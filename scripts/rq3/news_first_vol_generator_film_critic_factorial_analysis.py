"""Frozen Q3 selection and retrospective Q4 analysis for the FiLM/Critic factorial.

The module deliberately owns only analysis state.  Training and Q4 window
materialisation remain in :mod:`news_first_vol_generator_film_critic_factorial`.
The public hooks are called by that orchestrator at the three irreversible
boundaries:

* ``run_film_critic_q3_analysis`` evaluates all 16 development checkpoints on
  a common, pre-materialised Q3 panel and freezes the architecture decision;
* ``freeze_refit_recipes`` binds every development cell's learned epoch and
  generator/discriminator learning-rate history to an immutable replay recipe;
* ``run_film_critic_q4_analysis`` runs only after the orchestrator has opened
  the explicit Q4 gate and frozen the complete 16-checkpoint allowlist.

Inference is single-seed (seed 42).  Bootstrap uncertainty therefore resamples
whole CME sessions only; it must never be described as seed uncertainty.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.rq3 import news_first_vol_generator_film_critic_factorial as factorial
from scripts.rq3.news_first_vol_comparison_analysis import (
    RunSpec,
    TrainedRunEvaluator,
    _load_panel_source,
    aggregate_pair_metrics,
    compute_sample_metrics,
    holm_adjust,
)
from wgan_option.utils.text_ablation import REAL_TEXT, TEXT_SHUFFLE


SCHEMA_VERSION = 1
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260821
SEED = 42
EVALUATOR_COMMON_5M_NAMESPACE = "core"
ANCHOR_GENERATOR = "bottleneck_concat_v1"
FILM_GENERATOR = "film_conv_bottleneck_concat_v1"
ANCHOR_CRITIC = "lp_concat_v1"
DISABLED_CRITIC = "lp_disabled_same_shape_v1"
ANCHOR_ARCHITECTURE = (ANCHOR_GENERATOR, ANCHOR_CRITIC)
ARCHITECTURES = (
    ANCHOR_ARCHITECTURE,
    (ANCHOR_GENERATOR, DISABLED_CRITIC),
    (FILM_GENERATOR, ANCHOR_CRITIC),
    (FILM_GENERATOR, DISABLED_CRITIC),
)
ARCHITECTURE_PRIORITY = (
    (ANCHOR_GENERATOR, DISABLED_CRITIC),
    (FILM_GENERATOR, ANCHOR_CRITIC),
    (FILM_GENERATOR, DISABLED_CRITIC),
)
EXPECTED_Q3_COUNTS = {"rows": 148, "pairs": 135, "sessions": 33}
EXPECTED_Q4_COMMON_COUNTS = {"rows": 167, "pairs": 143, "sessions": 45}
HISTORICAL_Q4_PREDICTIONS = (
    factorial.REPO_ROOT
    / "outputs"
    / "experiments"
    / "rq3_news_first_vol_training_q097_103_ttm07_38_v1"
    / "analysis"
    / "predictions"
    / "regression_05m_core.csv.gz"
)

Q3_PAIR_METRICS = "film_critic_q3_pair_metrics.csv.gz"
Q3_CELL_SCORES = "film_critic_q3_cell_scores.csv"
Q3_ARCH_CONTRASTS = "film_critic_q3_architecture_contrasts.csv"
Q3_TEXT_CONTRASTS = "film_critic_q3_text_contrasts.csv"
Q3_FACTORIAL_EFFECTS = "film_critic_q3_factorial_effects.csv"
Q3_SELECTION = "film_critic_q3_selection.json"
REFIT_RECIPE_DIR = "refit_recipes"
REFIT_RECIPE_MANIFEST = "refit_recipe_manifest.json"
Q4_PAIR_METRICS = "film_critic_q4_pair_metrics.csv.gz"
Q4_PRIMARY_CONTRASTS = "film_critic_q4_primary_contrasts.csv"
Q4_SECONDARY = "film_critic_q4_30m_secondary.csv"
Q4_SUMMARY = "film_critic_q4_summary.json"


class FilmCriticAnalysisError(ValueError):
    """Raised when frozen analysis inputs or lineage do not match the protocol."""


def _analysis_dir(root: Path) -> Path:
    path = root / "analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(path: Path, label: str) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise FilmCriticAnalysisError(f"{label} must be a JSON object: {path}")
    return dict(value)


def _gzip_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    return path


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _self_hashed_payload(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    output = dict(payload)
    if field in output:
        raise FilmCriticAnalysisError(f"Self-hash field already present: {field}")
    output[field] = factorial._payload_sha256(output)
    return output


def _verify_self_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    observed = str(payload.get(field, ""))
    unsigned = {key: value for key, value in payload.items() if key != field}
    if not observed or observed != factorial._payload_sha256(unsigned):
        raise FilmCriticAnalysisError(f"{label} self-hash mismatch")


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "pairs": int(frame["pair_id"].astype(str).nunique()),
        "sessions": int(frame["session_id"].astype(str).nunique()),
    }


def _universe_sha(frame: pd.DataFrame, *, include_sample: bool) -> str:
    columns = ["session_id", "pair_id"]
    if include_sample:
        if "sample_id" in frame.columns:
            columns.append("sample_id")
        elif "news_row_id" in frame.columns:
            columns.append("news_row_id")
        else:
            raise FilmCriticAnalysisError("Panel lacks a stable sample key")
    rows = sorted(
        tuple(str(value) for value in row)
        for row in frame[columns].itertuples(index=False, name=None)
    )
    return factorial._payload_sha256(rows)


def _factor_key(job: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(job["generator_conditioning_mode"]),
        str(job["critic_conditioning_mode"]),
        str(job["text_ablation_mode"]),
        int(job["tolerance_minutes"]),
    )


def _expected_matrix() -> set[tuple[str, str, str, int]]:
    return {
        (generator, critic, text, tolerance)
        for generator in factorial.GENERATOR_MODES
        for critic in factorial.CRITIC_MODES
        for text in factorial.TEXT_MODES
        for tolerance in factorial.TOLERANCES
    }


def _status(root: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    path = factorial._job_status_path(root, str(job["job_id"]))
    value = _mapping(path, "job status")
    if not factorial._completed_job_is_valid(job, value):
        raise FilmCriticAnalysisError(
            f"Incomplete or hash-invalid job: {job['job_id']}"
        )
    if int(job.get("seed", -1)) != SEED:
        raise FilmCriticAnalysisError(f"Only seed {SEED} is permitted")
    return value


def _artifact(status: Mapping[str, Any], role: str) -> tuple[Path, str]:
    matches = [
        row for row in status.get("artifacts") or [] if row.get("artifact_role") == role
    ]
    if len(matches) != 1:
        raise FilmCriticAnalysisError(
            f"Expected exactly one {role} artifact, observed {len(matches)}"
        )
    row = matches[0]
    path = Path(str(row.get("path", ""))).resolve(strict=False)
    expected_sha = str(row.get("sha256", ""))
    if not path.is_file() or not expected_sha:
        raise FilmCriticAnalysisError(f"Missing {role} artifact: {path}")
    observed_sha = factorial._sha256_file(path)
    if observed_sha != expected_sha:
        raise FilmCriticAnalysisError(f"Hash drift for {role}: {path}")
    return path, observed_sha


def _stage_jobs(root: Path, stage: str) -> list[dict[str, Any]]:
    registry = factorial._load_registry(root)
    jobs = [
        dict(job)
        for job in registry.get("jobs") or []
        if str(job.get("experiment_stage")) == stage
    ]
    if len(jobs) != 16:
        raise FilmCriticAnalysisError(
            f"{stage} requires exactly 16 jobs, observed {len(jobs)}"
        )
    keys = [_factor_key(job) for job in jobs]
    if len(set(keys)) != 16 or set(keys) != _expected_matrix():
        raise FilmCriticAnalysisError(f"{stage} factorial matrix is incomplete")
    if len({str(job["job_id"]) for job in jobs}) != 16:
        raise FilmCriticAnalysisError(f"{stage} contains duplicate job IDs")
    for job in jobs:
        _status(root, job)
    return sorted(jobs, key=_factor_key)


def _assert_q3_gate(root: Path) -> None:
    registry = factorial._load_registry(root)
    forbidden = (
        "selection_frozen",
        "refit_complete",
        "q4_gate_open",
        "q4_window_materialized",
        "q4_loader_created",
        "q4_predictions_generated",
        "q4_evaluated",
    )
    active = [name for name in forbidden if bool(registry.get(name))]
    if active:
        raise FilmCriticAnalysisError(
            f"Q3 selection cannot run after frozen/downstream state: {active}"
        )
    if (root / "data_windows" / "q4").exists():
        raise FilmCriticAnalysisError("Q4 objects exist before Q3 selection freeze")
    manifest = _mapping(
        root / "data_windows" / "pre_q4_window_manifest.json",
        "pre-Q4 window manifest",
    )
    for field in (
        "q4_window_materialized",
        "q4_sample_objects_materialized",
        "q4_loader_created",
        "q4_predictions_generated",
        "q4_evaluated",
    ):
        if bool(manifest.get(field)):
            raise FilmCriticAnalysisError(f"Pre-Q4 manifest reports {field}=true")
    for job in registry.get("jobs") or []:
        status_path = factorial._job_status_path(root, str(job["job_id"]))
        if not status_path.is_file():
            continue
        status = _mapping(status_path, "job status")
        if any(
            bool(status.get(field))
            for field in (
                "q4_loader_created",
                "q4_predictions_generated",
                "q4_evaluated",
            )
        ):
            raise FilmCriticAnalysisError(
                f"Job accessed Q4 before selection: {job['job_id']}"
            )


def _resolved(root: Path) -> dict[str, Any]:
    resolved = factorial._validate_root_lineage(root)
    analysis = factorial._factorial(resolved).get("analysis") or {}
    if int(analysis.get("bootstrap_replicates", -1)) != BOOTSTRAP_REPLICATES:
        raise FilmCriticAnalysisError("Bootstrap replicates drifted from 10,000")
    if int(analysis.get("bootstrap_seed", -1)) != BOOTSTRAP_SEED:
        raise FilmCriticAnalysisError("Bootstrap master seed drifted")
    if not bool(analysis.get("single_seed_inference")):
        raise FilmCriticAnalysisError("Analysis must remain explicitly single-seed")
    return resolved


def _filter_panel(
    frame: pd.DataFrame,
    *,
    start: str,
    end: str,
    panel_name: str,
    expected: Mapping[str, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "effective_origin_utc" not in frame.columns:
        raise FilmCriticAnalysisError(f"{panel_name} lacks effective_origin_utc")
    timestamps = pd.to_datetime(
        frame["effective_origin_utc"], errors="coerce", utc=True
    )
    if timestamps.isna().any():
        raise FilmCriticAnalysisError(f"{panel_name} has invalid timestamps")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    selected = frame.loc[(timestamps >= start_ts) & (timestamps < end_ts)].copy()
    panel, lineage, _ = _load_panel_source(
        selected,
        sheet_name="gan_input_ready",
        panel_name=panel_name,
        freeze_test_window=False,
        enforce_expected_counts=False,
        support_mask_mode="raw_joint",
    )
    observed = _counts(panel)
    frozen = {key: int(value) for key, value in expected.items()}
    if observed != frozen:
        raise FilmCriticAnalysisError(
            f"{panel_name} counts drifted: {observed} != {frozen}"
        )
    if bool(
        (pd.to_datetime(panel["effective_origin_utc"], utc=True) < start_ts).any()
        or (pd.to_datetime(panel["effective_origin_utc"], utc=True) >= end_ts).any()
    ):
        raise FilmCriticAnalysisError(f"{panel_name} escaped its frozen interval")
    lineage.update(
        {
            "interval_start_utc_inclusive": start_ts.isoformat(),
            "interval_end_utc_exclusive": end_ts.isoformat(),
            "counts": observed,
            "pair_universe_sha256": _universe_sha(panel, include_sample=False),
            "sample_universe_sha256": _universe_sha(panel, include_sample=True),
        }
    )
    return panel.reset_index(drop=True), lineage


def _q3_panel(
    root: Path, resolved: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    workbook = root / "data_windows" / "pre_q4" / "tolerance_05m_pre_q4.xlsx"
    if not workbook.is_file():
        raise FileNotFoundError(workbook)
    frame = pd.read_excel(workbook, sheet_name=str(resolved["datasets"]["sheet_name"]))
    timestamps = pd.to_datetime(
        frame["effective_origin_utc"], errors="coerce", utc=True
    )
    if timestamps.isna().any() or bool(
        (timestamps >= pd.Timestamp(resolved["split"]["q4_start_utc"])).any()
    ):
        raise FilmCriticAnalysisError("Q3 source is not a strictly pre-Q4 workbook")
    panel, lineage = _filter_panel(
        frame,
        start=str(resolved["split"]["development_train_end_utc"]),
        end=str(resolved["split"]["development_validation_end_utc"]),
        panel_name="film_critic_q3_common_05m",
        expected=EXPECTED_Q3_COUNTS,
    )
    lineage.update(
        {
            "source_path": str(workbook.resolve()),
            "source_sha256": factorial._sha256_file(workbook),
            "q4_rows_read_by_selection": 0,
            "q4_loader_created": False,
            "q4_predictions_generated": False,
        }
    )
    return panel, lineage


def _q4_panel(
    root: Path, resolved: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    registry = factorial._load_registry(root)
    workbook = Path(str(registry.get("q4_window_path", "")))
    expected_sha = str(registry.get("q4_window_sha256", ""))
    if not workbook.is_file() or factorial._sha256_file(workbook) != expected_sha:
        raise FilmCriticAnalysisError("Frozen Q4 common workbook hash drift")
    frame = pd.read_excel(workbook, sheet_name=str(resolved["datasets"]["sheet_name"]))
    panel, lineage = _filter_panel(
        frame,
        start=str(resolved["split"]["q4_start_utc"]),
        end=str(resolved["split"]["q4_end_utc"]),
        panel_name="film_critic_q4_common_05m",
        expected=EXPECTED_Q4_COMMON_COUNTS,
    )
    lineage.update(
        {
            "source_path": str(workbook.resolve()),
            "source_sha256": factorial._sha256_file(workbook),
            "q4_gate_open_before_read": True,
        }
    )
    return panel, lineage


def _run_spec(
    root: Path, job: Mapping[str, Any], *, checkpoint_role: str
) -> tuple[RunSpec, str]:
    status = _status(root, job)
    checkpoint, checkpoint_sha = _artifact(status, checkpoint_role)
    return (
        RunSpec(
            run_id=str(job["job_id"]),
            run_dir=Path(str(status["run_dir"])).resolve(strict=False),
            model="wgan",
            tolerance_minutes=int(job["tolerance_minutes"]),
            seed=int(job["seed"]),
            checkpoint_path=checkpoint,
            text_ablation_mode=str(job["text_ablation_mode"]),
            support_mask_mode="raw_joint",
            generator_current_input_mode="current_support_masked",
            metadata={
                "experiment_stage": job["experiment_stage"],
                "generator_conditioning_mode": job["generator_conditioning_mode"],
                "critic_conditioning_mode": job["critic_conditioning_mode"],
                "model_contract_sha256": job["model_contract_sha256"],
            },
        ),
        checkpoint_sha,
    )


def _prediction_cache(
    root: Path,
    *,
    stage: str,
    job: Mapping[str, Any],
    spec: RunSpec,
    checkpoint_sha: str,
    panel: pd.DataFrame,
    panel_name: str,
    mc_samples: int,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    directory = _analysis_dir(root) / "predictions" / stage
    directory.mkdir(parents=True, exist_ok=True)
    prediction_path = directory / f"{job['job_id']}.csv.gz"
    manifest_path = directory / f"{job['job_id']}.manifest.json"
    contract = {
        "schema_version": SCHEMA_VERSION,
        "job_id": str(job["job_id"]),
        "job_spec_sha256": str(job["job_spec_sha256"]),
        "checkpoint_path": str(spec.checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "panel": panel_name,
        "evaluator_panel_namespace": EVALUATOR_COMMON_5M_NAMESPACE,
        "panel_sample_universe_sha256": _universe_sha(panel, include_sample=True),
        "mc_samples": int(mc_samples),
        "single_seed": SEED,
    }
    if prediction_path.exists() or manifest_path.exists():
        if not prediction_path.is_file() or not manifest_path.is_file():
            raise FilmCriticAnalysisError(f"Partial prediction cache: {job['job_id']}")
        manifest = _mapping(manifest_path, "prediction manifest")
        _verify_self_hash(manifest, "manifest_sha256", "prediction manifest")
        unsigned = {
            key: value
            for key, value in manifest.items()
            if key
            not in {
                "prediction_path",
                "prediction_sha256",
                "row_count",
                "manifest_sha256",
            }
        }
        if unsigned != contract:
            raise FilmCriticAnalysisError(
                f"Prediction cache contract drift: {job['job_id']}"
            )
        if str(manifest.get("prediction_path")) != str(
            prediction_path.resolve()
        ) or str(manifest.get("prediction_sha256")) != factorial._sha256_file(
            prediction_path
        ):
            raise FilmCriticAnalysisError(
                f"Prediction cache hash drift: {job['job_id']}"
            )
        predictions = pd.read_csv(prediction_path, low_memory=False)
        if int(manifest.get("row_count", -1)) != len(predictions):
            raise FilmCriticAnalysisError(
                f"Prediction cache row drift: {job['job_id']}"
            )
        return predictions
    # Every development and Q4 inference panel is the frozen common 5m panel.
    # Keeping the evaluator namespace identical also keeps text-shuffle mapping
    # identical across the 5m-primary and 30m-trained secondary checkpoints.
    predictions = evaluator(spec, EVALUATOR_COMMON_5M_NAMESPACE, panel.copy())
    if not isinstance(predictions, pd.DataFrame):
        raise FilmCriticAnalysisError("Evaluator must return a pandas DataFrame")
    _gzip_csv(prediction_path, predictions)
    manifest = _self_hashed_payload(
        {
            **contract,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": factorial._sha256_file(prediction_path),
            "row_count": int(len(predictions)),
        },
        "manifest_sha256",
    )
    factorial._write_json(manifest_path, manifest)
    return predictions


def _evaluate_jobs(
    root: Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    panel: pd.DataFrame,
    panel_name: str,
    stage: str,
    checkpoint_role: str,
    mc_samples: int,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None,
) -> pd.DataFrame:
    production = evaluator or TrainedRunEvaluator(mc_samples=mc_samples)
    parts: list[pd.DataFrame] = []
    for job in jobs:
        spec, checkpoint_sha = _run_spec(root, job, checkpoint_role=checkpoint_role)
        predictions = _prediction_cache(
            root,
            stage=stage,
            job=job,
            spec=spec,
            checkpoint_sha=checkpoint_sha,
            panel=panel,
            panel_name=panel_name,
            mc_samples=mc_samples,
            evaluator=production,
        )
        samples, exclusions, _ = compute_sample_metrics(
            spec,
            panel_name,
            panel,
            predictions,
            evaluate_embedded_atm_skew=False,
        )
        if not exclusions.empty:
            counts = exclusions["exclusion_code"].value_counts().to_dict()
            raise FilmCriticAnalysisError(
                f"Prediction exclusions for {job['job_id']}: {counts}"
            )
        pairs = aggregate_pair_metrics(samples)
        pairs = pairs[
            pairs["stratum_type"].astype(str).eq("overall")
            & pairs["stratum_value"].astype(str).eq("all")
        ].copy()
        if _counts(pairs) != {
            "rows": int(panel["pair_id"].nunique()),
            "pairs": int(panel["pair_id"].nunique()),
            "sessions": int(panel["session_id"].nunique()),
        }:
            raise FilmCriticAnalysisError(
                f"Pair aggregation coverage drift for {job['job_id']}"
            )
        for position, (name, value) in enumerate(
            (
                ("experiment_stage", job["experiment_stage"]),
                ("job_id", job["job_id"]),
                ("generator_conditioning_mode", job["generator_conditioning_mode"]),
                ("critic_conditioning_mode", job["critic_conditioning_mode"]),
                ("checkpoint_sha256", checkpoint_sha),
            )
        ):
            pairs.insert(position, name, value)
        parts.append(pairs)
    output = pd.concat(parts, ignore_index=True)
    keys = [
        "job_id",
        "generator_conditioning_mode",
        "critic_conditioning_mode",
        "text_ablation_mode",
        "tolerance_minutes",
        "seed",
        "pair_id",
    ]
    if output.duplicated(keys).any():
        raise FilmCriticAnalysisError("Duplicate factorial pair-metric key")
    coverages = {
        tuple(
            sorted(
                group[["session_id", "pair_id"]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
        )
        for _, group in output.groupby("job_id", sort=False)
    }
    if len(coverages) != 1:
        raise FilmCriticAnalysisError(
            "All factorial cells must share exact pair coverage"
        )
    return output.reset_index(drop=True)


def session_cluster_bootstrap(
    differences: Sequence[float],
    session_ids: Sequence[str],
    *,
    iterations: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Pair-balanced mean with whole-session resampling for one fixed seed."""

    if int(iterations) <= 0:
        raise FilmCriticAnalysisError("bootstrap iterations must be positive")
    frame = pd.DataFrame(
        {
            "difference": pd.to_numeric(pd.Series(differences), errors="coerce"),
            "session_id": pd.Series(session_ids, dtype="object"),
        }
    )
    finite = np.isfinite(frame["difference"].to_numpy(dtype=np.float64))
    frame = frame.loc[
        finite
        & frame["session_id"].notna()
        & frame["session_id"].astype(str).str.strip().ne("")
    ].copy()
    if frame.empty:
        raise FilmCriticAnalysisError(
            "Bootstrap received no finite paired observations"
        )
    grouped = frame.groupby("session_id", sort=True)["difference"].agg(["sum", "count"])
    if len(grouped) < 2:
        raise FilmCriticAnalysisError("Bootstrap requires at least two CME sessions")
    observed = float(frame["difference"].mean())
    if frame["difference"].eq(0.0).all():
        return {
            "mean_difference": 0.0,
            "bootstrap_se": 0.0,
            "ci_95_lower": 0.0,
            "ci_95_upper": 0.0,
            "p_two_sided": 1.0,
            "pair_count": int(len(frame)),
            "session_count": int(len(grouped)),
            "bootstrap_iterations": int(iterations),
            "bootstrap_seed": int(seed),
            "inference_unit": "cme_session_cluster_single_seed",
        }
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    selected = rng.integers(0, len(grouped), size=(int(iterations), len(grouped)))
    draws = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    centered = draws - observed
    p_two_sided = (float(np.sum(np.abs(centered) >= abs(observed))) + 1.0) / (
        float(iterations) + 1.0
    )
    return {
        "mean_difference": observed,
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_two_sided": float(p_two_sided),
        "pair_count": int(len(frame)),
        "session_count": int(len(grouped)),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
        "inference_unit": "cme_session_cluster_single_seed",
    }


def _cell(
    frame: pd.DataFrame,
    generator: str,
    critic: str,
    text: str,
    tolerance: int,
) -> pd.DataFrame:
    selected = frame[
        frame["generator_conditioning_mode"].astype(str).eq(generator)
        & frame["critic_conditioning_mode"].astype(str).eq(critic)
        & frame["text_ablation_mode"].astype(str).eq(text)
        & frame["tolerance_minutes"].astype(int).eq(int(tolerance))
    ].copy()
    if selected["job_id"].nunique() != 1:
        raise FilmCriticAnalysisError(
            f"Expected one cell for {(generator, critic, text, tolerance)}"
        )
    return selected


def _paired_values(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_column: str = "model_mae",
    right_column: str = "model_mae",
) -> pd.DataFrame:
    keys = ["session_id", "pair_id"]
    lhs = left[keys + [left_column]].rename(columns={left_column: "left_value"})
    rhs = right[keys + [right_column]].rename(columns={right_column: "right_value"})
    if lhs.duplicated(keys).any() or rhs.duplicated(keys).any():
        raise FilmCriticAnalysisError("Contrast inputs contain duplicate pair keys")
    if set(map(tuple, lhs[keys].astype(str).to_numpy())) != set(
        map(tuple, rhs[keys].astype(str).to_numpy())
    ):
        raise FilmCriticAnalysisError(
            "Contrast cells have different pair/session coverage"
        )
    merged = lhs.merge(rhs, on=keys, validate="one_to_one")
    merged["difference"] = pd.to_numeric(
        merged["left_value"], errors="raise"
    ) - pd.to_numeric(merged["right_value"], errors="raise")
    return merged


def _contrast(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    seed_offset: int,
    left_column: str = "model_mae",
    right_column: str = "model_mae",
) -> dict[str, Any]:
    merged = _paired_values(
        left,
        right,
        left_column=left_column,
        right_column=right_column,
    )
    return session_cluster_bootstrap(
        merged["difference"],
        merged["session_id"],
        seed=BOOTSTRAP_SEED + int(seed_offset),
    )


def _architecture_contrasts(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    anchor = _cell(pair_metrics, *ANCHOR_ARCHITECTURE, REAL_TEXT, 5)
    rows: list[dict[str, Any]] = []
    for offset, (generator, critic) in enumerate(ARCHITECTURES[1:]):
        candidate = _cell(pair_metrics, generator, critic, REAL_TEXT, 5)
        rows.append(
            {
                "candidate_generator_conditioning_mode": generator,
                "candidate_critic_conditioning_mode": critic,
                "anchor_generator_conditioning_mode": ANCHOR_GENERATOR,
                "anchor_critic_conditioning_mode": ANCHOR_CRITIC,
                "text_ablation_mode": REAL_TEXT,
                "tolerance_minutes": 5,
                **_contrast(candidate, anchor, seed_offset=100 + offset),
            }
        )
    if len(rows) != 3:
        raise AssertionError("Architecture Holm family must contain three contrasts")
    adjusted = holm_adjust([float(row["p_two_sided"]) for row in rows])
    for row, holm_p in zip(rows, adjusted):
        row["holm_p"] = float(holm_p)
        row["holm_family"] = "q3_real_architecture_vs_anchor_holm3"
        row["eligible"] = bool(
            float(row["mean_difference"]) < 0.0
            and float(row["ci_95_upper"]) < 0.0
            and float(holm_p) < 0.05
        )
    return pd.DataFrame(rows)


def _text_contrasts(pair_metrics: pd.DataFrame, *, tolerance: int = 5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for offset, (generator, critic) in enumerate(ARCHITECTURES):
        real = _cell(pair_metrics, generator, critic, REAL_TEXT, tolerance)
        shuffled = _cell(pair_metrics, generator, critic, TEXT_SHUFFLE, tolerance)
        rows.append(
            {
                "generator_conditioning_mode": generator,
                "critic_conditioning_mode": critic,
                "left_text_mode": REAL_TEXT,
                "right_text_mode": TEXT_SHUFFLE,
                "tolerance_minutes": int(tolerance),
                **_contrast(
                    real, shuffled, seed_offset=2000 + 100 * tolerance + offset
                ),
            }
        )
    adjusted = holm_adjust([float(row["p_two_sided"]) for row in rows])
    for row, holm_p in zip(rows, adjusted):
        row["holm_p"] = float(holm_p)
        row["holm_family"] = f"{tolerance}m_real_minus_shuffle_holm4"
        row["alignment_supported"] = bool(
            float(row["mean_difference"]) < 0.0
            and float(row["ci_95_upper"]) < 0.0
            and float(holm_p) < 0.05
        )
    return pd.DataFrame(rows)


_FACTOR_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("G", ("G",)),
    ("D", ("D",)),
    ("T", ("T",)),
    ("GxD", ("G", "D")),
    ("GxT", ("G", "T")),
    ("DxT", ("D", "T")),
    ("GxDxT", ("G", "D", "T")),
)


def _factor_signs(generator: str, critic: str, text: str) -> dict[str, int]:
    return {
        "G": 1 if generator == FILM_GENERATOR else -1,
        "D": 1 if critic == DISABLED_CRITIC else -1,
        "T": 1 if text == REAL_TEXT else -1,
    }


def _factorial_pair_contrast(
    pair_metrics: pd.DataFrame, *, tolerance: int, factors: Sequence[str]
) -> pd.DataFrame:
    keys = ["session_id", "pair_id"]
    merged: pd.DataFrame | None = None
    cell_columns: list[tuple[str, float]] = []
    denominator = float(2 ** (3 - len(factors)))
    for index, (generator, critic) in enumerate(ARCHITECTURES):
        for text in (TEXT_SHUFFLE, REAL_TEXT):
            cell = _cell(pair_metrics, generator, critic, text, tolerance)
            name = f"cell_{index}_{text}"
            selected = cell[keys + ["model_mae"]].rename(columns={"model_mae": name})
            merged = (
                selected
                if merged is None
                else merged.merge(selected, on=keys, validate="one_to_one")
            )
            signs = _factor_signs(generator, critic, text)
            coefficient = (
                float(np.prod([signs[factor] for factor in factors])) / denominator
            )
            cell_columns.append((name, coefficient))
    if merged is None or len(merged) == 0:
        raise FilmCriticAnalysisError("Empty factorial contrast")
    merged["difference"] = sum(
        coefficient * pd.to_numeric(merged[column], errors="raise")
        for column, coefficient in cell_columns
    )
    return merged[keys + ["difference"]]


def _factorial_effects(pair_metrics: pd.DataFrame, *, tolerance: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for offset, (term, factors) in enumerate(_FACTOR_TERMS):
        paired = _factorial_pair_contrast(
            pair_metrics, tolerance=tolerance, factors=factors
        )
        rows.append(
            {
                "term": term,
                "order": len(factors),
                "tolerance_minutes": int(tolerance),
                "coding": (
                    "G: film(+1)/concat(-1); D: LP-disabled(+1)/LP-concat(-1); "
                    "T: real(+1)/shuffle(-1)"
                ),
                **session_cluster_bootstrap(
                    paired["difference"],
                    paired["session_id"],
                    seed=BOOTSTRAP_SEED + 4000 + 100 * tolerance + offset,
                ),
            }
        )
    return pd.DataFrame(rows)


def _cell_scores(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for offset, (key, group) in enumerate(
        pair_metrics.groupby(
            [
                "generator_conditioning_mode",
                "critic_conditioning_mode",
                "text_ablation_mode",
                "tolerance_minutes",
            ],
            sort=True,
        )
    ):
        if group["job_id"].nunique() != 1:
            raise FilmCriticAnalysisError(f"Cell maps to multiple jobs: {key}")
        summary = _contrast(
            group,
            group,
            seed_offset=6000 + offset,
            left_column="model_mae",
            right_column="persistence_mae",
        )
        rows.append(
            {
                "generator_conditioning_mode": key[0],
                "critic_conditioning_mode": key[1],
                "text_ablation_mode": key[2],
                "tolerance_minutes": int(key[3]),
                "seed": SEED,
                "job_id": str(group["job_id"].iloc[0]),
                "model_mae": float(group["model_mae"].mean()),
                "persistence_mae": float(group["persistence_mae"].mean()),
                "mae_ratio": float(
                    group["model_mae"].mean() / group["persistence_mae"].mean()
                ),
                **summary,
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != 16:
        raise FilmCriticAnalysisError("Q3 cell score table must contain 16 cells")
    output["rank_within_tolerance"] = output.groupby("tolerance_minutes")[
        "model_mae"
    ].rank(method="first")
    return output.sort_values(
        ["tolerance_minutes", "model_mae"], kind="stable"
    ).reset_index(drop=True)


def _normalize_trace(
    rows: Sequence[Mapping[str, Any]], *, num_epochs: int, label: str
) -> list[dict[str, Any]]:
    selected = [
        {"epoch": int(row["epoch"]), "lr": float(row["lr"])}
        for row in rows
        if 1 <= int(row["epoch"]) <= int(num_epochs)
    ]
    if [row["epoch"] for row in selected] != list(range(1, int(num_epochs) + 1)):
        raise FilmCriticAnalysisError(f"{label} must cover epoch 1..E exactly")
    if any(not math.isfinite(row["lr"]) or row["lr"] <= 0.0 for row in selected):
        raise FilmCriticAnalysisError(f"{label} contains invalid learning rates")
    return selected


def _development_contract(root: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    status = _status(root, job)
    generator_checkpoint, generator_checkpoint_sha = _artifact(
        status, "generator_best_learned"
    )
    discriminator_checkpoint, discriminator_checkpoint_sha = _artifact(
        status, "discriminator_best_learned"
    )
    best_path, best_sha = _artifact(status, "best_learned_checkpoint")
    metrics_path, metrics_sha = _artifact(status, "training_metrics_json")
    best = _mapping(best_path, "best-learned checkpoint metadata")
    metrics = _read_json(metrics_path)
    if not isinstance(metrics, list):
        raise FilmCriticAnalysisError("training_metrics.json must be a list")
    epoch = int(best.get("best_learned_epoch_ge_1", best.get("best_epoch", 0)))
    if epoch < 1:
        raise FilmCriticAnalysisError(
            f"best_learned epoch must be >=1: {job['job_id']}"
        )
    by_epoch = {
        int(row["epoch"]): row
        for row in metrics
        if isinstance(row, Mapping) and int(row.get("epoch", -1)) >= 1
    }
    if sorted(value for value in by_epoch if value <= epoch) != list(
        range(1, epoch + 1)
    ):
        raise FilmCriticAnalysisError(f"Incomplete metrics LR trace: {job['job_id']}")
    generator_trace = _normalize_trace(
        [
            {"epoch": value, "lr": by_epoch[value]["g_lr"]}
            for value in range(1, epoch + 1)
        ],
        num_epochs=epoch,
        label="generator metrics trace",
    )
    discriminator_trace = _normalize_trace(
        [
            {"epoch": value, "lr": by_epoch[value]["d_lr"]}
            for value in range(1, epoch + 1)
        ],
        num_epochs=epoch,
        label="discriminator metrics trace",
    )
    metadata_g = _normalize_trace(
        list(best.get("generator_lr_trace") or []),
        num_epochs=epoch,
        label="generator best-metadata trace",
    )
    metadata_d = _normalize_trace(
        list(best.get("discriminator_lr_trace") or []),
        num_epochs=epoch,
        label="discriminator best-metadata trace",
    )
    if generator_trace != metadata_g or discriminator_trace != metadata_d:
        raise FilmCriticAnalysisError(f"LR trace disagreement: {job['job_id']}")
    config_path = Path(str(job["training_config_path"])).resolve(strict=False)
    if not config_path.is_file() or factorial._sha256_file(config_path) != str(
        job.get("config_sha256", "")
    ):
        raise FilmCriticAnalysisError(f"Development config hash drift: {job['job_id']}")
    code_manifest = root / "code_hashes.csv"
    if not code_manifest.is_file():
        raise FileNotFoundError(code_manifest)
    contract = {
        "development_job_id": str(job["job_id"]),
        "generator_conditioning_mode": str(job["generator_conditioning_mode"]),
        "critic_conditioning_mode": str(job["critic_conditioning_mode"]),
        "text_ablation_mode": str(job["text_ablation_mode"]),
        "tolerance_minutes": int(job["tolerance_minutes"]),
        "seed": int(job["seed"]),
        "best_learned_epoch": epoch,
        "generator_lr_trace": generator_trace,
        "discriminator_lr_trace": discriminator_trace,
        "best_learned_generator_path": str(generator_checkpoint),
        "best_learned_generator_sha256": generator_checkpoint_sha,
        "best_learned_discriminator_path": str(discriminator_checkpoint),
        "best_learned_discriminator_sha256": discriminator_checkpoint_sha,
        "best_learned_metadata_path": str(best_path),
        "best_learned_metadata_sha256": best_sha,
        "training_metrics_path": str(metrics_path),
        "training_metrics_sha256": metrics_sha,
        "training_config_path": str(config_path),
        "training_config_sha256": str(job["config_sha256"]),
        "code_manifest_path": str(code_manifest.resolve()),
        "code_manifest_sha256": factorial._sha256_file(code_manifest),
    }
    contract["training_contract_sha256"] = factorial._payload_sha256(contract)
    return contract


def _selection_from_tables(
    architecture: pd.DataFrame,
    text: pd.DataFrame,
    contracts: Sequence[Mapping[str, Any]],
    *,
    artifact_hashes: Mapping[str, str],
    panel_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = architecture[architecture["eligible"].astype(bool)].copy()
    if eligible.empty:
        winner = ANCHOR_ARCHITECTURE
        label = "no_supported_architecture_change"
        one_se_threshold: float | None = None
        point_leader: dict[str, str] | None = None
    else:
        ordered = eligible.sort_values("mean_difference", kind="stable").reset_index(
            drop=True
        )
        leader = ordered.iloc[0]
        threshold = float(leader["mean_difference"] + leader["bootstrap_se"])
        within = ordered[ordered["mean_difference"] <= threshold].copy()
        priority = {value: index for index, value in enumerate(ARCHITECTURE_PRIORITY)}
        within["priority"] = [
            priority[
                (
                    row.candidate_generator_conditioning_mode,
                    row.candidate_critic_conditioning_mode,
                )
            ]
            for row in within.itertuples()
        ]
        chosen = within.sort_values(
            ["priority", "mean_difference"], kind="stable"
        ).iloc[0]
        winner = (
            str(chosen["candidate_generator_conditioning_mode"]),
            str(chosen["candidate_critic_conditioning_mode"]),
        )
        label = "statistically_supported_architecture_change"
        one_se_threshold = threshold
        point_leader = {
            "generator_conditioning_mode": str(
                leader["candidate_generator_conditioning_mode"]
            ),
            "critic_conditioning_mode": str(
                leader["candidate_critic_conditioning_mode"]
            ),
        }
    winner_text = text[
        text["generator_conditioning_mode"].astype(str).eq(winner[0])
        & text["critic_conditioning_mode"].astype(str).eq(winner[1])
    ]
    if len(winner_text) != 1:
        raise FilmCriticAnalysisError("Winner text contrast is missing or duplicated")
    text_supported = bool(winner_text.iloc[0]["alignment_supported"])
    contract_rows = [dict(value) for value in contracts]
    if (
        len(contract_rows) != 16
        or len({row["development_job_id"] for row in contract_rows}) != 16
    ):
        raise FilmCriticAnalysisError("Selection requires 16 development contracts")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": factorial.EXPERIMENT_KIND,
        "selection_stage": factorial.DEVELOPMENT_STAGE,
        "selection_panel": "2023Q3_common_05m_raw_joint",
        "selection_tolerance_minutes": 5,
        "single_seed": SEED,
        "bootstrap_unit": "cme_session_cluster_single_seed",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed_uncertainty_estimated": False,
        "q4_rows_read_by_selection": 0,
        "q4_loader_created": False,
        "q4_predictions_generated": False,
        "anchor": {
            "generator_conditioning_mode": ANCHOR_GENERATOR,
            "critic_conditioning_mode": ANCHOR_CRITIC,
            "text_ablation_mode": REAL_TEXT,
        },
        "winner": {
            "generator_conditioning_mode": winner[0],
            "critic_conditioning_mode": winner[1],
            "text_ablation_mode": REAL_TEXT,
        },
        "selection_label": label,
        "point_leader": point_leader,
        "one_se_threshold": one_se_threshold,
        "architecture_holm_family_size": 3,
        "text_holm_family_size": 4,
        "text_alignment_supported": text_supported,
        "text_alignment_is_not_winner_gate": True,
        "panel_lineage": dict(panel_lineage),
        "development_training_contracts": contract_rows,
        "development_matrix_sha256": factorial._payload_sha256(
            sorted(row["training_contract_sha256"] for row in contract_rows)
        ),
        "analysis_artifact_sha256": dict(artifact_hashes),
    }
    return _self_hashed_payload(payload, "selection_sha256")


def run_film_critic_q3_analysis(
    root: str | Path,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Evaluate and freeze the single-seed Q3 architecture decision."""

    experiment_root = Path(root).resolve(strict=False)
    resolved = _resolved(experiment_root)
    _assert_q3_gate(experiment_root)
    jobs = _stage_jobs(experiment_root, factorial.DEVELOPMENT_STAGE)
    panel, panel_lineage = _q3_panel(experiment_root, resolved)
    pair_metrics = _evaluate_jobs(
        experiment_root,
        jobs,
        panel=panel,
        panel_name="film_critic_q3_common_05m",
        stage="q3_development_best_learned",
        checkpoint_role="generator_best_learned",
        mc_samples=int(resolved["split"]["validation_mc_samples"]),
        evaluator=evaluator,
    )
    analysis_dir = _analysis_dir(experiment_root)
    pair_path = _gzip_csv(analysis_dir / Q3_PAIR_METRICS, pair_metrics)
    cell_scores = _cell_scores(pair_metrics)
    cell_path = _write_csv(analysis_dir / Q3_CELL_SCORES, cell_scores)
    architecture = _architecture_contrasts(pair_metrics)
    architecture_path = _write_csv(analysis_dir / Q3_ARCH_CONTRASTS, architecture)
    text = _text_contrasts(pair_metrics, tolerance=5)
    text_path = _write_csv(analysis_dir / Q3_TEXT_CONTRASTS, text)
    factorial_effects = _factorial_effects(pair_metrics, tolerance=5)
    factorial_path = _write_csv(analysis_dir / Q3_FACTORIAL_EFFECTS, factorial_effects)
    contracts = [_development_contract(experiment_root, job) for job in jobs]
    artifacts = {
        Q3_PAIR_METRICS: factorial._sha256_file(pair_path),
        Q3_CELL_SCORES: factorial._sha256_file(cell_path),
        Q3_ARCH_CONTRASTS: factorial._sha256_file(architecture_path),
        Q3_TEXT_CONTRASTS: factorial._sha256_file(text_path),
        Q3_FACTORIAL_EFFECTS: factorial._sha256_file(factorial_path),
    }
    selection = _selection_from_tables(
        architecture,
        text,
        contracts,
        artifact_hashes=artifacts,
        panel_lineage=panel_lineage,
    )
    factorial._write_json(analysis_dir / Q3_SELECTION, selection)
    return selection


def _load_selection(
    root: Path, selection: Mapping[str, Any] | str | Path
) -> tuple[dict[str, Any], Path]:
    canonical = _analysis_dir(root) / Q3_SELECTION
    if isinstance(selection, Mapping):
        payload = dict(selection)
    else:
        provided = Path(selection)
        if not provided.is_absolute():
            provided = (root / provided).resolve(strict=False)
        payload = _mapping(provided, "Q3 selection")
    _verify_self_hash(payload, "selection_sha256", "Q3 selection")
    if not canonical.is_file():
        raise FileNotFoundError(canonical)
    disk = _mapping(canonical, "canonical Q3 selection")
    if payload != disk:
        raise FilmCriticAnalysisError(
            "Selection argument differs from canonical selection"
        )
    if int(payload.get("q4_rows_read_by_selection", -1)) != 0:
        raise FilmCriticAnalysisError("Selection reports Q4 access")
    return payload, canonical


def freeze_refit_recipes(
    root: str | Path, selection: Mapping[str, Any] | str | Path
) -> Path:
    """Freeze 16 exact epoch/LR replay recipes and their provenance manifest."""

    experiment_root = Path(root).resolve(strict=False)
    _resolved(experiment_root)
    _assert_q3_gate(experiment_root)
    payload, selection_path = _load_selection(experiment_root, selection)
    jobs = _stage_jobs(experiment_root, factorial.DEVELOPMENT_STAGE)
    frozen_contracts = {
        str(row["development_job_id"]): dict(row)
        for row in payload.get("development_training_contracts") or []
    }
    if len(frozen_contracts) != 16:
        raise FilmCriticAnalysisError(
            "Selection lacks the complete training contract map"
        )
    recipe_dir = _analysis_dir(experiment_root) / REFIT_RECIPE_DIR
    if recipe_dir.exists() and any(recipe_dir.iterdir()):
        raise FilmCriticAnalysisError("Refit recipes already exist; refusing overwrite")
    recipe_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for job in jobs:
        contract = _development_contract(experiment_root, job)
        frozen = frozen_contracts.get(str(job["job_id"]))
        if frozen != contract:
            raise FilmCriticAnalysisError(
                f"Development contract drift after Q3 selection: {job['job_id']}"
            )
        recipe = {
            "schema_version": SCHEMA_VERSION,
            "refit_mode": factorial.REFIT_MODE,
            "num_epochs": int(contract["best_learned_epoch"]),
            "generator_lr_trace": contract["generator_lr_trace"],
            "discriminator_lr_trace": contract["discriminator_lr_trace"],
        }
        factorial._validate_refit_recipe(recipe)
        path = recipe_dir / f"{job['job_id']}.json"
        factorial._write_json(path, recipe)
        rows.append(
            {
                "development_job_id": str(job["job_id"]),
                "generator_conditioning_mode": str(job["generator_conditioning_mode"]),
                "critic_conditioning_mode": str(job["critic_conditioning_mode"]),
                "text_ablation_mode": str(job["text_ablation_mode"]),
                "tolerance_minutes": int(job["tolerance_minutes"]),
                "seed": int(job["seed"]),
                "num_epochs": int(recipe["num_epochs"]),
                "recipe_path": str(path.resolve()),
                "recipe_sha256": factorial._sha256_file(path),
                "generator_lr_trace_sha256": factorial._payload_sha256(
                    recipe["generator_lr_trace"]
                ),
                "discriminator_lr_trace_sha256": factorial._payload_sha256(
                    recipe["discriminator_lr_trace"]
                ),
                "development_training_contract_sha256": str(
                    contract["training_contract_sha256"]
                ),
                "best_learned_generator_sha256": str(
                    contract["best_learned_generator_sha256"]
                ),
                "best_learned_discriminator_sha256": str(
                    contract["best_learned_discriminator_sha256"]
                ),
                "training_config_sha256": str(contract["training_config_sha256"]),
                "code_manifest_sha256": str(contract["code_manifest_sha256"]),
            }
        )
    manifest = _self_hashed_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_kind": factorial.EXPERIMENT_KIND,
            "refit_mode": factorial.REFIT_MODE,
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": factorial._sha256_file(selection_path),
            "selection_payload_sha256": str(payload["selection_sha256"]),
            "recipe_count": len(rows),
            "development_matrix_sha256": str(payload["development_matrix_sha256"]),
            "recipes": rows,
        },
        "manifest_sha256",
    )
    path = _analysis_dir(experiment_root) / REFIT_RECIPE_MANIFEST
    factorial._write_json(path, manifest)
    return path


def _verify_refit_manifest(root: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(registry.get("refit_recipe_manifest_path", "")))
    expected_file_sha = str(registry.get("refit_recipe_manifest_sha256", ""))
    if not path.is_file() or factorial._sha256_file(path) != expected_file_sha:
        raise FilmCriticAnalysisError("Refit recipe manifest path/hash drift")
    canonical = _analysis_dir(root) / REFIT_RECIPE_MANIFEST
    if path.resolve() != canonical.resolve():
        raise FilmCriticAnalysisError(
            "Registry points to a non-canonical recipe manifest"
        )
    manifest = _mapping(path, "refit recipe manifest")
    _verify_self_hash(manifest, "manifest_sha256", "refit recipe manifest")
    rows = list(manifest.get("recipes") or [])
    if len(rows) != 16 or len({row.get("development_job_id") for row in rows}) != 16:
        raise FilmCriticAnalysisError("Refit recipe manifest must bind 16 jobs")
    for row in rows:
        recipe_path = Path(str(row.get("recipe_path", "")))
        if not recipe_path.is_file() or factorial._sha256_file(recipe_path) != str(
            row.get("recipe_sha256", "")
        ):
            raise FilmCriticAnalysisError(f"Refit recipe hash drift: {recipe_path}")
        recipe = _mapping(recipe_path, "refit recipe")
        factorial._validate_refit_recipe(recipe)
        if factorial._payload_sha256(recipe["generator_lr_trace"]) != row.get(
            "generator_lr_trace_sha256"
        ) or factorial._payload_sha256(recipe["discriminator_lr_trace"]) != row.get(
            "discriminator_lr_trace_sha256"
        ):
            raise FilmCriticAnalysisError(f"Refit LR trace hash drift: {recipe_path}")
    return manifest


def _validate_allowlist(
    root: Path, registry: Mapping[str, Any]
) -> list[dict[str, str]]:
    path = Path(str(registry.get("q4_allowlist_path", "")))
    if not path.is_file() or factorial._sha256_file(path) != str(
        registry.get("q4_allowlist_sha256", "")
    ):
        raise FilmCriticAnalysisError("Q4 checkpoint allowlist path/hash drift")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 32:
        raise FilmCriticAnalysisError("Q4 allowlist must have 16 G + 16 D checkpoints")
    keys = set()
    for row in rows:
        checkpoint = Path(str(row.get("checkpoint_path", "")))
        if (
            str(row.get("q4_allowed", "")).lower() != "true"
            or not checkpoint.is_file()
            or factorial._sha256_file(checkpoint)
            != str(row.get("checkpoint_sha256", ""))
        ):
            raise FilmCriticAnalysisError(
                f"Q4 allowlist checkpoint drift: {checkpoint}"
            )
        key = (str(row.get("job_id")), str(row.get("checkpoint_role")))
        if key in keys:
            raise FilmCriticAnalysisError(f"Duplicate Q4 allowlist key: {key}")
        keys.add(key)
    return rows


def _q4_gate(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate every upstream freeze before any Q4 workbook is opened."""

    registry = factorial._load_registry(root)
    required_true = (
        "selection_frozen",
        "refit_complete",
        "q4_gate_open",
        "q4_window_materialized",
    )
    missing = [field for field in required_true if not bool(registry.get(field))]
    if missing:
        raise FilmCriticAnalysisError(f"Q4 gate is closed/incomplete: {missing}")
    if bool(registry.get("q4_evaluated")):
        raise FilmCriticAnalysisError("Q4 has already been evaluated")
    selection_path = Path(str(registry.get("selection_path", "")))
    if not selection_path.is_file() or factorial._sha256_file(selection_path) != str(
        registry.get("selection_sha256", "")
    ):
        raise FilmCriticAnalysisError("Frozen selection path/hash drift")
    selection = _mapping(selection_path, "frozen selection")
    _verify_self_hash(selection, "selection_sha256", "frozen selection")
    refit_manifest = _verify_refit_manifest(root, registry)
    if (
        str(Path(str(refit_manifest.get("selection_path", ""))).resolve(strict=False))
        != str(selection_path.resolve())
        or str(refit_manifest.get("selection_sha256", ""))
        != factorial._sha256_file(selection_path)
        or str(refit_manifest.get("selection_payload_sha256", ""))
        != str(selection["selection_sha256"])
    ):
        raise FilmCriticAnalysisError("Refit manifest is not bound to frozen selection")
    allowlist = _validate_allowlist(root, registry)
    jobs = _stage_jobs(root, factorial.REFIT_STAGE)
    recipes = {
        str(row["development_job_id"]): row
        for row in _mapping(
            Path(str(registry["refit_recipe_manifest_path"])), "refit manifest"
        )["recipes"]
    }
    for job in jobs:
        parent_id = str(job.get("parent_development_job_id", ""))
        if not parent_id or parent_id not in recipes:
            raise FilmCriticAnalysisError(
                f"Refit job lacks its frozen parent development job: {job['job_id']}"
            )
        row = recipes[parent_id]
        frozen_factors = (
            str(row.get("generator_conditioning_mode")),
            str(row.get("critic_conditioning_mode")),
            str(row.get("text_ablation_mode")),
            int(row.get("tolerance_minutes", -1)),
            int(row.get("seed", -1)),
        )
        refit_factors = (
            str(job["generator_conditioning_mode"]),
            str(job["critic_conditioning_mode"]),
            str(job["text_ablation_mode"]),
            int(job["tolerance_minutes"]),
            int(job["seed"]),
        )
        if frozen_factors != refit_factors:
            raise FilmCriticAnalysisError(
                f"Refit job factor drift from development recipe: {job['job_id']}"
            )
        if str(
            Path(str(job.get("refit_recipe_path", ""))).resolve(strict=False)
        ) != str(Path(str(row["recipe_path"])).resolve()) or str(
            job.get("refit_recipe_sha256", "")
        ) != str(row["recipe_sha256"]):
            raise FilmCriticAnalysisError(f"Refit job recipe drift: {job['job_id']}")
    job_map = {str(job["job_id"]): job for job in jobs}
    expected_allowlist_keys = {
        (job_id, role)
        for job_id in job_map
        for role in ("generator_final", "discriminator_final")
    }
    observed_allowlist_keys = {
        (str(row["job_id"]), str(row["checkpoint_role"])) for row in allowlist
    }
    if observed_allowlist_keys != expected_allowlist_keys:
        raise FilmCriticAnalysisError("Q4 allowlist does not match the 16 refit jobs")
    for row in allowlist:
        job = job_map[str(row["job_id"])]
        if any(
            str(row[name]) != str(job[name])
            for name in (
                "generator_conditioning_mode",
                "critic_conditioning_mode",
                "text_ablation_mode",
                "tolerance_minutes",
                "seed",
            )
        ):
            raise FilmCriticAnalysisError(f"Q4 allowlist factor drift: {row['job_id']}")
        status = _status(root, job)
        checkpoint, checkpoint_sha = _artifact(status, str(row["checkpoint_role"]))
        if str(checkpoint) != str(
            Path(row["checkpoint_path"]).resolve()
        ) or checkpoint_sha != str(row["checkpoint_sha256"]):
            raise FilmCriticAnalysisError(
                f"Q4 allowlist/status checkpoint mismatch: {row['job_id']}"
            )
    return registry, selection, jobs


def _primary_q4_contrasts(
    pair_metrics: pd.DataFrame, selection: Mapping[str, Any], *, tolerance: int
) -> pd.DataFrame:
    winner = selection["winner"]
    winner_real = _cell(
        pair_metrics,
        str(winner["generator_conditioning_mode"]),
        str(winner["critic_conditioning_mode"]),
        REAL_TEXT,
        tolerance,
    )
    anchor = _cell(pair_metrics, *ANCHOR_ARCHITECTURE, REAL_TEXT, tolerance)
    winner_shuffle = _cell(
        pair_metrics,
        str(winner["generator_conditioning_mode"]),
        str(winner["critic_conditioning_mode"]),
        TEXT_SHUFFLE,
        tolerance,
    )
    persistence = winner_real.copy()
    rows: list[dict[str, Any]] = []
    definitions = (
        (
            "winner_minus_persistence",
            _contrast(
                winner_real,
                persistence,
                seed_offset=10_000 + tolerance,
                left_column="model_mae",
                right_column="persistence_mae",
            ),
        ),
        (
            "winner_minus_anchor",
            _contrast(winner_real, anchor, seed_offset=11_000 + tolerance),
        ),
        (
            "winner_real_minus_winner_shuffle",
            _contrast(winner_real, winner_shuffle, seed_offset=12_000 + tolerance),
        ),
    )
    for name, summary in definitions:
        rows.append({"contrast": name, "tolerance_minutes": tolerance, **summary})
    gdt = _factorial_pair_contrast(
        pair_metrics, tolerance=tolerance, factors=("G", "D", "T")
    )
    rows.append(
        {
            "contrast": "generator_x_critic_x_text_interaction",
            "tolerance_minutes": tolerance,
            **session_cluster_bootstrap(
                gdt["difference"],
                gdt["session_id"],
                seed=BOOTSTRAP_SEED + 13_000 + tolerance,
            ),
        }
    )
    output = pd.DataFrame(rows)
    adjusted = holm_adjust(output["p_two_sided"].astype(float).tolist())
    output["holm_p"] = adjusted
    output["holm_family"] = (
        "q4_primary_holm4" if tolerance == 5 else "secondary_not_primary_family"
    )
    output["holm_significant"] = (
        (output["ci_95_lower"].astype(float) > 0.0)
        | (output["ci_95_upper"].astype(float) < 0.0)
    ) & (output["holm_p"].astype(float) < 0.05)
    output["improvement_supported"] = (output["ci_95_upper"].astype(float) < 0.0) & (
        output["holm_p"].astype(float) < 0.05
    )
    return output


def _historical_q4_overlap(panel: pd.DataFrame) -> dict[str, Any]:
    path = HISTORICAL_Q4_PREDICTIONS
    if not path.is_file():
        raise FilmCriticAnalysisError(
            f"Historical Q4 exposure evidence is missing: {path}"
        )
    previous = pd.read_csv(path, usecols=["pair_id", "session_id"], low_memory=False)
    current_pairs = set(panel["pair_id"].astype(str))
    previous_pairs = set(previous["pair_id"].astype(str))
    current_sessions = set(panel["session_id"].astype(str))
    previous_sessions = set(previous["session_id"].astype(str))
    pair_overlap = current_pairs & previous_pairs
    session_overlap = current_sessions & previous_sessions
    if len(current_pairs) != 143 or len(pair_overlap) != 143:
        raise FilmCriticAnalysisError(
            "Historical Q4 overlap drifted; expected 143/143 exact-grid pairs"
        )
    return {
        "historical_prediction_path": str(path.resolve()),
        "historical_prediction_sha256": factorial._sha256_file(path),
        "current_pair_count": len(current_pairs),
        "overlapping_pair_count": len(pair_overlap),
        "pair_overlap_label": "143/143",
        "current_session_count": len(current_sessions),
        "overlapping_session_count": len(session_overlap),
        "historically_exposed": True,
        "interpretation": "retrospective_frozen_exploratory_not_confirmatory",
    }


def run_film_critic_q4_analysis(
    root: str | Path,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen 16-checkpoint refit matrix after the explicit Q4 gate."""

    experiment_root = Path(root).resolve(strict=False)
    # This call must remain before _resolved/_q4_panel: tests enforce zero Q4 reads
    # whenever an upstream hash, recipe, checkpoint, or gate condition is invalid.
    registry, selection, jobs = _q4_gate(experiment_root)
    resolved = _resolved(experiment_root)
    common_q4, common_lineage = _q4_panel(experiment_root, resolved)
    jobs_5m = [job for job in jobs if int(job["tolerance_minutes"]) == 5]
    jobs_30m = [job for job in jobs if int(job["tolerance_minutes"]) == 30]
    if len(jobs_5m) != 8 or len(jobs_30m) != 8:
        raise FilmCriticAnalysisError("Q4 evaluation requires 8 jobs per tolerance")
    metrics_5m = _evaluate_jobs(
        experiment_root,
        jobs_5m,
        panel=common_q4,
        panel_name="film_critic_q4_common_05m",
        stage="q4_refit_final_05m",
        checkpoint_role="generator_final",
        mc_samples=int(resolved["split"]["q4_mc_samples"]),
        evaluator=evaluator,
    )
    metrics_30m = _evaluate_jobs(
        experiment_root,
        jobs_30m,
        panel=common_q4,
        panel_name="film_critic_q4_common_05m_training_30m_secondary",
        stage="q4_refit_final_30m",
        checkpoint_role="generator_final",
        mc_samples=int(resolved["split"]["q4_mc_samples"]),
        evaluator=evaluator,
    )
    pair_metrics = pd.concat([metrics_5m, metrics_30m], ignore_index=True)
    analysis_dir = _analysis_dir(experiment_root)
    pair_path = _gzip_csv(analysis_dir / Q4_PAIR_METRICS, pair_metrics)
    primary = _primary_q4_contrasts(metrics_5m, selection, tolerance=5)
    primary_path = _write_csv(analysis_dir / Q4_PRIMARY_CONTRASTS, primary)
    secondary = _primary_q4_contrasts(metrics_30m, selection, tolerance=30)
    secondary["analysis_role"] = "secondary_lagged_news_robustness"
    secondary_path = _write_csv(analysis_dir / Q4_SECONDARY, secondary)
    exposure = _historical_q4_overlap(common_q4)
    secondary_lineage = {
        **common_lineage,
        "analysis_role": "training_tolerance_30m_secondary_robustness",
        "evaluation_panel_tolerance_minutes": 5,
        "broad_30m_q4_loader_created": False,
        "broad_30m_q4_rows_passed_to_evaluator": 0,
    }
    summary = _self_hashed_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_kind": factorial.EXPERIMENT_KIND,
            "evaluation_stage": "q4_locked_evaluation",
            "selection_sha256": str(selection["selection_sha256"]),
            "selection_file_sha256": factorial._sha256_file(
                Path(str(registry["selection_path"]))
            ),
            "refit_recipe_manifest_sha256": str(
                registry["refit_recipe_manifest_sha256"]
            ),
            "q4_checkpoint_allowlist_sha256": str(registry["q4_allowlist_sha256"]),
            "winner": dict(selection["winner"]),
            "single_seed": SEED,
            "seed_uncertainty_estimated": False,
            "bootstrap_unit": "cme_session_cluster_single_seed",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "q4_mc_samples": int(resolved["split"]["q4_mc_samples"]),
            "q4_primary_holm_family_size": 4,
            "q4_confirmation_label": "frozen_exploratory_out_of_time",
            "confirmatory_claim_permitted": False,
            "historical_q4_exposure": exposure,
            "panel_lineage": {
                "5m_primary": {
                    **common_lineage,
                    "analysis_role": "training_tolerance_5m_primary",
                },
                "30m_secondary": secondary_lineage,
            },
            "artifact_sha256": {
                Q4_PAIR_METRICS: factorial._sha256_file(pair_path),
                Q4_PRIMARY_CONTRASTS: factorial._sha256_file(primary_path),
                Q4_SECONDARY: factorial._sha256_file(secondary_path),
            },
            "q4_loader_created": True,
            "q4_predictions_generated": True,
            "q4_evaluated": True,
        },
        "summary_sha256",
    )
    factorial._write_json(analysis_dir / Q4_SUMMARY, summary)
    return summary


__all__ = [
    "FilmCriticAnalysisError",
    "freeze_refit_recipes",
    "run_film_critic_q3_analysis",
    "run_film_critic_q4_analysis",
    "session_cluster_bootstrap",
]
