"""Q3 paired analysis for the deterministic-zero versus Gaussian WGAN ablation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from scripts.rq3.news_first_vol_comparison_analysis import (
    RunSpec,
    TrainedRunEvaluator,
    _load_panel_source,
    aggregate_pair_metrics,
    cme_session_cluster_bootstrap,
    compute_sample_metrics,
    holm_adjust,
)
from scripts.rq3.news_first_vol_zero_noise_ablation import (
    CAPACITY_PROFILE,
    EXPERIMENT_KIND,
    EXPERIMENT_STAGE,
    FIXED_LEARNING_RATE,
    FROZEN_SEED,
    FROZEN_TEXT_MODES,
    FROZEN_TOLERANCES,
    GENERATOR_NOISE_MODE,
    NOISE_DIM,
    REFERENCE_JOB_IDS,
    REFERENCE_NOISE_MODE,
    _job_id,
    _load_registry,
    _read_json,
    _sha256_file,
    _validate_job_lineage,
)
from wgan_option.models.common import generator_noise_fingerprint


Q3_START_UTC = "2023-07-01T00:00:00Z"
Q3_END_UTC = "2023-10-01T00:00:00Z"
SELECTION_PANEL = "common_validation_05m"
EXPECTED_PAIR_COUNT = 123
EXPECTED_SESSION_COUNT = 33
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260820


class ZeroNoiseAnalysisError(ValueError):
    """Raised when zero/Gaussian evidence cannot support paired inference."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(
    frame: pd.DataFrame, path: Path, *, compression: str | None = None
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression=compression)
    return path


def _load_jobs(root: Path) -> list[dict[str, Any]]:
    registry = _load_registry(root)
    if str(registry.get("experiment_stage", "")) != EXPERIMENT_STAGE:
        raise ZeroNoiseAnalysisError("Zero-noise registry stage mismatch")
    jobs = []
    for raw in registry.get("jobs", []):
        job = dict(raw)
        _validate_job_lineage(root, job)
        status_path = root / "registry" / "jobs" / f"{job['job_id']}.status.json"
        status = _read_json(status_path)
        if status.get("status") != "completed":
            raise ZeroNoiseAnalysisError(f"Incomplete zero-noise job: {job['job_id']}")
        if status.get("config_sha256") != job.get("config_sha256"):
            raise ZeroNoiseAnalysisError("Zero-noise status/config lineage mismatch")
        artifacts = list(status.get("artifacts") or [])
        if not artifacts:
            raise ZeroNoiseAnalysisError(
                "Completed zero-noise job lacks artifact hashes"
            )
        for artifact in artifacts:
            path = Path(str(artifact.get("path", "")))
            if not path.is_file() or _sha256_file(path) != str(
                artifact.get("sha256", "")
            ):
                raise ZeroNoiseAnalysisError(
                    f"Zero-noise artifact hash drift: {job['job_id']}"
                )
        jobs.append({**job, **status})
    expected = {
        _job_id(mode, tolerance)
        for mode in FROZEN_TEXT_MODES
        for tolerance in FROZEN_TOLERANCES
    }
    if len(jobs) != 4 or {str(job["job_id"]) for job in jobs} != expected:
        raise ZeroNoiseAnalysisError("Analysis requires the exact four zero-noise jobs")
    return jobs


def _training_config(job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(str(job["training_config_path"]))
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ZeroNoiseAnalysisError(f"Malformed training config: {path}")
    return path, dict(value)


def _learned_metadata(job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(str(job["run_dir"])) / "metrics" / "best_learned_checkpoint.json"
    value = _read_json(path)
    if (
        int(value.get("best_epoch", -1)) < 1
        or str(value.get("selection_scope", "")) != "trained_epochs_only"
    ):
        raise ZeroNoiseAnalysisError("Q3 analysis requires best-learned epoch >=1")
    expected_fingerprint = generator_noise_fingerprint(GENERATOR_NOISE_MODE, NOISE_DIM)
    if (
        str(value.get("generator_noise_mode", "")) != GENERATOR_NOISE_MODE
        or str(value.get("generator_noise_fingerprint", "")) != expected_fingerprint
    ):
        raise ZeroNoiseAnalysisError("Best-learned zero-noise metadata drifted")
    return path, value


def _run_spec(job: Mapping[str, Any]) -> RunSpec:
    metadata_path, metadata = _learned_metadata(job)
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ZeroNoiseAnalysisError("Learned metadata lacks checkpoint artifacts")
    checkpoint = Path(str(artifacts.get("generator", "")))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_path, config = _training_config(job)
    return RunSpec(
        run_id=str(job["job_id"]),
        run_dir=Path(str(job["run_dir"])),
        model="wgan",
        tolerance_minutes=int(job["tolerance_minutes"]),
        seed=FROZEN_SEED,
        checkpoint_path=checkpoint,
        text_ablation_mode=str(job["text_ablation_mode"]),
        support_mask_mode="raw_joint",
        manifest_path=config_path,
        metadata={
            **config,
            "generator_noise_mode": GENERATOR_NOISE_MODE,
            "generator_noise_fingerprint": generator_noise_fingerprint(
                GENERATOR_NOISE_MODE, NOISE_DIM
            ),
            "best_learned_metadata_path": str(metadata_path),
        },
    )


def _q3_panel(jobs: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    sources = set()
    for job in jobs:
        _, config = _training_config(job)
        sources.add(
            (
                str(config.get("news_first_common_eval_data_path", "")),
                str(config.get("sheet_name", "gan_input_ready")),
                str(config.get("support_mask_mode", "none")),
            )
        )
    if len(sources) != 1:
        raise ZeroNoiseAnalysisError("Zero-noise jobs disagree on the Q3 panel")
    raw_path, sheet_name, support_mode = next(iter(sources))
    workbook = Path(raw_path).resolve()
    raw = pd.read_excel(workbook, sheet_name=sheet_name)
    timestamps = pd.to_datetime(raw["effective_origin_utc"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ZeroNoiseAnalysisError("Invalid Q3 timestamps")
    start, end = pd.Timestamp(Q3_START_UTC), pd.Timestamp(Q3_END_UTC)
    q3 = raw.loc[(timestamps >= start) & (timestamps < end)].copy()
    panel, lineage, _ = _load_panel_source(
        q3,
        sheet_name=sheet_name,
        panel_name=SELECTION_PANEL,
        freeze_test_window=False,
        enforce_expected_counts=False,
        support_mask_mode=support_mode,
    )
    selected = pd.to_datetime(panel["effective_origin_utc"], errors="coerce", utc=True)
    if selected.isna().any() or bool(
        (selected < start).any() or (selected >= end).any()
    ):
        raise ZeroNoiseAnalysisError("A non-Q3 row reached zero-noise evaluation")
    lineage.update(
        {
            "path": str(workbook),
            "sha256": _sha256_file(workbook),
            "interval_start_utc": Q3_START_UTC,
            "interval_end_utc_exclusive": Q3_END_UTC,
            "q4_rows_passed_to_evaluator": 0,
        }
    )
    return panel, lineage


def evaluate_zero_q3_pair_metrics(
    experiment_root: str | Path,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
    prediction_artifact_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate four learned zero-noise checkpoints on the common Q3 panel."""

    root = Path(experiment_root).resolve()
    jobs = _load_jobs(root)
    panel, lineage = _q3_panel(jobs)
    production = evaluator or TrainedRunEvaluator(mc_samples=16)
    parts = []
    prediction_artifacts: list[dict[str, Any]] = []
    expected_fingerprint = generator_noise_fingerprint(GENERATOR_NOISE_MODE, NOISE_DIM)
    for job in jobs:
        spec = _run_spec(job)
        predictions = production(spec, SELECTION_PANEL, panel.copy())
        required_prediction = {
            "prediction_mc_samples",
            "prediction_fallback",
            "generator_noise_mode",
            "generator_noise_fingerprint",
        }
        missing = sorted(required_prediction - set(predictions.columns))
        if missing:
            raise ZeroNoiseAnalysisError(f"Prediction lineage missing {missing}")
        if (
            not pd.to_numeric(predictions["prediction_mc_samples"], errors="coerce")
            .eq(1)
            .all()
        ):
            raise ZeroNoiseAnalysisError("Zero-noise Q3 evaluation must be single-pass")
        if predictions["prediction_fallback"].astype(bool).any():
            raise ZeroNoiseAnalysisError("Prediction fallback is forbidden")
        if (
            not predictions["generator_noise_mode"]
            .astype(str)
            .eq(GENERATOR_NOISE_MODE)
            .all()
            or not predictions["generator_noise_fingerprint"]
            .astype(str)
            .eq(expected_fingerprint)
            .all()
        ):
            raise ZeroNoiseAnalysisError("Q3 predictions lost zero-noise lineage")
        if len(predictions) != len(panel):
            raise ZeroNoiseAnalysisError("Zero prediction export lost Q3 rows")
        if prediction_artifact_dir is not None:
            directory = Path(prediction_artifact_dir)
            directory.mkdir(parents=True, exist_ok=True)
            prediction_export = predictions.copy()
            prediction_export.insert(0, "run_id", spec.run_id)
            prediction_export.insert(1, "panel", SELECTION_PANEL)
            prediction_export.insert(2, "tolerance_minutes", spec.tolerance_minutes)
            path = _write_csv(
                prediction_export,
                directory / f"{spec.run_id}_q3_predictions.csv.gz",
                compression="gzip",
            )
            prediction_artifacts.append(
                {
                    "role": f"q3_predictions:{spec.run_id}",
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "row_count": int(len(prediction_export)),
                    "prediction_mc_samples": 1,
                    "prediction_fallback": False,
                    "generator_noise_mode": GENERATOR_NOISE_MODE,
                    "generator_noise_fingerprint": expected_fingerprint,
                }
            )
        samples, exclusions, _ = compute_sample_metrics(
            spec,
            SELECTION_PANEL,
            panel,
            predictions,
            evaluate_embedded_atm_skew=False,
        )
        if not exclusions.empty:
            raise ZeroNoiseAnalysisError(
                f"Q3 exclusions for {spec.run_id}: "
                f"{exclusions['exclusion_code'].value_counts().to_dict()}"
            )
        pairs = aggregate_pair_metrics(samples)
        pairs = pairs[
            pairs["stratum_type"].astype(str).eq("overall")
            & pairs["stratum_value"].astype(str).eq("all")
        ].copy()
        pairs.insert(0, "zero_job_id", str(job["job_id"]))
        pairs.insert(
            1, "gaussian_reference_job_id", str(job["gaussian_reference_job_id"])
        )
        pairs.insert(2, "generator_noise_mode", GENERATOR_NOISE_MODE)
        pairs.insert(3, "generator_noise_fingerprint", expected_fingerprint)
        pairs.insert(4, "prediction_mc_samples", 1)
        parts.append(pairs)
    output = pd.concat(parts, ignore_index=True)
    lineage.update(
        {
            "evaluated_run_count": 4,
            "q4_used_for_analysis": False,
            "q4_predictions_generated": False,
            "prediction_artifacts": prediction_artifacts,
        }
    )
    return validate_zero_pair_metrics(output), lineage


def validate_zero_pair_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "zero_job_id",
        "gaussian_reference_job_id",
        "generator_noise_mode",
        "generator_noise_fingerprint",
        "prediction_mc_samples",
        "text_ablation_mode",
        "tolerance_minutes",
        "pair_id",
        "session_id",
        "model_mae",
        "persistence_mae",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ZeroNoiseAnalysisError(f"Zero pair metrics missing {missing}")
    output = frame.copy()
    output["tolerance_minutes"] = pd.to_numeric(
        output["tolerance_minutes"], errors="coerce"
    ).astype(int)
    if set(output["generator_noise_mode"].astype(str)) != {GENERATOR_NOISE_MODE}:
        raise ZeroNoiseAnalysisError("Zero pair-metric mode drifted")
    if set(output["generator_noise_fingerprint"].astype(str)) != {
        generator_noise_fingerprint(GENERATOR_NOISE_MODE, NOISE_DIM)
    }:
        raise ZeroNoiseAnalysisError("Zero pair-metric fingerprint drifted")
    if not pd.to_numeric(output["prediction_mc_samples"], errors="coerce").eq(1).all():
        raise ZeroNoiseAnalysisError("Zero pair-metric MC count drifted")
    job_keys = ["text_ablation_mode", "tolerance_minutes"]
    expected = {
        (mode, tolerance)
        for mode in FROZEN_TEXT_MODES
        for tolerance in FROZEN_TOLERANCES
    }
    observed = set(output[job_keys].itertuples(index=False, name=None))
    if observed != expected or len(output) != 4 * EXPECTED_PAIR_COUNT:
        raise ZeroNoiseAnalysisError("Zero pair metrics are not the exact 4x123 matrix")
    coverage = []
    for key, group in output.groupby(job_keys, sort=False):
        if (
            group["pair_id"].nunique() != EXPECTED_PAIR_COUNT
            or group["session_id"].nunique() != EXPECTED_SESSION_COUNT
        ):
            raise ZeroNoiseAnalysisError(f"Zero Q3 coverage drifted: {key}")
        if group.duplicated("pair_id").any():
            raise ZeroNoiseAnalysisError(f"Duplicate zero pair: {key}")
        coverage.append(
            frozenset(
                group[["pair_id", "session_id"]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
        )
    if len(set(coverage)) != 1:
        raise ZeroNoiseAnalysisError("Zero cells do not share one paired Q3 panel")
    for column in ("model_mae", "persistence_mae"):
        values = pd.to_numeric(output[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or bool((values <= 0.0).any()):
            raise ZeroNoiseAnalysisError(f"Invalid zero {column}")
    return output.reset_index(drop=True)


def load_gaussian_pair_metrics(experiment_root: str | Path) -> pd.DataFrame:
    root = Path(experiment_root).resolve()
    manifest = pd.read_csv(root / "gaussian_reference_manifest.csv", low_memory=False)
    if len(manifest) != 4:
        raise ZeroNoiseAnalysisError("Gaussian reference manifest is incomplete")
    source_paths = manifest["reference_q3_pair_metrics_path"].astype(str).unique()
    source_hashes = manifest["reference_q3_pair_metrics_sha256"].astype(str).unique()
    if len(source_paths) != 1 or len(source_hashes) != 1:
        raise ZeroNoiseAnalysisError(
            "Gaussian pair-metric source lineage differs by cell"
        )
    path = Path(source_paths[0])
    if _sha256_file(path) != source_hashes[0]:
        raise ZeroNoiseAnalysisError("Gaussian pair-metric source hash changed")
    frame = pd.read_csv(path, low_memory=False)
    selected = frame[frame["run_id"].astype(str).isin(REFERENCE_JOB_IDS)].copy()
    expected = {
        (mode, tolerance)
        for mode in FROZEN_TEXT_MODES
        for tolerance in FROZEN_TOLERANCES
    }
    keys = ["text_ablation_mode", "tolerance_minutes"]
    observed = set(selected[keys].itertuples(index=False, name=None))
    if observed != expected or len(selected) != 4 * EXPECTED_PAIR_COUNT:
        raise ZeroNoiseAnalysisError(
            "Gaussian pair metrics are not the exact 4x123 matrix"
        )
    reference_map = {
        (str(row.text_ablation_mode), int(row.tolerance_minutes)): str(
            row.reference_job_id
        )
        for row in manifest.itertuples()
    }
    for key, group in selected.groupby(keys, sort=False):
        if set(group["run_id"].astype(str)) != {reference_map[key]}:
            raise ZeroNoiseAnalysisError(f"Gaussian run ID mismatch: {key}")
        if (
            group["pair_id"].nunique() != EXPECTED_PAIR_COUNT
            or group["session_id"].nunique() != EXPECTED_SESSION_COUNT
        ):
            raise ZeroNoiseAnalysisError(f"Gaussian Q3 coverage drifted: {key}")
    selected.insert(0, "generator_noise_mode", REFERENCE_NOISE_MODE)
    selected.insert(
        1,
        "generator_noise_fingerprint",
        generator_noise_fingerprint(REFERENCE_NOISE_MODE, NOISE_DIM),
    )
    selected.insert(2, "prediction_mc_samples", 16)
    return selected.reset_index(drop=True)


def build_paired_pair_metrics(
    zero_pair_metrics: pd.DataFrame,
    gaussian_pair_metrics: pd.DataFrame,
) -> pd.DataFrame:
    zero = validate_zero_pair_metrics(zero_pair_metrics)
    gaussian = gaussian_pair_metrics.copy()
    keys = ["text_ablation_mode", "tolerance_minutes", "pair_id", "session_id"]
    gaussian_required = set(keys) | {"run_id", "model_mae", "persistence_mae"}
    missing = sorted(gaussian_required - set(gaussian.columns))
    if missing:
        raise ZeroNoiseAnalysisError(f"Gaussian pair metrics missing {missing}")
    if gaussian.duplicated(keys).any() or len(gaussian) != len(zero):
        raise ZeroNoiseAnalysisError("Gaussian pair metrics are not one-to-one")
    left = zero[
        keys
        + [
            "zero_job_id",
            "gaussian_reference_job_id",
            "model_mae",
            "persistence_mae",
        ]
    ].rename(
        columns={
            "model_mae": "zero_mae",
            "persistence_mae": "zero_persistence_mae",
        }
    )
    right = gaussian[keys + ["run_id", "model_mae", "persistence_mae"]].rename(
        columns={
            "run_id": "gaussian_run_id",
            "model_mae": "gaussian_mae",
            "persistence_mae": "gaussian_persistence_mae",
        }
    )
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(paired) != len(zero):
        raise ZeroNoiseAnalysisError("Zero/Gaussian join lost paired rows")
    if not (paired["gaussian_run_id"] == paired["gaussian_reference_job_id"]).all():
        raise ZeroNoiseAnalysisError(
            "Gaussian joined run ID differs from frozen reference"
        )
    if not np.allclose(
        paired["zero_persistence_mae"],
        paired["gaussian_persistence_mae"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ZeroNoiseAnalysisError(
            "Persistence differs between paired evidence sources"
        )
    paired["persistence_mae"] = paired["zero_persistence_mae"]
    paired["zero_minus_gaussian"] = paired["zero_mae"] - paired["gaussian_mae"]
    paired["zero_minus_persistence"] = paired["zero_mae"] - paired["persistence_mae"]
    paired["gaussian_minus_persistence"] = (
        paired["gaussian_mae"] - paired["persistence_mae"]
    )
    return paired.sort_values(keys, kind="stable").reset_index(drop=True)


def build_cell_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Summarize the four exact paired cells without article duplication."""

    rows: list[dict[str, Any]] = []
    for (mode, tolerance), group in paired.groupby(
        ["text_ablation_mode", "tolerance_minutes"], sort=True
    ):
        rows.append(
            {
                "text_ablation_mode": str(mode),
                "tolerance_minutes": int(tolerance),
                "pair_count": int(len(group)),
                "session_count": int(group["session_id"].nunique()),
                "zero_mae": float(group["zero_mae"].mean()),
                "gaussian_mae": float(group["gaussian_mae"].mean()),
                "persistence_mae": float(group["persistence_mae"].mean()),
                "zero_minus_gaussian": float(group["zero_minus_gaussian"].mean()),
                "zero_minus_persistence": float(group["zero_minus_persistence"].mean()),
                "gaussian_minus_persistence": float(
                    group["gaussian_minus_persistence"].mean()
                ),
                "zero_beats_gaussian_pair_rate": float(
                    group["zero_minus_gaussian"].lt(0.0).mean()
                ),
                "zero_beats_persistence_pair_rate": float(
                    group["zero_minus_persistence"].lt(0.0).mean()
                ),
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != 4:
        raise ZeroNoiseAnalysisError("Cell summary requires four exact cells")
    return output


def _bootstrap_row(
    group: pd.DataFrame,
    *,
    difference_column: str,
    comparison: str,
    mode: str,
    tolerance_scope: str,
    analysis_role: str,
    family: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    result = cme_session_cluster_bootstrap(
        group[difference_column],
        group["session_id"].astype(str),
        iterations=iterations,
        seed=seed,
    )
    expected_pairs = EXPECTED_PAIR_COUNT * (
        2 if tolerance_scope == "combined_05m_30m" else 1
    )
    if (
        int(result["pair_count"]) != expected_pairs
        or int(result["session_count"]) != EXPECTED_SESSION_COUNT
    ):
        raise ZeroNoiseAnalysisError(
            f"Bootstrap panel drifted for {comparison}/{mode}/{tolerance_scope}"
        )
    return {
        "comparison": comparison,
        "text_ablation_mode": mode,
        "tolerance_scope": tolerance_scope,
        "analysis_role": analysis_role,
        "holm_family": family,
        "difference_definition": difference_column,
        "bootstrap_method": "paired_CME_session_cluster",
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
        **result,
    }


def build_zero_gaussian_bootstrap(
    paired: pd.DataFrame,
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Build the two primary combined and four sensitivity comparisons."""

    rows: list[dict[str, Any]] = []
    for mode in FROZEN_TEXT_MODES:
        group = paired[paired["text_ablation_mode"].astype(str).eq(mode)]
        rows.append(
            _bootstrap_row(
                group,
                difference_column="zero_minus_gaussian",
                comparison="zero_minus_gaussian",
                mode=mode,
                tolerance_scope="combined_05m_30m",
                analysis_role="primary",
                family="primary_combined_modes",
                iterations=iterations,
                seed=seed,
            )
        )
        for tolerance in FROZEN_TOLERANCES:
            cell = group[group["tolerance_minutes"].eq(tolerance)]
            rows.append(
                _bootstrap_row(
                    cell,
                    difference_column="zero_minus_gaussian",
                    comparison="zero_minus_gaussian",
                    mode=mode,
                    tolerance_scope=f"{tolerance:02d}m",
                    analysis_role="secondary_sensitivity",
                    family="secondary_per_tolerance",
                    iterations=iterations,
                    seed=seed,
                )
            )
    output = pd.DataFrame(rows)
    for family, indexes in output.groupby("holm_family", sort=False).groups.items():
        del family
        adjusted = holm_adjust(output.loc[indexes, "p_two_sided"].tolist())
        output.loc[indexes, "holm_adjusted_p"] = adjusted
    return output.sort_values(
        ["analysis_role", "text_ablation_mode", "tolerance_scope"], kind="stable"
    ).reset_index(drop=True)


def build_persistence_bootstrap(
    paired: pd.DataFrame,
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Report each noise treatment relative to persistence as secondary evidence."""

    rows: list[dict[str, Any]] = []
    for noise_mode, difference_column in (
        (GENERATOR_NOISE_MODE, "zero_minus_persistence"),
        (REFERENCE_NOISE_MODE, "gaussian_minus_persistence"),
    ):
        for mode in FROZEN_TEXT_MODES:
            group = paired[paired["text_ablation_mode"].astype(str).eq(mode)]
            rows.append(
                {
                    "generator_noise_mode": noise_mode,
                    **_bootstrap_row(
                        group,
                        difference_column=difference_column,
                        comparison=f"{noise_mode}_minus_persistence",
                        mode=mode,
                        tolerance_scope="combined_05m_30m",
                        analysis_role="secondary",
                        family=f"{noise_mode}_combined_modes",
                        iterations=iterations,
                        seed=seed,
                    ),
                }
            )
            for tolerance in FROZEN_TOLERANCES:
                cell = group[group["tolerance_minutes"].eq(tolerance)]
                rows.append(
                    {
                        "generator_noise_mode": noise_mode,
                        **_bootstrap_row(
                            cell,
                            difference_column=difference_column,
                            comparison=f"{noise_mode}_minus_persistence",
                            mode=mode,
                            tolerance_scope=f"{tolerance:02d}m",
                            analysis_role="secondary_sensitivity",
                            family=f"{noise_mode}_per_tolerance",
                            iterations=iterations,
                            seed=seed,
                        ),
                    }
                )
    output = pd.DataFrame(rows)
    for _, indexes in output.groupby("holm_family", sort=False).groups.items():
        output.loc[indexes, "holm_adjusted_p"] = holm_adjust(
            output.loc[indexes, "p_two_sided"].tolist()
        )
    return output.sort_values(
        [
            "generator_noise_mode",
            "analysis_role",
            "text_ablation_mode",
            "tolerance_scope",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _validate_analysis_tables(
    paired: pd.DataFrame,
    cell_summary: pd.DataFrame,
    zero_gaussian: pd.DataFrame,
    persistence: pd.DataFrame,
) -> None:
    if len(paired) != 4 * EXPECTED_PAIR_COUNT:
        raise ZeroNoiseAnalysisError("Paired output is not the exact 4x123 matrix")
    if len(cell_summary) != 4 or len(zero_gaussian) != 6 or len(persistence) != 12:
        raise ZeroNoiseAnalysisError("Analysis table cardinality drifted")
    primary = zero_gaussian[zero_gaussian["analysis_role"].eq("primary")]
    if len(primary) != 2 or set(primary["tolerance_scope"]) != {"combined_05m_30m"}:
        raise ZeroNoiseAnalysisError("Primary comparison family drifted")
    for frame in (zero_gaussian, persistence):
        if not frame["bootstrap_method"].eq("paired_CME_session_cluster").all():
            raise ZeroNoiseAnalysisError("Only paired session bootstrap is permitted")
        if not frame["bootstrap_iterations"].eq(DEFAULT_BOOTSTRAP_ITERATIONS).all():
            raise ZeroNoiseAnalysisError("Bootstrap iteration count drifted")
    if (
        set(paired["session_id"].astype(str))
        and paired["session_id"].nunique() != EXPECTED_SESSION_COUNT
    ):
        raise ZeroNoiseAnalysisError("Q3 session panel drifted")


def run_zero_noise_analysis(
    experiment_root: str | Path,
    *,
    zero_pair_metrics: pd.DataFrame | None = None,
    gaussian_pair_metrics: pd.DataFrame | None = None,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Path:
    """Run the Q3-only paired analysis and persist complete lineage."""

    if int(bootstrap_iterations) != DEFAULT_BOOTSTRAP_ITERATIONS:
        raise ZeroNoiseAnalysisError("Formal analysis is frozen to 10,000 draws")
    root = Path(experiment_root).resolve(strict=False)
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if zero_pair_metrics is None:
        zero, panel_lineage = evaluate_zero_q3_pair_metrics(
            root,
            evaluator=evaluator,
            prediction_artifact_dir=analysis_dir / "q3_predictions",
        )
        registry = _load_registry(root)
        if registry.get("experiment_kind") != EXPERIMENT_KIND:
            raise ZeroNoiseAnalysisError("Analysis root experiment kind mismatch")
        resolved_config_sha = str(registry["resolved_config_sha256"])
        reference_manifest_sha = _sha256_file(root / "gaussian_reference_manifest.csv")
    else:
        zero = validate_zero_pair_metrics(zero_pair_metrics)
        panel_lineage = {
            "fixture_injected": True,
            "interval_start_utc": Q3_START_UTC,
            "interval_end_utc_exclusive": Q3_END_UTC,
            "q4_rows_passed_to_evaluator": 0,
            "q4_used_for_analysis": False,
            "q4_predictions_generated": False,
        }
        resolved_config_sha = "fixture"
        reference_manifest_sha = "fixture"
    gaussian = (
        load_gaussian_pair_metrics(root)
        if gaussian_pair_metrics is None
        else gaussian_pair_metrics.copy()
    )
    paired = build_paired_pair_metrics(zero, gaussian)
    cell_summary = build_cell_summary(paired)
    zero_gaussian = build_zero_gaussian_bootstrap(
        paired, iterations=bootstrap_iterations, seed=bootstrap_seed
    )
    persistence = build_persistence_bootstrap(
        paired, iterations=bootstrap_iterations, seed=bootstrap_seed
    )
    _validate_analysis_tables(paired, cell_summary, zero_gaussian, persistence)

    paths = {
        "zero_pair_metrics": _write_csv(
            zero,
            analysis_dir / "zero_noise_q3_pair_metrics.csv.gz",
            compression="gzip",
        ),
        "paired_pair_metrics": _write_csv(
            paired,
            analysis_dir / "zero_noise_gaussian_paired_pair_metrics.csv.gz",
            compression="gzip",
        ),
        "cell_summary": _write_csv(
            cell_summary, analysis_dir / "zero_noise_cell_summary.csv"
        ),
        "zero_vs_gaussian_bootstrap": _write_csv(
            zero_gaussian,
            analysis_dir / "zero_noise_vs_gaussian_bootstrap.csv",
        ),
        "models_vs_persistence_bootstrap": _write_csv(
            persistence,
            analysis_dir / "noise_models_vs_persistence_bootstrap.csv",
        ),
    }
    for artifact in panel_lineage.get("prediction_artifacts", []):
        paths[str(artifact["role"])] = Path(str(artifact["path"]))
    primary = zero_gaussian[zero_gaussian["analysis_role"].eq("primary")]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "zero_noise_q3_paired_analysis",
        "experiment_kind": EXPERIMENT_KIND,
        "experiment_stage": EXPERIMENT_STAGE,
        "capacity_profile": CAPACITY_PROFILE,
        "learning_rate": FIXED_LEARNING_RATE,
        "seed": FROZEN_SEED,
        "generator_noise_mode": GENERATOR_NOISE_MODE,
        "generator_noise_fingerprint": generator_noise_fingerprint(
            GENERATOR_NOISE_MODE, NOISE_DIM
        ),
        "zero_prediction_mc_samples": 1,
        "gaussian_reference_noise_mode": REFERENCE_NOISE_MODE,
        "gaussian_reference_noise_fingerprint": generator_noise_fingerprint(
            REFERENCE_NOISE_MODE, NOISE_DIM
        ),
        "gaussian_reference_prediction_mc_samples": 16,
        "resolved_config_sha256": resolved_config_sha,
        "gaussian_reference_manifest_sha256": reference_manifest_sha,
        "q3_pair_count": EXPECTED_PAIR_COUNT,
        "q3_session_count": EXPECTED_SESSION_COUNT,
        "zero_job_count": 4,
        "bootstrap_iterations": DEFAULT_BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_layers": ["CME_session"],
        "seed_bootstrap_used": False,
        "primary_family": "two combined 5m/30m mode comparisons; Holm over 2",
        "secondary_family": "four per-tolerance comparisons; Holm over 4",
        "primary_results": primary.to_dict(orient="records"),
        "panel_lineage": panel_lineage,
        "source_workbook_materialized_before_q3_filter": bool(
            zero_pair_metrics is None
        ),
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "q4_used_for_checkpoint_selection": False,
        "artifacts": {
            role: {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for role, path in paths.items()
        },
        "created_at_utc": _utc_now(),
    }
    summary["analysis_sha256"] = _payload_sha256(summary)
    summary_path = _write_json(root / "zero_noise_analysis_summary.json", summary)
    validation = {
        "status": "pass",
        "experiment_kind": EXPERIMENT_KIND,
        "analysis_sha256": summary["analysis_sha256"],
        "pair_matrix_rows": int(len(paired)),
        "pair_count_per_cell": EXPECTED_PAIR_COUNT,
        "session_count": EXPECTED_SESSION_COUNT,
        "primary_comparison_count": 2,
        "secondary_tolerance_comparison_count": 4,
        "bootstrap_iterations": DEFAULT_BOOTSTRAP_ITERATIONS,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "validated_at_utc": _utc_now(),
    }
    validation["validation_sha256"] = _payload_sha256(validation)
    _write_json(analysis_dir / "zero_noise_validation_summary.json", validation)
    return summary_path


__all__ = [
    "ZeroNoiseAnalysisError",
    "build_cell_summary",
    "build_paired_pair_metrics",
    "build_persistence_bootstrap",
    "build_zero_gaussian_bootstrap",
    "evaluate_zero_q3_pair_metrics",
    "load_gaussian_pair_metrics",
    "run_zero_noise_analysis",
    "validate_zero_pair_metrics",
]
