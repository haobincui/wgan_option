"""Capacity selection and final evaluation for the news-first vol experiment.

Capacity is selected exclusively on the common 5-minute Q3 validation panel.
The Q4 panel is opened only by :func:`run_final_q4_analysis`, after the frozen
selection file proves that all required gates passed.  Keeping those entry
points separate makes accidental test-set selection difficult and auditable.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
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
    build_model_comparison,
    build_text_ablation_comparisons,
    cme_session_cluster_bootstrap,
    compute_sample_metrics,
    holm_adjust,
)


Q3_START_UTC = "2023-07-01T00:00:00Z"
Q3_END_UTC = "2023-10-01T00:00:00Z"
Q4_START_UTC = "2023-10-01T00:00:00Z"
Q4_END_UTC = "2024-01-01T00:00:00Z"
PRIMARY_TOLERANCES = (5, 30)
ALL_TOLERANCES = (5, 10, 15, 30)
GATE_MIN_IMPROVEMENT = 0.005
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260819

# Exact counts from real 16x16 model instantiation.  The table is deliberately
# local to the analysis boundary so selection cannot depend on lexicographic
# profile names or on an unreported estimate of model size.
PROFILE_PARAMETER_COUNTS: dict[str, dict[str, int]] = {
    "micro": {"regression": 20_070, "wgan": 25_850},
    "tiny": {"regression": 46_528, "wgan": 59_227},
    "small": {"regression": 119_376, "wgan": 149_333},
    "medium": {"regression": 344_800, "wgan": 422_953},
    "large": {"regression": 676_528, "wgan": 821_117},
    "legacy": {"regression": 3_929_728, "wgan": 4_691_653},
}

SUPPORTED_STAGES = (
    "regression_screen",
    "regression_confirm",
    "wgan_screen",
    "wgan_confirm",
)


class CapacityAnalysisError(ValueError):
    """Raised when a capacity decision cannot be reproduced safely."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CapacityAnalysisError(f"Expected a JSON object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise CapacityAnalysisError(f"Expected a YAML mapping: {path}")
    return dict(payload)


def _coerce_positive(value: Any, *, label: str, path: Path) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if not math.isfinite(float(numeric)) or float(numeric) <= 0.0:
        raise CapacityAnalysisError(f"{label} must be finite and positive in {path}")
    return float(numeric)


def _first_metric(payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    metrics = payload.get("metrics")
    candidates: list[Mapping[str, Any]] = [payload]
    if isinstance(metrics, Mapping):
        candidates.insert(0, metrics)
    for source in candidates:
        for name in names:
            if name in source and source[name] not in (None, ""):
                return source[name]
    return None


def _load_jobs(experiment_root: Path) -> list[dict[str, Any]]:
    registry = _read_json(experiment_root / "registry" / "jobs.json")
    raw_jobs = registry.get("jobs")
    if not isinstance(raw_jobs, list):
        raise CapacityAnalysisError("registry/jobs.json must contain a jobs list")
    jobs: list[dict[str, Any]] = []
    for raw in raw_jobs:
        if not isinstance(raw, Mapping):
            raise CapacityAnalysisError("Every registry job must be an object")
        job = dict(raw)
        job_id = str(job.get("job_id", "")).strip()
        if not job_id:
            raise CapacityAnalysisError("Every capacity job requires job_id")
        status_path = experiment_root / "registry" / "jobs" / f"{job_id}.status.json"
        status = _read_json(status_path) if status_path.is_file() else {}
        job.update(status)
        jobs.append(job)
    return jobs


def _job_stage(job: Mapping[str, Any]) -> str:
    return str(job.get("capacity_stage", job.get("stage", ""))).strip().lower()


def _job_model(job: Mapping[str, Any]) -> str:
    return str(job.get("model_family", job.get("model", ""))).strip().lower()


def _job_mode(job: Mapping[str, Any]) -> str:
    return str(job.get("text_ablation_mode", "")).strip().lower()


def _job_profile(job: Mapping[str, Any]) -> str:
    return str(job.get("capacity_profile", "")).strip().lower()


def _job_tolerance(job: Mapping[str, Any]) -> int:
    return int(job.get("tolerance_minutes", -1))


def _completed(job: Mapping[str, Any]) -> bool:
    return str(job.get("status", "")).strip().lower() == "completed"


def _run_dir(job: Mapping[str, Any]) -> Path:
    value = str(job.get("run_dir", "")).strip()
    if not value:
        raise CapacityAnalysisError(f"Completed job lacks run_dir: {job.get('job_id')}")
    path = Path(value).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _training_config(job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    value = str(job.get("training_config_path", "")).strip()
    if not value:
        raise CapacityAnalysisError(
            f"Job lacks training_config_path: {job.get('job_id')}"
        )
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, _load_yaml(path)


def _learned_metadata(job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _run_dir(job) / "metrics" / "best_learned_checkpoint.json"
    payload = _read_json(path)
    epoch = pd.to_numeric(payload.get("best_epoch"), errors="coerce")
    if not math.isfinite(float(epoch)) or int(epoch) < 1:
        raise CapacityAnalysisError(
            f"best_learned_checkpoint must select epoch >= 1: {path}"
        )
    return path, payload


def _baseline_inclusive_metadata(
    job: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = _run_dir(job) / "metrics" / "best_checkpoint.json"
    payload = _read_json(path)
    epoch = pd.to_numeric(payload.get("best_epoch"), errors="coerce")
    if not math.isfinite(float(epoch)) or int(epoch) < 0:
        raise CapacityAnalysisError(f"best_checkpoint must select epoch >= 0: {path}")
    if str(payload.get("selection_scope", "")).strip() != "baseline_inclusive":
        raise CapacityAnalysisError(
            f"best_checkpoint must be baseline-inclusive: {path}"
        )
    return path, payload


def _best_learned_epoch_metrics(
    job: Mapping[str, Any],
    *,
    learned_epoch: int,
    learned_metadata_path: Path,
    learned_metadata: Mapping[str, Any],
) -> tuple[Path, float, float]:
    path = _run_dir(job) / "metrics" / "training_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise CapacityAnalysisError(f"Expected a JSON list: {path}")
    matching = [
        row
        for row in payload
        if isinstance(row, Mapping)
        and pd.to_numeric(row.get("epoch"), errors="coerce") == learned_epoch
    ]
    if len(matching) != 1:
        raise CapacityAnalysisError(
            f"Expected one training-metrics row for learned epoch {learned_epoch}: {path}"
        )
    row = matching[0]
    train_recon = _coerce_positive(
        _first_metric(row, ("train_recon", "g_recon")),
        label="best-learned train reconstruction",
        path=path,
    )
    val_recon = _coerce_positive(
        _first_metric(row, ("val_recon",)),
        label="best-learned validation reconstruction",
        path=path,
    )
    metadata_val_recon = _coerce_positive(
        _first_metric(learned_metadata, ("val_recon",)),
        label="best-learned metadata validation reconstruction",
        path=learned_metadata_path,
    )
    tolerance = max(1.0e-12, abs(metadata_val_recon) * 1.0e-9)
    if not math.isclose(
        val_recon,
        metadata_val_recon,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise CapacityAnalysisError(
            f"Training metrics and best-learned metadata disagree on val_recon: {path}"
        )
    return path, train_recon, val_recon


def _resource_diagnostics(experiment_root: Path) -> dict[str, dict[str, Any]]:
    path = experiment_root / "resource_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    if "job_id" not in frame.columns:
        raise CapacityAnalysisError(f"resource_summary lacks job_id: {path}")
    if frame["job_id"].astype(str).duplicated().any():
        raise CapacityAnalysisError(
            f"resource_summary has duplicate job_id rows: {path}"
        )
    return {str(row["job_id"]): dict(row) for row in frame.to_dict(orient="records")}


def _optional_finite(value: Any) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if math.isfinite(float(numeric)) else float("nan")


def _learned_checkpoint(job: Mapping[str, Any], payload: Mapping[str, Any]) -> Path:
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, Mapping):
        for key in ("model", "generator", "checkpoint"):
            value = str(artifacts.get(key, "")).strip()
            if value:
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = _run_dir(job) / candidate
                if candidate.is_file():
                    return candidate.resolve()
    run_dir = _run_dir(job)
    candidates = (
        run_dir / "checkpoints" / "vol_regressor_best_learned.pt",
        run_dir / "checkpoints" / "generator_best_learned.pt",
    )
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise FileNotFoundError(
            f"Cannot resolve best-learned checkpoint for {job.get('job_id')}"
        )
    return found.resolve()


def collect_capacity_comparisons(
    experiment_root: str | Path,
    *,
    jobs: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Collect Q3 aggregate learned-checkpoint metrics from completed jobs."""

    root = Path(experiment_root).resolve()
    resources = _resource_diagnostics(root)
    rows: list[dict[str, Any]] = []
    for job in list(jobs) if jobs is not None else _load_jobs(root):
        profile = _job_profile(job)
        model = _job_model(job)
        if (
            not profile
            or profile not in PROFILE_PARAMETER_COUNTS
            or model
            not in {
                "regression",
                "wgan",
            }
        ):
            continue
        if not _completed(job):
            continue
        metadata_path, payload = _learned_metadata(job)
        baseline_path, baseline_payload = _baseline_inclusive_metadata(job)
        model_mae = _coerce_positive(
            _first_metric(
                payload,
                (
                    "masked_pair_balanced_mae",
                    "best_learned_masked_pair_balanced_mae",
                    "val_recon",
                ),
            ),
            label="best-learned masked pair-balanced MAE",
            path=metadata_path,
        )
        persistence_mae = _coerce_positive(
            _first_metric(
                payload,
                (
                    "persistence_masked_pair_balanced_mae",
                    "masked_pair_balanced_persistence_mae",
                    "val_current_recon",
                ),
            ),
            label="persistence masked pair-balanced MAE",
            path=metadata_path,
        )
        ratio = model_mae / persistence_mae
        profile_sha = str(job.get("capacity_profile_sha256", "")).strip()
        config_path, config = _training_config(job)
        config_profile = (
            str(config.get("news_first_capacity_profile", profile)).strip().lower()
        )
        config_sha = str(config.get("news_first_capacity_profile_sha256", "")).strip()
        metadata_profile = str(payload.get("capacity_profile", profile)).strip().lower()
        metadata_sha = str(payload.get("capacity_profile_sha256", "")).strip()
        if config_profile != profile:
            raise CapacityAnalysisError(
                f"Registry/config profile mismatch for {job.get('job_id')}"
            )
        if profile_sha and config_sha and profile_sha != config_sha:
            raise CapacityAnalysisError(
                f"Registry/config profile hash mismatch for {job.get('job_id')}"
            )
        if metadata_profile != profile:
            raise CapacityAnalysisError(
                f"Registry/learned-metadata profile mismatch for {job.get('job_id')}"
            )
        declared_hashes = {
            value for value in (profile_sha, config_sha, metadata_sha) if value
        }
        if len(declared_hashes) > 1:
            raise CapacityAnalysisError(
                f"Capacity profile hashes disagree for {job.get('job_id')}"
            )
        if str(config.get("support_mask_mode", "")).strip().lower() != "raw_joint":
            raise CapacityAnalysisError(
                f"Capacity selection requires support_mask_mode=raw_joint: {config_path}"
            )
        if str(config.get("news_first_train_end_utc", Q3_START_UTC)) != Q3_START_UTC:
            raise CapacityAnalysisError(
                f"Capacity selection train/Q3 boundary drift: {config_path}"
            )
        if str(config.get("news_first_validation_end_utc", Q3_END_UTC)) != Q3_END_UTC:
            raise CapacityAnalysisError(
                f"Capacity selection Q3/Q4 boundary drift: {config_path}"
            )
        if (
            str(payload.get("selection_scope", "trained_epochs_only"))
            != "trained_epochs_only"
        ):
            raise CapacityAnalysisError(
                f"Capacity selection requires trained-epochs-only metadata: {metadata_path}"
            )
        learned_epoch = int(payload["best_epoch"])
        declared_learned_epoch = int(
            payload.get("best_learned_epoch_ge_1", learned_epoch)
        )
        if declared_learned_epoch != learned_epoch:
            raise CapacityAnalysisError(
                f"Learned checkpoint epoch fields disagree: {metadata_path}"
            )
        training_metrics_path, train_recon, val_recon = _best_learned_epoch_metrics(
            job,
            learned_epoch=learned_epoch,
            learned_metadata_path=metadata_path,
            learned_metadata=payload,
        )
        job_id = str(job["job_id"])
        if job_id not in resources:
            raise CapacityAnalysisError(
                f"resource_summary lacks completed capacity job: {job_id}"
            )
        resource = resources[job_id]
        parameter_count = PROFILE_PARAMETER_COUNTS[profile][model]
        baseline_epoch = int(baseline_payload["best_epoch"])
        rows.append(
            {
                "job_id": job_id,
                "capacity_stage": _job_stage(job),
                "model_family": model,
                "capacity_profile": profile,
                "capacity_profile_sha256": profile_sha or config_sha or metadata_sha,
                "parameter_count": parameter_count,
                "parameter_log10": math.log10(parameter_count),
                "text_ablation_mode": _job_mode(job),
                "tolerance_minutes": _job_tolerance(job),
                "best_learned_epoch": learned_epoch,
                "best_learned_train_recon": train_recon,
                "best_learned_val_recon": val_recon,
                "best_learned_train_val_gap": val_recon - train_recon,
                "baseline_inclusive_best_epoch": baseline_epoch,
                "baseline_inclusive_epoch0_selected": baseline_epoch == 0,
                "model_mae": model_mae,
                "persistence_mae": persistence_mae,
                "mae_ratio": ratio,
                "log_mae_ratio": math.log(ratio),
                "improvement_fraction": 1.0 - ratio,
                "runtime_minutes": _optional_finite(resource.get("runtime_minutes")),
                "peak_memory_mib": _optional_finite(resource.get("peak_memory_mib")),
                "metadata_path": str(metadata_path),
                "metadata_sha256": _sha256(metadata_path),
                "baseline_metadata_path": str(baseline_path),
                "baseline_metadata_sha256": _sha256(baseline_path),
                "training_metrics_path": str(training_metrics_path),
                "training_metrics_sha256": _sha256(training_metrics_path),
                "training_config_path": str(config_path),
                "run_dir": str(_run_dir(job)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    duplicate_keys = [
        "model_family",
        "capacity_profile",
        "text_ablation_mode",
        "tolerance_minutes",
    ]
    if frame.duplicated(duplicate_keys).any():
        bad = frame.loc[frame.duplicated(duplicate_keys, keep=False), duplicate_keys]
        raise CapacityAnalysisError(
            f"Duplicate completed capacity jobs: {bad.to_dict(orient='records')[:5]}"
        )
    return frame.sort_values(duplicate_keys, kind="stable").reset_index(drop=True)


def summarize_profile_scores(
    comparisons: pd.DataFrame,
    *,
    model_family: str,
    text_ablation_mode: str,
    tolerances: Sequence[int],
    profiles: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute the frozen equal-tolerance mean log MAE ratio and gate."""

    expected_tolerances = tuple(int(value) for value in tolerances)
    selected = comparisons[
        comparisons["model_family"].astype(str).eq(str(model_family))
        & comparisons["text_ablation_mode"].astype(str).eq(str(text_ablation_mode))
        & comparisons["tolerance_minutes"].astype(int).isin(expected_tolerances)
    ].copy()
    if profiles is not None:
        selected = selected[
            selected["capacity_profile"].astype(str).isin([str(v) for v in profiles])
        ]
    rows: list[dict[str, Any]] = []
    for profile, group in selected.groupby("capacity_profile", sort=False):
        observed = tuple(sorted(group["tolerance_minutes"].astype(int).tolist()))
        if observed != tuple(sorted(expected_tolerances)):
            raise CapacityAnalysisError(
                f"Incomplete tolerances for {model_family}/{profile}/{text_ablation_mode}: "
                f"{observed} != {tuple(sorted(expected_tolerances))}"
            )
        score = float(group["log_mae_ratio"].mean())
        geometric_ratio = math.exp(score)
        all_not_worse = bool((group["mae_ratio"] <= 1.0 + 1.0e-12).all())
        gate_average = bool(1.0 - geometric_ratio >= GATE_MIN_IMPROVEMENT - 1.0e-12)
        rows.append(
            {
                "model_family": str(model_family),
                "capacity_profile": str(profile),
                "parameter_count": int(group["parameter_count"].iloc[0]),
                "selection_text_ablation_mode": str(text_ablation_mode),
                "selection_tolerances": ",".join(str(v) for v in expected_tolerances),
                "mean_log_mae_ratio": score,
                "geometric_mae_ratio": geometric_ratio,
                "mean_improvement_fraction": 1.0 - geometric_ratio,
                "all_tolerances_not_worse": all_not_worse,
                "average_improvement_gate_passed": gate_average,
                "profile_gate_passed": bool(all_not_worse and gate_average),
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        raise CapacityAnalysisError(
            f"No completed {model_family}/{text_ablation_mode} capacity rows"
        )
    return output.sort_values(
        ["mean_log_mae_ratio", "parameter_count", "capacity_profile"],
        kind="stable",
    ).reset_index(drop=True)


def _selection_document(root: Path) -> dict[str, Any]:
    path = root / "capacity_selection.json"
    if not path.is_file():
        return {
            "schema_version": 1,
            "selection_data_scope": "Q3 common_validation_05m only",
            "q3_start_utc": Q3_START_UTC,
            "q3_end_utc_exclusive": Q3_END_UTC,
            "q4_used_for_selection": False,
            "stages": {},
        }
    payload = _read_json(path)
    if bool(payload.get("q4_used_for_selection", False)):
        raise CapacityAnalysisError("Selection document claims Q4 was used")
    payload.setdefault("stages", {})
    return payload


def _prior_stage(root: Path, stage: str) -> dict[str, Any]:
    document = _selection_document(root)
    value = document.get("stages", {}).get(stage)
    if not isinstance(value, Mapping):
        raise CapacityAnalysisError(f"Required prior selection is missing: {stage}")
    return dict(value)


def _stage_rows(
    comparisons: pd.DataFrame,
    *,
    model: str,
    profiles: Sequence[str],
    modes: Sequence[str],
    tolerances: Sequence[int],
) -> pd.DataFrame:
    rows = comparisons[
        comparisons["model_family"].eq(model)
        & comparisons["capacity_profile"].isin(list(profiles))
        & comparisons["text_ablation_mode"].isin(list(modes))
        & comparisons["tolerance_minutes"].isin(list(tolerances))
    ].copy()
    expected = {
        (profile, mode, int(tolerance))
        for profile in profiles
        for mode in modes
        for tolerance in tolerances
    }
    observed = set(
        rows[
            ["capacity_profile", "text_ablation_mode", "tolerance_minutes"]
        ].itertuples(index=False, name=None)
    )
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise CapacityAnalysisError(
            f"Capacity stage matrix incomplete; missing={missing[:8]}, extra={extra[:8]}"
        )
    return rows


def _run_spec(job: Mapping[str, Any]) -> RunSpec:
    metadata_path, metadata = _learned_metadata(job)
    config_path, config = _training_config(job)
    return RunSpec(
        run_id=str(job["job_id"]),
        run_dir=_run_dir(job),
        model=_job_model(job),
        tolerance_minutes=_job_tolerance(job),
        seed=int(config.get("seed", 42)),
        checkpoint_path=_learned_checkpoint(job, metadata),
        text_ablation_mode=_job_mode(job),
        support_mask_mode=str(config.get("support_mask_mode", "none")),
        manifest_path=config_path,
        metadata={
            **config,
            "capacity_profile": _job_profile(job),
            "capacity_profile_sha256": str(job.get("capacity_profile_sha256", "")),
            "best_learned_metadata_path": str(metadata_path),
        },
    )


def _time_filtered_panel(
    workbook: Path,
    *,
    sheet_name: str,
    start_utc: str,
    end_utc: str,
    panel_name: str,
    support_mask_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Reading one xlsx sheet necessarily decodes its rows, but no row outside
    # the declared interval is passed to metric or inference code.
    raw = pd.read_excel(workbook, sheet_name=sheet_name)
    if "effective_origin_utc" not in raw.columns:
        raise CapacityAnalysisError(f"Panel lacks effective_origin_utc: {workbook}")
    timestamps = pd.to_datetime(raw["effective_origin_utc"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise CapacityAnalysisError(f"Unparseable panel timestamp in {workbook}")
    start = pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc)
    frame = raw.loc[(timestamps >= start) & (timestamps < end)].copy()
    if frame.empty:
        raise CapacityAnalysisError(f"No rows in {panel_name} interval")
    panel, lineage, _ = _load_panel_source(
        frame,
        sheet_name=sheet_name,
        panel_name=panel_name,
        freeze_test_window=False,
        enforce_expected_counts=False,
        support_mask_mode=support_mask_mode,
    )
    selected_times = pd.to_datetime(panel["effective_origin_utc"], utc=True)
    if bool((selected_times < start).any() or (selected_times >= end).any()):
        raise CapacityAnalysisError(f"Time guard failed for {panel_name}")
    lineage.update(
        {
            "path": str(workbook),
            "sha256": _sha256(workbook),
            "interval_start_utc": start_utc,
            "interval_end_utc_exclusive": end_utc,
            "selected_timestamp_min": selected_times.min().isoformat(),
            "selected_timestamp_max": selected_times.max().isoformat(),
        }
    )
    return panel, lineage


def _panel_for_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    q4: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths: set[tuple[str, str, str]] = set()
    for job in jobs:
        _, config = _training_config(job)
        paths.add(
            (
                str(config.get("news_first_common_eval_data_path", "")),
                str(config.get("sheet_name", "gan_input_ready")),
                str(config.get("support_mask_mode", "none")),
            )
        )
    if len(paths) != 1:
        raise CapacityAnalysisError(
            f"Compared jobs disagree on common evaluation panel: {sorted(paths)}"
        )
    raw_path, sheet, support_mode = next(iter(paths))
    workbook = Path(raw_path).resolve()
    if not workbook.is_file():
        raise FileNotFoundError(workbook)
    return _time_filtered_panel(
        workbook,
        sheet_name=sheet,
        start_utc=Q4_START_UTC if q4 else Q3_START_UTC,
        end_utc=Q4_END_UTC if q4 else Q3_END_UTC,
        panel_name="common_test_core_05m" if q4 else "common_validation_05m",
        support_mask_mode=support_mode,
    )


def evaluate_q3_pair_metrics(
    jobs: Sequence[Mapping[str, Any]],
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate best-learned checkpoints without exposing a Q4 row."""

    if not jobs:
        raise CapacityAnalysisError("No jobs supplied for Q3 pair evaluation")
    panel, lineage = _panel_for_jobs(jobs, q4=False)
    production = evaluator or TrainedRunEvaluator(mc_samples=16)
    parts: list[pd.DataFrame] = []
    for job in jobs:
        spec = _run_spec(job)
        # Selection only uses real_text or current_only.  For these modes the
        # old evaluator's namespace choice does not alter embeddings; passing
        # core also retains its stable 5m-noise contract.  Shuffled text is
        # deliberately excluded from every capacity-selection score.
        if spec.text_ablation_mode not in {"real_text", "current_only"}:
            raise CapacityAnalysisError(
                "Q3 capacity selection forbids text_shuffle predictions"
            )
        predictions = production(spec, "core", panel.copy())
        samples, exclusions, _ = compute_sample_metrics(
            spec,
            "common_validation_05m",
            panel,
            predictions,
            evaluate_embedded_atm_skew=False,
        )
        if not exclusions.empty:
            raise CapacityAnalysisError(
                f"Q3 evaluation exclusions for {spec.run_id}: "
                f"{exclusions['exclusion_code'].value_counts().to_dict()}"
            )
        pairs = aggregate_pair_metrics(samples)
        pairs = pairs[
            pairs["stratum_type"].astype(str).eq("overall")
            & pairs["stratum_value"].astype(str).eq("all")
        ].copy()
        metadata_path, learned = _learned_metadata(job)
        expected_model = _coerce_positive(
            _first_metric(learned, ("masked_pair_balanced_mae", "val_recon")),
            label="learned validation MAE",
            path=metadata_path,
        )
        expected_persistence = _coerce_positive(
            _first_metric(
                learned,
                ("persistence_masked_pair_balanced_mae", "val_current_recon"),
            ),
            label="validation persistence MAE",
            path=metadata_path,
        )
        actual_model = float(pairs["model_mae"].mean())
        actual_persistence = float(pairs["persistence_mae"].mean())
        for label, actual, expected in (
            ("model", actual_model, expected_model),
            ("persistence", actual_persistence, expected_persistence),
        ):
            # GPU convolution kernels may accumulate in a different order when
            # inference and validation use different batch sizes.  This bound
            # remains far below the 0.5% capacity gate while tolerating that
            # harmless float32 variation.
            tolerance = max(1.0e-7, abs(expected) * 1.0e-4)
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
                raise CapacityAnalysisError(
                    f"Q3 pair-balanced {label} MAE disagrees with learned metadata for "
                    f"{spec.run_id}: {actual} != {expected}"
                )
        pairs.insert(0, "capacity_profile", _job_profile(job))
        pairs.insert(
            1,
            "capacity_profile_sha256",
            str(job.get("capacity_profile_sha256", "")),
        )
        parts.append(pairs)
    output = pd.concat(parts, ignore_index=True)
    max_time = pd.to_datetime(panel["effective_origin_utc"], utc=True).max()
    if max_time >= pd.Timestamp(Q3_END_UTC):
        raise CapacityAnalysisError("Q4 row reached Q3 capacity evaluation")
    lineage["q4_rows_passed_to_evaluator"] = 0
    lineage["selection_scope_verified"] = True
    return output, lineage


def one_standard_error_selection(
    pair_metrics: pd.DataFrame,
    *,
    profiles: Sequence[str],
    tolerances: Sequence[int],
    model_family: str,
    text_ablation_mode: str,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Apply a paired CME-session one-standard-error capacity rule."""

    if int(iterations) < 2:
        raise CapacityAnalysisError("one-SE bootstrap requires at least two draws")
    selected = pair_metrics[
        pair_metrics["capacity_profile"].isin(list(profiles))
        & pair_metrics["model"].astype(str).eq(model_family)
        & pair_metrics["text_ablation_mode"].astype(str).eq(text_ablation_mode)
        & pair_metrics["tolerance_minutes"].isin(list(tolerances))
    ].copy()
    required = {
        "capacity_profile",
        "tolerance_minutes",
        "pair_id",
        "session_id",
        "model_mae",
        "persistence_mae",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise CapacityAnalysisError(f"one-SE pair metrics missing columns: {missing}")
    key_sets: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for (profile, tolerance), group in selected.groupby(
        ["capacity_profile", "tolerance_minutes"], sort=False
    ):
        key_sets[(str(profile), int(tolerance))] = set(
            group[["pair_id", "session_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
    expected_keys = {
        (str(profile), int(tolerance))
        for profile in profiles
        for tolerance in tolerances
    }
    if (
        set(key_sets) != expected_keys
        or len({frozenset(v) for v in key_sets.values()}) != 1
    ):
        raise CapacityAnalysisError(
            "one-SE comparison requires identical pair/session coverage for every profile/tolerance"
        )
    sessions = sorted(selected["session_id"].astype(str).unique().tolist())
    if len(sessions) < 2:
        raise CapacityAnalysisError(
            "one-SE comparison requires at least two CME sessions"
        )

    rng = np.random.default_rng(int(seed))
    sampled_indexes = rng.integers(
        0, len(sessions), size=(int(iterations), len(sessions))
    )
    point: dict[str, float] = {}
    draws: dict[str, np.ndarray] = {}
    for raw_profile in profiles:
        profile = str(raw_profile)
        point_terms: list[float] = []
        draw_terms: list[np.ndarray] = []
        profile_rows = selected[selected["capacity_profile"].astype(str).eq(profile)]
        for tolerance in tolerances:
            group = profile_rows[
                profile_rows["tolerance_minutes"].astype(int).eq(int(tolerance))
            ].copy()
            grouped = group.groupby("session_id", sort=True).agg(
                model_sum=("model_mae", "sum"),
                persistence_sum=("persistence_mae", "sum"),
                pair_count=("pair_id", "size"),
            )
            grouped = grouped.reindex(sessions)
            if grouped.isna().any().any():
                raise CapacityAnalysisError(
                    f"Missing session in one-SE profile/tolerance: {profile}/{tolerance}"
                )
            model_sums = grouped["model_sum"].to_numpy(dtype=float)
            persistence_sums = grouped["persistence_sum"].to_numpy(dtype=float)
            counts = grouped["pair_count"].to_numpy(dtype=float)
            model_mean = float(model_sums.sum() / counts.sum())
            persistence_mean = float(persistence_sums.sum() / counts.sum())
            if not (model_mean > 0 and persistence_mean > 0):
                raise CapacityAnalysisError("one-SE inputs must contain positive MAEs")
            point_terms.append(math.log(model_mean / persistence_mean))
            draw_model = model_sums[sampled_indexes].sum(axis=1)
            draw_persistence = persistence_sums[sampled_indexes].sum(axis=1)
            draw_counts = counts[sampled_indexes].sum(axis=1)
            ratios = (draw_model / draw_counts) / (draw_persistence / draw_counts)
            if not np.isfinite(ratios).all() or bool((ratios <= 0).any()):
                raise CapacityAnalysisError("Invalid one-SE bootstrap MAE ratio")
            draw_terms.append(np.log(ratios))
        point[profile] = float(np.mean(point_terms))
        draws[profile] = np.mean(np.stack(draw_terms, axis=0), axis=0)
    standard_errors = {
        profile: float(values.std(ddof=1)) for profile, values in draws.items()
    }
    best = min(
        [str(profile) for profile in profiles],
        key=lambda profile: (
            point[profile],
            PROFILE_PARAMETER_COUNTS[profile][model_family],
            profile,
        ),
    )
    threshold = point[best] + standard_errors[best]
    eligible = [
        str(profile)
        for profile in profiles
        if point[str(profile)] <= threshold + 1.0e-15
    ]
    winner = min(
        eligible,
        key=lambda profile: (
            PROFILE_PARAMETER_COUNTS[profile][model_family],
            point[profile],
            profile,
        ),
    )
    return {
        "method": "paired_CME_session_cluster_one_standard_error",
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
        "best_score_profile": best,
        "best_score": point[best],
        "best_score_standard_error": standard_errors[best],
        "one_se_threshold": threshold,
        "one_se_eligible_profiles": sorted(
            eligible,
            key=lambda profile: PROFILE_PARAMETER_COUNTS[profile][model_family],
        ),
        "winner_profile": winner,
        "profile_scores": point,
        "profile_standard_errors": standard_errors,
        "session_count": len(sessions),
    }


def _selection_jobs(
    all_jobs: Sequence[Mapping[str, Any]],
    *,
    model: str,
    profiles: Sequence[str],
    mode: str,
    tolerances: Sequence[int],
) -> list[dict[str, Any]]:
    jobs = [
        dict(job)
        for job in all_jobs
        if _completed(job)
        and _job_model(job) == model
        and _job_profile(job) in set(profiles)
        and _job_mode(job) == mode
        and _job_tolerance(job) in set(int(value) for value in tolerances)
    ]
    expected = {
        (str(profile), int(tolerance))
        for profile in profiles
        for tolerance in tolerances
    }
    observed = {(_job_profile(job), _job_tolerance(job)) for job in jobs}
    if observed != expected:
        raise CapacityAnalysisError(
            f"Selection checkpoint matrix incomplete: missing={sorted(expected - observed)}"
        )
    return jobs


def _decorate_comparisons(
    comparisons: pd.DataFrame,
    summaries: pd.DataFrame,
    *,
    stage: str,
    top_profiles: Sequence[str],
    winner: str,
    one_se: Mapping[str, Any] | None,
) -> pd.DataFrame:
    summary_map = summaries.set_index("capacity_profile").to_dict(orient="index")
    output = comparisons.copy()
    output.insert(0, "selection_stage", stage)
    for field in (
        "mean_log_mae_ratio",
        "geometric_mae_ratio",
        "mean_improvement_fraction",
        "all_tolerances_not_worse",
        "average_improvement_gate_passed",
        "profile_gate_passed",
    ):
        output[field] = output["capacity_profile"].map(
            {profile: values[field] for profile, values in summary_map.items()}
        )
    rank = {
        profile: index + 1
        for index, profile in enumerate(summaries["capacity_profile"].tolist())
    }
    output["score_rank"] = output["capacity_profile"].map(rank)
    output["selected_top2"] = output["capacity_profile"].isin(list(top_profiles))
    output["one_se_eligible"] = output["capacity_profile"].isin(
        list((one_se or {}).get("one_se_eligible_profiles", []))
    )
    output["winner_profile"] = winner
    return output


def evaluate_capacity_stage(
    experiment_root: str | Path,
    stage: str,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
    q3_pair_metrics: pd.DataFrame | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate one append-only capacity stage and persist its Q3 decision.

    The first two positional arguments are the stable orchestrator interface;
    keyword injections exist only to make fixture tests fast and deterministic.
    """

    normalized_stage = str(stage).strip().lower()
    if normalized_stage not in SUPPORTED_STAGES:
        raise CapacityAnalysisError(
            f"stage must be one of {list(SUPPORTED_STAGES)}, got {stage!r}"
        )
    root = Path(experiment_root).resolve()
    all_jobs = _load_jobs(root)
    comparisons = collect_capacity_comparisons(root, jobs=all_jobs)

    one_se: dict[str, Any] | None = None
    lineage: dict[str, Any] = {
        "selection_panel": "common_validation_05m",
        "q3_start_utc": Q3_START_UTC,
        "q3_end_utc_exclusive": Q3_END_UTC,
        "q4_rows_passed_to_evaluator": 0,
    }
    if normalized_stage == "regression_screen":
        model = "regression"
        profiles = tuple(PROFILE_PARAMETER_COUNTS)
        modes = ("current_only", "real_text")
        tolerances = PRIMARY_TOLERANCES
        stage_rows = _stage_rows(
            comparisons,
            model=model,
            profiles=profiles,
            modes=modes,
            tolerances=tolerances,
        )
        summaries = summarize_profile_scores(
            stage_rows,
            model_family=model,
            text_ablation_mode="real_text",
            tolerances=tolerances,
            profiles=profiles,
        )
        top_profiles = summaries["capacity_profile"].head(2).tolist()
        winner = top_profiles[0]
    elif normalized_stage == "regression_confirm":
        model = "regression"
        prior = _prior_stage(root, "regression_screen")
        profiles = tuple(str(value) for value in prior.get("selected_profiles", []))
        if len(profiles) != 2:
            raise CapacityAnalysisError(
                "regression_screen must freeze exactly two profiles"
            )
        modes = ("current_only", "real_text", "text_shuffle")
        tolerances = ALL_TOLERANCES
        stage_rows = _stage_rows(
            comparisons,
            model=model,
            profiles=profiles,
            modes=modes,
            tolerances=tolerances,
        )
        summaries = summarize_profile_scores(
            stage_rows,
            model_family=model,
            text_ablation_mode="real_text",
            tolerances=tolerances,
            profiles=profiles,
        )
        top_profiles = list(profiles)
        selection_jobs = _selection_jobs(
            all_jobs,
            model=model,
            profiles=profiles,
            mode="real_text",
            tolerances=tolerances,
        )
        if q3_pair_metrics is None:
            q3_pair_metrics, lineage = evaluate_q3_pair_metrics(
                selection_jobs, evaluator=evaluator
            )
        one_se = one_standard_error_selection(
            q3_pair_metrics,
            profiles=profiles,
            tolerances=tolerances,
            model_family=model,
            text_ablation_mode="real_text",
            iterations=int(bootstrap_iterations),
            seed=int(bootstrap_seed),
        )
        winner = str(one_se["winner_profile"])
    elif normalized_stage == "wgan_screen":
        model = "wgan"
        prior = _prior_stage(root, "regression_screen")
        profile_values = [str(value) for value in prior.get("selected_profiles", [])]
        profiles = tuple(dict.fromkeys(profile_values + ["legacy"]))
        if not 2 <= len(profiles) <= 3:
            raise CapacityAnalysisError(
                "wgan_screen requires regression top2 plus legacy"
            )
        modes = ("current_only",)
        tolerances = PRIMARY_TOLERANCES
        stage_rows = _stage_rows(
            comparisons,
            model=model,
            profiles=profiles,
            modes=modes,
            tolerances=tolerances,
        )
        summaries = summarize_profile_scores(
            stage_rows,
            model_family=model,
            text_ablation_mode="current_only",
            tolerances=tolerances,
            profiles=profiles,
        )
        top_profiles = summaries["capacity_profile"].head(2).tolist()
        selection_jobs = _selection_jobs(
            all_jobs,
            model=model,
            profiles=profiles,
            mode="current_only",
            tolerances=tolerances,
        )
        if q3_pair_metrics is None:
            q3_pair_metrics, lineage = evaluate_q3_pair_metrics(
                selection_jobs, evaluator=evaluator
            )
        one_se = one_standard_error_selection(
            q3_pair_metrics,
            profiles=profiles,
            tolerances=tolerances,
            model_family=model,
            text_ablation_mode="current_only",
            iterations=int(bootstrap_iterations),
            seed=int(bootstrap_seed),
        )
        winner = str(one_se["winner_profile"])
    else:
        model = "wgan"
        prior = _prior_stage(root, "wgan_screen")
        winner = str(prior.get("winner_profile", ""))
        if winner not in PROFILE_PARAMETER_COUNTS:
            raise CapacityAnalysisError("wgan_screen did not freeze a valid winner")
        profiles = (winner,)
        modes = ("current_only", "real_text", "text_shuffle")
        tolerances = ALL_TOLERANCES
        stage_rows = _stage_rows(
            comparisons,
            model=model,
            profiles=profiles,
            modes=modes,
            tolerances=tolerances,
        )
        summaries = summarize_profile_scores(
            stage_rows,
            model_family=model,
            text_ablation_mode="current_only",
            tolerances=tolerances,
            profiles=profiles,
        )
        top_profiles = [winner]

    winner_summary = summaries[summaries["capacity_profile"].astype(str).eq(winner)]
    if len(winner_summary) != 1:
        raise CapacityAnalysisError(f"Winner summary is not unique: {winner}")
    gate_passed = bool(winner_summary.iloc[0]["profile_gate_passed"])
    winner_profile_hashes = sorted(
        {
            str(value)
            for value in stage_rows.loc[
                stage_rows["capacity_profile"].astype(str).eq(winner),
                "capacity_profile_sha256",
            ].tolist()
            if str(value)
        }
    )
    if len(winner_profile_hashes) > 1:
        raise CapacityAnalysisError(
            f"Winner profile hash is not unique for {winner}: {winner_profile_hashes}"
        )
    ranked_top2 = summaries["capacity_profile"].head(2).astype(str).tolist()
    formal_winner = winner if gate_passed else ""
    formal_selected_profiles = list(top_profiles) if gate_passed else []
    formal_profile_hash = (
        winner_profile_hashes[0] if gate_passed and winner_profile_hashes else ""
    )
    one_se_payload = dict(one_se or {})
    if not gate_passed and "winner_profile" in one_se_payload:
        one_se_payload["ranked_candidate_profile"] = one_se_payload.pop(
            "winner_profile"
        )
    decorated = _decorate_comparisons(
        stage_rows,
        summaries,
        stage=normalized_stage,
        top_profiles=ranked_top2,
        winner=formal_winner,
        one_se=one_se,
    )
    decorated["ranked_top2"] = decorated["capacity_profile"].isin(ranked_top2)
    decorated["selected_top2"] = decorated["capacity_profile"].isin(
        formal_selected_profiles
    )
    decorated["ranked_leader_profile"] = winner

    result: dict[str, Any] = {
        "schema_version": 1,
        "capacity_stage": normalized_stage,
        "model_family": model,
        "selection_text_ablation_mode": (
            "current_only" if model == "wgan" else "real_text"
        ),
        "selection_tolerances": [int(value) for value in tolerances],
        "score_definition": (
            "equal-tolerance mean log(best_learned masked pair-balanced MAE / "
            "persistence masked pair-balanced MAE)"
        ),
        "gate_min_mean_improvement_fraction": GATE_MIN_IMPROVEMENT,
        "gate_requires_every_selection_tolerance_not_worse": True,
        "gate_passed": gate_passed,
        "selected_profiles": formal_selected_profiles,
        "winner_profile": formal_winner,
        "capacity_profile": formal_winner,
        "capacity_profile_sha256": formal_profile_hash,
        "winner_parameter_count": (
            PROFILE_PARAMETER_COUNTS[winner][model] if gate_passed else None
        ),
        "ranked_leader_profile": winner,
        "ranked_leader_capacity_profile_sha256": (
            winner_profile_hashes[0] if winner_profile_hashes else ""
        ),
        "ranked_leader_parameter_count": PROFILE_PARAMETER_COUNTS[winner][model],
        "ranked_top2_profiles": ranked_top2,
        "candidate_profiles": summaries.to_dict(orient="records"),
        "one_standard_error_selection": one_se_payload,
        "q3_lineage": lineage,
        "q4_used_for_selection": False,
    }
    result["selection_payload_sha256"] = _payload_sha256(result)

    document = _selection_document(root)
    stages = dict(document.get("stages", {}))
    stages[normalized_stage] = deepcopy(result)
    document["stages"] = stages
    document["updated_at_utc"] = _utc_now()
    _write_json(root / "capacity_selection.json", document)

    comparison_path = root / "capacity_comparisons.csv"
    if comparison_path.is_file():
        previous = pd.read_csv(comparison_path, low_memory=False)
        if "selection_stage" in previous.columns:
            previous = previous[
                ~previous["selection_stage"].astype(str).eq(normalized_stage)
            ]
        decorated = pd.concat([previous, decorated], ignore_index=True, sort=False)
    _write_csv(decorated, comparison_path)

    audit_row = {
        "capacity_stage": normalized_stage,
        "evaluated_at_utc": _utc_now(),
        "model_family": model,
        "candidate_profile_count": len(summaries),
        "selected_profiles": "|".join(formal_selected_profiles),
        "winner_profile": formal_winner,
        "capacity_profile": formal_winner,
        "capacity_profile_sha256": formal_profile_hash,
        "ranked_leader_profile": winner,
        "ranked_top2_profiles": "|".join(ranked_top2),
        "gate_passed": gate_passed,
        "selection_panel": "common_validation_05m",
        "q3_start_utc": Q3_START_UTC,
        "q3_end_utc_exclusive": Q3_END_UTC,
        "q4_rows_passed_to_evaluator": int(
            lineage.get("q4_rows_passed_to_evaluator", 0)
        ),
        "selection_payload_sha256": result["selection_payload_sha256"],
    }
    audit_path = root / "capacity_selection_audit.csv"
    audit = pd.read_csv(audit_path) if audit_path.is_file() else pd.DataFrame()
    if not audit.empty:
        audit = audit[~audit["capacity_stage"].astype(str).eq(normalized_stage)]
    audit = pd.concat([audit, pd.DataFrame([audit_row])], ignore_index=True)
    _write_csv(audit, audit_path)

    if q3_pair_metrics is not None and normalized_stage in {
        "regression_confirm",
        "wgan_screen",
    }:
        pair_path = root / "analysis" / f"q3_pair_metrics_{normalized_stage}.csv.gz"
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        q3_pair_metrics.to_csv(pair_path, index=False, compression="gzip")
    return result


def _final_jobs(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], str]:
    document = _selection_document(root)
    stages = document.get("stages", {})
    regression_screen = stages.get("regression_screen", {})
    regression = stages.get("regression_confirm", {})
    if not isinstance(regression_screen, Mapping) or not isinstance(
        regression, Mapping
    ):
        raise CapacityAnalysisError(
            "Final Q4 requires frozen regression_screen and regression_confirm"
        )
    if not bool(regression_screen.get("gate_passed")) or not bool(
        regression.get("gate_passed")
    ):
        raise CapacityAnalysisError(
            "Regression capacity gate failed; Q4 must remain unread"
        )
    winners = {"regression": str(regression.get("winner_profile", ""))}

    wgan_screen = stages.get("wgan_screen")
    wgan_confirm = stages.get("wgan_confirm")
    if not isinstance(wgan_screen, Mapping):
        wgan_status = "not_run"
    elif not bool(wgan_screen.get("gate_passed")):
        wgan_status = "gate_failed"
    elif not isinstance(wgan_confirm, Mapping):
        raise CapacityAnalysisError("WGAN screen passed but wgan_confirm is not frozen")
    elif not bool(wgan_confirm.get("gate_passed")):
        wgan_status = "gate_failed"
    else:
        wgan_status = "included"
        winners["wgan"] = str(wgan_confirm.get("winner_profile", ""))

    if any(value not in PROFILE_PARAMETER_COUNTS for value in winners.values()):
        raise CapacityAnalysisError(f"Invalid frozen final profile(s): {winners}")
    jobs = _load_jobs(root)
    selected = [
        job
        for job in jobs
        if _completed(job)
        and _job_model(job) in winners
        and _job_profile(job) == winners[_job_model(job)]
        and _job_mode(job) in {"current_only", "real_text", "text_shuffle"}
        and _job_tolerance(job) in ALL_TOLERANCES
    ]
    expected = {
        (model, winners[model], mode, tolerance)
        for model in winners
        for mode in ("current_only", "real_text", "text_shuffle")
        for tolerance in ALL_TOLERANCES
    }
    observed = {
        (_job_model(job), _job_profile(job), _job_mode(job), _job_tolerance(job))
        for job in selected
    }
    if observed != expected:
        raise CapacityAnalysisError(
            f"Final Q4 learned-checkpoint matrix incomplete: missing={sorted(expected - observed)}"
        )
    return [dict(job) for job in selected], winners, wgan_status


def run_final_q4_analysis(
    experiment_root: str | Path,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Path:
    """Evaluate frozen winners on Q4 and write paired session-bootstrap tables."""

    root = Path(experiment_root).resolve()
    jobs, winners, wgan_status = _final_jobs(root)
    panel, lineage = _panel_for_jobs(jobs, q4=True)
    production = evaluator or TrainedRunEvaluator(mc_samples=64)
    pair_parts: list[pd.DataFrame] = []
    for job in jobs:
        spec = _run_spec(job)
        predictions = production(spec, "core", panel.copy())
        samples, exclusions, _ = compute_sample_metrics(
            spec,
            "core",
            panel,
            predictions,
            evaluate_embedded_atm_skew=False,
        )
        if not exclusions.empty:
            raise CapacityAnalysisError(
                f"Final Q4 exclusions for {spec.run_id}: "
                f"{exclusions['exclusion_code'].value_counts().to_dict()}"
            )
        pairs = aggregate_pair_metrics(samples)
        pairs.insert(0, "capacity_profile", _job_profile(job))
        pairs.insert(
            1,
            "capacity_profile_sha256",
            str(job.get("capacity_profile_sha256", "")),
        )
        pair_parts.append(pairs)
    pair_metrics = pd.concat(pair_parts, ignore_index=True)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    pair_metrics.to_csv(
        analysis / "final_q4_pair_metrics.csv.gz", index=False, compression="gzip"
    )
    summary = build_model_comparison(pair_metrics)
    summary.insert(
        0,
        "capacity_profile",
        summary["model"].map(winners),
    )
    _write_csv(summary, analysis / "final_q4_model_comparison.csv")

    text_bootstrap = build_text_ablation_comparisons(
        pair_metrics,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )
    if not text_bootstrap.empty:
        text_bootstrap.insert(
            0, "capacity_profile", text_bootstrap["model"].map(winners)
        )
    _write_csv(text_bootstrap, analysis / "final_q4_text_ablation_bootstrap.csv")

    persistence_rows: list[dict[str, Any]] = []
    overall = pair_metrics[
        pair_metrics["stratum_type"].astype(str).eq("overall")
        & pair_metrics["stratum_value"].astype(str).eq("all")
    ]
    for keys, group in overall.groupby(
        ["model", "capacity_profile", "text_ablation_mode", "tolerance_minutes"],
        sort=True,
    ):
        model, profile, mode, tolerance = keys
        result = cme_session_cluster_bootstrap(
            group["mae_gap"],
            group["session_id"],
            iterations=int(bootstrap_iterations),
            seed=int(bootstrap_seed)
            + int(hashlib.sha256(str(keys).encode()).hexdigest()[:8], 16),
        )
        persistence_rows.append(
            {
                "model": model,
                "capacity_profile": profile,
                "text_ablation_mode": mode,
                "tolerance_minutes": int(tolerance),
                "comparison": "best_learned_minus_persistence_mae",
                **result,
            }
        )
    persistence = pd.DataFrame(persistence_rows)
    if not persistence.empty:
        persistence["p_holm"] = holm_adjust(persistence["p_two_sided"].tolist())
    _write_csv(persistence, analysis / "final_q4_persistence_bootstrap.csv")

    validation = {
        "schema_version": 1,
        "status": "pass",
        "regression_capacity_profile": winners["regression"],
        "wgan_capacity_profile": winners.get("wgan", ""),
        "wgan_status": wgan_status,
        "experiment_outcome": (
            "completed" if wgan_status == "included" else "completed_regression_only"
        ),
        "panel": "common_test_core_05m",
        "q4_lineage": lineage,
        "run_count": len(jobs),
        "pair_metric_rows": len(pair_metrics),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "holm_family_persistence_tests": len(persistence),
        "holm_family_text_ablation_tests": len(text_bootstrap),
        "selection_file_sha256": _sha256(root / "capacity_selection.json"),
        "completed_at_utc": _utc_now(),
    }
    output = _write_json(analysis / "final_q4_validation_summary.json", validation)
    return output


__all__ = [
    "ALL_TOLERANCES",
    "CapacityAnalysisError",
    "PROFILE_PARAMETER_COUNTS",
    "PRIMARY_TOLERANCES",
    "SUPPORTED_STAGES",
    "collect_capacity_comparisons",
    "evaluate_capacity_stage",
    "evaluate_q3_pair_metrics",
    "one_standard_error_selection",
    "run_final_q4_analysis",
    "summarize_profile_scores",
]
