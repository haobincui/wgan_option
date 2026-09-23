"""Q3-only analysis for the fixed-5e-7 capacity-by-seed experiment.

All model comparisons are made on the same unique market-pair panel.  Training
seed and CME session are kept as separate uncertainty levels: descriptive
statistics report the mean and sample SD across seeds, while inferential
statistics resample seeds and then paired CME-session clusters within each
sampled seed.  This module intentionally has no Q4 evaluation entry point.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from scripts.rq3.news_first_vol_capacity_analysis import PROFILE_PARAMETER_COUNTS
from scripts.rq3.news_first_vol_comparison_analysis import (
    RunSpec,
    TrainedRunEvaluator,
    _load_panel_source,
    aggregate_pair_metrics,
    compute_sample_metrics,
    holm_adjust,
)
from scripts.rq3.news_first_vol_fixed_lr_capacity_sweep import (
    EXPERIMENT_KIND,
    EXPERIMENT_STAGE,
    FIXED_LEARNING_RATE,
    FIXED_LR_PROFILE,
    FIXED_SCHEDULER_MIN_LR,
    FROZEN_PROFILES,
    FROZEN_SEEDS,
    FROZEN_TEXT_MODES,
    FROZEN_TOLERANCES,
    _capacity_profile_sha256,
    _capacity_seed_profile_sha256,
    _fixed_lr_profile_sha256,
)


Q3_START_UTC = "2023-07-01T00:00:00Z"
Q3_END_UTC = "2023-10-01T00:00:00Z"
SELECTION_PANEL = "common_validation_05m"
EXPECTED_PROFILES = tuple(FROZEN_PROFILES)
EXPECTED_SEEDS = tuple(FROZEN_SEEDS)
EXPECTED_TEXT_MODES = tuple(FROZEN_TEXT_MODES)
EXPECTED_TOLERANCES = tuple(FROZEN_TOLERANCES)
# Keep the capacity decision lane comparable with the preceding capacity
# experiment.  Current-only remains a fully paired diagnostic, but cannot
# determine the formal winner.
SELECTION_MODE = "real_text"
EXPECTED_PAIR_COUNT = 123
EXPECTED_SESSION_COUNT = 33
GATE_MIN_IMPROVEMENT = 0.005
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260820


class FixedLearningRateCapacityAnalysisError(ValueError):
    """Raised when fixed-LR capacity evidence violates its frozen contract."""


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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FixedLearningRateCapacityAnalysisError(f"Expected JSON object: {path}")
    return dict(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise FixedLearningRateCapacityAnalysisError(f"Expected YAML mapping: {path}")
    return dict(value)


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


def _finite(value: Any, *, label: str) -> float:
    result = float(pd.to_numeric(value, errors="coerce"))
    if not math.isfinite(result):
        raise FixedLearningRateCapacityAnalysisError(f"{label} must be finite")
    return result


def _positive(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if result <= 0.0:
        raise FixedLearningRateCapacityAnalysisError(f"{label} must be positive")
    return result


def _same_float(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-15)


def _first_metric(payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    nested = payload.get("metrics")
    sources: list[Mapping[str, Any]] = [payload]
    if isinstance(nested, Mapping):
        sources.insert(0, nested)
    for source in sources:
        for name in names:
            if name in source and source[name] not in (None, ""):
                return source[name]
    return None


def _profile(job: Mapping[str, Any]) -> str:
    return str(job.get("capacity_profile", "")).strip().lower()


def _seed(job: Mapping[str, Any]) -> int:
    value = _finite(job.get("seed"), label=f"seed for {job.get('job_id')}")
    if not float(value).is_integer():
        raise FixedLearningRateCapacityAnalysisError("Training seed must be an integer")
    return int(value)


def _mode(job: Mapping[str, Any]) -> str:
    return str(job.get("text_ablation_mode", "")).strip().lower()


def _tolerance(job: Mapping[str, Any]) -> int:
    return int(job.get("tolerance_minutes", -1))


def _run_dir(job: Mapping[str, Any]) -> Path:
    path = Path(str(job.get("run_dir", ""))).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _training_config(job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(str(job.get("training_config_path", ""))).resolve()
    return path, _read_yaml(path)


def _learned_metadata(job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _run_dir(job) / "metrics" / "best_learned_checkpoint.json"
    payload = _read_json(path)
    epoch = int(_finite(payload.get("best_epoch"), label="best learned epoch"))
    if epoch < 1 or int(payload.get("best_learned_epoch_ge_1", epoch)) != epoch:
        raise FixedLearningRateCapacityAnalysisError(
            f"best_learned must select epoch >=1: {path}"
        )
    if (
        str(payload.get("selection_scope", "trained_epochs_only"))
        != "trained_epochs_only"
    ):
        raise FixedLearningRateCapacityAnalysisError(
            f"best_learned selection scope drifted: {path}"
        )
    return path, payload


def _learned_checkpoint(job: Mapping[str, Any], payload: Mapping[str, Any]) -> Path:
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, Mapping):
        for key in ("model", "checkpoint"):
            raw = str(artifacts.get(key, "")).strip()
            if raw:
                path = Path(raw)
                if not path.is_absolute():
                    path = _run_dir(job) / path
                if path.is_file():
                    return path.resolve()
    candidates = (
        _run_dir(job) / "checkpoints" / "best_learned_epoch_ge_1.pt",
        _run_dir(job) / "checkpoints" / "vol_regressor_best_learned.pt",
    )
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise FileNotFoundError(candidates[0])
    return found.resolve()


def _load_jobs(root: Path) -> list[dict[str, Any]]:
    registry = _read_json(root / "registry" / "jobs.json")
    if str(registry.get("experiment_kind", "")) != EXPERIMENT_KIND:
        raise FixedLearningRateCapacityAnalysisError("Experiment kind mismatch")
    raw_jobs = registry.get("jobs")
    if not isinstance(raw_jobs, list):
        raise FixedLearningRateCapacityAnalysisError("Registry jobs must be a list")
    jobs: list[dict[str, Any]] = []
    for raw in raw_jobs:
        if not isinstance(raw, Mapping) or not str(raw.get("job_id", "")).strip():
            raise FixedLearningRateCapacityAnalysisError("Malformed registry job")
        job = dict(raw)
        status = root / "registry" / "jobs" / f"{job['job_id']}.status.json"
        if status.is_file():
            job.update(_read_json(status))
        jobs.append(job)
    return jobs


def validate_job_matrix(jobs: Sequence[Mapping[str, Any]]) -> None:
    """Require the exact completed 6x3x2x2 job matrix."""

    expected = {
        (profile, seed, mode, tolerance)
        for profile in EXPECTED_PROFILES
        for seed in EXPECTED_SEEDS
        for mode in EXPECTED_TEXT_MODES
        for tolerance in EXPECTED_TOLERANCES
    }
    observed: list[tuple[str, int, str, int]] = []
    for job in jobs:
        if str(job.get("status", "")).lower() != "completed":
            raise FixedLearningRateCapacityAnalysisError(
                f"Analysis requires every job completed: {job.get('job_id')}"
            )
        if str(job.get("experiment_stage", "")) != EXPERIMENT_STAGE:
            raise FixedLearningRateCapacityAnalysisError(
                f"Experiment stage mismatch: {job.get('job_id')}"
            )
        observed.append((_profile(job), _seed(job), _mode(job), _tolerance(job)))
    if len(observed) != 72 or len(set(observed)) != 72 or set(observed) != expected:
        raise FixedLearningRateCapacityAnalysisError(
            "Fixed-LR capacity analysis requires the exact 72-job matrix"
        )


def _expected_profile_sha(profile: str) -> str:
    return _capacity_profile_sha256(profile)


def _validate_profile_manifest(root: Path) -> pd.DataFrame:
    path = root / "capacity_seed_profile_manifest.csv"
    frame = pd.read_csv(path, low_memory=False)
    required = {"capacity_profile", "capacity_profile_sha256", "seed"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FixedLearningRateCapacityAnalysisError(
            f"Capacity-seed manifest missing {missing}"
        )
    keys = frame[["capacity_profile", "seed"]].copy()
    keys["capacity_profile"] = keys["capacity_profile"].astype(str)
    keys["seed"] = pd.to_numeric(keys["seed"], errors="coerce").astype(int)
    expected = {
        (profile, seed) for profile in EXPECTED_PROFILES for seed in EXPECTED_SEEDS
    }
    if (
        len(frame) != 18
        or keys.duplicated().any()
        or set(keys.itertuples(index=False, name=None)) != expected
    ):
        raise FixedLearningRateCapacityAnalysisError(
            "Capacity-seed manifest must contain 18 unique profile/seed rows"
        )
    for row in frame.to_dict(orient="records"):
        profile = str(row["capacity_profile"])
        if str(row["capacity_profile_sha256"]) != _expected_profile_sha(profile):
            raise FixedLearningRateCapacityAnalysisError(
                f"Capacity profile hash mismatch: {profile}"
            )
        if (
            "expected_regression_parameters" in frame.columns
            and int(row["expected_regression_parameters"])
            != PROFILE_PARAMETER_COUNTS[profile]["regression"]
        ):
            raise FixedLearningRateCapacityAnalysisError(
                f"Capacity parameter count mismatch: {profile}"
            )
        if "capacity_seed_profile_sha256" in frame.columns and str(
            row["capacity_seed_profile_sha256"]
        ) != _capacity_seed_profile_sha256(profile, int(row["seed"])):
            raise FixedLearningRateCapacityAnalysisError(
                f"Capacity+seed profile hash mismatch: {profile}/{row['seed']}"
            )
        for field, expected_value in (
            ("initial_learning_rate", FIXED_LEARNING_RATE),
            ("scheduler_min_lr", FIXED_SCHEDULER_MIN_LR),
        ):
            if field in frame.columns and not _same_float(row[field], expected_value):
                raise FixedLearningRateCapacityAnalysisError(
                    f"Capacity-seed manifest {field} mismatch"
                )
        if (
            "fixed_lr_profile" in frame.columns
            and str(row["fixed_lr_profile"]) != FIXED_LR_PROFILE
        ):
            raise FixedLearningRateCapacityAnalysisError("Fixed-LR profile ID mismatch")
        if (
            "fixed_lr_profile_sha256" in frame.columns
            and str(row["fixed_lr_profile_sha256"]) != _fixed_lr_profile_sha256()
        ):
            raise FixedLearningRateCapacityAnalysisError(
                "Fixed-LR profile hash mismatch"
            )
    return frame


def _resource_map(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "resource_summary.csv"
    frame = pd.read_csv(path, low_memory=False)
    if "job_id" not in frame.columns or frame["job_id"].astype(str).duplicated().any():
        raise FixedLearningRateCapacityAnalysisError("Invalid resource_summary.csv")
    return {str(row["job_id"]): dict(row) for row in frame.to_dict(orient="records")}


def _epoch_diagnostics(
    job: Mapping[str, Any], learned: Mapping[str, Any]
) -> tuple[float, float, int, bool]:
    epoch = int(learned["best_epoch"])
    path = _run_dir(job) / "metrics" / "training_metrics.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and int(row.get("epoch", -1)) == epoch
    ]
    if len(matches) != 1:
        raise FixedLearningRateCapacityAnalysisError(
            f"Cannot resolve learned epoch metrics: {path}"
        )
    train = _positive(
        _first_metric(matches[0], ("train_recon", "g_recon")),
        label="learned train reconstruction",
    )
    validation = _positive(
        _first_metric(matches[0], ("val_recon",)),
        label="learned validation reconstruction",
    )
    baseline_path = _run_dir(job) / "metrics" / "best_checkpoint.json"
    baseline = _read_json(baseline_path)
    baseline_epoch = int(_finite(baseline.get("best_epoch"), label="selected epoch"))
    if str(baseline.get("selection_scope", "")) != "baseline_inclusive":
        raise FixedLearningRateCapacityAnalysisError(
            f"Baseline-inclusive checkpoint scope drifted: {baseline_path}"
        )
    return train, validation, baseline_epoch, baseline_epoch == 0


def collect_run_diagnostics(experiment_root: str | Path) -> pd.DataFrame:
    """Collect learned-checkpoint, epoch and GPU lineage for all 72 jobs."""

    root = Path(experiment_root).resolve()
    jobs = _load_jobs(root)
    validate_job_matrix(jobs)
    manifest = _validate_profile_manifest(root)
    resources = _resource_map(root)
    manifest_index = manifest.set_index(["capacity_profile", "seed"])
    rows: list[dict[str, Any]] = []
    for job in jobs:
        profile, seed = _profile(job), _seed(job)
        declared = manifest_index.loc[(profile, seed)]
        profile_sha = str(job.get("capacity_profile_sha256", ""))
        seed_sha = _capacity_seed_profile_sha256(profile, seed)
        if profile_sha != str(declared["capacity_profile_sha256"]):
            raise FixedLearningRateCapacityAnalysisError(
                f"Registry/manifest capacity hash mismatch: {job.get('job_id')}"
            )
        job_contracts = (
            (
                str(job.get("capacity_seed_profile_sha256", "")) == seed_sha,
                "capacity+seed hash",
            ),
            (
                str(job.get("fixed_lr_profile", "")) == FIXED_LR_PROFILE,
                "fixed-LR profile",
            ),
            (
                str(job.get("fixed_lr_profile_sha256", ""))
                == _fixed_lr_profile_sha256(),
                "fixed-LR hash",
            ),
            (
                _same_float(job.get("initial_learning_rate"), FIXED_LEARNING_RATE),
                "job LR",
            ),
            (
                _same_float(job.get("scheduler_min_lr"), FIXED_SCHEDULER_MIN_LR),
                "job LR floor",
            ),
        )
        failed_jobs = [label for valid, label in job_contracts if not valid]
        if failed_jobs:
            raise FixedLearningRateCapacityAnalysisError(
                f"Registry/status lineage drift for {job.get('job_id')}: {failed_jobs}"
            )
        artifacts = list(job.get("artifacts") or [])
        if not artifacts:
            raise FixedLearningRateCapacityAnalysisError(
                f"Completed status lacks artifact hashes: {job.get('job_id')}"
            )
        for artifact in artifacts:
            path = Path(str(artifact.get("path", "")))
            if not path.is_file() or _sha256(path) != str(artifact.get("sha256", "")):
                raise FixedLearningRateCapacityAnalysisError(
                    f"Completed artifact hash mismatch: {job.get('job_id')}"
                )
        config_path, config = _training_config(job)
        if _sha256(config_path) != str(job.get("config_sha256", "")):
            raise FixedLearningRateCapacityAnalysisError(
                "Training config hash mismatch"
            )
        contracts = (
            (int(config.get("seed", -1)) == seed, "seed"),
            (str(config.get("news_first_capacity_profile", "")) == profile, "profile"),
            (
                str(config.get("news_first_capacity_profile_sha256", ""))
                == profile_sha,
                "profile hash",
            ),
            (
                str(config.get("news_first_capacity_seed_profile_sha256", ""))
                == seed_sha,
                "capacity+seed hash",
            ),
            (
                str(config.get("news_first_fixed_learning_rate_profile", ""))
                == FIXED_LR_PROFILE,
                "fixed-LR profile",
            ),
            (
                str(config.get("news_first_fixed_learning_rate_profile_sha256", ""))
                == _fixed_lr_profile_sha256(),
                "fixed-LR hash",
            ),
            (
                _same_float(config.get("learning_rate"), FIXED_LEARNING_RATE),
                "learning rate",
            ),
            (
                _same_float(config.get("reduce_lr_min_lr"), FIXED_SCHEDULER_MIN_LR),
                "minimum LR",
            ),
            (
                str(config.get("support_mask_mode", "")).lower() == "raw_joint",
                "support mask",
            ),
            (
                str(config.get("news_first_train_end_utc", Q3_START_UTC))
                == Q3_START_UTC,
                "train boundary",
            ),
            (
                str(config.get("news_first_validation_end_utc", Q3_END_UTC))
                == Q3_END_UTC,
                "Q3 boundary",
            ),
        )
        failed = [label for valid, label in contracts if not valid]
        if failed:
            raise FixedLearningRateCapacityAnalysisError(
                f"Training config contract drift for {job.get('job_id')}: {failed}"
            )
        metadata_path, learned = _learned_metadata(job)
        if int(learned.get("seed", -1)) != seed:
            raise FixedLearningRateCapacityAnalysisError("Checkpoint seed mismatch")
        if (
            str(learned.get("capacity_profile", "")) != profile
            or str(learned.get("capacity_profile_sha256", "")) != profile_sha
        ):
            raise FixedLearningRateCapacityAnalysisError("Checkpoint capacity mismatch")
        if str(learned.get("capacity_seed_profile_sha256", "")) != seed_sha:
            raise FixedLearningRateCapacityAnalysisError(
                "Checkpoint capacity+seed mismatch"
            )
        if (
            str(learned.get("fixed_lr_profile", "")) != FIXED_LR_PROFILE
            or str(learned.get("fixed_lr_profile_sha256", ""))
            != _fixed_lr_profile_sha256()
        ):
            raise FixedLearningRateCapacityAnalysisError(
                "Checkpoint fixed-LR lineage mismatch"
            )
        if not _same_float(learned.get("initial_learning_rate"), FIXED_LEARNING_RATE):
            raise FixedLearningRateCapacityAnalysisError("Checkpoint LR mismatch")
        if not _same_float(learned.get("scheduler_min_lr"), FIXED_SCHEDULER_MIN_LR):
            raise FixedLearningRateCapacityAnalysisError("Checkpoint LR floor mismatch")
        model_mae = _positive(
            _first_metric(learned, ("masked_pair_balanced_mae", "val_recon")),
            label="learned Q3 MAE",
        )
        persistence_mae = _positive(
            _first_metric(
                learned, ("persistence_masked_pair_balanced_mae", "val_current_recon")
            ),
            label="persistence Q3 MAE",
        )
        train_recon, val_recon, selected_epoch, epoch0 = _epoch_diagnostics(
            job, learned
        )
        job_id = str(job["job_id"])
        if job_id not in resources:
            raise FixedLearningRateCapacityAnalysisError(
                f"Missing resource row: {job_id}"
            )
        resource = resources[job_id]
        rows.append(
            {
                "job_id": job_id,
                "capacity_profile": profile,
                "capacity_profile_sha256": profile_sha,
                "capacity_seed_profile_sha256": str(
                    job.get("capacity_seed_profile_sha256", "")
                ),
                "parameter_count": PROFILE_PARAMETER_COUNTS[profile]["regression"],
                "parameter_log10": math.log10(
                    PROFILE_PARAMETER_COUNTS[profile]["regression"]
                ),
                "seed": seed,
                "text_ablation_mode": _mode(job),
                "tolerance_minutes": _tolerance(job),
                "initial_learning_rate": FIXED_LEARNING_RATE,
                "scheduler_min_lr": FIXED_SCHEDULER_MIN_LR,
                "best_learned_epoch": int(learned["best_epoch"]),
                "best_learned_train_recon": train_recon,
                "best_learned_val_recon": val_recon,
                "best_learned_train_val_gap": val_recon - train_recon,
                "baseline_inclusive_best_epoch": selected_epoch,
                "baseline_inclusive_epoch0_selected": epoch0,
                "model_mae": model_mae,
                "persistence_mae": persistence_mae,
                "mae_ratio": model_mae / persistence_mae,
                "runtime_minutes": _finite(
                    resource.get("runtime_minutes"), label="runtime"
                ),
                "gpu_hours": _finite(resource.get("gpu_hours", 0.0), label="GPU hours"),
                "peak_memory_mib": _finite(
                    resource.get("peak_memory_mib"), label="peak GPU memory"
                ),
                "mean_utilization_gpu_pct": _finite(
                    resource.get("mean_utilization_gpu_pct"),
                    label="mean GPU utilization",
                ),
                "gpu_id": int(resource.get("gpu_id", -1)),
                "metadata_path": str(metadata_path),
                "metadata_sha256": _sha256(metadata_path),
                "training_config_path": str(config_path),
                "run_dir": str(_run_dir(job)),
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != 72:
        raise FixedLearningRateCapacityAnalysisError("Run diagnostics lost jobs")
    return output.sort_values(
        ["parameter_count", "seed", "text_ablation_mode", "tolerance_minutes"],
        kind="stable",
    ).reset_index(drop=True)


def _run_spec(job: Mapping[str, Any]) -> RunSpec:
    metadata_path, learned = _learned_metadata(job)
    config_path, config = _training_config(job)
    return RunSpec(
        run_id=str(job["job_id"]),
        run_dir=_run_dir(job),
        model="regression",
        tolerance_minutes=_tolerance(job),
        seed=_seed(job),
        checkpoint_path=_learned_checkpoint(job, learned),
        text_ablation_mode=_mode(job),
        support_mask_mode=str(config.get("support_mask_mode", "none")),
        manifest_path=config_path,
        metadata={
            **config,
            "capacity_profile": _profile(job),
            "capacity_profile_sha256": str(job.get("capacity_profile_sha256", "")),
            "capacity_seed_profile_sha256": str(
                job.get("capacity_seed_profile_sha256", "")
            ),
            "best_learned_metadata_path": str(metadata_path),
        },
    )


def _q3_panel(jobs: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    sources: set[tuple[str, str, str]] = set()
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
        raise FixedLearningRateCapacityAnalysisError(
            "Jobs disagree on the common Q3 evaluation panel"
        )
    raw_path, sheet_name, support_mode = next(iter(sources))
    workbook = Path(raw_path).resolve()
    raw = pd.read_excel(workbook, sheet_name=sheet_name)
    timestamps = pd.to_datetime(
        raw.get("effective_origin_utc"), errors="coerce", utc=True
    )
    if timestamps.isna().any():
        raise FixedLearningRateCapacityAnalysisError("Invalid Q3 panel timestamps")
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
        raise FixedLearningRateCapacityAnalysisError("A non-Q3 row reached evaluation")
    lineage.update(
        {
            "path": str(workbook),
            "sha256": _sha256(workbook),
            "interval_start_utc": Q3_START_UTC,
            "interval_end_utc_exclusive": Q3_END_UTC,
            "q4_rows_passed_to_evaluator": 0,
        }
    )
    return panel, lineage


def evaluate_q3_pair_metrics(
    experiment_root: str | Path,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate all 72 best-learned checkpoints on one common Q3 panel."""

    root = Path(experiment_root).resolve()
    jobs = _load_jobs(root)
    validate_job_matrix(jobs)
    _validate_profile_manifest(root)
    panel, lineage = _q3_panel(jobs)
    production = evaluator or TrainedRunEvaluator(mc_samples=16)
    parts: list[pd.DataFrame] = []
    for job in jobs:
        spec = _run_spec(job)
        predictions = production(spec, "core", panel.copy())
        samples, exclusions, _ = compute_sample_metrics(
            spec,
            SELECTION_PANEL,
            panel,
            predictions,
            evaluate_embedded_atm_skew=False,
        )
        if not exclusions.empty:
            raise FixedLearningRateCapacityAnalysisError(
                f"Q3 prediction exclusions for {spec.run_id}: "
                f"{exclusions['exclusion_code'].value_counts().to_dict()}"
            )
        pairs = aggregate_pair_metrics(samples)
        pairs = pairs[
            pairs["stratum_type"].astype(str).eq("overall")
            & pairs["stratum_value"].astype(str).eq("all")
        ].copy()
        if not pairs["seed"].astype(int).eq(_seed(job)).all():
            raise FixedLearningRateCapacityAnalysisError(
                f"Pair metrics lost seed lineage: {spec.run_id}"
            )
        _, learned = _learned_metadata(job)
        expected = {
            "model": _positive(
                _first_metric(learned, ("masked_pair_balanced_mae", "val_recon")),
                label="metadata model MAE",
            ),
            "persistence": _positive(
                _first_metric(
                    learned,
                    ("persistence_masked_pair_balanced_mae", "val_current_recon"),
                ),
                label="metadata persistence MAE",
            ),
        }
        actual = {
            "model": float(pairs["model_mae"].mean()),
            "persistence": float(pairs["persistence_mae"].mean()),
        }
        for label in expected:
            tolerance = max(1.0e-7, abs(expected[label]) * 1.0e-4)
            if not math.isclose(
                actual[label], expected[label], rel_tol=0.0, abs_tol=tolerance
            ):
                raise FixedLearningRateCapacityAnalysisError(
                    f"Pair-balanced {label} MAE disagrees with metadata: {spec.run_id}"
                )
        profile, seed = _profile(job), _seed(job)
        pairs.insert(0, "capacity_profile", profile)
        pairs.insert(
            1, "capacity_profile_sha256", str(job.get("capacity_profile_sha256", ""))
        )
        pairs.insert(
            2,
            "capacity_seed_profile_sha256",
            str(job.get("capacity_seed_profile_sha256", "")),
        )
        pairs.insert(
            3, "parameter_count", PROFILE_PARAMETER_COUNTS[profile]["regression"]
        )
        pairs.insert(4, "initial_learning_rate", FIXED_LEARNING_RATE)
        if not pairs["seed"].astype(int).eq(seed).all():
            raise FixedLearningRateCapacityAnalysisError(
                "Pair seed changed unexpectedly"
            )
        parts.append(pairs)
    output = pd.concat(parts, ignore_index=True)
    lineage.update(
        {
            "selection_scope_verified": True,
            "q4_used_for_selection": False,
            "evaluated_run_count": len(jobs),
        }
    )
    return output, lineage


def validate_pair_metric_matrix(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    """Fail closed unless pair metrics preserve the exact 72-cell panel."""

    required = {
        "capacity_profile",
        "capacity_profile_sha256",
        "capacity_seed_profile_sha256",
        "parameter_count",
        "initial_learning_rate",
        "seed",
        "text_ablation_mode",
        "tolerance_minutes",
        "pair_id",
        "session_id",
        "model_mae",
        "persistence_mae",
    }
    missing = sorted(required - set(pair_metrics.columns))
    if missing:
        raise FixedLearningRateCapacityAnalysisError(
            f"Q3 pair metrics missing {missing}"
        )
    frame = pair_metrics.copy()
    frame["capacity_profile"] = frame["capacity_profile"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").astype(int)
    frame["tolerance_minutes"] = pd.to_numeric(
        frame["tolerance_minutes"], errors="coerce"
    ).astype(int)
    frame["initial_learning_rate"] = pd.to_numeric(
        frame["initial_learning_rate"], errors="coerce"
    )
    for column in ("model_mae", "persistence_mae"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all() or bool(
            (frame[column] <= 0.0).any()
        ):
            raise FixedLearningRateCapacityAnalysisError(
                f"{column} must be finite and positive"
            )
    if set(frame["capacity_profile"]) != set(EXPECTED_PROFILES):
        raise FixedLearningRateCapacityAnalysisError(
            "Capacity profiles differ from contract"
        )
    if set(frame["seed"]) != set(EXPECTED_SEEDS):
        raise FixedLearningRateCapacityAnalysisError(
            "Training seeds differ from contract"
        )
    if set(frame["text_ablation_mode"].astype(str)) != set(EXPECTED_TEXT_MODES):
        raise FixedLearningRateCapacityAnalysisError("Text modes differ from contract")
    if set(frame["tolerance_minutes"]) != set(EXPECTED_TOLERANCES):
        raise FixedLearningRateCapacityAnalysisError("Tolerances differ from contract")
    if (
        not frame["initial_learning_rate"]
        .map(lambda value: _same_float(value, FIXED_LEARNING_RATE))
        .all()
    ):
        raise FixedLearningRateCapacityAnalysisError(
            "Learning rate is not fixed at 5e-7"
        )
    for (profile, seed), group in frame.groupby(
        ["capacity_profile", "seed"], sort=False
    ):
        expected_profile_sha = _expected_profile_sha(str(profile))
        expected_seed_sha = _capacity_seed_profile_sha256(str(profile), int(seed))
        expected_parameters = PROFILE_PARAMETER_COUNTS[str(profile)]["regression"]
        if set(group["capacity_profile_sha256"].astype(str)) != {expected_profile_sha}:
            raise FixedLearningRateCapacityAnalysisError("Capacity profile hash drift")
        if set(group["capacity_seed_profile_sha256"].astype(str)) != {
            expected_seed_sha
        }:
            raise FixedLearningRateCapacityAnalysisError("Capacity+seed hash drift")
        if set(group["parameter_count"].astype(int)) != {expected_parameters}:
            raise FixedLearningRateCapacityAnalysisError("Parameter count drift")

    job_key = [
        "capacity_profile",
        "seed",
        "text_ablation_mode",
        "tolerance_minutes",
    ]
    if frame[job_key].drop_duplicates().shape[0] != 72:
        raise FixedLearningRateCapacityAnalysisError(
            "Pair metrics lost the 72-job matrix"
        )
    if frame.duplicated(job_key + ["pair_id"]).any():
        raise FixedLearningRateCapacityAnalysisError("Duplicate seed-aware pair metric")
    coverages: list[frozenset[tuple[str, str]]] = []
    for _, group in frame.groupby(job_key, sort=False):
        if group["pair_id"].nunique() != EXPECTED_PAIR_COUNT:
            raise FixedLearningRateCapacityAnalysisError(
                f"Expected {EXPECTED_PAIR_COUNT} Q3 pairs per run"
            )
        if group["session_id"].nunique() != EXPECTED_SESSION_COUNT:
            raise FixedLearningRateCapacityAnalysisError(
                f"Expected {EXPECTED_SESSION_COUNT} Q3 sessions per run"
            )
        coverages.append(
            frozenset(
                group[["pair_id", "session_id"]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
        )
    if len(set(coverages)) != 1:
        raise FixedLearningRateCapacityAnalysisError(
            "Every capacity/seed/mode/tolerance must use identical Q3 coverage"
        )
    return frame.reset_index(drop=True)


def build_seed_run_metrics(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse pairs to one row per capacity/seed/mode/tolerance run."""

    frame = validate_pair_metric_matrix(pair_metrics)
    keys = [
        "capacity_profile",
        "capacity_profile_sha256",
        "capacity_seed_profile_sha256",
        "parameter_count",
        "initial_learning_rate",
        "seed",
        "text_ablation_mode",
        "tolerance_minutes",
    ]
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, values))
        model = float(group["model_mae"].mean())
        persistence = float(group["persistence_mae"].mean())
        ratio = model / persistence
        row.update(
            {
                "pair_count": int(group["pair_id"].nunique()),
                "session_count": int(group["session_id"].nunique()),
                "model_mae": model,
                "persistence_mae": persistence,
                "mae_gap": model - persistence,
                "mae_ratio": ratio,
                "log_mae_ratio": math.log(ratio),
                "improvement_fraction": 1.0 - ratio,
                "pair_win_rate": float(
                    (group["model_mae"] < group["persistence_mae"]).mean()
                ),
            }
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    if len(output) != 72:
        raise FixedLearningRateCapacityAnalysisError("Expected 72 seed-run rows")
    return output.sort_values(
        ["parameter_count", "seed", "text_ablation_mode", "tolerance_minutes"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_across_seeds(seed_runs: pd.DataFrame) -> pd.DataFrame:
    """Report mean and sample SD across the three independent training seeds."""

    keys = [
        "capacity_profile",
        "capacity_profile_sha256",
        "parameter_count",
        "initial_learning_rate",
        "text_ablation_mode",
        "tolerance_minutes",
    ]
    metrics = (
        "model_mae",
        "persistence_mae",
        "mae_gap",
        "mae_ratio",
        "log_mae_ratio",
        "improvement_fraction",
        "pair_win_rate",
    )
    rows: list[dict[str, Any]] = []
    for values, group in seed_runs.groupby(keys, sort=True, dropna=False):
        seeds = tuple(sorted(group["seed"].astype(int).tolist()))
        if seeds != EXPECTED_SEEDS:
            raise FixedLearningRateCapacityAnalysisError("Across-seed cell lost a seed")
        row = dict(zip(keys, values))
        row.update(
            {
                "seed_count": len(seeds),
                "seeds": "|".join(str(value) for value in seeds),
                "pair_count_per_seed": int(group["pair_count"].iloc[0]),
                "session_count_per_seed": int(group["session_count"].iloc[0]),
            }
        )
        for metric in metrics:
            values_array = pd.to_numeric(group[metric], errors="coerce").to_numpy(
                dtype=float
            )
            if not np.isfinite(values_array).all():
                raise FixedLearningRateCapacityAnalysisError(
                    f"Non-finite across-seed metric: {metric}"
                )
            row[f"{metric}_mean"] = float(values_array.mean())
            row[f"{metric}_sd"] = float(values_array.std(ddof=1))
        rows.append(row)
    output = pd.DataFrame(rows)
    if len(output) != 24:
        raise FixedLearningRateCapacityAnalysisError("Expected 24 across-seed cells")
    return output.sort_values(
        ["parameter_count", "text_ablation_mode", "tolerance_minutes"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_capacity_scores(seed_runs: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight 5m/30m log-ratio score, then equal-weight seeds."""

    per_seed_rows: list[dict[str, Any]] = []
    keys = [
        "capacity_profile",
        "capacity_profile_sha256",
        "parameter_count",
        "seed",
        "text_ablation_mode",
    ]
    for values, group in seed_runs.groupby(keys, sort=True, dropna=False):
        tolerances = tuple(sorted(group["tolerance_minutes"].astype(int).tolist()))
        if tolerances != EXPECTED_TOLERANCES:
            raise FixedLearningRateCapacityAnalysisError(
                "Capacity score lost a tolerance"
            )
        score = float(group["log_mae_ratio"].mean())
        per_seed_rows.append(
            dict(zip(keys, values))
            | {
                "mean_log_mae_ratio": score,
                "geometric_mae_ratio": math.exp(score),
                "improvement_fraction": 1.0 - math.exp(score),
                "all_tolerances_not_worse": bool((group["mae_ratio"] <= 1.0).all()),
            }
        )
    per_seed = pd.DataFrame(per_seed_rows)
    rows: list[dict[str, Any]] = []
    outer_keys = [
        "capacity_profile",
        "capacity_profile_sha256",
        "parameter_count",
        "text_ablation_mode",
    ]
    for values, group in per_seed.groupby(outer_keys, sort=True, dropna=False):
        seeds = tuple(sorted(group["seed"].astype(int).tolist()))
        if seeds != EXPECTED_SEEDS:
            raise FixedLearningRateCapacityAnalysisError("Capacity score lost a seed")
        scores = group["mean_log_mae_ratio"].to_numpy(dtype=float)
        mean_score = float(scores.mean())
        geometric = math.exp(mean_score)
        rows.append(
            dict(zip(outer_keys, values))
            | {
                "seed_count": len(seeds),
                "mean_log_mae_ratio": mean_score,
                "sd_log_mae_ratio_across_seeds": float(scores.std(ddof=1)),
                "geometric_mae_ratio": geometric,
                "mean_improvement_fraction": 1.0 - geometric,
                "all_seed_tolerances_not_worse": bool(
                    group["all_tolerances_not_worse"].astype(bool).all()
                ),
            }
        )
    output = pd.DataFrame(rows)
    output["gate_passed"] = (
        output["mean_improvement_fraction"] >= GATE_MIN_IMPROVEMENT
    ) & output["all_seed_tolerances_not_worse"]
    if len(output) != 12:
        raise FixedLearningRateCapacityAnalysisError("Expected 12 capacity score rows")
    return output.sort_values(
        ["text_ablation_mode", "mean_log_mae_ratio", "parameter_count"],
        kind="stable",
    ).reset_index(drop=True)


def _two_level_bootstrap(
    differences: pd.DataFrame,
    *,
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    """Resample seeds, then paired CME-session clusters inside each seed."""

    if int(iterations) < 2:
        raise FixedLearningRateCapacityAnalysisError("Bootstrap needs at least 2 draws")
    required = {"seed", "session_id", "pair_id", "difference"}
    missing = sorted(required - set(differences.columns))
    if missing:
        raise FixedLearningRateCapacityAnalysisError(
            f"Bootstrap differences missing {missing}"
        )
    frame = differences.copy()
    frame["difference"] = pd.to_numeric(frame["difference"], errors="coerce")
    if not np.isfinite(frame["difference"].to_numpy(dtype=float)).all():
        raise FixedLearningRateCapacityAnalysisError(
            "Bootstrap differences are non-finite"
        )
    seeds = tuple(sorted(frame["seed"].astype(int).unique().tolist()))
    if seeds != EXPECTED_SEEDS:
        raise FixedLearningRateCapacityAnalysisError("Bootstrap lost a training seed")
    extras = ["tolerance_minutes"] if "tolerance_minutes" in frame.columns else []
    if frame.duplicated(["seed", "session_id", "pair_id", *extras]).any():
        raise FixedLearningRateCapacityAnalysisError("Duplicate paired bootstrap row")

    coverage: list[frozenset[tuple[str, ...]]] = []
    arrays: dict[int, tuple[np.ndarray, np.ndarray, int, int]] = {}
    points: list[float] = []
    for seed in seeds:
        selected = frame[frame["seed"].astype(int).eq(seed)]
        coverage.append(
            frozenset(
                selected[["session_id", "pair_id", *extras]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
        )
        grouped = selected.groupby("session_id", sort=True).agg(
            difference_sum=("difference", "sum"),
            row_count=("difference", "size"),
        )
        sums = grouped["difference_sum"].to_numpy(dtype=float)
        counts = grouped["row_count"].to_numpy(dtype=float)
        arrays[seed] = (sums, counts, len(grouped), len(selected))
        points.append(float(sums.sum() / counts.sum()))
    if len(set(coverage)) != 1:
        raise FixedLearningRateCapacityAnalysisError(
            "Two-level inference requires identical coverage across seeds"
        )

    rng = np.random.default_rng(int(random_seed))
    seed_indexes = rng.integers(0, len(seeds), size=(int(iterations), len(seeds)))
    draws = np.empty(int(iterations), dtype=float)
    for draw in range(int(iterations)):
        seed_means: list[float] = []
        for raw_index in seed_indexes[draw]:
            chosen = seeds[int(raw_index)]
            sums, counts, session_count, _ = arrays[chosen]
            sessions = rng.integers(0, session_count, size=session_count)
            seed_means.append(float(sums[sessions].sum() / counts[sessions].sum()))
        draws[draw] = float(np.mean(seed_means))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    p_lower = (float(np.sum(draws <= 0.0)) + 1.0) / (len(draws) + 1.0)
    p_upper = (float(np.sum(draws >= 0.0)) + 1.0) / (len(draws) + 1.0)
    return {
        "mean_difference": float(np.mean(points)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_two_sided": float(min(1.0, 2.0 * min(p_lower, p_upper))),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(random_seed),
        "seed_count": len(seeds),
        "session_count_per_seed": int(arrays[seeds[0]][2]),
        "pair_rows_per_seed": int(arrays[seeds[0]][3]),
        "resampling_method": "seed_then_paired_CME_session_cluster",
    }


def _scopes(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("05m", frame[frame["tolerance_minutes"].astype(int).eq(5)]),
        ("30m", frame[frame["tolerance_minutes"].astype(int).eq(30)]),
        ("combined_05m_30m", frame),
    ]


def build_bootstrap_tables(
    pair_metrics: pd.DataFrame,
    *,
    ranked_leader_profile: str,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build persistence, text and capacity paired two-level inference."""

    frame = validate_pair_metric_matrix(pair_metrics)
    persistence_rows: list[dict[str, Any]] = []
    for (profile, parameters, mode), group in frame.groupby(
        ["capacity_profile", "parameter_count", "text_ablation_mode"], sort=True
    ):
        for scope, scoped in _scopes(group):
            differences = scoped[
                ["seed", "session_id", "pair_id", "tolerance_minutes"]
            ].copy()
            differences["difference"] = scoped["model_mae"].to_numpy(
                dtype=float
            ) - scoped["persistence_mae"].to_numpy(dtype=float)
            persistence_rows.append(
                {
                    "capacity_profile": profile,
                    "parameter_count": int(parameters),
                    "text_ablation_mode": mode,
                    "tolerance_scope": scope,
                    "contrast": "model_minus_persistence",
                    **_two_level_bootstrap(
                        differences,
                        iterations=iterations,
                        random_seed=random_seed,
                    ),
                }
            )
    persistence = pd.DataFrame(persistence_rows)
    persistence["p_holm"] = holm_adjust(persistence["p_two_sided"].tolist())

    merge_keys = [
        "capacity_profile",
        "parameter_count",
        "seed",
        "tolerance_minutes",
        "pair_id",
        "session_id",
    ]
    current = frame[frame["text_ablation_mode"].eq("current_only")][
        merge_keys + ["model_mae"]
    ].rename(columns={"model_mae": "current_only_mae"})
    real = frame[frame["text_ablation_mode"].eq("real_text")][
        merge_keys + ["model_mae"]
    ].rename(columns={"model_mae": "real_text_mae"})
    paired_text = real.merge(current, on=merge_keys, validate="one_to_one")
    if len(paired_text) != len(real) or len(paired_text) != len(current):
        raise FixedLearningRateCapacityAnalysisError("Text contrast lost paired rows")
    text_rows: list[dict[str, Any]] = []
    for (profile, parameters), group in paired_text.groupby(
        ["capacity_profile", "parameter_count"], sort=True
    ):
        for scope, scoped in _scopes(group):
            differences = scoped[
                ["seed", "session_id", "pair_id", "tolerance_minutes"]
            ].copy()
            differences["difference"] = scoped["real_text_mae"].to_numpy(
                dtype=float
            ) - scoped["current_only_mae"].to_numpy(dtype=float)
            text_rows.append(
                {
                    "capacity_profile": profile,
                    "parameter_count": int(parameters),
                    "tolerance_scope": scope,
                    "contrast": "real_text_minus_current_only",
                    **_two_level_bootstrap(
                        differences,
                        iterations=iterations,
                        random_seed=random_seed + 101,
                    ),
                }
            )
    text = pd.DataFrame(text_rows)
    text["p_holm"] = holm_adjust(text["p_two_sided"].tolist())

    lane = frame[frame["text_ablation_mode"].eq(SELECTION_MODE)].copy()
    reference = lane[lane["capacity_profile"].eq(ranked_leader_profile)]
    if reference.empty:
        raise FixedLearningRateCapacityAnalysisError("Ranked capacity leader is absent")
    capacity_rows: list[dict[str, Any]] = []
    for (profile, parameters), candidate in lane[
        ~lane["capacity_profile"].eq(ranked_leader_profile)
    ].groupby(["capacity_profile", "parameter_count"], sort=True):
        pair_keys = ["seed", "tolerance_minutes", "pair_id", "session_id"]
        paired = candidate[pair_keys + ["model_mae"]].merge(
            reference[pair_keys + ["model_mae"]],
            on=pair_keys,
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        if len(paired) != len(candidate) or len(paired) != len(reference):
            raise FixedLearningRateCapacityAnalysisError(
                "Capacity contrast lost paired Q3 rows"
            )
        for scope, scoped in _scopes(paired):
            differences = scoped[
                ["seed", "session_id", "pair_id", "tolerance_minutes"]
            ].copy()
            differences["difference"] = scoped["model_mae_candidate"].to_numpy(
                dtype=float
            ) - scoped["model_mae_reference"].to_numpy(dtype=float)
            capacity_rows.append(
                {
                    "candidate_capacity_profile": profile,
                    "candidate_parameter_count": int(parameters),
                    "reference_capacity_profile": ranked_leader_profile,
                    "reference_parameter_count": PROFILE_PARAMETER_COUNTS[
                        ranked_leader_profile
                    ]["regression"],
                    "tolerance_scope": scope,
                    "text_ablation_mode": SELECTION_MODE,
                    "contrast": "candidate_minus_ranked_leader",
                    **_two_level_bootstrap(
                        differences,
                        iterations=iterations,
                        random_seed=random_seed + 202,
                    ),
                }
            )
    capacity = pd.DataFrame(capacity_rows)
    capacity["p_holm"] = holm_adjust(capacity["p_two_sided"].tolist())
    return persistence, text, capacity


def build_one_se_table(
    pair_metrics: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Compute paired seed/session SEs for the real-text capacity score."""

    frame = validate_pair_metric_matrix(pair_metrics)
    lane = frame[frame["text_ablation_mode"].eq(SELECTION_MODE)].copy()
    profiles = sorted(
        EXPECTED_PROFILES,
        key=lambda value: PROFILE_PARAMETER_COUNTS[value]["regression"],
    )
    sessions = sorted(lane["session_id"].astype(str).unique().tolist())
    if len(sessions) != EXPECTED_SESSION_COUNT:
        raise FixedLearningRateCapacityAnalysisError("one-SE session count drift")

    arrays: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for profile in profiles:
        for seed in EXPECTED_SEEDS:
            for tolerance in EXPECTED_TOLERANCES:
                selected = lane[
                    lane["capacity_profile"].eq(profile)
                    & lane["seed"].eq(seed)
                    & lane["tolerance_minutes"].eq(tolerance)
                ]
                grouped = (
                    selected.groupby("session_id", sort=True)
                    .agg(
                        model_sum=("model_mae", "sum"),
                        persistence_sum=("persistence_mae", "sum"),
                        pair_count=("pair_id", "size"),
                    )
                    .reindex(sessions)
                )
                if grouped.isna().any().any():
                    raise FixedLearningRateCapacityAnalysisError(
                        "one-SE capacity panel lacks a paired session"
                    )
                arrays[(profile, seed, tolerance)] = (
                    grouped["model_sum"].to_numpy(dtype=float),
                    grouped["persistence_sum"].to_numpy(dtype=float),
                    grouped["pair_count"].to_numpy(dtype=float),
                )

    rng = np.random.default_rng(int(random_seed))
    seed_draws = rng.integers(
        0, len(EXPECTED_SEEDS), size=(int(iterations), len(EXPECTED_SEEDS))
    )
    draws = {profile: np.empty(int(iterations), dtype=float) for profile in profiles}
    for draw_index in range(int(iterations)):
        sampled: list[tuple[int, np.ndarray]] = []
        for seed_index in seed_draws[draw_index]:
            sampled_seed = EXPECTED_SEEDS[int(seed_index)]
            session_indexes = rng.integers(0, len(sessions), size=len(sessions))
            sampled.append((sampled_seed, session_indexes))
        for profile in profiles:
            terms: list[float] = []
            for sampled_seed, indexes in sampled:
                for tolerance in EXPECTED_TOLERANCES:
                    model, persistence, counts = arrays[
                        (profile, sampled_seed, tolerance)
                    ]
                    model_mean = float(model[indexes].sum() / counts[indexes].sum())
                    persistence_mean = float(
                        persistence[indexes].sum() / counts[indexes].sum()
                    )
                    terms.append(math.log(model_mean / persistence_mean))
            draws[profile][draw_index] = float(np.mean(terms))

    selection_scores = scores[scores["text_ablation_mode"].eq(SELECTION_MODE)].copy()
    if len(selection_scores) != len(profiles):
        raise FixedLearningRateCapacityAnalysisError(
            "one-SE score matrix is incomplete"
        )
    point = {
        str(row.capacity_profile): float(row.mean_log_mae_ratio)
        for row in selection_scores.itertuples()
    }
    standard_errors = {
        profile: float(draws[profile].std(ddof=1)) for profile in profiles
    }
    diagnostic_best = min(
        profiles,
        key=lambda profile: (
            point[profile],
            PROFILE_PARAMETER_COUNTS[profile]["regression"],
        ),
    )
    diagnostic_threshold = point[diagnostic_best] + standard_errors[diagnostic_best]
    gate = {
        str(row.capacity_profile): bool(row.gate_passed)
        for row in selection_scores.itertuples()
    }
    gate_profiles = [profile for profile in profiles if gate[profile]]
    formal_best = (
        min(
            gate_profiles,
            key=lambda profile: (
                point[profile],
                PROFILE_PARAMETER_COUNTS[profile]["regression"],
            ),
        )
        if gate_profiles
        else ""
    )
    formal_threshold = (
        point[formal_best] + standard_errors[formal_best] if formal_best else None
    )
    formal_eligible = [
        profile
        for profile in gate_profiles
        if point[profile] <= float(formal_threshold) + 1.0e-15
    ]
    winner = (
        min(
            formal_eligible,
            key=lambda profile: PROFILE_PARAMETER_COUNTS[profile]["regression"],
        )
        if formal_eligible
        else ""
    )
    rows = []
    for profile in profiles:
        rows.append(
            {
                "capacity_profile": profile,
                "parameter_count": PROFILE_PARAMETER_COUNTS[profile]["regression"],
                "mean_log_mae_ratio": point[profile],
                "geometric_mae_ratio": math.exp(point[profile]),
                "bootstrap_standard_error": standard_errors[profile],
                "gate_passed": gate[profile],
                "diagnostic_best_score_profile": diagnostic_best,
                "diagnostic_one_se_threshold": diagnostic_threshold,
                "diagnostic_within_one_se": point[profile]
                <= diagnostic_threshold + 1.0e-15,
                "formal_best_gate_profile": formal_best,
                "formal_one_se_threshold": formal_threshold,
                "formal_one_se_eligible": profile in formal_eligible,
                "selected_capacity": profile == winner,
                "bootstrap_iterations": int(iterations),
                "bootstrap_seed": int(random_seed),
                "resampling_method": "seed_then_paired_CME_session_cluster",
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("parameter_count", kind="stable")
        .reset_index(drop=True)
    )


def summarize_resources(run_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarize epoch and assigned-GPU telemetry at capacity grain."""

    required = {
        "capacity_profile",
        "parameter_count",
        "seed",
        "best_learned_epoch",
        "best_learned_train_val_gap",
        "baseline_inclusive_epoch0_selected",
        "runtime_minutes",
        "gpu_hours",
        "peak_memory_mib",
        "mean_utilization_gpu_pct",
    }
    missing = sorted(required - set(run_diagnostics.columns))
    if missing:
        raise FixedLearningRateCapacityAnalysisError(
            f"Run diagnostics missing {missing}"
        )
    rows: list[dict[str, Any]] = []
    for (profile, parameters), group in run_diagnostics.groupby(
        ["capacity_profile", "parameter_count"], sort=True
    ):
        row: dict[str, Any] = {
            "capacity_profile": profile,
            "parameter_count": int(parameters),
            "run_count": int(len(group)),
            "seed_count": int(group["seed"].nunique()),
        }
        for column in (
            "best_learned_epoch",
            "best_learned_train_val_gap",
            "runtime_minutes",
            "peak_memory_mib",
            "mean_utilization_gpu_pct",
        ):
            numeric = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_mean"] = (
                float(numeric.mean()) if len(numeric) else float("nan")
            )
            row[f"{column}_sd"] = (
                float(numeric.std(ddof=1)) if len(numeric) > 1 else float("nan")
            )
        gpu_hours = pd.to_numeric(group["gpu_hours"], errors="coerce").dropna()
        runtimes = pd.to_numeric(group["runtime_minutes"], errors="coerce").dropna()
        peak = pd.to_numeric(group["peak_memory_mib"], errors="coerce").dropna()
        row["gpu_hours_total"] = (
            float(gpu_hours.sum()) if len(gpu_hours) else float("nan")
        )
        row["runtime_minutes_total"] = (
            float(runtimes.sum()) if len(runtimes) else float("nan")
        )
        row["peak_memory_mib_max"] = float(peak.max()) if len(peak) else float("nan")
        epoch0 = group["baseline_inclusive_epoch0_selected"].map(
            lambda value: str(value).strip().lower() in {"1", "true", "yes"}
        )
        row["baseline_inclusive_epoch0_rate"] = float(epoch0.mean())
        rows.append(row)
    output = pd.DataFrame(rows)
    if len(output) != 6:
        raise FixedLearningRateCapacityAnalysisError("Expected six resource profiles")
    return output.sort_values("parameter_count", kind="stable").reset_index(drop=True)


def _injected_run_diagnostics(seed_runs: pd.DataFrame) -> pd.DataFrame:
    """Provide schema-complete unavailable-resource rows for fixture analysis."""

    output = seed_runs[
        [
            "capacity_profile",
            "parameter_count",
            "seed",
            "text_ablation_mode",
            "tolerance_minutes",
        ]
    ].copy()
    for column in (
        "best_learned_epoch",
        "best_learned_train_val_gap",
        "runtime_minutes",
        "gpu_hours",
        "peak_memory_mib",
        "mean_utilization_gpu_pct",
    ):
        output[column] = float("nan")
    output["baseline_inclusive_epoch0_selected"] = False
    return output


def build_selection_summary(
    scores: pd.DataFrame,
    one_se: pd.DataFrame,
    *,
    resolved_config_sha256: str = "injected_pair_metrics",
) -> dict[str, Any]:
    """Freeze the real-text gate and one-SE decision without consulting Q4."""

    lane = scores[scores["text_ablation_mode"].eq(SELECTION_MODE)].sort_values(
        ["mean_log_mae_ratio", "parameter_count"], kind="stable"
    )
    diagnostic = scores[scores["text_ablation_mode"].eq("current_only")].sort_values(
        ["mean_log_mae_ratio", "parameter_count"], kind="stable"
    )
    if len(lane) != 6 or len(diagnostic) != 6 or len(one_se) != 6:
        raise FixedLearningRateCapacityAnalysisError(
            "Capacity selection matrix is incomplete"
        )
    leader = lane.iloc[0]
    winners = one_se[one_se["selected_capacity"].astype(bool)]
    if len(winners) > 1:
        raise FixedLearningRateCapacityAnalysisError(
            "one-SE selected multiple capacities"
        )
    winner = str(winners.iloc[0]["capacity_profile"]) if len(winners) else ""
    winner_row = lane[lane["capacity_profile"].eq(winner)] if winner else lane.iloc[0:0]
    gate_passed = bool(winner)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "artifact_kind": "fixed_lr_multi_seed_capacity_analysis",
        "resolved_config_sha256": str(resolved_config_sha256),
        "selection_scope": "Q3 common_validation_05m only",
        "selection_mode": SELECTION_MODE,
        "diagnostic_mode": "current_only",
        "selection_tolerances": list(EXPECTED_TOLERANCES),
        "fixed_initial_learning_rate": FIXED_LEARNING_RATE,
        "fixed_scheduler_min_lr": FIXED_SCHEDULER_MIN_LR,
        "score_definition": (
            "equal-tolerance mean log(model MAE / persistence MAE), then equal-seed mean"
        ),
        "ranked_leader_capacity_profile": str(leader["capacity_profile"]),
        "ranked_leader_capacity_profile_sha256": str(leader["capacity_profile_sha256"]),
        "ranked_leader_parameter_count": int(leader["parameter_count"]),
        "ranked_leader_mean_log_mae_ratio": float(leader["mean_log_mae_ratio"]),
        "ranked_leader_mean_improvement_fraction": float(
            leader["mean_improvement_fraction"]
        ),
        "gate_minimum_mean_improvement_fraction": GATE_MIN_IMPROVEMENT,
        "gate_requires_all_seed_tolerances_not_worse": True,
        "gate_passed": gate_passed,
        "winner_capacity_profile": winner,
        "winner_capacity_profile_sha256": (
            str(winner_row.iloc[0]["capacity_profile_sha256"]) if winner else ""
        ),
        "winner_parameter_count": (
            int(winner_row.iloc[0]["parameter_count"]) if winner else None
        ),
        "one_standard_error_rule": {
            "method": "seed_then_paired_CME_session_cluster_one_standard_error",
            "bootstrap_iterations": int(one_se["bootstrap_iterations"].iloc[0]),
            "bootstrap_seed": int(one_se["bootstrap_seed"].iloc[0]),
            "best_gate_profile": str(one_se["formal_best_gate_profile"].iloc[0]),
            "threshold": (
                float(one_se["formal_one_se_threshold"].dropna().iloc[0])
                if one_se["formal_one_se_threshold"].notna().any()
                else None
            ),
            "eligible_profiles": one_se.loc[
                one_se["formal_one_se_eligible"].astype(bool), "capacity_profile"
            ]
            .astype(str)
            .tolist(),
            "winner_profile": winner,
        },
        "selection_lane_scores": lane.to_dict(orient="records"),
        "current_only_diagnostic_scores": diagnostic.to_dict(orient="records"),
        "one_se_profile_statistics": one_se.to_dict(orient="records"),
        "seed_count": len(EXPECTED_SEEDS),
        "seeds": list(EXPECTED_SEEDS),
        "q3_start_utc": Q3_START_UTC,
        "q3_end_utc_exclusive": Q3_END_UTC,
        "q4_used_for_selection": False,
        "q4_predictions_generated": False,
        "created_at_utc": _utc_now(),
    }
    result["selection_payload_sha256"] = _payload_sha256(result)
    return result


def run_fixed_lr_capacity_analysis(
    experiment_root: str | Path,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
    q3_pair_metrics: pd.DataFrame | None = None,
    run_diagnostics: pd.DataFrame | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Path:
    """Write the complete fixed-LR capacity Q3 analysis artifact set."""

    root = Path(experiment_root).resolve()
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if q3_pair_metrics is None:
        diagnostics = collect_run_diagnostics(root)
        raw_pairs, lineage = evaluate_q3_pair_metrics(root, evaluator=evaluator)
        pair_frame = validate_pair_metric_matrix(raw_pairs)
        resolved_config_sha256 = (
            (root / "registry" / "resolved_config.sha256")
            .read_text(encoding="utf-8")
            .strip()
        )
    else:
        pair_frame = validate_pair_metric_matrix(q3_pair_metrics)
        lineage = {
            "panel": SELECTION_PANEL,
            "interval_start_utc": Q3_START_UTC,
            "interval_end_utc_exclusive": Q3_END_UTC,
            "q4_rows_passed_to_evaluator": 0,
            "q4_used_for_selection": False,
            "injected_pair_metrics": True,
        }
        diagnostics = run_diagnostics
        resolved_config_sha256 = "injected_pair_metrics"

    seed_runs = build_seed_run_metrics(pair_frame)
    if diagnostics is None:
        diagnostics = _injected_run_diagnostics(seed_runs)
    if q3_pair_metrics is None:
        comparison = diagnostics.merge(
            seed_runs[
                [
                    "capacity_profile",
                    "seed",
                    "text_ablation_mode",
                    "tolerance_minutes",
                    "model_mae",
                    "persistence_mae",
                ]
            ],
            on=[
                "capacity_profile",
                "seed",
                "text_ablation_mode",
                "tolerance_minutes",
            ],
            suffixes=("_metadata", "_pairs"),
            validate="one_to_one",
        )
        for metric in ("model_mae", "persistence_mae"):
            if not np.allclose(
                comparison[f"{metric}_metadata"],
                comparison[f"{metric}_pairs"],
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise FixedLearningRateCapacityAnalysisError(
                    f"Metadata and Q3 pair metrics disagree on {metric}"
                )

    across = summarize_across_seeds(seed_runs)
    scores = summarize_capacity_scores(seed_runs)
    one_se = build_one_se_table(
        pair_frame,
        scores,
        iterations=int(bootstrap_iterations),
        random_seed=int(bootstrap_seed),
    )
    selection = build_selection_summary(
        scores,
        one_se,
        resolved_config_sha256=resolved_config_sha256,
    )
    persistence, text, capacity = build_bootstrap_tables(
        pair_frame,
        ranked_leader_profile=str(selection["ranked_leader_capacity_profile"]),
        iterations=int(bootstrap_iterations),
        random_seed=int(bootstrap_seed),
    )
    resources = summarize_resources(diagnostics)

    _write_csv(
        pair_frame,
        analysis_dir / "fixed_lr_capacity_q3_pair_metrics.csv.gz",
        compression="gzip",
    )
    _write_csv(seed_runs, analysis_dir / "fixed_lr_capacity_seed_run_metrics.csv")
    _write_csv(across, analysis_dir / "fixed_lr_capacity_across_seed_summary.csv")
    _write_csv(scores, analysis_dir / "fixed_lr_capacity_scores.csv")
    _write_csv(one_se, analysis_dir / "fixed_lr_capacity_one_se.csv")
    _write_csv(diagnostics, analysis_dir / "fixed_lr_capacity_run_diagnostics.csv")
    _write_csv(resources, analysis_dir / "fixed_lr_capacity_resource_summary.csv")
    _write_csv(
        persistence, analysis_dir / "fixed_lr_capacity_persistence_bootstrap.csv"
    )
    _write_csv(text, analysis_dir / "fixed_lr_capacity_text_bootstrap.csv")
    _write_csv(capacity, analysis_dir / "fixed_lr_capacity_pairwise_bootstrap.csv")
    _write_json(root / "fixed_lr_capacity_selection.json", selection)

    validation = {
        "schema_version": 1,
        "status": "pass",
        "experiment_kind": EXPERIMENT_KIND,
        "selection_panel": SELECTION_PANEL,
        "selection_mode": SELECTION_MODE,
        "q3_start_utc": Q3_START_UTC,
        "q3_end_utc_exclusive": Q3_END_UTC,
        "q4_used_for_selection": False,
        "q4_predictions_generated": False,
        "q4_rows_passed_to_evaluator": int(
            lineage.get("q4_rows_passed_to_evaluator", 0)
        ),
        "job_count": 72,
        "capacity_count": len(EXPECTED_PROFILES),
        "capacity_profiles": list(EXPECTED_PROFILES),
        "seed_count": len(EXPECTED_SEEDS),
        "seeds": list(EXPECTED_SEEDS),
        "text_modes": list(EXPECTED_TEXT_MODES),
        "tolerances_minutes": list(EXPECTED_TOLERANCES),
        "pair_count_per_run": int(seed_runs["pair_count"].iloc[0]),
        "session_count_per_run": int(seed_runs["session_count"].iloc[0]),
        "fixed_initial_learning_rate": FIXED_LEARNING_RATE,
        "fixed_scheduler_min_lr": FIXED_SCHEDULER_MIN_LR,
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_method": "seed_then_paired_CME_session_cluster",
        "one_se_selection_mode": SELECTION_MODE,
        "selection_payload_sha256": str(selection["selection_payload_sha256"]),
        "resolved_config_sha256": resolved_config_sha256,
        "lineage": lineage,
        "created_at_utc": _utc_now(),
    }
    validation_path = analysis_dir / "fixed_lr_capacity_validation_summary.json"
    _write_json(validation_path, validation)
    return validation_path


__all__ = [
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_BOOTSTRAP_SEED",
    "EXPECTED_PROFILES",
    "EXPECTED_SEEDS",
    "EXPECTED_TEXT_MODES",
    "EXPECTED_TOLERANCES",
    "FixedLearningRateCapacityAnalysisError",
    "build_bootstrap_tables",
    "build_one_se_table",
    "build_seed_run_metrics",
    "build_selection_summary",
    "collect_run_diagnostics",
    "evaluate_q3_pair_metrics",
    "run_fixed_lr_capacity_analysis",
    "summarize_across_seeds",
    "summarize_capacity_scores",
    "summarize_resources",
    "validate_job_matrix",
    "validate_pair_metric_matrix",
]
