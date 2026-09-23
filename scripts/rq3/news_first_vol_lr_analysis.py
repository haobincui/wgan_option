"""Q3-only learning-rate selection for the news-first vol experiment.

The module deliberately has no Q4 entry point.  It evaluates trained-epoch
(``best_learned``) checkpoints on the common 5-minute Q3 validation panel and
uses ``current_only`` runs for learning-rate selection.  ``real_text`` remains
an explicitly diagnostic lane so text effects cannot change the optimizer
hyper-parameter decision.
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
    compute_sample_metrics,
)
from scripts.rq3.news_first_vol_lr_sweep import (
    FROZEN_LEARNING_RATES,
    FROZEN_SCHEDULER_MIN_LRS,
    _lr_profile_sha256,
)


Q3_START_UTC = "2023-07-01T00:00:00Z"
Q3_END_UTC = "2023-10-01T00:00:00Z"
SCREEN_TOLERANCES = (5, 30)
ALL_TOLERANCES = (5, 10, 15, 30)
TEXT_ABLATION_MODES = ("current_only", "real_text")
SELECTION_MODE = "current_only"
SELECTION_RULE = "one_standard_error_lower_lr"
SELECTION_PANEL = "common_validation_05m"
SUPPORTED_STAGES = ("lr_screen", "lr_confirm")
CAPACITY_PROFILE = "large"
GATE_MIN_IMPROVEMENT = 0.005
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260819

# Public alias retained in this analysis namespace for fixture users.  The
# source of truth is the orchestrator module, including its canonical hash.
FROZEN_LR_PROFILES = FROZEN_LEARNING_RATES


class LearningRateAnalysisError(ValueError):
    """Raised when an LR decision cannot be reproduced safely."""


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
        raise LearningRateAnalysisError(f"Expected a JSON object: {path}")
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
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise LearningRateAnalysisError(f"Expected a YAML mapping: {path}")
    return dict(payload)


def _finite_positive(value: Any, *, label: str, path: Path | None = None) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    try:
        result = float(numeric)
    except (TypeError, ValueError):
        result = float("nan")
    if not math.isfinite(result) or result <= 0.0:
        location = f" in {path}" if path is not None else ""
        raise LearningRateAnalysisError(
            f"{label} must be finite and positive{location}"
        )
    return result


def _optional_finite(value: Any) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    try:
        result = float(numeric)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _same_float(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-15)


def _first_metric(payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    nested = payload.get("metrics")
    candidates: list[Mapping[str, Any]] = [payload]
    if isinstance(nested, Mapping):
        candidates.insert(0, nested)
    for source in candidates:
        for name in names:
            if name in source and source[name] not in (None, ""):
                return source[name]
    return None


def _selection_contract(root: Path) -> dict[str, Any]:
    """Load and fail-closed validate the frozen LR selection contract."""

    path = root / "resolved_config.yaml"
    payload = _load_yaml(path)
    training = payload.get("news_first_vol_training")
    if not isinstance(training, Mapping):
        raise LearningRateAnalysisError(
            f"resolved_config lacks news_first_vol_training: {path}"
        )
    sweep = training.get("lr_sweep")
    if not isinstance(sweep, Mapping) or not bool(sweep.get("enabled")):
        raise LearningRateAnalysisError(f"LR sweep is not enabled: {path}")
    selection = sweep.get("selection")
    if not isinstance(selection, Mapping):
        raise LearningRateAnalysisError(f"LR selection contract is absent: {path}")
    if str(selection.get("selection_rule", "")).strip() != SELECTION_RULE:
        raise LearningRateAnalysisError(
            f"selection_rule must be {SELECTION_RULE}: {path}"
        )
    if str(selection.get("screen_selection_mode", "")).strip() != SELECTION_MODE:
        raise LearningRateAnalysisError(
            f"screen_selection_mode must be {SELECTION_MODE}: {path}"
        )
    if str(selection.get("selection_panel", "")).strip() != SELECTION_PANEL:
        raise LearningRateAnalysisError(
            f"selection_panel must be {SELECTION_PANEL}: {path}"
        )
    minimum = _finite_positive(
        selection.get("minimum_mean_improvement_fraction"),
        label="minimum_mean_improvement_fraction",
        path=path,
    )
    if not _same_float(minimum, GATE_MIN_IMPROVEMENT):
        raise LearningRateAnalysisError(
            f"minimum improvement must remain {GATE_MIN_IMPROVEMENT}: {path}"
        )
    iterations = int(selection.get("bootstrap_clusters", 0))
    if iterations < 2:
        raise LearningRateAnalysisError(f"bootstrap_clusters must be >=2: {path}")
    capacity = str(sweep.get("capacity_profile", "")).strip().lower()
    if capacity != CAPACITY_PROFILE:
        raise LearningRateAnalysisError(
            f"LR sweep capacity_profile must be {CAPACITY_PROFILE}: {path}"
        )
    profiles = sweep.get("profiles")
    if not isinstance(profiles, Mapping):
        raise LearningRateAnalysisError(f"LR profiles are absent: {path}")
    observed_profiles: dict[str, float] = {}
    for raw_name, raw_rate in profiles.items():
        name = str(raw_name).strip()
        observed_profiles[name] = _finite_positive(
            raw_rate, label=f"learning rate for {name}", path=path
        )
    if set(observed_profiles) != set(FROZEN_LR_PROFILES):
        raise LearningRateAnalysisError(
            f"Frozen LR profile IDs drifted: {sorted(observed_profiles)}"
        )
    for name, expected in FROZEN_LR_PROFILES.items():
        if not _same_float(observed_profiles[name], expected):
            raise LearningRateAnalysisError(
                f"Frozen learning rate drift for {name}: {observed_profiles[name]}"
            )
    screen_tolerances = tuple(
        int(value) for value in sweep.get("screen_tolerances_minutes", [])
    )
    confirm_new = tuple(
        int(value) for value in sweep.get("confirm_tolerances_minutes", [])
    )
    modes = tuple(str(value).strip() for value in sweep.get("text_ablation_modes", []))
    if tuple(sorted(screen_tolerances)) != SCREEN_TOLERANCES:
        raise LearningRateAnalysisError(
            f"screen_tolerances_minutes must be {SCREEN_TOLERANCES}: {path}"
        )
    if tuple(sorted(confirm_new)) != (10, 15):
        raise LearningRateAnalysisError(
            f"confirm_tolerances_minutes must be (10, 15): {path}"
        )
    if set(modes) != set(TEXT_ABLATION_MODES):
        raise LearningRateAnalysisError(
            f"LR text modes must be {TEXT_ABLATION_MODES}: {path}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "profiles": observed_profiles,
        "bootstrap_iterations": iterations,
        "selection_rule": SELECTION_RULE,
        "selection_mode": SELECTION_MODE,
    }


def _load_jobs(root: Path) -> list[dict[str, Any]]:
    registry = _read_json(root / "registry" / "jobs.json")
    raw_jobs = registry.get("jobs")
    if not isinstance(raw_jobs, list):
        raise LearningRateAnalysisError("registry/jobs.json must contain a jobs list")
    jobs: list[dict[str, Any]] = []
    for raw in raw_jobs:
        if not isinstance(raw, Mapping):
            raise LearningRateAnalysisError("Every registry job must be an object")
        job = dict(raw)
        job_id = str(job.get("job_id", "")).strip()
        if not job_id:
            raise LearningRateAnalysisError("Every LR job requires job_id")
        status_path = root / "registry" / "jobs" / f"{job_id}.status.json"
        if status_path.is_file():
            job.update(_read_json(status_path))
        jobs.append(job)
    return jobs


def _job_stage(job: Mapping[str, Any]) -> str:
    return str(job.get("lr_stage", job.get("stage", ""))).strip().lower()


def _job_profile(job: Mapping[str, Any]) -> str:
    return str(job.get("lr_profile", "")).strip()


def _job_mode(job: Mapping[str, Any]) -> str:
    return str(job.get("text_ablation_mode", "")).strip().lower()


def _job_tolerance(job: Mapping[str, Any]) -> int:
    return int(job.get("tolerance_minutes", -1))


def _completed(job: Mapping[str, Any]) -> bool:
    return str(job.get("status", "")).strip().lower() == "completed"


def _run_dir(job: Mapping[str, Any]) -> Path:
    raw = str(job.get("run_dir", "")).strip()
    if not raw:
        raise LearningRateAnalysisError(
            f"Completed job lacks run_dir: {job.get('job_id')}"
        )
    path = Path(raw).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _training_config(job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    raw = str(job.get("training_config_path", "")).strip()
    if not raw:
        raise LearningRateAnalysisError(
            f"Job lacks training_config_path: {job.get('job_id')}"
        )
    path = Path(raw).resolve()
    return path, _load_yaml(path)


def _learned_metadata(job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _run_dir(job) / "metrics" / "best_learned_checkpoint.json"
    payload = _read_json(path)
    epoch = pd.to_numeric(payload.get("best_epoch"), errors="coerce")
    if not math.isfinite(float(epoch)) or int(epoch) < 1:
        raise LearningRateAnalysisError(
            f"best_learned_checkpoint must select epoch >=1: {path}"
        )
    if (
        str(payload.get("selection_scope", "trained_epochs_only"))
        != "trained_epochs_only"
    ):
        raise LearningRateAnalysisError(
            f"best_learned selection_scope must be trained_epochs_only: {path}"
        )
    return path, payload


def _baseline_metadata(job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _run_dir(job) / "metrics" / "best_checkpoint.json"
    payload = _read_json(path)
    epoch = pd.to_numeric(payload.get("best_epoch"), errors="coerce")
    if not math.isfinite(float(epoch)) or int(epoch) < 0:
        raise LearningRateAnalysisError(f"Invalid baseline best epoch: {path}")
    if str(payload.get("selection_scope", "")).strip() != "baseline_inclusive":
        raise LearningRateAnalysisError(
            f"best_checkpoint must be baseline-inclusive: {path}"
        )
    return path, payload


def _best_epoch_metrics(
    job: Mapping[str, Any],
    *,
    learned_epoch: int,
    learned_path: Path,
    learned: Mapping[str, Any],
) -> tuple[Path, float, float]:
    path = _run_dir(job) / "metrics" / "training_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise LearningRateAnalysisError(f"Expected a JSON list: {path}")
    matches = [
        row
        for row in payload
        if isinstance(row, Mapping)
        and pd.to_numeric(row.get("epoch"), errors="coerce") == learned_epoch
    ]
    if len(matches) != 1:
        raise LearningRateAnalysisError(
            f"Expected one metrics row for learned epoch {learned_epoch}: {path}"
        )
    row = matches[0]
    train_recon = _finite_positive(
        _first_metric(row, ("train_recon", "g_recon")),
        label="best-learned train reconstruction",
        path=path,
    )
    val_recon = _finite_positive(
        _first_metric(row, ("val_recon",)),
        label="best-learned validation reconstruction",
        path=path,
    )
    metadata_val = _finite_positive(
        _first_metric(learned, ("val_recon",)),
        label="best-learned metadata validation reconstruction",
        path=learned_path,
    )
    tolerance = max(1.0e-12, abs(metadata_val) * 1.0e-9)
    if not math.isclose(val_recon, metadata_val, rel_tol=0.0, abs_tol=tolerance):
        raise LearningRateAnalysisError(
            f"Training metrics disagree with learned metadata: {path}"
        )
    return path, train_recon, val_recon


def _resource_rows(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "resource_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    if "job_id" not in frame.columns:
        raise LearningRateAnalysisError(f"resource_summary lacks job_id: {path}")
    if frame["job_id"].astype(str).duplicated().any():
        raise LearningRateAnalysisError(f"Duplicate resource job IDs: {path}")
    return {str(row["job_id"]): dict(row) for row in frame.to_dict(orient="records")}


def _learned_checkpoint(job: Mapping[str, Any], payload: Mapping[str, Any]) -> Path:
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, Mapping):
        for key in ("model", "checkpoint"):
            raw = str(artifacts.get(key, "")).strip()
            if raw:
                candidate = Path(raw)
                if not candidate.is_absolute():
                    candidate = _run_dir(job) / candidate
                if candidate.is_file():
                    return candidate.resolve()
    candidate = _run_dir(job) / "checkpoints" / "vol_regressor_best_learned.pt"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate.resolve()


def _validate_lr_lineage(
    job: Mapping[str, Any],
    *,
    config_path: Path,
    config: Mapping[str, Any],
    learned_path: Path,
    learned: Mapping[str, Any],
) -> tuple[str, str, float, float]:
    profile = _job_profile(job)
    if profile not in FROZEN_LR_PROFILES:
        raise LearningRateAnalysisError(f"Unknown LR profile for {job.get('job_id')}")
    expected_lr = FROZEN_LR_PROFILES[profile]
    expected_min_lr = FROZEN_SCHEDULER_MIN_LRS[profile]
    registry_lr = _finite_positive(
        job.get("initial_learning_rate"), label="registry initial_learning_rate"
    )
    registry_min_lr = _finite_positive(
        job.get("scheduler_min_lr"), label="registry scheduler_min_lr"
    )
    if not _same_float(registry_lr, expected_lr):
        raise LearningRateAnalysisError(
            f"Registry LR disagrees with frozen profile {profile}"
        )
    if not _same_float(registry_min_lr, expected_min_lr):
        raise LearningRateAnalysisError(
            f"scheduler_min_lr must equal initial_learning_rate/10 for {profile}"
        )
    config_profile = str(config.get("news_first_lr_profile", "")).strip()
    if config_profile != profile:
        raise LearningRateAnalysisError(
            f"Registry/config LR profile mismatch: {config_path}"
        )
    registry_hash = str(job.get("lr_profile_sha256", "")).strip()
    config_hash = str(config.get("news_first_lr_profile_sha256", "")).strip()
    expected_hash = _lr_profile_sha256(profile)
    if (
        not registry_hash
        or not config_hash
        or registry_hash != config_hash
        or registry_hash != expected_hash
    ):
        raise LearningRateAnalysisError(
            f"Registry/config LR profile hash mismatch: {config_path}"
        )
    config_lr = _finite_positive(
        config.get("learning_rate"), label="training learning_rate", path=config_path
    )
    config_min_lr = _finite_positive(
        config.get("reduce_lr_min_lr"),
        label="training reduce_lr_min_lr",
        path=config_path,
    )
    if not _same_float(config_lr, expected_lr) or not _same_float(
        config_min_lr, expected_min_lr
    ):
        raise LearningRateAnalysisError(
            f"Training optimizer LR lineage mismatch for {profile}: {config_path}"
        )
    metadata_profile = str(learned.get("lr_profile", profile)).strip()
    metadata_hash = str(learned.get("lr_profile_sha256", registry_hash)).strip()
    if metadata_profile != profile or metadata_hash != registry_hash:
        raise LearningRateAnalysisError(
            f"Learned checkpoint LR lineage mismatch: {learned_path}"
        )
    for field, expected in (
        ("initial_learning_rate", expected_lr),
        ("scheduler_min_lr", expected_min_lr),
    ):
        if field in learned and not _same_float(
            _finite_positive(learned[field], label=field, path=learned_path), expected
        ):
            raise LearningRateAnalysisError(
                f"Learned checkpoint {field} mismatch: {learned_path}"
            )
    return profile, registry_hash, expected_lr, expected_min_lr


def collect_lr_comparisons(
    experiment_root: str | Path,
    *,
    jobs: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Collect completed best-learned Q3 metrics and optimizer lineage."""

    root = Path(experiment_root).resolve()
    _selection_contract(root)
    resources = _resource_rows(root)
    rows: list[dict[str, Any]] = []
    source_jobs = list(jobs) if jobs is not None else _load_jobs(root)
    for job in source_jobs:
        if _job_stage(job) not in SUPPORTED_STAGES or not _completed(job):
            continue
        if str(job.get("model_family", "")).strip().lower() != "regression":
            raise LearningRateAnalysisError("LR sweep accepts regression jobs only")
        if str(job.get("capacity_profile", "")).strip().lower() != CAPACITY_PROFILE:
            raise LearningRateAnalysisError(
                f"LR sweep requires capacity_profile={CAPACITY_PROFILE}"
            )
        mode = _job_mode(job)
        if mode not in TEXT_ABLATION_MODES:
            raise LearningRateAnalysisError(f"Unexpected LR text mode: {mode}")
        config_path, config = _training_config(job)
        if str(config.get("support_mask_mode", "")).strip().lower() != "raw_joint":
            raise LearningRateAnalysisError(
                f"LR selection requires support_mask_mode=raw_joint: {config_path}"
            )
        if str(config.get("news_first_train_end_utc", Q3_START_UTC)) != Q3_START_UTC:
            raise LearningRateAnalysisError(f"Q3 train boundary drift: {config_path}")
        if str(config.get("news_first_validation_end_utc", Q3_END_UTC)) != Q3_END_UTC:
            raise LearningRateAnalysisError(
                f"Q3 validation boundary drift: {config_path}"
            )
        learned_path, learned = _learned_metadata(job)
        profile, profile_hash, initial_lr, min_lr = _validate_lr_lineage(
            job,
            config_path=config_path,
            config=config,
            learned_path=learned_path,
            learned=learned,
        )
        model_mae = _finite_positive(
            _first_metric(
                learned,
                (
                    "masked_pair_balanced_mae",
                    "best_learned_masked_pair_balanced_mae",
                    "val_recon",
                ),
            ),
            label="best-learned masked pair-balanced MAE",
            path=learned_path,
        )
        persistence_mae = _finite_positive(
            _first_metric(
                learned,
                (
                    "persistence_masked_pair_balanced_mae",
                    "masked_pair_balanced_persistence_mae",
                    "val_current_recon",
                ),
            ),
            label="persistence masked pair-balanced MAE",
            path=learned_path,
        )
        learned_epoch = int(learned["best_epoch"])
        metrics_path, train_recon, val_recon = _best_epoch_metrics(
            job,
            learned_epoch=learned_epoch,
            learned_path=learned_path,
            learned=learned,
        )
        baseline_path, baseline = _baseline_metadata(job)
        job_id = str(job["job_id"])
        if job_id not in resources:
            raise LearningRateAnalysisError(
                f"resource_summary lacks completed LR job: {job_id}"
            )
        resource = resources[job_id]
        ratio = model_mae / persistence_mae
        rows.append(
            {
                "job_id": job_id,
                "lr_stage": _job_stage(job),
                "model_family": "regression",
                "capacity_profile": CAPACITY_PROFILE,
                "capacity_profile_sha256": str(job.get("capacity_profile_sha256", "")),
                "lr_profile": profile,
                "lr_profile_sha256": profile_hash,
                "initial_learning_rate": initial_lr,
                "scheduler_min_lr": min_lr,
                "learning_rate_log10": math.log10(initial_lr),
                "text_ablation_mode": mode,
                "tolerance_minutes": _job_tolerance(job),
                "best_learned_epoch": learned_epoch,
                "best_learned_train_recon": train_recon,
                "best_learned_val_recon": val_recon,
                "best_learned_train_val_gap": val_recon - train_recon,
                "baseline_inclusive_best_epoch": int(baseline["best_epoch"]),
                "baseline_inclusive_epoch0_selected": int(baseline["best_epoch"]) == 0,
                "model_mae": model_mae,
                "persistence_mae": persistence_mae,
                "mae_ratio": ratio,
                "log_mae_ratio": math.log(ratio),
                "improvement_fraction": 1.0 - ratio,
                "runtime_minutes": _optional_finite(resource.get("runtime_minutes")),
                "peak_memory_mib": _optional_finite(resource.get("peak_memory_mib")),
                "mean_utilization_gpu_pct": _optional_finite(
                    resource.get("mean_utilization_gpu_pct")
                ),
                "learned_metadata_path": str(learned_path),
                "learned_metadata_sha256": _sha256(learned_path),
                "baseline_metadata_path": str(baseline_path),
                "baseline_metadata_sha256": _sha256(baseline_path),
                "training_metrics_path": str(metrics_path),
                "training_metrics_sha256": _sha256(metrics_path),
                "training_config_path": str(config_path),
                "run_dir": str(_run_dir(job)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    keys = ["lr_profile", "text_ablation_mode", "tolerance_minutes"]
    if frame.duplicated(keys).any():
        bad = frame.loc[frame.duplicated(keys, keep=False), keys]
        raise LearningRateAnalysisError(
            f"Duplicate completed LR jobs: {bad.to_dict(orient='records')[:5]}"
        )
    return frame.sort_values(
        ["initial_learning_rate", "text_ablation_mode", "tolerance_minutes"],
        kind="stable",
    ).reset_index(drop=True)


def build_lr_profile_manifest(
    experiment_root: str | Path,
    *,
    jobs: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build one auditable row per frozen LR profile."""

    root = Path(experiment_root).resolve()
    contract = _selection_contract(root)
    source_jobs = list(jobs) if jobs is not None else _load_jobs(root)
    manifest_path = root / "lr_profile_manifest.csv"
    if manifest_path.is_file():
        # The orchestrator owns the canonical profile manifest (including
        # model-shape and scheduler fields).  Analysis validates it in place
        # rather than replacing that immutable lineage artifact.
        existing = pd.read_csv(manifest_path, low_memory=False)
        required = {
            "lr_profile",
            "lr_profile_sha256",
            "initial_learning_rate",
            "scheduler_min_lr",
            "capacity_profile",
        }
        missing = sorted(required - set(existing.columns))
        if missing:
            raise LearningRateAnalysisError(
                f"Canonical LR manifest is missing columns: {missing}"
            )
        if existing["lr_profile"].astype(str).duplicated().any() or set(
            existing["lr_profile"].astype(str)
        ) != set(FROZEN_LR_PROFILES):
            raise LearningRateAnalysisError("Canonical LR manifest profile IDs drifted")
        for row in existing.to_dict(orient="records"):
            profile = str(row["lr_profile"])
            if (
                str(row["lr_profile_sha256"]) != _lr_profile_sha256(profile)
                or not _same_float(
                    _finite_positive(
                        row["initial_learning_rate"],
                        label="manifest initial_learning_rate",
                        path=manifest_path,
                    ),
                    FROZEN_LR_PROFILES[profile],
                )
                or not _same_float(
                    _finite_positive(
                        row["scheduler_min_lr"],
                        label="manifest scheduler_min_lr",
                        path=manifest_path,
                    ),
                    FROZEN_SCHEDULER_MIN_LRS[profile],
                )
                or str(row["capacity_profile"]).strip().lower() != CAPACITY_PROFILE
            ):
                raise LearningRateAnalysisError(
                    f"Canonical LR manifest lineage mismatch for {profile}"
                )
        return existing.sort_values("initial_learning_rate", kind="stable").reset_index(
            drop=True
        )
    rows: list[dict[str, Any]] = []
    for profile, initial_lr in sorted(
        FROZEN_LR_PROFILES.items(), key=lambda item: item[1]
    ):
        matching = [job for job in source_jobs if _job_profile(job) == profile]
        hashes = {
            str(job.get("lr_profile_sha256", "")).strip()
            for job in matching
            if str(job.get("lr_profile_sha256", "")).strip()
        }
        if len(hashes) != 1:
            raise LearningRateAnalysisError(
                f"LR profile hash must be unique for {profile}: {sorted(hashes)}"
            )
        observed_rates = {
            _finite_positive(
                job.get("initial_learning_rate"), label="initial_learning_rate"
            )
            for job in matching
        }
        observed_min = {
            _finite_positive(job.get("scheduler_min_lr"), label="scheduler_min_lr")
            for job in matching
        }
        expected_min_lr = FROZEN_SCHEDULER_MIN_LRS[profile]
        if observed_rates != {initial_lr} or observed_min != {expected_min_lr}:
            raise LearningRateAnalysisError(
                f"Optimizer lineage is inconsistent for {profile}"
            )
        row = {
            "lr_profile": profile,
            "lr_profile_sha256": next(iter(hashes)),
            "initial_learning_rate": initial_lr,
            "scheduler_min_lr": expected_min_lr,
            "capacity_profile": CAPACITY_PROFILE,
            "screen_job_count": sum(_job_stage(job) == "lr_screen" for job in matching),
            "screen_completed_job_count": sum(
                _job_stage(job) == "lr_screen" and _completed(job) for job in matching
            ),
            "confirm_job_count": sum(
                _job_stage(job) == "lr_confirm" for job in matching
            ),
            "confirm_completed_job_count": sum(
                _job_stage(job) == "lr_confirm" and _completed(job) for job in matching
            ),
            "selection_rule": contract["selection_rule"],
            "selection_mode": contract["selection_mode"],
        }
        row["profile_manifest_row_sha256"] = _payload_sha256(row)
        rows.append(row)
    frame = pd.DataFrame(rows)
    _write_csv(frame, manifest_path)
    return frame


def summarize_lr_scores(
    comparisons: pd.DataFrame,
    *,
    tolerances: Sequence[int],
    profiles: Sequence[str],
    selection_mode: str = SELECTION_MODE,
) -> pd.DataFrame:
    """Aggregate equal-tolerance log MAE ratios and evaluate the frozen gate."""

    expected_tolerances = tuple(int(value) for value in tolerances)
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        group = comparisons[
            comparisons["lr_profile"].astype(str).eq(str(profile))
            & comparisons["text_ablation_mode"].astype(str).eq(selection_mode)
            & comparisons["tolerance_minutes"].astype(int).isin(expected_tolerances)
        ].copy()
        observed = tuple(sorted(group["tolerance_minutes"].astype(int).tolist()))
        if observed != tuple(sorted(expected_tolerances)):
            raise LearningRateAnalysisError(
                f"Incomplete selection tolerances for {profile}: {observed}"
            )
        score = float(group["log_mae_ratio"].mean())
        ratio = math.exp(score)
        all_not_worse = bool((group["mae_ratio"] <= 1.0 + 1.0e-15).all())
        mean_improvement = 1.0 - ratio
        average_gate = mean_improvement >= GATE_MIN_IMPROVEMENT - 1.0e-15
        rows.append(
            {
                "lr_profile": str(profile),
                "initial_learning_rate": FROZEN_LR_PROFILES[str(profile)],
                "selection_mode": selection_mode,
                "selection_tolerances": ",".join(
                    str(value) for value in expected_tolerances
                ),
                "mean_log_mae_ratio": score,
                "geometric_mae_ratio": ratio,
                "mean_improvement_fraction": mean_improvement,
                "all_tolerances_not_worse": all_not_worse,
                "average_improvement_gate_passed": average_gate,
                "lr_gate_passed": all_not_worse and average_gate,
            }
        )
    output = pd.DataFrame(rows)
    return output.sort_values(
        ["mean_log_mae_ratio", "initial_learning_rate", "lr_profile"],
        kind="stable",
    ).reset_index(drop=True)


def _with_text_diagnostics(
    summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    tolerances: Sequence[int],
) -> pd.DataFrame:
    diagnostics: dict[str, dict[str, float]] = {}
    for profile in summaries["lr_profile"].astype(str):
        group = comparisons[
            comparisons["lr_profile"].astype(str).eq(profile)
            & comparisons["text_ablation_mode"].astype(str).eq("real_text")
            & comparisons["tolerance_minutes"].astype(int).isin(tolerances)
        ]
        observed = tuple(sorted(group["tolerance_minutes"].astype(int).tolist()))
        if observed != tuple(sorted(int(value) for value in tolerances)):
            raise LearningRateAnalysisError(
                f"Incomplete real_text diagnostics for {profile}: {observed}"
            )
        score = float(group["log_mae_ratio"].mean())
        diagnostics[profile] = {
            "real_text_mean_log_mae_ratio": score,
            "real_text_geometric_mae_ratio": math.exp(score),
        }
    output = summaries.copy()
    output["real_text_mean_log_mae_ratio"] = output["lr_profile"].map(
        {
            key: value["real_text_mean_log_mae_ratio"]
            for key, value in diagnostics.items()
        }
    )
    output["real_text_geometric_mae_ratio"] = output["lr_profile"].map(
        {
            key: value["real_text_geometric_mae_ratio"]
            for key, value in diagnostics.items()
        }
    )
    output["real_text_minus_current_log_ratio"] = (
        output["real_text_mean_log_mae_ratio"] - output["mean_log_mae_ratio"]
    )
    return output


def _stage_rows(
    comparisons: pd.DataFrame,
    *,
    profiles: Sequence[str],
    tolerances: Sequence[int],
) -> pd.DataFrame:
    rows = comparisons[
        comparisons["lr_profile"].isin(list(profiles))
        & comparisons["text_ablation_mode"].isin(TEXT_ABLATION_MODES)
        & comparisons["tolerance_minutes"].isin(list(tolerances))
    ].copy()
    expected = {
        (str(profile), mode, int(tolerance))
        for profile in profiles
        for mode in TEXT_ABLATION_MODES
        for tolerance in tolerances
    }
    observed = set(
        rows[["lr_profile", "text_ablation_mode", "tolerance_minutes"]].itertuples(
            index=False, name=None
        )
    )
    if observed != expected:
        raise LearningRateAnalysisError(
            "LR stage matrix incomplete; "
            f"missing={sorted(expected - observed)[:8]}, "
            f"extra={sorted(observed - expected)[:8]}"
        )
    return rows


def _run_spec(job: Mapping[str, Any]) -> RunSpec:
    learned_path, learned = _learned_metadata(job)
    config_path, config = _training_config(job)
    return RunSpec(
        run_id=str(job["job_id"]),
        run_dir=_run_dir(job),
        model="regression",
        tolerance_minutes=_job_tolerance(job),
        seed=int(config.get("seed", 42)),
        checkpoint_path=_learned_checkpoint(job, learned),
        text_ablation_mode=_job_mode(job),
        support_mask_mode=str(config.get("support_mask_mode", "none")),
        manifest_path=config_path,
        metadata={
            **config,
            "lr_profile": _job_profile(job),
            "lr_profile_sha256": str(job.get("lr_profile_sha256", "")),
            "best_learned_metadata_path": str(learned_path),
        },
    )


def _q3_panel_for_jobs(
    jobs: Sequence[Mapping[str, Any]],
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
        raise LearningRateAnalysisError(
            f"LR jobs disagree on common Q3 panel: {sorted(paths)}"
        )
    raw_path, sheet_name, support_mode = next(iter(paths))
    workbook = Path(raw_path).resolve()
    if not workbook.is_file():
        raise FileNotFoundError(workbook)
    raw = pd.read_excel(workbook, sheet_name=sheet_name)
    if "effective_origin_utc" not in raw.columns:
        raise LearningRateAnalysisError(
            f"Q3 panel lacks effective_origin_utc: {workbook}"
        )
    times = pd.to_datetime(raw["effective_origin_utc"], errors="coerce", utc=True)
    if times.isna().any():
        raise LearningRateAnalysisError(f"Unparseable Q3 timestamp: {workbook}")
    start = pd.Timestamp(Q3_START_UTC)
    end = pd.Timestamp(Q3_END_UTC)
    q3 = raw.loc[(times >= start) & (times < end)].copy()
    if q3.empty:
        raise LearningRateAnalysisError("Common Q3 panel is empty")
    panel, lineage, _ = _load_panel_source(
        q3,
        sheet_name=sheet_name,
        panel_name=SELECTION_PANEL,
        freeze_test_window=False,
        enforce_expected_counts=False,
        support_mask_mode=support_mode,
    )
    selected_times = pd.to_datetime(panel["effective_origin_utc"], utc=True)
    if bool((selected_times < start).any() or (selected_times >= end).any()):
        raise LearningRateAnalysisError("A Q4 row reached the LR selection panel")
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
    jobs: Sequence[Mapping[str, Any]],
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate best-learned current-only checkpoints on Q3, never Q4."""

    if not jobs:
        raise LearningRateAnalysisError("No jobs supplied for Q3 LR evaluation")
    if any(_job_mode(job) != SELECTION_MODE for job in jobs):
        raise LearningRateAnalysisError(
            "LR selection evaluator accepts current_only only"
        )
    panel, lineage = _q3_panel_for_jobs(jobs)
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
            raise LearningRateAnalysisError(
                f"Q3 exclusions for {spec.run_id}: "
                f"{exclusions['exclusion_code'].value_counts().to_dict()}"
            )
        pairs = aggregate_pair_metrics(samples)
        pairs = pairs[
            pairs["stratum_type"].astype(str).eq("overall")
            & pairs["stratum_value"].astype(str).eq("all")
        ].copy()
        learned_path, learned = _learned_metadata(job)
        expected_model = _finite_positive(
            _first_metric(learned, ("masked_pair_balanced_mae", "val_recon")),
            label="learned validation MAE",
            path=learned_path,
        )
        expected_persistence = _finite_positive(
            _first_metric(
                learned,
                ("persistence_masked_pair_balanced_mae", "val_current_recon"),
            ),
            label="validation persistence MAE",
            path=learned_path,
        )
        for label, actual, expected in (
            ("model", float(pairs["model_mae"].mean()), expected_model),
            (
                "persistence",
                float(pairs["persistence_mae"].mean()),
                expected_persistence,
            ),
        ):
            tolerance = max(1.0e-7, abs(expected) * 1.0e-4)
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
                raise LearningRateAnalysisError(
                    f"Q3 pair-balanced {label} MAE disagrees with metadata for "
                    f"{spec.run_id}: {actual} != {expected}"
                )
        pairs.insert(0, "lr_profile", _job_profile(job))
        pairs.insert(1, "lr_profile_sha256", str(job.get("lr_profile_sha256", "")))
        pairs.insert(2, "initial_learning_rate", FROZEN_LR_PROFILES[_job_profile(job)])
        parts.append(pairs)
    output = pd.concat(parts, ignore_index=True)
    lineage["selection_scope_verified"] = True
    return output, lineage


def one_standard_error_lr_selection(
    pair_metrics: pd.DataFrame,
    *,
    profiles: Sequence[str],
    tolerances: Sequence[int],
    gate_passed_profiles: Sequence[str],
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired session bootstrap and lower-LR one-standard-error rule."""

    if int(iterations) < 2:
        raise LearningRateAnalysisError("one-SE bootstrap requires at least two draws")
    profiles = tuple(str(value) for value in profiles)
    tolerances = tuple(int(value) for value in tolerances)
    selected = pair_metrics[
        pair_metrics["lr_profile"].astype(str).isin(profiles)
        & pair_metrics["model"].astype(str).eq("regression")
        & pair_metrics["text_ablation_mode"].astype(str).eq(SELECTION_MODE)
        & pair_metrics["tolerance_minutes"].astype(int).isin(tolerances)
    ].copy()
    required = {
        "lr_profile",
        "tolerance_minutes",
        "pair_id",
        "session_id",
        "model_mae",
        "persistence_mae",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise LearningRateAnalysisError(f"one-SE pair metrics missing: {missing}")
    key_sets: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for (profile, tolerance), group in selected.groupby(
        ["lr_profile", "tolerance_minutes"], sort=False
    ):
        key_sets[(str(profile), int(tolerance))] = set(
            group[["pair_id", "session_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
    expected_keys = {
        (profile, tolerance) for profile in profiles for tolerance in tolerances
    }
    if (
        set(key_sets) != expected_keys
        or len({frozenset(v) for v in key_sets.values()}) != 1
    ):
        raise LearningRateAnalysisError(
            "one-SE LR comparison requires identical pair/session coverage"
        )
    sessions = sorted(selected["session_id"].astype(str).unique())
    if len(sessions) < 2:
        raise LearningRateAnalysisError("one-SE LR comparison needs >=2 CME sessions")
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, len(sessions), size=(int(iterations), len(sessions)))
    point_scores: dict[str, float] = {}
    draws: dict[str, np.ndarray] = {}
    for profile in profiles:
        point_terms: list[float] = []
        draw_terms: list[np.ndarray] = []
        profile_rows = selected[selected["lr_profile"].astype(str).eq(profile)]
        for tolerance in tolerances:
            group = profile_rows[
                profile_rows["tolerance_minutes"].astype(int).eq(tolerance)
            ]
            by_session = group.groupby("session_id", sort=True).agg(
                model_sum=("model_mae", "sum"),
                persistence_sum=("persistence_mae", "sum"),
                pair_count=("pair_id", "size"),
            )
            by_session = by_session.reindex(sessions)
            if by_session.isna().any().any():
                raise LearningRateAnalysisError(
                    f"Missing Q3 session for {profile}/{tolerance}m"
                )
            model_sums = by_session["model_sum"].to_numpy(dtype=float)
            persistence_sums = by_session["persistence_sum"].to_numpy(dtype=float)
            point_ratio = model_sums.sum() / persistence_sums.sum()
            if not math.isfinite(point_ratio) or point_ratio <= 0.0:
                raise LearningRateAnalysisError("Invalid point MAE ratio")
            point_terms.append(math.log(point_ratio))
            draw_model = model_sums[sampled].sum(axis=1)
            draw_persistence = persistence_sums[sampled].sum(axis=1)
            ratios = draw_model / draw_persistence
            if not np.isfinite(ratios).all() or bool((ratios <= 0.0).any()):
                raise LearningRateAnalysisError("Invalid bootstrap MAE ratio")
            draw_terms.append(np.log(ratios))
        point_scores[profile] = float(np.mean(point_terms))
        draws[profile] = np.mean(np.stack(draw_terms, axis=0), axis=0)
    standard_errors = {
        profile: float(values.std(ddof=1)) for profile, values in draws.items()
    }
    ranked_leader = min(
        profiles,
        key=lambda profile: (
            point_scores[profile],
            FROZEN_LR_PROFILES[profile],
            profile,
        ),
    )
    passed = [profile for profile in profiles if profile in set(gate_passed_profiles)]
    if passed:
        best = min(
            passed,
            key=lambda profile: (
                point_scores[profile],
                FROZEN_LR_PROFILES[profile],
                profile,
            ),
        )
        threshold = point_scores[best] + standard_errors[best]
        eligible = [
            profile
            for profile in passed
            if point_scores[profile] <= threshold + 1.0e-15
        ]
        winner = min(
            eligible,
            key=lambda profile: (
                FROZEN_LR_PROFILES[profile],
                point_scores[profile],
                profile,
            ),
        )
    else:
        best = ranked_leader
        threshold = point_scores[best] + standard_errors[best]
        eligible = []
        winner = ""
    return {
        "method": "paired_CME_session_cluster_one_standard_error_lower_lr",
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
        "ranked_leader_lr_profile": ranked_leader,
        "best_gate_passing_lr_profile": best if passed else "",
        "best_score": point_scores[best],
        "best_score_standard_error": standard_errors[best],
        "one_se_threshold": threshold,
        "one_se_eligible_lr_profiles": sorted(
            eligible, key=lambda profile: FROZEN_LR_PROFILES[profile]
        ),
        "selected_lr_profile": winner,
        "profile_scores": point_scores,
        "profile_standard_errors": standard_errors,
        "session_count": len(sessions),
    }


def _selection_document(root: Path) -> dict[str, Any]:
    path = root / "lr_selection.json"
    if not path.is_file():
        return {
            "schema_version": 1,
            "experiment_kind": "learning_rate_sweep",
            "selection_data_scope": "Q3 common_validation_05m only",
            "q3_start_utc": Q3_START_UTC,
            "q3_end_utc_exclusive": Q3_END_UTC,
            "q4_used_for_selection": False,
            "stages": {},
        }
    payload = _read_json(path)
    if bool(payload.get("q4_used_for_selection", False)):
        raise LearningRateAnalysisError("LR selection file claims Q4 access")
    if (
        str(payload.get("experiment_kind", "learning_rate_sweep"))
        != "learning_rate_sweep"
    ):
        raise LearningRateAnalysisError("lr_selection experiment_kind mismatch")
    return payload


def _prior_screen(root: Path) -> dict[str, Any]:
    document = _selection_document(root)
    stages = document.get("stages")
    raw = stages.get("lr_screen") if isinstance(stages, Mapping) else None
    if not isinstance(raw, Mapping):
        raise LearningRateAnalysisError("lr_confirm requires frozen lr_screen")
    prior = dict(raw)
    if not bool(prior.get("gate_passed")):
        raise LearningRateAnalysisError("lr_screen gate failed; confirm is forbidden")
    selected = str(prior.get("selected_lr_profile", "")).strip()
    if selected not in FROZEN_LR_PROFILES:
        raise LearningRateAnalysisError("lr_screen lacks a valid selected LR")
    return prior


def _selection_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
    tolerances: Sequence[int],
) -> list[dict[str, Any]]:
    selected = [
        dict(job)
        for job in jobs
        if _completed(job)
        and _job_profile(job) in set(profiles)
        and _job_mode(job) == SELECTION_MODE
        and _job_tolerance(job) in set(int(value) for value in tolerances)
    ]
    expected = {
        (str(profile), int(tolerance))
        for profile in profiles
        for tolerance in tolerances
    }
    observed = {(_job_profile(job), _job_tolerance(job)) for job in selected}
    if observed != expected:
        raise LearningRateAnalysisError(
            f"Selection checkpoint matrix incomplete: missing={sorted(expected-observed)}"
        )
    return selected


def _decorate_comparisons(
    rows: pd.DataFrame,
    summaries: pd.DataFrame,
    *,
    stage: str,
    selected: str,
    ranked_leader: str,
    one_se: Mapping[str, Any],
) -> pd.DataFrame:
    output = rows.copy()
    output.insert(0, "selection_stage", stage)
    summary_map = summaries.set_index("lr_profile").to_dict(orient="index")
    for field in (
        "mean_log_mae_ratio",
        "geometric_mae_ratio",
        "mean_improvement_fraction",
        "all_tolerances_not_worse",
        "average_improvement_gate_passed",
        "lr_gate_passed",
        "real_text_mean_log_mae_ratio",
        "real_text_geometric_mae_ratio",
        "real_text_minus_current_log_ratio",
    ):
        output[field] = output["lr_profile"].map(
            {profile: values[field] for profile, values in summary_map.items()}
        )
    rank = {
        profile: index + 1
        for index, profile in enumerate(summaries["lr_profile"].astype(str))
    }
    output["score_rank"] = output["lr_profile"].map(rank)
    output["used_for_selection"] = output["text_ablation_mode"].eq(SELECTION_MODE)
    output["one_se_eligible"] = output["lr_profile"].isin(
        one_se.get("one_se_eligible_lr_profiles", [])
    )
    output["selected_lr_profile"] = selected
    output["selected_lr"] = output["lr_profile"].eq(selected) if selected else False
    output["ranked_leader_lr_profile"] = ranked_leader
    return output


def evaluate_lr_stage(
    experiment_root: str | Path,
    stage: str,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
    q3_pair_metrics: pd.DataFrame | None = None,
    bootstrap_iterations: int | None = None,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate ``lr_screen`` or ``lr_confirm`` using Q3 only."""

    normalized = str(stage).strip().lower()
    if normalized not in SUPPORTED_STAGES:
        raise LearningRateAnalysisError(
            f"stage must be one of {list(SUPPORTED_STAGES)}, got {stage!r}"
        )
    root = Path(experiment_root).resolve()
    contract = _selection_contract(root)
    iterations = (
        int(contract["bootstrap_iterations"])
        if bootstrap_iterations is None
        else int(bootstrap_iterations)
    )
    if iterations < 2:
        raise LearningRateAnalysisError("bootstrap_iterations must be >=2")
    jobs = _load_jobs(root)
    build_lr_profile_manifest(root, jobs=jobs)
    comparisons = collect_lr_comparisons(root, jobs=jobs)
    if normalized == "lr_screen":
        profiles = tuple(
            name
            for name, _ in sorted(FROZEN_LR_PROFILES.items(), key=lambda item: item[1])
        )
        tolerances = SCREEN_TOLERANCES
        provisional = True
        screen_selected = ""
    else:
        prior = _prior_screen(root)
        screen_selected = str(prior["selected_lr_profile"])
        profiles = (screen_selected,)
        tolerances = ALL_TOLERANCES
        provisional = False
    stage_rows = _stage_rows(comparisons, profiles=profiles, tolerances=tolerances)
    summaries = summarize_lr_scores(
        stage_rows, tolerances=tolerances, profiles=profiles
    )
    summaries = _with_text_diagnostics(summaries, stage_rows, tolerances=tolerances)
    gate_profiles = (
        summaries.loc[summaries["lr_gate_passed"].astype(bool), "lr_profile"]
        .astype(str)
        .tolist()
    )
    selection_jobs = _selection_jobs(jobs, profiles=profiles, tolerances=tolerances)
    lineage: dict[str, Any] = {
        "selection_panel": SELECTION_PANEL,
        "q3_start_utc": Q3_START_UTC,
        "q3_end_utc_exclusive": Q3_END_UTC,
        "q4_rows_passed_to_evaluator": 0,
    }
    if q3_pair_metrics is None:
        q3_pair_metrics, lineage = evaluate_q3_pair_metrics(
            selection_jobs, evaluator=evaluator
        )
    one_se = one_standard_error_lr_selection(
        q3_pair_metrics,
        profiles=profiles,
        tolerances=tolerances,
        gate_passed_profiles=gate_profiles,
        iterations=iterations,
        seed=int(bootstrap_seed),
    )
    selected = str(one_se["selected_lr_profile"])
    gate_passed = bool(selected)
    ranked_leader = str(one_se["ranked_leader_lr_profile"])
    if normalized == "lr_confirm" and gate_passed and selected != screen_selected:
        raise LearningRateAnalysisError("lr_confirm cannot change the frozen screen LR")
    selected_rows = stage_rows[stage_rows["lr_profile"].astype(str).eq(selected)]
    selected_hashes = sorted(
        {
            str(value)
            for value in selected_rows.get("lr_profile_sha256", pd.Series(dtype=str))
            if str(value)
        }
    )
    if gate_passed and len(selected_hashes) != 1:
        raise LearningRateAnalysisError("Selected LR profile hash is not unique")
    selected_hash = selected_hashes[0] if gate_passed else ""
    selected_lr = FROZEN_LR_PROFILES[selected] if gate_passed else None
    selected_min_lr = FROZEN_SCHEDULER_MIN_LRS[selected] if selected else None
    formal_winner = selected if gate_passed and not provisional else ""
    ranked_order = summaries["lr_profile"].astype(str).tolist()
    decorated = _decorate_comparisons(
        stage_rows,
        summaries,
        stage=normalized,
        selected=selected,
        ranked_leader=ranked_leader,
        one_se=one_se,
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_kind": "learning_rate_sweep",
        "lr_stage": normalized,
        "capacity_profile": CAPACITY_PROFILE,
        "selection_panel": SELECTION_PANEL,
        "selection_rule": SELECTION_RULE,
        "screen_selection_mode": SELECTION_MODE,
        "selection_text_ablation_mode": SELECTION_MODE,
        "selection_tolerances": [int(value) for value in tolerances],
        "score_definition": (
            "equal-tolerance mean log(best_learned masked pair-balanced MAE / "
            "persistence masked pair-balanced MAE)"
        ),
        "real_text_role": "diagnostic_only_not_used_for_lr_selection",
        "gate_min_mean_improvement_fraction": GATE_MIN_IMPROVEMENT,
        "gate_requires_every_selection_tolerance_not_worse": True,
        "gate_passed": gate_passed,
        "selection_is_provisional": provisional,
        "selected_lr_profile": selected,
        "selected_lr_profiles": [selected] if selected else [],
        "selected_lr_profile_sha256": selected_hash,
        "selected_initial_learning_rate": selected_lr,
        "selected_scheduler_min_lr": selected_min_lr,
        "winner_lr_profile": formal_winner,
        "winner_profile": formal_winner,
        "lr_profile": formal_winner,
        # These public lineage fields describe the selected candidate at
        # screen and the formal winner at confirm.  Formal winner identity is
        # still represented only by winner_lr_profile/lr_profile.
        "lr_profile_sha256": selected_hash,
        "initial_learning_rate": selected_lr,
        "scheduler_min_lr": selected_min_lr,
        "ranked_leader_lr_profile": ranked_leader,
        "ranked_lr_profiles": ranked_order,
        "screen_selected_lr_profile": screen_selected,
        "candidate_lr_profiles": summaries.to_dict(orient="records"),
        "one_standard_error_selection": one_se,
        "q3_lineage": lineage,
        "q4_used_for_selection": False,
        "resolved_config_sha256": contract["sha256"],
    }
    result["selection_payload_sha256"] = _payload_sha256(result)

    document = _selection_document(root)
    stages = dict(document.get("stages", {}))
    stages[normalized] = deepcopy(result)
    document["stages"] = stages
    document["updated_at_utc"] = _utc_now()
    _write_json(root / "lr_selection.json", document)

    comparison_path = root / "lr_comparisons.csv"
    if comparison_path.is_file():
        previous = pd.read_csv(comparison_path, low_memory=False)
        if "selection_stage" in previous.columns:
            previous = previous[~previous["selection_stage"].astype(str).eq(normalized)]
        decorated = pd.concat([previous, decorated], ignore_index=True, sort=False)
    _write_csv(decorated, comparison_path)

    audit_row = {
        "lr_stage": normalized,
        "evaluated_at_utc": _utc_now(),
        "candidate_lr_count": len(summaries),
        "gate_passing_lr_count": len(gate_profiles),
        "gate_passed": gate_passed,
        "selection_is_provisional": provisional,
        "selected_lr_profile": selected,
        "winner_lr_profile": formal_winner,
        "ranked_leader_lr_profile": ranked_leader,
        "selection_panel": SELECTION_PANEL,
        "selection_mode": SELECTION_MODE,
        "selection_tolerances": "|".join(str(value) for value in tolerances),
        "q3_start_utc": Q3_START_UTC,
        "q3_end_utc_exclusive": Q3_END_UTC,
        "q4_rows_passed_to_evaluator": int(
            lineage.get("q4_rows_passed_to_evaluator", 0)
        ),
        "q4_used_for_selection": False,
        "selection_payload_sha256": result["selection_payload_sha256"],
    }
    audit_path = root / "lr_selection_audit.csv"
    audit = pd.read_csv(audit_path) if audit_path.is_file() else pd.DataFrame()
    if not audit.empty and "lr_stage" in audit.columns:
        audit = audit[~audit["lr_stage"].astype(str).eq(normalized)]
    audit = pd.concat([audit, pd.DataFrame([audit_row])], ignore_index=True)
    _write_csv(audit, audit_path)

    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    q3_pair_metrics.to_csv(
        analysis / f"q3_pair_metrics_{normalized}.csv.gz",
        index=False,
        compression="gzip",
    )
    return result


__all__ = [
    "ALL_TOLERANCES",
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "FROZEN_LR_PROFILES",
    "LearningRateAnalysisError",
    "SCREEN_TOLERANCES",
    "SUPPORTED_STAGES",
    "build_lr_profile_manifest",
    "collect_lr_comparisons",
    "evaluate_lr_stage",
    "evaluate_q3_pair_metrics",
    "one_standard_error_lr_selection",
    "summarize_lr_scores",
]
