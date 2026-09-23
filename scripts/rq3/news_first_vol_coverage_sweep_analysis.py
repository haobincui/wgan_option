"""Q3-only analysis for the four-stage news-first coverage completion sweep.

The module is deliberately separate from the GPU orchestrator.  It evaluates
best-learned checkpoints on one common Q3 panel, joins the two immutable
reference experiments where a stage is completing an existing factorial
grid, and performs paired two-level inference.  Training seed and CME session
are distinct uncertainty levels: seeds are resampled first and complete CME
sessions are resampled inside every selected seed.

There is intentionally no Q4 loader or evaluator in this module.  A final
candidate can be frozen after all four Q3 stages, but Q4 remains locked until a
separate, explicitly authorised evaluation step.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    MAE_NUMERICAL_TIE_TOLERANCE,
    RunSpec,
    TrainedRunEvaluator,
    _load_panel_source,
    aggregate_pair_metrics,
    compute_sample_metrics,
    holm_adjust,
)


EXPERIMENT_KIND = "coverage_completion_sweep"
Q3_START_UTC = "2023-07-01T00:00:00Z"
Q3_END_UTC = "2023-10-01T00:00:00Z"
SELECTION_PANEL = "common_validation_05m"
SUPPORT_MASK_MODE = "raw_joint"
FIXED_LEARNING_RATE = 5.0e-7
LEARNING_RATES = (5.0e-7, 7.5e-7, 1.0e-6, 1.5e-6, 2.0e-6)
PROFILES = tuple(PROFILE_PARAMETER_COUNTS)
SEEDS = (42, 202, 404)
PRODUCTION_TEXT_MODES = ("current_only", "real_text")
ALL_TEXT_MODES = (*PRODUCTION_TEXT_MODES, "text_shuffle")
DEFAULT_PAIR_COUNT = 123
DEFAULT_SESSION_COUNT = 33
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260820

# The full experimental identity.  No summary or inferential table may drop
# one of these fields before first collapsing the unique Q3 pair rows.
COMBINATION_KEY = (
    "model_family",
    "capacity_profile",
    "initial_learning_rate",
    "tolerance_minutes",
    "text_ablation_mode",
    "seed",
)

STAGE_1 = "stage_1_regression_tolerance_completion"
STAGE_2 = "stage_2_regression_text_shuffle_completion"
STAGE_3 = "stage_3_wgan_low_lr_capacity"
STAGE_4 = "stage_4_regression_capacity_lr_interaction"
STAGE_ORDER = (STAGE_1, STAGE_2, STAGE_3, STAGE_4)


@dataclass(frozen=True)
class StageContract:
    stage_index: int
    model_families: tuple[str, ...]
    profiles: tuple[str, ...]
    learning_rates: tuple[float, ...]
    tolerances: tuple[int, ...]
    text_modes: tuple[str, ...]
    new_job_count: int
    evidence_job_count: int
    primary_mode: str
    purpose: str


STAGE_CONTRACTS: dict[str, StageContract] = {
    STAGE_1: StageContract(
        1,
        ("regression",),
        PROFILES,
        (FIXED_LEARNING_RATE,),
        (10, 15),
        PRODUCTION_TEXT_MODES,
        72,
        72,
        "real_text",
        "complete the 10m/15m Regression capacity comparison",
    ),
    STAGE_2: StageContract(
        2,
        ("regression",),
        PROFILES,
        (FIXED_LEARNING_RATE,),
        (5, 10, 15, 30),
        ALL_TEXT_MODES,
        72,
        216,
        "real_text",
        "compare text-shuffle with current-only and real text",
    ),
    STAGE_3: StageContract(
        3,
        ("wgan",),
        PROFILES,
        (FIXED_LEARNING_RATE,),
        (5, 30),
        PRODUCTION_TEXT_MODES,
        72,
        72,
        "current_only",
        "screen low-learning-rate WGAN capacity and text modes",
    ),
    STAGE_4: StageContract(
        4,
        ("regression",),
        PROFILES,
        LEARNING_RATES,
        (5, 30),
        PRODUCTION_TEXT_MODES,
        240,
        360,
        "real_text",
        "estimate the complete Regression capacity by learning-rate interaction",
    ),
}


class CoverageSweepAnalysisError(ValueError):
    """Raised when stage evidence violates the frozen Q3 contract."""


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
        raise CoverageSweepAnalysisError(f"Expected JSON object: {path}")
    return dict(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise CoverageSweepAnalysisError(f"Expected YAML mapping: {path}")
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


def _same_float(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-15)
    except (TypeError, ValueError):
        return False


def _lr_slug(rate: float) -> str:
    canonical = f"{float(rate):.8e}".replace(".", "p").replace("+", "")
    return f"lr_{canonical}"


def _job_model(job: Mapping[str, Any]) -> str:
    return str(job.get("model_family", job.get("model", ""))).strip().lower()


def _job_profile(job: Mapping[str, Any]) -> str:
    return str(job.get("capacity_profile", "")).strip().lower()


def _job_rate(job: Mapping[str, Any]) -> float:
    value = job.get("initial_learning_rate", job.get("learning_rate"))
    rate = float(pd.to_numeric(value, errors="coerce"))
    if not math.isfinite(rate) or rate <= 0.0:
        raise CoverageSweepAnalysisError(
            f"Invalid learning rate for {job.get('job_id')}: {value}"
        )
    return rate


def _job_seed(job: Mapping[str, Any]) -> int:
    value = float(pd.to_numeric(job.get("seed"), errors="coerce"))
    if not math.isfinite(value) or not value.is_integer():
        raise CoverageSweepAnalysisError(
            f"Invalid seed for {job.get('job_id')}: {job.get('seed')}"
        )
    return int(value)


def _job_tolerance(job: Mapping[str, Any]) -> int:
    return int(job.get("tolerance_minutes", -1))


def _job_mode(job: Mapping[str, Any]) -> str:
    return str(job.get("text_ablation_mode", "")).strip().lower()


def _job_stage(job: Mapping[str, Any]) -> str:
    return str(job.get("stage_id", job.get("experiment_stage", ""))).strip()


def _job_run_dir(job: Mapping[str, Any]) -> Path:
    value = str(job.get("run_dir", job.get("output_root", ""))).strip()
    path = Path(value).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _job_config(job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(str(job.get("training_config_path", ""))).resolve()
    return path, _read_yaml(path)


def _learned_metadata(job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _job_run_dir(job) / "metrics" / "best_learned_checkpoint.json"
    payload = _read_json(path)
    epoch = float(pd.to_numeric(payload.get("best_epoch"), errors="coerce"))
    if not math.isfinite(epoch) or int(epoch) < 1:
        raise CoverageSweepAnalysisError(
            f"best-learned checkpoint must select epoch >=1: {path}"
        )
    if str(payload.get("selection_scope", "trained_epochs_only")) != (
        "trained_epochs_only"
    ):
        raise CoverageSweepAnalysisError(
            f"best-learned checkpoint selection scope drifted: {path}"
        )
    return path, payload


def _learned_checkpoint(job: Mapping[str, Any], metadata: Mapping[str, Any]) -> Path:
    artifacts = metadata.get("artifacts")
    if isinstance(artifacts, Mapping):
        for key in ("model", "generator", "checkpoint"):
            raw = str(artifacts.get(key, "")).strip()
            if raw:
                candidate = Path(raw)
                if not candidate.is_absolute():
                    candidate = _job_run_dir(job) / candidate
                if candidate.is_file():
                    return candidate.resolve()
    run_dir = _job_run_dir(job)
    names = (
        ("vol_regressor_best_learned.pt",)
        if _job_model(job) == "regression"
        else ("generator_best_learned.pt",)
    )
    candidates = [run_dir / "checkpoints" / name for name in names]
    candidates.extend(run_dir / "models" / name for name in names)
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise FileNotFoundError(candidates[0])
    return found.resolve()


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


def _validate_completed_job_lineage(job: Mapping[str, Any]) -> None:
    """Verify immutable config and completed artifacts before inference."""

    config_path, config = _job_config(job)
    declared_config_sha = str(job.get("config_sha256", ""))
    if not declared_config_sha or _sha256(config_path) != declared_config_sha:
        raise CoverageSweepAnalysisError(
            f"Training config hash mismatch: {job.get('job_id')}"
        )
    model, profile = _job_model(job), _job_profile(job)
    expected_parameters = PROFILE_PARAMETER_COUNTS[profile][model]
    if (
        int(job.get("expected_model_parameters", expected_parameters))
        != expected_parameters
    ):
        raise CoverageSweepAnalysisError(
            f"Parameter-count lineage mismatch: {job.get('job_id')}"
        )
    contracts = (
        (int(config.get("seed", -1)) == _job_seed(job), "seed"),
        (
            str(config.get("news_first_capacity_profile", "")).lower() == profile,
            "capacity profile",
        ),
        (
            _same_float(config.get("learning_rate"), _job_rate(job)),
            "learning rate",
        ),
        (
            str(config.get("news_first_text_ablation_mode", "")).lower()
            == _job_mode(job),
            "text mode",
        ),
        (
            str(config.get("support_mask_mode", "")).lower() == SUPPORT_MASK_MODE,
            "support mask",
        ),
        (
            str(config.get("news_first_train_end_utc", Q3_START_UTC)) == Q3_START_UTC,
            "train boundary",
        ),
        (
            str(config.get("news_first_validation_end_utc", Q3_END_UTC)) == Q3_END_UTC,
            "Q3 boundary",
        ),
    )
    failed = [label for valid, label in contracts if not valid]
    if failed:
        raise CoverageSweepAnalysisError(
            f"Training config contract drift for {job.get('job_id')}: {failed}"
        )
    artifacts = job.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CoverageSweepAnalysisError(
            f"Completed job lacks artifact hashes: {job.get('job_id')}"
        )
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise CoverageSweepAnalysisError("Malformed completed artifact lineage")
        path = Path(str(artifact.get("path", "")))
        if not path.is_file() or _sha256(path) != str(artifact.get("sha256", "")):
            raise CoverageSweepAnalysisError(
                f"Completed artifact hash mismatch: {job.get('job_id')}"
            )


def _load_jobs(root: Path) -> list[dict[str, Any]]:
    registry = _read_json(root / "registry" / "jobs.json")
    if str(registry.get("experiment_kind", "")) != EXPERIMENT_KIND:
        raise CoverageSweepAnalysisError("Registry experiment_kind mismatch")
    raw_jobs = registry.get("jobs")
    if not isinstance(raw_jobs, list):
        raise CoverageSweepAnalysisError("Registry jobs must be a list")
    jobs: list[dict[str, Any]] = []
    for raw in raw_jobs:
        if not isinstance(raw, Mapping):
            raise CoverageSweepAnalysisError("Every registry job must be an object")
        job = dict(raw)
        job_id = str(job.get("job_id", "")).strip()
        if not job_id:
            raise CoverageSweepAnalysisError("Every registry job needs job_id")
        status_path = root / "registry" / "jobs" / f"{job_id}.status.json"
        if status_path.is_file():
            job.update(_read_json(status_path))
        jobs.append(job)
    return jobs


def _expected_job_keys(stage_id: str, *, evidence: bool) -> set[tuple[Any, ...]]:
    contract = STAGE_CONTRACTS[stage_id]
    if not evidence and stage_id == STAGE_2:
        modes = ("text_shuffle",)
    else:
        modes = contract.text_modes
    if not evidence and stage_id == STAGE_4:
        profiles = tuple(profile for profile in PROFILES if profile != "large")
        rates = tuple(rate for rate in LEARNING_RATES if rate != FIXED_LEARNING_RATE)
    else:
        profiles = contract.profiles
        rates = contract.learning_rates
    return {
        (model, profile, rate, tolerance, mode, seed)
        for model in contract.model_families
        for profile in profiles
        for rate in rates
        for tolerance in contract.tolerances
        for mode in modes
        for seed in SEEDS
    }


def validate_new_job_matrix(
    stage_id: str, jobs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Validate the exact set of newly trained jobs for one stage."""

    if stage_id not in STAGE_CONTRACTS:
        raise CoverageSweepAnalysisError(f"Unknown stage: {stage_id}")
    selected = [dict(job) for job in jobs if _job_stage(job) == stage_id]
    contract = STAGE_CONTRACTS[stage_id]
    if len(selected) != contract.new_job_count:
        raise CoverageSweepAnalysisError(
            f"{stage_id} requires {contract.new_job_count} new jobs; "
            f"observed={len(selected)}"
        )
    keys: list[tuple[Any, ...]] = []
    for job in selected:
        if str(job.get("status", "")).strip().lower() != "completed":
            raise CoverageSweepAnalysisError(
                f"Stage analysis requires completed job: {job.get('job_id')}"
            )
        keys.append(
            (
                _job_model(job),
                _job_profile(job),
                _job_rate(job),
                _job_tolerance(job),
                _job_mode(job),
                _job_seed(job),
            )
        )
    expected = _expected_job_keys(stage_id, evidence=False)
    if len(set(keys)) != len(keys) or set(keys) != expected:
        missing = sorted(expected - set(keys))
        extra = sorted(set(keys) - expected)
        raise CoverageSweepAnalysisError(
            f"{stage_id} new-job matrix drifted; missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    return selected


def _normalise_pair_metrics(
    frame: pd.DataFrame,
    *,
    evidence_source: str | None,
    forced_model: str | None = None,
    forced_profile: str | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    if "model_family" not in output:
        if "model" in output:
            output["model_family"] = output["model"].astype(str).str.lower()
        elif forced_model:
            output["model_family"] = forced_model
    if forced_model:
        output["model_family"] = forced_model
    if forced_profile:
        output["capacity_profile"] = forced_profile
    if "capacity_profile" not in output:
        raise CoverageSweepAnalysisError("Pair metrics lack capacity_profile")
    output["capacity_profile"] = output["capacity_profile"].astype(str).str.lower()
    if "parameter_count" not in output:
        output["parameter_count"] = [
            PROFILE_PARAMETER_COUNTS[profile][model]
            for profile, model in output[
                ["capacity_profile", "model_family"]
            ].itertuples(index=False, name=None)
        ]
    if "lr_profile" not in output:
        output["lr_profile"] = [
            _lr_slug(value) for value in output["initial_learning_rate"]
        ]
    if evidence_source is not None:
        output["evidence_source"] = evidence_source
    elif "evidence_source" not in output:
        output["evidence_source"] = "unspecified"
    output["model_family"] = output["model_family"].astype(str).str.lower()
    output["text_ablation_mode"] = output["text_ablation_mode"].astype(str).str.lower()
    for column in ("seed", "tolerance_minutes", "parameter_count"):
        output[column] = pd.to_numeric(output[column], errors="coerce").astype(int)
    for column in ("initial_learning_rate", "model_mae", "persistence_mae"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if "support_mask_mode" not in output:
        output["support_mask_mode"] = SUPPORT_MASK_MODE
    if "panel" not in output:
        output["panel"] = SELECTION_PANEL
    return output


def _reference_pair_metrics(
    fixed_reference_root: Path,
    local_lr_reference_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fixed_path = (
        fixed_reference_root / "analysis" / "fixed_lr_capacity_q3_pair_metrics.csv.gz"
    )
    local_path = (
        local_lr_reference_root / "analysis" / "local_lr_q3_pair_metrics.csv.gz"
    )
    fixed = _normalise_pair_metrics(
        pd.read_csv(fixed_path, low_memory=False),
        evidence_source="fixed_lr_capacity_reference",
        forced_model="regression",
    )
    local = _normalise_pair_metrics(
        pd.read_csv(local_path, low_memory=False),
        evidence_source="local_lr_large_reference",
        forced_model="regression",
        forced_profile="large",
    )
    lineage = {
        "fixed_lr_capacity": {
            "root": str(fixed_reference_root),
            "pair_metrics_path": str(fixed_path),
            "pair_metrics_sha256": _sha256(fixed_path),
        },
        "local_lr": {
            "root": str(local_lr_reference_root),
            "pair_metrics_path": str(local_path),
            "pair_metrics_sha256": _sha256(local_path),
        },
    }
    return fixed, local, lineage


def _validate_declared_reference_lineage(
    root: Path, lineage: Mapping[str, Mapping[str, Any]]
) -> None:
    registry = _read_json(root / "registry" / "jobs.json")
    declared = registry.get("reference_roots")
    if not isinstance(declared, Mapping):
        raise CoverageSweepAnalysisError("Registry lacks hashed reference_roots")
    for name in ("fixed_lr_capacity", "local_lr"):
        reference = declared.get(name)
        if not isinstance(reference, Mapping):
            raise CoverageSweepAnalysisError(f"Missing declared reference: {name}")
        files = reference.get("files")
        pair_file = files.get("pair_metrics") if isinstance(files, Mapping) else None
        if not isinstance(pair_file, Mapping):
            raise CoverageSweepAnalysisError(
                f"Declared reference lacks pair-metric hash: {name}"
            )
        observed = lineage[name]
        if Path(str(pair_file.get("path", ""))).resolve() != Path(
            str(observed["pair_metrics_path"])
        ).resolve() or str(pair_file.get("sha256", "")) != str(
            observed["pair_metrics_sha256"]
        ):
            raise CoverageSweepAnalysisError(
                f"Reference pair-metric lineage mismatch: {name}"
            )


def compose_stage_evidence(
    stage_id: str,
    new_stage_metrics: Mapping[str, pd.DataFrame],
    *,
    fixed_reference: pd.DataFrame | None = None,
    local_lr_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compose the complete evidence grid for a stage without duplicate cells."""

    if stage_id == STAGE_1:
        parts = [new_stage_metrics[STAGE_1]]
    elif stage_id == STAGE_2:
        if fixed_reference is None:
            raise CoverageSweepAnalysisError("Stage 2 requires fixed-LR reference")
        fixed = fixed_reference[
            fixed_reference["tolerance_minutes"].isin((5, 30))
            & fixed_reference["text_ablation_mode"].isin(PRODUCTION_TEXT_MODES)
        ]
        stage1 = new_stage_metrics[STAGE_1]
        shuffle = new_stage_metrics[STAGE_2]
        parts = [fixed, stage1, shuffle]
    elif stage_id == STAGE_3:
        parts = [new_stage_metrics[STAGE_3]]
    elif stage_id == STAGE_4:
        if fixed_reference is None or local_lr_reference is None:
            raise CoverageSweepAnalysisError(
                "Stage 4 requires fixed-LR and local-LR references"
            )
        fixed_non_large = fixed_reference[
            ~fixed_reference["capacity_profile"].eq("large")
            & fixed_reference["tolerance_minutes"].isin((5, 30))
            & fixed_reference["text_ablation_mode"].isin(PRODUCTION_TEXT_MODES)
        ]
        local_large = local_lr_reference[
            local_lr_reference["tolerance_minutes"].isin((5, 30))
            & local_lr_reference["text_ablation_mode"].isin(PRODUCTION_TEXT_MODES)
            & local_lr_reference["initial_learning_rate"].map(
                lambda value: any(_same_float(value, rate) for rate in LEARNING_RATES)
            )
        ]
        parts = [fixed_non_large, local_large, new_stage_metrics[STAGE_4]]
    else:
        raise CoverageSweepAnalysisError(f"Unknown stage: {stage_id}")
    output = pd.concat(parts, ignore_index=True)
    output["stage_id"] = stage_id
    return output


def validate_stage_pair_metrics(
    stage_id: str,
    pair_metrics: pd.DataFrame,
    *,
    expected_pair_count: int = DEFAULT_PAIR_COUNT,
    expected_session_count: int = DEFAULT_SESSION_COUNT,
) -> pd.DataFrame:
    """Fail closed unless a stage has its exact, paired three-seed Q3 grid."""

    if stage_id not in STAGE_CONTRACTS:
        raise CoverageSweepAnalysisError(f"Unknown stage: {stage_id}")
    required = {
        *COMBINATION_KEY,
        "stage_id",
        "parameter_count",
        "pair_id",
        "session_id",
        "model_mae",
        "persistence_mae",
        "support_mask_mode",
        "panel",
    }
    missing = sorted(required - set(pair_metrics.columns))
    if missing:
        raise CoverageSweepAnalysisError(f"{stage_id} pair metrics missing {missing}")
    frame = _normalise_pair_metrics(pair_metrics, evidence_source=None)
    if set(frame["stage_id"].astype(str)) != {stage_id}:
        raise CoverageSweepAnalysisError(f"{stage_id} evidence has wrong stage_id")
    if set(frame["panel"].astype(str)) != {SELECTION_PANEL}:
        raise CoverageSweepAnalysisError(
            f"{stage_id} contains a non-Q3 evaluation panel"
        )
    if set(frame["support_mask_mode"].astype(str).str.lower()) != {SUPPORT_MASK_MODE}:
        raise CoverageSweepAnalysisError(f"{stage_id} support-mask mode drifted")
    for column in ("model_mae", "persistence_mae"):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or bool((values <= 0.0).any()):
            raise CoverageSweepAnalysisError(
                f"{stage_id} {column} must be finite and positive"
            )
    expected = _expected_job_keys(stage_id, evidence=True)
    observed = set(frame[list(COMBINATION_KEY)].itertuples(index=False, name=None))
    if observed != expected:
        missing_keys = sorted(expected - observed)
        extra_keys = sorted(observed - expected)
        raise CoverageSweepAnalysisError(
            f"{stage_id} evidence matrix drifted; missing={missing_keys[:5]}, "
            f"extra={extra_keys[:5]}"
        )
    if len(observed) != STAGE_CONTRACTS[stage_id].evidence_job_count:
        raise CoverageSweepAnalysisError(f"{stage_id} evidence job count drifted")
    if frame.duplicated([*COMBINATION_KEY, "pair_id"]).any():
        raise CoverageSweepAnalysisError(f"{stage_id} has duplicate pair metrics")

    coverage: list[frozenset[tuple[str, str]]] = []
    for key, group in frame.groupby(list(COMBINATION_KEY), sort=False):
        if group["pair_id"].nunique() != int(expected_pair_count):
            raise CoverageSweepAnalysisError(
                f"{stage_id}/{key} expected {expected_pair_count} Q3 pairs"
            )
        if group["session_id"].nunique() != int(expected_session_count):
            raise CoverageSweepAnalysisError(
                f"{stage_id}/{key} expected {expected_session_count} Q3 sessions"
            )
        profile, model = str(key[1]), str(key[0])
        expected_parameters = PROFILE_PARAMETER_COUNTS[profile][model]
        if set(group["parameter_count"].astype(int)) != {expected_parameters}:
            raise CoverageSweepAnalysisError(
                f"{stage_id}/{key} parameter count drifted"
            )
        coverage.append(
            frozenset(
                group[["pair_id", "session_id"]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
        )
    if len(set(coverage)) != 1:
        raise CoverageSweepAnalysisError(
            f"{stage_id} cells do not share one paired Q3 panel"
        )

    # Persistence is a property of the common panel and must not depend on
    # model, capacity, LR, text mode, tolerance training source, or seed.
    persistence_span = frame.groupby("pair_id")["persistence_mae"].agg(
        lambda values: float(values.max() - values.min())
    )
    if bool((persistence_span > 1.0e-12).any()):
        raise CoverageSweepAnalysisError(
            f"{stage_id} persistence differs across paired experiment cells"
        )
    return frame.sort_values(
        [*COMBINATION_KEY, "session_id", "pair_id"], kind="stable"
    ).reset_index(drop=True)


def build_seed_run_metrics(stage_metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse pair rows to the exact six-field experimental key."""

    keys = ["stage_id", *COMBINATION_KEY, "parameter_count"]
    rows: list[dict[str, Any]] = []
    for values, group in stage_metrics.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, values))
        model = float(group["model_mae"].mean())
        persistence = float(group["persistence_mae"].mean())
        ratio = model / persistence
        if "win" in group.columns:
            pair_wins = pd.to_numeric(group["win"], errors="coerce")
            if pair_wins.isna().any() or not pair_wins.isin((0.0, 1.0)).all():
                raise CoverageSweepAnalysisError(
                    "Pair-level win must be a finite binary indicator"
                )
        else:
            # Reference and newly evaluated pair metrics normally carry the
            # canonical ``win`` emitted by aggregate_pair_metrics.  The
            # fallback keeps injected/legacy evidence on the same frozen
            # numerical-tie policy: abs(gap) <= 1e-8 IV is a tie and only a
            # gap strictly below -1e-8 IV is a win.
            pair_wins = (
                group["model_mae"] - group["persistence_mae"]
                < -MAE_NUMERICAL_TIE_TOLERANCE
            ).astype(float)
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
                "pair_win_rate": float(pair_wins.mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys, kind="stable").reset_index(drop=True)


def summarize_across_seeds(seed_runs: pd.DataFrame) -> pd.DataFrame:
    """Return mean and sample SD across exactly three independent seeds."""

    keys = [
        "stage_id",
        "model_family",
        "capacity_profile",
        "parameter_count",
        "initial_learning_rate",
        "tolerance_minutes",
        "text_ablation_mode",
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
        if seeds != SEEDS:
            raise CoverageSweepAnalysisError(
                f"Across-seed cell lost a seed: {dict(zip(keys, values))}"
            )
        row = dict(zip(keys, values))
        row.update(
            {
                "seed_count": 3,
                "seeds": "|".join(str(seed) for seed in seeds),
                "pair_count_per_seed": int(group["pair_count"].iloc[0]),
                "session_count_per_seed": int(group["session_count"].iloc[0]),
            }
        )
        for metric in metrics:
            array = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(array.mean())
            row[f"{metric}_sd"] = float(array.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys, kind="stable").reset_index(drop=True)


def build_stage_scores(stage_id: str, seed_runs: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight stage-tolerance log-MAE scores, with no improvement gate."""

    contract = STAGE_CONTRACTS[stage_id]
    selected = seed_runs[seed_runs["stage_id"].eq(stage_id)].copy()
    inner_keys = [
        "model_family",
        "capacity_profile",
        "parameter_count",
        "initial_learning_rate",
        "text_ablation_mode",
        "seed",
    ]
    per_seed: list[dict[str, Any]] = []
    for values, group in selected.groupby(inner_keys, sort=True, dropna=False):
        observed = tuple(sorted(group["tolerance_minutes"].astype(int).tolist()))
        if observed != tuple(sorted(contract.tolerances)):
            raise CoverageSweepAnalysisError(
                f"{stage_id} score lost a tolerance: {observed}"
            )
        score = float(group["log_mae_ratio"].mean())
        per_seed.append(
            dict(zip(inner_keys, values))
            | {
                "mean_log_mae_ratio": score,
                "geometric_mae_ratio": math.exp(score),
                "improvement_fraction": 1.0 - math.exp(score),
            }
        )
    per_seed_frame = pd.DataFrame(per_seed)
    outer_keys = inner_keys[:-1]
    rows: list[dict[str, Any]] = []
    for values, group in per_seed_frame.groupby(outer_keys, sort=True, dropna=False):
        if tuple(sorted(group["seed"].astype(int))) != SEEDS:
            raise CoverageSweepAnalysisError(f"{stage_id} score lost a seed")
        scores = group["mean_log_mae_ratio"].to_numpy(dtype=float)
        mean_score = float(scores.mean())
        rows.append(
            {"stage_id": stage_id, **dict(zip(outer_keys, values))}
            | {
                "tolerance_scope": "combined_"
                + "_".join(f"{value:02d}m" for value in contract.tolerances),
                "seed_count": 3,
                "mean_log_mae_ratio": mean_score,
                "sd_log_mae_ratio_across_seeds": float(scores.std(ddof=1)),
                "geometric_mae_ratio": math.exp(mean_score),
                "mean_improvement_fraction": 1.0 - math.exp(mean_score),
                "minimum_improvement_gate_applied": False,
            }
        )
    output = pd.DataFrame(rows)
    output["rank_within_model_and_mode"] = output.groupby(
        ["model_family", "text_ablation_mode"], sort=False
    )["mean_log_mae_ratio"].rank(method="first")
    return output.sort_values(
        ["model_family", "text_ablation_mode", "mean_log_mae_ratio", "parameter_count"],
        kind="stable",
    ).reset_index(drop=True)


def _two_level_bootstrap(
    differences: pd.DataFrame,
    *,
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    """Resample seeds, then paired complete CME-session clusters."""

    if int(iterations) < 2:
        raise CoverageSweepAnalysisError("Bootstrap requires at least two draws")
    required = {"seed", "session_id", "pair_id", "difference"}
    missing = sorted(required - set(differences.columns))
    if missing:
        raise CoverageSweepAnalysisError(f"Bootstrap input missing {missing}")
    frame = differences.copy()
    frame["difference"] = pd.to_numeric(frame["difference"], errors="coerce")
    if not np.isfinite(frame["difference"].to_numpy(dtype=float)).all():
        raise CoverageSweepAnalysisError("Bootstrap differences are non-finite")
    seeds = tuple(sorted(frame["seed"].astype(int).unique().tolist()))
    if seeds != SEEDS:
        raise CoverageSweepAnalysisError("Bootstrap lost a training seed")
    extras = [column for column in ("tolerance_minutes",) if column in frame.columns]
    if frame.duplicated(["seed", "session_id", "pair_id", *extras]).any():
        raise CoverageSweepAnalysisError("Duplicate paired bootstrap row")

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
        raise CoverageSweepAnalysisError(
            "Seed/session bootstrap requires identical paired coverage"
        )

    rng = np.random.default_rng(int(random_seed))
    seed_indexes = rng.integers(0, len(seeds), size=(iterations, len(seeds)))
    # Vectorise session resampling.  Axis 2 represents the three independently
    # drawn seed slots, so drawing the same training seed twice still produces
    # independent within-seed session resamples.
    session_draw_means = np.empty((len(seeds), iterations, len(seeds)), dtype=float)
    for seed_index, seed in enumerate(seeds):
        sums, counts, session_count, _ = arrays[seed]
        indexes = rng.integers(
            0,
            session_count,
            size=(iterations, len(seeds), session_count),
        )
        numerator = sums[indexes].sum(axis=2)
        denominator = counts[indexes].sum(axis=2)
        session_draw_means[seed_index] = numerator / denominator
    iteration_indexes = np.arange(iterations)[:, None]
    slot_indexes = np.arange(len(seeds))[None, :]
    draws = session_draw_means[
        seed_indexes,
        iteration_indexes,
        slot_indexes,
    ].mean(axis=1)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    p_lower = (float(np.sum(draws <= 0.0)) + 1.0) / (iterations + 1.0)
    p_upper = (float(np.sum(draws >= 0.0)) + 1.0) / (iterations + 1.0)
    return {
        "mean_difference": float(np.mean(points)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_two_sided": float(min(1.0, 2.0 * min(p_lower, p_upper))),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(random_seed),
        "seed_count": 3,
        "session_count_per_seed": int(arrays[seeds[0]][2]),
        "pair_rows_per_seed": int(arrays[seeds[0]][3]),
        "resampling_method": "seed_then_paired_CME_session_cluster",
    }


def _scope_frames(
    frame: pd.DataFrame, tolerances: Sequence[int]
) -> list[tuple[str, pd.DataFrame]]:
    parts = [
        (
            f"{tolerance:02d}m",
            frame[frame["tolerance_minutes"].astype(int).eq(tolerance)],
        )
        for tolerance in tolerances
    ]
    parts.append(
        (
            "combined_" + "_".join(f"{value:02d}m" for value in tolerances),
            frame,
        )
    )
    return parts


def _paired_difference(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["seed", "tolerance_minutes", "pair_id", "session_id"]
    paired = candidate[keys + ["model_mae"]].merge(
        reference[keys + ["model_mae"]],
        on=keys,
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    if len(paired) != len(candidate) or len(paired) != len(reference):
        raise CoverageSweepAnalysisError("Paired contrast lost Q3 rows")
    result = paired[keys].copy()
    result["difference"] = paired["model_mae_candidate"].to_numpy(dtype=float) - paired[
        "model_mae_reference"
    ].to_numpy(dtype=float)
    return result


def _contrast_record(
    *,
    stage_id: str,
    family: str,
    scope: str,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "contrast_family": family,
        "tolerance_scope": scope,
        "candidate_model_family": candidate.get("model_family", ""),
        "candidate_capacity_profile": candidate.get("capacity_profile", ""),
        "candidate_initial_learning_rate": candidate.get(
            "initial_learning_rate", float("nan")
        ),
        "candidate_text_ablation_mode": candidate.get("text_ablation_mode", ""),
        "reference_model_family": reference.get("model_family", ""),
        "reference_capacity_profile": reference.get("capacity_profile", ""),
        "reference_initial_learning_rate": reference.get(
            "initial_learning_rate", float("nan")
        ),
        "reference_text_ablation_mode": reference.get("text_ablation_mode", ""),
        **dict(statistics),
    }


def build_stage_bootstrap(
    stage_id: str,
    stage_metrics: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Build persistence, capacity, text, LR and interaction contrasts."""

    contract = STAGE_CONTRACTS[stage_id]
    frame = stage_metrics.copy()
    rows: list[dict[str, Any]] = []
    base_group_keys = [
        "model_family",
        "capacity_profile",
        "initial_learning_rate",
        "text_ablation_mode",
    ]

    # Every scored cell is contrasted with persistence on the same pairs.
    for values, group in frame.groupby(base_group_keys, sort=True, dropna=False):
        identity = dict(zip(base_group_keys, values))
        for scope, scoped in _scope_frames(group, contract.tolerances):
            differences = scoped[
                ["seed", "session_id", "pair_id", "tolerance_minutes"]
            ].copy()
            differences["difference"] = scoped["model_mae"].to_numpy(
                dtype=float
            ) - scoped["persistence_mae"].to_numpy(dtype=float)
            rows.append(
                _contrast_record(
                    stage_id=stage_id,
                    family="model_minus_persistence",
                    scope=scope,
                    candidate=identity,
                    reference={"model_family": "persistence"},
                    statistics=_two_level_bootstrap(
                        differences,
                        iterations=iterations,
                        random_seed=random_seed,
                    ),
                )
            )

    # Capacity contrasts use the point-estimate leader inside each
    # model/LR/mode lane.  The reference is chosen from the combined Q3 score,
    # never from Q4 and never through a minimum-improvement gate.
    for lane_values, lane_scores in scores.groupby(
        ["model_family", "initial_learning_rate", "text_ablation_mode"],
        sort=True,
        dropna=False,
    ):
        model, rate, mode = lane_values
        leader = lane_scores.sort_values(
            ["mean_log_mae_ratio", "parameter_count"], kind="stable"
        ).iloc[0]
        reference = frame[
            frame["model_family"].eq(model)
            & frame["capacity_profile"].eq(leader["capacity_profile"])
            & np.isclose(frame["initial_learning_rate"], float(rate))
            & frame["text_ablation_mode"].eq(mode)
        ]
        for profile in lane_scores["capacity_profile"].astype(str):
            if profile == str(leader["capacity_profile"]):
                continue
            candidate = frame[
                frame["model_family"].eq(model)
                & frame["capacity_profile"].eq(profile)
                & np.isclose(frame["initial_learning_rate"], float(rate))
                & frame["text_ablation_mode"].eq(mode)
            ]
            differences = _paired_difference(candidate, reference)
            for scope, scoped in _scope_frames(differences, contract.tolerances):
                rows.append(
                    _contrast_record(
                        stage_id=stage_id,
                        family="capacity_minus_lane_leader",
                        scope=scope,
                        candidate={
                            "model_family": model,
                            "capacity_profile": profile,
                            "initial_learning_rate": rate,
                            "text_ablation_mode": mode,
                        },
                        reference={
                            "model_family": model,
                            "capacity_profile": leader["capacity_profile"],
                            "initial_learning_rate": rate,
                            "text_ablation_mode": mode,
                        },
                        statistics=_two_level_bootstrap(
                            scoped,
                            iterations=iterations,
                            random_seed=random_seed + 101,
                        ),
                    )
                )

    # All available text modes are paired inside model/capacity/LR lanes.
    text_pairs = (
        (("real_text", "current_only"),)
        if stage_id in {STAGE_1, STAGE_3}
        else (
            ("real_text", "current_only"),
            ("text_shuffle", "current_only"),
            ("text_shuffle", "real_text"),
        )
        if stage_id == STAGE_2
        else ()
    )
    for model, profile, rate in (
        frame[["model_family", "capacity_profile", "initial_learning_rate"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        for candidate_mode, reference_mode in text_pairs:
            candidate = frame[
                frame["model_family"].eq(model)
                & frame["capacity_profile"].eq(profile)
                & np.isclose(frame["initial_learning_rate"], float(rate))
                & frame["text_ablation_mode"].eq(candidate_mode)
            ]
            reference = frame[
                frame["model_family"].eq(model)
                & frame["capacity_profile"].eq(profile)
                & np.isclose(frame["initial_learning_rate"], float(rate))
                & frame["text_ablation_mode"].eq(reference_mode)
            ]
            differences = _paired_difference(candidate, reference)
            for scope, scoped in _scope_frames(differences, contract.tolerances):
                rows.append(
                    _contrast_record(
                        stage_id=stage_id,
                        family="text_mode_difference",
                        scope=scope,
                        candidate={
                            "model_family": model,
                            "capacity_profile": profile,
                            "initial_learning_rate": rate,
                            "text_ablation_mode": candidate_mode,
                        },
                        reference={
                            "model_family": model,
                            "capacity_profile": profile,
                            "initial_learning_rate": rate,
                            "text_ablation_mode": reference_mode,
                        },
                        statistics=_two_level_bootstrap(
                            scoped,
                            iterations=iterations,
                            random_seed=random_seed + 202,
                        ),
                    )
                )

    if stage_id == STAGE_4:
        # LR contrasts are local to capacity and mode.
        for (model, profile, mode), lane_scores in scores.groupby(
            ["model_family", "capacity_profile", "text_ablation_mode"],
            sort=True,
            dropna=False,
        ):
            leader = lane_scores.sort_values("mean_log_mae_ratio", kind="stable").iloc[
                0
            ]
            leader_rate = float(leader["initial_learning_rate"])
            reference = frame[
                frame["model_family"].eq(model)
                & frame["capacity_profile"].eq(profile)
                & np.isclose(frame["initial_learning_rate"], leader_rate)
                & frame["text_ablation_mode"].eq(mode)
            ]
            for rate in lane_scores["initial_learning_rate"].astype(float):
                if _same_float(rate, leader_rate):
                    continue
                candidate = frame[
                    frame["model_family"].eq(model)
                    & frame["capacity_profile"].eq(profile)
                    & np.isclose(frame["initial_learning_rate"], rate)
                    & frame["text_ablation_mode"].eq(mode)
                ]
                differences = _paired_difference(candidate, reference)
                for scope, scoped in _scope_frames(differences, contract.tolerances):
                    rows.append(
                        _contrast_record(
                            stage_id=stage_id,
                            family="learning_rate_minus_lane_leader",
                            scope=scope,
                            candidate={
                                "model_family": model,
                                "capacity_profile": profile,
                                "initial_learning_rate": rate,
                                "text_ablation_mode": mode,
                            },
                            reference={
                                "model_family": model,
                                "capacity_profile": profile,
                                "initial_learning_rate": leader_rate,
                                "text_ablation_mode": mode,
                            },
                            statistics=_two_level_bootstrap(
                                scoped,
                                iterations=iterations,
                                random_seed=random_seed + 303,
                            ),
                        )
                    )

        # True difference-in-differences interaction, anchored at Large and
        # LR 5e-7.  Negative means the candidate capacity benefits more from
        # the candidate LR than Large does on the same pairs.
        pair_keys = ["seed", "tolerance_minutes", "pair_id", "session_id"]
        for profile in (value for value in PROFILES if value != "large"):
            for rate in (
                value for value in LEARNING_RATES if value != FIXED_LEARNING_RATE
            ):
                for mode in PRODUCTION_TEXT_MODES:
                    cells: dict[str, pd.DataFrame] = {}
                    for label, cell_profile, cell_rate in (
                        ("a", profile, rate),
                        ("b", profile, FIXED_LEARNING_RATE),
                        ("c", "large", rate),
                        ("d", "large", FIXED_LEARNING_RATE),
                    ):
                        cells[label] = frame[
                            frame["capacity_profile"].eq(cell_profile)
                            & np.isclose(
                                frame["initial_learning_rate"], float(cell_rate)
                            )
                            & frame["text_ablation_mode"].eq(mode)
                        ][pair_keys + ["model_mae"]].rename(
                            columns={"model_mae": f"mae_{label}"}
                        )
                    paired = cells["a"]
                    for label in ("b", "c", "d"):
                        paired = paired.merge(
                            cells[label], on=pair_keys, validate="one_to_one"
                        )
                    expected_rows = len(cells["a"])
                    if len(paired) != expected_rows or any(
                        len(cell) != expected_rows for cell in cells.values()
                    ):
                        raise CoverageSweepAnalysisError(
                            "Stage 4 interaction contrast lost paired rows"
                        )
                    differences = paired[pair_keys].copy()
                    differences["difference"] = (
                        paired["mae_a"]
                        - paired["mae_b"]
                        - paired["mae_c"]
                        + paired["mae_d"]
                    )
                    for scope, scoped in _scope_frames(
                        differences, contract.tolerances
                    ):
                        rows.append(
                            _contrast_record(
                                stage_id=stage_id,
                                family="capacity_by_learning_rate_interaction",
                                scope=scope,
                                candidate={
                                    "model_family": "regression",
                                    "capacity_profile": profile,
                                    "initial_learning_rate": rate,
                                    "text_ablation_mode": mode,
                                },
                                reference={
                                    "model_family": "regression",
                                    "capacity_profile": "large",
                                    "initial_learning_rate": FIXED_LEARNING_RATE,
                                    "text_ablation_mode": mode,
                                },
                                statistics=_two_level_bootstrap(
                                    scoped,
                                    iterations=iterations,
                                    random_seed=random_seed + 404,
                                ),
                            )
                        )

    output = pd.DataFrame(rows)
    if output.empty:
        raise CoverageSweepAnalysisError(f"{stage_id} produced no bootstrap rows")
    output["p_holm"] = np.nan
    for _, indexes in output.groupby("contrast_family", sort=False).groups.items():
        output.loc[indexes, "p_holm"] = holm_adjust(
            output.loc[indexes, "p_two_sided"].tolist()
        )
    return output.sort_values(
        ["contrast_family", "candidate_text_ablation_mode", "tolerance_scope"],
        kind="stable",
    ).reset_index(drop=True)


def build_stage_selection(stage_id: str, scores: pd.DataFrame) -> dict[str, Any]:
    """Freeze a transparent Q3 point-estimate ranking without the old gate."""

    contract = STAGE_CONTRACTS[stage_id]
    production = scores[scores["text_ablation_mode"].isin(PRODUCTION_TEXT_MODES)].copy()
    if production.empty:
        raise CoverageSweepAnalysisError(f"{stage_id} has no production-mode score")
    leaders: dict[str, Any] = {}
    for mode, group in production.groupby("text_ablation_mode", sort=True):
        winner = group.sort_values(
            ["mean_log_mae_ratio", "parameter_count", "initial_learning_rate"],
            kind="stable",
        ).iloc[0]
        leaders[str(mode)] = {
            "model_family": str(winner["model_family"]),
            "capacity_profile": str(winner["capacity_profile"]),
            "parameter_count": int(winner["parameter_count"]),
            "initial_learning_rate": float(winner["initial_learning_rate"]),
            "text_ablation_mode": str(winner["text_ablation_mode"]),
            "geometric_mae_ratio": float(winner["geometric_mae_ratio"]),
            "mean_improvement_fraction": float(winner["mean_improvement_fraction"]),
        }
    overall = production.sort_values(
        ["mean_log_mae_ratio", "parameter_count", "initial_learning_rate"],
        kind="stable",
    ).iloc[0]
    return {
        "schema_version": 1,
        "stage_id": stage_id,
        "stage_index": contract.stage_index,
        "purpose": contract.purpose,
        "selection_panel": SELECTION_PANEL,
        "selection_interval_start_utc": Q3_START_UTC,
        "selection_interval_end_utc_exclusive": Q3_END_UTC,
        "selection_rule": "minimum_Q3_equal_tolerance_mean_log_MAE_ratio",
        "minimum_improvement_gate_applied": False,
        "selection_is_exploratory": True,
        "q4_used_for_selection": False,
        "q4_predictions_generated": False,
        "primary_mode": contract.primary_mode,
        "mode_leaders": leaders,
        "overall_point_estimate_leader": {
            "model_family": str(overall["model_family"]),
            "capacity_profile": str(overall["capacity_profile"]),
            "parameter_count": int(overall["parameter_count"]),
            "initial_learning_rate": float(overall["initial_learning_rate"]),
            "text_ablation_mode": str(overall["text_ablation_mode"]),
            "geometric_mae_ratio": float(overall["geometric_mae_ratio"]),
            "mean_improvement_fraction": float(overall["mean_improvement_fraction"]),
        },
        "tolerances": list(contract.tolerances),
        "task_semantics_warning": (
            "Tolerance changes the news-wait/task composition; it is not merely "
            "an independent sample-size multiplier. The market forecast horizon "
            "remains current 5 minutes to target 5 minutes."
        ),
    }


def _run_spec(job: Mapping[str, Any]) -> RunSpec:
    metadata_path, metadata = _learned_metadata(job)
    config_path, config = _job_config(job)
    return RunSpec(
        run_id=str(job["job_id"]),
        run_dir=_job_run_dir(job),
        model=_job_model(job),
        tolerance_minutes=_job_tolerance(job),
        seed=_job_seed(job),
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


def _q3_panel(jobs: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    sources: set[tuple[str, str, str]] = set()
    for job in jobs:
        _, config = _job_config(job)
        sources.add(
            (
                str(config.get("news_first_common_eval_data_path", "")),
                str(config.get("sheet_name", "gan_input_ready")),
                str(config.get("support_mask_mode", "none")),
            )
        )
    if len(sources) != 1:
        raise CoverageSweepAnalysisError("Jobs disagree on common Q3 panel")
    raw_path, sheet_name, support_mode = next(iter(sources))
    if support_mode != SUPPORT_MASK_MODE:
        raise CoverageSweepAnalysisError("Q3 panel must use raw_joint support mask")
    workbook = Path(raw_path).resolve()
    raw = pd.read_excel(workbook, sheet_name=sheet_name)
    timestamps = pd.to_datetime(
        raw.get("effective_origin_utc"), errors="coerce", utc=True
    )
    if timestamps.isna().any():
        raise CoverageSweepAnalysisError("Invalid evaluation timestamps")
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
        raise CoverageSweepAnalysisError("A non-Q3 row reached evaluation")
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


def evaluate_new_stage_pair_metrics(
    experiment_root: str | Path,
    stage_id: str,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate every newly trained stage checkpoint on the common Q3 panel."""

    root = Path(experiment_root).resolve()
    jobs = validate_new_job_matrix(stage_id, _load_jobs(root))
    panel, lineage = _q3_panel(jobs)
    production = evaluator or TrainedRunEvaluator(mc_samples=16)
    parts: list[pd.DataFrame] = []
    for job in jobs:
        _validate_completed_job_lineage(job)
        spec = _run_spec(job)
        predictions = production(spec, "q3", panel.copy())
        samples, exclusions, _ = compute_sample_metrics(
            spec,
            SELECTION_PANEL,
            panel,
            predictions,
            evaluate_embedded_atm_skew=False,
        )
        if not exclusions.empty:
            raise CoverageSweepAnalysisError(
                f"Q3 prediction exclusions for {spec.run_id}: "
                f"{exclusions['exclusion_code'].value_counts().to_dict()}"
            )
        pairs = aggregate_pair_metrics(samples)
        pairs = pairs[
            pairs["stratum_type"].astype(str).eq("overall")
            & pairs["stratum_value"].astype(str).eq("all")
        ].copy()
        _, learned = _learned_metadata(job)
        expected_model = _first_metric(
            learned,
            (
                "masked_pair_balanced_mae",
                "best_learned_masked_pair_balanced_mae",
                "val_recon",
            ),
        )
        expected_persistence = _first_metric(
            learned,
            (
                "persistence_masked_pair_balanced_mae",
                "masked_pair_balanced_persistence_mae",
                "val_current_recon",
            ),
        )
        for label, expected, actual in (
            ("model", expected_model, float(pairs["model_mae"].mean())),
            (
                "persistence",
                expected_persistence,
                float(pairs["persistence_mae"].mean()),
            ),
        ):
            if expected is None:
                continue
            expected_value = float(pd.to_numeric(expected, errors="coerce"))
            tolerance = max(1.0e-7, abs(expected_value) * 1.0e-4)
            if not math.isfinite(expected_value) or not math.isclose(
                actual, expected_value, rel_tol=0.0, abs_tol=tolerance
            ):
                raise CoverageSweepAnalysisError(
                    f"Pair-balanced {label} MAE disagrees with learned metadata: "
                    f"{spec.run_id}"
                )
        profile, model = _job_profile(job), _job_model(job)
        pairs.insert(0, "model_family", model)
        pairs.insert(1, "capacity_profile", profile)
        pairs.insert(2, "parameter_count", PROFILE_PARAMETER_COUNTS[profile][model])
        pairs.insert(3, "initial_learning_rate", _job_rate(job))
        pairs.insert(
            4, "lr_profile", str(job.get("lr_profile", _lr_slug(_job_rate(job))))
        )
        pairs.insert(5, "evidence_source", "coverage_completion_new_job")
        parts.append(pairs)
    output = pd.concat(parts, ignore_index=True)
    output["stage_id"] = stage_id
    lineage.update(
        {
            "stage_id": stage_id,
            "evaluated_run_count": len(jobs),
            "selection_scope_verified": True,
            "q4_used_for_selection": False,
            "q4_predictions_generated": False,
        }
    )
    return output, lineage


def _discover_reference_roots(
    root: Path,
    fixed_reference_root: str | Path | None,
    local_lr_reference_root: str | Path | None,
) -> tuple[Path, Path]:
    if fixed_reference_root is not None and local_lr_reference_root is not None:
        return Path(fixed_reference_root).resolve(), Path(
            local_lr_reference_root
        ).resolve()
    candidates = (root / "run_manifest.json", root / "registry" / "jobs.json")
    mappings: list[Mapping[str, Any]] = []
    for path in candidates:
        if path.is_file():
            payload = _read_json(path)
            for key in ("reference_roots", "reference_experiments"):
                value = payload.get(key)
                if isinstance(value, Mapping):
                    mappings.append(value)

    def resolve(keys: Sequence[str], explicit: str | Path | None) -> Path:
        if explicit is not None:
            return Path(explicit).resolve()
        for mapping in mappings:
            for key in keys:
                value = mapping.get(key)
                if isinstance(value, Mapping):
                    value = value.get("root", value.get("path"))
                if value:
                    return Path(str(value)).resolve()
        raise CoverageSweepAnalysisError(
            f"Cannot resolve reference root; accepted keys={list(keys)}"
        )

    return (
        resolve(("fixed_lr_capacity", "fixed_lr_capacity_root"), fixed_reference_root),
        resolve(("local_lr", "local_lr_root"), local_lr_reference_root),
    )


def _stage_artifact_dir(root: Path, stage_id: str) -> Path:
    return root / "analysis" / "stages" / stage_id


def _validated_cached_new_metrics(root: Path, stage_id: str) -> pd.DataFrame | None:
    stage_dir = _stage_artifact_dir(root, stage_id)
    cache = stage_dir / "new_q3_pair_metrics.csv.gz"
    validation_path = stage_dir / "validation_summary.json"
    if not cache.is_file() or not validation_path.is_file():
        return None
    validation = _read_json(validation_path)
    artifact = validation.get("artifacts", {}).get("new_pair_metrics")
    if not isinstance(artifact, Mapping):
        return None
    if Path(str(artifact.get("path", ""))).resolve() != cache.resolve() or _sha256(
        cache
    ) != str(artifact.get("sha256", "")):
        raise CoverageSweepAnalysisError(
            f"Cached new-pair metric hash mismatch: {stage_id}"
        )
    if bool(validation.get("q4_used_for_selection", False)) or bool(
        validation.get("q4_predictions_generated", False)
    ):
        raise CoverageSweepAnalysisError(f"Cached stage claims Q4 access: {stage_id}")
    output = _normalise_pair_metrics(
        pd.read_csv(cache, low_memory=False), evidence_source=None
    )
    output["stage_id"] = stage_id
    return output


def _final_candidate(
    selections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    regression = dict(selections[STAGE_4]["overall_point_estimate_leader"])
    wgan = dict(selections[STAGE_3]["overall_point_estimate_leader"])
    overall = min((regression, wgan), key=lambda row: float(row["geometric_mae_ratio"]))
    return {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "selection_scope": "Q3_common_validation_05m_only",
        "selection_rule": "minimum_Q3_equal_tolerance_mean_log_MAE_ratio_no_gate",
        "minimum_improvement_gate_applied": False,
        "selection_is_exploratory": True,
        "all_four_stages_complete": True,
        "candidate_frozen": True,
        "q4_used_for_selection": False,
        "q4_predictions_generated": False,
        "q4_evaluation_permitted_by_this_artifact": False,
        "regression_candidate": regression,
        "wgan_candidate": wgan,
        "overall_point_estimate_candidate": overall,
        "task_semantics_warning": (
            "Tolerance-specific training datasets represent different news-wait "
            "tasks. Cross-tolerance rankings are exploratory even though every "
            "checkpoint is evaluated on the same Q3 market-pair panel."
        ),
    }


def run_coverage_sweep_analysis(
    experiment_root: str | Path,
    *,
    fixed_reference_root: str | Path | None = None,
    local_lr_reference_root: str | Path | None = None,
    stages: Sequence[str] | None = None,
    stage_pair_metrics: Mapping[str, pd.DataFrame] | None = None,
    fixed_reference_pair_metrics: pd.DataFrame | None = None,
    local_lr_reference_pair_metrics: pd.DataFrame | None = None,
    expected_pair_count: int = DEFAULT_PAIR_COUNT,
    expected_session_count: int = DEFAULT_SESSION_COUNT,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    reuse_cached_new_metrics: bool = True,
) -> Path:
    """Run independent Q3 analysis for completed stages and write lineage.

    ``stage_pair_metrics`` is an injection boundary for tests.  Production
    callers omit it and checkpoints are evaluated directly.  Injected frames
    should contain only the newly trained jobs; stage evidence is composed
    here in the same way as production evidence.
    """

    root = Path(experiment_root).resolve()
    requested = tuple(stages or STAGE_ORDER)
    if not requested or any(stage not in STAGE_ORDER for stage in requested):
        raise CoverageSweepAnalysisError("Invalid stage selection")
    if tuple(sorted(requested, key=STAGE_ORDER.index)) != requested:
        raise CoverageSweepAnalysisError("Stages must be analysed in declared order")

    injected = stage_pair_metrics is not None
    new_metrics: dict[str, pd.DataFrame] = {}
    panel_lineage: dict[str, Any] = {}
    for stage_id in requested:
        if injected:
            if stage_id not in stage_pair_metrics:
                raise CoverageSweepAnalysisError(f"Injected metrics missing {stage_id}")
            frame = _normalise_pair_metrics(
                stage_pair_metrics[stage_id],
                evidence_source="injected_new_stage_fixture",
            )
            frame["stage_id"] = stage_id
            new_metrics[stage_id] = frame
            panel_lineage[stage_id] = {
                "injected": True,
                "q4_rows_passed_to_evaluator": 0,
            }
        else:
            cached = (
                _validated_cached_new_metrics(root, stage_id)
                if reuse_cached_new_metrics
                else None
            )
            if cached is not None:
                new_metrics[stage_id] = cached
                panel_lineage[stage_id] = {
                    "reused_hashed_q3_pair_cache": True,
                    "q4_rows_passed_to_evaluator": 0,
                }
            else:
                frame, lineage = evaluate_new_stage_pair_metrics(root, stage_id)
                new_metrics[stage_id] = frame
                panel_lineage[stage_id] = lineage
        stage_dir = _stage_artifact_dir(root, stage_id)
        _write_csv(
            new_metrics[stage_id],
            stage_dir / "new_q3_pair_metrics.csv.gz",
            compression="gzip",
        )

    # Stage 2 completes a grid whose 10m/15m current/real cells were trained in
    # Stage 1.  Permit a genuinely stage-local analysis invocation by loading
    # the already hashed Stage-1 cache instead of re-running 72 checkpoints.
    if STAGE_2 in requested and STAGE_1 not in new_metrics:
        dependency = _stage_artifact_dir(root, STAGE_1) / "new_q3_pair_metrics.csv.gz"
        if not dependency.is_file():
            raise CoverageSweepAnalysisError(
                "Stage 2 analysis requires the completed Stage-1 Q3 pair cache"
            )
        cached = _normalise_pair_metrics(
            pd.read_csv(dependency, low_memory=False),
            evidence_source="cached_stage_1_q3_pairs",
        )
        cached["stage_id"] = STAGE_1
        new_metrics[STAGE_1] = cached

    reference_lineage: dict[str, Any] = {}
    fixed_reference = fixed_reference_pair_metrics
    local_reference = local_lr_reference_pair_metrics
    needs_references = any(stage in {STAGE_2, STAGE_4} for stage in requested)
    if needs_references and (fixed_reference is None or local_reference is None):
        if injected:
            if fixed_reference is None:
                raise CoverageSweepAnalysisError(
                    "Injected Stage 2/4 analysis requires fixed reference metrics"
                )
            if STAGE_4 in requested and local_reference is None:
                raise CoverageSweepAnalysisError(
                    "Injected Stage 4 analysis requires local-LR reference metrics"
                )
        else:
            fixed_root, local_root = _discover_reference_roots(
                root, fixed_reference_root, local_lr_reference_root
            )
            fixed_reference, local_reference, reference_lineage = (
                _reference_pair_metrics(fixed_root, local_root)
            )
            _validate_declared_reference_lineage(root, reference_lineage)
    if fixed_reference is not None:
        fixed_reference = _normalise_pair_metrics(
            fixed_reference,
            evidence_source="fixed_lr_capacity_reference",
            forced_model="regression",
        )
    if local_reference is not None:
        local_reference = _normalise_pair_metrics(
            local_reference,
            evidence_source="local_lr_large_reference",
            forced_model="regression",
            forced_profile="large",
        )

    all_seed_runs: list[pd.DataFrame] = []
    all_across: list[pd.DataFrame] = []
    all_scores: list[pd.DataFrame] = []
    all_bootstrap: list[pd.DataFrame] = []
    selections: dict[str, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}

    for stage_offset, stage_id in enumerate(requested):
        evidence = compose_stage_evidence(
            stage_id,
            new_metrics,
            fixed_reference=fixed_reference,
            local_lr_reference=local_reference,
        )
        evidence = validate_stage_pair_metrics(
            stage_id,
            evidence,
            expected_pair_count=expected_pair_count,
            expected_session_count=expected_session_count,
        )
        seed_runs = build_seed_run_metrics(evidence)
        across = summarize_across_seeds(seed_runs)
        scores = build_stage_scores(stage_id, seed_runs)
        bootstrap = build_stage_bootstrap(
            stage_id,
            evidence,
            scores,
            iterations=bootstrap_iterations,
            random_seed=bootstrap_seed + stage_offset * 10_000,
        )
        selection = build_stage_selection(stage_id, scores)
        stage_dir = _stage_artifact_dir(root, stage_id)
        paths = {
            "new_pair_metrics": stage_dir / "new_q3_pair_metrics.csv.gz",
            "pair_metrics": _write_csv(
                evidence,
                stage_dir / "q3_pair_metrics.csv.gz",
                compression="gzip",
            ),
            "seed_run_metrics": _write_csv(
                seed_runs, stage_dir / "q3_seed_run_metrics.csv"
            ),
            "across_seed_summary": _write_csv(
                across, stage_dir / "q3_across_seed_summary.csv"
            ),
            "scores": _write_csv(scores, stage_dir / "q3_scores.csv"),
            "bootstrap": _write_csv(bootstrap, stage_dir / "q3_bootstrap.csv"),
            "selection": _write_json(stage_dir / "q3_selection.json", selection),
        }
        validations[stage_id] = {
            "stage_id": stage_id,
            "stage_index": STAGE_CONTRACTS[stage_id].stage_index,
            "new_job_count": STAGE_CONTRACTS[stage_id].new_job_count,
            "evidence_job_count": STAGE_CONTRACTS[stage_id].evidence_job_count,
            "pair_count_per_job": int(expected_pair_count),
            "session_count_per_job": int(expected_session_count),
            "pair_metric_rows": len(evidence),
            "seed_run_rows": len(seed_runs),
            "across_seed_rows": len(across),
            "score_rows": len(scores),
            "bootstrap_rows": len(bootstrap),
            "combination_key": list(COMBINATION_KEY),
            "three_seed_complete": True,
            "paired_common_q3_panel": True,
            "q4_used_for_selection": False,
            "q4_predictions_generated": False,
            "minimum_improvement_gate_applied": False,
            "artifacts": {
                label: {"path": str(path), "sha256": _sha256(path)}
                for label, path in paths.items()
            },
        }
        _write_json(stage_dir / "validation_summary.json", validations[stage_id])
        selections[stage_id] = selection
        all_seed_runs.append(seed_runs)
        all_across.append(across)
        all_scores.append(scores)
        all_bootstrap.append(bootstrap)

    analysis_dir = root / "analysis"
    combined_paths = {
        "seed_run_metrics": _write_csv(
            pd.concat(all_seed_runs, ignore_index=True),
            analysis_dir / "coverage_q3_seed_run_metrics.csv",
        ),
        "across_seed_summary": _write_csv(
            pd.concat(all_across, ignore_index=True),
            analysis_dir / "coverage_q3_across_seed_summary.csv",
        ),
        "scores": _write_csv(
            pd.concat(all_scores, ignore_index=True),
            analysis_dir / "coverage_q3_scores.csv",
        ),
        "bootstrap": _write_csv(
            pd.concat(all_bootstrap, ignore_index=True),
            analysis_dir / "coverage_q3_bootstrap.csv",
        ),
    }
    final_candidate_path: Path | None = None
    if requested == STAGE_ORDER:
        final_candidate_path = _write_json(
            root / "coverage_final_candidate.json", _final_candidate(selections)
        )

    validation = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "experiment_kind": EXPERIMENT_KIND,
        "analysed_stages": list(requested),
        "all_four_stages_complete": requested == STAGE_ORDER,
        "selection_data_scope": "Q3 common_validation_05m only",
        "q3_start_utc": Q3_START_UTC,
        "q3_end_utc_exclusive": Q3_END_UTC,
        "q4_used_for_selection": False,
        "q4_predictions_generated": False,
        "q4_evaluation_permitted": False,
        "minimum_improvement_gate_applied": False,
        "numerical_tie_policy": {
            "mae_gap_tolerance_iv": MAE_NUMERICAL_TIE_TOLERANCE,
            "tie_definition": "abs(model_mae - persistence_mae) <= tolerance",
            "pair_win_definition": (
                "canonical pair win when present; otherwise "
                "model_mae - persistence_mae < -tolerance"
            ),
        },
        "combination_key": list(COMBINATION_KEY),
        "bootstrap_method": "seed_then_paired_CME_session_cluster",
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "stage_validations": validations,
        "stage_selections": selections,
        "panel_lineage": panel_lineage,
        "reference_lineage": reference_lineage,
        "combined_artifacts": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in combined_paths.items()
        },
        "final_candidate": (
            None
            if final_candidate_path is None
            else {
                "path": str(final_candidate_path),
                "sha256": _sha256(final_candidate_path),
            }
        ),
        "task_semantics_warning": (
            "5m/10m/15m/30m are cumulative news-wait limits and change the "
            "training task composition. All retain a fixed current-5m to "
            "target-5m market horizon and must not be treated as four independent "
            "copies of one task."
        ),
    }
    validation["payload_sha256"] = _payload_sha256(validation)
    return _write_json(analysis_dir / "coverage_validation_summary.json", validation)


__all__ = [
    "ALL_TEXT_MODES",
    "COMBINATION_KEY",
    "CoverageSweepAnalysisError",
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_BOOTSTRAP_SEED",
    "FIXED_LEARNING_RATE",
    "LEARNING_RATES",
    "MAE_NUMERICAL_TIE_TOLERANCE",
    "PROFILES",
    "SEEDS",
    "STAGE_1",
    "STAGE_2",
    "STAGE_3",
    "STAGE_4",
    "STAGE_CONTRACTS",
    "STAGE_ORDER",
    "build_seed_run_metrics",
    "build_stage_bootstrap",
    "build_stage_scores",
    "build_stage_selection",
    "compose_stage_evidence",
    "evaluate_new_stage_pair_metrics",
    "run_coverage_sweep_analysis",
    "summarize_across_seeds",
    "validate_new_job_matrix",
    "validate_stage_pair_metrics",
]
