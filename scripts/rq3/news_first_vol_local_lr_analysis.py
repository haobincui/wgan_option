"""Q3-only multi-seed analysis for the local low-learning-rate experiment.

The experiment deliberately treats training seeds and CME sessions as two
distinct uncertainty levels.  Every reported model metric is first computed
at unique market-pair grain.  Descriptive tables then give the mean and sample
standard deviation across seeds, while inferential contrasts resample seeds
and, within every sampled seed, complete CME sessions.

There is intentionally no Q4 entry point in this module.
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

from scripts.rq3.news_first_vol_comparison_analysis import (
    RunSpec,
    TrainedRunEvaluator,
    _load_panel_source,
    aggregate_pair_metrics,
    compute_sample_metrics,
    holm_adjust,
)
from scripts.rq3.news_first_vol_local_lr_sweep import (
    EXPERIMENT_KIND,
    EXPERIMENT_STAGE,
    FROZEN_LEARNING_RATES,
    FROZEN_SCHEDULER_MIN_LRS,
    FROZEN_SEEDS,
    FROZEN_TEXT_MODES,
    FROZEN_TOLERANCES,
    LR_PROFILE_IDS,
    _lr_profile_sha256,
    _lr_seed_profile_sha256,
    _validate_registry,
)


Q3_START_UTC = "2023-07-01T00:00:00Z"
Q3_END_UTC = "2023-10-01T00:00:00Z"
SELECTION_PANEL = "common_validation_05m"
EXPECTED_LEARNING_RATES = tuple(FROZEN_LEARNING_RATES.values())
EXPECTED_SEEDS = FROZEN_SEEDS
EXPECTED_TEXT_MODES = FROZEN_TEXT_MODES
EXPECTED_TOLERANCES = FROZEN_TOLERANCES
SELECTION_MODE = "current_only"
GATE_MIN_IMPROVEMENT = 0.005
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260819


class LocalLearningRateAnalysisError(ValueError):
    """Raised when the local-LR evidence cannot support a valid comparison."""


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
        raise LocalLearningRateAnalysisError(f"Expected a JSON object: {path}")
    return dict(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise LocalLearningRateAnalysisError(f"Expected a YAML mapping: {path}")
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
    numeric = pd.to_numeric(value, errors="coerce")
    try:
        result = float(numeric)
    except (TypeError, ValueError):
        result = float("nan")
    if not math.isfinite(result):
        raise LocalLearningRateAnalysisError(f"{label} must be finite")
    return result


def _positive(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if result <= 0.0:
        raise LocalLearningRateAnalysisError(f"{label} must be positive")
    return result


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


def _load_jobs(root: Path) -> list[dict[str, Any]]:
    registry = _read_json(root / "registry" / "jobs.json")
    if str(registry.get("experiment_kind", "")) != EXPERIMENT_KIND:
        raise LocalLearningRateAnalysisError(
            "Registry is not a local multi-seed learning-rate experiment"
        )
    raw_jobs = registry.get("jobs")
    if not isinstance(raw_jobs, list):
        raise LocalLearningRateAnalysisError("Registry jobs must be a list")
    jobs: list[dict[str, Any]] = []
    for raw in raw_jobs:
        if not isinstance(raw, Mapping):
            raise LocalLearningRateAnalysisError("Every registry job must be an object")
        job = dict(raw)
        job_id = str(job.get("job_id", "")).strip()
        if not job_id:
            raise LocalLearningRateAnalysisError("Every local-LR job requires job_id")
        status_path = root / "registry" / "jobs" / f"{job_id}.status.json"
        if status_path.is_file():
            job.update(_read_json(status_path))
        jobs.append(job)
    return jobs


def _job_profile(job: Mapping[str, Any]) -> str:
    return str(job.get("lr_profile", job.get("learning_rate_profile", ""))).strip()


def _job_rate(job: Mapping[str, Any]) -> float:
    return _positive(
        job.get("initial_learning_rate", job.get("learning_rate")),
        label=f"initial learning rate for {job.get('job_id')}",
    )


def _job_seed(job: Mapping[str, Any]) -> int:
    value = pd.to_numeric(job.get("seed"), errors="coerce")
    if not math.isfinite(float(value)):
        raise LocalLearningRateAnalysisError(
            f"Registry job lacks seed lineage: {job.get('job_id')}"
        )
    return int(value)


def _job_mode(job: Mapping[str, Any]) -> str:
    return str(job.get("text_ablation_mode", "")).strip().lower()


def _job_tolerance(job: Mapping[str, Any]) -> int:
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
    epoch = pd.to_numeric(payload.get("best_epoch"), errors="coerce")
    if not math.isfinite(float(epoch)) or int(epoch) < 1:
        raise LocalLearningRateAnalysisError(
            f"best_learned checkpoint must select epoch >=1: {path}"
        )
    if (
        str(payload.get("selection_scope", "trained_epochs_only"))
        != "trained_epochs_only"
    ):
        raise LocalLearningRateAnalysisError(
            f"best_learned checkpoint has invalid selection scope: {path}"
        )
    return path, payload


def _learned_checkpoint(job: Mapping[str, Any], payload: Mapping[str, Any]) -> Path:
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, Mapping):
        for name in ("model", "checkpoint"):
            raw = str(artifacts.get(name, "")).strip()
            if raw:
                path = Path(raw)
                if not path.is_absolute():
                    path = _run_dir(job) / path
                if path.is_file():
                    return path.resolve()
    path = _run_dir(job) / "checkpoints" / "vol_regressor_best_learned.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _profile_manifest(root: Path) -> pd.DataFrame:
    candidates = (
        root / "lr_seed_profile_manifest.csv",
        root / "local_lr_profile_manifest.csv",
        root / "lr_profile_manifest.csv",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(candidates[0])
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "lr_profile",
        "lr_profile_sha256",
        "lr_seed_profile_sha256",
        "initial_learning_rate",
        "scheduler_min_lr",
        "seed",
        "capacity_profile_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LocalLearningRateAnalysisError(f"LR profile manifest missing {missing}")
    manifest_keys = frame[["lr_profile", "seed"]]
    if len(frame) != 15 or manifest_keys.duplicated().any():
        raise LocalLearningRateAnalysisError(
            "LR seed-profile manifest must contain 15 unique profile/seed rows"
        )
    expected_keys = {
        (profile, seed) for profile in LR_PROFILE_IDS for seed in EXPECTED_SEEDS
    }
    observed_keys = set(
        manifest_keys.assign(
            seed=pd.to_numeric(manifest_keys["seed"], errors="coerce")
        ).itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        raise LocalLearningRateAnalysisError(
            "LR seed-profile manifest differs from the frozen profile/seed matrix"
        )
    for row in frame.to_dict(orient="records"):
        profile = str(row["lr_profile"])
        seed = int(row["seed"])
        capacity_sha = str(row["capacity_profile_sha256"])
        expected_lr_sha = _lr_profile_sha256(
            profile, capacity_profile_sha256=capacity_sha
        )
        expected_seed_sha = _lr_seed_profile_sha256(
            profile, seed, capacity_profile_sha256=capacity_sha
        )
        contracts = {
            "lr_profile_sha256": expected_lr_sha,
            "lr_seed_profile_sha256": expected_seed_sha,
            "initial_learning_rate": FROZEN_LEARNING_RATES[profile],
            "scheduler_min_lr": FROZEN_SCHEDULER_MIN_LRS[profile],
        }
        for field, expected in contracts.items():
            observed = row[field]
            if isinstance(expected, float):
                valid = _same_float(float(observed), expected)
            else:
                valid = str(observed) == str(expected)
            if not valid:
                raise LocalLearningRateAnalysisError(
                    f"LR seed-profile manifest mismatch: {profile}/{seed}/{field}"
                )
    return frame.copy()


def validate_job_matrix(jobs: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed unless all 60 seed-preserving jobs are complete and unique."""

    completed = [
        job for job in jobs if str(job.get("status", "")).lower() == "completed"
    ]
    for job in completed:
        if str(job.get("experiment_stage", "")) != EXPERIMENT_STAGE:
            raise LocalLearningRateAnalysisError(
                f"Local-LR stage mismatch for {job.get('job_id')}"
            )
    keys = [
        (_job_rate(job), _job_seed(job), _job_mode(job), _job_tolerance(job))
        for job in completed
    ]
    expected = {
        (rate, seed, mode, tolerance)
        for rate in EXPECTED_LEARNING_RATES
        for seed in EXPECTED_SEEDS
        for mode in EXPECTED_TEXT_MODES
        for tolerance in EXPECTED_TOLERANCES
    }
    observed = set(keys)
    if len(keys) != len(observed):
        raise LocalLearningRateAnalysisError(
            "Duplicate local-LR job key; seed must remain part of every key"
        )
    if observed != expected:
        raise LocalLearningRateAnalysisError(
            "Local-LR job matrix is incomplete; "
            f"missing={sorted(expected - observed)[:6]}, "
            f"extra={sorted(observed - expected)[:6]}"
        )
    if len(completed) != len(jobs):
        incomplete = [
            f"{job.get('job_id')}={job.get('status', 'missing')}"
            for job in jobs
            if str(job.get("status", "")).lower() != "completed"
        ]
        raise LocalLearningRateAnalysisError(
            f"Local-LR analysis requires all jobs completed: {incomplete[:6]}"
        )


def collect_run_metrics(experiment_root: str | Path) -> pd.DataFrame:
    """Collect Q3 best-learned metrics with seed and optimizer lineage."""

    root = Path(experiment_root).resolve()
    _validate_registry(root)
    jobs = _load_jobs(root)
    validate_job_matrix(jobs)
    manifest = _profile_manifest(root).set_index(["lr_profile", "seed"])
    rows: list[dict[str, Any]] = []
    for job in jobs:
        profile = _job_profile(job)
        seed = _job_seed(job)
        manifest_key = (profile, seed)
        if manifest_key not in manifest.index:
            raise LocalLearningRateAnalysisError(f"Unknown LR profile: {profile}")
        rate = _job_rate(job)
        declared = manifest.loc[manifest_key]
        if not _same_float(rate, float(declared["initial_learning_rate"])):
            raise LocalLearningRateAnalysisError(
                f"Registry/manifest learning-rate mismatch for {job.get('job_id')}"
            )
        profile_sha = str(job.get("lr_profile_sha256", "")).strip()
        if profile_sha != str(declared["lr_profile_sha256"]).strip():
            raise LocalLearningRateAnalysisError(
                f"Registry/manifest LR hash mismatch for {job.get('job_id')}"
            )
        seed_profile_sha = str(job.get("lr_seed_profile_sha256", "")).strip()
        if seed_profile_sha != str(declared["lr_seed_profile_sha256"]).strip():
            raise LocalLearningRateAnalysisError(
                f"Registry/manifest LR+seed hash mismatch for {job.get('job_id')}"
            )
        config_path, config = _training_config(job)
        if _sha256(config_path) != str(job.get("config_sha256", "")):
            raise LocalLearningRateAnalysisError(
                f"Training config hash mismatch for {job.get('job_id')}"
            )
        if int(config.get("seed", -1)) != seed:
            raise LocalLearningRateAnalysisError(
                f"Registry/training seed mismatch for {job.get('job_id')}"
            )
        if not _same_float(
            _positive(config.get("learning_rate"), label="training learning rate"),
            rate,
        ):
            raise LocalLearningRateAnalysisError(
                f"Registry/training LR mismatch for {job.get('job_id')}"
            )
        if str(config.get("support_mask_mode", "")).lower() != "raw_joint":
            raise LocalLearningRateAnalysisError("Local-LR analysis requires raw_joint")
        if str(config.get("news_first_train_end_utc", Q3_START_UTC)) != Q3_START_UTC:
            raise LocalLearningRateAnalysisError("Train/Q3 boundary drift")
        if str(config.get("news_first_validation_end_utc", Q3_END_UTC)) != Q3_END_UTC:
            raise LocalLearningRateAnalysisError("Q3/Q4 boundary drift")
        learned_path, learned = _learned_metadata(job)
        if int(learned.get("seed", -1)) != seed:
            raise LocalLearningRateAnalysisError(
                f"Checkpoint seed mismatch: {learned_path}"
            )
        if str(learned.get("lr_profile", profile)) != profile:
            raise LocalLearningRateAnalysisError(
                f"Checkpoint LR profile mismatch: {learned_path}"
            )
        if str(learned.get("lr_profile_sha256", profile_sha)) != profile_sha:
            raise LocalLearningRateAnalysisError(
                f"Checkpoint LR hash mismatch: {learned_path}"
            )
        model_mae = _positive(
            _first_metric(learned, ("masked_pair_balanced_mae", "val_recon")),
            label="best-learned Q3 MAE",
        )
        persistence_mae = _positive(
            _first_metric(
                learned,
                ("persistence_masked_pair_balanced_mae", "val_current_recon"),
            ),
            label="Q3 persistence MAE",
        )
        ratio = model_mae / persistence_mae
        rows.append(
            {
                "job_id": str(job["job_id"]),
                "model_family": "regression",
                "capacity_profile": str(job.get("capacity_profile", "")),
                "lr_profile": profile,
                "lr_profile_sha256": profile_sha,
                "lr_seed_profile_sha256": seed_profile_sha,
                "initial_learning_rate": rate,
                "seed": seed,
                "text_ablation_mode": _job_mode(job),
                "tolerance_minutes": _job_tolerance(job),
                "best_learned_epoch": int(learned["best_epoch"]),
                "model_mae": model_mae,
                "persistence_mae": persistence_mae,
                "mae_gap": model_mae - persistence_mae,
                "mae_ratio": ratio,
                "log_mae_ratio": math.log(ratio),
                "improvement_fraction": 1.0 - ratio,
                "training_config_path": str(config_path),
                "learned_metadata_path": str(learned_path),
                "learned_metadata_sha256": _sha256(learned_path),
                "run_dir": str(_run_dir(job)),
            }
        )
    frame = pd.DataFrame(rows)
    key = [
        "lr_profile",
        "seed",
        "text_ablation_mode",
        "tolerance_minutes",
    ]
    if frame.duplicated(key).any():
        raise LocalLearningRateAnalysisError(
            "Collected run metrics lost seed uniqueness"
        )
    return frame.sort_values(
        ["initial_learning_rate", "seed", "text_ablation_mode", "tolerance_minutes"],
        kind="stable",
    ).reset_index(drop=True)


def _run_spec(job: Mapping[str, Any]) -> RunSpec:
    learned_path, learned = _learned_metadata(job)
    config_path, config = _training_config(job)
    return RunSpec(
        run_id=str(job["job_id"]),
        run_dir=_run_dir(job),
        model="regression",
        tolerance_minutes=_job_tolerance(job),
        seed=_job_seed(job),
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
        raise LocalLearningRateAnalysisError(
            f"Local-LR jobs disagree on the common Q3 panel: {sorted(sources)}"
        )
    raw_path, sheet_name, support_mode = next(iter(sources))
    workbook = Path(raw_path).resolve()
    raw = pd.read_excel(workbook, sheet_name=sheet_name)
    if "effective_origin_utc" not in raw.columns:
        raise LocalLearningRateAnalysisError("Q3 panel lacks effective_origin_utc")
    timestamps = pd.to_datetime(raw["effective_origin_utc"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise LocalLearningRateAnalysisError("Q3 panel contains invalid timestamps")
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
        raise LocalLearningRateAnalysisError("A non-Q3 row reached local-LR evaluation")
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
    """Evaluate all 60 best-learned checkpoints on the common Q3 panel."""

    root = Path(experiment_root).resolve()
    _validate_registry(root)
    jobs = _load_jobs(root)
    validate_job_matrix(jobs)
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
            raise LocalLearningRateAnalysisError(
                f"Q3 prediction exclusions for {spec.run_id}: "
                f"{exclusions['exclusion_code'].value_counts().to_dict()}"
            )
        pairs = aggregate_pair_metrics(samples)
        pairs = pairs[
            pairs["stratum_type"].astype(str).eq("overall")
            & pairs["stratum_value"].astype(str).eq("all")
        ].copy()
        if not pairs["seed"].astype(int).eq(_job_seed(job)).all():
            raise LocalLearningRateAnalysisError(
                f"Pair metrics lost seed lineage for {job.get('job_id')}"
            )
        learned_path, learned = _learned_metadata(job)
        expected_model = _positive(
            _first_metric(learned, ("masked_pair_balanced_mae", "val_recon")),
            label="metadata model MAE",
        )
        expected_persistence = _positive(
            _first_metric(
                learned,
                ("persistence_masked_pair_balanced_mae", "val_current_recon"),
            ),
            label="metadata persistence MAE",
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
                raise LocalLearningRateAnalysisError(
                    f"Pair-balanced {label} MAE disagrees with metadata for "
                    f"{job.get('job_id')}: {actual} != {expected}"
                )
        pairs.insert(0, "lr_profile", _job_profile(job))
        pairs.insert(1, "lr_profile_sha256", str(job.get("lr_profile_sha256", "")))
        pairs.insert(
            2,
            "lr_seed_profile_sha256",
            str(job.get("lr_seed_profile_sha256", "")),
        )
        pairs.insert(3, "initial_learning_rate", _job_rate(job))
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
    """Validate exact seed-aware coverage and return a normalized copy."""

    required = {
        "lr_profile",
        "lr_profile_sha256",
        "lr_seed_profile_sha256",
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
        raise LocalLearningRateAnalysisError(f"Q3 pair metrics missing {missing}")
    frame = pair_metrics.copy()
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce")
    frame["initial_learning_rate"] = pd.to_numeric(
        frame["initial_learning_rate"], errors="coerce"
    )
    frame["tolerance_minutes"] = pd.to_numeric(
        frame["tolerance_minutes"], errors="coerce"
    )
    for column in ("model_mae", "persistence_mae"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all() or bool(
            (frame[column] <= 0.0).any()
        ):
            raise LocalLearningRateAnalysisError(
                f"{column} must be finite and positive"
            )
    if set(frame["seed"].astype(int)) != set(EXPECTED_SEEDS):
        raise LocalLearningRateAnalysisError(
            "Pair metrics do not contain all three seeds"
        )
    if set(frame["text_ablation_mode"].astype(str)) != set(EXPECTED_TEXT_MODES):
        raise LocalLearningRateAnalysisError(
            "Pair metrics text modes differ from contract"
        )
    if set(frame["tolerance_minutes"].astype(int)) != set(EXPECTED_TOLERANCES):
        raise LocalLearningRateAnalysisError(
            "Pair metrics tolerances differ from contract"
        )
    actual_rates = sorted(frame["initial_learning_rate"].drop_duplicates())
    if len(actual_rates) != len(EXPECTED_LEARNING_RATES) or any(
        not _same_float(actual, expected)
        for actual, expected in zip(actual_rates, sorted(EXPECTED_LEARNING_RATES))
    ):
        raise LocalLearningRateAnalysisError(
            "Pair metrics learning rates differ from contract"
        )
    if set(frame["lr_profile"].astype(str)) != set(LR_PROFILE_IDS):
        raise LocalLearningRateAnalysisError(
            "Pair metrics LR profile IDs differ from contract"
        )
    profile_rates = frame.groupby("lr_profile")["initial_learning_rate"].nunique()
    if bool((profile_rates != 1).any()):
        raise LocalLearningRateAnalysisError("An LR profile maps to multiple rates")
    for profile, expected_rate in FROZEN_LEARNING_RATES.items():
        observed_rate = float(
            frame.loc[
                frame["lr_profile"].astype(str).eq(profile), "initial_learning_rate"
            ].iloc[0]
        )
        if not _same_float(observed_rate, expected_rate):
            raise LocalLearningRateAnalysisError(f"LR profile/rate mismatch: {profile}")
    seed_hash_counts = frame.groupby(["lr_profile", "seed"])[
        "lr_seed_profile_sha256"
    ].nunique()
    if (
        bool((seed_hash_counts != 1).any())
        or frame["lr_seed_profile_sha256"].astype(str).str.strip().eq("").any()
    ):
        raise LocalLearningRateAnalysisError("LR+seed hash lineage is incomplete")
    for (profile, seed), group in frame.groupby(["lr_profile", "seed"], sort=False):
        expected_profile_sha = _lr_profile_sha256(str(profile))
        expected_seed_sha = _lr_seed_profile_sha256(str(profile), int(seed))
        if set(group["lr_profile_sha256"].astype(str)) != {expected_profile_sha}:
            raise LocalLearningRateAnalysisError(
                f"LR profile hash mismatch in pair metrics: {profile}"
            )
        if set(group["lr_seed_profile_sha256"].astype(str)) != {expected_seed_sha}:
            raise LocalLearningRateAnalysisError(
                f"LR+seed hash mismatch in pair metrics: {profile}/{seed}"
            )

    job_key = [
        "lr_profile",
        "seed",
        "text_ablation_mode",
        "tolerance_minutes",
    ]
    expected_jobs = len(EXPECTED_LEARNING_RATES) * len(EXPECTED_SEEDS) * 2 * 2
    if frame[job_key].drop_duplicates().shape[0] != expected_jobs:
        raise LocalLearningRateAnalysisError(
            "Pair metrics do not preserve the exact 60-job seed-aware matrix"
        )
    pair_key = job_key + ["pair_id"]
    if frame.duplicated(pair_key).any():
        raise LocalLearningRateAnalysisError("Duplicate seed-aware pair metric key")
    sessions_per_pair = frame.groupby(pair_key, dropna=False)["session_id"].nunique()
    if bool((sessions_per_pair != 1).any()):
        raise LocalLearningRateAnalysisError("A pair maps to multiple CME sessions")

    coverage_sets = []
    for _, group in frame.groupby(job_key, sort=False):
        coverage_sets.append(
            frozenset(
                group[["pair_id", "session_id"]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
        )
    if len(set(coverage_sets)) != 1:
        raise LocalLearningRateAnalysisError(
            "Paired LR inference requires identical pair/session coverage for all jobs"
        )
    return frame.reset_index(drop=True)


def build_seed_run_metrics(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse Q3 pairs to one row per LR/seed/mode/tolerance run."""

    frame = validate_pair_metric_matrix(pair_metrics)
    keys = [
        "lr_profile",
        "lr_profile_sha256",
        "lr_seed_profile_sha256",
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
    if len(output) != 60:
        raise LocalLearningRateAnalysisError(
            f"Expected 60 seed-run rows, got {len(output)}"
        )
    return output.sort_values(
        ["initial_learning_rate", "seed", "text_ablation_mode", "tolerance_minutes"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_across_seeds(seed_runs: pd.DataFrame) -> pd.DataFrame:
    """Report mean and sample SD across the three training seeds."""

    required = {
        "lr_profile",
        "lr_profile_sha256",
        "initial_learning_rate",
        "seed",
        "text_ablation_mode",
        "tolerance_minutes",
        "model_mae",
        "persistence_mae",
        "mae_gap",
        "mae_ratio",
        "log_mae_ratio",
        "improvement_fraction",
        "pair_win_rate",
        "pair_count",
        "session_count",
    }
    missing = sorted(required - set(seed_runs.columns))
    if missing:
        raise LocalLearningRateAnalysisError(f"Seed-run metrics missing {missing}")
    keys = [
        "lr_profile",
        "lr_profile_sha256",
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
            raise LocalLearningRateAnalysisError(
                f"Across-seed summary requires seeds {EXPECTED_SEEDS}, got {seeds}"
            )
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
            numeric = pd.to_numeric(group[metric], errors="coerce").to_numpy(
                dtype=float
            )
            if not np.isfinite(numeric).all():
                raise LocalLearningRateAnalysisError(
                    f"Non-finite seed metric: {metric}"
                )
            row[f"{metric}_mean"] = float(numeric.mean())
            row[f"{metric}_sd"] = float(numeric.std(ddof=1))
        rows.append(row)
    output = pd.DataFrame(rows)
    if len(output) != 20:
        raise LocalLearningRateAnalysisError(
            f"Expected 20 across-seed cells, got {len(output)}"
        )
    return output.sort_values(
        ["initial_learning_rate", "text_ablation_mode", "tolerance_minutes"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_lr_scores(seed_runs: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight 5m/30m score, then mean and SD across seeds."""

    per_seed_rows: list[dict[str, Any]] = []
    keys = [
        "lr_profile",
        "lr_profile_sha256",
        "initial_learning_rate",
        "seed",
        "text_ablation_mode",
    ]
    for values, group in seed_runs.groupby(keys, sort=True, dropna=False):
        tolerances = tuple(sorted(group["tolerance_minutes"].astype(int).tolist()))
        if tolerances != EXPECTED_TOLERANCES:
            raise LocalLearningRateAnalysisError(
                f"LR score requires both tolerances, got {tolerances}"
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
        "lr_profile",
        "lr_profile_sha256",
        "initial_learning_rate",
        "text_ablation_mode",
    ]
    for values, group in per_seed.groupby(outer_keys, sort=True, dropna=False):
        seeds = tuple(sorted(group["seed"].astype(int).tolist()))
        if seeds != EXPECTED_SEEDS:
            raise LocalLearningRateAnalysisError("LR score lost a training seed")
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
    return output.sort_values(
        ["text_ablation_mode", "mean_log_mae_ratio", "initial_learning_rate"],
        kind="stable",
    ).reset_index(drop=True)


def _two_level_bootstrap(
    differences: pd.DataFrame,
    *,
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    """Bootstrap seeds, then paired CME-session clusters within each seed."""

    if int(iterations) < 2:
        raise LocalLearningRateAnalysisError("Bootstrap iterations must be >=2")
    required = {"seed", "session_id", "pair_id", "difference"}
    missing = sorted(required - set(differences.columns))
    if missing:
        raise LocalLearningRateAnalysisError(f"Bootstrap differences missing {missing}")
    frame = differences.copy()
    frame["difference"] = pd.to_numeric(frame["difference"], errors="coerce")
    if not np.isfinite(frame["difference"].to_numpy(dtype=float)).all():
        raise LocalLearningRateAnalysisError("Bootstrap differences must be finite")
    seeds = tuple(sorted(frame["seed"].astype(int).unique().tolist()))
    if seeds != EXPECTED_SEEDS:
        raise LocalLearningRateAnalysisError(
            f"Two-level bootstrap requires seeds {EXPECTED_SEEDS}, got {seeds}"
        )
    extra_keys = ["tolerance_minutes"] if "tolerance_minutes" in frame.columns else []
    duplicate_key = ["seed", "session_id", "pair_id", *extra_keys]
    if frame.duplicated(duplicate_key).any():
        raise LocalLearningRateAnalysisError(
            "Bootstrap input has duplicate seed/session/pair comparison rows"
        )

    coverage: list[frozenset[tuple[str, ...]]] = []
    session_arrays: dict[int, tuple[np.ndarray, np.ndarray, int, int]] = {}
    seed_points: list[float] = []
    for seed in seeds:
        seed_frame = frame[frame["seed"].astype(int).eq(seed)].copy()
        coverage_columns = ["session_id", "pair_id", *extra_keys]
        coverage.append(
            frozenset(
                seed_frame[coverage_columns]
                .astype(str)
                .itertuples(index=False, name=None)
            )
        )
        grouped = seed_frame.groupby("session_id", sort=True).agg(
            difference_sum=("difference", "sum"),
            row_count=("difference", "size"),
        )
        sums = grouped["difference_sum"].to_numpy(dtype=float)
        counts = grouped["row_count"].to_numpy(dtype=float)
        session_arrays[seed] = (sums, counts, len(grouped), len(seed_frame))
        seed_points.append(float(sums.sum() / counts.sum()))
    if len(set(coverage)) != 1:
        raise LocalLearningRateAnalysisError(
            "Two-level paired bootstrap requires identical coverage across seeds"
        )

    rng = np.random.default_rng(int(random_seed))
    seed_draw_indexes = rng.integers(0, len(seeds), size=(int(iterations), len(seeds)))
    draws = np.empty(int(iterations), dtype=float)
    for draw_index in range(int(iterations)):
        sampled_seed_means: list[float] = []
        for seed_index in seed_draw_indexes[draw_index]:
            selected_seed = seeds[int(seed_index)]
            sums, counts, session_count, _ = session_arrays[selected_seed]
            session_indexes = rng.integers(0, session_count, size=session_count)
            sampled_seed_means.append(
                float(sums[session_indexes].sum() / counts[session_indexes].sum())
            )
        draws[draw_index] = float(np.mean(sampled_seed_means))
    point = float(np.mean(seed_points))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    p_lower = (float(np.sum(draws <= 0.0)) + 1.0) / (len(draws) + 1.0)
    p_upper = (float(np.sum(draws >= 0.0)) + 1.0) / (len(draws) + 1.0)
    return {
        "mean_difference": point,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_two_sided": float(min(1.0, 2.0 * min(p_lower, p_upper))),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(random_seed),
        "seed_count": len(seeds),
        "session_count_per_seed": int(session_arrays[seeds[0]][2]),
        "pair_rows_per_seed": int(session_arrays[seeds[0]][3]),
        "resampling_method": "seed_then_paired_CME_session_cluster",
    }


def _scoped_frames(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
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
    """Build persistence, text, and LR paired two-level bootstrap tables."""

    frame = validate_pair_metric_matrix(pair_metrics)

    persistence_rows: list[dict[str, Any]] = []
    for (profile, rate, mode), group in frame.groupby(
        ["lr_profile", "initial_learning_rate", "text_ablation_mode"], sort=True
    ):
        for scope, scoped in _scoped_frames(group):
            differences = scoped[
                ["seed", "session_id", "pair_id", "tolerance_minutes"]
            ].copy()
            differences["difference"] = scoped["model_mae"].to_numpy(
                dtype=float
            ) - scoped["persistence_mae"].to_numpy(dtype=float)
            persistence_rows.append(
                {
                    "lr_profile": profile,
                    "initial_learning_rate": float(rate),
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
        "lr_profile",
        "initial_learning_rate",
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
    text_paired = real.merge(current, on=merge_keys, how="inner", validate="one_to_one")
    if len(text_paired) != len(real) or len(text_paired) != len(current):
        raise LocalLearningRateAnalysisError("Text contrast lost paired Q3 rows")
    text_rows: list[dict[str, Any]] = []
    for (profile, rate), group in text_paired.groupby(
        ["lr_profile", "initial_learning_rate"], sort=True
    ):
        for scope, scoped in _scoped_frames(group):
            differences = scoped[
                ["seed", "session_id", "pair_id", "tolerance_minutes"]
            ].copy()
            differences["difference"] = scoped["real_text_mae"].to_numpy(
                dtype=float
            ) - scoped["current_only_mae"].to_numpy(dtype=float)
            text_rows.append(
                {
                    "lr_profile": profile,
                    "initial_learning_rate": float(rate),
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

    if ranked_leader_profile not in set(frame["lr_profile"].astype(str)):
        raise LocalLearningRateAnalysisError(
            "Ranked leader is absent from pair metrics"
        )
    current_frame = frame[frame["text_ablation_mode"].eq(SELECTION_MODE)].copy()
    lr_rows: list[dict[str, Any]] = []
    references = (
        ("ranked_leader", ranked_leader_profile),
        ("fixed_prior_center", "lr_1e_06"),
    )
    for reference_kind, reference_profile in references:
        reference = current_frame[current_frame["lr_profile"].eq(reference_profile)]
        if reference.empty:
            raise LocalLearningRateAnalysisError(
                f"LR contrast reference is absent: {reference_profile}"
            )
        candidates = current_frame[~current_frame["lr_profile"].eq(reference_profile)]
        for (profile, rate), candidate in candidates.groupby(
            ["lr_profile", "initial_learning_rate"], sort=True
        ):
            lr_keys = ["seed", "tolerance_minutes", "pair_id", "session_id"]
            paired = candidate[lr_keys + ["model_mae"]].merge(
                reference[lr_keys + ["model_mae"]],
                on=lr_keys,
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            if len(paired) != len(candidate) or len(paired) != len(reference):
                raise LocalLearningRateAnalysisError("LR contrast lost paired Q3 rows")
            for scope, scoped in _scoped_frames(paired):
                differences = scoped[
                    ["seed", "session_id", "pair_id", "tolerance_minutes"]
                ].copy()
                differences["difference"] = scoped["model_mae_candidate"].to_numpy(
                    dtype=float
                ) - scoped["model_mae_reference"].to_numpy(dtype=float)
                lr_rows.append(
                    {
                        "candidate_lr_profile": profile,
                        "candidate_learning_rate": float(rate),
                        "reference_kind": reference_kind,
                        "reference_lr_profile": reference_profile,
                        "reference_learning_rate": FROZEN_LEARNING_RATES[
                            reference_profile
                        ],
                        "tolerance_scope": scope,
                        "text_ablation_mode": SELECTION_MODE,
                        "contrast": "candidate_minus_reference",
                        **_two_level_bootstrap(
                            differences,
                            iterations=iterations,
                            random_seed=random_seed + 202,
                        ),
                    }
                )
    lr_contrasts = pd.DataFrame(lr_rows)
    hypothesis_keys = [
        "candidate_lr_profile",
        "reference_lr_profile",
        "tolerance_scope",
    ]
    unique_hypotheses = lr_contrasts.drop_duplicates(hypothesis_keys).copy()
    unique_hypotheses["p_holm"] = holm_adjust(unique_hypotheses["p_two_sided"].tolist())
    lr_contrasts = lr_contrasts.merge(
        unique_hypotheses[hypothesis_keys + ["p_holm"]],
        on=hypothesis_keys,
        how="left",
        validate="many_to_one",
    )
    return persistence, text, lr_contrasts


def build_selection_summary(
    scores: pd.DataFrame,
    *,
    resolved_config_sha256: str = "injected_pair_metrics",
) -> dict[str, Any]:
    current = scores[scores["text_ablation_mode"].eq(SELECTION_MODE)].sort_values(
        ["mean_log_mae_ratio", "initial_learning_rate"], kind="stable"
    )
    if len(current) != len(EXPECTED_LEARNING_RATES):
        raise LocalLearningRateAnalysisError(
            "Current-only LR score matrix is incomplete"
        )
    leader = current.iloc[0]
    gate_passed = bool(leader["gate_passed"])
    result = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "artifact_kind": "multi_seed_local_lr_analysis",
        "resolved_config_sha256": str(resolved_config_sha256),
        "selection_scope": "Q3 common_validation_05m only",
        "selection_mode": SELECTION_MODE,
        "selection_tolerances": list(EXPECTED_TOLERANCES),
        "score_definition": (
            "equal-tolerance mean log(model MAE / persistence MAE), then equal-seed mean"
        ),
        "ranked_leader_lr_profile": str(leader["lr_profile"]),
        "ranked_leader_learning_rate": float(leader["initial_learning_rate"]),
        "ranked_leader_mean_log_mae_ratio": float(leader["mean_log_mae_ratio"]),
        "ranked_leader_sd_log_mae_ratio_across_seeds": float(
            leader["sd_log_mae_ratio_across_seeds"]
        ),
        "ranked_leader_mean_improvement_fraction": float(
            leader["mean_improvement_fraction"]
        ),
        "gate_minimum_mean_improvement_fraction": GATE_MIN_IMPROVEMENT,
        "gate_requires_all_seed_tolerances_not_worse": True,
        "gate_passed": gate_passed,
        "winner_lr_profile": str(leader["lr_profile"]) if gate_passed else "",
        "winner_learning_rate": (
            float(leader["initial_learning_rate"]) if gate_passed else None
        ),
        "candidate_scores": current.to_dict(orient="records"),
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


def run_local_lr_analysis(
    experiment_root: str | Path,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
    q3_pair_metrics: pd.DataFrame | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Path:
    """Write all Q3-only local-LR statistics and validation artifacts."""

    root = Path(experiment_root).resolve()
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    lineage: dict[str, Any]
    if q3_pair_metrics is None:
        run_metrics = collect_run_metrics(root)
        q3_pair_metrics, lineage = evaluate_q3_pair_metrics(root, evaluator=evaluator)
        pair_frame = validate_pair_metric_matrix(q3_pair_metrics)
        derived_runs = build_seed_run_metrics(pair_frame)
        check_keys = [
            "lr_profile",
            "seed",
            "text_ablation_mode",
            "tolerance_minutes",
        ]
        reconciled = run_metrics.merge(
            derived_runs[check_keys + ["model_mae", "persistence_mae"]],
            on=check_keys,
            suffixes=("_metadata", "_pairs"),
            validate="one_to_one",
        )
        for metric in ("model_mae", "persistence_mae"):
            if not np.allclose(
                reconciled[f"{metric}_metadata"],
                reconciled[f"{metric}_pairs"],
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise LocalLearningRateAnalysisError(
                    f"Metadata and Q3 pair metrics disagree on {metric}"
                )
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
        resolved_config_sha256 = "injected_pair_metrics"

    seed_runs = build_seed_run_metrics(pair_frame)
    across_seeds = summarize_across_seeds(seed_runs)
    scores = summarize_lr_scores(seed_runs)
    selection = build_selection_summary(
        scores,
        resolved_config_sha256=resolved_config_sha256,
    )
    persistence, text, lr_contrasts = build_bootstrap_tables(
        pair_frame,
        ranked_leader_profile=str(selection["ranked_leader_lr_profile"]),
        iterations=int(bootstrap_iterations),
        random_seed=int(bootstrap_seed),
    )

    _write_csv(
        pair_frame,
        analysis_dir / "local_lr_q3_pair_metrics.csv.gz",
        compression="gzip",
    )
    _write_csv(seed_runs, analysis_dir / "local_lr_seed_run_metrics.csv")
    _write_csv(across_seeds, analysis_dir / "local_lr_across_seed_summary.csv")
    _write_csv(scores, analysis_dir / "local_lr_scores.csv")
    _write_csv(persistence, analysis_dir / "local_lr_persistence_bootstrap.csv")
    _write_csv(text, analysis_dir / "local_lr_text_bootstrap.csv")
    _write_csv(lr_contrasts, analysis_dir / "local_lr_pairwise_bootstrap.csv")
    _write_json(root / "local_lr_selection.json", selection)

    validation = {
        "schema_version": 1,
        "status": "pass",
        "selection_panel": SELECTION_PANEL,
        "q3_start_utc": Q3_START_UTC,
        "q3_end_utc_exclusive": Q3_END_UTC,
        "q4_used_for_selection": False,
        "q4_predictions_generated": False,
        "q4_rows_passed_to_evaluator": int(
            lineage.get("q4_rows_passed_to_evaluator", 0)
        ),
        "job_count": 60,
        "learning_rate_count": len(EXPECTED_LEARNING_RATES),
        "seed_count": len(EXPECTED_SEEDS),
        "seeds": list(EXPECTED_SEEDS),
        "text_modes": list(EXPECTED_TEXT_MODES),
        "tolerances_minutes": list(EXPECTED_TOLERANCES),
        "pair_count_per_run": int(seed_runs["pair_count"].iloc[0]),
        "session_count_per_run": int(seed_runs["session_count"].iloc[0]),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_method": "seed_then_paired_CME_session_cluster",
        "selection_payload_sha256": str(selection["selection_payload_sha256"]),
        "resolved_config_sha256": resolved_config_sha256,
        "lineage": lineage,
        "created_at_utc": _utc_now(),
    }
    validation_path = analysis_dir / "local_lr_validation_summary.json"
    _write_json(validation_path, validation)
    return validation_path


__all__ = [
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_BOOTSTRAP_SEED",
    "EXPECTED_LEARNING_RATES",
    "EXPECTED_SEEDS",
    "EXPECTED_TEXT_MODES",
    "EXPECTED_TOLERANCES",
    "LocalLearningRateAnalysisError",
    "build_bootstrap_tables",
    "build_seed_run_metrics",
    "build_selection_summary",
    "collect_run_metrics",
    "evaluate_q3_pair_metrics",
    "run_local_lr_analysis",
    "summarize_across_seeds",
    "summarize_lr_scores",
    "validate_job_matrix",
    "validate_pair_metric_matrix",
]
