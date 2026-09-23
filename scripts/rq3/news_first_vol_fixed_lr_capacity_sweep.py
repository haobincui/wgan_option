"""Q3-only fixed-learning-rate capacity sweep with repeated seeds.

This experiment is intentionally isolated from both the staged single-seed
capacity sweep and the local learning-rate sweep.  Its immutable unit is
``(capacity, seed, tolerance, text mode)`` at the already selected LR 5e-7.
"""

from __future__ import annotations

import csv
import json
import math
import os
import socket
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from scripts.rq3 import news_first_vol_lr_sweep as base_lr
from scripts.rq3 import news_first_vol_training as training
from wgan_option.utils.text_ablation import (
    REAL_TEXT,
    normalize_text_ablation_mode,
    text_information_path,
)


ROOT_KEY = training.ROOT_KEY
REPO_ROOT = training.REPO_ROOT
EXPERIMENT_KIND = "fixed_lr_capacity_seed_sweep"
EXPERIMENT_STAGE = "fixed_lr_capacity_seed_screen"
FIXED_LR_PROFILE = "lr_5e_07"
FIXED_LEARNING_RATE = 5.0e-7
FIXED_SCHEDULER_MIN_LR = 5.0e-8
FROZEN_PROFILES = training.CAPACITY_PROFILE_NAMES
FROZEN_CAPACITY_PROFILES = FROZEN_PROFILES
FROZEN_SEEDS = (42, 202, 404)
FROZEN_TOLERANCES = (5, 30)
FROZEN_TEXT_MODES = ("current_only", REAL_TEXT)
# Four slots are ordered GPU0/slot0, GPU0/slot1, GPU1/slot0, GPU1/slot1.
# This Latin-square-like order puts one mode and one tolerance on each GPU in
# every wave, preventing either experimental axis from being a GPU identity.
WAVE_CELL_ORDER = (
    ("current_only", 5),
    (REAL_TEXT, 30),
    (REAL_TEXT, 5),
    ("current_only", 30),
)
TERMINAL_EXPERIMENT_STATES = {"completed_q3_only", "dry_run_passed"}

_read_json = training._read_json
_write_json = training._write_json
_write_yaml = training._write_yaml
_write_csv = training._write_csv
_sha256_file = training._sha256_file
_payload_sha256 = training._payload_sha256
_atomic_write_text = training._atomic_write_text
_utc_now = training._utc_now
_require_mapping = training._require_mapping
_resolve_repo_path = training._resolve_repo_path
_job_status_path = training._job_status_path
_build_split_manifest = training._build_split_manifest


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _require_mapping(payload, label)


def _exact_float(value: Any, expected: float, label: str) -> None:
    observed = float(value)
    if not math.isfinite(observed) or observed != expected:
        raise ValueError(f"{label} is frozen to {expected}; observed={observed}")


def _sweep_config(config: Mapping[str, Any]) -> dict[str, Any]:
    sweep = _require_mapping(
        config.get("fixed_lr_capacity_seed_sweep"),
        "fixed_lr_capacity_seed_sweep",
    )
    if not bool(sweep.get("enabled", False)):
        raise ValueError("fixed_lr_capacity_seed_sweep.enabled must be true")
    return sweep


def _profiles(config: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    raw = _require_mapping(_sweep_config(config).get("profiles"), "profiles")
    names = tuple(str(name).strip().lower() for name in raw)
    if names != FROZEN_PROFILES:
        raise ValueError(f"profiles must preserve order {list(FROZEN_PROFILES)}")
    fields = set(
        training.CAPACITY_PROFILE_SHAPE_FIELDS
        + training.CAPACITY_PROFILE_METADATA_FIELDS
    )
    result: dict[str, dict[str, int]] = {}
    for name, values in raw.items():
        normalized = str(name).strip().lower()
        profile = _require_mapping(values, f"profiles.{name}")
        if set(profile) != fields:
            raise ValueError(
                f"profile {normalized} fields differ: "
                f"missing={sorted(fields - set(profile))}, "
                f"extra={sorted(set(profile) - fields)}"
            )
        observed = {field: int(profile[field]) for field in fields}
        expected = training.FROZEN_CAPACITY_PROFILES[normalized]
        if observed != expected:
            raise ValueError(
                f"profile {normalized} differs from frozen capacity: "
                f"expected={expected}, observed={observed}"
            )
        result[normalized] = observed
    return result


def _validate_frozen_config(config: Mapping[str, Any]) -> None:
    datasets = _require_mapping(config.get("datasets"), "datasets")
    split = _require_mapping(config.get("split"), "split")
    sweep = _sweep_config(config)
    runtime = _require_mapping(config.get("runtime"), "runtime")
    models = _require_mapping(config.get("models"), "models")

    if tuple(int(value) for value in datasets.get("tolerances_minutes", ())) != (
        5,
        10,
        15,
        30,
    ):
        raise ValueError("datasets.tolerances_minutes is frozen to 5/10/15/30")
    if int(datasets.get("common_evaluation_tolerance_minutes", -1)) != 5:
        raise ValueError("common evaluation tolerance is frozen to 5m")
    if str(datasets.get("sheet_name", "")) != "gan_input_ready":
        raise ValueError("sheet_name is frozen to gan_input_ready")
    if str(datasets.get("text_embedding_mode", "")).lower() != "lp":
        raise ValueError("text embedding is frozen to lp")
    if int(datasets.get("seed", -1)) != 42:
        raise ValueError("datasets.seed remains the split-manifest seed 42")
    if training._support_mask_mode(datasets) != "raw_joint":
        raise ValueError("support mask is frozen to raw_joint")
    if tuple(training._configured_text_ablation_modes(datasets)) != FROZEN_TEXT_MODES:
        raise ValueError("datasets text modes are frozen")

    if str(split.get("train_end_utc")) != "2023-07-01T00:00:00Z":
        raise ValueError("train_end_utc is frozen")
    if str(split.get("validation_end_utc")) != "2023-10-01T00:00:00Z":
        raise ValueError("validation_end_utc is frozen")
    if int(split.get("validation_mc_samples", -1)) != 16:
        raise ValueError("validation_mc_samples is frozen to 16")

    if str(sweep.get("experiment_stage", "")) != EXPERIMENT_STAGE:
        raise ValueError(f"experiment_stage is frozen to {EXPERIMENT_STAGE}")
    if str(sweep.get("fixed_lr_profile", "")) != FIXED_LR_PROFILE:
        raise ValueError(f"fixed_lr_profile is frozen to {FIXED_LR_PROFILE}")
    _exact_float(sweep.get("initial_learning_rate"), FIXED_LEARNING_RATE, "initial LR")
    _exact_float(
        sweep.get("scheduler_min_lr"), FIXED_SCHEDULER_MIN_LR, "scheduler floor"
    )
    _profiles(config)
    if tuple(int(value) for value in sweep.get("seeds", ())) != FROZEN_SEEDS:
        raise ValueError(f"seeds are frozen to {list(FROZEN_SEEDS)}")
    if tuple(int(value) for value in sweep.get("tolerances_minutes", ())) != (
        FROZEN_TOLERANCES
    ):
        raise ValueError(f"tolerances are frozen to {list(FROZEN_TOLERANCES)}")
    modes = tuple(
        normalize_text_ablation_mode(value)
        for value in sweep.get("text_ablation_modes", ())
    )
    if modes != FROZEN_TEXT_MODES:
        raise ValueError("text modes are frozen to current_only/real_text")
    analysis = _require_mapping(sweep.get("analysis"), "analysis")
    frozen_analysis = {
        "selection_panel": "common_validation_05m",
        "primary_text_ablation_mode": REAL_TEXT,
        "diagnostic_text_ablation_mode": "current_only",
        "selection_rule": "one_standard_error_smallest",
    }
    for field, expected in frozen_analysis.items():
        if str(analysis.get(field, "")) != expected:
            raise ValueError(f"analysis.{field} is frozen to {expected}")
    if int(analysis.get("bootstrap_clusters", -1)) != 10_000:
        raise ValueError("bootstrap_clusters is frozen to 10000")
    _exact_float(
        analysis.get("minimum_mean_improvement_fraction"),
        0.005,
        "minimum improvement",
    )
    if not bool(analysis.get("q4_prediction_and_evaluation_forbidden", False)):
        raise ValueError("Q4 prediction/evaluation must remain forbidden")

    if set(models) != {"regression"}:
        raise ValueError("Only Regression is permitted")
    model = _require_mapping(models["regression"], "models.regression")
    if str(model.get("trainer_command")) != "vol-regression-xlsx":
        raise ValueError("trainer command is frozen")
    values = _require_mapping(model.get("training"), "regression.training")
    frozen_ints = {
        "channels": 1,
        "embedding_dim": 1024,
        "noise_dim": 32,
        "gen_res_blocks": 0,
        "num_epochs": 100,
        "early_stopping_min_epochs": 30,
        "early_stopping_patience": 12,
        "reduce_lr_patience": 3,
        "batch_size": 16,
        "constraint_warmup_epochs": 0,
    }
    for field, expected in frozen_ints.items():
        if int(values.get(field, -1)) != expected:
            raise ValueError(f"Regression {field} is frozen to {expected}")
    if str(values.get("residual_output_mode", "")) != "identity_softplus_residual":
        raise ValueError("identity residual is required")
    if str(values.get("best_checkpoint_metric", "")) != "val_hybrid_score":
        raise ValueError("best_checkpoint_metric is frozen to val_hybrid_score")
    for field in (
        "evaluate_initial_checkpoint",
        "use_reduce_lr_on_plateau",
        "use_early_stopping",
        "use_calendar_constraint",
        "use_butterfly_constraint",
        "use_smooth_constraint",
    ):
        if not bool(values.get(field, False)):
            raise ValueError(f"{field} must remain true")
    _exact_float(values.get("learning_rate"), FIXED_LEARNING_RATE, "model LR")
    _exact_float(values.get("reduce_lr_min_lr"), FIXED_SCHEDULER_MIN_LR, "model floor")
    _exact_float(values.get("reduce_lr_factor"), 0.5, "reduce_lr_factor")
    frozen_floats = {
        "train_ratio": 0.8,
        "beta_1": 0.5,
        "beta_2": 0.9,
        "lambda_recon": 10.0,
        "lambda_calendar": 2.0,
        "lambda_butterfly": 2.0,
        "lambda_smooth": 0.1,
        "lambda_delta_shrink": 0.0,
        "baseline_penalty_weight": 2.0,
        "early_stopping_min_delta": 0.0,
    }
    for field, expected in frozen_floats.items():
        _exact_float(values.get(field), expected, field)
    if int(values.get("save_every", 0)) <= 100:
        raise ValueError("save_every must preserve best/final-only checkpoints")

    if int(runtime.get("slots_per_gpu", -1)) != 2:
        raise ValueError("slots_per_gpu is frozen to 2")
    if int(runtime.get("cpu_threads_per_job", -1)) != 8:
        raise ValueError("cpu_threads_per_job is frozen to 8")
    gpu_ids = tuple(int(value) for value in runtime.get("gpu_ids", ()))
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("Exactly two distinct GPUs are required")


def _resolved_config(config_path: str | Path) -> dict[str, Any]:
    path = _resolve_repo_path(config_path)
    root = _load_yaml_mapping(path, "fixed LR capacity config")
    config = _require_mapping(root.get(ROOT_KEY), ROOT_KEY)
    _validate_frozen_config(config)
    resolved = deepcopy(config)
    resolved["datasets"]["root"] = str(_resolve_repo_path(resolved["datasets"]["root"]))
    resolved["datasets"]["support_mask_mode"] = "raw_joint"
    resolved["datasets"]["text_ablation_modes"] = list(FROZEN_TEXT_MODES)
    resolved["runtime"]["python_executable"] = str(
        _resolve_repo_path(resolved["runtime"].get("python_executable", sys.executable))
    )
    resolved["source_config_path"] = str(path)
    return resolved


def _capacity_profile_sha256(profile: str) -> str:
    normalized = str(profile).strip().lower()
    if normalized not in FROZEN_PROFILES:
        raise ValueError(f"Unknown capacity profile: {profile}")
    return training._capacity_profile_sha256(
        normalized, training.FROZEN_CAPACITY_PROFILES[normalized]
    )


def _fixed_lr_profile_sha256() -> str:
    return _payload_sha256(
        {
            "schema_version": 1,
            "fixed_lr_profile": FIXED_LR_PROFILE,
            "initial_learning_rate": FIXED_LEARNING_RATE,
            "scheduler_min_lr": FIXED_SCHEDULER_MIN_LR,
            "scheduler_factor": 0.5,
            "scheduler_patience": 3,
            "support_mask_mode": "raw_joint",
            "residual_output_mode": "identity_softplus_residual",
        }
    )


def _capacity_seed_profile_sha256(profile: str, seed: int) -> str:
    if int(seed) not in FROZEN_SEEDS:
        raise ValueError(f"Unknown seed: {seed}")
    return _payload_sha256(
        {
            "schema_version": 1,
            "capacity_profile_sha256": _capacity_profile_sha256(profile),
            "fixed_lr_profile_sha256": _fixed_lr_profile_sha256(),
            "seed": int(seed),
        }
    )


def _job_id(profile: str, seed: int, mode: str, tolerance: int) -> str:
    return (
        f"fixed_lr_capacity_regression_{profile}_seed_{int(seed):03d}_"
        f"{normalize_text_ablation_mode(mode)}_{int(tolerance):02d}m"
    )


def _job_specs() -> list[dict[str, Any]]:
    return [
        {
            "capacity_profile": profile,
            "seed": seed,
            "text_ablation_mode": mode,
            "tolerance_minutes": tolerance,
        }
        for profile in FROZEN_PROFILES
        for seed in FROZEN_SEEDS
        for mode, tolerance in WAVE_CELL_ORDER
    ]


def _training_payload(
    resolved: Mapping[str, Any],
    root: Path,
    *,
    profile: str,
    seed: int,
    mode: str,
    tolerance: int,
) -> dict[str, Any]:
    payload = training._training_payload(
        resolved,
        family="regression",
        tolerance=int(tolerance),
        text_ablation_mode=mode,
        multi_mode=True,
        experiment_root=root,
        capacity_profile=None,
    )
    shape = _profiles(resolved)[profile]
    payload.update(
        {field: int(shape[field]) for field in training.CAPACITY_PROFILE_SHAPE_FIELDS}
    )
    payload.update(
        {
            "seed": int(seed),
            "news_first_capacity_profile": profile,
            "news_first_capacity_profile_sha256": _capacity_profile_sha256(profile),
            "news_first_capacity_seed_profile_sha256": (
                _capacity_seed_profile_sha256(profile, seed)
            ),
            "news_first_fixed_learning_rate_profile": FIXED_LR_PROFILE,
            "news_first_fixed_learning_rate_profile_sha256": (
                _fixed_lr_profile_sha256()
            ),
            "news_first_lr_profile": FIXED_LR_PROFILE,
            "news_first_lr_profile_sha256": _fixed_lr_profile_sha256(),
            "learning_rate": FIXED_LEARNING_RATE,
            "use_reduce_lr_on_plateau": True,
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": 3,
            "reduce_lr_min_lr": FIXED_SCHEDULER_MIN_LR,
            "num_epochs": 100,
            "use_early_stopping": True,
            "early_stopping_min_epochs": 30,
            "early_stopping_patience": 12,
            "output_root": str(
                root
                / "runs"
                / "regression"
                / profile
                / f"seed_{int(seed):03d}"
                / normalize_text_ablation_mode(mode)
                / f"tolerance_{int(tolerance):02d}m"
            ),
        }
    )
    return payload


def _job_spec_sha256(job: Mapping[str, Any]) -> str:
    return _payload_sha256(
        {key: value for key, value in job.items() if key != "job_spec_sha256"}
    )


def _source_rows(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    return training._source_rows(resolved)


def _code_rows() -> list[dict[str, Any]]:
    rows = training._code_rows()
    known = {str(row["relative_path"]) for row in rows}
    for relative in (
        "scripts/rq3/news_first_vol_fixed_lr_capacity_sweep.py",
        "scripts/rq3/news_first_vol_fixed_lr_capacity_analysis.py",
        "scripts/rq3/news_first_vol_fixed_lr_capacity_report.py",
    ):
        path = REPO_ROOT / relative
        if not path.is_file() or relative in known:
            continue
        rows.append(
            {
                "relative_path": relative,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _load_registry(root: Path) -> dict[str, Any]:
    registry = _require_mapping(
        _read_json(root / "registry" / "jobs.json"), "fixed LR capacity registry"
    )
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment root is not a fixed-LR capacity/seed sweep")
    return registry


def _validate_resolved_snapshot(root: Path) -> dict[str, Any]:
    resolved = _require_mapping(
        _load_yaml_mapping(root / "resolved_config.yaml", "resolved config").get(
            ROOT_KEY
        ),
        ROOT_KEY,
    )
    observed = _payload_sha256(resolved)
    expected = (
        (root / "registry" / "resolved_config.sha256")
        .read_text(encoding="utf-8")
        .strip()
    )
    if observed != expected:
        raise ValueError("Resolved fixed-LR capacity config hash mismatch")
    if str(_load_registry(root).get("resolved_config_sha256", "")) != expected:
        raise ValueError("Registry resolved-config hash mismatch")
    return resolved


def _validate_job_lineage(root: Path, job: Mapping[str, Any]) -> None:
    if str(job.get("job_spec_sha256", "")) != _job_spec_sha256(job):
        raise ValueError(f"job-spec hash mismatch for {job.get('job_id')}")
    profile = str(job.get("capacity_profile", "")).strip().lower()
    seed = int(job.get("seed", -1))
    mode = normalize_text_ablation_mode(job.get("text_ablation_mode", ""))
    tolerance = int(job.get("tolerance_minutes", -1))
    if profile not in FROZEN_PROFILES:
        raise ValueError(f"Unknown capacity profile for {job.get('job_id')}")
    if seed not in FROZEN_SEEDS:
        raise ValueError(f"Unknown seed for {job.get('job_id')}")
    if mode not in FROZEN_TEXT_MODES or tolerance not in FROZEN_TOLERANCES:
        raise ValueError(f"Out-of-contract axes for {job.get('job_id')}")
    if str(job.get("experiment_stage", "")) != EXPERIMENT_STAGE:
        raise ValueError(f"Experiment stage mismatch for {job.get('job_id')}")
    if str(job.get("job_id")) != _job_id(profile, seed, mode, tolerance):
        raise ValueError(f"Job ID does not encode immutable axes: {job.get('job_id')}")

    resolved = _validate_resolved_snapshot(root)
    shape = _profiles(resolved)[profile]
    contracts = {
        "capacity_profile_sha256": _capacity_profile_sha256(profile),
        "capacity_seed_profile_sha256": _capacity_seed_profile_sha256(profile, seed),
        "expected_regression_parameters": int(shape["expected_regression_parameters"]),
        "fixed_lr_profile": FIXED_LR_PROFILE,
        "fixed_lr_profile_sha256": _fixed_lr_profile_sha256(),
        "lr_profile": FIXED_LR_PROFILE,
        "lr_profile_sha256": _fixed_lr_profile_sha256(),
        "initial_learning_rate": FIXED_LEARNING_RATE,
        "scheduler_min_lr": FIXED_SCHEDULER_MIN_LR,
        "seed": seed,
        "support_mask_mode": "raw_joint",
    }
    for field, expected in contracts.items():
        if job.get(field) != expected:
            raise ValueError(f"Job lineage mismatch for {job['job_id']}: {field}")

    source_manifest = root / "source_hashes.csv"
    if _sha256_file(source_manifest) != str(job.get("source_manifest_sha256", "")):
        raise ValueError(f"Source-manifest hash mismatch for {job['job_id']}")
    with source_manifest.open("r", encoding="utf-8", newline="") as handle:
        sources = {row["source_role"]: row for row in csv.DictReader(handle)}
    for role in (
        f"training_workbook_{tolerance:02d}m",
        f"support_audit_{tolerance:02d}m",
        "orchestration_config",
    ):
        if role not in sources:
            raise ValueError(f"Source manifest is missing {role}")
        row = sources[role]
        path = Path(str(row["path"]))
        if not path.is_file() or _sha256_file(path) != str(row["sha256"]):
            raise ValueError(f"Source hash mismatch for {job['job_id']}: {role}")

    config_path = Path(str(job["training_config_path"]))
    if not config_path.is_file() or _sha256_file(config_path) != str(
        job["config_sha256"]
    ):
        raise ValueError(f"Training config hash mismatch for {job['job_id']}")
    payload = _load_yaml_mapping(config_path, f"training config {job['job_id']}")
    payload_contracts = {
        "seed": seed,
        "news_first_capacity_profile": profile,
        "news_first_capacity_profile_sha256": _capacity_profile_sha256(profile),
        "news_first_capacity_seed_profile_sha256": (
            _capacity_seed_profile_sha256(profile, seed)
        ),
        "news_first_fixed_learning_rate_profile": FIXED_LR_PROFILE,
        "news_first_fixed_learning_rate_profile_sha256": _fixed_lr_profile_sha256(),
        "news_first_lr_profile": FIXED_LR_PROFILE,
        "news_first_lr_profile_sha256": _fixed_lr_profile_sha256(),
        "learning_rate": FIXED_LEARNING_RATE,
        "reduce_lr_min_lr": FIXED_SCHEDULER_MIN_LR,
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 3,
        "early_stopping_min_epochs": 30,
        "early_stopping_patience": 12,
        "num_epochs": 100,
        "support_mask_mode": "raw_joint",
        "residual_output_mode": "identity_softplus_residual",
    }
    for field, expected in payload_contracts.items():
        if payload.get(field) != expected:
            raise ValueError(f"Training contract mismatch for {job['job_id']}: {field}")
    for field in training.CAPACITY_PROFILE_SHAPE_FIELDS:
        if int(payload.get(field, -1)) != int(shape[field]):
            raise ValueError(f"Capacity shape mismatch for {job['job_id']}: {field}")
    expected_output = (
        root
        / "runs"
        / "regression"
        / profile
        / f"seed_{seed:03d}"
        / mode
        / f"tolerance_{tolerance:02d}m"
    )
    if Path(str(job["output_root"])) != expected_output:
        raise ValueError(f"Output path mismatch for {job['job_id']}")
    if Path(str(payload["output_root"])) != expected_output:
        raise ValueError(f"Training output path mismatch for {job['job_id']}")
    dataset_path = Path(str(job["dataset_path"]))
    if _sha256_file(dataset_path) != str(job["dataset_sha256"]):
        raise ValueError(f"Dataset hash mismatch for {job['job_id']}")


def _validate_registry(root: Path) -> None:
    jobs = [dict(job) for job in _load_registry(root).get("jobs", [])]
    expected = {
        _job_id(profile, seed, mode, tolerance)
        for profile in FROZEN_PROFILES
        for seed in FROZEN_SEEDS
        for mode in FROZEN_TEXT_MODES
        for tolerance in FROZEN_TOLERANCES
    }
    observed = {str(job.get("job_id")) for job in jobs}
    if len(observed) != len(jobs):
        raise ValueError("Fixed-LR capacity registry contains duplicate job IDs")
    if observed != expected:
        raise ValueError(
            "Fixed-LR capacity registry must contain exactly the frozen 72-job "
            f"matrix; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    if sorted(int(job["wave"]) for job in jobs) != [
        wave for wave in range(1, 19) for _ in range(4)
    ]:
        raise ValueError("Fixed-LR capacity registry must contain 18 four-job waves")
    for job in jobs:
        _validate_job_lineage(root, job)


TASK_FIELDS = (
    "job_id",
    "wave",
    "model_family",
    "capacity_profile",
    "capacity_profile_sha256",
    "capacity_seed_profile_sha256",
    "expected_regression_parameters",
    "fixed_lr_profile",
    "fixed_lr_profile_sha256",
    "lr_profile",
    "lr_profile_sha256",
    "initial_learning_rate",
    "scheduler_min_lr",
    "lr_trace",
    "seed",
    "experiment_stage",
    "text_ablation_mode",
    "text_information_path",
    "support_mask_mode",
    "tolerance_minutes",
    "gpu_id",
    "gpu_slot",
    "numa_node",
    "job_spec_sha256",
    "status",
    "attempt",
    "pid",
    "config_sha256",
    "dataset_sha256",
    "source_manifest_sha256",
    "run_dir",
    "log_path",
    "exit_code",
    "started_at_utc",
    "completed_at_utc",
    "error",
)


def _refresh_exports(root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for job in _load_registry(root)["jobs"]:
        status_path = _job_status_path(root, str(job["job_id"]))
        status = (
            _read_json(status_path) if status_path.is_file() else {"status": "missing"}
        )
        row = {**job, **status}
        row["lr_trace"] = json.dumps(row.get("lr_trace", []), separators=(",", ":"))
        rows.append(row)
        artifacts.extend(
            {"job_id": job["job_id"], **artifact}
            for artifact in status.get("artifacts", [])
        )
    _write_csv(root / "task_registry.csv", rows, TASK_FIELDS)
    stable = [
        root / "task_registry.csv",
        root / "split_manifest.csv",
        root / "text_ablation_manifest.csv",
        root / "source_hashes.csv",
        root / "code_hashes.csv",
        root / "code_hashes_current.csv",
        root / "config_hashes.csv",
        root / "capacity_seed_profile_manifest.csv",
        root / "resource_usage.csv",
        root / "resource_summary.csv",
        root / "run_manifest.json",
        root / "registry" / "jobs.json",
        root / "registry" / "experiment_status.json",
        root / "registry" / "resolved_config.sha256",
    ]
    stable.extend((root / "registry" / "jobs").glob("*.status.json"))
    for directory in (root / "analysis", root / "report"):
        if directory.is_dir():
            stable.extend(path for path in directory.rglob("*") if path.is_file())
    seen = {str(row.get("path", "")) for row in artifacts}
    for path in stable:
        if path.is_file() and str(path) not in seen:
            artifacts.append(
                {
                    "job_id": "",
                    "artifact_role": f"experiment:{path.relative_to(root).as_posix()}",
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    _write_csv(
        root / "output_hashes.csv",
        artifacts,
        ("job_id", "artifact_role", "path", "size_bytes", "sha256"),
    )
    return root / "task_registry.csv"


def _refresh_config_hashes(root: Path) -> Path:
    rows = [
        {
            "config_role": "resolved_orchestration",
            "path": str(root / "resolved_config.yaml"),
            "sha256": _sha256_file(root / "resolved_config.yaml"),
        }
    ]
    rows.extend(
        {
            "config_role": str(job["job_id"]),
            "path": str(job["training_config_path"]),
            "sha256": str(job["config_sha256"]),
        }
        for job in _load_registry(root)["jobs"]
    )
    return _write_csv(root / "config_hashes.csv", rows, tuple(rows[0]))


def _maximum_attempt(root: Path) -> int:
    attempts = []
    for job in _load_registry(root).get("jobs", []):
        path = _job_status_path(root, str(job["job_id"]))
        if path.is_file():
            attempts.append(int(_read_json(path).get("attempt", 0)))
    return max(attempts, default=0)


def refresh_fixed_lr_capacity_lineage(
    config_path: str | Path, root: str | Path
) -> Path:
    experiment_root = Path(root).resolve(strict=False)
    expected_path = experiment_root / "registry" / "resolved_config.sha256"
    if not expected_path.is_file():
        raise ValueError(f"Not a prepared fixed-LR capacity sweep: {experiment_root}")
    resolved = _resolved_config(config_path)
    resolved_sha = _payload_sha256(resolved)
    if expected_path.read_text(encoding="utf-8").strip() != resolved_sha:
        raise ValueError("Prepared experiment config differs from requested config")
    source_rows = _source_rows(resolved)
    source_path = experiment_root / "source_hashes.csv"
    if source_path.is_file():
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            previous = {row["path"]: row["sha256"] for row in csv.DictReader(handle)}
        current = {row["path"]: row["sha256"] for row in source_rows}
        if previous != current:
            raise ValueError("Fixed-LR capacity source lineage changed")
    _write_csv(
        source_path, source_rows, ("source_role", "path", "size_bytes", "sha256")
    )

    code_rows = _code_rows()
    code_path = experiment_root / "code_hashes.csv"
    prepared: dict[str, str] = {}
    if code_path.is_file():
        with code_path.open("r", encoding="utf-8", newline="") as handle:
            prepared = {
                row["relative_path"]: row["sha256"] for row in csv.DictReader(handle)
            }
    current = {row["relative_path"]: row["sha256"] for row in code_rows}
    changed = bool(prepared and prepared != current)
    current_path: Path | None = None
    if not prepared or _maximum_attempt(experiment_root) == 0:
        _write_csv(
            code_path, code_rows, ("relative_path", "path", "size_bytes", "sha256")
        )
        changed = False
    elif changed:
        current_path = experiment_root / "code_hashes_current.csv"
        _write_csv(
            current_path,
            code_rows,
            ("relative_path", "path", "size_bytes", "sha256"),
        )
    manifest = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "experiment_root": str(experiment_root),
        "resolved_config_sha256": resolved_sha,
        "prepared_code_hashes_path": str(code_path),
        "prepared_code_hashes_sha256": _sha256_file(code_path),
        "current_code_hashes_path": str(current_path or code_path),
        "current_code_hashes_sha256": _sha256_file(current_path or code_path),
        "code_changed_since_prepared_snapshot": changed,
        "source_hashes_path": str(source_path),
        "source_hashes_sha256": _sha256_file(source_path),
        "q4_access_policy": (
            "loader_materialization_allowed; prediction_and_evaluation_forbidden"
        ),
        "updated_at_utc": _utc_now(),
    }
    _write_json(experiment_root / "run_manifest.json", manifest)
    return experiment_root / "run_manifest.json"


def prepare_fixed_lr_capacity_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    reuse: bool = False,
) -> Path:
    root = _resolve_repo_path(output_dir)
    resolved = _resolved_config(config_path)
    resolved_sha = _payload_sha256(resolved)
    expected_path = root / "registry" / "resolved_config.sha256"
    if root.exists() and any(root.iterdir()):
        if not reuse:
            raise FileExistsError(f"Experiment root exists; use --reuse: {root}")
        if not expected_path.is_file():
            raise ValueError(
                f"Existing directory is not a fixed-LR capacity sweep: {root}"
            )
        if expected_path.read_text(encoding="utf-8").strip() != resolved_sha:
            raise ValueError("Prepared experiment config differs from requested config")
        _validate_registry(root)
        refresh_fixed_lr_capacity_lineage(config_path, root)
        _refresh_exports(root)
        return root

    training._validate_dataset_summary(Path(resolved["datasets"]["root"]))
    source_rows = _source_rows(resolved)
    root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "registry/jobs",
        "configs/jobs",
        "logs",
        "resources",
        "runs",
        "analysis",
        "report",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _build_split_manifest(resolved, root)
    _write_yaml(root / "resolved_config.yaml", {ROOT_KEY: resolved})
    _atomic_write_text(expected_path, resolved_sha + "\n")
    _write_csv(
        root / "source_hashes.csv",
        source_rows,
        ("source_role", "path", "size_bytes", "sha256"),
    )
    source_sha = _sha256_file(root / "source_hashes.csv")
    source_by_path = {str(row["path"]): str(row["sha256"]) for row in source_rows}
    slots = training._capacity_worker_slots(resolved)
    profile_values = _profiles(resolved)

    jobs: list[dict[str, Any]] = []
    for index, spec in enumerate(_job_specs()):
        profile = str(spec["capacity_profile"])
        seed = int(spec["seed"])
        mode = normalize_text_ablation_mode(spec["text_ablation_mode"])
        tolerance = int(spec["tolerance_minutes"])
        job_id = _job_id(profile, seed, mode, tolerance)
        gpu_id, gpu_slot, numa_node = slots[index % len(slots)]
        payload = _training_payload(
            resolved,
            root,
            profile=profile,
            seed=seed,
            mode=mode,
            tolerance=tolerance,
        )
        config_path_for_job = root / "configs" / "jobs" / f"{job_id}.yaml"
        _write_yaml(config_path_for_job, payload)
        dataset_path = str(payload["data_path"])
        if dataset_path not in source_by_path:
            raise ValueError(f"Dataset source hash unavailable: {dataset_path}")
        profile_sha = _capacity_profile_sha256(profile)
        seed_sha = _capacity_seed_profile_sha256(profile, seed)
        job: dict[str, Any] = {
            "job_id": job_id,
            "wave": index // len(slots) + 1,
            "model_family": "regression",
            "trainer_command": "vol-regression-xlsx",
            "capacity_profile": profile,
            "capacity_profile_sha256": profile_sha,
            "capacity_seed_profile_sha256": seed_sha,
            "expected_regression_parameters": int(
                profile_values[profile]["expected_regression_parameters"]
            ),
            "fixed_lr_profile": FIXED_LR_PROFILE,
            "fixed_lr_profile_sha256": _fixed_lr_profile_sha256(),
            "lr_profile": FIXED_LR_PROFILE,
            "lr_profile_sha256": _fixed_lr_profile_sha256(),
            "initial_learning_rate": FIXED_LEARNING_RATE,
            "scheduler_min_lr": FIXED_SCHEDULER_MIN_LR,
            "lr_trace": [FIXED_LEARNING_RATE],
            "seed": seed,
            "experiment_stage": EXPERIMENT_STAGE,
            "text_ablation_mode": mode,
            "text_information_path": text_information_path(mode),
            "support_mask_mode": "raw_joint",
            "tolerance_minutes": tolerance,
            "gpu_id": gpu_id,
            "gpu_slot": gpu_slot,
            "numa_node": numa_node,
            "training_config_path": str(config_path_for_job),
            "config_sha256": _sha256_file(config_path_for_job),
            "dataset_path": dataset_path,
            "dataset_sha256": source_by_path[dataset_path],
            "source_manifest_sha256": source_sha,
            "output_root": str(payload["output_root"]),
        }
        job["job_spec_sha256"] = _job_spec_sha256(job)
        jobs.append(job)
        _write_json(
            _job_status_path(root, job_id),
            {
                "job_id": job_id,
                "status": "prepared",
                "attempt": 0,
                "config_sha256": job["config_sha256"],
                "capacity_profile": profile,
                "capacity_profile_sha256": profile_sha,
                "capacity_seed_profile_sha256": seed_sha,
                "expected_regression_parameters": job["expected_regression_parameters"],
                "fixed_lr_profile": FIXED_LR_PROFILE,
                "fixed_lr_profile_sha256": _fixed_lr_profile_sha256(),
                "lr_profile": FIXED_LR_PROFILE,
                "lr_profile_sha256": _fixed_lr_profile_sha256(),
                "initial_learning_rate": FIXED_LEARNING_RATE,
                "scheduler_min_lr": FIXED_SCHEDULER_MIN_LR,
                "lr_trace": [FIXED_LEARNING_RATE],
                "seed": seed,
                "experiment_stage": EXPERIMENT_STAGE,
                "updated_at_utc": _utc_now(),
            },
        )

    _write_json(
        root / "registry" / "jobs.json",
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "experiment_root": str(root),
            "resolved_config_sha256": resolved_sha,
            "created_at_utc": _utc_now(),
            "jobs": jobs,
        },
    )
    _write_json(
        root / "registry" / "experiment_status.json",
        {
            "status": "prepared",
            "current_stage": EXPERIMENT_STAGE,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
            "updated_at_utc": _utc_now(),
        },
    )
    manifest_rows = []
    for profile in FROZEN_PROFILES:
        values = profile_values[profile]
        for seed in FROZEN_SEEDS:
            manifest_rows.append(
                {
                    "capacity_profile": profile,
                    "capacity_profile_sha256": _capacity_profile_sha256(profile),
                    "capacity_seed_profile_sha256": (
                        _capacity_seed_profile_sha256(profile, seed)
                    ),
                    "seed": seed,
                    "expected_regression_parameters": int(
                        values["expected_regression_parameters"]
                    ),
                    "initial_learning_rate": FIXED_LEARNING_RATE,
                    "scheduler_min_lr": FIXED_SCHEDULER_MIN_LR,
                    "fixed_lr_profile": FIXED_LR_PROFILE,
                    "fixed_lr_profile_sha256": _fixed_lr_profile_sha256(),
                    "lr_profile": FIXED_LR_PROFILE,
                    "lr_profile_sha256": _fixed_lr_profile_sha256(),
                    **{
                        field: int(values[field])
                        for field in training.CAPACITY_PROFILE_SHAPE_FIELDS
                    },
                }
            )
    _write_csv(
        root / "capacity_seed_profile_manifest.csv",
        manifest_rows,
        tuple(manifest_rows[0]),
    )
    _refresh_config_hashes(root)
    _validate_registry(root)
    refresh_fixed_lr_capacity_lineage(config_path, root)
    _refresh_exports(root)
    return root


def _completed_job_is_valid(
    root: Path, job: Mapping[str, Any], status: Mapping[str, Any]
) -> bool:
    try:
        _validate_job_lineage(root, job)
    except (ValueError, FileNotFoundError):
        return False
    return training._completed_job_is_valid(root, job, status)


def _validate_run_contract(
    job: Mapping[str, Any], run_dir: Path, *, dry_run: bool
) -> list[dict[str, float | int]] | list[float]:
    trace = base_lr._validated_run_lr_trace(job, run_dir, dry_run=dry_run)
    if dry_run:
        return trace
    best = _require_mapping(
        _read_json(run_dir / "metrics" / "best_learned_checkpoint.json"),
        f"best learned checkpoint for {job['job_id']}",
    )
    if int(best.get("seed", -1)) != int(job["seed"]):
        raise ValueError(f"Trainer checkpoint seed mismatch for {job['job_id']}")
    if str(best.get("capacity_profile", "")) != str(job["capacity_profile"]):
        raise ValueError(f"Trainer capacity mismatch for {job['job_id']}")
    if str(best.get("capacity_profile_sha256", "")) != str(
        job["capacity_profile_sha256"]
    ):
        raise ValueError(f"Trainer capacity hash mismatch for {job['job_id']}")
    if str(best.get("capacity_seed_profile_sha256", "")) != str(
        job["capacity_seed_profile_sha256"]
    ):
        raise ValueError(f"Trainer capacity/seed hash mismatch for {job['job_id']}")
    if str(best.get("fixed_lr_profile", "")) != FIXED_LR_PROFILE:
        raise ValueError(f"Trainer fixed-LR profile mismatch for {job['job_id']}")
    if str(best.get("fixed_lr_profile_sha256", "")) != _fixed_lr_profile_sha256():
        raise ValueError(f"Trainer fixed-LR hash mismatch for {job['job_id']}")
    resolved_training = _load_yaml_mapping(
        run_dir / "metrics" / "training_resolved_config.yaml",
        f"resolved training config for {job['job_id']}",
    )
    if int(resolved_training.get("seed", -1)) != int(job["seed"]):
        raise ValueError(f"Resolved training seed mismatch for {job['job_id']}")
    return trace


def run_fixed_lr_capacity_worker(
    experiment_root: str | Path,
    job_id: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    root = _resolve_repo_path(experiment_root)
    matches = [
        dict(job)
        for job in _load_registry(root)["jobs"]
        if str(job["job_id"]) == str(job_id)
    ]
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicate fixed-LR capacity job: {job_id}")
    job = matches[0]
    _validate_job_lineage(root, job)
    status_path = _job_status_path(root, job_id)
    previous = _read_json(status_path)
    if previous.get("status") == "completed" and not dry_run:
        if resume and _completed_job_is_valid(root, job, previous):
            return Path(str(previous["run_dir"]))
        if not resume:
            raise RuntimeError(f"Job already completed: {job_id}")
    if previous.get("status") == "running" and training._pid_is_live(
        previous.get("pid")
    ):
        raise RuntimeError(f"Job is already running: {job_id}")
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    assigned = str(job["gpu_id"])
    if visible and visible.split(",")[0].strip() != assigned:
        raise RuntimeError(
            f"Worker GPU mismatch: assigned={assigned}, visible={visible}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = assigned
    resolved = _require_mapping(
        _load_yaml_mapping(root / "resolved_config.yaml", "resolved config").get(
            ROOT_KEY
        ),
        ROOT_KEY,
    )
    threads = str(int(resolved["runtime"]["cpu_threads_per_job"]))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = threads
    attempt = int(previous.get("attempt", 0)) + 1
    common = {
        "job_id": job_id,
        "attempt": attempt,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "gpu_id": job["gpu_id"],
        "config_sha256": job["config_sha256"],
        "capacity_profile": job["capacity_profile"],
        "capacity_profile_sha256": job["capacity_profile_sha256"],
        "capacity_seed_profile_sha256": job["capacity_seed_profile_sha256"],
        "expected_regression_parameters": job["expected_regression_parameters"],
        "fixed_lr_profile": FIXED_LR_PROFILE,
        "fixed_lr_profile_sha256": _fixed_lr_profile_sha256(),
        "lr_profile": FIXED_LR_PROFILE,
        "lr_profile_sha256": _fixed_lr_profile_sha256(),
        "initial_learning_rate": FIXED_LEARNING_RATE,
        "scheduler_min_lr": FIXED_SCHEDULER_MIN_LR,
        "lr_trace": list(job["lr_trace"]),
        "seed": int(job["seed"]),
        "experiment_stage": EXPERIMENT_STAGE,
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "dry_run": bool(dry_run),
        "log_path": str(os.environ.get("NEWS_FIRST_JOB_LOG_PATH", "")),
    }
    _write_json(status_path, {**common, "status": "running"})
    try:
        run_dir, artifacts = training._execute_training_job(job, dry_run=dry_run)
        trace = _validate_run_contract(job, run_dir, dry_run=dry_run)
        _write_json(
            status_path,
            {
                **common,
                "status": "dry_run_passed" if dry_run else "completed",
                "run_dir": str(run_dir),
                "artifacts": artifacts,
                "lr_trace": trace,
                "exit_code": 0,
                "completed_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
            },
        )
        return run_dir
    except BaseException as exc:
        _write_json(
            status_path,
            {
                **common,
                "status": "failed",
                "exit_code": 1,
                "error": f"{type(exc).__name__}: {exc}",
                "completed_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
            },
        )
        raise


def build_fixed_lr_capacity_worker_command(
    config_path: str | Path,
    root: str | Path,
    job: Mapping[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> list[str]:
    experiment_root = Path(root)
    resolved = _require_mapping(
        _load_yaml_mapping(
            experiment_root / "resolved_config.yaml", "resolved config"
        ).get(ROOT_KEY),
        ROOT_KEY,
    )
    runtime = resolved["runtime"]
    command: list[str] = []
    if bool(runtime.get("use_numa_binding", True)):
        command.extend(
            [
                str(runtime.get("numactl_executable", "numactl")),
                f"--cpunodebind={int(job['numa_node'])}",
                f"--membind={int(job['numa_node'])}",
            ]
        )
    command.extend(
        [
            str(runtime["python_executable"]),
            str(REPO_ROOT / "scripts" / "rq3" / "main.py"),
            "train-news-first-vol-fixed-lr-capacity-seed-sweep",
            "worker",
            "--config",
            str(_resolve_repo_path(config_path)),
            "--output-dir",
            str(experiment_root.resolve(strict=False)),
            "--job-id",
            str(job["job_id"]),
        ]
    )
    if dry_run:
        command.append("--worker-dry-run")
    if resume:
        command.append("--resume")
    return command


def _jobs_for_wave(
    root: Path, wave: int, *, resume: bool, dry_run: bool
) -> list[dict[str, Any]]:
    jobs = [
        dict(job)
        for job in _load_registry(root)["jobs"]
        if int(job["wave"]) == int(wave)
    ]
    selected = []
    for job in jobs:
        _validate_job_lineage(root, job)
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        if not dry_run and status.get("status") == "completed":
            if _completed_job_is_valid(root, job, status):
                if resume:
                    continue
                raise RuntimeError(
                    f"Job already completed; use --resume: {job['job_id']}"
                )
            if not resume:
                raise RuntimeError(
                    f"Invalid completed job requires --resume: {job['job_id']}"
                )
        if status.get("status") == "running" and training._pid_is_live(
            status.get("pid")
        ):
            raise RuntimeError(f"Refusing duplicate live job: {job['job_id']}")
        if status.get("status") in {"running", "failed"} and not (resume or dry_run):
            raise RuntimeError(
                f"Interrupted/failed job requires --resume: {job['job_id']}"
            )
        selected.append(job)
    return selected


def _validate_wave_completion(
    root: Path, jobs: Sequence[Mapping[str, Any]], *, dry_run: bool
) -> None:
    failures = []
    for job in jobs:
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        valid = (
            status.get("status") == "dry_run_passed"
            if dry_run
            else _completed_job_is_valid(root, job, status)
        )
        if not valid:
            failures.append(f"{job['job_id']}={status.get('status', 'missing')}")
    if failures:
        raise RuntimeError(f"Fixed-LR capacity wave did not complete: {failures}")


def _run_wave(
    root: Path,
    config_path: str | Path,
    wave: int,
    jobs: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
    resume: bool,
) -> None:
    if not jobs:
        return
    resolved = _require_mapping(
        _load_yaml_mapping(root / "resolved_config.yaml", "resolved config").get(
            ROOT_KEY
        ),
        ROOT_KEY,
    )
    runtime = resolved["runtime"]
    monitor = training._ResourceMonitor(
        root / "resource_usage.csv",
        executable=str(runtime.get("nvidia_smi_executable", "nvidia-smi")),
        interval_seconds=float(runtime.get("resource_sample_interval_seconds", 5)),
        wave=wave,
    )
    processes: list[subprocess.Popen[Any]] = []
    handles = []
    try:
        monitor.start()
        for job in jobs:
            previous = _read_json(_job_status_path(root, str(job["job_id"])))
            attempt = int(previous.get("attempt", 0)) + 1
            log_path = root / "logs" / f"{job['job_id']}.attempt_{attempt:02d}.log"
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
            env["NEWS_FIRST_JOB_LOG_PATH"] = str(log_path)
            env["PYTHONPATH"] = os.pathsep.join(
                (str(REPO_ROOT / "src"), str(REPO_ROOT))
            )
            threads = str(int(runtime["cpu_threads_per_job"]))
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                env[name] = threads
            process = subprocess.Popen(
                build_fixed_lr_capacity_worker_command(
                    config_path, root, job, dry_run=dry_run, resume=resume
                ),
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
        pending = set(range(len(processes)))
        while pending:
            for index in tuple(pending):
                code = processes[index].poll()
                if code is None:
                    continue
                pending.remove(index)
                if code != 0:
                    training._terminate_processes(processes)
                    raise RuntimeError(
                        f"Fixed-LR capacity wave {wave} job "
                        f"{jobs[index]['job_id']} exited {code}"
                    )
            if pending:
                time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        training._terminate_processes(processes)
        raise
    finally:
        monitor.stop()
        for handle in handles:
            handle.close()


RESOURCE_FIELDS = training.RESOURCE_SUMMARY_FIELDS + (
    "capacity_seed_profile_sha256",
    "expected_regression_parameters",
    "fixed_lr_profile",
    "fixed_lr_profile_sha256",
    "initial_learning_rate",
    "scheduler_min_lr",
    "lr_trace",
    "seed",
    "experiment_stage",
)


def _write_resource_summary(root: Path) -> Path:
    base_path = training._write_resource_summary(root)
    with base_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    jobs = {str(job["job_id"]): dict(job) for job in _load_registry(root)["jobs"]}
    for row in rows:
        job = jobs[str(row["job_id"])]
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        row.update(
            {
                "capacity_seed_profile_sha256": job["capacity_seed_profile_sha256"],
                "expected_regression_parameters": job["expected_regression_parameters"],
                "fixed_lr_profile": FIXED_LR_PROFILE,
                "fixed_lr_profile_sha256": _fixed_lr_profile_sha256(),
                "initial_learning_rate": FIXED_LEARNING_RATE,
                "scheduler_min_lr": FIXED_SCHEDULER_MIN_LR,
                "lr_trace": json.dumps(
                    status.get("lr_trace", job["lr_trace"]), separators=(",", ":")
                ),
                "seed": job["seed"],
                "experiment_stage": EXPERIMENT_STAGE,
            }
        )
    return _write_csv(root / "resource_summary.csv", rows, RESOURCE_FIELDS)


def _run_postprocess(root: Path) -> None:
    """Run Q3-only capacity analysis/report after all jobs complete."""

    from scripts.rq3.news_first_vol_fixed_lr_capacity_analysis import (
        run_fixed_lr_capacity_analysis,
    )
    from scripts.rq3.news_first_vol_fixed_lr_capacity_report import (
        render_fixed_lr_capacity_report,
    )

    run_fixed_lr_capacity_analysis(root)
    render_fixed_lr_capacity_report(root)


def _mark_status(root: Path, status: str, **details: Any) -> None:
    _write_json(
        root / "registry" / "experiment_status.json",
        {
            "status": status,
            "current_stage": EXPERIMENT_STAGE,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
            "updated_at_utc": _utc_now(),
            **details,
        },
    )


def launch_fixed_lr_capacity_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
    postprocess_hook: Callable[[Path], None] | None = None,
) -> Path:
    root = prepare_fixed_lr_capacity_experiment(config_path, output_dir, reuse=True)
    previous = _read_json(root / "registry" / "experiment_status.json")
    if previous.get("status") in TERMINAL_EXPERIMENT_STATES and not dry_run:
        if not resume:
            raise RuntimeError("Terminal fixed-LR capacity sweep requires --resume")
        if previous.get("status") == "completed_q3_only":
            _validate_registry(root)
            refresh_fixed_lr_capacity_lineage(config_path, root)
            _refresh_exports(root)
            return root
    if previous.get("status") in {"running", "failed"} and not (resume or dry_run):
        raise RuntimeError("Interrupted fixed-LR capacity sweep requires --resume")
    _mark_status(root, "dry_running" if dry_run else "running")
    try:
        waves = sorted({int(job["wave"]) for job in _load_registry(root)["jobs"]})
        for wave in waves:
            jobs = _jobs_for_wave(root, wave, resume=resume, dry_run=dry_run)
            _run_wave(
                root,
                config_path,
                wave,
                jobs,
                dry_run=dry_run,
                resume=resume,
            )
            _validate_wave_completion(root, jobs, dry_run=dry_run)
            _write_resource_summary(root)
            _refresh_exports(root)
        failures = []
        for job in _load_registry(root)["jobs"]:
            status = _read_json(_job_status_path(root, str(job["job_id"])))
            valid = (
                status.get("status") == "dry_run_passed"
                if dry_run
                else _completed_job_is_valid(root, job, status)
            )
            if not valid:
                failures.append(f"{job['job_id']}={status.get('status', 'missing')}")
        if failures:
            raise RuntimeError(f"Fixed-LR capacity matrix incomplete: {failures}")
        if dry_run:
            _mark_status(root, "dry_run_passed", completed_at_utc=_utc_now())
        else:
            (postprocess_hook or _run_postprocess)(root)
            _mark_status(root, "completed_q3_only", completed_at_utc=_utc_now())
    except BaseException as exc:
        _mark_status(root, "failed", error=f"{type(exc).__name__}: {exc}")
        refresh_fixed_lr_capacity_lineage(config_path, root)
        _refresh_exports(root)
        raise
    _write_resource_summary(root)
    refresh_fixed_lr_capacity_lineage(config_path, root)
    _refresh_exports(root)
    return root


def run_news_first_vol_fixed_lr_capacity_sweep(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    action: str = "launch",
    job_id: str = "",
    resume: bool = False,
    reuse: bool = False,
    worker_dry_run: bool = False,
    postprocess_hook: Callable[[Path], None] | None = None,
) -> Path:
    action = str(action).strip().lower()
    if action == "prepare":
        return prepare_fixed_lr_capacity_experiment(
            config_path, output_dir, reuse=bool(reuse or resume)
        )
    if action == "dry-run":
        return launch_fixed_lr_capacity_experiment(
            config_path, output_dir, resume=resume, dry_run=True
        )
    if action == "worker":
        if not job_id:
            raise ValueError("--job-id is required for worker")
        return run_fixed_lr_capacity_worker(
            output_dir, job_id, dry_run=worker_dry_run, resume=resume
        )
    if action == "launch":
        return launch_fixed_lr_capacity_experiment(
            config_path,
            output_dir,
            resume=resume,
            dry_run=False,
            postprocess_hook=postprocess_hook,
        )
    raise ValueError(f"Unsupported fixed-LR capacity action: {action}")


__all__ = [
    "EXPERIMENT_KIND",
    "EXPERIMENT_STAGE",
    "FIXED_LR_PROFILE",
    "FIXED_LEARNING_RATE",
    "FIXED_SCHEDULER_MIN_LR",
    "FROZEN_PROFILES",
    "FROZEN_CAPACITY_PROFILES",
    "FROZEN_SEEDS",
    "FROZEN_TOLERANCES",
    "FROZEN_TEXT_MODES",
    "WAVE_CELL_ORDER",
    "_capacity_profile_sha256",
    "_capacity_seed_profile_sha256",
    "_fixed_lr_profile_sha256",
    "build_fixed_lr_capacity_worker_command",
    "launch_fixed_lr_capacity_experiment",
    "prepare_fixed_lr_capacity_experiment",
    "refresh_fixed_lr_capacity_lineage",
    "run_fixed_lr_capacity_worker",
    "run_news_first_vol_fixed_lr_capacity_sweep",
]
