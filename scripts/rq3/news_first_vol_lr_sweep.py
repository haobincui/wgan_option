"""Fail-closed, Q3-only learning-rate sweep for news-first vol regression.

This is intentionally a separate branch-local experiment rather than another
mode of the capacity sweep.  It reuses the proven dataset split and trainer
plumbing, but gives learning-rate identity, scheduler floor, selection lineage,
and resume validation their own immutable registry contract.
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

from scripts.rq3 import news_first_vol_training as training
from wgan_option.utils.text_ablation import (
    REAL_TEXT,
    normalize_text_ablation_mode,
    text_information_path,
)


ROOT_KEY = training.ROOT_KEY
REPO_ROOT = training.REPO_ROOT
LR_PROFILE_IDS = (
    "lr_1e_06",
    "lr_3e_06",
    "lr_1e_05",
    "lr_3e_05",
    "lr_1e_04",
)
FROZEN_LEARNING_RATES = {
    "lr_1e_06": 1.0e-6,
    "lr_3e_06": 3.0e-6,
    "lr_1e_05": 1.0e-5,
    "lr_3e_05": 3.0e-5,
    "lr_1e_04": 1.0e-4,
}
FROZEN_SCHEDULER_MIN_LRS = {
    "lr_1e_06": 1.0e-7,
    "lr_3e_06": 3.0e-7,
    "lr_1e_05": 1.0e-6,
    "lr_3e_05": 3.0e-6,
    "lr_1e_04": 1.0e-5,
}
LR_STAGES = ("lr_screen", "lr_confirm")
SCREEN_TOLERANCES = (5, 30)
CONFIRM_TOLERANCES = (10, 15)
LR_TEXT_MODES = ("current_only", REAL_TEXT)
CAPACITY_PROFILE = "large"
LARGE_PROFILE = deepcopy(training.FROZEN_CAPACITY_PROFILES[CAPACITY_PROFILE])
TERMINAL_EXPERIMENT_STATES = {
    "completed_q3_only",
    "completed_no_learned_lr",
    "dry_run_passed",
}
SUCCESSFUL_JOB_STATES = {"completed", "dry_run_passed"}
EXPERIMENT_KIND = "learning_rate_sweep"

_read_json = training._read_json
_sha256_file = training._sha256_file
_payload_sha256 = training._payload_sha256
_write_json = training._write_json
_write_yaml = training._write_yaml
_write_csv = training._write_csv
_atomic_write_text = training._atomic_write_text
_utc_now = training._utc_now
_require_mapping = training._require_mapping
_resolve_repo_path = training._resolve_repo_path
_job_status_path = training._job_status_path
_build_split_manifest = training._build_split_manifest


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _require_mapping(payload, label)


def _validate_exact_float(value: Any, expected: float, label: str) -> None:
    observed = float(value)
    if not math.isfinite(observed) or observed != expected:
        raise ValueError(f"{label} is frozen to {expected}; observed={observed}")


def _large_profile_from_source(path: Path) -> tuple[dict[str, int], str]:
    root = _load_yaml_mapping(path, "capacity profile source")
    config = _require_mapping(root.get(ROOT_KEY), ROOT_KEY)
    sweep = _require_mapping(config.get("capacity_sweep"), "capacity_sweep")
    profiles = _require_mapping(sweep.get("profiles"), "capacity_sweep.profiles")
    profile = _require_mapping(profiles.get(CAPACITY_PROFILE), "capacity profile large")
    allowed = set(
        training.CAPACITY_PROFILE_SHAPE_FIELDS
        + training.CAPACITY_PROFILE_METADATA_FIELDS
    )
    if set(profile) != allowed:
        raise ValueError(
            "The referenced large capacity profile must contain exactly the frozen "
            f"shape/count fields; missing={sorted(allowed-set(profile))}, "
            f"unknown={sorted(set(profile)-allowed)}"
        )
    normalized = {field: int(profile[field]) for field in allowed}
    if normalized != LARGE_PROFILE:
        raise ValueError(
            "The referenced large profile differs from the frozen 12/96/48/384 "
            f"contract: expected={LARGE_PROFILE}, observed={normalized}"
        )
    return normalized, training._capacity_profile_sha256(CAPACITY_PROFILE, normalized)


def _lr_sweep_config(config: Mapping[str, Any]) -> dict[str, Any]:
    sweep = _require_mapping(config.get("lr_sweep"), "lr_sweep")
    if not bool(sweep.get("enabled", False)):
        raise ValueError("The LR-sweep command requires lr_sweep.enabled=true")
    return sweep


def _validate_frozen_config(config: Mapping[str, Any]) -> None:
    datasets = _require_mapping(config.get("datasets"), "datasets")
    split = _require_mapping(config.get("split"), "split")
    runtime = _require_mapping(config.get("runtime"), "runtime")
    models = _require_mapping(config.get("models"), "models")
    sweep = _lr_sweep_config(config)

    if tuple(int(value) for value in datasets.get("tolerances_minutes", ())) != training.TOLERANCES:
        raise ValueError("datasets.tolerances_minutes is frozen to 5/10/15/30")
    if int(datasets.get("common_evaluation_tolerance_minutes", -1)) != 5:
        raise ValueError("common_evaluation_tolerance_minutes is frozen to 5")
    if str(datasets.get("sheet_name", "")) != "gan_input_ready":
        raise ValueError("sheet_name is frozen to gan_input_ready")
    if str(datasets.get("text_embedding_mode", "")).lower() != "lp":
        raise ValueError("text_embedding_mode is frozen to lp")
    if int(datasets.get("seed", -1)) != 42:
        raise ValueError("seed is frozen to 42")
    if training._support_mask_mode(datasets) != "raw_joint":
        raise ValueError("The LR sweep requires support_mask_mode=raw_joint")
    dataset_modes = tuple(training._configured_text_ablation_modes(datasets))
    if dataset_modes != LR_TEXT_MODES:
        raise ValueError(
            f"datasets.text_ablation_modes is frozen to {list(LR_TEXT_MODES)}"
        )
    if int(datasets.get("text_shuffle_seed", -1)) != 42:
        raise ValueError("text_shuffle_seed is frozen to 42")

    if str(split.get("train_end_utc")) != "2023-07-01T00:00:00Z":
        raise ValueError("train_end_utc is frozen to 2023-07-01T00:00:00Z")
    if str(split.get("validation_end_utc")) != "2023-10-01T00:00:00Z":
        raise ValueError("validation_end_utc is frozen to 2023-10-01T00:00:00Z")
    if int(split.get("validation_mc_samples", -1)) != 16:
        raise ValueError("validation_mc_samples is frozen to 16")

    if str(sweep.get("capacity_profile", "")).strip().lower() != CAPACITY_PROFILE:
        raise ValueError("lr_sweep.capacity_profile is frozen to large")
    raw_profiles = _require_mapping(sweep.get("profiles"), "lr_sweep.profiles")
    if tuple(str(value) for value in raw_profiles) != LR_PROFILE_IDS:
        raise ValueError(
            "lr_sweep.profiles must preserve the frozen IDs and order: "
            f"{list(LR_PROFILE_IDS)}"
        )
    for profile_id, expected in FROZEN_LEARNING_RATES.items():
        _validate_exact_float(
            raw_profiles[profile_id], expected, f"lr_sweep.profiles.{profile_id}"
        )
    if tuple(int(value) for value in sweep.get("screen_tolerances_minutes", ())) != SCREEN_TOLERANCES:
        raise ValueError("screen_tolerances_minutes is frozen to [5, 30]")
    if tuple(int(value) for value in sweep.get("confirm_tolerances_minutes", ())) != CONFIRM_TOLERANCES:
        raise ValueError("confirm_tolerances_minutes is frozen to [10, 15]")
    if tuple(
        normalize_text_ablation_mode(value)
        for value in sweep.get("text_ablation_modes", ())
    ) != LR_TEXT_MODES:
        raise ValueError("lr_sweep.text_ablation_modes is frozen to current_only/real_text")
    selection = _require_mapping(sweep.get("selection"), "lr_sweep.selection")
    if str(selection.get("selection_panel", "")) != "common_validation_05m":
        raise ValueError("LR selection is frozen to common_validation_05m (Q3)")
    if str(selection.get("screen_selection_mode", "")) != "current_only":
        raise ValueError("LR screen_selection_mode is frozen to current_only")
    if str(selection.get("selection_rule", "")) != "one_standard_error_lower_lr":
        raise ValueError(
            "LR selection_rule is frozen to one_standard_error_lower_lr"
        )
    _validate_exact_float(
        selection.get("minimum_mean_improvement_fraction", -1),
        0.005,
        "minimum_mean_improvement_fraction",
    )
    if int(selection.get("bootstrap_clusters", -1)) != 10_000:
        raise ValueError("bootstrap_clusters is frozen to 10000")

    if set(models) != {"regression"}:
        raise ValueError("The LR sweep permits only the Regression model")
    model = _require_mapping(models["regression"], "models.regression")
    if str(model.get("trainer_command", "")) != "vol-regression-xlsx":
        raise ValueError("Regression trainer_command is frozen to vol-regression-xlsx")
    values = _require_mapping(model.get("training"), "models.regression.training")
    for field in training.CAPACITY_PROFILE_SHAPE_FIELDS:
        if int(values.get(field, -1)) != int(LARGE_PROFILE[field]):
            raise ValueError(f"Regression large shape mismatch: {field}")
    frozen_values = {
        "embedding_dim": 1024,
        "noise_dim": 32,
        "gen_res_blocks": 0,
        "num_epochs": 100,
        "early_stopping_min_epochs": 30,
        "early_stopping_patience": 12,
        "reduce_lr_patience": 3,
    }
    for field, expected in frozen_values.items():
        if int(values.get(field, -1)) != expected:
            raise ValueError(f"Regression {field} is frozen to {expected}")
    if str(values.get("residual_output_mode", "")).strip().lower() != "identity_softplus_residual":
        raise ValueError("residual_output_mode is frozen to identity_softplus_residual")
    if str(values.get("best_checkpoint_metric", "")) != "val_hybrid_score":
        raise ValueError("best_checkpoint_metric is frozen to val_hybrid_score")
    if not bool(values.get("evaluate_initial_checkpoint", False)):
        raise ValueError("evaluate_initial_checkpoint must remain enabled")
    if not bool(values.get("use_reduce_lr_on_plateau", False)):
        raise ValueError("ReduceLROnPlateau must remain enabled")
    if not bool(values.get("use_early_stopping", False)):
        raise ValueError("early stopping must remain enabled")
    _validate_exact_float(values.get("reduce_lr_factor", -1), 0.5, "reduce_lr_factor")
    if int(values.get("save_every", 0)) <= 100:
        raise ValueError("save_every must preserve the best/final-only checkpoint policy")

    if int(runtime.get("slots_per_gpu", -1)) != 2:
        raise ValueError("runtime.slots_per_gpu is frozen to 2")
    if int(runtime.get("cpu_threads_per_job", -1)) != 8:
        raise ValueError("runtime.cpu_threads_per_job is frozen to 8")
    gpu_ids = tuple(int(value) for value in runtime.get("gpu_ids", ()))
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("runtime.gpu_ids must contain exactly two distinct GPUs")


def _resolved_config(config_path: str | Path) -> dict[str, Any]:
    path = _resolve_repo_path(config_path)
    root = _load_yaml_mapping(path, "LR sweep config")
    config = _require_mapping(root.get(ROOT_KEY), ROOT_KEY)
    _validate_frozen_config(config)
    resolved = deepcopy(config)
    resolved["datasets"]["root"] = str(_resolve_repo_path(resolved["datasets"]["root"]))
    resolved["datasets"]["text_ablation_modes"] = list(LR_TEXT_MODES)
    resolved["datasets"]["support_mask_mode"] = "raw_joint"
    resolved["runtime"]["python_executable"] = str(
        _resolve_repo_path(resolved["runtime"].get("python_executable", sys.executable))
    )
    source_profile_path = _resolve_repo_path(
        resolved["lr_sweep"]["capacity_profile_source_config"]
    )
    profile, profile_sha = _large_profile_from_source(source_profile_path)
    resolved["lr_sweep"]["capacity_profile_source_config"] = str(source_profile_path)
    resolved["lr_sweep"]["large_profile"] = profile
    resolved["lr_sweep"]["large_profile_sha256"] = profile_sha
    resolved["source_config_path"] = str(path)
    return resolved


def _lr_profile_payload(
    profile_id: str,
    initial_learning_rate: float,
    *,
    capacity_profile_sha256: str,
) -> dict[str, Any]:
    profile = str(profile_id).strip().lower()
    if profile not in FROZEN_LEARNING_RATES:
        raise ValueError(f"Unknown LR profile: {profile}")
    value = float(initial_learning_rate)
    if value != FROZEN_LEARNING_RATES[profile]:
        raise ValueError(f"LR value does not match profile {profile}: {value}")
    return {
        "schema_version": 1,
        "lr_profile": profile,
        "initial_learning_rate": value,
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": 0.5,
            "patience": 3,
            "min_lr": FROZEN_SCHEDULER_MIN_LRS[profile],
        },
        "training_control": {
            "max_epochs": 100,
            "early_stopping_min_epochs": 30,
            "early_stopping_patience": 12,
        },
        "capacity_profile": CAPACITY_PROFILE,
        "capacity_profile_sha256": str(capacity_profile_sha256),
        "seed": 42,
        "text_embedding_mode": "lp",
        "support_mask_mode": "raw_joint",
        "residual_output_mode": "identity_softplus_residual",
    }


def _lr_profile_sha256(
    profile_id: str,
    initial_learning_rate: float | None = None,
    *,
    capacity_profile_sha256: str | None = None,
) -> str:
    value = (
        FROZEN_LEARNING_RATES[str(profile_id)]
        if initial_learning_rate is None
        else float(initial_learning_rate)
    )
    profile_sha = capacity_profile_sha256 or training._capacity_profile_sha256(
        CAPACITY_PROFILE, LARGE_PROFILE
    )
    return _payload_sha256(
        _lr_profile_payload(
            profile_id,
            value,
            capacity_profile_sha256=profile_sha,
        )
    )


def _lr_profile_manifest_rows(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    profile_sha = str(resolved["lr_sweep"]["large_profile_sha256"])
    return [
        {
            "lr_profile": name,
            "lr_profile_sha256": _lr_profile_sha256(
                name, value, capacity_profile_sha256=profile_sha
            ),
            "initial_learning_rate": value,
            "scheduler_min_lr": FROZEN_SCHEDULER_MIN_LRS[name],
            "scheduler_factor": 0.5,
            "scheduler_patience": 3,
            "max_epochs": 100,
            "early_stopping_min_epochs": 30,
            "early_stopping_patience": 12,
            "capacity_profile": CAPACITY_PROFILE,
            "capacity_profile_sha256": profile_sha,
            **{
                field: int(resolved["lr_sweep"]["large_profile"][field])
                for field in training.CAPACITY_PROFILE_SHAPE_FIELDS
            },
        }
        for name, value in FROZEN_LEARNING_RATES.items()
    ]


def _source_rows(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = training._source_rows(resolved)
    profile_path = Path(resolved["lr_sweep"]["capacity_profile_source_config"])
    rows.append(
        {
            "source_role": "large_capacity_profile_source",
            "path": str(profile_path),
            "size_bytes": profile_path.stat().st_size,
            "sha256": _sha256_file(profile_path),
        }
    )
    return rows


def _lr_job_id(profile: str, mode: str, tolerance: int) -> str:
    return (
        f"lr_regression_{CAPACITY_PROFILE}_{profile}_"
        f"{normalize_text_ablation_mode(mode)}_{int(tolerance):02d}m"
    )


def _stage_specs(stage: str, profiles: Sequence[str]) -> list[dict[str, Any]]:
    normalized = str(stage).strip().lower()
    if normalized not in LR_STAGES:
        raise ValueError(f"Unknown LR stage: {stage}")
    profile_ids = tuple(str(value).strip().lower() for value in profiles)
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("LR stage profiles must be unique")
    unknown = sorted(set(profile_ids) - set(LR_PROFILE_IDS))
    if unknown:
        raise ValueError(f"Unknown LR profiles: {unknown}")
    tolerances = SCREEN_TOLERANCES if normalized == "lr_screen" else CONFIRM_TOLERANCES
    return [
        {
            "model_family": "regression",
            "capacity_profile": CAPACITY_PROFILE,
            "lr_profile": profile,
            "text_ablation_mode": mode,
            "tolerance_minutes": tolerance,
            "lr_stage": normalized,
        }
        for profile in profile_ids
        for mode in LR_TEXT_MODES
        for tolerance in tolerances
    ]


def _selection_path(root: Path, stage: str) -> Path:
    return root / "registry" / "selections" / f"{stage}.json"


def _selection_hashes(root: Path) -> dict[str, str]:
    return {
        stage: _sha256_file(_selection_path(root, stage))
        for stage in LR_STAGES
        if _selection_path(root, stage).is_file()
    }


def _load_registry(root: Path) -> dict[str, Any]:
    registry = _require_mapping(
        _read_json(root / "registry" / "jobs.json"), "LR job registry"
    )
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment root is not a learning-rate sweep")
    return registry


def _training_payload(
    resolved: Mapping[str, Any],
    root: Path,
    *,
    profile_id: str,
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
    large = resolved["lr_sweep"]["large_profile"]
    payload.update(
        {field: int(large[field]) for field in training.CAPACITY_PROFILE_SHAPE_FIELDS}
    )
    value = FROZEN_LEARNING_RATES[profile_id]
    profile_sha = str(resolved["lr_sweep"]["large_profile_sha256"])
    lr_sha = _lr_profile_sha256(
        profile_id, value, capacity_profile_sha256=profile_sha
    )
    payload.update(
        {
            "news_first_capacity_profile": CAPACITY_PROFILE,
            "news_first_capacity_profile_sha256": profile_sha,
            "news_first_lr_profile": profile_id,
            "news_first_lr_profile_sha256": lr_sha,
            "learning_rate": value,
            "use_reduce_lr_on_plateau": True,
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": 3,
            "reduce_lr_min_lr": FROZEN_SCHEDULER_MIN_LRS[profile_id],
            "num_epochs": 100,
            "use_early_stopping": True,
            "early_stopping_min_epochs": 30,
            "early_stopping_patience": 12,
            "output_root": str(
                root
                / "runs"
                / "regression"
                / CAPACITY_PROFILE
                / profile_id
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


def _validate_resolved_snapshot(root: Path) -> dict[str, Any]:
    resolved_path = root / "resolved_config.yaml"
    payload = _load_yaml_mapping(resolved_path, "resolved LR-sweep config")
    resolved = _require_mapping(payload.get(ROOT_KEY), ROOT_KEY)
    observed = _payload_sha256(resolved)
    expected = (root / "registry" / "resolved_config.sha256").read_text(
        encoding="utf-8"
    ).strip()
    if observed != expected:
        raise ValueError("Resolved LR-sweep snapshot hash mismatch")
    registry = _require_mapping(
        _read_json(root / "registry" / "jobs.json"), "LR job registry"
    )
    if str(registry.get("resolved_config_sha256", "")) != expected:
        raise ValueError("LR registry resolved-config hash mismatch")
    return resolved


def _refresh_config_hashes(root: Path) -> Path:
    registry = _load_registry(root)
    resolved_path = root / "resolved_config.yaml"
    rows = [
        {
            "config_role": "resolved_orchestration",
            "path": str(resolved_path),
            "sha256": _sha256_file(resolved_path),
        }
    ]
    rows.extend(
        {
            "config_role": str(job["job_id"]),
            "path": str(job["training_config_path"]),
            "sha256": str(job["config_sha256"]),
        }
        for job in registry["jobs"]
    )
    return _write_csv(root / "config_hashes.csv", rows, tuple(rows[0]))


def _append_lr_stage_jobs(
    root: Path,
    resolved: Mapping[str, Any],
    *,
    stage: str,
    profiles: Sequence[str],
    selection_sha256: str = "",
) -> tuple[list[str], list[str]]:
    specs = _stage_specs(stage, profiles)
    registry = _load_registry(root)
    existing = {str(job["job_id"]): dict(job) for job in registry["jobs"]}
    source_hashes: dict[str, str] = {}
    with (root / "source_hashes.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_hashes[str(row["path"])] = str(row["sha256"])
    slots = training._capacity_worker_slots(resolved)
    next_wave = max((int(job["wave"]) for job in existing.values()), default=0) + 1
    required_ids: list[str] = []
    missing_specs: list[dict[str, Any]] = []
    for spec in specs:
        job_id = _lr_job_id(
            str(spec["lr_profile"]),
            str(spec["text_ablation_mode"]),
            int(spec["tolerance_minutes"]),
        )
        required_ids.append(job_id)
        if job_id in existing:
            prior = existing[job_id]
            immutable = {
                "model_family": "regression",
                "capacity_profile": CAPACITY_PROFILE,
                "lr_profile": spec["lr_profile"],
                "text_ablation_mode": spec["text_ablation_mode"],
                "tolerance_minutes": spec["tolerance_minutes"],
                "lr_stage": str(stage),
                "selection_sha256": str(selection_sha256),
            }
            mismatches = {
                key: (prior.get(key), expected)
                for key, expected in immutable.items()
                if prior.get(key) != expected
            }
            if mismatches:
                raise ValueError(f"Existing LR job conflicts with requested spec: {job_id}: {mismatches}")
            _validate_lr_job_lineage(root, prior)
            continue
        missing_specs.append(spec)

    selection_chain = _selection_hashes(root)
    selection_chain_sha = _payload_sha256(selection_chain) if selection_chain else ""
    new_jobs: list[dict[str, Any]] = []
    new_ids: list[str] = []
    for index, spec in enumerate(missing_specs):
        wave = next_wave + index // len(slots)
        gpu_id, gpu_slot, numa_node = slots[index % len(slots)]
        profile_id = str(spec["lr_profile"])
        mode = normalize_text_ablation_mode(spec["text_ablation_mode"])
        tolerance = int(spec["tolerance_minutes"])
        job_id = _lr_job_id(profile_id, mode, tolerance)
        payload = _training_payload(
            resolved,
            root,
            profile_id=profile_id,
            mode=mode,
            tolerance=tolerance,
        )
        config_path = root / "configs" / "jobs" / f"{job_id}.yaml"
        _write_yaml(config_path, payload)
        config_sha = _sha256_file(config_path)
        dataset_path = str(payload["data_path"])
        if dataset_path not in source_hashes:
            raise ValueError(f"Dataset source hash is unavailable for {dataset_path}")
        value = FROZEN_LEARNING_RATES[profile_id]
        lr_sha = str(payload["news_first_lr_profile_sha256"])
        job: dict[str, Any] = {
            "job_id": job_id,
            "wave": wave,
            "model_family": "regression",
            "trainer_command": "vol-regression-xlsx",
            "capacity_profile": CAPACITY_PROFILE,
            "capacity_profile_sha256": str(payload["news_first_capacity_profile_sha256"]),
            "expected_regression_parameters": int(
                resolved["lr_sweep"]["large_profile"]["expected_regression_parameters"]
            ),
            "lr_profile": profile_id,
            "lr_profile_sha256": lr_sha,
            "initial_learning_rate": value,
            "scheduler_min_lr": FROZEN_SCHEDULER_MIN_LRS[profile_id],
            "lr_trace": [value],
            "lr_stage": str(stage),
            "text_ablation_mode": mode,
            "text_information_path": text_information_path(mode),
            "support_mask_mode": "raw_joint",
            "tolerance_minutes": tolerance,
            "gpu_id": gpu_id,
            "gpu_slot": gpu_slot,
            "numa_node": numa_node,
            "training_config_path": str(config_path),
            "config_sha256": config_sha,
            "dataset_path": dataset_path,
            "dataset_sha256": source_hashes[dataset_path],
            "source_manifest_sha256": _sha256_file(root / "source_hashes.csv"),
            "output_root": str(payload["output_root"]),
            "selection_sha256": str(selection_sha256),
            "selection_chain_sha256": selection_chain_sha,
        }
        job["job_spec_sha256"] = _job_spec_sha256(job)
        new_jobs.append(job)
        new_ids.append(job_id)
        _write_json(
            _job_status_path(root, job_id),
            {
                "job_id": job_id,
                "status": "prepared",
                "attempt": 0,
                "config_sha256": config_sha,
                "capacity_profile": CAPACITY_PROFILE,
                "capacity_profile_sha256": job["capacity_profile_sha256"],
                "lr_profile": profile_id,
                "lr_profile_sha256": lr_sha,
                "initial_learning_rate": value,
                "scheduler_min_lr": FROZEN_SCHEDULER_MIN_LRS[profile_id],
                "lr_trace": [value],
                "lr_stage": str(stage),
                "updated_at_utc": _utc_now(),
            },
        )
    if new_jobs:
        registry["jobs"].extend(new_jobs)
        registry["updated_at_utc"] = _utc_now()
        _write_json(root / "registry" / "jobs.json", registry)

    state_path = root / "lr_stage_status.json"
    state = _read_json(state_path)
    stages = dict(state.get("stages") or {})
    previous = dict(stages.get(stage) or {})
    stages[stage] = {
        **previous,
        "status": previous.get("status", "prepared"),
        "profiles": list(profiles),
        "required_job_ids": required_ids,
        "new_job_ids": new_ids or list(previous.get("new_job_ids") or []),
        "selection_sha256": str(selection_sha256),
        "updated_at_utc": _utc_now(),
    }
    state.update(
        {
            "status": state.get("status", "prepared"),
            "current_stage": str(stage),
            "stages": stages,
            "updated_at_utc": _utc_now(),
        }
    )
    _write_json(state_path, state)
    _refresh_config_hashes(root)
    _refresh_registry_exports(root)
    return required_ids, new_ids


TASK_FIELDS = (
    "job_id",
    "wave",
    "model_family",
    "capacity_profile",
    "capacity_profile_sha256",
    "lr_profile",
    "lr_profile_sha256",
    "initial_learning_rate",
    "scheduler_min_lr",
    "lr_trace",
    "lr_stage",
    "text_ablation_mode",
    "text_information_path",
    "support_mask_mode",
    "tolerance_minutes",
    "gpu_id",
    "gpu_slot",
    "numa_node",
    "selection_sha256",
    "selection_chain_sha256",
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


def _refresh_registry_exports(root: Path) -> Path:
    registry = _load_registry(root)
    rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for job in registry["jobs"]:
        status_path = _job_status_path(root, str(job["job_id"]))
        status = _read_json(status_path) if status_path.is_file() else {"status": "missing"}
        row = {**job, **status}
        row["lr_trace"] = json.dumps(row.get("lr_trace", []), separators=(",", ":"))
        rows.append(row)
        for artifact in status.get("artifacts", []):
            output_rows.append({"job_id": job["job_id"], **artifact})
    _write_csv(root / "task_registry.csv", rows, TASK_FIELDS)
    stable = [
        root / "task_registry.csv",
        root / "split_manifest.csv",
        root / "text_ablation_manifest.csv",
        root / "resource_usage.csv",
        root / "resource_summary.csv",
        root / "config_hashes.csv",
        root / "source_hashes.csv",
        root / "code_hashes.csv",
        root / "code_hashes_current.csv",
        root / "run_manifest.json",
        root / "lr_profile_manifest.csv",
        root / "lr_comparisons.csv",
        root / "lr_selection_audit.csv",
        root / "lr_stage_status.json",
        root / "lr_selection.json",
        root / "registry" / "jobs.json",
        root / "registry" / "experiment_status.json",
        root / "registry" / "resolved_config.sha256",
    ]
    status_dir = root / "registry" / "jobs"
    if status_dir.is_dir():
        stable.extend(status_dir.glob("*.status.json"))
    selection_dir = root / "registry" / "selections"
    if selection_dir.is_dir():
        stable.extend(selection_dir.glob("*.json"))
    for directory_name in ("analysis", "report"):
        directory = root / directory_name
        if directory.is_dir():
            stable.extend(path for path in directory.rglob("*") if path.is_file())
    seen = {str(row.get("path", "")) for row in output_rows}
    for path in stable:
        if not path.is_file() or str(path) in seen:
            continue
        output_rows.append(
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
        output_rows,
        ("job_id", "artifact_role", "path", "size_bytes", "sha256"),
    )
    return root / "task_registry.csv"


def _code_rows() -> list[dict[str, Any]]:
    rows = training._code_rows()
    existing = {str(row["relative_path"]) for row in rows}
    for relative in (
        "scripts/rq3/news_first_vol_lr_sweep.py",
        "scripts/rq3/news_first_vol_lr_analysis.py",
        "scripts/rq3/news_first_vol_lr_report.py",
    ):
        if relative in existing:
            continue
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required LR experiment code is missing: {path}")
        rows.append(
            {
                "relative_path": relative,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _maximum_attempt(root: Path) -> int:
    attempts = []
    if not (root / "registry" / "jobs.json").is_file():
        return 0
    for job in _load_registry(root)["jobs"]:
        path = _job_status_path(root, str(job["job_id"]))
        if path.is_file():
            attempts.append(int(_read_json(path).get("attempt", 0)))
    return max(attempts, default=0)


def refresh_lr_sweep_lineage(config_path: str | Path, root: str | Path) -> Path:
    experiment_root = Path(root).resolve(strict=False)
    expected_path = experiment_root / "registry" / "resolved_config.sha256"
    if not expected_path.is_file():
        raise ValueError(f"Not a prepared LR-sweep root: {experiment_root}")
    resolved = _resolved_config(config_path)
    resolved_sha = _payload_sha256(resolved)
    if expected_path.read_text(encoding="utf-8").strip() != resolved_sha:
        raise ValueError("Prepared LR-sweep config hash differs from requested config")
    source_rows = _source_rows(resolved)
    prepared_sources = {}
    source_path = experiment_root / "source_hashes.csv"
    if source_path.is_file():
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            prepared_sources = {row["path"]: row["sha256"] for row in csv.DictReader(handle)}
    current_sources = {row["path"]: row["sha256"] for row in source_rows}
    if prepared_sources and prepared_sources != current_sources:
        raise ValueError("LR-sweep source lineage changed after preparation")
    _write_csv(source_path, source_rows, ("source_role", "path", "size_bytes", "sha256"))

    code_rows = _code_rows()
    code_path = experiment_root / "code_hashes.csv"
    current_map = {row["relative_path"]: row["sha256"] for row in code_rows}
    prepared_map: dict[str, str] = {}
    if code_path.is_file():
        with code_path.open("r", encoding="utf-8", newline="") as handle:
            prepared_map = {row["relative_path"]: row["sha256"] for row in csv.DictReader(handle)}
    changed = bool(prepared_map and prepared_map != current_map)
    current_path: Path | None = None
    if not prepared_map or _maximum_attempt(experiment_root) == 0:
        _write_csv(code_path, code_rows, ("relative_path", "path", "size_bytes", "sha256"))
        changed = False
    elif changed:
        current_path = experiment_root / "code_hashes_current.csv"
        _write_csv(current_path, code_rows, ("relative_path", "path", "size_bytes", "sha256"))
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
        "q4_access_policy": "forbidden_during_lr_selection",
        "updated_at_utc": _utc_now(),
    }
    _write_json(experiment_root / "run_manifest.json", manifest)
    return experiment_root / "run_manifest.json"


def _validate_lr_job_lineage(root: Path, job: Mapping[str, Any]) -> None:
    if str(job.get("job_spec_sha256", "")) != _job_spec_sha256(job):
        raise ValueError(f"LR job-spec hash mismatch for {job.get('job_id')}")
    profile_id = str(job.get("lr_profile", ""))
    if profile_id not in FROZEN_LEARNING_RATES:
        raise ValueError(f"LR job references an unknown profile: {profile_id}")
    resolved = _validate_resolved_snapshot(root)
    source_manifest = root / "source_hashes.csv"
    if _sha256_file(source_manifest) != str(job.get("source_manifest_sha256", "")):
        raise ValueError(f"Source-manifest hash mismatch for {job['job_id']}")
    with source_manifest.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    source_by_role = {str(row["source_role"]): row for row in source_rows}
    required_source_roles = {
        f"support_audit_{int(job['tolerance_minutes']):02d}m",
        "orchestration_config",
        "large_capacity_profile_source",
    }
    missing_sources = sorted(required_source_roles - set(source_by_role))
    if missing_sources:
        raise ValueError(
            f"Source manifest is missing LR job dependencies: {missing_sources}"
        )
    for role in required_source_roles:
        row = source_by_role[role]
        path = Path(str(row["path"]))
        if not path.is_file() or _sha256_file(path) != str(row["sha256"]):
            raise ValueError(f"Source hash mismatch for {job['job_id']}: {role}")
    capacity_sha = str(resolved["lr_sweep"]["large_profile_sha256"])
    expected_lr_sha = _lr_profile_sha256(
        profile_id,
        FROZEN_LEARNING_RATES[profile_id],
        capacity_profile_sha256=capacity_sha,
    )
    expected = FROZEN_LEARNING_RATES[profile_id]
    if str(job.get("lr_profile_sha256", "")) != expected_lr_sha:
        raise ValueError(f"LR profile hash mismatch for {job['job_id']}")
    if float(job.get("initial_learning_rate", -1)) != expected:
        raise ValueError(f"Initial learning rate mismatch for {job['job_id']}")
    if float(job.get("scheduler_min_lr", -1)) != FROZEN_SCHEDULER_MIN_LRS[profile_id]:
        raise ValueError(f"Scheduler min_lr mismatch for {job['job_id']}")
    if str(job.get("capacity_profile", "")) != CAPACITY_PROFILE:
        raise ValueError(f"Capacity profile mismatch for {job['job_id']}")
    if str(job.get("capacity_profile_sha256", "")) != capacity_sha:
        raise ValueError(f"Capacity profile hash mismatch for {job['job_id']}")
    config_path = Path(str(job["training_config_path"]))
    if not config_path.is_file() or _sha256_file(config_path) != str(job["config_sha256"]):
        raise ValueError(f"Training config hash mismatch for {job['job_id']}")
    payload = _load_yaml_mapping(config_path, f"training config {job['job_id']}")
    contracts = {
        "news_first_capacity_profile": CAPACITY_PROFILE,
        "news_first_capacity_profile_sha256": capacity_sha,
        "news_first_lr_profile": profile_id,
        "news_first_lr_profile_sha256": expected_lr_sha,
        "support_mask_mode": "raw_joint",
        "residual_output_mode": "identity_softplus_residual",
        "early_stopping_min_epochs": 30,
        "early_stopping_patience": 12,
        "reduce_lr_patience": 3,
        "num_epochs": 100,
    }
    for field, expected_value in contracts.items():
        if payload.get(field) != expected_value:
            raise ValueError(f"Training LR contract mismatch for {job['job_id']}: {field}")
    for field in training.CAPACITY_PROFILE_SHAPE_FIELDS:
        if int(payload.get(field, -1)) != int(LARGE_PROFILE[field]):
            raise ValueError(f"Training large shape mismatch for {job['job_id']}: {field}")
    for field, expected_value in {
        "learning_rate": expected,
        "reduce_lr_min_lr": FROZEN_SCHEDULER_MIN_LRS[profile_id],
        "reduce_lr_factor": 0.5,
    }.items():
        if float(payload.get(field, -1)) != expected_value:
            raise ValueError(f"Training LR contract mismatch for {job['job_id']}: {field}")
    expected_output = (
        root
        / "runs"
        / "regression"
        / CAPACITY_PROFILE
        / profile_id
        / str(job["text_ablation_mode"])
        / f"tolerance_{int(job['tolerance_minutes']):02d}m"
    )
    if Path(str(job["output_root"])) != expected_output or Path(str(payload["output_root"])) != expected_output:
        raise ValueError(f"LR output path mismatch for {job['job_id']}")
    stage = str(job.get("lr_stage", ""))
    if stage not in LR_STAGES:
        raise ValueError(f"Unknown LR stage for {job['job_id']}: {stage}")
    parents = () if stage == "lr_screen" else ("lr_screen",)
    hashes: dict[str, str] = {}
    for parent in parents:
        path = _selection_path(root, parent)
        if not path.is_file():
            raise ValueError(f"LR job is missing frozen parent selection: {parent}")
        hashes[parent] = _sha256_file(path)
    expected_parent = hashes[parents[-1]] if parents else ""
    if str(job.get("selection_sha256", "")) != expected_parent:
        raise ValueError(f"LR selection hash mismatch for {job['job_id']}")
    expected_chain = _payload_sha256(hashes) if hashes else ""
    if str(job.get("selection_chain_sha256", "")) != expected_chain:
        raise ValueError(f"LR selection-chain hash mismatch for {job['job_id']}")


def _validate_registry(root: Path) -> None:
    registry = _load_registry(root)
    jobs = [dict(job) for job in registry.get("jobs", [])]
    by_id = {str(job["job_id"]): job for job in jobs}
    if len(by_id) != len(jobs):
        raise ValueError("LR registry contains duplicate job IDs")
    expected_screen = {
        _lr_job_id(profile, mode, tolerance)
        for profile in LR_PROFILE_IDS
        for mode in LR_TEXT_MODES
        for tolerance in SCREEN_TOLERANCES
    }
    if not expected_screen.issubset(by_id):
        raise ValueError("LR registry is missing one or more frozen screen jobs")
    allowed = set(expected_screen)
    screen_path = _selection_path(root, "lr_screen")
    if screen_path.is_file():
        selection = _read_json(screen_path)
        if bool(selection.get("gate_passed")):
            profile = _selected_lr_profile(selection)
            allowed.update(
                _lr_job_id(profile, mode, tolerance)
                for mode in LR_TEXT_MODES
                for tolerance in CONFIRM_TOLERANCES
            )
    unknown = sorted(set(by_id) - allowed)
    if unknown:
        raise ValueError(f"LR registry contains out-of-contract jobs: {unknown}")
    for job in jobs:
        _validate_lr_job_lineage(root, job)


def prepare_lr_sweep_experiment(
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
            raise FileExistsError(f"Experiment root exists; use --resume/--reuse: {root}")
        if not expected_path.is_file():
            raise ValueError(f"Existing directory is not a prepared LR sweep: {root}")
        if expected_path.read_text(encoding="utf-8").strip() != resolved_sha:
            raise ValueError("Prepared LR-sweep config hash differs from requested config")
        _validate_registry(root)
        refresh_lr_sweep_lineage(config_path, root)
        _refresh_registry_exports(root)
        return root

    training._validate_dataset_summary(Path(resolved["datasets"]["root"]))
    source_rows = _source_rows(resolved)
    root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "registry/jobs",
        "registry/selections",
        "configs/jobs",
        "logs",
        "resources",
        "runs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _build_split_manifest(resolved, root)
    _write_yaml(root / "resolved_config.yaml", {ROOT_KEY: resolved})
    _atomic_write_text(expected_path, resolved_sha + "\n")
    _write_json(
        root / "registry" / "jobs.json",
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "experiment_root": str(root),
            "resolved_config_sha256": resolved_sha,
            "created_at_utc": _utc_now(),
            "jobs": [],
        },
    )
    _write_json(
        root / "registry" / "experiment_status.json",
        {"status": "prepared", "current_stage": "lr_screen", "updated_at_utc": _utc_now()},
    )
    _write_json(
        root / "lr_stage_status.json",
        {
            "schema_version": 1,
            "status": "prepared",
            "current_stage": "lr_screen",
            "q4_access_policy": "forbidden",
            "stages": {},
            "updated_at_utc": _utc_now(),
        },
    )
    _write_csv(root / "source_hashes.csv", source_rows, ("source_role", "path", "size_bytes", "sha256"))
    manifest_rows = _lr_profile_manifest_rows(resolved)
    _write_csv(root / "lr_profile_manifest.csv", manifest_rows, tuple(manifest_rows[0]))
    _append_lr_stage_jobs(root, resolved, stage="lr_screen", profiles=LR_PROFILE_IDS)
    _validate_registry(root)
    refresh_lr_sweep_lineage(config_path, root)
    _refresh_registry_exports(root)
    return root


def _completed_job_is_valid(root: Path, job: Mapping[str, Any], status: Mapping[str, Any]) -> bool:
    try:
        _validate_lr_job_lineage(root, job)
    except (ValueError, FileNotFoundError):
        return False
    return training._completed_job_is_valid(root, job, status)


def _validated_run_lr_trace(
    job: Mapping[str, Any], run_dir: Path, *, dry_run: bool
) -> list[dict[str, float | int]] | list[float]:
    """Read the trainer's real epoch trace and verify its immutable LR identity."""

    if dry_run:
        return [float(job["initial_learning_rate"])]
    checkpoint = _require_mapping(
        _read_json(run_dir / "metrics" / "best_learned_checkpoint.json"),
        f"best-learned checkpoint for {job['job_id']}",
    )
    expected_contract = {
        "lr_profile": str(job["lr_profile"]),
        "lr_profile_sha256": str(job["lr_profile_sha256"]),
        "initial_learning_rate": float(job["initial_learning_rate"]),
        "scheduler_min_lr": float(job["scheduler_min_lr"]),
    }
    for field, expected in expected_contract.items():
        observed = checkpoint.get(field)
        if observed != expected:
            raise ValueError(
                f"Trainer LR metadata mismatch for {job['job_id']}: "
                f"{field}={observed!r}, expected={expected!r}"
            )
    raw_rows = _read_json(run_dir / "metrics" / "training_metrics.json")
    if not isinstance(raw_rows, list) or len(raw_rows) < 2:
        raise ValueError(
            f"Trainer LR trace for {job['job_id']} must contain epoch 0 and a learned epoch"
        )
    result: list[dict[str, float | int]] = []
    previous_lr = float(job["initial_learning_rate"])
    floor = float(job["scheduler_min_lr"])
    for raw in raw_rows:
        row = _require_mapping(raw, f"training metric row for {job['job_id']}")
        epoch = int(row["epoch"])
        value = float(row["lr"])
        if not math.isfinite(value) or value < floor or value > previous_lr:
            raise ValueError(
                f"Trainer LR trace violates the non-increasing floor contract for "
                f"{job['job_id']}: epoch={epoch}, lr={value}"
            )
        result.append({"epoch": epoch, "lr": value})
        previous_lr = value
    if int(result[0]["epoch"]) != 0 or float(result[0]["lr"]) != float(
        job["initial_learning_rate"]
    ):
        raise ValueError(f"Trainer LR trace has an invalid epoch-0 origin for {job['job_id']}")
    return result


def run_lr_sweep_worker(
    experiment_root: str | Path,
    job_id: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    root = _resolve_repo_path(experiment_root)
    registry = _load_registry(root)
    matches = [dict(job) for job in registry["jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicate LR job_id: {job_id}")
    job = matches[0]
    _validate_lr_job_lineage(root, job)
    actual_dataset_sha = _sha256_file(job["dataset_path"])
    if actual_dataset_sha.lower() != str(job["dataset_sha256"]).lower():
        raise ValueError(f"Dataset hash mismatch for {job_id}")
    status_path = _job_status_path(root, job_id)
    previous = _read_json(status_path)
    if previous.get("status") == "completed" and not dry_run:
        if resume and _completed_job_is_valid(root, job, previous):
            return Path(previous["run_dir"])
        if not resume:
            raise RuntimeError(f"Job already completed: {job_id}")
    if previous.get("status") == "running" and training._pid_is_live(previous.get("pid")):
        raise RuntimeError(f"Job is already running: {job_id}")
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    assigned = str(job["gpu_id"])
    if visible and visible.split(",")[0].strip() != assigned:
        raise RuntimeError(f"Worker GPU mismatch: assigned={assigned}, visible={visible}")
    os.environ["CUDA_VISIBLE_DEVICES"] = assigned
    resolved = _load_yaml_mapping(root / "resolved_config.yaml", "resolved config")[ROOT_KEY]
    threads = str(int(resolved["runtime"]["cpu_threads_per_job"]))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = threads
    attempt = int(previous.get("attempt", 0)) + 1
    common = {
        "job_id": job_id,
        "attempt": attempt,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "gpu_id": job["gpu_id"],
        "config_sha256": job["config_sha256"],
        "capacity_profile": CAPACITY_PROFILE,
        "capacity_profile_sha256": job["capacity_profile_sha256"],
        "lr_profile": job["lr_profile"],
        "lr_profile_sha256": job["lr_profile_sha256"],
        "initial_learning_rate": job["initial_learning_rate"],
        "scheduler_min_lr": job["scheduler_min_lr"],
        "lr_trace": list(job["lr_trace"]),
        "lr_stage": job["lr_stage"],
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "dry_run": bool(dry_run),
        "log_path": str(os.environ.get("NEWS_FIRST_JOB_LOG_PATH", "")),
    }
    _write_json(status_path, {**common, "status": "running"})
    try:
        run_dir, artifacts = training._execute_training_job(job, dry_run=dry_run)
        lr_trace = _validated_run_lr_trace(job, run_dir, dry_run=dry_run)
        _write_json(
            status_path,
            {
                **common,
                "status": "dry_run_passed" if dry_run else "completed",
                "run_dir": str(run_dir),
                "artifacts": artifacts,
                "lr_trace": lr_trace,
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


def build_lr_worker_command(
    config_path: str | Path,
    root: str | Path,
    job: Mapping[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> list[str]:
    experiment_root = Path(root)
    resolved = _load_yaml_mapping(experiment_root / "resolved_config.yaml", "resolved config")[ROOT_KEY]
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
            "train-news-first-vol-lr-sweep",
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


def _jobs_for_wave(root: Path, wave: int, *, resume: bool, dry_run: bool) -> list[dict[str, Any]]:
    jobs = [dict(job) for job in _load_registry(root)["jobs"] if int(job["wave"]) == int(wave)]
    selected: list[dict[str, Any]] = []
    for job in jobs:
        _validate_lr_job_lineage(root, job)
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        if not dry_run and status.get("status") == "completed":
            if _completed_job_is_valid(root, job, status):
                if resume:
                    continue
                raise RuntimeError(f"Job already completed; use --resume: {job['job_id']}")
            if not resume:
                raise RuntimeError(f"Completed job is invalid and requires --resume: {job['job_id']}")
        if status.get("status") == "running" and training._pid_is_live(status.get("pid")):
            raise RuntimeError(f"Refusing duplicate live job: {job['job_id']}")
        if status.get("status") in {"running", "failed"} and not (resume or dry_run):
            raise RuntimeError(f"Interrupted/failed job requires --resume: {job['job_id']}")
        selected.append(job)
    return selected


def _validate_wave_completion(root: Path, jobs: Sequence[Mapping[str, Any]], *, dry_run: bool) -> None:
    failures = []
    for job in jobs:
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        valid = status.get("status") == "dry_run_passed" if dry_run else _completed_job_is_valid(root, job, status)
        if not valid:
            failures.append(f"{job['job_id']}={status.get('status', 'missing')}")
    if failures:
        raise RuntimeError(f"LR wave did not produce valid terminal states: {failures}")


def _run_lr_wave(
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
    resolved = _load_yaml_mapping(root / "resolved_config.yaml", "resolved config")[ROOT_KEY]
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
            env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT)))
            threads = str(int(runtime["cpu_threads_per_job"]))
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[name] = threads
            process = subprocess.Popen(
                build_lr_worker_command(
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
                        f"LR wave {wave} job {jobs[index]['job_id']} exited with code {code}"
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
    "lr_profile",
    "lr_profile_sha256",
    "initial_learning_rate",
    "scheduler_min_lr",
    "lr_trace",
    "lr_stage",
)


def _write_resource_summary(root: Path) -> Path:
    base_path = training._write_resource_summary(root)
    with base_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    jobs = {
        str(job["job_id"]): dict(job) for job in _load_registry(root)["jobs"]
    }
    for row in rows:
        job = jobs[str(row["job_id"])]
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        row.update(
            {
                "lr_profile": job["lr_profile"],
                "lr_profile_sha256": job["lr_profile_sha256"],
                "initial_learning_rate": job["initial_learning_rate"],
                "scheduler_min_lr": job["scheduler_min_lr"],
                "lr_trace": json.dumps(
                    status.get("lr_trace", job["lr_trace"]), separators=(",", ":")
                ),
                "lr_stage": job["lr_stage"],
            }
        )
    return _write_csv(root / "resource_summary.csv", rows, RESOURCE_FIELDS)


def _write_stage_state(root: Path, stage: str, stage_status: str, *, overall_status: str | None = None, **details: Any) -> None:
    path = root / "lr_stage_status.json"
    payload = _read_json(path)
    stages = dict(payload.get("stages") or {})
    entry = dict(stages.get(stage) or {})
    entry.update({"status": stage_status, "updated_at_utc": _utc_now(), **details})
    stages[stage] = entry
    payload.update(
        {
            "status": overall_status or payload.get("status", "running"),
            "current_stage": stage,
            "stages": stages,
            "updated_at_utc": _utc_now(),
        }
    )
    _write_json(path, payload)


def _run_registered_stage(root: Path, config_path: str | Path, stage: str, *, resume: bool, dry_run: bool) -> None:
    state = _read_json(root / "lr_stage_status.json")
    entry = _require_mapping(
        _require_mapping(state.get("stages"), "lr stages").get(stage),
        f"LR stage {stage}",
    )
    required = [str(value) for value in entry.get("required_job_ids", ())]
    new_ids = set(str(value) for value in entry.get("new_job_ids", ()))
    registry = _load_registry(root)
    by_id = {str(job["job_id"]): dict(job) for job in registry["jobs"]}
    missing = sorted(set(required) - set(by_id))
    if missing:
        raise ValueError(f"LR stage registry is missing jobs: {missing}")
    _write_stage_state(
        root,
        stage,
        "dry_running" if dry_run else "running",
        overall_status="dry_running" if dry_run else "running",
    )
    waves = sorted({int(by_id[job_id]["wave"]) for job_id in new_ids})
    for wave in waves:
        selected_ids = {job_id for job_id in new_ids if int(by_id[job_id]["wave"]) == wave}
        jobs = [
            job
            for job in _jobs_for_wave(root, wave, resume=resume, dry_run=dry_run)
            if str(job["job_id"]) in selected_ids
        ]
        _run_lr_wave(root, config_path, wave, jobs, dry_run=dry_run, resume=resume)
        _validate_wave_completion(root, jobs, dry_run=dry_run)
        _write_resource_summary(root)
        _refresh_registry_exports(root)
    failures = []
    for job_id in required:
        job = by_id[job_id]
        status = _read_json(_job_status_path(root, job_id))
        valid = status.get("status") == "dry_run_passed" if dry_run else _completed_job_is_valid(root, job, status)
        if not valid:
            failures.append(f"{job_id}={status.get('status', 'missing')}")
    if failures:
        raise RuntimeError(f"LR stage did not complete all required jobs: {failures}")
    _write_stage_state(
        root,
        stage,
        "dry_run_passed" if dry_run else "completed",
        overall_status="dry_run_passed" if dry_run else "running",
        completed_at_utc=_utc_now(),
    )


def _selected_lr_profile(selection: Mapping[str, Any]) -> str:
    value = str(
        selection.get(
            "selected_lr_profile",
            selection.get("winner_lr_profile", selection.get("lr_profile", "")),
        )
    ).strip().lower()
    if value not in LR_PROFILE_IDS:
        raise ValueError(f"LR selection has an invalid selected profile: {value!r}")
    expected_sha = _lr_profile_sha256(value)
    supplied_sha = str(selection.get("lr_profile_sha256", "")).strip()
    if supplied_sha and supplied_sha != expected_sha:
        raise ValueError("LR selection profile hash does not match the frozen profile")
    supplied_lr = selection.get("initial_learning_rate")
    if supplied_lr is not None and float(supplied_lr) != FROZEN_LEARNING_RATES[value]:
        raise ValueError("LR selection value does not match the selected profile")
    supplied_floor = selection.get("scheduler_min_lr")
    if supplied_floor is not None and float(supplied_floor) != FROZEN_SCHEDULER_MIN_LRS[value]:
        raise ValueError("LR selection scheduler floor does not match the selected profile")
    return value


def _validate_selection_result(selection: Mapping[str, Any], stage: str) -> None:
    if "gate_passed" not in selection:
        raise ValueError(f"{stage} selection is missing gate_passed")
    if selection.get("q4_used_for_selection") is not False:
        raise ValueError("LR selection must explicitly declare q4_used_for_selection=false")
    if str(selection.get("selection_panel", "")) != "common_validation_05m":
        raise ValueError("LR selection must use only common_validation_05m (Q3)")
    if bool(selection["gate_passed"]):
        _selected_lr_profile(selection)
    else:
        retained = {
            field: selection.get(field)
            for field in (
                "selected_lr_profile",
                "winner_lr_profile",
                "winner_profile",
                "lr_profile",
            )
            if str(selection.get(field, "") or "").strip()
        }
        selected_profiles = selection.get("selected_lr_profiles", [])
        if retained or (isinstance(selected_profiles, Sequence) and selected_profiles):
            raise ValueError(
                "A failed LR gate must not retain a selected/winner LR profile"
            )


def _freeze_selection(
    root: Path,
    stage: str,
    *,
    selection_hook: Callable[[Path, str], Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], str]:
    path = _selection_path(root, stage)
    if path.is_file():
        payload = _require_mapping(_read_json(path), f"LR selection {stage}")
        _validate_selection_result(payload, stage)
        state = _read_json(root / "lr_stage_status.json")
        stage_entry = _require_mapping(
            _require_mapping(state.get("stages"), "lr stages").get(stage),
            f"LR stage {stage}",
        )
        if str(stage_entry.get("selection_sha256", "")) != _sha256_file(path):
            raise ValueError(f"Frozen LR selection hash mismatch for {stage}")
        cumulative_path = root / "lr_selection.json"
        if not cumulative_path.is_file():
            raise ValueError(f"Cumulative LR selection is missing for {stage}")
        cumulative = _require_mapping(
            _read_json(cumulative_path), "cumulative LR selection"
        )
        cumulative_stages = _require_mapping(
            cumulative.get("stages"), "cumulative LR selection stages"
        )
        expected_result = {key: value for key, value in payload.items() if key != "stage"}
        observed_result = _require_mapping(
            cumulative_stages.get(stage), f"cumulative LR selection stage {stage}"
        )
        if training._canonical_json(expected_result) != training._canonical_json(
            observed_result
        ):
            raise ValueError(
                f"Cumulative LR selection disagrees with frozen snapshot: {stage}"
            )
        return payload, _sha256_file(path)
    if selection_hook is None:
        from scripts.rq3.news_first_vol_lr_analysis import evaluate_lr_stage

        selection_hook = evaluate_lr_stage
    result = _require_mapping(selection_hook(root, stage), f"LR selection result {stage}")
    _validate_selection_result(result, stage)
    payload = {"schema_version": 1, "stage": stage, **result}
    _write_json(path, payload)
    sha = _sha256_file(path)
    _write_stage_state(root, stage, "selected", selection_path=str(path), selection_sha256=sha)
    cumulative_path = root / "lr_selection.json"
    cumulative = (
        _read_json(cumulative_path)
        if cumulative_path.is_file()
        else {"schema_version": 1, "selection_scope": "Q3_only", "stages": {}}
    )
    stages = dict(cumulative.get("stages") or {})
    stages[stage] = {key: value for key, value in payload.items() if key != "stage"}
    cumulative.update({"stages": stages, "updated_at_utc": _utc_now()})
    _write_json(cumulative_path, cumulative)
    return payload, sha


def _mark_terminal(root: Path, stage: str, status: str) -> None:
    _write_stage_state(
        root,
        stage,
        status,
        overall_status=status,
        completed_at_utc=_utc_now(),
    )
    _write_json(
        root / "registry" / "experiment_status.json",
        {
            "status": status,
            "current_stage": stage,
            "q4_accessed": False,
            "completed_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
        },
    )


def _render_terminal_report(
    root: Path,
    config_path: str | Path,
    *,
    stage: str,
    status: str,
) -> Path:
    """Finalize a Q3-only terminal stage and hash its portable report."""

    _mark_terminal(root, stage, status)
    _write_resource_summary(root)
    refresh_lr_sweep_lineage(config_path, root)
    _refresh_registry_exports(root)
    from scripts.rq3.news_first_vol_lr_report import render_lr_sweep_report

    report = Path(render_lr_sweep_report(root))
    refresh_lr_sweep_lineage(config_path, root)
    _refresh_registry_exports(root)
    return report


def launch_lr_sweep_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
    selection_hook: Callable[[Path, str], Mapping[str, Any]] | None = None,
) -> Path:
    root = prepare_lr_sweep_experiment(config_path, output_dir, reuse=True)
    previous = _read_json(root / "registry" / "experiment_status.json")
    if previous.get("status") in TERMINAL_EXPERIMENT_STATES and not dry_run:
        if not resume:
            raise RuntimeError("Terminal LR sweep requires --resume for idempotent reuse")
        if previous.get("status") != "dry_run_passed":
            _validate_registry(root)
            for stage in LR_STAGES:
                path = _selection_path(root, stage)
                if path.is_file():
                    _freeze_selection(root, stage, selection_hook=selection_hook)
            report_path = root / "report" / "lr_sweep_report.html"
            if not report_path.is_file():
                raise ValueError("Terminal LR sweep is missing its Q3-only report")
            refresh_lr_sweep_lineage(config_path, root)
            _refresh_registry_exports(root)
            return root
    if previous.get("status") in {"failed", "running"} and not (resume or dry_run):
        raise RuntimeError("Interrupted LR sweep requires --resume")
    _write_json(
        root / "registry" / "experiment_status.json",
        {
            "status": "dry_running" if dry_run else "running",
            "current_stage": "lr_screen",
            "q4_accessed": False,
            "updated_at_utc": _utc_now(),
        },
    )
    try:
        _run_registered_stage(root, config_path, "lr_screen", resume=resume, dry_run=dry_run)
        if dry_run:
            _mark_terminal(root, "lr_screen", "dry_run_passed")
            _write_resource_summary(root)
            refresh_lr_sweep_lineage(config_path, root)
            _refresh_registry_exports(root)
            return root
        selection, selection_sha = _freeze_selection(
            root, "lr_screen", selection_hook=selection_hook
        )
        if not bool(selection["gate_passed"]):
            _render_terminal_report(
                root,
                config_path,
                stage="lr_screen",
                status="completed_no_learned_lr",
            )
            return root
        selected = _selected_lr_profile(selection)
        resolved = _load_yaml_mapping(root / "resolved_config.yaml", "resolved config")[ROOT_KEY]
        _append_lr_stage_jobs(
            root,
            resolved,
            stage="lr_confirm",
            profiles=(selected,),
            selection_sha256=selection_sha,
        )
        _run_registered_stage(root, config_path, "lr_confirm", resume=resume, dry_run=False)
        confirm, _ = _freeze_selection(
            root, "lr_confirm", selection_hook=selection_hook
        )
        if bool(confirm["gate_passed"]) and _selected_lr_profile(confirm) != selected:
            raise ValueError("LR confirm cannot switch the Q3-screen-selected profile")
        terminal_status = (
            "completed_q3_only"
            if bool(confirm["gate_passed"])
            else "completed_no_learned_lr"
        )
        _render_terminal_report(
            root,
            config_path,
            stage="lr_confirm",
            status=terminal_status,
        )
    except BaseException as exc:
        _write_json(
            root / "registry" / "experiment_status.json",
            {
                "status": "failed",
                "q4_accessed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at_utc": _utc_now(),
            },
        )
        state = _read_json(root / "lr_stage_status.json")
        state.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at_utc": _utc_now(),
            }
        )
        _write_json(root / "lr_stage_status.json", state)
        refresh_lr_sweep_lineage(config_path, root)
        _refresh_registry_exports(root)
        raise
    _write_resource_summary(root)
    refresh_lr_sweep_lineage(config_path, root)
    _refresh_registry_exports(root)
    return root


def run_news_first_vol_lr_sweep(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    action: str = "launch",
    job_id: str = "",
    resume: bool = False,
    reuse: bool = False,
    worker_dry_run: bool = False,
    selection_hook: Callable[[Path, str], Mapping[str, Any]] | None = None,
) -> Path:
    action = str(action).strip().lower()
    if action == "prepare":
        return prepare_lr_sweep_experiment(
            config_path, output_dir, reuse=bool(reuse or resume)
        )
    if action == "dry-run":
        return launch_lr_sweep_experiment(
            config_path, output_dir, resume=resume, dry_run=True
        )
    if action == "worker":
        if not job_id:
            raise ValueError("--job-id is required for the worker action")
        return run_lr_sweep_worker(
            output_dir, job_id, dry_run=worker_dry_run, resume=resume
        )
    if action == "launch":
        return launch_lr_sweep_experiment(
            config_path,
            output_dir,
            resume=resume,
            dry_run=False,
            selection_hook=selection_hook,
        )
    raise ValueError(f"Unsupported LR-sweep action: {action}")


__all__ = [
    "FROZEN_LEARNING_RATES",
    "FROZEN_SCHEDULER_MIN_LRS",
    "LR_PROFILE_IDS",
    "build_lr_worker_command",
    "launch_lr_sweep_experiment",
    "prepare_lr_sweep_experiment",
    "refresh_lr_sweep_lineage",
    "run_lr_sweep_worker",
    "run_news_first_vol_lr_sweep",
]
