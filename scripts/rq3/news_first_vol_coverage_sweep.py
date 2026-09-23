"""Fail-closed four-stage completion sweep for News-first Vol experiments.

The sweep adds only cells that are absent from the two completed Q3 reference
experiments.  It is deliberately sequential: a stage cannot start until the
preceding stage has a hash-verified QA record.  The immutable training unit is
``(stage, model family, capacity, LR, seed, tolerance, text mode)``.

This module owns a small ``python -m`` worker CLI so that the branch-local
orchestrator can be prepared and tested without changing ``scripts/rq3/main.py``.
"""

from __future__ import annotations

import argparse
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
EXPERIMENT_KIND = "coverage_completion_sweep"
FROZEN_PROFILES = training.CAPACITY_PROFILE_NAMES
FROZEN_SEEDS = (42, 202, 404)
FROZEN_LR_PROFILES: dict[str, tuple[float, float]] = {
    "lr_5e_07": (5.0e-7, 5.0e-8),
    "lr_7_5e_07": (7.5e-7, 7.5e-8),
    "lr_1e_06": (1.0e-6, 1.0e-7),
    "lr_1_5e_06": (1.5e-6, 1.5e-7),
    "lr_2e_06": (2.0e-6, 2.0e-7),
}
STAGE_IDS = (
    "stage_1_regression_tolerance_completion",
    "stage_2_regression_text_shuffle_completion",
    "stage_3_wgan_low_lr_capacity",
    "stage_4_regression_capacity_lr_interaction",
)
STAGE_CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "stage_id": STAGE_IDS[0],
        "model_family": "regression",
        "capacity_profiles": FROZEN_PROFILES,
        "lr_profiles": ("lr_5e_07",),
        "seeds": FROZEN_SEEDS,
        "tolerances_minutes": (10, 15),
        "text_ablation_modes": ("current_only", REAL_TEXT),
        "expected_jobs": 72,
    },
    {
        "stage_id": STAGE_IDS[1],
        "model_family": "regression",
        "capacity_profiles": FROZEN_PROFILES,
        "lr_profiles": ("lr_5e_07",),
        "seeds": FROZEN_SEEDS,
        "tolerances_minutes": (5, 10, 15, 30),
        "text_ablation_modes": ("text_shuffle",),
        "expected_jobs": 72,
    },
    {
        "stage_id": STAGE_IDS[2],
        "model_family": "wgan",
        "capacity_profiles": FROZEN_PROFILES,
        "lr_profiles": ("lr_5e_07",),
        "seeds": FROZEN_SEEDS,
        "tolerances_minutes": (5, 30),
        "text_ablation_modes": ("current_only", REAL_TEXT),
        "expected_jobs": 72,
    },
    {
        "stage_id": STAGE_IDS[3],
        "model_family": "regression",
        "capacity_profiles": ("micro", "tiny", "small", "medium", "legacy"),
        "lr_profiles": ("lr_7_5e_07", "lr_1e_06", "lr_1_5e_06", "lr_2e_06"),
        "seeds": FROZEN_SEEDS,
        "tolerances_minutes": (5, 30),
        "text_ablation_modes": ("current_only", REAL_TEXT),
        "expected_jobs": 240,
    },
)
EXPECTED_STAGE_JOBS = {
    row["stage_id"]: int(row["expected_jobs"]) for row in STAGE_CONTRACTS
}
EXPECTED_TOTAL_JOBS = sum(EXPECTED_STAGE_JOBS.values())
ALLOWED_SLOTS_PER_GPU = (2, 4, 6, 8, 12, 16, 24)
TERMINAL_SUCCESS = {"completed", "dry_run_passed"}
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
        config.get("coverage_completion_sweep"), "coverage_completion_sweep"
    )
    if not bool(sweep.get("enabled", False)):
        raise ValueError("coverage_completion_sweep.enabled must be true")
    return sweep


def _profiles(config: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    raw = _require_mapping(_sweep_config(config).get("profiles"), "profiles")
    names = tuple(str(name).strip().lower() for name in raw)
    if names != FROZEN_PROFILES:
        raise ValueError(f"profiles must preserve frozen order {list(FROZEN_PROFILES)}")
    fields = set(
        training.CAPACITY_PROFILE_SHAPE_FIELDS
        + training.CAPACITY_PROFILE_METADATA_FIELDS
    )
    result: dict[str, dict[str, int]] = {}
    for raw_name, raw_values in raw.items():
        name = str(raw_name).strip().lower()
        values = _require_mapping(raw_values, f"profiles.{name}")
        if set(values) != fields:
            raise ValueError(
                f"profile {name} fields differ: missing={sorted(fields - set(values))}, "
                f"extra={sorted(set(values) - fields)}"
            )
        observed = {field: int(values[field]) for field in fields}
        expected = training.FROZEN_CAPACITY_PROFILES[name]
        if observed != expected:
            raise ValueError(
                f"profile {name} differs from frozen shape: "
                f"expected={expected}, observed={observed}"
            )
        result[name] = observed
    return result


def _lr_profiles(config: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    raw = _require_mapping(
        _sweep_config(config).get("learning_rate_profiles"),
        "learning_rate_profiles",
    )
    if tuple(raw) != tuple(FROZEN_LR_PROFILES):
        raise ValueError(
            "learning_rate_profiles must preserve the frozen names and order"
        )
    result: dict[str, dict[str, float]] = {}
    for name, (expected_lr, expected_floor) in FROZEN_LR_PROFILES.items():
        values = _require_mapping(raw[name], f"learning_rate_profiles.{name}")
        if set(values) != {"initial_learning_rate", "scheduler_min_lr"}:
            raise ValueError(f"Unexpected LR profile fields for {name}")
        _exact_float(values["initial_learning_rate"], expected_lr, f"{name} LR")
        _exact_float(values["scheduler_min_lr"], expected_floor, f"{name} floor")
        result[name] = {
            "initial_learning_rate": expected_lr,
            "scheduler_min_lr": expected_floor,
        }
    return result


def _stages(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_stages = _sweep_config(config).get("stages")
    if not isinstance(raw_stages, list):
        raise TypeError("coverage_completion_sweep.stages must be a list")
    if len(raw_stages) != len(STAGE_CONTRACTS):
        raise ValueError("Exactly four frozen stages are required")
    normalized: list[dict[str, Any]] = []
    for index, (raw, expected) in enumerate(zip(raw_stages, STAGE_CONTRACTS), 1):
        stage = _require_mapping(raw, f"stages[{index - 1}]")
        observed = {
            "stage_id": str(stage.get("stage_id", "")),
            "model_family": str(stage.get("model_family", "")).lower(),
            "capacity_profiles": tuple(
                str(value).lower() for value in stage.get("capacity_profiles", ())
            ),
            "lr_profiles": tuple(str(value) for value in stage.get("lr_profiles", ())),
            "seeds": tuple(int(value) for value in stage.get("seeds", ())),
            "tolerances_minutes": tuple(
                int(value) for value in stage.get("tolerances_minutes", ())
            ),
            "text_ablation_modes": tuple(
                normalize_text_ablation_mode(value)
                for value in stage.get("text_ablation_modes", ())
            ),
            "expected_jobs": int(stage.get("expected_jobs", -1)),
        }
        for field, value in expected.items():
            if observed[field] != value:
                raise ValueError(
                    f"Stage {index} {field} differs: expected={value}, "
                    f"observed={observed[field]}"
                )
        product = (
            len(observed["capacity_profiles"])
            * len(observed["lr_profiles"])
            * len(observed["seeds"])
            * len(observed["tolerances_minutes"])
            * len(observed["text_ablation_modes"])
        )
        if product != observed["expected_jobs"]:
            raise ValueError(f"Stage {index} Cartesian product is {product}")
        normalized.append(observed)
    return normalized


def _validate_model_training(
    family: str, model: Mapping[str, Any], *, expected_patience: int
) -> None:
    expected_command = "vol-xlsx" if family == "wgan" else "vol-regression-xlsx"
    if str(model.get("trainer_command", "")) != expected_command:
        raise ValueError(f"{family} trainer_command must be {expected_command}")
    values = _require_mapping(model.get("training"), f"models.{family}.training")
    frozen_ints = {
        "channels": 1,
        "embedding_dim": 1024,
        "noise_dim": 32,
        "gen_res_blocks": 0,
        "disc_res_blocks": 0,
        "num_epochs": 100,
        "early_stopping_min_epochs": 30,
        "early_stopping_patience": expected_patience,
        "reduce_lr_patience": 3,
        "batch_size": 16,
        "constraint_warmup_epochs": 0,
    }
    for field, expected in frozen_ints.items():
        if int(values.get(field, -1)) != expected:
            raise ValueError(f"{family} {field} is frozen to {expected}")
    if family == "wgan" and int(values.get("discriminator_iter", -1)) != 5:
        raise ValueError("wgan discriminator_iter is frozen to 5")
    if str(values.get("residual_output_mode", "")) != "identity_softplus_residual":
        raise ValueError(f"{family} requires identity_softplus_residual")
    if str(values.get("best_checkpoint_metric", "")) != "val_hybrid_score":
        raise ValueError(f"{family} best checkpoint metric is frozen")
    for field in (
        "evaluate_initial_checkpoint",
        "use_reduce_lr_on_plateau",
        "use_early_stopping",
        "use_calendar_constraint",
        "use_butterfly_constraint",
        "use_smooth_constraint",
    ):
        if not bool(values.get(field, False)):
            raise ValueError(f"{family} {field} must remain true")
    _exact_float(values.get("learning_rate"), 5.0e-7, f"{family} base LR")
    _exact_float(values.get("reduce_lr_min_lr"), 5.0e-8, f"{family} base floor")
    frozen_floats = {
        "train_ratio": 0.8,
        "reduce_lr_factor": 0.5,
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
    if family == "wgan":
        frozen_floats["lambda_gp"] = 10.0
    for field, expected in frozen_floats.items():
        _exact_float(values.get(field), expected, f"{family}.{field}")
    if int(values.get("save_every", 0)) <= 100:
        raise ValueError(f"{family} save_every must preserve best/final only")


def _validate_frozen_config(config: Mapping[str, Any]) -> None:
    datasets = _require_mapping(config.get("datasets"), "datasets")
    split = _require_mapping(config.get("split"), "split")
    runtime = _require_mapping(config.get("runtime"), "runtime")
    models = _require_mapping(config.get("models"), "models")
    sweep = _sweep_config(config)

    if tuple(int(value) for value in datasets.get("tolerances_minutes", ())) != (
        5,
        10,
        15,
        30,
    ):
        raise ValueError("Dataset tolerances are frozen to 5/10/15/30")
    if int(datasets.get("common_evaluation_tolerance_minutes", -1)) != 5:
        raise ValueError("Common evaluation tolerance is frozen to 5m")
    if str(datasets.get("sheet_name", "")) != "gan_input_ready":
        raise ValueError("sheet_name is frozen to gan_input_ready")
    if str(datasets.get("text_embedding_mode", "")).lower() != "lp":
        raise ValueError("text embedding is frozen to LP")
    if training._support_mask_mode(datasets) != "raw_joint":
        raise ValueError("support mask is frozen to raw_joint")
    if tuple(training._configured_text_ablation_modes(datasets)) != (
        "current_only",
        REAL_TEXT,
        "text_shuffle",
    ):
        raise ValueError("Dataset text modes must cover current/real/shuffle")
    if int(datasets.get("text_shuffle_seed", -1)) != 42:
        raise ValueError("text shuffle seed is frozen to 42")
    if str(split.get("train_end_utc")) != "2023-07-01T00:00:00Z":
        raise ValueError("train_end_utc is frozen")
    if str(split.get("validation_end_utc")) != "2023-10-01T00:00:00Z":
        raise ValueError("validation_end_utc is frozen")
    if int(split.get("validation_mc_samples", -1)) != 16:
        raise ValueError("validation_mc_samples is frozen to 16")
    if not bool(sweep.get("q4_prediction_and_evaluation_forbidden", False)):
        raise ValueError("Q4 prediction/evaluation must remain forbidden")
    _profiles(config)
    _lr_profiles(config)
    _stages(config)

    if set(models) != {"regression", "wgan"}:
        raise ValueError("Exactly regression and wgan model configs are required")
    _validate_model_training("regression", models["regression"], expected_patience=12)
    _validate_model_training("wgan", models["wgan"], expected_patience=16)

    gpu_ids = tuple(int(value) for value in runtime.get("gpu_ids", ()))
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("Exactly two distinct GPUs are required")
    slots = int(runtime.get("slots_per_gpu", -1))
    if slots not in ALLOWED_SLOTS_PER_GPU:
        raise ValueError(f"slots_per_gpu must be one of {ALLOWED_SLOTS_PER_GPU}")
    if int(runtime.get("cpu_threads_per_job", -1)) <= 0:
        raise ValueError("cpu_threads_per_job must be positive")
    stage_slots = dict(runtime.get("stage_slots_per_gpu") or {})
    unknown_stage_slots = set(stage_slots) - set(STAGE_IDS)
    if unknown_stage_slots:
        raise ValueError(
            f"stage_slots_per_gpu contains unknown stages: {sorted(unknown_stage_slots)}"
        )
    for stage_id, value in stage_slots.items():
        if int(value) not in ALLOWED_SLOTS_PER_GPU:
            raise ValueError(
                f"stage_slots_per_gpu.{stage_id} must be one of {ALLOWED_SLOTS_PER_GPU}"
            )
    policy = str(runtime.get("numa_memory_policy", "preferred")).strip().lower()
    if policy not in {"preferred", "strict", "none"}:
        raise ValueError("numa_memory_policy must be preferred/strict/none")
    numa = {
        int(key): int(value)
        for key, value in dict(runtime.get("gpu_numa_nodes") or {}).items()
    }
    if set(numa) != set(gpu_ids):
        raise ValueError("gpu_numa_nodes must cover exactly the configured GPUs")

    references = _require_mapping(
        sweep.get("reference_experiments"), "reference_experiments"
    )
    if set(references) != {"fixed_lr_capacity", "local_lr"}:
        raise ValueError("Both fixed_lr_capacity and local_lr references are required")
    for name, raw in references.items():
        reference = _require_mapping(raw, f"reference_experiments.{name}")
        if set(reference) != {
            "root",
            "pair_metrics",
            "registry",
            "resolved_config_hash",
        }:
            raise ValueError(f"Unexpected reference fields for {name}")


def _resolved_config(config_path: str | Path) -> dict[str, Any]:
    path = _resolve_repo_path(config_path)
    root = _load_yaml_mapping(path, "coverage completion config")
    config = _require_mapping(root.get(ROOT_KEY), ROOT_KEY)
    _validate_frozen_config(config)
    resolved = deepcopy(config)
    resolved["datasets"]["root"] = str(_resolve_repo_path(resolved["datasets"]["root"]))
    resolved["datasets"]["support_mask_mode"] = "raw_joint"
    resolved["datasets"]["text_ablation_modes"] = [
        "current_only",
        REAL_TEXT,
        "text_shuffle",
    ]
    resolved["runtime"]["python_executable"] = str(
        _resolve_repo_path(resolved["runtime"].get("python_executable", sys.executable))
    )
    references = resolved["coverage_completion_sweep"]["reference_experiments"]
    for reference in references.values():
        reference["root"] = str(_resolve_repo_path(reference["root"]))
        for field in ("pair_metrics", "registry", "resolved_config_hash"):
            reference[field] = str(_resolve_repo_path(reference[field]))
    resolved["source_config_path"] = str(path)
    return resolved


def _capacity_profile_sha256(profile: str) -> str:
    name = str(profile).strip().lower()
    if name not in FROZEN_PROFILES:
        raise ValueError(f"Unknown capacity profile: {profile}")
    return training._capacity_profile_sha256(
        name, training.FROZEN_CAPACITY_PROFILES[name]
    )


def _lr_profile_sha256(profile: str) -> str:
    name = str(profile).strip()
    if name not in FROZEN_LR_PROFILES:
        raise ValueError(f"Unknown LR profile: {profile}")
    learning_rate, scheduler_min_lr = FROZEN_LR_PROFILES[name]
    return _payload_sha256(
        {
            "schema_version": 1,
            "lr_profile": name,
            "initial_learning_rate": learning_rate,
            "scheduler_min_lr": scheduler_min_lr,
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": 3,
        }
    )


def _capacity_seed_profile_sha256(
    profile: str, lr_profile: str, seed: int, family: str
) -> str:
    if int(seed) not in FROZEN_SEEDS:
        raise ValueError(f"Unknown seed: {seed}")
    if str(family) not in {"regression", "wgan"}:
        raise ValueError(f"Unknown model family: {family}")
    return _payload_sha256(
        {
            "schema_version": 1,
            "capacity_profile_sha256": _capacity_profile_sha256(profile),
            "lr_profile_sha256": _lr_profile_sha256(lr_profile),
            "seed": int(seed),
            "model_family": str(family),
        }
    )


def _coverage_cell_sha256(spec: Mapping[str, Any]) -> str:
    return _payload_sha256(
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "stage_index": int(spec["stage_index"]),
            "stage_id": str(spec["stage_id"]),
            "model_family": str(spec["model_family"]),
            "capacity_profile_sha256": _capacity_profile_sha256(
                str(spec["capacity_profile"])
            ),
            "lr_profile_sha256": _lr_profile_sha256(str(spec["lr_profile"])),
            "seed": int(spec["seed"]),
            "tolerance_minutes": int(spec["tolerance_minutes"]),
            "text_ablation_mode": normalize_text_ablation_mode(
                spec["text_ablation_mode"]
            ),
            "support_mask_mode": "raw_joint",
            "residual_output_mode": "identity_softplus_residual",
        }
    )


def _job_id(spec: Mapping[str, Any]) -> str:
    return (
        f"s{int(spec['stage_index'])}_{spec['model_family']}_"
        f"{spec['capacity_profile']}_{spec['lr_profile']}_"
        f"seed_{int(spec['seed']):03d}_"
        f"{normalize_text_ablation_mode(spec['text_ablation_mode'])}_"
        f"{int(spec['tolerance_minutes']):02d}m"
    )


def _cell_order(stage: Mapping[str, Any]) -> list[tuple[str, int]]:
    modes = tuple(stage["text_ablation_modes"])
    tolerances = tuple(int(value) for value in stage["tolerances_minutes"])
    if len(modes) == 2 and len(tolerances) == 2:
        return [
            (modes[0], tolerances[0]),
            (modes[1], tolerances[1]),
            (modes[1], tolerances[0]),
            (modes[0], tolerances[1]),
        ]
    if len(modes) == 1 and set(tolerances) == {5, 10, 15, 30}:
        return [(modes[0], tolerance) for tolerance in (5, 30, 10, 15)]
    return [(mode, tolerance) for mode in modes for tolerance in tolerances]


def _stage_slots_per_gpu(resolved: Mapping[str, Any], stage_id: str) -> int:
    runtime = _require_mapping(resolved["runtime"], "runtime")
    overrides = dict(runtime.get("stage_slots_per_gpu") or {})
    value = int(overrides.get(stage_id, runtime["slots_per_gpu"]))
    if value not in ALLOWED_SLOTS_PER_GPU:
        raise ValueError(f"Invalid slots_per_gpu for {stage_id}: {value}")
    return value


def _worker_slots(
    resolved: Mapping[str, Any], stage_id: str
) -> tuple[tuple[int, int, int], ...]:
    runtime = _require_mapping(resolved["runtime"], "runtime")
    gpu_ids = tuple(int(value) for value in runtime["gpu_ids"])
    slots_per_gpu = _stage_slots_per_gpu(resolved, stage_id)
    if slots_per_gpu not in ALLOWED_SLOTS_PER_GPU:
        raise ValueError(f"slots_per_gpu must be one of {ALLOWED_SLOTS_PER_GPU}")
    numa = {
        int(key): int(value) for key, value in dict(runtime["gpu_numa_nodes"]).items()
    }
    # Interleave physical GPUs so every four-cell block is split evenly.
    return tuple(
        (gpu_id, gpu_slot, numa[gpu_id])
        for gpu_slot in range(slots_per_gpu)
        for gpu_id in gpu_ids
    )


def _balanced_group_order(stage: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    """Return every capacity/LR/seed group once in a Latin-style order."""

    profiles = tuple(str(value) for value in stage["capacity_profiles"])
    lr_profiles = tuple(str(value) for value in stage["lr_profiles"])
    seeds = tuple(int(value) for value in stage["seeds"])
    result: list[tuple[str, str, int]] = []
    if len(lr_profiles) == 1:
        # Every six-group round contains all capacities and exactly two of each
        # seed.  Thus the 12/6 group waves produced by 24 slots/GPU are balanced
        # exactly, while smaller WGAN benchmark waves remain interleaved.
        lr_profile = lr_profiles[0]
        for round_index in range(len(seeds)):
            for profile_index, profile in enumerate(profiles):
                seed = seeds[(round_index + profile_index) % len(seeds)]
                result.append((profile, lr_profile, seed))
    elif len(profiles) == 5 and len(lr_profiles) == 4 and len(seeds) == 3:
        # Stage 4 has 60 groups and, at 24 slots/GPU, exactly five 12-group
        # waves. Each wave contains 3 groups/LR, 4 groups/seed, and 2-3 groups
        # per capacity; cycling the wave index covers every Cartesian cell once.
        for wave_index in range(len(profiles)):
            for lr_index, lr_profile in enumerate(lr_profiles):
                for seed_index, seed in enumerate(seeds):
                    profile = profiles[
                        (wave_index + lr_index + seed_index) % len(profiles)
                    ]
                    result.append((profile, lr_profile, seed))
    else:
        result = [
            (profile, lr_profile, seed)
            for profile in profiles
            for lr_profile in lr_profiles
            for seed in seeds
        ]
    expected = {
        (profile, lr_profile, seed)
        for profile in profiles
        for lr_profile in lr_profiles
        for seed in seeds
    }
    if len(result) != len(expected) or set(result) != expected:
        raise AssertionError("Balanced group order does not cover the Cartesian grid")
    return result


def _job_specs(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    global_wave_offset = 0
    for stage_index, stage in enumerate(_stages(resolved), 1):
        slots = _worker_slots(resolved, str(stage["stage_id"]))
        if len(slots) % 4:
            raise ValueError(
                "Each GPU worker wave must be divisible into four-cell blocks"
            )
        groups = _balanced_group_order(stage)
        groups_per_wave = len(slots) // 4
        group_waves = [
            groups[index : index + groups_per_wave]
            for index in range(0, len(groups), groups_per_wave)
        ]
        stage_specs: list[dict[str, Any]] = []
        for stage_wave, wave_groups in enumerate(group_waves, 1):
            for profile, lr_profile, seed in wave_groups:
                for mode, tolerance in _cell_order(stage):
                    stage_specs.append(
                        {
                            "stage_index": stage_index,
                            "stage_id": stage["stage_id"],
                            "model_family": stage["model_family"],
                            "capacity_profile": profile,
                            "lr_profile": lr_profile,
                            "seed": int(seed),
                            "text_ablation_mode": mode,
                            "tolerance_minutes": int(tolerance),
                            "stage_wave": stage_wave,
                        }
                    )
        if len(stage_specs) != int(stage["expected_jobs"]):
            raise AssertionError(f"Unexpected generated count for {stage['stage_id']}")
        wave_slot_indexes: dict[int, int] = {}
        for stage_job_index, spec in enumerate(stage_specs, 1):
            stage_wave = int(spec["stage_wave"])
            slot_index = wave_slot_indexes.get(stage_wave, 0)
            wave_slot_indexes[stage_wave] = slot_index + 1
            gpu_id, gpu_slot, numa_node = slots[slot_index]
            result.append(
                {
                    **spec,
                    "stage_job_index": stage_job_index,
                    "stage_wave": stage_wave,
                    "global_wave": global_wave_offset + stage_wave,
                    "wave": global_wave_offset + stage_wave,
                    "gpu_id": gpu_id,
                    "gpu_slot": gpu_slot,
                    "numa_node": numa_node,
                }
            )
        global_wave_offset += len(group_waves)
    if len(result) != EXPECTED_TOTAL_JOBS:
        raise AssertionError(f"Expected {EXPECTED_TOTAL_JOBS} jobs, got {len(result)}")
    return result


def _training_payload(
    resolved: Mapping[str, Any], root: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    family = str(spec["model_family"])
    profile = str(spec["capacity_profile"])
    lr_profile = str(spec["lr_profile"])
    seed = int(spec["seed"])
    tolerance = int(spec["tolerance_minutes"])
    mode = normalize_text_ablation_mode(spec["text_ablation_mode"])
    payload = training._training_payload(
        resolved,
        family=family,
        tolerance=tolerance,
        text_ablation_mode=mode,
        multi_mode=True,
        experiment_root=root,
        capacity_profile=None,
    )
    shape = _profiles(resolved)[profile]
    payload.update(
        {field: int(shape[field]) for field in training.CAPACITY_PROFILE_SHAPE_FIELDS}
    )
    learning_rate, scheduler_min_lr = FROZEN_LR_PROFILES[lr_profile]
    capacity_seed_sha = _capacity_seed_profile_sha256(profile, lr_profile, seed, family)
    output_root = (
        root
        / "runs"
        / str(spec["stage_id"])
        / family
        / profile
        / lr_profile
        / f"seed_{seed:03d}"
        / mode
        / f"tolerance_{tolerance:02d}m"
    )
    payload.update(
        {
            "seed": seed,
            "news_first_capacity_profile": profile,
            "news_first_capacity_profile_sha256": _capacity_profile_sha256(profile),
            "news_first_capacity_seed_profile_sha256": capacity_seed_sha,
            "news_first_lr_profile": lr_profile,
            "news_first_lr_profile_sha256": _lr_profile_sha256(lr_profile),
            "news_first_fixed_learning_rate_profile": lr_profile,
            "news_first_fixed_learning_rate_profile_sha256": _lr_profile_sha256(
                lr_profile
            ),
            "learning_rate": learning_rate,
            "use_reduce_lr_on_plateau": True,
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": 3,
            "reduce_lr_min_lr": scheduler_min_lr,
            "num_epochs": 100,
            "use_early_stopping": True,
            "early_stopping_min_epochs": 30,
            "early_stopping_patience": 16 if family == "wgan" else 12,
            "output_root": str(output_root),
        }
    )
    return payload


def _reference_roots(resolved: Mapping[str, Any]) -> dict[str, Any]:
    references = resolved["coverage_completion_sweep"]["reference_experiments"]
    result: dict[str, Any] = {}
    for name, reference in references.items():
        files: dict[str, Any] = {}
        for role in ("pair_metrics", "registry", "resolved_config_hash"):
            path = Path(str(reference[role]))
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name} reference {role}: {path}")
            files[role] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        payload = {"root": str(Path(str(reference["root"]))), "files": files}
        result[name] = {**payload, "reference_sha256": _payload_sha256(payload)}
    return result


def _source_rows(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = training._source_rows(resolved)
    for name, reference in _reference_roots(resolved).items():
        for role, file_row in reference["files"].items():
            rows.append(
                {
                    "source_role": f"reference_{name}_{role}",
                    "path": file_row["path"],
                    "size_bytes": file_row["size_bytes"],
                    "sha256": file_row["sha256"],
                }
            )
    return rows


def _code_rows() -> list[dict[str, Any]]:
    rows = training._code_rows()
    observed = {str(row["relative_path"]) for row in rows}
    for relative in (
        "scripts/rq3/news_first_vol_coverage_sweep.py",
        "scripts/rq3/news_first_vol_coverage_sweep_analysis.py",
        "scripts/rq3/news_first_vol_coverage_sweep_report.py",
    ):
        if relative in observed:
            continue
        path = REPO_ROOT / relative
        rows.append(
            {
                "relative_path": relative,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
        observed.add(relative)
    return rows


def _job_spec_sha256(job: Mapping[str, Any]) -> str:
    return _payload_sha256(
        {key: value for key, value in job.items() if key != "job_spec_sha256"}
    )


def _load_registry(root: Path) -> dict[str, Any]:
    registry = _require_mapping(
        _read_json(root / "registry" / "jobs.json"), "coverage registry"
    )
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment root is not a coverage completion sweep")
    return registry


def _read_manifest_rows(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row[key]): row for row in rows}


def _validate_root_lineage(
    root: Path, config_path: str | Path | None = None
) -> dict[str, Any]:
    resolved = _require_mapping(
        _load_yaml_mapping(root / "resolved_config.yaml", "resolved config").get(
            ROOT_KEY
        ),
        ROOT_KEY,
    )
    expected_sha = (
        (root / "registry" / "resolved_config.sha256")
        .read_text(encoding="utf-8")
        .strip()
    )
    if _payload_sha256(resolved) != expected_sha:
        raise ValueError("Resolved coverage config hash mismatch")
    if config_path is not None:
        if _payload_sha256(_resolved_config(config_path)) != expected_sha:
            raise ValueError("Prepared experiment config differs from requested config")
    registry = _load_registry(root)
    if str(registry.get("resolved_config_sha256", "")) != expected_sha:
        raise ValueError("Registry resolved-config hash mismatch")

    source_path = root / "source_hashes.csv"
    source_rows = _read_manifest_rows(source_path, "source_role")
    for role, row in source_rows.items():
        path = Path(row["path"])
        if not path.is_file() or _sha256_file(path) != row["sha256"]:
            raise ValueError(f"Source lineage changed: {role}")
    current_references = _reference_roots(resolved)
    if current_references != registry.get("reference_roots"):
        raise ValueError("Reference experiment lineage changed")

    prepared_code = _read_manifest_rows(root / "code_hashes.csv", "relative_path")
    current_code = {str(row["relative_path"]): row for row in _code_rows()}
    if set(prepared_code) != set(current_code):
        raise ValueError("Coverage code manifest membership changed")
    changed = [
        name
        for name, row in prepared_code.items()
        if row["sha256"] != str(current_code[name]["sha256"])
    ]
    if changed:
        raise ValueError(f"Coverage code changed after preparation: {changed}")
    return resolved


def _validate_job_lineage(
    root: Path,
    job: Mapping[str, Any],
    resolved: Mapping[str, Any],
    source_rows: Mapping[str, Mapping[str, str]],
) -> None:
    if str(job.get("job_spec_sha256", "")) != _job_spec_sha256(job):
        raise ValueError(f"job-spec hash mismatch for {job.get('job_id')}")
    stage_id = str(job.get("stage_id", ""))
    if stage_id not in STAGE_IDS:
        raise ValueError(f"Unknown stage for {job.get('job_id')}")
    spec = {
        field: job[field]
        for field in (
            "stage_index",
            "stage_id",
            "model_family",
            "capacity_profile",
            "lr_profile",
            "seed",
            "text_ablation_mode",
            "tolerance_minutes",
        )
    }
    if str(job["job_id"]) != _job_id(spec):
        raise ValueError(f"Job ID does not encode immutable axes: {job['job_id']}")
    if str(job.get("coverage_cell_sha256", "")) != _coverage_cell_sha256(spec):
        raise ValueError(f"Coverage-cell hash mismatch for {job['job_id']}")
    profile = str(job["capacity_profile"])
    lr_profile = str(job["lr_profile"])
    family = str(job["model_family"])
    seed = int(job["seed"])
    tolerance = int(job["tolerance_minutes"])
    mode = normalize_text_ablation_mode(job["text_ablation_mode"])
    shape = _profiles(resolved)[profile]
    learning_rate, scheduler_min_lr = FROZEN_LR_PROFILES[lr_profile]
    contracts = {
        "capacity_profile_sha256": _capacity_profile_sha256(profile),
        "capacity_seed_profile_sha256": _capacity_seed_profile_sha256(
            profile, lr_profile, seed, family
        ),
        "lr_profile_sha256": _lr_profile_sha256(lr_profile),
        "initial_learning_rate": learning_rate,
        "scheduler_min_lr": scheduler_min_lr,
        "support_mask_mode": "raw_joint",
        "trainer_command": "vol-xlsx" if family == "wgan" else "vol-regression-xlsx",
    }
    for field, expected in contracts.items():
        if job.get(field) != expected:
            raise ValueError(f"Job lineage mismatch for {job['job_id']}: {field}")
    expected_parameters = int(
        shape[
            "expected_wgan_parameters"
            if family == "wgan"
            else "expected_regression_parameters"
        ]
    )
    if int(job.get("expected_model_parameters", -1)) != expected_parameters:
        raise ValueError(f"Parameter-count lineage mismatch for {job['job_id']}")
    if str(job.get("source_manifest_sha256", "")) != _sha256_file(
        root / "source_hashes.csv"
    ):
        raise ValueError(f"Source-manifest hash mismatch for {job['job_id']}")
    for role in (
        f"training_workbook_{tolerance:02d}m",
        f"support_audit_{tolerance:02d}m",
        "orchestration_config",
    ):
        if role not in source_rows:
            raise ValueError(f"Source manifest missing {role}")
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
        "news_first_capacity_seed_profile_sha256": _capacity_seed_profile_sha256(
            profile, lr_profile, seed, family
        ),
        "news_first_lr_profile": lr_profile,
        "news_first_lr_profile_sha256": _lr_profile_sha256(lr_profile),
        "news_first_fixed_learning_rate_profile": lr_profile,
        "news_first_fixed_learning_rate_profile_sha256": _lr_profile_sha256(lr_profile),
        "learning_rate": learning_rate,
        "reduce_lr_min_lr": scheduler_min_lr,
        "early_stopping_min_epochs": 30,
        "early_stopping_patience": 16 if family == "wgan" else 12,
        "support_mask_mode": "raw_joint",
        "residual_output_mode": "identity_softplus_residual",
        "news_first_dataset_tolerance_minutes": tolerance,
        "news_first_text_ablation_mode": mode,
    }
    for field, expected in payload_contracts.items():
        if payload.get(field) != expected:
            raise ValueError(f"Training contract mismatch for {job['job_id']}: {field}")
    for field in training.CAPACITY_PROFILE_SHAPE_FIELDS:
        if int(payload.get(field, -1)) != int(shape[field]):
            raise ValueError(f"Capacity shape mismatch for {job['job_id']}: {field}")
    dataset_role = f"training_workbook_{tolerance:02d}m"
    if str(job["dataset_sha256"]) != source_rows[dataset_role]["sha256"]:
        raise ValueError(f"Dataset hash mismatch for {job['job_id']}")
    if str(job["dataset_path"]) != source_rows[dataset_role]["path"]:
        raise ValueError(f"Dataset path mismatch for {job['job_id']}")
    if Path(str(job["output_root"])) != Path(str(payload["output_root"])):
        raise ValueError(f"Output path mismatch for {job['job_id']}")


def _validate_registry(root: Path, config_path: str | Path | None = None) -> None:
    resolved = _validate_root_lineage(root, config_path)
    registry = _load_registry(root)
    jobs = [dict(job) for job in registry.get("jobs", [])]
    expected_specs = _job_specs(resolved)
    expected_by_id = {_job_id(spec): spec for spec in expected_specs}
    expected_ids = set(expected_by_id)
    observed_ids = {str(job.get("job_id")) for job in jobs}
    if len(observed_ids) != len(jobs):
        raise ValueError("Coverage registry contains duplicate job IDs")
    if observed_ids != expected_ids or len(jobs) != EXPECTED_TOTAL_JOBS:
        raise ValueError(
            f"Coverage registry must contain exactly {EXPECTED_TOTAL_JOBS} jobs; "
            f"missing={len(expected_ids - observed_ids)}, extra={len(observed_ids - expected_ids)}"
        )
    source_rows = _read_manifest_rows(root / "source_hashes.csv", "source_role")
    for job in jobs:
        expected = expected_by_id[str(job["job_id"])]
        for field in (
            "stage_index",
            "stage_id",
            "stage_job_index",
            "stage_wave",
            "global_wave",
            "wave",
            "model_family",
            "capacity_profile",
            "lr_profile",
            "seed",
            "text_ablation_mode",
            "tolerance_minutes",
            "gpu_id",
            "gpu_slot",
            "numa_node",
        ):
            if job.get(field) != expected.get(field):
                raise ValueError(
                    f"Registry schedule mismatch for {job['job_id']}: {field}"
                )
        _validate_job_lineage(root, job, resolved, source_rows)
    manifest = _read_manifest_rows(root / "stage_manifest.csv", "job_id")
    if set(manifest) != observed_ids:
        raise ValueError("stage_manifest.csv does not match registry jobs")


TASK_FIELDS = (
    "job_id",
    "stage_index",
    "stage_id",
    "stage_job_index",
    "stage_wave",
    "global_wave",
    "model_family",
    "trainer_command",
    "capacity_profile",
    "capacity_profile_sha256",
    "capacity_seed_profile_sha256",
    "expected_model_parameters",
    "expected_regression_parameters",
    "expected_wgan_parameters",
    "lr_profile",
    "lr_profile_sha256",
    "initial_learning_rate",
    "scheduler_min_lr",
    "lr_trace",
    "seed",
    "text_ablation_mode",
    "text_information_path",
    "support_mask_mode",
    "tolerance_minutes",
    "gpu_id",
    "gpu_slot",
    "numa_node",
    "coverage_cell_sha256",
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
        root / "stage_manifest.csv",
        root / "split_manifest.csv",
        root / "text_ablation_manifest.csv",
        root / "source_hashes.csv",
        root / "code_hashes.csv",
        root / "code_hashes_current.csv",
        root / "config_hashes.csv",
        root / "capacity_lr_profile_manifest.csv",
        root / "coverage_stage_status.json",
        root / "resource_usage.csv",
        root / "resource_summary.csv",
        root / "run_manifest.json",
        root / "registry" / "jobs.json",
        root / "registry" / "experiment_status.json",
        root / "registry" / "resolved_config.sha256",
    ]
    stable.extend((root / "registry" / "jobs").glob("*.status.json"))
    stable.extend((root / "registry" / "stages").glob("*.qa.json"))
    stable.extend(path for path in (root / "analysis").rglob("*") if path.is_file())
    stable.extend(path for path in (root / "report").rglob("*") if path.is_file())
    final_candidate = root / "coverage_final_candidate.json"
    if final_candidate.is_file():
        stable.append(final_candidate)
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


def _write_config_hashes(root: Path) -> Path:
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


def _initial_stage_status() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "status": "prepared",
        "current_stage": STAGE_IDS[0],
        "stages": {
            stage_id: {
                "stage_index": index,
                "status": "prepared",
                "expected_jobs": EXPECTED_STAGE_JOBS[stage_id],
                "completed_jobs": 0,
            }
            for index, stage_id in enumerate(STAGE_IDS, 1)
        },
        "updated_at_utc": _utc_now(),
    }


def _write_run_manifest(root: Path, resolved: Mapping[str, Any]) -> Path:
    registry = _load_registry(root)
    prepared_code = _read_manifest_rows(root / "code_hashes.csv", "relative_path")
    current_code_rows = _code_rows()
    current_code = {str(row["relative_path"]): row for row in current_code_rows}
    changed_code = sorted(
        relative
        for relative in set(prepared_code) | set(current_code)
        if relative not in prepared_code
        or relative not in current_code
        or str(prepared_code[relative]["sha256"])
        != str(current_code[relative]["sha256"])
    )
    current_code_path = _write_csv(
        root / "code_hashes_current.csv",
        current_code_rows,
        ("relative_path", "path", "size_bytes", "sha256"),
    )
    payload = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "experiment_root": str(root),
        "resolved_config_sha256": registry["resolved_config_sha256"],
        "source_hashes_sha256": _sha256_file(root / "source_hashes.csv"),
        "code_hashes_sha256": _sha256_file(root / "code_hashes.csv"),
        "current_code_hashes_sha256": _sha256_file(current_code_path),
        "code_changed_since_prepared_snapshot": bool(changed_code),
        "changed_code_files": changed_code,
        "lineage_note": (
            "The prepared ledger records code available before worker execution; "
            "the current ledger records code used for the latest postprocess render."
        ),
        "reference_roots": registry["reference_roots"],
        "stage_ids": list(STAGE_IDS),
        "expected_total_jobs": EXPECTED_TOTAL_JOBS,
        "default_slots_per_gpu": int(resolved["runtime"]["slots_per_gpu"]),
        "stage_slots_per_gpu": {
            stage_id: _stage_slots_per_gpu(resolved, stage_id) for stage_id in STAGE_IDS
        },
        "stage_total_gpu_worker_slots": {
            stage_id: len(_worker_slots(resolved, stage_id)) for stage_id in STAGE_IDS
        },
        "numa_memory_policy": str(resolved["runtime"]["numa_memory_policy"]),
        "q4_access_policy": "prediction_and_evaluation_forbidden",
        "updated_at_utc": _utc_now(),
    }
    _write_json(root / "run_manifest.json", payload)
    return root / "run_manifest.json"


def prepare_coverage_sweep_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    reuse: bool = False,
) -> Path:
    root = _resolve_repo_path(output_dir)
    resolved = _resolved_config(config_path)
    resolved_sha = _payload_sha256(resolved)
    hash_path = root / "registry" / "resolved_config.sha256"
    if root.exists() and any(root.iterdir()):
        if not reuse:
            raise FileExistsError(f"Experiment root exists; use --reuse: {root}")
        if not hash_path.is_file():
            raise ValueError(f"Existing directory is not a coverage sweep: {root}")
        _validate_registry(root, config_path)
        _refresh_exports(root)
        return root

    training._validate_dataset_summary(Path(resolved["datasets"]["root"]))
    source_rows = _source_rows(resolved)
    code_rows = _code_rows()
    root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "registry/jobs",
        "registry/stages",
        "configs/jobs",
        "logs",
        "resources",
        "runs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _build_split_manifest(resolved, root)
    _write_yaml(root / "resolved_config.yaml", {ROOT_KEY: resolved})
    _atomic_write_text(hash_path, resolved_sha + "\n")
    _write_csv(
        root / "source_hashes.csv",
        source_rows,
        ("source_role", "path", "size_bytes", "sha256"),
    )
    _write_csv(
        root / "code_hashes.csv",
        code_rows,
        ("relative_path", "path", "size_bytes", "sha256"),
    )
    source_sha = _sha256_file(root / "source_hashes.csv")
    source_by_path = {str(row["path"]): str(row["sha256"]) for row in source_rows}
    profile_values = _profiles(resolved)

    jobs: list[dict[str, Any]] = []
    for spec in _job_specs(resolved):
        family = str(spec["model_family"])
        profile = str(spec["capacity_profile"])
        lr_profile = str(spec["lr_profile"])
        seed = int(spec["seed"])
        mode = normalize_text_ablation_mode(spec["text_ablation_mode"])
        job_id = _job_id(spec)
        payload = _training_payload(resolved, root, spec)
        job_config_path = root / "configs" / "jobs" / f"{job_id}.yaml"
        _write_yaml(job_config_path, payload)
        dataset_path = str(payload["data_path"])
        if dataset_path not in source_by_path:
            raise ValueError(f"Dataset source hash unavailable: {dataset_path}")
        shape = profile_values[profile]
        expected_parameters = int(
            shape[
                "expected_wgan_parameters"
                if family == "wgan"
                else "expected_regression_parameters"
            ]
        )
        learning_rate, scheduler_min_lr = FROZEN_LR_PROFILES[lr_profile]
        job: dict[str, Any] = {
            **spec,
            "job_id": job_id,
            "trainer_command": "vol-xlsx"
            if family == "wgan"
            else "vol-regression-xlsx",
            "capacity_profile_sha256": _capacity_profile_sha256(profile),
            "capacity_seed_profile_sha256": _capacity_seed_profile_sha256(
                profile, lr_profile, seed, family
            ),
            "expected_model_parameters": expected_parameters,
            "expected_regression_parameters": int(
                shape["expected_regression_parameters"]
            ),
            "expected_wgan_parameters": int(shape["expected_wgan_parameters"]),
            "lr_profile_sha256": _lr_profile_sha256(lr_profile),
            "initial_learning_rate": learning_rate,
            "scheduler_min_lr": scheduler_min_lr,
            "lr_trace": [learning_rate],
            "text_information_path": text_information_path(mode),
            "support_mask_mode": "raw_joint",
            "coverage_cell_sha256": _coverage_cell_sha256(spec),
            "training_config_path": str(job_config_path),
            "config_sha256": _sha256_file(job_config_path),
            "dataset_path": dataset_path,
            "dataset_sha256": source_by_path[dataset_path],
            "source_manifest_sha256": source_sha,
            "output_root": str(payload["output_root"]),
            "capacity_stage": str(spec["stage_id"]),
        }
        job["job_spec_sha256"] = _job_spec_sha256(job)
        jobs.append(job)
        _write_json(
            _job_status_path(root, job_id),
            {
                "job_id": job_id,
                "status": "prepared",
                "attempt": 0,
                "stage_index": spec["stage_index"],
                "stage_id": spec["stage_id"],
                "model_family": family,
                "capacity_profile": profile,
                "capacity_profile_sha256": job["capacity_profile_sha256"],
                "capacity_seed_profile_sha256": job["capacity_seed_profile_sha256"],
                "lr_profile": lr_profile,
                "lr_profile_sha256": job["lr_profile_sha256"],
                "initial_learning_rate": learning_rate,
                "scheduler_min_lr": scheduler_min_lr,
                "seed": seed,
                "coverage_cell_sha256": job["coverage_cell_sha256"],
                "config_sha256": job["config_sha256"],
                "updated_at_utc": _utc_now(),
            },
        )

    reference_roots = _reference_roots(resolved)
    _write_json(
        root / "registry" / "jobs.json",
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "experiment_root": str(root),
            "resolved_config_sha256": resolved_sha,
            "reference_roots": reference_roots,
            "stage_ids": list(STAGE_IDS),
            "expected_total_jobs": EXPECTED_TOTAL_JOBS,
            "created_at_utc": _utc_now(),
            "jobs": jobs,
        },
    )
    _write_json(
        root / "registry" / "experiment_status.json",
        {
            "status": "prepared",
            "current_stage": STAGE_IDS[0],
            "q4_predictions_generated": False,
            "q4_evaluated": False,
            "updated_at_utc": _utc_now(),
        },
    )
    _write_json(root / "coverage_stage_status.json", _initial_stage_status())
    stage_rows = [
        {
            "job_id": job["job_id"],
            "stage_index": job["stage_index"],
            "stage_id": job["stage_id"],
            "stage_job_index": job["stage_job_index"],
            "stage_wave": job["stage_wave"],
            "global_wave": job["global_wave"],
            "model_family": job["model_family"],
            "capacity_profile": job["capacity_profile"],
            "lr_profile": job["lr_profile"],
            "seed": job["seed"],
            "text_ablation_mode": job["text_ablation_mode"],
            "tolerance_minutes": job["tolerance_minutes"],
            "gpu_id": job["gpu_id"],
            "gpu_slot": job["gpu_slot"],
            "coverage_cell_sha256": job["coverage_cell_sha256"],
            "job_spec_sha256": job["job_spec_sha256"],
        }
        for job in jobs
    ]
    _write_csv(root / "stage_manifest.csv", stage_rows, tuple(stage_rows[0]))
    manifest_rows = []
    for family in ("regression", "wgan"):
        for profile in FROZEN_PROFILES:
            shape = profile_values[profile]
            for lr_profile, (
                learning_rate,
                scheduler_min_lr,
            ) in FROZEN_LR_PROFILES.items():
                for seed in FROZEN_SEEDS:
                    manifest_rows.append(
                        {
                            "model_family": family,
                            "capacity_profile": profile,
                            "capacity_profile_sha256": _capacity_profile_sha256(
                                profile
                            ),
                            "capacity_seed_profile_sha256": _capacity_seed_profile_sha256(
                                profile, lr_profile, seed, family
                            ),
                            "lr_profile": lr_profile,
                            "lr_profile_sha256": _lr_profile_sha256(lr_profile),
                            "initial_learning_rate": learning_rate,
                            "scheduler_min_lr": scheduler_min_lr,
                            "seed": seed,
                            "expected_model_parameters": int(
                                shape[
                                    "expected_wgan_parameters"
                                    if family == "wgan"
                                    else "expected_regression_parameters"
                                ]
                            ),
                            **{
                                field: int(shape[field])
                                for field in training.CAPACITY_PROFILE_SHAPE_FIELDS
                            },
                        }
                    )
    _write_csv(
        root / "capacity_lr_profile_manifest.csv",
        manifest_rows,
        tuple(manifest_rows[0]),
    )
    _write_config_hashes(root)
    _write_run_manifest(root, resolved)
    _validate_registry(root, config_path)
    _refresh_exports(root)
    return root


def _validate_run_contract(
    job: Mapping[str, Any], run_dir: Path, *, dry_run: bool
) -> list[Any]:
    if dry_run:
        return [float(job["initial_learning_rate"])]
    if job["model_family"] == "regression":
        trace = base_lr._validated_run_lr_trace(job, run_dir, dry_run=False)
    else:
        rows = _read_json(run_dir / "metrics" / "training_metrics.json")
        if not isinstance(rows, list) or len(rows) < 2:
            raise ValueError(
                f"WGAN LR trace for {job['job_id']} requires epoch 0 and a learned epoch"
            )
        trace = []
        previous_g = float(job["initial_learning_rate"])
        previous_d = float(job["initial_learning_rate"])
        floor = float(job["scheduler_min_lr"])
        for raw in rows:
            row = _require_mapping(raw, f"WGAN training metric for {job['job_id']}")
            epoch = int(row["epoch"])
            g_lr = float(row["g_lr"])
            d_lr = float(row["d_lr"])
            if not (floor <= g_lr <= previous_g and floor <= d_lr <= previous_d):
                raise ValueError(
                    f"WGAN LR trace violates non-increasing floor contract for "
                    f"{job['job_id']} at epoch {epoch}"
                )
            trace.append({"epoch": epoch, "g_lr": g_lr, "d_lr": d_lr})
            previous_g, previous_d = g_lr, d_lr
        first = trace[0]
        if (
            int(first["epoch"]) != 0
            or float(first["g_lr"]) != float(job["initial_learning_rate"])
            or float(first["d_lr"]) != float(job["initial_learning_rate"])
        ):
            raise ValueError(
                f"WGAN LR trace has invalid epoch-0 origin for {job['job_id']}"
            )
    best = _require_mapping(
        _read_json(run_dir / "metrics" / "best_learned_checkpoint.json"),
        f"best learned checkpoint for {job['job_id']}",
    )
    contracts = {
        "capacity_profile": job["capacity_profile"],
        "capacity_profile_sha256": job["capacity_profile_sha256"],
        "lr_profile": job["lr_profile"],
        "lr_profile_sha256": job["lr_profile_sha256"],
        "initial_learning_rate": job["initial_learning_rate"],
        "scheduler_min_lr": job["scheduler_min_lr"],
    }
    if job["model_family"] == "regression":
        contracts.update(
            {
                "seed": job["seed"],
                "capacity_seed_profile_sha256": job["capacity_seed_profile_sha256"],
                "fixed_lr_profile": job["lr_profile"],
                "fixed_lr_profile_sha256": job["lr_profile_sha256"],
            }
        )
    for field, expected in contracts.items():
        if best.get(field) != expected:
            raise ValueError(
                f"Trainer checkpoint mismatch for {job['job_id']}: {field}"
            )
    resolved_training = _load_yaml_mapping(
        run_dir / "metrics" / "training_resolved_config.yaml",
        f"resolved training config for {job['job_id']}",
    )
    if int(resolved_training.get("seed", -1)) != int(job["seed"]):
        raise ValueError(f"Resolved training seed mismatch for {job['job_id']}")
    return trace


def _completed_job_is_valid(
    root: Path, job: Mapping[str, Any], status: Mapping[str, Any]
) -> bool:
    if not training._completed_job_is_valid(root, job, status):
        return False
    try:
        run_dir = Path(str(status["run_dir"]))
        _validate_run_contract(job, run_dir, dry_run=False)
    except (ValueError, KeyError, FileNotFoundError):
        return False
    return True


def run_coverage_sweep_worker(
    experiment_root: str | Path,
    job_id: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    root = _resolve_repo_path(experiment_root)
    resolved = _validate_root_lineage(root)
    matches = [
        dict(job)
        for job in _load_registry(root)["jobs"]
        if str(job["job_id"]) == str(job_id)
    ]
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicate coverage job: {job_id}")
    job = matches[0]
    source_rows = _read_manifest_rows(root / "source_hashes.csv", "source_role")
    _validate_job_lineage(root, job, resolved, source_rows)
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
        "stage_index": job["stage_index"],
        "stage_id": job["stage_id"],
        "model_family": job["model_family"],
        "capacity_profile": job["capacity_profile"],
        "capacity_profile_sha256": job["capacity_profile_sha256"],
        "capacity_seed_profile_sha256": job["capacity_seed_profile_sha256"],
        "lr_profile": job["lr_profile"],
        "lr_profile_sha256": job["lr_profile_sha256"],
        "initial_learning_rate": job["initial_learning_rate"],
        "scheduler_min_lr": job["scheduler_min_lr"],
        "seed": job["seed"],
        "coverage_cell_sha256": job["coverage_cell_sha256"],
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


def build_coverage_worker_command(
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
        node = int(job["numa_node"])
        command.extend(
            [str(runtime.get("numactl_executable", "numactl")), f"--cpunodebind={node}"]
        )
        policy = str(runtime.get("numa_memory_policy", "preferred")).lower()
        if policy == "preferred":
            command.append(f"--preferred={node}")
        elif policy == "strict":
            command.append(f"--membind={node}")
        elif policy != "none":
            raise ValueError(f"Unsupported NUMA memory policy: {policy}")
    command.extend(
        [
            str(runtime["python_executable"]),
            "-m",
            "scripts.rq3.news_first_vol_coverage_sweep",
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
    root: Path,
    stage_id: str,
    wave: int,
    *,
    resume: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    jobs = [
        dict(job)
        for job in _load_registry(root)["jobs"]
        if str(job["stage_id"]) == stage_id and int(job["global_wave"]) == int(wave)
    ]
    selected: list[dict[str, Any]] = []
    for job in jobs:
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
        if dry_run and resume and status.get("status") == "dry_run_passed":
            continue
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
        interval_seconds=float(runtime.get("resource_sample_interval_seconds", 2)),
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
                build_coverage_worker_command(
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
                        f"Coverage wave {wave} job {jobs[index]['job_id']} exited {code}"
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


def _write_resource_summary(root: Path) -> Path:
    base_path = training._write_resource_summary(root)
    with base_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    jobs = {str(job["job_id"]): job for job in _load_registry(root)["jobs"]}
    resolved = _require_mapping(
        _load_yaml_mapping(root / "resolved_config.yaml", "resolved config").get(
            ROOT_KEY
        ),
        ROOT_KEY,
    )
    for row in rows:
        job = jobs[str(row["job_id"])]
        slots_per_gpu = _stage_slots_per_gpu(resolved, str(job["stage_id"]))
        row.update(
            {
                "stage_index": job["stage_index"],
                "stage_id": job["stage_id"],
                "stage_wave": job["stage_wave"],
                "global_wave": job["global_wave"],
                "lr_profile": job["lr_profile"],
                "lr_profile_sha256": job["lr_profile_sha256"],
                "initial_learning_rate": job["initial_learning_rate"],
                "scheduler_min_lr": job["scheduler_min_lr"],
                "seed": job["seed"],
                "expected_model_parameters": job["expected_model_parameters"],
                "concurrent_slots_on_gpu": slots_per_gpu,
            }
        )
    fields = training.RESOURCE_SUMMARY_FIELDS + (
        "stage_index",
        "stage_id",
        "stage_wave",
        "global_wave",
        "lr_profile",
        "lr_profile_sha256",
        "initial_learning_rate",
        "scheduler_min_lr",
        "seed",
        "expected_model_parameters",
    )
    return _write_csv(root / "resource_summary.csv", rows, fields)


def _stage_status(root: Path) -> dict[str, Any]:
    return _require_mapping(
        _read_json(root / "coverage_stage_status.json"), "stage status"
    )


def _mark_stage(
    root: Path,
    stage_id: str,
    status: str,
    *,
    overall_status: str | None = None,
    **details: Any,
) -> None:
    payload = _stage_status(root)
    stages = _require_mapping(payload.get("stages"), "stage status stages")
    entry = dict(_require_mapping(stages.get(stage_id), stage_id))
    entry.update({"status": status, "updated_at_utc": _utc_now(), **details})
    stages[stage_id] = entry
    payload["stages"] = stages
    payload["current_stage"] = stage_id
    if overall_status is not None:
        payload["status"] = overall_status
    payload["updated_at_utc"] = _utc_now()
    _write_json(root / "coverage_stage_status.json", payload)


def _qa_sha256(payload: Mapping[str, Any]) -> str:
    return _payload_sha256(
        {key: value for key, value in payload.items() if key != "qa_sha256"}
    )


def run_stage_qa(root: str | Path, stage_id: str, *, dry_run: bool = False) -> Path:
    experiment_root = _resolve_repo_path(root)
    _validate_registry(experiment_root)
    if stage_id not in STAGE_IDS:
        raise ValueError(f"Unknown stage: {stage_id}")
    jobs = [
        dict(job)
        for job in _load_registry(experiment_root)["jobs"]
        if str(job["stage_id"]) == stage_id
    ]
    failures = []
    artifact_count = 0
    for job in jobs:
        status = _read_json(_job_status_path(experiment_root, str(job["job_id"])))
        valid = (
            status.get("status") == "dry_run_passed"
            if dry_run
            else _completed_job_is_valid(experiment_root, job, status)
        )
        if not valid:
            failures.append(f"{job['job_id']}={status.get('status', 'missing')}")
        artifact_count += len(status.get("artifacts", []))
    if len(jobs) != EXPECTED_STAGE_JOBS[stage_id]:
        failures.append(f"job_count={len(jobs)}")
    axes = {
        "model_families": sorted({str(job["model_family"]) for job in jobs}),
        "capacity_profiles": sorted({str(job["capacity_profile"]) for job in jobs}),
        "lr_profiles": sorted({str(job["lr_profile"]) for job in jobs}),
        "seeds": sorted({int(job["seed"]) for job in jobs}),
        "tolerances_minutes": sorted({int(job["tolerance_minutes"]) for job in jobs}),
        "text_ablation_modes": sorted({str(job["text_ablation_mode"]) for job in jobs}),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "stage_id": stage_id,
        "stage_index": STAGE_IDS.index(stage_id) + 1,
        "dry_run": bool(dry_run),
        "status": "pass" if not failures else "fail",
        "expected_jobs": EXPECTED_STAGE_JOBS[stage_id],
        "observed_jobs": len(jobs),
        "artifact_count": artifact_count,
        "axes": axes,
        "job_matrix_sha256": _payload_sha256(
            sorted((job["job_id"], job["job_spec_sha256"]) for job in jobs)
        ),
        "failures": failures,
        "validated_at_utc": _utc_now(),
    }
    payload["qa_sha256"] = _qa_sha256(payload)
    path = experiment_root / "registry" / "stages" / f"{stage_id}.qa.json"
    _write_json(path, payload)
    if failures:
        raise RuntimeError(f"Stage QA failed for {stage_id}: {failures[:10]}")
    return path


def _assert_stage_prerequisite(root: Path, stage_index: int, *, dry_run: bool) -> None:
    if stage_index <= 1:
        return
    prior = STAGE_IDS[stage_index - 2]
    status = _stage_status(root)
    entry = _require_mapping(_require_mapping(status["stages"], "stages")[prior], prior)
    expected_status = "dry_run_passed" if dry_run else "completed"
    if str(entry.get("status")) != expected_status:
        raise RuntimeError(
            f"Cannot start {STAGE_IDS[stage_index - 1]}: {prior} is {entry.get('status')}"
        )
    qa_path = root / "registry" / "stages" / f"{prior}.qa.json"
    if not qa_path.is_file():
        raise RuntimeError(f"Missing prerequisite QA: {qa_path}")
    qa = _require_mapping(_read_json(qa_path), f"QA {prior}")
    if qa.get("status") != "pass" or bool(qa.get("dry_run")) != bool(dry_run):
        raise RuntimeError(f"Prerequisite QA is not valid for {prior}")
    if str(qa.get("qa_sha256", "")) != _qa_sha256(qa):
        raise RuntimeError(f"Prerequisite QA hash mismatch for {prior}")
    if str(entry.get("qa_sha256", "")) != str(qa["qa_sha256"]):
        raise RuntimeError(f"Stage-status QA hash mismatch for {prior}")


def _mark_experiment(root: Path, status: str, stage_id: str, **details: Any) -> None:
    _write_json(
        root / "registry" / "experiment_status.json",
        {
            "status": status,
            "current_stage": stage_id,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
            "updated_at_utc": _utc_now(),
            **details,
        },
    )


def launch_coverage_sweep_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
    stage_postprocess_hook: Callable[[Path, str], None] | None = None,
    postprocess_hook: Callable[[Path], None] | None = None,
) -> Path:
    root = prepare_coverage_sweep_experiment(config_path, output_dir, reuse=True)
    previous = _read_json(root / "registry" / "experiment_status.json")
    if previous.get("status") in TERMINAL_EXPERIMENT_STATES and not dry_run:
        if not resume:
            raise RuntimeError("Terminal coverage sweep requires --resume")
        if previous.get("status") == "completed_q3_only":
            _validate_registry(root, config_path)
            _refresh_exports(root)
            return root
    if previous.get("status") in {"running", "failed"} and not (resume or dry_run):
        raise RuntimeError("Interrupted coverage sweep requires --resume")
    _mark_experiment(root, "dry_running" if dry_run else "running", STAGE_IDS[0])
    try:
        for stage_index, stage_id in enumerate(STAGE_IDS, 1):
            _assert_stage_prerequisite(root, stage_index, dry_run=dry_run)
            current_entry = _require_mapping(
                _require_mapping(_stage_status(root)["stages"], "stages")[stage_id],
                stage_id,
            )
            terminal = "dry_run_passed" if dry_run else "completed"
            if current_entry.get("status") == terminal and resume:
                qa_path = root / "registry" / "stages" / f"{stage_id}.qa.json"
                qa = _read_json(qa_path)
                if qa.get("status") != "pass" or qa.get("qa_sha256") != _qa_sha256(qa):
                    raise RuntimeError(f"Invalid completed-stage QA for {stage_id}")
                continue
            _mark_experiment(root, "dry_running" if dry_run else "running", stage_id)
            _mark_stage(
                root,
                stage_id,
                "dry_running" if dry_run else "running",
                overall_status="dry_running" if dry_run else "running",
                started_at_utc=_utc_now(),
            )
            waves = sorted(
                {
                    int(job["global_wave"])
                    for job in _load_registry(root)["jobs"]
                    if str(job["stage_id"]) == stage_id
                }
            )
            for wave in waves:
                jobs = _jobs_for_wave(
                    root,
                    stage_id,
                    wave,
                    resume=resume,
                    dry_run=dry_run,
                )
                _run_wave(
                    root,
                    config_path,
                    wave,
                    jobs,
                    dry_run=dry_run,
                    resume=resume,
                )
                _write_resource_summary(root)
                _refresh_exports(root)
            qa_path = run_stage_qa(root, stage_id, dry_run=dry_run)
            qa = _read_json(qa_path)
            if not dry_run and stage_postprocess_hook is not None:
                stage_postprocess_hook(root, stage_id)
            _mark_stage(
                root,
                stage_id,
                terminal,
                overall_status="dry_running" if dry_run else "running",
                completed_jobs=EXPECTED_STAGE_JOBS[stage_id],
                qa_path=str(qa_path),
                qa_sha256=qa["qa_sha256"],
                completed_at_utc=_utc_now(),
            )
        if dry_run:
            _mark_experiment(
                root, "dry_run_passed", STAGE_IDS[-1], completed_at_utc=_utc_now()
            )
            status = _stage_status(root)
            status["status"] = "dry_run_passed"
            status["updated_at_utc"] = _utc_now()
            _write_json(root / "coverage_stage_status.json", status)
        else:
            if postprocess_hook is not None:
                postprocess_hook(root)
            _mark_experiment(
                root, "completed_q3_only", STAGE_IDS[-1], completed_at_utc=_utc_now()
            )
            status = _stage_status(root)
            status["status"] = "completed_q3_only"
            status["updated_at_utc"] = _utc_now()
            _write_json(root / "coverage_stage_status.json", status)
    except BaseException as exc:
        current_stage = str(_stage_status(root).get("current_stage", STAGE_IDS[0]))
        _mark_stage(
            root,
            current_stage,
            "failed",
            overall_status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        _mark_experiment(
            root,
            "failed",
            current_stage,
            error=f"{type(exc).__name__}: {exc}",
        )
        _refresh_exports(root)
        raise
    _write_resource_summary(root)
    _write_run_manifest(root, _validate_root_lineage(root, config_path))
    _refresh_exports(root)
    return root


def run_news_first_vol_coverage_sweep(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    action: str = "launch",
    job_id: str = "",
    stage_id: str = "",
    resume: bool = False,
    reuse: bool = False,
    worker_dry_run: bool = False,
    stage_postprocess_hook: Callable[[Path, str], None] | None = None,
    postprocess_hook: Callable[[Path], None] | None = None,
) -> Path:
    normalized = str(action).strip().lower()
    if normalized == "prepare":
        return prepare_coverage_sweep_experiment(
            config_path, output_dir, reuse=bool(reuse or resume)
        )
    if normalized == "dry-run":
        return launch_coverage_sweep_experiment(
            config_path, output_dir, resume=resume, dry_run=True
        )
    if normalized == "worker":
        if not job_id:
            raise ValueError("--job-id is required for worker")
        return run_coverage_sweep_worker(
            output_dir, job_id, dry_run=worker_dry_run, resume=resume
        )
    if normalized == "qa":
        if not stage_id:
            raise ValueError("--stage-id is required for qa")
        return run_stage_qa(output_dir, stage_id, dry_run=worker_dry_run)
    if normalized == "launch":
        return launch_coverage_sweep_experiment(
            config_path,
            output_dir,
            resume=resume,
            dry_run=False,
            stage_postprocess_hook=stage_postprocess_hook,
            postprocess_hook=postprocess_hook,
        )
    raise ValueError(f"Unsupported coverage sweep action: {action}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "dry-run", "launch", "worker", "qa")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--stage-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--worker-dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = run_news_first_vol_coverage_sweep(
        args.config,
        args.output_dir,
        action=args.action,
        job_id=args.job_id,
        stage_id=args.stage_id,
        resume=args.resume,
        reuse=args.reuse,
        worker_dry_run=args.worker_dry_run,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_SLOTS_PER_GPU",
    "EXPECTED_STAGE_JOBS",
    "EXPECTED_TOTAL_JOBS",
    "EXPERIMENT_KIND",
    "FROZEN_LR_PROFILES",
    "FROZEN_PROFILES",
    "FROZEN_SEEDS",
    "STAGE_IDS",
    "build_coverage_worker_command",
    "launch_coverage_sweep_experiment",
    "prepare_coverage_sweep_experiment",
    "run_coverage_sweep_worker",
    "run_news_first_vol_coverage_sweep",
    "run_stage_qa",
]
