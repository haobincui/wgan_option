"""Direct five-arm FiLM U-Net experiment for rolling 2023 evaluation.

This branch-local orchestrator deliberately removes the historical
parent/continuation/branch state chain.  Every fold/arm cell starts from the
same seed-42 Generator and Critic initialization, trains with its own dynamic
validation scheduler, and freezes its own ``best_learned`` checkpoint before
any fold-test row is opened.

The heavy, already-audited data and inference primitives live in the unified
RQ1--RQ3 orchestrator.  They are used here through a temporary profile so the
old experiment and its completed roots remain byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd
import torch
import yaml

from scripts.rq123 import news_first_vol_film_nolp_10seed as core
from scripts.rq123 import news_first_vol_film_unet_nolp_10seed as unet
from scripts.rq3.news_first_vol_film_unet_direct_5arm_seed42_analysis import (
    analyze_experiment,
    fold_session_paired_bootstrap,
    holm_adjust,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/rq3/news_first_vol_film_unet_direct_5arm_seed42.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq12_news_first_vol_film_unet_text128_c32_nolp_direct_5arm_seed42_"
    "exact_ttm_rolling_v1"
)
EXPERIMENT_KIND = (
    "rq12_news_first_vol_film_unet_text128_c32_nolp_direct_5arm_seed42_rolling_v1"
)
INTERPRETATION = "retrospective_rolling_development_single_seed_descriptive"
DIRECT_STAGE = "direct_arms"
DIRECT_ARMS = ("lp_matched", "lp_shuffle", "no_text", "bow", "sentiment")
NO_TEXT_ARM = "no_text"
FOLDS = ("f1_2023q1", "f2_2023q2", "f3_2023q3", "f4_2023q4")
SEED = 42
TOLERANCE_MINUTES = 5
EXPECTED_TRAINING_JOBS = 20
EXPECTED_PREDICTION_CELLS = 20
EXPECTED_PAIR_METRIC_ROWS = 2_500
EXPECTED_PARAMETER_COUNTS = {
    "generator": 827_745,
    "critic": 729_157,
    "total": 1_556_902,
}
GENERATOR_MODE = "film_unet_mask_coords_v1"
CRITIC_MODE = "lp_disabled_same_shape_v1"
CAPACITY_PROFILE = "c32"
EXPECTED_ARCHITECTURE_PROFILE_SHA256 = unet.EXPECTED_ARCHITECTURE_PROFILE_SHA256
WORKER_MODULE = "scripts.rq3.news_first_vol_film_unet_direct_5arm_seed42"
GRID = (1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38)
GRID_FINGERPRINT = core.GRID_FINGERPRINT
INFERENCE_DETERMINISM_KIND = "direct_5arm_inference_determinism_v1"
BENCHMARK_RESULT_KIND = "direct_5arm_full_matrix_epoch1_benchmark_v1"
REGISTRY_KIND = "direct_5arm_task_registry_v1"

SOURCE_CODE_RELATIVE_PATHS = (
    "scripts/rq3/news_first_vol_film_unet_direct_5arm_seed42.py",
    "scripts/rq3/news_first_vol_film_unet_direct_5arm_seed42_analysis.py",
    "scripts/rq123/news_first_vol_film_nolp_10seed.py",
    "scripts/rq123/news_first_vol_film_unet_nolp_10seed.py",
    "scripts/rq3/news_first_vol_comparison_analysis.py",
    "scripts/rq3/news_first_vol_training.py",
    "scripts/rq2_pair/pair_features.py",
    "scripts/rq2_pair/rq2_pair_experiment.py",
)

sha256_file = core.sha256_file
payload_sha256 = core.payload_sha256
read_json = core.read_json
write_json = core.write_json
write_yaml = core.write_yaml
write_csv = core.write_csv
manifest_row = core.manifest_row
verify_manifest_rows = core.verify_manifest_rows
utc_now = core.utc_now


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _verify_frozen_file(path: str | Path, expected_sha256: str) -> Path:
    target = Path(path).resolve()
    if not target.is_file() or sha256_file(target) != str(expected_sha256):
        raise ValueError(f"Frozen file drift: {target}")
    return target


@contextmanager
def direct_profile() -> Iterator[None]:
    """Install only this experiment's globals while calling shared helpers."""

    replacements: dict[str, Any] = {
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "EXPERIMENT_KIND": EXPERIMENT_KIND,
        "INTERPRETATION": INTERPRETATION,
        "SEEDS": (SEED,),
        "TOLERANCES": (TOLERANCE_MINUTES,),
        "FOLDS": FOLDS,
        "PARENT_ARM": NO_TEXT_ARM,
        "CONTINUATION_ARM": "__unused_continuation__",
        "TEXT_ARMS_5M": tuple(arm for arm in DIRECT_ARMS if arm != NO_TEXT_ARM),
        "TEXT_ARMS_30M": (),
        "ALL_ARMS": DIRECT_ARMS,
        "PARENT_STAGE": DIRECT_STAGE,
        "CONTINUATION_STAGE": "__unused_continuations__",
        "BRANCH_STAGE": "__unused_text_branches__",
        "STAGES": (DIRECT_STAGE,),
        "EXPECTED_STAGE_COUNTS": {DIRECT_STAGE: EXPECTED_TRAINING_JOBS},
        "EXPECTED_TRAINING_JOBS": EXPECTED_TRAINING_JOBS,
        "EXPECTED_PREDICTION_CELLS": EXPECTED_PREDICTION_CELLS,
        "WORKER_MODULE": WORKER_MODULE,
        "SOURCE_CODE_RELATIVE_PATHS": SOURCE_CODE_RELATIVE_PATHS,
        "INFERENCE_DETERMINISM_KIND": INFERENCE_DETERMINISM_KIND,
        "GENERATOR_MODE": GENERATOR_MODE,
        "CRITIC_MODE": CRITIC_MODE,
        "CAPACITY_PROFILE": CAPACITY_PROFILE,
        "EXPECTED_PARAMETER_COUNTS": EXPECTED_PARAMETER_COUNTS,
        "EXPECTED_ARCHITECTURE_PROFILE_SHA256": (EXPECTED_ARCHITECTURE_PROFILE_SHA256),
    }
    with unet.unet_nolp_profile():
        originals = {name: getattr(core, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(core, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(core, name, value)


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = resolve_path(config_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Direct five-arm config must be a mapping")
    config = deepcopy(dict(raw))
    config["source_config_path"] = str(source)
    config["source_config_sha256"] = sha256_file(source)
    validate_config(config)
    return config


def _load_frozen_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Frozen direct-five-arm config must be a mapping")
    config = dict(raw)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    experiment = _mapping(config.get("experiment"), "experiment")
    data = _mapping(config.get("data"), "data")
    matrix = _mapping(config.get("matrix"), "matrix")
    model = _mapping(config.get("model"), "model")
    training = _mapping(config.get("training"), "training")
    analysis = _mapping(config.get("analysis"), "analysis")
    runtime = _mapping(config.get("runtime"), "runtime")
    unsupported_lineage = {
        "parent_arms",
        "continuation_arms",
        "branch_recipes",
        "parent_state_allowlist",
        "lr_replay",
    }
    present_lineage = sorted(unsupported_lineage & set(matrix))
    if present_lineage:
        raise ValueError(
            "Direct experiment contains unsupported parent/continuation lineage: "
            f"{present_lineage}"
        )
    if int(experiment.get("schema_version", -1)) != 1:
        raise ValueError("experiment.schema_version must be 1")
    if experiment.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment kind drift")
    if experiment.get("interpretation") != INTERPRETATION:
        raise ValueError("Interpretation label drift")
    if tuple(map(int, data.get("tolerances_minutes", ()))) != (5,):
        raise ValueError("Direct experiment requires exactly the 5m alignment")
    if tuple(map(int, data.get("maturity_days_grid", ()))) != GRID:
        raise ValueError("Exact-TTM grid drift")
    if str(data.get("support_mask_mode")) != "raw_joint":
        raise ValueError("support_mask_mode must be raw_joint")
    if int(data.get("embedding_dimension", -1)) != 1024:
        raise ValueError("Text overlay dimension must be 1024")
    folds = list(config.get("folds") or [])
    if tuple(str(row.get("id")) for row in folds) != FOLDS:
        raise ValueError("Rolling-fold identifiers/order drift")
    for row in folds:
        train_end = pd.Timestamp(row["train_end_utc"])
        validation_start = pd.Timestamp(row["validation_start_utc"])
        validation_end = pd.Timestamp(row["validation_end_utc"])
        test_start = pd.Timestamp(row["test_start_utc"])
        test_end = pd.Timestamp(row["test_end_utc"])
        if not (
            train_end == validation_start < validation_end == test_start < test_end
        ):
            raise ValueError(f"Invalid half-open fold windows: {row['id']}")
    if tuple(map(int, matrix.get("seeds", ()))) != (SEED,):
        raise ValueError("Direct experiment seed must be exactly 42")
    if tuple(map(int, matrix.get("tolerances_minutes", ()))) != (5,):
        raise ValueError("matrix.tolerances_minutes must be [5]")
    if tuple(matrix.get("direct_arms", ())) != DIRECT_ARMS:
        raise ValueError("Direct five-arm order drift")
    if int(matrix.get("expected_training_jobs", -1)) != EXPECTED_TRAINING_JOBS:
        raise ValueError("Expected exactly 20 training jobs")
    if int(matrix.get("expected_prediction_cells", -1)) != EXPECTED_PREDICTION_CELLS:
        raise ValueError("Expected exactly 20 prediction cells")
    if int(matrix.get("expected_pair_metric_rows", -1)) != EXPECTED_PAIR_METRIC_ROWS:
        raise ValueError("Expected exactly 2,500 pair-metric rows")
    if model.get("generator_conditioning_mode") != GENERATOR_MODE:
        raise ValueError(f"Generator must be {GENERATOR_MODE}")
    if model.get("critic_conditioning_mode") != CRITIC_MODE:
        raise ValueError(f"Critic must be {CRITIC_MODE}")
    if model.get("capacity_profile") != CAPACITY_PROFILE:
        raise ValueError(f"Capacity must be {CAPACITY_PROFILE}")
    counts = {
        "generator": int(model.get("expected_generator_parameters", -1)),
        "critic": int(model.get("expected_critic_parameters", -1)),
        "total": int(model.get("expected_total_parameters", -1)),
    }
    if counts != EXPECTED_PARAMETER_COUNTS:
        raise ValueError("Executable parameter-count contract drift")
    required_training = {
        "protocol": "independent_random_init_v1",
        "num_epochs": 240,
        "early_stopping_min_epochs": 30,
        "early_stopping_patience": 20,
        "lr_warmup_epochs": 0,
        "batch_size": 16,
        "discriminator_steps": 5,
        "validation_mc_samples": 16,
        "prediction_mc_samples": 64,
    }
    for key, expected in required_training.items():
        observed = training.get(key)
        if observed != expected:
            raise ValueError(f"training.{key} drift: {observed!r} != {expected!r}")
    if float(training.get("initial_learning_rate", math.nan)) != 5e-7:
        raise ValueError("Initial learning rate must be 5e-7")
    if float(training.get("scheduler_min_lr", math.nan)) != 5e-8:
        raise ValueError("Scheduler floor must be 5e-8")
    if int(analysis.get("bootstrap_replicates", -1)) != 10_000:
        raise ValueError("Analysis requires 10,000 bootstrap replicates")
    if bool(analysis.get("cross_seed_inference_enabled", True)):
        raise ValueError("Cross-seed inference must be disabled")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != (0, 1):
        raise ValueError("Runtime requires GPU 0 and GPU 1")
    if int(runtime.get("benchmark_jobs", -1)) != EXPECTED_TRAINING_JOBS:
        raise ValueError("Benchmark must exercise all 20 cells")
    if int(runtime.get("benchmark_epochs", -1)) != 1:
        raise ValueError("Benchmark must use one epoch")
    primary_workers = int(runtime.get("benchmark_workers_per_gpu", -1))
    fallback_workers = int(runtime.get("fallback_workers_per_gpu", -1))
    if primary_workers < 1:
        raise ValueError("Primary benchmark concurrency must be positive")
    if fallback_workers < 1 or fallback_workers > primary_workers:
        raise ValueError(
            "Fallback concurrency must be positive and no larger than primary"
        )
    formal = runtime.get("formal_workers_per_gpu")
    if formal is not None and int(formal) not in (fallback_workers, primary_workers):
        raise ValueError(
            "Frozen formal concurrency must be null, fallback, or primary workers/GPU"
        )
    with direct_profile():
        # This also verifies the executable U-Net architecture-profile digest.
        core.architecture_profile_contract(config)


def grid_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    with direct_profile():
        return core.grid_contract(config)


def model_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    with direct_profile():
        return core.model_contract(config)


def _state_dict_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_state_sha256(path: str | Path) -> str:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("state_dict"), Mapping
    ):
        raise ValueError(f"Checkpoint lacks a state_dict: {path}")
    digest = hashlib.sha256()
    for key, tensor in payload["state_dict"].items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Non-tensor state entry {key!r}: {path}")
        value = tensor.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _initial_state_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    """Hash the exact isolated G-then-D initialization used by WGAN_GP."""

    import numpy as np
    from wgan_option.models.discriminator import Discriminator
    from wgan_option.models.generator import Generator

    model = _mapping(config["model"], "model")
    cpu_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(SEED)
        generator = Generator(
            channels=int(model["channels"]),
            embedding_dim=int(model["embedding_dim"]),
            noise_dim=int(model["noise_dim"]),
            surface_height=16,
            surface_width=16,
            base_channels=int(model["gen_base_channels"]),
            res_blocks=int(model["gen_res_blocks"]),
            text_hidden_dim=int(model["gen_text_hidden_dim"]),
            text_out_dim=int(model["gen_text_out_dim"]),
            hidden_dim=int(model["gen_hidden_dim"]),
            residual_output_mode=str(model["residual_output_mode"]),
            generator_noise_mode=str(model["generator_noise_mode"]),
            generator_current_input_mode=str(model["generator_current_input_mode"]),
            generator_conditioning_mode=str(model["generator_conditioning_mode"]),
            strike_grid=np.asarray(config["data"]["strike_grid"], dtype=np.float32),
            maturity_grid_days=np.asarray(
                config["data"]["maturity_days_grid"], dtype=np.float32
            ),
        )
        critic = Discriminator(
            channels=int(model["channels"]),
            embedding_dim=int(model["embedding_dim"]),
            surface_height=16,
            surface_width=16,
            base_channels=int(model["disc_base_channels"]),
            res_blocks=int(model["disc_res_blocks"]),
            text_hidden_dim=int(model["disc_text_hidden_dim"]),
            hidden_dim=int(model["disc_hidden_dim"]),
            critic_normalization_mode=str(model["critic_normalization_mode"]),
            critic_conditioning_mode=str(model["critic_conditioning_mode"]),
        )
        observed = {
            "generator": sum(value.numel() for value in generator.parameters()),
            "critic": sum(value.numel() for value in critic.parameters()),
        }
        if observed != {
            "generator": EXPECTED_PARAMETER_COUNTS["generator"],
            "critic": EXPECTED_PARAMETER_COUNTS["critic"],
        }:
            raise ValueError(f"Executable parameter count drift: {observed}")
        return {
            "initial_generator_state_sha256": _state_dict_sha256(generator),
            "initial_critic_state_sha256": _state_dict_sha256(critic),
        }
    finally:
        torch.random.set_rng_state(cpu_state)


def _job_id(fold: str, arm: str) -> str:
    return f"{DIRECT_STAGE}_05m_{fold}_seed_{SEED}_{arm}"


def planned_specs(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    resolved = load_config(DEFAULT_CONFIG) if config is None else dict(config)
    validate_config(resolved)
    initial = _initial_state_hashes(resolved)
    assignment = _mapping(resolved["runtime"]["gpu_fold_assignment"], "gpu assignment")
    specs: list[dict[str, Any]] = []
    for fold in FOLDS:
        for arm in DIRECT_ARMS:
            specs.append(
                {
                    "stage": DIRECT_STAGE,
                    "tolerance_minutes": TOLERANCE_MINUTES,
                    "fold": fold,
                    "seed": SEED,
                    "arm": arm,
                    "gpu_id": int(assignment[fold]),
                    "job_id": _job_id(fold, arm),
                    **initial,
                }
            )
    if (
        len(specs) != EXPECTED_TRAINING_JOBS
        or len({str(row["job_id"]) for row in specs}) != EXPECTED_TRAINING_JOBS
    ):
        raise AssertionError(
            f"Direct experiment must contain {EXPECTED_TRAINING_JOBS} unique jobs"
        )
    validate_gpu_balance(specs)
    return specs


def validate_gpu_balance(specs: Sequence[Mapping[str, Any]]) -> None:
    fold_gpu: dict[str, int] = {}
    for row in specs:
        fold = str(row["fold"])
        gpu = int(row["gpu_id"])
        if fold_gpu.setdefault(fold, gpu) != gpu:
            raise ValueError(f"Arms in fold {fold} do not share one physical GPU")
    counts = {gpu: sum(int(row["gpu_id"]) == gpu for row in specs) for gpu in (0, 1)}
    expected_per_gpu = EXPECTED_TRAINING_JOBS // 2
    if EXPECTED_TRAINING_JOBS % 2 or counts != {
        0: expected_per_gpu,
        1: expected_per_gpu,
    }:
        raise ValueError(
            f"GPU job balance must be {expected_per_gpu}/{expected_per_gpu}, "
            f"got {counts}"
        )
    if fold_gpu != {
        "f1_2023q1": 0,
        "f2_2023q2": 1,
        "f3_2023q3": 0,
        "f4_2023q4": 1,
    }:
        raise ValueError(f"Frozen fold/GPU assignment drift: {fold_gpu}")


def _overlay_mode(arm: str) -> str:
    normalized = str(arm)
    if normalized == NO_TEXT_ARM:
        return "current_only"
    return {
        "lp_matched": "lp_mean_l2",
        "lp_shuffle": "lp_shuffle",
        "bow": "bow1024",
        "sentiment": "sentiment_pad1024",
    }[normalized]


def _fold(config: Mapping[str, Any], fold_id: str) -> dict[str, Any]:
    rows = [dict(row) for row in config["folds"] if str(row["id"]) == fold_id]
    if len(rows) != 1:
        raise KeyError(fold_id)
    return rows[0]


def _expected_counts(config: Mapping[str, Any], fold_id: str) -> dict[str, int]:
    all_counts = _mapping(config["expected_pair_session_counts"], "expected counts")
    tolerance = all_counts.get(5, all_counts.get("5"))
    return {
        key: int(value)
        for key, value in _mapping(tolerance, "expected counts.5")[fold_id].items()
    }


def _source_paths(config: Mapping[str, Any]) -> list[tuple[str, Path]]:
    data = _mapping(config["data"], "data")
    dataset = resolve_path(data["root"])
    return [
        ("source_config", resolve_path(config["source_config_path"])),
        ("dataset_manifest", dataset / "dataset_output_sha256.txt"),
        ("dataset_validation", dataset / "validation_summary.json"),
        ("dataset_news_master", resolve_path(data["news_master_path"])),
        ("sentiment_workbook", resolve_path(data["sentiment_workbook_path"])),
        (
            "dataset_workbook_05m",
            dataset / str(data["workbook_template"]).format(tolerance02="05"),
        ),
        ("support_audit_05m", dataset / "tolerance_05m/surface_support_audit.csv.gz"),
    ]


def _code_paths() -> list[Path]:
    paths = {resolve_path(value) for value in SOURCE_CODE_RELATIVE_PATHS}
    paths.update(path.resolve() for path in (REPO_ROOT / "src").rglob("*.py"))
    paths.update(path.resolve() for path in (REPO_ROOT / "scripts/rq123").rglob("*.py"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Direct experiment runtime source missing: {missing}")
    return sorted(paths, key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def _write_hash_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    result = write_csv(
        path,
        rows,
        ("artifact_role", "path", "size_bytes", "sha256"),
    )
    verify_manifest_rows(rows)
    return result


def _assert_hash_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    observed = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = pd.DataFrame(rows).astype({"size_bytes": str})
    columns = ["artifact_role", "path", "size_bytes", "sha256"]
    observed = observed[columns].sort_values(columns[:2]).reset_index(drop=True)
    expected = expected[columns].sort_values(columns[:2]).reset_index(drop=True)
    if not observed.equals(expected):
        raise ValueError(f"Frozen hash manifest drift: {path}")
    verify_manifest_rows(observed.to_dict(orient="records"))


def _registry_path(root: Path) -> Path:
    return root / "registry/task_registry.json"


def _status_path(root: Path, job_id: str) -> Path:
    return root / "registry/job_status" / f"{job_id}.json"


def read_registry(root: Path) -> dict[str, Any]:
    return read_json(_registry_path(root))


def write_registry(root: Path, registry: Mapping[str, Any]) -> Path:
    payload = dict(registry)
    payload["jobs_sha256"] = payload_sha256(payload.get("jobs", []))
    payload["updated_at_utc"] = utc_now()
    return write_json(_registry_path(root), payload)


def _experiment_status(root: Path, status: str, **details: Any) -> Path:
    return write_json(
        root / "registry/experiment_status.json",
        {"status": status, "updated_at_utc": utc_now(), **details},
    )


def _run_directory(root: Path, spec: Mapping[str, Any]) -> Path:
    return (
        root
        / "runs/direct_arms/tolerance_05m"
        / str(spec["fold"])
        / f"seed_{SEED}"
        / str(spec["arm"])
    ).resolve()


def _overlay_path(root: Path, fold: str, arm: str) -> Path:
    return (
        root / "inputs/pair_text_overlays/tolerance_05m" / fold / f"{arm}.json"
    ).resolve()


def _materialize_pair_universes(config: Mapping[str, Any], root: Path) -> Path:
    with direct_profile():
        return core.materialize_pair_universes(config, root)


def _materialize_pair_text_overlays(config: Mapping[str, Any], root: Path) -> Path:
    """Create exactly the configured development overlays for each fold."""

    from wgan_option.utils.news_first_experiment_core import (
        build_pair_text_overlay_manifests,
        write_pair_text_overlay_manifest,
    )

    output_dir = root / "inputs/pair_text_overlays"
    generated = build_pair_text_overlay_manifests(
        config=config,
        pair_universe_path=root / "inputs/pair_universes.csv",
        output_dir=output_dir,
    )
    if len(generated) != 24:
        raise ValueError(
            f"Shared builder produced {len(generated)} rather than 24 overlays"
        )
    for fold in FOLDS:
        parent = output_dir / "tolerance_05m" / fold / "parent_current_only.json"
        if NO_TEXT_ARM in DIRECT_ARMS:
            payload = read_json(parent)
            write_pair_text_overlay_manifest(
                _overlay_path(root, fold, NO_TEXT_ARM),
                mode="current_only",
                namespace=(
                    f"tol05/{fold}/{NO_TEXT_ARM}/direct_independent_training_v1"
                ),
                records=list(payload["records"]),
                transform={
                    **dict(payload.get("transform") or {}),
                    "direct_arm": NO_TEXT_ARM,
                    "parent_or_continuation_state_input": False,
                },
            )
    expected = {_overlay_path(root, fold, arm) for fold in FOLDS for arm in DIRECT_ARMS}
    for path in output_dir.glob("tolerance_*/*/*.json"):
        if path.resolve() not in expected:
            path.unlink()
    paths = sorted(output_dir.glob("tolerance_05m/*/*.json"))
    if (
        set(path.resolve() for path in paths) != expected
        or len(paths) != EXPECTED_TRAINING_JOBS
    ):
        raise ValueError(
            "Development overlay universe does not match the configured fold/arm matrix"
        )
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        unsigned = {
            key: value for key, value in payload.items() if key != "profile_sha256"
        }
        if payload_sha256(unsigned) != payload.get("profile_sha256"):
            raise ValueError(f"Overlay self-hash drift: {path}")
        arm = path.stem
        if payload.get("mode") != _overlay_mode(arm):
            raise ValueError(f"Overlay mode drift: {path}")
        records = list(payload.get("records") or [])
        if not records:
            raise ValueError(f"Overlay is empty: {path}")
        if arm == NO_TEXT_ARM and any(
            any(float(value) != 0.0 for value in record["embedding"])
            for record in records
        ):
            raise ValueError("no_text overlay must be bit-exact zero")
        if arm == "lp_shuffle" and any(
            str(record["pair_id"]) == str(record.get("donor_pair_id", ""))
            for record in records
        ):
            raise ValueError("lp_shuffle contains a fixed point")
        rows.append(manifest_row(f"pair_overlay:{path.relative_to(root)}", path))
    return _write_hash_manifest(root / "inputs/pair_text_overlay_hashes.csv", rows)


def _with_workers(config: Mapping[str, Any], workers: int) -> dict[str, Any]:
    runtime = _mapping(config["runtime"], "runtime")
    candidates = {
        int(runtime["benchmark_workers_per_gpu"]),
        int(runtime["fallback_workers_per_gpu"]),
    }
    if int(workers) not in candidates:
        raise ValueError(f"Workers/GPU must be one of {sorted(candidates)}")
    resolved = deepcopy(dict(config))
    resolved["runtime"] = dict(resolved["runtime"])
    resolved["runtime"]["formal_workers_per_gpu"] = int(workers)
    validate_config(resolved)
    return resolved


def _training_payload(
    config: Mapping[str, Any],
    root: Path,
    spec: Mapping[str, Any],
    *,
    num_epochs: int | None = None,
    slots_per_gpu: int | None = None,
) -> dict[str, Any]:
    """Return the public direct-training contract without touching test data.

    The complete hash-bound payload is emitted by :func:`_build_job` after its
    pair universe and text overlay exist.  This compact view is intentionally
    pure so prelaunch tests can inspect every arm before any root is created.
    """

    del root, slots_per_gpu
    validate_config(config)
    training = _mapping(config["training"], "training")
    model = _mapping(config["model"], "model")
    arm = str(spec["arm"])
    if arm not in DIRECT_ARMS:
        raise ValueError(f"Unknown direct arm: {arm}")
    return {
        "generator_conditioning_mode": model["generator_conditioning_mode"],
        "critic_conditioning_mode": model["critic_conditioning_mode"],
        "num_epochs": int(num_epochs or training["num_epochs"]),
        "early_stopping_min_epochs": int(training["early_stopping_min_epochs"]),
        "early_stopping_patience": int(training["early_stopping_patience"]),
        "validation_mc_samples": int(training["validation_mc_samples"]),
        "news_first_materialize_validation_loader": True,
        "news_first_materialize_test_loader": False,
        "news_first_pair_text_overlay_mode": _overlay_mode(arm),
        "news_first_full_training_state_mode": "save_dynamic_v1",
        "news_first_refit_mode": "none",
        "use_reduce_lr_on_plateau": True,
        "use_early_stopping": True,
        "seed": SEED,
    }


def _build_job(
    config: Mapping[str, Any],
    root: Path,
    spec: Mapping[str, Any],
    *,
    slots_per_gpu: int,
    num_epochs: int,
) -> dict[str, Any]:
    core_spec = {
        key: value
        for key, value in spec.items()
        if key
        not in {
            "job_id",
            "initial_generator_state_sha256",
            "initial_critic_state_sha256",
        }
    }
    with direct_profile():
        job, _ = core.build_job(
            config,
            root,
            core_spec,
            slots_per_gpu=slots_per_gpu,
            num_epochs_override=num_epochs,
        )
    job.update(
        initial_generator_state_sha256=str(spec["initial_generator_state_sha256"]),
        initial_critic_state_sha256=str(spec["initial_critic_state_sha256"]),
        parent_state_path="",
        parent_state_sha256="",
        continuation_job_id="",
        recipe_path="",
        recipe_sha256="",
    )
    job["job_spec_sha256"] = core._job_spec_sha(job)
    return job


def _prepare_hash_manifests(config: Mapping[str, Any], root: Path) -> None:
    source_rows = [manifest_row(role, path) for role, path in _source_paths(config)]
    code_rows = [
        manifest_row(f"code:{path.relative_to(REPO_ROOT)}", path)
        for path in _code_paths()
    ]
    config_rows = [
        manifest_row("resolved_config", root / "resolved_config.yaml"),
        manifest_row("grid_contract", root / "grid_contract.json"),
        manifest_row("model_contract", root / "model_contract.json"),
        manifest_row(
            "pair_universe_manifest", root / "inputs/pair_universe_manifest.json"
        ),
        manifest_row(
            "pair_text_overlay_hashes", root / "inputs/pair_text_overlay_hashes.csv"
        ),
    ]
    _write_hash_manifest(root / "source_hashes.csv", source_rows)
    _write_hash_manifest(root / "code_hashes.csv", code_rows)
    _write_hash_manifest(root / "config_hashes.csv", config_rows)


def prepare(
    config_or_path: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    *,
    resume: bool = False,
    benchmark_evidence: Mapping[str, Any] | None = None,
    workers_per_gpu: int | None = None,
    num_epochs: int | None = None,
    root_mode: str = "formal",
) -> Path:
    config = (
        load_config(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else deepcopy(dict(config_or_path))
    )
    validate_config(config)
    root = Path(output_root).resolve()
    if root.exists():
        for candidate in (
            root / "registry/task_registry.json",
            root / "registry/jobs.json",
        ):
            if candidate.is_file() and bool(
                read_json(candidate).get("terminal_complete")
            ):
                raise RuntimeError("Terminal-complete root is strictly read-only")
        if resume:
            validate_root(root)
            return root
        raise FileExistsError(root)
    workers = int(
        workers_per_gpu
        or config["runtime"].get("formal_workers_per_gpu")
        or config["runtime"]["benchmark_workers_per_gpu"]
    )
    epochs = int(num_epochs or config["training"]["num_epochs"])
    if root_mode == "formal":
        declared = resolve_path(config["experiment"]["output_root"])
        if root != declared:
            raise ValueError(f"Formal output root drift: {root} != {declared}")
        if not benchmark_evidence or benchmark_evidence.get("status") != "passed":
            raise RuntimeError("Formal prepare requires a passed frozen benchmark")
    config = _with_workers(config, workers)
    root.mkdir(parents=True)
    for relative in (
        "registry/job_status",
        "registry/job_locks",
        "configs/jobs",
        "configs/full_state_contracts",
        "inputs/pair_text_overlays",
        "logs",
        "analysis",
        "evaluation",
        "report",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    write_yaml(root / "resolved_config.yaml", config)
    write_json(root / "grid_contract.json", grid_contract(config))
    write_json(root / "model_contract.json", model_contract(config))
    _materialize_pair_universes(config, root)
    _materialize_pair_text_overlays(config, root)
    specs = planned_specs(config)
    assigned = core.assign_waves(specs, slots_per_gpu=workers)
    jobs = [
        _build_job(
            config,
            root,
            spec,
            slots_per_gpu=workers,
            num_epochs=epochs,
        )
        for spec in assigned
    ]
    if len(jobs) != EXPECTED_TRAINING_JOBS or any(
        job.get("parent_state_path") or job.get("recipe_path") for job in jobs
    ):
        raise AssertionError("Direct task registry contains state-chain inputs")
    for job in jobs:
        write_json(
            _status_path(root, str(job["job_id"])),
            {
                "schema_version": 1,
                "job_id": job["job_id"],
                "job_spec_sha256": job["job_spec_sha256"],
                "stage": DIRECT_STAGE,
                "status": "pending",
                "attempt": 0,
                "artifacts": [],
            },
        )
    registry = {
        "schema_version": 1,
        "kind": REGISTRY_KIND,
        "experiment_kind": EXPERIMENT_KIND,
        "interpretation": INTERPRETATION,
        "root_mode": root_mode,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "formal_workers_per_gpu": workers,
        "training_num_epochs": epochs,
        "expected_training_jobs": EXPECTED_TRAINING_JOBS,
        "expected_prediction_cells": EXPECTED_PREDICTION_CELLS,
        "expected_pair_metric_rows": EXPECTED_PAIR_METRIC_ROWS,
        "jobs": jobs,
        "evaluation_frozen": False,
        "test_data_opened": False,
        "predictions_frozen": False,
        "analysis_complete": False,
        "terminal_complete": False,
        "benchmark_evidence": None
        if benchmark_evidence is None
        else dict(benchmark_evidence),
    }
    write_registry(root, registry)
    _experiment_status(root, "prepared", registered_jobs=len(jobs), root_mode=root_mode)
    _prepare_hash_manifests(config, root)
    validate_root(root)
    return root


def _job_from_registry(root: Path, job_id: str) -> dict[str, Any]:
    rows = [
        dict(row) for row in read_registry(root)["jobs"] if str(row["job_id"]) == job_id
    ]
    if len(rows) != 1:
        raise KeyError(job_id)
    return rows[0]


def _completed_valid(job: Mapping[str, Any], status: Mapping[str, Any]) -> bool:
    if (
        status.get("status") != "completed"
        or status.get("job_id") != job.get("job_id")
        or status.get("job_spec_sha256") != job.get("job_spec_sha256")
        or status.get("training_config_sha256") != job.get("training_config_sha256")
    ):
        return False
    try:
        verify_manifest_rows(status.get("artifacts") or [], require_unique_roles=True)
    except (KeyError, FileNotFoundError, TypeError, ValueError):
        return False
    return True


def validate_root(
    root_or_path: str | Path, *, verify_large_inputs: bool = True
) -> dict[str, Any]:
    root = Path(root_or_path).resolve()
    registry = read_registry(root)
    if (
        registry.get("kind") != REGISTRY_KIND
        or registry.get("experiment_kind") != EXPERIMENT_KIND
        or registry.get("interpretation") != INTERPRETATION
    ):
        raise ValueError("Direct experiment registry identity drift")
    jobs = list(registry.get("jobs") or [])
    if len(jobs) != EXPECTED_TRAINING_JOBS or registry.get(
        "jobs_sha256"
    ) != payload_sha256(jobs):
        raise ValueError(f"Direct {EXPECTED_TRAINING_JOBS}-job registry drift")
    if len({str(job["job_id"]) for job in jobs}) != EXPECTED_TRAINING_JOBS:
        raise ValueError("Duplicate direct job IDs")
    config = _load_frozen_config(root / "resolved_config.yaml")
    if read_json(root / "grid_contract.json") != grid_contract(config):
        raise ValueError("Grid contract drift")
    if read_json(root / "model_contract.json") != model_contract(config):
        raise ValueError("Model contract drift")
    source_rows = [manifest_row(role, path) for role, path in _source_paths(config)]
    if verify_large_inputs:
        _assert_hash_manifest(root / "source_hashes.csv", source_rows)
    else:
        # Workers trust the supervisor's just-completed full validation for
        # large raw files and bind that decision to the manifest file hash.
        expected_manifest_sha = os.environ.get("DIRECT_SOURCE_MANIFEST_SHA256", "")
        if not expected_manifest_sha:
            _assert_hash_manifest(root / "source_hashes.csv", source_rows)
        elif sha256_file(root / "source_hashes.csv") != expected_manifest_sha:
            raise ValueError("Supervisor source-manifest attestation drift")
    code_rows = [
        manifest_row(f"code:{path.relative_to(REPO_ROOT)}", path)
        for path in _code_paths()
    ]
    _assert_hash_manifest(root / "code_hashes.csv", code_rows)
    config_rows = [
        manifest_row("resolved_config", root / "resolved_config.yaml"),
        manifest_row("grid_contract", root / "grid_contract.json"),
        manifest_row("model_contract", root / "model_contract.json"),
        manifest_row(
            "pair_universe_manifest", root / "inputs/pair_universe_manifest.json"
        ),
        manifest_row(
            "pair_text_overlay_hashes", root / "inputs/pair_text_overlay_hashes.csv"
        ),
    ]
    _assert_hash_manifest(root / "config_hashes.csv", config_rows)
    initial_g = {str(job["initial_generator_state_sha256"]) for job in jobs}
    initial_d = {str(job["initial_critic_state_sha256"]) for job in jobs}
    if len(initial_g) != 1 or len(initial_d) != 1:
        raise ValueError(
            "Planned epoch-0 state hashes are not common across arms/folds"
        )
    for job in jobs:
        if core._job_spec_sha(job) != job.get("job_spec_sha256"):
            raise ValueError(f"Job spec hash drift: {job['job_id']}")
        if job.get("parent_state_path") or job.get("recipe_path"):
            raise ValueError(
                f"Direct job consumes forbidden parent/recipe: {job['job_id']}"
            )
        for path_key, sha_key in (
            ("training_config_path", "training_config_sha256"),
            ("full_state_contract_path", "full_state_contract_sha256"),
            ("overlay_path", "overlay_sha256"),
        ):
            _verify_frozen_file(job[path_key], job[sha_key])
        if verify_large_inputs:
            _verify_frozen_file(job["dataset_path"], job["dataset_sha256"])
            _verify_frozen_file(job["support_path"], job["support_sha256"])
        status = read_json(_status_path(root, str(job["job_id"])))
        if (
            status.get("job_id") != job["job_id"]
            or status.get("job_spec_sha256") != job["job_spec_sha256"]
        ):
            raise ValueError(f"Job status lineage drift: {job['job_id']}")
        if status.get("status") == "completed" and not _completed_valid(job, status):
            raise ValueError(f"Completed artifact drift: {job['job_id']}")
    for flag, path_key, sha_key in (
        (
            "evaluation_frozen",
            "checkpoint_allowlist_path",
            "checkpoint_allowlist_sha256",
        ),
        ("test_data_opened", "test_input_manifest_path", "test_input_manifest_sha256"),
        (
            "predictions_frozen",
            "prediction_manifest_path",
            "prediction_manifest_sha256",
        ),
        ("predictions_frozen", "pair_metrics_path", "pair_metrics_sha256"),
        ("analysis_complete", "analysis_manifest_path", "analysis_manifest_sha256"),
    ):
        if registry.get(flag):
            _verify_frozen_file(registry[path_key], registry[sha_key])
    if registry.get("test_data_opened") and not registry.get("evaluation_frozen"):
        raise ValueError("Test data opened before checkpoint freeze")
    if registry.get("predictions_frozen") and not registry.get("test_data_opened"):
        raise ValueError("Predictions frozen before test-input freeze")
    return config


_CORE_ARTIFACT_PATHS = core._artifact_paths_for_job


def _direct_artifact_paths(
    job: Mapping[str, Any], run_dir: Path
) -> list[dict[str, Any]]:
    rows = _CORE_ARTIFACT_PATHS(job, run_dir)
    for role, filename in (
        ("generator_initial_epoch0", "generator_initial_epoch0.pt"),
        ("discriminator_initial_epoch0", "discriminator_initial_epoch0.pt"),
    ):
        path = run_dir / "checkpoints" / filename
        if not path.is_file():
            raise RuntimeError(f"Training did not retain {role}: {path}")
        rows.append(manifest_row(role, path))
    verify_manifest_rows(rows)
    return rows


def _direct_prune(job: Mapping[str, Any], run_dir: Path) -> None:
    del job
    checkpoint_dir = (run_dir / "checkpoints").resolve()
    if checkpoint_dir.parent != run_dir.resolve():
        raise RuntimeError("Checkpoint pruning escaped the selected run")
    for name in (
        "generator.pt",
        "discriminator.pt",
        "generator_best.pt",
        "discriminator_best.pt",
    ):
        (checkpoint_dir / name).unlink(missing_ok=True)
    for pattern in ("generator_epoch_*.pt", "discriminator_epoch_*.pt"):
        for path in checkpoint_dir.glob(pattern):
            if path.parent.resolve() != checkpoint_dir:
                raise RuntimeError(f"Refusing out-of-run checkpoint cleanup: {path}")
            path.unlink()


@contextmanager
def _training_execution_profile() -> Iterator[None]:
    with direct_profile():
        original_artifacts = core._artifact_paths_for_job
        original_prune = core._prune_unselected_training_checkpoints
        try:
            core._artifact_paths_for_job = _direct_artifact_paths
            core._prune_unselected_training_checkpoints = _direct_prune
            yield
        finally:
            core._artifact_paths_for_job = original_artifacts
            core._prune_unselected_training_checkpoints = original_prune


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError(f"Live lock already exists: {path}") from exc
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.fsync(descriptor)
    return descriptor


def _release_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def worker(
    output_dir: str | Path,
    job_id: str,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> Path:
    root = Path(output_dir).resolve()
    validate_root(root, verify_large_inputs=False)
    job = _job_from_registry(root, job_id)
    descriptor = _acquire_lock(root / "registry/job_locks" / f"{job_id}.lock")
    try:
        registry = read_registry(root)
        if registry.get("evaluation_frozen") or registry.get("test_data_opened"):
            raise RuntimeError("Training is forbidden after evaluation freeze")
        status_path = _status_path(root, job_id)
        previous = read_json(status_path)
        if _completed_valid(job, previous):
            if resume and not dry_run:
                return Path(str(previous["run_dir"]))
            raise RuntimeError(f"Job already completed: {job_id}")
        if previous.get("status") in {"running", "failed"} and not resume:
            raise RuntimeError(f"Interrupted job requires --resume: {job_id}")
        running = {
            "schema_version": 1,
            "job_id": job_id,
            "job_spec_sha256": job["job_spec_sha256"],
            "stage": DIRECT_STAGE,
            "status": "running",
            "attempt": int(previous.get("attempt", 0)) + 1,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "dry_run": bool(dry_run),
            "started_at_utc": utc_now(),
            "training_config_sha256": job["training_config_sha256"],
            "artifacts": [],
        }
        write_json(status_path, running)
        try:
            core._cleanup_unselected_job_attempts(root, job, keep=None)
            with _training_execution_profile():
                run_dir, artifacts = core._execute_job(job, dry_run=dry_run)
            if not dry_run:
                core._cleanup_unselected_job_attempts(root, job, keep=run_dir)
                by_role = {str(row["artifact_role"]): row for row in artifacts}
                observed_g = _checkpoint_state_sha256(
                    by_role["generator_initial_epoch0"]["path"]
                )
                observed_d = _checkpoint_state_sha256(
                    by_role["discriminator_initial_epoch0"]["path"]
                )
                if observed_g != job["initial_generator_state_sha256"]:
                    raise RuntimeError(f"Generator epoch-0 state drift: {job_id}")
                if observed_d != job["initial_critic_state_sha256"]:
                    raise RuntimeError(f"Critic epoch-0 state drift: {job_id}")
                if observed_g == _checkpoint_state_sha256(
                    by_role["generator_best_learned"]["path"]
                ) or observed_d == _checkpoint_state_sha256(
                    by_role["discriminator_best_learned"]["path"]
                ):
                    raise RuntimeError(f"G/D parameters did not both update: {job_id}")
            completed = {
                **running,
                "status": "dry_run_passed" if dry_run else "completed",
                "run_dir": str(run_dir),
                "artifacts": artifacts,
                "completed_at_utc": utc_now(),
            }
            write_json(status_path, completed)
            return run_dir
        except BaseException as exc:
            write_json(
                status_path,
                {
                    **running,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "completed_at_utc": utc_now(),
                },
            )
            raise
    finally:
        _release_lock(descriptor)


def _host_ram_fraction() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
    total = values.get("MemTotal", 0)
    return 1.0 - values.get("MemAvailable", 0) / total if total else 1.0


def _terminate_processes(processes: Sequence[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                process.terminate()
    deadline = time.monotonic() + 10.0
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                process.kill()


def _worker_command(
    config: Mapping[str, Any], root: Path, job: Mapping[str, Any], resume: bool
) -> list[str]:
    command = [
        str(config["runtime"]["python_executable"]),
        "-m",
        WORKER_MODULE,
        "worker",
        "--output-dir",
        str(root),
        "--job-id",
        str(job["job_id"]),
    ]
    if resume:
        command.append("--resume")
    return command


def _run_wave(
    root: Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    wave: int,
    resume: bool,
) -> float:
    if not jobs:
        return _host_ram_fraction()
    from scripts.rq3.news_first_vol_training import _ResourceMonitor

    config = validate_root(root)
    runtime = _mapping(config["runtime"], "runtime")
    monitor = _ResourceMonitor(
        root / "resource_usage.csv",
        executable=str(runtime["nvidia_smi_executable"]),
        interval_seconds=float(runtime["resource_sample_interval_seconds"]),
        wave=int(wave),
    )
    processes: list[subprocess.Popen[Any]] = []
    handles: list[Any] = []
    peak_ram = _host_ram_fraction()
    try:
        monitor.start()
        for job in jobs:
            previous = read_json(_status_path(root, str(job["job_id"])))
            log_path = (
                root
                / "logs"
                / (
                    f"{job['job_id']}.attempt_{int(previous.get('attempt', 0)) + 1:02d}.log"
                )
            )
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
            environment["NEWS_FIRST_JOB_LOG_PATH"] = str(log_path.resolve())
            environment["DIRECT_SOURCE_MANIFEST_SHA256"] = sha256_file(
                root / "source_hashes.csv"
            )
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(REPO_ROOT / "src"), str(REPO_ROOT))
            )
            threads = str(int(runtime["cpu_threads_per_job"]))
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment[name] = threads
            processes.append(
                subprocess.Popen(
                    _worker_command(config, root, job, resume),
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        pending = set(range(len(processes)))
        while pending:
            peak_ram = max(peak_ram, _host_ram_fraction())
            for index in tuple(pending):
                code = processes[index].poll()
                if code is None:
                    continue
                pending.remove(index)
                if code != 0:
                    raise RuntimeError(
                        f"Wave {wave} job {jobs[index]['job_id']} exited with {code}"
                    )
            if pending:
                time.sleep(0.5)
    finally:
        _terminate_processes(processes)
        monitor.stop()
        for handle in handles:
            handle.close()
    return peak_ram


def launch(output_dir: str | Path, *, resume: bool = False) -> Path:
    root = Path(output_dir).resolve()
    validate_root(root)
    registry = read_registry(root)
    if registry.get("evaluation_frozen"):
        raise RuntimeError("Training cannot restart after evaluation freeze")
    jobs = [dict(job) for job in registry["jobs"]]
    peak_ram = 0.0
    for wave in sorted({int(job["wave"]) for job in jobs}):
        selected: list[dict[str, Any]] = []
        for job in jobs:
            if int(job["wave"]) != wave:
                continue
            status = read_json(_status_path(root, str(job["job_id"])))
            if _completed_valid(job, status):
                if resume:
                    continue
                raise RuntimeError(f"Completed job requires --resume: {job['job_id']}")
            selected.append(job)
        peak_ram = max(peak_ram, _run_wave(root, selected, wave=wave, resume=resume))
    failures = [
        str(job["job_id"])
        for job in jobs
        if not _completed_valid(job, read_json(_status_path(root, str(job["job_id"]))))
    ]
    if failures:
        raise RuntimeError(f"Direct training incomplete: {failures}")
    registry = read_registry(root)
    registry.update(
        status="all_training_complete",
        all_training_complete_at_utc=utc_now(),
        peak_host_ram_fraction=float(peak_ram),
    )
    write_registry(root, registry)
    _experiment_status(
        root, "all_training_complete", completed_jobs=EXPECTED_TRAINING_JOBS
    )
    return root


def _resource_peaks(root: Path) -> dict[str, Any]:
    frame = pd.read_csv(root / "resource_usage.csv")
    if frame.empty or "memory_used_mib" not in frame:
        raise ValueError("Benchmark resource telemetry is empty")
    memory = pd.to_numeric(frame["memory_used_mib"], errors="raise")
    return {
        "peak_gpu_memory_mib": float(memory.max()),
        "peak_gpu_memory_gib": float(memory.max()) / 1024.0,
        "telemetry_rows": int(len(frame)),
        "telemetry_sha256": sha256_file(root / "resource_usage.csv"),
    }


def _validate_benchmark_updates(root: Path) -> None:
    registry = read_registry(root)
    for job in registry["jobs"]:
        status = read_json(_status_path(root, str(job["job_id"])))
        if not _completed_valid(job, status):
            raise ValueError(f"Incomplete benchmark cell: {job['job_id']}")
        artifacts = {row["artifact_role"]: row for row in status["artifacts"]}
        if _checkpoint_state_sha256(
            artifacts["generator_initial_epoch0"]["path"]
        ) == _checkpoint_state_sha256(artifacts["generator_best_learned"]["path"]):
            raise ValueError(f"Benchmark Generator did not update: {job['job_id']}")
        if _checkpoint_state_sha256(
            artifacts["discriminator_initial_epoch0"]["path"]
        ) == _checkpoint_state_sha256(artifacts["discriminator_best_learned"]["path"]):
            raise ValueError(f"Benchmark Critic did not update: {job['job_id']}")
        metrics = pd.read_csv(artifacts["training_metrics_csv"]["path"])
        numeric = metrics.select_dtypes(include="number")
        if numeric.empty or not all(
            math.isfinite(float(value))
            for value in numeric.to_numpy().ravel()
            if not pd.isna(value)
        ):
            raise ValueError(f"Benchmark contains NaN/Inf metrics: {job['job_id']}")


def _benchmark_result_path(formal_root: Path) -> Path:
    return (
        formal_root.with_name(formal_root.name + "_control") / "benchmark_result.json"
    )


def benchmark(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    formal_root = Path(output_dir).resolve()
    result_path = _benchmark_result_path(formal_root)
    if result_path.is_file():
        result = read_json(result_path)
        unsigned = {
            key: value for key, value in result.items() if key != "payload_sha256"
        }
        if (
            result.get("kind") == BENCHMARK_RESULT_KIND
            and result.get("status") == "passed"
            and payload_sha256(unsigned) == result.get("payload_sha256")
            and result.get("source_config_sha256") == config["source_config_sha256"]
        ):
            _verify_frozen_file(
                result["benchmark_root_registry_path"],
                result["benchmark_root_registry_sha256"],
            )
            benchmark_root = Path(str(result["benchmark_root"])).resolve()
            validate_root(benchmark_root)
            return result_path
        raise ValueError("Existing benchmark result drift")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    worker_candidates = tuple(
        dict.fromkeys(
            (
                int(config["runtime"]["benchmark_workers_per_gpu"]),
                int(config["runtime"]["fallback_workers_per_gpu"]),
            )
        )
    )
    for candidate_index, workers in enumerate(worker_candidates):
        benchmark_root = formal_root.with_name(
            formal_root.name + f"_benchmark_{workers}"
        )
        try:
            if not benchmark_root.exists():
                prepare(
                    config,
                    benchmark_root,
                    workers_per_gpu=workers,
                    num_epochs=1,
                    root_mode="benchmark",
                )
            elif not resume:
                raise FileExistsError(benchmark_root)
            launch(benchmark_root, resume=True)
            _validate_benchmark_updates(benchmark_root)
            resources = _resource_peaks(benchmark_root)
            peak_ram = float(
                read_registry(benchmark_root).get("peak_host_ram_fraction", 1.0)
            )
            if resources["peak_gpu_memory_gib"] >= float(
                config["runtime"]["preflight_max_peak_gpu_memory_gib"]
            ):
                raise RuntimeError("Benchmark GPU-memory gate failed")
            if peak_ram >= float(config["runtime"]["preflight_max_host_ram_fraction"]):
                raise RuntimeError("Benchmark host-RAM gate failed")
            registry_path = _registry_path(benchmark_root)
            result = {
                "schema_version": 1,
                "kind": BENCHMARK_RESULT_KIND,
                "status": "passed",
                "source_config_sha256": config["source_config_sha256"],
                "selected_workers_per_gpu": workers,
                "job_count": EXPECTED_TRAINING_JOBS,
                "epoch_count": 1,
                "benchmark_root": str(benchmark_root.resolve()),
                "benchmark_root_registry_path": str(registry_path.resolve()),
                "benchmark_root_registry_sha256": sha256_file(registry_path),
                "benchmark_code_manifest_sha256": sha256_file(
                    benchmark_root / "code_hashes.csv"
                ),
                "benchmark_source_manifest_sha256": sha256_file(
                    benchmark_root / "source_hashes.csv"
                ),
                "benchmark_config_manifest_sha256": sha256_file(
                    benchmark_root / "config_hashes.csv"
                ),
                "peak_host_ram_fraction": peak_ram,
                **resources,
                "completed_at_utc": utc_now(),
            }
            result["payload_sha256"] = payload_sha256(result)
            return write_json(result_path, result)
        except BaseException as exc:
            errors.append(f"{workers}/GPU: {type(exc).__name__}: {exc}")
            capacity_related = (
                "GPU-memory gate failed" in str(exc)
                or "host-RAM gate failed" in str(exc)
                or core._benchmark_failure_is_capacity_related(benchmark_root, exc)
            )
            if candidate_index + 1 < len(worker_candidates) and capacity_related:
                continue
            if candidate_index + 1 == len(worker_candidates) and capacity_related:
                raise RuntimeError(
                    f"Both benchmark concurrency candidates failed: {errors}"
                ) from exc
            raise RuntimeError(
                "Benchmark failed for a non-capacity reason; fallback is forbidden: "
                f"{errors[-1]}"
            ) from exc
    raise AssertionError("Unreachable benchmark path")


def prepare_experiment(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    formal_root = Path(output_dir).resolve()
    result_path = _benchmark_result_path(formal_root)
    if not result_path.is_file():
        raise RuntimeError("Run benchmark before formal prepare")
    result = read_json(result_path)
    unsigned = {key: value for key, value in result.items() if key != "payload_sha256"}
    if (
        result.get("kind") != BENCHMARK_RESULT_KIND
        or result.get("status") != "passed"
        or payload_sha256(unsigned) != result.get("payload_sha256")
        or result.get("source_config_sha256") != config["source_config_sha256"]
    ):
        raise ValueError("Benchmark evidence drift")
    benchmark_root = Path(str(result["benchmark_root"])).resolve()
    benchmark_config = validate_root(benchmark_root)
    if int(benchmark_config["runtime"]["formal_workers_per_gpu"]) != int(
        result["selected_workers_per_gpu"]
    ):
        raise ValueError("Benchmark concurrency/config drift")
    for key, filename in (
        ("benchmark_code_manifest_sha256", "code_hashes.csv"),
        ("benchmark_source_manifest_sha256", "source_hashes.csv"),
        ("benchmark_config_manifest_sha256", "config_hashes.csv"),
    ):
        if result.get(key) != sha256_file(benchmark_root / filename):
            raise ValueError(f"Benchmark {key} drift")
    benchmark_bytes = sum(
        path.stat().st_size for path in benchmark_root.rglob("*") if path.is_file()
    )
    projected_bytes = int(
        benchmark_bytes
        * 2.0
        * float(config["runtime"]["disk_projection_safety_factor"])
    )
    free_bytes = shutil.disk_usage(formal_root.parent).free
    required_remaining = (
        int(config["runtime"]["minimum_free_disk_after_projected_bytes_gib"]) * 1024**3
    )
    if free_bytes - projected_bytes < required_remaining:
        raise RuntimeError(
            "Formal disk gate failed: projected artifacts plus safety margin "
            "would leave less than 30 GiB"
        )
    return prepare(
        config,
        formal_root,
        resume=resume,
        benchmark_evidence=result,
        workers_per_gpu=int(result["selected_workers_per_gpu"]),
        num_epochs=int(config["training"]["num_epochs"]),
        root_mode="formal",
    )


def dry_run(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    """The full-matrix one-epoch benchmark is this experiment's dry run."""

    return benchmark(config_path, output_dir, resume=resume)


def _artifact(status: Mapping[str, Any], role: str) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in status.get("artifacts", [])
        if str(row.get("artifact_role")) == role
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one artifact role {role!r}")
    verify_manifest_rows(rows)
    return rows[0]


def _all_training_complete(root: Path) -> list[dict[str, Any]]:
    jobs = [dict(job) for job in read_registry(root)["jobs"]]
    failures = [
        str(job["job_id"])
        for job in jobs
        if not _completed_valid(job, read_json(_status_path(root, str(job["job_id"]))))
    ]
    if failures:
        raise RuntimeError(
            f"Checkpoint freeze requires {EXPECTED_TRAINING_JOBS} completed jobs: "
            f"{failures}"
        )
    return jobs


def _initial_state_fairness_path(root: Path) -> Path:
    return root / "registry/initial_state_fairness.json"


def _freeze_initial_state_fairness(
    root: Path, jobs: Sequence[Mapping[str, Any]]
) -> Path:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        status = read_json(_status_path(root, str(job["job_id"])))
        initial_g = _artifact(status, "generator_initial_epoch0")
        initial_d = _artifact(status, "discriminator_initial_epoch0")
        best_g = _artifact(status, "generator_best_learned")
        best_d = _artifact(status, "discriminator_best_learned")
        observed_g = _checkpoint_state_sha256(initial_g["path"])
        observed_d = _checkpoint_state_sha256(initial_d["path"])
        if observed_g != job["initial_generator_state_sha256"]:
            raise ValueError(f"Frozen Generator initial state drift: {job['job_id']}")
        if observed_d != job["initial_critic_state_sha256"]:
            raise ValueError(f"Frozen Critic initial state drift: {job['job_id']}")
        if observed_g == _checkpoint_state_sha256(best_g["path"]):
            raise ValueError(f"Generator did not update: {job['job_id']}")
        if observed_d == _checkpoint_state_sha256(best_d["path"]):
            raise ValueError(f"Critic did not update: {job['job_id']}")
        rows.append(
            {
                "job_id": str(job["job_id"]),
                "fold": str(job["fold"]),
                "arm": str(job["arm"]),
                "generator_initial_state_sha256": observed_g,
                "critic_initial_state_sha256": observed_d,
                "generator_best_state_sha256": _checkpoint_state_sha256(best_g["path"]),
                "critic_best_state_sha256": _checkpoint_state_sha256(best_d["path"]),
            }
        )
    generator_hashes = {row["generator_initial_state_sha256"] for row in rows}
    critic_hashes = {row["critic_initial_state_sha256"] for row in rows}
    if len(generator_hashes) != 1 or len(critic_hashes) != 1:
        raise ValueError("Actual epoch-0 G/D states are not identical across all jobs")
    payload = {
        "schema_version": 1,
        "kind": "direct_5arm_common_initial_state_fairness_v1",
        "seed": SEED,
        "initialization_order": "seed_everything(seed), Generator, Critic",
        "generator_initial_state_sha256": next(iter(generator_hashes)),
        "critic_initial_state_sha256": next(iter(critic_hashes)),
        "job_count": len(rows),
        "all_generator_parameters_updated": True,
        "all_critic_parameters_updated": True,
        "rows": sorted(rows, key=lambda row: row["job_id"]),
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return write_json(_initial_state_fairness_path(root), payload)


def _inference_contract() -> dict[str, Any]:
    with direct_profile():
        return core._inference_determinism_contract()


def _checkpoint_allowlist_path(root: Path) -> Path:
    return root / "registry/evaluation_checkpoint_allowlist.csv"


def _checkpoint_manifest_path(root: Path) -> Path:
    return root / "analysis/checkpoint_manifest.csv"


def _checkpoint_map(root: Path) -> dict[str, dict[str, str]]:
    registry = read_registry(root)
    allowlist_path = _verify_frozen_file(
        registry["checkpoint_allowlist_path"], registry["checkpoint_allowlist_sha256"]
    )
    frame = pd.read_csv(allowlist_path, dtype=str, keep_default_na=False)
    expected_checkpoint_rows = EXPECTED_TRAINING_JOBS * 2
    if (
        len(frame) != expected_checkpoint_rows
        or frame.duplicated(["job_id", "checkpoint_role"]).any()
    ):
        raise ValueError(
            f"Checkpoint allowlist must contain {EXPECTED_TRAINING_JOBS} G/D pairs"
        )
    expected_jobs = {str(job["job_id"]) for job in registry["jobs"]}
    if set(frame["job_id"]) != expected_jobs:
        raise ValueError("Checkpoint allowlist job universe drift")
    for row in frame.to_dict(orient="records"):
        path = _verify_frozen_file(row["checkpoint_path"], row["checkpoint_sha256"])
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Checkpoint size drift: {path}")
    generators = frame.loc[frame["checkpoint_role"].eq("generator_best_learned")]
    if len(generators) != EXPECTED_TRAINING_JOBS:
        raise ValueError(
            "Checkpoint allowlist lacks "
            f"{EXPECTED_TRAINING_JOBS} best-learned Generators"
        )
    contract_path = _verify_frozen_file(
        registry["inference_determinism_contract_path"],
        registry["inference_determinism_contract_sha256"],
    )
    if read_json(contract_path) != _inference_contract():
        raise ValueError("Inference determinism contract drift")
    return {
        str(row.job_id): {
            "path": str(row.checkpoint_path),
            "sha256": str(row.checkpoint_sha256),
            "role": "generator_best_learned",
        }
        for row in generators.itertuples(index=False)
    }


def _freeze_checkpoints(root: Path) -> Path:
    registry = read_registry(root)
    if registry.get("evaluation_frozen"):
        _checkpoint_map(root)
        return _checkpoint_allowlist_path(root)
    jobs = _all_training_complete(root)
    fairness = _freeze_initial_state_fairness(root, jobs)
    rows: list[dict[str, Any]] = []
    generator_rows: list[dict[str, Any]] = []
    for job in jobs:
        status = read_json(_status_path(root, str(job["job_id"])))
        for role in ("generator_best_learned", "discriminator_best_learned"):
            artifact = _artifact(status, role)
            row = {
                "job_id": str(job["job_id"]),
                "arm": str(job["arm"]),
                "fold": str(job["fold"]),
                "seed": int(job["seed"]),
                "tolerance_minutes": 5,
                "checkpoint_role": role,
                "checkpoint_path": str(Path(artifact["path"]).resolve()),
                "size_bytes": int(artifact["size_bytes"]),
                "checkpoint_sha256": str(artifact["sha256"]),
            }
            rows.append(row)
            if role == "generator_best_learned":
                generator_rows.append(
                    {
                        "job_id": row["job_id"],
                        "arm": row["arm"],
                        "fold": row["fold"],
                        "seed": row["seed"],
                        "tolerance_minutes": 5,
                        "checkpoint_path": row["checkpoint_path"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "size_bytes": row["size_bytes"],
                    }
                )
    if (
        len(rows) != EXPECTED_TRAINING_JOBS * 2
        or len(generator_rows) != EXPECTED_TRAINING_JOBS
    ):
        raise ValueError(
            f"Checkpoint universe is not exactly {EXPECTED_TRAINING_JOBS} G/D pairs"
        )
    allowlist = write_csv(_checkpoint_allowlist_path(root), rows, tuple(rows[0]))
    checkpoint_manifest = write_csv(
        _checkpoint_manifest_path(root), generator_rows, tuple(generator_rows[0])
    )
    inference = write_json(
        root / "evaluation/inference_determinism_contract.json", _inference_contract()
    )
    registry.update(
        status="evaluation_frozen",
        evaluation_frozen=True,
        evaluation_frozen_at_utc=utc_now(),
        checkpoint_allowlist_path=str(allowlist.resolve()),
        checkpoint_allowlist_sha256=sha256_file(allowlist),
        checkpoint_manifest_path=str(checkpoint_manifest.resolve()),
        checkpoint_manifest_sha256=sha256_file(checkpoint_manifest),
        initial_state_fairness_path=str(fairness.resolve()),
        initial_state_fairness_sha256=sha256_file(fairness),
        inference_determinism_contract_path=str(inference.resolve()),
        inference_determinism_contract_sha256=sha256_file(inference),
        test_data_opened=False,
    )
    write_registry(root, registry)
    _experiment_status(
        root,
        "evaluation_frozen",
        frozen_checkpoint_rows=EXPECTED_TRAINING_JOBS * 2,
    )
    _checkpoint_map(root)
    return allowlist


def _test_panel_path(root: Path, fold: str) -> Path:
    return root / "evaluation/test_panels/tolerance_05m" / f"{fold}.csv.gz"


def _test_overlay_path(root: Path, fold: str, arm: str) -> Path:
    return (
        root / "evaluation/test_pair_text_overlays/tolerance_05m" / fold / f"{arm}.json"
    )


def _test_input_manifest_path(root: Path) -> Path:
    return root / "evaluation/test_input_hashes.csv"


def _validate_test_inputs(root: Path) -> Path:
    registry = read_registry(root)
    if not registry.get("evaluation_frozen") or not registry.get("test_data_opened"):
        raise ValueError("Test inputs are not frozen")
    path = _verify_frozen_file(
        registry["test_input_manifest_path"], registry["test_input_manifest_sha256"]
    )
    rows = core.read_manifest(path)
    expected_roles = {
        *(f"test_panel:05m:{fold}" for fold in FOLDS),
        *(f"test_overlay:05m:{fold}:{arm}" for fold in FOLDS for arm in DIRECT_ARMS),
    }
    expected_input_files = len(FOLDS) * (1 + len(DIRECT_ARMS))
    if (
        len(rows) != expected_input_files
        or {str(row["artifact_role"]) for row in rows} != expected_roles
    ):
        raise ValueError(
            "Frozen test-input artifact universe must contain "
            f"{expected_input_files} files"
        )
    return path


def _materialize_test_inputs(config: Mapping[str, Any], root: Path) -> Path:
    registry = read_registry(root)
    if not registry.get("evaluation_frozen"):
        raise RuntimeError("Test inputs require a frozen checkpoint allowlist")
    existing = _test_input_manifest_path(root)
    if existing.is_file():
        if registry.get("test_data_opened"):
            return _validate_test_inputs(root)
        raise ValueError("Partial test-input freeze requires manual audit")

    import numpy as np
    from wgan_option.utils.news_first_experiment_core import (
        _fixed_partition_derangement,
        _l2,
        _normalized_article_key,
        _parsed_vector,
        write_pair_text_overlay_manifest,
    )

    data = _mapping(config["data"], "data")
    matrix = _mapping(config["matrix"], "matrix")
    universe = pd.read_csv(root / "inputs/pair_universes.csv", dtype=str)
    test_universe = universe.loc[
        universe["tolerance_minutes"].astype(int).eq(5)
        & universe["partition"].eq("test")
    ].copy()
    required_pairs = set(test_universe["pair_id"].astype(str))
    sentiment_frame = pd.read_excel(
        resolve_path(data["sentiment_workbook_path"]),
        sheet_name="features",
        usecols=["news_row_id", "sentiment_embedding"],
    )
    if sentiment_frame["news_row_id"].astype(str).duplicated().any():
        raise ValueError("Sentiment source contains duplicate news_row_id")
    raw_sentiment = dict(
        zip(
            sentiment_frame["news_row_id"].astype(str),
            sentiment_frame["sentiment_embedding"],
        )
    )
    workbook = resolve_path(data["root"]) / str(data["workbook_template"]).format(
        tolerance02="05"
    )
    frame = pd.read_excel(workbook, sheet_name=str(data["sheet_name"]))
    frame["pair_id"] = frame["pair_id"].astype(str)
    frame = frame.loc[frame["pair_id"].isin(required_pairs)].copy()
    if set(frame["pair_id"]) != required_pairs:
        raise ValueError("Test workbook coverage drift")
    rows_by_pair: dict[str, list[dict[str, Any]]] = {}
    canonical_rows: dict[str, dict[str, Any]] = {}
    for pair_id, pair_rows in frame.groupby("pair_id", sort=False):
        ordered = pair_rows.assign(
            _news_row_sort=pd.to_numeric(pair_rows["news_row_id"], errors="raise")
        ).sort_values(["_news_row_sort", "sample_id"], kind="stable")
        canonical_rows[str(pair_id)] = (
            ordered.iloc[0].drop(labels="_news_row_sort").to_dict()
        )
        articles: dict[str, dict[str, Any]] = {}
        for row in ordered.itertuples(index=False):
            article_key = _normalized_article_key(row)
            news_key = str(row.news_row_id)
            if news_key not in raw_sentiment:
                raise ValueError(f"Test sentiment missing news_row_id={news_key}")
            article = {
                "lp_text": core._canonical_article_text(row.lp_text),
                "lp_embedding": _parsed_vector(
                    row.lp_embedding,
                    dimension=1024,
                    label=f"test LP {pair_id}/{article_key}",
                ),
                "sentiment": _parsed_vector(
                    raw_sentiment[news_key],
                    dimension=1024,
                    label=f"test sentiment {news_key}",
                )[:3],
            }
            previous = articles.get(article_key)
            if previous is not None:
                if previous["lp_text"] != article["lp_text"]:
                    raise ValueError(
                        f"Test article text drift: {pair_id}/{article_key}"
                    )
                continue
            articles[article_key] = article
        if not articles:
            raise ValueError(f"No usable test articles: {pair_id}")
        rows_by_pair[str(pair_id)] = [articles[key] for key in sorted(articles)]
    lp_by_pair = {
        pair_id: _l2(
            np.mean(
                np.stack([article["lp_embedding"] for article in articles], axis=0),
                axis=0,
            )
        )
        for pair_id, articles in rows_by_pair.items()
    }
    sentiment_by_pair = {
        pair_id: np.mean(
            np.stack([article["sentiment"] for article in articles], axis=0), axis=0
        ).astype(np.float32)
        for pair_id, articles in rows_by_pair.items()
    }
    artifact_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        selected = test_universe.loc[test_universe["fold"].eq(fold)].copy()
        pair_ids = sorted(selected["pair_id"].astype(str))
        sessions = dict(
            zip(selected["pair_id"].astype(str), selected["session_id"].astype(str))
        )
        expected = _expected_counts(config, fold)
        if (
            len(pair_ids) != expected["test_pairs"]
            or len(set(sessions.values())) != expected["test_sessions"]
        ):
            raise ValueError(f"Frozen test counts drift: {fold}")
        panel = pd.DataFrame([canonical_rows[pair_id] for pair_id in pair_ids])
        panel["sample_id"] = panel["pair_id"].map(lambda value: f"pair::{value}")
        panel["sample_weight"] = 1.0
        panel_path = core._write_dataframe_csv(
            _test_panel_path(root, fold), panel, gzip=True
        )
        artifact_rows.append(manifest_row(f"test_panel:05m:{fold}", panel_path))
        for arm in DIRECT_ARMS:
            semantic_arm = {
                "film_lp_matched": "lp_matched",
                "film_lp_shuffle": "lp_shuffle",
                "film_bow": "bow",
                "film_sentiment": "sentiment",
            }.get(str(arm), str(arm))
            training_path = _overlay_path(root, fold, arm)
            training_payload = read_json(training_path)
            transform = {
                **dict(training_payload.get("transform") or {}),
                "evaluation_partition": "test",
                "training_overlay_path": str(training_path),
                "training_overlay_sha256": sha256_file(training_path),
            }
            donor_by_pair: dict[str, str] = {}
            if arm == NO_TEXT_ARM:
                vectors = {
                    pair_id: np.zeros(1024, dtype=np.float32) for pair_id in pair_ids
                }
                transform["method"] = "zero_vector_v1"
            elif semantic_arm == "lp_matched":
                vectors = {pair_id: lp_by_pair[pair_id] for pair_id in pair_ids}
                transform["method"] = "unique_article_lp_mean_l2_v1"
            elif semantic_arm == "lp_shuffle":
                donor_by_pair = _fixed_partition_derangement(
                    pair_ids,
                    master_seed=int(matrix["shuffle_seed"]),
                    namespace=f"evaluation/test/05m/{fold}",
                )
                vectors = {
                    pair_id: lp_by_pair[donor_by_pair[pair_id]] for pair_id in pair_ids
                }
                transform.update(
                    method="test_partition_pair_derangement_v1",
                    master_seed=int(matrix["shuffle_seed"]),
                    mapping_sha256=payload_sha256(sorted(donor_by_pair.items())),
                )
            elif semantic_arm == "bow":
                vocabulary = list(transform.get("vocabulary") or [])
                if not vocabulary or len(vocabulary) > 1024:
                    raise ValueError("Frozen train-only BoW vocabulary is invalid")
                vectors = {
                    pair_id: core._bow_vector_from_article_texts(
                        [article["lp_text"] for article in rows_by_pair[pair_id]],
                        vocabulary,
                    )
                    for pair_id in pair_ids
                }
            elif semantic_arm == "sentiment":
                mean = np.asarray(transform.get("train_mean"), dtype=np.float32)
                std = np.asarray(transform.get("train_std"), dtype=np.float32)
                if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
                    raise ValueError("Frozen train-only sentiment transform is invalid")
                vectors = {}
                for pair_id in pair_ids:
                    vector = np.zeros(1024, dtype=np.float32)
                    vector[:3] = (sentiment_by_pair[pair_id] - mean) / std
                    vectors[pair_id] = vector
            else:  # pragma: no cover
                raise AssertionError(arm)
            records = [
                {
                    "pair_id": pair_id,
                    "session_id": sessions[pair_id],
                    "embedding": vectors[pair_id],
                    **(
                        {"donor_pair_id": donor_by_pair[pair_id]}
                        if semantic_arm == "lp_shuffle"
                        else {}
                    ),
                }
                for pair_id in pair_ids
            ]
            overlay = _test_overlay_path(root, fold, arm)
            write_pair_text_overlay_manifest(
                overlay,
                mode=_overlay_mode(arm),
                namespace=f"evaluation/test/05m/{fold}/{arm}",
                records=records,
                transform=transform,
            )
            artifact_rows.append(
                manifest_row(f"test_overlay:05m:{fold}:{arm}", overlay)
            )
    expected_input_files = len(FOLDS) * (1 + len(DIRECT_ARMS))
    if len(artifact_rows) != expected_input_files:
        raise ValueError(
            f"Expected {expected_input_files} frozen test inputs, "
            f"got {len(artifact_rows)}"
        )
    result = _write_hash_manifest(_test_input_manifest_path(root), artifact_rows)
    registry = read_registry(root)
    registry.update(
        status="evaluation_inputs_frozen",
        test_data_opened=True,
        test_data_opened_at_utc=utc_now(),
        test_input_manifest_path=str(result.resolve()),
        test_input_manifest_sha256=sha256_file(result),
    )
    write_registry(root, registry)
    _experiment_status(
        root,
        "evaluation_inputs_frozen",
        test_panels=len(FOLDS),
        test_overlays=EXPECTED_PREDICTION_CELLS,
    )
    return _validate_test_inputs(root)


def freeze_evaluation(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    config = validate_root(root)
    _freeze_checkpoints(root)
    _materialize_test_inputs(config, root)
    return root


def _prediction_path(root: Path, job: Mapping[str, Any]) -> Path:
    return (
        root
        / "evaluation/predictions/tolerance_05m"
        / str(job["fold"])
        / f"seed_{SEED}"
        / f"{job['arm']}.csv.gz"
    )


def _prediction_manifest_path(root: Path, job: Mapping[str, Any]) -> Path:
    return _prediction_path(root, job).with_suffix(".manifest.json")


def _validate_prediction_cell(
    root: Path,
    job: Mapping[str, Any],
    checkpoint: Mapping[str, str],
    *,
    expected_pairs: int,
) -> dict[str, Any]:
    with direct_profile():
        return core._validate_prediction_job(
            root, job, checkpoint, expected_pairs=expected_pairs
        )


def _validate_predictions(root: Path) -> tuple[Path, Path]:
    config = validate_root(root)
    registry = read_registry(root)
    if not registry.get("predictions_frozen"):
        raise ValueError("Predictions are not frozen")
    manifest_path = _verify_frozen_file(
        registry["prediction_manifest_path"], registry["prediction_manifest_sha256"]
    )
    pair_path = _verify_frozen_file(
        registry["pair_metrics_path"], registry["pair_metrics_sha256"]
    )
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    evidence = pd.read_csv(pair_path)
    if (
        len(manifest) != EXPECTED_PREDICTION_CELLS
        or manifest["job_id"].duplicated().any()
    ):
        raise ValueError(
            "Prediction manifest must contain exactly "
            f"{EXPECTED_PREDICTION_CELLS} cells"
        )
    if len(evidence) != EXPECTED_PAIR_METRIC_ROWS:
        raise ValueError("Global pair metrics must contain exactly 2,500 rows")
    checkpoints = _checkpoint_map(root)
    jobs = [dict(job) for job in registry["jobs"]]
    profiles: dict[str, set[str]] = {fold: set() for fold in FOLDS}
    for job in jobs:
        expected_pairs = _expected_counts(config, str(job["fold"]))["test_pairs"]
        payload = _validate_prediction_cell(
            root,
            job,
            checkpoints[str(job["job_id"])],
            expected_pairs=expected_pairs,
        )
        profiles[str(job["fold"])].add(str(payload["noise_bank_profile_sha256"]))
        cell = evidence.loc[evidence["job_id"].astype(str).eq(str(job["job_id"]))]
        if len(cell) != expected_pairs:
            raise ValueError(f"Global pair-evidence count drift: {job['job_id']}")
    if any(len(values) != 1 for values in profiles.values()):
        raise ValueError("Arms within a fold do not share one MC64 noise bank")
    expected_cells = {(fold, arm) for fold in FOLDS for arm in DIRECT_ARMS}
    if (
        set(
            evidence[["fold", "arm"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        != expected_cells
    ):
        raise ValueError("Pair-evidence fold/arm universe drift")
    return manifest_path, pair_path


def predict(output_dir: str | Path, *, resume: bool = False) -> Path:
    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise RuntimeError("Prediction requires a frozen checkpoint allowlist root")
    try:
        registry = read_registry(root)
    except FileNotFoundError as exc:
        raise RuntimeError("Prediction requires a frozen checkpoint allowlist") from exc
    if not registry.get("evaluation_frozen"):
        raise RuntimeError("Prediction requires a frozen checkpoint allowlist")
    freeze_evaluation(root)
    config = validate_root(root)
    registry = read_registry(root)
    if registry.get("predictions_frozen"):
        _validate_predictions(root)
        return root
    checkpoints = _checkpoint_map(root)
    _validate_test_inputs(root)
    with direct_profile():
        core._configure_prediction_determinism(SEED)
    prediction_rows: list[dict[str, Any]] = []
    evidence_frames: list[pd.DataFrame] = []
    jobs = sorted(
        (dict(job) for job in registry["jobs"]),
        key=lambda job: (str(job["fold"]), str(job["arm"])),
    )
    for job in jobs:
        checkpoint = checkpoints[str(job["job_id"])]
        expected_pairs = _expected_counts(config, str(job["fold"]))["test_pairs"]
        path = _prediction_path(root, job)
        manifest_path = _prediction_manifest_path(root, job)
        evidence_path = manifest_path.with_suffix(".pair_metrics.csv")
        if path.is_file() and manifest_path.is_file() and evidence_path.is_file():
            payload = _validate_prediction_cell(
                root, job, checkpoint, expected_pairs=expected_pairs
            )
            evidence = pd.read_csv(evidence_path)
        else:
            if (
                any(value.is_file() for value in (path, manifest_path, evidence_path))
                and not resume
            ):
                raise ValueError(
                    f"Partial prediction requires --resume: {job['job_id']}"
                )
            with direct_profile():
                payload, evidence = core._evaluate_prediction_job(
                    root, job, checkpoint, expected_pairs=expected_pairs
                )
            core._write_dataframe_csv(evidence_path, evidence)
            payload = read_json(manifest_path)
            payload.pop("payload_sha256", None)
            payload.update(
                pair_metrics_path=str(evidence_path.resolve()),
                pair_metrics_sha256=sha256_file(evidence_path),
                pair_metrics_row_count=int(len(evidence)),
            )
            payload["payload_sha256"] = payload_sha256(payload)
            write_json(manifest_path, payload)
        payload = _validate_prediction_cell(
            root, job, checkpoint, expected_pairs=expected_pairs
        )
        prediction_rows.append(
            {
                "job_id": str(job["job_id"]),
                "arm": str(job["arm"]),
                "fold": str(job["fold"]),
                "seed": SEED,
                "tolerance_minutes": 5,
                "prediction_path": str(path.resolve()),
                "prediction_sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "prediction_manifest_path": str(manifest_path.resolve()),
                "prediction_manifest_sha256": sha256_file(manifest_path),
                "noise_bank_profile_sha256": payload["noise_bank_profile_sha256"],
                "inference_determinism_contract_sha256": payload[
                    "inference_determinism_contract_sha256"
                ],
            }
        )
        evidence_frames.append(evidence)
    if (
        len(prediction_rows) != EXPECTED_PREDICTION_CELLS
        or len(evidence_frames) != EXPECTED_PREDICTION_CELLS
    ):
        raise ValueError(
            f"Prediction stage did not produce {EXPECTED_PREDICTION_CELLS} cells"
        )
    prediction_manifest = write_csv(
        root / "analysis/prediction_manifest.csv",
        prediction_rows,
        tuple(prediction_rows[0]),
    )
    evidence = pd.concat(evidence_frames, ignore_index=True)
    pair_metrics = core._write_dataframe_csv(
        root / "analysis/rq12_pair_metrics.csv.gz", evidence, gzip=True
    )
    if len(evidence) != EXPECTED_PAIR_METRIC_ROWS:
        raise ValueError(f"Expected 2,500 pair rows, got {len(evidence)}")
    registry = read_registry(root)
    registry.update(
        status="predictions_frozen",
        predictions_frozen=True,
        predictions_frozen_at_utc=utc_now(),
        prediction_manifest_path=str(prediction_manifest.resolve()),
        prediction_manifest_sha256=sha256_file(prediction_manifest),
        pair_metrics_path=str(pair_metrics.resolve()),
        pair_metrics_sha256=sha256_file(pair_metrics),
        prediction_cell_count=EXPECTED_PREDICTION_CELLS,
        pair_metric_row_count=EXPECTED_PAIR_METRIC_ROWS,
    )
    write_registry(root, registry)
    _experiment_status(
        root,
        "predictions_frozen",
        prediction_cells=EXPECTED_PREDICTION_CELLS,
        pair_rows=EXPECTED_PAIR_METRIC_ROWS,
    )
    _validate_predictions(root)
    return root


def _training_summary(root: Path) -> Path:
    destination = root / "analysis/training_summary.csv"
    if destination.is_file():
        return destination
    checkpoints = _checkpoint_map(root)
    rows: list[dict[str, Any]] = []
    for job in read_registry(root)["jobs"]:
        status = read_json(_status_path(root, str(job["job_id"])))
        best_path = Path(_artifact(status, "best_learned_checkpoint")["path"])
        best = read_json(best_path)
        metrics = pd.read_csv(_artifact(status, "training_metrics_csv")["path"])
        epochs = pd.to_numeric(metrics["epoch"], errors="raise").astype(int)
        learned = epochs[epochs >= 1]
        if learned.empty:
            raise ValueError(f"Training metrics have no learned epoch: {job['job_id']}")
        g_trace = list(best.get("generator_lr_trace") or [])
        d_trace = list(best.get("discriminator_lr_trace") or [])
        if not g_trace or not d_trace:
            lr_trace = list(best.get("lr_trace") or [])
            g_trace = [
                {"epoch": row["epoch"], "lr": row.get("g_lr", row.get("lr"))}
                for row in lr_trace
            ]
            d_trace = [
                {"epoch": row["epoch"], "lr": row.get("d_lr", row.get("lr"))}
                for row in lr_trace
            ]
        rows.append(
            {
                "job_id": str(job["job_id"]),
                "fold": str(job["fold"]),
                "arm": str(job["arm"]),
                "best_epoch": int(best["best_epoch"]),
                "epochs_ran": int(learned.max()),
                "final_generator_lr": float(g_trace[-1]["lr"]),
                "final_discriminator_lr": float(d_trace[-1]["lr"]),
                "best_validation_score": float(best["best_metric"]),
                "checkpoint_sha256": checkpoints[str(job["job_id"])]["sha256"],
                "monitor_metric": str(best.get("monitor_metric", "val_hybrid_score")),
                "early_stopped": int(learned.max())
                < int(read_registry(root)["training_num_epochs"]),
            }
        )
    if len(rows) != EXPECTED_TRAINING_JOBS:
        raise ValueError(f"Training summary requires {EXPECTED_TRAINING_JOBS} cells")
    return write_csv(destination, rows, tuple(rows[0]))


def _historical_reference(config: Mapping[str, Any], root: Path) -> Path:
    """Freeze the old seed-42 chain as a clearly non-compute-matched appendix."""

    historical = _mapping(
        config["analysis"]["historical_reference"], "historical reference"
    )
    source_root = resolve_path(historical["experiment_root"])
    source = source_root / "analysis/rq123_pair_metrics.csv.gz"
    source_sha = sha256_file(source)
    frame = pd.read_csv(source)
    selected = frame.loc[
        pd.to_numeric(frame["seed"], errors="raise")
        .astype(int)
        .eq(int(historical["seed"]))
        & pd.to_numeric(frame["tolerance_minutes"], errors="raise")
        .astype(int)
        .eq(int(historical["tolerance_minutes"]))
        & frame["arm"].astype(str).isin(list(historical["arms"]))
    ].copy()
    if len(selected) != 1_000:
        raise ValueError("Historical seed-42 reference must contain 2 arms x 500 pairs")
    rows: list[dict[str, Any]] = []
    for arm, group in selected.groupby("arm", sort=True):
        fold_means = group.groupby("fold", sort=True).agg(
            model_mae=("target_mae", "mean"),
            persistence_mae=("persistence_mae", "mean"),
        )
        if set(fold_means.index.astype(str)) != set(FOLDS):
            raise ValueError(f"Historical fold universe drift: {arm}")
        pooled_model = float(pd.to_numeric(group["target_mae"], errors="raise").mean())
        pooled_persistence = float(
            pd.to_numeric(group["persistence_mae"], errors="raise").mean()
        )
        equal_model = float(fold_means["model_mae"].mean())
        equal_persistence = float(fold_means["persistence_mae"].mean())
        rows.append(
            {
                "historical_arm": str(arm),
                "seed": SEED,
                "pooled_mae": pooled_model,
                "equal_fold_mae": equal_model,
                "pooled_improvement_vs_persistence_percent": 100.0
                * (1.0 - pooled_model / pooled_persistence),
                "equal_fold_improvement_vs_persistence_percent": 100.0
                * (1.0 - equal_model / equal_persistence),
                "pair_count": int(len(group)),
                "fold_count": int(len(fold_means)),
                "historical_epoch_budget": str(historical["historical_epoch_budget"]),
                "direct_epoch_budget": int(historical["direct_epoch_budget"]),
                "usage": str(historical["usage"]),
                "strict_compute_matched_comparison": False,
                "source_pair_metrics_sha256": source_sha,
            }
        )
    destination = write_csv(
        root / "analysis/historical_seed42_parent_chain_reference.csv",
        rows,
        tuple(rows[0]),
    )
    markdown_lines = [
        "# Historical seed-42 parent-chain reference",
        "",
        "This appendix is descriptive only and is not a compute-matched comparison: "
        "the historical chain had a 30-epoch parent plus up to 240 continuation "
        "epochs, while each new direct arm has at most 240 total epochs.",
        "",
        "| arm | pooled MAE | equal-fold MAE | equal-fold improvement vs persistence |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        markdown_lines.append(
            f"| {row['historical_arm']} | {row['pooled_mae']:.10g} | "
            f"{row['equal_fold_mae']:.10g} | "
            f"{row['equal_fold_improvement_vs_persistence_percent']:.6g}% |"
        )
    markdown = "\n".join(markdown_lines) + "\n"
    markdown_path = root / "report/historical_seed42_parent_chain_reference.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    table = pd.DataFrame(rows)[
        [
            "historical_arm",
            "pooled_mae",
            "equal_fold_mae",
            "equal_fold_improvement_vs_persistence_percent",
            "usage",
        ]
    ].to_html(index=False, border=0)
    html_path = root / "report/historical_seed42_parent_chain_reference.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Historical "
        "seed-42 reference</title></head><body><h1>Historical seed-42 "
        "parent-chain reference</h1><p>"
        + html.escape(
            "Descriptive only; not compute-matched. Historical budget: parent30 + "
            "continuation240. New direct budget: 240 total epochs."
        )
        + f"</p>{table}</body></html>\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "kind": "direct_5arm_historical_reference_manifest_v1",
        "usage": str(historical["usage"]),
        "strict_compute_matched_comparison": False,
        "source_pair_metrics_path": str(source.resolve()),
        "source_pair_metrics_sha256": source_sha,
        "artifacts": [
            manifest_row("historical_reference_csv", destination),
            manifest_row("historical_reference_markdown", markdown_path),
            manifest_row("historical_reference_html", html_path),
        ],
    }
    manifest["payload_sha256"] = payload_sha256(manifest)
    return write_json(root / "analysis/historical_reference_manifest.json", manifest)


def _resource_summary(root: Path) -> Path:
    destination = root / "resource_summary.csv"
    if destination.is_file():
        return destination
    frame = pd.read_csv(root / "resource_usage.csv")
    rows: list[dict[str, Any]] = []
    for gpu, group in frame.groupby("gpu_index", sort=True):
        rows.append(
            {
                "gpu_index": int(gpu),
                "telemetry_rows": int(len(group)),
                "peak_memory_used_mib": float(
                    pd.to_numeric(group["memory_used_mib"], errors="raise").max()
                ),
            }
        )
    if {row["gpu_index"] for row in rows} != {0, 1}:
        raise ValueError("Resource summary requires both A30 GPUs")
    return write_csv(destination, rows, tuple(rows[0]))


def _terminal_files(root: Path) -> list[Path]:
    excluded = {(root / "output_hashes.csv").resolve()}
    return sorted(
        (
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.resolve() not in excluded
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _validate_output_hashes(root: Path) -> Path:
    path = root / "output_hashes.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = {item.relative_to(root).as_posix() for item in _terminal_files(root)}
    if (
        set(frame["relative_path"]) != expected
        or frame["relative_path"].duplicated().any()
    ):
        raise ValueError("Terminal output manifest universe drift")
    for row in frame.itertuples(index=False):
        target = root / row.relative_path
        if (
            str(target.resolve()) != row.path
            or target.stat().st_size != int(row.size_bytes)
            or sha256_file(target) != row.sha256
        ):
            raise ValueError(f"Terminal output hash drift: {target}")
    return path


def qa(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    config = validate_root(root)
    registry = read_registry(root)
    if registry.get("terminal_complete"):
        _validate_output_hashes(root)
        return root / "qa.json"
    _all_training_complete(root)
    _checkpoint_map(root)
    _validate_test_inputs(root)
    _validate_predictions(root)
    analysis_manifest = _verify_frozen_file(
        registry["analysis_manifest_path"], registry["analysis_manifest_sha256"]
    )
    if analysis_manifest != (root / "analysis/analysis_manifest.json").resolve():
        raise ValueError("Analysis manifest canonical path drift")
    historical_manifest = _verify_frozen_file(
        registry["historical_reference_manifest_path"],
        registry["historical_reference_manifest_sha256"],
    )
    historical_payload = read_json(historical_manifest)
    historical_unsigned = {
        key: value
        for key, value in historical_payload.items()
        if key != "payload_sha256"
    }
    if (
        historical_payload.get("kind") != "direct_5arm_historical_reference_manifest_v1"
        or payload_sha256(historical_unsigned)
        != historical_payload.get("payload_sha256")
        or bool(historical_payload.get("strict_compute_matched_comparison"))
    ):
        raise ValueError("Historical reference manifest drift")
    for artifact in historical_payload.get("artifacts") or []:
        verify_manifest_rows([artifact])
    training = pd.read_csv(root / "analysis/training_summary.csv")
    pairs = pd.read_csv(root / "analysis/rq12_pair_metrics.csv.gz")
    if len(training) != 20 or len(pairs) != 2500:
        raise ValueError("Terminal evidence count drift")
    fairness = _verify_frozen_file(
        registry["initial_state_fairness_path"],
        registry["initial_state_fairness_sha256"],
    )
    if read_json(fairness).get("job_count") != 20:
        raise ValueError("Initial-state fairness evidence drift")
    qa_payload = {
        "schema_version": 1,
        "kind": "direct_5arm_terminal_qa_v1",
        "status": "passed",
        "interpretation": INTERPRETATION,
        "training_jobs_completed": 20,
        "prediction_cells": 20,
        "pair_metric_rows": 2500,
        "parent_jobs": 0,
        "continuation_jobs": 0,
        "branch_recipes": 0,
        "test_opened_after_checkpoint_freeze": True,
        "shared_mc64_noise_bank_within_fold": True,
        "generator_parameters": EXPECTED_PARAMETER_COUNTS["generator"],
        "critic_parameters": EXPECTED_PARAMETER_COUNTS["critic"],
        "source_config_sha256": config["source_config_sha256"],
        "completed_at_utc": utc_now(),
    }
    qa_payload["payload_sha256"] = payload_sha256(qa_payload)
    qa_path = write_json(root / "qa.json", qa_payload)
    registry = read_registry(root)
    registry.update(
        status="completed", terminal_complete=True, completed_at_utc=utc_now()
    )
    write_registry(root, registry)
    _experiment_status(root, "completed", current_stage="terminal")
    files = _terminal_files(root)
    rows = [
        {
            "artifact_role": f"output:{path.relative_to(root).as_posix()}",
            "relative_path": path.relative_to(root).as_posix(),
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    write_csv(root / "output_hashes.csv", rows, tuple(rows[0]))
    _validate_output_hashes(root)
    return qa_path


def postprocess(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    config = validate_root(root)
    registry = read_registry(root)
    if registry.get("terminal_complete"):
        return qa(root)
    _validate_predictions(root)
    training_summary = _training_summary(root)
    historical_manifest = _historical_reference(config, root)
    analysis_manifest = analyze_experiment(root, config)
    _resource_summary(root)
    registry = read_registry(root)
    registry.update(
        status="analysis_complete",
        analysis_complete=True,
        analysis_completed_at_utc=utc_now(),
        training_summary_path=str(training_summary.resolve()),
        training_summary_sha256=sha256_file(training_summary),
        historical_reference_manifest_path=str(historical_manifest.resolve()),
        historical_reference_manifest_sha256=sha256_file(historical_manifest),
        analysis_manifest_path=str(analysis_manifest.resolve()),
        analysis_manifest_sha256=sha256_file(analysis_manifest),
    )
    write_registry(root, registry)
    _experiment_status(
        root, "analysis_complete", analysis_manifest=str(analysis_manifest)
    )
    return qa(root)


def status(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    if not root.exists():
        return {"status": "absent", "root": str(root)}
    try:
        registry = read_registry(root)
    except (FileNotFoundError, ValueError):
        return {"status": "invalid_partial_root", "root": str(root)}
    counts: dict[str, int] = {}
    for job in registry.get("jobs") or []:
        status_path = _status_path(root, str(job["job_id"]))
        value = (
            read_json(status_path).get("status", "missing")
            if status_path.is_file()
            else "missing"
        )
        counts[str(value)] = counts.get(str(value), 0) + 1
    return {
        "status": registry.get("status"),
        "root": str(root),
        "registered_jobs": len(registry.get("jobs") or []),
        "expected_jobs": EXPECTED_TRAINING_JOBS,
        "job_status_counts": counts,
        "evaluation_frozen": bool(registry.get("evaluation_frozen")),
        "test_data_opened": bool(registry.get("test_data_opened")),
        "predictions_frozen": bool(registry.get("predictions_frozen")),
        "analysis_complete": bool(registry.get("analysis_complete")),
        "terminal_complete": bool(registry.get("terminal_complete")),
    }


_PIPELINE_LOCKS: dict[Path, int] = {}


def _pipeline_lock(root: Path) -> tuple[Path, int]:
    control = root.with_name(root.name + "_control")
    control.mkdir(parents=True, exist_ok=True)
    path = (control / "pipeline.lock").resolve()
    descriptor = _acquire_lock(path)
    _PIPELINE_LOCKS[path] = descriptor
    (control / "pipeline.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    return path, descriptor


def _pipeline_journal(
    root: Path, stage: str, status_value: str, **details: Any
) -> Path:
    return write_json(
        root.with_name(root.name + "_control") / "pipeline_journal.json",
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "stage": stage,
            "status": status_value,
            "pid": os.getpid(),
            "updated_at_utc": utc_now(),
            **details,
        },
    )


def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    root = Path(output_dir).resolve()
    if root.is_dir():
        try:
            registry = read_registry(root)
        except (FileNotFoundError, ValueError):
            registry = {}
        if registry.get("terminal_complete"):
            validate_root(root)
            _validate_output_hashes(root)
            return root
    lock_path, descriptor = _pipeline_lock(root)
    del lock_path
    try:
        _pipeline_journal(root, "benchmark", "running")
        benchmark(config_path, root, resume=resume)
        _pipeline_journal(root, "prepare", "running")
        prepare_experiment(config_path, root, resume=True)
        registry = read_registry(root)
        if registry.get("status") == "prepared":
            _pipeline_journal(
                root,
                "direct_arms",
                "running",
                completed=status(root)["job_status_counts"].get("completed", 0),
            )
            launch(root, resume=True)
        if not read_registry(root).get("evaluation_frozen"):
            _pipeline_journal(root, "freeze_evaluation", "running")
            freeze_evaluation(root, resume=True)
        if not read_registry(root).get("predictions_frozen"):
            _pipeline_journal(root, "predict", "running")
            predict(root, resume=True)
        _pipeline_journal(root, "postprocess", "running")
        postprocess(root, resume=True)
        _pipeline_journal(root, "terminal", "completed")
        return root
    except BaseException as exc:
        _pipeline_journal(
            root, "pipeline", "failed", error=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        _PIPELINE_LOCKS.pop(
            root.with_name(root.name + "_control") / "pipeline.lock", None
        )
        _release_lock(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "benchmark",
            "prepare",
            "dry-run",
            "worker",
            "launch",
            "freeze-evaluation",
            "predict",
            "postprocess",
            "qa",
            "status",
            "run-pipeline",
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    action = str(args.action)
    if action == "benchmark":
        result: Any = benchmark(args.config, args.output_dir, resume=args.resume)
    elif action == "prepare":
        result = prepare_experiment(args.config, args.output_dir, resume=args.resume)
    elif action == "dry-run":
        result = dry_run(args.config, args.output_dir, resume=args.resume)
    elif action == "worker":
        if not args.job_id:
            raise SystemExit("worker requires --job-id")
        result = worker(
            args.output_dir,
            args.job_id,
            resume=args.resume,
            dry_run=args.worker_dry_run,
        )
    elif action == "launch":
        result = launch(args.output_dir, resume=args.resume)
    elif action == "freeze-evaluation":
        result = freeze_evaluation(args.output_dir, resume=args.resume)
    elif action == "predict":
        result = predict(args.output_dir, resume=args.resume)
    elif action == "postprocess":
        result = postprocess(args.output_dir, resume=args.resume)
    elif action == "qa":
        result = qa(args.output_dir, resume=args.resume)
    elif action == "status":
        result = status(args.output_dir)
    else:
        result = run_pipeline(args.config, args.output_dir, resume=args.resume)
    if isinstance(result, Mapping):
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DIRECT_ARMS",
    "DIRECT_STAGE",
    "EXPECTED_PAIR_METRIC_ROWS",
    "EXPECTED_PREDICTION_CELLS",
    "EXPECTED_TRAINING_JOBS",
    "FOLDS",
    "INTERPRETATION",
    "SEED",
    "analyze_experiment",
    "benchmark",
    "fold_session_paired_bootstrap",
    "freeze_evaluation",
    "holm_adjust",
    "load_config",
    "planned_specs",
    "predict",
    "prepare",
    "run_pipeline",
    "sha256_file",
    "validate_config",
    "validate_gpu_balance",
]
