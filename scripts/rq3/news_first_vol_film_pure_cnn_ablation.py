"""Run the Q3-only FiLM dense-to-CNN architecture ablation.

The experiment is intentionally isolated from earlier capacity and conditioning
sweeps.  It owns an immutable 7-arm x 3-seed registry, never materializes a Q4
loader, and ranks arms only on the frozen Q3 validation panel.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

import torch
import yaml

import scripts._path_setup  # noqa: F401
from scripts.rq3 import news_first_vol_training as training
from wgan_option.models.common import (
    critic_conditioning_fingerprint,
    generator_conditioning_fingerprint,
)
from wgan_option.models.discriminator import Discriminator
from wgan_option.models.generator import Generator


EXPERIMENT_KIND = "film_pure_cnn_ablation_3seed_q3_development_v1"
DEFAULT_CONFIG = "configs/rq3/news_first_vol_film_pure_cnn_ablation_3seed.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/rq3_news_first_vol_film_pure_cnn_ablation_3seed_exact_ttm_v1"
)
DEFAULT_BENCHMARK_DIR = (
    "outputs/benchmarks/rq3_news_first_vol_film_pure_cnn_ablation_3seed_epoch1_v1"
)
BENCHMARK_EXPERIMENT_KIND = "film_pure_cnn_ablation_3seed_epoch1_benchmark_v1"
SEEDS = (42, 202, 404)
GPU_IDS = (0, 1)
LEARNING_RATE = 5e-7
GRID_SHA256 = "7b72f2d3d12a55999415351ba071f8d87863ed57445f1bbf03543e39f9f186b8"
GENERATOR_FILM_DENSE_CONCAT = "film_conv_bottleneck_concat_v1"
GENERATOR_FILM_DENSE_NO_CONCAT = "film_conv_no_bottleneck_concat_v1"
GENERATOR_FILM_UNET = "film_unet_v1"
GENERATOR_FILM_UNET_MASK_COORDS = "film_unet_mask_coords_v1"
CRITIC_LP_CONCAT = "lp_concat_v1"
CRITIC_NOLP_SAME_SHAPE = "lp_disabled_same_shape_v1"
CRITIC_LP_PROJECTION = "lp_projection_v1"

ARM_CONTRACTS: dict[str, dict[str, Any]] = {
    "baseline": {
        "generator_conditioning_mode": GENERATOR_FILM_DENSE_CONCAT,
        "critic_conditioning_mode": CRITIC_LP_CONCAT,
        "gen_text_hidden_dim": 256,
        "gen_text_out_dim": 128,
    },
    "film_dense_no_text_concat": {
        "generator_conditioning_mode": GENERATOR_FILM_DENSE_NO_CONCAT,
        "critic_conditioning_mode": CRITIC_LP_CONCAT,
        "gen_text_hidden_dim": 256,
        "gen_text_out_dim": 128,
    },
    "film_unet_text128": {
        "generator_conditioning_mode": GENERATOR_FILM_UNET,
        "critic_conditioning_mode": CRITIC_LP_CONCAT,
        "gen_text_hidden_dim": 256,
        "gen_text_out_dim": 128,
    },
    "film_unet_mask_coords_text128": {
        "generator_conditioning_mode": GENERATOR_FILM_UNET_MASK_COORDS,
        "critic_conditioning_mode": CRITIC_LP_CONCAT,
        "gen_text_hidden_dim": 256,
        "gen_text_out_dim": 128,
    },
    "film_unet_mask_coords_text64": {
        "generator_conditioning_mode": GENERATOR_FILM_UNET_MASK_COORDS,
        "critic_conditioning_mode": CRITIC_LP_CONCAT,
        "gen_text_hidden_dim": 64,
        "gen_text_out_dim": 64,
    },
    "film_unet_mask_coords_text64_nolp": {
        "generator_conditioning_mode": GENERATOR_FILM_UNET_MASK_COORDS,
        "critic_conditioning_mode": CRITIC_NOLP_SAME_SHAPE,
        "gen_text_hidden_dim": 64,
        "gen_text_out_dim": 64,
    },
    "film_unet_mask_coords_text64_projection": {
        "generator_conditioning_mode": GENERATOR_FILM_UNET_MASK_COORDS,
        "critic_conditioning_mode": CRITIC_LP_PROJECTION,
        "gen_text_hidden_dim": 64,
        "gen_text_out_dim": 64,
    },
}
ARM_IDS = tuple(ARM_CONTRACTS)
EXPECTED_JOB_COUNT = len(ARM_IDS) * len(SEEDS)

LEGACY_PROFILE = {
    "gen_base_channels": 32,
    "gen_res_blocks": 0,
    "gen_hidden_dim": 1024,
    "disc_base_channels": 32,
    "disc_res_blocks": 0,
    "disc_text_hidden_dim": 128,
    "disc_hidden_dim": 786,
}
REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_PATH = Path(__file__).resolve()
SOURCE_PATHS = (
    ORCHESTRATOR_PATH,
    REPO_ROOT / "src/wgan_option/models/common.py",
    REPO_ROOT / "src/wgan_option/models/generator.py",
    REPO_ROOT / "src/wgan_option/models/discriminator.py",
    REPO_ROOT / "src/wgan_option/models/gan_model.py",
    REPO_ROOT / "src/wgan_option/config.py",
    REPO_ROOT / "scripts/rq3/news_first_vol_training.py",
)

_payload_sha256 = training._payload_sha256
_sha256_file = training._sha256_file
_utc_now = training._utc_now
_write_json = training._write_json
_write_yaml = training._write_yaml
_atomic_write_text = training._atomic_write_text


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _read_json(path: Path) -> dict[str, Any]:
    return _require_mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _validate_resolved(resolved: Mapping[str, Any]) -> None:
    if int(resolved.get("schema_version", -1)) != 1:
        raise ValueError("Sweep schema_version must be 1")
    if str(resolved.get("experiment_kind")) != EXPERIMENT_KIND:
        raise ValueError(f"experiment_kind must be {EXPERIMENT_KIND!r}")
    if tuple(map(int, resolved.get("seeds", ()))) != SEEDS:
        raise ValueError(f"seeds must be {SEEDS}")
    if not math.isclose(float(resolved.get("learning_rate", 0.0)), LEARNING_RATE):
        raise ValueError("learning_rate must be 5e-7")
    if int(resolved.get("max_epochs", -1)) != 30:
        raise ValueError("max_epochs must be 30")
    if int(resolved.get("lr_warmup_epochs", -1)) != 0:
        raise ValueError("lr_warmup_epochs must be 0")
    if int(resolved.get("early_stopping_min_epochs", -1)) != 15:
        raise ValueError("early_stopping_min_epochs must be 15")
    if int(resolved.get("early_stopping_patience", -1)) != 16:
        raise ValueError("early_stopping_patience must be 16")
    if int(resolved.get("validation_mc_samples", -1)) != 16:
        raise ValueError("validation_mc_samples must be 16")

    arms = _require_mapping(resolved.get("arms"), "arms")
    if tuple(arms) != ARM_IDS:
        raise ValueError(f"arm order must be {ARM_IDS}")
    for arm_id, expected in ARM_CONTRACTS.items():
        arm = _require_mapping(arms.get(arm_id), f"arms.{arm_id}")
        if not str(arm.get("label", "")).strip():
            raise ValueError(f"arms.{arm_id}.label must be non-empty")
        for key, expected_value in expected.items():
            actual = arm.get(key)
            if isinstance(expected_value, int):
                actual = int(actual)
            else:
                actual = str(actual)
            if actual != expected_value:
                raise ValueError(
                    f"arms.{arm_id}.{key} must be {expected_value!r}, got {actual!r}"
                )

    profile = _require_mapping(resolved.get("legacy_profile"), "legacy_profile")
    if set(profile) != set(LEGACY_PROFILE):
        raise ValueError("legacy_profile fields drifted")
    for key, expected in LEGACY_PROFILE.items():
        if int(profile.get(key, -1)) != expected:
            raise ValueError(f"legacy_profile.{key} must be {expected}")

    runtime = _require_mapping(resolved.get("runtime"), "runtime")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != GPU_IDS:
        raise ValueError("runtime.gpu_ids must be [0, 1]")
    if int(runtime.get("workers_per_gpu", -1)) != 11:
        raise ValueError("workers_per_gpu must be 11 for the frozen single wave")
    if int(runtime.get("cpu_threads_per_job", -1)) != 1:
        raise ValueError("cpu_threads_per_job must be 1")
    for field in ("poll_interval_seconds", "resource_sample_interval_seconds"):
        if float(runtime.get(field, 0.0)) <= 0.0:
            raise ValueError(f"runtime.{field} must be positive")


def _validate_base_training_config(base: Mapping[str, Any]) -> None:
    exact = {
        "channels": 1,
        "embedding_dim": 1024,
        "noise_dim": 32,
        "generator_noise_mode": "gaussian",
        "generator_current_input_mode": "current_support_masked",
        "residual_output_mode": "identity_softplus_residual",
        "batch_size": 16,
        "discriminator_iter": 5,
        "support_mask_mode": "raw_joint",
        "news_first_dataset_tolerance_minutes": 5,
        "news_first_text_ablation_mode": "real_text",
        "news_first_label_reliability_mode": "none",
        "news_first_train_end_utc": "2023-07-01T00:00:00Z",
        "news_first_validation_end_utc": "2023-10-01T00:00:00Z",
        "news_first_surface_grid_profile": "exact_ttm_16x16_v1",
        "news_first_surface_grid_sha256": GRID_SHA256,
    }
    for key, expected in exact.items():
        if base.get(key) != expected:
            raise ValueError(
                f"Base training contract drifted for {key}: "
                f"expected {expected!r}, got {base.get(key)!r}"
            )
    if not bool(base.get("news_first_materialize_validation_loader")):
        raise ValueError("Base config must materialize the Q3 validation loader")
    if bool(base.get("news_first_materialize_test_loader")):
        raise ValueError("Base config must not materialize a test/Q4 loader")
    if not bool(base.get("use_reduce_lr_on_plateau")):
        raise ValueError("Existing ReduceLROnPlateau scheduler must remain enabled")
    if not bool(base.get("use_early_stopping")):
        raise ValueError("Existing early stopping must remain enabled")


def resolve_config(config_path: str | Path) -> dict[str, Any]:
    source = _resolve_repo_path(config_path)
    root = _require_mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "root")
    resolved = _require_mapping(
        root.get("film_pure_cnn_ablation_sweep"),
        "film_pure_cnn_ablation_sweep",
    )
    resolved["source_config_path"] = str(source)
    resolved["base_training_config"] = str(
        _resolve_repo_path(str(resolved["base_training_config"]))
    )
    resolved["runtime"] = _require_mapping(resolved.get("runtime"), "runtime")
    resolved["runtime"]["python_executable"] = str(
        _resolve_repo_path(str(resolved["runtime"]["python_executable"]))
    )
    _validate_resolved(resolved)
    base = _require_mapping(
        yaml.safe_load(
            Path(str(resolved["base_training_config"])).read_text(encoding="utf-8")
        ),
        "base training config",
    )
    _validate_base_training_config(base)
    return resolved


def _arm_payload(resolved: Mapping[str, Any], arm_id: str) -> dict[str, Any]:
    arm = _require_mapping(resolved["arms"][arm_id], f"arms.{arm_id}")
    profile = _require_mapping(resolved["legacy_profile"], "legacy_profile")
    return {
        "arm_id": arm_id,
        "arm_label": str(arm["label"]),
        "generator_conditioning_mode": str(arm["generator_conditioning_mode"]),
        "critic_conditioning_mode": str(arm["critic_conditioning_mode"]),
        "gen_text_hidden_dim": int(arm["gen_text_hidden_dim"]),
        "gen_text_out_dim": int(arm["gen_text_out_dim"]),
        **{key: int(profile[key]) for key in LEGACY_PROFILE},
    }


def _parameter_counts(arm: Mapping[str, Any]) -> dict[str, int]:
    """Instantiate the executable graph and report its true parameter counts."""

    cpu_rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(0)
        generator = Generator(
            channels=1,
            embedding_dim=1024,
            noise_dim=32,
            surface_height=16,
            surface_width=16,
            base_channels=int(arm["gen_base_channels"]),
            res_blocks=int(arm["gen_res_blocks"]),
            text_hidden_dim=int(arm["gen_text_hidden_dim"]),
            text_out_dim=int(arm["gen_text_out_dim"]),
            hidden_dim=int(arm["gen_hidden_dim"]),
            residual_output_mode="identity_softplus_residual",
            generator_noise_mode="gaussian",
            generator_current_input_mode="current_support_masked",
            generator_conditioning_mode=str(arm["generator_conditioning_mode"]),
        )
        critic = Discriminator(
            channels=1,
            embedding_dim=1024,
            surface_height=16,
            surface_width=16,
            base_channels=int(arm["disc_base_channels"]),
            res_blocks=int(arm["disc_res_blocks"]),
            text_hidden_dim=int(arm["disc_text_hidden_dim"]),
            hidden_dim=int(arm["disc_hidden_dim"]),
            critic_normalization_mode="legacy_instance_norm_v1",
            critic_conditioning_mode=str(arm["critic_conditioning_mode"]),
        )
        generator_parameters = sum(
            parameter.numel() for parameter in generator.parameters()
        )
        critic_parameters = sum(parameter.numel() for parameter in critic.parameters())
    finally:
        torch.random.set_rng_state(cpu_rng_state)
    return {
        "generator_parameters": generator_parameters,
        "critic_parameters": critic_parameters,
        "wgan_parameters": generator_parameters + critic_parameters,
    }


def _state_dict_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_state_sha256(path: Path) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Checkpoint lacks state_dict: {path}")
    digest = hashlib.sha256()
    for key, raw_tensor in state.items():
        if not isinstance(raw_tensor, torch.Tensor):
            raise ValueError(f"Non-tensor checkpoint state {key!r}: {path}")
        tensor = raw_tensor.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _initial_state_hashes(
    resolved: Mapping[str, Any], arm_id: str, seed: int
) -> dict[str, str]:
    """Hash a deterministic isolated G-then-D initialization for fairness QA."""

    arm = _arm_payload(resolved, arm_id)
    cpu_rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        generator = Generator(
            channels=1,
            embedding_dim=1024,
            noise_dim=32,
            surface_height=16,
            surface_width=16,
            base_channels=int(arm["gen_base_channels"]),
            res_blocks=int(arm["gen_res_blocks"]),
            text_hidden_dim=int(arm["gen_text_hidden_dim"]),
            text_out_dim=int(arm["gen_text_out_dim"]),
            hidden_dim=int(arm["gen_hidden_dim"]),
            residual_output_mode="identity_softplus_residual",
            generator_noise_mode="gaussian",
            generator_current_input_mode="current_support_masked",
            generator_conditioning_mode=str(arm["generator_conditioning_mode"]),
        )
        critic = Discriminator(
            channels=1,
            embedding_dim=1024,
            surface_height=16,
            surface_width=16,
            base_channels=int(arm["disc_base_channels"]),
            res_blocks=int(arm["disc_res_blocks"]),
            text_hidden_dim=int(arm["disc_text_hidden_dim"]),
            hidden_dim=int(arm["disc_hidden_dim"]),
            critic_normalization_mode="legacy_instance_norm_v1",
            critic_conditioning_mode=str(arm["critic_conditioning_mode"]),
        )
        return {
            "initial_generator_state_sha256": _state_dict_sha256(generator),
            "initial_critic_state_sha256": _state_dict_sha256(critic),
        }
    finally:
        torch.random.set_rng_state(cpu_rng_state)


def _initial_state_fairness(specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(str(spec["arm_id"]), int(spec["seed"])): spec for spec in specs}
    generator_group = (
        "film_unet_mask_coords_text64",
        "film_unet_mask_coords_text64_nolp",
        "film_unet_mask_coords_text64_projection",
    )
    critic_group = (
        "film_unet_mask_coords_text64",
        "film_unet_mask_coords_text64_nolp",
    )
    seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        generator_hashes = {
            str(by_key[(arm_id, seed)]["initial_generator_state_sha256"])
            for arm_id in generator_group
        }
        critic_hashes = {
            str(by_key[(arm_id, seed)]["initial_critic_state_sha256"])
            for arm_id in critic_group
        }
        if len(generator_hashes) != 1:
            raise AssertionError(
                f"Compact Generator initial state drifted for seed={seed}"
            )
        if len(critic_hashes) != 1:
            raise AssertionError(
                f"LP/NoLP Critic initial state drifted for seed={seed}"
            )
        seed_rows.append(
            {
                "seed": seed,
                "generator_equal_arms": list(generator_group),
                "generator_state_sha256": next(iter(generator_hashes)),
                "lp_nolp_critic_equal_arms": list(critic_group),
                "lp_nolp_critic_state_sha256": next(iter(critic_hashes)),
                "projection_critic_excluded_from_state_equality": True,
            }
        )
    payload = {
        "schema_version": 1,
        "initialization_order": "isolated torch.manual_seed(seed), Generator then Critic",
        "scope_note": (
            "Only identical executable graphs are required to share initial state; "
            "dense/decoder/coordinate-width changes are not claimed weight-identical."
        ),
        "seeds": seed_rows,
    }
    return {**payload, "fairness_contract_sha256": _payload_sha256(payload)}


def _architecture_contract(resolved: Mapping[str, Any], arm_id: str) -> dict[str, Any]:
    arm = _arm_payload(resolved, arm_id)
    generator_mode = str(arm["generator_conditioning_mode"])
    critic_mode = str(arm["critic_conditioning_mode"])
    payload = {
        "schema_version": 1,
        **arm,
        "generator_conditioning_fingerprint": generator_conditioning_fingerprint(
            generator_mode
        ),
        "critic_conditioning_fingerprint": critic_conditioning_fingerprint(critic_mode),
        **_parameter_counts(arm),
    }
    return {**payload, "architecture_profile_sha256": _payload_sha256(payload)}


def _model_contract(resolved: Mapping[str, Any], arm_id: str) -> dict[str, Any]:
    architecture = _architecture_contract(resolved, arm_id)
    payload = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "architecture_profile_sha256": architecture["architecture_profile_sha256"],
        "architecture": {
            key: value
            for key, value in architecture.items()
            if key != "architecture_profile_sha256"
        },
        "surface_grid_profile": "exact_ttm_16x16_v1",
        "surface_grid_sha256": GRID_SHA256,
        "alignment_tolerance_minutes": 5,
        "train_end_utc": "2023-07-01T00:00:00Z",
        "validation_end_utc": "2023-10-01T00:00:00Z",
        "q4_loader_materialized": False,
        "embedding_mode": "lp",
        "embedding_dim": 1024,
        "generator_current_input_mode": "current_support_masked",
        "generator_noise_mode": "gaussian",
        "noise_dim": 32,
        "initial_learning_rate": LEARNING_RATE,
        "lr_warmup_epochs": 0,
    }
    return {**payload, "model_contract_sha256": _payload_sha256(payload)}


def experiment_specs(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    architectures = {
        arm_id: _architecture_contract(resolved, arm_id) for arm_id in ARM_IDS
    }
    models = {arm_id: _model_contract(resolved, arm_id) for arm_id in ARM_IDS}
    specs: list[dict[str, Any]] = []
    for arm_index, arm_id in enumerate(ARM_IDS):
        architecture = architectures[arm_id]
        model = models[arm_id]
        for seed_index, seed in enumerate(SEEDS):
            initial_state = _initial_state_hashes(resolved, arm_id, seed)
            spec = {
                "job_id": f"{arm_id}_seed_{seed:03d}",
                "experiment_kind": EXPERIMENT_KIND,
                "arm_id": arm_id,
                "arm_label": architecture["arm_label"],
                "seed": seed,
                "gpu_id": GPU_IDS[(arm_index + seed_index) % len(GPU_IDS)],
                "learning_rate": LEARNING_RATE,
                "generator_conditioning_mode": architecture[
                    "generator_conditioning_mode"
                ],
                "generator_conditioning_fingerprint": architecture[
                    "generator_conditioning_fingerprint"
                ],
                "critic_conditioning_mode": architecture["critic_conditioning_mode"],
                "critic_conditioning_fingerprint": architecture[
                    "critic_conditioning_fingerprint"
                ],
                "gen_text_hidden_dim": architecture["gen_text_hidden_dim"],
                "gen_text_out_dim": architecture["gen_text_out_dim"],
                "generator_parameters": architecture["generator_parameters"],
                "critic_parameters": architecture["critic_parameters"],
                "wgan_parameters": architecture["wgan_parameters"],
                "architecture_profile_sha256": architecture[
                    "architecture_profile_sha256"
                ],
                "model_contract_sha256": model["model_contract_sha256"],
                **initial_state,
            }
            spec["job_spec_sha256"] = _payload_sha256(spec)
            specs.append(spec)
    if len(specs) != EXPECTED_JOB_COUNT:
        raise AssertionError("Unexpected architecture-ablation job count")
    if len({str(spec["job_id"]) for spec in specs}) != EXPECTED_JOB_COUNT:
        raise AssertionError("Duplicate architecture-ablation job IDs")
    gpu_counts = {
        gpu: sum(int(spec["gpu_id"]) == gpu for spec in specs) for gpu in GPU_IDS
    }
    if gpu_counts != {0: 11, 1: 10}:
        raise AssertionError(f"GPU assignment drifted: {gpu_counts}")
    for seed in SEEDS:
        seed_counts = {
            gpu: sum(
                int(spec["seed"]) == seed and int(spec["gpu_id"]) == gpu
                for spec in specs
            )
            for gpu in GPU_IDS
        }
        if abs(seed_counts[0] - seed_counts[1]) > 1:
            raise AssertionError(
                f"Seed/GPU balance drifted for seed={seed}: {seed_counts}"
            )
    _initial_state_fairness(specs)
    return specs


def _training_payload(
    resolved: Mapping[str, Any], output_root: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    base_path = Path(str(resolved["base_training_config"]))
    payload = _require_mapping(
        yaml.safe_load(base_path.read_text(encoding="utf-8")), "base config"
    )
    arm = _arm_payload(resolved, str(spec["arm_id"]))
    payload.update(
        {
            "generator_conditioning_mode": str(spec["generator_conditioning_mode"]),
            "critic_conditioning_mode": str(spec["critic_conditioning_mode"]),
            "gen_base_channels": int(arm["gen_base_channels"]),
            "gen_res_blocks": int(arm["gen_res_blocks"]),
            "gen_text_hidden_dim": int(arm["gen_text_hidden_dim"]),
            "gen_text_out_dim": int(arm["gen_text_out_dim"]),
            "gen_hidden_dim": int(arm["gen_hidden_dim"]),
            "disc_base_channels": int(arm["disc_base_channels"]),
            "disc_res_blocks": int(arm["disc_res_blocks"]),
            "disc_text_hidden_dim": int(arm["disc_text_hidden_dim"]),
            "disc_hidden_dim": int(arm["disc_hidden_dim"]),
            "learning_rate": LEARNING_RATE,
            "generator_learning_rate": LEARNING_RATE,
            "discriminator_learning_rate": LEARNING_RATE,
            "lr_warmup_epochs": 0,
            "lr_warmup_start_factor": 0.1,
            "num_epochs": int(resolved["max_epochs"]),
            "early_stopping_min_epochs": int(resolved["early_stopping_min_epochs"]),
            "early_stopping_patience": int(resolved["early_stopping_patience"]),
            "validation_mc_samples": int(resolved["validation_mc_samples"]),
            "seed": int(spec["seed"]),
            "news_first_capacity_profile": "legacy",
            "news_first_lr_profile": "pure_cnn_ablation_lr_5e_07_no_warmup",
            "news_first_architecture_profile_sha256": str(
                spec["architecture_profile_sha256"]
            ),
            "news_first_model_contract_sha256": str(spec["model_contract_sha256"]),
            "news_first_materialize_validation_loader": True,
            "news_first_materialize_test_loader": False,
            "output_root": str(
                (
                    output_root
                    / "runs"
                    / str(spec["arm_id"])
                    / f"seed_{int(spec['seed']):03d}"
                ).resolve()
            ),
        }
    )
    _validate_base_training_config(payload)
    return payload


def _registry_path(output_root: Path) -> Path:
    return output_root / "registry/jobs.json"


def _job_status_path(output_root: Path, job_id: str) -> Path:
    return output_root / f"registry/jobs/{job_id}.status.json"


def _artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "artifact_role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _source_hashes(resolved: Mapping[str, Any]) -> dict[str, str]:
    paths = (*SOURCE_PATHS, Path(str(resolved["source_config_path"])))
    hashes = {
        f"source_sha256::{path.relative_to(REPO_ROOT)}": _sha256_file(path)
        for path in paths
    }
    base = Path(str(resolved["base_training_config"]))
    hashes[f"source_sha256::{base.relative_to(REPO_ROOT)}"] = _sha256_file(base)
    return hashes


def prepare(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    registry_path = _registry_path(output_root)
    source_hashes = _source_hashes(resolved)
    specs = experiment_specs(resolved)
    initial_state_fairness = _initial_state_fairness(specs)
    fairness_path = output_root / "registry/initial_state_fairness.json"
    if registry_path.is_file():
        if not resume:
            raise FileExistsError(
                f"Registry already exists; pass --resume: {registry_path}"
            )
        registry = _read_json(registry_path)
        if registry.get("experiment_kind") != EXPERIMENT_KIND:
            raise ValueError("Existing registry experiment kind drifted")
        if registry.get("source_hashes") != source_hashes:
            raise ValueError("Source/config drift detected; resume refused")
        if registry.get("jobs_payload_sha256") != _payload_sha256(specs):
            raise ValueError("Job matrix drift detected; resume refused")
        if (
            not fairness_path.is_file()
            or _sha256_file(fairness_path)
            != registry.get("initial_state_fairness_sha256")
            or _read_json(fairness_path) != initial_state_fairness
        ):
            raise ValueError("Initial-state fairness manifest drift detected")
        for job in registry["jobs"]:
            config_path = Path(str(job["config_path"]))
            if (
                not config_path.is_file()
                or _sha256_file(config_path) != job["config_sha256"]
            ):
                raise ValueError(f"Generated job config drift: {config_path}")
        return registry

    output_root.mkdir(parents=True, exist_ok=True)
    for directory in (
        "analysis",
        "configs",
        "control",
        "control/job_locks",
        "logs",
        "registry/jobs",
    ):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for spec in specs:
        config_path = output_root / f"configs/{spec['job_id']}.yaml"
        payload = _training_payload(resolved, output_root, spec)
        _write_yaml(config_path, payload)
        job = {
            **spec,
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256_file(config_path),
            "run_root": str(Path(str(payload["output_root"])).resolve()),
        }
        jobs.append(job)
        _write_json(
            _job_status_path(output_root, str(spec["job_id"])),
            {
                "job_id": spec["job_id"],
                "state": "pending",
                "attempt": 0,
                "updated_at": _utc_now(),
                "job_spec_sha256": spec["job_spec_sha256"],
            },
        )
    registry = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "created_at": _utc_now(),
        "source_config_path": str(resolved["source_config_path"]),
        "base_training_config_path": str(resolved["base_training_config"]),
        "source_hashes": source_hashes,
        "expected_job_count": EXPECTED_JOB_COUNT,
        "jobs_payload_sha256": _payload_sha256(specs),
        "initial_state_fairness_path": str(fairness_path.resolve()),
        "jobs": jobs,
    }
    _write_json(fairness_path, initial_state_fairness)
    registry["initial_state_fairness_sha256"] = _sha256_file(fairness_path)
    _write_json(registry_path, registry)
    return registry


def _prepare_benchmark(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    if output_root.resolve() == _resolve_repo_path(DEFAULT_OUTPUT_DIR):
        raise ValueError(
            "Benchmark root must be separate from the formal experiment root"
        )
    registry_path = _registry_path(output_root)
    source_hashes = _source_hashes(resolved)
    formal_specs = experiment_specs(resolved)
    specs: list[dict[str, Any]] = []
    for formal in formal_specs:
        spec = {key: value for key, value in formal.items() if key != "job_spec_sha256"}
        spec.update(
            {
                "experiment_kind": BENCHMARK_EXPERIMENT_KIND,
                "num_epochs": 1,
                "benchmark": True,
            }
        )
        spec["job_spec_sha256"] = _payload_sha256(spec)
        specs.append(spec)
    if registry_path.is_file():
        if not resume:
            raise FileExistsError(
                f"Benchmark registry exists; pass --resume: {registry_path}"
            )
        registry = _read_json(registry_path)
        if registry.get("experiment_kind") != BENCHMARK_EXPERIMENT_KIND:
            raise ValueError("Benchmark experiment kind drifted")
        if registry.get("source_hashes") != source_hashes:
            raise ValueError("Benchmark source/config drift detected")
        if registry.get("jobs_payload_sha256") != _payload_sha256(specs):
            raise ValueError("Benchmark job matrix drift detected")
        for job in registry["jobs"]:
            config_path = Path(str(job["config_path"]))
            if (
                not config_path.is_file()
                or _sha256_file(config_path) != job["config_sha256"]
            ):
                raise ValueError(f"Benchmark generated config drift: {config_path}")
        return registry

    for directory in ("analysis", "configs", "control", "logs", "registry/jobs"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for spec in specs:
        config_path = output_root / f"configs/{spec['job_id']}.yaml"
        payload = _training_payload(resolved, output_root, spec)
        payload.update(
            {
                "num_epochs": 1,
                "use_early_stopping": False,
                "early_stopping_min_epochs": 1,
                "early_stopping_patience": 1,
                "save_every": 1_000_000,
                "news_first_lr_profile": "pure_cnn_ablation_epoch1_benchmark",
            }
        )
        _write_yaml(config_path, payload)
        job = {
            **spec,
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256_file(config_path),
            "run_root": str(Path(str(payload["output_root"])).resolve()),
        }
        jobs.append(job)
        _write_json(
            _job_status_path(output_root, str(job["job_id"])),
            {
                "job_id": job["job_id"],
                "state": "pending",
                "attempt": 0,
                "updated_at": _utc_now(),
                "job_spec_sha256": job["job_spec_sha256"],
            },
        )
    registry = {
        "schema_version": 1,
        "experiment_kind": BENCHMARK_EXPERIMENT_KIND,
        "created_at": _utc_now(),
        "source_config_path": str(resolved["source_config_path"]),
        "source_hashes": source_hashes,
        "expected_job_count": EXPECTED_JOB_COUNT,
        "jobs_payload_sha256": _payload_sha256(specs),
        "jobs": jobs,
    }
    _write_json(registry_path, registry)
    return registry


def _verify_artifacts(status_payload: Mapping[str, Any]) -> None:
    artifacts = status_payload.get("artifacts")
    if not isinstance(artifacts, Sequence) or not artifacts:
        raise ValueError("Completed status has no artifacts")
    for raw_artifact in artifacts:
        artifact = _require_mapping(raw_artifact, "artifact")
        path = Path(str(artifact["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact["size_bytes"])
            or _sha256_file(path) != artifact["sha256"]
        ):
            raise ValueError(f"Artifact drift: {path}")


def _discover_completed(job: Mapping[str, Any]) -> dict[str, Any] | None:
    run_root = Path(str(job["run_root"]))
    candidates = sorted(run_root.glob("*/metrics/best_learned_checkpoint.json"))
    for best_path in reversed(candidates):
        run_dir = best_path.parents[1]
        paths = {
            "best_learned": best_path,
            "generator_best_learned": run_dir / "checkpoints/generator_best_learned.pt",
            "discriminator_best_learned": run_dir
            / "checkpoints/discriminator_best_learned.pt",
            "training_metrics": run_dir / "metrics/training_metrics.csv",
            "resolved_config": run_dir / "metrics/training_resolved_config.yaml",
            "run_log": run_dir / "run.log",
            "generator_initial": run_dir / "checkpoints/generator_initial_epoch0.pt",
            "critic_initial": run_dir / "checkpoints/discriminator_initial_epoch0.pt",
            "generator_final": run_dir / "checkpoints/generator.pt",
            "critic_final": run_dir / "checkpoints/discriminator.pt",
        }
        if not all(path.is_file() for path in paths.values()):
            continue
        if "Training complete" not in paths["run_log"].read_text(
            encoding="utf-8", errors="replace"
        ):
            continue
        best = _read_json(best_path)
        config = _require_mapping(
            yaml.safe_load(paths["resolved_config"].read_text(encoding="utf-8")),
            "resolved training config",
        )
        expected = {
            "seed": int(job["seed"]),
            "generator_conditioning_mode": str(job["generator_conditioning_mode"]),
            "critic_conditioning_mode": str(job["critic_conditioning_mode"]),
            "gen_text_hidden_dim": int(job["gen_text_hidden_dim"]),
            "gen_text_out_dim": int(job["gen_text_out_dim"]),
            "gen_base_channels": 32,
            "disc_base_channels": 32,
            "num_epochs": int(job.get("num_epochs", 30)),
            "lr_warmup_epochs": 0,
            "news_first_architecture_profile_sha256": str(
                job["architecture_profile_sha256"]
            ),
            "news_first_model_contract_sha256": str(job["model_contract_sha256"]),
            "news_first_train_end_utc": "2023-07-01T00:00:00Z",
            "news_first_validation_end_utc": "2023-10-01T00:00:00Z",
            "news_first_materialize_validation_loader": True,
            "news_first_materialize_test_loader": False,
        }
        if any(config.get(key) != value for key, value in expected.items()):
            continue
        if not (
            math.isclose(
                float(config.get("generator_learning_rate", 0.0)), LEARNING_RATE
            )
            and math.isclose(
                float(config.get("discriminator_learning_rate", 0.0)), LEARNING_RATE
            )
            and str(best.get("model_contract_sha256"))
            == str(job["model_contract_sha256"])
            and str(best.get("architecture_profile_sha256"))
            == str(job["architecture_profile_sha256"])
            and str(best.get("generator_conditioning_fingerprint"))
            == str(job["generator_conditioning_fingerprint"])
            and str(best.get("critic_conditioning_fingerprint"))
            == str(job["critic_conditioning_fingerprint"])
            and int(best.get("best_epoch", 0)) >= 1
            and bool(best.get("best_learned_epoch_ge_1"))
            and not bool(best.get("news_first_materialize_test_loader"))
        ):
            continue
        with paths["training_metrics"].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        epochs = [int(row["epoch"]) for row in rows]
        if not epochs or epochs[0] != 0 or epochs != list(range(epochs[-1] + 1)):
            continue
        finite_metrics = True
        for row in rows:
            for value in row.values():
                if value in {None, ""}:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    finite_metrics = False
                    break
            if not finite_metrics:
                break
        if not finite_metrics:
            continue
        generator_initial_sha = _checkpoint_state_sha256(paths["generator_initial"])
        generator_final_sha = _checkpoint_state_sha256(paths["generator_final"])
        critic_initial_sha = _checkpoint_state_sha256(paths["critic_initial"])
        critic_final_sha = _checkpoint_state_sha256(paths["critic_final"])
        if (
            generator_initial_sha == generator_final_sha
            or critic_initial_sha == critic_final_sha
        ):
            continue
        metrics = _require_mapping(best.get("metrics"), "best metrics")
        best_mae = float(metrics["val_recon"])
        persistence_mae = float(metrics["val_current_recon"])
        if not (
            math.isfinite(best_mae)
            and math.isfinite(persistence_mae)
            and persistence_mae > 0.0
        ):
            continue
        return {
            "run_dir": str(run_dir.resolve()),
            "best_epoch": int(best["best_epoch"]),
            "best_mae": best_mae,
            "persistence_mae": persistence_mae,
            "completed_epochs": epochs[-1],
            "generator_parameters_updated": True,
            "critic_parameters_updated": True,
            "generator_initial_state_sha256": generator_initial_sha,
            "generator_final_state_sha256": generator_final_sha,
            "critic_initial_state_sha256": critic_initial_sha,
            "critic_final_state_sha256": critic_final_sha,
            "artifacts": [_artifact(path, role) for role, path in paths.items()],
        }
    return None


def _read_status(output_root: Path, job_id: str) -> dict[str, Any]:
    return _read_json(_job_status_path(output_root, job_id))


def _write_status(output_root: Path, job_id: str, payload: Mapping[str, Any]) -> None:
    _write_json(_job_status_path(output_root, job_id), dict(payload))


def _find_job(registry: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    matches = [dict(job) for job in registry["jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate job_id={job_id!r}")
    return matches[0]


def _counts(registry: Mapping[str, Any], output_root: Path) -> dict[str, int]:
    counts = {"pending": 0, "running": 0, "complete": 0, "failed": 0}
    for job in registry["jobs"]:
        state = str(_read_status(output_root, str(job["job_id"])).get("state"))
        counts[state] = counts.get(state, 0) + 1
    return counts


def run_worker(
    resolved: Mapping[str, Any], output_root: Path, *, job_id: str
) -> dict[str, Any]:
    registry = prepare(resolved, output_root, resume=True)
    job = _find_job(registry, job_id)
    lock_path = output_root / f"control/job_locks/{job_id}.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another worker already owns {job_id}") from exc
    previous = _read_status(output_root, job_id)
    if previous.get("state") == "complete":
        _verify_artifacts(previous)
        return previous
    recovered = _discover_completed(job)
    if recovered is not None:
        completed_status = {
            "job_id": job_id,
            "state": "complete",
            "attempt": int(previous.get("attempt", 0)),
            "updated_at": _utc_now(),
            "job_spec_sha256": job["job_spec_sha256"],
            **recovered,
        }
        _write_status(output_root, job_id, completed_status)
        return completed_status

    attempt = int(previous.get("attempt", 0)) + 1
    stdout_log = str(os.environ.get("ABLATION_WORKER_LOG_PATH", ""))
    _write_status(
        output_root,
        job_id,
        {
            "job_id": job_id,
            "state": "running",
            "attempt": attempt,
            "pid": os.getpid(),
            "gpu_id": int(job["gpu_id"]),
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "job_spec_sha256": job["job_spec_sha256"],
            "stdout_log": stdout_log,
        },
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:."
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
    runtime = _require_mapping(resolved["runtime"], "runtime")
    env["OMP_NUM_THREADS"] = str(runtime["cpu_threads_per_job"])
    env["MKL_NUM_THREADS"] = str(runtime["cpu_threads_per_job"])
    command = [
        str(runtime["python_executable"]),
        "scripts/train/train_vol.py",
        "--config",
        str(job["config_path"]),
        "--train-only",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if result.returncode != 0:
        failed = {
            "job_id": job_id,
            "state": "failed",
            "attempt": attempt,
            "returncode": int(result.returncode),
            "updated_at": _utc_now(),
            "job_spec_sha256": job["job_spec_sha256"],
            "stdout_log": stdout_log,
        }
        _write_status(output_root, job_id, failed)
        raise RuntimeError(f"{job_id} training exited with code {result.returncode}")
    completed = _discover_completed(job)
    if completed is None:
        failed = {
            "job_id": job_id,
            "state": "failed",
            "attempt": attempt,
            "returncode": 0,
            "updated_at": _utc_now(),
            "job_spec_sha256": job["job_spec_sha256"],
            "stdout_log": stdout_log,
            "message": "Training exited 0 but frozen artifacts are incomplete",
        }
        _write_status(output_root, job_id, failed)
        raise RuntimeError(str(failed["message"]))
    completed_status = {
        "job_id": job_id,
        "state": "complete",
        "attempt": attempt,
        "returncode": 0,
        "updated_at": _utc_now(),
        "job_spec_sha256": job["job_spec_sha256"],
        "stdout_log": stdout_log,
        **completed,
    }
    _write_status(output_root, job_id, completed_status)
    return completed_status


def dry_run(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    registry = prepare(resolved, output_root, resume=resume)
    jobs = [_find_job(registry, f"{arm_id}_seed_{SEEDS[0]:03d}") for arm_id in ARM_IDS]
    expected_payload = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "jobs": [
            {
                "job_id": job["job_id"],
                "config_sha256": job["config_sha256"],
                "job_spec_sha256": job["job_spec_sha256"],
            }
            for job in jobs
        ],
        "source_hashes": registry["source_hashes"],
    }
    manifest_path = output_root / "control/dry_run_manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("contract_sha256") != _payload_sha256(expected_payload):
            raise ValueError("Dry-run contract drift detected")
        for artifact in manifest.get("artifacts", []):
            path = Path(str(artifact["path"]))
            if (
                not path.is_file()
                or path.stat().st_size != int(artifact["size_bytes"])
                or _sha256_file(path) != artifact["sha256"]
            ):
                raise ValueError(f"Dry-run log drift: {path}")
        return manifest

    runtime = _require_mapping(resolved["runtime"], "runtime")
    artifacts: list[dict[str, Any]] = []
    for job in jobs:
        log_path = output_root / f"logs/dry_run_{job['arm_id']}.log"
        env = os.environ.copy()
        env["PYTHONPATH"] = "src:."
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
        env["OMP_NUM_THREADS"] = str(runtime["cpu_threads_per_job"])
        env["MKL_NUM_THREADS"] = str(runtime["cpu_threads_per_job"])
        command = [
            str(runtime["python_executable"]),
            "scripts/train/train_vol.py",
            "--config",
            str(job["config_path"]),
            "--dry-run",
        ]
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"Dry run failed for {job['arm_id']}; see {log_path}")
        artifacts.append(_artifact(log_path, f"dry_run::{job['arm_id']}"))
    manifest = {
        **expected_payload,
        "contract_sha256": _payload_sha256(expected_payload),
        "completed_at": _utc_now(),
        "artifacts": artifacts,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _resource_snapshot(output_root: Path) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    memory: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                memory[f"{key}_kib"] = int(raw_value.strip().split()[0])
    payload = {
        "timestamp": _utc_now(),
        "returncode": result.returncode,
        "rows": result.stdout.strip().splitlines(),
        **memory,
    }
    with (output_root / "control/resource_snapshots.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_supervisor_status(
    registry: Mapping[str, Any],
    output_root: Path,
    *,
    state: str,
    active: Mapping[str, Any],
    message: str = "",
) -> None:
    _write_json(
        output_root / "control/status.json",
        {
            "experiment_kind": registry["experiment_kind"],
            "state": state,
            "pid": os.getpid(),
            "pid_alive": True,
            "updated_at": _utc_now(),
            "counts": _counts(registry, output_root),
            "active_jobs": sorted(active),
            "message": message,
        },
    )


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def benchmark(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    """Run the isolated 21-cell, one-epoch GPU prelaunch benchmark."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "control/benchmark.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another benchmark supervisor owns the lock") from exc
    registry = _prepare_benchmark(resolved, output_root, resume=resume)
    _atomic_write_text(output_root / "control/supervisor.pid", f"{os.getpid()}\n")
    stop_requested = False

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    pending_by_gpu: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in GPU_IDS}
    for raw_job in registry["jobs"]:
        job = dict(raw_job)
        job_id = str(job["job_id"])
        job_status = _read_status(output_root, job_id)
        if job_status.get("state") == "complete":
            _verify_artifacts(job_status)
            continue
        recovered = _discover_completed(job)
        if recovered is not None:
            _write_status(
                output_root,
                job_id,
                {
                    "job_id": job_id,
                    "state": "complete",
                    "attempt": int(job_status.get("attempt", 0)),
                    "updated_at": _utc_now(),
                    "job_spec_sha256": job["job_spec_sha256"],
                    **recovered,
                },
            )
            continue
        pending_by_gpu[int(job["gpu_id"])].append(job)

    runtime = _require_mapping(resolved["runtime"], "runtime")
    workers_per_gpu = int(runtime["workers_per_gpu"])
    poll_interval = float(runtime["poll_interval_seconds"])
    resource_interval = float(runtime["resource_sample_interval_seconds"])
    active: dict[str, dict[str, Any]] = {}
    failure = ""
    last_resource = 0.0
    _write_supervisor_status(registry, output_root, state="running", active=active)
    while any(pending_by_gpu.values()) or active:
        if stop_requested or failure:
            for record in active.values():
                _terminate_process_group(record["process"])
                record["handle"].close()
            state = "interrupted" if stop_requested else "failed"
            _write_supervisor_status(
                registry,
                output_root,
                state=state,
                active={},
                message=failure or "Benchmark supervisor received a stop signal",
            )
            raise RuntimeError(failure or "Benchmark interrupted")

        for gpu in GPU_IDS:
            running_on_gpu = sum(record["gpu_id"] == gpu for record in active.values())
            while running_on_gpu < workers_per_gpu and pending_by_gpu[gpu]:
                job = pending_by_gpu[gpu].pop(0)
                job_id = str(job["job_id"])
                previous = _read_status(output_root, job_id)
                attempt = int(previous.get("attempt", 0)) + 1
                log_path = output_root / f"logs/{job_id}.attempt_{attempt:02d}.log"
                handle = log_path.open("w", encoding="utf-8")
                env = os.environ.copy()
                env["PYTHONPATH"] = "src:."
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["OMP_NUM_THREADS"] = str(runtime["cpu_threads_per_job"])
                env["MKL_NUM_THREADS"] = str(runtime["cpu_threads_per_job"])
                command = [
                    str(runtime["python_executable"]),
                    "scripts/train/train_vol.py",
                    "--config",
                    str(job["config_path"]),
                    "--train-only",
                ]
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[job_id] = {
                    "process": process,
                    "handle": handle,
                    "gpu_id": gpu,
                    "job": job,
                    "attempt": attempt,
                    "log_path": log_path,
                }
                _write_status(
                    output_root,
                    job_id,
                    {
                        "job_id": job_id,
                        "state": "running",
                        "attempt": attempt,
                        "pid": process.pid,
                        "gpu_id": gpu,
                        "started_at": _utc_now(),
                        "updated_at": _utc_now(),
                        "job_spec_sha256": job["job_spec_sha256"],
                        "stdout_log": str(log_path.resolve()),
                    },
                )
                print(
                    f"benchmark launched {job_id} pid={process.pid} gpu={gpu}",
                    flush=True,
                )
                running_on_gpu += 1

        for job_id, record in list(active.items()):
            returncode = record["process"].poll()
            if returncode is None:
                continue
            record["handle"].close()
            job = record["job"]
            if returncode == 0:
                completed = _discover_completed(job)
            else:
                completed = None
            if completed is None:
                failure = (
                    f"Benchmark {job_id} failed verification with return code "
                    f"{returncode}; see {record['log_path']}"
                )
                _write_status(
                    output_root,
                    job_id,
                    {
                        "job_id": job_id,
                        "state": "failed",
                        "attempt": record["attempt"],
                        "returncode": returncode,
                        "updated_at": _utc_now(),
                        "job_spec_sha256": job["job_spec_sha256"],
                        "stdout_log": str(record["log_path"].resolve()),
                    },
                )
            else:
                _write_status(
                    output_root,
                    job_id,
                    {
                        "job_id": job_id,
                        "state": "complete",
                        "attempt": record["attempt"],
                        "returncode": 0,
                        "updated_at": _utc_now(),
                        "job_spec_sha256": job["job_spec_sha256"],
                        "stdout_log": str(record["log_path"].resolve()),
                        **completed,
                    },
                )
                print(f"benchmark completed {job_id}", flush=True)
            del active[job_id]

        now = time.monotonic()
        if now - last_resource >= resource_interval:
            _resource_snapshot(output_root)
            last_resource = now
        _write_supervisor_status(
            registry,
            output_root,
            state="running",
            active=active,
            message=failure,
        )
        if any(pending_by_gpu.values()) or active:
            time.sleep(poll_interval)

    rows: list[dict[str, Any]] = []
    for job in registry["jobs"]:
        job_status = _read_status(output_root, str(job["job_id"]))
        if job_status.get("state") != "complete":
            raise ValueError("Benchmark registry is incomplete")
        _verify_artifacts(job_status)
        rows.append(
            {
                "job_id": job["job_id"],
                "arm_id": job["arm_id"],
                "seed": job["seed"],
                "gpu_id": job["gpu_id"],
                "completed_epochs": job_status["completed_epochs"],
                "generator_parameters_updated": job_status[
                    "generator_parameters_updated"
                ],
                "critic_parameters_updated": job_status["critic_parameters_updated"],
                "best_q3_mae": job_status["best_mae"],
                "run_dir": job_status["run_dir"],
            }
        )
    _atomic_write_text(
        output_root / "analysis/benchmark_jobs.csv",
        _csv_text(rows),
    )
    summary = {
        "schema_version": 1,
        "experiment_kind": BENCHMARK_EXPERIMENT_KIND,
        "completed_jobs": len(rows),
        "expected_jobs": EXPECTED_JOB_COUNT,
        "all_training_complete": len(rows) == EXPECTED_JOB_COUNT,
        "all_metrics_finite": True,
        "all_generator_parameters_updated": all(
            bool(row["generator_parameters_updated"]) for row in rows
        ),
        "all_critic_parameters_updated": all(
            bool(row["critic_parameters_updated"]) for row in rows
        ),
        "workers_per_gpu": workers_per_gpu,
        "gpu_job_counts": {
            str(gpu): sum(int(row["gpu_id"]) == gpu for row in rows) for gpu in GPU_IDS
        },
        "resource_snapshots_path": str(
            (output_root / "control/resource_snapshots.jsonl").resolve()
        ),
        "completed_at": _utc_now(),
    }
    _write_json(output_root / "analysis/benchmark_summary.json", summary)
    _write_supervisor_status(registry, output_root, state="complete", active={})
    return summary


def launch(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "control/supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another supervisor owns the experiment lock") from exc
    registry = prepare(resolved, output_root, resume=resume)
    _atomic_write_text(output_root / "control/supervisor.pid", f"{os.getpid()}\n")
    stop_requested = False

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    pending_by_gpu: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in GPU_IDS}
    for raw_job in registry["jobs"]:
        job = dict(raw_job)
        job_id = str(job["job_id"])
        job_status = _read_status(output_root, job_id)
        if job_status.get("state") == "complete":
            _verify_artifacts(job_status)
            continue
        recovered = _discover_completed(job)
        if recovered is not None:
            _write_status(
                output_root,
                job_id,
                {
                    "job_id": job_id,
                    "state": "complete",
                    "attempt": int(job_status.get("attempt", 0)),
                    "updated_at": _utc_now(),
                    "job_spec_sha256": job["job_spec_sha256"],
                    **recovered,
                },
            )
            continue
        pending_by_gpu[int(job["gpu_id"])].append(job)

    runtime = _require_mapping(resolved["runtime"], "runtime")
    workers_per_gpu = int(runtime["workers_per_gpu"])
    poll_interval = float(runtime["poll_interval_seconds"])
    resource_interval = float(runtime["resource_sample_interval_seconds"])
    source_config = str(resolved["source_config_path"])
    active: dict[str, dict[str, Any]] = {}
    failure = ""
    last_resource = 0.0
    _write_supervisor_status(registry, output_root, state="running", active=active)
    while any(pending_by_gpu.values()) or active:
        if stop_requested or failure:
            for record in active.values():
                _terminate_process_group(record["process"])
                record["handle"].close()
            state = "interrupted" if stop_requested else "failed"
            _write_supervisor_status(
                registry,
                output_root,
                state=state,
                active={},
                message=failure or "Supervisor received a stop signal",
            )
            raise RuntimeError(failure or "Supervisor interrupted")

        for gpu in GPU_IDS:
            running_on_gpu = sum(record["gpu_id"] == gpu for record in active.values())
            while running_on_gpu < workers_per_gpu and pending_by_gpu[gpu]:
                job = pending_by_gpu[gpu].pop(0)
                job_id = str(job["job_id"])
                previous = _read_status(output_root, job_id)
                attempt = int(previous.get("attempt", 0)) + 1
                log_path = output_root / f"logs/{job_id}.attempt_{attempt:02d}.log"
                handle = log_path.open("w", encoding="utf-8")
                env = os.environ.copy()
                env["PYTHONPATH"] = "src:."
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["OMP_NUM_THREADS"] = str(runtime["cpu_threads_per_job"])
                env["MKL_NUM_THREADS"] = str(runtime["cpu_threads_per_job"])
                env["ABLATION_WORKER_LOG_PATH"] = str(log_path.resolve())
                command = [
                    str(runtime["python_executable"]),
                    "-m",
                    "scripts.rq3.news_first_vol_film_pure_cnn_ablation",
                    "worker",
                    "--config",
                    source_config,
                    "--output-dir",
                    str(output_root),
                    "--job-id",
                    job_id,
                ]
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[job_id] = {
                    "process": process,
                    "handle": handle,
                    "gpu_id": gpu,
                    "job": job,
                    "attempt": attempt,
                    "log_path": log_path,
                }
                print(
                    f"launched {job_id} pid={process.pid} gpu={gpu} attempt={attempt}",
                    flush=True,
                )
                running_on_gpu += 1

        for job_id, record in list(active.items()):
            returncode = record["process"].poll()
            if returncode is None:
                continue
            record["handle"].close()
            job_status = _read_status(output_root, job_id)
            if returncode != 0 or job_status.get("state") != "complete":
                failure = (
                    f"{job_id} worker exited with code {returncode}; "
                    f"see {record['log_path']}"
                )
                if job_status.get("state") != "failed":
                    _write_status(
                        output_root,
                        job_id,
                        {
                            "job_id": job_id,
                            "state": "failed",
                            "attempt": record["attempt"],
                            "returncode": returncode,
                            "updated_at": _utc_now(),
                            "job_spec_sha256": record["job"]["job_spec_sha256"],
                            "stdout_log": str(record["log_path"].resolve()),
                        },
                    )
            else:
                _verify_artifacts(job_status)
                print(
                    f"completed {job_id} epoch={job_status['best_epoch']} "
                    f"mae={float(job_status['best_mae']):.12g}",
                    flush=True,
                )
            del active[job_id]

        now = time.monotonic()
        if now - last_resource >= resource_interval:
            _resource_snapshot(output_root)
            last_resource = now
        _write_supervisor_status(
            registry,
            output_root,
            state="running",
            active=active,
            message=failure,
        )
        if any(pending_by_gpu.values()) or active:
            time.sleep(poll_interval)

    result = postprocess(output_root)
    _write_supervisor_status(registry, output_root, state="complete", active={})
    print(f"all {EXPECTED_JOB_COUNT} jobs completed", flush=True)
    return result


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot serialize an empty CSV")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def postprocess(output_root: Path) -> dict[str, Any]:
    registry = _read_json(_registry_path(output_root))
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Unexpected experiment kind")
    rows: list[dict[str, Any]] = []
    for job in registry["jobs"]:
        job_status = _read_status(output_root, str(job["job_id"]))
        if job_status.get("state") != "complete":
            raise ValueError("Cannot postprocess an incomplete registry")
        _verify_artifacts(job_status)
        mae = float(job_status["best_mae"])
        persistence = float(job_status["persistence_mae"])
        rows.append(
            {
                "job_id": job["job_id"],
                "arm_id": job["arm_id"],
                "arm_label": job["arm_label"],
                "seed": job["seed"],
                "gpu_id": job["gpu_id"],
                "generator_conditioning_mode": job["generator_conditioning_mode"],
                "critic_conditioning_mode": job["critic_conditioning_mode"],
                "gen_text_hidden_dim": job["gen_text_hidden_dim"],
                "gen_text_out_dim": job["gen_text_out_dim"],
                "generator_parameters": job["generator_parameters"],
                "critic_parameters": job["critic_parameters"],
                "wgan_parameters": job["wgan_parameters"],
                "best_epoch": job_status["best_epoch"],
                "completed_epochs": job_status["completed_epochs"],
                "best_q3_mae": mae,
                "persistence_q3_mae": persistence,
                "log_mae_ratio_vs_persistence": math.log(mae / persistence),
                "improvement_vs_persistence_pct": (persistence - mae)
                / persistence
                * 100.0,
                "run_dir": job_status["run_dir"],
            }
        )
    if len(rows) != EXPECTED_JOB_COUNT:
        raise ValueError("Postprocess row count drifted")
    for seed in SEEDS:
        persistence_values = {
            float(row["persistence_q3_mae"]) for row in rows if int(row["seed"]) == seed
        }
        if len(persistence_values) != 1:
            raise ValueError(
                f"Persistence baseline drifted across arms for seed={seed}"
            )

    baseline_by_seed = {
        int(row["seed"]): float(row["best_q3_mae"])
        for row in rows
        if row["arm_id"] == "baseline"
    }
    arm_rows: list[dict[str, Any]] = []
    for arm_id in ARM_IDS:
        selected = [row for row in rows if row["arm_id"] == arm_id]
        if len(selected) != len(SEEDS):
            raise ValueError(f"Incomplete arm={arm_id}")
        mean_mae = sum(float(row["best_q3_mae"]) for row in selected) / len(selected)
        mean_persistence = sum(
            float(row["persistence_q3_mae"]) for row in selected
        ) / len(selected)
        mean_log_ratio = sum(
            float(row["log_mae_ratio_vs_persistence"]) for row in selected
        ) / len(selected)
        baseline_log_ratios = [
            math.log(float(row["best_q3_mae"]) / baseline_by_seed[int(row["seed"])])
            for row in selected
        ]
        arm_rows.append(
            {
                "arm_id": arm_id,
                "arm_label": selected[0]["arm_label"],
                "generator_conditioning_mode": selected[0][
                    "generator_conditioning_mode"
                ],
                "critic_conditioning_mode": selected[0]["critic_conditioning_mode"],
                "generator_parameters": selected[0]["generator_parameters"],
                "critic_parameters": selected[0]["critic_parameters"],
                "wgan_parameters": selected[0]["wgan_parameters"],
                "seed_count": len(selected),
                "mean_best_q3_mae": mean_mae,
                "mean_persistence_q3_mae": mean_persistence,
                "mean_log_mae_ratio_vs_persistence": mean_log_ratio,
                "geomean_mae_ratio_vs_persistence": math.exp(mean_log_ratio),
                "mean_improvement_vs_persistence_pct": (1.0 - math.exp(mean_log_ratio))
                * 100.0,
                "seeds_beating_persistence": sum(
                    float(row["best_q3_mae"]) <= float(row["persistence_q3_mae"])
                    for row in selected
                ),
                "mean_log_mae_ratio_vs_baseline": sum(baseline_log_ratios)
                / len(baseline_log_ratios),
                "mean_best_epoch": sum(int(row["best_epoch"]) for row in selected)
                / len(selected),
            }
        )
    arm_rows.sort(key=lambda row: float(row["mean_log_mae_ratio_vs_persistence"]))
    for rank, row in enumerate(arm_rows, start=1):
        row["rank"] = rank
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(analysis_dir / "job_summary.csv", _csv_text(rows))
    _atomic_write_text(analysis_dir / "arm_ranking.csv", _csv_text(arm_rows))
    result = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "selection_scope": "Q3 development only; Q4 not materialized",
        "completed_jobs": len(rows),
        "rank_metric": "mean_seed_log(best_q3_mae / persistence_q3_mae)",
        "point_leader": arm_rows[0]["arm_id"],
        "arms": arm_rows,
        "created_at": _utc_now(),
    }
    _write_json(analysis_dir / "arm_ranking.json", result)
    return result


def status(output_root: Path) -> dict[str, Any]:
    registry_path = _registry_path(output_root)
    if not registry_path.is_file():
        return {"state": "not_prepared", "output_root": str(output_root)}
    registry = _read_json(registry_path)
    status_path = output_root / "control/status.json"
    payload = _read_json(status_path) if status_path.is_file() else {}
    pid_path = output_root / "control/supervisor.pid"
    pid = int(pid_path.read_text(encoding="utf-8").strip()) if pid_path.is_file() else 0
    payload.update(
        {
            "output_root": str(output_root),
            "pid": pid,
            "pid_alive": bool(pid and _pid_alive(pid)),
            "counts": _counts(registry, output_root),
        }
    )
    return payload


def run_pipeline(
    resolved: Mapping[str, Any],
    output_root: Path,
    *,
    resume: bool,
    benchmark_root: Path | None = None,
) -> dict[str, Any]:
    resolved_benchmark_root = (
        _resolve_repo_path(DEFAULT_BENCHMARK_DIR)
        if benchmark_root is None
        else benchmark_root.resolve()
    )
    benchmark(
        resolved,
        resolved_benchmark_root,
        resume=_registry_path(resolved_benchmark_root).is_file(),
    )
    registry_exists = _registry_path(output_root).is_file()
    prepare(resolved, output_root, resume=resume if registry_exists else False)
    dry_run(resolved, output_root, resume=True)
    return launch(resolved, output_root, resume=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in (
        "benchmark",
        "prepare",
        "dry-run",
        "worker",
        "launch",
        "status",
        "postprocess",
        "run-pipeline",
    ):
        sub = subparsers.add_parser(action)
        sub.add_argument("--config", default=DEFAULT_CONFIG)
        sub.add_argument(
            "--output-dir",
            default=(
                DEFAULT_BENCHMARK_DIR if action == "benchmark" else DEFAULT_OUTPUT_DIR
            ),
        )
        if action in {"benchmark", "prepare", "dry-run", "launch", "run-pipeline"}:
            sub.add_argument("--resume", action="store_true")
        if action == "run-pipeline":
            sub.add_argument("--benchmark-output-dir", default=DEFAULT_BENCHMARK_DIR)
        if action == "worker":
            sub.add_argument("--job-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    output_root = _resolve_repo_path(args.output_dir)
    if args.action == "status":
        print(json.dumps(status(output_root), indent=2, sort_keys=True))
        return 0
    if args.action == "postprocess":
        print(json.dumps(postprocess(output_root), indent=2, sort_keys=True))
        return 0
    resolved = resolve_config(args.config)
    if args.action == "benchmark":
        result = benchmark(resolved, output_root, resume=bool(args.resume))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.action == "prepare":
        registry = prepare(resolved, output_root, resume=bool(args.resume))
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "job_count": len(registry["jobs"]),
                    "gpu_counts": {
                        str(gpu): sum(
                            int(job["gpu_id"]) == gpu for job in registry["jobs"]
                        )
                        for gpu in GPU_IDS
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.action == "dry-run":
        manifest = dry_run(resolved, output_root, resume=bool(args.resume))
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.action == "worker":
        result = run_worker(resolved, output_root, job_id=str(args.job_id))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.action == "launch":
        result = launch(resolved, output_root, resume=bool(args.resume))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = run_pipeline(
        resolved,
        output_root,
        resume=bool(args.resume),
        benchmark_root=_resolve_repo_path(args.benchmark_output_dir),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
