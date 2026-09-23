"""Independent four-job zero-latent-noise WGAN ablation.

The experiment trains only the zero-noise cells and treats the matching
Gaussian coverage-stage runs as immutable references.  Its worker CLI lives in
this module so the branch-local experiment does not depend on modifying an
older public command surface.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import torch
import yaml

from scripts.rq3 import news_first_vol_training as training
from scripts.rq3.news_first_vol_coverage_sweep import (
    _capacity_seed_profile_sha256 as _coverage_capacity_seed_profile_sha256,
    _lr_profile_sha256 as _coverage_lr_profile_sha256,
    _qa_sha256,
)
from wgan_option.models.common import (
    GAUSSIAN_GENERATOR_NOISE_MODE,
    ZERO_GENERATOR_NOISE_MODE,
    generator_noise_fingerprint,
    normalize_generator_noise_mode,
)
from wgan_option.models.generator import Generator
from wgan_option.utils.inference_helpers import (
    resolve_checkpoint_generator_noise_contract,
)
from wgan_option.utils.text_ablation import (
    REAL_TEXT,
    normalize_text_ablation_mode,
    text_information_path,
)


ROOT_KEY = training.ROOT_KEY
REPO_ROOT = training.REPO_ROOT
EXPERIMENT_KIND = "zero_noise_seed_ablation"
EXPERIMENT_STAGE = "zero_noise_wgan_small_seed42"
CAPACITY_PROFILE = "small"
LR_PROFILE = "lr_5e_07"
FIXED_LEARNING_RATE = 5.0e-7
FIXED_SCHEDULER_MIN_LR = 5.0e-8
FROZEN_SEED = 42
FROZEN_TOLERANCES = (5, 30)
FROZEN_TEXT_MODES = ("current_only", REAL_TEXT)
GENERATOR_NOISE_MODE = ZERO_GENERATOR_NOISE_MODE
REFERENCE_NOISE_MODE = GAUSSIAN_GENERATOR_NOISE_MODE
NOISE_DIM = 32
EXPECTED_Q3_PAIRS = 123
EXPECTED_Q3_SESSIONS = 33
REFERENCE_STAGE_ID = "stage_3_wgan_low_lr_capacity"
REFERENCE_JOB_IDS = (
    "s3_wgan_small_lr_5e_07_seed_042_current_only_05m",
    "s3_wgan_small_lr_5e_07_seed_042_real_text_30m",
    "s3_wgan_small_lr_5e_07_seed_042_real_text_05m",
    "s3_wgan_small_lr_5e_07_seed_042_current_only_30m",
)
REFERENCE_RESOLVED_CONFIG_SHA256 = (
    "58b0c712ed04ae80d3cb53b8f86aaff55c3305b84c76fa10d5ab8a3db49c0661"
)
REFERENCE_Q3_PAIR_METRICS_SHA256 = (
    "eb49b862485e370460ccf23295e184715016c19037c58099a9a55328a6738eab"
)
EXPECTED_INITIAL_GENERATOR_STATE_SHA256 = (
    "ec724ab94c5f66de246b7e396cf8283ed2967a43989c818a30f89605cd7bc077"
)
EXPECTED_INITIAL_DISCRIMINATOR_STATE_SHA256 = (
    "5da9039c0cf6ff046097ab94ae0a0a8596762046717366967683f5bb909bb0b8"
)
EXPECTED_REFERENCE_HASHES: dict[str, dict[str, str]] = {
    "s3_wgan_small_lr_5e_07_seed_042_current_only_05m": {
        "config": "43378f521b6ec89ab03d5431d4e0ac8530e0dfcec1abfd715b49e73afa96a4e8",
        "generator": "51a38e23a90de2fb3c38a118ba311e85692ed11f5bc7d7e37dcbad2d13ffd417",
        "discriminator": "96ea5fdc8809c9c0f26c3a54edb20efef12356c327a3ccbbe132c9c3c77a7d9f",
        "metadata": "2e1012c0f64af50718c7314458b9cf9c33730fc729efab2076b118c9768fb2ea",
    },
    "s3_wgan_small_lr_5e_07_seed_042_current_only_30m": {
        "config": "68009bc859dc30efe0206246563b21bc67e423724b8210f3056f3a139e562a57",
        "generator": "5aa4e27eed88326b74f36e1b69b5533b6f282b860de77438ab3b49ea119e9bca",
        "discriminator": "0db1f2dec9c183d16504cdcd0c610884216de5296c6698c36b5501cb2093dcec",
        "metadata": "fad5ee1e6fc86e608f5e7109312c90fba2eada9c9adce135f22e5f3c7a4f0e48",
    },
    "s3_wgan_small_lr_5e_07_seed_042_real_text_05m": {
        "config": "553bb3dda9df81fb92aaa6a8979d599ab73973fd8acdbc05d21dc88b83d1e4d8",
        "generator": "62738b3a6965c5f49a0932d8e272dcfe5fa8db8d158f6d5ef808571b3fd64dca",
        "discriminator": "82530707692d16e85f282b0c6c423aa5557a4a433806b096ea97b7d37803f1e5",
        "metadata": "1260ade2a358dcb4a305726fe7a5b66133297335dd63c484096b8a4d2f0e90e2",
    },
    "s3_wgan_small_lr_5e_07_seed_042_real_text_30m": {
        "config": "3ea456a8ef828df31fb7212e00e956cdc19b5fd6cce0167525149b78e6f6341d",
        "generator": "a220276c10dd0b5383fc65f90232bae4a491983a5da748994ac7425b6e61a7df",
        "discriminator": "b2919daa702fa4c759592e8f53761490632c2890125dbc6623e9245ceb6441a1",
        "metadata": "3743884339d49bdd52ada1d4f3a632f6a9b53c3fc759362d120cd42fe502a611",
    },
}
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
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _require_mapping(value, label)


def _exact_float(value: Any, expected: float, label: str) -> None:
    observed = float(value)
    if not math.isfinite(observed) or observed != expected:
        raise ValueError(f"{label} is frozen to {expected}; observed={observed}")


def _ablation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(config.get("zero_noise_ablation"), "zero_noise_ablation")
    if not bool(value.get("enabled", False)):
        raise ValueError("zero_noise_ablation.enabled must be true")
    return value


def _validate_frozen_config(config: Mapping[str, Any]) -> None:
    datasets = _require_mapping(config.get("datasets"), "datasets")
    split = _require_mapping(config.get("split"), "split")
    ablation = _ablation_config(config)
    models = _require_mapping(config.get("models"), "models")
    wgan = _require_mapping(
        _require_mapping(models.get("wgan"), "models.wgan").get("training"),
        "models.wgan.training",
    )
    runtime = _require_mapping(config.get("runtime"), "runtime")

    if tuple(int(v) for v in datasets.get("tolerances_minutes", ())) != (5, 10, 15, 30):
        raise ValueError("Dataset tolerances are frozen to 5/10/15/30")
    if int(datasets.get("common_evaluation_tolerance_minutes", -1)) != 5:
        raise ValueError("Common Q3 evaluation data are frozen to 5m")
    if str(datasets.get("sheet_name", "")) != "gan_input_ready":
        raise ValueError("sheet_name is frozen to gan_input_ready")
    if str(datasets.get("text_embedding_mode", "")).lower() != "lp":
        raise ValueError("Text embedding is frozen to LP")
    if int(datasets.get("seed", -1)) != FROZEN_SEED:
        raise ValueError("Dataset split seed is frozen to 42")
    if training._support_mask_mode(datasets) != "raw_joint":
        raise ValueError("Support mask is frozen to raw_joint")
    if tuple(training._configured_text_ablation_modes(datasets)) != FROZEN_TEXT_MODES:
        raise ValueError("Text modes are frozen to current_only/real_text")
    if str(split.get("train_end_utc")) != "2023-07-01T00:00:00Z":
        raise ValueError("Train/Q3 boundary drift")
    if str(split.get("validation_end_utc")) != "2023-10-01T00:00:00Z":
        raise ValueError("Q3/Q4 boundary drift")
    if int(split.get("validation_mc_samples", -1)) != 1:
        raise ValueError("Zero-noise validation is a frozen single pass")

    contracts = {
        "experiment_kind": EXPERIMENT_KIND,
        "experiment_stage": EXPERIMENT_STAGE,
        "capacity_profile": CAPACITY_PROFILE,
        "lr_profile": LR_PROFILE,
        "generator_noise_mode": GENERATOR_NOISE_MODE,
        "reference_generator_noise_mode": REFERENCE_NOISE_MODE,
    }
    for field, expected in contracts.items():
        if str(ablation.get(field, "")) != expected:
            raise ValueError(f"zero_noise_ablation.{field} is frozen to {expected}")
    if int(ablation.get("seed", -1)) != FROZEN_SEED:
        raise ValueError("Ablation seed is frozen to 42")
    if (
        tuple(int(v) for v in ablation.get("tolerances_minutes", ()))
        != FROZEN_TOLERANCES
    ):
        raise ValueError("Ablation tolerances are frozen to 5/30")
    if (
        tuple(
            normalize_text_ablation_mode(v)
            for v in ablation.get("text_ablation_modes", ())
        )
        != FROZEN_TEXT_MODES
    ):
        raise ValueError("Ablation text modes are frozen")
    _exact_float(
        ablation.get("initial_learning_rate"), FIXED_LEARNING_RATE, "initial LR"
    )
    _exact_float(
        ablation.get("scheduler_min_lr"), FIXED_SCHEDULER_MIN_LR, "scheduler floor"
    )
    if int(ablation.get("bootstrap_clusters", -1)) != 10_000:
        raise ValueError("Bootstrap iterations are frozen to 10000")
    if (
        int(ablation.get("expected_q3_pairs", -1)) != EXPECTED_Q3_PAIRS
        or int(ablation.get("expected_q3_sessions", -1)) != EXPECTED_Q3_SESSIONS
    ):
        raise ValueError("Q3 panel counts are frozen to 123 pairs / 33 sessions")
    if not bool(ablation.get("q4_prediction_and_evaluation_forbidden", False)):
        raise ValueError("Q4 prediction/evaluation must remain forbidden")
    reference = _require_mapping(
        ablation.get("gaussian_reference"), "gaussian_reference"
    )
    if tuple(str(v) for v in reference.get("job_ids", ())) != REFERENCE_JOB_IDS:
        raise ValueError("Gaussian reference job IDs differ from the frozen four jobs")

    expected_ints = {
        "channels": 1,
        "embedding_dim": 1024,
        "noise_dim": NOISE_DIM,
        "gen_base_channels": 4,
        "gen_res_blocks": 0,
        "gen_text_hidden_dim": 32,
        "gen_text_out_dim": 16,
        "gen_hidden_dim": 128,
        "disc_base_channels": 4,
        "disc_res_blocks": 0,
        "disc_text_hidden_dim": 16,
        "disc_hidden_dim": 96,
        "batch_size": 16,
        "discriminator_iter": 5,
        "num_epochs": 100,
        "reduce_lr_patience": 3,
        "early_stopping_patience": 16,
        "early_stopping_min_epochs": 30,
    }
    for field, expected in expected_ints.items():
        if int(wgan.get(field, -1)) != expected:
            raise ValueError(f"WGAN {field} is frozen to {expected}")
    expected_strings = {
        "generator_noise_mode": GENERATOR_NOISE_MODE,
        "residual_output_mode": "identity_softplus_residual",
        "best_checkpoint_metric": "val_hybrid_score",
    }
    for field, expected in expected_strings.items():
        if str(wgan.get(field, "")) != expected:
            raise ValueError(f"WGAN {field} is frozen to {expected}")
    expected_floats = {
        "learning_rate": FIXED_LEARNING_RATE,
        "reduce_lr_factor": 0.5,
        "reduce_lr_min_lr": FIXED_SCHEDULER_MIN_LR,
        "beta_1": 0.5,
        "beta_2": 0.9,
        "lambda_gp": 10.0,
        "lambda_recon": 10.0,
        "lambda_calendar": 2.0,
        "lambda_butterfly": 2.0,
        "lambda_smooth": 0.1,
        "lambda_delta_shrink": 0.0,
        "baseline_penalty_weight": 2.0,
    }
    for field, expected in expected_floats.items():
        _exact_float(wgan.get(field), expected, f"WGAN {field}")
    for field in (
        "use_reduce_lr_on_plateau",
        "use_calendar_constraint",
        "use_butterfly_constraint",
        "use_smooth_constraint",
        "evaluate_initial_checkpoint",
        "use_early_stopping",
    ):
        if not bool(wgan.get(field, False)):
            raise ValueError(f"WGAN {field} must remain true")
    gpu_ids = tuple(int(v) for v in runtime.get("gpu_ids", ()))
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("Exactly two distinct physical GPUs are required")
    if int(runtime.get("slots_per_gpu", -1)) != 2:
        raise ValueError("Two worker slots per GPU are frozen for the four-job wave")


def _resolved_config(config_path: str | Path) -> dict[str, Any]:
    path = _resolve_repo_path(config_path)
    root = _load_yaml_mapping(path, "zero-noise config")
    config = _require_mapping(root.get(ROOT_KEY), ROOT_KEY)
    _validate_frozen_config(config)
    resolved = deepcopy(config)
    resolved["datasets"]["root"] = str(_resolve_repo_path(resolved["datasets"]["root"]))
    reference = resolved["zero_noise_ablation"]["gaussian_reference"]
    for field in (
        "root",
        "registry",
        "stage_qa",
        "resolved_config_hash",
        "q3_pair_metrics",
    ):
        reference[field] = str(_resolve_repo_path(reference[field]))
    resolved["runtime"]["python_executable"] = str(
        _resolve_repo_path(resolved["runtime"].get("python_executable", sys.executable))
    )
    resolved["source_config_path"] = str(path)
    return resolved


def _capacity_profile_sha256() -> str:
    profile = training.FROZEN_CAPACITY_PROFILES[CAPACITY_PROFILE]
    return training._capacity_profile_sha256(CAPACITY_PROFILE, profile)


def _lr_profile_sha256() -> str:
    # Reuse the immutable Gaussian experiment's profile identity.  The noise
    # treatment has its own fingerprint and must not silently redefine the LR.
    return _coverage_lr_profile_sha256(LR_PROFILE)


def _capacity_seed_profile_sha256() -> str:
    return _coverage_capacity_seed_profile_sha256(
        CAPACITY_PROFILE, LR_PROFILE, FROZEN_SEED, "wgan"
    )


def _noise_profile_sha256(mode: str) -> str:
    normalized = normalize_generator_noise_mode(mode)
    return _payload_sha256(
        {
            "schema_version": 1,
            "generator_noise_mode": normalized,
            "generator_noise_fingerprint": generator_noise_fingerprint(
                normalized, NOISE_DIM
            ),
            "noise_dim": NOISE_DIM,
            "capacity_profile_sha256": _capacity_profile_sha256(),
            "lr_profile_sha256": _lr_profile_sha256(),
            "seed": FROZEN_SEED,
            "support_mask_mode": "raw_joint",
            "residual_output_mode": "identity_softplus_residual",
        }
    )


def _job_id(mode: str, tolerance: int) -> str:
    return (
        f"zero_noise_wgan_small_lr_5e_07_seed_042_zero_"
        f"{normalize_text_ablation_mode(mode)}_{int(tolerance):02d}m"
    )


def _job_specs() -> tuple[dict[str, Any], ...]:
    # One wave, cross-balanced so both GPUs see both experimental axes.
    return (
        {
            "text_ablation_mode": "current_only",
            "tolerance_minutes": 5,
            "gpu_index": 0,
            "gpu_slot": 0,
        },
        {
            "text_ablation_mode": "real_text",
            "tolerance_minutes": 5,
            "gpu_index": 1,
            "gpu_slot": 0,
        },
        {
            "text_ablation_mode": "current_only",
            "tolerance_minutes": 30,
            "gpu_index": 1,
            "gpu_slot": 1,
        },
        {
            "text_ablation_mode": "real_text",
            "tolerance_minutes": 30,
            "gpu_index": 0,
            "gpu_slot": 1,
        },
    )


def _gaussian_reference_rows(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    reference = resolved["zero_noise_ablation"]["gaussian_reference"]
    root = Path(str(reference["root"]))
    registry_path = Path(str(reference["registry"]))
    qa_path = Path(str(reference["stage_qa"]))
    resolved_hash_path = Path(str(reference["resolved_config_hash"]))
    pair_path = Path(str(reference["q3_pair_metrics"]))
    for path in (registry_path, qa_path, resolved_hash_path, pair_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    qa = _read_json(qa_path)
    if (
        qa.get("status") != "pass"
        or qa.get("stage_id") != REFERENCE_STAGE_ID
        or qa.get("qa_sha256") != _qa_sha256(qa)
    ):
        raise ValueError("Gaussian reference Stage 3 QA is invalid")
    if (
        resolved_hash_path.read_text(encoding="utf-8").strip()
        != REFERENCE_RESOLVED_CONFIG_SHA256
    ):
        raise ValueError("Frozen Gaussian resolved-config hash changed")
    if _sha256_file(pair_path) != REFERENCE_Q3_PAIR_METRICS_SHA256:
        raise ValueError("Frozen Gaussian Q3 pair-metric hash changed")
    registry = _read_json(registry_path)
    if registry.get("experiment_kind") != "coverage_completion_sweep":
        raise ValueError("Gaussian reference registry kind mismatch")
    jobs = {str(job["job_id"]): dict(job) for job in registry.get("jobs", [])}
    if not set(REFERENCE_JOB_IDS).issubset(jobs):
        raise ValueError("Gaussian reference registry lacks a frozen job")
    rows: list[dict[str, Any]] = []
    for job_id in REFERENCE_JOB_IDS:
        job = jobs[job_id]
        mode = normalize_text_ablation_mode(job.get("text_ablation_mode", ""))
        tolerance = int(job.get("tolerance_minutes", -1))
        contracts = {
            "stage_id": REFERENCE_STAGE_ID,
            "model_family": "wgan",
            "capacity_profile": CAPACITY_PROFILE,
            "lr_profile": LR_PROFILE,
            "seed": FROZEN_SEED,
        }
        if any(job.get(field) != expected for field, expected in contracts.items()):
            raise ValueError(f"Gaussian reference axes drifted: {job_id}")
        if mode not in FROZEN_TEXT_MODES or tolerance not in FROZEN_TOLERANCES:
            raise ValueError(f"Gaussian reference mode/tolerance drifted: {job_id}")
        status_path = root / "registry" / "jobs" / f"{job_id}.status.json"
        status = _read_json(status_path)
        if status.get("status") != "completed" or status.get(
            "config_sha256"
        ) != job.get("config_sha256"):
            raise ValueError(f"Gaussian reference status invalid: {job_id}")
        artifacts = {
            str(row["artifact_role"]): dict(row) for row in status.get("artifacts", [])
        }
        for artifact in artifacts.values():
            path = Path(str(artifact.get("path", "")))
            if not path.is_file() or _sha256_file(path) != str(
                artifact.get("sha256", "")
            ):
                raise ValueError(f"Gaussian reference artifact hash drifted: {job_id}")
        required = (
            "generator_best_learned",
            "discriminator_best_learned",
            "best_learned_checkpoint",
            "generator_initial_epoch0",
            "discriminator_initial_epoch0",
            "initial_checkpoint",
        )
        if not set(required).issubset(artifacts):
            raise ValueError(f"Gaussian reference learned artifacts missing: {job_id}")
        frozen_hashes = EXPECTED_REFERENCE_HASHES[job_id]
        observed_hashes = {
            "config": str(job.get("config_sha256", "")),
            "generator": str(artifacts["generator_best_learned"]["sha256"]),
            "discriminator": str(artifacts["discriminator_best_learned"]["sha256"]),
            "metadata": str(artifacts["best_learned_checkpoint"]["sha256"]),
        }
        if observed_hashes != frozen_hashes:
            raise ValueError(f"Exact Gaussian reference hashes drifted: {job_id}")
        generator_path = Path(artifacts["generator_best_learned"]["path"])
        checkpoint = torch.load(generator_path, map_location="cpu", weights_only=False)
        reference_mode, reference_fingerprint = (
            resolve_checkpoint_generator_noise_contract(checkpoint)
        )
        if reference_mode != REFERENCE_NOISE_MODE:
            raise ValueError(f"Gaussian reference checkpoint is not Gaussian: {job_id}")
        generator_initial_path = Path(artifacts["generator_initial_epoch0"]["path"])
        discriminator_initial_path = Path(
            artifacts["discriminator_initial_epoch0"]["path"]
        )
        initial_generator_state_sha = _checkpoint_state_sha256(
            torch.load(generator_initial_path, map_location="cpu", weights_only=False)
        )
        initial_discriminator_state_sha = _checkpoint_state_sha256(
            torch.load(
                discriminator_initial_path, map_location="cpu", weights_only=False
            )
        )
        if initial_generator_state_sha != EXPECTED_INITIAL_GENERATOR_STATE_SHA256:
            raise ValueError(f"Gaussian initial G tensors drifted: {job_id}")
        if (
            initial_discriminator_state_sha
            != EXPECTED_INITIAL_DISCRIMINATOR_STATE_SHA256
        ):
            raise ValueError(f"Gaussian initial D tensors drifted: {job_id}")
        rows.append(
            {
                "reference_job_id": job_id,
                "text_ablation_mode": mode,
                "tolerance_minutes": tolerance,
                "generator_noise_mode": reference_mode,
                "generator_noise_fingerprint": reference_fingerprint,
                "job_spec_sha256": str(job.get("job_spec_sha256", "")),
                "config_sha256": str(job.get("config_sha256", "")),
                "status_path": str(status_path),
                "status_sha256": _sha256_file(status_path),
                "run_dir": str(status["run_dir"]),
                "best_learned_metadata_path": str(
                    artifacts["best_learned_checkpoint"]["path"]
                ),
                "best_learned_metadata_sha256": str(
                    artifacts["best_learned_checkpoint"]["sha256"]
                ),
                "generator_checkpoint_path": str(generator_path),
                "generator_checkpoint_sha256": str(
                    artifacts["generator_best_learned"]["sha256"]
                ),
                "discriminator_checkpoint_path": str(
                    artifacts["discriminator_best_learned"]["path"]
                ),
                "discriminator_checkpoint_sha256": str(
                    artifacts["discriminator_best_learned"]["sha256"]
                ),
                "initial_generator_checkpoint_path": str(generator_initial_path),
                "initial_generator_checkpoint_sha256": str(
                    artifacts["generator_initial_epoch0"]["sha256"]
                ),
                "initial_generator_state_sha256": initial_generator_state_sha,
                "initial_discriminator_checkpoint_path": str(
                    discriminator_initial_path
                ),
                "initial_discriminator_checkpoint_sha256": str(
                    artifacts["discriminator_initial_epoch0"]["sha256"]
                ),
                "initial_discriminator_state_sha256": initial_discriminator_state_sha,
                "initial_metadata_path": str(artifacts["initial_checkpoint"]["path"]),
                "initial_metadata_sha256": str(
                    artifacts["initial_checkpoint"]["sha256"]
                ),
                "reference_registry_path": str(registry_path),
                "reference_registry_sha256": _sha256_file(registry_path),
                "reference_stage_qa_path": str(qa_path),
                "reference_stage_qa_sha256": _sha256_file(qa_path),
                "reference_resolved_config_sha256": resolved_hash_path.read_text(
                    encoding="utf-8"
                ).strip(),
                "reference_q3_pair_metrics_path": str(pair_path),
                "reference_q3_pair_metrics_sha256": _sha256_file(pair_path),
            }
        )
    observed = {(row["text_ablation_mode"], row["tolerance_minutes"]) for row in rows}
    expected = {
        (mode, tolerance)
        for mode in FROZEN_TEXT_MODES
        for tolerance in FROZEN_TOLERANCES
    }
    if observed != expected or len(rows) != 4:
        raise ValueError("Gaussian reference matrix is not exactly 2x2")
    return rows


def _reference_by_cell(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["text_ablation_mode"]), int(row["tolerance_minutes"])): dict(row)
        for row in rows
    }


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor content independently of torch serialization metadata."""

    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_state_sha256(checkpoint: Mapping[str, Any]) -> str:
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint state_dict must be a mapping")
    return _tensor_state_sha256(state)


def _training_payload(
    resolved: Mapping[str, Any],
    root: Path,
    *,
    mode: str,
    tolerance: int,
) -> dict[str, Any]:
    payload = training._training_payload(
        resolved,
        family="wgan",
        tolerance=int(tolerance),
        text_ablation_mode=mode,
        multi_mode=True,
        experiment_root=root,
        capacity_profile=None,
    )
    shape = training.FROZEN_CAPACITY_PROFILES[CAPACITY_PROFILE]
    payload.update(
        {field: int(shape[field]) for field in training.CAPACITY_PROFILE_SHAPE_FIELDS}
    )
    payload.update(
        {
            "seed": FROZEN_SEED,
            "generator_noise_mode": GENERATOR_NOISE_MODE,
            "noise_dim": NOISE_DIM,
            "news_first_capacity_profile": CAPACITY_PROFILE,
            "news_first_capacity_profile_sha256": _capacity_profile_sha256(),
            "news_first_capacity_seed_profile_sha256": (
                _capacity_seed_profile_sha256()
            ),
            "news_first_lr_profile": LR_PROFILE,
            "news_first_lr_profile_sha256": _lr_profile_sha256(),
            "news_first_fixed_learning_rate_profile": LR_PROFILE,
            "news_first_fixed_learning_rate_profile_sha256": _lr_profile_sha256(),
            "learning_rate": FIXED_LEARNING_RATE,
            "use_reduce_lr_on_plateau": True,
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": 3,
            "reduce_lr_min_lr": FIXED_SCHEDULER_MIN_LR,
            "num_epochs": 100,
            "use_early_stopping": True,
            "early_stopping_min_epochs": 30,
            "early_stopping_patience": 16,
            "validation_mc_samples": 1,
            "output_root": str(
                root
                / "runs"
                / "wgan"
                / CAPACITY_PROFILE
                / LR_PROFILE
                / f"seed_{FROZEN_SEED:03d}"
                / GENERATOR_NOISE_MODE
                / normalize_text_ablation_mode(mode)
                / f"tolerance_{int(tolerance):02d}m"
            ),
        }
    )
    return payload


def _source_rows(
    resolved: Mapping[str, Any], reference_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = training._source_rows(resolved)
    seen = {str(row["path"]) for row in rows}
    paths: list[tuple[str, str, str]] = []
    reference = resolved["zero_noise_ablation"]["gaussian_reference"]
    for role in ("registry", "stage_qa", "resolved_config_hash", "q3_pair_metrics"):
        path = Path(str(reference[role]))
        paths.append((f"gaussian_reference_{role}", str(path), _sha256_file(path)))
    for row in reference_rows:
        job_id = str(row["reference_job_id"])
        paths.extend(
            (
                (
                    f"gaussian_{job_id}_status",
                    str(row["status_path"]),
                    str(row["status_sha256"]),
                ),
                (
                    f"gaussian_{job_id}_best_learned_metadata",
                    str(row["best_learned_metadata_path"]),
                    str(row["best_learned_metadata_sha256"]),
                ),
                (
                    f"gaussian_{job_id}_generator_best_learned",
                    str(row["generator_checkpoint_path"]),
                    str(row["generator_checkpoint_sha256"]),
                ),
                (
                    f"gaussian_{job_id}_discriminator_best_learned",
                    str(row["discriminator_checkpoint_path"]),
                    str(row["discriminator_checkpoint_sha256"]),
                ),
                (
                    f"gaussian_{job_id}_generator_initial_epoch0",
                    str(row["initial_generator_checkpoint_path"]),
                    str(row["initial_generator_checkpoint_sha256"]),
                ),
                (
                    f"gaussian_{job_id}_discriminator_initial_epoch0",
                    str(row["initial_discriminator_checkpoint_path"]),
                    str(row["initial_discriminator_checkpoint_sha256"]),
                ),
                (
                    f"gaussian_{job_id}_initial_metadata",
                    str(row["initial_metadata_path"]),
                    str(row["initial_metadata_sha256"]),
                ),
            )
        )
    for role, raw_path, expected_sha in paths:
        if raw_path in seen:
            continue
        path = Path(raw_path)
        observed_sha = _sha256_file(path)
        if observed_sha != expected_sha:
            raise ValueError(f"Gaussian source hash changed: {role}")
        rows.append(
            {
                "source_role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": observed_sha,
            }
        )
        seen.add(raw_path)
    return rows


def _code_rows() -> list[dict[str, Any]]:
    rows = training._code_rows()
    known = {str(row["relative_path"]) for row in rows}
    for relative in (
        "scripts/rq3/news_first_vol_zero_noise_ablation.py",
        "scripts/rq3/news_first_vol_zero_noise_analysis.py",
        "scripts/rq3/news_first_vol_zero_noise_report.py",
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


def _job_spec_sha256(job: Mapping[str, Any]) -> str:
    return _payload_sha256(
        {key: value for key, value in job.items() if key != "job_spec_sha256"}
    )


def _load_registry(root: Path) -> dict[str, Any]:
    registry = _require_mapping(
        _read_json(root / "registry" / "jobs.json"), "zero-noise registry"
    )
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment root is not a zero-noise seed ablation")
    return registry


def _validate_resolved_snapshot(root: Path) -> dict[str, Any]:
    resolved = _require_mapping(
        _load_yaml_mapping(root / "resolved_config.yaml", "resolved config").get(
            ROOT_KEY
        ),
        ROOT_KEY,
    )
    observed = _payload_sha256(resolved)
    recorded = (
        (root / "registry" / "resolved_config.sha256")
        .read_text(encoding="utf-8")
        .strip()
    )
    if observed != recorded:
        raise ValueError("Resolved zero-noise config hash mismatch")
    if str(_load_registry(root).get("resolved_config_sha256", "")) != recorded:
        raise ValueError("Registry resolved-config hash mismatch")
    return resolved


def _validate_reference_manifest(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    path = root / "gaussian_reference_manifest.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != 4:
        raise ValueError("Gaussian reference manifest must have four rows")
    expected_sha = str(
        _load_registry(root).get("gaussian_reference_manifest_sha256", "")
    )
    if _sha256_file(path) != expected_sha:
        raise ValueError("Gaussian reference manifest hash mismatch")
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            normalize_text_ablation_mode(row["text_ablation_mode"]),
            int(row["tolerance_minutes"]),
        )
        if key in observed:
            raise ValueError("Duplicate Gaussian reference cell")
        if str(row["generator_noise_mode"]) != REFERENCE_NOISE_MODE:
            raise ValueError("Gaussian reference mode drifted")
        if str(row["generator_noise_fingerprint"]) != generator_noise_fingerprint(
            REFERENCE_NOISE_MODE, NOISE_DIM
        ):
            raise ValueError("Gaussian reference fingerprint drifted")
        for path_field, sha_field in (
            ("status_path", "status_sha256"),
            ("best_learned_metadata_path", "best_learned_metadata_sha256"),
            ("generator_checkpoint_path", "generator_checkpoint_sha256"),
            ("discriminator_checkpoint_path", "discriminator_checkpoint_sha256"),
            (
                "initial_generator_checkpoint_path",
                "initial_generator_checkpoint_sha256",
            ),
            (
                "initial_discriminator_checkpoint_path",
                "initial_discriminator_checkpoint_sha256",
            ),
            ("initial_metadata_path", "initial_metadata_sha256"),
        ):
            target = Path(str(row[path_field]))
            if not target.is_file() or _sha256_file(target) != str(row[sha_field]):
                raise ValueError(f"Gaussian reference artifact changed: {path_field}")
        if (
            str(row["initial_generator_state_sha256"])
            != EXPECTED_INITIAL_GENERATOR_STATE_SHA256
            or str(row["initial_discriminator_state_sha256"])
            != EXPECTED_INITIAL_DISCRIMINATOR_STATE_SHA256
        ):
            raise ValueError("Gaussian initial tensor fingerprint drifted")
        observed[key] = row
    expected = {
        (mode, tolerance)
        for mode in FROZEN_TEXT_MODES
        for tolerance in FROZEN_TOLERANCES
    }
    if set(observed) != expected:
        raise ValueError("Gaussian reference manifest cells drifted")
    return observed


def _validate_job_lineage(root: Path, job: Mapping[str, Any]) -> None:
    if str(job.get("job_spec_sha256", "")) != _job_spec_sha256(job):
        raise ValueError(f"Job-spec hash mismatch: {job.get('job_id')}")
    mode = normalize_text_ablation_mode(job.get("text_ablation_mode", ""))
    tolerance = int(job.get("tolerance_minutes", -1))
    if mode not in FROZEN_TEXT_MODES or tolerance not in FROZEN_TOLERANCES:
        raise ValueError(f"Job axes are out of contract: {job.get('job_id')}")
    if str(job.get("job_id")) != _job_id(mode, tolerance):
        raise ValueError("Job ID does not encode zero noise and immutable axes")
    contracts = {
        "model_family": "wgan",
        "capacity_profile": CAPACITY_PROFILE,
        "capacity_profile_sha256": _capacity_profile_sha256(),
        "capacity_seed_profile_sha256": _capacity_seed_profile_sha256(),
        "lr_profile": LR_PROFILE,
        "lr_profile_sha256": _lr_profile_sha256(),
        "seed": FROZEN_SEED,
        "experiment_stage": EXPERIMENT_STAGE,
        "generator_noise_mode": GENERATOR_NOISE_MODE,
        "generator_noise_fingerprint": generator_noise_fingerprint(
            GENERATOR_NOISE_MODE, NOISE_DIM
        ),
        "noise_profile_sha256": _noise_profile_sha256(GENERATOR_NOISE_MODE),
        "initial_learning_rate": FIXED_LEARNING_RATE,
        "scheduler_min_lr": FIXED_SCHEDULER_MIN_LR,
        "support_mask_mode": "raw_joint",
    }
    for field, expected in contracts.items():
        if job.get(field) != expected:
            raise ValueError(f"Job lineage mismatch {job['job_id']}: {field}")
    if int(job.get("wave", -1)) != 1:
        raise ValueError("All four zero-noise jobs must be in one wave")
    resolved = _validate_resolved_snapshot(root)
    runtime = resolved["runtime"]
    gpu_ids = tuple(int(value) for value in runtime["gpu_ids"])
    if int(job.get("gpu_id", -1)) not in gpu_ids:
        raise ValueError("Job references an unknown GPU")
    references = _validate_reference_manifest(root)
    reference = references[(mode, tolerance)]
    reference_contracts = {
        "gaussian_reference_job_id": reference["reference_job_id"],
        "gaussian_generator_checkpoint_sha256": reference[
            "generator_checkpoint_sha256"
        ],
        "gaussian_best_learned_metadata_sha256": reference[
            "best_learned_metadata_sha256"
        ],
        "gaussian_initial_metadata_sha256": reference["initial_metadata_sha256"],
        "gaussian_initial_generator_state_sha256": reference[
            "initial_generator_state_sha256"
        ],
        "gaussian_initial_discriminator_state_sha256": reference[
            "initial_discriminator_state_sha256"
        ],
        "gaussian_reference_manifest_sha256": _sha256_file(
            root / "gaussian_reference_manifest.csv"
        ),
    }
    for field, expected in reference_contracts.items():
        if str(job.get(field, "")) != str(expected):
            raise ValueError(f"Gaussian reference lineage mismatch: {field}")
    source_path = root / "source_hashes.csv"
    if _sha256_file(source_path) != str(job.get("source_manifest_sha256", "")):
        raise ValueError("Source manifest hash mismatch")
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = Path(str(row["path"]))
            if not path.is_file() or _sha256_file(path) != str(row["sha256"]):
                raise ValueError(f"Source hash changed: {row['source_role']}")
    config_path = Path(str(job["training_config_path"]))
    if not config_path.is_file() or _sha256_file(config_path) != str(
        job["config_sha256"]
    ):
        raise ValueError("Training config hash mismatch")
    payload = _load_yaml_mapping(config_path, f"training config {job['job_id']}")
    payload_contracts = {
        "seed": FROZEN_SEED,
        "generator_noise_mode": GENERATOR_NOISE_MODE,
        "noise_dim": NOISE_DIM,
        "news_first_capacity_profile": CAPACITY_PROFILE,
        "news_first_capacity_profile_sha256": _capacity_profile_sha256(),
        "news_first_capacity_seed_profile_sha256": (_capacity_seed_profile_sha256()),
        "news_first_lr_profile": LR_PROFILE,
        "news_first_lr_profile_sha256": _lr_profile_sha256(),
        "learning_rate": FIXED_LEARNING_RATE,
        "reduce_lr_min_lr": FIXED_SCHEDULER_MIN_LR,
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 3,
        "early_stopping_min_epochs": 30,
        "early_stopping_patience": 16,
        "validation_mc_samples": 1,
        "support_mask_mode": "raw_joint",
        "residual_output_mode": "identity_softplus_residual",
    }
    for field, expected in payload_contracts.items():
        if payload.get(field) != expected:
            raise ValueError(f"Training config contract mismatch: {field}")
    if Path(str(payload["output_root"])) != Path(str(job["output_root"])):
        raise ValueError("Training output root mismatch")
    dataset_path = Path(str(job["dataset_path"]))
    if _sha256_file(dataset_path) != str(job["dataset_sha256"]):
        raise ValueError("Dataset hash mismatch")


def _validate_registry(root: Path) -> None:
    _validate_split_manifest(root)
    registry = _load_registry(root)
    jobs = [dict(job) for job in registry.get("jobs", [])]
    expected = {
        _job_id(mode, tolerance)
        for mode in FROZEN_TEXT_MODES
        for tolerance in FROZEN_TOLERANCES
    }
    observed = {str(job.get("job_id")) for job in jobs}
    if len(jobs) != 4 or len(observed) != 4 or observed != expected:
        raise ValueError("Zero-noise registry must contain the exact four jobs")
    if set(int(job["gpu_id"]) for job in jobs) != set(
        int(value) for value in _validate_resolved_snapshot(root)["runtime"]["gpu_ids"]
    ):
        raise ValueError("Both configured GPUs must be used")
    gpu_cells: dict[int, set[tuple[str, int]]] = {}
    for job in jobs:
        gpu_cells.setdefault(int(job["gpu_id"]), set()).add(
            (str(job["text_ablation_mode"]), int(job["tolerance_minutes"]))
        )
        _validate_job_lineage(root, job)
    if any(len(cells) != 2 for cells in gpu_cells.values()):
        raise ValueError("Each physical GPU must receive two cross-balanced cells")


def _validate_gaussian_q3_panel(resolved: Mapping[str, Any]) -> None:
    path = Path(
        str(resolved["zero_noise_ablation"]["gaussian_reference"]["q3_pair_metrics"])
    )
    if _sha256_file(path) != REFERENCE_Q3_PAIR_METRICS_SHA256:
        raise ValueError("Frozen Gaussian Q3 pair-metric SHA-256 changed")
    frame = pd.read_csv(path, low_memory=False)
    selected = frame[
        frame["run_id"].astype(str).isin(REFERENCE_JOB_IDS)
        & frame["stage_id"].astype(str).eq(REFERENCE_STAGE_ID)
        & frame["model"].astype(str).eq("wgan")
        & frame["capacity_profile"].astype(str).eq(CAPACITY_PROFILE)
        & frame["lr_profile"].astype(str).eq(LR_PROFILE)
        & frame["seed"].astype(int).eq(FROZEN_SEED)
    ].copy()
    keys = ["text_ablation_mode", "tolerance_minutes"]
    observed = set(selected[keys].itertuples(index=False, name=None))
    expected = {
        (mode, tolerance)
        for mode in FROZEN_TEXT_MODES
        for tolerance in FROZEN_TOLERANCES
    }
    if observed != expected or len(selected) != 4 * EXPECTED_Q3_PAIRS:
        raise ValueError("Gaussian reference Q3 matrix is not exactly 4x123")
    for _, group in selected.groupby(keys, sort=False):
        if (
            group["pair_id"].nunique() != EXPECTED_Q3_PAIRS
            or group["session_id"].nunique() != EXPECTED_Q3_SESSIONS
        ):
            raise ValueError("Gaussian Q3 reference panel count drifted")
    coverage = [
        frozenset(
            group[["pair_id", "session_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        for _, group in selected.groupby(keys, sort=False)
    ]
    if len(set(coverage)) != 1:
        raise ValueError("Gaussian reference cells do not share one paired Q3 panel")


def _validate_split_manifest(root: Path) -> None:
    frame = pd.read_csv(root / "split_manifest.csv", low_memory=False)
    relevant = frame[frame["tolerance_minutes"].isin(FROZEN_TOLERANCES)].copy()
    if set(relevant["tolerance_minutes"].astype(int)) != set(FROZEN_TOLERANCES):
        raise ValueError("Split manifest lacks the exact 5m/30m rows")
    expected_by_tolerance = {
        5: (936, 720, 210),
        30: (1591, 1050, 237),
    }
    for row in relevant.itertuples(index=False):
        tolerance = int(row.tolerance_minutes)
        observed_train = (
            int(row.train_rows),
            int(row.train_pairs),
            int(row.train_sessions),
        )
        if observed_train != expected_by_tolerance[tolerance]:
            raise ValueError(f"Training split count drifted for {tolerance}m")
        if (
            int(row.validation_rows),
            int(row.validation_pairs),
            int(row.validation_sessions),
        ) != (133, EXPECTED_Q3_PAIRS, EXPECTED_Q3_SESSIONS):
            raise ValueError(f"Q3 split count drifted for {tolerance}m")
        if (
            int(row.test_rows),
            int(row.test_pairs),
            int(row.test_sessions),
        ) != (152, 130, 45):
            raise ValueError(f"Q4 audit split count drifted for {tolerance}m")
        overlap_fields = (
            "train_validation_pair_overlap",
            "train_test_pair_overlap",
            "validation_test_pair_overlap",
            "train_validation_session_overlap",
            "train_test_session_overlap",
            "validation_test_session_overlap",
        )
        if any(int(getattr(row, field)) != 0 for field in overlap_fields):
            raise ValueError(f"Split leakage detected for {tolerance}m")
        if str(row.support_mask_mode) != "raw_joint" or str(row.status) != "pass":
            raise ValueError(f"Split support/status contract drifted for {tolerance}m")


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


def refresh_zero_noise_lineage(config_path: str | Path, root: str | Path) -> Path:
    experiment_root = Path(root).resolve(strict=False)
    expected_path = experiment_root / "registry" / "resolved_config.sha256"
    if not expected_path.is_file():
        raise ValueError(f"Not a prepared zero-noise experiment: {experiment_root}")
    resolved = _resolved_config(config_path)
    resolved_sha = _payload_sha256(resolved)
    if expected_path.read_text(encoding="utf-8").strip() != resolved_sha:
        raise ValueError("Prepared zero-noise config differs from requested config")
    reference_rows = _gaussian_reference_rows(resolved)
    source_rows = _source_rows(resolved, reference_rows)
    source_path = experiment_root / "source_hashes.csv"
    if source_path.is_file():
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            previous = {row["path"]: row["sha256"] for row in csv.DictReader(handle)}
        current_sources = {row["path"]: row["sha256"] for row in source_rows}
        if previous != current_sources:
            raise ValueError("Zero-noise source lineage changed")
    _write_csv(
        source_path,
        source_rows,
        ("source_role", "path", "size_bytes", "sha256"),
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
    if changed and _maximum_attempt(experiment_root) > 0:
        raise ValueError(
            "Runtime code changed after an experiment attempt; refusing a mixed-code "
            "four-job matrix. Prepare a new experiment root."
        )
    if not prepared or _maximum_attempt(experiment_root) == 0:
        _write_csv(
            code_path,
            code_rows,
            ("relative_path", "path", "size_bytes", "sha256"),
        )
        changed = False
    elif changed:
        current_path = experiment_root / "code_hashes_current.csv"
        _write_csv(
            current_path,
            code_rows,
            ("relative_path", "path", "size_bytes", "sha256"),
        )
    _write_json(
        experiment_root / "run_manifest.json",
        {
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
            "gaussian_reference_manifest_path": str(
                experiment_root / "gaussian_reference_manifest.csv"
            ),
            "gaussian_reference_manifest_sha256": _sha256_file(
                experiment_root / "gaussian_reference_manifest.csv"
            ),
            "q4_access_policy": (
                "loader_materialization_allowed; prediction_and_evaluation_forbidden"
            ),
            "updated_at_utc": _utc_now(),
        },
    )
    return experiment_root / "run_manifest.json"


TASK_FIELDS = (
    "job_id",
    "wave",
    "model_family",
    "capacity_profile",
    "capacity_profile_sha256",
    "capacity_seed_profile_sha256",
    "lr_profile",
    "lr_profile_sha256",
    "seed",
    "experiment_stage",
    "generator_noise_mode",
    "generator_noise_fingerprint",
    "noise_profile_sha256",
    "gaussian_reference_job_id",
    "gaussian_generator_checkpoint_sha256",
    "gaussian_best_learned_metadata_sha256",
    "gaussian_initial_metadata_sha256",
    "gaussian_initial_generator_state_sha256",
    "gaussian_initial_discriminator_state_sha256",
    "gaussian_reference_manifest_sha256",
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
        rows.append({**job, **status})
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
        root / "noise_profile_manifest.csv",
        root / "gaussian_reference_manifest.csv",
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


def prepare_zero_noise_experiment(
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
        if (
            not hash_path.is_file()
            or hash_path.read_text(encoding="utf-8").strip() != resolved_sha
        ):
            raise ValueError("Existing root is not this zero-noise experiment/config")
        _validate_registry(root)
        refresh_zero_noise_lineage(config_path, root)
        _refresh_config_hashes(root)
        _refresh_exports(root)
        return root

    training._validate_dataset_summary(Path(resolved["datasets"]["root"]))
    _validate_gaussian_q3_panel(resolved)
    reference_rows = _gaussian_reference_rows(resolved)
    source_rows = _source_rows(resolved, reference_rows)
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
    _validate_split_manifest(root)
    _write_yaml(root / "resolved_config.yaml", {ROOT_KEY: resolved})
    _atomic_write_text(hash_path, resolved_sha + "\n")
    _write_csv(
        root / "gaussian_reference_manifest.csv",
        reference_rows,
        tuple(reference_rows[0]),
    )
    reference_manifest_sha = _sha256_file(root / "gaussian_reference_manifest.csv")
    reference_by_cell = _reference_by_cell(reference_rows)
    _write_csv(
        root / "source_hashes.csv",
        source_rows,
        ("source_role", "path", "size_bytes", "sha256"),
    )
    source_sha = _sha256_file(root / "source_hashes.csv")
    source_by_path = {str(row["path"]): str(row["sha256"]) for row in source_rows}
    runtime = resolved["runtime"]
    gpu_ids = tuple(int(value) for value in runtime["gpu_ids"])
    numa = {
        int(key): int(value) for key, value in dict(runtime["gpu_numa_nodes"]).items()
    }
    jobs: list[dict[str, Any]] = []
    for spec in _job_specs():
        mode = normalize_text_ablation_mode(spec["text_ablation_mode"])
        tolerance = int(spec["tolerance_minutes"])
        gpu_id = gpu_ids[int(spec["gpu_index"])]
        reference = reference_by_cell[(mode, tolerance)]
        job_id = _job_id(mode, tolerance)
        payload = _training_payload(resolved, root, mode=mode, tolerance=tolerance)
        training_path = root / "configs" / "jobs" / f"{job_id}.yaml"
        _write_yaml(training_path, payload)
        dataset_path = str(payload["data_path"])
        if dataset_path not in source_by_path:
            raise ValueError(f"Dataset source hash unavailable: {dataset_path}")
        job: dict[str, Any] = {
            "job_id": job_id,
            "wave": 1,
            "model_family": "wgan",
            "trainer_command": "vol-xlsx",
            "capacity_profile": CAPACITY_PROFILE,
            "capacity_profile_sha256": _capacity_profile_sha256(),
            "capacity_seed_profile_sha256": _capacity_seed_profile_sha256(),
            "expected_wgan_parameters": int(
                training.FROZEN_CAPACITY_PROFILES[CAPACITY_PROFILE][
                    "expected_wgan_parameters"
                ]
            ),
            "lr_profile": LR_PROFILE,
            "lr_profile_sha256": _lr_profile_sha256(),
            "initial_learning_rate": FIXED_LEARNING_RATE,
            "scheduler_min_lr": FIXED_SCHEDULER_MIN_LR,
            "lr_trace": [FIXED_LEARNING_RATE],
            "seed": FROZEN_SEED,
            "experiment_stage": EXPERIMENT_STAGE,
            "generator_noise_mode": GENERATOR_NOISE_MODE,
            "generator_noise_fingerprint": generator_noise_fingerprint(
                GENERATOR_NOISE_MODE, NOISE_DIM
            ),
            "noise_profile_sha256": _noise_profile_sha256(GENERATOR_NOISE_MODE),
            "gaussian_reference_job_id": reference["reference_job_id"],
            "gaussian_generator_checkpoint_sha256": reference[
                "generator_checkpoint_sha256"
            ],
            "gaussian_best_learned_metadata_sha256": reference[
                "best_learned_metadata_sha256"
            ],
            "gaussian_initial_metadata_sha256": reference["initial_metadata_sha256"],
            "gaussian_initial_generator_state_sha256": reference[
                "initial_generator_state_sha256"
            ],
            "gaussian_initial_discriminator_state_sha256": reference[
                "initial_discriminator_state_sha256"
            ],
            "gaussian_reference_manifest_sha256": reference_manifest_sha,
            "text_ablation_mode": mode,
            "text_information_path": text_information_path(mode),
            "support_mask_mode": "raw_joint",
            "tolerance_minutes": tolerance,
            "gpu_id": gpu_id,
            "gpu_slot": int(spec["gpu_slot"]),
            "numa_node": numa[gpu_id],
            "training_config_path": str(training_path),
            "config_sha256": _sha256_file(training_path),
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
                "experiment_stage": EXPERIMENT_STAGE,
                "generator_noise_mode": GENERATOR_NOISE_MODE,
                "generator_noise_fingerprint": job["generator_noise_fingerprint"],
                "noise_profile_sha256": job["noise_profile_sha256"],
                "gaussian_reference_job_id": reference["reference_job_id"],
                "gaussian_reference_manifest_sha256": reference_manifest_sha,
                "updated_at_utc": _utc_now(),
            },
        )
    _write_json(
        root / "registry" / "jobs.json",
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "experiment_root": str(root),
            "experiment_stage": EXPERIMENT_STAGE,
            "resolved_config_sha256": resolved_sha,
            "gaussian_reference_manifest_sha256": reference_manifest_sha,
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
    noise_rows = []
    for mode in (GENERATOR_NOISE_MODE, REFERENCE_NOISE_MODE):
        noise_rows.append(
            {
                "generator_noise_mode": mode,
                "generator_noise_fingerprint": generator_noise_fingerprint(
                    mode, NOISE_DIM
                ),
                "noise_profile_sha256": _noise_profile_sha256(mode),
                "noise_dim": NOISE_DIM,
                "architecture_noise_width_preserved": True,
                "training_policy": (
                    "burn_standard_normal_then_zero"
                    if mode == GENERATOR_NOISE_MODE
                    else "standard_normal"
                ),
                "q3_evaluation_policy": (
                    "literal_zero_single_pass"
                    if mode == GENERATOR_NOISE_MODE
                    else "immutable_reference_MC16"
                ),
            }
        )
    _write_csv(root / "noise_profile_manifest.csv", noise_rows, tuple(noise_rows[0]))
    _refresh_config_hashes(root)
    _validate_registry(root)
    refresh_zero_noise_lineage(config_path, root)
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


def _checkpoint_parameter_count(checkpoint: Mapping[str, Any]) -> int:
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint state_dict must be a mapping")
    return int(sum(int(value.numel()) for value in state.values()))


def _validated_wgan_lr_trace(
    job: Mapping[str, Any], run_dir: Path, *, dry_run: bool
) -> list[dict[str, float | int]] | list[float]:
    if dry_run:
        return [float(job["initial_learning_rate"])]
    rows = _read_json(run_dir / "metrics" / "training_metrics.json")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("WGAN LR trace requires epoch 0 and a learned epoch")
    trace: list[dict[str, float | int]] = []
    previous_g = float(job["initial_learning_rate"])
    previous_d = float(job["initial_learning_rate"])
    floor = float(job["scheduler_min_lr"])
    learned_gp: list[float] = []
    for expected_epoch, raw in enumerate(rows):
        row = _require_mapping(raw, f"WGAN metric {job['job_id']}")
        epoch = int(row["epoch"])
        g_lr, d_lr = float(row["g_lr"]), float(row["d_lr"])
        if epoch != expected_epoch:
            raise ValueError("WGAN epochs must be contiguous from epoch 0")
        if not (floor <= g_lr <= previous_g and floor <= d_lr <= previous_d):
            raise ValueError("WGAN LR trace violates its monotone floor contract")
        for field in (
            "val_recon",
            "val_current_recon",
            "val_baseline_gap",
            "val_hybrid_score",
            "val_calendar",
            "val_butterfly",
            "val_delta_shrink",
            "g_lr",
            "d_lr",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite WGAN metric: epoch={epoch} {field}")
        if epoch == 0:
            if not math.isclose(
                float(row["val_recon"]),
                float(row["val_current_recon"]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError("Epoch-0 generator is not the persistence origin")
        else:
            for field in (
                "d_total",
                "d_real",
                "d_fake",
                "gp",
                "g_total",
                "g_adv",
                "g_recon",
                "g_calendar",
                "g_butterfly",
                "g_smooth",
                "g_delta_shrink",
            ):
                if not math.isfinite(float(row[field])):
                    raise ValueError(
                        f"Non-finite learned WGAN metric: epoch={epoch} {field}"
                    )
            learned_gp.append(float(row["gp"]))
        trace.append({"epoch": epoch, "g_lr": g_lr, "d_lr": d_lr})
        previous_g, previous_d = g_lr, d_lr
    if trace[0] != {
        "epoch": 0,
        "g_lr": float(job["initial_learning_rate"]),
        "d_lr": float(job["initial_learning_rate"]),
    }:
        raise ValueError("WGAN LR trace has an invalid epoch-0 origin")
    if int(trace[-1]["epoch"]) < 30:
        raise ValueError("WGAN stopped before the frozen minimum 30 epochs")
    if (
        not learned_gp
        or min(learned_gp) <= 0.0
        or max(learned_gp) - min(learned_gp) <= 1.0e-12
    ):
        raise ValueError("Gradient-penalty trace is degenerate")
    return trace


def _generator_from_checkpoint(checkpoint: Mapping[str, Any]) -> Generator:
    config = _require_mapping(checkpoint.get("config"), "generator config")
    model = Generator(
        channels=int(config["channels"]),
        embedding_dim=int(
            checkpoint.get("embedding_dim", config.get("embedding_dim", 1024))
        ),
        noise_dim=int(config["noise_dim"]),
        surface_height=16,
        surface_width=16,
        base_channels=int(config["gen_base_channels"]),
        res_blocks=int(config["gen_res_blocks"]),
        text_hidden_dim=int(config["gen_text_hidden_dim"]),
        text_out_dim=int(config["gen_text_out_dim"]),
        hidden_dim=int(config["gen_hidden_dim"]),
        residual_output_mode=str(config["residual_output_mode"]),
        generator_noise_mode=str(config["generator_noise_mode"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _zero_noise_invariance_max_abs(checkpoint: Mapping[str, Any]) -> float:
    model = _generator_from_checkpoint(checkpoint)
    embedding_dim = int(model.text_encoder[0].in_features)
    current = torch.linspace(0.1, 0.3, 2 * 16 * 16).reshape(2, 1, 16, 16)
    text = torch.linspace(-0.2, 0.2, 2 * embedding_dim).reshape(2, embedding_dim)
    zeros = torch.zeros((2, NOISE_DIM))
    attempted_nonzero = torch.linspace(-3.0, 3.0, 2 * NOISE_DIM).reshape(2, NOISE_DIM)
    with torch.no_grad():
        baseline = model(current, text, noise=zeros)
        attempted = model(current, text, noise=attempted_nonzero)
    return float(torch.max(torch.abs(baseline - attempted)).item())


def _validate_run_contract(
    root: Path, job: Mapping[str, Any], run_dir: Path, *, dry_run: bool
) -> tuple[list[dict[str, float | int]] | list[float], Path | None]:
    trace = _validated_wgan_lr_trace(job, run_dir, dry_run=dry_run)
    if dry_run:
        return trace, None
    best_path = run_dir / "metrics" / "best_learned_checkpoint.json"
    best = _require_mapping(
        _read_json(best_path), f"best learned checkpoint for {job['job_id']}"
    )
    if int(best.get("best_epoch", -1)) < 1 or int(
        best.get("best_learned_epoch_ge_1", best.get("best_epoch", -1))
    ) != int(best.get("best_epoch", -1)):
        raise ValueError("Zero-noise comparison requires best-learned epoch >=1")
    if str(best.get("selection_scope", "")) != "trained_epochs_only":
        raise ValueError("Zero-noise best learned selection scope drifted")
    expected_fingerprint = generator_noise_fingerprint(GENERATOR_NOISE_MODE, NOISE_DIM)
    if (
        str(best.get("generator_noise_mode", "")) != GENERATOR_NOISE_MODE
        or str(best.get("generator_noise_fingerprint", "")) != expected_fingerprint
    ):
        raise ValueError("Best learned metadata lost zero-noise lineage")
    artifacts = _require_mapping(best.get("artifacts"), "learned artifacts")
    generator_path = Path(str(artifacts.get("generator", "")))
    discriminator_path = Path(str(artifacts.get("discriminator", "")))
    if not generator_path.is_file() or not discriminator_path.is_file():
        raise FileNotFoundError("Learned zero-noise WGAN checkpoint is missing")
    generator = torch.load(generator_path, map_location="cpu", weights_only=False)
    discriminator = torch.load(
        discriminator_path, map_location="cpu", weights_only=False
    )
    for checkpoint, label, expected_parameters in (
        (generator, "generator", 123_472),
        (discriminator, "discriminator", 25_861),
    ):
        mode, fingerprint = resolve_checkpoint_generator_noise_contract(checkpoint)
        if mode != GENERATOR_NOISE_MODE or fingerprint != expected_fingerprint:
            raise ValueError(f"{label} checkpoint noise contract drifted")
        raw_config = _require_mapping(checkpoint.get("config"), f"{label} config")
        if int(raw_config.get("seed", -1)) != FROZEN_SEED:
            raise ValueError(f"{label} checkpoint seed drifted")
        if int(raw_config.get("noise_dim", -1)) != NOISE_DIM:
            raise ValueError(f"{label} checkpoint noise width drifted")
        if _checkpoint_parameter_count(checkpoint) != expected_parameters:
            raise ValueError(f"{label} checkpoint parameter count drifted")
    if _checkpoint_parameter_count(generator) + _checkpoint_parameter_count(
        discriminator
    ) != int(job["expected_wgan_parameters"]):
        raise ValueError("Small WGAN total parameter count drifted")
    initial_generator_path = run_dir / "checkpoints" / "generator_initial_epoch0.pt"
    initial_discriminator_path = (
        run_dir / "checkpoints" / "discriminator_initial_epoch0.pt"
    )
    final_generator_path = run_dir / "checkpoints" / "generator.pt"
    for path in (
        initial_generator_path,
        initial_discriminator_path,
        final_generator_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    initial_generator = torch.load(
        initial_generator_path, map_location="cpu", weights_only=False
    )
    initial_discriminator = torch.load(
        initial_discriminator_path, map_location="cpu", weights_only=False
    )
    final_generator = torch.load(
        final_generator_path, map_location="cpu", weights_only=False
    )
    initial_g_sha = _checkpoint_state_sha256(initial_generator)
    initial_d_sha = _checkpoint_state_sha256(initial_discriminator)
    if (
        initial_g_sha != EXPECTED_INITIAL_GENERATOR_STATE_SHA256
        or str(job["gaussian_initial_generator_state_sha256"]) != initial_g_sha
    ):
        raise ValueError("Zero/Gaussian initial generator tensors differ")
    if (
        initial_d_sha != EXPECTED_INITIAL_DISCRIMINATOR_STATE_SHA256
        or str(job["gaussian_initial_discriminator_state_sha256"]) != initial_d_sha
    ):
        raise ValueError("Zero/Gaussian initial discriminator tensors differ")
    reference = _validate_reference_manifest(root)[
        (str(job["text_ablation_mode"]), int(job["tolerance_minutes"]))
    ]
    zero_initial_metadata = _read_json(run_dir / "metrics" / "initial_checkpoint.json")
    gaussian_initial_metadata = _read_json(Path(reference["initial_metadata_path"]))
    zero_initial_metrics = _require_mapping(
        zero_initial_metadata.get("metrics"), "zero epoch-0 metrics"
    )
    gaussian_initial_metrics = _require_mapping(
        gaussian_initial_metadata.get("metrics"), "Gaussian epoch-0 metrics"
    )
    if set(zero_initial_metrics) != set(gaussian_initial_metrics):
        raise ValueError("Zero/Gaussian epoch-0 metric fields differ")
    epoch0_metric_max_abs = max(
        abs(float(zero_initial_metrics[key]) - float(gaussian_initial_metrics[key]))
        for key in zero_initial_metrics
    )
    if epoch0_metric_max_abs > 1.0e-8:
        raise ValueError("Zero/Gaussian epoch-0 validation metrics differ materially")
    if int(zero_initial_metadata.get("best_epoch", -1)) != 0:
        raise ValueError("Zero initial-checkpoint metadata is not epoch 0")
    initial_noise_columns = initial_generator["state_dict"]["fusion.0.weight"][
        :, -NOISE_DIM:
    ]
    best_noise_columns = generator["state_dict"]["fusion.0.weight"][:, -NOISE_DIM:]
    final_noise_columns = final_generator["state_dict"]["fusion.0.weight"][
        :, -NOISE_DIM:
    ]
    if int(initial_noise_columns.numel()) != 4_096:
        raise ValueError("Small generator must have 4,096 latent fusion weights")
    if not torch.equal(initial_noise_columns, best_noise_columns) or not torch.equal(
        initial_noise_columns, final_noise_columns
    ):
        raise ValueError("Zero-mode latent fusion weights changed during training")
    invariance_max_abs = _zero_noise_invariance_max_abs(generator)
    if invariance_max_abs != 0.0:
        raise ValueError("Explicit nonzero z changed a zero-mode prediction")
    fallback_path = run_dir / "metrics" / "fallback_calibration.json"
    if fallback_path.is_file():
        fallback = _read_json(fallback_path)
        if str(fallback.get("generator_noise_mode", "")) != GENERATOR_NOISE_MODE:
            raise ValueError("Fallback calibration noise mode drifted")
        if str(fallback.get("generator_noise_fingerprint", "")) != expected_fingerprint:
            raise ValueError("Fallback calibration fingerprint drifted")
        if int(fallback.get("effective_mc_samples", -1)) != 1:
            raise ValueError("Zero-noise fallback calibration must be a single pass")
    audit = {
        "schema_version": 1,
        "job_id": job["job_id"],
        "generator_noise_mode": GENERATOR_NOISE_MODE,
        "generator_noise_fingerprint": expected_fingerprint,
        "effective_q3_mc_samples": 1,
        "initial_generator_state_sha256": initial_g_sha,
        "initial_discriminator_state_sha256": initial_d_sha,
        "matching_gaussian_reference_job_id": job["gaussian_reference_job_id"],
        "matching_gaussian_epoch0_metrics_within_tolerance": True,
        "matching_gaussian_epoch0_metric_max_abs_difference": (epoch0_metric_max_abs),
        "matching_gaussian_epoch0_metric_tolerance": 1.0e-8,
        "divergence_first_allowed_epoch": 1,
        "latent_fusion_weight_count": 4_096,
        "latent_fusion_weights_frozen_initial_to_best_learned": True,
        "latent_fusion_weights_frozen_initial_to_final": True,
        "attempted_nonzero_z_prediction_max_abs_difference": invariance_max_abs,
        "best_learned_epoch": int(best["best_epoch"]),
        "validated_at_utc": _utc_now(),
    }
    audit["audit_sha256"] = _payload_sha256(audit)
    audit_path = _write_json(
        run_dir / "metrics" / "zero_noise_contract_audit.json", audit
    )
    return trace, audit_path


def run_zero_noise_worker(
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
        raise KeyError(f"Unknown or duplicate zero-noise job: {job_id}")
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
    if visible and visible.split(",")[0].strip() != str(job["gpu_id"]):
        raise RuntimeError("Worker CUDA_VISIBLE_DEVICES disagrees with registry")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
    resolved = _validate_resolved_snapshot(root)
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
        "experiment_stage": EXPERIMENT_STAGE,
        "generator_noise_mode": GENERATOR_NOISE_MODE,
        "generator_noise_fingerprint": job["generator_noise_fingerprint"],
        "noise_profile_sha256": job["noise_profile_sha256"],
        "gaussian_reference_job_id": job["gaussian_reference_job_id"],
        "gaussian_reference_manifest_sha256": job["gaussian_reference_manifest_sha256"],
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "dry_run": bool(dry_run),
        "log_path": str(os.environ.get("NEWS_FIRST_JOB_LOG_PATH", "")),
    }
    _write_json(status_path, {**common, "status": "running"})
    try:
        run_dir, artifacts = training._execute_training_job(job, dry_run=dry_run)
        trace, audit_path = _validate_run_contract(root, job, run_dir, dry_run=dry_run)
        if audit_path is not None:
            artifacts = list(artifacts) + [
                {
                    "artifact_role": "zero_noise_contract_audit",
                    "path": str(audit_path),
                    "size_bytes": audit_path.stat().st_size,
                    "sha256": _sha256_file(audit_path),
                }
            ]
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


def build_zero_noise_worker_command(
    config_path: str | Path,
    root: str | Path,
    job: Mapping[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> list[str]:
    experiment_root = Path(root)
    resolved = _validate_resolved_snapshot(experiment_root)
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
            "-m",
            "scripts.rq3.news_first_vol_zero_noise_ablation",
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


def _jobs_for_wave(root: Path, *, resume: bool, dry_run: bool) -> list[dict[str, Any]]:
    jobs = [dict(job) for job in _load_registry(root)["jobs"]]
    selected: list[dict[str, Any]] = []
    for job in jobs:
        _validate_job_lineage(root, job)
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        if not dry_run and status.get("status") == "completed":
            if _completed_job_is_valid(root, job, status):
                if resume:
                    continue
                raise RuntimeError(f"Completed job requires --resume: {job['job_id']}")
            if not resume:
                raise RuntimeError(
                    f"Invalid completed job requires --resume: {job['job_id']}"
                )
        if status.get("status") == "running" and training._pid_is_live(
            status.get("pid")
        ):
            raise RuntimeError(f"Refusing duplicate live job: {job['job_id']}")
        if status.get("status") in {"running", "failed"} and not (resume or dry_run):
            raise RuntimeError(f"Interrupted job requires --resume: {job['job_id']}")
        selected.append(job)
    return selected


def _run_wave(
    root: Path,
    config_path: str | Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
    resume: bool,
) -> None:
    if not jobs:
        return
    resolved = _validate_resolved_snapshot(root)
    runtime = resolved["runtime"]
    monitor = training._ResourceMonitor(
        root / "resource_usage.csv",
        executable=str(runtime.get("nvidia_smi_executable", "nvidia-smi")),
        interval_seconds=float(runtime.get("resource_sample_interval_seconds", 2)),
        wave=1,
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
                build_zero_noise_worker_command(
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
                        f"Zero-noise job {jobs[index]['job_id']} exited {code}"
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
        raise RuntimeError(f"Zero-noise wave did not complete: {failures}")


RESOURCE_FIELDS = training.RESOURCE_SUMMARY_FIELDS + (
    "generator_noise_mode",
    "generator_noise_fingerprint",
    "noise_profile_sha256",
    "gaussian_reference_job_id",
    "gaussian_generator_checkpoint_sha256",
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
        row.update(
            {
                "generator_noise_mode": job["generator_noise_mode"],
                "generator_noise_fingerprint": job["generator_noise_fingerprint"],
                "noise_profile_sha256": job["noise_profile_sha256"],
                "gaussian_reference_job_id": job["gaussian_reference_job_id"],
                "gaussian_generator_checkpoint_sha256": job[
                    "gaussian_generator_checkpoint_sha256"
                ],
                "seed": job["seed"],
                "experiment_stage": EXPERIMENT_STAGE,
            }
        )
    return _write_csv(root / "resource_summary.csv", rows, RESOURCE_FIELDS)


def _run_postprocess(root: Path) -> None:
    from scripts.rq3.news_first_vol_zero_noise_analysis import (
        run_zero_noise_analysis,
    )
    from scripts.rq3.news_first_vol_zero_noise_report import (
        render_zero_noise_report,
    )

    run_zero_noise_analysis(root)
    render_zero_noise_report(root)


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


def launch_zero_noise_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
    postprocess_hook: Callable[[Path], None] | None = None,
) -> Path:
    root = prepare_zero_noise_experiment(config_path, output_dir, reuse=True)
    previous = _read_json(root / "registry" / "experiment_status.json")
    if previous.get("status") in TERMINAL_EXPERIMENT_STATES and not dry_run:
        if not resume:
            raise RuntimeError("Terminal zero-noise experiment requires --resume")
        if previous.get("status") == "completed_q3_only":
            _validate_registry(root)
            refresh_zero_noise_lineage(config_path, root)
            _refresh_exports(root)
            return root
    if previous.get("status") in {"running", "failed"} and not (resume or dry_run):
        raise RuntimeError("Interrupted zero-noise experiment requires --resume")
    _mark_status(root, "dry_running" if dry_run else "running")
    try:
        jobs = _jobs_for_wave(root, resume=resume, dry_run=dry_run)
        _run_wave(
            root,
            config_path,
            jobs,
            dry_run=dry_run,
            resume=resume,
        )
        _validate_wave_completion(root, jobs, dry_run=dry_run)
        _write_resource_summary(root)
        _refresh_exports(root)
        all_jobs = [dict(job) for job in _load_registry(root)["jobs"]]
        failures = []
        for job in all_jobs:
            status = _read_json(_job_status_path(root, str(job["job_id"])))
            valid = (
                status.get("status") == "dry_run_passed"
                if dry_run
                else _completed_job_is_valid(root, job, status)
            )
            if not valid:
                failures.append(f"{job['job_id']}={status.get('status', 'missing')}")
        if failures:
            raise RuntimeError(f"Zero-noise matrix incomplete: {failures}")
        if dry_run:
            _mark_status(root, "dry_run_passed", completed_at_utc=_utc_now())
        else:
            (postprocess_hook or _run_postprocess)(root)
            _mark_status(root, "completed_q3_only", completed_at_utc=_utc_now())
    except BaseException as exc:
        _mark_status(root, "failed", error=f"{type(exc).__name__}: {exc}")
        refresh_zero_noise_lineage(config_path, root)
        _refresh_exports(root)
        raise
    _write_resource_summary(root)
    refresh_zero_noise_lineage(config_path, root)
    _refresh_exports(root)
    return root


def run_news_first_vol_zero_noise_ablation(
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
        return prepare_zero_noise_experiment(
            config_path, output_dir, reuse=bool(reuse or resume)
        )
    if action == "dry-run":
        return launch_zero_noise_experiment(
            config_path, output_dir, resume=resume, dry_run=True
        )
    if action == "worker":
        if not job_id:
            raise ValueError("--job-id is required for worker")
        return run_zero_noise_worker(
            output_dir, job_id, dry_run=worker_dry_run, resume=resume
        )
    if action == "launch":
        return launch_zero_noise_experiment(
            config_path,
            output_dir,
            resume=resume,
            dry_run=False,
            postprocess_hook=postprocess_hook,
        )
    raise ValueError(f"Unsupported zero-noise action: {action}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "dry-run", "worker", "launch"))
    parser.add_argument(
        "--config",
        default="configs/rq3/news_first_vol_zero_noise_ablation.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--worker-dry-run", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = _parser().parse_args(argv)
    output = run_news_first_vol_zero_noise_ablation(
        args.config,
        args.output_dir,
        action=args.action,
        job_id=args.job_id,
        resume=bool(args.resume),
        reuse=bool(args.reuse),
        worker_dry_run=bool(args.worker_dry_run),
    )
    print(output)
    return output


if __name__ == "__main__":
    main()


__all__ = [
    "CAPACITY_PROFILE",
    "EXPERIMENT_KIND",
    "EXPERIMENT_STAGE",
    "FIXED_LEARNING_RATE",
    "FIXED_SCHEDULER_MIN_LR",
    "FROZEN_SEED",
    "FROZEN_TEXT_MODES",
    "FROZEN_TOLERANCES",
    "GENERATOR_NOISE_MODE",
    "LR_PROFILE",
    "NOISE_DIM",
    "REFERENCE_JOB_IDS",
    "REFERENCE_NOISE_MODE",
    "build_zero_noise_worker_command",
    "launch_zero_noise_experiment",
    "prepare_zero_noise_experiment",
    "refresh_zero_noise_lineage",
    "run_news_first_vol_zero_noise_ablation",
    "run_zero_noise_worker",
]
