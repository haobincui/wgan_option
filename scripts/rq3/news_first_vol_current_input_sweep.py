"""Formal Q3-only WGAN current-support conditioning experiment.

The six masked-input jobs are trained in a new immutable root.  Each job is
paired with an already completed full-current coverage run having the same
seed and tolerance.  ``prepare`` freezes those references and ``dry-run``
exercises the exact 4+2 GPU schedule without starting optimization.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import gzip
import hashlib
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml

from scripts.rq3 import news_first_vol_training as training
from scripts.rq3.news_first_vol_coverage_sweep import (
    _capacity_seed_profile_sha256 as _coverage_capacity_seed_profile_sha256,
    _lr_profile_sha256 as _coverage_lr_profile_sha256,
    _qa_sha256,
)
from wgan_option.models.common import (
    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    FULL_CURRENT_GENERATOR_INPUT_MODE,
    GAUSSIAN_GENERATOR_NOISE_MODE,
    generator_current_input_fingerprint,
    generator_noise_fingerprint,
)
from wgan_option.utils.inference_helpers import (
    resolve_checkpoint_generator_current_input_contract,
    resolve_checkpoint_generator_noise_contract,
)
from wgan_option.utils.text_ablation import REAL_TEXT, text_information_path


ROOT_KEY = training.ROOT_KEY
REPO_ROOT = training.REPO_ROOT
EXPERIMENT_KIND = "current_support_masked_seed_sweep"
EXPERIMENT_STAGE = "current_support_masked_wgan_small_three_seed"
CAPACITY_PROFILE = "small"
LR_PROFILE = "lr_5e_07"
FIXED_LEARNING_RATE = 5.0e-7
FIXED_SCHEDULER_MIN_LR = 5.0e-8
FROZEN_SEEDS = (42, 202, 404)
FROZEN_TOLERANCES = (5, 30)
TEXT_ABLATION_MODE = REAL_TEXT
GENERATOR_NOISE_MODE = GAUSSIAN_GENERATOR_NOISE_MODE
GENERATOR_CURRENT_INPUT_MODE = CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
REFERENCE_CURRENT_INPUT_MODE = FULL_CURRENT_GENERATOR_INPUT_MODE
NOISE_DIM = 32
EXPECTED_Q3_PAIRS = 123
EXPECTED_Q3_SESSIONS = 33
REFERENCE_STAGE_ID = "stage_3_wgan_low_lr_capacity"
REFERENCE_REGISTRY_SHA256 = (
    "348466bbf9d74709e8b91e24232279c5333ad71df4cecf58d3ee408db2ba9d3e"
)
REFERENCE_RESOLVED_CONFIG_SHA256 = (
    "58b0c712ed04ae80d3cb53b8f86aaff55c3305b84c76fa10d5ab8a3db49c0661"
)
REFERENCE_Q3_PAIR_METRICS_SHA256 = (
    "eb49b862485e370460ccf23295e184715016c19037c58099a9a55328a6738eab"
)
REFERENCE_STAGE_QA_SHA256 = (
    "cb0aa1816c7c097739b9bc4abeec6ddcfc289bfcef90994c0fcca984af2999d7"
)
REFERENCE_JOB_IDS = tuple(
    f"s3_wgan_small_lr_5e_07_seed_{seed:03d}_real_text_{tolerance:02d}m"
    for seed in FROZEN_SEEDS
    for tolerance in FROZEN_TOLERANCES
)
ALLOWED_POSTPROCESS_CODE_DRIFT_PATHS = frozenset(
    {
        "scripts/rq3/news_first_vol_current_input_sweep.py",
        "scripts/rq3/news_first_vol_current_input_analysis.py",
        "scripts/rq3/news_first_vol_current_input_report.py",
    }
)
CURRENT_CODE_MANIFEST_FILENAME = "code_hashes_current.csv"
POSTPROCESS_CODE_DRIFT_LEDGER_FILENAME = "postprocess_code_drift.json"
PRE_POSTPROCESS_OUTPUT_HASH_MANIFEST_FILENAME = "output_hashes_pre_postprocess.csv"

# Exact content identities of the matched coverage artifacts.  Paths remain
# configurable so a repository checkout can move; bytes and experimental axes
# cannot drift.
EXPECTED_REFERENCE_HASHES: dict[str, dict[str, str]] = {
    "s3_wgan_small_lr_5e_07_seed_042_real_text_05m": {
        "config": "553bb3dda9df81fb92aaa6a8979d599ab73973fd8acdbc05d21dc88b83d1e4d8",
        "status": "d5fea5386b774a97bd2bac3ab8fc238da2f3cc3f7c3e5d21aa9bea5a0ebb1fc5",
        "generator": "62738b3a6965c5f49a0932d8e272dcfe5fa8db8d158f6d5ef808571b3fd64dca",
        "discriminator": "82530707692d16e85f282b0c6c423aa5557a4a433806b096ea97b7d37803f1e5",
        "metadata": "1260ade2a358dcb4a305726fe7a5b66133297335dd63c484096b8a4d2f0e90e2",
        "generator_initial": "60a201f4b57827d82f3ba03f874ca0abf1fcec0561f44d2ea490b995ee82102c",
        "discriminator_initial": "57d63240e6716f55604eee164a4832dc2b6780dfdc5de6a31d5564b73083db1d",
        "initial_metadata": "8995955acb2cf32d79897340fc104a32639013a9236aeae842cc0c6bcf1c79e5",
        "pair_panel": "58140554e649c4e3c7d567746b60af1b51dcdd9ba6366e4154effb4b5360adc1",
    },
    "s3_wgan_small_lr_5e_07_seed_042_real_text_30m": {
        "config": "3ea456a8ef828df31fb7212e00e956cdc19b5fd6cce0167525149b78e6f6341d",
        "status": "1a85f57d7a48cc2ea5a0778a38acadddb6dec0e52530c7599202d716525d3291",
        "generator": "a220276c10dd0b5383fc65f90232bae4a491983a5da748994ac7425b6e61a7df",
        "discriminator": "b2919daa702fa4c759592e8f53761490632c2890125dbc6623e9245ceb6441a1",
        "metadata": "3743884339d49bdd52ada1d4f3a632f6a9b53c3fc759362d120cd42fe502a611",
        "generator_initial": "85e0e8e7b07fd8cd7406ff844a0ca8b7ec199f9617dbd802635eca7e371b7734",
        "discriminator_initial": "d91d510c9520c89d97b4778af2a9f2bc7a721446f68d89f54b106ecc5be8aad4",
        "initial_metadata": "89e22598f12a1d2a442a745e609717bc2c47487fa121df6538035cbf6fd5244a",
        "pair_panel": "5d77ca25b25b3d8bcdf9d20d0cadad14957dcc605000566a03952932740b57fb",
    },
    "s3_wgan_small_lr_5e_07_seed_202_real_text_05m": {
        "config": "dc168d1ec5cfd5b38c6125384ffb25f29a375cc0fa5302dc75fbc0c80723cf60",
        "status": "f33dc22e03156443573f8d441f2b93670066d9f81ba289eb08d2456f6379aba4",
        "generator": "7b9a315cace92532ef2f1fcb86968b84f7e6dbb4109d40bb4b882ebae332cc41",
        "discriminator": "ee513f455103271d77f916f9da5cfde0ddc76f1305fb949b30dfcb4368314475",
        "metadata": "f3da931f59088ce22e1efc59bdc4ae776299dfb96f9e52547a70ce2521d9edb6",
        "generator_initial": "16a859abf129738f8997030e033530f151cd249773ed76ecb676235f6459b3bf",
        "discriminator_initial": "a8f02065ad24f2f20f6b54c7aa304b7e3a00e39f394bcdcb13da61cac5b58044",
        "initial_metadata": "345e2a90f529cfae3faaa96beb64bd6fe054680e397da72b1c64fa6a1f8e9890",
        "pair_panel": "46663bb47e90f19a5f83f30d07c7d849feed6f4c88783ec635d29cf580c8d238",
    },
    "s3_wgan_small_lr_5e_07_seed_202_real_text_30m": {
        "config": "e1aded894d83cc7793cf616156c2bd676a96844e61a19316d71b12ffe95945a5",
        "status": "72f577ae7c8193054aa01b84496b212ad1c85d607671a3064a5ede41f3858cc9",
        "generator": "fb0d2067492be40aa6b8142ea93cb3b649264c20e24a76b218c0c97143d25b3a",
        "discriminator": "bf711aec4e35e4918125e3addeafe49606246647125bddd34b7cce6d4272978b",
        "metadata": "919bcc06b7d48abacd2a0490f5a80b1d787e9fdfa4aa1291e9df0b95f2fead06",
        "generator_initial": "2c357ec6d4394f8d39d0f0f38c051732a051ac3e8f784e901de265b89c9c503f",
        "discriminator_initial": "9892c80af803650091c1a2285ff08ddb295368994908e8a74473b5a127887af5",
        "initial_metadata": "e28433de430fb51fefbd94c38065d7a0de59cf914710249de34355d4b14a41d2",
        "pair_panel": "b3afdb631663354620f5570c9e938ca5fa1e65234a24185b1529c4421f43c6d0",
    },
    "s3_wgan_small_lr_5e_07_seed_404_real_text_05m": {
        "config": "24be1dc2d5a49c7606ab355a9871eb199a151b639442824db5df303644e50e81",
        "status": "42ea77d836f7d1e3f616bc655361768e23fa97dc1e0f87a279cd3ebe93b83862",
        "generator": "2677917e5eb4627fe380f4b7a6373b54cbcf08e169b91d41fc150ee277d821cf",
        "discriminator": "82f6186db4a0822a23a01da8de3aad3b105dd8388c6d96d653b3558eb18cd3bf",
        "metadata": "3b5f69351ba01c42b5edf5b46a1746eb36f7ed468a50ea2ce1f03d67fd3b2437",
        "generator_initial": "c50a7063dc3b25d803c673ddc898fc61bd0a87d2cda54c03edd94bf054cecde8",
        "discriminator_initial": "7f0a1378994791c229b5a78b4c0257b122e9d02279fb37e58362591be6653064",
        "initial_metadata": "c2499e5562b482879b87a8ab8da20a6218303c07de6d2d1b2f00bd917ea2a76a",
        "pair_panel": "dcd08e28cf8c2ffe3a65e34f6b4edb279ea3183f1a2d0d9f9d61dc42743587ec",
    },
    "s3_wgan_small_lr_5e_07_seed_404_real_text_30m": {
        "config": "7092f2779e50efe6759c1fdf0dd3c145decac2708a58a2ae6e7e564175abf4b5",
        "status": "1ebb599597a0e0d4b79985cfe92a1a20ebc2d15cf94f6595b6405a917d2acd1b",
        "generator": "f955e2bc276742d937b8ba4a4a902da29258dd7dbeadeda2388ade0e69185537",
        "discriminator": "418546853ef6d5223dc1ae92bb8f26bce0256bb7699675f8a0d4b38e3e4b5f03",
        "metadata": "cee7912158267a8fafb1285b52057a886e6a94be9084798a625e58df3748105d",
        "generator_initial": "2be9b1f03fcea67a8729fae1bb48d4402b60b34aeda2a467b2d6fdbd8caf5951",
        "discriminator_initial": "6371f36c1faff1952158ae4a6d960af6578fec803411765b2ca7f7a2b4656bc8",
        "initial_metadata": "9425244c0a28e5c02e396358a968d95ed7a566a466445e911dfb1ead743f55a4",
        "pair_panel": "880c7cfc51d29faf43cfc86efb26491f14d8c4cd82f53f8f3248f6992a63633c",
    },
}

EXPECTED_INITIAL_STATE_HASHES = {
    42: (
        "ec724ab94c5f66de246b7e396cf8283ed2967a43989c818a30f89605cd7bc077",
        "5da9039c0cf6ff046097ab94ae0a0a8596762046717366967683f5bb909bb0b8",
    ),
    202: (
        "380a80af83bb8fd37e2f7fef238971c53d58852bbf6a99a05944e03122396460",
        "fea93c830cf421a33d1e022514792583675f420316de5f34c52778b2e18ed101",
    ),
    404: (
        "8e5b810dd0948c2b3d4be405e73919366fd71f3acce01e8454818ab9a0bf5dd4",
        "ff3077252c530e7f0c9c49e3f07515f088d767b2bac8ca5374182c25840ff985",
    ),
}

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


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _require_mapping(payload, label)


def _exact_float(value: Any, expected: float, label: str) -> None:
    observed = float(value)
    if not math.isfinite(observed) or observed != expected:
        raise ValueError(f"{label} is frozen to {expected}; observed={observed}")


def _sweep_config(config: Mapping[str, Any]) -> dict[str, Any]:
    sweep = _require_mapping(
        config.get("current_support_masked_seed_sweep"),
        "current_support_masked_seed_sweep",
    )
    if not bool(sweep.get("enabled", False)):
        raise ValueError("current_support_masked_seed_sweep.enabled must be true")
    return sweep


def _validate_frozen_config(config: Mapping[str, Any]) -> None:
    datasets = _require_mapping(config.get("datasets"), "datasets")
    split = _require_mapping(config.get("split"), "split")
    sweep = _sweep_config(config)
    models = _require_mapping(config.get("models"), "models")
    runtime = _require_mapping(config.get("runtime"), "runtime")

    if tuple(int(value) for value in datasets.get("tolerances_minutes", ())) != (
        5,
        10,
        15,
        30,
    ):
        raise ValueError("dataset tolerances are frozen to 5/10/15/30")
    dataset_contract = {
        "common_evaluation_tolerance_minutes": 5,
        "sheet_name": "gan_input_ready",
        "text_embedding_mode": "lp",
        "seed": 42,
        "support_mask_mode": "raw_joint",
    }
    for field, expected in dataset_contract.items():
        if datasets.get(field) != expected:
            raise ValueError(f"datasets.{field} is frozen to {expected}")
    if tuple(datasets.get("text_ablation_modes", ())) != (REAL_TEXT,):
        raise ValueError("Only real_text is permitted")
    if str(split.get("train_end_utc")) != "2023-07-01T00:00:00Z":
        raise ValueError("train boundary drifted")
    if str(split.get("validation_end_utc")) != "2023-10-01T00:00:00Z":
        raise ValueError("validation boundary drifted")
    if int(split.get("validation_mc_samples", -1)) != 16:
        raise ValueError("validation_mc_samples must match the Gaussian reference (16)")

    sweep_contract = {
        "experiment_kind": EXPERIMENT_KIND,
        "experiment_stage": EXPERIMENT_STAGE,
        "capacity_profile": CAPACITY_PROFILE,
        "lr_profile": LR_PROFILE,
        "text_ablation_mode": REAL_TEXT,
        "generator_noise_mode": GENERATOR_NOISE_MODE,
        "noise_dim": NOISE_DIM,
        "generator_current_input_mode": GENERATOR_CURRENT_INPUT_MODE,
        "reference_generator_current_input_mode": REFERENCE_CURRENT_INPUT_MODE,
        "expected_q3_pairs": EXPECTED_Q3_PAIRS,
        "expected_q3_sessions": EXPECTED_Q3_SESSIONS,
        "q4_evaluator_calls": 0,
    }
    for field, expected in sweep_contract.items():
        if sweep.get(field) != expected:
            raise ValueError(f"sweep.{field} is frozen to {expected}")
    if tuple(int(value) for value in sweep.get("seeds", ())) != FROZEN_SEEDS:
        raise ValueError(f"seeds are frozen to {list(FROZEN_SEEDS)}")
    if tuple(int(value) for value in sweep.get("tolerances_minutes", ())) != (
        FROZEN_TOLERANCES
    ):
        raise ValueError(f"tolerances are frozen to {list(FROZEN_TOLERANCES)}")
    if not bool(sweep.get("q4_prediction_and_evaluation_forbidden", False)):
        raise ValueError("Q4 prediction/evaluation must remain forbidden")
    _exact_float(sweep.get("initial_learning_rate"), FIXED_LEARNING_RATE, "LR")
    _exact_float(
        sweep.get("scheduler_min_lr"), FIXED_SCHEDULER_MIN_LR, "scheduler floor"
    )
    _exact_float(
        sweep.get("epoch0_metric_absolute_tolerance"), 1.0e-8, "epoch-0 tolerance"
    )

    if set(models) != {"wgan"}:
        raise ValueError("Only WGAN is permitted")
    model = _require_mapping(models["wgan"], "models.wgan")
    if model.get("trainer_command") != "vol-xlsx":
        raise ValueError("trainer command must remain vol-xlsx")
    values = _require_mapping(model.get("training"), "models.wgan.training")
    expected_training_fields = {
        "train_ratio",
        "channels",
        "embedding_dim",
        "noise_dim",
        "generator_noise_mode",
        "generator_current_input_mode",
        "gen_base_channels",
        "gen_res_blocks",
        "gen_text_hidden_dim",
        "gen_text_out_dim",
        "gen_hidden_dim",
        "disc_base_channels",
        "disc_res_blocks",
        "disc_text_hidden_dim",
        "disc_hidden_dim",
        "residual_output_mode",
        "learning_rate",
        "use_reduce_lr_on_plateau",
        "reduce_lr_factor",
        "reduce_lr_patience",
        "reduce_lr_min_lr",
        "num_epochs",
        "batch_size",
        "beta_1",
        "beta_2",
        "discriminator_iter",
        "lambda_gp",
        "lambda_recon",
        "lambda_calendar",
        "lambda_butterfly",
        "lambda_smooth",
        "lambda_delta_shrink",
        "use_calendar_constraint",
        "use_butterfly_constraint",
        "use_smooth_constraint",
        "constraint_warmup_epochs",
        "best_checkpoint_metric",
        "evaluate_initial_checkpoint",
        "baseline_penalty_weight",
        "use_early_stopping",
        "early_stopping_patience",
        "early_stopping_min_epochs",
        "early_stopping_min_delta",
        "save_every",
    }
    if set(values) != expected_training_fields:
        raise ValueError(
            "WGAN training fields drifted: "
            f"missing={sorted(expected_training_fields - set(values))}, "
            f"extra={sorted(set(values) - expected_training_fields)}"
        )
    frozen_values: dict[str, Any] = {
        "channels": 1,
        "embedding_dim": 1024,
        "noise_dim": 32,
        "generator_noise_mode": GENERATOR_NOISE_MODE,
        "generator_current_input_mode": GENERATOR_CURRENT_INPUT_MODE,
        "gen_base_channels": 4,
        "gen_res_blocks": 0,
        "gen_text_hidden_dim": 32,
        "gen_text_out_dim": 16,
        "gen_hidden_dim": 128,
        "disc_base_channels": 4,
        "disc_res_blocks": 0,
        "disc_text_hidden_dim": 16,
        "disc_hidden_dim": 96,
        "residual_output_mode": "identity_softplus_residual",
        "num_epochs": 100,
        "batch_size": 16,
        "discriminator_iter": 5,
        "constraint_warmup_epochs": 0,
        "best_checkpoint_metric": "val_hybrid_score",
        "early_stopping_patience": 16,
        "early_stopping_min_epochs": 30,
        "reduce_lr_patience": 3,
    }
    for field, expected in frozen_values.items():
        if values.get(field) != expected:
            raise ValueError(f"WGAN {field} is frozen to {expected}")
    for field in (
        "use_reduce_lr_on_plateau",
        "use_calendar_constraint",
        "use_butterfly_constraint",
        "use_smooth_constraint",
        "evaluate_initial_checkpoint",
        "use_early_stopping",
    ):
        if not bool(values.get(field, False)):
            raise ValueError(f"WGAN {field} must remain true")
    for field, expected in {
        "train_ratio": 0.8,
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
        "early_stopping_min_delta": 0.0,
    }.items():
        _exact_float(values.get(field), expected, f"WGAN {field}")
    if int(values.get("save_every", 0)) != 1_000_000:
        raise ValueError("save_every is frozen to best/final-only checkpoints")

    gpu_ids = tuple(int(value) for value in runtime.get("gpu_ids", ()))
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("Exactly two distinct GPUs are required")
    if int(runtime.get("slots_per_gpu", -1)) != 2:
        raise ValueError("The formal sweep is frozen to two workers per GPU")
    if int(runtime.get("cpu_threads_per_job", -1)) != 8:
        raise ValueError("Each worker is frozen to eight CPU threads")


def _resolved_config(config_path: str | Path) -> dict[str, Any]:
    path = _resolve_repo_path(config_path)
    root = _load_yaml_mapping(path, "current-input sweep config")
    config = _require_mapping(root.get(ROOT_KEY), ROOT_KEY)
    _validate_frozen_config(config)
    resolved = deepcopy(config)
    resolved["datasets"]["root"] = str(_resolve_repo_path(resolved["datasets"]["root"]))
    reference = resolved["current_support_masked_seed_sweep"]["full_current_reference"]
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
    return training._capacity_profile_sha256(
        CAPACITY_PROFILE, training.FROZEN_CAPACITY_PROFILES[CAPACITY_PROFILE]
    )


def _lr_profile_sha256() -> str:
    return _coverage_lr_profile_sha256(LR_PROFILE)


def _capacity_seed_profile_sha256(seed: int) -> str:
    if int(seed) not in FROZEN_SEEDS:
        raise ValueError(f"Unknown seed: {seed}")
    return _coverage_capacity_seed_profile_sha256(
        CAPACITY_PROFILE, LR_PROFILE, int(seed), "wgan"
    )


def _job_id(seed: int, tolerance: int) -> str:
    return (
        f"current_support_masked_wgan_small_lr_5e_07_seed_{int(seed):03d}_"
        f"real_text_{int(tolerance):02d}m"
    )


def _job_specs() -> tuple[dict[str, int], ...]:
    # Preserve the matched coverage run's physical-GPU lane (5m on GPU0, 30m
    # on GPU1) while balancing seeds/worker count: four jobs then two jobs.
    return (
        {"seed": 42, "tolerance_minutes": 5, "wave": 1, "gpu_index": 0, "gpu_slot": 0},
        {"seed": 404, "tolerance_minutes": 5, "wave": 1, "gpu_index": 0, "gpu_slot": 1},
        {"seed": 42, "tolerance_minutes": 30, "wave": 1, "gpu_index": 1, "gpu_slot": 0},
        {
            "seed": 404,
            "tolerance_minutes": 30,
            "wave": 1,
            "gpu_index": 1,
            "gpu_slot": 1,
        },
        {"seed": 202, "tolerance_minutes": 5, "wave": 2, "gpu_index": 0, "gpu_slot": 0},
        {
            "seed": 202,
            "tolerance_minutes": 30,
            "wave": 2,
            "gpu_index": 1,
            "gpu_slot": 0,
        },
    )


def _tensor_state_sha256(state: Mapping[str, Any]) -> str:
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


def _reference_pair_rows(path: Path, job_id: str) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("run_id") == job_id]
    rows.sort(
        key=lambda row: (
            row.get("panel", ""),
            row.get("stratum_type", ""),
            row.get("stratum_value", ""),
            row.get("pair_id", ""),
            row.get("session_id", ""),
        )
    )
    if len(rows) != EXPECTED_Q3_PAIRS:
        raise ValueError(f"Reference Q3 pair panel is not 123 rows: {job_id}")
    if len({row["pair_id"] for row in rows}) != EXPECTED_Q3_PAIRS:
        raise ValueError(f"Reference Q3 pair IDs drifted: {job_id}")
    if len({row["session_id"] for row in rows}) != EXPECTED_Q3_SESSIONS:
        raise ValueError(f"Reference Q3 session IDs drifted: {job_id}")
    return rows


def _shared_training_contract_sha256(payload: Mapping[str, Any]) -> str:
    canonical = deepcopy(dict(payload))
    canonical.pop("output_root", None)
    canonical.pop("generator_current_input_mode", None)
    # The historical configs predate this explicit field; absence has always
    # meant Gaussian sampling.  Normalize semantics before comparison.
    canonical["generator_noise_mode"] = GAUSSIAN_GENERATOR_NOISE_MODE
    return _payload_sha256(canonical)


def _reference_rows(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    import torch

    reference = resolved["current_support_masked_seed_sweep"]["full_current_reference"]
    root = Path(str(reference["root"]))
    registry_path = Path(str(reference["registry"]))
    resolved_hash_path = Path(str(reference["resolved_config_hash"]))
    pair_path = Path(str(reference["q3_pair_metrics"]))
    qa_path = Path(str(reference["stage_qa"]))
    for path in (registry_path, resolved_hash_path, pair_path, qa_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256_file(registry_path) != REFERENCE_REGISTRY_SHA256:
        raise ValueError("Frozen full-current registry hash changed")
    if (
        resolved_hash_path.read_text(encoding="utf-8").strip()
        != REFERENCE_RESOLVED_CONFIG_SHA256
    ):
        raise ValueError("Frozen full-current resolved-config hash changed")
    if _sha256_file(pair_path) != REFERENCE_Q3_PAIR_METRICS_SHA256:
        raise ValueError("Frozen full-current Q3 pair-metric hash changed")
    if _sha256_file(qa_path) != REFERENCE_STAGE_QA_SHA256:
        raise ValueError("Frozen full-current Stage 3 QA hash changed")
    qa = _read_json(qa_path)
    if (
        qa.get("status") != "pass"
        or bool(qa.get("dry_run", True))
        or qa.get("stage_id") != REFERENCE_STAGE_ID
        or qa.get("qa_sha256") != _qa_sha256(qa)
    ):
        raise ValueError("Frozen full-current Stage 3 QA is invalid")
    registry = _read_json(registry_path)
    if registry.get("experiment_kind") != "coverage_completion_sweep":
        raise ValueError("Reference registry kind mismatch")
    jobs = {str(job["job_id"]): dict(job) for job in registry.get("jobs", [])}
    configured_ids = tuple(str(value) for value in reference.get("job_ids", ()))
    if configured_ids != REFERENCE_JOB_IDS:
        raise ValueError("Reference job IDs differ from the frozen six-cell order")
    if not set(REFERENCE_JOB_IDS).issubset(jobs):
        raise ValueError("Reference registry lacks a frozen job")

    rows: list[dict[str, Any]] = []
    panel_key_sets: list[frozenset[tuple[str, str]]] = []
    for job_id in REFERENCE_JOB_IDS:
        job = jobs[job_id]
        seed = int(job.get("seed", -1))
        tolerance = int(job.get("tolerance_minutes", -1))
        expected_id = (
            f"s3_wgan_small_lr_5e_07_seed_{seed:03d}_real_text_{tolerance:02d}m"
        )
        contracts = {
            "stage_id": REFERENCE_STAGE_ID,
            "model_family": "wgan",
            "capacity_profile": CAPACITY_PROFILE,
            "lr_profile": LR_PROFILE,
            "text_ablation_mode": REAL_TEXT,
            "support_mask_mode": "raw_joint",
        }
        if job_id != expected_id or seed not in FROZEN_SEEDS:
            raise ValueError(f"Reference axes drifted: {job_id}")
        if tolerance not in FROZEN_TOLERANCES:
            raise ValueError(f"Reference tolerance drifted: {job_id}")
        if any(job.get(field) != expected for field, expected in contracts.items()):
            raise ValueError(f"Reference contract drifted: {job_id}")

        config_path = Path(str(job["training_config_path"]))
        status_path = root / "registry" / "jobs" / f"{job_id}.status.json"
        if not config_path.is_file() or not status_path.is_file():
            raise FileNotFoundError(f"Reference config/status missing: {job_id}")
        status = _read_json(status_path)
        if status.get("status") != "completed":
            raise ValueError(f"Reference job is not completed: {job_id}")
        artifacts = {
            str(row["artifact_role"]): dict(row) for row in status.get("artifacts", [])
        }
        required = {
            "generator_best_learned",
            "discriminator_best_learned",
            "best_learned_checkpoint",
            "generator_initial_epoch0",
            "discriminator_initial_epoch0",
            "initial_checkpoint",
        }
        if not required.issubset(artifacts):
            raise ValueError(f"Reference artifacts missing: {job_id}")
        expected = EXPECTED_REFERENCE_HASHES[job_id]
        observed = {
            "config": _sha256_file(config_path),
            "status": _sha256_file(status_path),
            "generator": str(artifacts["generator_best_learned"]["sha256"]),
            "discriminator": str(artifacts["discriminator_best_learned"]["sha256"]),
            "metadata": str(artifacts["best_learned_checkpoint"]["sha256"]),
            "generator_initial": str(artifacts["generator_initial_epoch0"]["sha256"]),
            "discriminator_initial": str(
                artifacts["discriminator_initial_epoch0"]["sha256"]
            ),
            "initial_metadata": str(artifacts["initial_checkpoint"]["sha256"]),
        }
        for role, artifact in artifacts.items():
            path = Path(str(artifact.get("path", "")))
            if not path.is_file() or _sha256_file(path) != str(
                artifact.get("sha256", "")
            ):
                raise ValueError(f"Reference artifact hash drifted: {job_id}/{role}")
        pair_rows = _reference_pair_rows(pair_path, job_id)
        observed["pair_panel"] = _payload_sha256(pair_rows)
        if observed != expected:
            raise ValueError(f"Exact reference hashes drifted: {job_id}")
        panel_key_sets.append(
            frozenset((row["pair_id"], row["session_id"]) for row in pair_rows)
        )

        generator_path = Path(artifacts["generator_best_learned"]["path"])
        discriminator_path = Path(artifacts["discriminator_best_learned"]["path"])
        generator = torch.load(generator_path, map_location="cpu", weights_only=False)
        discriminator = torch.load(
            discriminator_path, map_location="cpu", weights_only=False
        )
        for checkpoint, label in (
            (generator, "generator"),
            (discriminator, "discriminator"),
        ):
            input_mode, input_fingerprint = (
                resolve_checkpoint_generator_current_input_contract(checkpoint)
            )
            noise_mode, _ = resolve_checkpoint_generator_noise_contract(checkpoint)
            if (
                input_mode != REFERENCE_CURRENT_INPUT_MODE
                or input_fingerprint
                != generator_current_input_fingerprint(REFERENCE_CURRENT_INPUT_MODE)
                or noise_mode != GENERATOR_NOISE_MODE
            ):
                raise ValueError(f"Historical {label} contract drifted: {job_id}")
        initial_generator_path = Path(artifacts["generator_initial_epoch0"]["path"])
        initial_discriminator_path = Path(
            artifacts["discriminator_initial_epoch0"]["path"]
        )
        initial_g_state_sha = _checkpoint_state_sha256(
            torch.load(initial_generator_path, map_location="cpu", weights_only=False)
        )
        initial_d_state_sha = _checkpoint_state_sha256(
            torch.load(
                initial_discriminator_path, map_location="cpu", weights_only=False
            )
        )
        if (initial_g_state_sha, initial_d_state_sha) != EXPECTED_INITIAL_STATE_HASHES[
            seed
        ]:
            raise ValueError(f"Reference epoch-0 tensor identity drifted: {job_id}")
        reference_config = _load_yaml_mapping(config_path, f"reference {job_id}")
        initial_metadata_path = Path(artifacts["initial_checkpoint"]["path"])
        best_metadata_path = Path(artifacts["best_learned_checkpoint"]["path"])
        best_metadata = _read_json(best_metadata_path)
        best_learned_epoch = int(
            best_metadata.get(
                "best_learned_epoch_ge_1", best_metadata.get("best_epoch", -1)
            )
        )
        if best_learned_epoch < 1:
            raise ValueError(f"Reference has no learned epoch: {job_id}")
        rows.append(
            {
                "seed": seed,
                "tolerance_minutes": tolerance,
                "reference_job_id": job_id,
                "reference_root": str(root),
                "reference_registry_path": str(registry_path),
                "reference_registry_sha256": REFERENCE_REGISTRY_SHA256,
                "reference_stage_qa_path": str(qa_path),
                "reference_stage_qa_sha256": REFERENCE_STAGE_QA_SHA256,
                "reference_resolved_config_hash_path": str(resolved_hash_path),
                "reference_resolved_config_sha256": (REFERENCE_RESOLVED_CONFIG_SHA256),
                "reference_status_path": str(status_path),
                "reference_status_sha256": observed["status"],
                "reference_training_config_path": str(config_path),
                "reference_training_config_sha256": observed["config"],
                "reference_shared_training_contract_sha256": (
                    _shared_training_contract_sha256(reference_config)
                ),
                "best_learned_metadata_path": str(
                    artifacts["best_learned_checkpoint"]["path"]
                ),
                "best_learned_metadata_sha256": observed["metadata"],
                "best_learned_epoch": best_learned_epoch,
                "generator_checkpoint_path": str(generator_path),
                "generator_checkpoint_sha256": observed["generator"],
                "discriminator_checkpoint_path": str(discriminator_path),
                "discriminator_checkpoint_sha256": observed["discriminator"],
                "initial_generator_checkpoint_path": str(initial_generator_path),
                "initial_generator_checkpoint_sha256": observed["generator_initial"],
                "initial_generator_state_sha256": initial_g_state_sha,
                "initial_discriminator_checkpoint_path": str(
                    initial_discriminator_path
                ),
                "initial_discriminator_checkpoint_sha256": observed[
                    "discriminator_initial"
                ],
                "initial_discriminator_state_sha256": initial_d_state_sha,
                "initial_metadata_path": str(initial_metadata_path),
                "initial_metadata_sha256": observed["initial_metadata"],
                "reference_q3_pair_metrics_path": str(pair_path),
                "reference_q3_pair_metrics_sha256": (REFERENCE_Q3_PAIR_METRICS_SHA256),
                "reference_q3_pair_panel_sha256": observed["pair_panel"],
                "reference_q3_pair_count": EXPECTED_Q3_PAIRS,
                "reference_q3_session_count": EXPECTED_Q3_SESSIONS,
                "generator_current_input_mode": REFERENCE_CURRENT_INPUT_MODE,
                "generator_current_input_fingerprint": (
                    generator_current_input_fingerprint(REFERENCE_CURRENT_INPUT_MODE)
                ),
                "generator_noise_mode": GENERATOR_NOISE_MODE,
                "generator_noise_fingerprint": generator_noise_fingerprint(
                    GENERATOR_NOISE_MODE, NOISE_DIM
                ),
            }
        )
    if len(rows) != 6 or len(set(panel_key_sets)) != 1:
        raise ValueError("Reference matrix must be six runs on one common Q3 panel")
    return rows


def _reference_by_cell(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(row["seed"]), int(row["tolerance_minutes"])): dict(row) for row in rows
    }


def _training_payload(
    reference: Mapping[str, Any], root: Path, *, seed: int, tolerance: int
) -> dict[str, Any]:
    reference_path = Path(str(reference["reference_training_config_path"]))
    payload = deepcopy(_load_yaml_mapping(reference_path, "reference training config"))
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError("Reference training config seed mismatch")
    if int(payload.get("news_first_dataset_tolerance_minutes", -1)) != int(tolerance):
        raise ValueError("Reference training config tolerance mismatch")
    if str(payload.get("news_first_text_ablation_mode", "")) != REAL_TEXT:
        raise ValueError("Reference training config is not real_text")
    payload["generator_noise_mode"] = GENERATOR_NOISE_MODE
    payload["generator_current_input_mode"] = GENERATOR_CURRENT_INPUT_MODE
    payload["output_root"] = str(
        root
        / "runs"
        / "wgan"
        / CAPACITY_PROFILE
        / LR_PROFILE
        / f"seed_{int(seed):03d}"
        / REAL_TEXT
        / f"tolerance_{int(tolerance):02d}m"
    )
    if _shared_training_contract_sha256(payload) != str(
        reference["reference_shared_training_contract_sha256"]
    ):
        raise ValueError("Masked payload changed more than the current-input contract")
    return payload


def _source_rows(
    resolved: Mapping[str, Any], reference_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = training._source_rows(resolved)
    seen = {str(row["path"]) for row in rows}
    reference = resolved["current_support_masked_seed_sweep"]["full_current_reference"]
    candidates: list[tuple[str, str]] = [
        ("full_current_reference_registry", str(reference["registry"])),
        ("full_current_reference_stage_qa", str(reference["stage_qa"])),
        (
            "full_current_reference_resolved_config_hash",
            str(reference["resolved_config_hash"]),
        ),
        ("full_current_reference_q3_pair_metrics", str(reference["q3_pair_metrics"])),
    ]
    artifact_fields = {
        "reference_status_path": "status",
        "reference_training_config_path": "training_config",
        "best_learned_metadata_path": "best_learned_metadata",
        "generator_checkpoint_path": "generator_best_learned",
        "discriminator_checkpoint_path": "discriminator_best_learned",
        "initial_generator_checkpoint_path": "generator_initial_epoch0",
        "initial_discriminator_checkpoint_path": "discriminator_initial_epoch0",
        "initial_metadata_path": "initial_metadata",
    }
    for row in reference_rows:
        job_id = str(row["reference_job_id"])
        for field, role in artifact_fields.items():
            candidates.append((f"full_current_{job_id}_{role}", str(row[field])))
    for role, raw_path in candidates:
        path = Path(raw_path)
        if raw_path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "source_role": role,
                "path": raw_path,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
        seen.add(raw_path)
    return rows


def _code_rows() -> list[dict[str, Any]]:
    rows = training._code_rows()
    known = {str(row["relative_path"]) for row in rows}
    for relative in (
        "scripts/rq3/news_first_vol_current_input_sweep.py",
        "scripts/rq3/news_first_vol_current_input_analysis.py",
        "scripts/rq3/news_first_vol_current_input_report.py",
    ):
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required current-input code is missing: {path}")
        if relative in known:
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
        _read_json(root / "registry" / "jobs.json"), "current-input registry"
    )
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment root is not a current-support masked sweep")
    return registry


def _validate_resolved_snapshot(root: Path) -> dict[str, Any]:
    resolved = _require_mapping(
        _load_yaml_mapping(root / "resolved_config.yaml", "resolved config").get(
            ROOT_KEY
        ),
        ROOT_KEY,
    )
    _validate_frozen_config(resolved)
    observed = _payload_sha256(resolved)
    expected = (
        (root / "registry" / "resolved_config.sha256")
        .read_text(encoding="utf-8")
        .strip()
    )
    if observed != expected:
        raise ValueError("Resolved current-input config hash mismatch")
    if str(_load_registry(root).get("resolved_config_sha256", "")) != expected:
        raise ValueError("Registry resolved-config hash mismatch")
    return resolved


def _validate_reference_manifest(
    root: Path,
) -> dict[tuple[int, int], dict[str, str]]:
    path = root / "full_current_reference_manifest.csv"
    registry = _load_registry(root)
    expected_sha = str(registry.get("full_current_reference_manifest_sha256", ""))
    if not path.is_file() or _sha256_file(path) != expected_sha:
        raise ValueError("Full-current reference manifest hash mismatch")
    companion = root / "full_current_reference_manifest.sha256"
    if (
        not companion.is_file()
        or companion.read_text(encoding="utf-8").strip() != expected_sha
    ):
        raise ValueError("Full-current reference manifest companion hash mismatch")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        raise ValueError("Full-current reference manifest must contain six rows")
    by_cell: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        seed = int(row["seed"])
        tolerance = int(row["tolerance_minutes"])
        key = (seed, tolerance)
        if (
            key in by_cell
            or seed not in FROZEN_SEEDS
            or tolerance not in FROZEN_TOLERANCES
        ):
            raise ValueError("Reference manifest contains duplicate/unknown axes")
        expected_job_id = (
            f"s3_wgan_small_lr_5e_07_seed_{seed:03d}_real_text_{tolerance:02d}m"
        )
        if row["reference_job_id"] != expected_job_id:
            raise ValueError("Reference manifest job ID/axes mismatch")
        if row["generator_current_input_mode"] != REFERENCE_CURRENT_INPUT_MODE or row[
            "generator_current_input_fingerprint"
        ] != generator_current_input_fingerprint(REFERENCE_CURRENT_INPUT_MODE):
            raise ValueError("Reference manifest is not explicitly full-current")
        path_hash_fields = (
            ("reference_registry_path", "reference_registry_sha256"),
            ("reference_stage_qa_path", "reference_stage_qa_sha256"),
            ("reference_status_path", "reference_status_sha256"),
            (
                "reference_training_config_path",
                "reference_training_config_sha256",
            ),
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
            (
                "reference_q3_pair_metrics_path",
                "reference_q3_pair_metrics_sha256",
            ),
        )
        for path_field, hash_field in path_hash_fields:
            artifact = Path(row[path_field])
            if not artifact.is_file() or _sha256_file(artifact) != row[hash_field]:
                raise ValueError(
                    f"Reference manifest artifact drifted: {expected_job_id}/{path_field}"
                )
        expected_hashes = EXPECTED_REFERENCE_HASHES[expected_job_id]
        checks = {
            "config": row["reference_training_config_sha256"],
            "status": row["reference_status_sha256"],
            "generator": row["generator_checkpoint_sha256"],
            "discriminator": row["discriminator_checkpoint_sha256"],
            "metadata": row["best_learned_metadata_sha256"],
            "generator_initial": row["initial_generator_checkpoint_sha256"],
            "discriminator_initial": row["initial_discriminator_checkpoint_sha256"],
            "initial_metadata": row["initial_metadata_sha256"],
            "pair_panel": row["reference_q3_pair_panel_sha256"],
        }
        if checks != expected_hashes:
            raise ValueError(
                f"Reference manifest exact hashes drifted: {expected_job_id}"
            )
        by_cell[key] = row
    if set(by_cell) != {
        (seed, tolerance) for seed in FROZEN_SEEDS for tolerance in FROZEN_TOLERANCES
    }:
        raise ValueError("Reference manifest matrix is incomplete")
    return by_cell


def _validate_source_manifest(root: Path, job: Mapping[str, Any]) -> None:
    path = root / "source_hashes.csv"
    if _sha256_file(path) != str(job.get("source_manifest_sha256", "")):
        raise ValueError(f"Source-manifest hash mismatch: {job['job_id']}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        source = Path(str(row["path"]))
        if not source.is_file() or _sha256_file(source) != str(row["sha256"]):
            raise ValueError(
                f"Source hash mismatch: {job['job_id']}/{row['source_role']}"
            )


def _validate_code_manifest(root: Path) -> None:
    path = root / "code_hashes.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Code manifest is empty")
    for row in rows:
        source = Path(str(row["path"]))
        if not source.is_file() or _sha256_file(source) != str(row["sha256"]):
            raise ValueError(f"Experiment code drifted: {row['relative_path']}")


CODE_MANIFEST_FIELDS = ("relative_path", "path", "size_bytes", "sha256")


def _read_code_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CODE_MANIFEST_FIELDS:
            raise ValueError(f"Code manifest schema drifted: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Code manifest is empty: {path}")
    relative_paths = [str(row.get("relative_path", "")) for row in rows]
    if "" in relative_paths or len(relative_paths) != len(set(relative_paths)):
        raise ValueError(f"Code manifest paths are empty or duplicated: {path}")
    for row in rows:
        try:
            size_bytes = int(row["size_bytes"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Code manifest size is invalid: {path}") from exc
        digest = str(row["sha256"])
        if (
            size_bytes < 0
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Code manifest content is invalid: {path}")
        row["relative_path"] = str(row["relative_path"])
        row["path"] = str(row["path"])
        row["size_bytes"] = size_bytes
        row["sha256"] = digest
    return rows


def _compare_postprocess_code_ledgers(
    prepared_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    prepared = {str(row["relative_path"]): dict(row) for row in prepared_rows}
    current = {str(row["relative_path"]): dict(row) for row in current_rows}
    if len(prepared) != len(prepared_rows) or len(current) != len(current_rows):
        raise ValueError("Prepared/current code ledger contains duplicate paths")
    if set(prepared) != set(current):
        raise ValueError(
            "Prepared/current code path sets differ: "
            f"missing={sorted(set(prepared) - set(current))}, "
            f"extra={sorted(set(current) - set(prepared))}"
        )
    changed: list[str] = []
    for relative in sorted(prepared):
        before, after = prepared[relative], current[relative]
        if Path(str(before["path"])).resolve(strict=False) != Path(
            str(after["path"])
        ).resolve(strict=False):
            raise ValueError(f"Code path target drifted: {relative}")
        if str(before["sha256"]) == str(after["sha256"]) and int(
            before["size_bytes"]
        ) == int(after["size_bytes"]):
            continue
        if relative not in ALLOWED_POSTPROCESS_CODE_DRIFT_PATHS:
            raise ValueError(f"Forbidden postprocess code drift: {relative}")
        changed.append(relative)
    return changed


def _validate_prepared_code_ledger_anchor(root: Path, prepared_sha256: str) -> None:
    lineage = _validate_self_hashed_json(
        root / "experiment_lineage.json", "lineage_sha256"
    )
    if str(lineage.get("code_manifest_sha256", "")) != prepared_sha256:
        raise ValueError("Prepared code ledger differs from frozen experiment lineage")
    output_path = root / "output_hashes.csv"
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    expected_path = str((root / "code_hashes.csv").resolve())
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if str(Path(str(row["path"])).resolve(strict=False)) == expected_path
        ]
    if len(matches) != 1 or str(matches[0].get("sha256", "")) != prepared_sha256:
        raise ValueError("Prepared code ledger lost its frozen output-hash anchor")


def _postprocess_code_snapshot(root: Path, *, persist: bool) -> dict[str, Any]:
    prepared_path = root / "code_hashes.csv"
    prepared_rows = _read_code_manifest(prepared_path)
    prepared_sha = _sha256_file(prepared_path)
    _validate_prepared_code_ledger_anchor(root, prepared_sha)

    current_rows = _code_rows()
    changed_paths = _compare_postprocess_code_ledgers(prepared_rows, current_rows)
    current_path = root / CURRENT_CODE_MANIFEST_FILENAME
    if current_path.is_file():
        saved_current_rows = _read_code_manifest(current_path)
        if saved_current_rows != current_rows:
            raise ValueError(
                "Current postprocess code ledger changed after it was frozen"
            )
    elif persist:
        _write_csv(current_path, current_rows, CODE_MANIFEST_FIELDS)
    else:
        raise FileNotFoundError(current_path)
    current_sha = _sha256_file(current_path)
    output_manifest = root / "output_hashes.csv"
    preserved_output_manifest = root / PRE_POSTPROCESS_OUTPUT_HASH_MANIFEST_FILENAME
    if preserved_output_manifest.is_file():
        if not preserved_output_manifest.read_text(encoding="utf-8").strip():
            raise ValueError("Preserved pre-postprocess output ledger is empty")
    elif persist:
        _atomic_write_text(
            preserved_output_manifest,
            output_manifest.read_text(encoding="utf-8"),
        )
    else:
        raise FileNotFoundError(preserved_output_manifest)
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "prepared_code_manifest_path": str(prepared_path),
        "prepared_code_manifest_sha256": prepared_sha,
        "current_code_manifest_path": str(current_path),
        "current_code_manifest_sha256": current_sha,
        "pre_postprocess_output_hash_manifest_path": str(preserved_output_manifest),
        "pre_postprocess_output_hash_manifest_sha256": _sha256_file(
            preserved_output_manifest
        ),
        "allowed_postprocess_code_drift_paths": sorted(
            ALLOWED_POSTPROCESS_CODE_DRIFT_PATHS
        ),
        "changed_paths": changed_paths,
        "code_changed_since_prepared_snapshot": bool(changed_paths),
    }
    snapshot["postprocess_code_drift_ledger_sha256"] = _payload_sha256(snapshot)
    ledger_path = root / POSTPROCESS_CODE_DRIFT_LEDGER_FILENAME
    if ledger_path.is_file():
        saved = _read_json(ledger_path)
        if saved != snapshot:
            raise ValueError(
                "Postprocess code-drift ledger changed after it was frozen"
            )
    elif persist:
        _write_json(ledger_path, snapshot)
    else:
        raise FileNotFoundError(ledger_path)
    return {
        **snapshot,
        "postprocess_code_drift_ledger_path": str(ledger_path),
        "postprocess_code_drift_ledger_file_sha256": _sha256_file(ledger_path),
    }


def _postprocess_status_fields(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the immutable dual-ledger snapshot into status/lineage fields."""

    return {
        "prepared_code_manifest_path": str(snapshot["prepared_code_manifest_path"]),
        "prepared_code_manifest_sha256": str(snapshot["prepared_code_manifest_sha256"]),
        "current_code_manifest_path": str(snapshot["current_code_manifest_path"]),
        "current_code_manifest_sha256": str(snapshot["current_code_manifest_sha256"]),
        "pre_postprocess_output_hash_manifest_path": str(
            snapshot["pre_postprocess_output_hash_manifest_path"]
        ),
        "pre_postprocess_output_hash_manifest_sha256": str(
            snapshot["pre_postprocess_output_hash_manifest_sha256"]
        ),
        "postprocess_allowed_code_drift_paths": list(
            snapshot["allowed_postprocess_code_drift_paths"]
        ),
        "postprocess_code_changed_paths": list(snapshot["changed_paths"]),
        "code_changed_since_prepared_snapshot": bool(
            snapshot["code_changed_since_prepared_snapshot"]
        ),
        "postprocess_code_drift_ledger_path": str(
            snapshot["postprocess_code_drift_ledger_path"]
        ),
        "postprocess_code_drift_ledger_payload_sha256": str(
            snapshot["postprocess_code_drift_ledger_sha256"]
        ),
        "postprocess_code_drift_ledger_sha256": str(
            snapshot["postprocess_code_drift_ledger_file_sha256"]
        ),
    }


def _validate_job_lineage(root: Path, job: Mapping[str, Any]) -> None:
    if str(job.get("job_spec_sha256", "")) != _job_spec_sha256(job):
        raise ValueError(f"Job-spec hash mismatch: {job.get('job_id')}")
    seed = int(job.get("seed", -1))
    tolerance = int(job.get("tolerance_minutes", -1))
    if seed not in FROZEN_SEEDS or tolerance not in FROZEN_TOLERANCES:
        raise ValueError(f"Out-of-contract job axes: {job.get('job_id')}")
    if str(job.get("job_id")) != _job_id(seed, tolerance):
        raise ValueError("Job ID does not encode immutable axes")
    expected_contracts: dict[str, Any] = {
        "experiment_stage": EXPERIMENT_STAGE,
        "model_family": "wgan",
        "trainer_command": "vol-xlsx",
        "capacity_profile": CAPACITY_PROFILE,
        "capacity_profile_sha256": _capacity_profile_sha256(),
        "capacity_seed_profile_sha256": _capacity_seed_profile_sha256(seed),
        "expected_wgan_parameters": 149_333,
        "lr_profile": LR_PROFILE,
        "lr_profile_sha256": _lr_profile_sha256(),
        "initial_learning_rate": FIXED_LEARNING_RATE,
        "scheduler_min_lr": FIXED_SCHEDULER_MIN_LR,
        "text_ablation_mode": REAL_TEXT,
        "text_information_path": text_information_path(REAL_TEXT),
        "support_mask_mode": "raw_joint",
        "generator_noise_mode": GENERATOR_NOISE_MODE,
        "generator_noise_fingerprint": generator_noise_fingerprint(
            GENERATOR_NOISE_MODE, NOISE_DIM
        ),
        "generator_current_input_mode": GENERATOR_CURRENT_INPUT_MODE,
        "generator_current_input_fingerprint": (
            generator_current_input_fingerprint(GENERATOR_CURRENT_INPUT_MODE)
        ),
        "q4_evaluator_calls": 0,
    }
    for field, expected in expected_contracts.items():
        if job.get(field) != expected:
            raise ValueError(f"Job lineage mismatch: {job['job_id']}/{field}")
    registry = _load_registry(root)
    manifest_sha = str(registry["full_current_reference_manifest_sha256"])
    if str(job.get("full_current_reference_manifest_sha256", "")) != manifest_sha:
        raise ValueError("Job lost its immutable full-current manifest link")
    reference = _validate_reference_manifest(root)[(seed, tolerance)]
    if str(job.get("full_current_reference_job_id", "")) != str(
        reference["reference_job_id"]
    ):
        raise ValueError("Job links the wrong full-current cell")
    _validate_source_manifest(root, job)

    config_path = Path(str(job["training_config_path"]))
    if not config_path.is_file() or _sha256_file(config_path) != str(
        job["config_sha256"]
    ):
        raise ValueError(f"Training config hash mismatch: {job['job_id']}")
    payload = _load_yaml_mapping(config_path, f"training config {job['job_id']}")
    if payload.get("generator_current_input_mode") != GENERATOR_CURRENT_INPUT_MODE:
        raise ValueError("Masked training config lost its input mode")
    if payload.get("generator_noise_mode") != GENERATOR_NOISE_MODE:
        raise ValueError("Masked training config lost Gaussian noise")
    if payload.get("support_mask_mode") != "raw_joint":
        raise ValueError("Masked training config lost raw_joint support")
    if _shared_training_contract_sha256(payload) != str(
        reference["reference_shared_training_contract_sha256"]
    ):
        raise ValueError("Training config differs from reference beyond input mode")
    expected_output = (
        root
        / "runs"
        / "wgan"
        / CAPACITY_PROFILE
        / LR_PROFILE
        / f"seed_{seed:03d}"
        / REAL_TEXT
        / f"tolerance_{tolerance:02d}m"
    )
    if Path(str(job["output_root"])) != expected_output:
        raise ValueError("Job output root drifted")
    if Path(str(payload["output_root"])) != expected_output:
        raise ValueError("Training output root drifted")
    dataset_path = Path(str(job["dataset_path"]))
    if not dataset_path.is_file() or _sha256_file(dataset_path) != str(
        job["dataset_sha256"]
    ):
        raise ValueError("Training workbook hash drifted")


def _validate_registry(root: Path) -> None:
    registry = _load_registry(root)
    jobs = [dict(job) for job in registry.get("jobs", [])]
    expected_ids = {
        _job_id(seed, tolerance)
        for seed in FROZEN_SEEDS
        for tolerance in FROZEN_TOLERANCES
    }
    observed_ids = {str(job.get("job_id")) for job in jobs}
    if len(jobs) != 6 or observed_ids != expected_ids:
        raise ValueError("Registry must contain exactly the frozen six-job matrix")
    if sorted(int(job["wave"]) for job in jobs) != [1, 1, 1, 1, 2, 2]:
        raise ValueError("Registry must preserve the 4+2 wave schedule")
    if int(registry.get("q4_evaluator_calls", -1)) != 0:
        raise ValueError("Registry records a forbidden Q4 evaluator")
    counts: dict[int, int] = {}
    for job in jobs:
        counts[int(job["gpu_id"])] = counts.get(int(job["gpu_id"]), 0) + 1
        expected_gpu_index = 0 if int(job["tolerance_minutes"]) == 5 else 1
        resolved = _validate_resolved_snapshot(root)
        expected_gpu = int(resolved["runtime"]["gpu_ids"][expected_gpu_index])
        if int(job["gpu_id"]) != expected_gpu:
            raise ValueError("Physical GPU lane differs from the matched reference")
        _validate_job_lineage(root, job)
    if sorted(counts.values()) != [3, 3]:
        raise ValueError("Both GPUs must receive exactly three jobs")


def _validate_split_manifest(root: Path) -> None:
    with (root / "split_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["tolerance_minutes"]) in FROZEN_TOLERANCES
        ]
    if {int(row["tolerance_minutes"]) for row in rows} != set(FROZEN_TOLERANCES):
        raise ValueError("Split manifest lacks the exact 5m/30m rows")
    expected_train = {5: (936, 720, 210), 30: (1591, 1050, 237)}
    overlap_fields = (
        "train_validation_pair_overlap",
        "train_test_pair_overlap",
        "validation_test_pair_overlap",
        "train_validation_session_overlap",
        "train_test_session_overlap",
        "validation_test_session_overlap",
    )
    for row in rows:
        tolerance = int(row["tolerance_minutes"])
        if (
            tuple(
                int(row[field])
                for field in ("train_rows", "train_pairs", "train_sessions")
            )
            != expected_train[tolerance]
        ):
            raise ValueError(f"Training split count drifted for {tolerance}m")
        if tuple(
            int(row[field])
            for field in ("validation_rows", "validation_pairs", "validation_sessions")
        ) != (133, EXPECTED_Q3_PAIRS, EXPECTED_Q3_SESSIONS):
            raise ValueError(f"Q3 split count drifted for {tolerance}m")
        if tuple(
            int(row[field]) for field in ("test_rows", "test_pairs", "test_sessions")
        ) != (152, 130, 45):
            raise ValueError(f"Q4 audit split count drifted for {tolerance}m")
        if any(int(row[field]) != 0 for field in overlap_fields):
            raise ValueError(f"Split leakage detected for {tolerance}m")
        if row["support_mask_mode"] != "raw_joint" or row["status"] != "pass":
            raise ValueError(f"Split support/status drifted for {tolerance}m")


TASK_FIELDS = (
    "job_id",
    "status",
    "attempt",
    "wave",
    "seed",
    "tolerance_minutes",
    "gpu_id",
    "gpu_slot",
    "capacity_profile",
    "lr_profile",
    "generator_noise_mode",
    "generator_noise_fingerprint",
    "generator_current_input_mode",
    "generator_current_input_fingerprint",
    "full_current_reference_job_id",
    "full_current_reference_manifest_sha256",
    "config_sha256",
    "job_spec_sha256",
    "run_dir",
    "exit_code",
)


def _refresh_exports(root: Path) -> Path:
    registry = _load_registry(root)
    rows = []
    for job in registry["jobs"]:
        status_path = _job_status_path(root, str(job["job_id"]))
        status = _read_json(status_path) if status_path.is_file() else {}
        rows.append(
            {field: status.get(field, job.get(field, "")) for field in TASK_FIELDS}
        )
    return _write_csv(root / "task_registry.csv", rows, TASK_FIELDS)


def _write_config_hashes(root: Path) -> Path:
    rows = [
        {
            "config_role": "resolved_orchestration",
            "path": str(root / "resolved_config.yaml"),
            "sha256": _sha256_file(root / "resolved_config.yaml"),
        },
        {
            "config_role": "full_current_reference_manifest",
            "path": str(root / "full_current_reference_manifest.csv"),
            "sha256": _sha256_file(root / "full_current_reference_manifest.csv"),
        },
        {
            "config_role": "full_current_reference_manifest_hash",
            "path": str(root / "full_current_reference_manifest.sha256"),
            "sha256": _sha256_file(root / "full_current_reference_manifest.sha256"),
        },
    ]
    for job in _load_registry(root)["jobs"]:
        path = Path(str(job["training_config_path"]))
        rows.append(
            {
                "config_role": str(job["job_id"]),
                "path": str(path),
                "sha256": _sha256_file(path),
            }
        )
    return _write_csv(
        root / "config_hashes.csv",
        rows,
        ("config_role", "path", "sha256"),
    )


def refresh_current_input_lineage(root: str | Path) -> Path:
    experiment_root = _resolve_repo_path(root)
    registry = _load_registry(experiment_root)
    status = _read_json(experiment_root / "registry" / "experiment_status.json")
    jobs = []
    output_rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    def add_output(job_id: str, role: str, path: Path) -> None:
        resolved_path = str(path.resolve(strict=False))
        if resolved_path in seen_paths or not path.is_file():
            return
        seen_paths.add(resolved_path)
        output_rows.append(
            {
                "job_id": job_id,
                "artifact_role": role,
                "path": resolved_path,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )

    for job in registry["jobs"]:
        status_path = _job_status_path(experiment_root, str(job["job_id"]))
        job_status = _read_json(status_path)
        jobs.append(
            {
                "job_id": job["job_id"],
                "job_spec_sha256": job["job_spec_sha256"],
                "config_sha256": job["config_sha256"],
                "status": job_status.get("status"),
                "status_sha256": _sha256_file(status_path),
                "full_current_reference_job_id": job["full_current_reference_job_id"],
                "generator_current_input_mode": job["generator_current_input_mode"],
                "generator_current_input_fingerprint": job[
                    "generator_current_input_fingerprint"
                ],
                "q4_evaluator_calls": int(job_status.get("q4_evaluator_calls", 0)),
            }
        )
        add_output(str(job["job_id"]), "job_status", status_path)
        for artifact in job_status.get("artifacts", []):
            add_output(
                str(job["job_id"]),
                str(artifact["artifact_role"]),
                Path(str(artifact["path"])),
            )

    lineage = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "experiment_stage": EXPERIMENT_STAGE,
        "experiment_root": str(experiment_root),
        "experiment_status": status.get("status"),
        "resolved_config_sha256": registry["resolved_config_sha256"],
        "source_manifest_sha256": _sha256_file(experiment_root / "source_hashes.csv"),
        "code_manifest_sha256": _sha256_file(experiment_root / "code_hashes.csv"),
        "config_manifest_sha256": _sha256_file(experiment_root / "config_hashes.csv"),
        "full_current_reference_manifest_sha256": registry[
            "full_current_reference_manifest_sha256"
        ],
        "generator_current_input_mode": GENERATOR_CURRENT_INPUT_MODE,
        "generator_current_input_fingerprint": generator_current_input_fingerprint(
            GENERATOR_CURRENT_INPUT_MODE
        ),
        "q4_evaluator_calls": 0,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "jobs": jobs,
        "updated_at_utc": _utc_now(),
    }
    postprocess_fields = (
        "prepared_code_manifest_path",
        "prepared_code_manifest_sha256",
        "current_code_manifest_path",
        "current_code_manifest_sha256",
        "pre_postprocess_output_hash_manifest_path",
        "pre_postprocess_output_hash_manifest_sha256",
        "postprocess_allowed_code_drift_paths",
        "postprocess_code_changed_paths",
        "code_changed_since_prepared_snapshot",
        "postprocess_code_drift_ledger_path",
        "postprocess_code_drift_ledger_payload_sha256",
        "postprocess_code_drift_ledger_sha256",
    )
    present_postprocess_fields = [
        field for field in postprocess_fields if field in status
    ]
    if present_postprocess_fields:
        if len(present_postprocess_fields) != len(postprocess_fields):
            raise ValueError(
                "Experiment status has an incomplete postprocess code ledger"
            )
        lineage.update({field: status[field] for field in postprocess_fields})
    lineage["lineage_sha256"] = _payload_sha256(lineage)
    lineage_path = _write_json(experiment_root / "experiment_lineage.json", lineage)

    stable_files = (
        "resolved_config.yaml",
        "registry/resolved_config.sha256",
        "registry/jobs.json",
        "registry/experiment_status.json",
        "full_current_reference_manifest.csv",
        "full_current_reference_manifest.sha256",
        "source_hashes.csv",
        "code_hashes.csv",
        CURRENT_CODE_MANIFEST_FILENAME,
        POSTPROCESS_CODE_DRIFT_LEDGER_FILENAME,
        PRE_POSTPROCESS_OUTPUT_HASH_MANIFEST_FILENAME,
        "config_hashes.csv",
        "split_manifest.csv",
        "text_ablation_manifest.csv",
        "task_registry.csv",
        "resource_usage.csv",
        "resource_summary.csv",
        "experiment_lineage.json",
    )
    for relative in stable_files:
        add_output("", f"experiment:{relative}", experiment_root / relative)
    for directory in (experiment_root / "analysis", experiment_root / "report"):
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    add_output(
                        "",
                        f"experiment:{path.relative_to(experiment_root).as_posix()}",
                        path,
                    )
    _write_csv(
        experiment_root / "output_hashes.csv",
        output_rows,
        ("job_id", "artifact_role", "path", "size_bytes", "sha256"),
    )
    return lineage_path


def prepare_current_input_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    reuse: bool = False,
) -> Path:
    root = _resolve_repo_path(output_dir)
    resolved = _resolved_config(config_path)
    resolved_sha = _payload_sha256(resolved)
    resolved_hash_path = root / "registry" / "resolved_config.sha256"
    if root.exists() and any(root.iterdir()):
        if not reuse:
            raise FileExistsError(f"Experiment root exists; use --reuse: {root}")
        if (
            not resolved_hash_path.is_file()
            or resolved_hash_path.read_text(encoding="utf-8").strip() != resolved_sha
        ):
            raise ValueError("Existing root is not this immutable current-input sweep")
        _validate_registry(root)
        _validate_split_manifest(root)
        existing_status = _read_json(root / "registry" / "experiment_status.json")
        if existing_status.get("status") == "completed_q3_only":
            _validate_completed_experiment(root)
            return root
        _validate_code_manifest(root)
        _refresh_exports(root)
        refresh_current_input_lineage(root)
        return root

    training._validate_dataset_summary(Path(resolved["datasets"]["root"]))
    reference_rows = _reference_rows(resolved)
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
    training._build_split_manifest(resolved, root)
    _validate_split_manifest(root)
    _write_yaml(root / "resolved_config.yaml", {ROOT_KEY: resolved})
    _atomic_write_text(resolved_hash_path, resolved_sha + "\n")
    _write_csv(
        root / "full_current_reference_manifest.csv",
        reference_rows,
        tuple(reference_rows[0]),
    )
    reference_manifest_sha = _sha256_file(root / "full_current_reference_manifest.csv")
    _atomic_write_text(
        root / "full_current_reference_manifest.sha256",
        reference_manifest_sha + "\n",
    )
    reference_by_cell = _reference_by_cell(reference_rows)
    _write_csv(
        root / "source_hashes.csv",
        source_rows,
        ("source_role", "path", "size_bytes", "sha256"),
    )
    source_manifest_sha = _sha256_file(root / "source_hashes.csv")
    source_by_path = {str(row["path"]): str(row["sha256"]) for row in source_rows}
    code_rows = _code_rows()
    _write_csv(
        root / "code_hashes.csv",
        code_rows,
        ("relative_path", "path", "size_bytes", "sha256"),
    )

    runtime = resolved["runtime"]
    gpu_ids = tuple(int(value) for value in runtime["gpu_ids"])
    numa_nodes = {
        int(key): int(value) for key, value in dict(runtime["gpu_numa_nodes"]).items()
    }
    jobs: list[dict[str, Any]] = []
    for spec in _job_specs():
        seed = int(spec["seed"])
        tolerance = int(spec["tolerance_minutes"])
        reference = reference_by_cell[(seed, tolerance)]
        job_id = _job_id(seed, tolerance)
        payload = _training_payload(reference, root, seed=seed, tolerance=tolerance)
        training_path = root / "configs" / "jobs" / f"{job_id}.yaml"
        _write_yaml(training_path, payload)
        dataset_path = str(payload["data_path"])
        if dataset_path not in source_by_path:
            raise ValueError(f"Dataset source hash unavailable: {dataset_path}")
        gpu_id = gpu_ids[int(spec["gpu_index"])]
        job: dict[str, Any] = {
            "job_id": job_id,
            "wave": int(spec["wave"]),
            "model_family": "wgan",
            "trainer_command": "vol-xlsx",
            "experiment_stage": EXPERIMENT_STAGE,
            "capacity_profile": CAPACITY_PROFILE,
            "capacity_profile_sha256": _capacity_profile_sha256(),
            "capacity_seed_profile_sha256": _capacity_seed_profile_sha256(seed),
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
            "seed": seed,
            "text_ablation_mode": REAL_TEXT,
            "text_information_path": text_information_path(REAL_TEXT),
            "support_mask_mode": "raw_joint",
            "generator_noise_mode": GENERATOR_NOISE_MODE,
            "generator_noise_fingerprint": generator_noise_fingerprint(
                GENERATOR_NOISE_MODE, NOISE_DIM
            ),
            "generator_current_input_mode": GENERATOR_CURRENT_INPUT_MODE,
            "generator_current_input_fingerprint": (
                generator_current_input_fingerprint(GENERATOR_CURRENT_INPUT_MODE)
            ),
            "full_current_reference_job_id": reference["reference_job_id"],
            "full_current_reference_manifest_sha256": reference_manifest_sha,
            "full_current_initial_generator_state_sha256": reference[
                "initial_generator_state_sha256"
            ],
            "full_current_initial_discriminator_state_sha256": reference[
                "initial_discriminator_state_sha256"
            ],
            "full_current_initial_metadata_path": reference["initial_metadata_path"],
            "full_current_initial_metadata_sha256": reference[
                "initial_metadata_sha256"
            ],
            "full_current_q3_pair_panel_sha256": reference[
                "reference_q3_pair_panel_sha256"
            ],
            "q4_evaluator_calls": 0,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
            "tolerance_minutes": tolerance,
            "gpu_id": gpu_id,
            "gpu_slot": int(spec["gpu_slot"]),
            "numa_node": numa_nodes[gpu_id],
            "training_config_path": str(training_path),
            "config_sha256": _sha256_file(training_path),
            "dataset_path": dataset_path,
            "dataset_sha256": source_by_path[dataset_path],
            "source_manifest_sha256": source_manifest_sha,
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
                "job_spec_sha256": job["job_spec_sha256"],
                "wave": job["wave"],
                "seed": seed,
                "tolerance_minutes": tolerance,
                "gpu_id": gpu_id,
                "gpu_slot": job["gpu_slot"],
                "capacity_profile": CAPACITY_PROFILE,
                "lr_profile": LR_PROFILE,
                "generator_noise_mode": GENERATOR_NOISE_MODE,
                "generator_noise_fingerprint": job["generator_noise_fingerprint"],
                "generator_current_input_mode": GENERATOR_CURRENT_INPUT_MODE,
                "generator_current_input_fingerprint": job[
                    "generator_current_input_fingerprint"
                ],
                "full_current_reference_job_id": reference["reference_job_id"],
                "full_current_reference_manifest_sha256": reference_manifest_sha,
                "q4_evaluator_calls": 0,
                "q4_predictions_generated": False,
                "q4_evaluated": False,
                "updated_at_utc": _utc_now(),
            },
        )

    _write_json(
        root / "registry" / "jobs.json",
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "experiment_stage": EXPERIMENT_STAGE,
            "experiment_root": str(root),
            "resolved_config_sha256": resolved_sha,
            "full_current_reference_manifest_sha256": reference_manifest_sha,
            "generator_current_input_mode": GENERATOR_CURRENT_INPUT_MODE,
            "generator_current_input_fingerprint": (
                generator_current_input_fingerprint(GENERATOR_CURRENT_INPUT_MODE)
            ),
            "q4_evaluator_calls": 0,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
            "created_at_utc": _utc_now(),
            "jobs": jobs,
        },
    )
    _write_json(
        root / "registry" / "experiment_status.json",
        {
            "status": "prepared",
            "current_stage": EXPERIMENT_STAGE,
            "q4_evaluator_calls": 0,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
            "updated_at_utc": _utc_now(),
        },
    )
    _write_config_hashes(root)
    _validate_code_manifest(root)
    _validate_registry(root)
    _refresh_exports(root)
    refresh_current_input_lineage(root)
    return root


def _checkpoint_parameter_count(checkpoint: Mapping[str, Any]) -> int:
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint state_dict must be a mapping")
    return int(sum(int(value.numel()) for value in state.values()))


def _validated_wgan_lr_trace(
    job: Mapping[str, Any], run_dir: Path, *, dry_run: bool
) -> list[dict[str, float | int]] | list[float]:
    # Reuse the already audited WGAN schedule gate.  It validates epoch 0,
    # monotone/floored G+D LR traces, finite GP/reconstruction diagnostics and
    # the frozen minimum of 30 learned epochs.
    from scripts.rq3.news_first_vol_zero_noise_ablation import (
        _validated_wgan_lr_trace as validate,
    )

    return validate(job, run_dir, dry_run=dry_run)


def _validate_dry_run_contract(job: Mapping[str, Any], run_dir: Path) -> None:
    path = run_dir / "metrics" / "training_resolved_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _load_yaml_mapping(path, f"dry-run config {job['job_id']}")
    contracts = {
        "seed": int(job["seed"]),
        "support_mask_mode": "raw_joint",
        "generator_noise_mode": GENERATOR_NOISE_MODE,
        "generator_current_input_mode": GENERATOR_CURRENT_INPUT_MODE,
        "validation_mc_samples": 16,
        "learning_rate": FIXED_LEARNING_RATE,
        "reduce_lr_min_lr": FIXED_SCHEDULER_MIN_LR,
    }
    for field, expected in contracts.items():
        if payload.get(field) != expected:
            raise ValueError(f"Dry-run contract mismatch: {job['job_id']}/{field}")
    if Path(str(payload["data_path"])) != Path(str(job["dataset_path"])):
        raise ValueError("Dry-run loaded a different training workbook")


def _validate_run_contract(
    root: Path,
    job: Mapping[str, Any],
    run_dir: Path,
    *,
    dry_run: bool,
) -> tuple[list[dict[str, float | int]] | list[float], Path | None]:
    import torch

    trace = _validated_wgan_lr_trace(job, run_dir, dry_run=dry_run)
    if dry_run:
        _validate_dry_run_contract(job, run_dir)
        return trace, None
    reference = _validate_reference_manifest(root)[
        (int(job["seed"]), int(job["tolerance_minutes"]))
    ]
    best_path = run_dir / "metrics" / "best_learned_checkpoint.json"
    best = _require_mapping(
        _read_json(best_path), f"best learned checkpoint {job['job_id']}"
    )
    if int(best.get("best_epoch", -1)) < 1:
        raise ValueError("Formal comparison requires a learned epoch")
    if str(best.get("selection_scope", "")) != "trained_epochs_only":
        raise ValueError("Best-learned selection scope drifted")
    current_fingerprint = generator_current_input_fingerprint(
        GENERATOR_CURRENT_INPUT_MODE
    )
    if (
        best.get("generator_current_input_mode") != GENERATOR_CURRENT_INPUT_MODE
        or best.get("generator_current_input_fingerprint") != current_fingerprint
    ):
        raise ValueError("Best-learned metadata lost masked-input lineage")
    if best.get("generator_noise_mode") != GENERATOR_NOISE_MODE or best.get(
        "generator_noise_fingerprint"
    ) != generator_noise_fingerprint(GENERATOR_NOISE_MODE, NOISE_DIM):
        raise ValueError("Best-learned metadata lost Gaussian-noise lineage")

    artifacts = _require_mapping(best.get("artifacts"), "best learned artifacts")
    generator_path = Path(str(artifacts.get("generator", "")))
    discriminator_path = Path(str(artifacts.get("discriminator", "")))
    if not generator_path.is_file() or not discriminator_path.is_file():
        raise FileNotFoundError("Best-learned G/D checkpoint is missing")
    generator = torch.load(generator_path, map_location="cpu", weights_only=False)
    discriminator = torch.load(
        discriminator_path, map_location="cpu", weights_only=False
    )
    expected_noise_fingerprint = generator_noise_fingerprint(
        GENERATOR_NOISE_MODE, NOISE_DIM
    )
    for checkpoint, label, parameters in (
        (generator, "generator", 123_472),
        (discriminator, "discriminator", 25_861),
    ):
        input_mode, input_fingerprint = (
            resolve_checkpoint_generator_current_input_contract(checkpoint)
        )
        noise_mode, noise_fingerprint = resolve_checkpoint_generator_noise_contract(
            checkpoint
        )
        if (
            input_mode != GENERATOR_CURRENT_INPUT_MODE
            or input_fingerprint != current_fingerprint
        ):
            raise ValueError(f"{label} masked-input checkpoint contract drifted")
        if (
            noise_mode != GENERATOR_NOISE_MODE
            or noise_fingerprint != expected_noise_fingerprint
        ):
            raise ValueError(f"{label} Gaussian checkpoint contract drifted")
        if _checkpoint_parameter_count(checkpoint) != parameters:
            raise ValueError(f"Small WGAN {label} parameter count drifted")
    if _checkpoint_parameter_count(generator) + _checkpoint_parameter_count(
        discriminator
    ) != int(job["expected_wgan_parameters"]):
        raise ValueError("Small WGAN total parameter count drifted")

    initial_generator_path = run_dir / "checkpoints" / "generator_initial_epoch0.pt"
    initial_discriminator_path = (
        run_dir / "checkpoints" / "discriminator_initial_epoch0.pt"
    )
    initial_generator = torch.load(
        initial_generator_path, map_location="cpu", weights_only=False
    )
    initial_discriminator = torch.load(
        initial_discriminator_path, map_location="cpu", weights_only=False
    )
    initial_g_sha = _checkpoint_state_sha256(initial_generator)
    initial_d_sha = _checkpoint_state_sha256(initial_discriminator)
    if initial_g_sha != str(reference["initial_generator_state_sha256"]):
        raise ValueError("Masked/full-current initial generator tensors differ")
    if initial_d_sha != str(reference["initial_discriminator_state_sha256"]):
        raise ValueError("Masked/full-current initial discriminator tensors differ")
    # This also proves both tolerance lanes for a seed started from the same
    # exact tensor state.
    if (initial_g_sha, initial_d_sha) != EXPECTED_INITIAL_STATE_HASHES[
        int(job["seed"])
    ]:
        raise ValueError("Initial tensor state differs from the frozen seed identity")

    masked_initial = _read_json(run_dir / "metrics" / "initial_checkpoint.json")
    full_initial = _read_json(Path(str(reference["initial_metadata_path"])))
    masked_metrics = _require_mapping(
        masked_initial.get("metrics"), "masked epoch-0 metrics"
    )
    full_metrics = _require_mapping(
        full_initial.get("metrics"), "full-current epoch-0 metrics"
    )
    if set(masked_metrics) != set(full_metrics):
        raise ValueError("Epoch-0 validation metric fields differ")
    epoch0_max_abs = max(
        abs(float(masked_metrics[key]) - float(full_metrics[key]))
        for key in masked_metrics
    )
    if epoch0_max_abs > 1.0e-8:
        raise ValueError("Masked/full-current epoch-0 metrics differ materially")
    for metrics, label in ((masked_metrics, "masked"), (full_metrics, "full")):
        if not math.isclose(
            float(metrics["val_recon"]),
            float(metrics["val_current_recon"]),
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise ValueError(f"{label} epoch-0 prediction is not persistence")

    audit = {
        "schema_version": 1,
        "job_id": job["job_id"],
        "seed": int(job["seed"]),
        "tolerance_minutes": int(job["tolerance_minutes"]),
        "generator_current_input_mode": GENERATOR_CURRENT_INPUT_MODE,
        "generator_current_input_fingerprint": current_fingerprint,
        "generator_noise_mode": GENERATOR_NOISE_MODE,
        "generator_noise_fingerprint": expected_noise_fingerprint,
        "full_current_reference_job_id": reference["reference_job_id"],
        "full_current_reference_manifest_sha256": job[
            "full_current_reference_manifest_sha256"
        ],
        "initial_generator_state_sha256": initial_g_sha,
        "initial_discriminator_state_sha256": initial_d_sha,
        "initial_tensors_exactly_match_full_current": True,
        "epoch0_metrics_match_full_current": True,
        "epoch0_metric_max_abs_difference": epoch0_max_abs,
        "epoch0_metric_absolute_tolerance": 1.0e-8,
        "q4_evaluator_calls": 0,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "validated_at_utc": _utc_now(),
    }
    audit["audit_sha256"] = _payload_sha256(audit)
    audit_path = _write_json(
        run_dir / "metrics" / "current_input_contract_audit.json", audit
    )
    return trace, audit_path


def _completed_job_is_valid(
    root: Path, job: Mapping[str, Any], status: Mapping[str, Any]
) -> bool:
    try:
        _validate_job_lineage(root, job)
    except (ValueError, FileNotFoundError):
        return False
    if int(status.get("q4_evaluator_calls", -1)) != 0:
        return False
    return training._completed_job_is_valid(root, job, status)


def _validate_completed_job_matrix(root: Path) -> None:
    """Validate all six frozen jobs and their last exported artifact hashes."""

    output_manifest = root / "output_hashes.csv"
    if not output_manifest.is_file():
        raise FileNotFoundError(output_manifest)
    with output_manifest.open("r", encoding="utf-8", newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    if not output_rows:
        raise ValueError("Output-hash manifest is empty")
    output_by_path: dict[str, dict[str, str]] = {}
    for row in output_rows:
        resolved_path = str(Path(str(row.get("path", ""))).resolve(strict=False))
        if not resolved_path or resolved_path in output_by_path:
            raise ValueError("Output-hash manifest has empty or duplicate paths")
        output_by_path[resolved_path] = row

    failures: list[str] = []
    jobs = [dict(job) for job in _load_registry(root)["jobs"]]
    for job in jobs:
        job_id = str(job["job_id"])
        status_path = _job_status_path(root, job_id)
        if not status_path.is_file():
            failures.append(f"{job_id}=missing_status")
            continue
        status_sha = _sha256_file(status_path)
        anchored_status = output_by_path.get(str(status_path.resolve(strict=False)))
        if (
            anchored_status is None
            or str(anchored_status.get("sha256", "")) != status_sha
        ):
            failures.append(f"{job_id}=unanchored_status")
            continue
        status = _read_json(status_path)
        if (
            not _completed_job_is_valid(root, job, status)
            or str(status.get("job_spec_sha256", "")) != str(job["job_spec_sha256"])
            or bool(status.get("q4_predictions_generated", True))
            or bool(status.get("q4_evaluated", True))
        ):
            failures.append(f"{job_id}={status.get('status', 'invalid')}")
            continue
        for artifact in status.get("artifacts", []):
            artifact_path = Path(str(artifact.get("path", "")))
            anchored = output_by_path.get(str(artifact_path.resolve(strict=False)))
            if anchored is None or str(anchored.get("sha256", "")) != str(
                artifact.get("sha256", "")
            ):
                failures.append(f"{job_id}=unanchored_artifact")
                break
    if failures:
        raise RuntimeError(
            "Current-input six-job matrix is incomplete or drifted: "
            + ", ".join(failures)
        )


def run_current_input_worker(
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
        raise KeyError(f"Unknown or duplicate current-input job: {job_id}")
    job = matches[0]
    _validate_code_manifest(root)
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
    if previous.get("status") in {"running", "failed"} and not (resume or dry_run):
        raise RuntimeError(f"Interrupted job requires --resume: {job_id}")
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    assigned = str(job["gpu_id"])
    if visible and visible.split(",")[0].strip() != assigned:
        raise RuntimeError(
            f"Worker GPU mismatch: assigned={assigned}, visible={visible}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = assigned
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
        "wave": int(job["wave"]),
        "gpu_id": int(job["gpu_id"]),
        "gpu_slot": int(job["gpu_slot"]),
        "config_sha256": job["config_sha256"],
        "job_spec_sha256": job["job_spec_sha256"],
        "seed": int(job["seed"]),
        "tolerance_minutes": int(job["tolerance_minutes"]),
        "capacity_profile": CAPACITY_PROFILE,
        "lr_profile": LR_PROFILE,
        "generator_noise_mode": GENERATOR_NOISE_MODE,
        "generator_noise_fingerprint": job["generator_noise_fingerprint"],
        "generator_current_input_mode": GENERATOR_CURRENT_INPUT_MODE,
        "generator_current_input_fingerprint": job[
            "generator_current_input_fingerprint"
        ],
        "full_current_reference_job_id": job["full_current_reference_job_id"],
        "full_current_reference_manifest_sha256": job[
            "full_current_reference_manifest_sha256"
        ],
        "q4_evaluator_calls": 0,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
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
                    "artifact_role": "current_input_contract_audit",
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


def build_current_input_worker_command(
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
            "scripts.rq3.news_first_vol_current_input_sweep",
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
    wave: int,
    *,
    resume: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    jobs = [
        dict(job)
        for job in _load_registry(root)["jobs"]
        if int(job["wave"]) == int(wave)
    ]
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
    wave: int,
    jobs: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
    resume: bool,
) -> None:
    if not jobs:
        return
    expected_count = 4 if int(wave) == 1 else 2
    if len(jobs) > expected_count:
        raise ValueError(f"Wave {wave} exceeds its frozen concurrency")
    resolved = _validate_resolved_snapshot(root)
    runtime = resolved["runtime"]
    monitor = training._ResourceMonitor(
        root / "resource_usage.csv",
        executable=str(runtime.get("nvidia_smi_executable", "nvidia-smi")),
        interval_seconds=float(runtime.get("resource_sample_interval_seconds", 2)),
        wave=int(wave),
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
                build_current_input_worker_command(
                    config_path,
                    root,
                    job,
                    dry_run=dry_run,
                    resume=resume,
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
                        f"Current-input job {jobs[index]['job_id']} exited {code}"
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
    root: Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
) -> None:
    failures = []
    for job in jobs:
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        valid = (
            status.get("status") == "dry_run_passed"
            if dry_run
            else _completed_job_is_valid(root, job, status)
        )
        if not valid or int(status.get("q4_evaluator_calls", -1)) != 0:
            failures.append(f"{job['job_id']}={status.get('status', 'missing')}")
    if failures:
        raise RuntimeError(f"Current-input wave did not complete: {failures}")


def _mark_status(root: Path, status: str, **details: Any) -> None:
    _write_json(
        root / "registry" / "experiment_status.json",
        {
            "status": status,
            "current_stage": EXPERIMENT_STAGE,
            "q4_evaluator_calls": 0,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
            "updated_at_utc": _utc_now(),
            **details,
        },
    )


def _validate_self_hashed_json(path: Path, hash_field: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _read_json(path)
    saved = str(payload.pop(hash_field, ""))
    if not saved or saved != _payload_sha256(payload):
        raise ValueError(f"Self-hash mismatch: {path}")
    payload[hash_field] = saved
    return payload


def _validate_completed_experiment(root: Path) -> None:
    status_path = root / "registry" / "experiment_status.json"
    status = _read_json(status_path)
    if status.get("status") != "completed_q3_only":
        raise ValueError("Experiment is not in the completed Q3-only state")
    if (
        int(status.get("q4_evaluator_calls", -1)) != 0
        or bool(status.get("q4_predictions_generated", True))
        or bool(status.get("q4_evaluated", True))
    ):
        raise ValueError("Completed experiment violated Q4 isolation")
    postprocess_contract: dict[str, Any] | None = None
    if "current_code_manifest_sha256" in status:
        snapshot = _postprocess_code_snapshot(root, persist=False)
        postprocess_contract = _postprocess_status_fields(snapshot)
        for field, expected in postprocess_contract.items():
            if status.get(field) != expected:
                raise ValueError(f"Completed postprocess code ledger drifted: {field}")
        _validate_postprocess_exports(root)
    else:
        _validate_code_manifest(root)
    for job in _load_registry(root)["jobs"]:
        job_status = _read_json(_job_status_path(root, str(job["job_id"])))
        if not _completed_job_is_valid(root, job, job_status):
            raise ValueError(
                f"Completed experiment has an invalid job: {job['job_id']}"
            )
    analysis_path = Path(str(status.get("analysis_path", "")))
    report_path = Path(str(status.get("report_path", "")))
    for path, field in (
        (analysis_path, "analysis_sha256"),
        (report_path, "report_sha256"),
    ):
        if not path.is_file() or _sha256_file(path) != str(status.get(field, "")):
            raise ValueError(f"Completed experiment artifact hash mismatch: {path}")

    analysis_dir = root / "analysis" / "current_input_ablation"
    analysis_summary = _validate_self_hashed_json(
        analysis_dir / "current_input_analysis_summary.json", "analysis_sha256"
    )
    validation = _validate_self_hashed_json(
        analysis_dir / "current_input_validation_summary.json", "validation_sha256"
    )
    if (
        validation.get("status") != "pass"
        or validation.get("analysis_sha256") != analysis_summary["analysis_sha256"]
    ):
        raise ValueError("Completed current-input analysis validation did not pass")
    report_manifest = _validate_self_hashed_json(
        root / "report" / "current_input_ablation_report_manifest.json",
        "report_manifest_sha256",
    )
    if (
        Path(str(report_manifest.get("report_path", ""))) != report_path
        or report_manifest.get("report_sha256") != _sha256_file(report_path)
        or report_manifest.get("analysis_sha256") != analysis_summary["analysis_sha256"]
        or bool(report_manifest.get("q4_predictions_generated", True))
        or bool(report_manifest.get("q4_evaluated", True))
    ):
        raise ValueError("Completed report manifest contract drifted")
    lineage = _validate_self_hashed_json(
        root / "experiment_lineage.json", "lineage_sha256"
    )
    if (
        lineage.get("experiment_status") != "completed_q3_only"
        or int(lineage.get("q4_evaluator_calls", -1)) != 0
    ):
        raise ValueError("Completed experiment lineage drifted")
    if postprocess_contract is not None:
        for field, expected in postprocess_contract.items():
            if lineage.get(field) != expected:
                raise ValueError(f"Completed lineage code ledger drifted: {field}")

    output_manifest = root / "output_hashes.csv"
    if not output_manifest.is_file():
        raise FileNotFoundError(output_manifest)
    found = {str(analysis_path.resolve()), str(report_path.resolve())}
    with output_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Completed output-hash manifest is empty")
    for row in rows:
        path = Path(str(row["path"]))
        if not path.is_file() or _sha256_file(path) != str(row["sha256"]):
            raise ValueError(f"Completed output artifact drifted: {path}")
        found.discard(str(path.resolve()))
    if found:
        raise ValueError("Completed output manifest omits analysis/report artifacts")


def _run_postprocess(root: Path) -> tuple[Path, Path]:
    # Import only after all six formal jobs pass artifact QA.  This keeps
    # worker processes lean and makes analysis/report failures terminal rather
    # than allowing a misleading completed experiment state.
    from scripts.rq3.news_first_vol_current_input_analysis import (
        run_current_input_analysis,
    )
    from scripts.rq3.news_first_vol_current_input_report import (
        render_current_input_report,
    )

    resolved = _validate_resolved_snapshot(root)
    reference_root = Path(
        str(
            resolved["current_support_masked_seed_sweep"]["full_current_reference"][
                "root"
            ]
        )
    )
    analysis_path = run_current_input_analysis(
        root,
        reference_root=reference_root,
    )
    report_path = render_current_input_report(root)
    if not Path(analysis_path).is_file() or not Path(report_path).is_file():
        raise RuntimeError("Current-input postprocess did not persist analysis/report")
    return Path(analysis_path), Path(report_path)


def _validate_postprocess_exports(root: Path) -> None:
    resource_path = root / "resource_summary.csv"
    task_path = root / "task_registry.csv"
    with resource_path.open("r", encoding="utf-8", newline="") as handle:
        resource_rows = list(csv.DictReader(handle))
    with task_path.open("r", encoding="utf-8", newline="") as handle:
        task_rows = list(csv.DictReader(handle))
    expected_ids = {
        _job_id(seed, tolerance)
        for seed in FROZEN_SEEDS
        for tolerance in FROZEN_TOLERANCES
    }
    if {str(row.get("job_id", "")) for row in resource_rows} != expected_ids:
        raise ValueError("Formal resource summary does not contain the six jobs")
    if any(
        str(row.get("status", "")) != "completed"
        or str(row.get("dry_run", "")).strip().lower() not in {"false", "0"}
        for row in resource_rows
    ):
        raise ValueError("Formal resource summary still contains dry-run jobs")
    if {str(row.get("job_id", "")) for row in task_rows} != expected_ids:
        raise ValueError("Formal task registry does not contain the six jobs")
    if any(
        str(row.get("status", "")) != "completed" or int(row.get("attempt", -1)) != 2
        for row in task_rows
    ):
        raise ValueError("Formal task registry does not record six attempt-2 jobs")


def postprocess_current_input_experiment(output_dir: str | Path) -> Path:
    """Recover analysis/report only; never invoke a worker or training wave."""

    root = _resolve_repo_path(output_dir)
    _validate_resolved_snapshot(root)
    _validate_registry(root)
    _validate_split_manifest(root)
    previous = _read_json(root / "registry" / "experiment_status.json")
    if previous.get("status") == "completed_q3_only":
        _validate_completed_experiment(root)
        return root
    if previous.get("status") != "failed":
        raise RuntimeError(
            "Postprocess-only recovery requires a failed experiment with completed jobs"
        )
    if (
        int(previous.get("q4_evaluator_calls", -1)) != 0
        or bool(previous.get("q4_predictions_generated", True))
        or bool(previous.get("q4_evaluated", True))
    ):
        raise ValueError("Postprocess-only recovery found forbidden Q4 activity")

    # This is intentionally before any new ledger/status write.  An incomplete
    # or tampered training matrix must leave the failed root untouched.
    _validate_completed_job_matrix(root)
    snapshot = _postprocess_code_snapshot(root, persist=True)
    provenance = _postprocess_status_fields(snapshot)
    previous_error = str(previous.get("error", ""))
    _mark_status(
        root,
        "postprocessing",
        postprocess_only=True,
        recovered_from_status="failed",
        recovered_from_error=previous_error,
        pid=os.getpid(),
        host=socket.gethostname(),
        **provenance,
    )
    try:
        analysis_path, report_path = _run_postprocess(root)
        _mark_status(
            root,
            "completed_q3_only",
            postprocess_only=True,
            recovered_from_status="failed",
            recovered_from_error=previous_error,
            analysis_path=str(analysis_path),
            analysis_sha256=_sha256_file(analysis_path),
            report_path=str(report_path),
            report_sha256=_sha256_file(report_path),
            completed_at_utc=_utc_now(),
            **provenance,
        )
        training._write_resource_summary(root)
        _refresh_exports(root)
        _validate_postprocess_exports(root)
        refresh_current_input_lineage(root)
        _validate_completed_experiment(root)
    except BaseException as exc:
        _mark_status(
            root,
            "failed",
            postprocess_only=True,
            recovered_from_status="failed",
            recovered_from_error=previous_error,
            error=f"{type(exc).__name__}: {exc}",
            **provenance,
        )
        try:
            _refresh_exports(root)
            refresh_current_input_lineage(root)
        except BaseException as refresh_exc:
            _mark_status(
                root,
                "failed",
                postprocess_only=True,
                recovered_from_status="failed",
                recovered_from_error=previous_error,
                error=f"{type(exc).__name__}: {exc}",
                recovery_refresh_error=(f"{type(refresh_exc).__name__}: {refresh_exc}"),
                **provenance,
            )
        raise
    return root


def launch_current_input_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> Path:
    root = prepare_current_input_experiment(config_path, output_dir, reuse=True)
    previous = _read_json(root / "registry" / "experiment_status.json")
    if previous.get("status") == "completed_q3_only" and not dry_run:
        if not resume:
            raise RuntimeError("Completed formal experiment requires --resume")
        _validate_registry(root)
        _validate_completed_experiment(root)
        return root
    if previous.get("status") in {"running", "failed"} and not (resume or dry_run):
        raise RuntimeError("Interrupted formal experiment requires --resume")
    _mark_status(root, "dry_running" if dry_run else "running")
    try:
        for wave in (1, 2):
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
            raise RuntimeError(f"Current-input six-job matrix incomplete: {failures}")
        if dry_run:
            _mark_status(root, "ready_after_dry_run", completed_at_utc=_utc_now())
        else:
            analysis_path, report_path = _run_postprocess(root)
            _mark_status(
                root,
                "completed_q3_only",
                analysis_path=str(analysis_path),
                analysis_sha256=_sha256_file(analysis_path),
                report_path=str(report_path),
                report_sha256=_sha256_file(report_path),
                completed_at_utc=_utc_now(),
            )
        training._write_resource_summary(root)
        _refresh_exports(root)
        refresh_current_input_lineage(root)
        if not dry_run:
            _validate_completed_experiment(root)
    except BaseException as exc:
        _mark_status(root, "failed", error=f"{type(exc).__name__}: {exc}")
        _refresh_exports(root)
        refresh_current_input_lineage(root)
        raise
    return root


def run_news_first_vol_current_input_sweep(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    action: str = "prepare",
    job_id: str = "",
    resume: bool = False,
    reuse: bool = False,
    worker_dry_run: bool = False,
) -> Path:
    action = str(action).strip().lower()
    if action == "prepare":
        return prepare_current_input_experiment(
            config_path, output_dir, reuse=bool(reuse or resume)
        )
    if action == "dry-run":
        return launch_current_input_experiment(
            config_path, output_dir, resume=resume, dry_run=True
        )
    if action == "worker":
        if not job_id:
            raise ValueError("--job-id is required for worker")
        return run_current_input_worker(
            output_dir, job_id, dry_run=worker_dry_run, resume=resume
        )
    if action == "postprocess":
        return postprocess_current_input_experiment(output_dir)
    if action == "launch":
        return launch_current_input_experiment(
            config_path, output_dir, resume=resume, dry_run=False
        )
    raise ValueError(f"Unsupported current-input action: {action}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "dry-run", "worker", "launch", "postprocess"),
    )
    parser.add_argument(
        "--config",
        default="configs/rq3/news_first_vol_current_input_sweep.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--worker-dry-run", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = _parser().parse_args(argv)
    output = run_news_first_vol_current_input_sweep(
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
    "EXPERIMENT_KIND",
    "EXPERIMENT_STAGE",
    "FROZEN_SEEDS",
    "FROZEN_TOLERANCES",
    "GENERATOR_CURRENT_INPUT_MODE",
    "REFERENCE_CURRENT_INPUT_MODE",
    "build_current_input_worker_command",
    "launch_current_input_experiment",
    "postprocess_current_input_experiment",
    "prepare_current_input_experiment",
    "refresh_current_input_lineage",
    "run_current_input_worker",
    "run_news_first_vol_current_input_sweep",
]
