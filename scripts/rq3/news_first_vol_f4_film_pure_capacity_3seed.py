"""Run the F4-only FiLM-CNN/Pure-CNN capacity robustness experiment.

Training and checkpoint selection stop at the end of 2023Q3.  The frozen
2023Q4 panel is deliberately absent from every preparation and training code
path; it becomes eligible only after all 36 checkpoint selections are frozen.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

import torch
import yaml

import scripts._path_setup  # noqa: F401
from scripts.rq3 import news_first_vol_film_unet_capacity_epoch_3seed as proven
from scripts.rq3 import news_first_vol_training as training
from wgan_option.models.common import (
    critic_conditioning_fingerprint,
    generator_conditioning_fingerprint,
)
from wgan_option.models.discriminator import Discriminator
from wgan_option.models.generator import Generator


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/rq3/news_first_vol_f4_film_pure_capacity_3seed.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq3_news_first_vol_f4_film_pure_capacity_3seed_exact_ttm_v1"
)
DEFAULT_BENCHMARK_DIR = (
    "outputs/benchmarks/"
    "rq3_news_first_vol_f4_film_pure_capacity_3seed_epoch1_v1"
)
EXPERIMENT_KIND = "news_first_vol_f4_film_pure_capacity_3seed_v1"
BENCHMARK_KIND = f"{EXPERIMENT_KIND}_epoch1_benchmark"
ARCHITECTURES = ("film_cnn", "pure_cnn")
CAPACITIES = ("c08", "c12", "c16", "c24", "c32", "c48")
SEEDS = (42, 202, 404)
GPU_IDS = (0, 1)
PROFILE_FIELDS = (
    "gen_base_channels",
    "gen_res_blocks",
    "gen_hidden_dim",
    "disc_base_channels",
    "disc_res_blocks",
    "disc_text_hidden_dim",
    "disc_hidden_dim",
)
EXPECTED_JOB_COUNT = 36
C32_REGRESSION_RELATIVE_TOLERANCE = 5e-5
C32_REGRESSION_CONTROLS = {
    "film_cnn": {
        "anchor_observed_mae": 0.0016690598968354838,
        "archive_root": REPO_ROOT
        / "outputs/experiments/"
        "rq12_news_first_vol_film_unet_text128_c32_nolp_direct_text_10seed_"
        "lr2p5e5_exact_ttm_rolling_v1",
        "pair_metrics_sha256": (
            "c500e835263d9421b02213c788c23d99bc623ef3be9c9db127fa9b0b81877d9c"
        ),
        "checkpoint_allowlist_sha256": (
            "8917bc368965a6f7ec6736e4aaf8ab866dacf698382a1b3c18ab841c0ed3c391"
        ),
        "panel_sha256": (
            "eb2d930cd3b5dfb35c9e1a6e977ac32551d9e04647c4221f62d988e20dd01de3"
        ),
        "overlay_sha256": (
            "37154b244930934a10a6b974b1a70f7e53ad0f62685180bfa93c2b480858e32e"
        ),
        "arm": "lp_matched",
        "overlay_name": "lp_matched.json",
    },
    "pure_cnn": {
        "anchor_observed_mae": 0.0016693063748754267,
        "archive_root": REPO_ROOT
        / "outputs/experiments/"
        "rq12_news_first_vol_cnn_unet_c32_nolp_pure_no_text_10seed_"
        "exact_ttm_rolling_v1",
        "pair_metrics_sha256": (
            "7219e2f297a411ca8d8a2a6e8e0e3333fe53e04340d3ea5c3c6c64564a25fe48"
        ),
        "checkpoint_allowlist_sha256": (
            "1a0200330ecd976285d0d619870cbaa728545011a1013cc92e8514cdd27bcd25"
        ),
        "panel_sha256": (
            "eb2d930cd3b5dfb35c9e1a6e977ac32551d9e04647c4221f62d988e20dd01de3"
        ),
        "overlay_sha256": (
            "392bf412a5b433d03ba1ed69fdd77ebe86ae8fa8be814e850009f2b6f32a930a"
        ),
        "arm": "pure_cnn_no_text",
        "overlay_name": "pure_cnn_no_text.json",
    },
}
BASE_TRAINING_CONFIG = (
    REPO_ROOT / "configs/rq3/news_first_vol_film_lp_critic_capacity_seed42_lr2p5e7.yaml"
)
SOURCE_PATHS = (
    Path(__file__).resolve(),
    REPO_ROOT / "scripts/rq3/news_first_vol_film_unet_capacity_epoch_3seed.py",
    REPO_ROOT / "scripts/rq3/news_first_vol_training.py",
    REPO_ROOT / "scripts/train/train_vol.py",
    REPO_ROOT / "src/wgan_option/config.py",
    REPO_ROOT / "src/wgan_option/models/common.py",
    REPO_ROOT / "src/wgan_option/models/generator.py",
    REPO_ROOT / "src/wgan_option/models/discriminator.py",
    REPO_ROOT / "src/wgan_option/models/gan_model.py",
    BASE_TRAINING_CONFIG,
)

_payload_sha256 = training._payload_sha256
_sha256_file = training._sha256_file
_utc_now = training._utc_now
_write_json = training._write_json
_write_yaml = training._write_yaml
_atomic_write_text = training._atomic_write_text


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _read_json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = _path(config_path)
    config = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "root")
    config["source_config_path"] = str(source)
    config["source_config_sha256"] = _sha256_file(source)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    experiment = _mapping(config.get("experiment"), "experiment")
    matrix = _mapping(config.get("matrix"), "matrix")
    data = _mapping(config.get("data"), "data")
    training_cfg = _mapping(config.get("training"), "training")
    runtime = _mapping(config.get("runtime"), "runtime")
    if int(experiment.get("schema_version", -1)) != 1:
        raise ValueError("experiment.schema_version must be 1")
    if experiment.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError(f"experiment_kind must be {EXPERIMENT_KIND}")
    if tuple(matrix.get("architectures", ())) != ARCHITECTURES:
        raise ValueError(f"architecture order must be {ARCHITECTURES}")
    if tuple(matrix.get("capacities", ())) != CAPACITIES:
        raise ValueError(f"capacity order must be {CAPACITIES}")
    if tuple(map(int, matrix.get("seeds", ()))) != SEEDS:
        raise ValueError(f"seeds must be {SEEDS}")
    if matrix.get("fold") != "f4_2023q4":
        raise ValueError("Only f4_2023q4 is permitted")
    if int(matrix.get("expected_training_jobs", -1)) != EXPECTED_JOB_COUNT:
        raise ValueError("Expected exactly 36 training jobs")
    boundaries = {
        "train_end_utc": "2023-07-01T00:00:00Z",
        "validation_end_utc": "2023-10-01T00:00:00Z",
        "test_end_utc": "2024-01-01T00:00:00Z",
    }
    for key, expected in boundaries.items():
        if data.get(key) != expected:
            raise ValueError(f"data.{key} must be {expected}")
    expected_counts = {
        "train": {"pairs": 748, "sessions": 203},
        "validation": {"pairs": 135, "sessions": 33},
        "test": {"pairs": 143, "sessions": 45},
    }
    if data.get("expected_counts") != expected_counts:
        raise ValueError("F4 pair/session counts drifted")
    if int(training_cfg.get("num_epochs", -1)) != 240:
        raise ValueError("training.num_epochs must be 240")
    if int(training_cfg.get("early_stopping_min_epochs", -1)) != 30:
        raise ValueError("early_stopping_min_epochs must be 30")
    if int(training_cfg.get("early_stopping_patience", -1)) != 20:
        raise ValueError("early_stopping_patience must be 20")
    if int(training_cfg.get("validation_mc_samples", -1)) != 16:
        raise ValueError("validation_mc_samples must be 16")
    if int(training_cfg.get("prediction_mc_samples", -1)) != 64:
        raise ValueError("prediction_mc_samples must be 64")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != GPU_IDS:
        raise ValueError("runtime.gpu_ids must be [0, 1]")
    for key in ("workers_per_gpu", "benchmark_workers_per_gpu"):
        if int(runtime.get(key, -1)) != 18:
            raise ValueError(f"runtime.{key} must be 18")
    if int(runtime.get("cpu_threads_per_job", -1)) != 1:
        raise ValueError("runtime.cpu_threads_per_job must be 1")
    profiles = _mapping(config.get("profiles"), "profiles")
    if tuple(profiles) != CAPACITIES:
        raise ValueError("profile order drifted")
    if set(profiles["c48"]) != set(PROFILE_FIELDS):
        raise ValueError("c48 profile fields drifted")
    if profiles["c48"] != {
        "gen_base_channels": 48,
        "gen_res_blocks": 0,
        "gen_hidden_dim": 1536,
        "disc_base_channels": 48,
        "disc_res_blocks": 0,
        "disc_text_hidden_dim": 128,
        "disc_hidden_dim": 1179,
    }:
        raise ValueError("c48 width contract drifted")
    # Training paths may be bound now; frozen Q4 paths must remain unread.
    workbook = _path(str(data["workbook"]))
    if not workbook.is_file():
        raise FileNotFoundError(workbook)
    for architecture in ARCHITECTURES:
        overlay = _mapping(
            _mapping(data["development_overlays"], "development_overlays")[architecture],
            f"development_overlays.{architecture}",
        )
        overlay_path = _path(str(overlay["path"]))
        if not overlay_path.is_file() or _sha256_file(overlay_path) != overlay["sha256"]:
            raise ValueError(f"Development overlay drifted: {overlay_path}")


def _architecture_payload(
    config: Mapping[str, Any], architecture: str, capacity: str
) -> dict[str, Any]:
    arch = _mapping(config["architectures"][architecture], architecture)
    profile = _mapping(config["profiles"][capacity], capacity)
    return {
        "architecture": architecture,
        "architecture_label": str(arch["label"]),
        "capacity_id": capacity,
        "generator_conditioning_mode": str(arch["generator_conditioning_mode"]),
        "critic_conditioning_mode": str(arch["critic_conditioning_mode"]),
        "gen_text_hidden_dim": int(arch["gen_text_hidden_dim"]),
        "gen_text_out_dim": int(arch["gen_text_out_dim"]),
        **{field: int(profile[field]) for field in PROFILE_FIELDS},
    }


def _modules(payload: Mapping[str, Any], seed: int) -> tuple[Generator, Discriminator]:
    torch.manual_seed(int(seed))
    generator = Generator(
        channels=1,
        embedding_dim=1024,
        noise_dim=32,
        surface_height=16,
        surface_width=16,
        base_channels=int(payload["gen_base_channels"]),
        res_blocks=int(payload["gen_res_blocks"]),
        text_hidden_dim=int(payload["gen_text_hidden_dim"]),
        text_out_dim=int(payload["gen_text_out_dim"]),
        hidden_dim=int(payload["gen_hidden_dim"]),
        residual_output_mode="identity_softplus_residual",
        generator_noise_mode="gaussian",
        generator_current_input_mode="current_support_masked",
        generator_conditioning_mode=str(payload["generator_conditioning_mode"]),
    )
    critic = Discriminator(
        channels=1,
        embedding_dim=1024,
        surface_height=16,
        surface_width=16,
        base_channels=int(payload["disc_base_channels"]),
        res_blocks=int(payload["disc_res_blocks"]),
        text_hidden_dim=int(payload["disc_text_hidden_dim"]),
        hidden_dim=int(payload["disc_hidden_dim"]),
        critic_normalization_mode="legacy_instance_norm_v1",
        critic_conditioning_mode=str(payload["critic_conditioning_mode"]),
    )
    return generator, critic


def _state_sha(module: torch.nn.Module, keys: Sequence[str] | None = None) -> str:
    digest = hashlib.sha256()
    state = module.state_dict()
    selected = sorted(state) if keys is None else sorted(keys)
    for key in selected:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def experiment_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected_counts = _mapping(config["expected_parameter_counts"], "counts")
    specs: list[dict[str, Any]] = []
    cpu_state = torch.random.get_rng_state()
    try:
        for arch_index, architecture in enumerate(ARCHITECTURES):
            for capacity_index, capacity in enumerate(CAPACITIES):
                payload = _architecture_payload(config, architecture, capacity)
                generator0, critic0 = _modules(payload, 0)
                counts = {
                    "generator_parameters": sum(p.numel() for p in generator0.parameters()),
                    "critic_parameters": sum(p.numel() for p in critic0.parameters()),
                }
                counts["wgan_parameters"] = (
                    counts["generator_parameters"] + counts["critic_parameters"]
                )
                frozen = _mapping(expected_counts[architecture][capacity], "counts cell")
                actual_tuple = (
                    counts["generator_parameters"],
                    counts["critic_parameters"],
                    counts["wgan_parameters"],
                )
                expected_tuple = (
                    int(frozen["generator"]),
                    int(frozen["critic"]),
                    int(frozen["total"]),
                )
                if actual_tuple != expected_tuple:
                    raise ValueError(
                        f"Parameter-count drift for {architecture}/{capacity}: "
                        f"{actual_tuple} != {expected_tuple}"
                    )
                architecture_contract = {
                    **payload,
                    **counts,
                    "generator_conditioning_fingerprint": (
                        generator_conditioning_fingerprint(
                            str(payload["generator_conditioning_mode"])
                        )
                    ),
                    "critic_conditioning_fingerprint": critic_conditioning_fingerprint(
                        str(payload["critic_conditioning_mode"])
                    ),
                }
                architecture_sha = _payload_sha256(architecture_contract)
                model_contract = {
                    "experiment_kind": EXPERIMENT_KIND,
                    "fold_id": "f4_2023q4",
                    "architecture_profile_sha256": architecture_sha,
                    "surface_grid_sha256": config["data"]["surface_grid_sha256"],
                    "train_end_utc": config["data"]["train_end_utc"],
                    "validation_end_utc": config["data"]["validation_end_utc"],
                    "q4_loader_materialized": False,
                }
                model_sha = _payload_sha256(model_contract)
                for seed_index, seed in enumerate(SEEDS):
                    generator, critic = _modules(payload, seed)
                    spec = {
                        "job_id": f"{architecture}_{capacity}_seed_{seed:03d}",
                        "experiment_kind": EXPERIMENT_KIND,
                        **architecture_contract,
                        "architecture_profile_sha256": architecture_sha,
                        "model_contract_sha256": model_sha,
                        "seed": seed,
                        "gpu_id": GPU_IDS[
                            (capacity_index + arch_index + seed_index) % len(GPU_IDS)
                        ],
                        "num_epochs": int(config["training"]["num_epochs"]),
                        "lr_warmup_epochs": int(config["training"]["lr_warmup_epochs"]),
                        "early_stopping_min_epochs": int(
                            config["training"]["early_stopping_min_epochs"]
                        ),
                        "early_stopping_patience": int(
                            config["training"]["early_stopping_patience"]
                        ),
                        "initial_generator_state_sha256": _state_sha(generator),
                        "initial_critic_state_sha256": _state_sha(critic),
                    }
                    spec["job_spec_sha256"] = _payload_sha256(spec)
                    specs.append(spec)
    finally:
        torch.random.set_rng_state(cpu_state)
    if len(specs) != EXPECTED_JOB_COUNT or len({row["job_id"] for row in specs}) != 36:
        raise AssertionError("Expected 36 unique jobs")
    if {gpu: sum(row["gpu_id"] == gpu for row in specs) for gpu in GPU_IDS} != {
        0: 18,
        1: 18,
    }:
        raise AssertionError("GPU assignment must be balanced 18/18")
    for capacity in CAPACITIES:
        rows = [row for row in specs if row["capacity_id"] == capacity]
        if {gpu: sum(row["gpu_id"] == gpu for row in rows) for gpu in GPU_IDS} != {
            0: 3,
            1: 3,
        }:
            raise AssertionError(f"Capacity/GPU imbalance for {capacity}")
    return specs


def _fairness_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cpu_state = torch.random.get_rng_state()
    try:
        for capacity in CAPACITIES:
            film_payload = _architecture_payload(config, "film_cnn", capacity)
            pure_payload = _architecture_payload(config, "pure_cnn", capacity)
            for seed in SEEDS:
                film_g, film_d = _modules(film_payload, seed)
                pure_g, pure_d = _modules(pure_payload, seed)
                if _state_sha(film_d) != _state_sha(pure_d):
                    raise AssertionError(f"Critic initialization drift: {capacity}/{seed}")
                film_state = film_g.state_dict()
                pure_state = pure_g.state_dict()
                shared = [
                    key
                    for key in film_state.keys() & pure_state.keys()
                    if film_state[key].shape == pure_state[key].shape
                ]
                if not shared or _state_sha(film_g, shared) != _state_sha(pure_g, shared):
                    raise AssertionError(
                        f"Shared Generator initialization drift: {capacity}/{seed}"
                    )
                rows.append(
                    {
                        "capacity_id": capacity,
                        "seed": seed,
                        "shared_generator_state_sha256": _state_sha(film_g, shared),
                        "critic_state_sha256": _state_sha(film_d),
                        "shared_generator_tensor_count": len(shared),
                    }
                )
    finally:
        torch.random.set_rng_state(cpu_state)
    payload = {
        "schema_version": 1,
        "contract": "same_seed_same_capacity_shared_backbone_and_critic_v1",
        "rows": rows,
    }
    return {**payload, "fairness_contract_sha256": _payload_sha256(payload)}


def _training_payload(
    config: Mapping[str, Any], root: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _mapping(
        yaml.safe_load(BASE_TRAINING_CONFIG.read_text(encoding="utf-8")), "base config"
    )
    data = _mapping(config["data"], "data")
    train = _mapping(config["training"], "training")
    overlay = _mapping(data["development_overlays"][spec["architecture"]], "overlay")
    workbook = _path(str(data["workbook"]))
    overlay_path = _path(str(overlay["path"]))
    is_film = spec["architecture"] == "film_cnn"
    payload.update(
        {
            "data_path": str(workbook),
            "sheet_name": str(data["sheet_name"]),
            "text_embedding_mode": "lp",
            "news_first_common_eval_data_path": str(workbook),
            "news_first_train_end_utc": str(data["train_end_utc"]),
            "news_first_validation_end_utc": str(data["validation_end_utc"]),
            "news_first_data_window_end_utc_exclusive": str(
                data["validation_end_utc"]
            ),
            "news_first_dataset_tolerance_minutes": 5,
            "support_mask_mode": str(data["support_mask_mode"]),
            "news_first_text_ablation_mode": "real_text",
            "news_first_text_information_path": (
                "lp_matched" if is_film else "pure_cnn_no_text"
            ),
            "news_first_pair_text_overlay_mode": str(overlay["mode"]),
            "news_first_pair_text_manifest_path": str(overlay_path),
            "news_first_pair_text_manifest_sha256": str(overlay["sha256"]),
            "news_first_pair_text_profile_sha256": str(overlay["profile_sha256"]),
            "news_first_full_training_state_mode": "none",
            "news_first_full_training_state_contract_path": "",
            "news_first_full_training_state_contract_sha256": "",
            "news_first_materialize_validation_loader": True,
            "news_first_materialize_test_loader": False,
            "news_first_capacity_profile": str(spec["capacity_id"]),
            "news_first_capacity_profile_sha256": str(
                spec["architecture_profile_sha256"]
            ),
            "news_first_architecture_profile_sha256": str(
                spec["architecture_profile_sha256"]
            ),
            "news_first_model_contract_sha256": str(spec["model_contract_sha256"]),
            "news_first_surface_grid_profile": str(data["surface_grid_profile"]),
            "news_first_surface_grid_sha256": str(data["surface_grid_sha256"]),
            "generator_conditioning_mode": str(
                spec["generator_conditioning_mode"]
            ),
            "critic_conditioning_mode": str(spec["critic_conditioning_mode"]),
            "gen_text_hidden_dim": int(spec["gen_text_hidden_dim"]),
            "gen_text_out_dim": int(spec["gen_text_out_dim"]),
            **{field: int(spec[field]) for field in PROFILE_FIELDS},
            "learning_rate": float(train["initial_learning_rate"]),
            "generator_learning_rate": float(train["generator_learning_rate"]),
            "discriminator_learning_rate": float(
                train["discriminator_learning_rate"]
            ),
            "reduce_lr_factor": float(train["reduce_lr_factor"]),
            "reduce_lr_patience": int(train["reduce_lr_patience"]),
            "reduce_lr_min_lr": float(train["scheduler_min_lr"]),
            "lr_warmup_epochs": int(spec["lr_warmup_epochs"]),
            "num_epochs": int(spec["num_epochs"]),
            "batch_size": int(train["batch_size"]),
            "discriminator_iter": int(train["discriminator_steps"]),
            "beta_1": float(train["beta_1"]),
            "beta_2": float(train["beta_2"]),
            "lambda_gp": float(train["lambda_gp"]),
            "lambda_recon": float(train["lambda_recon"]),
            "lambda_calendar": float(train["lambda_calendar"]),
            "lambda_butterfly": float(train["lambda_butterfly"]),
            "lambda_smooth": float(train["lambda_smooth"]),
            "lambda_delta_shrink": float(train["lambda_delta_shrink"]),
            "constraint_warmup_epochs": int(train["constraint_warmup_epochs"]),
            "best_checkpoint_metric": str(train["best_checkpoint_metric"]),
            "baseline_penalty_weight": float(train["baseline_penalty_weight"]),
            "use_early_stopping": True,
            "early_stopping_min_epochs": int(spec["early_stopping_min_epochs"]),
            "early_stopping_patience": int(spec["early_stopping_patience"]),
            "early_stopping_min_delta": float(train["early_stopping_min_delta"]),
            "validation_mc_samples": int(train["validation_mc_samples"]),
            "evaluate_initial_checkpoint": True,
            "seed": int(spec["seed"]),
            "cuda": True,
            "num_workers": 0,
            "save_every": 1_000_000,
            "generator_optimizer_profile": (
                "film_unet_split_lr_v1" if is_film else "uniform_v1"
            ),
            "generator_text_learning_rate": (
                float(train["generator_text_learning_rate"]) if is_film else 0.0
            ),
            "generator_film_learning_rate": (
                float(train["generator_film_learning_rate"]) if is_film else 0.0
            ),
            "generator_text_min_learning_rate": (
                float(train["generator_text_min_learning_rate"]) if is_film else 0.0
            ),
            "generator_film_min_learning_rate": (
                float(train["generator_film_min_learning_rate"]) if is_film else 0.0
            ),
            "news_first_lr_profile": (
                "film_unet_split_lr_v1" if is_film else "pure_cnn_uniform_5e7_v1"
            ),
            "output_root": str(
                (
                    root
                    / "runs"
                    / str(spec["architecture"])
                    / str(spec["capacity_id"])
                    / f"seed_{int(spec['seed']):03d}"
                ).resolve()
            ),
        }
    )
    return payload


def _source_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    data = _mapping(config["data"], "data")
    overlays = _mapping(data["development_overlays"], "development_overlays")
    paths = [*SOURCE_PATHS, Path(str(config["source_config_path"])), _path(data["workbook"])]
    paths.extend(_path(overlays[architecture]["path"]) for architecture in ARCHITECTURES)
    return {str(path.resolve()): _sha256_file(path) for path in paths}


def _registry_path(root: Path) -> Path:
    return root / "registry/jobs.json"


def _status_path(root: Path, job_id: str) -> Path:
    return root / f"registry/jobs/{job_id}.status.json"


def _counts(registry: Mapping[str, Any], root: Path) -> dict[str, int]:
    result = {"pending": 0, "running": 0, "complete": 0, "failed": 0}
    for job in registry["jobs"]:
        state = str(_read_json(_status_path(root, str(job["job_id"]))).get("state"))
        result[state] = result.get(state, 0) + 1
    return result


def _prepare(
    config: Mapping[str, Any], root: Path, *, resume: bool, benchmark: bool
) -> dict[str, Any]:
    registry_path = _registry_path(root)
    specs = experiment_specs(config)
    if benchmark:
        converted: list[dict[str, Any]] = []
        for formal in specs:
            spec = {key: value for key, value in formal.items() if key != "job_spec_sha256"}
            spec.update(
                experiment_kind=BENCHMARK_KIND,
                num_epochs=1,
                early_stopping_min_epochs=1,
                early_stopping_patience=1,
                benchmark=True,
            )
            spec["job_spec_sha256"] = _payload_sha256(spec)
            converted.append(spec)
        specs = converted
    source_hashes = _source_hashes(config)
    if registry_path.is_file():
        if not resume:
            raise FileExistsError(f"Registry exists; pass --resume: {registry_path}")
        registry = _read_json(registry_path)
        expected_kind = BENCHMARK_KIND if benchmark else EXPERIMENT_KIND
        if registry.get("experiment_kind") != expected_kind:
            raise ValueError("Registry experiment kind drifted")
        if registry.get("source_hashes") != source_hashes:
            raise ValueError("Source/config/data drift detected; resume refused")
        if registry.get("jobs_payload_sha256") != _payload_sha256(specs):
            raise ValueError("Job matrix drift detected; resume refused")
        for job in registry["jobs"]:
            path = Path(str(job["config_path"]))
            if not path.is_file() or _sha256_file(path) != job["config_sha256"]:
                raise ValueError(f"Generated training config drift: {path}")
        return registry
    for directory in (
        "analysis",
        "configs",
        "control",
        "control/job_locks",
        "logs",
        "registry/jobs",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for spec in specs:
        path = root / f"configs/{spec['job_id']}.yaml"
        payload = _training_payload(config, root, spec)
        if benchmark:
            payload.update(
                num_epochs=1,
                use_early_stopping=False,
                early_stopping_min_epochs=1,
                early_stopping_patience=1,
                news_first_lr_profile="f4_capacity_epoch1_benchmark",
            )
        _write_yaml(path, payload)
        job = {
            **spec,
            "config_path": str(path.resolve()),
            "config_sha256": _sha256_file(path),
            "run_root": str(Path(payload["output_root"]).resolve()),
        }
        jobs.append(job)
        _write_json(
            _status_path(root, str(spec["job_id"])),
            {
                "job_id": spec["job_id"],
                "state": "pending",
                "attempt": 0,
                "updated_at": _utc_now(),
                "job_spec_sha256": spec["job_spec_sha256"],
            },
        )
    fairness_path = root / "registry/initial_state_fairness.json"
    fairness = _fairness_manifest(config)
    _write_json(fairness_path, fairness)
    registry = {
        "schema_version": 1,
        "experiment_kind": BENCHMARK_KIND if benchmark else EXPERIMENT_KIND,
        "created_at": _utc_now(),
        "source_config_path": str(config["source_config_path"]),
        "source_hashes": source_hashes,
        "expected_job_count": EXPECTED_JOB_COUNT,
        "jobs_payload_sha256": _payload_sha256(specs),
        "initial_state_fairness_path": str(fairness_path.resolve()),
        "initial_state_fairness_sha256": _sha256_file(fairness_path),
        "jobs": jobs,
    }
    _write_json(registry_path, registry)
    return registry


def _benchmark_summary(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _path(config["experiment"]["benchmark_root"]) / "analysis/benchmark_summary.json"
    if not path.is_file():
        raise RuntimeError("Formal preparation requires the completed resource benchmark")
    payload = _read_json(path)
    if (
        payload.get("experiment_kind") != BENCHMARK_KIND
        or not bool(payload.get("gate_passed"))
        or int(payload.get("completed_jobs", -1)) != EXPECTED_JOB_COUNT
    ):
        raise RuntimeError("Resource benchmark did not pass the frozen gate")
    return path, payload


def prepare(config: Mapping[str, Any], root: Path, *, resume: bool) -> dict[str, Any]:
    _benchmark_summary(config)
    return _prepare(config, root, resume=resume, benchmark=False)


def dry_run(config: Mapping[str, Any], root: Path, *, resume: bool) -> dict[str, Any]:
    registry = prepare(config, root, resume=resume)
    manifest_path = root / "control/dry_run_manifest.json"
    if manifest_path.is_file():
        return _read_json(manifest_path)
    chosen = [
        job
        for job in registry["jobs"]
        if int(job["seed"]) == SEEDS[0]
    ]
    runtime = _mapping(config["runtime"], "runtime")
    artifacts: list[dict[str, Any]] = []
    for job in chosen:
        log_path = root / f"logs/dry_run_{job['architecture']}_{job['capacity_id']}.log"
        env = os.environ.copy()
        env.update(
            PYTHONPATH="src:.",
            CUDA_VISIBLE_DEVICES=str(job["gpu_id"]),
            OMP_NUM_THREADS=str(runtime["cpu_threads_per_job"]),
            MKL_NUM_THREADS=str(runtime["cpu_threads_per_job"]),
        )
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
        if result.returncode:
            raise RuntimeError(f"Dry run failed; see {log_path}")
        artifacts.append(proven._artifact(log_path, f"dry_run::{job['job_id']}"))
    result = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "checked_jobs": len(chosen),
        "artifacts": artifacts,
        "completed_at": _utc_now(),
    }
    _write_json(manifest_path, result)
    return result


def _write_supervisor_status(
    registry: Mapping[str, Any], root: Path, state: str, active: Mapping[str, Any], message: str = ""
) -> None:
    _write_json(
        root / "control/status.json",
        {
            "experiment_kind": registry["experiment_kind"],
            "state": state,
            "pid": os.getpid(),
            "pid_alive": True,
            "updated_at": _utc_now(),
            "counts": _counts(registry, root),
            "active_jobs": sorted(active),
            "message": message,
        },
    )


def _supervise(
    config: Mapping[str, Any], root: Path, registry: Mapping[str, Any], *, benchmark: bool
) -> dict[str, Any]:
    lock_path = root / f"control/{'benchmark' if benchmark else 'supervisor'}.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another supervisor owns this output root") from exc
    _atomic_write_text(root / "control/supervisor.pid", f"{os.getpid()}\n")
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    pending: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in GPU_IDS}
    for raw in registry["jobs"]:
        job = dict(raw)
        job_status = _read_json(_status_path(root, str(job["job_id"])))
        if job_status.get("state") == "complete":
            proven._verify_artifacts(job_status)
            continue
        recovered = proven._discover_completed(job)
        if recovered is not None:
            _write_json(
                _status_path(root, str(job["job_id"])),
                {
                    "job_id": job["job_id"],
                    "state": "complete",
                    "attempt": int(job_status.get("attempt", 0)),
                    "updated_at": _utc_now(),
                    "job_spec_sha256": job["job_spec_sha256"],
                    **recovered,
                },
            )
            continue
        pending[int(job["gpu_id"])].append(job)
    runtime = _mapping(config["runtime"], "runtime")
    workers = int(
        runtime["benchmark_workers_per_gpu" if benchmark else "workers_per_gpu"]
    )
    active: dict[str, dict[str, Any]] = {}
    failure = ""
    last_resource = 0.0
    _write_supervisor_status(registry, root, "running", active)
    while any(pending.values()) or active:
        if stop_requested or failure:
            for record in active.values():
                proven._terminate_process_group(record["process"])
                record["handle"].close()
            state = "interrupted" if stop_requested else "failed"
            _write_supervisor_status(registry, root, state, {}, failure)
            raise RuntimeError(failure or "Supervisor interrupted")
        for gpu in GPU_IDS:
            running = sum(record["gpu_id"] == gpu for record in active.values())
            while running < workers and pending[gpu]:
                job = pending[gpu].pop(0)
                job_id = str(job["job_id"])
                previous = _read_json(_status_path(root, job_id))
                attempt = int(previous.get("attempt", 0)) + 1
                log_path = root / f"logs/{job_id}.attempt_{attempt:02d}.log"
                handle = log_path.open("w", encoding="utf-8")
                env = os.environ.copy()
                env.update(
                    PYTHONPATH="src:.",
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    OMP_NUM_THREADS=str(runtime["cpu_threads_per_job"]),
                    MKL_NUM_THREADS=str(runtime["cpu_threads_per_job"]),
                )
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
                _write_json(
                    _status_path(root, job_id),
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
                print(f"launched {job_id} pid={process.pid} gpu={gpu}", flush=True)
                running += 1
        for job_id, record in list(active.items()):
            returncode = record["process"].poll()
            if returncode is None:
                continue
            record["handle"].close()
            completed = (
                proven._discover_completed(record["job"]) if returncode == 0 else None
            )
            if completed is None:
                failure = (
                    f"{job_id} failed verification with return code {returncode}; "
                    f"see {record['log_path']}"
                )
                _write_json(
                    _status_path(root, job_id),
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
                _write_json(
                    _status_path(root, job_id),
                    {
                        "job_id": job_id,
                        "state": "complete",
                        "attempt": record["attempt"],
                        "returncode": 0,
                        "updated_at": _utc_now(),
                        "job_spec_sha256": record["job"]["job_spec_sha256"],
                        "stdout_log": str(record["log_path"].resolve()),
                        **completed,
                    },
                )
                print(f"completed {job_id}", flush=True)
            del active[job_id]
        now = time.monotonic()
        if now - last_resource >= float(runtime["resource_sample_interval_seconds"]):
            proven._resource_snapshot(root)
            last_resource = now
        _write_supervisor_status(registry, root, "running", active, failure)
        if any(pending.values()) or active:
            time.sleep(float(runtime["poll_interval_seconds"]))
    proven._resource_snapshot(root)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_kind": registry["experiment_kind"],
        "completed_jobs": EXPECTED_JOB_COUNT,
        "expected_jobs": EXPECTED_JOB_COUNT,
        "workers_per_gpu": workers,
        "gpu_job_counts": {"0": 18, "1": 18},
        "completed_at": _utc_now(),
    }
    if benchmark:
        peaks = proven._resource_peaks(root)
        gate = proven._benchmark_resource_gate(peaks, runtime)
        result.update(
            all_training_complete=True,
            all_metrics_finite=True,
            all_generator_parameters_updated=True,
            all_critic_parameters_updated=True,
            resource_peaks=peaks,
            resource_gate=gate,
            gate_passed=bool(gate["passed"]),
        )
        _write_json(root / "analysis/benchmark_summary.json", result)
        if not gate["passed"]:
            _write_supervisor_status(registry, root, "failed", {}, "resource gate failed")
            raise RuntimeError("Benchmark resource gate failed")
    _write_supervisor_status(registry, root, "complete", {})
    return result


def benchmark(config: Mapping[str, Any], root: Path, *, resume: bool) -> dict[str, Any]:
    registry = _prepare(config, root, resume=resume, benchmark=True)
    return _supervise(config, root, registry, benchmark=True)


def launch(config: Mapping[str, Any], root: Path, *, resume: bool) -> dict[str, Any]:
    _benchmark_summary(config)
    registry = prepare(config, root, resume=resume)
    return _supervise(config, root, registry, benchmark=False)


def worker(config: Mapping[str, Any], root: Path, job_id: str) -> dict[str, Any]:
    registry = prepare(config, root, resume=True)
    matches = [job for job in registry["jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown job id: {job_id}")
    job = matches[0]
    runtime = _mapping(config["runtime"], "runtime")
    env = os.environ.copy()
    env.update(
        PYTHONPATH="src:.",
        CUDA_VISIBLE_DEVICES=str(job["gpu_id"]),
        OMP_NUM_THREADS=str(runtime["cpu_threads_per_job"]),
        MKL_NUM_THREADS=str(runtime["cpu_threads_per_job"]),
    )
    result = subprocess.run(
        [
            str(runtime["python_executable"]),
            "scripts/train/train_vol.py",
            "--config",
            str(job["config_path"]),
            "--train-only",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Training exited with {result.returncode}")
    completed = proven._discover_completed(job)
    if completed is None:
        raise RuntimeError("Training exited zero but artifacts failed verification")
    return completed


def status(root: Path) -> dict[str, Any]:
    registry_path = _registry_path(root)
    if not registry_path.is_file():
        return {"state": "not_prepared", "output_root": str(root)}
    registry = _read_json(registry_path)
    status_path = root / "control/status.json"
    result = _read_json(status_path) if status_path.is_file() else {}
    pid_path = root / "control/supervisor.pid"
    pid = int(pid_path.read_text(encoding="utf-8").strip()) if pid_path.is_file() else 0
    result.update(
        output_root=str(root),
        pid=pid,
        pid_alive=bool(pid and proven._pid_alive(pid)),
        counts=_counts(registry, root),
    )
    return result


def freeze_checkpoints(root: Path) -> Path:
    """Freeze and validate the 36 Q3-selected Generator/Critic pairs."""

    from scripts.rq3.news_first_vol_f4_film_pure_capacity_3seed_evaluation import (
        freeze_checkpoints as freeze_evaluation_checkpoints,
    )

    return freeze_evaluation_checkpoints(root)


def evaluate_f4(
    config: Mapping[str, Any], root: Path, *, resume: bool
) -> Path:
    """Evaluate every frozen checkpoint on the common F4 MC64 panel."""

    from scripts.rq3.news_first_vol_f4_film_pure_capacity_3seed_evaluation import (
        evaluate_f4 as evaluate_frozen_f4,
    )

    return evaluate_frozen_f4(config, root, resume=resume)


def postprocess(config: Mapping[str, Any], root: Path) -> Mapping[str, Path]:
    """Create the auditable single-panel capacity summary and LaTeX table."""

    from scripts.rq3.news_first_vol_f4_film_pure_capacity_3seed_analysis import (
        postprocess_experiment,
    )

    pair_metrics_path = root / "evaluation/f4_pair_metrics.csv.gz"
    return postprocess_experiment(
        root,
        pair_metrics_path,
        parameter_counts=_mapping(
            config["expected_parameter_counts"], "expected_parameter_counts"
        ),
    )


def _read_signed_json(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    signature = payload.get("payload_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if not isinstance(signature, str) or signature != _payload_sha256(unsigned):
        raise ValueError(f"Signed JSON self-hash drift: {path}")
    return payload


def _verify_declared_file(path_value: Any, sha256_value: Any, label: str) -> Path:
    path = Path(str(path_value)).resolve()
    expected_sha256 = str(sha256_value)
    if (
        not path.is_file()
        or not expected_sha256
        or _sha256_file(path) != expected_sha256
    ):
        raise ValueError(f"{label} is missing or its declared SHA-256 drifted: {path}")
    return path


def _pair_panel_identity(frame: Any, *, label: str) -> str:
    required = {"pair_id", "session_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{label} lacks pair/session identifiers")
    pairs = frame.loc[:, ["pair_id", "session_id"]].astype(str).drop_duplicates()
    if len(frame) != 143 or len(pairs) != 143 or pairs["session_id"].nunique() != 45:
        raise ValueError(f"{label} must contain 143 unique pairs in 45 sessions")
    rows = sorted(
        pairs.to_dict(orient="records"),
        key=lambda row: (str(row["pair_id"]), str(row["session_id"])),
    )
    return _payload_sha256(
        {
            "schema_version": 1,
            "kind": "f4_pair_session_panel_identity_v1",
            "rows": rows,
        }
    )


def _c32_mae_regression(observed: float, anchor: float) -> dict[str, Any]:
    delta = float(observed) - float(anchor)
    relative_delta = delta / float(anchor)
    return {
        "anchor_observed_mae": float(anchor),
        "current_observed_mae": float(observed),
        "delta_current_minus_anchor": delta,
        "relative_delta_current_minus_anchor": relative_delta,
        "absolute_relative_delta": abs(relative_delta),
        "within_relative_tolerance": bool(
            abs(relative_delta) <= C32_REGRESSION_RELATIVE_TOLERANCE
        ),
    }


def _write_c32_regression_control(
    root: Path,
    pair_metrics: Any,
    checkpoint_allowlist_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Compare the rerun c32 cells with immutable three-seed F4 controls.

    This is deliberately a numerical regression gate, not a bitwise-
    determinism claim.  Both selected Generator and Critic file hashes are
    reported per seed and role, even when the MAE-level control passes.
    """

    import pandas as pd

    required_current = {
        "architecture",
        "capacity_id",
        "seed",
        "pair_id",
        "session_id",
        "target_mae",
        "checkpoint_sha256",
        "noise_bank_profile_sha256",
        "panel_sha256",
        "test_overlay_sha256",
    }
    if not required_current.issubset(pair_metrics.columns):
        missing = sorted(required_current - set(pair_metrics.columns))
        raise ValueError(f"Current F4 pair metrics lack c32 control fields: {missing}")
    current_allowlist = pd.read_csv(
        checkpoint_allowlist_path, dtype=str, keep_default_na=False
    )
    expected_roles = {"generator_best_learned", "discriminator_best_learned"}
    architecture_results: dict[str, Any] = {}
    all_checkpoints_identical = True
    all_panels_identical = True
    all_within_tolerance = True

    for architecture in ARCHITECTURES:
        control = _mapping(
            C32_REGRESSION_CONTROLS[architecture],
            f"C32_REGRESSION_CONTROLS[{architecture}]",
        )
        archive_root = Path(control["archive_root"]).resolve()
        archive_pair_path = _verify_declared_file(
            archive_root / "analysis/rq12_pair_metrics.csv.gz",
            control["pair_metrics_sha256"],
            f"{architecture} archived c32 pair metrics",
        )
        archive_allowlist_path = _verify_declared_file(
            archive_root / "registry/evaluation_checkpoint_allowlist.csv",
            control["checkpoint_allowlist_sha256"],
            f"{architecture} archived checkpoint allowlist",
        )
        archive_panel_path = _verify_declared_file(
            archive_root
            / "evaluation/test_panels/tolerance_05m/f4_2023q4.csv.gz",
            control["panel_sha256"],
            f"{architecture} archived F4 panel",
        )
        archive_overlay_path = _verify_declared_file(
            archive_root
            / "evaluation/test_pair_text_overlays/tolerance_05m/f4_2023q4"
            / str(control["overlay_name"]),
            control["overlay_sha256"],
            f"{architecture} archived F4 overlay",
        )
        archive_pairs = pd.read_csv(archive_pair_path, low_memory=False)
        archive_pairs = archive_pairs.loc[
            archive_pairs["fold"].astype(str).eq("f4_2023q4")
            & archive_pairs["arm"].astype(str).eq(str(control["arm"]))
            & archive_pairs["seed"].astype(int).isin(SEEDS)
        ].copy()
        current_pairs = pair_metrics.loc[
            pair_metrics["architecture"].astype(str).eq(architecture)
            & pair_metrics["capacity_id"].astype(str).eq("c32")
        ].copy()
        for label, frame in (
            (f"current {architecture} c32", current_pairs),
            (f"archived {architecture} c32", archive_pairs),
        ):
            if (
                len(frame) != len(SEEDS) * 143
                or set(frame["seed"].astype(int)) != set(SEEDS)
                or frame.duplicated(["seed", "pair_id"]).any()
            ):
                raise ValueError(f"{label} is not the expected 3 x 143 panel")

        current_panel_hashes = set(current_pairs["panel_sha256"].astype(str))
        current_overlay_hashes = set(current_pairs["test_overlay_sha256"].astype(str))
        if current_panel_hashes != {str(control["panel_sha256"])}:
            raise ValueError(f"{architecture} current panel SHA differs from its control")
        if current_overlay_hashes != {str(control["overlay_sha256"])}:
            raise ValueError(f"{architecture} current overlay SHA differs from its control")

        archive_allowlist = pd.read_csv(
            archive_allowlist_path, dtype=str, keep_default_na=False
        )
        archive_allowlist = archive_allowlist.loc[
            archive_allowlist["fold"].eq("f4_2023q4")
            & archive_allowlist["arm"].eq(str(control["arm"]))
            & archive_allowlist["seed"].astype(int).isin(SEEDS)
            & archive_allowlist["checkpoint_role"].isin(expected_roles)
        ].copy()
        current_checkpoints = current_allowlist.loc[
            current_allowlist["architecture"].eq(architecture)
            & current_allowlist["capacity_id"].eq("c32")
            & current_allowlist["seed"].astype(int).isin(SEEDS)
            & current_allowlist["checkpoint_role"].isin(expected_roles)
        ].copy()
        expected_checkpoint_keys = {
            (seed, role) for seed in SEEDS for role in expected_roles
        }
        for label, frame in (
            ("current", current_checkpoints),
            ("archived", archive_allowlist),
        ):
            keys = set(
                zip(
                    frame["seed"].astype(int),
                    frame["checkpoint_role"].astype(str),
                    strict=True,
                )
            )
            if len(frame) != 6 or keys != expected_checkpoint_keys:
                raise ValueError(
                    f"{architecture} {label} c32 allowlist is not three G/D pairs"
                )
            for row in frame.itertuples(index=False):
                _verify_declared_file(
                    row.checkpoint_path,
                    row.checkpoint_sha256,
                    f"{architecture} {label} {row.seed}/{row.checkpoint_role}",
                )

        seed_results: list[dict[str, Any]] = []
        architecture_checkpoints_identical = True
        architecture_panels_identical = True
        for seed in SEEDS:
            current_seed = current_pairs.loc[current_pairs["seed"].astype(int).eq(seed)]
            archive_seed = archive_pairs.loc[archive_pairs["seed"].astype(int).eq(seed)]
            current_panel_id = _pair_panel_identity(
                current_seed, label=f"current {architecture}/seed {seed}"
            )
            archive_panel_id = _pair_panel_identity(
                archive_seed, label=f"archived {architecture}/seed {seed}"
            )
            panel_identical = current_panel_id == archive_panel_id
            architecture_panels_identical &= panel_identical

            checkpoint_rows: list[dict[str, Any]] = []
            seed_checkpoints_identical = True
            for role in sorted(expected_roles):
                current_checkpoint = current_checkpoints.loc[
                    current_checkpoints["seed"].astype(int).eq(seed)
                    & current_checkpoints["checkpoint_role"].eq(role)
                ].iloc[0]
                archive_checkpoint = archive_allowlist.loc[
                    archive_allowlist["seed"].astype(int).eq(seed)
                    & archive_allowlist["checkpoint_role"].eq(role)
                ].iloc[0]
                bitwise_identical = str(current_checkpoint["checkpoint_sha256"]) == str(
                    archive_checkpoint["checkpoint_sha256"]
                )
                seed_checkpoints_identical &= bitwise_identical
                checkpoint_rows.append(
                    {
                        "checkpoint_role": role,
                        "current_sha256": str(current_checkpoint["checkpoint_sha256"]),
                        "archive_sha256": str(archive_checkpoint["checkpoint_sha256"]),
                        "bitwise_identical": bool(bitwise_identical),
                    }
                )
            architecture_checkpoints_identical &= seed_checkpoints_identical

            current_generator_sha = next(
                row["current_sha256"]
                for row in checkpoint_rows
                if row["checkpoint_role"] == "generator_best_learned"
            )
            archive_generator_sha = next(
                row["archive_sha256"]
                for row in checkpoint_rows
                if row["checkpoint_role"] == "generator_best_learned"
            )
            if set(current_seed["checkpoint_sha256"].astype(str)) != {
                current_generator_sha
            } or set(archive_seed["checkpoint_sha256"].astype(str)) != {
                archive_generator_sha
            }:
                raise ValueError(
                    f"{architecture}/seed {seed} pair evidence checkpoint drift"
                )
            current_noise = sorted(
                set(current_seed["noise_bank_profile_sha256"].astype(str))
            )
            archive_noise = sorted(
                set(archive_seed["noise_bank_profile_sha256"].astype(str))
            )
            if len(current_noise) != 1 or len(archive_noise) != 1:
                raise ValueError(
                    f"{architecture}/seed {seed} must have one noise-bank profile"
                )
            seed_results.append(
                {
                    "seed": seed,
                    "current_observed_mae": float(current_seed["target_mae"].mean()),
                    "archive_observed_mae": float(archive_seed["target_mae"].mean()),
                    "delta_current_minus_archive": float(
                        current_seed["target_mae"].mean()
                        - archive_seed["target_mae"].mean()
                    ),
                    "current_pair_panel_id_sha256": current_panel_id,
                    "archive_pair_panel_id_sha256": archive_panel_id,
                    "pair_panel_identical": bool(panel_identical),
                    "current_noise_bank_profile_sha256": current_noise[0],
                    "archive_noise_bank_profile_sha256": archive_noise[0],
                    "noise_bank_profile_sha256_identical": current_noise == archive_noise,
                    "checkpoints": checkpoint_rows,
                    "all_checkpoints_bitwise_identical": bool(
                        seed_checkpoints_identical
                    ),
                }
            )

        anchor = float(control["anchor_observed_mae"])
        current_observed = float(
            current_pairs.groupby(current_pairs["seed"].astype(int))["target_mae"]
            .mean()
            .mean()
        )
        archive_observed = float(
            archive_pairs.groupby(archive_pairs["seed"].astype(int))["target_mae"]
            .mean()
            .mean()
        )
        if abs(archive_observed - anchor) > 1e-15:
            raise ValueError(
                f"{architecture} archived evidence no longer reproduces its MAE anchor"
            )
        comparison = _c32_mae_regression(current_observed, anchor)
        within_tolerance = bool(comparison["within_relative_tolerance"])
        all_within_tolerance &= within_tolerance
        all_panels_identical &= architecture_panels_identical
        all_checkpoints_identical &= architecture_checkpoints_identical
        architecture_results[architecture] = {
            **comparison,
            "archive_recomputed_observed_mae": archive_observed,
            "pair_panels_identical": bool(architecture_panels_identical),
            "all_checkpoints_bitwise_identical": bool(
                architecture_checkpoints_identical
            ),
            "archive_sources": {
                "pair_metrics_path": str(archive_pair_path),
                "pair_metrics_sha256": _sha256_file(archive_pair_path),
                "checkpoint_allowlist_path": str(archive_allowlist_path),
                "checkpoint_allowlist_sha256": _sha256_file(
                    archive_allowlist_path
                ),
                "panel_path": str(archive_panel_path),
                "panel_sha256": _sha256_file(archive_panel_path),
                "overlay_path": str(archive_overlay_path),
                "overlay_sha256": _sha256_file(archive_overlay_path),
            },
            "seeds": seed_results,
        }

    passed = all_within_tolerance and all_panels_identical
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "f4_c32_regression_control_v1",
        "status": "passed" if passed else "failed",
        "fold_id": "f4_2023q4",
        "capacity_id": "c32",
        "seeds": list(SEEDS),
        "aggregation": "mean_143_pairs_within_seed_then_equal_mean_across_3_seeds",
        "relative_mae_tolerance": C32_REGRESSION_RELATIVE_TOLERANCE,
        "tolerance_rationale": (
            "The 5e-5 relative gate is 0.005% of the approximately 1.67e-3 "
            "MAE anchor. It is a tight numerical regression guard for an "
            "independent full rerun; checkpoint and noise-profile hashes are "
            "reported separately and are not allowed to imply bitwise determinism."
        ),
        "current_pair_metrics_path": str(
            (root / "evaluation/f4_pair_metrics.csv.gz").resolve()
        ),
        "current_pair_metrics_sha256": _sha256_file(
            root / "evaluation/f4_pair_metrics.csv.gz"
        ),
        "current_checkpoint_allowlist_path": str(checkpoint_allowlist_path.resolve()),
        "current_checkpoint_allowlist_sha256": _sha256_file(
            checkpoint_allowlist_path
        ),
        "all_mae_controls_within_tolerance": bool(all_within_tolerance),
        "all_pair_panels_identical": bool(all_panels_identical),
        "all_checkpoints_bitwise_identical": bool(all_checkpoints_identical),
        "exact_determinism_claimed": False,
        "interpretation": (
            "A passing MAE regression control establishes close numerical "
            "agreement on the same pair/session panel only. Differing checkpoint "
            "or noise-profile hashes mean the rerun is not bitwise identical."
        ),
        "architectures": architecture_results,
        "completed_at": _utc_now(),
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    path = root / "analysis/f4_c32_regression_control.json"
    _write_json(path, payload)
    return path, payload


def qa(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Validate all frozen training, F4-evaluation, and analysis artifacts."""

    import pandas as pd

    from scripts.rq3 import (
        news_first_vol_f4_film_pure_capacity_3seed_analysis as analysis,
    )

    # The idempotent freeze call performs the full 72-checkpoint allowlist and
    # source-artifact validation without opening any new Q4 inputs.
    allowlist_path = freeze_checkpoints(root)
    state_path = root / "registry/f4_evaluation_state.json"
    state = _read_signed_json(state_path)
    required_state = {
        "checkpoints_frozen": True,
        "completed_training_jobs": EXPECTED_JOB_COUNT,
        "checkpoint_allowlist_rows": EXPECTED_JOB_COUNT * 2,
        "generator_checkpoint_count": EXPECTED_JOB_COUNT,
        "q4_data_opened": True,
        "predictions_frozen": True,
        "prediction_cell_count": EXPECTED_JOB_COUNT,
        "pair_metric_rows": analysis.EXPECTED_ROWS,
    }
    for key, expected in required_state.items():
        if state.get(key) != expected:
            raise ValueError(f"F4 evaluation state {key}={state.get(key)!r}, expected {expected!r}")

    declared_allowlist = _verify_declared_file(
        state.get("checkpoint_allowlist_path"),
        state.get("checkpoint_allowlist_sha256"),
        "checkpoint allowlist",
    )
    if declared_allowlist != allowlist_path.resolve():
        raise ValueError("Checkpoint allowlist path differs from the frozen evaluator path")
    checkpoint_manifest = _verify_declared_file(
        state.get("checkpoint_manifest_path"),
        state.get("checkpoint_manifest_sha256"),
        "Generator checkpoint manifest",
    )
    determinism_contract = _verify_declared_file(
        state.get("inference_determinism_contract_path"),
        state.get("inference_determinism_contract_sha256"),
        "inference determinism contract",
    )
    _read_signed_json(determinism_contract)
    input_manifest_path = _verify_declared_file(
        state.get("test_input_manifest_path"),
        state.get("test_input_manifest_sha256"),
        "frozen F4 input manifest",
    )
    input_manifest = _read_signed_json(input_manifest_path)
    if (
        input_manifest.get("fold_id") != "f4_2023q4"
        or int(input_manifest.get("pair_count", -1)) != 143
        or int(input_manifest.get("session_count", -1)) != 45
    ):
        raise ValueError("Frozen F4 input manifest count/fold drift")
    input_sources = input_manifest.get("sources")
    if not isinstance(input_sources, list) or {
        str(row.get("role")) for row in input_sources if isinstance(row, Mapping)
    } != {"panel", "film_overlay", "pure_overlay"}:
        raise ValueError("Frozen F4 input manifest must contain panel and two overlays")
    for source in input_sources:
        if not isinstance(source, Mapping):
            raise ValueError("Malformed frozen F4 input source")
        destination = _verify_declared_file(
            source.get("destination_path"), source.get("sha256"), "frozen F4 input"
        )
        if destination.stat().st_size != int(source.get("size_bytes", -1)):
            raise ValueError(f"Frozen F4 input size drift: {destination}")

    prediction_manifest_path = _verify_declared_file(
        state.get("prediction_manifest_path"),
        state.get("prediction_manifest_sha256"),
        "F4 prediction manifest",
    )
    prediction_manifest = pd.read_csv(
        prediction_manifest_path, dtype=str, keep_default_na=False
    )
    registry = _read_json(_registry_path(root))
    expected_job_ids = {str(row["job_id"]) for row in registry["jobs"]}
    if (
        len(prediction_manifest) != EXPECTED_JOB_COUNT
        or prediction_manifest["job_id"].nunique() != EXPECTED_JOB_COUNT
        or set(prediction_manifest["job_id"]) != expected_job_ids
    ):
        raise ValueError("F4 prediction manifest must contain 36 unique training jobs")
    for row in prediction_manifest.to_dict(orient="records"):
        for prefix in ("prediction", "pair_metrics", "manifest"):
            artifact = _verify_declared_file(
                row.get(f"{prefix}_path"),
                row.get(f"{prefix}_sha256"),
                f"{row['job_id']} {prefix}",
            )
            if prefix == "manifest":
                _read_signed_json(artifact)

    pair_metrics_path = _verify_declared_file(
        state.get("pair_metrics_path"),
        state.get("pair_metrics_sha256"),
        "frozen F4 pair metrics",
    )
    pair_metrics = pd.read_csv(pair_metrics_path, low_memory=False)
    summary = analysis.summarize_pair_metrics(
        pair_metrics,
        parameter_counts=_mapping(
            config["expected_parameter_counts"], "expected_parameter_counts"
        ),
    )
    if len(summary) != len(CAPACITIES):
        raise ValueError("F4 capacity summary must contain six capacity rows")

    summary_json_path = root / "analysis/f4_capacity_summary.json"
    summary_payload = _read_json(summary_json_path)
    if (
        summary_payload.get("kind") != analysis.ANALYSIS_KIND
        or int(summary_payload.get("pair_metric_rows", -1)) != analysis.EXPECTED_ROWS
        or summary_payload.get("input_pair_metrics_sha256") != _sha256_file(pair_metrics_path)
    ):
        raise ValueError("F4 capacity summary lineage/count drift")
    declared_outputs = summary_payload.get("artifacts")
    expected_outputs = {
        analysis.PAIR_METRICS_NAME,
        analysis.SUMMARY_CSV_NAME,
        analysis.TABLE_TEX_NAME,
    }
    if not isinstance(declared_outputs, Mapping) or set(declared_outputs) != expected_outputs:
        raise ValueError("F4 capacity summary artifact set drift")
    verified_outputs: dict[str, str] = {}
    for name, expected_sha256 in declared_outputs.items():
        path = _verify_declared_file(
            root / "analysis" / str(name), expected_sha256, f"analysis artifact {name}"
        )
        verified_outputs[str(name)] = _sha256_file(path)

    observed_summary = pd.read_csv(root / "analysis" / analysis.SUMMARY_CSV_NAME)
    try:
        pd.testing.assert_frame_equal(
            observed_summary,
            summary,
            check_dtype=False,
            check_exact=False,
            rtol=1e-13,
            atol=1e-15,
        )
    except AssertionError as exc:
        raise ValueError("Frozen summary CSV differs from recomputed evidence") from exc
    current = observed_summary.loc[observed_summary["capacity_id"].eq("c32")]
    if (
        len(current) != 1
        or float(current.iloc[0]["film_cnn_improvement_vs_c32_pct"]) != 0.0
        or float(current.iloc[0]["pure_cnn_improvement_vs_c32_pct"]) != 0.0
    ):
        raise ValueError("Both architecture-specific c32 improvements must equal zero")

    regression_path, regression = _write_c32_regression_control(
        root, pair_metrics, allowlist_path
    )
    if regression.get("status") != "passed":
        raise ValueError(
            "The independent c32 rerun exceeded its frozen numerical regression "
            "tolerance or changed the F4 pair/session panel"
        )

    qa_payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "f4_film_pure_capacity_3seed_terminal_qa_v1",
        "status": "passed",
        "experiment_kind": EXPERIMENT_KIND,
        "training_jobs": EXPECTED_JOB_COUNT,
        "checkpoint_rows": EXPECTED_JOB_COUNT * 2,
        "prediction_cells": EXPECTED_JOB_COUNT,
        "pair_metric_rows": analysis.EXPECTED_ROWS,
        "pair_count": analysis.PAIR_COUNT,
        "session_count": analysis.SESSION_COUNT,
        "capacity_rows": len(CAPACITIES),
        "checkpoint_manifest_sha256": _sha256_file(checkpoint_manifest),
        "pair_metrics_sha256": _sha256_file(pair_metrics_path),
        "summary_json_sha256": _sha256_file(summary_json_path),
        "c32_regression_control_status": regression["status"],
        "c32_regression_control_sha256": _sha256_file(regression_path),
        "c32_regression_all_mae_controls_within_tolerance": regression[
            "all_mae_controls_within_tolerance"
        ],
        "c32_regression_all_pair_panels_identical": regression[
            "all_pair_panels_identical"
        ],
        "c32_regression_all_checkpoints_bitwise_identical": regression[
            "all_checkpoints_bitwise_identical"
        ],
        "c32_regression_exact_determinism_claimed": regression[
            "exact_determinism_claimed"
        ],
        "verified_analysis_artifacts": verified_outputs,
        "completed_at": _utc_now(),
    }
    qa_payload["payload_sha256"] = _payload_sha256(qa_payload)
    qa_path = root / "analysis/f4_capacity_qa.json"
    _write_json(qa_path, qa_payload)
    return {
        **qa_payload,
        "qa_path": str(qa_path.resolve()),
        "qa_sha256": _sha256_file(qa_path),
    }


def run_news_first_vol_f4_film_pure_capacity_3seed(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    action: str,
    job_id: str = "",
    resume: bool = False,
    reuse: bool = False,
    worker_dry_run: bool = False,
) -> Path:
    del reuse, worker_dry_run
    config = load_config(config_path)
    root = _path(output_dir)
    if action == "benchmark":
        root = _path(config["experiment"]["benchmark_root"])
        result: Any = benchmark(config, root, resume=resume)
    elif action == "prepare":
        result = prepare(config, root, resume=resume)
    elif action == "dry-run":
        result = dry_run(config, root, resume=resume)
    elif action == "launch":
        result = launch(config, root, resume=resume)
    elif action == "worker":
        if not job_id:
            raise ValueError("worker requires --job-id")
        result = worker(config, root, job_id)
    elif action == "status":
        result = status(root)
    elif action in {"freeze-checkpoints", "evaluate-f4", "postprocess", "qa"}:
        current = status(root)
        if current.get("counts", {}).get("complete") != EXPECTED_JOB_COUNT:
            raise RuntimeError(f"{action} requires all 36 completed training jobs")
        if action == "freeze-checkpoints":
            result = freeze_checkpoints(root)
        elif action == "evaluate-f4":
            result = evaluate_f4(config, root, resume=resume)
        elif action == "postprocess":
            result = postprocess(config, root)
        else:
            result = qa(config, root)
    else:
        raise ValueError(f"Unknown action: {action}")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "benchmark",
            "prepare",
            "dry-run",
            "launch",
            "freeze-checkpoints",
            "evaluate-f4",
            "postprocess",
            "qa",
            "worker",
            "status",
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    run_news_first_vol_f4_film_pure_capacity_3seed(
        args.config,
        args.output_dir,
        action=args.action,
        job_id=args.job_id,
        resume=bool(args.resume),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
