"""Run the Q3-only Text128+c32 Mask+Coords FiLM U-Net for five seeds.

The five jobs start from fresh deterministic initialization, use the selected
LP-concat Critic, and train for at most 240 epochs.  Q4/test loaders are never
materialized.  This branch-local wrapper reuses the hardened c32 supervisor
without mutating either predecessor experiment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

import scripts._path_setup  # noqa: F401
from scripts.rq3 import news_first_vol_film_unet_c32_epoch120_seed42 as previous


core = previous.core

EXPERIMENT_KIND = "film_unet_mask_coords_text128_c32_epoch240_5seed_q3_v1"
BENCHMARK_EXPERIMENT_KIND = (
    "film_unet_mask_coords_text128_c32_epoch240_5seed_epoch1_benchmark_v1"
)
DEFAULT_CONFIG = "configs/rq3/news_first_vol_film_unet_text128_c32_epoch240_5seed.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq3_news_first_vol_film_unet_text128_c32_epoch240_5seed_exact_ttm_v1"
)
DEFAULT_BENCHMARK_DIR = (
    "outputs/benchmarks/"
    "rq3_news_first_vol_film_unet_text128_c32_epoch240_5seed_epoch1_v1"
)
SEEDS = (42, 202, 404, 382624741, 1607127774)
GPU_IDS = (0, 1)
ARM_ID = "film_unet_mask_coords_text128"
ARM_IDS = (ARM_ID,)
ARM_CONTRACTS = {ARM_ID: dict(previous.ARM_CONTRACTS[ARM_ID])}
CAPACITY_IDS = ("c32",)
PROFILE_FIELDS = previous.PROFILE_FIELDS
CAPACITY_CONTRACTS = {
    "c32": dict(zip(PROFILE_FIELDS, (32, 0, 1024, 32, 0, 128, 786), strict=True))
}
EXPECTED_JOB_COUNT = 5
LEARNING_RATE = 5e-7
MAX_EPOCHS = 240
REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_PATH = Path(__file__).resolve()
SOURCE_PATHS = (
    ORCHESTRATOR_PATH,
    previous.ORCHESTRATOR_PATH,
    previous.CORE_ORCHESTRATOR_PATH,
    *tuple(
        path
        for path in core.SOURCE_PATHS
        if path not in {previous.CORE_ORCHESTRATOR_PATH, previous.ORCHESTRATOR_PATH}
    ),
)
WORKER_MODULE = "scripts.rq3.news_first_vol_film_unet_text128_c32_epoch240_5seed"

_payload_sha256 = previous._payload_sha256
_sha256_file = previous._sha256_file
_utc_now = previous._utc_now
_write_json = previous._write_json
_write_yaml = previous._write_yaml
_atomic_write_text = previous._atomic_write_text
_require_mapping = previous._require_mapping
_read_json = previous._read_json
_registry_path = previous._registry_path
_read_status = previous._read_status
_verify_status_binding = previous._verify_status_binding
_csv_text = previous._csv_text

_ORIGINAL_TRAINING_PAYLOAD = previous._CORE_TRAINING_PAYLOAD


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _validate_resolved(resolved: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "lr_warmup_epochs": 0,
        "early_stopping_min_epochs": 30,
        "early_stopping_patience": 20,
        "validation_mc_samples": 16,
    }
    for key, expected in exact.items():
        actual = resolved.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected):
                raise ValueError(f"{key} must be {expected}")
        elif actual != expected:
            raise ValueError(f"{key} must be {expected!r}, got {actual!r}")

    if tuple(map(int, resolved.get("seeds", ()))) != SEEDS:
        raise ValueError(f"seeds must be {SEEDS}")
    if tuple(map(str, resolved.get("capacity_profiles", ()))) != CAPACITY_IDS:
        raise ValueError(f"capacity_profiles must be {CAPACITY_IDS}")

    arms = _require_mapping(resolved.get("arms"), "arms")
    if tuple(arms) != ARM_IDS:
        raise ValueError(f"arm order must be {ARM_IDS}")
    arm = _require_mapping(arms[ARM_ID], f"arms.{ARM_ID}")
    if not str(arm.get("label", "")).strip():
        raise ValueError(f"arms.{ARM_ID}.label must be non-empty")
    for key, expected in ARM_CONTRACTS[ARM_ID].items():
        actual = int(arm[key]) if isinstance(expected, int) else str(arm[key])
        if actual != expected:
            raise ValueError(f"arms.{ARM_ID}.{key} must be {expected!r}")

    profiles = _require_mapping(resolved.get("profiles"), "profiles")
    if tuple(profiles) != CAPACITY_IDS:
        raise ValueError("Only the c32 profile is permitted")
    profile = _require_mapping(profiles["c32"], "profiles.c32")
    if set(profile) != set(PROFILE_FIELDS):
        raise ValueError("c32 profile fields drifted")
    for key, expected in CAPACITY_CONTRACTS["c32"].items():
        if int(profile[key]) != expected:
            raise ValueError(f"profiles.c32.{key} must be {expected}")

    comparison = _require_mapping(resolved.get("comparison"), "comparison")
    for path_key, sha_key in (
        ("epoch60_job_summary", "epoch60_job_summary_sha256"),
        ("epoch120_ranking", "epoch120_ranking_sha256"),
    ):
        path = Path(str(comparison.get(path_key, "")))
        if not path.is_file():
            raise FileNotFoundError(f"Frozen comparison is missing: {path}")
        if _sha256_file(path) != str(comparison.get(sha_key, "")):
            raise ValueError(f"Frozen comparison SHA drifted: {path}")

    runtime = _require_mapping(resolved.get("runtime"), "runtime")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != GPU_IDS:
        raise ValueError("runtime.gpu_ids must be [0, 1]")
    for key in ("workers_per_gpu", "benchmark_workers_per_gpu"):
        if int(runtime.get(key, -1)) != 3:
            raise ValueError(f"runtime.{key} must be 3")
    if int(runtime.get("cpu_threads_per_job", -1)) != 1:
        raise ValueError("runtime.cpu_threads_per_job must be 1")
    for key in ("poll_interval_seconds", "resource_sample_interval_seconds"):
        if float(runtime.get(key, 0.0)) <= 0.0:
            raise ValueError(f"runtime.{key} must be positive")
    if not math.isclose(float(runtime.get("max_peak_gpu_memory_gib", 0.0)), 20.0):
        raise ValueError("runtime.max_peak_gpu_memory_gib must be 20")
    if not math.isclose(float(runtime.get("max_host_ram_fraction", 0.0)), 0.85):
        raise ValueError("runtime.max_host_ram_fraction must be 0.85")


def resolve_config(config_path: str | Path) -> dict[str, Any]:
    source = _resolve_repo_path(config_path)
    root = _require_mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "root")
    resolved = _require_mapping(
        root.get("film_unet_text128_c32_epoch240_5seed"),
        "film_unet_text128_c32_epoch240_5seed",
    )
    resolved["source_config_path"] = str(source)
    resolved["base_training_config"] = str(
        _resolve_repo_path(str(resolved["base_training_config"]))
    )
    runtime = _require_mapping(resolved.get("runtime"), "runtime")
    runtime["python_executable"] = str(
        _resolve_repo_path(str(runtime["python_executable"]))
    )
    resolved["runtime"] = runtime
    comparison = _require_mapping(resolved.get("comparison"), "comparison")
    comparison["epoch60_job_summary"] = str(
        _resolve_repo_path(str(comparison["epoch60_job_summary"]))
    )
    comparison["epoch120_ranking"] = str(
        _resolve_repo_path(str(comparison["epoch120_ranking"]))
    )
    resolved["comparison"] = comparison
    _validate_resolved(resolved)
    base_config = _require_mapping(
        yaml.safe_load(
            Path(str(resolved["base_training_config"])).read_text(encoding="utf-8")
        ),
        "base training config",
    )
    core._validate_base_training_config(base_config)
    return resolved


def _source_hashes_impl(resolved: Mapping[str, Any]) -> dict[str, str]:
    paths = (*SOURCE_PATHS, Path(str(resolved["source_config_path"])))
    hashes = {
        f"source_sha256::{path.relative_to(REPO_ROOT)}": _sha256_file(path)
        for path in paths
    }
    base_config = Path(str(resolved["base_training_config"]))
    hashes[f"source_sha256::{base_config.relative_to(REPO_ROOT)}"] = _sha256_file(
        base_config
    )
    comparison = _require_mapping(resolved["comparison"], "comparison")
    for key in ("epoch60_job_summary", "epoch120_ranking"):
        path = Path(str(comparison[key]))
        hashes[f"input_sha256::{path.relative_to(REPO_ROOT)}"] = _sha256_file(path)
    return hashes


def _training_payload_impl(
    resolved: Mapping[str, Any], output_root: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _ORIGINAL_TRAINING_PAYLOAD(resolved, output_root, spec)
    payload["news_first_lr_profile"] = (
        "film_unet_text128_c32_epoch240_5seed_lr_5e_07_no_warmup"
    )
    return payload


def _initial_state_fairness_impl(
    specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(specs) != EXPECTED_JOB_COUNT:
        raise AssertionError("Initial-state manifest requires exactly five jobs")
    by_seed = {int(spec["seed"]): spec for spec in specs}
    if tuple(by_seed) != SEEDS:
        raise AssertionError("Initial-state seed order drifted")
    rows = [
        {
            "seed": seed,
            "generator_state_sha256": str(
                by_seed[seed]["initial_generator_state_sha256"]
            ),
            "critic_state_sha256": str(by_seed[seed]["initial_critic_state_sha256"]),
        }
        for seed in SEEDS
    ]
    if len({row["generator_state_sha256"] for row in rows}) != len(SEEDS):
        raise AssertionError("Generator initial states must differ across seeds")
    if len({row["critic_state_sha256"] for row in rows}) != len(SEEDS):
        raise AssertionError("Critic initial states must differ across seeds")
    payload = {
        "schema_version": 1,
        "initialization_order": "isolated torch.manual_seed(seed), Generator then Critic",
        "scope_note": "One fixed executable graph; each seed owns a distinct fresh state.",
        "seeds": rows,
    }
    return {**payload, "fairness_contract_sha256": _payload_sha256(payload)}


def _experiment_specs_impl(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    architecture = core._architecture_contract(resolved, ARM_ID, "c32")
    model = core._model_contract(resolved, ARM_ID, "c32")
    specs: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        initial_state = core._initial_state_hashes(resolved, ARM_ID, "c32", seed)
        spec = {
            "job_id": f"{ARM_ID}_c32_seed_{seed:03d}",
            "experiment_kind": EXPERIMENT_KIND,
            "arm_id": ARM_ID,
            "arm_label": architecture["arm_label"],
            "capacity_id": "c32",
            "seed": seed,
            "gpu_id": GPU_IDS[seed_index % len(GPU_IDS)],
            "learning_rate": LEARNING_RATE,
            "training_state_inherited": False,
            "num_epochs": int(resolved["max_epochs"]),
            "lr_warmup_epochs": int(resolved["lr_warmup_epochs"]),
            "early_stopping_min_epochs": int(resolved["early_stopping_min_epochs"]),
            "early_stopping_patience": int(resolved["early_stopping_patience"]),
            "generator_conditioning_mode": architecture["generator_conditioning_mode"],
            "generator_conditioning_fingerprint": architecture[
                "generator_conditioning_fingerprint"
            ],
            "critic_conditioning_mode": architecture["critic_conditioning_mode"],
            "critic_conditioning_fingerprint": architecture[
                "critic_conditioning_fingerprint"
            ],
            "gen_text_hidden_dim": architecture["gen_text_hidden_dim"],
            "gen_text_out_dim": architecture["gen_text_out_dim"],
            **{field: int(architecture[field]) for field in PROFILE_FIELDS},
            "generator_parameters": architecture["generator_parameters"],
            "critic_parameters": architecture["critic_parameters"],
            "wgan_parameters": architecture["wgan_parameters"],
            "architecture_profile_sha256": architecture["architecture_profile_sha256"],
            "model_contract_sha256": model["model_contract_sha256"],
            **initial_state,
        }
        spec["job_spec_sha256"] = _payload_sha256(spec)
        specs.append(spec)
    if (
        len(specs) != EXPECTED_JOB_COUNT
        or len({str(spec["job_id"]) for spec in specs}) != EXPECTED_JOB_COUNT
    ):
        raise AssertionError("Expected five unique formal jobs")
    gpu_counts = {
        gpu: sum(int(spec["gpu_id"]) == gpu for spec in specs) for gpu in GPU_IDS
    }
    if gpu_counts != {0: 3, 1: 2}:
        raise AssertionError(f"GPU assignment drifted: {gpu_counts}")
    _initial_state_fairness_impl(specs)
    return specs


@contextmanager
def _configured_core() -> Iterator[None]:
    replacements: dict[str, Any] = {
        "EXPERIMENT_KIND": EXPERIMENT_KIND,
        "BENCHMARK_EXPERIMENT_KIND": BENCHMARK_EXPERIMENT_KIND,
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "DEFAULT_BENCHMARK_DIR": DEFAULT_BENCHMARK_DIR,
        "SEEDS": SEEDS,
        "GPU_IDS": GPU_IDS,
        "LEARNING_RATE": LEARNING_RATE,
        "ARM_CONTRACTS": ARM_CONTRACTS,
        "ARM_IDS": ARM_IDS,
        "CAPACITY_IDS": CAPACITY_IDS,
        "CAPACITY_CONTRACTS": CAPACITY_CONTRACTS,
        "EXPECTED_JOB_COUNT": EXPECTED_JOB_COUNT,
        "ORCHESTRATOR_PATH": ORCHESTRATOR_PATH,
        "SOURCE_PATHS": SOURCE_PATHS,
        "experiment_specs": _experiment_specs_impl,
        "_training_payload": _training_payload_impl,
        "_source_hashes": _source_hashes_impl,
        "_initial_state_fairness": _initial_state_fairness_impl,
    }
    original = {name: getattr(core, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(core, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(core, name, value)


def _prepare_benchmark_impl(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with _configured_core():
        registry = core._prepare_benchmark(resolved, output_root, resume=resume)
    expected_label = "film_unet_text128_c32_epoch240_5seed_epoch1_benchmark"
    changed = False
    for job in registry["jobs"]:
        config_path = Path(str(job["config_path"]))
        payload = _require_mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            "benchmark job config",
        )
        if str(payload.get("news_first_lr_profile", "")) == expected_label:
            continue
        if resume:
            raise ValueError("Benchmark LR-profile label drift detected")
        payload["news_first_lr_profile"] = expected_label
        _write_yaml(config_path, payload)
        job["config_sha256"] = _sha256_file(config_path)
        changed = True
    if changed:
        _write_json(_registry_path(output_root), registry)
    previous._verify_registry_bindings(registry, output_root, expected_specs=None)
    return registry


def _prepare_benchmark(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with _configured_previous():
        return _prepare_benchmark_impl(resolved, output_root, resume=resume)


@contextmanager
def _configured_previous() -> Iterator[None]:
    replacements: dict[str, Any] = {
        "EXPERIMENT_KIND": EXPERIMENT_KIND,
        "BENCHMARK_EXPERIMENT_KIND": BENCHMARK_EXPERIMENT_KIND,
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "DEFAULT_BENCHMARK_DIR": DEFAULT_BENCHMARK_DIR,
        "SEEDS": SEEDS,
        "GPU_IDS": GPU_IDS,
        "CAPACITY_IDS": CAPACITY_IDS,
        "EXPECTED_JOB_COUNT": EXPECTED_JOB_COUNT,
        "LEARNING_RATE": LEARNING_RATE,
        "ARM_IDS": ARM_IDS,
        "ARM_CONTRACTS": ARM_CONTRACTS,
        "CAPACITY_CONTRACTS": CAPACITY_CONTRACTS,
        "ORCHESTRATOR_PATH": ORCHESTRATOR_PATH,
        "SOURCE_PATHS": SOURCE_PATHS,
        "WORKER_MODULE": WORKER_MODULE,
        "_configured_core": _configured_core,
        "_experiment_specs_impl": _experiment_specs_impl,
        "_training_payload_impl": _training_payload_impl,
        "_source_hashes_impl": _source_hashes_impl,
        "_prepare_benchmark": _prepare_benchmark_impl,
        "postprocess": _postprocess_impl,
    }
    original = {name: getattr(previous, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(previous, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(previous, name, value)


def experiment_specs(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    with _configured_core():
        return _experiment_specs_impl(resolved)


def _training_payload(
    resolved: Mapping[str, Any], output_root: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    with _configured_core():
        return _training_payload_impl(resolved, output_root, spec)


def _initial_state_fairness(
    specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _initial_state_fairness_impl(specs)


def prepare(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with _configured_previous():
        return previous.prepare(resolved, output_root, resume=resume)


def dry_run(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with _configured_previous():
        return previous.dry_run(resolved, output_root, resume=resume)


def run_worker(
    resolved: Mapping[str, Any], output_root: Path, *, job_id: str
) -> dict[str, Any]:
    with _configured_previous():
        return previous.run_worker(resolved, output_root, job_id=job_id)


def benchmark(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with _configured_previous():
        return previous.benchmark(resolved, output_root, resume=resume)


def _historical_rows(
    resolved: Mapping[str, Any],
) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    comparison = _require_mapping(resolved["comparison"], "comparison")
    epoch60_path = Path(str(comparison["epoch60_job_summary"]))
    epoch120_path = Path(str(comparison["epoch120_ranking"]))
    if _sha256_file(epoch60_path) != str(comparison["epoch60_job_summary_sha256"]):
        raise ValueError("Frozen epoch-60 comparison SHA drifted")
    if _sha256_file(epoch120_path) != str(comparison["epoch120_ranking_sha256"]):
        raise ValueError("Frozen epoch-120 comparison SHA drifted")
    with epoch60_path.open(encoding="utf-8", newline="") as handle:
        epoch60 = {
            int(row["seed"]): row
            for row in csv.DictReader(handle)
            if row["arm_id"] == ARM_ID and row["capacity_id"] == "c32"
        }
    with epoch120_path.open(encoding="utf-8", newline="") as handle:
        epoch120 = {
            int(row["seed"]): row
            for row in csv.DictReader(handle)
            if row["arm_id"] == ARM_ID and row["capacity_id"] == "c32"
        }
    if set(epoch60) != {42, 202, 404} or set(epoch120) != {42}:
        raise ValueError("Frozen historical comparison coverage drifted")
    return epoch60, epoch120


def _postprocess_impl(output_root: Path) -> dict[str, Any]:
    registry = _read_json(_registry_path(output_root))
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Unexpected experiment kind")
    resolved = resolve_config(str(registry["source_config_path"]))
    prepare(resolved, output_root, resume=True)
    epoch60, epoch120 = _historical_rows(resolved)
    rows: list[dict[str, Any]] = []
    for job in registry["jobs"]:
        status_payload = _read_status(output_root, str(job["job_id"]))
        _verify_status_binding(job, status_payload, require_complete=True)
        mae = float(status_payload["best_mae"])
        persistence = float(status_payload["persistence_mae"])
        best_epoch = int(status_payload["best_epoch"])
        completed_epochs = int(status_payload["completed_epochs"])
        if not (
            math.isfinite(mae)
            and mae > 0.0
            and math.isfinite(persistence)
            and persistence > 0.0
        ):
            raise ValueError(f"Invalid Q3 metric for {job['job_id']}")
        if not (1 <= best_epoch <= completed_epochs <= MAX_EPOCHS):
            raise ValueError(f"Invalid epoch bounds for {job['job_id']}")
        seed = int(job["seed"])
        old60 = epoch60.get(seed)
        old120 = epoch120.get(seed)
        old60_mae = float(old60["best_q3_mae"]) if old60 else None
        old120_mae = float(old120["best_q3_mae_120_run"]) if old120 else None
        log_ratio = math.log(mae / persistence)
        rows.append(
            {
                "job_id": job["job_id"],
                "arm_id": job["arm_id"],
                "arm_label": job["arm_label"],
                "capacity_id": job["capacity_id"],
                "seed": seed,
                "gpu_id": job["gpu_id"],
                "generator_conditioning_mode": job["generator_conditioning_mode"],
                "critic_conditioning_mode": job["critic_conditioning_mode"],
                "gen_text_hidden_dim": job["gen_text_hidden_dim"],
                "gen_text_out_dim": job["gen_text_out_dim"],
                "generator_parameters": job["generator_parameters"],
                "critic_parameters": job["critic_parameters"],
                "wgan_parameters": job["wgan_parameters"],
                "training_state_inherited": False,
                "best_epoch": best_epoch,
                "completed_epochs": completed_epochs,
                "completed_max_epoch": completed_epochs == MAX_EPOCHS,
                "best_at_max_epoch": best_epoch == MAX_EPOCHS,
                "best_q3_mae": mae,
                "persistence_q3_mae": persistence,
                "log_mae_ratio_vs_persistence": log_ratio,
                "improvement_vs_persistence_pct": (1.0 - math.exp(log_ratio)) * 100.0,
                "historical_epoch60_best_epoch": (
                    int(old60["best_epoch"]) if old60 else None
                ),
                "historical_epoch60_best_q3_mae": old60_mae,
                "mae_delta_vs_epoch60": (
                    mae - old60_mae if old60_mae is not None else None
                ),
                "historical_epoch120_best_epoch": (
                    int(old120["best_epoch_120_run"]) if old120 else None
                ),
                "historical_epoch120_best_q3_mae": old120_mae,
                "mae_delta_vs_epoch120": (
                    mae - old120_mae if old120_mae is not None else None
                ),
                "run_dir": status_payload["run_dir"],
            }
        )
    if len(rows) != EXPECTED_JOB_COUNT or {int(row["seed"]) for row in rows} != set(
        SEEDS
    ):
        raise ValueError("Postprocess job coverage drifted")
    if len({float(row["persistence_q3_mae"]) for row in rows}) != 1:
        raise ValueError("Persistence baseline drifted across seeds")
    rows.sort(key=lambda row: SEEDS.index(int(row["seed"])))
    log_ratios = [float(row["log_mae_ratio_vs_persistence"]) for row in rows]
    maes = [float(row["best_q3_mae"]) for row in rows]
    mean_log_ratio = sum(log_ratios) / len(log_ratios)
    log_ratio_variance = sum((value - mean_log_ratio) ** 2 for value in log_ratios) / (
        len(log_ratios) - 1
    )
    mean_mae = sum(maes) / len(maes)
    mae_variance = sum((value - mean_mae) ** 2 for value in maes) / (len(maes) - 1)
    summary = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "selection_scope": "Q3 development only; Q4 not materialized",
        "inference_scope": "five-seed descriptive epoch extension",
        "fresh_initialization": True,
        "training_state_inherited": False,
        "completed_jobs": len(rows),
        "seed_count": len(rows),
        "mean_best_q3_mae": mean_mae,
        "sample_std_best_q3_mae": math.sqrt(mae_variance),
        "mean_log_mae_ratio_vs_persistence": mean_log_ratio,
        "seed_standard_error_log_ratio": math.sqrt(
            log_ratio_variance / len(log_ratios)
        ),
        "geomean_mae_ratio_vs_persistence": math.exp(mean_log_ratio),
        "geomean_improvement_vs_persistence_pct": (1.0 - math.exp(mean_log_ratio))
        * 100.0,
        "seeds_beating_persistence": sum(
            float(row["best_q3_mae"]) <= float(row["persistence_q3_mae"])
            for row in rows
        ),
        "completed_max_epoch_count": sum(
            bool(row["completed_max_epoch"]) for row in rows
        ),
        "best_at_max_epoch_count": sum(bool(row["best_at_max_epoch"]) for row in rows),
        "rows": rows,
        "created_at": _utc_now(),
    }
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(analysis_dir / "job_summary.csv", _csv_text(rows))
    _write_json(analysis_dir / "five_seed_summary.json", summary)
    return summary


def postprocess(output_root: Path) -> dict[str, Any]:
    with _configured_previous():
        return _postprocess_impl(output_root)


def launch(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with _configured_previous():
        return previous.launch(resolved, output_root, resume=resume)


def status(output_root: Path) -> dict[str, Any]:
    with _configured_previous():
        return previous.status(output_root)


def run_pipeline(
    resolved: Mapping[str, Any],
    output_root: Path,
    *,
    resume: bool,
    benchmark_root: Path | None = None,
) -> dict[str, Any]:
    with _configured_previous():
        return previous.run_pipeline(
            resolved,
            output_root,
            resume=resume,
            benchmark_root=benchmark_root,
        )


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
    elif args.action == "prepare":
        registry = prepare(resolved, output_root, resume=bool(args.resume))
        result = {
            "output_root": str(output_root),
            "job_count": len(registry["jobs"]),
            "gpu_counts": {
                str(gpu): sum(int(job["gpu_id"]) == gpu for job in registry["jobs"])
                for gpu in GPU_IDS
            },
        }
    elif args.action == "dry-run":
        result = dry_run(resolved, output_root, resume=bool(args.resume))
    elif args.action == "worker":
        result = run_worker(resolved, output_root, job_id=str(args.job_id))
    elif args.action == "launch":
        result = launch(resolved, output_root, resume=bool(args.resume))
    else:
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
