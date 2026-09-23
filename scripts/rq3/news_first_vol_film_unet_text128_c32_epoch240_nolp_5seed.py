"""Run the exact five-seed NoLP-Critic counterpart to the frozen LP run.

Only the Critic conditioning mode changes.  The selected Text128+c32 FiLM
U-Net, seeds, optimization budget, validation panel, and initialization are
bound to the completed LP-Critic experiment.  Q4/test loaders are never
materialized by this runner.
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
from scripts.rq3 import (
    news_first_vol_film_unet_text128_c32_epoch240_5seed as lp,
)


EXPERIMENT_KIND = "film_unet_mask_coords_text128_c32_epoch240_nolp_5seed_q3_v1"
BENCHMARK_EXPERIMENT_KIND = (
    "film_unet_mask_coords_text128_c32_epoch240_nolp_5seed_epoch1_benchmark_v1"
)
DEFAULT_CONFIG = (
    "configs/rq3/news_first_vol_film_unet_text128_c32_epoch240_nolp_5seed.yaml"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq3_news_first_vol_film_unet_text128_c32_epoch240_nolp_5seed_exact_ttm_v1"
)
DEFAULT_BENCHMARK_DIR = (
    "outputs/benchmarks/"
    "rq3_news_first_vol_film_unet_text128_c32_epoch240_nolp_5seed_epoch1_v1"
)
SEEDS = lp.SEEDS
GPU_IDS = lp.GPU_IDS
ARM_ID = "film_unet_mask_coords_text128_nolp"
LP_REFERENCE_ARM_ID = "film_unet_mask_coords_text128"
LP_REFERENCE_EXPERIMENT_KIND = lp.EXPERIMENT_KIND
ARM_IDS = (ARM_ID,)
ARM_CONTRACTS = {
    ARM_ID: {
        "generator_conditioning_mode": "film_unet_mask_coords_v1",
        "critic_conditioning_mode": "lp_disabled_same_shape_v1",
        "gen_text_hidden_dim": 256,
        "gen_text_out_dim": 128,
    }
}
CAPACITY_IDS = lp.CAPACITY_IDS
PROFILE_FIELDS = lp.PROFILE_FIELDS
CAPACITY_CONTRACTS = lp.CAPACITY_CONTRACTS
EXPECTED_JOB_COUNT = 5
LEARNING_RATE = lp.LEARNING_RATE
MAX_EPOCHS = lp.MAX_EPOCHS
REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_PATH = Path(__file__).resolve()
SOURCE_PATHS = (ORCHESTRATOR_PATH, *lp.SOURCE_PATHS)
WORKER_MODULE = "scripts.rq3.news_first_vol_film_unet_text128_c32_epoch240_nolp_5seed"

_payload_sha256 = lp._payload_sha256
_sha256_file = lp._sha256_file
_utc_now = lp._utc_now
_write_json = lp._write_json
_write_yaml = lp._write_yaml
_atomic_write_text = lp._atomic_write_text
_require_mapping = lp._require_mapping
_read_json = lp._read_json
_registry_path = lp._registry_path
_read_status = lp._read_status
_csv_text = lp._csv_text
_ORIGINAL_VERIFY_STATUS_BINDING = lp.previous._verify_status_binding


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _reference_paths(resolved: Mapping[str, Any]) -> dict[str, Path]:
    comparison = _require_mapping(resolved["comparison"], "comparison")
    return {
        key: Path(str(comparison[key])).resolve()
        for key in (
            "lp_registry",
            "lp_initial_state_fairness",
            "lp_job_summary",
            "lp_five_seed_summary",
        )
    }


def _verify_frozen_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen comparison input is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Frozen comparison SHA drifted for {path}: {actual} != {expected_sha256}"
        )


def _load_lp_reference(resolved: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    comparison = _require_mapping(resolved["comparison"], "comparison")
    paths = _reference_paths(resolved)
    for key, path in paths.items():
        _verify_frozen_file(path, str(comparison[f"{key}_sha256"]))

    registry = _read_json(paths["lp_registry"])
    if registry.get("experiment_kind") != LP_REFERENCE_EXPERIMENT_KIND:
        raise ValueError("Frozen LP registry has the wrong experiment kind")
    jobs = [_require_mapping(job, "LP registry job") for job in registry["jobs"]]
    by_seed = {int(job["seed"]): job for job in jobs}
    if tuple(by_seed) != SEEDS or len(jobs) != EXPECTED_JOB_COUNT:
        raise ValueError("Frozen LP registry seed coverage drifted")
    if {str(job["arm_id"]) for job in jobs} != {LP_REFERENCE_ARM_ID}:
        raise ValueError("Frozen LP registry arm drifted")

    fairness = _read_json(paths["lp_initial_state_fairness"])
    fairness_by_seed = {
        int(row["seed"]): _require_mapping(row, "LP fairness row")
        for row in fairness["seeds"]
    }
    if set(fairness_by_seed) != set(SEEDS):
        raise ValueError("Frozen LP fairness seed coverage drifted")

    summary = _read_json(paths["lp_five_seed_summary"])
    summary_by_seed = {
        int(row["seed"]): _require_mapping(row, "LP summary row")
        for row in summary["rows"]
    }
    if set(summary_by_seed) != set(SEEDS):
        raise ValueError("Frozen LP summary seed coverage drifted")

    with paths["lp_job_summary"].open(encoding="utf-8", newline="") as handle:
        csv_by_seed = {int(row["seed"]): row for row in csv.DictReader(handle)}
    if set(csv_by_seed) != set(SEEDS):
        raise ValueError("Frozen LP job-summary seed coverage drifted")

    status_sha_by_seed = _require_mapping(
        comparison["lp_status_sha256_by_seed"],
        "comparison.lp_status_sha256_by_seed",
    )
    if {int(seed) for seed in status_sha_by_seed} != set(SEEDS):
        raise ValueError("Frozen LP status-SHA seed coverage drifted")

    result: dict[int, dict[str, Any]] = {}
    status_dir = paths["lp_registry"].parent / "jobs"
    for seed in SEEDS:
        job = by_seed[seed]
        status_path = status_dir / f"{job['job_id']}.status.json"
        _verify_frozen_file(status_path, str(status_sha_by_seed[str(seed)]))
        status = _read_json(status_path)
        _ORIGINAL_VERIFY_STATUS_BINDING(job, status, require_complete=True)
        fairness_row = fairness_by_seed[seed]
        for component in ("generator", "critic"):
            expected = str(job[f"initial_{component}_state_sha256"])
            actual = str(status[f"{component}_initial_state_sha256"])
            frozen = str(fairness_row[f"{component}_state_sha256"])
            if len({expected, actual, frozen}) != 1:
                raise ValueError(
                    f"Frozen LP {component} initial state drifted for seed {seed}"
                )
        csv_row = csv_by_seed[seed]
        summary_row = summary_by_seed[seed]
        if not math.isclose(
            float(csv_row["best_q3_mae"]),
            float(summary_row["best_q3_mae"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Frozen LP metric drifted for seed {seed}")
        result[seed] = {
            "job": job,
            "status": status,
            "summary_row": summary_row,
            "status_path": status_path,
        }
    return result


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

    runtime = _require_mapping(resolved.get("runtime"), "runtime")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != GPU_IDS:
        raise ValueError("runtime.gpu_ids must be [0, 1]")
    for key in ("workers_per_gpu", "benchmark_workers_per_gpu"):
        if int(runtime.get(key, -1)) != 3:
            raise ValueError(f"runtime.{key} must be 3")
    if int(runtime.get("cpu_threads_per_job", -1)) != 1:
        raise ValueError("runtime.cpu_threads_per_job must be 1")
    if not math.isclose(float(runtime.get("max_peak_gpu_memory_gib", 0.0)), 20.0):
        raise ValueError("runtime.max_peak_gpu_memory_gib must be 20")
    if not math.isclose(float(runtime.get("max_host_ram_fraction", 0.0)), 0.85):
        raise ValueError("runtime.max_host_ram_fraction must be 0.85")

    data_contract = _require_mapping(resolved.get("data_contract"), "data_contract")
    workbook = Path(str(data_contract["training_workbook"]))
    _verify_frozen_file(workbook, str(data_contract["training_workbook_sha256"]))
    base = _require_mapping(
        yaml.safe_load(
            Path(str(resolved["base_training_config"])).read_text(encoding="utf-8")
        ),
        "base training config",
    )
    configured_data = _resolve_repo_path(str(base["data_path"]))
    if configured_data != workbook:
        raise ValueError("Base config data_path drifted from the frozen workbook")
    lp.core._validate_base_training_config(base)
    _load_lp_reference(resolved)


def resolve_config(config_path: str | Path) -> dict[str, Any]:
    source = _resolve_repo_path(config_path)
    root = _require_mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "root")
    resolved = _require_mapping(
        root.get("film_unet_text128_c32_epoch240_nolp_5seed"),
        "film_unet_text128_c32_epoch240_nolp_5seed",
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
    data_contract = _require_mapping(resolved.get("data_contract"), "data_contract")
    data_contract["training_workbook"] = str(
        _resolve_repo_path(str(data_contract["training_workbook"]))
    )
    resolved["data_contract"] = data_contract
    comparison = _require_mapping(resolved.get("comparison"), "comparison")
    for key in (
        "lp_registry",
        "lp_initial_state_fairness",
        "lp_job_summary",
        "lp_five_seed_summary",
    ):
        comparison[key] = str(_resolve_repo_path(str(comparison[key])))
    resolved["comparison"] = comparison
    _validate_resolved(resolved)
    return resolved


def _source_hashes_impl(resolved: Mapping[str, Any]) -> dict[str, str]:
    paths = (*SOURCE_PATHS, Path(str(resolved["source_config_path"])))
    hashes = {
        f"source_sha256::{path.relative_to(REPO_ROOT)}": _sha256_file(path)
        for path in paths
    }
    base = Path(str(resolved["base_training_config"]))
    hashes[f"source_sha256::{base.relative_to(REPO_ROOT)}"] = _sha256_file(base)
    data_contract = _require_mapping(resolved["data_contract"], "data_contract")
    workbook = Path(str(data_contract["training_workbook"]))
    _verify_frozen_file(workbook, str(data_contract["training_workbook_sha256"]))
    hashes[f"input_sha256::{workbook.relative_to(REPO_ROOT)}"] = _sha256_file(workbook)
    comparison = _require_mapping(resolved["comparison"], "comparison")
    for key, path in _reference_paths(resolved).items():
        _verify_frozen_file(path, str(comparison[f"{key}_sha256"]))
        hashes[f"input_sha256::{path.relative_to(REPO_ROOT)}"] = _sha256_file(path)
    for seed, reference in _load_lp_reference(resolved).items():
        status_path = Path(str(reference["status_path"]))
        hashes[f"input_sha256::{status_path.relative_to(REPO_ROOT)}"] = _sha256_file(
            status_path
        )
    return hashes


def _training_payload_impl(
    resolved: Mapping[str, Any], output_root: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    payload = lp._ORIGINAL_TRAINING_PAYLOAD(resolved, output_root, spec)
    payload["news_first_lr_profile"] = (
        "film_unet_text128_c32_epoch240_nolp_5seed_lr_5e_07_no_warmup"
    )
    return payload


def _experiment_specs_impl(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    reference = _load_lp_reference(resolved)
    architecture = lp.core._architecture_contract(resolved, ARM_ID, "c32")
    model = lp.core._model_contract(resolved, ARM_ID, "c32")
    specs: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        initial_state = lp.core._initial_state_hashes(resolved, ARM_ID, "c32", seed)
        lp_job = reference[seed]["job"]
        lp_status = reference[seed]["status"]
        for component in ("generator", "critic"):
            predicted = str(initial_state[f"initial_{component}_state_sha256"])
            old_predicted = str(lp_job[f"initial_{component}_state_sha256"])
            old_actual = str(lp_status[f"{component}_initial_state_sha256"])
            if len({predicted, old_predicted, old_actual}) != 1:
                raise ValueError(
                    f"NoLP/LP {component} initial state mismatch for seed {seed}"
                )
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
            "paired_lp_job_id": str(lp_job["job_id"]),
            "paired_lp_job_spec_sha256": str(lp_job["job_spec_sha256"]),
            "paired_lp_generator_initial_state_sha256": str(
                lp_status["generator_initial_state_sha256"]
            ),
            "paired_lp_critic_initial_state_sha256": str(
                lp_status["critic_initial_state_sha256"]
            ),
            **initial_state,
        }
        spec["job_spec_sha256"] = _payload_sha256(spec)
        specs.append(spec)
    if (
        len(specs) != EXPECTED_JOB_COUNT
        or len({str(spec["job_id"]) for spec in specs}) != EXPECTED_JOB_COUNT
    ):
        raise AssertionError("Expected five unique NoLP formal jobs")
    gpu_counts = {
        gpu: sum(int(spec["gpu_id"]) == gpu for spec in specs) for gpu in GPU_IDS
    }
    if gpu_counts != {0: 3, 1: 2}:
        raise AssertionError(f"GPU assignment drifted: {gpu_counts}")
    _initial_state_fairness_impl(specs)
    return specs


def _initial_state_fairness_impl(
    specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(specs) != EXPECTED_JOB_COUNT:
        raise AssertionError("Initial-state manifest requires exactly five jobs")
    by_seed = {int(spec["seed"]): spec for spec in specs}
    if tuple(by_seed) != SEEDS:
        raise AssertionError("Initial-state seed order drifted")
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        spec = by_seed[seed]
        row = {
            "seed": seed,
            "generator_state_sha256": str(spec["initial_generator_state_sha256"]),
            "critic_state_sha256": str(spec["initial_critic_state_sha256"]),
            "paired_lp_job_id": str(spec["paired_lp_job_id"]),
            "paired_lp_generator_state_sha256": str(
                spec["paired_lp_generator_initial_state_sha256"]
            ),
            "paired_lp_critic_state_sha256": str(
                spec["paired_lp_critic_initial_state_sha256"]
            ),
        }
        if row["generator_state_sha256"] != row["paired_lp_generator_state_sha256"]:
            raise AssertionError(f"Generator initial state differs for seed {seed}")
        if row["critic_state_sha256"] != row["paired_lp_critic_state_sha256"]:
            raise AssertionError(f"Critic initial state differs for seed {seed}")
        rows.append(row)
    if len({row["generator_state_sha256"] for row in rows}) != len(SEEDS):
        raise AssertionError("Generator initial states must differ across seeds")
    if len({row["critic_state_sha256"] for row in rows}) != len(SEEDS):
        raise AssertionError("Critic initial states must differ across seeds")
    payload = {
        "schema_version": 1,
        "initialization_order": "isolated torch.manual_seed(seed), Generator then Critic",
        "scope_note": "NoLP and frozen LP arms have identical epoch-0 state by seed.",
        "seeds": rows,
    }
    return {**payload, "fairness_contract_sha256": _payload_sha256(payload)}


def _verify_status_binding_impl(
    job: Mapping[str, Any],
    status_payload: Mapping[str, Any],
    *,
    require_complete: bool,
) -> None:
    _ORIGINAL_VERIFY_STATUS_BINDING(
        job, status_payload, require_complete=require_complete
    )
    if str(status_payload.get("state", "")) != "complete":
        return
    for component in ("generator", "critic"):
        actual = str(status_payload.get(f"{component}_initial_state_sha256", ""))
        expected = str(job[f"initial_{component}_state_sha256"])
        paired = str(job[f"paired_lp_{component}_initial_state_sha256"])
        if not actual or len({actual, expected, paired}) != 1:
            raise ValueError(
                f"Actual NoLP {component} epoch-0 state is not paired for "
                f"{job['job_id']}"
            )


def _prepare_benchmark_impl(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with lp._configured_core():
        registry = lp.core._prepare_benchmark(resolved, output_root, resume=resume)
    expected_label = "film_unet_text128_c32_epoch240_nolp_5seed_epoch1_benchmark"
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
    lp.previous._verify_registry_bindings(registry, output_root, expected_specs=None)
    return registry


def _sample_std(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _postprocess_impl(output_root: Path) -> dict[str, Any]:
    registry = _read_json(_registry_path(output_root))
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Unexpected experiment kind")
    resolved = resolve_config(str(registry["source_config_path"]))
    lp.prepare(resolved, output_root, resume=True)
    reference = _load_lp_reference(resolved)
    rows: list[dict[str, Any]] = []
    for job in registry["jobs"]:
        status = _read_status(output_root, str(job["job_id"]))
        _verify_status_binding_impl(job, status, require_complete=True)
        seed = int(job["seed"])
        nolp_mae = float(status["best_mae"])
        persistence = float(status["persistence_mae"])
        lp_row = reference[seed]["summary_row"]
        lp_mae = float(lp_row["best_q3_mae"])
        lp_persistence = float(lp_row["persistence_q3_mae"])
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (nolp_mae, lp_mae, persistence, lp_persistence)
        ):
            raise ValueError(f"Invalid paired metric for seed {seed}")
        if not math.isclose(persistence, lp_persistence, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Persistence baseline drifted for seed {seed}")
        best_epoch = int(status["best_epoch"])
        completed_epochs = int(status["completed_epochs"])
        if not (1 <= best_epoch <= completed_epochs <= MAX_EPOCHS):
            raise ValueError(f"Invalid epoch bounds for seed {seed}")
        paired_log_ratio = math.log(nolp_mae / lp_mae)
        rows.append(
            {
                "seed": seed,
                "nolp_job_id": str(job["job_id"]),
                "lp_job_id": str(job["paired_lp_job_id"]),
                "nolp_best_epoch": best_epoch,
                "nolp_completed_epochs": completed_epochs,
                "lp_best_epoch": int(lp_row["best_epoch"]),
                "lp_completed_epochs": int(lp_row["completed_epochs"]),
                "nolp_q3_mae": nolp_mae,
                "lp_q3_mae": lp_mae,
                "persistence_q3_mae": persistence,
                "nolp_minus_lp_mae": nolp_mae - lp_mae,
                "log_nolp_over_lp_mae": paired_log_ratio,
                "nolp_improvement_over_lp_pct": (1.0 - math.exp(paired_log_ratio))
                * 100.0,
                "nolp_beats_lp": nolp_mae <= lp_mae,
                "nolp_improvement_vs_persistence_pct": (1.0 - nolp_mae / persistence)
                * 100.0,
                "lp_improvement_vs_persistence_pct": (1.0 - lp_mae / persistence)
                * 100.0,
                "nolp_run_dir": str(status["run_dir"]),
                "lp_run_dir": str(reference[seed]["status"]["run_dir"]),
            }
        )
    rows.sort(key=lambda row: SEEDS.index(int(row["seed"])))
    if len(rows) != EXPECTED_JOB_COUNT:
        raise ValueError("Paired comparison coverage drifted")
    nolp_maes = [float(row["nolp_q3_mae"]) for row in rows]
    lp_maes = [float(row["lp_q3_mae"]) for row in rows]
    paired_logs = [float(row["log_nolp_over_lp_mae"]) for row in rows]
    nolp_logs_vs_persistence = [
        math.log(float(row["nolp_q3_mae"]) / float(row["persistence_q3_mae"]))
        for row in rows
    ]
    mean_paired_log = sum(paired_logs) / len(paired_logs)
    mean_nolp_log = sum(nolp_logs_vs_persistence) / len(nolp_logs_vs_persistence)
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(analysis_dir / "job_summary.csv", _csv_text(rows))
    _atomic_write_text(analysis_dir / "critic_paired_comparison.csv", _csv_text(rows))
    summary = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "selection_scope": "Q3 development only; Q4 not materialized",
        "comparison_scope": (
            "exact paired critic-only comparison; Text128+c32, same five seeds, "
            "same initialization and max-240 training protocol"
        ),
        "completed_jobs": len(rows),
        "seed_count": len(rows),
        "nolp_mean_best_q3_mae": sum(nolp_maes) / len(nolp_maes),
        "nolp_sample_std_best_q3_mae": _sample_std(nolp_maes),
        "lp_mean_best_q3_mae": sum(lp_maes) / len(lp_maes),
        "lp_sample_std_best_q3_mae": _sample_std(lp_maes),
        "mean_log_nolp_over_lp_mae": mean_paired_log,
        "geomean_nolp_over_lp_mae_ratio": math.exp(mean_paired_log),
        "geomean_nolp_improvement_over_lp_pct": (1.0 - math.exp(mean_paired_log))
        * 100.0,
        "nolp_seeds_beating_lp": sum(bool(row["nolp_beats_lp"]) for row in rows),
        "lp_seeds_beating_nolp": sum(not bool(row["nolp_beats_lp"]) for row in rows),
        "point_winner": "nolp" if mean_paired_log < 0.0 else "lp",
        "nolp_geomean_improvement_vs_persistence_pct": (1.0 - math.exp(mean_nolp_log))
        * 100.0,
        "rows": rows,
        "created_at": _utc_now(),
    }
    _write_json(analysis_dir / "five_seed_summary.json", summary)
    _write_json(analysis_dir / "critic_paired_comparison.json", summary)
    return summary


@contextmanager
def _configured_lp() -> Iterator[None]:
    replacements: dict[str, Any] = {
        "EXPERIMENT_KIND": EXPERIMENT_KIND,
        "BENCHMARK_EXPERIMENT_KIND": BENCHMARK_EXPERIMENT_KIND,
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "DEFAULT_BENCHMARK_DIR": DEFAULT_BENCHMARK_DIR,
        "SEEDS": SEEDS,
        "GPU_IDS": GPU_IDS,
        "ARM_ID": ARM_ID,
        "ARM_IDS": ARM_IDS,
        "ARM_CONTRACTS": ARM_CONTRACTS,
        "CAPACITY_IDS": CAPACITY_IDS,
        "CAPACITY_CONTRACTS": CAPACITY_CONTRACTS,
        "EXPECTED_JOB_COUNT": EXPECTED_JOB_COUNT,
        "LEARNING_RATE": LEARNING_RATE,
        "MAX_EPOCHS": MAX_EPOCHS,
        "ORCHESTRATOR_PATH": ORCHESTRATOR_PATH,
        "SOURCE_PATHS": SOURCE_PATHS,
        "WORKER_MODULE": WORKER_MODULE,
        "_experiment_specs_impl": _experiment_specs_impl,
        "_training_payload_impl": _training_payload_impl,
        "_source_hashes_impl": _source_hashes_impl,
        "_initial_state_fairness_impl": _initial_state_fairness_impl,
        "_prepare_benchmark_impl": _prepare_benchmark_impl,
        "_postprocess_impl": _postprocess_impl,
        "_verify_status_binding": _verify_status_binding_impl,
    }
    originals = {name: getattr(lp, name) for name in replacements}
    original_previous_verify = lp.previous._verify_status_binding
    try:
        for name, value in replacements.items():
            setattr(lp, name, value)
        lp.previous._verify_status_binding = _verify_status_binding_impl
        yield
    finally:
        lp.previous._verify_status_binding = original_previous_verify
        for name, value in originals.items():
            setattr(lp, name, value)


def experiment_specs(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    with _configured_lp():
        return lp.experiment_specs(resolved)


def _training_payload(
    resolved: Mapping[str, Any], output_root: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    with _configured_lp():
        return lp._training_payload(resolved, output_root, spec)


def _initial_state_fairness(
    specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _initial_state_fairness_impl(specs)


def prepare(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with _configured_lp():
        return lp.prepare(resolved, output_root, resume=resume)


def _prepare_benchmark(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with _configured_lp():
        return lp._prepare_benchmark(resolved, output_root, resume=resume)


def dry_run(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with _configured_lp():
        return lp.dry_run(resolved, output_root, resume=resume)


def run_worker(
    resolved: Mapping[str, Any], output_root: Path, *, job_id: str
) -> dict[str, Any]:
    with _configured_lp():
        return lp.run_worker(resolved, output_root, job_id=job_id)


def benchmark(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with _configured_lp():
        return lp.benchmark(resolved, output_root, resume=resume)


def postprocess(output_root: Path) -> dict[str, Any]:
    with _configured_lp():
        return _postprocess_impl(output_root)


def launch(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with _configured_lp():
        return lp.launch(resolved, output_root, resume=resume)


def status(output_root: Path) -> dict[str, Any]:
    with _configured_lp():
        return lp.status(output_root)


def run_pipeline(
    resolved: Mapping[str, Any],
    output_root: Path,
    *,
    resume: bool,
    benchmark_root: Path | None = None,
) -> dict[str, Any]:
    with _configured_lp():
        return lp.run_pipeline(
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
