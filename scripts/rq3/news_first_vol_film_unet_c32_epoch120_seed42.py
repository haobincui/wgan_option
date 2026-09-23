"""Run the independent c32, seed-42, 120-epoch Mask+Coords Q3 experiment.

The four jobs start from fresh deterministic initialization.  This module
reuses the already-tested capacity-sweep primitives while owning a separate
config, registry, source-hash contract, benchmark, and output root.  It never
materializes a Q4/test loader.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Iterator, Mapping, Sequence

import yaml

import scripts._path_setup  # noqa: F401
from scripts.rq3 import news_first_vol_film_unet_capacity_epoch_3seed as core


EXPERIMENT_KIND = "film_unet_mask_coords_c32_epoch120_seed42_q3_v1"
BENCHMARK_EXPERIMENT_KIND = (
    "film_unet_mask_coords_c32_epoch120_seed42_epoch1_benchmark_v1"
)
DEFAULT_CONFIG = "configs/rq3/news_first_vol_film_unet_c32_epoch120_seed42.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/rq3_news_first_vol_film_unet_c32_epoch120_seed42_exact_ttm_v1"
)
DEFAULT_BENCHMARK_DIR = (
    "outputs/benchmarks/rq3_news_first_vol_film_unet_c32_epoch120_seed42_epoch1_v1"
)
SEEDS = (42,)
GPU_IDS = (0, 1)
CAPACITY_IDS = ("c32",)
EXPECTED_JOB_COUNT = 4
LEARNING_RATE = 5e-7
ARM_IDS = core.ARM_IDS
ARM_CONTRACTS = core.ARM_CONTRACTS
PROFILE_FIELDS = core.PROFILE_FIELDS
CAPACITY_CONTRACTS = {
    "c32": dict(zip(PROFILE_FIELDS, (32, 0, 1024, 32, 0, 128, 786), strict=True))
}
REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_PATH = Path(__file__).resolve()
CORE_ORCHESTRATOR_PATH = Path(core.__file__).resolve()
SOURCE_PATHS = (
    ORCHESTRATOR_PATH,
    CORE_ORCHESTRATOR_PATH,
    *tuple(path for path in core.SOURCE_PATHS if path != CORE_ORCHESTRATOR_PATH),
)
WORKER_MODULE = "scripts.rq3.news_first_vol_film_unet_c32_epoch120_seed42"
REQUIRED_ARTIFACT_ROLES = frozenset(
    {
        "best_learned",
        "generator_best_learned",
        "discriminator_best_learned",
        "training_metrics",
        "resolved_config",
        "run_log",
        "generator_initial",
        "critic_initial",
        "generator_final",
        "critic_final",
    }
)

_payload_sha256 = core._payload_sha256
_sha256_file = core._sha256_file
_utc_now = core._utc_now
_write_json = core._write_json
_write_yaml = core._write_yaml
_atomic_write_text = core._atomic_write_text
_require_mapping = core._require_mapping
_read_json = core._read_json
_artifact = core._artifact
_registry_path = core._registry_path
_job_status_path = core._job_status_path
_read_status = core._read_status
_write_status = core._write_status
_verify_artifacts = core._verify_artifacts
_discover_completed = core._discover_completed
_find_job = core._find_job
_counts = core._counts
_resource_snapshot = core._resource_snapshot
_resource_peaks = core._resource_peaks
_benchmark_resource_gate = core._benchmark_resource_gate
_write_supervisor_status = core._write_supervisor_status
_terminate_process_group = core._terminate_process_group
_csv_text = core._csv_text
_pid_alive = core._pid_alive

_CORE_TRAINING_PAYLOAD = core._training_payload
_CORE_SOURCE_HASHES = core._source_hashes


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
        "max_epochs": 120,
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
    for arm_id, expected_arm in ARM_CONTRACTS.items():
        arm = _require_mapping(arms.get(arm_id), f"arms.{arm_id}")
        if not str(arm.get("label", "")).strip():
            raise ValueError(f"arms.{arm_id}.label must be non-empty")
        for key, expected in expected_arm.items():
            actual = int(arm[key]) if isinstance(expected, int) else str(arm[key])
            if actual != expected:
                raise ValueError(f"arms.{arm_id}.{key} drifted")

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
    comparison_path = Path(str(comparison.get("epoch60_job_summary", "")))
    if not comparison_path.is_absolute():
        comparison_path = REPO_ROOT / comparison_path
    if not comparison_path.is_file():
        raise FileNotFoundError(
            f"Frozen epoch-60 comparison is missing: {comparison_path}"
        )
    expected_sha = str(comparison.get("epoch60_job_summary_sha256", ""))
    if _sha256_file(comparison_path) != expected_sha:
        raise ValueError("Frozen epoch-60 comparison SHA drifted")

    runtime = _require_mapping(resolved.get("runtime"), "runtime")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != GPU_IDS:
        raise ValueError("runtime.gpu_ids must be [0, 1]")
    for key in ("workers_per_gpu", "benchmark_workers_per_gpu"):
        if int(runtime.get(key, -1)) != 2:
            raise ValueError(f"runtime.{key} must be 2")
    if int(runtime.get("cpu_threads_per_job", -1)) != 1:
        raise ValueError("runtime.cpu_threads_per_job must be 1")
    for key in ("poll_interval_seconds", "resource_sample_interval_seconds"):
        if float(runtime.get(key, 0.0)) <= 0:
            raise ValueError(f"runtime.{key} must be positive")
    if not math.isclose(float(runtime.get("max_peak_gpu_memory_gib", 0)), 20.0):
        raise ValueError("runtime.max_peak_gpu_memory_gib must be 20")
    if not math.isclose(float(runtime.get("max_host_ram_fraction", 0)), 0.85):
        raise ValueError("runtime.max_host_ram_fraction must be 0.85")


def resolve_config(config_path: str | Path) -> dict[str, Any]:
    source = _resolve_repo_path(config_path)
    root = _require_mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "root")
    resolved = _require_mapping(
        root.get("film_unet_c32_epoch120_seed42"),
        "film_unet_c32_epoch120_seed42",
    )
    resolved["source_config_path"] = str(source)
    resolved["base_training_config"] = str(
        _resolve_repo_path(str(resolved["base_training_config"]))
    )
    resolved["runtime"] = _require_mapping(resolved.get("runtime"), "runtime")
    resolved["runtime"]["python_executable"] = str(
        _resolve_repo_path(str(resolved["runtime"]["python_executable"]))
    )
    comparison = _require_mapping(resolved.get("comparison"), "comparison")
    comparison["epoch60_job_summary"] = str(
        _resolve_repo_path(str(comparison["epoch60_job_summary"]))
    )
    resolved["comparison"] = comparison
    _validate_resolved(resolved)
    base = _require_mapping(
        yaml.safe_load(
            Path(str(resolved["base_training_config"])).read_text(encoding="utf-8")
        ),
        "base training config",
    )
    core._validate_base_training_config(base)
    return resolved


def _source_hashes_impl(resolved: Mapping[str, Any]) -> dict[str, str]:
    hashes = _CORE_SOURCE_HASHES(resolved)
    comparison = _require_mapping(resolved["comparison"], "comparison")
    comparison_path = Path(str(comparison["epoch60_job_summary"]))
    actual_sha = _sha256_file(comparison_path)
    if actual_sha != str(comparison["epoch60_job_summary_sha256"]):
        raise ValueError("Frozen epoch-60 comparison SHA drifted")
    hashes[f"input_sha256::{comparison_path.relative_to(REPO_ROOT)}"] = actual_sha
    return hashes


def _training_payload_impl(
    resolved: Mapping[str, Any], output_root: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _CORE_TRAINING_PAYLOAD(resolved, output_root, spec)
    payload["news_first_lr_profile"] = "film_unet_c32_epoch120_lr_5e_07_no_warmup"
    return payload


def _experiment_specs_impl(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    capacity_id = "c32"
    comparison = _require_mapping(resolved["comparison"], "comparison")
    gpu_by_arm = {
        "film_unet_mask_coords_text128": 1,
        "film_unet_mask_coords_text64": 0,
        "film_unet_mask_coords_text64_nolp": 1,
        "film_unet_mask_coords_text64_projection": 0,
    }
    specs: list[dict[str, Any]] = []
    for arm_id in ARM_IDS:
        architecture = core._architecture_contract(resolved, arm_id, capacity_id)
        model = core._model_contract(resolved, arm_id, capacity_id)
        initial_state = core._initial_state_hashes(
            resolved, arm_id, capacity_id, SEEDS[0]
        )
        spec = {
            "job_id": f"{arm_id}_{capacity_id}_seed_{SEEDS[0]:03d}",
            "experiment_kind": EXPERIMENT_KIND,
            "arm_id": arm_id,
            "arm_label": architecture["arm_label"],
            "capacity_id": capacity_id,
            "seed": SEEDS[0],
            "gpu_id": gpu_by_arm[arm_id],
            "learning_rate": LEARNING_RATE,
            "training_state_inherited": False,
            "predecessor_job_summary_path": str(comparison["epoch60_job_summary"]),
            "predecessor_job_summary_sha256": str(
                comparison["epoch60_job_summary_sha256"]
            ),
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
        raise AssertionError("Expected four unique formal jobs")
    gpu_counts = {
        gpu: sum(int(spec["gpu_id"]) == gpu for spec in specs) for gpu in GPU_IDS
    }
    if gpu_counts != {0: 2, 1: 2}:
        raise AssertionError(f"GPU assignment drifted: {gpu_counts}")
    core._initial_state_fairness(specs)
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
        "CAPACITY_IDS": CAPACITY_IDS,
        "CAPACITY_CONTRACTS": CAPACITY_CONTRACTS,
        "EXPECTED_JOB_COUNT": EXPECTED_JOB_COUNT,
        "ORCHESTRATOR_PATH": ORCHESTRATOR_PATH,
        "SOURCE_PATHS": SOURCE_PATHS,
        "experiment_specs": _experiment_specs_impl,
        "_training_payload": _training_payload_impl,
        "_source_hashes": _source_hashes_impl,
    }
    previous = {name: getattr(core, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(core, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(core, name, value)


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
    with _configured_core():
        return core._initial_state_fairness(specs)


def _verify_status_binding(
    job: Mapping[str, Any],
    status_payload: Mapping[str, Any],
    *,
    require_complete: bool,
) -> None:
    """Bind a status and its artifacts to exactly one immutable registry job."""

    job_id = str(job["job_id"])
    if str(status_payload.get("job_id", "")) != job_id:
        raise ValueError(f"Status job_id binding drift for {job_id}")
    if str(status_payload.get("job_spec_sha256", "")) != str(job["job_spec_sha256"]):
        raise ValueError(f"Status job_spec_sha256 binding drift for {job_id}")
    state = str(status_payload.get("state", ""))
    if state not in {"pending", "running", "complete", "failed"}:
        raise ValueError(f"Invalid status state for {job_id}: {state!r}")
    if require_complete and state != "complete":
        raise ValueError(f"Job is not complete: {job_id}")
    if state != "complete":
        return

    _verify_artifacts(status_payload)
    artifacts = status_payload.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise ValueError(f"Malformed artifact list for {job_id}")
    roles = [
        str(_require_mapping(item, "artifact")["artifact_role"]) for item in artifacts
    ]
    if len(roles) != len(set(roles)) or set(roles) != REQUIRED_ARTIFACT_ROLES:
        raise ValueError(f"Completed artifact-role contract drift for {job_id}")

    run_root = Path(str(job["run_root"])).resolve()
    run_dir = Path(str(status_payload.get("run_dir", ""))).resolve()
    try:
        run_dir.relative_to(run_root)
    except ValueError as exc:
        raise ValueError(
            f"Completed run_dir escapes job run_root for {job_id}"
        ) from exc
    for raw_artifact in artifacts:
        artifact_path = Path(
            str(_require_mapping(raw_artifact, "artifact")["path"])
        ).resolve()
        try:
            artifact_path.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError(
                f"Completed artifact escapes run_dir for {job_id}: {artifact_path}"
            ) from exc


def _unsigned_job_spec(job: Mapping[str, Any]) -> dict[str, Any]:
    generated_fields = {"config_path", "config_sha256", "run_root", "job_spec_sha256"}
    return {key: value for key, value in job.items() if key not in generated_fields}


def _verify_registry_bindings(
    registry: Mapping[str, Any],
    output_root: Path,
    *,
    expected_specs: Sequence[Mapping[str, Any]] | None,
) -> None:
    jobs = [_require_mapping(job, "registry job") for job in registry.get("jobs", ())]
    if len(jobs) != EXPECTED_JOB_COUNT:
        raise ValueError("Registry must contain exactly four jobs")
    job_ids = [str(job.get("job_id", "")) for job in jobs]
    if len(set(job_ids)) != EXPECTED_JOB_COUNT:
        raise ValueError("Registry job IDs are not unique")
    unsigned_specs = [_unsigned_job_spec(job) for job in jobs]
    for job, unsigned in zip(jobs, unsigned_specs, strict=True):
        if _payload_sha256(unsigned) != str(job.get("job_spec_sha256", "")):
            raise ValueError(f"Registry job_spec SHA drift for {job['job_id']}")
    signed_specs = [
        {**unsigned, "job_spec_sha256": str(job["job_spec_sha256"])}
        for job, unsigned in zip(jobs, unsigned_specs, strict=True)
    ]
    if _payload_sha256(signed_specs) != str(registry.get("jobs_payload_sha256", "")):
        raise ValueError("Registry jobs payload SHA drift")

    if expected_specs is not None:
        expected_by_id = {
            str(spec["job_id"]): _require_mapping(spec, "expected spec")
            for spec in expected_specs
        }
        if set(expected_by_id) != set(job_ids):
            raise ValueError("Registry job set drifted from fresh experiment specs")
        for job in jobs:
            expected = expected_by_id[str(job["job_id"])]
            if any(job.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Registry job spec drift for {job['job_id']}")

    for job in jobs:
        status_payload = _read_status(output_root, str(job["job_id"]))
        _verify_status_binding(job, status_payload, require_complete=False)


def prepare(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with _configured_core():
        registry = core.prepare(resolved, output_root, resume=resume)
        expected_specs = _experiment_specs_impl(resolved)
    _verify_registry_bindings(registry, output_root, expected_specs=expected_specs)
    return registry


def _prepare_benchmark(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    with _configured_core():
        registry = core._prepare_benchmark(resolved, output_root, resume=resume)
    # The inherited builder is otherwise exact; freeze the semantically precise
    # branch-local benchmark label before any training occurs.
    expected_label = "film_unet_c32_epoch120_seed42_epoch1_benchmark"
    changed = False
    for job in registry["jobs"]:
        config_path = Path(str(job["config_path"]))
        payload = _require_mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            "benchmark job config",
        )
        current = str(payload.get("news_first_lr_profile", ""))
        if current == expected_label:
            continue
        if resume:
            raise ValueError("Benchmark LR-profile label drift detected")
        payload["news_first_lr_profile"] = expected_label
        _write_yaml(config_path, payload)
        job["config_sha256"] = _sha256_file(config_path)
        changed = True
    if changed:
        _write_json(_registry_path(output_root), registry)
    _verify_registry_bindings(registry, output_root, expected_specs=None)
    return registry


def dry_run(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    registry_exists = _registry_path(output_root).is_file()
    prepare(resolved, output_root, resume=resume if registry_exists else False)
    with _configured_core():
        manifest = core.dry_run(resolved, output_root, resume=True)
    prepare(resolved, output_root, resume=True)
    return manifest


def run_worker(
    resolved: Mapping[str, Any], output_root: Path, *, job_id: str
) -> dict[str, Any]:
    with _configured_core():
        result = core.run_worker(resolved, output_root, job_id=job_id)
    registry = prepare(resolved, output_root, resume=True)
    job = _find_job(registry, job_id)
    _verify_status_binding(job, result, require_complete=True)
    return result


def _pending_jobs(
    registry: Mapping[str, Any], output_root: Path
) -> dict[int, list[dict[str, Any]]]:
    pending = {gpu: [] for gpu in GPU_IDS}
    for raw_job in registry["jobs"]:
        job = dict(raw_job)
        job_id = str(job["job_id"])
        job_status = _read_status(output_root, job_id)
        if job_status.get("state") == "complete":
            _verify_status_binding(job, job_status, require_complete=True)
            continue
        recovered = _discover_completed(job)
        if recovered is not None:
            recovered_status = {
                "job_id": job_id,
                "state": "complete",
                "attempt": int(job_status.get("attempt", 0)),
                "updated_at": _utc_now(),
                "job_spec_sha256": job["job_spec_sha256"],
                **recovered,
            }
            _verify_status_binding(job, recovered_status, require_complete=True)
            _write_status(output_root, job_id, recovered_status)
            continue
        pending[int(job["gpu_id"])].append(job)
    return pending


def _supervise(
    resolved: Mapping[str, Any],
    output_root: Path,
    registry: Mapping[str, Any],
    *,
    benchmark_mode: bool,
) -> None:
    runtime = _require_mapping(resolved["runtime"], "runtime")
    workers_key = "benchmark_workers_per_gpu" if benchmark_mode else "workers_per_gpu"
    workers_per_gpu = int(runtime[workers_key])
    pending = _pending_jobs(registry, output_root)
    active: dict[str, dict[str, Any]] = {}
    stop_requested = False
    failure = ""
    last_resource = 0.0

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    _write_supervisor_status(registry, output_root, state="running", active=active)
    while any(pending.values()) or active:
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
            running = sum(record["gpu_id"] == gpu for record in active.values())
            while running < workers_per_gpu and pending[gpu]:
                job = pending[gpu].pop(0)
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
                if benchmark_mode:
                    command = [
                        str(runtime["python_executable"]),
                        "scripts/train/train_vol.py",
                        "--config",
                        str(job["config_path"]),
                        "--train-only",
                    ]
                else:
                    command = [
                        str(runtime["python_executable"]),
                        "-m",
                        WORKER_MODULE,
                        "worker",
                        "--config",
                        str(resolved["source_config_path"]),
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
                if benchmark_mode:
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
                print(f"launched {job_id} pid={process.pid} gpu={gpu}", flush=True)
                running += 1

        for job_id, record in list(active.items()):
            returncode = record["process"].poll()
            if returncode is None:
                continue
            record["handle"].close()
            job = record["job"]
            completed = (
                _discover_completed(job)
                if benchmark_mode and returncode == 0
                else _read_status(output_root, job_id)
            )
            valid = (
                completed is not None
                and (benchmark_mode or str(completed.get("state")) == "complete")
                and returncode == 0
            )
            if not valid:
                failure = (
                    f"{job_id} failed verification with return code {returncode}; "
                    f"see {record['log_path']}"
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
            elif benchmark_mode:
                completed_status = {
                    "job_id": job_id,
                    "state": "complete",
                    "attempt": record["attempt"],
                    "returncode": 0,
                    "updated_at": _utc_now(),
                    "job_spec_sha256": job["job_spec_sha256"],
                    "stdout_log": str(record["log_path"].resolve()),
                    **dict(completed),
                }
                _verify_status_binding(job, completed_status, require_complete=True)
                _write_status(output_root, job_id, completed_status)
            else:
                _verify_status_binding(job, completed, require_complete=True)
            if valid:
                print(f"completed {job_id}", flush=True)
            del active[job_id]

        now = time.monotonic()
        if now - last_resource >= float(runtime["resource_sample_interval_seconds"]):
            _resource_snapshot(output_root)
            last_resource = now
        _write_supervisor_status(
            registry,
            output_root,
            state="running",
            active=active,
            message=failure,
        )
        if any(pending.values()) or active:
            time.sleep(float(runtime["poll_interval_seconds"]))


def benchmark(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    if output_root.resolve() == _resolve_repo_path(DEFAULT_OUTPUT_DIR):
        raise ValueError("Benchmark root must differ from the formal root")
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
    _supervise(resolved, output_root, registry, benchmark_mode=True)

    rows: list[dict[str, Any]] = []
    for job in registry["jobs"]:
        job_status = _read_status(output_root, str(job["job_id"]))
        _verify_status_binding(job, job_status, require_complete=True)
        rows.append(
            {
                "job_id": job["job_id"],
                "arm_id": job["arm_id"],
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
    _atomic_write_text(output_root / "analysis/benchmark_jobs.csv", _csv_text(rows))
    _resource_snapshot(output_root)
    peaks = _resource_peaks(output_root)
    gate = _benchmark_resource_gate(peaks, resolved["runtime"])
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
        "workers_per_gpu": int(resolved["runtime"]["benchmark_workers_per_gpu"]),
        "gpu_job_counts": {
            str(gpu): sum(int(row["gpu_id"]) == gpu for row in rows) for gpu in GPU_IDS
        },
        "resource_peaks": peaks,
        "resource_gate": gate,
        "gate_passed": bool(gate["passed"]),
        "completed_at": _utc_now(),
    }
    _write_json(output_root / "analysis/benchmark_summary.json", summary)
    if not gate["passed"]:
        message = "Benchmark resource gate failed; formal root was not prepared"
        _write_supervisor_status(
            registry, output_root, state="failed", active={}, message=message
        )
        raise RuntimeError(message)
    _write_supervisor_status(registry, output_root, state="complete", active={})
    return summary


def _epoch60_rows(resolved: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    comparison = _require_mapping(resolved["comparison"], "comparison")
    path = Path(str(comparison["epoch60_job_summary"]))
    if _sha256_file(path) != str(comparison["epoch60_job_summary_sha256"]):
        raise ValueError("Frozen epoch-60 comparison SHA drifted")
    with path.open(encoding="utf-8", newline="") as handle:
        selected = {
            str(row["arm_id"]): row
            for row in csv.DictReader(handle)
            if str(row["capacity_id"]) == "c32" and int(row["seed"]) == SEEDS[0]
        }
    if tuple(selected) != ARM_IDS:
        raise ValueError("Frozen epoch-60 comparison does not contain the four arms")
    return selected


def postprocess(output_root: Path) -> dict[str, Any]:
    registry = _read_json(_registry_path(output_root))
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Unexpected experiment kind")
    resolved = resolve_config(str(registry["source_config_path"]))
    # Revalidate source/config/generated-config hashes before reading metrics.
    prepare(resolved, output_root, resume=True)
    old_rows = _epoch60_rows(resolved)
    rows: list[dict[str, Any]] = []
    for job in registry["jobs"]:
        job_status = _read_status(output_root, str(job["job_id"]))
        _verify_status_binding(job, job_status, require_complete=True)
        mae = float(job_status["best_mae"])
        persistence = float(job_status["persistence_mae"])
        best_epoch = int(job_status["best_epoch"])
        completed_epochs = int(job_status["completed_epochs"])
        if not (
            math.isfinite(mae)
            and mae > 0.0
            and math.isfinite(persistence)
            and persistence > 0.0
        ):
            raise ValueError(
                f"Non-finite or non-positive Q3 metric for {job['job_id']}"
            )
        if not (1 <= best_epoch <= completed_epochs <= 120):
            raise ValueError(f"Invalid epoch bounds for {job['job_id']}")
        old = old_rows[str(job["arm_id"])]
        old_mae = float(old["best_q3_mae"])
        if not math.isfinite(old_mae) or old_mae <= 0.0:
            raise ValueError(f"Invalid epoch-60 comparison for {job['job_id']}")
        rows.append(
            {
                "job_id": job["job_id"],
                "arm_id": job["arm_id"],
                "arm_label": job["arm_label"],
                "capacity_id": "c32",
                "seed": SEEDS[0],
                "gpu_id": job["gpu_id"],
                "generator_conditioning_mode": job["generator_conditioning_mode"],
                "critic_conditioning_mode": job["critic_conditioning_mode"],
                "gen_text_hidden_dim": job["gen_text_hidden_dim"],
                "gen_text_out_dim": job["gen_text_out_dim"],
                "generator_parameters": job["generator_parameters"],
                "critic_parameters": job["critic_parameters"],
                "wgan_parameters": job["wgan_parameters"],
                "training_state_inherited": False,
                "best_epoch_120_run": best_epoch,
                "best_epoch_after_60": best_epoch > 60,
                "completed_epochs_120_run": completed_epochs,
                "best_q3_mae_120_run": mae,
                "persistence_q3_mae": persistence,
                "improvement_vs_persistence_pct": (persistence - mae)
                / persistence
                * 100.0,
                "best_epoch_60_run": int(old["best_epoch"]),
                "best_q3_mae_60_run": old_mae,
                "mae_delta_120_minus_60": mae - old_mae,
                "improvement_vs_epoch60_pct": (old_mae - mae) / old_mae * 100.0,
                "run_dir": job_status["run_dir"],
            }
        )
    if len(rows) != EXPECTED_JOB_COUNT:
        raise ValueError("Postprocess row count drifted")
    if len({float(row["persistence_q3_mae"]) for row in rows}) != 1:
        raise ValueError("Persistence baseline drifted across arms")
    rows.sort(key=lambda row: float(row["best_q3_mae_120_run"]))
    for rank, row in enumerate(rows, start=1):
        row["descriptive_rank"] = rank
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(analysis_dir / "job_summary.csv", _csv_text(rows))
    _atomic_write_text(analysis_dir / "architecture_ranking.csv", _csv_text(rows))
    result = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "selection_scope": "Q3 development only; Q4 not materialized",
        "inference_scope": "single-seed descriptive epoch extension",
        "fresh_initialization": True,
        "training_state_inherited": False,
        "completed_jobs": len(rows),
        "rank_metric": "best_q3_mae_120_run",
        "point_leader_arm_id": rows[0]["arm_id"],
        "epoch60_comparison_path": str(resolved["comparison"]["epoch60_job_summary"]),
        "epoch60_comparison_sha256": resolved["comparison"][
            "epoch60_job_summary_sha256"
        ],
        "rows": rows,
        "created_at": _utc_now(),
    }
    _write_json(analysis_dir / "architecture_ranking.json", result)
    return result


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
    _supervise(resolved, output_root, registry, benchmark_mode=False)
    result = postprocess(output_root)
    _write_supervisor_status(registry, output_root, state="complete", active={})
    print(f"all {EXPECTED_JOB_COUNT} jobs completed", flush=True)
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
    selected_benchmark_root = (
        _resolve_repo_path(DEFAULT_BENCHMARK_DIR)
        if benchmark_root is None
        else benchmark_root.resolve()
    )
    if selected_benchmark_root == output_root.resolve():
        raise ValueError("Benchmark root must differ from the formal output root")
    benchmark(
        resolved,
        selected_benchmark_root,
        resume=_registry_path(selected_benchmark_root).is_file(),
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
