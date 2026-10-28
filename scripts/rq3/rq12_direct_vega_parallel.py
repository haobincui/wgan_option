"""Hash-bound, work-conserving two-GPU scheduler for the RQ1/2 direct matrix.

The two audited child workers remain responsible for training, checkpoint
selection and research validation. This supervisor owns one shared GPU budget
across both families and never opens a test loader during benchmarking/training.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "outputs/experiments/rq12_direct_vega_20261007_v1"
DEFAULT_FILM_CONFIG = REPO_ROOT / "configs/rq3/rq12_direct_vega_20261007_film.yaml"
DEFAULT_PURE_CONFIG = REPO_ROOT / "configs/rq3/rq12_direct_vega_20261007_pure.yaml"
DEFAULT_COMPOSITE_CONFIG = REPO_ROOT / "configs/rq3/rq12_direct_vega_20261007_composite.yaml"
DEFAULT_BENCHMARK_WORKSPACE = Path("/tmp/rq12_direct_vega_20261007_v1")
CANDIDATES = (8, 10, 12, 16, 18, 24, 32, 48)
FALLBACK_CANDIDATES = (4, 2, 1)
SCHEDULER_MODE = "rq12_vega_shared_slots_v1"
BENCHMARK_KIND = "rq12_direct_vega_global_benchmark_v1"
OVERRIDE_KIND = "rq12_direct_vega_user_concurrency_override_v1"
OVERRIDE_INSTRUCTION = "直接开始每张卡20个任务，不用测试"
OVERRIDE_INSTRUCTIONS = {
    20: OVERRIDE_INSTRUCTION,
    40: "帮我看一下是不是在计算mae的时候使用了vega weight，其他地方没有使用，如果不是的话帮我改成这个版本，然后重新开始每张卡40并发的训练",
}
CONTRACT_KIND = "rq12_direct_vega_parallel_runtime_v1"
MODULES = {
    "film": "scripts.rq3.news_first_vol_film_unet_text_10seed_lr2p5e5",
    "pure": "scripts.rq3.news_first_vol_cnn_unet_pure_no_text_10seed",
}
EXPECTED_COUNTS = {"film": 160, "pure": 40}
FOLD_GPU = {"f1_2023q1": 0, "f2_2023q2": 1, "f3_2023q3": 0, "f4_2023q4": 1}


class ResourceLimitError(RuntimeError):
    """A measured GPU/RAM admission limit was exceeded."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in payload.items() if key != "payload_sha256"}
    result["payload_sha256"] = payload_sha256(result)
    return result


def read_json(path: str | Path) -> dict[str, Any]:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def read_signed(path: str | Path, *, kind: str | None = None) -> dict[str, Any]:
    result = read_json(path)
    unsigned = {key: value for key, value in result.items() if key != "payload_sha256"}
    if result.get("payload_sha256") != payload_sha256(unsigned):
        raise ValueError(f"Signed payload drift: {path}")
    if kind is not None and result.get("kind") != kind:
        raise ValueError(f"Payload kind drift: {path}")
    return result


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True,
                                   allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def resolve(path: str | Path) -> Path:
    result = Path(path).expanduser()
    return (REPO_ROOT / result).resolve() if not result.is_absolute() else result.resolve()


@contextmanager
def supervisor_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / "supervisor.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another supervisor owns {root}") from exc
        yield
    finally:
        os.close(descriptor)


def acquire_gpu_slot(root: Path, gpu: int, capacity: int) -> tuple[int, int] | None:
    """The child inherits this open description; parent close cannot unlock it."""
    directory = root / "gpu_slots" / f"gpu_{gpu}"
    directory.mkdir(parents=True, exist_ok=True)
    for slot in range(capacity):
        descriptor = os.open(directory / f"slot_{slot:03d}.lock",
                             os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            continue
        os.set_inheritable(descriptor, True)
        return descriptor, slot
    return None


def close_gpu_slot(descriptor: int) -> None:
    # Never LOCK_UN here: the worker's inherited description may still live.
    os.close(descriptor)


@dataclass(frozen=True)
class Child:
    family: str
    root: Path
    config_path: Path
    module: str

    def as_dict(self) -> dict[str, str]:
        return {"family": self.family, "root": str(self.root),
                "config_path": str(self.config_path), "module": self.module}


@dataclass(frozen=True)
class Job:
    child: Child
    spec: Mapping[str, Any]

    @property
    def key(self) -> str:
        return f"{self.child.family}:{self.spec['job_id']}"

    @property
    def gpu(self) -> int:
        return int(self.spec["gpu_id"])

    @property
    def status_path(self) -> Path:
        return self.child.root / "registry/job_status" / f"{self.spec['job_id']}.json"


def _child_call(child: Child, function: str, *args: Any, **kwargs: Any) -> Any:
    module = importlib.import_module(child.module)
    # These audited wrappers install mutable globals in two shared layers.
    # Enter BOTH layers even for validators/public helpers: preparation enters
    # them itself, while validate_root/postprocess otherwise see generic source
    # lists and report false frozen-manifest drift.
    if child.family == "film":
        outer = module.film_text_profile
        shared = module.multiseed
    elif child.family == "pure":
        outer = module.pure_multiseed_profile
        shared = module.multi
    else:
        raise ValueError(f"Unknown child family: {child.family}")
    with outer():
        with shared.multiseed_profile():
            target = getattr(module, function, None)
            if not callable(target):
                target = getattr(shared, function, None)
            if callable(target):
                return target(*args, **kwargs)
            # Shared benchmark helpers live in the underlying direct module.
            return shared._call(function, *args, **kwargs)


def _children_from_configs(film_config: str | Path, pure_config: str | Path,
                           *, benchmark_root: Path | None = None) -> list[Child]:
    result = []
    for family, config_path in (("film", film_config), ("pure", pure_config)):
        path = resolve(config_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw["runtime"].get("parallel_scheduler_mode") != SCHEDULER_MODE:
            raise ValueError(f"Shared scheduler mode missing: {path}")
        root = (benchmark_root / family if benchmark_root is not None
                else resolve(raw["experiment"]["output_root"]))
        result.append(Child(family, root, path, MODULES[family]))
    return result


def child_binding(child: Child) -> dict[str, Any]:
    rows = {"source_config": child.config_path,
            "resolved_config": child.root / "resolved_config.yaml",
            "code_manifest": child.root / "code_hashes.csv",
            "source_manifest": child.root / "source_hashes.csv",
            "config_manifest": child.root / "config_hashes.csv"}
    return {**child.as_dict(), "source_config_sha256": sha256_file(child.config_path),
            "files": {role: {"path": str(path), "sha256": sha256_file(path)}
                      for role, path in rows.items()}}


def verify_bindings(bindings: Sequence[Mapping[str, Any]]) -> None:
    for child in bindings:
        if sha256_file(child["config_path"]) != child["source_config_sha256"]:
            raise ValueError(f"Source config drift: {child['config_path']}")
        for role, row in child["files"].items():
            if sha256_file(row["path"]) != row["sha256"]:
                raise ValueError(f"Child {role} drift: {row['path']}")
            if role.endswith("_manifest"):
                with Path(row["path"]).open(newline="", encoding="utf-8") as stream:
                    for entry in csv.DictReader(stream):
                        path = Path(entry["path"])
                        if sha256_file(path) != entry["sha256"]:
                            raise ValueError(f"Manifest member drift: {path}")


def _registry(child: Child) -> dict[str, Any]:
    return read_json(child.root / "registry/task_registry.json")


def load_jobs(children: Sequence[Child], *, validate: bool = True,
              training: bool = True) -> list[Job]:
    result = []
    for child in children:
        if validate:
            _child_call(child, "validate_root", child.root)
        registry = _registry(child)
        specs = registry["jobs"]
        if len(specs) != EXPECTED_COUNTS[child.family]:
            raise ValueError(f"Child matrix size drift: {child.family}")
        if training and (registry.get("evaluation_frozen") or registry.get("test_data_opened")):
            raise RuntimeError(f"Training forbidden after evaluation freeze: {child.root}")
        counts = {gpu: sum(int(row["gpu_id"]) == gpu for row in specs) for gpu in (0, 1)}
        expected = EXPECTED_COUNTS[child.family] // 2
        if counts != {0: expected, 1: expected}:
            raise ValueError(f"Child GPU balance drift: {counts}")
        for spec in specs:
            if FOLD_GPU.get(str(spec["fold"])) != int(spec["gpu_id"]):
                raise ValueError(f"Fold/GPU contract drift: {spec['job_id']}")
            if spec.get("parent_state_path") or spec.get("recipe_path"):
                raise ValueError("Direct job must start from a fresh initialization")
            result.append(Job(child, dict(spec)))
    if len(result) != 200 or len({job.key for job in result}) != 200:
        raise ValueError("Global matrix must contain exactly 200 unique jobs")
    return result


def completed_valid(job: Job) -> bool:
    try:
        state = read_json(job.status_path)
        if state.get("status") != "completed":
            return False
        for key in ("job_id", "job_spec_sha256", "training_config_sha256"):
            if state.get(key) != job.spec.get(key):
                return False
        artifacts = state.get("artifacts") or []
        if not artifacts:
            return False
        roles = [str(row["artifact_role"]) for row in artifacts]
        if len(set(roles)) != len(roles):
            return False
        for row in artifacts:
            path = Path(row["path"])
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                return False
        return True
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False


def job_lock_held(job: Job) -> bool:
    path = job.child.root / "registry/job_locks" / f"{job.spec['job_id']}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(descriptor)


def balanced_queue(jobs: Sequence[Job], gpu: int) -> deque[Job]:
    """Four FiLM jobs then one Pure; heavier folds first within each family."""
    selected = [job for job in jobs if job.gpu == gpu]
    order = lambda job: (-int(str(job.spec["fold"])[1]), int(job.spec["seed"]),
                         str(job.spec["arm"]), job.key)
    families = {family: deque(sorted((job for job in selected
                                      if job.child.family == family), key=order))
                for family in ("film", "pure")}
    result: deque[Job] = deque()
    while any(families.values()):
        for family in ("film", "film", "film", "film", "pure"):
            if families[family]:
                result.append(families[family].popleft())
    return result


def resource_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    cards = {}
    for row in csv.reader(result.stdout.splitlines()):
        gpu, memory, utilization = (float(value.strip()) for value in row)
        if int(gpu) in (0, 1):
            cards[str(int(gpu))] = {"memory_used_mib": memory,
                                   "utilization_gpu_percent": utilization}
    if set(cards) != {"0", "1"}:
        raise RuntimeError("GPU telemetry must include physical cards 0 and 1")
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = int(value.strip().split()[0])
    return {"at_utc": utc_now(), "gpu": cards,
            "host_ram_fraction": 1.0 - memory["MemAvailable"] / memory["MemTotal"]}


def enforce_resources(snapshot: Mapping[str, Any], max_gpu_gib: float,
                      max_host_fraction: float) -> None:
    if any(float(card["memory_used_mib"]) >= max_gpu_gib * 1024
           for card in snapshot["gpu"].values()):
        raise ResourceLimitError(f"GPU memory reached {max_gpu_gib:g} GiB limit")
    if float(snapshot["host_ram_fraction"]) >= max_host_fraction:
        raise ResourceLimitError(f"Host RAM reached {max_host_fraction:.0%} limit")


def _worker_command(job: Job, python: str, descriptor: int, *, resume: bool) -> list[str]:
    result = [python, "-m", "scripts.rq3.rq12_direct_vega_parallel", "slot-worker",
              "--child-module", job.child.module, "--child-root", str(job.child.root),
              "--job-id", str(job.spec["job_id"]), "--slot-fd", str(descriptor)]
    if resume:
        result.append("--resume")
    return result


def _worker_environment(job: Job, log: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    environment["NEWS_FIRST_JOB_LOG_PATH"] = str(log.resolve())
    environment["DIRECT_SOURCE_MANIFEST_SHA256"] = sha256_file(job.child.root / "source_hashes.csv")
    environment["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    return environment


def _terminate(active: Mapping[str, tuple[Any, ...]]) -> None:
    for process, *_ in active.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for process, *_ in active.values():
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_queue(children: Sequence[Child], queue_root: str | Path, workers_per_gpu: int,
              *, resume: bool = False, max_gpu_gib: float = 20,
              max_host_fraction: float = .85, poll_seconds: float = 2,
              allow_benchmark: bool = False,
              job_keys: Sequence[str] | None = None) -> dict[str, Any]:
    root = resolve(queue_root)
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    jobs = load_jobs(children)
    if job_keys is not None:
        if not allow_benchmark or len(set(job_keys)) != len(job_keys):
            raise ValueError("Only benchmark smoke runs may select a job subset")
        requested = set(job_keys)
        jobs = [job for job in jobs if job.key in requested]
        if len(jobs) != len(requested) or not jobs:
            raise ValueError("Unknown or empty smoke job subset")
    expected_total = len(jobs)
    for child in children:
        registry = _registry(child)
        if registry["root_mode"] == "benchmark" and not allow_benchmark:
            raise ValueError("Benchmark worker roots cannot be launched as formal training")
    initial = [job for job in jobs if completed_valid(job)]
    if initial and not resume:
        raise RuntimeError("Completed jobs require --resume")
    remaining = [job for job in jobs if job not in initial]
    queues = {gpu: balanced_queue(remaining, gpu) for gpu in (0, 1)}
    bindings = [child_binding(child) for child in children]
    contract = signed({"schema_version": 1, "kind": CONTRACT_KIND,
                       "scheduler_mode": SCHEDULER_MODE, "gpu_ids": [0, 1],
                       "workers_per_gpu": workers_per_gpu, "expected_training_jobs": expected_total,
                       "expected_family_jobs": {family: sum(job.child.family == family for job in jobs)
                                                for family in MODULES},
                       "job_keys": sorted(job.key for job in jobs), "fold_gpu_assignment": FOLD_GPU,
                       "max_gpu_memory_gib": max_gpu_gib, "max_host_ram_fraction": max_host_fraction,
                       "supervisor_code_sha256": sha256_file(__file__), "children": bindings})
    contract_path = root / "runtime_contract.json"
    if contract_path.exists():
        if read_signed(contract_path, kind=CONTRACT_KIND) != contract:
            raise ValueError("Global runtime contract drift")
    else:
        write_json(contract_path, contract)
    configs = {child.family: yaml.safe_load((child.root / "resolved_config.yaml").read_text())
               for child in children}
    active: dict[str, tuple[Any, ...]] = {}
    completed = len(initial)
    failed = 0
    started = time.monotonic()
    start_utc = utc_now()
    peak_memory = {"0": 0.0, "1": 0.0}
    peak_ram = 0.0
    result: dict[str, Any] = {}

    def progress(state: str, *, error: str = "") -> dict[str, Any]:
        elapsed = time.monotonic() - started
        newly_completed = completed - len(initial)
        throughput = newly_completed / elapsed if elapsed > 0 else 0.0
        pending = expected_total - completed - len(active)
        payload = {"schema_version": 1, "status": state, "pid": os.getpid(),
                   "started_at_utc": start_utc, "updated_at_utc": utc_now(),
                   "expected": expected_total, "completed": completed, "failed": failed,
                   "running": len(active), "pending": pending,
                   "running_per_gpu": {str(gpu): sum(record[2].gpu == gpu for record in active.values())
                                       for gpu in (0, 1)},
                   "elapsed_seconds": elapsed, "throughput_jobs_per_second": throughput,
                   "eta_seconds": ((expected_total - completed) / throughput if throughput > 0 else None),
                   "workers_per_gpu": workers_per_gpu, "error": error,
                   "peak_gpu_memory_mib": peak_memory, "peak_host_ram_fraction": peak_ram,
                   "runtime_contract_sha256": sha256_file(contract_path)}
        write_json(root / "progress.json", payload)
        return payload

    with supervisor_lock(root):
        try:
            while any(queues.values()) or active:
                snapshot = resource_snapshot()
                peak_ram = max(peak_ram, float(snapshot["host_ram_fraction"]))
                for gpu, card in snapshot["gpu"].items():
                    peak_memory[gpu] = max(peak_memory[gpu], float(card["memory_used_mib"]))
                with (root / "resource_samples.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
                enforce_resources(snapshot, max_gpu_gib, max_host_fraction)
                for key, (process, handle, job, descriptor, _slot) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    handle.close()
                    close_gpu_slot(descriptor)
                    del active[key]
                    if code != 0 or not completed_valid(job):
                        failed += 1
                        raise RuntimeError(f"Worker failed: {key}, exit={code}")
                    completed += 1
                for gpu in (0, 1):
                    count = sum(record[2].gpu == gpu for record in active.values())
                    available_jobs = len(queues[gpu])
                    while count < workers_per_gpu and queues[gpu] and available_jobs:
                        available_jobs -= 1
                        job = queues[gpu].popleft()
                        if completed_valid(job):
                            # An orphan inherited its slot after a killed
                            # supervisor and has now completed. Adopt its audit.
                            completed += 1
                            continue
                        if job_lock_held(job):
                            queues[gpu].append(job)
                            continue
                        acquired = acquire_gpu_slot(root, gpu, workers_per_gpu)
                        if acquired is None:
                            queues[gpu].appendleft(job)
                            break
                        descriptor, slot = acquired
                        state = read_json(job.status_path)
                        log = job.child.root / "logs" / f"{job.spec['job_id']}.attempt_{int(state.get('attempt', 0)) + 1:02d}.log"
                        handle = log.open("a", encoding="utf-8")
                        try:
                            python = str(configs[job.child.family]["runtime"]["python_executable"])
                            process = subprocess.Popen(
                                _worker_command(job, python, descriptor, resume=resume),
                                cwd=REPO_ROOT, env=_worker_environment(job, log),
                                stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True, pass_fds=(descriptor,))
                        except BaseException:
                            handle.close()
                            close_gpu_slot(descriptor)
                            raise
                        active[job.key] = (process, handle, job, descriptor, slot)
                        count += 1
                progress("running")
                if active or any(queues.values()):
                    time.sleep(poll_seconds)
            if completed != expected_total:
                raise RuntimeError(f"Incomplete queue: {completed}/{expected_total}")
            result = progress("completed")
        except BaseException as exc:
            _terminate(active)
            for _process, handle, _job, descriptor, _slot in active.values():
                handle.close()
                close_gpu_slot(descriptor)
            active.clear()
            progress("failed", error=f"{type(exc).__name__}: {exc}")
            raise
    return result


def _prepare_children(children: Sequence[Child], workers: int, *, benchmark: bool,
                      evidence: Mapping[str, Any] | None = None,
                      resume: bool = False) -> None:
    for child in children:
        config = _child_call(child, "load_config", child.config_path)
        if resume and (child.root / "registry/task_registry.json").is_file():
            registry = _registry(child)
            if registry.get("terminal_complete"):
                _child_call(child, "validate_root", child.root)
                continue
        _child_call(child, "prepare", config, child.root, resume=resume,
                    workers_per_gpu=workers,
                    num_epochs=1 if benchmark else int(config["training"]["num_epochs"]),
                    root_mode="benchmark" if benchmark else "formal",
                    benchmark_evidence=None if benchmark else evidence)


def _validate_real_updates(children: Sequence[Child]) -> None:
    for child in children:
        # Each wrapper installs the correct seed/family globals for the shared
        # validator, which checks finite metrics and actual G/D parameter updates.
        _child_call(child, "_validate_benchmark_updates", child.root)


def select_candidate(attempts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    passed = [row for row in attempts if row.get("status") == "passed"]
    if not passed:
        raise RuntimeError("No concurrency candidate passed the resource/update gate")
    return max(passed, key=lambda row: (float(row["throughput_jobs_per_second"]),
                                        -int(row["workers_per_gpu"])))


def plateau_reached(attempts: Sequence[Mapping[str, Any]], threshold: float = .03) -> bool:
    passed = [row for row in attempts if row.get("status") == "passed"]
    if len(passed) < 3:
        return False
    recent = passed[-3:]
    gains = [float(right["throughput_jobs_per_second"]) / float(left["throughput_jobs_per_second"]) - 1
             for left, right in zip(recent, recent[1:])]
    return all(gain < threshold for gain in gains)


def smoke(film_config: str | Path, pure_config: str | Path, global_root: str | Path,
          *, workspace: str | Path = DEFAULT_BENCHMARK_WORKSPACE,
          resume: bool = False, poll_seconds: float = 2) -> Path:
    root = resolve(global_root)
    smoke_root = resolve(workspace) / "smoke"
    children = _children_from_configs(film_config, pure_config, benchmark_root=smoke_root)
    _prepare_children(children, 1, benchmark=True, resume=resume)
    jobs = load_jobs(children)
    selected = []
    for family, fold in (("film", "f1_2023q1"), ("pure", "f2_2023q2")):
        matching = [job for job in jobs if job.child.family == family and
                    str(job.spec["fold"]) == fold and int(job.spec["seed"]) == 42 and
                    (family == "pure" or str(job.spec["arm"]) == "lp_matched")]
        if len(matching) != 1:
            raise ValueError(f"Smoke representative drift: {family}/{fold}")
        selected.append(matching[0])
    run = run_queue(children, smoke_root / "scheduler", 1, resume=resume,
                    poll_seconds=poll_seconds, allow_benchmark=True,
                    job_keys=[job.key for job in selected])
    # The audited worker verifies epoch-0 hashes and that both G/D parameters
    # changed. The complete candidate benchmarks add the finite-metrics audit.
    if not all(completed_valid(job) for job in selected):
        raise RuntimeError("Smoke did not produce two valid updated checkpoints")
    result = signed({**run, "kind": "rq12_direct_vega_smoke_v1", "status": "passed",
                     "children": [child_binding(child) for child in children],
                     "job_keys": [job.key for job in selected], "epoch_count": 1,
                     "formal_benchmark_gate": False,
                     "supervisor_code_sha256": sha256_file(__file__), "completed_at_utc": utc_now()})
    return write_json(root / "smoke_result.json", result)


def benchmark(film_config: str | Path, pure_config: str | Path,
              global_root: str | Path, *, workspace: str | Path = DEFAULT_BENCHMARK_WORKSPACE,
              resume: bool = False, candidates: Sequence[int] = CANDIDATES,
              fallback_candidates: Sequence[int] = FALLBACK_CANDIDATES,
              poll_seconds: float = 2) -> Path:
    root = resolve(global_root)
    workspace = resolve(workspace)
    result_path = root / "benchmark_result.json"
    source_configs = {"film": {"path": str(resolve(film_config)), "sha256": sha256_file(resolve(film_config))},
                      "pure": {"path": str(resolve(pure_config)), "sha256": sha256_file(resolve(pure_config))}}
    if result_path.exists():
        result = read_signed(result_path, kind=BENCHMARK_KIND)
        if result["source_configs"] != source_configs or result["supervisor_code_sha256"] != sha256_file(__file__):
            raise ValueError("Global benchmark source/code drift")
        verify_bindings(result["children"])
        return result_path
    if tuple(candidates) != CANDIDATES or tuple(fallback_candidates) != FALLBACK_CANDIDATES:
        raise ValueError("Benchmark candidate universe drift")
    attempts = []
    with supervisor_lock(root / "benchmark_control"):
        pending = deque(map(int, candidates))
        while pending:
            workers = pending.popleft()
            candidate_root = workspace / f"workers_{workers:02d}"
            children = _children_from_configs(film_config, pure_config, benchmark_root=candidate_root)
            attempt_path = candidate_root / "attempt.json"
            if attempt_path.exists():
                attempt = read_signed(attempt_path)
                if attempt.get("source_configs") != source_configs:
                    raise ValueError("Candidate source config drift")
                if attempt["status"] == "passed":
                    verify_bindings(attempt["children"])
                elif not resume:
                    raise RuntimeError("Interrupted benchmark requires --resume")
            else:
                _prepare_children(children, workers, benchmark=True, resume=resume)
                try:
                    run = run_queue(children, candidate_root / "scheduler", workers,
                                    resume=resume, poll_seconds=poll_seconds, allow_benchmark=True)
                    _validate_real_updates(children)
                    attempt = {**run, "status": "passed",
                               "children": [child_binding(child) for child in children]}
                except (ResourceLimitError, RuntimeError) as exc:
                    # A failed worker can indicate a correctness error rather
                    # than capacity. Do not silently treat non-OOM failures as
                    # permission to reduce concurrency.
                    resource_failure = isinstance(exc, ResourceLimitError)
                    if not resource_failure:
                        logs = "\n".join(path.read_text(errors="replace")[-8000:]
                                         for child in children for path in (child.root / "logs").glob("*.log"))
                        resource_failure = "out of memory" in logs.lower()
                    if not resource_failure:
                        raise
                    progress_path = candidate_root / "scheduler/progress.json"
                    attempt = {**(read_json(progress_path) if progress_path.exists() else {}),
                               "status": "resource_failed", "error": str(exc)}
                attempt = signed({**attempt, "kind": "rq12_direct_vega_candidate_v1",
                                  "workers_per_gpu": workers, "source_configs": source_configs,
                                  "supervisor_code_sha256": sha256_file(__file__), "completed_at_utc": utc_now()})
                write_json(attempt_path, attempt)
            attempts.append(attempt)
            write_json(root / "benchmark_progress.json", {"updated_at_utc": utc_now(),
                       "attempts": attempts, "pending_candidates": list(pending)})
            if attempt["status"] != "passed":
                if workers == int(candidates[0]) and len(attempts) == 1:
                    pending = deque(map(int, fallback_candidates))
                    continue
                if any(row.get("status") == "passed" for row in attempts):
                    break
                continue
            if workers in fallback_candidates:
                break
            # Exercise every mandatory candidate unless capacity fails. Only
            # after the final mandatory point may the throughput stopping rule
            # end the search; extend in +8 steps when the last gain is >=3%.
            passed = [row for row in attempts if row.get("status") == "passed"]
            if workers == int(candidates[-1]) and not pending:
                gain = (float(passed[-1]["throughput_jobs_per_second"]) /
                        float(passed[-2]["throughput_jobs_per_second"]) - 1)
                if gain >= .03:
                    pending.append(workers + 8)
            elif workers > int(candidates[-1]):
                if plateau_reached(attempts):
                    break
                pending.append(workers + 8)
        selected = select_candidate(attempts)
        evidence = signed({"schema_version": 1, "kind": BENCHMARK_KIND, "status": "passed",
                           "scheduler_mode": SCHEDULER_MODE, "selected_workers_per_gpu": selected["workers_per_gpu"],
                           "gpu_ids": [0, 1], "job_count": 200, "epoch_count": 1,
                           "selection_metric": "completed_jobs_per_wall_second",
                           "plateau_rule": "all_mandatory_then_extend_by_8_until_two_gains_below_3_percent",
                           "candidate_workers_per_gpu": list(candidates),
                           "fallback_workers_per_gpu": list(fallback_candidates),
                           "max_gpu_memory_gib": 20, "max_host_ram_fraction": .85,
                           "source_configs": source_configs, "children": selected["children"],
                           "attempts": attempts, "benchmark_workspace": str(workspace),
                           "supervisor_code_sha256": sha256_file(__file__), "completed_at_utc": utc_now()})
        write_json(result_path, evidence)
    return result_path


def user_concurrency_override(film_config: str | Path, pure_config: str | Path,
                              global_root: str | Path, *, workers_per_gpu: int,
                              workspace: str | Path = DEFAULT_BENCHMARK_WORKSPACE) -> Path:
    """Record the explicit user instruction; prior trials supply disk sizing only."""
    if workers_per_gpu not in OVERRIDE_INSTRUCTIONS:
        raise ValueError("The explicit user instructions authorize 20 or 40 workers per GPU")
    instruction = OVERRIDE_INSTRUCTIONS[workers_per_gpu]
    root = resolve(global_root)
    path = root / "concurrency_override.json"
    children = _children_from_configs(film_config, pure_config)
    sources = {child.family: {"path": str(child.config_path), "sha256": sha256_file(child.config_path)}
               for child in children}
    if path.exists():
        existing = read_signed(path, kind=OVERRIDE_KIND)
        if (existing.get("status") != "authorized_by_user" or
                existing.get("authorization") != "explicit_user_instruction" or
                existing.get("user_instruction") != instruction or
                existing.get("selected_workers_per_gpu") != workers_per_gpu or
                existing.get("source_configs") != sources or
                existing.get("supervisor_code_sha256") != sha256_file(__file__)):
            raise ValueError("Existing user concurrency override drift")
        return path
    reference_path = resolve(workspace) / "workers_08/attempt.json"
    reference = read_signed(reference_path, kind="rq12_direct_vega_candidate_v1")
    if (reference.get("status") != "passed" or reference.get("completed") != 200 or
            reference.get("failed") != 0 or reference.get("workers_per_gpu") != 8):
        raise ValueError("Disk projection requires the completed prior 8-worker matrix")
    _verify_disk_bindings(reference["children"])
    payload = signed({"schema_version": 1, "kind": OVERRIDE_KIND,
                      "status": "authorized_by_user", "authorization": "explicit_user_instruction",
                      "user_instruction": instruction,
                      "selected_workers_per_gpu": workers_per_gpu, "gpu_ids": [0, 1],
                      "job_count": 200, "scheduler_mode": SCHEDULER_MODE,
                      "source_configs": sources, "supervisor_code_sha256": sha256_file(__file__),
                      "performance_benchmark_passed": False, "benchmark_skipped": True,
                      "max_gpu_memory_gib": 20, "max_host_ram_fraction": .85,
                      "formal_children": [child.as_dict() for child in children],
                      "children": reference["children"],
                      "children_scope": "prior_8_worker_one_epoch_disk_projection_only",
                      "disk_projection_reference": {"path": str(reference_path),
                          "sha256": sha256_file(reference_path),
                          "supervisor_code_sha256": reference["supervisor_code_sha256"],
                          "training_code_reused": False},
                      "authorized_at_utc": utc_now()})
    return write_json(path, payload)


def _verify_disk_bindings(bindings: Sequence[Mapping[str, Any]]) -> None:
    # The prior supervisor/code snapshot is deliberately NOT a training gate.
    # Its unchanged data, weights and generated configs remain disk references.
    for child in bindings:
        for role, row in child["files"].items():
            if role == "code_manifest":
                continue
            if sha256_file(row["path"]) != row["sha256"]:
                raise ValueError(f"Disk reference {role} drift: {row['path']}")
            if role.endswith("_manifest"):
                with Path(row["path"]).open(newline="", encoding="utf-8") as stream:
                    for entry in csv.DictReader(stream):
                        if sha256_file(entry["path"]) != entry["sha256"]:
                            raise ValueError(f"Disk reference member drift: {entry['path']}")


def _formal_evidence(root: Path) -> dict[str, Any]:
    override = root / "concurrency_override.json"
    return read_signed(override, kind=OVERRIDE_KIND) if override.exists() else read_signed(
        root / "benchmark_result.json", kind=BENCHMARK_KIND)


def prepare_global(film_config: str | Path, pure_config: str | Path,
                   global_root: str | Path, *, resume: bool = False) -> list[Child]:
    root = resolve(global_root)
    evidence = _formal_evidence(root)
    override = evidence["kind"] == OVERRIDE_KIND
    if (evidence.get("status") != ("authorized_by_user" if override else "passed") or
            evidence["supervisor_code_sha256"] != sha256_file(__file__)):
        raise ValueError("Global training authorization is not current")
    if override and (evidence.get("authorization") != "explicit_user_instruction" or
                     evidence.get("selected_workers_per_gpu") not in OVERRIDE_INSTRUCTIONS or
                     evidence.get("user_instruction") != OVERRIDE_INSTRUCTIONS.get(
                         evidence.get("selected_workers_per_gpu"))):
        raise ValueError("User concurrency authorization drift")
    children = _children_from_configs(film_config, pure_config)
    expected = {child.family: {"path": str(child.config_path), "sha256": sha256_file(child.config_path)}
                for child in children}
    if evidence["source_configs"] != expected:
        raise ValueError("Formal source configs differ from training authorization")
    if override:
        reference = evidence["disk_projection_reference"]
        if sha256_file(reference["path"]) != reference["sha256"]:
            raise ValueError("Disk projection reference evidence drift")
        _verify_disk_bindings(evidence["children"])
    else:
        verify_bindings(evidence["children"])
    formal_disk_gate(children, evidence, root)
    _prepare_children(children, int(evidence["selected_workers_per_gpu"]),
                      benchmark=False, evidence=evidence, resume=resume)
    return children


def formal_disk_gate(children: Sequence[Child], evidence: Mapping[str, Any],
                     global_root: Path) -> dict[str, Any]:
    """Project checkpoint artifacts from the actual selected one-epoch matrix."""
    bindings = {str(row["family"]): row for row in evidence["children"]}
    rows = []
    groups: dict[int, dict[str, Any]] = {}
    for child in children:
        config = yaml.safe_load(child.config_path.read_text(encoding="utf-8"))
        runtime = config["runtime"]
        safety = float(runtime.get("disk_projection_safety_factor", 1.3))
        minimum_gib = float(runtime.get("minimum_free_disk_after_projected_bytes_gib", 30))
        if safety < 1.3 or minimum_gib < 30:
            raise ValueError("Disk admission requires >=1.3 safety and >=30 GiB remaining")
        benchmark_root = Path(bindings[child.family]["root"])
        files = {path.resolve() for path in benchmark_root.rglob("*") if path.is_file()}
        registry = read_json(benchmark_root / "registry/task_registry.json")
        for spec in registry["jobs"]:
            state = read_json(benchmark_root / "registry/job_status" / f"{spec['job_id']}.json")
            files.update(Path(row["path"]).resolve() for row in state.get("artifacts", []))
        actual_bytes = sum(path.stat().st_size for path in files)
        remaining = EXPECTED_COUNTS[child.family]
        registry_path = child.root / "registry/task_registry.json"
        if registry_path.exists():
            existing = _registry(child)
            remaining -= sum(completed_valid(Job(child, spec)) for spec in existing["jobs"])
        projected = math.ceil(actual_bytes * safety * remaining / EXPECTED_COUNTS[child.family])
        existing_path = child.root
        while not existing_path.exists():
            existing_path = existing_path.parent
        device = existing_path.stat().st_dev
        group = groups.setdefault(device, {"path": str(existing_path), "projected_bytes": 0,
                                            "minimum_remaining_bytes": 0})
        group["projected_bytes"] += projected
        group["minimum_remaining_bytes"] = max(group["minimum_remaining_bytes"], int(minimum_gib * 1024**3))
        rows.append({"family": child.family, "benchmark_bytes": actual_bytes,
                     "remaining_jobs": remaining, "expected_jobs": EXPECTED_COUNTS[child.family],
                     "projected_bytes": projected, "safety_factor": safety})
    for group in groups.values():
        group["free_bytes"] = shutil.disk_usage(group["path"]).free
        if group["free_bytes"] - group["projected_bytes"] < group["minimum_remaining_bytes"]:
            raise ResourceLimitError("Formal disk admission would leave less than 30 GiB")
    admission = signed({"kind": "rq12_direct_vega_disk_admission_v1", "status": "passed",
                        "children": rows, "filesystems": list(groups.values()),
                        "authorization_kind": evidence["kind"],
                        "disk_projection_reference": evidence.get("disk_projection_reference"),
                        "checked_at_utc": utc_now()})
    write_json(global_root / "disk_admission.json", admission)
    return admission


def export_shared_resources(children: Sequence[Child], global_root: Path) -> None:
    """Export physical-card totals with explicit shared-family attribution."""
    samples_path = global_root / "resource_samples.jsonl"
    if not samples_path.is_file():
        raise ValueError("Completed shared scheduler has no resource telemetry")
    samples_hash = sha256_file(samples_path)
    rows = []
    fields = ("timestamp_utc", "wave", "gpu_index", "gpu_uuid", "gpu_name",
              "utilization_gpu_pct", "utilization_memory_pct", "memory_used_mib",
              "memory_total_mib", "power_draw_w", "temperature_c", "sample_status", "error",
              "measurement_scope")
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        if set(sample["gpu"]) != {"0", "1"}:
            raise ValueError("Shared resource sample must cover both physical cards")
        for gpu, card in sorted(sample["gpu"].items()):
            rows.append({"timestamp_utc": sample["at_utc"], "wave": 1, "gpu_index": int(gpu),
                         "utilization_gpu_pct": card["utilization_gpu_percent"],
                         "memory_used_mib": card["memory_used_mib"], "sample_status": "ok",
                         "error": "", "measurement_scope": "shared_film_and_pure_physical_gpu_total"})
    if not rows:
        raise ValueError("Shared resource telemetry is empty")
    for child in children:
        destination = child.root / "resource_usage.csv"
        provenance_path = child.root / "resource_usage_provenance.json"
        if provenance_path.exists():
            old = read_signed(provenance_path, kind="rq12_direct_vega_shared_resource_export_v1")
            if (old["source_sha256"] != samples_hash or old["resource_usage_sha256"] != sha256_file(destination)):
                raise ValueError("Shared resource export drift")
            continue
        if _registry(child).get("terminal_complete"):
            raise ValueError("Cannot add resource telemetry after terminal output freeze")
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
        write_json(provenance_path, signed({"kind": "rq12_direct_vega_shared_resource_export_v1",
                   "source_path": str(samples_path), "source_sha256": samples_hash,
                   "resource_usage_path": str(destination), "resource_usage_sha256": sha256_file(destination),
                   "row_count": len(rows), "gpu_ids": [0, 1],
                   "measurement_scope": "physical_gpu_totals_for_shared_film_and_pure_scheduler",
                   "per_model_memory_attribution": False, "wave_meaning": "single_shared_scheduler_matrix",
                   "sample_interval_seconds": 2, "exported_at_utc": utc_now()}))


def finalize_global(children: Sequence[Child], global_root: str | Path, *,
                    composite_config: str | Path = DEFAULT_COMPOSITE_CONFIG) -> Path:
    root = resolve(global_root)
    jobs = load_jobs(children, training=False)
    if not all(completed_valid(job) for job in jobs):
        raise RuntimeError("All 200 jobs must complete before checkpoint freeze")
    export_shared_resources(children, root)
    # The audited launch skips completed jobs, updates child bookkeeping and
    # provides an independent completion audit. It launches no new worker here.
    for child in children:
        if not _registry(child).get("evaluation_frozen"):
            _child_call(child, "launch", child.root, resume=True)
    for child in children:
        if not _registry(child).get("evaluation_frozen"):
            _child_call(child, "freeze_evaluation", child.root, resume=True)
    for child in children:
        if not _registry(child).get("predictions_frozen"):
            _child_call(child, "predict", child.root, resume=True)
        if not _registry(child).get("terminal_complete"):
            _child_call(child, "postprocess", child.root, resume=True)
        _child_call(child, "qa", child.root, resume=True)
    from scripts.rq3 import news_first_vol_text_10seed_composite as composite
    from scripts.rq3.rq12_vega_rescore import run_rescore

    composite.run_pipeline(resolve(composite_config), root / "composite", resume=True)
    run_rescore(root, controls_only=False)
    write_json(root / "finalize.json", signed({"kind": "rq12_direct_vega_finalize_v1",
               "status": "completed", "children": [child.as_dict() for child in children],
               "completed_at_utc": utc_now()}))
    return root


def _slot_worker(args: argparse.Namespace) -> None:
    if args.child_module not in MODULES.values() or args.slot_fd < 0:
        raise ValueError("Unknown child worker or missing inherited slot descriptor")
    os.fstat(args.slot_fd)
    try:
        family = next(family for family, module in MODULES.items() if module == args.child_module)
        child_root = resolve(args.child_root)
        child = Child(family, child_root, child_root / "resolved_config.yaml", args.child_module)
        _child_call(child, "worker", child_root, args.job_id, resume=args.resume)
    finally:
        os.close(args.slot_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "benchmark", "prepare", "launch", "run", "status", "finalize", "slot-worker"))
    parser.add_argument("--film-config", default=str(DEFAULT_FILM_CONFIG))
    parser.add_argument("--pure-config", default=str(DEFAULT_PURE_CONFIG))
    parser.add_argument("--composite-config", default=str(DEFAULT_COMPOSITE_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_ROOT))
    parser.add_argument("--benchmark-workspace", default=str(DEFAULT_BENCHMARK_WORKSPACE))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--workers-per-gpu", type=int)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--child-module", default="")
    parser.add_argument("--child-root", default="")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--slot-fd", type=int, default=-1)
    args = parser.parse_args(argv)
    root = resolve(args.output_dir)
    if args.action == "slot-worker":
        _slot_worker(args)
        return 0
    if args.skip_benchmark and (args.action not in {"prepare", "launch", "run"} or
                                args.workers_per_gpu not in OVERRIDE_INSTRUCTIONS):
        raise ValueError("--skip-benchmark requires prepare/launch/run --workers-per-gpu 20 or 40")
    if args.workers_per_gpu is not None and not args.skip_benchmark:
        raise ValueError("--workers-per-gpu requires explicit --skip-benchmark")
    if args.detach:
        if args.action not in {"smoke", "benchmark", "launch", "run", "finalize"}:
            raise ValueError("--detach is only available for sustained work")
        root.mkdir(parents=True, exist_ok=True)
        command_args = list(sys.argv[1:] if argv is None else argv)
        command_args.remove("--detach")
        with (root / "supervisor.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "scripts.rq3.rq12_direct_vega_parallel", *command_args],
                                       cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        write_json(root / "supervisor_process.json", {"pid": process.pid, "started_at_utc": utc_now(),
                   "command": command_args})
        print(json.dumps({"pid": process.pid, "log": str(root / "supervisor.log")}))
        return 0
    if args.action == "status":
        paths = (root / "progress.json", root / "benchmark_progress.json", root / "finalize.json")
        print(json.dumps({path.name: read_json(path) for path in paths if path.exists()}, indent=2))
        return 0
    if args.action == "smoke":
        result = smoke(args.film_config, args.pure_config, root,
                       workspace=args.benchmark_workspace, resume=args.resume,
                       poll_seconds=args.poll_seconds)
        print(str(result))
        return 0
    if args.skip_benchmark:
        user_concurrency_override(args.film_config, args.pure_config, root,
                                  workers_per_gpu=args.workers_per_gpu, workspace=args.benchmark_workspace)
    if args.action == "benchmark" or (args.action == "run" and not (root / "concurrency_override.json").exists()):
        benchmark(args.film_config, args.pure_config, root,
                  workspace=args.benchmark_workspace, resume=args.resume, poll_seconds=args.poll_seconds)
        if args.action == "benchmark":
            print(str(root / "benchmark_result.json"))
            return 0
    if args.action in {"prepare", "launch", "run"}:
        children = prepare_global(args.film_config, args.pure_config, root, resume=args.resume)
        if args.action == "prepare":
            print(json.dumps([child.as_dict() for child in children]))
            return 0
        evidence = _formal_evidence(root)
        if any(_registry(child).get("evaluation_frozen") for child in children):
            if not all(completed_valid(job) for job in load_jobs(children, training=False)):
                raise RuntimeError("Evaluation was frozen before the global matrix completed")
        else:
            run_queue(children, root, int(evidence["selected_workers_per_gpu"]),
                      resume=args.resume, poll_seconds=args.poll_seconds)
        if args.action == "launch":
            return 0
    children = _children_from_configs(args.film_config, args.pure_config)
    finalize_global(children, root, composite_config=args.composite_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
