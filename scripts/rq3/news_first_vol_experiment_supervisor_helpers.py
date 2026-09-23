"""Reusable resource and lifecycle helpers for RQ3 background supervisors.

This module contains no experiment-specific training logic.  In particular it
never kills, pauses, or reconfigures a process found on a GPU.  An external
vLLM (or any other unapproved CUDA process) makes the availability gate wait
rather than pre-empting that workload.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any


class SupervisorHelperError(RuntimeError):
    """Raised when a resource or supervisor contract cannot be trusted."""


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    import datetime as dt

    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace a JSON control artifact on the same filesystem."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(payload))
    os.replace(temporary, destination)
    return destination


def _signed_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    unsigned["payload_sha256"] = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    return unsigned


def _read_signed_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupervisorHelperError(f"Invalid JSON control artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise SupervisorHelperError(f"Control artifact must be a mapping: {path}")
    payload = dict(value)
    observed = str(payload.pop("payload_sha256", ""))
    expected = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if observed != expected:
        raise SupervisorHelperError(f"Control artifact hash drift: {path}")
    payload["payload_sha256"] = observed
    return payload


def parse_gpu_rows(text: str) -> list[dict[str, Any]]:
    """Parse the frozen nvidia-smi GPU query used by :func:`gpu_snapshot`."""

    rows: list[dict[str, Any]] = []
    for line_number, fields in enumerate(csv.reader(text.splitlines()), start=1):
        if not fields or all(not field.strip() for field in fields):
            continue
        if len(fields) != 6:
            raise SupervisorHelperError(
                f"Malformed GPU telemetry row {line_number}: {fields!r}"
            )
        try:
            index = int(fields[0].strip())
            memory_used = float(fields[3].strip())
            memory_total = float(fields[4].strip())
            utilization = float(fields[5].strip())
        except ValueError as exc:
            raise SupervisorHelperError(
                f"Non-numeric GPU telemetry row {line_number}"
            ) from exc
        values = (memory_used, memory_total, utilization)
        if not all(math.isfinite(value) for value in values) or (
            memory_used < 0 or memory_total <= 0 or not 0 <= utilization <= 100
        ):
            raise SupervisorHelperError(f"Invalid GPU telemetry row {line_number}")
        rows.append(
            {
                "gpu_index": index,
                "gpu_uuid": fields[1].strip(),
                "gpu_name": fields[2].strip(),
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
                "utilization_gpu_pct": utilization,
            }
        )
    return rows


def parse_compute_process_rows(
    text: str,
    *,
    command_lookup: Callable[[int], str] | None = None,
) -> list[dict[str, Any]]:
    """Parse CUDA compute processes and enrich them with ``/proc`` commands."""

    lookup = command_lookup or process_command
    rows: list[dict[str, Any]] = []
    for line_number, fields in enumerate(csv.reader(text.splitlines()), start=1):
        if not fields or all(not field.strip() for field in fields):
            continue
        # nvidia-smi reports this literal when there are no compute applications.
        if len(fields) == 1 and "no running" in fields[0].lower():
            continue
        if len(fields) != 4:
            raise SupervisorHelperError(
                f"Malformed compute-process row {line_number}: {fields!r}"
            )
        try:
            pid = int(fields[1].strip())
            used_mib = float(fields[3].strip())
        except ValueError as exc:
            raise SupervisorHelperError(
                f"Non-numeric compute-process row {line_number}"
            ) from exc
        command = lookup(pid)
        process_name = fields[2].strip()
        searchable = f"{process_name} {command}".lower()
        rows.append(
            {
                "gpu_uuid": fields[0].strip(),
                "pid": pid,
                "process_name": process_name,
                "used_memory_mib": used_mib,
                "command": command,
                "protected_vllm": "vllm" in searchable,
            }
        )
    return rows


def process_command(pid: int) -> str:
    """Read a process command without treating a disappearing PID as an error."""

    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(
        part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part
    )


def host_memory_snapshot(meminfo: str | Path = "/proc/meminfo") -> dict[str, Any]:
    """Return host memory totals and used fraction from Linux meminfo."""

    path = Path(meminfo)
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SupervisorHelperError(
            f"Cannot read host memory telemetry: {path}"
        ) from exc
    for line in lines:
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            try:
                values[key] = int(raw.strip().split()[0])
            except (IndexError, ValueError) as exc:
                raise SupervisorHelperError(f"Malformed {key} in {path}") from exc
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", -1)
    if total <= 0 or not 0 <= available <= total:
        raise SupervisorHelperError(f"Incomplete host memory telemetry: {path}")
    return {
        "memory_total_kib": total,
        "memory_available_kib": available,
        "memory_used_fraction": (total - available) / total,
    }


def _run_nvidia_smi(
    executable: str,
    query: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    result = runner(
        [executable, query, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if int(result.returncode) != 0:
        raise SupervisorHelperError(
            f"nvidia-smi failed: {str(result.stderr).strip() or result.returncode}"
        )
    return str(result.stdout)


def gpu_snapshot(
    *,
    executable: str = "nvidia-smi",
    command_lookup: Callable[[int], str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Collect GPU and compute-process telemetry without changing GPU state."""

    gpu_text = _run_nvidia_smi(
        executable,
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        runner=runner,
    )
    process_text = _run_nvidia_smi(
        executable,
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        runner=runner,
    )
    return {
        "timestamp_utc": utc_now(),
        "gpus": parse_gpu_rows(gpu_text),
        "compute_processes": parse_compute_process_rows(
            process_text, command_lookup=command_lookup
        ),
    }


def system_resource_snapshot(
    *,
    executable: str = "nvidia-smi",
    meminfo: str | Path = "/proc/meminfo",
    command_lookup: Callable[[int], str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Collect the GPU, host-memory, and lightweight CPU availability snapshot."""

    gpu = gpu_snapshot(
        executable=executable, command_lookup=command_lookup, runner=runner
    )
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except OSError:
        load_1m = load_5m = load_15m = math.nan
    return {
        **gpu,
        "host_memory": host_memory_snapshot(meminfo),
        "cpu": {
            "logical_cpu_count": os.cpu_count(),
            "load_average_1m": load_1m,
            "load_average_5m": load_5m,
            "load_average_15m": load_15m,
        },
    }


def gpu_availability_decision(
    snapshot: Mapping[str, Any],
    *,
    gpu_ids: Sequence[int] = (0, 1),
    allowed_pids: Sequence[int] = (),
    maximum_idle_memory_mib: float = 1024.0,
    maximum_idle_utilization_pct: float = 5.0,
) -> dict[str, Any]:
    """Decide whether all requested GPUs are idle and safe to claim.

    Every unapproved compute process blocks launch.  vLLM processes receive a
    separate protected classification so callers can explain why they are
    waiting.  This function never signals a process.
    """

    requested = tuple(map(int, gpu_ids))
    if len(requested) != 2 or len(set(requested)) != 2:
        raise SupervisorHelperError("The formal gate requires two distinct GPUs")
    gpu_rows = {int(row["gpu_index"]): dict(row) for row in snapshot.get("gpus", ())}
    missing = sorted(set(requested) - set(gpu_rows))
    allowed = set(map(int, allowed_pids)) | {os.getpid()}
    uuid_to_index = {str(row["gpu_uuid"]): index for index, row in gpu_rows.items()}
    external: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    for raw in snapshot.get("compute_processes", ()):
        row = dict(raw)
        gpu_index = uuid_to_index.get(str(row.get("gpu_uuid", "")))
        if gpu_index not in requested or int(row.get("pid", -1)) in allowed:
            continue
        row["gpu_index"] = gpu_index
        external.append(row)
        if bool(row.get("protected_vllm")):
            protected.append(row)
    memory_busy = [
        index
        for index in requested
        if index in gpu_rows
        and float(gpu_rows[index]["memory_used_mib"]) > float(maximum_idle_memory_mib)
    ]
    utilization_busy = [
        index
        for index in requested
        if index in gpu_rows
        and float(gpu_rows[index]["utilization_gpu_pct"])
        > float(maximum_idle_utilization_pct)
    ]
    reasons: list[str] = []
    if missing:
        reasons.append("requested_gpu_missing")
    if protected:
        reasons.append("protected_external_vllm")
    if external:
        reasons.append("unapproved_external_gpu_process")
    if memory_busy:
        reasons.append("gpu_memory_not_idle")
    if utilization_busy:
        reasons.append("gpu_utilization_not_idle")
    return {
        "ready": not reasons,
        "requested_gpu_ids": list(requested),
        "missing_gpu_ids": missing,
        "memory_busy_gpu_ids": memory_busy,
        "utilization_busy_gpu_ids": utilization_busy,
        "external_processes": external,
        "protected_vllm_processes": protected,
        "reasons": reasons,
        "policy": "wait_only_never_signal_or_preempt_external_processes",
    }


def wait_for_dual_gpu_availability(
    snapshot_provider: Callable[[], Mapping[str, Any]],
    *,
    gpu_ids: Sequence[int] = (0, 1),
    allowed_pids: Sequence[int] = (),
    maximum_idle_memory_mib: float = 1024.0,
    maximum_idle_utilization_pct: float = 5.0,
    timeout_seconds: float = 3600.0,
    poll_interval_seconds: float = 15.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait until both GPUs pass the non-preemptive availability gate."""

    if timeout_seconds < 0 or poll_interval_seconds <= 0:
        raise SupervisorHelperError("Invalid GPU wait timing")
    start = monotonic()
    attempts = 0
    last: dict[str, Any] | None = None
    while True:
        attempts += 1
        snapshot = snapshot_provider()
        last = gpu_availability_decision(
            snapshot,
            gpu_ids=gpu_ids,
            allowed_pids=allowed_pids,
            maximum_idle_memory_mib=maximum_idle_memory_mib,
            maximum_idle_utilization_pct=maximum_idle_utilization_pct,
        )
        elapsed = max(0.0, monotonic() - start)
        if last["ready"]:
            return {**last, "attempts": attempts, "waited_seconds": elapsed}
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                "Timed out waiting for two safe GPUs; "
                f"reasons={last['reasons']}, protected_vllm={len(last['protected_vllm_processes'])}"
            )
        sleeper(min(float(poll_interval_seconds), float(timeout_seconds) - elapsed))


def evaluate_canary_or_benchmark(
    evidence: Mapping[str, Any],
    *,
    expected_jobs: int,
    gpu_ids: Sequence[int] = (0, 1),
    maximum_peak_gpu_memory_gib: float = 20.0,
    maximum_host_ram_fraction: float = 0.85,
) -> dict[str, Any]:
    """Apply the shared correctness and capacity gates to run evidence."""

    peaks = {
        int(key): float(value)
        for key, value in dict(evidence.get("peak_gpu_memory_gib_by_gpu", {})).items()
    }
    expected_gpus = set(map(int, gpu_ids))
    completed_jobs = int(evidence.get("completed_jobs", -1))
    failed_jobs = int(evidence.get("failed_jobs", 0))
    oom_count = int(evidence.get("oom_count", 0))
    complete = completed_jobs == int(expected_jobs)
    telemetry_complete = set(peaks) == expected_gpus and all(
        math.isfinite(value) and value >= 0.0 for value in peaks.values()
    )
    correctness_reasons: list[str] = []
    if completed_jobs + failed_jobs != int(expected_jobs):
        correctness_reasons.append("incomplete_job_accounting")
    elif not complete and oom_count == 0:
        correctness_reasons.append("incomplete_jobs")
    if failed_jobs > oom_count:
        correctness_reasons.append("noncapacity_failed_jobs")
    if int(evidence.get("nan_count", 0)) != 0:
        correctness_reasons.append("nan")
    if oom_count == 0 and int(evidence.get("generator_updated_jobs", -1)) != int(
        expected_jobs
    ):
        correctness_reasons.append("generator_not_updated")
    if oom_count == 0 and int(evidence.get("critic_updated_jobs", -1)) != int(
        expected_jobs
    ):
        correctness_reasons.append("critic_not_updated")
    if not telemetry_complete:
        correctness_reasons.append("incomplete_telemetry")
    capacity_reasons: list[str] = []
    if oom_count:
        capacity_reasons.append("oom")
    if telemetry_complete and max(peaks.values()) >= float(maximum_peak_gpu_memory_gib):
        capacity_reasons.append("gpu_memory_cap")
    host_fraction = float(evidence.get("peak_host_ram_fraction", math.inf))
    if not math.isfinite(host_fraction):
        correctness_reasons.append("invalid_host_telemetry")
    elif host_fraction >= float(maximum_host_ram_fraction):
        capacity_reasons.append("host_ram_cap")
    passed = not correctness_reasons and not capacity_reasons
    return {
        "passed": passed,
        "failure_category": (
            "none" if passed else "correctness" if correctness_reasons else "capacity"
        ),
        "correctness_reasons": correctness_reasons,
        "capacity_reasons": capacity_reasons,
        "expected_jobs": int(expected_jobs),
        "peak_gpu_memory_gib_by_gpu": peaks,
        "peak_host_ram_fraction": host_fraction,
        "gpu_limit_gib_exclusive": float(maximum_peak_gpu_memory_gib),
        "host_ram_limit_fraction_exclusive": float(maximum_host_ram_fraction),
    }


def benchmark_concurrency_decision(
    primary: Mapping[str, Any],
    *,
    primary_workers_per_gpu: int,
    fallback_workers_per_gpu: int,
    fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select concurrency; only a capacity failure may enter fallback."""

    if bool(primary.get("passed")):
        return {
            "action": "launch_formal",
            "selected_workers_per_gpu": int(primary_workers_per_gpu),
            "decision_reason": "primary_benchmark_passed",
        }
    if str(primary.get("failure_category")) != "capacity":
        return {
            "action": "abort",
            "selected_workers_per_gpu": None,
            "decision_reason": "primary_noncapacity_failure",
        }
    if fallback is None:
        return {
            "action": "run_fallback_benchmark",
            "selected_workers_per_gpu": None,
            "decision_reason": "primary_capacity_failure",
        }
    if bool(fallback.get("passed")):
        return {
            "action": "launch_formal",
            "selected_workers_per_gpu": int(fallback_workers_per_gpu),
            "decision_reason": "fallback_benchmark_passed",
        }
    return {
        "action": "abort",
        "selected_workers_per_gpu": None,
        "decision_reason": "fallback_benchmark_failed",
    }


def process_start_ticks(pid: int) -> int | None:
    """Read Linux process start ticks, guarding against PID reuse."""

    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    closing = raw.rfind(")")
    if closing < 0:
        return None
    fields_after_command = raw[closing + 2 :].split()
    # Field 22 overall; fields_after_command starts at field 3.
    try:
        return int(fields_after_command[19])
    except (IndexError, ValueError):
        return None


def pid_alive(pid: int, *, expected_start_ticks: int | None = None) -> bool:
    """Check liveness and optionally reject a reused PID."""

    try:
        os.kill(int(pid), 0)
    except (OSError, ProcessLookupError):
        return False
    return expected_start_ticks is None or process_start_ticks(int(pid)) == int(
        expected_start_ticks
    )


@dataclass
class SupervisorLock:
    """Non-blocking advisory lock with a hash-bound PID identity artifact."""

    control_dir: Path
    name: str = "pipeline"
    descriptor: int | None = None

    @property
    def lock_path(self) -> Path:
        return self.control_dir.resolve() / f"{self.name}.lock"

    @property
    def pid_path(self) -> Path:
        return self.control_dir.resolve() / f"{self.name}.pid.json"

    def acquire(self) -> "SupervisorLock":
        if self.descriptor is not None:
            raise SupervisorHelperError("Supervisor lock is already acquired")
        self.control_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise SupervisorHelperError(
                f"Live supervisor lock exists: {self.lock_path}"
            ) from exc
        self.descriptor = descriptor
        try:
            identity = _signed_payload(
                {
                    "schema_version": 1,
                    "kind": "rq3_supervisor_pid_v1",
                    "pid": os.getpid(),
                    "process_start_ticks": process_start_ticks(os.getpid()),
                    "acquired_at_utc": utc_now(),
                    "lock_path": str(self.lock_path),
                }
            )
            atomic_write_json(self.pid_path, identity)
        except BaseException:
            self.release()
            raise
        return self

    def release(self) -> None:
        if self.descriptor is None:
            return
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)
            self.descriptor = None

    def __enter__(self) -> "SupervisorLock":
        return self.acquire()

    def __exit__(self, *_args: object) -> None:
        self.release()


def append_stage_journal(
    control_dir: str | Path,
    *,
    experiment_kind: str,
    stage: str,
    status: str,
    details: Mapping[str, Any] | None = None,
    name: str = "pipeline_journal.json",
) -> Path:
    """Append one hash-bound stage event and publish the journal atomically."""

    path = Path(control_dir).resolve() / name
    events: list[dict[str, Any]] = []
    if path.is_file():
        existing = _read_signed_json(path)
        if str(existing.get("experiment_kind")) != str(experiment_kind):
            raise SupervisorHelperError("Journal experiment kind drift")
        raw_events = existing.get("events")
        if not isinstance(raw_events, list):
            raise SupervisorHelperError("Journal events are malformed")
        events = [dict(event) for event in raw_events]
    event = {
        "sequence": len(events),
        "stage": str(stage),
        "status": str(status),
        "pid": os.getpid(),
        "process_start_ticks": process_start_ticks(os.getpid()),
        "updated_at_utc": utc_now(),
        "details": dict(details or {}),
    }
    events.append(event)
    payload = _signed_payload(
        {
            "schema_version": 1,
            "kind": "rq3_supervisor_stage_journal_v1",
            "experiment_kind": str(experiment_kind),
            "events": events,
            "latest": event,
        }
    )
    return atomic_write_json(path, payload)


def advisory_lock_is_held(path: str | Path) -> bool:
    """Probe an existing advisory lock without retaining or modifying it."""

    lock_path = Path(path).resolve()
    if not lock_path.is_file():
        return False
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def assess_resume_state(
    output_root: str | Path,
    control_dir: str | Path,
    *,
    terminal_marker: str | Path,
    terminal_validator: Callable[[Path], None] | None = None,
    partial_validator: Callable[[Path], None] | None = None,
    pid_name: str = "pipeline.pid.json",
    lock_name: str = "pipeline.lock",
    journal_name: str = "pipeline_journal.json",
) -> dict[str, Any]:
    """Classify new, live, safely resumable, and terminal read-only states."""

    root = Path(output_root).resolve()
    control = Path(control_dir).resolve()
    marker = root / terminal_marker
    if root.exists() and marker.is_file():
        if terminal_validator is None:
            return {
                "state": "invalid_terminal",
                "reason": "terminal_validator_required",
            }
        try:
            terminal_validator(root)
        except Exception as exc:  # fail-closed boundary supplied by caller
            return {
                "state": "invalid_terminal",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        return {
            "state": "terminal_read_only",
            "may_write": False,
            "output_root": str(root),
        }

    pid_path = control / pid_name
    lock_held = advisory_lock_is_held(control / lock_name)
    if lock_held and pid_path.is_file():
        identity = _read_signed_json(pid_path)
        pid = int(identity.get("pid", -1))
        ticks = identity.get("process_start_ticks")
        if pid > 0 and pid_alive(
            pid, expected_start_ticks=None if ticks is None else int(ticks)
        ):
            return {
                "state": "blocked_live_supervisor",
                "may_write": False,
                "pid": pid,
            }
    if lock_held:
        return {
            "state": "blocked_live_supervisor",
            "may_write": False,
            "pid": None,
            "reason": "held_lock_without_live_hash_bound_pid",
        }

    if not root.exists():
        return {"state": "new", "may_write": True, "output_root": str(root)}
    if not root.is_dir():
        return {"state": "invalid_partial", "reason": "output_root_is_not_directory"}
    if partial_validator is None:
        return {"state": "invalid_partial", "reason": "partial_validator_required"}
    try:
        partial_validator(root)
    except Exception as exc:  # caller owns the experiment-specific contract
        return {"state": "invalid_partial", "reason": f"{type(exc).__name__}: {exc}"}
    journal_path = control / journal_name
    latest: Mapping[str, Any] = {}
    if journal_path.is_file():
        latest = dict(_read_signed_json(journal_path).get("latest") or {})
    return {
        "state": "resumable_partial",
        "may_write": True,
        "last_stage": latest.get("stage"),
        "last_status": latest.get("status"),
        "output_root": str(root),
    }


__all__ = [
    "SupervisorHelperError",
    "SupervisorLock",
    "advisory_lock_is_held",
    "append_stage_journal",
    "assess_resume_state",
    "atomic_write_json",
    "benchmark_concurrency_decision",
    "evaluate_canary_or_benchmark",
    "gpu_availability_decision",
    "gpu_snapshot",
    "host_memory_snapshot",
    "parse_compute_process_rows",
    "parse_gpu_rows",
    "pid_alive",
    "process_command",
    "process_start_ticks",
    "system_resource_snapshot",
    "utc_now",
    "wait_for_dual_gpu_availability",
]
