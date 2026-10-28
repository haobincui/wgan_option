"""Read-only pipeline monitor; atomically publish main-root/live_status.json."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.rq3.rq12_direct_vega_parallel import (
    DEFAULT_BENCHMARK_WORKSPACE, DEFAULT_ROOT, resource_snapshot,
)

EXPECTED_JOBS = 200
DEATH_GRACE_SECONDS = 30.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_identity(pid: Any, root: Path, module: str) -> dict[str, Any]:
    """Reject PID reuse by checking the actual module and output-directory args."""
    if not isinstance(pid, int) or pid <= 0:
        return {"pid": None, "alive": None, "cmdline_verified": False, "reason": "pid_unavailable", "command": None}
    proc = Path("/proc") / str(pid)
    try:
        command = [part.decode(errors="replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
        state = (proc / "stat").read_text().rsplit(")", 1)[1].strip().split()[0]
        cwd = (proc / "cwd").resolve(strict=True)
    except FileNotFoundError:
        return {"pid": pid, "alive": False, "cmdline_verified": False, "reason": "process_missing", "command": None}
    except OSError as exc:
        return {"pid": pid, "alive": None, "cmdline_verified": False, "reason": f"proc_read_error: {exc}", "command": None}
    expected_script = module.rsplit(".", 1)[-1] + ".py"
    module_matches = module in command or any(Path(arg).name == expected_script for arg in command)
    directory_flag = "--output-dir" if module.endswith("rq12_direct_vega_parallel") else "--main-root"
    try:
        directory = Path(command[command.index(directory_flag) + 1])
        bound_root = (directory if directory.is_absolute() else cwd / directory).resolve()
    except (ValueError, IndexError):
        bound_root = None
    verified = module_matches and bound_root == root
    if module.endswith("rq12_direct_vega_parallel"):
        verified = verified and "run" in command
    alive = bool(command) and state not in {"Z", "X"} and verified
    return {"pid": pid, "alive": alive, "cmdline_verified": verified, "reason": None if alive else "process_not_running_or_cmdline_mismatch", "command": command}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def queue_summary(progress: dict[str, Any] | None, *, not_started: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "expected": EXPECTED_JOBS, "status": "not_started" if not_started else None,
        "completed": 0 if not_started else None, "failed": 0 if not_started else None,
        "running": 0 if not_started else None, "pending": EXPECTED_JOBS if not_started else None,
        "jobs_per_hour": None, "eta_seconds": None, "error": None,
        "updated_at_utc": None, "elapsed_seconds": None,
    }
    if progress is None:
        return result
    result["status"] = progress.get("status")
    result["expected"] = progress.get("expected", EXPECTED_JOBS)
    for field in ("completed", "failed", "running", "pending"):
        value = _number(progress.get(field))
        result[field] = int(value) if value is not None and value.is_integer() else None
    throughput = _number(progress.get("throughput_jobs_per_second"))
    result["jobs_per_hour"] = throughput * 3600 if throughput is not None and throughput > 0 else None
    result["eta_seconds"] = _number(progress.get("eta_seconds"))
    result["elapsed_seconds"] = _number(progress.get("elapsed_seconds"))
    result["updated_at_utc"] = progress.get("updated_at_utc")
    result["error"] = progress.get("error") or None
    if result["status"] == "completed":
        result["eta_seconds"] = 0.0
    return result


def benchmark_candidate(workspace: Path, benchmark_progress: dict[str, Any] | None) -> dict[str, Any]:
    directories = list(workspace.glob("workers_[0-9]*")) if workspace.exists() else []
    directories = [path for path in directories if path.is_dir() and path.name[8:].isdigit()]
    candidate = max(directories, key=lambda path: path.stat().st_mtime_ns) if directories else None
    progress_path = candidate / "scheduler/progress.json" if candidate else None
    progress = read_json(progress_path) if progress_path else None
    pending = (benchmark_progress or {}).get("pending_candidates", [])
    # The completed candidate is no longer active when the supervisor starts
    # preparing the next capacity. Its 200/200 must not describe that next run.
    if pending and (candidate is None or (progress or {}).get("status") in {"completed", "failed"}):
        candidate = workspace / f"workers_{int(pending[0]):02d}"
        progress_path = candidate / "scheduler/progress.json"
        progress = read_json(progress_path)
    capacity = int(candidate.name[8:]) if candidate else None
    result = queue_summary(progress)
    result.update({"workers_per_gpu": capacity, "capacity_total_two_gpus": capacity * 2 if capacity is not None else None,
                   "stage": progress.get("status") if progress else "preparing",
                   "progress_path": str(progress_path) if progress_path else None,
                   "candidate_root": str(candidate) if candidate else None})
    return result


def select_phase(*, finalized: bool, supervisor_alive: bool | None, formal: Mapping[str, Any], benchmark_passed: bool, candidate: Mapping[str, Any], concurrency_configured: bool = False, awaiting_new_supervisor: bool = False) -> str:
    if finalized:
        return "completed"
    if awaiting_new_supervisor:
        return "formal_preparation"
    if supervisor_alive is False:
        return "supervisor_exited"
    if formal.get("status") == "failed":
        return "training_failed"
    if formal.get("status") == "completed":
        return "finalizing_predictions_and_analysis"
    if formal.get("status") == "running":
        return "formal_training"
    if concurrency_configured or benchmark_passed:
        return "formal_preparation"
    if candidate.get("stage") == "running":
        return "benchmark_training"
    if candidate.get("workers_per_gpu") is not None:
        return "preparing_benchmark" if candidate.get("stage") == "preparing" else "benchmark_capacity_selection"
    return "preparing_benchmark" if supervisor_alive else "unknown"


def collect_status(root: Path) -> dict[str, Any]:
    process_path = root / "supervisor_process.json"
    process = read_json(process_path) or {}
    supervisor = process_identity(process.get("pid"), root, "scripts.rq3.rq12_direct_vega_parallel")
    benchmark = read_json(root / "benchmark_result.json")
    benchmark_progress = read_json(root / "benchmark_progress.json")
    override_path = root / "concurrency_override.json"
    override = read_json(override_path)
    override_workers = _number((override or {}).get("selected_workers_per_gpu"))
    override_accepted = bool(override and override.get("kind") == "rq12_direct_vega_user_concurrency_override_v1"
                             and override.get("status") == "authorized_by_user"
                             and override_workers is not None and override_workers > 0 and override_workers.is_integer())
    command = supervisor.get("command") or []
    workspace = Path((benchmark or {}).get("benchmark_workspace", DEFAULT_BENCHMARK_WORKSPACE)).resolve()
    if "--benchmark-workspace" in command:
        value = Path(command[command.index("--benchmark-workspace") + 1])
        workspace = (value if value.is_absolute() else (Path("/proc") / str(supervisor["pid"]) / "cwd").resolve() / value).resolve()
    progress = read_json(root / "progress.json")
    formal = queue_summary(progress, not_started=progress is None and (supervisor.get("alive") is True or override_accepted))
    candidate = benchmark_candidate(workspace, benchmark_progress)
    passed = (benchmark or {}).get("status") == "passed"
    if override_accepted:
        candidate.update({"raw_scheduler_status": candidate.get("status"),
                          "raw_scheduler_error": candidate.get("error"),
                          "raw_scheduler_progress": read_json(Path(candidate["progress_path"])) if candidate.get("progress_path") else None,
                          "stage": "stopped_by_user", "status": "stopped_by_user", "history_only": True,
                          "stop_reason": "user_requested_direct_formal_training", "error": None, "jobs_per_hour": None, "eta_seconds": None})
    elif passed:
        candidate["stage"] = "selection_completed"
    configured = override_accepted or passed
    # The override is written after the old benchmark process was stopped and
    # before its replacement is announced. Do not call that handover a failure.
    announced_command = supervisor.get("command") or process.get("command") or []
    formal_launch_announced = "--skip-benchmark" in announced_command
    awaiting_new = bool(override_accepted and not formal_launch_announced and progress is None and supervisor.get("alive") is not True
                        and (not process_path.exists() or process_path.stat().st_mtime_ns < override_path.stat().st_mtime_ns))
    supervisor["awaiting_new_formal_launch"] = awaiting_new
    finalize = read_json(root / "finalize.json")
    finalized = (finalize or {}).get("status") == "completed"
    phase = select_phase(finalized=finalized, supervisor_alive=supervisor.get("alive"), formal=formal, benchmark_passed=passed, candidate=candidate,
                         concurrency_configured=configured, awaiting_new_supervisor=awaiting_new)
    try:
        resources = resource_snapshot()
        resource_error = None
    except Exception as exc:
        resources, resource_error = None, f"{type(exc).__name__}: {exc}"
    final_error = (finalize or {}).get("error") or None
    finalize_status = (finalize or {}).get("status") or ("running" if formal.get("status") == "completed" and supervisor.get("alive") else "pending")
    if not finalized and supervisor.get("alive") is False and not awaiting_new:
        # /proc cannot establish an exit code or the cause of a missing process.
        # Preserve observed completions, but do not present stale ETA as live.
        if finalize is None:
            finalize_status = None
        for queue in (formal, candidate):
            queue["jobs_per_hour"] = None
            queue["eta_seconds"] = None
            if queue.get("status") == "running":
                queue["status"] = "last_observed_running_supervisor_exited"
                queue["running"] = None
                queue["pending"] = None
    return {"schema_version": 1, "updated_at_utc": utc_now(), "main_root": str(root), "phase": phase,
            "supervisor": supervisor, "formal_200_jobs": formal,
            "concurrency": {"concurrency_configured": configured, "source": "user_instruction" if override_accepted else ("benchmark" if passed else None),
                            "selected_workers_per_gpu": int(override_workers) if override_accepted else (benchmark or {}).get("selected_workers_per_gpu"),
                            "benchmark_passed": passed, "override_accepted": override_accepted,
                            "override_path": str(override_path) if override_accepted else None},
            "benchmark": {"workspace": str(workspace), "selection_passed": passed,
                          "selected_workers_per_gpu": (benchmark or {}).get("selected_workers_per_gpu"),
                          "completed_candidate_count": len((benchmark_progress or {}).get("attempts", [])),
                          "current_candidate": candidate},
            "resources": resources, "resource_error": resource_error,
            "finalize": {"completed": finalized, "status": finalize_status, "error": final_error,
                         "completed_at_utc": (finalize or {}).get("completed_at_utc")},
            "monitor": {"pid": os.getpid(), "status": "running", "supervisor_death_grace_seconds": DEATH_GRACE_SECONDS}}


def watch(root: Path, interval: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "live_status_monitor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A live-status monitor already holds this root's lock.") from None
        atomic_json(root / "live_status_monitor_process.json", {"pid": os.getpid(), "started_at_utc": utc_now(), "interval_seconds": interval, "script": str(Path(__file__).resolve())})
        dead_since = None
        while True:
            status = collect_status(root)
            if status["supervisor"]["alive"] is False and not status["supervisor"].get("awaiting_new_formal_launch"):
                dead_since = time.monotonic() if dead_since is None else dead_since
            else:
                dead_since = None
            status["monitor"]["interval_seconds"] = interval
            status["monitor"]["supervisor_missing_seconds"] = time.monotonic() - dead_since if dead_since is not None else None
            stop = status["finalize"]["completed"] or (dead_since is not None and time.monotonic() - dead_since >= DEATH_GRACE_SECONDS)
            if stop:
                status["monitor"]["status"] = "exited_pipeline_complete" if status["finalize"]["completed"] else "exited_supervisor_missing"
            atomic_json(root / "live_status.json", status)
            if stop:
                return
            time.sleep(interval)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--interval", type=float, default=12.0)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error("--interval must be finite and at least one second")
    root = args.main_root.resolve()
    if args.detach:
        root.mkdir(parents=True, exist_ok=True)
        previous = read_json(root / "live_status_monitor_process.json") or {}
        existing = process_identity(previous.get("pid"), root, "scripts.rq3.rq12_vega_watch_status")
        if existing["alive"]:
            print(json.dumps({"pid": existing["pid"], "status": "already_running", "status_path": str(root / "live_status.json")}))
            return 0
        with (root / "live_status_monitor.log").open("a") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--main-root", str(root), "--interval", str(args.interval)],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True, cwd=Path(__file__).resolve().parents[2])
        print(json.dumps({"pid": child.pid, "status": "launched", "status_path": str(root / "live_status.json")}))
        return 0
    watch(root, args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
