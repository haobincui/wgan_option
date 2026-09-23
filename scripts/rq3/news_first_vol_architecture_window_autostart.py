"""Fail-closed autostart monitor for the architecture/window RQ3 studies.

The monitor is deliberately separate from both scientific pipelines.  It waits
for the currently-running 280-job text-effectiveness experiment to become a
hash-bound terminal root, waits for the five production model files to be
installed and attested, runs the two real GPU benchmarks serially, and finally
starts the architecture and alignment-window supervisors.  It never signals or
pre-empts a process using a GPU.

All mutable monitor evidence lives below ``DEFAULT_CONTROL_ROOT``.  Formal
experiment roots and terminal logs are never truncated on a repeated call.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, IO

from scripts.rq3.news_first_vol_experiment_supervisor_helpers import (
    SupervisorHelperError,
    SupervisorLock,
    append_stage_journal,
    atomic_write_json,
    gpu_availability_decision,
    pid_alive,
    process_start_ticks,
    system_resource_snapshot,
    utc_now,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_KIND = "rq3_architecture_window_autostart_v1"
DEFAULT_CONTROL_ROOT = (
    REPO_ROOT / "outputs/experiments/rq3_architecture_window_autostart_control"
)
DEFAULT_OLD_ROOT = REPO_ROOT / (
    "outputs/experiments/"
    "rq12_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_5m_v1"
)
DEFAULT_CORE_READY = (
    REPO_ROOT
    / "outputs/experiments/rq3_architecture_window_autostart_v1/core_ready.json"
)
PYTHON_EXECUTABLE = "/home/haobin_cui/.conda/envs/py312/bin/python"
STUDY_MODULE = "scripts.rq3.news_first_vol_architecture_window_study"

ARCHITECTURE_CONFIG = (
    REPO_ROOT / "configs/rq3/news_first_vol_generator_architecture_3seed_5m.yaml"
)
WINDOW_CONFIG = REPO_ROOT / "configs/rq3/news_first_vol_alignment_tolerance_3seed.yaml"
ARCHITECTURE_ROOT = REPO_ROOT / (
    "outputs/experiments/"
    "rq3_news_first_vol_generator_architecture_3seed_5m_exact_ttm_v1"
)
WINDOW_ROOT = REPO_ROOT / (
    "outputs/experiments/rq3_news_first_vol_alignment_tolerance_3seed_exact_ttm_v1"
)
SHARED_BENCHMARK_SUITE = (
    REPO_ROOT
    / "outputs/experiments/rq3_architecture_window_shared_gpu_slots_v1/benchmark_suite.json"
)

CORE_FILES = (
    "src/wgan_option/models/common.py",
    "src/wgan_option/models/generator.py",
    "src/wgan_option/models/gan_model.py",
    "src/wgan_option/config.py",
    "src/wgan_option/utils/inference_helpers.py",
)
OLD_PROCESS_NEEDLES = (
    "news_first_vol_film_unet_pure_cnn_backbone_text_effect",
    DEFAULT_OLD_ROOT.name,
)
NEW_PROCESS_NEEDLES = (
    STUDY_MODULE,
    ARCHITECTURE_ROOT.name,
    WINDOW_ROOT.name,
)
MINIMUM_FREE_BYTES = 30 * 1024**3


class AutostartError(RuntimeError):
    """Raised when an autostart safety contract cannot be established."""


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else REPO_ROOT / value).resolve()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _legacy_payload_sha256(value: object) -> str:
    """Hash payloads using the frozen 280-job registry's compact encoding."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = payload_sha256(result)
    return result


def _write_signed(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return atomic_write_json(path, _signed(payload))


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutostartError(f"Invalid JSON artifact: {source}") from exc
    if not isinstance(value, Mapping):
        raise AutostartError(f"JSON artifact must be a mapping: {source}")
    return dict(value)


def _read_signed(path: str | Path, *, kind: str | None = None) -> dict[str, Any]:
    source = Path(path)
    payload = _read_json(source)
    observed = str(payload.get("payload_sha256", ""))
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    # New supervisor/control artifacts use the shared pretty-JSON canonical
    # profile.  The already-frozen 280-job lifecycle and the separately-built
    # production-core attestation use the repository's older compact-JSON
    # profile.  Both profiles are deterministic; accepting exactly these two
    # preserves their immutable signatures without rewriting either producer.
    expected_digests = {payload_sha256(unsigned), _legacy_payload_sha256(unsigned)}
    if not observed or observed not in expected_digests:
        raise AutostartError(f"Signed payload drift: {source}")
    if kind is not None and payload.get("kind") != kind:
        raise AutostartError(f"Signed payload kind drift: {source}")
    return payload


def _verify_file(path: str | Path, expected_sha: object, label: str) -> Path:
    value = Path(path).resolve()
    if not value.is_file() or sha256_file(value) != str(expected_sha):
        raise AutostartError(f"{label} is missing or has SHA drift: {value}")
    return value


def _python_processes_with(needles: Sequence[str]) -> list[dict[str, Any]]:
    """Return live related Python processes, excluding this monitor."""

    rows: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            parts = [
                part.decode("utf-8", errors="replace")
                for part in (proc / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except OSError:
            continue
        if not parts or "python" not in Path(parts[0]).name.lower():
            continue
        command = " ".join(parts)
        if any(needle in command for needle in needles):
            rows.append(
                {
                    "pid": int(proc.name),
                    "process_start_ticks": process_start_ticks(int(proc.name)),
                    "command": command,
                }
            )
    return sorted(rows, key=lambda row: int(row["pid"]))


def validate_old_terminal(old_root: str | Path) -> dict[str, Any]:
    """Validate the immutable terminal identity of the 40+240 experiment."""

    root = _resolve(old_root)
    registry_path = root / "registry/task_registry.json"
    registry = _read_json(registry_path)
    jobs = list(registry.get("jobs") or ())
    counts = Counter(str(row.get("status")) for row in jobs if isinstance(row, Mapping))
    if (
        registry.get("kind") != "pure_cnn_parent_film_text_effectiveness_registry_v1"
        or registry.get("status") != "completed"
        or not bool(registry.get("terminal_complete"))
        or len(jobs) != 280
        or counts != Counter({"completed": 280})
    ):
        raise AutostartError("Old experiment is not terminal at 280/280 completed")
    # This registry intentionally stores a jobs hash rather than an outer hash.
    if registry.get("jobs_sha256") != _legacy_payload_sha256(jobs):
        raise AutostartError("Old experiment jobs registry SHA drift")
    qa_path = _verify_file(
        registry.get("terminal_qa_path", root / "qa.json"),
        registry.get("terminal_qa_sha256", ""),
        "old terminal QA",
    )
    qa = _read_signed(qa_path, kind="pure_cnn_parent_film_text_effect_terminal_qa_v1")
    if (
        qa.get("status") != "passed"
        or int(qa.get("training_jobs_completed", -1)) != 280
    ):
        raise AutostartError("Old terminal QA count/status drift")
    output_manifest = _verify_file(
        registry.get("output_manifest_path", ""),
        registry.get("output_manifest_sha256", ""),
        "old output manifest",
    )
    snapshot = _verify_file(
        registry.get("final_registry_snapshot_path", ""),
        registry.get("final_registry_snapshot_sha256", ""),
        "old final registry snapshot",
    )
    _read_signed(
        snapshot, kind="pure_cnn_parent_film_text_effect_final_registry_snapshot_v1"
    )
    source_manifest = root / "stages/backbones/code_hashes.csv"
    if not source_manifest.is_file():
        raise AutostartError("Old source manifest is missing")
    with source_manifest.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 288 or any(not row.get("sha256") for row in source_rows):
        raise AutostartError("Old source manifest must contain exactly 288 hashed rows")
    return {
        "root": str(root),
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": sha256_file(registry_path),
        "qa_path": str(qa_path),
        "qa_sha256": sha256_file(qa_path),
        "output_manifest_path": str(output_manifest),
        "output_manifest_sha256": sha256_file(output_manifest),
        "final_registry_snapshot_path": str(snapshot),
        "final_registry_snapshot_sha256": sha256_file(snapshot),
        "source_manifest_path": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_manifest_rows": len(source_rows),
        "training_jobs_completed": 280,
    }


def validate_core_ready(path: str | Path) -> dict[str, Any]:
    """Validate the exact five-file production-core readiness attestation."""

    source = _resolve(path)
    ready = _read_signed(source, kind="architecture_window_core_ready_v1")
    if int(ready.get("schema_version", -1)) != 1 or ready.get("status") != "ready":
        raise AutostartError("Production core readiness status drift")
    raw_files = ready.get("files")
    if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
        raise AutostartError("core_ready files must be a sequence")
    by_relative: dict[str, dict[str, Any]] = {}
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise AutostartError("Malformed core_ready file row")
        row = dict(raw)
        value = Path(str(row.get("path", row.get("relative_path", ""))))
        try:
            relative = value.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            relative = value.as_posix()
        if relative in by_relative:
            raise AutostartError("Duplicate core_ready file row")
        by_relative[relative] = row
    if set(by_relative) != set(CORE_FILES):
        raise AutostartError("core_ready must bind the exact five production files")
    verified = []
    for relative in CORE_FILES:
        file_path = REPO_ROOT / relative
        row = by_relative[relative]
        if (
            not file_path.is_file()
            or file_path.stat().st_size != int(row.get("size_bytes", -1))
            or sha256_file(file_path) != str(row.get("sha256", ""))
        ):
            raise AutostartError(f"Production core hash drift: {relative}")
        verified.append(
            {
                "path": relative,
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    if (
        ready.get("staging_removed") is not True
        or ready.get("gpu_started") is not False
    ):
        raise AutostartError("core_ready cleanup/GPU-start proof drift")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "files": verified,
        "tests": ready.get("tests", ready.get("test_summary", {})),
    }


def _wait_until(
    probe: Callable[[], Any],
    *,
    description: str,
    poll_seconds: float,
    sleeper: Callable[[float], None] = time.sleep,
    on_wait: Callable[[Exception], None] | None = None,
) -> Any:
    while True:
        try:
            return probe()
        except (AutostartError, FileNotFoundError, SupervisorHelperError) as exc:
            if on_wait is not None:
                on_wait(exc)
            sleeper(poll_seconds)


def _wait_no_processes(
    needles: Sequence[str], *, poll_seconds: float, sleeper: Callable[[float], None]
) -> None:
    def probe() -> bool:
        rows = _python_processes_with(needles)
        if rows:
            raise AutostartError(f"Related Python processes are still live: {rows}")
        return True

    _wait_until(
        probe,
        description="process drain",
        poll_seconds=poll_seconds,
        sleeper=sleeper,
    )


def wait_for_stable_resources(
    *,
    samples: int,
    sample_seconds: float,
    snapshot_provider: Callable[[], Mapping[str, Any]] = system_resource_snapshot,
    disk_usage_provider: Callable[[str | Path], Any] = shutil.disk_usage,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if samples < 1 or sample_seconds <= 0:
        raise AutostartError("Invalid stable-resource sampling contract")
    snapshots: list[dict[str, Any]] = []
    while len(snapshots) < samples:
        snapshot = dict(snapshot_provider())
        decision = gpu_availability_decision(snapshot, gpu_ids=(0, 1))
        free_bytes = int(disk_usage_provider(REPO_ROOT).free)
        if not decision["ready"] or free_bytes < MINIMUM_FREE_BYTES:
            snapshots.clear()
            sleeper(sample_seconds)
            continue
        snapshots.append(
            {"snapshot": snapshot, "decision": decision, "disk_free_bytes": free_bytes}
        )
        if len(snapshots) < samples:
            sleeper(sample_seconds)
    return {
        "status": "passed",
        "stable_samples": samples,
        "minimum_disk_free_bytes": MINIMUM_FREE_BYTES,
        "samples": snapshots,
    }


def _command(config: Path, root: Path, action: str) -> list[str]:
    result = [
        PYTHON_EXECUTABLE,
        "-m",
        STUDY_MODULE,
        action,
        "--config",
        str(config.resolve()),
    ]
    if action != "dry-run":
        result.extend(("--output-dir", str(root.resolve())))
    if action == "run-pipeline":
        result.append("--resume")
    return result


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT)))
    return env


def _run_checked(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AutostartError(
            "Command failed: " + " ".join(command) + f"\n{result.stderr[-4000:]}"
        )
    return result


def validate_dry_run(payload: Mapping[str, Any], *, line: str) -> dict[str, Any]:
    expected_jobs = {"architecture": 69, "window": 96}[line]
    if (
        payload.get("line") != line
        or int(payload.get("new_training_jobs", -1)) != expected_jobs
        or payload.get("formal_root_created") is not False
        or int(payload.get("test_metric_files_read", -1)) != 0
        or payload.get("baseline_status") != "passed"
        or dict(payload.get("input_preflight") or {}).get("status") != "passed"
        or dict(payload.get("core_runtime_preflight") or {}).get("status") != "passed"
    ):
        raise AutostartError(f"{line} dry-run contract did not pass")
    return dict(payload)


def run_dry_runs() -> dict[str, Any]:
    if ARCHITECTURE_ROOT.exists() or WINDOW_ROOT.exists():
        raise AutostartError("Formal roots must be absent before initial dry-runs")
    rows: dict[str, Any] = {}
    for line, config, root in (
        ("architecture", ARCHITECTURE_CONFIG, ARCHITECTURE_ROOT),
        ("window", WINDOW_CONFIG, WINDOW_ROOT),
    ):
        result = _run_checked(_command(config, root, "dry-run"))
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise AutostartError(f"{line} dry-run emitted invalid JSON") from exc
        rows[line] = validate_dry_run(payload, line=line)
    return rows


def _benchmark_path(root: Path) -> Path:
    return root.with_name(root.name + "_control") / "benchmark.json"


def validate_benchmark(
    path: str | Path, *, root: Path, require_formal_root_absent: bool = True
) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = _read_signed(source, kind="architecture_window_gpu_epoch1_benchmark_v1")
    if (
        payload.get("status") != "passed"
        or Path(str(payload.get("formal_output_root", ""))).resolve() != root.resolve()
        or int(payload.get("selected_workers_per_gpu", 0)) not in {4, 8, 12, 18}
        or payload.get("formal_root_created_before_benchmark_pass") is not False
    ):
        raise AutostartError(f"Benchmark evidence drift: {source}")
    if require_formal_root_absent and root.exists():
        raise AutostartError(f"Benchmark illegally created formal root: {root}")
    return payload


def run_benchmarks() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for line, config, root in (
        ("architecture", ARCHITECTURE_CONFIG, ARCHITECTURE_ROOT),
        ("window", WINDOW_CONFIG, WINDOW_ROOT),
    ):
        _run_checked(_command(config, root, "benchmark"))
        evidence = validate_benchmark(_benchmark_path(root), root=root)
        rows[line] = {
            "path": str(_benchmark_path(root).resolve()),
            "sha256": sha256_file(_benchmark_path(root)),
            "selected_workers_per_gpu": evidence["selected_workers_per_gpu"],
        }
    return rows


def validate_resume_gates(control: Path) -> dict[str, Any]:
    """Revalidate immutable prelaunch evidence before a dead-partial resume."""

    old = _read_signed(
        control / "old_terminal_gate.json",
        kind="architecture_window_old_terminal_gate_v1",
    )
    if (
        int(old.get("training_jobs_completed", -1)) != 280
        or int(old.get("source_manifest_rows", -1)) != 288
        or int(old.get("related_python_processes", -1)) != 0
    ):
        raise AutostartError("Old-terminal resume gate drift")
    for prefix in ("registry", "qa", "output_manifest", "final_registry_snapshot"):
        _verify_file(
            old.get(f"{prefix}_path", ""), old.get(f"{prefix}_sha256", ""), prefix
        )

    core = _read_signed(
        control / "core_ready_gate.json",
        kind="architecture_window_core_ready_gate_v1",
    )
    validate_core_ready(core.get("path", ""))

    dry = _read_signed(
        control / "dry_run_gate.json",
        kind="architecture_window_autostart_dry_run_gate_v1",
    )
    if dry.get("status") != "passed":
        raise AutostartError("Dry-run resume gate is not passed")
    raw_dry_runs = dict(dry.get("dry_runs") or {})
    validate_dry_run(dict(raw_dry_runs.get("architecture") or {}), line="architecture")
    validate_dry_run(dict(raw_dry_runs.get("window") or {}), line="window")

    benchmark = _read_signed(
        control / "benchmark_gate.json",
        kind="architecture_window_autostart_benchmark_gate_v1",
    )
    if benchmark.get("status") != "passed":
        raise AutostartError("Benchmark resume gate is not passed")
    rows = dict(benchmark.get("benchmarks") or {})
    for line, root in (("architecture", ARCHITECTURE_ROOT), ("window", WINDOW_ROOT)):
        row = dict(rows.get(line) or {})
        path = Path(str(row.get("path", ""))).resolve()
        _verify_file(path, row.get("sha256", ""), f"{line} resume benchmark")
        validate_benchmark(path, root=root, require_formal_root_absent=False)
    return {"old": old, "core": core, "dry_run": dry, "benchmark": benchmark}


def validate_shared_benchmark_suite(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = _read_signed(source, kind="architecture_window_dual_benchmark_suite_v1")
    rows = list(payload.get("benchmarks") or ())
    expected = {ARCHITECTURE_ROOT.resolve(), WINDOW_ROOT.resolve()}
    observed = {Path(str(row.get("formal_root", ""))).resolve() for row in rows}
    if observed != expected or len(rows) != 2:
        raise AutostartError("Shared benchmark suite root coverage drift")
    for row in rows:
        benchmark_path = Path(str(row.get("benchmark_path", ""))).resolve()
        root = Path(str(row["formal_root"])).resolve()
        if benchmark_path != _benchmark_path(root).resolve():
            raise AutostartError("Shared benchmark path drift")
        _verify_file(benchmark_path, row.get("benchmark_sha256"), "shared benchmark")
        validate_benchmark(benchmark_path, root=root, require_formal_root_absent=False)
    return payload


def _launch_process(command: Sequence[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is what makes an attempt log immutable/non-truncating.
    handle: IO[bytes] = log_path.open("xb")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=_subprocess_env(),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        handle.close()
    ticks = process_start_ticks(process.pid)
    if ticks is None or not pid_alive(process.pid, expected_start_ticks=ticks):
        raise AutostartError("Launched supervisor did not remain alive")
    return {
        "pid": process.pid,
        "process_start_ticks": ticks,
        "command": list(command),
        "log_path": str(log_path.resolve()),
        "launched_at_utc": utc_now(),
    }


def _next_attempt(control: Path) -> tuple[int, Path]:
    existing = []
    for value in (control / "attempts").glob("attempt_*"):
        try:
            existing.append(int(value.name.removeprefix("attempt_")))
        except ValueError:
            continue
    attempt = max(existing, default=0) + 1
    root = control / "attempts" / f"attempt_{attempt:04d}"
    root.mkdir(parents=True, exist_ok=False)
    return attempt, root


def _terminal_new_root(root: Path) -> bool:
    registry_path = root / "registry/task_registry.json"
    if not registry_path.is_file():
        return False
    try:
        registry = _read_signed(
            registry_path, kind="architecture_window_study_registry_v1"
        )
    except AutostartError:
        return False
    return bool(registry.get("terminal_complete")) and (root / "qa.json").is_file()


def _launch_state(control: Path) -> dict[str, Any] | None:
    path = control / "launch_manifest.json"
    return (
        _read_signed(path, kind="architecture_window_autostart_launch_v1")
        if path.is_file()
        else None
    )


def _live_launch_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in manifest.get("pipelines", ())
        if pid_alive(
            int(row.get("pid", -1)),
            expected_start_ticks=int(row.get("process_start_ticks", -1)),
        )
    ]


def classify_existing(control: Path) -> str:
    manifest = _launch_state(control)
    terminal = _terminal_new_root(ARCHITECTURE_ROOT) and _terminal_new_root(WINDOW_ROOT)
    if terminal:
        return "terminal_read_only"
    if manifest is not None and _live_launch_rows(manifest):
        return "live_monitor_only"
    roots_exist = ARCHITECTURE_ROOT.exists() or WINDOW_ROOT.exists()
    if manifest is not None or roots_exist:
        return "dead_partial"
    return "new"


def _wait_shared_suite(
    *, poll_seconds: float, architecture: Mapping[str, Any]
) -> dict[str, Any]:
    while True:
        if not pid_alive(
            int(architecture["pid"]),
            expected_start_ticks=int(architecture["process_start_ticks"]),
        ):
            raise AutostartError(
                "Architecture supervisor exited before freezing shared benchmark suite"
            )
        try:
            return validate_shared_benchmark_suite(SHARED_BENCHMARK_SUITE)
        except (AutostartError, FileNotFoundError):
            time.sleep(poll_seconds)


def launch_pipelines(control: Path, *, poll_seconds: float) -> Path:
    attempt, attempt_root = _next_attempt(control)
    architecture = _launch_process(
        _command(ARCHITECTURE_CONFIG, ARCHITECTURE_ROOT, "run-pipeline"),
        attempt_root / "architecture_pipeline.log",
    )
    suite = _wait_shared_suite(poll_seconds=poll_seconds, architecture=architecture)
    window = _launch_process(
        _command(WINDOW_CONFIG, WINDOW_ROOT, "run-pipeline"),
        attempt_root / "window_pipeline.log",
    )
    return _write_signed(
        control / "launch_manifest.json",
        {
            "schema_version": 1,
            "kind": "architecture_window_autostart_launch_v1",
            "status": "launched",
            "attempt": attempt,
            "pipelines": [
                {
                    "line": "architecture",
                    "root": str(ARCHITECTURE_ROOT),
                    **architecture,
                },
                {"line": "window", "root": str(WINDOW_ROOT), **window},
            ],
            "shared_benchmark_suite_path": str(SHARED_BENCHMARK_SUITE.resolve()),
            "shared_benchmark_suite_sha256": sha256_file(SHARED_BENCHMARK_SUITE),
            "shared_benchmark_suite_payload_sha256": suite["payload_sha256"],
            "launched_at_utc": utc_now(),
        },
    )


def status(control_root: str | Path = DEFAULT_CONTROL_ROOT) -> dict[str, Any]:
    control = _resolve(control_root)
    state = classify_existing(control)
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "architecture_window_autostart_status_v1",
        "state": state,
        "control_root": str(control),
        "old_terminal": (DEFAULT_OLD_ROOT / "qa.json").is_file(),
        "core_ready": DEFAULT_CORE_READY.is_file(),
        "architecture_terminal": _terminal_new_root(ARCHITECTURE_ROOT),
        "window_terminal": _terminal_new_root(WINDOW_ROOT),
    }
    manifest = _launch_state(control)
    if manifest is not None:
        result["launch_manifest_path"] = str(
            (control / "launch_manifest.json").resolve()
        )
        result["attempt"] = manifest.get("attempt")
        result["pipelines"] = [
            {
                **dict(row),
                "live": pid_alive(
                    int(row.get("pid", -1)),
                    expected_start_ticks=int(row.get("process_start_ticks", -1)),
                ),
            }
            for row in manifest.get("pipelines", ())
        ]
    journal = control / "pipeline_journal.json"
    if journal.is_file():
        result["journal"] = _read_signed(
            journal, kind="rq3_supervisor_stage_journal_v1"
        ).get("latest")
    return result


def run(
    *,
    control_root: str | Path = DEFAULT_CONTROL_ROOT,
    old_root: str | Path = DEFAULT_OLD_ROOT,
    core_ready_path: str | Path = DEFAULT_CORE_READY,
    resume: bool = False,
    poll_seconds: float = 30.0,
    gpu_stable_samples: int = 3,
) -> dict[str, Any]:
    """Run the durable monitor until both downstream supervisors are launched."""

    if poll_seconds <= 0:
        raise AutostartError("poll_seconds must be positive")
    control = _resolve(control_root)
    control.mkdir(parents=True, exist_ok=True)
    initial = classify_existing(control)
    if initial in {"terminal_read_only", "live_monitor_only"}:
        return status(control)
    if initial == "dead_partial" and not resume:
        raise AutostartError("Dead partial state requires explicit --resume")
    with SupervisorLock(control, name="autostart"):
        append_stage_journal(
            control,
            experiment_kind=EXPERIMENT_KIND,
            stage="autostart",
            status="running",
            details={"resume": bool(resume), "initial_state": initial},
        )
        try:
            if initial == "dead_partial":
                validate_resume_gates(control)
                _wait_no_processes(
                    NEW_PROCESS_NEEDLES,
                    poll_seconds=poll_seconds,
                    sleeper=time.sleep,
                )
                resources = wait_for_stable_resources(
                    samples=gpu_stable_samples, sample_seconds=poll_seconds
                )
                _write_signed(
                    control / "resume_resource_gate.json",
                    {
                        "schema_version": 1,
                        "kind": "architecture_window_autostart_resume_resource_gate_v1",
                        **resources,
                        "passed_at_utc": utc_now(),
                    },
                )
                launch_manifest = launch_pipelines(control, poll_seconds=poll_seconds)
                append_stage_journal(
                    control,
                    experiment_kind=EXPERIMENT_KIND,
                    stage="autostart_resume",
                    status="completed",
                    details={"launch_manifest": str(launch_manifest)},
                )
                return status(control)

            old = _wait_until(
                lambda: validate_old_terminal(old_root),
                description="old terminal experiment",
                poll_seconds=poll_seconds,
            )
            _wait_no_processes(
                OLD_PROCESS_NEEDLES, poll_seconds=poll_seconds, sleeper=time.sleep
            )
            _write_signed(
                control / "old_terminal_gate.json",
                {
                    "schema_version": 1,
                    "kind": "architecture_window_old_terminal_gate_v1",
                    **old,
                    "related_python_processes": 0,
                    "passed_at_utc": utc_now(),
                },
            )
            append_stage_journal(
                control,
                experiment_kind=EXPERIMENT_KIND,
                stage="old_terminal_gate",
                status="completed",
                details={"training_jobs_completed": 280},
            )

            core = _wait_until(
                lambda: validate_core_ready(core_ready_path),
                description="production core readiness",
                poll_seconds=poll_seconds,
            )
            _write_signed(
                control / "core_ready_gate.json",
                {
                    "schema_version": 1,
                    "kind": "architecture_window_core_ready_gate_v1",
                    **core,
                    "passed_at_utc": utc_now(),
                },
            )
            _wait_no_processes(
                NEW_PROCESS_NEEDLES, poll_seconds=poll_seconds, sleeper=time.sleep
            )
            resources = wait_for_stable_resources(
                samples=gpu_stable_samples, sample_seconds=poll_seconds
            )
            _write_signed(
                control / "resource_gate.json",
                {
                    "schema_version": 1,
                    "kind": "architecture_window_autostart_resource_gate_v1",
                    **resources,
                    "passed_at_utc": utc_now(),
                },
            )
            dry_runs = run_dry_runs()
            _write_signed(
                control / "dry_run_gate.json",
                {
                    "schema_version": 1,
                    "kind": "architecture_window_autostart_dry_run_gate_v1",
                    "status": "passed",
                    "dry_runs": dry_runs,
                    "passed_at_utc": utc_now(),
                },
            )
            benchmarks = run_benchmarks()
            _write_signed(
                control / "benchmark_gate.json",
                {
                    "schema_version": 1,
                    "kind": "architecture_window_autostart_benchmark_gate_v1",
                    "status": "passed",
                    "benchmarks": benchmarks,
                    "passed_at_utc": utc_now(),
                },
            )
            launch_manifest = launch_pipelines(control, poll_seconds=poll_seconds)
            append_stage_journal(
                control,
                experiment_kind=EXPERIMENT_KIND,
                stage="autostart",
                status="completed",
                details={"launch_manifest": str(launch_manifest)},
            )
            return status(control)
        except BaseException as exc:
            append_stage_journal(
                control,
                experiment_kind=EXPERIMENT_KIND,
                stage="autostart",
                status="failed",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"))
    parser.add_argument("--control-root", default=str(DEFAULT_CONTROL_ROOT))
    parser.add_argument("--old-root", default=str(DEFAULT_OLD_ROOT))
    parser.add_argument("--core-ready", default=str(DEFAULT_CORE_READY))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--gpu-stable-samples", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "status":
        result = status(args.control_root)
    else:
        result = run(
            control_root=args.control_root,
            old_root=args.old_root,
            core_ready_path=args.core_ready,
            resume=args.resume,
            poll_seconds=args.poll_seconds,
            gpu_stable_samples=args.gpu_stable_samples,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AutostartError",
    "classify_existing",
    "launch_pipelines",
    "main",
    "run",
    "run_benchmarks",
    "run_dry_runs",
    "status",
    "validate_benchmark",
    "validate_core_ready",
    "validate_dry_run",
    "validate_old_terminal",
    "validate_resume_gates",
    "validate_shared_benchmark_suite",
    "wait_for_stable_resources",
]
