"""Audited recovery runner for the frozen Pure-CNN -> FiLM experiment.

The active formal root froze the orchestration source before a checkpoint-freeze
route bug was discovered.  Editing that frozen source (or its hash manifest)
would invalidate 40 completed parents.  This module therefore provides an
explicit, hash-recorded execution-layer repair:

* the frozen experiment source must still match its original code manifest;
* only the missing nested runtime profiles are installed around checkpoint
  freeze calls;
* training workers continue to execute the original frozen worker module; and
* concurrency is dynamically capped by both the frozen 10-worker benchmark and
  currently free GPU memory.

The recovery manifest is stored inside the formal root so the exceptional
orchestration path remains part of the final output lineage.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed as experiment,
)
from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle as lifecycle,
)
from scripts.rq3.news_first_vol_experiment_supervisor_helpers import (
    SupervisorLock,
    append_stage_journal,
    system_resource_snapshot,
)


RECOVERY_KIND = "pure_cnn_parent_text_effect_freeze_route_recovery_v1"
RECOVERY_PATH = "registry/runtime_recovery_manifest.json"
FROZEN_MAIN_RELATIVE_PATH = (
    "scripts/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed.py"
)
DEFAULT_SHARED_WORKERS_PER_GPU = 4
DEFAULT_IDLE_WORKERS_PER_GPU = 10
DEFAULT_WORKER_MEMORY_MIB = 450
DEFAULT_FREE_MEMORY_RESERVE_MIB = 1_536


def _root(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _recovery_path(root: Path) -> Path:
    return root / RECOVERY_PATH


def _frozen_main_attestation(root: Path) -> dict[str, Any]:
    """Prove that recovery did not rewrite any frozen experiment source."""

    manifest_path = root / "stages/backbones/code_hashes.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {"artifact_role", "path", "size_bytes", "sha256"}
    if not required.issubset(frame.columns):
        raise ValueError("Backbone code manifest is malformed")
    source_path = (experiment.REPO_ROOT / FROZEN_MAIN_RELATIVE_PATH).resolve()
    selected = frame.loc[
        frame["path"].map(lambda value: Path(value).resolve()).eq(source_path)
    ]
    if len(selected) != 1:
        raise ValueError("Frozen experiment source has no unique code-manifest row")
    row = selected.iloc[0]
    observed_sha = experiment.sha256_file(source_path)
    observed_size = source_path.stat().st_size
    if observed_sha != row["sha256"] or observed_size != int(row["size_bytes"]):
        raise ValueError("Frozen experiment source drift; recovery is not permitted")

    recovery_source = Path(__file__).resolve()
    if frame["path"].map(lambda value: Path(value).resolve()).eq(recovery_source).any():
        raise ValueError(
            "Recovery source unexpectedly belongs to the frozen code manifest"
        )
    return {
        "frozen_code_manifest_path": str(manifest_path.resolve()),
        "frozen_code_manifest_sha256": experiment.sha256_file(manifest_path),
        "frozen_experiment_source_path": str(source_path),
        "frozen_experiment_source_size_bytes": observed_size,
        "frozen_experiment_source_sha256": observed_sha,
        "recovery_source_path": str(recovery_source),
        "recovery_source_size_bytes": recovery_source.stat().st_size,
        "recovery_source_sha256": experiment.sha256_file(recovery_source),
        "frozen_manifests_modified": False,
    }


def _write_recovery_manifest(
    root: Path,
    *,
    status: str,
    shared_workers_per_gpu: int,
    idle_workers_per_gpu: int,
    details: Mapping[str, Any] | None = None,
) -> Path:
    path = _recovery_path(root)
    previous: dict[str, Any] = {}
    if path.is_file():
        previous = lifecycle._read_signed(path, kind=RECOVERY_KIND)
    attestation = _frozen_main_attestation(root)
    if previous and previous.get("source_attestation") != attestation:
        raise ValueError("Recovery source or frozen-source attestation drift")
    benchmark_path = experiment._control_root(root) / "benchmark_result.json"
    if not benchmark_path.is_file():
        raise FileNotFoundError(benchmark_path)
    history = [dict(row) for row in previous.get("history", [])]
    history.append(
        {
            "sequence": len(history),
            "status": str(status),
            "pid": os.getpid(),
            "updated_at_utc": experiment.direct.utc_now(),
            "details": dict(details or {}),
        }
    )
    return lifecycle._write_signed(
        path,
        {
            "schema_version": 1,
            "kind": RECOVERY_KIND,
            "reason": (
                "checkpoint freeze called the shared direct implementation without "
                "the stage runtime profile"
            ),
            "repair_scope": "orchestration_routing_and_resource_scheduling_only",
            "scientific_contract_changed": False,
            "training_worker_module": experiment.WORKER_MODULE,
            "formal_workers_per_gpu": 10,
            "maximum_shared_workers_per_gpu": int(shared_workers_per_gpu),
            "maximum_idle_workers_per_gpu": int(idle_workers_per_gpu),
            "benchmark_result_path": str(benchmark_path.resolve()),
            "benchmark_result_sha256": experiment.sha256_file(benchmark_path),
            "source_attestation": attestation,
            "history": history,
            "status": str(status),
        },
    )


@contextmanager
def _checkpoint_freeze_router(root: Path) -> Iterator[None]:
    """Route shared freeze calls through their complete stage profiles."""

    roots = {
        key: value.resolve() for key, value in experiment._stage_roots(root).items()
    }
    original = experiment.direct._freeze_checkpoints

    def routed(stage_root: str | Path) -> Path:
        resolved = Path(stage_root).resolve()
        if resolved in {roots["backbones"], roots["pure_continuation"]}:
            # The caller already installed _pure_stage_profile.  This second
            # layer propagates its values into the shared direct module.
            with experiment.pure._runtime_profile():
                return original(resolved)
        if resolved == roots["film_continuations"]:
            # Likewise propagate the active five-arm stage through both thin
            # FiLM wrappers before entering the shared direct freezer.
            with (
                experiment.film_text.film_text_profile(),
                experiment.film_text.multiseed.multiseed_profile(),
            ):
                return original(resolved)
        raise ValueError(f"Recovery refused an unknown stage root: {resolved}")

    experiment.direct._freeze_checkpoints = routed
    try:
        yield
    finally:
        experiment.direct._freeze_checkpoints = original


def freeze_backbones(root: str | Path) -> Path:
    resolved = _root(root)
    with _checkpoint_freeze_router(resolved):
        return experiment.freeze_backbones(resolved, resume=True)


def freeze_evaluation(root: str | Path) -> Path:
    resolved = _root(root)
    with _checkpoint_freeze_router(resolved):
        return experiment.freeze_evaluation(resolved, resume=True)


@contextmanager
def _stage_runtime_profile(root: Path, stage_key: str) -> Iterator[None]:
    if stage_key == "pure_continuation":
        with (
            experiment._pure_stage_profile(
                main_root=root,
                stage_name="pure_continuation",
                arm=experiment.PURE_CONTINUATION_ARM,
                use_graft=True,
            ),
            experiment.pure._runtime_profile(),
        ):
            yield
        return
    if stage_key == "film_continuations":
        with (
            experiment._film_stage_profile(
                main_root=root, stage_name="film_continuations"
            ),
            experiment.film_text.film_text_profile(),
            experiment.film_text.multiseed.multiseed_profile(),
        ):
            yield
        return
    raise ValueError(f"Unsupported shared-training stage: {stage_key}")


def _workers_by_gpu(
    snapshot: Mapping[str, Any],
    *,
    gpu_ids: Sequence[int],
    shared_cap: int,
    idle_cap: int,
    worker_memory_mib: int,
    reserve_mib: int,
) -> dict[int, int]:
    """Choose a conservative per-GPU batch size from live free memory."""

    gpu_rows = {int(row["gpu_index"]): dict(row) for row in snapshot.get("gpus", [])}
    uuid_to_gpu = {
        str(row["gpu_uuid"]): int(row["gpu_index"]) for row in snapshot.get("gpus", [])
    }
    external_by_gpu = {int(gpu): False for gpu in gpu_ids}
    for process in snapshot.get("compute_processes", []):
        gpu = uuid_to_gpu.get(str(process.get("gpu_uuid", "")))
        if gpu in external_by_gpu:
            external_by_gpu[gpu] = True
    selected: dict[int, int] = {}
    for gpu in map(int, gpu_ids):
        if gpu not in gpu_rows:
            raise ValueError(f"GPU {gpu} missing from resource snapshot")
        row = gpu_rows[gpu]
        free_mib = float(row["memory_total_mib"]) - float(row["memory_used_mib"])
        memory_cap = max(0, math.floor((free_mib - reserve_mib) / worker_memory_mib))
        execution_cap = shared_cap if external_by_gpu[gpu] else idle_cap
        selected[gpu] = min(int(execution_cap), int(memory_cap))
    return selected


def _pending_jobs_for_wave(
    stage_root: Path, jobs: Sequence[Mapping[str, Any]], wave: int
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for raw in jobs:
        job = dict(raw)
        if int(job["wave"]) != int(wave):
            continue
        state = experiment.direct.read_json(
            experiment.direct._status_path(stage_root, str(job["job_id"]))
        )
        if not experiment.direct._completed_valid(job, state):
            pending.append(job)
    return pending


def _select_batch(
    pending: Sequence[Mapping[str, Any]], limits: Mapping[int, int]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for gpu in sorted(map(int, limits)):
        candidates = [dict(job) for job in pending if int(job["gpu_id"]) == gpu]
        selected.extend(candidates[: int(limits[gpu])])
    return selected


def _launch_stage_shared(
    root: Path,
    *,
    stage_key: str,
    shared_workers_per_gpu: int,
    idle_workers_per_gpu: int,
    wait_seconds: float = 30.0,
) -> None:
    stage_root = experiment._stage_roots(root)[stage_key]
    config = experiment._load_frozen_config(root / "resolved_config.yaml")
    gpu_ids = tuple(map(int, config["runtime"]["gpu_ids"]))
    control = experiment._control_root(root)
    with _stage_runtime_profile(root, stage_key):
        registry = experiment.direct.read_registry(stage_root)
        jobs = [dict(row) for row in registry["jobs"]]
        for wave in sorted({int(job["wave"]) for job in jobs}):
            while True:
                pending = _pending_jobs_for_wave(stage_root, jobs, wave)
                if not pending:
                    break
                snapshot = system_resource_snapshot(
                    executable=str(config["runtime"]["nvidia_smi_executable"])
                )
                limits = _workers_by_gpu(
                    snapshot,
                    gpu_ids=gpu_ids,
                    shared_cap=shared_workers_per_gpu,
                    idle_cap=idle_workers_per_gpu,
                    worker_memory_mib=DEFAULT_WORKER_MEMORY_MIB,
                    reserve_mib=DEFAULT_FREE_MEMORY_RESERVE_MIB,
                )
                batch = _select_batch(pending, limits)
                if not batch:
                    append_stage_journal(
                        control,
                        experiment_kind=experiment.EXPERIMENT_KIND,
                        stage=f"shared_{stage_key}_wave_{wave}",
                        status="waiting_for_safe_free_memory",
                        details={"limits": limits, "pending": len(pending)},
                    )
                    time.sleep(wait_seconds)
                    continue
                append_stage_journal(
                    control,
                    experiment_kind=experiment.EXPERIMENT_KIND,
                    stage=f"shared_{stage_key}_wave_{wave}",
                    status="running",
                    details={
                        "limits": limits,
                        "batch_jobs": [str(job["job_id"]) for job in batch],
                        "remaining_before_batch": len(pending),
                        "resource_snapshot": snapshot,
                    },
                )
                experiment.direct._run_wave(
                    stage_root,
                    batch,
                    wave=wave,
                    resume=True,
                )
                append_stage_journal(
                    control,
                    experiment_kind=experiment.EXPERIMENT_KIND,
                    stage=f"shared_{stage_key}_wave_{wave}",
                    status="batch_completed",
                    details={
                        "batch_jobs": [str(job["job_id"]) for job in batch],
                        "remaining_after_batch": len(
                            _pending_jobs_for_wave(stage_root, jobs, wave)
                        ),
                    },
                )


def _stage(
    root: Path,
    name: str,
    operation: Any,
) -> Any:
    control = experiment._control_root(root)
    append_stage_journal(
        control,
        experiment_kind=experiment.EXPERIMENT_KIND,
        stage=name,
        status="running",
        details={"recovery_runner": True},
    )
    result = operation()
    append_stage_journal(
        control,
        experiment_kind=experiment.EXPERIMENT_KIND,
        stage=name,
        status="completed",
        details={"result": str(result), "recovery_runner": True},
    )
    return result


def run_shared_pipeline(
    output_dir: str | Path = experiment.DEFAULT_OUTPUT_DIR,
    *,
    shared_workers_per_gpu: int = DEFAULT_SHARED_WORKERS_PER_GPU,
    idle_workers_per_gpu: int = DEFAULT_IDLE_WORKERS_PER_GPU,
) -> Path:
    """Resume the frozen root while safely sharing GPUs with external jobs."""

    root = _root(output_dir)
    if not 1 <= int(shared_workers_per_gpu) <= 10:
        raise ValueError("shared_workers_per_gpu must be in [1, 10]")
    if not 1 <= int(idle_workers_per_gpu) <= 10:
        raise ValueError("idle_workers_per_gpu must be in [1, 10]")
    if (root / "qa.json").is_file():
        lifecycle.validate_terminal(root)
        return root
    experiment.validate_partial_root(root, verify_stage_roots=True)
    attestation = _frozen_main_attestation(root)
    del attestation
    control = experiment._control_root(root)
    with SupervisorLock(control, name="pipeline"):
        _write_recovery_manifest(
            root,
            status="running",
            shared_workers_per_gpu=shared_workers_per_gpu,
            idle_workers_per_gpu=idle_workers_per_gpu,
            details={"phase": "start"},
        )
        try:
            _stage(root, "freeze_backbones_repaired", lambda: freeze_backbones(root))
            _stage(root, "graft", lambda: experiment.graft(root, resume=True))
            _stage(
                root,
                "shared_pure_continuations",
                lambda: _launch_stage_shared(
                    root,
                    stage_key="pure_continuation",
                    shared_workers_per_gpu=shared_workers_per_gpu,
                    idle_workers_per_gpu=idle_workers_per_gpu,
                ),
            )
            _stage(
                root,
                "shared_film_continuations",
                lambda: _launch_stage_shared(
                    root,
                    stage_key="film_continuations",
                    shared_workers_per_gpu=shared_workers_per_gpu,
                    idle_workers_per_gpu=idle_workers_per_gpu,
                ),
            )
            _stage(
                root,
                "finalize_continuations",
                lambda: experiment.launch_continuations(root, resume=True),
            )
            _stage(
                root,
                "freeze_evaluation_repaired",
                lambda: freeze_evaluation(root),
            )
            _stage(
                root,
                "predict_validation_trajectories",
                lambda: experiment.predict_validation_trajectories(root, resume=True),
            )
            _stage(
                root,
                "predict_test",
                lambda: experiment.predict_test(root, resume=True),
            )
            _stage(
                root,
                "predict_interventions",
                lambda: experiment.predict_interventions(root, resume=True),
            )
            _stage(root, "analyze", lambda: lifecycle.analyze(root, resume=True))
            _stage(root, "bootstrap", lambda: lifecycle.bootstrap(root, resume=True))
            _stage(root, "report", lambda: lifecycle.report(root, resume=True))
            # The recovery artifact is part of the terminal output hash
            # universe.  Freeze it before QA builds output_hashes.csv; a
            # terminal root must never be mutated afterward.
            _write_recovery_manifest(
                root,
                status="completed",
                shared_workers_per_gpu=shared_workers_per_gpu,
                idle_workers_per_gpu=idle_workers_per_gpu,
                details={"phase": "terminal_qa_next"},
            )
            _stage(root, "qa", lambda: lifecycle.qa(root, resume=True))
        except BaseException as exc:
            # Do not mutate a root after QA has committed its terminal marker.
            if not (root / "qa.json").is_file():
                _write_recovery_manifest(
                    root,
                    status="failed",
                    shared_workers_per_gpu=shared_workers_per_gpu,
                    idle_workers_per_gpu=idle_workers_per_gpu,
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            append_stage_journal(
                control,
                experiment_kind=experiment.EXPERIMENT_KIND,
                stage="recovery_pipeline",
                status="failed",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
            raise
    return root


def status(output_dir: str | Path = experiment.DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _root(output_dir)
    payload = experiment.status(root)
    path = _recovery_path(root)
    if path.is_file():
        payload["runtime_recovery"] = lifecycle._read_signed(path, kind=RECOVERY_KIND)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run-shared-pipeline", "status"))
    parser.add_argument("--output-dir", default=experiment.DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--shared-workers-per-gpu",
        type=int,
        default=DEFAULT_SHARED_WORKERS_PER_GPU,
    )
    parser.add_argument(
        "--idle-workers-per-gpu",
        type=int,
        default=DEFAULT_IDLE_WORKERS_PER_GPU,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run-shared-pipeline":
        print(
            run_shared_pipeline(
                args.output_dir,
                shared_workers_per_gpu=args.shared_workers_per_gpu,
                idle_workers_per_gpu=args.idle_workers_per_gpu,
            )
        )
    else:
        import json

        print(json.dumps(status(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "freeze_backbones",
    "freeze_evaluation",
    "run_shared_pipeline",
    "status",
]
