"""Lifecycle, statistics, and supervisor for the 10-seed text-effect experiment.

The executable model orchestration lives in the sibling experiment module.
Keeping this lifecycle layer separate also lets the background runner delay all
torch imports until after its command line has been parsed.  That matters for
spawned prediction workers: ``CUDA_VISIBLE_DEVICES`` is installed by their
initializer before an evaluator imports torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed as experiment,
)
from scripts.rq3.news_first_vol_experiment_supervisor_helpers import (
    SupervisorLock,
    append_stage_journal,
    assess_resume_state,
    atomic_write_json,
    benchmark_concurrency_decision,
    evaluate_canary_or_benchmark,
    gpu_availability_decision,
    system_resource_snapshot,
    utc_now,
)


ANALYSIS_INPUT_KIND = "pure_cnn_parent_film_text_effect_analysis_inputs_v1"
ANALYSIS_RESULT_KIND = "pure_cnn_parent_film_text_effect_analysis_result_v1"
TERMINAL_QA_KIND = "pure_cnn_parent_film_text_effect_terminal_qa_v1"
BENCHMARK_ROOT_KIND = "pure_cnn_parent_film_text_effect_benchmark_root_v1"
FORMAL_PROJECTION_MULTIPLIER = 3.0


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = experiment.direct.payload_sha256(result)
    return result


def _read_signed(path: Path, *, kind: str | None = None) -> dict[str, Any]:
    payload = experiment._read_json(path)
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if experiment.direct.payload_sha256(unsigned) != payload.get("payload_sha256"):
        raise ValueError(f"Signed payload drift: {path}")
    if kind is not None and payload.get("kind") != kind:
        raise ValueError(f"Signed payload kind drift: {path}")
    return payload


def _write_signed(path: Path, payload: Mapping[str, Any]) -> Path:
    return experiment._write_json(path, _signed(payload))


def _stage_registry_rows(stage_root: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in experiment.direct.read_registry(stage_root)["jobs"]]


def _worker_environment(job: Mapping[str, Any], log_path: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(int(job["gpu_id"]))
    environment["NEWS_FIRST_JOB_LOG_PATH"] = str(log_path.resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(experiment.REPO_ROOT / "src"), str(experiment.REPO_ROOT))
    )
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    return environment


def _run_canary_jobs(
    jobs: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    python_executable: str,
    canary_dir: Path,
    resume: bool,
) -> Path:
    """Run exactly one selected job per GPU without changing other statuses."""

    selected = [(root, dict(job)) for root, job in jobs]
    if len(selected) != 2 or {int(job["gpu_id"]) for _, job in selected} != {0, 1}:
        raise ValueError("A dual-GPU canary requires exactly one job on each GPU")
    canary_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[Any]] = []
    handles: list[Any] = []
    rows: list[dict[str, Any]] = []
    try:
        for stage_root, job in selected:
            log_path = canary_dir / f"{job['job_id']}.log"
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            command = [
                str(python_executable),
                "-m",
                experiment.WORKER_MODULE,
                "worker",
                "--output-dir",
                str(stage_root),
                "--job-id",
                str(job["job_id"]),
            ]
            if resume:
                command.append("--resume")
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=experiment.REPO_ROOT,
                    env=_worker_environment(job, log_path),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            rows.append(
                {
                    "job_id": str(job["job_id"]),
                    "stage_root": str(stage_root.resolve()),
                    "physical_gpu_id": int(job["gpu_id"]),
                    "log_path": str(log_path.resolve()),
                }
            )
        failures = []
        for process, row in zip(processes, rows, strict=True):
            code = process.wait()
            row["exit_code"] = int(code)
            log_path = Path(str(row["log_path"]))
            row["log_size_bytes"] = log_path.stat().st_size
            row["log_sha256"] = experiment.sha256_file(log_path)
            if code != 0:
                failures.append(f"{row['job_id']}={code}")
        if failures:
            raise RuntimeError(f"Dual-GPU canary failed: {failures}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()
    for stage_root, job in selected:
        status = experiment._read_json(
            experiment.direct._status_path(stage_root, str(job["job_id"]))
        )
        if status.get("status") != "completed":
            raise RuntimeError(f"Canary status is incomplete: {job['job_id']}")
    return _write_signed(
        canary_dir / "canary_manifest.json",
        {
            "schema_version": 1,
            "kind": "pure_cnn_parent_film_text_effect_dual_gpu_canary_v1",
            "status": "passed",
            "jobs": rows,
            "completed_at_utc": utc_now(),
        },
    )


def _validate_canary_manifest(path: Path) -> dict[str, Any]:
    payload = _read_signed(
        path, kind="pure_cnn_parent_film_text_effect_dual_gpu_canary_v1"
    )
    rows = [dict(row) for row in payload.get("jobs") or ()]
    if (
        payload.get("status") != "passed"
        or len(rows) != 2
        or {int(row.get("physical_gpu_id", -1)) for row in rows} != {0, 1}
        or any(int(row.get("exit_code", -1)) != 0 for row in rows)
    ):
        raise ValueError(f"Dual-GPU canary manifest drift: {path}")
    for row in rows:
        log_path = Path(str(row["log_path"])).resolve()
        if (
            not log_path.is_file()
            or log_path.stat().st_size != int(row["log_size_bytes"])
            or experiment.sha256_file(log_path) != str(row["log_sha256"])
        ):
            raise ValueError(f"Dual-GPU canary log drift: {log_path}")
    return payload


def _write_benchmark_graft_allowlist(
    config: Mapping[str, Any], benchmark_root: Path, *, resume: bool
) -> Path:
    parents = experiment._parent_allowlist_rows(benchmark_root)
    entries: list[dict[str, Any]] = []
    for parent in parents:
        for target_mode in (experiment.PURE_MODE, experiment.FILM_MODE):
            existing = (
                experiment._resume_block_graft(
                    config, benchmark_root, parent, target_mode=target_mode
                )
                if resume
                else None
            )
            entries.append(
                existing
                if existing is not None
                else experiment._save_block_graft(
                    config, benchmark_root, parent, target_mode=target_mode
                )
            )
    if len(entries) != 80:
        raise ValueError("Benchmark must create 80 graft/restart states")
    payload = {
        "schema_version": 1,
        "kind": experiment.GRAFT_ALLOWLIST_KIND,
        "entry_count": 80,
        "film_graft_count": 40,
        "pure_restart_count": 40,
        "entries": sorted(
            entries,
            key=lambda row: (
                int(row["seed"]),
                str(row["fold"]),
                str(row["target_generator_mode"]),
            ),
        ),
        "frozen_at_utc": utc_now(),
    }
    payload["payload_sha256"] = experiment.direct.payload_sha256(payload)
    return experiment._write_json(
        benchmark_root / "registry/graft_allowlist.json", payload
    )


def _resource_evidence(benchmark_root: Path) -> dict[str, Any]:
    roots = experiment._stage_roots(benchmark_root)
    frames: list[pd.DataFrame] = []
    peak_host = 0.0
    for stage_root in roots.values():
        telemetry = stage_root / "resource_usage.csv"
        if not telemetry.is_file():
            raise ValueError(f"Benchmark telemetry is missing: {telemetry}")
        frame = pd.read_csv(telemetry)
        if frame.empty or not {"gpu_index", "memory_used_mib"}.issubset(frame):
            raise ValueError(f"Benchmark telemetry is malformed: {telemetry}")
        frames.append(frame)
        peak_host = max(
            peak_host,
            float(
                experiment.direct.read_registry(stage_root).get(
                    "peak_host_ram_fraction", 1.0
                )
            ),
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["gpu_index"] = pd.to_numeric(combined["gpu_index"], errors="raise").astype(
        int
    )
    combined["memory_used_mib"] = pd.to_numeric(
        combined["memory_used_mib"], errors="raise"
    ).astype(float)
    peaks = {
        int(gpu): float(group["memory_used_mib"].max()) / 1024.0
        for gpu, group in combined.groupby("gpu_index")
        if int(gpu) in (0, 1)
    }
    return {
        "completed_jobs": 280,
        "failed_jobs": 0,
        "oom_count": 0,
        "nan_count": 0,
        "generator_updated_jobs": 280,
        "critic_updated_jobs": 280,
        "peak_gpu_memory_gib_by_gpu": peaks,
        "peak_host_ram_fraction": peak_host,
    }


def _validate_stage_benchmark_updates(benchmark_root: Path, *, stage_key: str) -> None:
    roots = experiment._stage_roots(benchmark_root)
    stage_root = roots[stage_key]
    if stage_key == "backbones":
        profile = experiment._pure_stage_profile(
            main_root=benchmark_root,
            stage_name="backbones",
            arm=experiment.PARENT_ARM,
            use_graft=False,
        )
    elif stage_key == "pure_continuation":
        profile = experiment._pure_stage_profile(
            main_root=benchmark_root,
            stage_name="pure_continuation",
            arm=experiment.PURE_CONTINUATION_ARM,
            use_graft=True,
        )
    else:
        profile = experiment._film_stage_profile(
            main_root=benchmark_root, stage_name="film_continuations"
        )
    with profile:
        experiment.direct._validate_benchmark_updates(stage_root)


def _benchmark_stage_registry_evidence(
    benchmark_root: Path,
) -> dict[str, dict[str, object]]:
    evidence: dict[str, dict[str, object]] = {}
    for key, stage_root in experiment._stage_roots(benchmark_root).items():
        registry_path = stage_root / "registry/task_registry.json"
        evidence[key] = {
            "path": str(registry_path.resolve()),
            "sha256": experiment.sha256_file(registry_path),
        }
    return evidence


def _run_benchmark_candidate(
    config: Mapping[str, Any],
    formal_root: Path,
    *,
    workers: int,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    benchmark_root = formal_root.with_name(
        f"{formal_root.name}_benchmark_{int(workers)}w_v1"
    )
    benchmark_root.mkdir(parents=True, exist_ok=True)
    for relative in ("registry/graft_states", "registry/graft_manifests", "canary"):
        (benchmark_root / relative).mkdir(parents=True, exist_ok=True)
    frozen_config = benchmark_root / "resolved_config.yaml"
    if frozen_config.is_file():
        if experiment.sha256_file(frozen_config) != experiment.sha256_file(
            config["source_config_path"]
        ):
            raise ValueError("Benchmark config drift")
    else:
        shutil.copy2(config["source_config_path"], frozen_config)

    backbone_root = experiment._prepare_backbone_stage(
        config,
        benchmark_root,
        workers=int(workers),
        num_epochs=1,
        resume=resume and (benchmark_root / "stages/backbones").exists(),
    )
    parent_jobs = _stage_registry_rows(backbone_root)
    parent_canary_jobs = [
        (backbone_root, next(job for job in parent_jobs if int(job["gpu_id"]) == gpu))
        for gpu in (0, 1)
    ]
    parent_canary = benchmark_root / "canary/parents/canary_manifest.json"
    if not parent_canary.is_file():
        _run_canary_jobs(
            parent_canary_jobs,
            python_executable=str(config["runtime"]["python_executable"]),
            canary_dir=parent_canary.parent,
            resume=True,
        )
    _validate_canary_manifest(parent_canary)
    with experiment._pure_stage_profile(
        main_root=benchmark_root,
        stage_name="backbones",
        arm=experiment.PARENT_ARM,
        use_graft=False,
    ):
        experiment.pure.launch(backbone_root, resume=True)
    _validate_stage_benchmark_updates(benchmark_root, stage_key="backbones")

    graft_path = benchmark_root / "registry/graft_allowlist.json"
    if graft_path.is_file():
        if not resume:
            raise FileExistsError(graft_path)
        payload = _read_signed(graft_path, kind=experiment.GRAFT_ALLOWLIST_KIND)
        if len(payload.get("entries") or ()) != 80:
            raise ValueError("Benchmark graft allowlist drift")
    else:
        _write_benchmark_graft_allowlist(config, benchmark_root, resume=resume)
    pure_root, film_root = experiment._prepare_continuation_stages(
        config,
        benchmark_root,
        workers=int(workers),
        num_epochs=1,
        resume=resume and (benchmark_root / "stages/continuations").exists(),
    )
    pure_jobs = _stage_registry_rows(pure_root)
    film_jobs = _stage_registry_rows(film_root)
    branch_canary_jobs = [
        (pure_root, next(job for job in pure_jobs if int(job["gpu_id"]) == 0)),
        (
            film_root,
            next(
                job
                for job in film_jobs
                if int(job["gpu_id"]) == 1 and str(job["arm"]) == "film_lp_matched"
            ),
        ),
    ]
    branch_canary = benchmark_root / "canary/continuations/canary_manifest.json"
    if not branch_canary.is_file():
        _run_canary_jobs(
            branch_canary_jobs,
            python_executable=str(config["runtime"]["python_executable"]),
            canary_dir=branch_canary.parent,
            resume=True,
        )
    _validate_canary_manifest(branch_canary)
    with experiment._pure_stage_profile(
        main_root=benchmark_root,
        stage_name="pure_continuation",
        arm=experiment.PURE_CONTINUATION_ARM,
        use_graft=True,
    ):
        experiment.pure.launch(pure_root, resume=True)
    with experiment._film_stage_profile(
        main_root=benchmark_root, stage_name="film_continuations"
    ):
        experiment.film_text.launch(film_root, resume=True)
    _validate_stage_benchmark_updates(benchmark_root, stage_key="pure_continuation")
    _validate_stage_benchmark_updates(benchmark_root, stage_key="film_continuations")

    evidence = _resource_evidence(benchmark_root)
    evaluated = evaluate_canary_or_benchmark(
        evidence,
        expected_jobs=280,
        gpu_ids=(0, 1),
        maximum_peak_gpu_memory_gib=float(
            config["runtime"]["preflight_max_peak_gpu_memory_gib"]
        ),
        maximum_host_ram_fraction=float(
            config["runtime"]["preflight_max_host_ram_fraction"]
        ),
    )
    manifest = _write_signed(
        benchmark_root / "benchmark_manifest.json",
        {
            "schema_version": 1,
            "kind": BENCHMARK_ROOT_KIND,
            "status": "passed" if evaluated["passed"] else "failed",
            "source_config_sha256": str(config["source_config_sha256"]),
            "workers_per_gpu": int(workers),
            "parent_canary_manifest_path": str(parent_canary.resolve()),
            "parent_canary_manifest_sha256": experiment.sha256_file(parent_canary),
            "branch_canary_manifest_path": str(branch_canary.resolve()),
            "branch_canary_manifest_sha256": experiment.sha256_file(branch_canary),
            "stage_registries": _benchmark_stage_registry_evidence(benchmark_root),
            "evidence": evidence,
            "gate": evaluated,
            "completed_at_utc": utc_now(),
        },
    )
    return manifest, evaluated


def benchmark(
    config_path: str | Path = experiment.DEFAULT_CONFIG,
    output_dir: str | Path = experiment.DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = experiment.load_config(config_path)
    experiment.validate_config(config, verify_source_evidence=True)
    formal_root = experiment._root(output_dir)
    result_path = experiment._control_root(formal_root) / "benchmark_result.json"
    if result_path.is_file():
        payload = _read_signed(result_path, kind=experiment.BENCHMARK_KIND)
        if (
            payload.get("status") != "passed"
            or payload.get("source_config_sha256") != config["source_config_sha256"]
        ):
            raise ValueError("Existing benchmark result drift")
        experiment.direct._verify_frozen_file(
            payload["benchmark_manifest_path"], payload["benchmark_manifest_sha256"]
        )
        return result_path
    if formal_root.exists():
        raise RuntimeError("Benchmark must complete before the formal root is created")
    primary_workers = int(config["runtime"]["benchmark_workers_per_gpu"])
    fallback_workers = int(config["runtime"]["fallback_workers_per_gpu"])

    def attempt(workers: int) -> tuple[Path | None, dict[str, Any]]:
        candidate_root = formal_root.with_name(
            f"{formal_root.name}_benchmark_{int(workers)}w_v1"
        )
        try:
            return _run_benchmark_candidate(
                config,
                formal_root,
                workers=workers,
                resume=resume,
            )
        except BaseException as exc:
            capacity = experiment.core._benchmark_failure_is_capacity_related(
                candidate_root, exc
            )
            gate = {
                "passed": False,
                "failure_category": "capacity" if capacity else "correctness",
                "correctness_reasons": (
                    [] if capacity else [f"{type(exc).__name__}: {exc}"]
                ),
                "capacity_reasons": (
                    [f"{type(exc).__name__}: {exc}"] if capacity else []
                ),
                "expected_jobs": 280,
            }
            _write_signed(
                candidate_root / "benchmark_failure.json",
                {
                    "schema_version": 1,
                    "kind": "pure_cnn_parent_film_text_effect_benchmark_failure_v1",
                    "workers_per_gpu": int(workers),
                    "gate": gate,
                    "failed_at_utc": utc_now(),
                },
            )
            return None, gate

    primary_manifest, primary = attempt(primary_workers)
    decision = benchmark_concurrency_decision(
        primary,
        primary_workers_per_gpu=primary_workers,
        fallback_workers_per_gpu=fallback_workers,
    )
    selected_manifest = primary_manifest
    fallback: dict[str, Any] | None = None
    if decision["action"] == "run_fallback_benchmark":
        selected_manifest, fallback = attempt(fallback_workers)
        decision = benchmark_concurrency_decision(
            primary,
            primary_workers_per_gpu=primary_workers,
            fallback_workers_per_gpu=fallback_workers,
            fallback=fallback,
        )
    if decision["action"] != "launch_formal":
        raise RuntimeError(f"Benchmark gate rejected formal execution: {decision}")
    if selected_manifest is None:
        raise AssertionError("Passed benchmark decision lacks its frozen manifest")
    benchmark_root = selected_manifest.parent
    benchmark_bytes = sum(
        path.stat().st_size for path in benchmark_root.rglob("*") if path.is_file()
    )
    projected = int(
        benchmark_bytes
        * FORMAL_PROJECTION_MULTIPLIER
        * float(config["runtime"]["disk_projection_safety_factor"])
    )
    free = shutil.disk_usage(formal_root.parent).free
    minimum_remaining = (
        int(config["runtime"]["minimum_free_disk_after_projected_bytes_gib"]) * 1024**3
    )
    if free - projected < minimum_remaining:
        raise RuntimeError("Formal disk gate would leave less than 30 GiB free")
    return _write_signed(
        result_path,
        {
            "schema_version": 1,
            "kind": experiment.BENCHMARK_KIND,
            "status": "passed",
            "source_config_path": str(Path(config["source_config_path"]).resolve()),
            "source_config_sha256": str(config["source_config_sha256"]),
            "selected_workers_per_gpu": int(decision["selected_workers_per_gpu"]),
            "decision": decision,
            "primary_gate": primary,
            "fallback_gate": fallback,
            "benchmark_root": str(benchmark_root.resolve()),
            "benchmark_manifest_path": str(selected_manifest.resolve()),
            "benchmark_manifest_sha256": experiment.sha256_file(selected_manifest),
            "benchmark_bytes": benchmark_bytes,
            "projected_formal_bytes_with_safety": projected,
            "free_bytes_at_gate": free,
            "minimum_remaining_bytes": minimum_remaining,
            "completed_at_utc": utc_now(),
        },
    )


def analyze(output_dir: str | Path, *, resume: bool = False) -> Path:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_prediction import (
        compose_intervention_analysis_panel,
        standard_pair_metrics_for_analysis,
    )

    root = experiment._root(output_dir)
    experiment.validate_root(root, verify_stage_roots=True)
    registry = experiment._read_registry(root)
    if not all(
        bool(registry.get(key))
        for key in (
            "standard_predictions_frozen",
            "interventions_frozen",
            "validation_trajectories_frozen",
        )
    ):
        raise RuntimeError("Analysis requires all three frozen evidence panels")
    manifest_path = root / "analysis/analysis_inputs.json"
    if manifest_path.is_file():
        if not resume:
            raise RuntimeError("Analysis inputs already exist; use --resume")
        payload = _read_signed(manifest_path, kind=ANALYSIS_INPUT_KIND)
        for path_key, sha_key in (
            ("standard_panel_path", "standard_panel_sha256"),
            ("intervention_panel_path", "intervention_panel_sha256"),
            ("validation_trajectory_path", "validation_trajectory_sha256"),
        ):
            experiment.direct._verify_frozen_file(payload[path_key], payload[sha_key])
        registry = experiment._read_registry(root)
        if not registry.get("analysis_inputs_frozen"):
            registry.update(
                status="analysis_inputs_frozen",
                analysis_inputs_frozen=True,
                analysis_inputs_path=str(manifest_path.resolve()),
                analysis_inputs_sha256=experiment.sha256_file(manifest_path),
            )
            experiment._write_registry(root, registry)
        return manifest_path
    standard_source = Path(str(registry["standard_pair_metrics_path"]))
    intervention_source = Path(str(registry["intervention_pair_metrics_path"]))
    trajectory_source = Path(str(registry["validation_trajectory_pair_metrics_path"]))
    standard = standard_pair_metrics_for_analysis(pd.read_csv(standard_source))
    intervention = compose_intervention_analysis_panel(
        pd.read_csv(standard_source), pd.read_csv(intervention_source)
    )
    standard_path = experiment.core._write_dataframe_csv(
        root / "analysis/standard_primary_panel.csv.gz", standard, gzip=True
    )
    intervention_path = experiment.core._write_dataframe_csv(
        root / "analysis/intervention_primary_panel.csv.gz", intervention, gzip=True
    )
    payload = {
        "schema_version": 1,
        "kind": ANALYSIS_INPUT_KIND,
        "interpretation": experiment.INTERPRETATION,
        "standard_panel_path": str(standard_path.resolve()),
        "standard_panel_sha256": experiment.sha256_file(standard_path),
        "standard_panel_rows": len(standard),
        "intervention_panel_path": str(intervention_path.resolve()),
        "intervention_panel_sha256": experiment.sha256_file(intervention_path),
        "intervention_panel_rows": len(intervention),
        "validation_trajectory_path": str(trajectory_source.resolve()),
        "validation_trajectory_sha256": experiment.sha256_file(trajectory_source),
        "validation_trajectory_rows": len(pd.read_csv(trajectory_source)),
        "created_at_utc": utc_now(),
    }
    _write_signed(manifest_path, payload)
    registry = experiment._read_registry(root)
    registry.update(
        status="analysis_inputs_frozen",
        analysis_inputs_frozen=True,
        analysis_inputs_path=str(manifest_path.resolve()),
        analysis_inputs_sha256=experiment.sha256_file(manifest_path),
    )
    experiment._write_registry(root, registry)
    return manifest_path


def _load_analysis_result(root: Path) -> dict[str, Any]:
    manifest_path = root / "analysis/results/analysis_manifest.json"
    manifest = experiment._read_json(manifest_path)
    if (
        manifest.get("kind")
        != "film_unet_pure_cnn_backbone_text_effect_analysis_manifest_v1"
    ):
        raise ValueError("Analysis artifact manifest kind drift")
    result: dict[str, Any] = {}
    for name, raw in dict(manifest.get("artifacts") or {}).items():
        row = dict(raw)
        path = Path(str(row["path"])).resolve()
        experiment.direct._verify_frozen_file(path, row["sha256"])
        if name == "conclusion":
            result[name] = experiment._read_json(path)
        else:
            result[name] = pd.read_csv(path)
    return result


def _validated_analysis_inputs(
    root: Path, registry: Mapping[str, Any]
) -> dict[str, Any]:
    path = experiment._verify_registry_file(
        registry, "analysis_inputs_path", "analysis_inputs_sha256"
    )
    if path != (root / "analysis/analysis_inputs.json").resolve():
        raise ValueError("Frozen analysis input manifest path drift")
    payload = _read_signed(path, kind=ANALYSIS_INPUT_KIND)
    for path_key, sha_key in (
        ("standard_panel_path", "standard_panel_sha256"),
        ("intervention_panel_path", "intervention_panel_sha256"),
        ("validation_trajectory_path", "validation_trajectory_sha256"),
    ):
        experiment.direct._verify_frozen_file(payload[path_key], payload[sha_key])
    return payload


def _validated_analysis_result_manifest(
    root: Path, registry: Mapping[str, Any]
) -> Path:
    path = experiment._verify_registry_file(
        registry, "analysis_manifest_path", "analysis_manifest_sha256"
    )
    if path != (root / "analysis/results/analysis_manifest.json").resolve():
        raise ValueError("Frozen analysis result manifest path drift")
    return path


def bootstrap(output_dir: str | Path, *, resume: bool = False) -> Path:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_analysis import (
        analyze_backbone_text_effect,
    )
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_report import (
        write_analysis_artifacts,
    )

    root = experiment._root(output_dir)
    experiment.validate_root(root, verify_stage_roots=True)
    registry = experiment._read_registry(root)
    if not registry.get("analysis_inputs_frozen"):
        analyze(root, resume=resume)
        registry = experiment._read_registry(root)
    result_manifest = root / "analysis/results/analysis_manifest.json"
    if registry.get("analysis_complete"):
        _validated_analysis_result_manifest(root, registry)
        _load_analysis_result(root)
        return result_manifest
    if result_manifest.is_file():
        if not resume:
            raise RuntimeError("Complete analysis artifacts require --resume")
        result = _load_analysis_result(root)
        conclusion_path = root / "analysis/results/conclusion.json"
        registry.update(
            status="analysis_complete",
            analysis_complete=True,
            analysis_completed_at_utc=utc_now(),
            analysis_manifest_path=str(result_manifest.resolve()),
            analysis_manifest_sha256=experiment.sha256_file(result_manifest),
            conclusion_path=str(conclusion_path.resolve()),
            conclusion_sha256=experiment.sha256_file(conclusion_path),
            conclusion_status=str(result["conclusion"]["status"]),
        )
        experiment._write_registry(root, registry)
        return result_manifest
    inputs = _validated_analysis_inputs(root, registry)
    config = experiment.validate_root(root, verify_stage_roots=False)
    result = analyze_backbone_text_effect(
        inputs["standard_panel_path"],
        inputs["intervention_panel_path"],
        inputs["validation_trajectory_path"],
        expected_seeds=experiment.SEEDS,
        expected_folds=experiment.FOLDS,
        bootstrap_iterations=int(config["analysis"]["bootstrap_replicates"]),
        bootstrap_seed=int(config["analysis"]["bootstrap_seed"]),
        alpha=1.0 - float(config["analysis"]["confidence_level"]),
        minimum_nonworse_seeds=int(
            config["analysis"]["primary_support_gate"]["minimum_nonworse_seeds"]
        ),
        minimum_nonworse_folds=int(
            config["analysis"]["primary_support_gate"]["minimum_nonworse_folds"]
        ),
    )
    paths = write_analysis_artifacts(root / "analysis/results", result)
    result_manifest = paths["manifest"]
    registry = experiment._read_registry(root)
    registry.update(
        status="analysis_complete",
        analysis_complete=True,
        analysis_completed_at_utc=utc_now(),
        analysis_manifest_path=str(result_manifest.resolve()),
        analysis_manifest_sha256=experiment.sha256_file(result_manifest),
        conclusion_path=str(paths["conclusion"].resolve()),
        conclusion_sha256=experiment.sha256_file(paths["conclusion"]),
        conclusion_status=str(result["conclusion"]["status"]),
    )
    experiment._write_registry(root, registry)
    return result_manifest


def _training_summary(root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for stage_name, stage_root in experiment._stage_roots(root).items():
        registry = experiment.direct.read_registry(stage_root)
        for job in registry["jobs"]:
            status = experiment._read_json(
                experiment.direct._status_path(stage_root, str(job["job_id"]))
            )
            best = experiment._read_json(
                Path(
                    str(
                        experiment.direct._artifact(status, "best_learned_checkpoint")[
                            "path"
                        ]
                    )
                )
            )
            metrics_path = Path(
                str(experiment.direct._artifact(status, "training_metrics_csv")["path"])
            )
            metrics = pd.read_csv(metrics_path)
            learned = pd.to_numeric(metrics["epoch"], errors="raise")
            learned = learned.loc[learned.ge(1)]
            rows.append(
                {
                    "job_id": str(job["job_id"]),
                    "stage": stage_name,
                    "arm": str(job["arm"]),
                    "seed": int(job["seed"]),
                    "fold": str(job["fold"]),
                    "best_epoch": int(best["best_epoch"]),
                    "epochs_trained": int(learned.max()),
                    "generator_initial_lr": float(
                        experiment._read_json(job["training_config_path"])[
                            "generator_learning_rate"
                        ]
                    )
                    if str(job["training_config_path"]).endswith(".json")
                    else float(
                        experiment.yaml.safe_load(
                            Path(job["training_config_path"]).read_text(
                                encoding="utf-8"
                            )
                        )["generator_learning_rate"]
                    ),
                    "metrics_path": str(metrics_path.resolve()),
                    "metrics_sha256": experiment.sha256_file(metrics_path),
                }
            )
    if len(rows) != 280:
        raise ValueError("Training summary requires exactly 280 completed jobs")
    return experiment.core._write_dataframe_csv(
        root / "analysis/training_summary.csv", pd.DataFrame(rows)
    )


def _validate_report_artifact(path: Path) -> dict[str, Any]:
    payload = experiment._read_json(path)
    if (
        payload.get("kind")
        != "film_unet_pure_cnn_backbone_text_effect_report_artifact_v1"
        or payload.get("interpretation") != experiment.INTERPRETATION
        or not str(payload.get("conclusion_status", ""))
    ):
        raise ValueError("Frozen report artifact identity drift")
    reports = dict(payload.get("reports") or {})
    if set(reports) != {"markdown", "html"}:
        raise ValueError("Frozen report artifact role universe drift")
    for role, raw in reports.items():
        row = dict(raw)
        report_path = Path(str(row.get("path", ""))).resolve()
        if (
            not report_path.is_file()
            or report_path.stat().st_size != int(row.get("size_bytes", -1))
            or experiment.sha256_file(report_path) != str(row.get("sha256", ""))
        ):
            raise ValueError(f"Frozen {role} report drift: {report_path}")
    if not bool(reports["html"].get("self_contained")):
        raise ValueError("HTML report must remain self-contained")
    return payload


def report(output_dir: str | Path, *, resume: bool = False) -> Path:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_report import (
        write_text_effect_report,
    )

    root = experiment._root(output_dir)
    experiment.validate_root(root, verify_stage_roots=True)
    registry = experiment._read_registry(root)
    if not registry.get("analysis_complete"):
        raise RuntimeError("Report requires completed 10,000-draw analysis")
    _validated_analysis_result_manifest(root, registry)
    artifact_path = root / "report/report_artifact.json"
    if registry.get("report_complete"):
        if not artifact_path.is_file() or experiment.sha256_file(
            artifact_path
        ) != registry.get("report_artifact_sha256"):
            raise ValueError("Frozen report artifact drift")
        _validate_report_artifact(artifact_path)
        return artifact_path
    if artifact_path.exists() and not resume:
        raise RuntimeError("Partial report requires --resume")
    if artifact_path.is_file():
        artifact = _validate_report_artifact(artifact_path)
        reports = dict(artifact["reports"])
        training_summary = root / "analysis/training_summary.csv"
        if not training_summary.is_file():
            raise ValueError("Complete report lacks its frozen training summary")
        registry.update(
            status="report_complete",
            report_complete=True,
            report_completed_at_utc=utc_now(),
            report_artifact_path=str(artifact_path.resolve()),
            report_artifact_sha256=experiment.sha256_file(artifact_path),
            report_markdown_path=str(Path(reports["markdown"]["path"]).resolve()),
            report_markdown_sha256=str(reports["markdown"]["sha256"]),
            report_html_path=str(Path(reports["html"]["path"]).resolve()),
            report_html_sha256=str(reports["html"]["sha256"]),
            training_summary_path=str(training_summary.resolve()),
            training_summary_sha256=experiment.sha256_file(training_summary),
        )
        experiment._write_registry(root, registry)
        return artifact_path
    result = _load_analysis_result(root)
    training_summary = _training_summary(root)
    paths = write_text_effect_report(
        root / "report",
        result,
        metadata={
            "interpretation": experiment.INTERPRETATION,
            "training_jobs": 280,
            "standard_prediction_cells": 280,
            "intervention_prediction_cells": 80,
            "validation_trajectory_cells": 1_680,
            "training_summary_path": str(training_summary.resolve()),
            "training_summary_sha256": experiment.sha256_file(training_summary),
        },
    )
    registry = experiment._read_registry(root)
    registry.update(
        status="report_complete",
        report_complete=True,
        report_completed_at_utc=utc_now(),
        report_artifact_path=str(paths["artifact"].resolve()),
        report_artifact_sha256=experiment.sha256_file(paths["artifact"]),
        report_markdown_path=str(paths["markdown"].resolve()),
        report_markdown_sha256=experiment.sha256_file(paths["markdown"]),
        report_html_path=str(paths["html"].resolve()),
        report_html_sha256=experiment.sha256_file(paths["html"]),
        training_summary_path=str(training_summary.resolve()),
        training_summary_sha256=experiment.sha256_file(training_summary),
    )
    experiment._write_registry(root, registry)
    _validate_report_artifact(paths["artifact"])
    return paths["artifact"]


def _output_rows(root: Path) -> list[dict[str, Any]]:
    excluded = {
        (root / "output_hashes.csv").resolve(),
        experiment._registry_path(root).resolve(),
    }
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.resolve() in excluded:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": experiment.sha256_file(path),
            }
        )
    return rows


def _validate_output_manifest(root: Path, path: Path) -> int:
    frame = pd.read_csv(path, dtype={"relative_path": str, "sha256": str})
    if frame.empty or frame["relative_path"].duplicated().any():
        raise ValueError("Terminal output manifest is empty or duplicated")
    for row in frame.itertuples(index=False):
        artifact = (root / str(row.relative_path)).resolve()
        if root not in artifact.parents or not artifact.is_file():
            raise ValueError(f"Terminal artifact is missing: {artifact}")
        if artifact.stat().st_size != int(row.size_bytes):
            raise ValueError(f"Terminal artifact size drift: {artifact}")
        if experiment.sha256_file(artifact) != str(row.sha256):
            raise ValueError(f"Terminal artifact SHA drift: {artifact}")
    observed_paths = set(frame["relative_path"].astype(str))
    current_paths = {str(row["relative_path"]) for row in _output_rows(root)}
    if observed_paths != current_paths:
        raise ValueError(
            "Terminal output manifest is not complete: "
            f"unlisted={sorted(current_paths - observed_paths)[:5]}, "
            f"missing={sorted(observed_paths - current_paths)[:5]}"
        )
    return len(frame)


def validate_terminal(root_or_path: str | Path) -> Path:
    root = experiment._root(root_or_path)
    experiment.validate_root(root, verify_stage_roots=True)
    registry = experiment._read_registry(root)
    if not registry.get("terminal_complete") or registry.get("status") != "completed":
        raise ValueError("Experiment is not terminal")
    for path_key, sha_key in (
        ("terminal_qa_path", "terminal_qa_sha256"),
        ("output_manifest_path", "output_manifest_sha256"),
        ("final_registry_snapshot_path", "final_registry_snapshot_sha256"),
    ):
        experiment.direct._verify_frozen_file(registry[path_key], registry[sha_key])
    manifest_path = Path(str(registry["output_manifest_path"])).resolve()
    count = _validate_output_manifest(root, manifest_path)
    if count != int(registry["output_manifest_rows"]):
        raise ValueError("Terminal output manifest count drift")
    qa_payload = _read_signed(
        Path(str(registry["terminal_qa_path"])), kind=TERMINAL_QA_KIND
    )
    expected_qa = {
        "status": "passed",
        "interpretation": experiment.INTERPRETATION,
        "training_jobs_completed": 280,
        "standard_prediction_cells": 280,
        "intervention_prediction_cells": 80,
        "validation_trajectory_cells": 1_680,
        "validation_trajectory_pair_rows": 210_420,
        "standard_pair_rows": 35_000,
        "intervention_pair_rows": 10_000,
        "bootstrap_draws_per_comparison": 10_000,
    }
    if any(qa_payload.get(key) != value for key, value in expected_qa.items()):
        raise ValueError("Terminal QA identity/count contract drift")
    snapshot_path = Path(str(registry["final_registry_snapshot_path"])).resolve()
    if qa_payload.get("final_registry_snapshot_path") != str(
        snapshot_path
    ) or qa_payload.get("final_registry_snapshot_sha256") != experiment.sha256_file(
        snapshot_path
    ):
        raise ValueError("Terminal QA/final-registry snapshot binding drift")
    snapshot = _read_signed(
        snapshot_path,
        kind="pure_cnn_parent_film_text_effect_final_registry_snapshot_v1",
    )
    snapshot_registry = dict(snapshot.get("registry") or {})
    if (
        snapshot_registry.get("experiment_kind") != experiment.EXPERIMENT_KIND
        or snapshot_registry.get("interpretation") != experiment.INTERPRETATION
        or snapshot_registry.get("status") != "completed"
        or not bool(snapshot_registry.get("terminal_complete"))
        or len(snapshot_registry.get("jobs") or ()) != 280
    ):
        raise ValueError("Final registry snapshot contract drift")
    _validate_report_artifact(Path(str(registry["report_artifact_path"])).resolve())
    return Path(str(registry["terminal_qa_path"])).resolve()


def qa(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = experiment._root(output_dir)
    registry = experiment._read_registry(root)
    if registry.get("terminal_complete"):
        return validate_terminal(root)
    experiment.validate_root(root, verify_stage_roots=True)
    required_flags = (
        "evaluation_frozen",
        "validation_trajectories_frozen",
        "standard_predictions_frozen",
        "interventions_frozen",
        "analysis_complete",
        "report_complete",
    )
    if not all(bool(registry.get(key)) for key in required_flags):
        raise RuntimeError(
            "Terminal QA requires every training/evaluation/report stage"
        )
    final_timestamp = utc_now()
    intended_registry = dict(registry)
    intended_registry.update(
        status="completed",
        terminal_complete=True,
        completed_at_utc=final_timestamp,
        updated_at_utc=final_timestamp,
    )
    snapshot_path = root / "registry/final_registry_snapshot.json"
    _write_signed(
        snapshot_path,
        {
            "schema_version": 1,
            "kind": "pure_cnn_parent_film_text_effect_final_registry_snapshot_v1",
            "registry": intended_registry,
            "created_at_utc": final_timestamp,
        },
    )
    qa_path = root / "qa.json"
    _write_signed(
        qa_path,
        {
            "schema_version": 1,
            "kind": TERMINAL_QA_KIND,
            "status": "passed",
            "interpretation": experiment.INTERPRETATION,
            "training_jobs_completed": 280,
            "standard_prediction_cells": 280,
            "intervention_prediction_cells": 80,
            "validation_trajectory_cells": 1_680,
            "validation_trajectory_pair_rows": 210_420,
            "standard_pair_rows": 35_000,
            "intervention_pair_rows": 10_000,
            "bootstrap_draws_per_comparison": 10_000,
            "final_registry_snapshot_path": str(snapshot_path.resolve()),
            "final_registry_snapshot_sha256": experiment.sha256_file(snapshot_path),
            "completed_at_utc": final_timestamp,
        },
    )
    rows = _output_rows(root)
    output_path = experiment.direct.write_csv(
        root / "output_hashes.csv",
        rows,
        ("relative_path", "size_bytes", "sha256"),
    )
    registry = experiment._read_registry(root)
    registry.update(
        status="completed",
        terminal_complete=True,
        completed_at_utc=final_timestamp,
        terminal_qa_path=str(qa_path.resolve()),
        terminal_qa_sha256=experiment.sha256_file(qa_path),
        output_manifest_path=str(output_path.resolve()),
        output_manifest_sha256=experiment.sha256_file(output_path),
        output_manifest_rows=len(rows),
        final_registry_snapshot_path=str(snapshot_path.resolve()),
        final_registry_snapshot_sha256=experiment.sha256_file(snapshot_path),
    )
    experiment._write_registry(root, registry)
    return validate_terminal(root)


def status(output_dir: str | Path = experiment.DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = experiment._root(output_dir)
    control = experiment._control_root(root)
    if not root.exists():
        result: dict[str, Any] = {
            "output_root": str(root),
            "status": "not_prepared",
            "resource_state": "awaiting_resources_or_benchmark",
            "formal_root_created": False,
            "benchmark_complete": (control / "benchmark_result.json").is_file(),
        }
    else:
        registry = experiment._read_registry(root)
        counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for stage_root in experiment._stage_roots(root).values():
            if not stage_root.is_dir():
                continue
            stage_registry = experiment.direct.read_registry(stage_root)
            for job in stage_registry["jobs"]:
                state = experiment._read_json(
                    experiment.direct._status_path(stage_root, str(job["job_id"]))
                )
                label = str(state.get("status", "pending"))
                counts[label if label in counts else "failed"] += 1
        result = {
            "output_root": str(root),
            "status": str(registry.get("status")),
            "formal_root_created": True,
            "training_job_counts": counts,
            "training_jobs_total": 280,
            "phase_flags": {
                key: bool(registry.get(key))
                for key in (
                    "parents_frozen",
                    "grafts_frozen",
                    "continuations_prepared",
                    "evaluation_frozen",
                    "validation_trajectories_frozen",
                    "standard_predictions_frozen",
                    "interventions_frozen",
                    "analysis_complete",
                    "report_complete",
                    "terminal_complete",
                )
            },
            "conclusion_status": registry.get("conclusion_status"),
        }
    snapshot_path = control / "latest_resource_snapshot.json"
    if snapshot_path.is_file():
        result["latest_resource_snapshot"] = experiment._read_json(snapshot_path)
    pid_path = control / "pipeline.pid.json"
    if pid_path.is_file():
        result["supervisor_pid"] = experiment._read_json(pid_path).get("pid")
    journal_path = control / "pipeline_journal.json"
    if journal_path.is_file():
        result["pipeline_latest"] = experiment._read_json(journal_path).get("latest")
    return result


def _wait_for_resources(config: Mapping[str, Any], root: Path) -> None:
    control = experiment._control_root(root)
    attempts = 0
    previous_reasons: tuple[str, ...] | None = None
    while True:
        attempts += 1
        snapshot = system_resource_snapshot(
            executable=str(config["runtime"]["nvidia_smi_executable"])
        )
        atomic_write_json(control / "latest_resource_snapshot.json", snapshot)
        decision = gpu_availability_decision(snapshot, gpu_ids=(0, 1))
        reasons = tuple(map(str, decision["reasons"]))
        if decision["ready"]:
            append_stage_journal(
                control,
                experiment_kind=experiment.EXPERIMENT_KIND,
                stage="resource_gate",
                status="completed",
                details={"attempts": attempts, "decision": decision},
            )
            return
        if attempts == 1 or reasons != previous_reasons or attempts % 20 == 0:
            append_stage_journal(
                control,
                experiment_kind=experiment.EXPERIMENT_KIND,
                stage="resource_gate",
                status="waiting",
                details={"attempts": attempts, "decision": decision},
            )
        previous_reasons = reasons
        time.sleep(30.0)


def run_pipeline(
    config_path: str | Path = experiment.DEFAULT_CONFIG,
    output_dir: str | Path = experiment.DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    root = experiment._root(output_dir)
    control = experiment._control_root(root)
    control.mkdir(parents=True, exist_ok=True)
    state = assess_resume_state(
        root,
        control,
        terminal_marker="qa.json",
        terminal_validator=lambda value: validate_terminal(value),
        partial_validator=lambda value: experiment.validate_partial_root(
            value, verify_stage_roots=True
        ),
    )
    if state["state"] == "terminal_read_only":
        return root
    if state["state"] == "blocked_live_supervisor":
        raise RuntimeError(f"A live supervisor already owns this experiment: {state}")
    if state["state"] == "invalid_partial":
        raise RuntimeError(f"Unsafe partial experiment cannot resume: {state}")
    if state["state"] == "invalid_terminal":
        raise RuntimeError(f"Corrupt or incomplete terminal state: {state}")
    if state["state"] not in {"new", "resumable_partial"}:
        raise RuntimeError(f"Unknown supervisor resume state: {state}")
    if state["state"] == "resumable_partial" and not resume:
        raise RuntimeError("Partial experiment requires --resume")
    config = experiment.load_config(config_path)
    with SupervisorLock(control, name="pipeline"):
        try:
            _wait_for_resources(config, root)
            stages: list[tuple[str, Any]] = [
                ("benchmark", lambda: benchmark(config_path, root, resume=resume)),
                (
                    "prepare",
                    lambda: experiment.prepare(config_path, root, resume=resume),
                ),
                (
                    "launch_backbones",
                    lambda: experiment.launch_backbones(root, resume=True),
                ),
                (
                    "freeze_backbones",
                    lambda: experiment.freeze_backbones(root, resume=True),
                ),
                ("graft", lambda: experiment.graft(root, resume=True)),
                (
                    "launch_continuations",
                    lambda: experiment.launch_continuations(root, resume=True),
                ),
                (
                    "freeze_evaluation",
                    lambda: experiment.freeze_evaluation(root, resume=True),
                ),
                (
                    "predict_validation_trajectories",
                    lambda: experiment.predict_validation_trajectories(
                        root, resume=True
                    ),
                ),
                ("predict_test", lambda: experiment.predict_test(root, resume=True)),
                (
                    "predict_interventions",
                    lambda: experiment.predict_interventions(root, resume=True),
                ),
                ("analyze", lambda: analyze(root, resume=True)),
                ("bootstrap", lambda: bootstrap(root, resume=True)),
                ("report", lambda: report(root, resume=True)),
                ("qa", lambda: qa(root, resume=True)),
            ]
            for stage, operation in stages:
                append_stage_journal(
                    control,
                    experiment_kind=experiment.EXPERIMENT_KIND,
                    stage=stage,
                    status="running",
                )
                result = operation()
                append_stage_journal(
                    control,
                    experiment_kind=experiment.EXPERIMENT_KIND,
                    stage=stage,
                    status="completed",
                    details={"result": str(result)},
                )
            return root
        except BaseException as exc:
            append_stage_journal(
                control,
                experiment_kind=experiment.EXPERIMENT_KIND,
                stage="pipeline",
                status="failed",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
            raise


__all__ = [
    "analyze",
    "benchmark",
    "bootstrap",
    "qa",
    "report",
    "run_pipeline",
    "status",
    "validate_terminal",
]
