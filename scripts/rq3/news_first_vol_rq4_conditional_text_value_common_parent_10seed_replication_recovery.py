"""Audited resume path for the RQ4 common-parent replication.

The formal root froze its code before a checkpoint-freeze routing bug was
encountered.  The 40 trained parents are valid and must not be rewritten.  This
runner therefore keeps every frozen artifact intact, installs the already
tested stage-profile router only around checkpoint-freeze calls, and resumes
the remaining lifecycle with the benchmark-selected concurrency.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.rq3 import (
    news_first_vol_rq4_conditional_text_value_common_parent_10seed_replication as wrapper,
)

# Import these only after the replication wrapper has rebound the shared
# experiment module to the frozen RQ4 config, output root, and worker module.
from scripts.rq3 import (  # noqa: E402
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed as experiment,
)
from scripts.rq3 import (  # noqa: E402
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle as lifecycle,
)
from scripts.rq3 import (  # noqa: E402
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_recovery as freeze_recovery,
)
from scripts.rq3.news_first_vol_experiment_supervisor_helpers import (  # noqa: E402
    SupervisorLock,
    append_stage_journal,
)


RECOVERY_KIND = "rq4_common_parent_checkpoint_freeze_route_recovery_v1"
RECOVERY_PATH = "registry/runtime_recovery_manifest_v2.json"
RECOVERY_SOURCE = (
    "scripts/rq3/"
    "news_first_vol_rq4_conditional_text_value_common_parent_10seed_replication_recovery.py"
)
RECOVERY_SUPERVISOR_SOURCE = (
    "scripts/rq3/"
    "news_first_vol_rq4_conditional_text_value_common_parent_10seed_replication_recovery_supervisor.py"
)


def _root(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _manifest_entry(manifest_path: Path, source_path: Path) -> dict[str, Any]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if Path(str(row.get("path", ""))).resolve() == source_path.resolve()
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Frozen manifest has no unique row for {source_path}: {manifest_path}"
        )
    row = selected[0]
    observed_sha = experiment.sha256_file(source_path)
    observed_size = source_path.stat().st_size
    if observed_sha != str(row["sha256"]) or observed_size != int(row["size_bytes"]):
        raise ValueError(f"Frozen source/config drift: {source_path}")
    return {
        "artifact_role": str(row["artifact_role"]),
        "path": str(source_path.resolve()),
        "size_bytes": observed_size,
        "sha256": observed_sha,
    }


def _source_attestation(root: Path) -> dict[str, Any]:
    """Prove that only an unhashed execution-layer recovery was added."""

    frozen = freeze_recovery._frozen_main_attestation(root)
    code_manifest = root / "stages/backbones/code_hashes.csv"
    source_manifest = root / "registry/source_hashes.csv"
    wrapper_path = (experiment.REPO_ROOT / wrapper.WRAPPER_SOURCE).resolve()
    supervisor_path = (experiment.REPO_ROOT / wrapper.SUPERVISOR_SOURCE).resolve()
    config_path = (experiment.REPO_ROOT / wrapper.DEFAULT_CONFIG).resolve()
    recovery_path = (experiment.REPO_ROOT / RECOVERY_SOURCE).resolve()
    recovery_supervisor_path = (
        experiment.REPO_ROOT / RECOVERY_SUPERVISOR_SOURCE
    ).resolve()

    with code_manifest.open(newline="", encoding="utf-8") as handle:
        frozen_code_paths = {
            Path(str(row["path"])).resolve() for row in csv.DictReader(handle)
        }
    unexpected = [
        str(path)
        for path in (recovery_path, recovery_supervisor_path)
        if path in frozen_code_paths
    ]
    if unexpected:
        raise ValueError(f"Recovery files unexpectedly entered frozen code: {unexpected}")

    recovery_rows = []
    for role, path in (
        ("checkpoint_freeze_router", Path(frozen["recovery_source_path"])),
        ("rq4_recovery_runner", recovery_path),
        ("rq4_recovery_supervisor", recovery_supervisor_path),
    ):
        recovery_rows.append(
            {
                "artifact_role": role,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": experiment.sha256_file(path),
            }
        )
    return {
        "frozen_experiment": frozen,
        "frozen_replication_wrapper": _manifest_entry(code_manifest, wrapper_path),
        "frozen_replication_supervisor": _manifest_entry(
            code_manifest, supervisor_path
        ),
        "frozen_replication_config": _manifest_entry(source_manifest, config_path),
        "recovery_sources": recovery_rows,
        "frozen_code_manifest_path": str(code_manifest.resolve()),
        "frozen_code_manifest_sha256": experiment.sha256_file(code_manifest),
        "frozen_source_manifest_path": str(source_manifest.resolve()),
        "frozen_source_manifest_sha256": experiment.sha256_file(source_manifest),
        "frozen_sources_or_manifests_modified": False,
    }


def _benchmark_attestation(root: Path) -> dict[str, Any]:
    path = experiment._control_root(root) / "benchmark_result.json"
    payload = lifecycle._read_signed(path, kind=experiment.BENCHMARK_KIND)
    registry = experiment._read_registry(root)
    selected = int(payload.get("selected_workers_per_gpu", -1))
    formal = int(registry.get("formal_workers_per_gpu", -2))
    if payload.get("status") != "passed" or selected != formal:
        raise ValueError(
            "Passed benchmark concurrency does not match the frozen formal registry"
        )
    return {
        "path": str(path.resolve()),
        "sha256": experiment.sha256_file(path),
        "status": "passed",
        "selected_workers_per_gpu": selected,
        "formal_workers_per_gpu": formal,
    }


def _recovery_path(root: Path) -> Path:
    return root / RECOVERY_PATH


def _write_recovery_manifest(
    root: Path, *, status: str, details: Mapping[str, Any] | None = None
) -> Path:
    path = _recovery_path(root)
    previous: dict[str, Any] = {}
    if path.is_file():
        previous = lifecycle._read_signed(path, kind=RECOVERY_KIND)
    source_attestation = _source_attestation(root)
    benchmark_attestation = _benchmark_attestation(root)
    if previous and previous.get("source_attestation") != source_attestation:
        raise ValueError("Recovery or frozen-source attestation drift")
    if previous and previous.get("benchmark_attestation") != benchmark_attestation:
        raise ValueError("Recovery benchmark attestation drift")
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
                "checkpoint freeze reached the shared direct implementation "
                "without propagating the active stage runtime profile"
            ),
            "repair_scope": "checkpoint_freeze_runtime_profile_routing_only",
            "scientific_contract_changed": False,
            "parent_training_reused": True,
            "training_worker_module": experiment.WORKER_MODULE,
            "source_attestation": source_attestation,
            "benchmark_attestation": benchmark_attestation,
            "history": history,
            "status": str(status),
        },
    )


def _stage(root: Path, name: str, operation: Callable[[], Any]) -> Any:
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


def _freeze_backbones(root: Path) -> Path:
    with freeze_recovery._checkpoint_freeze_router(root):
        return experiment.freeze_backbones(root, resume=True)


def _freeze_evaluation(root: Path) -> Path:
    with freeze_recovery._checkpoint_freeze_router(root):
        return experiment.freeze_evaluation(root, resume=True)


def run_recovery(
    output_dir: str | Path = wrapper.DEFAULT_OUTPUT_DIR,
) -> Path:
    """Resume after parent training and finish the ordinary frozen lifecycle."""

    wrapper._configure()
    root = _root(output_dir)
    if (root / "qa.json").is_file():
        lifecycle.validate_terminal(root)
        return root
    experiment.validate_partial_root(root, verify_stage_roots=True)
    _source_attestation(root)
    _benchmark_attestation(root)
    control = experiment._control_root(root)
    with SupervisorLock(control, name="pipeline"):
        _write_recovery_manifest(
            root,
            status="running",
            details={"phase": "checkpoint_freeze_repair_start"},
        )
        try:
            config = experiment._load_frozen_config(root / "resolved_config.yaml")
            lifecycle._wait_for_resources(config, root)
            _stage(root, "freeze_backbones_repaired", lambda: _freeze_backbones(root))
            _stage(root, "graft", lambda: experiment.graft(root, resume=True))
            _stage(
                root,
                "launch_continuations",
                lambda: experiment.launch_continuations(root, resume=True),
            )
            _stage(root, "freeze_evaluation_repaired", lambda: _freeze_evaluation(root))
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
            # QA hashes the terminal output universe, so the recovery artifact
            # must become immutable immediately before that stage.
            _write_recovery_manifest(
                root,
                status="completed",
                details={"phase": "terminal_qa_next"},
            )
            _stage(root, "qa", lambda: lifecycle.qa(root, resume=True))
        except BaseException as exc:
            if not (root / "qa.json").is_file():
                _write_recovery_manifest(
                    root,
                    status="failed",
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            append_stage_journal(
                control,
                experiment_kind=experiment.EXPERIMENT_KIND,
                stage="rq4_recovery_pipeline",
                status="failed",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
            raise
    return root


def status(output_dir: str | Path = wrapper.DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _root(output_dir)
    payload = experiment.status(root)
    path = _recovery_path(root)
    if path.is_file():
        payload["runtime_recovery"] = lifecycle._read_signed(
            path, kind=RECOVERY_KIND
        )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run-pipeline", "status"))
    parser.add_argument("--output-dir", default=wrapper.DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run-pipeline":
        print(run_recovery(args.output_dir))
    else:
        import json

        print(json.dumps(status(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["run_recovery", "status", "main"]
