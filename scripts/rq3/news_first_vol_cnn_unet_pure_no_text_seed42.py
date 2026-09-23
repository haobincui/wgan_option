"""Pure-CNN U-Net no-text rolling experiment for seed 42.

This thin branch-local profile reuses the audited direct-training lifecycle but
replaces the FiLM Generator with ``cnn_unet_mask_coords_v1``.  The Generator
contains neither a text encoder nor FiLM projections and ignores the frozen
zero-valued text overlay.  Four rolling folds are trained independently and
all test inputs remain closed until the four best-learned checkpoints freeze.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from scripts.rq3 import (
    news_first_vol_cnn_unet_pure_no_text_seed42_analysis as pure_analysis,
)
from scripts.rq3 import news_first_vol_film_unet_direct_5arm_seed42 as direct


DEFAULT_CONFIG = "configs/rq3/news_first_vol_cnn_unet_pure_no_text_seed42.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq12_news_first_vol_cnn_unet_c32_nolp_pure_no_text_seed42_"
    "exact_ttm_rolling_v1"
)
EXPERIMENT_KIND = "rq12_news_first_vol_cnn_unet_c32_nolp_pure_no_text_seed42_rolling_v1"
INTERPRETATION = "retrospective_rolling_development_single_seed_descriptive"
DIRECT_ARMS = ("pure_cnn_no_text",)
NO_TEXT_ARM = "pure_cnn_no_text"
EXPECTED_TRAINING_JOBS = 4
EXPECTED_PREDICTION_CELLS = 4
EXPECTED_PAIR_METRIC_ROWS = 500
EXPECTED_PARAMETER_COUNTS = {
    "generator": 416_353,
    "critic": 729_157,
    "total": 1_145_510,
}
GENERATOR_MODE = "cnn_unet_mask_coords_v1"
CRITIC_MODE = "lp_disabled_same_shape_v1"
CAPACITY_PROFILE = "c32"
WORKER_MODULE = "scripts.rq3.news_first_vol_cnn_unet_pure_no_text_seed42"
INFERENCE_DETERMINISM_KIND = "cnn_unet_pure_no_text_inference_determinism_v1"
BENCHMARK_RESULT_KIND = "cnn_unet_pure_no_text_full_matrix_epoch1_benchmark_v1"
REGISTRY_KIND = "cnn_unet_pure_no_text_task_registry_v1"
INITIAL_STATE_FAIRNESS_KIND = "cnn_unet_pure_no_text_common_initial_state_v1"
QA_KIND = "cnn_unet_pure_no_text_terminal_qa_v1"
SOURCE_CODE_RELATIVE_PATHS = (
    "scripts/rq3/news_first_vol_cnn_unet_pure_no_text_seed42.py",
    "scripts/rq3/news_first_vol_cnn_unet_pure_no_text_seed42_analysis.py",
    "scripts/rq3/news_first_vol_film_unet_direct_5arm_seed42.py",
    "scripts/rq3/news_first_vol_film_unet_direct_5arm_seed42_analysis.py",
    "scripts/rq123/news_first_vol_film_nolp_10seed.py",
    "scripts/rq123/news_first_vol_film_unet_nolp_10seed.py",
    "scripts/rq3/news_first_vol_comparison_analysis.py",
    "scripts/rq3/news_first_vol_training.py",
    "scripts/rq2_pair/pair_features.py",
    "scripts/rq2_pair/rq2_pair_experiment.py",
)


@contextmanager
def pure_cnn_profile() -> Iterator[None]:
    """Install the four-cell pure-CNN contract and restore shared globals."""

    replacements: dict[str, Any] = {
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "EXPERIMENT_KIND": EXPERIMENT_KIND,
        "INTERPRETATION": INTERPRETATION,
        "DIRECT_ARMS": DIRECT_ARMS,
        "NO_TEXT_ARM": NO_TEXT_ARM,
        "EXPECTED_TRAINING_JOBS": EXPECTED_TRAINING_JOBS,
        "EXPECTED_PREDICTION_CELLS": EXPECTED_PREDICTION_CELLS,
        "EXPECTED_PAIR_METRIC_ROWS": EXPECTED_PAIR_METRIC_ROWS,
        "EXPECTED_PARAMETER_COUNTS": EXPECTED_PARAMETER_COUNTS,
        "GENERATOR_MODE": GENERATOR_MODE,
        "CRITIC_MODE": CRITIC_MODE,
        "CAPACITY_PROFILE": CAPACITY_PROFILE,
        "WORKER_MODULE": WORKER_MODULE,
        "INFERENCE_DETERMINISM_KIND": INFERENCE_DETERMINISM_KIND,
        "BENCHMARK_RESULT_KIND": BENCHMARK_RESULT_KIND,
        "REGISTRY_KIND": REGISTRY_KIND,
        "SOURCE_CODE_RELATIVE_PATHS": SOURCE_CODE_RELATIVE_PATHS,
    }
    originals = {name: getattr(direct, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(direct, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(direct, name, value)


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with pure_cnn_profile():
        config = direct.load_config(config_path)
    analysis = direct._mapping(config["analysis"], "analysis")
    reference = direct.resolve_path(analysis["historical_pair_metrics_path"])
    if direct.sha256_file(reference) != str(analysis["historical_pair_metrics_sha256"]):
        raise ValueError("Frozen historical pair-metrics reference drift")
    if str(analysis.get("historical_reference_arm")) != "no_text":
        raise ValueError("Historical reference arm must be no_text")
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    with pure_cnn_profile():
        direct.validate_config(config)


def planned_specs(
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    resolved = load_config() if config is None else dict(config)
    with pure_cnn_profile():
        return direct.planned_specs(resolved)


def validate_gpu_balance(specs: Sequence[Mapping[str, Any]]) -> None:
    with pure_cnn_profile():
        direct.validate_gpu_balance(specs)


def model_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    with pure_cnn_profile():
        return direct.model_contract(config)


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    with pure_cnn_profile():
        return getattr(direct, name)(*args, **kwargs)


def benchmark(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    load_config(config_path)
    return _call("benchmark", config_path, output_dir, resume=resume)


def prepare_experiment(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    load_config(config_path)
    return _call("prepare_experiment", config_path, output_dir, resume=resume)


def prepare(
    config_or_path: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    **kwargs: Any,
) -> Path:
    with pure_cnn_profile():
        return direct.prepare(config_or_path, output_root, **kwargs)


def dry_run(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    return benchmark(config_path, output_dir, resume=resume)


def worker(
    output_dir: str | Path,
    job_id: str,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> Path:
    return _call("worker", output_dir, job_id, resume=resume, dry_run=dry_run)


def launch(output_dir: str | Path, *, resume: bool = False) -> Path:
    return _call("launch", output_dir, resume=resume)


def freeze_evaluation(output_dir: str | Path, *, resume: bool = False) -> Path:
    return _call("freeze_evaluation", output_dir, resume=resume)


def predict(output_dir: str | Path, *, resume: bool = False) -> Path:
    return _call("predict", output_dir, resume=resume)


def status(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    return _call("status", output_dir)


def _verify_analysis_manifest(path: Path) -> dict[str, Any]:
    manifest = direct.read_json(path)
    if (
        manifest.get("kind")
        != "news_first_vol_cnn_unet_pure_no_text_seed42_analysis_manifest_v1"
    ):
        raise ValueError("Pure-CNN analysis manifest kind drift")
    rows = [
        (str(section), dict(row))
        for section in ("inputs", "artifacts")
        for row in manifest.get(section) or []
    ]
    if not rows:
        raise ValueError("Pure-CNN analysis manifest is empty")
    identities = {(section, str(row.get("role", ""))) for section, row in rows}
    if len(identities) != len(rows) or any(not role for _, role in identities):
        raise ValueError("Pure-CNN analysis manifest roles are empty or duplicated")
    for section, row in rows:
        target = Path(str(row.get("path", ""))).resolve()
        if (
            not target.is_file()
            or target.stat().st_size != int(row.get("size_bytes", -1))
            or direct.sha256_file(target) != str(row.get("sha256", ""))
        ):
            raise ValueError(f"Pure-CNN analysis {section} artifact drift: {target}")
    return manifest


def qa(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    with pure_cnn_profile():
        config = direct.validate_root(root)
        registry = direct.read_registry(root)
        if registry.get("terminal_complete"):
            direct._validate_output_hashes(root)
            return root / "qa.json"
        direct._all_training_complete(root)
        direct._checkpoint_map(root)
        direct._validate_test_inputs(root)
        direct._validate_predictions(root)
        manifest_path = direct._verify_frozen_file(
            registry["analysis_manifest_path"], registry["analysis_manifest_sha256"]
        )
        _verify_analysis_manifest(manifest_path)
        training = pd.read_csv(root / "analysis/training_summary.csv")
        pairs = pd.read_csv(root / "analysis/rq12_pair_metrics.csv.gz")
        if (
            len(training) != EXPECTED_TRAINING_JOBS
            or len(pairs) != EXPECTED_PAIR_METRIC_ROWS
        ):
            raise ValueError("Pure-CNN terminal evidence count drift")
        fairness = direct._verify_frozen_file(
            registry["initial_state_fairness_path"],
            registry["initial_state_fairness_sha256"],
        )
        if direct.read_json(fairness).get("job_count") != EXPECTED_TRAINING_JOBS:
            raise ValueError("Pure-CNN initial-state fairness evidence drift")
        qa_payload = {
            "schema_version": 1,
            "kind": QA_KIND,
            "status": "passed",
            "interpretation": INTERPRETATION,
            "training_jobs_completed": EXPECTED_TRAINING_JOBS,
            "prediction_cells": EXPECTED_PREDICTION_CELLS,
            "pair_metric_rows": EXPECTED_PAIR_METRIC_ROWS,
            "parent_jobs": 0,
            "continuation_jobs": 0,
            "text_encoder_registered": False,
            "film_layers_registered": False,
            "test_opened_after_checkpoint_freeze": True,
            "shared_mc64_noise_bank_within_fold": True,
            "generator_parameters": EXPECTED_PARAMETER_COUNTS["generator"],
            "critic_parameters": EXPECTED_PARAMETER_COUNTS["critic"],
            "source_config_sha256": config["source_config_sha256"],
            "completed_at_utc": direct.utc_now(),
        }
        qa_payload["payload_sha256"] = direct.payload_sha256(qa_payload)
        qa_path = direct.write_json(root / "qa.json", qa_payload)
        registry.update(
            status="completed",
            terminal_complete=True,
            completed_at_utc=direct.utc_now(),
        )
        direct.write_registry(root, registry)
        direct._experiment_status(root, "completed", current_stage="terminal")
        files = direct._terminal_files(root)
        rows = [
            {
                "artifact_role": f"output:{path.relative_to(root).as_posix()}",
                "relative_path": path.relative_to(root).as_posix(),
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": direct.sha256_file(path),
            }
            for path in files
        ]
        direct.write_csv(root / "output_hashes.csv", rows, tuple(rows[0]))
        direct._validate_output_hashes(root)
        return qa_path


def postprocess(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    with pure_cnn_profile():
        config = direct.validate_root(root)
        registry = direct.read_registry(root)
        if registry.get("terminal_complete"):
            return qa(root)
        direct._validate_predictions(root)
        training_summary = direct._training_summary(root)
        analysis_manifest = pure_analysis.analyze_experiment(root, config)
        _verify_analysis_manifest(analysis_manifest)
        direct._resource_summary(root)
        registry = direct.read_registry(root)
        registry.update(
            status="analysis_complete",
            analysis_complete=True,
            analysis_completed_at_utc=direct.utc_now(),
            training_summary_path=str(training_summary.resolve()),
            training_summary_sha256=direct.sha256_file(training_summary),
            analysis_manifest_path=str(analysis_manifest.resolve()),
            analysis_manifest_sha256=direct.sha256_file(analysis_manifest),
        )
        direct.write_registry(root, registry)
        direct._experiment_status(
            root, "analysis_complete", analysis_manifest=str(analysis_manifest)
        )
    return qa(root)


def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    root = Path(output_dir).resolve()
    if root.is_dir():
        try:
            registry = _call("read_registry", root)
        except (FileNotFoundError, ValueError):
            registry = {}
        if registry.get("terminal_complete"):
            with pure_cnn_profile():
                direct.validate_root(root)
                direct._validate_output_hashes(root)
            return root
    with pure_cnn_profile():
        lock_path, descriptor = direct._pipeline_lock(root)
    try:
        with pure_cnn_profile():
            direct._pipeline_journal(root, "benchmark", "running")
        benchmark(config_path, root, resume=resume)
        with pure_cnn_profile():
            direct._pipeline_journal(root, "prepare", "running")
        prepare_experiment(config_path, root, resume=True)
        registry = _call("read_registry", root)
        if registry.get("status") == "prepared":
            with pure_cnn_profile():
                direct._pipeline_journal(
                    root,
                    "direct_arms",
                    "running",
                    completed=status(root)["job_status_counts"].get("completed", 0),
                )
            launch(root, resume=True)
        if not _call("read_registry", root).get("evaluation_frozen"):
            with pure_cnn_profile():
                direct._pipeline_journal(root, "freeze_evaluation", "running")
            freeze_evaluation(root, resume=True)
        if not _call("read_registry", root).get("predictions_frozen"):
            with pure_cnn_profile():
                direct._pipeline_journal(root, "predict", "running")
            predict(root, resume=True)
        with pure_cnn_profile():
            direct._pipeline_journal(root, "postprocess", "running")
        postprocess(root, resume=True)
        with pure_cnn_profile():
            direct._pipeline_journal(root, "terminal", "completed")
        return root
    except BaseException as exc:
        with pure_cnn_profile():
            direct._pipeline_journal(
                root, "pipeline", "failed", error=f"{type(exc).__name__}: {exc}"
            )
        raise
    finally:
        direct._PIPELINE_LOCKS.pop(lock_path, None)
        direct._release_lock(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "benchmark",
            "prepare",
            "dry-run",
            "worker",
            "launch",
            "freeze-evaluation",
            "predict",
            "postprocess",
            "qa",
            "status",
            "run-pipeline",
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    action = str(args.action)
    if action == "benchmark":
        result: Any = benchmark(args.config, args.output_dir, resume=args.resume)
    elif action == "prepare":
        result = prepare_experiment(args.config, args.output_dir, resume=args.resume)
    elif action == "dry-run":
        result = dry_run(args.config, args.output_dir, resume=args.resume)
    elif action == "worker":
        if not args.job_id:
            raise SystemExit("worker requires --job-id")
        result = worker(
            args.output_dir,
            args.job_id,
            resume=args.resume,
            dry_run=args.worker_dry_run,
        )
    elif action == "launch":
        result = launch(args.output_dir, resume=args.resume)
    elif action == "freeze-evaluation":
        result = freeze_evaluation(args.output_dir, resume=args.resume)
    elif action == "predict":
        result = predict(args.output_dir, resume=args.resume)
    elif action == "postprocess":
        result = postprocess(args.output_dir, resume=args.resume)
    elif action == "qa":
        result = qa(args.output_dir, resume=args.resume)
    elif action == "status":
        result = status(args.output_dir)
    else:
        result = run_pipeline(args.config, args.output_dir, resume=args.resume)
    print(
        json.dumps(result, indent=2, sort_keys=True)
        if isinstance(result, Mapping)
        else result
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DIRECT_ARMS",
    "EXPECTED_PAIR_METRIC_ROWS",
    "EXPECTED_PREDICTION_CELLS",
    "EXPECTED_TRAINING_JOBS",
    "benchmark",
    "freeze_evaluation",
    "load_config",
    "model_contract",
    "planned_specs",
    "predict",
    "prepare",
    "run_pipeline",
    "validate_config",
    "validate_gpu_balance",
]
