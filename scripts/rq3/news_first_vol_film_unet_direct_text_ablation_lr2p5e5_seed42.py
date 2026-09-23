"""Direct seed-42 text ablation at a fixed FiLM LR of 2.5e-5.

The profile trains only ``lp_shuffle``, ``no_text``, ``bow`` and ``sentiment``
for four rolling folds.  The already-completed matched-LP ``film_lr_2p5e5``
cell is hash-bound as a read-only comparison and is never retrained.  All new
jobs start independently from the same seed-42 G/D initialization and use the
same split Generator optimizer as the matched-LP reference.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd
import yaml

from scripts.rq3 import news_first_vol_film_unet_direct_5arm_seed42 as direct
from scripts.rq3 import news_first_vol_film_unet_direct_matched_lr_seed42 as lr_base
from scripts.rq3 import (
    news_first_vol_film_unet_direct_text_ablation_lr2p5e5_seed42_analysis as ablation_analysis,
)


DEFAULT_CONFIG = (
    "configs/rq3/news_first_vol_film_unet_direct_text_ablation_lr2p5e5_seed42.yaml"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq12_news_first_vol_film_unet_text128_c32_nolp_direct_text_ablation_"
    "lr2p5e5_seed42_exact_ttm_rolling_v1"
)
EXPERIMENT_KIND = (
    "rq12_news_first_vol_film_unet_text128_c32_nolp_direct_text_ablation_"
    "lr2p5e5_seed42_rolling_v1"
)
INTERPRETATION = "retrospective_rolling_development_single_seed_descriptive"
DIRECT_ARMS = ("lp_shuffle", "no_text", "bow", "sentiment")
NO_TEXT_ARM = "no_text"
EXPECTED_TRAINING_JOBS = 16
EXPECTED_PREDICTION_CELLS = 16
EXPECTED_PAIR_METRIC_ROWS = 2_000
EXPECTED_COMBINED_PAIR_METRIC_ROWS = 2_500
WORKER_MODULE = (
    "scripts.rq3.news_first_vol_film_unet_direct_text_ablation_lr2p5e5_seed42"
)
INFERENCE_DETERMINISM_KIND = "direct_text_ablation_lr2p5e5_inference_v1"
BENCHMARK_RESULT_KIND = "direct_text_ablation_lr2p5e5_full_matrix_epoch1_v1"
REGISTRY_KIND = "direct_text_ablation_lr2p5e5_task_registry_v1"
QA_KIND = "direct_text_ablation_lr2p5e5_terminal_qa_v1"
GENERATOR_OPTIMIZER_PROFILE = "film_unet_split_lr_v1"
BACKBONE_LEARNING_RATE = 5.0e-7
TEXT_LEARNING_RATE = 2.5e-6
FILM_LEARNING_RATE = 2.5e-5
CRITIC_LEARNING_RATE = 5.0e-7
BACKBONE_MIN_LEARNING_RATE = 5.0e-8
TEXT_MIN_LEARNING_RATE = 2.5e-7
FILM_MIN_LEARNING_RATE = 2.5e-6
GROUP_FLOOR_RATIO = 0.1

SOURCE_CODE_RELATIVE_PATHS = tuple(
    dict.fromkeys(
        (
            "scripts/rq3/"
            "news_first_vol_film_unet_direct_text_ablation_lr2p5e5_seed42.py",
            "scripts/rq3/"
            "news_first_vol_film_unet_direct_text_ablation_lr2p5e5_seed42_analysis.py",
            *lr_base.SOURCE_CODE_RELATIVE_PATHS,
        )
    )
)


_BASE_VALIDATE_CONFIG = direct.validate_config
_BASE_SOURCE_PATHS = direct._source_paths
_BASE_BENCHMARK = direct.benchmark
_BASE_PREPARE_EXPERIMENT = direct.prepare_experiment
_BASE_PREPARE = direct.prepare
_BASE_WORKER = direct.worker
_BASE_LAUNCH = direct.launch
_BASE_FREEZE_EVALUATION = direct.freeze_evaluation
_BASE_PREDICT = direct.predict
_BASE_STATUS = direct.status
_BASE_RUN_PIPELINE = direct.run_pipeline
_BASE_MODEL_CONTRACT = direct.model_contract
_BASE_VALIDATE_GPU_BALANCE = direct.validate_gpu_balance


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = direct.resolve_path(config_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Direct text-ablation config must be a mapping")
    config = deepcopy(dict(raw))
    config["source_config_path"] = str(source)
    config["source_config_sha256"] = direct.sha256_file(source)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    with text_ablation_profile():
        _BASE_VALIDATE_CONFIG(config)
    matrix = direct._mapping(config.get("matrix"), "matrix")
    training = direct._mapping(config.get("training"), "training")
    analysis = direct._mapping(config.get("analysis"), "analysis")
    if tuple(map(str, matrix.get("direct_arms", ()))) != DIRECT_ARMS:
        raise ValueError("Text-ablation arm order drift")
    if int(matrix.get("expected_combined_pair_metric_rows", -1)) != 2_500:
        raise ValueError("Combined matched-reference evidence must have 2,500 rows")
    fixed = {
        "initial_learning_rate": BACKBONE_LEARNING_RATE,
        "generator_learning_rate": BACKBONE_LEARNING_RATE,
        "generator_text_learning_rate": TEXT_LEARNING_RATE,
        "generator_film_learning_rate": FILM_LEARNING_RATE,
        "generator_text_min_learning_rate": TEXT_MIN_LEARNING_RATE,
        "generator_film_min_learning_rate": FILM_MIN_LEARNING_RATE,
        "discriminator_learning_rate": CRITIC_LEARNING_RATE,
        "scheduler_min_lr": BACKBONE_MIN_LEARNING_RATE,
        "group_scheduler_floor_ratio": GROUP_FLOOR_RATIO,
    }
    if training.get("generator_optimizer_profile") != GENERATOR_OPTIMIZER_PROFILE:
        raise ValueError("Split Generator optimizer profile is required")
    for key, expected in fixed.items():
        if not math.isclose(
            float(training.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1e-18
        ):
            raise ValueError(f"training.{key} drift")
    representations = direct._mapping(
        config.get("text_representations"), "text_representations"
    )
    expected_modes = {
        "lp_shuffle": "lp_pair_mean_l2_v1",
        "no_text": "current_only",
        "bow": "bow_train_pair_vocab_log1p_l2_v1",
        "sentiment": "sentiment_pair_mean_train_zscore_pad_v1",
    }
    if set(representations) != set(DIRECT_ARMS) or any(
        direct._mapping(representations[arm], f"text_representations.{arm}").get("mode")
        != expected_modes[arm]
        for arm in DIRECT_ARMS
    ):
        raise ValueError("Text-representation contract drift")
    if analysis.get("direct_contrast_family") != (
        "matched_lp_vs_four_alternatives_holm4"
    ) or bool(analysis.get("test_based_representation_selection_permitted", True)):
        raise ValueError("Frozen text-ablation analysis contract drift")
    ablation_analysis.verify_frozen_reference(config)


def _source_paths(config: Mapping[str, Any]) -> list[tuple[str, Path]]:
    rows = list(_BASE_SOURCE_PATHS(config))
    reference = direct._mapping(
        direct._mapping(config["analysis"], "analysis").get("frozen_matched_lp"),
        "analysis.frozen_matched_lp",
    )
    rows.extend(
        (
            role,
            direct.resolve_path(reference[key]),
        )
        for role, key in (
            ("frozen_matched_output_hash_manifest", "output_hash_manifest_path"),
            ("frozen_matched_pair_metrics", "pair_metrics_path"),
            ("frozen_matched_training_summary", "training_summary_path"),
            ("frozen_matched_checkpoint_allowlist", "checkpoint_allowlist_path"),
        )
    )
    if len({role for role, _path in rows}) != len(rows):
        raise ValueError("Source-manifest roles must be unique")
    return rows


def planned_specs(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    resolved = load_config() if config is None else deepcopy(dict(config))
    validate_config(resolved)
    with text_ablation_profile():
        initial = direct._initial_state_hashes(resolved)
    assignment = direct._mapping(
        resolved["runtime"]["gpu_fold_assignment"], "gpu assignment"
    )
    specs: list[dict[str, Any]] = []
    for fold in direct.FOLDS:
        for arm in DIRECT_ARMS:
            specs.append(
                {
                    "stage": direct.DIRECT_STAGE,
                    "tolerance_minutes": direct.TOLERANCE_MINUTES,
                    "fold": fold,
                    "seed": direct.SEED,
                    "arm": arm,
                    "gpu_id": int(assignment[fold]),
                    "job_id": direct._job_id(fold, arm),
                    "pair_text_overlay_mode": direct._overlay_mode(arm),
                    "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
                    "generator_text_learning_rate": TEXT_LEARNING_RATE,
                    "generator_film_learning_rate": FILM_LEARNING_RATE,
                    "generator_text_min_learning_rate": TEXT_MIN_LEARNING_RATE,
                    "generator_film_min_learning_rate": FILM_MIN_LEARNING_RATE,
                    **initial,
                }
            )
    if len(specs) != 16 or len({row["job_id"] for row in specs}) != 16:
        raise AssertionError("Text ablation requires 16 unique jobs")
    with text_ablation_profile():
        _BASE_VALIDATE_GPU_BALANCE(specs)
    return specs


def _training_payload(
    config: Mapping[str, Any],
    root: Path,
    spec: Mapping[str, Any],
    *,
    num_epochs: int | None = None,
    slots_per_gpu: int | None = None,
) -> dict[str, Any]:
    del root, slots_per_gpu
    validate_config(config)
    arm = str(spec["arm"])
    if arm not in DIRECT_ARMS:
        raise ValueError(f"Unknown text-ablation arm: {arm}")
    training = direct._mapping(config["training"], "training")
    model = direct._mapping(config["model"], "model")
    return {
        "generator_conditioning_mode": model["generator_conditioning_mode"],
        "critic_conditioning_mode": model["critic_conditioning_mode"],
        "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
        "generator_learning_rate": BACKBONE_LEARNING_RATE,
        "generator_text_learning_rate": TEXT_LEARNING_RATE,
        "generator_film_learning_rate": FILM_LEARNING_RATE,
        "generator_text_min_learning_rate": TEXT_MIN_LEARNING_RATE,
        "generator_film_min_learning_rate": FILM_MIN_LEARNING_RATE,
        "discriminator_learning_rate": CRITIC_LEARNING_RATE,
        "reduce_lr_min_lr": BACKBONE_MIN_LEARNING_RATE,
        "num_epochs": int(num_epochs or training["num_epochs"]),
        "early_stopping_min_epochs": int(training["early_stopping_min_epochs"]),
        "early_stopping_patience": int(training["early_stopping_patience"]),
        "validation_mc_samples": int(training["validation_mc_samples"]),
        "news_first_materialize_validation_loader": True,
        "news_first_materialize_test_loader": False,
        "news_first_pair_text_overlay_mode": direct._overlay_mode(arm),
        "news_first_full_training_state_mode": "save_dynamic_v1",
        "news_first_refit_mode": "none",
        "use_reduce_lr_on_plateau": True,
        "use_early_stopping": True,
        "seed": direct.SEED,
    }


@contextmanager
def text_ablation_profile() -> Iterator[None]:
    """Install this additive 16-cell profile and restore shared globals."""

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
        "WORKER_MODULE": WORKER_MODULE,
        "INFERENCE_DETERMINISM_KIND": INFERENCE_DETERMINISM_KIND,
        "BENCHMARK_RESULT_KIND": BENCHMARK_RESULT_KIND,
        "REGISTRY_KIND": REGISTRY_KIND,
        "SOURCE_CODE_RELATIVE_PATHS": SOURCE_CODE_RELATIVE_PATHS,
        "load_config": load_config,
        "validate_config": validate_config,
        "planned_specs": planned_specs,
        "_source_paths": _source_paths,
        "_training_payload": _training_payload,
        "postprocess": _postprocess_active,
        "qa": _qa_active,
    }
    originals = {name: getattr(direct, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(direct, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(direct, name, value)


def model_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    with text_ablation_profile():
        return _BASE_MODEL_CONTRACT(config)


def validate_gpu_balance(specs: Sequence[Mapping[str, Any]]) -> None:
    with text_ablation_profile():
        _BASE_VALIDATE_GPU_BALANCE(specs)


def benchmark(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    load_config(config_path)
    with text_ablation_profile():
        return _BASE_BENCHMARK(config_path, output_dir, resume=resume)


def prepare_experiment(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    load_config(config_path)
    with text_ablation_profile():
        return _BASE_PREPARE_EXPERIMENT(config_path, output_dir, resume=resume)


def prepare(
    config_or_path: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    **kwargs: Any,
) -> Path:
    with text_ablation_profile():
        return _BASE_PREPARE(config_or_path, output_root, **kwargs)


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
    with text_ablation_profile():
        return _BASE_WORKER(output_dir, job_id, resume=resume, dry_run=dry_run)


def launch(output_dir: str | Path, *, resume: bool = False) -> Path:
    with text_ablation_profile():
        return _BASE_LAUNCH(output_dir, resume=resume)


def freeze_evaluation(output_dir: str | Path, *, resume: bool = False) -> Path:
    with text_ablation_profile():
        return _BASE_FREEZE_EVALUATION(output_dir, resume=resume)


def predict(output_dir: str | Path, *, resume: bool = False) -> Path:
    with text_ablation_profile():
        return _BASE_PREDICT(output_dir, resume=resume)


def status(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    with text_ablation_profile():
        return _BASE_STATUS(output_dir)


def _training_summary(root: Path) -> Path:
    destination = root / "analysis/training_summary.csv"
    if destination.is_file():
        return destination
    checkpoints = direct._checkpoint_map(root)
    registry = direct.read_registry(root)
    rows: list[dict[str, Any]] = []
    for job in registry["jobs"]:
        status_payload = direct.read_json(direct._status_path(root, str(job["job_id"])))
        best = direct.read_json(
            Path(direct._artifact(status_payload, "best_learned_checkpoint")["path"])
        )
        metrics = pd.read_csv(
            direct._artifact(status_payload, "training_metrics_csv")["path"]
        )
        epochs = pd.to_numeric(metrics["epoch"], errors="raise").astype(int)
        learned = epochs[epochs >= 1]
        if learned.empty:
            raise ValueError(f"Training metrics have no learned epoch: {job['job_id']}")
        epochs_ran = int(learned.max())
        contract = lr_base._optimizer_contract(
            job=job,
            metrics=metrics.loc[epochs <= epochs_ran].copy(),
            epochs_ran=epochs_ran,
        )
        rows.append(
            {
                "job_id": str(job["job_id"]),
                "fold": str(job["fold"]),
                "arm": str(job["arm"]),
                "best_epoch": int(best["best_epoch"]),
                "epochs_ran": epochs_ran,
                "final_generator_lr": float(metrics.iloc[-1]["g_lr_backbone"]),
                "final_discriminator_lr": float(metrics.iloc[-1]["d_lr"]),
                "best_validation_score": float(best["best_metric"]),
                "checkpoint_sha256": checkpoints[str(job["job_id"])]["sha256"],
                "monitor_metric": str(best.get("monitor_metric", "val_hybrid_score")),
                "early_stopped": epochs_ran < int(registry["training_num_epochs"]),
                "optimizer_contract_json": json.dumps(
                    contract,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        )
    if len(rows) != EXPECTED_TRAINING_JOBS:
        raise ValueError("Training summary requires exactly 16 cells")
    return direct.write_csv(destination, rows, tuple(rows[0]))


def _verify_analysis_manifest(path: Path) -> dict[str, Any]:
    payload = direct.read_json(path)
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("kind") != ablation_analysis.ANALYSIS_KIND or direct.payload_sha256(
        unsigned
    ) != payload.get("payload_sha256"):
        # The analysis module hashes its pretty canonical JSON bytes, whereas
        # the orchestrator payload helper hashes compact canonical JSON.  Fall
        # back to the exact byte-level definition used by that module.
        import hashlib

        observed = hashlib.sha256(
            (
                json.dumps(
                    unsigned,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        if payload.get(
            "kind"
        ) != ablation_analysis.ANALYSIS_KIND or observed != payload.get(
            "payload_sha256"
        ):
            raise ValueError("Text-ablation analysis manifest drift")
    for row in payload.get("inputs") or []:
        target = Path(str(row["path"])).resolve()
        if not target.is_file() or direct.sha256_file(target) != str(row["sha256"]):
            raise ValueError(f"Text-ablation analysis input drift: {target}")
    inner = direct._mapping(payload.get("inner_analysis_manifest"), "inner manifest")
    direct._verify_frozen_file(inner["path"], inner["sha256"])
    return payload


def _postprocess_active(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    config = direct.validate_root(root)
    registry = direct.read_registry(root)
    if registry.get("terminal_complete"):
        return _qa_active(root)
    direct._validate_predictions(root)
    training_summary = _training_summary(root)
    analysis_manifest = ablation_analysis.analyze_experiment(root, config)
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
    return _qa_active(root)


def postprocess(output_dir: str | Path, *, resume: bool = False) -> Path:
    with text_ablation_profile():
        return _postprocess_active(output_dir, resume=resume)


def _qa_active(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    config = direct.validate_root(root)
    registry = direct.read_registry(root)
    if registry.get("terminal_complete"):
        direct._validate_output_hashes(root)
        return root / "qa.json"
    direct._all_training_complete(root)
    direct._checkpoint_map(root)
    direct._validate_test_inputs(root)
    direct._validate_predictions(root)
    manifest = direct._verify_frozen_file(
        registry["analysis_manifest_path"], registry["analysis_manifest_sha256"]
    )
    _verify_analysis_manifest(manifest)
    ablation_analysis.verify_frozen_reference(config)
    training = pd.read_csv(root / "analysis/training_summary.csv")
    pairs = pd.read_csv(root / "analysis/rq12_pair_metrics.csv.gz")
    combined = pd.read_csv(root / "analysis/combined_five_arm_pair_metrics.csv")
    if (
        len(training) != EXPECTED_TRAINING_JOBS
        or len(pairs) != EXPECTED_PAIR_METRIC_ROWS
        or len(combined) != EXPECTED_COMBINED_PAIR_METRIC_ROWS
    ):
        raise ValueError("Text-ablation terminal evidence count drift")
    fairness = direct._verify_frozen_file(
        registry["initial_state_fairness_path"],
        registry["initial_state_fairness_sha256"],
    )
    if direct.read_json(fairness).get("job_count") != EXPECTED_TRAINING_JOBS:
        raise ValueError("Text-ablation initial-state fairness evidence drift")
    qa_payload = {
        "schema_version": 1,
        "kind": QA_KIND,
        "status": "passed",
        "interpretation": INTERPRETATION,
        "training_jobs_completed": EXPECTED_TRAINING_JOBS,
        "prediction_cells": EXPECTED_PREDICTION_CELLS,
        "new_pair_metric_rows": EXPECTED_PAIR_METRIC_ROWS,
        "combined_pair_metric_rows": EXPECTED_COMBINED_PAIR_METRIC_ROWS,
        "matched_lp_reference_jobs": 4,
        "matched_lp_retrained": False,
        "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
        "film_learning_rate": FILM_LEARNING_RATE,
        "parent_jobs": 0,
        "continuation_jobs": 0,
        "test_opened_after_checkpoint_freeze": True,
        "shared_mc64_noise_bank_within_new_fold": True,
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


def qa(output_dir: str | Path, *, resume: bool = False) -> Path:
    with text_ablation_profile():
        return _qa_active(output_dir, resume=resume)


def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    load_config(config_path)
    with text_ablation_profile():
        return _BASE_RUN_PIPELINE(config_path, output_dir, resume=resume)


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
    elif action == "run-pipeline":
        result = run_pipeline(args.config, args.output_dir, resume=args.resume)
    else:  # pragma: no cover
        raise AssertionError(action)
    print(json.dumps(str(result) if isinstance(result, Path) else result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "DIRECT_ARMS",
    "EXPECTED_TRAINING_JOBS",
    "FILM_LEARNING_RATE",
    "benchmark",
    "freeze_evaluation",
    "launch",
    "load_config",
    "main",
    "planned_specs",
    "postprocess",
    "predict",
    "prepare",
    "prepare_experiment",
    "qa",
    "run_pipeline",
    "status",
    "text_ablation_profile",
    "validate_config",
    "worker",
]
