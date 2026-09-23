"""Ten-seed direct-training baseline for the text-free convolutional U-Net.

This branch-local profile reuses the audited rolling direct-training lifecycle.
Every seed/fold cell starts from an independent seeded random initialization;
there are no parent, continuation, or branch-recipe inputs.  The Generator is
``cnn_unet_mask_coords_v1`` and therefore registers neither a text encoder nor
FiLM projections.  The 2.5e-5 rate used by the companion experiment belongs
only to FiLM projections; the shared CNN backbone and NoLP Critic stay at
5e-7 here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from scripts.rq3 import news_first_vol_cnn_unet_pure_no_text_seed42 as single
from scripts.rq3 import news_first_vol_film_unet_direct_5arm_seed42 as direct
from scripts.rq3 import news_first_vol_film_unet_direct_matched_lr_5seed as multi


DEFAULT_CONFIG = "configs/rq3/news_first_vol_cnn_unet_pure_no_text_10seed.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq12_news_first_vol_cnn_unet_c32_nolp_pure_no_text_10seed_"
    "exact_ttm_rolling_v1"
)
EXPERIMENT_KIND = "rq12_news_first_vol_cnn_unet_c32_nolp_pure_no_text_10seed_rolling_v1"
INTERPRETATION = "retrospective_rolling_development_10seed_architecture_baseline"
DIRECT_ARMS = ("pure_cnn_no_text",)
NO_TEXT_ARM = DIRECT_ARMS[0]
SEEDS = (
    42,
    202,
    404,
    382624741,
    1607127774,
    1662128673,
    2041145538,
    2014889368,
    1343862330,
    779214671,
)
FOLDS = direct.FOLDS
TOLERANCE_MINUTES = 5
EXPECTED_TRAINING_JOBS = 40
EXPECTED_PREDICTION_CELLS = 40
EXPECTED_PAIR_METRIC_ROWS = 5_000
EXPECTED_PARAMETER_COUNTS = {
    "generator": 416_353,
    "critic": 729_157,
    "total": 1_145_510,
}
GENERATOR_MODE = "cnn_unet_mask_coords_v1"
CRITIC_MODE = "lp_disabled_same_shape_v1"
CAPACITY_PROFILE = "c32"
GENERATOR_OPTIMIZER_PROFILE = "uniform_v1"
GENERATOR_LEARNING_RATE = 5.0e-7
CRITIC_LEARNING_RATE = 5.0e-7
SCHEDULER_MIN_LR = 5.0e-8
WORKER_MODULE = "scripts.rq3.news_first_vol_cnn_unet_pure_no_text_10seed"
INFERENCE_DETERMINISM_KIND = "cnn_unet_pure_no_text_10seed_inference_v1"
BENCHMARK_RESULT_KIND = "cnn_unet_pure_no_text_10seed_full_matrix_epoch1_v1"
REGISTRY_KIND = "cnn_unet_pure_no_text_10seed_task_registry_v1"
QA_KIND = "cnn_unet_pure_no_text_10seed_terminal_qa_v1"
ANALYSIS_MANIFEST_KIND = "cnn_unet_pure_no_text_10seed_analysis_manifest_v1"
FAIRNESS_KIND = "cnn_unet_pure_no_text_10seed_initial_state_fairness_v1"
SOURCE_CODE_RELATIVE_PATHS = tuple(
    dict.fromkeys(
        (
            "scripts/rq3/news_first_vol_cnn_unet_pure_no_text_10seed.py",
            "scripts/rq3/news_first_vol_cnn_unet_pure_no_text_seed42.py",
            "scripts/rq3/news_first_vol_cnn_unet_pure_no_text_seed42_analysis.py",
            *multi.SOURCE_CODE_RELATIVE_PATHS,
        )
    )
)


_BASE_MULTI_VALIDATE_ROOT = multi.validate_root
_BASE_MULTI_VALIDATE_PREDICTIONS = multi._validate_predictions
_BASE_DIRECT_SOURCE_PATHS = direct._source_paths
_BASE_DIRECT_MATERIALIZE_TEST_INPUTS = direct._materialize_test_inputs
_BASE_DIRECT_GRID_CONTRACT = direct.grid_contract
_BASE_DIRECT_MODEL_CONTRACT = direct.model_contract


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = direct.resolve_path(config_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Pure-CNN ten-seed config must be a mapping")
    config = deepcopy(dict(raw))
    config["source_config_path"] = str(source)
    config["source_config_sha256"] = direct.sha256_file(source)
    validate_config(config)
    return config


def _load_frozen_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Frozen Pure-CNN ten-seed config must be a mapping")
    config = dict(raw)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    experiment = _mapping(config.get("experiment"), "experiment")
    data = _mapping(config.get("data"), "data")
    matrix = _mapping(config.get("matrix"), "matrix")
    model = _mapping(config.get("model"), "model")
    training = _mapping(config.get("training"), "training")
    analysis = _mapping(config.get("analysis"), "analysis")
    runtime = _mapping(config.get("runtime"), "runtime")
    if int(experiment.get("schema_version", -1)) != 1:
        raise ValueError("experiment.schema_version must be 1")
    if experiment.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment kind drift")
    if experiment.get("interpretation") != INTERPRETATION:
        raise ValueError("Interpretation label drift")
    if tuple(map(int, data.get("tolerances_minutes", ()))) != (5,):
        raise ValueError("Only the 5m alignment is permitted")
    if tuple(map(int, data.get("maturity_days_grid", ()))) != direct.GRID:
        raise ValueError("Exact-TTM grid drift")
    if data.get("support_mask_mode") != "raw_joint":
        raise ValueError("support_mask_mode must be raw_joint")
    if int(data.get("embedding_dimension", -1)) != 1024:
        raise ValueError("Overlay width must remain 1024")
    folds = list(config.get("folds") or [])
    if tuple(str(row.get("id")) for row in folds) != FOLDS:
        raise ValueError("Rolling-fold identifiers/order drift")
    for row in folds:
        train_end = pd.Timestamp(row["train_end_utc"])
        validation_start = pd.Timestamp(row["validation_start_utc"])
        validation_end = pd.Timestamp(row["validation_end_utc"])
        test_start = pd.Timestamp(row["test_start_utc"])
        test_end = pd.Timestamp(row["test_end_utc"])
        if not (
            train_end == validation_start < validation_end == test_start < test_end
        ):
            raise ValueError(f"Invalid half-open fold windows: {row['id']}")
    if tuple(map(int, matrix.get("seeds", ()))) != SEEDS:
        raise ValueError("Ten-seed order drift")
    if tuple(map(int, matrix.get("tolerances_minutes", ()))) != (5,):
        raise ValueError("matrix.tolerances_minutes must be [5]")
    if tuple(map(str, matrix.get("direct_arms", ()))) != DIRECT_ARMS:
        raise ValueError("Pure-CNN arm universe drift")
    expected_matrix = {
        "expected_training_jobs": EXPECTED_TRAINING_JOBS,
        "expected_prediction_cells": EXPECTED_PREDICTION_CELLS,
        "expected_pair_metric_rows": EXPECTED_PAIR_METRIC_ROWS,
    }
    for key, expected in expected_matrix.items():
        if int(matrix.get(key, -1)) != expected:
            raise ValueError(f"matrix.{key} drift")
    representations = _mapping(
        config.get("text_representations"), "text_representations"
    )
    representation = _mapping(
        representations.get(NO_TEXT_ARM), f"text_representations.{NO_TEXT_ARM}"
    )
    if (
        set(representations) != {NO_TEXT_ARM}
        or representation.get("mode") != "current_only"
        or representation.get("normalization") != "zero_vector_v1"
        or bool(representation.get("consumed_by_generator", True))
    ):
        raise ValueError("Pure-CNN must consume no text representation")
    if model.get("generator_conditioning_mode") != GENERATOR_MODE:
        raise ValueError(f"Generator must be {GENERATOR_MODE}")
    if model.get("critic_conditioning_mode") != CRITIC_MODE:
        raise ValueError(f"Critic must be {CRITIC_MODE}")
    if model.get("capacity_profile") != CAPACITY_PROFILE:
        raise ValueError(f"Capacity must be {CAPACITY_PROFILE}")
    counts = {
        "generator": int(model.get("expected_generator_parameters", -1)),
        "critic": int(model.get("expected_critic_parameters", -1)),
        "total": int(model.get("expected_total_parameters", -1)),
    }
    if counts != EXPECTED_PARAMETER_COUNTS:
        raise ValueError("Executable parameter-count contract drift")
    required_training: Mapping[str, Any] = {
        "protocol": "independent_random_init_v1",
        "initial_state_contract": (
            "common_within_seed_distinct_across_seeds_g_and_d_state_sha_v1"
        ),
        "checkpoint_selection_mode": "best_learned_dynamic_validation_v1",
        "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
        "num_epochs": 240,
        "early_stopping_min_epochs": 30,
        "early_stopping_patience": 20,
        "lr_warmup_epochs": 0,
        "batch_size": 16,
        "discriminator_steps": 5,
        "validation_mc_samples": 16,
        "prediction_mc_samples": 64,
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} drift")
    for key, expected in {
        "initial_learning_rate": GENERATOR_LEARNING_RATE,
        "generator_learning_rate": GENERATOR_LEARNING_RATE,
        "discriminator_learning_rate": CRITIC_LEARNING_RATE,
        "scheduler_min_lr": SCHEDULER_MIN_LR,
    }.items():
        if not math.isclose(
            float(training.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1e-18
        ):
            raise ValueError(f"training.{key} drift")
    if analysis.get("interpretation") != INTERPRETATION:
        raise ValueError("analysis.interpretation drift")
    if not bool(analysis.get("cross_seed_inference_enabled")):
        raise ValueError("Cross-seed summaries must be enabled")
    if not bool(analysis.get("best_observed_seed_is_descriptive_only")):
        raise ValueError("Best-observed seed must be labelled descriptive")
    if bool(analysis.get("test_based_seed_selection_permitted", True)):
        raise ValueError("Test-based seed selection is prohibited")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != (0, 1):
        raise ValueError("Runtime requires GPU 0 and GPU 1")
    if int(runtime.get("benchmark_jobs", -1)) != EXPECTED_TRAINING_JOBS:
        raise ValueError("Benchmark must exercise all 40 cells")
    if int(runtime.get("benchmark_epochs", -1)) != 1:
        raise ValueError("Benchmark must use one epoch")
    primary = int(runtime.get("benchmark_workers_per_gpu", -1))
    fallback = int(runtime.get("fallback_workers_per_gpu", -1))
    if primary < 1 or fallback < 1 or fallback > primary:
        raise ValueError("Invalid benchmark/fallback concurrency")
    formal = runtime.get("formal_workers_per_gpu")
    if formal is not None and int(formal) not in (primary, fallback):
        raise ValueError("Formal concurrency was not benchmarked")


def _overlay_mode(arm: str) -> str:
    if str(arm) != NO_TEXT_ARM:
        raise ValueError(f"Unknown Pure-CNN arm: {arm}")
    return "current_only"


def _initial_state_hashes(config: Mapping[str, Any], seed: int) -> dict[str, str]:
    original_seed = direct.SEED
    try:
        with single.pure_cnn_profile():
            direct.SEED = int(seed)
            return direct._initial_state_hashes(config)
    finally:
        direct.SEED = original_seed


def _job_id(seed: int, fold: str) -> str:
    return f"direct_arms_05m_{fold}_seed_{int(seed)}_{NO_TEXT_ARM}"


def planned_specs(
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    resolved = load_config() if config is None else deepcopy(dict(config))
    validate_config(resolved)
    initial_by_seed = {seed: _initial_state_hashes(resolved, seed) for seed in SEEDS}
    if len(
        {row["initial_generator_state_sha256"] for row in initial_by_seed.values()}
    ) != len(SEEDS) or len(
        {row["initial_critic_state_sha256"] for row in initial_by_seed.values()}
    ) != len(SEEDS):
        raise ValueError("Fresh seeds did not produce distinct G/D initial states")
    assignment = _mapping(resolved["runtime"]["gpu_fold_assignment"], "gpu assignment")
    specs = [
        {
            "stage": direct.DIRECT_STAGE,
            "tolerance_minutes": TOLERANCE_MINUTES,
            "fold": fold,
            "seed": seed,
            "arm": NO_TEXT_ARM,
            "gpu_id": int(assignment[fold]),
            "job_id": _job_id(seed, fold),
            "pair_text_overlay_mode": "current_only",
            "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
            **initial_by_seed[seed],
        }
        for seed in SEEDS
        for fold in FOLDS
    ]
    if len(specs) != 40 or len({row["job_id"] for row in specs}) != 40:
        raise AssertionError("Pure-CNN ten-seed profile requires 40 unique jobs")
    validate_gpu_balance(specs)
    return specs


def validate_gpu_balance(specs: Sequence[Mapping[str, Any]]) -> None:
    blocks = {(int(row["seed"]), str(row["fold"])): int(row["gpu_id"]) for row in specs}
    if len(blocks) != 40:
        raise ValueError("Each seed/fold block must contain exactly one Pure-CNN job")
    counts = {gpu: sum(int(row["gpu_id"]) == gpu for row in specs) for gpu in (0, 1)}
    if counts != {0: 20, 1: 20}:
        raise ValueError(f"GPU job balance must be 20/20, got {counts}")
    expected = {
        (seed, fold): (0 if fold in {"f1_2023q1", "f3_2023q3"} else 1)
        for seed in SEEDS
        for fold in FOLDS
    }
    if blocks != expected:
        raise ValueError("Frozen seed/fold GPU assignment drift")


def _source_paths(config: Mapping[str, Any]) -> list[tuple[str, Path]]:
    return list(_BASE_DIRECT_SOURCE_PATHS(config))


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
    if str(spec["arm"]) != NO_TEXT_ARM:
        raise ValueError(f"Unknown Pure-CNN arm: {spec['arm']}")
    training = _mapping(config["training"], "training")
    model = _mapping(config["model"], "model")
    return {
        "generator_conditioning_mode": model["generator_conditioning_mode"],
        "critic_conditioning_mode": model["critic_conditioning_mode"],
        "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
        "generator_learning_rate": GENERATOR_LEARNING_RATE,
        "discriminator_learning_rate": CRITIC_LEARNING_RATE,
        "reduce_lr_min_lr": SCHEDULER_MIN_LR,
        "num_epochs": int(num_epochs or training["num_epochs"]),
        "early_stopping_min_epochs": int(training["early_stopping_min_epochs"]),
        "early_stopping_patience": int(training["early_stopping_patience"]),
        "validation_mc_samples": int(training["validation_mc_samples"]),
        "news_first_materialize_validation_loader": True,
        "news_first_materialize_test_loader": False,
        "news_first_pair_text_overlay_mode": "current_only",
        "news_first_full_training_state_mode": "save_dynamic_v1",
        "news_first_refit_mode": "none",
        "use_reduce_lr_on_plateau": True,
        "use_early_stopping": True,
        "seed": int(spec["seed"]),
    }


def _materialize_pair_text_overlays(config: Mapping[str, Any], root: Path) -> Path:
    from wgan_option.utils.news_first_experiment_core import (
        build_pair_text_overlay_manifests,
        write_pair_text_overlay_manifest,
    )

    output_dir = root / "inputs/pair_text_overlays"
    generated = build_pair_text_overlay_manifests(
        config=config,
        pair_universe_path=root / "inputs/pair_universes.csv",
        output_dir=output_dir,
    )
    if len(generated) != 24:
        raise ValueError("Shared builder did not produce the canonical 24 overlays")
    for fold in FOLDS:
        source = output_dir / "tolerance_05m" / fold / "parent_current_only.json"
        payload = direct.read_json(source)
        destination = direct._overlay_path(root, fold, NO_TEXT_ARM)
        write_pair_text_overlay_manifest(
            destination,
            mode="current_only",
            namespace=f"tol05/{fold}/{NO_TEXT_ARM}/direct_10seed_v1",
            records=list(payload["records"]),
            transform={
                **dict(payload.get("transform") or {}),
                "direct_arm": NO_TEXT_ARM,
                "training_seeds": list(SEEDS),
                "parent_or_continuation_state_input": False,
                "text_consumed_by_generator": False,
            },
        )
    expected = {direct._overlay_path(root, fold, NO_TEXT_ARM) for fold in FOLDS}
    for path in output_dir.glob("tolerance_*/*/*.json"):
        if path.resolve() not in expected:
            path.unlink()
    paths = sorted(output_dir.glob("tolerance_05m/*/*.json"))
    if len(paths) != 4 or {path.resolve() for path in paths} != expected:
        raise ValueError("Pure-CNN development overlay universe drift")
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = direct.read_json(path)
        unsigned = {
            key: value for key, value in payload.items() if key != "profile_sha256"
        }
        records = list(payload.get("records") or [])
        if (
            payload.get("mode") != "current_only"
            or direct.payload_sha256(unsigned) != payload.get("profile_sha256")
            or not records
            or any(
                any(float(value) != 0.0 for value in record["embedding"])
                for record in records
            )
        ):
            raise ValueError(f"Pure-CNN zero-text overlay drift: {path}")
        rows.append(direct.manifest_row(f"pair_overlay:{path.relative_to(root)}", path))
    return direct._write_hash_manifest(
        root / "inputs/pair_text_overlay_hashes.csv", rows
    )


def grid_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    with single.pure_cnn_profile():
        return _BASE_DIRECT_GRID_CONTRACT(config)


def model_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    with single.pure_cnn_profile():
        return _BASE_DIRECT_MODEL_CONTRACT(config)


def validate_root(
    root_or_path: str | Path, *, verify_large_inputs: bool = True
) -> dict[str, Any]:
    return _BASE_MULTI_VALIDATE_ROOT(
        root_or_path, verify_large_inputs=verify_large_inputs
    )


def _freeze_initial_state_fairness(
    root: Path, jobs: Sequence[Mapping[str, Any]]
) -> Path:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        status = direct.read_json(direct._status_path(root, str(job["job_id"])))
        initial_g = direct._artifact(status, "generator_initial_epoch0")
        initial_d = direct._artifact(status, "discriminator_initial_epoch0")
        best_g = direct._artifact(status, "generator_best_learned")
        best_d = direct._artifact(status, "discriminator_best_learned")
        observed_g = direct._checkpoint_state_sha256(initial_g["path"])
        observed_d = direct._checkpoint_state_sha256(initial_d["path"])
        best_g_sha = direct._checkpoint_state_sha256(best_g["path"])
        best_d_sha = direct._checkpoint_state_sha256(best_d["path"])
        if (
            observed_g != job["initial_generator_state_sha256"]
            or observed_d != job["initial_critic_state_sha256"]
        ):
            raise ValueError(f"Frozen epoch-0 state drift: {job['job_id']}")
        if observed_g == best_g_sha or observed_d == best_d_sha:
            raise ValueError(f"G/D parameters did not both update: {job['job_id']}")
        if job.get("parent_state_path") or job.get("recipe_path"):
            raise ValueError(
                f"Direct Pure-CNN job consumed forbidden state: {job['job_id']}"
            )
        rows.append(
            {
                "job_id": str(job["job_id"]),
                "seed": int(job["seed"]),
                "fold": str(job["fold"]),
                "arm": str(job["arm"]),
                "generator_initial_state_sha256": observed_g,
                "critic_initial_state_sha256": observed_d,
                "generator_best_state_sha256": best_g_sha,
                "critic_best_state_sha256": best_d_sha,
            }
        )
    by_seed: dict[str, Any] = {}
    for seed in SEEDS:
        selected = [row for row in rows if row["seed"] == seed]
        g_hashes = {row["generator_initial_state_sha256"] for row in selected}
        d_hashes = {row["critic_initial_state_sha256"] for row in selected}
        if len(selected) != 4 or len(g_hashes) != 1 or len(d_hashes) != 1:
            raise ValueError(f"Actual epoch-0 fairness drift within seed {seed}")
        by_seed[str(seed)] = {
            "job_count": 4,
            "generator_initial_state_sha256": next(iter(g_hashes)),
            "critic_initial_state_sha256": next(iter(d_hashes)),
        }
    if len({row["generator_initial_state_sha256"] for row in by_seed.values()}) != 10:
        raise ValueError("Generator epoch-0 states are not distinct across seeds")
    if len({row["critic_initial_state_sha256"] for row in by_seed.values()}) != 10:
        raise ValueError("Critic epoch-0 states are not distinct across seeds")
    payload = {
        "schema_version": 1,
        "kind": FAIRNESS_KIND,
        "seeds": list(SEEDS),
        "initialization_order": "seed_everything(seed), Generator, Critic",
        "job_count": len(rows),
        "common_within_seed": True,
        "distinct_across_seeds": True,
        "all_generator_parameters_updated": True,
        "all_critic_parameters_updated": True,
        "parent_jobs": 0,
        "continuation_jobs": 0,
        "by_seed": by_seed,
        "rows": sorted(rows, key=lambda row: row["job_id"]),
    }
    payload["payload_sha256"] = direct.payload_sha256(payload)
    return direct.write_json(direct._initial_state_fairness_path(root), payload)


def _materialize_test_inputs(config: Mapping[str, Any], root: Path) -> Path:
    return _BASE_DIRECT_MATERIALIZE_TEST_INPUTS(config, root)


def _validate_predictions(root: Path) -> tuple[Path, Path]:
    return _BASE_MULTI_VALIDATE_PREDICTIONS(root)


@contextmanager
def _direct_multiseed_profile() -> Iterator[None]:
    """Install the Pure-CNN contract in the shared direct orchestrator."""

    replacements: dict[str, Any] = {
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "EXPERIMENT_KIND": EXPERIMENT_KIND,
        "INTERPRETATION": INTERPRETATION,
        "DIRECT_ARMS": DIRECT_ARMS,
        "NO_TEXT_ARM": NO_TEXT_ARM,
        "SEED": SEEDS[0],
        "EXPECTED_TRAINING_JOBS": EXPECTED_TRAINING_JOBS,
        "EXPECTED_PREDICTION_CELLS": EXPECTED_PREDICTION_CELLS,
        "EXPECTED_PAIR_METRIC_ROWS": EXPECTED_PAIR_METRIC_ROWS,
        "WORKER_MODULE": WORKER_MODULE,
        "INFERENCE_DETERMINISM_KIND": INFERENCE_DETERMINISM_KIND,
        "BENCHMARK_RESULT_KIND": BENCHMARK_RESULT_KIND,
        "REGISTRY_KIND": REGISTRY_KIND,
        "SOURCE_CODE_RELATIVE_PATHS": SOURCE_CODE_RELATIVE_PATHS,
        "GENERATOR_MODE": GENERATOR_MODE,
        "CRITIC_MODE": CRITIC_MODE,
        "CAPACITY_PROFILE": CAPACITY_PROFILE,
        "EXPECTED_PARAMETER_COUNTS": EXPECTED_PARAMETER_COUNTS,
        "direct_profile": multi._core_profile,
        "load_config": load_config,
        "_load_frozen_config": _load_frozen_config,
        "validate_config": validate_config,
        "grid_contract": grid_contract,
        "model_contract": model_contract,
        "planned_specs": planned_specs,
        "validate_gpu_balance": validate_gpu_balance,
        "_source_paths": _source_paths,
        "_overlay_mode": _overlay_mode,
        "_run_directory": multi._run_directory,
        "_materialize_pair_text_overlays": _materialize_pair_text_overlays,
        "_training_payload": _training_payload,
        "validate_root": validate_root,
        "_freeze_initial_state_fairness": _freeze_initial_state_fairness,
        "_materialize_test_inputs": _materialize_test_inputs,
        "_prediction_path": multi._prediction_path,
        "_prediction_manifest_path": multi._prediction_manifest_path,
        "_validate_prediction_cell": multi._validate_prediction_cell,
        "_validate_predictions": _validate_predictions,
    }
    with multi._core_profile():
        originals = {name: getattr(direct, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(direct, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(direct, name, value)


@contextmanager
def pure_multiseed_profile() -> Iterator[None]:
    """Install this 40-cell profile into the existing multiseed lifecycle."""

    replacements: dict[str, Any] = {
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "EXPERIMENT_KIND": EXPERIMENT_KIND,
        "INTERPRETATION": INTERPRETATION,
        "DIRECT_ARMS": DIRECT_ARMS,
        "FILM_LEARNING_RATES": {NO_TEXT_ARM: GENERATOR_LEARNING_RATE},
        "SEEDS": SEEDS,
        "FOLDS": FOLDS,
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
        "QA_KIND": QA_KIND,
        "GENERATOR_OPTIMIZER_PROFILE": GENERATOR_OPTIMIZER_PROFILE,
        "BACKBONE_LEARNING_RATE": GENERATOR_LEARNING_RATE,
        "CRITIC_LEARNING_RATE": CRITIC_LEARNING_RATE,
        "BACKBONE_MIN_LEARNING_RATE": SCHEDULER_MIN_LR,
        "SOURCE_CODE_RELATIVE_PATHS": SOURCE_CODE_RELATIVE_PATHS,
        "multiseed_profile": _direct_multiseed_profile,
        "load_config": load_config,
        "_load_frozen_config": _load_frozen_config,
        "validate_config": validate_config,
        "_overlay_mode": _overlay_mode,
        "planned_specs": planned_specs,
        "validate_gpu_balance": validate_gpu_balance,
        "_source_paths": _source_paths,
        "_training_payload": _training_payload,
        "_materialize_pair_text_overlays": _materialize_pair_text_overlays,
        "grid_contract": grid_contract,
        "model_contract": model_contract,
        "validate_root": validate_root,
        "_freeze_initial_state_fairness": _freeze_initial_state_fairness,
        "_materialize_test_inputs": _materialize_test_inputs,
        "_validate_predictions": _validate_predictions,
        "postprocess": postprocess,
        "qa": qa,
    }
    originals = {name: getattr(multi, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(multi, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(multi, name, value)


@contextmanager
def _runtime_profile() -> Iterator[None]:
    with pure_multiseed_profile():
        with multi.multiseed_profile():
            yield


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    with pure_multiseed_profile():
        return getattr(multi, name)(*args, **kwargs)


def benchmark(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    return _call("benchmark", config_path, output_dir, resume=resume)


def prepare_experiment(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    return _call("prepare_experiment", config_path, output_dir, resume=resume)


def prepare(
    config_or_path: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    **kwargs: Any,
) -> Path:
    with pure_multiseed_profile():
        return multi.prepare(config_or_path, output_root, **kwargs)


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


def _training_summary(root: Path) -> Path:
    destination = root / "analysis/training_summary.csv"
    if destination.is_file():
        return destination
    checkpoints = direct._checkpoint_map(root)
    registry = direct.read_registry(root)
    rows: list[dict[str, Any]] = []
    for job in registry["jobs"]:
        if job.get("parent_state_path") or job.get("recipe_path"):
            raise ValueError(f"Direct job consumed forbidden state: {job['job_id']}")
        state = direct.read_json(direct._status_path(root, str(job["job_id"])))
        best = direct.read_json(
            Path(direct._artifact(state, "best_learned_checkpoint")["path"])
        )
        metrics = pd.read_csv(direct._artifact(state, "training_metrics_csv")["path"])
        epochs = pd.to_numeric(metrics["epoch"], errors="raise").astype(int)
        learned = epochs[epochs >= 1]
        if learned.empty:
            raise ValueError(f"Training metrics have no learned epoch: {job['job_id']}")
        g_trace = list(best.get("generator_lr_trace") or [])
        d_trace = list(best.get("discriminator_lr_trace") or [])
        if not g_trace or not d_trace:
            trace = list(best.get("lr_trace") or [])
            g_trace = [
                {"epoch": row["epoch"], "lr": row.get("g_lr", row.get("lr"))}
                for row in trace
            ]
            d_trace = [
                {"epoch": row["epoch"], "lr": row.get("d_lr", row.get("lr"))}
                for row in trace
            ]
        best_epoch = int(best["best_epoch"])
        if best_epoch < 1 or best_epoch > 240:
            raise ValueError(f"Invalid best learned epoch: {job['job_id']}")
        rows.append(
            {
                "job_id": str(job["job_id"]),
                "seed": int(job["seed"]),
                "fold": str(job["fold"]),
                "arm": str(job["arm"]),
                "best_epoch": best_epoch,
                "epochs_ran": int(learned.max()),
                "final_generator_lr": float(g_trace[-1]["lr"]),
                "final_discriminator_lr": float(d_trace[-1]["lr"]),
                "best_validation_score": float(best["best_metric"]),
                "checkpoint_sha256": checkpoints[str(job["job_id"])]["sha256"],
                "monitor_metric": str(best.get("monitor_metric", "val_hybrid_score")),
                "early_stopped": int(learned.max()) < 240,
            }
        )
    if len(rows) != 40:
        raise ValueError("Training summary requires exactly 40 direct cells")
    return direct.write_csv(destination, rows, tuple(rows[0]))


def _summarize_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "seed",
        "fold",
        "arm",
        "pair_id",
        "session_id",
        "target_mae",
        "persistence_mae",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Pair metrics missing columns: {sorted(required - set(frame))}"
        )
    evidence = frame.copy()
    evidence["seed"] = pd.to_numeric(evidence["seed"], errors="raise").astype(int)
    evidence["target_mae"] = pd.to_numeric(evidence["target_mae"], errors="raise")
    evidence["persistence_mae"] = pd.to_numeric(
        evidence["persistence_mae"], errors="raise"
    )
    if (
        len(evidence) != EXPECTED_PAIR_METRIC_ROWS
        or set(evidence["seed"]) != set(SEEDS)
        or set(evidence["fold"].astype(str)) != set(FOLDS)
        or set(evidence["arm"].astype(str)) != {NO_TEXT_ARM}
        or not np.isfinite(evidence[["target_mae", "persistence_mae"]].to_numpy()).all()
    ):
        raise ValueError("Pure-CNN pair-metric universe drift")
    fold = (
        evidence.groupby(["seed", "fold"], sort=True)
        .agg(
            fold_mae=("target_mae", "mean"),
            fold_persistence_mae=("persistence_mae", "mean"),
            pair_count=("pair_id", "size"),
            session_count=("session_id", "nunique"),
        )
        .reset_index()
    )
    if len(fold) != 40:
        raise ValueError("Expected 40 seed/fold summaries")
    seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        selected = evidence.loc[evidence["seed"].eq(seed)]
        selected_folds = fold.loc[fold["seed"].eq(seed)]
        if len(selected) != 500 or len(selected_folds) != 4:
            raise ValueError(f"Seed {seed} does not contain four folds/500 pairs")
        equal_fold_mae = float(selected_folds["fold_mae"].mean())
        equal_fold_persistence = float(selected_folds["fold_persistence_mae"].mean())
        seed_rows.append(
            {
                "seed": seed,
                "arm": NO_TEXT_ARM,
                "pooled_mae": float(selected["target_mae"].mean()),
                "equal_fold_mae": equal_fold_mae,
                "pooled_persistence_mae": float(selected["persistence_mae"].mean()),
                "equal_fold_persistence_mae": equal_fold_persistence,
                "equal_fold_improvement_vs_persistence_percent": 100.0
                * (1.0 - equal_fold_mae / equal_fold_persistence),
                "pair_rows": len(selected),
                "unique_market_pairs": selected["pair_id"].nunique(),
                "fold_session_count_sum": int(selected_folds["session_count"].sum()),
            }
        )
    seeds = pd.DataFrame(seed_rows).sort_values(
        ["equal_fold_mae", "seed"], kind="stable"
    )
    seeds.insert(0, "best_observed_seed_rank", np.arange(1, len(seeds) + 1))
    overall = pd.DataFrame(
        [
            {
                "arm": NO_TEXT_ARM,
                "seed_count": len(seeds),
                "market_pair_count_per_seed": 500,
                "pair_metric_rows": len(evidence),
                "mean_seed_equal_fold_mae": float(seeds["equal_fold_mae"].mean()),
                "median_seed_equal_fold_mae": float(seeds["equal_fold_mae"].median()),
                "sd_seed_equal_fold_mae": float(seeds["equal_fold_mae"].std(ddof=1)),
                "minimum_seed_equal_fold_mae": float(seeds["equal_fold_mae"].min()),
                "maximum_seed_equal_fold_mae": float(seeds["equal_fold_mae"].max()),
                "best_observed_seed": int(seeds.iloc[0]["seed"]),
                "best_observed_seed_equal_fold_mae": float(
                    seeds.iloc[0]["equal_fold_mae"]
                ),
                "best_observed_seed_is_descriptive_only": True,
                "test_based_seed_selection_permitted": False,
            }
        ]
    )
    return seeds.reset_index(drop=True), overall


def _analysis_manifest(
    root: Path,
    inputs: Sequence[tuple[str, Path]],
    artifacts: Sequence[tuple[str, Path]],
) -> Path:
    payload = {
        "schema_version": 1,
        "kind": ANALYSIS_MANIFEST_KIND,
        "interpretation": INTERPRETATION,
        "inputs": [direct.manifest_row(role, path) for role, path in inputs],
        "artifacts": [direct.manifest_row(role, path) for role, path in artifacts],
    }
    payload["payload_sha256"] = direct.payload_sha256(payload)
    return direct.write_json(root / "analysis/analysis_manifest.json", payload)


def _analyze_experiment(root: Path) -> Path:
    pair_path = root / "analysis/rq12_pair_metrics.csv.gz"
    training_path = root / "analysis/training_summary.csv"
    seeds, overall = _summarize_pairs(pd.read_csv(pair_path))
    seed_path = direct.write_csv(
        root / "analysis/pure_cnn_seed_summary.csv",
        seeds.to_dict("records"),
        tuple(seeds.columns),
    )
    overall_path = direct.write_csv(
        root / "analysis/pure_cnn_overall_summary.csv",
        overall.to_dict("records"),
        tuple(overall.columns),
    )
    row = overall.iloc[0]
    report = root / "analysis/pure_cnn_10seed_report.md"
    report.write_text(
        "\n".join(
            (
                "# Pure-CNN No-text 10-seed direct-training summary",
                "",
                f"- Architecture: `{GENERATOR_MODE} + {CRITIC_MODE}`.",
                f"- Direct jobs: {EXPECTED_TRAINING_JOBS}; parent jobs: 0; continuation jobs: 0.",
                f"- Mean seed-equal-fold MAE: {row['mean_seed_equal_fold_mae']:.10f}.",
                f"- Median seed-equal-fold MAE: {row['median_seed_equal_fold_mae']:.10f}.",
                f"- Across-seed SD: {row['sd_seed_equal_fold_mae']:.10f}.",
                f"- Best observed seed (descriptive only): {int(row['best_observed_seed'])}, "
                f"MAE={row['best_observed_seed_equal_fold_mae']:.10f}.",
                "",
                "The best-observed seed is an oracle descriptive statistic selected on the same rolling test evidence. It must not replace the all-seed aggregate or be presented as an independently selected model.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return _analysis_manifest(
        root,
        (("pair_metrics", pair_path), ("training_summary", training_path)),
        (
            ("seed_summary", seed_path),
            ("overall_summary", overall_path),
            ("markdown_report", report),
        ),
    )


def _verify_analysis_manifest(path: Path) -> dict[str, Any]:
    payload = direct.read_json(path)
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("kind") != ANALYSIS_MANIFEST_KIND or direct.payload_sha256(
        unsigned
    ) != payload.get("payload_sha256"):
        raise ValueError("Pure-CNN analysis manifest identity drift")
    rows = [
        (section, dict(row))
        for section in ("inputs", "artifacts")
        for row in payload.get(section) or []
    ]
    if not rows or len(
        {(section, row.get("artifact_role")) for section, row in rows}
    ) != len(rows):
        raise ValueError("Analysis manifest roles are empty or duplicated")
    for _section, row in rows:
        target = Path(str(row["path"])).resolve()
        if (
            not target.is_file()
            or target.stat().st_size != int(row["size_bytes"])
            or direct.sha256_file(target) != str(row["sha256"])
        ):
            raise ValueError(f"Analysis artifact drift: {target}")
    return payload


def postprocess(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    with _runtime_profile():
        validate_root(root)
        registry = direct.read_registry(root)
        if registry.get("terminal_complete"):
            return qa(root)
        _validate_predictions(root)
        training_summary = _training_summary(root)
        analysis_manifest = _analyze_experiment(root)
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


def qa(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    with _runtime_profile():
        config = validate_root(root)
        registry = direct.read_registry(root)
        if registry.get("terminal_complete"):
            direct._validate_output_hashes(root)
            return root / "qa.json"
        direct._all_training_complete(root)
        direct._checkpoint_map(root)
        direct._validate_test_inputs(root)
        _validate_predictions(root)
        manifest = direct._verify_frozen_file(
            registry["analysis_manifest_path"], registry["analysis_manifest_sha256"]
        )
        _verify_analysis_manifest(manifest)
        training = pd.read_csv(root / "analysis/training_summary.csv")
        pairs = pd.read_csv(root / "analysis/rq12_pair_metrics.csv.gz")
        if len(training) != 40 or len(pairs) != 5_000:
            raise ValueError("Pure-CNN terminal evidence count drift")
        if set(training["seed"].astype(int)) != set(SEEDS):
            raise ValueError("Training summary seed universe drift")
        jobs = list(registry.get("jobs") or [])
        if any(job.get("parent_state_path") or job.get("recipe_path") for job in jobs):
            raise ValueError("Pure-CNN registry contains parent or recipe state")
        fairness_path = direct._verify_frozen_file(
            registry["initial_state_fairness_path"],
            registry["initial_state_fairness_sha256"],
        )
        fairness = direct.read_json(fairness_path)
        if (
            fairness.get("kind") != FAIRNESS_KIND
            or fairness.get("job_count") != 40
            or fairness.get("parent_jobs") != 0
            or fairness.get("continuation_jobs") != 0
            or not fairness.get("common_within_seed")
            or not fairness.get("distinct_across_seeds")
        ):
            raise ValueError("Pure-CNN initial-state fairness evidence drift")
        qa_payload = {
            "schema_version": 1,
            "kind": QA_KIND,
            "status": "passed",
            "interpretation": INTERPRETATION,
            "training_jobs_completed": 40,
            "prediction_cells": 40,
            "pair_metric_rows": 5_000,
            "seeds": list(SEEDS),
            "parent_jobs": 0,
            "continuation_jobs": 0,
            "branch_recipes": 0,
            "training_protocol": "independent_random_init_v1",
            "checkpoint_selection_mode": "best_learned_dynamic_validation_v1",
            "maximum_epochs": 240,
            "generator_parameters": EXPECTED_PARAMETER_COUNTS["generator"],
            "critic_parameters": EXPECTED_PARAMETER_COUNTS["critic"],
            "text_encoder_registered": False,
            "film_layers_registered": False,
            "test_opened_after_checkpoint_freeze": True,
            "shared_mc64_noise_bank_within_seed_fold": True,
            "best_observed_seed_is_descriptive_only": True,
            "test_based_seed_selection_permitted": False,
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


def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    with pure_multiseed_profile():
        return multi.run_pipeline(config_path, output_dir, resume=resume)


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
    "SEEDS",
    "benchmark",
    "freeze_evaluation",
    "load_config",
    "model_contract",
    "planned_specs",
    "predict",
    "prepare",
    "pure_multiseed_profile",
    "run_pipeline",
    "validate_config",
    "validate_gpu_balance",
]
