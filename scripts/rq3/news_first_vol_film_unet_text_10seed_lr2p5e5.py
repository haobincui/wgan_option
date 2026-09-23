"""Ten-seed direct FiLM text-representation experiment at FiLM LR 2.5e-5.

This branch-local wrapper reuses the audited direct/multi-seed lifecycle.  It
contains four FiLM U-Net arms (matched LP, shuffled LP, BoW, and sentiment),
ten seeds, and four rolling folds.  Every one of the 160 cells starts from a
fresh seed-specific random initialization; parent, continuation, and recipe
state are forbidden.
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

from scripts.rq123 import news_first_vol_film_nolp_10seed as core
from scripts.rq3 import news_first_vol_film_unet_direct_5arm_seed42 as direct
from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_lr_5seed as multiseed,
)


DEFAULT_CONFIG = "configs/rq3/news_first_vol_film_unet_text_10seed_lr2p5e5.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq12_news_first_vol_film_unet_text128_c32_nolp_"
    "direct_text_10seed_lr2p5e5_exact_ttm_rolling_v1"
)
EXPERIMENT_KIND = (
    "rq12_news_first_vol_film_unet_text128_c32_nolp_film_text_10seed_lr2p5e5_rolling_v1"
)
INTERPRETATION = "retrospective_rolling_development_10seed_text_representation"
DIRECT_ARMS = ("lp_matched", "lp_shuffle", "bow", "sentiment")
NO_TEXT_ARM = "__no_text_arm_is_not_part_of_this_subexperiment__"
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
FILM_LEARNING_RATE = 2.5e-5
FILM_LEARNING_RATES: Mapping[str, float] = {
    arm: FILM_LEARNING_RATE for arm in DIRECT_ARMS
}
EXPECTED_TRAINING_JOBS = 160
EXPECTED_PREDICTION_CELLS = 160
EXPECTED_PAIR_METRIC_ROWS = 20_000
EXPECTED_PARAMETER_COUNTS = direct.EXPECTED_PARAMETER_COUNTS
GENERATOR_MODE = direct.GENERATOR_MODE
CRITIC_MODE = direct.CRITIC_MODE
CAPACITY_PROFILE = direct.CAPACITY_PROFILE
WORKER_MODULE = "scripts.rq3.news_first_vol_film_unet_text_10seed_lr2p5e5"
INFERENCE_DETERMINISM_KIND = "film_text_10seed_lr2p5e5_inference_v1"
BENCHMARK_RESULT_KIND = "film_text_10seed_lr2p5e5_full_matrix_epoch1_v1"
REGISTRY_KIND = "film_text_10seed_lr2p5e5_task_registry_v1"
QA_KIND = "film_text_10seed_lr2p5e5_terminal_qa_v1"
ANALYSIS_KIND = "film_text_10seed_lr2p5e5_analysis_manifest_v1"

GENERATOR_OPTIMIZER_PROFILE = multiseed.GENERATOR_OPTIMIZER_PROFILE
BACKBONE_LEARNING_RATE = multiseed.BACKBONE_LEARNING_RATE
TEXT_LEARNING_RATE = multiseed.TEXT_LEARNING_RATE
CRITIC_LEARNING_RATE = multiseed.CRITIC_LEARNING_RATE
BACKBONE_MIN_LEARNING_RATE = multiseed.BACKBONE_MIN_LEARNING_RATE
TEXT_MIN_LEARNING_RATE = multiseed.TEXT_MIN_LEARNING_RATE
GROUP_FLOOR_RATIO = multiseed.GROUP_FLOOR_RATIO
FILM_MIN_LEARNING_RATE = FILM_LEARNING_RATE * GROUP_FLOOR_RATIO

SOURCE_CODE_RELATIVE_PATHS = tuple(
    dict.fromkeys(
        (
            "scripts/rq3/news_first_vol_film_unet_text_10seed_lr2p5e5.py",
            *direct.SOURCE_CODE_RELATIVE_PATHS,
            "scripts/rq3/news_first_vol_film_unet_direct_matched_lr_5seed.py",
        )
    )
)

_BASE_INITIAL_STATE_HASHES = direct._initial_state_hashes
_BASE_SOURCE_PATHS = direct._source_paths
_BASE_MATERIALIZE_TEST_INPUTS = direct._materialize_test_inputs
_BASE_TRAINING_SUMMARY = multiseed._training_summary


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = direct.resolve_path(config_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Ten-seed FiLM-text config must be a mapping")
    config = deepcopy(dict(raw))
    config["source_config_path"] = str(source)
    config["source_config_sha256"] = direct.sha256_file(source)
    validate_config(config)
    return config


def _load_frozen_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Frozen ten-seed FiLM-text config must be a mapping")
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

    prohibited = {
        "parent_arms",
        "continuation_arms",
        "branch_recipes",
        "parent_state_allowlist",
        "lr_replay",
    }
    present = sorted(prohibited & (set(matrix) | set(training)))
    if present:
        raise ValueError(
            "Direct experiment contains prohibited parent/continuation state: "
            f"{present}"
        )

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
        raise ValueError("Text overlay dimension must be 1024")

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
        raise ValueError("Ten-seed universe/order drift")
    if tuple(map(int, matrix.get("tolerances_minutes", ()))) != (5,):
        raise ValueError("matrix.tolerances_minutes must be [5]")
    if tuple(matrix.get("direct_arms", ())) != DIRECT_ARMS:
        raise ValueError("Four-arm text representation order drift")
    rates = {
        str(key): float(value)
        for key, value in _mapping(
            matrix.get("arm_film_learning_rates"), "arm FiLM LRs"
        ).items()
    }
    if rates != dict(FILM_LEARNING_RATES):
        raise ValueError("Every text arm must use FiLM LR 2.5e-5")
    for key, expected in {
        "expected_training_jobs": EXPECTED_TRAINING_JOBS,
        "expected_prediction_cells": EXPECTED_PREDICTION_CELLS,
        "expected_pair_metric_rows": EXPECTED_PAIR_METRIC_ROWS,
    }.items():
        if int(matrix.get(key, -1)) != expected:
            raise ValueError(f"matrix.{key} drift")

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
        "initial_learning_rate": BACKBONE_LEARNING_RATE,
        "generator_learning_rate": BACKBONE_LEARNING_RATE,
        "generator_text_learning_rate": TEXT_LEARNING_RATE,
        "generator_film_learning_rate": FILM_LEARNING_RATE,
        "generator_text_min_learning_rate": TEXT_MIN_LEARNING_RATE,
        "generator_film_min_learning_rate": FILM_MIN_LEARNING_RATE,
        "discriminator_learning_rate": CRITIC_LEARNING_RATE,
        "scheduler_min_lr": BACKBONE_MIN_LEARNING_RATE,
        "group_scheduler_floor_ratio": GROUP_FLOOR_RATIO,
    }.items():
        if not math.isclose(
            float(training.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1e-18
        ):
            raise ValueError(f"training.{key} drift")

    representations = _mapping(
        config.get("text_representations"), "text_representations"
    )
    expected_modes = {
        "lp_matched": "lp_pair_mean_l2_v1",
        "lp_shuffle": "lp_pair_mean_l2_v1",
        "bow": "bow_train_pair_vocab_log1p_l2_v1",
        "sentiment": "sentiment_pair_mean_train_zscore_pad_v1",
    }
    if set(representations) != set(DIRECT_ARMS) or any(
        _mapping(representations[arm], f"text_representations.{arm}").get("mode")
        != mode
        for arm, mode in expected_modes.items()
    ):
        raise ValueError("Text-representation contract drift")
    if (
        analysis.get("interpretation") != INTERPRETATION
        or int(analysis.get("bootstrap_replicates", -1)) != 10_000
        or not bool(analysis.get("cross_seed_inference_enabled"))
        or analysis.get("best_seed_reporting_mode") != "descriptive_test_selected_v1"
        or bool(analysis.get("test_based_model_selection_permitted", True))
    ):
        raise ValueError("Analysis contract drift")

    if tuple(map(int, runtime.get("gpu_ids", ()))) != (0, 1):
        raise ValueError("Runtime requires GPU 0 and GPU 1")
    if int(runtime.get("benchmark_jobs", -1)) != EXPECTED_TRAINING_JOBS:
        raise ValueError("Benchmark must exercise all 160 cells")
    if int(runtime.get("benchmark_epochs", -1)) != 1:
        raise ValueError("Benchmark must use one epoch")
    primary = int(runtime.get("benchmark_workers_per_gpu", -1))
    fallback = int(runtime.get("fallback_workers_per_gpu", -1))
    if primary < 1 or fallback < 1 or fallback > primary:
        raise ValueError("Invalid benchmark/fallback concurrency")
    formal = runtime.get("formal_workers_per_gpu")
    if formal is not None and int(formal) not in (primary, fallback):
        raise ValueError("Frozen formal concurrency was not benchmarked")

    with multiseed._core_profile():
        core.architecture_profile_contract(config)


def _overlay_mode(arm: str) -> str:
    return {
        "lp_matched": "lp_mean_l2",
        "lp_shuffle": "lp_shuffle",
        "bow": "bow1024",
        "sentiment": "sentiment_pad1024",
    }[str(arm)]


def _initial_state_hashes(config: Mapping[str, Any], seed: int) -> dict[str, str]:
    original = direct.SEED
    try:
        direct.SEED = int(seed)
        return _BASE_INITIAL_STATE_HASHES(config)
    finally:
        direct.SEED = original


def _job_id(seed: int, fold: str, arm: str) -> str:
    return core.job_id(
        {
            "stage": direct.DIRECT_STAGE,
            "tolerance_minutes": 5,
            "fold": fold,
            "seed": int(seed),
            "arm": arm,
        }
    )


def planned_specs(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
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
    specs: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            for arm in DIRECT_ARMS:
                specs.append(
                    {
                        "stage": direct.DIRECT_STAGE,
                        "tolerance_minutes": 5,
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "gpu_id": int(assignment[fold]),
                        "job_id": _job_id(seed, fold, arm),
                        "pair_text_overlay_mode": _overlay_mode(arm),
                        "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
                        "generator_text_learning_rate": TEXT_LEARNING_RATE,
                        "generator_film_learning_rate": FILM_LEARNING_RATE,
                        "generator_text_min_learning_rate": TEXT_MIN_LEARNING_RATE,
                        "generator_film_min_learning_rate": FILM_MIN_LEARNING_RATE,
                        **initial_by_seed[seed],
                    }
                )
    if (
        len(specs) != EXPECTED_TRAINING_JOBS
        or len({str(row["job_id"]) for row in specs}) != EXPECTED_TRAINING_JOBS
    ):
        raise AssertionError("Ten-seed FiLM-text sweep requires 160 unique jobs")
    validate_gpu_balance(specs)
    return specs


def validate_gpu_balance(specs: Sequence[Mapping[str, Any]]) -> None:
    blocks: dict[tuple[int, str], int] = {}
    for row in specs:
        key = (int(row["seed"]), str(row["fold"]))
        gpu = int(row["gpu_id"])
        if blocks.setdefault(key, gpu) != gpu:
            raise ValueError(f"Text arms in block {key} do not share one GPU")
    counts = {gpu: sum(int(row["gpu_id"]) == gpu for row in specs) for gpu in (0, 1)}
    if counts != {0: 80, 1: 80}:
        raise ValueError(f"GPU job balance must be 80/80, got {counts}")
    expected = {
        (seed, fold): (0 if fold in {"f1_2023q1", "f3_2023q3"} else 1)
        for seed in SEEDS
        for fold in FOLDS
    }
    if blocks != expected:
        raise ValueError(f"Frozen seed/fold GPU assignment drift: {blocks}")


def _source_paths(config: Mapping[str, Any]) -> list[tuple[str, Path]]:
    return list(_BASE_SOURCE_PATHS(config))


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
        raise ValueError(f"Unknown FiLM-text arm: {arm}")
    training = _mapping(config["training"], "training")
    model = _mapping(config["model"], "model")
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
        "news_first_pair_text_overlay_mode": _overlay_mode(arm),
        "news_first_full_training_state_mode": "save_dynamic_v1",
        "news_first_refit_mode": "none",
        "use_reduce_lr_on_plateau": True,
        "use_early_stopping": True,
        "seed": int(spec["seed"]),
    }


def _materialize_pair_text_overlays(config: Mapping[str, Any], root: Path) -> Path:
    from wgan_option.utils.news_first_experiment_core import (
        build_pair_text_overlay_manifests,
    )

    output_dir = root / "inputs/pair_text_overlays"
    generated = build_pair_text_overlay_manifests(
        config=config,
        pair_universe_path=root / "inputs/pair_universes.csv",
        output_dir=output_dir,
    )
    if len(generated) != 24:
        raise ValueError("Shared builder did not produce the canonical 24 overlays")
    expected = {
        direct._overlay_path(root, fold, arm) for fold in FOLDS for arm in DIRECT_ARMS
    }
    for path in output_dir.glob("tolerance_*/*/*.json"):
        if path.resolve() not in expected:
            path.unlink()
    paths = sorted(output_dir.glob("tolerance_05m/*/*.json"))
    if len(paths) != 16 or {path.resolve() for path in paths} != expected:
        raise ValueError("FiLM-text development overlay universe drift")
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = direct.read_json(path)
        unsigned = {
            key: value for key, value in payload.items() if key != "profile_sha256"
        }
        if direct.payload_sha256(unsigned) != payload.get("profile_sha256"):
            raise ValueError(f"Overlay self-hash drift: {path}")
        if payload.get("mode") != _overlay_mode(path.stem):
            raise ValueError(f"Overlay mode drift: {path}")
        records = list(payload.get("records") or [])
        if not records:
            raise ValueError(f"Overlay is empty: {path}")
        if path.stem == "lp_shuffle" and any(
            str(row["pair_id"]) == str(row.get("donor_pair_id", "")) for row in records
        ):
            raise ValueError("lp_shuffle contains a fixed point")
        rows.append(direct.manifest_row(f"pair_overlay:{path.relative_to(root)}", path))
    return direct._write_hash_manifest(
        root / "inputs/pair_text_overlay_hashes.csv", rows
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
        if observed_g != job["initial_generator_state_sha256"]:
            raise ValueError(f"Generator initial-state drift: {job['job_id']}")
        if observed_d != job["initial_critic_state_sha256"]:
            raise ValueError(f"Critic initial-state drift: {job['job_id']}")
        best_g_sha = direct._checkpoint_state_sha256(best_g["path"])
        best_d_sha = direct._checkpoint_state_sha256(best_d["path"])
        if observed_g == best_g_sha or observed_d == best_d_sha:
            raise ValueError(f"G/D parameters did not both update: {job['job_id']}")
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
        if len(selected) != 16 or len(g_hashes) != 1 or len(d_hashes) != 1:
            raise ValueError(f"Epoch-0 fairness drift within seed {seed}")
        by_seed[str(seed)] = {
            "job_count": 16,
            "generator_initial_state_sha256": next(iter(g_hashes)),
            "critic_initial_state_sha256": next(iter(d_hashes)),
        }
    if len({row["generator_initial_state_sha256"] for row in by_seed.values()}) != 10:
        raise ValueError("Generator initial states are not distinct across seeds")
    if len({row["critic_initial_state_sha256"] for row in by_seed.values()}) != 10:
        raise ValueError("Critic initial states are not distinct across seeds")
    payload = {
        "schema_version": 1,
        "kind": "film_text_10seed_lr2p5e5_initial_state_fairness_v1",
        "seeds": list(SEEDS),
        "initialization_order": "seed_everything(seed), Generator, Critic",
        "job_count": len(rows),
        "common_within_seed": True,
        "distinct_across_seeds": True,
        "all_generator_parameters_updated": True,
        "all_critic_parameters_updated": True,
        "by_seed": by_seed,
        "rows": sorted(rows, key=lambda row: row["job_id"]),
    }
    payload["payload_sha256"] = direct.payload_sha256(payload)
    return direct.write_json(direct._initial_state_fairness_path(root), payload)


def _materialize_test_inputs(config: Mapping[str, Any], root: Path) -> Path:
    # The direct implementation already handles matched/shuffle/BoW/sentiment.
    # Under ``multiseed_profile`` its module globals describe this four-arm root.
    return _BASE_MATERIALIZE_TEST_INPUTS(config, root)


def _analysis_bundle(root: Path, config: Mapping[str, Any]) -> Path:
    pair_path = root / "analysis/rq12_pair_metrics.csv.gz"
    training_path = root / "analysis/training_summary.csv"
    pairs = pd.read_csv(pair_path)
    training = pd.read_csv(training_path)
    if (
        len(pairs) != EXPECTED_PAIR_METRIC_ROWS
        or len(training) != EXPECTED_TRAINING_JOBS
    ):
        raise ValueError("Analysis evidence count drift")
    expected_cells = {
        (seed, fold, arm) for seed in SEEDS for fold in FOLDS for arm in DIRECT_ARMS
    }
    observed_cells = set(
        pairs[["seed", "fold", "arm"]]
        .assign(
            seed=lambda frame: pd.to_numeric(frame["seed"], errors="raise").astype(int)
        )
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise ValueError("Analysis seed/fold/arm universe drift")

    cell_rows: list[dict[str, Any]] = []
    for (seed, fold, arm), group in pairs.groupby(["seed", "fold", "arm"], sort=True):
        expected_pairs = direct._expected_counts(config, str(fold))["test_pairs"]
        if len(group) != expected_pairs:
            raise ValueError(
                f"Pair count drift for seed={seed}, fold={fold}, arm={arm}"
            )
        model_mae = float(pd.to_numeric(group["target_mae"], errors="raise").mean())
        persistence_mae = float(
            pd.to_numeric(group["persistence_mae"], errors="raise").mean()
        )
        cell_rows.append(
            {
                "seed": int(seed),
                "fold": str(fold),
                "arm": str(arm),
                "mean_mae": model_mae,
                "persistence_mae": persistence_mae,
                "improvement_vs_persistence_percent": 100.0
                * (1.0 - model_mae / persistence_mae),
                "pair_count": int(len(group)),
                "session_count": int(group["session_id"].nunique()),
            }
        )
    cells = pd.DataFrame(cell_rows).sort_values(["arm", "seed", "fold"], kind="stable")
    seed_arm = (
        cells.groupby(["arm", "seed"], as_index=False)
        .agg(
            equal_fold_mean_mae=("mean_mae", "mean"),
            equal_fold_persistence_mae=("persistence_mae", "mean"),
            fold_count=("fold", "nunique"),
        )
        .sort_values(["arm", "equal_fold_mean_mae", "seed"], kind="stable")
    )
    seed_arm["improvement_vs_persistence_percent"] = 100.0 * (
        1.0 - seed_arm["equal_fold_mean_mae"] / seed_arm["equal_fold_persistence_mae"]
    )
    seed_arm["descriptive_seed_rank"] = (
        seed_arm.groupby("arm")["equal_fold_mean_mae"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    best = seed_arm.loc[seed_arm["descriptive_seed_rank"].eq(1)].copy()
    best["selection_scope"] = "test_selected_descriptive_only"
    arms = (
        seed_arm.groupby("arm", as_index=False)
        .agg(
            ten_seed_equal_fold_mean_mae=("equal_fold_mean_mae", "mean"),
            seed_mae_std=("equal_fold_mean_mae", "std"),
            seed_count=("seed", "nunique"),
        )
        .sort_values("ten_seed_equal_fold_mean_mae", kind="stable")
    )
    arms["descriptive_arm_rank"] = range(1, len(arms) + 1)

    direct.core._write_dataframe_csv(root / "analysis/seed_fold_arm_summary.csv", cells)
    direct.core._write_dataframe_csv(root / "analysis/seed_arm_summary.csv", seed_arm)
    direct.core._write_dataframe_csv(root / "analysis/best_seed_by_arm.csv", best)
    direct.core._write_dataframe_csv(root / "analysis/arm_summary.csv", arms)
    summary = {
        "schema_version": 1,
        "kind": "film_text_10seed_lr2p5e5_analysis_summary_v1",
        "interpretation": INTERPRETATION,
        "seeds": list(SEEDS),
        "arms": list(DIRECT_ARMS),
        "training_jobs": EXPECTED_TRAINING_JOBS,
        "prediction_cells": EXPECTED_PREDICTION_CELLS,
        "pair_metric_rows": EXPECTED_PAIR_METRIC_ROWS,
        "best_seed_reporting_mode": "descriptive_test_selected_v1",
        "best_seed_is_unbiased_model_estimate": False,
        "test_based_model_selection_permitted": False,
        "best_seed_by_arm": {
            str(row.arm): {
                "seed": int(row.seed),
                "equal_fold_mean_mae": float(row.equal_fold_mean_mae),
            }
            for row in best.itertuples(index=False)
        },
    }
    summary_path = direct.write_json(root / "analysis/analysis_summary.json", summary)
    report_lines = [
        "# FiLM text 10-seed LR=2.5e-5",
        "",
        "All 160 cells are independent direct fits with no parent or continuation.",
        "The best seed per arm is test-selected and therefore descriptive/optimistic; "
        "the ten-seed mean remains the primary stability summary.",
        "",
        "| arm | 10-seed mean MAE | seed SD | best seed | best-seed MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    best_by_arm = best.set_index("arm")
    for row in arms.itertuples(index=False):
        selected = best_by_arm.loc[str(row.arm)]
        report_lines.append(
            f"| {row.arm} | {row.ten_seed_equal_fold_mean_mae:.10f} | "
            f"{row.seed_mae_std:.10f} | {int(selected['seed'])} | "
            f"{float(selected['equal_fold_mean_mae']):.10f} |"
        )
    report_path = root / "analysis/film_text_10seed_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    artifacts = [
        root / "analysis/seed_fold_arm_summary.csv",
        root / "analysis/seed_arm_summary.csv",
        root / "analysis/best_seed_by_arm.csv",
        root / "analysis/arm_summary.csv",
        summary_path,
        report_path,
    ]
    manifest = {
        "schema_version": 1,
        "kind": ANALYSIS_KIND,
        "interpretation": INTERPRETATION,
        "inputs": [
            direct.manifest_row("pair_metrics", pair_path),
            direct.manifest_row("training_summary", training_path),
        ],
        "artifacts": [
            direct.manifest_row(f"analysis:{path.name}", path) for path in artifacts
        ],
    }
    manifest["payload_sha256"] = direct.payload_sha256(manifest)
    return direct.write_json(root / "analysis/analysis_manifest.json", manifest)


def _verify_analysis_manifest(path: Path) -> dict[str, Any]:
    payload = direct.read_json(path)
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("kind") != ANALYSIS_KIND or direct.payload_sha256(
        unsigned
    ) != payload.get("payload_sha256"):
        raise ValueError("FiLM-text analysis manifest drift")
    for section in ("inputs", "artifacts"):
        direct.verify_manifest_rows(payload.get(section) or [])
    return payload


def _postprocess_active(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    config = multiseed.validate_root(root)
    registry = direct.read_registry(root)
    if registry.get("terminal_complete"):
        return _qa_active(root)
    multiseed._validate_predictions(root)
    training_summary = _BASE_TRAINING_SUMMARY(root)
    analysis_manifest = _analysis_bundle(root, config)
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


def _qa_active(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    config = multiseed.validate_root(root)
    registry = direct.read_registry(root)
    if registry.get("terminal_complete"):
        direct._validate_output_hashes(root)
        return root / "qa.json"
    direct._all_training_complete(root)
    direct._checkpoint_map(root)
    direct._validate_test_inputs(root)
    multiseed._validate_predictions(root)
    manifest = direct._verify_frozen_file(
        registry["analysis_manifest_path"], registry["analysis_manifest_sha256"]
    )
    _verify_analysis_manifest(manifest)
    training = pd.read_csv(root / "analysis/training_summary.csv")
    pairs = pd.read_csv(root / "analysis/rq12_pair_metrics.csv.gz")
    if (
        len(training) != EXPECTED_TRAINING_JOBS
        or len(pairs) != EXPECTED_PAIR_METRIC_ROWS
    ):
        raise ValueError("Terminal evidence count drift")
    jobs = [dict(job) for job in registry.get("jobs") or []]
    if len(jobs) != EXPECTED_TRAINING_JOBS or any(
        str(job.get("stage")) != direct.DIRECT_STAGE
        or str(job.get("parent_state_path", ""))
        or str(job.get("recipe_path", ""))
        or str(job.get("continuation_job_id", ""))
        for job in jobs
    ):
        raise ValueError("Direct-only no-parent/no-continuation registry drift")
    fairness_path = direct._verify_frozen_file(
        registry["initial_state_fairness_path"],
        registry["initial_state_fairness_sha256"],
    )
    fairness = direct.read_json(fairness_path)
    if (
        fairness.get("job_count") != EXPECTED_TRAINING_JOBS
        or not fairness.get("common_within_seed")
        or not fairness.get("distinct_across_seeds")
        or set(map(int, fairness.get("seeds", ()))) != set(SEEDS)
    ):
        raise ValueError("Initial-state fairness evidence drift")
    qa_payload = {
        "schema_version": 1,
        "kind": QA_KIND,
        "status": "passed",
        "interpretation": INTERPRETATION,
        "training_jobs_completed": EXPECTED_TRAINING_JOBS,
        "prediction_cells": EXPECTED_PREDICTION_CELLS,
        "pair_metric_rows": EXPECTED_PAIR_METRIC_ROWS,
        "seeds": list(SEEDS),
        "arms": list(DIRECT_ARMS),
        "parent_jobs": 0,
        "continuation_jobs": 0,
        "branch_recipes": 0,
        "maximum_epochs": 240,
        "checkpoint_selection_mode": "best_learned_dynamic_validation_v1",
        "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
        "film_learning_rate": FILM_LEARNING_RATE,
        "text_learning_rate": TEXT_LEARNING_RATE,
        "backbone_learning_rate": BACKBONE_LEARNING_RATE,
        "critic_learning_rate": CRITIC_LEARNING_RATE,
        "test_opened_after_checkpoint_freeze": True,
        "shared_mc64_noise_bank_within_seed_fold": True,
        "source_config_sha256": config["source_config_sha256"],
        "completed_at_utc": direct.utc_now(),
    }
    qa_payload["payload_sha256"] = direct.payload_sha256(qa_payload)
    qa_path = direct.write_json(root / "qa.json", qa_payload)
    registry.update(
        status="completed", terminal_complete=True, completed_at_utc=direct.utc_now()
    )
    direct.write_registry(root, registry)
    direct._experiment_status(root, "completed", current_stage="terminal")
    rows = [
        {
            "artifact_role": f"output:{path.relative_to(root).as_posix()}",
            "relative_path": path.relative_to(root).as_posix(),
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": direct.sha256_file(path),
        }
        for path in direct._terminal_files(root)
    ]
    direct.write_csv(root / "output_hashes.csv", rows, tuple(rows[0]))
    direct._validate_output_hashes(root)
    return qa_path


@contextmanager
def film_text_profile() -> Iterator[None]:
    """Temporarily install this 160-cell profile into the reusable wrapper."""

    replacements: dict[str, Any] = {
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "EXPERIMENT_KIND": EXPERIMENT_KIND,
        "INTERPRETATION": INTERPRETATION,
        "DIRECT_ARMS": DIRECT_ARMS,
        "NO_TEXT_ARM": NO_TEXT_ARM,
        "SEEDS": SEEDS,
        "FILM_LEARNING_RATES": FILM_LEARNING_RATES,
        "EXPECTED_TRAINING_JOBS": EXPECTED_TRAINING_JOBS,
        "EXPECTED_PREDICTION_CELLS": EXPECTED_PREDICTION_CELLS,
        "EXPECTED_PAIR_METRIC_ROWS": EXPECTED_PAIR_METRIC_ROWS,
        "WORKER_MODULE": WORKER_MODULE,
        "INFERENCE_DETERMINISM_KIND": INFERENCE_DETERMINISM_KIND,
        "BENCHMARK_RESULT_KIND": BENCHMARK_RESULT_KIND,
        "REGISTRY_KIND": REGISTRY_KIND,
        "QA_KIND": QA_KIND,
        "SOURCE_CODE_RELATIVE_PATHS": SOURCE_CODE_RELATIVE_PATHS,
        "load_config": load_config,
        "_load_frozen_config": _load_frozen_config,
        "validate_config": validate_config,
        "planned_specs": planned_specs,
        "validate_gpu_balance": validate_gpu_balance,
        "_source_paths": _source_paths,
        "_overlay_mode": _overlay_mode,
        "_materialize_pair_text_overlays": _materialize_pair_text_overlays,
        "_training_payload": _training_payload,
        "_freeze_initial_state_fairness": _freeze_initial_state_fairness,
        "_materialize_test_inputs": _materialize_test_inputs,
        "postprocess": _postprocess_active,
        "qa": _qa_active,
    }
    originals = {name: getattr(multiseed, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(multiseed, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(multiseed, name, value)


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    with film_text_profile():
        return getattr(multiseed, name)(*args, **kwargs)


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
    return _call("prepare", config_or_path, output_root, **kwargs)


def dry_run(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    return benchmark(config_path, output_dir, resume=resume)


def worker(
    output_dir: str | Path, job_id: str, *, resume: bool = False, dry_run: bool = False
) -> Path:
    return _call("worker", output_dir, job_id, resume=resume, dry_run=dry_run)


def launch(output_dir: str | Path, *, resume: bool = False) -> Path:
    return _call("launch", output_dir, resume=resume)


def freeze_evaluation(output_dir: str | Path, *, resume: bool = False) -> Path:
    return _call("freeze_evaluation", output_dir, resume=resume)


def predict(output_dir: str | Path, *, resume: bool = False) -> Path:
    return _call("predict", output_dir, resume=resume)


def postprocess(output_dir: str | Path, *, resume: bool = False) -> Path:
    with film_text_profile():
        return _postprocess_active(output_dir, resume=resume)


def qa(output_dir: str | Path, *, resume: bool = False) -> Path:
    with film_text_profile():
        return _qa_active(output_dir, resume=resume)


def status(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    return _call("status", output_dir)


def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    return _call("run_pipeline", config_path, output_dir, resume=resume)


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
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "DIRECT_ARMS",
    "EXPECTED_PAIR_METRIC_ROWS",
    "EXPECTED_PREDICTION_CELLS",
    "EXPECTED_TRAINING_JOBS",
    "FILM_LEARNING_RATE",
    "SEEDS",
    "film_text_profile",
    "load_config",
    "main",
    "planned_specs",
    "run_pipeline",
    "validate_config",
]
