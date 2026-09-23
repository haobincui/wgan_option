"""Five-seed robustness run for two frozen FiLM learning rates.

The seed-42 learning-rate sweep selected ``1e-5`` and ``2.5e-5`` as the two
post-selection candidates.  This branch-local experiment validates those two
rates on five fresh seeds and four rolling folds.  Every cell is trained from
scratch; there is no parent/continuation state and no test-based LR selection.
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

from scripts.rq123 import news_first_vol_film_nolp_10seed as core
from scripts.rq123 import news_first_vol_film_unet_nolp_10seed as unet
from scripts.rq3 import news_first_vol_film_unet_direct_5arm_seed42 as direct
from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_lr_5seed_analysis as analysis,
)
from scripts.rq3 import news_first_vol_film_unet_direct_matched_lr_seed42 as single_lr


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/rq3/news_first_vol_film_unet_direct_matched_lr_5seed.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_lr_"
    "1e5_2p5e5_5seed_exact_ttm_rolling_v1"
)
EXPERIMENT_KIND = (
    "rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_lr_"
    "1e5_2p5e5_5seed_rolling_v1"
)
INTERPRETATION = "retrospective_rolling_development_post_selection_seed_robustness"
DIRECT_STAGE = "direct_arms"
DIRECT_ARMS = ("film_lr_1e5", "film_lr_2p5e5")
FILM_LEARNING_RATES: Mapping[str, float] = {
    "film_lr_1e5": 1.0e-5,
    "film_lr_2p5e5": 2.5e-5,
}
SEEDS = (202, 404, 382624741, 1607127774, 1662128673)
FOLDS = direct.FOLDS
TOLERANCE_MINUTES = 5
NO_TEXT_ARM = "__unused_no_text_arm__"
EXPECTED_TRAINING_JOBS = 40
EXPECTED_PREDICTION_CELLS = 40
EXPECTED_PAIR_METRIC_ROWS = 5_000
EXPECTED_PARAMETER_COUNTS = direct.EXPECTED_PARAMETER_COUNTS
GENERATOR_MODE = direct.GENERATOR_MODE
CRITIC_MODE = direct.CRITIC_MODE
CAPACITY_PROFILE = direct.CAPACITY_PROFILE
EXPECTED_ARCHITECTURE_PROFILE_SHA256 = direct.EXPECTED_ARCHITECTURE_PROFILE_SHA256
WORKER_MODULE = "scripts.rq3.news_first_vol_film_unet_direct_matched_lr_5seed"
INFERENCE_DETERMINISM_KIND = "direct_matched_film_lr_5seed_inference_determinism_v1"
BENCHMARK_RESULT_KIND = "direct_matched_film_lr_5seed_full_matrix_epoch1_benchmark_v1"
REGISTRY_KIND = "direct_matched_film_lr_5seed_task_registry_v1"
QA_KIND = "direct_matched_film_lr_5seed_terminal_qa_v1"
GENERATOR_OPTIMIZER_PROFILE = single_lr.GENERATOR_OPTIMIZER_PROFILE
BACKBONE_LEARNING_RATE = single_lr.BACKBONE_LEARNING_RATE
TEXT_LEARNING_RATE = single_lr.TEXT_LEARNING_RATE
CRITIC_LEARNING_RATE = single_lr.CRITIC_LEARNING_RATE
BACKBONE_MIN_LEARNING_RATE = single_lr.BACKBONE_MIN_LEARNING_RATE
TEXT_MIN_LEARNING_RATE = single_lr.TEXT_MIN_LEARNING_RATE
GROUP_FLOOR_RATIO = single_lr.GROUP_FLOOR_RATIO
SOURCE_CODE_RELATIVE_PATHS = (
    "scripts/rq3/news_first_vol_film_unet_direct_matched_lr_5seed.py",
    "scripts/rq3/news_first_vol_film_unet_direct_matched_lr_5seed_analysis.py",
    "scripts/rq3/news_first_vol_film_unet_direct_5arm_seed42.py",
    "scripts/rq3/news_first_vol_film_unet_direct_matched_lr_seed42.py",
    "scripts/rq3/news_first_vol_film_unet_direct_matched_lr_seed42_analysis.py",
    "scripts/rq123/news_first_vol_film_nolp_10seed.py",
    "scripts/rq123/news_first_vol_film_unet_nolp_10seed.py",
    "scripts/rq3/news_first_vol_comparison_analysis.py",
    "scripts/rq3/news_first_vol_training.py",
    "scripts/rq2_pair/pair_features.py",
    "scripts/rq2_pair/rq2_pair_experiment.py",
)

_BASE_INITIAL_STATE_HASHES = direct._initial_state_hashes
_BASE_SOURCE_PATHS = direct._source_paths


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _selection_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(
        _mapping(config.get("analysis"), "analysis").get("selection_provenance"),
        "analysis.selection_provenance",
    )


def _validate_selection_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    provenance = _selection_provenance(config)
    discovery_seed = int(provenance.get("discovery_seed", -1))
    if discovery_seed != 42 or discovery_seed in SEEDS:
        raise ValueError("Discovery seed 42 must be excluded from five-seed validation")
    if not bool(provenance.get("excluded_from_validation")):
        raise ValueError("Selection provenance must declare discovery-seed exclusion")
    source_root = direct.resolve_path(provenance["experiment_root"])
    report = direct.resolve_path(provenance["report_path"])
    output_manifest = direct.resolve_path(provenance["output_hash_manifest_path"])
    if source_root not in report.parents or source_root not in output_manifest.parents:
        raise ValueError("Selection provenance paths must remain inside discovery root")
    if direct.sha256_file(report) != str(provenance["report_sha256"]):
        raise ValueError("Selection report SHA drift")
    if direct.sha256_file(output_manifest) != str(
        provenance["output_hash_manifest_sha256"]
    ):
        raise ValueError("Selection output-manifest SHA drift")
    manifest = pd.read_csv(output_manifest, dtype=str, keep_default_na=False)
    relative = report.relative_to(source_root).as_posix()
    selected = manifest.loc[manifest["relative_path"].eq(relative)]
    if (
        len(selected) != 1
        or selected.iloc[0]["sha256"] != str(provenance["report_sha256"])
        or int(selected.iloc[0]["size_bytes"]) != report.stat().st_size
    ):
        raise ValueError("Discovery output manifest does not bind selection report")
    return provenance


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = direct.resolve_path(config_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Five-seed FiLM-LR config must be a mapping")
    config = deepcopy(dict(raw))
    config["source_config_path"] = str(source)
    config["source_config_sha256"] = direct.sha256_file(source)
    validate_config(config)
    return config


def _load_frozen_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Frozen five-seed FiLM-LR config must be a mapping")
    config = dict(raw)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    experiment = _mapping(config.get("experiment"), "experiment")
    data = _mapping(config.get("data"), "data")
    matrix = _mapping(config.get("matrix"), "matrix")
    model = _mapping(config.get("model"), "model")
    training = _mapping(config.get("training"), "training")
    analysis_config = _mapping(config.get("analysis"), "analysis")
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
        raise ValueError("Five fresh validation seeds/order drift")
    if tuple(map(int, matrix.get("tolerances_minutes", ()))) != (5,):
        raise ValueError("matrix.tolerances_minutes must be [5]")
    if tuple(matrix.get("direct_arms", ())) != DIRECT_ARMS:
        raise ValueError("Frozen two-arm order drift")
    rates = {
        str(key): float(value)
        for key, value in _mapping(
            matrix.get("arm_film_learning_rates"), "arm FiLM LRs"
        ).items()
    }
    if rates != dict(FILM_LEARNING_RATES):
        raise ValueError("FiLM learning-rate arm contract drift")
    expected_matrix = {
        "expected_training_jobs": EXPECTED_TRAINING_JOBS,
        "expected_prediction_cells": EXPECTED_PREDICTION_CELLS,
        "expected_pair_metric_rows": EXPECTED_PAIR_METRIC_ROWS,
    }
    for key, expected in expected_matrix.items():
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
    fixed_rates = {
        "initial_learning_rate": BACKBONE_LEARNING_RATE,
        "generator_learning_rate": BACKBONE_LEARNING_RATE,
        "generator_text_learning_rate": TEXT_LEARNING_RATE,
        "generator_text_min_learning_rate": TEXT_MIN_LEARNING_RATE,
        "discriminator_learning_rate": CRITIC_LEARNING_RATE,
        "scheduler_min_lr": BACKBONE_MIN_LEARNING_RATE,
        "group_scheduler_floor_ratio": GROUP_FLOOR_RATIO,
    }
    for key, expected in fixed_rates.items():
        if not math.isclose(
            float(training.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1e-18
        ):
            raise ValueError(f"training.{key} drift")
    representations = _mapping(
        config.get("text_representations"), "text_representations"
    )
    if set(representations) != set(DIRECT_ARMS) or any(
        _mapping(representations[arm], f"text_representations.{arm}").get("mode")
        != "lp_pair_mean_l2_v1"
        for arm in DIRECT_ARMS
    ):
        raise ValueError("Every LR arm must use matched pair-level LP")
    if analysis_config.get("interpretation") != INTERPRETATION:
        raise ValueError("analysis.interpretation drift")
    if int(analysis_config.get("bootstrap_replicates", -1)) != 10_000:
        raise ValueError("Analysis requires 10,000 bootstrap replicates")
    if analysis_config.get("primary_family") != "film_lr_2p5e5_vs_film_lr_1e5":
        raise ValueError("Primary comparison must remain frozen")
    if bool(analysis_config.get("test_based_lr_selection_permitted", True)):
        raise ValueError("Test-based LR selection is prohibited")
    if not bool(analysis_config.get("cross_seed_inference_enabled")):
        raise ValueError("Cross-seed inference must be enabled")
    if (
        tuple(analysis_config.get("frozen_post_selection_candidates", ()))
        != DIRECT_ARMS
    ):
        raise ValueError("Frozen post-selection candidates drift")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != (0, 1):
        raise ValueError("Runtime requires GPU 0 and GPU 1")
    if int(runtime.get("benchmark_jobs", -1)) != EXPECTED_TRAINING_JOBS:
        raise ValueError("Benchmark must exercise all 40 cells")
    if int(runtime.get("benchmark_epochs", -1)) != 1:
        raise ValueError("Benchmark must use one epoch")
    primary_workers = int(runtime.get("benchmark_workers_per_gpu", -1))
    fallback_workers = int(runtime.get("fallback_workers_per_gpu", -1))
    if (
        primary_workers < 1
        or fallback_workers < 1
        or fallback_workers > primary_workers
    ):
        raise ValueError("Invalid benchmark/fallback concurrency")
    formal = runtime.get("formal_workers_per_gpu")
    if formal is not None and int(formal) not in (fallback_workers, primary_workers):
        raise ValueError("Frozen formal concurrency must equal a benchmarked setting")
    _validate_selection_provenance(config)
    with _core_profile():
        core.architecture_profile_contract(config)


def _overlay_mode(arm: str) -> str:
    if str(arm) not in DIRECT_ARMS:
        raise ValueError(f"Unknown matched-FiLM-LR arm: {arm}")
    return "lp_mean_l2"


@contextmanager
def _core_profile() -> Iterator[None]:
    replacements: dict[str, Any] = {
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "EXPERIMENT_KIND": EXPERIMENT_KIND,
        "INTERPRETATION": INTERPRETATION,
        "SEEDS": SEEDS,
        "TOLERANCES": (TOLERANCE_MINUTES,),
        "FOLDS": FOLDS,
        "PARENT_ARM": NO_TEXT_ARM,
        "CONTINUATION_ARM": "__unused_continuation__",
        "TEXT_ARMS_5M": DIRECT_ARMS,
        "TEXT_ARMS_30M": (),
        "ALL_ARMS": DIRECT_ARMS,
        "PARENT_STAGE": DIRECT_STAGE,
        "CONTINUATION_STAGE": "__unused_continuations__",
        "BRANCH_STAGE": "__unused_text_branches__",
        "STAGES": (DIRECT_STAGE,),
        "EXPECTED_STAGE_COUNTS": {DIRECT_STAGE: EXPECTED_TRAINING_JOBS},
        "EXPECTED_TRAINING_JOBS": EXPECTED_TRAINING_JOBS,
        "EXPECTED_PREDICTION_CELLS": EXPECTED_PREDICTION_CELLS,
        "WORKER_MODULE": WORKER_MODULE,
        "SOURCE_CODE_RELATIVE_PATHS": SOURCE_CODE_RELATIVE_PATHS,
        "INFERENCE_DETERMINISM_KIND": INFERENCE_DETERMINISM_KIND,
        "GENERATOR_MODE": GENERATOR_MODE,
        "CRITIC_MODE": CRITIC_MODE,
        "CAPACITY_PROFILE": CAPACITY_PROFILE,
        "EXPECTED_PARAMETER_COUNTS": EXPECTED_PARAMETER_COUNTS,
        "EXPECTED_ARCHITECTURE_PROFILE_SHA256": (EXPECTED_ARCHITECTURE_PROFILE_SHA256),
        "_overlay_mode": _overlay_mode,
    }
    with unet.unet_nolp_profile():
        originals = {name: getattr(core, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(core, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(core, name, value)


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
            "stage": DIRECT_STAGE,
            "tolerance_minutes": TOLERANCE_MINUTES,
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
                film_lr = float(FILM_LEARNING_RATES[arm])
                specs.append(
                    {
                        "stage": DIRECT_STAGE,
                        "tolerance_minutes": TOLERANCE_MINUTES,
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "gpu_id": int(assignment[fold]),
                        "job_id": _job_id(seed, fold, arm),
                        "pair_text_overlay_mode": "lp_mean_l2",
                        "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
                        "generator_text_learning_rate": TEXT_LEARNING_RATE,
                        "generator_film_learning_rate": film_lr,
                        "generator_text_min_learning_rate": TEXT_MIN_LEARNING_RATE,
                        "generator_film_min_learning_rate": (
                            film_lr * GROUP_FLOOR_RATIO
                        ),
                        **initial_by_seed[seed],
                    }
                )
    if (
        len(specs) != EXPECTED_TRAINING_JOBS
        or len({str(row["job_id"]) for row in specs}) != EXPECTED_TRAINING_JOBS
    ):
        raise AssertionError("Five-seed FiLM-LR sweep requires 40 unique jobs")
    validate_gpu_balance(specs)
    return specs


def validate_gpu_balance(specs: Sequence[Mapping[str, Any]]) -> None:
    blocks: dict[tuple[int, str], int] = {}
    for row in specs:
        key = (int(row["seed"]), str(row["fold"]))
        gpu = int(row["gpu_id"])
        if blocks.setdefault(key, gpu) != gpu:
            raise ValueError(f"LR arms in block {key} do not share one physical GPU")
    counts = {gpu: sum(int(row["gpu_id"]) == gpu for row in specs) for gpu in (0, 1)}
    if counts != {0: 20, 1: 20}:
        raise ValueError(f"GPU job balance must be 20/20, got {counts}")
    expected_blocks = {
        (seed, fold): (0 if fold in {"f1_2023q1", "f3_2023q3"} else 1)
        for seed in SEEDS
        for fold in FOLDS
    }
    if blocks != expected_blocks:
        raise ValueError(f"Frozen seed/fold GPU assignment drift: {blocks}")


def _run_directory(root: Path, spec: Mapping[str, Any]) -> Path:
    with _core_profile():
        return core._run_directory(root, spec)


def _prediction_path(root: Path, job: Mapping[str, Any]) -> Path:
    return core._prediction_path(root, job)


def _prediction_manifest_path(root: Path, job: Mapping[str, Any]) -> Path:
    return core._prediction_job_manifest_path(root, job)


def _source_paths(config: Mapping[str, Any]) -> list[tuple[str, Path]]:
    rows = list(_BASE_SOURCE_PATHS(config))
    provenance = _selection_provenance(config)
    rows.extend(
        [
            ("selection_report", direct.resolve_path(provenance["report_path"])),
            (
                "selection_output_hash_manifest",
                direct.resolve_path(provenance["output_hash_manifest_path"]),
            ),
        ]
    )
    if len({role for role, _path in rows}) != len(rows):
        raise ValueError("Source-manifest roles must be unique")
    return rows


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
        raise ValueError(f"Unknown matched-FiLM-LR arm: {arm}")
    training = _mapping(config["training"], "training")
    model = _mapping(config["model"], "model")
    return {
        "generator_conditioning_mode": model["generator_conditioning_mode"],
        "critic_conditioning_mode": model["critic_conditioning_mode"],
        "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
        "generator_learning_rate": BACKBONE_LEARNING_RATE,
        "generator_text_learning_rate": float(spec["generator_text_learning_rate"]),
        "generator_film_learning_rate": float(spec["generator_film_learning_rate"]),
        "generator_text_min_learning_rate": float(
            spec["generator_text_min_learning_rate"]
        ),
        "generator_film_min_learning_rate": float(
            spec["generator_film_min_learning_rate"]
        ),
        "discriminator_learning_rate": CRITIC_LEARNING_RATE,
        "reduce_lr_min_lr": BACKBONE_MIN_LEARNING_RATE,
        "num_epochs": int(num_epochs or training["num_epochs"]),
        "early_stopping_min_epochs": int(training["early_stopping_min_epochs"]),
        "early_stopping_patience": int(training["early_stopping_patience"]),
        "validation_mc_samples": int(training["validation_mc_samples"]),
        "news_first_materialize_validation_loader": True,
        "news_first_materialize_test_loader": False,
        "news_first_pair_text_overlay_mode": "lp_mean_l2",
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
        source = output_dir / "tolerance_05m" / fold / "lp_matched.json"
        payload = direct.read_json(source)
        for arm in DIRECT_ARMS:
            film_lr = float(FILM_LEARNING_RATES[arm])
            write_pair_text_overlay_manifest(
                direct._overlay_path(root, fold, arm),
                mode="lp_mean_l2",
                namespace=f"tol05/{fold}/{arm}/direct_matched_film_lr_5seed_v1",
                records=list(payload["records"]),
                transform={
                    **dict(payload.get("transform") or {}),
                    "direct_arm": arm,
                    "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
                    "generator_film_learning_rate": film_lr,
                    "training_seeds": list(SEEDS),
                    "parent_or_continuation_state_input": False,
                },
            )
    expected = {
        direct._overlay_path(root, fold, arm) for fold in FOLDS for arm in DIRECT_ARMS
    }
    for path in output_dir.glob("tolerance_*/*/*.json"):
        if path.resolve() not in expected:
            path.unlink()
    paths = sorted(output_dir.glob("tolerance_05m/*/*.json"))
    if len(paths) != 8 or {path.resolve() for path in paths} != expected:
        raise ValueError("Matched-FiLM-LR development overlay universe drift")
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        payloads = [
            direct.read_json(direct._overlay_path(root, fold, arm))
            for arm in DIRECT_ARMS
        ]
        if payloads[0]["records"] != payloads[1]["records"]:
            raise ValueError(f"Matched LP records differ across LR arms: {fold}")
        for arm, payload in zip(DIRECT_ARMS, payloads, strict=True):
            unsigned = {
                key: value for key, value in payload.items() if key != "profile_sha256"
            }
            if payload.get("mode") != "lp_mean_l2" or direct.payload_sha256(
                unsigned
            ) != payload.get("profile_sha256"):
                raise ValueError(f"Matched LP overlay self-hash drift: {fold}/{arm}")
            path = direct._overlay_path(root, fold, arm)
            rows.append(
                direct.manifest_row(f"pair_overlay:{path.relative_to(root)}", path)
            )
    return direct._write_hash_manifest(
        root / "inputs/pair_text_overlay_hashes.csv", rows
    )


def validate_root(
    root_or_path: str | Path, *, verify_large_inputs: bool = True
) -> dict[str, Any]:
    root = Path(root_or_path).resolve()
    registry = direct.read_registry(root)
    if (
        registry.get("kind") != REGISTRY_KIND
        or registry.get("experiment_kind") != EXPERIMENT_KIND
        or registry.get("interpretation") != INTERPRETATION
    ):
        raise ValueError("Five-seed FiLM-LR registry identity drift")
    jobs = list(registry.get("jobs") or [])
    if len(jobs) != EXPECTED_TRAINING_JOBS or registry.get(
        "jobs_sha256"
    ) != direct.payload_sha256(jobs):
        raise ValueError("Five-seed 40-job registry drift")
    if len({str(job["job_id"]) for job in jobs}) != EXPECTED_TRAINING_JOBS:
        raise ValueError("Duplicate five-seed job IDs")
    expected_cells = {
        (seed, fold, arm) for seed in SEEDS for fold in FOLDS for arm in DIRECT_ARMS
    }
    observed_cells = {
        (int(job["seed"]), str(job["fold"]), str(job["arm"])) for job in jobs
    }
    if observed_cells != expected_cells:
        raise ValueError("Registry seed/fold/arm universe drift")
    config = _load_frozen_config(root / "resolved_config.yaml")
    if direct.read_json(root / "grid_contract.json") != grid_contract(config):
        raise ValueError("Grid contract drift")
    if direct.read_json(root / "model_contract.json") != model_contract(config):
        raise ValueError("Model contract drift")
    source_rows = [
        direct.manifest_row(role, path) for role, path in _source_paths(config)
    ]
    if verify_large_inputs:
        direct._assert_hash_manifest(root / "source_hashes.csv", source_rows)
    else:
        import os

        attestation = os.environ.get("DIRECT_SOURCE_MANIFEST_SHA256", "")
        if not attestation:
            direct._assert_hash_manifest(root / "source_hashes.csv", source_rows)
        elif direct.sha256_file(root / "source_hashes.csv") != attestation:
            raise ValueError("Supervisor source-manifest attestation drift")
    code_rows = [
        direct.manifest_row(f"code:{path.relative_to(REPO_ROOT)}", path)
        for path in direct._code_paths()
    ]
    direct._assert_hash_manifest(root / "code_hashes.csv", code_rows)
    config_rows = [
        direct.manifest_row("resolved_config", root / "resolved_config.yaml"),
        direct.manifest_row("grid_contract", root / "grid_contract.json"),
        direct.manifest_row("model_contract", root / "model_contract.json"),
        direct.manifest_row(
            "pair_universe_manifest", root / "inputs/pair_universe_manifest.json"
        ),
        direct.manifest_row(
            "pair_text_overlay_hashes", root / "inputs/pair_text_overlay_hashes.csv"
        ),
    ]
    direct._assert_hash_manifest(root / "config_hashes.csv", config_rows)
    grouped_g: dict[int, set[str]] = {seed: set() for seed in SEEDS}
    grouped_d: dict[int, set[str]] = {seed: set() for seed in SEEDS}
    for job in jobs:
        seed = int(job["seed"])
        grouped_g[seed].add(str(job["initial_generator_state_sha256"]))
        grouped_d[seed].add(str(job["initial_critic_state_sha256"]))
    if any(len(values) != 1 for values in (*grouped_g.values(), *grouped_d.values())):
        raise ValueError("Planned epoch-0 states are not common within each seed")
    if len({next(iter(values)) for values in grouped_g.values()}) != len(SEEDS) or len(
        {next(iter(values)) for values in grouped_d.values()}
    ) != len(SEEDS):
        raise ValueError("Planned epoch-0 states are not distinct across seeds")
    for job in jobs:
        if core._job_spec_sha(job) != job.get("job_spec_sha256"):
            raise ValueError(f"Job spec hash drift: {job['job_id']}")
        if job.get("parent_state_path") or job.get("recipe_path"):
            raise ValueError(f"Direct job consumes forbidden state: {job['job_id']}")
        for path_key, sha_key in (
            ("training_config_path", "training_config_sha256"),
            ("full_state_contract_path", "full_state_contract_sha256"),
            ("overlay_path", "overlay_sha256"),
        ):
            direct._verify_frozen_file(job[path_key], job[sha_key])
        if verify_large_inputs:
            direct._verify_frozen_file(job["dataset_path"], job["dataset_sha256"])
            direct._verify_frozen_file(job["support_path"], job["support_sha256"])
        status_payload = direct.read_json(direct._status_path(root, str(job["job_id"])))
        if (
            status_payload.get("job_id") != job["job_id"]
            or status_payload.get("job_spec_sha256") != job["job_spec_sha256"]
        ):
            raise ValueError(f"Job status lineage drift: {job['job_id']}")
        if status_payload.get("status") == "completed" and not direct._completed_valid(
            job, status_payload
        ):
            raise ValueError(f"Completed artifact drift: {job['job_id']}")
    for flag, path_key, sha_key in (
        (
            "evaluation_frozen",
            "checkpoint_allowlist_path",
            "checkpoint_allowlist_sha256",
        ),
        ("test_data_opened", "test_input_manifest_path", "test_input_manifest_sha256"),
        (
            "predictions_frozen",
            "prediction_manifest_path",
            "prediction_manifest_sha256",
        ),
        ("predictions_frozen", "pair_metrics_path", "pair_metrics_sha256"),
        ("analysis_complete", "analysis_manifest_path", "analysis_manifest_sha256"),
    ):
        if registry.get(flag):
            direct._verify_frozen_file(registry[path_key], registry[sha_key])
    if registry.get("test_data_opened") and not registry.get("evaluation_frozen"):
        raise ValueError("Test data opened before checkpoint freeze")
    if registry.get("predictions_frozen") and not registry.get("test_data_opened"):
        raise ValueError("Predictions frozen before test-input freeze")
    return config


def _freeze_initial_state_fairness(
    root: Path, jobs: Sequence[Mapping[str, Any]]
) -> Path:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        status_payload = direct.read_json(direct._status_path(root, str(job["job_id"])))
        initial_g = direct._artifact(status_payload, "generator_initial_epoch0")
        initial_d = direct._artifact(status_payload, "discriminator_initial_epoch0")
        best_g = direct._artifact(status_payload, "generator_best_learned")
        best_d = direct._artifact(status_payload, "discriminator_best_learned")
        observed_g = direct._checkpoint_state_sha256(initial_g["path"])
        observed_d = direct._checkpoint_state_sha256(initial_d["path"])
        if observed_g != job["initial_generator_state_sha256"]:
            raise ValueError(f"Frozen Generator initial state drift: {job['job_id']}")
        if observed_d != job["initial_critic_state_sha256"]:
            raise ValueError(f"Frozen Critic initial state drift: {job['job_id']}")
        best_g_sha = direct._checkpoint_state_sha256(best_g["path"])
        best_d_sha = direct._checkpoint_state_sha256(best_d["path"])
        if observed_g == best_g_sha:
            raise ValueError(f"Generator did not update: {job['job_id']}")
        if observed_d == best_d_sha:
            raise ValueError(f"Critic did not update: {job['job_id']}")
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
        if len(selected) != 8 or len(g_hashes) != 1 or len(d_hashes) != 1:
            raise ValueError(f"Actual epoch-0 fairness drift within seed {seed}")
        by_seed[str(seed)] = {
            "job_count": 8,
            "generator_initial_state_sha256": next(iter(g_hashes)),
            "critic_initial_state_sha256": next(iter(d_hashes)),
        }
    if len({row["generator_initial_state_sha256"] for row in by_seed.values()}) != 5:
        raise ValueError("Generator epoch-0 states are not distinct across seeds")
    if len({row["critic_initial_state_sha256"] for row in by_seed.values()}) != 5:
        raise ValueError("Critic epoch-0 states are not distinct across seeds")
    payload = {
        "schema_version": 1,
        "kind": "direct_matched_film_lr_5seed_initial_state_fairness_v1",
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


def grid_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    with _core_profile():
        return core.grid_contract(config)


def model_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    with _core_profile():
        return core.model_contract(config)


def _materialize_test_inputs(config: Mapping[str, Any], root: Path) -> Path:
    registry = direct.read_registry(root)
    if not registry.get("evaluation_frozen"):
        raise RuntimeError("Test inputs require a frozen checkpoint allowlist")
    existing = direct._test_input_manifest_path(root)
    if existing.is_file():
        if registry.get("test_data_opened"):
            return direct._validate_test_inputs(root)
        raise ValueError("Partial test-input freeze requires manual audit")

    from wgan_option.utils.news_first_experiment_core import (
        _l2,
        _normalized_article_key,
        _parsed_vector,
        write_pair_text_overlay_manifest,
    )

    data = _mapping(config["data"], "data")
    universe = pd.read_csv(root / "inputs/pair_universes.csv", dtype=str)
    test_universe = universe.loc[
        universe["tolerance_minutes"].astype(int).eq(5)
        & universe["partition"].eq("test")
    ].copy()
    required_pairs = set(test_universe["pair_id"].astype(str))
    workbook = direct.resolve_path(data["root"]) / str(
        data["workbook_template"]
    ).format(tolerance02="05")
    frame = pd.read_excel(workbook, sheet_name=str(data["sheet_name"]))
    frame["pair_id"] = frame["pair_id"].astype(str)
    frame = frame.loc[frame["pair_id"].isin(required_pairs)].copy()
    if set(frame["pair_id"]) != required_pairs:
        raise ValueError("Test workbook coverage drift")
    canonical_rows: dict[str, dict[str, Any]] = {}
    lp_by_pair: dict[str, np.ndarray] = {}
    for pair_id, pair_rows in frame.groupby("pair_id", sort=False):
        ordered = pair_rows.assign(
            _news_row_sort=pd.to_numeric(pair_rows["news_row_id"], errors="raise")
        ).sort_values(["_news_row_sort", "sample_id"], kind="stable")
        canonical_rows[str(pair_id)] = (
            ordered.iloc[0].drop(labels="_news_row_sort").to_dict()
        )
        articles: dict[str, np.ndarray] = {}
        for row in ordered.itertuples(index=False):
            key = _normalized_article_key(row)
            if key not in articles:
                articles[key] = _parsed_vector(
                    row.lp_embedding,
                    dimension=1024,
                    label=f"test LP {pair_id}/{key}",
                )
        if not articles:
            raise ValueError(f"No usable test articles: {pair_id}")
        lp_by_pair[str(pair_id)] = _l2(
            np.mean(np.stack([articles[key] for key in sorted(articles)]), axis=0)
        )

    artifact_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        selected = test_universe.loc[test_universe["fold"].eq(fold)].copy()
        pair_ids = sorted(selected["pair_id"].astype(str))
        sessions = dict(
            zip(selected["pair_id"].astype(str), selected["session_id"].astype(str))
        )
        expected = direct._expected_counts(config, fold)
        if (
            len(pair_ids) != expected["test_pairs"]
            or len(set(sessions.values())) != expected["test_sessions"]
        ):
            raise ValueError(f"Frozen test counts drift: {fold}")
        panel = pd.DataFrame([canonical_rows[pair_id] for pair_id in pair_ids])
        panel["sample_id"] = panel["pair_id"].map(lambda value: f"pair::{value}")
        panel["sample_weight"] = 1.0
        panel_path = core._write_dataframe_csv(
            direct._test_panel_path(root, fold), panel, gzip=True
        )
        artifact_rows.append(direct.manifest_row(f"test_panel:05m:{fold}", panel_path))
        records = [
            {
                "pair_id": pair_id,
                "session_id": sessions[pair_id],
                "embedding": lp_by_pair[pair_id],
            }
            for pair_id in pair_ids
        ]
        for arm in DIRECT_ARMS:
            training_path = direct._overlay_path(root, fold, arm)
            training_payload = direct.read_json(training_path)
            overlay = direct._test_overlay_path(root, fold, arm)
            write_pair_text_overlay_manifest(
                overlay,
                mode="lp_mean_l2",
                namespace=f"evaluation/test/05m/{fold}/{arm}",
                records=records,
                transform={
                    **dict(training_payload.get("transform") or {}),
                    "evaluation_partition": "test",
                    "method": "unique_article_lp_mean_l2_v1",
                    "training_overlay_path": str(training_path),
                    "training_overlay_sha256": direct.sha256_file(training_path),
                },
            )
            artifact_rows.append(
                direct.manifest_row(f"test_overlay:05m:{fold}:{arm}", overlay)
            )
    if len(artifact_rows) != 12:
        raise ValueError("Expected four panels and eight frozen test overlays")
    result = direct._write_hash_manifest(existing, artifact_rows)
    registry = direct.read_registry(root)
    registry.update(
        status="evaluation_inputs_frozen",
        test_data_opened=True,
        test_data_opened_at_utc=direct.utc_now(),
        test_input_manifest_path=str(result.resolve()),
        test_input_manifest_sha256=direct.sha256_file(result),
    )
    direct.write_registry(root, registry)
    direct._experiment_status(
        root,
        "evaluation_inputs_frozen",
        test_panels=len(FOLDS),
        test_overlays=len(FOLDS) * len(DIRECT_ARMS),
    )
    return direct._validate_test_inputs(root)


def _validate_prediction_cell(
    root: Path,
    job: Mapping[str, Any],
    checkpoint: Mapping[str, str],
    *,
    expected_pairs: int,
) -> dict[str, Any]:
    with _core_profile():
        return core._validate_prediction_job(
            root, job, checkpoint, expected_pairs=expected_pairs
        )


def _validate_predictions(root: Path) -> tuple[Path, Path]:
    config = validate_root(root)
    registry = direct.read_registry(root)
    if not registry.get("predictions_frozen"):
        raise ValueError("Predictions are not frozen")
    manifest_path = direct._verify_frozen_file(
        registry["prediction_manifest_path"], registry["prediction_manifest_sha256"]
    )
    pair_path = direct._verify_frozen_file(
        registry["pair_metrics_path"], registry["pair_metrics_sha256"]
    )
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    evidence = pd.read_csv(pair_path)
    if (
        len(manifest) != EXPECTED_PREDICTION_CELLS
        or manifest["job_id"].duplicated().any()
    ):
        raise ValueError("Prediction manifest must contain exactly 40 unique cells")
    if len(evidence) != EXPECTED_PAIR_METRIC_ROWS:
        raise ValueError("Global pair metrics must contain exactly 5,000 rows")
    checkpoints = direct._checkpoint_map(root)
    jobs = [dict(job) for job in registry["jobs"]]
    profiles: dict[tuple[int, str], set[str]] = {
        (seed, fold): set() for seed in SEEDS for fold in FOLDS
    }
    for job in jobs:
        expected_pairs = direct._expected_counts(config, str(job["fold"]))["test_pairs"]
        payload = _validate_prediction_cell(
            root,
            job,
            checkpoints[str(job["job_id"])],
            expected_pairs=expected_pairs,
        )
        profiles[(int(job["seed"]), str(job["fold"]))].add(
            str(payload["noise_bank_profile_sha256"])
        )
        cell = evidence.loc[evidence["job_id"].astype(str).eq(str(job["job_id"]))]
        if len(cell) != expected_pairs:
            raise ValueError(f"Global pair-evidence count drift: {job['job_id']}")
    if any(len(values) != 1 for values in profiles.values()):
        raise ValueError("LR arms within a seed/fold do not share one MC64 noise bank")
    expected_cells = {
        (seed, fold, arm) for seed in SEEDS for fold in FOLDS for arm in DIRECT_ARMS
    }
    observed_cells = set(
        evidence[["seed", "fold", "arm"]]
        .assign(
            seed=lambda frame: pd.to_numeric(frame["seed"], errors="raise").astype(int)
        )
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise ValueError("Pair-evidence seed/fold/arm universe drift")
    return manifest_path, pair_path


@contextmanager
def multiseed_profile() -> Iterator[None]:
    """Install the 40-cell profile and restore every shared global on exit."""

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
        "direct_profile": _core_profile,
        "load_config": load_config,
        "_load_frozen_config": _load_frozen_config,
        "validate_config": validate_config,
        "grid_contract": grid_contract,
        "model_contract": model_contract,
        "planned_specs": planned_specs,
        "validate_gpu_balance": validate_gpu_balance,
        "_source_paths": _source_paths,
        "_overlay_mode": _overlay_mode,
        "_run_directory": _run_directory,
        "_materialize_pair_text_overlays": _materialize_pair_text_overlays,
        "_training_payload": _training_payload,
        "validate_root": validate_root,
        "_freeze_initial_state_fairness": _freeze_initial_state_fairness,
        "_materialize_test_inputs": _materialize_test_inputs,
        "_prediction_path": _prediction_path,
        "_prediction_manifest_path": _prediction_manifest_path,
        "_validate_prediction_cell": _validate_prediction_cell,
        "_validate_predictions": _validate_predictions,
    }
    with _core_profile():
        originals = {name: getattr(direct, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(direct, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(direct, name, value)


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    with multiseed_profile():
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
    with multiseed_profile():
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
    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise RuntimeError("Prediction requires a frozen checkpoint allowlist root")
    try:
        registry = direct.read_registry(root)
    except FileNotFoundError as exc:
        raise RuntimeError("Prediction requires a frozen checkpoint allowlist") from exc
    if not registry.get("evaluation_frozen"):
        raise RuntimeError("Prediction requires a frozen checkpoint allowlist")
    freeze_evaluation(root)
    with multiseed_profile():
        config = validate_root(root)
        registry = direct.read_registry(root)
        if registry.get("predictions_frozen"):
            _validate_predictions(root)
            return root
        checkpoints = direct._checkpoint_map(root)
        direct._validate_test_inputs(root)
        prediction_rows: list[dict[str, Any]] = []
        evidence_frames: list[pd.DataFrame] = []
        jobs = sorted(
            (dict(job) for job in registry["jobs"]),
            key=lambda job: (int(job["seed"]), str(job["fold"]), str(job["arm"])),
        )
        for job in jobs:
            checkpoint = checkpoints[str(job["job_id"])]
            expected_pairs = direct._expected_counts(config, str(job["fold"]))[
                "test_pairs"
            ]
            path = _prediction_path(root, job)
            manifest_path = _prediction_manifest_path(root, job)
            evidence_path = manifest_path.with_suffix(".pair_metrics.csv")
            if path.is_file() and manifest_path.is_file() and evidence_path.is_file():
                payload = _validate_prediction_cell(
                    root, job, checkpoint, expected_pairs=expected_pairs
                )
                evidence = pd.read_csv(evidence_path)
            else:
                if (
                    any(
                        value.is_file()
                        for value in (path, manifest_path, evidence_path)
                    )
                    and not resume
                ):
                    raise ValueError(
                        f"Partial prediction requires --resume: {job['job_id']}"
                    )
                with _core_profile():
                    _payload, evidence = core._evaluate_prediction_job(
                        root, job, checkpoint, expected_pairs=expected_pairs
                    )
                core._write_dataframe_csv(evidence_path, evidence)
                payload = direct.read_json(manifest_path)
                payload.pop("payload_sha256", None)
                payload.update(
                    pair_metrics_path=str(evidence_path.resolve()),
                    pair_metrics_sha256=direct.sha256_file(evidence_path),
                    pair_metrics_row_count=int(len(evidence)),
                )
                payload["payload_sha256"] = direct.payload_sha256(payload)
                direct.write_json(manifest_path, payload)
            payload = _validate_prediction_cell(
                root, job, checkpoint, expected_pairs=expected_pairs
            )
            prediction_rows.append(
                {
                    "job_id": str(job["job_id"]),
                    "arm": str(job["arm"]),
                    "fold": str(job["fold"]),
                    "seed": int(job["seed"]),
                    "tolerance_minutes": 5,
                    "prediction_path": str(path.resolve()),
                    "prediction_sha256": direct.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "prediction_manifest_path": str(manifest_path.resolve()),
                    "prediction_manifest_sha256": direct.sha256_file(manifest_path),
                    "noise_bank_profile_sha256": payload["noise_bank_profile_sha256"],
                    "inference_determinism_contract_sha256": payload[
                        "inference_determinism_contract_sha256"
                    ],
                }
            )
            evidence_frames.append(evidence)
        if (
            len(prediction_rows) != EXPECTED_PREDICTION_CELLS
            or len(evidence_frames) != EXPECTED_PREDICTION_CELLS
        ):
            raise ValueError("Prediction stage did not produce 40 cells")
        prediction_manifest = direct.write_csv(
            root / "analysis/prediction_manifest.csv",
            prediction_rows,
            tuple(prediction_rows[0]),
        )
        evidence = pd.concat(evidence_frames, ignore_index=True)
        pair_metrics = core._write_dataframe_csv(
            root / "analysis/rq12_pair_metrics.csv.gz", evidence, gzip=True
        )
        if len(evidence) != EXPECTED_PAIR_METRIC_ROWS:
            raise ValueError(f"Expected 5,000 pair rows, got {len(evidence)}")
        registry = direct.read_registry(root)
        registry.update(
            status="predictions_frozen",
            predictions_frozen=True,
            predictions_frozen_at_utc=direct.utc_now(),
            prediction_manifest_path=str(prediction_manifest.resolve()),
            prediction_manifest_sha256=direct.sha256_file(prediction_manifest),
            pair_metrics_path=str(pair_metrics.resolve()),
            pair_metrics_sha256=direct.sha256_file(pair_metrics),
            prediction_cell_count=EXPECTED_PREDICTION_CELLS,
            pair_metric_row_count=EXPECTED_PAIR_METRIC_ROWS,
        )
        direct.write_registry(root, registry)
        direct._experiment_status(
            root,
            "predictions_frozen",
            prediction_cells=EXPECTED_PREDICTION_CELLS,
            pair_rows=EXPECTED_PAIR_METRIC_ROWS,
        )
        _validate_predictions(root)
        return root


def status(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    return _call("status", output_dir)


def _training_summary(root: Path) -> Path:
    destination = root / "analysis/training_summary.csv"
    if destination.is_file():
        return destination
    with multiseed_profile():
        checkpoints = direct._checkpoint_map(root)
        registry = direct.read_registry(root)
        rows: list[dict[str, Any]] = []
        for job in registry["jobs"]:
            status_payload = direct.read_json(
                direct._status_path(root, str(job["job_id"]))
            )
            best = direct.read_json(
                Path(
                    direct._artifact(status_payload, "best_learned_checkpoint")["path"]
                )
            )
            metrics = pd.read_csv(
                direct._artifact(status_payload, "training_metrics_csv")["path"]
            )
            epochs = pd.to_numeric(metrics["epoch"], errors="raise").astype(int)
            learned = epochs[epochs >= 1]
            if learned.empty:
                raise ValueError(
                    f"Training metrics have no learned epoch: {job['job_id']}"
                )
            epochs_ran = int(learned.max())
            contract = single_lr._optimizer_contract(
                job=job,
                metrics=metrics.loc[epochs <= epochs_ran].copy(),
                epochs_ran=epochs_ran,
            )
            rows.append(
                {
                    "job_id": str(job["job_id"]),
                    "seed": int(job["seed"]),
                    "fold": str(job["fold"]),
                    "arm": str(job["arm"]),
                    "best_epoch": int(best["best_epoch"]),
                    "epochs_ran": epochs_ran,
                    "final_generator_lr": float(metrics.iloc[-1]["g_lr_backbone"]),
                    "final_discriminator_lr": float(metrics.iloc[-1]["d_lr"]),
                    "best_validation_score": float(best["best_metric"]),
                    "checkpoint_sha256": checkpoints[str(job["job_id"])]["sha256"],
                    "monitor_metric": str(
                        best.get("monitor_metric", "val_hybrid_score")
                    ),
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
        raise ValueError("Training summary requires exactly 40 FiLM-LR cells")
    return direct.write_csv(destination, rows, tuple(rows[0]))


def _verify_analysis_manifest(path: Path) -> dict[str, Any]:
    payload = direct.read_json(path)
    if payload.get("kind") != analysis.ANALYSIS_MANIFEST_KIND:
        raise ValueError("Five-seed FiLM-LR analysis manifest kind drift")
    rows = [
        (str(section), dict(row))
        for section in ("inputs", "artifacts")
        for row in payload.get(section) or []
    ]
    if not rows or len({(section, row.get("role")) for section, row in rows}) != len(
        rows
    ):
        raise ValueError("Analysis manifest roles are empty or duplicated")
    for section, row in rows:
        target = Path(str(row.get("path", ""))).resolve()
        if (
            not target.is_file()
            or target.stat().st_size != int(row.get("size_bytes", -1))
            or direct.sha256_file(target) != str(row.get("sha256", ""))
        ):
            raise ValueError(f"Analysis {section} artifact drift: {target}")
    return payload


def postprocess(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    with multiseed_profile():
        config = validate_root(root)
        registry = direct.read_registry(root)
        if registry.get("terminal_complete"):
            return qa(root)
        _validate_predictions(root)
        training_summary = _training_summary(root)
        analysis_manifest = analysis.analyze_experiment(root, config)
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
    with multiseed_profile():
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
        _validate_selection_provenance(config)
        training = pd.read_csv(root / "analysis/training_summary.csv")
        pairs = pd.read_csv(root / "analysis/rq12_pair_metrics.csv.gz")
        if (
            len(training) != EXPECTED_TRAINING_JOBS
            or len(pairs) != EXPECTED_PAIR_METRIC_ROWS
        ):
            raise ValueError("Five-seed terminal evidence count drift")
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
            raise ValueError("Five-seed initial-state fairness evidence drift")
        qa_payload = {
            "schema_version": 1,
            "kind": QA_KIND,
            "status": "passed",
            "interpretation": INTERPRETATION,
            "training_jobs_completed": EXPECTED_TRAINING_JOBS,
            "prediction_cells": EXPECTED_PREDICTION_CELLS,
            "pair_metric_rows": EXPECTED_PAIR_METRIC_ROWS,
            "seeds": list(SEEDS),
            "discovery_seed_excluded": True,
            "parent_jobs": 0,
            "continuation_jobs": 0,
            "matched_lp_arms": len(DIRECT_ARMS),
            "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
            "test_opened_after_checkpoint_freeze": True,
            "shared_mc64_noise_bank_within_seed_fold": True,
            "test_based_lr_selection_permitted": False,
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
    root = Path(output_dir).resolve()
    if root.is_dir():
        try:
            registry = _call("read_registry", root)
        except (FileNotFoundError, ValueError):
            registry = {}
        if registry.get("terminal_complete"):
            with multiseed_profile():
                validate_root(root)
                direct._validate_output_hashes(root)
            return root
    with multiseed_profile():
        lock_path, descriptor = direct._pipeline_lock(root)
    try:
        with multiseed_profile():
            direct._pipeline_journal(root, "benchmark", "running")
        benchmark(config_path, root, resume=resume)
        with multiseed_profile():
            direct._pipeline_journal(root, "prepare", "running")
        prepare_experiment(config_path, root, resume=True)
        if _call("read_registry", root).get("status") == "prepared":
            with multiseed_profile():
                direct._pipeline_journal(
                    root,
                    "direct_arms",
                    "running",
                    completed=status(root)["job_status_counts"].get("completed", 0),
                )
            launch(root, resume=True)
        if not _call("read_registry", root).get("evaluation_frozen"):
            with multiseed_profile():
                direct._pipeline_journal(root, "freeze_evaluation", "running")
            freeze_evaluation(root, resume=True)
        if not _call("read_registry", root).get("predictions_frozen"):
            with multiseed_profile():
                direct._pipeline_journal(root, "predict", "running")
            predict(root, resume=True)
        with multiseed_profile():
            direct._pipeline_journal(root, "postprocess", "running")
        postprocess(root, resume=True)
        with multiseed_profile():
            direct._pipeline_journal(root, "terminal", "completed")
        return root
    except BaseException as exc:
        with multiseed_profile():
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
    "FILM_LEARNING_RATES",
    "SEEDS",
    "benchmark",
    "freeze_evaluation",
    "load_config",
    "model_contract",
    "multiseed_profile",
    "planned_specs",
    "predict",
    "prepare",
    "run_pipeline",
    "validate_config",
    "validate_gpu_balance",
]
