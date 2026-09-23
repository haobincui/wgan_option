"""Direct matched-LP FiLM learning-rate sweep for seed 42.

This branch-local profile reuses the audited direct-training lifecycle while
varying only the FiLM-projection learning rate.  Every one of the five arms
uses the same matched pair-level LP overlay, the same seeded model weights,
and the same backbone/text/Critic learning rates.  There is no parent,
continuation, or branch-replay state in this experiment.
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

from scripts.rq3 import (
    news_first_vol_film_unet_direct_5arm_seed42 as direct,
)
from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_lr_seed42_analysis as lr_analysis,
)


DEFAULT_CONFIG = "configs/rq3/news_first_vol_film_unet_direct_matched_lr_seed42.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_lr_"
    "seed42_exact_ttm_rolling_v1"
)
EXPERIMENT_KIND = (
    "rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_lr_seed42_rolling_v1"
)
INTERPRETATION = "retrospective_rolling_development_single_seed_descriptive"
DIRECT_ARMS = (
    "film_lr_2p5e7",
    "film_lr_5e7",
    "film_lr_1e6",
    "film_lr_2p5e6",
    "film_lr_5e6",
)
FILM_LEARNING_RATES: Mapping[str, float] = {
    "film_lr_2p5e7": 2.5e-7,
    "film_lr_5e7": 5.0e-7,
    "film_lr_1e6": 1.0e-6,
    "film_lr_2p5e6": 2.5e-6,
    "film_lr_5e6": 5.0e-6,
}
# The shared orchestrator needs a parent-arm identity for generic stage
# routing.  It is deliberately not a member of DIRECT_ARMS.
NO_TEXT_ARM = "__unused_no_text_arm__"
EXPECTED_TRAINING_JOBS = 20
EXPECTED_PREDICTION_CELLS = 20
EXPECTED_PAIR_METRIC_ROWS = 2_500
WORKER_MODULE = "scripts.rq3.news_first_vol_film_unet_direct_matched_lr_seed42"
INFERENCE_DETERMINISM_KIND = "direct_matched_film_lr_inference_determinism_v1"
BENCHMARK_RESULT_KIND = "direct_matched_film_lr_full_matrix_epoch1_benchmark_v1"
REGISTRY_KIND = "direct_matched_film_lr_task_registry_v1"
QA_KIND = "direct_matched_film_lr_terminal_qa_v1"
GENERATOR_OPTIMIZER_PROFILE = "film_unet_split_lr_v1"
BACKBONE_LEARNING_RATE = 5.0e-7
TEXT_LEARNING_RATE = 2.5e-6
CRITIC_LEARNING_RATE = 5.0e-7
BACKBONE_MIN_LEARNING_RATE = 5.0e-8
TEXT_MIN_LEARNING_RATE = 2.5e-7
GROUP_FLOOR_RATIO = 0.1
GENERATOR_GROUP_PARAMETER_COUNTS: Mapping[str, int] = {
    "backbone": 416_353,
    "text_encoder": 295_808,
    "film_projection": 115_584,
}
SOURCE_CODE_RELATIVE_PATHS = (
    "scripts/rq3/news_first_vol_film_unet_direct_matched_lr_seed42.py",
    "scripts/rq3/news_first_vol_film_unet_direct_matched_lr_seed42_analysis.py",
    "scripts/rq3/news_first_vol_film_unet_direct_5arm_seed42.py",
    "scripts/rq3/news_first_vol_film_unet_direct_5arm_seed42_analysis.py",
    "scripts/rq123/news_first_vol_film_nolp_10seed.py",
    "scripts/rq123/news_first_vol_film_unet_nolp_10seed.py",
    "scripts/rq3/news_first_vol_comparison_analysis.py",
    "scripts/rq3/news_first_vol_training.py",
    "scripts/rq2_pair/pair_features.py",
    "scripts/rq2_pair/rq2_pair_experiment.py",
)


_BASE_VALIDATE_CONFIG = direct.validate_config
_BASE_SOURCE_PATHS = direct._source_paths


def _pure_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    analysis = direct._mapping(config.get("analysis"), "analysis")
    return direct._mapping(analysis.get("frozen_pure_cnn"), "analysis.frozen_pure_cnn")


def _validate_pure_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the immutable Pure-CNN comparator and its checkpoint lineage."""

    reference = _pure_reference(config)
    if reference.get("arm") != "pure_cnn_no_text" or bool(
        reference.get("retrain", True)
    ):
        raise ValueError("Pure-CNN reference must be frozen and must not be retrained")
    root = direct.resolve_path(reference["experiment_root"])
    declared = {
        "output_hash_manifest": (
            direct.resolve_path(reference["output_hash_manifest_path"]),
            str(reference["output_hash_manifest_sha256"]),
        ),
        "pair_metrics": (
            direct.resolve_path(reference["pair_metrics_path"]),
            str(reference["pair_metrics_sha256"]),
        ),
        "checkpoint_allowlist": (
            direct.resolve_path(reference["checkpoint_allowlist_path"]),
            str(reference["checkpoint_allowlist_sha256"]),
        ),
    }
    for role, (path, expected_sha) in declared.items():
        if root not in path.parents or direct.sha256_file(path) != expected_sha:
            raise ValueError(f"Frozen Pure-CNN {role} path/SHA drift")
    output_rows = pd.read_csv(
        declared["output_hash_manifest"][0], dtype=str, keep_default_na=False
    )
    for role in ("pair_metrics", "checkpoint_allowlist"):
        path, expected_sha = declared[role]
        relative = path.relative_to(root).as_posix()
        selected = output_rows.loc[output_rows["relative_path"].eq(relative)]
        if (
            len(selected) != 1
            or selected.iloc[0]["sha256"] != expected_sha
            or int(selected.iloc[0]["size_bytes"]) != path.stat().st_size
        ):
            raise ValueError(f"Pure-CNN output manifest does not bind {role}")
    allowlist = pd.read_csv(
        declared["checkpoint_allowlist"][0], dtype=str, keep_default_na=False
    )
    generators = allowlist.loc[
        allowlist["checkpoint_role"].eq("generator_best_learned")
    ]
    expected_by_fold = {
        str(key): str(value)
        for key, value in direct._mapping(
            reference.get("generator_checkpoint_sha256_by_fold"),
            "Pure-CNN generator hashes",
        ).items()
    }
    observed_by_fold = dict(zip(generators["fold"], generators["checkpoint_sha256"]))
    if expected_by_fold != observed_by_fold or set(expected_by_fold) != set(
        direct.FOLDS
    ):
        raise ValueError("Frozen Pure-CNN Generator allowlist drift")
    for row in generators.itertuples(index=False):
        path = Path(row.checkpoint_path).resolve()
        if (
            root not in path.parents
            or path.stat().st_size != int(row.size_bytes)
            or direct.sha256_file(path) != row.checkpoint_sha256
        ):
            raise ValueError(f"Frozen Pure-CNN checkpoint drift: {path}")
    return reference


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = direct.resolve_path(config_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Direct matched-FiLM-LR config must be a mapping")
    config = deepcopy(dict(raw))
    config["source_config_path"] = str(source)
    config["source_config_sha256"] = direct.sha256_file(source)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    with lr_sweep_profile():
        _BASE_VALIDATE_CONFIG(config)
    matrix = direct._mapping(config.get("matrix"), "matrix")
    training = direct._mapping(config.get("training"), "training")
    rates = {
        str(key): float(value)
        for key, value in direct._mapping(
            matrix.get("arm_film_learning_rates"), "arm FiLM LRs"
        ).items()
    }
    if rates != dict(FILM_LEARNING_RATES):
        raise ValueError("FiLM learning-rate arm contract drift")
    if training.get("generator_optimizer_profile") != GENERATOR_OPTIMIZER_PROFILE:
        raise ValueError("Split Generator optimizer profile is required")
    fixed = {
        "generator_learning_rate": BACKBONE_LEARNING_RATE,
        "generator_text_learning_rate": TEXT_LEARNING_RATE,
        "discriminator_learning_rate": CRITIC_LEARNING_RATE,
        "scheduler_min_lr": BACKBONE_MIN_LEARNING_RATE,
        "generator_text_min_learning_rate": TEXT_MIN_LEARNING_RATE,
        "group_scheduler_floor_ratio": GROUP_FLOOR_RATIO,
    }
    for key, expected in fixed.items():
        if not math.isclose(
            float(training.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1e-18
        ):
            raise ValueError(f"training.{key} drift")
    representations = direct._mapping(
        config.get("text_representations"), "text_representations"
    )
    if set(representations) != set(DIRECT_ARMS) or any(
        direct._mapping(representations[arm], f"text_representations.{arm}").get("mode")
        != "lp_pair_mean_l2_v1"
        for arm in DIRECT_ARMS
    ):
        raise ValueError("Every LR arm must use matched pair-level LP")
    _validate_pure_reference(config)


def _source_paths(config: Mapping[str, Any]) -> list[tuple[str, Path]]:
    rows = list(_BASE_SOURCE_PATHS(config))
    reference = _pure_reference(config)
    rows.extend(
        (
            role,
            direct.resolve_path(reference[key]),
        )
        for role, key in (
            ("frozen_pure_output_hash_manifest", "output_hash_manifest_path"),
            ("frozen_pure_pair_metrics", "pair_metrics_path"),
            ("frozen_pure_checkpoint_allowlist", "checkpoint_allowlist_path"),
        )
    )
    if len({role for role, _path in rows}) != len(rows):
        raise ValueError("Source-manifest roles must be unique")
    return rows


def _overlay_mode(arm: str) -> str:
    if str(arm) not in DIRECT_ARMS:
        raise ValueError(f"Unknown matched-FiLM-LR arm: {arm}")
    return "lp_mean_l2"


def planned_specs(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    resolved = load_config() if config is None else deepcopy(dict(config))
    validate_config(resolved)
    with lr_sweep_profile():
        initial = direct._initial_state_hashes(resolved)
    assignment = direct._mapping(
        resolved["runtime"]["gpu_fold_assignment"], "gpu assignment"
    )
    specs: list[dict[str, Any]] = []
    for fold in direct.FOLDS:
        for arm in DIRECT_ARMS:
            film_lr = float(FILM_LEARNING_RATES[arm])
            specs.append(
                {
                    "stage": direct.DIRECT_STAGE,
                    "tolerance_minutes": direct.TOLERANCE_MINUTES,
                    "fold": fold,
                    "seed": direct.SEED,
                    "arm": arm,
                    "gpu_id": int(assignment[fold]),
                    "job_id": direct._job_id(fold, arm),
                    "pair_text_overlay_mode": "lp_mean_l2",
                    "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
                    "generator_text_learning_rate": TEXT_LEARNING_RATE,
                    "generator_film_learning_rate": film_lr,
                    "generator_text_min_learning_rate": TEXT_MIN_LEARNING_RATE,
                    "generator_film_min_learning_rate": film_lr * GROUP_FLOOR_RATIO,
                    **initial,
                }
            )
    if (
        len(specs) != EXPECTED_TRAINING_JOBS
        or len({str(row["job_id"]) for row in specs}) != EXPECTED_TRAINING_JOBS
    ):
        raise AssertionError("Matched-FiLM-LR sweep requires 20 unique jobs")
    with lr_sweep_profile():
        direct.validate_gpu_balance(specs)
    return specs


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
    for fold in direct.FOLDS:
        source = output_dir / "tolerance_05m" / fold / "lp_matched.json"
        payload = direct.read_json(source)
        for arm in DIRECT_ARMS:
            film_lr = float(FILM_LEARNING_RATES[arm])
            write_pair_text_overlay_manifest(
                direct._overlay_path(root, fold, arm),
                mode="lp_mean_l2",
                namespace=f"tol05/{fold}/{arm}/direct_matched_film_lr_v1",
                records=list(payload["records"]),
                transform={
                    **dict(payload.get("transform") or {}),
                    "direct_arm": arm,
                    "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
                    "generator_film_learning_rate": film_lr,
                    "parent_or_continuation_state_input": False,
                },
            )
    expected = {
        direct._overlay_path(root, fold, arm)
        for fold in direct.FOLDS
        for arm in DIRECT_ARMS
    }
    for path in output_dir.glob("tolerance_*/*/*.json"):
        if path.resolve() not in expected:
            path.unlink()
    paths = sorted(output_dir.glob("tolerance_05m/*/*.json"))
    if (
        len(paths) != EXPECTED_TRAINING_JOBS
        or {path.resolve() for path in paths} != expected
    ):
        raise ValueError("Matched-FiLM-LR development overlay universe drift")
    rows: list[dict[str, Any]] = []
    for fold in direct.FOLDS:
        payloads = [
            direct.read_json(direct._overlay_path(root, fold, arm))
            for arm in DIRECT_ARMS
        ]
        reference_records = payloads[0]["records"]
        if any(payload["records"] != reference_records for payload in payloads[1:]):
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
    training = direct._mapping(config["training"], "training")
    model = direct._mapping(config["model"], "model")
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
        "seed": direct.SEED,
    }


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

    data = direct._mapping(config["data"], "data")
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
    for fold in direct.FOLDS:
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
        panel_path = direct.core._write_dataframe_csv(
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
    expected_files = len(direct.FOLDS) * (1 + len(DIRECT_ARMS))
    if len(artifact_rows) != expected_files:
        raise ValueError(f"Expected {expected_files} frozen test inputs")
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
        test_panels=len(direct.FOLDS),
        test_overlays=EXPECTED_PREDICTION_CELLS,
    )
    return direct._validate_test_inputs(root)


@contextmanager
def lr_sweep_profile() -> Iterator[None]:
    """Install the 20-cell LR-sweep contract and restore shared globals."""

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
        "_overlay_mode": _overlay_mode,
        "_materialize_pair_text_overlays": _materialize_pair_text_overlays,
        "_training_payload": _training_payload,
        "_materialize_test_inputs": _materialize_test_inputs,
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
    with lr_sweep_profile():
        return direct.model_contract(config)


def validate_gpu_balance(specs: Sequence[Mapping[str, Any]]) -> None:
    with lr_sweep_profile():
        direct.validate_gpu_balance(specs)


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    with lr_sweep_profile():
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
    with lr_sweep_profile():
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


def _optimizer_contract(
    *,
    job: Mapping[str, Any],
    metrics: pd.DataFrame,
    epochs_ran: int,
) -> dict[str, Any]:
    expected_epochs = list(range(0, int(epochs_ran) + 1))
    observed_epochs = pd.to_numeric(metrics["epoch"], errors="raise").astype(int)
    if observed_epochs.tolist() != expected_epochs:
        raise ValueError(f"Non-contiguous training metrics: {job['job_id']}")
    configured = {
        "backbone": BACKBONE_LEARNING_RATE,
        "text_encoder": float(job["generator_text_learning_rate"]),
        "film_projection": float(job["generator_film_learning_rate"]),
        "critic": CRITIC_LEARNING_RATE,
    }
    columns = {
        "backbone": "g_lr_backbone",
        "text_encoder": "g_lr_text_encoder",
        "film_projection": "g_lr_film_projection",
        "critic": "d_lr",
    }
    counts = {**GENERATOR_GROUP_PARAMETER_COUNTS, "critic": 729_157}
    groups: dict[str, Any] = {}
    for name, column in columns.items():
        if column not in metrics.columns:
            raise ValueError(f"Missing {column} LR evidence: {job['job_id']}")
        values = pd.to_numeric(metrics[column], errors="raise").astype(float)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError(f"Invalid {name} LR trace: {job['job_id']}")
        if not math.isclose(
            float(values.iloc[0]), configured[name], rel_tol=0.0, abs_tol=1e-18
        ):
            raise ValueError(f"Initial {name} LR drift: {job['job_id']}")
        groups[name] = {
            "parameter_count": counts[name],
            "configured_learning_rate": configured[name],
            "initial_learning_rate": float(values.iloc[0]),
            "final_learning_rate": float(values.iloc[-1]),
            "lr_trace": [
                {"epoch": int(epoch), "lr": float(value)}
                for epoch, value in zip(observed_epochs, values, strict=True)
            ],
        }
    return {
        "schema_version": 1,
        "kind": "film_unet_split_lr_optimizer_contract_v1",
        "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
        "groups": groups,
    }


def _training_summary(root: Path) -> Path:
    destination = root / "analysis/training_summary.csv"
    if destination.is_file():
        return destination
    checkpoints = direct._checkpoint_map(root)
    rows: list[dict[str, Any]] = []
    registry = direct.read_registry(root)
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
        contract = _optimizer_contract(
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
        raise ValueError("Training summary requires exactly 20 LR-sweep cells")
    return direct.write_csv(destination, rows, tuple(rows[0]))


def _verify_analysis_manifest(path: Path) -> dict[str, Any]:
    payload = direct.read_json(path)
    if payload.get("kind") != "film_unet_direct_matched_lr_seed42_analysis_manifest_v1":
        raise ValueError("FiLM-LR analysis manifest kind drift")
    rows = [
        (str(section), dict(row))
        for section in ("inputs", "artifacts")
        for row in payload.get(section) or []
    ]
    if not rows or len({(section, row.get("role")) for section, row in rows}) != len(
        rows
    ):
        raise ValueError("FiLM-LR analysis manifest roles are empty or duplicated")
    for section, row in rows:
        target = Path(str(row.get("path", ""))).resolve()
        if (
            not target.is_file()
            or target.stat().st_size != int(row.get("size_bytes", -1))
            or direct.sha256_file(target) != str(row.get("sha256", ""))
        ):
            raise ValueError(f"FiLM-LR analysis {section} artifact drift: {target}")
    return payload


def postprocess(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    with lr_sweep_profile():
        config = direct.validate_root(root)
        registry = direct.read_registry(root)
        if registry.get("terminal_complete"):
            return qa(root)
        direct._validate_predictions(root)
        training_summary = _training_summary(root)
        analysis_manifest = lr_analysis.analyze_experiment(root, config)
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
    with lr_sweep_profile():
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
        _validate_pure_reference(config)
        training = pd.read_csv(root / "analysis/training_summary.csv")
        pairs = pd.read_csv(root / "analysis/rq12_pair_metrics.csv.gz")
        if (
            len(training) != EXPECTED_TRAINING_JOBS
            or len(pairs) != EXPECTED_PAIR_METRIC_ROWS
        ):
            raise ValueError("FiLM-LR terminal evidence count drift")
        fairness = direct._verify_frozen_file(
            registry["initial_state_fairness_path"],
            registry["initial_state_fairness_sha256"],
        )
        if direct.read_json(fairness).get("job_count") != EXPECTED_TRAINING_JOBS:
            raise ValueError("FiLM-LR initial-state fairness evidence drift")
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
            "matched_lp_arms": len(DIRECT_ARMS),
            "generator_optimizer_profile": GENERATOR_OPTIMIZER_PROFILE,
            "test_opened_after_checkpoint_freeze": True,
            "shared_mc64_noise_bank_with_frozen_pure_cnn": True,
            "pure_cnn_retrained": False,
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
            with lr_sweep_profile():
                direct.validate_root(root)
                direct._validate_output_hashes(root)
            return root
    with lr_sweep_profile():
        lock_path, descriptor = direct._pipeline_lock(root)
    try:
        with lr_sweep_profile():
            direct._pipeline_journal(root, "benchmark", "running")
        benchmark(config_path, root, resume=resume)
        with lr_sweep_profile():
            direct._pipeline_journal(root, "prepare", "running")
        prepare_experiment(config_path, root, resume=True)
        if _call("read_registry", root).get("status") == "prepared":
            with lr_sweep_profile():
                direct._pipeline_journal(
                    root,
                    "direct_arms",
                    "running",
                    completed=status(root)["job_status_counts"].get("completed", 0),
                )
            launch(root, resume=True)
        if not _call("read_registry", root).get("evaluation_frozen"):
            with lr_sweep_profile():
                direct._pipeline_journal(root, "freeze_evaluation", "running")
            freeze_evaluation(root, resume=True)
        if not _call("read_registry", root).get("predictions_frozen"):
            with lr_sweep_profile():
                direct._pipeline_journal(root, "predict", "running")
            predict(root, resume=True)
        with lr_sweep_profile():
            direct._pipeline_journal(root, "postprocess", "running")
        postprocess(root, resume=True)
        with lr_sweep_profile():
            direct._pipeline_journal(root, "terminal", "completed")
        return root
    except BaseException as exc:
        with lr_sweep_profile():
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
