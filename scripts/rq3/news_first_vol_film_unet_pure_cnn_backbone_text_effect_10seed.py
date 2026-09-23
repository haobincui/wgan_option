"""Pure-CNN parent -> FiLM text-effectiveness experiment.

This branch-local orchestrator deliberately keeps the two executable model
profiles in separate stage roots beneath one public experiment root.  It
reuses the audited direct-training lifecycle for data materialization,
training, checkpoint selection, and inference while adding a strict
weights-only graft boundary.  Test data remain inaccessible until all 280
selected checkpoints have been frozen by the public registry.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd
import torch
import yaml

from scripts.rq123 import news_first_vol_film_nolp_10seed as core
from scripts.rq3 import news_first_vol_cnn_unet_pure_no_text_10seed as pure
from scripts.rq3 import news_first_vol_film_unet_direct_5arm_seed42 as direct
from scripts.rq3 import news_first_vol_film_unet_text_10seed_lr2p5e5 as film_text


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    "configs/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed.yaml"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq12_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_5m_v1"
)
EXPERIMENT_KIND = (
    "rq12_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_rolling_v1"
)
INTERPRETATION = "retrospective_rolling_development_text_effectiveness"
REGISTRY_KIND = "pure_cnn_parent_film_text_effectiveness_registry_v1"
QA_KIND = "pure_cnn_parent_film_text_effectiveness_terminal_qa_v1"
BENCHMARK_KIND = "pure_cnn_parent_film_text_effectiveness_benchmark_v1"
ALLOWLIST_KIND = "pure_cnn_parent_film_text_effectiveness_allowlist_v1"
GRAFT_ALLOWLIST_KIND = "pure_cnn_parent_film_graft_allowlist_v1"
SUPERVISOR_KIND = "pure_cnn_parent_film_text_effectiveness_supervisor_v1"

SEEDS = pure.SEEDS
FOLDS = pure.FOLDS
PARENT_ARM = "pure_cnn_parent"
PURE_CONTINUATION_ARM = "pure_cnn_continue_no_text"
FILM_ARMS = (
    "film_zero_text",
    "film_lp_matched",
    "film_lp_shuffle",
    "film_bow",
    "film_sentiment",
)
CONTINUATION_ARMS = (PURE_CONTINUATION_ARM, *FILM_ARMS)
STANDARD_ARMS = (PARENT_ARM, *CONTINUATION_ARMS)
INTERVENTION_MODES = (
    "matched_checkpoint_zero_text_input",
    "matched_checkpoint_independent_lp_shuffle_input",
)
SNAPSHOT_EPOCHS = (0, 1, 5, 10, 20, 30)
LEARNED_SNAPSHOT_EPOCHS = SNAPSHOT_EPOCHS[1:]
EXPECTED_PARENT_JOBS = 40
EXPECTED_CONTINUATION_JOBS = 240
EXPECTED_TRAINING_JOBS = 280
EXPECTED_STANDARD_PREDICTIONS = 280
EXPECTED_INTERVENTION_PREDICTIONS = 80
EXPECTED_VALIDATION_TRAJECTORY_UNITS = 1_680
EXPECTED_VALIDATION_TRAJECTORY_PAIR_ROWS = 210_420
EXPECTED_STANDARD_PAIR_ROWS = 35_000
EXPECTED_INTERVENTION_PAIR_ROWS = 10_000
PURE_COUNTS = {"generator": 416_353, "critic": 729_157, "total": 1_145_510}
FILM_COUNTS = {"generator": 827_745, "critic": 729_157, "total": 1_556_902}
PURE_MODE = "cnn_unet_mask_coords_v1"
FILM_MODE = "film_unet_mask_coords_v1"
CRITIC_MODE = "lp_disabled_same_shape_v1"
WORKER_MODULE = (
    "scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed"
)

_BASE_PURE_INITIAL_STATE_HASHES = pure._initial_state_hashes
_BASE_DIRECT_ARTIFACT_PATHS = direct._direct_artifact_paths
_BASE_DIRECT_SOURCE_PATHS = direct._source_paths
_BASE_DIRECT_CHECKPOINT_STATE_SHA256 = direct._checkpoint_state_sha256


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256_file(path: str | Path) -> str:
    return direct.sha256_file(Path(path))


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = resolve_path(config_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Text-effectiveness config must be a mapping")
    config = deepcopy(dict(raw))
    config["source_config_path"] = str(source)
    config["source_config_sha256"] = sha256_file(source)
    validate_config(config, verify_source_evidence=False)
    return config


def _load_frozen_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Frozen text-effectiveness config must be a mapping")
    config = dict(raw)
    validate_config(config, verify_source_evidence=False)
    return config


def _expected_fold_counts() -> dict[str, tuple[int, int, int, int, int, int]]:
    return {
        "f1_2023q1": (382, 93, 144, 40, 110, 34),
        "f2_2023q2": (526, 133, 110, 34, 112, 36),
        "f3_2023q3": (636, 167, 112, 36, 135, 33),
        "f4_2023q4": (748, 203, 135, 33, 143, 45),
    }


def _verify_source_evidence(config: Mapping[str, Any]) -> None:
    evidence = _mapping(config.get("source_evidence"), "source_evidence")
    if set(evidence) != {"film_text_direct", "pure_cnn_direct", "direct_composite"}:
        raise ValueError("Frozen direct-evidence universe drift")
    for label, raw in evidence.items():
        record = _mapping(raw, f"source_evidence.{label}")
        if bool(record.get("checkpoint_reuse_allowed", True)):
            raise ValueError(f"Historical checkpoint reuse is forbidden: {label}")
        for key in (
            "source_config",
            "resolved_config",
            "qa",
            "analysis_manifest",
            "pair_metrics",
            "output_manifest",
        ):
            path = resolve_path(record[f"{key}_path"])
            expected = str(record[f"{key}_sha256"])
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Frozen source evidence drift: {label}/{key}")


def validate_config(
    config: Mapping[str, Any], *, verify_source_evidence: bool = True
) -> None:
    experiment = _mapping(config.get("experiment"), "experiment")
    data = _mapping(config.get("data"), "data")
    matrix = _mapping(config.get("matrix"), "matrix")
    models = _mapping(config.get("models"), "models")
    common_model = _mapping(models.get("common"), "models.common")
    pure_model = _mapping(models.get("pure_cnn"), "models.pure_cnn")
    film_model = _mapping(models.get("film_unet"), "models.film_unet")
    training = _mapping(config.get("training"), "training")
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    analysis = _mapping(config.get("analysis"), "analysis")
    runtime = _mapping(config.get("runtime"), "runtime")
    if int(experiment.get("schema_version", -1)) != 1:
        raise ValueError("experiment.schema_version must be 1")
    if experiment.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment kind drift")
    if experiment.get("interpretation") != INTERPRETATION:
        raise ValueError("Interpretation drift")
    if resolve_path(experiment.get("output_root", "")) != resolve_path(
        DEFAULT_OUTPUT_DIR
    ):
        raise ValueError("Formal output root drift")
    if tuple(map(int, data.get("tolerances_minutes", ()))) != (5,):
        raise ValueError("Only 5m alignment is permitted")
    if tuple(map(int, data.get("maturity_days_grid", ()))) != direct.GRID:
        raise ValueError("Exact-TTM axis drift")
    if data.get("support_mask_mode") != "raw_joint":
        raise ValueError("support_mask_mode must be raw_joint")
    if int(data.get("embedding_dimension", -1)) != 1024:
        raise ValueError("Text width must be 1024")
    folds = list(config.get("folds") or [])
    if tuple(str(row.get("id")) for row in folds) != FOLDS:
        raise ValueError("Rolling-fold order drift")
    counts = _mapping(config.get("expected_pair_session_counts"), "counts")
    observed_counts = _mapping(counts.get(5, counts.get("5")), "counts.5")
    keys = (
        "train_pairs",
        "train_sessions",
        "validation_pairs",
        "validation_sessions",
        "test_pairs",
        "test_sessions",
    )
    for fold, expected in _expected_fold_counts().items():
        observed = tuple(int(observed_counts[fold][key]) for key in keys)
        if observed != expected:
            raise ValueError(f"Pair/session count drift: {fold}")
    if tuple(map(int, matrix.get("seeds", ()))) != SEEDS:
        raise ValueError("Ten-seed universe drift")
    if tuple(matrix.get("parent_arms", ())) != (PARENT_ARM,):
        raise ValueError("Parent arm drift")
    if tuple(matrix.get("branch_arms", ())) != CONTINUATION_ARMS:
        raise ValueError("Continuation arm drift")
    if tuple(matrix.get("standard_prediction_arms", ())) != STANDARD_ARMS:
        raise ValueError("Standard prediction arm drift")
    expected_matrix = {
        "expected_parent_jobs": EXPECTED_PARENT_JOBS,
        "expected_branch_jobs": EXPECTED_CONTINUATION_JOBS,
        "expected_training_jobs": EXPECTED_TRAINING_JOBS,
        "expected_standard_prediction_cells": EXPECTED_STANDARD_PREDICTIONS,
        "expected_intervention_prediction_cells": EXPECTED_INTERVENTION_PREDICTIONS,
        "expected_total_prediction_cells": 360,
        "expected_standard_pair_metric_rows": EXPECTED_STANDARD_PAIR_ROWS,
        "expected_intervention_pair_metric_rows": EXPECTED_INTERVENTION_PAIR_ROWS,
        "expected_total_pair_metric_rows": 45_000,
    }
    for key, expected in expected_matrix.items():
        if int(matrix.get(key, -1)) != expected:
            raise ValueError(f"matrix.{key} drift")
    if common_model.get("critic_conditioning_mode") != CRITIC_MODE:
        raise ValueError("NoLP Critic contract drift")
    if pure_model.get("generator_conditioning_mode") != PURE_MODE:
        raise ValueError("Pure-CNN mode drift")
    if film_model.get("generator_conditioning_mode") != FILM_MODE:
        raise ValueError("FiLM mode drift")
    if {
        "generator": int(pure_model.get("expected_generator_parameters", -1)),
        "critic": int(common_model.get("expected_critic_parameters", -1)),
        "total": int(pure_model.get("expected_total_parameters", -1)),
    } != PURE_COUNTS:
        raise ValueError("Pure-CNN parameter-count drift")
    if {
        "generator": int(film_model.get("expected_generator_parameters", -1)),
        "critic": int(common_model.get("expected_critic_parameters", -1)),
        "total": int(film_model.get("expected_total_parameters", -1)),
    } != FILM_COUNTS:
        raise ValueError("FiLM parameter-count drift")
    required_training: Mapping[str, Any] = {
        "parent_num_epochs": 240,
        "branch_num_epochs": 240,
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
    if (
        tuple(map(int, training.get("validation_snapshot_epochs", ())))
        != SNAPSHOT_EPOCHS
    ):
        raise ValueError("Validation snapshot schedule drift")
    for key, expected in {
        "generator_backbone_learning_rate": 5e-7,
        "generator_text_learning_rate": 2.5e-6,
        "generator_film_learning_rate": 2.5e-5,
        "discriminator_learning_rate": 5e-7,
        "generator_backbone_min_learning_rate": 5e-8,
        "generator_text_min_learning_rate": 2.5e-7,
        "generator_film_min_learning_rate": 2.5e-6,
        "discriminator_min_learning_rate": 5e-8,
    }.items():
        if not math.isclose(
            float(training.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1e-18
        ):
            raise ValueError(f"training.{key} drift")
    interventions = _mapping(evaluation.get("interventions"), "interventions")
    if not bool(evaluation.get("freeze_all_checkpoints_before_test")):
        raise ValueError("All 280 checkpoints must be frozen before test access")
    if bool(evaluation.get("test_loader_allowed_before_freeze", True)):
        raise ValueError("Test loaders must remain disabled before evaluation freeze")
    if int(evaluation.get("standard_predictions", {}).get("expected_cells", -1)) != 280:
        raise ValueError("Standard prediction count drift")
    if int(interventions.get("expected_cells", -1)) != 80:
        raise ValueError("Intervention count drift")
    input_modes = [
        _mapping(row, "evaluation.interventions.input_modes")
        for row in interventions.get("input_modes", ())
    ]
    if (
        interventions.get("source_checkpoint_arm") != "film_lp_matched"
        or tuple(row.get("id") for row in input_modes) != INTERVENTION_MODES
        or len(input_modes) != 2
        or input_modes[0].get("representation") != "zero_vector_v1"
        or input_modes[1].get("representation") != "lp_pair_mean_l2_v1"
        or input_modes[1].get("pair_mapping") != "split_local_derangement_v1"
        or int(input_modes[1].get("mapping_seed", -1)) != 20260906
        or not bool(interventions.get("mapping_must_differ_from_training_shuffle"))
    ):
        raise ValueError("Frozen intervention source/input/mapping contract drift")
    if int(analysis.get("bootstrap_replicates", -1)) != 10_000:
        raise ValueError("Bootstrap replicate drift")
    if analysis.get("multiple_testing_method") != "holm":
        raise ValueError("Primary multiplicity method must be Holm")
    if analysis.get("interpretation") != INTERPRETATION:
        raise ValueError("Analysis interpretation drift")
    primary = _mapping(analysis.get("primary_family"), "analysis.primary_family")
    support_gate = _mapping(
        analysis.get("primary_support_gate"), "analysis.primary_support_gate"
    )
    if int(primary.get("family_size", -1)) != 2:
        raise ValueError("Primary family must remain Holm-2")
    if (
        int(support_gate.get("minimum_nonworse_seeds", -1)) != 7
        or int(support_gate.get("minimum_nonworse_folds", -1)) != 3
    ):
        raise ValueError("Primary 7/10-seed and 3/4-fold support gate drift")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != (0, 1):
        raise ValueError("Both A30 devices are required")
    if int(runtime.get("parent_blocks_per_gpu", -1)) != 20:
        raise ValueError("Parent GPU balance drift")
    if int(runtime.get("branch_jobs_per_gpu", -1)) != 120:
        raise ValueError("Continuation GPU balance drift")
    if verify_source_evidence:
        _verify_source_evidence(config)


def parent_job_id(seed: int, fold: str) -> str:
    return f"backbone_05m_{fold}_seed_{int(seed)}_{PARENT_ARM}"


def continuation_job_id(seed: int, fold: str, arm: str) -> str:
    if arm not in CONTINUATION_ARMS:
        raise ValueError(f"Unknown continuation arm: {arm}")
    return f"continuation_05m_{fold}_seed_{int(seed)}_{arm}"


def planned_job_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_config(config, verify_source_evidence=False)
    assignment = _mapping(config["runtime"]["gpu_fold_assignment"], "GPU assignment")
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            gpu = int(assignment[fold])
            rows.append(
                {
                    "job_id": parent_job_id(seed, fold),
                    "stage": "backbone",
                    "seed": seed,
                    "fold": fold,
                    "arm": PARENT_ARM,
                    "gpu_id": gpu,
                    "parent_job_id": "",
                    "graft_state_path": "",
                    "graft_state_sha256": "",
                    "status": "pending",
                }
            )
            for arm in CONTINUATION_ARMS:
                rows.append(
                    {
                        "job_id": continuation_job_id(seed, fold, arm),
                        "stage": "continuation",
                        "seed": seed,
                        "fold": fold,
                        "arm": arm,
                        "gpu_id": gpu,
                        "parent_job_id": parent_job_id(seed, fold),
                        "graft_state_path": "",
                        "graft_state_sha256": "",
                        "status": "blocked_on_graft",
                    }
                )
    if (
        len(rows) != EXPECTED_TRAINING_JOBS
        or len({str(row["job_id"]) for row in rows}) != EXPECTED_TRAINING_JOBS
    ):
        raise AssertionError("Expected exactly 280 unique training jobs")
    parent_balance = {
        gpu: sum(row["stage"] == "backbone" and row["gpu_id"] == gpu for row in rows)
        for gpu in (0, 1)
    }
    continuation_balance = {
        gpu: sum(
            row["stage"] == "continuation" and row["gpu_id"] == gpu for row in rows
        )
        for gpu in (0, 1)
    }
    if parent_balance != {0: 20, 1: 20} or continuation_balance != {0: 120, 1: 120}:
        raise ValueError("Frozen 20/20 parent or 120/120 continuation balance drift")
    return rows


def _root(output_dir: str | Path) -> Path:
    return Path(output_dir).expanduser().resolve()


def _control_root(root: Path) -> Path:
    return root.with_name(root.name + "_control")


def _registry_path(root: Path) -> Path:
    return root / "registry/task_registry.json"


def _read_json(path: Path) -> dict[str, Any]:
    return direct.read_json(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    return direct.write_json(path, payload)


def _write_registry(root: Path, payload: Mapping[str, Any]) -> Path:
    registry = dict(payload)
    registry["jobs_sha256"] = direct.payload_sha256(registry.get("jobs", []))
    registry["updated_at_utc"] = direct.utc_now()
    return _write_json(_registry_path(root), registry)


def _read_registry(root: Path) -> dict[str, Any]:
    return _read_json(_registry_path(root))


def _stage_roots(root: Path) -> dict[str, Path]:
    return {
        "backbones": root / "stages/backbones",
        "pure_continuation": root / "stages/continuations/pure_cnn",
        "film_continuations": root / "stages/continuations/film_unet",
    }


def _validate_registry_state(
    config: Mapping[str, Any], registry: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Validate the public 40+240 state machine without touching stage data."""

    if (
        int(registry.get("schema_version", -1)) != 1
        or registry.get("kind") != REGISTRY_KIND
        or registry.get("experiment_kind") != EXPERIMENT_KIND
        or registry.get("interpretation") != INTERPRETATION
    ):
        raise ValueError("Text-effectiveness registry identity drift")
    raw_jobs = list(registry.get("jobs") or [])
    if not all(isinstance(row, Mapping) for row in raw_jobs):
        raise ValueError("Text-effectiveness registry jobs must be mappings")
    jobs = [dict(row) for row in raw_jobs]
    if (
        len(jobs) != EXPECTED_TRAINING_JOBS
        or len({str(row.get("job_id", "")) for row in jobs}) != EXPECTED_TRAINING_JOBS
        or registry.get("jobs_sha256") != direct.payload_sha256(jobs)
    ):
        raise ValueError("Text-effectiveness registry must contain 280 unique jobs")
    expected = {row["job_id"]: row for row in planned_job_specs(config)}
    observed = {str(row["job_id"]): row for row in jobs}
    if set(observed) != set(expected):
        raise ValueError("Text-effectiveness logical job universe drift")
    invariant_fields = (
        "stage",
        "seed",
        "fold",
        "arm",
        "gpu_id",
        "parent_job_id",
    )
    for job_id_value, expected_job in expected.items():
        job = observed[job_id_value]
        if any(job.get(field) != expected_job.get(field) for field in invariant_fields):
            raise ValueError(f"Text-effectiveness job identity drift: {job_id_value}")
    stage_counts = {
        stage: sum(str(row["stage"]) == stage for row in jobs)
        for stage in ("backbone", "continuation")
    }
    if stage_counts != {
        "backbone": EXPECTED_PARENT_JOBS,
        "continuation": EXPECTED_CONTINUATION_JOBS,
    }:
        raise ValueError("Text-effectiveness 40/240 stage count drift")

    implications = (
        ("grafts_frozen", "parents_frozen"),
        ("continuations_prepared", "grafts_frozen"),
        ("evaluation_frozen", "continuations_prepared"),
        ("test_data_opened", "evaluation_frozen"),
        ("standard_predictions_frozen", "test_data_opened"),
        ("interventions_frozen", "standard_predictions_frozen"),
        ("validation_trajectories_frozen", "evaluation_frozen"),
        ("terminal_complete", "analysis_complete"),
    )
    for consequence, prerequisite in implications:
        if bool(registry.get(consequence)) and not bool(registry.get(prerequisite)):
            raise ValueError(
                f"Invalid registry phase: {consequence}=true requires {prerequisite}=true"
            )
    if bool(registry.get("analysis_complete")) and not (
        bool(registry.get("standard_predictions_frozen"))
        and bool(registry.get("interventions_frozen"))
        and bool(registry.get("validation_trajectories_frozen"))
    ):
        raise ValueError(
            "Invalid registry phase: analysis_complete requires all three evidence layers"
        )
    continuation_jobs = [row for row in jobs if row["stage"] == "continuation"]
    if registry.get("grafts_frozen"):
        required_graft_fields = (
            "graft_state_path",
            "graft_state_sha256",
            "graft_manifest_path",
            "graft_manifest_sha256",
            "initial_generator_state_sha256",
            "initial_critic_state_sha256",
        )
        for job in continuation_jobs:
            if any(not str(job.get(field, "")) for field in required_graft_fields):
                raise ValueError(f"Continuation lacks frozen graft: {job['job_id']}")
    elif any(
        str(job.get(field, ""))
        for job in continuation_jobs
        for field in ("graft_state_path", "graft_state_sha256")
    ):
        raise ValueError("Unfrozen registry contains continuation graft bindings")
    if (
        registry.get("evaluation_frozen")
        and int(registry.get("checkpoint_allowlist_rows", -1))
        != EXPECTED_TRAINING_JOBS * 2
    ):
        raise ValueError("Frozen evaluation must contain 560 checkpoint rows")
    if registry.get("standard_predictions_frozen") and (
        int(registry.get("standard_prediction_cells", -1))
        != EXPECTED_STANDARD_PREDICTIONS
        or int(registry.get("standard_pair_metric_rows", -1))
        != EXPECTED_STANDARD_PAIR_ROWS
    ):
        raise ValueError("Frozen standard prediction 280/35,000 count drift")
    if registry.get("interventions_frozen") and (
        int(registry.get("intervention_prediction_cells", -1))
        != EXPECTED_INTERVENTION_PREDICTIONS
        or int(registry.get("intervention_pair_metric_rows", -1))
        != EXPECTED_INTERVENTION_PAIR_ROWS
    ):
        raise ValueError("Frozen intervention prediction 80/10,000 count drift")
    if registry.get("validation_trajectories_frozen") and (
        int(registry.get("validation_trajectory_prediction_units", -1))
        != EXPECTED_VALIDATION_TRAJECTORY_UNITS
        or int(registry.get("validation_trajectory_pair_metric_rows", -1))
        != EXPECTED_VALIDATION_TRAJECTORY_PAIR_ROWS
    ):
        raise ValueError("Frozen validation trajectory 1,680/210,420 count drift")
    return jobs


def _verify_registry_file(
    registry: Mapping[str, Any], path_key: str, sha_key: str
) -> Path:
    raw_path = str(registry.get(path_key, "")).strip()
    expected_sha = str(registry.get(sha_key, "")).strip()
    if not raw_path or not expected_sha:
        raise ValueError(f"Registry lacks frozen file lineage: {path_key}/{sha_key}")
    return direct._verify_frozen_file(raw_path, expected_sha)


def _validate_parent_allowlist(
    root: Path, registry: Mapping[str, Any], *, verify_artifacts: bool
) -> dict[tuple[int, str], dict[str, Any]]:
    path = _verify_registry_file(
        registry, "parent_allowlist_path", "parent_allowlist_sha256"
    )
    if path != (root / "registry/parent_allowlist.json").resolve():
        raise ValueError("Parent allowlist escaped the experiment root")
    payload = _read_json(path)
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    entries = [dict(row) for row in payload.get("entries") or []]
    if (
        payload.get("kind") != "pure_cnn_parent_allowlist_v1"
        or direct.payload_sha256(unsigned) != payload.get("payload_sha256")
        or len(entries) != EXPECTED_PARENT_JOBS
        or int(payload.get("test_loader_count", -1)) != 0
        or int(payload.get("test_prediction_count", -1)) != 0
    ):
        raise ValueError("Frozen parent allowlist identity/count drift")
    by_cell: dict[tuple[int, str], dict[str, Any]] = {}
    for row in entries:
        key = (int(row["seed"]), str(row["fold"]))
        if key in by_cell or str(row["parent_job_id"]) != parent_job_id(*key):
            raise ValueError(f"Duplicate or invalid parent allowlist cell: {key}")
        if int(row["best_epoch"]) != int(row["full_state_completed_epoch"]):
            raise ValueError(f"Parent full-state/best-epoch drift: {key}")
        if verify_artifacts:
            for path_key, sha_key in (
                ("generator_path", "generator_sha256"),
                ("discriminator_path", "discriminator_sha256"),
                ("full_state_path", "full_state_sha256"),
            ):
                direct._verify_frozen_file(row[path_key], row[sha_key])
            full_state = torch.load(
                Path(str(row["full_state_path"])),
                map_location="cpu",
                weights_only=False,
            )
            if (
                not isinstance(full_state, Mapping)
                or full_state.get("kind") != core.FULL_STATE_KIND
                or int(full_state.get("completed_epoch", 0)) != int(row["best_epoch"])
            ):
                raise ValueError(f"Frozen parent full-state payload drift: {key}")
            generator_state_sha = _tensor_state_sha256(
                _mapping(
                    full_state.get("generator_state_dict"),
                    "parent generator state",
                )
            )
            discriminator_state_sha = _tensor_state_sha256(
                _mapping(
                    full_state.get("discriminator_state_dict"),
                    "parent discriminator state",
                )
            )
            if (
                generator_state_sha
                != str(row.get("full_state_generator_state_sha256", ""))
                or discriminator_state_sha
                != str(row.get("full_state_discriminator_state_sha256", ""))
                or generator_state_sha
                != _canonical_checkpoint_state_sha256(row["generator_path"])
                or discriminator_state_sha
                != _canonical_checkpoint_state_sha256(row["discriminator_path"])
            ):
                raise ValueError(f"Frozen parent full-state/checkpoint drift: {key}")
        by_cell[key] = row
    expected = {(seed, fold) for seed in SEEDS for fold in FOLDS}
    if set(by_cell) != expected:
        raise ValueError("Frozen parent allowlist seed/fold universe drift")
    return by_cell


def _validate_graft_allowlist(
    root: Path, registry: Mapping[str, Any], *, verify_artifacts: bool
) -> list[dict[str, Any]]:
    path = _verify_registry_file(
        registry, "graft_allowlist_path", "graft_allowlist_sha256"
    )
    if path != (root / "registry/graft_allowlist.json").resolve():
        raise ValueError("Graft allowlist escaped the experiment root")
    payload = _read_json(path)
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    entries = [dict(row) for row in payload.get("entries") or []]
    if (
        payload.get("kind") != GRAFT_ALLOWLIST_KIND
        or direct.payload_sha256(unsigned) != payload.get("payload_sha256")
        or int(payload.get("entry_count", -1)) != 80
        or int(payload.get("film_graft_count", -1)) != 40
        or int(payload.get("pure_restart_count", -1)) != 40
        or len(entries) != 80
    ):
        raise ValueError("Frozen graft allowlist identity/count drift")
    expected = {
        (seed, fold, target_mode)
        for seed in SEEDS
        for fold in FOLDS
        for target_mode in (PURE_MODE, FILM_MODE)
    }
    observed = {
        (int(row["seed"]), str(row["fold"]), str(row["target_generator_mode"]))
        for row in entries
    }
    if observed != expected or len(observed) != len(entries):
        raise ValueError("Frozen graft seed/fold/target universe drift")
    for row in entries:
        if float(row.get("epoch0_max_abs", math.inf)) > 1e-7:
            raise ValueError("Frozen graft failed epoch-0 equivalence")
        proof = _mapping(row.get("optimizer_reset_proof"), "optimizer reset proof")
        if not all(
            bool(proof.get(key))
            for key in (
                "generator_optimizer_state_empty",
                "discriminator_optimizer_state_empty",
                "generator_scheduler_fresh",
                "discriminator_scheduler_fresh",
            )
        ) or bool(proof.get("apply_loads_optimizer_or_scheduler_state", True)):
            raise ValueError("Frozen graft lacks fresh optimizer/scheduler proof")
        if verify_artifacts:
            for path_key, sha_key in (
                ("artifact_path", "artifact_sha256"),
                ("manifest_path", "manifest_sha256"),
            ):
                direct._verify_frozen_file(row[path_key], row[sha_key])
    if verify_artifacts:
        _graft_entries(root, PURE_MODE)
        _graft_entries(root, FILM_MODE)
    return entries


def _stage_experiment_kind(stage_name: str) -> str:
    return f"{EXPERIMENT_KIND}_{stage_name}_internal_v1"


def _stage_config(
    config: Mapping[str, Any],
    *,
    stage_name: str,
    arms: Sequence[str],
    generator_mode: str,
    expected_counts: Mapping[str, int],
    workers_per_gpu: int,
    output_root: Path,
    num_epochs: int = 240,
) -> dict[str, Any]:
    common = _mapping(config["models"]["common"], "models.common")
    model_specific = _mapping(
        config["models"]["pure_cnn" if generator_mode == PURE_MODE else "film_unet"],
        "models stage",
    )
    training = _mapping(config["training"], "training")
    text_representations = _mapping(
        config["text_representations"], "text_representations"
    )
    stage_training: dict[str, Any] = {
        "protocol": "selected_parent_weights_fresh_optimizer_dynamic_validation_v1",
        "initial_state_contract": "hash_bound_weights_rng_graft_v1",
        "checkpoint_selection_mode": "best_learned_dynamic_validation_v1",
        "generator_optimizer_profile": (
            "uniform_v1" if generator_mode == PURE_MODE else "film_unet_split_lr_v1"
        ),
        "initial_learning_rate": float(training["initial_learning_rate"]),
        "generator_learning_rate": float(training["generator_backbone_learning_rate"]),
        "discriminator_learning_rate": float(training["discriminator_learning_rate"]),
        "scheduler_min_lr": float(training["generator_backbone_min_learning_rate"]),
        "reduce_lr_factor": float(training["reduce_lr_factor"]),
        "reduce_lr_patience": int(training["reduce_lr_patience"]),
        "lr_warmup_epochs": 0,
        "batch_size": int(training["batch_size"]),
        "discriminator_steps": int(training["discriminator_steps"]),
        "beta_1": float(training["beta_1"]),
        "beta_2": float(training["beta_2"]),
        "lambda_gp": float(training["lambda_gp"]),
        "lambda_recon": float(training["lambda_recon"]),
        "lambda_calendar": float(training["lambda_calendar"]),
        "lambda_butterfly": float(training["lambda_butterfly"]),
        "lambda_smooth": float(training["lambda_smooth"]),
        "lambda_delta_shrink": float(training["lambda_delta_shrink"]),
        "constraint_warmup_epochs": int(training["constraint_warmup_epochs"]),
        "num_epochs": int(num_epochs),
        "best_checkpoint_metric": str(training["best_checkpoint_metric"]),
        "baseline_penalty_weight": float(training["baseline_penalty_weight"]),
        "early_stopping_patience": int(training["early_stopping_patience"]),
        "early_stopping_min_epochs": min(
            int(training["early_stopping_min_epochs"]), int(num_epochs)
        ),
        "early_stopping_min_delta": float(training["early_stopping_min_delta"]),
        "label_reliability_mode": "none",
        "validation_mc_samples": int(training["validation_mc_samples"]),
        "prediction_mc_samples": int(training["prediction_mc_samples"]),
        "shared_prediction_noise_bank": True,
    }
    if generator_mode == FILM_MODE:
        stage_training.update(
            generator_text_learning_rate=float(
                training["generator_text_learning_rate"]
            ),
            generator_film_learning_rate=float(
                training["generator_film_learning_rate"]
            ),
            generator_text_min_learning_rate=float(
                training["generator_text_min_learning_rate"]
            ),
            generator_film_min_learning_rate=float(
                training["generator_film_min_learning_rate"]
            ),
        )
    runtime = deepcopy(dict(config["runtime"]))
    runtime["formal_workers_per_gpu"] = int(workers_per_gpu)
    runtime["benchmark_jobs"] = int(expected_counts["training_jobs"])
    runtime["benchmark_epochs"] = 1
    model = {
        **common,
        **model_specific,
        "generator_conditioning_mode": generator_mode,
        "expected_generator_parameters": int(expected_counts["generator"]),
        "expected_critic_parameters": int(expected_counts["critic"]),
        "expected_total_parameters": int(expected_counts["total"]),
    }
    # The shared Generator constructor requires the text-projection widths even
    # for ``cnn_unet_mask_coords_v1``.  Pure-CNN never instantiates or consumes
    # those layers, but keeping the constructor-only dimensions in the internal
    # stage config lets the audited common worker build the no-text model.
    # Bind them to the target FiLM profile so the later graft cannot silently
    # change its text architecture.
    film_model = _mapping(config["models"]["film_unet"], "models.film_unet")
    model.setdefault("gen_text_hidden_dim", int(film_model["gen_text_hidden_dim"]))
    model.setdefault("gen_text_out_dim", int(film_model["gen_text_out_dim"]))
    model.pop("expected_backbone_parameters", None)
    model.pop("expected_text_encoder_parameters", None)
    model.pop("expected_film_projection_parameters", None)
    return {
        "experiment": {
            "schema_version": 1,
            "experiment_kind": _stage_experiment_kind(stage_name),
            "interpretation": INTERPRETATION,
            "interface_stability": "branch_local_internal_stage",
            "output_root": str(output_root),
        },
        "data": {
            **deepcopy(dict(config["data"])),
            "surface_sheet_name": str(config["data"]["sheet_name"]),
        },
        "folds": deepcopy(list(config["folds"])),
        "expected_pair_session_counts": deepcopy(
            dict(config["expected_pair_session_counts"])
        ),
        "matrix": {
            "seeds": list(SEEDS),
            "tolerances_minutes": [5],
            "direct_arms": list(arms),
            "arm_film_learning_rates": {
                arm: float(training["generator_film_learning_rate"])
                for arm in arms
                if generator_mode == FILM_MODE
            },
            "expected_training_jobs": int(expected_counts["training_jobs"]),
            "expected_prediction_cells": int(expected_counts["training_jobs"]),
            "expected_pair_metric_rows": int(expected_counts["pair_rows"]),
            "shuffle_seed": int(config["matrix"]["training_shuffle_seed"]),
        },
        "text_representations": {
            arm: deepcopy(dict(text_representations[arm])) for arm in arms
        },
        "model": model,
        "training": stage_training,
        "analysis": {
            "interpretation": INTERPRETATION,
            "bootstrap_replicates": int(config["analysis"]["bootstrap_replicates"]),
            "bootstrap_seed": int(config["analysis"]["bootstrap_seed"]),
            "confidence_level": float(config["analysis"]["confidence_level"]),
            "resampling_hierarchy": str(config["analysis"]["resampling_hierarchy"]),
            "cross_seed_inference_enabled": True,
            "best_observed_seed_is_descriptive_only": True,
            "test_based_seed_selection_permitted": False,
            "best_seed_reporting_mode": "descriptive_only",
            "test_based_model_selection_permitted": False,
        },
        "runtime": runtime,
        "source_config_path": str(config["source_config_path"]),
        "source_config_sha256": str(config["source_config_sha256"]),
        "internal_stage_name": stage_name,
    }


def _validate_stage_config(
    config: Mapping[str, Any],
    *,
    stage_name: str,
    arms: Sequence[str],
    generator_mode: str,
    expected_jobs: int,
) -> None:
    if config.get("internal_stage_name") != stage_name:
        raise ValueError("Internal stage identity drift")
    if config.get("experiment", {}).get("experiment_kind") != _stage_experiment_kind(
        stage_name
    ):
        raise ValueError("Internal stage experiment kind drift")
    if config.get("experiment", {}).get("interpretation") != INTERPRETATION:
        raise ValueError("Internal stage interpretation drift")
    if tuple(map(int, config.get("matrix", {}).get("seeds", ()))) != SEEDS:
        raise ValueError("Internal stage seed drift")
    if tuple(config.get("matrix", {}).get("direct_arms", ())) != tuple(arms):
        raise ValueError("Internal stage arm drift")
    if int(config.get("matrix", {}).get("expected_training_jobs", -1)) != expected_jobs:
        raise ValueError("Internal stage job-count drift")
    if config.get("model", {}).get("generator_conditioning_mode") != generator_mode:
        raise ValueError("Internal stage Generator mode drift")
    if config.get("model", {}).get("critic_conditioning_mode") != CRITIC_MODE:
        raise ValueError("Internal stage Critic mode drift")
    if (
        tuple(map(int, config.get("data", {}).get("maturity_days_grid", ())))
        != direct.GRID
    ):
        raise ValueError("Internal stage exact-TTM drift")
    if config.get("data", {}).get("support_mask_mode") != "raw_joint":
        raise ValueError("Internal stage support mode drift")
    training = _mapping(config.get("training"), "stage training")
    if int(training.get("num_epochs", -1)) not in {1, 240}:
        raise ValueError("Internal stage epoch budget must be 1 or 240")
    if int(training.get("validation_mc_samples", -1)) != 16:
        raise ValueError("Internal stage validation MC drift")
    if bool(training.get("lr_warmup_epochs")):
        raise ValueError("Warmup is forbidden")


def _stage_source_paths(config: Mapping[str, Any]) -> list[tuple[str, Path]]:
    # ``pure.prepare`` temporarily installs this callback into the shared
    # direct orchestrator.  Calling ``direct._source_paths`` here would then
    # call this function recursively, so retain the unpatched implementation.
    return _BASE_DIRECT_SOURCE_PATHS(config)


def _stage_overlay_mode(arm: str) -> str:
    return {
        PARENT_ARM: "current_only",
        PURE_CONTINUATION_ARM: "current_only",
        "film_zero_text": "current_only",
        "film_lp_matched": "lp_mean_l2",
        "film_lp_shuffle": "lp_shuffle",
        "film_bow": "bow1024",
        "film_sentiment": "sentiment_pad1024",
        "film_lp_matched__zero_input": "current_only",
        "film_lp_matched__wrong_input": "lp_mean_l2",
    }[str(arm)]


def _stage_code_paths() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                "scripts/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed.py",
                "scripts/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_analysis.py",
                "scripts/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle.py",
                "scripts/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_prediction.py",
                "scripts/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_report.py",
                "scripts/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_parallel_prediction.py",
                "scripts/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_prediction_supervisor.py",
                "scripts/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_prediction_worker.py",
                "scripts/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_supervisor.py",
                "scripts/rq3/news_first_vol_experiment_supervisor_helpers.py",
                "src/wgan_option/utils/pure_cnn_film_graft.py",
                *direct.SOURCE_CODE_RELATIVE_PATHS,
            )
        )
    )


def _tensor_state_sha256(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Non-tensor model state entry: {key}")
        value = tensor.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _canonical_checkpoint_state_sha256(path: str | Path) -> str:
    """Hash a weights checkpoint with the experiment's canonical key order.

    The shared direct-training helper preserves ``state_dict`` insertion order,
    while this experiment deliberately sorts parameter keys so tensor lineage
    is independent of serialization order.  Comparing those two digest formats
    rejects byte-identical model states.  Load the checkpoint and apply the
    same canonical tensor-state digest on both sides of the comparison.
    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint lacks a state_dict: {path}")
    return _tensor_state_sha256(state)


def _graft_entries(
    root: Path, target_mode: str
) -> dict[tuple[int, str], dict[str, Any]]:
    path = root / "registry/graft_allowlist.json"
    payload = _read_json(path)
    if payload.get("kind") != GRAFT_ALLOWLIST_KIND:
        raise ValueError("Graft allowlist kind drift")
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if direct.payload_sha256(unsigned) != payload.get("payload_sha256"):
        raise ValueError("Graft allowlist self-hash drift")
    entries: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in payload.get("entries", []):
        row = dict(raw)
        if str(row["target_generator_mode"]) != target_mode:
            continue
        key = (int(row["seed"]), str(row["fold"]))
        if key in entries:
            raise ValueError(f"Duplicate graft entry: {key}/{target_mode}")
        for path_key, sha_key in (
            ("artifact_path", "artifact_sha256"),
            ("manifest_path", "manifest_sha256"),
        ):
            target = Path(str(row[path_key])).resolve()
            if not target.is_file() or sha256_file(target) != str(row[sha_key]):
                raise ValueError(f"Frozen graft entry drift: {target}")
        artifact = torch.load(
            Path(str(row["artifact_path"])), map_location="cpu", weights_only=False
        )
        if _tensor_state_sha256(artifact["generator_state_dict"]) != str(
            row["generator_state_sha256"]
        ) or _tensor_state_sha256(artifact["discriminator_state_dict"]) != str(
            row["discriminator_state_sha256"]
        ):
            raise ValueError(f"Graft state digest drift: {key}/{target_mode}")
        entries[key] = row
    expected = {(seed, fold) for seed in SEEDS for fold in FOLDS}
    if set(entries) != expected:
        raise ValueError(f"Incomplete graft target-mode universe: {target_mode}")
    return entries


def _planned_stage_specs(
    config: Mapping[str, Any],
    *,
    arm_universe: Sequence[str],
    generator_mode: str,
    main_root: Path,
    use_graft: bool,
) -> list[dict[str, Any]]:
    assignment = _mapping(config["runtime"]["gpu_fold_assignment"], "GPU assignment")
    grafts = _graft_entries(main_root, generator_mode) if use_graft else {}
    initial_by_seed = (
        {}
        if use_graft
        else {seed: _BASE_PURE_INITIAL_STATE_HASHES(config, seed) for seed in SEEDS}
    )
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            for arm in arm_universe:
                spec: dict[str, Any] = {
                    "stage": direct.DIRECT_STAGE,
                    "tolerance_minutes": 5,
                    "fold": fold,
                    "seed": seed,
                    "arm": arm,
                    "gpu_id": int(assignment[fold]),
                    "job_id": core.job_id(
                        {
                            "stage": direct.DIRECT_STAGE,
                            "tolerance_minutes": 5,
                            "fold": fold,
                            "seed": seed,
                            "arm": arm,
                        }
                    ),
                    "pair_text_overlay_mode": _stage_overlay_mode(arm),
                    "generator_optimizer_profile": (
                        "uniform_v1"
                        if generator_mode == PURE_MODE
                        else "film_unet_split_lr_v1"
                    ),
                    "validation_snapshot_epochs": tuple(
                        epoch
                        for epoch in LEARNED_SNAPSHOT_EPOCHS
                        if epoch <= int(config["training"]["num_epochs"])
                    ),
                }
                if use_graft:
                    graft = grafts[(seed, fold)]
                    spec.update(
                        initial_generator_state_sha256=str(
                            graft["generator_state_sha256"]
                        ),
                        initial_critic_state_sha256=str(
                            graft["discriminator_state_sha256"]
                        ),
                        graft_state_path=str(graft["artifact_path"]),
                        graft_state_sha256=str(graft["artifact_sha256"]),
                    )
                else:
                    spec.update(initial_by_seed[seed])
                if generator_mode == FILM_MODE:
                    training = config["training"]
                    spec.update(
                        generator_text_learning_rate=float(
                            training["generator_text_learning_rate"]
                        ),
                        generator_film_learning_rate=float(
                            training["generator_film_learning_rate"]
                        ),
                        generator_text_min_learning_rate=float(
                            training["generator_text_min_learning_rate"]
                        ),
                        generator_film_min_learning_rate=float(
                            training["generator_film_min_learning_rate"]
                        ),
                    )
                rows.append(spec)
    expected = len(SEEDS) * len(FOLDS) * len(arm_universe)
    if len(rows) != expected or len({str(row["job_id"]) for row in rows}) != expected:
        raise AssertionError(f"Internal stage expected {expected} unique jobs")
    expected_gpu = expected // 2
    counts = {gpu: sum(int(row["gpu_id"]) == gpu for row in rows) for gpu in (0, 1)}
    if counts != {0: expected_gpu, 1: expected_gpu}:
        raise ValueError(f"Internal stage GPU imbalance: {counts}")
    if use_graft:
        for seed in SEEDS:
            for fold in FOLDS:
                selected = [
                    row
                    for row in rows
                    if int(row["seed"]) == seed and str(row["fold"]) == fold
                ]
                if len({str(row["graft_state_sha256"]) for row in selected}) != 1:
                    raise ValueError(f"Graft state differs within block: {seed}/{fold}")
    return rows


def _materialize_stage_overlays(
    config: Mapping[str, Any], root: Path, arms: Sequence[str]
) -> Path:
    from wgan_option.utils.news_first_experiment_core import (
        build_pair_text_overlay_manifests,
        write_pair_text_overlay_manifest,
    )

    output_dir = root / "inputs/pair_text_overlays"
    build_pair_text_overlay_manifests(
        config=config,
        pair_universe_path=root / "inputs/pair_universes.csv",
        output_dir=output_dir,
    )
    canonical_sources = {
        PARENT_ARM: "parent_current_only",
        PURE_CONTINUATION_ARM: "parent_current_only",
        "film_zero_text": "parent_current_only",
        "film_lp_matched": "lp_matched",
        "film_lp_shuffle": "lp_shuffle",
        "film_bow": "bow",
        "film_sentiment": "sentiment",
    }
    for fold in FOLDS:
        directory = output_dir / "tolerance_05m" / fold
        source_payloads = {
            arm: _read_json(directory / f"{canonical_sources[arm]}.json")
            for arm in arms
        }
        for arm, payload in source_payloads.items():
            write_pair_text_overlay_manifest(
                direct._overlay_path(root, fold, arm),
                mode=_stage_overlay_mode(arm),
                namespace=f"text_effectiveness/{fold}/{arm}/v1",
                records=list(payload["records"]),
                transform={
                    **dict(payload.get("transform") or {}),
                    "text_effectiveness_arm": arm,
                    "source_overlay_name": canonical_sources[arm],
                },
            )
    expected = {direct._overlay_path(root, fold, arm) for fold in FOLDS for arm in arms}
    for path in output_dir.glob("tolerance_*/*/*.json"):
        if path.resolve() not in expected:
            path.unlink()
    paths = sorted(output_dir.glob("tolerance_05m/*/*.json"))
    if {path.resolve() for path in paths} != expected:
        raise ValueError("Internal stage overlay universe drift")
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _read_json(path)
        unsigned = {
            key: value for key, value in payload.items() if key != "profile_sha256"
        }
        if direct.payload_sha256(unsigned) != payload.get("profile_sha256"):
            raise ValueError(f"Overlay self-hash drift: {path}")
        if payload.get("mode") != _stage_overlay_mode(path.stem):
            raise ValueError(f"Overlay mode drift: {path}")
        records = list(payload.get("records") or [])
        if not records:
            raise ValueError(f"Empty overlay: {path}")
        if path.stem == "film_lp_shuffle" and any(
            str(row["pair_id"]) == str(row.get("donor_pair_id", "")) for row in records
        ):
            raise ValueError("Training shuffle contains a fixed point")
        rows.append(direct.manifest_row(f"pair_overlay:{path.relative_to(root)}", path))
    return direct._write_hash_manifest(
        root / "inputs/pair_text_overlay_hashes.csv", rows
    )


def _stage_training_payload(
    config: Mapping[str, Any],
    root: Path,
    spec: Mapping[str, Any],
    *,
    num_epochs: int | None = None,
    slots_per_gpu: int | None = None,
) -> dict[str, Any]:
    del root, slots_per_gpu
    training = _mapping(config["training"], "training")
    payload: dict[str, Any] = {
        "generator_conditioning_mode": config["model"]["generator_conditioning_mode"],
        "critic_conditioning_mode": CRITIC_MODE,
        "generator_optimizer_profile": spec["generator_optimizer_profile"],
        "generator_learning_rate": float(training["generator_learning_rate"]),
        "discriminator_learning_rate": float(training["discriminator_learning_rate"]),
        "reduce_lr_min_lr": float(training["scheduler_min_lr"]),
        "num_epochs": int(num_epochs or training["num_epochs"]),
        "early_stopping_min_epochs": int(training["early_stopping_min_epochs"]),
        "early_stopping_patience": int(training["early_stopping_patience"]),
        "validation_mc_samples": 16,
        "evaluate_initial_checkpoint": True,
        "news_first_materialize_validation_loader": True,
        "news_first_materialize_test_loader": False,
        "news_first_pair_text_overlay_mode": _stage_overlay_mode(str(spec["arm"])),
        "news_first_full_training_state_mode": "save_dynamic_v1",
        "news_first_refit_mode": "none",
        "news_first_graft_state_path": str(spec.get("graft_state_path", "")),
        "news_first_graft_state_sha256": str(spec.get("graft_state_sha256", "")),
        "news_first_validation_snapshot_epochs": list(
            spec.get("validation_snapshot_epochs", ())
        ),
        "use_reduce_lr_on_plateau": True,
        "use_early_stopping": True,
        "seed": int(spec["seed"]),
    }
    if spec["generator_optimizer_profile"] == "film_unet_split_lr_v1":
        for name in (
            "generator_text_learning_rate",
            "generator_film_learning_rate",
            "generator_text_min_learning_rate",
            "generator_film_min_learning_rate",
        ):
            payload[name] = float(spec[name])
    return payload


def _stage_artifact_paths(
    job: Mapping[str, Any], run_dir: Path
) -> list[dict[str, Any]]:
    rows = list(_BASE_DIRECT_ARTIFACT_PATHS(job, run_dir))
    for epoch in job.get("validation_snapshot_epochs", []):
        for network in ("generator", "discriminator"):
            path = (
                run_dir
                / "checkpoints"
                / f"{network}_validation_epoch_{int(epoch):04d}.pt"
            )
            if not path.is_file():
                raise RuntimeError(f"Missing validation snapshot: {path}")
            rows.append(
                direct.manifest_row(
                    f"{network}_validation_epoch_{int(epoch):04d}", path
                )
            )
    if job.get("graft_state_path"):
        rows.append(
            direct.manifest_row("graft_initialization_state", job["graft_state_path"])
        )
    direct.verify_manifest_rows(rows, require_unique_roles=True)
    return rows


def _stage_freeze_initial_state_fairness(
    root: Path, jobs: Sequence[Mapping[str, Any]]
) -> Path:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        status_payload = _read_json(direct._status_path(root, str(job["job_id"])))
        initial_g = direct._artifact(status_payload, "generator_initial_epoch0")
        initial_d = direct._artifact(status_payload, "discriminator_initial_epoch0")
        # Fresh random parents inherit the historical insertion-order digest
        # emitted by the shared direct-training planner.  Grafted branches use
        # this experiment's canonical sorted-key digest stored in the graft
        # allowlist.  Compare each checkpoint with the algorithm that produced
        # its frozen expected hash.
        checkpoint_digest = (
            _canonical_checkpoint_state_sha256
            if str(job.get("graft_state_path", "")).strip()
            else direct._checkpoint_state_sha256
        )
        observed_g = checkpoint_digest(initial_g["path"])
        observed_d = checkpoint_digest(initial_d["path"])
        if observed_g != str(job["initial_generator_state_sha256"]):
            raise ValueError(f"Generator epoch-0 drift: {job['job_id']}")
        if observed_d != str(job["initial_critic_state_sha256"]):
            raise ValueError(f"Critic epoch-0 drift: {job['job_id']}")
        rows.append(
            {
                "job_id": str(job["job_id"]),
                "seed": int(job["seed"]),
                "fold": str(job["fold"]),
                "arm": str(job["arm"]),
                "generator_initial_state_sha256": observed_g,
                "critic_initial_state_sha256": observed_d,
                "graft_state_sha256": str(job.get("graft_state_sha256", "")),
            }
        )
    for seed in SEEDS:
        for fold in FOLDS:
            selected = [
                row for row in rows if row["seed"] == seed and row["fold"] == fold
            ]
            if len({row["generator_initial_state_sha256"] for row in selected}) != 1:
                raise ValueError(f"Generator initial states differ: {seed}/{fold}")
            if len({row["critic_initial_state_sha256"] for row in selected}) != 1:
                raise ValueError(f"Critic initial states differ: {seed}/{fold}")
    payload = {
        "schema_version": 1,
        "kind": "text_effectiveness_stage_initial_state_fairness_v1",
        "job_count": len(rows),
        "common_within_seed_fold": True,
        "fresh_optimizer_and_scheduler": True,
        "rows": sorted(rows, key=lambda row: row["job_id"]),
    }
    payload["payload_sha256"] = direct.payload_sha256(payload)
    return _write_json(direct._initial_state_fairness_path(root), payload)


def _validate_planned_initial_state_fairness(
    jobs: Sequence[Mapping[str, Any]],
    *,
    use_graft: bool,
) -> None:
    """Validate fresh seed-level or grafted seed/fold-level initialization."""
    if any(
        bool(str(job.get("graft_state_path", "")).strip()) != use_graft for job in jobs
    ):
        raise ValueError("Planned epoch-0 graft-state mode drift")
    grouped_g: dict[object, set[str]] = {}
    grouped_d: dict[object, set[str]] = {}
    for job in jobs:
        block: object = (
            (int(job["seed"]), str(job["fold"])) if use_graft else int(job["seed"])
        )
        grouped_g.setdefault(block, set()).add(
            str(job["initial_generator_state_sha256"])
        )
        grouped_d.setdefault(block, set()).add(str(job["initial_critic_state_sha256"]))
    expected_blocks: set[object] = (
        {(seed, fold) for seed in SEEDS for fold in FOLDS} if use_graft else set(SEEDS)
    )
    if set(grouped_g) != expected_blocks or set(grouped_d) != expected_blocks:
        raise ValueError("Planned epoch-0 fairness block universe drift")
    if any(len(values) != 1 for values in (*grouped_g.values(), *grouped_d.values())):
        unit = "seed/fold block" if use_graft else "seed"
        raise ValueError(f"Planned epoch-0 states are not common within each {unit}")
    if len({next(iter(values)) for values in grouped_g.values()}) != len(
        expected_blocks
    ) or len({next(iter(values)) for values in grouped_d.values()}) != len(
        expected_blocks
    ):
        unit = "seed/fold blocks" if use_graft else "seeds"
        raise ValueError(f"Planned epoch-0 states are not distinct across {unit}")


def _validate_stage_root(
    root_or_path: str | Path,
    *,
    stage_name: str,
    arms: Sequence[str],
    generator_mode: str,
    expected_jobs: int,
    use_graft: bool,
    verify_large_inputs: bool = True,
) -> dict[str, Any]:
    root = Path(root_or_path).resolve()
    registry = direct.read_registry(root)
    if (
        registry.get("experiment_kind") != _stage_experiment_kind(stage_name)
        or registry.get("interpretation") != INTERPRETATION
    ):
        raise ValueError("Internal stage registry identity drift")
    jobs = list(registry.get("jobs") or [])
    if len(jobs) != expected_jobs or registry.get(
        "jobs_sha256"
    ) != direct.payload_sha256(jobs):
        raise ValueError("Internal stage registry count/hash drift")
    if len({str(job["job_id"]) for job in jobs}) != expected_jobs:
        raise ValueError("Duplicate internal stage job IDs")
    expected_cells = {
        (seed, fold, arm) for seed in SEEDS for fold in FOLDS for arm in arms
    }
    observed_cells = {
        (int(job["seed"]), str(job["fold"]), str(job["arm"])) for job in jobs
    }
    if observed_cells != expected_cells:
        raise ValueError("Internal stage seed/fold/arm universe drift")
    _validate_planned_initial_state_fairness(jobs, use_graft=use_graft)
    config = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Internal stage resolved config must be a mapping")
    _validate_stage_config(
        config,
        stage_name=stage_name,
        arms=arms,
        generator_mode=generator_mode,
        expected_jobs=expected_jobs,
    )
    contract_module = pure if generator_mode == PURE_MODE else direct
    if _read_json(root / "grid_contract.json") != contract_module.grid_contract(config):
        raise ValueError("Internal stage grid contract drift")
    if _read_json(root / "model_contract.json") != contract_module.model_contract(
        config
    ):
        raise ValueError("Internal stage model contract drift")
    source_rows = [
        direct.manifest_row(role, path) for role, path in _stage_source_paths(config)
    ]
    if verify_large_inputs:
        direct._assert_hash_manifest(root / "source_hashes.csv", source_rows)
    code_paths = {resolve_path(value) for value in _stage_code_paths()}
    code_paths.update(path.resolve() for path in (REPO_ROOT / "src").rglob("*.py"))
    code_paths.update(
        path.resolve() for path in (REPO_ROOT / "scripts/rq123").rglob("*.py")
    )
    code_rows = [
        direct.manifest_row(f"code:{path.relative_to(REPO_ROOT)}", path)
        for path in sorted(
            code_paths, key=lambda value: value.relative_to(REPO_ROOT).as_posix()
        )
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
    for job in jobs:
        if core._job_spec_sha(job) != job.get("job_spec_sha256"):
            raise ValueError(f"Internal job spec drift: {job['job_id']}")
        for path_key, sha_key in (
            ("training_config_path", "training_config_sha256"),
            ("full_state_contract_path", "full_state_contract_sha256"),
            ("overlay_path", "overlay_sha256"),
        ):
            direct._verify_frozen_file(job[path_key], job[sha_key])
        if job.get("graft_state_path"):
            direct._verify_frozen_file(
                job["graft_state_path"], job["graft_state_sha256"]
            )
        if verify_large_inputs:
            direct._verify_frozen_file(job["dataset_path"], job["dataset_sha256"])
            direct._verify_frozen_file(job["support_path"], job["support_sha256"])
        training_payload = yaml.safe_load(
            Path(str(job["training_config_path"])).read_text(encoding="utf-8")
        )
        if not isinstance(training_payload, Mapping) or not bool(
            training_payload.get("evaluate_initial_checkpoint")
        ):
            raise ValueError(
                f"Internal direct job disabled epoch-0 evaluation: {job['job_id']}"
            )
        status_payload = _read_json(direct._status_path(root, str(job["job_id"])))
        if (
            status_payload.get("job_id") != job["job_id"]
            or status_payload.get("job_spec_sha256") != job["job_spec_sha256"]
        ):
            raise ValueError(f"Internal job status lineage drift: {job['job_id']}")
        if status_payload.get("status") == "completed" and not direct._completed_valid(
            job, status_payload
        ):
            raise ValueError(f"Completed internal artifact drift: {job['job_id']}")
    if registry.get("test_data_opened") and not registry.get("evaluation_frozen"):
        raise ValueError("Internal test data opened before evaluation freeze")
    return dict(config)


def _validate_public_prepare_base(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deeply validate the public artifacts committed before stage preparation."""

    config_path = root / "resolved_config.yaml"
    registry_path = _registry_path(root)
    prepare_path = root / "registry/prepare_manifest.json"
    for required in (config_path, registry_path, prepare_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"Prepared experiment artifact is missing: {required}"
            )
    config = _load_frozen_config(config_path)
    validate_config(config, verify_source_evidence=True)
    if root != resolve_path(config["experiment"]["output_root"]):
        raise ValueError("Prepared root differs from the frozen formal output root")
    registry = _read_registry(root)
    _validate_registry_state(config, registry)

    source_path = root / "registry/source_hashes.csv"
    direct._assert_hash_manifest(source_path, _source_manifest_rows(config))
    prepare_manifest = _read_json(prepare_path)
    expected_initial_jobs_sha = direct.payload_sha256(planned_job_specs(config))
    if (
        prepare_manifest.get("kind") != "text_effectiveness_prepare_manifest_v1"
        or prepare_manifest.get("config_sha256") != sha256_file(config_path)
        or prepare_manifest.get("source_manifest_sha256") != sha256_file(source_path)
        or prepare_manifest.get("jobs_sha256") != expected_initial_jobs_sha
        or int(prepare_manifest.get("formal_workers_per_gpu", -1))
        != int(registry.get("formal_workers_per_gpu", -2))
        or int(prepare_manifest.get("test_loader_count", -1)) != 0
        or int(prepare_manifest.get("test_prediction_count", -1)) != 0
    ):
        raise ValueError("Formal prepare manifest drift")
    return config, registry


def validate_partial_root(
    root_or_path: str | Path,
    *,
    verify_stage_roots: bool = True,
    verify_large_inputs: bool = True,
) -> dict[str, Any]:
    """Validate an owned formal root, including the pre-backbone prepare boundary."""

    root = _root(root_or_path)
    config, registry = _validate_public_prepare_base(root)
    roots = _stage_roots(root)
    if not roots["backbones"].exists():
        if any(path.exists() for key, path in roots.items() if key != "backbones"):
            raise ValueError("Continuation stage exists before the backbone stage")
        if (
            registry.get("status") != "prepared"
            or registry.get("jobs") != planned_job_specs(config)
            or any(
                bool(registry.get(flag))
                for flag in (
                    "parents_frozen",
                    "grafts_frozen",
                    "continuations_prepared",
                    "evaluation_frozen",
                    "test_data_opened",
                    "validation_trajectories_frozen",
                    "standard_predictions_frozen",
                    "interventions_frozen",
                    "analysis_complete",
                    "terminal_complete",
                )
            )
        ):
            raise ValueError("Pre-backbone formal prepare boundary drift")
        return config
    return validate_root(
        root,
        verify_stage_roots=verify_stage_roots,
        verify_large_inputs=verify_large_inputs,
    )


def validate_root(
    root_or_path: str | Path,
    *,
    verify_stage_roots: bool = True,
    verify_large_inputs: bool = True,
) -> dict[str, Any]:
    """Validate a prepared public root while permitting partial continuations."""

    root = _root(root_or_path)
    config, registry = _validate_public_prepare_base(root)

    roots = _stage_roots(root)
    continuation_exists = any(
        stage_root.exists() for key, stage_root in roots.items() if key != "backbones"
    )
    if continuation_exists and not registry.get("grafts_frozen"):
        raise ValueError("Continuation stage exists before the graft freeze")
    if registry.get("continuations_prepared") and any(
        not roots[key].is_dir() for key in ("pure_continuation", "film_continuations")
    ):
        raise ValueError("Frozen continuation prepare is missing a stage root")
    if verify_stage_roots:
        if not roots["backbones"].is_dir():
            raise ValueError("Prepared experiment is missing the backbone stage")
        _validate_stage_root(
            roots["backbones"],
            stage_name="backbones",
            arms=(PARENT_ARM,),
            generator_mode=PURE_MODE,
            expected_jobs=EXPECTED_PARENT_JOBS,
            use_graft=False,
            verify_large_inputs=verify_large_inputs,
        )
        if roots["pure_continuation"].exists():
            _validate_stage_root(
                roots["pure_continuation"],
                stage_name="pure_continuation",
                arms=(PURE_CONTINUATION_ARM,),
                generator_mode=PURE_MODE,
                expected_jobs=EXPECTED_PARENT_JOBS,
                use_graft=True,
                verify_large_inputs=verify_large_inputs,
            )
        if roots["film_continuations"].exists():
            _validate_stage_root(
                roots["film_continuations"],
                stage_name="film_continuations",
                arms=FILM_ARMS,
                generator_mode=FILM_MODE,
                expected_jobs=200,
                use_graft=True,
                verify_large_inputs=verify_large_inputs,
            )

    if registry.get("parents_frozen"):
        _validate_parent_allowlist(root, registry, verify_artifacts=verify_large_inputs)
    if registry.get("grafts_frozen"):
        entries = _validate_graft_allowlist(
            root, registry, verify_artifacts=verify_large_inputs
        )
        by_target = {
            (int(row["seed"]), str(row["fold"]), str(row["target_generator_mode"])): row
            for row in entries
        }
        for job in registry["jobs"]:
            if job["stage"] != "continuation":
                continue
            target_mode = (
                PURE_MODE if job["arm"] == PURE_CONTINUATION_ARM else FILM_MODE
            )
            entry = by_target[(int(job["seed"]), str(job["fold"]), target_mode)]
            if any(
                str(job[field]) != str(entry[entry_field])
                for field, entry_field in (
                    ("graft_state_path", "artifact_path"),
                    ("graft_state_sha256", "artifact_sha256"),
                    ("graft_manifest_path", "manifest_path"),
                    ("graft_manifest_sha256", "manifest_sha256"),
                    ("initial_generator_state_sha256", "generator_state_sha256"),
                    ("initial_critic_state_sha256", "discriminator_state_sha256"),
                )
            ):
                raise ValueError(f"Continuation/graft registry drift: {job['job_id']}")

    if registry.get("evaluation_frozen"):
        from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_prediction import (
            require_frozen_test_access,
        )

        require_frozen_test_access(
            registry,
            expected_training_jobs=EXPECTED_TRAINING_JOBS,
            verify_allowlist_artifacts=verify_large_inputs,
        )
    if registry.get("standard_predictions_frozen"):
        manifest_path = _verify_registry_file(
            registry,
            "standard_prediction_manifest_path",
            "standard_prediction_manifest_sha256",
        )
        pair_path = _verify_registry_file(
            registry,
            "standard_pair_metrics_path",
            "standard_pair_metrics_sha256",
        )
        manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        if len(manifest) != EXPECTED_STANDARD_PREDICTIONS:
            raise ValueError("Frozen standard prediction manifest count drift")
        if (
            verify_large_inputs
            and len(pd.read_csv(pair_path)) != EXPECTED_STANDARD_PAIR_ROWS
        ):
            raise ValueError("Frozen standard prediction pair-row count drift")
    if registry.get("interventions_frozen"):
        manifest_path = _verify_registry_file(
            registry,
            "intervention_prediction_manifest_path",
            "intervention_prediction_manifest_sha256",
        )
        pair_path = _verify_registry_file(
            registry,
            "intervention_pair_metrics_path",
            "intervention_pair_metrics_sha256",
        )
        manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        if len(manifest) != EXPECTED_INTERVENTION_PREDICTIONS:
            raise ValueError("Frozen intervention prediction manifest count drift")
        if (
            verify_large_inputs
            and len(pd.read_csv(pair_path)) != EXPECTED_INTERVENTION_PAIR_ROWS
        ):
            raise ValueError("Frozen intervention prediction pair-row count drift")
    if registry.get("validation_trajectories_frozen"):
        inventory_path = _verify_registry_file(
            registry,
            "validation_trajectory_inventory_path",
            "validation_trajectory_inventory_sha256",
        )
        units_path = _verify_registry_file(
            registry,
            "validation_trajectory_units_path",
            "validation_trajectory_units_sha256",
        )
        pair_path = _verify_registry_file(
            registry,
            "validation_trajectory_pair_metrics_path",
            "validation_trajectory_pair_metrics_sha256",
        )
        if len(pd.read_csv(inventory_path)) != EXPECTED_VALIDATION_TRAJECTORY_UNITS:
            raise ValueError("Frozen validation trajectory inventory count drift")
        if len(pd.read_csv(units_path)) != EXPECTED_VALIDATION_TRAJECTORY_UNITS:
            raise ValueError("Frozen validation trajectory unit count drift")
        if verify_large_inputs and len(pd.read_csv(pair_path)) != (
            EXPECTED_VALIDATION_TRAJECTORY_PAIR_ROWS
        ):
            raise ValueError("Frozen validation trajectory pair-row count drift")
    return config


_PATCH_ATTRIBUTE_MISSING = object()


@contextmanager
def _patched(module: object, replacements: Mapping[str, object]) -> Iterator[None]:
    """Temporarily patch both existing and branch-local module attributes."""

    originals = {
        name: getattr(module, name, _PATCH_ATTRIBUTE_MISSING) for name in replacements
    }
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            if value is _PATCH_ATTRIBUTE_MISSING:
                delattr(module, name)
            else:
                setattr(module, name, value)


@contextmanager
def _pure_stage_profile(
    *,
    main_root: Path,
    stage_name: str,
    arm: str,
    use_graft: bool,
) -> Iterator[None]:
    expected_jobs = 40

    def validate(config: Mapping[str, Any]) -> None:
        _validate_stage_config(
            config,
            stage_name=stage_name,
            arms=(arm,),
            generator_mode=PURE_MODE,
            expected_jobs=expected_jobs,
        )

    def load_frozen(path: Path) -> dict[str, Any]:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        validate(payload)
        return dict(payload)

    def specs(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        if config is None:
            raise ValueError("Internal stage planning requires a frozen config")
        return _planned_stage_specs(
            config,
            arm_universe=(arm,),
            generator_mode=PURE_MODE,
            main_root=main_root,
            use_graft=use_graft,
        )

    def validate_root_local(
        path: str | Path, *, verify_large_inputs: bool = True
    ) -> dict[str, Any]:
        return _validate_stage_root(
            path,
            stage_name=stage_name,
            arms=(arm,),
            generator_mode=PURE_MODE,
            expected_jobs=expected_jobs,
            use_graft=use_graft,
            verify_large_inputs=verify_large_inputs,
        )

    replacements = {
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "DEFAULT_OUTPUT_DIR": str(main_root),
        "EXPERIMENT_KIND": _stage_experiment_kind(stage_name),
        "INTERPRETATION": INTERPRETATION,
        "DIRECT_ARMS": (arm,),
        "NO_TEXT_ARM": arm,
        "EXPECTED_TRAINING_JOBS": expected_jobs,
        "EXPECTED_PREDICTION_CELLS": expected_jobs,
        "EXPECTED_PAIR_METRIC_ROWS": 5_000,
        "WORKER_MODULE": WORKER_MODULE,
        "REGISTRY_KIND": f"{stage_name}_task_registry_v1",
        "SOURCE_CODE_RELATIVE_PATHS": _stage_code_paths(),
        "load_config": lambda path=DEFAULT_CONFIG: load_frozen(Path(path)),
        "_load_frozen_config": load_frozen,
        "validate_config": validate,
        "_overlay_mode": _stage_overlay_mode,
        "planned_specs": specs,
        "validate_gpu_balance": lambda rows: None,
        "_source_paths": _stage_source_paths,
        "_training_payload": _stage_training_payload,
        "_materialize_pair_text_overlays": lambda config,
        root: _materialize_stage_overlays(config, root, (arm,)),
        "validate_root": validate_root_local,
        "_freeze_initial_state_fairness": _stage_freeze_initial_state_fairness,
    }
    direct_replacements = {
        "_direct_artifact_paths": _stage_artifact_paths,
        # Grafted continuation checkpoints are bound to the canonical
        # sorted-key digest used by the graft allowlist.  Fresh parents retain
        # the historical insertion-order digest for compatibility.
        "_checkpoint_state_sha256": (
            _canonical_checkpoint_state_sha256
            if use_graft
            else _BASE_DIRECT_CHECKPOINT_STATE_SHA256
        ),
    }
    with (
        _patched(pure, replacements),
        _patched(direct, direct_replacements),
        _patched(core, {"_overlay_mode": _stage_overlay_mode}),
    ):
        yield


@contextmanager
def _film_stage_profile(*, main_root: Path, stage_name: str) -> Iterator[None]:
    arms = FILM_ARMS
    expected_jobs = 200

    def validate(config: Mapping[str, Any]) -> None:
        _validate_stage_config(
            config,
            stage_name=stage_name,
            arms=arms,
            generator_mode=FILM_MODE,
            expected_jobs=expected_jobs,
        )

    def load_frozen(path: Path) -> dict[str, Any]:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        validate(payload)
        return dict(payload)

    def specs(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        if config is None:
            raise ValueError("Internal stage planning requires a frozen config")
        return _planned_stage_specs(
            config,
            arm_universe=arms,
            generator_mode=FILM_MODE,
            main_root=main_root,
            use_graft=True,
        )

    def validate_root_local(
        path: str | Path, *, verify_large_inputs: bool = True
    ) -> dict[str, Any]:
        return _validate_stage_root(
            path,
            stage_name=stage_name,
            arms=arms,
            generator_mode=FILM_MODE,
            expected_jobs=expected_jobs,
            use_graft=True,
            verify_large_inputs=verify_large_inputs,
        )

    replacements = {
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "DEFAULT_OUTPUT_DIR": str(main_root),
        "EXPERIMENT_KIND": _stage_experiment_kind(stage_name),
        "INTERPRETATION": INTERPRETATION,
        "DIRECT_ARMS": arms,
        "NO_TEXT_ARM": "film_zero_text",
        "EXPECTED_TRAINING_JOBS": expected_jobs,
        "EXPECTED_PREDICTION_CELLS": expected_jobs,
        "EXPECTED_PAIR_METRIC_ROWS": 25_000,
        "WORKER_MODULE": WORKER_MODULE,
        "REGISTRY_KIND": f"{stage_name}_task_registry_v1",
        "SOURCE_CODE_RELATIVE_PATHS": _stage_code_paths(),
        "load_config": lambda path=DEFAULT_CONFIG: load_frozen(Path(path)),
        "_load_frozen_config": load_frozen,
        "validate_config": validate,
        "_overlay_mode": _stage_overlay_mode,
        "planned_specs": specs,
        "validate_gpu_balance": lambda rows: None,
        "_source_paths": _stage_source_paths,
        "_training_payload": _stage_training_payload,
        "_materialize_pair_text_overlays": lambda config,
        root: _materialize_stage_overlays(config, root, arms),
        "validate_root": validate_root_local,
        "_freeze_initial_state_fairness": _stage_freeze_initial_state_fairness,
    }
    direct_replacements = {
        "_direct_artifact_paths": _stage_artifact_paths,
        "_checkpoint_state_sha256": _canonical_checkpoint_state_sha256,
    }
    with (
        _patched(film_text, replacements),
        _patched(film_text.multiseed, {"validate_root": validate_root_local}),
        _patched(direct, direct_replacements),
        _patched(core, {"_overlay_mode": _stage_overlay_mode}),
    ):
        yield


def _workers_from_benchmark(root: Path) -> int:
    path = _control_root(root) / "benchmark_result.json"
    if not path.is_file():
        raise RuntimeError("Formal prepare requires a completed benchmark")
    payload = _read_json(path)
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if (
        payload.get("kind") != BENCHMARK_KIND
        or payload.get("status") != "passed"
        or direct.payload_sha256(unsigned) != payload.get("payload_sha256")
        or payload.get("source_config_sha256")
        != sha256_file(resolve_path(DEFAULT_CONFIG))
    ):
        raise ValueError("Benchmark result drift")
    manifest = direct._verify_frozen_file(
        payload["benchmark_manifest_path"], payload["benchmark_manifest_sha256"]
    )
    benchmark_manifest = _read_json(manifest)
    benchmark_unsigned = {
        key: value
        for key, value in benchmark_manifest.items()
        if key != "payload_sha256"
    }
    workers = int(payload["selected_workers_per_gpu"])
    if (
        benchmark_manifest.get("kind")
        != "pure_cnn_parent_film_text_effect_benchmark_root_v1"
        or benchmark_manifest.get("status") != "passed"
        or direct.payload_sha256(benchmark_unsigned)
        != benchmark_manifest.get("payload_sha256")
        or int(benchmark_manifest.get("workers_per_gpu", -1)) != workers
        or workers
        not in {
            int(load_config(DEFAULT_CONFIG)["runtime"]["benchmark_workers_per_gpu"]),
            int(load_config(DEFAULT_CONFIG)["runtime"]["fallback_workers_per_gpu"]),
        }
    ):
        raise ValueError("Benchmark manifest/concurrency drift")
    projected = int(payload.get("projected_formal_bytes_with_safety", -1))
    minimum_remaining = int(payload.get("minimum_remaining_bytes", -1))
    if projected <= 0 or minimum_remaining != 30 * 1024**3:
        raise ValueError("Benchmark disk projection contract drift")
    if shutil.disk_usage(root.parent).free - projected < minimum_remaining:
        raise RuntimeError("Formal disk gate no longer leaves the frozen 30 GiB margin")
    return workers


def _prepare_backbone_stage(
    config: Mapping[str, Any],
    root: Path,
    *,
    workers: int,
    num_epochs: int,
    resume: bool,
) -> Path:
    stage_root = _stage_roots(root)["backbones"]
    stage_config = _stage_config(
        config,
        stage_name="backbones",
        arms=(PARENT_ARM,),
        generator_mode=PURE_MODE,
        expected_counts={
            "training_jobs": 40,
            "pair_rows": 5_000,
            **PURE_COUNTS,
        },
        workers_per_gpu=workers,
        output_root=stage_root,
        num_epochs=num_epochs,
    )
    stage_config["training"]["protocol"] = "pure_cnn_fresh_dynamic_validation_v1"
    stage_config["training"]["initial_state_contract"] = (
        "canonical_seeded_random_initialization_v1"
    )
    with _pure_stage_profile(
        main_root=root,
        stage_name="backbones",
        arm=PARENT_ARM,
        use_graft=False,
    ):
        return pure.prepare(
            stage_config,
            stage_root,
            resume=resume,
            workers_per_gpu=workers,
            num_epochs=num_epochs,
            root_mode="internal_stage",
        )


def _prepare_continuation_stages(
    config: Mapping[str, Any],
    root: Path,
    *,
    workers: int,
    num_epochs: int,
    resume: bool,
) -> tuple[Path, Path]:
    roots = _stage_roots(root)
    pure_root = roots["pure_continuation"]
    film_root = roots["film_continuations"]
    pure_config = _stage_config(
        config,
        stage_name="pure_continuation",
        arms=(PURE_CONTINUATION_ARM,),
        generator_mode=PURE_MODE,
        expected_counts={
            "training_jobs": 40,
            "pair_rows": 5_000,
            **PURE_COUNTS,
        },
        workers_per_gpu=workers,
        output_root=pure_root,
        num_epochs=num_epochs,
    )
    film_config = _stage_config(
        config,
        stage_name="film_continuations",
        arms=FILM_ARMS,
        generator_mode=FILM_MODE,
        expected_counts={
            "training_jobs": 200,
            "pair_rows": 25_000,
            **FILM_COUNTS,
        },
        workers_per_gpu=workers,
        output_root=film_root,
        num_epochs=num_epochs,
    )
    with _pure_stage_profile(
        main_root=root,
        stage_name="pure_continuation",
        arm=PURE_CONTINUATION_ARM,
        use_graft=True,
    ):
        pure.prepare(
            pure_config,
            pure_root,
            resume=resume,
            workers_per_gpu=workers,
            num_epochs=num_epochs,
            root_mode="internal_stage",
        )
    with _film_stage_profile(main_root=root, stage_name="film_continuations"):
        film_text.prepare(
            film_config,
            film_root,
            resume=resume,
            workers_per_gpu=workers,
            num_epochs=num_epochs,
            root_mode="internal_stage",
        )
    return pure_root, film_root


def _source_manifest_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        direct.manifest_row("formal_config", config["source_config_path"]),
        direct.manifest_row(
            "dataset_manifest",
            resolve_path(config["data"]["root"]) / "dataset_output_sha256.txt",
        ),
        direct.manifest_row(
            "dataset_validation",
            resolve_path(config["data"]["root"]) / "validation_summary.json",
        ),
        direct.manifest_row(
            "news_master", resolve_path(config["data"]["news_master_path"])
        ),
        direct.manifest_row(
            "sentiment_workbook",
            resolve_path(config["data"]["sentiment_workbook_path"]),
        ),
    ]
    for label, raw in _mapping(config["source_evidence"], "source_evidence").items():
        evidence = _mapping(raw, label)
        for key in (
            "source_config",
            "resolved_config",
            "qa",
            "analysis_manifest",
            "pair_metrics",
            "output_manifest",
        ):
            rows.append(
                direct.manifest_row(
                    f"historical:{label}:{key}", resolve_path(evidence[f"{key}_path"])
                )
            )
    return rows


def _source_manifest(config: Mapping[str, Any], root: Path) -> Path:
    return direct._write_hash_manifest(
        root / "registry/source_hashes.csv", _source_manifest_rows(config)
    )


def _prepare_formal(
    config: Mapping[str, Any],
    root: Path,
    *,
    workers: int,
    resume: bool,
    num_epochs: int = 240,
) -> Path:
    if root.exists():
        if not resume:
            raise FileExistsError(root)
        frozen = validate_partial_root(root, verify_stage_roots=True)
        if direct.payload_sha256(frozen) != direct.payload_sha256(config):
            raise ValueError("Resume config differs from the frozen formal config")
        registry = _read_registry(root)
        if int(registry.get("formal_workers_per_gpu", -1)) != int(workers):
            raise ValueError("Resume benchmark concurrency differs from formal prepare")
        if not _stage_roots(root)["backbones"].exists():
            _prepare_backbone_stage(
                config,
                root,
                workers=workers,
                num_epochs=num_epochs,
                resume=False,
            )
        validate_root(root, verify_stage_roots=True)
        return root
    if root != resolve_path(config["experiment"]["output_root"]):
        raise ValueError("Formal root differs from the frozen config")
    root.mkdir(parents=True)
    for relative in (
        "registry",
        "registry/graft_manifests",
        "registry/graft_states",
        "evaluation",
        "analysis",
        "report",
        "logs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    direct.write_yaml(root / "resolved_config.yaml", config)
    jobs = planned_job_specs(config)
    registry = {
        "schema_version": 1,
        "kind": REGISTRY_KIND,
        "experiment_kind": EXPERIMENT_KIND,
        "interpretation": INTERPRETATION,
        "status": "prepared",
        "created_at_utc": direct.utc_now(),
        "formal_workers_per_gpu": int(workers),
        "jobs": jobs,
        "parents_frozen": False,
        "grafts_frozen": False,
        "continuations_prepared": False,
        "evaluation_frozen": False,
        "test_data_opened": False,
        "validation_trajectories_frozen": False,
        "standard_predictions_frozen": False,
        "interventions_frozen": False,
        "analysis_complete": False,
        "terminal_complete": False,
    }
    _write_registry(root, registry)
    _source_manifest(config, root)
    _write_json(
        root / "registry/prepare_manifest.json",
        {
            "schema_version": 1,
            "kind": "text_effectiveness_prepare_manifest_v1",
            "config_sha256": sha256_file(root / "resolved_config.yaml"),
            "source_manifest_sha256": sha256_file(root / "registry/source_hashes.csv"),
            "jobs_sha256": _read_registry(root)["jobs_sha256"],
            "formal_workers_per_gpu": workers,
            "test_loader_count": 0,
            "test_prediction_count": 0,
            "created_at_utc": direct.utc_now(),
        },
    )
    _prepare_backbone_stage(
        config,
        root,
        workers=workers,
        num_epochs=num_epochs,
        resume=False,
    )
    validate_root(root, verify_stage_roots=True)
    return root


def prepare(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    validate_config(config, verify_source_evidence=True)
    root = _root(output_dir)
    workers = _workers_from_benchmark(root)
    return _prepare_formal(config, root, workers=workers, resume=resume)


def launch_backbones(output_dir: str | Path, *, resume: bool = False) -> Path:
    root = _root(output_dir)
    validate_root(root, verify_stage_roots=True)
    registry = _read_registry(root)
    if registry.get("parents_frozen"):
        if resume:
            return _stage_roots(root)["backbones"]
        raise RuntimeError("Backbone checkpoints are already frozen")
    stage_root = _stage_roots(root)["backbones"]
    with _pure_stage_profile(
        main_root=root,
        stage_name="backbones",
        arm=PARENT_ARM,
        use_graft=False,
    ):
        result = pure.launch(stage_root, resume=resume)
    registry = _read_registry(root)
    registry.update(
        status="backbones_trained", backbones_completed_at_utc=direct.utc_now()
    )
    _write_registry(root, registry)
    return result


def _parent_allowlist_row(stage_root: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    status_payload = _read_json(direct._status_path(stage_root, str(job["job_id"])))
    if status_payload.get("status") != "completed":
        raise RuntimeError(f"Parent is incomplete: {job['job_id']}")
    best = direct._artifact(status_payload, "best_learned_checkpoint")
    best_payload = _read_json(Path(str(best["path"])))
    best_epoch = int(best_payload["best_epoch"])
    training_payload = yaml.safe_load(
        Path(str(job["training_config_path"])).read_text(encoding="utf-8")
    )
    if not isinstance(training_payload, Mapping):
        raise ValueError(f"Invalid parent training config: {job['job_id']}")
    maximum_epoch = int(training_payload["num_epochs"])
    if not 1 <= best_epoch <= maximum_epoch:
        raise ValueError(f"Invalid parent best epoch: {job['job_id']}")
    full_state = direct._artifact(status_payload, "full_training_state")
    generator = direct._artifact(status_payload, "generator_best_learned")
    discriminator = direct._artifact(status_payload, "discriminator_best_learned")
    full_state_payload = core._validated_full_state_payload(job, status_payload)
    generator_state_sha = _tensor_state_sha256(
        full_state_payload["generator_state_dict"]
    )
    discriminator_state_sha = _tensor_state_sha256(
        full_state_payload["discriminator_state_dict"]
    )
    if generator_state_sha != _canonical_checkpoint_state_sha256(generator["path"]):
        raise ValueError(
            f"Parent full-state Generator differs from best checkpoint: {job['job_id']}"
        )
    if discriminator_state_sha != _canonical_checkpoint_state_sha256(
        discriminator["path"]
    ):
        raise ValueError(
            f"Parent full-state Critic differs from best checkpoint: {job['job_id']}"
        )
    for epoch in job.get("validation_snapshot_epochs", ()):
        direct._artifact(status_payload, f"generator_validation_epoch_{epoch:04d}")
        direct._artifact(status_payload, f"discriminator_validation_epoch_{epoch:04d}")
    return {
        "parent_job_id": parent_job_id(int(job["seed"]), str(job["fold"])),
        "internal_job_id": str(job["job_id"]),
        "seed": int(job["seed"]),
        "fold": str(job["fold"]),
        "best_epoch": best_epoch,
        "full_state_completed_epoch": int(full_state_payload["completed_epoch"]),
        "full_state_generator_state_sha256": generator_state_sha,
        "full_state_discriminator_state_sha256": discriminator_state_sha,
        "generator_path": str(generator["path"]),
        "generator_sha256": str(generator["sha256"]),
        "discriminator_path": str(discriminator["path"]),
        "discriminator_sha256": str(discriminator["sha256"]),
        "full_state_path": str(full_state["path"]),
        "full_state_sha256": str(full_state["sha256"]),
        "training_metrics_path": str(
            direct._artifact(status_payload, "training_metrics_csv")["path"]
        ),
    }


def _parent_allowlist_rows(root: Path) -> list[dict[str, Any]]:
    stage_root = _stage_roots(root)["backbones"]
    registry = direct.read_registry(stage_root)
    rows = [_parent_allowlist_row(stage_root, job) for job in registry["jobs"]]
    if len(rows) != 40:
        raise ValueError("Parent allowlist must contain 40 entries")
    return sorted(rows, key=lambda row: row["parent_job_id"])


def freeze_backbones(output_dir: str | Path, *, resume: bool = False) -> Path:
    root = _root(output_dir)
    validate_root(root, verify_stage_roots=True)
    registry = _read_registry(root)
    path = root / "registry/parent_allowlist.json"
    if registry.get("parents_frozen"):
        if not path.is_file() or sha256_file(path) != registry.get(
            "parent_allowlist_sha256"
        ):
            raise ValueError("Frozen parent allowlist drift")
        _validate_parent_allowlist(root, registry, verify_artifacts=True)
        return path
    stage_root = _stage_roots(root)["backbones"]
    with _pure_stage_profile(
        main_root=root,
        stage_name="backbones",
        arm=PARENT_ARM,
        use_graft=False,
    ):
        direct._freeze_checkpoints(stage_root)
    entries = _parent_allowlist_rows(root)
    payload = {
        "schema_version": 1,
        "kind": "pure_cnn_parent_allowlist_v1",
        "entries": entries,
        "test_loader_count": 0,
        "test_prediction_count": 0,
        "frozen_at_utc": direct.utc_now(),
    }
    payload["payload_sha256"] = direct.payload_sha256(payload)
    _write_json(path, payload)
    registry = _read_registry(root)
    by_parent = {str(row["parent_job_id"]): row for row in entries}
    jobs: list[dict[str, Any]] = []
    for raw in registry["jobs"]:
        job = dict(raw)
        if job["stage"] == "backbone":
            parent = by_parent[str(job["job_id"])]
            job.update(
                status="completed",
                selected_best_epoch=int(parent["best_epoch"]),
                selected_generator_path=str(parent["generator_path"]),
                selected_generator_sha256=str(parent["generator_sha256"]),
                selected_discriminator_path=str(parent["discriminator_path"]),
                selected_discriminator_sha256=str(parent["discriminator_sha256"]),
                selected_full_state_path=str(parent["full_state_path"]),
                selected_full_state_sha256=str(parent["full_state_sha256"]),
            )
        jobs.append(job)
    registry.update(
        status="parents_frozen",
        jobs=jobs,
        parents_frozen=True,
        parent_allowlist_path=str(path),
        parent_allowlist_sha256=sha256_file(path),
    )
    _write_registry(root, registry)
    validate_root(root, verify_stage_roots=True)
    del resume
    return path


def _block_seed(seed: int, fold: str, namespace: str) -> int:
    digest = hashlib.sha256(f"{namespace}|{int(seed)}|{fold}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _instantiate_modules(
    config: Mapping[str, Any], *, generator_mode: str, seed: int
) -> tuple[torch.nn.Module, torch.nn.Module]:
    from wgan_option.models.discriminator import Discriminator
    from wgan_option.models.generator import Generator
    from wgan_option.utils.reproducibility import seed_everything

    seed_everything(int(seed))
    common = config["models"]["common"]
    film = config["models"]["film_unet"]
    data = config["data"]
    generator = Generator(
        channels=int(common["channels"]),
        embedding_dim=int(common["embedding_dim"]),
        noise_dim=int(common["noise_dim"]),
        surface_height=16,
        surface_width=16,
        base_channels=int(common["gen_base_channels"]),
        res_blocks=int(common["gen_res_blocks"]),
        text_hidden_dim=int(film["gen_text_hidden_dim"]),
        text_out_dim=int(film["gen_text_out_dim"]),
        hidden_dim=int(common["gen_hidden_dim"]),
        residual_output_mode=str(common["residual_output_mode"]),
        generator_noise_mode=str(common["generator_noise_mode"]),
        generator_current_input_mode=str(common["generator_current_input_mode"]),
        generator_conditioning_mode=generator_mode,
        strike_grid=list(data["strike_grid"]),
        maturity_grid_days=list(data["maturity_days_grid"]),
    )
    discriminator = Discriminator(
        channels=int(common["channels"]),
        embedding_dim=int(common["embedding_dim"]),
        surface_height=16,
        surface_width=16,
        base_channels=int(common["disc_base_channels"]),
        res_blocks=int(common["disc_res_blocks"]),
        text_hidden_dim=int(common["disc_text_hidden_dim"]),
        hidden_dim=int(common["disc_hidden_dim"]),
        critic_normalization_mode=str(common["critic_normalization_mode"]),
        critic_conditioning_mode=str(common["critic_conditioning_mode"]),
    )
    expected = PURE_COUNTS if generator_mode == PURE_MODE else FILM_COUNTS
    actual = {
        "generator": sum(parameter.numel() for parameter in generator.parameters()),
        "critic": sum(parameter.numel() for parameter in discriminator.parameters()),
    }
    actual["total"] = actual["generator"] + actual["critic"]
    if actual != expected:
        raise ValueError(f"Executable parameter-count drift: {actual} != {expected}")
    return generator, discriminator


def _fresh_generator_optimizer(
    config: Mapping[str, Any], generator: torch.nn.Module, *, target_mode: str
) -> torch.optim.Optimizer:
    training = config["training"]
    if target_mode == PURE_MODE:
        return torch.optim.Adam(
            generator.parameters(),
            lr=float(training["generator_backbone_learning_rate"]),
            betas=(float(training["beta_1"]), float(training["beta_2"])),
        )
    grouped: dict[str, list[torch.nn.Parameter]] = {
        "backbone": [],
        "text_encoder": [],
        "film_projection": [],
    }
    for name, parameter in generator.named_parameters():
        if name.startswith("text_encoder."):
            grouped["text_encoder"].append(parameter)
        elif name.startswith(
            (
                "encoder_film_layers.",
                "bottleneck_film_layer.",
                "decoder_film_layers.",
            )
        ):
            grouped["film_projection"].append(parameter)
        else:
            grouped["backbone"].append(parameter)
    if any(not values for values in grouped.values()):
        raise ValueError("FiLM optimizer groups must all be non-empty")
    return torch.optim.Adam(
        [
            {
                "params": grouped["backbone"],
                "lr": float(training["generator_backbone_learning_rate"]),
                "group_name": "backbone",
            },
            {
                "params": grouped["text_encoder"],
                "lr": float(training["generator_text_learning_rate"]),
                "group_name": "text_encoder",
            },
            {
                "params": grouped["film_projection"],
                "lr": float(training["generator_film_learning_rate"]),
                "group_name": "film_projection",
            },
        ],
        betas=(float(training["beta_1"]), float(training["beta_2"])),
    )


def _block_graft_paths(
    root: Path, parent: Mapping[str, Any], target_mode: str
) -> tuple[Path, Path]:
    suffix = "film" if target_mode == FILM_MODE else "pure_restart"
    base = f"{parent['parent_job_id']}__{suffix}"
    return (
        root / "registry/graft_states" / f"{base}.pt",
        root / "registry/graft_manifests" / f"{base}.json",
    )


def _expected_graft_lineage(
    config: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    target_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = int(parent["seed"])
    fold = str(parent["fold"])
    return (
        {
            "parent_job_id": str(parent["parent_job_id"]),
            "internal_parent_job_id": str(parent["internal_job_id"]),
            "seed": seed,
            "fold": fold,
            "best_epoch": int(parent["best_epoch"]),
            "full_state_sha256": str(parent["full_state_sha256"]),
        },
        {
            "experiment_kind": EXPERIMENT_KIND,
            "interpretation": INTERPRETATION,
            "formal_config_sha256": str(config["source_config_sha256"]),
            "seed": seed,
            "fold": fold,
            "target_generator_mode": target_mode,
            "training_rng_seed": _block_seed(
                seed, fold, "shared_continuation_training_rng"
            ),
            "optimizer_state_mode": "fresh_zero_state_for_all_branches_v1",
        },
    )


def _save_block_graft(
    config: Mapping[str, Any],
    root: Path,
    parent: Mapping[str, Any],
    *,
    target_mode: str,
) -> dict[str, Any]:
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from wgan_option.utils.pure_cnn_film_graft import (
        save_pure_cnn_to_film_graft_state,
    )
    from wgan_option.utils.reproducibility import seed_everything

    seed = int(parent["seed"])
    fold = str(parent["fold"])
    artifact_path, manifest_path = _block_graft_paths(root, parent, target_mode)
    if artifact_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing graft attempt: "
            f"{artifact_path} / {manifest_path}"
        )
    full_state_path = Path(str(parent["full_state_path"])).resolve()
    if sha256_file(full_state_path) != str(parent["full_state_sha256"]):
        raise ValueError(f"Parent full-state drift: {seed}/{fold}")
    parent_state = torch.load(full_state_path, map_location="cpu", weights_only=False)
    if parent_state.get("kind") != "news_first_wgan_full_training_state_v1":
        raise ValueError("Parent is not a complete full-training-state artifact")
    source_g, source_d = _instantiate_modules(
        config,
        generator_mode=PURE_MODE,
        seed=_block_seed(seed, fold, "source_model"),
    )
    source_g.load_state_dict(parent_state["generator_state_dict"], strict=True)
    source_d.load_state_dict(parent_state["discriminator_state_dict"], strict=True)
    target_g, target_d = _instantiate_modules(
        config,
        generator_mode=target_mode,
        seed=_block_seed(seed, fold, f"target_model:{target_mode}"),
    )
    g_optimizer = _fresh_generator_optimizer(config, target_g, target_mode=target_mode)
    d_optimizer = torch.optim.Adam(
        target_d.parameters(),
        lr=float(config["training"]["discriminator_learning_rate"]),
        betas=(
            float(config["training"]["beta_1"]),
            float(config["training"]["beta_2"]),
        ),
    )
    g_scheduler = ReduceLROnPlateau(
        g_optimizer,
        factor=float(config["training"]["reduce_lr_factor"]),
        patience=int(config["training"]["reduce_lr_patience"]),
        min_lr=(
            [
                float(config["training"]["generator_backbone_min_learning_rate"]),
                float(config["training"]["generator_text_min_learning_rate"]),
                float(config["training"]["generator_film_min_learning_rate"]),
            ]
            if target_mode == FILM_MODE
            else float(config["training"]["generator_backbone_min_learning_rate"])
        ),
    )
    d_scheduler = ReduceLROnPlateau(
        d_optimizer,
        factor=float(config["training"]["reduce_lr_factor"]),
        patience=int(config["training"]["reduce_lr_patience"]),
        min_lr=float(config["training"]["discriminator_min_learning_rate"]),
    )
    training_seed = _block_seed(seed, fold, "shared_continuation_training_rng")
    seed_everything(training_seed)
    loader_generator = torch.Generator().manual_seed(training_seed)
    current = torch.linspace(0.12, 0.28, 2 * 16 * 16).reshape(2, 1, 16, 16)
    support = torch.ones_like(current)
    noise = torch.linspace(-1.0, 1.0, 2 * 32).reshape(2, 32)
    zero = torch.zeros(2, 1024)
    matched = torch.linspace(-0.5, 0.5, 2 * 1024).reshape(2, 1024)
    shuffle = torch.flip(matched, dims=(0,))
    parent_lineage, graft_lineage = _expected_graft_lineage(
        config, parent, target_mode=target_mode
    )
    result = save_pure_cnn_to_film_graft_state(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        parent_checkpoint_path=full_state_path,
        expected_parent_checkpoint_sha256=str(parent["full_state_sha256"]),
        pure_cnn_generator=source_g,
        target_generator=target_g,
        pure_cnn_critic=source_d,
        target_discriminator=target_d,
        generator_optimizer=g_optimizer,
        critic_optimizer=d_optimizer,
        generator_scheduler=g_scheduler,
        critic_scheduler=d_scheduler,
        loader_generator=loader_generator,
        parent_lineage=parent_lineage,
        graft_lineage=graft_lineage,
        current_surface=current,
        current_support_mask=support,
        noise=noise,
        text_embeddings={"zero": zero, "matched": matched, "shuffle": shuffle},
        verification_tolerance=1e-7,
        expected_cuda_device_count=1,
        cuda_source_device_index=int(config["runtime"]["gpu_fold_assignment"][fold]),
    )
    artifact = torch.load(
        Path(str(result["artifact_path"])), map_location="cpu", weights_only=False
    )
    return {
        "parent_job_id": str(parent["parent_job_id"]),
        "seed": seed,
        "fold": fold,
        "target_generator_mode": target_mode,
        "artifact_path": str(result["artifact_path"]),
        "artifact_sha256": str(result["artifact_sha256"]),
        "manifest_path": str(result["manifest_path"]),
        "manifest_sha256": str(result["manifest_sha256"]),
        "generator_state_sha256": _tensor_state_sha256(
            artifact["generator_state_dict"]
        ),
        "discriminator_state_sha256": _tensor_state_sha256(
            artifact["discriminator_state_dict"]
        ),
        "epoch0_max_abs": float(result["epoch0_equivalence"]["maximum_absolute_error"]),
        "optimizer_reset_proof": result["optimizer_reset_proof"],
    }


def _resume_block_graft(
    config: Mapping[str, Any],
    root: Path,
    parent: Mapping[str, Any],
    *,
    target_mode: str,
) -> dict[str, Any] | None:
    """Reuse only a complete, deeply valid artifact/manifest pair."""

    from wgan_option.utils.pure_cnn_film_graft import (
        GRAFT_MANIFEST_KIND,
        read_pure_cnn_graft_metadata,
    )

    artifact_path, manifest_path = _block_graft_paths(root, parent, target_mode)
    existence = (artifact_path.is_file(), manifest_path.is_file())
    if existence == (False, False):
        return None
    if existence != (True, True):
        raise RuntimeError(
            "Partial graft artifact/manifest pair requires manual audit: "
            f"{artifact_path} / {manifest_path}"
        )
    manifest = _read_json(manifest_path)
    artifact_sha = sha256_file(artifact_path)
    parent_lineage, graft_lineage = _expected_graft_lineage(
        config, parent, target_mode=target_mode
    )
    metadata = read_pure_cnn_graft_metadata(
        artifact_path=artifact_path,
        expected_artifact_sha256=artifact_sha,
    )
    if (
        manifest.get("kind") != GRAFT_MANIFEST_KIND
        or Path(str(manifest.get("artifact_path", ""))).resolve()
        != artifact_path.resolve()
        or int(manifest.get("artifact_size_bytes", -1)) != artifact_path.stat().st_size
        or manifest.get("artifact_sha256") != artifact_sha
        or Path(str(manifest.get("parent_checkpoint_path", ""))).resolve()
        != Path(str(parent["full_state_path"])).resolve()
        or manifest.get("parent_checkpoint_sha256") != str(parent["full_state_sha256"])
        or manifest.get("target_generator_mode") != target_mode
        or manifest.get("parent_lineage") != parent_lineage
        or manifest.get("graft_lineage") != graft_lineage
        or metadata.get("parent_lineage") != parent_lineage
        or metadata.get("graft_lineage") != graft_lineage
        or metadata.get("target_generator_mode") != target_mode
        or metadata.get("cuda_rng_topology", {}).get("worker_cuda_device_count") != 1
        or metadata.get("cuda_rng_topology", {}).get("source_cuda_device_index")
        != int(config["runtime"]["gpu_fold_assignment"][str(parent["fold"])])
        or float(
            metadata.get("epoch0_equivalence", {}).get(
                "maximum_absolute_error", math.inf
            )
        )
        > 1e-7
    ):
        raise ValueError(f"Existing graft lineage drift: {artifact_path}")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    return {
        "parent_job_id": str(parent["parent_job_id"]),
        "seed": int(parent["seed"]),
        "fold": str(parent["fold"]),
        "target_generator_mode": target_mode,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": artifact_sha,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "generator_state_sha256": _tensor_state_sha256(
            artifact["generator_state_dict"]
        ),
        "discriminator_state_sha256": _tensor_state_sha256(
            artifact["discriminator_state_dict"]
        ),
        "epoch0_max_abs": float(
            metadata["epoch0_equivalence"]["maximum_absolute_error"]
        ),
        "optimizer_reset_proof": dict(metadata["optimizer_reset_proof"]),
    }


def graft(output_dir: str | Path, *, resume: bool = False) -> Path:
    root = _root(output_dir)
    config = validate_root(root, verify_stage_roots=True)
    registry = _read_registry(root)
    allowlist_path = root / "registry/graft_allowlist.json"
    if registry.get("grafts_frozen"):
        _validate_graft_allowlist(root, registry, verify_artifacts=True)
        if registry.get("continuations_prepared"):
            return allowlist_path
        if not resume:
            raise RuntimeError(
                "Grafts are frozen but continuation preparation is incomplete; "
                "rerun with --resume"
            )
        _prepare_continuation_stages(
            config,
            root,
            workers=int(registry["formal_workers_per_gpu"]),
            num_epochs=240,
            resume=True,
        )
        registry = _read_registry(root)
        registry.update(
            status="continuations_prepared",
            continuations_prepared=True,
            continuations_prepared_at_utc=direct.utc_now(),
        )
        _write_registry(root, registry)
        validate_root(root, verify_stage_roots=True)
        return allowlist_path
    if not registry.get("parents_frozen"):
        raise RuntimeError("Graft requires a frozen 40-parent allowlist")
    parents = _read_json(root / "registry/parent_allowlist.json")["entries"]
    entries: list[dict[str, Any]] = []
    for parent in parents:
        for target_mode in (PURE_MODE, FILM_MODE):
            existing = (
                _resume_block_graft(config, root, parent, target_mode=target_mode)
                if resume
                else None
            )
            entries.append(
                existing
                if existing is not None
                else _save_block_graft(config, root, parent, target_mode=target_mode)
            )
    if len(entries) != 80:
        raise AssertionError("Expected 40 FiLM grafts and 40 Pure-CNN restarts")
    for seed in SEEDS:
        for fold in FOLDS:
            block = [
                row
                for row in entries
                if int(row["seed"]) == seed and str(row["fold"]) == fold
            ]
            if len(block) != 2:
                raise ValueError(f"Incomplete graft block: {seed}/{fold}")
            if len({row["discriminator_state_sha256"] for row in block}) != 1:
                raise ValueError(
                    f"Critic copy differs across graft targets: {seed}/{fold}"
                )
            if any(float(row["epoch0_max_abs"]) > 1e-7 for row in block):
                raise ValueError(f"Epoch-0 graft equivalence failed: {seed}/{fold}")
    payload = {
        "schema_version": 1,
        "kind": GRAFT_ALLOWLIST_KIND,
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
        "frozen_at_utc": direct.utc_now(),
    }
    payload["payload_sha256"] = direct.payload_sha256(payload)
    _write_json(allowlist_path, payload)
    by_target = {
        (int(row["seed"]), str(row["fold"]), str(row["target_generator_mode"])): row
        for row in entries
    }
    jobs = []
    for raw in registry["jobs"]:
        job = dict(raw)
        if job["stage"] == "continuation":
            target_mode = (
                PURE_MODE if job["arm"] == PURE_CONTINUATION_ARM else FILM_MODE
            )
            entry = by_target[(int(job["seed"]), str(job["fold"]), target_mode)]
            job.update(
                graft_state_path=str(entry["artifact_path"]),
                graft_state_sha256=str(entry["artifact_sha256"]),
                graft_manifest_path=str(entry["manifest_path"]),
                graft_manifest_sha256=str(entry["manifest_sha256"]),
                initial_generator_state_sha256=str(entry["generator_state_sha256"]),
                initial_critic_state_sha256=str(entry["discriminator_state_sha256"]),
                status="pending",
            )
        jobs.append(job)
    registry.update(
        status="grafts_frozen",
        jobs=jobs,
        grafts_frozen=True,
        graft_allowlist_path=str(allowlist_path),
        graft_allowlist_sha256=sha256_file(allowlist_path),
    )
    _write_registry(root, registry)
    workers = int(registry["formal_workers_per_gpu"])
    _prepare_continuation_stages(
        config,
        root,
        workers=workers,
        num_epochs=240,
        resume=resume,
    )
    registry = _read_registry(root)
    registry.update(
        status="continuations_prepared",
        continuations_prepared=True,
        continuations_prepared_at_utc=direct.utc_now(),
    )
    _write_registry(root, registry)
    validate_root(root, verify_stage_roots=True)
    return allowlist_path


def launch_continuations(output_dir: str | Path, *, resume: bool = False) -> Path:
    root = _root(output_dir)
    validate_root(root, verify_stage_roots=True)
    registry = _read_registry(root)
    if not registry.get("continuations_prepared"):
        raise RuntimeError("Continuation launch requires frozen grafts and stage specs")
    if registry.get("evaluation_frozen"):
        if resume:
            return _stage_roots(root)["film_continuations"]
        raise RuntimeError("Continuation checkpoints are already frozen")
    roots = _stage_roots(root)
    with _pure_stage_profile(
        main_root=root,
        stage_name="pure_continuation",
        arm=PURE_CONTINUATION_ARM,
        use_graft=True,
    ):
        pure.launch(roots["pure_continuation"], resume=resume)
    with _film_stage_profile(main_root=root, stage_name="film_continuations"):
        result = film_text.launch(roots["film_continuations"], resume=resume)
    registry = _read_registry(root)
    registry.update(
        status="continuations_trained",
        continuations_completed_at_utc=direct.utc_now(),
    )
    _write_registry(root, registry)
    return result


def _selected_checkpoint_rows(
    stage_root: Path,
    *,
    arm_mapping: Mapping[str, str],
    main_job_id_builder: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    registry = direct.read_registry(stage_root)
    for job in registry["jobs"]:
        status_payload = _read_json(direct._status_path(stage_root, str(job["job_id"])))
        if status_payload.get("status") != "completed":
            raise RuntimeError(f"Incomplete checkpoint cell: {job['job_id']}")
        arm = arm_mapping[str(job["arm"])]
        main_job_id = main_job_id_builder(int(job["seed"]), str(job["fold"]), arm)
        best_payload = _read_json(
            Path(
                str(direct._artifact(status_payload, "best_learned_checkpoint")["path"])
            )
        )
        best_epoch = int(best_payload["best_epoch"])
        if not 1 <= best_epoch <= 240:
            raise ValueError(f"Invalid selected learned epoch: {main_job_id}")
        for role in ("generator_best_learned", "discriminator_best_learned"):
            artifact = direct._artifact(status_payload, role)
            path = Path(str(artifact["path"])).resolve()
            rows.append(
                {
                    "job_id": main_job_id,
                    "internal_job_id": str(job["job_id"]),
                    "seed": int(job["seed"]),
                    "fold": str(job["fold"]),
                    "arm": arm,
                    "best_epoch": best_epoch,
                    "checkpoint_role": role,
                    "checkpoint_path": str(path),
                    "checkpoint_sha256": str(artifact["sha256"]),
                    "size_bytes": path.stat().st_size,
                    "stage_root": str(stage_root),
                }
            )
    return rows


def _checkpoint_allowlist_rows(root: Path) -> list[dict[str, Any]]:
    roots = _stage_roots(root)
    rows: list[dict[str, Any]] = []
    rows.extend(
        _selected_checkpoint_rows(
            roots["backbones"],
            arm_mapping={PARENT_ARM: PARENT_ARM},
            main_job_id_builder=lambda seed, fold, arm: parent_job_id(seed, fold),
        )
    )
    rows.extend(
        _selected_checkpoint_rows(
            roots["pure_continuation"],
            arm_mapping={PURE_CONTINUATION_ARM: PURE_CONTINUATION_ARM},
            main_job_id_builder=continuation_job_id,
        )
    )
    rows.extend(
        _selected_checkpoint_rows(
            roots["film_continuations"],
            arm_mapping={arm: arm for arm in FILM_ARMS},
            main_job_id_builder=continuation_job_id,
        )
    )
    if len(rows) != EXPECTED_TRAINING_JOBS * 2:
        raise ValueError("Checkpoint allowlist must contain 280 G/D pairs")
    expected = {
        (job["job_id"], role)
        for job in _read_registry(root)["jobs"]
        for role in ("generator_best_learned", "discriminator_best_learned")
    }
    observed = {(row["job_id"], row["checkpoint_role"]) for row in rows}
    if observed != expected:
        raise ValueError("Checkpoint allowlist logical job universe drift")
    return sorted(rows, key=lambda row: (row["job_id"], row["checkpoint_role"]))


def freeze_evaluation(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = _root(output_dir)
    validate_root(root, verify_stage_roots=True)
    registry = _read_registry(root)
    allowlist_path = root / "evaluation/checkpoint_allowlist.csv"
    if registry.get("evaluation_frozen"):
        if not allowlist_path.is_file() or sha256_file(allowlist_path) != registry.get(
            "checkpoint_allowlist_sha256"
        ):
            raise ValueError("Frozen checkpoint allowlist drift")
        return allowlist_path
    if not registry.get("continuations_prepared") or not registry.get("grafts_frozen"):
        raise RuntimeError("Evaluation freeze requires all continuation contracts")
    roots = _stage_roots(root)
    with _pure_stage_profile(
        main_root=root,
        stage_name="pure_continuation",
        arm=PURE_CONTINUATION_ARM,
        use_graft=True,
    ):
        direct._freeze_checkpoints(roots["pure_continuation"])
    with _film_stage_profile(main_root=root, stage_name="film_continuations"):
        direct._freeze_checkpoints(roots["film_continuations"])
    rows = _checkpoint_allowlist_rows(root)
    direct.write_csv(
        allowlist_path,
        rows,
        (
            "job_id",
            "internal_job_id",
            "seed",
            "fold",
            "arm",
            "best_epoch",
            "checkpoint_role",
            "checkpoint_path",
            "checkpoint_sha256",
            "size_bytes",
            "stage_root",
        ),
    )
    registry = _read_registry(root)
    selected: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        selected.setdefault(str(row["job_id"]), {})[str(row["checkpoint_role"])] = row
    jobs: list[dict[str, Any]] = []
    for raw in registry["jobs"]:
        job = dict(raw)
        checkpoints = selected[str(job["job_id"])]
        generator = checkpoints["generator_best_learned"]
        discriminator = checkpoints["discriminator_best_learned"]
        job.update(
            status="completed",
            selected_best_epoch=int(generator["best_epoch"]),
            selected_generator_path=str(generator["checkpoint_path"]),
            selected_generator_sha256=str(generator["checkpoint_sha256"]),
            selected_discriminator_path=str(discriminator["checkpoint_path"]),
            selected_discriminator_sha256=str(discriminator["checkpoint_sha256"]),
        )
        jobs.append(job)
    registry.update(
        status="evaluation_frozen",
        jobs=jobs,
        evaluation_frozen=True,
        evaluation_frozen_at_utc=direct.utc_now(),
        checkpoint_allowlist_path=str(allowlist_path),
        checkpoint_allowlist_sha256=sha256_file(allowlist_path),
        checkpoint_allowlist_rows=len(rows),
        test_data_opened=False,
    )
    _write_registry(root, registry)
    validate_root(root, verify_stage_roots=True)
    return allowlist_path


def _logical_arm(internal_arm: str) -> str:
    return str(internal_arm)


def _logical_job_id(seed: int, fold: str, arm: str) -> str:
    return (
        parent_job_id(seed, fold)
        if arm == PARENT_ARM
        else continuation_job_id(seed, fold, arm)
    )


_PARALLEL_PREDICTION_WORKER = (
    "scripts.rq3."
    "news_first_vol_film_unet_pure_cnn_backbone_text_effect_prediction_worker:"
    "run_prediction_unit"
)


def _materialize_parallel_test_inputs(config: Mapping[str, Any], root: Path) -> None:
    """Freeze the three child-stage test panels only after the public freeze."""

    registry = _read_registry(root)
    if not registry.get("evaluation_frozen"):
        raise RuntimeError("Child test inputs require the public evaluation freeze")
    roots = _stage_roots(root)
    pure_stages = (
        ("backbones", PARENT_ARM, False),
        ("pure_continuation", PURE_CONTINUATION_ARM, True),
    )
    for stage_name, arm, use_graft in pure_stages:
        stage_root = roots[stage_name]
        stage_config = yaml.safe_load(
            (stage_root / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        with (
            _pure_stage_profile(
                main_root=root,
                stage_name=stage_name,
                arm=arm,
                use_graft=use_graft,
            ),
            pure._runtime_profile(),
        ):
            direct._materialize_test_inputs(stage_config, stage_root)
            direct._validate_test_inputs(stage_root)
    film_root = roots["film_continuations"]
    film_config = yaml.safe_load(
        (film_root / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    with (
        _film_stage_profile(main_root=root, stage_name="film_continuations"),
        film_text.film_text_profile(),
        film_text.multiseed.multiseed_profile(),
    ):
        direct._materialize_test_inputs(film_config, film_root)
        direct._validate_test_inputs(film_root)
    del config


def _test_sample_ids(root: Path) -> dict[tuple[str, str], list[str]]:
    """Return one SHA-independent sample universe shared by all child stages."""

    roots = _stage_roots(root)
    output: dict[tuple[str, str], list[str]] = {}
    for fold in FOLDS:
        reference: list[str] | None = None
        for stage_root in roots.values():
            panel_path = direct._test_panel_path(stage_root, fold)
            if not panel_path.is_file():
                raise ValueError(f"Frozen child test panel is missing: {panel_path}")
            panel = pd.read_csv(panel_path, dtype=str, keep_default_na=False)
            if (
                "sample_id" not in panel.columns
                or panel["sample_id"].duplicated().any()
            ):
                raise ValueError(f"Child test sample-ID contract drift: {panel_path}")
            observed = sorted(panel["sample_id"].astype(str))
            if reference is None:
                reference = observed
            elif observed != reference:
                raise ValueError(f"Child test panels disagree for {fold}")
        if reference is None:
            raise ValueError(f"No child test panel found for {fold}")
        output[("test", fold)] = reference
    return output


def _freeze_parallel_unit_manifest(path: Path, units: pd.DataFrame) -> Path:
    """Write once, or require byte-equivalent scientific prediction units."""

    serial = units.copy()
    if "expected_artifact_roles" in serial.columns:
        serial["expected_artifact_roles"] = serial["expected_artifact_roles"].map(
            lambda value: json.dumps(list(value), separators=(",", ":"))
        )
    expected = serial.fillna("").astype(str)
    if path.is_file():
        observed = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(observed.columns) != list(expected.columns) or observed.to_dict(
            orient="records"
        ) != expected.to_dict(orient="records"):
            raise ValueError(f"Frozen prediction-unit manifest drift: {path}")
        return path
    return direct.write_csv(
        path, serial.to_dict(orient="records"), tuple(serial.columns)
    )


def _run_parallel_prediction_stage(
    root: Path,
    config: Mapping[str, Any],
    units: pd.DataFrame,
    *,
    stage: str,
    workers_key: str,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_parallel_prediction import (
        run_parallel_prediction_units,
    )

    unit_path = _freeze_parallel_unit_manifest(
        root / f"evaluation/{stage}_prediction_units.csv", units
    )
    payload = run_parallel_prediction_units(
        units.to_dict(orient="records"),
        worker_entrypoint=_PARALLEL_PREDICTION_WORKER,
        control_dir=root / f"control/parallel_predictions/{stage}",
        artifact_root=root,
        gpu_ids=tuple(map(int, config["runtime"]["gpu_ids"])),
        workers_per_gpu=int(config["runtime"][workers_key]),
        resume=resume,
    )
    return unit_path, payload


def _parallel_cell_artifacts(
    root: Path, stage: str, unit_id: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_parallel_prediction import (
        prediction_cell_manifest_path,
    )

    path = prediction_cell_manifest_path(
        root / f"control/parallel_predictions/{stage}", unit_id
    )
    payload = _read_json(path)
    if payload.get("prediction_unit_id") != unit_id:
        raise ValueError(f"Parallel cell identity drift: {unit_id}")
    artifacts = {
        str(row["role"]): dict(row) for row in list(payload.get("artifacts") or [])
    }
    if len(artifacts) != len(list(payload.get("artifacts") or [])):
        raise ValueError(f"Parallel cell artifact-role drift: {unit_id}")
    for artifact in artifacts.values():
        target = Path(str(artifact["path"])).resolve()
        if (
            not target.is_file()
            or target.stat().st_size != int(artifact["size_bytes"])
            or sha256_file(target) != str(artifact["sha256"])
            or (target != root and root not in target.parents)
        ):
            raise ValueError(f"Parallel cell artifact drift: {unit_id}/{target}")
    return payload, artifacts


def _parallel_standard_aggregate(root: Path, units: pd.DataFrame) -> tuple[Path, Path]:
    evidence_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for unit in units.to_dict(orient="records"):
        unit_id = str(unit["prediction_unit_id"])
        cell, artifacts = _parallel_cell_artifacts(root, "standard", unit_id)
        if set(artifacts) != {
            "prediction",
            "core_prediction_manifest",
            "pair_metrics",
        }:
            raise ValueError(f"Standard artifact universe drift: {unit_id}")
        evidence = pd.read_csv(artifacts["pair_metrics"]["path"])
        expected_pairs = int(_expected_fold_counts()[str(unit["fold"])][4])
        if len(evidence) != expected_pairs:
            raise ValueError(f"Standard pair coverage drift: {unit_id}")
        evidence = evidence.copy()
        evidence["job_id"] = str(unit["job_id"])
        evidence["seed"] = int(unit["seed"])
        evidence["fold"] = str(unit["fold"])
        evidence["arm"] = str(unit["arm"])
        if set(evidence["noise_bank_profile_sha256"].astype(str)) != {
            str(unit["noise_bank_profile_sha256"])
        }:
            raise ValueError(f"Standard MC64 lineage drift: {unit_id}")
        evidence_frames.append(evidence)
        manifest_rows.append(
            {
                "prediction_unit_id": unit_id,
                "job_id": str(unit["job_id"]),
                "seed": int(unit["seed"]),
                "fold": str(unit["fold"]),
                "arm": str(unit["arm"]),
                "checkpoint_path": str(unit["checkpoint_path"]),
                "checkpoint_sha256": str(unit["checkpoint_sha256"]),
                "prediction_path": str(artifacts["prediction"]["path"]),
                "prediction_sha256": str(artifacts["prediction"]["sha256"]),
                "core_prediction_manifest_path": str(
                    artifacts["core_prediction_manifest"]["path"]
                ),
                "core_prediction_manifest_sha256": str(
                    artifacts["core_prediction_manifest"]["sha256"]
                ),
                "pair_metrics_path": str(artifacts["pair_metrics"]["path"]),
                "pair_metrics_sha256": str(artifacts["pair_metrics"]["sha256"]),
                "noise_bank_profile_sha256": str(unit["noise_bank_profile_sha256"]),
                "physical_gpu_id": int(cell["physical_gpu_id"]),
            }
        )
    evidence = pd.concat(evidence_frames, ignore_index=True)
    if len(evidence) != EXPECTED_STANDARD_PAIR_ROWS or len(manifest_rows) != (
        EXPECTED_STANDARD_PREDICTIONS
    ):
        raise ValueError("Parallel standard prediction count drift")
    expected_cells = {
        (seed, fold, arm) for seed in SEEDS for fold in FOLDS for arm in STANDARD_ARMS
    }
    if (
        set(
            evidence[["seed", "fold", "arm"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        != expected_cells
    ):
        raise ValueError("Parallel standard cell universe drift")
    for _key, group in evidence.groupby(["seed", "fold", "pair_id"], sort=False):
        if (
            group["arm"].nunique() != len(STANDARD_ARMS)
            or group["noise_bank_profile_sha256"].astype(str).nunique() != 1
        ):
            raise ValueError("Standard arms do not share complete paired MC64 evidence")
    pair_path = core._write_dataframe_csv(
        root / "evaluation/standard_pair_metrics.csv.gz",
        evidence.sort_values(["seed", "fold", "arm", "pair_id"], kind="stable"),
        gzip=True,
    )
    manifest_path = direct.write_csv(
        root / "evaluation/standard_prediction_manifest.csv",
        sorted(manifest_rows, key=lambda row: row["prediction_unit_id"]),
        tuple(manifest_rows[0]),
    )
    return manifest_path, pair_path


def _aggregate_standard_predictions(root: Path) -> tuple[Path, Path]:
    rows: list[pd.DataFrame] = []
    manifest_rows: list[pd.DataFrame] = []
    for stage_root in _stage_roots(root).values():
        registry = direct.read_registry(stage_root)
        pair_path = Path(str(registry["pair_metrics_path"])).resolve()
        manifest_path = Path(str(registry["prediction_manifest_path"])).resolve()
        if sha256_file(pair_path) != str(registry["pair_metrics_sha256"]):
            raise ValueError(f"Internal pair metrics drift: {stage_root}")
        if sha256_file(manifest_path) != str(registry["prediction_manifest_sha256"]):
            raise ValueError(f"Internal prediction manifest drift: {stage_root}")
        frame = pd.read_csv(pair_path)
        frame["arm"] = frame["arm"].astype(str).map(_logical_arm)
        frame["job_id"] = [
            _logical_job_id(int(seed), str(fold), str(arm))
            for seed, fold, arm in frame[["seed", "fold", "arm"]].itertuples(
                index=False, name=None
            )
        ]
        rows.append(frame)
        manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        manifest["arm"] = manifest["arm"].astype(str).map(_logical_arm)
        manifest["job_id"] = [
            _logical_job_id(int(seed), str(fold), str(arm))
            for seed, fold, arm in manifest[["seed", "fold", "arm"]].itertuples(
                index=False, name=None
            )
        ]
        manifest_rows.append(manifest)
    evidence = pd.concat(rows, ignore_index=True)
    manifests = pd.concat(manifest_rows, ignore_index=True)
    if len(evidence) != EXPECTED_STANDARD_PAIR_ROWS:
        raise ValueError("Standard pair-metric row count drift")
    if len(manifests) != EXPECTED_STANDARD_PREDICTIONS:
        raise ValueError("Standard prediction manifest count drift")
    expected_cells = {
        (seed, fold, arm) for seed in SEEDS for fold in FOLDS for arm in STANDARD_ARMS
    }
    observed_cells = set(
        evidence[["seed", "fold", "arm"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise ValueError("Standard prediction cell universe drift")
    key_columns = ["seed", "fold", "pair_id"]
    lineage_columns = [
        column
        for column in (
            "session_id",
            "effective_origin_utc",
            "persistence_mae",
        )
        if column in evidence.columns
    ]
    for _key, group in evidence.groupby(key_columns, sort=False):
        for column in lineage_columns:
            if group[column].astype(str).nunique() != 1:
                raise ValueError(f"Standard prediction lineage differs: {column}")
        if group["arm"].nunique() != len(STANDARD_ARMS):
            raise ValueError("A standard pair is missing one or more arms")
        if group["noise_bank_profile_sha256"].astype(str).nunique() != 1:
            raise ValueError("Standard arms do not share the MC64 noise bank")
    pair_path = core._write_dataframe_csv(
        root / "evaluation/standard_pair_metrics.csv.gz",
        evidence.sort_values(["seed", "fold", "arm", "pair_id"], kind="stable"),
        gzip=True,
    )
    manifest_path = direct.write_csv(
        root / "evaluation/standard_prediction_manifest.csv",
        manifests.sort_values(["seed", "fold", "arm"], kind="stable").to_dict(
            orient="records"
        ),
        tuple(manifests.columns),
    )
    return manifest_path, pair_path


def predict_test(output_dir: str | Path, *, resume: bool = False) -> Path:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_prediction import (
        attach_noise_bank_profiles,
        plan_standard_prediction_units,
        require_frozen_test_access,
    )

    root = _root(output_dir)
    config = validate_root(root, verify_stage_roots=True)
    registry = _read_registry(root)
    require_frozen_test_access(registry)
    if registry.get("standard_predictions_frozen"):
        path = Path(str(registry["standard_pair_metrics_path"])).resolve()
        if not path.is_file() or sha256_file(path) != registry.get(
            "standard_pair_metrics_sha256"
        ):
            raise ValueError("Frozen standard predictions drift")
        return path
    if registry.get("test_data_opened") and not resume:
        raise RuntimeError(
            "A standard prediction attempt already opened test data; rerun with --resume"
        )
    registry.update(test_data_opened=True, test_data_opened_at_utc=direct.utc_now())
    _write_registry(root, registry)
    _materialize_parallel_test_inputs(config, root)
    registry = _read_registry(root)
    units = plan_standard_prediction_units(
        pd.DataFrame(registry["jobs"]), registry["checkpoint_allowlist_path"]
    )
    units = attach_noise_bank_profiles(
        units, sample_ids_by_split_fold=_test_sample_ids(root)
    )
    units["experiment_root"] = str(root)
    units["expected_artifact_roles"] = [
        ["prediction", "core_prediction_manifest", "pair_metrics"]
        for _ in range(len(units))
    ]
    unit_path, execution = _run_parallel_prediction_stage(
        root,
        config,
        units,
        stage="standard",
        workers_key="prediction_workers_per_gpu",
        resume=resume,
    )
    manifest_path, pair_path = _parallel_standard_aggregate(root, units)
    registry = _read_registry(root)
    registry.update(
        status="standard_predictions_frozen",
        standard_predictions_frozen=True,
        standard_prediction_manifest_path=str(manifest_path),
        standard_prediction_manifest_sha256=sha256_file(manifest_path),
        standard_pair_metrics_path=str(pair_path),
        standard_pair_metrics_sha256=sha256_file(pair_path),
        standard_prediction_cells=EXPECTED_STANDARD_PREDICTIONS,
        standard_pair_metric_rows=EXPECTED_STANDARD_PAIR_ROWS,
        standard_prediction_units_path=str(unit_path),
        standard_prediction_units_sha256=sha256_file(unit_path),
        standard_parallel_execution_manifest_path=str(
            (
                root
                / "control/parallel_predictions/standard/parallel_prediction_execution_manifest.json"
            ).resolve()
        ),
        standard_parallel_execution_manifest_sha256=sha256_file(
            root
            / "control/parallel_predictions/standard/parallel_prediction_execution_manifest.json"
        ),
        standard_parallel_worker_count=int(execution["prediction_unit_count"]),
    )
    _write_registry(root, registry)
    validate_root(root, verify_stage_roots=True)
    return pair_path


def _materialize_intervention_overlays(config: Mapping[str, Any], root: Path) -> Path:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_prediction import (
        build_intervention_overlay_frames,
    )
    from wgan_option.utils.news_first_experiment_core import (
        write_pair_text_overlay_manifest,
    )

    film_root = _stage_roots(root)["film_continuations"]
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        matched_path = direct._test_overlay_path(film_root, fold, "film_lp_matched")
        shuffle_path = direct._test_overlay_path(film_root, fold, "film_lp_shuffle")
        matched_payload = _read_json(matched_path)
        shuffle_payload = _read_json(shuffle_path)
        matched_frame = pd.DataFrame(matched_payload["records"])
        forbidden = {
            str(row["pair_id"]): str(row["donor_pair_id"])
            for row in shuffle_payload["records"]
        }
        frames = build_intervention_overlay_frames(
            matched_frame,
            master_seed=int(
                config["evaluation"]["interventions"]["input_modes"][1]["mapping_seed"]
            ),
            namespace=f"text_effectiveness/test/{fold}/independent_wrong/v1",
            training_shuffle_mapping=forbidden,
        )
        mapping = frames["mapping"]
        if any(
            forbidden[str(row.pair_id)] == str(row.donor_pair_id)
            for row in mapping.itertuples(index=False)
        ):
            raise ValueError("Intervention wrong mapping repeats training shuffle")
        mapping_path = core._write_dataframe_csv(
            root / f"evaluation/intervention_mappings/{fold}.csv", mapping
        )
        rows.append(direct.manifest_row(f"intervention_mapping:{fold}", mapping_path))
        for condition, arm, mode in (
            ("zero_input", "film_lp_matched__zero_input", "current_only"),
            ("wrong_input", "film_lp_matched__wrong_input", "lp_mean_l2"),
        ):
            records = []
            for raw in frames[condition].to_dict(orient="records"):
                records.append(
                    {
                        "pair_id": str(raw["pair_id"]),
                        "session_id": str(raw["session_id"]),
                        "embedding": raw["embedding"],
                        **(
                            {"donor_pair_id": str(raw["donor_pair_id"])}
                            if condition == "wrong_input"
                            else {}
                        ),
                    }
                )
            destination = direct._test_overlay_path(film_root, fold, arm)
            write_pair_text_overlay_manifest(
                destination,
                mode=mode,
                namespace=f"evaluation/test/05m/{fold}/{arm}",
                records=records,
                transform={
                    "method": (
                        "zero_vector_v1"
                        if condition == "zero_input"
                        else "independent_cross_session_derangement_v1"
                    ),
                    "matched_overlay_path": str(matched_path),
                    "matched_overlay_sha256": sha256_file(matched_path),
                    "training_shuffle_overlay_path": str(shuffle_path),
                    "training_shuffle_overlay_sha256": sha256_file(shuffle_path),
                    "mapping_path": str(mapping_path),
                    "mapping_sha256": sha256_file(mapping_path),
                },
            )
            rows.append(
                direct.manifest_row(
                    f"intervention_overlay:{fold}:{condition}", destination
                )
            )
    return direct._write_hash_manifest(
        root / "evaluation/intervention_input_hashes.csv", rows
    )


def _parallel_intervention_aggregate(
    root: Path, units: pd.DataFrame
) -> tuple[Path, Path]:
    evidence_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for unit in units.to_dict(orient="records"):
        unit_id = str(unit["prediction_unit_id"])
        cell, artifacts = _parallel_cell_artifacts(root, "interventions", unit_id)
        if set(artifacts) != {
            "prediction",
            "core_prediction_manifest",
            "pair_metrics",
        }:
            raise ValueError(f"Intervention artifact universe drift: {unit_id}")
        evidence = pd.read_csv(artifacts["pair_metrics"]["path"])
        expected_pairs = int(_expected_fold_counts()[str(unit["fold"])][4])
        if len(evidence) != expected_pairs:
            raise ValueError(f"Intervention pair coverage drift: {unit_id}")
        evidence = evidence.copy()
        evidence["input_condition"] = str(unit["input_condition"])
        evidence["source_job_id"] = str(unit["source_job_id"])
        evidence["job_id"] = str(unit["source_job_id"])
        evidence["seed"] = int(unit["seed"])
        evidence["fold"] = str(unit["fold"])
        if set(evidence["noise_bank_profile_sha256"].astype(str)) != {
            str(unit["noise_bank_profile_sha256"])
        }:
            raise ValueError(f"Intervention MC64 lineage drift: {unit_id}")
        evidence_frames.append(evidence)
        manifest_rows.append(
            {
                "prediction_unit_id": unit_id,
                "source_job_id": str(unit["source_job_id"]),
                "seed": int(unit["seed"]),
                "fold": str(unit["fold"]),
                "input_condition": str(unit["input_condition"]),
                "checkpoint_path": str(unit["checkpoint_path"]),
                "checkpoint_sha256": str(unit["checkpoint_sha256"]),
                "prediction_path": str(artifacts["prediction"]["path"]),
                "prediction_sha256": str(artifacts["prediction"]["sha256"]),
                "core_prediction_manifest_path": str(
                    artifacts["core_prediction_manifest"]["path"]
                ),
                "core_prediction_manifest_sha256": str(
                    artifacts["core_prediction_manifest"]["sha256"]
                ),
                "pair_metrics_path": str(artifacts["pair_metrics"]["path"]),
                "pair_metrics_sha256": str(artifacts["pair_metrics"]["sha256"]),
                "noise_bank_profile_sha256": str(unit["noise_bank_profile_sha256"]),
                "physical_gpu_id": int(cell["physical_gpu_id"]),
            }
        )
    evidence = pd.concat(evidence_frames, ignore_index=True)
    if len(evidence) != EXPECTED_INTERVENTION_PAIR_ROWS or len(manifest_rows) != (
        EXPECTED_INTERVENTION_PREDICTIONS
    ):
        raise ValueError("Parallel intervention count drift")
    expected_cells = {
        (seed, fold, condition)
        for seed in SEEDS
        for fold in FOLDS
        for condition in ("zero_input", "wrong_input")
    }
    if (
        set(
            evidence[["seed", "fold", "input_condition"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        != expected_cells
    ):
        raise ValueError("Parallel intervention cell universe drift")
    standard = pd.read_csv(root / "evaluation/standard_pair_metrics.csv.gz")
    matched = standard.loc[standard["arm"].eq("film_lp_matched")]
    profiles = matched.set_index(["seed", "fold", "pair_id"])[
        "noise_bank_profile_sha256"
    ]
    for row in evidence.itertuples(index=False):
        if str(row.noise_bank_profile_sha256) != str(
            profiles.loc[(int(row.seed), str(row.fold), str(row.pair_id))]
        ):
            raise ValueError("Intervention does not reuse matched MC64 noise")
    pair_path = core._write_dataframe_csv(
        root / "evaluation/intervention_pair_metrics.csv.gz",
        evidence.sort_values(
            ["seed", "fold", "input_condition", "pair_id"], kind="stable"
        ),
        gzip=True,
    )
    manifest_path = direct.write_csv(
        root / "evaluation/intervention_prediction_manifest.csv",
        sorted(manifest_rows, key=lambda row: row["prediction_unit_id"]),
        tuple(manifest_rows[0]),
    )
    return manifest_path, pair_path


def predict_interventions(output_dir: str | Path, *, resume: bool = False) -> Path:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_prediction import (
        attach_noise_bank_profiles,
        plan_intervention_prediction_units,
    )

    root = _root(output_dir)
    config = validate_root(root, verify_stage_roots=True)
    registry = _read_registry(root)
    if not registry.get("standard_predictions_frozen"):
        raise RuntimeError("Interventions require the frozen standard MC64 panel")
    pair_path = root / "evaluation/intervention_pair_metrics.csv.gz"
    if registry.get("interventions_frozen"):
        if not pair_path.is_file() or sha256_file(pair_path) != registry.get(
            "intervention_pair_metrics_sha256"
        ):
            raise ValueError("Frozen intervention predictions drift")
        return pair_path
    inputs = _materialize_intervention_overlays(config, root)
    registry = _read_registry(root)
    units = plan_intervention_prediction_units(
        pd.DataFrame(registry["jobs"]), registry["checkpoint_allowlist_path"]
    )
    units = attach_noise_bank_profiles(
        units, sample_ids_by_split_fold=_test_sample_ids(root)
    )
    units["experiment_root"] = str(root)
    units["expected_artifact_roles"] = [
        ["prediction", "core_prediction_manifest", "pair_metrics"]
        for _ in range(len(units))
    ]
    unit_path, execution = _run_parallel_prediction_stage(
        root,
        config,
        units,
        stage="interventions",
        workers_key="prediction_workers_per_gpu",
        resume=resume,
    )
    manifest_path, result = _parallel_intervention_aggregate(root, units)
    registry = _read_registry(root)
    registry.update(
        status="interventions_frozen",
        interventions_frozen=True,
        intervention_inputs_path=str(inputs),
        intervention_inputs_sha256=sha256_file(inputs),
        intervention_prediction_manifest_path=str(manifest_path),
        intervention_prediction_manifest_sha256=sha256_file(manifest_path),
        intervention_pair_metrics_path=str(result),
        intervention_pair_metrics_sha256=sha256_file(result),
        intervention_prediction_cells=EXPECTED_INTERVENTION_PREDICTIONS,
        intervention_pair_metric_rows=EXPECTED_INTERVENTION_PAIR_ROWS,
        intervention_prediction_units_path=str(unit_path),
        intervention_prediction_units_sha256=sha256_file(unit_path),
        intervention_parallel_execution_manifest_path=str(
            (
                root
                / "control/parallel_predictions/interventions/parallel_prediction_execution_manifest.json"
            ).resolve()
        ),
        intervention_parallel_execution_manifest_sha256=sha256_file(
            root
            / "control/parallel_predictions/interventions/parallel_prediction_execution_manifest.json"
        ),
        intervention_parallel_worker_count=int(execution["prediction_unit_count"]),
    )
    _write_registry(root, registry)
    validate_root(root, verify_stage_roots=True)
    return result


def _trajectory_inventory(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage_key in ("pure_continuation", "film_continuations"):
        stage_root = _stage_roots(root)[stage_key]
        registry = direct.read_registry(stage_root)
        for job in registry["jobs"]:
            status_payload = _read_json(
                direct._status_path(stage_root, str(job["job_id"]))
            )
            logical_job = continuation_job_id(
                int(job["seed"]), str(job["fold"]), str(job["arm"])
            )
            best_payload = _read_json(
                Path(
                    str(
                        direct._artifact(status_payload, "best_learned_checkpoint")[
                            "path"
                        ]
                    )
                )
            )
            roles = [
                ("epoch_0", 0, "generator_initial_epoch0"),
                *[
                    (
                        f"epoch_{epoch}",
                        epoch,
                        f"generator_validation_epoch_{epoch:04d}",
                    )
                    for epoch in LEARNED_SNAPSHOT_EPOCHS
                ],
                (
                    "best",
                    int(best_payload["best_epoch"]),
                    "generator_best_learned",
                ),
            ]
            for label, epoch, role in roles:
                artifact = direct._artifact(status_payload, role)
                path = Path(str(artifact["path"])).resolve()
                rows.append(
                    {
                        "job_id": logical_job,
                        "internal_job_id": str(job["job_id"]),
                        "seed": int(job["seed"]),
                        "fold": str(job["fold"]),
                        "arm": str(job["arm"]),
                        "checkpoint_label": label,
                        "epoch": int(epoch),
                        "checkpoint_path": str(path),
                        "checkpoint_sha256": str(artifact["sha256"]),
                        "size_bytes": path.stat().st_size,
                        "stage_root": str(stage_root),
                    }
                )
    frame = pd.DataFrame(rows)
    if (
        len(frame) != EXPECTED_VALIDATION_TRAJECTORY_UNITS
        or frame.duplicated(["job_id", "checkpoint_label"]).any()
    ):
        raise ValueError("Validation trajectory checkpoint inventory drift")
    return frame.sort_values(["seed", "fold", "arm", "checkpoint_label"], kind="stable")


def _validation_pair_ids(root: Path, fold: str) -> tuple[list[str], dict[str, str]]:
    universe = pd.read_csv(
        _stage_roots(root)["backbones"] / "inputs/pair_universes.csv", dtype=str
    )
    selected = universe.loc[
        universe["tolerance_minutes"].astype(int).eq(5)
        & universe["fold"].eq(fold)
        & universe["partition"].eq("validation")
    ].copy()
    pair_ids = sorted(selected["pair_id"].astype(str))
    sessions = dict(
        selected[["pair_id", "session_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    return pair_ids, sessions


def _canonical_validation_rows(
    config: Mapping[str, Any], root: Path
) -> dict[str, dict[str, Any]]:
    required = {
        pair_id for fold in FOLDS for pair_id in _validation_pair_ids(root, fold)[0]
    }
    workbook = resolve_path(config["data"]["root"]) / str(
        config["data"]["workbook_template"]
    ).format(tolerance02="05")
    frame = pd.read_excel(workbook, sheet_name=str(config["data"]["sheet_name"]))
    frame["pair_id"] = frame["pair_id"].astype(str)
    frame = frame.loc[frame["pair_id"].isin(required)].copy()
    if set(frame["pair_id"]) != required:
        raise ValueError("Validation workbook pair coverage drift")
    rows: dict[str, dict[str, Any]] = {}
    for pair_id, pair_rows in frame.groupby("pair_id", sort=False):
        ordered = pair_rows.assign(
            _news_row_sort=pd.to_numeric(pair_rows["news_row_id"], errors="raise")
        ).sort_values(["_news_row_sort", "sample_id"], kind="stable")
        rows[str(pair_id)] = ordered.iloc[0].drop(labels="_news_row_sort").to_dict()
    return rows


def _validation_panel(
    root: Path,
    canonical_rows: Mapping[str, Mapping[str, Any]],
    *,
    fold: str,
    arm: str,
) -> pd.DataFrame:
    pair_ids, sessions = _validation_pair_ids(root, fold)
    stage_root = (
        _stage_roots(root)["pure_continuation"]
        if arm == PURE_CONTINUATION_ARM
        else _stage_roots(root)["film_continuations"]
    )
    overlay = _read_json(direct._overlay_path(stage_root, fold, arm))
    by_pair = {str(row["pair_id"]): row for row in list(overlay.get("records") or [])}
    if not set(pair_ids).issubset(by_pair):
        raise ValueError(f"Validation overlay coverage drift: {fold}/{arm}")
    panel = pd.DataFrame([dict(canonical_rows[pair_id]) for pair_id in pair_ids])
    panel["pair_id"] = panel["pair_id"].astype(str)
    panel["session_id"] = panel["pair_id"].map(sessions)
    panel["lp_embedding"] = panel["pair_id"].map(
        lambda pair_id: json.dumps(by_pair[pair_id]["embedding"], separators=(",", ":"))
    )
    panel["hd_embedding"] = ""
    panel["sample_id"] = panel["pair_id"].map(lambda value: f"pair::{value}")
    panel["sample_weight"] = 1.0
    return panel


def _evaluate_validation_unit(
    unit: Mapping[str, Any], panel: pd.DataFrame, *, gpu_id: int
) -> pd.DataFrame:
    from scripts.rq3.news_first_vol_comparison_analysis import (
        RunSpec,
        TrainedRunEvaluator,
        aggregate_pair_metrics,
        compute_sample_metrics,
    )

    run = RunSpec(
        run_id=str(unit["prediction_unit_id"]),
        run_dir=Path(str(unit["stage_root"])),
        model="wgan",
        tolerance_minutes=5,
        seed=int(unit["seed"]),
        checkpoint_path=Path(str(unit["checkpoint_path"])),
        text_ablation_mode="real_text",
        support_mask_mode="raw_joint",
        generator_current_input_mode="current_support_masked",
        metadata={"fold": unit["fold"], "arm": unit["arm"]},
    )
    evaluator = TrainedRunEvaluator(
        mc_samples=16,
        sample_batch_size=32,
        draw_batch_size=16,
        device=f"cuda:{int(gpu_id)}",
    )
    panel_name = f"{unit['fold']}_validation_05m"
    predictions = evaluator(run, panel_name, panel)
    samples, _general, _metric = compute_sample_metrics(
        run,
        panel_name,
        panel,
        predictions,
        evaluate_embedded_atm_skew=False,
    )
    standard = aggregate_pair_metrics(samples)
    standard = standard.loc[
        standard["stratum_type"].eq("overall") & standard["stratum_value"].eq("all")
    ].copy()
    if len(standard) != len(panel):
        raise ValueError(
            f"Validation prediction coverage drift: {unit['prediction_unit_id']}"
        )
    standard = standard.merge(
        panel[["pair_id", "effective_origin_utc"]].assign(
            pair_id=lambda frame: frame["pair_id"].astype(str)
        ),
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    return pd.DataFrame(
        {
            "prediction_unit_id": str(unit["prediction_unit_id"]),
            "job_id": str(unit["job_id"]),
            "parent_job_id": str(unit["parent_job_id"]),
            "seed": int(unit["seed"]),
            "fold": str(unit["fold"]),
            "arm": str(unit["arm"]),
            "checkpoint_label": str(unit["checkpoint_label"]),
            "epoch": int(unit["epoch"]),
            "pair_id": standard["pair_id"].astype(str),
            "session_id": standard["session_id"].astype(str),
            "effective_origin_utc": standard["effective_origin_utc"].astype(str),
            "target_mae": standard["model_mae"].astype(float),
            "persistence_mae": standard["persistence_mae"].astype(float),
            "checkpoint_sha256": str(unit["checkpoint_sha256"]),
            "noise_bank_profile_sha256": str(unit["noise_bank_profile_sha256"]),
        }
    )


def _parallel_trajectory_aggregate(root: Path, units: pd.DataFrame) -> Path:
    frames: list[pd.DataFrame] = []
    for unit in units.to_dict(orient="records"):
        unit_id = str(unit["prediction_unit_id"])
        _cell, artifacts = _parallel_cell_artifacts(root, "trajectories", unit_id)
        if set(artifacts) != {"pair_metrics", "trajectory_manifest"}:
            raise ValueError(f"Trajectory artifact universe drift: {unit_id}")
        frame = pd.read_csv(artifacts["pair_metrics"]["path"])
        expected_pairs = int(_expected_fold_counts()[str(unit["fold"])][2])
        if (
            len(frame) != expected_pairs
            or set(frame["prediction_unit_id"].astype(str)) != {unit_id}
            or set(frame["checkpoint_sha256"].astype(str))
            != {str(unit["checkpoint_sha256"])}
            or set(frame["noise_bank_profile_sha256"].astype(str))
            != {str(unit["noise_bank_profile_sha256"])}
        ):
            raise ValueError(f"Trajectory pair evidence drift: {unit_id}")
        frames.append(frame)
    result_frame = pd.concat(frames, ignore_index=True)
    if len(result_frame) != EXPECTED_VALIDATION_TRAJECTORY_PAIR_ROWS:
        raise ValueError("Parallel validation trajectory pair-row count drift")
    if result_frame["prediction_unit_id"].nunique() != (
        EXPECTED_VALIDATION_TRAJECTORY_UNITS
    ):
        raise ValueError("Parallel validation trajectory cell universe drift")
    for _key, group in result_frame.groupby(["seed", "fold", "pair_id"], sort=False):
        if group["noise_bank_profile_sha256"].astype(str).nunique() != 1:
            raise ValueError("Validation snapshots do not share paired MC16 noise")
    return core._write_dataframe_csv(
        root / "evaluation/validation_trajectory_pair_metrics.csv.gz",
        result_frame.sort_values(
            ["seed", "fold", "arm", "checkpoint_label", "pair_id"], kind="stable"
        ),
        gzip=True,
    )


def predict_validation_trajectories(
    output_dir: str | Path, *, resume: bool = False
) -> Path:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_prediction import (
        attach_noise_bank_profiles,
        plan_validation_trajectory_units,
    )

    root = _root(output_dir)
    config = validate_root(root, verify_stage_roots=True)
    registry = _read_registry(root)
    if not registry.get("evaluation_frozen"):
        raise RuntimeError(
            "Trajectory prediction requires a frozen checkpoint allowlist"
        )
    output_path = root / "evaluation/validation_trajectory_pair_metrics.csv.gz"
    if registry.get("validation_trajectories_frozen"):
        if not output_path.is_file() or sha256_file(output_path) != registry.get(
            "validation_trajectory_pair_metrics_sha256"
        ):
            raise ValueError("Frozen validation trajectories drift")
        return output_path
    inventory = _trajectory_inventory(root)
    inventory_path = direct.write_csv(
        root / "evaluation/validation_trajectory_checkpoint_inventory.csv",
        inventory.to_dict(orient="records"),
        tuple(inventory.columns),
    )
    jobs = pd.DataFrame(registry["jobs"])
    units = plan_validation_trajectory_units(
        jobs,
        inventory,
        registry["checkpoint_allowlist_path"],
    )
    sample_ids = {
        ("validation", fold): [
            f"pair::{pair_id}" for pair_id in _validation_pair_ids(root, fold)[0]
        ]
        for fold in FOLDS
    }
    units = attach_noise_bank_profiles(units, sample_ids_by_split_fold=sample_ids)
    stage_by_job = inventory.drop_duplicates("job_id").set_index("job_id")["stage_root"]
    units["stage_root"] = units["job_id"].map(stage_by_job)
    canonical = _canonical_validation_rows(config, root)
    panel_cache = {
        (fold, arm): _validation_panel(root, canonical, fold=fold, arm=arm)
        for fold in FOLDS
        for arm in CONTINUATION_ARMS
    }
    panel_paths: dict[tuple[str, str], tuple[Path, str]] = {}
    for key, panel in panel_cache.items():
        fold, arm = key
        path = core._write_dataframe_csv(
            root / f"evaluation/validation_panels/{fold}/{arm}.csv.gz",
            panel,
            gzip=True,
        )
        panel_paths[key] = (path, sha256_file(path))
    units["validation_panel_path"] = [
        str(panel_paths[(str(row.fold), str(row.arm))][0])
        for row in units.itertuples(index=False)
    ]
    units["validation_panel_sha256"] = [
        panel_paths[(str(row.fold), str(row.arm))][1]
        for row in units.itertuples(index=False)
    ]
    units["experiment_root"] = str(root)
    units["expected_artifact_roles"] = [
        ["pair_metrics", "trajectory_manifest"] for _ in range(len(units))
    ]
    unit_manifest_path, execution = _run_parallel_prediction_stage(
        root,
        config,
        units,
        stage="trajectories",
        workers_key="validation_prediction_workers_per_gpu",
        resume=resume,
    )
    result = _parallel_trajectory_aggregate(root, units)
    registry = _read_registry(root)
    registry.update(
        status="validation_trajectories_frozen",
        validation_trajectories_frozen=True,
        validation_trajectory_inventory_path=str(inventory_path),
        validation_trajectory_inventory_sha256=sha256_file(inventory_path),
        validation_trajectory_units_path=str(unit_manifest_path),
        validation_trajectory_units_sha256=sha256_file(unit_manifest_path),
        validation_trajectory_pair_metrics_path=str(result),
        validation_trajectory_pair_metrics_sha256=sha256_file(result),
        validation_trajectory_prediction_units=EXPECTED_VALIDATION_TRAJECTORY_UNITS,
        validation_trajectory_pair_metric_rows=EXPECTED_VALIDATION_TRAJECTORY_PAIR_ROWS,
        validation_parallel_execution_manifest_path=str(
            (
                root
                / "control/parallel_predictions/trajectories/parallel_prediction_execution_manifest.json"
            ).resolve()
        ),
        validation_parallel_execution_manifest_sha256=sha256_file(
            root
            / "control/parallel_predictions/trajectories/parallel_prediction_execution_manifest.json"
        ),
        validation_parallel_worker_count=int(execution["prediction_unit_count"]),
    )
    _write_registry(root, registry)
    validate_root(root, verify_stage_roots=True)
    return result


def _public_root_for_stage(stage_root: Path, stage_name: str) -> Path:
    if stage_name == "backbones":
        main_root = stage_root.parents[1]
        key = "backbones"
    elif stage_name == "pure_continuation":
        main_root = stage_root.parents[2]
        key = "pure_continuation"
    elif stage_name == "film_continuations":
        main_root = stage_root.parents[2]
        key = "film_continuations"
    else:
        raise ValueError(f"Unknown internal stage name: {stage_name}")
    if _stage_roots(main_root)[key] != stage_root:
        raise ValueError(f"Internal stage escaped the public root: {stage_root}")
    return main_root


def _validate_worker_owner_root(main_root: Path) -> None:
    """Validate either the formal root or one pre-formal benchmark root.

    Benchmark workers are permitted only under the two frozen concurrency-root
    names, while the formal root is still absent, and with an exact copy of the
    source config.  This narrow exception does not make benchmark checkpoints
    eligible for the formal graft/allowlist.
    """

    if (main_root / "registry/prepare_manifest.json").is_file():
        validate_root(main_root, verify_stage_roots=False, verify_large_inputs=False)
        return
    config = load_config(DEFAULT_CONFIG)
    formal_root = resolve_path(config["experiment"]["output_root"])
    allowed = {
        formal_root.with_name(f"{formal_root.name}_benchmark_{int(workers)}w_v1")
        for workers in (
            config["runtime"]["benchmark_workers_per_gpu"],
            config["runtime"]["fallback_workers_per_gpu"],
        )
    }
    frozen_config = main_root / "resolved_config.yaml"
    if (
        main_root not in allowed
        or formal_root.exists()
        or not frozen_config.is_file()
        or sha256_file(frozen_config) != sha256_file(config["source_config_path"])
    ):
        raise ValueError(
            f"Worker owner root is neither formal nor benchmark: {main_root}"
        )
    _load_frozen_config(frozen_config)


def worker(
    output_dir: str | Path,
    job_id: str,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> Path:
    """Dispatch an internal worker without weakening its stage-specific profile."""

    supplied = _root(output_dir)
    frozen_config = supplied / "resolved_config.yaml"
    if frozen_config.is_file():
        payload = yaml.safe_load(frozen_config.read_text(encoding="utf-8"))
    else:
        payload = None
    if isinstance(payload, Mapping) and payload.get("internal_stage_name"):
        stage_root = supplied
        stage_name = str(payload["internal_stage_name"])
        main_root = _public_root_for_stage(stage_root, stage_name)
    else:
        main_root = supplied
        candidates: list[tuple[str, Path]] = []
        for key, candidate in _stage_roots(main_root).items():
            registry_path = candidate / "registry/task_registry.json"
            if not registry_path.is_file():
                continue
            stage_registry = direct.read_registry(candidate)
            if any(str(row["job_id"]) == str(job_id) for row in stage_registry["jobs"]):
                stage_name = {
                    "backbones": "backbones",
                    "pure_continuation": "pure_continuation",
                    "film_continuations": "film_continuations",
                }[key]
                candidates.append((stage_name, candidate))
        if len(candidates) != 1:
            raise ValueError(
                f"Worker job must resolve to exactly one internal stage: {job_id}"
            )
        stage_name, stage_root = candidates[0]
    _validate_worker_owner_root(main_root)
    if stage_name == "backbones":
        with _pure_stage_profile(
            main_root=main_root,
            stage_name=stage_name,
            arm=PARENT_ARM,
            use_graft=False,
        ):
            return pure.worker(stage_root, job_id, resume=resume, dry_run=dry_run)
    if stage_name == "pure_continuation":
        with _pure_stage_profile(
            main_root=main_root,
            stage_name=stage_name,
            arm=PURE_CONTINUATION_ARM,
            use_graft=True,
        ):
            return pure.worker(stage_root, job_id, resume=resume, dry_run=dry_run)
    if stage_name == "film_continuations":
        with _film_stage_profile(main_root=main_root, stage_name=stage_name):
            return film_text.worker(stage_root, job_id, resume=resume, dry_run=dry_run)
    raise AssertionError(stage_name)


def benchmark(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle import (
        benchmark as lifecycle_benchmark,
    )

    return lifecycle_benchmark(config_path, output_dir, resume=resume)


def analyze(output_dir: str | Path, *, resume: bool = False) -> Path:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle import (
        analyze as lifecycle_analyze,
    )

    return lifecycle_analyze(output_dir, resume=resume)


def bootstrap(output_dir: str | Path, *, resume: bool = False) -> Path:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle import (
        bootstrap as lifecycle_bootstrap,
    )

    return lifecycle_bootstrap(output_dir, resume=resume)


def report(output_dir: str | Path, *, resume: bool = False) -> Path:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle import (
        report as lifecycle_report,
    )

    return lifecycle_report(output_dir, resume=resume)


def qa(output_dir: str | Path, *, resume: bool = False) -> Path:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle import (
        qa as lifecycle_qa,
    )

    return lifecycle_qa(output_dir, resume=resume)


def status(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle import (
        status as lifecycle_status,
    )

    result = lifecycle_status(output_dir)
    return {
        "experiment_kind": EXPERIMENT_KIND,
        "expected_training_jobs": EXPECTED_TRAINING_JOBS,
        "expected_standard_predictions": EXPECTED_STANDARD_PREDICTIONS,
        "expected_intervention_predictions": EXPECTED_INTERVENTION_PREDICTIONS,
        **result,
    }


def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle import (
        run_pipeline as lifecycle_run_pipeline,
    )

    return lifecycle_run_pipeline(config_path, output_dir, resume=resume)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "benchmark",
            "prepare",
            "worker",
            "launch-backbones",
            "freeze-backbones",
            "graft",
            "launch-continuations",
            "freeze-evaluation",
            "predict",
            "predict-test",
            "predict-interventions",
            "predict-validation-trajectories",
            "analyze",
            "bootstrap",
            "report",
            "qa",
            "status",
            "validate",
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
        result: Any = prepare(args.config, args.output_dir, resume=args.resume)
    elif action == "run-pipeline":
        result = run_pipeline(args.config, args.output_dir, resume=args.resume)
    elif action == "worker":
        if not args.job_id:
            raise SystemExit("worker requires --job-id")
        result = worker(
            args.output_dir,
            args.job_id,
            resume=args.resume,
            dry_run=args.worker_dry_run,
        )
    else:
        handlers = {
            "launch-backbones": launch_backbones,
            "freeze-backbones": freeze_backbones,
            "graft": graft,
            "launch-continuations": launch_continuations,
            "freeze-evaluation": freeze_evaluation,
            "predict": predict_test,
            "predict-test": predict_test,
            "predict-interventions": predict_interventions,
            "predict-validation-trajectories": predict_validation_trajectories,
            "analyze": analyze,
            "bootstrap": bootstrap,
            "report": report,
            "qa": qa,
            "status": status,
            "validate": validate_root,
        }
        result = (
            handlers[action](args.output_dir, resume=args.resume)
            if action
            not in {
                "status",
                "validate",
            }
            else handlers[action](args.output_dir)
        )
    print(
        json.dumps(result, indent=2, sort_keys=True, default=str)
        if isinstance(result, Mapping)
        else result
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EXPECTED_INTERVENTION_PREDICTIONS",
    "EXPECTED_STANDARD_PREDICTIONS",
    "EXPECTED_TRAINING_JOBS",
    "analyze",
    "benchmark",
    "bootstrap",
    "freeze_backbones",
    "freeze_evaluation",
    "graft",
    "launch_backbones",
    "launch_continuations",
    "load_config",
    "main",
    "planned_job_specs",
    "predict_interventions",
    "predict_test",
    "predict_validation_trajectories",
    "prepare",
    "qa",
    "report",
    "run_pipeline",
    "status",
    "validate_config",
    "validate_partial_root",
    "validate_partial_root",
    "validate_root",
    "worker",
]
