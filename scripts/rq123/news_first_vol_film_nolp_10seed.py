"""Fail-closed FiLM/NoLP 10-seed RQ1--RQ3 experiment.

This module owns a branch-local experiment registry.  It never mutates the
older RQ1/RQ2/RQ3 roots and it materializes downstream jobs only after the
parent state or branch recipe they consume has been frozen and hashed.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import fcntl
import gzip as gzip_module
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/rq123/news_first_vol_film_nolp_legacy_10seed_v2.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq123_news_first_vol_film_nolp_legacy_10seed_exact_ttm_rolling_v2"
)
EXPERIMENT_KIND = "rq123_news_first_vol_film_nolp_legacy_10seed_rolling_v2"
INTERPRETATION = "retrospective_rolling_development"

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
TOLERANCES = (5, 30)
FOLDS = ("f1_2023q1", "f2_2023q2", "f3_2023q3", "f4_2023q4")
PARENT_ARM = "parent_current_only"
CONTINUATION_ARM = "continuation_no_text"
TEXT_ARMS_5M = ("lp_matched", "lp_shuffle", "bow", "sentiment")
TEXT_ARMS_30M = ("lp_matched", "lp_shuffle")
ALL_ARMS = (
    PARENT_ARM,
    CONTINUATION_ARM,
    *TEXT_ARMS_5M,
)
PARENT_STAGE = "parents"
CONTINUATION_STAGE = "continuations"
BRANCH_STAGE = "text_branches"
BENCHMARK_STAGE = "benchmark"
MATRIX_SMOKE_STAGE = "matrix_smoke"
RECOVERY_CANARY_STAGE = "recovery_canary"
STAGES = (PARENT_STAGE, CONTINUATION_STAGE, BRANCH_STAGE)
EXPECTED_STAGE_COUNTS = {
    PARENT_STAGE: 80,
    CONTINUATION_STAGE: 80,
    BRANCH_STAGE: 240,
}
EXPECTED_TRAINING_JOBS = 400
EXPECTED_PREDICTION_CELLS = 400
EXPECTED_MATRIX_SMOKE_JOBS = 400
MATRIX_SMOKE_RESULT_KIND = "rq123_full_matrix_one_epoch_smoke_result_v1"
RECOVERY_CANARY_RESULT_KIND = "rq123_parent_continuation_checkpoint_canary_result_v1"
RECOVERY_CANARY_PARENT_JOB_ID = (
    "benchmark_30m_f4_2023q4_seed_202_parent_current_only_01"
)
WAVE_ATTESTATION_KIND = "rq123_validated_training_wave_v1"
ATTEMPT_OWNERSHIP_KIND = "rq123_owned_training_attempt_v1"
INFERENCE_DETERMINISM_KIND = "rq123_inference_determinism_contract_v1"
GENERATOR_MODE = "film_conv_bottleneck_concat_v1"
CRITIC_MODE = "lp_disabled_same_shape_v1"
CAPACITY_PROFILE = "legacy"
WORKER_MODULE = "scripts.rq123.news_first_vol_film_nolp_10seed"
GRID = (1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38)
GRID_FINGERPRINT = "7b72f2d3d12a55999415351ba071f8d87863ed57445f1bbf03543e39f9f186b8"
EXPECTED_PARAMETER_COUNTS = {
    "generator": 4_020_288,
    "critic": 729_157,
    "total": 4_749_445,
}
ARCHITECTURE_PROFILE_SHAPE_FIELDS = (
    "gen_base_channels",
    "gen_text_hidden_dim",
    "gen_text_out_dim",
    "gen_hidden_dim",
    "disc_base_channels",
    "disc_text_hidden_dim",
    "disc_hidden_dim",
)
EXPECTED_ARCHITECTURE_PROFILE_SHA256 = (
    "5285d6c973d66b3b0fa78491bed99a7d707c68b02b8a4a6634ecbbea50c44124"
)
PAIR_MANIFEST_KIND = "pair_text_overlay_manifest_v1"
FULL_STATE_KIND = "news_first_wgan_full_training_state_v1"
FULL_STATE_CONTRACT_KIND = "full_training_state_contract_v1"
FULL_STATE_PHASE = "end_of_epoch_after_scheduler_step"
JOB_ARTIFACT_ROLES = (
    "generator_initial_epoch0",
    "discriminator_initial_epoch0",
    "generator_best_learned",
    "discriminator_best_learned",
    "generator_final",
    "discriminator_final",
    "training_metrics_csv",
    "training_metrics_json",
    "best_learned_checkpoint",
    "resolved_training_config",
    "run_log",
    "pair_text_overlay_manifest",
    "full_state_contract",
)
PREPARE_CONTRACT_KIND = "rq123_prepare_contract_v1"
RUNTIME_CONTRACT_KIND = "rq123_runtime_concurrency_contract_v1"


class BenchmarkConcurrencyGateError(RuntimeError):
    """The 18-worker candidate may safely fall back to 12 workers/GPU."""


SOURCE_CODE_RELATIVE_PATHS = (
    "scripts/_path_setup.py",
    "scripts/rq3/__init__.py",
    "scripts/rq123/news_first_vol_film_nolp_10seed.py",
    "scripts/rq123/news_first_vol_film_nolp_10seed_analysis.py",
    "scripts/rq123/news_first_vol_film_nolp_10seed_report.py",
    "scripts/rq3/news_first_vol_capacity_analysis.py",
    "scripts/rq3/news_first_vol_capacity_report.py",
    "scripts/rq3/news_first_vol_comparison_analysis.py",
    "scripts/rq3/news_first_vol_training.py",
    "scripts/rq3/news_first_vol_training_report.py",
    "scripts/rq2_pair/pair_features.py",
    "scripts/rq2_pair/rq2_pair_experiment.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")


def read_json(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw


def write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def write_yaml(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(dict(value), sort_keys=False), encoding="utf-8")
    os.replace(temporary, target)
    return target


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        raise ValueError(f"Cannot infer columns for empty CSV: {target}")
    columns = tuple(fieldnames or rows[0].keys())
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, "") for column in columns} for row in rows
        )
    os.replace(temporary, target)
    return target


def manifest_row(role: str, path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    return {
        "artifact_role": str(role),
        "path": str(target),
        "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def verify_manifest_rows(
    rows: Sequence[Mapping[str, Any]], *, require_unique_roles: bool = True
) -> None:
    if not rows:
        raise ValueError("Artifact manifest is empty")
    paths = [str(Path(str(row["path"])).resolve()) for row in rows]
    roles = [str(row["artifact_role"]) for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Artifact manifest contains duplicate paths")
    if require_unique_roles and len(roles) != len(set(roles)):
        raise ValueError("Artifact manifest contains duplicate roles")
    for row, normalized in zip(rows, paths, strict=True):
        target = Path(normalized)
        if (
            not target.is_file()
            or target.stat().st_size != int(row["size_bytes"])
            or sha256_file(target) != str(row["sha256"])
        ):
            raise ValueError(f"Artifact hash drift: {target}")


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    verify_manifest_rows(rows)
    return rows


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    source = resolve_path(config_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Unified RQ123 config must be a mapping")
    resolved = deepcopy(raw)
    resolved["source_config_path"] = str(source)
    resolved["source_config_sha256"] = sha256_file(source)
    validate_config(resolved)
    return resolved


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _stage_num_epochs(
    training: Mapping[str, Any],
    stage: str,
    *,
    recipe: Mapping[str, Any] | None = None,
    override: int | None = None,
) -> int:
    """Resolve the epoch cap without changing legacy config serialization.

    Parent jobs may opt into a smaller search horizon while continuation jobs
    retain the historical global cap.  Missing ``parent_num_epochs`` therefore
    preserves the exact behavior of already-frozen v1/v2 configs.
    """

    if override is not None:
        epochs = int(override)
    elif stage in {BENCHMARK_STAGE, MATRIX_SMOKE_STAGE}:
        epochs = 1
    elif recipe is not None:
        epochs = int(recipe["num_epochs"])
    elif stage == PARENT_STAGE:
        epochs = int(training.get("parent_num_epochs", training["num_epochs"]))
    else:
        epochs = int(training["num_epochs"])
    if epochs < 1:
        raise ValueError("Resolved num_epochs must be positive")
    return epochs


def _stage_early_stopping_min_epochs(training: Mapping[str, Any], stage: str) -> int:
    if stage == PARENT_STAGE:
        return int(
            training.get(
                "parent_early_stopping_min_epochs",
                training["early_stopping_min_epochs"],
            )
        )
    return int(training["early_stopping_min_epochs"])


def validate_config(config: Mapping[str, Any]) -> None:
    experiment = _mapping(config.get("experiment"), "experiment")
    data = _mapping(config.get("data"), "data")
    matrix = _mapping(config.get("matrix"), "matrix")
    model = _mapping(config.get("model"), "model")
    training = _mapping(config.get("training"), "training")
    runtime = _mapping(config.get("runtime"), "runtime")
    analysis = _mapping(config.get("analysis"), "analysis")
    if int(experiment.get("schema_version", -1)) != 1:
        raise ValueError("Experiment schema_version must be 1")
    if experiment.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment kind drift")
    if experiment.get("interpretation") != INTERPRETATION:
        raise ValueError("Interpretation label drift")
    if tuple(map(int, data.get("tolerances_minutes", ()))) != TOLERANCES:
        raise ValueError("Tolerances must be [5, 30]")
    if tuple(map(int, data.get("maturity_days_grid", ()))) != GRID:
        raise ValueError("Exact-TTM maturity grid drift")
    if str(data.get("support_mask_mode")) != "raw_joint":
        raise ValueError("support_mask_mode must be raw_joint")
    if int(data.get("embedding_dimension", -1)) != 1024:
        raise ValueError("Embedding dimension must be 1024")
    folds = list(config.get("folds") or [])
    if tuple(str(row.get("id")) for row in folds) != FOLDS:
        raise ValueError("Fold identifiers/order drift")
    for row in folds:
        train_end = pd.Timestamp(row["train_end_utc"])
        val_start = pd.Timestamp(row["validation_start_utc"])
        val_end = pd.Timestamp(row["validation_end_utc"])
        test_start = pd.Timestamp(row["test_start_utc"])
        test_end = pd.Timestamp(row["test_end_utc"])
        if not (train_end == val_start < val_end == test_start < test_end):
            raise ValueError(f"Fold half-open windows drift: {row['id']}")
    if tuple(map(int, matrix.get("seeds", ()))) != SEEDS:
        raise ValueError("Frozen ten-seed universe drift")
    if tuple(matrix.get("parent_arms", ())) != (PARENT_ARM,):
        raise ValueError("Parent arm drift")
    if tuple(matrix.get("continuation_arms", ())) != (CONTINUATION_ARM,):
        raise ValueError("Continuation arm drift")
    if tuple(matrix.get("text_arms_5m", ())) != TEXT_ARMS_5M:
        raise ValueError("5m text arms drift")
    if tuple(matrix.get("text_arms_30m", ())) != TEXT_ARMS_30M:
        raise ValueError("30m text arms drift")
    if int(matrix.get("expected_training_jobs", -1)) != EXPECTED_TRAINING_JOBS:
        raise ValueError("Training job count must be 400")
    if int(matrix.get("expected_prediction_cells", -1)) != EXPECTED_PREDICTION_CELLS:
        raise ValueError("Prediction cell count must be 400")
    if model.get("generator_conditioning_mode") != GENERATOR_MODE:
        raise ValueError("Generator conditioning mode drift")
    if model.get("critic_conditioning_mode") != CRITIC_MODE:
        raise ValueError("Critic conditioning mode drift")
    if model.get("capacity_profile") != CAPACITY_PROFILE:
        raise ValueError("Capacity profile drift")
    observed_parameters = {
        "generator": int(model.get("expected_generator_parameters", -1)),
        "critic": int(model.get("expected_critic_parameters", -1)),
        "total": int(model.get("expected_total_parameters", -1)),
    }
    if observed_parameters != EXPECTED_PARAMETER_COUNTS:
        raise ValueError("Frozen parameter-count contract drift")
    if (
        int(model.get("gen_res_blocks", -1)) != 0
        or int(model.get("disc_res_blocks", -1)) != 0
    ):
        raise ValueError("Legacy residual-block contract drift")
    if float(training.get("initial_learning_rate", math.nan)) != 5e-7:
        raise ValueError("Initial LR must be 5e-7")
    if float(training.get("scheduler_min_lr", math.nan)) != 5e-8:
        raise ValueError("Scheduler floor must be 5e-8")
    if int(training.get("lr_warmup_epochs", 0)) != 0:
        raise ValueError("Unified RQ123 training requires zero LR warmup epochs")
    num_epochs = int(training.get("num_epochs", 0))
    parent_num_epochs = int(training.get("parent_num_epochs", num_epochs))
    if num_epochs < 1:
        raise ValueError("training.num_epochs must be positive")
    if parent_num_epochs < 1 or parent_num_epochs > num_epochs:
        raise ValueError(
            "training.parent_num_epochs must be in [1, training.num_epochs]"
        )
    early_stopping_min_epochs = int(training.get("early_stopping_min_epochs", -1))
    parent_early_stopping_min_epochs = int(
        training.get("parent_early_stopping_min_epochs", early_stopping_min_epochs)
    )
    if early_stopping_min_epochs < 0:
        raise ValueError("training.early_stopping_min_epochs must be non-negative")
    if not 0 <= parent_early_stopping_min_epochs <= parent_num_epochs:
        raise ValueError(
            "training.parent_early_stopping_min_epochs must be in "
            "[0, training.parent_num_epochs]"
        )
    if int(training.get("prediction_mc_samples", -1)) != 64:
        raise ValueError("Frozen test prediction MC must be 64")
    if int(training.get("validation_mc_samples", -1)) != 16:
        raise ValueError("Validation MC must be 16")
    if str(training.get("full_state_schema")) != FULL_STATE_KIND:
        raise ValueError("Full-state kind drift")
    if str(training.get("full_state_phase")) != FULL_STATE_PHASE:
        raise ValueError("Full-state phase drift")
    if int(analysis.get("bootstrap_replicates", -1)) != 10_000:
        raise ValueError("Bootstrap replicates must be 10,000")
    if int(analysis.get("minimum_nonworse_seeds", -1)) != 7:
        raise ValueError("Seed consistency gate must be 7/10")
    if int(analysis.get("minimum_nonworse_folds", -1)) != 3:
        raise ValueError("Fold consistency gate must be 3/4")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != (0, 1):
        raise ValueError("Exactly GPU 0/1 are required")
    if int(runtime.get("benchmark_jobs", -1)) != 36:
        raise ValueError("Benchmark must contain 36 jobs")
    if int(runtime.get("benchmark_workers_per_gpu", -1)) != 18:
        raise ValueError("Benchmark concurrency must be 18/GPU")
    if int(runtime.get("fallback_workers_per_gpu", -1)) != 12:
        raise ValueError("Fallback concurrency must be 12/GPU")
    formal_workers = runtime.get("formal_workers_per_gpu")
    if formal_workers is not None and int(formal_workers) not in (12, 18):
        raise ValueError("Frozen formal concurrency must be null, 12, or 18/GPU")
    architecture_profile_contract(config)


def _validated_formal_root(config: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Reject a config/root mismatch before creating any control artifact."""

    experiment = _mapping(config.get("experiment"), "experiment")
    declared = resolve_path(str(experiment.get("output_root", "")))
    observed = Path(output_dir).resolve()
    if observed != declared:
        raise ValueError(
            "Output root differs from the source config experiment.output_root: "
            f"{observed} != {declared}"
        )
    return observed


def _with_selected_workers(
    config: Mapping[str, Any], workers_per_gpu: int
) -> dict[str, Any]:
    """Return a resolved config with the benchmark decision frozen in-band."""

    selected = int(workers_per_gpu)
    if selected not in (12, 18):
        raise ValueError("Selected workers/GPU must be 12 or 18")
    frozen = deepcopy(dict(config))
    runtime = _mapping(frozen.get("runtime"), "runtime")
    runtime["formal_workers_per_gpu"] = selected
    frozen["runtime"] = runtime
    validate_config(frozen)
    return frozen


def grid_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(config["data"], "data")
    payload = {
        "schema_version": 1,
        "profile": data["surface_grid_profile"],
        "strike_grid": [float(value) for value in data["strike_grid"]],
        "maturity_days_grid": [int(value) for value in data["maturity_days_grid"]],
        "shape": [16, 16],
        "source_grid_fingerprint": GRID_FINGERPRINT,
    }
    payload["grid_contract_sha256"] = payload_sha256(payload)
    return payload


def architecture_profile_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable frozen-width architecture profile.

    Keeping the capacity profile distinct from the broader model contract lets
    explicit-grid checkpoints prove both their tensor widths and their
    conditioning/grid semantics.
    """

    model = _mapping(config["model"], "model")
    payload = {
        "schema_version": 1,
        "capacity_profile": CAPACITY_PROFILE,
        **{field: int(model[field]) for field in ARCHITECTURE_PROFILE_SHAPE_FIELDS},
    }
    digest = payload_sha256(payload)
    if digest != EXPECTED_ARCHITECTURE_PROFILE_SHA256:
        raise ValueError("Frozen architecture-profile contract drift")
    return {**payload, "architecture_profile_sha256": digest}


def model_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    model = _mapping(config["model"], "model")
    training = _mapping(config["training"], "training")
    architecture = architecture_profile_contract(config)
    payload = {
        "schema_version": 2,
        "capacity_profile": CAPACITY_PROFILE,
        "architecture_profile_sha256": architecture["architecture_profile_sha256"],
        "architecture": {
            key: value
            for key, value in architecture.items()
            if key != "architecture_profile_sha256"
        },
        "generator_conditioning_mode": GENERATOR_MODE,
        "critic_conditioning_mode": CRITIC_MODE,
        "critic_normalization_mode": model["critic_normalization_mode"],
        "generator_noise_mode": model["generator_noise_mode"],
        "generator_current_input_mode": model["generator_current_input_mode"],
        "residual_output_mode": model["residual_output_mode"],
        "embedding_dim": int(model["embedding_dim"]),
        "noise_dim": int(model["noise_dim"]),
        "gen_base_channels": int(model["gen_base_channels"]),
        "gen_res_blocks": int(model["gen_res_blocks"]),
        "gen_text_hidden_dim": int(model["gen_text_hidden_dim"]),
        "gen_text_out_dim": int(model["gen_text_out_dim"]),
        "gen_hidden_dim": int(model["gen_hidden_dim"]),
        "disc_base_channels": int(model["disc_base_channels"]),
        "disc_res_blocks": int(model["disc_res_blocks"]),
        "disc_text_hidden_dim": int(model["disc_text_hidden_dim"]),
        "disc_hidden_dim": int(model["disc_hidden_dim"]),
        "generator_parameters": int(model["expected_generator_parameters"]),
        "critic_parameters": int(model["expected_critic_parameters"]),
        "total_parameters": int(model["expected_total_parameters"]),
        "initial_learning_rate": float(training["initial_learning_rate"]),
        "scheduler_min_lr": float(training["scheduler_min_lr"]),
        "grid_contract_sha256": grid_contract(config)["grid_contract_sha256"],
    }
    payload["model_contract_sha256"] = payload_sha256(payload)
    return payload


def _arms_for_tolerance(tolerance: int) -> tuple[str, ...]:
    return TEXT_ARMS_5M if int(tolerance) == 5 else TEXT_ARMS_30M


def block_gpu(tolerance: int, fold: str, seed: int) -> int:
    if tolerance not in TOLERANCES or fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Unknown tolerance/fold/seed block")
    return (TOLERANCES.index(tolerance) + FOLDS.index(fold) + SEEDS.index(seed)) % 2


def experiment_specs(stage: str) -> list[dict[str, Any]]:
    if stage not in (*STAGES, BENCHMARK_STAGE):
        raise ValueError(f"Unknown stage: {stage}")
    specs: list[dict[str, Any]] = []
    if stage == BENCHMARK_STAGE:
        for index in range(36):
            # Pair consecutive jobs across the two devices, then alternate the
            # tolerance inside each device.  This keeps the 18/18 assignment
            # while ensuring both GPUs see both horizons and every arm that is
            # valid for that horizon.  In particular, 30m never references the
            # 5m-only BoW/sentiment overlays.
            gpu = index % 2
            block_index = index // 2
            tolerance = 5 if (block_index + gpu) % 2 == 0 else 30
            occurrence = block_index // 2
            benchmark_arms = (
                (PARENT_ARM, CONTINUATION_ARM, *TEXT_ARMS_5M)
                if tolerance == 5
                else (PARENT_ARM, CONTINUATION_ARM, *TEXT_ARMS_30M)
            )
            arm = benchmark_arms[occurrence % len(benchmark_arms)]
            specs.append(
                {
                    "stage": BENCHMARK_STAGE,
                    "tolerance_minutes": tolerance,
                    "fold": "f4_2023q4",
                    "seed": SEEDS[index % len(SEEDS)],
                    "arm": arm,
                    "benchmark_index": index,
                    "gpu_id": gpu,
                }
            )
        return specs
    for tolerance in TOLERANCES:
        for fold in FOLDS:
            for seed in SEEDS:
                arms: Iterable[str]
                if stage == PARENT_STAGE:
                    arms = (PARENT_ARM,)
                elif stage == CONTINUATION_STAGE:
                    arms = (CONTINUATION_ARM,)
                else:
                    arms = _arms_for_tolerance(tolerance)
                for arm in arms:
                    specs.append(
                        {
                            "stage": stage,
                            "tolerance_minutes": tolerance,
                            "fold": fold,
                            "seed": seed,
                            "arm": arm,
                            "gpu_id": block_gpu(tolerance, fold, seed),
                        }
                    )
    expected = EXPECTED_STAGE_COUNTS[stage]
    if len(specs) != expected:
        raise AssertionError(f"Expected {expected} {stage} specs, got {len(specs)}")
    return specs


def planned_specs() -> list[dict[str, Any]]:
    rows = [row for stage in STAGES for row in experiment_specs(stage)]
    if len(rows) != EXPECTED_TRAINING_JOBS:
        raise AssertionError("Unified task matrix is not exactly 400")
    keys = {
        (
            row["stage"],
            row["tolerance_minutes"],
            row["fold"],
            row["seed"],
            row["arm"],
        )
        for row in rows
    }
    if len(keys) != len(rows):
        raise AssertionError("Unified task matrix contains duplicate cells")
    return rows


def matrix_smoke_specs() -> list[dict[str, Any]]:
    """Return one independent one-epoch smoke cell per formal training cell."""

    rows = [
        {
            **row,
            "formal_stage": str(row["stage"]),
            "stage": MATRIX_SMOKE_STAGE,
        }
        for row in planned_specs()
    ]
    if len(rows) != EXPECTED_MATRIX_SMOKE_JOBS:
        raise AssertionError("Matrix smoke universe is not exactly 400 cells")
    identifiers = [job_id(row) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("Matrix smoke universe contains duplicate job IDs")
    validate_gpu_balance(rows)
    return rows


def validate_gpu_balance(rows: Sequence[Mapping[str, Any]]) -> None:
    blocks: dict[tuple[int, str, int], int] = {}
    for row in rows:
        key = (
            int(row["tolerance_minutes"]),
            str(row["fold"]),
            int(row["seed"]),
        )
        gpu = int(row["gpu_id"])
        previous = blocks.setdefault(key, gpu)
        if previous != gpu:
            raise ValueError(f"Arms in block {key} do not share a physical GPU")
    counts = {gpu: sum(value == gpu for value in blocks.values()) for gpu in (0, 1)}
    if counts != {0: 40, 1: 40}:
        raise ValueError(f"Block GPU assignment must be 40/40, got {counts}")


def job_id(spec: Mapping[str, Any]) -> str:
    benchmark_suffix = (
        f"_{int(spec['benchmark_index']):02d}"
        if spec.get("stage") == BENCHMARK_STAGE
        else ""
    )
    return (
        f"{spec['stage']}_{int(spec['tolerance_minutes']):02d}m_"
        f"{spec['fold']}_seed_{int(spec['seed'])}_{spec['arm']}{benchmark_suffix}"
    )


def assign_waves(
    specs: Sequence[Mapping[str, Any]], *, slots_per_gpu: int
) -> list[dict[str, Any]]:
    if int(slots_per_gpu) <= 0:
        raise ValueError("slots_per_gpu must be positive")
    counts = {0: 0, 1: 0}
    result: list[dict[str, Any]] = []
    for raw in specs:
        row = dict(raw)
        gpu = int(row["gpu_id"])
        local_index = counts[gpu]
        counts[gpu] += 1
        row["gpu_slot"] = local_index % int(slots_per_gpu)
        row["wave"] = local_index // int(slots_per_gpu) + 1
        result.append(row)
    return result


def _fold(config: Mapping[str, Any], fold_id: str) -> dict[str, Any]:
    matches = [dict(row) for row in config["folds"] if row["id"] == fold_id]
    if len(matches) != 1:
        raise KeyError(fold_id)
    return matches[0]


def _expected_counts(
    config: Mapping[str, Any], tolerance: int, fold_id: str
) -> dict[str, int]:
    raw = _mapping(config["expected_pair_session_counts"], "expected counts")
    tolerance_rows = raw.get(tolerance, raw.get(str(tolerance)))
    return {
        key: int(value)
        for key, value in _mapping(tolerance_rows, f"counts.{tolerance}")[
            fold_id
        ].items()
    }


def _validate_five_minute_forecast_horizon(
    frame: pd.DataFrame, *, tolerance_minutes: int
) -> None:
    required = {"current_snapshot_time_utc", "target_snapshot_time_utc"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(
            f"Forecast-horizon inputs are incomplete for {tolerance_minutes}m"
        )
    current = pd.to_datetime(
        frame["current_snapshot_time_utc"], errors="coerce", utc=True
    )
    target = pd.to_datetime(
        frame["target_snapshot_time_utc"], errors="coerce", utc=True
    )
    delta_seconds = (target - current).dt.total_seconds()
    invalid = current.isna() | target.isna() | ~delta_seconds.eq(300.0)
    if invalid.any():
        examples = frame.loc[
            invalid, ["current_snapshot_time_utc", "target_snapshot_time_utc"]
        ].head(3)
        raise ValueError(
            f"{tolerance_minutes}m alignment must still forecast exactly 5 minutes; "
            f"invalid_rows={int(invalid.sum())}, "
            f"examples={examples.to_dict(orient='records')}"
        )


def materialize_pair_universes(config: Mapping[str, Any], root: Path) -> Path:
    output_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    data = _mapping(config["data"], "data")
    dataset_root = resolve_path(data["root"])
    for tolerance in TOLERANCES:
        workbook = dataset_root / str(data["workbook_template"]).format(
            tolerance02=f"{tolerance:02d}"
        )
        support_path = (
            dataset_root / f"tolerance_{tolerance:02d}m/surface_support_audit.csv.gz"
        )
        frame = pd.read_excel(
            workbook,
            sheet_name=str(data["sheet_name"]),
            usecols=[
                "pair_id",
                "session_id",
                "effective_origin_utc",
                "current_snapshot_time_utc",
                "target_snapshot_time_utc",
                "article_id",
                "news_row_id",
                "sample_id",
            ],
        )
        support = pd.read_csv(
            support_path,
            usecols=[
                "pair_id",
                "surface_training_eligible",
                "joint_zero_support",
                "joint_strict_support_cell_count",
                "grid_fingerprint",
            ],
        )
        if set(support["grid_fingerprint"].astype(str)) != {GRID_FINGERPRINT}:
            raise ValueError(f"Grid fingerprint drift for {tolerance}m")
        support["surface_training_eligible"] = support[
            "surface_training_eligible"
        ].astype(bool)
        support["joint_zero_support"] = support["joint_zero_support"].astype(bool)
        support = support.loc[
            support["surface_training_eligible"]
            & ~support["joint_zero_support"]
            & (support["joint_strict_support_cell_count"].astype(int) > 0)
        ].copy()
        support_ids = set(support["pair_id"].astype(str))
        frame["pair_id"] = frame["pair_id"].astype(str)
        frame = frame.loc[frame["pair_id"].isin(support_ids)].copy()
        _validate_five_minute_forecast_horizon(frame, tolerance_minutes=tolerance)
        frame["effective_origin_utc"] = pd.to_datetime(
            frame["effective_origin_utc"], errors="raise", utc=True
        )
        if frame.empty or set(frame["pair_id"]) != support_ids:
            raise ValueError(
                f"Workbook/support pair universe mismatch for {tolerance}m"
            )
        invariants = [
            "session_id",
            "effective_origin_utc",
            "current_snapshot_time_utc",
            "target_snapshot_time_utc",
        ]
        for column in invariants:
            if int(frame.groupby("pair_id", sort=False)[column].nunique().max()) != 1:
                raise ValueError(f"Pair invariant drift: {tolerance}m {column}")
        article_counts = frame.groupby("pair_id", sort=False)["article_id"].nunique()
        row_counts = frame.groupby("pair_id", sort=False).size()
        pairs = frame.sort_values(["effective_origin_utc", "pair_id"]).drop_duplicates(
            "pair_id", keep="first"
        )
        pairs = pairs.merge(
            support[["pair_id", "joint_strict_support_cell_count"]],
            on="pair_id",
            validate="one_to_one",
        )
        for fold_id in FOLDS:
            fold = _fold(config, fold_id)
            train_end = pd.Timestamp(fold["train_end_utc"])
            validation_end = pd.Timestamp(fold["validation_end_utc"])
            test_end = pd.Timestamp(fold["test_end_utc"])
            partition_masks = {
                "train": pairs["effective_origin_utc"] < train_end,
                "validation": (pairs["effective_origin_utc"] >= train_end)
                & (pairs["effective_origin_utc"] < validation_end),
                "test": (pairs["effective_origin_utc"] >= validation_end)
                & (pairs["effective_origin_utc"] < test_end),
            }
            expected = _expected_counts(config, tolerance, fold_id)
            for partition, mask in partition_masks.items():
                selected = pairs.loc[mask].copy()
                pair_count = int(len(selected))
                session_count = int(selected["session_id"].astype(str).nunique())
                if pair_count != expected[f"{partition}_pairs"]:
                    raise ValueError(
                        f"{tolerance}m {fold_id} {partition} pair count drift: "
                        f"{pair_count} != {expected[f'{partition}_pairs']}"
                    )
                if session_count != expected[f"{partition}_sessions"]:
                    raise ValueError(
                        f"{tolerance}m {fold_id} {partition} session count drift: "
                        f"{session_count} != {expected[f'{partition}_sessions']}"
                    )
                ids = sorted(selected["pair_id"].astype(str))
                universe_sha = payload_sha256(ids)
                summary_rows.append(
                    {
                        "tolerance_minutes": tolerance,
                        "fold": fold_id,
                        "partition": partition,
                        "forecast_horizon_minutes": 5,
                        "pair_count": pair_count,
                        "session_count": session_count,
                        "pair_universe_sha256": universe_sha,
                    }
                )
                for row in selected.itertuples(index=False):
                    output_rows.append(
                        {
                            "tolerance_minutes": tolerance,
                            "fold": fold_id,
                            "partition": partition,
                            "pair_id": str(row.pair_id),
                            "session_id": str(row.session_id),
                            "effective_origin_utc": pd.Timestamp(
                                row.effective_origin_utc
                            )
                            .isoformat()
                            .replace("+00:00", "Z"),
                            "current_snapshot_time_utc": str(
                                row.current_snapshot_time_utc
                            ),
                            "target_snapshot_time_utc": str(
                                row.target_snapshot_time_utc
                            ),
                            "joint_support_cells": int(
                                row.joint_strict_support_cell_count
                            ),
                            "article_count": int(article_counts[str(row.pair_id)]),
                            "source_row_count": int(row_counts[str(row.pair_id)]),
                            "pair_universe_sha256": universe_sha,
                        }
                    )
    universe_path = write_csv(root / "inputs/pair_universes.csv", output_rows)
    summary_path = write_csv(root / "inputs/pair_universe_summary.csv", summary_rows)
    manifest = {
        "schema_version": 1,
        "kind": "rq123_pair_universe_manifest_v1",
        "grid_fingerprint": GRID_FINGERPRINT,
        "forecast_horizon_minutes": 5,
        "pair_universes": manifest_row("pair_universes", universe_path),
        "summary": manifest_row("pair_universe_summary", summary_path),
        "row_count": len(output_rows),
        "summary_row_count": len(summary_rows),
    }
    manifest["manifest_sha256"] = payload_sha256(manifest)
    return write_json(root / "inputs/pair_universe_manifest.json", manifest)


def _code_paths() -> list[Path]:
    # Freeze the complete runtime package, not only the handful of modules
    # imported directly by this orchestrator.  Training, checkpoint loading,
    # support reconstruction and result generation all contain lazy imports;
    # a short hand-maintained list allowed those dependencies to drift during
    # a long background run without invalidating resume.
    paths = {resolve_path(relative) for relative in SOURCE_CODE_RELATIVE_PATHS}
    paths.update(path.resolve() for path in (REPO_ROOT / "src").rglob("*.py"))
    paths.update(path.resolve() for path in (REPO_ROOT / "scripts/rq123").rglob("*.py"))
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Frozen runtime code paths are missing: {missing[:8]}")
    return sorted(paths, key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def _source_paths(config: Mapping[str, Any]) -> list[tuple[str, Path]]:
    data = _mapping(config["data"], "data")
    dataset_root = resolve_path(data["root"])
    rows: list[tuple[str, Path]] = [
        ("source_config", resolve_path(config["source_config_path"])),
        ("dataset_manifest", dataset_root / "dataset_output_sha256.txt"),
        ("dataset_validation", dataset_root / "validation_summary.json"),
        ("dataset_news_master", resolve_path(data["news_master_path"])),
        ("sentiment_workbook", resolve_path(data["sentiment_workbook_path"])),
        ("market_jump_pairs", resolve_path(data["market_jump_candidate_pairs"])),
        ("market_jump_episodes", resolve_path(data["market_jump_candidate_episodes"])),
        (
            "market_jump_validation",
            resolve_path(data["market_jump_validation_summary"]),
        ),
    ]
    for tolerance in TOLERANCES:
        rows.extend(
            [
                (
                    f"dataset_workbook_{tolerance:02d}m",
                    dataset_root
                    / str(data["workbook_template"]).format(
                        tolerance02=f"{tolerance:02d}"
                    ),
                ),
                (
                    f"support_audit_{tolerance:02d}m",
                    dataset_root
                    / f"tolerance_{tolerance:02d}m/surface_support_audit.csv.gz",
                ),
            ]
        )
    rows.extend(
        (f"scheduled_calendar_{index + 1}", resolve_path(path))
        for index, path in enumerate(data["scheduled_event_paths"])
    )
    return rows


def _manifest_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    result = write_csv(
        path,
        rows,
        ("artifact_role", "path", "size_bytes", "sha256"),
    )
    verify_manifest_rows([dict(row) for row in rows])
    return result


def _registry_path(root: Path) -> Path:
    return root / "registry/task_registry.json"


def _status_path(root: Path, job_id_value: str) -> Path:
    return root / "registry/job_status" / f"{job_id_value}.json"


def _experiment_status_path(root: Path) -> Path:
    return root / "registry/experiment_status.json"


def read_registry(root: Path) -> dict[str, Any]:
    return read_json(_registry_path(root))


def write_registry(
    root: Path,
    registry: Mapping[str, Any],
    *,
    updated_at_utc: str | None = None,
) -> Path:
    payload = dict(registry)
    payload["updated_at_utc"] = updated_at_utc or utc_now()
    payload["jobs_sha256"] = payload_sha256(payload.get("jobs", []))
    return write_json(_registry_path(root), payload)


def write_experiment_status(root: Path, status: str, **details: Any) -> Path:
    return write_json(
        _experiment_status_path(root),
        {"status": status, "updated_at_utc": utc_now(), **details},
    )


def initial_job_status(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "job_id": job["job_id"],
        "job_spec_sha256": job["job_spec_sha256"],
        "stage": job["stage"],
        "status": "pending",
        "attempt": 0,
        "artifacts": [],
    }


def _job_spec_sha(job: Mapping[str, Any]) -> str:
    return payload_sha256(
        {key: value for key, value in job.items() if key != "job_spec_sha256"}
    )


def _pair_summary_rows(root: Path) -> dict[tuple[int, str, str], dict[str, str]]:
    with (root / "inputs/pair_universe_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return {
        (int(row["tolerance_minutes"]), row["fold"], row["partition"]): row
        for row in rows
    }


def _run_directory(root: Path, spec: Mapping[str, Any]) -> Path:
    leaf = str(spec["arm"])
    if spec.get("stage") == BENCHMARK_STAGE:
        leaf = f"{leaf}_{int(spec['benchmark_index']):02d}"
    return (
        root
        / "runs"
        / str(spec["stage"])
        / f"tolerance_{int(spec['tolerance_minutes']):02d}m"
        / str(spec["fold"])
        / f"seed_{int(spec['seed'])}"
        / leaf
    ).resolve()


def _overlay_manifest_path(root: Path, spec: Mapping[str, Any]) -> Path:
    return (
        root
        / "inputs/pair_text_overlays"
        / f"tolerance_{int(spec['tolerance_minutes']):02d}m"
        / str(spec["fold"])
        / f"{spec['arm']}.json"
    ).resolve()


def _parent_id(spec: Mapping[str, Any]) -> str:
    return job_id({**dict(spec), "stage": PARENT_STAGE, "arm": PARENT_ARM})


def _continuation_id(spec: Mapping[str, Any]) -> str:
    return job_id({**dict(spec), "stage": CONTINUATION_STAGE, "arm": CONTINUATION_ARM})


def build_job(
    config: Mapping[str, Any],
    root: Path,
    spec: Mapping[str, Any],
    *,
    slots_per_gpu: int,
    parent_state: Mapping[str, Any] | None = None,
    recipe: Mapping[str, Any] | None = None,
    num_epochs_override: int | None = None,
    parent_job_id_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from wgan_option.utils.news_first_experiment_core import (
        training_config_payload_sha256,
    )

    fold = _fold(config, str(spec["fold"]))
    tolerance = int(spec["tolerance_minutes"])
    summaries = _pair_summary_rows(root)
    train_summary = summaries[(tolerance, str(spec["fold"]), "train")]
    validation_summary = summaries[(tolerance, str(spec["fold"]), "validation")]
    output_root = _run_directory(root, spec)
    overlay_path = _overlay_manifest_path(root, spec)
    if not overlay_path.is_file():
        raise FileNotFoundError(f"Pair text overlay is not frozen: {overlay_path}")
    model = _mapping(config["model"], "model")
    training = _mapping(config["training"], "training")
    data = _mapping(config["data"], "data")
    architecture = architecture_profile_contract(config)
    model_lineage = model_contract(config)
    grid_lineage = grid_contract(config)
    if num_epochs_override is not None and int(num_epochs_override) < 1:
        raise ValueError("num_epochs_override must be positive")
    dataset_root = resolve_path(data["root"])
    dataset_path = dataset_root / str(data["workbook_template"]).format(
        tolerance02=f"{tolerance:02d}"
    )
    support_path = (
        dataset_root / f"tolerance_{tolerance:02d}m/surface_support_audit.csv.gz"
    )
    stage = str(spec["stage"])
    resolved_num_epochs = _stage_num_epochs(
        training,
        stage,
        recipe=recipe,
        override=num_epochs_override,
    )
    state_mode = {
        PARENT_STAGE: "save_dynamic_v1",
        CONTINUATION_STAGE: "resume_dynamic_v1",
        BRANCH_STAGE: "resume_frozen_lr_replay_v1",
        BENCHMARK_STAGE: "save_dynamic_v1",
        MATRIX_SMOKE_STAGE: "save_dynamic_v1",
        RECOVERY_CANARY_STAGE: "resume_dynamic_v1",
    }[stage]
    if (
        stage not in {PARENT_STAGE, BENCHMARK_STAGE, MATRIX_SMOKE_STAGE}
        and parent_state is None
    ):
        raise ValueError(f"{stage} requires a frozen parent state")
    if stage == BRANCH_STAGE and recipe is None:
        raise ValueError("Text branch requires a frozen continuation recipe")
    default_overlay_modes = {
        PARENT_ARM: "current_only",
        CONTINUATION_ARM: "current_only",
        "lp_matched": "lp_mean_l2",
        "lp_shuffle": "lp_shuffle",
        "bow": "bow1024",
        "sentiment": "sentiment_pad1024",
    }
    declared_overlay_mode = spec.get("pair_text_overlay_mode")
    overlay_mode = (
        str(declared_overlay_mode).strip()
        if declared_overlay_mode is not None
        else default_overlay_modes[str(spec["arm"])]
    )
    if not overlay_mode:
        raise ValueError("pair_text_overlay_mode must be non-empty when declared")
    run_config: dict[str, Any] = {
        "data_path": str(dataset_path.resolve()),
        "sheet_name": str(data["sheet_name"]),
        "text_embedding_mode": "lp",
        "news_first_common_eval_data_path": (
            "" if stage == BRANCH_STAGE else str(dataset_path.resolve())
        ),
        "news_first_train_end_utc": str(fold["train_end_utc"]),
        "news_first_validation_end_utc": (
            str(fold["train_end_utc"])
            if stage == BRANCH_STAGE
            else str(fold["validation_end_utc"])
        ),
        "news_first_data_window_end_utc_exclusive": (
            str(fold["train_end_utc"])
            if stage == BRANCH_STAGE
            else str(fold["validation_end_utc"])
        ),
        "news_first_dataset_tolerance_minutes": tolerance,
        "support_mask_mode": "raw_joint",
        "news_first_text_ablation_mode": "real_text",
        "news_first_text_information_path": str(spec["arm"]),
        "news_first_text_shuffle_seed": int(config["matrix"]["shuffle_seed"]),
        "news_first_pair_text_overlay_mode": overlay_mode,
        "news_first_pair_text_manifest_path": str(overlay_path),
        "news_first_pair_text_manifest_sha256": sha256_file(overlay_path),
        "news_first_pair_text_profile_sha256": read_json(overlay_path)[
            "profile_sha256"
        ],
        "news_first_full_training_state_mode": state_mode,
        "news_first_full_training_state_contract_path": "",
        "news_first_full_training_state_contract_sha256": "",
        "news_first_materialize_validation_loader": stage != BRANCH_STAGE,
        "news_first_materialize_test_loader": False,
        "news_first_capacity_profile": CAPACITY_PROFILE,
        "news_first_capacity_profile_sha256": architecture[
            "architecture_profile_sha256"
        ],
        "news_first_architecture_profile_sha256": architecture[
            "architecture_profile_sha256"
        ],
        "news_first_model_contract_sha256": model_lineage["model_contract_sha256"],
        "news_first_surface_grid_profile": data["surface_grid_profile"],
        "news_first_surface_grid_sha256": grid_lineage["grid_contract_sha256"],
        "news_first_label_reliability_mode": "none",
        "news_first_refit_mode": (
            "frozen_epoch_lr_replay_v1" if stage == BRANCH_STAGE else "none"
        ),
        "news_first_refit_recipe_path": "" if recipe is None else str(recipe["path"]),
        "news_first_refit_recipe_sha256": ""
        if recipe is None
        else str(recipe["sha256"]),
        "strike_bins": 16,
        "maturity_bins": 16,
        "moneyness_min": 0.97,
        "moneyness_max": 1.03,
        "maturity_min_days": 1,
        "maturity_max_days": 38,
        "channels": int(model["channels"]),
        "embedding_dim": int(model["embedding_dim"]),
        "noise_dim": int(model["noise_dim"]),
        "generator_noise_mode": model["generator_noise_mode"],
        "generator_current_input_mode": model["generator_current_input_mode"],
        "generator_conditioning_mode": GENERATOR_MODE,
        "critic_conditioning_mode": CRITIC_MODE,
        "critic_normalization_mode": model["critic_normalization_mode"],
        "gen_base_channels": int(model["gen_base_channels"]),
        "gen_res_blocks": int(model["gen_res_blocks"]),
        "gen_text_hidden_dim": int(model["gen_text_hidden_dim"]),
        "gen_text_out_dim": int(model["gen_text_out_dim"]),
        "gen_hidden_dim": int(model["gen_hidden_dim"]),
        "disc_base_channels": int(model["disc_base_channels"]),
        "disc_res_blocks": int(model["disc_res_blocks"]),
        "disc_text_hidden_dim": int(model["disc_text_hidden_dim"]),
        "disc_hidden_dim": int(model["disc_hidden_dim"]),
        "residual_output_mode": model["residual_output_mode"],
        "learning_rate": float(training["initial_learning_rate"]),
        "generator_learning_rate": float(training["generator_learning_rate"]),
        "discriminator_learning_rate": float(training["discriminator_learning_rate"]),
        "lr_scheduler_type": "none",
        "use_reduce_lr_on_plateau": stage != BRANCH_STAGE,
        "reduce_lr_factor": float(training["reduce_lr_factor"]),
        "reduce_lr_patience": int(training["reduce_lr_patience"]),
        "reduce_lr_min_lr": float(training["scheduler_min_lr"]),
        "num_epochs": resolved_num_epochs,
        "batch_size": int(training["batch_size"]),
        "beta_1": float(training["beta_1"]),
        "beta_2": float(training["beta_2"]),
        "discriminator_iter": int(training["discriminator_steps"]),
        "lambda_gp": float(training["lambda_gp"]),
        "lambda_recon": float(training["lambda_recon"]),
        "lambda_calendar": float(training["lambda_calendar"]),
        "lambda_butterfly": float(training["lambda_butterfly"]),
        "lambda_smooth": float(training["lambda_smooth"]),
        "lambda_delta_shrink": float(training["lambda_delta_shrink"]),
        "use_calendar_constraint": True,
        "use_butterfly_constraint": True,
        "use_smooth_constraint": True,
        "constraint_warmup_epochs": int(training["constraint_warmup_epochs"]),
        "best_checkpoint_metric": str(training["best_checkpoint_metric"]),
        "evaluate_initial_checkpoint": stage != BRANCH_STAGE,
        "baseline_penalty_weight": float(training["baseline_penalty_weight"]),
        "use_early_stopping": stage
        not in {BRANCH_STAGE, BENCHMARK_STAGE, MATRIX_SMOKE_STAGE},
        "early_stopping_patience": int(training["early_stopping_patience"]),
        "early_stopping_min_epochs": _stage_early_stopping_min_epochs(training, stage),
        "early_stopping_min_delta": float(training["early_stopping_min_delta"]),
        "seed": int(spec["seed"]),
        "cuda": True,
        "num_workers": 0,
        "validation_mc_samples": int(training["validation_mc_samples"]),
        "output_root": str(output_root),
        "save_every": 1_000_000,
    }
    # Optional branch-local epoch-0 initialization.  This is deliberately
    # separate from full-state continuation: the trainer restores only G/D
    # weights (and the frozen RNG contract) while Adam and schedulers remain
    # newly constructed.  Historical job specs omit both fields and therefore
    # retain their existing payload hashes and behavior.
    graft_state_path = str(spec.get("graft_state_path", "") or "").strip()
    graft_state_sha256 = str(spec.get("graft_state_sha256", "") or "").strip()
    if bool(graft_state_path) != bool(graft_state_sha256):
        raise ValueError("graft_state_path and graft_state_sha256 must be paired")
    if graft_state_path:
        graft_path = Path(graft_state_path).expanduser().resolve()
        if not graft_path.is_file() or sha256_file(graft_path) != graft_state_sha256:
            raise ValueError(f"Frozen graft-state drift: {graft_path}")
        run_config["news_first_graft_state_path"] = str(graft_path)
        run_config["news_first_graft_state_sha256"] = graft_state_sha256
    snapshot_epochs = tuple(
        int(epoch)
        for epoch in spec.get("validation_snapshot_epochs", ())
    )
    if snapshot_epochs:
        if tuple(sorted(set(snapshot_epochs))) != snapshot_epochs:
            raise ValueError("validation_snapshot_epochs must be sorted and unique")
        if snapshot_epochs[-1] > resolved_num_epochs:
            raise ValueError("A validation snapshot exceeds the job epoch budget")
        run_config["news_first_validation_snapshot_epochs"] = list(snapshot_epochs)
    # New profiles freeze the explicit zero-warmup decision into each job
    # payload.  Older configs omit the field, so their already-frozen job and
    # training-config hashes remain byte-for-byte compatible.
    if "lr_warmup_epochs" in training:
        run_config["lr_warmup_epochs"] = int(training["lr_warmup_epochs"])
    # Branch-local optimizer profiles may declare split Generator parameter
    # groups in the job spec.  Omit these fields entirely for historical specs
    # so their frozen training-config and job hashes remain byte-for-byte
    # compatible with the original uniform-optimizer contract.
    optional_optimizer_fields = (
        "generator_optimizer_profile",
        "generator_text_learning_rate",
        "generator_film_learning_rate",
        "generator_text_min_learning_rate",
        "generator_film_min_learning_rate",
    )
    for field in optional_optimizer_fields:
        declared = spec.get(field, training.get(field))
        if declared is None:
            continue
        run_config[field] = (
            str(declared) if field == "generator_optimizer_profile" else float(declared)
        )
    canonical_config_sha = training_config_payload_sha256(run_config)
    state_path = output_root / "checkpoints/full_training_state_best_learned.pt"
    lineage = {
        "fold_id": str(spec["fold"]),
        "seed": int(spec["seed"]),
        "arm": str(spec["arm"]),
        "architecture_profile_sha256": architecture["architecture_profile_sha256"],
        "model_contract_sha256": model_lineage["model_contract_sha256"],
        "grid_sha256": grid_lineage["grid_contract_sha256"],
        "training_config_payload_sha256": canonical_config_sha,
        "code_sha256": payload_sha256(
            [
                manifest_row(f"code:{path.relative_to(REPO_ROOT)}", path)
                for path in _code_paths()
            ]
        ),
        "dataset_sha256": sha256_file(dataset_path),
        "support_sha256": sha256_file(support_path),
        "pair_universe_sha256": train_summary["pair_universe_sha256"],
        "text_manifest_sha256": sha256_file(overlay_path),
    }
    if graft_state_path:
        lineage["graft_state_sha256"] = graft_state_sha256
    provisional_job = {
        **dict(spec),
        "job_id": job_id(spec),
        "experiment_kind": EXPERIMENT_KIND,
        "architecture_profile_sha256": architecture["architecture_profile_sha256"],
        "model_contract_sha256": model_lineage["model_contract_sha256"],
        "grid_contract_sha256": grid_lineage["grid_contract_sha256"],
        "train_pair_universe_sha256": train_summary["pair_universe_sha256"],
        "validation_pair_universe_sha256": validation_summary["pair_universe_sha256"],
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "support_path": str(support_path.resolve()),
        "support_sha256": sha256_file(support_path),
        "overlay_path": str(overlay_path),
        "overlay_sha256": sha256_file(overlay_path),
        "run_dir": str(output_root),
        "gpu_id": int(spec["gpu_id"]),
        "gpu_slot": int(spec.get("gpu_slot", 0)),
        "wave": int(spec.get("wave", 1)),
        "parent_job_id": (
            ""
            if stage in {PARENT_STAGE, BENCHMARK_STAGE, MATRIX_SMOKE_STAGE}
            else str(parent_job_id_override or _parent_id(spec))
        ),
        "parent_state_path": "" if parent_state is None else str(parent_state["path"]),
        "parent_state_sha256": ""
        if parent_state is None
        else str(parent_state["sha256"]),
        "continuation_job_id": "" if stage != BRANCH_STAGE else _continuation_id(spec),
        "recipe_path": "" if recipe is None else str(recipe["path"]),
        "recipe_sha256": "" if recipe is None else str(recipe["sha256"]),
        "graft_state_path": graft_state_path,
        "graft_state_sha256": graft_state_sha256,
        "validation_snapshot_epochs": list(snapshot_epochs),
        "expected_generator_parameters": EXPECTED_PARAMETER_COUNTS["generator"],
        "expected_critic_parameters": EXPECTED_PARAMETER_COUNTS["critic"],
        "expected_total_parameters": EXPECTED_PARAMETER_COUNTS["total"],
        "canonical_training_config_sha256": canonical_config_sha,
    }
    lineage["job_sha256"] = payload_sha256(provisional_job)
    contract = {
        "schema_version": 1,
        "kind": FULL_STATE_CONTRACT_KIND,
        "mode": state_mode,
        "output_path": str(state_path) if stage != BRANCH_STAGE else "",
        "output_lineage": lineage,
        "input": None,
    }
    if parent_state is not None:
        contract["input"] = {
            "path": str(parent_state["path"]),
            "sha256": str(parent_state["sha256"]),
            "expected_lineage": dict(parent_state["lineage"]),
        }
    contract_path = (
        root / "configs/full_state_contracts" / f"{provisional_job['job_id']}.json"
    ).resolve()
    write_json(contract_path, contract)
    run_config["news_first_full_training_state_contract_path"] = str(contract_path)
    run_config["news_first_full_training_state_contract_sha256"] = sha256_file(
        contract_path
    )
    training_config_path = (
        root / "configs/jobs" / f"{provisional_job['job_id']}.yaml"
    ).resolve()
    write_yaml(training_config_path, run_config)
    job = {
        **provisional_job,
        "training_config_path": str(training_config_path),
        "training_config_sha256": sha256_file(training_config_path),
        "full_state_contract_path": str(contract_path),
        "full_state_contract_sha256": sha256_file(contract_path),
    }
    job["job_spec_sha256"] = _job_spec_sha(job)
    return job, run_config


def _benchmark_result_path(formal_root: Path) -> Path:
    return (
        formal_root.with_name(formal_root.name + "_control") / "benchmark_result.json"
    )


def _recovery_canary_result_path(formal_root: Path) -> Path:
    return (
        formal_root.with_name(formal_root.name + "_control")
        / "parent_continuation_checkpoint_canary.json"
    )


def _recovery_canary_evidence_path(formal_root: Path) -> Path:
    return (
        formal_root.with_name(formal_root.name + "_control")
        / "parent_continuation_checkpoint_canary_evidence.csv"
    )


def _recovery_canary_root(formal_root: Path, workers_per_gpu: int) -> Path:
    return formal_root.with_name(
        formal_root.name + f"_restore_canary_gpu_{int(workers_per_gpu)}"
    ).resolve()


def _recovery_canary_ownership_path(canary_root: Path) -> Path:
    return canary_root / "registry/recovery_canary_ownership.json"


def _matrix_smoke_result_path(formal_root: Path) -> Path:
    return (
        formal_root.with_name(formal_root.name + "_control")
        / "matrix_smoke_result.json"
    )


def _matrix_smoke_evidence_path(formal_root: Path) -> Path:
    return (
        formal_root.with_name(formal_root.name + "_control")
        / "matrix_smoke_evidence.csv"
    )


def _matrix_smoke_storage_gate_path(formal_root: Path) -> Path:
    return (
        formal_root.with_name(formal_root.name + "_control")
        / "matrix_smoke_storage_gate.json"
    )


def _matrix_smoke_root(formal_root: Path, workers_per_gpu: int) -> Path:
    return formal_root.with_name(
        formal_root.name + f"_matrix_smoke_{int(workers_per_gpu)}"
    ).resolve()


def _matrix_smoke_ownership_path(smoke_root: Path) -> Path:
    return smoke_root / "registry/matrix_smoke_ownership.json"


def _write_matrix_smoke_ownership(
    smoke_root: Path,
    formal_root: Path,
    config: Mapping[str, Any],
    *,
    workers_per_gpu: int,
) -> Path:
    payload = {
        "schema_version": 1,
        "kind": "rq123_matrix_smoke_owned_temporary_root_v1",
        "smoke_root": str(smoke_root.resolve()),
        "formal_root": str(formal_root.resolve()),
        "workers_per_gpu": int(workers_per_gpu),
        "source_config_sha256": str(config["source_config_sha256"]),
        "created_at_utc": utc_now(),
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return write_json(_matrix_smoke_ownership_path(smoke_root), payload)


def _load_frozen_config(path: Path) -> dict[str, Any]:
    """Load a resolved config without replacing its original-source lineage."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Resolved config must be a mapping: {path}")
    validate_config(raw)
    source = resolve_path(str(raw.get("source_config_path", "")))
    if not source.is_file() or sha256_file(source) != str(
        raw.get("source_config_sha256", "")
    ):
        raise ValueError("Resolved config source lineage drift")
    return raw


def _runtime_contract_path(root: Path) -> Path:
    return root / "runtime_contract.json"


def _runtime_contract(
    config: Mapping[str, Any],
    *,
    workers_per_gpu: int,
    mode: str,
    benchmark_result_path: Path | None = None,
) -> dict[str, Any]:
    runtime = _mapping(config["runtime"], "runtime")
    selected = int(workers_per_gpu)
    if selected not in (12, 18):
        raise ValueError("Runtime contract workers/GPU must be 12 or 18")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": RUNTIME_CONTRACT_KIND,
        "mode": str(mode),
        "selected_workers_per_gpu": selected,
        "gpu_ids": [int(value) for value in runtime["gpu_ids"]],
        "benchmark_workers_per_gpu": int(runtime["benchmark_workers_per_gpu"]),
        "fallback_workers_per_gpu": int(runtime["fallback_workers_per_gpu"]),
        "source_config_path": str(resolve_path(config["source_config_path"])),
        "source_config_sha256": str(config["source_config_sha256"]),
        "benchmark_result_path": "",
        "benchmark_result_sha256": "",
    }
    if benchmark_result_path is not None:
        result_path = benchmark_result_path.resolve()
        payload["benchmark_result_path"] = str(result_path)
        payload["benchmark_result_sha256"] = sha256_file(result_path)
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def _validate_runtime_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_json(_runtime_contract_path(root))
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    selected = int(config["runtime"]["formal_workers_per_gpu"])
    if (
        payload.get("kind") != RUNTIME_CONTRACT_KIND
        or payload_sha256(unsigned) != payload.get("payload_sha256")
        or int(payload.get("selected_workers_per_gpu", -1)) != selected
        or payload.get("source_config_sha256") != config["source_config_sha256"]
    ):
        raise ValueError("Runtime concurrency contract drift")
    return payload


def _prepare_contract_path(root: Path) -> Path:
    return root / "registry/prepare_contract.json"


def _write_prepare_contract(
    root: Path,
    config: Mapping[str, Any],
    *,
    workers_per_gpu: int,
    mode: str,
    status: str,
    benchmark_result_path: Path | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": PREPARE_CONTRACT_KIND,
        "mode": str(mode),
        "status": str(status),
        "root": str(root.resolve()),
        "source_config_path": str(resolve_path(config["source_config_path"])),
        "source_config_sha256": str(config["source_config_sha256"]),
        "selected_workers_per_gpu": int(workers_per_gpu),
        "benchmark_result_path": "",
        "benchmark_result_sha256": "",
    }
    if benchmark_result_path is not None:
        result_path = benchmark_result_path.resolve()
        payload["benchmark_result_path"] = str(result_path)
        payload["benchmark_result_sha256"] = sha256_file(result_path)
    payload["payload_sha256"] = payload_sha256(payload)
    return write_json(_prepare_contract_path(root), payload)


def _validate_prepare_contract(
    root: Path,
    config: Mapping[str, Any],
    *,
    workers_per_gpu: int,
    mode: str,
    benchmark_result_path: Path | None = None,
) -> dict[str, Any]:
    payload = read_json(_prepare_contract_path(root))
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    expected_result_path = ""
    expected_result_sha = ""
    if benchmark_result_path is not None:
        expected_result_path = str(benchmark_result_path.resolve())
        expected_result_sha = sha256_file(benchmark_result_path)
    if (
        payload.get("kind") != PREPARE_CONTRACT_KIND
        or payload_sha256(unsigned) != payload.get("payload_sha256")
        or payload.get("root") != str(root.resolve())
        or payload.get("mode") != mode
        or payload.get("source_config_sha256") != config["source_config_sha256"]
        or int(payload.get("selected_workers_per_gpu", -1)) != int(workers_per_gpu)
        or payload.get("benchmark_result_path") != expected_result_path
        or payload.get("benchmark_result_sha256") != expected_result_sha
    ):
        raise ValueError("Preparation contract drift")
    return payload


def _validate_prepare_marker_integrity(root: Path) -> dict[str, Any]:
    payload = read_json(_prepare_contract_path(root))
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    result_path = str(payload.get("benchmark_result_path", "")).strip()
    result_sha = str(payload.get("benchmark_result_sha256", "")).strip()
    if (
        payload.get("kind") != PREPARE_CONTRACT_KIND
        or payload.get("status") != "prepared"
        or payload.get("root") != str(root.resolve())
        or payload_sha256(unsigned) != payload.get("payload_sha256")
        or bool(result_path) != bool(result_sha)
    ):
        raise ValueError("Preparation marker integrity drift")
    if result_path and sha256_file(result_path) != result_sha:
        raise ValueError("Preparation marker benchmark result drift")
    return payload


def _validate_job_contract(
    root: Path,
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    hash_cache: dict[Path, str] | None = None,
    validate_large_inputs: bool = True,
) -> dict[str, Any]:
    """Validate one registry cell against its canonical immutable files."""

    cache = {} if hash_cache is None else hash_cache

    def checked_sha(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in cache:
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            cache[resolved] = sha256_file(resolved)
        return cache[resolved]

    job_id_value = str(job["job_id"])
    if _job_spec_sha(job) != job.get("job_spec_sha256"):
        raise ValueError(f"Job spec SHA drift: {job_id_value}")
    architecture = architecture_profile_contract(config)
    expected_architecture_sha = str(architecture["architecture_profile_sha256"])
    expected_model_sha = str(model_contract(config)["model_contract_sha256"])
    expected_grid_sha = str(grid_contract(config)["grid_contract_sha256"])
    if (
        str(job.get("architecture_profile_sha256", "")) != expected_architecture_sha
        or str(job.get("model_contract_sha256", "")) != expected_model_sha
        or str(job.get("grid_contract_sha256", "")) != expected_grid_sha
    ):
        raise ValueError(f"Job architecture/model/grid contract drift: {job_id_value}")
    data = _mapping(config["data"], "data")
    dataset_root = resolve_path(data["root"])
    tolerance = int(job["tolerance_minutes"])
    expected_files = (
        (
            "training config",
            "training_config_path",
            "training_config_sha256",
            (root / "configs/jobs" / f"{job_id_value}.yaml").resolve(),
        ),
        (
            "full-state contract",
            "full_state_contract_path",
            "full_state_contract_sha256",
            (root / "configs/full_state_contracts" / f"{job_id_value}.json").resolve(),
        ),
        (
            "pair overlay",
            "overlay_path",
            "overlay_sha256",
            _overlay_manifest_path(root, job),
        ),
        (
            "dataset",
            "dataset_path",
            "dataset_sha256",
            (
                dataset_root
                / str(data["workbook_template"]).format(tolerance02=f"{tolerance:02d}")
            ).resolve(),
        ),
        (
            "support audit",
            "support_path",
            "support_sha256",
            (
                dataset_root
                / f"tolerance_{tolerance:02d}m/surface_support_audit.csv.gz"
            ).resolve(),
        ),
    )
    for label, path_key, sha_key, canonical in expected_files:
        declared = Path(str(job.get(path_key, ""))).resolve()
        actual_hash = (
            checked_sha(declared)
            if validate_large_inputs or label not in {"dataset", "support audit"}
            else str(job.get(sha_key, ""))
        )
        if declared != canonical or actual_hash != str(job.get(sha_key, "")):
            raise ValueError(f"Job {label} drift: {job_id_value}")
    training_payload = yaml.safe_load(
        Path(str(job["training_config_path"])).read_text(encoding="utf-8")
    )
    if not isinstance(training_payload, Mapping):
        raise ValueError(f"Invalid training config: {job_id_value}")
    expected_training_lineage = {
        "news_first_capacity_profile_sha256": expected_architecture_sha,
        "news_first_architecture_profile_sha256": expected_architecture_sha,
        "news_first_model_contract_sha256": expected_model_sha,
        "news_first_surface_grid_sha256": expected_grid_sha,
    }
    if any(
        str(training_payload.get(field, "")) != value
        for field, value in expected_training_lineage.items()
    ):
        raise ValueError(f"Training architecture/model/grid drift: {job_id_value}")
    expected_root = _run_directory(root, job)
    if Path(str(job.get("run_dir", ""))).resolve() != expected_root:
        raise ValueError(f"Job run directory drift: {job_id_value}")
    status = read_json(_status_path(root, job_id_value))
    if (
        status.get("job_id") != job_id_value
        or status.get("stage") != job["stage"]
        or status.get("job_spec_sha256") != job["job_spec_sha256"]
    ):
        raise ValueError(f"Job status lineage drift: {job_id_value}")
    if status.get("status") == "completed":
        actual_run = Path(str(status.get("run_dir", ""))).resolve()
        if (
            actual_run.parent != expected_root
            or re.fullmatch(r"\d{8}_\d{6}", actual_run.name) is None
            or not actual_run.is_dir()
            or status.get("training_config_sha256") != job["training_config_sha256"]
        ):
            raise ValueError(f"Completed job run lineage drift: {job_id_value}")
    return status


def validate_root(root: Path) -> dict[str, Any]:
    registry = read_registry(root)
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Registry experiment kind drift")
    jobs = list(registry.get("jobs") or [])
    if registry.get("jobs_sha256") != payload_sha256(jobs):
        raise ValueError("Registry job-universe SHA drift")
    identifiers = [str(job["job_id"]) for job in jobs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Registry contains duplicate job IDs")
    config = _load_frozen_config(root / "resolved_config.yaml")
    if read_json(root / "grid_contract.json") != grid_contract(config):
        raise ValueError("Persisted grid contract drift")
    if read_json(root / "model_contract.json") != model_contract(config):
        raise ValueError("Persisted model contract drift")
    hash_cache: dict[Path, str] = {}

    def cached_manifest_row(role: str, path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        digest = hash_cache.get(resolved)
        if digest is None:
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            digest = sha256_file(resolved)
            hash_cache[resolved] = digest
        return {
            "artifact_role": role,
            "path": str(resolved),
            "size_bytes": resolved.stat().st_size,
            "sha256": digest,
        }

    for job in jobs:
        _validate_job_contract(root, config, job, hash_cache=hash_cache)
    manifest = read_json(root / "inputs/pair_universe_manifest.json")
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if payload_sha256(unsigned) != manifest.get("manifest_sha256"):
        raise ValueError("Pair universe manifest self-hash drift")
    for role in ("pair_universes", "summary"):
        verify_manifest_rows([manifest[role]])
    _assert_manifest_equals_current(
        root / "source_hashes.csv",
        [cached_manifest_row(role, path) for role, path in _source_paths(config)],
    )
    _assert_manifest_equals_current(
        root / "code_hashes.csv",
        [
            manifest_row(f"code:{path.relative_to(REPO_ROOT)}", path)
            for path in _code_paths()
        ],
    )
    _assert_manifest_equals_current(
        root / "config_hashes.csv",
        [
            manifest_row("resolved_config", root / "resolved_config.yaml"),
            manifest_row("grid_contract", root / "grid_contract.json"),
            manifest_row("model_contract", root / "model_contract.json"),
            manifest_row("runtime_contract", root / "runtime_contract.json"),
            manifest_row(
                "pair_universe_manifest",
                root / "inputs/pair_universe_manifest.json",
            ),
            manifest_row(
                "pair_text_overlay_hashes",
                root / "inputs/pair_text_overlay_hashes.csv",
            ),
        ],
    )

    stage_dependencies = (
        ("branch_recipes_frozen", "parent_states_frozen"),
        ("evaluation_frozen", "branch_recipes_frozen"),
        ("test_data_opened", "evaluation_frozen"),
        ("predictions_frozen", "test_data_opened"),
        ("terminal_complete", "predictions_frozen"),
    )
    for child, parent in stage_dependencies:
        if bool(registry.get(child)) and not bool(registry.get(parent)):
            raise ValueError(
                f"Registry stage dependency drift: {child} requires {parent}"
            )

    frozen_anchors = (
        (
            "parent_states_frozen",
            "parent_state_allowlist_path",
            "parent_state_allowlist_sha256",
            root / "registry/parent_state_allowlist.csv",
        ),
        (
            "branch_recipes_frozen",
            "branch_recipe_manifest_path",
            "branch_recipe_manifest_sha256",
            root / "registry/branch_recipe_manifest.csv",
        ),
        (
            "evaluation_frozen",
            "checkpoint_allowlist_path",
            "checkpoint_allowlist_sha256",
            root / "registry/evaluation_checkpoint_allowlist.csv",
        ),
        (
            "evaluation_frozen",
            "checkpoint_manifest_path",
            "checkpoint_manifest_sha256",
            root / "analysis/checkpoint_manifest.csv",
        ),
        (
            "evaluation_frozen",
            "inference_determinism_contract_path",
            "inference_determinism_contract_sha256",
            root / "evaluation/inference_determinism_contract.json",
        ),
        (
            "test_data_opened",
            "test_input_manifest_path",
            "test_input_manifest_sha256",
            root / "evaluation/test_input_hashes.csv",
        ),
        (
            "predictions_frozen",
            "prediction_manifest_path",
            "prediction_manifest_sha256",
            root / "analysis/prediction_manifest.csv",
        ),
        (
            "predictions_frozen",
            "pair_metrics_path",
            "pair_metrics_sha256",
            root / "analysis/rq123_pair_metrics.csv.gz",
        ),
    )
    for flag, path_key, sha_key, canonical_path in frozen_anchors:
        if not bool(registry.get(flag)):
            continue
        declared = Path(str(registry.get(path_key, ""))).resolve()
        if (
            declared != canonical_path.resolve()
            or not declared.is_file()
            or sha256_file(declared) != registry.get(sha_key)
        ):
            raise ValueError(f"Frozen registry anchor drift: {path_key}")
    if registry.get("analysis_manifest_path"):
        analysis_manifest = Path(str(registry["analysis_manifest_path"])).resolve()
        if (
            analysis_manifest != _analysis_manifest_path(root).resolve()
            or not analysis_manifest.is_file()
            or sha256_file(analysis_manifest)
            != registry.get("analysis_manifest_sha256")
        ):
            raise ValueError("Frozen analysis manifest anchor drift")
    if registry.get("matrix_smoke_result_path"):
        smoke_result = Path(str(registry["matrix_smoke_result_path"])).resolve()
        if (
            smoke_result != _matrix_smoke_result_path(root).resolve()
            or not smoke_result.is_file()
            or sha256_file(smoke_result) != registry.get("matrix_smoke_result_sha256")
        ):
            raise ValueError("Frozen matrix-smoke result anchor drift")
    if registry.get("recovery_canary_result_path"):
        canary_result = Path(str(registry["recovery_canary_result_path"])).resolve()
        if (
            canary_result != _recovery_canary_result_path(root).resolve()
            or not canary_result.is_file()
            or sha256_file(canary_result)
            != registry.get("recovery_canary_result_sha256")
        ):
            raise ValueError("Frozen recovery-canary result anchor drift")
    _validate_runtime_contract(root, config)
    _validate_prepare_marker_integrity(root)
    return config


def _prepare_hash_manifests(config: Mapping[str, Any], root: Path) -> None:
    source_rows = [manifest_row(role, path) for role, path in _source_paths(config)]
    code_rows = [
        manifest_row(f"code:{path.relative_to(REPO_ROOT)}", path)
        for path in _code_paths()
    ]
    config_rows = [
        manifest_row("resolved_config", root / "resolved_config.yaml"),
        manifest_row("grid_contract", root / "grid_contract.json"),
        manifest_row("model_contract", root / "model_contract.json"),
        manifest_row("runtime_contract", root / "runtime_contract.json"),
        manifest_row(
            "pair_universe_manifest", root / "inputs/pair_universe_manifest.json"
        ),
        manifest_row(
            "pair_text_overlay_hashes",
            root / "inputs/pair_text_overlay_hashes.csv",
        ),
    ]
    _manifest_csv(root / "source_hashes.csv", source_rows)
    _manifest_csv(root / "code_hashes.csv", code_rows)
    _manifest_csv(root / "config_hashes.csv", config_rows)


def _assert_manifest_equals_current(
    manifest_path: Path, expected_rows: Sequence[Mapping[str, Any]]
) -> None:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        observed = [dict(row) for row in csv.DictReader(handle)]
    expected = [dict(row) for row in expected_rows]

    def normalize(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, int, str]]:
        return sorted(
            (
                str(row["artifact_role"]),
                str(Path(str(row["path"])).resolve()),
                int(row["size_bytes"]),
                str(row["sha256"]),
            )
            for row in rows
        )

    if normalize(observed) != normalize(expected):
        raise ValueError(f"Manifest no longer matches current inputs: {manifest_path}")


def _benchmark_completion_sha256(root: Path) -> str:
    jobs = _validate_stage_complete(root, BENCHMARK_STAGE)
    rows: list[dict[str, Any]] = []
    for job in jobs:
        status_path = _status_path(root, str(job["job_id"]))
        status = read_json(status_path)
        rows.append(
            {
                "job_id": str(job["job_id"]),
                "job_spec_sha256": str(job["job_spec_sha256"]),
                "status_sha256": sha256_file(status_path),
                "artifacts": sorted(
                    (
                        str(row["artifact_role"]),
                        str(row["sha256"]),
                        int(row["size_bytes"]),
                    )
                    for row in status["artifacts"]
                ),
            }
        )
    return payload_sha256(sorted(rows, key=lambda row: row["job_id"]))


def _require_benchmark(formal_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    result_path = _benchmark_result_path(formal_root)
    result = read_json(result_path)
    unsigned = {key: value for key, value in result.items() if key != "payload_sha256"}
    selected = int(result.get("selected_workers_per_gpu", -1))
    expected_benchmark_root = formal_root.with_name(
        formal_root.name + f"_benchmark_{selected}"
    ).resolve()
    if (
        result.get("status") != "passed"
        or payload_sha256(unsigned) != result.get("payload_sha256")
        or result.get("config_sha256") != config["source_config_sha256"]
        or selected not in (12, 18)
        or int(result.get("job_count", -1)) != 36
        or Path(str(result.get("benchmark_root", ""))).resolve()
        != expected_benchmark_root
    ):
        raise ValueError("Benchmark result/config contract drift")
    benchmark_root = expected_benchmark_root
    frozen_config = _with_selected_workers(config, selected)
    observed_config = validate_root(benchmark_root)
    if observed_config != frozen_config:
        raise ValueError("Benchmark resolved config differs from the formal config")
    registry = read_registry(benchmark_root)
    if (
        int(registry.get("formal_workers_per_gpu", -1)) != selected
        or len(registry.get("jobs") or []) != 36
        or {str(job.get("stage")) for job in registry.get("jobs") or []}
        != {BENCHMARK_STAGE}
    ):
        raise ValueError("Benchmark registry matrix drift")
    _assert_manifest_equals_current(
        benchmark_root / "source_hashes.csv",
        [manifest_row(role, path) for role, path in _source_paths(frozen_config)],
    )
    _assert_manifest_equals_current(
        benchmark_root / "code_hashes.csv",
        [
            manifest_row(f"code:{path.relative_to(REPO_ROOT)}", path)
            for path in _code_paths()
        ],
    )
    for key, filename in (
        ("benchmark_root_code_manifest_sha256", "code_hashes.csv"),
        ("benchmark_root_source_manifest_sha256", "source_hashes.csv"),
        ("benchmark_root_config_manifest_sha256", "config_hashes.csv"),
    ):
        if result.get(key) != sha256_file(benchmark_root / filename):
            raise ValueError(f"Benchmark result {key} drift")
    completion_sha = _benchmark_completion_sha256(benchmark_root)
    if result.get("benchmark_completion_sha256") != completion_sha:
        raise ValueError("Benchmark completion/artifact digest drift")
    resources = _resource_peaks(benchmark_root)
    if result.get("telemetry_sha256") != resources["telemetry_sha256"]:
        raise ValueError("Benchmark telemetry hash drift")
    _assert_benchmark_updates(benchmark_root)
    return result


def prepare_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> Path:
    source_config = load_config(config_path)
    root = _validated_formal_root(source_config, output_dir)
    benchmark = _require_benchmark(root, source_config)
    recovery_canary = _require_recovery_canary(root, source_config)
    matrix_smoke = _require_matrix_smoke(root, source_config)
    slots = int(benchmark["selected_workers_per_gpu"])
    if int(matrix_smoke["selected_workers_per_gpu"]) != slots:
        raise ValueError("Matrix smoke concurrency differs from the benchmark decision")
    smoke_root = _matrix_smoke_root(root, slots)
    if smoke_root.exists():
        raise RuntimeError(
            "Validated matrix-smoke evidence exists but its temporary root was not "
            "cleaned; rerun the public dry-run action with --resume"
        )
    canary_root = _recovery_canary_root(root, slots)
    if canary_root.exists():
        raise RuntimeError(
            "Validated recovery-canary evidence exists but its temporary root was "
            "not cleaned; rerun the recovery-canary action with --resume"
        )
    config = _with_selected_workers(source_config, slots)
    benchmark_path = _benchmark_result_path(root)
    if root.exists():
        if not resume:
            raise FileExistsError(root)
        try:
            existing = validate_root(root)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            marker = _validate_prepare_contract(
                root,
                config,
                workers_per_gpu=slots,
                mode="formal",
                benchmark_result_path=benchmark_path,
            )
            if marker.get("status") != "preparing":
                raise ValueError("Invalid formal root is not safely resumable")
            registry_path = _registry_path(root)
            if registry_path.is_file():
                partial_registry = read_registry(root)
                for job in partial_registry.get("jobs") or []:
                    status_path = _status_path(root, str(job["job_id"]))
                    if not status_path.is_file():
                        continue
                    status = read_json(status_path)
                    if (
                        status.get("status") != "pending"
                        or int(status.get("attempt", 0)) != 0
                        or status.get("artifacts")
                    ):
                        raise ValueError(
                            "Formal preparation cannot resume after training started"
                        )
        else:
            if existing != config:
                raise ValueError("Existing formal resolved config drift")
            registry = read_registry(root)
            if (
                registry.get("benchmark_result_sha256") != sha256_file(benchmark_path)
                or registry.get("recovery_canary_result_sha256")
                != sha256_file(_recovery_canary_result_path(root))
                or registry.get("matrix_smoke_result_sha256")
                != sha256_file(_matrix_smoke_result_path(root))
                or int(registry.get("formal_workers_per_gpu", -1)) != slots
            ):
                raise ValueError("Existing formal preflight anchor drift")
            return root
    else:
        root.mkdir(parents=True)
    _write_prepare_contract(
        root,
        config,
        workers_per_gpu=slots,
        mode="formal",
        status="preparing",
        benchmark_result_path=benchmark_path,
    )
    for relative in (
        "registry/job_status",
        "configs/jobs",
        "configs/full_state_contracts",
        "inputs/pair_text_overlays",
        "logs",
        "analysis",
        "report",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    write_yaml(root / "resolved_config.yaml", config)
    write_json(root / "grid_contract.json", grid_contract(config))
    write_json(root / "model_contract.json", model_contract(config))
    write_json(
        _runtime_contract_path(root),
        _runtime_contract(
            config,
            workers_per_gpu=slots,
            mode="formal_frozen",
            benchmark_result_path=benchmark_path,
        ),
    )
    materialize_pair_universes(config, root)
    # Pair overlays are materialized before job configs so every job spec can
    # bind an immutable text universe and profile hash.
    materialize_pair_text_overlays(config, root)
    parent_specs = assign_waves(experiment_specs(PARENT_STAGE), slots_per_gpu=slots)
    jobs: list[dict[str, Any]] = []
    for spec in parent_specs:
        job, _ = build_job(config, root, spec, slots_per_gpu=slots)
        jobs.append(job)
        write_json(_status_path(root, job["job_id"]), initial_job_status(job))
    registry = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "interpretation": INTERPRETATION,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "formal_workers_per_gpu": slots,
        "expected_training_jobs": EXPECTED_TRAINING_JOBS,
        "expected_prediction_cells": EXPECTED_PREDICTION_CELLS,
        "jobs": jobs,
        "parent_states_frozen": False,
        "branch_recipes_frozen": False,
        "evaluation_frozen": False,
        "predictions_frozen": False,
        "rq1_evaluated": False,
        "rq2_evaluated": False,
        "rq3_evaluated": False,
        "test_data_opened": False,
        "terminal_complete": False,
        "benchmark_result_path": str(_benchmark_result_path(root).resolve()),
        "benchmark_result_sha256": sha256_file(_benchmark_result_path(root)),
        "recovery_canary_result_path": str(
            _recovery_canary_result_path(root).resolve()
        ),
        "recovery_canary_result_sha256": sha256_file(
            _recovery_canary_result_path(root)
        ),
        "recovery_canary_payload_sha256": recovery_canary["payload_sha256"],
        "matrix_smoke_result_path": str(_matrix_smoke_result_path(root).resolve()),
        "matrix_smoke_result_sha256": sha256_file(_matrix_smoke_result_path(root)),
    }
    write_registry(root, registry)
    write_experiment_status(root, "prepared", registered_jobs=len(jobs))
    _prepare_hash_manifests(config, root)
    _write_prepare_contract(
        root,
        config,
        workers_per_gpu=slots,
        mode="formal",
        status="prepared",
        benchmark_result_path=benchmark_path,
    )
    validate_root(root)
    return root


def materialize_pair_text_overlays(config: Mapping[str, Any], root: Path) -> Path:
    """Create deterministic pair-level embeddings for every fold/arm.

    The implementation is intentionally delegated to the shared runtime core;
    this wrapper freezes a single manifest of the generated files and validates
    exact coverage against ``pair_universes.csv``.
    """

    from wgan_option.utils.news_first_experiment_core import (
        build_pair_text_overlay_manifests,
    )

    generated = build_pair_text_overlay_manifests(
        config=dict(config),
        pair_universe_path=root / "inputs/pair_universes.csv",
        output_dir=root / "inputs/pair_text_overlays",
    )
    paths = [Path(path).resolve() for path in generated]
    expected = sum(6 if tolerance == 5 else 4 for tolerance in TOLERANCES) * len(FOLDS)
    if len(paths) != expected or len(paths) != len(set(paths)):
        raise ValueError(f"Expected {expected} unique pair overlay manifests")
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        payload = read_json(path)
        unsigned = {
            key: value for key, value in payload.items() if key != "profile_sha256"
        }
        if (
            int(payload.get("schema_version", -1)) != 1
            or payload.get("kind") != PAIR_MANIFEST_KIND
            or payload_sha256(unsigned) != payload.get("profile_sha256")
        ):
            raise ValueError(f"Invalid pair overlay manifest: {path}")
        rows.append(manifest_row(f"pair_overlay:{path.relative_to(root)}", path))
    manifest_path = _manifest_csv(root / "inputs/pair_text_overlay_hashes.csv", rows)
    return manifest_path


def status_experiment(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    if not root.exists():
        return {"status": "absent", "root": str(root)}
    registry = read_registry(root)
    counts: dict[str, int] = {}
    for job in registry.get("jobs", []):
        state = str(read_json(_status_path(root, str(job["job_id"]))).get("status"))
        counts[state] = counts.get(state, 0) + 1
    return {
        "status": registry.get("status"),
        "root": str(root),
        "registered_jobs": len(registry.get("jobs", [])),
        "expected_jobs": EXPECTED_TRAINING_JOBS,
        "job_status_counts": counts,
        "parent_states_frozen": bool(registry.get("parent_states_frozen")),
        "branch_recipes_frozen": bool(registry.get("branch_recipes_frozen")),
        "evaluation_frozen": bool(registry.get("evaluation_frozen")),
        "predictions_frozen": bool(registry.get("predictions_frozen")),
        "test_data_opened": bool(registry.get("test_data_opened")),
        "terminal_complete": bool(registry.get("terminal_complete")),
    }


def _pid_is_live(value: object) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _completed_valid(job: Mapping[str, Any], status: Mapping[str, Any]) -> bool:
    artifacts = list(status.get("artifacts") or [])
    expected_root = Path(str(job.get("run_dir", ""))).resolve()
    actual_run = Path(str(status.get("run_dir", ""))).resolve()
    if (
        status.get("status") != "completed"
        or status.get("job_id") != job["job_id"]
        or status.get("stage") != job["stage"]
        or status.get("job_spec_sha256") != job["job_spec_sha256"]
        or status.get("training_config_sha256") != job["training_config_sha256"]
        or actual_run.parent != expected_root
        or re.fullmatch(r"\d{8}_\d{6}", actual_run.name) is None
        or not artifacts
    ):
        return False
    try:
        verify_manifest_rows(artifacts)
    except (FileNotFoundError, KeyError, ValueError):
        return False
    return True


def _artifact_paths_for_job(
    job: Mapping[str, Any], run_dir: Path
) -> list[dict[str, Any]]:
    checkpoint_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    stage = str(job["stage"])
    required: list[tuple[str, Path]] = []
    if stage in {
        PARENT_STAGE,
        CONTINUATION_STAGE,
        BENCHMARK_STAGE,
        MATRIX_SMOKE_STAGE,
        RECOVERY_CANARY_STAGE,
    }:
        required.extend(
            [
                (
                    "generator_best_learned",
                    checkpoint_dir / "generator_best_learned.pt",
                ),
                (
                    "discriminator_best_learned",
                    checkpoint_dir / "discriminator_best_learned.pt",
                ),
                (
                    "best_learned_checkpoint",
                    metrics_dir / "best_learned_checkpoint.json",
                ),
            ]
        )
    if stage in {
        BRANCH_STAGE,
        BENCHMARK_STAGE,
        MATRIX_SMOKE_STAGE,
        RECOVERY_CANARY_STAGE,
    }:
        required.extend(
            [
                ("generator_final", checkpoint_dir / "generator.pt"),
                ("discriminator_final", checkpoint_dir / "discriminator.pt"),
            ]
        )
    if stage in {BENCHMARK_STAGE, MATRIX_SMOKE_STAGE, RECOVERY_CANARY_STAGE}:
        required.extend(
            [
                (
                    "generator_initial_epoch0",
                    checkpoint_dir / "generator_initial_epoch0.pt",
                ),
                (
                    "discriminator_initial_epoch0",
                    checkpoint_dir / "discriminator_initial_epoch0.pt",
                ),
            ]
        )
    required.extend(
        [
            ("training_metrics_csv", metrics_dir / "training_metrics.csv"),
            ("training_metrics_json", metrics_dir / "training_metrics.json"),
            ("resolved_training_config", metrics_dir / "training_resolved_config.yaml"),
            ("run_log", run_dir / "run.log"),
            ("pair_text_overlay_manifest", Path(str(job["overlay_path"]))),
            ("full_state_contract", Path(str(job["full_state_contract_path"]))),
        ]
    )
    if stage in {
        PARENT_STAGE,
        CONTINUATION_STAGE,
        BENCHMARK_STAGE,
        MATRIX_SMOKE_STAGE,
        RECOVERY_CANARY_STAGE,
    }:
        contract = read_json(job["full_state_contract_path"])
        required.append(("full_training_state", Path(str(contract["output_path"]))))
    paths = [str(path.resolve()) for _, path in required]
    roles = [role for role, _ in required]
    if len(paths) != len(set(paths)) or len(roles) != len(set(roles)):
        raise RuntimeError(f"Duplicate artifact path/role for {job['job_id']}")
    missing = [str(path) for _, path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Training completed without required artifacts: {missing}")
    return [manifest_row(role, path) for role, path in required]


def _prune_unselected_training_checkpoints(
    job: Mapping[str, Any], run_dir: Path
) -> None:
    """Retain only the checkpoint roles frozen by the formal protocol."""

    stage = str(job["stage"])
    if stage not in {PARENT_STAGE, CONTINUATION_STAGE, BRANCH_STAGE}:
        return
    checkpoint_dir = (run_dir / "checkpoints").resolve()
    if not checkpoint_dir.is_relative_to(run_dir.resolve()):
        raise RuntimeError(f"Checkpoint directory escaped its run: {checkpoint_dir}")
    names = [
        "generator_initial_epoch0.pt",
        "discriminator_initial_epoch0.pt",
        "generator_best.pt",
        "discriminator_best.pt",
    ]
    if stage in {PARENT_STAGE, CONTINUATION_STAGE}:
        names.extend(("generator.pt", "discriminator.pt"))
    else:
        names.extend(
            (
                "generator_best_learned.pt",
                "discriminator_best_learned.pt",
            )
        )
    candidates = [checkpoint_dir / name for name in names]
    candidates.extend(checkpoint_dir.glob("generator_epoch_*.pt"))
    candidates.extend(checkpoint_dir.glob("discriminator_epoch_*.pt"))
    for candidate in candidates:
        resolved_parent = candidate.parent.resolve()
        if resolved_parent != checkpoint_dir:
            raise RuntimeError(
                f"Refusing to prune an out-of-run checkpoint: {candidate}"
            )
        candidate.unlink(missing_ok=True)


def _validate_instantiated_model(trainer: Any, job: Mapping[str, Any]) -> None:
    if trainer.model is None:
        raise RuntimeError("Trainer did not instantiate a WGAN model")
    generator_count = sum(value.numel() for value in trainer.model.G.parameters())
    critic_count = sum(value.numel() for value in trainer.model.D.parameters())
    if generator_count != int(job["expected_generator_parameters"]):
        raise RuntimeError(f"Generator parameter drift: {generator_count}")
    if critic_count != int(job["expected_critic_parameters"]):
        raise RuntimeError(f"Critic parameter drift: {critic_count}")
    if generator_count + critic_count != int(job["expected_total_parameters"]):
        raise RuntimeError("Total parameter-count drift")
    if trainer.model.G.generator_conditioning_mode != GENERATOR_MODE:
        raise RuntimeError("Generator conditioning mode drift")
    if trainer.model.D.critic_conditioning_mode != CRITIC_MODE:
        raise RuntimeError("Critic conditioning mode drift")


def _attempt_ownership_path(run_dir: Path) -> Path:
    return run_dir / ".rq123_job_attempt.json"


def _write_attempt_ownership(job: Mapping[str, Any], run_dir: Path) -> Path:
    expected_root = Path(str(job["run_dir"])).resolve()
    resolved = run_dir.resolve()
    if (
        resolved.parent != expected_root
        or re.fullmatch(r"\d{8}_\d{6}", resolved.name) is None
        or resolved.is_symlink()
    ):
        raise RuntimeError(f"Training attempt escaped canonical job root: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": ATTEMPT_OWNERSHIP_KIND,
        "job_id": str(job["job_id"]),
        "job_spec_sha256": str(job["job_spec_sha256"]),
        "training_config_sha256": str(job["training_config_sha256"]),
        "canonical_job_root": str(expected_root),
        "attempt_run_dir": str(resolved),
        "created_at_utc": utc_now(),
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return write_json(_attempt_ownership_path(resolved), payload)


def _validate_owned_attempt(job: Mapping[str, Any], run_dir: Path) -> None:
    resolved = run_dir.resolve()
    payload = read_json(_attempt_ownership_path(resolved))
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if (
        run_dir.is_symlink()
        or not resolved.is_dir()
        or resolved.parent != Path(str(job["run_dir"])).resolve()
        or re.fullmatch(r"\d{8}_\d{6}", resolved.name) is None
        or payload.get("kind") != ATTEMPT_OWNERSHIP_KIND
        or payload_sha256(unsigned) != payload.get("payload_sha256")
        or payload.get("job_id") != job["job_id"]
        or payload.get("job_spec_sha256") != job["job_spec_sha256"]
        or payload.get("training_config_sha256") != job["training_config_sha256"]
        or payload.get("canonical_job_root") != str(Path(str(job["run_dir"])).resolve())
        or payload.get("attempt_run_dir") != str(resolved)
    ):
        raise RuntimeError(f"Refusing unowned training-attempt cleanup: {resolved}")


def _cleanup_unselected_job_attempts(
    root: Path,
    job: Mapping[str, Any],
    *,
    keep: Path | None,
) -> None:
    job_root = Path(str(job["run_dir"])).resolve()
    if not job_root.exists():
        return
    if job_root.is_symlink() or not job_root.is_dir():
        raise RuntimeError(f"Invalid canonical job root: {job_root}")
    keep_resolved = None if keep is None else keep.resolve()
    candidates = sorted(job_root.iterdir(), key=lambda path: path.name)
    removable: list[Path] = []
    for candidate in candidates:
        if candidate.name == "checkpoints":
            if candidate.is_symlink() or not candidate.is_dir():
                raise RuntimeError(f"Invalid selected-state directory: {candidate}")
            continue
        if re.fullmatch(r"\d{8}_\d{6}", candidate.name) is None:
            raise RuntimeError(f"Unexpected entry in canonical job root: {candidate}")
        if keep_resolved is not None and candidate.resolve() == keep_resolved:
            continue
        _validate_owned_attempt(job, candidate)
        removable.append(candidate.resolve())
    if not removable:
        return
    journal_path = root / "registry/job_cleanup" / f"{job['job_id']}.json"
    journal = {
        "schema_version": 1,
        "kind": "rq123_selected_attempt_cleanup_v1",
        "job_id": str(job["job_id"]),
        "job_spec_sha256": str(job["job_spec_sha256"]),
        "status": "planned",
        "removed_attempts": [str(path) for path in removable],
        "keep_attempt": "" if keep_resolved is None else str(keep_resolved),
        "created_at_utc": utc_now(),
    }
    journal["payload_sha256"] = payload_sha256(journal)
    write_json(journal_path, journal)
    for candidate in removable:
        _validate_owned_attempt(job, candidate)
        shutil.rmtree(candidate)
    journal["status"] = "completed"
    journal["completed_at_utc"] = utc_now()
    journal["payload_sha256"] = payload_sha256(
        {key: value for key, value in journal.items() if key != "payload_sha256"}
    )
    write_json(journal_path, journal)


def _execute_job(
    job: Mapping[str, Any], *, dry_run: bool
) -> tuple[Path, list[dict[str, Any]]]:
    from wgan_option.config import load_config as load_training_config
    from wgan_option.train_vol_xlsx import VolSurfaceXlsxTrainer

    config_path = Path(str(job["training_config_path"])).resolve()
    if sha256_file(config_path) != job["training_config_sha256"]:
        raise ValueError(f"Training config hash drift: {job['job_id']}")
    config = load_training_config(str(config_path))
    trainer = VolSurfaceXlsxTrainer(config, config_path=str(config_path))
    trainer._ensure_runtime_prepared()
    if trainer.run_dir is None:
        raise RuntimeError("Trainer did not materialize an attempt run directory")
    _write_attempt_ownership(job, trainer.run_dir)
    result = trainer.dry_run() if dry_run else trainer.train()
    if result is None:
        raise RuntimeError("Trainer returned no run directory")
    _validate_instantiated_model(trainer, job)
    run_dir = Path(result).resolve()
    _validate_owned_attempt(job, run_dir)
    if dry_run:
        configured_root = Path(str(job["run_dir"])).resolve()
        if run_dir == configured_root or not run_dir.is_relative_to(configured_root):
            raise RuntimeError(f"Dry-run output escaped its job root: {run_dir}")
        # A trainer dry run creates only a timestamped config/log directory.
        # It is a smoke artifact rather than a selected research artifact, so
        # remove that exact leaf after model/data validation succeeds.
        shutil.rmtree(run_dir)
        return run_dir, []
    artifacts = [] if dry_run else _artifact_paths_for_job(job, run_dir)
    # All selected artifacts have already been existence/hash checked by
    # `_artifact_paths_for_job`.  Remove only the explicit non-selected
    # checkpoint copies produced by the generic trainer.
    _prune_unselected_training_checkpoints(job, run_dir)
    return run_dir, artifacts


def _find_job(root: Path, job_id_value: str) -> dict[str, Any]:
    matches = [
        dict(job)
        for job in read_registry(root).get("jobs", [])
        if str(job["job_id"]) == str(job_id_value)
    ]
    if len(matches) != 1:
        raise KeyError(job_id_value)
    return matches[0]


def _wave_attestation_path(root: Path, wave: int) -> Path:
    return root / "registry/wave_attestations" / f"wave_{int(wave):04d}.json"


def _write_wave_attestation(
    root: Path,
    *,
    wave: int,
    jobs: Sequence[Mapping[str, Any]],
) -> Path:
    registry = read_registry(root)
    payload = {
        "schema_version": 1,
        "kind": WAVE_ATTESTATION_KIND,
        "root": str(root.resolve()),
        "wave": int(wave),
        "jobs_sha256": str(registry["jobs_sha256"]),
        "job_ids": sorted(str(job["job_id"]) for job in jobs),
        "source_manifest_sha256": sha256_file(root / "source_hashes.csv"),
        "code_manifest_sha256": sha256_file(root / "code_hashes.csv"),
        "config_manifest_sha256": sha256_file(root / "config_hashes.csv"),
        "created_at_utc": utc_now(),
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return write_json(_wave_attestation_path(root, wave), payload)


def _validated_wave_job(
    root: Path, job_id_value: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use a supervisor-issued attestation to avoid rehashing raw workbooks."""

    raw_path = os.environ.get("RQ123_WAVE_ATTESTATION_PATH", "").strip()
    token = os.environ.get("RQ123_WAVE_ATTESTATION_SHA256", "").strip()
    if not raw_path or not token:
        config = validate_root(root)
        return config, _find_job(root, job_id_value)
    path = Path(raw_path).resolve()
    if not path.is_relative_to((root / "registry/wave_attestations").resolve()):
        raise ValueError("Wave attestation escaped its canonical directory")
    payload = read_json(path)
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    registry = read_registry(root)
    expected_manifest_hashes = {
        "source_manifest_sha256": sha256_file(root / "source_hashes.csv"),
        "code_manifest_sha256": sha256_file(root / "code_hashes.csv"),
        "config_manifest_sha256": sha256_file(root / "config_hashes.csv"),
    }
    if (
        payload.get("kind") != WAVE_ATTESTATION_KIND
        or payload_sha256(unsigned) != payload.get("payload_sha256")
        or payload.get("payload_sha256") != token
        or sha256_file(path) != os.environ.get("RQ123_WAVE_ATTESTATION_FILE_SHA256")
        or payload.get("root") != str(root.resolve())
        or payload.get("jobs_sha256") != registry.get("jobs_sha256")
        or job_id_value not in set(payload.get("job_ids") or [])
        or any(
            payload.get(key) != value for key, value in expected_manifest_hashes.items()
        )
    ):
        raise ValueError("Wave attestation drift")
    config = _load_frozen_config(root / "resolved_config.yaml")
    job = _find_job(root, job_id_value)
    # The supervisor performed one complete root validation immediately before
    # issuing the attestation.  Workers still verify their own immutable files,
    # status lineage, and actual hashes before acquiring the job lock.
    _validate_job_contract(root, config, job, validate_large_inputs=False)
    return config, job


def _stage_is_open(registry: Mapping[str, Any], stage: str) -> bool:
    if stage == PARENT_STAGE:
        return not bool(registry.get("parent_states_frozen"))
    if stage == CONTINUATION_STAGE:
        return bool(registry.get("parent_states_frozen")) and not bool(
            registry.get("branch_recipes_frozen")
        )
    if stage == BRANCH_STAGE:
        return bool(registry.get("branch_recipes_frozen")) and not bool(
            registry.get("evaluation_frozen")
        )
    if stage == BENCHMARK_STAGE:
        return True
    if stage == MATRIX_SMOKE_STAGE:
        return True
    if stage == RECOVERY_CANARY_STAGE:
        return True
    return False


def _acquire_job_lock(root: Path, job_id_value: str) -> tuple[Path, int]:
    lock = root / "registry/job_locks" / f"{job_id_value}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError(f"Duplicate live job: {job_id_value}") from exc
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.fsync(descriptor)
    return lock, descriptor


def _release_file_lock(path: Path, descriptor: int) -> None:
    del path
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def run_worker(
    output_dir: str | Path,
    job_id_value: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    root = Path(output_dir).resolve()
    _, job = _validated_wave_job(root, job_id_value)
    lock_path, lock_descriptor = _acquire_job_lock(root, job_id_value)
    try:
        registry = read_registry(root)
        if bool(registry.get("test_data_opened")):
            raise RuntimeError("Training is forbidden after test data opens")
        if not _stage_is_open(registry, str(job["stage"])):
            raise RuntimeError(f"Training stage is not open: {job['stage']}")
        status_path = _status_path(root, job_id_value)
        previous = read_json(status_path)
        if _completed_valid(job, previous):
            if resume and not dry_run:
                return Path(str(previous["run_dir"]))
            raise RuntimeError(f"Job already completed: {job_id_value}")
        if previous.get("status") in {"running", "failed"} and not (resume or dry_run):
            raise RuntimeError(f"Interrupted job requires --resume: {job_id_value}")
        running = {
            **initial_job_status(job),
            "status": "running",
            "attempt": int(previous.get("attempt", 0)) + 1,
            "dry_run": bool(dry_run),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at_utc": utc_now(),
            "training_config_sha256": job["training_config_sha256"],
        }
        write_json(status_path, running)
        try:
            _cleanup_unselected_job_attempts(root, job, keep=None)
            run_dir, artifacts = _execute_job(job, dry_run=dry_run)
            if not dry_run:
                _cleanup_unselected_job_attempts(root, job, keep=run_dir)
            completed = {
                **running,
                "status": "dry_run_passed" if dry_run else "completed",
                "run_dir": str(run_dir),
                "artifacts": artifacts,
                "completed_at_utc": utc_now(),
            }
            write_json(status_path, completed)
            return run_dir
        except BaseException as exc:
            write_json(
                status_path,
                {
                    **running,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "completed_at_utc": utc_now(),
                },
            )
            raise
    finally:
        _release_file_lock(lock_path, lock_descriptor)


def _worker_command(
    root: Path,
    job: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    dry_run: bool,
    resume: bool,
) -> list[str]:
    command = [
        str(config["runtime"]["python_executable"]),
        "-m",
        WORKER_MODULE,
        "worker",
        "--config",
        str(config["source_config_path"]),
        "--output-dir",
        str(root),
        "--job-id",
        str(job["job_id"]),
    ]
    if dry_run:
        command.append("--worker-dry-run")
    if resume:
        command.append("--resume")
    return command


def _host_ram_fraction() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
    total = int(values.get("MemTotal", 0))
    return 1.0 - int(values.get("MemAvailable", 0)) / total if total else 1.0


def _terminate_processes(processes: Sequence[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (AttributeError, ProcessLookupError, OSError):
                process.terminate()
    deadline = time.monotonic() + 10.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (AttributeError, ProcessLookupError, OSError):
                process.kill()


def _run_wave(
    root: Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    wave: int,
    dry_run: bool,
    resume: bool,
) -> float:
    if not jobs:
        return _host_ram_fraction()
    from scripts.rq3 import news_first_vol_training as training_orchestrator

    config = validate_root(root)
    attestation = _write_wave_attestation(root, wave=wave, jobs=jobs)
    attestation_payload = read_json(attestation)
    runtime = _mapping(config["runtime"], "runtime")
    monitor = training_orchestrator._ResourceMonitor(
        root / "resource_usage.csv",
        executable=str(runtime["nvidia_smi_executable"]),
        interval_seconds=float(runtime["resource_sample_interval_seconds"]),
        wave=int(wave),
    )
    processes: list[subprocess.Popen[Any]] = []
    handles: list[Any] = []
    peak_host = _host_ram_fraction()
    try:
        monitor.start()
        for job in jobs:
            previous = read_json(_status_path(root, str(job["job_id"])))
            log_path = (
                root
                / "logs"
                / f"{job['job_id']}.attempt_{int(previous.get('attempt', 0)) + 1:02d}.log"
            )
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
            environment["NEWS_FIRST_JOB_LOG_PATH"] = str(log_path.resolve())
            environment["RQ123_WAVE_ATTESTATION_PATH"] = str(attestation.resolve())
            environment["RQ123_WAVE_ATTESTATION_SHA256"] = str(
                attestation_payload["payload_sha256"]
            )
            environment["RQ123_WAVE_ATTESTATION_FILE_SHA256"] = sha256_file(attestation)
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(REPO_ROOT / "src"), str(REPO_ROOT))
            )
            threads = str(int(runtime["cpu_threads_per_job"]))
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment[name] = threads
            processes.append(
                subprocess.Popen(
                    _worker_command(
                        root,
                        job,
                        config=config,
                        dry_run=dry_run,
                        resume=resume,
                    ),
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        pending = set(range(len(processes)))
        while pending:
            peak_host = max(peak_host, _host_ram_fraction())
            for index in tuple(pending):
                return_code = processes[index].poll()
                if return_code is None:
                    continue
                pending.remove(index)
                if return_code != 0:
                    raise RuntimeError(
                        f"Wave {wave} job {jobs[index]['job_id']} exited {return_code}"
                    )
            if pending:
                time.sleep(0.5)
    finally:
        # KeyboardInterrupt, SIGTERM translated by the supervisor, and any
        # unexpected exception must not leave detached worker sessions alive.
        _terminate_processes(processes)
        monitor.stop()
        for handle in handles:
            handle.close()
    return peak_host


def _stage_jobs(root: Path, stage: str) -> list[dict[str, Any]]:
    jobs = [
        dict(job)
        for job in read_registry(root).get("jobs", [])
        if str(job["stage"]) == stage
    ]
    expected = EXPECTED_STAGE_COUNTS.get(
        stage,
        36
        if stage == BENCHMARK_STAGE
        else EXPECTED_MATRIX_SMOKE_JOBS
        if stage == MATRIX_SMOKE_STAGE
        else 1
        if stage == RECOVERY_CANARY_STAGE
        else -1,
    )
    if len(jobs) != expected:
        raise ValueError(
            f"Expected {expected} registered {stage} jobs, got {len(jobs)}"
        )
    return jobs


def _launch_stage(root: Path, stage: str, *, dry_run: bool, resume: bool) -> float:
    validate_root(root)
    jobs = _stage_jobs(root, stage)
    peak_host = 0.0
    for wave in sorted({int(job["wave"]) for job in jobs}):
        selected: list[dict[str, Any]] = []
        for job in jobs:
            if int(job["wave"]) != wave:
                continue
            status = read_json(_status_path(root, str(job["job_id"])))
            if not dry_run and _completed_valid(job, status):
                if resume:
                    continue
                raise RuntimeError(f"Completed job requires --resume: {job['job_id']}")
            if dry_run and status.get("status") == "dry_run_passed" and resume:
                continue
            selected.append(job)
        peak_host = max(
            peak_host,
            _run_wave(root, selected, wave=wave, dry_run=dry_run, resume=resume),
        )
    failures: list[str] = []
    for job in jobs:
        status = read_json(_status_path(root, str(job["job_id"])))
        valid = (
            status.get("status") == "dry_run_passed"
            if dry_run
            else _completed_valid(job, status)
        )
        if not valid:
            failures.append(f"{job['job_id']}={status.get('status')}")
    if failures:
        raise RuntimeError(f"Incomplete {stage} stage: {failures[:8]}")
    return peak_host


def dry_run(output_dir: str | Path, *, resume: bool = False) -> Path:
    root = Path(output_dir).resolve()
    _launch_stage(root, PARENT_STAGE, dry_run=True, resume=resume)
    return root


def launch_parents(output_dir: str | Path, *, resume: bool = False) -> Path:
    root = Path(output_dir).resolve()
    registry = read_registry(root)
    if registry.get("parent_states_frozen"):
        _validate_parent_state_allowlist(root)
        return root
    _launch_stage(root, PARENT_STAGE, dry_run=False, resume=resume)
    _freeze_parent_states_and_materialize_continuations(root)
    return root


def _artifact(status: Mapping[str, Any], role: str) -> dict[str, Any]:
    matches = [
        dict(row) for row in status.get("artifacts", []) if row["artifact_role"] == role
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one artifact role {role!r}")
    verify_manifest_rows(matches)
    return matches[0]


def _validate_stage_complete(root: Path, stage: str) -> list[dict[str, Any]]:
    jobs = _stage_jobs(root, stage)
    for job in jobs:
        status = read_json(_status_path(root, str(job["job_id"])))
        if not _completed_valid(job, status):
            raise ValueError(f"Incomplete or drifted job: {job['job_id']}")
    return jobs


def _parent_state_allowlist_path(root: Path) -> Path:
    return root / "registry/parent_state_allowlist.csv"


def _validated_full_state_payload(
    job: Mapping[str, Any], status: Mapping[str, Any]
) -> dict[str, Any]:
    """Deeply validate a selected continuation state before freezing lineage."""

    import torch

    state = _artifact(status, "full_training_state")
    contract_path = Path(str(job["full_state_contract_path"])).resolve()
    if sha256_file(contract_path) != str(job["full_state_contract_sha256"]):
        raise ValueError(f"Full-state contract hash drift: {job['job_id']}")
    contract = read_json(contract_path)
    payload = torch.load(state["path"], map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Full training state root is invalid: {job['job_id']}")
    required = {
        "generator_state_dict",
        "discriminator_state_dict",
        "generator_optimizer_state_dict",
        "discriminator_optimizer_state_dict",
        "generator_scheduler_state_dict",
        "discriminator_scheduler_state_dict",
        "rng_state",
    }
    rng = payload.get("rng_state")
    required_rng = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "loader_generator",
    }
    best = read_json(_artifact(status, "best_learned_checkpoint")["path"])
    completed_epoch = int(payload.get("completed_epoch", 0))
    if (
        int(payload.get("schema_version", -1)) != 1
        or payload.get("kind") != FULL_STATE_KIND
        or payload.get("save_phase") != FULL_STATE_PHASE
        or payload.get("contract_sha256") != str(job["full_state_contract_sha256"])
        or payload.get("lineage") != contract.get("output_lineage")
        or not required.issubset(payload)
        or not isinstance(rng, Mapping)
        or not required_rng.issubset(rng)
        or completed_epoch < 1
        or completed_epoch != int(best.get("best_epoch", 0))
    ):
        raise ValueError(f"Full training state contract/payload drift: {job['job_id']}")
    for key in required - {"rng_state"}:
        if not isinstance(payload[key], Mapping):
            raise ValueError(
                f"Full training state field {key!r} is invalid: {job['job_id']}"
            )
    if not isinstance(rng["torch_cuda"], list):
        raise ValueError(f"Full training state CUDA RNG is invalid: {job['job_id']}")
    return dict(payload)


def _validate_parent_state_allowlist(root: Path) -> dict[str, dict[str, Any]]:
    path = _parent_state_allowlist_path(root)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != EXPECTED_STAGE_COUNTS[PARENT_STAGE]:
        raise ValueError("Parent state allowlist must contain exactly 80 rows")
    jobs = {
        str(job["job_id"]): dict(job)
        for job in read_registry(root).get("jobs", [])
        if str(job["stage"]) == PARENT_STAGE
    }
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        state_path = Path(row["path"])
        if (
            row["parent_job_id"] in result
            or not state_path.is_file()
            or state_path.stat().st_size != int(row["size_bytes"])
            or sha256_file(state_path) != row["sha256"]
        ):
            raise ValueError("Parent state allowlist drift")
        lineage = json.loads(row["lineage_json"])
        if not isinstance(lineage, dict):
            raise ValueError("Parent state lineage must be a mapping")
        job = jobs.get(row["parent_job_id"])
        if job is None:
            raise ValueError("Parent state allowlist references an unknown job")
        status = read_json(_status_path(root, row["parent_job_id"]))
        payload = _validated_full_state_payload(job, status)
        if payload["lineage"] != lineage:
            raise ValueError("Parent state allowlist lineage/payload drift")
        result[row["parent_job_id"]] = {
            "path": str(state_path.resolve()),
            "sha256": row["sha256"],
            "lineage": lineage,
        }
    return result


def _freeze_parent_states_and_materialize_continuations(root: Path) -> Path:
    config = validate_root(root)
    registry = read_registry(root)
    if registry.get("parent_states_frozen"):
        _validate_parent_state_allowlist(root)
        return _parent_state_allowlist_path(root)
    parents = _validate_stage_complete(root, PARENT_STAGE)
    rows: list[dict[str, Any]] = []
    parent_states: dict[str, dict[str, Any]] = {}
    for job in parents:
        status = read_json(_status_path(root, str(job["job_id"])))
        state = _artifact(status, "full_training_state")
        contract = read_json(job["full_state_contract_path"])
        payload = _validated_full_state_payload(job, status)
        lineage = _mapping(contract["output_lineage"], "parent output lineage")
        if payload["lineage"] != lineage:
            raise ValueError(f"Parent state lineage drift: {job['job_id']}")
        rows.append(
            {
                "parent_job_id": job["job_id"],
                "path": state["path"],
                "size_bytes": int(state["size_bytes"]),
                "sha256": state["sha256"],
                "lineage_json": json.dumps(
                    lineage, sort_keys=True, separators=(",", ":")
                ),
            }
        )
        parent_states[job["job_id"]] = {
            "path": state["path"],
            "sha256": state["sha256"],
            "lineage": lineage,
        }
    allowlist_path = write_csv(_parent_state_allowlist_path(root), rows)
    slots = int(registry["formal_workers_per_gpu"])
    specs = assign_waves(experiment_specs(CONTINUATION_STAGE), slots_per_gpu=slots)
    new_jobs: list[dict[str, Any]] = []
    for spec in specs:
        parent = parent_states[_parent_id(spec)]
        job, _ = build_job(
            config,
            root,
            spec,
            slots_per_gpu=slots,
            parent_state=parent,
        )
        new_jobs.append(job)
    if {job["job_id"] for job in registry["jobs"]} & {
        job["job_id"] for job in new_jobs
    }:
        raise ValueError("Continuation materialization would duplicate jobs")
    for job in new_jobs:
        write_json(_status_path(root, job["job_id"]), initial_job_status(job))
    registry["jobs"] = [*registry["jobs"], *new_jobs]
    registry.update(
        status="parents_complete_continuations_prepared",
        parent_states_frozen=True,
        parent_state_allowlist_path=str(allowlist_path.resolve()),
        parent_state_allowlist_sha256=sha256_file(allowlist_path),
    )
    write_registry(root, registry)
    write_experiment_status(
        root, registry["status"], registered_jobs=len(registry["jobs"])
    )
    validate_root(root)
    return allowlist_path


def launch_continuations(output_dir: str | Path, *, resume: bool = False) -> Path:
    root = Path(output_dir).resolve()
    registry = read_registry(root)
    if not registry.get("parent_states_frozen"):
        _freeze_parent_states_and_materialize_continuations(root)
    if registry.get("branch_recipes_frozen"):
        _validate_recipe_manifest(root)
        return root
    _launch_stage(root, CONTINUATION_STAGE, dry_run=False, resume=resume)
    registry = read_registry(root)
    registry["status"] = "continuations_complete"
    write_registry(root, registry)
    write_experiment_status(
        root, "continuations_complete", registered_jobs=len(registry["jobs"])
    )
    return root


def _recipe_manifest_path(root: Path) -> Path:
    return root / "registry/branch_recipe_manifest.csv"


def _validate_recipe_manifest(root: Path) -> dict[str, dict[str, Any]]:
    path = _recipe_manifest_path(root)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != EXPECTED_STAGE_COUNTS[CONTINUATION_STAGE]:
        raise ValueError("Branch recipe manifest must contain exactly 80 rows")
    recipes: dict[str, dict[str, Any]] = {}
    for row in rows:
        path_value = Path(row["path"])
        if (
            row["continuation_job_id"] in recipes
            or not path_value.is_file()
            or path_value.stat().st_size != int(row["size_bytes"])
            or sha256_file(path_value) != row["sha256"]
        ):
            raise ValueError("Branch recipe manifest drift")
        payload = read_json(path_value)
        if (
            payload.get("refit_mode") != "frozen_epoch_lr_replay_v1"
            or int(payload.get("num_epochs", 0)) <= 0
            or len(payload.get("generator_lr_trace", [])) != int(payload["num_epochs"])
            or len(payload.get("discriminator_lr_trace", []))
            != int(payload["num_epochs"])
        ):
            raise ValueError(f"Invalid branch recipe: {path_value}")
        recipes[row["continuation_job_id"]] = {
            "path": str(path_value.resolve()),
            "sha256": row["sha256"],
            "num_epochs": int(payload["num_epochs"]),
        }
    return recipes


def freeze_branch_recipes(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    config = validate_root(root)
    registry = read_registry(root)
    if registry.get("branch_recipes_frozen"):
        _validate_recipe_manifest(root)
        return _recipe_manifest_path(root)
    continuations = _validate_stage_complete(root, CONTINUATION_STAGE)
    parent_states = _validate_parent_state_allowlist(root)
    recipe_rows: list[dict[str, Any]] = []
    recipes: dict[str, dict[str, Any]] = {}
    recipe_dir = root / "registry/branch_recipes"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    for job in continuations:
        status = read_json(_status_path(root, str(job["job_id"])))
        state = _artifact(status, "full_training_state")
        contract = read_json(job["full_state_contract_path"])
        state_payload = _validated_full_state_payload(job, status)
        completed_epoch = int(state_payload["completed_epoch"])
        metrics = pd.read_csv(_artifact(status, "training_metrics_csv")["path"])
        selected = metrics.loc[
            (pd.to_numeric(metrics["epoch"], errors="coerce") >= 1)
            & (pd.to_numeric(metrics["epoch"], errors="coerce") <= completed_epoch)
        ].sort_values("epoch")
        if list(selected["epoch"].astype(int)) != list(range(1, completed_epoch + 1)):
            raise ValueError(f"Continuation LR trace is incomplete: {job['job_id']}")
        recipe_payload = {
            "schema_version": 1,
            "refit_mode": "frozen_epoch_lr_replay_v1",
            "num_epochs": completed_epoch,
            "generator_lr_trace": [
                {"epoch": int(row.epoch), "lr": float(row.g_lr)}
                for row in selected.itertuples(index=False)
            ],
            "discriminator_lr_trace": [
                {"epoch": int(row.epoch), "lr": float(row.d_lr)}
                for row in selected.itertuples(index=False)
            ],
            "parent_job_id": job["parent_job_id"],
            "parent_state_sha256": job["parent_state_sha256"],
            "continuation_job_id": job["job_id"],
            "continuation_state_sha256": state["sha256"],
            "continuation_output_lineage": contract["output_lineage"],
        }
        recipe_path = write_json(recipe_dir / f"{job['job_id']}.json", recipe_payload)
        recipe_rows.append(
            {
                "continuation_job_id": job["job_id"],
                "parent_job_id": job["parent_job_id"],
                "path": str(recipe_path.resolve()),
                "size_bytes": recipe_path.stat().st_size,
                "sha256": sha256_file(recipe_path),
                "num_epochs": completed_epoch,
            }
        )
        recipes[job["job_id"]] = {
            "path": str(recipe_path.resolve()),
            "sha256": sha256_file(recipe_path),
            "num_epochs": completed_epoch,
        }
    recipe_manifest = write_csv(_recipe_manifest_path(root), recipe_rows)
    slots = int(registry["formal_workers_per_gpu"])
    specs = assign_waves(experiment_specs(BRANCH_STAGE), slots_per_gpu=slots)
    branches: list[dict[str, Any]] = []
    for spec in specs:
        parent = parent_states[_parent_id(spec)]
        recipe = recipes[_continuation_id(spec)]
        branch, _ = build_job(
            config,
            root,
            spec,
            slots_per_gpu=slots,
            parent_state=parent,
            recipe=recipe,
        )
        branches.append(branch)
    if {job["job_id"] for job in registry["jobs"]} & {
        job["job_id"] for job in branches
    }:
        raise ValueError("Branch materialization would duplicate jobs")
    for job in branches:
        write_json(_status_path(root, job["job_id"]), initial_job_status(job))
    registry["jobs"] = [*registry["jobs"], *branches]
    if len(registry["jobs"]) != EXPECTED_TRAINING_JOBS:
        raise ValueError("Terminal registry must contain exactly 400 jobs")
    validate_gpu_balance(registry["jobs"])
    registry.update(
        status="branch_recipes_frozen_text_branches_prepared",
        branch_recipes_frozen=True,
        branch_recipe_manifest_path=str(recipe_manifest.resolve()),
        branch_recipe_manifest_sha256=sha256_file(recipe_manifest),
    )
    write_registry(root, registry)
    write_experiment_status(
        root, registry["status"], registered_jobs=len(registry["jobs"])
    )
    validate_root(root)
    return recipe_manifest


def launch_text_branches(output_dir: str | Path, *, resume: bool = False) -> Path:
    root = Path(output_dir).resolve()
    if not read_registry(root).get("branch_recipes_frozen"):
        freeze_branch_recipes(root)
    _launch_stage(root, BRANCH_STAGE, dry_run=False, resume=resume)
    registry = read_registry(root)
    registry["status"] = "all_training_complete"
    write_registry(root, registry)
    write_experiment_status(root, "all_training_complete", completed_jobs=400)
    return root


def _overlay_mode(arm: str) -> str:
    return {
        PARENT_ARM: "current_only",
        CONTINUATION_ARM: "current_only",
        "lp_matched": "lp_mean_l2",
        "lp_shuffle": "lp_shuffle",
        "bow": "bow1024",
        "sentiment": "sentiment_pad1024",
    }[str(arm)]


def _selected_checkpoint_roles(job: Mapping[str, Any]) -> tuple[str, str]:
    if str(job["stage"]) in {PARENT_STAGE, CONTINUATION_STAGE}:
        return "generator_best_learned", "discriminator_best_learned"
    if str(job["stage"]) == BRANCH_STAGE:
        return "generator_final", "discriminator_final"
    raise ValueError(f"No evaluation checkpoint for stage={job['stage']!r}")


def _checkpoint_allowlist_path(root: Path) -> Path:
    return root / "registry/evaluation_checkpoint_allowlist.csv"


def _checkpoint_manifest_path(root: Path) -> Path:
    return root / "analysis/checkpoint_manifest.csv"


def _inference_determinism_contract_path(root: Path) -> Path:
    return root / "evaluation/inference_determinism_contract.json"


def _inference_determinism_contract() -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": INFERENCE_DETERMINISM_KIND,
        "seed_scope": "checkpoint_job_seed_before_each_cell_v1",
        "python_numpy_torch_seeded": True,
        "torch_cuda_manual_seed_all": True,
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": ":4096:8",
        "noise_method": "stable_noise_for_keys_v1",
        "prediction_mc_samples": 64,
        "noise_dim": 32,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def _configure_prediction_determinism(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from wgan_option.utils.reproducibility import seed_everything

    seed_everything(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _validate_inference_determinism_contract(root: Path) -> Path:
    path = _inference_determinism_contract_path(root)
    if read_json(path) != _inference_determinism_contract():
        raise ValueError("Inference determinism contract drift")
    return path


def _validate_checkpoint_allowlist(root: Path) -> dict[str, dict[str, str]]:
    registry = read_registry(root)
    allowlist_path = Path(str(registry.get("checkpoint_allowlist_path", "")))
    checkpoint_manifest = Path(str(registry.get("checkpoint_manifest_path", "")))
    if (
        not allowlist_path.is_file()
        or sha256_file(allowlist_path) != registry.get("checkpoint_allowlist_sha256")
        or not checkpoint_manifest.is_file()
        or sha256_file(checkpoint_manifest)
        != registry.get("checkpoint_manifest_sha256")
    ):
        raise ValueError("Frozen evaluation checkpoint lineage drift")
    determinism = _validate_inference_determinism_contract(root)
    if registry.get("inference_determinism_contract_path") != str(
        determinism.resolve()
    ) or registry.get("inference_determinism_contract_sha256") != sha256_file(
        determinism
    ):
        raise ValueError("Frozen inference determinism lineage drift")
    allowlist = pd.read_csv(allowlist_path, dtype=str, keep_default_na=False)
    expected_columns = {
        "job_id",
        "arm",
        "fold",
        "seed",
        "tolerance_minutes",
        "checkpoint_role",
        "checkpoint_path",
        "size_bytes",
        "checkpoint_sha256",
    }
    if set(allowlist.columns) != expected_columns or len(allowlist) != 800:
        raise ValueError("Evaluation allowlist must contain exactly 800 G/D rows")
    if allowlist.duplicated(["job_id", "checkpoint_role"]).any():
        raise ValueError("Evaluation allowlist contains duplicate job/role rows")
    expected_jobs = {str(job["job_id"]) for job in registry.get("jobs", [])}
    if set(allowlist["job_id"]) != expected_jobs:
        raise ValueError("Evaluation allowlist job universe drift")
    for row in allowlist.itertuples(index=False):
        path = Path(row.checkpoint_path)
        if (
            not path.is_file()
            or path.stat().st_size != int(row.size_bytes)
            or sha256_file(path) != row.checkpoint_sha256
        ):
            raise ValueError(f"Evaluation checkpoint drift: {path}")
    manifest = pd.read_csv(checkpoint_manifest, dtype=str, keep_default_na=False)
    if (
        len(manifest) != EXPECTED_PREDICTION_CELLS
        or set(manifest["job_id"]) != expected_jobs
    ):
        raise ValueError("Generator checkpoint manifest must contain exactly 400 jobs")
    generators = allowlist.loc[allowlist["checkpoint_role"].str.startswith("generator")]
    expected = generators.set_index("job_id")["checkpoint_sha256"].sort_index()
    actual = manifest.set_index("job_id")["checkpoint_sha256"].sort_index()
    if not expected.equals(actual):
        raise ValueError("Generator checkpoint manifest differs from allowlist")
    return {
        str(row.job_id): {
            "path": str(row.checkpoint_path),
            "sha256": str(row.checkpoint_sha256),
            "role": str(row.checkpoint_role),
        }
        for row in generators.itertuples(index=False)
    }


def _freeze_checkpoint_allowlist(root: Path) -> Path:
    registry = read_registry(root)
    if bool(registry.get("evaluation_frozen")):
        _validate_checkpoint_allowlist(root)
        return _checkpoint_allowlist_path(root)
    jobs = [dict(job) for job in registry.get("jobs", [])]
    if len(jobs) != EXPECTED_TRAINING_JOBS:
        raise ValueError("Evaluation freeze requires exactly 400 registered jobs")
    for stage in STAGES:
        _validate_stage_complete(root, stage)
    rows: list[dict[str, Any]] = []
    generator_rows: list[dict[str, Any]] = []
    for job in jobs:
        status = read_json(_status_path(root, str(job["job_id"])))
        generator_role, critic_role = _selected_checkpoint_roles(job)
        for role in (generator_role, critic_role):
            artifact = _artifact(status, role)
            row = {
                "job_id": str(job["job_id"]),
                "arm": str(job["arm"]),
                "fold": str(job["fold"]),
                "seed": int(job["seed"]),
                "tolerance_minutes": int(job["tolerance_minutes"]),
                "checkpoint_role": role,
                "checkpoint_path": str(Path(artifact["path"]).resolve()),
                "size_bytes": int(artifact["size_bytes"]),
                "checkpoint_sha256": str(artifact["sha256"]),
            }
            rows.append(row)
            if role == generator_role:
                generator_rows.append(
                    {
                        "job_id": row["job_id"],
                        "arm": row["arm"],
                        "fold": row["fold"],
                        "seed": row["seed"],
                        "tolerance_minutes": row["tolerance_minutes"],
                        "checkpoint_path": row["checkpoint_path"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "size_bytes": row["size_bytes"],
                    }
                )
    if (
        len(rows) != 800
        or len(generator_rows) != EXPECTED_PREDICTION_CELLS
        or len({row["checkpoint_path"] for row in rows}) != 800
    ):
        raise ValueError("Evaluation checkpoint universe is not exactly 400 G/D pairs")
    allowlist_path = write_csv(_checkpoint_allowlist_path(root), rows, tuple(rows[0]))
    checkpoint_manifest = write_csv(
        _checkpoint_manifest_path(root), generator_rows, tuple(generator_rows[0])
    )
    determinism_contract = write_json(
        _inference_determinism_contract_path(root),
        _inference_determinism_contract(),
    )
    # This is the irreversible data-access gate: the checkpoint universe is
    # immutable before any fold-test feature, surface, event, or error is read.
    registry.update(
        status="evaluation_frozen",
        evaluation_frozen=True,
        evaluation_frozen_at_utc=utc_now(),
        checkpoint_allowlist_path=str(allowlist_path.resolve()),
        checkpoint_allowlist_sha256=sha256_file(allowlist_path),
        checkpoint_manifest_path=str(checkpoint_manifest.resolve()),
        checkpoint_manifest_sha256=sha256_file(checkpoint_manifest),
        inference_determinism_contract_path=str(determinism_contract.resolve()),
        inference_determinism_contract_sha256=sha256_file(determinism_contract),
        test_data_opened=False,
    )
    write_registry(root, registry)
    write_experiment_status(root, "evaluation_frozen", completed_jobs=400)
    _validate_checkpoint_allowlist(root)
    return allowlist_path


def _write_dataframe_csv(
    path: Path, frame: pd.DataFrame, *, gzip: bool = False
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if gzip:
        with temporary.open("wb") as raw:
            with gzip_module.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    frame.to_csv(text, index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return path


def _materialize_frozen_event_sources(config: Mapping[str, Any], root: Path) -> Path:
    destination = root / "evaluation/frozen_scheduled_events.csv"
    manifest_path = root / "evaluation/frozen_event_sources.json"
    if destination.is_file() and manifest_path.is_file():
        manifest = read_json(manifest_path)
        unsigned = {
            key: value for key, value in manifest.items() if key != "payload_sha256"
        }
        if payload_sha256(unsigned) != manifest.get("payload_sha256") or manifest.get(
            "scheduled_events_sha256"
        ) != sha256_file(destination):
            raise ValueError("Frozen event-source manifest drift")
        for source in manifest.get("sources", []):
            verify_manifest_rows([source])
        return manifest_path
    data = _mapping(config["data"], "data")
    source_paths = [resolve_path(value) for value in data["scheduled_event_paths"]]
    frames = [pd.read_csv(path) for path in source_paths]
    scheduled = pd.concat(frames, ignore_index=True)
    if scheduled["event_id"].astype(str).duplicated().any():
        raise ValueError("Scheduled event_id values must be unique")
    times = pd.to_datetime(scheduled["release_time_utc"], errors="coerce", utc=True)
    if times.isna().any() or len(scheduled) != 168:
        raise ValueError("Scheduled event calendar must contain 168 valid UTC events")
    scheduled["release_time_utc"] = times.map(
        lambda value: value.isoformat().replace("+00:00", "Z")
    )
    scheduled = scheduled.sort_values(["release_time_utc", "event_id"], kind="stable")
    recovery = destination.with_name(f".{destination.name}.recovery.{os.getpid()}")
    _write_dataframe_csv(recovery, scheduled)
    expected_scheduled_sha = sha256_file(recovery)
    if destination.is_file():
        if sha256_file(destination) != expected_scheduled_sha:
            recovery.unlink(missing_ok=True)
            raise ValueError("Partial frozen scheduled-event file is not reproducible")
        recovery.unlink()
    else:
        os.replace(recovery, destination)
    market_paths = [
        resolve_path(data["market_jump_candidate_pairs"]),
        resolve_path(data["market_jump_candidate_episodes"]),
        resolve_path(data["market_jump_validation_summary"]),
    ]
    source_rows = [
        manifest_row(f"scheduled_source_{index + 1}", path)
        for index, path in enumerate(source_paths)
    ] + [
        manifest_row(f"market_jump_source_{index + 1}", path)
        for index, path in enumerate(market_paths)
    ]
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        unsigned = {
            key: value for key, value in manifest.items() if key != "payload_sha256"
        }
        if (
            payload_sha256(unsigned) != manifest.get("payload_sha256")
            or manifest.get("scheduled_events_sha256") != expected_scheduled_sha
            or manifest.get("sources") != source_rows
        ):
            raise ValueError("Partial frozen event manifest is not reproducible")
        return manifest_path
    manifest = {
        "schema_version": 1,
        "kind": "rq123_frozen_event_sources_v1",
        "interpretation": INTERPRETATION,
        "sources": source_rows,
        "scheduled_events_path": str(destination.resolve()),
        "scheduled_events_sha256": sha256_file(destination),
        "scheduled_event_count": int(len(scheduled)),
        "market_jump_pairs_path": str(market_paths[0]),
        "market_jump_pairs_sha256": sha256_file(market_paths[0]),
    }
    manifest["payload_sha256"] = payload_sha256(manifest)
    return write_json(manifest_path, manifest)


def _test_input_manifest_path(root: Path) -> Path:
    return root / "evaluation/test_input_hashes.csv"


def _test_overlay_path(root: Path, tolerance: int, fold: str, arm: str) -> Path:
    return (
        root
        / "evaluation/test_pair_text_overlays"
        / f"tolerance_{int(tolerance):02d}m"
        / str(fold)
        / f"{arm}.json"
    )


def _test_panel_path(root: Path, tolerance: int, fold: str) -> Path:
    return (
        root
        / "evaluation/test_panels"
        / f"tolerance_{int(tolerance):02d}m"
        / f"{fold}.csv.gz"
    )


def _validate_test_input_manifest_file(root: Path, path: Path) -> None:
    rows = read_manifest(path)
    expected_roles = {
        *(
            f"test_panel:{tolerance:02d}m:{fold}"
            for tolerance in TOLERANCES
            for fold in FOLDS
        ),
        *(
            f"test_overlay:{tolerance:02d}m:{fold}:{arm}"
            for tolerance in TOLERANCES
            for fold in FOLDS
            for arm in (PARENT_ARM, CONTINUATION_ARM, *_arms_for_tolerance(tolerance))
        ),
        "frozen_event_manifest",
        "frozen_scheduled_events",
    }
    if len(rows) != 50 or {str(row["artifact_role"]) for row in rows} != expected_roles:
        raise ValueError("Frozen test-input artifact universe drift")


def _canonical_article_text(value: object) -> str:
    """Share training's missing-text semantics for frozen test transforms."""

    from wgan_option.utils.news_first_experiment_core import _normalized_article_text

    return _normalized_article_text(value)


def _bow_vector_from_article_texts(
    texts: Sequence[object], vocabulary: Sequence[str]
) -> Any:
    import numpy as np
    from wgan_option.utils.news_first_experiment_core import _l2, _ngram_counts

    vocabulary_index = {term: index for index, term in enumerate(vocabulary)}
    counts: dict[str, int] = {}
    for text in texts:
        for term, count in _ngram_counts(text).items():
            counts[term] = counts.get(term, 0) + int(count)
    vector = np.zeros(1024, dtype=np.float32)
    for term, count in counts.items():
        index = vocabulary_index.get(term)
        if index is not None:
            vector[index] = np.log1p(float(count))
    return _l2(vector)


def _validate_test_inputs(root: Path) -> Path:
    registry = read_registry(root)
    path = Path(str(registry.get("test_input_manifest_path", "")))
    if (
        not bool(registry.get("evaluation_frozen"))
        or not bool(registry.get("test_data_opened"))
        or not path.is_file()
        or sha256_file(path) != registry.get("test_input_manifest_sha256")
    ):
        raise ValueError("Frozen test-input lineage drift")
    _validate_test_input_manifest_file(root, path)
    return path


def _materialize_test_inputs(config: Mapping[str, Any], root: Path) -> Path:
    """Open fold-test rows only after the 400-checkpoint freeze.

    BoW and sentiment reuse the exact train-only transform stored in the
    training overlay.  LP shuffle creates one deterministic derangement within
    each fold's test partition; it never reuses a validation donor.
    """

    registry = read_registry(root)
    if not bool(registry.get("evaluation_frozen")):
        raise RuntimeError("Test inputs require a frozen checkpoint allowlist")
    existing = _test_input_manifest_path(root)
    if existing.is_file():
        _validate_test_input_manifest_file(root, existing)
        if bool(registry.get("test_data_opened")):
            return _validate_test_inputs(root)
        if registry.get("test_input_manifest_path") or registry.get(
            "test_input_manifest_sha256"
        ):
            raise ValueError("Partial test-input registry anchor is inconsistent")
        registry.update(
            status="evaluation_inputs_frozen",
            test_data_opened=True,
            test_data_opened_at_utc=utc_now(),
            test_input_manifest_path=str(existing.resolve()),
            test_input_manifest_sha256=sha256_file(existing),
        )
        write_registry(root, registry)
        write_experiment_status(root, "evaluation_inputs_frozen", test_panels=8)
        return _validate_test_inputs(root)

    import numpy as np
    from wgan_option.utils.news_first_experiment_core import (
        _fixed_partition_derangement,
        _l2,
        _normalized_article_key,
        _parsed_vector,
        write_pair_text_overlay_manifest,
    )

    data = _mapping(config["data"], "data")
    matrix = _mapping(config["matrix"], "matrix")
    universe = pd.read_csv(root / "inputs/pair_universes.csv", dtype=str)
    sentiment_source = resolve_path(data["sentiment_workbook_path"])
    sentiment_frame = pd.read_excel(
        sentiment_source,
        sheet_name="features",
        usecols=["news_row_id", "sentiment_embedding"],
    )
    if sentiment_frame["news_row_id"].astype(str).duplicated().any():
        raise ValueError("Sentiment source contains duplicate news_row_id values")
    raw_sentiment = dict(
        zip(
            sentiment_frame["news_row_id"].astype(str),
            sentiment_frame["sentiment_embedding"],
        )
    )
    artifact_rows: list[dict[str, Any]] = []
    for tolerance in TOLERANCES:
        test_universe = universe.loc[
            (universe["tolerance_minutes"].astype(int) == tolerance)
            & universe["partition"].eq("test")
        ].copy()
        required_pairs = set(test_universe["pair_id"].astype(str))
        workbook = resolve_path(data["root"]) / str(data["workbook_template"]).format(
            tolerance02=f"{tolerance:02d}"
        )
        frame = pd.read_excel(workbook, sheet_name=str(data["sheet_name"]))
        frame["pair_id"] = frame["pair_id"].astype(str)
        frame = frame.loc[frame["pair_id"].isin(required_pairs)].copy()
        if set(frame["pair_id"]) != required_pairs:
            raise ValueError(f"Test workbook coverage drift for {tolerance}m")
        rows_by_pair: dict[str, list[dict[str, Any]]] = {}
        canonical_rows: dict[str, dict[str, Any]] = {}
        for pair_id, pair_rows in frame.groupby("pair_id", sort=False):
            ordered = pair_rows.assign(
                _news_row_sort=pd.to_numeric(pair_rows["news_row_id"], errors="raise")
            ).sort_values(["_news_row_sort", "sample_id"], kind="stable")
            canonical_rows[str(pair_id)] = (
                ordered.iloc[0].drop(labels="_news_row_sort").to_dict()
            )
            articles: dict[str, dict[str, Any]] = {}
            for row in ordered.itertuples(index=False):
                article_key = _normalized_article_key(row)
                news_key = str(row.news_row_id)
                if news_key not in raw_sentiment:
                    raise ValueError(
                        f"Test sentiment coverage missing news_row_id={news_key}"
                    )
                article = {
                    "lp_text": _canonical_article_text(row.lp_text),
                    "lp_embedding": _parsed_vector(
                        row.lp_embedding,
                        dimension=1024,
                        label=f"test LP pair={pair_id}/{article_key}",
                    ),
                    "sentiment": _parsed_vector(
                        raw_sentiment[news_key],
                        dimension=1024,
                        label=f"test sentiment news={news_key}",
                    )[:3],
                }
                previous = articles.get(article_key)
                if previous is not None:
                    if previous["lp_text"] != article["lp_text"]:
                        raise ValueError(
                            f"Test article text drift: {pair_id}/{article_key}"
                        )
                    continue
                articles[article_key] = article
            rows_by_pair[str(pair_id)] = [articles[key] for key in sorted(articles)]

        lp_by_pair = {
            pair_id: _l2(
                np.mean(
                    np.stack([article["lp_embedding"] for article in articles], axis=0),
                    axis=0,
                )
            )
            for pair_id, articles in rows_by_pair.items()
        }
        sentiment_by_pair = {
            pair_id: np.mean(
                np.stack([article["sentiment"] for article in articles], axis=0),
                axis=0,
            ).astype(np.float32)
            for pair_id, articles in rows_by_pair.items()
        }
        for fold_id in FOLDS:
            selected_universe = test_universe.loc[test_universe["fold"].eq(fold_id)]
            pair_ids = sorted(selected_universe["pair_id"].astype(str))
            sessions = dict(
                zip(
                    selected_universe["pair_id"].astype(str),
                    selected_universe["session_id"].astype(str),
                )
            )
            expected = _expected_counts(config, tolerance, fold_id)
            if (
                len(pair_ids) != expected["test_pairs"]
                or len(set(sessions.values())) != expected["test_sessions"]
            ):
                raise ValueError(f"Frozen test counts drift: {tolerance}m {fold_id}")
            panel = pd.DataFrame([canonical_rows[pair_id] for pair_id in pair_ids])
            panel["sample_id"] = panel["pair_id"].map(lambda value: f"pair::{value}")
            panel["sample_weight"] = 1.0
            panel_path = _write_dataframe_csv(
                _test_panel_path(root, tolerance, fold_id), panel, gzip=True
            )
            artifact_rows.append(
                manifest_row(f"test_panel:{tolerance:02d}m:{fold_id}", panel_path)
            )

            donor_by_pair: dict[str, str] = {}
            for arm in (PARENT_ARM, CONTINUATION_ARM, *_arms_for_tolerance(tolerance)):
                training_path = _overlay_manifest_path(
                    root,
                    {
                        "tolerance_minutes": tolerance,
                        "fold": fold_id,
                        "arm": arm,
                    },
                )
                training_payload = read_json(training_path)
                transform = dict(training_payload.get("transform") or {})
                transform.update(
                    {
                        "evaluation_partition": "test",
                        "training_overlay_path": str(training_path),
                        "training_overlay_sha256": sha256_file(training_path),
                        "missing_text_article_count_test": sum(
                            not bool(article["lp_text"])
                            for pair_id in pair_ids
                            for article in rows_by_pair[pair_id]
                        ),
                        "missing_text_pair_count_test": sum(
                            all(
                                not bool(article["lp_text"])
                                for article in rows_by_pair[pair_id]
                            )
                            for pair_id in pair_ids
                        ),
                    }
                )
                if arm in {PARENT_ARM, CONTINUATION_ARM}:
                    vectors = {
                        pair_id: np.zeros(1024, dtype=np.float32)
                        for pair_id in pair_ids
                    }
                    transform["method"] = "zero_vector_v1"
                elif arm == "lp_matched":
                    vectors = {pair_id: lp_by_pair[pair_id] for pair_id in pair_ids}
                    transform["method"] = "unique_article_lp_mean_l2_v1"
                elif arm == "lp_shuffle":
                    donor_by_pair = _fixed_partition_derangement(
                        pair_ids,
                        master_seed=int(matrix["shuffle_seed"]),
                        namespace=f"evaluation/test/{tolerance:02d}m/{fold_id}",
                    )
                    vectors = {
                        pair_id: lp_by_pair[donor_by_pair[pair_id]]
                        for pair_id in pair_ids
                    }
                    transform.update(
                        method="test_partition_pair_derangement_v1",
                        master_seed=int(matrix["shuffle_seed"]),
                        mapping_sha256=payload_sha256(sorted(donor_by_pair.items())),
                    )
                elif arm == "bow":
                    vocabulary = list(transform.get("vocabulary") or [])
                    if not vocabulary or len(vocabulary) > 1024:
                        raise ValueError("Frozen train-only BoW vocabulary is invalid")
                    vectors = {
                        pair_id: _bow_vector_from_article_texts(
                            [article["lp_text"] for article in rows_by_pair[pair_id]],
                            vocabulary,
                        )
                        for pair_id in pair_ids
                    }
                elif arm == "sentiment":
                    mean = np.asarray(transform.get("train_mean"), dtype=np.float32)
                    std = np.asarray(transform.get("train_std"), dtype=np.float32)
                    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
                        raise ValueError(
                            "Frozen train-only sentiment transform is invalid"
                        )
                    vectors = {}
                    for pair_id in pair_ids:
                        vector = np.zeros(1024, dtype=np.float32)
                        vector[:3] = (sentiment_by_pair[pair_id] - mean) / std
                        vectors[pair_id] = vector
                else:  # pragma: no cover - arm universe is frozen above
                    raise AssertionError(arm)
                records = [
                    {
                        "pair_id": pair_id,
                        "session_id": sessions[pair_id],
                        "embedding": vectors[pair_id],
                        **(
                            {"donor_pair_id": donor_by_pair[pair_id]}
                            if arm == "lp_shuffle"
                            else {}
                        ),
                    }
                    for pair_id in pair_ids
                ]
                overlay_path = _test_overlay_path(root, tolerance, fold_id, arm)
                write_pair_text_overlay_manifest(
                    overlay_path,
                    mode=_overlay_mode(arm),
                    namespace=f"evaluation/test/{tolerance:02d}m/{fold_id}/{arm}",
                    records=records,
                    transform=transform,
                )
                artifact_rows.append(
                    manifest_row(
                        f"test_overlay:{tolerance:02d}m:{fold_id}:{arm}",
                        overlay_path,
                    )
                )
    event_manifest = _materialize_frozen_event_sources(config, root)
    artifact_rows.extend(
        [
            manifest_row("frozen_event_manifest", event_manifest),
            manifest_row(
                "frozen_scheduled_events",
                root / "evaluation/frozen_scheduled_events.csv",
            ),
        ]
    )
    if len(artifact_rows) != 50:
        raise ValueError(f"Expected 50 test-input artifacts, got {len(artifact_rows)}")
    result = _manifest_csv(_test_input_manifest_path(root), artifact_rows)
    registry = read_registry(root)
    registry.update(
        status="evaluation_inputs_frozen",
        test_data_opened=True,
        test_data_opened_at_utc=utc_now(),
        test_input_manifest_path=str(result.resolve()),
        test_input_manifest_sha256=sha256_file(result),
    )
    write_registry(root, registry)
    write_experiment_status(root, "evaluation_inputs_frozen", test_panels=8)
    _validate_test_inputs(root)
    return result


def freeze_evaluation(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    config = validate_root(root)
    _freeze_checkpoint_allowlist(root)
    _materialize_test_inputs(config, root)
    return root


def _prediction_path(root: Path, job: Mapping[str, Any]) -> Path:
    return (
        root
        / "evaluation/predictions"
        / f"tolerance_{int(job['tolerance_minutes']):02d}m"
        / str(job["fold"])
        / f"seed_{int(job['seed'])}"
        / f"{job['arm']}.csv.gz"
    )


def _prediction_job_manifest_path(root: Path, job: Mapping[str, Any]) -> Path:
    prediction = _prediction_path(root, job)
    return prediction.with_suffix(".manifest.json")


def _noise_bank_profile_sha256(
    job: Mapping[str, Any], sample_ids: Sequence[str]
) -> str:
    return payload_sha256(
        {
            "schema_version": 1,
            "method": "stable_noise_for_keys_v1",
            "tolerance_minutes": int(job["tolerance_minutes"]),
            "fold": str(job["fold"]),
            "seed": int(job["seed"]),
            "sample_ids": sorted(str(value) for value in sample_ids),
            "draws": 64,
            "noise_dim": 32,
        }
    )


def _expected_noise_bank_profile(root: Path, job: Mapping[str, Any]) -> str:
    panel = pd.read_csv(
        _test_panel_path(root, int(job["tolerance_minutes"]), str(job["fold"])),
        usecols=["sample_id"],
        dtype=str,
    )
    return _noise_bank_profile_sha256(job, panel["sample_id"].astype(str).tolist())


def _validate_prediction_job(
    root: Path,
    job: Mapping[str, Any],
    checkpoint: Mapping[str, str],
    *,
    expected_pairs: int,
) -> dict[str, Any]:
    path = _prediction_path(root, job)
    manifest_path = _prediction_job_manifest_path(root, job)
    evidence_path = manifest_path.with_suffix(".pair_metrics.csv")
    if not path.is_file() or not manifest_path.is_file() or not evidence_path.is_file():
        raise ValueError(f"Partial prediction artifact: {job['job_id']}")
    payload = read_json(manifest_path)
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    panel_path = _test_panel_path(root, int(job["tolerance_minutes"]), str(job["fold"]))
    overlay_path = _test_overlay_path(
        root,
        int(job["tolerance_minutes"]),
        str(job["fold"]),
        str(job["arm"]),
    )
    expected_noise_profile = _expected_noise_bank_profile(root, job)
    determinism_contract = _validate_inference_determinism_contract(root)
    training_config = yaml.safe_load(
        Path(str(job["training_config_path"])).read_text(encoding="utf-8")
    )
    if (
        not isinstance(training_config, Mapping)
        or training_config.get("generator_noise_mode") != "gaussian"
        or int(training_config.get("noise_dim", -1)) != 32
    ):
        raise ValueError(f"Prediction checkpoint noise contract drift: {job['job_id']}")
    if (
        payload_sha256(unsigned) != payload.get("payload_sha256")
        or payload.get("job_id") != job["job_id"]
        or payload.get("job_spec_sha256") != job["job_spec_sha256"]
        or payload.get("checkpoint_path") != str(Path(checkpoint["path"]).resolve())
        or payload.get("checkpoint_sha256") != checkpoint["sha256"]
        or payload.get("panel_path") != str(panel_path.resolve())
        or payload.get("panel_sha256") != sha256_file(panel_path)
        or payload.get("test_overlay_path") != str(overlay_path.resolve())
        or payload.get("test_overlay_sha256") != sha256_file(overlay_path)
        or payload.get("prediction_path") != str(path.resolve())
        or payload.get("prediction_sha256") != sha256_file(path)
        or payload.get("pair_metrics_path") != str(evidence_path.resolve())
        or payload.get("pair_metrics_sha256") != sha256_file(evidence_path)
        or int(payload.get("pair_metrics_row_count", -1)) != int(expected_pairs)
        or int(payload.get("prediction_mc_samples", -1)) != 64
        or payload.get("noise_bank_profile_sha256") != expected_noise_profile
        or payload.get("inference_determinism_contract_sha256")
        != sha256_file(determinism_contract)
        or int(payload.get("row_count", -1)) != int(expected_pairs)
    ):
        raise ValueError(f"Prediction manifest drift: {job['job_id']}")
    evidence = pd.read_csv(evidence_path)
    required = {
        "job_id",
        "pair_id",
        "target_mae",
        "persistence_mae",
        "checkpoint_sha256",
        "prediction_sha256",
        "noise_bank_profile_sha256",
        "inference_determinism_contract_sha256",
    }
    if (
        len(evidence) != int(expected_pairs)
        or not required.issubset(evidence.columns)
        or evidence["pair_id"].astype(str).duplicated().any()
        or set(evidence["job_id"].astype(str)) != {str(job["job_id"])}
        or set(evidence["checkpoint_sha256"].astype(str)) != {str(checkpoint["sha256"])}
        or set(evidence["prediction_sha256"].astype(str))
        != {str(payload["prediction_sha256"])}
        or set(evidence["noise_bank_profile_sha256"].astype(str))
        != {expected_noise_profile}
        or set(evidence["inference_determinism_contract_sha256"].astype(str))
        != {sha256_file(determinism_contract)}
        or not all(
            math.isfinite(float(value))
            for column in ("target_mae", "persistence_mae")
            for value in pd.to_numeric(evidence[column], errors="coerce")
        )
    ):
        raise ValueError(f"Prediction pair evidence drift: {job['job_id']}")
    return payload


def _panel_with_overlay(root: Path, job: Mapping[str, Any]) -> pd.DataFrame:
    from wgan_option.utils.news_first_experiment_core import (
        load_pair_text_overlay_manifest,
    )

    tolerance = int(job["tolerance_minutes"])
    fold = str(job["fold"])
    arm = str(job["arm"])
    panel_path = _test_panel_path(root, tolerance, fold)
    panel = pd.read_csv(panel_path, low_memory=False)
    overlay_path = _test_overlay_path(root, tolerance, fold, arm)
    overlay_payload = read_json(overlay_path)
    manifest = load_pair_text_overlay_manifest(
        overlay_path,
        sha256_file(overlay_path),
        str(overlay_payload["profile_sha256"]),
        expected_mode=_overlay_mode(arm),
    )
    pair_ids = panel["pair_id"].astype(str)
    if set(pair_ids) != set(manifest.embeddings) or pair_ids.duplicated().any():
        raise ValueError(f"Test overlay/panel universe drift: {job['job_id']}")
    if any(
        str(session) != manifest.sessions[str(pair_id)]
        for pair_id, session in zip(pair_ids, panel["session_id"], strict=True)
    ):
        raise ValueError(f"Test overlay/panel session drift: {job['job_id']}")
    panel["lp_embedding"] = [
        json.dumps(
            manifest.embeddings[pair_id].astype(float).tolist(),
            separators=(",", ":"),
        )
        for pair_id in pair_ids
    ]
    panel["hd_embedding"] = ""
    panel["sample_id"] = pair_ids.map(lambda value: f"pair::{value}")
    panel["sample_weight"] = 1.0
    return panel


def _current_state_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    from scripts.rq3.news_first_vol_comparison_analysis import (
        _current_support_mask,
        _parse_vector,
        _surface_axes,
    )

    rows: list[dict[str, Any]] = []
    for raw in panel.to_dict(orient="records"):
        current = _parse_vector(
            raw["current_surface_flat"], label="current_surface_flat"
        )
        strike, maturity, shape = _surface_axes(raw, cell_count=len(current))
        surface = current.reshape(shape)
        mask = _current_support_mask(raw, strike_grid=strike, maturity_grid=maturity)
        supported = surface[mask]
        if supported.size == 0 or not np.isfinite(supported).all():
            raise ValueError(f"Current-state support is empty: {raw['pair_id']}")
        atm_index = int(np.argmin(np.abs(strike - 1.0)))
        short_indices = np.flatnonzero(maturity <= 10.0)
        if short_indices.size == 0:
            short_indices = np.asarray([0])
        strike_span = max(float(strike[-1] - strike[0]), 1e-12)
        maturity_span = max(float(maturity[-1] - maturity[0]), 1e-12)
        rows.append(
            {
                "pair_id": str(raw["pair_id"]),
                "current_surface_mean": float(np.mean(supported)),
                "current_surface_std": float(np.std(supported)),
                "current_short_atm_mean": float(
                    np.mean(surface[short_indices, atm_index])
                ),
                "current_strike_slope": float(
                    np.mean((surface[:, -1] - surface[:, 0]) / strike_span)
                ),
                "current_term_slope": float(
                    (surface[-1, atm_index] - surface[0, atm_index]) / maturity_span
                ),
                "current_curvature": float(
                    np.mean(np.abs(np.diff(surface, n=2, axis=1)))
                ),
                "current_supported_cell_fraction": float(mask.mean()),
            }
        )
    result = pd.DataFrame(rows)
    if (
        result["pair_id"].duplicated().any()
        or not np.isfinite(result.drop(columns="pair_id").to_numpy(float)).all()
    ):
        raise ValueError("Current-state matching covariates are invalid")
    return result


def _evaluate_prediction_job(
    root: Path,
    job: Mapping[str, Any],
    checkpoint: Mapping[str, str],
    *,
    expected_pairs: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    from scripts.rq3.news_first_vol_comparison_analysis import (
        RunSpec,
        TrainedRunEvaluator,
        _enforce_formal_run_coverage,
        _prediction_export_frame,
        aggregate_pair_metrics,
        compute_sample_metrics,
    )

    _configure_prediction_determinism(int(job["seed"]))
    determinism_contract = _validate_inference_determinism_contract(root)
    panel = _panel_with_overlay(root, job)
    noise_bank_profile_sha = _noise_bank_profile_sha256(
        job, panel["sample_id"].astype(str).tolist()
    )
    run = RunSpec(
        run_id=str(job["job_id"]),
        run_dir=Path(str(read_json(_status_path(root, str(job["job_id"])))["run_dir"])),
        model="wgan",
        tolerance_minutes=int(job["tolerance_minutes"]),
        seed=int(job["seed"]),
        checkpoint_path=Path(checkpoint["path"]),
        text_ablation_mode="real_text",
        support_mask_mode="raw_joint",
        generator_current_input_mode="current_support_masked",
        metadata={"fold": job["fold"], "arm": job["arm"]},
    )
    evaluator = TrainedRunEvaluator(
        mc_samples=64,
        sample_batch_size=32,
        draw_batch_size=64,
        device=f"cuda:{int(job['gpu_id'])}",
    )
    panel_name = f"{job['fold']}_test_{int(job['tolerance_minutes']):02d}m"
    predictions = evaluator(run, panel_name, panel)
    samples, general_exclusions, metric_exclusions = compute_sample_metrics(
        run,
        panel_name,
        panel,
        predictions,
        evaluate_embedded_atm_skew=False,
    )
    export = _prediction_export_frame(run, panel_name, panel, predictions, samples)
    _enforce_formal_run_coverage(
        run, panel_name, panel, samples, general_exclusions, export
    )
    export["job_id"] = str(job["job_id"])
    export["fold"] = str(job["fold"])
    export["arm"] = str(job["arm"])
    export["checkpoint_sha256"] = checkpoint["sha256"]
    overlay_path = _test_overlay_path(
        root,
        int(job["tolerance_minutes"]),
        str(job["fold"]),
        str(job["arm"]),
    )
    export["test_overlay_sha256"] = sha256_file(overlay_path)
    export["noise_bank_profile_sha256"] = noise_bank_profile_sha
    export["inference_determinism_contract_sha256"] = sha256_file(determinism_contract)
    prediction_path = _write_dataframe_csv(
        _prediction_path(root, job), export, gzip=True
    )
    prediction_sha = sha256_file(prediction_path)

    standard = aggregate_pair_metrics(samples)
    standard = standard.loc[
        standard["stratum_type"].eq("overall") & standard["stratum_value"].eq("all")
    ].copy()
    origin = panel[["pair_id", "effective_origin_utc"]].copy()
    origin["pair_id"] = origin["pair_id"].astype(str)
    standard = standard.merge(origin, on="pair_id", how="left", validate="one_to_one")
    state = _current_state_metrics(panel)
    standard = standard.merge(state, on="pair_id", how="left", validate="one_to_one")
    evidence = pd.DataFrame(
        {
            "job_id": str(job["job_id"]),
            "tolerance_minutes": int(job["tolerance_minutes"]),
            "fold": str(job["fold"]),
            "seed": int(job["seed"]),
            "arm": str(job["arm"]),
            "pair_id": standard["pair_id"].astype(str),
            "session_id": standard["session_id"].astype(str),
            "effective_origin_utc": standard["effective_origin_utc"].astype(str),
            "target_mae": standard["model_mae"].astype(float),
            "persistence_mae": standard["persistence_mae"].astype(float),
            "checkpoint_sha256": checkpoint["sha256"],
            "prediction_sha256": prediction_sha,
            "noise_bank_profile_sha256": noise_bank_profile_sha,
            "inference_determinism_contract_sha256": sha256_file(determinism_contract),
        }
    )
    for column in (
        "current_surface_mean",
        "current_surface_std",
        "current_short_atm_mean",
        "current_strike_slope",
        "current_term_slope",
        "current_curvature",
        "current_supported_cell_fraction",
    ):
        evidence[column] = standard[column].astype(float)
    if len(export) != expected_pairs or len(evidence) != expected_pairs:
        raise ValueError(f"Prediction pair count drift: {job['job_id']}")
    manifest = {
        "schema_version": 1,
        "kind": "rq123_prediction_manifest_v1",
        "interpretation": INTERPRETATION,
        "job_id": str(job["job_id"]),
        "job_spec_sha256": str(job["job_spec_sha256"]),
        "arm": str(job["arm"]),
        "fold": str(job["fold"]),
        "seed": int(job["seed"]),
        "tolerance_minutes": int(job["tolerance_minutes"]),
        "checkpoint_role": checkpoint["role"],
        "checkpoint_path": str(Path(checkpoint["path"]).resolve()),
        "checkpoint_sha256": checkpoint["sha256"],
        "panel_path": str(
            _test_panel_path(
                root, int(job["tolerance_minutes"]), str(job["fold"])
            ).resolve()
        ),
        "panel_sha256": sha256_file(
            _test_panel_path(root, int(job["tolerance_minutes"]), str(job["fold"]))
        ),
        "test_overlay_path": str(overlay_path.resolve()),
        "test_overlay_sha256": sha256_file(overlay_path),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": prediction_sha,
        "prediction_mc_samples": 64,
        "noise_bank_profile_sha256": noise_bank_profile_sha,
        "inference_determinism_contract_path": str(determinism_contract.resolve()),
        "inference_determinism_contract_sha256": sha256_file(determinism_contract),
        "row_count": int(len(export)),
        "optional_metric_exclusion_count": int(len(metric_exclusions)),
    }
    manifest["payload_sha256"] = payload_sha256(manifest)
    write_json(_prediction_job_manifest_path(root, job), manifest)
    return manifest, evidence


def _prediction_manifest_path(root: Path) -> Path:
    return root / "analysis/prediction_manifest.csv"


def _pair_metrics_path(root: Path) -> Path:
    return root / "analysis/rq123_pair_metrics.csv.gz"


def _validate_frozen_predictions(root: Path) -> tuple[Path, Path]:
    config = validate_root(root)
    registry = read_registry(root)
    prediction_manifest = Path(str(registry.get("prediction_manifest_path", "")))
    pair_metrics = Path(str(registry.get("pair_metrics_path", "")))
    if (
        not bool(registry.get("predictions_frozen"))
        or not prediction_manifest.is_file()
        or sha256_file(prediction_manifest)
        != registry.get("prediction_manifest_sha256")
        or not pair_metrics.is_file()
        or sha256_file(pair_metrics) != registry.get("pair_metrics_sha256")
    ):
        raise ValueError("Frozen prediction bundle drift")
    checkpoint_map = _validate_checkpoint_allowlist(root)
    jobs = [dict(job) for job in registry.get("jobs", [])]
    manifest = pd.read_csv(prediction_manifest, dtype=str, keep_default_na=False)
    if len(manifest) != 400 or set(manifest["job_id"]) != {
        job["job_id"] for job in jobs
    }:
        raise ValueError("Prediction manifest must contain exactly 400 jobs")
    if manifest["job_id"].duplicated().any():
        raise ValueError("Prediction manifest contains duplicate jobs")
    manifest_by_job = manifest.set_index("job_id", drop=False)
    global_evidence = pd.read_csv(pair_metrics)
    if len(global_evidence) != 58_840:
        raise ValueError("Global pair evidence row-count drift")
    block_profiles: dict[tuple[int, str, int], set[str]] = {}
    for job in jobs:
        expected_pairs = _expected_counts(
            config, int(job["tolerance_minutes"]), str(job["fold"])
        )["test_pairs"]
        payload = _validate_prediction_job(
            root,
            job,
            checkpoint_map[str(job["job_id"])],
            expected_pairs=expected_pairs,
        )
        global_row = manifest_by_job.loc[str(job["job_id"])]
        per_job_manifest = _prediction_job_manifest_path(root, job)
        prediction_path = _prediction_path(root, job)
        if (
            global_row["prediction_path"] != str(prediction_path.resolve())
            or global_row["prediction_sha256"] != sha256_file(prediction_path)
            or global_row["prediction_manifest_path"] != str(per_job_manifest.resolve())
            or global_row["prediction_manifest_sha256"] != sha256_file(per_job_manifest)
            or global_row["noise_bank_profile_sha256"]
            != payload["noise_bank_profile_sha256"]
            or global_row["inference_determinism_contract_sha256"]
            != payload["inference_determinism_contract_sha256"]
        ):
            raise ValueError(f"Global prediction manifest drift: {job['job_id']}")
        block_profiles.setdefault(
            (
                int(job["tolerance_minutes"]),
                str(job["fold"]),
                int(job["seed"]),
            ),
            set(),
        ).add(str(payload["noise_bank_profile_sha256"]))
        cell = pd.read_csv(payload["pair_metrics_path"]).sort_values(
            "pair_id", kind="stable"
        )
        combined = global_evidence.loc[
            global_evidence["job_id"].astype(str).eq(str(job["job_id"]))
        ].sort_values("pair_id", kind="stable")
        if list(cell.columns) != list(combined.columns):
            raise ValueError(f"Global/cell evidence columns drift: {job['job_id']}")
        try:
            pd.testing.assert_frame_equal(
                cell.reset_index(drop=True),
                combined.reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as exc:
            raise ValueError(
                f"Global/cell pair evidence mismatch: {job['job_id']}"
            ) from exc
    if any(len(profiles) != 1 for profiles in block_profiles.values()):
        raise ValueError("Prediction arms do not share one frozen MC noise bank")
    from scripts.rq123.news_first_vol_film_nolp_10seed_analysis import (
        validate_paired_evidence,
    )

    validate_paired_evidence(
        pair_metrics,
        prediction_manifest,
        _checkpoint_manifest_path(root),
        expected_arms=ALL_ARMS,
        expected_seeds=SEEDS,
        expected_folds=FOLDS,
        expected_tolerances=TOLERANCES,
        expected_arms_by_tolerance={
            5: (PARENT_ARM, CONTINUATION_ARM, *TEXT_ARMS_5M),
            30: (PARENT_ARM, CONTINUATION_ARM, *TEXT_ARMS_30M),
        },
    )
    return prediction_manifest, pair_metrics


def predict(output_dir: str | Path, *, resume: bool = False) -> Path:
    root = Path(output_dir).resolve()
    freeze_evaluation(root)
    # A resumed prediction supervisor may be a fresh process. Establish the
    # frozen RNG/CUDA contract before any checkpoint is loaded for inference.
    _configure_prediction_determinism(SEEDS[0])
    config = validate_root(root)
    registry = read_registry(root)
    if bool(registry.get("predictions_frozen")):
        _validate_frozen_predictions(root)
        return root
    checkpoints = _validate_checkpoint_allowlist(root)
    _validate_test_inputs(root)
    prediction_rows: list[dict[str, Any]] = []
    evidence_frames: list[pd.DataFrame] = []
    jobs = sorted(
        (dict(job) for job in registry.get("jobs", [])),
        key=lambda job: (
            int(job["tolerance_minutes"]),
            str(job["fold"]),
            int(job["seed"]),
            str(job["arm"]),
        ),
    )
    for job in jobs:
        checkpoint = checkpoints[str(job["job_id"])]
        expected_pairs = _expected_counts(
            config, int(job["tolerance_minutes"]), str(job["fold"])
        )["test_pairs"]
        path = _prediction_path(root, job)
        manifest_path = _prediction_job_manifest_path(root, job)
        evidence_path = manifest_path.with_suffix(".pair_metrics.csv")
        if path.is_file() and manifest_path.is_file() and evidence_path.is_file():
            payload = _validate_prediction_job(
                root, job, checkpoint, expected_pairs=expected_pairs
            )
            evidence = pd.read_csv(evidence_path)
        else:
            if (
                path.is_file() or manifest_path.is_file() or evidence_path.is_file()
            ) and not resume:
                raise ValueError(
                    f"Partial prediction requires --resume: {job['job_id']}"
                )
            payload, evidence = _evaluate_prediction_job(
                root, job, checkpoint, expected_pairs=expected_pairs
            )
        evidence_path = manifest_path.with_suffix(".pair_metrics.csv")
        _write_dataframe_csv(evidence_path, evidence)
        payload = read_json(manifest_path)
        payload.pop("payload_sha256", None)
        payload.update(
            pair_metrics_path=str(evidence_path.resolve()),
            pair_metrics_sha256=sha256_file(evidence_path),
            pair_metrics_row_count=int(len(evidence)),
        )
        payload["payload_sha256"] = payload_sha256(payload)
        write_json(manifest_path, payload)
        prediction_rows.append(
            {
                "job_id": job["job_id"],
                "arm": job["arm"],
                "fold": job["fold"],
                "seed": int(job["seed"]),
                "tolerance_minutes": int(job["tolerance_minutes"]),
                "prediction_path": str(path.resolve()),
                "prediction_sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "prediction_manifest_path": str(manifest_path.resolve()),
                "prediction_manifest_sha256": sha256_file(manifest_path),
                "noise_bank_profile_sha256": payload["noise_bank_profile_sha256"],
                "inference_determinism_contract_sha256": payload[
                    "inference_determinism_contract_sha256"
                ],
            }
        )
        evidence_frames.append(evidence)
    if len(prediction_rows) != 400 or len(evidence_frames) != 400:
        raise ValueError("Prediction stage did not produce exactly 400 cells")
    prediction_manifest = write_csv(
        _prediction_manifest_path(root), prediction_rows, tuple(prediction_rows[0])
    )
    pair_metrics = pd.concat(evidence_frames, ignore_index=True)
    _write_dataframe_csv(_pair_metrics_path(root), pair_metrics, gzip=True)
    registry = read_registry(root)
    registry.update(
        status="predictions_frozen",
        predictions_frozen=True,
        predictions_frozen_at_utc=utc_now(),
        prediction_manifest_path=str(prediction_manifest.resolve()),
        prediction_manifest_sha256=sha256_file(prediction_manifest),
        pair_metrics_path=str(_pair_metrics_path(root).resolve()),
        pair_metrics_sha256=sha256_file(_pair_metrics_path(root)),
        prediction_cell_count=400,
    )
    write_registry(root, registry)
    write_experiment_status(root, "predictions_frozen", prediction_cells=400)
    _validate_frozen_predictions(root)
    return root


def _analysis_manifest_path(root: Path) -> Path:
    return root / "analysis/unified/analysis_artifact_manifest.json"


def _validate_analysis_bundle(root: Path) -> Path:
    from scripts.rq123.news_first_vol_film_nolp_10seed_analysis import (
        validate_frozen_artifact_manifest,
    )

    registry = read_registry(root)
    path = Path(str(registry.get("analysis_manifest_path", "")))
    if (
        not path.is_file()
        or path != _analysis_manifest_path(root)
        or sha256_file(path) != registry.get("analysis_manifest_sha256")
    ):
        raise ValueError("Unified analysis manifest drift")
    validate_frozen_artifact_manifest(path)
    return path


def _ensure_unified_analysis(root: Path, *, resume: bool) -> Path:
    from scripts.rq123.news_first_vol_film_nolp_10seed_analysis import (
        run_unified_analysis,
    )

    config = validate_root(root)
    prediction_manifest, pair_metrics = _validate_frozen_predictions(root)
    registry = read_registry(root)
    if registry.get("analysis_manifest_path"):
        return _validate_analysis_bundle(root)
    destination = root / "analysis/unified"
    if destination.exists():
        if not resume:
            raise RuntimeError("Partial analysis exists; use --resume")
        # A prior interrupted attempt is preserved for audit.  It is never
        # scanned or reused as evidence.
        attempt = int(registry.get("analysis_attempt", 1)) + 1
        archived = root / "analysis" / f"unified_interrupted_attempt_{attempt - 1:02d}"
        if archived.exists():
            raise RuntimeError(f"Analysis recovery target already exists: {archived}")
        os.replace(destination, archived)
    else:
        attempt = int(registry.get("analysis_attempt", 0)) + 1
    registry.update(
        status="analysis_running",
        analysis_attempt=attempt,
        analysis_started_at_utc=utc_now(),
    )
    write_registry(root, registry)
    event_manifest = read_json(root / "evaluation/frozen_event_sources.json")
    analysis = _mapping(config["analysis"], "analysis")
    scheduled = _mapping(analysis["scheduled"], "analysis.scheduled")
    market = _mapping(analysis["market_jump"], "analysis.market_jump")
    run_unified_analysis(
        pair_metrics_path=pair_metrics,
        pair_metrics_sha256=sha256_file(pair_metrics),
        prediction_manifest_path=prediction_manifest,
        prediction_manifest_sha256=sha256_file(prediction_manifest),
        checkpoint_manifest_path=_checkpoint_manifest_path(root),
        checkpoint_manifest_sha256=sha256_file(_checkpoint_manifest_path(root)),
        scheduled_events_path=event_manifest["scheduled_events_path"],
        scheduled_events_sha256=event_manifest["scheduled_events_sha256"],
        market_jump_path=event_manifest["market_jump_pairs_path"],
        market_jump_sha256=event_manifest["market_jump_pairs_sha256"],
        output_dir=destination,
        expected_arms=ALL_ARMS,
        expected_seeds=SEEDS,
        expected_folds=FOLDS,
        expected_tolerances=TOLERANCES,
        expected_arms_by_tolerance={
            5: (PARENT_ARM, CONTINUATION_ARM, *TEXT_ARMS_5M),
            30: (PARENT_ARM, CONTINUATION_ARM, *TEXT_ARMS_30M),
        },
        bootstrap_iterations=int(analysis["bootstrap_replicates"]),
        bootstrap_seed=int(analysis["bootstrap_seed"]),
        ordinary_minimum_distance_minutes=int(
            scheduled["ordinary_minimum_distance_minutes"]
        ),
        scheduled_clock_caliper_minutes=int(scheduled["clock_caliper_minutes"]),
        scheduled_minimum_primary_pairs=int(scheduled["minimum_primary_pairs"]),
        scheduled_minimum_primary_releases=int(scheduled["minimum_primary_releases"]),
        scheduled_minimum_primary_sessions=int(scheduled["minimum_primary_sessions"]),
        market_jump_minimum_pairs=int(market["minimum_pairs"]),
        market_jump_minimum_sessions=int(market["minimum_sessions"]),
        required_market_root=resolve_path(config["data"]["market_jump_root"]),
    )
    manifest = _analysis_manifest_path(root)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    registry = read_registry(root)
    registry.update(
        status="analysis_complete",
        analysis_manifest_path=str(manifest.resolve()),
        analysis_manifest_sha256=sha256_file(manifest),
        analysis_completed_at_utc=utc_now(),
    )
    write_registry(root, registry)
    write_experiment_status(root, "analysis_complete", analysis_attempt=attempt)
    return _validate_analysis_bundle(root)


def _mark_analysis_question(root: Path, question: str, *, resume: bool) -> Path:
    if question not in {"rq1", "rq2", "rq3"}:
        raise ValueError(question)
    manifest = _ensure_unified_analysis(root, resume=resume)
    registry = read_registry(root)
    registry[f"{question}_evaluated"] = True
    registry[f"{question}_evaluated_at_utc"] = utc_now()
    registry["status"] = f"{question}_evaluated"
    write_registry(root, registry)
    write_experiment_status(root, registry["status"], analysis_manifest=str(manifest))
    return root


def evaluate_rq1(output_dir: str | Path, *, resume: bool = False) -> Path:
    return _mark_analysis_question(Path(output_dir).resolve(), "rq1", resume=resume)


def evaluate_rq2(output_dir: str | Path, *, resume: bool = False) -> Path:
    return _mark_analysis_question(Path(output_dir).resolve(), "rq2", resume=resume)


def evaluate_rq3(output_dir: str | Path, *, resume: bool = False) -> Path:
    return _mark_analysis_question(Path(output_dir).resolve(), "rq3", resume=resume)


def _resource_summary(root: Path) -> Path:
    source = root / "resource_usage.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    required = {"gpu_index", "memory_used_mib"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Formal resource telemetry is incomplete")
    if (
        "sample_status" in frame
        and not frame["sample_status"].astype(str).eq("ok").all()
    ):
        raise ValueError("Formal resource telemetry contains failed samples")
    utilization_column = next(
        (
            column
            for column in ("utilization_gpu_pct", "gpu_utilization_pct")
            if column in frame
        ),
        None,
    )
    rows: list[dict[str, Any]] = []
    for gpu, group in frame.groupby("gpu_index", sort=True):
        rows.append(
            {
                "gpu_index": int(gpu),
                "telemetry_rows": int(len(group)),
                "peak_memory_used_mib": float(
                    pd.to_numeric(group["memory_used_mib"], errors="raise").max()
                ),
                "mean_gpu_utilization_pct": (
                    float(
                        pd.to_numeric(group[utilization_column], errors="raise").mean()
                    )
                    if utilization_column is not None
                    else math.nan
                ),
            }
        )
    if {row["gpu_index"] for row in rows} != {0, 1}:
        raise ValueError("Resource summary requires GPU0 and GPU1")
    return write_csv(root / "resource_summary.csv", rows, tuple(rows[0]))


def _terminal_file_universe(root: Path) -> list[Path]:
    excluded = {
        (root / "output_hashes.csv").resolve(),
        _registry_path(root).resolve(),
        _experiment_status_path(root).resolve(),
    }
    return sorted(
        (
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.resolve() not in excluded
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _validate_final_snapshot(
    root: Path, *, allow_postprocessing: bool = False
) -> dict[str, Any]:
    payload = read_json(root / "registry/final_registry_snapshot.json")
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    snapshot_registry = _mapping(payload.get("registry"), "snapshot.registry")
    snapshot_status = _mapping(
        payload.get("experiment_status"), "snapshot.experiment_status"
    )
    live = read_registry(root)
    live_status = read_json(_experiment_status_path(root))
    anchors = {
        "terminal_output_manifest_path",
        "terminal_output_manifest_sha256",
        "terminal_output_artifact_count",
    }
    normalized_live = {key: value for key, value in live.items() if key not in anchors}
    valid_header = (
        payload_sha256(unsigned) != payload.get("payload_sha256")
        or snapshot_registry.get("status") != "completed"
        or not bool(snapshot_registry.get("terminal_complete"))
        or snapshot_status.get("status") != "completed"
    )
    if valid_header:
        raise ValueError("Final registry snapshot drift")
    if live.get("status") == "completed":
        if snapshot_registry != normalized_live or snapshot_status != live_status:
            raise ValueError("Final registry/status snapshot drift")
        return payload
    if not allow_postprocessing or live.get("status") != "postprocessing":
        raise ValueError("Final registry snapshot is not yet live")
    transient_registry = {
        *anchors,
        "status",
        "terminal_complete",
        "completed_at_utc",
        "updated_at_utc",
    }
    stable_snapshot = {
        key: value
        for key, value in snapshot_registry.items()
        if key not in transient_registry
    }
    stable_live = {
        key: value for key, value in live.items() if key not in transient_registry
    }
    transient_status = {"status", "updated_at_utc"}
    if (
        stable_snapshot != stable_live
        or {
            key: value
            for key, value in snapshot_status.items()
            if key not in transient_status
        }
        != {
            key: value
            for key, value in live_status.items()
            if key not in transient_status
        }
        or live_status.get("status") not in {"postprocessing", "completed"}
    ):
        raise ValueError("Postprocessing snapshot/live state drift")
    return payload


def _validate_report_manifest(root: Path) -> Path:
    path = root / "report/report_manifest.json"
    payload = read_json(path)
    analysis_manifest = _validate_analysis_bundle(root)
    analysis = _mapping(payload.get("analysis_manifest"), "report.analysis_manifest")
    reports = payload.get("reports")
    expected_paths = {
        (root / "report/news_first_vol_film_nolp_10seed_unified_report.md").resolve(),
        (root / "report/news_first_vol_film_nolp_10seed_unified_report.html").resolve(),
    }
    if (
        payload.get("interpretation") != INTERPRETATION
        or analysis.get("path") != str(analysis_manifest.resolve())
        or analysis.get("sha256") != sha256_file(analysis_manifest)
        or not isinstance(reports, list)
        or len(reports) != 2
    ):
        raise ValueError("Report manifest input lineage drift")
    observed_paths: set[Path] = set()
    for row in reports:
        if not isinstance(row, Mapping):
            raise ValueError("Report manifest rows must be mappings")
        report_path = Path(str(row.get("path", ""))).resolve()
        if (
            report_path in observed_paths
            or not report_path.is_file()
            or report_path.stat().st_size != int(row.get("size_bytes", -1))
            or sha256_file(report_path) != row.get("sha256")
        ):
            raise ValueError(f"Report artifact drift: {report_path}")
        observed_paths.add(report_path)
    if observed_paths != expected_paths:
        raise ValueError("Report artifact universe drift")
    return path


def _validate_terminal_qa_payload(root: Path, *, expected_count: int) -> Path:
    qa_path = root / "qa.json"
    qa = read_json(qa_path)
    unsigned = {key: value for key, value in qa.items() if key != "payload_sha256"}
    if (
        qa.get("kind") != "rq123_terminal_qa_v1"
        or qa.get("status") != "passed"
        or qa.get("interpretation") != INTERPRETATION
        or payload_sha256(unsigned) != qa.get("payload_sha256")
        or int(qa.get("training_jobs_completed", -1)) != 400
        or int(qa.get("prediction_cells", -1)) != 400
        or int(qa.get("pair_metric_rows", -1)) != 58_840
        or int(qa.get("terminal_output_artifact_count", -1)) != expected_count
    ):
        raise ValueError("Terminal QA payload drift")
    return qa_path


def _validate_terminal_output_manifest(
    root: Path, *, require_registry_anchor: bool
) -> tuple[Path, int]:
    output_manifest = root / "output_hashes.csv"
    rows = pd.read_csv(output_manifest, dtype=str, keep_default_na=False)
    expected_paths = {
        path.relative_to(root).as_posix() for path in _terminal_file_universe(root)
    }
    if (
        len(rows) != len(set(rows["relative_path"]))
        or len(rows) != len(set(rows["path"]))
        or len(rows) != len(set(rows["artifact_role"]))
        or set(rows["relative_path"]) != expected_paths
    ):
        raise ValueError("Terminal output manifest universe drift")
    for row in rows.itertuples(index=False):
        path = root / row.relative_path
        if (
            row.path != str(path.resolve())
            or row.artifact_role != f"output:{row.relative_path}"
            or path.stat().st_size != int(row.size_bytes)
            or sha256_file(path) != row.sha256
        ):
            raise ValueError(f"Terminal output hash drift: {path}")
    if require_registry_anchor:
        registry = read_registry(root)
        if (
            registry.get("terminal_output_manifest_path")
            != str(output_manifest.resolve())
            or registry.get("terminal_output_manifest_sha256")
            != sha256_file(output_manifest)
            or int(registry.get("terminal_output_artifact_count", -1)) != len(rows)
        ):
            raise ValueError("Terminal output registry anchor drift")
    _validate_terminal_qa_payload(root, expected_count=len(rows))
    return output_manifest, len(rows)


def qa_experiment(output_dir: str | Path, *, resume: bool = False) -> Path:
    del resume
    root = Path(output_dir).resolve()
    validate_root(root)
    registry = read_registry(root)
    allow_postprocessing = registry.get("status") == "postprocessing"
    if len(registry.get("jobs", [])) != 400:
        raise ValueError("Terminal QA requires exactly 400 jobs")
    for stage in STAGES:
        _validate_stage_complete(root, stage)
    _validate_parent_state_allowlist(root)
    _validate_recipe_manifest(root)
    _validate_checkpoint_allowlist(root)
    _validate_test_inputs(root)
    _validate_frozen_predictions(root)
    _validate_analysis_bundle(root)
    if not all(bool(registry.get(f"rq{index}_evaluated")) for index in (1, 2, 3)):
        raise ValueError("Terminal QA requires completed RQ1/RQ2/RQ3 analyses")
    required = (
        root / "report/news_first_vol_film_nolp_10seed_unified_report.md",
        root / "report/news_first_vol_film_nolp_10seed_unified_report.html",
        root / "report/report_manifest.json",
        root / "resource_summary.csv",
        root / "registry/final_registry_snapshot.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Terminal artifacts missing: {missing}")
    _validate_report_manifest(root)
    _validate_final_snapshot(root, allow_postprocessing=allow_postprocessing)
    output_manifest = root / "output_hashes.csv"
    qa_path = root / "qa.json"
    if output_manifest.is_file():
        _validate_terminal_output_manifest(
            root, require_registry_anchor=not allow_postprocessing
        )
        return qa_path
    expected_count = len(_terminal_file_universe(root)) + (
        0 if qa_path.is_file() else 1
    )
    if qa_path.is_file():
        return _validate_terminal_qa_payload(root, expected_count=expected_count)
    qa = {
        "schema_version": 1,
        "kind": "rq123_terminal_qa_v1",
        "status": "passed",
        "interpretation": INTERPRETATION,
        "training_jobs_completed": 400,
        "parent_full_states": 80,
        "branch_recipes": 80,
        "prediction_cells": 400,
        "pair_metric_rows": 58_840,
        "rq1_evaluated": True,
        "rq2_evaluated": True,
        "rq3_evaluated": True,
        "terminal_output_artifact_count": expected_count,
        "completed_at_utc": utc_now(),
    }
    actual_pair_rows = len(pd.read_csv(_pair_metrics_path(root)))
    if actual_pair_rows != int(qa["pair_metric_rows"]):
        raise ValueError(f"Pair evidence row-count drift: {actual_pair_rows}")
    qa["payload_sha256"] = payload_sha256(qa)
    return write_json(qa_path, qa)


def postprocess(output_dir: str | Path, *, resume: bool = False) -> Path:
    from scripts.rq123.news_first_vol_film_nolp_10seed_report import (
        render_unified_report,
    )

    root = Path(output_dir).resolve()
    validate_root(root)
    registry = read_registry(root)
    if registry.get("status") == "completed" and bool(
        registry.get("terminal_complete")
    ):
        terminal_paths = (
            root / "registry/final_registry_snapshot.json",
            root / "qa.json",
            root / "output_hashes.csv",
        )
        if all(path.is_file() for path in terminal_paths):
            return qa_experiment(root, resume=True)
        if not resume:
            raise RuntimeError("Interrupted terminal commit requires --resume")
    if not all(bool(registry.get(f"rq{index}_evaluated")) for index in (1, 2, 3)):
        raise RuntimeError("Postprocess requires completed RQ1/RQ2/RQ3 analyses")
    manifest = _validate_analysis_bundle(root)
    report_dir = root / "report"
    report_manifest = report_dir / "report_manifest.json"
    if report_manifest.exists():
        if not resume:
            raise RuntimeError("Report already exists; use --resume")
        _validate_report_manifest(root)
    else:
        render_unified_report(
            analysis_manifest_path=manifest,
            analysis_manifest_sha256=sha256_file(manifest),
            output_dir=report_dir,
        )
        _validate_report_manifest(root)
    resource = root / "resource_summary.csv"
    if not resource.is_file():
        _resource_summary(root)

    snapshot_path = root / "registry/final_registry_snapshot.json"
    output_manifest = root / "output_hashes.csv"
    if not snapshot_path.is_file():
        registry = read_registry(root)
        for key in (
            "terminal_output_manifest_path",
            "terminal_output_manifest_sha256",
            "terminal_output_artifact_count",
        ):
            registry.pop(key, None)
        registry.update(
            status="postprocessing",
            terminal_complete=False,
            postprocessing_started_at_utc=registry.get(
                "postprocessing_started_at_utc", utc_now()
            ),
        )
        write_registry(root, registry)
        write_experiment_status(root, "postprocessing", current_stage="terminal")
        live = read_registry(root)
        final_timestamp = str(live.get("completed_at_utc") or utc_now())
        intended_registry = {
            **live,
            "status": "completed",
            "terminal_complete": True,
            "completed_at_utc": final_timestamp,
            "updated_at_utc": final_timestamp,
        }
        intended_registry["jobs_sha256"] = payload_sha256(
            intended_registry.get("jobs", [])
        )
        intended_status = {
            "status": "completed",
            "updated_at_utc": final_timestamp,
            "current_stage": "terminal",
        }
        snapshot = {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "registry": intended_registry,
            "experiment_status": intended_status,
            "created_at_utc": final_timestamp,
        }
        snapshot["payload_sha256"] = payload_sha256(snapshot)
        write_json(snapshot_path, snapshot)
    snapshot = _validate_final_snapshot(root, allow_postprocessing=True)
    qa_experiment(root, resume=True)

    if output_manifest.is_file():
        _, artifact_count = _validate_terminal_output_manifest(
            root, require_registry_anchor=False
        )
    else:
        files = _terminal_file_universe(root)
        rows = [
            {
                "artifact_role": f"output:{path.relative_to(root).as_posix()}",
                "relative_path": path.relative_to(root).as_posix(),
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
        output_manifest = write_csv(output_manifest, rows, tuple(rows[0]))
        _, artifact_count = _validate_terminal_output_manifest(
            root, require_registry_anchor=False
        )

    intended_registry = dict(snapshot["registry"])
    intended_registry.update(
        terminal_output_manifest_path=str(output_manifest.resolve()),
        terminal_output_manifest_sha256=sha256_file(output_manifest),
        terminal_output_artifact_count=int(artifact_count),
    )
    # The experiment-status file is installed first.  The registry's
    # `status=completed` transition is the final commit point.
    write_json(_experiment_status_path(root), snapshot["experiment_status"])
    write_registry(
        root,
        intended_registry,
        updated_at_utc=str(snapshot["registry"]["updated_at_utc"]),
    )
    return qa_experiment(root, resume=True)


_PIPELINE_LOCK_DESCRIPTORS: dict[Path, int] = {}


def _pipeline_lock(root: Path) -> Path:
    control = root.with_name(root.name + "_control")
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / "pipeline.lock").resolve()
    if lock in _PIPELINE_LOCK_DESCRIPTORS:
        raise RuntimeError(f"Live pipeline supervisor already exists: {lock}")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError(f"Live pipeline supervisor already exists: {lock}") from exc
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.fsync(descriptor)
    _PIPELINE_LOCK_DESCRIPTORS[lock] = descriptor
    (control / "pipeline.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    return lock


def _release_pipeline_lock(lock: Path) -> None:
    resolved = lock.resolve()
    descriptor = _PIPELINE_LOCK_DESCRIPTORS.pop(resolved, None)
    if descriptor is None:
        return
    _release_file_lock(resolved, descriptor)


class _PipelineInterrupted(RuntimeError):
    pass


def _pipeline_signal_handler(signum: int, frame: object) -> None:
    del frame
    raise _PipelineInterrupted(f"Pipeline interrupted by signal {signum}")


def run_pipeline(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> Path:
    source_config = load_config(config_path)
    root = _validated_formal_root(source_config, output_dir)
    lock = _pipeline_lock(root)
    previous_handlers: dict[int, Any] = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _pipeline_signal_handler)
        if root.is_dir():
            try:
                registry = read_registry(root)
            except (FileNotFoundError, ValueError):
                registry = {}
            if registry.get("status") in {"completed", "postprocessing"}:
                postprocess(root, resume=True)
                return root
        if not _benchmark_result_path(root).is_file():
            run_benchmark(config_path, root, resume=resume)
        run_recovery_canary(config_path, root, resume=True)
        # The formal root is intentionally absent here.  The complete 400-cell
        # one-epoch smoke runs only after the 36-cell capacity benchmark has
        # frozen 12/18 workers per GPU, then deletes its temporary root after a
        # self-hashed evidence bundle is durable in the control directory.
        run_matrix_smoke(config_path, root, resume=True)
        prepare_experiment(config_path, root, resume=True)
        registry = read_registry(root)
        if not registry.get("parent_states_frozen"):
            _launch_stage(root, PARENT_STAGE, dry_run=True, resume=True)
            launch_parents(root, resume=True)
        registry = read_registry(root)
        if not registry.get("branch_recipes_frozen"):
            _launch_stage(root, CONTINUATION_STAGE, dry_run=True, resume=True)
            launch_continuations(root, resume=True)
            freeze_branch_recipes(root, resume=True)
        if not read_registry(root).get("evaluation_frozen"):
            _launch_stage(root, BRANCH_STAGE, dry_run=True, resume=True)
            launch_text_branches(root, resume=True)
            freeze_evaluation(root, resume=True)
        if not read_registry(root).get("predictions_frozen"):
            predict(root, resume=True)
        if not read_registry(root).get("rq1_evaluated"):
            evaluate_rq1(root, resume=True)
        if not read_registry(root).get("rq2_evaluated"):
            evaluate_rq2(root, resume=True)
        if not read_registry(root).get("rq3_evaluated"):
            evaluate_rq3(root, resume=True)
        postprocess(root, resume=True)
        return root
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        _release_pipeline_lock(lock)


def _prepare_benchmark_root(
    config_path: str | Path,
    benchmark_root: Path,
    *,
    workers_per_gpu: int,
    resume: bool = False,
) -> Path:
    source_config = load_config(config_path)
    config = _with_selected_workers(source_config, workers_per_gpu)
    if benchmark_root.exists():
        if not resume:
            raise FileExistsError(benchmark_root)
        try:
            existing = validate_root(benchmark_root)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            marker = _validate_prepare_contract(
                benchmark_root,
                config,
                workers_per_gpu=workers_per_gpu,
                mode="benchmark",
            )
            if marker.get("status") != "preparing":
                raise ValueError("Invalid benchmark root is not safely resumable")
            registry_path = _registry_path(benchmark_root)
            if registry_path.is_file():
                partial_registry = read_registry(benchmark_root)
                for job in partial_registry.get("jobs") or []:
                    status_path = _status_path(benchmark_root, str(job["job_id"]))
                    if not status_path.is_file():
                        continue
                    status = read_json(status_path)
                    if (
                        status.get("status") != "pending"
                        or int(status.get("attempt", 0)) != 0
                        or status.get("artifacts")
                    ):
                        raise ValueError(
                            "Benchmark preparation cannot resume after training started"
                        )
        else:
            if existing != config:
                raise ValueError("Existing benchmark resolved config drift")
            registry = read_registry(benchmark_root)
            if int(registry.get("formal_workers_per_gpu", -1)) != int(workers_per_gpu):
                raise ValueError("Existing benchmark concurrency drift")
            return benchmark_root
    else:
        benchmark_root.mkdir(parents=True)
    _write_prepare_contract(
        benchmark_root,
        config,
        workers_per_gpu=workers_per_gpu,
        mode="benchmark",
        status="preparing",
    )
    for relative in (
        "registry/job_status",
        "configs/jobs",
        "configs/full_state_contracts",
        "inputs/pair_text_overlays",
        "logs",
    ):
        (benchmark_root / relative).mkdir(parents=True, exist_ok=True)
    write_yaml(benchmark_root / "resolved_config.yaml", config)
    write_json(benchmark_root / "grid_contract.json", grid_contract(config))
    write_json(benchmark_root / "model_contract.json", model_contract(config))
    write_json(
        _runtime_contract_path(benchmark_root),
        _runtime_contract(
            config,
            workers_per_gpu=workers_per_gpu,
            mode="benchmark_candidate",
        ),
    )
    materialize_pair_universes(config, benchmark_root)
    materialize_pair_text_overlays(config, benchmark_root)
    specs = experiment_specs(BENCHMARK_STAGE)
    # Benchmark assignment is deliberately 18/18 and each physical GPU sees
    # both tolerances and every text arm.  A fallback root changes only waves.
    counts = {0: 0, 1: 0}
    assigned: list[dict[str, Any]] = []
    for raw in specs:
        spec = dict(raw)
        gpu = int(spec["gpu_id"])
        local = counts[gpu]
        counts[gpu] += 1
        spec["gpu_slot"] = local % int(workers_per_gpu)
        spec["wave"] = local // int(workers_per_gpu) + 1
        assigned.append(spec)
    if counts != {0: 18, 1: 18}:
        raise AssertionError(f"Benchmark GPU assignment drift: {counts}")
    jobs: list[dict[str, Any]] = []
    for spec in assigned:
        job, _ = build_job(
            config,
            benchmark_root,
            spec,
            slots_per_gpu=workers_per_gpu,
        )
        jobs.append(job)
        write_json(_status_path(benchmark_root, job["job_id"]), initial_job_status(job))
    write_registry(
        benchmark_root,
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "interpretation": INTERPRETATION,
            "status": "benchmark_prepared",
            "created_at_utc": utc_now(),
            "formal_workers_per_gpu": workers_per_gpu,
            "expected_training_jobs": 36,
            "expected_prediction_cells": 0,
            "jobs": jobs,
            "parent_states_frozen": False,
            "branch_recipes_frozen": False,
            "evaluation_frozen": False,
            "predictions_frozen": False,
            "test_data_opened": False,
            "terminal_complete": False,
        },
    )
    write_experiment_status(benchmark_root, "benchmark_prepared", registered_jobs=36)
    _prepare_hash_manifests(config, benchmark_root)
    _write_prepare_contract(
        benchmark_root,
        config,
        workers_per_gpu=workers_per_gpu,
        mode="benchmark",
        status="prepared",
    )
    validate_root(benchmark_root)
    return benchmark_root


def _checkpoint_state(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("state_dict"), Mapping
    ):
        raise ValueError(f"Invalid model checkpoint: {path}")
    return dict(payload["state_dict"])


def _validate_training_metrics(
    path: Path, *, require_exactly_one_learned_epoch: bool
) -> dict[str, Any]:
    """Validate finite learned losses while allowing epoch-0 train diagnostics to be NaN."""

    frame = pd.read_csv(path)
    required_columns = (
        "epoch",
        "d_total",
        "g_total",
        "g_recon",
        "gp",
        "val_recon",
        "g_lr",
        "d_lr",
    )
    missing = [column for column in required_columns if column not in frame]
    if missing or frame.empty:
        raise ValueError(f"Training metrics are incomplete: missing={missing}")
    epochs = pd.to_numeric(frame["epoch"], errors="raise").astype(int)
    if epochs.duplicated().any() or (epochs < 0).any():
        raise ValueError("Training metric epochs must be unique non-negative integers")
    learned = frame.loc[epochs >= 1]
    if learned.empty:
        raise ValueError("Training metrics contain no learned epoch")
    if require_exactly_one_learned_epoch and set(epochs.tolist()) not in ({1}, {0, 1}):
        raise ValueError("One-epoch smoke metrics must contain only epoch 0/1")
    for column in required_columns[1:]:
        values = pd.to_numeric(learned[column], errors="coerce")
        if values.isna().any() or not all(
            math.isfinite(float(value)) for value in values
        ):
            raise ValueError(f"Learned metric {column!r} contains NaN/Inf")
    epoch_zero = frame.loc[epochs == 0]
    if not epoch_zero.empty:
        # Initial-checkpoint rows have no training step, so their training-loss
        # fields are intentionally NaN.  Validation and LR fields remain strict.
        for column in ("val_recon", "g_lr", "d_lr"):
            values = pd.to_numeric(epoch_zero[column], errors="coerce")
            if values.isna().any() or not all(
                math.isfinite(float(value)) for value in values
            ):
                raise ValueError(f"Epoch-0 metric {column!r} contains NaN/Inf")
    numeric = frame.select_dtypes(include="number")
    for value in numeric.to_numpy().ravel():
        if pd.notna(value) and not math.isfinite(float(value)):
            raise ValueError("Training metrics contain an infinite numeric value")
    return {
        "row_count": int(len(frame)),
        "learned_epoch_count": int(len(learned)),
        "maximum_epoch": int(epochs.max()),
        "metrics_sha256": sha256_file(path),
    }


def _state_dicts_differ(initial_path: Path, final_path: Path) -> bool:
    import torch

    initial = _checkpoint_state(initial_path)
    final = _checkpoint_state(final_path)
    return initial.keys() == final.keys() and any(
        not torch.equal(initial[key], final[key]) for key in initial
    )


def _state_dicts_equal(first_path: Path, second_path: Path) -> bool:
    import torch

    first = _checkpoint_state(first_path)
    second = _checkpoint_state(second_path)
    return first.keys() == second.keys() and all(
        torch.equal(first[key], second[key]) for key in first
    )


def _assert_stage_one_epoch_updates(root: Path, stage: str) -> list[dict[str, Any]]:
    jobs = _validate_stage_complete(root, stage)
    evidence: list[dict[str, Any]] = []
    for job in jobs:
        status = read_json(_status_path(root, str(job["job_id"])))
        initial_g = _artifact(status, "generator_initial_epoch0")
        final_g = _artifact(status, "generator_final")
        initial_d = _artifact(status, "discriminator_initial_epoch0")
        final_d = _artifact(status, "discriminator_final")
        generator_updated = _state_dicts_differ(
            Path(initial_g["path"]), Path(final_g["path"])
        )
        critic_updated = _state_dicts_differ(
            Path(initial_d["path"]), Path(final_d["path"])
        )
        if not generator_updated:
            raise ValueError(f"One-epoch Generator did not update: {job['job_id']}")
        if not critic_updated:
            raise ValueError(f"One-epoch Critic did not update: {job['job_id']}")
        metrics = _validate_training_metrics(
            Path(_artifact(status, "training_metrics_csv")["path"]),
            require_exactly_one_learned_epoch=True,
        )
        evidence.append(
            {
                "job_id": str(job["job_id"]),
                "generator_updated": True,
                "critic_updated": True,
                **metrics,
            }
        )
    return evidence


def _assert_benchmark_updates(root: Path) -> None:
    _assert_stage_one_epoch_updates(root, BENCHMARK_STAGE)


def _resource_peaks(root: Path) -> dict[str, Any]:
    path = root / "resource_usage.csv"
    frame = pd.read_csv(path)
    if frame.empty or "gpu_index" not in frame or "memory_used_mib" not in frame:
        raise ValueError("Benchmark resource telemetry is incomplete")
    if (
        "sample_status" in frame
        and not frame["sample_status"].astype(str).eq("ok").all()
    ):
        raise ValueError("Benchmark resource telemetry contains failed samples")
    peak_by_gpu = {
        str(int(gpu)): float(group["memory_used_mib"].astype(float).max())
        for gpu, group in frame.groupby("gpu_index")
    }
    if set(peak_by_gpu) != {"0", "1"}:
        raise ValueError(f"Expected benchmark telemetry for GPU0/GPU1: {peak_by_gpu}")
    return {
        "peak_memory_mib_by_gpu": peak_by_gpu,
        "peak_memory_gib_by_gpu": {
            key: value / 1024.0 for key, value in peak_by_gpu.items()
        },
        "telemetry_rows": int(len(frame)),
        "telemetry_sha256": sha256_file(path),
    }


def _project_formal_storage_breakdown(benchmark_root: Path) -> dict[str, int]:
    """Project formal storage from benchmark role maxima and stage cardinality."""

    jobs = _stage_jobs(benchmark_root, BENCHMARK_STAGE)
    sizes_by_role: dict[str, list[int]] = {}
    for job in jobs:
        status = read_json(_status_path(benchmark_root, str(job["job_id"])))
        if not _completed_valid(job, status):
            raise ValueError(
                f"Cannot project from incomplete benchmark: {job['job_id']}"
            )
        for row in status["artifacts"]:
            sizes_by_role.setdefault(str(row["artifact_role"]), []).append(
                int(row["size_bytes"])
            )

    def maximum(role: str) -> int:
        values = sizes_by_role.get(role, [])
        if not values:
            raise ValueError(f"Benchmark does not measure artifact role {role!r}")
        return max(values)

    dynamic_jobs = (
        EXPECTED_STAGE_COUNTS[PARENT_STAGE] + EXPECTED_STAGE_COUNTS[CONTINUATION_STAGE]
    )
    branch_jobs = EXPECTED_STAGE_COUNTS[BRANCH_STAGE]
    training_jobs = dynamic_jobs + branch_jobs
    checkpoint_roles = (
        "generator_best_learned",
        "discriminator_best_learned",
    )
    dynamic_checkpoints = sum(maximum(role) for role in checkpoint_roles) * dynamic_jobs
    branch_checkpoints = (
        maximum("generator_final") + maximum("discriminator_final")
    ) * branch_jobs
    full_states = maximum("full_training_state") * dynamic_jobs
    best_checkpoint_metadata = maximum("best_learned_checkpoint") * dynamic_jobs
    per_job_contracts = (
        maximum("resolved_training_config") + maximum("full_state_contract")
    ) * training_jobs
    # Benchmark metrics/logs cover one epoch. Scale their maxima by the formal
    # epoch cap so a long, non-early-stopped cell cannot invalidate the disk gate.
    config = validate_root(benchmark_root)
    epoch_cap = int(config["training"]["num_epochs"])
    epoch_scaled_diagnostics = (
        sum(
            maximum(role)
            for role in ("training_metrics_csv", "training_metrics_json", "run_log")
        )
        * training_jobs
        * epoch_cap
    )
    overlay_bytes = sum(
        path.stat().st_size
        for path in (benchmark_root / "inputs").rglob("*")
        if path.is_file()
    )
    # Reserve a second input bank for test-only overlays/panels and 12 GiB for
    # 400 MC64 prediction cells, paired-bootstrap outputs, reports and journals.
    evaluation_and_reports = 12 * 1024**3
    breakdown = {
        "dynamic_checkpoints": dynamic_checkpoints,
        "branch_checkpoints": branch_checkpoints,
        "selected_full_training_states": full_states,
        "best_checkpoint_metadata": best_checkpoint_metadata,
        "per_job_configs_and_contracts": per_job_contracts,
        "epoch_scaled_metrics_and_logs": epoch_scaled_diagnostics,
        "development_and_test_inputs": overlay_bytes * 2,
        "evaluation_predictions_reports_allowance": evaluation_and_reports,
    }
    breakdown["total"] = sum(breakdown.values())
    return breakdown


def _project_formal_storage_bytes(benchmark_root: Path) -> int:
    return _project_formal_storage_breakdown(benchmark_root)["total"]


def _benchmark_capacity_gate_path(root: Path) -> Path:
    return root / "benchmark_capacity_gate.json"


def _record_benchmark_capacity_gate(
    root: Path,
    config: Mapping[str, Any],
    *,
    workers_per_gpu: int,
    reason: str,
) -> Path:
    payload = {
        "schema_version": 1,
        "kind": "rq123_benchmark_capacity_gate_v1",
        "benchmark_root": str(root.resolve()),
        "source_config_sha256": str(config["source_config_sha256"]),
        "workers_per_gpu": int(workers_per_gpu),
        "reason": str(reason),
        "created_at_utc": utc_now(),
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return write_json(_benchmark_capacity_gate_path(root), payload)


def _read_benchmark_capacity_gate(
    root: Path, config: Mapping[str, Any], *, workers_per_gpu: int
) -> dict[str, Any] | None:
    path = _benchmark_capacity_gate_path(root)
    if not path.is_file():
        return None
    payload = read_json(path)
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if (
        payload.get("kind") != "rq123_benchmark_capacity_gate_v1"
        or payload_sha256(unsigned) != payload.get("payload_sha256")
        or payload.get("benchmark_root") != str(root.resolve())
        or payload.get("source_config_sha256") != config["source_config_sha256"]
        or int(payload.get("workers_per_gpu", -1)) != int(workers_per_gpu)
    ):
        raise ValueError("Benchmark capacity-gate journal drift")
    return payload


def _run_one_benchmark_candidate(
    config_path: str | Path,
    benchmark_root: Path,
    *,
    workers_per_gpu: int,
    resume: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    _prepare_benchmark_root(
        config_path,
        benchmark_root,
        workers_per_gpu=workers_per_gpu,
        resume=resume,
    )
    previous_gate = _read_benchmark_capacity_gate(
        benchmark_root, config, workers_per_gpu=workers_per_gpu
    )
    if previous_gate is not None:
        raise BenchmarkConcurrencyGateError(str(previous_gate["reason"]))
    started = time.monotonic()
    try:
        peak_host = _launch_stage(
            benchmark_root, BENCHMARK_STAGE, dry_run=False, resume=resume
        )
    except BaseException as exc:
        if _benchmark_failure_is_capacity_related(benchmark_root, exc):
            reason = f"Benchmark capacity/OOM gate failed: {exc}"
            _record_benchmark_capacity_gate(
                benchmark_root,
                config,
                workers_per_gpu=workers_per_gpu,
                reason=reason,
            )
            raise BenchmarkConcurrencyGateError(reason) from exc
        raise
    elapsed = time.monotonic() - started
    _assert_benchmark_updates(benchmark_root)
    resources = _resource_peaks(benchmark_root)
    maximum_gpu = max(resources["peak_memory_gib_by_gpu"].values())
    runtime = _mapping(config["runtime"], "runtime")
    if maximum_gpu >= float(runtime["preflight_max_peak_gpu_memory_gib"]):
        reason = f"Benchmark GPU memory gate failed: {maximum_gpu:.3f} GiB"
        _record_benchmark_capacity_gate(
            benchmark_root,
            config,
            workers_per_gpu=workers_per_gpu,
            reason=reason,
        )
        raise BenchmarkConcurrencyGateError(reason)
    if peak_host >= float(runtime["preflight_max_host_ram_fraction"]):
        reason = f"Benchmark host RAM gate failed: {peak_host:.4f}"
        _record_benchmark_capacity_gate(
            benchmark_root,
            config,
            workers_per_gpu=workers_per_gpu,
            reason=reason,
        )
        raise BenchmarkConcurrencyGateError(reason)
    projection = _project_formal_storage_breakdown(benchmark_root)
    projected = projection["total"]
    safety_factor = float(runtime["disk_projection_safety_factor"])
    required = int(math.ceil(projected * safety_factor))
    free = shutil.disk_usage(REPO_ROOT).free
    remaining = free - required
    minimum_remaining = (
        int(runtime["minimum_free_disk_after_projected_bytes_gib"]) * 1024**3
    )
    if remaining < minimum_remaining:
        raise RuntimeError(
            "Projected formal storage fails disk gate: "
            f"free={free}, projected_with_margin={required}, remaining={remaining}"
        )
    return {
        "status": "passed",
        "benchmark_root": str(benchmark_root.resolve()),
        "benchmark_root_code_manifest_sha256": sha256_file(
            benchmark_root / "code_hashes.csv"
        ),
        "benchmark_root_source_manifest_sha256": sha256_file(
            benchmark_root / "source_hashes.csv"
        ),
        "benchmark_root_config_manifest_sha256": sha256_file(
            benchmark_root / "config_hashes.csv"
        ),
        "config_sha256": config["source_config_sha256"],
        "job_count": 36,
        "benchmark_completion_sha256": _benchmark_completion_sha256(benchmark_root),
        "selected_workers_per_gpu": int(workers_per_gpu),
        "peak_host_ram_fraction": float(peak_host),
        "elapsed_seconds": float(elapsed),
        "projected_formal_bytes": int(projected),
        "projected_formal_storage_breakdown": projection,
        "projected_with_safety_margin_bytes": int(required),
        "free_bytes_at_gate": int(free),
        "projected_free_bytes": int(remaining),
        **resources,
    }


def _benchmark_failure_is_capacity_related(root: Path, exc: BaseException) -> bool:
    if isinstance(exc, MemoryError) or "outofmemory" in type(exc).__name__.lower():
        return True
    indicators = (
        "cuda out of memory",
        "cuda error: out of memory",
        "out of memory",
        "cublas_status_alloc_failed",
        "hip out of memory",
    )
    messages = [f"{type(exc).__name__}: {exc}".lower()]
    status_dir = root / "registry/job_status"
    if status_dir.is_dir():
        for path in status_dir.glob("*.json"):
            try:
                status = read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if status.get("status") == "failed":
                messages.append(str(status.get("error", "")).lower())
    logs_dir = root / "logs"
    if logs_dir.is_dir():
        for path in logs_dir.glob("*.log"):
            try:
                with path.open("rb") as handle:
                    handle.seek(max(0, path.stat().st_size - 128 * 1024))
                    messages.append(
                        handle.read().decode("utf-8", errors="replace").lower()
                    )
            except OSError:
                continue
    return any(indicator in message for message in messages for indicator in indicators)


def run_benchmark(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    formal_root = _validated_formal_root(config, output_dir)
    if formal_root.exists():
        raise RuntimeError("Formal root must not exist before benchmark")
    result_path = _benchmark_result_path(formal_root)
    if result_path.is_file():
        if not resume:
            raise FileExistsError(result_path)
        _require_benchmark(formal_root, config)
        return result_path
    control = result_path.parent
    control.mkdir(parents=True, exist_ok=True)
    candidates = (18, 12)
    failures: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    for workers in candidates:
        suffix = "benchmark_18" if workers == 18 else "benchmark_12"
        benchmark_root = formal_root.with_name(formal_root.name + f"_{suffix}")
        try:
            result = _run_one_benchmark_candidate(
                config_path,
                benchmark_root,
                workers_per_gpu=workers,
                resume=resume,
            )
            break
        except BenchmarkConcurrencyGateError as exc:
            failures.append(
                {
                    "workers_per_gpu": workers,
                    "benchmark_root": str(benchmark_root),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if workers == 12:
                raise
    if result is None:
        raise RuntimeError("No benchmark candidate passed")
    result["candidate_failures"] = failures
    result["created_at_utc"] = utc_now()
    result["payload_sha256"] = payload_sha256(result)
    write_json(result_path, result)
    _require_benchmark(formal_root, config)
    return result_path


def _write_recovery_canary_ownership(
    canary_root: Path,
    formal_root: Path,
    config: Mapping[str, Any],
    *,
    workers_per_gpu: int,
) -> Path:
    payload = {
        "schema_version": 1,
        "kind": "rq123_recovery_canary_owned_temporary_root_v1",
        "canary_root": str(canary_root.resolve()),
        "formal_root": str(formal_root.resolve()),
        "workers_per_gpu": int(workers_per_gpu),
        "source_config_sha256": str(config["source_config_sha256"]),
        "created_at_utc": utc_now(),
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return write_json(_recovery_canary_ownership_path(canary_root), payload)


def _validate_recovery_canary_ownership(
    canary_root: Path,
    formal_root: Path,
    config: Mapping[str, Any],
    *,
    workers_per_gpu: int,
) -> dict[str, Any]:
    payload = read_json(_recovery_canary_ownership_path(canary_root))
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if (
        payload.get("kind") != "rq123_recovery_canary_owned_temporary_root_v1"
        or payload_sha256(unsigned) != payload.get("payload_sha256")
        or payload.get("canary_root") != str(canary_root.resolve())
        or payload.get("formal_root") != str(formal_root.resolve())
        or int(payload.get("workers_per_gpu", -1)) != int(workers_per_gpu)
        or payload.get("source_config_sha256") != config["source_config_sha256"]
    ):
        raise ValueError("Recovery-canary temporary-root ownership drift")
    return payload


def _benchmark_canary_parent(
    benchmark_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    job = _find_job(benchmark_root, RECOVERY_CANARY_PARENT_JOB_ID)
    status = read_json(_status_path(benchmark_root, RECOVERY_CANARY_PARENT_JOB_ID))
    if not _completed_valid(job, status):
        raise ValueError("Recovery-canary benchmark parent is incomplete")
    state_artifact = _artifact(status, "full_training_state")
    state_payload = _validated_full_state_payload(job, status)
    parent = {
        "path": str(Path(state_artifact["path"]).resolve()),
        "sha256": str(state_artifact["sha256"]),
        "lineage": dict(state_payload["lineage"]),
    }
    return job, status, parent, state_payload


def _prepare_recovery_canary_root(
    config_path: str | Path,
    formal_root: Path,
    *,
    workers_per_gpu: int,
    resume: bool,
) -> Path:
    source_config = load_config(config_path)
    _validated_formal_root(source_config, formal_root)
    config = _with_selected_workers(source_config, workers_per_gpu)
    root = _recovery_canary_root(formal_root, workers_per_gpu)
    benchmark_path = _benchmark_result_path(formal_root)
    benchmark_root = formal_root.with_name(
        formal_root.name + f"_benchmark_{int(workers_per_gpu)}"
    ).resolve()
    if root.exists():
        if not resume:
            raise FileExistsError(root)
        _validate_recovery_canary_ownership(
            root,
            formal_root,
            config,
            workers_per_gpu=workers_per_gpu,
        )
        observed = validate_root(root)
        registry = read_registry(root)
        jobs = list(registry.get("jobs") or [])
        if (
            observed != config
            or len(jobs) != 1
            or jobs[0].get("stage") != RECOVERY_CANARY_STAGE
            or int(registry.get("expected_training_jobs", -1)) != 1
            or registry.get("benchmark_result_sha256") != sha256_file(benchmark_path)
        ):
            raise ValueError("Existing recovery-canary root drift")
        return root
    root.mkdir(parents=True)
    (root / "registry").mkdir(parents=True, exist_ok=True)
    _write_recovery_canary_ownership(
        root,
        formal_root,
        config,
        workers_per_gpu=workers_per_gpu,
    )
    _write_prepare_contract(
        root,
        config,
        workers_per_gpu=workers_per_gpu,
        mode=RECOVERY_CANARY_STAGE,
        status="preparing",
        benchmark_result_path=benchmark_path,
    )
    for relative in (
        "registry/job_status",
        "configs/jobs",
        "configs/full_state_contracts",
        "inputs/pair_text_overlays",
        "logs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    write_yaml(root / "resolved_config.yaml", config)
    write_json(root / "grid_contract.json", grid_contract(config))
    write_json(root / "model_contract.json", model_contract(config))
    write_json(
        _runtime_contract_path(root),
        _runtime_contract(
            config,
            workers_per_gpu=workers_per_gpu,
            mode="parent_continuation_checkpoint_canary",
            benchmark_result_path=benchmark_path,
        ),
    )
    materialize_pair_universes(config, root)
    materialize_pair_text_overlays(config, root)
    parent_job, parent_status, parent_state, parent_payload = _benchmark_canary_parent(
        benchmark_root
    )
    spec = {
        "stage": RECOVERY_CANARY_STAGE,
        "tolerance_minutes": 30,
        "fold": "f4_2023q4",
        "seed": 202,
        "arm": CONTINUATION_ARM,
        "gpu_id": int(parent_job["gpu_id"]),
        "gpu_slot": 0,
        "wave": 1,
    }
    job, run_config = build_job(
        config,
        root,
        spec,
        slots_per_gpu=workers_per_gpu,
        parent_state=parent_state,
        num_epochs_override=1,
        parent_job_id_override=RECOVERY_CANARY_PARENT_JOB_ID,
    )
    contract = read_json(job["full_state_contract_path"])
    if (
        run_config.get("news_first_full_training_state_mode") != "resume_dynamic_v1"
        or int(run_config.get("num_epochs", -1)) != 1
        or bool(run_config.get("news_first_materialize_test_loader"))
        or contract.get("mode") != "resume_dynamic_v1"
        or contract.get("input", {}).get("path") != parent_state["path"]
        or contract.get("input", {}).get("sha256") != parent_state["sha256"]
        or contract.get("input", {}).get("expected_lineage")
        != parent_payload["lineage"]
    ):
        raise ValueError("Recovery-canary continuation contract drift")
    write_json(_status_path(root, job["job_id"]), initial_job_status(job))
    parent_allowlist = write_csv(
        _parent_state_allowlist_path(root),
        [
            {
                "job_id": RECOVERY_CANARY_PARENT_JOB_ID,
                "job_spec_sha256": parent_job["job_spec_sha256"],
                "status_sha256": sha256_file(
                    _status_path(benchmark_root, RECOVERY_CANARY_PARENT_JOB_ID)
                ),
                "full_training_state_path": parent_state["path"],
                "full_training_state_sha256": parent_state["sha256"],
                "lineage_sha256": payload_sha256(parent_state["lineage"]),
                "completed_epoch": int(parent_payload["completed_epoch"]),
            }
        ],
    )
    write_registry(
        root,
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "interpretation": INTERPRETATION,
            "status": "recovery_canary_prepared",
            "created_at_utc": utc_now(),
            "formal_workers_per_gpu": int(workers_per_gpu),
            "expected_training_jobs": 1,
            "expected_prediction_cells": 0,
            "jobs": [job],
            "parent_states_frozen": True,
            "parent_state_allowlist_path": str(parent_allowlist.resolve()),
            "parent_state_allowlist_sha256": sha256_file(parent_allowlist),
            "branch_recipes_frozen": False,
            "evaluation_frozen": False,
            "predictions_frozen": False,
            "test_data_opened": False,
            "terminal_complete": False,
            "benchmark_result_path": str(benchmark_path.resolve()),
            "benchmark_result_sha256": sha256_file(benchmark_path),
            "benchmark_parent_job_id": RECOVERY_CANARY_PARENT_JOB_ID,
            "benchmark_parent_state_sha256": parent_state["sha256"],
        },
    )
    write_experiment_status(root, "recovery_canary_prepared", registered_jobs=1)
    _prepare_hash_manifests(config, root)
    _write_prepare_contract(
        root,
        config,
        workers_per_gpu=workers_per_gpu,
        mode=RECOVERY_CANARY_STAGE,
        status="prepared",
        benchmark_result_path=benchmark_path,
    )
    validate_root(root)
    return root


def _checkpoint_load_canary_evidence(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    *,
    physical_gpu_id: int,
) -> dict[str, Any]:
    """Load an explicit-grid checkpoint through the production inference path."""

    from types import SimpleNamespace

    import numpy as np
    import torch

    from wgan_option.utils.inference_helpers import load_vol_generator

    architecture_sha = architecture_profile_contract(config)[
        "architecture_profile_sha256"
    ]
    model_sha = model_contract(config)["model_contract_sha256"]
    grid = grid_contract(config)
    gpu_id = int(physical_gpu_id)
    if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"Recovery canary requires visible physical CUDA GPU {gpu_id}"
        )
    device = torch.device(f"cuda:{gpu_id}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sample = SimpleNamespace(
        current_surface=np.zeros((1, 16, 16), dtype=np.float32),
        strike_grid=np.asarray(grid["strike_grid"], dtype=np.float32),
        maturity_grid_days=np.asarray(grid["maturity_days_grid"], dtype=np.float32),
    )
    generator, training_config, embedding_dim = load_vol_generator(
        checkpoint_path, sample, device
    )
    parameter_count = sum(parameter.numel() for parameter in generator.parameters())
    current = torch.full((1, 1, 16, 16), 0.2, dtype=torch.float32, device=device)
    text_embedding = torch.zeros(
        (1, int(embedding_dim)), dtype=torch.float32, device=device
    )
    noise = torch.zeros(
        (1, int(training_config.noise_dim)), dtype=torch.float32, device=device
    )
    current_support = torch.ones_like(current)
    with torch.no_grad():
        output = generator(
            current,
            text_embedding,
            noise=noise,
            current_support_mask=current_support,
        )
    if (
        str(checkpoint.get("architecture_profile_sha256", "")) != architecture_sha
        or str(checkpoint.get("model_contract_sha256", "")) != model_sha
        or str(checkpoint.get("surface_grid_sha256", ""))
        != grid["grid_contract_sha256"]
        or str(training_config.news_first_architecture_profile_sha256)
        != architecture_sha
        or str(training_config.news_first_model_contract_sha256) != model_sha
        or parameter_count != EXPECTED_PARAMETER_COUNTS["generator"]
        or tuple(output.shape) != (1, 1, 16, 16)
        or not bool(torch.isfinite(output).all())
    ):
        raise ValueError("Production Generator checkpoint-load canary failed")
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "architecture_profile_sha256": architecture_sha,
        "model_contract_sha256": model_sha,
        "grid_contract_sha256": grid["grid_contract_sha256"],
        "generator_parameter_count": parameter_count,
        "embedding_dim": int(embedding_dim),
        "physical_gpu_id": gpu_id,
        "cuda_device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        "production_loader_passed": True,
        "finite_forward_passed": True,
    }


def _recovery_canary_evidence_row(
    formal_root: Path,
    canary_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    benchmark = _require_benchmark(formal_root, config)
    benchmark_root = Path(str(benchmark["benchmark_root"])).resolve()
    parent_job, parent_status, parent_state, parent_payload = _benchmark_canary_parent(
        benchmark_root
    )
    parent_checkpoint = _artifact(parent_status, "generator_best_learned")
    parent_discriminator = _artifact(parent_status, "discriminator_best_learned")
    parent_load = _checkpoint_load_canary_evidence(
        Path(parent_checkpoint["path"]),
        config,
        physical_gpu_id=int(parent_job["gpu_id"]),
    )
    updates = _assert_stage_one_epoch_updates(canary_root, RECOVERY_CANARY_STAGE)
    if len(updates) != 1:
        raise ValueError("Recovery canary must contain one update row")
    job = _stage_jobs(canary_root, RECOVERY_CANARY_STAGE)[0]
    status = read_json(_status_path(canary_root, str(job["job_id"])))
    continuation_payload = _validated_full_state_payload(job, status)
    continuation_state = _artifact(status, "full_training_state")
    continuation_log = _artifact(status, "run_log")
    generator = _artifact(status, "generator_best_learned")
    discriminator = _artifact(status, "discriminator_best_learned")
    continuation_initial_generator = _artifact(status, "generator_initial_epoch0")
    continuation_initial_discriminator = _artifact(
        status, "discriminator_initial_epoch0"
    )
    generator_restore_exact = _state_dicts_equal(
        Path(parent_checkpoint["path"]),
        Path(continuation_initial_generator["path"]),
    )
    critic_restore_exact = _state_dicts_equal(
        Path(parent_discriminator["path"]),
        Path(continuation_initial_discriminator["path"]),
    )
    continuation_load = _checkpoint_load_canary_evidence(
        Path(generator["path"]),
        config,
        physical_gpu_id=int(job["gpu_id"]),
    )
    contract = read_json(job["full_state_contract_path"])
    run_config = yaml.safe_load(
        Path(job["training_config_path"]).read_text(encoding="utf-8")
    )
    if (
        not isinstance(run_config, Mapping)
        or contract.get("input", {}).get("path") != parent_state["path"]
        or contract.get("input", {}).get("sha256") != parent_state["sha256"]
        or contract.get("input", {}).get("expected_lineage")
        != parent_payload["lineage"]
        or run_config.get("news_first_full_training_state_mode") != "resume_dynamic_v1"
        or bool(run_config.get("news_first_materialize_test_loader"))
        or int(run_config.get("num_epochs", -1)) != 1
        or int(continuation_payload.get("completed_epoch", -1)) != 1
        or not generator_restore_exact
        or not critic_restore_exact
        or parent_load["physical_gpu_id"] != continuation_load["physical_gpu_id"]
    ):
        raise ValueError("Recovery-canary restore/output lineage drift")
    update = updates[0]
    return {
        "parent_job_id": str(parent_job["job_id"]),
        "parent_job_spec_sha256": str(parent_job["job_spec_sha256"]),
        "parent_state_path": parent_state["path"],
        "parent_state_size_bytes": int(Path(parent_state["path"]).stat().st_size),
        "parent_state_sha256": parent_state["sha256"],
        "parent_state_lineage_sha256": payload_sha256(parent_payload["lineage"]),
        "parent_checkpoint_path": parent_load["checkpoint_path"],
        "parent_checkpoint_size_bytes": parent_load["checkpoint_size_bytes"],
        "parent_checkpoint_sha256": parent_load["checkpoint_sha256"],
        "parent_checkpoint_loader_passed": True,
        "parent_discriminator_checkpoint_path": str(parent_discriminator["path"]),
        "parent_discriminator_checkpoint_size_bytes": int(
            parent_discriminator["size_bytes"]
        ),
        "parent_discriminator_checkpoint_sha256": str(parent_discriminator["sha256"]),
        "continuation_job_id": str(job["job_id"]),
        "continuation_job_spec_sha256": str(job["job_spec_sha256"]),
        "continuation_training_config_sha256": str(job["training_config_sha256"]),
        "continuation_full_state_contract_sha256": str(
            job["full_state_contract_sha256"]
        ),
        "continuation_state_path": str(continuation_state["path"]),
        "continuation_state_size_bytes": int(continuation_state["size_bytes"]),
        "continuation_state_sha256": str(continuation_state["sha256"]),
        "generator_checkpoint_path": continuation_load["checkpoint_path"],
        "generator_checkpoint_size_bytes": continuation_load["checkpoint_size_bytes"],
        "generator_checkpoint_sha256": continuation_load["checkpoint_sha256"],
        "discriminator_checkpoint_path": str(discriminator["path"]),
        "discriminator_checkpoint_size_bytes": int(discriminator["size_bytes"]),
        "discriminator_checkpoint_sha256": str(discriminator["sha256"]),
        "continuation_initial_generator_path": str(
            continuation_initial_generator["path"]
        ),
        "continuation_initial_generator_size_bytes": int(
            continuation_initial_generator["size_bytes"]
        ),
        "continuation_initial_generator_sha256": str(
            continuation_initial_generator["sha256"]
        ),
        "continuation_initial_discriminator_path": str(
            continuation_initial_discriminator["path"]
        ),
        "continuation_initial_discriminator_size_bytes": int(
            continuation_initial_discriminator["size_bytes"]
        ),
        "continuation_initial_discriminator_sha256": str(
            continuation_initial_discriminator["sha256"]
        ),
        "architecture_profile_sha256": continuation_load["architecture_profile_sha256"],
        "model_contract_sha256": continuation_load["model_contract_sha256"],
        "grid_contract_sha256": continuation_load["grid_contract_sha256"],
        "generator_parameter_count": continuation_load["generator_parameter_count"],
        "physical_gpu_id": continuation_load["physical_gpu_id"],
        "cuda_device": continuation_load["cuda_device"],
        "cuda_device_name": continuation_load["cuda_device_name"],
        "cuda_visible_devices": continuation_load["cuda_visible_devices"],
        "worker_physical_gpu_id": int(job["gpu_id"]),
        "worker_cuda_visible_devices": str(job["gpu_id"]),
        "worker_run_log_path": str(continuation_log["path"]),
        "worker_run_log_size_bytes": int(continuation_log["size_bytes"]),
        "worker_run_log_sha256": str(continuation_log["sha256"]),
        "metrics_sha256": str(update["metrics_sha256"]),
        "metrics_row_count": int(update["row_count"]),
        "maximum_epoch": int(update["maximum_epoch"]),
        "generator_restore_exact": generator_restore_exact,
        "critic_restore_exact": critic_restore_exact,
        "restored_parent_state_exact": (
            generator_restore_exact and critic_restore_exact
        ),
        "generator_updated": bool(update["generator_updated"]),
        "critic_updated": bool(update["critic_updated"]),
        "metrics_finite": True,
        "one_epoch": True,
        "test_loader_disabled": True,
        "production_parent_checkpoint_loader_passed": True,
        "production_continuation_checkpoint_loader_passed": True,
        "finite_forward_passed": True,
    }


def _read_recovery_canary_evidence(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != 1:
        raise ValueError("Recovery-canary evidence must contain exactly one row")
    row = rows[0]
    truth_fields = (
        "generator_restore_exact",
        "critic_restore_exact",
        "restored_parent_state_exact",
        "generator_updated",
        "critic_updated",
        "metrics_finite",
        "one_epoch",
        "test_loader_disabled",
        "production_parent_checkpoint_loader_passed",
        "production_continuation_checkpoint_loader_passed",
        "finite_forward_passed",
    )
    if any(row.get(field) != "True" for field in truth_fields):
        raise ValueError("Recovery-canary evidence contains a failed acceptance flag")
    if (
        row.get("parent_job_id") != RECOVERY_CANARY_PARENT_JOB_ID
        or int(row.get("maximum_epoch", -1)) != 1
        or int(row.get("generator_parameter_count", -1))
        != EXPECTED_PARAMETER_COUNTS["generator"]
        or row.get("architecture_profile_sha256")
        != EXPECTED_ARCHITECTURE_PROFILE_SHA256
        or int(row.get("physical_gpu_id", -1)) != 1
        or row.get("cuda_device") != "cuda:1"
        or int(row.get("worker_physical_gpu_id", -1)) != 1
        or row.get("worker_cuda_visible_devices") != "1"
    ):
        raise ValueError("Recovery-canary evidence contract drift")
    sha_fields = (
        "parent_state_sha256",
        "parent_checkpoint_sha256",
        "parent_discriminator_checkpoint_sha256",
        "continuation_initial_generator_sha256",
        "continuation_initial_discriminator_sha256",
        "generator_checkpoint_sha256",
        "discriminator_checkpoint_sha256",
        "continuation_state_sha256",
        "worker_run_log_sha256",
    )
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(row.get(field, ""))) is None
        for field in sha_fields
    ):
        raise ValueError("Recovery-canary evidence contains an invalid SHA")
    return row


def _require_recovery_canary(
    formal_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    benchmark = _require_benchmark(formal_root, config)
    result_path = _recovery_canary_result_path(formal_root)
    evidence_path = _recovery_canary_evidence_path(formal_root)
    result = read_json(result_path)
    unsigned = {key: value for key, value in result.items() if key != "payload_sha256"}
    evidence = _read_recovery_canary_evidence(evidence_path)
    workers = int(benchmark["selected_workers_per_gpu"])
    expected_root = _recovery_canary_root(formal_root, workers)
    architecture_sha = architecture_profile_contract(config)[
        "architecture_profile_sha256"
    ]
    if (
        result.get("kind") != RECOVERY_CANARY_RESULT_KIND
        or result.get("status") != "passed"
        or payload_sha256(unsigned) != result.get("payload_sha256")
        or result.get("source_config_sha256") != config["source_config_sha256"]
        or int(result.get("selected_workers_per_gpu", -1)) != workers
        or Path(str(result.get("temporary_root", ""))).resolve() != expected_root
        or result.get("benchmark_result_sha256")
        != sha256_file(_benchmark_result_path(formal_root))
        or result.get("benchmark_completion_sha256")
        != benchmark["benchmark_completion_sha256"]
        or result.get("code_lineage_sha256") != _lineage_digest(config, code=True)
        or result.get("source_lineage_sha256") != _lineage_digest(config, code=False)
        or result.get("architecture_profile_sha256") != architecture_sha
        or result.get("model_contract_sha256")
        != model_contract(config)["model_contract_sha256"]
        or result.get("grid_contract_sha256")
        != grid_contract(config)["grid_contract_sha256"]
        or result.get("evidence_path") != str(evidence_path.resolve())
        or int(result.get("evidence_size_bytes", -1)) != evidence_path.stat().st_size
        or result.get("evidence_sha256") != sha256_file(evidence_path)
        or int(result.get("evidence_row_count", -1)) != 1
        or result.get("parent_state_sha256") != evidence["parent_state_sha256"]
        or result.get("generator_checkpoint_sha256")
        != evidence["generator_checkpoint_sha256"]
        or result.get("discriminator_checkpoint_sha256")
        != evidence["discriminator_checkpoint_sha256"]
    ):
        raise ValueError("Recovery-canary result/config/evidence contract drift")
    benchmark_root = Path(str(benchmark["benchmark_root"])).resolve()
    _, parent_status, _, _ = _benchmark_canary_parent(benchmark_root)
    if (
        _artifact(parent_status, "full_training_state")["sha256"]
        != evidence["parent_state_sha256"]
        or _artifact(parent_status, "generator_best_learned")["sha256"]
        != evidence["parent_checkpoint_sha256"]
    ):
        raise ValueError("Recovery-canary benchmark-parent artifact drift")
    return result


def _cleanup_recovery_canary_root(
    formal_root: Path,
    config: Mapping[str, Any],
    *,
    workers_per_gpu: int,
) -> None:
    root = _recovery_canary_root(formal_root, workers_per_gpu)
    expected = formal_root.with_name(
        formal_root.name + f"_restore_canary_gpu_{int(workers_per_gpu)}"
    ).resolve()
    if root != expected or root.parent != formal_root.parent.resolve():
        raise RuntimeError(f"Refusing unsafe recovery-canary cleanup target: {root}")
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Refusing non-directory recovery-canary root: {root}")
    _validate_recovery_canary_ownership(
        root,
        formal_root,
        config,
        workers_per_gpu=workers_per_gpu,
    )
    prepare = _validate_prepare_marker_integrity(root)
    registry = read_registry(root)
    jobs = list(registry.get("jobs") or [])
    if (
        prepare.get("mode") != RECOVERY_CANARY_STAGE
        or prepare.get("root") != str(root)
        or len(jobs) != 1
        or jobs[0].get("stage") != RECOVERY_CANARY_STAGE
        or int(registry.get("expected_training_jobs", -1)) != 1
        or not _completed_valid(
            jobs[0], read_json(_status_path(root, str(jobs[0]["job_id"])))
        )
    ):
        raise RuntimeError(f"Refusing unowned recovery-canary cleanup target: {root}")
    shutil.rmtree(root)


def run_recovery_canary(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    formal_root = _validated_formal_root(config, output_dir)
    benchmark = _require_benchmark(formal_root, config)
    workers = int(benchmark["selected_workers_per_gpu"])
    result_path = _recovery_canary_result_path(formal_root)
    if result_path.is_file():
        if not resume:
            raise FileExistsError(result_path)
        _require_recovery_canary(formal_root, config)
        _cleanup_recovery_canary_root(formal_root, config, workers_per_gpu=workers)
        _require_recovery_canary(formal_root, config)
        return result_path
    if formal_root.exists():
        raise RuntimeError("Formal root must not exist before the recovery canary")
    canary_root = _prepare_recovery_canary_root(
        config_path,
        formal_root,
        workers_per_gpu=workers,
        resume=resume,
    )
    started = time.monotonic()
    peak_host = _launch_stage(
        canary_root, RECOVERY_CANARY_STAGE, dry_run=False, resume=resume
    )
    elapsed = time.monotonic() - started
    evidence = _recovery_canary_evidence_row(formal_root, canary_root, config)
    evidence_path = write_csv(_recovery_canary_evidence_path(formal_root), [evidence])
    persisted = _read_recovery_canary_evidence(evidence_path)
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": RECOVERY_CANARY_RESULT_KIND,
        "status": "passed",
        "source_config_sha256": config["source_config_sha256"],
        "selected_workers_per_gpu": workers,
        "temporary_root": str(canary_root),
        "temporary_root_cleanup_authorized_after_evidence": True,
        "benchmark_result_path": str(_benchmark_result_path(formal_root).resolve()),
        "benchmark_result_sha256": sha256_file(_benchmark_result_path(formal_root)),
        "benchmark_completion_sha256": benchmark["benchmark_completion_sha256"],
        "code_lineage_sha256": _lineage_digest(config, code=True),
        "source_lineage_sha256": _lineage_digest(config, code=False),
        "architecture_profile_sha256": architecture_profile_contract(config)[
            "architecture_profile_sha256"
        ],
        "model_contract_sha256": model_contract(config)["model_contract_sha256"],
        "grid_contract_sha256": grid_contract(config)["grid_contract_sha256"],
        "evidence_path": str(evidence_path.resolve()),
        "evidence_size_bytes": evidence_path.stat().st_size,
        "evidence_sha256": sha256_file(evidence_path),
        "evidence_row_count": 1,
        "parent_state_sha256": persisted["parent_state_sha256"],
        "generator_checkpoint_sha256": persisted["generator_checkpoint_sha256"],
        "discriminator_checkpoint_sha256": persisted["discriminator_checkpoint_sha256"],
        "peak_host_ram_fraction": float(peak_host),
        "elapsed_seconds": float(elapsed),
        "created_at_utc": utc_now(),
    }
    result["payload_sha256"] = payload_sha256(result)
    write_json(result_path, result)
    _require_recovery_canary(formal_root, config)
    _cleanup_recovery_canary_root(formal_root, config, workers_per_gpu=workers)
    if canary_root.exists():
        raise RuntimeError("Recovery-canary temporary root cleanup did not complete")
    _require_recovery_canary(formal_root, config)
    return result_path


def _lineage_digest(config: Mapping[str, Any], *, code: bool) -> str:
    rows: list[dict[str, Any]]
    if code:
        rows = [
            manifest_row(f"code:{path.relative_to(REPO_ROOT)}", path)
            for path in _code_paths()
        ]
    else:
        rows = [manifest_row(role, path) for role, path in _source_paths(config)]
    return payload_sha256(
        [
            {
                "artifact_role": str(row["artifact_role"]),
                "path": str(Path(str(row["path"])).resolve()),
                "size_bytes": int(row["size_bytes"]),
                "sha256": str(row["sha256"]),
            }
            for row in rows
        ]
    )


def _formal_cell_digest() -> str:
    return payload_sha256(
        sorted(
            (
                str(row["stage"]),
                int(row["tolerance_minutes"]),
                str(row["fold"]),
                int(row["seed"]),
                str(row["arm"]),
                int(row["gpu_id"]),
            )
            for row in planned_specs()
        )
    )


def _project_matrix_smoke_storage_bytes(benchmark_root: Path) -> int:
    """Conservatively project the temporary peak from observed benchmark files."""

    sizes: dict[str, list[int]] = {}
    for job in _validate_stage_complete(benchmark_root, BENCHMARK_STAGE):
        status = read_json(_status_path(benchmark_root, str(job["job_id"])))
        for row in status["artifacts"]:
            role = str(row["artifact_role"])
            if role == "pair_text_overlay_manifest":
                continue
            sizes.setdefault(role, []).append(int(row["size_bytes"]))
    required_roles = {
        "generator_initial_epoch0",
        "discriminator_initial_epoch0",
        "generator_best_learned",
        "discriminator_best_learned",
        "generator_final",
        "discriminator_final",
        "training_metrics_csv",
        "training_metrics_json",
        "best_learned_checkpoint",
        "resolved_training_config",
        "run_log",
        "full_state_contract",
        "full_training_state",
    }
    if not required_roles.issubset(sizes):
        raise ValueError(
            "Benchmark cannot project matrix-smoke storage; missing roles="
            f"{sorted(required_roles - set(sizes))}"
        )
    per_job = sum(max(sizes[role]) for role in required_roles)
    inputs = sum(
        path.stat().st_size
        for path in (benchmark_root / "inputs").rglob("*")
        if path.is_file()
    )
    journals_and_configs = 512 * 1024 * EXPECTED_MATRIX_SMOKE_JOBS
    return int(per_job * EXPECTED_MATRIX_SMOKE_JOBS + inputs + journals_and_configs)


def _matrix_smoke_storage_gate(
    config: Mapping[str, Any], benchmark_root: Path
) -> dict[str, int | float]:
    runtime = _mapping(config["runtime"], "runtime")
    projected = _project_matrix_smoke_storage_bytes(benchmark_root)
    safety_factor = float(runtime["disk_projection_safety_factor"])
    required = int(math.ceil(projected * safety_factor))
    free = int(shutil.disk_usage(REPO_ROOT).free)
    minimum_remaining = (
        int(runtime["minimum_free_disk_after_projected_bytes_gib"]) * 1024**3
    )
    remaining = free - required
    if remaining < minimum_remaining:
        raise RuntimeError(
            "Projected 400-cell one-epoch smoke storage fails disk gate: "
            f"free={free}, projected_with_margin={required}, remaining={remaining}"
        )
    return {
        "projected_bytes": projected,
        "safety_factor": safety_factor,
        "projected_with_margin_bytes": required,
        "free_bytes_at_gate": free,
        "projected_free_bytes": remaining,
    }


def _freeze_matrix_smoke_storage_gate(
    formal_root: Path,
    config: Mapping[str, Any],
    benchmark_root: Path,
) -> dict[str, Any]:
    path = _matrix_smoke_storage_gate_path(formal_root)
    if path.is_file():
        payload = read_json(path)
        unsigned = {
            key: value for key, value in payload.items() if key != "payload_sha256"
        }
        if (
            payload.get("kind") != "rq123_matrix_smoke_storage_gate_v1"
            or payload_sha256(unsigned) != payload.get("payload_sha256")
            or payload.get("source_config_sha256") != config["source_config_sha256"]
            or payload.get("benchmark_result_sha256")
            != sha256_file(_benchmark_result_path(formal_root))
        ):
            raise ValueError("Frozen matrix-smoke storage gate drift")
        return dict(payload["storage_gate"])
    if _matrix_smoke_root(
        formal_root,
        int(read_json(_benchmark_result_path(formal_root))["selected_workers_per_gpu"]),
    ).exists():
        raise ValueError("Matrix-smoke root exists without a frozen storage gate")
    gate = _matrix_smoke_storage_gate(config, benchmark_root)
    payload = {
        "schema_version": 1,
        "kind": "rq123_matrix_smoke_storage_gate_v1",
        "source_config_sha256": config["source_config_sha256"],
        "benchmark_result_sha256": sha256_file(_benchmark_result_path(formal_root)),
        "storage_gate": gate,
        "created_at_utc": utc_now(),
    }
    payload["payload_sha256"] = payload_sha256(payload)
    write_json(path, payload)
    return gate


def _prepare_matrix_smoke_root(
    config_path: str | Path,
    formal_root: Path,
    *,
    workers_per_gpu: int,
    storage_gate: Mapping[str, Any] | None,
    resume: bool,
) -> Path:
    source_config = load_config(config_path)
    config = _with_selected_workers(source_config, workers_per_gpu)
    root = _matrix_smoke_root(formal_root, workers_per_gpu)
    benchmark_path = _benchmark_result_path(formal_root)
    if root.exists():
        if not resume:
            raise FileExistsError(root)
        try:
            existing = validate_root(root)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            marker = _validate_prepare_contract(
                root,
                config,
                workers_per_gpu=workers_per_gpu,
                mode=MATRIX_SMOKE_STAGE,
                benchmark_result_path=benchmark_path,
            )
            if marker.get("status") != "preparing":
                raise ValueError("Invalid matrix-smoke root is not safely resumable")
            registry_path = _registry_path(root)
            if registry_path.is_file():
                partial = read_registry(root)
                for job in partial.get("jobs") or []:
                    status_path = _status_path(root, str(job["job_id"]))
                    if not status_path.is_file():
                        continue
                    status = read_json(status_path)
                    if (
                        status.get("status") != "pending"
                        or int(status.get("attempt", 0)) != 0
                        or status.get("artifacts")
                    ):
                        raise ValueError(
                            "Matrix-smoke preparation cannot resume after training started"
                        )
        else:
            if existing != config:
                raise ValueError("Existing matrix-smoke resolved config drift")
            registry = read_registry(root)
            if (
                int(registry.get("formal_workers_per_gpu", -1)) != int(workers_per_gpu)
                or int(registry.get("expected_training_jobs", -1))
                != EXPECTED_MATRIX_SMOKE_JOBS
                or registry.get("benchmark_result_sha256")
                != sha256_file(benchmark_path)
            ):
                raise ValueError("Existing matrix-smoke preflight anchor drift")
            return root
    else:
        if storage_gate is None:
            raise ValueError("A new matrix-smoke root requires a frozen storage gate")
        root.mkdir(parents=True)
        (root / "registry").mkdir(parents=True, exist_ok=True)
        _write_matrix_smoke_ownership(
            root,
            formal_root,
            config,
            workers_per_gpu=workers_per_gpu,
        )
    _write_prepare_contract(
        root,
        config,
        workers_per_gpu=workers_per_gpu,
        mode=MATRIX_SMOKE_STAGE,
        status="preparing",
        benchmark_result_path=benchmark_path,
    )
    for relative in (
        "registry/job_status",
        "configs/jobs",
        "configs/full_state_contracts",
        "inputs/pair_text_overlays",
        "logs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    write_yaml(root / "resolved_config.yaml", config)
    write_json(root / "grid_contract.json", grid_contract(config))
    write_json(root / "model_contract.json", model_contract(config))
    write_json(
        _runtime_contract_path(root),
        _runtime_contract(
            config,
            workers_per_gpu=workers_per_gpu,
            mode="matrix_smoke_candidate",
            benchmark_result_path=benchmark_path,
        ),
    )
    materialize_pair_universes(config, root)
    materialize_pair_text_overlays(config, root)
    specs = assign_waves(matrix_smoke_specs(), slots_per_gpu=workers_per_gpu)
    jobs: list[dict[str, Any]] = []
    for spec in specs:
        job, run_config = build_job(
            config,
            root,
            spec,
            slots_per_gpu=workers_per_gpu,
        )
        if (
            int(run_config["num_epochs"]) != 1
            or run_config["news_first_full_training_state_mode"] != "save_dynamic_v1"
            or bool(run_config["news_first_materialize_test_loader"])
            or run_config["news_first_data_window_end_utc_exclusive"]
            != _fold(config, str(spec["fold"]))["validation_end_utc"]
        ):
            raise ValueError(f"Matrix-smoke training contract drift: {job['job_id']}")
        jobs.append(job)
        write_json(_status_path(root, job["job_id"]), initial_job_status(job))
    gate_payload = dict(storage_gate or {})
    write_registry(
        root,
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "interpretation": INTERPRETATION,
            "status": "matrix_smoke_prepared",
            "created_at_utc": utc_now(),
            "formal_workers_per_gpu": int(workers_per_gpu),
            "expected_training_jobs": EXPECTED_MATRIX_SMOKE_JOBS,
            "expected_prediction_cells": 0,
            "jobs": jobs,
            "parent_states_frozen": False,
            "branch_recipes_frozen": False,
            "evaluation_frozen": False,
            "predictions_frozen": False,
            "test_data_opened": False,
            "terminal_complete": False,
            "benchmark_result_path": str(benchmark_path.resolve()),
            "benchmark_result_sha256": sha256_file(benchmark_path),
            "matrix_smoke_storage_gate": gate_payload,
        },
    )
    write_experiment_status(root, "matrix_smoke_prepared", registered_jobs=len(jobs))
    _prepare_hash_manifests(config, root)
    _write_prepare_contract(
        root,
        config,
        workers_per_gpu=workers_per_gpu,
        mode=MATRIX_SMOKE_STAGE,
        status="prepared",
        benchmark_result_path=benchmark_path,
    )
    validate_root(root)
    return root


def _matrix_smoke_evidence_rows(root: Path) -> list[dict[str, Any]]:
    config = validate_root(root)
    registry = read_registry(root)
    update_rows = {
        row["job_id"]: row
        for row in _assert_stage_one_epoch_updates(root, MATRIX_SMOKE_STAGE)
    }
    rows: list[dict[str, Any]] = []
    for job in sorted(
        _stage_jobs(root, MATRIX_SMOKE_STAGE), key=lambda row: str(row["job_id"])
    ):
        status = read_json(_status_path(root, str(job["job_id"])))
        config_path = Path(str(job["training_config_path"])).resolve()
        contract_path = Path(str(job["full_state_contract_path"])).resolve()
        if (
            sha256_file(config_path) != job["training_config_sha256"]
            or sha256_file(contract_path) != job["full_state_contract_sha256"]
        ):
            raise ValueError(
                f"Matrix-smoke config/contract hash drift: {job['job_id']}"
            )
        run_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(run_config, Mapping):
            raise ValueError(f"Invalid smoke training config: {config_path}")
        fold = _fold(config, str(job["fold"]))
        contract = read_json(contract_path)
        no_test = (
            not bool(run_config.get("news_first_materialize_test_loader"))
            and run_config.get("news_first_data_window_end_utc_exclusive")
            == fold["validation_end_utc"]
            and not bool(registry.get("test_data_opened"))
        )
        save_dynamic = (
            run_config.get("news_first_full_training_state_mode") == "save_dynamic_v1"
            and contract.get("mode") == "save_dynamic_v1"
            and contract.get("input") is None
        )
        artifacts = sorted(
            (
                str(row["artifact_role"]),
                int(row["size_bytes"]),
                str(row["sha256"]),
            )
            for row in status["artifacts"]
        )
        updates = update_rows[str(job["job_id"])]
        if (
            not no_test
            or not save_dynamic
            or int(run_config.get("num_epochs", -1)) != 1
        ):
            raise ValueError(f"Invalid matrix-smoke evidence: {job['job_id']}")
        rows.append(
            {
                "job_id": str(job["job_id"]),
                "formal_stage": str(job["formal_stage"]),
                "tolerance_minutes": int(job["tolerance_minutes"]),
                "fold": str(job["fold"]),
                "seed": int(job["seed"]),
                "arm": str(job["arm"]),
                "gpu_id": int(job["gpu_id"]),
                "wave": int(job["wave"]),
                "job_spec_sha256": str(job["job_spec_sha256"]),
                "training_config_sha256": str(job["training_config_sha256"]),
                "dataset_sha256": str(job["dataset_sha256"]),
                "support_sha256": str(job["support_sha256"]),
                "overlay_sha256": str(job["overlay_sha256"]),
                "artifact_evidence_sha256": payload_sha256(artifacts),
                "metrics_sha256": str(updates["metrics_sha256"]),
                "metrics_row_count": int(updates["row_count"]),
                "maximum_epoch": int(updates["maximum_epoch"]),
                "generator_updated": True,
                "critic_updated": True,
                "metrics_finite": True,
                "one_epoch": True,
                "save_dynamic": True,
                "test_loader_disabled": True,
            }
        )
    if len(rows) != EXPECTED_MATRIX_SMOKE_JOBS:
        raise ValueError("Matrix-smoke evidence must contain exactly 400 rows")
    return rows


def _read_matrix_smoke_evidence(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != EXPECTED_MATRIX_SMOKE_JOBS:
        raise ValueError("Matrix-smoke evidence row count drift")
    observed = {
        (
            row["formal_stage"],
            int(row["tolerance_minutes"]),
            row["fold"],
            int(row["seed"]),
            row["arm"],
            int(row["gpu_id"]),
        )
        for row in rows
    }
    expected = {
        (
            str(row["stage"]),
            int(row["tolerance_minutes"]),
            str(row["fold"]),
            int(row["seed"]),
            str(row["arm"]),
            int(row["gpu_id"]),
        )
        for row in planned_specs()
    }
    if observed != expected or len(observed) != EXPECTED_MATRIX_SMOKE_JOBS:
        raise ValueError("Matrix-smoke evidence does not cover the exact formal matrix")
    truth_fields = (
        "generator_updated",
        "critic_updated",
        "metrics_finite",
        "one_epoch",
        "save_dynamic",
        "test_loader_disabled",
    )
    if any(row.get(field) != "True" for row in rows for field in truth_fields):
        raise ValueError("Matrix-smoke evidence contains a failed acceptance flag")
    return rows


def _require_matrix_smoke(
    formal_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    benchmark = _require_benchmark(formal_root, config)
    result_path = _matrix_smoke_result_path(formal_root)
    evidence_path = _matrix_smoke_evidence_path(formal_root)
    storage_gate_path = _matrix_smoke_storage_gate_path(formal_root)
    result = read_json(result_path)
    unsigned = {key: value for key, value in result.items() if key != "payload_sha256"}
    rows = _read_matrix_smoke_evidence(evidence_path)
    expected_root = _matrix_smoke_root(
        formal_root, int(benchmark["selected_workers_per_gpu"])
    )
    if (
        result.get("kind") != MATRIX_SMOKE_RESULT_KIND
        or result.get("status") != "passed"
        or payload_sha256(unsigned) != result.get("payload_sha256")
        or result.get("source_config_sha256") != config["source_config_sha256"]
        or int(result.get("selected_workers_per_gpu", -1))
        != int(benchmark["selected_workers_per_gpu"])
        or int(result.get("job_count", -1)) != EXPECTED_MATRIX_SMOKE_JOBS
        or Path(str(result.get("temporary_root", ""))).resolve() != expected_root
        or result.get("formal_cell_sha256") != _formal_cell_digest()
        or result.get("evidence_path") != str(evidence_path.resolve())
        or result.get("evidence_sha256") != sha256_file(evidence_path)
        or int(result.get("evidence_row_count", -1)) != len(rows)
        or result.get("storage_gate_path") != str(storage_gate_path.resolve())
        or result.get("storage_gate_sha256") != sha256_file(storage_gate_path)
        or result.get("benchmark_result_sha256")
        != sha256_file(_benchmark_result_path(formal_root))
        or result.get("code_lineage_sha256") != _lineage_digest(config, code=True)
        or result.get("source_lineage_sha256") != _lineage_digest(config, code=False)
    ):
        raise ValueError("Matrix-smoke result/config/evidence contract drift")
    return result


def _cleanup_matrix_smoke_root(formal_root: Path, workers_per_gpu: int) -> None:
    root = _matrix_smoke_root(formal_root, workers_per_gpu)
    expected = formal_root.with_name(
        formal_root.name + f"_matrix_smoke_{int(workers_per_gpu)}"
    ).resolve()
    if root != expected or root.parent != formal_root.parent.resolve():
        raise RuntimeError(f"Refusing unsafe matrix-smoke cleanup target: {root}")
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(
            f"Refusing non-directory matrix-smoke cleanup target: {root}"
        )
    ownership_path = _matrix_smoke_ownership_path(root)
    ownership = read_json(ownership_path)
    ownership_unsigned = {
        key: value for key, value in ownership.items() if key != "payload_sha256"
    }
    prepare = _validate_prepare_marker_integrity(root)
    registry = read_registry(root)
    if (
        ownership.get("kind") != "rq123_matrix_smoke_owned_temporary_root_v1"
        or payload_sha256(ownership_unsigned) != ownership.get("payload_sha256")
        or ownership.get("smoke_root") != str(root)
        or ownership.get("formal_root") != str(formal_root.resolve())
        or int(ownership.get("workers_per_gpu", -1)) != int(workers_per_gpu)
        or prepare.get("mode") != MATRIX_SMOKE_STAGE
        or prepare.get("root") != str(root)
        or int(prepare.get("selected_workers_per_gpu", -1)) != int(workers_per_gpu)
        or registry.get("experiment_kind") != EXPERIMENT_KIND
        or int(registry.get("expected_training_jobs", -1)) != EXPECTED_MATRIX_SMOKE_JOBS
        or len(registry.get("jobs") or []) != EXPECTED_MATRIX_SMOKE_JOBS
    ):
        raise RuntimeError(f"Refusing unowned matrix-smoke cleanup target: {root}")
    shutil.rmtree(root)


def run_matrix_smoke(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    formal_root = _validated_formal_root(config, output_dir)
    benchmark = _require_benchmark(formal_root, config)
    _require_recovery_canary(formal_root, config)
    workers = int(benchmark["selected_workers_per_gpu"])
    result_path = _matrix_smoke_result_path(formal_root)
    if result_path.is_file():
        if not resume:
            raise FileExistsError(result_path)
        _require_matrix_smoke(formal_root, config)
        _cleanup_matrix_smoke_root(formal_root, workers)
        return result_path
    if formal_root.exists():
        raise RuntimeError("Formal root must not exist before the full-matrix smoke")
    smoke_root = _matrix_smoke_root(formal_root, workers)
    storage_gate = _freeze_matrix_smoke_storage_gate(
        formal_root,
        config,
        Path(str(benchmark["benchmark_root"])).resolve(),
    )
    _prepare_matrix_smoke_root(
        config_path,
        formal_root,
        workers_per_gpu=workers,
        storage_gate=storage_gate,
        resume=resume,
    )
    started = time.monotonic()
    peak_host = _launch_stage(
        smoke_root, MATRIX_SMOKE_STAGE, dry_run=False, resume=resume
    )
    elapsed = time.monotonic() - started
    rows = _matrix_smoke_evidence_rows(smoke_root)
    evidence_path = write_csv(_matrix_smoke_evidence_path(formal_root), rows)
    persisted_rows = _read_matrix_smoke_evidence(evidence_path)
    resources = _resource_peaks(smoke_root)
    registry = read_registry(smoke_root)
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": MATRIX_SMOKE_RESULT_KIND,
        "status": "passed",
        "source_config_sha256": config["source_config_sha256"],
        "selected_workers_per_gpu": workers,
        "job_count": EXPECTED_MATRIX_SMOKE_JOBS,
        "formal_cell_sha256": _formal_cell_digest(),
        "temporary_root": str(smoke_root),
        "temporary_root_cleanup_authorized_after_evidence": True,
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": sha256_file(evidence_path),
        "evidence_row_count": len(persisted_rows),
        "storage_gate_path": str(
            _matrix_smoke_storage_gate_path(formal_root).resolve()
        ),
        "storage_gate_sha256": sha256_file(
            _matrix_smoke_storage_gate_path(formal_root)
        ),
        "benchmark_result_path": str(_benchmark_result_path(formal_root).resolve()),
        "benchmark_result_sha256": sha256_file(_benchmark_result_path(formal_root)),
        "code_lineage_sha256": _lineage_digest(config, code=True),
        "source_lineage_sha256": _lineage_digest(config, code=False),
        "storage_gate": dict(registry.get("matrix_smoke_storage_gate") or {}),
        "peak_host_ram_fraction": float(peak_host),
        "elapsed_seconds": float(elapsed),
        "created_at_utc": utc_now(),
        **resources,
    }
    result["payload_sha256"] = payload_sha256(result)
    write_json(result_path, result)
    _require_matrix_smoke(formal_root, config)
    _cleanup_matrix_smoke_root(formal_root, workers)
    if smoke_root.exists():
        raise RuntimeError("Matrix-smoke temporary root cleanup did not complete")
    _require_matrix_smoke(formal_root, config)
    return result_path


# The stage actions below are completed further down in this module.  Keeping
# this dispatcher stable makes the module directly invokable with ``python -m``.
def run_action(
    action: str,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    resume: bool = False,
    job_id_value: str = "",
    worker_dry_run: bool = False,
) -> Path | dict[str, Any]:
    normalized = str(action).strip().lower()
    if normalized == "benchmark":
        return run_benchmark(config_path, output_dir, resume=resume)
    if normalized == "recovery-canary":
        return run_recovery_canary(config_path, output_dir, resume=resume)
    if normalized == "prepare":
        root = Path(output_dir).resolve()
        if not _recovery_canary_result_path(root).is_file():
            run_recovery_canary(config_path, root, resume=resume)
        if not _matrix_smoke_result_path(root).is_file():
            run_matrix_smoke(config_path, root, resume=resume)
        return prepare_experiment(config_path, output_dir, resume=resume)
    if normalized == "status":
        return status_experiment(output_dir)
    if normalized == "worker":
        if not job_id_value:
            raise ValueError("--job-id is required for worker")
        return run_worker(
            output_dir,
            job_id_value,
            dry_run=worker_dry_run,
            resume=resume,
        )
    if normalized == "dry-run" and not Path(output_dir).resolve().exists():
        # Before the formal root exists, the public dry-run action is the full
        # restore/load canary followed by the full 400-cell one-epoch preflight.
        # Once prepared, dry-run retains its historical parent-stage check.
        run_recovery_canary(config_path, output_dir, resume=resume)
        return run_matrix_smoke(config_path, output_dir, resume=resume)
    handlers = {
        "dry-run": dry_run,
        "launch-parents": launch_parents,
        "launch-continuations": launch_continuations,
        "freeze-branch-recipes": freeze_branch_recipes,
        "launch-text-branches": launch_text_branches,
        "freeze-evaluation": freeze_evaluation,
        "predict": predict,
        "evaluate-rq1": evaluate_rq1,
        "evaluate-rq2": evaluate_rq2,
        "evaluate-rq3": evaluate_rq3,
        "postprocess": postprocess,
        "qa": qa_experiment,
        "run-pipeline": lambda root, resume=False: run_pipeline(
            config_path, root, resume=resume
        ),
    }
    if normalized not in handlers:
        raise ValueError(f"Unsupported action: {action}")
    return handlers[normalized](output_dir, resume=resume)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "benchmark",
            "recovery-canary",
            "prepare",
            "dry-run",
            "worker",
            "launch-parents",
            "launch-continuations",
            "freeze-branch-recipes",
            "launch-text-branches",
            "freeze-evaluation",
            "predict",
            "evaluate-rq1",
            "evaluate-rq2",
            "evaluate-rq3",
            "postprocess",
            "qa",
            "status",
            "run-pipeline",
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--worker-dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_action(
        args.action,
        config_path=args.config,
        output_dir=args.output_dir,
        resume=bool(args.resume),
        job_id_value=args.job_id,
        worker_dry_run=bool(args.worker_dry_run),
    )
    print(
        json.dumps(result, indent=2, default=str)
        if isinstance(result, dict)
        else result
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
