"""Audited orchestration shell for the architecture and alignment-window studies.

The two studies intentionally share only lifecycle machinery.  Scientific
choices live in separate YAML files and terminal 5-minute baselines are bound
by SHA-256.  ``prepare`` never reads baseline test metrics, and no test-facing
action is permitted before the checkpoint allowlist has been frozen.

Model-specific code may materialize per-job ``train_vol.py`` configurations in
``registry/materialized_worker_specs.json``.  The generic worker verifies that
signed manifest and then invokes the repository's existing training entrypoint.
This keeps the new orchestration independent of experimental Generator
implementations while providing a concrete, resumable worker boundary.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import fcntl
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scripts.rq3.news_first_vol_experiment_supervisor_helpers import (
    SupervisorLock,
    append_stage_journal,
    atomic_write_json,
    gpu_availability_decision,
    pid_alive,
    process_start_ticks,
    system_resource_snapshot,
    utc_now,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHITECTURE_CONFIG = (
    "configs/rq3/news_first_vol_generator_architecture_3seed_5m.yaml"
)
DEFAULT_WINDOW_CONFIG = "configs/rq3/news_first_vol_alignment_tolerance_3seed.yaml"
ARCHITECTURE_KIND = "generator_architecture_3seed_5m_v1"
WINDOW_KIND = "alignment_tolerance_3seed_v1"
REGISTRY_KIND = "architecture_window_study_registry_v1"
BASELINE_KIND = "architecture_window_baseline_bindings_v1"
WORKER_SPEC_KIND = "architecture_window_materialized_worker_specs_v1"
MATCHED_CAPACITY_KIND = "architecture_matched_capacity_profiles_v1"
SEEDS = (42, 202, 404)
FOLDS = ("f1_2023q1", "f2_2023q2", "f3_2023q3", "f4_2023q4")
ARCHITECTURE_MODES = (
    "crossattn_unet_mask_coords_v1",
    "transformer_tokens_mask_coords_v1",
    "stylemod_unet_mask_coords_v1",
)
ARCHITECTURE_LRS = (1.0e-5, 2.5e-5, 5.0e-5)
WINDOW_TOLERANCES = (5, 10, 15, 20, 30)
WINDOW_ARMS = ("film_lp_matched", "pure_cnn_no_text")
CHECKPOINT_ROLES = (
    "generator_best_learned",
    "discriminator_best_learned",
)
EXACT_TTM_DAYS = (1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38)
EXACT_MONEYNESS = tuple(float(value) for value in np.linspace(0.97, 1.03, 16))
TARGET_SCALED_GENERATOR_PARAMETERS = 1_655_490
TARGET_MATCHED_GENERATOR_PARAMETERS = 827_745
MATCHED_GENERATOR_PARAMETER_COUNTS = {
    "crossattn_unet_mask_coords_v1": 824_644,
    "transformer_tokens_mask_coords_v1": 834_561,
    "stylemod_unet_mask_coords_v1": 829_635,
}
WINDOW_RELATIONS = (
    "current_pre_news_target_post_news",
    "current_partially_post_news",
    "current_fully_post_news",
)
PAIR_UNIVERSE_KIND = "architecture_window_pair_universe_v1"
PREDICTION_PLAN_KIND = "architecture_window_prediction_plan_v1"
GPU_BENCHMARK_KIND = "architecture_window_gpu_epoch1_benchmark_v1"
GLOBAL_GPU_SLOT_FD_ENV = "ARCHITECTURE_WINDOW_GLOBAL_GPU_SLOT_FD"


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else REPO_ROOT / value).resolve()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = _payload_sha256(result)
    return result


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    return _mapping(value, str(path))


def _read_signed(path: str | Path, *, kind: str | None = None) -> dict[str, Any]:
    payload = _read_json(path)
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("payload_sha256") != _payload_sha256(unsigned):
        raise ValueError(f"Signed payload drift: {path}")
    if kind is not None and payload.get("kind") != kind:
        raise ValueError(f"Signed payload kind drift: {path}")
    return payload


def _write_signed(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return atomic_write_json(path, _signed(payload))


def _job_id(*parts: object) -> str:
    text = "__".join(str(part) for part in parts)
    return text.replace(".", "p").replace("-", "m")


def _job_spec(cell: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(cell)
    result["job_spec_sha256"] = _payload_sha256(result)
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    source = _resolve(config_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    top = _mapping(raw, "config")
    config = _mapping(top.get("study"), "study config")
    for key in (
        "data",
        "folds",
        "matrix",
        "baselines",
        "training",
        "analysis",
        "runtime",
    ):
        config[key] = top.get(key)
    config["source_config_path"] = str(source)
    config["source_config_sha256"] = _sha256_file(source)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("study.schema_version must be 1")
    kind = str(config.get("study_kind", ""))
    line = str(config.get("line", ""))
    if (kind, line) not in (
        (ARCHITECTURE_KIND, "architecture"),
        (WINDOW_KIND, "window"),
    ):
        raise ValueError("Unknown study kind/line")
    if config.get("interpretation") != "retrospective_rolling_development_3seed":
        raise ValueError(
            "Interpretation label must remain retrospective_rolling_development_3seed"
        )
    expected_scope = {
        "architecture": "generator_architecture_selection",
        "window": "alignment_tolerance_sensitivity",
    }[line]
    if config.get("study_scope") != expected_scope:
        raise ValueError(f"Study scope drift: expected {expected_scope}")
    expected_root_name = {
        "architecture": "rq3_news_first_vol_generator_architecture_3seed_5m_exact_ttm_v1",
        "window": "rq3_news_first_vol_alignment_tolerance_3seed_exact_ttm_v1",
    }[line]
    if Path(str(config.get("output_root", ""))).name != expected_root_name:
        raise ValueError(
            f"Formal output-root basename drift: expected {expected_root_name}"
        )
    data = _mapping(config.get("data"), "data")
    matrix = _mapping(config.get("matrix"), "matrix")
    runtime = _mapping(config.get("runtime"), "runtime")
    folds = list(config.get("folds") or [])
    if tuple(str(row["id"]) for row in folds) != FOLDS:
        raise ValueError("Rolling fold order drift")
    if tuple(map(int, matrix.get("seeds", ()))) != SEEDS:
        raise ValueError("Three-seed contract drift")
    if data.get("surface_grid_profile") != "exact_ttm_16x16_v1":
        raise ValueError("Exact-TTM grid is required")
    if data.get("support_mask_mode") != "raw_joint":
        raise ValueError("raw_joint support is required")
    if int(data.get("target_horizon_minutes", -1)) != 5:
        raise ValueError("Prediction target must remain five minutes")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != (0, 1):
        raise ValueError("Runtime requires GPU 0 and GPU 1")
    if int(runtime.get("workers_per_gpu", 0)) < 1:
        raise ValueError("workers_per_gpu must be positive")
    if tuple(map(int, runtime.get("benchmark_worker_candidates_per_gpu", ()))) != (
        4,
        8,
        12,
        18,
    ):
        raise ValueError("Benchmark worker candidates must be [4, 8, 12, 18]")
    if (
        int(runtime.get("benchmark_epochs", -1)) != 1
        or float(runtime.get("preflight_max_peak_gpu_memory_gib", math.nan)) != 20.0
        or float(runtime.get("preflight_max_host_ram_fraction", math.nan)) != 0.85
    ):
        raise ValueError("Benchmark resource gate drift")

    if kind == ARCHITECTURE_KIND:
        if tuple(matrix.get("modes", ())) != ARCHITECTURE_MODES:
            raise ValueError("Architecture mode order drift")
        screen = _mapping(matrix.get("screen"), "matrix.screen")
        formal = _mapping(matrix.get("formal"), "matrix.formal")
        scaled = _mapping(matrix.get("scaled"), "matrix.scaled")
        if (
            tuple(map(float, screen.get("conditioning_learning_rates", ())))
            != ARCHITECTURE_LRS
        ):
            raise ValueError("Architecture LR screen drift")
        if int(screen.get("num_epochs", -1)) != 60:
            raise ValueError("Architecture screen must train for exactly 60 epochs")
        if (
            screen.get("selection_rule")
            != "lowest_validation_mae_at_best_hybrid_checkpoint_per_mode_v1"
        ):
            raise ValueError("Architecture screen checkpoint-selection rule drift")
        expected = (
            int(screen.get("expected_training_jobs", -1)),
            int(formal.get("expected_training_jobs", -1)),
            int(scaled.get("expected_training_jobs", -1)),
            int(matrix.get("expected_new_training_jobs", -1)),
            int(matrix.get("expected_reused_baseline_cells", -1)),
        )
        if expected != (9, 36, 24, 69, 24):
            raise ValueError(f"Architecture count contract drift: {expected}")
        target = int(scaled.get("target_generator_parameters", -1))
        relative = float(scaled.get("maximum_relative_deviation", math.nan))
        search = _mapping(
            scaled.get("deterministic_search_grid"), "scaled deterministic search grid"
        )
        if (
            target != TARGET_SCALED_GENERATOR_PARAMETERS
            or relative != 0.05
            or set(search) != {*ARCHITECTURE_MODES, "film_unet_mask_coords_v1"}
        ):
            raise ValueError("Scaled executable parameter-search contract drift")
        matched = _mapping(matrix.get("matched_capacity"), "matched capacity")
        matched_search = _mapping(
            matched.get("deterministic_search_grid"),
            "matched deterministic search grid",
        )
        matched_expected = _mapping(
            matched.get("expected_winner_generator_parameters"),
            "matched expected winner parameters",
        )
        if (
            int(matched.get("target_generator_parameters", -1))
            != TARGET_MATCHED_GENERATOR_PARAMETERS
            or float(matched.get("maximum_relative_deviation", math.nan)) != 0.05
            or matched.get("profile_selection")
            != "closest_executable_parameter_count_then_smallest_base_channels_v1"
            or set(matched_search) != set(ARCHITECTURE_MODES)
            or {
                mode: int(matched_expected.get(mode, -1)) for mode in ARCHITECTURE_MODES
            }
            != MATCHED_GENERATOR_PARAMETER_COUNTS
        ):
            raise ValueError("Matched executable parameter-search contract drift")
        required_search_keys = {
            "crossattn_unet_mask_coords_v1": {
                "gen_base_channels",
                "gen_crossattn_heads",
                "gen_crossattn_text_tokens",
                "gen_crossattn_dim",
            },
            "transformer_tokens_mask_coords_v1": {
                "gen_base_channels",
                "gen_transformer_model_dim",
                "gen_transformer_layers",
                "gen_transformer_heads",
                "gen_transformer_ffn_dim",
                "gen_transformer_dropout",
            },
            "stylemod_unet_mask_coords_v1": {
                "gen_base_channels",
                "gen_style_dim",
                "gen_style_demodulate",
            },
        }
        for mode, required_keys in required_search_keys.items():
            grid = _mapping(matched_search[mode], f"matched search grid {mode}")
            if set(grid) != required_keys:
                raise ValueError(f"Matched search-grid fields drift: {mode}")
            for key, raw_values in grid.items():
                _grid_values(raw_values, label=f"matched.{mode}.{key}")
        for raw in scaled.get("variants", []):
            variant = _mapping(raw, "scaled variant")
            if (
                variant.get("capacity_profile_source")
                != "deterministic_parameter_search"
            ):
                raise ValueError("Scaled variants must use executable parameter search")
    else:
        equivalence = _mapping(
            data.get("reused_5m_semantic_equivalence"),
            "data.reused_5m_semantic_equivalence",
        )
        required = {
            "reference_workbook",
            "reference_support",
            "expected_workbook_semantic_sha256",
            "expected_support_semantic_sha256",
            "profile",
        }
        if set(equivalence) != required:
            raise ValueError("Reused 5m semantic-equivalence contract drift")
        if equivalence["profile"] != "exact_cellwise_string_equivalence_v1":
            raise ValueError("Unknown reused 5m semantic-equivalence profile")
        for key in (
            "expected_workbook_semantic_sha256",
            "expected_support_semantic_sha256",
        ):
            value = str(equivalence[key])
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"Invalid semantic SHA-256: {key}")
        if tuple(map(int, matrix.get("tolerances_minutes", ()))) != WINDOW_TOLERANCES:
            raise ValueError("Window tolerance order drift")
        if tuple(matrix.get("arms", {}).keys()) != WINDOW_ARMS:
            raise ValueError("Window arm order drift")
        expected = (
            int(matrix.get("expected_logical_cells", -1)),
            int(matrix.get("expected_reused_baseline_cells", -1)),
            int(matrix.get("expected_new_training_jobs", -1)),
        )
        if expected != (120, 24, 96):
            raise ValueError(f"Window count contract drift: {expected}")
        analysis = _mapping(config.get("analysis"), "analysis")
        if (
            analysis.get("primary_evaluation_panel") != "common_tolerance_05m_v1"
            or analysis.get("secondary_evaluation_panel")
            != "own_alignment_tolerance_v1"
            or int(analysis.get("primary_prediction_units", -1)) != 120
            or int(analysis.get("secondary_additional_prediction_units", -1)) != 96
            or int(analysis.get("expected_total_prediction_units", -1)) != 216
            or int(analysis.get("expected_total_evaluation_units", -1)) != 324
        ):
            raise ValueError("Window common-5m/own-panel prediction contract drift")


def _gpu(config: Mapping[str, Any], fold: str) -> int:
    assignment = _mapping(
        _mapping(config["runtime"], "runtime")["gpu_fold_assignment"], "gpu assignment"
    )
    return int(assignment[fold])


def _matched_capacity_profile_name(mode: str) -> str:
    if mode not in ARCHITECTURE_MODES:
        raise ValueError(f"Unsupported matched-capacity mode: {mode}")
    return f"auto_match827745_{mode}"


def planned_cells(config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic new-training and read-only baseline cell universes."""

    validate_config(config)
    kind = str(config["study_kind"])
    new: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    if kind == ARCHITECTURE_KIND:
        matrix = _mapping(config["matrix"], "matrix")
        screen = _mapping(matrix["screen"], "matrix.screen")
        for mode in ARCHITECTURE_MODES:
            for lr in ARCHITECTURE_LRS:
                screen_index = ARCHITECTURE_MODES.index(mode) * len(
                    ARCHITECTURE_LRS
                ) + ARCHITECTURE_LRS.index(lr)
                new.append(
                    _job_spec(
                        {
                            "job_id": _job_id(
                                "arch",
                                "screen",
                                mode,
                                f"lr_{lr:.1e}",
                                "seed_42",
                                screen["fold"],
                            ),
                            "stage": "screen",
                            "generator_mode": mode,
                            "conditioning_learning_rate": lr,
                            "seed": 42,
                            "fold": str(screen["fold"]),
                            "tolerance_minutes": 5,
                            "capacity_profile": _matched_capacity_profile_name(mode),
                            "arm": "lp_matched",
                            "gpu_id": int(
                                _mapping(config["runtime"], "runtime")["gpu_ids"][
                                    screen_index % 2
                                ]
                            ),
                            "initial_status": "pending",
                        }
                    )
                )
        for mode in ARCHITECTURE_MODES:
            for seed in SEEDS:
                for fold in FOLDS:
                    new.append(
                        _job_spec(
                            {
                                "job_id": _job_id(
                                    "arch", "formal", mode, f"seed_{seed}", fold
                                ),
                                "stage": "formal",
                                "generator_mode": mode,
                                "conditioning_learning_rate": None,
                                "conditioning_learning_rate_source": f"screen_selection:{mode}",
                                "seed": seed,
                                "fold": fold,
                                "tolerance_minutes": 5,
                                "capacity_profile": _matched_capacity_profile_name(
                                    mode
                                ),
                                "arm": "lp_matched",
                                "gpu_id": _gpu(config, fold),
                                "initial_status": "blocked_on_screen_selection",
                            }
                        )
                    )
        variants = list(_mapping(matrix["scaled"], "matrix.scaled")["variants"])
        for variant in variants:
            spec = _mapping(variant, "scaled variant")
            for seed in SEEDS:
                for fold in FOLDS:
                    new.append(
                        _job_spec(
                            {
                                "job_id": _job_id(
                                    "arch", "scaled", spec["id"], f"seed_{seed}", fold
                                ),
                                "stage": "scaled",
                                "variant": str(spec["id"]),
                                "generator_mode": spec.get("generator_mode"),
                                "generator_mode_source": spec.get(
                                    "generator_mode_source"
                                ),
                                "conditioning_learning_rate": None,
                                "conditioning_learning_rate_source": "frozen_architecture_selection",
                                "seed": seed,
                                "fold": fold,
                                "tolerance_minutes": 5,
                                "capacity_profile": None,
                                "capacity_profile_source": str(
                                    spec["capacity_profile_source"]
                                ),
                                "arm": "lp_matched",
                                "gpu_id": _gpu(config, fold),
                                "initial_status": "blocked_on_architecture_selection",
                            }
                        )
                    )
        for baseline in ("film_reference", "pure_cnn_reference"):
            for seed in SEEDS:
                for fold in FOLDS:
                    reused.append(
                        {
                            "cell_id": _job_id(
                                "arch", "baseline", baseline, f"seed_{seed}", fold
                            ),
                            "baseline": baseline,
                            "seed": seed,
                            "fold": fold,
                            "tolerance_minutes": 5,
                            "execution": "reuse_terminal_checkpoint",
                        }
                    )
        counts = Counter(row["stage"] for row in new)
        if counts != {"screen": 9, "formal": 36, "scaled": 24} or len(reused) != 24:
            raise AssertionError(
                f"Architecture matrix count drift: {counts}/{len(reused)}"
            )
    else:
        matrix = _mapping(config["matrix"], "matrix")
        reused_tolerances = set(map(int, matrix["reused_tolerances_minutes"]))
        for tolerance in WINDOW_TOLERANCES:
            for arm in WINDOW_ARMS:
                for seed in SEEDS:
                    for fold in FOLDS:
                        base = {
                            "cell_id": _job_id(
                                "window", f"t{tolerance:02d}", arm, f"seed_{seed}", fold
                            ),
                            "stage": "window",
                            "arm": arm,
                            "seed": seed,
                            "fold": fold,
                            "tolerance_minutes": tolerance,
                            "gpu_id": _gpu(config, fold),
                        }
                        if tolerance in reused_tolerances:
                            reused.append(
                                {
                                    **base,
                                    "baseline": arm,
                                    "execution": "reuse_terminal_checkpoint",
                                }
                            )
                        else:
                            arm_config = _mapping(
                                _mapping(matrix["arms"], "matrix.arms")[arm],
                                f"arm {arm}",
                            )
                            new.append(
                                _job_spec(
                                    {
                                        **base,
                                        "job_id": base["cell_id"],
                                        "generator_mode": str(
                                            arm_config["generator_mode"]
                                        ),
                                        "capacity_profile": str(
                                            arm_config["capacity_profile"]
                                        ),
                                        "conditioning_learning_rate": arm_config.get(
                                            "generator_conditioning_learning_rate"
                                        ),
                                        "initial_status": "pending",
                                    }
                                )
                            )
        if len(new) != 96 or len(reused) != 24 or len(new) + len(reused) != 120:
            raise AssertionError(
                "Window matrix must contain 120 logical / 96 new / 24 reused cells"
            )
    if len({row["job_id"] for row in new}) != len(new):
        raise AssertionError("Duplicate training job IDs")
    return {"new_training_jobs": new, "reused_baseline_cells": reused}


def _semantic_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash an ordered table independently of its xlsx/gzip container bytes."""

    digest = hashlib.sha256()
    digest.update(
        _canonical_bytes({"columns": list(frame.columns), "shape": frame.shape})
    )
    for row in frame.itertuples(index=False, name=None):
        for raw in row:
            value = str(raw).encode("utf-8")
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def _reused_5m_semantic_equivalence(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Prove that reused old-root checkpoints see the identical 5m table."""

    if config["line"] != "window":
        return None
    data = _mapping(config["data"], "data")
    contract = _mapping(
        data["reused_5m_semantic_equivalence"],
        "data.reused_5m_semantic_equivalence",
    )
    candidate_root = _resolve(str(data["root"]))
    paths = {
        "reference_workbook": _resolve(str(contract["reference_workbook"])),
        "candidate_workbook": candidate_root
        / str(data["workbook_template"]).format(tolerance02="05"),
        "reference_support": _resolve(str(contract["reference_support"])),
        "candidate_support": candidate_root
        / str(data["support_template"]).format(tolerance02="05"),
    }
    if not all(path.is_file() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        raise FileNotFoundError(f"Reused 5m equivalence input missing: {missing}")
    workbook_frames = {
        role: pd.read_excel(
            path,
            sheet_name=str(data.get("sheet_name", "gan_input_ready")),
            dtype=str,
            keep_default_na=False,
        )
        for role, path in paths.items()
        if role.endswith("workbook")
    }
    support_frames = {
        role: pd.read_csv(path, dtype=str, keep_default_na=False)
        for role, path in paths.items()
        if role.endswith("support")
    }
    reference_workbook = workbook_frames["reference_workbook"]
    candidate_workbook = workbook_frames["candidate_workbook"]
    reference_support = support_frames["reference_support"]
    candidate_support = support_frames["candidate_support"]
    if not reference_workbook.equals(candidate_workbook):
        raise ValueError("Old-root and with20 5m workbook semantics differ")
    if not reference_support.equals(candidate_support):
        raise ValueError("Old-root and with20 5m support semantics differ")
    workbook_sha = _semantic_frame_sha256(candidate_workbook)
    support_sha = _semantic_frame_sha256(candidate_support)
    if workbook_sha != str(contract["expected_workbook_semantic_sha256"]):
        raise ValueError("Reused 5m workbook semantic SHA drift")
    if support_sha != str(contract["expected_support_semantic_sha256"]):
        raise ValueError("Reused 5m support semantic SHA drift")
    return {
        "profile": contract["profile"],
        "workbook_semantic_sha256": workbook_sha,
        "support_semantic_sha256": support_sha,
        "inputs": [
            {
                "role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for role, path in sorted(paths.items())
        ],
    }


def input_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(config["data"], "data")
    root = _resolve(str(data["root"]))
    tolerances = tuple(map(int, data["tolerances_minutes"]))
    rows: list[dict[str, Any]] = []
    declared_hashes: dict[str, str] = {}
    output_manifest = data.get("output_sha_manifest")
    lineage_inputs: list[dict[str, Any]] = []
    for role in ("build_config", "validation_manifest"):
        raw_path = data.get(role)
        if not raw_path:
            continue
        path = _resolve(str(raw_path))
        if not path.is_file():
            raise FileNotFoundError(f"Dataset {role} missing: {path}")
        lineage_inputs.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if output_manifest:
        manifest_path = _resolve(str(output_manifest))
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Dataset output SHA manifest missing: {manifest_path}"
            )
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.strip().split(maxsplit=1)
            declared_hashes[relative.strip()] = digest
    for tolerance in tolerances:
        values = {"tolerance_minutes": tolerance}
        for key in ("workbook", "support", "validation"):
            template = str(data[f"{key}_template"])
            path = root / template.format(tolerance02=f"{tolerance:02d}")
            values[f"{key}_path"] = str(path.resolve())
            values[f"{key}_exists"] = path.is_file()
            if path.is_file():
                values[f"{key}_sha256"] = _sha256_file(path)
                values[f"{key}_size_bytes"] = path.stat().st_size
                if declared_hashes:
                    relative = path.relative_to(root).as_posix()
                    if declared_hashes.get(relative) != values[f"{key}_sha256"]:
                        raise ValueError(f"Dataset output SHA drift: {relative}")
        validation_path = Path(str(values["validation_path"]))
        if validation_path.is_file():
            validation = _read_json(validation_path)
            if validation.get("status") != "pass":
                raise ValueError(f"Dataset validation did not pass: {validation_path}")
            expected_pairs = _mapping(
                data.get("expected_joint_strict_support_pairs", {}),
                "expected joint support counts",
            )
            expected = expected_pairs.get(tolerance, expected_pairs.get(str(tolerance)))
            if expected is not None and int(
                validation.get("joint_strict_support_pairs", -1)
            ) != int(expected):
                raise ValueError(f"Joint-support count drift for {tolerance}m")
        rows.append(values)
    missing = [
        values[f"{key}_path"]
        for values in rows
        for key in ("workbook", "support", "validation")
        if not values[f"{key}_exists"]
    ]
    return {
        "status": "passed" if not missing else "blocked",
        "rows": rows,
        "missing": missing,
        "output_sha_manifest_path": (
            str(_resolve(str(output_manifest))) if output_manifest else ""
        ),
        "output_sha_manifest_sha256": (
            _sha256_file(_resolve(str(output_manifest))) if output_manifest else ""
        ),
        "dataset_lineage_inputs": lineage_inputs,
        "reused_5m_semantic_equivalence": _reused_5m_semantic_equivalence(config),
    }


def _verify_file(path: Path, expected_sha: str, label: str) -> None:
    if not path.is_file() or _sha256_file(path) != str(expected_sha):
        raise ValueError(f"{label} missing or SHA drift: {path}")


def bind_baselines(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind QA/registry/checkpoints without opening historical test metrics."""

    baselines = _mapping(config["baselines"], "baselines")
    rows: list[dict[str, Any]] = []
    for name, raw in baselines.items():
        baseline = _mapping(raw, f"baseline {name}")
        root = _resolve(str(baseline["root"]))
        qa_path = root / "qa.json"
        registry_path = root / "registry/task_registry.json"
        allowlist_path = root / "registry/evaluation_checkpoint_allowlist.csv"
        _verify_file(qa_path, str(baseline["qa_sha256"]), f"{name} QA")
        _verify_file(
            registry_path, str(baseline["registry_sha256"]), f"{name} registry"
        )
        _verify_file(
            allowlist_path,
            str(baseline["checkpoint_allowlist_sha256"]),
            f"{name} checkpoint allowlist",
        )
        qa = _read_json(qa_path)
        if qa.get("status") != "passed":
            raise ValueError(f"Baseline {name} is not terminal-passed")
        frame = pd.read_csv(allowlist_path)
        selected = frame.loc[
            frame["seed"].astype(int).isin(SEEDS)
            & frame["fold"].astype(str).isin(FOLDS)
            & frame["arm"].astype(str).eq(str(baseline["arm"]))
            & frame["tolerance_minutes"].astype(int).eq(5)
            & frame["checkpoint_role"].astype(str).isin(CHECKPOINT_ROLES)
        ].copy()
        if len(selected) != 24:
            raise ValueError(f"Baseline {name} must bind 12 G/D checkpoint cells")
        grouped = selected.groupby(["seed", "fold"])["checkpoint_role"].agg(set)
        if len(grouped) != 12 or any(
            value != set(CHECKPOINT_ROLES) for value in grouped
        ):
            raise ValueError(f"Baseline {name} checkpoint role coverage drift")
        for row in selected.to_dict(orient="records"):
            checkpoint = Path(str(row["checkpoint_path"])).resolve()
            if (
                not checkpoint.is_file()
                or checkpoint.stat().st_size != int(row["size_bytes"])
                or _sha256_file(checkpoint) != str(row["checkpoint_sha256"])
            ):
                raise ValueError(f"Baseline checkpoint drift: {checkpoint}")
            rows.append({"baseline": str(name), **row})
    return {
        "schema_version": 1,
        "kind": BASELINE_KIND,
        "bound_at_utc": utc_now(),
        "test_metric_files_read": 0,
        "selection_uses_test_metrics": False,
        "checkpoint_rows": rows,
    }


def _fold(config: Mapping[str, Any], fold_id: str) -> dict[str, Any]:
    rows = [dict(row) for row in config["folds"] if str(row["id"]) == fold_id]
    if len(rows) != 1:
        raise KeyError(fold_id)
    return rows[0]


def _as_bool(series: pd.Series, *, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - {"true", "false", "1", "0"})
    if unknown:
        raise ValueError(f"{label} contains invalid booleans: {unknown[:5]}")
    return normalized.isin({"true", "1"})


def _relation_from_post_news_overlap(overlap_minutes: float) -> str:
    """Map the five-minute current-window overlap to its audit stratum."""

    value = float(overlap_minutes)
    if value == 0.0:
        return "current_pre_news_target_post_news"
    if 0.0 < value < 5.0:
        return "current_partially_post_news"
    if value == 5.0:
        return "current_fully_post_news"
    raise ValueError(f"Post-news current overlap is outside [0, 5]: {value}")


def _adjacent_window_tolerance_pairs() -> tuple[tuple[int, int], ...]:
    """Return all four ordered nesting comparisons without unequal strict zip."""

    return tuple(zip(WINDOW_TOLERANCES[:-1], WINDOW_TOLERANCES[1:], strict=True))


def _collapse_pair_window_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse article-level timing metadata to one conservative pair stratum.

    Multiple news articles can legitimately align to the same market pair.  The
    model then consumes their pair-level aggregate embedding, so article-level
    ``window_relation`` is not a pair invariant.  We classify the pair by its
    maximum post-news current-window overlap: if *any* contributing article has
    already affected more of the current surface, the pair remains in the more
    conservative contamination stratum.  This is target-free, deterministic,
    preserves every frozen pair, and never understates possible post-news input.
    """

    required = {
        "pair_id",
        "article_id",
        "sample_id",
        "news_row_id",
        "news_available_time_utc",
        "current_window_start_utc",
        "current_window_end_utc",
        "window_relation",
        "post_news_current_overlap_minutes",
    }
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Pair window metadata is missing or empty")
    from wgan_option.utils.news_first_experiment_core import _normalized_article_key

    working = frame[list(required)].copy()
    working["pair_id"] = working["pair_id"].astype(str)
    for column in (
        "news_available_time_utc",
        "current_window_start_utc",
        "current_window_end_utc",
    ):
        try:
            working[column] = pd.to_datetime(working[column], errors="raise", utc=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid article-level window timestamp: {column}"
            ) from exc
    duration_seconds = (
        working["current_window_end_utc"] - working["current_window_start_utc"]
    ).dt.total_seconds()
    if not duration_seconds.eq(300.0).all():
        raise ValueError("Article-level current window must be exactly five minutes")
    if (working["news_available_time_utc"] > working["current_window_end_utc"]).any():
        raise ValueError("News availability is after the aligned market origin")
    recomputed_overlap = (
        (working["current_window_end_utc"] - working["news_available_time_utc"])
        .dt.total_seconds()
        .div(60.0)
        .clip(lower=0.0, upper=5.0)
    )
    if not np.isclose(
        recomputed_overlap.to_numpy(), np.rint(recomputed_overlap.to_numpy())
    ).all():
        raise ValueError("Article-level window timestamps are not minute aligned")
    recomputed_overlap = recomputed_overlap.astype(float)
    recomputed_relation = recomputed_overlap.map(_relation_from_post_news_overlap)
    stored_overlap = pd.to_numeric(
        frame["post_news_current_overlap_minutes"], errors="coerce"
    ).astype(float)
    if (
        not np.isfinite(stored_overlap.to_numpy()).all()
        or not np.array_equal(stored_overlap.to_numpy(), recomputed_overlap.to_numpy())
        or not frame["window_relation"].astype(str).eq(recomputed_relation).all()
    ):
        raise ValueError(
            "Stored article-level window metadata disagrees with source timestamps"
        )
    working["window_relation"] = recomputed_relation
    working["post_news_current_overlap_minutes"] = recomputed_overlap
    pair_order = {
        pair_id: index
        for index, pair_id in enumerate(dict.fromkeys(working["pair_id"]))
    }
    working["_pair_order"] = working["pair_id"].map(pair_order)
    working["_news_sort"] = pd.to_numeric(working["news_row_id"], errors="raise")
    working["_sample_sort"] = working["sample_id"].fillna("").astype(str)
    working = working.sort_values(
        ["_pair_order", "_news_sort", "_sample_sort"], kind="stable"
    )
    working["_article_key"] = [
        _normalized_article_key(row) for row in working.itertuples(index=False)
    ]
    working = working.drop_duplicates(["pair_id", "_article_key"], keep="first")
    collapsed = (
        working.groupby("pair_id", sort=False, as_index=False)
        .agg(
            aligned_article_count=("_article_key", "size"),
            window_relation_variant_count=("window_relation", "nunique"),
            post_news_current_overlap_variant_count=(
                "post_news_current_overlap_minutes",
                "nunique",
            ),
            post_news_current_overlap_min_minutes=(
                "post_news_current_overlap_minutes",
                "min",
            ),
            post_news_current_overlap_minutes=(
                "post_news_current_overlap_minutes",
                "max",
            ),
        )
        .reset_index(drop=True)
    )
    collapsed["window_relation"] = collapsed["post_news_current_overlap_minutes"].map(
        _relation_from_post_news_overlap
    )
    return collapsed


def _materialize_pair_universe(config: Mapping[str, Any], root: Path) -> Path:
    """Freeze support-filtered fold partitions without reading target values/errors."""

    data = _mapping(config["data"], "data")
    dataset_root = _resolve(str(data["root"]))
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    relation_audits: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for tolerance in map(int, data["tolerances_minutes"]):
        workbook = dataset_root / str(data["workbook_template"]).format(
            tolerance02=f"{tolerance:02d}"
        )
        support_path = dataset_root / str(data["support_template"]).format(
            tolerance02=f"{tolerance:02d}"
        )
        frame = pd.read_excel(
            workbook,
            sheet_name=str(data.get("sheet_name", "gan_input_ready")),
            usecols=[
                "pair_id",
                "session_id",
                "effective_origin_utc",
                "current_snapshot_time_utc",
                "target_snapshot_time_utc",
                "article_id",
                "news_row_id",
                "sample_id",
                "news_available_time_utc",
                "current_window_start_utc",
                "current_window_end_utc",
                "window_relation",
                "post_news_current_overlap_minutes",
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
        observed_fingerprints = set(support["grid_fingerprint"].astype(str))
        if len(observed_fingerprints) != 1:
            raise ValueError(f"Grid fingerprint is not unique for {tolerance}m")
        fingerprints.update(observed_fingerprints)
        eligible = _as_bool(
            support["surface_training_eligible"],
            label=f"{tolerance}m surface_training_eligible",
        )
        zero = _as_bool(
            support["joint_zero_support"], label=f"{tolerance}m joint_zero_support"
        )
        support = support.loc[
            eligible
            & ~zero
            & (
                pd.to_numeric(
                    support["joint_strict_support_cell_count"], errors="raise"
                )
                > 0
            )
        ].copy()
        support["pair_id"] = support["pair_id"].astype(str)
        if support["pair_id"].duplicated().any():
            raise ValueError(f"Duplicate support pair IDs for {tolerance}m")
        frame["pair_id"] = frame["pair_id"].astype(str)
        frame = frame.loc[frame["pair_id"].isin(set(support["pair_id"]))].copy()
        if set(frame["pair_id"]) != set(support["pair_id"]):
            raise ValueError(f"Workbook/support pair coverage drift for {tolerance}m")
        frame["effective_origin_utc"] = pd.to_datetime(
            frame["effective_origin_utc"], errors="raise", utc=True
        )
        current = pd.to_datetime(
            frame["current_snapshot_time_utc"], errors="raise", utc=True
        )
        target = pd.to_datetime(
            frame["target_snapshot_time_utc"], errors="raise", utc=True
        )
        current_start = pd.to_datetime(
            frame["current_window_start_utc"], errors="raise", utc=True
        )
        current_end = pd.to_datetime(
            frame["current_window_end_utc"], errors="raise", utc=True
        )
        if not (target - current).dt.total_seconds().eq(300.0).all():
            raise ValueError(f"{tolerance}m data do not retain a five-minute target")
        if (
            not current_end.eq(current).all()
            or not (current_end - current_start).dt.total_seconds().eq(300.0).all()
        ):
            raise ValueError(f"{tolerance}m current-window timestamps drift")
        for column in (
            "session_id",
            "effective_origin_utc",
            "current_snapshot_time_utc",
            "target_snapshot_time_utc",
            "current_window_start_utc",
            "current_window_end_utc",
        ):
            if int(frame.groupby("pair_id", sort=False)[column].nunique().max()) != 1:
                raise ValueError(f"Pair invariant drift for {tolerance}m: {column}")
        pair_window = _collapse_pair_window_metadata(frame)
        pair_rows = (
            frame[
                [
                    "pair_id",
                    "session_id",
                    "effective_origin_utc",
                    "current_snapshot_time_utc",
                    "target_snapshot_time_utc",
                ]
            ]
            .sort_values(["effective_origin_utc", "pair_id"], kind="stable")
            .drop_duplicates("pair_id", keep="first")
            .merge(pair_window, on="pair_id", validate="one_to_one")
            .merge(
                support[["pair_id", "joint_strict_support_cell_count"]],
                on="pair_id",
                validate="one_to_one",
            )
        )
        observed_relations = set(pair_rows["window_relation"].astype(str))
        if observed_relations != set(WINDOW_RELATIONS):
            raise ValueError(
                f"Window-relation strata drift for {tolerance}m: "
                f"{sorted(observed_relations)}"
            )
        overlap = pd.to_numeric(
            pair_rows["post_news_current_overlap_minutes"], errors="coerce"
        )
        if not np.isfinite(overlap.to_numpy(float)).all() or (overlap < 0).any():
            raise ValueError(f"Invalid post-news current overlap for {tolerance}m")
        relation_audits.append(
            {
                "tolerance_minutes": tolerance,
                "pair_count": len(pair_rows),
                "multi_article_pair_count": int(
                    pair_rows["aligned_article_count"].gt(1).sum()
                ),
                "mixed_window_relation_pair_count": int(
                    pair_rows["window_relation_variant_count"].gt(1).sum()
                ),
                "overlap_variant_pair_count": int(
                    pair_rows["post_news_current_overlap_variant_count"].gt(1).sum()
                ),
            }
        )
        for fold_id in FOLDS:
            fold = _fold(config, fold_id)
            train_end = pd.Timestamp(fold["train_end_utc"])
            validation_end = pd.Timestamp(fold["validation_end_utc"])
            test_end = pd.Timestamp(fold["test_end_utc"])
            partitions = {
                "train": pair_rows["effective_origin_utc"] < train_end,
                "validation": (pair_rows["effective_origin_utc"] >= train_end)
                & (pair_rows["effective_origin_utc"] < validation_end),
                "test": (pair_rows["effective_origin_utc"] >= validation_end)
                & (pair_rows["effective_origin_utc"] < test_end),
            }
            for partition, mask in partitions.items():
                selected = pair_rows.loc[mask].copy()
                if selected.empty:
                    raise ValueError(
                        f"Empty pair universe for {tolerance}m/{fold_id}/{partition}"
                    )
                pair_ids = sorted(selected["pair_id"].astype(str))
                universe_sha = _payload_sha256(pair_ids)
                summaries.append(
                    {
                        "tolerance_minutes": tolerance,
                        "fold": fold_id,
                        "partition": partition,
                        "pair_count": len(pair_ids),
                        "session_count": int(
                            selected["session_id"].astype(str).nunique()
                        ),
                        "pair_universe_sha256": universe_sha,
                    }
                )
                for item in selected.itertuples(index=False):
                    rows.append(
                        {
                            "tolerance_minutes": tolerance,
                            "fold": fold_id,
                            "partition": partition,
                            "pair_id": str(item.pair_id),
                            "session_id": str(item.session_id),
                            "effective_origin_utc": pd.Timestamp(
                                item.effective_origin_utc
                            ).isoformat(),
                            "joint_support_cells": int(
                                item.joint_strict_support_cell_count
                            ),
                            "window_relation": str(item.window_relation),
                            "post_news_current_overlap_minutes": float(
                                item.post_news_current_overlap_minutes
                            ),
                            "post_news_current_overlap_min_minutes": float(
                                item.post_news_current_overlap_min_minutes
                            ),
                            "aligned_article_count": int(item.aligned_article_count),
                            "window_relation_variant_count": int(
                                item.window_relation_variant_count
                            ),
                            "post_news_current_overlap_variant_count": int(
                                item.post_news_current_overlap_variant_count
                            ),
                            "pair_universe_sha256": universe_sha,
                        }
                    )
    if len(fingerprints) != 1:
        raise ValueError(
            f"Exact-TTM grid fingerprint drift across tolerances: {fingerprints}"
        )
    expected_20 = data.get("expected_20m_rolling_counts")
    if expected_20:
        summary_by_key = {
            (
                int(row["tolerance_minutes"]),
                str(row["fold"]),
                str(row["partition"]),
            ): row
            for row in summaries
        }
        for fold_id, raw_partitions in _mapping(
            expected_20, "expected 20m rolling counts"
        ).items():
            for partition, raw_counts in _mapping(
                raw_partitions, f"20m counts {fold_id}"
            ).items():
                expected = _mapping(raw_counts, f"20m counts {fold_id}/{partition}")
                observed = summary_by_key[(20, str(fold_id), str(partition))]
                if int(observed["pair_count"]) != int(expected["pairs"]) or int(
                    observed["session_count"]
                ) != int(expected["sessions"]):
                    raise ValueError(f"20m rolling count drift: {fold_id}/{partition}")
    if config["line"] == "window":
        universe_frame = pd.DataFrame(rows)
        for fold_id in FOLDS:
            for partition in ("train", "validation", "test"):
                nested_sets = {
                    tolerance: set(
                        universe_frame.loc[
                            universe_frame["tolerance_minutes"]
                            .astype(int)
                            .eq(tolerance)
                            & universe_frame["fold"].astype(str).eq(fold_id)
                            & universe_frame["partition"].astype(str).eq(partition),
                            "pair_id",
                        ].astype(str)
                    )
                    for tolerance in WINDOW_TOLERANCES
                }
                for lower, upper in _adjacent_window_tolerance_pairs():
                    if not nested_sets[lower].issubset(nested_sets[upper]):
                        raise ValueError(
                            f"Alignment nesting drift: {fold_id}/{partition}/{lower}m⊄{upper}m"
                        )
    universe_path = root / "inputs/pair_universes.csv"
    summary_path = root / "inputs/pair_universe_summary.csv"
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(universe_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    return _write_signed(
        root / "inputs/pair_universe_manifest.json",
        {
            "schema_version": 1,
            "kind": PAIR_UNIVERSE_KIND,
            "grid_fingerprint": next(iter(fingerprints)),
            "target_horizon_minutes": 5,
            "target_surface_values_read": False,
            "test_error_metrics_read": False,
            "pair_window_relation_aggregation": (
                "maximum_post_news_overlap_across_aligned_articles_conservative_v1"
            ),
            "pair_window_relation_source": (
                "recomputed_from_news_available_and_current_window_timestamps_v1"
            ),
            "pair_window_article_deduplication": (
                "normalized_article_key_then_minimum_news_row_v1"
            ),
            "pair_window_relation_audits": relation_audits,
            "universe_path": str(universe_path.resolve()),
            "universe_sha256": _sha256_file(universe_path),
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": _sha256_file(summary_path),
            "row_count": len(rows),
        },
    )


def _parse_lp_vector(value: object, *, label: str) -> np.ndarray:
    from wgan_option.utils.news_first_experiment_core import _parsed_vector

    return _parsed_vector(value, dimension=1024, label=label)


def _lp_vectors_for_pairs(
    workbook: Path, *, sheet_name: str, required_pairs: set[str]
) -> dict[str, np.ndarray]:
    from wgan_option.utils.news_first_experiment_core import (
        _l2,
        _normalized_article_key,
    )

    frame = pd.read_excel(
        workbook,
        sheet_name=sheet_name,
        usecols=["pair_id", "article_id", "sample_id", "news_row_id", "lp_embedding"],
    )
    frame["pair_id"] = frame["pair_id"].astype(str)
    frame = frame.loc[frame["pair_id"].isin(required_pairs)].copy()
    if set(frame["pair_id"]) != required_pairs:
        raise ValueError(f"LP workbook coverage drift: {workbook}")
    result: dict[str, np.ndarray] = {}
    for pair_id, group in frame.groupby("pair_id", sort=False):
        ordered = group.assign(
            _news_sort=pd.to_numeric(group["news_row_id"], errors="raise")
        ).sort_values(["_news_sort", "sample_id"], kind="stable")
        articles: dict[str, np.ndarray] = {}
        for row in ordered.itertuples(index=False):
            key = _normalized_article_key(row)
            vector = _parse_lp_vector(
                row.lp_embedding, label=f"LP pair={pair_id}, article={key}"
            )
            previous = articles.get(key)
            if previous is not None:
                if not np.array_equal(previous, vector):
                    # Match the established deterministic minimum-news-row rule.
                    continue
                continue
            articles[key] = vector
        if not articles:
            raise ValueError(f"No LP articles for pair {pair_id}")
        result[str(pair_id)] = _l2(np.mean(np.stack(list(articles.values())), axis=0))
    return result


def _materialize_development_overlays(config: Mapping[str, Any], root: Path) -> Path:
    from wgan_option.utils.news_first_experiment_core import (
        write_pair_text_overlay_manifest,
    )

    universe = pd.read_csv(root / "inputs/pair_universes.csv", dtype=str)
    data = _mapping(config["data"], "data")
    dataset_root = _resolve(str(data["root"]))
    rows: list[dict[str, Any]] = []
    for tolerance in map(int, data["tolerances_minutes"]):
        selected_tolerance = universe.loc[
            universe["tolerance_minutes"].astype(int).eq(tolerance)
            & universe["partition"].isin(("train", "validation"))
        ]
        required = set(selected_tolerance["pair_id"].astype(str))
        workbook = dataset_root / str(data["workbook_template"]).format(
            tolerance02=f"{tolerance:02d}"
        )
        vectors = _lp_vectors_for_pairs(
            workbook,
            sheet_name=str(data.get("sheet_name", "gan_input_ready")),
            required_pairs=required,
        )
        for fold in FOLDS:
            fold_rows = selected_tolerance.loc[selected_tolerance["fold"].eq(fold)]
            pair_ids = sorted(fold_rows["pair_id"].astype(str))
            sessions = dict(
                zip(
                    fold_rows["pair_id"].astype(str),
                    fold_rows["session_id"].astype(str),
                )
            )
            arms = (
                ("lp_matched",)
                if config["line"] == "architecture"
                else ("film_lp_matched", "pure_cnn_no_text")
            )
            for arm in arms:
                is_zero = arm == "pure_cnn_no_text"
                records = [
                    {
                        "pair_id": pair_id,
                        "session_id": sessions[pair_id],
                        "embedding": (
                            np.zeros(1024, dtype=np.float32)
                            if is_zero
                            else vectors[pair_id]
                        ),
                    }
                    for pair_id in pair_ids
                ]
                path = (
                    root
                    / "inputs/pair_text_overlays"
                    / f"tolerance_{tolerance:02d}m"
                    / fold
                    / f"{arm}.json"
                )
                write_pair_text_overlay_manifest(
                    path,
                    mode="current_only" if is_zero else "lp_mean_l2",
                    namespace=f"development/{tolerance:02d}m/{fold}/{arm}",
                    records=records,
                    transform={
                        "method": (
                            "zero_vector_v1"
                            if is_zero
                            else "unique_article_lp_mean_l2_v1"
                        ),
                        "partition_scope": "train_and_validation_only",
                    },
                )
                rows.append(
                    {
                        "artifact_role": f"development_overlay:{tolerance:02d}m:{fold}:{arm}",
                        "path": str(path.resolve()),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
    manifest = root / "inputs/development_overlay_hashes.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _registry_path(root: Path) -> Path:
    return root / "registry/task_registry.json"


def _status_path(root: Path, job_id: str) -> Path:
    return root / "registry/job_status" / f"{job_id}.json"


def _atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(payload), handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise
    return path


def _baseline_training_config(
    config: Mapping[str, Any], *, baseline_name: str, seed: int, fold: str
) -> tuple[Path, dict[str, Any]]:
    baseline = _mapping(
        _mapping(config["baselines"], "baselines")[baseline_name],
        f"baseline {baseline_name}",
    )
    source_arm = str(baseline["arm"])
    path = (
        _resolve(str(baseline["root"]))
        / "configs/jobs"
        / f"direct_arms_05m_{fold}_seed_{int(seed)}_{source_arm}.yaml"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Baseline training template is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return path, _mapping(payload, str(path))


def _resolved_job_conditioning_learning_rate(
    root: Path, job: Mapping[str, Any]
) -> float | None:
    value = job.get("conditioning_learning_rate")
    if value is not None:
        return float(value)
    if job["stage"] == "formal":
        selection = _read_signed(
            root / "registry/screen_selection.json",
            kind="architecture_screen_selection_v1",
        )
        return float(
            selection["selected_by_mode"][str(job["generator_mode"])][
                "conditioning_learning_rate"
            ]
        )
    if job["stage"] == "scaled":
        if str(job["variant"]) == "film_reference_scaled":
            config = yaml.safe_load(
                (root / "resolved_config.yaml").read_text(encoding="utf-8")
            )
            scaled = _mapping(_mapping(config["matrix"], "matrix")["scaled"], "scaled")
            return float(scaled["conditioning_reference_learning_rate"])
        selection = _read_signed(
            root / "registry/screen_selection.json",
            kind="architecture_screen_selection_v1",
        )
        architecture = _read_signed(
            root / "registry/architecture_selection.json",
            kind="architecture_formal_selection_v1",
        )
        mode = str(architecture["point_leader_mode"])
        return float(selection["selected_by_mode"][mode]["conditioning_learning_rate"])
    return None


def _resolved_job_mode(root: Path, job: Mapping[str, Any]) -> str:
    mode = job.get("generator_mode")
    if mode:
        return str(mode)
    if job["stage"] != "scaled":
        raise ValueError(f"Job has no generator mode: {job['job_id']}")
    selection = _read_signed(
        root / "registry/architecture_selection.json",
        kind="architecture_formal_selection_v1",
    )
    return str(selection["point_leader_mode"])


def _generator_parameter_count(mode: str, profile: Mapping[str, Any]) -> int:
    from wgan_option.models.generator import Generator

    profile = dict(profile)
    base_channels = int(profile["gen_base_channels"])
    mode_kwargs = {
        key.removeprefix("gen_"): value
        for key, value in profile.items()
        if key.startswith("gen_")
        and key
        not in {
            "gen_base_channels",
            "gen_hidden_dim",
        }
    }
    generator = Generator(
        channels=1,
        embedding_dim=1024,
        noise_dim=32,
        surface_height=16,
        surface_width=16,
        base_channels=int(base_channels),
        res_blocks=0,
        text_hidden_dim=256,
        text_out_dim=128,
        hidden_dim=int(profile.get("gen_hidden_dim", base_channels * 32)),
        residual_output_mode="identity_softplus_residual",
        generator_conditioning_mode=str(mode),
        generator_current_input_mode="current_support_masked",
        strike_grid=np.asarray(EXACT_MONEYNESS, dtype=np.float32),
        maturity_grid_days=np.asarray(EXACT_TTM_DAYS, dtype=np.float32),
        **mode_kwargs,
    )
    return sum(int(parameter.numel()) for parameter in generator.parameters())


def core_runtime_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that the branch-local modes and split-LR API are installed.

    The architecture implementations are staged separately while older runs are
    bound to source hashes.  This check therefore deliberately happens before a
    formal output root is created.  It becomes a positive executable contract
    once that staged code is integrated into ``src``.
    """

    errors: list[str] = []
    counts: dict[str, int] = {}
    critic_parameters: int | None = None
    try:
        from wgan_option.config import Config

        fields = set(getattr(Config, "__dataclass_fields__", {}))
        required_fields = {
            "generator_conditioning_learning_rate",
            "generator_conditioning_min_learning_rate",
        }
        if config["line"] == "architecture":
            required_fields.update(
                {
                    "gen_crossattn_heads",
                    "gen_crossattn_text_tokens",
                    "gen_crossattn_dim",
                    "gen_transformer_model_dim",
                    "gen_transformer_layers",
                    "gen_transformer_heads",
                    "gen_transformer_ffn_dim",
                    "gen_transformer_dropout",
                    "gen_style_dim",
                    "gen_style_demodulate",
                }
            )
        missing = sorted(required_fields - fields)
        if missing:
            errors.append("missing Config fields: " + ", ".join(missing))
        else:
            # Construction exercises mode normalization and the optimizer
            # profile validator, rather than merely trusting dataclass fields.
            Config(
                generator_conditioning_mode=(
                    "crossattn_unet_mask_coords_v1"
                    if config["line"] == "architecture"
                    else "film_unet_mask_coords_v1"
                ),
                generator_current_input_mode="current_support_masked",
                support_mask_mode="raw_joint",
                residual_output_mode="identity_softplus_residual",
                generator_optimizer_profile="conditioning_split_lr_v2",
                generator_text_learning_rate=2.5e-6,
                generator_text_min_learning_rate=2.5e-7,
                generator_conditioning_learning_rate=2.5e-5,
                generator_conditioning_min_learning_rate=2.5e-6,
            )
    except Exception as exc:
        errors.append(f"configuration API unavailable: {type(exc).__name__}: {exc}")

    try:
        if config["line"] == "architecture":
            matched_profiles = _compute_matched_capacity_profiles(config)
            counts.update(
                {
                    mode: int(matched_profiles[mode]["generator_parameters"])
                    for mode in ARCHITECTURE_MODES
                }
            )
        else:
            for mode, expected in (
                ("film_unet_mask_coords_v1", 827_745),
                ("cnn_unet_mask_coords_v1", 416_353),
            ):
                profile = {"gen_base_channels": 32, "gen_hidden_dim": 1024}
                observed = _generator_parameter_count(mode, profile)
                counts[mode] = observed
                if observed != expected:
                    errors.append(f"{mode} parameter count {observed} != {expected}")
    except Exception as exc:
        errors.append(f"Generator API unavailable: {type(exc).__name__}: {exc}")

    try:
        from wgan_option.models.discriminator import Discriminator

        critic = Discriminator(
            channels=1,
            embedding_dim=1024,
            base_channels=32,
            res_blocks=0,
            text_hidden_dim=128,
            hidden_dim=786,
            surface_height=16,
            surface_width=16,
            critic_conditioning_mode="lp_disabled_same_shape_v1",
        )
        critic_parameters = sum(
            int(parameter.numel()) for parameter in critic.parameters()
        )
        if critic_parameters != 729_157:
            errors.append(f"NoLP Critic parameter count {critic_parameters} != 729157")
    except Exception as exc:
        errors.append(f"Critic API unavailable: {type(exc).__name__}: {exc}")

    return {
        "status": "passed" if not errors else "blocked",
        "profile": "conditioning_split_lr_v2",
        "generator_parameter_counts": counts,
        "critic_parameters": critic_parameters,
        "errors": errors,
    }


def _grid_values(raw: object, *, label: str) -> list[Any]:
    if isinstance(raw, Mapping):
        values = _mapping(raw, label)
        lower = int(values["minimum"])
        upper = int(values["maximum"])
        if lower < 1 or upper < lower:
            raise ValueError(f"Invalid integer search interval: {label}")
        return list(range(lower, upper + 1))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError(f"Search grid {label} must be non-empty")
    return list(raw)


def _capacity_candidates(mode: str, grid: Mapping[str, Any]) -> list[dict[str, Any]]:
    keys = sorted(grid)
    values = [_grid_values(grid[key], label=f"{mode}.{key}") for key in keys]
    candidates: list[dict[str, Any]] = []
    for combination in itertools.product(*values):
        profile = dict(zip(keys, combination, strict=True))
        profile.setdefault("gen_hidden_dim", int(profile["gen_base_channels"]) * 32)
        try:
            count = _generator_parameter_count(mode, profile)
        except (AssertionError, ValueError):
            # Invalid divisibility/head combinations are excluded by construction.
            continue
        candidates.append({**profile, "generator_parameters": count})
    if not candidates:
        raise ValueError(f"No executable capacity candidates for {mode}")
    return candidates


def _freeze_scaled_capacity_profiles(
    config: Mapping[str, Any], root: Path, *, point_leader_mode: str
) -> Path:
    scaled = _mapping(_mapping(config["matrix"], "matrix")["scaled"], "scaled")
    target = int(scaled["target_generator_parameters"])
    maximum_deviation = float(scaled["maximum_relative_deviation"])
    search = _mapping(scaled["deterministic_search_grid"], "deterministic search grid")
    variants = {
        "validation_point_leader_scaled": str(point_leader_mode),
        "film_reference_scaled": "film_unet_mask_coords_v1",
    }
    selected: dict[str, dict[str, Any]] = {}
    for variant, mode in variants.items():
        winner = _selected_capacity_candidate(
            mode=mode,
            target=target,
            maximum_deviation=maximum_deviation,
            search_grid=_mapping(search[mode], f"search grid {mode}"),
        )
        profile = {
            "variant": variant,
            "generator_mode": mode,
            "target_generator_parameters": target,
            **winner,
            "capacity_profile": (
                f"auto_budget1655490_{mode}_c{winner['gen_base_channels']}"
            ),
        }
        profile["profile_sha256"] = _payload_sha256(profile)
        selected[variant] = profile
    path = root / "registry/scaled_capacity_profiles.json"
    return _write_signed(
        path,
        {
            "schema_version": 1,
            "kind": "architecture_scaled_capacity_profiles_v1",
            "selection_rule": scaled["profile_selection"],
            "maximum_relative_deviation": maximum_deviation,
            "profiles": selected,
            "frozen_at_utc": utc_now(),
        },
    )


def _selected_capacity_candidate(
    *,
    mode: str,
    target: int,
    maximum_deviation: float,
    search_grid: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = [
        {
            **candidate,
            "absolute_deviation": abs(int(candidate["generator_parameters"]) - target),
            "relative_deviation": abs(int(candidate["generator_parameters"]) - target)
            / target,
        }
        for candidate in _capacity_candidates(mode, search_grid)
    ]
    winner = min(
        candidates,
        key=lambda row: (
            row["absolute_deviation"],
            int(row["gen_base_channels"]),
            _payload_sha256(row),
        ),
    )
    if float(winner["relative_deviation"]) > maximum_deviation:
        raise ValueError(
            f"No executable {mode} capacity is within {maximum_deviation:.1%} "
            f"of {target:,} parameters; nearest={winner}"
        )
    return winner


def _matched_capacity_contract(
    config: Mapping[str, Any],
) -> tuple[int, float, str, dict[str, Any], dict[str, int]]:
    matched = _mapping(
        _mapping(config["matrix"], "matrix")["matched_capacity"],
        "matched capacity",
    )
    search = _mapping(
        matched["deterministic_search_grid"], "matched deterministic search grid"
    )
    expected = {
        str(mode): int(value)
        for mode, value in _mapping(
            matched["expected_winner_generator_parameters"],
            "matched expected winner parameters",
        ).items()
    }
    return (
        int(matched["target_generator_parameters"]),
        float(matched["maximum_relative_deviation"]),
        str(matched["profile_selection"]),
        search,
        expected,
    )


def _compute_matched_capacity_profiles(
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate the configured grid and return its deterministic winners."""

    target, maximum_deviation, selection_rule, search, expected = (
        _matched_capacity_contract(config)
    )
    profiles: dict[str, dict[str, Any]] = {}
    for mode in ARCHITECTURE_MODES:
        winner = _selected_capacity_candidate(
            mode=mode,
            target=target,
            maximum_deviation=maximum_deviation,
            search_grid=_mapping(search[mode], f"matched search grid {mode}"),
        )
        actual = int(winner["generator_parameters"])
        if actual != int(expected[mode]):
            raise ValueError(
                f"Matched-capacity winner drift for {mode}: "
                f"{actual} != {int(expected[mode])}"
            )
        profile = {
            "generator_mode": mode,
            "capacity_profile": _matched_capacity_profile_name(mode),
            "target_generator_parameters": target,
            "maximum_relative_deviation": maximum_deviation,
            "selection_rule": selection_rule,
            "search_grid_sha256": _payload_sha256(search[mode]),
            **winner,
            "actual_generator_parameters": actual,
        }
        profile["profile_sha256"] = _payload_sha256(profile)
        profiles[mode] = profile
    return profiles


def _matched_capacity_artifact_payload(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    target, maximum_deviation, selection_rule, search, expected = (
        _matched_capacity_contract(config)
    )
    return {
        "schema_version": 1,
        "kind": MATCHED_CAPACITY_KIND,
        "target_generator_parameters": target,
        "maximum_relative_deviation": maximum_deviation,
        "selection_rule": selection_rule,
        "expected_winner_generator_parameters": expected,
        "deterministic_search_grid": deepcopy(search),
        "search_grid_sha256": _payload_sha256(search),
        "profiles": _compute_matched_capacity_profiles(config),
    }


def _freeze_matched_capacity_profiles(config: Mapping[str, Any], root: Path) -> Path:
    path = root / "registry/matched_capacity_profiles.json"
    return _write_signed(
        path,
        {
            **_matched_capacity_artifact_payload(config),
            "frozen_at_utc": utc_now(),
        },
    )


def _verify_matched_capacity_profiles(
    root: Path,
    registry: Mapping[str, Any],
    *,
    recompute_winners: bool = True,
) -> dict[str, Any]:
    expected_path = (root / "registry/matched_capacity_profiles.json").resolve()
    path = Path(str(registry.get("matched_capacity_profiles_path", ""))).resolve()
    if path != expected_path:
        raise ValueError("Matched-capacity artifact path drift")
    _verify_file(
        path,
        str(registry.get("matched_capacity_profiles_sha256", "")),
        "matched-capacity profiles",
    )
    artifact = _read_signed(path, kind=MATCHED_CAPACITY_KIND)
    if int(artifact.get("schema_version", -1)) != 1:
        raise ValueError("Matched-capacity artifact schema drift")
    frozen_config = _mapping(
        yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8")),
        "resolved config",
    )
    target, maximum_deviation, selection_rule, search, expected = (
        _matched_capacity_contract(frozen_config)
    )
    expected_top = {
        "target_generator_parameters": target,
        "maximum_relative_deviation": maximum_deviation,
        "selection_rule": selection_rule,
        "expected_winner_generator_parameters": expected,
        "deterministic_search_grid": search,
        "search_grid_sha256": _payload_sha256(search),
    }
    if any(artifact.get(key) != value for key, value in expected_top.items()):
        raise ValueError("Matched-capacity search contract drift")
    profiles = _mapping(artifact.get("profiles"), "matched-capacity profiles")
    if set(profiles) != set(ARCHITECTURE_MODES):
        raise ValueError("Matched-capacity winner universe drift")
    for mode in ARCHITECTURE_MODES:
        profile = _mapping(profiles[mode], f"matched-capacity winner {mode}")
        unsigned = {
            key: value for key, value in profile.items() if key != "profile_sha256"
        }
        if profile.get("profile_sha256") != _payload_sha256(unsigned):
            raise ValueError(f"Matched-capacity profile SHA drift: {mode}")
        if int(profile.get("generator_parameters", -1)) != int(
            profile.get("actual_generator_parameters", -2)
        ):
            raise ValueError(f"Matched-capacity actual-count drift: {mode}")
    if recompute_winners:
        expected_profiles = _compute_matched_capacity_profiles(frozen_config)
        if profiles != expected_profiles:
            raise ValueError("Frozen matched-capacity winners are not grid optima")
    return artifact


def _job_capacity(root: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    if job["stage"] != "scaled":
        registry = _load_registry(root)
        if registry["line"] == "window":
            profile = {
                "capacity_profile": "c32",
                "gen_base_channels": 32,
                "gen_hidden_dim": 1024,
            }
            mode = _resolved_job_mode(root, job)
            actual = _generator_parameter_count(mode, profile)
            expected = {
                "film_unet_mask_coords_v1": 827_745,
                "cnn_unet_mask_coords_v1": 416_353,
            }.get(mode)
            if expected is None or actual != expected:
                raise ValueError(
                    f"Window c32 Generator parameter drift for {mode}: "
                    f"{actual} != {expected}"
                )
            return {
                **profile,
                "generator_parameters": actual,
            }
        matched = _verify_matched_capacity_profiles(
            root, registry, recompute_winners=False
        )
        profiles = _mapping(matched["profiles"], "matched-capacity profiles")
        mode = _resolved_job_mode(root, job)
        profile = _mapping(profiles[mode], "matched architecture profile")
        actual = _generator_parameter_count(mode, profile)
        if (
            actual != int(profile["actual_generator_parameters"])
            or actual != MATCHED_GENERATOR_PARAMETER_COUNTS[mode]
            or str(job.get("capacity_profile")) != str(profile["capacity_profile"])
        ):
            raise ValueError(
                f"Matched profile parameter drift for {mode}: "
                f"actual={actual}, frozen={profile['actual_generator_parameters']}, "
                f"job_profile={job.get('capacity_profile')}"
            )
        return dict(profile)
    profiles = _read_signed(
        root / "registry/scaled_capacity_profiles.json",
        kind="architecture_scaled_capacity_profiles_v1",
    )
    return dict(profiles["profiles"][str(job["variant"])])


def _materialized_training_payload(
    config: Mapping[str, Any], root: Path, job: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    is_pure = str(job["arm"]) == "pure_cnn_no_text"
    baseline_name = (
        "pure_cnn_no_text"
        if config["line"] == "window" and is_pure
        else "film_lp_matched"
        if config["line"] == "window"
        else "film_reference"
    )
    template_path, payload = _baseline_training_config(
        config,
        baseline_name=baseline_name,
        seed=int(job["seed"]),
        fold=str(job["fold"]),
    )
    payload = deepcopy(payload)
    data = _mapping(config["data"], "data")
    training = _mapping(config["training"], "training")
    tolerance = int(job["tolerance_minutes"])
    fold = _fold(config, str(job["fold"]))
    mode = _resolved_job_mode(root, job)
    capacity = _job_capacity(root, job)
    conditioning_lr = _resolved_job_conditioning_learning_rate(root, job)
    dataset = _resolve(str(data["root"])) / str(data["workbook_template"]).format(
        tolerance02=f"{tolerance:02d}"
    )
    overlay_arm = (
        "pure_cnn_no_text"
        if is_pure
        else ("film_lp_matched" if config["line"] == "window" else "lp_matched")
    )
    overlay = (
        root
        / "inputs/pair_text_overlays"
        / f"tolerance_{tolerance:02d}m"
        / str(job["fold"])
        / f"{overlay_arm}.json"
    )
    overlay_payload = _read_json(overlay)
    epochs = int(training["num_epochs"])
    if job["stage"] == "screen":
        epochs = int(
            _mapping(_mapping(config["matrix"], "matrix")["screen"], "screen")[
                "num_epochs"
            ]
        )
    payload.update(
        data_path=str(dataset.resolve()),
        news_first_common_eval_data_path=str(dataset.resolve()),
        news_first_train_end_utc=str(fold["train_end_utc"]),
        news_first_validation_end_utc=str(fold["validation_end_utc"]),
        news_first_data_window_end_utc_exclusive=str(fold["validation_end_utc"]),
        news_first_dataset_tolerance_minutes=tolerance,
        news_first_text_ablation_mode="real_text",
        news_first_text_information_path=str(job["arm"]),
        news_first_pair_text_overlay_mode=("current_only" if is_pure else "lp_mean_l2"),
        news_first_pair_text_manifest_path=str(overlay.resolve()),
        news_first_pair_text_manifest_sha256=_sha256_file(overlay),
        news_first_pair_text_profile_sha256=str(overlay_payload["profile_sha256"]),
        news_first_full_training_state_mode="none",
        news_first_full_training_state_contract_path="",
        news_first_full_training_state_contract_sha256="",
        news_first_graft_state_path="",
        news_first_graft_state_sha256="",
        news_first_materialize_validation_loader=True,
        news_first_materialize_test_loader=False,
        news_first_refit_mode="none",
        news_first_refit_recipe_path="",
        news_first_refit_recipe_sha256="",
        news_first_capacity_profile=str(capacity["capacity_profile"]),
        news_first_capacity_profile_sha256=str(
            capacity.get("profile_sha256") or _payload_sha256(capacity)
        ),
        news_first_architecture_profile_sha256=_payload_sha256(
            {"mode": mode, "capacity": capacity}
        ),
        news_first_model_contract_sha256=_payload_sha256(
            {
                "generator_mode": mode,
                "critic_mode": training["critic_conditioning_mode"],
                "capacity": capacity,
            }
        ),
        generator_conditioning_mode=mode,
        critic_conditioning_mode=str(training["critic_conditioning_mode"]),
        generator_noise_mode="gaussian",
        noise_dim=32,
        residual_output_mode="identity_softplus_residual",
        support_mask_mode="raw_joint",
        generator_current_input_mode="current_support_masked",
        gen_base_channels=int(capacity["gen_base_channels"]),
        gen_hidden_dim=int(capacity["gen_hidden_dim"]),
        generator_learning_rate=float(training["backbone_learning_rate"]),
        discriminator_learning_rate=float(training["critic_learning_rate"]),
        learning_rate=float(training["backbone_learning_rate"]),
        reduce_lr_min_lr=float(training["backbone_learning_rate"]) * 0.1,
        num_epochs=epochs,
        early_stopping_min_epochs=(
            epochs
            if job["stage"] == "screen"
            else int(training["early_stopping_min_epochs"])
        ),
        early_stopping_patience=(
            epochs
            if job["stage"] == "screen"
            else int(training["early_stopping_patience"])
        ),
        batch_size=int(training["batch_size"]),
        discriminator_iter=int(training["discriminator_steps"]),
        validation_mc_samples=int(training["validation_mc_samples"]),
        seed=int(job["seed"]),
        output_root=str(
            (
                root
                / "runs"
                / str(job["stage"])
                / f"tolerance_{tolerance:02d}m"
                / str(job["fold"])
                / f"seed_{int(job['seed'])}"
                / str(job["job_id"])
            ).resolve()
        ),
        save_every=1_000_000,
        lr_warmup_epochs=0,
    )
    if is_pure:
        payload.update(
            generator_optimizer_profile="uniform_v1",
            generator_text_learning_rate=0.0,
            generator_film_learning_rate=0.0,
            generator_text_min_learning_rate=0.0,
            generator_film_min_learning_rate=0.0,
            generator_conditioning_learning_rate=0.0,
            generator_conditioning_min_learning_rate=0.0,
        )
    else:
        if conditioning_lr is None or conditioning_lr <= 0:
            raise ValueError(
                f"Conditional job lacks positive conditioning LR: {job['job_id']}"
            )
        payload.update(
            generator_optimizer_profile="conditioning_split_lr_v2",
            generator_text_learning_rate=float(training["text_learning_rate"]),
            generator_film_learning_rate=0.0,
            generator_text_min_learning_rate=float(training["text_learning_rate"])
            * 0.1,
            generator_film_min_learning_rate=0.0,
            generator_conditioning_learning_rate=float(conditioning_lr),
            generator_conditioning_min_learning_rate=float(conditioning_lr) * 0.1,
        )
    for key, value in capacity.items():
        if key.startswith("gen_") and key not in {
            "gen_base_channels",
            "gen_hidden_dim",
        }:
            payload[key] = value
    return payload, template_path


def materialize(
    config_path: str | Path, output_dir: str | Path, *, stage: str | None = None
) -> Path:
    """Materialize immutable ``train_vol.py`` configs for currently unblocked jobs."""

    config = load_config(config_path)
    root = _resolve(output_dir)
    registry = _load_registry(root)
    _verify_prepare_lineage(root, registry)
    if registry["source_config_sha256"] != config["source_config_sha256"]:
        raise ValueError("Materialization source config drift")
    existing_path = root / "registry/materialized_worker_specs.json"
    existing: dict[str, dict[str, Any]] = {}
    if existing_path.is_file():
        prior = _read_signed(existing_path, kind=WORKER_SPEC_KIND)
        existing = {str(row["job_id"]): dict(row) for row in prior.get("jobs", [])}
    for job in registry["jobs"]:
        if stage is not None and str(job["stage"]) != str(stage):
            continue
        state = _read_json(_status_path(root, str(job["job_id"])))
        if str(state.get("status")) not in {"pending", "failed", "completed"}:
            continue
        payload, template_path = _materialized_training_payload(config, root, job)
        path = root / "configs/jobs" / f"{job['job_id']}.yaml"
        if path.is_file():
            observed = _mapping(
                yaml.safe_load(path.read_text(encoding="utf-8")), str(path)
            )
            if observed != payload:
                raise ValueError(f"Materialized training config drift: {job['job_id']}")
        else:
            _atomic_write_yaml(path, payload)
        spec = {
            "job_id": str(job["job_id"]),
            "job_spec_sha256": str(job["job_spec_sha256"]),
            "training_config_path": str(path.resolve()),
            "training_config_sha256": _sha256_file(path),
            "template_path": str(template_path.resolve()),
            "template_sha256": _sha256_file(template_path),
            "generator_mode": str(payload["generator_conditioning_mode"]),
            "conditioning_learning_rate": float(
                payload.get("generator_conditioning_learning_rate", 0.0)
            ),
            "capacity_profile": str(payload["news_first_capacity_profile"]),
            "expected_generator_parameters": int(
                _job_capacity(root, job)["generator_parameters"]
            ),
        }
        if str(job["job_id"]) in existing and existing[str(job["job_id"])] != spec:
            raise ValueError(f"Materialized worker spec drift: {job['job_id']}")
        existing[str(job["job_id"])] = spec
    return _write_signed(
        existing_path,
        {
            "schema_version": 1,
            "kind": WORKER_SPEC_KIND,
            "study_kind": registry["study_kind"],
            "jobs": sorted(existing.values(), key=lambda row: row["job_id"]),
            "updated_at_utc": utc_now(),
        },
    )


def _write_commands(
    root: Path, config: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]
) -> Path:
    path = root / "registry/worker_commands.jsonl"
    lines = []
    for job in jobs:
        command = [
            str(_mapping(config["runtime"], "runtime")["python_executable"]),
            "-m",
            "scripts.rq3.news_first_vol_architecture_window_study",
            "worker",
            "--config",
            str(config["source_config_path"]),
            "--output-dir",
            str(root),
            "--job-id",
            str(job["job_id"]),
        ]
        lines.append(
            json.dumps(
                {"job_id": job["job_id"], "gpu_id": job["gpu_id"], "command": command},
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _source_code_bindings(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = {
        Path(__file__).resolve(),
        (
            Path(__file__).with_name(
                "news_first_vol_architecture_window_study_analysis.py"
            )
        ).resolve(),
        Path(__file__)
        .with_name("news_first_vol_experiment_supervisor_helpers.py")
        .resolve(),
        Path(__file__).with_name("news_first_vol_comparison_analysis.py").resolve(),
        (REPO_ROOT / "scripts/rq123/news_first_vol_film_nolp_10seed.py").resolve(),
        _resolve(str(_mapping(config["runtime"], "runtime")["training_entrypoint"])),
    }
    paths.update(path.resolve() for path in (REPO_ROOT / "src").rglob("*.py"))
    paths.update(path.resolve() for path in (REPO_ROOT / "scripts/train").rglob("*.py"))
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"Source-code lineage file missing: {missing}")
    return [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(paths, key=str)
    ]


def _verify_prepare_lineage(root: Path, registry: Mapping[str, Any]) -> None:
    _verify_file(
        root / "resolved_config.yaml",
        str(registry["resolved_config_sha256"]),
        "resolved config",
    )
    _verify_file(
        root / "registry/input_preflight.json",
        str(registry["input_preflight_sha256"]),
        "input preflight",
    )
    input_preflight = _read_json(root / "registry/input_preflight.json")
    for row in input_preflight.get("rows", []):
        for role in ("workbook", "support", "validation"):
            path = Path(str(row[f"{role}_path"])).resolve()
            if (
                not path.is_file()
                or path.stat().st_size != int(row[f"{role}_size_bytes"])
                or _sha256_file(path) != str(row[f"{role}_sha256"])
            ):
                raise ValueError(f"Prepared dataset input drift: {path}")
    for row in input_preflight.get("dataset_lineage_inputs", []):
        path = Path(str(row["path"])).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or _sha256_file(path) != str(row["sha256"])
        ):
            raise ValueError(f"Prepared dataset lineage drift: {path}")
    equivalence = input_preflight.get("reused_5m_semantic_equivalence")
    if equivalence is not None:
        for row in _mapping(equivalence, "reused 5m equivalence").get("inputs", []):
            path = Path(str(row["path"])).resolve()
            if (
                not path.is_file()
                or path.stat().st_size != int(row["size_bytes"])
                or _sha256_file(path) != str(row["sha256"])
            ):
                raise ValueError(f"Reused 5m equivalence input drift: {path}")
    output_manifest_value = str(
        input_preflight.get("output_sha_manifest_path", "") or ""
    )
    if output_manifest_value:
        output_manifest = Path(output_manifest_value).resolve()
        _verify_file(
            output_manifest,
            str(input_preflight["output_sha_manifest_sha256"]),
            "dataset output SHA manifest",
        )
        for line in output_manifest.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split(maxsplit=1)
            _verify_file(
                output_manifest.parent / relative.strip(),
                digest,
                "dataset manifest member",
            )
    _verify_file(
        root / "registry/core_runtime_preflight.json",
        str(registry["core_runtime_preflight_sha256"]),
        "core runtime preflight",
    )
    if registry.get("line") == "architecture":
        _verify_matched_capacity_profiles(root, registry, recompute_winners=True)
    if registry.get("benchmark_binding_sha256") != "benchmark_workspace_bypass":
        _verify_file(
            root / "control/benchmark.json",
            str(registry["benchmark_binding_sha256"]),
            "GPU benchmark binding",
        )
        external_benchmark = root.with_name(root.name + "_control") / "benchmark.json"
        _verify_file(
            external_benchmark,
            str(registry["benchmark_binding_sha256"]),
            "external GPU benchmark binding",
        )
    _verify_file(
        Path(str(registry["worker_commands_path"])),
        str(registry["worker_commands_sha256"]),
        "worker commands",
    )
    pair_manifest_path = Path(str(registry["pair_universe_manifest_path"]))
    _verify_file(
        pair_manifest_path,
        str(registry["pair_universe_manifest_sha256"]),
        "pair-universe manifest",
    )
    pair_manifest = _read_signed(pair_manifest_path, kind=PAIR_UNIVERSE_KIND)
    _verify_file(
        Path(str(pair_manifest["universe_path"])),
        str(pair_manifest["universe_sha256"]),
        "pair universe",
    )
    _verify_file(
        Path(str(pair_manifest["summary_path"])),
        str(pair_manifest["summary_sha256"]),
        "pair-universe summary",
    )
    overlay_manifest_path = Path(str(registry["development_overlay_manifest_path"]))
    _verify_file(
        overlay_manifest_path,
        str(registry["development_overlay_manifest_sha256"]),
        "development overlay manifest",
    )
    overlay_rows = pd.read_csv(overlay_manifest_path)
    for row in overlay_rows.to_dict(orient="records"):
        path = Path(str(row["path"])).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or _sha256_file(path) != str(row["sha256"])
        ):
            raise ValueError(f"Development overlay drift: {path}")
    for row in registry.get("source_code_bindings", []):
        path = Path(str(row["path"])).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or _sha256_file(path) != str(row["sha256"])
        ):
            raise ValueError(f"Source-code lineage drift: {path}")


def prepare(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    resume: bool = False,
    _skip_benchmark_gate: bool = False,
) -> Path:
    config = load_config(config_path)
    root = _resolve(output_dir or str(config["output_root"]))
    core_preflight = core_runtime_preflight(config)
    if core_preflight["status"] != "passed":
        raise RuntimeError(
            "Formal root was not created because the branch-local Generator/"
            "conditioning_split_lr_v2 API is not installed: "
            + "; ".join(core_preflight["errors"])
        )
    benchmark_binding: dict[str, Any] | None = None
    if not _skip_benchmark_gate:
        external_control = root.with_name(root.name + "_control")
        benchmark_path = external_control / "benchmark.json"
        benchmark_binding = _read_signed(benchmark_path, kind=GPU_BENCHMARK_KIND)
        if (
            benchmark_binding.get("status") != "passed"
            or benchmark_binding.get("source_config_sha256")
            != config["source_config_sha256"]
            or Path(str(benchmark_binding.get("formal_output_root", ""))).resolve()
            != root
        ):
            raise RuntimeError(
                "Formal prepare requires its frozen external GPU benchmark"
            )
    preflight = input_preflight(config)
    if preflight["status"] != "passed":
        raise FileNotFoundError(
            "Required tolerance data are missing: " + ", ".join(preflight["missing"])
        )
    plan = planned_cells(config)
    bindings = bind_baselines(config)
    registry_path = _registry_path(root)
    if registry_path.is_file():
        if not resume:
            raise FileExistsError(f"Experiment exists; pass --resume: {root}")
        registry = _read_signed(registry_path, kind=REGISTRY_KIND)
        if registry.get("source_config_sha256") != config["source_config_sha256"]:
            raise ValueError("Source config drift; resume refused")
        if registry.get("jobs_sha256") != _payload_sha256(plan["new_training_jobs"]):
            raise ValueError("Training registry drift; resume refused")
        bind_path = root / "registry/baseline_bindings.json"
        if (
            not bind_path.is_file()
            or _read_signed(bind_path, kind=BASELINE_KIND)["checkpoint_rows"]
            != bindings["checkpoint_rows"]
        ):
            raise ValueError("Baseline bindings drift; resume refused")
        _verify_prepare_lineage(root, registry)
        return root

    root.mkdir(parents=True, exist_ok=False)
    for directory in (
        "analysis",
        "configs/jobs",
        "control",
        "evaluation",
        "inputs/pair_text_overlays",
        "logs",
        "predictions",
        "registry/job_status",
        "registry/job_locks",
        "report",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    frozen_config = {
        key: value
        for key, value in config.items()
        if key not in {"source_config_path", "source_config_sha256"}
    }
    frozen_config["source_config_path"] = config["source_config_path"]
    frozen_config["source_config_sha256"] = config["source_config_sha256"]
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(frozen_config, sort_keys=False), encoding="utf-8"
    )
    matched_capacity_path = (
        _freeze_matched_capacity_profiles(config, root)
        if config["line"] == "architecture"
        else None
    )
    _write_signed(root / "registry/baseline_bindings.json", bindings)
    atomic_write_json(root / "registry/input_preflight.json", preflight)
    atomic_write_json(root / "registry/core_runtime_preflight.json", core_preflight)
    input_preflight_sha = _sha256_file(root / "registry/input_preflight.json")
    core_preflight_sha = _sha256_file(root / "registry/core_runtime_preflight.json")
    if benchmark_binding is not None:
        _write_signed(
            root / "control/benchmark.json",
            {
                key: value
                for key, value in benchmark_binding.items()
                if key != "payload_sha256"
            },
        )
    pair_universe_manifest = _materialize_pair_universe(config, root)
    development_overlay_manifest = _materialize_development_overlays(config, root)
    for job in plan["new_training_jobs"]:
        atomic_write_json(
            _status_path(root, str(job["job_id"])),
            {
                "schema_version": 1,
                "job_id": job["job_id"],
                "job_spec_sha256": job["job_spec_sha256"],
                "status": job["initial_status"],
                "attempt": 0,
                "updated_at_utc": utc_now(),
            },
        )
    command_path = _write_commands(root, config, plan["new_training_jobs"])
    registry = {
        "schema_version": 1,
        "kind": REGISTRY_KIND,
        "study_kind": config["study_kind"],
        "line": config["line"],
        "interpretation": config["interpretation"],
        "created_at_utc": utc_now(),
        "source_config_path": config["source_config_path"],
        "source_config_sha256": config["source_config_sha256"],
        "resolved_config_sha256": _sha256_file(root / "resolved_config.yaml"),
        "input_preflight_sha256": input_preflight_sha,
        "core_runtime_preflight_sha256": core_preflight_sha,
        "source_code_bindings": _source_code_bindings(config),
        "benchmark_binding_sha256": (
            _sha256_file(root / "control/benchmark.json")
            if benchmark_binding is not None
            else "benchmark_workspace_bypass"
        ),
        "jobs_sha256": _payload_sha256(plan["new_training_jobs"]),
        "reused_cells_sha256": _payload_sha256(plan["reused_baseline_cells"]),
        "new_training_jobs": len(plan["new_training_jobs"]),
        "reused_baseline_cells": len(plan["reused_baseline_cells"]),
        "logical_cells": len(plan["new_training_jobs"])
        + len(plan["reused_baseline_cells"]),
        "jobs": plan["new_training_jobs"],
        "baseline_cells": plan["reused_baseline_cells"],
        "worker_commands_path": str(command_path.resolve()),
        "worker_commands_sha256": _sha256_file(command_path),
        "pair_universe_manifest_path": str(pair_universe_manifest.resolve()),
        "pair_universe_manifest_sha256": _sha256_file(pair_universe_manifest),
        "development_overlay_manifest_path": str(
            development_overlay_manifest.resolve()
        ),
        "development_overlay_manifest_sha256": _sha256_file(
            development_overlay_manifest
        ),
        "screen_selection_frozen": False,
        "architecture_selection_frozen": False,
        "evaluation_frozen": False,
        "test_data_opened": False,
        "predictions_frozen": False,
        "analysis_complete": False,
        "terminal_complete": False,
    }
    if matched_capacity_path is not None:
        registry.update(
            matched_capacity_profiles_path=str(matched_capacity_path.resolve()),
            matched_capacity_profiles_sha256=_sha256_file(matched_capacity_path),
        )
    _write_signed(registry_path, registry)
    materialize(
        config_path,
        root,
        stage="screen" if config["line"] == "architecture" else "window",
    )
    return root


def dry_run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    plan = planned_cells(config)
    preflight = input_preflight(config)
    core_preflight = core_runtime_preflight(config)
    baseline_status = "passed"
    baseline_error = ""
    try:
        bindings = bind_baselines(config)
        bound_checkpoint_rows = len(bindings["checkpoint_rows"])
    except Exception as exc:
        baseline_status = "blocked"
        baseline_error = f"{type(exc).__name__}: {exc}"
        bound_checkpoint_rows = 0
    return {
        "study_kind": config["study_kind"],
        "line": config["line"],
        "new_training_jobs": len(plan["new_training_jobs"]),
        "reused_baseline_cells": len(plan["reused_baseline_cells"]),
        "logical_cells": len(plan["new_training_jobs"])
        + len(plan["reused_baseline_cells"]),
        "stage_counts": dict(
            Counter(row["stage"] for row in plan["new_training_jobs"])
        ),
        "input_preflight": preflight,
        "core_runtime_preflight": core_preflight,
        "baseline_status": baseline_status,
        "baseline_error": baseline_error,
        "bound_checkpoint_rows": bound_checkpoint_rows,
        "formal_root_created": False,
        "test_metric_files_read": 0,
    }


def _load_registry(root: Path) -> dict[str, Any]:
    return _read_signed(_registry_path(root), kind=REGISTRY_KIND)


def _save_registry(root: Path, registry: Mapping[str, Any]) -> None:
    _write_signed(_registry_path(root), registry)


def _job(root: Path, registry: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    rows = [dict(row) for row in registry["jobs"] if str(row["job_id"]) == str(job_id)]
    if len(rows) != 1:
        raise ValueError(f"Unknown or duplicate job: {job_id}")
    return rows[0]


def _verify_artifacts(artifacts: object) -> None:
    if not isinstance(artifacts, Sequence) or not artifacts:
        raise ValueError("Completed job has no artifacts")
    for raw in artifacts:
        row = _mapping(raw, "artifact")
        path = Path(str(row["path"])).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or _sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"Artifact drift: {path}")


def _materialized_worker_spec(root: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    path = root / "registry/materialized_worker_specs.json"
    if not path.is_file():
        raise RuntimeError(
            "Model-specific worker configs are not materialized. Add a signed "
            f"{WORKER_SPEC_KIND} manifest before launching training."
        )
    manifest = _read_signed(path, kind=WORKER_SPEC_KIND)
    if manifest.get("study_kind") != _load_registry(root)["study_kind"]:
        raise ValueError("Worker materialization study drift")
    matches = [
        dict(row)
        for row in manifest.get("jobs", [])
        if row.get("job_id") == job["job_id"]
    ]
    if len(matches) != 1:
        raise ValueError(f"No unique materialized config for {job['job_id']}")
    spec = matches[0]
    path = Path(str(spec["training_config_path"])).resolve()
    _verify_file(
        path, str(spec["training_config_sha256"]), "materialized training config"
    )
    if spec.get("job_spec_sha256") != job["job_spec_sha256"]:
        raise ValueError("Materialized worker job lineage drift")
    training = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
    frozen_config = _mapping(
        yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8")),
        "resolved config",
    )
    contract = _mapping(frozen_config["training"], "training contract")
    expected_text_lr = float(contract.get("text_learning_rate", -1.0))
    expected_mode = _resolved_job_mode(root, job)
    is_pure = str(job["arm"]) == "pure_cnn_no_text"
    expected_lr = _resolved_job_conditioning_learning_rate(root, job)
    expected_epochs = 60 if job["stage"] == "screen" else 240
    expected_min_epochs = expected_epochs if job["stage"] == "screen" else 30
    expected_patience = expected_epochs if job["stage"] == "screen" else 20
    if (
        training.get("generator_conditioning_mode") != expected_mode
        or bool(training.get("news_first_materialize_test_loader"))
        or not bool(training.get("news_first_materialize_validation_loader"))
        or int(training.get("seed", -1)) != int(job["seed"])
        or int(training.get("news_first_dataset_tolerance_minutes", -1))
        != int(job["tolerance_minutes"])
    ):
        raise ValueError(f"Materialized training contract drift: {job['job_id']}")
    exact_contract = {
        "news_first_full_training_state_mode": "none",
        "news_first_full_training_state_contract_path": "",
        "news_first_full_training_state_contract_sha256": "",
        "news_first_refit_mode": "none",
        "news_first_refit_recipe_path": "",
        "news_first_refit_recipe_sha256": "",
        "news_first_graft_state_path": "",
        "news_first_graft_state_sha256": "",
        "generator_noise_mode": "gaussian",
        "noise_dim": 32,
        "residual_output_mode": "identity_softplus_residual",
        "support_mask_mode": "raw_joint",
        "generator_current_input_mode": "current_support_masked",
        "critic_conditioning_mode": "lp_disabled_same_shape_v1",
        "generator_learning_rate": 5.0e-7,
        "discriminator_learning_rate": 5.0e-7,
        "learning_rate": 5.0e-7,
        "reduce_lr_min_lr": 5.0e-8,
        "num_epochs": expected_epochs,
        "early_stopping_min_epochs": expected_min_epochs,
        "early_stopping_patience": expected_patience,
        "batch_size": 16,
        "discriminator_iter": 5,
        "validation_mc_samples": 16,
        "lr_warmup_epochs": 0,
        "use_early_stopping": True,
        "best_checkpoint_metric": "val_hybrid_score",
        "text_embedding_mode": "lp",
    }
    drift = {
        key: (training.get(key), expected)
        for key, expected in exact_contract.items()
        if training.get(key) != expected
    }
    if drift:
        raise ValueError(f"Materialized scientific training contract drift: {drift}")
    if (
        float(contract.get("prediction_mc_samples", -1)) != 64.0
        or float(contract.get("backbone_learning_rate", -1)) != 5.0e-7
        or expected_text_lr != 2.5e-6
        or float(contract.get("critic_learning_rate", -1)) != 5.0e-7
    ):
        raise ValueError("Resolved training LR/MC64 contract drift")
    if is_pure:
        if training.get("generator_optimizer_profile") != "uniform_v1" or any(
            float(training.get(key, 0.0)) != 0.0
            for key in (
                "generator_text_learning_rate",
                "generator_text_min_learning_rate",
                "generator_film_learning_rate",
                "generator_film_min_learning_rate",
                "generator_conditioning_learning_rate",
                "generator_conditioning_min_learning_rate",
            )
        ):
            raise ValueError("Pure-CNN job unexpectedly carries conditioning LR")
    elif (
        training.get("generator_optimizer_profile") != "conditioning_split_lr_v2"
        or expected_lr is None
        or float(training.get("generator_conditioning_learning_rate", -1.0))
        != float(expected_lr)
        or float(training.get("generator_text_learning_rate", -1.0)) <= 0.0
        or float(training.get("generator_text_learning_rate", -1.0)) != expected_text_lr
        or float(training.get("generator_text_min_learning_rate", -1.0))
        != expected_text_lr * 0.1
        or float(training.get("generator_conditioning_min_learning_rate", -1.0))
        != float(expected_lr) * 0.1
    ):
        raise ValueError(f"Conditioning LR contract drift: {job['job_id']}")
    return spec


def _discover_training_artifacts(
    training_config_path: Path,
) -> tuple[str, list[dict[str, Any]], float, float, float, int]:
    config = _mapping(
        yaml.safe_load(training_config_path.read_text(encoding="utf-8")),
        "training config",
    )
    if bool(config.get("news_first_materialize_test_loader")):
        raise ValueError("Training config attempted to materialize a test loader")
    run_root = Path(str(config["output_root"])).resolve()
    candidates = sorted(run_root.glob("*/metrics/best_learned_checkpoint.json"))
    if not candidates:
        raise RuntimeError(f"No completed training run under {run_root}")
    best = candidates[-1]
    run_dir = best.parents[1]
    paths = {
        "best_learned_checkpoint": best,
        "generator_best_learned": run_dir / "checkpoints/generator_best_learned.pt",
        "discriminator_best_learned": run_dir
        / "checkpoints/discriminator_best_learned.pt",
        "training_metrics": run_dir / "metrics/training_metrics.csv",
        "training_resolved_config": run_dir / "metrics/training_resolved_config.yaml",
        "run_log": run_dir / "run.log",
    }
    if not all(path.is_file() for path in paths.values()):
        raise RuntimeError(f"Incomplete training artifacts under {run_dir}")
    rows = [
        {
            "artifact_role": role,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for role, path in paths.items()
    ]
    checkpoint = _read_json(best)
    best_epoch = int(
        checkpoint.get("best_learned_epoch_ge_1", checkpoint.get("best_epoch", -1))
    )
    metrics = pd.read_csv(paths["training_metrics"])
    selected = metrics.loc[
        pd.to_numeric(metrics["epoch"], errors="raise").eq(best_epoch)
    ]
    if len(selected) != 1 or not {"val_recon", "val_current_recon"}.issubset(
        selected.columns
    ):
        raise ValueError(f"Cannot recover selected validation MAE: {run_dir}")
    validation_mae = float(selected.iloc[0]["val_recon"])
    validation_persistence_mae = float(selected.iloc[0]["val_current_recon"])
    if (
        not math.isfinite(validation_mae)
        or validation_mae < 0
        or not math.isfinite(validation_persistence_mae)
        or validation_persistence_mae <= 0
    ):
        raise ValueError(f"Invalid selected validation MAE: {run_dir}")
    validation_log_mae_ratio = math.log(validation_mae / validation_persistence_mae)
    return (
        str(run_dir),
        rows,
        validation_mae,
        validation_persistence_mae,
        validation_log_mae_ratio,
        best_epoch,
    )


def _run_training_subprocess(
    command: Sequence[str], *, env: Mapping[str, str]
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        env=dict(env),
        check=False,
        pass_fds=_inherited_global_slot_descriptors(),
    )


def worker(
    config_path: str | Path, output_dir: str | Path, job_id: str
) -> dict[str, Any]:
    config = load_config(config_path)
    root = _resolve(output_dir)
    registry = _load_registry(root)
    _verify_prepare_lineage(root, registry)
    if registry["source_config_sha256"] != config["source_config_sha256"]:
        raise ValueError("Worker source config drift")
    if registry.get("evaluation_frozen") or registry.get("test_data_opened"):
        raise RuntimeError("Training is forbidden after evaluation freeze")
    job = _job(root, registry, job_id)
    state = _read_json(_status_path(root, job_id))
    allowed = {
        "screen": {"pending", "failed"},
        "formal": {"pending", "failed"},
        "scaled": {"pending", "failed"},
        "window": {"pending", "failed"},
    }
    if state.get("status") == "completed":
        _verify_artifacts(state.get("artifacts"))
        return state
    if state.get("status") not in allowed[str(job["stage"])]:
        raise RuntimeError(f"Job {job_id} is gated in state {state.get('status')}")
    lock_path = root / "registry/job_locks" / f"{job_id}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Job lock is held: {job_id}") from exc
        materialized = _materialized_worker_spec(root, job)
        attempt = int(state.get("attempt", 0)) + 1
        running = {
            "schema_version": 1,
            "job_id": job_id,
            "job_spec_sha256": job["job_spec_sha256"],
            "status": "running",
            "attempt": attempt,
            "pid": os.getpid(),
            "process_start_ticks": process_start_ticks(os.getpid()),
            "gpu_id": int(job["gpu_id"]),
            "started_at_utc": utc_now(),
        }
        atomic_write_json(_status_path(root, job_id), running)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT)))
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
        threads = str(_mapping(config["runtime"], "runtime")["cpu_threads_per_job"])
        env.update(
            {
                name: threads
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            }
        )
        command = [
            str(_mapping(config["runtime"], "runtime")["python_executable"]),
            str(
                _resolve(
                    str(_mapping(config["runtime"], "runtime")["training_entrypoint"])
                )
            ),
            "--config",
            str(materialized["training_config_path"]),
            "--train-only",
        ]
        result = _run_training_subprocess(command, env=env)
        if result.returncode != 0:
            failed = {
                **running,
                "status": "failed",
                "returncode": result.returncode,
                "completed_at_utc": utc_now(),
            }
            atomic_write_json(_status_path(root, job_id), failed)
            raise RuntimeError(f"Training failed for {job_id}: {result.returncode}")
        try:
            (
                run_dir,
                artifacts,
                validation_mae,
                validation_persistence_mae,
                validation_log_mae_ratio,
                best_epoch,
            ) = _discover_training_artifacts(
                Path(str(materialized["training_config_path"]))
            )
            if job["stage"] == "screen":
                metrics_artifact = next(
                    row
                    for row in artifacts
                    if row["artifact_role"] == "training_metrics"
                )
                screen_metrics = pd.read_csv(str(metrics_artifact["path"]))
                if (
                    int(pd.to_numeric(screen_metrics["epoch"], errors="raise").max())
                    != 60
                ):
                    raise ValueError(
                        f"Screen job did not complete exactly 60 epochs: {job_id}"
                    )
        except BaseException as exc:
            atomic_write_json(
                _status_path(root, job_id),
                {
                    **running,
                    "status": "failed",
                    "returncode": 0,
                    "failure_reason": "artifact_validation_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed_at_utc": utc_now(),
                },
            )
            raise
        completed = {
            **running,
            "status": "completed",
            "returncode": 0,
            "run_dir": run_dir,
            "artifacts": artifacts,
            "validation_mae": validation_mae,
            "validation_persistence_mae": validation_persistence_mae,
            "validation_log_mae_ratio": validation_log_mae_ratio,
            "best_epoch": best_epoch,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(_status_path(root, job_id), completed)
        return completed


def _all_complete(root: Path, jobs: Sequence[Mapping[str, Any]]) -> bool:
    for job in jobs:
        state = _read_json(_status_path(root, str(job["job_id"])))
        if state.get("status") != "completed":
            return False
        _verify_artifacts(state.get("artifacts"))
    return True


def _inherited_global_slot_descriptors() -> tuple[int, ...]:
    value = os.environ.get(GLOBAL_GPU_SLOT_FD_ENV, "").strip()
    if not value:
        return ()
    try:
        descriptor = int(value)
        os.fstat(descriptor)
    except (OSError, ValueError) as exc:
        raise RuntimeError("Worker inherited an invalid global GPU slot FD") from exc
    return (descriptor,)


def freeze_screen(output_dir: str | Path) -> Path:
    root = _resolve(output_dir)
    registry = _load_registry(root)
    if registry["line"] != "architecture":
        raise ValueError("freeze-screen is architecture-only")
    if registry.get("screen_selection_frozen"):
        path = Path(str(registry["screen_selection_path"])).resolve()
        _verify_file(path, str(registry["screen_selection_sha256"]), "screen selection")
        _read_signed(path, kind="architecture_screen_selection_v1")
        return path
    screen = [row for row in registry["jobs"] if row["stage"] == "screen"]
    if not _all_complete(root, screen):
        raise RuntimeError("All nine screen jobs must complete before selection")
    rows = []
    for job in screen:
        state = _read_json(_status_path(root, str(job["job_id"])))
        metric = float(state.get("validation_mae", math.nan))
        if not math.isfinite(metric):
            raise ValueError(f"Screen job lacks validation_mae: {job['job_id']}")
        rows.append(
            {
                "mode": job["generator_mode"],
                "conditioning_learning_rate": job["conditioning_learning_rate"],
                "validation_mae": metric,
            }
        )
    selected: dict[str, dict[str, Any]] = {}
    for mode in ARCHITECTURE_MODES:
        selected[mode] = min(
            (row for row in rows if row["mode"] == mode),
            key=lambda row: row["validation_mae"],
        )
    point = min(selected.values(), key=lambda row: row["validation_mae"])
    path = root / "registry/screen_selection.json"
    _write_signed(
        path,
        {
            "schema_version": 1,
            "kind": "architecture_screen_selection_v1",
            "selection_metric": "validation_mae_at_best_hybrid_checkpoint",
            "checkpoint_metric": "val_hybrid_score",
            "test_metrics_read": 0,
            "selected_by_mode": selected,
            "screen_point_leader": point,
            "frozen_at_utc": utc_now(),
        },
    )
    for job in registry["jobs"]:
        if job["stage"] == "formal":
            status = _read_json(_status_path(root, str(job["job_id"])))
            if status.get("status") == "blocked_on_screen_selection":
                status.update(
                    status="pending",
                    screen_selection_sha256=_sha256_file(path),
                    updated_at_utc=utc_now(),
                )
                atomic_write_json(_status_path(root, str(job["job_id"])), status)
    registry.update(
        screen_selection_frozen=True,
        screen_selection_path=str(path),
        screen_selection_sha256=_sha256_file(path),
    )
    _save_registry(root, registry)
    materialize(registry["source_config_path"], root, stage="formal")
    return path


def freeze_architecture(output_dir: str | Path) -> Path:
    root = _resolve(output_dir)
    registry = _load_registry(root)
    if not registry.get("screen_selection_frozen"):
        raise RuntimeError("Screen selection must be frozen first")
    if registry.get("architecture_selection_frozen"):
        path = Path(str(registry["architecture_selection_path"])).resolve()
        _verify_file(
            path,
            str(registry["architecture_selection_sha256"]),
            "architecture selection",
        )
        _read_signed(path, kind="architecture_formal_selection_v1")
        _verify_file(
            Path(str(registry["scaled_capacity_profiles_path"])),
            str(registry["scaled_capacity_profiles_sha256"]),
            "scaled capacity profiles",
        )
        return path
    formal = [row for row in registry["jobs"] if row["stage"] == "formal"]
    if not _all_complete(root, formal):
        raise RuntimeError(
            "All 36 formal jobs must complete before architecture selection"
        )
    values: dict[str, list[float]] = {mode: [] for mode in ARCHITECTURE_MODES}
    cells: dict[str, set[tuple[int, str]]] = {
        mode: set() for mode in ARCHITECTURE_MODES
    }
    for job in formal:
        metric = float(
            _read_json(_status_path(root, str(job["job_id"]))).get(
                "validation_log_mae_ratio", math.nan
            )
        )
        if not math.isfinite(metric):
            raise ValueError(
                f"Formal job lacks validation log-MAE ratio: {job['job_id']}"
            )
        values[str(job["generator_mode"])].append(metric)
        cells[str(job["generator_mode"])].add((int(job["seed"]), str(job["fold"])))
    expected_cells = {(seed, fold) for seed in SEEDS for fold in FOLDS}
    if any(observed != expected_cells for observed in cells.values()):
        raise ValueError("Formal selection is missing a seed/fold validation cell")
    means = {mode: sum(rows) / len(rows) for mode, rows in values.items()}
    point_leader = min(means, key=means.__getitem__)
    path = root / "registry/architecture_selection.json"
    _write_signed(
        path,
        {
            "schema_version": 1,
            "kind": "architecture_formal_selection_v1",
            "selection_metric": "equal_seed_fold_mean_log_validation_mae_ratio_vs_persistence",
            "test_metrics_read": 0,
            "mean_log_validation_mae_ratio_by_mode": means,
            "point_leader_mode": point_leader,
            "frozen_at_utc": utc_now(),
        },
    )
    frozen_config = load_config(registry["source_config_path"])
    capacity_path = _freeze_scaled_capacity_profiles(
        frozen_config, root, point_leader_mode=point_leader
    )
    for job in registry["jobs"]:
        if job["stage"] == "scaled":
            status = _read_json(_status_path(root, str(job["job_id"])))
            if status.get("status") == "blocked_on_architecture_selection":
                status.update(
                    status="pending",
                    architecture_selection_sha256=_sha256_file(path),
                    scaled_capacity_profiles_sha256=_sha256_file(capacity_path),
                    updated_at_utc=utc_now(),
                )
                atomic_write_json(_status_path(root, str(job["job_id"])), status)
    registry.update(
        architecture_selection_frozen=True,
        architecture_selection_path=str(path),
        architecture_selection_sha256=_sha256_file(path),
        scaled_capacity_profiles_path=str(capacity_path),
        scaled_capacity_profiles_sha256=_sha256_file(capacity_path),
    )
    _save_registry(root, registry)
    materialize(registry["source_config_path"], root, stage="scaled")
    return path


def freeze_evaluation(output_dir: str | Path) -> Path:
    root = _resolve(output_dir)
    registry = _load_registry(root)
    if registry.get("evaluation_frozen"):
        path = Path(str(registry["checkpoint_allowlist_path"])).resolve()
        _verify_file(
            path, str(registry["checkpoint_allowlist_sha256"]), "evaluation allowlist"
        )
        _read_signed(path, kind="architecture_window_evaluation_allowlist_v1")
        freeze_prediction_plan(root)
        return path
    if registry.get("test_data_opened"):
        raise RuntimeError("Cannot freeze after test data were opened")
    if not _all_complete(root, registry["jobs"]):
        raise RuntimeError(
            "Every new training job must complete before evaluation freeze"
        )
    bindings = _read_signed(
        root / "registry/baseline_bindings.json", kind=BASELINE_KIND
    )
    rows = [dict(row) for row in bindings["checkpoint_rows"]]
    for job in registry["jobs"]:
        state = _read_json(_status_path(root, str(job["job_id"])))
        artifacts = {row["artifact_role"]: row for row in state["artifacts"]}
        for role in CHECKPOINT_ROLES:
            if role not in artifacts:
                raise ValueError(f"Missing {role}: {job['job_id']}")
            rows.append(
                {"job_id": job["job_id"], "stage": job["stage"], **artifacts[role]}
            )
    path = root / "registry/evaluation_checkpoint_allowlist.json"
    _write_signed(
        path,
        {
            "schema_version": 1,
            "kind": "architecture_window_evaluation_allowlist_v1",
            "test_metrics_read": 0,
            "checkpoint_rows": rows,
            "frozen_at_utc": utc_now(),
        },
    )
    registry.update(
        evaluation_frozen=True,
        checkpoint_allowlist_path=str(path),
        checkpoint_allowlist_sha256=_sha256_file(path),
    )
    _save_registry(root, registry)
    freeze_prediction_plan(root)
    return path


def _baseline_checkpoint(
    bindings: Mapping[str, Any], *, baseline: str, seed: int, fold: str
) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in bindings["checkpoint_rows"]
        if str(row["baseline"]) == baseline
        and int(row["seed"]) == int(seed)
        and str(row["fold"]) == fold
        and str(row["checkpoint_role"]) == "generator_best_learned"
    ]
    if len(rows) != 1:
        raise ValueError(f"No unique baseline Generator: {baseline}/{seed}/{fold}")
    return rows[0]


def _new_checkpoint(root: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    status = _read_json(_status_path(root, str(job["job_id"])))
    rows = [
        dict(row)
        for row in status["artifacts"]
        if str(row["artifact_role"]) == "generator_best_learned"
    ]
    if len(rows) != 1:
        raise ValueError(f"No unique selected Generator: {job['job_id']}")
    return rows[0]


def _standard_prediction_cells(
    root: Path, registry: Mapping[str, Any]
) -> list[dict[str, Any]]:
    bindings = _read_signed(
        root / "registry/baseline_bindings.json", kind=BASELINE_KIND
    )
    cells: list[dict[str, Any]] = []
    if registry["line"] == "architecture":
        for job in registry["jobs"]:
            if job["stage"] == "screen":
                continue
            checkpoint = _new_checkpoint(root, job)
            model_id = (
                f"formal:{job['generator_mode']}"
                if job["stage"] == "formal"
                else f"scaled:{job['variant']}"
            )
            cells.append(
                {
                    "source_cell_id": str(job["job_id"]),
                    "model_id": model_id,
                    "generator_mode": _resolved_job_mode(root, job),
                    "train_tolerance_minutes": 5,
                    "panel_tolerance_minutes": 5,
                    "panel_role": "common_5m_primary",
                    "fold": str(job["fold"]),
                    "seed": int(job["seed"]),
                    "gpu_id": int(job["gpu_id"]),
                    "run_dir": str(
                        _read_json(_status_path(root, str(job["job_id"])))["run_dir"]
                    ),
                    "checkpoint_path": str(checkpoint["path"]),
                    "checkpoint_sha256": str(checkpoint["sha256"]),
                }
            )
        config = yaml.safe_load(
            (root / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        for cell in registry["baseline_cells"]:
            baseline = str(cell["baseline"])
            checkpoint = _baseline_checkpoint(
                bindings,
                baseline=baseline,
                seed=int(cell["seed"]),
                fold=str(cell["fold"]),
            )
            baseline_config = _mapping(
                _mapping(config["baselines"], "baselines")[baseline], baseline
            )
            cells.append(
                {
                    "source_cell_id": str(cell["cell_id"]),
                    "model_id": baseline,
                    "generator_mode": str(baseline_config["generator_mode"]),
                    "train_tolerance_minutes": 5,
                    "panel_tolerance_minutes": 5,
                    "panel_role": "common_5m_primary",
                    "fold": str(cell["fold"]),
                    "seed": int(cell["seed"]),
                    "gpu_id": _gpu(config, str(cell["fold"])),
                    "run_dir": str(
                        Path(str(checkpoint["checkpoint_path"])).resolve().parents[1]
                    ),
                    "checkpoint_path": str(checkpoint["checkpoint_path"]),
                    "checkpoint_sha256": str(checkpoint["checkpoint_sha256"]),
                }
            )
        if len(cells) != 84:
            raise AssertionError(
                f"Architecture prediction cells must be 84, got {len(cells)}"
            )
    else:
        logical: list[dict[str, Any]] = []
        for job in registry["jobs"]:
            checkpoint = _new_checkpoint(root, job)
            logical.append(
                {
                    "source_cell_id": str(job["job_id"]),
                    "model_id": str(job["arm"]),
                    "generator_mode": str(job["generator_mode"]),
                    "train_tolerance_minutes": int(job["tolerance_minutes"]),
                    "fold": str(job["fold"]),
                    "seed": int(job["seed"]),
                    "gpu_id": int(job["gpu_id"]),
                    "run_dir": str(
                        _read_json(_status_path(root, str(job["job_id"])))["run_dir"]
                    ),
                    "checkpoint_path": str(checkpoint["path"]),
                    "checkpoint_sha256": str(checkpoint["sha256"]),
                }
            )
        config = yaml.safe_load(
            (root / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        for cell in registry["baseline_cells"]:
            baseline = str(cell["baseline"])
            checkpoint = _baseline_checkpoint(
                bindings,
                baseline=baseline,
                seed=int(cell["seed"]),
                fold=str(cell["fold"]),
            )
            baseline_config = _mapping(
                _mapping(config["baselines"], "baselines")[baseline], baseline
            )
            logical.append(
                {
                    "source_cell_id": str(cell["cell_id"]),
                    "model_id": baseline,
                    "generator_mode": str(baseline_config["generator_mode"]),
                    "train_tolerance_minutes": 5,
                    "fold": str(cell["fold"]),
                    "seed": int(cell["seed"]),
                    "gpu_id": _gpu(config, str(cell["fold"])),
                    "run_dir": str(
                        Path(str(checkpoint["checkpoint_path"])).resolve().parents[1]
                    ),
                    "checkpoint_path": str(checkpoint["checkpoint_path"]),
                    "checkpoint_sha256": str(checkpoint["checkpoint_sha256"]),
                }
            )
        if len(logical) != 120:
            raise AssertionError(
                f"Window logical prediction cells must be 120, got {len(logical)}"
            )
        for cell in logical:
            cells.append(
                {
                    **cell,
                    "panel_tolerance_minutes": 5,
                    "panel_role": "common_5m_primary",
                }
            )
            if int(cell["train_tolerance_minutes"]) != 5:
                cells.append(
                    {
                        **cell,
                        "panel_tolerance_minutes": int(cell["train_tolerance_minutes"]),
                        "panel_role": "own_tolerance_secondary",
                    }
                )
        if len(cells) != 216:
            raise AssertionError(
                f"Window standard predictions must be 216, got {len(cells)}"
            )
    return cells


def freeze_prediction_plan(output_dir: str | Path) -> Path:
    root = _resolve(output_dir)
    registry = _load_registry(root)
    if not registry.get("evaluation_frozen"):
        raise RuntimeError("Prediction plan requires frozen checkpoints")
    path = root / "registry/prediction_plan.json"
    standard = _standard_prediction_cells(root, registry)
    evaluations: list[dict[str, Any]] = []
    for cell in standard:
        conditional = str(cell["generator_mode"]) != "cnn_unet_mask_coords_v1"
        conditions = _prediction_text_conditions(
            line=str(registry["line"]), conditional=conditional
        )
        for condition in conditions:
            item = {
                **cell,
                "text_condition": condition,
                "evaluation_id": _job_id(
                    "eval",
                    cell["source_cell_id"],
                    cell["panel_role"],
                    condition,
                ),
            }
            item["evaluation_spec_sha256"] = _payload_sha256(item)
            evaluations.append(item)
    if len({row["evaluation_id"] for row in evaluations}) != len(evaluations):
        raise ValueError("Duplicate prediction evaluation IDs")
    expected = 228 if registry["line"] == "architecture" else 324
    if len(evaluations) != expected:
        raise ValueError(
            f"Prediction+intervention count drift: {len(evaluations)} != {expected}"
        )
    payload = {
        "schema_version": 1,
        "kind": PREDICTION_PLAN_KIND,
        "study_kind": registry["study_kind"],
        "standard_prediction_units": len(standard),
        "counterfactual_additional_units": len(evaluations) - len(standard),
        "total_evaluation_units": len(evaluations),
        "test_data_read": False,
        "evaluations": evaluations,
        "frozen_at_utc": utc_now(),
    }
    if path.is_file():
        existing = _read_signed(path, kind=PREDICTION_PLAN_KIND)
        if existing["evaluations"] != evaluations:
            raise ValueError("Frozen prediction plan drift")
    else:
        _write_signed(path, payload)
    registry = _load_registry(root)
    existing_sha = registry.get("prediction_plan_sha256")
    if existing_sha is not None and existing_sha != _sha256_file(path):
        raise ValueError("Frozen prediction-plan registry binding drift")
    registry.update(
        prediction_plan_path=str(path.resolve()),
        prediction_plan_sha256=_sha256_file(path),
    )
    _save_registry(root, registry)
    return path


def _prediction_text_conditions(*, line: str, conditional: bool) -> tuple[str, ...]:
    if not conditional:
        return ("zero",)
    return (
        ("matched", "zero", "shuffle")
        if line == "architecture"
        else ("matched", "zero")
    )


def _prediction_embedding(
    *,
    pair_id: str,
    condition: str,
    lp_vectors: Mapping[str, np.ndarray],
    donor_by_pair: Mapping[str, str],
) -> tuple[np.ndarray, str | None]:
    """Resolve one counterfactual embedding without touching unused donors."""

    if condition == "matched":
        return lp_vectors[pair_id], None
    if condition == "zero":
        return np.zeros(1024, dtype=np.float32), None
    if condition == "shuffle":
        donor = donor_by_pair[pair_id]
        return lp_vectors[donor], donor
    raise ValueError(f"Unsupported prediction text condition: {condition}")


def _write_frame(path: Path, frame: pd.DataFrame, *, gzip: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if gzip:
        frame.to_csv(
            temporary,
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return path


def _test_panel_path(root: Path, tolerance: int, fold: str) -> Path:
    return (
        root
        / "evaluation/test_panels"
        / f"tolerance_{tolerance:02d}m"
        / f"{fold}.csv.gz"
    )


def _test_overlay_path(root: Path, tolerance: int, fold: str, condition: str) -> Path:
    return (
        root
        / "evaluation/test_text_overlays"
        / f"tolerance_{tolerance:02d}m"
        / fold
        / f"{condition}.json"
    )


def _verify_test_input_manifest(root: Path) -> Path:
    path = root / "evaluation/test_input_manifest.json"
    payload = _read_signed(path, kind="architecture_window_test_inputs_v1")
    for row in payload["artifacts"]:
        _verify_file(
            Path(str(row["path"])), str(row["sha256"]), str(row["artifact_role"])
        )
        if Path(str(row["path"])).stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Test input size drift: {row['path']}")
    return path


def _verify_evaluation_bundle(root: Path, registry: Mapping[str, Any]) -> None:
    """Verify every frozen pre-test input before any test-facing action."""

    _verify_prepare_lineage(root, registry)
    allowlist_path = Path(str(registry["checkpoint_allowlist_path"])).resolve()
    _verify_file(
        allowlist_path,
        str(registry["checkpoint_allowlist_sha256"]),
        "evaluation checkpoint allowlist",
    )
    allowlist = _read_signed(
        allowlist_path, kind="architecture_window_evaluation_allowlist_v1"
    )
    for row in allowlist["checkpoint_rows"]:
        checkpoint = Path(str(row.get("path") or row.get("checkpoint_path"))).resolve()
        digest = str(row.get("sha256") or row.get("checkpoint_sha256"))
        if (
            not checkpoint.is_file()
            or checkpoint.stat().st_size != int(row["size_bytes"])
            or _sha256_file(checkpoint) != digest
        ):
            raise ValueError(f"Frozen checkpoint allowlist drift: {checkpoint}")
    plan_path = Path(str(registry["prediction_plan_path"])).resolve()
    _verify_file(plan_path, str(registry["prediction_plan_sha256"]), "prediction plan")
    _read_signed(plan_path, kind=PREDICTION_PLAN_KIND)


def _verify_prediction_bundle(root: Path, registry: Mapping[str, Any]) -> None:
    """Verify aggregate and per-cell prediction evidence as one frozen bundle."""

    if not registry.get("predictions_frozen"):
        raise RuntimeError("Prediction bundle is not frozen")
    _verify_evaluation_bundle(root, registry)
    test_manifest = Path(str(registry["test_input_manifest_path"])).resolve()
    _verify_file(
        test_manifest,
        str(registry["test_input_manifest_sha256"]),
        "test input manifest",
    )
    _verify_test_input_manifest(root)
    pair_metrics_path = Path(str(registry["pair_metrics_path"])).resolve()
    prediction_manifest_path = Path(str(registry["prediction_manifest_path"])).resolve()
    _verify_file(
        pair_metrics_path, str(registry["pair_metrics_sha256"]), "pair metrics"
    )
    _verify_file(
        prediction_manifest_path,
        str(registry["prediction_manifest_sha256"]),
        "prediction manifest",
    )
    plan = _read_signed(
        Path(str(registry["prediction_plan_path"])), kind=PREDICTION_PLAN_KIND
    )
    manifest = pd.read_csv(prediction_manifest_path)
    expected_ids = {str(row["evaluation_id"]) for row in plan["evaluations"]}
    observed_ids = set(manifest["evaluation_id"].astype(str))
    if (
        len(manifest) != int(plan["total_evaluation_units"])
        or manifest["evaluation_id"].astype(str).duplicated().any()
        or observed_ids != expected_ids
    ):
        raise ValueError("Prediction manifest evaluation universe drift")
    for row in manifest.to_dict(orient="records"):
        cell_manifest_path = Path(str(row["manifest_path"])).resolve()
        _verify_file(
            cell_manifest_path,
            str(row["manifest_sha256"]),
            "prediction cell manifest",
        )
        cell = _read_signed(
            cell_manifest_path, kind="architecture_window_prediction_cell_v1"
        )
        prediction_path = Path(str(cell["prediction_path"])).resolve()
        cell_metrics_path = Path(str(cell["pair_metrics_path"])).resolve()
        _verify_file(
            prediction_path, str(cell["prediction_sha256"]), "cell raw prediction"
        )
        _verify_file(
            cell_metrics_path,
            str(cell["pair_metrics_sha256"]),
            "cell pair metrics",
        )
        if len(pd.read_csv(cell_metrics_path)) != int(cell["row_count"]):
            raise ValueError(f"Prediction cell row-count drift: {cell_manifest_path}")
    metrics = pd.read_csv(pair_metrics_path)
    if (
        len(metrics) != int(registry["prediction_pair_rows"])
        or metrics.duplicated(["evaluation_id", "pair_id"]).any()
    ):
        raise ValueError("Aggregate prediction evidence drift")


def _verify_analysis_bundle(root: Path, registry: Mapping[str, Any]) -> None:
    path = Path(str(registry["analysis_manifest_path"])).resolve()
    _verify_file(path, str(registry["analysis_manifest_sha256"]), "analysis manifest")
    manifest = _read_signed(path, kind="architecture_window_analysis_manifest_v1")
    if manifest["pair_metrics_sha256"] != str(registry["pair_metrics_sha256"]):
        raise ValueError("Analysis/prediction lineage drift")
    _verify_file(
        Path(str(manifest["summary_path"])),
        str(manifest["summary_sha256"]),
        "analysis summary",
    )


def _verify_bootstrap_bundle(root: Path, registry: Mapping[str, Any]) -> None:
    path = Path(str(registry["bootstrap_manifest_path"])).resolve()
    _verify_file(path, str(registry["bootstrap_manifest_sha256"]), "bootstrap manifest")
    manifest = _read_signed(path, kind="architecture_window_bootstrap_manifest_v1")
    _verify_file(
        Path(str(manifest["result_path"])),
        str(manifest["result_sha256"]),
        "bootstrap results",
    )


def _verify_report_bundle(root: Path, registry: Mapping[str, Any]) -> None:
    path = Path(str(registry["report_manifest_path"])).resolve()
    _verify_file(path, str(registry["report_manifest_sha256"]), "report manifest")
    manifest = _read_signed(path, kind="architecture_window_report_manifest_v1")
    for role in ("markdown", "html"):
        _verify_file(
            Path(str(manifest[f"{role}_path"])),
            str(manifest[f"{role}_sha256"]),
            f"{role} report",
        )


def _materialize_test_inputs(config: Mapping[str, Any], root: Path) -> Path:
    """Open fold-test rows only after both selection and checkpoint freeze."""

    registry = _load_registry(root)
    if not registry.get("evaluation_frozen"):
        raise RuntimeError("Test inputs require evaluation freeze")
    existing = root / "evaluation/test_input_manifest.json"
    if existing.is_file():
        path = _verify_test_input_manifest(root)
        if not registry.get("test_data_opened"):
            raise ValueError("Partial test-input freeze requires audit")
        return path
    from wgan_option.utils.news_first_experiment_core import (
        _fixed_partition_derangement,
        write_pair_text_overlay_manifest,
    )

    data = _mapping(config["data"], "data")
    dataset_root = _resolve(str(data["root"]))
    universe = pd.read_csv(root / "inputs/pair_universes.csv", dtype=str)
    tolerances = (5,) if registry["line"] == "architecture" else WINDOW_TOLERANCES
    artifacts: list[dict[str, Any]] = []
    for tolerance in tolerances:
        test_universe = universe.loc[
            universe["tolerance_minutes"].astype(int).eq(tolerance)
            & universe["partition"].eq("test")
        ].copy()
        required = set(test_universe["pair_id"].astype(str))
        workbook = dataset_root / str(data["workbook_template"]).format(
            tolerance02=f"{tolerance:02d}"
        )
        full = pd.read_excel(
            workbook, sheet_name=str(data.get("sheet_name", "gan_input_ready"))
        )
        full["pair_id"] = full["pair_id"].astype(str)
        full = full.loc[full["pair_id"].isin(required)].copy()
        if set(full["pair_id"]) != required:
            raise ValueError(f"Test workbook coverage drift: {tolerance}m")
        canonical: dict[str, dict[str, Any]] = {}
        for pair_id, group in full.groupby("pair_id", sort=False):
            ordered = group.assign(
                _news_sort=pd.to_numeric(group["news_row_id"], errors="raise")
            ).sort_values(["_news_sort", "sample_id"], kind="stable")
            canonical[str(pair_id)] = (
                ordered.iloc[0].drop(labels="_news_sort").to_dict()
            )
        lp_vectors = _lp_vectors_for_pairs(
            workbook,
            sheet_name=str(data.get("sheet_name", "gan_input_ready")),
            required_pairs=required,
        )
        for fold in FOLDS:
            selected = test_universe.loc[test_universe["fold"].eq(fold)].copy()
            pair_ids = sorted(selected["pair_id"].astype(str))
            sessions = dict(
                zip(
                    selected["pair_id"].astype(str),
                    selected["session_id"].astype(str),
                )
            )
            panel = pd.DataFrame([canonical[pair_id] for pair_id in pair_ids])
            panel["sample_id"] = (
                panel["pair_id"].astype(str).map(lambda value: f"pair::{value}")
            )
            panel["sample_weight"] = 1.0
            panel_path = _write_frame(
                _test_panel_path(root, tolerance, fold), panel, gzip=True
            )
            artifacts.append(
                {
                    "artifact_role": f"test_panel:{tolerance:02d}m:{fold}",
                    "path": str(panel_path.resolve()),
                    "size_bytes": panel_path.stat().st_size,
                    "sha256": _sha256_file(panel_path),
                }
            )
            test_conditions = _prediction_text_conditions(
                line=str(registry["line"]), conditional=True
            )
            donor_by_pair = (
                _fixed_partition_derangement(
                    pair_ids,
                    master_seed=20260907,
                    namespace=f"architecture_window/test/{tolerance:02d}m/{fold}/shuffle",
                )
                if "shuffle" in test_conditions
                else {}
            )
            for condition in test_conditions:
                records = []
                for pair_id in pair_ids:
                    vector, donor = _prediction_embedding(
                        pair_id=pair_id,
                        condition=condition,
                        lp_vectors=lp_vectors,
                        donor_by_pair=donor_by_pair,
                    )
                    records.append(
                        {
                            "pair_id": pair_id,
                            "session_id": sessions[pair_id],
                            "embedding": vector,
                            **({"donor_pair_id": donor} if donor is not None else {}),
                        }
                    )
                overlay_path = _test_overlay_path(root, tolerance, fold, condition)
                write_pair_text_overlay_manifest(
                    overlay_path,
                    mode={
                        "matched": "lp_mean_l2",
                        "zero": "current_only",
                        "shuffle": "lp_shuffle",
                    }[condition],
                    namespace=f"architecture_window/test/{tolerance:02d}m/{fold}/{condition}",
                    records=records,
                    transform={
                        "method": {
                            "matched": "unique_article_lp_mean_l2_v1",
                            "zero": "zero_vector_v1",
                            "shuffle": "test_partition_pair_derangement_v1",
                        }[condition],
                        "master_seed": 20260907 if condition == "shuffle" else None,
                    },
                )
                artifacts.append(
                    {
                        "artifact_role": f"test_overlay:{tolerance:02d}m:{fold}:{condition}",
                        "path": str(overlay_path.resolve()),
                        "size_bytes": overlay_path.stat().st_size,
                        "sha256": _sha256_file(overlay_path),
                    }
                )
    path = _write_signed(
        existing,
        {
            "schema_version": 1,
            "kind": "architecture_window_test_inputs_v1",
            "test_opened_after_evaluation_freeze": True,
            "artifacts": artifacts,
            "frozen_at_utc": utc_now(),
        },
    )
    registry.update(
        test_data_opened=True,
        test_input_manifest_path=str(path.resolve()),
        test_input_manifest_sha256=_sha256_file(path),
    )
    _save_registry(root, registry)
    return path


def _panel_with_text(
    root: Path, *, tolerance: int, fold: str, condition: str
) -> pd.DataFrame:
    from wgan_option.utils.news_first_experiment_core import (
        load_pair_text_overlay_manifest,
    )

    panel = pd.read_csv(_test_panel_path(root, tolerance, fold), low_memory=False)
    overlay_path = _test_overlay_path(root, tolerance, fold, condition)
    payload = _read_json(overlay_path)
    mode = {"matched": "lp_mean_l2", "zero": "current_only", "shuffle": "lp_shuffle"}[
        condition
    ]
    overlay = load_pair_text_overlay_manifest(
        overlay_path,
        _sha256_file(overlay_path),
        str(payload["profile_sha256"]),
        expected_mode=mode,
    )
    pair_ids = panel["pair_id"].astype(str)
    if pair_ids.duplicated().any() or set(pair_ids) != set(overlay.embeddings):
        raise ValueError("Test panel/text overlay universe drift")
    panel["lp_embedding"] = [
        json.dumps(
            overlay.embeddings[pair_id].astype(float).tolist(), separators=(",", ":")
        )
        for pair_id in pair_ids
    ]
    panel["hd_embedding"] = ""
    return panel


def _prediction_noise_sha(cell: Mapping[str, Any], sample_ids: Sequence[str]) -> str:
    return _payload_sha256(
        {
            "method": "stable_noise_for_keys_v1",
            "panel_tolerance_minutes": int(cell["panel_tolerance_minutes"]),
            "fold": str(cell["fold"]),
            "seed": int(cell["seed"]),
            "sample_ids": sorted(map(str, sample_ids)),
            "draws": 64,
            "noise_dim": 32,
        }
    )


def _evaluate_prediction_cell(root: Path, cell: Mapping[str, Any]) -> pd.DataFrame:
    from scripts.rq123.news_first_vol_film_nolp_10seed import (
        _configure_prediction_determinism,
    )
    from scripts.rq3.news_first_vol_comparison_analysis import (
        RunSpec,
        TrainedRunEvaluator,
        _enforce_formal_run_coverage,
        _prediction_export_frame,
        aggregate_pair_metrics,
        compute_sample_metrics,
    )

    _configure_prediction_determinism(int(cell["seed"]))
    panel = _panel_with_text(
        root,
        tolerance=int(cell["panel_tolerance_minutes"]),
        fold=str(cell["fold"]),
        condition=str(cell["text_condition"]),
    )
    run = RunSpec(
        run_id=str(cell["evaluation_id"]),
        run_dir=Path(str(cell["run_dir"])),
        model="wgan",
        tolerance_minutes=int(cell["train_tolerance_minutes"]),
        seed=int(cell["seed"]),
        checkpoint_path=Path(str(cell["checkpoint_path"])),
        text_ablation_mode="real_text",
        support_mask_mode="raw_joint",
        generator_current_input_mode="current_support_masked",
        metadata={"fold": cell["fold"], "model_id": cell["model_id"]},
    )
    panel_name = f"{cell['panel_role']}__{int(cell['panel_tolerance_minutes']):02d}m__{cell['fold']}"
    evaluator = TrainedRunEvaluator(
        mc_samples=64,
        sample_batch_size=32,
        draw_batch_size=64,
        device=f"cuda:{int(cell['gpu_id'])}",
    )
    predictions = evaluator(run, panel_name, panel)
    samples, exclusions, metric_exclusions = compute_sample_metrics(
        run, panel_name, panel, predictions, evaluate_embedded_atm_skew=False
    )
    export = _prediction_export_frame(run, panel_name, panel, predictions, samples)
    _enforce_formal_run_coverage(run, panel_name, panel, samples, exclusions, export)
    prediction_path = root / "predictions/cells" / f"{cell['evaluation_id']}.csv.gz"
    _write_frame(prediction_path, export, gzip=True)
    overall = aggregate_pair_metrics(samples)
    overall = overall.loc[
        overall["stratum_type"].eq("overall") & overall["stratum_value"].eq("all")
    ].copy()
    required = {
        "pair_id",
        "session_id",
        "model_mae",
        "persistence_mae",
        "skill",
        "predicted_calendar_violation_rate",
        "target_calendar_violation_rate",
        "calendar_violation_rate_gap",
        "predicted_butterfly_violation_rate",
        "target_butterfly_violation_rate",
        "butterfly_violation_rate_gap",
    }
    missing = sorted(required - set(overall.columns))
    if missing:
        raise ValueError(f"Prediction diagnostics missing columns: {missing}")
    noise_sha = _prediction_noise_sha(cell, panel["sample_id"].astype(str).tolist())
    relation_by_pair = panel.assign(pair_id=panel["pair_id"].astype(str)).set_index(
        "pair_id"
    )[["window_relation", "post_news_current_overlap_minutes"]]
    metric_pair_ids = overall["pair_id"].astype(str)
    if set(metric_pair_ids) != set(relation_by_pair.index):
        raise ValueError("Prediction metrics lost window-relation lineage")
    evidence = pd.DataFrame(
        {
            "evaluation_id": str(cell["evaluation_id"]),
            "source_cell_id": str(cell["source_cell_id"]),
            "model_id": str(cell["model_id"]),
            "generator_mode": str(cell["generator_mode"]),
            "train_tolerance_minutes": int(cell["train_tolerance_minutes"]),
            "panel_tolerance_minutes": int(cell["panel_tolerance_minutes"]),
            "tolerance_minutes": int(cell["panel_tolerance_minutes"]),
            "panel_role": str(cell["panel_role"]),
            "text_condition": str(cell["text_condition"]),
            "fold": str(cell["fold"]),
            "seed": int(cell["seed"]),
            "pair_id": overall["pair_id"].astype(str),
            "session_id": overall["session_id"].astype(str),
            "window_relation": metric_pair_ids.map(
                relation_by_pair["window_relation"]
            ).astype(str),
            "post_news_current_overlap_minutes": pd.to_numeric(
                metric_pair_ids.map(
                    relation_by_pair["post_news_current_overlap_minutes"]
                ),
                errors="raise",
            ),
            "target_mae": pd.to_numeric(overall["model_mae"], errors="raise"),
            "persistence_mae": pd.to_numeric(
                overall["persistence_mae"], errors="raise"
            ),
            "persistence_skill": pd.to_numeric(overall["skill"], errors="raise"),
            "predicted_calendar_violation_rate": pd.to_numeric(
                overall["predicted_calendar_violation_rate"], errors="raise"
            ),
            "target_calendar_violation_rate": pd.to_numeric(
                overall["target_calendar_violation_rate"], errors="raise"
            ),
            "calendar_violation_rate_gap": pd.to_numeric(
                overall["calendar_violation_rate_gap"], errors="raise"
            ),
            "predicted_butterfly_violation_rate": pd.to_numeric(
                overall["predicted_butterfly_violation_rate"], errors="raise"
            ),
            "target_butterfly_violation_rate": pd.to_numeric(
                overall["target_butterfly_violation_rate"], errors="raise"
            ),
            "butterfly_violation_rate_gap": pd.to_numeric(
                overall["butterfly_violation_rate_gap"], errors="raise"
            ),
            "checkpoint_sha256": str(cell["checkpoint_sha256"]),
            "prediction_sha256": _sha256_file(prediction_path),
            "noise_bank_profile_sha256": noise_sha,
        }
    )
    numeric = evidence.select_dtypes(include="number").to_numpy(dtype=float)
    if len(evidence) != len(panel) or not np.isfinite(numeric).all():
        raise ValueError(
            f"Incomplete/non-finite prediction evidence: {cell['evaluation_id']}"
        )
    evidence_path = (
        root / "predictions/cells" / f"{cell['evaluation_id']}.pair_metrics.csv"
    )
    _write_frame(evidence_path, evidence)
    manifest_path = (
        root / "predictions/cells" / f"{cell['evaluation_id']}.manifest.json"
    )
    _write_signed(
        manifest_path,
        {
            "schema_version": 1,
            "kind": "architecture_window_prediction_cell_v1",
            "evaluation_spec_sha256": cell["evaluation_spec_sha256"],
            "checkpoint_path": str(Path(str(cell["checkpoint_path"])).resolve()),
            "checkpoint_sha256": str(cell["checkpoint_sha256"]),
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": _sha256_file(prediction_path),
            "pair_metrics_path": str(evidence_path.resolve()),
            "pair_metrics_sha256": _sha256_file(evidence_path),
            "row_count": len(evidence),
            "noise_bank_profile_sha256": noise_sha,
            "optional_metric_exclusion_count": len(metric_exclusions),
        },
    )
    return evidence


def predict(
    config_path: str | Path, output_dir: str | Path, *, resume: bool = False
) -> Path:
    config = load_config(config_path)
    root = _resolve(output_dir)
    registry = _load_registry(root)
    if registry.get("source_config_sha256") != config["source_config_sha256"]:
        raise ValueError("Prediction source config drift")
    if not registry.get("evaluation_frozen"):
        raise RuntimeError("Prediction requires frozen evaluation allowlist")
    _verify_evaluation_bundle(root, registry)
    if registry.get("predictions_frozen"):
        _verify_prediction_bundle(root, registry)
        return Path(str(registry["pair_metrics_path"]))
    plan = _read_signed(
        root / "registry/prediction_plan.json", kind=PREDICTION_PLAN_KIND
    )
    allowlist = _read_signed(
        root / "registry/evaluation_checkpoint_allowlist.json",
        kind="architecture_window_evaluation_allowlist_v1",
    )
    allowed: set[tuple[str, str]] = set()
    for row in allowlist["checkpoint_rows"]:
        checkpoint_path = Path(
            str(row.get("path") or row.get("checkpoint_path"))
        ).resolve()
        checkpoint_sha = str(row.get("sha256") or row.get("checkpoint_sha256"))
        expected_size = int(row["size_bytes"])
        if (
            not checkpoint_path.is_file()
            or checkpoint_path.stat().st_size != expected_size
            or _sha256_file(checkpoint_path) != checkpoint_sha
        ):
            raise ValueError(f"Frozen checkpoint allowlist drift: {checkpoint_path}")
        allowed.add((str(checkpoint_path), checkpoint_sha))
    if len({str(row["evaluation_id"]) for row in plan["evaluations"]}) != int(
        plan["total_evaluation_units"]
    ):
        raise ValueError("Frozen prediction plan evaluation IDs are not unique")
    for cell in plan["evaluations"]:
        identity = (
            str(Path(str(cell["checkpoint_path"])).resolve()),
            str(cell["checkpoint_sha256"]),
        )
        if identity not in allowed:
            raise ValueError(
                f"Prediction checkpoint is outside allowlist: {identity[0]}"
            )
    _materialize_test_inputs(config, root)
    evidence_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for cell in plan["evaluations"]:
        manifest_path = (
            root / "predictions/cells" / f"{cell['evaluation_id']}.manifest.json"
        )
        if manifest_path.is_file():
            if not resume:
                raise ValueError(
                    f"Existing prediction requires --resume: {cell['evaluation_id']}"
                )
            manifest = _read_signed(
                manifest_path, kind="architecture_window_prediction_cell_v1"
            )
            if manifest["evaluation_spec_sha256"] != cell["evaluation_spec_sha256"]:
                raise ValueError(f"Prediction spec drift: {cell['evaluation_id']}")
            evidence_path = Path(str(manifest["pair_metrics_path"]))
            _verify_file(
                evidence_path, str(manifest["pair_metrics_sha256"]), "cell pair metrics"
            )
            _verify_file(
                Path(str(manifest["prediction_path"])),
                str(manifest["prediction_sha256"]),
                "cell raw prediction",
            )
            evidence = pd.read_csv(evidence_path)
            if len(evidence) != int(manifest["row_count"]):
                raise ValueError(f"Prediction row-count drift: {cell['evaluation_id']}")
        else:
            evidence = _evaluate_prediction_cell(root, cell)
            manifest = _read_signed(
                manifest_path, kind="architecture_window_prediction_cell_v1"
            )
        evidence_frames.append(evidence)
        manifest_rows.append(
            {
                "evaluation_id": str(cell["evaluation_id"]),
                "source_cell_id": str(cell["source_cell_id"]),
                "model_id": str(cell["model_id"]),
                "train_tolerance_minutes": int(cell["train_tolerance_minutes"]),
                "panel_tolerance_minutes": int(cell["panel_tolerance_minutes"]),
                "panel_role": str(cell["panel_role"]),
                "text_condition": str(cell["text_condition"]),
                "fold": str(cell["fold"]),
                "seed": int(cell["seed"]),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": _sha256_file(manifest_path),
                "row_count": int(manifest["row_count"]),
                "noise_bank_profile_sha256": str(manifest["noise_bank_profile_sha256"]),
            }
        )
    if len(evidence_frames) != int(plan["total_evaluation_units"]):
        raise ValueError("Prediction evaluation universe is incomplete")
    combined = pd.concat(evidence_frames, ignore_index=True)
    pair_metrics = _write_frame(
        root / "predictions/pair_metrics.csv.gz", combined, gzip=True
    )
    manifest_csv = _write_frame(
        root / "predictions/prediction_manifest.csv", pd.DataFrame(manifest_rows)
    )
    if pd.DataFrame(manifest_rows)["evaluation_id"].astype(str).duplicated().any():
        raise ValueError("Prediction manifest evaluation IDs are not unique")
    if combined.duplicated(["evaluation_id", "pair_id"]).any():
        raise ValueError("Prediction evidence has duplicate evaluation/pair keys")
    # Same seed/fold/panel must use one noise bank across architectures and interventions.
    noise_counts = (
        pd.DataFrame(manifest_rows)
        .groupby(["seed", "fold", "panel_tolerance_minutes"], sort=True)[
            "noise_bank_profile_sha256"
        ]
        .nunique()
    )
    if not noise_counts.eq(1).all():
        raise ValueError("Prediction cells do not share the frozen MC64 noise bank")
    registry = _load_registry(root)
    registry.update(
        predictions_frozen=True,
        predictions_frozen_at_utc=utc_now(),
        pair_metrics_path=str(pair_metrics.resolve()),
        pair_metrics_sha256=_sha256_file(pair_metrics),
        prediction_manifest_path=str(manifest_csv.resolve()),
        prediction_manifest_sha256=_sha256_file(manifest_csv),
        prediction_evaluation_units=len(manifest_rows),
        prediction_pair_rows=len(combined),
    )
    _save_registry(root, registry)
    return pair_metrics


def _worker_materialization_ready(
    root: Path, jobs: Sequence[Mapping[str, Any]]
) -> bool:
    path = root / "registry/materialized_worker_specs.json"
    if not path.is_file():
        return False
    manifest = _read_signed(path, kind=WORKER_SPEC_KIND)
    available = {str(row.get("job_id")) for row in manifest.get("jobs", [])}
    return all(str(job["job_id"]) in available for job in jobs)


FORMAL_ROOT_BASENAMES = (
    "rq3_news_first_vol_generator_architecture_3seed_5m_exact_ttm_v1",
    "rq3_news_first_vol_alignment_tolerance_3seed_exact_ttm_v1",
)
FORMAL_ROOT_CONFIGS = {
    FORMAL_ROOT_BASENAMES[0]: DEFAULT_ARCHITECTURE_CONFIG,
    FORMAL_ROOT_BASENAMES[1]: DEFAULT_WINDOW_CONFIG,
}


def _shared_gpu_capacity(
    root: Path, current: Mapping[str, Any]
) -> tuple[int, list[dict[str, Any]]]:
    shared = root.parent / "rq3_architecture_window_shared_gpu_slots_v1"
    suite = _read_signed(
        shared / "benchmark_suite.json",
        kind="architecture_window_dual_benchmark_suite_v1",
    )
    suite_rows = {
        Path(str(row["formal_root"])).resolve().name: dict(row)
        for row in suite.get("benchmarks", [])
    }
    if set(suite_rows) != set(FORMAL_ROOT_BASENAMES):
        raise ValueError("Dual benchmark suite root coverage drift")
    bindings: list[dict[str, Any]] = []
    for basename in FORMAL_ROOT_BASENAMES:
        path = root.parent / f"{basename}_control" / "benchmark.json"
        if not path.is_file():
            raise RuntimeError(
                "Both architecture/window benchmarks must be frozen before either "
                f"training pipeline starts; missing {path}"
            )
        payload = _read_signed(path, kind=GPU_BENCHMARK_KIND)
        if (
            payload.get("status") != "passed"
            or Path(str(payload.get("formal_output_root", ""))).resolve().name
            != basename
            or _sha256_file(path) != str(suite_rows[basename]["benchmark_sha256"])
        ):
            raise ValueError(f"Peer benchmark is not passed: {path}")
        bindings.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "selected_workers_per_gpu": int(payload["selected_workers_per_gpu"]),
            }
        )
    current_capacity = int(current["selected_workers_per_gpu"])
    capacities = [
        current_capacity,
        *(row["selected_workers_per_gpu"] for row in bindings),
    ]
    return min(capacities), bindings


def _acquire_global_gpu_slot(
    root: Path, *, gpu_id: int, capacity: int
) -> tuple[int, int] | None:
    slot_root = (
        root.parent / "rq3_architecture_window_shared_gpu_slots_v1" / f"gpu_{gpu_id}"
    )
    slot_root.mkdir(parents=True, exist_ok=True)
    for slot in range(capacity):
        path = slot_root / f"slot_{slot:02d}.lock"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            continue
        return descriptor, slot
    return None


def _release_global_gpu_slot(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _terminate_worker_process_groups(
    active: Mapping[
        str, tuple[subprocess.Popen[Any], Any, Mapping[str, Any], int, int]
    ],
) -> None:
    """Stop wrappers and their real-training children before releasing GPU slots."""

    for process, _handle, _job, _descriptor, _slot in active.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        live_groups = []
        for process, _handle, _job, _descriptor, _slot in active.values():
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                continue
            live_groups.append(process.pid)
        if not live_groups:
            break
        time.sleep(0.1)
    for process, _handle, _job, _descriptor, _slot in active.values():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process, handle, _job, descriptor, _slot in active.values():
        process.wait()
        handle.close()
        _release_global_gpu_slot(descriptor)


def launch_stage(config_path: str | Path, output_dir: str | Path, stage: str) -> Path:
    config = load_config(config_path)
    root = _resolve(output_dir)
    registry = _load_registry(root)
    materialize(config_path, root, stage=stage)
    registry = _load_registry(root)
    for row in registry["jobs"]:
        if str(row["stage"]) != stage:
            continue
        path = _status_path(root, str(row["job_id"]))
        state = _read_json(path)
        if state.get("status") == "running":
            if pid_alive(
                int(state.get("pid", -1)),
                expected_start_ticks=state.get("process_start_ticks"),
            ):
                raise RuntimeError(f"Live worker still owns {row['job_id']}")
            state.update(
                status="failed",
                failure_reason="stale_running_pid_recovered",
                recovered_at_utc=utc_now(),
            )
            atomic_write_json(path, state)
    jobs = [
        dict(row)
        for row in registry["jobs"]
        if row["stage"] == stage
        and _read_json(_status_path(root, str(row["job_id"]))).get("status")
        != "completed"
    ]
    if not jobs:
        return root
    if not _worker_materialization_ready(root, jobs):
        raise RuntimeError(
            f"Stage {stage} is planned but model-specific worker configs are not materialized"
        )
    runtime = _mapping(config["runtime"], "runtime")
    benchmark_path = root / "control/benchmark.json"
    benchmark_evidence = _read_signed(
        benchmark_path, kind="architecture_window_gpu_epoch1_benchmark_v1"
    )
    if benchmark_evidence.get("status") != "passed":
        raise RuntimeError("Training launch requires a passed GPU epoch-1 benchmark")
    workers_per_gpu, peer_benchmarks = _shared_gpu_capacity(root, benchmark_evidence)
    resource_contract = {
        "schema_version": 1,
        "kind": "architecture_window_shared_gpu_slot_contract_v1",
        "global_workers_per_gpu": workers_per_gpu,
        "slot_lock_root": str(
            (root.parent / "rq3_architecture_window_shared_gpu_slots_v1").resolve()
        ),
        "current_benchmark_sha256": _sha256_file(benchmark_path),
        "peer_benchmarks": peer_benchmarks,
        "policy": "cross_root_global_flock_semaphore_v1",
    }
    resource_contract_path = root / "control/launch_resource_contract.json"
    if resource_contract_path.is_file():
        existing_contract = _read_signed(
            resource_contract_path,
            kind="architecture_window_shared_gpu_slot_contract_v1",
        )
        unsigned = {
            key: value
            for key, value in existing_contract.items()
            if key != "payload_sha256"
        }
        if unsigned != resource_contract:
            raise ValueError("Shared GPU resource contract drift")
    else:
        _write_signed(resource_contract_path, resource_contract)
    queues = {gpu: [job for job in jobs if int(job["gpu_id"]) == gpu] for gpu in (0, 1)}
    active: dict[str, tuple[subprocess.Popen[Any], Any, dict[str, Any], int, int]] = {}
    with SupervisorLock(root / "control", name=f"launch_{stage}"):
        try:
            while any(queues.values()) or active:
                for gpu in (0, 1):
                    on_gpu = sum(
                        int(record[2]["gpu_id"]) == gpu for record in active.values()
                    )
                    while on_gpu < workers_per_gpu and queues[gpu]:
                        acquired = _acquire_global_gpu_slot(
                            root, gpu_id=gpu, capacity=workers_per_gpu
                        )
                        if acquired is None:
                            break
                        slot_descriptor, global_slot = acquired
                        job = queues[gpu].pop(0)
                        state = _read_json(_status_path(root, str(job["job_id"])))
                        log = (
                            root
                            / "logs"
                            / f"{job['job_id']}.attempt_{int(state.get('attempt', 0)) + 1:02d}.log"
                        )
                        handle = log.open("a", encoding="utf-8")
                        command = [
                            str(runtime["python_executable"]),
                            "-m",
                            "scripts.rq3.news_first_vol_architecture_window_study",
                            "worker",
                            "--config",
                            str(config["source_config_path"]),
                            "--output-dir",
                            str(root),
                            "--job-id",
                            str(job["job_id"]),
                        ]
                        try:
                            # The worker wrapper retains the advisory slot if this
                            # supervisor is killed.  Releasing the parent's copy
                            # therefore cannot over-subscribe a second root while
                            # an orphaned worker is still alive.
                            os.set_inheritable(slot_descriptor, True)
                            worker_env = dict(os.environ)
                            worker_env[GLOBAL_GPU_SLOT_FD_ENV] = str(slot_descriptor)
                            process = subprocess.Popen(
                                command,
                                cwd=REPO_ROOT,
                                env=worker_env,
                                stdout=handle,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                                pass_fds=(slot_descriptor,),
                            )
                        except BaseException:
                            handle.close()
                            _release_global_gpu_slot(slot_descriptor)
                            raise
                        active[str(job["job_id"])] = (
                            process,
                            handle,
                            job,
                            slot_descriptor,
                            global_slot,
                        )
                        on_gpu += 1
                for job_id, (process, handle, _job_row, slot_descriptor, _slot) in list(
                    active.items()
                ):
                    code = process.poll()
                    if code is None:
                        continue
                    handle.close()
                    _release_global_gpu_slot(slot_descriptor)
                    del active[job_id]
                    if code != 0:
                        raise RuntimeError(f"Worker {job_id} exited {code}")
                if active or any(queues.values()):
                    time.sleep(float(runtime["poll_interval_seconds"]))
        except BaseException:
            _terminate_worker_process_groups(active)
            raise
    return root


def analyze(output_dir: str | Path) -> Path:
    root = _resolve(output_dir)
    registry = _load_registry(root)
    if not registry.get("evaluation_frozen") or not registry.get("predictions_frozen"):
        raise RuntimeError("Analysis requires frozen checkpoints and predictions")
    _verify_prediction_bundle(root, registry)
    if registry.get("analysis_complete"):
        path = Path(str(registry["analysis_manifest_path"])).resolve()
        _verify_file(
            path, str(registry["analysis_manifest_sha256"]), "analysis manifest"
        )
        return path
    module = importlib.import_module(
        "scripts.rq3.news_first_vol_architecture_window_study_analysis"
    )
    result = module.analyze(root, registry)
    registry["analysis_complete"] = True
    registry["analysis_manifest_path"] = str(result)
    registry["analysis_manifest_sha256"] = _sha256_file(result)
    _save_registry(root, registry)
    return result


def bootstrap(output_dir: str | Path) -> Path:
    root = _resolve(output_dir)
    registry = _load_registry(root)
    if not registry.get("analysis_complete"):
        raise RuntimeError("Bootstrap requires analysis")
    _verify_prediction_bundle(root, registry)
    _verify_analysis_bundle(root, registry)
    if registry.get("bootstrap_complete"):
        path = Path(str(registry["bootstrap_manifest_path"])).resolve()
        _verify_file(
            path, str(registry["bootstrap_manifest_sha256"]), "bootstrap manifest"
        )
        return path
    module = importlib.import_module(
        "scripts.rq3.news_first_vol_architecture_window_study_analysis"
    )
    result = module.bootstrap(root, registry)
    registry.update(
        bootstrap_complete=True,
        bootstrap_manifest_path=str(result),
        bootstrap_manifest_sha256=_sha256_file(result),
    )
    _save_registry(root, registry)
    return result


def report(output_dir: str | Path) -> Path:
    root = _resolve(output_dir)
    registry = _load_registry(root)
    if not registry.get("bootstrap_complete"):
        raise RuntimeError("Report requires bootstrap")
    _verify_prediction_bundle(root, registry)
    _verify_analysis_bundle(root, registry)
    _verify_bootstrap_bundle(root, registry)
    if registry.get("report_complete"):
        path = Path(str(registry["report_manifest_path"])).resolve()
        _verify_file(path, str(registry["report_manifest_sha256"]), "report manifest")
        return path
    module = importlib.import_module(
        "scripts.rq3.news_first_vol_architecture_window_study_analysis"
    )
    result = module.report(root, registry)
    registry.update(
        report_complete=True,
        report_manifest_path=str(result),
        report_manifest_sha256=_sha256_file(result),
    )
    _save_registry(root, registry)
    return result


def _checkpoint_tensor_sha256(path: Path) -> str:
    """Hash checkpoint tensor values independently of torch serialization bytes."""

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = (
        payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    )
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Checkpoint has no state dict: {path}")
    digest = hashlib.sha256()
    tensor_count = 0
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Checkpoint contains NaN/Inf: {path}:{key}")
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        # Flatten before the byte reinterpretation: ``Tensor.view(dtype)`` rejects
        # zero-dimensional tensors because they have no final dimension.  The
        # contiguous flatten is canonical for strided tensors and preserves the
        # exact byte stream (and therefore existing hashes) for every previously
        # supported non-scalar tensor.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        tensor_count += 1
    if tensor_count == 0:
        raise ValueError(f"Checkpoint has no tensors: {path}")
    return digest.hexdigest()


def _checkpoint_tensor_count(path: Path) -> int:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = (
        payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    )
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint has no state dict: {path}")
    return sum(
        int(value.numel())
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _terminate_benchmark_processes(
    active: Sequence[tuple[subprocess.Popen[Any], Any, Mapping[str, Any]]],
) -> None:
    for process, _handle, _row in active:
        if process.poll() is None:
            process.terminate()
    for process, handle, _row in active:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        handle.close()


def _benchmark_external_processes(
    snapshot: Mapping[str, Any], *, gpu_ids: Sequence[int], allowed_pids: set[int]
) -> list[dict[str, Any]]:
    uuid_to_gpu = {
        str(row["gpu_uuid"]): int(row["gpu_index"]) for row in snapshot.get("gpus", ())
    }
    return [
        dict(row)
        for row in snapshot.get("compute_processes", ())
        if uuid_to_gpu.get(str(row.get("gpu_uuid"))) in set(map(int, gpu_ids))
        and int(row.get("pid", -1)) not in allowed_pids
    ]


def _validate_benchmark_run(row: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(str(row["output_root"])).resolve()
    best_files = sorted(output_root.glob("*/metrics/best_learned_checkpoint.json"))
    if len(best_files) != 1:
        raise ValueError(f"Benchmark did not create one completed run: {output_root}")
    run_dir = best_files[0].parents[1]
    checkpoints = {
        role: run_dir / "checkpoints" / filename
        for role, filename in {
            "generator_initial": "generator_initial_epoch0.pt",
            "generator_learned": "generator_best_learned.pt",
            "critic_initial": "discriminator_initial_epoch0.pt",
            "critic_learned": "discriminator_best_learned.pt",
        }.items()
    }
    if not all(path.is_file() for path in checkpoints.values()):
        raise ValueError(f"Benchmark checkpoint set is incomplete: {run_dir}")
    tensor_hashes = {
        role: _checkpoint_tensor_sha256(path) for role, path in checkpoints.items()
    }
    parameter_counts = {
        role: _checkpoint_tensor_count(path) for role, path in checkpoints.items()
    }
    if parameter_counts["generator_learned"] != int(row["generator_parameters"]):
        raise ValueError(f"Benchmark Generator parameter count drift: {run_dir}")
    if parameter_counts["critic_learned"] != 729_157:
        raise ValueError(f"Benchmark Critic parameter count drift: {run_dir}")
    if tensor_hashes["generator_initial"] == tensor_hashes["generator_learned"]:
        raise ValueError(f"Benchmark Generator did not update: {run_dir}")
    if tensor_hashes["critic_initial"] == tensor_hashes["critic_learned"]:
        raise ValueError(f"Benchmark Critic did not update: {run_dir}")
    metrics_path = run_dir / "metrics/training_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    learned = metrics.loc[pd.to_numeric(metrics["epoch"], errors="coerce").eq(1)]
    numeric = learned.apply(pd.to_numeric, errors="coerce")
    if (
        len(learned) != 1
        or not {"val_recon", "g_recon", "g_total", "d_total", "gp"}.issubset(
            learned.columns
        )
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise ValueError(f"Benchmark metrics are incomplete/non-finite: {run_dir}")
    return {
        **dict(row),
        "run_dir": str(run_dir),
        "generator_updated": True,
        "critic_updated": True,
        "nan_or_inf_detected": False,
        "tensor_state_sha256": tensor_hashes,
        "checkpoint_parameter_counts": parameter_counts,
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256_file(metrics_path),
    }


def _benchmark_candidate(
    *,
    config: Mapping[str, Any],
    root: Path,
    workers_per_gpu: int,
    representatives: Sequence[Mapping[str, Any]],
    materialized: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one real one-epoch concurrency candidate on both physical GPUs."""

    runtime = _mapping(config["runtime"], "runtime")
    gpu_ids = tuple(map(int, runtime["gpu_ids"]))
    before = system_resource_snapshot()
    availability = gpu_availability_decision(before, gpu_ids=gpu_ids)
    if not availability["ready"]:
        raise RuntimeError(
            "GPU benchmark refused because an external/busy GPU was observed: "
            + ",".join(availability["reasons"])
        )
    attempt_parent = root / "control/benchmark_runs" / f"workers_{workers_per_gpu:02d}"
    attempt_index = len(list(attempt_parent.glob("attempt_*"))) + 1
    attempt_root = attempt_parent / f"attempt_{attempt_index:02d}"
    attempt_root.mkdir(parents=True, exist_ok=False)
    jobs: list[dict[str, Any]] = []
    required_profile_ids = [
        f"{row['generator_mode']}:{row.get('capacity_profile', 'matched')}"
        for row in representatives
    ]
    if len(required_profile_ids) != len(set(required_profile_ids)):
        raise ValueError("Benchmark representative capacity profiles are not unique")
    for gpu_position, gpu_id in enumerate(gpu_ids):
        for slot in range(workers_per_gpu):
            representative = dict(
                representatives[
                    (gpu_position * workers_per_gpu + slot) % len(representatives)
                ]
            )
            spec = materialized[str(representative["job_id"])]
            source = _mapping(
                yaml.safe_load(
                    Path(str(spec["training_config_path"])).read_text(encoding="utf-8")
                ),
                "benchmark source config",
            )
            output_root = attempt_root / f"gpu_{gpu_id}" / f"slot_{slot:02d}" / "run"
            source.update(
                num_epochs=1,
                early_stopping_min_epochs=1,
                early_stopping_patience=1,
                validation_mc_samples=int(
                    _mapping(config["training"], "training")["validation_mc_samples"]
                ),
                num_workers=0,
                seed=9_000_000 + workers_per_gpu * 1_000 + gpu_id * 100 + slot,
                output_root=str(output_root.resolve()),
                news_first_materialize_validation_loader=True,
                news_first_materialize_test_loader=False,
                save_every=1_000_000,
            )
            config_file = (
                attempt_root / "configs" / f"gpu_{gpu_id}_slot_{slot:02d}.yaml"
            )
            _atomic_write_yaml(config_file, source)
            jobs.append(
                {
                    "gpu_id": gpu_id,
                    "slot": slot,
                    "generator_mode": source["generator_conditioning_mode"],
                    "capacity_profile": str(
                        f"{representative['generator_mode']}:"
                        f"{representative.get('capacity_profile', 'matched')}"
                    ),
                    "generator_parameters": int(spec["expected_generator_parameters"]),
                    "training_config_path": str(config_file.resolve()),
                    "training_config_sha256": _sha256_file(config_file),
                    "output_root": str(output_root.resolve()),
                }
            )

    observed_profiles = {str(row["capacity_profile"]) for row in jobs}
    if not set(required_profile_ids).issubset(observed_profiles):
        raise ValueError(
            "Benchmark candidate does not cover every matched/scaled profile: "
            f"missing={sorted(set(required_profile_ids) - observed_profiles)}"
        )

    active: list[tuple[subprocess.Popen[Any], Any, Mapping[str, Any]]] = []
    snapshots: list[dict[str, Any]] = [before]
    resource_failure = ""
    try:
        for row in jobs:
            log_path = (
                attempt_root
                / "logs"
                / f"gpu_{row['gpu_id']}_slot_{row['slot']:02d}.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                (str(REPO_ROOT / "src"), str(REPO_ROOT))
            )
            env["CUDA_VISIBLE_DEVICES"] = str(row["gpu_id"])
            threads = str(runtime["cpu_threads_per_job"])
            env.update(
                {
                    name: threads
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                }
            )
            command = [
                str(runtime["python_executable"]),
                str(_resolve(str(runtime["training_entrypoint"]))),
                "--config",
                str(row["training_config_path"]),
                "--train-only",
            ]
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active.append((process, handle, {**row, "log_path": str(log_path)}))
        while any(process.poll() is None for process, _handle, _row in active):
            snapshot = system_resource_snapshot()
            snapshots.append(snapshot)
            allowed = {os.getpid(), *(process.pid for process, _handle, _row in active)}
            external = _benchmark_external_processes(
                snapshot, gpu_ids=gpu_ids, allowed_pids=allowed
            )
            if external:
                raise RuntimeError(
                    "Unapproved external GPU process appeared during benchmark"
                )
            memory_by_gpu = {
                int(row["gpu_index"]): float(row["memory_used_mib"])
                for row in snapshot["gpus"]
            }
            if any(
                memory_by_gpu.get(gpu_id, math.inf) >= 20.0 * 1024.0
                for gpu_id in gpu_ids
            ):
                resource_failure = "peak_gpu_memory_reached_20GiB"
                break
            if float(snapshot["host_memory"]["memory_used_fraction"]) >= 0.85:
                resource_failure = "host_ram_reached_85_percent"
                break
            time.sleep(max(0.25, float(runtime["poll_interval_seconds"])))
        if resource_failure:
            _terminate_benchmark_processes(active)
        else:
            for process, handle, _row in active:
                process.wait()
                handle.close()
    except BaseException:
        _terminate_benchmark_processes(active)
        raise

    logs = [Path(str(row[2]["log_path"])) for row in active]
    combined_log = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in logs
    ).lower()
    returncodes = [int(process.returncode or 0) for process, _handle, _row in active]
    capacity_tokens = (
        "out of memory",
        "cuda error: out of memory",
        "resource temporarily unavailable",
    )
    oom = any(token in combined_log for token in capacity_tokens)
    if any(code != 0 for code in returncodes) and not oom and not resource_failure:
        raise RuntimeError(
            f"Non-capacity benchmark training failure at {workers_per_gpu}/GPU: "
            f"returncodes={returncodes}"
        )
    passed = not resource_failure and not oom and all(code == 0 for code in returncodes)
    verified = [_validate_benchmark_run(row[2]) for row in active] if passed else []
    gpu_peaks = {
        str(gpu_id): max(
            float(row["memory_used_mib"])
            for snapshot in snapshots
            for row in snapshot["gpus"]
            if int(row["gpu_index"]) == gpu_id
        )
        for gpu_id in gpu_ids
    }
    host_peak = max(
        float(snapshot["host_memory"]["memory_used_fraction"]) for snapshot in snapshots
    )
    return {
        "workers_per_gpu": workers_per_gpu,
        "status": "passed" if passed else "capacity_failed",
        "resource_failure": resource_failure,
        "oom_detected": oom,
        "nan_or_inf_detected": False if passed else None,
        "returncodes": returncodes,
        "peak_gpu_memory_mib": gpu_peaks,
        "peak_host_ram_fraction": host_peak,
        "external_gpu_processes": [],
        "attempt_root": str(attempt_root.resolve()),
        "training_jobs": verified,
    }


def _architecture_scaled_benchmark_representatives(
    *,
    config: Mapping[str, Any],
    root: Path,
    materialized: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize all possible scaled leaders without consulting validation."""

    scaled = _mapping(_mapping(config["matrix"], "matrix")["scaled"], "scaled")
    search = _mapping(scaled["deterministic_search_grid"], "scaled search")
    target = int(scaled["target_generator_parameters"])
    maximum_deviation = float(scaled["maximum_relative_deviation"])
    base_spec = next(iter(materialized.values()))
    source = _mapping(
        yaml.safe_load(
            Path(str(base_spec["training_config_path"])).read_text(encoding="utf-8")
        ),
        "benchmark source config",
    )
    mode_fields = {
        "gen_crossattn_heads",
        "gen_crossattn_text_tokens",
        "gen_crossattn_dim",
        "gen_transformer_model_dim",
        "gen_transformer_layers",
        "gen_transformer_heads",
        "gen_transformer_ffn_dim",
        "gen_transformer_dropout",
        "gen_style_dim",
        "gen_style_demodulate",
    }
    lineage_fields = {
        "news_first_capacity_profile",
        "news_first_capacity_profile_sha256",
        "news_first_architecture_profile_sha256",
        "news_first_model_contract_sha256",
    }
    representatives: list[dict[str, Any]] = []
    for mode in (*ARCHITECTURE_MODES, "film_unet_mask_coords_v1"):
        winner = _selected_capacity_candidate(
            mode=mode,
            target=target,
            maximum_deviation=maximum_deviation,
            search_grid=_mapping(search[mode], f"scaled search {mode}"),
        )
        payload = deepcopy(source)
        for key in mode_fields | lineage_fields:
            payload.pop(key, None)
        capacity = {
            **winner,
            "capacity_profile": f"benchmark_scaled_{mode}",
        }
        architecture_profile = {"mode": mode, "capacity": capacity}
        model_contract = {
            "generator_mode": mode,
            "critic_mode": _mapping(config["training"], "training")[
                "critic_conditioning_mode"
            ],
            "capacity": capacity,
        }
        payload.update(
            generator_conditioning_mode=mode,
            generator_optimizer_profile="conditioning_split_lr_v2",
            generator_conditioning_learning_rate=2.5e-5,
            generator_conditioning_min_learning_rate=2.5e-6,
            generator_text_learning_rate=float(
                _mapping(config["training"], "training")["text_learning_rate"]
            ),
            generator_text_min_learning_rate=float(
                _mapping(config["training"], "training")["text_learning_rate"]
            )
            * 0.1,
            generator_film_learning_rate=0.0,
            generator_film_min_learning_rate=0.0,
            gen_base_channels=int(winner["gen_base_channels"]),
            gen_hidden_dim=int(winner["gen_hidden_dim"]),
            news_first_capacity_profile=capacity["capacity_profile"],
            news_first_capacity_profile_sha256=_payload_sha256(capacity),
            news_first_architecture_profile_sha256=_payload_sha256(
                architecture_profile
            ),
            news_first_model_contract_sha256=_payload_sha256(model_contract),
        )
        payload.update(
            {
                key: value
                for key, value in winner.items()
                if key.startswith("gen_")
                and key not in {"gen_base_channels", "gen_hidden_dim"}
            }
        )

        job_id = _job_id("benchmark", "scaled", mode)
        path = root / "control/benchmark_scaled_configs" / f"{job_id}.yaml"
        _atomic_write_yaml(path, payload)
        materialized[job_id] = {
            "job_id": job_id,
            "training_config_path": str(path.resolve()),
            "training_config_sha256": _sha256_file(path),
            "expected_generator_parameters": int(winner["generator_parameters"]),
        }
        representatives.append(
            {
                "job_id": job_id,
                "generator_mode": mode,
                "capacity_profile": capacity["capacity_profile"],
                "tolerance_minutes": 5,
                "fold": "f2_2023q2",
                "seed": 42,
            }
        )
    return representatives


def _benchmark_prepared_root(config_path: str | Path, output_dir: str | Path) -> Path:
    """Select safe concurrency using real one-epoch G/D updates on both GPUs."""

    config = load_config(config_path)
    root = _resolve(output_dir)
    registry = _load_registry(root)
    destination = root / "control/benchmark.json"
    if destination.is_file():
        existing = _read_signed(destination, kind=GPU_BENCHMARK_KIND)
        if existing.get("status") != "passed":
            raise RuntimeError("Existing benchmark is not passed")
        return destination
    stage = "screen" if registry["line"] == "architecture" else "window"
    materialize(config_path, root, stage=stage)
    manifest = _read_signed(
        root / "registry/materialized_worker_specs.json", kind=WORKER_SPEC_KIND
    )
    materialized = {str(row["job_id"]): dict(row) for row in manifest["jobs"]}
    representatives: dict[str, dict[str, Any]] = {}
    candidates_by_mode: dict[str, list[dict[str, Any]]] = {}
    for job in registry["jobs"]:
        if job["stage"] == stage:
            candidates_by_mode.setdefault(_resolved_job_mode(root, job), []).append(
                dict(job)
            )
    for mode, jobs in candidates_by_mode.items():
        representatives[mode] = max(
            jobs,
            key=lambda row: (
                int(row["tolerance_minutes"]),
                FOLDS.index(str(row["fold"])),
                int(row["seed"]),
            ),
        )
    expected_modes = 3 if registry["line"] == "architecture" else 2
    if len(representatives) != expected_modes:
        raise ValueError("Benchmark model coverage drift")
    representative_rows = list(representatives.values())
    if registry["line"] == "architecture":
        representative_rows.extend(
            _architecture_scaled_benchmark_representatives(
                config=config,
                root=root,
                materialized=materialized,
            )
        )
    required_profile_ids = [
        f"{row['generator_mode']}:{row.get('capacity_profile', 'matched')}"
        for row in representative_rows
    ]
    runtime = _mapping(config["runtime"], "runtime")
    candidates = sorted(map(int, runtime["benchmark_worker_candidates_per_gpu"]))
    attempts: list[dict[str, Any]] = []
    capacity_blocked = False
    with SupervisorLock(root / "control", name="benchmark"):
        for workers_per_gpu in candidates:
            if capacity_blocked:
                attempts.append(
                    {
                        "workers_per_gpu": workers_per_gpu,
                        "status": "safety_skipped_after_lower_capacity_failure",
                    }
                )
                continue
            attempt = _benchmark_candidate(
                config=config,
                root=root,
                workers_per_gpu=workers_per_gpu,
                representatives=representative_rows,
                materialized=materialized,
            )
            attempts.append(attempt)
            capacity_blocked = attempt["status"] != "passed"
        passed = [row for row in attempts if row["status"] == "passed"]
        if not passed:
            raise RuntimeError(
                "No benchmark concurrency candidate passed the resource gate"
            )
        selected = max(int(row["workers_per_gpu"]) for row in passed)
        return _write_signed(
            destination,
            {
                "schema_version": 1,
                "kind": GPU_BENCHMARK_KIND,
                "status": "passed",
                "candidate_workers_per_gpu": candidates,
                "selection_order": "ascending_stop_after_first_capacity_failure_v1",
                "selected_workers_per_gpu": selected,
                "benchmark_epochs": 1,
                "generator_and_critic_updated": True,
                "critic_parameters": 729_157,
                "nan_or_inf_detected": False,
                "oom_detected_in_selected_attempt": False,
                "maximum_gpu_memory_mib_exclusive": 20.0 * 1024.0,
                "maximum_host_ram_fraction_exclusive": 0.85,
                "test_loader_created": False,
                "shared_global_gpu_slot_contract": "architecture_window_global_flock_slots_v1",
                "required_benchmark_capacity_profiles": required_profile_ids,
                "attempts": attempts,
                "completed_at_utc": utc_now(),
            },
        )


def _fresh_benchmark_workspace(control: Path) -> Path:
    """Choose a never-before-written workspace while preserving failed evidence."""

    legacy = control / "benchmark_workspace"
    if not legacy.exists():
        return legacy
    attempts = control / "benchmark_attempts"
    index = 1
    while True:
        candidate = attempts / f"attempt_{index:04d}" / "benchmark_workspace"
        if not candidate.exists():
            return candidate
        index += 1


def benchmark(config_path: str | Path, output_dir: str | Path) -> Path:
    """Benchmark in a control workspace before the formal root may exist."""

    config = load_config(config_path)
    formal_root = _resolve(output_dir)
    control = formal_root.with_name(formal_root.name + "_control")
    destination = control / "benchmark.json"
    if destination.is_file():
        existing = _read_signed(destination, kind=GPU_BENCHMARK_KIND)
        if (
            existing.get("status") != "passed"
            or existing.get("source_config_sha256") != config["source_config_sha256"]
            or Path(str(existing.get("formal_output_root", ""))).resolve()
            != formal_root
        ):
            raise ValueError("External benchmark binding drift")
        workspace = Path(str(existing["benchmark_workspace"])).resolve()
        internal = workspace / "control/benchmark.json"
        _verify_file(
            internal,
            str(existing["benchmark_workspace_evidence_sha256"]),
            "benchmark workspace evidence",
        )
        workspace_registry = _load_registry(workspace)
        _verify_prepare_lineage(workspace, workspace_registry)
        return destination
    # The private workspace has the same immutable registry/materialized configs
    # but is not the formal experiment root and can never open test inputs.
    # A failed prepare can leave a valuable pre-GPU audit workspace without a
    # registry.  Never delete, rename, or resume that ambiguous directory;
    # retry in a fresh attempt namespace and bind the successful path below.
    workspace = _fresh_benchmark_workspace(control)
    workspace = prepare(
        config_path,
        workspace,
        resume=_registry_path(workspace).is_file(),
        _skip_benchmark_gate=True,
    )
    internal = _benchmark_prepared_root(config_path, workspace)
    evidence = _read_signed(internal, kind=GPU_BENCHMARK_KIND)
    return _write_signed(
        destination,
        {
            **{
                key: value for key, value in evidence.items() if key != "payload_sha256"
            },
            "source_config_sha256": config["source_config_sha256"],
            "formal_output_root": str(formal_root),
            "benchmark_workspace": str(workspace),
            "benchmark_workspace_evidence_sha256": _sha256_file(internal),
            "formal_root_created_before_benchmark_pass": False,
        },
    )


def _ensure_dual_benchmarks(config_path: str | Path, root: Path) -> None:
    """Serialize both GPU benchmarks before either formal pipeline can train."""

    if root.name not in FORMAL_ROOT_CONFIGS:
        benchmark(config_path, root)
        return
    shared = root.parent / "rq3_architecture_window_shared_gpu_slots_v1"
    shared.mkdir(parents=True, exist_ok=True)
    lock_path = shared / "benchmark_suite.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        # This lock is deliberately blocking: two independently launched
        # run-pipeline processes may arrive together, but only the first is
        # allowed to occupy both GPUs for the benchmark suite.
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        rows: list[dict[str, Any]] = []
        for basename in FORMAL_ROOT_BASENAMES:
            peer_root = root.parent / basename
            path = benchmark(FORMAL_ROOT_CONFIGS[basename], peer_root)
            rows.append(
                {
                    "formal_root": str(peer_root.resolve()),
                    "benchmark_path": str(path.resolve()),
                    "benchmark_sha256": _sha256_file(path),
                }
            )
        _write_signed(
            shared / "benchmark_suite.json",
            {
                "schema_version": 1,
                "kind": "architecture_window_dual_benchmark_suite_v1",
                "benchmarks": rows,
                "completed_at_utc": utc_now(),
            },
        )
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def qa(output_dir: str | Path) -> Path:
    root = _resolve(output_dir)
    registry = _load_registry(root)
    _verify_prepare_lineage(root, registry)
    if not all(
        bool(registry.get(key))
        for key in (
            "evaluation_frozen",
            "test_data_opened",
            "predictions_frozen",
            "analysis_complete",
            "bootstrap_complete",
            "report_complete",
        )
    ):
        raise RuntimeError(
            "Terminal QA requires frozen predictions and completed analysis"
        )
    if not _all_complete(root, registry["jobs"]):
        raise RuntimeError("Terminal QA found incomplete training jobs")
    _verify_prediction_bundle(root, registry)
    _verify_analysis_bundle(root, registry)
    _verify_bootstrap_bundle(root, registry)
    _verify_report_bundle(root, registry)
    benchmark_evidence = _read_signed(
        root / "control/benchmark.json", kind=GPU_BENCHMARK_KIND
    )
    if (
        benchmark_evidence.get("status") != "passed"
        or not benchmark_evidence.get("generator_and_critic_updated")
        or int(benchmark_evidence.get("critic_parameters", -1)) != 729_157
        or benchmark_evidence.get("nan_or_inf_detected")
        or benchmark_evidence.get("oom_detected_in_selected_attempt")
    ):
        raise ValueError("Terminal QA found invalid GPU benchmark evidence")
    expected_candidates = [4, 8, 12, 18]
    attempts = list(benchmark_evidence.get("attempts", []))
    if (
        list(map(int, benchmark_evidence.get("candidate_workers_per_gpu", [])))
        != expected_candidates
        or [int(row.get("workers_per_gpu", -1)) for row in attempts]
        != expected_candidates
    ):
        raise ValueError("Terminal QA found incomplete 4/8/12/18 benchmark evidence")
    passed_attempts = [row for row in attempts if row.get("status") == "passed"]
    if not passed_attempts or int(
        benchmark_evidence["selected_workers_per_gpu"]
    ) != max(int(row["workers_per_gpu"]) for row in passed_attempts):
        raise ValueError("Terminal QA found invalid benchmark concurrency selection")
    required_profiles = set(
        map(str, benchmark_evidence.get("required_benchmark_capacity_profiles", []))
    )
    if not required_profiles:
        raise ValueError("Terminal QA found empty benchmark model coverage")
    for attempt in passed_attempts:
        training_jobs = list(attempt.get("training_jobs", []))
        if not required_profiles.issubset(
            {str(row.get("capacity_profile")) for row in training_jobs}
        ):
            raise ValueError("Terminal QA found incomplete benchmark profile coverage")
        for benchmark_job in training_jobs:
            training_config = _mapping(
                yaml.safe_load(
                    Path(str(benchmark_job["training_config_path"])).read_text(
                        encoding="utf-8"
                    )
                ),
                "benchmark training config",
            )
            if (
                int(training_config.get("num_epochs", -1)) != 1
                or int(training_config.get("validation_mc_samples", -1)) != 16
                or bool(training_config.get("news_first_materialize_test_loader"))
            ):
                raise ValueError("Terminal QA found benchmark training-contract drift")
    core_preflight = _read_json(root / "registry/core_runtime_preflight.json")
    if int(core_preflight.get("critic_parameters", -1)) != 729_157:
        raise ValueError("Terminal QA found NoLP Critic parameter drift")
    resource_contract = _read_signed(
        root / "control/launch_resource_contract.json",
        kind="architecture_window_shared_gpu_slot_contract_v1",
    )
    if (
        int(resource_contract.get("global_workers_per_gpu", -1))
        > int(benchmark_evidence["selected_workers_per_gpu"])
        or len(resource_contract.get("peer_benchmarks", [])) != 2
    ):
        raise ValueError("Terminal QA found cross-root GPU slot contract drift")
    materialized = _read_signed(
        root / "registry/materialized_worker_specs.json", kind=WORKER_SPEC_KIND
    )
    if {str(row["job_id"]) for row in materialized["jobs"]} != {
        str(row["job_id"]) for row in registry["jobs"]
    }:
        raise ValueError("Terminal QA found incomplete materialized training configs")
    universe_manifest = _read_signed(
        root / "inputs/pair_universe_manifest.json", kind=PAIR_UNIVERSE_KIND
    )
    universe_path = Path(str(universe_manifest["universe_path"])).resolve()
    _verify_file(
        universe_path, str(universe_manifest["universe_sha256"]), "pair universe"
    )
    if registry["line"] == "window":
        universe = pd.read_csv(universe_path, dtype={"pair_id": str})
        for fold in FOLDS:
            for partition in ("train", "validation", "test"):
                sets = {
                    tolerance: set(
                        universe.loc[
                            pd.to_numeric(universe["tolerance_minutes"]).eq(tolerance)
                            & universe["fold"].astype(str).eq(fold)
                            & universe["partition"].astype(str).eq(partition),
                            "pair_id",
                        ].astype(str)
                    )
                    for tolerance in WINDOW_TOLERANCES
                }
                if any(
                    not sets[lower].issubset(sets[upper])
                    for lower, upper in _adjacent_window_tolerance_pairs()
                ):
                    raise ValueError(
                        f"Terminal alignment nesting drift: {fold}/{partition}"
                    )
    prediction_plan = _read_signed(
        root / "registry/prediction_plan.json", kind=PREDICTION_PLAN_KIND
    )
    manifest = pd.read_csv(root / "predictions/prediction_manifest.csv")
    metrics = pd.read_csv(root / "predictions/pair_metrics.csv.gz")
    required_metrics = {
        "persistence_mae",
        "persistence_skill",
        "predicted_calendar_violation_rate",
        "predicted_butterfly_violation_rate",
        "text_condition",
        "panel_role",
    }
    if (
        len(manifest) != int(prediction_plan["total_evaluation_units"])
        or manifest["evaluation_id"].astype(str).duplicated().any()
        or not required_metrics.issubset(metrics.columns)
        or metrics.empty
    ):
        raise ValueError("Terminal prediction evidence contract drift")
    qa_path = _write_signed(
        root / "qa.json",
        {
            "schema_version": 1,
            "kind": "architecture_window_terminal_qa_v1",
            "status": "passed",
            "training_jobs": len(registry["jobs"]),
            "reused_baseline_cells": len(registry["baseline_cells"]),
            "prediction_evaluation_units": len(manifest),
            "pair_metric_rows": len(metrics),
            "benchmark_candidates_per_gpu": expected_candidates,
            "selected_global_workers_per_gpu": int(
                resource_contract["global_workers_per_gpu"]
            ),
            "completed_at_utc": utc_now(),
        },
    )
    registry.update(
        terminal_complete=True,
        terminal_qa_path=str(qa_path.resolve()),
        terminal_qa_sha256=_sha256_file(qa_path),
    )
    _save_registry(root, registry)
    _write_output_sha_manifest(root)
    return qa_path


def _output_manifest_path(root: Path) -> Path:
    return root / "output_sha256.txt"


def _output_files(root: Path) -> list[Path]:
    excluded = _output_manifest_path(root).resolve()
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.resolve() != excluded
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _write_output_sha_manifest(root: Path) -> Path:
    path = _output_manifest_path(root)
    lines = [
        f"{_sha256_file(item)}  {item.relative_to(root).as_posix()}"
        for item in _output_files(root)
    ]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _verify_output_sha_manifest(root: Path) -> None:
    path = _output_manifest_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Terminal output SHA manifest is missing: {path}")
    expected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        expected[relative.strip()] = digest
    observed_paths = {
        item.relative_to(root).as_posix(): item for item in _output_files(root)
    }
    if set(expected) != set(observed_paths):
        raise ValueError("Terminal output file universe drift")
    for relative, item in observed_paths.items():
        if _sha256_file(item) != expected[relative]:
            raise ValueError(f"Terminal output SHA drift: {relative}")


def _control_output_manifest_path(control: Path) -> Path:
    return control / "control_output_sha256.txt"


def _control_output_files(control: Path) -> list[Path]:
    excluded = _control_output_manifest_path(control).resolve()
    return sorted(
        (
            path
            for path in control.rglob("*")
            if path.is_file() and path.resolve() != excluded
        ),
        key=lambda path: path.relative_to(control).as_posix(),
    )


def _write_control_output_sha_manifest(control: Path) -> Path:
    """Freeze external PID/journal/log/resource/benchmark evidence as one universe."""

    path = _control_output_manifest_path(control)
    lines = [
        f"{_sha256_file(item)}  {item.relative_to(control).as_posix()}"
        for item in _control_output_files(control)
    ]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _verify_control_output_sha_manifest(control: Path) -> None:
    path = _control_output_manifest_path(control)
    if not path.is_file():
        raise FileNotFoundError(f"Terminal control SHA manifest is missing: {path}")
    expected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        expected[relative.strip()] = digest
    observed = {
        item.relative_to(control).as_posix(): item
        for item in _control_output_files(control)
    }
    if set(expected) != set(observed):
        raise ValueError("Terminal control output file universe drift")
    for relative, item in observed.items():
        if _sha256_file(item) != expected[relative]:
            raise ValueError(f"Terminal control output SHA drift: {relative}")


def _append_resource_snapshot(control: Path, *, phase: str) -> Path:
    path = control / "resource_snapshots.json"
    snapshots: list[dict[str, Any]] = []
    if path.is_file():
        existing = _read_signed(path, kind="architecture_window_resource_snapshots_v1")
        snapshots = [dict(row) for row in existing.get("snapshots", [])]
    snapshots.append({"phase": phase, "snapshot": system_resource_snapshot()})
    return _write_signed(
        path,
        {
            "schema_version": 1,
            "kind": "architecture_window_resource_snapshots_v1",
            "snapshots": snapshots,
        },
    )


def status(
    config_path: str | Path, output_dir: str | Path | None = None
) -> dict[str, Any]:
    config = load_config(config_path)
    root = _resolve(output_dir or str(config["output_root"]))
    if not _registry_path(root).is_file():
        return {
            "state": "not_prepared",
            "output_root": str(root),
            **dry_run(config_path),
        }
    registry = _load_registry(root)
    counts = Counter()
    stages: dict[str, Counter[str]] = {}
    for job in registry["jobs"]:
        state = str(
            _read_json(_status_path(root, str(job["job_id"]))).get("status", "missing")
        )
        counts[state] += 1
        stages.setdefault(str(job["stage"]), Counter())[state] += 1
    return {
        "state": "terminal" if registry.get("terminal_complete") else "active",
        "study_kind": registry["study_kind"],
        "output_root": str(root),
        "counts": dict(counts),
        "stage_counts": {key: dict(value) for key, value in stages.items()},
        "new_training_jobs": registry["new_training_jobs"],
        "reused_baseline_cells": registry["reused_baseline_cells"],
        "logical_cells": registry["logical_cells"],
        "screen_selection_frozen": registry["screen_selection_frozen"],
        "architecture_selection_frozen": registry["architecture_selection_frozen"],
        "evaluation_frozen": registry["evaluation_frozen"],
        "test_data_opened": registry["test_data_opened"],
        "predictions_frozen": registry["predictions_frozen"],
    }


def registry(output_dir: str | Path) -> dict[str, Any]:
    """Return the hash-verified immutable registry."""

    return _load_registry(_resolve(output_dir))


def commands(output_dir: str | Path) -> Path:
    """Return the hash-verified worker-command manifest."""

    root = _resolve(output_dir)
    payload = _load_registry(root)
    path = Path(str(payload["worker_commands_path"])).resolve()
    _verify_file(path, str(payload["worker_commands_sha256"]), "worker commands")
    return path


def screen_lr(config_path: str | Path, output_dir: str | Path) -> Path:
    """Run and freeze the nine architecture learning-rate screen cells."""

    root = _resolve(output_dir)
    registry_payload = _load_registry(root)
    if registry_payload["line"] != "architecture":
        raise ValueError("screen-lr is architecture-only")
    launch_stage(config_path, root, "screen")
    return freeze_screen(root)


def launch_training(config_path: str | Path, output_dir: str | Path) -> Path:
    """Launch the next design-frozen formal training stage."""

    root = _resolve(output_dir)
    registry_payload = _load_registry(root)
    if registry_payload["line"] == "window":
        return launch_stage(config_path, root, "window")
    if not registry_payload.get("screen_selection_frozen"):
        raise RuntimeError("Run screen-lr before launch-training")
    stage = (
        "scaled" if registry_payload.get("architecture_selection_frozen") else "formal"
    )
    return launch_stage(config_path, root, stage)


def freeze_selection(output_dir: str | Path) -> Path:
    """Freeze validation-only architecture selection or static window design."""

    root = _resolve(output_dir)
    registry_payload = _load_registry(root)
    if registry_payload["line"] == "architecture":
        if not registry_payload.get("screen_selection_frozen"):
            return freeze_screen(root)
        return freeze_architecture(root)
    path = root / "registry/window_design_freeze.json"
    if registry_payload.get("window_design_frozen"):
        _verify_file(
            path,
            str(registry_payload["window_design_freeze_sha256"]),
            "window design freeze",
        )
        _read_signed(path, kind="alignment_window_static_design_freeze_v1")
        return path
    _write_signed(
        path,
        {
            "schema_version": 1,
            "kind": "alignment_window_static_design_freeze_v1",
            "tolerances_minutes": list(WINDOW_TOLERANCES),
            "model_arms": list(WINDOW_ARMS),
            "test_metrics_read": 0,
            "selection_performed": False,
            "frozen_at_utc": utc_now(),
        },
    )
    registry_payload.update(
        window_design_frozen=True,
        window_design_freeze_path=str(path.resolve()),
        window_design_freeze_sha256=_sha256_file(path),
    )
    _save_registry(root, registry_payload)
    return path


def run_pipeline(
    config_path: str | Path, output_dir: str | Path | None, *, resume: bool
) -> Path:
    config = load_config(config_path)
    root = _resolve(output_dir or str(config["output_root"]))
    if _registry_path(root).is_file():
        existing = _load_registry(root)
        if existing.get("terminal_complete"):
            if existing.get("source_config_sha256") != config["source_config_sha256"]:
                raise ValueError("Terminal source config drift")
            _verify_prepare_lineage(root, existing)
            _verify_output_sha_manifest(root)
            _verify_control_output_sha_manifest(root.with_name(root.name + "_control"))
            return root
    control = root.with_name(root.name + "_control")
    kind = str(config["study_kind"])
    with SupervisorLock(control, name="pipeline"):
        _append_resource_snapshot(control, phase="pipeline_start")
        append_stage_journal(
            control,
            experiment_kind=kind,
            stage="pipeline",
            status="running",
            details={"output_root": str(root), "resume": bool(resume)},
        )
        try:
            _ensure_dual_benchmarks(config_path, root)
            append_stage_journal(
                control, experiment_kind=kind, stage="benchmark", status="completed"
            )
            _append_resource_snapshot(control, phase="post_benchmark")
            root = prepare(config_path, output_dir, resume=resume)
            append_stage_journal(
                control, experiment_kind=kind, stage="prepare", status="completed"
            )
            registry = _load_registry(root)
            if registry["line"] == "architecture":
                launch_stage(config_path, root, "screen")
                freeze_screen(root)
                launch_stage(config_path, root, "formal")
                freeze_architecture(root)
                launch_stage(config_path, root, "scaled")
            else:
                freeze_selection(root)
                launch_stage(config_path, root, "window")
            append_stage_journal(
                control, experiment_kind=kind, stage="training", status="completed"
            )
            _append_resource_snapshot(control, phase="post_training")
            freeze_evaluation(root)
            predict(config_path, root, resume=resume)
            analyze(root)
            bootstrap(root)
            report(root)
            qa(root)
            append_stage_journal(
                control, experiment_kind=kind, stage="pipeline", status="completed"
            )
            _append_resource_snapshot(control, phase="pipeline_complete")
            # The external control journal is intentionally outside the formal
            # root; regenerating here catches every final in-root mutation.
            _write_output_sha_manifest(root)
            _write_control_output_sha_manifest(control)
            return root
        except BaseException as exc:
            _append_resource_snapshot(control, phase="pipeline_failed")
            append_stage_journal(
                control,
                experiment_kind=kind,
                stage="pipeline",
                status="failed",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "benchmark",
            "prepare",
            "dry-run",
            "materialize",
            "worker",
            "launch-screen",
            "screen-lr",
            "launch-training",
            "freeze-screen",
            "launch-formal",
            "freeze-architecture",
            "freeze-selection",
            "launch-scaled",
            "launch-window",
            "freeze-evaluation",
            "predict",
            "analyze",
            "bootstrap",
            "report",
            "qa",
            "status",
            "registry",
            "commands",
            "run-pipeline",
        ),
    )
    parser.add_argument("--config", default=DEFAULT_ARCHITECTURE_CONFIG)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--resume", action="store_true")
    return parser


def _is_terminal_root(root: Path) -> bool:
    return _registry_path(root).is_file() and bool(
        _load_registry(root).get("terminal_complete")
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    output = args.output_dir or str(config["output_root"])
    output_root = _resolve(output)
    entered_terminal = args.action == "run-pipeline" and _is_terminal_root(output_root)
    if args.action == "dry-run":
        result: Any = dry_run(args.config)
    elif args.action == "benchmark":
        result = benchmark(args.config, output)
    elif args.action == "prepare":
        result = prepare(args.config, output, resume=args.resume)
    elif args.action == "materialize":
        result = materialize(args.config, output, stage=args.stage or None)
    elif args.action == "worker":
        if not args.job_id:
            raise SystemExit("worker requires --job-id")
        result = worker(args.config, output, args.job_id)
    elif args.action == "screen-lr":
        result = screen_lr(args.config, output)
    elif args.action == "launch-training":
        result = launch_training(args.config, output)
    elif args.action.startswith("launch-"):
        result = launch_stage(args.config, output, args.action.removeprefix("launch-"))
    elif args.action == "freeze-screen":
        result = freeze_screen(output)
    elif args.action == "freeze-architecture":
        result = freeze_architecture(output)
    elif args.action == "freeze-selection":
        result = freeze_selection(output)
    elif args.action == "freeze-evaluation":
        result = freeze_evaluation(output)
    elif args.action == "predict":
        result = predict(args.config, output, resume=args.resume)
    elif args.action == "analyze":
        result = analyze(output)
    elif args.action == "bootstrap":
        result = bootstrap(output)
    elif args.action == "report":
        result = report(output)
    elif args.action == "qa":
        result = qa(output)
    elif args.action == "status":
        result = status(args.config, output)
    elif args.action == "registry":
        result = registry(output)
    elif args.action == "commands":
        result = commands(output)
    else:
        result = run_pipeline(args.config, output, resume=args.resume)
    if isinstance(result, Mapping):
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    else:
        print(result, flush=True)
    if (
        args.action == "run-pipeline"
        and not entered_terminal
        and _is_terminal_root(output_root)
    ):
        # The final stdout line is part of the externally redirected pipeline.log.
        # Freeze the control universe only after that line has been flushed.  A
        # later invocation that entered terminal state remains strictly read-only.
        _write_control_output_sha_manifest(
            output_root.with_name(output_root.name + "_control")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURE_KIND",
    "WINDOW_KIND",
    "bind_baselines",
    "dry_run",
    "freeze_architecture",
    "freeze_evaluation",
    "freeze_screen",
    "input_preflight",
    "load_config",
    "planned_cells",
    "prepare",
    "status",
    "worker",
]
