"""Two-stage single-seed Generator-FiLM by Critic-text factorial experiment.

The branch-local protocol has three deliberately separate phases.  Development
jobs train through 2023Q2 and use Q3 for checkpoint/architecture selection.
Refit jobs then replay the frozen per-epoch learning-rate recipes while using
all observations before Q4 and materializing no validation or test loader.
Only the explicit ``evaluate-q4`` action may create the Q4 workbook, loader, or
predictions.  Q4 remains exploratory because it was inspected by earlier work.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import yaml

from scripts.rq3 import news_first_vol_training as training
from wgan_option.utils.text_ablation import (
    REAL_TEXT,
    TEXT_SHUFFLE,
    text_information_path,
)


ROOT_KEY = training.ROOT_KEY
REPO_ROOT = training.REPO_ROOT
EXPERIMENT_KIND = "generator_film_critic_factorial_q4_refit"
DEFAULT_CONFIG = "configs/rq3/news_first_vol_generator_film_critic_factorial.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq3_news_first_vol_generator_film_critic_factorial_exact_ttm_seed42_v1"
)
DEVELOPMENT_STAGE = "development_q3_selection"
REFIT_STAGE = "full_history_refit"
BENCHMARK_STAGE = "benchmark"
REFIT_MODE = "frozen_epoch_lr_replay_v1"
GENERATOR_MODES = (
    "bottleneck_concat_v1",
    "film_conv_bottleneck_concat_v1",
)
CRITIC_MODES = ("lp_concat_v1", "lp_disabled_same_shape_v1")
TEXT_MODES = (REAL_TEXT, TEXT_SHUFFLE)
TOLERANCES = (5, 30)
SEED = 42
EXPECTED_REFIT_COUNTS = {
    5: {"rows": 1116, "pairs": 883, "sessions": 236},
    30: {"rows": 1897, "pairs": 1269, "sessions": 278},
}
EXPECTED_DEVELOPMENT_COUNTS = {
    5: {"rows": 968, "pairs": 748, "sessions": 203},
    30: {"rows": 1659, "pairs": 1083, "sessions": 237},
}
EXPECTED_Q3_COUNTS = {
    5: {"rows": 148, "pairs": 135, "sessions": 33},
    30: {"rows": 238, "pairs": 186, "sessions": 41},
}
EXPECTED_Q4_COMMON_COUNTS = {"rows": 167, "pairs": 143, "sessions": 45}
EXACT_TTM_STRIKE_GRID = tuple(0.97 + 0.004 * index for index in range(16))
EXACT_TTM_MATURITY_GRID = (1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38)
EXPECTED_WGAN_PARAMETERS = {
    GENERATOR_MODES[0]: 149_333,
    GENERATOR_MODES[1]: 150_285,
}
SUCCESS_STATUSES = {"completed", "dry_run_passed"}
DEVELOPMENT_ARTIFACT_ROLES = (
    "generator_initial_epoch0",
    "discriminator_initial_epoch0",
    "generator_best",
    "discriminator_best",
    "generator_best_learned",
    "discriminator_best_learned",
    "generator_final",
    "discriminator_final",
    "training_metrics_csv",
    "training_metrics_json",
    "best_checkpoint",
    "initial_checkpoint",
    "best_learned_checkpoint",
    "resolved_training_config",
    "run_log",
)
REFIT_ARTIFACT_ROLES = (
    "generator_final",
    "discriminator_final",
    "training_metrics_csv",
    "training_metrics_json",
    "resolved_training_config",
    "run_log",
)

_read_json = training._read_json
_write_json = training._write_json
_write_yaml = training._write_yaml
_write_csv = training._write_csv
_sha256_file = training._sha256_file
_payload_sha256 = training._payload_sha256
_atomic_write_text = training._atomic_write_text
_utc_now = training._utc_now
_require_mapping = training._require_mapping
_resolve_repo_path = training._resolve_repo_path
_job_status_path = training._job_status_path


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    return _require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)


def _factorial(resolved: Mapping[str, Any]) -> dict[str, Any]:
    return _require_mapping(
        resolved.get("generator_film_critic_factorial"),
        "generator_film_critic_factorial",
    )


def _exact_float(value: Any, expected: float, label: str) -> None:
    observed = float(value)
    if not math.isfinite(observed) or observed != float(expected):
        raise ValueError(f"{label} is frozen to {expected}; observed={observed}")


def _surface_grid_contract() -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "support_method": "raw_bracket_intersection_v1",
        "strike_grid": [float(value) for value in EXACT_TTM_STRIKE_GRID],
        "maturity_days_grid": [int(value) for value in EXACT_TTM_MATURITY_GRID],
        "surface_shape": [16, 16],
    }
    return {
        **payload,
        "surface_grid_profile": "exact_ttm_16x16_v1",
        "surface_grid_sha256": _payload_sha256(payload),
    }


def _architecture_contract() -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "capacity_profile": "small",
        "gen_base_channels": 4,
        "gen_res_blocks": 0,
        "gen_text_hidden_dim": 32,
        "gen_text_out_dim": 16,
        "gen_hidden_dim": 128,
        "disc_base_channels": 4,
        "disc_res_blocks": 0,
        "disc_text_hidden_dim": 16,
        "disc_hidden_dim": 96,
    }
    return {**payload, "architecture_profile_sha256": _payload_sha256(payload)}


def _validate_config(resolved: Mapping[str, Any]) -> None:
    datasets = _require_mapping(resolved.get("datasets"), "datasets")
    split = _require_mapping(resolved.get("split"), "split")
    factorial = _factorial(resolved)
    models = _require_mapping(resolved.get("models"), "models")
    runtime = _require_mapping(resolved.get("runtime"), "runtime")
    if str(factorial.get("experiment_kind")) != EXPERIMENT_KIND:
        raise ValueError(f"experiment_kind must be {EXPERIMENT_KIND}")
    if not bool(factorial.get("enabled")):
        raise ValueError("The factorial experiment must be enabled")
    if tuple(map(int, datasets.get("tolerances_minutes", ()))) != TOLERANCES:
        raise ValueError("Only 5m and 30m datasets are permitted")
    if int(datasets.get("common_evaluation_tolerance_minutes", -1)) != 5:
        raise ValueError("The common evaluation panel is frozen to 5m")
    if str(datasets.get("text_embedding_mode", "")).lower() != "lp":
        raise ValueError("LP embeddings are frozen")
    if str(datasets.get("support_mask_mode", "")).lower() != "raw_joint":
        raise ValueError("raw_joint support is frozen")
    if tuple(datasets.get("text_ablation_modes", ())) != TEXT_MODES:
        raise ValueError("Dataset text modes must be real_text/text_shuffle")
    if int(datasets.get("seed", -1)) != SEED:
        raise ValueError("The experiment is single-seed seed=42")
    frozen_dates = {
        "development_train_end_utc": "2023-07-01T00:00:00Z",
        "development_validation_end_utc": "2023-10-01T00:00:00Z",
        "refit_train_end_utc": "2023-10-01T00:00:00Z",
        "q4_start_utc": "2023-10-01T00:00:00Z",
        "q4_end_utc": "2024-01-01T00:00:00Z",
    }
    for key, expected in frozen_dates.items():
        if str(split.get(key)) != expected:
            raise ValueError(f"split.{key} is frozen to {expected}")
    if int(split.get("validation_mc_samples", -1)) != 16:
        raise ValueError("Development validation MC is frozen to 16")
    if int(split.get("q4_mc_samples", -1)) != 64:
        raise ValueError("Q4 evaluation MC is frozen to 64")
    if tuple(factorial.get("generator_conditioning_modes", ())) != GENERATOR_MODES:
        raise ValueError(f"Generator modes/order must be {GENERATOR_MODES}")
    if tuple(factorial.get("critic_conditioning_modes", ())) != CRITIC_MODES:
        raise ValueError(f"Critic modes/order must be {CRITIC_MODES}")
    if tuple(factorial.get("text_ablation_modes", ())) != TEXT_MODES:
        raise ValueError(f"Text modes/order must be {TEXT_MODES}")
    if tuple(map(int, factorial.get("tolerances_minutes", ()))) != TOLERANCES:
        raise ValueError("Factorial tolerances are frozen to 5m/30m")
    if int(factorial.get("seed", -1)) != SEED:
        raise ValueError("Factorial seed is frozen to 42")
    if str(factorial.get("capacity_profile")) != "small":
        raise ValueError("Capacity is frozen to Small")
    if str(factorial.get("lr_profile")) != "lr_5e_07":
        raise ValueError("LR profile is frozen to lr_5e_07")
    if str(factorial.get("refit_mode")) != REFIT_MODE:
        raise ValueError(f"Refit mode must be {REFIT_MODE}")
    grid = _surface_grid_contract()
    if str(factorial.get("surface_grid_profile")) != grid["surface_grid_profile"]:
        raise ValueError("Exact-TTM surface grid profile drifted")
    configured_strikes = [float(value) for value in factorial.get("strike_grid", ())]
    if len(configured_strikes) != 16 or any(
        abs(actual - expected) > 1e-12
        for actual, expected in zip(configured_strikes, grid["strike_grid"])
    ):
        raise ValueError("Exact-TTM strike grid drifted")
    if [int(value) for value in factorial.get("maturity_days_grid", ())] != list(
        EXACT_TTM_MATURITY_GRID
    ):
        raise ValueError("Exact-TTM maturity grid drifted")
    configured_parameters = {
        str(key): int(value)
        for key, value in _require_mapping(
            factorial.get("expected_wgan_parameters_by_generator_mode"),
            "expected_wgan_parameters_by_generator_mode",
        ).items()
    }
    if configured_parameters != EXPECTED_WGAN_PARAMETERS:
        raise ValueError("Mode-specific Small parameter counts drifted")
    configured_refit = {
        int(key): {name: int(value) for name, value in raw.items()}
        for key, raw in _require_mapping(
            factorial.get("expected_refit_counts"), "expected_refit_counts"
        ).items()
    }
    if configured_refit != EXPECTED_REFIT_COUNTS:
        raise ValueError("Frozen refit data counts drifted")
    for key, expected in (
        ("expected_development_counts", EXPECTED_DEVELOPMENT_COUNTS),
        ("expected_q3_counts", EXPECTED_Q3_COUNTS),
    ):
        configured = {
            int(tolerance): {name: int(value) for name, value in raw.items()}
            for tolerance, raw in _require_mapping(factorial.get(key), key).items()
        }
        if configured != expected:
            raise ValueError(f"Frozen {key} drifted")
    configured_q4 = {
        name: int(value)
        for name, value in _require_mapping(
            factorial.get("expected_q4_common_5m_counts"),
            "expected_q4_common_5m_counts",
        ).items()
    }
    if configured_q4 != EXPECTED_Q4_COMMON_COUNTS:
        raise ValueError("Frozen Q4 common-panel counts drifted")
    if set(models) != {"wgan"}:
        raise ValueError("Only WGAN is permitted")
    values = _require_mapping(models["wgan"].get("training"), "wgan.training")
    frozen_ints = {
        "embedding_dim": 1024,
        "noise_dim": 32,
        "gen_base_channels": 4,
        "gen_res_blocks": 0,
        "gen_text_hidden_dim": 32,
        "gen_text_out_dim": 16,
        "gen_hidden_dim": 128,
        "disc_base_channels": 4,
        "disc_res_blocks": 0,
        "disc_text_hidden_dim": 16,
        "disc_hidden_dim": 96,
        "batch_size": 16,
        "discriminator_iter": 5,
        "num_epochs": 100,
        "reduce_lr_patience": 3,
        "early_stopping_patience": 16,
        "early_stopping_min_epochs": 30,
    }
    for key, expected in frozen_ints.items():
        if int(values.get(key, -1)) != expected:
            raise ValueError(f"WGAN {key} is frozen to {expected}")
    frozen_strings = {
        "generator_noise_mode": "gaussian",
        "generator_current_input_mode": "current_support_masked",
        "generator_conditioning_mode": GENERATOR_MODES[0],
        "critic_conditioning_mode": CRITIC_MODES[0],
        "critic_normalization_mode": "legacy_instance_norm_v1",
        "residual_output_mode": "identity_softplus_residual",
        "news_first_label_reliability_mode": "none",
        "news_first_refit_mode": "none",
    }
    for key, expected in frozen_strings.items():
        if str(values.get(key, "")).lower() != expected:
            raise ValueError(f"WGAN {key} is frozen to {expected}")
    for key, expected in {
        "learning_rate": 5e-7,
        "generator_learning_rate": 5e-7,
        "discriminator_learning_rate": 5e-7,
        "reduce_lr_min_lr": 5e-8,
        "reduce_lr_factor": 0.5,
        "lambda_gp": 10.0,
        "lambda_recon": 10.0,
        "lambda_calendar": 2.0,
        "lambda_butterfly": 2.0,
        "lambda_smooth": 0.1,
        "lambda_delta_shrink": 0.0,
    }.items():
        _exact_float(values.get(key), expected, key)
    for key in (
        "use_reduce_lr_on_plateau",
        "evaluate_initial_checkpoint",
        "use_early_stopping",
        "news_first_materialize_validation_loader",
    ):
        if not bool(values.get(key)):
            raise ValueError(f"Development WGAN requires {key}=true")
    if bool(values.get("news_first_materialize_test_loader")):
        raise ValueError("Training-stage test loader must remain disabled")
    gpu_ids = tuple(map(int, runtime.get("gpu_ids", ())))
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("Exactly two distinct GPUs are required")
    if int(runtime.get("benchmark_workers_per_gpu", -1)) != 8:
        raise ValueError("Benchmark concurrency is frozen to 8 workers/GPU")
    if int(runtime.get("fallback_slots_per_gpu", -1)) != 4:
        raise ValueError("Fallback concurrency is frozen to 4 workers/GPU")
    if int(runtime.get("slots_per_gpu", -1)) not in {4, 8}:
        raise ValueError("Formal concurrency must be 4 or 8 workers/GPU")


def resolve_config(config_path: str | Path) -> dict[str, Any]:
    source = _resolve_repo_path(config_path)
    root = _load_yaml(source, "factorial config")
    resolved = deepcopy(_require_mapping(root.get(ROOT_KEY), ROOT_KEY))
    resolved["datasets"]["root"] = str(_resolve_repo_path(resolved["datasets"]["root"]))
    resolved["runtime"]["python_executable"] = str(
        _resolve_repo_path(resolved["runtime"].get("python_executable", sys.executable))
    )
    resolved["source_config_path"] = str(source)
    _validate_config(resolved)
    return resolved


def factorial_specs(stage: str = DEVELOPMENT_STAGE) -> list[dict[str, Any]]:
    normalized = str(stage).strip().lower()
    if normalized not in {DEVELOPMENT_STAGE, REFIT_STAGE, BENCHMARK_STAGE}:
        raise ValueError(f"Unknown factorial stage: {stage}")
    return [
        {
            "experiment_stage": normalized,
            "generator_conditioning_mode": generator_mode,
            "critic_conditioning_mode": critic_mode,
            "text_ablation_mode": text_mode,
            "tolerance_minutes": tolerance,
            "seed": SEED,
        }
        for generator_mode in GENERATOR_MODES
        for critic_mode in CRITIC_MODES
        for text_mode in TEXT_MODES
        for tolerance in TOLERANCES
    ]


def _balanced_assignments(
    specs: Sequence[Mapping[str, Any]],
    *,
    gpu_ids: Sequence[int],
    slots_per_gpu: int,
    invert_gpu: bool = False,
) -> list[dict[str, Any]]:
    if len(gpu_ids) != 2 or int(slots_per_gpu) <= 0:
        raise ValueError("Balanced assignment requires two GPUs and positive slots")
    index_by_generator = {value: index for index, value in enumerate(GENERATOR_MODES)}
    index_by_critic = {value: index for index, value in enumerate(CRITIC_MODES)}
    index_by_text = {value: index for index, value in enumerate(TEXT_MODES)}
    index_by_tolerance = {value: index for index, value in enumerate(TOLERANCES)}
    per_gpu_count = {int(gpu): 0 for gpu in gpu_ids}
    assigned: list[dict[str, Any]] = []
    for raw in specs:
        spec = dict(raw)
        parity = (
            index_by_generator[str(spec["generator_conditioning_mode"])]
            + index_by_critic[str(spec["critic_conditioning_mode"])]
            + index_by_text[str(spec["text_ablation_mode"])]
            + index_by_tolerance[int(spec["tolerance_minutes"])]
            + int(bool(invert_gpu))
        ) % 2
        gpu_id = int(gpu_ids[parity])
        local_index = per_gpu_count[gpu_id]
        per_gpu_count[gpu_id] += 1
        spec.update(
            {
                "gpu_id": gpu_id,
                "gpu_slot": local_index % int(slots_per_gpu),
                "wave": local_index // int(slots_per_gpu) + 1,
            }
        )
        assigned.append(spec)
    _assert_gpu_balance(assigned, gpu_ids=gpu_ids)
    return assigned


def _assert_gpu_balance(
    rows: Sequence[Mapping[str, Any]], *, gpu_ids: Sequence[int]
) -> None:
    for factor in (
        "generator_conditioning_mode",
        "critic_conditioning_mode",
        "text_ablation_mode",
        "tolerance_minutes",
    ):
        values = {row[factor] for row in rows}
        for value in values:
            counts = {
                int(gpu): sum(
                    row[factor] == value and int(row["gpu_id"]) == int(gpu)
                    for row in rows
                )
                for gpu in gpu_ids
            }
            if len(set(counts.values())) != 1:
                raise ValueError(f"GPU imbalance for {factor}={value}: {counts}")


def _mode_slug(value: str) -> str:
    return {
        GENERATOR_MODES[0]: "gconcat",
        GENERATOR_MODES[1]: "gfilm",
        CRITIC_MODES[0]: "dlp",
        CRITIC_MODES[1]: "doff",
        REAL_TEXT: "real",
        TEXT_SHUFFLE: "shuffle",
    }[str(value)]


def _job_id(stage: str, spec: Mapping[str, Any]) -> str:
    prefix = {
        DEVELOPMENT_STAGE: "dev",
        REFIT_STAGE: "refit",
        BENCHMARK_STAGE: "benchmark",
    }[str(stage)]
    return (
        f"{prefix}_{_mode_slug(str(spec['generator_conditioning_mode']))}_"
        f"{_mode_slug(str(spec['critic_conditioning_mode']))}_"
        f"{_mode_slug(str(spec['text_ablation_mode']))}_"
        f"{int(spec['tolerance_minutes']):02d}m_seed_{int(spec['seed']):03d}"
    )


def _dataset_path(resolved: Mapping[str, Any], tolerance: int) -> Path:
    datasets = resolved["datasets"]
    return Path(datasets["root"]) / str(datasets["workbook_template"]).format(
        tolerance=int(tolerance), tolerance02=f"{int(tolerance):02d}"
    )


def _counts(frame: Any) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "pairs": int(frame["pair_id"].nunique()),
        "sessions": int(frame["session_id"].nunique()),
    }


def _supported_frame(
    resolved: Mapping[str, Any], workbook: Path, tolerance: int
) -> Any:
    frame = training._read_split_keys(workbook, str(resolved["datasets"]["sheet_name"]))
    return training._apply_support_eligibility(
        frame,
        dataset_root=Path(resolved["datasets"]["root"]),
        tolerance=int(tolerance),
        support_mode="raw_joint",
    )


def _verify_exact_ttm_grid(frame: Any, *, label: str) -> None:
    expected = _surface_grid_contract()
    columns = {
        "strike_grid": expected["strike_grid"],
        "maturity_days_grid": expected["maturity_days_grid"],
        "surface_shape": expected["surface_shape"],
    }
    for column, frozen in columns.items():
        if column not in frame:
            raise ValueError(f"{label} is missing {column}")
        values = frame[column].dropna().astype(str).unique().tolist()
        if len(values) != 1:
            raise ValueError(f"{label} contains multiple {column} contracts")
        observed = yaml.safe_load(values[0])
        if column == "strike_grid":
            if len(observed) != len(frozen) or any(
                abs(float(actual) - float(expected_value)) > 1e-12
                for actual, expected_value in zip(observed, frozen)
            ):
                raise ValueError(f"{label} exact-TTM strike grid drift")
        elif [int(value) for value in observed] != [int(value) for value in frozen]:
            raise ValueError(f"{label} exact-TTM {column} drift")


def _materialize_pre_q4_windows(
    resolved: Mapping[str, Any], root: Path
) -> tuple[list[dict[str, Any]], Path]:
    """Scan source sheets but write only rows with origin strictly before Q4."""

    import pandas as pd

    cutoff = pd.Timestamp(resolved["split"]["refit_train_end_utc"])
    directory = root / "data_windows" / "pre_q4"
    directory.mkdir(parents=True, exist_ok=True)
    sheet = str(resolved["datasets"]["sheet_name"])
    rows: list[dict[str, Any]] = []
    source_q4_counts: dict[str, int] = {}
    for tolerance in TOLERANCES:
        source = _dataset_path(resolved, tolerance)
        frame = pd.read_excel(source, sheet_name=sheet)
        _verify_exact_ttm_grid(frame, label=f"{tolerance}m source workbook")
        timestamps = pd.to_datetime(
            frame["effective_origin_utc"], errors="coerce", utc=True
        )
        if timestamps.isna().any():
            raise ValueError(f"Invalid effective_origin_utc in {source}")
        selected = frame.loc[timestamps < cutoff].copy()
        source_q4_counts[str(tolerance)] = int((timestamps >= cutoff).sum())
        if selected.empty or bool(
            (pd.to_datetime(selected["effective_origin_utc"], utc=True) >= cutoff).any()
        ):
            raise ValueError(f"Failed to exclude Q4 from {tolerance}m window")
        target = directory / f"tolerance_{tolerance:02d}m_pre_q4.xlsx"
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            selected.to_excel(writer, sheet_name=sheet, index=False)
        supported = _supported_frame(resolved, target, tolerance)
        observed = _counts(supported)
        if observed != EXPECTED_REFIT_COUNTS[tolerance]:
            raise ValueError(
                f"Frozen {tolerance}m refit counts drifted: "
                f"{observed} != {EXPECTED_REFIT_COUNTS[tolerance]}"
            )
        rows.append(
            {
                "source_role": f"materialized_pre_q4_{tolerance:02d}m",
                "path": str(target.resolve()),
                "size_bytes": target.stat().st_size,
                "sha256": _sha256_file(target),
            }
        )
    manifest = {
        "schema_version": 1,
        "source_sheet_scanned_for_window_materialization": True,
        "origin_end_utc_exclusive": str(resolved["split"]["refit_train_end_utc"]),
        "source_rows_at_or_after_q4_start": source_q4_counts,
        "q4_window_materialized": False,
        "q4_sample_objects_materialized": False,
        "q4_loader_created": False,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "workbooks": rows,
    }
    manifest["payload_sha256"] = _payload_sha256(manifest)
    manifest_path = _write_json(
        root / "data_windows" / "pre_q4_window_manifest.json", manifest
    )
    rows.append(
        {
            "source_role": "pre_q4_window_manifest",
            "path": str(manifest_path.resolve()),
            "size_bytes": manifest_path.stat().st_size,
            "sha256": _sha256_file(manifest_path),
        }
    )
    return rows, manifest_path


def _write_split_manifest(resolved: Mapping[str, Any], root: Path) -> Path:
    import pandas as pd

    train_end = pd.Timestamp(resolved["split"]["development_train_end_utc"])
    validation_end = pd.Timestamp(resolved["split"]["development_validation_end_utc"])
    rows: list[dict[str, Any]] = []
    for tolerance in TOLERANCES:
        workbook = (
            root / "data_windows" / "pre_q4" / f"tolerance_{tolerance:02d}m_pre_q4.xlsx"
        )
        frame = _supported_frame(resolved, workbook, tolerance)
        partitions = {
            "development_train": frame[frame["effective_origin_utc"] < train_end],
            "development_q3_validation": frame[
                (frame["effective_origin_utc"] >= train_end)
                & (frame["effective_origin_utc"] < validation_end)
            ],
            "full_history_refit": frame[frame["effective_origin_utc"] < validation_end],
        }
        train = partitions["development_train"]
        validation = partitions["development_q3_validation"]
        if _counts(train) != EXPECTED_DEVELOPMENT_COUNTS[tolerance]:
            raise ValueError(f"Development split counts drifted for {tolerance}m")
        if _counts(validation) != EXPECTED_Q3_COUNTS[tolerance]:
            raise ValueError(f"Q3 validation split counts drifted for {tolerance}m")
        if set(train["pair_id"]) & set(validation["pair_id"]):
            raise ValueError(f"Development pair leakage for {tolerance}m")
        if set(train["session_id"]) & set(validation["session_id"]):
            raise ValueError(f"Development session leakage for {tolerance}m")
        if (
            _counts(partitions["full_history_refit"])
            != EXPECTED_REFIT_COUNTS[tolerance]
        ):
            raise ValueError(f"Refit split counts drifted for {tolerance}m")
        for name, selected in partitions.items():
            rows.append(
                {
                    "split": name,
                    "tolerance_minutes": tolerance,
                    **_counts(selected),
                    "pair_universe_sha256": _payload_sha256(
                        sorted(set(selected["pair_id"]))
                    ),
                    "session_universe_sha256": _payload_sha256(
                        sorted(set(selected["session_id"]))
                    ),
                    "q4_window_materialized": False,
                    "q4_loader_created": False,
                }
            )
    return _write_csv(root / "rolling_split_manifest.csv", rows, tuple(rows[0]))


def _source_rows(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset_root = Path(resolved["datasets"]["root"])
    declaration_path = dataset_root / "dataset_output_sha256.txt"
    declared = training._declared_dataset_hashes(dataset_root)
    if not declaration_path.is_file() or not declared:
        raise FileNotFoundError(
            f"Missing dataset output hash declaration: {declaration_path}"
        )
    candidates: list[tuple[str, Path]] = [
        ("dataset_output_manifest", declaration_path),
        ("dataset_summary", dataset_root / "dataset_summary.csv"),
        ("orchestration_config", Path(resolved["source_config_path"])),
    ]
    for tolerance in TOLERANCES:
        candidates.extend(
            [
                (
                    f"source_workbook_{tolerance:02d}m",
                    _dataset_path(resolved, tolerance),
                ),
                (
                    f"support_audit_{tolerance:02d}m",
                    dataset_root
                    / f"tolerance_{tolerance:02d}m"
                    / "surface_support_audit.csv.gz",
                ),
            ]
        )
    rows: list[dict[str, Any]] = []
    for role, path in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"Missing factorial source {role}: {path}")
        actual = _sha256_file(path)
        if path != declaration_path and path.is_relative_to(dataset_root):
            relative = path.relative_to(dataset_root).as_posix()
            expected = declared.get(relative)
            if expected is None:
                raise ValueError(
                    f"Dataset hash declaration is missing source: {relative}"
                )
            if actual.lower() != expected.lower():
                raise ValueError(
                    f"Dataset source hash mismatch for {relative}: "
                    f"declared={expected}, actual={actual}"
                )
        rows.append(
            {
                "source_role": role,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    return rows


def _code_rows() -> list[dict[str, Any]]:
    rows = training._code_rows()
    known = {row["relative_path"] for row in rows}
    factorial_modules = (
        "scripts/rq3/news_first_vol_generator_film_critic_factorial.py",
        "scripts/rq3/news_first_vol_generator_film_critic_factorial_analysis.py",
        "scripts/rq3/news_first_vol_generator_film_critic_factorial_report.py",
    )
    for relative in factorial_modules:
        if not (REPO_ROOT / relative).is_file():
            raise FileNotFoundError(
                f"Factorial code lineage is incomplete; missing {relative}"
            )
    for relative in (
        *factorial_modules,
        "scripts/rq3/main.py",
        "src/wgan_option/config.py",
        "src/wgan_option/models/generator.py",
        "src/wgan_option/models/discriminator.py",
        "src/wgan_option/models/gan_model.py",
        "src/wgan_option/utils/news_first_dataloaders.py",
        "src/wgan_option/utils/text_ablation.py",
    ):
        path = REPO_ROOT / relative
        if path.is_file() and relative not in known:
            rows.append(
                {
                    "relative_path": relative,
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return rows


def _write_hash_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"Cannot write empty hash manifest: {path}")
    return _write_csv(path, rows, tuple(rows[0]))


def _verify_hash_manifest(path: Path, *, path_field: str = "path") -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty hash manifest: {path}")
    for row in rows:
        target = Path(row[path_field])
        if not target.is_file() or _sha256_file(target) != row["sha256"]:
            raise ValueError(f"Hash drift in {path.name}: {target}")


def _conditioning_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    from wgan_option.models.common import (
        critic_conditioning_fingerprint,
        generator_conditioning_fingerprint,
    )

    generator_mode = str(spec["generator_conditioning_mode"])
    critic_mode = str(spec["critic_conditioning_mode"])
    grid = _surface_grid_contract()
    architecture = _architecture_contract()
    conditioning = {
        "schema_version": 1,
        "generator_conditioning_mode": generator_mode,
        "generator_conditioning_fingerprint": generator_conditioning_fingerprint(
            generator_mode
        ),
        "critic_conditioning_mode": critic_mode,
        "critic_conditioning_fingerprint": critic_conditioning_fingerprint(critic_mode),
    }
    payload = {
        **conditioning,
        "conditioning_contract_sha256": _payload_sha256(conditioning),
        "embedding_mode": "lp",
        "embedding_dim": 1024,
        "generator_current_input_mode": "current_support_masked",
        "generator_noise_mode": "gaussian",
        "noise_dim": 32,
        "capacity_profile": "small",
        "initial_learning_rate": 5e-7,
        "surface_grid_profile": grid["surface_grid_profile"],
        "surface_grid_sha256": grid["surface_grid_sha256"],
        "architecture_profile_sha256": architecture["architecture_profile_sha256"],
        "architecture": {
            key: value
            for key, value in architecture.items()
            if key not in {"schema_version", "architecture_profile_sha256"}
        },
        "expected_wgan_parameters": EXPECTED_WGAN_PARAMETERS[generator_mode],
    }
    payload["model_contract_sha256"] = _payload_sha256(payload)
    return payload


def _stage_resolved(resolved: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    output = deepcopy(dict(resolved))
    if stage in {DEVELOPMENT_STAGE, BENCHMARK_STAGE}:
        train_end = str(resolved["split"]["development_train_end_utc"])
        validation_end = str(resolved["split"]["development_validation_end_utc"])
    elif stage == REFIT_STAGE:
        train_end = str(resolved["split"]["refit_train_end_utc"])
        validation_end = train_end
    else:
        raise ValueError(f"Unknown training stage: {stage}")
    output["split"] = {
        "train_end_utc": train_end,
        "validation_end_utc": validation_end,
        "validation_mc_samples": int(resolved["split"]["validation_mc_samples"]),
    }
    return output


def _training_payload(
    resolved: Mapping[str, Any],
    root: Path,
    *,
    spec: Mapping[str, Any],
    stage: str,
    recipe_path: Path | None = None,
    benchmark: bool = False,
) -> dict[str, Any]:
    tolerance = int(spec["tolerance_minutes"])
    mode = str(spec["text_ablation_mode"])
    contract = _conditioning_contract(spec)
    payload = training._training_payload(
        _stage_resolved(resolved, stage=stage),
        family="wgan",
        tolerance=tolerance,
        text_ablation_mode=mode,
        multi_mode=True,
        experiment_root=root,
        capacity_profile=None,
    )
    pre_q4 = (
        root / "data_windows" / "pre_q4" / f"tolerance_{tolerance:02d}m_pre_q4.xlsx"
    )
    common = root / "data_windows" / "pre_q4" / "tolerance_05m_pre_q4.xlsx"
    output_root = (
        root
        / "runs"
        / ("benchmark" if benchmark else stage)
        / str(spec["generator_conditioning_mode"])
        / str(spec["critic_conditioning_mode"])
        / mode
        / f"tolerance_{tolerance:02d}m"
        / f"seed_{int(spec['seed']):03d}"
    )
    payload.update(
        {
            "data_path": str(pre_q4.resolve()),
            "news_first_common_eval_data_path": str(common.resolve()),
            "news_first_dataset_tolerance_minutes": tolerance,
            "news_first_text_ablation_mode": mode,
            "news_first_text_information_path": text_information_path(mode),
            "news_first_text_shuffle_seed": int(
                resolved["datasets"]["text_shuffle_seed"]
            ),
            "news_first_capacity_profile": "small",
            "news_first_lr_profile": "lr_5e_07",
            "generator_conditioning_mode": str(spec["generator_conditioning_mode"]),
            "critic_conditioning_mode": str(spec["critic_conditioning_mode"]),
            "news_first_architecture_profile_sha256": contract[
                "architecture_profile_sha256"
            ],
            "news_first_model_contract_sha256": contract["model_contract_sha256"],
            "news_first_surface_grid_profile": contract["surface_grid_profile"],
            "news_first_surface_grid_sha256": contract["surface_grid_sha256"],
            "seed": int(spec["seed"]),
            "learning_rate": 5e-7,
            "generator_learning_rate": 5e-7,
            "discriminator_learning_rate": 5e-7,
            "reduce_lr_min_lr": 5e-8,
            "news_first_materialize_test_loader": False,
            "output_root": str(output_root.resolve()),
        }
    )
    if stage == REFIT_STAGE:
        if recipe_path is None or not recipe_path.is_file():
            raise FileNotFoundError("Refit payload requires a frozen recipe")
        recipe = _require_mapping(_read_json(recipe_path), "refit recipe")
        payload.update(
            {
                "news_first_train_end_utc": str(
                    resolved["split"]["refit_train_end_utc"]
                ),
                "news_first_validation_end_utc": str(
                    resolved["split"]["refit_train_end_utc"]
                ),
                "news_first_refit_mode": REFIT_MODE,
                "news_first_refit_recipe_path": str(recipe_path.resolve()),
                "news_first_refit_recipe_sha256": _sha256_file(recipe_path),
                "news_first_materialize_validation_loader": False,
                "num_epochs": int(recipe["num_epochs"]),
                "lr_scheduler_type": "none",
                "use_reduce_lr_on_plateau": False,
                "use_early_stopping": False,
                "evaluate_initial_checkpoint": False,
            }
        )
    else:
        payload.update(
            {
                "news_first_train_end_utc": str(
                    resolved["split"]["development_train_end_utc"]
                ),
                "news_first_validation_end_utc": str(
                    resolved["split"]["development_validation_end_utc"]
                ),
                "news_first_refit_mode": "none",
                "news_first_refit_recipe_path": "",
                "news_first_refit_recipe_sha256": "",
                "news_first_materialize_validation_loader": True,
            }
        )
    if benchmark:
        payload.update(
            {
                "num_epochs": 1,
                "use_early_stopping": False,
                "early_stopping_min_epochs": 1,
                "early_stopping_patience": 1,
            }
        )
    return payload


def _job_spec_sha(job: Mapping[str, Any]) -> str:
    return _payload_sha256(
        {key: value for key, value in job.items() if key != "job_spec_sha256"}
    )


def _load_registry(root: Path) -> dict[str, Any]:
    registry = _require_mapping(_read_json(root / "registry" / "jobs.json"), "registry")
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment root is not a Generator/Critic factorial")
    return registry


def _write_registry_exports(root: Path) -> None:
    registry = _load_registry(root)
    jobs = list(registry["jobs"])
    if jobs:
        _write_csv(root / "task_registry.csv", jobs, tuple(jobs[0]))
    artifacts: list[dict[str, Any]] = []
    for job in jobs:
        status_path = _job_status_path(root, job["job_id"])
        if status_path.is_file():
            artifacts.extend(_read_json(status_path).get("artifacts") or [])
    if artifacts:
        dedup = {str(row["path"]): dict(row) for row in artifacts}
        _write_csv(
            root / "output_hashes.csv",
            list(dedup.values()),
            tuple(next(iter(dedup.values()))),
        )


def _anchor_manifests(root: Path) -> None:
    registry = _load_registry(root)
    for name in ("source_hashes.csv", "code_hashes.csv", "config_hashes.csv"):
        registry[f"{name.removesuffix('.csv')}_sha256"] = _sha256_file(root / name)
    _write_json(root / "registry" / "jobs.json", registry)


def _validate_root_lineage(root: Path) -> dict[str, Any]:
    registry = _load_registry(root)
    for name in ("source_hashes.csv", "code_hashes.csv", "config_hashes.csv"):
        path = root / name
        anchor = f"{name.removesuffix('.csv')}_sha256"
        if not path.is_file() or _sha256_file(path) != registry.get(anchor):
            raise ValueError(f"Manifest anchor mismatch: {name}")
        _verify_hash_manifest(path)
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY), ROOT_KEY
    )
    if _payload_sha256(resolved) != registry["resolved_config_sha256"]:
        raise ValueError("Resolved config hash drift")
    ids: list[str] = []
    outputs: list[str] = []
    for job in registry["jobs"]:
        ids.append(str(job["job_id"]))
        outputs.append(str(job["output_root"]))
        if _job_spec_sha(job) != job["job_spec_sha256"]:
            raise ValueError(f"Job spec hash mismatch: {job['job_id']}")
        if _sha256_file(Path(job["training_config_path"])) != job["config_sha256"]:
            raise ValueError(f"Job config drift: {job['job_id']}")
        contract = _conditioning_contract(job)
        for key in (
            "generator_conditioning_fingerprint",
            "critic_conditioning_fingerprint",
            "conditioning_contract_sha256",
            "architecture_profile_sha256",
            "model_contract_sha256",
            "surface_grid_profile",
            "surface_grid_sha256",
            "expected_wgan_parameters",
        ):
            if job.get(key) != contract[key]:
                raise ValueError(f"Job {key} drift: {job['job_id']}")
        training_config = _load_yaml(
            Path(job["training_config_path"]), "training config"
        )
        expected_config = {
            "generator_conditioning_mode": job["generator_conditioning_mode"],
            "critic_conditioning_mode": job["critic_conditioning_mode"],
            "news_first_architecture_profile_sha256": job[
                "architecture_profile_sha256"
            ],
            "news_first_model_contract_sha256": job["model_contract_sha256"],
            "news_first_surface_grid_profile": job["surface_grid_profile"],
            "news_first_surface_grid_sha256": job["surface_grid_sha256"],
        }
        for key, expected in expected_config.items():
            if training_config.get(key) != expected:
                raise ValueError(f"Training config {key} drift: {job['job_id']}")
        if _sha256_file(Path(job["dataset_path"])) != job["dataset_sha256"]:
            raise ValueError(f"Job dataset drift: {job['job_id']}")
        recipe_path = str(job.get("refit_recipe_path", ""))
        if recipe_path and (
            not Path(recipe_path).is_file()
            or _sha256_file(Path(recipe_path)) != job["refit_recipe_sha256"]
        ):
            raise ValueError(f"Refit recipe drift: {job['job_id']}")
    if len(ids) != len(set(ids)) or len(outputs) != len(set(outputs)):
        raise ValueError("Duplicate job ID or output path")
    return resolved


def _benchmark_result_path(formal_root: Path) -> Path:
    return (
        formal_root.with_name(formal_root.name + "_benchmark") / "benchmark_result.json"
    )


def _validated_benchmark_result(
    formal_root: Path, config_path: str | Path
) -> dict[str, Any]:
    benchmark_root = formal_root.with_name(formal_root.name + "_benchmark")
    benchmark_resolved = _validate_root_lineage(benchmark_root)
    current_resolved = resolve_config(config_path)
    if _payload_sha256(benchmark_resolved) != _payload_sha256(current_resolved):
        raise ValueError(
            "Benchmark config/source/code lineage does not match formal prepare"
        )
    path = _benchmark_result_path(formal_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Formal prepare requires the independent 16-worker benchmark first: {path}"
        )
    result = _require_mapping(_read_json(path), "benchmark result")
    payload_sha = str(result.get("payload_sha256", ""))
    payload = {key: value for key, value in result.items() if key != "payload_sha256"}
    if not payload_sha or _payload_sha256(payload) != payload_sha:
        raise ValueError("Benchmark result self-hash mismatch")
    if result.get("status") != "passed" or int(result.get("worker_count", -1)) != 16:
        raise ValueError("Benchmark did not complete the frozen 16-worker wave")
    if int(result.get("workers_per_gpu", -1)) != 8:
        raise ValueError("Benchmark must use 8 workers/GPU")
    if int(result.get("selected_slots_per_gpu", -1)) not in {4, 8}:
        raise ValueError("Benchmark did not freeze 4 or 8 slots/GPU")
    return result


def _manifest_row(role: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "source_role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_training_config(path: Path, payload: Mapping[str, Any]) -> Path:
    return _write_yaml(path, dict(payload))


def _build_jobs(
    resolved: Mapping[str, Any],
    root: Path,
    *,
    stage: str,
    slots_per_gpu: int,
    recipes: Mapping[str, Path] | None = None,
    benchmark: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gpu_ids = tuple(map(int, resolved["runtime"]["gpu_ids"]))
    specs = _balanced_assignments(
        factorial_specs(BENCHMARK_STAGE if benchmark else stage),
        gpu_ids=gpu_ids,
        slots_per_gpu=int(slots_per_gpu),
        invert_gpu=stage == REFIT_STAGE,
    )
    jobs: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for spec in specs:
        development_id = _job_id(DEVELOPMENT_STAGE, spec)
        recipe_path = None if recipes is None else recipes.get(development_id)
        payload = _training_payload(
            resolved,
            root,
            spec=spec,
            stage=stage,
            recipe_path=recipe_path,
            benchmark=benchmark,
        )
        job_stage = BENCHMARK_STAGE if benchmark else stage
        job_id = _job_id(job_stage, spec)
        config_path = (root / "configs" / job_stage / f"{job_id}.yaml").resolve(
            strict=False
        )
        _write_training_config(config_path, payload)
        dataset_path = Path(payload["data_path"])
        contract = _conditioning_contract(spec)
        job: dict[str, Any] = {
            "job_id": job_id,
            "experiment_kind": EXPERIMENT_KIND,
            "experiment_stage": job_stage,
            "model_family": "wgan",
            "capacity_profile": "small",
            "lr_profile": "lr_5e_07",
            "generator_conditioning_mode": spec["generator_conditioning_mode"],
            "critic_conditioning_mode": spec["critic_conditioning_mode"],
            "generator_conditioning_fingerprint": contract[
                "generator_conditioning_fingerprint"
            ],
            "critic_conditioning_fingerprint": contract[
                "critic_conditioning_fingerprint"
            ],
            "conditioning_contract_sha256": contract["conditioning_contract_sha256"],
            "architecture_profile_sha256": contract["architecture_profile_sha256"],
            "model_contract_sha256": contract["model_contract_sha256"],
            "surface_grid_profile": contract["surface_grid_profile"],
            "surface_grid_sha256": contract["surface_grid_sha256"],
            "expected_wgan_parameters": contract["expected_wgan_parameters"],
            "text_ablation_mode": spec["text_ablation_mode"],
            "support_mask_mode": "raw_joint",
            "tolerance_minutes": int(spec["tolerance_minutes"]),
            "seed": int(spec["seed"]),
            "gpu_id": int(spec["gpu_id"]),
            "gpu_slot": int(spec["gpu_slot"]),
            "wave": int(spec["wave"]),
            "training_config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": _sha256_file(dataset_path),
            "output_root": str(Path(payload["output_root"]).resolve()),
            "development_job_id": development_id,
            "parent_development_job_id": (
                development_id if stage == REFIT_STAGE else ""
            ),
            "refit_recipe_path": str(recipe_path.resolve()) if recipe_path else "",
            "refit_recipe_sha256": _sha256_file(recipe_path) if recipe_path else "",
        }
        job["job_spec_sha256"] = _job_spec_sha(job)
        jobs.append(job)
        config_rows.append(_manifest_row(f"training_config_{job_id}", config_path))
    if len(jobs) != 16:
        raise ValueError(f"Expected 16 {stage} jobs, observed {len(jobs)}")
    if len({job["job_id"] for job in jobs}) != 16:
        raise ValueError(f"Duplicate {stage} job IDs")
    if len({job["output_root"] for job in jobs}) != 16:
        raise ValueError(f"Duplicate {stage} output roots")
    return jobs, config_rows


def _initial_job_status(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "experiment_stage": job["experiment_stage"],
        "status": "prepared",
        "attempt": 0,
        "config_sha256": job["config_sha256"],
        "q4_loader_created": False,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "updated_at_utc": _utc_now(),
    }


def _write_experiment_status(root: Path, status: str, **details: Any) -> Path:
    return _write_json(
        root / "registry" / "experiment_status.json",
        {
            "status": status,
            "q4_window_materialized": bool(
                details.pop("q4_window_materialized", False)
            ),
            "q4_loader_created": bool(details.pop("q4_loader_created", False)),
            "q4_predictions_generated": bool(
                details.pop("q4_predictions_generated", False)
            ),
            "q4_evaluated": bool(details.pop("q4_evaluated", False)),
            "updated_at_utc": _utc_now(),
            **details,
        },
    )


def _write_model_contract_manifest(root: Path) -> Path:
    contracts = []
    for generator_mode in GENERATOR_MODES:
        for critic_mode in CRITIC_MODES:
            contracts.append(
                _conditioning_contract(
                    {
                        "generator_conditioning_mode": generator_mode,
                        "critic_conditioning_mode": critic_mode,
                        "text_ablation_mode": REAL_TEXT,
                    }
                )
            )
    payload = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "surface_grid": _surface_grid_contract(),
        "architecture": _architecture_contract(),
        "contracts": contracts,
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return _write_json(root / "model_contract_manifest.json", payload)


def _prepare_root(
    config_path: str | Path,
    root: Path,
    *,
    slots_per_gpu: int,
    benchmark: bool,
) -> Path:
    resolved = resolve_config(config_path)
    resolved["runtime"]["slots_per_gpu"] = int(slots_per_gpu)
    _validate_config(resolved)
    for relative in (
        "analysis",
        "configs",
        "data_windows",
        "logs",
        "registry/jobs",
        "report",
        "runs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    resolved_path = _write_yaml(root / "resolved_config.yaml", {ROOT_KEY: resolved})
    model_contract_manifest = _write_model_contract_manifest(root)
    window_rows, window_manifest = _materialize_pre_q4_windows(resolved, root)
    split_manifest = _write_split_manifest(resolved, root)
    stage = BENCHMARK_STAGE if benchmark else DEVELOPMENT_STAGE
    jobs, job_config_rows = _build_jobs(
        resolved,
        root,
        stage=DEVELOPMENT_STAGE,
        slots_per_gpu=int(slots_per_gpu),
        benchmark=benchmark,
    )
    source_rows = _source_rows(resolved) + window_rows
    _write_hash_rows(root / "source_hashes.csv", source_rows)
    _write_hash_rows(root / "code_hashes.csv", _code_rows())
    config_rows = [
        _manifest_row("resolved_config", resolved_path),
        _manifest_row("rolling_split_manifest", split_manifest),
        _manifest_row("pre_q4_window_manifest", window_manifest),
        _manifest_row("model_contract_manifest", model_contract_manifest),
        *job_config_rows,
    ]
    _write_hash_rows(root / "config_hashes.csv", config_rows)
    registry = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "status": "prepared",
        "benchmark_root": bool(benchmark),
        "resolved_config_sha256": _payload_sha256(resolved),
        "slots_per_gpu": int(slots_per_gpu),
        "development_job_count": 0 if benchmark else 16,
        "benchmark_job_count": 16 if benchmark else 0,
        "refit_job_count": 0,
        "selection_frozen": False,
        "refit_complete": False,
        "q4_gate_open": False,
        "q4_window_materialized": False,
        "q4_loader_created": False,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "jobs": jobs,
        "created_at_utc": _utc_now(),
    }
    _write_json(root / "registry" / "jobs.json", registry)
    for job in jobs:
        _write_json(_job_status_path(root, job["job_id"]), _initial_job_status(job))
    _anchor_manifests(root)
    _write_registry_exports(root)
    _write_experiment_status(
        root,
        "benchmark_prepared" if benchmark else "development_prepared",
        current_stage=stage,
    )
    _validate_root_lineage(root)
    return root


def prepare_factorial_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    reuse: bool = False,
) -> Path:
    root = Path(output_dir).resolve(strict=False)
    if root.exists():
        if not reuse:
            raise FileExistsError(f"Experiment root already exists: {root}")
        _validate_root_lineage(root)
        return root
    benchmark = _validated_benchmark_result(root, config_path)
    selected = int(benchmark["selected_slots_per_gpu"])
    prepared = _prepare_root(
        config_path,
        root,
        slots_per_gpu=selected,
        benchmark=False,
    )
    benchmark_path = _benchmark_result_path(root)
    with (root / "source_hashes.csv").open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    source_rows.append(_manifest_row("formal_concurrency_benchmark", benchmark_path))
    _write_hash_rows(root / "source_hashes.csv", source_rows)
    registry = _load_registry(root)
    registry.update(
        {
            "benchmark_result_path": str(benchmark_path.resolve()),
            "benchmark_result_sha256": _sha256_file(benchmark_path),
            "benchmark_payload_sha256": benchmark["payload_sha256"],
        }
    )
    _write_json(root / "registry" / "jobs.json", registry)
    _anchor_manifests(root)
    _validate_root_lineage(root)
    return prepared


def _custom_artifact_rows(
    job: Mapping[str, Any], run_dir: Path
) -> list[dict[str, Any]]:
    checkpoints = run_dir / "checkpoints"
    metrics = run_dir / "metrics"
    if job["experiment_stage"] == REFIT_STAGE:
        required = {
            "generator_final": checkpoints / "generator.pt",
            "discriminator_final": checkpoints / "discriminator.pt",
            "training_metrics_csv": metrics / "training_metrics.csv",
            "training_metrics_json": metrics / "training_metrics.json",
            "resolved_training_config": metrics / "training_resolved_config.yaml",
            "run_log": run_dir / "run.log",
        }
        if tuple(required) != REFIT_ARTIFACT_ROLES:
            raise AssertionError("Refit artifact-role contract drift")
    else:
        required = {
            "generator_initial_epoch0": checkpoints / "generator_initial_epoch0.pt",
            "discriminator_initial_epoch0": checkpoints
            / "discriminator_initial_epoch0.pt",
            "generator_best": checkpoints / "generator_best.pt",
            "discriminator_best": checkpoints / "discriminator_best.pt",
            "generator_best_learned": checkpoints / "generator_best_learned.pt",
            "discriminator_best_learned": checkpoints / "discriminator_best_learned.pt",
            "generator_final": checkpoints / "generator.pt",
            "discriminator_final": checkpoints / "discriminator.pt",
            "training_metrics_csv": metrics / "training_metrics.csv",
            "training_metrics_json": metrics / "training_metrics.json",
            "best_checkpoint": metrics / "best_checkpoint.json",
            "initial_checkpoint": metrics / "initial_checkpoint.json",
            "best_learned_checkpoint": metrics / "best_learned_checkpoint.json",
            "resolved_training_config": metrics / "training_resolved_config.yaml",
            "run_log": run_dir / "run.log",
        }
        if tuple(required) != DEVELOPMENT_ARTIFACT_ROLES:
            raise AssertionError("Development artifact-role contract drift")
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"{job['experiment_stage']} completed without artifacts: {missing}"
        )
    unexpected = sorted(
        path
        for path in checkpoints.glob("*_epoch_*.pt")
        if "initial_epoch0" not in path.name
    )
    if unexpected:
        raise RuntimeError(f"Best/final-only checkpoint policy violated: {unexpected}")
    return [
        {
            "artifact_role": role,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for role, path in required.items()
    ]


def _execute_factorial_job(
    job: Mapping[str, Any], *, dry_run: bool
) -> tuple[Path, list[dict[str, Any]]]:
    from wgan_option.config import load_config
    from wgan_option.train_vol_xlsx import VolSurfaceXlsxTrainer

    config = load_config(str(job["training_config_path"]))
    trainer = VolSurfaceXlsxTrainer(
        config, config_path=str(job["training_config_path"])
    )
    result = trainer.dry_run() if dry_run else trainer.train()
    if result is None:
        raise RuntimeError("Trainer did not return a run directory")
    if trainer.model is None:
        raise RuntimeError("Trainer did not instantiate the WGAN")
    actual_parameters = sum(
        parameter.numel()
        for model in (trainer.model.G, trainer.model.D)
        for parameter in model.parameters()
    )
    if actual_parameters != int(job["expected_wgan_parameters"]):
        raise RuntimeError(
            f"WGAN parameter contract drift: {actual_parameters} != "
            f"{job['expected_wgan_parameters']}"
        )
    actual_generator_fingerprint = str(
        trainer.model.G.generator_conditioning_fingerprint
    )
    actual_critic_fingerprint = str(trainer.model.D.critic_conditioning_fingerprint)
    if actual_generator_fingerprint != job["generator_conditioning_fingerprint"]:
        raise RuntimeError("Instantiated Generator conditioning fingerprint drift")
    if actual_critic_fingerprint != job["critic_conditioning_fingerprint"]:
        raise RuntimeError("Instantiated Critic conditioning fingerprint drift")
    if str(config.news_first_model_contract_sha256) != job["model_contract_sha256"]:
        raise RuntimeError("Instantiated model-contract SHA drift")
    if str(config.news_first_surface_grid_sha256) != job["surface_grid_sha256"]:
        raise RuntimeError("Instantiated exact-TTM grid SHA drift")
    run_dir = Path(result).resolve(strict=False)
    return run_dir, [] if dry_run else _custom_artifact_rows(job, run_dir)


def _completed_job_is_valid(job: Mapping[str, Any], status: Mapping[str, Any]) -> bool:
    if (
        status.get("status") != "completed"
        or status.get("config_sha256") != job["config_sha256"]
    ):
        return False
    artifacts = list(status.get("artifacts") or [])
    if not artifacts:
        return False
    return all(
        Path(str(row.get("path", ""))).is_file()
        and _sha256_file(Path(str(row["path"]))) == row.get("sha256")
        for row in artifacts
    )


def _find_job(root: Path, job_id: str) -> dict[str, Any]:
    for job in _load_registry(root)["jobs"]:
        if str(job["job_id"]) == str(job_id):
            return dict(job)
    raise KeyError(f"Unknown job_id: {job_id}")


def _validate_worker_gate(root: Path, job: Mapping[str, Any]) -> None:
    registry = _load_registry(root)
    stage = str(job["experiment_stage"])
    if bool(registry.get("q4_gate_open")) or bool(registry.get("q4_evaluated")):
        raise RuntimeError("Training is forbidden after the Q4 gate opens")
    if stage == DEVELOPMENT_STAGE and bool(registry.get("selection_frozen")):
        raise RuntimeError("Development jobs are frozen after selection")
    if stage == REFIT_STAGE and not bool(registry.get("selection_frozen")):
        raise RuntimeError("Refit jobs require frozen Q3 selection and recipes")
    if stage == BENCHMARK_STAGE and not bool(registry.get("benchmark_root")):
        raise RuntimeError("Benchmark jobs require an isolated benchmark root")
    payload = _load_yaml(Path(job["training_config_path"]), "training config")
    if stage == REFIT_STAGE:
        forbidden = {
            "news_first_materialize_validation_loader": False,
            "news_first_materialize_test_loader": False,
            "use_reduce_lr_on_plateau": False,
            "use_early_stopping": False,
            "evaluate_initial_checkpoint": False,
        }
        for key, expected in forbidden.items():
            if bool(payload.get(key)) != expected:
                raise ValueError(f"Refit {key} must equal {expected}")
        if str(payload.get("news_first_refit_mode")) != REFIT_MODE:
            raise ValueError("Refit payload is missing frozen LR replay mode")


def run_factorial_worker(
    output_dir: str | Path,
    job_id: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    root = Path(output_dir).resolve(strict=False)
    _validate_root_lineage(root)
    job = _find_job(root, job_id)
    _validate_worker_gate(root, job)
    status_path = _job_status_path(root, job_id)
    previous = _require_mapping(_read_json(status_path), "job status")
    if previous.get("status") == "completed" and _completed_job_is_valid(job, previous):
        if resume and not dry_run:
            return Path(str(previous["run_dir"])).resolve(strict=False)
        raise RuntimeError(f"Job already completed; use --resume: {job_id}")
    if previous.get("status") == "running" and training._pid_is_live(
        previous.get("pid")
    ):
        raise RuntimeError(f"Refusing duplicate live job: {job_id}")
    if previous.get("status") in {"running", "failed"} and not (resume or dry_run):
        raise RuntimeError(f"Interrupted job requires --resume: {job_id}")
    attempt = int(previous.get("attempt", 0)) + 1
    running = {
        **_initial_job_status(job),
        "status": "running",
        "attempt": attempt,
        "dry_run": bool(dry_run),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at_utc": _utc_now(),
    }
    _write_json(status_path, running)
    try:
        run_dir, artifacts = _execute_factorial_job(job, dry_run=dry_run)
        completed = {
            **running,
            "status": "dry_run_passed" if dry_run else "completed",
            "run_dir": str(run_dir),
            "artifacts": artifacts,
            "completed_at_utc": _utc_now(),
        }
        _write_json(status_path, completed)
        _write_registry_exports(root)
        return run_dir
    except BaseException as exc:
        _write_json(
            status_path,
            {
                **running,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "completed_at_utc": _utc_now(),
            },
        )
        raise


def build_factorial_worker_command(
    config_path: str | Path,
    root: Path,
    job: Mapping[str, Any],
    *,
    dry_run: bool,
    resume: bool,
) -> list[str]:
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY),
        ROOT_KEY,
    )
    command = [
        str(resolved["runtime"]["python_executable"]),
        str(REPO_ROOT / "scripts" / "rq3" / "main.py"),
        "train-news-first-vol-generator-film-critic-factorial",
        "worker",
        "--config",
        str(_resolve_repo_path(config_path)),
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


def _selected_jobs(
    root: Path,
    *,
    stage: str,
    wave: int,
    resume: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for raw in _load_registry(root)["jobs"]:
        job = dict(raw)
        if job["experiment_stage"] != stage or int(job["wave"]) != int(wave):
            continue
        status = _require_mapping(
            _read_json(_job_status_path(root, job["job_id"])), "job status"
        )
        if not dry_run and _completed_job_is_valid(job, status):
            if resume:
                continue
            raise RuntimeError(f"Job already completed; use --resume: {job['job_id']}")
        if status.get("status") == "running" and training._pid_is_live(
            status.get("pid")
        ):
            raise RuntimeError(f"Refusing duplicate live job: {job['job_id']}")
        if status.get("status") in {"running", "failed"} and not (resume or dry_run):
            raise RuntimeError(
                f"Interrupted/failed job requires --resume: {job['job_id']}"
            )
        selected.append(job)
    return selected


def _host_ram_fraction() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    return 1.0 - available / total if total > 0 else 1.0


def _run_wave(
    config_path: str | Path,
    root: Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    wave: int,
    dry_run: bool,
    resume: bool,
) -> float:
    if not jobs:
        return _host_ram_fraction()
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY),
        ROOT_KEY,
    )
    runtime = resolved["runtime"]
    monitor = training._ResourceMonitor(
        root / "resource_usage.csv",
        executable=str(runtime.get("nvidia_smi_executable", "nvidia-smi")),
        interval_seconds=float(runtime.get("resource_sample_interval_seconds", 2)),
        wave=int(wave),
    )
    processes: list[subprocess.Popen[Any]] = []
    handles: list[Any] = []
    peak_host = _host_ram_fraction()
    try:
        monitor.start()
        for job in jobs:
            previous = _read_json(_job_status_path(root, str(job["job_id"])))
            attempt = int(previous.get("attempt", 0)) + 1
            log_path = root / "logs" / f"{job['job_id']}.attempt_{attempt:02d}.log"
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
            env["NEWS_FIRST_JOB_LOG_PATH"] = str(log_path)
            env["PYTHONPATH"] = os.pathsep.join(
                (str(REPO_ROOT / "src"), str(REPO_ROOT))
            )
            threads = str(int(runtime["cpu_threads_per_job"]))
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                env[name] = threads
            processes.append(
                subprocess.Popen(
                    build_factorial_worker_command(
                        config_path,
                        root,
                        job,
                        dry_run=dry_run,
                        resume=resume,
                    ),
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        pending = set(range(len(processes)))
        while pending:
            peak_host = max(peak_host, _host_ram_fraction())
            for index in tuple(pending):
                code = processes[index].poll()
                if code is None:
                    continue
                pending.remove(index)
                if code != 0:
                    training._terminate_processes(processes)
                    raise RuntimeError(
                        f"Wave {wave} job {jobs[index]['job_id']} exited {code}"
                    )
            if pending:
                time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        training._terminate_processes(processes)
        raise
    finally:
        monitor.stop()
        for handle in handles:
            handle.close()
    return peak_host


def _validate_stage_complete(root: Path, stage: str, *, dry_run: bool) -> None:
    jobs = [
        dict(job)
        for job in _load_registry(root)["jobs"]
        if job["experiment_stage"] == stage
    ]
    expected = 16
    if len(jobs) != expected:
        raise RuntimeError(f"{stage} matrix must contain {expected} jobs")
    failures = []
    for job in jobs:
        status = _read_json(_job_status_path(root, job["job_id"]))
        valid = (
            status.get("status") == "dry_run_passed"
            if dry_run
            else _completed_job_is_valid(job, status)
        )
        if not valid:
            failures.append(f"{job['job_id']}={status.get('status')}")
    if failures:
        raise RuntimeError(f"Incomplete {stage} matrix: {failures}")


def _launch_stage(
    config_path: str | Path,
    root: Path,
    *,
    stage: str,
    resume: bool,
    dry_run: bool,
) -> float:
    _validate_root_lineage(root)
    registry = _load_registry(root)
    jobs = [job for job in registry["jobs"] if job["experiment_stage"] == stage]
    if len(jobs) != 16:
        raise RuntimeError(f"Expected 16 {stage} jobs, observed {len(jobs)}")
    _assert_gpu_balance(
        jobs,
        gpu_ids=tuple(
            map(
                int,
                _require_mapping(
                    _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY),
                    ROOT_KEY,
                )["runtime"]["gpu_ids"],
            )
        ),
    )
    peak_host = 0.0
    for wave in sorted({int(job["wave"]) for job in jobs}):
        selected = _selected_jobs(
            root,
            stage=stage,
            wave=wave,
            resume=resume,
            dry_run=dry_run,
        )
        peak_host = max(
            peak_host,
            _run_wave(
                config_path,
                root,
                selected,
                wave=wave,
                dry_run=dry_run,
                resume=resume,
            ),
        )
    _validate_stage_complete(root, stage, dry_run=dry_run)
    _write_registry_exports(root)
    return peak_host


def _resource_peaks(root: Path, gpu_ids: Sequence[int]) -> dict[int, float]:
    path = root / "resource_usage.csv"
    if not path.is_file():
        raise RuntimeError("Benchmark produced no GPU resource telemetry")
    peaks = {int(gpu): 0.0 for gpu in gpu_ids}
    samples = {int(gpu): 0 for gpu in gpu_ids}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("sample_status")) != "ok":
                continue
            gpu = int(row["gpu_index"])
            if gpu not in peaks:
                continue
            samples[gpu] += 1
            peaks[gpu] = max(peaks[gpu], float(row["memory_used_mib"]))
    if any(samples[gpu] == 0 for gpu in gpu_ids):
        raise RuntimeError(f"Benchmark telemetry missing GPU samples: {samples}")
    return peaks


def run_factorial_benchmark(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> Path:
    formal_root = Path(output_dir).resolve(strict=False)
    if formal_root.exists():
        raise FileExistsError(
            f"Benchmark must run before the formal root is created: {formal_root}"
        )
    root = formal_root.with_name(formal_root.name + "_benchmark")
    if root.exists():
        if not resume:
            raise FileExistsError(f"Benchmark root exists; use --resume: {root}")
        _validate_root_lineage(root)
    else:
        _prepare_root(config_path, root, slots_per_gpu=8, benchmark=True)
    _write_experiment_status(root, "benchmark_running", current_stage=BENCHMARK_STAGE)
    try:
        peak_host = _launch_stage(
            config_path,
            root,
            stage=BENCHMARK_STAGE,
            resume=resume,
            dry_run=False,
        )
        resolved = _require_mapping(
            _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY),
            ROOT_KEY,
        )
        gpu_ids = tuple(map(int, resolved["runtime"]["gpu_ids"]))
        peaks = _resource_peaks(root, gpu_ids)
        memory_limit = (
            float(resolved["runtime"]["preflight_max_peak_gpu_memory_gib"]) * 1024.0
        )
        ram_limit = float(resolved["runtime"]["preflight_max_host_ram_fraction"])
        selected = (
            8
            if all(value < memory_limit for value in peaks.values())
            and peak_host < ram_limit
            else 4
        )
        result = {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "status": "passed",
            "worker_count": 16,
            "workers_per_gpu": 8,
            "num_epochs": 1,
            "peak_gpu_memory_mib": {str(key): value for key, value in peaks.items()},
            "peak_host_ram_fraction": peak_host,
            "gpu_memory_limit_mib_exclusive": memory_limit,
            "host_ram_limit_fraction_exclusive": ram_limit,
            "selected_slots_per_gpu": selected,
            "formal_root_must_be_new": True,
            "completed_at_utc": _utc_now(),
        }
        result["payload_sha256"] = _payload_sha256(result)
        _write_json(root / "benchmark_result.json", result)
        _write_experiment_status(
            root,
            "benchmark_completed",
            current_stage=BENCHMARK_STAGE,
            selected_slots_per_gpu=selected,
        )
    except BaseException as exc:
        _write_experiment_status(
            root,
            "failed",
            current_stage=BENCHMARK_STAGE,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return root


def _artifact_path(status: Mapping[str, Any], role: str) -> Path:
    matches = [
        Path(str(row["path"]))
        for row in status.get("artifacts") or []
        if row.get("artifact_role") == role
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {role} artifact, observed {len(matches)}")
    return matches[0]


def _validate_refit_recipe(recipe: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "refit_mode",
        "num_epochs",
        "generator_lr_trace",
        "discriminator_lr_trace",
    }
    if set(recipe) != required:
        raise ValueError(f"Refit recipe fields must be exactly {sorted(required)}")
    if int(recipe["schema_version"]) != 1 or recipe["refit_mode"] != REFIT_MODE:
        raise ValueError("Invalid refit recipe schema/mode")
    num_epochs = int(recipe["num_epochs"])
    if num_epochs < 1:
        raise ValueError("Refit recipe num_epochs must be >=1")
    expected = list(range(1, num_epochs + 1))
    for name in ("generator_lr_trace", "discriminator_lr_trace"):
        trace = list(recipe[name])
        if [int(row["epoch"]) for row in trace] != expected:
            raise ValueError(f"{name} must cover epoch1..N exactly")
        if any(
            set(row) != {"epoch", "lr"}
            or not math.isfinite(float(row["lr"]))
            or float(row["lr"]) <= 0.0
            for row in trace
        ):
            raise ValueError(f"Invalid {name}")


def run_film_critic_q3_analysis(root: Path) -> Mapping[str, Any] | Path:
    from scripts.rq3.news_first_vol_generator_film_critic_factorial_analysis import (
        run_film_critic_q3_analysis as implementation,
    )

    return implementation(root)


def run_film_critic_q4_analysis(root: Path) -> Mapping[str, Any] | Path:
    from scripts.rq3.news_first_vol_generator_film_critic_factorial_analysis import (
        run_film_critic_q4_analysis as implementation,
    )

    return implementation(root)


def render_film_critic_report(root: Path) -> Path:
    from scripts.rq3.news_first_vol_generator_film_critic_factorial_report import (
        render_film_critic_report as implementation,
    )

    return implementation(root)


def freeze_refit_recipes(
    root: Path, selection: Mapping[str, Any] | str | Path
) -> Mapping[str, Any] | Path:
    from scripts.rq3.news_first_vol_generator_film_critic_factorial_analysis import (
        freeze_refit_recipes as implementation,
    )

    return implementation(root, selection)


def _validate_analysis_recipe_manifest(
    path: Path,
    *,
    selection_path: Path,
    recipes: Mapping[str, Path],
) -> None:
    manifest = _require_mapping(_read_json(path), "refit recipe manifest")
    manifest_sha = str(manifest.get("manifest_sha256", ""))
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if not manifest_sha or _payload_sha256(unsigned) != manifest_sha:
        raise ValueError("Refit recipe manifest self-hash mismatch")
    if (
        int(manifest.get("recipe_count", -1)) != 16
        or manifest.get("refit_mode") != REFIT_MODE
        or Path(str(manifest.get("selection_path", ""))).resolve()
        != selection_path.resolve()
        or str(manifest.get("selection_sha256", "")) != _sha256_file(selection_path)
    ):
        raise ValueError("Refit recipe manifest selection/count contract drift")
    rows = list(manifest.get("recipes") or [])
    indexed = {str(row.get("development_job_id", "")): row for row in rows}
    if len(rows) != 16 or set(indexed) != set(recipes):
        raise ValueError("Refit recipe manifest does not bind all 16 jobs")
    for development_job_id, recipe_path in recipes.items():
        row = indexed[development_job_id]
        if Path(
            str(row.get("recipe_path", ""))
        ).resolve() != recipe_path.resolve() or str(
            row.get("recipe_sha256", "")
        ) != _sha256_file(recipe_path):
            raise ValueError(
                f"Refit recipe manifest path/hash drift: {development_job_id}"
            )


def _recoverable_analysis_freeze_candidate(
    root: Path, development_jobs: Sequence[Mapping[str, Any]]
) -> Path | None:
    """Return a complete unregistered freeze, while rejecting partial state."""

    selection_path = root / "analysis" / "film_critic_q3_selection.json"
    recipe_directory = root / "analysis" / "refit_recipes"
    recipe_manifest = root / "analysis" / "refit_recipe_manifest.json"
    expected_ids = {str(job["job_id"]) for job in development_jobs}
    recipe_paths = (
        sorted(recipe_directory.glob("*.json")) if recipe_directory.is_dir() else []
    )
    any_freeze_state = (
        selection_path.exists() or recipe_directory.exists() or recipe_manifest.exists()
    )
    if not any_freeze_state:
        return None
    observed_ids = {path.stem for path in recipe_paths}
    complete = (
        selection_path.is_file()
        and recipe_directory.is_dir()
        and recipe_manifest.is_file()
        and observed_ids == expected_ids
        and len(recipe_paths) == 16
    )
    if not complete:
        raise ValueError(
            "Partial Q3 selection/refit freeze detected; fail-closed. "
            "Restore the canonical selection, all 16 recipes, and recipe manifest "
            "from one run, or remove the partial freeze artifacts before retrying"
        )
    selection = _require_mapping(_read_json(selection_path), "canonical Q3 selection")
    selection_sha = str(selection.get("selection_sha256", ""))
    unsigned = {
        key: value for key, value in selection.items() if key != "selection_sha256"
    }
    if not selection_sha or _payload_sha256(unsigned) != selection_sha:
        raise ValueError("Canonical Q3 selection self-hash mismatch")
    recipes: dict[str, Path] = {}
    for path in recipe_paths:
        _validate_refit_recipe(_require_mapping(_read_json(path), "refit recipe"))
        recipes[path.stem] = path
    _validate_analysis_recipe_manifest(
        recipe_manifest,
        selection_path=selection_path,
        recipes=recipes,
    )
    return selection_path


def _validate_registered_refit_freeze(root: Path) -> Path:
    """Validate a fully registered Stage-B freeze without mutating it."""

    registry = _load_registry(root)
    if not bool(registry.get("selection_frozen")):
        raise ValueError("Registry does not report a frozen selection")
    development_jobs = [
        dict(job)
        for job in registry["jobs"]
        if job["experiment_stage"] == DEVELOPMENT_STAGE
    ]
    refit_jobs = [
        dict(job) for job in registry["jobs"] if job["experiment_stage"] == REFIT_STAGE
    ]
    if len(development_jobs) != 16 or len(refit_jobs) != 16:
        raise ValueError("Frozen registry must contain 16 development + 16 refit jobs")
    selection_path = _recoverable_analysis_freeze_candidate(root, development_jobs)
    if selection_path is None:
        raise ValueError("Frozen registry is missing canonical freeze artifacts")
    recipe_manifest = root / "analysis" / "refit_recipe_manifest.json"
    if (
        Path(str(registry.get("selection_path", ""))).resolve()
        != selection_path.resolve()
        or registry.get("selection_sha256") != _sha256_file(selection_path)
        or Path(str(registry.get("refit_recipe_manifest_path", ""))).resolve()
        != recipe_manifest.resolve()
        or registry.get("refit_recipe_manifest_sha256") != _sha256_file(recipe_manifest)
        or int(registry.get("refit_job_count", -1)) != 16
    ):
        raise ValueError("Frozen selection/recipe registry binding drift")
    development_by_id = {str(job["job_id"]): job for job in development_jobs}
    for job in refit_jobs:
        parent_id = str(job.get("parent_development_job_id", ""))
        parent = development_by_id.get(parent_id)
        recipe_path = root / "analysis" / "refit_recipes" / f"{parent_id}.json"
        if parent is None or (
            Path(str(job.get("refit_recipe_path", ""))).resolve()
            != recipe_path.resolve()
            or job.get("refit_recipe_sha256") != _sha256_file(recipe_path)
        ):
            raise ValueError(f"Frozen refit job recipe binding drift: {job['job_id']}")
        for key in (
            "generator_conditioning_mode",
            "critic_conditioning_mode",
            "text_ablation_mode",
            "tolerance_minutes",
            "seed",
        ):
            if job.get(key) != parent.get(key):
                raise ValueError(f"Frozen refit job factor drift: {job['job_id']}")
        status_path = _job_status_path(root, str(job["job_id"]))
        if not status_path.is_file():
            raise ValueError(f"Frozen refit job status is missing: {job['job_id']}")
        status = _require_mapping(_read_json(status_path), "refit job status")
        if (
            status.get("job_id") != job["job_id"]
            or status.get("experiment_stage") != REFIT_STAGE
            or status.get("config_sha256") != job["config_sha256"]
        ):
            raise ValueError(f"Frozen refit job status drift: {job['job_id']}")
    with (root / "config_hashes.csv").open("r", encoding="utf-8", newline="") as handle:
        config_rows = list(csv.DictReader(handle))
    by_role: dict[str, list[dict[str, str]]] = {}
    for row in config_rows:
        by_role.setdefault(str(row.get("source_role", "")), []).append(row)
    required_roles = {
        "q3_selection": selection_path,
        "refit_recipe_manifest": recipe_manifest,
        **{
            f"refit_recipe_{job_id}": root
            / "analysis"
            / "refit_recipes"
            / f"{job_id}.json"
            for job_id in development_by_id
        },
        **{
            f"training_config_{job['job_id']}": Path(job["training_config_path"])
            for job in refit_jobs
        },
    }
    for role, expected_path in required_roles.items():
        rows = by_role.get(role, [])
        if len(rows) != 1 or Path(rows[0]["path"]).resolve() != expected_path.resolve():
            raise ValueError(f"Frozen config manifest role drift: {role}")
    task_registry = root / "task_registry.csv"
    if not task_registry.is_file():
        raise ValueError("Frozen task registry export is missing")
    with task_registry.open("r", encoding="utf-8", newline="") as handle:
        exported = list(csv.DictReader(handle))
    if len(exported) != 32 or {row["job_id"] for row in exported} != {
        str(job["job_id"]) for job in registry["jobs"]
    }:
        raise ValueError("Frozen task registry export is incomplete")
    return selection_path


def _selection_registration_journal_path(root: Path) -> Path:
    return root / "registry" / "selection_registration.pending.json"


def _recover_selection_registration(root: Path) -> None:
    """Finish the small multi-file registration transaction after interruption."""

    journal_path = _selection_registration_journal_path(root)
    if not journal_path.is_file():
        return
    journal = _require_mapping(_read_json(journal_path), "selection registration")
    digest = str(journal.get("payload_sha256", ""))
    unsigned = {key: value for key, value in journal.items() if key != "payload_sha256"}
    if not digest or _payload_sha256(unsigned) != digest:
        raise ValueError("Selection registration journal self-hash mismatch")
    final_registry = _require_mapping(
        journal.get("final_registry"), "selection registration registry"
    )
    if final_registry.get("experiment_kind") != EXPERIMENT_KIND or not bool(
        final_registry.get("selection_frozen")
    ):
        raise ValueError("Selection registration journal contract drift")
    for name in ("source_hashes.csv", "code_hashes.csv"):
        anchor = f"{name.removesuffix('.csv')}_sha256"
        if not (root / name).is_file() or _sha256_file(
            root / name
        ) != final_registry.get(anchor):
            raise ValueError(f"Selection registration upstream anchor drift: {name}")
    current = _load_registry(root)
    current_ids = frozenset(str(job["job_id"]) for job in current["jobs"])
    final_ids = frozenset(str(job["job_id"]) for job in final_registry["jobs"])
    development_ids = frozenset(
        str(job["job_id"])
        for job in final_registry["jobs"]
        if job["experiment_stage"] == DEVELOPMENT_STAGE
    )
    if current_ids not in {development_ids, final_ids}:
        raise ValueError("Selection registration journal conflicts with registry jobs")
    config_rows = list(journal.get("config_rows") or [])
    if not config_rows:
        raise ValueError("Selection registration journal lacks config hashes")
    for row in config_rows:
        target = Path(str(row.get("path", "")))
        if (
            not target.is_file()
            or _sha256_file(target) != row.get("sha256")
            or target.stat().st_size != int(row.get("size_bytes", -1))
        ):
            raise ValueError(f"Selection registration input drift: {target}")
    for job in final_registry["jobs"]:
        if job["experiment_stage"] != REFIT_STAGE:
            continue
        status_path = _job_status_path(root, str(job["job_id"]))
        if status_path.is_file():
            status = _require_mapping(_read_json(status_path), "refit job status")
            if (
                status.get("job_id") != job["job_id"]
                or status.get("experiment_stage") != REFIT_STAGE
                or status.get("config_sha256") != job["config_sha256"]
            ):
                raise ValueError(f"Interrupted refit status drift: {job['job_id']}")
        else:
            _write_json(status_path, _initial_job_status(job))
    _write_hash_rows(root / "config_hashes.csv", config_rows)
    final_registry["config_hashes_sha256"] = _sha256_file(root / "config_hashes.csv")
    _write_json(root / "registry" / "jobs.json", final_registry)
    _write_registry_exports(root)
    _write_experiment_status(
        root,
        "selection_frozen",
        current_stage=REFIT_STAGE,
        selection_sha256=final_registry["selection_sha256"],
    )
    _validate_root_lineage(root)
    _validate_registered_refit_freeze(root)
    journal_path.unlink()


def _register_frozen_refit_recipes(
    root: Path, selection: Mapping[str, Any] | str | Path
) -> Path:
    """Validate analysis-owned recipes, register their hashes, and add Stage B."""

    _validate_root_lineage(root)
    registry = _load_registry(root)
    if bool(registry.get("selection_frozen")):
        return _validate_registered_refit_freeze(root)
    _validate_stage_complete(root, DEVELOPMENT_STAGE, dry_run=False)
    development_jobs = [
        dict(job)
        for job in registry["jobs"]
        if job["experiment_stage"] == DEVELOPMENT_STAGE
    ]
    if len(development_jobs) != 16:
        raise RuntimeError("Selection requires the complete 16-cell development matrix")
    canonical_selection = root / "analysis" / "film_critic_q3_selection.json"
    if isinstance(selection, Mapping):
        selection_path = canonical_selection
        if not selection_path.is_file():
            raise FileNotFoundError(
                f"Analysis did not write canonical selection: {selection_path}"
            )
        if _require_mapping(
            _read_json(selection_path), "canonical Q3 selection"
        ) != dict(selection):
            raise ValueError("Returned Q3 selection differs from canonical file")
    else:
        selection_path = Path(selection).resolve()
        if not selection_path.is_file():
            raise FileNotFoundError(selection_path)
        if selection_path != canonical_selection.resolve():
            raise ValueError("Q3 selection must use the canonical analysis path")
    selection_payload = _require_mapping(
        _read_json(selection_path), "canonical Q3 selection"
    )
    selection_sha = str(selection_payload.get("selection_sha256", ""))
    selection_body = {
        key: value
        for key, value in selection_payload.items()
        if key != "selection_sha256"
    }
    if not selection_sha or _payload_sha256(selection_body) != selection_sha:
        raise ValueError("Canonical Q3 selection self-hash mismatch")
    recipes: dict[str, Path] = {}
    recipe_rows: list[dict[str, Any]] = []
    recipe_directory = root / "analysis" / "refit_recipes"
    for job in development_jobs:
        path = recipe_directory / f"{job['job_id']}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Analysis did not freeze recipe: {path}")
        _validate_refit_recipe(_require_mapping(_read_json(path), "refit recipe"))
        recipes[str(job["job_id"])] = path
        recipe_rows.append(_manifest_row(f"refit_recipe_{job['job_id']}", path))
    unexpected = sorted(
        path.name
        for path in recipe_directory.glob("*.json")
        if path.stem not in recipes
    )
    if unexpected:
        raise ValueError(f"Unexpected analysis-owned refit recipes: {unexpected}")
    recipe_manifest = root / "analysis" / "refit_recipe_manifest.json"
    if not recipe_manifest.is_file():
        raise FileNotFoundError(recipe_manifest)
    _validate_analysis_recipe_manifest(
        recipe_manifest,
        selection_path=selection_path,
        recipes=recipes,
    )
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY),
        ROOT_KEY,
    )
    refit_jobs, refit_config_rows = _build_jobs(
        resolved,
        root,
        stage=REFIT_STAGE,
        slots_per_gpu=int(registry["slots_per_gpu"]),
        recipes=recipes,
    )
    all_jobs = development_jobs + refit_jobs
    if len(all_jobs) != 32:
        raise RuntimeError(
            "Frozen registry must contain 16 development + 16 refit jobs"
        )
    final_registry = deepcopy(registry)
    final_registry.update(
        {
            "status": "selection_frozen",
            "selection_frozen": True,
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": _sha256_file(selection_path),
            "refit_recipe_manifest_path": str(recipe_manifest.resolve()),
            "refit_recipe_manifest_sha256": _sha256_file(recipe_manifest),
            "refit_job_count": 16,
            "jobs": all_jobs,
            "selection_frozen_at_utc": _utc_now(),
        }
    )
    with (root / "config_hashes.csv").open("r", encoding="utf-8", newline="") as handle:
        config_rows = list(csv.DictReader(handle))
    config_rows.extend(
        [
            _manifest_row("q3_selection", selection_path),
            _manifest_row("refit_recipe_manifest", recipe_manifest),
            *recipe_rows,
            *refit_config_rows,
        ]
    )
    roles = [str(row["source_role"]) for row in config_rows]
    paths = [str(Path(row["path"]).resolve()) for row in config_rows]
    if len(roles) != len(set(roles)) or len(paths) != len(set(paths)):
        raise ValueError("Frozen config manifest contains duplicate roles/paths")
    journal = {
        "schema_version": 1,
        "transaction": "selection_registration_v1",
        "final_registry": final_registry,
        "config_rows": config_rows,
    }
    journal["payload_sha256"] = _payload_sha256(journal)
    _write_json(_selection_registration_journal_path(root), journal)
    _recover_selection_registration(root)
    return selection_path


def freeze_factorial_selection(
    output_dir: str | Path,
    *,
    analysis_hook: Callable[[Path], Mapping[str, Any] | Path] | None = None,
    recipe_hook: Callable[
        [Path, Mapping[str, Any] | str | Path], Mapping[str, Any] | Path
    ]
    | None = None,
) -> Path:
    root = Path(output_dir).resolve(strict=False)
    _recover_selection_registration(root)
    _validate_root_lineage(root)
    registry = _load_registry(root)
    if bool(registry.get("selection_frozen")):
        _validate_registered_refit_freeze(root)
        return root
    _validate_stage_complete(root, DEVELOPMENT_STAGE, dry_run=False)
    development_jobs = [
        dict(job)
        for job in registry["jobs"]
        if job["experiment_stage"] == DEVELOPMENT_STAGE
    ]
    if len(development_jobs) != 16:
        raise RuntimeError("Selection requires the complete 16-cell development matrix")
    recovered = _recoverable_analysis_freeze_candidate(root, development_jobs)
    if recovered is None:
        selection = (analysis_hook or run_film_critic_q3_analysis)(root)
        (recipe_hook or freeze_refit_recipes)(root, selection)
        recovered = _recoverable_analysis_freeze_candidate(root, development_jobs)
        if recovered is None:
            raise RuntimeError("Selection hooks did not create the frozen artifacts")
    _register_frozen_refit_recipes(root, recovered)
    return root


def _freeze_q4_allowlist(root: Path) -> Path:
    registry = _load_registry(root)
    if not bool(registry.get("selection_frozen")):
        raise RuntimeError("Q4 allowlist requires frozen Q3 selection")
    _validate_stage_complete(root, REFIT_STAGE, dry_run=False)
    rows: list[dict[str, Any]] = []
    for job in registry["jobs"]:
        if job["experiment_stage"] != REFIT_STAGE:
            continue
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        for role in ("generator_final", "discriminator_final"):
            checkpoint = _artifact_path(status, role)
            rows.append(
                {
                    "job_id": job["job_id"],
                    "generator_conditioning_mode": job["generator_conditioning_mode"],
                    "critic_conditioning_mode": job["critic_conditioning_mode"],
                    "text_ablation_mode": job["text_ablation_mode"],
                    "tolerance_minutes": job["tolerance_minutes"],
                    "seed": job["seed"],
                    "checkpoint_role": role,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": _sha256_file(checkpoint),
                    "q4_allowed": True,
                }
            )
    if len(rows) != 32 or len({row["checkpoint_path"] for row in rows}) != 32:
        raise RuntimeError(
            "Q4 allowlist requires exactly two unique checkpoints per refit"
        )
    path = _write_csv(root / "q4_checkpoint_allowlist.csv", rows, tuple(rows[0]))
    registry.update(
        {
            "status": "refit_complete_q4_locked",
            "refit_complete": True,
            "q4_allowlist_path": str(path.resolve()),
            "q4_allowlist_sha256": _sha256_file(path),
            "q4_gate_open": False,
            "q4_window_materialized": False,
            "q4_loader_created": False,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
            "refit_completed_at_utc": _utc_now(),
        }
    )
    _write_json(root / "registry" / "jobs.json", registry)
    _write_experiment_status(
        root,
        "refit_complete_q4_locked",
        current_stage="q4_locked",
        q4_allowlist_sha256=_sha256_file(path),
    )
    return path


def _validate_q4_allowlist(root: Path) -> list[dict[str, str]]:
    registry = _load_registry(root)
    path = Path(str(registry.get("q4_allowlist_path", "")))
    if not path.is_file() or _sha256_file(path) != registry.get("q4_allowlist_sha256"):
        raise ValueError("Q4 checkpoint allowlist drift")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 32:
        raise ValueError("Q4 allowlist must contain exactly 32 checkpoint rows")
    for row in rows:
        checkpoint = Path(row["checkpoint_path"])
        if (
            not checkpoint.is_file()
            or _sha256_file(checkpoint) != row["checkpoint_sha256"]
            or row["q4_allowed"].lower() != "true"
        ):
            raise ValueError(f"Q4 allowlist checkpoint drift: {checkpoint}")
    return rows


def launch_factorial_development(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> Path:
    root = prepare_factorial_experiment(
        config_path, output_dir, reuse=bool(resume or Path(output_dir).exists())
    )
    registry = _load_registry(root)
    if bool(registry.get("selection_frozen")) and not dry_run:
        raise RuntimeError("Development stage is already frozen")
    _write_experiment_status(
        root,
        "development_dry_running" if dry_run else "development_running",
        current_stage=DEVELOPMENT_STAGE,
    )
    try:
        _launch_stage(
            config_path,
            root,
            stage=DEVELOPMENT_STAGE,
            resume=resume,
            dry_run=dry_run,
        )
        _write_experiment_status(
            root,
            "development_dry_run_passed" if dry_run else "development_complete",
            current_stage=DEVELOPMENT_STAGE,
        )
    except BaseException as exc:
        _write_experiment_status(
            root,
            "failed",
            current_stage=DEVELOPMENT_STAGE,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return root


def launch_factorial_refit(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> Path:
    root = Path(output_dir).resolve(strict=False)
    _validate_root_lineage(root)
    registry = _load_registry(root)
    if not bool(registry.get("selection_frozen")):
        raise RuntimeError("Refit launch requires freeze-selection first")
    if bool(registry.get("q4_gate_open")) or bool(registry.get("q4_evaluated")):
        raise RuntimeError("Refit is forbidden after Q4 access")
    if bool(registry.get("refit_complete")):
        if resume:
            _validate_q4_allowlist(root)
            return root
        raise RuntimeError("Refit is complete; use --resume for validation")
    _write_experiment_status(root, "refit_running", current_stage=REFIT_STAGE)
    try:
        _launch_stage(
            config_path,
            root,
            stage=REFIT_STAGE,
            resume=resume,
            dry_run=False,
        )
        _freeze_q4_allowlist(root)
    except BaseException as exc:
        _write_experiment_status(
            root,
            "failed",
            current_stage=REFIT_STAGE,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return root


def _materialize_q4_common_window(
    resolved: Mapping[str, Any], root: Path
) -> tuple[Path, Path]:
    import pandas as pd

    registry = _load_registry(root)
    if not bool(registry.get("q4_gate_open")):
        raise RuntimeError("Q4 materialization requires the explicit open gate")
    if not bool(registry.get("refit_complete")):
        raise RuntimeError("Q4 materialization requires completed refits")
    source = _dataset_path(resolved, 5)
    sheet = str(resolved["datasets"]["sheet_name"])
    frame = pd.read_excel(source, sheet_name=sheet)
    timestamp = pd.to_datetime(frame["effective_origin_utc"], errors="coerce", utc=True)
    if timestamp.isna().any():
        raise ValueError("Invalid Q4 timestamps in common 5m workbook")
    start = pd.Timestamp(resolved["split"]["q4_start_utc"])
    end = pd.Timestamp(resolved["split"]["q4_end_utc"])
    selected = frame.loc[(timestamp >= start) & (timestamp < end)].copy()
    if selected.empty:
        raise ValueError("Q4 common window is empty")
    directory = root / "data_windows" / "q4"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "common_05m_q4.xlsx"
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        selected.to_excel(writer, sheet_name=sheet, index=False)
    supported = _supported_frame(resolved, target, 5)
    observed = _counts(supported)
    if observed != EXPECTED_Q4_COMMON_COUNTS:
        raise ValueError(
            f"Frozen Q4 common counts drifted: {observed} != "
            f"{EXPECTED_Q4_COMMON_COUNTS}"
        )
    manifest = {
        "schema_version": 1,
        "explicit_action": "evaluate-q4",
        "source_sheet_scanned_for_window_materialization": True,
        "start_utc_inclusive": str(resolved["split"]["q4_start_utc"]),
        "end_utc_exclusive": str(resolved["split"]["q4_end_utc"]),
        "tolerance_minutes": 5,
        "supported_counts": observed,
        "workbook_path": str(target.resolve()),
        "workbook_sha256": _sha256_file(target),
        "q4_window_materialized": True,
        "q4_loader_created": False,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "materialized_at_utc": _utc_now(),
    }
    manifest["payload_sha256"] = _payload_sha256(manifest)
    manifest_path = _write_json(directory / "q4_window_manifest.json", manifest)
    return target, manifest_path


def evaluate_factorial_q4(
    output_dir: str | Path,
    *,
    resume: bool = False,
    analysis_hook: Callable[[Path], Mapping[str, Any] | Path] | None = None,
) -> Path:
    root = Path(output_dir).resolve(strict=False)
    resolved = _validate_root_lineage(root)
    registry = _load_registry(root)
    if not bool(registry.get("refit_complete")):
        raise RuntimeError("evaluate-q4 requires the completed 16-cell refit matrix")
    _validate_q4_allowlist(root)
    if bool(registry.get("q4_evaluated")):
        if resume:
            _validate_completed_q4_outputs(root)
            return root
        raise RuntimeError("Q4 has already been evaluated; use --resume to validate")
    registry.update(
        {
            "status": "q4_gate_open",
            "q4_gate_open": True,
            "q4_gate_opened_at_utc": _utc_now(),
        }
    )
    _write_json(root / "registry" / "jobs.json", registry)
    _write_experiment_status(root, "q4_gate_open", current_stage="q4_evaluation")
    try:
        q4_window, manifest_path = _materialize_q4_common_window(resolved, root)
        registry = _load_registry(root)
        registry.update(
            {
                "q4_window_materialized": True,
                "q4_window_path": str(q4_window.resolve()),
                "q4_window_sha256": _sha256_file(q4_window),
                "q4_window_manifest_path": str(manifest_path.resolve()),
                "q4_window_manifest_sha256": _sha256_file(manifest_path),
            }
        )
        _write_json(root / "registry" / "jobs.json", registry)
        result = (analysis_hook or run_film_critic_q4_analysis)(root)
        result_path = None if isinstance(result, Mapping) else Path(result)
        result_sha = (
            _payload_sha256(dict(result))
            if isinstance(result, Mapping)
            else _sha256_file(result_path)
        )
        manifest = _require_mapping(_read_json(manifest_path), "Q4 window manifest")
        manifest.pop("payload_sha256", None)
        manifest.update(
            {
                "q4_loader_created": True,
                "q4_predictions_generated": True,
                "q4_evaluated": True,
                "analysis_result_sha256": result_sha,
                "evaluated_at_utc": _utc_now(),
            }
        )
        manifest["payload_sha256"] = _payload_sha256(manifest)
        _write_json(manifest_path, manifest)
        registry = _load_registry(root)
        registry.update(
            {
                "status": "q4_evaluated",
                "q4_loader_created": True,
                "q4_predictions_generated": True,
                "q4_evaluated": True,
                "q4_analysis_result_sha256": result_sha,
                "q4_window_manifest_sha256": _sha256_file(manifest_path),
                "q4_evaluated_at_utc": _utc_now(),
            }
        )
        _write_json(root / "registry" / "jobs.json", registry)
        _write_experiment_status(
            root,
            "q4_evaluated",
            current_stage="q4_evaluation",
            q4_window_materialized=True,
            q4_loader_created=True,
            q4_predictions_generated=True,
            q4_evaluated=True,
        )
    except BaseException as exc:
        _write_experiment_status(
            root,
            "failed_after_q4_gate_open",
            current_stage="q4_evaluation",
            q4_window_materialized=(root / "data_windows" / "q4").exists(),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return root


RESOURCE_SUMMARY_FIELDS = (
    "scope",
    "experiment_stage",
    "job_id",
    "wave",
    "gpu_id",
    "gpu_slot",
    "started_at_utc",
    "completed_at_utc",
    "wall_hours",
    "job_process_hours",
    "physical_gpu_hours",
    "peak_memory_mib",
    "mean_utilization_gpu_pct",
    "peak_utilization_gpu_pct",
    "sample_count",
)


def _write_factorial_resource_summary(root: Path) -> tuple[Path, Path]:
    """Separate summed job process time from non-duplicated physical GPU time."""

    samples: list[dict[str, Any]] = []
    usage_path = root / "resource_usage.csv"
    if usage_path.is_file():
        with usage_path.open("r", encoding="utf-8", newline="") as handle:
            samples = list(csv.DictReader(handle))
    jobs = [
        dict(job)
        for job in _load_registry(root)["jobs"]
        if job["experiment_stage"] in {DEVELOPMENT_STAGE, REFIT_STAGE}
    ]
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, int], list[tuple[dict[str, Any], Any, Any]]] = {}
    total_process_hours = 0.0
    for job in jobs:
        status = _require_mapping(
            _read_json(_job_status_path(root, str(job["job_id"]))), "job status"
        )
        if status.get("status") != "completed":
            continue
        started = training._parse_utc(status.get("started_at_utc"))
        completed = training._parse_utc(status.get("completed_at_utc"))
        if started is None or completed is None or completed < started:
            raise ValueError(f"Invalid resource timestamps: {job['job_id']}")
        process_hours = (completed - started).total_seconds() / 3600.0
        total_process_hours += process_hours
        rows.append(
            {
                "scope": "job_process",
                "experiment_stage": job["experiment_stage"],
                "job_id": job["job_id"],
                "wave": job["wave"],
                "gpu_id": job["gpu_id"],
                "gpu_slot": job["gpu_slot"],
                "started_at_utc": status["started_at_utc"],
                "completed_at_utc": status["completed_at_utc"],
                "wall_hours": process_hours,
                "job_process_hours": process_hours,
                "physical_gpu_hours": "",
                "peak_memory_mib": "",
                "mean_utilization_gpu_pct": "",
                "peak_utilization_gpu_pct": "",
                "sample_count": "",
            }
        )
        key = (
            str(job["experiment_stage"]),
            int(job["wave"]),
            int(job["gpu_id"]),
        )
        grouped.setdefault(key, []).append((job, started, completed))
    total_physical_gpu_hours = 0.0
    physical_rows: list[dict[str, Any]] = []
    for (stage, wave, gpu_id), values in sorted(grouped.items()):
        started = min(value[1] for value in values)
        completed = max(value[2] for value in values)
        wall_hours = (completed - started).total_seconds() / 3600.0
        total_physical_gpu_hours += wall_hours
        selected_samples = []
        for sample in samples:
            sample_time = training._parse_utc(sample.get("timestamp_utc"))
            if (
                str(sample.get("sample_status")) == "ok"
                and int(sample.get("wave", -1)) == wave
                and int(sample.get("gpu_index", -1)) == gpu_id
                and sample_time is not None
                and started <= sample_time <= completed
            ):
                selected_samples.append(sample)
        memory = [float(sample["memory_used_mib"]) for sample in selected_samples]
        utilization = [
            float(sample["utilization_gpu_pct"]) for sample in selected_samples
        ]
        physical_rows.append(
            {
                "scope": "physical_gpu_wave",
                "experiment_stage": stage,
                "job_id": "",
                "wave": wave,
                "gpu_id": gpu_id,
                "gpu_slot": "",
                "started_at_utc": started.isoformat(),
                "completed_at_utc": completed.isoformat(),
                "wall_hours": wall_hours,
                "job_process_hours": "",
                "physical_gpu_hours": wall_hours,
                "peak_memory_mib": max(memory) if memory else "",
                "mean_utilization_gpu_pct": (
                    sum(utilization) / len(utilization) if utilization else ""
                ),
                "peak_utilization_gpu_pct": max(utilization) if utilization else "",
                "sample_count": len(selected_samples),
            }
        )
    rows.extend(physical_rows)
    csv_path = _write_csv(root / "resource_summary.csv", rows, RESOURCE_SUMMARY_FIELDS)
    summary = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "accounting_contract": {
            "job_process_hours": "sum_of_each_worker_elapsed_hours",
            "physical_gpu_hours": (
                "sum_once_per_physical_gpu_per_stage_wave_elapsed_window"
            ),
            "gpu_telemetry_scope": "physical_gpu_stage_wave_not_per_job",
            "do_not_sum_job_process_hours_as_gpu_hours": True,
        },
        "completed_job_count": len(
            [row for row in rows if row["scope"] == "job_process"]
        ),
        "physical_gpu_wave_count": len(physical_rows),
        "total_job_process_hours": total_process_hours,
        "total_physical_gpu_hours": total_physical_gpu_hours,
        "resource_summary_csv": str(csv_path.resolve()),
        "resource_summary_csv_sha256": _sha256_file(csv_path),
        "raw_resource_usage_path": str(usage_path.resolve())
        if usage_path.is_file()
        else "",
        "raw_resource_usage_sha256": _sha256_file(usage_path)
        if usage_path.is_file()
        else "",
        "created_at_utc": _utc_now(),
    }
    summary["payload_sha256"] = _payload_sha256(summary)
    json_path = _write_json(root / "resource_summary.json", summary)
    return csv_path, json_path


def _terminal_deliverable_paths(root: Path) -> list[tuple[str, Path]]:
    deliverables: dict[str, tuple[str, Path]] = {}
    role_paths: dict[str, str] = {}

    def add(role: str, path: Path) -> None:
        target = path.resolve(strict=False)
        if target.is_file() and target != (root / "output_hashes.csv").resolve():
            target_key = str(target)
            if role in role_paths:
                raise ValueError(f"Duplicate terminal artifact role: {role}")
            if target_key in deliverables:
                previous = deliverables[target_key][0]
                raise ValueError(
                    f"Duplicate terminal artifact path: {target} ({previous}, {role})"
                )
            role_paths[role] = target_key
            deliverables[target_key] = (role, target)

    registry = _load_registry(root)
    for job in registry["jobs"]:
        status_path = _job_status_path(root, str(job["job_id"]))
        add(f"job_status:{job['job_id']}", status_path)
        if status_path.is_file():
            status = _read_json(status_path)
            for artifact in status.get("artifacts") or []:
                add(
                    f"job_artifact:{job['job_id']}:{artifact['artifact_role']}",
                    Path(str(artifact["path"])),
                )
    for directory in (root / "analysis", root / "report", root / "data_windows"):
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                add(f"derived:{path.relative_to(root).as_posix()}", path)
    for relative in (
        "code_hashes.csv",
        "config_hashes.csv",
        "source_hashes.csv",
        "model_contract_manifest.json",
        "qa.json",
        "q4_checkpoint_allowlist.csv",
        "resolved_config.yaml",
        "resource_usage.csv",
        "resource_summary.csv",
        "resource_summary.json",
        "rolling_split_manifest.csv",
        "task_registry.csv",
        "registry/experiment_status.json",
        "registry/final_registry_snapshot.json",
    ):
        add(f"terminal:{relative}", root / relative)
    return [deliverables[key] for key in sorted(deliverables)]


def _required_prediction_paths(root: Path) -> set[str]:
    required: set[str] = set()
    expectations = {
        "analysis/predictions/q3_development_best_learned": 16,
        "analysis/predictions/q4_refit_final_05m": 8,
        "analysis/predictions/q4_refit_final_30m": 8,
    }
    for relative, expected_count in expectations.items():
        directory = root / relative
        predictions = sorted(directory.glob("*.csv.gz"))
        manifests = sorted(directory.glob("*.manifest.json"))
        if len(predictions) != expected_count or len(manifests) != expected_count:
            raise ValueError(
                f"Terminal prediction cache is incomplete: {relative} requires "
                f"{expected_count} CSVs and manifests"
            )
        if {path.name.removesuffix(".csv.gz") for path in predictions} != {
            path.name.removesuffix(".manifest.json") for path in manifests
        }:
            raise ValueError(
                f"Terminal prediction CSV/manifest pairing drift: {relative}"
            )
        required.update(path.relative_to(root).as_posix() for path in predictions)
        required.update(path.relative_to(root).as_posix() for path in manifests)
    return required


def _required_terminal_paths(root: Path) -> set[str]:
    registry = _load_registry(root)
    if _selection_registration_journal_path(root).exists():
        raise ValueError("Terminal state contains a pending selection transaction")
    development = [
        job for job in registry["jobs"] if job["experiment_stage"] == DEVELOPMENT_STAGE
    ]
    refit = [job for job in registry["jobs"] if job["experiment_stage"] == REFIT_STAGE]
    if len(registry["jobs"]) != 32 or len(development) != 16 or len(refit) != 16:
        raise ValueError(
            "Terminal registry requires exactly 16 development + 16 refit jobs"
        )
    required: set[str] = {
        "analysis/film_critic_q3_pair_metrics.csv.gz",
        "analysis/film_critic_q3_cell_scores.csv",
        "analysis/film_critic_q3_architecture_contrasts.csv",
        "analysis/film_critic_q3_text_contrasts.csv",
        "analysis/film_critic_q3_factorial_effects.csv",
        "analysis/film_critic_q3_selection.json",
        "analysis/refit_recipe_manifest.json",
        "analysis/film_critic_q4_pair_metrics.csv.gz",
        "analysis/film_critic_q4_primary_contrasts.csv",
        "analysis/film_critic_q4_30m_secondary.csv",
        "analysis/film_critic_q4_summary.json",
        "data_windows/q4/common_05m_q4.xlsx",
        "data_windows/q4/q4_window_manifest.json",
        "q4_checkpoint_allowlist.csv",
        "report/film_critic_factorial_conclusion.md",
        "report/film_critic_factorial_conclusion.html",
        "resource_usage.csv",
        "resource_summary.csv",
        "resource_summary.json",
        "qa.json",
        "code_hashes.csv",
        "config_hashes.csv",
        "source_hashes.csv",
        "model_contract_manifest.json",
        "resolved_config.yaml",
        "rolling_split_manifest.csv",
        "task_registry.csv",
        "registry/experiment_status.json",
        "registry/final_registry_snapshot.json",
    }
    required.update(_required_prediction_paths(root))
    development_ids = {str(job["job_id"]) for job in development}
    recipe_paths = sorted((root / "analysis" / "refit_recipes").glob("*.json"))
    if (
        len(recipe_paths) != 16
        or {path.stem for path in recipe_paths} != development_ids
    ):
        raise ValueError("Terminal freeze requires exactly 16 development recipes")
    required.update(path.relative_to(root).as_posix() for path in recipe_paths)
    for job in registry["jobs"]:
        job_id = str(job["job_id"])
        status_path = _job_status_path(root, job_id)
        if not status_path.is_file():
            raise ValueError(f"Terminal job status is missing: {job_id}")
        status = _require_mapping(_read_json(status_path), "terminal job status")
        if (
            status.get("status") != "completed"
            or status.get("job_id") != job_id
            or status.get("experiment_stage") != job["experiment_stage"]
            or status.get("config_sha256") != job["config_sha256"]
        ):
            raise ValueError(f"Terminal job status is not completed/bound: {job_id}")
        expected_roles = (
            DEVELOPMENT_ARTIFACT_ROLES
            if job["experiment_stage"] == DEVELOPMENT_STAGE
            else REFIT_ARTIFACT_ROLES
        )
        artifacts = list(status.get("artifacts") or [])
        artifact_roles = [str(row.get("artifact_role", "")) for row in artifacts]
        artifact_paths = [
            str(Path(str(row.get("path", ""))).resolve()) for row in artifacts
        ]
        if (
            set(artifact_roles) != set(expected_roles)
            or len(artifact_roles) != len(expected_roles)
            or len(artifact_paths) != len(set(artifact_paths))
        ):
            raise ValueError(
                f"Terminal job artifact-role/path contract drift: {job_id}"
            )
        required.add(status_path.relative_to(root).as_posix())
        for artifact in artifacts:
            artifact_path = Path(str(artifact["path"])).resolve()
            if not artifact_path.is_relative_to(root):
                raise ValueError(
                    f"Terminal job artifact escapes experiment root: {artifact_path}"
                )
            if (
                not artifact_path.is_file()
                or _sha256_file(artifact_path) != artifact.get("sha256")
                or artifact_path.stat().st_size != int(artifact.get("size_bytes", -1))
            ):
                raise ValueError(f"Terminal job artifact drift: {artifact_path}")
            required.add(artifact_path.relative_to(root).as_posix())
    return required


def _finalize_terminal_lineage(root: Path) -> Path:
    registry = _load_registry(root)
    snapshot = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "registry": registry,
        "experiment_status": _read_json(root / "registry" / "experiment_status.json"),
        "snapshot_excludes_live_terminal_manifest_anchor": True,
        "created_at_utc": _utc_now(),
    }
    snapshot["payload_sha256"] = _payload_sha256(snapshot)
    _write_json(root / "registry" / "final_registry_snapshot.json", snapshot)
    deliverables = _terminal_deliverable_paths(root)
    available = {path.relative_to(root).as_posix() for _, path in deliverables}
    missing = sorted(_required_terminal_paths(root) - available)
    if missing:
        raise ValueError(
            f"Cannot commit incomplete terminal output manifest: {missing}"
        )
    rows = []
    for role, path in deliverables:
        rows.append(
            {
                "artifact_role": role,
                "relative_path": (
                    path.relative_to(root).as_posix()
                    if path.is_relative_to(root)
                    else ""
                ),
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not rows:
        raise RuntimeError("Cannot finalize an empty terminal output manifest")
    manifest_path = _write_csv(root / "output_hashes.csv", rows, tuple(rows[0]))
    registry = _load_registry(root)
    registry.update(
        {
            "terminal_output_manifest_path": str(manifest_path.resolve()),
            "terminal_output_manifest_sha256": _sha256_file(manifest_path),
            "terminal_output_artifact_count": len(rows),
            "terminal_output_manifest_excludes_itself": True,
            "terminal_lineage_finalized_at_utc": _utc_now(),
        }
    )
    _write_json(root / "registry" / "jobs.json", registry)
    return manifest_path


def _validate_terminal_output_manifest(root: Path) -> list[dict[str, str]]:
    registry = _load_registry(root)
    path = Path(str(registry.get("terminal_output_manifest_path", "")))
    if (
        not path.is_file()
        or path.resolve() != (root / "output_hashes.csv").resolve()
        or _sha256_file(path) != registry.get("terminal_output_manifest_sha256")
    ):
        raise ValueError("Terminal output manifest path/hash drift")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(registry.get("terminal_output_artifact_count", -1)):
        raise ValueError("Terminal output manifest row-count drift")
    roles = [row["artifact_role"] for row in rows]
    absolute_paths = [str(Path(row["path"]).resolve()) for row in rows]
    relative_paths = [row["relative_path"] for row in rows]
    if (
        len(roles) != len(set(roles))
        or len(absolute_paths) != len(set(absolute_paths))
        or len(relative_paths) != len(set(relative_paths))
    ):
        raise ValueError("Terminal output manifest role/path uniqueness drift")
    if any(Path(row["path"]).resolve() == path.resolve() for row in rows):
        raise ValueError("Terminal output manifest must not include itself")
    for row in rows:
        artifact = Path(row["path"]).resolve()
        relative_path = str(row["relative_path"])
        if (
            not relative_path
            or not artifact.is_relative_to(root)
            or (root / relative_path).resolve() != artifact
        ):
            raise ValueError(f"Terminal output artifact path drift: {artifact}")
        if (
            not artifact.is_file()
            or _sha256_file(artifact) != row["sha256"]
            or artifact.stat().st_size != int(row["size_bytes"])
        ):
            raise ValueError(f"Terminal output artifact drift: {artifact}")
    relative = set(relative_paths)
    required = _required_terminal_paths(root)
    missing = sorted(required - relative)
    if missing:
        raise ValueError(f"Terminal output manifest is incomplete: {missing}")
    expected = {
        (role, path.relative_to(root).as_posix(), str(path))
        for role, path in _terminal_deliverable_paths(root)
    }
    observed = {
        (row["artifact_role"], row["relative_path"], str(Path(row["path"]).resolve()))
        for row in rows
    }
    if observed != expected:
        raise ValueError("Terminal output manifest does not exactly cover deliverables")
    qa = _require_mapping(_read_json(root / "qa.json"), "terminal QA")
    qa_sha = str(qa.get("payload_sha256", ""))
    qa_unsigned = {key: value for key, value in qa.items() if key != "payload_sha256"}
    if (
        qa.get("status") != "passed"
        or not qa_sha
        or _payload_sha256(qa_unsigned) != qa_sha
        or bool(qa.get("terminal_output_manifest_pending", True))
        or int(qa.get("terminal_output_artifact_count", 0)) != len(rows)
    ):
        raise ValueError("Terminal QA does not describe the finalized manifest")
    return rows


def _validate_historical_q4_exposure(summary: Mapping[str, Any]) -> None:
    historical = _require_mapping(
        summary.get("historical_q4_exposure"), "historical Q4 exposure"
    )
    historical_path = Path(str(historical.get("historical_prediction_path", "")))
    if (
        not historical_path.is_file()
        or _sha256_file(historical_path)
        != str(historical.get("historical_prediction_sha256", ""))
        or int(historical.get("current_pair_count", -1)) != 143
        or int(historical.get("overlapping_pair_count", -1)) != 143
        or str(historical.get("pair_overlap_label", "")) != "143/143"
        or not bool(historical.get("historically_exposed"))
        or str(historical.get("interpretation", ""))
        != "retrospective_frozen_exploratory_not_confirmatory"
    ):
        raise ValueError("Historical Q4 exposure evidence/path/hash drift")


def _validate_completed_q4_outputs(root: Path) -> None:
    registry = _load_registry(root)
    _validate_q4_allowlist(root)
    summary_path = root / "analysis" / "film_critic_q4_summary.json"
    summary = _require_mapping(_read_json(summary_path), "Q4 summary")
    summary_sha = str(summary.get("summary_sha256", ""))
    unsigned_summary = {
        key: value for key, value in summary.items() if key != "summary_sha256"
    }
    if not summary_sha or _payload_sha256(unsigned_summary) != summary_sha:
        raise ValueError("Q4 summary self-hash mismatch")
    if registry.get("q4_analysis_result_sha256") != _payload_sha256(summary):
        raise ValueError("Q4 summary differs from the registry result hash")
    if not all(
        bool(summary.get(field))
        for field in (
            "q4_loader_created",
            "q4_predictions_generated",
            "q4_evaluated",
        )
    ):
        raise ValueError("Q4 summary does not report completed evaluation")
    _validate_historical_q4_exposure(summary)
    for relative, expected in _require_mapping(
        summary.get("artifact_sha256"), "Q4 artifact hashes"
    ).items():
        artifact = root / "analysis" / str(relative)
        if not artifact.is_file() or _sha256_file(artifact) != str(expected):
            raise ValueError(f"Q4 analysis artifact drift: {artifact}")
    prediction_manifests = sorted(
        path
        for directory in (
            root / "analysis" / "predictions" / "q4_refit_final_05m",
            root / "analysis" / "predictions" / "q4_refit_final_30m",
        )
        for path in directory.glob("*.manifest.json")
    )
    if len(prediction_manifests) != 16:
        raise ValueError("Completed Q4 evaluation requires 16 prediction manifests")
    for manifest_path in prediction_manifests:
        manifest = _require_mapping(_read_json(manifest_path), "Q4 prediction manifest")
        digest = str(manifest.get("manifest_sha256", ""))
        unsigned = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        if not digest or _payload_sha256(unsigned) != digest:
            raise ValueError(f"Q4 prediction manifest self-hash drift: {manifest_path}")
        prediction = Path(str(manifest.get("prediction_path", "")))
        if not prediction.is_file() or _sha256_file(prediction) != manifest.get(
            "prediction_sha256"
        ):
            raise ValueError(f"Q4 prediction cache drift: {prediction}")
    window_manifest_path = Path(str(registry.get("q4_window_manifest_path", "")))
    window_path = Path(str(registry.get("q4_window_path", "")))
    if not window_path.is_file() or _sha256_file(window_path) != registry.get(
        "q4_window_sha256"
    ):
        raise ValueError("Q4 common workbook path/hash drift")
    if not window_manifest_path.is_file() or _sha256_file(
        window_manifest_path
    ) != registry.get("q4_window_manifest_sha256"):
        raise ValueError("Q4 window manifest path/hash drift")
    window = _require_mapping(_read_json(window_manifest_path), "Q4 window manifest")
    window_sha = str(window.get("payload_sha256", ""))
    unsigned_window = {
        key: value for key, value in window.items() if key != "payload_sha256"
    }
    if not window_sha or _payload_sha256(unsigned_window) != window_sha:
        raise ValueError("Q4 window manifest self-hash mismatch")
    if not all(
        bool(window.get(field))
        for field in (
            "q4_window_materialized",
            "q4_loader_created",
            "q4_predictions_generated",
            "q4_evaluated",
        )
    ):
        raise ValueError("Q4 window manifest is not terminal")
    if registry.get("terminal_output_manifest_path"):
        _validate_terminal_output_manifest(root)


def postprocess_factorial_experiment(
    output_dir: str | Path,
    *,
    report_hook: Callable[[Path], Path] | None = None,
) -> Path:
    root = Path(output_dir).resolve(strict=False)
    _validate_root_lineage(root)
    registry = _load_registry(root)
    if registry.get("status") == "completed" and registry.get(
        "terminal_output_manifest_path"
    ):
        _validate_completed_q4_outputs(root)
        _validate_terminal_output_manifest(root)
        return root
    if not bool(registry.get("q4_evaluated")):
        raise RuntimeError("postprocess requires the explicit Q4 evaluation first")
    _validate_completed_q4_outputs(root)
    (report_hook or render_film_critic_report)(root)
    registry = _load_registry(root)
    registry["status"] = "completed"
    registry["completed_at_utc"] = _utc_now()
    _write_json(root / "registry" / "jobs.json", registry)
    _write_experiment_status(
        root,
        "completed",
        current_stage="postprocess",
        q4_window_materialized=True,
        q4_loader_created=True,
        q4_predictions_generated=True,
        q4_evaluated=True,
    )
    _write_factorial_resource_summary(root)
    qa_factorial_experiment(root)
    _finalize_terminal_lineage(root)
    _validate_terminal_output_manifest(root)
    return root


def qa_factorial_experiment(output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve(strict=False)
    _validate_root_lineage(root)
    registry = _load_registry(root)
    development = [
        job for job in registry["jobs"] if job["experiment_stage"] == DEVELOPMENT_STAGE
    ]
    refit = [job for job in registry["jobs"] if job["experiment_stage"] == REFIT_STAGE]
    if len(development) != 16 or len(refit) not in {0, 16}:
        raise ValueError("Registry must contain 16 development and zero/16 refit jobs")
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY), ROOT_KEY
    )
    gpu_ids = tuple(map(int, resolved["runtime"]["gpu_ids"]))
    _assert_gpu_balance(development, gpu_ids=gpu_ids)
    if refit:
        _assert_gpu_balance(refit, gpu_ids=gpu_ids)
        for job in refit:
            config = _load_yaml(Path(job["training_config_path"]), "refit config")
            if any(
                bool(config[key])
                for key in (
                    "news_first_materialize_validation_loader",
                    "news_first_materialize_test_loader",
                    "use_reduce_lr_on_plateau",
                    "use_early_stopping",
                    "evaluate_initial_checkpoint",
                )
            ):
                raise ValueError(
                    f"Refit created forbidden evaluation state: {job['job_id']}"
                )
            _validate_refit_recipe(_read_json(Path(job["refit_recipe_path"])))
    q4_directory = root / "data_windows" / "q4"
    if not bool(registry.get("q4_gate_open")) and q4_directory.exists():
        raise ValueError("Q4 objects exist before the explicit gate")
    terminal_rows = 0
    if bool(registry.get("q4_evaluated")):
        _validate_completed_q4_outputs(root)
    if registry.get("terminal_output_manifest_path"):
        terminal_rows = len(_validate_terminal_output_manifest(root))
    terminal_finalizing = registry.get("status") == "completed" and bool(
        registry.get("q4_evaluated")
    )
    if terminal_finalizing and not terminal_rows:
        existing_paths = _terminal_deliverable_paths(root)
        existing = {path.resolve() for _, path in existing_paths}
        future_paths = {
            (root / "qa.json").resolve(),
            (root / "registry" / "final_registry_snapshot.json").resolve(),
        }
        terminal_rows = len(existing_paths) + len(
            [path for path in future_paths if path not in existing]
        )
    report = {
        "schema_version": 1,
        "status": "passed",
        "development_jobs": len(development),
        "refit_jobs": len(refit),
        "gpu_axes_balanced": True,
        "q4_gate_open": bool(registry.get("q4_gate_open")),
        "q4_window_materialized": bool(registry.get("q4_window_materialized")),
        "q4_loader_created": bool(registry.get("q4_loader_created")),
        "q4_predictions_generated": bool(registry.get("q4_predictions_generated")),
        "q4_evaluated": bool(registry.get("q4_evaluated")),
        "terminal_output_artifact_count": terminal_rows,
        "terminal_output_manifest_pending": not (
            terminal_finalizing or bool(registry.get("terminal_output_manifest_path"))
        ),
        "terminal_output_manifest_finalization_ready": bool(terminal_finalizing),
        "completed_at_utc": _utc_now(),
    }
    report["payload_sha256"] = _payload_sha256(report)
    qa_path = root / "qa.json"
    if registry.get("terminal_output_manifest_path"):
        if not qa_path.is_file():
            raise FileNotFoundError("Terminal QA artifact is missing")
        return qa_path
    return _write_json(qa_path, report)


def run_news_first_vol_generator_film_critic_factorial(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    action: str = "prepare",
    job_id: str = "",
    resume: bool = False,
    reuse: bool = False,
    worker_dry_run: bool = False,
    q3_analysis_hook: Callable[[Path], Mapping[str, Any] | Path] | None = None,
    q4_analysis_hook: Callable[[Path], Mapping[str, Any] | Path] | None = None,
    report_hook: Callable[[Path], Path] | None = None,
) -> Path:
    normalized = str(action).strip().lower()
    if normalized == "prepare":
        return prepare_factorial_experiment(
            config_path, output_dir, reuse=bool(reuse or resume)
        )
    if normalized == "benchmark":
        return run_factorial_benchmark(config_path, output_dir, resume=resume)
    if normalized == "dry-run":
        return launch_factorial_development(
            config_path, output_dir, resume=resume, dry_run=True
        )
    if normalized == "launch-development":
        return launch_factorial_development(
            config_path, output_dir, resume=resume, dry_run=False
        )
    if normalized == "freeze-selection":
        return freeze_factorial_selection(output_dir, analysis_hook=q3_analysis_hook)
    if normalized == "launch-refit":
        return launch_factorial_refit(config_path, output_dir, resume=resume)
    if normalized == "evaluate-q4":
        return evaluate_factorial_q4(
            output_dir, resume=resume, analysis_hook=q4_analysis_hook
        )
    if normalized == "postprocess":
        return postprocess_factorial_experiment(output_dir, report_hook=report_hook)
    if normalized == "worker":
        if not job_id:
            raise ValueError("--job-id is required for worker")
        return run_factorial_worker(
            output_dir,
            job_id,
            dry_run=worker_dry_run,
            resume=resume,
        )
    if normalized == "qa":
        return qa_factorial_experiment(output_dir)
    raise ValueError(f"Unsupported Generator/Critic factorial action: {action}")


__all__ = [
    "BENCHMARK_STAGE",
    "CRITIC_MODES",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "DEVELOPMENT_STAGE",
    "EXPERIMENT_KIND",
    "GENERATOR_MODES",
    "REFIT_MODE",
    "REFIT_STAGE",
    "build_factorial_worker_command",
    "factorial_specs",
    "freeze_refit_recipes",
    "prepare_factorial_experiment",
    "resolve_config",
    "run_news_first_vol_generator_film_critic_factorial",
]
