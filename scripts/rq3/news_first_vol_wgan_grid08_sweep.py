"""Independent 8x8 WGAN capacity-by-learning-rate experiment.

This module owns a branch-local durable schema.  It deliberately does not
reinterpret the earlier 16x16 capacity/LR experiments.  Q2 is the only
selection panel; Q3 is unlocked only after the 180-job selection and the six
diagnostic checkpoints have been frozen.  Q4 is never materialized.
"""

from __future__ import annotations

import csv
import math
import os
import socket
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from scripts.rq3 import news_first_vol_training as training
from wgan_option.models.common import (
    INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
    critic_normalization_fingerprint,
)
from wgan_option.utils.text_ablation import REAL_TEXT, text_information_path


ROOT_KEY = training.ROOT_KEY
REPO_ROOT = training.REPO_ROOT
EXPERIMENT_KIND = "wgan_grid08_capacity_lr_sweep"
GRID_PROFILE = "grid08_q097_103_ttm07_38_uniform_v1"
PRIMARY_STAGE = "q2_primary"
DIAGNOSTIC_STAGE = "q2_current_only_diagnostic"
PROFILES = ("micro", "tiny", "small", "medium", "large", "legacy")
LR_PROFILES = (
    "lr_2_5e_07",
    "lr_5e_07",
    "lr_7_5e_07",
    "lr_1e_06",
    "lr_1_5e_06",
)
LRS = {
    "lr_2_5e_07": 2.5e-7,
    "lr_5e_07": 5.0e-7,
    "lr_7_5e_07": 7.5e-7,
    "lr_1e_06": 1.0e-6,
    "lr_1_5e_06": 1.5e-6,
}
SEEDS = (42, 202, 404)
TOLERANCES = (5, 30)
EXPECTED_GRID = {
    "strike_grid": [
        0.970000,
        0.978571,
        0.987143,
        0.995714,
        1.004286,
        1.012857,
        1.021429,
        1.030000,
    ],
    "maturity_grid_days": [7, 11, 16, 20, 25, 29, 34, 38],
    "surface_shape": [8, 8],
}
EXPECTED_PANEL_COUNTS = {
    "pre_q3_05m": {"pairs": 584, "sessions": 189},
    "pre_q3_30m": {"pairs": 840, "sessions": 225},
    "common_q2": {"pairs": 95, "sessions": 31},
    "common_q3": {"pairs": 97, "sessions": 29},
}
SUCCESS = {"completed", "dry_run_passed"}

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


def _sweep(resolved: Mapping[str, Any]) -> dict[str, Any]:
    return _require_mapping(
        resolved.get("wgan_grid08_capacity_lr_sweep"),
        "wgan_grid08_capacity_lr_sweep",
    )


def _exact(value: Any, expected: float, label: str) -> None:
    observed = float(value)
    if not math.isfinite(observed) or observed != float(expected):
        raise ValueError(f"{label} is frozen to {expected}; observed={observed}")


def _validate_config(resolved: Mapping[str, Any]) -> None:
    datasets = _require_mapping(resolved.get("datasets"), "datasets")
    split = _require_mapping(resolved.get("split"), "split")
    sweep = _sweep(resolved)
    models = _require_mapping(resolved.get("models"), "models")
    runtime = _require_mapping(resolved.get("runtime"), "runtime")
    if str(sweep.get("experiment_kind")) != EXPERIMENT_KIND:
        raise ValueError(f"experiment_kind must be {EXPERIMENT_KIND}")
    if tuple(map(int, datasets.get("tolerances_minutes", ()))) != TOLERANCES:
        raise ValueError("The grid08 sweep trains only 5m and 30m tolerances")
    if int(datasets.get("common_evaluation_tolerance_minutes", -1)) != 5:
        raise ValueError("The common evaluation workbook must be 5m")
    if str(datasets.get("text_embedding_mode", "")).lower() != "lp":
        raise ValueError("LP embeddings are frozen")
    if str(datasets.get("support_mask_mode", "")).lower() != "raw_joint":
        raise ValueError("raw_joint support is frozen")
    if list(datasets.get("text_ablation_modes", ())) != [REAL_TEXT]:
        raise ValueError("The 180-job primary matrix must be real_text only")
    if int(datasets.get("seed", -1)) != 42:
        raise ValueError("datasets.seed must remain 42 for split lineage")
    frozen_dates = {
        "train_end_utc": "2023-04-01T00:00:00Z",
        "validation_end_utc": "2023-07-01T00:00:00Z",
        "q3_start_utc": "2023-07-01T00:00:00Z",
        "q3_end_utc": "2023-10-01T00:00:00Z",
    }
    for key, expected in frozen_dates.items():
        if str(split.get(key)) != expected:
            raise ValueError(f"split.{key} is frozen to {expected}")
    if int(split.get("validation_mc_samples", -1)) != 16:
        raise ValueError("Q2 validation MC is frozen to 16")
    if int(split.get("q3_mc_samples", -1)) != 64:
        raise ValueError("Q3 MC is frozen to 64")
    if tuple(sweep.get("profiles", {})) != PROFILES:
        raise ValueError(f"Capacity profiles/order must be {PROFILES}")
    if tuple(sweep.get("learning_rate_profiles", {})) != LR_PROFILES:
        raise ValueError(f"LR profiles/order must be {LR_PROFILES}")
    if tuple(map(int, sweep.get("seeds", ()))) != SEEDS:
        raise ValueError("Training seeds are frozen to 42/202/404")
    if tuple(map(int, sweep.get("tolerances_minutes", ()))) != TOLERANCES:
        raise ValueError("Sweep tolerances are frozen to 5/30")
    if str(sweep.get("surface_grid_profile")) != GRID_PROFILE:
        raise ValueError("Unexpected surface grid profile")
    if (
        str(sweep.get("critic_normalization_mode"))
        != INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE
    ):
        raise ValueError("8x8 requires the versioned GroupNorm critic tail")
    for profile, raw in sweep["profiles"].items():
        values = _require_mapping(raw, f"profiles.{profile}")
        if int(values["expected_generator_parameters"]) <= 0:
            raise ValueError(f"Invalid generator count for {profile}")
        if int(values["expected_generator_parameters"]) + int(
            values["expected_discriminator_parameters"]
        ) != int(values["expected_wgan_parameters"]):
            raise ValueError(f"WGAN parameter arithmetic mismatch for {profile}")
    for name, expected in LRS.items():
        raw = _require_mapping(sweep["learning_rate_profiles"][name], name)
        _exact(raw["initial_learning_rate"], expected, f"{name}.initial_learning_rate")
        _exact(raw["scheduler_min_lr"], expected / 10.0, f"{name}.scheduler_min_lr")
    if set(models) != {"wgan"}:
        raise ValueError("Only WGAN is permitted")
    values = _require_mapping(models["wgan"].get("training"), "wgan.training")
    frozen = {
        "embedding_dim": 1024,
        "noise_dim": 32,
        "gen_res_blocks": 0,
        "disc_res_blocks": 0,
        "batch_size": 16,
        "discriminator_iter": 5,
        "num_epochs": 100,
        "early_stopping_min_epochs": 30,
        "early_stopping_patience": 16,
        "reduce_lr_patience": 3,
    }
    for key, expected in frozen.items():
        if int(values.get(key, -1)) != expected:
            raise ValueError(f"WGAN {key} is frozen to {expected}")
    string_frozen = {
        "generator_noise_mode": "gaussian",
        "generator_current_input_mode": "current_support_masked",
        "critic_normalization_mode": INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
        "residual_output_mode": "identity_softplus_residual",
    }
    for key, expected in string_frozen.items():
        if str(values.get(key, "")).lower() != expected:
            raise ValueError(f"WGAN {key} is frozen to {expected}")
    if str(values.get("news_first_label_reliability_mode", "none")) != "none":
        raise ValueError("Label reliability weighting is excluded")
    if not bool(values.get("evaluate_initial_checkpoint")):
        raise ValueError("Epoch-0 evaluation is required")
    if not bool(values.get("use_reduce_lr_on_plateau")):
        raise ValueError("ReduceLROnPlateau is required")
    _exact(values.get("reduce_lr_factor"), 0.5, "reduce_lr_factor")
    gpu_ids = tuple(map(int, runtime.get("gpu_ids", ())))
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("Exactly two distinct GPUs are required")
    if int(runtime.get("slots_per_gpu", -1)) not in {12, 24}:
        raise ValueError("Formal concurrency must be 12 or 24 workers/GPU")


def resolve_config(config_path: str | Path) -> dict[str, Any]:
    source = _resolve_repo_path(config_path)
    root = _load_yaml(source, "grid08 sweep config")
    resolved = deepcopy(_require_mapping(root.get(ROOT_KEY), ROOT_KEY))
    resolved["datasets"]["root"] = str(_resolve_repo_path(resolved["datasets"]["root"]))
    resolved["runtime"]["python_executable"] = str(
        _resolve_repo_path(resolved["runtime"].get("python_executable", sys.executable))
    )
    resolved["source_config_path"] = str(source)
    _validate_config(resolved)
    return resolved


def _grid_contract(resolved: Mapping[str, Any]) -> dict[str, Any]:
    from wgan_option.surface_grid import build_surface_grids

    strike, maturity = build_surface_grids(
        strike_bins=8,
        maturity_bins=8,
        moneyness_min=0.97,
        moneyness_max=1.03,
        maturity_min_days=7,
        maturity_max_days=38,
        integer_maturity_days=True,
        dtype=np.float64,
    )
    observed = {
        "schema_version": 1,
        "profile": GRID_PROFILE,
        "strike_grid": [float(value) for value in strike],
        "maturity_grid_days": [int(value) for value in maturity],
        "surface_shape": [8, 8],
        "critic_normalization_mode": INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
        "critic_normalization_fingerprint": critic_normalization_fingerprint(
            INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE
        ),
    }
    for actual, expected in zip(observed["strike_grid"], EXPECTED_GRID["strike_grid"]):
        if abs(float(actual) - float(expected)) > 5e-7:
            raise ValueError(f"8x8 strike node drift: {actual} != {expected}")
    if observed["maturity_grid_days"] != EXPECTED_GRID["maturity_grid_days"]:
        raise ValueError("8x8 maturity node drift")
    observed["grid_sha256"] = _payload_sha256(
        {
            "schema_version": 1,
            "support_method": "raw_bracket_intersection_v1",
            "strike_grid": observed["strike_grid"],
            "maturity_days_grid": observed["maturity_grid_days"],
            "surface_shape": observed["surface_shape"],
        }
    )
    return observed


def _architecture_payload(name: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "gen_base_channels",
        "gen_text_hidden_dim",
        "gen_text_out_dim",
        "gen_hidden_dim",
        "disc_base_channels",
        "disc_text_hidden_dim",
        "disc_hidden_dim",
    )
    return {
        "schema_version": 1,
        "profile": name,
        **{key: int(raw[key]) for key in fields},
    }


def _model_contract(
    name: str, raw: Mapping[str, Any], grid: Mapping[str, Any]
) -> dict[str, Any]:
    architecture = _architecture_payload(name, raw)
    architecture_sha = _payload_sha256(architecture)
    payload = {
        "schema_version": 1,
        "capacity_profile": name,
        "architecture_profile_sha256": architecture_sha,
        "surface_grid_profile": GRID_PROFILE,
        "surface_grid_sha256": grid["grid_sha256"],
        "surface_shape": [8, 8],
        "critic_normalization_mode": INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
        "critic_normalization_fingerprint": grid["critic_normalization_fingerprint"],
        "embedding_mode": "lp",
        "embedding_dim": 1024,
        "noise_mode": "gaussian",
        "noise_dim": 32,
        "generator_current_input_mode": "current_support_masked",
        "gen_res_blocks": 0,
        "disc_res_blocks": 0,
        "expected_generator_parameters": int(raw["expected_generator_parameters"]),
        "expected_discriminator_parameters": int(
            raw["expected_discriminator_parameters"]
        ),
        "expected_wgan_parameters": int(raw["expected_wgan_parameters"]),
    }
    payload["model_contract_sha256"] = _payload_sha256(payload)
    payload["architecture"] = architecture
    return payload


def _lr_sha(name: str) -> str:
    return _payload_sha256(
        {
            "schema_version": 1,
            "lr_profile": name,
            "initial_learning_rate": LRS[name],
            "scheduler_min_lr": LRS[name] / 10.0,
            "factor": 0.5,
            "patience": 3,
        }
    )


def _job_id(profile: str, lr_profile: str, seed: int, mode: str, tolerance: int) -> str:
    return (
        f"grid08_wgan_{profile}_{lr_profile}_seed_{int(seed):03d}_"
        f"{mode}_{int(tolerance):02d}m"
    )


def primary_specs() -> list[dict[str, Any]]:
    # Pair 5m/30m per base cell.  Alternating the first tolerance's GPU gives
    # every factor level equal exposure to both physical devices.
    return [
        {
            "capacity_profile": profile,
            "lr_profile": lr,
            "seed": seed,
            "text_ablation_mode": REAL_TEXT,
            "tolerance_minutes": tolerance,
            "experiment_stage": PRIMARY_STAGE,
        }
        for profile in PROFILES
        for lr in LR_PROFILES
        for seed in SEEDS
        for tolerance in TOLERANCES
    ]


def _balanced_assignments(
    specs: Sequence[Mapping[str, Any]], *, gpu_ids: Sequence[int], slots_per_gpu: int
) -> list[dict[str, Any]]:
    per_wave = len(gpu_ids) * int(slots_per_gpu)
    if per_wave <= 0:
        raise ValueError("No worker slots")
    assigned: list[dict[str, Any]] = []
    per_gpu_slot = {int(gpu): 0 for gpu in gpu_ids}
    for index, raw in enumerate(specs):
        spec = dict(raw)
        pair_index = index // 2
        tolerance_index = index % 2
        first_gpu = pair_index % 2
        gpu_index = (first_gpu + tolerance_index) % 2
        gpu_id = int(gpu_ids[gpu_index])
        wave = index // per_wave + 1
        within_wave_gpu_index = per_gpu_slot[gpu_id] % int(slots_per_gpu)
        per_gpu_slot[gpu_id] += 1
        spec.update(
            {
                "wave": wave,
                "gpu_id": gpu_id,
                "gpu_slot": within_wave_gpu_index,
            }
        )
        assigned.append(spec)
    return assigned


def _dataset_path(resolved: Mapping[str, Any], tolerance: int) -> Path:
    datasets = resolved["datasets"]
    template = str(datasets["workbook_template"])
    return Path(datasets["root"]) / template.format(
        tolerance=int(tolerance), tolerance02=f"{int(tolerance):02d}"
    )


def _source_rows(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset_root = Path(resolved["datasets"]["root"])
    candidates = [
        ("dataset_summary", dataset_root / "dataset_summary.csv"),
        ("dataset_output_manifest", dataset_root / "dataset_output_sha256.txt"),
        ("dataset_manifest", dataset_root / "dataset_manifest.json"),
        ("orchestration_config", Path(resolved["source_config_path"])),
    ]
    for tolerance in TOLERANCES:
        candidates.extend(
            [
                (
                    f"training_workbook_{tolerance:02d}m",
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
    rows = []
    for role, path in candidates:
        if not path.is_file():
            if role == "dataset_manifest":
                continue
            raise FileNotFoundError(f"Missing source {role}: {path}")
        rows.append(
            {
                "source_role": role,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _materialize_pre_q3_workbooks(
    resolved: Mapping[str, Any], root: Path
) -> list[dict[str, Any]]:
    """Create physical train+Q2-only workbooks before any training process.

    This is intentionally stronger than merely not constructing a test loader:
    neither a trainer nor Q2 analysis can physically read a Q3/Q4 workbook.
    """

    import pandas as pd

    end = pd.Timestamp(resolved["split"]["validation_end_utc"])
    output_rows = []
    directory = root / "data_windows"
    directory.mkdir(parents=True, exist_ok=True)
    sheet = str(resolved["datasets"]["sheet_name"])
    for tolerance in TOLERANCES:
        source = _dataset_path(resolved, tolerance)
        frame = pd.read_excel(source, sheet_name=sheet)
        timestamps = pd.to_datetime(
            frame["effective_origin_utc"], errors="coerce", utc=True
        )
        if timestamps.isna().any():
            raise ValueError(f"Invalid timestamps in {source}")
        selected = frame.loc[timestamps < end].copy()
        if selected.empty or bool(
            (pd.to_datetime(selected["effective_origin_utc"], utc=True) >= end).any()
        ):
            raise ValueError(f"Failed to enforce the pre-Q3 window for {tolerance}m")
        target = directory / f"tolerance_{tolerance:02d}m_train_q2_only.xlsx"
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            selected.to_excel(writer, sheet_name=sheet, index=False)
        output_rows.append(
            {
                "source_role": f"materialized_train_q2_window_{tolerance:02d}m",
                "path": str(target.resolve()),
                "size_bytes": target.stat().st_size,
                "sha256": _sha256_file(target),
            }
        )
    contract = {
        "schema_version": 1,
        "origin_end_utc_exclusive": str(resolved["split"]["validation_end_utc"]),
        "q3_rows_present": 0,
        "q4_rows_present": 0,
        "workbooks": output_rows,
    }
    contract["payload_sha256"] = _payload_sha256(contract)
    _write_json(root / "data_windows" / "pre_q3_window_manifest.json", contract)
    output_rows.append(
        {
            "source_role": "pre_q3_window_manifest",
            "path": str(
                (root / "data_windows" / "pre_q3_window_manifest.json").resolve()
            ),
            "size_bytes": (root / "data_windows" / "pre_q3_window_manifest.json")
            .stat()
            .st_size,
            "sha256": _sha256_file(
                root / "data_windows" / "pre_q3_window_manifest.json"
            ),
        }
    )
    return output_rows


def _code_rows() -> list[dict[str, Any]]:
    rows = training._code_rows()
    known = {row["relative_path"] for row in rows}
    for relative in (
        "scripts/rq3/news_first_vol_wgan_grid08_sweep.py",
        "scripts/rq3/news_first_vol_wgan_grid08_analysis.py",
        "scripts/rq3/news_first_vol_wgan_grid08_report.py",
        "scripts/rq3/news_first_vol_comparison_analysis.py",
        "src/wgan_option/models/common.py",
        "src/wgan_option/models/discriminator.py",
        "src/wgan_option/utils/inference_helpers.py",
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


def _support_filtered_frame(resolved: Mapping[str, Any], tolerance: int):
    frame = training._read_split_keys(
        _dataset_path(resolved, tolerance), str(resolved["datasets"]["sheet_name"])
    )
    return training._apply_support_eligibility(
        frame,
        dataset_root=Path(resolved["datasets"]["root"]),
        tolerance=int(tolerance),
        support_mode="raw_joint",
    )


def _counts(frame: Any) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "pairs": int(frame["pair_id"].nunique()),
        "sessions": int(frame["session_id"].nunique()),
    }


def _build_split_manifest(resolved: Mapping[str, Any], root: Path) -> Path:
    import pandas as pd

    train_end = pd.Timestamp(resolved["split"]["train_end_utc"])
    validation_end = pd.Timestamp(resolved["split"]["validation_end_utc"])
    q3_end = pd.Timestamp(resolved["split"]["q3_end_utc"])
    common = _support_filtered_frame(resolved, 5)
    source_05 = _support_filtered_frame(resolved, 5)
    source_30 = _support_filtered_frame(resolved, 30)
    frames = {
        "train_05m": source_05[source_05["effective_origin_utc"] < train_end],
        "train_30m": source_30[source_30["effective_origin_utc"] < train_end],
        "pre_q3_05m": source_05[source_05["effective_origin_utc"] < validation_end],
        "pre_q3_30m": source_30[source_30["effective_origin_utc"] < validation_end],
        "common_q2": common[
            (common["effective_origin_utc"] >= train_end)
            & (common["effective_origin_utc"] < validation_end)
        ],
        "common_q3": common[
            (common["effective_origin_utc"] >= validation_end)
            & (common["effective_origin_utc"] < q3_end)
        ],
    }
    for name, expected in EXPECTED_PANEL_COUNTS.items():
        observed = _counts(frames[name])
        for key in ("pairs", "sessions"):
            if observed[key] != expected[key]:
                raise ValueError(
                    f"Frozen grid08 {name} {key} drift: {observed[key]} != {expected[key]}"
                )
    if set(frames["common_q2"]["pair_id"]) & set(frames["common_q3"]["pair_id"]):
        raise ValueError("Q2/Q3 pair leakage")
    if set(frames["common_q2"]["session_id"]) & set(frames["common_q3"]["session_id"]):
        raise ValueError("Q2/Q3 session leakage")
    rows = []
    for name, frame in frames.items():
        counts = _counts(frame)
        rows.append(
            {
                "split": name,
                "tolerance_minutes": 30 if name in {"train_30m", "pre_q3_30m"} else 5,
                **counts,
                "pair_universe_sha256": _payload_sha256(sorted(set(frame["pair_id"]))),
                "session_universe_sha256": _payload_sha256(
                    sorted(set(frame["session_id"]))
                ),
                "q3_used_for_selection": False,
                "q4_loader_created": False,
            }
        )
    return _write_csv(root / "grid08_split_manifest.csv", rows, tuple(rows[0]))


def _training_payload(
    resolved: Mapping[str, Any],
    root: Path,
    *,
    spec: Mapping[str, Any],
    model_contract: Mapping[str, Any],
    benchmark: bool = False,
) -> dict[str, Any]:
    tolerance = int(spec["tolerance_minutes"])
    mode = str(spec["text_ablation_mode"])
    payload = training._training_payload(
        resolved,
        family="wgan",
        tolerance=tolerance,
        text_ablation_mode=mode,
        multi_mode=True,
        experiment_root=root,
        capacity_profile=None,
    )
    profile = str(spec["capacity_profile"])
    raw = _sweep(resolved)["profiles"][profile]
    for field in training.CAPACITY_PROFILE_SHAPE_FIELDS:
        payload[field] = int(raw[field])
    lr_profile = str(spec["lr_profile"])
    initial_lr = LRS[lr_profile]
    payload.update(
        {
            "seed": int(spec["seed"]),
            "learning_rate": initial_lr,
            "generator_learning_rate": initial_lr,
            "discriminator_learning_rate": initial_lr,
            "reduce_lr_min_lr": initial_lr / 10.0,
            "news_first_capacity_profile": profile,
            "news_first_capacity_profile_sha256": model_contract[
                "architecture_profile_sha256"
            ],
            "news_first_architecture_profile_sha256": model_contract[
                "architecture_profile_sha256"
            ],
            "news_first_model_contract_sha256": model_contract["model_contract_sha256"],
            "news_first_surface_grid_profile": GRID_PROFILE,
            "news_first_surface_grid_sha256": model_contract["surface_grid_sha256"],
            "news_first_lr_profile": lr_profile,
            "news_first_lr_profile_sha256": _lr_sha(lr_profile),
            "news_first_label_reliability_mode": "none",
            "news_first_materialize_test_loader": False,
            "critic_normalization_mode": INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
            "data_path": str(
                root / "data_windows" / f"tolerance_{tolerance:02d}m_train_q2_only.xlsx"
            ),
            "news_first_common_eval_data_path": str(
                root / "data_windows" / "tolerance_05m_train_q2_only.xlsx"
            ),
            "news_first_data_window_start_utc_inclusive": "",
            "news_first_data_window_end_utc_exclusive": str(
                resolved["split"]["validation_end_utc"]
            ),
            "output_root": str(
                root
                / "runs"
                / "wgan"
                / "grid08"
                / profile
                / lr_profile
                / f"seed_{int(spec['seed']):03d}"
                / mode
                / f"tolerance_{tolerance:02d}m"
            ),
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


def _instantiate_profile_contracts(
    resolved: Mapping[str, Any], grid: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    from wgan_option.models.discriminator import Discriminator
    from wgan_option.models.generator import Generator

    result = {}
    for name in PROFILES:
        raw = _sweep(resolved)["profiles"][name]
        generator = Generator(
            channels=1,
            embedding_dim=1024,
            noise_dim=32,
            surface_height=8,
            surface_width=8,
            base_channels=int(raw["gen_base_channels"]),
            res_blocks=0,
            text_hidden_dim=int(raw["gen_text_hidden_dim"]),
            text_out_dim=int(raw["gen_text_out_dim"]),
            hidden_dim=int(raw["gen_hidden_dim"]),
            residual_output_mode="identity_softplus_residual",
            generator_noise_mode="gaussian",
            generator_current_input_mode="current_support_masked",
        )
        critic = Discriminator(
            channels=1,
            embedding_dim=1024,
            surface_height=8,
            surface_width=8,
            base_channels=int(raw["disc_base_channels"]),
            res_blocks=0,
            text_hidden_dim=int(raw["disc_text_hidden_dim"]),
            hidden_dim=int(raw["disc_hidden_dim"]),
            critic_normalization_mode=INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
        )
        counts = {
            "generator": sum(parameter.numel() for parameter in generator.parameters()),
            "discriminator": sum(
                parameter.numel() for parameter in critic.parameters()
            ),
        }
        counts["wgan"] = counts["generator"] + counts["discriminator"]
        for key in ("generator", "discriminator", "wgan"):
            if counts[key] != int(raw[f"expected_{key}_parameters"]):
                raise ValueError(
                    f"{name} {key} parameter count drift: {counts[key]} != "
                    f"{raw[f'expected_{key}_parameters']}"
                )
        result[name] = _model_contract(name, raw, grid)
    return result


def _load_registry(root: Path) -> dict[str, Any]:
    registry = _require_mapping(_read_json(root / "registry" / "jobs.json"), "registry")
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment root is not a grid08 WGAN sweep")
    return registry


def _write_registry_exports(root: Path) -> None:
    registry = _load_registry(root)
    jobs = list(registry["jobs"])
    _write_csv(root / "task_registry.csv", jobs, tuple(jobs[0]))
    output_rows = []
    for job in jobs:
        status_path = _job_status_path(root, job["job_id"])
        if not status_path.is_file():
            continue
        status = _read_json(status_path)
        output_rows.extend(status.get("artifacts") or [])
    if output_rows:
        dedup = {str(row["path"]): dict(row) for row in output_rows}
        _write_csv(
            root / "output_hashes.csv",
            list(dedup.values()),
            tuple(next(iter(dedup.values()))),
        )


def _write_code_manifest(root: Path) -> None:
    _write_hash_rows(root / "code_hashes.csv", _code_rows())


def _active_code_manifest(root: Path, registry: Mapping[str, Any]) -> Path:
    active_name = str(registry.get("active_code_hashes_filename", "code_hashes.csv"))
    if active_name not in {"code_hashes.csv", "code_hashes_current.csv"}:
        raise ValueError(f"Unsupported active code manifest: {active_name}")
    return root / active_name


def _validate_root_lineage(root: Path) -> dict[str, Any]:
    registry = _load_registry(root)
    for name in ("source_hashes.csv", "config_hashes.csv"):
        path = root / name
        anchor = f"{name.removesuffix('.csv')}_sha256"
        if registry.get(anchor) and _sha256_file(path) != registry[anchor]:
            raise ValueError(f"Manifest anchor mismatch: {name}")
        _verify_hash_manifest(path)
    prepared_code = root / "code_hashes.csv"
    if _sha256_file(prepared_code) != registry.get("code_hashes_sha256"):
        raise ValueError("Prepared code-manifest anchor mismatch")
    active_code = _active_code_manifest(root, registry)
    active_anchor = (
        "code_hashes_sha256"
        if active_code.name == "code_hashes.csv"
        else "code_hashes_current_sha256"
    )
    if _sha256_file(active_code) != registry.get(active_anchor):
        raise ValueError("Active code-manifest anchor mismatch")
    _verify_hash_manifest(active_code)
    resolved_path = root / "resolved_config.yaml"
    resolved = _require_mapping(
        _load_yaml(resolved_path, "resolved").get(ROOT_KEY), ROOT_KEY
    )
    if _payload_sha256(resolved) != registry["resolved_config_sha256"]:
        raise ValueError("Resolved config hash drift")
    jobs = list(registry["jobs"])
    ids = [job["job_id"] for job in jobs]
    outputs = [job["output_root"] for job in jobs]
    if len(ids) != len(set(ids)) or len(outputs) != len(set(outputs)):
        raise ValueError("Duplicate job ID or output path")
    for job in jobs:
        if _job_spec_sha(job) != job["job_spec_sha256"]:
            raise ValueError(f"Job spec hash mismatch: {job['job_id']}")
        if _sha256_file(Path(job["training_config_path"])) != job["config_sha256"]:
            raise ValueError(f"Job config drift: {job['job_id']}")
        if _sha256_file(Path(job["dataset_path"])) != job["dataset_sha256"]:
            raise ValueError(f"Dataset drift: {job['job_id']}")
    return resolved


def _validate_worker_lineage(root: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed without making 48 workers rehash every multi-GB source."""

    registry = _load_registry(root)
    for name in ("source_hashes.csv", "config_hashes.csv"):
        anchor = f"{name.removesuffix('.csv')}_sha256"
        path = root / name
        if not path.is_file() or _sha256_file(path) != registry.get(anchor):
            raise ValueError(f"Worker manifest anchor mismatch: {name}")
    prepared_code = root / "code_hashes.csv"
    if _sha256_file(prepared_code) != registry.get("code_hashes_sha256"):
        raise ValueError("Worker prepared code-manifest anchor mismatch")
    active_code = _active_code_manifest(root, registry)
    active_anchor = (
        "code_hashes_sha256"
        if active_code.name == "code_hashes.csv"
        else "code_hashes_current_sha256"
    )
    if _sha256_file(active_code) != registry.get(active_anchor):
        raise ValueError("Worker active code-manifest anchor mismatch")
    _verify_hash_manifest(active_code)
    if _sha256_file(Path(job["training_config_path"])) != job["config_sha256"]:
        raise ValueError(f"Worker config drift: {job['job_id']}")
    if _sha256_file(Path(job["dataset_path"])) != job["dataset_sha256"]:
        raise ValueError(f"Worker dataset drift: {job['job_id']}")
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY), ROOT_KEY
    )
    if _payload_sha256(resolved) != registry["resolved_config_sha256"]:
        raise ValueError("Worker resolved-config drift")
    return resolved


def _anchor_manifests_in_registry(root: Path) -> None:
    registry = _load_registry(root)
    for name in ("source_hashes.csv", "code_hashes.csv", "config_hashes.csv"):
        registry[f"{name.removesuffix('.csv')}_sha256"] = _sha256_file(root / name)
    _write_json(root / "registry" / "jobs.json", registry)


def prepare_grid08_experiment(
    config_path: str | Path, output_dir: str | Path, *, reuse: bool = False
) -> Path:
    root = _resolve_repo_path(output_dir)
    resolved = resolve_config(config_path)
    benchmark_path = root.with_name(root.name + "_benchmark") / "benchmark_result.json"
    if not benchmark_path.is_file():
        raise FileNotFoundError(
            "Formal prepare requires the independent 48-worker benchmark first: "
            f"{benchmark_path}"
        )
    benchmark = _read_json(benchmark_path)
    slots = int(benchmark.get("selected_slots_per_gpu", -1))
    if slots not in {12, 24}:
        raise ValueError("Benchmark did not freeze a valid concurrency choice")
    resolved["runtime"]["slots_per_gpu"] = slots
    resolved["runtime"]["benchmark_result_path"] = str(benchmark_path)
    resolved["runtime"]["benchmark_result_sha256"] = _sha256_file(benchmark_path)
    resolved_sha = _payload_sha256(resolved)
    hash_path = root / "registry" / "resolved_config.sha256"
    if root.exists() and any(root.iterdir()):
        if not reuse:
            raise FileExistsError(f"Experiment root exists; use --reuse: {root}")
        if not hash_path.is_file() or hash_path.read_text().strip() != resolved_sha:
            raise ValueError("Existing grid08 root has a different resolved config")
        _validate_root_lineage(root)
        return root
    dataset_root = Path(resolved["datasets"]["root"])
    if not (dataset_root / "dataset_output_sha256.txt").is_file():
        raise FileNotFoundError("The immutable grid08 dataset has not been built")
    grid = _grid_contract(resolved)
    contracts = _instantiate_profile_contracts(resolved, grid)
    root.mkdir(parents=True)
    for relative in (
        "registry/jobs",
        "configs/jobs",
        "logs",
        "resources",
        "runs",
        "analysis",
        "report",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _write_yaml(root / "resolved_config.yaml", {ROOT_KEY: resolved})
    _atomic_write_text(hash_path, resolved_sha + "\n")
    _write_json(root / "grid08_dataset_manifest.json", grid)
    profile_rows = []
    for name, contract in contracts.items():
        profile_rows.append(
            {
                "capacity_profile": name,
                "architecture_profile_sha256": contract["architecture_profile_sha256"],
                "model_contract_sha256": contract["model_contract_sha256"],
                "surface_grid_sha256": contract["surface_grid_sha256"],
                "expected_generator_parameters": contract[
                    "expected_generator_parameters"
                ],
                "expected_discriminator_parameters": contract[
                    "expected_discriminator_parameters"
                ],
                "expected_wgan_parameters": contract["expected_wgan_parameters"],
            }
        )
    _write_csv(
        root / "grid08_profile_manifest.csv", profile_rows, tuple(profile_rows[0])
    )
    _build_split_manifest(resolved, root)
    window_rows = _materialize_pre_q3_workbooks(resolved, root)
    source_rows = _source_rows(resolved) + window_rows
    _write_hash_rows(root / "source_hashes.csv", source_rows)
    source_by_path = {row["path"]: row["sha256"] for row in source_rows}
    assigned = _balanced_assignments(
        primary_specs(),
        gpu_ids=resolved["runtime"]["gpu_ids"],
        slots_per_gpu=slots,
    )
    jobs = []
    config_rows = []
    numa = {
        int(key): int(value)
        for key, value in resolved["runtime"]["gpu_numa_nodes"].items()
    }
    for spec in assigned:
        profile = str(spec["capacity_profile"])
        lr_profile = str(spec["lr_profile"])
        tolerance = int(spec["tolerance_minutes"])
        seed = int(spec["seed"])
        mode = str(spec["text_ablation_mode"])
        job_id = _job_id(profile, lr_profile, seed, mode, tolerance)
        payload = _training_payload(
            resolved, root, spec=spec, model_contract=contracts[profile]
        )
        config_path_for_job = root / "configs" / "jobs" / f"{job_id}.yaml"
        _write_yaml(config_path_for_job, payload)
        dataset_path = str(Path(payload["data_path"]).resolve())
        job = {
            "job_id": job_id,
            "experiment_kind": EXPERIMENT_KIND,
            "experiment_stage": PRIMARY_STAGE,
            "wave": int(spec["wave"]),
            "model_family": "wgan",
            "trainer_command": "vol-xlsx",
            "capacity_profile": profile,
            "capacity_profile_sha256": contracts[profile][
                "architecture_profile_sha256"
            ],
            "architecture_profile_sha256": contracts[profile][
                "architecture_profile_sha256"
            ],
            "model_contract_sha256": contracts[profile]["model_contract_sha256"],
            "expected_generator_parameters": contracts[profile][
                "expected_generator_parameters"
            ],
            "expected_discriminator_parameters": contracts[profile][
                "expected_discriminator_parameters"
            ],
            "expected_wgan_parameters": contracts[profile]["expected_wgan_parameters"],
            "lr_profile": lr_profile,
            "lr_profile_sha256": _lr_sha(lr_profile),
            "initial_learning_rate": LRS[lr_profile],
            "scheduler_min_lr": LRS[lr_profile] / 10.0,
            "seed": seed,
            "text_ablation_mode": mode,
            "text_information_path": text_information_path(mode),
            "support_mask_mode": "raw_joint",
            "generator_current_input_mode": "current_support_masked",
            "generator_noise_mode": "gaussian",
            "critic_normalization_mode": INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
            "surface_grid_profile": GRID_PROFILE,
            "surface_grid_sha256": grid["grid_sha256"],
            "surface_shape": [8, 8],
            "tolerance_minutes": tolerance,
            "gpu_id": int(spec["gpu_id"]),
            "gpu_slot": int(spec["gpu_slot"]),
            "numa_node": numa[int(spec["gpu_id"])],
            "training_config_path": str(config_path_for_job.resolve()),
            "config_sha256": _sha256_file(config_path_for_job),
            "dataset_path": dataset_path,
            "dataset_sha256": source_by_path[dataset_path],
            "output_root": str(payload["output_root"]),
        }
        job["job_spec_sha256"] = _job_spec_sha(job)
        jobs.append(job)
        config_rows.append(
            {
                "config_role": job_id,
                "path": str(config_path_for_job.resolve()),
                "sha256": job["config_sha256"],
            }
        )
        _write_json(
            _job_status_path(root, job_id),
            {
                "job_id": job_id,
                "status": "prepared",
                "attempt": 0,
                "config_sha256": job["config_sha256"],
                "job_spec_sha256": job["job_spec_sha256"],
                "updated_at_utc": _utc_now(),
            },
        )
    if len(jobs) != 180:
        raise AssertionError(f"Primary matrix must have 180 jobs, got {len(jobs)}")
    _write_json(
        root / "registry" / "jobs.json",
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "experiment_root": str(root),
            "resolved_config_sha256": resolved_sha,
            "benchmark_result_path": str(benchmark_path),
            "benchmark_result_sha256": _sha256_file(benchmark_path),
            "q3_unlocked": False,
            "jobs": jobs,
        },
    )
    _write_json(
        root / "registry" / "experiment_status.json",
        {
            "status": "prepared",
            "q3_predictions_generated": False,
            "q3_evaluated": False,
            "q3_used_for_selection": False,
            "q4_loader_created": False,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
            "updated_at_utc": _utc_now(),
        },
    )
    _write_code_manifest(root)
    config_rows.insert(
        0,
        {
            "config_role": "resolved",
            "path": str((root / "resolved_config.yaml").resolve()),
            "sha256": _sha256_file(root / "resolved_config.yaml"),
        },
    )
    _write_hash_rows(root / "config_hashes.csv", config_rows)
    _anchor_manifests_in_registry(root)
    _write_registry_exports(root)
    _validate_root_lineage(root)
    return root


def _validated_lr_trace(job: Mapping[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    rows = _read_json(run_dir / "metrics" / "training_metrics.json")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(
            f"WGAN trace requires epoch 0 and learned epochs: {job['job_id']}"
        )
    initial = float(job["initial_learning_rate"])
    floor = float(job["scheduler_min_lr"])
    previous_g = initial
    previous_d = initial
    trace = []
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, f"metric {job['job_id']}[{index}]")
        epoch = int(row["epoch"])
        g_lr = float(row["g_lr"])
        d_lr = float(row["d_lr"])
        if not (math.isfinite(g_lr) and math.isfinite(d_lr)):
            raise ValueError(f"Non-finite LR: {job['job_id']}")
        if not (floor <= g_lr <= previous_g and floor <= d_lr <= previous_d):
            raise ValueError(
                f"LR trace violates monotone/floor contract: {job['job_id']}"
            )
        trace.append({"epoch": epoch, "g_lr": g_lr, "d_lr": d_lr})
        previous_g, previous_d = g_lr, d_lr
    if trace[0] != {"epoch": 0, "g_lr": initial, "d_lr": initial}:
        raise ValueError(f"LR trace has the wrong epoch-0 origin: {job['job_id']}")
    return trace


def _validate_checkpoint_contract(job: Mapping[str, Any], run_dir: Path) -> None:
    import torch

    best = _require_mapping(
        _read_json(run_dir / "metrics" / "best_learned_checkpoint.json"),
        f"best learned {job['job_id']}",
    )
    if int(best.get("best_epoch", best.get("epoch", 0))) < 1:
        raise ValueError(f"best_learned epoch must be >=1: {job['job_id']}")
    expected = {
        "capacity_profile": job["capacity_profile"],
        "architecture_profile_sha256": job["architecture_profile_sha256"],
        "model_contract_sha256": job["model_contract_sha256"],
        "surface_grid_profile": job["surface_grid_profile"],
        "surface_grid_sha256": job["surface_grid_sha256"],
        "critic_normalization_mode": job["critic_normalization_mode"],
        "lr_profile": job["lr_profile"],
        "lr_profile_sha256": job["lr_profile_sha256"],
        "initial_learning_rate": job["initial_learning_rate"],
        "scheduler_min_lr": job["scheduler_min_lr"],
    }
    for key, value in expected.items():
        if best.get(key) != value:
            raise ValueError(
                f"Best-learned checkpoint lineage mismatch {job['job_id']}: {key}"
            )
    generator_path = run_dir / "checkpoints" / "generator_best_learned.pt"
    discriminator_path = run_dir / "checkpoints" / "discriminator_best_learned.pt"
    for path, count_key in (
        (generator_path, "expected_generator_parameters"),
        (discriminator_path, "expected_discriminator_parameters"),
    ):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_expected = {
            key: value
            for key, value in expected.items()
            if key
            not in {
                "lr_profile",
                "lr_profile_sha256",
                "initial_learning_rate",
                "scheduler_min_lr",
            }
        }
        for key, value in checkpoint_expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Checkpoint lineage mismatch {path.name}: {key}")
        state = checkpoint.get("state_dict", checkpoint)
        count = sum(int(value.numel()) for value in state.values())
        if count != int(job[count_key]):
            raise ValueError(
                f"Checkpoint parameter count mismatch {path.name}: {count} != {job[count_key]}"
            )


def _completed_valid(job: Mapping[str, Any], status: Mapping[str, Any]) -> bool:
    if (
        status.get("status") != "completed"
        or status.get("config_sha256") != job["config_sha256"]
    ):
        return False
    artifacts = status.get("artifacts") or []
    return bool(artifacts) and all(
        Path(row["path"]).is_file() and _sha256_file(Path(row["path"])) == row["sha256"]
        for row in artifacts
    )


def run_grid08_worker(
    experiment_root: str | Path,
    job_id: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    root = _resolve_repo_path(experiment_root)
    matches = [
        dict(job) for job in _load_registry(root)["jobs"] if job["job_id"] == job_id
    ]
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicate grid08 job: {job_id}")
    job = matches[0]
    resolved = _validate_worker_lineage(root, job)
    status_path = _job_status_path(root, job_id)
    previous = _read_json(status_path)
    if previous.get("status") == "completed" and not dry_run:
        if resume and _completed_valid(job, previous):
            return Path(previous["run_dir"])
        raise RuntimeError(f"Job already completed or invalid: {job_id}")
    if previous.get("status") == "running" and training._pid_is_live(
        previous.get("pid")
    ):
        raise RuntimeError(f"Duplicate live worker: {job_id}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if visible and visible != str(job["gpu_id"]):
        raise RuntimeError(
            f"Worker GPU mismatch: visible={visible}, assigned={job['gpu_id']}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
    threads = str(int(resolved["runtime"]["cpu_threads_per_job"]))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = threads
    attempt = int(previous.get("attempt", 0)) + 1
    common = {
        "job_id": job_id,
        "attempt": attempt,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "gpu_id": job["gpu_id"],
        "gpu_slot": job["gpu_slot"],
        "config_sha256": job["config_sha256"],
        "job_spec_sha256": job["job_spec_sha256"],
        "experiment_stage": job["experiment_stage"],
        "capacity_profile": job["capacity_profile"],
        "lr_profile": job["lr_profile"],
        "seed": job["seed"],
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "dry_run": bool(dry_run),
    }
    _write_json(status_path, {**common, "status": "running"})
    try:
        run_dir, artifacts = training._execute_training_job(job, dry_run=dry_run)
        trace: list[Any] = [float(job["initial_learning_rate"])]
        if not dry_run:
            trace = _validated_lr_trace(job, run_dir)
            _validate_checkpoint_contract(job, run_dir)
        _write_json(
            status_path,
            {
                **common,
                "status": "dry_run_passed" if dry_run else "completed",
                "run_dir": str(run_dir),
                "artifacts": artifacts,
                "lr_trace": trace,
                "exit_code": 0,
                "completed_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
            },
        )
        return run_dir
    except BaseException as exc:
        _write_json(
            status_path,
            {
                **common,
                "status": "failed",
                "exit_code": 1,
                "error": f"{type(exc).__name__}: {exc}",
                "completed_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
            },
        )
        raise


def build_worker_command(
    root: Path, job: Mapping[str, Any], *, dry_run: bool, resume: bool
) -> list[str]:
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved")[ROOT_KEY], ROOT_KEY
    )
    command = [
        str(resolved["runtime"]["python_executable"]),
        "-m",
        "scripts.rq3.news_first_vol_wgan_grid08_sweep",
        "worker",
        "--config",
        str(resolved["source_config_path"]),
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


def _jobs_for_wave(
    root: Path, wave: int, *, dry_run: bool, resume: bool
) -> list[dict[str, Any]]:
    selected = []
    for raw in _load_registry(root)["jobs"]:
        if int(raw["wave"]) != int(wave):
            continue
        job = dict(raw)
        status = _read_json(_job_status_path(root, job["job_id"]))
        if (
            not dry_run
            and status.get("status") == "completed"
            and _completed_valid(job, status)
        ):
            if resume:
                continue
            raise RuntimeError(
                f"Completed job requires --resume to skip: {job['job_id']}"
            )
        if status.get("status") == "running" and training._pid_is_live(
            status.get("pid")
        ):
            raise RuntimeError(f"Live job already exists: {job['job_id']}")
        selected.append(job)
    return selected


def _host_ram_fraction() -> float:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
        for line in handle:
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
    total = float(values["MemTotal"])
    available = float(values.get("MemAvailable", values.get("MemFree", 0)))
    return 1.0 - available / total


def _run_wave(
    root: Path,
    wave: int,
    jobs: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
    resume: bool,
) -> float:
    if not jobs:
        return _host_ram_fraction()
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved")[ROOT_KEY], ROOT_KEY
    )
    monitor = training._ResourceMonitor(
        root / "resource_usage.csv",
        executable=str(resolved["runtime"]["nvidia_smi_executable"]),
        interval_seconds=float(resolved["runtime"]["resource_sample_interval_seconds"]),
        wave=wave,
    )
    processes: list[subprocess.Popen[Any]] = []
    logs = []
    monitor.start()
    peak_host_ram = _host_ram_fraction()
    try:
        for job in jobs:
            log_path = root / "logs" / f"{job['job_id']}.attempt.log"
            handle = log_path.open("a", encoding="utf-8")
            logs.append(handle)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
            env["NEWS_FIRST_JOB_LOG_PATH"] = str(log_path)
            threads = str(int(resolved["runtime"]["cpu_threads_per_job"]))
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                env[key] = threads
            processes.append(
                subprocess.Popen(
                    build_worker_command(root, job, dry_run=dry_run, resume=resume),
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        while any(process.poll() is None for process in processes):
            peak_host_ram = max(peak_host_ram, _host_ram_fraction())
            time.sleep(1.0)
        failures = []
        for job, process in zip(jobs, processes):
            code = int(process.returncode or 0)
            if code:
                failures.append(f"{job['job_id']}={code}")
        if failures:
            raise RuntimeError(f"Grid08 wave {wave} failed: {failures}")
    except BaseException:
        training._terminate_processes(processes)
        raise
    finally:
        monitor.stop()
        for handle in logs:
            handle.close()
    for job in jobs:
        status = _read_json(_job_status_path(root, job["job_id"]))
        expected = "dry_run_passed" if dry_run else "completed"
        if status.get("status") != expected:
            raise RuntimeError(f"Wave completion state mismatch: {job['job_id']}")
    return peak_host_ram


def _all_primary_completed(root: Path) -> bool:
    jobs = [
        job
        for job in _load_registry(root)["jobs"]
        if job["experiment_stage"] == PRIMARY_STAGE
    ]
    return len(jobs) == 180 and all(
        _completed_valid(job, _read_json(_job_status_path(root, job["job_id"])))
        for job in jobs
    )


def _validate_frozen_selection(
    path: Path, returned_selection: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], str]:
    selection = _read_json(path)
    saved_payload_sha = str(selection.get("selection_sha256", ""))
    payload = {
        key: value for key, value in selection.items() if key != "selection_sha256"
    }
    if not saved_payload_sha or _payload_sha256(payload) != saved_payload_sha:
        raise ValueError("Selection self-hash mismatch")
    if returned_selection is not None:
        returned_sha = str(returned_selection.get("selection_sha256", ""))
        if returned_sha != saved_payload_sha:
            raise ValueError(
                "Returned selection payload does not match frozen selection"
            )
    return selection, _sha256_file(path)


def append_diagnostic_jobs(root: Path, winner: Mapping[str, Any]) -> list[str]:
    if not _all_primary_completed(root):
        raise RuntimeError(
            "Cannot append diagnostics before all 180 primary jobs complete"
        )
    selection_path = root / "analysis" / "grid08_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Frozen Q2 selection is missing")
    frozen_selection, selection_sha = _validate_frozen_selection(selection_path, winner)
    resolved = _validate_root_lineage(root)
    registry = _load_registry(root)
    if any(job["experiment_stage"] == DIAGNOSTIC_STAGE for job in registry["jobs"]):
        return [
            job["job_id"]
            for job in registry["jobs"]
            if job["experiment_stage"] == DIAGNOSTIC_STAGE
        ]
    profile = str(frozen_selection["one_se_candidate_capacity_profile"])
    lr_profile = str(frozen_selection["one_se_candidate_lr_profile"])
    grid = _read_json(root / "grid08_dataset_manifest.json")
    contracts = _instantiate_profile_contracts(resolved, grid)
    start_wave = max(int(job["wave"]) for job in registry["jobs"]) + 1
    raw_specs = [
        {
            "capacity_profile": profile,
            "lr_profile": lr_profile,
            "seed": seed,
            "text_ablation_mode": "current_only",
            "tolerance_minutes": tolerance,
            "experiment_stage": DIAGNOSTIC_STAGE,
        }
        for seed in SEEDS
        for tolerance in TOLERANCES
    ]
    assigned = _balanced_assignments(
        raw_specs,
        gpu_ids=resolved["runtime"]["gpu_ids"],
        slots_per_gpu=int(resolved["runtime"]["slots_per_gpu"]),
    )
    source_by_path = {}
    with (root / "source_hashes.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_by_path[row["path"]] = row["sha256"]
    config_rows = []
    with (root / "config_hashes.csv").open("r", encoding="utf-8", newline="") as handle:
        config_rows = list(csv.DictReader(handle))
    new_ids = []
    for spec in assigned:
        spec["wave"] = start_wave
        profile = str(spec["capacity_profile"])
        tolerance = int(spec["tolerance_minutes"])
        seed = int(spec["seed"])
        job_id = _job_id(profile, lr_profile, seed, "current_only", tolerance)
        payload = _training_payload(
            resolved, root, spec=spec, model_contract=contracts[profile]
        )
        path = root / "configs" / "jobs" / f"{job_id}.yaml"
        _write_yaml(path, payload)
        dataset_path = str(Path(payload["data_path"]).resolve())
        job = {
            "job_id": job_id,
            "experiment_kind": EXPERIMENT_KIND,
            "experiment_stage": DIAGNOSTIC_STAGE,
            "parent_selection_sha256": selection_sha,
            "wave": start_wave,
            "model_family": "wgan",
            "trainer_command": "vol-xlsx",
            "capacity_profile": profile,
            "capacity_profile_sha256": contracts[profile][
                "architecture_profile_sha256"
            ],
            "architecture_profile_sha256": contracts[profile][
                "architecture_profile_sha256"
            ],
            "model_contract_sha256": contracts[profile]["model_contract_sha256"],
            "expected_generator_parameters": contracts[profile][
                "expected_generator_parameters"
            ],
            "expected_discriminator_parameters": contracts[profile][
                "expected_discriminator_parameters"
            ],
            "expected_wgan_parameters": contracts[profile]["expected_wgan_parameters"],
            "lr_profile": lr_profile,
            "lr_profile_sha256": _lr_sha(lr_profile),
            "initial_learning_rate": LRS[lr_profile],
            "scheduler_min_lr": LRS[lr_profile] / 10,
            "seed": seed,
            "text_ablation_mode": "current_only",
            "text_information_path": text_information_path("current_only"),
            "support_mask_mode": "raw_joint",
            "generator_current_input_mode": "current_support_masked",
            "generator_noise_mode": "gaussian",
            "critic_normalization_mode": INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
            "surface_grid_profile": GRID_PROFILE,
            "surface_grid_sha256": grid["grid_sha256"],
            "surface_shape": [8, 8],
            "tolerance_minutes": tolerance,
            "gpu_id": int(spec["gpu_id"]),
            "gpu_slot": int(spec["gpu_slot"]),
            "numa_node": int(
                resolved["runtime"]["gpu_numa_nodes"][int(spec["gpu_id"])]
            ),
            "training_config_path": str(path.resolve()),
            "config_sha256": _sha256_file(path),
            "dataset_path": dataset_path,
            "dataset_sha256": source_by_path[dataset_path],
            "output_root": str(payload["output_root"]),
        }
        job["job_spec_sha256"] = _job_spec_sha(job)
        registry["jobs"].append(job)
        new_ids.append(job_id)
        config_rows.append(
            {
                "config_role": job_id,
                "path": str(path.resolve()),
                "sha256": job["config_sha256"],
            }
        )
        _write_json(
            _job_status_path(root, job_id),
            {
                "job_id": job_id,
                "status": "prepared",
                "attempt": 0,
                "config_sha256": job["config_sha256"],
                "job_spec_sha256": job["job_spec_sha256"],
                "updated_at_utc": _utc_now(),
            },
        )
    registry["selection_sha256"] = selection_sha
    _write_json(root / "registry" / "jobs.json", registry)
    _write_hash_rows(root / "config_hashes.csv", config_rows)
    _anchor_manifests_in_registry(root)
    _write_registry_exports(root)
    return new_ids


def _freeze_q3_allowlist(root: Path) -> Path:
    selection = _read_json(root / "analysis" / "grid08_selection.json")
    wanted = {
        (
            selection["one_se_candidate_capacity_profile"],
            selection["one_se_candidate_lr_profile"],
            REAL_TEXT,
        ),
        (
            selection["point_leader_capacity_profile"],
            selection["point_leader_lr_profile"],
            REAL_TEXT,
        ),
        ("small", "lr_5e_07", REAL_TEXT),
        (
            selection["one_se_candidate_capacity_profile"],
            selection["one_se_candidate_lr_profile"],
            "current_only",
        ),
    }
    rows = []
    for job in _load_registry(root)["jobs"]:
        key = (job["capacity_profile"], job["lr_profile"], job["text_ablation_mode"])
        if key not in wanted:
            continue
        status = _read_json(_job_status_path(root, job["job_id"]))
        if not _completed_valid(job, status):
            raise ValueError(
                f"Allowlisted checkpoint job is incomplete: {job['job_id']}"
            )
        path = Path(status["run_dir"]) / "checkpoints" / "generator_best_learned.pt"
        rows.append(
            {
                "job_id": job["job_id"],
                "capacity_profile": job["capacity_profile"],
                "lr_profile": job["lr_profile"],
                "seed": job["seed"],
                "tolerance_minutes": job["tolerance_minutes"],
                "text_ablation_mode": job["text_ablation_mode"],
                "checkpoint_path": str(path),
                "checkpoint_sha256": _sha256_file(path),
                "selection_sha256": _sha256_file(
                    root / "analysis" / "grid08_selection.json"
                ),
            }
        )
    expected = len({(a, b, c) for a, b, c in wanted}) * 3 * 2
    if len(rows) != expected:
        raise ValueError(f"Q3 allowlist incomplete: {len(rows)} != {expected}")
    path = _write_csv(root / "q3_checkpoint_allowlist.csv", rows, tuple(rows[0]))
    registry = _load_registry(root)
    registry["q3_unlocked"] = True
    registry["q3_checkpoint_allowlist_sha256"] = _sha256_file(path)
    _write_json(root / "registry" / "jobs.json", registry)
    return path


def launch_grid08_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    q2_hook: Callable[[Path], Mapping[str, Any]] | None = None,
    final_hook: Callable[[Path], None] | None = None,
) -> Path:
    root = prepare_grid08_experiment(config_path, output_dir, reuse=True)
    _validate_root_lineage(root)
    status_path = root / "registry" / "experiment_status.json"
    status = _read_json(status_path)
    status.update(
        {
            "status": "running_dry_run" if dry_run else "running_primary",
            "updated_at_utc": _utc_now(),
        }
    )
    _write_json(status_path, status)
    primary_waves = sorted(
        {
            int(job["wave"])
            for job in _load_registry(root)["jobs"]
            if job["experiment_stage"] == PRIMARY_STAGE
        }
    )
    for wave in primary_waves:
        jobs = _jobs_for_wave(root, wave, dry_run=dry_run, resume=resume)
        _run_wave(root, wave, jobs, dry_run=dry_run, resume=resume)
    if dry_run:
        status.update({"status": "dry_run_passed", "updated_at_utc": _utc_now()})
        _write_json(status_path, status)
        _write_registry_exports(root)
        return root
    if not _all_primary_completed(root):
        raise RuntimeError("Primary matrix did not complete")
    selection_path = root / "analysis" / "grid08_selection.json"
    diagnostics_already_registered = any(
        job["experiment_stage"] == DIAGNOSTIC_STAGE
        for job in _load_registry(root)["jobs"]
    )
    if diagnostics_already_registered:
        selection, _ = _validate_frozen_selection(selection_path)
    else:
        if q2_hook is None:
            from scripts.rq3.news_first_vol_wgan_grid08_analysis import (
                run_grid08_q2_analysis,
            )

            q2_hook = run_grid08_q2_analysis
        selection = dict(q2_hook(root))
    append_diagnostic_jobs(root, selection)
    diagnostic_waves = sorted(
        {
            int(job["wave"])
            for job in _load_registry(root)["jobs"]
            if job["experiment_stage"] == DIAGNOSTIC_STAGE
        }
    )
    for wave in diagnostic_waves:
        jobs = _jobs_for_wave(root, wave, dry_run=False, resume=resume)
        _run_wave(root, wave, jobs, dry_run=False, resume=resume)
    _freeze_q3_allowlist(root)
    if final_hook is None:
        from scripts.rq3.news_first_vol_wgan_grid08_analysis import (
            run_grid08_q3_analysis,
        )
        from scripts.rq3.news_first_vol_wgan_grid08_report import render_grid08_report

        run_grid08_q3_analysis(root)
        render_grid08_report(root)
    else:
        final_hook(root)
    status = _read_json(status_path)
    status.update(
        {
            "status": "completed_q3_only",
            "q3_predictions_generated": True,
            "q3_evaluated": True,
            "q3_used_for_selection": False,
            "q4_loader_created": False,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
            "updated_at_utc": _utc_now(),
        }
    )
    _write_json(status_path, status)
    _write_registry_exports(root)
    # The terminal registry/status mutation must itself be covered by the
    # delivery manifest, so render/finalize once more after terminal state.
    from scripts.rq3.news_first_vol_wgan_grid08_report import render_grid08_report

    render_grid08_report(root)
    return root


def _benchmark_specs() -> list[dict[str, Any]]:
    # Four base cells per capacity, rotated across LR and seed, with both
    # tolerances.  Each capacity/LR/seed and both tolerances occur in the
    # single 48-worker wave; the legacy profile supplies the true worst case.
    specs = []
    for capacity_index, profile in enumerate(PROFILES):
        for offset in range(4):
            lr_profile = LR_PROFILES[(capacity_index + offset) % len(LR_PROFILES)]
            seed = SEEDS[(capacity_index + 2 * offset) % len(SEEDS)]
            for tolerance in TOLERANCES:
                specs.append(
                    {
                        "capacity_profile": profile,
                        "lr_profile": lr_profile,
                        "seed": seed,
                        "text_ablation_mode": REAL_TEXT,
                        "tolerance_minutes": tolerance,
                        "experiment_stage": "benchmark",
                    }
                )
    if len(specs) != 48:
        raise AssertionError("Benchmark must contain exactly 48 workers")
    return specs


def prepare_grid08_benchmark(
    config_path: str | Path, formal_output_dir: str | Path, *, reuse: bool = False
) -> Path:
    formal = _resolve_repo_path(formal_output_dir)
    root = formal.with_name(formal.name + "_benchmark")
    resolved = resolve_config(config_path)
    resolved["runtime"]["slots_per_gpu"] = 24
    resolved_sha = _payload_sha256(resolved)
    hash_path = root / "registry" / "resolved_config.sha256"
    if root.exists() and any(root.iterdir()):
        if not reuse:
            raise FileExistsError(
                f"Benchmark root exists; use --resume/--reuse: {root}"
            )
        if not hash_path.is_file() or hash_path.read_text().strip() != resolved_sha:
            raise ValueError("Benchmark root resolved config drift")
        _validate_root_lineage(root)
        return root
    grid = _grid_contract(resolved)
    contracts = _instantiate_profile_contracts(resolved, grid)
    root.mkdir(parents=True)
    for relative in ("registry/jobs", "configs/jobs", "logs", "resources", "runs"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _write_yaml(root / "resolved_config.yaml", {ROOT_KEY: resolved})
    _atomic_write_text(hash_path, resolved_sha + "\n")
    _write_json(root / "grid08_dataset_manifest.json", grid)
    _build_split_manifest(resolved, root)
    window_rows = _materialize_pre_q3_workbooks(resolved, root)
    source_rows = _source_rows(resolved) + window_rows
    _write_hash_rows(root / "source_hashes.csv", source_rows)
    source_by_path = {row["path"]: row["sha256"] for row in source_rows}
    assigned = _balanced_assignments(
        _benchmark_specs(), gpu_ids=resolved["runtime"]["gpu_ids"], slots_per_gpu=24
    )
    jobs = []
    config_rows = []
    numa = {
        int(key): int(value)
        for key, value in resolved["runtime"]["gpu_numa_nodes"].items()
    }
    for spec in assigned:
        profile = str(spec["capacity_profile"])
        lr_profile = str(spec["lr_profile"])
        seed = int(spec["seed"])
        tolerance = int(spec["tolerance_minutes"])
        job_id = "benchmark_" + _job_id(profile, lr_profile, seed, REAL_TEXT, tolerance)
        payload = _training_payload(
            resolved, root, spec=spec, model_contract=contracts[profile], benchmark=True
        )
        payload["output_root"] = str(root / "runs" / job_id)
        path = root / "configs" / "jobs" / f"{job_id}.yaml"
        _write_yaml(path, payload)
        dataset_path = str(Path(payload["data_path"]).resolve())
        job = {
            "job_id": job_id,
            "experiment_kind": EXPERIMENT_KIND,
            "experiment_stage": "benchmark",
            "wave": 1,
            "model_family": "wgan",
            "trainer_command": "vol-xlsx",
            "capacity_profile": profile,
            "capacity_profile_sha256": contracts[profile][
                "architecture_profile_sha256"
            ],
            "architecture_profile_sha256": contracts[profile][
                "architecture_profile_sha256"
            ],
            "model_contract_sha256": contracts[profile]["model_contract_sha256"],
            "expected_generator_parameters": contracts[profile][
                "expected_generator_parameters"
            ],
            "expected_discriminator_parameters": contracts[profile][
                "expected_discriminator_parameters"
            ],
            "expected_wgan_parameters": contracts[profile]["expected_wgan_parameters"],
            "lr_profile": lr_profile,
            "lr_profile_sha256": _lr_sha(lr_profile),
            "initial_learning_rate": LRS[lr_profile],
            "scheduler_min_lr": LRS[lr_profile] / 10,
            "seed": seed,
            "text_ablation_mode": REAL_TEXT,
            "text_information_path": text_information_path(REAL_TEXT),
            "support_mask_mode": "raw_joint",
            "generator_current_input_mode": "current_support_masked",
            "generator_noise_mode": "gaussian",
            "critic_normalization_mode": INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
            "surface_grid_profile": GRID_PROFILE,
            "surface_grid_sha256": grid["grid_sha256"],
            "surface_shape": [8, 8],
            "tolerance_minutes": tolerance,
            "gpu_id": int(spec["gpu_id"]),
            "gpu_slot": int(spec["gpu_slot"]),
            "numa_node": numa[int(spec["gpu_id"])],
            "training_config_path": str(path.resolve()),
            "config_sha256": _sha256_file(path),
            "dataset_path": dataset_path,
            "dataset_sha256": source_by_path[dataset_path],
            "output_root": str(payload["output_root"]),
        }
        job["job_spec_sha256"] = _job_spec_sha(job)
        jobs.append(job)
        config_rows.append(
            {
                "config_role": job_id,
                "path": str(path.resolve()),
                "sha256": job["config_sha256"],
            }
        )
        _write_json(
            _job_status_path(root, job_id),
            {
                "job_id": job_id,
                "status": "prepared",
                "attempt": 0,
                "config_sha256": job["config_sha256"],
                "job_spec_sha256": job["job_spec_sha256"],
                "updated_at_utc": _utc_now(),
            },
        )
    _write_json(
        root / "registry" / "jobs.json",
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "experiment_root": str(root),
            "resolved_config_sha256": resolved_sha,
            "benchmark": True,
            "q3_unlocked": False,
            "jobs": jobs,
        },
    )
    _write_json(
        root / "registry" / "experiment_status.json",
        {
            "status": "benchmark_prepared",
            "q3_evaluated": False,
            "q4_loader_created": False,
            "q4_evaluated": False,
            "updated_at_utc": _utc_now(),
        },
    )
    _write_code_manifest(root)
    config_rows.insert(
        0,
        {
            "config_role": "resolved",
            "path": str((root / "resolved_config.yaml").resolve()),
            "sha256": _sha256_file(root / "resolved_config.yaml"),
        },
    )
    _write_hash_rows(root / "config_hashes.csv", config_rows)
    _anchor_manifests_in_registry(root)
    _write_registry_exports(root)
    _validate_root_lineage(root)
    return root


def run_grid08_benchmark(
    config_path: str | Path, formal_output_dir: str | Path, *, resume: bool = False
) -> Path:
    root = prepare_grid08_benchmark(config_path, formal_output_dir, reuse=resume)
    jobs = _jobs_for_wave(root, 1, dry_run=False, resume=resume)
    peak_host_ram = _run_wave(root, 1, jobs, dry_run=False, resume=resume)
    peak_by_gpu: dict[int, float] = {0: 0.0, 1: 0.0}
    usage = root / "resource_usage.csv"
    if usage.is_file():
        with usage.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("sample_status") != "ok":
                    continue
                gpu = int(row["gpu_index"])
                peak_by_gpu[gpu] = max(
                    peak_by_gpu.get(gpu, 0.0), float(row["memory_used_mib"])
                )
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved")[ROOT_KEY], ROOT_KEY
    )
    memory_limit = (
        float(resolved["runtime"]["preflight_max_peak_gpu_memory_gib"]) * 1024
    )
    ram_limit = float(resolved["runtime"]["preflight_max_host_ram_fraction"])
    selected = (
        24
        if max(peak_by_gpu.values()) < memory_limit and peak_host_ram < ram_limit
        else 12
    )
    result = {
        "schema_version": 1,
        "status": "passed",
        "worker_count": 48,
        "workers_per_gpu": 24,
        "epochs": 1,
        "peak_gpu_memory_mib": {str(key): value for key, value in peak_by_gpu.items()},
        "peak_host_ram_fraction": peak_host_ram,
        "gpu_memory_limit_mib": memory_limit,
        "host_ram_fraction_limit": ram_limit,
        "selected_slots_per_gpu": selected,
        "formal_root_must_be_new": True,
        "q3_evaluated": False,
        "q4_loader_created": False,
        "q4_evaluated": False,
    }
    result["payload_sha256"] = _payload_sha256(result)
    _write_json(root / "benchmark_result.json", result)
    status = _read_json(root / "registry" / "experiment_status.json")
    status.update(
        {
            "status": "benchmark_completed",
            "selected_slots_per_gpu": selected,
            "updated_at_utc": _utc_now(),
        }
    )
    _write_json(root / "registry" / "experiment_status.json", status)
    _write_registry_exports(root)
    return root


def _activate_postprocess_code_ledger(root: Path) -> None:
    """Record the narrowly audited post-selection bug fix without rewriting history."""

    registry = _load_registry(root)
    if registry.get("active_code_hashes_filename") == "code_hashes_current.csv":
        active = root / "code_hashes_current.csv"
        if _sha256_file(active) != registry.get("code_hashes_current_sha256"):
            raise ValueError("Current postprocess code ledger anchor mismatch")
        _verify_hash_manifest(active)
        return
    prepared_path = root / "code_hashes.csv"
    if _sha256_file(prepared_path) != registry.get("code_hashes_sha256"):
        raise ValueError("Prepared code ledger anchor mismatch before recovery")
    with prepared_path.open("r", encoding="utf-8", newline="") as handle:
        prepared = {row["relative_path"]: row for row in csv.DictReader(handle)}
    current_rows = _code_rows()
    current = {row["relative_path"]: row for row in current_rows}
    if set(prepared) != set(current):
        raise ValueError("Postprocess recovery code universe changed")
    changed = sorted(
        relative
        for relative in prepared
        if prepared[relative]["sha256"] != current[relative]["sha256"]
        or int(prepared[relative]["size_bytes"]) != int(current[relative]["size_bytes"])
    )
    allowed = ["scripts/rq3/news_first_vol_wgan_grid08_sweep.py"]
    if changed != allowed:
        raise ValueError(
            f"Postprocess recovery refuses unexpected code drift: {changed}"
        )
    current_path = root / "code_hashes_current.csv"
    _write_hash_rows(current_path, current_rows)
    drift = {
        "schema_version": 1,
        "reason": "selection_payload_sha_was_compared_to_selection_file_sha",
        "training_reexecuted": False,
        "q3_was_unlocked_before_recovery": bool(registry.get("q3_unlocked")),
        "prepared_code_manifest_sha256": _sha256_file(prepared_path),
        "current_code_manifest_sha256": _sha256_file(current_path),
        "changed_paths": changed,
    }
    drift["payload_sha256"] = _payload_sha256(drift)
    drift_path = root / "postprocess_code_drift.json"
    _write_json(drift_path, drift)
    registry["active_code_hashes_filename"] = current_path.name
    registry["code_hashes_current_sha256"] = _sha256_file(current_path)
    registry["postprocess_code_drift_sha256"] = _sha256_file(drift_path)
    registry["code_changed_since_prepared_snapshot"] = True
    _write_json(root / "registry" / "jobs.json", registry)


def postprocess_grid08_experiment(root: str | Path) -> Path:
    experiment_root = _resolve_repo_path(root)
    _activate_postprocess_code_ledger(experiment_root)
    _validate_root_lineage(experiment_root)
    registry = _load_registry(experiment_root)
    if not registry.get("q3_unlocked"):
        if not _all_primary_completed(experiment_root):
            raise RuntimeError("Cannot postprocess an incomplete primary matrix")
        from scripts.rq3.news_first_vol_wgan_grid08_analysis import (
            run_grid08_q2_analysis,
        )

        selection_path = experiment_root / "analysis" / "grid08_selection.json"
        if selection_path.is_file():
            selection, _ = _validate_frozen_selection(selection_path)
        else:
            selection = run_grid08_q2_analysis(experiment_root)
        append_diagnostic_jobs(experiment_root, selection)
        status_path = experiment_root / "registry" / "experiment_status.json"
        status = _read_json(status_path)
        status.update(
            {
                "status": "diagnostics_prepared",
                "q3_evaluated": False,
                "q3_predictions_generated": False,
                "q3_used_for_selection": False,
                "q4_loader_created": False,
                "q4_predictions_generated": False,
                "q4_evaluated": False,
                "updated_at_utc": _utc_now(),
            }
        )
        _write_json(status_path, status)
        _write_registry_exports(experiment_root)
        return experiment_root
    from scripts.rq3.news_first_vol_wgan_grid08_analysis import run_grid08_q3_analysis
    from scripts.rq3.news_first_vol_wgan_grid08_report import render_grid08_report

    run_grid08_q3_analysis(experiment_root)
    render_grid08_report(experiment_root)
    _write_registry_exports(experiment_root)
    return experiment_root


def run_news_first_vol_wgan_grid08_sweep(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    action: str = "launch",
    job_id: str = "",
    resume: bool = False,
    reuse: bool = False,
    worker_dry_run: bool = False,
) -> Path:
    normalized = str(action).strip().lower()
    if normalized == "prepare":
        return prepare_grid08_experiment(config_path, output_dir, reuse=reuse)
    if normalized == "benchmark":
        return run_grid08_benchmark(config_path, output_dir, resume=resume or reuse)
    if normalized == "worker":
        if not job_id:
            raise ValueError("worker requires --job-id")
        return run_grid08_worker(
            output_dir, job_id, dry_run=worker_dry_run, resume=resume
        )
    if normalized == "dry-run":
        return launch_grid08_experiment(
            config_path, output_dir, dry_run=True, resume=resume
        )
    if normalized == "launch":
        return launch_grid08_experiment(
            config_path, output_dir, dry_run=False, resume=resume
        )
    if normalized == "postprocess":
        return postprocess_grid08_experiment(output_dir)
    raise ValueError(f"Unknown grid08 action: {action}")


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "benchmark", "dry-run", "worker", "launch", "postprocess"),
    )
    parser.add_argument(
        "--config", default="configs/rq3/news_first_vol_wgan_grid08_capacity_lr.yaml"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--worker-dry-run", action="store_true")
    args = parser.parse_args(argv)
    output = run_news_first_vol_wgan_grid08_sweep(
        args.config,
        args.output_dir,
        action=args.action,
        job_id=args.job_id,
        resume=args.resume,
        reuse=args.reuse,
        worker_dry_run=args.worker_dry_run,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
