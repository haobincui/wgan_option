"""Immutable, conditional News-first Vol label-reliability experiment.

Actions are deliberately explicit:

``prepare`` freezes inputs/config/code and the 48-job stage-one schedule;
``bootstrap`` runs the audit-only raw-trade bootstrap and materializes training
profiles; ``dry-run`` exercises model/data/resource contracts without optimizer
steps; ``worker`` executes one registered unit; ``launch`` runs the conditional
three-stage experiment; and ``postprocess`` performs the allowlisted Q3 analysis.

Bootstrap surfaces are never provided to a trainer.  The only training input
derived from the audit is a six-column pair profile with hash-checked lineage.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from scripts.rq3 import news_first_vol_training as training


ROOT_KEY = training.ROOT_KEY
REPO_ROOT = training.REPO_ROOT
EXPERIMENT_KIND = "label_reliability_sweep"
SCHEMA_VERSION = 1
DEFAULT_CONFIG = "configs/rq3/news_first_vol_label_reliability.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq3_news_first_vol_label_reliability_q097_103_ttm07_38_v1"
)
FROZEN_SEEDS = (42, 202, 404)
FROZEN_TOLERANCES = (5, 30)
FROZEN_PROFILE = "small"
FROZEN_LR = 5.0e-7
FROZEN_LR_FLOOR = 5.0e-8
FROZEN_LR_PROFILE = "lr_5e_07"
FROZEN_TEXT_MODE = "real_text"
FROZEN_SUPPORT_MODE = "raw_joint"
FROZEN_RESIDUAL_MODE = "identity_softplus_residual"
FROZEN_CURRENT_INPUT_MODE = "current_support_masked"
FROZEN_NOISE_MODE = "gaussian"
FROZEN_NOISE_DIM = 32
Q3_START_UTC = "2023-07-01T00:00:00Z"
Q3_END_UTC = "2023-10-01T00:00:00Z"
Q4_START_UTC = Q3_END_UTC
FOLDS: tuple[dict[str, str], ...] = (
    {
        "fold_id": "F1",
        "train_end_utc": "2022-07-01T00:00:00Z",
        "validation_start_utc": "2022-07-01T00:00:00Z",
        "validation_end_utc": "2022-10-01T00:00:00Z",
    },
    {
        "fold_id": "F2",
        "train_end_utc": "2022-10-01T00:00:00Z",
        "validation_start_utc": "2022-10-01T00:00:00Z",
        "validation_end_utc": "2023-01-01T00:00:00Z",
    },
    {
        "fold_id": "F3",
        "train_end_utc": "2023-01-01T00:00:00Z",
        "validation_start_utc": "2023-01-01T00:00:00Z",
        "validation_end_utc": "2023-04-01T00:00:00Z",
    },
    {
        "fold_id": "F4",
        "train_end_utc": "2023-04-01T00:00:00Z",
        "validation_start_utc": "2023-04-01T00:00:00Z",
        "validation_end_utc": "2023-07-01T00:00:00Z",
    },
)
ARMS: tuple[dict[str, Any], ...] = (
    {
        "arm_id": "A",
        "mode": "none",
        "filter_min_joint_cells": None,
        "soft_weight": False,
    },
    {
        "arm_id": "B",
        "mode": "support_filter",
        "filter_min_joint_cells": 16,
        "soft_weight": False,
    },
    {
        "arm_id": "C",
        "mode": "soft_weight",
        "filter_min_joint_cells": None,
        "soft_weight": True,
    },
    {
        "arm_id": "D",
        "mode": "support_filter_soft_weight",
        "filter_min_joint_cells": 16,
        "soft_weight": True,
    },
)
STAGE_1 = "stage1_regression_05m"
STAGE_2 = "stage2_regression_30m"
STAGE_3 = "stage3_wgan_05m"
STAGES = (STAGE_1, STAGE_2, STAGE_3)
EXPECTED_STAGE_JOBS = {STAGE_1: 48, STAGE_2: 24, STAGE_3: 24}
EXPECTED_MAX_JOBS = 96
PROFILE_COLUMNS = (
    "tolerance_minutes",
    "fold_id",
    "pair_id",
    "included",
    "normalized_label_weight",
    "reliability_score",
)
PROFILE_FILE_COLUMNS = PROFILE_COLUMNS + ("train_pair_universe_sha256",)
PROFILE_MANIFEST_COLUMNS = (
    "arm_id",
    "label_reliability_mode",
    "tolerance_minutes",
    "fold_id",
    "pair_count",
    "session_count",
    "included_pair_count",
    "included_session_count",
    "pair_retention_fraction",
    "session_retention_fraction",
    "tau_u_q75",
    "u_estimable_pair_fraction",
    "u_estimable_session_fraction",
    "global_estimability_gate_valid",
    "retention_gate_valid",
    "profile_path",
    "manifest_sha256",
    "profile_sha256",
    "train_pair_universe_sha256",
)

_read_json = training._read_json
_write_json = training._write_json
_write_yaml = training._write_yaml
_write_csv = training._write_csv
_sha256_file = training._sha256_file
_payload_sha256 = training._payload_sha256
_utc_now = training._utc_now
_require_mapping = training._require_mapping
_resolve_repo_path = training._resolve_repo_path
_job_status_path = training._job_status_path


def _capacity_profile_sha256() -> str:
    return training._capacity_profile_sha256(
        FROZEN_PROFILE, training.FROZEN_CAPACITY_PROFILES[FROZEN_PROFILE]
    )


def _lr_profile_sha256() -> str:
    return _payload_sha256(
        {
            "schema_version": 1,
            "lr_profile": FROZEN_LR_PROFILE,
            "initial_learning_rate": FROZEN_LR,
            "scheduler_min_lr": FROZEN_LR_FLOOR,
        }
    )


class LabelReliabilityExperimentError(ValueError):
    """Raised when a persisted experiment contract fails closed."""


def _load_yaml_mapping(path: str | Path, label: str) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return _require_mapping(value, label)


def _exact(value: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        observed = float(value)
        if not math.isfinite(observed) or observed != expected:
            raise LabelReliabilityExperimentError(
                f"{label} is frozen to {expected}; observed={observed}"
            )
    elif value != expected:
        raise LabelReliabilityExperimentError(
            f"{label} is frozen to {expected!r}; observed={value!r}"
        )


def _experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(
        config.get("label_reliability_experiment"),
        "label_reliability_experiment",
    )
    if not bool(value.get("enabled", False)):
        raise LabelReliabilityExperimentError(
            "label_reliability_experiment.enabled must be true"
        )
    return value


def _validate_frozen_config(config: Mapping[str, Any]) -> None:
    datasets = _require_mapping(config.get("datasets"), "datasets")
    experiment = _experiment_config(config)
    runtime = _require_mapping(config.get("runtime"), "runtime")
    models = _require_mapping(config.get("models"), "models")
    _exact(
        tuple(int(value) for value in datasets.get("tolerances_minutes", ())),
        (5, 10, 15, 30),
        "datasets.tolerances_minutes",
    )
    _exact(
        int(datasets.get("common_evaluation_tolerance_minutes", -1)),
        5,
        "common evaluation tolerance",
    )
    _exact(str(datasets.get("sheet_name", "")), "gan_input_ready", "sheet_name")
    _exact(str(datasets.get("text_embedding_mode", "")), "lp", "text embedding")
    _exact(str(datasets.get("support_mask_mode", "")), FROZEN_SUPPORT_MODE, "mask")
    _exact(
        tuple(str(value) for value in datasets.get("text_ablation_modes", ())),
        (FROZEN_TEXT_MODE,),
        "text modes",
    )
    _exact(tuple(int(value) for value in experiment.get("seeds", ())), FROZEN_SEEDS, "seeds")
    _exact(
        tuple(int(value) for value in experiment.get("tolerances_minutes", ())),
        FROZEN_TOLERANCES,
        "experiment tolerances",
    )
    _exact(str(experiment.get("capacity_profile", "")), FROZEN_PROFILE, "capacity")
    _exact(float(experiment.get("initial_learning_rate", -1)), FROZEN_LR, "LR")
    _exact(float(experiment.get("scheduler_min_lr", -1)), FROZEN_LR_FLOOR, "LR floor")
    _exact(str(experiment.get("text_ablation_mode", "")), FROZEN_TEXT_MODE, "text mode")
    _exact(str(experiment.get("generator_current_input_mode", "")), FROZEN_CURRENT_INPUT_MODE, "current input")
    _exact(str(experiment.get("generator_noise_mode", "")), FROZEN_NOISE_MODE, "noise mode")
    _exact(int(experiment.get("noise_dim", -1)), FROZEN_NOISE_DIM, "noise dim")
    _exact(bool(experiment.get("q4_loader_prediction_forbidden", False)), True, "Q4 guard")
    observed_folds = tuple(
        {
            key: str(row.get(key, ""))
            for key in (
                "fold_id",
                "train_end_utc",
                "validation_start_utc",
                "validation_end_utc",
            )
        }
        for row in experiment.get("folds", ())
    )
    _exact(observed_folds, FOLDS, "folds")
    observed_arms = tuple(
        {
            "arm_id": str(row.get("arm_id", "")),
            "mode": str(row.get("mode", "")),
            "filter_min_joint_cells": row.get("filter_min_joint_cells"),
            "soft_weight": bool(row.get("soft_weight", False)),
        }
        for row in experiment.get("arms", ())
    )
    _exact(observed_arms, ARMS, "arms")
    selection = _require_mapping(experiment.get("selection"), "selection")
    frozen_selection = {
        "bootstrap_iterations": 10_000,
        "bootstrap_seed": 20260820,
        "holm_family_size": 3,
        "alpha": 0.05,
        "minimum_nonworse_folds": 3,
        "minimum_nonworse_seeds": 2,
        "one_se_priority": ["C", "B", "D"],
        "q3_used_for_selection": False,
    }
    for key, expected in frozen_selection.items():
        _exact(selection.get(key), expected, f"selection.{key}")
    preflight = _require_mapping(experiment.get("preflight"), "preflight")
    _exact(float(preflight.get("max_peak_gpu_memory_gib", -1)), 20.0, "GPU cap")
    _exact(float(preflight.get("max_host_ram_fraction", -1)), 0.85, "RAM cap")
    audit = _require_mapping(config.get("label_reliability"), "label_reliability")
    _exact(
        tuple(str(value) for value in audit.get("methods", ())),
        ("support_preserving_bayesian", "strike_cluster"),
        "label_reliability.methods",
    )
    _exact(int(audit.get("bayesian_draws", -1)), 1000, "bayesian draws")
    _exact(int(audit.get("cluster_draws", -1)), 1000, "cluster draws")
    _exact(str(audit.get("priced_target_direction", "")), "backward", "priced target direction")
    _exact(
        str(audit.get("priced_target_calibration_time_column", "")),
        "calibration_datetime_utc",
        "priced target time column",
    )
    _exact(int(audit.get("target_anchor_horizon_minutes", -1)), 5, "target horizon")
    raw_globs = tuple(str(value).strip() for value in audit.get("raw_file_globs", ()))
    if len(raw_globs) != 1 or not raw_globs[0].endswith("/*.csv.gz"):
        raise LabelReliabilityExperimentError(
            "label_reliability.raw_file_globs must declare one daily *.csv.gz glob"
        )
    slots = int(runtime.get("slots_per_gpu", -1))
    if slots not in {12, 24}:
        raise LabelReliabilityExperimentError("runtime.slots_per_gpu must be 12 or 24")
    gpu_ids = tuple(int(value) for value in runtime.get("gpu_ids", ()))
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise LabelReliabilityExperimentError(
            "runtime.gpu_ids must contain exactly two distinct GPUs"
        )
    shape = _require_mapping(experiment.get("small_profile"), "small_profile")
    expected_shape = training.FROZEN_CAPACITY_PROFILES[FROZEN_PROFILE]
    if {key: int(value) for key, value in shape.items()} != expected_shape:
        raise LabelReliabilityExperimentError("small_profile differs from frozen Small")
    for family in ("regression", "wgan"):
        model = _require_mapping(models.get(family), f"models.{family}")
        command = "vol-regression-xlsx" if family == "regression" else "vol-xlsx"
        _exact(str(model.get("trainer_command", "")), command, f"{family} command")
        values = _require_mapping(model.get("training"), f"models.{family}.training")
        contracts = {
            "learning_rate": FROZEN_LR,
            "reduce_lr_min_lr": FROZEN_LR_FLOOR,
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": 3,
            "num_epochs": 100,
            "early_stopping_min_epochs": 30,
            "early_stopping_patience": 16 if family == "wgan" else 12,
            "residual_output_mode": FROZEN_RESIDUAL_MODE,
            "generator_current_input_mode": FROZEN_CURRENT_INPUT_MODE,
            "noise_dim": FROZEN_NOISE_DIM,
            "evaluate_initial_checkpoint": True,
        }
        if family == "wgan":
            contracts["generator_noise_mode"] = FROZEN_NOISE_MODE
        for key, expected in contracts.items():
            _exact(values.get(key), expected, f"{family}.{key}")
        for field in training.CAPACITY_PROFILE_SHAPE_FIELDS:
            _exact(int(values.get(field, -1)), int(expected_shape[field]), f"{family}.{field}")


def _resolved_config(config_path: str | Path) -> dict[str, Any]:
    path = _resolve_repo_path(config_path)
    root = _load_yaml_mapping(path, "label-reliability config")
    config = _require_mapping(root.get(ROOT_KEY), ROOT_KEY)
    _validate_frozen_config(config)
    resolved = deepcopy(config)
    resolved["source_config_path"] = str(path)
    resolved["datasets"]["root"] = str(_resolve_repo_path(resolved["datasets"]["root"]))
    resolved["runtime"]["python_executable"] = str(
        _resolve_repo_path(resolved["runtime"].get("python_executable", sys.executable))
    )
    audit = _require_mapping(resolved.get("label_reliability"), "label_reliability")
    raw_globs = audit.get("raw_file_globs")
    if not isinstance(raw_globs, list) or not raw_globs:
        raise LabelReliabilityExperimentError("raw_file_globs must be a non-empty list")
    audit["raw_file_globs"] = [
        str(_resolve_repo_path(Path(value).parent) / Path(value).name)
        for value in raw_globs
    ]
    for field in (
        "priced_target_rows",
        "current_surface_params",
        "rate_curve_path",
        "calendar_path",
    ):
        value = audit.get(field)
        if isinstance(value, list):
            audit[field] = [str(_resolve_repo_path(item)) for item in value]
        elif value not in (None, ""):
            audit[field] = str(_resolve_repo_path(value))
    pair_manifests = _require_mapping(audit.get("pair_manifests"), "pair_manifests")
    audit["pair_manifests"] = {
        str(key): str(_resolve_repo_path(value)) for key, value in pair_manifests.items()
    }
    resolved["label_reliability"] = audit
    _validate_pricing_source_contract(resolved)
    return resolved


def _validate_pricing_source_contract(resolved: Mapping[str, Any]) -> None:
    dataset_root = Path(str(resolved["datasets"]["root"]))
    audit = _require_mapping(resolved["label_reliability"], "label_reliability")
    surface_config_path = dataset_root / "shared_surface_inputs" / "surface-resolved_config.yaml"
    surface_root = _load_yaml_mapping(surface_config_path, "surface resolved config")
    builder = _require_mapping(surface_root.get("surface_builder"), "surface_builder")
    generation = _require_mapping(builder.get("generate_surface"), "generate_surface")
    if str(generation.get("pricing_model", "")).lower() != "black76":
        raise LabelReliabilityExperimentError("Surface pricing model must be black76")
    if Path(str(generation.get("rate_curve_path", ""))).resolve() != Path(
        str(audit["rate_curve_path"])
    ).resolve():
        raise LabelReliabilityExperimentError(
            "Label audit rate curve differs from surface-generation contract"
        )
    source_manifest = dataset_root / "source_manifest.csv"
    with source_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["source_role"]: row for row in csv.DictReader(handle)}
    for role, field in (
        ("rate_curve_csv", "rate_curve_path"),
        ("session_calendar_csv", "calendar_path"),
    ):
        if role not in rows:
            raise LabelReliabilityExperimentError(
                f"Dataset source manifest lacks {role}"
            )
        path = Path(str(audit[field])).resolve()
        if path != Path(rows[role]["path"]).resolve():
            raise LabelReliabilityExperimentError(f"{role} path lineage mismatch")
        if _sha256_file(path) != str(rows[role]["sha256"]):
            raise LabelReliabilityExperimentError(f"{role} SHA256 lineage mismatch")


def _raw_daily_files(
    resolved: Mapping[str, Any], *, required_dates: Iterable[str] | None = None
) -> list[Path]:
    """Expand the daily glob and retain files overlapping required UTC dates."""

    audit = _require_mapping(resolved["label_reliability"], "label_reliability")
    paths: list[Path] = []
    pattern = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv\.gz$")
    required = {str(value)[:10] for value in (required_dates or ())}
    for raw_pattern in audit["raw_file_globs"]:
        for raw_path in glob.glob(str(raw_pattern)):
            path = Path(raw_path).resolve()
            match = pattern.search(path.name)
            if match is None:
                continue
            first_day, second_day = match.group(1), match.group(2)
            in_scope = (
                any(first_day <= day <= second_day for day in required)
                if required
                else "2022-01-01" <= first_day < "2023-07-01"
            )
            if in_scope:
                paths.append(path)
    unique = sorted(set(paths))
    if not unique:
        raise LabelReliabilityExperimentError(
            "Daily raw glob produced no pre-Q3 2022--2023 files"
        )
    return unique


def _required_raw_dates(pair_manifests: Mapping[str, str]) -> set[str]:
    import pandas as pd

    dates: set[str] = set()
    for path in pair_manifests.values():
        frame = pd.read_csv(
            path,
            usecols=[
                "effective_origin_utc",
                "current_snapshot_time_utc",
                "target_snapshot_time_utc",
            ],
            dtype=str,
        )
        for column in frame.columns:
            values = pd.to_datetime(frame[column], errors="coerce", utc=True)
            if values.isna().any():
                raise LabelReliabilityExperimentError(
                    f"Invalid raw-date lineage in compact manifest: {path}"
                )
            dates.update(value.strftime("%Y-%m-%d") for value in values)
    if not dates:
        raise LabelReliabilityExperimentError("Compact manifests contain no raw dates")
    return dates


def _workbook_path(resolved: Mapping[str, Any], tolerance: int) -> Path:
    datasets = _require_mapping(resolved["datasets"], "datasets")
    return Path(datasets["root"]) / str(datasets["workbook_template"]).format(
        tolerance=int(tolerance), tolerance02=f"{int(tolerance):02d}"
    )


def _source_rows(
    resolved: Mapping[str, Any],
    *,
    pair_manifests: Mapping[str, str],
    fold_workbook_manifest: Any,
) -> list[dict[str, Any]]:
    paths: list[tuple[str, Path]] = [
        ("orchestration_config", Path(str(resolved["source_config_path"]))),
        (
            "surface_resolved_config",
            Path(resolved["datasets"]["root"])
            / "shared_surface_inputs"
            / "surface-resolved_config.yaml",
        ),
        (
            "dataset_source_manifest",
            Path(resolved["datasets"]["root"]) / "source_manifest.csv",
        ),
    ]
    for tolerance in FROZEN_TOLERANCES:
        paths.extend(
            [
                (f"training_workbook_{tolerance:02d}m", _workbook_path(resolved, tolerance)),
                (
                    f"support_audit_{tolerance:02d}m",
                    Path(resolved["datasets"]["root"])
                    / f"tolerance_{tolerance:02d}m"
                    / "surface_support_audit.csv.gz",
                ),
            ]
        )
    audit = _require_mapping(resolved["label_reliability"], "label_reliability")
    for tolerance, raw_path in pair_manifests.items():
        paths.append((f"bootstrap_pair_manifest_{int(tolerance):02d}m", Path(raw_path)))
    for row in fold_workbook_manifest.itertuples(index=False):
        paths.append(
            (
                f"fold_{row.data_role}_{int(row.tolerance_minutes):02d}m_{row.fold_id}",
                Path(str(row.path)),
            )
        )
    paths.extend(
        [
            (
                "fold_window_workbook_manifest",
                Path(str(fold_workbook_manifest.attrs["manifest_path"])),
            ),
            (
                "fold_window_contracts",
                Path(str(fold_workbook_manifest.attrs["contracts_path"])),
            ),
        ]
    )
    raw_dates = _required_raw_dates(pair_manifests)
    for index, path in enumerate(
        _raw_daily_files(resolved, required_dates=raw_dates)
    ):
        paths.append((f"raw_daily_option_file_{index:04d}", path))
    for field in (
        "priced_target_rows",
        "current_surface_params",
        "rate_curve_path",
        "calendar_path",
    ):
        value = audit.get(field)
        values = value if isinstance(value, list) else [value]
        for index, raw in enumerate(values):
            if raw in (None, ""):
                continue
            paths.append((f"bootstrap_{field}_{index}", Path(str(raw))))
    unique: dict[str, tuple[str, Path]] = {}
    for role, path in paths:
        resolved_path = path.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(resolved_path)
        unique[f"{role}:{resolved_path}"] = (role, resolved_path)
    return [
        {
            "source_role": role,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for role, path in unique.values()
    ]


def _code_rows() -> list[dict[str, Any]]:
    rows = [dict(row) for row in training._code_rows()]
    known = {str(row["relative_path"]) for row in rows}
    relatives = (
        "scripts/rq3/news_first_vol_label_reliability.py",
        "scripts/rq3/news_first_vol_label_reliability_analysis.py",
        "scripts/rq3/news_first_vol_label_reliability_report.py",
        "src/wgan_option/surface_generation/label_reliability.py",
    )
    for relative in relatives:
        if relative in known:
            continue
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "relative_path": relative,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
        known.add(relative)
    return rows


def canonical_profile_sha256(frame: Any) -> str:
    """Hash the exact six-column training profile independent of CSV encoding."""

    import pandas as pd

    missing = sorted(set(PROFILE_COLUMNS) - set(frame.columns))
    if missing:
        raise LabelReliabilityExperimentError(f"Profile columns missing: {missing}")
    selected = frame.loc[:, PROFILE_COLUMNS].copy().sort_values("pair_id", kind="stable")
    selected["tolerance_minutes"] = pd.to_numeric(
        selected["tolerance_minutes"], errors="raise"
    ).astype(int)
    selected["fold_id"] = selected["fold_id"].astype(str)
    selected["pair_id"] = selected["pair_id"].astype(str)
    def parse_bool(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
        raise LabelReliabilityExperimentError(f"Invalid included value: {value!r}")

    import numpy as np

    selected["included"] = selected["included"].map(parse_bool)
    for field in ("normalized_label_weight", "reliability_score"):
        selected[field] = pd.to_numeric(selected[field], errors="raise").astype(float)
    records = selected.to_dict(orient="records")
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    local_digest = hashlib.sha256(encoded).hexdigest()
    # Keep the writer and loader on one public hashing implementation.  The
    # explicit local computation above documents the persisted contract and
    # lets this orchestrator fail loudly if either side ever drifts.
    try:
        from wgan_option.utils.news_first_dataloaders import (
            label_reliability_profile_sha256,
        )
    except ModuleNotFoundError as exc:
        # Lightweight orchestration/test environments need not install torch;
        # training workers do, and re-run this cross-check before consumption.
        if not str(exc.name or "").startswith("torch"):
            raise
        return local_digest
    public_digest = label_reliability_profile_sha256(selected)
    if public_digest != local_digest:
        raise LabelReliabilityExperimentError(
            "Writer/loader canonical profile SHA256 implementations disagree"
        )
    return public_digest


def train_pair_universe_sha256(
    pair_ids: Iterable[str], *, fold_id: str, tolerance_minutes: int
) -> str:
    """Hash the exact pre-treatment train-pair universe for one fold."""

    normalized = sorted({str(value).strip() for value in pair_ids})
    if not normalized or any(not value for value in normalized):
        raise LabelReliabilityExperimentError("Train pair universe must be non-empty")
    payload = {
        "schema_version": 1,
        "fold_id": str(fold_id),
        "tolerance_minutes": int(tolerance_minutes),
        "pair_ids": normalized,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    local_digest = hashlib.sha256(encoded).hexdigest()
    try:
        from wgan_option.utils.news_first_dataloaders import (
            label_reliability_train_pair_universe_sha256,
        )
    except ModuleNotFoundError as exc:
        if not str(exc.name or "").startswith("torch"):
            raise
        return local_digest
    public_digest = label_reliability_train_pair_universe_sha256(
        normalized,
        fold_id=str(fold_id),
        tolerance_minutes=int(tolerance_minutes),
    )
    if public_digest != local_digest:
        raise LabelReliabilityExperimentError(
            "Writer/loader train-pair universe SHA256 implementations disagree"
        )
    return public_digest


def _read_training_pair_universe(
    resolved: Mapping[str, Any], tolerance: int, train_end_utc: str
):
    import pandas as pd

    workbook = _workbook_path(resolved, tolerance)
    sheet = str(resolved["datasets"]["sheet_name"])
    rows = pd.read_excel(
        workbook,
        sheet_name=sheet,
        usecols=["pair_id", "session_id", "effective_origin_utc"],
        dtype=str,
        engine="openpyxl",
    )
    timestamps = pd.to_datetime(rows["effective_origin_utc"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise LabelReliabilityExperimentError(f"Invalid timestamp in {workbook}")
    rows = rows.loc[timestamps < pd.Timestamp(train_end_utc)].copy()
    audit_path = (
        Path(resolved["datasets"]["root"])
        / f"tolerance_{int(tolerance):02d}m"
        / "surface_support_audit.csv.gz"
    )
    support = pd.read_csv(
        audit_path,
        usecols=[
            "pair_id",
            "surface_training_eligible",
            "joint_zero_support",
            "joint_strict_support_cell_count",
        ],
        low_memory=False,
    )
    truth = lambda values: values.map(  # noqa: E731
        lambda value: str(value).strip().lower() in {"1", "true", "yes"}
    )
    counts = pd.to_numeric(support["joint_strict_support_cell_count"], errors="coerce")
    eligible = truth(support["surface_training_eligible"]) & ~truth(
        support["joint_zero_support"]
    ) & counts.gt(0)
    supported = set(support.loc[eligible, "pair_id"].astype(str))
    rows = rows[rows["pair_id"].astype(str).isin(supported)].copy()
    if rows.empty:
        raise LabelReliabilityExperimentError(
            f"No training pairs for {tolerance}m before {train_end_utc}"
        )
    grouped = rows.groupby("pair_id", sort=True, as_index=False).agg(
        session_id=("session_id", "first"), row_count=("pair_id", "size")
    )
    if grouped["session_id"].fillna("").astype(str).str.strip().eq("").any():
        raise LabelReliabilityExperimentError("Training pair universe lacks session_id")
    return grouped


def materialize_reliability_profiles(
    pair_metrics: Any,
    resolved: Mapping[str, Any],
    output_dir: str | Path,
) -> Any:
    """Create 32 fold/tolerance/arm profiles from audit sufficient statistics."""

    import numpy as np
    import pandas as pd

    required = {
        "pair_id",
        "tolerance_minutes",
        "session_id",
        "effective_origin_utc",
        "c_i",
        "q_i",
        "m_i",
        "u_i",
        "u_estimable",
        "v_i",
        "h_i",
    }
    missing = sorted(required - set(pair_metrics.columns))
    if missing:
        raise LabelReliabilityExperimentError(
            f"Bootstrap pair metrics are missing: {missing}"
        )
    metrics = pair_metrics.copy()
    metrics["pair_id"] = metrics["pair_id"].astype(str)
    metrics["tolerance_minutes"] = pd.to_numeric(
        metrics["tolerance_minutes"], errors="raise"
    ).astype(int)
    if metrics.duplicated(["tolerance_minutes", "pair_id"]).any():
        raise LabelReliabilityExperimentError(
            "Bootstrap pair metrics must be unique by tolerance/pair"
        )
    times = pd.to_datetime(metrics["effective_origin_utc"], errors="coerce", utc=True)
    if times.isna().any() or bool((times >= pd.Timestamp(Q3_START_UTC)).any()):
        raise LabelReliabilityExperimentError(
            "Bootstrap pair metrics must be strictly pre-Q3"
        )
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for tolerance in FROZEN_TOLERANCES:
        tol_metrics = metrics[metrics["tolerance_minutes"] == tolerance].copy()
        for fold in FOLDS:
            fold_id = fold["fold_id"]
            universe = _read_training_pair_universe(
                resolved, tolerance, fold["train_end_utc"]
            )
            by_id = tol_metrics.set_index("pair_id", drop=False)
            expected_ids = set(universe["pair_id"].astype(str))
            universe_sha256 = train_pair_universe_sha256(
                expected_ids, fold_id=fold_id, tolerance_minutes=tolerance
            )
            observed_ids = set(by_id.index)
            if not expected_ids.issubset(observed_ids):
                raise LabelReliabilityExperimentError(
                    f"Bootstrap is missing {len(expected_ids - observed_ids)} "
                    f"training pairs for {tolerance}m/{fold_id}"
                )
            selected = by_id.loc[sorted(expected_ids)].copy()
            for field in ("c_i", "q_i", "m_i", "u_i", "v_i", "h_i"):
                selected[field] = pd.to_numeric(selected[field], errors="coerce")
            for field in ("c_i", "q_i", "m_i"):
                values = selected[field].to_numpy(dtype=float)
                if (
                    not np.isfinite(values).all()
                    or bool((values < 0.0).any())
                    or not np.equal(values, np.floor(values)).all()
                ):
                    raise LabelReliabilityExperimentError(
                        f"{field} must be finite nonnegative integers for "
                        f"{tolerance}m/{fold_id}"
                    )
            for field in ("v_i", "h_i"):
                values = selected[field].to_numpy(dtype=float)
                if (
                    not np.isfinite(values).all()
                    or bool((values < 0.0).any())
                    or bool((values > 1.0).any())
                ):
                    raise LabelReliabilityExperimentError(
                        f"{field} must be finite in [0,1] for {tolerance}m/{fold_id}"
                    )
            estimable = selected["u_estimable"].map(
                lambda value: str(value).strip().lower() in {"1", "true", "yes"}
            )
            finite_u = np.isfinite(selected["u_i"])
            if bool((estimable & (~finite_u | selected["u_i"].le(0.0))).any()):
                raise LabelReliabilityExperimentError(
                    f"Estimable u_i must be finite and positive for {tolerance}m/{fold_id}"
                )
            if bool((~estimable & finite_u).any()):
                raise LabelReliabilityExperimentError(
                    f"Unestimable u_i must be NaN for {tolerance}m/{fold_id}"
                )
            valid_u = selected.loc[estimable, "u_i"]
            if valid_u.empty:
                raise LabelReliabilityExperimentError(
                    f"No estimable u_i for {tolerance}m/{fold_id}"
                )
            tau = float(valid_u.quantile(0.75))
            if not math.isfinite(tau) or tau <= 0.0:
                raise LabelReliabilityExperimentError(
                    f"tau_f must be positive for {tolerance}m/{fold_id}"
                )
            session_lookup = universe.set_index("pair_id")["session_id"].astype(str)
            estimable_ids = set(selected.loc[estimable, "pair_id"].astype(str))
            all_sessions = set(session_lookup)
            estimable_sessions = set(session_lookup.loc[sorted(estimable_ids)])
            estimable_pair_fraction = len(estimable_ids) / len(selected)
            estimable_session_fraction = len(estimable_sessions) / len(all_sessions)
            global_estimability_valid = bool(
                estimable_pair_fraction >= 0.60
                and estimable_session_fraction >= 0.80
            )
            s = np.minimum(1.0, np.sqrt(np.maximum(selected["c_i"], 0.0) / 16.0))
            d = np.minimum(1.0, np.sqrt(np.maximum(selected["q_i"], 0.0) / 8.0))
            e = np.minimum(1.0, np.sqrt(np.maximum(selected["m_i"], 0.0) / 4.0))
            b = pd.Series(0.0, index=selected.index)
            b.loc[estimable] = 1.0 / (1.0 + (selected.loc[estimable, "u_i"] / tau) ** 2)
            v = selected["v_i"]
            h = selected["h_i"]
            product = (s * d * e * b * v * h).clip(lower=0.0, upper=1.0)
            reliability = product ** (1.0 / 6.0)
            if not np.isfinite(reliability.to_numpy(dtype=float)).all():
                raise LabelReliabilityExperimentError(
                    f"Reliability score is non-finite for {tolerance}m/{fold_id}"
                )
            raw_weight = 0.5 + 0.5 * reliability
            for arm in ARMS:
                arm_id = str(arm["arm_id"])
                included = pd.Series(True, index=selected.index)
                if arm["filter_min_joint_cells"] is not None:
                    included = selected["c_i"].ge(int(arm["filter_min_joint_cells"]))
                weights = pd.Series(1.0, index=selected.index)
                if bool(arm["soft_weight"]):
                    if not bool(included.any()):
                        raise LabelReliabilityExperimentError(
                            f"{arm_id}/{tolerance}m/{fold_id} retains no pairs"
                        )
                    weights.loc[included] = (
                        raw_weight.loc[included] / float(raw_weight.loc[included].mean())
                    )
                weights.loc[~included] = 0.0
                included_weights = weights.loc[included]
                if (
                    not np.isfinite(included_weights.to_numpy()).all()
                    or not math.isclose(
                        float(included_weights.mean()), 1.0, rel_tol=0.0, abs_tol=1e-12
                    )
                    or float(included_weights.min()) < 0.5 - 1e-12
                    or float(included_weights.max()) > 2.0 + 1e-12
                ):
                    raise LabelReliabilityExperimentError(
                        f"Normalized weights violate [0.5,2]/mean-one: "
                        f"{arm_id}/{tolerance}m/{fold_id}"
                    )
                profile = pd.DataFrame(
                    {
                        "tolerance_minutes": tolerance,
                        "fold_id": fold_id,
                        "pair_id": selected["pair_id"].astype(str).to_numpy(),
                        "included": included.astype(bool).to_numpy(),
                        "normalized_label_weight": weights.astype(float).to_numpy(),
                        "reliability_score": reliability.astype(float).to_numpy(),
                        "train_pair_universe_sha256": universe_sha256,
                    }
                ).sort_values("pair_id", kind="stable")
                path = target / arm_id / f"tolerance_{tolerance:02d}m_{fold_id}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                profile.to_csv(path, index=False, columns=PROFILE_FILE_COLUMNS)
                included_ids = set(profile.loc[profile["included"], "pair_id"])
                included_sessions = set(session_lookup.loc[sorted(included_ids)])
                all_sessions = set(session_lookup)
                pair_retention = len(included_ids) / len(profile)
                session_retention = len(included_sessions) / len(all_sessions)
                retention_valid = bool(
                    arm_id not in {"B", "D"}
                    or (pair_retention >= 0.60 and session_retention >= 0.80)
                )
                manifest_rows.append(
                    {
                        "arm_id": arm_id,
                        "label_reliability_mode": arm["mode"],
                        "tolerance_minutes": tolerance,
                        "fold_id": fold_id,
                        "pair_count": len(profile),
                        "session_count": len(all_sessions),
                        "included_pair_count": len(included_ids),
                        "included_session_count": len(included_sessions),
                        "pair_retention_fraction": pair_retention,
                        "session_retention_fraction": session_retention,
                        "tau_u_q75": tau,
                        "u_estimable_pair_fraction": estimable_pair_fraction,
                        "u_estimable_session_fraction": estimable_session_fraction,
                        "global_estimability_gate_valid": global_estimability_valid,
                        "retention_gate_valid": retention_valid,
                        "profile_path": str(path),
                        "manifest_sha256": _sha256_file(path),
                        "profile_sha256": canonical_profile_sha256(profile),
                        "train_pair_universe_sha256": universe_sha256,
                    }
                )
    manifest = pd.DataFrame(manifest_rows, columns=PROFILE_MANIFEST_COLUMNS)
    if len(manifest) != len(FROZEN_TOLERANCES) * len(FOLDS) * len(ARMS):
        raise AssertionError("Expected exactly 32 reliability profiles")
    manifest.to_csv(target / "reliability_profile_manifest.csv", index=False)
    return manifest


def _stage_specs(
    stage_id: str,
    *,
    selected_arm: str | None = None,
) -> list[dict[str, Any]]:
    if stage_id == STAGE_1:
        family, tolerance, arms = "regression", 5, tuple(row["arm_id"] for row in ARMS)
    elif stage_id == STAGE_2:
        family, tolerance, arms = "regression", 30, ("A", str(selected_arm or ""))
    elif stage_id == STAGE_3:
        family, tolerance, arms = "wgan", 5, ("A", str(selected_arm or ""))
    else:
        raise LabelReliabilityExperimentError(f"Unknown stage: {stage_id}")
    if stage_id != STAGE_1 and str(selected_arm) not in {"B", "C", "D"}:
        raise LabelReliabilityExperimentError(
            f"{stage_id} requires one selected candidate arm"
        )
    rows = []
    for arm_id in arms:
        for fold in FOLDS:
            for seed in FROZEN_SEEDS:
                rows.append(
                    {
                        "stage_id": stage_id,
                        "model_family": family,
                        "tolerance_minutes": tolerance,
                        "arm_id": arm_id,
                        "fold_id": fold["fold_id"],
                        "seed": seed,
                    }
                )
    if len(rows) != EXPECTED_STAGE_JOBS[stage_id]:
        raise AssertionError(f"{stage_id} job count drifted")
    return rows


def _job_id(spec: Mapping[str, Any]) -> str:
    return (
        f"{spec['stage_id']}_{spec['model_family']}_{spec['arm_id']}_"
        f"{spec['fold_id']}_seed_{int(spec['seed']):03d}_"
        f"{int(spec['tolerance_minutes']):02d}m"
    ).lower()


def schedule_job_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    gpu_ids: Sequence[int],
    slots_per_gpu: int,
    wave_offset: int = 0,
) -> list[dict[str, Any]]:
    """Cross-balance consecutive arm/fold/seed cells over two GPUs."""

    gpus = tuple(int(value) for value in gpu_ids)
    if len(gpus) != 2 or len(set(gpus)) != 2 or int(slots_per_gpu) not in {12, 24}:
        raise LabelReliabilityExperimentError("Invalid frozen GPU schedule")
    capacity = len(gpus) * int(slots_per_gpu)
    rows = []
    for index, raw in enumerate(specs):
        gpu_index = index % len(gpus)
        rows.append(
            {
                **dict(raw),
                "job_id": _job_id(raw),
                "stage_job_index": index,
                "wave": int(wave_offset) + index // capacity,
                "gpu_id": gpus[gpu_index],
                "gpu_slot": (index // len(gpus)) % int(slots_per_gpu),
            }
        )
    if len({row["job_id"] for row in rows}) != len(rows):
        raise AssertionError("Job IDs are not unique")
    return rows


def build_full_job_matrix(
    selected_arm: str,
    *,
    gpu_ids: Sequence[int] = (0, 1),
    slots_per_gpu: int = 24,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wave_offset = 0
    for stage in STAGES:
        scheduled = schedule_job_specs(
            _stage_specs(stage, selected_arm=selected_arm),
            gpu_ids=gpu_ids,
            slots_per_gpu=slots_per_gpu,
            wave_offset=wave_offset,
        )
        rows.extend(scheduled)
        wave_offset = max(row["wave"] for row in rows) + 1
    if len(rows) != EXPECTED_MAX_JOBS:
        raise AssertionError("Full conditional matrix must contain 96 jobs")
    return rows


def _job_spec_hash(job: Mapping[str, Any]) -> str:
    return _payload_sha256(
        {key: value for key, value in job.items() if key != "job_spec_sha256"}
    )


def _initial_registry(resolved: Mapping[str, Any], root: Path) -> dict[str, Any]:
    runtime = resolved["runtime"]
    jobs = schedule_job_specs(
        _stage_specs(STAGE_1),
        gpu_ids=runtime["gpu_ids"],
        slots_per_gpu=int(runtime["slots_per_gpu"]),
    )
    source_hash = _sha256_file(root / "source_hashes.csv")
    for job in jobs:
        job.update(
            {
                "capacity_profile": FROZEN_PROFILE,
                "capacity_profile_sha256": _capacity_profile_sha256(),
                "expected_model_parameters": int(
                    training.FROZEN_CAPACITY_PROFILES[FROZEN_PROFILE][
                        "expected_regression_parameters"
                    ]
                ),
                "lr_profile": FROZEN_LR_PROFILE,
                "lr_profile_sha256": _lr_profile_sha256(),
                "initial_learning_rate": FROZEN_LR,
                "scheduler_min_lr": FROZEN_LR_FLOOR,
                "text_ablation_mode": FROZEN_TEXT_MODE,
                "support_mask_mode": FROZEN_SUPPORT_MODE,
                "profile_path": "",
                "profile_manifest_sha256": "",
                "profile_sha256": "",
                "training_config_path": "",
                "config_sha256": "",
                "source_manifest_sha256": source_hash,
                "state": "awaiting_bootstrap",
            }
        )
        job["job_spec_sha256"] = _job_spec_hash(job)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": EXPERIMENT_KIND,
        "resolved_config_sha256": _payload_sha256(resolved),
        "conditional_registry": True,
        "maximum_jobs": EXPECTED_MAX_JOBS,
        "jobs": jobs,
    }


def _stage_status_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": EXPERIMENT_KIND,
        "status": "prepared_bootstrap_required",
        "current_stage": "bootstrap",
        "q3_predictions_generated": False,
        "q3_used_for_selection": False,
        "q4_loader_materialized": False,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "stages": {
            STAGE_1: {"status": "awaiting_bootstrap", "expected_jobs": 48},
            STAGE_2: {"status": "conditional", "expected_jobs": 24},
            STAGE_3: {"status": "conditional", "expected_jobs": 24},
        },
        "updated_at_utc": _utc_now(),
    }


def _write_split_manifest(root: Path, resolved: Mapping[str, Any]) -> Path:
    rows: list[dict[str, Any]] = []
    for tolerance in FROZEN_TOLERANCES:
        for fold in FOLDS:
            universe = _read_training_pair_universe(
                resolved, tolerance, fold["train_end_utc"]
            )
            rows.append(
                {
                    "tolerance_minutes": tolerance,
                    "fold_id": fold["fold_id"],
                    "split": "train",
                    "interval_start_utc": "-infinity",
                    "interval_end_utc_exclusive": fold["train_end_utc"],
                    "pair_count": len(universe),
                    "session_count": universe["session_id"].nunique(),
                    "q3_rows_passed_to_training_loader": 0,
                    "q4_rows_passed_to_training_loader": 0,
                    "prepare_source_sheet_physically_read": True,
                    "prepare_read_scope_note": (
                        "prepare reads the mixed source sheet only to materialize "
                        "strict fold windows; models and selection receive no "
                        "post-validation rows"
                    ),
                }
            )
    return _write_csv(root / "fold_split_manifest.csv", rows, tuple(rows[0]))


def data_window_contract_sha256(
    *,
    fold_id: str,
    train_end_utc: str,
    validation_end_utc: str,
    train_workbook_sha256: str,
    validation_workbook_sha256: str,
) -> str:
    """Hash the strict pre-deserialization train/validation window contract."""

    payload = {
        "schema_version": 1,
        "fold_id": str(fold_id),
        "train": {
            "origin_start_utc_inclusive": None,
            "origin_end_utc_exclusive": str(train_end_utc),
            "workbook_sha256": str(train_workbook_sha256),
        },
        "validation": {
            "origin_start_utc_inclusive": str(train_end_utc),
            "origin_end_utc_exclusive": str(validation_end_utc),
            "workbook_sha256": str(validation_workbook_sha256),
        },
        "post_validation": {
            "forbidden_from_utc_inclusive": str(validation_end_utc),
            "materialized": False,
        },
    }
    return _payload_sha256(payload)


def materialize_fold_window_workbooks(
    resolved: Mapping[str, Any], output_dir: str | Path
):
    """Write hash-bound workbooks containing only each job's inner window."""

    import pandas as pd

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    validation_by_fold: dict[str, dict[str, Any]] = {}
    for tolerance in FROZEN_TOLERANCES:
        source = _workbook_path(resolved, tolerance)
        frame = pd.read_excel(source, sheet_name="gan_input_ready", engine="openpyxl")
        timestamps = pd.to_datetime(
            frame["effective_origin_utc"], errors="coerce", utc=True
        )
        if timestamps.isna().any():
            raise LabelReliabilityExperimentError(
                f"Invalid effective_origin_utc in fold source: {source}"
            )
        for fold in FOLDS:
            train_end = pd.Timestamp(fold["train_end_utc"])
            validation_end = pd.Timestamp(fold["validation_end_utc"])
            train = frame.loc[timestamps < train_end].copy()
            if train.empty:
                raise LabelReliabilityExperimentError(
                    f"Empty fold train workbook: {tolerance}m/{fold['fold_id']}"
                )
            train_path = (
                root
                / f"tolerance_{tolerance:02d}m"
                / f"{fold['fold_id']}_train.xlsx"
            )
            train_path.parent.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(train_path, engine="openpyxl") as writer:
                train.to_excel(writer, sheet_name="gan_input_ready", index=False)
            train_times = pd.to_datetime(
                train["effective_origin_utc"], errors="raise", utc=True
            )
            if bool((train_times >= train_end).any()):
                raise LabelReliabilityExperimentError("Train fold slice leaked forward")
            rows.append(
                {
                    "data_role": "train",
                    "tolerance_minutes": tolerance,
                    "fold_id": fold["fold_id"],
                    "origin_start_utc_inclusive": "",
                    "origin_end_utc_exclusive": fold["train_end_utc"],
                    "row_count": len(train),
                    "pair_count": train["pair_id"].astype(str).nunique(),
                    "session_count": train["session_id"].astype(str).nunique(),
                    "path": str(train_path),
                    "sha256": _sha256_file(train_path),
                    "source_path": str(source),
                    "source_sha256": _sha256_file(source),
                }
            )
            if tolerance == 5:
                validation = frame.loc[
                    (timestamps >= train_end) & (timestamps < validation_end)
                ].copy()
                if validation.empty:
                    raise LabelReliabilityExperimentError(
                        f"Empty validation workbook: {fold['fold_id']}"
                    )
                validation_path = root / "common_05m" / f"{fold['fold_id']}_validation.xlsx"
                validation_path.parent.mkdir(parents=True, exist_ok=True)
                with pd.ExcelWriter(validation_path, engine="openpyxl") as writer:
                    validation.to_excel(writer, sheet_name="gan_input_ready", index=False)
                validation_times = pd.to_datetime(
                    validation["effective_origin_utc"], errors="raise", utc=True
                )
                if not (
                    (validation_times >= train_end)
                    & (validation_times < validation_end)
                ).all():
                    raise LabelReliabilityExperimentError(
                        "Validation fold slice escaped its half-open interval"
                    )
                validation_row = {
                    "data_role": "validation",
                    "tolerance_minutes": 5,
                    "fold_id": fold["fold_id"],
                    "origin_start_utc_inclusive": fold["train_end_utc"],
                    "origin_end_utc_exclusive": fold["validation_end_utc"],
                    "row_count": len(validation),
                    "pair_count": validation["pair_id"].astype(str).nunique(),
                    "session_count": validation["session_id"].astype(str).nunique(),
                    "path": str(validation_path),
                    "sha256": _sha256_file(validation_path),
                    "source_path": str(source),
                    "source_sha256": _sha256_file(source),
                }
                rows.append(validation_row)
                validation_by_fold[fold["fold_id"]] = validation_row
    manifest = pd.DataFrame(rows)
    expected_rows = len(FROZEN_TOLERANCES) * len(FOLDS) + len(FOLDS)
    if len(manifest) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} fold-window workbooks")
    contracts = []
    for fold in FOLDS:
        validation = validation_by_fold[fold["fold_id"]]
        for tolerance in FROZEN_TOLERANCES:
            train = manifest[
                (manifest["data_role"] == "train")
                & (manifest["fold_id"] == fold["fold_id"])
                & (manifest["tolerance_minutes"] == tolerance)
            ].iloc[0]
            contracts.append(
                {
                    "tolerance_minutes": tolerance,
                    "fold_id": fold["fold_id"],
                    "train_path": train["path"],
                    "train_sha256": train["sha256"],
                    "validation_path": validation["path"],
                    "validation_sha256": validation["sha256"],
                    "data_window_contract_sha256": data_window_contract_sha256(
                        fold_id=fold["fold_id"],
                        train_end_utc=fold["train_end_utc"],
                        validation_end_utc=fold["validation_end_utc"],
                        train_workbook_sha256=train["sha256"],
                        validation_workbook_sha256=validation["sha256"],
                    ),
                }
            )
    contract_frame = pd.DataFrame(contracts)
    manifest.to_csv(root / "fold_window_workbook_manifest.csv", index=False)
    contract_frame.to_csv(root / "fold_window_contracts.csv", index=False)
    return manifest, contract_frame


def _fold_window_contract_map(root: Path) -> dict[tuple[int, str], Any]:
    import pandas as pd

    path = root / "fold_workbooks" / "fold_window_contracts.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) != 8:
        raise LabelReliabilityExperimentError("Fold-window contract matrix must have 8 rows")
    return {
        (int(row.tolerance_minutes), str(row.fold_id)): row
        for row in frame.itertuples(index=False)
    }


def prepare_label_reliability_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    reuse: bool = False,
) -> Path:
    """Freeze an immutable root and the 48 placeholder screen jobs."""

    resolved = _resolved_config(config_path)
    root = _resolve_repo_path(output_dir)
    if root.exists() and any(root.iterdir()):
        if not reuse:
            raise FileExistsError(
                f"Experiment root already exists; pass --resume/--reuse: {root}"
            )
        _validate_registry(root, config_path)
        return root
    root.mkdir(parents=True, exist_ok=True)
    for directory in (
        "registry/jobs",
        "registry/stages",
        "bootstrap",
        "reliability_profiles",
        "training_configs",
        "fold_workbooks",
        "runs",
        "logs",
        "analysis",
        "report",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    _write_yaml(root / "resolved_config.yaml", {ROOT_KEY: resolved})
    (root / "registry" / "resolved_config.sha256").write_text(
        _payload_sha256(resolved) + "\n", encoding="utf-8"
    )
    pair_manifests = _prepare_bootstrap_pair_manifests(root, resolved)
    fold_manifest, _ = materialize_fold_window_workbooks(
        resolved, root / "fold_workbooks"
    )
    fold_manifest.attrs["manifest_path"] = str(
        root / "fold_workbooks" / "fold_window_workbook_manifest.csv"
    )
    fold_manifest.attrs["contracts_path"] = str(
        root / "fold_workbooks" / "fold_window_contracts.csv"
    )
    source_rows = _source_rows(
        resolved,
        pair_manifests=pair_manifests,
        fold_workbook_manifest=fold_manifest,
    )
    _write_csv(root / "source_hashes.csv", source_rows, tuple(source_rows[0]))
    code_rows = _code_rows()
    _write_csv(root / "code_hashes.csv", code_rows, tuple(code_rows[0]))
    registry = _initial_registry(resolved, root)
    _write_json(root / "registry" / "jobs.json", registry)
    for job in registry["jobs"]:
        _write_json(
            _job_status_path(root, job["job_id"]),
            {"job_id": job["job_id"], "status": "awaiting_bootstrap", "attempt": 0},
        )
    _write_json(root / "label_reliability_stage_status.json", _stage_status_payload())
    _write_split_manifest(root, resolved)
    _refresh_exports(root)
    return root


def _manifest_map(root: Path):
    import pandas as pd

    path = root / "reliability_profiles" / "reliability_profile_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) != 32:
        raise LabelReliabilityExperimentError("Reliability manifest must have 32 rows")
    return {
        (str(row.arm_id), int(row.tolerance_minutes), str(row.fold_id)): row
        for row in frame.itertuples(index=False)
    }


def _training_payload(
    resolved: Mapping[str, Any],
    root: Path,
    job: Mapping[str, Any],
    profile_row: Any,
) -> dict[str, Any]:
    fold = next(row for row in FOLDS if row["fold_id"] == job["fold_id"])
    family = str(job["model_family"])
    payload = training._training_payload(
        resolved,
        family=family,
        tolerance=int(job["tolerance_minutes"]),
        text_ablation_mode=FROZEN_TEXT_MODE,
        multi_mode=True,
        experiment_root=root,
        capacity_profile=None,
    )
    shape = _require_mapping(
        _experiment_config(resolved)["small_profile"], "small_profile"
    )
    payload.update(
        {field: int(shape[field]) for field in training.CAPACITY_PROFILE_SHAPE_FIELDS}
    )
    arm = next(row for row in ARMS if row["arm_id"] == job["arm_id"])
    window = _fold_window_contract_map(root)[
        (int(job["tolerance_minutes"]), str(job["fold_id"]))
    ]
    output_root = (
        root
        / "runs"
        / str(job["stage_id"])
        / family
        / str(job["arm_id"])
        / str(job["fold_id"])
        / f"seed_{int(job['seed']):03d}"
        / f"tolerance_{int(job['tolerance_minutes']):02d}m"
    )
    payload.update(
        {
            "seed": int(job["seed"]),
            "data_path": str(window.train_path),
            "news_first_common_eval_data_path": str(window.validation_path),
            "news_first_capacity_profile": FROZEN_PROFILE,
            "news_first_capacity_profile_sha256": _capacity_profile_sha256(),
            "news_first_lr_profile": FROZEN_LR_PROFILE,
            "news_first_lr_profile_sha256": _lr_profile_sha256(),
            "news_first_fixed_learning_rate_profile": FROZEN_LR_PROFILE,
            "news_first_fixed_learning_rate_profile_sha256": _lr_profile_sha256(),
            "learning_rate": FROZEN_LR,
            "reduce_lr_min_lr": FROZEN_LR_FLOOR,
            "use_reduce_lr_on_plateau": True,
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": 3,
            "news_first_train_end_utc": fold["train_end_utc"],
            "news_first_validation_end_utc": fold["validation_end_utc"],
            "news_first_materialize_test_loader": False,
            "news_first_label_reliability_train_data_sha256": str(
                window.train_sha256
            ),
            "news_first_label_reliability_validation_data_sha256": str(
                window.validation_sha256
            ),
            "news_first_label_reliability_data_window_contract_sha256": str(
                window.data_window_contract_sha256
            ),
            "news_first_label_reliability_mode": arm["mode"],
            "news_first_label_reliability_manifest_path": str(profile_row.profile_path),
            "news_first_label_reliability_manifest_sha256": str(
                profile_row.manifest_sha256
            ),
            "news_first_label_reliability_profile_sha256": str(
                profile_row.profile_sha256
            ),
            "news_first_label_reliability_fold_id": str(job["fold_id"]),
            "news_first_label_reliability_train_pair_universe_sha256": str(
                profile_row.train_pair_universe_sha256
            ),
            "generator_current_input_mode": FROZEN_CURRENT_INPUT_MODE,
            "generator_noise_mode": FROZEN_NOISE_MODE,
            "residual_output_mode": FROZEN_RESIDUAL_MODE,
            "output_root": str(output_root),
        }
    )
    return payload


def _materialize_job_configs(
    root: Path,
    resolved: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    profiles = _manifest_map(root)
    result: list[dict[str, Any]] = []
    source_hash = _sha256_file(root / "source_hashes.csv")
    for raw in jobs:
        job = dict(raw)
        job.update(
            {
                "capacity_profile": FROZEN_PROFILE,
                "capacity_profile_sha256": _capacity_profile_sha256(),
                "expected_model_parameters": int(
                    training.FROZEN_CAPACITY_PROFILES[FROZEN_PROFILE][
                        "expected_wgan_parameters"
                        if job["model_family"] == "wgan"
                        else "expected_regression_parameters"
                    ]
                ),
                "lr_profile": FROZEN_LR_PROFILE,
                "lr_profile_sha256": _lr_profile_sha256(),
                "initial_learning_rate": FROZEN_LR,
                "scheduler_min_lr": FROZEN_LR_FLOOR,
                "text_ablation_mode": FROZEN_TEXT_MODE,
                "support_mask_mode": FROZEN_SUPPORT_MODE,
            }
        )
        key = (job["arm_id"], int(job["tolerance_minutes"]), job["fold_id"])
        if key not in profiles:
            raise LabelReliabilityExperimentError(f"Missing profile for job {job['job_id']}")
        row = profiles[key]
        window = _fold_window_contract_map(root)[
            (int(job["tolerance_minutes"]), str(job["fold_id"]))
        ]
        profile_path = Path(str(row.profile_path))
        if _sha256_file(profile_path) != str(row.manifest_sha256):
            raise LabelReliabilityExperimentError("Profile byte hash mismatch")
        payload = _training_payload(resolved, root, job, row)
        config_path = root / "training_configs" / f"{job['job_id']}.yaml"
        _write_yaml(config_path, payload)
        job.update(
            {
                "profile_path": str(profile_path),
                "profile_manifest_sha256": str(row.manifest_sha256),
                "profile_sha256": str(row.profile_sha256),
                "train_pair_universe_sha256": str(row.train_pair_universe_sha256),
                "train_data_path": str(window.train_path),
                "train_data_sha256": str(window.train_sha256),
                "validation_data_path": str(window.validation_path),
                "validation_data_sha256": str(window.validation_sha256),
                "data_window_contract_sha256": str(
                    window.data_window_contract_sha256
                ),
                "profile_retention_gate_valid": bool(row.retention_gate_valid),
                "training_config_path": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "source_manifest_sha256": source_hash,
                "state": "prepared",
            }
        )
        job["job_spec_sha256"] = _job_spec_hash(job)
        result.append(job)
        status_path = _job_status_path(root, job["job_id"])
        previous = _read_json(status_path) if status_path.is_file() else {}
        if previous.get("status") in {"completed", "running"}:
            raise LabelReliabilityExperimentError(
                f"Cannot rematerialize an active/completed job: {job['job_id']}"
            )
        _write_json(
            status_path,
            {"job_id": job["job_id"], "status": "prepared", "attempt": 0},
        )
    return result


def _prepare_bootstrap_pair_manifests(
    root: Path, resolved: Mapping[str, Any]
) -> dict[str, str]:
    """Freeze compact, pre-Q3 pair rows with both surfaces and strict support."""

    import pandas as pd

    surface_fields = (
        "tolerance_minutes",
        "pair_id",
        "session_id",
        "effective_origin_utc",
        "current_snapshot_time_utc",
        "target_snapshot_time_utc",
        "current_surface_param_json",
        "target_surface_param_json",
        "current_surface_flat",
        "target_surface_flat",
        "strike_grid",
        "maturity_days_grid",
    )
    result: dict[str, str] = {}
    for tolerance in FROZEN_TOLERANCES:
        workbook = _workbook_path(resolved, tolerance)
        available = pd.read_excel(
            workbook,
            sheet_name="gan_input_ready",
            nrows=0,
            engine="openpyxl",
        ).columns
        required_without_tolerance = set(surface_fields) - {"tolerance_minutes"}
        missing = sorted(required_without_tolerance - set(available))
        if missing:
            raise LabelReliabilityExperimentError(
                f"Bootstrap source {workbook} is missing fields: {missing}"
            )
        surface = pd.read_excel(
            workbook,
            sheet_name="gan_input_ready",
            usecols=sorted(required_without_tolerance),
            dtype=str,
            engine="openpyxl",
        )
        timestamps = pd.to_datetime(
            surface["effective_origin_utc"], errors="coerce", utc=True
        )
        if timestamps.isna().any():
            raise LabelReliabilityExperimentError(
                f"Invalid effective_origin_utc in {workbook}"
            )
        surface = surface.loc[timestamps < pd.Timestamp(Q3_START_UTC)].copy()
        if surface.empty:
            raise LabelReliabilityExperimentError(
                f"No pre-Q3 bootstrap pairs in {workbook}"
            )
        equality_fields = sorted(required_without_tolerance - {"pair_id"})
        disagreement = surface.groupby("pair_id", dropna=False)[equality_fields].nunique(
            dropna=False
        )
        bad = disagreement.gt(1).any(axis=1)
        if bad.any():
            preview = disagreement.index[bad].astype(str).tolist()[:5]
            raise LabelReliabilityExperimentError(
                "Rows sharing pair_id disagree on canonical surface lineage: "
                f"{preview}"
            )
        surface = surface.sort_values("pair_id", kind="stable").drop_duplicates(
            "pair_id", keep="first"
        )
        surface.insert(0, "tolerance_minutes", int(tolerance))
        support_path = (
            Path(resolved["datasets"]["root"])
            / f"tolerance_{tolerance:02d}m"
            / "surface_support_audit.csv.gz"
        )
        support = pd.read_csv(support_path, low_memory=False)
        if support["pair_id"].astype(str).duplicated().any():
            raise LabelReliabilityExperimentError(
                f"Support audit is not pair unique: {support_path}"
            )
        support["pair_id"] = support["pair_id"].astype(str)
        duplicate_fields = set(surface.columns) & set(support.columns) - {"pair_id"}
        support = support.drop(columns=sorted(duplicate_fields))
        compact = surface.merge(
            support, on="pair_id", how="left", validate="one_to_one", indicator=True
        )
        if not compact["_merge"].eq("both").all():
            raise LabelReliabilityExperimentError(
                f"Surface/support pair join is incomplete for {tolerance}m"
            )
        compact = compact.drop(columns="_merge")
        output = root / "bootstrap" / f"pair_manifest_{tolerance:02d}m.csv.gz"
        compact.to_csv(output, index=False, compression="gzip")
        result[str(tolerance)] = str(output)
    return result


def run_label_reliability_bootstrap_action(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> Path:
    """Run the audit core and turn sufficient statistics into train profiles."""

    root = prepare_label_reliability_experiment(
        config_path, output_dir, reuse=bool(resume)
    )
    _validate_registry(root, config_path)
    resolved = _require_mapping(
        _load_yaml_mapping(root / "resolved_config.yaml", "resolved config")[ROOT_KEY],
        ROOT_KEY,
    )
    profile_manifest = root / "reliability_profiles" / "reliability_profile_manifest.csv"
    if profile_manifest.is_file():
        if not resume:
            raise FileExistsError("Bootstrap profiles already exist; pass --resume")
        _validate_profiles(root)
        return root
    from wgan_option.surface_generation.label_reliability import (
        run_label_reliability_bootstrap,
    )

    audit_config = deepcopy(resolved["label_reliability"])
    audit_config["pair_manifests"] = {
        str(tolerance): str(
            root / "bootstrap" / f"pair_manifest_{int(tolerance):02d}m.csv.gz"
        )
        for tolerance in FROZEN_TOLERANCES
    }
    audit_config["raw_files"] = [
        str(path)
        for path in _raw_daily_files(
            resolved,
            required_dates=_required_raw_dates(audit_config["pair_manifests"]),
        )
    ]
    audit_output_dir = root / "bootstrap" / "audit"
    audit_manifest = run_label_reliability_bootstrap(
        audit_config, audit_output_dir, resume=bool(resume)
    )
    audit_manifest = Path(audit_manifest).resolve()
    if audit_output_dir.resolve() not in audit_manifest.parents:
        raise LabelReliabilityExperimentError("Audit manifest escaped bootstrap root")
    pair_path = audit_manifest.parent / "pair_metrics.csv.gz"
    if not pair_path.is_file():
        raise FileNotFoundError(pair_path)
    import pandas as pd

    pair_metrics = pd.read_csv(pair_path, low_memory=False)
    profile_rows = materialize_reliability_profiles(
        pair_metrics, resolved, root / "reliability_profiles"
    )
    eligibility = profile_rows[
        [
            "tolerance_minutes",
            "fold_id",
            "u_estimable_pair_fraction",
            "u_estimable_session_fraction",
            "global_estimability_gate_valid",
        ]
    ].drop_duplicates()
    eligibility_path = root / "bootstrap_fold_eligibility.csv"
    eligibility.to_csv(eligibility_path, index=False)
    eligibility_valid = bool(eligibility["global_estimability_gate_valid"].all())
    eligibility_payload = {
        "schema_version": 1,
        "status": "pass" if eligibility_valid else "fail",
        "minimum_estimable_pair_fraction": 0.60,
        "minimum_estimable_session_fraction": 0.80,
        "fold_count": int(len(eligibility)),
        "eligibility_path": str(eligibility_path),
        "eligibility_sha256": _sha256_file(eligibility_path),
        "checked_before_training": True,
    }
    _write_json(root / "bootstrap_eligibility.json", eligibility_payload)
    if not eligibility_valid:
        registry = _load_registry(root)
        registry["bootstrap_manifest_path"] = str(audit_manifest)
        registry["bootstrap_manifest_sha256"] = _sha256_file(audit_manifest)
        registry["profile_manifest_sha256"] = _sha256_file(profile_manifest)
        registry["terminal_reason"] = "insufficient_estimable_label_reliability"
        _write_json(root / "registry" / "jobs.json", registry)
        status = _stage_status(root)
        status.update(
            {
                "status": "completed_insufficient_label_reliability_evidence",
                "current_stage": "bootstrap",
                "terminal_reason": (
                    "现有标签不足以支持可靠性加权实验"
                ),
                "q3_predictions_generated": False,
                "q4_loader_materialized": False,
                "updated_at_utc": _utc_now(),
            }
        )
        _write_json(root / "label_reliability_stage_status.json", status)
        from scripts.rq3.news_first_vol_label_reliability_report import (
            render_label_reliability_report,
        )

        render_label_reliability_report(root)
        _refresh_exports(root)
        return root
    registry = _load_registry(root)
    registry["jobs"] = _materialize_job_configs(root, resolved, registry["jobs"])
    registry["bootstrap_manifest_path"] = str(audit_manifest)
    registry["bootstrap_manifest_sha256"] = _sha256_file(audit_manifest)
    registry["profile_manifest_sha256"] = _sha256_file(profile_manifest)
    _write_json(root / "registry" / "jobs.json", registry)
    status = _stage_status(root)
    status.update(
        {
            "status": "bootstrapped",
            "current_stage": STAGE_1,
            "bootstrap_manifest_path": str(audit_manifest),
            "bootstrap_manifest_sha256": _sha256_file(audit_manifest),
            "updated_at_utc": _utc_now(),
        }
    )
    status["stages"][STAGE_1]["status"] = "prepared"
    _write_json(root / "label_reliability_stage_status.json", status)
    _validate_registry(root, config_path)
    _refresh_exports(root)
    return root


def _load_registry(root: Path) -> dict[str, Any]:
    registry = _require_mapping(
        _read_json(root / "registry" / "jobs.json"), "label-reliability registry"
    )
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise LabelReliabilityExperimentError("Experiment kind mismatch")
    return registry


def _stage_status(root: Path) -> dict[str, Any]:
    return _require_mapping(
        _read_json(root / "label_reliability_stage_status.json"), "stage status"
    )


def _validate_root_lineage(
    root: Path, config_path: str | Path | None = None
) -> dict[str, Any]:
    resolved = _require_mapping(
        _load_yaml_mapping(root / "resolved_config.yaml", "resolved config").get(
            ROOT_KEY
        ),
        ROOT_KEY,
    )
    _validate_frozen_config(resolved)
    expected = (root / "registry" / "resolved_config.sha256").read_text(
        encoding="utf-8"
    ).strip()
    if _payload_sha256(resolved) != expected:
        raise LabelReliabilityExperimentError("Resolved config was modified")
    registry = _load_registry(root)
    if str(registry.get("resolved_config_sha256", "")) != expected:
        raise LabelReliabilityExperimentError(
            "Registry/resolved-config hash mismatch"
        )
    if config_path is not None:
        requested = _resolved_config(config_path)
        if _payload_sha256(requested) != expected:
            raise LabelReliabilityExperimentError(
                "Requested config differs from immutable experiment config"
            )
    for path_name, key in (("source_hashes.csv", "path"), ("code_hashes.csv", "path")):
        with (root / path_name).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            path = Path(row[key])
            if not path.is_file() or _sha256_file(path) != row["sha256"]:
                raise LabelReliabilityExperimentError(
                    f"Immutable lineage mismatch: {path}"
                )
    return resolved


def _validate_profiles(root: Path) -> None:
    import pandas as pd

    manifest_path = root / "reliability_profiles" / "reliability_profile_manifest.csv"
    manifest = pd.read_csv(manifest_path, low_memory=False)
    if len(manifest) != 32:
        raise LabelReliabilityExperimentError("Profile manifest row count drifted")
    for row in manifest.itertuples(index=False):
        path = Path(str(row.profile_path))
        if _sha256_file(path) != str(row.manifest_sha256):
            raise LabelReliabilityExperimentError(f"Profile file changed: {path}")
        frame = pd.read_csv(path, low_memory=False)
        if tuple(frame.columns) != PROFILE_FILE_COLUMNS:
            raise LabelReliabilityExperimentError(f"Profile schema changed: {path}")
        if canonical_profile_sha256(frame) != str(row.profile_sha256):
            raise LabelReliabilityExperimentError(f"Profile content hash changed: {path}")
        universe_values = set(
            frame["train_pair_universe_sha256"].fillna("").astype(str)
        )
        if universe_values != {str(row.train_pair_universe_sha256)}:
            raise LabelReliabilityExperimentError(
                f"Train-pair universe hash changed: {path}"
            )


def _validate_job(root: Path, job: Mapping[str, Any]) -> None:
    if str(job.get("job_id")) != _job_id(job):
        raise LabelReliabilityExperimentError("Job ID/axes mismatch")
    observed_hash = _job_spec_hash(job)
    if observed_hash != str(job.get("job_spec_sha256", "")):
        raise LabelReliabilityExperimentError(
            f"Job spec hash mismatch: {job.get('job_id')}"
        )
    if job.get("state") != "prepared":
        raise LabelReliabilityExperimentError(f"Job is not prepared: {job['job_id']}")
    config_path = Path(str(job["training_config_path"]))
    profile_path = Path(str(job["profile_path"]))
    if _sha256_file(config_path) != str(job["config_sha256"]):
        raise LabelReliabilityExperimentError("Training config hash mismatch")
    if _sha256_file(profile_path) != str(job["profile_manifest_sha256"]):
        raise LabelReliabilityExperimentError("Reliability manifest hash mismatch")
    if _sha256_file(Path(str(job["train_data_path"]))) != str(job["train_data_sha256"]):
        raise LabelReliabilityExperimentError("Fold train workbook hash mismatch")
    if _sha256_file(Path(str(job["validation_data_path"]))) != str(
        job["validation_data_sha256"]
    ):
        raise LabelReliabilityExperimentError("Fold validation workbook hash mismatch")
    payload = _load_yaml_mapping(config_path, f"training config {job['job_id']}")
    contracts = {
        "seed": int(job["seed"]),
        "news_first_train_end_utc": next(
            row["train_end_utc"] for row in FOLDS if row["fold_id"] == job["fold_id"]
        ),
        "news_first_validation_end_utc": next(
            row["validation_end_utc"]
            for row in FOLDS
            if row["fold_id"] == job["fold_id"]
        ),
        "news_first_materialize_test_loader": False,
        "news_first_label_reliability_fold_id": job["fold_id"],
        "news_first_label_reliability_manifest_sha256": job[
            "profile_manifest_sha256"
        ],
        "news_first_label_reliability_profile_sha256": job["profile_sha256"],
        "news_first_label_reliability_train_pair_universe_sha256": job[
            "train_pair_universe_sha256"
        ],
        "data_path": job["train_data_path"],
        "news_first_common_eval_data_path": job["validation_data_path"],
        "news_first_label_reliability_train_data_sha256": job["train_data_sha256"],
        "news_first_label_reliability_validation_data_sha256": job[
            "validation_data_sha256"
        ],
        "news_first_label_reliability_data_window_contract_sha256": job[
            "data_window_contract_sha256"
        ],
        "news_first_capacity_profile": FROZEN_PROFILE,
        "news_first_capacity_profile_sha256": job["capacity_profile_sha256"],
        "news_first_lr_profile": FROZEN_LR_PROFILE,
        "news_first_lr_profile_sha256": job["lr_profile_sha256"],
        "support_mask_mode": FROZEN_SUPPORT_MODE,
        "news_first_text_ablation_mode": FROZEN_TEXT_MODE,
        "generator_current_input_mode": FROZEN_CURRENT_INPUT_MODE,
    }
    for key, expected in contracts.items():
        if payload.get(key) != expected:
            raise LabelReliabilityExperimentError(
                f"Training config contract changed: {job['job_id']} {key}"
            )
    expected_window_hash = data_window_contract_sha256(
        fold_id=str(job["fold_id"]),
        train_end_utc=str(payload["news_first_train_end_utc"]),
        validation_end_utc=str(payload["news_first_validation_end_utc"]),
        train_workbook_sha256=str(job["train_data_sha256"]),
        validation_workbook_sha256=str(job["validation_data_sha256"]),
    )
    if expected_window_hash != str(job["data_window_contract_sha256"]):
        raise LabelReliabilityExperimentError(
            f"Data-window contract hash mismatch: {job['job_id']}"
        )


def _validate_registry(root: Path, config_path: str | Path | None = None) -> None:
    resolved = _validate_root_lineage(root, config_path)
    registry = _load_registry(root)
    jobs = [dict(row) for row in registry.get("jobs", [])]
    if len(jobs) not in {48, 96} or len({row["job_id"] for row in jobs}) != len(jobs):
        raise LabelReliabilityExperimentError("Conditional registry has invalid size/IDs")
    if len(jobs) == 48:
        expected = schedule_job_specs(
            _stage_specs(STAGE_1),
            gpu_ids=resolved["runtime"]["gpu_ids"],
            slots_per_gpu=int(resolved["runtime"]["slots_per_gpu"]),
        )
    else:
        winner = str(registry.get("selected_arm", ""))
        expected = build_full_job_matrix(
            winner,
            gpu_ids=resolved["runtime"]["gpu_ids"],
            slots_per_gpu=int(resolved["runtime"]["slots_per_gpu"]),
        )
    expected_by_id = {row["job_id"]: row for row in expected}
    if set(expected_by_id) != {row["job_id"] for row in jobs}:
        raise LabelReliabilityExperimentError("Registry Cartesian matrix changed")
    axis_fields = (
        "stage_id",
        "model_family",
        "tolerance_minutes",
        "arm_id",
        "fold_id",
        "seed",
        "wave",
        "gpu_id",
        "gpu_slot",
    )
    for job in jobs:
        expected_job = expected_by_id[job["job_id"]]
        for field in axis_fields:
            if job.get(field) != expected_job.get(field):
                raise LabelReliabilityExperimentError(
                    f"Registry schedule changed: {job['job_id']} {field}"
                )
        if _job_spec_hash(job) != str(job.get("job_spec_sha256", "")):
            raise LabelReliabilityExperimentError(
                f"Registry job hash changed: {job['job_id']}"
            )
    if len(jobs) >= 48 and all(row.get("state") == "prepared" for row in jobs):
        _validate_profiles(root)
        for job in jobs:
            _validate_job(root, job)


def _find_job(root: Path, job_id: str) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in _load_registry(root).get("jobs", [])
        if str(row.get("job_id")) == str(job_id)
    ]
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicate job: {job_id}")
    return matches[0]


def run_label_reliability_worker(
    experiment_root: str | Path,
    job_id: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    """Execute one hash-verified registered job."""

    root = _resolve_repo_path(experiment_root)
    _validate_registry(root)
    job = _find_job(root, job_id)
    _validate_job(root, job)
    status_path = _job_status_path(root, job_id)
    previous = _read_json(status_path)
    if previous.get("status") == "completed" and not dry_run:
        if resume and training._completed_job_is_valid(root, job, previous):
            return Path(str(previous["run_dir"]))
        raise RuntimeError(f"Job already completed: {job_id}")
    if previous.get("status") == "running" and training._pid_is_live(previous.get("pid")):
        raise RuntimeError(f"Job already running: {job_id}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if visible and visible != str(job["gpu_id"]):
        raise RuntimeError(
            f"Worker GPU mismatch: visible={visible}, assigned={job['gpu_id']}"
        )
    common = {
        "job_id": job_id,
        "attempt": int(previous.get("attempt", 0)) + 1,
        "config_sha256": job["config_sha256"],
        "job_spec_sha256": job["job_spec_sha256"],
        "profile_manifest_sha256": job["profile_manifest_sha256"],
        "profile_sha256": job["profile_sha256"],
        "gpu_id": job["gpu_id"],
        "pid": os.getpid(),
        "dry_run": bool(dry_run),
        "started_at_utc": _utc_now(),
    }
    _write_json(status_path, {**common, "status": "running"})
    try:
        run_dir, artifacts = training._execute_training_job(job, dry_run=bool(dry_run))
        _write_json(
            status_path,
            {
                **common,
                "status": "dry_run_passed" if dry_run else "completed",
                "run_dir": str(run_dir),
                "artifacts": artifacts,
                "exit_code": 0,
                "completed_at_utc": _utc_now(),
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
            },
        )
        raise


class _HostMemoryMonitor:
    def __init__(self, path: Path, interval_seconds: float) -> None:
        self.path = path
        self.interval = max(0.25, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("timestamp_utc,used_fraction\n", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            import psutil

            while not self._stop.is_set():
                value = float(psutil.virtual_memory().percent) / 100.0
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{_utc_now()},{value:.8f}\n")
                self._stop.wait(self.interval)
        except Exception:
            return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 2.0))


def build_worker_command(
    config_path: str | Path,
    root: Path,
    job: Mapping[str, Any],
    *,
    dry_run: bool,
    resume: bool,
) -> list[str]:
    resolved = _require_mapping(
        _load_yaml_mapping(root / "resolved_config.yaml", "resolved config")[ROOT_KEY],
        ROOT_KEY,
    )
    command = [
        str(resolved["runtime"]["python_executable"]),
        "-m",
        "scripts.rq3.news_first_vol_label_reliability",
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


def _run_wave(
    root: Path,
    config_path: str | Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
    resume: bool,
) -> None:
    if not jobs:
        return
    resolved = _require_mapping(
        _load_yaml_mapping(root / "resolved_config.yaml", "resolved config")[ROOT_KEY],
        ROOT_KEY,
    )
    wave = int(jobs[0]["wave"])
    monitor = training._ResourceMonitor(
        root / "resource_usage.csv",
        executable=str(resolved["runtime"].get("nvidia_smi_executable", "nvidia-smi")),
        interval_seconds=float(
            resolved["runtime"].get("resource_sample_interval_seconds", 2)
        ),
        wave=wave,
    )
    host_monitor = _HostMemoryMonitor(
        root / "host_memory_usage.csv",
        float(resolved["runtime"].get("resource_sample_interval_seconds", 2)),
    )
    processes: list[subprocess.Popen[Any]] = []
    handles = []
    try:
        monitor.start()
        host_monitor.start()
        for job in jobs:
            previous = _read_json(_job_status_path(root, job["job_id"]))
            if not dry_run and previous.get("status") == "completed":
                if resume and training._completed_job_is_valid(root, job, previous):
                    continue
                raise RuntimeError(f"Completed job requires --resume: {job['job_id']}")
            if dry_run and resume and previous.get("status") == "dry_run_passed":
                continue
            attempt = int(previous.get("attempt", 0)) + 1
            log_path = root / "logs" / f"{job['job_id']}.attempt_{attempt:02d}.log"
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
            env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT)))
            threads = str(int(resolved["runtime"].get("cpu_threads_per_job", 1)))
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[name] = threads
            process = subprocess.Popen(
                build_worker_command(
                    config_path, root, job, dry_run=dry_run, resume=resume
                ),
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
        pending = set(range(len(processes)))
        while pending:
            for index in tuple(pending):
                code = processes[index].poll()
                if code is None:
                    continue
                pending.remove(index)
                if code != 0:
                    training._terminate_processes(processes)
                    raise RuntimeError(
                        f"Wave {wave} worker exited {code}: {jobs[index]['job_id']}"
                    )
            if pending:
                time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        training._terminate_processes(processes)
        raise
    finally:
        monitor.stop()
        host_monitor.stop()
        for handle in handles:
            handle.close()


def _run_registered_stage(
    root: Path,
    config_path: str | Path,
    stage_id: str,
    *,
    dry_run: bool,
    resume: bool,
) -> None:
    jobs = [
        dict(row)
        for row in _load_registry(root)["jobs"]
        if row["stage_id"] == stage_id
    ]
    if len(jobs) != EXPECTED_STAGE_JOBS[stage_id]:
        raise LabelReliabilityExperimentError(f"{stage_id} registry incomplete")
    for wave in sorted({int(row["wave"]) for row in jobs}):
        _run_wave(
            root,
            config_path,
            [row for row in jobs if int(row["wave"]) == wave],
            dry_run=dry_run,
            resume=resume,
        )
    failures = []
    for job in jobs:
        status = _read_json(_job_status_path(root, job["job_id"]))
        expected = "dry_run_passed" if dry_run else "completed"
        if status.get("status") != expected:
            failures.append(f"{job['job_id']}={status.get('status')}")
    if failures:
        raise RuntimeError(f"Stage workers failed: {failures[:10]}")


def _preflight_limits(root: Path, resolved: Mapping[str, Any]) -> dict[str, Any]:
    import pandas as pd

    gpu_peak_mib = 0.0
    resource_path = root / "resource_usage.csv"
    if resource_path.is_file():
        resource = pd.read_csv(resource_path, low_memory=False)
        for field in ("memory_used_mib", "memory_used_mb", "memory_used"):
            if field in resource.columns:
                values = pd.to_numeric(resource[field], errors="coerce").dropna()
                if not values.empty:
                    gpu_peak_mib = float(values.max())
                    break
    host_peak = 0.0
    host_path = root / "host_memory_usage.csv"
    if host_path.is_file():
        values = pd.to_numeric(
            pd.read_csv(host_path)["used_fraction"], errors="coerce"
        ).dropna()
        if not values.empty:
            host_peak = float(values.max())
    cap = _experiment_config(resolved)["preflight"]
    gpu_limit_mib = float(cap["max_peak_gpu_memory_gib"]) * 1024.0
    host_limit = float(cap["max_host_ram_fraction"])
    passed = gpu_peak_mib < gpu_limit_mib and host_peak < host_limit
    return {
        "status": "pass" if passed else "fail",
        "slots_per_gpu": int(resolved["runtime"]["slots_per_gpu"]),
        "peak_gpu_memory_mib": gpu_peak_mib,
        "gpu_memory_limit_mib": gpu_limit_mib,
        "peak_host_ram_fraction": host_peak,
        "host_ram_limit_fraction": host_limit,
        "requires_new_12_slot_root": bool(
            not passed and int(resolved["runtime"]["slots_per_gpu"]) == 24
        ),
    }


def _append_conditional_jobs(
    root: Path, resolved: Mapping[str, Any], selected_arm: str
) -> None:
    registry = _load_registry(root)
    if len(registry["jobs"]) == EXPECTED_MAX_JOBS:
        return
    if len(registry["jobs"]) != 48:
        raise LabelReliabilityExperimentError("Conditional registry is partially appended")
    wave_offset = max(int(row["wave"]) for row in registry["jobs"]) + 1
    additions = []
    for stage in (STAGE_2, STAGE_3):
        scheduled = schedule_job_specs(
            _stage_specs(stage, selected_arm=selected_arm),
            gpu_ids=resolved["runtime"]["gpu_ids"],
            slots_per_gpu=int(resolved["runtime"]["slots_per_gpu"]),
            wave_offset=wave_offset,
        )
        additions.extend(scheduled)
        wave_offset = max(row["wave"] for row in scheduled) + 1
    prepared = _materialize_job_configs(root, resolved, additions)
    registry["jobs"].extend(prepared)
    registry["selected_arm"] = selected_arm
    registry["selection_sha256"] = _sha256_file(
        root / "label_reliability_selection.json"
    )
    _write_json(root / "registry" / "jobs.json", registry)


def _best_checkpoint_from_status(root: Path, job: Mapping[str, Any]) -> tuple[Path, str]:
    status = _read_json(_job_status_path(root, job["job_id"]))
    roles = {
        "regressor_best_learned"
        if job["model_family"] == "regression"
        else "generator_best_learned"
    }
    matches = [row for row in status.get("artifacts", []) if row.get("artifact_role") in roles]
    if len(matches) != 1:
        raise LabelReliabilityExperimentError(
            f"Missing unique best-learned checkpoint for {job['job_id']}"
        )
    path = Path(str(matches[0]["path"]))
    digest = str(matches[0]["sha256"])
    if not path.is_file() or _sha256_file(path) != digest:
        raise LabelReliabilityExperimentError("Best checkpoint hash mismatch")
    return path, digest


def evaluate_jobs_on_interval(
    root: Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    start_utc: str,
    end_utc: str,
    panel_name: str,
    evaluator: Callable[..., Any] | None = None,
):
    """Infer an explicit half-open interval and return overall pair metrics."""

    import pandas as pd
    from scripts.rq3.news_first_vol_comparison_analysis import (
        RunSpec,
        TrainedRunEvaluator,
        _load_panel_source,
        aggregate_pair_metrics,
        compute_sample_metrics,
    )

    start, end = pd.Timestamp(start_utc), pd.Timestamp(end_utc)
    if end > pd.Timestamp(Q3_END_UTC):
        raise LabelReliabilityExperimentError("Q4 prediction/evaluation is forbidden")
    workbook = _workbook_path(
        _require_mapping(
            _load_yaml_mapping(root / "resolved_config.yaml", "resolved")[ROOT_KEY],
            ROOT_KEY,
        ),
        5,
    )
    raw = pd.read_excel(workbook, sheet_name="gan_input_ready")
    times = pd.to_datetime(raw["effective_origin_utc"], errors="coerce", utc=True)
    if times.isna().any():
        raise LabelReliabilityExperimentError("Invalid evaluation timestamps")
    sliced = raw.loc[(times >= start) & (times < end)].copy()
    panel, _, _ = _load_panel_source(
        sliced,
        sheet_name="gan_input_ready",
        panel_name=panel_name,
        freeze_test_window=False,
        enforce_expected_counts=False,
        support_mask_mode=FROZEN_SUPPORT_MODE,
    )
    selected_times = pd.to_datetime(panel["effective_origin_utc"], errors="coerce", utc=True)
    if selected_times.isna().any() or not ((selected_times >= start) & (selected_times < end)).all():
        raise LabelReliabilityExperimentError("Evaluation interval guard failed")
    production = evaluator or TrainedRunEvaluator(mc_samples=16)
    parts = []
    for job in jobs:
        checkpoint, digest = _best_checkpoint_from_status(root, job)
        status = _read_json(_job_status_path(root, job["job_id"]))
        run = RunSpec(
            run_id=job["job_id"],
            run_dir=Path(status["run_dir"]),
            model=job["model_family"],
            tolerance_minutes=int(job["tolerance_minutes"]),
            seed=int(job["seed"]),
            checkpoint_path=checkpoint,
            text_ablation_mode=FROZEN_TEXT_MODE,
            support_mask_mode=FROZEN_SUPPORT_MODE,
            generator_current_input_mode=FROZEN_CURRENT_INPUT_MODE,
            manifest_path=Path(job["training_config_path"]),
            metadata={"checkpoint_sha256": digest},
        )
        predictions = production(run, panel_name, panel.copy())
        samples, exclusions, _ = compute_sample_metrics(
            run, panel_name, panel, predictions, evaluate_embedded_atm_skew=False
        )
        if not exclusions.empty:
            raise LabelReliabilityExperimentError(
                f"Prediction exclusions for {job['job_id']}: "
                f"{exclusions['exclusion_code'].value_counts().to_dict()}"
            )
        pairs = aggregate_pair_metrics(samples)
        pairs = pairs[
            pairs["stratum_type"].astype(str).eq("overall")
            & pairs["stratum_value"].astype(str).eq("all")
        ].copy()
        pairs.insert(0, "arm_id", job["arm_id"])
        pairs.insert(1, "fold_id", job["fold_id"])
        parts.append(pairs)
    return pd.concat(parts, ignore_index=True)


def _run_stage1_selection(root: Path, resolved: Mapping[str, Any]) -> dict[str, Any]:
    import pandas as pd
    from scripts.rq3.news_first_vol_label_reliability_analysis import (
        build_arm_comparisons,
        select_reliability_winner,
    )

    jobs = [row for row in _load_registry(root)["jobs"] if row["stage_id"] == STAGE_1]
    parts = []
    for fold in FOLDS:
        fold_jobs = [row for row in jobs if row["fold_id"] == fold["fold_id"]]
        parts.append(
            evaluate_jobs_on_interval(
                root,
                fold_jobs,
                start_utc=fold["validation_start_utc"],
                end_utc=fold["validation_end_utc"],
                panel_name=f"inner_validation_{fold['fold_id']}",
            )
        )
    metrics = pd.concat(parts, ignore_index=True)
    metrics_path = root / "analysis" / "fold_pair_metrics.csv.gz"
    metrics.to_csv(metrics_path, index=False, compression="gzip")
    selection_config = _experiment_config(resolved)["selection"]
    comparisons = build_arm_comparisons(
        metrics,
        iterations=int(selection_config["bootstrap_iterations"]),
        seed=int(selection_config["bootstrap_seed"]),
    )
    profile_manifest = pd.read_csv(
        root / "reliability_profiles" / "reliability_profile_manifest.csv"
    )
    retention_valid = profile_manifest["retention_gate_valid"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes"}
    )
    invalid = set(
        profile_manifest.loc[
            (profile_manifest["tolerance_minutes"] == 5)
            & ~retention_valid,
            "arm_id",
        ].astype(str)
    )
    comparisons.loc[
        comparisons["candidate_arm"].astype(str).isin(invalid), "passes_gate"
    ] = False
    comparisons["retention_gate_valid_all_folds"] = ~comparisons[
        "candidate_arm"
    ].astype(str).isin(invalid)
    comparisons_path = root / "analysis" / "label_reliability_comparisons.csv"
    comparisons.to_csv(comparisons_path, index=False)
    selection = select_reliability_winner(comparisons)
    selection["retention_invalid_arms"] = sorted(invalid)
    selection["selection_payload_sha256"] = _payload_sha256(
        {key: value for key, value in selection.items() if key != "selection_payload_sha256"}
    )
    path = root / "label_reliability_selection.json"
    encoded = json.dumps(selection, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise LabelReliabilityExperimentError("Immutable selection changed")
    path.write_text(encoded, encoding="utf-8")
    return selection


def _freeze_q3_checkpoints(root: Path, selected_arm: str) -> Path:
    import pandas as pd
    from scripts.rq3.news_first_vol_label_reliability_analysis import (
        q3_checkpoint_allowlist,
    )

    enriched = []
    for job in _load_registry(root)["jobs"]:
        row = dict(job)
        if row["fold_id"] == "F4" and row["arm_id"] in {"A", selected_arm}:
            path, digest = _best_checkpoint_from_status(root, row)
            row["best_learned_checkpoint_path"] = str(path)
            row["best_learned_checkpoint_sha256"] = digest
        enriched.append(row)
    allowlist = q3_checkpoint_allowlist(enriched, selected_arm)
    path = root / "q3_checkpoint_manifest.csv"
    pd.DataFrame(allowlist).to_csv(path, index=False)
    return path


def postprocess_label_reliability_experiment(
    config_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Evaluate only frozen A/winner F4 checkpoints on Q3 and render the report."""

    import pandas as pd

    root = _resolve_repo_path(output_dir)
    _validate_root_lineage(root, config_path)
    selection_path = root / "label_reliability_selection.json"
    if not selection_path.is_file():
        raise LabelReliabilityExperimentError("Selection must be frozen before Q3")
    selection = _read_json(selection_path)
    from scripts.rq3.news_first_vol_label_reliability_analysis import (
        validate_selection_payload,
    )

    selection = validate_selection_payload(selection)
    if selection.get("status") != "selected":
        from scripts.rq3.news_first_vol_label_reliability_report import (
            render_label_reliability_report,
        )

        render_label_reliability_report(root)
        return root
    winner = str(selection["selected_arm"])
    checkpoint_manifest = _freeze_q3_checkpoints(root, winner)
    frozen = pd.read_csv(checkpoint_manifest)
    jobs_by_id = {row["job_id"]: row for row in _load_registry(root)["jobs"]}
    jobs = [jobs_by_id[value] for value in frozen["job_id"].astype(str)]
    q3 = evaluate_jobs_on_interval(
        root,
        jobs,
        start_utc=Q3_START_UTC,
        end_utc=Q3_END_UTC,
        panel_name="exploratory_repeated_q3",
    )
    q3_path = root / "analysis" / "q3_pair_metrics.csv.gz"
    q3.to_csv(q3_path, index=False, compression="gzip")
    status = _stage_status(root)
    status.update(
        {
            "status": "completed_q3_only",
            "current_stage": "postprocess",
            "q3_predictions_generated": True,
            "q3_used_for_selection": False,
            "q4_loader_materialized": False,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
            "updated_at_utc": _utc_now(),
        }
    )
    _write_json(root / "label_reliability_stage_status.json", status)
    from scripts.rq3.news_first_vol_label_reliability_report import (
        render_label_reliability_report,
    )

    render_label_reliability_report(root)
    _refresh_exports(root)
    return root


def launch_label_reliability_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> Path:
    root = run_label_reliability_bootstrap_action(
        config_path, output_dir, resume=bool(resume)
    )
    resolved = _validate_root_lineage(root, config_path)
    bootstrap_status = _stage_status(root)
    if bootstrap_status.get("status") == "completed_insufficient_label_reliability_evidence":
        return root
    eligibility = _read_json(root / "bootstrap_eligibility.json")
    if eligibility.get("status") != "pass" or not bool(
        eligibility.get("checked_before_training", False)
    ):
        raise LabelReliabilityExperimentError(
            "Bootstrap estimability gate must pass before any model task"
        )
    if dry_run:
        # Stage one already exercises 24/12 slots per GPU and every arm/fold
        # profile.  Validate all 96 conditional cells/configs in memory without
        # persisting a pretend winner or mutating the formal registry.
        build_full_job_matrix(
            str(_experiment_config(resolved).get("dry_run_assumed_winner", "C")),
            gpu_ids=resolved["runtime"]["gpu_ids"],
            slots_per_gpu=int(resolved["runtime"]["slots_per_gpu"]),
        )
        _run_registered_stage(root, config_path, STAGE_1, dry_run=True, resume=resume)
        preflight = _preflight_limits(root, resolved)
        _write_json(root / "preflight.json", preflight)
        if preflight["status"] != "pass":
            status = _stage_status(root)
            status["status"] = (
                "preflight_requires_new_12_slot_root"
                if preflight["requires_new_12_slot_root"]
                else "preflight_failed"
            )
            status["updated_at_utc"] = _utc_now()
            _write_json(root / "label_reliability_stage_status.json", status)
            raise RuntimeError(
                "Resource preflight failed; 24-slot roots must be replaced by a "
                "new immutable 12-slot root"
            )
        status = _stage_status(root)
        status.update({"status": "dry_run_passed", "updated_at_utc": _utc_now()})
        _write_json(root / "label_reliability_stage_status.json", status)
        _refresh_exports(root)
        return root
    status = _stage_status(root)
    if status.get("status") == "completed_q3_only":
        if not resume:
            raise RuntimeError("Completed experiment requires --resume")
        return root
    try:
        _run_registered_stage(root, config_path, STAGE_1, dry_run=False, resume=resume)
        selection = _run_stage1_selection(root, resolved)
        if selection["status"] == "terminal_no_winner":
            status = _stage_status(root)
            status.update(
                {
                    "status": "completed_no_reliable_label_policy",
                    "current_stage": STAGE_1,
                    "winner": None,
                    "terminal_reason": "terminal_no_winner",
                    "q3_predictions_generated": False,
                    "q4_loader_materialized": False,
                    "updated_at_utc": _utc_now(),
                }
            )
            _write_json(root / "label_reliability_stage_status.json", status)
            from scripts.rq3.news_first_vol_label_reliability_report import (
                render_label_reliability_report,
            )

            render_label_reliability_report(root)
            _refresh_exports(root)
            return root
        winner = str(selection["selected_arm"])
        _append_conditional_jobs(root, resolved, winner)
        _validate_registry(root, config_path)
        for stage in (STAGE_2, STAGE_3):
            _run_registered_stage(root, config_path, stage, dry_run=False, resume=resume)
        return postprocess_label_reliability_experiment(config_path, root)
    except BaseException as exc:
        status = _stage_status(root)
        status.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at_utc": _utc_now(),
            }
        )
        _write_json(root / "label_reliability_stage_status.json", status)
        _refresh_exports(root)
        raise


def _refresh_exports(root: Path) -> Path:
    registry_path = root / "registry" / "jobs.json"
    if not registry_path.is_file():
        return root
    rows = []
    artifacts = []
    for job in _load_registry(root).get("jobs", []):
        status_path = _job_status_path(root, job["job_id"])
        status = _read_json(status_path) if status_path.is_file() else {"status": "missing"}
        rows.append({**job, **status})
        artifacts.extend(
            {"job_id": job["job_id"], **artifact}
            for artifact in status.get("artifacts", [])
        )
    if rows:
        import pandas as pd

        pd.DataFrame(rows).to_csv(root / "task_registry.csv", index=False)
    stable = [path for path in root.rglob("*") if path.is_file() and path.name != "output_hashes.csv"]
    seen = {str(row.get("path", "")) for row in artifacts}
    for path in stable:
        if str(path) not in seen:
            artifacts.append(
                {
                    "job_id": "",
                    "artifact_role": f"experiment:{path.relative_to(root).as_posix()}",
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    if artifacts:
        _write_csv(
            root / "output_hashes.csv",
            artifacts,
            ("job_id", "artifact_role", "path", "size_bytes", "sha256"),
        )
    return root


def run_news_first_vol_label_reliability(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    action: str = "prepare",
    job_id: str = "",
    resume: bool = False,
    reuse: bool = False,
    worker_dry_run: bool = False,
) -> Path:
    """Dispatch one of the six public experiment actions."""

    normalized = str(action).strip().lower()
    if normalized == "prepare":
        return prepare_label_reliability_experiment(
            config_path, output_dir, reuse=bool(reuse or resume)
        )
    if normalized == "bootstrap":
        return run_label_reliability_bootstrap_action(
            config_path, output_dir, resume=resume
        )
    if normalized == "dry-run":
        return launch_label_reliability_experiment(
            config_path, output_dir, resume=resume, dry_run=True
        )
    if normalized == "worker":
        if not job_id:
            raise ValueError("worker requires --job-id")
        return run_label_reliability_worker(
            output_dir, job_id, dry_run=worker_dry_run, resume=resume
        )
    if normalized == "launch":
        return launch_label_reliability_experiment(
            config_path, output_dir, resume=resume, dry_run=False
        )
    if normalized == "postprocess":
        return postprocess_label_reliability_experiment(config_path, output_dir)
    raise ValueError(
        "action must be prepare/bootstrap/dry-run/worker/launch/postprocess"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="prepare",
        choices=("prepare", "bootstrap", "dry-run", "worker", "launch", "postprocess"),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--worker-dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_news_first_vol_label_reliability(
        args.config,
        args.output_dir,
        action=args.action,
        job_id=args.job_id,
        resume=args.resume,
        reuse=args.reuse,
        worker_dry_run=args.worker_dry_run,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "FOLDS",
    "build_full_job_matrix",
    "canonical_profile_sha256",
    "launch_label_reliability_experiment",
    "materialize_reliability_profiles",
    "postprocess_label_reliability_experiment",
    "prepare_label_reliability_experiment",
    "run_label_reliability_bootstrap_action",
    "run_label_reliability_worker",
    "run_news_first_vol_label_reliability",
    "schedule_job_specs",
]
