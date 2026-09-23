"""Independent evaluation and inference for news-first vol-surface runs.

This module deliberately does not depend on the training loop.  An
orchestrator supplies an ``evaluator`` callable that turns one run checkpoint
and one fixed evaluation panel into keyed surface predictions.  Everything
after that boundary is deterministic, dataframe-based, and unit-testable.

The statistical unit is a market ``pair_id``.  Article-level results are
retained for audit, but model summaries balance pairs equally and the formal
WGAN tolerance comparisons resample complete CME sessions.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import yaml


if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

import scripts._path_setup  # noqa: E402,F401
from film_wgan.support import SUPPORT_METHOD, parse_raw_surface_params, raw_support_mask  # noqa: E402
from wgan_option.utils.text_ablation import (  # noqa: E402
    CURRENT_ONLY,
    REAL_TEXT,
    TEXT_SHUFFLE,
    normalize_text_ablation_mode,
    text_information_path,
    transform_embedding_matrix,
)


DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260819
DEFAULT_BASE_TOLERANCE_MINUTES = 5
DEFAULT_FOCAL_TOLERANCES = (10, 15, 30)
SHORT_ATM_Q_MIN = 0.98
SHORT_ATM_Q_MAX = 1.02
SHORT_ATM_MAX_BUSINESS_DAYS = 60.0
ARBITRAGE_VIOLATION_TOLERANCE = 1.0e-8
MAE_NUMERICAL_TIE_TOLERANCE = 1.0e-8
DEFAULT_BOOTSTRAP_METRICS = (
    "model_mae",
    "mae_gap",
    "skill",
    "win",
    "short_atm_model_mae",
    "short_atm_mae_gap",
    "atm_model_abs_error",
    "atm_error_gap",
    "skew_model_abs_error",
    "skew_error_gap",
    "calendar_violation_rate_gap",
    "butterfly_violation_rate_gap",
)
TEST_START_UTC = "2023-10-01T00:00:00Z"
TEST_END_UTC = "2024-01-01T00:00:00Z"
EXPECTED_PANEL_COUNTS = {
    "core": {"rows": 200, "pairs": 170, "sessions": 49, "tolerance_minutes": 5},
    "broad": {"rows": 361, "pairs": 263, "sessions": 55, "tolerance_minutes": 30},
}
EXPECTED_RAW_JOINT_PANEL_COUNTS = {
    "core": {"rows": 152, "pairs": 130, "sessions": 45, "tolerance_minutes": 5},
    "broad": {"rows": 245, "pairs": 182, "sessions": 51, "tolerance_minutes": 30},
}

SAMPLE_METRIC_COLUMNS = (
    "run_id",
    "model",
    "text_ablation_mode",
    "support_mask_mode",
    "generator_current_input_mode",
    "seed",
    "tolerance_minutes",
    "panel",
    "sample_id",
    "news_row_id",
    "pair_id",
    "session_id",
    "first_included_tolerance_minutes",
    "origin_shift_minutes",
    "atm_ab_flag",
    "sample_weight",
    "supported_cell_count",
    "supported_cell_fraction",
    "support_method",
    "support_grid_fingerprint",
    "model_mae",
    "model_rmse",
    "model_max_abs",
    "persistence_mae",
    "persistence_rmse",
    "persistence_max_abs",
    "mae_gap",
    "mae_tie",
    "skill",
    "win",
    "surface_diagnostic_status",
    "short_atm_metric_status",
    "calendar_metric_status",
    "butterfly_metric_status",
    "short_atm_cell_count",
    "short_atm_model_mae",
    "short_atm_persistence_mae",
    "short_atm_mae_gap",
    "short_atm_tie",
    "short_atm_skill",
    "short_atm_win",
    "predicted_calendar_violation_count",
    "predicted_calendar_constraint_count",
    "predicted_calendar_violation_rate",
    "target_calendar_violation_count",
    "target_calendar_constraint_count",
    "target_calendar_violation_rate",
    "current_calendar_violation_count",
    "current_calendar_constraint_count",
    "current_calendar_violation_rate",
    "calendar_violation_rate_gap",
    "predicted_butterfly_violation_count",
    "predicted_butterfly_constraint_count",
    "predicted_butterfly_violation_rate",
    "target_butterfly_violation_count",
    "target_butterfly_constraint_count",
    "target_butterfly_violation_rate",
    "current_butterfly_violation_count",
    "current_butterfly_constraint_count",
    "current_butterfly_violation_rate",
    "butterfly_violation_rate_gap",
    "atm_metric_status",
    "atm_maturity_count",
    "atm_model_abs_error",
    "atm_persistence_abs_error",
    "atm_error_gap",
    "skew_metric_status",
    "skew_maturity_count",
    "skew_model_abs_error",
    "skew_persistence_abs_error",
    "skew_error_gap",
)

PAIR_METRIC_COLUMNS = (
    "run_id",
    "model",
    "text_ablation_mode",
    "support_mask_mode",
    "generator_current_input_mode",
    "seed",
    "tolerance_minutes",
    "panel",
    "stratum_type",
    "stratum_value",
    "pair_id",
    "session_id",
    "first_included_tolerance_minutes",
    "origin_shift_minutes",
    "atm_ab_flag",
    "article_count",
    "article_weight_sum",
    "model_mae",
    "model_rmse",
    "model_max_abs",
    "persistence_mae",
    "persistence_rmse",
    "persistence_max_abs",
    "mae_gap",
    "mae_tie",
    "skill",
    "win",
    "short_atm_cell_count",
    "short_atm_model_mae",
    "short_atm_persistence_mae",
    "short_atm_mae_gap",
    "short_atm_tie",
    "short_atm_skill",
    "short_atm_win",
    "predicted_calendar_violation_count",
    "predicted_calendar_constraint_count",
    "predicted_calendar_violation_rate",
    "target_calendar_violation_count",
    "target_calendar_constraint_count",
    "target_calendar_violation_rate",
    "current_calendar_violation_count",
    "current_calendar_constraint_count",
    "current_calendar_violation_rate",
    "calendar_violation_rate_gap",
    "predicted_butterfly_violation_count",
    "predicted_butterfly_constraint_count",
    "predicted_butterfly_violation_rate",
    "target_butterfly_violation_count",
    "target_butterfly_constraint_count",
    "target_butterfly_violation_rate",
    "current_butterfly_violation_count",
    "current_butterfly_constraint_count",
    "current_butterfly_violation_rate",
    "butterfly_violation_rate_gap",
    "atm_maturity_count",
    "atm_model_abs_error",
    "atm_persistence_abs_error",
    "atm_error_gap",
    "skew_maturity_count",
    "skew_model_abs_error",
    "skew_persistence_abs_error",
    "skew_error_gap",
)

MODEL_COMPARISON_COLUMNS = (
    "model",
    "text_ablation_mode",
    "support_mask_mode",
    "generator_current_input_mode",
    "tolerance_minutes",
    "panel",
    "stratum_type",
    "stratum_value",
    "run_count",
    "seed_count",
    "pair_count",
    "session_count",
    "mae",
    "rmse",
    "persistence_mae",
    "gap",
    "tie_rate",
    "skill",
    "win",
    "short_atm_mae",
    "short_atm_persistence_mae",
    "short_atm_gap",
    "short_atm_tie_rate",
    "short_atm_skill",
    "short_atm_win",
    "predicted_calendar_violation_count_mean",
    "predicted_calendar_violation_rate",
    "target_calendar_violation_count_mean",
    "target_calendar_violation_rate",
    "current_calendar_violation_count_mean",
    "current_calendar_violation_rate",
    "calendar_violation_rate_gap",
    "predicted_butterfly_violation_count_mean",
    "predicted_butterfly_violation_rate",
    "target_butterfly_violation_count_mean",
    "target_butterfly_violation_rate",
    "current_butterfly_violation_count_mean",
    "current_butterfly_violation_rate",
    "butterfly_violation_rate_gap",
    "atm_pair_count",
    "atm_maturity_count",
    "atm_mae",
    "atm_gap",
    "skew_pair_count",
    "skew_maturity_count",
    "skew_mae",
    "skew_gap",
)

BOOTSTRAP_COMPARISON_COLUMNS = (
    "model",
    "text_ablation_mode",
    "focal_tolerance_minutes",
    "base_tolerance_minutes",
    "panel",
    "metric",
    "stratum_type",
    "stratum_value",
    "difference_direction",
    "negative_means_focal_better",
    "pair_count",
    "session_count",
    "mean_diff",
    "ci_95_lower",
    "ci_95_upper",
    "p_two_sided",
    "ci_low",
    "ci_high",
    "p_value",
    "p_holm",
    "holm_family",
    "bootstrap_iterations",
    "bootstrap_seed",
    "numerical_tie_tolerance",
    "status",
)

MATURITY_METRIC_COLUMNS = (
    "run_id",
    "model",
    "text_ablation_mode",
    "support_mask_mode",
    "generator_current_input_mode",
    "seed",
    "tolerance_minutes",
    "panel",
    "sample_id",
    "news_row_id",
    "pair_id",
    "session_id",
    "slice_pair_id",
    "maturity_date",
    "underlying_contract_id",
    "origin_business_days",
    "target_business_days",
    "maturity_weight",
    "pair_atm_quality",
    "atm_metric_status",
    "predicted_atm_iv",
    "current_atm_iv",
    "target_atm_iv",
    "atm_model_abs_error",
    "atm_persistence_abs_error",
    "atm_error_gap",
    "skew_quality",
    "skew_metric_status",
    "predicted_skew",
    "current_skew",
    "target_skew",
    "skew_model_abs_error",
    "skew_persistence_abs_error",
    "skew_error_gap",
)

TEXT_ABLATION_COMPARISON_COLUMNS = (
    "model",
    "tolerance_minutes",
    "panel",
    "metric",
    "control_mode",
    "difference_direction",
    "negative_means_real_text_better",
    "pair_count",
    "session_count",
    "mean_diff",
    "ci_95_lower",
    "ci_95_upper",
    "p_two_sided",
    "p_holm",
    "holm_family",
    "bootstrap_iterations",
    "bootstrap_seed",
    "numerical_tie_tolerance",
    "status",
)

CHECKPOINT_SUMMARY_COLUMNS = (
    "run_id",
    "model",
    "text_ablation_mode",
    "support_mask_mode",
    "generator_current_input_mode",
    "seed",
    "tolerance_minutes",
    "monitor_metric",
    "best_epoch",
    "best_metric",
    "metadata_path",
    "status",
)


class ComparisonAnalysisError(ValueError):
    """Raised when an input cannot support a trustworthy comparison."""


class GridInterpolationError(ComparisonAnalysisError):
    """Raised when a requested grid value would require extrapolation."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = str(code)
        self.detail = str(detail)


@dataclass(frozen=True)
class RunSpec:
    """Resolved metadata for one trained run."""

    run_id: str
    run_dir: Path
    model: str
    tolerance_minutes: int
    seed: int
    checkpoint_path: Path
    text_ablation_mode: str = REAL_TEXT
    support_mask_mode: str = "none"
    generator_current_input_mode: str = "full_current"
    manifest_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        for key in ("run_dir", "checkpoint_path", "manifest_path"):
            value = record[key]
            record[key] = "" if value is None else str(value)
        record["metadata"] = dict(self.metadata)
        return record


class PredictionEvaluator(Protocol):
    """Boundary between model-specific inference and pure analysis.

    The returned dataframe must contain ``sample_id`` or ``news_row_id`` and a
    ``predicted_surface_flat`` column.  It may contain only a subset of panel
    rows; missing or explicitly failed predictions are written to the general
    exclusion audit.
    """

    def __call__(
        self,
        run: RunSpec,
        panel_name: str,
        panel: pd.DataFrame,
    ) -> pd.DataFrame: ...


def _stable_offset(label: str) -> int:
    digest = hashlib.sha256(str(label).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 1_000_000_007


def _mae_gap_is_tie(gap: float) -> bool:
    return bool(
        math.isfinite(float(gap)) and abs(float(gap)) <= MAE_NUMERICAL_TIE_TOLERANCE
    )


def _mae_gap_is_win(gap: float) -> bool:
    return bool(math.isfinite(float(gap)) and float(gap) < -MAE_NUMERICAL_TIE_TOLERANCE)


def _zero_numerical_ties(values: pd.Series) -> pd.Series:
    """Canonicalize comparison differences inside the declared IV precision."""

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.mask(numeric.abs().le(MAE_NUMERICAL_TIE_TOLERANCE), 0.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle)
        else:
            payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ComparisonAnalysisError(f"Manifest must be a mapping: {path}")
    return dict(payload)


def _recursive_first(payload: Any, keys: Sequence[str]) -> Any:
    wanted = {str(key) for key in keys}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key) in wanted and value not in (None, ""):
                return value
        for value in payload.values():
            found = _recursive_first(value, keys)
            if found not in (None, ""):
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _recursive_first(value, keys)
            if found not in (None, ""):
                return found
    return None


def _resolve_relative_path(
    value: str | Path, *, run_dir: Path, manifest_path: Path | None
) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    roots = [run_dir]
    if manifest_path is not None:
        roots.insert(0, manifest_path.parent)
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (run_dir / candidate).resolve()


def discover_run_spec(run_dir: str | Path) -> RunSpec:
    """Resolve a run manifest and checkpoint without assuming one trainer layout."""

    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {root}")

    manifest_candidates = [
        root / "run_manifest.json",
        root / "manifest.json",
        root / "metrics" / "run_manifest.json",
        root / "metrics" / "resolved_config.json",
        root / "metrics" / "resolved_config.yaml",
        root / "metrics" / "training_resolved_config.yaml",
    ]
    manifest_path = next((path for path in manifest_candidates if path.is_file()), None)
    payload: dict[str, Any] = _read_mapping(manifest_path) if manifest_path else {}

    run_id = str(_recursive_first(payload, ("run_id", "name")) or root.name)
    model_value = _recursive_first(
        payload, ("model", "model_family", "architecture", "trainer")
    )

    tolerance_value = _recursive_first(
        payload,
        ("tolerance_minutes", "news_tolerance_minutes", "alignment_tolerance_minutes"),
    )
    if tolerance_value is None:
        search_text = "|".join(
            [root.name, root.parent.name, root.parent.parent.name]
            + [
                str(_recursive_first(payload, ("data_path", "output_root")) or ""),
            ]
        )
        match = re.search(
            r"(?:tolerance[_-]?)?(05|5|10|15|30)m(?:in)?\b",
            search_text,
            flags=re.I,
        )
        if match:
            tolerance_value = match.group(1)
    if tolerance_value is None:
        raise ComparisonAnalysisError(
            f"Cannot resolve tolerance_minutes from manifest or run directory: {root}"
        )
    tolerance_minutes = int(tolerance_value)

    seed_value = _recursive_first(payload, ("seed", "random_seed", "training_seed"))
    if seed_value is None:
        match = re.search(r"seed[_-]?(\d+)", root.name, flags=re.I)
        seed_value = match.group(1) if match else 0
    seed = int(seed_value)
    text_mode = normalize_text_ablation_mode(
        _recursive_first(
            payload, ("news_first_text_ablation_mode", "text_ablation_mode")
        )
        or REAL_TEXT
    )
    support_mask_mode = (
        str(_recursive_first(payload, ("support_mask_mode",)) or "none").strip().lower()
    )
    if support_mask_mode not in {"none", "raw_joint"}:
        raise ComparisonAnalysisError(
            f"Unsupported support_mask_mode={support_mask_mode!r} for {root}"
        )
    generator_current_input_mode = (
        str(
            _recursive_first(payload, ("generator_current_input_mode",))
            or "full_current"
        )
        .strip()
        .lower()
    )
    if generator_current_input_mode not in {"full_current", "current_support_masked"}:
        raise ComparisonAnalysisError(
            "Unsupported generator_current_input_mode="
            f"{generator_current_input_mode!r} for {root}"
        )
    if (
        generator_current_input_mode == "current_support_masked"
        and support_mask_mode != "raw_joint"
    ):
        raise ComparisonAnalysisError(
            "current_support_masked runs require support_mask_mode='raw_joint'"
        )

    checkpoint_value = _recursive_first(
        payload,
        (
            "checkpoint_path",
            "generator_checkpoint_path",
            "best_checkpoint_path",
            "selected_checkpoint_path",
        ),
    )
    checkpoint_path: Path | None = None
    if checkpoint_value:
        checkpoint_path = _resolve_relative_path(
            checkpoint_value,
            run_dir=root,
            manifest_path=manifest_path,
        )
    if checkpoint_path is None or not checkpoint_path.is_file():
        checkpoint_candidates = [
            root / "checkpoints" / "generator_best.pt",
            root / "checkpoints" / "generator.pt",
            root / "generator_best.pt",
            root / "generator.pt",
            root / "checkpoints" / "model_best.pt",
            root / "model_best.pt",
            root / "checkpoints" / "vol_regressor_best.pt",
            root / "checkpoints" / "vol_regressor.pt",
            root / "vol_regressor_best.pt",
            root / "vol_regressor.pt",
        ]
        checkpoint_path = next(
            (path for path in checkpoint_candidates if path.is_file()), checkpoint_path
        )
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Cannot resolve a checkpoint under run directory: {root}"
        )

    model_context = "|".join(
        [str(model_value or ""), root.name, root.parent.name, checkpoint_path.name]
    ).lower()
    if "regress" in model_context or checkpoint_path.name.startswith("vol_regressor"):
        model = "regression"
    elif "wgan" in model_context or checkpoint_path.name.startswith("generator"):
        model = "wgan"
    elif model_value:
        model = str(model_value).strip()
    else:
        raise ComparisonAnalysisError(f"Cannot resolve model family for run: {root}")

    return RunSpec(
        run_id=run_id,
        run_dir=root,
        model=model,
        tolerance_minutes=tolerance_minutes,
        seed=seed,
        checkpoint_path=checkpoint_path.resolve(),
        text_ablation_mode=text_mode,
        support_mask_mode=support_mask_mode,
        generator_current_input_mode=generator_current_input_mode,
        manifest_path=manifest_path.resolve() if manifest_path else None,
        metadata=payload,
    )


def _parse_vector(value: Any, *, label: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        result = value.astype(np.float64, copy=False).reshape(-1)
    elif isinstance(value, (list, tuple)):
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ComparisonAnalysisError(f"{label} is empty")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
        result = np.asarray(parsed, dtype=np.float64).reshape(-1)
    else:
        raise ComparisonAnalysisError(
            f"{label} has unsupported type {type(value).__name__}"
        )
    if result.size == 0:
        raise ComparisonAnalysisError(f"{label} is empty")
    if not np.isfinite(result).all():
        raise ComparisonAnalysisError(f"{label} contains non-finite values")
    return result


def _parse_surface_shape(value: Any, *, cell_count: int) -> tuple[int, int]:
    has_value = value is not None
    if isinstance(value, str):
        has_value = bool(value.strip())
    elif isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        has_value = False
    if has_value:
        parsed = _parse_vector(value, label="surface_shape")
        if parsed.size != 2:
            raise ComparisonAnalysisError("surface_shape must contain two dimensions")
        shape = (int(parsed[0]), int(parsed[1]))
        if shape[0] * shape[1] != int(cell_count):
            raise ComparisonAnalysisError(
                f"surface_shape {shape} does not match {cell_count} cells"
            )
        return shape
    side = int(round(math.sqrt(int(cell_count))))
    if side * side != int(cell_count):
        raise ComparisonAnalysisError(
            "surface_shape is required when surface cell count is not square"
        )
    return side, side


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _first_numeric(
    row: Mapping[str, Any], columns: Sequence[str]
) -> tuple[float | None, str | None]:
    for column in columns:
        if column in row:
            value = _coerce_optional_float(row[column])
            if value is not None:
                return value, column
    return None, None


def _coerce_flag(value: Any) -> str:
    if (
        value is None
        or value is pd.NA
        or (isinstance(value, float) and math.isnan(value))
    ):
        return "unknown"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "atm_ab", "a/b", "ab"}:
            return "atm_ab"
        if normalized in {"0", "false", "no", "n", "not_atm_ab", "non_atm_ab"}:
            return "not_atm_ab"
        return normalized or "unknown"
    return "atm_ab" if bool(value) else "not_atm_ab"


def _coerce_strict_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ComparisonAnalysisError(f"{label} must be an explicit boolean, got {value!r}")


def _panel_atm_flag(row: Mapping[str, Any]) -> str:
    for column in ("atm_ab_flag", "atm_ab_training_eligible", "has_atm_ab_maturity"):
        if column in row:
            return _coerce_flag(row[column])
    return "unknown"


def _surface_axes(
    row: Mapping[str, Any],
    *,
    cell_count: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if "strike_grid" not in row or "maturity_days_grid" not in row:
        raise GridInterpolationError(
            "missing_grid_fields",
            "strike_grid and maturity_days_grid are required for ATM/skew interpolation",
        )
    strike_grid = _parse_vector(row["strike_grid"], label="strike_grid")
    maturity_grid = _parse_vector(row["maturity_days_grid"], label="maturity_days_grid")
    shape = _parse_surface_shape(row.get("surface_shape"), cell_count=cell_count)
    if shape != (len(maturity_grid), len(strike_grid)):
        raise GridInterpolationError(
            "grid_shape_mismatch",
            f"surface shape {shape} does not match maturity/strike grids "
            f"({len(maturity_grid)}, {len(strike_grid)})",
        )
    if len(np.unique(strike_grid)) != len(strike_grid) or len(
        np.unique(maturity_grid)
    ) != len(maturity_grid):
        raise GridInterpolationError(
            "duplicate_grid_coordinate", "surface grids must be unique"
        )
    return strike_grid, maturity_grid, shape


def _support_grid_fingerprint(
    strike_grid: np.ndarray,
    maturity_grid: np.ndarray,
) -> str:
    payload = {
        "schema_version": 1,
        "support_method": SUPPORT_METHOD,
        "strike_grid": [float(value) for value in strike_grid],
        "maturity_days_grid": [int(round(float(value))) for value in maturity_grid],
        "surface_shape": [int(len(maturity_grid)), int(len(strike_grid))],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _joint_support_mask(
    row: Mapping[str, Any],
    *,
    strike_grid: np.ndarray,
    maturity_grid: np.ndarray,
    support_mask_mode: str,
) -> tuple[np.ndarray, str, str]:
    """Derive the exact current/target raw-support intersection used in loss."""

    shape = (len(maturity_grid), len(strike_grid))
    mode = str(support_mask_mode or "none").strip().lower()
    fingerprint = _support_grid_fingerprint(strike_grid, maturity_grid)
    if mode == "none":
        return np.ones(shape, dtype=bool), "none", fingerprint
    if mode != "raw_joint":
        raise ComparisonAnalysisError(f"Unsupported support_mask_mode={mode!r}")
    missing = [
        column
        for column in ("current_surface_param_json", "target_surface_param_json")
        if column not in row or str(row.get(column, "")).strip() == ""
    ]
    if missing:
        raise ComparisonAnalysisError(
            f"raw_joint evaluation requires workbook parameter columns: {missing}"
        )
    try:
        current = raw_support_mask(
            parse_raw_surface_params(row["current_surface_param_json"]),
            strike_grid=strike_grid,
            maturity_days_grid=maturity_grid,
        )
        target = raw_support_mask(
            parse_raw_surface_params(row["target_surface_param_json"]),
            strike_grid=strike_grid,
            maturity_days_grid=maturity_grid,
        )
    except (TypeError, ValueError) as exc:
        raise ComparisonAnalysisError(
            f"Cannot derive raw joint support: {exc}"
        ) from exc
    joint = np.logical_and(current, target)
    if joint.shape != shape or not bool(joint.any()):
        raise ComparisonAnalysisError(
            f"raw_joint support must contain at least one of {int(np.prod(shape))} cells"
        )
    return joint, SUPPORT_METHOD, fingerprint


def _current_support_mask(
    row: Mapping[str, Any],
    *,
    strike_grid: np.ndarray,
    maturity_grid: np.ndarray,
) -> np.ndarray:
    """Derive conditioning support from current parameters only."""

    shape = (len(maturity_grid), len(strike_grid))
    value = row.get("current_surface_param_json")
    if value is None or str(value).strip() == "":
        raise ComparisonAnalysisError(
            "current_support_masked evaluation requires current_surface_param_json"
        )
    try:
        mask = raw_support_mask(
            parse_raw_surface_params(value),
            strike_grid=strike_grid,
            maturity_days_grid=maturity_grid,
        )
    except (TypeError, ValueError) as exc:
        raise ComparisonAnalysisError(
            f"Cannot derive raw current support: {exc}"
        ) from exc
    if tuple(mask.shape) != shape or not bool(mask.any()):
        raise ComparisonAnalysisError(
            f"raw current support must contain at least one of {int(np.prod(shape))} cells"
        )
    return np.asarray(mask, dtype=bool)


def _current_support_mask_fingerprint(
    mask: np.ndarray,
    *,
    strike_grid: np.ndarray,
    maturity_grid: np.ndarray,
) -> str:
    """Fingerprint the current-only conditioning mask and its fixed grid."""

    boolean_mask = np.asarray(mask, dtype=bool)
    payload = {
        "schema_version": 1,
        "support_method": SUPPORT_METHOD,
        "role": "current",
        "support_grid_fingerprint": _support_grid_fingerprint(
            strike_grid, maturity_grid
        ),
        "surface_shape": [int(value) for value in boolean_mask.shape],
        "supported_cells": [int(value) for value in boolean_mask.reshape(-1)],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    """Standard-normal CDF without adding a SciPy runtime dependency."""

    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    result = np.fromiter(
        (0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in flattened),
        dtype=np.float64,
        count=flattened.size,
    )
    return result.reshape(np.asarray(values).shape)


def surface_arbitrage_violations(
    surface: np.ndarray | Sequence[float],
    strike_grid: np.ndarray | Sequence[float],
    maturity_days_grid: np.ndarray | Sequence[float],
    *,
    tolerance: float = ARBITRAGE_VIOLATION_TOLERANCE,
    support_mask: np.ndarray | Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Compute the trainer-compatible discrete call-price violation metrics.

    The definition matches the active WGAN and regression trainers: calendar
    monotonicity is tested on total variance ``sigma**2 * tau`` and butterfly
    convexity on Black-76 relative call prices, with ``tau=maturity_days/365``.
    Counts and denominators accompany rates so the output remains auditable.
    """

    strikes = np.asarray(strike_grid, dtype=np.float64).reshape(-1)
    maturities = np.asarray(maturity_days_grid, dtype=np.float64).reshape(-1)
    values = np.asarray(surface, dtype=np.float64)
    expected_shape = (len(maturities), len(strikes))
    if values.shape != expected_shape:
        raise ComparisonAnalysisError(
            f"surface shape {values.shape} does not match grid shape "
            f"({len(maturities)}, {len(strikes)})"
        )
    if (
        not np.isfinite(values).all()
        or not np.isfinite(strikes).all()
        or not np.isfinite(maturities).all()
        or bool(np.any(strikes <= 0.0))
    ):
        raise ComparisonAnalysisError(
            "surface/grid must be finite and strike moneyness must be positive"
        )
    if float(tolerance) < 0.0:
        raise ComparisonAnalysisError(
            "arbitrage violation tolerance must be non-negative"
        )
    strike_order = np.argsort(strikes)
    maturity_order = np.argsort(maturities)
    strikes = strikes[strike_order]
    maturities = maturities[maturity_order]
    values = values[np.ix_(maturity_order, strike_order)]
    mask_was_provided = support_mask is not None
    if support_mask is None:
        supported = np.ones_like(values, dtype=bool)
    else:
        supported = np.asarray(support_mask, dtype=bool)
        if supported.shape != expected_shape:
            raise ComparisonAnalysisError(
                f"support mask shape {supported.shape} does not match {expected_shape}"
            )
        supported = supported[np.ix_(maturity_order, strike_order)]
    if len(np.unique(strikes)) != len(strikes) or len(np.unique(maturities)) != len(
        maturities
    ):
        raise ComparisonAnalysisError("surface grids must contain unique coordinates")
    if len(strikes) >= 3 and not np.allclose(
        np.diff(strikes),
        np.diff(strikes)[0],
        rtol=1.0e-8,
        atol=1.0e-10,
    ):
        raise ComparisonAnalysisError(
            "trainer-compatible butterfly diagnostics require an equally spaced strike grid"
        )

    sigma = np.clip(values, 1.0e-4, None)
    strike = strikes.reshape(1, -1)
    tau = np.clip(maturities.reshape(-1, 1) / 365.0, 1.0 / 365.0, None)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(1.0 / strike) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    calls = _normal_cdf(d1) - strike * _normal_cdf(d2)

    total_variance = sigma**2 * tau
    calendar_residual = total_variance[:-1, :] - total_variance[1:, :]
    calendar_supported = supported[:-1, :] & supported[1:, :]
    calendar_count = int(
        np.sum((calendar_residual > float(tolerance)) & calendar_supported)
    )
    calendar_constraints = int(calendar_supported.sum())
    second_diff = calls[:, 2:] - 2.0 * calls[:, 1:-1] + calls[:, :-2]
    butterfly_supported = supported[:, :-2] & supported[:, 1:-1] & supported[:, 2:]
    butterfly_count = int(
        np.sum((second_diff < -float(tolerance)) & butterfly_supported)
    )
    butterfly_constraints = int(butterfly_supported.sum())
    return {
        "calendar_violation_count": calendar_count,
        "calendar_constraint_count": calendar_constraints,
        "calendar_violation_rate": (
            float(calendar_count / calendar_constraints)
            if calendar_constraints
            else (float("nan") if mask_was_provided else 0.0)
        ),
        "butterfly_violation_count": butterfly_count,
        "butterfly_constraint_count": butterfly_constraints,
        "butterfly_violation_rate": (
            float(butterfly_count / butterfly_constraints)
            if butterfly_constraints
            else (float("nan") if mask_was_provided else 0.0)
        ),
    }


def surface_diagnostic_metrics(
    current_surface: np.ndarray | Sequence[float],
    target_surface: np.ndarray | Sequence[float],
    predicted_surface: np.ndarray | Sequence[float],
    strike_grid: np.ndarray | Sequence[float],
    maturity_days_grid: np.ndarray | Sequence[float],
    *,
    support_mask: np.ndarray | Sequence[float] | None = None,
) -> dict[str, float | int | str]:
    """Return short-ATM forecast errors and structural violation diagnostics."""

    strikes = np.asarray(strike_grid, dtype=np.float64).reshape(-1)
    maturities = np.asarray(maturity_days_grid, dtype=np.float64).reshape(-1)
    expected_shape = (len(maturities), len(strikes))
    current = np.asarray(current_surface, dtype=np.float64).reshape(expected_shape)
    target = np.asarray(target_surface, dtype=np.float64).reshape(expected_shape)
    predicted = np.asarray(predicted_surface, dtype=np.float64).reshape(expected_shape)
    if not (
        np.isfinite(current).all()
        and np.isfinite(target).all()
        and np.isfinite(predicted).all()
    ):
        raise ComparisonAnalysisError("surface diagnostics require finite surfaces")
    supported = (
        np.ones(expected_shape, dtype=bool)
        if support_mask is None
        else np.asarray(support_mask, dtype=bool)
    )
    if supported.shape != expected_shape or not bool(supported.any()):
        raise ComparisonAnalysisError(
            f"surface diagnostics require a non-empty {expected_shape} support mask"
        )

    short_mask = (
        (maturities.reshape(-1, 1) <= SHORT_ATM_MAX_BUSINESS_DAYS + 1.0e-6)
        & (strikes.reshape(1, -1) >= SHORT_ATM_Q_MIN - 1.0e-6)
        & (strikes.reshape(1, -1) <= SHORT_ATM_Q_MAX + 1.0e-6)
    )
    short_mask = short_mask & supported
    cell_count = int(short_mask.sum())
    if cell_count > 0:
        model_short = float(np.mean(np.abs(predicted[short_mask] - target[short_mask])))
        persistence_short = float(
            np.mean(np.abs(current[short_mask] - target[short_mask]))
        )
        short_skill = (
            float(1.0 - model_short / persistence_short)
            if persistence_short > 0.0
            else float("nan")
        )
        short_status = "ok"
    else:
        model_short = float("nan")
        persistence_short = float("nan")
        short_skill = float("nan")
        short_status = "no_supported_cells"
    short_gap = float(model_short - persistence_short)
    output: dict[str, float | int | str] = {
        "surface_diagnostic_status": (
            "ok"
            if cell_count > 0 or support_mask is not None
            else "short_atm_unavailable"
        ),
        "short_atm_metric_status": short_status,
        "short_atm_cell_count": cell_count,
        "short_atm_model_mae": model_short,
        "short_atm_persistence_mae": persistence_short,
        "short_atm_mae_gap": short_gap,
        "short_atm_tie": (
            float(_mae_gap_is_tie(short_gap)) if cell_count > 0 else float("nan")
        ),
        "short_atm_skill": short_skill,
        "short_atm_win": (
            float(_mae_gap_is_win(short_gap)) if cell_count > 0 else float("nan")
        ),
    }
    for label, surface in (
        ("predicted", predicted),
        ("target", target),
        ("current", current),
    ):
        diagnostics = surface_arbitrage_violations(
            surface,
            strikes,
            maturities,
            tolerance=ARBITRAGE_VIOLATION_TOLERANCE,
            support_mask=supported if support_mask is not None else None,
        )
        for metric, value in diagnostics.items():
            output[f"{label}_{metric}"] = value
    output["calendar_violation_rate_gap"] = float(
        output["predicted_calendar_violation_rate"]
        - output["target_calendar_violation_rate"]
    )
    output["butterfly_violation_rate_gap"] = float(
        output["predicted_butterfly_violation_rate"]
        - output["target_butterfly_violation_rate"]
    )
    output["calendar_metric_status"] = (
        "ok"
        if int(output["predicted_calendar_constraint_count"]) > 0
        else "no_supported_constraints"
    )
    output["butterfly_metric_status"] = (
        "ok"
        if int(output["predicted_butterfly_constraint_count"]) > 0
        else "no_supported_constraints"
    )
    return output


def _observed_short_atm_cell_counts(
    sample_metrics: pd.DataFrame,
) -> dict[str, list[int]]:
    """Return the evaluated short-ATM grid widths by panel.

    The news-first experiments intentionally keep a fixed 16x16 tensor shape,
    but the number of cells inside the pre-registered short-ATM band changes
    with the configured strike and maturity axes.  Deriving this value from
    evaluated rows keeps both the legacy wide grid and the narrow grid honest.
    """

    required = {"panel", "short_atm_cell_count"}
    if sample_metrics.empty or not required.issubset(sample_metrics.columns):
        return {}
    output: dict[str, list[int]] = {}
    for panel, group in sample_metrics.groupby("panel", sort=True, dropna=False):
        counts = (
            pd.to_numeric(group["short_atm_cell_count"], errors="coerce")
            .dropna()
            .astype(int)
        )
        output[str(panel)] = sorted(set(counts.tolist()))
    return output


def _format_short_atm_cell_counts(counts_by_panel: Mapping[str, Sequence[int]]) -> str:
    if not counts_by_panel:
        return "observed grid cell count unavailable"
    parts = []
    for panel, counts in sorted(counts_by_panel.items()):
        values = sorted({int(value) for value in counts})
        rendered = str(values[0]) if len(values) == 1 else json.dumps(values)
        parts.append(f"{panel}={rendered}")
    return "observed grid cell count(s): " + ", ".join(parts)


def strict_grid_interpolate(
    surface: np.ndarray | Sequence[float],
    strike_grid: np.ndarray | Sequence[float],
    maturity_days_grid: np.ndarray | Sequence[float],
    *,
    moneyness: float,
    maturity_days: float,
    support_mask: np.ndarray | Sequence[float] | None = None,
) -> float:
    """Bilinearly interpolate inside the observed grid, never extrapolate."""

    strikes = np.asarray(strike_grid, dtype=np.float64).reshape(-1)
    maturities = np.asarray(maturity_days_grid, dtype=np.float64).reshape(-1)
    values = np.asarray(surface, dtype=np.float64)
    if values.shape != (len(maturities), len(strikes)):
        raise GridInterpolationError(
            "grid_shape_mismatch",
            f"surface shape {values.shape} does not match ({len(maturities)}, {len(strikes)})",
        )
    if (
        not np.isfinite(values).all()
        or not np.isfinite(strikes).all()
        or not np.isfinite(maturities).all()
    ):
        raise GridInterpolationError(
            "non_finite_grid", "surface and grid values must be finite"
        )
    if len(np.unique(strikes)) != len(strikes) or len(np.unique(maturities)) != len(
        maturities
    ):
        raise GridInterpolationError(
            "duplicate_grid_coordinate", "surface grids must be unique"
        )

    strike_order = np.argsort(strikes)
    maturity_order = np.argsort(maturities)
    strikes = strikes[strike_order]
    maturities = maturities[maturity_order]
    values = values[np.ix_(maturity_order, strike_order)]
    supported = None
    if support_mask is not None:
        supported = np.asarray(support_mask, dtype=bool)
        if supported.shape != (len(maturity_order), len(strike_order)):
            raise GridInterpolationError(
                "support_mask_shape_mismatch",
                "support mask shape does not match interpolation grid",
            )
        supported = supported[np.ix_(maturity_order, strike_order)]
    x = float(moneyness)
    y = float(maturity_days)
    if not np.isfinite(x) or not np.isfinite(y):
        raise GridInterpolationError(
            "non_finite_query", "interpolation coordinates must be finite"
        )
    tolerance = 1e-12
    if x < strikes[0] - tolerance or x > strikes[-1] + tolerance:
        raise GridInterpolationError(
            "moneyness_outside_grid",
            f"moneyness={x} is outside [{strikes[0]}, {strikes[-1]}]",
        )
    if y < maturities[0] - tolerance or y > maturities[-1] + tolerance:
        raise GridInterpolationError(
            "maturity_outside_grid",
            f"maturity_days={y} is outside [{maturities[0]}, {maturities[-1]}]",
        )
    x = float(np.clip(x, strikes[0], strikes[-1]))
    y = float(np.clip(y, maturities[0], maturities[-1]))

    def bracket(axis: np.ndarray, value: float) -> tuple[int, int, float]:
        right = int(np.searchsorted(axis, value, side="left"))
        if right < len(axis) and math.isclose(
            float(axis[right]), value, abs_tol=tolerance
        ):
            return right, right, 0.0
        if right == 0 or right == len(axis):
            index = 0 if right == 0 else len(axis) - 1
            return index, index, 0.0
        left = right - 1
        weight = (value - float(axis[left])) / (float(axis[right]) - float(axis[left]))
        return left, right, float(weight)

    x0, x1, wx = bracket(strikes, x)
    y0, y1, wy = bracket(maturities, y)
    if supported is not None:
        required = {(y0, x0)}
        if x1 != x0 and wx > tolerance:
            required.add((y0, x1))
        if y1 != y0 and wy > tolerance:
            required.add((y1, x0))
            if x1 != x0 and wx > tolerance:
                required.add((y1, x1))
        if any(not bool(supported[row, column]) for row, column in required):
            raise GridInterpolationError(
                "unsupported_interpolation_cells",
                "ATM/skew interpolation requires grid cells outside raw joint support",
            )
    if x0 == x1 and y0 == y1:
        return float(values[y0, x0])
    if y0 == y1:
        return float((1.0 - wx) * values[y0, x0] + wx * values[y0, x1])
    if x0 == x1:
        return float((1.0 - wy) * values[y0, x0] + wy * values[y1, x0])
    lower = (1.0 - wx) * values[y0, x0] + wx * values[y0, x1]
    upper = (1.0 - wx) * values[y1, x0] + wx * values[y1, x1]
    return float((1.0 - wy) * lower + wy * upper)


def _surface_skew(
    surface: np.ndarray,
    strike_grid: np.ndarray,
    maturity_grid: np.ndarray,
    *,
    left_q: float,
    right_q: float,
    maturity_days: float,
    support_mask: np.ndarray | None = None,
) -> float:
    if left_q <= 0.0 or right_q <= 0.0 or not left_q < right_q:
        raise GridInterpolationError(
            "invalid_skew_coordinates",
            "skew requires finite positive left_q < right_q",
        )
    left_iv = strict_grid_interpolate(
        surface,
        strike_grid,
        maturity_grid,
        moneyness=left_q,
        maturity_days=maturity_days,
        support_mask=support_mask,
    )
    right_iv = strict_grid_interpolate(
        surface,
        strike_grid,
        maturity_grid,
        moneyness=right_q,
        maturity_days=maturity_days,
        support_mask=support_mask,
    )
    span = math.log(right_q) - math.log(left_q)
    if span <= 0.0:
        raise GridInterpolationError(
            "invalid_skew_log_span", "skew log-moneyness span must be positive"
        )
    return float((right_iv - left_iv) / span)


def _metric_exclusion(
    base: Mapping[str, Any],
    *,
    metric_family: str,
    code: str,
    detail: str,
) -> dict[str, Any]:
    return {
        key: base.get(key, "")
        for key in (
            "run_id",
            "model",
            "text_ablation_mode",
            "support_mask_mode",
            "generator_current_input_mode",
            "seed",
            "tolerance_minutes",
            "panel",
            "sample_id",
            "news_row_id",
            "pair_id",
            "session_id",
        )
    } | {
        "metric_family": str(metric_family),
        "exclusion_code": str(code),
        "detail": str(detail),
    }


def _evaluate_atm_skew(
    row: Mapping[str, Any],
    *,
    current_surface: np.ndarray,
    target_surface: np.ndarray,
    predicted_surface: np.ndarray,
    support_mask: np.ndarray | None,
    base: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics: dict[str, Any] = {
        "atm_metric_status": "excluded",
        "atm_model_abs_error": np.nan,
        "atm_persistence_abs_error": np.nan,
        "atm_error_gap": np.nan,
        "skew_metric_status": "excluded",
        "skew_model_abs_error": np.nan,
        "skew_persistence_abs_error": np.nan,
        "skew_error_gap": np.nan,
    }
    exclusions: list[dict[str, Any]] = []
    try:
        strike_grid, maturity_grid, shape = _surface_axes(
            row, cell_count=target_surface.size
        )
    except (ComparisonAnalysisError, GridInterpolationError) as exc:
        code = getattr(exc, "code", "invalid_grid")
        for family in ("atm", "skew"):
            exclusions.append(
                _metric_exclusion(
                    base, metric_family=family, code=code, detail=str(exc)
                )
            )
        return metrics, exclusions

    current = current_surface.reshape(shape)
    target = target_surface.reshape(shape)
    predicted = predicted_surface.reshape(shape)
    maturity, maturity_field = _first_numeric(
        row,
        ("target_business_days", "atm_maturity_days", "business_days", "maturity_days"),
    )
    atm_q, atm_q_field = _first_numeric(row, ("target_atm_q", "atm_q", "approx_atm_q"))
    if maturity is None or atm_q is None:
        missing = []
        if maturity is None:
            missing.append("target_business_days/atm_maturity_days")
        if atm_q is None:
            missing.append("target_atm_q/atm_q")
        exclusions.append(
            _metric_exclusion(
                base,
                metric_family="atm",
                code="missing_atm_coordinates",
                detail="Missing "
                + ", ".join(missing)
                + "; q=1 is deliberately not assumed.",
            )
        )
    else:
        try:
            predicted_atm = strict_grid_interpolate(
                predicted,
                strike_grid,
                maturity_grid,
                moneyness=atm_q,
                maturity_days=maturity,
                support_mask=support_mask,
            )
            current_atm = strict_grid_interpolate(
                current,
                strike_grid,
                maturity_grid,
                moneyness=atm_q,
                maturity_days=maturity,
                support_mask=support_mask,
            )
            observed_atm, _ = _first_numeric(row, ("target_atm_iv", "atm_target_iv"))
            target_atm = (
                float(observed_atm)
                if observed_atm is not None
                else strict_grid_interpolate(
                    target,
                    strike_grid,
                    maturity_grid,
                    moneyness=atm_q,
                    maturity_days=maturity,
                    support_mask=support_mask,
                )
            )
            model_error = abs(predicted_atm - target_atm)
            persistence_error = abs(current_atm - target_atm)
            metrics.update(
                {
                    "atm_metric_status": "ok",
                    "atm_model_abs_error": float(model_error),
                    "atm_persistence_abs_error": float(persistence_error),
                    "atm_error_gap": float(model_error - persistence_error),
                    "atm_query_q": float(atm_q),
                    "atm_query_maturity_days": float(maturity),
                    "atm_query_q_field": str(atm_q_field),
                    "atm_query_maturity_field": str(maturity_field),
                }
            )
        except GridInterpolationError as exc:
            exclusions.append(
                _metric_exclusion(
                    base, metric_family="atm", code=exc.code, detail=exc.detail
                )
            )

    left_q, _ = _first_numeric(row, ("target_skew_left_q", "skew_left_q"))
    right_q, _ = _first_numeric(row, ("target_skew_right_q", "skew_right_q"))
    if left_q is None:
        left_k, _ = _first_numeric(row, ("target_skew_left_k", "skew_left_k"))
        left_q = math.exp(left_k) if left_k is not None else None
    if right_q is None:
        right_k, _ = _first_numeric(row, ("target_skew_right_k", "skew_right_k"))
        right_q = math.exp(right_k) if right_k is not None else None
    if maturity is None or left_q is None or right_q is None:
        exclusions.append(
            _metric_exclusion(
                base,
                metric_family="skew",
                code="missing_skew_coordinates",
                detail=(
                    "Skew requires maturity and explicit left/right q (or k) coordinates; "
                    "single-wing or implicit grid endpoints are not used."
                ),
            )
        )
    else:
        try:
            predicted_skew = _surface_skew(
                predicted,
                strike_grid,
                maturity_grid,
                left_q=left_q,
                right_q=right_q,
                maturity_days=maturity,
                support_mask=support_mask,
            )
            current_skew = _surface_skew(
                current,
                strike_grid,
                maturity_grid,
                left_q=left_q,
                right_q=right_q,
                maturity_days=maturity,
                support_mask=support_mask,
            )
            observed_skew, _ = _first_numeric(
                row,
                ("target_atm_iv_skew_secant", "target_skew_secant"),
            )
            target_skew = (
                float(observed_skew)
                if observed_skew is not None
                else _surface_skew(
                    target,
                    strike_grid,
                    maturity_grid,
                    left_q=left_q,
                    right_q=right_q,
                    maturity_days=maturity,
                    support_mask=support_mask,
                )
            )
            model_error = abs(predicted_skew - target_skew)
            persistence_error = abs(current_skew - target_skew)
            metrics.update(
                {
                    "skew_metric_status": "ok",
                    "skew_model_abs_error": float(model_error),
                    "skew_persistence_abs_error": float(persistence_error),
                    "skew_error_gap": float(model_error - persistence_error),
                    "skew_left_q": float(left_q),
                    "skew_right_q": float(right_q),
                    "skew_query_maturity_days": float(maturity),
                }
            )
        except GridInterpolationError as exc:
            exclusions.append(
                _metric_exclusion(
                    base, metric_family="skew", code=exc.code, detail=exc.detail
                )
            )
    return metrics, exclusions


def _sample_join_key(frame: pd.DataFrame) -> str:
    if "sample_id" in frame.columns and frame["sample_id"].notna().all():
        return "sample_id"
    if "news_row_id" in frame.columns and frame["news_row_id"].notna().all():
        return "news_row_id"
    raise ComparisonAnalysisError(
        "Panel/predictions require complete sample_id or news_row_id"
    )


def validate_panel(panel: pd.DataFrame, *, panel_name: str) -> pd.DataFrame:
    required = {
        "pair_id",
        "session_id",
        "current_surface_flat",
        "target_surface_flat",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ComparisonAnalysisError(
            f"Panel {panel_name!r} is missing columns: {missing}"
        )
    output = panel.copy()
    key = _sample_join_key(output)
    if output[key].duplicated().any():
        duplicates = (
            output.loc[output[key].duplicated(keep=False), key]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise ComparisonAnalysisError(
            f"Panel {panel_name!r} has duplicate {key} values: {duplicates}"
        )
    if (
        output["pair_id"].isna().any()
        or output["pair_id"].astype(str).str.strip().eq("").any()
    ):
        raise ComparisonAnalysisError(
            f"Panel {panel_name!r} has missing pair_id values"
        )
    if (
        output["session_id"].isna().any()
        or output["session_id"].astype(str).str.strip().eq("").any()
    ):
        raise ComparisonAnalysisError(
            f"Panel {panel_name!r} has missing session_id; CME-session inference cannot be audited"
        )
    sessions_per_pair = output.groupby("pair_id")["session_id"].nunique(dropna=False)
    if bool((sessions_per_pair > 1).any()):
        bad = sessions_per_pair[sessions_per_pair > 1].index.astype(str).tolist()[:5]
        raise ComparisonAnalysisError(
            f"Panel {panel_name!r} maps a pair_id to multiple CME sessions: {bad}"
        )
    return output.reset_index(drop=True)


def _prediction_surface_column(predictions: pd.DataFrame) -> str:
    for column in (
        "predicted_surface_flat",
        "generated_surface_flat",
        "prediction_surface_flat",
    ):
        if column in predictions.columns:
            return column
    raise ComparisonAnalysisError(
        "Evaluator output requires predicted_surface_flat (or generated/prediction_surface_flat alias)"
    )


def compute_sample_metrics(
    run: RunSpec,
    panel_name: str,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    evaluate_embedded_atm_skew: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join keyed predictions and compute article-grain metrics and exclusions."""

    panel = validate_panel(panel, panel_name=panel_name)
    if not isinstance(predictions, pd.DataFrame):
        raise ComparisonAnalysisError("Evaluator must return a pandas DataFrame")
    panel_key = _sample_join_key(panel)
    prediction_key = _sample_join_key(predictions)
    if panel_key != prediction_key:
        if {panel_key, prediction_key} == {"sample_id", "news_row_id"}:
            raise ComparisonAnalysisError(
                f"Panel joins on {panel_key}, evaluator output joins on {prediction_key}; return the same stable key"
            )
    key = panel_key
    if predictions[key].duplicated().any():
        duplicates = (
            predictions.loc[predictions[key].duplicated(keep=False), key]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise ComparisonAnalysisError(
            f"Evaluator returned duplicate {key} values: {duplicates}"
        )
    surface_column = _prediction_surface_column(predictions)

    prediction_columns = [key, surface_column]
    for optional in ("prediction_status", "prediction_exclusion_reason"):
        if optional in predictions.columns:
            prediction_columns.append(optional)
    renamed_surface = "_predicted_surface_flat"
    keyed_predictions = predictions[prediction_columns].rename(
        columns={surface_column: renamed_surface}
    )
    merged = panel.merge(
        keyed_predictions, on=key, how="left", validate="one_to_one", indicator=True
    )

    sample_rows: list[dict[str, Any]] = []
    general_exclusions: list[dict[str, Any]] = []
    metric_exclusions: list[dict[str, Any]] = []
    for raw in merged.to_dict(orient="records"):
        base = {
            "run_id": run.run_id,
            "model": run.model,
            "text_ablation_mode": run.text_ablation_mode,
            "support_mask_mode": run.support_mask_mode,
            "generator_current_input_mode": run.generator_current_input_mode,
            "seed": int(run.seed),
            "tolerance_minutes": int(run.tolerance_minutes),
            "panel": str(panel_name),
            "sample_id": str(raw.get("sample_id", "")),
            "news_row_id": raw.get("news_row_id", ""),
            "pair_id": str(raw.get("pair_id", "")),
            "session_id": str(raw.get("session_id", "")),
        }
        prediction_status = str(raw.get("prediction_status", "ok")).strip().lower()
        if raw.get("_merge") != "both":
            general_exclusions.append(
                base
                | {
                    "exclusion_code": "missing_prediction",
                    "detail": "Evaluator did not return a prediction for the panel row.",
                }
            )
            continue
        if prediction_status not in {"", "ok", "success", "usable"}:
            general_exclusions.append(
                base
                | {
                    "exclusion_code": "evaluator_rejected_prediction",
                    "detail": str(
                        raw.get("prediction_exclusion_reason", prediction_status)
                    ),
                }
            )
            continue
        try:
            current = _parse_vector(
                raw["current_surface_flat"], label="current_surface_flat"
            )
            target = _parse_vector(
                raw["target_surface_flat"], label="target_surface_flat"
            )
            predicted = _parse_vector(
                raw[renamed_surface], label="predicted_surface_flat"
            )
            if not (len(current) == len(target) == len(predicted)):
                raise ComparisonAnalysisError(
                    f"surface length mismatch current={len(current)}, target={len(target)}, predicted={len(predicted)}"
                )
            strike_grid, maturity_grid, shape = _surface_axes(
                raw, cell_count=len(target)
            )
            support_mask, support_method, support_fingerprint = _joint_support_mask(
                raw,
                strike_grid=strike_grid,
                maturity_grid=maturity_grid,
                support_mask_mode=run.support_mask_mode,
            )
        except (ComparisonAnalysisError, ValueError, SyntaxError) as exc:
            general_exclusions.append(
                base | {"exclusion_code": "invalid_surface", "detail": str(exc)}
            )
            continue

        flat_support = support_mask.reshape(-1)
        model_diff = (predicted - target)[flat_support]
        persistence_diff = (current - target)[flat_support]
        model_mae = float(np.mean(np.abs(model_diff)))
        persistence_mae = float(np.mean(np.abs(persistence_diff)))
        mae_gap = float(model_mae - persistence_mae)
        skill = (
            float(1.0 - model_mae / persistence_mae)
            if persistence_mae > 0.0
            else float("nan")
        )
        weight = _coerce_optional_float(raw.get("sample_weight"))
        if weight is None or weight <= 0.0:
            weight = 1.0
        row = base | {
            "first_included_tolerance_minutes": _coerce_optional_float(
                raw.get("first_included_tolerance_minutes")
            ),
            "origin_shift_minutes": _coerce_optional_float(
                raw.get("origin_shift_minutes")
            ),
            "atm_ab_flag": _panel_atm_flag(raw),
            "sample_weight": float(weight),
            "supported_cell_count": int(flat_support.sum()),
            "supported_cell_fraction": float(flat_support.mean()),
            "support_method": support_method,
            "support_grid_fingerprint": support_fingerprint,
            "model_mae": model_mae,
            "model_rmse": float(np.sqrt(np.mean(model_diff**2))),
            "model_max_abs": float(np.max(np.abs(model_diff))),
            "persistence_mae": persistence_mae,
            "persistence_rmse": float(np.sqrt(np.mean(persistence_diff**2))),
            "persistence_max_abs": float(np.max(np.abs(persistence_diff))),
            "mae_gap": mae_gap,
            "mae_tie": float(_mae_gap_is_tie(mae_gap)),
            "skill": skill,
            "win": float(_mae_gap_is_win(mae_gap)),
        }
        try:
            row.update(
                surface_diagnostic_metrics(
                    current.reshape(shape),
                    target.reshape(shape),
                    predicted.reshape(shape),
                    strike_grid,
                    maturity_grid,
                    support_mask=(
                        support_mask if run.support_mask_mode == "raw_joint" else None
                    ),
                )
            )
            if run.support_mask_mode == "raw_joint":
                for family, status_column in (
                    ("short_atm", "short_atm_metric_status"),
                    ("calendar", "calendar_metric_status"),
                    ("butterfly", "butterfly_metric_status"),
                ):
                    if str(row.get(status_column, "")) != "ok":
                        metric_exclusions.append(
                            _metric_exclusion(
                                base,
                                metric_family=family,
                                code=str(row.get(status_column, "unavailable")),
                                detail=(
                                    "No raw-joint-supported grid cells/constraints are "
                                    "available for this auxiliary metric."
                                ),
                            )
                        )
        except (ComparisonAnalysisError, ValueError) as exc:
            row["surface_diagnostic_status"] = "unavailable"
            metric_exclusions.append(
                _metric_exclusion(
                    base,
                    metric_family="surface_diagnostics",
                    code="invalid_surface_diagnostics",
                    detail=str(exc),
                )
            )
        if evaluate_embedded_atm_skew:
            atm_skew, exclusions = _evaluate_atm_skew(
                raw,
                current_surface=current,
                target_surface=target,
                predicted_surface=predicted,
                support_mask=(
                    support_mask if run.support_mask_mode == "raw_joint" else None
                ),
                base=base,
            )
            row.update(atm_skew)
        else:
            row.update(
                {
                    "atm_metric_status": "pending_maturity_bridge",
                    "atm_model_abs_error": np.nan,
                    "atm_persistence_abs_error": np.nan,
                    "atm_error_gap": np.nan,
                    "skew_metric_status": "pending_maturity_bridge",
                    "skew_model_abs_error": np.nan,
                    "skew_persistence_abs_error": np.nan,
                    "skew_error_gap": np.nan,
                }
            )
            exclusions = []
        sample_rows.append(row)
        metric_exclusions.extend(exclusions)

    sample_metrics = pd.DataFrame(sample_rows)
    if sample_metrics.empty:
        sample_metrics = pd.DataFrame(columns=SAMPLE_METRIC_COLUMNS)
    else:
        for column in SAMPLE_METRIC_COLUMNS:
            if column not in sample_metrics.columns:
                sample_metrics[column] = np.nan
        ordered = list(SAMPLE_METRIC_COLUMNS) + sorted(
            set(sample_metrics.columns) - set(SAMPLE_METRIC_COLUMNS)
        )
        sample_metrics = sample_metrics[ordered]
    return (
        sample_metrics,
        pd.DataFrame(general_exclusions),
        pd.DataFrame(metric_exclusions),
    )


def compute_maturity_metrics(
    run: RunSpec,
    panel_name: str,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    maturity_bridge: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every actual ATM A/B and skew A/B maturity in the bridge.

    Model ATM/skew values are extracted only by interpolation inside the fixed
    16x16 grid.  The approximate ATM coordinate is the observed ``target_atm_q``;
    q=1 is never substituted.  Skew wing q values are recovered from the
    nominal wing strikes and the target forward implied by the paired ATM
    strike/q observation.
    """

    if maturity_bridge.empty:
        return pd.DataFrame(columns=MATURITY_METRIC_COLUMNS), pd.DataFrame()
    panel = validate_panel(panel, panel_name=panel_name)
    panel_key = _sample_join_key(panel)
    prediction_key = _sample_join_key(predictions)
    if panel_key != prediction_key or panel_key not in maturity_bridge.columns:
        raise ComparisonAnalysisError(
            "Panel, predictions and maturity bridge must share the same stable sample key"
        )
    if predictions[panel_key].duplicated().any():
        raise ComparisonAnalysisError(
            "Maturity evaluation received duplicate predictions"
        )
    surface_column = _prediction_surface_column(predictions)
    prediction_map = predictions.set_index(panel_key)[surface_column].to_dict()
    panel_map = panel.set_index(panel_key).to_dict(orient="index")

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for bridge_row in maturity_bridge.to_dict(orient="records"):
        key_value = bridge_row.get(panel_key)
        panel_row = panel_map.get(key_value)
        if panel_row is None or key_value not in prediction_map:
            continue
        base = {
            "run_id": run.run_id,
            "model": run.model,
            "text_ablation_mode": run.text_ablation_mode,
            "support_mask_mode": run.support_mask_mode,
            "generator_current_input_mode": run.generator_current_input_mode,
            "seed": int(run.seed),
            "tolerance_minutes": int(run.tolerance_minutes),
            "panel": panel_name,
            "sample_id": str(
                bridge_row.get("sample_id", panel_row.get("sample_id", ""))
            ),
            "news_row_id": bridge_row.get(
                "news_row_id", panel_row.get("news_row_id", "")
            ),
            "pair_id": str(bridge_row.get("pair_id", panel_row.get("pair_id", ""))),
            "session_id": str(
                bridge_row.get("session_id", panel_row.get("session_id", ""))
            ),
            "slice_pair_id": str(bridge_row.get("slice_pair_id", "")),
            "maturity_date": str(bridge_row.get("maturity_date", "")),
            "underlying_contract_id": str(bridge_row.get("underlying_contract_id", "")),
        }
        try:
            current = _parse_vector(
                panel_row["current_surface_flat"], label="current_surface_flat"
            )
            target = _parse_vector(
                panel_row["target_surface_flat"], label="target_surface_flat"
            )
            predicted = _parse_vector(
                prediction_map[key_value], label="predicted_surface_flat"
            )
            if not (len(current) == len(target) == len(predicted)):
                raise ComparisonAnalysisError("surface length mismatch")
            strike_grid, maturity_grid, shape = _surface_axes(
                panel_row, cell_count=len(target)
            )
            support_mask, _, _ = _joint_support_mask(
                panel_row,
                strike_grid=strike_grid,
                maturity_grid=maturity_grid,
                support_mask_mode=run.support_mask_mode,
            )
            predicted_surface = predicted.reshape(shape)
        except (
            ComparisonAnalysisError,
            GridInterpolationError,
            ValueError,
            SyntaxError,
        ) as exc:
            for family in ("atm", "skew"):
                exclusion = _metric_exclusion(
                    base,
                    metric_family=family,
                    code=getattr(exc, "code", "invalid_surface_or_grid"),
                    detail=str(exc),
                )
                exclusion.update(
                    {
                        "slice_pair_id": base["slice_pair_id"],
                        "maturity_date": base["maturity_date"],
                        "underlying_contract_id": base["underlying_contract_id"],
                    }
                )
                exclusions.append(exclusion)
            continue

        pair_maturity_count = _coerce_optional_float(
            bridge_row.get("pair_maturity_count")
        )
        maturity_weight = (
            1.0 / pair_maturity_count
            if pair_maturity_count and pair_maturity_count > 0
            else 1.0
        )
        record = base | {
            "origin_business_days": _coerce_optional_float(
                bridge_row.get("origin_business_days")
            ),
            "target_business_days": _coerce_optional_float(
                bridge_row.get("target_business_days")
            ),
            "maturity_weight": float(maturity_weight),
            "pair_atm_quality": str(bridge_row.get("pair_atm_quality", "")),
            "atm_metric_status": "excluded",
            "predicted_atm_iv": np.nan,
            "current_atm_iv": _coerce_optional_float(bridge_row.get("current_atm_iv")),
            "target_atm_iv": _coerce_optional_float(bridge_row.get("target_atm_iv")),
            "atm_model_abs_error": np.nan,
            "atm_persistence_abs_error": np.nan,
            "atm_error_gap": np.nan,
            "skew_quality": str(bridge_row.get("skew_quality", "")),
            "skew_metric_status": "excluded",
            "predicted_skew": np.nan,
            "current_skew": _coerce_optional_float(
                bridge_row.get("current_atm_iv_skew_secant")
            ),
            "target_skew": _coerce_optional_float(
                bridge_row.get("target_atm_iv_skew_secant")
            ),
            "skew_model_abs_error": np.nan,
            "skew_persistence_abs_error": np.nan,
            "skew_error_gap": np.nan,
        }

        atm_quality = str(bridge_row.get("pair_atm_quality", "")).strip().upper()
        metric_status = str(bridge_row.get("metric_status", "")).strip().lower()
        if metric_status != "ok" or atm_quality not in {"A", "B"}:
            code = (
                "atm_metric_status_not_ok"
                if metric_status != "ok"
                else "atm_quality_not_ab"
            )
            exclusion = _metric_exclusion(
                base,
                metric_family="atm",
                code=code,
                detail=f"metric_status={metric_status!r}, pair_atm_quality={atm_quality!r}",
            )
            exclusion.update(
                {
                    "slice_pair_id": base["slice_pair_id"],
                    "maturity_date": base["maturity_date"],
                    "underlying_contract_id": base["underlying_contract_id"],
                }
            )
            exclusions.append(exclusion)
        else:
            target_q = _coerce_optional_float(bridge_row.get("target_atm_q"))
            target_days = _coerce_optional_float(bridge_row.get("target_business_days"))
            current_iv = record["current_atm_iv"]
            target_iv = record["target_atm_iv"]
            if None in (target_q, target_days, current_iv, target_iv):
                exclusion = _metric_exclusion(
                    base,
                    metric_family="atm",
                    code="missing_atm_bridge_fields",
                    detail="target_atm_q, target_business_days and current/target_atm_iv are required",
                )
                exclusion.update(
                    {
                        "slice_pair_id": base["slice_pair_id"],
                        "maturity_date": base["maturity_date"],
                        "underlying_contract_id": base["underlying_contract_id"],
                    }
                )
                exclusions.append(exclusion)
            else:
                try:
                    predicted_atm = strict_grid_interpolate(
                        predicted_surface,
                        strike_grid,
                        maturity_grid,
                        moneyness=float(target_q),
                        maturity_days=float(target_days),
                        support_mask=support_mask,
                    )
                    model_error = abs(predicted_atm - float(target_iv))
                    persistence_error = abs(float(current_iv) - float(target_iv))
                    record.update(
                        {
                            "atm_metric_status": "ok",
                            "predicted_atm_iv": float(predicted_atm),
                            "atm_model_abs_error": float(model_error),
                            "atm_persistence_abs_error": float(persistence_error),
                            "atm_error_gap": float(model_error - persistence_error),
                        }
                    )
                except GridInterpolationError as exc:
                    exclusion = _metric_exclusion(
                        base,
                        metric_family="atm",
                        code=exc.code,
                        detail=exc.detail,
                    )
                    exclusion.update(
                        {
                            "slice_pair_id": base["slice_pair_id"],
                            "maturity_date": base["maturity_date"],
                            "underlying_contract_id": base["underlying_contract_id"],
                        }
                    )
                    exclusions.append(exclusion)

        skew_quality = str(bridge_row.get("skew_quality", "")).strip().upper()
        skew_status = str(bridge_row.get("skew_status", "")).strip().lower()
        if skew_status != "ok" or skew_quality not in {"A", "B"}:
            code = (
                "skew_status_not_ok" if skew_status != "ok" else "skew_quality_not_ab"
            )
            exclusion = _metric_exclusion(
                base,
                metric_family="skew",
                code=code,
                detail=f"skew_status={skew_status!r}, skew_quality={skew_quality!r}",
            )
            exclusion.update(
                {
                    "slice_pair_id": base["slice_pair_id"],
                    "maturity_date": base["maturity_date"],
                    "underlying_contract_id": base["underlying_contract_id"],
                }
            )
            exclusions.append(exclusion)
        else:
            pair_atm_strike = _coerce_optional_float(bridge_row.get("pair_atm_strike"))
            target_atm_q = _coerce_optional_float(bridge_row.get("target_atm_q"))
            left_strike = _coerce_optional_float(bridge_row.get("skew_left_strike"))
            right_strike = _coerce_optional_float(bridge_row.get("skew_right_strike"))
            target_days = _coerce_optional_float(bridge_row.get("target_business_days"))
            current_skew = record["current_skew"]
            target_skew = record["target_skew"]
            required = (
                pair_atm_strike,
                target_atm_q,
                left_strike,
                right_strike,
                target_days,
                current_skew,
                target_skew,
            )
            if (
                any(value is None for value in required)
                or float(target_atm_q or 0.0) <= 0.0
            ):
                exclusion = _metric_exclusion(
                    base,
                    metric_family="skew",
                    code="missing_skew_bridge_fields",
                    detail=(
                        "pair_atm_strike, target_atm_q, wing strikes, target days and "
                        "current/target secant slopes are required"
                    ),
                )
                exclusion.update(
                    {
                        "slice_pair_id": base["slice_pair_id"],
                        "maturity_date": base["maturity_date"],
                        "underlying_contract_id": base["underlying_contract_id"],
                    }
                )
                exclusions.append(exclusion)
            else:
                target_forward = float(pair_atm_strike) / float(target_atm_q)
                left_q = float(left_strike) / target_forward
                right_q = float(right_strike) / target_forward
                try:
                    predicted_skew = _surface_skew(
                        predicted_surface,
                        strike_grid,
                        maturity_grid,
                        left_q=left_q,
                        right_q=right_q,
                        maturity_days=float(target_days),
                        support_mask=support_mask,
                    )
                    model_error = abs(predicted_skew - float(target_skew))
                    persistence_error = abs(float(current_skew) - float(target_skew))
                    record.update(
                        {
                            "skew_metric_status": "ok",
                            "predicted_skew": float(predicted_skew),
                            "skew_model_abs_error": float(model_error),
                            "skew_persistence_abs_error": float(persistence_error),
                            "skew_error_gap": float(model_error - persistence_error),
                        }
                    )
                except GridInterpolationError as exc:
                    exclusion = _metric_exclusion(
                        base,
                        metric_family="skew",
                        code=exc.code,
                        detail=exc.detail,
                    )
                    exclusion.update(
                        {
                            "slice_pair_id": base["slice_pair_id"],
                            "maturity_date": base["maturity_date"],
                            "underlying_contract_id": base["underlying_contract_id"],
                        }
                    )
                    exclusions.append(exclusion)
        rows.append(record)

    maturity_metrics = pd.DataFrame(rows)
    if maturity_metrics.empty:
        maturity_metrics = pd.DataFrame(columns=MATURITY_METRIC_COLUMNS)
    else:
        for column in MATURITY_METRIC_COLUMNS:
            if column not in maturity_metrics.columns:
                maturity_metrics[column] = np.nan
        maturity_metrics = maturity_metrics[list(MATURITY_METRIC_COLUMNS)]
    return maturity_metrics, pd.DataFrame(exclusions)


def attach_maturity_summaries(
    sample_metrics: pd.DataFrame,
    maturity_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Attach equal-maturity summaries to article-level surface metrics."""

    if sample_metrics.empty or maturity_metrics.empty:
        return sample_metrics.copy()
    output = sample_metrics.copy()
    keys = ["run_id", "panel", "sample_id"]
    summaries: list[dict[str, Any]] = []
    for values, group in maturity_metrics.groupby(keys, sort=True, dropna=False):
        weights = pd.to_numeric(group["maturity_weight"], errors="coerce").fillna(0.0)
        record = dict(zip(keys, values))
        for metric in (
            "atm_model_abs_error",
            "atm_persistence_abs_error",
            "skew_model_abs_error",
            "skew_persistence_abs_error",
        ):
            record[metric] = _weighted_mean(group[metric], weights)
        record["atm_metric_status"] = (
            "ok"
            if group["atm_metric_status"].astype(str).eq("ok").any()
            else "excluded"
        )
        record["atm_maturity_count"] = int(
            group["atm_metric_status"].astype(str).eq("ok").sum()
        )
        record["skew_metric_status"] = (
            "ok"
            if group["skew_metric_status"].astype(str).eq("ok").any()
            else "excluded"
        )
        record["skew_maturity_count"] = int(
            group["skew_metric_status"].astype(str).eq("ok").sum()
        )
        summaries.append(record)
    summary = pd.DataFrame(summaries)
    update_columns = [
        "atm_metric_status",
        "atm_maturity_count",
        "atm_model_abs_error",
        "atm_persistence_abs_error",
        "skew_metric_status",
        "skew_maturity_count",
        "skew_model_abs_error",
        "skew_persistence_abs_error",
    ]
    output = output.drop(columns=update_columns, errors="ignore").merge(
        summary,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    output["atm_metric_status"] = output["atm_metric_status"].fillna("excluded")
    output["skew_metric_status"] = output["skew_metric_status"].fillna("excluded")
    output["atm_error_gap"] = (
        output["atm_model_abs_error"] - output["atm_persistence_abs_error"]
    )
    output["skew_error_gap"] = (
        output["skew_model_abs_error"] - output["skew_persistence_abs_error"]
    )
    for column in SAMPLE_METRIC_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
    extras = sorted(set(output.columns) - set(SAMPLE_METRIC_COLUMNS))
    return output[list(SAMPLE_METRIC_COLUMNS) + extras]


def _shift_bucket(value: Any) -> str:
    numeric = _coerce_optional_float(value)
    if numeric is None:
        return "missing"
    if numeric < 0.0:
        return "negative_invalid"
    if math.isclose(numeric, 0.0, abs_tol=1e-12):
        return "0m"
    if numeric <= 4.0:
        return "1-4m"
    return ">=5m"


def _first_tolerance_bucket(value: Any) -> str:
    numeric = _coerce_optional_float(value)
    return "missing" if numeric is None else f"{int(numeric)}m"


def _expand_strata(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    if sample_metrics.empty:
        return sample_metrics.assign(
            stratum_type=pd.Series(dtype=str), stratum_value=pd.Series(dtype=str)
        )
    frames: list[pd.DataFrame] = []
    overall = sample_metrics.copy()
    overall["stratum_type"] = "overall"
    overall["stratum_value"] = "all"
    frames.append(overall)
    definitions = {
        "first_included_tolerance": sample_metrics[
            "first_included_tolerance_minutes"
        ].map(_first_tolerance_bucket),
        "origin_shift": sample_metrics["origin_shift_minutes"].map(_shift_bucket),
        "atm_ab": sample_metrics["atm_ab_flag"].map(_coerce_flag),
    }
    for name, values in definitions.items():
        frame = sample_metrics.copy()
        frame["stratum_type"] = name
        frame["stratum_value"] = values.astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    weight_values = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric) & np.isfinite(weight_values) & (weight_values > 0.0)
    if not valid.any():
        return float("nan")
    return float(np.average(numeric[valid], weights=weight_values[valid]))


def aggregate_pair_metrics(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate articles within each pair, then expose overall and audit strata."""

    if sample_metrics.empty:
        return pd.DataFrame(columns=PAIR_METRIC_COLUMNS)
    prepared = sample_metrics.copy()
    if "text_ablation_mode" not in prepared.columns:
        prepared["text_ablation_mode"] = REAL_TEXT
    if "support_mask_mode" not in prepared.columns:
        prepared["support_mask_mode"] = "none"
    if "generator_current_input_mode" not in prepared.columns:
        prepared["generator_current_input_mode"] = "full_current"
    expanded = _expand_strata(prepared)
    group_columns = [
        "run_id",
        "model",
        "text_ablation_mode",
        "support_mask_mode",
        "generator_current_input_mode",
        "seed",
        "tolerance_minutes",
        "panel",
        "stratum_type",
        "stratum_value",
        "pair_id",
    ]
    metric_columns = [
        "model_mae",
        "model_rmse",
        "model_max_abs",
        "persistence_mae",
        "persistence_rmse",
        "persistence_max_abs",
        "short_atm_cell_count",
        "short_atm_model_mae",
        "short_atm_persistence_mae",
        "predicted_calendar_violation_count",
        "predicted_calendar_constraint_count",
        "predicted_calendar_violation_rate",
        "target_calendar_violation_count",
        "target_calendar_constraint_count",
        "target_calendar_violation_rate",
        "current_calendar_violation_count",
        "current_calendar_constraint_count",
        "current_calendar_violation_rate",
        "predicted_butterfly_violation_count",
        "predicted_butterfly_constraint_count",
        "predicted_butterfly_violation_rate",
        "target_butterfly_violation_count",
        "target_butterfly_constraint_count",
        "target_butterfly_violation_rate",
        "current_butterfly_violation_count",
        "current_butterfly_constraint_count",
        "current_butterfly_violation_rate",
        "atm_maturity_count",
        "atm_model_abs_error",
        "atm_persistence_abs_error",
        "skew_maturity_count",
        "skew_model_abs_error",
        "skew_persistence_abs_error",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in expanded.groupby(group_columns, sort=True, dropna=False):
        sessions = group["session_id"].astype(str).drop_duplicates().tolist()
        if len(sessions) != 1:
            raise ComparisonAnalysisError(
                f"pair_id={keys[-1]} maps to multiple CME sessions within one run/panel"
            )
        weights = pd.to_numeric(group["sample_weight"], errors="coerce").fillna(0.0)
        if not bool((weights > 0.0).any()):
            weights = pd.Series(np.ones(len(group)), index=group.index, dtype=float)
        record = dict(zip(group_columns, keys))
        record.update(
            {
                "session_id": sessions[0],
                "article_count": int(len(group)),
                "article_weight_sum": float(weights.sum()),
                "first_included_tolerance_minutes": _weighted_mean(
                    group["first_included_tolerance_minutes"], weights
                ),
                "origin_shift_minutes": _weighted_mean(
                    group["origin_shift_minutes"], weights
                ),
                "atm_ab_flag": (
                    group["atm_ab_flag"].astype(str).iloc[0]
                    if group["atm_ab_flag"].astype(str).nunique() == 1
                    else "mixed"
                ),
            }
        )
        for metric in metric_columns:
            values = (
                group[metric]
                if metric in group.columns
                else pd.Series(np.nan, index=group.index, dtype=float)
            )
            record[metric] = _weighted_mean(values, weights)
        record["mae_gap"] = record["model_mae"] - record["persistence_mae"]
        record["mae_tie"] = float(_mae_gap_is_tie(record["mae_gap"]))
        record["skill"] = (
            1.0 - record["model_mae"] / record["persistence_mae"]
            if record["persistence_mae"] > 0.0
            else float("nan")
        )
        record["win"] = float(_mae_gap_is_win(record["mae_gap"]))
        short_metrics_finite = np.isfinite(
            record["short_atm_model_mae"]
        ) and np.isfinite(record["short_atm_persistence_mae"])
        record["short_atm_mae_gap"] = (
            record["short_atm_model_mae"] - record["short_atm_persistence_mae"]
            if short_metrics_finite
            else float("nan")
        )
        record["short_atm_tie"] = (
            float(_mae_gap_is_tie(record["short_atm_mae_gap"]))
            if short_metrics_finite
            else float("nan")
        )
        record["short_atm_skill"] = (
            1.0 - record["short_atm_model_mae"] / record["short_atm_persistence_mae"]
            if short_metrics_finite and record["short_atm_persistence_mae"] > 0.0
            else float("nan")
        )
        record["short_atm_win"] = (
            float(_mae_gap_is_win(record["short_atm_mae_gap"]))
            if short_metrics_finite
            else float("nan")
        )
        record["calendar_violation_rate_gap"] = (
            record["predicted_calendar_violation_rate"]
            - record["target_calendar_violation_rate"]
        )
        record["butterfly_violation_rate_gap"] = (
            record["predicted_butterfly_violation_rate"]
            - record["target_butterfly_violation_rate"]
        )
        record["atm_error_gap"] = (
            record["atm_model_abs_error"] - record["atm_persistence_abs_error"]
            if np.isfinite(record["atm_model_abs_error"])
            and np.isfinite(record["atm_persistence_abs_error"])
            else float("nan")
        )
        record["skew_error_gap"] = (
            record["skew_model_abs_error"] - record["skew_persistence_abs_error"]
            if np.isfinite(record["skew_model_abs_error"])
            and np.isfinite(record["skew_persistence_abs_error"])
            else float("nan")
        )
        rows.append(record)
    output = pd.DataFrame(rows)
    for column in PAIR_METRIC_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
    return output[list(PAIR_METRIC_COLUMNS)]


def _mean_finite(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _collapse_runs_to_model_pair(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average repeated seeds/runs without treating them as extra market pairs."""

    if pair_metrics.empty:
        return pair_metrics.copy()
    pair_metrics = pair_metrics.copy()
    if "text_ablation_mode" not in pair_metrics.columns:
        pair_metrics["text_ablation_mode"] = REAL_TEXT
    if "support_mask_mode" not in pair_metrics.columns:
        pair_metrics["support_mask_mode"] = "none"
    if "generator_current_input_mode" not in pair_metrics.columns:
        pair_metrics["generator_current_input_mode"] = "full_current"
    keys = [
        "model",
        "text_ablation_mode",
        "support_mask_mode",
        "generator_current_input_mode",
        "tolerance_minutes",
        "panel",
        "stratum_type",
        "stratum_value",
        "pair_id",
        "session_id",
    ]
    metric_columns = [
        "model_mae",
        "model_rmse",
        "model_max_abs",
        "persistence_mae",
        "persistence_rmse",
        "persistence_max_abs",
        "mae_gap",
        "mae_tie",
        "skill",
        "win",
        "short_atm_cell_count",
        "short_atm_model_mae",
        "short_atm_persistence_mae",
        "short_atm_mae_gap",
        "short_atm_tie",
        "short_atm_skill",
        "short_atm_win",
        "predicted_calendar_violation_count",
        "predicted_calendar_constraint_count",
        "predicted_calendar_violation_rate",
        "target_calendar_violation_count",
        "target_calendar_constraint_count",
        "target_calendar_violation_rate",
        "current_calendar_violation_count",
        "current_calendar_constraint_count",
        "current_calendar_violation_rate",
        "calendar_violation_rate_gap",
        "predicted_butterfly_violation_count",
        "predicted_butterfly_constraint_count",
        "predicted_butterfly_violation_rate",
        "target_butterfly_violation_count",
        "target_butterfly_constraint_count",
        "target_butterfly_violation_rate",
        "current_butterfly_violation_count",
        "current_butterfly_constraint_count",
        "current_butterfly_violation_rate",
        "butterfly_violation_rate_gap",
        "atm_maturity_count",
        "atm_model_abs_error",
        "atm_persistence_abs_error",
        "atm_error_gap",
        "skew_maturity_count",
        "skew_model_abs_error",
        "skew_persistence_abs_error",
        "skew_error_gap",
    ]
    rows: list[dict[str, Any]] = []
    for values, group in pair_metrics.groupby(keys, sort=True, dropna=False):
        record = dict(zip(keys, values))
        record["run_count"] = int(group["run_id"].nunique())
        record["seed_count"] = int(group["seed"].nunique())
        for metric in metric_columns:
            values = (
                group[metric]
                if metric in group.columns
                else pd.Series(np.nan, index=group.index, dtype=float)
            )
            record[metric] = _mean_finite(values)
        rows.append(record)
    return pd.DataFrame(rows)


def build_model_comparison(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    collapsed = _collapse_runs_to_model_pair(pair_metrics)
    if collapsed.empty:
        return pd.DataFrame(columns=MODEL_COMPARISON_COLUMNS)
    keys = [
        "model",
        "text_ablation_mode",
        "support_mask_mode",
        "generator_current_input_mode",
        "tolerance_minutes",
        "panel",
        "stratum_type",
        "stratum_value",
    ]
    rows: list[dict[str, Any]] = []
    for values, group in collapsed.groupby(keys, sort=True, dropna=False):
        record = dict(zip(keys, values))
        atm_errors = pd.to_numeric(group["atm_model_abs_error"], errors="coerce")
        skew_errors = pd.to_numeric(group["skew_model_abs_error"], errors="coerce")
        atm_maturity_counts = pd.to_numeric(
            group["atm_maturity_count"], errors="coerce"
        )
        skew_maturity_counts = pd.to_numeric(
            group["skew_maturity_count"], errors="coerce"
        )
        record.update(
            {
                "run_count": int(group["run_count"].max()),
                "seed_count": int(group["seed_count"].max()),
                "pair_count": int(group["pair_id"].nunique()),
                "session_count": int(group["session_id"].nunique()),
                "mae": _mean_finite(group["model_mae"]),
                "rmse": _mean_finite(group["model_rmse"]),
                "persistence_mae": _mean_finite(group["persistence_mae"]),
                "gap": _mean_finite(group["mae_gap"]),
                "tie_rate": _mean_finite(group["mae_tie"]),
                "skill": _mean_finite(group["skill"]),
                "win": _mean_finite(group["win"]),
                "short_atm_mae": _mean_finite(group["short_atm_model_mae"]),
                "short_atm_persistence_mae": _mean_finite(
                    group["short_atm_persistence_mae"]
                ),
                "short_atm_gap": _mean_finite(group["short_atm_mae_gap"]),
                "short_atm_tie_rate": _mean_finite(group["short_atm_tie"]),
                "short_atm_skill": _mean_finite(group["short_atm_skill"]),
                "short_atm_win": _mean_finite(group["short_atm_win"]),
                "predicted_calendar_violation_count_mean": _mean_finite(
                    group["predicted_calendar_violation_count"]
                ),
                "predicted_calendar_violation_rate": _mean_finite(
                    group["predicted_calendar_violation_rate"]
                ),
                "target_calendar_violation_count_mean": _mean_finite(
                    group["target_calendar_violation_count"]
                ),
                "target_calendar_violation_rate": _mean_finite(
                    group["target_calendar_violation_rate"]
                ),
                "current_calendar_violation_count_mean": _mean_finite(
                    group["current_calendar_violation_count"]
                ),
                "current_calendar_violation_rate": _mean_finite(
                    group["current_calendar_violation_rate"]
                ),
                "calendar_violation_rate_gap": _mean_finite(
                    group["calendar_violation_rate_gap"]
                ),
                "predicted_butterfly_violation_count_mean": _mean_finite(
                    group["predicted_butterfly_violation_count"]
                ),
                "predicted_butterfly_violation_rate": _mean_finite(
                    group["predicted_butterfly_violation_rate"]
                ),
                "target_butterfly_violation_count_mean": _mean_finite(
                    group["target_butterfly_violation_count"]
                ),
                "target_butterfly_violation_rate": _mean_finite(
                    group["target_butterfly_violation_rate"]
                ),
                "current_butterfly_violation_count_mean": _mean_finite(
                    group["current_butterfly_violation_count"]
                ),
                "current_butterfly_violation_rate": _mean_finite(
                    group["current_butterfly_violation_rate"]
                ),
                "butterfly_violation_rate_gap": _mean_finite(
                    group["butterfly_violation_rate_gap"]
                ),
                "atm_pair_count": int(
                    np.isfinite(atm_errors.to_numpy(dtype=float)).sum()
                ),
                "atm_maturity_count": int(
                    round(
                        float(
                            atm_maturity_counts[np.isfinite(atm_maturity_counts)].sum()
                        )
                    )
                ),
                "atm_mae": _mean_finite(group["atm_model_abs_error"]),
                "atm_gap": _mean_finite(group["atm_error_gap"]),
                "skew_pair_count": int(
                    np.isfinite(skew_errors.to_numpy(dtype=float)).sum()
                ),
                "skew_maturity_count": int(
                    round(
                        float(
                            skew_maturity_counts[
                                np.isfinite(skew_maturity_counts)
                            ].sum()
                        )
                    )
                ),
                "skew_mae": _mean_finite(group["skew_model_abs_error"]),
                "skew_gap": _mean_finite(group["skew_error_gap"]),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)[list(MODEL_COMPARISON_COLUMNS)]


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down family-wise error correction, preserving NaNs."""

    array = np.asarray(p_values, dtype=np.float64)
    finite = np.isfinite(array)
    output = np.full(len(array), np.nan, dtype=np.float64)
    if not finite.any():
        return output.tolist()
    indexes = np.flatnonzero(finite)
    order = indexes[np.argsort(array[indexes], kind="stable")]
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * float(array[index])))
        output[index] = running
    return output.tolist()


def cme_session_cluster_bootstrap(
    differences: Sequence[float],
    session_ids: Sequence[str],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Pair-balanced mean with whole-CME-session cluster resampling."""

    if int(iterations) <= 0:
        raise ComparisonAnalysisError("bootstrap iterations must be positive")
    frame = pd.DataFrame(
        {
            "difference": pd.to_numeric(pd.Series(differences), errors="coerce"),
            "session_id": pd.Series(session_ids, dtype="object"),
        }
    )
    frame = frame[
        frame["difference"].notna()
        & np.isfinite(frame["difference"].to_numpy(dtype=np.float64))
        & frame["session_id"].notna()
        & frame["session_id"].astype(str).str.strip().ne("")
    ].copy()
    if frame.empty:
        return {
            "mean_diff": float("nan"),
            "ci_95_lower": float("nan"),
            "ci_95_upper": float("nan"),
            "p_two_sided": float("nan"),
            "pair_count": 0,
            "session_count": 0,
            "status": "no_valid_pairs",
        }
    grouped = frame.groupby("session_id", sort=True)["difference"].agg(["sum", "count"])
    observed = float(frame["difference"].mean())
    if frame["difference"].eq(0.0).all():
        return {
            "mean_diff": 0.0,
            "ci_95_lower": 0.0,
            "ci_95_upper": 0.0,
            "p_two_sided": 1.0,
            "pair_count": int(len(frame)),
            "session_count": int(len(grouped)),
            "status": "numerical_tie",
        }
    if len(grouped) < 2:
        return {
            "mean_diff": observed,
            "ci_95_lower": float("nan"),
            "ci_95_upper": float("nan"),
            "p_two_sided": float("nan"),
            "pair_count": int(len(frame)),
            "session_count": int(len(grouped)),
            "status": "insufficient_sessions",
        }
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    selected = rng.integers(0, len(grouped), size=(int(iterations), len(grouped)))
    draws = np.sum(sums[selected], axis=1) / np.sum(counts[selected], axis=1)
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    centered = draws - observed
    p_two = (float(np.sum(np.abs(centered) >= abs(observed))) + 1.0) / (
        float(iterations) + 1.0
    )
    return {
        "mean_diff": observed,
        "ci_95_lower": float(ci_low),
        "ci_95_upper": float(ci_high),
        "p_two_sided": float(p_two),
        "pair_count": int(len(frame)),
        "session_count": int(len(grouped)),
        "status": "ok",
    }


def _is_wgan_model(model: str) -> bool:
    return "wgan" in str(model).strip().lower()


def build_bootstrap_comparisons(
    pair_metrics: pd.DataFrame,
    *,
    base_tolerance_minutes: int = DEFAULT_BASE_TOLERANCE_MINUTES,
    focal_tolerances: Sequence[int] = DEFAULT_FOCAL_TOLERANCES,
    metrics: Sequence[str] = DEFAULT_BOOTSTRAP_METRICS,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    comparison_models: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare WGAN 10/15/30m against 5m on identical market pairs."""

    collapsed = _collapse_runs_to_model_pair(pair_metrics)
    if collapsed.empty:
        return pd.DataFrame(columns=BOOTSTRAP_COMPARISON_COLUMNS)
    if comparison_models is None:
        models = sorted(
            model
            for model in collapsed["model"].astype(str).unique()
            if _is_wgan_model(model)
        )
    else:
        models = [str(model) for model in comparison_models]
    rows: list[dict[str, Any]] = []
    grouping = ["text_ablation_mode", "panel", "stratum_type", "stratum_value"]
    for model in models:
        model_frame = collapsed[collapsed["model"].astype(str).eq(model)]
        for group_values, group in model_frame.groupby(
            grouping, sort=True, dropna=False
        ):
            text_mode, panel, stratum_type, stratum_value = group_values
            base = group[
                pd.to_numeric(group["tolerance_minutes"], errors="coerce").eq(
                    int(base_tolerance_minutes)
                )
            ]
            for focal_tolerance in focal_tolerances:
                focal = group[
                    pd.to_numeric(group["tolerance_minutes"], errors="coerce").eq(
                        int(focal_tolerance)
                    )
                ]
                for metric in metrics:
                    if metric not in group.columns:
                        continue
                    left = focal[["pair_id", "session_id", metric]].rename(
                        columns={
                            "session_id": "focal_session_id",
                            metric: "focal_value",
                        }
                    )
                    right = base[["pair_id", "session_id", metric]].rename(
                        columns={"session_id": "base_session_id", metric: "base_value"}
                    )
                    matched = left.merge(
                        right, on="pair_id", how="inner", validate="one_to_one"
                    )
                    if not matched.empty and not matched["focal_session_id"].astype(
                        str
                    ).equals(matched["base_session_id"].astype(str)):
                        raise ComparisonAnalysisError(
                            f"CME session mismatch while comparing {model} {focal_tolerance}m vs "
                            f"{base_tolerance_minutes}m"
                        )
                    differences = _zero_numerical_ties(
                        pd.to_numeric(matched["focal_value"], errors="coerce")
                        - pd.to_numeric(matched["base_value"], errors="coerce")
                    )
                    row_seed = int(bootstrap_seed) + _stable_offset(
                        f"{model}|{panel}|{stratum_type}|{stratum_value}|{metric}|"
                        f"{focal_tolerance}|{base_tolerance_minutes}"
                    )
                    result = cme_session_cluster_bootstrap(
                        differences,
                        matched["focal_session_id"],
                        iterations=int(bootstrap_iterations),
                        seed=row_seed,
                    )
                    negative_better = metric not in {"skill", "win"}
                    rows.append(
                        {
                            "model": model,
                            "text_ablation_mode": text_mode,
                            "focal_tolerance_minutes": int(focal_tolerance),
                            "base_tolerance_minutes": int(base_tolerance_minutes),
                            "panel": panel,
                            "metric": metric,
                            "stratum_type": stratum_type,
                            "stratum_value": stratum_value,
                            "difference_direction": "focal_minus_base",
                            "negative_means_focal_better": bool(negative_better),
                            **result,
                            "ci_low": result["ci_95_lower"],
                            "ci_high": result["ci_95_upper"],
                            "p_value": result["p_two_sided"],
                            "p_holm": np.nan,
                            "holm_family": (
                                f"{model}|{text_mode}|{panel}|{metric}|{stratum_type}|{stratum_value}|"
                                f"vs_{base_tolerance_minutes}m"
                            ),
                            "bootstrap_iterations": int(bootstrap_iterations),
                            "bootstrap_seed": int(row_seed),
                            "numerical_tie_tolerance": MAE_NUMERICAL_TIE_TOLERANCE,
                        }
                    )
    output = pd.DataFrame(rows)
    if output.empty:
        return pd.DataFrame(columns=BOOTSTRAP_COMPARISON_COLUMNS)
    for _, indexes in output.groupby("holm_family", sort=True).groups.items():
        index_list = list(indexes)
        output.loc[index_list, "p_holm"] = holm_adjust(
            output.loc[index_list, "p_two_sided"]
        )
    for column in BOOTSTRAP_COMPARISON_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
    return (
        output[list(BOOTSTRAP_COMPARISON_COLUMNS)]
        .sort_values(
            [
                "model",
                "text_ablation_mode",
                "panel",
                "metric",
                "stratum_type",
                "stratum_value",
                "focal_tolerance_minutes",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_text_ablation_comparisons(
    pair_metrics: pd.DataFrame,
    *,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Compare real news text with current-only and shuffled-text controls.

    The estimand is always ``real_text model_mae - control model_mae`` on the
    identical core-Q4 market pairs. Therefore a negative difference is the
    pre-registered direction indicating incremental predictive value in the
    real news embedding.
    """

    collapsed = _collapse_runs_to_model_pair(pair_metrics)
    if collapsed.empty or "text_ablation_mode" not in collapsed.columns:
        return pd.DataFrame(columns=TEXT_ABLATION_COMPARISON_COLUMNS)
    scope = collapsed[
        collapsed["panel"].astype(str).eq("core")
        & collapsed["stratum_type"].astype(str).eq("overall")
        & collapsed["stratum_value"].astype(str).eq("all")
    ].copy()
    modes = set(scope["text_ablation_mode"].astype(str))
    if not {REAL_TEXT, CURRENT_ONLY, TEXT_SHUFFLE}.issubset(modes):
        return pd.DataFrame(columns=TEXT_ABLATION_COMPARISON_COLUMNS)

    rows: list[dict[str, Any]] = []
    for model in sorted(scope["model"].astype(str).unique()):
        for tolerance in sorted(
            pd.to_numeric(scope["tolerance_minutes"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        ):
            subset = scope[
                scope["model"].astype(str).eq(model)
                & pd.to_numeric(scope["tolerance_minutes"], errors="coerce").eq(
                    tolerance
                )
            ]
            real = subset[subset["text_ablation_mode"].astype(str).eq(REAL_TEXT)][
                ["pair_id", "session_id", "model_mae"]
            ].rename(
                columns={"session_id": "real_session_id", "model_mae": "real_value"}
            )
            for control_mode in (CURRENT_ONLY, TEXT_SHUFFLE):
                control = subset[
                    subset["text_ablation_mode"].astype(str).eq(control_mode)
                ][["pair_id", "session_id", "model_mae"]].rename(
                    columns={
                        "session_id": "control_session_id",
                        "model_mae": "control_value",
                    }
                )
                matched = real.merge(
                    control, on="pair_id", how="inner", validate="one_to_one"
                )
                if not matched.empty and not matched["real_session_id"].astype(
                    str
                ).equals(matched["control_session_id"].astype(str)):
                    raise ComparisonAnalysisError(
                        f"CME session mismatch in {model}/{tolerance}m real-vs-{control_mode}"
                    )
                differences = _zero_numerical_ties(
                    pd.to_numeric(matched["real_value"], errors="coerce")
                    - pd.to_numeric(matched["control_value"], errors="coerce")
                )
                row_seed = int(bootstrap_seed) + _stable_offset(
                    f"text_ablation|{model}|{tolerance}|{control_mode}|model_mae"
                )
                result = cme_session_cluster_bootstrap(
                    differences,
                    matched["real_session_id"],
                    iterations=int(bootstrap_iterations),
                    seed=row_seed,
                )
                rows.append(
                    {
                        "model": model,
                        "tolerance_minutes": int(tolerance),
                        "panel": "core",
                        "metric": "model_mae",
                        "control_mode": control_mode,
                        "difference_direction": "real_text_minus_control",
                        "negative_means_real_text_better": True,
                        **result,
                        "p_holm": np.nan,
                        "holm_family": "global|core|model_mae|real_vs_controls",
                        "bootstrap_iterations": int(bootstrap_iterations),
                        "bootstrap_seed": int(row_seed),
                        "numerical_tie_tolerance": MAE_NUMERICAL_TIE_TOLERANCE,
                    }
                )
    output = pd.DataFrame(rows)
    if output.empty:
        return pd.DataFrame(columns=TEXT_ABLATION_COMPARISON_COLUMNS)
    for _, indexes in output.groupby("holm_family", sort=True).groups.items():
        selected = list(indexes)
        output.loc[selected, "p_holm"] = holm_adjust(
            output.loc[selected, "p_two_sided"]
        )
    for column in TEXT_ABLATION_COMPARISON_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
    return (
        output[list(TEXT_ABLATION_COMPARISON_COLUMNS)]
        .sort_values(["model", "tolerance_minutes", "control_mode"], kind="stable")
        .reset_index(drop=True)
    )


def select_primary_tolerance_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the three pre-registered real-text WGAN tolerance comparisons."""

    if frame.empty:
        return frame.copy()
    mask = (
        frame["model"].astype(str).map(_is_wgan_model)
        & frame["panel"].astype(str).eq("core")
        & frame["metric"].astype(str).eq("model_mae")
        & frame["stratum_type"].astype(str).eq("overall")
        & frame["stratum_value"].astype(str).eq("all")
    )
    if "text_ablation_mode" in frame.columns:
        mask &= frame["text_ablation_mode"].astype(str).eq(REAL_TEXT)
    return frame.loc[mask].copy()


def _stable_sample_text(row: Mapping[str, Any]) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    return (
        f"news={row.get('news_row_id', '')}|pair={row.get('pair_id', '')}|"
        f"origin={row.get('effective_origin_utc', row.get('current_snapshot_time_utc', ''))}"
    )


class TrainedRunEvaluator:
    """Production evaluator for the frozen WGAN/regression checkpoints.

    Gaussian-WGAN predictions are the arithmetic mean of 64 stable-key noise
    draws.  Zero-noise WGAN and regression checkpoints require one explicit
    deterministic pass.  Draws and samples are batched; no uncertainty
    fallback or validation-data calibration is applied.
    """

    def __init__(
        self,
        *,
        mc_samples: int = 64,
        sample_batch_size: int = 32,
        draw_batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        if (
            int(mc_samples) <= 0
            or int(sample_batch_size) <= 0
            or int(draw_batch_size) <= 0
        ):
            raise ComparisonAnalysisError(
                "mc_samples and evaluator batch sizes must be positive"
            )
        self.mc_samples = int(mc_samples)
        self.sample_batch_size = int(sample_batch_size)
        self.draw_batch_size = int(draw_batch_size)
        self.device_name = device or os.environ.get("NEWS_FIRST_EVAL_DEVICE", "auto")
        self._cache_key: tuple[str, tuple[int, int], str, str, str] | None = None
        self._cached_model: Any = None
        self._cached_config: Any = None
        self._cached_embedding_dim: int | None = None
        self._cached_device: Any = None

    def _device(self):
        import torch

        normalized = str(self.device_name).strip().lower()
        if normalized in {"", "auto"}:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(normalized)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ComparisonAnalysisError(
                f"Requested CUDA evaluator device is unavailable: {device}"
            )
        return device

    def _load(
        self,
        run: RunSpec,
        shape: tuple[int, int],
        *,
        strike_grid: np.ndarray,
        maturity_grid: np.ndarray,
    ):
        import torch
        from wgan_option.utils.inference_helpers import (
            load_vol_generator,
            load_vol_regressor,
        )

        device = self._device()
        grid_key = hashlib.sha256(
            json.dumps(
                {
                    "strike_grid": [float(value) for value in strike_grid],
                    "maturity_grid_days": [float(value) for value in maturity_grid],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        key = (
            str(run.checkpoint_path.resolve()),
            tuple(shape),
            str(device),
            str(run.generator_current_input_mode),
            grid_key,
        )
        if key == self._cache_key:
            return (
                self._cached_model,
                self._cached_config,
                int(self._cached_embedding_dim),
                device,
            )
        if self._cached_model is not None:
            del self._cached_model
            self._cached_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        sample = SimpleNamespace(
            current_surface=np.zeros((1, shape[0], shape[1]), dtype=np.float32),
            strike_grid=np.asarray(strike_grid, dtype=np.float32),
            maturity_grid_days=np.asarray(maturity_grid, dtype=np.float32),
        )
        if str(run.model).strip().lower() == "regression":
            model, config, embedding_dim = load_vol_regressor(
                run.checkpoint_path, sample, device
            )
        elif _is_wgan_model(run.model):
            model, config, embedding_dim = load_vol_generator(
                run.checkpoint_path, sample, device
            )
        else:
            raise ComparisonAnalysisError(
                f"Production evaluator supports only WGAN/regression, got model={run.model!r}"
            )
        if str(getattr(config, "text_embedding_mode", "lp")).strip().lower() != "lp":
            raise ComparisonAnalysisError(
                f"Frozen news-first evaluation requires LP embeddings; checkpoint uses "
                f"{getattr(config, 'text_embedding_mode', None)!r}"
            )
        checkpoint_mode = normalize_text_ablation_mode(
            getattr(config, "news_first_text_ablation_mode", REAL_TEXT)
        )
        if checkpoint_mode != normalize_text_ablation_mode(run.text_ablation_mode):
            raise ComparisonAnalysisError(
                f"Run/checkpoint text mode mismatch: {run.text_ablation_mode!r} != "
                f"{checkpoint_mode!r}"
            )
        checkpoint_support = (
            str(getattr(config, "support_mask_mode", "none") or "none").strip().lower()
        )
        if checkpoint_support != str(run.support_mask_mode).strip().lower():
            raise ComparisonAnalysisError(
                f"Run/checkpoint support mode mismatch: {run.support_mask_mode!r} != "
                f"{checkpoint_support!r}"
            )
        checkpoint_current_input = (
            str(getattr(config, "generator_current_input_mode", "full_current"))
            .strip()
            .lower()
        )
        if (
            checkpoint_current_input
            != str(run.generator_current_input_mode).strip().lower()
        ):
            raise ComparisonAnalysisError(
                "Run/checkpoint generator current-input mode mismatch: "
                f"{run.generator_current_input_mode!r} != "
                f"{checkpoint_current_input!r}"
            )
        self._cache_key = key
        self._cached_model = model
        self._cached_config = config
        self._cached_embedding_dim = int(embedding_dim)
        self._cached_device = device
        return model, config, int(embedding_dim), device

    def __call__(
        self,
        run: RunSpec,
        panel_name: str,
        panel: pd.DataFrame,
    ) -> pd.DataFrame:
        import torch
        from wgan_option.utils.merged_xlsx_parsing import (
            _parse_serialized_vector,
            _resolve_text_embedding,
        )
        from wgan_option.models.common import (
            CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
            GAUSSIAN_GENERATOR_NOISE_MODE,
            ZERO_GENERATOR_NOISE_MODE,
            generator_current_input_fingerprint,
            generator_noise_fingerprint,
            normalize_generator_current_input_mode,
            normalize_generator_noise_mode,
        )
        from wgan_option.utils.weighted_training import (
            stable_key_to_int64,
            stable_noise_for_keys,
        )

        if panel.empty:
            return pd.DataFrame(columns=["sample_id", "predicted_surface_flat"])
        first_current = _parse_serialized_vector(panel.iloc[0]["current_surface_flat"])
        shape = _parse_surface_shape(
            panel.iloc[0].get("surface_shape"), cell_count=len(first_current)
        )
        if {
            "strike_grid",
            "maturity_days_grid",
        }.issubset(panel.columns):
            strike_grid, maturity_grid, parsed_shape = _surface_axes(
                panel.iloc[0], cell_count=len(first_current)
            )
        else:
            # Historical full-current checkpoints did not persist an explicit
            # grid contract.  Their loader deliberately preserves that legacy
            # interpretation; masked-current/explicit-grid runs still fail
            # closed later because they require the real axes.
            strike_grid = np.arange(shape[1], dtype=np.float32)
            maturity_grid = np.arange(shape[0], dtype=np.float32)
            parsed_shape = shape
        if tuple(parsed_shape) != tuple(shape):
            raise ComparisonAnalysisError(
                f"Panel surface/grid shape mismatch: {shape} != {parsed_shape}"
            )
        model, config, embedding_dim, device = self._load(
            run,
            shape,
            strike_grid=strike_grid,
            maturity_grid=maturity_grid,
        )

        currents: list[np.ndarray] = []
        current_support_masks: list[np.ndarray] = []
        current_support_cell_counts: list[int] = []
        current_support_mask_fingerprints: list[str] = []
        embeddings: list[np.ndarray] = []
        stable_keys: list[int] = []
        sample_ids: list[str] = []
        news_ids: list[Any] = []
        stable_texts: list[str] = []
        pair_ids: list[str] = []
        session_ids: list[str] = []
        current_input_mode = normalize_generator_current_input_mode(
            run.generator_current_input_mode
        )
        for row in panel.to_dict(orient="records"):
            current = _parse_serialized_vector(row["current_surface_flat"])
            if (
                int(current.size) != int(shape[0] * shape[1])
                or not np.isfinite(current).all()
            ):
                raise ComparisonAnalysisError(
                    f"Invalid current surface for sample {_stable_sample_text(row)}"
                )
            embedding = _resolve_text_embedding(
                row.get("hd_embedding", ""),
                row.get("lp_embedding", ""),
                "lp",
            )
            if (
                int(embedding.size) != int(embedding_dim)
                or not np.isfinite(embedding).all()
            ):
                raise ComparisonAnalysisError(
                    f"LP embedding width/values invalid for sample {_stable_sample_text(row)}: "
                    f"{embedding.size} != {embedding_dim}"
                )
            stable_text = _stable_sample_text(row)
            currents.append(current.reshape(1, shape[0], shape[1]).astype(np.float32))
            if current_input_mode == CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE:
                strike_grid, maturity_grid, _ = _surface_axes(
                    row,
                    cell_count=int(current.size),
                )
                current_mask = _current_support_mask(
                    row,
                    strike_grid=strike_grid,
                    maturity_grid=maturity_grid,
                )
                current_support_masks.append(
                    current_mask.reshape(1, shape[0], shape[1]).astype(np.float32)
                )
                current_support_cell_counts.append(int(current_mask.sum()))
                current_support_mask_fingerprints.append(
                    _current_support_mask_fingerprint(
                        current_mask,
                        strike_grid=strike_grid,
                        maturity_grid=maturity_grid,
                    )
                )
            else:
                current_support_cell_counts.append(int(current.size))
                current_support_mask_fingerprints.append("")
            embeddings.append(embedding.astype(np.float32))
            stable_keys.append(stable_key_to_int64(stable_text))
            sample_ids.append(str(row.get("sample_id", stable_text)))
            news_ids.append(row.get("news_row_id", ""))
            stable_texts.append(stable_text)
            pair_ids.append(str(row.get("pair_id", "")))
            session_ids.append(str(row.get("session_id", "")))

        current_array = np.stack(currents, axis=0)
        current_support_mask_array = (
            np.stack(current_support_masks, axis=0)
            if current_input_mode == CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
            else None
        )
        embedding_array = np.stack(embeddings, axis=0)
        namespace = (
            "common_test_core_05m" if str(panel_name) == "core" else "broad_test_30m"
        )
        embedding_array, text_audit = transform_embedding_matrix(
            embedding_array,
            stable_texts,
            mode=run.text_ablation_mode,
            seed=int(getattr(config, "news_first_text_shuffle_seed", run.seed)),
            namespace=namespace,
            pair_ids=pair_ids,
            session_ids=session_ids,
        )
        predictions: list[np.ndarray] = []
        is_regression = str(run.model).strip().lower() == "regression"
        noise_mode = (
            "not_applicable"
            if is_regression
            else normalize_generator_noise_mode(
                getattr(config, "generator_noise_mode", GAUSSIAN_GENERATOR_NOISE_MODE)
            )
        )
        noise_fingerprint = (
            ""
            if is_regression
            else generator_noise_fingerprint(noise_mode, int(config.noise_dim))
        )
        current_input_fingerprint = (
            ""
            if is_regression
            else generator_current_input_fingerprint(current_input_mode)
        )
        effective_mc_samples = (
            1
            if is_regression or noise_mode == ZERO_GENERATOR_NOISE_MODE
            else int(self.mc_samples)
        )
        model.eval()
        with torch.no_grad():
            for start in range(0, len(panel), self.sample_batch_size):
                stop = min(len(panel), start + self.sample_batch_size)
                current_tensor = torch.tensor(
                    current_array[start:stop], dtype=torch.float32, device=device
                )
                embedding_tensor = torch.tensor(
                    embedding_array[start:stop], dtype=torch.float32, device=device
                )
                current_support_mask_tensor = (
                    None
                    if current_support_mask_array is None
                    else torch.tensor(
                        current_support_mask_array[start:stop],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                if is_regression:
                    predicted = model(current_tensor, embedding_tensor)
                elif noise_mode == ZERO_GENERATOR_NOISE_MODE:
                    call_kwargs: dict[str, Any] = {
                        "noise": torch.zeros(
                            (stop - start, int(config.noise_dim)),
                            dtype=current_tensor.dtype,
                            device=device,
                        )
                    }
                    if current_support_mask_tensor is not None:
                        call_kwargs["current_support_mask"] = (
                            current_support_mask_tensor
                        )
                    predicted = model(current_tensor, embedding_tensor, **call_kwargs)
                else:
                    keys = stable_keys[start:stop]
                    accumulated = torch.zeros_like(current_tensor)
                    completed = 0
                    for draw_start in range(0, self.mc_samples, self.draw_batch_size):
                        draw_stop = min(
                            self.mc_samples, draw_start + self.draw_batch_size
                        )
                        noises = [
                            stable_noise_for_keys(
                                keys,
                                noise_dim=int(config.noise_dim),
                                base_seed=int(run.seed),
                                draw_index=draw_index,
                                device=device,
                                dtype=current_tensor.dtype,
                            )
                            for draw_index in range(draw_start, draw_stop)
                        ]
                        draw_count = len(noises)
                        repeated_current = current_tensor.repeat((draw_count, 1, 1, 1))
                        repeated_embedding = embedding_tensor.repeat((draw_count, 1))
                        call_kwargs = {"noise": torch.cat(noises, dim=0)}
                        if current_support_mask_tensor is not None:
                            call_kwargs["current_support_mask"] = (
                                current_support_mask_tensor.repeat(
                                    (draw_count, 1, 1, 1)
                                )
                            )
                        generated = model(
                            repeated_current,
                            repeated_embedding,
                            **call_kwargs,
                        ).reshape(draw_count, stop - start, *current_tensor.shape[1:])
                        accumulated += generated.sum(dim=0)
                        completed += draw_count
                    predicted = accumulated / float(completed)
                predictions.extend(
                    predicted.detach().cpu().numpy()[:, 0].astype(np.float32)
                )
        return pd.DataFrame(
            {
                "sample_id": sample_ids,
                "news_row_id": news_ids,
                "predicted_surface_flat": [
                    value.reshape(-1).tolist() for value in predictions
                ],
                "prediction_status": "ok",
                "prediction_mc_samples": effective_mc_samples,
                "prediction_fallback": False,
                "generator_noise_mode": noise_mode,
                "generator_noise_fingerprint": noise_fingerprint,
                "generator_current_input_mode": (
                    "not_applicable" if is_regression else current_input_mode
                ),
                "generator_current_input_fingerprint": current_input_fingerprint,
                "current_support_cell_count": current_support_cell_counts,
                "current_support_mask_fingerprint": (current_support_mask_fingerprints),
                "text_ablation_mode": run.text_ablation_mode,
                "text_information_path": text_information_path(run.text_ablation_mode),
                "text_shuffle_mapping_sha256": str(
                    text_audit.get("text_shuffle_mapping_sha256", "")
                ),
            }
        )


_DEFAULT_TRAINED_EVALUATOR: TrainedRunEvaluator | None = None


def evaluate_trained_run(
    run: RunSpec,
    panel_name: str,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Direct production evaluator usable as the ``run_analysis`` callback."""

    global _DEFAULT_TRAINED_EVALUATOR
    if _DEFAULT_TRAINED_EVALUATOR is None:
        _DEFAULT_TRAINED_EVALUATOR = TrainedRunEvaluator()
    return _DEFAULT_TRAINED_EVALUATOR(run, panel_name, panel)


def precomputed_prediction_evaluator(
    run: RunSpec,
    panel_name: str,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Load keyed predictions previously materialized inside a run directory."""

    candidates = [
        run.run_dir / "predictions" / f"{panel_name}.csv.gz",
        run.run_dir / "predictions" / f"{panel_name}_predictions.csv.gz",
        run.run_dir / f"{panel_name}_predictions.csv.gz",
        run.run_dir / "evaluation" / panel_name / "predictions.csv.gz",
        run.run_dir / "predictions" / f"{panel_name}.csv",
        run.run_dir / f"{panel_name}_predictions.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"No precomputed {panel_name!r} predictions found for run {run.run_id}; "
            "supply a model-specific evaluator callable."
        )
    return pd.read_csv(path, low_memory=False)


def _load_panel_source(
    source: str | Path | pd.DataFrame,
    *,
    sheet_name: str,
    panel_name: str,
    freeze_test_window: bool,
    enforce_expected_counts: bool,
    support_mask_mode: str = "none",
    expected_panel_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    source_path: Path | None = None
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
        lineage = {
            "source_type": "dataframe",
            "row_count": int(len(source)),
        }
    else:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Panel workbook does not exist: {source_path}")
        frame = pd.read_excel(source_path, sheet_name=sheet_name)
        lineage = {
            "source_type": "workbook",
            "path": str(source_path),
            "sheet_name": str(sheet_name),
            "sha256": _sha256(source_path),
            "source_row_count": int(len(frame)),
        }

    if freeze_test_window:
        if "effective_origin_utc" not in frame.columns:
            raise ComparisonAnalysisError(
                f"Panel {panel_name!r} lacks effective_origin_utc required for the frozen Q4 test"
            )
        timestamps = pd.to_datetime(
            frame["effective_origin_utc"], errors="coerce", utc=True
        )
        if timestamps.isna().any():
            raise ComparisonAnalysisError(
                f"Panel {panel_name!r} contains unparseable effective_origin_utc values"
            )
        start = pd.Timestamp(TEST_START_UTC)
        end = pd.Timestamp(TEST_END_UTC)
        frame = frame[(timestamps >= start) & (timestamps < end)].copy()
        lineage.update(
            {
                "test_start_utc": TEST_START_UTC,
                "test_end_utc": TEST_END_UTC,
                "test_filter_column": "effective_origin_utc",
            }
        )

    frame = validate_panel(frame, panel_name=panel_name)
    support_mode = str(support_mask_mode or "none").strip().lower()
    pre_support_rows = int(len(frame))
    if support_mode == "raw_joint":
        keep: list[bool] = []
        counts: list[int] = []
        for row in frame.to_dict(orient="records"):
            strike_grid, maturity_grid, _ = _surface_axes(
                row,
                cell_count=len(
                    _parse_vector(
                        row["target_surface_flat"], label="target_surface_flat"
                    )
                ),
            )
            try:
                mask, _, _ = _joint_support_mask(
                    row,
                    strike_grid=strike_grid,
                    maturity_grid=maturity_grid,
                    support_mask_mode=support_mode,
                )
            except ComparisonAnalysisError as exc:
                if "at least one" not in str(exc):
                    raise
                keep.append(False)
                counts.append(0)
            else:
                keep.append(True)
                counts.append(int(mask.sum()))
        frame = frame.loc[np.asarray(keep, dtype=bool)].reset_index(drop=True)
        if frame.empty:
            raise ComparisonAnalysisError(
                f"Panel {panel_name!r} has no positive raw-joint-support samples"
            )
        kept_counts = [count for count in counts if count > 0]
        lineage.update(
            {
                "support_mask_mode": "raw_joint",
                "support_mask_applied_in_evaluation": True,
                "pre_support_filter_rows": pre_support_rows,
                "excluded_zero_joint_support_rows": int(pre_support_rows - len(frame)),
                "supported_cell_count_min": int(min(kept_counts)),
                "supported_cell_count_median": float(np.median(kept_counts)),
                "supported_cell_count_max": int(max(kept_counts)),
                "support_method": SUPPORT_METHOD,
            }
        )
    else:
        lineage.update(
            {
                "support_mask_mode": "none",
                "support_mask_applied_in_evaluation": False,
                "pre_support_filter_rows": pre_support_rows,
                "excluded_zero_joint_support_rows": 0,
            }
        )
    lineage["row_count"] = int(len(frame))
    lineage["pair_count"] = int(frame["pair_id"].nunique())
    lineage["session_count"] = int(frame["session_id"].nunique())
    expected = dict(expected_panel_counts or EXPECTED_PANEL_COUNTS).get(panel_name)
    if enforce_expected_counts and expected:
        actual = {
            "rows": int(len(frame)),
            "pairs": int(frame["pair_id"].nunique()),
            "sessions": int(frame["session_id"].nunique()),
        }
        mismatches = {
            key: (actual[key], int(expected[key]))
            for key in ("rows", "pairs", "sessions")
            if actual[key] != int(expected[key])
        }
        if "dataset_tolerance_minutes" in frame.columns:
            tolerances = set(
                pd.to_numeric(frame["dataset_tolerance_minutes"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            if tolerances != {int(expected["tolerance_minutes"])}:
                mismatches["tolerance_minutes"] = (
                    sorted(tolerances),
                    int(expected["tolerance_minutes"]),
                )
        if mismatches:
            raise ComparisonAnalysisError(
                f"Frozen {panel_name} Q4 panel count/tolerance mismatch: {mismatches}"
            )
        lineage["expected_counts_verified"] = True

    bridge = pd.DataFrame()
    if source_path is not None:
        bridge_path = source_path.parent / "news_atm_ab_maturity_bridge.csv.gz"
        if not bridge_path.is_file():
            raise FileNotFoundError(
                f"ATM/skew maturity bridge required beside panel workbook: {bridge_path}"
            )
        bridge = pd.read_csv(bridge_path, low_memory=False)
        key = _sample_join_key(frame)
        if key not in bridge.columns:
            raise ComparisonAnalysisError(
                f"Maturity bridge is missing panel key {key}: {bridge_path}"
            )
        allowed = set(frame[key].astype(str))
        bridge = bridge[bridge[key].astype(str).isin(allowed)].copy()
        lineage["maturity_bridge_path"] = str(bridge_path.resolve())
        lineage["maturity_bridge_sha256"] = _sha256(bridge_path)
        lineage["maturity_bridge_rows"] = int(len(bridge))
    return frame, lineage, bridge


def _write_csv(frame: pd.DataFrame, path: Path, *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if compressed else None
    frame.to_csv(path, index=False, compression=compression)


def _prediction_export_frame(
    run: RunSpec,
    panel_name: str,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    evaluated_samples: pd.DataFrame,
) -> pd.DataFrame:
    """Build a finite, panel-keyed prediction artifact for one run and panel."""

    panel_key = _sample_join_key(panel)
    prediction_key = _sample_join_key(predictions)
    if panel_key != prediction_key:
        raise ComparisonAnalysisError(
            f"Prediction export key mismatch: panel={panel_key}, predictions={prediction_key}"
        )
    if predictions[prediction_key].duplicated().any():
        raise ComparisonAnalysisError(
            f"Prediction export contains duplicate {prediction_key} values"
        )
    surface_column = _prediction_surface_column(predictions)
    successful_keys = set(evaluated_samples[panel_key].astype(str))
    prediction_keys = set(predictions[prediction_key].astype(str))
    extra_keys = sorted(prediction_keys - set(panel[panel_key].astype(str)))
    if extra_keys:
        raise ComparisonAnalysisError(
            f"Evaluator returned prediction keys outside panel {panel_name}: {extra_keys[:5]}"
        )
    prediction_columns = [prediction_key, surface_column]
    prediction_columns.extend(
        column
        for column in (
            "prediction_status",
            "prediction_mc_samples",
            "prediction_fallback",
            "generator_noise_mode",
            "generator_noise_fingerprint",
            "generator_current_input_mode",
            "generator_current_input_fingerprint",
            "current_support_cell_count",
            "current_support_mask_fingerprint",
            "text_ablation_mode",
            "text_information_path",
            "text_shuffle_mapping_sha256",
        )
        if column in predictions.columns
    )
    selected = predictions.loc[
        predictions[prediction_key].astype(str).isin(successful_keys),
        prediction_columns,
    ].copy()
    metadata_columns = [
        column
        for column in ("sample_id", "news_row_id", "pair_id", "session_id")
        if column in panel.columns
    ]
    metadata = panel[
        [panel_key] + [c for c in metadata_columns if c != panel_key]
    ].copy()
    selected[prediction_key] = selected[prediction_key].astype(str)
    metadata[panel_key] = metadata[panel_key].astype(str)
    merged = metadata.merge(
        selected, on=prediction_key, how="inner", validate="one_to_one"
    )
    if len(merged) != len(evaluated_samples):
        raise ComparisonAnalysisError(
            f"Prediction export row mismatch for {run.run_id}/{panel_name}: "
            f"export={len(merged)} evaluated={len(evaluated_samples)}"
        )
    panel_lengths = {
        str(row[panel_key]): len(
            _parse_vector(row["target_surface_flat"], label="target_surface_flat")
        )
        for row in panel.to_dict(orient="records")
    }
    panel_records = {
        str(row[panel_key]): row for row in panel.to_dict(orient="records")
    }
    serialized: list[str] = []
    serialized_masks: list[str] = []
    support_counts: list[int] = []
    support_fractions: list[float] = []
    support_methods: list[str] = []
    support_fingerprints: list[str] = []
    for raw in merged.to_dict(orient="records"):
        values = _parse_vector(raw[surface_column], label="predicted_surface_flat")
        expected = panel_lengths[str(raw[panel_key])]
        if len(values) != expected:
            raise ComparisonAnalysisError(
                f"Prediction surface width mismatch for {raw[panel_key]}: {len(values)} != {expected}"
            )
        serialized.append(
            json.dumps(values.astype(float).tolist(), separators=(",", ":"))
        )
        panel_row = panel_records[str(raw[panel_key])]
        strike_grid, maturity_grid, _ = _surface_axes(panel_row, cell_count=expected)
        mask, method, fingerprint = _joint_support_mask(
            panel_row,
            strike_grid=strike_grid,
            maturity_grid=maturity_grid,
            support_mask_mode=run.support_mask_mode,
        )
        serialized_masks.append(
            json.dumps(mask.reshape(-1).astype(int).tolist(), separators=(",", ":"))
        )
        support_counts.append(int(mask.sum()))
        support_fractions.append(float(mask.mean()))
        support_methods.append(method)
        support_fingerprints.append(fingerprint)
    merged["predicted_surface_flat"] = serialized
    status = (
        merged["prediction_status"].astype(str)
        if "prediction_status" in merged.columns
        else pd.Series("ok", index=merged.index, dtype=str)
    )
    mc_default = 1 if str(run.model).strip().lower() == "regression" else 64
    mc_samples = (
        pd.to_numeric(merged["prediction_mc_samples"], errors="coerce")
        if "prediction_mc_samples" in merged.columns
        else pd.Series(mc_default, index=merged.index, dtype=int)
    )
    fallback = (
        merged["prediction_fallback"].map(
            lambda value: _coerce_strict_bool(value, label="prediction_fallback")
        )
        if "prediction_fallback" in merged.columns
        else pd.Series(False, index=merged.index, dtype=bool)
    )
    default_noise_mode = (
        "not_applicable"
        if str(run.model).strip().lower() == "regression"
        else "gaussian"
    )
    generator_noise_mode = (
        merged["generator_noise_mode"].astype(str)
        if "generator_noise_mode" in merged.columns
        else pd.Series(default_noise_mode, index=merged.index, dtype=str)
    )
    generator_noise_fingerprint = (
        merged["generator_noise_fingerprint"].astype(str)
        if "generator_noise_fingerprint" in merged.columns
        else pd.Series("", index=merged.index, dtype=str)
    )
    generator_current_input_mode = (
        merged["generator_current_input_mode"].astype(str)
        if "generator_current_input_mode" in merged.columns
        else pd.Series(
            run.generator_current_input_mode,
            index=merged.index,
            dtype=str,
        )
    )
    generator_current_input_fingerprint = (
        merged["generator_current_input_fingerprint"].astype(str)
        if "generator_current_input_fingerprint" in merged.columns
        else pd.Series("", index=merged.index, dtype=str)
    )
    current_support_cell_count = (
        pd.to_numeric(merged["current_support_cell_count"], errors="coerce")
        if "current_support_cell_count" in merged.columns
        else pd.Series(np.nan, index=merged.index, dtype=float)
    )
    current_support_mask_fingerprint = (
        merged["current_support_mask_fingerprint"].astype(str)
        if "current_support_mask_fingerprint" in merged.columns
        else pd.Series("", index=merged.index, dtype=str)
    )
    output = pd.DataFrame(
        {
            "run_id": run.run_id,
            "model": run.model,
            "text_ablation_mode": run.text_ablation_mode,
            "text_information_path": text_information_path(run.text_ablation_mode),
            "support_mask_mode": run.support_mask_mode,
            "generator_current_input_mode": generator_current_input_mode,
            "seed": int(run.seed),
            "tolerance_minutes": int(run.tolerance_minutes),
            "panel": str(panel_name),
            "sample_id": merged.get("sample_id", merged[panel_key]).astype(str),
            "news_row_id": merged.get("news_row_id", ""),
            "pair_id": merged["pair_id"].astype(str),
            "session_id": merged["session_id"].astype(str),
            "predicted_surface_flat": merged["predicted_surface_flat"],
            "support_mask_flat": serialized_masks,
            "supported_cell_count": support_counts,
            "supported_cell_fraction": support_fractions,
            "support_method": support_methods,
            "support_grid_fingerprint": support_fingerprints,
            "text_shuffle_mapping_sha256": merged.get(
                "text_shuffle_mapping_sha256", ""
            ),
            "prediction_status": status,
            "prediction_mc_samples": mc_samples,
            "prediction_fallback": fallback,
            "generator_noise_mode": generator_noise_mode,
            "generator_noise_fingerprint": generator_noise_fingerprint,
            "generator_current_input_fingerprint": (
                generator_current_input_fingerprint
            ),
            "current_support_cell_count": current_support_cell_count,
            "current_support_mask_fingerprint": (current_support_mask_fingerprint),
        }
    )
    return output.sort_values("sample_id", kind="stable").reset_index(drop=True)


def _enforce_formal_run_coverage(
    run: RunSpec,
    panel_name: str,
    panel: pd.DataFrame,
    samples: pd.DataFrame,
    general_exclusions: pd.DataFrame,
    prediction_export: pd.DataFrame,
) -> None:
    """Fail closed unless a formal run covers the frozen panel exactly."""

    if not general_exclusions.empty:
        counts = (
            general_exclusions["exclusion_code"].astype(str).value_counts().to_dict()
        )
        raise ComparisonAnalysisError(
            f"Formal evaluation excluded rows for {run.run_id}/{panel_name}: {counts}"
        )
    expected = {
        "rows": int(len(panel)),
        "pairs": int(panel["pair_id"].nunique()),
        "sessions": int(panel["session_id"].nunique()),
    }
    actual = {
        "rows": int(len(samples)),
        "pairs": int(samples["pair_id"].nunique()),
        "sessions": int(samples["session_id"].nunique()),
    }
    if actual != expected:
        raise ComparisonAnalysisError(
            f"Formal evaluation coverage mismatch for {run.run_id}/{panel_name}: "
            f"actual={actual}, expected={expected}"
        )
    panel_keys = set(panel[_sample_join_key(panel)].astype(str))
    sample_keys = set(samples[_sample_join_key(samples)].astype(str))
    prediction_keys = set(prediction_export["sample_id"].astype(str))
    if panel_keys != sample_keys or panel_keys != prediction_keys:
        raise ComparisonAnalysisError(
            f"Formal stable-key coverage mismatch for {run.run_id}/{panel_name}"
        )
    if not samples["surface_diagnostic_status"].astype(str).eq("ok").all():
        counts = (
            samples["surface_diagnostic_status"].astype(str).value_counts().to_dict()
        )
        raise ComparisonAnalysisError(
            f"Formal surface diagnostics unavailable for {run.run_id}/{panel_name}: {counts}"
        )
    finite_columns = [
        "model_mae",
        "model_rmse",
        "persistence_mae",
        "persistence_rmse",
        "supported_cell_count",
        "supported_cell_fraction",
        "predicted_calendar_violation_count",
        "target_calendar_violation_count",
        "predicted_butterfly_violation_count",
        "target_butterfly_violation_count",
    ]
    missing_columns = sorted(set(finite_columns) - set(samples.columns))
    if missing_columns:
        raise ComparisonAnalysisError(
            f"Formal metrics are missing columns for {run.run_id}/{panel_name}: {missing_columns}"
        )
    non_finite = {}
    for column in finite_columns:
        values = pd.to_numeric(samples[column], errors="coerce").to_numpy(dtype=float)
        count = int((~np.isfinite(values)).sum())
        if count:
            non_finite[column] = count
    if non_finite:
        raise ComparisonAnalysisError(
            f"Formal metrics contain non-finite values for {run.run_id}/{panel_name}: {non_finite}"
        )
    sample_support = samples.set_index("sample_id")["supported_cell_count"].astype(int)
    artifact_support = prediction_export.set_index("sample_id")[
        "supported_cell_count"
    ].astype(int)
    if not sample_support.sort_index().equals(artifact_support.sort_index()):
        raise ComparisonAnalysisError(
            f"Prediction/sample support counts differ for {run.run_id}/{panel_name}"
        )
    if bool((artifact_support <= 0).any()):
        raise ComparisonAnalysisError(
            f"Formal prediction mask is empty for {run.run_id}/{panel_name}"
        )
    panel_lengths = {
        str(row["sample_id"]): len(
            _parse_vector(row["target_surface_flat"], label="target_surface_flat")
        )
        for row in panel.to_dict(orient="records")
    }
    for raw in prediction_export.to_dict(orient="records"):
        mask = _parse_vector(raw["support_mask_flat"], label="support_mask_flat")
        if (
            len(mask) != panel_lengths[str(raw["sample_id"])]
            or not np.isin(mask, [0.0, 1.0]).all()
        ):
            raise ComparisonAnalysisError(
                f"Invalid serialized support mask for {run.run_id}/{raw['sample_id']}"
            )
        if int(mask.sum()) != int(raw["supported_cell_count"]):
            raise ComparisonAnalysisError(
                f"Serialized support-mask count mismatch for {run.run_id}/{raw['sample_id']}"
            )
    if (
        run.support_mask_mode == "raw_joint"
        and not prediction_export["support_method"].astype(str).eq(SUPPORT_METHOD).all()
    ):
        raise ComparisonAnalysisError(
            f"Formal raw-joint support method mismatch for {run.run_id}/{panel_name}"
        )
    normalized_status = (
        prediction_export["prediction_status"].astype(str).str.strip().str.lower()
    )
    if not normalized_status.eq("ok").all():
        raise ComparisonAnalysisError(
            f"Formal prediction artifact contains rejected status for {run.run_id}/{panel_name}"
        )
    if prediction_export["prediction_fallback"].astype(bool).any():
        raise ComparisonAnalysisError(
            f"Formal prediction fallback is forbidden for {run.run_id}/{panel_name}"
        )
    observed_noise_modes = set(
        prediction_export.get(
            "generator_noise_mode",
            pd.Series(
                "not_applicable"
                if str(run.model).strip().lower() == "regression"
                else "gaussian",
                index=prediction_export.index,
                dtype=str,
            ),
        )
        .astype(str)
        .str.strip()
        .str.lower()
        .tolist()
    )
    expected_noise_mode = (
        "not_applicable"
        if str(run.model).strip().lower() == "regression"
        else next(iter(observed_noise_modes), "gaussian")
    )
    if observed_noise_modes != {expected_noise_mode} or expected_noise_mode not in {
        "not_applicable",
        "gaussian",
        "zero",
    }:
        raise ComparisonAnalysisError(
            f"Formal prediction noise-mode mismatch for {run.run_id}/{panel_name}: "
            f"{sorted(observed_noise_modes)}"
        )
    if expected_noise_mode == "zero":
        fingerprints = set(
            prediction_export["generator_noise_fingerprint"]
            .astype(str)
            .str.strip()
            .tolist()
        )
        if (
            len(fingerprints) != 1
            or re.fullmatch(r"[0-9a-f]{64}", next(iter(fingerprints), "")) is None
        ):
            raise ComparisonAnalysisError(
                f"Formal zero-noise fingerprint mismatch for {run.run_id}/{panel_name}: "
                f"{sorted(fingerprints)}"
            )
    expected_current_input_mode = (
        "not_applicable"
        if str(run.model).strip().lower() == "regression"
        else str(run.generator_current_input_mode).strip().lower()
    )
    observed_current_input_modes = set(
        prediction_export["generator_current_input_mode"]
        .astype(str)
        .str.strip()
        .str.lower()
        .tolist()
    )
    if observed_current_input_modes != {expected_current_input_mode}:
        raise ComparisonAnalysisError(
            "Formal prediction generator-current-input mode mismatch for "
            f"{run.run_id}/{panel_name}: {sorted(observed_current_input_modes)} "
            f"!= {[expected_current_input_mode]}"
        )
    if expected_current_input_mode == "current_support_masked":
        fingerprints = prediction_export["generator_current_input_fingerprint"].astype(
            str
        )
        if (
            fingerprints.nunique() != 1
            or re.fullmatch(r"[0-9a-f]{64}", fingerprints.iloc[0]) is None
        ):
            raise ComparisonAnalysisError(
                "Formal current-input contract fingerprint mismatch for "
                f"{run.run_id}/{panel_name}"
            )
        current_counts = pd.to_numeric(
            prediction_export["current_support_cell_count"], errors="coerce"
        )
        mask_fingerprints = prediction_export[
            "current_support_mask_fingerprint"
        ].astype(str)
        if (
            current_counts.isna().any()
            or bool((current_counts <= 0).any())
            or not mask_fingerprints.map(
                lambda value: re.fullmatch(r"[0-9a-f]{64}", value) is not None
            ).all()
        ):
            raise ComparisonAnalysisError(
                f"Formal current-support lineage is invalid for {run.run_id}/{panel_name}"
            )
    expected_mc = 1 if expected_noise_mode in {"not_applicable", "zero"} else 64
    observed_mc = set(
        pd.to_numeric(prediction_export["prediction_mc_samples"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    if observed_mc != {expected_mc}:
        raise ComparisonAnalysisError(
            f"Formal prediction MC count mismatch for {run.run_id}/{panel_name}: "
            f"{sorted(observed_mc)} != {[expected_mc]}"
        )


def collect_training_curves(specs: Sequence[RunSpec]) -> pd.DataFrame:
    """Collect trainer-produced epoch metrics without interpreting trainer internals."""

    parts: list[pd.DataFrame] = []
    for spec in specs:
        candidates = [
            spec.run_dir / "metrics" / "training_metrics.csv",
            spec.run_dir / "training_metrics.csv",
            spec.run_dir / "metrics" / "training_metrics.json",
            spec.run_dir / "training_metrics.json",
        ]
        path = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if path is None:
            continue
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, low_memory=False)
        else:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, Mapping):
                payload = payload.get("metrics", payload.get("epochs", []))
            frame = pd.DataFrame(payload)
        if frame.empty:
            continue
        frame.insert(0, "run_id", spec.run_id)
        frame.insert(1, "model", spec.model)
        frame.insert(2, "text_ablation_mode", spec.text_ablation_mode)
        frame.insert(3, "support_mask_mode", spec.support_mask_mode)
        frame.insert(
            4,
            "generator_current_input_mode",
            spec.generator_current_input_mode,
        )
        frame.insert(5, "seed", int(spec.seed))
        frame.insert(6, "tolerance_minutes", int(spec.tolerance_minutes))
        frame["source_metrics_path"] = str(path.resolve())
        parts.append(frame)
    if not parts:
        return pd.DataFrame(
            columns=[
                "run_id",
                "model",
                "text_ablation_mode",
                "support_mask_mode",
                "generator_current_input_mode",
                "seed",
                "tolerance_minutes",
                "epoch",
                "source_metrics_path",
            ]
        )
    return pd.concat(parts, ignore_index=True, sort=False)


def collect_checkpoint_summary(specs: Sequence[RunSpec]) -> pd.DataFrame:
    """Collect the trainer's authoritative best-checkpoint selection metadata."""

    rows: list[dict[str, Any]] = []
    for spec in specs:
        candidates = (
            spec.run_dir / "metrics" / "best_checkpoint.json",
            spec.run_dir / "best_checkpoint.json",
        )
        path = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        base = {
            "run_id": spec.run_id,
            "model": spec.model,
            "text_ablation_mode": spec.text_ablation_mode,
            "support_mask_mode": spec.support_mask_mode,
            "generator_current_input_mode": spec.generator_current_input_mode,
            "seed": int(spec.seed),
            "tolerance_minutes": int(spec.tolerance_minutes),
        }
        if path is None:
            rows.append(
                base
                | {
                    "monitor_metric": "",
                    "best_epoch": np.nan,
                    "best_metric": np.nan,
                    "metadata_path": "",
                    "status": "missing",
                }
            )
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ComparisonAnalysisError(
                f"Best-checkpoint metadata must be an object: {path}"
            )
        best_epoch = pd.to_numeric(payload.get("best_epoch"), errors="coerce")
        if not math.isfinite(float(best_epoch)) or float(best_epoch) < 0:
            raise ComparisonAnalysisError(
                f"Best-checkpoint metadata has invalid best_epoch: {path}"
            )
        rows.append(
            base
            | {
                "monitor_metric": str(payload.get("monitor_metric", "")),
                "best_epoch": int(best_epoch),
                "best_metric": _coerce_optional_float(payload.get("best_metric")),
                "metadata_path": path.relative_to(spec.run_dir).as_posix(),
                "status": "ok",
            }
        )
    frame = pd.DataFrame(rows)
    for column in CHECKPOINT_SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[list(CHECKPOINT_SUMMARY_COLUMNS)]


def _schema_payload() -> dict[str, Any]:
    return {
        "numerical_tie_policy": {
            "mae_gap_tolerance_iv": MAE_NUMERICAL_TIE_TOLERANCE,
            "raw_values_preserved": ["model_mae", "persistence_mae", "mae_gap"],
            "win_definition": "mae_gap < -mae_gap_tolerance_iv",
            "tie_definition": "abs(mae_gap) <= mae_gap_tolerance_iv",
            "bootstrap_rule": "paired differences inside the tolerance are canonicalized to zero",
        },
        "sample_metrics.csv.gz": {
            "grain": "run x panel x news/article sample",
            "required_columns": list(SAMPLE_METRIC_COLUMNS),
            "tie_columns": {
                "mae_tie": "abs(mae_gap) <= 1e-8 IV",
                "short_atm_tie": "abs(short_atm_mae_gap) <= 1e-8 IV",
            },
        },
        "maturity_metrics.csv.gz": {
            "grain": "run x panel x news/article x actual pair maturity",
            "required_columns": list(MATURITY_METRIC_COLUMNS),
            "selection": "ATM quality A/B and, independently, skew quality A/B",
        },
        "pair_metrics.csv.gz": {
            "grain": "run x panel x stratum x unique market pair",
            "required_columns": list(PAIR_METRIC_COLUMNS),
            "weighting": "sample_weight normalized within pair and stratum; pairs are equal downstream",
        },
        "model_comparison.csv": {
            "grain": "model x tolerance x panel x stratum",
            "required_columns": list(MODEL_COMPARISON_COLUMNS),
            "win_and_tie_rates": "pair-balanced rates using the 1e-8 IV numerical tie tolerance",
        },
        "stratified_metrics.csv": {
            "grain": "model x tolerance x panel x non-overall audit stratum",
            "source": "model_comparison rows where stratum_type != overall",
        },
        "training_curves.csv": {
            "grain": "run x trainer-emitted epoch",
            "source": "training_metrics.csv/json discovered under each run directory",
        },
        "checkpoint_summary.csv": {
            "grain": "one row per formal run",
            "required_columns": list(CHECKPOINT_SUMMARY_COLUMNS),
            "source": "trainer-emitted metrics/best_checkpoint.json",
        },
        "bootstrap_comparisons.csv": {
            "grain": "WGAN focal-vs-5m on core x overall/all x model_mae",
            "required_columns": list(BOOTSTRAP_COMPARISON_COLUMNS),
            "difference": "focal tolerance minus 5m on common pair_id",
            "resampling": "whole CME session, pair-balanced statistic",
            "role": "primary confirmatory comparison; exactly 10/15/30m vs 5m",
        },
        "bootstrap_sensitivity.csv": {
            "grain": "WGAN focal-vs-5m x secondary panel/metric/stratum",
            "required_columns": list(BOOTSTRAP_COMPARISON_COLUMNS),
            "role": "secondary/sensitivity only",
        },
        "text_ablation_comparisons.csv": {
            "grain": "model x tolerance x current-only/shuffled-text control",
            "required_columns": list(TEXT_ABLATION_COMPARISON_COLUMNS),
            "difference": "real_text model_mae minus control model_mae on identical core-Q4 pairs",
            "interpretation": "negative means real news text adds predictive value relative to the control",
            "resampling": "10,000 whole-CME-session cluster-bootstrap draws; one global Holm family across all 16 tests",
        },
        "predictions/{run_id}_{panel}.csv.gz": {
            "grain": "run x panel x stable article sample",
            "required_columns": [
                "run_id",
                "model",
                "text_ablation_mode",
                "text_information_path",
                "support_mask_mode",
                "generator_current_input_mode",
                "seed",
                "tolerance_minutes",
                "panel",
                "sample_id",
                "news_row_id",
                "pair_id",
                "session_id",
                "predicted_surface_flat",
                "support_mask_flat",
                "supported_cell_count",
                "support_method",
                "support_grid_fingerprint",
                "text_shuffle_mapping_sha256",
                "prediction_status",
                "prediction_mc_samples",
                "prediction_fallback",
                "generator_noise_mode",
                "generator_noise_fingerprint",
                "generator_current_input_fingerprint",
                "current_support_cell_count",
                "current_support_mask_fingerprint",
            ],
            "foreign_key": "sample_id -> frozen panel sample_id",
        },
        "analysis_config.json": {
            "grain": "one resolved post-training analysis specification",
            "role": "freezes test window, structural metrics, MC, coverage, and bootstrap choices",
        },
        "evaluation_exclusions.csv.gz": {
            "grain": "run x panel x excluded sample",
        },
        "atm_skew_exclusions.csv.gz": {
            "grain": "run x panel x sample x actual maturity x ATM/skew metric family",
            "rule": "interpolation inside strike/maturity grid only; no extrapolation and no implicit q=1 ATM",
        },
    }


def run_analysis(
    run_dirs: Sequence[str | Path | RunSpec],
    core_workbook: str | Path | pd.DataFrame,
    broad_workbook: str | Path | pd.DataFrame,
    evaluator: PredictionEvaluator,
    output_dir: str | Path,
    *,
    core_sheet: str = "gan_input_ready",
    broad_sheet: str = "gan_input_ready",
    expected_run_count: int | None = 8,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    comparison_models: Sequence[str] | None = None,
    freeze_test_window: bool = True,
    enforce_expected_panel_counts: bool = True,
) -> Path:
    """Evaluate runs on fixed panels and write all formal comparison tables."""

    specs = [
        item if isinstance(item, RunSpec) else discover_run_spec(item)
        for item in run_dirs
    ]
    if expected_run_count is not None and len(specs) != int(expected_run_count):
        raise ComparisonAnalysisError(
            f"Expected {expected_run_count} run directories/specs, received {len(specs)}"
        )
    run_ids = [spec.run_id for spec in specs]
    if len(set(run_ids)) != len(run_ids):
        raise ComparisonAnalysisError("run_id values must be unique")
    support_modes = {str(spec.support_mask_mode).strip().lower() for spec in specs}
    if len(support_modes) != 1:
        raise ComparisonAnalysisError(
            f"All compared runs must use one support-mask mode, got {sorted(support_modes)}"
        )
    support_mode = next(iter(support_modes), "none")
    expected_counts = (
        EXPECTED_RAW_JOINT_PANEL_COUNTS
        if support_mode == "raw_joint"
        else EXPECTED_PANEL_COUNTS
    )
    panels: dict[str, pd.DataFrame] = {}
    maturity_bridges: dict[str, pd.DataFrame] = {}
    panel_lineage: dict[str, Any] = {}
    panels["core"], panel_lineage["core"], maturity_bridges["core"] = (
        _load_panel_source(
            core_workbook,
            sheet_name=core_sheet,
            panel_name="core",
            freeze_test_window=bool(freeze_test_window),
            enforce_expected_counts=bool(enforce_expected_panel_counts),
            support_mask_mode=support_mode,
            expected_panel_counts=expected_counts,
        )
    )
    panels["broad"], panel_lineage["broad"], maturity_bridges["broad"] = (
        _load_panel_source(
            broad_workbook,
            sheet_name=broad_sheet,
            panel_name="broad",
            freeze_test_window=bool(freeze_test_window),
            enforce_expected_counts=bool(enforce_expected_panel_counts),
            support_mask_mode=support_mode,
            expected_panel_counts=expected_counts,
        )
    )

    sample_parts: list[pd.DataFrame] = []
    general_exclusion_parts: list[pd.DataFrame] = []
    metric_exclusion_parts: list[pd.DataFrame] = []
    maturity_parts: list[pd.DataFrame] = []
    prediction_exports: dict[tuple[str, str], pd.DataFrame] = {}
    coverage_rows: list[dict[str, Any]] = []
    formal_mode = bool(freeze_test_window and enforce_expected_panel_counts)
    for spec in specs:
        for panel_name, panel in panels.items():
            predictions = evaluator(spec, panel_name, panel.copy())
            samples, general_exclusions, metric_exclusions = compute_sample_metrics(
                spec,
                panel_name,
                panel,
                predictions,
                evaluate_embedded_atm_skew=maturity_bridges[panel_name].empty,
            )
            if not maturity_bridges[panel_name].empty:
                maturity_metrics, maturity_exclusions = compute_maturity_metrics(
                    spec,
                    panel_name,
                    panel,
                    predictions,
                    maturity_bridges[panel_name],
                )
                maturity_parts.append(maturity_metrics)
                samples = attach_maturity_summaries(samples, maturity_metrics)
                if not maturity_exclusions.empty:
                    metric_exclusions = pd.concat(
                        [metric_exclusions, maturity_exclusions],
                        ignore_index=True,
                    )
            prediction_export = _prediction_export_frame(
                spec,
                panel_name,
                panel,
                predictions,
                samples,
            )
            if formal_mode:
                _enforce_formal_run_coverage(
                    spec,
                    panel_name,
                    panel,
                    samples,
                    general_exclusions,
                    prediction_export,
                )
            prediction_exports[(spec.run_id, panel_name)] = prediction_export
            sample_parts.append(samples)
            if not general_exclusions.empty:
                general_exclusion_parts.append(general_exclusions)
            if not metric_exclusions.empty:
                metric_exclusion_parts.append(metric_exclusions)
            coverage_rows.append(
                {
                    "run_id": spec.run_id,
                    "model": spec.model,
                    "text_ablation_mode": spec.text_ablation_mode,
                    "support_mask_mode": spec.support_mask_mode,
                    "generator_current_input_mode": (spec.generator_current_input_mode),
                    "tolerance_minutes": int(spec.tolerance_minutes),
                    "seed": int(spec.seed),
                    "panel": panel_name,
                    "panel_rows": int(len(panel)),
                    "evaluated_rows": int(len(samples)),
                    "excluded_rows": int(len(general_exclusions)),
                    "evaluated_pairs": int(samples["pair_id"].nunique())
                    if not samples.empty
                    else 0,
                    "evaluated_sessions": int(samples["session_id"].nunique())
                    if not samples.empty
                    else 0,
                    "prediction_rows": int(len(prediction_export)),
                    "full_panel_coverage_verified": bool(formal_mode),
                    "supported_cell_count_min": int(
                        pd.to_numeric(
                            samples["supported_cell_count"], errors="raise"
                        ).min()
                    ),
                    "supported_cell_count_median": float(
                        pd.to_numeric(
                            samples["supported_cell_count"], errors="raise"
                        ).median()
                    ),
                    "supported_cell_count_max": int(
                        pd.to_numeric(
                            samples["supported_cell_count"], errors="raise"
                        ).max()
                    ),
                    "short_atm_available_rows": int(
                        samples["short_atm_metric_status"].astype(str).eq("ok").sum()
                    ),
                    "calendar_available_rows": int(
                        samples["calendar_metric_status"].astype(str).eq("ok").sum()
                    ),
                    "butterfly_available_rows": int(
                        samples["butterfly_metric_status"].astype(str).eq("ok").sum()
                    ),
                }
            )

    sample_metrics = (
        pd.concat(sample_parts, ignore_index=True)
        if sample_parts
        else pd.DataFrame(columns=SAMPLE_METRIC_COLUMNS)
    )
    maturity_metrics = (
        pd.concat(maturity_parts, ignore_index=True)
        if maturity_parts
        else pd.DataFrame(columns=MATURITY_METRIC_COLUMNS)
    )
    pair_metrics = aggregate_pair_metrics(sample_metrics)
    model_comparison = build_model_comparison(pair_metrics)
    stratified_metrics = model_comparison[
        ~model_comparison["stratum_type"].astype(str).eq("overall")
    ].copy()
    training_curves = collect_training_curves(specs)
    checkpoint_summary = collect_checkpoint_summary(specs)
    all_bootstrap = build_bootstrap_comparisons(
        pair_metrics,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
        comparison_models=comparison_models,
    )
    selected_primary = select_primary_tolerance_bootstrap(all_bootstrap)
    bootstrap_sensitivity = all_bootstrap.drop(
        index=selected_primary.index
    ).reset_index(drop=True)
    bootstrap = selected_primary.reset_index(drop=True)
    text_ablation_comparisons = build_text_ablation_comparisons(
        pair_metrics,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )
    if formal_mode:
        checkpoint_ok = checkpoint_summary["status"].astype(str).eq("ok")
        if len(checkpoint_summary) != len(specs) or not checkpoint_ok.all():
            raise ComparisonAnalysisError(
                "Formal analysis requires valid best-checkpoint metadata for every run"
            )
        focal_values = set(
            pd.to_numeric(bootstrap["focal_tolerance_minutes"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
        if len(bootstrap) != 3 or focal_values != set(DEFAULT_FOCAL_TOLERANCES):
            raise ComparisonAnalysisError(
                "Formal primary bootstrap must contain exactly WGAN core/overall/model_mae "
                f"10/15/30m-vs-5m rows; found rows={len(bootstrap)}, focal={sorted(focal_values)}"
            )
        formal_modes = {
            normalize_text_ablation_mode(spec.text_ablation_mode) for spec in specs
        }
        if formal_modes == {REAL_TEXT, CURRENT_ONLY, TEXT_SHUFFLE}:
            if len(text_ablation_comparisons) != 16:
                raise ComparisonAnalysisError(
                    "Formal text ablation requires exactly 16 real-vs-control rows; "
                    f"found {len(text_ablation_comparisons)}"
                )
    general_exclusions = (
        pd.concat(general_exclusion_parts, ignore_index=True)
        if general_exclusion_parts
        else pd.DataFrame(
            columns=[
                "run_id",
                "model",
                "seed",
                "tolerance_minutes",
                "panel",
                "sample_id",
                "news_row_id",
                "pair_id",
                "session_id",
                "exclusion_code",
                "detail",
            ]
        )
    )
    metric_exclusions = (
        pd.concat(metric_exclusion_parts, ignore_index=True)
        if metric_exclusion_parts
        else pd.DataFrame(
            columns=[
                "run_id",
                "model",
                "seed",
                "tolerance_minutes",
                "panel",
                "sample_id",
                "news_row_id",
                "pair_id",
                "session_id",
                "metric_family",
                "exclusion_code",
                "detail",
            ]
        )
    )

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(sample_metrics, output / "sample_metrics.csv.gz", compressed=True)
    _write_csv(maturity_metrics, output / "maturity_metrics.csv.gz", compressed=True)
    _write_csv(pair_metrics, output / "pair_metrics.csv.gz", compressed=True)
    _write_csv(model_comparison, output / "model_comparison.csv", compressed=False)
    _write_csv(stratified_metrics, output / "stratified_metrics.csv", compressed=False)
    _write_csv(training_curves, output / "training_curves.csv", compressed=False)
    _write_csv(
        checkpoint_summary,
        output / "checkpoint_summary.csv",
        compressed=False,
    )
    _write_csv(bootstrap, output / "bootstrap_comparisons.csv", compressed=False)
    _write_csv(
        bootstrap_sensitivity,
        output / "bootstrap_sensitivity.csv",
        compressed=False,
    )
    _write_csv(
        text_ablation_comparisons,
        output / "text_ablation_comparisons.csv",
        compressed=False,
    )
    _write_csv(
        general_exclusions, output / "evaluation_exclusions.csv.gz", compressed=True
    )
    _write_csv(
        metric_exclusions, output / "atm_skew_exclusions.csv.gz", compressed=True
    )
    _write_csv(
        pd.DataFrame(coverage_rows),
        output / "evaluation_coverage.csv",
        compressed=False,
    )
    prediction_paths: dict[str, str] = {}
    for (run_id, panel_name), frame in sorted(prediction_exports.items()):
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id)).strip("._") or "run"
        relative = Path("predictions") / f"{safe_run_id}_{panel_name}.csv.gz"
        if str(relative) in prediction_paths.values():
            raise ComparisonAnalysisError(
                f"Prediction artifact filename collision: {relative}"
            )
        _write_csv(frame, output / relative, compressed=True)
        prediction_paths[f"{run_id}|{panel_name}"] = str(relative)
    with (output / "output_schema.json").open("w", encoding="utf-8") as handle:
        json.dump(_schema_payload(), handle, indent=2, ensure_ascii=False)
    short_atm_cell_counts = _observed_short_atm_cell_counts(sample_metrics)
    first_panel_row = panels["core"].iloc[0].to_dict()
    first_target = _parse_vector(
        first_panel_row["target_surface_flat"], label="target_surface_flat"
    )
    first_strikes, first_maturities, _ = _surface_axes(
        first_panel_row, cell_count=len(first_target)
    )
    configured_short_atm_cell_count = int(
        (
            (first_maturities.reshape(-1, 1) <= SHORT_ATM_MAX_BUSINESS_DAYS + 1.0e-6)
            & (first_strikes.reshape(1, -1) >= SHORT_ATM_Q_MIN - 1.0e-6)
            & (first_strikes.reshape(1, -1) <= SHORT_ATM_Q_MAX + 1.0e-6)
        ).sum()
    )
    analysis_config = {
        "test_start_utc": TEST_START_UTC,
        "test_end_utc": TEST_END_UTC,
        "expected_panel_counts": expected_counts,
        "support_mask_mode": support_mode,
        "support_mask_applied_in_training_and_evaluation": support_mode == "raw_joint",
        "formal_full_coverage_required": bool(formal_mode),
        "numerical_tie_policy": {
            "mae_gap_tolerance_iv": MAE_NUMERICAL_TIE_TOLERANCE,
            "tie_definition": "abs(model_mae - persistence_mae) <= tolerance",
            "win_definition": "model_mae < persistence_mae - tolerance",
            "raw_mae_and_gap_preserved": True,
            "bootstrap_differences_within_tolerance_set_to_zero": True,
        },
        "short_atm": {
            "q_min": SHORT_ATM_Q_MIN,
            "q_max": SHORT_ATM_Q_MAX,
            "max_business_days": SHORT_ATM_MAX_BUSINESS_DAYS,
            "boundary_tolerance": 1.0e-6,
            "configured_grid_candidate_cell_count": configured_short_atm_cell_count,
            "observed_cell_counts_by_panel": short_atm_cell_counts,
        },
        "arbitrage": {
            "violation_tolerance": ARBITRAGE_VIOLATION_TOLERANCE,
            "calendar_definition": "adjacent non-decreasing total variance sigma^2*(business_days/365)",
            "butterfly_definition": "non-negative equally-spaced strike second difference of Black-76 relative calls",
        },
        "bootstrap": {
            "iterations": int(bootstrap_iterations),
            "seed": int(bootstrap_seed),
            "primary_scope": "WGAN core panel, overall/all, model_mae, 10/15/30m vs 5m",
            "text_ablation_scope": (
                "real_text minus current_only/text_shuffle model_mae on identical core pairs; "
                "negative means real text better"
            ),
            "cluster": "CME session_id",
        },
        "wgan_prediction_mc_samples": {
            "gaussian": 64,
            "zero": 1,
        },
        "prediction_fallback_allowed": False,
    }
    with (output / "analysis_config.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis_config, handle, indent=2, ensure_ascii=False)
    analysis_output_names = (
        "sample_metrics.csv.gz",
        "maturity_metrics.csv.gz",
        "pair_metrics.csv.gz",
        "model_comparison.csv",
        "stratified_metrics.csv",
        "training_curves.csv",
        "checkpoint_summary.csv",
        "bootstrap_comparisons.csv",
        "bootstrap_sensitivity.csv",
        "text_ablation_comparisons.csv",
        "evaluation_exclusions.csv.gz",
        "atm_skew_exclusions.csv.gz",
        "evaluation_coverage.csv",
        "output_schema.json",
        "analysis_config.json",
    )
    hash_paths = [output / name for name in analysis_output_names]
    hash_paths.extend(output / relative for relative in prediction_paths.values())
    output_hashes = {
        str(path.relative_to(output)): _sha256(path) for path in sorted(hash_paths)
    }
    manifest = {
        "status": "pass",
        "run_count": int(len(specs)),
        "runs": [spec.as_record() for spec in specs],
        "panels": panel_lineage,
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "base_tolerance_minutes": DEFAULT_BASE_TOLERANCE_MINUTES,
        "focal_tolerances": list(DEFAULT_FOCAL_TOLERANCES),
        "comparison_models": None
        if comparison_models is None
        else list(comparison_models),
        "formal_full_coverage_verified": bool(formal_mode),
        "prediction_artifacts": prediction_paths,
        "output_sha256": output_hashes,
        "row_counts": {
            "sample_metrics": int(len(sample_metrics)),
            "maturity_metrics": int(len(maturity_metrics)),
            "pair_metrics": int(len(pair_metrics)),
            "model_comparison": int(len(model_comparison)),
            "stratified_metrics": int(len(stratified_metrics)),
            "training_curves": int(len(training_curves)),
            "checkpoint_summary": int(len(checkpoint_summary)),
            "bootstrap_comparisons": int(len(bootstrap)),
            "bootstrap_sensitivity": int(len(bootstrap_sensitivity)),
            "text_ablation_comparisons": int(len(text_ablation_comparisons)),
            "prediction_rows": int(
                sum(len(frame) for frame in prediction_exports.values())
            ),
            "evaluation_exclusions": int(len(general_exclusions)),
            "atm_skew_exclusions": int(len(metric_exclusions)),
        },
        "methodology": {
            "article_weighting": "sample_weight normalized within pair/stratum",
            "pair_weighting": "equal",
            "bootstrap_cluster": "CME session_id",
            "bootstrap_ci": "percentile 95%",
            "bootstrap_p": "two-sided centered cluster bootstrap",
            "multiple_testing": "Holm within model/panel/metric/stratum 10/15/30m-vs-5m family",
            "text_ablation_estimand": (
                "real_text model_mae minus control model_mae; negative indicates "
                "incremental predictive value from real news text"
            ),
            "text_ablation_multiple_testing": (
                "one global Holm family across 2 models x 4 tolerances x 2 controls"
            ),
            "numerical_ties": (
                f"raw model/persistence MAE and gap are retained; abs(gap) <= "
                f"{MAE_NUMERICAL_TIE_TOLERANCE:g} IV is a tie, win requires gap < "
                f"-{MAE_NUMERICAL_TIE_TOLERANCE:g} IV, and paired bootstrap/text "
                "differences within this tolerance are canonicalized to zero"
            ),
            "support_mask": (
                "raw_joint excludes zero-support samples, then applies the current/target "
                "raw-support intersection cell-wise to model, persistence, short-ATM, "
                "ATM/skew interpolation and structural diagnostics"
                if support_mode == "raw_joint"
                else "none (all grid cells evaluated)"
            ),
            "support_mask_input_scope": (
                "The raw_joint mask is applied to objectives, critic/gradient penalty and "
                "formal evaluation, while generator/regression forward receives the complete "
                "current surface. The joint mask contains target-support information and is "
                "therefore not used to mask model inputs; current inputs may still contain "
                "extrapolated cells."
                if support_mode == "raw_joint"
                else "No support mask is applied to objectives, evaluation or inputs."
            ),
            "atm_definition": "explicit approximate-ATM q supplied by input; never implicit q=1",
            "interpolation": "bilinear within grid; extrapolation forbidden",
            "short_atm": (
                f"grid cells q in [{SHORT_ATM_Q_MIN}, {SHORT_ATM_Q_MAX}] and "
                f"maturity_days <= {SHORT_ATM_MAX_BUSINESS_DAYS}; "
                f"{_format_short_atm_cell_counts(short_atm_cell_counts)}"
            ),
            "arbitrage": (
                "active-trainer definitions: total-variance calendar monotonicity and "
                "Black-76 call-price discrete butterfly convexity"
            ),
            "primary_bootstrap_scope": "WGAN core/overall/all model_mae only; all others are sensitivity",
        },
    }
    with (output / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return output


def _experiment_run_specs(experiment_root: Path) -> list[RunSpec]:
    registry_path = experiment_root / "registry" / "jobs.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"Experiment registry does not exist: {registry_path}")
    registry = _read_mapping(registry_path)
    jobs = registry.get("jobs")
    if not isinstance(jobs, list):
        raise ComparisonAnalysisError("Experiment registry jobs must be a list")
    specs: list[RunSpec] = []
    for job in jobs:
        if not isinstance(job, Mapping):
            raise ComparisonAnalysisError(
                "Experiment registry contains a non-mapping job"
            )
        job_id = str(job.get("job_id", ""))
        status_path = experiment_root / "registry" / "jobs" / f"{job_id}.status.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"Job status is missing: {status_path}")
        status = _read_mapping(status_path)
        if str(status.get("status", "")) != "completed":
            raise ComparisonAnalysisError(
                f"All registered jobs must be completed before analysis: {job_id}={status.get('status')}"
            )
        run_dir = Path(str(status.get("run_dir", ""))).resolve()
        family = str(job.get("model_family", "")).strip().lower()
        if family == "wgan":
            checkpoint = run_dir / "checkpoints" / "generator_best.pt"
        elif family == "regression":
            checkpoint = run_dir / "checkpoints" / "vol_regressor_best.pt"
        else:
            raise ComparisonAnalysisError(
                f"Unknown experiment model family: {family!r}"
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Best checkpoint is missing for {job_id}: {checkpoint}"
            )
        resolved_training = run_dir / "metrics" / "training_resolved_config.yaml"
        config_payload = (
            _read_mapping(resolved_training) if resolved_training.is_file() else {}
        )
        seed = int(_recursive_first(config_payload, ("seed",)) or 42)
        text_mode = normalize_text_ablation_mode(
            job.get("text_ablation_mode")
            or _recursive_first(
                config_payload,
                ("news_first_text_ablation_mode", "text_ablation_mode"),
            )
            or REAL_TEXT
        )
        support_mode = (
            str(
                job.get("support_mask_mode")
                or _recursive_first(config_payload, ("support_mask_mode",))
                or "none"
            )
            .strip()
            .lower()
        )
        current_input_mode = (
            str(
                job.get("generator_current_input_mode")
                or _recursive_first(
                    config_payload,
                    ("generator_current_input_mode",),
                )
                or "full_current"
            )
            .strip()
            .lower()
        )
        specs.append(
            RunSpec(
                run_id=job_id,
                run_dir=run_dir,
                model=family,
                tolerance_minutes=int(job["tolerance_minutes"]),
                seed=seed,
                checkpoint_path=checkpoint.resolve(),
                text_ablation_mode=text_mode,
                support_mask_mode=support_mode,
                generator_current_input_mode=current_input_mode,
                manifest_path=resolved_training.resolve()
                if resolved_training.is_file()
                else status_path.resolve(),
                metadata={"job": dict(job), "status": status},
            )
        )
    specs.sort(
        key=lambda item: (
            item.model,
            item.text_ablation_mode,
            int(item.tolerance_minutes),
            item.run_id,
        )
    )
    if len(specs) != len(jobs):
        raise ComparisonAnalysisError(
            f"Expected all {len(jobs)} registered jobs, found {len(specs)}"
        )
    modes = {spec.text_ablation_mode for spec in specs}
    if modes == {REAL_TEXT, CURRENT_ONLY, TEXT_SHUFFLE}:
        observed = {
            (spec.model, spec.text_ablation_mode, int(spec.tolerance_minutes))
            for spec in specs
        }
        expected = {
            (model, mode, tolerance)
            for model in ("wgan", "regression")
            for mode in (REAL_TEXT, CURRENT_ONLY, TEXT_SHUFFLE)
            for tolerance in (5, 10, 15, 30)
        }
        if observed != expected:
            raise ComparisonAnalysisError(
                "Formal text-ablation registry must contain the complete 24-job "
                f"model x mode x tolerance product; missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )
    return specs


def run_analysis_from_experiment(
    experiment_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    evaluator: PredictionEvaluator | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Path:
    """Run formal Q4 analysis directly from every registered experiment job."""

    root = Path(experiment_root).resolve()
    resolved_path = root / "resolved_config.yaml"
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Resolved experiment config is missing: {resolved_path}"
        )
    resolved_payload = _read_mapping(resolved_path)
    config = resolved_payload.get("news_first_vol_training", resolved_payload)
    if not isinstance(config, Mapping) or not isinstance(
        config.get("datasets"), Mapping
    ):
        raise ComparisonAnalysisError(
            "Resolved experiment config lacks datasets mapping"
        )
    datasets = dict(config["datasets"])
    dataset_root = Path(str(datasets["root"])).resolve()
    template = str(
        datasets.get("workbook_template", "tolerance_{tolerance02}m/merged_vol.xlsx")
    )

    def workbook(tolerance: int) -> Path:
        return dataset_root / template.format(
            tolerance=int(tolerance), tolerance02=f"{int(tolerance):02d}"
        )

    specs = _experiment_run_specs(root)
    production_evaluator = evaluator or TrainedRunEvaluator(
        mc_samples=64,
        sample_batch_size=32,
        draw_batch_size=64,
    )
    return run_analysis(
        specs,
        workbook(5),
        workbook(30),
        production_evaluator,
        output_dir or (root / "analysis"),
        core_sheet=str(datasets.get("sheet_name", "gan_input_ready")),
        broad_sheet=str(datasets.get("sheet_name", "gan_input_ready")),
        expected_run_count=len(specs),
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
        comparison_models=["wgan"],
        freeze_test_window=True,
        enforce_expected_panel_counts=True,
    )


def run_experiment_analysis(experiment_root: str | Path) -> Path:
    """Launcher-friendly alias using all frozen production defaults."""

    return run_analysis_from_experiment(experiment_root)


def _load_dotted_evaluator(value: str) -> PredictionEvaluator:
    normalized = str(value).strip()
    if normalized in {"", "precomputed"}:
        return precomputed_prediction_evaluator
    if normalized == "trained":
        return evaluate_trained_run
    if ":" not in normalized:
        raise ComparisonAnalysisError(
            "--evaluator must be 'trained', 'precomputed' or module.path:function"
        )
    module_name, function_name = normalized.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise ComparisonAnalysisError(f"Evaluator is not callable: {normalized}")
    return function


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate eight news-first vol runs on fixed core/broad panels.",
    )
    parser.add_argument("--experiment-root", default="")
    parser.add_argument(
        "--run-dir", action="append", help="Repeat exactly eight times."
    )
    parser.add_argument("--core-workbook")
    parser.add_argument("--broad-workbook")
    parser.add_argument("--core-sheet", default="gan_input_ready")
    parser.add_argument("--broad-sheet", default="gan_input_ready")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--evaluator",
        default="trained",
        help="trained, precomputed, or an importable module.path:function callable.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--comparison-model", action="append", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    evaluator: PredictionEvaluator = (
        TrainedRunEvaluator(device=args.device)
        if args.evaluator == "trained"
        else _load_dotted_evaluator(args.evaluator)
    )
    if args.experiment_root:
        if args.run_dir or args.core_workbook or args.broad_workbook:
            raise ComparisonAnalysisError(
                "--experiment-root cannot be combined with explicit run/workbook arguments"
            )
        output = run_analysis_from_experiment(
            args.experiment_root,
            output_dir=args.output_dir or None,
            evaluator=evaluator,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(f"News-first vol comparison outputs written to {output}")
        return 0
    if (
        not args.run_dir
        or not args.core_workbook
        or not args.broad_workbook
        or not args.output_dir
    ):
        raise ComparisonAnalysisError(
            "Explicit mode requires eight --run-dir values, both panel workbooks and --output-dir"
        )
    output = run_analysis(
        args.run_dir,
        args.core_workbook,
        args.broad_workbook,
        evaluator,
        args.output_dir,
        core_sheet=args.core_sheet,
        broad_sheet=args.broad_sheet,
        expected_run_count=8,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        comparison_models=args.comparison_model,
    )
    print(f"News-first vol comparison outputs written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_COMPARISON_COLUMNS",
    "CHECKPOINT_SUMMARY_COLUMNS",
    "MAE_NUMERICAL_TIE_TOLERANCE",
    "MATURITY_METRIC_COLUMNS",
    "MODEL_COMPARISON_COLUMNS",
    "PAIR_METRIC_COLUMNS",
    "SAMPLE_METRIC_COLUMNS",
    "TEXT_ABLATION_COMPARISON_COLUMNS",
    "ComparisonAnalysisError",
    "GridInterpolationError",
    "PredictionEvaluator",
    "RunSpec",
    "TrainedRunEvaluator",
    "aggregate_pair_metrics",
    "build_bootstrap_comparisons",
    "build_model_comparison",
    "build_text_ablation_comparisons",
    "cme_session_cluster_bootstrap",
    "compute_sample_metrics",
    "compute_maturity_metrics",
    "collect_checkpoint_summary",
    "collect_training_curves",
    "discover_run_spec",
    "evaluate_trained_run",
    "holm_adjust",
    "precomputed_prediction_evaluator",
    "run_analysis",
    "run_analysis_from_experiment",
    "run_experiment_analysis",
    "select_primary_tolerance_bootstrap",
    "surface_arbitrage_violations",
    "surface_diagnostic_metrics",
    "strict_grid_interpolate",
    "validate_panel",
]
