"""Core logic for building a paired vol-surface audit workbook."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from wgan_option.merge_support import (
    DEFAULT_DAYS_IN_YEAR as _DEFAULT_DAYS_IN_YEAR,
    DEFAULT_NEWS_XLSX_PATH,
    DEFAULT_OFFSET_MINUTES,
    DEFAULT_SOURCE_TIMEZONE,
    build_surface_from_params,
    coerce_optional_numeric,
    evaluate_raw_row_against_surface,
    is_boundary_surface_slice,
    is_placeholder_surface_slice,
    load_json_direction_map,
    load_news_base_frame,
    load_precalib_csv,
    max_abs_error,
    normalize_bool,
    normalize_optional_text,
    offset_column_name,
    resolve_existing_path,
    resolve_surface_csv_path,
    resolve_surface_json_path,
    serialize_json,
    serialize_list,
    weighted_mae,
    weighted_rmse,
    write_workbook,
)
from wgan_option.surface_grid import (
    DEFAULT_MATURITY_BINS,
    DEFAULT_MATURITY_MAX_DAYS,
    DEFAULT_MATURITY_MIN_DAYS,
    DEFAULT_MONEYNESS_MAX,
    DEFAULT_MONEYNESS_MIN,
    DEFAULT_STRIKE_BINS,
    build_surface_grids,
    surface_shape,
)

DEFAULT_OUTPUT_NAME = "merged_vol.xlsx"
DAYS_IN_YEAR = _DEFAULT_DAYS_IN_YEAR

PAIR_AUDIT_SHEET = "news_surface_pair_audit"
SIDE_DETAIL_SHEET = "surface_side_detail"
GAN_SHEET = "gan_input_ready"

PAIR_AUDIT_HEADERS = [
    "sample_id",
    "news_row_id",
    "article_id",
    "source_file",
    "news_timestamp_utc",
    "current_snapshot_time_utc",
    "target_snapshot_time_utc",
    "current_json_target_timestamp_utc",
    "target_json_target_timestamp_utc",
    "hd_text",
    "lp_text",
    "hd_embedding",
    "lp_embedding",
    "hd_dim",
    "lp_dim",
    "surface_model",
    "strike_grid",
    "maturity_days_grid",
    "surface_shape",
    "current_has_surface",
    "target_has_surface",
    "current_surface_slice_count",
    "target_surface_slice_count",
    "current_has_svi",
    "target_has_svi",
    "current_svi_slice_count",
    "target_svi_slice_count",
    "current_surface_flat",
    "target_surface_flat",
    "current_raw_point_count",
    "target_raw_point_count",
    "current_raw_point_pass_count",
    "target_raw_point_pass_count",
    "current_exact_slice_point_ratio",
    "target_exact_slice_point_ratio",
    "current_weighted_iv_rmse",
    "target_weighted_iv_rmse",
    "current_weighted_iv_mae",
    "target_weighted_iv_mae",
    "current_max_abs_iv_error",
    "target_max_abs_iv_error",
    "current_weighted_total_var_rmse",
    "target_weighted_total_var_rmse",
    "current_boundary_flag",
    "target_boundary_flag",
    "current_placeholder_flag",
    "target_placeholder_flag",
    "pair_quality_label",
    "training_candidate_flag",
    "exclude_reason",
]

SIDE_DETAIL_HEADERS = [
    "sample_id",
    "news_row_id",
    "side",
    "source_direction",
    "matched_snapshot_time_utc",
    "json_target_timestamp_utc",
    "surface_model",
    "has_surface",
    "surface_slice_count",
    "has_svi",
    "svi_slice_count",
    "svi_business_days_list",
    "svi_a_list",
    "svi_b_list",
    "svi_rho_list",
    "svi_m_list",
    "svi_sigma_list",
    "surface_param_json",
    "strike_grid",
    "maturity_days_grid",
    "surface_flat",
    "surface_min",
    "surface_max",
    "surface_mean",
    "surface_std",
    "raw_point_count",
    "raw_point_pass_count",
    "exact_slice_point_ratio",
    "weighted_iv_rmse",
    "weighted_iv_mae",
    "max_abs_iv_error",
    "weighted_total_var_rmse",
    "boundary_flag",
    "placeholder_flag",
    "side_quality_label",
]

GAN_HEADERS = [
    "sample_id",
    "news_timestamp_utc",
    "current_snapshot_time_utc",
    "target_snapshot_time_utc",
    "surface_model",
    "current_has_surface",
    "target_has_surface",
    "current_surface_slice_count",
    "target_surface_slice_count",
    "hd_embedding",
    "lp_embedding",
    "hd_dim",
    "lp_dim",
    "strike_grid",
    "maturity_days_grid",
    "surface_shape",
    "current_surface_flat",
    "target_surface_flat",
    "current_weighted_iv_rmse",
    "target_weighted_iv_rmse",
    "pair_quality_label",
    "training_candidate_flag",
]


def _grid_definition(
    *,
    strike_bins: int = DEFAULT_STRIKE_BINS,
    maturity_bins: int = DEFAULT_MATURITY_BINS,
    moneyness_min: float = DEFAULT_MONEYNESS_MIN,
    moneyness_max: float = DEFAULT_MONEYNESS_MAX,
    maturity_min_days: int = DEFAULT_MATURITY_MIN_DAYS,
    maturity_max_days: int = DEFAULT_MATURITY_MAX_DAYS,
) -> Tuple[List[float], List[float], str]:
    strike_grid, maturity_days_grid = build_surface_grids(
        strike_bins=strike_bins,
        maturity_bins=maturity_bins,
        moneyness_min=moneyness_min,
        moneyness_max=moneyness_max,
        maturity_min_days=maturity_min_days,
        maturity_max_days=maturity_max_days,
        dtype=np.float64,
    )
    return (
        [float(value) for value in strike_grid.tolist()],
        [float(value) for value in maturity_days_grid.tolist()],
        serialize_list(surface_shape(strike_bins=strike_bins, maturity_bins=maturity_bins)),
    )


def _reconstruct_surface_flat(
    surface,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
) -> List[float]:
    surface_grid = surface.implied_vol_surface(
        percent_strikes=[float(value) for value in strike_grid],
        business_days=[int(round(value)) for value in maturity_days_grid],
        forward=1.0,
    )
    return [
        float(value)
        for row in surface_grid
        for value in row
    ]


def _surface_stats(surface_flat: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not surface_flat:
        return None, None, None, None
    surface = np.asarray(surface_flat, dtype=np.float64)
    return float(surface.min()), float(surface.max()), float(surface.mean()), float(surface.std())


def _side_quality_label(
    *,
    surface_model: str,
    has_surface: bool,
    placeholder_flag: int,
    raw_point_pass_count: int,
    exact_slice_point_ratio: float,
    weighted_iv_rmse: Optional[float],
) -> str:
    if not has_surface:
        return "no_svi" if surface_model == "svi" else "no_surface"
    if placeholder_flag:
        return "placeholder"
    if raw_point_pass_count == 0:
        return "no_raw_points"
    if exact_slice_point_ratio < 0.5:
        return "poor"
    if weighted_iv_rmse is None or weighted_iv_rmse > 0.05:
        return "poor"
    return "usable"


def _missing_surface_label(prefix: str, surface_model: str) -> str:
    suffix = "svi" if surface_model == "svi" else "surface"
    return f"no_{prefix}_{suffix}"


def _pair_quality_label(current_metrics: Mapping[str, Any], target_metrics: Mapping[str, Any]) -> Tuple[str, str]:
    current_model = str(current_metrics.get("surface_model") or "svi")
    target_model = str(target_metrics.get("surface_model") or current_model or "svi")
    if not bool(current_metrics["has_surface"]):
        label = _missing_surface_label("current", current_model)
        return label, label
    if not bool(target_metrics["has_surface"]):
        label = _missing_surface_label("target", target_model)
        return label, label
    if int(current_metrics["placeholder_flag"]) == 1:
        return "current_placeholder", "current_placeholder"
    if int(target_metrics["placeholder_flag"]) == 1:
        return "target_placeholder", "target_placeholder"
    if int(current_metrics["raw_point_pass_count"]) == 0:
        return "no_current_raw_points", "no_current_raw_points"
    if int(target_metrics["raw_point_pass_count"]) == 0:
        return "no_target_raw_points", "no_target_raw_points"

    current_exact = float(current_metrics["exact_slice_point_ratio"])
    target_exact = float(target_metrics["exact_slice_point_ratio"])
    current_rmse = current_metrics["weighted_iv_rmse"]
    target_rmse = target_metrics["weighted_iv_rmse"]
    if (
        current_exact < 0.5
        or target_exact < 0.5
        or current_rmse is None
        or target_rmse is None
        or float(current_rmse) > 0.05
        or float(target_rmse) > 0.05
    ):
        return "poor_fit", "poor_fit"
    return "usable", ""


def _base_pair_fields(news_row: pd.Series) -> Dict[str, Any]:
    return {
        "sample_id": f"news_{int(news_row['news_row_id'])}",
        "news_row_id": int(news_row["news_row_id"]),
        "article_id": normalize_optional_text(news_row.get("ArticleID", "")),
        "source_file": normalize_optional_text(news_row.get("SourceFile", "")),
        "news_timestamp_utc": normalize_optional_text(news_row.get("timestamp_utc", "")),
        "hd_text": normalize_optional_text(news_row.get("HD", "")),
        "lp_text": normalize_optional_text(news_row.get("LP", "")),
        "hd_embedding": normalize_optional_text(news_row.get("HD_embedding", "")),
        "lp_embedding": normalize_optional_text(news_row.get("LP_embedding", "")),
        "hd_dim": normalize_optional_text(news_row.get("HD_dim", "")),
        "lp_dim": normalize_optional_text(news_row.get("LP_dim", "")),
    }


def _evaluate_side(
    *,
    sample_id: str,
    news_row_id: int,
    side: str,
    source_direction: str,
    matched_snapshot: str,
    raw_rows: Sequence[Mapping[str, Any]],
    json_entry: Optional[Mapping[str, Any]],
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
    strike_grid_text: str,
    maturity_grid_text: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    json_target_timestamp = ""
    slices: List[Dict[str, Any]] = []
    surface_model = "svi"
    surface_params: Optional[Mapping[str, Any]] = None
    has_surface = False
    if json_entry is not None:
        json_target_timestamp = normalize_optional_text(json_entry.get("json_target_timestamp_utc", ""))
        surface_model = normalize_optional_text(json_entry.get("surface_model", "svi")) or "svi"
        surface_params = json_entry.get("surface_params")
        slices = list(json_entry.get("slices", []))
        has_surface = bool(json_entry.get("has_surface_params", False))

    raw_point_count = len(raw_rows)
    raw_pass_rows = [row for row in raw_rows if normalize_bool(row.get("passes_precalib_filter"))]
    raw_point_pass_count = len(raw_pass_rows)

    surface = None
    if has_surface and surface_params is not None:
        valuation_date = (
            pd.Timestamp(matched_snapshot).tz_convert("UTC").date()
            if matched_snapshot
            else date(2000, 1, 3)
        )
        try:
            surface = build_surface_from_params(
                surface_model=surface_model,
                surface_params=surface_params,
                valuation_date=valuation_date,
                days_in_year=DAYS_IN_YEAR,
            )
        except Exception:
            surface = None
            has_surface = False

    pass_assignments = []
    if surface is not None:
        pass_assignments = [
            assignment
            for assignment in (
                evaluate_raw_row_against_surface(
                    raw_row,
                    slices=slices,
                    surface=surface,
                    days_in_year=DAYS_IN_YEAR,
                )
                for raw_row in raw_pass_rows
            )
            if assignment is not None
        ]
    exact_slice_point_count = sum(1 for assignment in pass_assignments if assignment["is_exact"])
    exact_slice_point_ratio = float(exact_slice_point_count / raw_point_pass_count) if raw_point_pass_count else 0.0

    iv_errors = [
        float(assignment["iv_error"])
        for assignment in pass_assignments
        if assignment["iv_error"] is not None and assignment["weight"] > 0
    ]
    total_var_errors = [
        float(assignment["total_var_error"])
        for assignment in pass_assignments
        if assignment["total_var_error"] is not None and assignment["weight"] > 0
    ]
    iv_weights = [
        float(assignment["weight"])
        for assignment in pass_assignments
        if assignment["iv_error"] is not None and assignment["weight"] > 0
    ]
    total_var_weights = [
        float(assignment["weight"])
        for assignment in pass_assignments
        if assignment["total_var_error"] is not None and assignment["weight"] > 0
    ]

    w_iv_rmse = weighted_rmse(iv_errors, iv_weights)
    w_iv_mae = weighted_mae(iv_errors, iv_weights)
    max_abs_iv_err = max_abs_error(iv_errors)
    w_total_var_rmse = weighted_rmse(total_var_errors, total_var_weights)

    placeholder_flag = int(bool(slices) and all(is_placeholder_surface_slice(surface_model, slice_row) for slice_row in slices))
    boundary_flag = int(bool(slices) and any(is_boundary_surface_slice(surface_model, slice_row) for slice_row in slices))
    side_ql = _side_quality_label(
        surface_model=surface_model,
        has_surface=has_surface,
        placeholder_flag=placeholder_flag,
        raw_point_pass_count=raw_point_pass_count,
        exact_slice_point_ratio=exact_slice_point_ratio,
        weighted_iv_rmse=w_iv_rmse,
    )

    surface_flat = _reconstruct_surface_flat(surface, strike_grid, maturity_days_grid) if surface is not None else []
    surface_min, surface_max, surface_mean, surface_std = _surface_stats(surface_flat)
    surface_business_days = [slice_row["business_days"] for slice_row in slices]
    svi_fields = {
        "svi_a_list": "",
        "svi_b_list": "",
        "svi_rho_list": "",
        "svi_m_list": "",
        "svi_sigma_list": "",
    }
    if surface_model == "svi" and slices:
        svi_fields = {
            "svi_a_list": serialize_list([slice_row["a"] for slice_row in slices]),
            "svi_b_list": serialize_list([slice_row["b"] for slice_row in slices]),
            "svi_rho_list": serialize_list([slice_row["rho"] for slice_row in slices]),
            "svi_m_list": serialize_list([slice_row["m"] for slice_row in slices]),
            "svi_sigma_list": serialize_list([slice_row["sigma"] for slice_row in slices]),
        }

    side_row = {
        "sample_id": sample_id,
        "news_row_id": news_row_id,
        "side": side,
        "source_direction": source_direction,
        "matched_snapshot_time_utc": normalize_optional_text(matched_snapshot),
        "json_target_timestamp_utc": json_target_timestamp,
        "surface_model": surface_model,
        "has_surface": has_surface,
        "surface_slice_count": len(slices),
        "has_svi": has_surface,
        "svi_slice_count": len(slices),
        "svi_business_days_list": serialize_list(surface_business_days) if slices else "",
        **svi_fields,
        "surface_param_json": serialize_json(surface_params) if surface_params is not None else "",
        "strike_grid": strike_grid_text,
        "maturity_days_grid": maturity_grid_text,
        "surface_flat": serialize_list(surface_flat) if surface_flat else "",
        "surface_min": coerce_optional_numeric(surface_min),
        "surface_max": coerce_optional_numeric(surface_max),
        "surface_mean": coerce_optional_numeric(surface_mean),
        "surface_std": coerce_optional_numeric(surface_std),
        "raw_point_count": raw_point_count,
        "raw_point_pass_count": raw_point_pass_count,
        "exact_slice_point_ratio": exact_slice_point_ratio,
        "weighted_iv_rmse": coerce_optional_numeric(w_iv_rmse),
        "weighted_iv_mae": coerce_optional_numeric(w_iv_mae),
        "max_abs_iv_error": coerce_optional_numeric(max_abs_iv_err),
        "weighted_total_var_rmse": coerce_optional_numeric(w_total_var_rmse),
        "boundary_flag": boundary_flag,
        "placeholder_flag": placeholder_flag,
        "side_quality_label": side_ql,
    }

    metrics = {
        "matched_snapshot_time_utc": side_row["matched_snapshot_time_utc"],
        "json_target_timestamp_utc": json_target_timestamp,
        "surface_model": surface_model,
        "has_surface": has_surface,
        "surface_slice_count": len(slices),
        "has_svi": has_surface,
        "svi_slice_count": len(slices),
        "surface_flat": side_row["surface_flat"],
        "raw_point_count": raw_point_count,
        "raw_point_pass_count": raw_point_pass_count,
        "exact_slice_point_ratio": exact_slice_point_ratio,
        "weighted_iv_rmse": coerce_optional_numeric(w_iv_rmse),
        "weighted_iv_mae": coerce_optional_numeric(w_iv_mae),
        "max_abs_iv_error": coerce_optional_numeric(max_abs_iv_err),
        "weighted_total_var_rmse": coerce_optional_numeric(w_total_var_rmse),
        "boundary_flag": boundary_flag,
        "placeholder_flag": placeholder_flag,
        "side_quality_label": side_ql,
    }
    return side_row, metrics


def build_vol_workbook_frames(
    input_dir: Path,
    *,
    news_xlsx_path: Path = DEFAULT_NEWS_XLSX_PATH,
    source_timezone: str = DEFAULT_SOURCE_TIMEZONE,
    offset_minutes: int = DEFAULT_OFFSET_MINUTES,
    strike_bins: int = DEFAULT_STRIKE_BINS,
    maturity_bins: int = DEFAULT_MATURITY_BINS,
    moneyness_min: float = DEFAULT_MONEYNESS_MIN,
    moneyness_max: float = DEFAULT_MONEYNESS_MAX,
    maturity_min_days: int = DEFAULT_MATURITY_MIN_DAYS,
    maturity_max_days: int = DEFAULT_MATURITY_MAX_DAYS,
) -> Dict[str, pd.DataFrame]:
    """Build the three-sheet paired vol-surface workbook from raw surface results."""

    input_dir = resolve_existing_path(Path(input_dir), "Input directory")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_dir}")

    news_xlsx_path = resolve_existing_path(Path(news_xlsx_path), "News xlsx")
    csv_path = resolve_surface_csv_path(input_dir)
    json_path = resolve_surface_json_path(input_dir)

    news_df = load_news_base_frame(news_xlsx_path, source_timezone=source_timezone, offset_minutes=offset_minutes)
    csv_df = load_precalib_csv(csv_path)
    json_direction_map = load_json_direction_map(json_path)
    csv_groups = {
        str(timestamp): group.to_dict(orient="records")
        for timestamp, group in csv_df.groupby("calibration_datetime_utc", dropna=False)
        if str(timestamp).strip()
    }

    strike_grid, maturity_days_grid, surface_shape_text = _grid_definition(
        strike_bins=strike_bins,
        maturity_bins=maturity_bins,
        moneyness_min=moneyness_min,
        moneyness_max=moneyness_max,
        maturity_min_days=maturity_min_days,
        maturity_max_days=maturity_max_days,
    )
    strike_grid_text = serialize_list(strike_grid)
    maturity_grid_text = serialize_list(maturity_days_grid)

    offset_column = offset_column_name(offset_minutes)
    pair_rows: List[Dict[str, Any]] = []
    side_rows: List[Dict[str, Any]] = []

    for _, news_row in news_df.iterrows():
        base = _base_pair_fields(news_row)
        current_snapshot = normalize_optional_text(news_row.get("timestamp_utc", "")).strip()
        target_snapshot = normalize_optional_text(news_row.get(offset_column, "")).strip()

        current_side_row, current_metrics = _evaluate_side(
            sample_id=base["sample_id"],
            news_row_id=base["news_row_id"],
            side="current_back",
            source_direction="backward",
            matched_snapshot=current_snapshot,
            raw_rows=list(csv_groups.get(current_snapshot, [])) if current_snapshot else [],
            json_entry=json_direction_map.get((current_snapshot, "backward")) if current_snapshot else None,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
            strike_grid_text=strike_grid_text,
            maturity_grid_text=maturity_grid_text,
        )
        target_side_row, target_metrics = _evaluate_side(
            sample_id=base["sample_id"],
            news_row_id=base["news_row_id"],
            side="target_forward",
            source_direction="forward",
            matched_snapshot=target_snapshot,
            raw_rows=list(csv_groups.get(target_snapshot, [])) if target_snapshot else [],
            json_entry=json_direction_map.get((target_snapshot, "forward")) if target_snapshot else None,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
            strike_grid_text=strike_grid_text,
            maturity_grid_text=maturity_grid_text,
        )
        pair_quality, exclude_reason = _pair_quality_label(current_metrics, target_metrics)
        training_candidate_flag = int(pair_quality == "usable")

        pair_rows.append(
            {
                **base,
                "current_snapshot_time_utc": current_metrics["matched_snapshot_time_utc"],
                "target_snapshot_time_utc": target_metrics["matched_snapshot_time_utc"],
                "current_json_target_timestamp_utc": current_metrics["json_target_timestamp_utc"],
                "target_json_target_timestamp_utc": target_metrics["json_target_timestamp_utc"],
                "surface_model": current_metrics["surface_model"] or target_metrics["surface_model"],
                "current_has_surface": current_metrics["has_surface"],
                "target_has_surface": target_metrics["has_surface"],
                "current_surface_slice_count": current_metrics["surface_slice_count"],
                "target_surface_slice_count": target_metrics["surface_slice_count"],
                "strike_grid": strike_grid_text,
                "maturity_days_grid": maturity_grid_text,
                "surface_shape": surface_shape_text,
                "current_has_svi": current_metrics["has_svi"],
                "target_has_svi": target_metrics["has_svi"],
                "current_svi_slice_count": current_metrics["svi_slice_count"],
                "target_svi_slice_count": target_metrics["svi_slice_count"],
                "current_surface_flat": current_metrics["surface_flat"],
                "target_surface_flat": target_metrics["surface_flat"],
                "current_raw_point_count": current_metrics["raw_point_count"],
                "target_raw_point_count": target_metrics["raw_point_count"],
                "current_raw_point_pass_count": current_metrics["raw_point_pass_count"],
                "target_raw_point_pass_count": target_metrics["raw_point_pass_count"],
                "current_exact_slice_point_ratio": current_metrics["exact_slice_point_ratio"],
                "target_exact_slice_point_ratio": target_metrics["exact_slice_point_ratio"],
                "current_weighted_iv_rmse": current_metrics["weighted_iv_rmse"],
                "target_weighted_iv_rmse": target_metrics["weighted_iv_rmse"],
                "current_weighted_iv_mae": current_metrics["weighted_iv_mae"],
                "target_weighted_iv_mae": target_metrics["weighted_iv_mae"],
                "current_max_abs_iv_error": current_metrics["max_abs_iv_error"],
                "target_max_abs_iv_error": target_metrics["max_abs_iv_error"],
                "current_weighted_total_var_rmse": current_metrics["weighted_total_var_rmse"],
                "target_weighted_total_var_rmse": target_metrics["weighted_total_var_rmse"],
                "current_boundary_flag": current_metrics["boundary_flag"],
                "target_boundary_flag": target_metrics["boundary_flag"],
                "current_placeholder_flag": current_metrics["placeholder_flag"],
                "target_placeholder_flag": target_metrics["placeholder_flag"],
                "pair_quality_label": pair_quality,
                "training_candidate_flag": training_candidate_flag,
                "exclude_reason": exclude_reason,
            }
        )
        side_rows.extend([current_side_row, target_side_row])

    pair_df = pd.DataFrame(pair_rows, columns=PAIR_AUDIT_HEADERS)
    side_df = pd.DataFrame(side_rows, columns=SIDE_DETAIL_HEADERS)
    gan_df = pair_df.loc[pair_df["training_candidate_flag"] == 1, GAN_HEADERS].reset_index(drop=True)
    return {
        PAIR_AUDIT_SHEET: pair_df,
        SIDE_DETAIL_SHEET: side_df,
        GAN_SHEET: gan_df,
    }
