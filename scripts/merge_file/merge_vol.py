"""Build a paired vol-surface audit workbook from minute-SVI results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from scripts.generate_surface.common.minute_svi_excel_common import DEFAULT_SOURCE_TIMEZONE  # noqa: E402
from scripts.merge_file import merge_svi  # noqa: E402
from scripts.merge_file._merge_common import (  # noqa: E402
    assign_raw_row_to_slice,
    coerce_optional_numeric,
    is_boundary_slice,
    is_placeholder_slice,
    load_json_direction_map,
    load_news_base_frame,
    load_precalib_csv,
    max_abs_error,
    normalize_bool,
    normalize_optional_text,
    offset_column_name,
    resolve_existing_path,
    serialize_list,
    weighted_mae,
    weighted_rmse,
    write_workbook,
)
from wgan_option.config import default_config  # noqa: E402

DEFAULT_NEWS_XLSX_PATH = merge_svi.DEFAULT_NEWS_XLSX_PATH
DEFAULT_CSV_NAME = merge_svi.DEFAULT_CSV_NAME
DEFAULT_JSON_NAME = merge_svi.DEFAULT_JSON_NAME
DEFAULT_OUTPUT_NAME = "merged_vol.xlsx"
DEFAULT_OFFSET_MINUTES = merge_svi.DEFAULT_OFFSET_MINUTES
DAYS_IN_YEAR = merge_svi.DAYS_IN_YEAR

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
    "strike_grid",
    "maturity_days_grid",
    "surface_shape",
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
    "has_svi",
    "svi_slice_count",
    "svi_business_days_list",
    "svi_a_list",
    "svi_b_list",
    "svi_rho_list",
    "svi_m_list",
    "svi_sigma_list",
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


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge minute SVI results into a paired vol-surface workbook.")
    parser.add_argument("--input-dir", required=True, help="Directory containing minute_svi_precalib_points.csv and minute_svi_params.json.")
    parser.add_argument(
        "--source-timezone",
        default=DEFAULT_SOURCE_TIMEZONE,
        help="Timezone used to parse PD + ET in the news xlsx.",
    )
    parser.add_argument(
        "--offset-minutes",
        type=int,
        default=DEFAULT_OFFSET_MINUTES,
        help="Forward direction offset in minutes.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _grid_definition() -> Tuple[List[float], List[float], str]:
    strike_grid = np.linspace(
        default_config.moneyness_min,
        default_config.moneyness_max,
        default_config.strike_bins,
        dtype=np.float64,
    )
    maturity_days_grid = np.linspace(
        default_config.maturity_min_days,
        default_config.maturity_max_days,
        default_config.maturity_bins,
        dtype=np.float64,
    )
    return (
        [float(value) for value in strike_grid.tolist()],
        [float(value) for value in maturity_days_grid.tolist()],
        serialize_list([int(default_config.maturity_bins), int(default_config.strike_bins)]),
    )


def _interp_param(days: np.ndarray, values: np.ndarray, day: float) -> float:
    if day <= float(days[0]):
        return float(values[0])
    if day >= float(days[-1]):
        return float(values[-1])
    idx = int(np.searchsorted(days, day))
    left_day = float(days[idx - 1])
    right_day = float(days[idx])
    left_value = float(values[idx - 1])
    right_value = float(values[idx])
    weight = 0.0 if abs(right_day - left_day) < 1e-12 else (float(day) - left_day) / (right_day - left_day)
    return float(left_value + weight * (right_value - left_value))


def _slice_total_variance(log_moneyness: float, a: float, b: float, rho: float, m: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-8)
    core = float(rho) * (float(log_moneyness) - float(m)) + math.sqrt((float(log_moneyness) - float(m)) ** 2 + sigma * sigma)
    return float(max(float(a) + float(b) * core, 0.0))


def _reconstruct_surface_flat(
    slices: Sequence[Mapping[str, Any]],
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
) -> List[float]:
    if not slices:
        return []
    days = np.asarray([float(slice_row["business_days"]) for slice_row in slices], dtype=np.float64)
    a_arr = np.asarray([float(slice_row["a"]) for slice_row in slices], dtype=np.float64)
    b_arr = np.asarray([float(slice_row["b"]) for slice_row in slices], dtype=np.float64)
    rho_arr = np.asarray([float(slice_row["rho"]) for slice_row in slices], dtype=np.float64)
    m_arr = np.asarray([float(slice_row["m"]) for slice_row in slices], dtype=np.float64)
    sigma_arr = np.asarray([float(slice_row["sigma"]) for slice_row in slices], dtype=np.float64)

    surface_flat: List[float] = []
    for day in maturity_days_grid:
        maturity_day = max(float(day), 1.0)
        a = _interp_param(days, a_arr, maturity_day)
        b = _interp_param(days, b_arr, maturity_day)
        rho = _interp_param(days, rho_arr, maturity_day)
        m = _interp_param(days, m_arr, maturity_day)
        sigma = _interp_param(days, sigma_arr, maturity_day)
        for percent_strike in strike_grid:
            log_moneyness = math.log(max(float(percent_strike), 1e-12))
            total_var = _slice_total_variance(log_moneyness, a, b, rho, m, sigma)
            model_iv = math.sqrt(max(total_var, 0.0) * float(DAYS_IN_YEAR) / maturity_day)
            surface_flat.append(float(model_iv))
    return surface_flat


def _surface_stats(surface_flat: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not surface_flat:
        return None, None, None, None
    surface = np.asarray(surface_flat, dtype=np.float64)
    return float(surface.min()), float(surface.max()), float(surface.mean()), float(surface.std())


def _side_quality_label(
    *,
    has_svi: bool,
    placeholder_flag: int,
    raw_point_pass_count: int,
    exact_slice_point_ratio: float,
    weighted_iv_rmse: Optional[float],
) -> str:
    if not has_svi:
        return "no_svi"
    if placeholder_flag:
        return "placeholder"
    if raw_point_pass_count == 0:
        return "no_raw_points"
    if exact_slice_point_ratio < 0.5:
        return "poor"
    if weighted_iv_rmse is None or weighted_iv_rmse > 0.05:
        return "poor"
    return "usable"


def _pair_quality_label(current_metrics: Mapping[str, Any], target_metrics: Mapping[str, Any]) -> Tuple[str, str]:
    if not bool(current_metrics["has_svi"]):
        return "no_current_svi", "no_current_svi"
    if not bool(target_metrics["has_svi"]):
        return "no_target_svi", "no_target_svi"
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
    has_svi = False
    if json_entry is not None:
        json_target_timestamp = normalize_optional_text(json_entry.get("json_target_timestamp_utc", ""))
        slices = list(json_entry.get("slices", []))
        has_svi = bool(json_entry.get("has_svi_params", False))

    raw_point_count = len(raw_rows)
    raw_pass_rows = [row for row in raw_rows if normalize_bool(row.get("passes_precalib_filter"))]
    raw_point_pass_count = len(raw_pass_rows)

    pass_assignments = [
        assignment
        for assignment in (assign_raw_row_to_slice(raw_row, slices) for raw_row in raw_pass_rows)
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

    weighted_iv_rmse = weighted_rmse(iv_errors, iv_weights)
    weighted_iv_mae = weighted_mae(iv_errors, iv_weights)
    max_abs_iv_error = max_abs_error(iv_errors)
    weighted_total_var_rmse = weighted_rmse(total_var_errors, total_var_weights)

    placeholder_flag = int(bool(slices) and all(is_placeholder_slice(slice_row) for slice_row in slices))
    boundary_flag = int(bool(slices) and any(is_boundary_slice(slice_row) for slice_row in slices))
    side_quality_label = _side_quality_label(
        has_svi=has_svi,
        placeholder_flag=placeholder_flag,
        raw_point_pass_count=raw_point_pass_count,
        exact_slice_point_ratio=exact_slice_point_ratio,
        weighted_iv_rmse=weighted_iv_rmse,
    )

    surface_flat = _reconstruct_surface_flat(slices, strike_grid, maturity_days_grid) if has_svi else []
    surface_min, surface_max, surface_mean, surface_std = _surface_stats(surface_flat)

    side_row = {
        "sample_id": sample_id,
        "news_row_id": news_row_id,
        "side": side,
        "source_direction": source_direction,
        "matched_snapshot_time_utc": normalize_optional_text(matched_snapshot),
        "json_target_timestamp_utc": json_target_timestamp,
        "has_svi": has_svi,
        "svi_slice_count": len(slices),
        "svi_business_days_list": serialize_list([slice_row["business_days"] for slice_row in slices]) if slices else "",
        "svi_a_list": serialize_list([slice_row["a"] for slice_row in slices]) if slices else "",
        "svi_b_list": serialize_list([slice_row["b"] for slice_row in slices]) if slices else "",
        "svi_rho_list": serialize_list([slice_row["rho"] for slice_row in slices]) if slices else "",
        "svi_m_list": serialize_list([slice_row["m"] for slice_row in slices]) if slices else "",
        "svi_sigma_list": serialize_list([slice_row["sigma"] for slice_row in slices]) if slices else "",
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
        "weighted_iv_rmse": coerce_optional_numeric(weighted_iv_rmse),
        "weighted_iv_mae": coerce_optional_numeric(weighted_iv_mae),
        "max_abs_iv_error": coerce_optional_numeric(max_abs_iv_error),
        "weighted_total_var_rmse": coerce_optional_numeric(weighted_total_var_rmse),
        "boundary_flag": boundary_flag,
        "placeholder_flag": placeholder_flag,
        "side_quality_label": side_quality_label,
    }

    metrics = {
        "matched_snapshot_time_utc": side_row["matched_snapshot_time_utc"],
        "json_target_timestamp_utc": json_target_timestamp,
        "has_svi": has_svi,
        "svi_slice_count": len(slices),
        "surface_flat": side_row["surface_flat"],
        "raw_point_count": raw_point_count,
        "raw_point_pass_count": raw_point_pass_count,
        "exact_slice_point_ratio": exact_slice_point_ratio,
        "weighted_iv_rmse": coerce_optional_numeric(weighted_iv_rmse),
        "weighted_iv_mae": coerce_optional_numeric(weighted_iv_mae),
        "max_abs_iv_error": coerce_optional_numeric(max_abs_iv_error),
        "weighted_total_var_rmse": coerce_optional_numeric(weighted_total_var_rmse),
        "boundary_flag": boundary_flag,
        "placeholder_flag": placeholder_flag,
        "side_quality_label": side_quality_label,
    }
    return side_row, metrics


def build_workbook_frames(
    input_dir: Path,
    *,
    news_xlsx_path: Path = DEFAULT_NEWS_XLSX_PATH,
    source_timezone: str = DEFAULT_SOURCE_TIMEZONE,
    offset_minutes: int = DEFAULT_OFFSET_MINUTES,
) -> Dict[str, pd.DataFrame]:
    input_dir = resolve_existing_path(Path(input_dir), "Input directory")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_dir}")

    news_xlsx_path = resolve_existing_path(Path(news_xlsx_path), "News xlsx")
    csv_path = resolve_existing_path(input_dir / DEFAULT_CSV_NAME, "CSV")
    json_path = resolve_existing_path(input_dir / DEFAULT_JSON_NAME, "JSON")

    news_df = load_news_base_frame(news_xlsx_path, source_timezone=source_timezone, offset_minutes=offset_minutes)
    csv_df = load_precalib_csv(csv_path)
    json_direction_map = load_json_direction_map(json_path)
    csv_groups = {
        str(timestamp): group.to_dict(orient="records")
        for timestamp, group in csv_df.groupby("calibration_datetime_utc", dropna=False)
        if str(timestamp).strip()
    }

    strike_grid, maturity_days_grid, surface_shape_text = _grid_definition()
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
        pair_quality_label, exclude_reason = _pair_quality_label(current_metrics, target_metrics)
        training_candidate_flag = int(pair_quality_label == "usable")

        pair_rows.append(
            {
                **base,
                "current_snapshot_time_utc": current_metrics["matched_snapshot_time_utc"],
                "target_snapshot_time_utc": target_metrics["matched_snapshot_time_utc"],
                "current_json_target_timestamp_utc": current_metrics["json_target_timestamp_utc"],
                "target_json_target_timestamp_utc": target_metrics["json_target_timestamp_utc"],
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
                "pair_quality_label": pair_quality_label,
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
def main(argv: Optional[Iterable[str]] = None) -> Path:
    args = _parse_args(argv)
    input_dir = Path(args.input_dir).expanduser()
    workbook_frames = build_workbook_frames(
        input_dir,
        news_xlsx_path=DEFAULT_NEWS_XLSX_PATH,
        source_timezone=str(args.source_timezone),
        offset_minutes=int(args.offset_minutes),
    )
    output_path = input_dir / DEFAULT_OUTPUT_NAME
    write_workbook(output_path, workbook_frames)
    print(f"Merged workbook written to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
