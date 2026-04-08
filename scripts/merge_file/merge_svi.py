"""Build an SVI audit workbook from minute-SVI results."""

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
from scripts._path_setup import ROOT_DIR

from quantlib.vol_surface.algo.svi_algo import _svi_function, _vars_to_vols  # noqa: E402
from scripts.generate_surface.common.minute_svi_excel_common import (  # noqa: E402
    DEFAULT_SOURCE_TIMEZONE,
)
from scripts.merge_file._merge_common import (  # noqa: E402
    assign_raw_row_to_slice as _assign_raw_row_to_slice,
    coerce_optional_numeric as _coerce_optional_numeric,
    int_range_summary as _int_range_summary,
    is_boundary_slice as _is_boundary_slice,
    is_placeholder_slice as _is_placeholder_slice,
    load_json_direction_map as _load_json_direction_map,
    load_news_base_frame,
    load_precalib_csv as _load_precalib_csv,
    max_abs_error as _max_abs_error,
    normalize_bool as _normalize_bool,
    normalize_optional_text as _normalize_optional_text,
    offset_column_name as _offset_column_name,
    range_summary as _range_summary,
    resolve_existing_path as _resolve_existing_path,
    safe_float as _safe_float,
    safe_int as _safe_int,
    serialize_list as _serialize_list,
    to_utc_string as _to_utc_string,
    weighted_mae as _weighted_mae,
    weighted_rmse as _weighted_rmse,
    write_workbook,
)

DEFAULT_NEWS_XLSX_PATH = ROOT_DIR / "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
DEFAULT_CSV_NAME = "minute_svi_precalib_points.csv"
DEFAULT_JSON_NAME = "minute_svi_params.json"
DEFAULT_OUTPUT_NAME = "merged_svi.xlsx"
DEFAULT_OFFSET_MINUTES = 5
DAYS_IN_YEAR = 250

AUDIT_SHEET = "news_direction_audit"
SLICE_SHEET = "svi_slice_detail"
GAN_SHEET = "gan_input_ready"

AUDIT_HEADERS = [
    "sample_id",
    "news_row_id",
    "article_id",
    "source_file",
    "direction",
    "news_timestamp_utc",
    "matched_snapshot_time_utc",
    "json_target_timestamp_utc",
    "hd_text",
    "lp_text",
    "hd_embedding",
    "lp_embedding",
    "hd_dim",
    "lp_dim",
    "has_csv_points",
    "raw_point_count",
    "raw_point_pass_count",
    "raw_point_fail_count",
    "raw_point_pass_ratio",
    "raw_business_days_count",
    "raw_business_days_min",
    "raw_business_days_max",
    "strike_min",
    "strike_max",
    "percent_strike_min",
    "percent_strike_max",
    "has_svi_params",
    "svi_slice_count",
    "svi_business_days_min",
    "svi_business_days_max",
    "svi_business_days_list",
    "svi_a_list",
    "svi_b_list",
    "svi_rho_list",
    "svi_m_list",
    "svi_sigma_list",
    "exact_slice_point_count",
    "exact_slice_point_ratio",
    "nearest_slice_gap_days_mean",
    "nearest_slice_gap_days_max",
    "weighted_iv_rmse",
    "weighted_iv_mae",
    "max_abs_iv_error",
    "weighted_total_var_rmse",
    "svi_boundary_flag",
    "svi_placeholder_flag",
    "fit_quality_label",
    "training_candidate_flag",
    "exclude_reason",
]

SLICE_HEADERS = [
    "sample_id",
    "news_row_id",
    "direction",
    "matched_snapshot_time_utc",
    "json_target_timestamp_utc",
    "slice_index",
    "business_days",
    "a",
    "b",
    "rho",
    "m",
    "sigma",
    "raw_point_count_on_slice",
    "raw_point_pass_count_on_slice",
    "percent_strike_min_on_slice",
    "percent_strike_max_on_slice",
    "weighted_iv_rmse_slice",
    "weighted_iv_mae_slice",
    "max_abs_iv_error_slice",
    "boundary_flag_slice",
    "placeholder_flag_slice",
]

GAN_HEADERS = [
    "sample_id",
    "news_timestamp_utc",
    "direction",
    "matched_snapshot_time_utc",
    "hd_embedding",
    "lp_embedding",
    "hd_dim",
    "lp_dim",
    "svi_business_days_list",
    "svi_a_list",
    "svi_b_list",
    "svi_rho_list",
    "svi_m_list",
    "svi_sigma_list",
    "raw_point_pass_count",
    "exact_slice_point_ratio",
    "weighted_iv_rmse",
    "fit_quality_label",
    "training_candidate_flag",
]


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge minute SVI results into an audit workbook.")
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


def _compute_raw_total_variance(implied_vol: float, business_days: int) -> float:
    return float(implied_vol * implied_vol * float(business_days) / float(DAYS_IN_YEAR))


def _compute_model_values(slice_row: Mapping[str, Any], percent_strike: float) -> Tuple[Optional[float], Optional[float]]:
    if percent_strike <= 0:
        return None, None
    total_var = _svi_function(
        a=float(slice_row["a"]),
        b=float(slice_row["b"]),
        rho=float(slice_row["rho"]),
        m=float(slice_row["m"]),
        sigma=float(slice_row["sigma"]),
        percent_strike=[float(percent_strike)],
    )[0]
    model_iv = _vars_to_vols(np.asarray([total_var], dtype=np.float64), t=float(slice_row["business_days"]), days_in_year=DAYS_IN_YEAR)[0]
    return float(total_var), float(model_iv)


def _sample_base_fields(news_row: pd.Series, direction: str, matched_snapshot: str) -> Dict[str, Any]:
    return {
        "sample_id": f"news_{int(news_row['news_row_id'])}_{direction}",
        "news_row_id": int(news_row["news_row_id"]),
        "article_id": _normalize_optional_text(news_row.get("ArticleID", "")),
        "source_file": _normalize_optional_text(news_row.get("SourceFile", "")),
        "direction": direction,
        "news_timestamp_utc": _normalize_optional_text(news_row.get("timestamp_utc", "")),
        "matched_snapshot_time_utc": _normalize_optional_text(matched_snapshot),
        "hd_text": _normalize_optional_text(news_row.get("HD", "")),
        "lp_text": _normalize_optional_text(news_row.get("LP", "")),
        "hd_embedding": _normalize_optional_text(news_row.get("HD_embedding", "")),
        "lp_embedding": _normalize_optional_text(news_row.get("LP_embedding", "")),
        "hd_dim": _normalize_optional_text(news_row.get("HD_dim", "")),
        "lp_dim": _normalize_optional_text(news_row.get("LP_dim", "")),
    }


def _build_sample_rows(
    news_row: pd.Series,
    direction: str,
    matched_snapshot: str,
    raw_rows: List[Dict[str, Any]],
    json_entry: Optional[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    base = _sample_base_fields(news_row, direction, matched_snapshot)
    matched_snapshot_text = _normalize_optional_text(matched_snapshot)
    json_target_timestamp = ""
    slices: List[Dict[str, Any]] = []
    has_svi_params = False
    if json_entry is not None:
        json_target_timestamp = _normalize_optional_text(json_entry.get("json_target_timestamp_utc", ""))
        slices = list(json_entry.get("slices", []))
        has_svi_params = bool(json_entry.get("has_svi_params", False))

    raw_point_count = len(raw_rows)
    raw_pass_rows = [row for row in raw_rows if _normalize_bool(row.get("passes_precalib_filter"))]
    raw_point_pass_count = len(raw_pass_rows)
    raw_point_fail_count = raw_point_count - raw_point_pass_count
    raw_point_pass_ratio = float(raw_point_pass_count / raw_point_count) if raw_point_count else 0.0
    raw_business_days_values = [row.get("business_days") for row in raw_rows]
    raw_bday_min, raw_bday_max = _int_range_summary(raw_business_days_values)
    strike_min, strike_max = _range_summary([row.get("strike") for row in raw_rows])
    pct_min, pct_max = _range_summary([row.get("percent_strike") for row in raw_rows])

    slice_business_days = [slice_row["business_days"] for slice_row in slices]
    svi_business_days_min, svi_business_days_max = _int_range_summary(slice_business_days)
    svi_placeholder_flag = int(bool(slices) and all(_is_placeholder_slice(slice_row) for slice_row in slices))
    svi_boundary_flag = int(bool(slices) and any(_is_boundary_slice(slice_row) for slice_row in slices))

    all_assignments = [
        assignment
        for assignment in (_assign_raw_row_to_slice(raw_row, slices) for raw_row in raw_rows)
        if assignment is not None
    ]
    pass_assignments = [
        assignment
        for assignment in (_assign_raw_row_to_slice(raw_row, slices) for raw_row in raw_pass_rows)
        if assignment is not None
    ]

    exact_slice_point_count = sum(1 for assignment in pass_assignments if assignment["is_exact"])
    exact_slice_point_ratio = float(exact_slice_point_count / raw_point_pass_count) if raw_point_pass_count else 0.0

    gap_values = [float(assignment["slice_day_gap"]) for assignment in pass_assignments]
    nearest_slice_gap_days_mean = float(np.mean(gap_values)) if gap_values else None
    nearest_slice_gap_days_max = float(np.max(gap_values)) if gap_values else None

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
    weights = [float(assignment["weight"]) for assignment in pass_assignments if assignment["iv_error"] is not None and assignment["weight"] > 0]
    total_var_weights = [
        float(assignment["weight"])
        for assignment in pass_assignments
        if assignment["total_var_error"] is not None and assignment["weight"] > 0
    ]
    weighted_iv_rmse = _weighted_rmse(iv_errors, weights)
    weighted_iv_mae = _weighted_mae(iv_errors, weights)
    max_abs_iv_error = _max_abs_error(iv_errors)
    weighted_total_var_rmse = _weighted_rmse(total_var_errors, total_var_weights)

    if not has_svi_params:
        fit_quality_label = "no_svi"
        exclude_reason = "no_svi"
    elif svi_placeholder_flag:
        fit_quality_label = "placeholder"
        exclude_reason = "placeholder"
    elif raw_point_pass_count == 0:
        fit_quality_label = "no_raw_points"
        exclude_reason = "no_raw_points"
    elif exact_slice_point_ratio < 0.5:
        fit_quality_label = "poor"
        exclude_reason = "low_exact_slice_ratio"
    elif weighted_iv_rmse is None or weighted_iv_rmse > 0.05:
        fit_quality_label = "poor"
        exclude_reason = "high_weighted_iv_rmse"
    else:
        fit_quality_label = "usable"
        exclude_reason = ""
    training_candidate_flag = int(fit_quality_label == "usable")

    audit_row = {
        **base,
        "json_target_timestamp_utc": json_target_timestamp,
        "has_csv_points": bool(raw_point_count > 0),
        "raw_point_count": raw_point_count,
        "raw_point_pass_count": raw_point_pass_count,
        "raw_point_fail_count": raw_point_fail_count,
        "raw_point_pass_ratio": raw_point_pass_ratio,
        "raw_business_days_count": len({value for value in (_safe_int(v) for v in raw_business_days_values) if value is not None}),
        "raw_business_days_min": raw_bday_min,
        "raw_business_days_max": raw_bday_max,
        "strike_min": strike_min,
        "strike_max": strike_max,
        "percent_strike_min": pct_min,
        "percent_strike_max": pct_max,
        "has_svi_params": has_svi_params,
        "svi_slice_count": len(slices),
        "svi_business_days_min": svi_business_days_min,
        "svi_business_days_max": svi_business_days_max,
        "svi_business_days_list": _serialize_list([slice_row["business_days"] for slice_row in slices]) if slices else "",
        "svi_a_list": _serialize_list([slice_row["a"] for slice_row in slices]) if slices else "",
        "svi_b_list": _serialize_list([slice_row["b"] for slice_row in slices]) if slices else "",
        "svi_rho_list": _serialize_list([slice_row["rho"] for slice_row in slices]) if slices else "",
        "svi_m_list": _serialize_list([slice_row["m"] for slice_row in slices]) if slices else "",
        "svi_sigma_list": _serialize_list([slice_row["sigma"] for slice_row in slices]) if slices else "",
        "exact_slice_point_count": exact_slice_point_count,
        "exact_slice_point_ratio": exact_slice_point_ratio,
        "nearest_slice_gap_days_mean": _coerce_optional_numeric(nearest_slice_gap_days_mean),
        "nearest_slice_gap_days_max": _coerce_optional_numeric(nearest_slice_gap_days_max),
        "weighted_iv_rmse": _coerce_optional_numeric(weighted_iv_rmse),
        "weighted_iv_mae": _coerce_optional_numeric(weighted_iv_mae),
        "max_abs_iv_error": _coerce_optional_numeric(max_abs_iv_error),
        "weighted_total_var_rmse": _coerce_optional_numeric(weighted_total_var_rmse),
        "svi_boundary_flag": svi_boundary_flag,
        "svi_placeholder_flag": svi_placeholder_flag,
        "fit_quality_label": fit_quality_label,
        "training_candidate_flag": training_candidate_flag,
        "exclude_reason": exclude_reason,
    }

    slice_rows: List[Dict[str, Any]] = []
    for slice_row in slices:
        all_for_slice = [assignment for assignment in all_assignments if assignment["slice"]["slice_index"] == slice_row["slice_index"]]
        pass_for_slice = [assignment for assignment in pass_assignments if assignment["slice"]["slice_index"] == slice_row["slice_index"]]
        pass_pct_values = [assignment["raw_row"].get("percent_strike") for assignment in pass_for_slice]
        pct_min_slice, pct_max_slice = _range_summary(pass_pct_values)
        slice_iv_errors = [
            float(assignment["iv_error"])
            for assignment in pass_for_slice
            if assignment["iv_error"] is not None and assignment["weight"] > 0
        ]
        slice_weights = [
            float(assignment["weight"])
            for assignment in pass_for_slice
            if assignment["iv_error"] is not None and assignment["weight"] > 0
        ]
        slice_rows.append(
            {
                "sample_id": base["sample_id"],
                "news_row_id": base["news_row_id"],
                "direction": direction,
                "matched_snapshot_time_utc": matched_snapshot_text,
                "json_target_timestamp_utc": json_target_timestamp,
                "slice_index": int(slice_row["slice_index"]),
                "business_days": int(slice_row["business_days"]),
                "a": float(slice_row["a"]),
                "b": float(slice_row["b"]),
                "rho": float(slice_row["rho"]),
                "m": float(slice_row["m"]),
                "sigma": float(slice_row["sigma"]),
                "raw_point_count_on_slice": len(all_for_slice),
                "raw_point_pass_count_on_slice": len(pass_for_slice),
                "percent_strike_min_on_slice": pct_min_slice,
                "percent_strike_max_on_slice": pct_max_slice,
                "weighted_iv_rmse_slice": _coerce_optional_numeric(_weighted_rmse(slice_iv_errors, slice_weights)),
                "weighted_iv_mae_slice": _coerce_optional_numeric(_weighted_mae(slice_iv_errors, slice_weights)),
                "max_abs_iv_error_slice": _coerce_optional_numeric(_max_abs_error(slice_iv_errors)),
                "boundary_flag_slice": int(_is_boundary_slice(slice_row)),
                "placeholder_flag_slice": int(_is_placeholder_slice(slice_row)),
            }
        )

    return audit_row, slice_rows


def build_workbook_frames(
    input_dir: Path,
    *,
    news_xlsx_path: Path = DEFAULT_NEWS_XLSX_PATH,
    source_timezone: str = DEFAULT_SOURCE_TIMEZONE,
    offset_minutes: int = DEFAULT_OFFSET_MINUTES,
) -> Dict[str, pd.DataFrame]:
    input_dir = _resolve_existing_path(Path(input_dir), "Input directory")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_dir}")

    news_xlsx_path = _resolve_existing_path(Path(news_xlsx_path), "News xlsx")
    csv_path = _resolve_existing_path(input_dir / DEFAULT_CSV_NAME, "CSV")
    json_path = _resolve_existing_path(input_dir / DEFAULT_JSON_NAME, "JSON")

    news_df = load_news_base_frame(news_xlsx_path, source_timezone=source_timezone, offset_minutes=offset_minutes)
    csv_df = _load_precalib_csv(csv_path)
    json_direction_map = _load_json_direction_map(json_path)
    csv_groups = {
        str(timestamp): group.to_dict(orient="records")
        for timestamp, group in csv_df.groupby("calibration_datetime_utc", dropna=False)
        if str(timestamp).strip()
    }

    offset_column = _offset_column_name(offset_minutes)
    audit_rows: List[Dict[str, Any]] = []
    slice_rows: List[Dict[str, Any]] = []

    for _, news_row in news_df.iterrows():
        backward_snapshot = _normalize_optional_text(news_row.get("timestamp_utc", "")).strip()
        forward_snapshot = _normalize_optional_text(news_row.get(offset_column, "")).strip()
        for direction, matched_snapshot in (("backward", backward_snapshot), ("forward", forward_snapshot)):
            raw_rows = list(csv_groups.get(matched_snapshot, [])) if matched_snapshot else []
            json_entry = json_direction_map.get((matched_snapshot, direction)) if matched_snapshot else None
            audit_row, sample_slice_rows = _build_sample_rows(news_row, direction, matched_snapshot, raw_rows, json_entry)
            audit_rows.append(audit_row)
            slice_rows.extend(sample_slice_rows)

    audit_df = pd.DataFrame(audit_rows, columns=AUDIT_HEADERS)
    slice_df = pd.DataFrame(slice_rows, columns=SLICE_HEADERS)
    gan_df = audit_df.loc[audit_df["training_candidate_flag"] == 1, GAN_HEADERS].reset_index(drop=True)
    return {
        AUDIT_SHEET: audit_df,
        SLICE_SHEET: slice_df,
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
