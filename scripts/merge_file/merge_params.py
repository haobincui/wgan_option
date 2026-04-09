"""Build a model-neutral surface-parameter audit workbook from minute surface results."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from scripts.generate_surface.data_helperd.excel import DEFAULT_SOURCE_TIMEZONE  # noqa: E402
from scripts.merge_file import merge_svi  # noqa: E402
from scripts.merge_file._merge_common import (  # noqa: E402
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
    serialize_json,
    serialize_list,
    weighted_mae,
    weighted_rmse,
    write_workbook,
)

DEFAULT_NEWS_XLSX_PATH = merge_svi.DEFAULT_NEWS_XLSX_PATH
DEFAULT_CSV_NAME = merge_svi.DEFAULT_CSV_NAME
DEFAULT_JSON_NAME = merge_svi.DEFAULT_JSON_NAME
DEFAULT_OUTPUT_NAME = "merged_params.xlsx"
DEFAULT_OFFSET_MINUTES = merge_svi.DEFAULT_OFFSET_MINUTES
DAYS_IN_YEAR = merge_svi.DAYS_IN_YEAR

AUDIT_SHEET = "news_direction_audit"
SLICE_SHEET = "surface_slice_detail"
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
    "surface_model",
    "has_surface_params",
    "surface_slice_count",
    "surface_business_days_list",
    "surface_param_json",
    "raw_point_count",
    "raw_point_pass_count",
    "raw_point_fail_count",
    "raw_point_pass_ratio",
    "exact_slice_point_count",
    "exact_slice_point_ratio",
    "nearest_slice_gap_days_mean",
    "nearest_slice_gap_days_max",
    "weighted_iv_rmse",
    "weighted_iv_mae",
    "max_abs_iv_error",
    "weighted_total_var_rmse",
    "surface_boundary_flag",
    "surface_placeholder_flag",
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
    "surface_model",
    "slice_index",
    "business_days",
    "slice_param_json",
    "raw_point_count_on_slice",
    "raw_point_pass_count_on_slice",
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
    "surface_model",
    "hd_embedding",
    "lp_embedding",
    "hd_dim",
    "lp_dim",
    "surface_slice_count",
    "surface_business_days_list",
    "surface_param_json",
    "raw_point_pass_count",
    "exact_slice_point_ratio",
    "weighted_iv_rmse",
    "fit_quality_label",
    "training_candidate_flag",
]


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge minute surface params into a model-neutral audit workbook.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing minute_svi_precalib_points.csv and minute_svi_params.json.",
    )
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


def _sample_base_fields(news_row: pd.Series, direction: str, matched_snapshot: str) -> Dict[str, Any]:
    return {
        "sample_id": f"news_{int(news_row['news_row_id'])}_{direction}",
        "news_row_id": int(news_row["news_row_id"]),
        "article_id": normalize_optional_text(news_row.get("ArticleID", "")),
        "source_file": normalize_optional_text(news_row.get("SourceFile", "")),
        "direction": direction,
        "news_timestamp_utc": normalize_optional_text(news_row.get("timestamp_utc", "")),
        "matched_snapshot_time_utc": normalize_optional_text(matched_snapshot),
        "hd_text": normalize_optional_text(news_row.get("HD", "")),
        "lp_text": normalize_optional_text(news_row.get("LP", "")),
        "hd_embedding": normalize_optional_text(news_row.get("HD_embedding", "")),
        "lp_embedding": normalize_optional_text(news_row.get("LP_embedding", "")),
        "hd_dim": normalize_optional_text(news_row.get("HD_dim", "")),
        "lp_dim": normalize_optional_text(news_row.get("LP_dim", "")),
    }


def _fit_quality_label(
    *,
    has_surface_params: bool,
    placeholder_flag: int,
    raw_point_pass_count: int,
    exact_slice_point_ratio: float,
    weighted_iv_rmse: Optional[float],
) -> Tuple[str, str]:
    if not has_surface_params:
        return "no_surface_params", "no_surface_params"
    if placeholder_flag:
        return "placeholder", "placeholder"
    if raw_point_pass_count == 0:
        return "no_raw_points", "no_raw_points"
    if exact_slice_point_ratio < 0.5:
        return "poor", "low_exact_slice_ratio"
    if weighted_iv_rmse is None or weighted_iv_rmse > 0.05:
        return "poor", "high_weighted_iv_rmse"
    return "usable", ""


def _slice_param_payload(surface_model: str, slice_row: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {
        "business_days": int(slice_row["business_days"]),
    }
    if surface_model == "svi":
        payload.update(
            {
                "a": float(slice_row["a"]),
                "b": float(slice_row["b"]),
                "rho": float(slice_row["rho"]),
                "m": float(slice_row["m"]),
                "sigma": float(slice_row["sigma"]),
            }
        )
    elif surface_model == "sabr":
        payload.update(
            {
                "alpha": float(slice_row["alpha"]),
                "beta": float(slice_row["beta"]),
                "rho": float(slice_row["rho"]),
                "nu": float(slice_row["nu"]),
            }
        )
    else:
        payload.update(
            {
                "percent_strikes": list(slice_row["percent_strikes"]),
                "implied_vols": list(slice_row["implied_vols"]),
            }
        )
    return payload


def _build_sample_rows(
    news_row: pd.Series,
    direction: str,
    matched_snapshot: str,
    raw_rows: List[Dict[str, Any]],
    json_entry: Optional[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    base = _sample_base_fields(news_row, direction, matched_snapshot)
    json_target_timestamp = ""
    surface_model = "svi"
    surface_params: Optional[Mapping[str, Any]] = None
    slices: List[Dict[str, Any]] = []
    has_surface_params = False
    if json_entry is not None:
        json_target_timestamp = normalize_optional_text(json_entry.get("json_target_timestamp_utc", ""))
        surface_model = normalize_optional_text(json_entry.get("surface_model", "svi")) or "svi"
        surface_params = json_entry.get("surface_params")
        slices = list(json_entry.get("slices", []))
        has_surface_params = bool(json_entry.get("has_surface_params", False))

    raw_point_count = len(raw_rows)
    raw_pass_rows = [row for row in raw_rows if normalize_bool(row.get("passes_precalib_filter"))]
    raw_point_pass_count = len(raw_pass_rows)
    raw_point_fail_count = raw_point_count - raw_point_pass_count
    raw_point_pass_ratio = float(raw_point_pass_count / raw_point_count) if raw_point_count else 0.0

    surface = None
    if has_surface_params and surface_params is not None:
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
            has_surface_params = False

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

    gap_values = [
        float(assignment["slice_day_gap"])
        for assignment in pass_assignments
        if assignment["slice_day_gap"] is not None
    ]
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

    placeholder_flag = int(bool(slices) and all(is_placeholder_surface_slice(surface_model, slice_row) for slice_row in slices))
    boundary_flag = int(bool(slices) and any(is_boundary_surface_slice(surface_model, slice_row) for slice_row in slices))
    fit_quality_label, exclude_reason = _fit_quality_label(
        has_surface_params=has_surface_params,
        placeholder_flag=placeholder_flag,
        raw_point_pass_count=raw_point_pass_count,
        exact_slice_point_ratio=exact_slice_point_ratio,
        weighted_iv_rmse=weighted_iv_rmse,
    )
    training_candidate_flag = int(fit_quality_label == "usable")

    audit_row = {
        **base,
        "json_target_timestamp_utc": json_target_timestamp,
        "surface_model": surface_model,
        "has_surface_params": has_surface_params,
        "surface_slice_count": len(slices),
        "surface_business_days_list": serialize_list([slice_row["business_days"] for slice_row in slices]) if slices else "",
        "surface_param_json": serialize_json(surface_params) if surface_params is not None else "",
        "raw_point_count": raw_point_count,
        "raw_point_pass_count": raw_point_pass_count,
        "raw_point_fail_count": raw_point_fail_count,
        "raw_point_pass_ratio": raw_point_pass_ratio,
        "exact_slice_point_count": exact_slice_point_count,
        "exact_slice_point_ratio": exact_slice_point_ratio,
        "nearest_slice_gap_days_mean": coerce_optional_numeric(nearest_slice_gap_days_mean),
        "nearest_slice_gap_days_max": coerce_optional_numeric(nearest_slice_gap_days_max),
        "weighted_iv_rmse": coerce_optional_numeric(weighted_iv_rmse),
        "weighted_iv_mae": coerce_optional_numeric(weighted_iv_mae),
        "max_abs_iv_error": coerce_optional_numeric(max_abs_iv_error),
        "weighted_total_var_rmse": coerce_optional_numeric(weighted_total_var_rmse),
        "surface_boundary_flag": boundary_flag,
        "surface_placeholder_flag": placeholder_flag,
        "fit_quality_label": fit_quality_label,
        "training_candidate_flag": training_candidate_flag,
        "exclude_reason": exclude_reason,
    }

    slice_rows: List[Dict[str, Any]] = []
    for slice_row in slices:
        slice_assignments = [
            assignment
            for assignment in pass_assignments
            if assignment["slice"] is not None
            and int(assignment["slice"]["slice_index"]) == int(slice_row["slice_index"])
        ]
        slice_iv_errors = [
            float(assignment["iv_error"])
            for assignment in slice_assignments
            if assignment["iv_error"] is not None and assignment["weight"] > 0
        ]
        slice_weights = [
            float(assignment["weight"])
            for assignment in slice_assignments
            if assignment["iv_error"] is not None and assignment["weight"] > 0
        ]
        raw_rows_on_slice = [
            row
            for row in raw_rows
            if int(round(float(row.get("business_days", -1)))) == int(slice_row["business_days"])
        ]
        raw_pass_rows_on_slice = [
            row
            for row in raw_rows_on_slice
            if normalize_bool(row.get("passes_precalib_filter"))
        ]
        slice_rows.append(
            {
                "sample_id": base["sample_id"],
                "news_row_id": base["news_row_id"],
                "direction": direction,
                "matched_snapshot_time_utc": base["matched_snapshot_time_utc"],
                "json_target_timestamp_utc": json_target_timestamp,
                "surface_model": surface_model,
                "slice_index": int(slice_row["slice_index"]),
                "business_days": int(slice_row["business_days"]),
                "slice_param_json": serialize_json(_slice_param_payload(surface_model, slice_row)),
                "raw_point_count_on_slice": len(raw_rows_on_slice),
                "raw_point_pass_count_on_slice": len(raw_pass_rows_on_slice),
                "weighted_iv_rmse_slice": coerce_optional_numeric(weighted_rmse(slice_iv_errors, slice_weights)),
                "weighted_iv_mae_slice": coerce_optional_numeric(weighted_mae(slice_iv_errors, slice_weights)),
                "max_abs_iv_error_slice": coerce_optional_numeric(max_abs_error(slice_iv_errors)),
                "boundary_flag_slice": int(is_boundary_surface_slice(surface_model, slice_row)),
                "placeholder_flag_slice": int(is_placeholder_surface_slice(surface_model, slice_row)),
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

    offset_column = offset_column_name(offset_minutes)
    audit_rows: List[Dict[str, Any]] = []
    slice_rows: List[Dict[str, Any]] = []

    for _, news_row in news_df.iterrows():
        for direction, snapshot_column in (("backward", "timestamp_utc"), ("forward", offset_column)):
            matched_snapshot = normalize_optional_text(news_row.get(snapshot_column, "")).strip()
            audit_row, detail_rows = _build_sample_rows(
                news_row=news_row,
                direction=direction,
                matched_snapshot=matched_snapshot,
                raw_rows=list(csv_groups.get(matched_snapshot, [])) if matched_snapshot else [],
                json_entry=json_direction_map.get((matched_snapshot, direction)) if matched_snapshot else None,
            )
            audit_rows.append(audit_row)
            slice_rows.extend(detail_rows)

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
