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

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quantlib.vol_surface.algo.svi_algo import _svi_function, _vars_to_vols  # noqa: E402
from scripts.generate_surface.common.minute_svi_excel_common import (  # noqa: E402
    DEFAULT_SOURCE_TIMEZONE,
    _normalize_date_value,
    _normalize_time_value,
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


def _resolve_existing_path(path_value: Path, label: str) -> Path:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _offset_column_name(offset_minutes: int) -> str:
    return f"timestamp_utc_plus_{int(offset_minutes)}m"


def _normalize_optional_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def _to_utc_string(ts: Any) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _serialize_list(values: Sequence[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _safe_int(value: Any) -> Optional[int]:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _coerce_optional_numeric(value: Optional[float]) -> Any:
    return None if value is None else float(value)


def _weighted_rmse(errors: List[float], weights: List[float]) -> Optional[float]:
    if not errors or not weights:
        return None
    denom = float(np.sum(weights))
    if not math.isfinite(denom) or denom <= 0:
        return None
    sq = np.square(np.asarray(errors, dtype=np.float64))
    return float(np.sqrt(np.dot(sq, np.asarray(weights, dtype=np.float64)) / denom))


def _weighted_mae(errors: List[float], weights: List[float]) -> Optional[float]:
    if not errors or not weights:
        return None
    denom = float(np.sum(weights))
    if not math.isfinite(denom) or denom <= 0:
        return None
    abs_errors = np.abs(np.asarray(errors, dtype=np.float64))
    return float(np.dot(abs_errors, np.asarray(weights, dtype=np.float64)) / denom)


def _max_abs_error(errors: List[float]) -> Optional[float]:
    if not errors:
        return None
    return float(np.max(np.abs(np.asarray(errors, dtype=np.float64))))


def _range_summary(values: Sequence[Any]) -> Tuple[Optional[float], Optional[float]]:
    numeric = [v for v in (_safe_float(value) for value in values) if v is not None]
    if not numeric:
        return None, None
    return float(min(numeric)), float(max(numeric))


def _int_range_summary(values: Sequence[Any]) -> Tuple[Optional[int], Optional[int]]:
    numeric = [v for v in (_safe_int(value) for value in values) if v is not None]
    if not numeric:
        return None, None
    return int(min(numeric)), int(max(numeric))


def load_news_base_frame(
    xlsx_path: Path,
    *,
    source_timezone: str,
    offset_minutes: int,
) -> pd.DataFrame:
    news_df = pd.read_excel(xlsx_path, engine="openpyxl", dtype=object)
    for column in ("SourceFile", "ArticleID", "HD", "LP", "HD_embedding", "LP_embedding", "HD_dim", "LP_dim"):
        if column not in news_df.columns:
            news_df[column] = ""
    missing_columns = [column for column in ("PD", "ET") if column not in news_df.columns]
    if missing_columns:
        raise ValueError(f"News xlsx is missing required columns {missing_columns}: {xlsx_path}")

    news_df = news_df.copy()
    news_df.insert(0, "news_row_id", range(1, len(news_df) + 1))
    date_text = news_df["PD"].map(_normalize_date_value)
    time_text = news_df["ET"].map(_normalize_time_value)
    combined = (date_text + " " + time_text).where((date_text != "") & (time_text != ""), None)
    naive_ts = pd.to_datetime(combined, errors="coerce")
    localized = naive_ts.dt.tz_localize(source_timezone, ambiguous="NaT", nonexistent="NaT")
    utc_ts = localized.dt.tz_convert("UTC")
    shifted = utc_ts + pd.Timedelta(minutes=int(offset_minutes))

    news_df["timestamp_utc"] = utc_ts.map(_to_utc_string)
    news_df[_offset_column_name(offset_minutes)] = shifted.map(_to_utc_string)
    return news_df


def _load_precalib_csv(csv_path: Path) -> pd.DataFrame:
    csv_df = pd.read_csv(csv_path)
    required = {
        "calibration_datetime_utc",
        "business_days",
        "contract_id",
        "strike",
        "percent_strike",
        "implied_vol",
        "passes_precalib_filter",
        "weight",
    }
    missing = sorted(required - set(csv_df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns {missing}: {csv_path}")
    csv_df = csv_df.copy()
    csv_df["passes_precalib_filter"] = csv_df["passes_precalib_filter"].map(_normalize_bool)
    return csv_df


def _extract_svi_slices(params: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if params is None:
        return []
    required = ("business_days", "a", "b", "rho", "m", "sigma")
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"SVI params missing required keys {missing}")
    lengths = []
    for key in required:
        values = params[key]
        if not isinstance(values, list):
            raise ValueError(f"SVI param `{key}` must be a list.")
        lengths.append(len(values))
    if len(set(lengths)) != 1:
        raise ValueError(f"SVI param list lengths must match: {dict(zip(required, lengths))}")

    slices: List[Dict[str, Any]] = []
    for slice_index in range(lengths[0]):
        business_days = _safe_int(params["business_days"][slice_index])
        a = _safe_float(params["a"][slice_index])
        b = _safe_float(params["b"][slice_index])
        rho = _safe_float(params["rho"][slice_index])
        m = _safe_float(params["m"][slice_index])
        sigma = _safe_float(params["sigma"][slice_index])
        if None in {business_days, a, b, rho, m, sigma}:
            raise ValueError(f"Invalid numeric SVI slice at index {slice_index}: {params}")
        slices.append(
            {
                "slice_index": slice_index,
                "business_days": int(business_days),
                "a": float(a),
                "b": float(b),
                "rho": float(rho),
                "m": float(m),
                "sigma": float(sigma),
            }
        )
    return slices


def _load_json_direction_map(json_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {json_path}")

    direction_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for target_timestamp, target_payload in payload.items():
        if not isinstance(target_payload, Mapping):
            raise ValueError(f"JSON target `{target_timestamp}` must be an object.")
        for direction in ("backward", "forward"):
            side_payload = target_payload.get(direction)
            if not isinstance(side_payload, Mapping):
                raise ValueError(f"JSON target `{target_timestamp}` missing direction `{direction}`.")
            snapshot = str(side_payload.get("snapshot_time_utc", "")).strip()
            if not snapshot:
                raise ValueError(
                    f"JSON target `{target_timestamp}` direction `{direction}` missing `snapshot_time_utc`."
                )
            slices = _extract_svi_slices(side_payload.get("svi_params"))
            key = (snapshot, direction)
            if key in direction_map:
                raise ValueError(f"Duplicate JSON direction entry for snapshot={snapshot}, direction={direction}.")
            direction_map[key] = {
                "json_target_timestamp_utc": str(target_timestamp),
                "snapshot_time_utc": snapshot,
                "direction": direction,
                "slices": slices,
                "has_svi_params": bool(slices),
            }
    return direction_map


def _is_placeholder_slice(slice_row: Mapping[str, Any]) -> bool:
    return (
        float(slice_row["b"]) == 0.0
        and float(slice_row["rho"]) == 0.0
        and float(slice_row["m"]) == 0.0
        and float(slice_row["sigma"]) == 0.0
    )


def _is_boundary_slice(slice_row: Mapping[str, Any]) -> bool:
    return (
        abs(float(slice_row["rho"])) >= 0.999
        or float(slice_row["sigma"]) <= 5e-4
        or float(slice_row["sigma"]) >= 5.0
        or float(slice_row["a"]) <= 1e-12
        or float(slice_row["b"]) <= 1e-12
    )


def _compute_raw_total_variance(implied_vol: float, business_days: int) -> float:
    return float(implied_vol * implied_vol * float(business_days) / float(DAYS_IN_YEAR))


def _compute_model_values(slice_row: Mapping[str, Any], percent_strike: float) -> Tuple[Optional[float], Optional[float]]:
    if percent_strike <= 0:
        return None, None
    total_var = _svi_function(
        float(slice_row["a"]),
        float(slice_row["b"]),
        float(slice_row["rho"]),
        float(slice_row["m"]),
        float(slice_row["sigma"]),
        [float(percent_strike)],
    )[0]
    model_iv = _vars_to_vols(np.asarray([total_var], dtype=np.float64), t=float(slice_row["business_days"]), days_in_year=DAYS_IN_YEAR)[0]
    return float(total_var), float(model_iv)


def _assign_raw_row_to_slice(raw_row: Mapping[str, Any], slices: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not slices:
        return None
    raw_business_days = _safe_int(raw_row.get("business_days"))
    percent_strike = _safe_float(raw_row.get("percent_strike"))
    implied_vol = _safe_float(raw_row.get("implied_vol"))
    weight = _safe_float(raw_row.get("weight"))
    if raw_business_days is None or percent_strike is None or implied_vol is None or weight is None:
        return None
    nearest_slice = min(
        slices,
        key=lambda slice_row: (
            abs(int(raw_business_days) - int(slice_row["business_days"])),
            int(slice_row["slice_index"]),
        ),
    )
    model_total_var, model_iv = _compute_model_values(nearest_slice, percent_strike)
    raw_total_var = _compute_raw_total_variance(implied_vol, int(raw_business_days))
    iv_error = None if model_iv is None else float(implied_vol - model_iv)
    total_var_error = None if model_total_var is None else float(raw_total_var - model_total_var)
    return {
        "raw_row": raw_row,
        "slice": nearest_slice,
        "is_exact": int(raw_business_days) == int(nearest_slice["business_days"]),
        "slice_day_gap": abs(int(raw_business_days) - int(nearest_slice["business_days"])),
        "weight": float(weight),
        "model_total_variance": model_total_var,
        "model_implied_vol": model_iv,
        "raw_total_variance": raw_total_var,
        "iv_error": iv_error,
        "total_var_error": total_var_error,
    }


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


def write_workbook(output_path: Path, workbook_frames: Mapping[str, pd.DataFrame]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name in (AUDIT_SHEET, SLICE_SHEET, GAN_SHEET):
            workbook_frames[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
    return output_path


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
