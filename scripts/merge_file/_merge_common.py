"""Shared utility functions for merge_svi.py and merge_vol.py."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from quantlib.vol_surface.algo.svi_algo import _svi_function, _vars_to_vols
from scripts.generate_surface.common.minute_svi_excel_common import (
    _normalize_date_value,
    _normalize_time_value,
)

DEFAULT_DAYS_IN_YEAR = 250


# ---------------------------------------------------------------------------
# Path / text / type helpers
# ---------------------------------------------------------------------------


def resolve_existing_path(path_value: Path, label: str) -> Path:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def offset_column_name(offset_minutes: int) -> str:
    return f"timestamp_utc_plus_{int(offset_minutes)}m"


def normalize_optional_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def to_utc_string(ts: Any) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def serialize_list(values: Sequence[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def safe_int(value: Any) -> Optional[int]:
    numeric = safe_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def coerce_optional_numeric(value: Optional[float]) -> Any:
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# Error / metric helpers
# ---------------------------------------------------------------------------


def weighted_rmse(errors: List[float], weights: List[float]) -> Optional[float]:
    if not errors or not weights:
        return None
    denom = float(np.sum(weights))
    if not math.isfinite(denom) or denom <= 0:
        return None
    sq = np.square(np.asarray(errors, dtype=np.float64))
    return float(np.sqrt(np.dot(sq, np.asarray(weights, dtype=np.float64)) / denom))


def weighted_mae(errors: List[float], weights: List[float]) -> Optional[float]:
    if not errors or not weights:
        return None
    denom = float(np.sum(weights))
    if not math.isfinite(denom) or denom <= 0:
        return None
    abs_errors = np.abs(np.asarray(errors, dtype=np.float64))
    return float(np.dot(abs_errors, np.asarray(weights, dtype=np.float64)) / denom)


def max_abs_error(errors: List[float]) -> Optional[float]:
    if not errors:
        return None
    return float(np.max(np.abs(np.asarray(errors, dtype=np.float64))))


def range_summary(values: Sequence[Any]) -> Tuple[Optional[float], Optional[float]]:
    numeric = [v for v in (safe_float(value) for value in values) if v is not None]
    if not numeric:
        return None, None
    return float(min(numeric)), float(max(numeric))


def int_range_summary(values: Sequence[Any]) -> Tuple[Optional[int], Optional[int]]:
    numeric = [v for v in (safe_int(value) for value in values) if v is not None]
    if not numeric:
        return None, None
    return int(min(numeric)), int(max(numeric))


# ---------------------------------------------------------------------------
# Shared merge input loaders and SVI/raw-row alignment helpers
# ---------------------------------------------------------------------------


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

    news_df["timestamp_utc"] = utc_ts.map(to_utc_string)
    news_df[offset_column_name(offset_minutes)] = shifted.map(to_utc_string)
    return news_df


def load_precalib_csv(csv_path: Path) -> pd.DataFrame:
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
    csv_df["passes_precalib_filter"] = csv_df["passes_precalib_filter"].map(normalize_bool)
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
        business_days = safe_int(params["business_days"][slice_index])
        a = safe_float(params["a"][slice_index])
        b = safe_float(params["b"][slice_index])
        rho = safe_float(params["rho"][slice_index])
        m = safe_float(params["m"][slice_index])
        sigma = safe_float(params["sigma"][slice_index])
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


def load_json_direction_map(json_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
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


def _compute_raw_total_variance(implied_vol: float, business_days: int, *, days_in_year: int = DEFAULT_DAYS_IN_YEAR) -> float:
    return float(implied_vol * implied_vol * float(business_days) / float(days_in_year))


def _compute_model_values(
    slice_row: Mapping[str, Any],
    percent_strike: float,
    *,
    days_in_year: int = DEFAULT_DAYS_IN_YEAR,
) -> Tuple[Optional[float], Optional[float]]:
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
    model_iv = _vars_to_vols(
        np.asarray([total_var], dtype=np.float64),
        t=float(slice_row["business_days"]),
        days_in_year=days_in_year,
    )[0]
    return float(total_var), float(model_iv)


def assign_raw_row_to_slice(
    raw_row: Mapping[str, Any],
    slices: Sequence[Mapping[str, Any]],
    *,
    days_in_year: int = DEFAULT_DAYS_IN_YEAR,
) -> Optional[Dict[str, Any]]:
    if not slices:
        return None
    raw_business_days = safe_int(raw_row.get("business_days"))
    percent_strike = safe_float(raw_row.get("percent_strike"))
    implied_vol = safe_float(raw_row.get("implied_vol"))
    weight = safe_float(raw_row.get("weight"))
    if raw_business_days is None or percent_strike is None or implied_vol is None or weight is None:
        return None
    nearest_slice = min(
        slices,
        key=lambda slice_row: (
            abs(int(raw_business_days) - int(slice_row["business_days"])),
            int(slice_row["slice_index"]),
        ),
    )
    model_total_var, model_iv = _compute_model_values(nearest_slice, percent_strike, days_in_year=days_in_year)
    raw_total_var = _compute_raw_total_variance(implied_vol, int(raw_business_days), days_in_year=days_in_year)
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


# ---------------------------------------------------------------------------
# SVI slice helpers
# ---------------------------------------------------------------------------


def is_placeholder_slice(slice_row: Mapping[str, Any]) -> bool:
    return (
        float(slice_row["b"]) == 0.0
        and float(slice_row["rho"]) == 0.0
        and float(slice_row["m"]) == 0.0
        and float(slice_row["sigma"]) == 0.0
    )


def is_boundary_slice(slice_row: Mapping[str, Any]) -> bool:
    return (
        abs(float(slice_row["rho"])) >= 0.999
        or float(slice_row["sigma"]) <= 5e-4
        or float(slice_row["sigma"]) >= 5.0
        or float(slice_row["a"]) <= 1e-12
        or float(slice_row["b"]) <= 1e-12
    )


# ---------------------------------------------------------------------------
# Excel workbook writer
# ---------------------------------------------------------------------------


def write_workbook(output_path: Path, sheet_map: Mapping[str, pd.DataFrame]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheet_map.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return output_path
