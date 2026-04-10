"""Neutral shared helpers for merge_* jobs and related workbook alignment."""

from __future__ import annotations

import json
import math
from datetime import date
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from quantlib.calendar.daycount import DayCountBusN
from quantlib.calendar.holidays import usd_calendar
from quantlib.vol_surface.algo.cubic_spline_surface import CubicSplineVolSurface
from quantlib.vol_surface.algo.raw_surface import RawVolSurface
from quantlib.vol_surface.algo.sabr_surface import SabrVolSurface
from quantlib.vol_surface.algo.svi_algo import _svi_function, _vars_to_vols
from quantlib.vol_surface.algo.svi_surface import SviVolSurface

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_NEWS_XLSX_PATH = ROOT_DIR / "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
DEFAULT_SOURCE_TIMEZONE = "America/New_York"
DEFAULT_OFFSET_MINUTES = 5
DEFAULT_DAYS_IN_YEAR = 250
SUPPORTED_SURFACE_MODELS = {"svi", "sabr", "cubic", "raw"}
LEGACY_SURFACE_JSON_NAME = "minute_svi_params.json"
LEGACY_SURFACE_CSV_NAME = "minute_svi_precalib_points.csv"
NEW_SURFACE_JSON_GLOB = "surface-*.json"
NEW_SURFACE_CSV_GLOB = "surface-*-precalib-points.csv"
NEW_RESOLVED_CONFIG_NAME = "surface-resolved_config.yaml"
LEGACY_RESOLVED_CONFIG_NAME = "resolved_config.yaml"


def _normalize_date_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dt_datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dt_date):
        return value.isoformat()
    text = str(value).strip()
    if text.lower() in {"", "nan", "nat", "none"}:
        return ""
    return text


def _excel_time_fraction_to_hms(value: float) -> str:
    total_seconds = int(round(max(0.0, min(float(value), 1.0)) * 24 * 60 * 60))
    total_seconds = total_seconds % (24 * 60 * 60)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _normalize_time_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%H:%M:%S")
    if isinstance(value, dt_datetime):
        return value.strftime("%H:%M:%S")
    if isinstance(value, dt_time):
        return value.strftime("%H:%M:%S")
    if isinstance(value, (int, float)):
        numeric = float(value)
        if 0.0 <= numeric < 1.0:
            return _excel_time_fraction_to_hms(numeric)
        text = str(value).strip()
        return "" if text.lower() in {"", "nan", "nat", "none"} else text
    text = str(value).strip()
    if text.lower() in {"", "nan", "nat", "none"}:
        return ""
    return text


def resolve_existing_path(path_value: Path, label: str) -> Path:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _discover_named_input_file(
    input_dir: Path,
    *,
    pattern: str,
    legacy_name: str,
    label: str,
) -> Path:
    input_dir = resolve_existing_path(Path(input_dir), "Input directory")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_dir}")

    matches = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        described = ", ".join(str(path.name) for path in matches)
        raise ValueError(f"Expected one {label} matching `{pattern}` in {input_dir}, found: {described}")

    return resolve_existing_path(input_dir / legacy_name, label)


def resolve_surface_json_path(input_dir: Path) -> Path:
    return _discover_named_input_file(
        input_dir,
        pattern=NEW_SURFACE_JSON_GLOB,
        legacy_name=LEGACY_SURFACE_JSON_NAME,
        label="JSON",
    )


def resolve_surface_csv_path(input_dir: Path) -> Path:
    return _discover_named_input_file(
        input_dir,
        pattern=NEW_SURFACE_CSV_GLOB,
        legacy_name=LEGACY_SURFACE_CSV_NAME,
        label="CSV",
    )


def resolve_surface_resolved_config_path(input_dir: Path) -> Optional[Path]:
    input_dir = resolve_existing_path(Path(input_dir), "Input directory")
    for candidate in (NEW_RESOLVED_CONFIG_NAME, LEGACY_RESOLVED_CONFIG_NAME):
        path = input_dir / candidate
        if path.exists():
            return path
    return None


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


def serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


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


def normalize_surface_model(value: Any) -> str:
    model = str(value or "svi").strip().lower()
    if model not in SUPPORTED_SURFACE_MODELS:
        raise ValueError(f"Unsupported surface model `{value}`. Expected one of {sorted(SUPPORTED_SURFACE_MODELS)}")
    return model


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


def _extract_sabr_slices(params: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if params is None:
        return []
    required = ("business_days", "alpha", "beta", "rho", "nu")
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"SABR params missing required keys {missing}")
    lengths = []
    for key in required:
        values = params[key]
        if not isinstance(values, list):
            raise ValueError(f"SABR param `{key}` must be a list.")
        lengths.append(len(values))
    if len(set(lengths)) != 1:
        raise ValueError(f"SABR param list lengths must match: {dict(zip(required, lengths))}")

    slices: List[Dict[str, Any]] = []
    for slice_index in range(lengths[0]):
        business_days = safe_int(params["business_days"][slice_index])
        alpha = safe_float(params["alpha"][slice_index])
        beta = safe_float(params["beta"][slice_index])
        rho = safe_float(params["rho"][slice_index])
        nu = safe_float(params["nu"][slice_index])
        if None in {business_days, alpha, beta, rho, nu}:
            raise ValueError(f"Invalid numeric SABR slice at index {slice_index}: {params}")
        slices.append(
            {
                "slice_index": slice_index,
                "business_days": int(business_days),
                "alpha": float(alpha),
                "beta": float(beta),
                "rho": float(rho),
                "nu": float(nu),
            }
        )
    return slices


def _extract_grid_slices(params: Optional[Mapping[str, Any]], *, model_name: str) -> List[Dict[str, Any]]:
    if params is None:
        return []
    required = ("business_days", "percent_strikes", "implied_vols")
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"{model_name} params missing required keys {missing}")

    business_days = params["business_days"]
    percent_strikes = params["percent_strikes"]
    implied_vols = params["implied_vols"]
    if not isinstance(business_days, list) or not isinstance(percent_strikes, list) or not isinstance(implied_vols, list):
        raise ValueError(
            f"{model_name} params must use list values for business_days/percent_strikes/implied_vols"
        )
    lengths = [len(business_days), len(percent_strikes), len(implied_vols)]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"{model_name} param outer-list lengths must match: "
            f"{{'business_days': {lengths[0]}, 'percent_strikes': {lengths[1]}, 'implied_vols': {lengths[2]}}}"
        )

    slices: List[Dict[str, Any]] = []
    for slice_index in range(lengths[0]):
        business_day = safe_int(business_days[slice_index])
        strike_slice = percent_strikes[slice_index]
        vol_slice = implied_vols[slice_index]
        if business_day is None or not isinstance(strike_slice, list) or not isinstance(vol_slice, list):
            raise ValueError(f"Invalid {model_name} slice at index {slice_index}: {params}")
        if len(strike_slice) != len(vol_slice):
            raise ValueError(f"{model_name} slice {slice_index} strike/vol lengths must match.")
        cleaned_strikes = []
        cleaned_vols = []
        for strike_value, vol_value in zip(strike_slice, vol_slice):
            strike_numeric = safe_float(strike_value)
            vol_numeric = safe_float(vol_value)
            if strike_numeric is None or vol_numeric is None:
                raise ValueError(f"Invalid numeric {model_name} slice entry at index {slice_index}: {params}")
            cleaned_strikes.append(float(strike_numeric))
            cleaned_vols.append(float(vol_numeric))
        slices.append(
            {
                "slice_index": slice_index,
                "business_days": int(business_day),
                "percent_strikes": cleaned_strikes,
                "implied_vols": cleaned_vols,
            }
        )
    return slices


def extract_surface_slices(surface_model: str, params: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    model = normalize_surface_model(surface_model)
    if model == "svi":
        return _extract_svi_slices(params)
    if model == "sabr":
        return _extract_sabr_slices(params)
    if model == "cubic":
        return _extract_grid_slices(params, model_name="Cubic")
    if model == "raw":
        return _extract_grid_slices(params, model_name="Raw")
    raise ValueError(f"Unsupported surface model: {surface_model}")


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
            if "surface_model" in side_payload or "surface_params" in side_payload:
                surface_model = normalize_surface_model(side_payload.get("surface_model", "svi"))
                surface_params = side_payload.get("surface_params")
            else:
                surface_model = "svi"
                surface_params = side_payload.get("svi_params")
            slices = extract_surface_slices(surface_model, surface_params)
            key = (snapshot, direction)
            if key in direction_map:
                raise ValueError(f"Duplicate JSON direction entry for snapshot={snapshot}, direction={direction}.")
            direction_map[key] = {
                "json_target_timestamp_utc": str(target_timestamp),
                "snapshot_time_utc": snapshot,
                "direction": direction,
                "surface_model": surface_model,
                "surface_params": surface_params,
                "slices": slices,
                "has_surface_params": surface_params is not None,
                "has_svi_params": surface_model == "svi" and bool(slices),
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
        a=float(slice_row["a"]),
        b=float(slice_row["b"]),
        rho=float(slice_row["rho"]),
        m=float(slice_row["m"]),
        sigma=float(slice_row["sigma"]),
        percent_strike=[float(percent_strike)],
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


def build_surface_from_params(
    *,
    surface_model: str,
    surface_params: Mapping[str, Any],
    valuation_date: date,
    days_in_year: int = DEFAULT_DAYS_IN_YEAR,
):
    model = normalize_surface_model(surface_model)
    vol_daycount = DayCountBusN(
        f"BUS{int(days_in_year)}USD",
        usd_calendar(),
        int(days_in_year),
    )
    if model == "svi":
        return SviVolSurface(
            valuation_date=valuation_date,
            svi_params=dict(surface_params),
            vol_daycount=vol_daycount,
        )
    if model == "sabr":
        return SabrVolSurface(
            valuation_date=valuation_date,
            sabr_params=dict(surface_params),
            vol_daycount=vol_daycount,
        )
    if model == "cubic":
        return CubicSplineVolSurface(
            valuation_date=valuation_date,
            vols=list(surface_params["implied_vols"]),
            percent_strikes=list(surface_params["percent_strikes"]),
            business_days=list(surface_params["business_days"]),
            vol_daycount=vol_daycount,
        )
    if model == "raw":
        return RawVolSurface(
            valuation_date=valuation_date,
            vols=list(surface_params["implied_vols"]),
            percent_strikes=list(surface_params["percent_strikes"]),
            business_days=list(surface_params["business_days"]),
            vol_daycount=vol_daycount,
        )
    raise ValueError(f"Unsupported surface model: {surface_model}")


def evaluate_raw_row_against_surface(
    raw_row: Mapping[str, Any],
    *,
    slices: Sequence[Mapping[str, Any]],
    surface,
    days_in_year: int = DEFAULT_DAYS_IN_YEAR,
) -> Optional[Dict[str, Any]]:
    raw_business_days = safe_int(raw_row.get("business_days"))
    percent_strike = safe_float(raw_row.get("percent_strike"))
    implied_vol = safe_float(raw_row.get("implied_vol"))
    weight = safe_float(raw_row.get("weight"))
    if raw_business_days is None or percent_strike is None or implied_vol is None or weight is None:
        return None

    nearest_slice = None
    if slices:
        nearest_slice = min(
            slices,
            key=lambda slice_row: (
                abs(int(raw_business_days) - int(slice_row["business_days"])),
                int(slice_row["slice_index"]),
            ),
        )

    model_iv = None
    try:
        surface_grid = surface.implied_vol_surface(
            percent_strikes=[float(percent_strike)],
            business_days=[int(raw_business_days)],
            forward=1.0,
        )
        if surface_grid and surface_grid[0]:
            candidate_iv = safe_float(surface_grid[0][0])
            if candidate_iv is not None:
                model_iv = float(candidate_iv)
    except Exception:
        model_iv = None

    raw_total_var = _compute_raw_total_variance(
        implied_vol,
        int(raw_business_days),
        days_in_year=days_in_year,
    )
    model_total_var = None
    if model_iv is not None:
        model_total_var = _compute_raw_total_variance(
            model_iv,
            int(raw_business_days),
            days_in_year=days_in_year,
        )
    iv_error = None if model_iv is None else float(implied_vol - model_iv)
    total_var_error = None if model_total_var is None else float(raw_total_var - model_total_var)
    return {
        "raw_row": raw_row,
        "slice": nearest_slice,
        "is_exact": bool(nearest_slice is not None and int(raw_business_days) == int(nearest_slice["business_days"])),
        "slice_day_gap": (
            abs(int(raw_business_days) - int(nearest_slice["business_days"]))
            if nearest_slice is not None
            else None
        ),
        "weight": float(weight),
        "model_total_variance": model_total_var,
        "model_implied_vol": model_iv,
        "raw_total_variance": raw_total_var,
        "iv_error": iv_error,
        "total_var_error": total_var_error,
    }


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


def is_placeholder_surface_slice(surface_model: str, slice_row: Mapping[str, Any]) -> bool:
    model = normalize_surface_model(surface_model)
    if model == "svi":
        return is_placeholder_slice(slice_row)
    if model == "sabr":
        return float(slice_row["rho"]) == 0.0 and float(slice_row["nu"]) == 0.0
    return False


def is_boundary_surface_slice(surface_model: str, slice_row: Mapping[str, Any]) -> bool:
    model = normalize_surface_model(surface_model)
    if model == "svi":
        return is_boundary_slice(slice_row)
    if model == "sabr":
        return (
            abs(float(slice_row["rho"])) >= 0.999
            or float(slice_row["alpha"]) <= 1e-12
            or float(slice_row["nu"]) >= 5.0
        )
    return False


def write_workbook(output_path: Path, sheet_map: Mapping[str, pd.DataFrame]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheet_map.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return output_path
