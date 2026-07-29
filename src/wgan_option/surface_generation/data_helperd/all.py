"""Shared all-minute surface generation helpers for CPU and GPU entrypoints."""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import logging
import math
import re
import shlex
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from wgan_option.surface_generation.common.config_utils import (  # noqa: E402
    build_config_scope,
    load_surface_builder_section,
    parse_bool_from_config,
    resolve_config_path,
    resolve_config_variables,
    write_yaml_mapping,
)
from wgan_option.surface_generation.model import build_surface_params  # noqa: E402
from logger import LoggingConfig, setup_logging  # noqa: E402
from market_data.contract_handler.future_contract import FutureContract  # noqa: E402
from market_data.contract_handler.option_contract import OptionContract  # noqa: E402
from market_data.contract_handler.utils import ContractTerminationRule  # noqa: E402
from market_data.dto.tradedata_do import TradeDataDO  # noqa: E402
from quantlib.calendar.daycount import DayCountBusN  # noqa: E402
from quantlib.calendar.holidays import usd_calendar  # noqa: E402
from quantlib.calendar.utils import (  # noqa: E402
    future_maturity_month_map,
    month_map,
    option_maturity_month_map,
)
from wgan_option.market.black76 import (  # noqa: E402
    black76_implied_vol,
    black76_no_arbitrage_bounds,
)
from wgan_option.market.rates import TreasuryParYieldCurve  # noqa: E402
from wgan_option.market.treasury_options import (  # noqa: E402
    cme_treasury_calendar,
    resolve_contract_year,
    resolve_ty_option_contract_dates,
)

logger = logging.getLogger(__name__)

PRECALIB_CSV_HEADERS = [
    "target_datetime_utc",
    "window_side",
    "window_start_utc",
    "window_end_utc",
    "trade_datetime_utc",
    "calibration_datetime_utc",
    "business_days",
    "maturity_date",
    "expiration_datetime_utc",
    "contract_id",
    "option_type",
    "strike",
    "price",
    "spot",
    "underlying_contract_id",
    "underlying_trade_datetime_utc",
    "underlying_staleness_seconds",
    "underlying_match_mode",
    "percent_strike",
    "pricing_model",
    "rate_curve_date",
    "continuous_rate",
    "discount_factor",
    "rate_curve_sha256",
    "implied_vol",
    "is_otm",
    "surface_input_role",
    "passes_precalib_filter",
    "filter_reason",
    "weight",
]

DEFAULT_CONFIG_PATH = "configs/surface_builder/svi/generate_surface-svi-all.yaml"
DEFAULT_EXPIRATION_TIME_UTC = "20:00:00"
DEFAULT_MAX_PRECALIB_IV = 3.0
DEFAULT_CALIBRATION_WORKERS = 0
DEFAULT_SURFACE_MODEL = "svi"
DEFAULT_DATA_RANGE = "all"
DEFAULT_PRICING_MODEL = "legacy_black_scholes"
DEFAULT_RATE_CURVE_PATH = "data/reference/us_treasury_par_yield_curve_2022_2023.csv"
DEFAULT_MAX_RATE_STALENESS_DAYS = 7
DEFAULT_MAX_UNDERLYING_STALENESS_SECONDS = 60
DEFAULT_MAX_ITM_MONEYNESS_DISTANCE = 0.05
DEFAULT_OPTION_FILTER_MODE = "none"
DEFAULT_IV_AGGREGATION_MODE = "volume_weighted_mean"
SUPPORTED_SURFACE_MODELS = {"svi", "sabr", "cubic", "raw"}
SUPPORTED_DATA_RANGES = {"all", "window", "excel"}
SUPPORTED_PRICING_MODELS = {"legacy_black_scholes", "black76"}
SUPPORTED_OPTION_FILTER_MODES = {
    "none",
    "otm_only",
    "otm_preferred_itm_fallback",
}
SUPPORTED_IV_AGGREGATION_MODES = {"volume_weighted_mean", "volume_weighted_median"}
RESOLVED_CONFIG_FILENAME = "surface-resolved_config.yaml"
SUPPORTED_GENERATE_SURFACE_CONFIG_KEYS = {
    "input_glob",
    "output_dir",
    "output_json",
    "log_file",
    "resolved_config_path",
    "model",
    "data_range",
    "run_ts",
    "data_date",
    "expiration_time_utc",
    "max_precalib_iv",
    "days_in_year",
    "min_strikes_per_expiry",
    "min_expiries_per_minute",
    "max_files",
    "max_minutes",
    "chunk_size",
    "calibration_workers",
    "save_precalib_csv",
    "precalib_csv",
    "target_datetimes",
    "target_datetimes_file",
    "window_minutes",
    "target_xlsx",
    "sheet_name",
    "date_column",
    "time_column",
    "source_timezone",
    "publication_availability_lag_minutes",
    "max_target_datetimes",
    "pricing_model",
    "rate_curve_path",
    "max_rate_staleness_days",
    "max_underlying_staleness_seconds",
    "underlying_match_mode",
    "option_filter_mode",
    "max_itm_moneyness_distance",
    "iv_aggregation_mode",
    "window_audit_json",
    "news_alignment_mode",
    "intraday_tolerance_minutes",
    "max_session_shift_minutes",
    "require_complete_pair",
    "require_origin_not_before_news",
    "include_session_shifted",
    "collision_policy",
}

FILE_DATE_RANGE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv(?:\.gz)?$")
SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class ContractMeta:
    contract_type: str
    underlying: str
    maturity_month_code: Optional[str] = None
    maturity_year: Optional[int] = None
    strike: Optional[float] = None
    option_type: Any = None
    expiry_date: Optional[date] = None
    expiry_dt_utc: Optional[datetime] = None
    underlying_future_month_code: Optional[str] = None
    underlying_future_year: Optional[int] = None
    contract_id: Optional[str] = None


@dataclass(frozen=True)
class MinuteTradeRow:
    trade_ts: pd.Timestamp
    meta: ContractMeta
    price: float
    volume: float


SpotKey = Tuple[Any, ...]


@dataclass(frozen=True)
class UnderlyingTrade:
    trade_ts: pd.Timestamp
    price: float
    contract_id: str


@dataclass(frozen=True)
class SurfacePricingContext:
    pricing_model: str = DEFAULT_PRICING_MODEL
    rate_curve_path: str = DEFAULT_RATE_CURVE_PATH
    max_rate_staleness_days: int = DEFAULT_MAX_RATE_STALENESS_DAYS
    max_underlying_staleness_seconds: int = DEFAULT_MAX_UNDERLYING_STALENESS_SECONDS
    underlying_match_mode: str = "last_prior_trade"
    option_filter_mode: str = DEFAULT_OPTION_FILTER_MODE
    max_itm_moneyness_distance: float = DEFAULT_MAX_ITM_MONEYNESS_DISTANCE
    iv_aggregation_mode: str = DEFAULT_IV_AGGREGATION_MODE
    target_datetime_utc: str = ""
    window_side: str = ""
    window_start_utc: str = ""
    window_end_utc: str = ""


@dataclass(frozen=True)
class MinuteOptionCandidate:
    meta: ContractMeta
    price: float
    weight: float
    strike: float
    spot: float
    tau: float
    business_days: int
    trade_ts: Optional[pd.Timestamp] = None
    underlying_trade_ts: Optional[pd.Timestamp] = None
    underlying_contract_id: str = ""
    underlying_staleness_seconds: float = math.nan
    pricing_model: str = DEFAULT_PRICING_MODEL
    rate_curve_date: Optional[date] = None
    continuous_rate: float = 0.0
    discount_factor: float = 1.0
    rate_curve_sha256: str = ""
    is_otm: bool = True
    target_datetime_utc: str = ""
    window_side: str = ""
    window_start_utc: str = ""
    window_end_utc: str = ""


ProcessMinuteFn = Callable[..., None]


_resolve_config_path = resolve_config_path
_parse_bool_from_config = parse_bool_from_config


def _default_run_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _normalize_surface_model(value: Any) -> str:
    model = str(value or DEFAULT_SURFACE_MODEL).strip().lower()
    if model not in SUPPORTED_SURFACE_MODELS:
        raise ValueError(
            f"Unsupported surface model `{value}`. Expected one of: {sorted(SUPPORTED_SURFACE_MODELS)}"
        )
    return model


def _normalize_data_range(value: Any) -> str:
    data_range = str(value or DEFAULT_DATA_RANGE).strip().lower()
    if data_range not in SUPPORTED_DATA_RANGES:
        raise ValueError(
            f"Unsupported data_range `{value}`. Expected one of: {sorted(SUPPORTED_DATA_RANGES)}"
        )
    return data_range


def _normalize_pricing_model(value: Any) -> str:
    pricing_model = str(value or DEFAULT_PRICING_MODEL).strip().lower()
    if pricing_model not in SUPPORTED_PRICING_MODELS:
        raise ValueError(
            f"Unsupported pricing model `{value}`. Expected one of: "
            f"{sorted(SUPPORTED_PRICING_MODELS)}"
        )
    return pricing_model


def _normalize_option_filter_mode(value: Any) -> str:
    mode = str(value or DEFAULT_OPTION_FILTER_MODE).strip().lower()
    if mode not in SUPPORTED_OPTION_FILTER_MODES:
        raise ValueError(
            f"Unsupported option filter mode `{value}`. Expected one of: "
            f"{sorted(SUPPORTED_OPTION_FILTER_MODES)}"
        )
    return mode


def _normalize_iv_aggregation_mode(value: Any) -> str:
    mode = str(value or DEFAULT_IV_AGGREGATION_MODE).strip().lower()
    if mode not in SUPPORTED_IV_AGGREGATION_MODES:
        raise ValueError(
            f"Unsupported IV aggregation mode `{value}`. Expected one of: "
            f"{sorted(SUPPORTED_IV_AGGREGATION_MODES)}"
        )
    return mode


def _parse_expiration_time_utc(value: Any, key: str = "expiration_time_utc") -> dt_time:
    if isinstance(value, dt_time):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"Invalid time value for `{key}` in config: {value!r}")
        try:
            parsed = dt_time.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid time value for `{key}` in config: {value!r}. Expected HH:MM[:SS]."
            ) from exc
    else:
        raise ValueError(f"Invalid time value for `{key}` in config: {value!r}")

    offset = parsed.utcoffset()
    if offset not in {None, timezone.utc.utcoffset(None)}:
        raise ValueError(f"`{key}` must be UTC or naive HH:MM[:SS], got {value!r}")

    return parsed.replace(tzinfo=timezone.utc)


def _load_generate_surface_config(
    config_path_value: str,
    *,
    runtime_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config_path, config_root, defaults = load_surface_builder_section(
        config_path_value,
        section_key="generate_surface",
        supported_keys=SUPPORTED_GENERATE_SURFACE_CONFIG_KEYS,
    )
    defaults = dict(defaults)
    shared_glob = config_root.get("option_data_glob")
    if "input_glob" not in defaults and isinstance(shared_glob, str):
        defaults["input_glob"] = shared_glob

    if runtime_overrides:
        for key, value in runtime_overrides.items():
            if value is None:
                continue
            text = str(value).strip() if isinstance(value, str) else value
            if text == "":
                continue
            defaults[key] = value

    defaults["model"] = _normalize_surface_model(defaults.get("model", DEFAULT_SURFACE_MODEL))
    defaults["data_range"] = _normalize_data_range(defaults.get("data_range", DEFAULT_DATA_RANGE))
    defaults["pricing_model"] = _normalize_pricing_model(
        defaults.get(
            "pricing_model",
            "black76" if defaults["model"] == "raw" else DEFAULT_PRICING_MODEL,
        )
    )
    defaults["option_filter_mode"] = _normalize_option_filter_mode(
        defaults.get(
            "option_filter_mode",
            "otm_only" if defaults["model"] == "raw" else DEFAULT_OPTION_FILTER_MODE,
        )
    )
    defaults["max_itm_moneyness_distance"] = float(
        defaults.get(
            "max_itm_moneyness_distance",
            DEFAULT_MAX_ITM_MONEYNESS_DISTANCE,
        )
    )
    defaults["iv_aggregation_mode"] = _normalize_iv_aggregation_mode(
        defaults.get(
            "iv_aggregation_mode",
            "volume_weighted_median"
            if defaults["model"] == "raw"
            else DEFAULT_IV_AGGREGATION_MODE,
        )
    )
    defaults["run_ts"] = str(defaults.get("run_ts", "")).strip() or _default_run_ts()
    defaults.setdefault("output_dir", "data/processed/${model}-${data_range}/${run_ts}")
    defaults.setdefault("output_json", "${output_dir}/surface-${model}-${data_range}.json")
    defaults.setdefault("log_file", "${output_dir}/surface-${model}-${data_range}.log")
    defaults.setdefault("precalib_csv", "${output_dir}/surface-${model}-${data_range}-precalib-points.csv")
    defaults.setdefault("resolved_config_path", "${output_dir}/surface-resolved_config.yaml")

    return resolve_config_variables(defaults, extra_scope=build_config_scope(config_root))


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="YAML config path for surface_builder settings.",
    )
    pre_parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    pre_parser.add_argument(
        "--data_range",
        "--data-range",
        dest="data_range",
        type=str,
        default=None,
    )
    pre_parser.add_argument(
        "--run-ts",
        type=str,
        default=None,
    )
    pre_args, _ = pre_parser.parse_known_args(argv_list)

    if "-h" in argv_list or "--help" in argv_list:
        config_defaults: Dict[str, Any] = {}
    else:
        config_defaults = _load_generate_surface_config(
            pre_args.config,
            runtime_overrides={
                "model": pre_args.model,
                "data_range": pre_args.data_range,
                "run_ts": pre_args.run_ts,
            },
        )

    default_save_precalib_csv = _parse_bool_from_config(
        config_defaults.get("save_precalib_csv", False),
        key="save_precalib_csv",
    )

    parser = argparse.ArgumentParser(
        description="Generate minute-level surface parameters from option trade files."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=pre_args.config,
        help="YAML config path with a required `surface_builder.generate_surface` mapping.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default=str(config_defaults.get("input_glob", "data/raw/option_data/**/*.csv.gz")),
        help="Glob pattern for raw option trade files.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(SUPPORTED_SURFACE_MODELS),
        default=str(config_defaults.get("model", DEFAULT_SURFACE_MODEL)),
        help="Surface model used for minute generation.",
    )
    parser.add_argument(
        "--data_range",
        "--data-range",
        dest="data_range",
        choices=sorted(SUPPORTED_DATA_RANGES),
        default=str(config_defaults.get("data_range", DEFAULT_DATA_RANGE)),
        help="Minute data range selector used by the unified generate_surface CLI.",
    )
    parser.add_argument(
        "--run-ts",
        type=str,
        default=str(config_defaults.get("run_ts", _default_run_ts())),
        help="Run timestamp used inside the default output directory.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(config_defaults.get("output_json", f"data/processed/{DEFAULT_SURFACE_MODEL}-{DEFAULT_DATA_RANGE}/surface-{DEFAULT_SURFACE_MODEL}-{DEFAULT_DATA_RANGE}.json")),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(config_defaults.get("log_file", f"data/processed/{DEFAULT_SURFACE_MODEL}-{DEFAULT_DATA_RANGE}/surface-{DEFAULT_SURFACE_MODEL}-{DEFAULT_DATA_RANGE}.log")),
        help="Path to save run logs.",
    )
    parser.add_argument(
        "--data-date",
        type=str,
        default=str(config_defaults.get("data_date", "2026-03-09")),
        help="Reference date for contract expiry inference (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--expiration-time-utc",
        type=str,
        default=str(config_defaults.get("expiration_time_utc", DEFAULT_EXPIRATION_TIME_UTC)),
        help="UTC expiration time used to build expiry datetime (HH:MM[:SS]).",
    )
    parser.add_argument(
        "--days-in-year",
        type=int,
        default=int(config_defaults.get("days_in_year", 250)),
        help="Business days in year for tau/vol conversion.",
    )
    parser.add_argument(
        "--min-strikes-per-expiry",
        type=int,
        default=int(config_defaults.get("min_strikes_per_expiry", 3)),
        help="Minimum strike points required per expiry term.",
    )
    parser.add_argument(
        "--min-expiries-per-minute",
        type=int,
        default=int(config_defaults.get("min_expiries_per_minute", 1)),
        help="Minimum valid expiries required to calibrate one minute surface.",
    )
    parser.add_argument(
        "--max-precalib-iv",
        type=float,
        default=float(config_defaults.get("max_precalib_iv", DEFAULT_MAX_PRECALIB_IV)),
        help="Maximum implied vol allowed into minute surface inputs; <= 0 disables the filter.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=int(config_defaults.get("max_files", 0)),
        help="Optional debug cap on input files (0 means no cap).",
    )
    parser.add_argument(
        "--max-minutes",
        type=int,
        default=int(config_defaults.get("max_minutes", 0)),
        help="Optional debug cap on processed minutes (0 means no cap).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(config_defaults.get("chunk_size", 100000)),
        help="CSV chunk size.",
    )
    parser.add_argument(
        "--calibration-workers",
        type=int,
        default=int(config_defaults.get("calibration_workers", DEFAULT_CALIBRATION_WORKERS)),
        help="CPU worker processes for SVI window/excel calibration (0 means serial).",
    )
    parser.add_argument(
        "--save-precalib-csv",
        dest="save_precalib_csv",
        action="store_true",
        default=default_save_precalib_csv,
        help="If set, save pre-calibration SVI input points to CSV.",
    )
    parser.add_argument(
        "--no-save-precalib-csv",
        dest="save_precalib_csv",
        action="store_false",
        help="Disable saving pre-calibration SVI input points CSV.",
    )
    parser.add_argument(
        "--precalib-csv",
        type=str,
        default=str(config_defaults.get("precalib_csv", f"data/processed/{DEFAULT_SURFACE_MODEL}-{DEFAULT_DATA_RANGE}/surface-{DEFAULT_SURFACE_MODEL}-{DEFAULT_DATA_RANGE}-precalib-points.csv")),
        help="Path for pre-calibration SVI input points CSV.",
    )
    parser.add_argument(
        "--pricing-model",
        choices=sorted(SUPPORTED_PRICING_MODELS),
        default=str(config_defaults.get("pricing_model", DEFAULT_PRICING_MODEL)),
        help="Option pricing model used for implied-volatility inversion.",
    )
    parser.add_argument(
        "--rate-curve-path",
        default=str(config_defaults.get("rate_curve_path", DEFAULT_RATE_CURVE_PATH)),
        help="Frozen U.S. Treasury curve used by Black-76.",
    )
    parser.add_argument(
        "--max-rate-staleness-days",
        type=int,
        default=int(
            config_defaults.get(
                "max_rate_staleness_days",
                DEFAULT_MAX_RATE_STALENESS_DAYS,
            )
        ),
    )
    parser.add_argument(
        "--max-underlying-staleness-seconds",
        type=int,
        default=int(
            config_defaults.get(
                "max_underlying_staleness_seconds",
                DEFAULT_MAX_UNDERLYING_STALENESS_SECONDS,
            )
        ),
    )
    parser.add_argument(
        "--underlying-match-mode",
        choices=["last_prior_trade"],
        default=str(config_defaults.get("underlying_match_mode", "last_prior_trade")),
    )
    parser.add_argument(
        "--option-filter-mode",
        choices=sorted(SUPPORTED_OPTION_FILTER_MODES),
        default=str(
            config_defaults.get("option_filter_mode", DEFAULT_OPTION_FILTER_MODE)
        ),
    )
    parser.add_argument(
        "--max-itm-moneyness-distance",
        type=float,
        default=float(
            config_defaults.get(
                "max_itm_moneyness_distance",
                DEFAULT_MAX_ITM_MONEYNESS_DISTANCE,
            )
        ),
        help=(
            "Maximum abs(strike / futures - 1) for ITM fallback observations. "
            "Used only by otm_preferred_itm_fallback."
        ),
    )
    parser.add_argument(
        "--iv-aggregation-mode",
        choices=sorted(SUPPORTED_IV_AGGREGATION_MODES),
        default=str(
            config_defaults.get(
                "iv_aggregation_mode",
                DEFAULT_IV_AGGREGATION_MODE,
            )
        ),
    )
    parser.add_argument(
        "--window-audit-json",
        default=str(config_defaults.get("window_audit_json", "")),
        help="Optional path for per-target temporal-window audit JSON.",
    )
    args = parser.parse_args(argv_list)
    args.config = str(_resolve_config_path(args.config))
    args.model = _normalize_surface_model(args.model)
    args.data_range = _normalize_data_range(args.data_range)
    args.pricing_model = _normalize_pricing_model(args.pricing_model)
    args.option_filter_mode = _normalize_option_filter_mode(args.option_filter_mode)
    args.iv_aggregation_mode = _normalize_iv_aggregation_mode(args.iv_aggregation_mode)
    args.run_ts = str(args.run_ts).strip() or _default_run_ts()
    args.calibration_workers = int(args.calibration_workers)
    if args.calibration_workers < 0:
        raise ValueError(
            f"Invalid --calibration-workers: {args.calibration_workers}. Expected >= 0."
        )
    if args.max_underlying_staleness_seconds < 0:
        raise ValueError("--max-underlying-staleness-seconds must be >= 0")
    if args.max_rate_staleness_days < 0:
        raise ValueError("--max-rate-staleness-days must be >= 0")
    if not 0.0 < args.max_itm_moneyness_distance < 1.0:
        raise ValueError(
            "--max-itm-moneyness-distance must be strictly between 0 and 1"
        )
    if args.model == "raw" and args.min_expiries_per_minute < 2:
        raise ValueError(
            "Corrected raw-vol surfaces require --min-expiries-per-minute >= 2"
        )

    resolved_defaults = _load_generate_surface_config(
        args.config,
        runtime_overrides={
            "model": args.model,
            "data_range": args.data_range,
            "run_ts": args.run_ts,
        },
    )
    args.output_dir = str(resolved_defaults.get("output_dir", Path(args.output_json).parent))
    args.resolved_config_path = str(
        resolved_defaults.get(
            "resolved_config_path",
            Path(args.output_dir) / RESOLVED_CONFIG_FILENAME,
        )
    )
    return args


def _to_utc_minute_string(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_utc_datetime_string(value: pd.Timestamp | datetime) -> str:
    ts = _to_utc_timestamp(value)
    iso = ts.isoformat()
    if iso.endswith("+00:00"):
        return iso[:-6] + "Z"
    return iso


def _parse_data_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid --data-date: {value}. Expected YYYY-MM-DD.") from exc


def _to_json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_native(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_native(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _option_type_name(option_type: Any) -> str:
    name = getattr(option_type, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    text = str(option_type)
    if "." in text:
        text = text.split(".")[-1]
    return text.upper()


def _evaluate_precalib_filter(implied_vol: float, max_precalib_iv: float) -> Tuple[bool, str]:
    if max_precalib_iv > 0 and implied_vol > max_precalib_iv:
        return False, "implied_vol_above_cap"
    return True, ""


def _weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    pairs = sorted(
        (
            (float(value), max(float(weight), 0.0))
            for value, weight in zip(values, weights)
            if math.isfinite(float(value)) and math.isfinite(float(weight))
        ),
        key=lambda item: item[0],
    )
    if not pairs:
        return math.nan
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return float(np.median([value for value, _ in pairs]))
    threshold = total_weight / 2.0
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def _log_cli_arguments(args: argparse.Namespace) -> None:
    logger.info("CLI argv: %s", " ".join(shlex.quote(x) for x in sys.argv))
    logger.info("Parsed CLI arguments:")
    for key in sorted(vars(args)):
        logger.info("  %s=%r", key, getattr(args, key))


def _build_resolved_config_payload(args: argparse.Namespace) -> Dict[str, Any]:
    model = _normalize_surface_model(getattr(args, "model", DEFAULT_SURFACE_MODEL))
    data_range = _normalize_data_range(getattr(args, "data_range", DEFAULT_DATA_RANGE))
    run_ts = str(getattr(args, "run_ts", "")).strip() or _default_run_ts()
    output_dir = str(getattr(args, "output_dir", Path(getattr(args, "output_json")).parent))
    section_payload: Dict[str, Any] = {
        "model": model,
        "data_range": data_range,
        "run_ts": run_ts,
        "input_glob": str(args.input_glob),
        "output_dir": output_dir,
        "output_json": str(args.output_json),
        "log_file": str(args.log_file),
        "resolved_config_path": str(getattr(args, "resolved_config_path", Path(output_dir) / RESOLVED_CONFIG_FILENAME)),
        "data_date": str(args.data_date),
        "expiration_time_utc": str(args.expiration_time_utc),
        "days_in_year": int(args.days_in_year),
        "min_strikes_per_expiry": int(args.min_strikes_per_expiry),
        "min_expiries_per_minute": int(args.min_expiries_per_minute),
        "max_precalib_iv": float(args.max_precalib_iv),
        "max_files": int(args.max_files),
        "max_minutes": int(args.max_minutes),
        "chunk_size": int(args.chunk_size),
        "calibration_workers": int(args.calibration_workers),
        "save_precalib_csv": bool(args.save_precalib_csv),
        "precalib_csv": str(args.precalib_csv),
        "pricing_model": str(
            getattr(args, "pricing_model", DEFAULT_PRICING_MODEL)
        ),
        "rate_curve_path": str(
            getattr(args, "rate_curve_path", DEFAULT_RATE_CURVE_PATH)
        ),
        "max_rate_staleness_days": int(
            getattr(
                args,
                "max_rate_staleness_days",
                DEFAULT_MAX_RATE_STALENESS_DAYS,
            )
        ),
        "max_underlying_staleness_seconds": int(
            getattr(
                args,
                "max_underlying_staleness_seconds",
                DEFAULT_MAX_UNDERLYING_STALENESS_SECONDS,
            )
        ),
        "underlying_match_mode": str(
            getattr(args, "underlying_match_mode", "last_prior_trade")
        ),
        "option_filter_mode": str(
            getattr(args, "option_filter_mode", DEFAULT_OPTION_FILTER_MODE)
        ),
        "max_itm_moneyness_distance": float(
            getattr(
                args,
                "max_itm_moneyness_distance",
                DEFAULT_MAX_ITM_MONEYNESS_DISTANCE,
            )
        ),
        "iv_aggregation_mode": str(
            getattr(
                args,
                "iv_aggregation_mode",
                DEFAULT_IV_AGGREGATION_MODE,
            )
        ),
        "window_audit_json": str(getattr(args, "window_audit_json", "")),
    }
    optional_keys = (
        "target_datetimes",
        "target_datetimes_file",
        "window_minutes",
        "target_xlsx",
        "sheet_name",
        "date_column",
        "time_column",
        "source_timezone",
        "publication_availability_lag_minutes",
        "max_target_datetimes",
        "news_alignment_mode",
        "intraday_tolerance_minutes",
        "max_session_shift_minutes",
        "require_complete_pair",
        "require_origin_not_before_news",
        "include_session_shifted",
        "collision_policy",
    )
    for key in optional_keys:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                section_payload[key] = _to_json_native(value)
    payload: Dict[str, Any] = {
        "surface_builder": {
            "job": "generate_surface",
            "generate_surface": section_payload,
        },
        "runtime": {
            "config_path": str(args.config),
            "cli_argv": list(sys.argv),
        },
    }
    return payload


def _write_resolved_config(args: argparse.Namespace) -> Path:
    return write_yaml_mapping(
        _build_resolved_config_payload(args),
        Path(getattr(args, "resolved_config_path", Path(getattr(args, "output_dir", Path(args.output_json).parent)) / RESOLVED_CONFIG_FILENAME)),
    )


def _get_target_future_month_code(current_month: int) -> str:
    if current_month <= 3:
        return "H"
    if current_month <= 6:
        return "M"
    if current_month <= 9:
        return "U"
    return "Z"


def _infer_file_date_range(path_value: Path | str) -> Tuple[Optional[date], Optional[date]]:
    matched = FILE_DATE_RANGE_RE.search(Path(path_value).name)
    if not matched:
        return None, None
    start_raw, end_raw = matched.groups()
    return date.fromisoformat(start_raw), date.fromisoformat(end_raw)


def _get_file_target_future_month_code(path_value: Path | str) -> Optional[str]:
    start_date, _ = _infer_file_date_range(path_value)
    if start_date is None:
        return None
    return _get_target_future_month_code(start_date.month)


def _to_utc_timestamp(value: pd.Timestamp | datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _make_expiry_dt_utc(
    expiry_date: date,
    expiration_time_utc: Optional[dt_time] = None,
) -> datetime:
    target_time = expiration_time_utc or _parse_expiration_time_utc(DEFAULT_EXPIRATION_TIME_UTC)
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)
    return datetime.combine(expiry_date, target_time)


def _coerce_expiry_dt_utc(
    expiry_value: date | datetime,
    expiration_time_utc: dt_time,
) -> Tuple[date, datetime]:
    if isinstance(expiry_value, datetime):
        expiry_ts = pd.Timestamp(expiry_value)
        if expiry_ts.tzinfo is None:
            expiry_ts = expiry_ts.tz_localize("UTC")
        else:
            expiry_ts = expiry_ts.tz_convert("UTC")
        return expiry_ts.date(), expiry_ts.to_pydatetime(warn=False)
    return expiry_value, _make_expiry_dt_utc(expiry_value, expiration_time_utc)


def _make_spot_cache_key(
    underlying: str,
    target_future_month_code: Optional[str],
    target_future_year: Optional[int] = None,
) -> SpotKey:
    base_key = (
        str(underlying or "").upper(),
        (target_future_month_code or "").upper(),
    )
    if target_future_year is None:
        return base_key
    return (*base_key, int(target_future_year))


def _pricing_context_from_args(
    args: argparse.Namespace,
    **window_overrides: str,
) -> SurfacePricingContext:
    return SurfacePricingContext(
        pricing_model=_normalize_pricing_model(
            getattr(args, "pricing_model", DEFAULT_PRICING_MODEL)
        ),
        rate_curve_path=str(
            getattr(args, "rate_curve_path", DEFAULT_RATE_CURVE_PATH)
        ),
        max_rate_staleness_days=int(
            getattr(
                args,
                "max_rate_staleness_days",
                DEFAULT_MAX_RATE_STALENESS_DAYS,
            )
        ),
        max_underlying_staleness_seconds=int(
            getattr(
                args,
                "max_underlying_staleness_seconds",
                DEFAULT_MAX_UNDERLYING_STALENESS_SECONDS,
            )
        ),
        underlying_match_mode=str(
            getattr(args, "underlying_match_mode", "last_prior_trade")
        ),
        option_filter_mode=_normalize_option_filter_mode(
            getattr(args, "option_filter_mode", DEFAULT_OPTION_FILTER_MODE)
        ),
        max_itm_moneyness_distance=float(
            getattr(
                args,
                "max_itm_moneyness_distance",
                DEFAULT_MAX_ITM_MONEYNESS_DISTANCE,
            )
        ),
        iv_aggregation_mode=_normalize_iv_aggregation_mode(
            getattr(args, "iv_aggregation_mode", DEFAULT_IV_AGGREGATION_MODE)
        ),
        target_datetime_utc=str(window_overrides.get("target_datetime_utc", "")),
        window_side=str(window_overrides.get("window_side", "")),
        window_start_utc=str(window_overrides.get("window_start_utc", "")),
        window_end_utc=str(window_overrides.get("window_end_utc", "")),
    )


@lru_cache(maxsize=8)
def _load_rate_curve(
    path: str,
    max_staleness_days: int,
) -> TreasuryParYieldCurve:
    return TreasuryParYieldCurve(
        path,
        max_staleness_days=int(max_staleness_days),
    )


def _get_ty_option_underlying_future_month_code(
    option_month_code: Optional[str] = None,
    expiry_date: Optional[date] = None,
) -> Optional[str]:
    normalized_option_month = str(option_month_code or "").strip().upper()
    if normalized_option_month:
        month_name = option_maturity_month_map.get(normalized_option_month)
        if month_name:
            calendar_month = month_map.get(month_name)
            if calendar_month is not None:
                return _get_target_future_month_code(int(calendar_month))

    if expiry_date is not None:
        return _get_target_future_month_code(int(expiry_date.month))

    return None


def _resolve_option_target_future_month_code(
    meta: ContractMeta,
    fallback_target_future_month_code: Optional[str],
) -> str:
    normalized_fallback = (fallback_target_future_month_code or "").upper()
    if meta.contract_type != "option":
        return normalized_fallback

    if str(meta.underlying or "").upper() == "TY":
        resolved = _get_ty_option_underlying_future_month_code(
            option_month_code=meta.maturity_month_code,
            expiry_date=meta.expiry_date,
        )
        if resolved:
            return resolved

    return normalized_fallback


def _resolve_future_year(
    *,
    contract: FutureContract,
    trade_date: date,
) -> int:
    month_name = future_maturity_month_map.get(
        str(contract.get_maturity_month_code()).upper()
    )
    if month_name is None:
        raise ValueError(
            f"Unsupported futures month code: {contract.get_maturity_month_code()!r}"
        )
    return resolve_contract_year(
        contract.get_maturity_year_code(),
        trade_date=trade_date,
        contract_month=int(month_map[month_name]),
    )


def _build_contract_meta(
    trade_do: TradeDataDO,
    expiry_inference_date: date,
    calendar,
    expiration_time_utc: dt_time,
    pricing_model: str = DEFAULT_PRICING_MODEL,
) -> Optional[ContractMeta]:
    contract = trade_do.to_contract()
    trade_date = _to_utc_timestamp(trade_do.get_data_time()).date()
    if isinstance(contract, FutureContract):
        return ContractMeta(
            contract_type="future",
            underlying=contract.get_underlying(),
            maturity_month_code=contract.get_maturity_month_code(),
            maturity_year=_resolve_future_year(
                contract=contract,
                trade_date=trade_date,
            ),
            contract_id=trade_do.contract_id,
        )

    if isinstance(contract, OptionContract):
        if (
            str(contract.get_underlying()).upper() == "TY"
            and _normalize_pricing_model(pricing_model) == "black76"
        ):
            ty_dates = resolve_ty_option_contract_dates(
                option_month_code=contract.get_maturity_month_code(),
                option_year_code=contract.get_maturity_year_code(),
                trade_date=trade_date,
            )
            expiry_date = ty_dates.last_trading_date
            expiry_dt_utc = ty_dates.last_trading_datetime_utc
            maturity_year = ty_dates.named_year
            underlying_future_month_code = (
                ty_dates.underlying_future_month_code
            )
            underlying_future_year = ty_dates.underlying_future_year
        else:
            expiry = contract.get_contract_maturity_dates_by_contract_id(
                data_date=expiry_inference_date,
                calendars=[calendar],
                termination_rule=ContractTerminationRule.EndOfMonth,
                expiration_time=expiration_time_utc,
            )
            expiry_date, expiry_dt_utc = _coerce_expiry_dt_utc(
                expiry,
                expiration_time_utc,
            )
            maturity_year = expiry_date.year
            underlying_future_month_code = (
                _get_ty_option_underlying_future_month_code(
                    option_month_code=contract.get_maturity_month_code(),
                    expiry_date=expiry_date,
                )
            )
            underlying_future_year = expiry_date.year
        return ContractMeta(
            contract_type="option",
            underlying=contract.get_underlying(),
            maturity_month_code=contract.get_maturity_month_code(),
            maturity_year=maturity_year,
            strike=float(contract.get_strike()),
            option_type=contract.get_option_type(),
            expiry_date=expiry_date,
            expiry_dt_utc=expiry_dt_utc,
            underlying_future_month_code=underlying_future_month_code,
            underlying_future_year=underlying_future_year,
            contract_id=trade_do.contract_id,
        )

    return None


def _collect_minute_spot(
    rows: List[MinuteTradeRow],
    target_future_month_code: Optional[str],
    last_spot_by_key: Dict[Tuple[str, str], float],
    stats: Dict[str, int],
) -> Tuple[Dict[Tuple[str, str], float], List[MinuteTradeRow]]:
    return _collect_spot_and_option_rows(
        rows=rows,
        target_future_month_code=target_future_month_code,
        last_spot_by_key=last_spot_by_key,
        stats=stats,
    )


def _collect_spot_and_option_rows(
    rows: List[MinuteTradeRow],
    target_future_month_code: Optional[str],
    last_spot_by_key: Optional[Dict[Tuple[str, str], float]] = None,
    stats: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[Tuple[str, str], float], List[MinuteTradeRow]]:
    fut_sum: Dict[Tuple[str, str], float] = defaultdict(float)
    fut_vol: Dict[Tuple[str, str], float] = defaultdict(float)
    option_rows: List[MinuteTradeRow] = []
    normalized_target_month = (target_future_month_code or "").upper()

    for row in rows:
        meta = row.meta
        if meta.contract_type == "future":
            if stats is not None:
                stats["future_rows"] += 1
            future_month_code = (meta.maturity_month_code or "").upper()
            if (
                normalized_target_month
                and str(meta.underlying or "").upper() != "TY"
                and future_month_code
                and future_month_code != normalized_target_month
            ):
                if stats is not None:
                    stats["skip_future_non_target_month"] += 1
                continue
            spot_key = _make_spot_cache_key(
                underlying=meta.underlying,
                target_future_month_code=future_month_code or normalized_target_month,
            )
            fut_sum[spot_key] += row.price * row.volume
            fut_vol[spot_key] += row.volume
        elif meta.contract_type == "option":
            option_rows.append(row)
            if stats is not None:
                stats["option_rows"] += 1

    minute_spot: Dict[Tuple[str, str], float] = {}
    for spot_key, total_pxv in fut_sum.items():
        total_vol = fut_vol.get(spot_key, 0.0)
        if total_vol <= 0:
            continue
        spot = total_pxv / total_vol
        if not math.isfinite(spot) or spot <= 0:
            continue
        minute_spot[spot_key] = spot
        if last_spot_by_key is not None:
            last_spot_by_key[spot_key] = spot
    return minute_spot, option_rows


def _update_last_underlying_trade(
    rows: Iterable[MinuteTradeRow],
    state: Dict[SpotKey, UnderlyingTrade],
) -> None:
    """Update exact-contract futures state using chronologically visible trades."""

    for row in sorted(rows, key=lambda item: _to_utc_timestamp(item.trade_ts)):
        meta = row.meta
        if meta.contract_type != "future":
            continue
        key = _make_spot_cache_key(
            meta.underlying,
            meta.maturity_month_code,
            meta.maturity_year,
        )
        trade = UnderlyingTrade(
            trade_ts=_to_utc_timestamp(row.trade_ts),
            price=float(row.price),
            contract_id=str(meta.contract_id or ""),
        )
        previous = state.get(key)
        if previous is None or trade.trade_ts >= previous.trade_ts:
            state[key] = trade


def _build_underlying_timelines(
    rows: Iterable[MinuteTradeRow],
    initial_state: Mapping[SpotKey, UnderlyingTrade],
) -> Dict[SpotKey, List[UnderlyingTrade]]:
    timelines: Dict[SpotKey, List[UnderlyingTrade]] = defaultdict(list)
    for key, trade in initial_state.items():
        if isinstance(trade, UnderlyingTrade):
            timelines[key].append(trade)
    for row in rows:
        meta = row.meta
        if meta.contract_type != "future":
            continue
        key = _make_spot_cache_key(
            meta.underlying,
            meta.maturity_month_code,
            meta.maturity_year,
        )
        timelines[key].append(
            UnderlyingTrade(
                trade_ts=_to_utc_timestamp(row.trade_ts),
                price=float(row.price),
                contract_id=str(meta.contract_id or ""),
            )
        )
    for key in list(timelines):
        deduped = {
            (
                int(trade.trade_ts.value),
                float(trade.price),
                trade.contract_id,
            ): trade
            for trade in timelines[key]
        }
        timelines[key] = sorted(
            deduped.values(),
            key=lambda trade: trade.trade_ts,
        )
    return dict(timelines)


def _find_last_prior_underlying_trade(
    timeline: List[UnderlyingTrade],
    option_trade_ts: pd.Timestamp,
) -> Optional[UnderlyingTrade]:
    option_ns = int(_to_utc_timestamp(option_trade_ts).value)
    low = 0
    high = len(timeline)
    while low < high:
        middle = (low + high) // 2
        if int(timeline[middle].trade_ts.value) <= option_ns:
            low = middle + 1
        else:
            high = middle
    return timeline[low - 1] if low > 0 else None


def _is_otm_option(option_type: Any, strike: float, futures_price: float) -> bool:
    option_name = _option_type_name(option_type)
    if option_name == "CALL":
        return float(strike) >= float(futures_price)
    if option_name == "PUT":
        return float(strike) < float(futures_price)
    raise ValueError(f"Unsupported option type: {option_type!r}")


def _is_within_itm_fallback_range(
    strike: float,
    futures_price: float,
    max_distance: float,
) -> bool:
    if futures_price <= 0:
        return False
    return bool(
        abs((float(strike) / float(futures_price)) - 1.0)
        <= float(max_distance) + 1.0e-12
    )


def _pricing_tau_act365(
    trade_ts: pd.Timestamp,
    expiry_dt_utc: datetime,
) -> float:
    trade = _to_utc_timestamp(trade_ts)
    expiry = _to_utc_timestamp(expiry_dt_utc)
    return float((expiry - trade).total_seconds() / (365.0 * SECONDS_PER_DAY))


def _day_fraction_utc(value: pd.Timestamp | datetime) -> float:
    ts_utc = _to_utc_timestamp(value)
    midnight = ts_utc.normalize()
    return (ts_utc - midnight).total_seconds() / SECONDS_PER_DAY


def _tau_years_from_trade_to_expiry(
    trade_ts: pd.Timestamp,
    expiry_dt_utc: datetime,
    vol_daycount: DayCountBusN,
) -> float:
    trade_ts_utc = _to_utc_timestamp(trade_ts)
    expiry_ts_utc = _to_utc_timestamp(expiry_dt_utc)
    base_tau = float(vol_daycount(trade_ts_utc.date(), expiry_ts_utc.date()))
    quote_fraction = _day_fraction_utc(trade_ts_utc)
    expiry_fraction = _day_fraction_utc(expiry_ts_utc)
    return float(
        base_tau + (expiry_fraction - quote_fraction) / float(vol_daycount.days_in_year)
    )


def _prepare_option_candidates(
    minute_ts: pd.Timestamp,
    option_rows: List[MinuteTradeRow],
    minute_spot: Mapping[Any, Any],
    last_spot_by_key: Mapping[Any, Any],
    target_future_month_code: Optional[str],
    vol_daycount: DayCountBusN,
    calendar,
    stats: Dict[str, int],
    tau_anchor_ts: Optional[pd.Timestamp] = None,
    all_rows: Optional[List[MinuteTradeRow]] = None,
    pricing_context: Optional[SurfacePricingContext] = None,
    rejected_audit_rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[date, List[MinuteOptionCandidate]]:
    minute_ts_utc = _to_utc_timestamp(minute_ts)
    valuation_date = minute_ts_utc.date()
    pricing_context = pricing_context or SurfacePricingContext()
    corrected_black76 = pricing_context.pricing_model == "black76"
    underlying_timelines = (
        _build_underlying_timelines(
            all_rows or option_rows,
            {
                key: value
                for key, value in last_spot_by_key.items()
                if isinstance(key, tuple)
                and len(key) == 3
                and isinstance(value, UnderlyingTrade)
            },
        )
        if corrected_black76
        else {}
    )
    rate_curve = (
        _load_rate_curve(
            pricing_context.rate_curve_path,
            pricing_context.max_rate_staleness_days,
        )
        if corrected_black76
        else None
    )

    def reject(
        row: MinuteTradeRow,
        reason: str,
        *,
        underlying: Optional[UnderlyingTrade] = None,
        staleness_seconds: float = math.nan,
        is_otm: Optional[bool] = None,
    ) -> None:
        stats[f"skip_{reason}"] += 1
        if rejected_audit_rows is None:
            return
        meta = row.meta
        rejected_audit_rows.append(
            {
                "target_datetime_utc": pricing_context.target_datetime_utc,
                "window_side": pricing_context.window_side,
                "window_start_utc": pricing_context.window_start_utc,
                "window_end_utc": pricing_context.window_end_utc,
                "trade_datetime_utc": _to_utc_datetime_string(row.trade_ts),
                "calibration_datetime_utc": _to_utc_minute_string(minute_ts),
                "business_days": "",
                "maturity_date": (
                    meta.expiry_date.isoformat()
                    if meta.expiry_date is not None
                    else ""
                ),
                "expiration_datetime_utc": (
                    _to_utc_datetime_string(meta.expiry_dt_utc)
                    if meta.expiry_dt_utc is not None
                    else ""
                ),
                "contract_id": meta.contract_id or "",
                "option_type": _option_type_name(meta.option_type),
                "strike": float(meta.strike or 0.0),
                "price": float(row.price),
                "spot": float(underlying.price) if underlying is not None else "",
                "underlying_contract_id": (
                    underlying.contract_id if underlying is not None else ""
                ),
                "underlying_trade_datetime_utc": (
                    _to_utc_datetime_string(underlying.trade_ts)
                    if underlying is not None
                    else ""
                ),
                "underlying_staleness_seconds": (
                    float(staleness_seconds)
                    if math.isfinite(staleness_seconds)
                    else ""
                ),
                "underlying_match_mode": pricing_context.underlying_match_mode,
                "percent_strike": (
                    float(meta.strike or 0.0) / float(underlying.price)
                    if underlying is not None and underlying.price > 0
                    else ""
                ),
                "pricing_model": pricing_context.pricing_model,
                "rate_curve_date": "",
                "continuous_rate": "",
                "discount_factor": "",
                "rate_curve_sha256": "",
                "implied_vol": "",
                "is_otm": (
                    "true" if is_otm else "false"
                    if is_otm is not None
                    else ""
                ),
                "surface_input_role": "",
                "passes_precalib_filter": "false",
                "filter_reason": reason,
                "weight": float(row.volume),
            }
        )

    candidates: List[MinuteOptionCandidate] = []
    for row in option_rows:
        meta = row.meta
        assert meta.contract_type == "option"

        underlying_trade: Optional[UnderlyingTrade] = None
        underlying_staleness_seconds = math.nan
        if corrected_black76:
            spot_month_code = (
                meta.underlying_future_month_code
                or _resolve_option_target_future_month_code(
                    meta,
                    fallback_target_future_month_code=target_future_month_code,
                )
            )
            spot_key = _make_spot_cache_key(
                meta.underlying,
                spot_month_code,
                meta.underlying_future_year,
            )
            underlying_trade = _find_last_prior_underlying_trade(
                underlying_timelines.get(spot_key, []),
                row.trade_ts,
            )
            if underlying_trade is None:
                reject(row, "no_prior_underlying")
                continue
            underlying_staleness_seconds = float(
                (
                    _to_utc_timestamp(row.trade_ts)
                    - underlying_trade.trade_ts
                ).total_seconds()
            )
            if underlying_staleness_seconds < 0:
                reject(
                    row,
                    "future_underlying_lookup",
                    underlying=underlying_trade,
                    staleness_seconds=underlying_staleness_seconds,
                )
                continue
            if (
                underlying_staleness_seconds
                > pricing_context.max_underlying_staleness_seconds
            ):
                reject(
                    row,
                    "stale_underlying",
                    underlying=underlying_trade,
                    staleness_seconds=underlying_staleness_seconds,
                )
                continue
            spot = float(underlying_trade.price)
        else:
            spot_month_code = _resolve_option_target_future_month_code(
                meta,
                fallback_target_future_month_code=target_future_month_code,
            )
            legacy_key = _make_spot_cache_key(
                meta.underlying,
                spot_month_code,
                None,
            )
            spot = minute_spot.get(
                legacy_key,
                last_spot_by_key.get(legacy_key),
            )
            if isinstance(spot, UnderlyingTrade):
                spot = spot.price
            if spot is None or not math.isfinite(float(spot)) or float(spot) <= 0:
                stats["skip_no_spot"] += 1
                continue
            spot = float(spot)

        expiry_date = meta.expiry_date
        expiry_dt_utc = meta.expiry_dt_utc
        if expiry_date is None or expiry_dt_utc is None:
            reject(row, "missing_expiry", underlying=underlying_trade)
            continue

        surface_anchor_ts = (
            _to_utc_timestamp(tau_anchor_ts)
            if tau_anchor_ts is not None
            else minute_ts_utc
        )
        trade_ts_utc = _to_utc_timestamp(row.trade_ts)
        business_days = int(
            calendar.count_business_days(
                surface_anchor_ts.date(),
                expiry_date,
                include_start=False,
                include_end=True,
            )
        )
        if business_days <= 0:
            reject(row, "tau_nonpositive", underlying=underlying_trade)
            continue

        tau = (
            _pricing_tau_act365(trade_ts_utc, expiry_dt_utc)
            if corrected_black76
            else _tau_years_from_trade_to_expiry(
                surface_anchor_ts
                if tau_anchor_ts is not None
                else trade_ts_utc,
                expiry_dt_utc,
                vol_daycount,
            )
        )
        if tau <= 0:
            reject(row, "tau_nonpositive", underlying=underlying_trade)
            continue

        strike = float(meta.strike or 0.0)
        if not math.isfinite(strike) or strike <= 0:
            reject(row, "invalid_strike", underlying=underlying_trade)
            continue

        is_otm = _is_otm_option(meta.option_type, strike, spot)
        if pricing_context.option_filter_mode == "otm_only" and not is_otm:
            reject(
                row,
                "not_otm",
                underlying=underlying_trade,
                staleness_seconds=underlying_staleness_seconds,
                is_otm=False,
            )
            continue
        if (
            pricing_context.option_filter_mode
            == "otm_preferred_itm_fallback"
            and not is_otm
            and not _is_within_itm_fallback_range(
                strike,
                spot,
                pricing_context.max_itm_moneyness_distance,
            )
        ):
            reject(
                row,
                "itm_outside_moneyness_range",
                underlying=underlying_trade,
                staleness_seconds=underlying_staleness_seconds,
                is_otm=False,
            )
            continue

        if corrected_black76:
            assert rate_curve is not None
            try:
                curve_point = rate_curve.point(trade_ts_utc.date(), tau)
            except Exception:
                reject(
                    row,
                    "rate_curve",
                    underlying=underlying_trade,
                    staleness_seconds=underlying_staleness_seconds,
                    is_otm=is_otm,
                )
                continue
            try:
                lower_bound, upper_bound = black76_no_arbitrage_bounds(
                    futures_price=spot,
                    strike=strike,
                    discount_factor=curve_point.discount_factor,
                    option_type=meta.option_type,
                )
            except Exception:
                reject(
                    row,
                    "price_bounds",
                    underlying=underlying_trade,
                    staleness_seconds=underlying_staleness_seconds,
                    is_otm=is_otm,
                )
                continue
            tolerance = 1.0e-10 * max(1.0, upper_bound)
            if (
                float(row.price) < lower_bound - tolerance
                or float(row.price) > upper_bound + tolerance
            ):
                reject(
                    row,
                    "price_bounds",
                    underlying=underlying_trade,
                    staleness_seconds=underlying_staleness_seconds,
                    is_otm=is_otm,
                )
                continue
            rate_curve_date = curve_point.curve_date
            continuous_rate = curve_point.continuous_rate
            discount_factor = curve_point.discount_factor
            rate_curve_sha256 = curve_point.source_sha256
        else:
            rate_curve_date = None
            continuous_rate = 0.0
            discount_factor = 1.0
            rate_curve_sha256 = ""

        candidates.append(
            MinuteOptionCandidate(
                meta=meta,
                price=float(row.price),
                weight=float(row.volume),
                strike=strike,
                spot=float(spot),
                tau=float(tau),
                business_days=business_days,
                trade_ts=_to_utc_timestamp(row.trade_ts),
                underlying_trade_ts=(
                    underlying_trade.trade_ts
                    if underlying_trade is not None
                    else None
                ),
                underlying_contract_id=(
                    underlying_trade.contract_id
                    if underlying_trade is not None
                    else ""
                ),
                underlying_staleness_seconds=underlying_staleness_seconds,
                pricing_model=pricing_context.pricing_model,
                rate_curve_date=rate_curve_date,
                continuous_rate=float(continuous_rate),
                discount_factor=float(discount_factor),
                rate_curve_sha256=rate_curve_sha256,
                is_otm=bool(is_otm),
                target_datetime_utc=pricing_context.target_datetime_utc,
                window_side=pricing_context.window_side,
                window_start_utc=pricing_context.window_start_utc,
                window_end_utc=pricing_context.window_end_utc,
            )
        )

    return valuation_date, candidates


def _apply_otm_preferred_itm_fallback(
    records: List[Dict[str, Any]],
    stats: Dict[str, int],
) -> None:
    """Select bounded ITM observations only where OTM coverage is missing."""

    def exclude(record: Dict[str, Any], reason: str) -> None:
        if not record["selected"]:
            return
        record["selected"] = False
        record["audit_row"]["passes_precalib_filter"] = "false"
        record["audit_row"]["surface_input_role"] = ""
        record["audit_row"]["filter_reason"] = reason
        stats[f"skip_{reason}"] += 1

    by_maturity_strike: Dict[
        Tuple[int, float],
        List[Dict[str, Any]],
    ] = defaultdict(list)
    for record in records:
        if record["selected"]:
            candidate = record["candidate"]
            by_maturity_strike[
                (int(candidate.business_days), float(candidate.strike))
            ].append(record)

    # A valid OTM observation owns its strike. ITM is considered only when
    # that strike has no usable OTM observation.
    for strike_records in by_maturity_strike.values():
        if any(
            record["selected"] and record["candidate"].is_otm
            for record in strike_records
        ):
            for record in strike_records:
                if record["selected"] and not record["candidate"].is_otm:
                    exclude(record, "itm_shadowed_by_otm")

    by_maturity: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["selected"]:
            by_maturity[int(record["candidate"].business_days)].append(record)

    for maturity_records in by_maturity.values():
        otm_strikes = {
            float(record["candidate"].strike)
            for record in maturity_records
            if record["candidate"].is_otm
        }
        itm_by_strike: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        for record in maturity_records:
            if not record["candidate"].is_otm:
                itm_by_strike[float(record["candidate"].strike)].append(record)

        if not itm_by_strike:
            continue
        if not otm_strikes:
            for strike_records in itm_by_strike.values():
                for record in strike_records:
                    exclude(record, "itm_no_otm_anchor")
            continue

        # Keep the nearest-to-ATM fallback strikes and cap their number at the
        # number of OTM strikes, so ITM never dominates a maturity slice.
        ranked_itm_strikes = sorted(
            itm_by_strike,
            key=lambda strike: (
                min(
                    abs(float(record["percent_strike"]) - 1.0)
                    for record in itm_by_strike[strike]
                ),
                strike,
            ),
        )
        allowed_itm_strikes = set(ranked_itm_strikes[: len(otm_strikes)])
        for strike, strike_records in itm_by_strike.items():
            if strike in allowed_itm_strikes:
                continue
            for record in strike_records:
                exclude(record, "itm_fallback_cap")

    for record in records:
        if not record["selected"]:
            continue
        record["audit_row"]["surface_input_role"] = (
            "otm" if record["candidate"].is_otm else "itm_fallback"
        )


def _finalize_minute_surface(
    minute_ts: pd.Timestamp,
    valuation_date: date,
    candidates: List[MinuteOptionCandidate],
    implied_vols: List[Optional[float]],
    min_strikes_per_expiry: int,
    min_expiries_per_minute: int,
    max_precalib_iv: float,
    vol_daycount: DayCountBusN,
    results: Dict[str, Dict[str, Any]],
    stats: Dict[str, int],
    precalib_writer: Optional[csv.DictWriter] = None,
    surface_model: str = DEFAULT_SURFACE_MODEL,
    iv_aggregation_mode: str = DEFAULT_IV_AGGREGATION_MODE,
    rejected_audit_rows: Optional[List[Dict[str, Any]]] = None,
    option_filter_mode: str = DEFAULT_OPTION_FILTER_MODE,
    max_itm_moneyness_distance: float = DEFAULT_MAX_ITM_MONEYNESS_DISTANCE,
) -> None:
    iv_aggregation_mode = _normalize_iv_aggregation_mode(iv_aggregation_mode)
    option_filter_mode = _normalize_option_filter_mode(option_filter_mode)
    grouped: Dict[int, Dict[float, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "iv_weighted_sum": 0.0,
                "weight_sum": 0.0,
                "price_weighted_sum": 0.0,
                "spot_weighted_sum": 0.0,
                "percent_strike_weighted_sum": 0.0,
                "iv_observations": [],
            }
        )
    )
    minute_key = _to_utc_minute_string(minute_ts)
    precalib_rows: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []

    for candidate, iv in zip(candidates, implied_vols):
        if iv is None or not math.isfinite(iv) or iv <= 0:
            stats["skip_iv_fail"] += 1
            continue

        percent_strike = candidate.strike / candidate.spot
        if not math.isfinite(percent_strike) or percent_strike <= 0:
            stats["skip_iv_fail"] += 1
            continue

        passes_precalib_filter, filter_reason = _evaluate_precalib_filter(
            implied_vol=float(iv),
            max_precalib_iv=float(max_precalib_iv),
        )
        if not passes_precalib_filter:
            stats["skip_precalib_iv_above_cap"] += 1

        trade_ts = candidate.trade_ts if candidate.trade_ts is not None else minute_ts
        audit_row = {
                "target_datetime_utc": candidate.target_datetime_utc,
                "window_side": candidate.window_side,
                "window_start_utc": candidate.window_start_utc,
                "window_end_utc": candidate.window_end_utc,
                "trade_datetime_utc": _to_utc_datetime_string(trade_ts),
                "calibration_datetime_utc": minute_key,
                "business_days": int(candidate.business_days),
                "maturity_date": (
                    candidate.meta.expiry_date.isoformat()
                    if candidate.meta.expiry_date is not None
                    else ""
                ),
                "expiration_datetime_utc": (
                    _to_utc_datetime_string(candidate.meta.expiry_dt_utc)
                    if candidate.meta.expiry_dt_utc is not None
                    else ""
                ),
                "contract_id": candidate.meta.contract_id or "",
                "option_type": _option_type_name(candidate.meta.option_type),
                "strike": float(candidate.strike),
                "price": float(candidate.price),
                "spot": float(candidate.spot),
                "underlying_contract_id": candidate.underlying_contract_id,
                "underlying_trade_datetime_utc": (
                    _to_utc_datetime_string(candidate.underlying_trade_ts)
                    if candidate.underlying_trade_ts is not None
                    else ""
                ),
                "underlying_staleness_seconds": (
                    float(candidate.underlying_staleness_seconds)
                    if math.isfinite(candidate.underlying_staleness_seconds)
                    else ""
                ),
                "underlying_match_mode": (
                    "last_prior_trade"
                    if candidate.pricing_model == "black76"
                    else "minute_vwap_legacy"
                ),
                "percent_strike": float(percent_strike),
                "pricing_model": candidate.pricing_model,
                "rate_curve_date": (
                    candidate.rate_curve_date.isoformat()
                    if candidate.rate_curve_date is not None
                    else ""
                ),
                "continuous_rate": float(candidate.continuous_rate),
                "discount_factor": float(candidate.discount_factor),
                "rate_curve_sha256": candidate.rate_curve_sha256,
                "implied_vol": float(iv),
                "is_otm": "true" if candidate.is_otm else "false",
                "surface_input_role": (
                    "otm" if candidate.is_otm else "unfiltered"
                )
                if passes_precalib_filter
                else "",
                "passes_precalib_filter": "true" if passes_precalib_filter else "false",
                "filter_reason": filter_reason,
                "weight": float(candidate.weight),
            }
        precalib_rows.append(audit_row)
        records.append(
            {
                "candidate": candidate,
                "implied_vol": float(iv),
                "percent_strike": float(percent_strike),
                "selected": bool(passes_precalib_filter),
                "audit_row": audit_row,
            }
        )
        stats["used_option_rows"] += 1

    if option_filter_mode == "otm_preferred_itm_fallback":
        _apply_otm_preferred_itm_fallback(records, stats)

    for record in records:
        if not record["selected"]:
            continue
        candidate = record["candidate"]
        implied_vol = float(record["implied_vol"])
        percent_strike = float(record["percent_strike"])
        bucket = grouped[candidate.business_days][candidate.strike]
        bucket["iv_weighted_sum"] += implied_vol * candidate.weight
        bucket["weight_sum"] += candidate.weight
        bucket["price_weighted_sum"] += candidate.price * candidate.weight
        bucket["spot_weighted_sum"] += candidate.spot * candidate.weight
        bucket["percent_strike_weighted_sum"] += percent_strike * candidate.weight
        bucket["iv_observations"].append(
            (implied_vol, float(candidate.weight))
        )

    business_days_list: List[int] = []
    vols: List[List[float]] = []
    percent_strikes: List[List[float]] = []

    for bdays in sorted(grouped.keys()):
        strike_map = grouped[bdays]
        points: List[Dict[str, Any]] = []
        for strike_value, values in strike_map.items():
            weight_sum = float(values["weight_sum"])
            if weight_sum <= 0:
                continue
            if iv_aggregation_mode == "volume_weighted_median":
                avg_iv = _weighted_median(
                    [value for value, _ in values["iv_observations"]],
                    [weight for _, weight in values["iv_observations"]],
                )
            else:
                avg_iv = float(values["iv_weighted_sum"]) / weight_sum
            avg_price = float(values["price_weighted_sum"]) / weight_sum
            avg_spot = float(values["spot_weighted_sum"]) / weight_sum
            avg_pct = float(values["percent_strike_weighted_sum"]) / weight_sum
            if not math.isfinite(avg_iv) or avg_iv <= 0:
                continue
            if not math.isfinite(avg_price) or avg_price <= 0:
                continue
            if not math.isfinite(avg_spot) or avg_spot <= 0:
                continue
            if not math.isfinite(avg_pct) or avg_pct <= 0:
                continue

            points.append(
                {
                    "percent_strike": avg_pct,
                    "implied_vol": avg_iv,
                }
            )

        points.sort(key=lambda x: x["percent_strike"])
        if len(points) < min_strikes_per_expiry:
            continue

        business_days_list.append(int(bdays))
        percent_strikes.append([float(p["percent_strike"]) for p in points])
        vols.append([float(p["implied_vol"]) for p in points])

    all_audit_rows = [
        *(rejected_audit_rows or []),
        *precalib_rows,
    ]
    if precalib_writer is not None and all_audit_rows:
        for row in all_audit_rows:
            precalib_writer.writerow(row)
        stats["precalib_rows_written"] += len(all_audit_rows)
        stats["precalib_minutes_written"] += 1

    if len(business_days_list) < min_expiries_per_minute:
        stats["skip_sample_insufficient"] += 1
        return

    surface_model = _normalize_surface_model(surface_model)

    try:
        params = _to_json_native(
            build_surface_params(
                surface_model=surface_model,
                valuation_date=valuation_date,
                vols=vols,
                percent_strikes=percent_strikes,
                business_days_list=business_days_list,
                vol_daycount=vol_daycount,
                stats=stats,
            )
        )
    except Exception:
        stats["skip_calibration_exception"] += 1
        logger.debug("Surface fit failed at minute %s for model=%s", minute_ts, surface_model, exc_info=True)
        return

    results[minute_key] = {
        "surface_model": surface_model,
        "surface_params": params,
        "surface_audit": {
            "pricing_model": (
                candidates[0].pricing_model if candidates else ""
            ),
            "iv_aggregation_mode": iv_aggregation_mode,
            "option_filter_mode": option_filter_mode,
            "max_itm_moneyness_distance": float(
                max_itm_moneyness_distance
            ),
            "valid_option_observations": len(precalib_rows),
            "selected_option_observations": sum(
                bool(record["selected"]) for record in records
            ),
            "selected_otm_observations": sum(
                bool(record["selected"])
                and bool(record["candidate"].is_otm)
                for record in records
            ),
            "selected_itm_fallback_observations": sum(
                bool(record["selected"])
                and not bool(record["candidate"].is_otm)
                for record in records
            ),
            "rejected_option_observations": len(rejected_audit_rows or []),
            "expiry_slice_count": len(business_days_list),
        },
    }
    stats["calibrated_minutes"] += 1


def _build_rows_for_minute(
    minute_df: pd.DataFrame,
    expiry_inference_date: date,
    expiration_time_utc: dt_time,
    calendar,
    contract_cache: Dict[str, Optional[ContractMeta]],
    stats: Dict[str, int],
    pricing_model: str = DEFAULT_PRICING_MODEL,
) -> List[MinuteTradeRow]:
    rows: List[MinuteTradeRow] = []

    for rec in minute_df.itertuples(index=False):
        ric = str(rec.ric)
        trade_ts: pd.Timestamp = rec.trade_dt
        price = float(rec.price)
        volume_raw = float(rec.volume)
        if not math.isfinite(volume_raw) or volume_raw <= 0:
            stats["skip_nonpositive_volume"] += 1
            continue
        volume = volume_raw

        trade_do = TradeDataDO(
            contract_id=ric,
            trade_time=trade_ts.to_pydatetime(warn=False),
            trade_price=price,
            trade_volume=volume,
        )

        meta = contract_cache.get(ric)
        if meta is None and ric not in contract_cache:
            try:
                meta = _build_contract_meta(
                    trade_do=trade_do,
                    expiry_inference_date=expiry_inference_date,
                    calendar=calendar,
                    expiration_time_utc=expiration_time_utc,
                    pricing_model=pricing_model,
                )
            except Exception:
                stats["skip_contract_parse"] += 1
                logger.debug("Contract parsing failed for ric=%s", ric, exc_info=True)
                contract_cache[ric] = None
                continue
            contract_cache[ric] = meta

        if meta is None:
            continue

        rows.append(
            MinuteTradeRow(
                trade_ts=trade_ts,
                meta=meta,
                price=price,
                volume=volume,
            )
        )

    return rows


def _setup_runtime(
    args: argparse.Namespace,
) -> Tuple[Path, Path, Path, date, Any, DayCountBusN, dt_time, List[str]]:
    output_dir = Path(getattr(args, "output_dir", Path(args.output_json).parent))
    args.output_dir = str(output_dir)
    args.model = _normalize_surface_model(getattr(args, "model", DEFAULT_SURFACE_MODEL))
    args.data_range = _normalize_data_range(getattr(args, "data_range", DEFAULT_DATA_RANGE))
    args.run_ts = str(getattr(args, "run_ts", "")).strip() or _default_run_ts()
    args.pricing_model = _normalize_pricing_model(
        getattr(args, "pricing_model", DEFAULT_PRICING_MODEL)
    )
    args.rate_curve_path = str(
        getattr(args, "rate_curve_path", DEFAULT_RATE_CURVE_PATH)
    )
    args.max_rate_staleness_days = int(
        getattr(
            args,
            "max_rate_staleness_days",
            DEFAULT_MAX_RATE_STALENESS_DAYS,
        )
    )
    args.max_underlying_staleness_seconds = int(
        getattr(
            args,
            "max_underlying_staleness_seconds",
            DEFAULT_MAX_UNDERLYING_STALENESS_SECONDS,
        )
    )
    args.underlying_match_mode = str(
        getattr(args, "underlying_match_mode", "last_prior_trade")
    )
    args.option_filter_mode = _normalize_option_filter_mode(
        getattr(args, "option_filter_mode", DEFAULT_OPTION_FILTER_MODE)
    )
    args.iv_aggregation_mode = _normalize_iv_aggregation_mode(
        getattr(args, "iv_aggregation_mode", DEFAULT_IV_AGGREGATION_MODE)
    )
    args.window_audit_json = str(getattr(args, "window_audit_json", ""))
    output_json_path = Path(args.output_json)
    log_path = Path(args.log_file)
    resolved_config_path = Path(
        getattr(args, "resolved_config_path", output_dir / RESOLVED_CONFIG_FILENAME)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    args.resolved_config_path = str(resolved_config_path)
    resolved_config_path = _write_resolved_config(args)

    setup_logging(
        LoggingConfig(
            level="INFO",
            log_file=log_path,
            file_level="INFO",
            force=True,
        )
    )

    expiry_inference_date = _parse_data_date(args.data_date)
    pricing_model = args.pricing_model
    calendar = (
        cme_treasury_calendar()
        if pricing_model == "black76"
        else usd_calendar()
    )
    vol_daycount = DayCountBusN(
        (
            f"BUS{int(args.days_in_year)}CME_TREASURY"
            if pricing_model == "black76"
            else f"BUS{int(args.days_in_year)}USD"
        ),
        calendar,
        int(args.days_in_year),
    )
    expiration_time_utc = _parse_expiration_time_utc(args.expiration_time_utc)
    if pricing_model == "black76":
        _load_rate_curve(
            str(args.rate_curve_path),
            int(args.max_rate_staleness_days),
        )

    files = sorted(glob.glob(args.input_glob, recursive=True))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise FileNotFoundError(f"No input files found for pattern: {args.input_glob}")

    return (
        output_json_path,
        log_path,
        resolved_config_path,
        expiry_inference_date,
        calendar,
        vol_daycount,
        expiration_time_utc,
        files,
    )


def resolve_parallel_calibration_workers(
    args: argparse.Namespace,
    *,
    device: str,
    data_range: str,
) -> int:
    requested = int(getattr(args, "calibration_workers", 0) or 0)
    if requested <= 0:
        return 0

    normalized_model = _normalize_surface_model(getattr(args, "model", DEFAULT_SURFACE_MODEL))
    normalized_data_range = _normalize_data_range(data_range)
    if (
        device == "cpu"
        and normalized_model in {"svi", "raw"}
        and normalized_data_range in {"window", "excel"}
    ):
        return requested

    logger.warning(
        "Ignoring calibration_workers=%d; only CPU SVI/raw window/excel jobs support parallel calibration.",
        requested,
    )
    return 0


def run_minute_svi_job(args: argparse.Namespace, process_minute_fn: ProcessMinuteFn) -> Dict[str, Dict[str, Any]]:
    (
        output_json_path,
        log_path,
        resolved_config_path,
        expiry_inference_date,
        calendar,
        vol_daycount,
        expiration_time_utc,
        files,
    ) = _setup_runtime(args)

    logger.info("Start minute surface generation")
    _log_cli_arguments(args)
    logger.info("Config file=%s", Path(args.config))
    logger.info("Surface model=%s", str(args.model))
    logger.info("Run timestamp=%s", str(args.run_ts))
    logger.info("Output directory=%s", Path(args.output_dir))
    logger.info("Expiry inference data_date=%s", expiry_inference_date.isoformat())
    logger.info("Input files=%d", len(files))
    logger.info("Output JSON=%s", output_json_path)
    logger.info("Log file=%s", log_path)
    logger.info("Resolved config=%s", resolved_config_path)
    logger.info("Expiration time UTC=%s", expiration_time_utc.isoformat())
    logger.info("Max pre-calib IV=%s", float(args.max_precalib_iv))
    logger.info("Save pre-calib CSV=%s", bool(args.save_precalib_csv))
    logger.info("Pre-calib CSV path=%s", Path(args.precalib_csv))
    parsed_file_ranges = [rng for rng in (_infer_file_date_range(path) for path in files) if rng[0] is not None]
    if parsed_file_ranges:
        logger.info(
            "Input file date span=%s..%s",
            min(start for start, _ in parsed_file_ranges).isoformat(),
            max(end for _, end in parsed_file_ranges if end is not None).isoformat(),
        )

    start_ts = time.time()
    stats: Dict[str, int] = defaultdict(int)
    results: Dict[str, Dict[str, Any]] = {}
    contract_cache: Dict[str, Optional[ContractMeta]] = {}
    last_spot_by_key: Dict[Any, Any] = {}
    pricing_context = _pricing_context_from_args(args)
    precalib_csv_path = Path(args.precalib_csv)
    precalib_writer: Optional[csv.DictWriter] = None
    precalib_fp = None

    if args.save_precalib_csv:
        precalib_csv_path.parent.mkdir(parents=True, exist_ok=True)
        precalib_fp = precalib_csv_path.open("w", encoding="utf-8", newline="")
        precalib_writer = csv.DictWriter(precalib_fp, fieldnames=PRECALIB_CSV_HEADERS)
        precalib_writer.writeheader()

    stop_due_to_max_minutes = False

    try:
        for file_idx, path in enumerate(files, start=1):
            logger.info("Processing file %d/%d: %s", file_idx, len(files), path)
            stats["total_files"] += 1
            file_start_date, file_end_date = _infer_file_date_range(path)
            target_future_month_code = _get_file_target_future_month_code(path)
            logger.info(
                "File context: date_range=%s..%s target_future_month=%s",
                file_start_date.isoformat() if file_start_date is not None else "unknown",
                file_end_date.isoformat() if file_end_date is not None else "unknown",
                target_future_month_code or "unknown",
            )

            try:
                chunk_iter = pd.read_csv(
                    path,
                    usecols=["#RIC", "Date-Time", "Price", "Volume"],
                    chunksize=max(1, int(args.chunk_size)),
                )
            except ValueError:
                logger.warning("Skip unreadable/empty file: %s", path)
                stats["skip_empty_file"] += 1
                continue
            except (gzip.BadGzipFile, EOFError, OSError, pd.errors.ParserError) as exc:
                logger.warning("Skip corrupted file: %s (%s)", path, exc)
                stats["skip_bad_gzip"] += 1
                continue

            pending_minute_df: Optional[pd.DataFrame] = None
            try:
                for chunk in chunk_iter:
                    if chunk.empty:
                        continue

                    chunk = chunk.rename(
                        columns={
                            "#RIC": "ric",
                            "Date-Time": "raw_time",
                            "Price": "price",
                            "Volume": "volume",
                        }
                    )

                    chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
                    chunk["volume"] = pd.to_numeric(chunk["volume"], errors="coerce")
                    chunk["trade_dt"] = pd.to_datetime(chunk["raw_time"], errors="coerce", utc=True)
                    chunk = chunk[
                        chunk["ric"].notna()
                        & chunk["price"].notna()
                        & (chunk["price"] > 0)
                        & chunk["volume"].notna()
                        & (chunk["volume"] > 0)
                        & chunk["trade_dt"].notna()
                    ].copy()
                    if chunk.empty:
                        continue

                    if pending_minute_df is not None and not pending_minute_df.empty:
                        chunk = pd.concat([pending_minute_df, chunk], axis=0, ignore_index=True)
                        pending_minute_df = None

                    chunk["minute"] = chunk["trade_dt"].dt.floor("min")
                    chunk = chunk.sort_values(["minute", "trade_dt"])
                    if chunk.empty:
                        continue

                    last_minute = chunk["minute"].iloc[-1]
                    pending_mask = chunk["minute"] == last_minute
                    pending_minute_df = chunk[pending_mask].copy()
                    ready_df = chunk[~pending_mask]

                    for minute_ts, minute_df in ready_df.groupby("minute", sort=True):
                        if args.max_minutes and args.max_minutes > 0 and stats["total_minutes"] >= args.max_minutes:
                            stop_due_to_max_minutes = True
                            break

                        rows = _build_rows_for_minute(
                            minute_df=minute_df,
                            expiry_inference_date=expiry_inference_date,
                            expiration_time_utc=expiration_time_utc,
                            calendar=calendar,
                            contract_cache=contract_cache,
                            stats=stats,
                            pricing_model=pricing_context.pricing_model,
                        )
                        if not rows:
                            stats["total_minutes"] += 1
                            continue

                        process_minute_fn(
                            minute_ts=minute_ts,
                            rows=rows,
                            days_in_year=int(args.days_in_year),
                            min_strikes_per_expiry=int(args.min_strikes_per_expiry),
                            min_expiries_per_minute=int(args.min_expiries_per_minute),
                            max_precalib_iv=float(args.max_precalib_iv),
                            vol_daycount=vol_daycount,
                            calendar=calendar,
                            target_future_month_code=target_future_month_code,
                            last_spot_by_key=last_spot_by_key,
                            results=results,
                            stats=stats,
                            precalib_writer=precalib_writer,
                            surface_model=str(args.model),
                            pricing_context=pricing_context,
                        )
                        if pricing_context.pricing_model == "black76":
                            _update_last_underlying_trade(rows, last_spot_by_key)

                    if stop_due_to_max_minutes:
                        break
            except (gzip.BadGzipFile, EOFError, OSError, pd.errors.ParserError) as exc:
                logger.warning("Skip corrupted file: %s (%s)", path, exc)
                stats["skip_bad_gzip"] += 1
                pending_minute_df = None
                continue

            if not stop_due_to_max_minutes and pending_minute_df is not None and not pending_minute_df.empty:
                for minute_ts, minute_df in pending_minute_df.groupby("minute", sort=True):
                    if args.max_minutes and args.max_minutes > 0 and stats["total_minutes"] >= args.max_minutes:
                        stop_due_to_max_minutes = True
                        break

                    rows = _build_rows_for_minute(
                        minute_df=minute_df,
                        expiry_inference_date=expiry_inference_date,
                        expiration_time_utc=expiration_time_utc,
                        calendar=calendar,
                        contract_cache=contract_cache,
                        stats=stats,
                        pricing_model=pricing_context.pricing_model,
                    )
                    if not rows:
                        stats["total_minutes"] += 1
                        continue

                    process_minute_fn(
                        minute_ts=minute_ts,
                        rows=rows,
                        days_in_year=int(args.days_in_year),
                        min_strikes_per_expiry=int(args.min_strikes_per_expiry),
                        min_expiries_per_minute=int(args.min_expiries_per_minute),
                        max_precalib_iv=float(args.max_precalib_iv),
                        vol_daycount=vol_daycount,
                        calendar=calendar,
                        target_future_month_code=target_future_month_code,
                        last_spot_by_key=last_spot_by_key,
                        results=results,
                        stats=stats,
                        precalib_writer=precalib_writer,
                        surface_model=str(args.model),
                        pricing_context=pricing_context,
                    )
                    if pricing_context.pricing_model == "black76":
                        _update_last_underlying_trade(rows, last_spot_by_key)

            if stop_due_to_max_minutes:
                logger.info("Stop early due to --max-minutes=%d", args.max_minutes)
                break
    finally:
        if precalib_fp is not None:
            precalib_fp.close()

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_ts
    logger.info("Finished minute surface generation")
    logger.info("Elapsed seconds: %.2f", elapsed)
    logger.info("Summary stats:")
    logger.info("  total_files=%d", stats["total_files"])
    logger.info("  total_minutes=%d", stats["total_minutes"])
    logger.info("  calibrated_minutes=%d", stats["calibrated_minutes"])
    logger.info("  unique_contracts_cached=%d", len(contract_cache))
    logger.info("  option_rows=%d", stats["option_rows"])
    logger.info("  used_option_rows=%d", stats["used_option_rows"])
    logger.info("  future_rows=%d", stats["future_rows"])
    logger.info("  skip_future_non_target_month=%d", stats["skip_future_non_target_month"])
    logger.info("  skip_no_spot=%d", stats["skip_no_spot"])
    logger.info("  skip_tau_nonpositive=%d", stats["skip_tau_nonpositive"])
    logger.info("  skip_iv_fail=%d", stats["skip_iv_fail"])
    logger.info("  skip_precalib_iv_above_cap=%d", stats["skip_precalib_iv_above_cap"])
    logger.info("  skip_sample_insufficient=%d", stats["skip_sample_insufficient"])
    logger.info("  skip_calibration_exception=%d", stats["skip_calibration_exception"])
    logger.info("  qls_boundary_retry_slices=%d", stats["qls_boundary_retry_slices"])
    logger.info("  qls_fallback_attempt_slices=%d", stats["qls_fallback_attempt_slices"])
    logger.info("  qls_fallback_success_slices=%d", stats["qls_fallback_success_slices"])
    logger.info("  qls_boundary_reject_slices=%d", stats["qls_boundary_reject_slices"])
    logger.info("  qls_stage1_kept_slices=%d", stats["qls_stage1_kept_slices"])
    logger.info("  skip_contract_parse=%d", stats["skip_contract_parse"])
    logger.info("  skip_empty_file=%d", stats["skip_empty_file"])
    logger.info("  skip_bad_gzip=%d", stats["skip_bad_gzip"])
    logger.info("  precalib_minutes_written=%d", stats["precalib_minutes_written"])
    logger.info("  precalib_rows_written=%d", stats["precalib_rows_written"])
    logger.info("Saved %d calibrated minute surfaces to %s", len(results), output_json_path)
    return results
