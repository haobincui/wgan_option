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
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

import scripts._path_setup  # noqa: F401

from scripts.generate_surface.common.config_utils import (  # noqa: E402
    build_config_scope,
    load_surface_builder_section,
    parse_bool_from_config,
    resolve_config_path,
    resolve_config_variables,
    write_yaml_mapping,
)
from scripts.generate_surface.model import build_surface_params  # noqa: E402
from logger import LoggingConfig, setup_logging  # noqa: E402
from market_data.contract_handler.future_contract import FutureContract  # noqa: E402
from market_data.contract_handler.option_contract import OptionContract  # noqa: E402
from market_data.contract_handler.utils import ContractTerminationRule  # noqa: E402
from market_data.dto.tradedata_do import TradeDataDO  # noqa: E402
from quantlib.calendar.daycount import DayCountBusN  # noqa: E402
from quantlib.calendar.holidays import usd_calendar  # noqa: E402

logger = logging.getLogger(__name__)

PRECALIB_CSV_HEADERS = [
    "trade_datetime_utc",
    "calibration_datetime_utc",
    "business_days",
    "maturity_date",
    "contract_id",
    "option_type",
    "strike",
    "price",
    "spot",
    "percent_strike",
    "implied_vol",
    "passes_precalib_filter",
    "filter_reason",
    "weight",
]

DEFAULT_CONFIG_PATH = "configs/surface_builder/svi/minute-svi-all.yaml"
DEFAULT_EXPIRATION_TIME_UTC = "20:00:00"
DEFAULT_MAX_PRECALIB_IV = 3.0
DEFAULT_SURFACE_MODEL = "svi"
SUPPORTED_SURFACE_MODELS = {"svi", "sabr", "cubic", "raw"}
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
SUPPORTED_MINUTE_SVI_CONFIG_KEYS = {
    "input_glob",
    "output_dir",
    "output_json",
    "log_file",
    "model",
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
    "save_precalib_csv",
    "precalib_csv",
}

FILE_DATE_RANGE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv(?:\.gz)?$")
SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class ContractMeta:
    contract_type: str
    underlying: str
    maturity_month_code: Optional[str] = None
    strike: Optional[float] = None
    option_type: Any = None
    expiry_date: Optional[date] = None
    expiry_dt_utc: Optional[datetime] = None
    contract_id: Optional[str] = None


@dataclass(frozen=True)
class MinuteTradeRow:
    trade_ts: pd.Timestamp
    meta: ContractMeta
    price: float
    volume: float


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


ProcessMinuteFn = Callable[
    [
        pd.Timestamp,
        List[MinuteTradeRow],
        int,
        int,
        int,
        float,
        DayCountBusN,
        Any,
        Optional[str],
        Dict[Tuple[str, str], float],
        Dict[str, Dict[str, Any]],
        Dict[str, int],
        Optional[csv.DictWriter],
    ],
    None,
]


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


def _load_minute_svi_config(
    config_path_value: str,
    *,
    runtime_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config_path, config_root, defaults = load_surface_builder_section(
        config_path_value,
        section_key="minute_svi",
        supported_keys=SUPPORTED_MINUTE_SVI_CONFIG_KEYS,
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
    defaults["run_ts"] = str(defaults.get("run_ts", "")).strip() or _default_run_ts()
    defaults.setdefault("output_dir", "data/processed/${model}/${run_ts}")
    defaults.setdefault("output_json", "${output_dir}/minute_svi_params.json")
    defaults.setdefault("log_file", "${output_dir}/minute_svi_params.log")
    defaults.setdefault("precalib_csv", "${output_dir}/minute_svi_precalib_points.csv")

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
        "--run-ts",
        type=str,
        default=None,
    )
    pre_args, _ = pre_parser.parse_known_args(argv_list)

    if "-h" in argv_list or "--help" in argv_list:
        config_defaults: Dict[str, Any] = {}
    else:
        config_defaults = _load_minute_svi_config(
            pre_args.config,
            runtime_overrides={
                "model": pre_args.model,
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
        help="YAML config path with a required `surface_builder.minute_svi` mapping.",
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
        "--run-ts",
        type=str,
        default=str(config_defaults.get("run_ts", _default_run_ts())),
        help="Run timestamp used inside the default output directory.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(config_defaults.get("output_json", "data/processed/minute_svi_params.json")),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(config_defaults.get("log_file", "data/processed/minute_svi_params.log")),
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
        default=str(config_defaults.get("precalib_csv", "data/processed/minute_svi_precalib_points.csv")),
        help="Path for pre-calibration SVI input points CSV.",
    )
    args = parser.parse_args(argv_list)
    args.config = str(_resolve_config_path(args.config))
    args.model = _normalize_surface_model(args.model)
    args.run_ts = str(args.run_ts).strip() or _default_run_ts()

    resolved_defaults = _load_minute_svi_config(
        args.config,
        runtime_overrides={
            "model": args.model,
            "run_ts": args.run_ts,
        },
    )
    args.output_dir = str(resolved_defaults.get("output_dir", Path(args.output_json).parent))
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


def _log_cli_arguments(args: argparse.Namespace) -> None:
    logger.info("CLI argv: %s", " ".join(shlex.quote(x) for x in sys.argv))
    logger.info("Parsed CLI arguments:")
    for key in sorted(vars(args)):
        logger.info("  %s=%r", key, getattr(args, key))


def _build_resolved_config_payload(args: argparse.Namespace) -> Dict[str, Any]:
    model = _normalize_surface_model(getattr(args, "model", DEFAULT_SURFACE_MODEL))
    run_ts = str(getattr(args, "run_ts", "")).strip() or _default_run_ts()
    output_dir = str(getattr(args, "output_dir", Path(getattr(args, "output_json")).parent))
    payload: Dict[str, Any] = {
        "surface_builder": {
            "minute_svi": {
                "model": model,
                "run_ts": run_ts,
                "input_glob": str(args.input_glob),
                "output_dir": output_dir,
                "output_json": str(args.output_json),
                "log_file": str(args.log_file),
                "data_date": str(args.data_date),
                "expiration_time_utc": str(args.expiration_time_utc),
                "days_in_year": int(args.days_in_year),
                "min_strikes_per_expiry": int(args.min_strikes_per_expiry),
                "min_expiries_per_minute": int(args.min_expiries_per_minute),
                "max_precalib_iv": float(args.max_precalib_iv),
                "max_files": int(args.max_files),
                "max_minutes": int(args.max_minutes),
                "chunk_size": int(args.chunk_size),
                "save_precalib_csv": bool(args.save_precalib_csv),
                "precalib_csv": str(args.precalib_csv),
            }
        },
        "runtime": {
            "config_path": str(args.config),
            "cli_argv": list(sys.argv),
        },
    }
    if hasattr(args, "target_datetimes"):
        payload["surface_builder"]["minute_svi_window"] = {
            "target_datetimes": list(args.target_datetimes),
            "target_datetimes_file": str(getattr(args, "target_datetimes_file", "")),
            "window_minutes": int(getattr(args, "window_minutes", 3)),
        }
    if hasattr(args, "target_xlsx"):
        payload["surface_builder"]["minute_svi_excel"] = {
            "target_xlsx": str(args.target_xlsx),
            "sheet_name": str(args.sheet_name),
            "date_column": str(args.date_column),
            "time_column": str(args.time_column),
            "source_timezone": str(args.source_timezone),
            "max_target_datetimes": int(args.max_target_datetimes),
            "window_minutes": int(getattr(args, "window_minutes", 3)),
        }
    return payload


def _write_resolved_config(args: argparse.Namespace) -> Path:
    return write_yaml_mapping(
        _build_resolved_config_payload(args),
        Path(getattr(args, "output_dir", Path(args.output_json).parent)) / RESOLVED_CONFIG_FILENAME,
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


def _make_spot_cache_key(underlying: str, target_future_month_code: Optional[str]) -> Tuple[str, str]:
    return underlying, (target_future_month_code or "").upper()


def _build_contract_meta(
    trade_do: TradeDataDO,
    expiry_inference_date: date,
    calendar,
    expiration_time_utc: dt_time,
) -> Optional[ContractMeta]:
    contract = trade_do.to_contract()
    if isinstance(contract, FutureContract):
        return ContractMeta(
            contract_type="future",
            underlying=contract.get_underlying(),
            maturity_month_code=contract.get_maturity_month_code(),
            contract_id=trade_do.contract_id,
        )

    if isinstance(contract, OptionContract):
        expiry = contract.get_contract_maturity_dates_by_contract_id(
            data_date=expiry_inference_date,
            calendars=[calendar],
            termination_rule=ContractTerminationRule.EndOfMonth,
            expiration_time=expiration_time_utc,
        )
        expiry_date, expiry_dt_utc = _coerce_expiry_dt_utc(expiry, expiration_time_utc)
        return ContractMeta(
            contract_type="option",
            underlying=contract.get_underlying(),
            strike=float(contract.get_strike()),
            option_type=contract.get_option_type(),
            expiry_date=expiry_date,
            expiry_dt_utc=expiry_dt_utc,
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
            if normalized_target_month and meta.maturity_month_code != normalized_target_month:
                if stats is not None:
                    stats["skip_future_non_target_month"] += 1
                continue
            spot_key = _make_spot_cache_key(
                underlying=meta.underlying,
                target_future_month_code=normalized_target_month or meta.maturity_month_code,
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
    minute_spot: Dict[Tuple[str, str], float],
    last_spot_by_key: Dict[Tuple[str, str], float],
    target_future_month_code: Optional[str],
    vol_daycount: DayCountBusN,
    calendar,
    stats: Dict[str, int],
    tau_anchor_ts: Optional[pd.Timestamp] = None,
) -> Tuple[date, List[MinuteOptionCandidate]]:
    normalized_target_month = (target_future_month_code or "").upper()
    minute_ts_utc = _to_utc_timestamp(minute_ts)
    valuation_date = minute_ts_utc.date()

    candidates: List[MinuteOptionCandidate] = []
    for row in option_rows:
        meta = row.meta
        assert meta.contract_type == "option"

        spot_key = _make_spot_cache_key(meta.underlying, normalized_target_month)
        spot = minute_spot.get(spot_key, last_spot_by_key.get(spot_key))
        if spot is None or not math.isfinite(spot) or spot <= 0:
            stats["skip_no_spot"] += 1
            continue

        expiry_date = meta.expiry_date
        expiry_dt_utc = meta.expiry_dt_utc
        if expiry_date is None or expiry_dt_utc is None:
            stats["skip_tau_nonpositive"] += 1
            continue

        effective_trade_ts = tau_anchor_ts if tau_anchor_ts is not None else row.trade_ts
        trade_ts_utc = _to_utc_timestamp(effective_trade_ts)
        business_days = int(
            calendar.count_business_days(
                trade_ts_utc.date(),
                expiry_date,
                include_start=False,
                include_end=True,
            )
        )
        if business_days <= 0:
            stats["skip_tau_nonpositive"] += 1
            continue

        tau = _tau_years_from_trade_to_expiry(trade_ts_utc, expiry_dt_utc, vol_daycount)
        if tau <= 0:
            stats["skip_tau_nonpositive"] += 1
            continue

        strike = float(meta.strike or 0.0)
        if not math.isfinite(strike) or strike <= 0:
            stats["skip_iv_fail"] += 1
            continue

        candidates.append(
            MinuteOptionCandidate(
                meta=meta,
                price=float(row.price),
                weight=max(float(row.volume), 1.0),
                strike=strike,
                spot=float(spot),
                tau=float(tau),
                business_days=business_days,
                trade_ts=_to_utc_timestamp(row.trade_ts),
            )
        )

    return valuation_date, candidates


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
) -> None:
    grouped: Dict[int, Dict[float, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "iv_weighted_sum": 0.0,
                "weight_sum": 0.0,
                "price_weighted_sum": 0.0,
                "spot_weighted_sum": 0.0,
                "percent_strike_weighted_sum": 0.0,
            }
        )
    )
    minute_key = _to_utc_minute_string(minute_ts)
    precalib_rows: List[Dict[str, Any]] = []

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
        precalib_rows.append(
            {
                "trade_datetime_utc": _to_utc_datetime_string(trade_ts),
                "calibration_datetime_utc": minute_key,
                "business_days": int(candidate.business_days),
                "maturity_date": (
                    candidate.meta.expiry_date.isoformat()
                    if candidate.meta.expiry_date is not None
                    else ""
                ),
                "contract_id": candidate.meta.contract_id or "",
                "option_type": _option_type_name(candidate.meta.option_type),
                "strike": float(candidate.strike),
                "price": float(candidate.price),
                "spot": float(candidate.spot),
                "percent_strike": float(percent_strike),
                "implied_vol": float(iv),
                "passes_precalib_filter": "true" if passes_precalib_filter else "false",
                "filter_reason": filter_reason,
                "weight": float(candidate.weight),
            }
        )
        bucket = grouped[candidate.business_days][candidate.strike]
        stats["used_option_rows"] += 1
        if not passes_precalib_filter:
            continue

        bucket["iv_weighted_sum"] += float(iv) * candidate.weight
        bucket["weight_sum"] += candidate.weight
        bucket["price_weighted_sum"] += candidate.price * candidate.weight
        bucket["spot_weighted_sum"] += candidate.spot * candidate.weight
        bucket["percent_strike_weighted_sum"] += percent_strike * candidate.weight

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

    if precalib_writer is not None and precalib_rows:
        for row in precalib_rows:
            precalib_writer.writerow(row)
        stats["precalib_rows_written"] += len(precalib_rows)
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
    }
    stats["calibrated_minutes"] += 1


def _build_rows_for_minute(
    minute_df: pd.DataFrame,
    expiry_inference_date: date,
    expiration_time_utc: dt_time,
    calendar,
    contract_cache: Dict[str, Optional[ContractMeta]],
    stats: Dict[str, int],
) -> List[MinuteTradeRow]:
    rows: List[MinuteTradeRow] = []

    for rec in minute_df.itertuples(index=False):
        ric = str(rec.ric)
        trade_ts: pd.Timestamp = rec.trade_dt
        price = float(rec.price)
        volume_raw = float(rec.volume)
        volume = max(volume_raw, 1.0) if math.isfinite(volume_raw) else 1.0

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
    args.run_ts = str(getattr(args, "run_ts", "")).strip() or _default_run_ts()
    output_json_path = Path(args.output_json)
    log_path = Path(args.log_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    calendar = usd_calendar()
    vol_daycount = DayCountBusN(
        f"BUS{int(args.days_in_year)}USD",
        calendar,
        int(args.days_in_year),
    )
    expiration_time_utc = _parse_expiration_time_utc(args.expiration_time_utc)

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
    last_spot_by_key: Dict[Tuple[str, str], float] = {}
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
                    chunk["volume"] = pd.to_numeric(chunk["volume"], errors="coerce").fillna(1.0)
                    chunk["trade_dt"] = pd.to_datetime(chunk["raw_time"], errors="coerce", utc=True)
                    chunk = chunk[
                        chunk["ric"].notna()
                        & chunk["price"].notna()
                        & (chunk["price"] > 0)
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
                        )

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
                    )

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
