"""Shared minute-SVI generation helpers for CPU and GPU entrypoints."""

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
import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.generate_surface.common.config_utils import (  # noqa: E402
    build_config_scope,
    resolve_config_variables,
)
from logger import LoggingConfig, setup_logging  # noqa: E402
from market_data.contract_handler.future_contract import FutureContract  # noqa: E402
from market_data.contract_handler.option_contract import OptionContract  # noqa: E402
from market_data.contract_handler.utils import ContractTerminationRule  # noqa: E402
from market_data.dto.tradedata_do import TradeDataDO  # noqa: E402
from quantlib.calendar.daycount import DayCountBusN  # noqa: E402
from quantlib.calendar.holidays import usd_calendar  # noqa: E402
from quantlib.vol_surface.algo.svi_algo import SviCalibrationQuasiExplicit  # noqa: E402

logger = logging.getLogger(__name__)

PRECALIB_CSV_HEADERS = [
    "snapshot_time_utc",
    "business_days",
    "maturity_date",
    "option_type",
    "strike",
    "price",
    "spot",
    "percent_strike",
    "implied_vol",
    "weight_sum",
]

DEFAULT_CONFIG_PATH = "configs/surface_builder/default.yaml"
SUPPORTED_MINUTE_SVI_CONFIG_KEYS = {
    "input_glob",
    "output_dir",
    "output_json",
    "log_file",
    "data_date",
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


ProcessMinuteFn = Callable[
    [
        pd.Timestamp,
        List[MinuteTradeRow],
        int,
        int,
        int,
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


def _resolve_config_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return ROOT_DIR / path


def _parse_bool_from_config(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Invalid boolean value for `{key}` in config: {value!r}")


def _load_minute_svi_config(config_path_value: str) -> Dict[str, Any]:
    config_path = _resolve_config_path(config_path_value)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}
    if not isinstance(raw_data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    config_root = raw_data.get("surface_builder", raw_data)
    if not isinstance(config_root, dict):
        raise ValueError(f"`surface_builder` must be a mapping in config file: {config_path}")

    minute_section = config_root.get("minute_svi")
    if minute_section is None:
        minute_section = {k: v for k, v in config_root.items() if k in SUPPORTED_MINUTE_SVI_CONFIG_KEYS}
    if not isinstance(minute_section, dict):
        raise ValueError(f"`minute_svi` must be a mapping in config file: {config_path}")

    defaults = dict(minute_section)
    shared_glob = config_root.get("option_data_glob")
    if "input_glob" not in defaults and isinstance(shared_glob, str):
        defaults["input_glob"] = shared_glob
    if "output_dir" in defaults and isinstance(defaults["output_dir"], str):
        defaults.setdefault("output_json", "${output_dir}/minute_svi_params.json")
        defaults.setdefault("log_file", "${output_dir}/minute_svi_params.log")
        defaults.setdefault("precalib_csv", "${output_dir}/minute_svi_precalib_points.csv")

    unknown_keys = sorted(set(defaults.keys()) - SUPPORTED_MINUTE_SVI_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown minute_svi config keys in {config_path}: {unknown_keys}")
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
    pre_args, _ = pre_parser.parse_known_args(argv_list)

    if "-h" in argv_list or "--help" in argv_list:
        config_defaults: Dict[str, Any] = {}
    else:
        config_defaults = _load_minute_svi_config(pre_args.config)

    default_save_precalib_csv = _parse_bool_from_config(
        config_defaults.get("save_precalib_csv", False),
        key="save_precalib_csv",
    )

    parser = argparse.ArgumentParser(
        description="Generate minute-level SVI parameters from option trade files."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=pre_args.config,
        help=(
            "YAML config path. Supports either top-level `minute_svi` mapping, "
            "or `surface_builder.minute_svi` in a consolidated config."
        ),
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default=str(config_defaults.get("input_glob", "data/raw/option_data/**/*.csv.gz")),
        help="Glob pattern for raw option trade files.",
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
    return args


def _to_utc_minute_string(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _log_cli_arguments(args: argparse.Namespace) -> None:
    logger.info("CLI argv: %s", " ".join(shlex.quote(x) for x in sys.argv))
    logger.info("Parsed CLI arguments:")
    for key in sorted(vars(args)):
        logger.info("  %s=%r", key, getattr(args, key))


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


def _make_expiry_dt_utc(expiry_date: date) -> datetime:
    return datetime.combine(expiry_date, dt_time(23, 59, 59, tzinfo=timezone.utc))


def _make_spot_cache_key(underlying: str, target_future_month_code: Optional[str]) -> Tuple[str, str]:
    return underlying, (target_future_month_code or "").upper()


def _build_contract_meta(trade_do: TradeDataDO, expiry_inference_date: date, calendar) -> Optional[ContractMeta]:
    contract = trade_do.to_contract()
    if isinstance(contract, FutureContract):
        return ContractMeta(
            contract_type="future",
            underlying=contract.get_underlying(),
            maturity_month_code=contract.get_maturity_month_code(),
        )

    if isinstance(contract, OptionContract):
        expiry = contract.get_contract_maturity_dates_by_contract_id(
            data_date=expiry_inference_date,
            calendars=[calendar],
            termination_rule=ContractTerminationRule.EndOfMonth,
        )
        if isinstance(expiry, datetime):
            expiry = expiry.date()
        return ContractMeta(
            contract_type="option",
            underlying=contract.get_underlying(),
            strike=float(contract.get_strike()),
            option_type=contract.get_option_type(),
            expiry_date=expiry,
            expiry_dt_utc=_make_expiry_dt_utc(expiry),
        )

    return None


def _collect_minute_spot(
    rows: List[MinuteTradeRow],
    target_future_month_code: Optional[str],
    last_spot_by_key: Dict[Tuple[str, str], float],
    stats: Dict[str, int],
) -> Tuple[Dict[Tuple[str, str], float], List[MinuteTradeRow]]:
    fut_sum: Dict[Tuple[str, str], float] = defaultdict(float)
    fut_vol: Dict[Tuple[str, str], float] = defaultdict(float)
    option_rows: List[MinuteTradeRow] = []
    normalized_target_month = (target_future_month_code or "").upper()

    for row in rows:
        meta = row.meta
        if meta.contract_type == "future":
            stats["future_rows"] += 1
            if normalized_target_month and meta.maturity_month_code != normalized_target_month:
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
        last_spot_by_key[spot_key] = spot
    return minute_spot, option_rows


def _tau_years_from_trade_to_expiry(trade_ts: pd.Timestamp, expiry_dt_utc: datetime) -> float:
    return (pd.Timestamp(expiry_dt_utc) - _to_utc_timestamp(trade_ts)).total_seconds() / SECONDS_PER_DAY / 365.0


def _prepare_option_candidates(
    minute_ts: pd.Timestamp,
    option_rows: List[MinuteTradeRow],
    minute_spot: Dict[Tuple[str, str], float],
    last_spot_by_key: Dict[Tuple[str, str], float],
    target_future_month_code: Optional[str],
    calendar,
    stats: Dict[str, int],
) -> Tuple[date, List[MinuteOptionCandidate]]:
    normalized_target_month = (target_future_month_code or "").upper()
    if minute_ts.tzinfo is None:
        valuation_date = minute_ts.date()
    else:
        valuation_date = minute_ts.tz_convert("UTC").date()

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

        trade_ts_utc = _to_utc_timestamp(row.trade_ts)
        business_days = int(calendar.count_business_days(trade_ts_utc.date(), expiry_date, False, True))
        if business_days <= 0:
            stats["skip_tau_nonpositive"] += 1
            continue

        tau = _tau_years_from_trade_to_expiry(row.trade_ts, expiry_dt_utc)
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
    vol_daycount: DayCountBusN,
    results: Dict[str, Dict[str, Any]],
    stats: Dict[str, int],
    precalib_writer: Optional[csv.DictWriter] = None,
) -> None:
    grouped: Dict[int, Dict[float, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "iv_weighted_sum": 0.0,
                "weight_sum": 0.0,
                "price_weighted_sum": 0.0,
                "spot_weighted_sum": 0.0,
                "percent_strike_weighted_sum": 0.0,
                "option_weight_map": defaultdict(float),
                "maturity_weight_map": defaultdict(float),
            }
        )
    )

    for candidate, iv in zip(candidates, implied_vols):
        if iv is None or not math.isfinite(iv) or iv <= 0:
            stats["skip_iv_fail"] += 1
            continue

        percent_strike = candidate.strike / candidate.spot
        if not math.isfinite(percent_strike) or percent_strike <= 0:
            stats["skip_iv_fail"] += 1
            continue

        bucket = grouped[candidate.business_days][candidate.strike]
        bucket["iv_weighted_sum"] += iv * candidate.weight
        bucket["weight_sum"] += candidate.weight
        bucket["price_weighted_sum"] += candidate.price * candidate.weight
        bucket["spot_weighted_sum"] += candidate.spot * candidate.weight
        bucket["percent_strike_weighted_sum"] += percent_strike * candidate.weight
        bucket["option_weight_map"][_option_type_name(candidate.meta.option_type)] += candidate.weight
        if candidate.meta.expiry_date is not None:
            bucket["maturity_weight_map"][candidate.meta.expiry_date.isoformat()] += candidate.weight
        stats["used_option_rows"] += 1

    business_days_list: List[int] = []
    vols: List[List[float]] = []
    percent_strikes: List[List[float]] = []
    precalib_rows: List[Dict[str, Any]] = []

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

            maturity_weight_map = values["maturity_weight_map"]
            maturity_date = ""
            if maturity_weight_map:
                maturity_date = max(maturity_weight_map.items(), key=lambda x: x[1])[0]

            option_weight_map = values["option_weight_map"]
            option_type = ""
            if option_weight_map:
                option_type = max(option_weight_map.items(), key=lambda x: x[1])[0]

            points.append(
                {
                    "percent_strike": avg_pct,
                    "implied_vol": avg_iv,
                    "strike": float(strike_value),
                    "price": avg_price,
                    "spot": avg_spot,
                    "weight_sum": weight_sum,
                    "maturity_date": maturity_date,
                    "option_type": option_type,
                }
            )

        points.sort(key=lambda x: x["percent_strike"])
        if len(points) < min_strikes_per_expiry:
            continue

        business_days_list.append(int(bdays))
        percent_strikes.append([float(p["percent_strike"]) for p in points])
        vols.append([float(p["implied_vol"]) for p in points])
        for point in points:
            precalib_rows.append(
                {
                    "business_days": int(bdays),
                    "maturity_date": point["maturity_date"],
                    "option_type": point["option_type"],
                    "strike": point["strike"],
                    "price": point["price"],
                    "spot": point["spot"],
                    "percent_strike": point["percent_strike"],
                    "implied_vol": point["implied_vol"],
                    "weight_sum": point["weight_sum"],
                }
            )

    if len(business_days_list) < min_expiries_per_minute:
        stats["skip_sample_insufficient"] += 1
        return

    minute_key = _to_utc_minute_string(minute_ts)
    if precalib_writer is not None and precalib_rows:
        for row in precalib_rows:
            precalib_writer.writerow(
                {
                    "snapshot_time_utc": minute_key,
                    "business_days": row["business_days"],
                    "maturity_date": row["maturity_date"],
                    "option_type": row["option_type"],
                    "strike": row["strike"],
                    "price": row["price"],
                    "spot": row["spot"],
                    "percent_strike": row["percent_strike"],
                    "implied_vol": row["implied_vol"],
                    "weight_sum": row["weight_sum"],
                }
            )
        stats["precalib_rows_written"] += len(precalib_rows)
        stats["precalib_minutes_written"] += 1

    try:
        calibration = SviCalibrationQuasiExplicit(
            vols=vols,
            percent_strikes=percent_strikes,
            business_days=business_days_list,
            vol_daycount=vol_daycount,
            valuation_date=valuation_date,
        )
        params = _to_json_native(calibration.params)
    except Exception:
        stats["skip_calibration_exception"] += 1
        logger.debug("Calibration failed at minute %s", minute_ts, exc_info=True)
        return

    results[minute_key] = params
    stats["calibrated_minutes"] += 1


def _build_rows_for_minute(
    minute_df: pd.DataFrame,
    expiry_inference_date: date,
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


def _setup_runtime(args: argparse.Namespace) -> Tuple[Path, Path, date, Any, DayCountBusN, List[str]]:
    output_json_path = Path(args.output_json)
    log_path = Path(args.log_file)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

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
        name=f"BUS{int(args.days_in_year)}USD",
        calendar=calendar,
        days_in_year=int(args.days_in_year),
    )

    files = sorted(glob.glob(args.input_glob, recursive=True))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise FileNotFoundError(f"No input files found for pattern: {args.input_glob}")

    return output_json_path, log_path, expiry_inference_date, calendar, vol_daycount, files


def run_minute_svi_job(args: argparse.Namespace, process_minute_fn: ProcessMinuteFn) -> Dict[str, Dict[str, Any]]:
    output_json_path, log_path, expiry_inference_date, calendar, vol_daycount, files = _setup_runtime(args)

    logger.info("Start minute SVI generation")
    _log_cli_arguments(args)
    logger.info("Config file=%s", Path(args.config))
    logger.info("Expiry inference data_date=%s", expiry_inference_date.isoformat())
    logger.info("Input files=%d", len(files))
    logger.info("Output JSON=%s", output_json_path)
    logger.info("Log file=%s", log_path)
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
                            vol_daycount=vol_daycount,
                            calendar=calendar,
                            target_future_month_code=target_future_month_code,
                            last_spot_by_key=last_spot_by_key,
                            results=results,
                            stats=stats,
                            precalib_writer=precalib_writer,
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
                        vol_daycount=vol_daycount,
                        calendar=calendar,
                        target_future_month_code=target_future_month_code,
                        last_spot_by_key=last_spot_by_key,
                        results=results,
                        stats=stats,
                        precalib_writer=precalib_writer,
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
    logger.info("Finished minute SVI generation")
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
    logger.info("  skip_sample_insufficient=%d", stats["skip_sample_insufficient"])
    logger.info("  skip_calibration_exception=%d", stats["skip_calibration_exception"])
    logger.info("  skip_contract_parse=%d", stats["skip_contract_parse"])
    logger.info("  skip_empty_file=%d", stats["skip_empty_file"])
    logger.info("  skip_bad_gzip=%d", stats["skip_bad_gzip"])
    logger.info("  precalib_minutes_written=%d", stats["precalib_minutes_written"])
    logger.info("  precalib_rows_written=%d", stats["precalib_rows_written"])
    logger.info("Saved %d calibrated minute surfaces to %s", len(results), output_json_path)
    return results
