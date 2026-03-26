"""Shared Excel-driven minute-SVI helpers."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

if __package__ in {None, ""}:
    ROOT_DIR = Path(__file__).resolve().parents[3]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from scripts.generate_surface.common.config_utils import (  # noqa: E402
        build_config_scope,
        resolve_config_variables,
    )
    from scripts.generate_surface.common.minute_svi_common import (  # noqa: E402
        DEFAULT_CONFIG_PATH,
        _parse_args as _parse_base_args,
        _resolve_config_path,
    )
    from scripts.generate_surface.common.minute_svi_window_common import (  # noqa: E402
        ProcessMinuteFn,
        generate_surfaces_for_datetime_windows,
    )
else:
    from .config_utils import build_config_scope, resolve_config_variables  # noqa: E402
    from .minute_svi_common import (  # noqa: E402
        DEFAULT_CONFIG_PATH,
        _parse_args as _parse_base_args,
        _resolve_config_path,
    )
    from .minute_svi_window_common import (  # noqa: E402
        ProcessMinuteFn,
        generate_surfaces_for_datetime_windows,
    )

logger = logging.getLogger(__name__)

DEFAULT_TARGET_XLSX = "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
DEFAULT_SHEET_NAME = "Sheet1"
DEFAULT_DATE_COLUMN = "PD"
DEFAULT_TIME_COLUMN = "ET"
DEFAULT_SOURCE_TIMEZONE = "America/New_York"
SUPPORTED_MINUTE_SVI_EXCEL_CONFIG_KEYS = {
    "target_xlsx",
    "sheet_name",
    "date_column",
    "time_column",
    "source_timezone",
    "max_target_datetimes",
    "window_minutes",
}


def _load_excel_config(config_path_value: str) -> Dict[str, Any]:
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

    excel_section = config_root.get("minute_svi_excel")
    if excel_section is None:
        excel_section = {k: v for k, v in config_root.items() if k in SUPPORTED_MINUTE_SVI_EXCEL_CONFIG_KEYS}
    if not isinstance(excel_section, dict):
        raise ValueError(f"`minute_svi_excel` must be a mapping in config file: {config_path}")

    unknown_keys = sorted(set(excel_section.keys()) - SUPPORTED_MINUTE_SVI_EXCEL_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown minute_svi_excel config keys in {config_path}: {unknown_keys}")
    return resolve_config_variables(dict(excel_section), extra_scope=build_config_scope(config_root))


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    pre_args, _ = pre_parser.parse_known_args(argv_list)

    if "-h" in argv_list or "--help" in argv_list:
        config_defaults: Dict[str, Any] = {}
        try:
            _parse_base_args(["--help"])
        except SystemExit:
            print("")
            print("Excel target-datetime arguments (added by this script):")
            print(f"  --target-xlsx PATH (default: {DEFAULT_TARGET_XLSX})")
            print(f"  --sheet-name NAME (default: {DEFAULT_SHEET_NAME})")
            print(f"  --date-column NAME (default: {DEFAULT_DATE_COLUMN})")
            print(f"  --time-column NAME (default: {DEFAULT_TIME_COLUMN})")
            print(f"  --source-timezone TZ (default: {DEFAULT_SOURCE_TIMEZONE})")
            print("  --max-target-datetimes N (default: 0 means no cap)")
            print("  --window-minutes N (default: 3)")
            raise
    else:
        config_defaults = _load_excel_config(pre_args.config)

    excel_parser = argparse.ArgumentParser(add_help=False)
    excel_parser.add_argument(
        "--target-xlsx",
        type=str,
        default=str(config_defaults.get("target_xlsx", DEFAULT_TARGET_XLSX)),
        help="Path to the Excel file that contains PD and ET columns.",
    )
    excel_parser.add_argument(
        "--sheet-name",
        type=str,
        default=str(config_defaults.get("sheet_name", DEFAULT_SHEET_NAME)),
        help="Worksheet name to load from target xlsx.",
    )
    excel_parser.add_argument(
        "--date-column",
        type=str,
        default=str(config_defaults.get("date_column", DEFAULT_DATE_COLUMN)),
        help="Date column name in xlsx (for example: PD).",
    )
    excel_parser.add_argument(
        "--time-column",
        type=str,
        default=str(config_defaults.get("time_column", DEFAULT_TIME_COLUMN)),
        help="Time column name in xlsx (for example: ET).",
    )
    excel_parser.add_argument(
        "--source-timezone",
        type=str,
        default=str(config_defaults.get("source_timezone", DEFAULT_SOURCE_TIMEZONE)),
        help="Timezone for PD+ET source timestamps before converting to UTC.",
    )
    excel_parser.add_argument(
        "--max-target-datetimes",
        type=int,
        default=int(config_defaults.get("max_target_datetimes", 0)),
        help="Optional cap on target datetimes after dedupe (0 means no cap).",
    )
    excel_parser.add_argument(
        "--window-minutes",
        type=int,
        default=int(config_defaults.get("window_minutes", 3)),
        help="Calibrate minute surfaces within +/- this many minutes around each target datetime.",
    )

    excel_args, remaining_argv = excel_parser.parse_known_args(argv_list)
    args = _parse_base_args(remaining_argv)
    args.target_xlsx = str(excel_args.target_xlsx)
    args.sheet_name = str(excel_args.sheet_name)
    args.date_column = str(excel_args.date_column)
    args.time_column = str(excel_args.time_column)
    args.source_timezone = str(excel_args.source_timezone)
    args.max_target_datetimes = int(excel_args.max_target_datetimes)
    args.window_minutes = int(excel_args.window_minutes)
    return args


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


def _to_utc_string(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_target_datetimes_from_excel(args: argparse.Namespace) -> Tuple[List[pd.Timestamp], Dict[str, int]]:
    path = Path(args.target_xlsx)
    if not path.exists():
        raise FileNotFoundError(f"Target xlsx does not exist: {path}")

    try:
        df = pd.read_excel(
            path,
            sheet_name=args.sheet_name,
            usecols=[args.date_column, args.time_column],
            engine="openpyxl",
            dtype=object,
        )
    except ValueError as exc:
        raise ValueError(
            f"Failed to read columns `{args.date_column}`, `{args.time_column}` "
            f"from sheet `{args.sheet_name}` in {path}: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(f"No rows found in xlsx: {path} (sheet={args.sheet_name})")

    date_text = df[args.date_column].map(_normalize_date_value)
    time_text = df[args.time_column].map(_normalize_time_value)
    complete_mask = (date_text != "") & (time_text != "")
    combined_text = (date_text + " " + time_text).where(complete_mask, None)

    naive_ts = pd.to_datetime(combined_text, errors="coerce")
    localized_ts = naive_ts.dt.tz_localize(
        args.source_timezone,
        ambiguous="NaT",
        nonexistent="NaT",
    )
    utc_ts = localized_ts.dt.tz_convert("UTC")

    parsed_mask = utc_ts.notna()
    parsed_rows = int(parsed_mask.sum())
    excel_rows_total = int(len(df))
    invalid_rows = excel_rows_total - parsed_rows

    deduped_targets: List[pd.Timestamp] = []
    seen = set()
    for ts in utc_ts[parsed_mask]:
        target_ts = pd.Timestamp(ts).tz_convert("UTC")
        key = target_ts.isoformat()
        if key in seen:
            continue
        seen.add(key)
        deduped_targets.append(target_ts)

    deduped_count = len(deduped_targets)
    max_targets = int(args.max_target_datetimes)
    if max_targets > 0:
        deduped_targets = deduped_targets[:max_targets]

    final_targets_used = len(deduped_targets)
    if final_targets_used == 0:
        raise ValueError(
            "No valid UTC target datetimes parsed from Excel. "
            "Please check date/time columns and timezone settings."
        )

    stats = {
        "excel_rows_total": excel_rows_total,
        "parsed_rows": parsed_rows,
        "invalid_rows": invalid_rows,
        "deduped_targets": deduped_count,
        "final_targets_used": final_targets_used,
    }
    return deduped_targets, stats


def run_excel_job(args: argparse.Namespace, process_minute_fn: ProcessMinuteFn):
    target_datetimes, stats = _load_target_datetimes_from_excel(args)

    surfaces_by_target = generate_surfaces_for_datetime_windows(
        args=args,
        target_datetimes=target_datetimes,
        process_minute_fn=process_minute_fn,
        window_minutes=int(args.window_minutes),
    )

    logger.info("Excel target datetime extraction summary:")
    logger.info("  excel_rows_total=%d", stats["excel_rows_total"])
    logger.info("  parsed_rows=%d", stats["parsed_rows"])
    logger.info("  invalid_rows=%d", stats["invalid_rows"])
    logger.info("  deduped_targets=%d", stats["deduped_targets"])
    logger.info("  final_targets_used=%d", stats["final_targets_used"])
    logger.info("  source_xlsx=%s", Path(args.target_xlsx))
    logger.info("  source_sheet=%s", args.sheet_name)
    logger.info("  source_columns=(%s, %s)", args.date_column, args.time_column)
    logger.info("  source_timezone=%s", args.source_timezone)

    if target_datetimes:
        min_target = min(target_datetimes)
        max_target = max(target_datetimes)
        logger.info("  utc_target_min=%s", _to_utc_string(min_target))
        logger.info("  utc_target_max=%s", _to_utc_string(max_target))
        sample = [_to_utc_string(ts) for ts in target_datetimes[:5]]
        logger.info("  utc_target_samples_first=%s", sample)

    return surfaces_by_target
