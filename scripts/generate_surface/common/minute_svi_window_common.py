"""Shared datetime-window minute-SVI helpers."""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import logging
import shlex
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

if __package__ in {None, ""}:
    import scripts._path_setup  # noqa: F401
    from scripts.generate_surface.common.config_utils import (  # noqa: E402
        build_config_scope,
        load_surface_builder_section,
        resolve_config_variables,
    )
    from scripts.generate_surface.common.minute_svi_common import (  # noqa: E402
        DEFAULT_CONFIG_PATH,
        PRECALIB_CSV_HEADERS,
        _build_rows_for_minute,
        _collect_spot_and_option_rows,
        _get_file_target_future_month_code,
        _get_target_future_month_code,
        _infer_file_date_range,
        _log_cli_arguments,
        _parse_args as _parse_base_args,
        _resolve_config_path,
        _setup_runtime,
        _to_utc_minute_string,
        ProcessMinuteFn,
    )
else:
    from .config_utils import build_config_scope, load_surface_builder_section, resolve_config_variables  # noqa: E402
    from .minute_svi_common import (  # noqa: E402
        DEFAULT_CONFIG_PATH,
        PRECALIB_CSV_HEADERS,
        ProcessMinuteFn,
        _build_rows_for_minute,
        _collect_spot_and_option_rows,
        _get_file_target_future_month_code,
        _get_target_future_month_code,
        _infer_file_date_range,
        _log_cli_arguments,
        _parse_args as _parse_base_args,
        _resolve_config_path,
        _setup_runtime,
        _to_utc_minute_string,
    )

logger = logging.getLogger(__name__)

DatetimeLike = Union[str, datetime, pd.Timestamp]
SUPPORTED_MINUTE_SVI_WINDOW_CONFIG_KEYS = {
    "target_datetimes",
    "target_datetimes_file",
    "window_minutes",
}


def _coerce_target_datetime_defaults(value: Any) -> List[str]:
    def _normalize_item(item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, str):
            return item.strip()
        try:
            ts = pd.Timestamp(item)
        except Exception:
            return str(item).strip()
        if pd.isna(ts):
            return ""
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [normalized for item in value if (normalized := _normalize_item(item))]
    normalized = _normalize_item(value)
    return [normalized] if normalized else []


def _load_window_config(config_path_value: str) -> Dict[str, Any]:
    _, config_root, window_section = load_surface_builder_section(
        config_path_value,
        section_key="minute_svi_window",
        supported_keys=SUPPORTED_MINUTE_SVI_WINDOW_CONFIG_KEYS,
    )
    return resolve_config_variables(window_section, extra_scope=build_config_scope(config_root))


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
            print("Datetime-window arguments (added by this script):")
            print("  --target-datetime VALUE")
            print("      Repeatable; comma-separated values are also supported.")
            print("  --target-datetimes-file PATH")
            print("      Optional text file containing datetime values.")
            print("  --window-minutes N")
            print("      Calibrate in +/-N minutes around each target datetime (default: 3).")
            raise
    else:
        config_defaults = _load_window_config(pre_args.config)

    window_parser = argparse.ArgumentParser(add_help=False)
    window_parser.add_argument(
        "--target-datetime",
        dest="target_datetimes",
        action="append",
        default=_coerce_target_datetime_defaults(config_defaults.get("target_datetimes", [])),
        help=(
            "Target datetime in UTC by default (repeatable, supports comma-separated "
            "values in each argument). Example: 2026-03-09T14:35:00Z"
        ),
    )
    window_parser.add_argument(
        "--target-datetimes-file",
        type=str,
        default=str(config_defaults.get("target_datetimes_file", "")),
        help="Optional text file containing target datetimes (one per line, comma also supported).",
    )
    window_parser.add_argument(
        "--window-minutes",
        type=int,
        default=int(config_defaults.get("window_minutes", 3)),
        help="Calibrate minute surfaces within +/- this many minutes around each target datetime.",
    )

    window_args, remaining_argv = window_parser.parse_known_args(argv_list)
    args = _parse_base_args(remaining_argv)
    args.target_datetimes = list(window_args.target_datetimes)
    args.target_datetimes_file = str(window_args.target_datetimes_file)
    args.window_minutes = int(window_args.window_minutes)
    return args


def _split_datetime_tokens(values: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for raw in values:
        for token in str(raw).split(","):
            stripped = token.strip()
            if stripped:
                tokens.append(stripped)
    return tokens


def _collect_target_datetime_tokens(cli_values: Sequence[str], file_path: str) -> List[str]:
    tokens = _split_datetime_tokens(cli_values)

    file_path = file_path.strip()
    if file_path:
        src = Path(file_path)
        if not src.exists():
            raise FileNotFoundError(f"target datetime file does not exist: {src}")
        file_tokens: List[str] = []
        for line in src.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            file_tokens.extend(_split_datetime_tokens([stripped]))
        tokens.extend(file_tokens)

    deduped: List[str] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _to_utc_minute_ts(value: DatetimeLike) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("min")


def _build_target_window_map(
    target_datetimes: Sequence[DatetimeLike],
    window_minutes: int,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if window_minutes < 0:
        raise ValueError(f"window_minutes must be >= 0, got {window_minutes}")
    if not target_datetimes:
        raise ValueError("target_datetimes is empty; please provide at least one datetime.")

    window_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for raw_value in target_datetimes:
        center_minute = _to_utc_minute_ts(raw_value)
        target_key = _to_utc_minute_string(center_minute)
        backward_minutes = [
            center_minute + pd.Timedelta(minutes=offset)
            for offset in range(-window_minutes, 1)
        ]
        forward_minutes = [
            center_minute + pd.Timedelta(minutes=offset)
            for offset in range(0, window_minutes + 1)
        ]
        window_map[target_key] = {
            "target_ts": center_minute,
            "backward": {
                "anchor_ts": center_minute,
                "minutes": sorted(backward_minutes),
            },
            "forward": {
                "anchor_ts": center_minute + pd.Timedelta(minutes=window_minutes),
                "minutes": sorted(forward_minutes),
            },
        }
    return window_map


def _filter_files_for_target_windows(
    files: Sequence[str],
    target_window_map: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[str]:
    if not target_window_map:
        return list(files)

    window_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for target_spec in target_window_map.values():
        window_minutes = sorted(
            set(target_spec["backward"]["minutes"]) | set(target_spec["forward"]["minutes"])
        )
        if not window_minutes:
            continue
        window_ranges.append((window_minutes[0], window_minutes[-1]))

    if not window_ranges:
        return list(files)

    filtered_files: List[str] = []
    for path in files:
        file_start_date, file_end_date = _infer_file_date_range(path)
        if file_start_date is None or file_end_date is None:
            filtered_files.append(path)
            continue

        for range_start_ts, range_end_ts in window_ranges:
            if file_end_date >= range_start_ts.date() and file_start_date <= range_end_ts.date():
                filtered_files.append(path)
                break
    return filtered_files


def _extract_target_surfaces(
    side_results: Dict[Tuple[str, str], Optional[Dict[str, Any]]],
    target_window_map: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_target: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for target_key, target_spec in target_window_map.items():
        target_results: Dict[str, Dict[str, Any]] = {}
        for direction in ("backward", "forward"):
            anchor_ts = target_spec[direction]["anchor_ts"]
            target_results[direction] = {
                "snapshot_time_utc": _to_utc_minute_string(anchor_ts),
                "svi_params": side_results.get((target_key, direction)),
            }
        by_target[target_key] = target_results
    return by_target


def generate_surfaces_for_datetime_windows(
    args: argparse.Namespace,
    target_datetimes: Sequence[DatetimeLike],
    process_minute_fn: ProcessMinuteFn,
    window_minutes: int = 3,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    output_json_path, log_path, expiry_inference_date, calendar, vol_daycount, expiration_time_utc, all_files = (
        _setup_runtime(args)
    )

    target_window_map = _build_target_window_map(
        target_datetimes=target_datetimes,
        window_minutes=int(window_minutes),
    )
    files = _filter_files_for_target_windows(all_files, target_window_map)
    if not files:
        raise FileNotFoundError("No input files overlap the requested target datetime windows.")
    bucket_rows: Dict[Tuple[str, str], List[Any]] = {}
    minute_bucket_map: Dict[pd.Timestamp, List[Tuple[str, str]]] = defaultdict(list)
    pending_buckets: List[Tuple[pd.Timestamp, str, str]] = []
    all_window_minutes: set[pd.Timestamp] = set()

    for target_key, target_spec in target_window_map.items():
        for direction in ("backward", "forward"):
            bucket_key = (target_key, direction)
            bucket_rows[bucket_key] = []
            anchor_ts = target_spec[direction]["anchor_ts"]
            pending_buckets.append((anchor_ts, target_key, direction))
            for minute_ts in target_spec[direction]["minutes"]:
                minute_bucket_map[minute_ts].append(bucket_key)
                all_window_minutes.add(minute_ts)
            all_window_minutes.add(anchor_ts)

    pending_buckets.sort(key=lambda item: (item[0], item[1], item[2]))

    logger.info("Start minute SVI generation for target datetime windows")
    _log_cli_arguments(args)
    logger.info("Config file=%s", Path(args.config))
    logger.info("Expiry inference data_date=%s", expiry_inference_date.isoformat())
    logger.info("Input files=%d", len(files))
    logger.info("Output JSON=%s", output_json_path)
    logger.info("Log file=%s", log_path)
    logger.info("Expiration time UTC=%s", expiration_time_utc.isoformat())
    logger.info("Max pre-calib IV=%s", float(args.max_precalib_iv))
    logger.info("Save pre-calib CSV=%s", bool(args.save_precalib_csv))
    logger.info("Pre-calib CSV path=%s", Path(args.precalib_csv))
    logger.info("Target datetimes=%d", len(target_window_map))
    logger.info("Window minutes=%d", int(window_minutes))
    logger.info("Window surface sides=%d", len(pending_buckets))
    if all_window_minutes:
        logger.info(
            "Target window minute range=%s..%s",
            _to_utc_minute_string(min(all_window_minutes)),
            _to_utc_minute_string(max(all_window_minutes)),
        )
    parsed_file_ranges = [rng for rng in (_infer_file_date_range(path) for path in files) if rng[0] is not None]
    if parsed_file_ranges:
        logger.info(
            "Input file date span=%s..%s",
            min(start for start, _ in parsed_file_ranges).isoformat(),
            max(end for _, end in parsed_file_ranges if end is not None).isoformat(),
        )

    start_ts = time.time()
    stats: Dict[str, int] = defaultdict(int)
    side_results: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
    contract_cache: Dict[str, Any] = {}
    visible_spot_by_key: Dict[tuple[str, str], float] = {}
    precalib_csv_path = Path(args.precalib_csv)
    precalib_writer: Optional[csv.DictWriter] = None
    precalib_fp = None

    if args.save_precalib_csv:
        precalib_csv_path.parent.mkdir(parents=True, exist_ok=True)
        precalib_fp = precalib_csv_path.open("w", encoding="utf-8", newline="")
        precalib_writer = csv.DictWriter(precalib_fp, fieldnames=PRECALIB_CSV_HEADERS)
        precalib_writer.writeheader()

    stop_due_to_max_minutes = False
    next_bucket_idx = 0

    def finalize_pending_buckets(current_minute: Optional[pd.Timestamp], include_equal: bool) -> None:
        nonlocal next_bucket_idx
        while next_bucket_idx < len(pending_buckets):
            anchor_ts, target_key, direction = pending_buckets[next_bucket_idx]
            if current_minute is not None:
                if anchor_ts > current_minute:
                    break
                if anchor_ts == current_minute and not include_equal:
                    break

            local_results: Dict[str, Dict[str, Any]] = {}
            process_minute_fn(
                minute_ts=anchor_ts,
                rows=bucket_rows[(target_key, direction)],
                days_in_year=int(args.days_in_year),
                min_strikes_per_expiry=int(args.min_strikes_per_expiry),
                min_expiries_per_minute=int(args.min_expiries_per_minute),
                max_precalib_iv=float(args.max_precalib_iv),
                vol_daycount=vol_daycount,
                calendar=calendar,
                target_future_month_code=_get_target_future_month_code(anchor_ts.month),
                last_spot_by_key=dict(visible_spot_by_key),
                results=local_results,
                stats=stats,
                precalib_writer=precalib_writer,
                tau_anchor_ts=anchor_ts,
                count_stat_key="window_surface_attempts",
            )
            side_results[(target_key, direction)] = local_results.get(_to_utc_minute_string(anchor_ts))
            next_bucket_idx += 1

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
                        if args.max_minutes and args.max_minutes > 0 and stats["encountered_minutes"] >= args.max_minutes:
                            stop_due_to_max_minutes = True
                            break
                        stats["encountered_minutes"] += 1
                        finalize_pending_buckets(current_minute=minute_ts, include_equal=False)
                        rows = _build_rows_for_minute(
                            minute_df=minute_df,
                            expiry_inference_date=expiry_inference_date,
                            expiration_time_utc=expiration_time_utc,
                            calendar=calendar,
                            contract_cache=contract_cache,
                            stats=stats,
                        )
                        if rows:
                            _collect_spot_and_option_rows(
                                rows=rows,
                                target_future_month_code=None,
                                last_spot_by_key=visible_spot_by_key,
                            )
                            for bucket_key in minute_bucket_map.get(minute_ts, []):
                                bucket_rows[bucket_key].extend(rows)
                        if minute_ts not in minute_bucket_map:
                            stats["skip_not_target_minute"] += 1
                        finalize_pending_buckets(current_minute=minute_ts, include_equal=True)

                    if stop_due_to_max_minutes:
                        break
            except (gzip.BadGzipFile, EOFError, OSError, pd.errors.ParserError) as exc:
                logger.warning("Skip corrupted file: %s (%s)", path, exc)
                stats["skip_bad_gzip"] += 1
                pending_minute_df = None
                continue

            if not stop_due_to_max_minutes and pending_minute_df is not None and not pending_minute_df.empty:
                for minute_ts, minute_df in pending_minute_df.groupby("minute", sort=True):
                    if args.max_minutes and args.max_minutes > 0 and stats["encountered_minutes"] >= args.max_minutes:
                        stop_due_to_max_minutes = True
                        break
                    stats["encountered_minutes"] += 1
                    finalize_pending_buckets(current_minute=minute_ts, include_equal=False)
                    rows = _build_rows_for_minute(
                        minute_df=minute_df,
                        expiry_inference_date=expiry_inference_date,
                        expiration_time_utc=expiration_time_utc,
                        calendar=calendar,
                        contract_cache=contract_cache,
                        stats=stats,
                    )
                    if rows:
                        _collect_spot_and_option_rows(
                            rows=rows,
                            target_future_month_code=None,
                            last_spot_by_key=visible_spot_by_key,
                        )
                        for bucket_key in minute_bucket_map.get(minute_ts, []):
                            bucket_rows[bucket_key].extend(rows)
                    if minute_ts not in minute_bucket_map:
                        stats["skip_not_target_minute"] += 1
                    finalize_pending_buckets(current_minute=minute_ts, include_equal=True)

            if stop_due_to_max_minutes:
                logger.info("Stop early due to --max-minutes=%d", args.max_minutes)
                break
    finally:
        finalize_pending_buckets(current_minute=None, include_equal=True)
        if precalib_fp is not None:
            precalib_fp.close()

    surfaces_by_target = _extract_target_surfaces(side_results, target_window_map)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(surfaces_by_target, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_ts
    logger.info("Finished minute SVI generation for target datetime windows")
    logger.info("Elapsed seconds: %.2f", elapsed)
    logger.info("Summary stats:")
    logger.info("  total_files=%d", stats["total_files"])
    logger.info("  encountered_minutes=%d", stats["encountered_minutes"])
    logger.info("  window_surface_sides=%d", len(pending_buckets))
    logger.info("  window_surface_attempts=%d", stats["window_surface_attempts"])
    logger.info("  calibrated_window_surfaces=%d", stats["calibrated_minutes"])
    logger.info("  skip_not_target_minute=%d", stats["skip_not_target_minute"])
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
    logger.info("Saved %d target window surface bundles to %s", len(surfaces_by_target), output_json_path)

    covered_targets = sum(
        1
        for direction_map in surfaces_by_target.values()
        if direction_map["backward"]["svi_params"] is not None
        or direction_map["forward"]["svi_params"] is not None
    )
    logger.info("Targets with at least one calibrated direction=%d/%d", covered_targets, len(surfaces_by_target))
    return surfaces_by_target


def run_window_job(args: argparse.Namespace, process_minute_fn: ProcessMinuteFn) -> Dict[str, Dict[str, Dict[str, Any]]]:
    target_tokens = _collect_target_datetime_tokens(
        cli_values=args.target_datetimes,
        file_path=args.target_datetimes_file,
    )
    if not target_tokens:
        raise ValueError(
            "Please provide target datetimes via --target-datetime and/or --target-datetimes-file."
        )

    logger.info("Target datetime tokens=%s", " ".join(shlex.quote(x) for x in target_tokens))
    return generate_surfaces_for_datetime_windows(
        args=args,
        target_datetimes=target_tokens,
        process_minute_fn=process_minute_fn,
        window_minutes=int(args.window_minutes),
    )
