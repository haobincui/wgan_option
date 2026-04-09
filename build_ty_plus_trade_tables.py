"""Build merged TY raw trades, minute trade counts, and a news-joined workbook."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.merge_raw_option_data import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_PATH,
    EXPECTED_HEADER,
    MergeStats,
    merge_raw_option_data,
)

DEFAULT_MINUTE_COUNTS_OUTPUT = Path("data/raw/option_data/ty_plus_minute_trade_counts.csv")
DEFAULT_NEWS_XLSX = Path("data/raw/text_embedding/news_with_openai_embeddings_large.xlsx")
DEFAULT_NEWS_OUTPUT = Path(
    "data/raw/text_embedding/news_with_openai_embeddings_large_with_ty_plus_trade_counts.xlsx"
)
DEFAULT_SOURCE_TIMEZONE = "America/New_York"
DEFAULT_SHEET_NAME = "Sheet1"
DEFAULT_DATE_COLUMN = "PD"
DEFAULT_TIME_COLUMN = "ET"
DEFAULT_OFFSET_MINUTES = 5
EMBEDDING_COLUMNS_TO_DROP = ("HD_embedding", "LP_embedding")


def _normalize_date_value(value: object) -> str:
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
    return "" if text.lower() in {"", "nan", "nat", "none"} else text


def _excel_time_fraction_to_hms(value: float) -> str:
    total_seconds = int(round(max(0.0, min(float(value), 1.0)) * 24 * 60 * 60))
    total_seconds = total_seconds % (24 * 60 * 60)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _normalize_time_value(value: object) -> str:
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
    return "" if text.lower() in {"", "nan", "nat", "none"} else text


def load_news_base_frame(
    xlsx_path: Path,
    *,
    sheet_name: str,
    date_column: str,
    time_column: str,
    source_timezone: str,
    offset_minutes: int,
) -> pd.DataFrame:
    news_df = pd.read_excel(
        xlsx_path,
        sheet_name=sheet_name,
        engine="openpyxl",
        dtype=object,
    )
    missing_columns = [column for column in (date_column, time_column) if column not in news_df.columns]
    if missing_columns:
        raise ValueError(f"News xlsx is missing required columns {missing_columns}: {xlsx_path}")

    news_df = news_df.copy()
    news_df.insert(0, "news_row_id", range(1, len(news_df) + 1))
    date_text = news_df[date_column].map(_normalize_date_value)
    time_text = news_df[time_column].map(_normalize_time_value)
    combined = (date_text + " " + time_text).where((date_text != "") & (time_text != ""), None)
    naive_ts = pd.to_datetime(combined, errors="coerce")
    localized = naive_ts.dt.tz_localize(source_timezone, ambiguous="NaT", nonexistent="NaT")
    utc_ts = localized.dt.tz_convert("UTC")
    shifted = utc_ts + pd.Timedelta(minutes=int(offset_minutes))

    news_df["timestamp_utc"] = utc_ts.map(lambda ts: ts.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(ts) else None)
    news_df[f"timestamp_utc_plus_{int(offset_minutes)}m"] = shifted.map(
        lambda ts: ts.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(ts) else None
    )
    return news_df


@dataclass(frozen=True)
class MinuteCountSummary:
    counted_rows: int
    filtered_non_positive_price_rows: int
    filtered_non_positive_volume_rows: int
    filtered_invalid_numeric_rows: int


@dataclass(frozen=True)
class PipelineStats:
    merge_stats: MergeStats
    count_summary: MinuteCountSummary
    minute_count_rows: int
    news_rows: int
    minute_counts_output: Path
    news_output: Path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge TY raw trades, build minute-level trade counts, and merge those counts "
            "onto the news workbook timestamps."
        ),
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--merged-output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--minute-counts-output", type=Path, default=DEFAULT_MINUTE_COUNTS_OUTPUT)
    parser.add_argument("--news-xlsx", type=Path, default=DEFAULT_NEWS_XLSX)
    parser.add_argument("--news-output", type=Path, default=DEFAULT_NEWS_OUTPUT)
    parser.add_argument("--sheet-name", type=str, default=DEFAULT_SHEET_NAME)
    parser.add_argument("--date-column", type=str, default=DEFAULT_DATE_COLUMN)
    parser.add_argument("--time-column", type=str, default=DEFAULT_TIME_COLUMN)
    parser.add_argument("--source-timezone", type=str, default=DEFAULT_SOURCE_TIMEZONE)
    parser.add_argument("--offset-minutes", type=int, default=DEFAULT_OFFSET_MINUTES)
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Fail instead of skipping invalid non-gzip or schema-mismatched source files.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _to_utc_minute_string(value: str) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("min").strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_numeric(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def build_minute_trade_counts(
    merged_output: Path,
    minute_counts_output: Path,
) -> tuple[pd.DataFrame, MinuteCountSummary]:
    counts: Counter[str] = Counter()
    counted_rows = 0
    filtered_non_positive_price_rows = 0
    filtered_non_positive_volume_rows = 0
    filtered_invalid_numeric_rows = 0

    with gzip.open(merged_output, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_HEADER:
            raise ValueError(
                f"Merged CSV header mismatch: expected {EXPECTED_HEADER}, got {reader.fieldnames}"
            )
        for row in reader:
            price = _parse_numeric(row.get("Price"))
            volume = _parse_numeric(row.get("Volume"))
            if price is None or volume is None:
                filtered_invalid_numeric_rows += 1
                continue
            if price <= 0:
                filtered_non_positive_price_rows += 1
                continue
            if volume <= 0:
                filtered_non_positive_volume_rows += 1
                continue

            raw_time = (row.get("Date-Time") or "").strip()
            if not raw_time:
                continue
            counts[_to_utc_minute_string(raw_time)] += 1
            counted_rows += 1

    minute_counts_output.parent.mkdir(parents=True, exist_ok=True)
    minute_df = pd.DataFrame(
        {
            "timestamp_utc": sorted(counts.keys()),
        }
    )
    minute_df["trade_count"] = minute_df["timestamp_utc"].map(counts).astype(int)
    minute_df.to_csv(minute_counts_output, index=False)
    return minute_df, MinuteCountSummary(
        counted_rows=counted_rows,
        filtered_non_positive_price_rows=filtered_non_positive_price_rows,
        filtered_non_positive_volume_rows=filtered_non_positive_volume_rows,
        filtered_invalid_numeric_rows=filtered_invalid_numeric_rows,
    )


def build_news_trade_count_workbook(
    *,
    news_xlsx: Path,
    minute_df: pd.DataFrame,
    news_output: Path,
    sheet_name: str,
    date_column: str,
    time_column: str,
    source_timezone: str,
    offset_minutes: int,
) -> pd.DataFrame:
    minute_lookup = minute_df.set_index("timestamp_utc")["trade_count"]
    news_df = load_news_base_frame(
        news_xlsx,
        sheet_name=sheet_name,
        date_column=date_column,
        time_column=time_column,
        source_timezone=source_timezone,
        offset_minutes=offset_minutes,
    )
    news_df = news_df.drop(columns=list(EMBEDDING_COLUMNS_TO_DROP), errors="ignore")
    news_df["trade_count_at_timestamp_utc"] = (
        news_df["timestamp_utc"].map(minute_lookup).fillna(0).astype(int)
    )

    offset_column = f"timestamp_utc_plus_{int(offset_minutes)}m"
    if offset_column in news_df.columns:
        news_df[f"trade_count_at_{offset_column}"] = (
            news_df[offset_column].map(minute_lookup).fillna(0).astype(int)
        )

    news_output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(news_output, engine="openpyxl") as writer:
        news_df.to_excel(writer, sheet_name="news_with_trade_counts", index=False)
        minute_df.to_excel(writer, sheet_name="minute_trade_counts", index=False)
    return news_df


def run_pipeline(
    *,
    input_dir: Path,
    merged_output: Path,
    minute_counts_output: Path,
    news_xlsx: Path,
    news_output: Path,
    source_timezone: str,
    offset_minutes: int,
    sheet_name: str,
    date_column: str,
    time_column: str,
    fail_on_invalid: bool,
) -> PipelineStats:
    merge_stats = merge_raw_option_data(
        input_dir=input_dir,
        output_path=merged_output,
        skip_invalid=not fail_on_invalid,
    )
    minute_df, count_summary = build_minute_trade_counts(
        merged_output=merged_output,
        minute_counts_output=minute_counts_output,
    )
    news_df = build_news_trade_count_workbook(
        news_xlsx=news_xlsx,
        minute_df=minute_df,
        news_output=news_output,
        sheet_name=sheet_name,
        date_column=date_column,
        time_column=time_column,
        source_timezone=source_timezone,
        offset_minutes=offset_minutes,
    )
    return PipelineStats(
        merge_stats=merge_stats,
        count_summary=count_summary,
        minute_count_rows=len(minute_df),
        news_rows=len(news_df),
        minute_counts_output=minute_counts_output,
        news_output=news_output,
    )


def main(argv: Iterable[str] | None = None) -> PipelineStats:
    args = _parse_args(argv)
    stats = run_pipeline(
        input_dir=Path(args.input_dir),
        merged_output=Path(args.merged_output),
        minute_counts_output=Path(args.minute_counts_output),
        news_xlsx=Path(args.news_xlsx),
        news_output=Path(args.news_output),
        source_timezone=str(args.source_timezone),
        offset_minutes=int(args.offset_minutes),
        sheet_name=str(args.sheet_name),
        date_column=str(args.date_column),
        time_column=str(args.time_column),
        fail_on_invalid=bool(args.fail_on_invalid),
    )
    print(f"Merged files: {stats.merge_stats.files_merged}")
    print(f"Merged rows: {stats.merge_stats.rows_written}")
    if stats.merge_stats.skipped_files:
        print(f"Skipped invalid files: {len(stats.merge_stats.skipped_files)}")
        for path in stats.merge_stats.skipped_files:
            print(f"  {path}")
    print(f"Counted rows: {stats.count_summary.counted_rows}")
    print(
        "Filtered non-positive price rows: "
        f"{stats.count_summary.filtered_non_positive_price_rows}"
    )
    print(
        "Filtered non-positive volume rows: "
        f"{stats.count_summary.filtered_non_positive_volume_rows}"
    )
    print(
        "Filtered invalid numeric rows: "
        f"{stats.count_summary.filtered_invalid_numeric_rows}"
    )
    print(f"Minute-count rows: {stats.minute_count_rows}")
    print(f"News rows written: {stats.news_rows}")
    print(f"Minute-count CSV: {stats.minute_counts_output}")
    print(f"News workbook: {stats.news_output}")
    return stats


if __name__ == "__main__":
    main()
