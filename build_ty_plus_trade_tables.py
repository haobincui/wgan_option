"""Build TY minute trade counts and comparison workbooks from a merged raw trade CSV."""

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

from scripts.merge_raw_option_data import DEFAULT_OUTPUT_PATH, EXPECTED_HEADER  # noqa: E402

DEFAULT_MERGED_CSV = DEFAULT_OUTPUT_PATH
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
TRADE_COUNT_PAIR_THRESHOLD = 5
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


def _offset_timestamp_column(offset_minutes: int) -> str:
    return f"timestamp_utc_plus_{int(offset_minutes)}m"


def _trade_count_columns(offset_minutes: int) -> tuple[str, str, str]:
    t_col = "trade_count_at_timestamp_utc"
    offset_column = _offset_timestamp_column(offset_minutes)
    t_plus_col = f"trade_count_at_{offset_column}"
    flag_col = "trade_count_pair_ge_5_flag"
    return t_col, t_plus_col, flag_col


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
    news_df[_offset_timestamp_column(offset_minutes)] = shifted.map(
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
    count_summary: MinuteCountSummary
    minute_count_rows: int
    news_rows: int
    minute_counts_output: Path
    news_output: Path
    merged_vol_output: Path | None


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build TY minute-level trade counts from a merged raw trade CSV, merge those counts "
            "onto the news workbook timestamps, and optionally enrich merged_vol.xlsx."
        ),
    )
    parser.add_argument("--merged-csv", type=Path, default=DEFAULT_MERGED_CSV)
    parser.add_argument("--minute-counts-output", type=Path, default=DEFAULT_MINUTE_COUNTS_OUTPUT)
    parser.add_argument("--news-xlsx", type=Path, default=DEFAULT_NEWS_XLSX)
    parser.add_argument("--news-output", type=Path, default=DEFAULT_NEWS_OUTPUT)
    parser.add_argument("--merged-vol-xlsx", type=Path, default=None)
    parser.add_argument("--merged-vol-output", type=Path, default=None)
    parser.add_argument("--sheet-name", type=str, default=DEFAULT_SHEET_NAME)
    parser.add_argument("--date-column", type=str, default=DEFAULT_DATE_COLUMN)
    parser.add_argument("--time-column", type=str, default=DEFAULT_TIME_COLUMN)
    parser.add_argument("--source-timezone", type=str, default=DEFAULT_SOURCE_TIMEZONE)
    parser.add_argument("--offset-minutes", type=int, default=DEFAULT_OFFSET_MINUTES)
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


def _normalize_news_row_id_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _default_merged_vol_output_path(merged_vol_xlsx: Path) -> Path:
    return merged_vol_xlsx.with_name(
        f"{merged_vol_xlsx.stem}_with_trade_counts{merged_vol_xlsx.suffix}"
    )


def _apply_trade_counts_to_news_df(
    news_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    *,
    offset_minutes: int,
) -> pd.DataFrame:
    news_df = news_df.copy()
    minute_lookup = minute_df.set_index("timestamp_utc")["trade_count"]
    t_col, t_plus_col, flag_col = _trade_count_columns(offset_minutes)
    offset_column = _offset_timestamp_column(offset_minutes)

    news_df[t_col] = news_df["timestamp_utc"].map(minute_lookup).fillna(0).astype(int)
    news_df[t_plus_col] = news_df[offset_column].map(minute_lookup).fillna(0).astype(int)
    news_df[flag_col] = (
        (news_df[t_col] >= TRADE_COUNT_PAIR_THRESHOLD)
        & (news_df[t_plus_col] >= TRADE_COUNT_PAIR_THRESHOLD)
    )
    return news_df


def _build_news_trade_lookup(news_df: pd.DataFrame, *, offset_minutes: int) -> pd.DataFrame:
    t_col, t_plus_col, flag_col = _trade_count_columns(offset_minutes)
    lookup = news_df[["news_row_id", t_col, t_plus_col, flag_col]].copy()
    lookup["news_row_id"] = _normalize_news_row_id_series(lookup["news_row_id"])
    return lookup


def _finalize_trade_count_columns(
    df: pd.DataFrame,
    *,
    offset_minutes: int,
) -> pd.DataFrame:
    t_col, t_plus_col, flag_col = _trade_count_columns(offset_minutes)
    df[t_col] = pd.to_numeric(df[t_col], errors="coerce").fillna(0).astype(int)
    df[t_plus_col] = pd.to_numeric(df[t_plus_col], errors="coerce").fillna(0).astype(int)
    df[flag_col] = df[flag_col].fillna(False).astype(bool)
    return df


def build_minute_trade_counts(
    merged_csv: Path,
    minute_counts_output: Path,
) -> tuple[pd.DataFrame, MinuteCountSummary]:
    counts: Counter[str] = Counter()
    counted_rows = 0
    filtered_non_positive_price_rows = 0
    filtered_non_positive_volume_rows = 0
    filtered_invalid_numeric_rows = 0

    with gzip.open(merged_csv, "rt", encoding="utf-8", newline="") as handle:
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
    minute_df = pd.DataFrame({"timestamp_utc": sorted(counts.keys())})
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
    news_df = load_news_base_frame(
        news_xlsx,
        sheet_name=sheet_name,
        date_column=date_column,
        time_column=time_column,
        source_timezone=source_timezone,
        offset_minutes=offset_minutes,
    )
    news_df = _apply_trade_counts_to_news_df(news_df, minute_df, offset_minutes=offset_minutes)
    news_df = news_df.drop(columns=list(EMBEDDING_COLUMNS_TO_DROP), errors="ignore")

    news_output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(news_output, engine="openpyxl") as writer:
        news_df.to_excel(writer, sheet_name="news_with_trade_counts", index=False)
        minute_df.to_excel(writer, sheet_name="minute_trade_counts", index=False)
    return news_df


def build_merged_vol_trade_count_workbook(
    *,
    merged_vol_xlsx: Path,
    merged_vol_output: Path,
    news_df: pd.DataFrame,
    offset_minutes: int,
) -> Path:
    with pd.ExcelFile(merged_vol_xlsx) as workbook:
        sheet_names = list(workbook.sheet_names)
        sheets = {sheet_name: workbook.parse(sheet_name) for sheet_name in sheet_names}

        if "news_surface_pair_audit" not in sheets:
            raise ValueError(
                f"merged_vol workbook is missing news_surface_pair_audit: {merged_vol_xlsx}"
            )
        if "gan_input_ready" not in sheets:
            raise ValueError(f"merged_vol workbook is missing gan_input_ready: {merged_vol_xlsx}")

        trade_lookup = _build_news_trade_lookup(news_df, offset_minutes=offset_minutes)
        t_col, t_plus_col, flag_col = _trade_count_columns(offset_minutes)
        trade_columns = [t_col, t_plus_col, flag_col]

        audit_df = sheets["news_surface_pair_audit"].copy()
        if "news_row_id" not in audit_df.columns:
            raise ValueError(f"news_surface_pair_audit is missing news_row_id: {merged_vol_xlsx}")
        audit_df = audit_df.drop(columns=trade_columns, errors="ignore")
        audit_df["news_row_id"] = _normalize_news_row_id_series(audit_df["news_row_id"])
        audit_df = audit_df.merge(trade_lookup, on="news_row_id", how="left")
        audit_df = _finalize_trade_count_columns(audit_df, offset_minutes=offset_minutes)
        sheets["news_surface_pair_audit"] = audit_df

        sample_trade_lookup = (
            audit_df[["sample_id", *trade_columns]]
            .dropna(subset=["sample_id"])
            .drop_duplicates(subset=["sample_id"], keep="first")
        )
        gan_df = sheets["gan_input_ready"].copy()
        if "sample_id" not in gan_df.columns:
            raise ValueError(f"gan_input_ready is missing sample_id: {merged_vol_xlsx}")
        gan_df = gan_df.drop(columns=trade_columns, errors="ignore")
        gan_df = gan_df.merge(sample_trade_lookup, on="sample_id", how="left")
        gan_df = _finalize_trade_count_columns(gan_df, offset_minutes=offset_minutes)
        sheets["gan_input_ready"] = gan_df

    merged_vol_output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(merged_vol_output, engine="openpyxl") as writer:
        for sheet_name in sheet_names:
            sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
    return merged_vol_output


def run_pipeline(
    *,
    merged_csv: Path,
    minute_counts_output: Path,
    news_xlsx: Path,
    news_output: Path,
    source_timezone: str,
    offset_minutes: int,
    sheet_name: str,
    date_column: str,
    time_column: str,
    merged_vol_xlsx: Path | None,
    merged_vol_output: Path | None,
) -> PipelineStats:
    minute_df, count_summary = build_minute_trade_counts(
        merged_csv=merged_csv,
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

    written_merged_vol_output: Path | None = None
    if merged_vol_xlsx is not None:
        target_output = (
            merged_vol_output
            if merged_vol_output is not None
            else _default_merged_vol_output_path(merged_vol_xlsx)
        )
        written_merged_vol_output = build_merged_vol_trade_count_workbook(
            merged_vol_xlsx=merged_vol_xlsx,
            merged_vol_output=target_output,
            news_df=news_df,
            offset_minutes=offset_minutes,
        )

    return PipelineStats(
        count_summary=count_summary,
        minute_count_rows=len(minute_df),
        news_rows=len(news_df),
        minute_counts_output=minute_counts_output,
        news_output=news_output,
        merged_vol_output=written_merged_vol_output,
    )


def main(argv: Iterable[str] | None = None) -> PipelineStats:
    args = _parse_args(argv)
    stats = run_pipeline(
        merged_csv=Path(args.merged_csv),
        minute_counts_output=Path(args.minute_counts_output),
        news_xlsx=Path(args.news_xlsx),
        news_output=Path(args.news_output),
        source_timezone=str(args.source_timezone),
        offset_minutes=int(args.offset_minutes),
        sheet_name=str(args.sheet_name),
        date_column=str(args.date_column),
        time_column=str(args.time_column),
        merged_vol_xlsx=Path(args.merged_vol_xlsx) if args.merged_vol_xlsx is not None else None,
        merged_vol_output=Path(args.merged_vol_output) if args.merged_vol_output is not None else None,
    )
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
    if stats.merged_vol_output is not None:
        print(f"Merged-vol workbook: {stats.merged_vol_output}")
    else:
        print("Merged-vol workbook: not requested")
    return stats


if __name__ == "__main__":
    main()
