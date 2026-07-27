"""Canonical Factiva publication-time parsing for thesis data pipelines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from typing import Any, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

DEFAULT_NEWS_SOURCE_TIMEZONE = "Europe/London"


def normalize_news_date(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, (pd.Timestamp, dt_datetime)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dt_date):
        return value.isoformat()
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "nat", "none"} else text


def excel_time_fraction_to_hms(value: float) -> str:
    total_seconds = int(round(max(0.0, min(float(value), 1.0)) * 24 * 60 * 60))
    total_seconds %= 24 * 60 * 60
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def normalize_news_time(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, (pd.Timestamp, dt_datetime)):
        return value.strftime("%H:%M:%S")
    if isinstance(value, dt_time):
        return value.strftime("%H:%M:%S")
    if isinstance(value, (int, float)):
        numeric = float(value)
        if 0.0 <= numeric < 1.0:
            return excel_time_fraction_to_hms(numeric)
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "nat", "none"} else text


@dataclass(frozen=True)
class ParsedNewsTimestamps:
    """Vectorized timestamp result with explicit audit fields."""

    source_local_timestamp: pd.Series
    timestamp_utc: pd.Series
    utc_offset_minutes: pd.Series
    parse_status: pd.Series

    @property
    def parsed_count(self) -> int:
        return int((self.parse_status == "ok").sum())

    @property
    def invalid_count(self) -> int:
        return int((self.parse_status != "ok").sum())


def _validate_timezone(source_timezone: str) -> str:
    value = str(source_timezone).strip()
    if not value:
        raise ValueError("Factiva source timezone must be explicit and non-empty.")
    try:
        ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown Factiva source timezone: {value}") from exc
    return value


def parse_news_timestamps(
    date_values: Sequence[Any] | pd.Series,
    time_values: Sequence[Any] | pd.Series,
    *,
    source_timezone: str = DEFAULT_NEWS_SOURCE_TIMEZONE,
) -> ParsedNewsTimestamps:
    """Parse Factiva PD/ET as local Europe/London time and convert to UTC.

    The timezone remains configurable for historical fixtures, but all canonical
    RQ1-RQ3 configs explicitly use ``Europe/London``. Ambiguous/nonexistent DST
    rows are retained as audit failures instead of being silently interpreted.
    """

    timezone_name = _validate_timezone(source_timezone)
    dates = pd.Series(date_values, copy=False).reset_index(drop=True)
    times = pd.Series(time_values, copy=False).reset_index(drop=True)
    if len(dates) != len(times):
        raise ValueError(
            f"Factiva date/time column length mismatch: {len(dates)} != {len(times)}."
        )

    date_text = dates.map(normalize_news_date)
    time_text = times.map(normalize_news_time)
    complete = (date_text != "") & (time_text != "")
    combined = (date_text + " " + time_text).where(complete, None)
    naive = pd.to_datetime(combined, errors="coerce")
    localized = naive.dt.tz_localize(
        timezone_name,
        ambiguous="NaT",
        nonexistent="NaT",
    )
    utc = localized.dt.tz_convert("UTC")

    status = pd.Series(
        np.where(
            ~complete,
            "missing_pd_or_et",
            np.where(
                naive.isna(),
                "invalid_pd_or_et",
                np.where(localized.isna(), "dst_ambiguous_or_nonexistent", "ok"),
            ),
        ),
        index=dates.index,
        dtype="object",
    )
    local_text = naive.map(
        lambda value: (
            ""
            if value is None or pd.isna(value)
            else pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S")
        )
    )
    utc_text = utc.map(
        lambda value: (
            None
            if value is None or pd.isna(value)
            else pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        )
    )
    offsets = localized.map(
        lambda value: (
            np.nan
            if value is None or pd.isna(value)
            else float(pd.Timestamp(value).utcoffset().total_seconds() / 60.0)
        )
    )
    return ParsedNewsTimestamps(
        source_local_timestamp=local_text,
        timestamp_utc=utc_text,
        utc_offset_minutes=offsets,
        parse_status=status,
    )
