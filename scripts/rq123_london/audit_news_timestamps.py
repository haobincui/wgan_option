#!/usr/bin/env python3
"""Audit Factiva PD/ET Europe/London conversion against embedded GMT stamps."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wgan_option.news_time import parse_news_timestamps  # noqa: E402


GMT_PATTERN = re.compile(r"^\s*(\d{2})(\d{2})\s+GMT\b", re.IGNORECASE)


def _embedded_gmt_minutes(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    match = GMT_PATTERN.search(str(value))
    if match is None:
        return float("nan")
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        return float("nan")
    return float(hour * 60 + minute)


def _circular_minute_difference(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    raw = abs(float(left) - float(right))
    return float(min(raw, 1440.0 - raw))


def run(args: argparse.Namespace) -> Path:
    source = Path(args.news_xlsx).expanduser()
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_excel(
        source,
        sheet_name=args.sheet_name,
        usecols=lambda column: column in {"PD", "ET", "LP", "ArticleID", "SourceFile"},
        dtype=object,
    )
    missing = sorted({"PD", "ET"} - set(frame.columns))
    if missing:
        raise ValueError(f"News workbook is missing required columns {missing}: {source}")
    if "LP" not in frame.columns:
        frame["LP"] = ""

    parsed = parse_news_timestamps(
        frame["PD"],
        frame["ET"],
        source_timezone=args.source_timezone,
    )
    utc = pd.to_datetime(parsed.timestamp_utc, utc=True, errors="coerce")
    utc_minutes = utc.map(
        lambda value: (
            float("nan")
            if value is None or pd.isna(value)
            else float(pd.Timestamp(value).hour * 60 + pd.Timestamp(value).minute)
        )
    )
    embedded_minutes = frame["LP"].map(_embedded_gmt_minutes)
    mismatch = pd.Series(
        [
            _circular_minute_difference(left, right)
            for left, right in zip(utc_minutes, embedded_minutes)
        ],
        dtype=float,
    )
    has_embedded = embedded_minutes.notna()
    within_tolerance = has_embedded & mismatch.le(float(args.match_tolerance_minutes))

    audit = pd.DataFrame(
        {
            "news_row_id": np.arange(1, len(frame) + 1, dtype=int),
            "article_id": frame.get("ArticleID", ""),
            "source_file": frame.get("SourceFile", ""),
            "pd": frame["PD"],
            "et": frame["ET"],
            "source_local_timestamp": parsed.source_local_timestamp,
            "source_timezone": args.source_timezone,
            "source_utc_offset_minutes": parsed.utc_offset_minutes,
            "timestamp_utc": parsed.timestamp_utc,
            "timestamp_parse_status": parsed.parse_status,
            "embedded_gmt_minutes": embedded_minutes,
            "gmt_crosscheck_abs_diff_minutes": mismatch,
            "gmt_crosscheck_within_tolerance": within_tolerance.astype(int),
        }
    )
    audit_path = output / "factiva_timestamp_audit.csv"
    audit.to_csv(audit_path, index=False)

    embedded_count = int(has_embedded.sum())
    match_count = int(within_tolerance.sum())
    match_rate = float(match_count / embedded_count) if embedded_count else float("nan")
    summary = {
        "news_xlsx": str(source),
        "source_timezone": str(args.source_timezone),
        "row_count": int(len(frame)),
        "parse_status_counts": {
            str(key): int(value)
            for key, value in parsed.parse_status.value_counts(dropna=False).items()
        },
        "embedded_gmt_count": embedded_count,
        "gmt_match_tolerance_minutes": float(args.match_tolerance_minutes),
        "embedded_gmt_match_count": match_count,
        "embedded_gmt_match_rate": match_rate,
        "minimum_required_match_rate": float(args.minimum_gmt_match_rate),
        "status": (
            "ok"
            if embedded_count > 0
            and match_rate >= float(args.minimum_gmt_match_rate)
            else "failed"
        ),
        "audit_csv": str(audit_path),
    }
    summary_path = output / "factiva_timestamp_audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if summary["status"] != "ok":
        raise ValueError(
            "Factiva GMT cross-check failed: "
            f"match_rate={match_rate:.6f}, required={args.minimum_gmt_match_rate}."
        )
    print(summary_path)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-xlsx", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sheet-name", default="Sheet1")
    parser.add_argument("--source-timezone", default="Europe/London")
    parser.add_argument("--match-tolerance-minutes", type=float, default=2.0)
    parser.add_argument("--minimum-gmt-match-rate", type=float, default=0.95)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
