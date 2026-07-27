#!/usr/bin/env python3
"""Freeze official rate inputs and rule-derived TY option expiry references."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from wgan_option.market.treasury_options import (
    cme_treasury_option_last_trading_datetime,
    underlying_quarterly_future,
)


TREASURY_URL_TEMPLATE = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
)
CME_RULE_URL = "https://www.cmegroup.com/rulebook/CBOT/II/19A.pdf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "wgan-option-thesis-reproducibility/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def build_treasury_curve(output_dir: Path, years: Iterable[int]) -> tuple[Path, list[dict]]:
    frames = []
    sources: list[dict] = []
    for year in years:
        url = TREASURY_URL_TEMPLATE.format(year=int(year))
        raw_path = output_dir / f"us_treasury_daily_par_yield_curve_{year}_official.csv"
        raw_path.write_bytes(_download(url))
        frame = pd.read_csv(raw_path)
        frame["source_year"] = int(year)
        frames.append(frame)
        sources.append(
            {
                "category": "treasury_curve",
                "url": url,
                "relative_path": raw_path.name,
                "sha256": _sha256(raw_path),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="raise")
    combined = (
        combined.sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .assign(Date=lambda frame: frame["Date"].dt.strftime("%m/%d/%Y"))
    )
    output_path = output_dir / "us_treasury_par_yield_curve_2022_2023.csv"
    combined.to_csv(output_path, index=False, lineterminator="\n")
    sources.append(
        {
            "category": "treasury_curve_combined",
            "url": "",
            "relative_path": output_path.name,
            "sha256": _sha256(output_path),
        }
    )
    return output_path, sources


def build_ty_expiries(output_dir: Path, years: Iterable[int]) -> tuple[Path, dict]:
    output_path = output_dir / "cme_ty_monthly_expirations_2022_2024.csv"
    rows = []
    for year in years:
        for month in range(1, 13):
            expiry = cme_treasury_option_last_trading_datetime(int(year), month)
            future_month, future_year = underlying_quarterly_future(int(year), month)
            rows.append(
                {
                    "named_option_year": int(year),
                    "named_option_month": month,
                    "last_trading_date": expiry.date().isoformat(),
                    "last_trading_time_chicago": "16:00:00",
                    "last_trading_datetime_utc": expiry.isoformat().replace("+00:00", "Z"),
                    "underlying_future_month_code": future_month,
                    "underlying_future_year": future_year,
                    "contract_class": "quarterly_or_serial",
                    "rule_reference": "CBOT Rule 19A01.I",
                }
            )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path, {
        "category": "cme_ty_expiry_rule_reference",
        "url": CME_RULE_URL,
        "relative_path": output_path.name,
        "sha256": _sha256(output_path),
        "notes": "Rule-derived quarterly/serial dates; weekly options are excluded.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, sources = build_treasury_curve(output_dir, years=(2022, 2023))
    _, expiry_source = build_ty_expiries(output_dir, years=(2022, 2023, 2024))
    sources.append(expiry_source)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/raw_vol/build_reference_data.py",
        "sources": sources,
    }
    manifest_path = output_dir / "rq123_market_reference_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
