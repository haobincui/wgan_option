"""Thin CLI wrapper — build a paired vol-surface audit workbook from minute-SVI results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from wgan_option.merge.merge_vol_core import (  # noqa: E402
    DAYS_IN_YEAR,
    DEFAULT_OUTPUT_NAME,
    build_vol_workbook_frames,
    build_vol_workbook_frames as build_workbook_frames,
)
from wgan_option.merge_support import (  # noqa: E402
    DEFAULT_NEWS_XLSX_PATH,
    DEFAULT_OFFSET_MINUTES,
    DEFAULT_SOURCE_TIMEZONE,
    write_workbook,
)
from wgan_option.surface_grid import (  # noqa: E402
    DEFAULT_MATURITY_BINS,
    DEFAULT_MATURITY_MAX_DAYS,
    DEFAULT_MATURITY_MIN_DAYS,
    DEFAULT_MONEYNESS_MAX,
    DEFAULT_MONEYNESS_MIN,
    DEFAULT_STRIKE_BINS,
)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge minute surface results into a paired vol-surface workbook.")
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing surface-<model>-<data_range>.json and surface-<model>-<data_range>-precalib-points.csv.",
    )
    parser.add_argument("--news-xlsx", default=str(DEFAULT_NEWS_XLSX_PATH), help="Path to the news embedding workbook.")
    parser.add_argument("--source-timezone", default=DEFAULT_SOURCE_TIMEZONE, help="Timezone for PD + ET in news xlsx.")
    parser.add_argument("--offset-minutes", type=int, default=DEFAULT_OFFSET_MINUTES, help="Forward direction offset in minutes.")
    parser.add_argument("--strike-bins", type=int, default=DEFAULT_STRIKE_BINS, help="Number of moneyness bins.")
    parser.add_argument("--maturity-bins", type=int, default=DEFAULT_MATURITY_BINS, help="Number of maturity bins.")
    parser.add_argument("--moneyness-min", type=float, default=DEFAULT_MONEYNESS_MIN, help="Minimum percent-strike grid value.")
    parser.add_argument("--moneyness-max", type=float, default=DEFAULT_MONEYNESS_MAX, help="Maximum percent-strike grid value.")
    parser.add_argument("--maturity-min-days", type=int, default=DEFAULT_MATURITY_MIN_DAYS, help="Minimum maturity in business days.")
    parser.add_argument("--maturity-max-days", type=int, default=DEFAULT_MATURITY_MAX_DAYS, help="Maximum maturity in business days.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> Path:
    args = _parse_args(argv)
    input_dir = Path(args.input_dir).expanduser()
    workbook_frames = build_vol_workbook_frames(
        input_dir,
        news_xlsx_path=Path(args.news_xlsx).expanduser(),
        source_timezone=str(args.source_timezone),
        offset_minutes=int(args.offset_minutes),
        strike_bins=int(args.strike_bins),
        maturity_bins=int(args.maturity_bins),
        moneyness_min=float(args.moneyness_min),
        moneyness_max=float(args.moneyness_max),
        maturity_min_days=int(args.maturity_min_days),
        maturity_max_days=int(args.maturity_max_days),
    )
    output_path = input_dir / DEFAULT_OUTPUT_NAME
    write_workbook(output_path, workbook_frames)
    print(f"Merged workbook written to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
