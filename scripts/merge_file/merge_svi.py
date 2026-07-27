"""Thin CLI wrapper — build an SVI audit workbook from minute-SVI results."""

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

from wgan_option.merge.merge_svi_core import (  # noqa: E402
    DAYS_IN_YEAR,
    DEFAULT_OUTPUT_NAME,
    build_svi_workbook_frames,
    build_svi_workbook_frames as build_workbook_frames,
)
from wgan_option.merge_support import (  # noqa: E402
    DEFAULT_NEWS_XLSX_PATH,
    DEFAULT_OFFSET_MINUTES,
    DEFAULT_SOURCE_TIMEZONE,
    write_workbook,
)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge minute SVI results into an audit workbook.")
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing surface-<model>-<data_range>.json and surface-<model>-<data_range>-precalib-points.csv.",
    )
    parser.add_argument("--news-xlsx", default=str(DEFAULT_NEWS_XLSX_PATH), help="Path to the news embedding workbook.")
    parser.add_argument(
        "--source-timezone",
        default=None,
        help=(
            "Timezone for PD + ET. Defaults to the surface resolved config, "
            f"then {DEFAULT_SOURCE_TIMEZONE} for legacy inputs."
        ),
    )
    parser.add_argument("--offset-minutes", type=int, default=DEFAULT_OFFSET_MINUTES, help="Forward direction offset in minutes.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> Path:
    args = _parse_args(argv)
    input_dir = Path(args.input_dir).expanduser()
    workbook_frames = build_svi_workbook_frames(
        input_dir,
        news_xlsx_path=Path(args.news_xlsx).expanduser(),
        source_timezone=args.source_timezone,
        offset_minutes=int(args.offset_minutes),
    )
    output_path = input_dir / DEFAULT_OUTPUT_NAME
    write_workbook(output_path, workbook_frames)
    print(f"Merged workbook written to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
