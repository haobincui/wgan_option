"""Argument parser for standalone surface generation script."""

import argparse
from typing import Optional, Sequence


class GenerateSurfaceArgParser:
    """Thin wrapper to keep parser construction reusable and testable."""

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Generate daily volatility-like surfaces from raw option data."
        )
        parser.add_argument("--input-glob", type=str, default="data/raw/option_data/*.csv.gz")
        parser.add_argument("--output-dir", type=str, default="outputs/vol_surface")
        parser.add_argument("--strike-bins", type=int, default=16)
        parser.add_argument("--maturity-bins", type=int, default=16)
        parser.add_argument("--moneyness-min", type=float, default=0.7)
        parser.add_argument("--moneyness-max", type=float, default=1.3)
        parser.add_argument("--maturity-min-days", type=int, default=7)
        parser.add_argument("--maturity-max-days", type=int, default=365)
        parser.add_argument(
            "--no-daily-csv",
            action="store_true",
            help="If set, only save compressed NPZ and skip per-day CSV files.",
        )
        self._parser = parser

    def parse_args(self, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
        """Parse argv and return typed argparse namespace."""
        return self._parser.parse_args(argv)
