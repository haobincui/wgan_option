"""Frozen U.S. Treasury par-yield curve loader for Black-76 discounting."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


_TENOR_COLUMNS: Dict[str, float] = {
    "1 Mo": 1.0 / 12.0,
    "1.5 Mo": 1.5 / 12.0,
    "2 Mo": 2.0 / 12.0,
    "3 Mo": 3.0 / 12.0,
    "4 Mo": 4.0 / 12.0,
    "6 Mo": 6.0 / 12.0,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CurvePoint:
    curve_date: date
    maturity_years: float
    par_yield_percent: float
    continuous_rate: float
    discount_factor: float
    source_sha256: str


class TreasuryParYieldCurve:
    """Read a frozen daily Treasury curve and interpolate by maturity.

    Treasury publishes par yields. For the short option maturities used here,
    the annual par yield is converted to a continuously compounded proxy via
    ``2 * log(1 + y / 2)`` before discounting. The approximation and source
    date are preserved in every pre-calibration audit row.
    """

    def __init__(self, path: str | Path, *, max_staleness_days: int = 7) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Frozen Treasury curve does not exist: {self.path}")
        self.max_staleness_days = int(max_staleness_days)
        if self.max_staleness_days < 0:
            raise ValueError("max_staleness_days must be >= 0")
        self.source_sha256 = _sha256(self.path)

        frame = pd.read_csv(self.path)
        if "Date" not in frame.columns:
            raise ValueError(f"Treasury curve is missing Date column: {self.path}")
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.date
        frame = frame[frame["Date"].notna()].copy()
        available = [column for column in _TENOR_COLUMNS if column in frame.columns]
        if len(available) < 2:
            raise ValueError(
                f"Treasury curve must provide at least two supported tenors: {self.path}"
            )
        for column in available:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.sort_values("Date").drop_duplicates("Date", keep="last")
        if frame.empty:
            raise ValueError(f"Treasury curve has no valid rows: {self.path}")

        self._frame = frame.set_index("Date")
        self._available_tenors = tuple(available)
        self._dates = tuple(self._frame.index)

    def _row_on_or_before(self, valuation_date: date) -> Tuple[date, pd.Series]:
        eligible = [curve_date for curve_date in self._dates if curve_date <= valuation_date]
        if not eligible:
            raise ValueError(
                f"No Treasury curve is available on or before {valuation_date.isoformat()}"
            )
        curve_date = eligible[-1]
        staleness = (valuation_date - curve_date).days
        if staleness > self.max_staleness_days:
            raise ValueError(
                "Treasury curve is too stale: "
                f"valuation={valuation_date.isoformat()}, curve={curve_date.isoformat()}, "
                f"staleness_days={staleness}, max={self.max_staleness_days}"
            )
        return curve_date, self._frame.loc[curve_date]

    @staticmethod
    def _valid_points(row: pd.Series, columns: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
        points = [
            (_TENOR_COLUMNS[column], float(row[column]))
            for column in columns
            if pd.notna(row[column]) and math.isfinite(float(row[column]))
        ]
        if len(points) < 2:
            raise ValueError("Treasury curve row has fewer than two finite tenors")
        points.sort(key=lambda item: item[0])
        return (
            np.asarray([item[0] for item in points], dtype=np.float64),
            np.asarray([item[1] for item in points], dtype=np.float64),
        )

    def point(self, valuation_date: date, maturity_years: float) -> CurvePoint:
        maturity_years = float(maturity_years)
        if not math.isfinite(maturity_years) or maturity_years <= 0:
            raise ValueError("maturity_years must be finite and positive")
        curve_date, row = self._row_on_or_before(valuation_date)
        tenors, yields = self._valid_points(row, self._available_tenors)
        clipped_maturity = float(np.clip(maturity_years, tenors[0], tenors[-1]))
        par_yield_percent = float(np.interp(clipped_maturity, tenors, yields))
        annual_rate = par_yield_percent / 100.0
        continuous_rate = 2.0 * math.log1p(annual_rate / 2.0)
        discount_factor = math.exp(-continuous_rate * maturity_years)
        return CurvePoint(
            curve_date=curve_date,
            maturity_years=maturity_years,
            par_yield_percent=par_yield_percent,
            continuous_rate=continuous_rate,
            discount_factor=discount_factor,
            source_sha256=self.source_sha256,
        )
