from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List

from quantlib.calendar.daycount import DayCount

from .cubic_spline_surface import CubicSplineVolSurface


@dataclass
class CubicSplineVolSurfaceBuilder:
    valuation_date: date
    vols: List[List[float]]
    percent_strikes: List[List[float]]
    business_days: List[int]
    vol_daycount: DayCount
    surface: CubicSplineVolSurface = field(init=False)

    def __post_init__(self):
        self.surface = CubicSplineVolSurface(
            valuation_date=self.valuation_date,
            vols=self.vols,
            percent_strikes=self.percent_strikes,
            business_days=self.business_days,
            vol_daycount=self.vol_daycount,
        )

    def get_vol_surface(self) -> CubicSplineVolSurface:
        return self.surface

    def get_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        return self.surface.implied_vol(
            forward=forward,
            strike=strike,
            expiration_date=expiration_date,
        )
