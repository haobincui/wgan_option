"""Cubic-spline-specific surface parameter construction."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

from quantlib.calendar.daycount import DayCountBusN
from quantlib.vol_surface.algo.cubic_spline_algo import CubicSplineVolSurfaceBuilder


def build_cubic_surface_params(
    *,
    valuation_date: date,
    vols: List[List[float]],
    percent_strikes: List[List[float]],
    business_days_list: List[int],
    vol_daycount: DayCountBusN,
    stats: Dict[str, int],
) -> Dict[str, Any]:
    del stats
    builder = CubicSplineVolSurfaceBuilder(
        valuation_date=valuation_date,
        vols=vols,
        percent_strikes=percent_strikes,
        business_days=business_days_list,
        vol_daycount=vol_daycount,
    )
    builder.get_vol_surface()
    return {
        "business_days": business_days_list,
        "percent_strikes": percent_strikes,
        "implied_vols": vols,
    }
