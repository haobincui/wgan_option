"""Model-specific surface-parameter builders."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

from quantlib.calendar.daycount import DayCountBusN

from .cubic import build_cubic_surface_params
from .raw import build_raw_surface_params
from .sabr import build_sabr_surface_params
from .svi import build_svi_surface_params


def build_surface_params(
    *,
    surface_model: str,
    valuation_date: date,
    vols: List[List[float]],
    percent_strikes: List[List[float]],
    business_days_list: List[int],
    vol_daycount: DayCountBusN,
    stats: Dict[str, int],
) -> Dict[str, Any]:
    if surface_model == "svi":
        return build_svi_surface_params(
            valuation_date=valuation_date,
            vols=vols,
            percent_strikes=percent_strikes,
            business_days_list=business_days_list,
            vol_daycount=vol_daycount,
            stats=stats,
        )
    if surface_model == "sabr":
        return build_sabr_surface_params(
            valuation_date=valuation_date,
            vols=vols,
            percent_strikes=percent_strikes,
            business_days_list=business_days_list,
            vol_daycount=vol_daycount,
            stats=stats,
        )
    if surface_model == "cubic":
        return build_cubic_surface_params(
            valuation_date=valuation_date,
            vols=vols,
            percent_strikes=percent_strikes,
            business_days_list=business_days_list,
            vol_daycount=vol_daycount,
            stats=stats,
        )
    if surface_model == "raw":
        return build_raw_surface_params(
            valuation_date=valuation_date,
            vols=vols,
            percent_strikes=percent_strikes,
            business_days_list=business_days_list,
            vol_daycount=vol_daycount,
            stats=stats,
        )
    raise ValueError(f"Unsupported surface model: {surface_model}")
