"""SABR-specific surface parameter construction."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

from quantlib.calendar.daycount import DayCountBusN
from quantlib.vol_surface.algo.sabr_algo import SabrCalibrationHagan


def build_sabr_surface_params(
    *,
    valuation_date: date,
    vols: List[List[float]],
    percent_strikes: List[List[float]],
    business_days_list: List[int],
    vol_daycount: DayCountBusN,
    stats: Dict[str, int],
) -> Dict[str, Any]:
    calibration = SabrCalibrationHagan(
        vols=vols,
        percent_strikes=percent_strikes,
        business_days=business_days_list,
        vol_daycount=vol_daycount,
        valuation_date=valuation_date,
        stats=stats,
    )
    return calibration.params
