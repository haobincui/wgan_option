"""Raw non-parametric surface parameter construction.

The raw-vol RQ workflow stores observed implied-vol points by maturity slice.
The reconstruction rule lives in :class:`quantlib.vol_surface.algo.raw_surface.RawVolSurface`:
linear interpolation over percent strike inside each slice and linear
interpolation of total variance across maturity.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

from quantlib.calendar.daycount import DayCountBusN
from quantlib.vol_surface.algo.raw_surface import RawVolSurface

RAW_SURFACE_PARAM_KEYS = ("business_days", "percent_strikes", "implied_vols")


def build_raw_surface_params(
    *,
    valuation_date: date,
    vols: List[List[float]],
    percent_strikes: List[List[float]],
    business_days_list: List[int],
    vol_daycount: DayCountBusN,
    stats: Dict[str, int],
) -> Dict[str, Any]:
    """Return model-aware raw surface params compatible with merge_vol.py."""

    del stats
    RawVolSurface(
        valuation_date=valuation_date,
        vols=vols,
        percent_strikes=percent_strikes,
        business_days=business_days_list,
        vol_daycount=vol_daycount,
    )
    return {
        "business_days": business_days_list,
        "percent_strikes": percent_strikes,
        "implied_vols": vols,
    }
