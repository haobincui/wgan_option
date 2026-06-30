from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Sequence

import numpy as np

from quantlib.calendar.daycount import DayCountBusN
from quantlib.vol_surface.surface import ImpliedVolSurface


_UNSUPPORTED_SURFACE_OPERATION_MESSAGE = (
    "This model surface currently supports implied-vol queries only; "
    "parallel_bump/roll/shift_valuation_date are not implemented."
)


def _sorted_unique_slice(
    percent_strikes: Sequence[float],
    vols: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[float, List[float]] = {}
    for percent_strike, vol in zip(percent_strikes, vols):
        strike_value = float(percent_strike)
        vol_value = float(vol)
        if not np.isfinite(strike_value) or not np.isfinite(vol_value):
            continue
        if strike_value <= 0 or vol_value <= 0:
            continue
        grouped.setdefault(strike_value, []).append(vol_value)

    if not grouped:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    ordered = sorted(grouped.items(), key=lambda item: item[0])
    strikes = np.asarray([item[0] for item in ordered], dtype=np.float64)
    vol_values = np.asarray([np.mean(item[1]) for item in ordered], dtype=np.float64)
    return strikes, vol_values


@dataclass
class RawVolSurface(ImpliedVolSurface):
    """Non-parametric IV surface reconstructed from raw implied-vol points.

    For each maturity slice, duplicate percent-strike observations are averaged
    and queried values are linearly interpolated in percent-strike space with
    boundary clamping. Across maturities, the interpolation is linear in total
    variance, then converted back to implied volatility.
    """

    vols: List[List[float]]
    percent_strikes: List[List[float]]
    business_days: List[int]
    vol_daycount: DayCountBusN

    def __post_init__(self):
        if not (len(self.vols) == len(self.percent_strikes) == len(self.business_days)):
            raise ValueError("vols, percent_strikes, and business_days must have the same length.")

        self._days_in_year = float(self.vol_daycount.days_in_year)
        cleaned_slices = sorted(
            zip(self.business_days, self.percent_strikes, self.vols),
            key=lambda item: item[0],
        )

        self._business_days: List[int] = []
        self._slice_bounds: List[tuple[float, float]] = []
        self._slice_strikes: List[np.ndarray] = []
        self._slice_total_variances: List[np.ndarray] = []
        for business_day, percent_strike_slice, vol_slice in cleaned_slices:
            if int(business_day) <= 0:
                continue
            strikes_arr, vols_arr = _sorted_unique_slice(percent_strikes=percent_strike_slice, vols=vol_slice)
            if strikes_arr.size == 0:
                continue
            self._business_days.append(int(business_day))
            self._slice_bounds.append((float(strikes_arr[0]), float(strikes_arr[-1])))
            self._slice_strikes.append(strikes_arr)
            self._slice_total_variances.append(
                np.square(vols_arr) * float(business_day) / self._days_in_year
            )

        if not self._business_days:
            raise ValueError("RawVolSurface requires at least one valid slice.")

        self._days = np.asarray(self._business_days, dtype=np.float64)

    def _slice_total_variance(self, slice_idx: int, percent_strike: float) -> float:
        lower_bound, upper_bound = self._slice_bounds[slice_idx]
        clamped_percent = float(np.clip(percent_strike, lower_bound, upper_bound))
        total_variance = np.interp(
            clamped_percent,
            self._slice_strikes[slice_idx],
            self._slice_total_variances[slice_idx],
            left=float(self._slice_total_variances[slice_idx][0]),
            right=float(self._slice_total_variances[slice_idx][-1]),
        )
        return float(max(float(total_variance), 0.0))

    def _implied_vol_by_percent_strike_and_day(self, percent_strike: float, day: float) -> float:
        day = max(float(day), 1.0)
        percent_strike = max(float(percent_strike), 1e-12)

        if len(self._business_days) == 1:
            return float(np.sqrt(self._slice_total_variance(0, percent_strike) * self._days_in_year / day))

        if day <= self._days[0]:
            return float(np.sqrt(self._slice_total_variance(0, percent_strike) * self._days_in_year / day))
        if day >= self._days[-1]:
            return float(
                np.sqrt(
                    self._slice_total_variance(len(self._business_days) - 1, percent_strike)
                    * self._days_in_year
                    / day
                )
            )

        idx = int(np.searchsorted(self._days, day))
        lower_idx = idx - 1
        upper_idx = idx
        lower_day = self._days[lower_idx]
        upper_day = self._days[upper_idx]
        lower_var = self._slice_total_variance(lower_idx, percent_strike)
        upper_var = self._slice_total_variance(upper_idx, percent_strike)
        weight = (day - lower_day) / (upper_day - lower_day)
        interpolated_var = lower_var + (upper_var - lower_var) * weight
        return float(np.sqrt(max(interpolated_var, 0.0) * self._days_in_year / day))

    def implied_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        day = self.vol_daycount.calendar.count_business_days(
            self.valuation_date,
            expiration_date,
            include_start=False,
            include_end=True,
        )
        percent_strike = max(float(strike) / max(float(forward), 1e-12), 1e-12)
        return self._implied_vol_by_percent_strike_and_day(percent_strike=percent_strike, day=day)

    def implied_vol_by_spot(self, spot: float, strike: float, expiration_date: date) -> float:
        return self.implied_vol(
            forward=spot,
            strike=strike,
            expiration_date=expiration_date,
        )

    def parallel_bump(self, amount: float) -> ImpliedVolSurface:
        del amount
        raise NotImplementedError(_UNSUPPORTED_SURFACE_OPERATION_MESSAGE)

    def roll(self, new_valuation_date: date) -> ImpliedVolSurface:
        del new_valuation_date
        raise NotImplementedError(_UNSUPPORTED_SURFACE_OPERATION_MESSAGE)

    def shift_valuation_date(self, new_valuation_date: date) -> ImpliedVolSurface:
        del new_valuation_date
        raise NotImplementedError(_UNSUPPORTED_SURFACE_OPERATION_MESSAGE)

    def implied_vol_surface(
        self,
        percent_strikes: Sequence[float],
        business_days: Sequence[int],
        forward: float = 1.0,
    ) -> List[List[float]]:
        del forward
        return [
            [
                self._implied_vol_by_percent_strike_and_day(
                    percent_strike=float(percent_strike),
                    day=float(business_day),
                )
                for percent_strike in percent_strikes
            ]
            for business_day in business_days
        ]
