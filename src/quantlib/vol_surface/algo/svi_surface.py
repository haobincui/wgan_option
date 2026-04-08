from dataclasses import dataclass
from datetime import date
from typing import Dict, List

import numpy as np

from quantlib.calendar.daycount import DayCountBusN


@dataclass
class TermVolSurfaceByDays:
    valuation_date: date
    vols: List[float]
    business_days: List[int]
    vol_daycount: DayCountBusN

    def __post_init__(self):
        if len(self.vols) != len(self.business_days):
            raise ValueError("vols and business_days must have the same length")
        if len(self.vols) == 0:
            raise ValueError("vols cannot be empty")
        self._days = np.asarray(self.business_days, dtype=np.float64)
        self._vols = np.asarray(self.vols, dtype=np.float64)
        self._days_in_year = float(self.vol_daycount.days_in_year)
        self._vars = self._vols * self._vols * self._days / self._days_in_year

    def _interp_variance(self, day: float) -> float:
        x = self._days
        y = self._vars
        if day <= x[0]:
            return float(y[0])
        if day >= x[-1]:
            if len(x) < 2:
                return float(y[-1])
            slope = (y[-1] - y[-2]) / max(x[-1] - x[-2], 1.0)
            return float(y[-1] + slope * (day - x[-1]))
        idx = int(np.searchsorted(x, day))
        x0, x1 = x[idx - 1], x[idx]
        y0, y1 = y[idx - 1], y[idx]
        w = (day - x0) / (x1 - x0)
        return float(y0 + w * (y1 - y0))

    def implied_vol(self, expiration_date: date) -> float:
        day = self.vol_daycount.calendar.count_business_days(
            self.valuation_date,
            expiration_date,
            include_start=False,
            include_end=True,
        )
        day = max(1.0, float(day))
        variance = max(self._interp_variance(day), 0.0)
        return float(np.sqrt(variance * self._days_in_year / day))


@dataclass
class SviVolSurface:
    valuation_date: date
    svi_params: Dict[str, List[float]]
    vol_daycount: DayCountBusN

    def __post_init__(self):
        required = {"a", "b", "rho", "m", "sigma", "business_days"}
        missing = sorted(required - set(self.svi_params.keys()))
        if missing:
            raise ValueError(f"svi_params missing keys: {missing}")

        self._days = np.asarray(self.svi_params["business_days"], dtype=np.float64)
        self._a = np.asarray(self.svi_params["a"], dtype=np.float64)
        self._b = np.asarray(self.svi_params["b"], dtype=np.float64)
        self._rho = np.asarray(self.svi_params["rho"], dtype=np.float64)
        self._m = np.asarray(self.svi_params["m"], dtype=np.float64)
        self._sigma = np.asarray(self.svi_params["sigma"], dtype=np.float64)
        self._days_in_year = float(self.vol_daycount.days_in_year)

    def _interp_param(self, arr: np.ndarray, day: float) -> float:
        if day <= self._days[0]:
            return float(arr[0])
        if day >= self._days[-1]:
            return float(arr[-1])
        idx = int(np.searchsorted(self._days, day))
        x0, x1 = self._days[idx - 1], self._days[idx]
        y0, y1 = arr[idx - 1], arr[idx]
        w = (day - x0) / (x1 - x0)
        return float(y0 + w * (y1 - y0))

    @staticmethod
    def _slice_total_variance(
        log_moneyness: float, a: float, b: float, rho: float, m: float, sigma: float
    ) -> float:
        core = rho * (log_moneyness - m) + np.sqrt((log_moneyness - m) ** 2 + sigma * sigma)
        return float(max(a + b * core, 0.0))

    def implied_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        day = self.vol_daycount.calendar.count_business_days(
            self.valuation_date,
            expiration_date,
            include_start=False,
            include_end=True,
        )
        day = max(1.0, float(day))
        lm = float(np.log(max(strike / forward, 1e-12)))
        a = self._interp_param(self._a, day)
        b = self._interp_param(self._b, day)
        rho = self._interp_param(self._rho, day)
        m = self._interp_param(self._m, day)
        sigma = max(self._interp_param(self._sigma, day), 1e-8)
        total_var = self._slice_total_variance(lm, a, b, rho, m, sigma)
        return float(np.sqrt(total_var * self._days_in_year / day))

    def implied_vol_surface(
        self, percent_strikes: List[float], business_days: List[int], forward: float = 1.0
    ) -> List[List[float]]:
        vols: List[List[float]] = []
        for day in business_days:
            row = []
            exp = self.valuation_date
            for _ in range(int(day)):
                exp = self.vol_daycount.calendar.next(exp)
            for p in percent_strikes:
                row.append(self.implied_vol(forward=forward, strike=p * forward, expiration_date=exp))
            vols.append(row)
        return vols
