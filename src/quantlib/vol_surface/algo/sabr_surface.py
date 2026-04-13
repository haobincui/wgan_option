from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List

import numpy as np

from quantlib.calendar.daycount import DayCountBusN
from quantlib.vol_surface.surface import ImpliedVolSurface


_UNSUPPORTED_SURFACE_OPERATION_MESSAGE = (
    "This model surface currently supports implied-vol queries only; "
    "parallel_bump/roll/shift_valuation_date are not implemented."
)


def hagan_lognormal_implied_vol(
    forward: float | np.ndarray,
    strike: float | np.ndarray,
    maturity: float | np.ndarray,
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
    rho: float | np.ndarray,
    nu: float | np.ndarray,
) -> np.ndarray:
    forward_arr = np.maximum(np.asarray(forward, dtype=np.float64), 1e-12)
    strike_arr = np.maximum(np.asarray(strike, dtype=np.float64), 1e-12)
    maturity_arr = np.maximum(np.asarray(maturity, dtype=np.float64), 1e-12)
    alpha_arr = np.maximum(np.asarray(alpha, dtype=np.float64), 1e-12)
    beta_arr = np.asarray(beta, dtype=np.float64)
    rho_arr = np.clip(np.asarray(rho, dtype=np.float64), -0.999, 0.999)
    nu_arr = np.maximum(np.asarray(nu, dtype=np.float64), 0.0)

    one_minus_beta = 1.0 - beta_arr
    log_fk = np.log(forward_arr / strike_arr)
    log_fk_sq = log_fk * log_fk
    log_fk_four = log_fk_sq * log_fk_sq
    fk = forward_arr * strike_arr
    fk_beta = np.power(fk, 0.5 * one_minus_beta)

    z = (nu_arr / alpha_arr) * fk_beta * log_fk
    sqrt_term = np.sqrt(np.maximum(1.0 - 2.0 * rho_arr * z + z * z, 1e-12))
    x_z = np.log((sqrt_term + z - rho_arr) / np.maximum(1.0 - rho_arr, 1e-12))
    z_over_x = np.where(np.abs(z) < 1e-10, 1.0, z / np.where(np.abs(x_z) < 1e-12, 1.0, x_z))

    base_denom = fk_beta * (
        1.0
        + (one_minus_beta * one_minus_beta / 24.0) * log_fk_sq
        + (one_minus_beta ** 4 / 1920.0) * log_fk_four
    )
    time_adjustment = 1.0 + (
        (one_minus_beta * one_minus_beta / 24.0) * (alpha_arr * alpha_arr) / np.maximum(fk ** one_minus_beta, 1e-12)
        + 0.25 * rho_arr * beta_arr * nu_arr * alpha_arr / np.maximum(fk_beta, 1e-12)
        + ((2.0 - 3.0 * rho_arr * rho_arr) / 24.0) * (nu_arr * nu_arr)
    ) * maturity_arr
    non_atm = alpha_arr / np.maximum(base_denom, 1e-12) * z_over_x * time_adjustment

    forward_beta = np.power(forward_arr, one_minus_beta)
    atm_adjustment = 1.0 + (
        (one_minus_beta * one_minus_beta / 24.0) * (alpha_arr * alpha_arr) / np.maximum(forward_beta * forward_beta, 1e-12)
        + 0.25 * rho_arr * beta_arr * nu_arr * alpha_arr / np.maximum(forward_beta, 1e-12)
        + ((2.0 - 3.0 * rho_arr * rho_arr) / 24.0) * (nu_arr * nu_arr)
    ) * maturity_arr
    atm = alpha_arr / np.maximum(forward_beta, 1e-12) * atm_adjustment

    implied_vol = np.where(np.abs(log_fk) < 1e-10, atm, non_atm)
    return np.maximum(implied_vol, 0.0).astype(np.float64)


@dataclass
class SabrVolSurface(ImpliedVolSurface):
    sabr_params: Dict[str, List[float]]
    vol_daycount: DayCountBusN

    def __post_init__(self):
        required = {"alpha", "beta", "rho", "nu", "business_days"}
        missing = sorted(required - set(self.sabr_params.keys()))
        if missing:
            raise ValueError(f"sabr_params missing keys: {missing}")

        self._days = np.asarray(self.sabr_params["business_days"], dtype=np.float64)
        self._alpha = np.asarray(self.sabr_params["alpha"], dtype=np.float64)
        self._beta = np.asarray(self.sabr_params["beta"], dtype=np.float64)
        self._rho = np.asarray(self.sabr_params["rho"], dtype=np.float64)
        self._nu = np.asarray(self.sabr_params["nu"], dtype=np.float64)
        self._days_in_year = float(self.vol_daycount.days_in_year)

        if self._days.size == 0:
            raise ValueError("sabr_params cannot be empty")
        if not (
            self._alpha.size == self._beta.size == self._rho.size == self._nu.size == self._days.size
        ):
            raise ValueError("All SABR parameter arrays must have the same length")
        if np.any(~np.isfinite(self._days)) or np.any(self._days <= 0):
            raise ValueError("business_days must be strictly positive and finite")
        if np.any(np.diff(self._days) <= 0):
            raise ValueError("business_days must be strictly increasing")

    def _interp_param(self, arr: np.ndarray, day: float) -> float:
        if day <= self._days[0]:
            return float(arr[0])
        if day >= self._days[-1]:
            return float(arr[-1])
        idx = int(np.searchsorted(self._days, day))
        x0, x1 = self._days[idx - 1], self._days[idx]
        y0, y1 = arr[idx - 1], arr[idx]
        weight = (day - x0) / (x1 - x0)
        return float(y0 + weight * (y1 - y0))

    def _implied_vol_by_percent_strike_and_day(self, percent_strike: float, day: float) -> float:
        day = max(float(day), 1.0)
        tau = day / self._days_in_year
        alpha = max(self._interp_param(self._alpha, day), 1e-12)
        beta = float(np.clip(self._interp_param(self._beta, day), 0.0, 1.0))
        rho = float(np.clip(self._interp_param(self._rho, day), -0.999, 0.999))
        nu = max(self._interp_param(self._nu, day), 0.0)
        implied_vol = hagan_lognormal_implied_vol(
            forward=1.0,
            strike=max(float(percent_strike), 1e-12),
            maturity=tau,
            alpha=alpha,
            beta=beta,
            rho=rho,
            nu=nu,
        )
        return float(np.asarray(implied_vol, dtype=np.float64))

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
        self, percent_strikes: List[float], business_days: List[int], forward: float = 1.0
    ) -> List[List[float]]:
        del forward
        vols: List[List[float]] = []
        for business_day in business_days:
            row = [
                self._implied_vol_by_percent_strike_and_day(
                    percent_strike=float(percent_strike),
                    day=float(business_day),
                )
                for percent_strike in percent_strikes
            ]
            vols.append(row)
        return vols
