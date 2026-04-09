from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple
import logging

import numpy as np
from scipy import optimize

from quantlib.calendar.daycount import DayCount
from .sabr_surface import SabrVolSurface, hagan_lognormal_implied_vol

logger = logging.getLogger(__name__)

_MIN_ALPHA = 1e-6
_MAX_ALPHA = 10.0
_MIN_RHO = -0.999
_MAX_RHO = 0.999
_MIN_NU = 0.0
_MAX_NU = 10.0
_DEFAULT_BETA = 1.0
_MAX_NFEV = 2000


def _increment_stat(stats: Optional[Dict[str, int]], key: str) -> None:
    if stats is None:
        return
    stats[key] = int(stats.get(key, 0)) + 1


def _prepare_slice_inputs(
    percent_strikes: Sequence[float],
    vols: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    pairs: Dict[float, List[float]] = {}
    for percent_strike, vol in zip(percent_strikes, vols):
        strike_value = float(percent_strike)
        vol_value = float(vol)
        if not np.isfinite(strike_value) or not np.isfinite(vol_value):
            continue
        if strike_value <= 0 or vol_value <= 0:
            continue
        pairs.setdefault(strike_value, []).append(vol_value)

    if not pairs:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    ordered = sorted(pairs.items(), key=lambda item: item[0])
    strikes = np.asarray([item[0] for item in ordered], dtype=np.float64)
    slice_vols = np.asarray([np.mean(item[1]) for item in ordered], dtype=np.float64)
    return strikes, slice_vols


def _slice_rmse(target: np.ndarray, model: np.ndarray) -> float:
    diff = np.asarray(target, dtype=np.float64) - np.asarray(model, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(diff))))


def _alpha_upper_bound(target_vols: np.ndarray) -> float:
    return float(max(_MAX_ALPHA, 5.0 * max(float(np.max(target_vols)), float(np.mean(target_vols)), 0.1)))


def _seed_alphas(target_vols: np.ndarray) -> List[float]:
    atm_idx = int(np.abs(target_vols - np.median(target_vols)).argmin())
    atm_vol = float(target_vols[atm_idx])
    seeds = [atm_vol, float(np.mean(target_vols)), float(np.max(target_vols)), max(float(np.min(target_vols)), 0.05)]
    deduped: List[float] = []
    for seed in seeds:
        clipped = float(np.clip(seed, _MIN_ALPHA, _MAX_ALPHA))
        if any(np.isclose(clipped, existing, atol=1e-12, rtol=0.0) for existing in deduped):
            continue
        deduped.append(clipped)
    return deduped


@dataclass(frozen=True)
class _SabrSliceCalibrationResult:
    params: List[float]
    rmse: float
    success: bool


def _calibrate_sabr_slice(
    percent_strikes: np.ndarray,
    target_vols: np.ndarray,
    maturity: float,
    fixed_beta: Optional[float],
) -> _SabrSliceCalibrationResult:
    if percent_strikes.size == 0 or target_vols.size == 0 or maturity <= 0:
        return _SabrSliceCalibrationResult(params=[0.0, _DEFAULT_BETA, 0.0, 0.0], rmse=float("inf"), success=False)

    alpha_upper = _alpha_upper_bound(target_vols)
    rho_seeds = [-0.25, 0.0, 0.25]
    nu_seeds = [0.1, 0.5, 1.0]
    alpha_seeds = _seed_alphas(target_vols)

    if fixed_beta is not None:
        beta_value = float(np.clip(fixed_beta, 0.0, 1.0))

        def unpack(raw_params: np.ndarray) -> Tuple[float, float, float, float]:
            alpha, rho, nu = [float(x) for x in raw_params]
            return alpha, beta_value, rho, nu

        lower_bounds = np.asarray([_MIN_ALPHA, _MIN_RHO, _MIN_NU], dtype=np.float64)
        upper_bounds = np.asarray([alpha_upper, _MAX_RHO, _MAX_NU], dtype=np.float64)
        initial_guesses = [
            np.asarray([alpha_seed, rho_seed, nu_seed], dtype=np.float64)
            for alpha_seed in alpha_seeds
            for rho_seed in rho_seeds
            for nu_seed in nu_seeds
        ]
    else:
        beta_seeds = [0.2, 0.5, 0.8, 1.0]

        def unpack(raw_params: np.ndarray) -> Tuple[float, float, float, float]:
            alpha, beta, rho, nu = [float(x) for x in raw_params]
            return alpha, beta, rho, nu

        lower_bounds = np.asarray([_MIN_ALPHA, 0.0, _MIN_RHO, _MIN_NU], dtype=np.float64)
        upper_bounds = np.asarray([alpha_upper, 1.0, _MAX_RHO, _MAX_NU], dtype=np.float64)
        initial_guesses = [
            np.asarray([alpha_seed, beta_seed, rho_seed, nu_seed], dtype=np.float64)
            for alpha_seed in alpha_seeds
            for beta_seed in beta_seeds
            for rho_seed in rho_seeds
            for nu_seed in nu_seeds
        ]

    best_result: Optional[_SabrSliceCalibrationResult] = None
    for initial_guess in initial_guesses:

        def residuals(raw_params: np.ndarray) -> np.ndarray:
            alpha, beta, rho, nu = unpack(raw_params)
            model_vols = hagan_lognormal_implied_vol(
                forward=1.0,
                strike=percent_strikes,
                maturity=maturity,
                alpha=alpha,
                beta=beta,
                rho=rho,
                nu=nu,
            )
            return np.asarray(model_vols, dtype=np.float64) - target_vols

        try:
            result = optimize.least_squares(
                residuals,
                initial_guess,
                bounds=(lower_bounds, upper_bounds),
                method="trf",
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=_MAX_NFEV,
            )
        except Exception:
            logger.debug("SABR slice calibration failed for initial_guess=%s", initial_guess, exc_info=True)
            continue

        alpha, beta, rho, nu = unpack(np.asarray(result.x, dtype=np.float64))
        model_vols = hagan_lognormal_implied_vol(
            forward=1.0,
            strike=percent_strikes,
            maturity=maturity,
            alpha=alpha,
            beta=beta,
            rho=rho,
            nu=nu,
        )
        rmse = _slice_rmse(target=target_vols, model=np.asarray(model_vols, dtype=np.float64))
        candidate = _SabrSliceCalibrationResult(
            params=[alpha, beta, rho, nu],
            rmse=rmse,
            success=bool(np.isfinite(rmse)),
        )
        if best_result is None or candidate.rmse < best_result.rmse - 1e-12:
            best_result = candidate

    if best_result is None:
        return _SabrSliceCalibrationResult(params=[0.0, _DEFAULT_BETA, 0.0, 0.0], rmse=float("inf"), success=False)
    return best_result


@dataclass
class SabrCalibration(ABC):
    valuation_date: date

    @abstractmethod
    def get_calibrated_vol_surface(self) -> SabrVolSurface:
        pass

    @abstractmethod
    def get_calibrated_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        pass


@dataclass
class SabrCalibrationHagan(SabrCalibration):
    vols: List[List[float]]
    percent_strikes: List[List[float]]
    business_days: List[int]
    vol_daycount: DayCount
    beta: Optional[float] = None
    stats: Optional[Dict[str, int]] = None
    days_in_year: int = field(init=False)
    params: Dict[str, List[float]] = field(init=False)

    def __post_init__(self):
        try:
            self.days_in_year = self.vol_daycount.days_in_year
        except Exception:
            self.days_in_year = 365

        cleaned_slices = sorted(
            zip(self.business_days, self.percent_strikes, self.vols),
            key=lambda item: item[0],
        )

        alpha_values: List[float] = []
        beta_values: List[float] = []
        rho_values: List[float] = []
        nu_values: List[float] = []
        day_values: List[int] = []

        for business_day, percent_strike_slice, vol_slice in cleaned_slices:
            _increment_stat(self.stats, "sabr_total_slices")
            if int(business_day) <= 0:
                _increment_stat(self.stats, "sabr_failed_slices")
                continue

            strikes_arr, vols_arr = _prepare_slice_inputs(percent_strikes=percent_strike_slice, vols=vol_slice)
            if strikes_arr.size == 0:
                _increment_stat(self.stats, "sabr_failed_slices")
                continue

            result = _calibrate_sabr_slice(
                percent_strikes=strikes_arr,
                target_vols=vols_arr,
                maturity=float(business_day) / float(self.days_in_year),
                fixed_beta=self.beta,
            )
            if not result.success:
                _increment_stat(self.stats, "sabr_failed_slices")
                continue

            alpha, beta_value, rho, nu = result.params
            alpha_values.append(float(alpha))
            beta_values.append(float(beta_value))
            rho_values.append(float(rho))
            nu_values.append(float(nu))
            day_values.append(int(business_day))
            _increment_stat(self.stats, "sabr_calibrated_slices")
            if self.beta is None:
                _increment_stat(self.stats, "sabr_free_beta_slices")
            else:
                _increment_stat(self.stats, "sabr_fixed_beta_slices")

        if not day_values:
            _increment_stat(self.stats, "sabr_placeholder_surface")
            fallback_day = int(min(self.business_days)) if self.business_days else 1
            fallback_idx = int(np.argmin(np.asarray(self.business_days, dtype=np.float64))) if self.business_days else 0
            fallback_slice = self.vols[fallback_idx] if self.vols else [0.2]
            fallback_vols = [float(vol) for vol in fallback_slice if np.isfinite(vol) and float(vol) > 0]
            placeholder_alpha = float(np.mean(fallback_vols)) if fallback_vols else 0.2
            alpha_values = [placeholder_alpha]
            beta_values = [float(self.beta) if self.beta is not None else _DEFAULT_BETA]
            rho_values = [0.0]
            nu_values = [0.0]
            day_values = [max(fallback_day, 1)]

        self.params = {
            "alpha": alpha_values,
            "beta": beta_values,
            "rho": rho_values,
            "nu": nu_values,
            "business_days": day_values,
        }

    def get_calibrated_vol_surface(self) -> SabrVolSurface:
        return SabrVolSurface(
            valuation_date=self.valuation_date,
            sabr_params=self.params,
            vol_daycount=self.vol_daycount,
        )

    def get_calibrated_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        return self.get_calibrated_vol_surface().implied_vol(
            forward=forward,
            strike=strike,
            expiration_date=expiration_date,
        )
