import logging
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import norm

from quantlib.calendar.daycount import DayCountBusN

logger = logging.getLogger(__name__)


def _as_1d_float_array(values: Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"{name} must have at least 3 points (got {arr.size}).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _broadcast_term_vector(
    value: Union[float, Sequence[float]],
    n_terms: int,
    *,
    name: str,
) -> np.ndarray:
    if isinstance(value, (int, float, np.floating, np.integer)):
        return np.full((n_terms, 1), float(value), dtype=np.float64)
    arr = np.asarray(list(value), dtype=np.float64).reshape(-1)
    if arr.size != n_terms:
        raise ValueError(f"{name} must be scalar or length {n_terms} (got {arr.size}).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr.reshape(n_terms, 1)


def _sorted_unique_slice(
    percent_strikes: Sequence[float],
    vols: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
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


def _interp_slice_with_clamp(
    source_percent_strikes: np.ndarray,
    source_vols: np.ndarray,
    target_percent_strikes: np.ndarray,
) -> np.ndarray:
    if source_percent_strikes.size == 0:
        raise ValueError("Cannot interpolate an empty implied-vol slice.")
    return np.interp(
        target_percent_strikes,
        source_percent_strikes,
        source_vols,
        left=float(source_vols[0]),
        right=float(source_vols[-1]),
    ).astype(np.float64)


def _fill_invalid_local_var_nearest(local_var: np.ndarray) -> np.ndarray:
    filled = np.asarray(local_var, dtype=np.float64).copy()
    valid_mask = np.isfinite(filled) & (filled > 0.0)
    if not np.any(valid_mask):
        raise ValueError("Local-vol surface is fully invalid after Dupire inversion.")
    if np.all(valid_mask):
        return filled

    valid_indices = np.argwhere(valid_mask)
    invalid_indices = np.argwhere(~valid_mask)
    for invalid_index in invalid_indices:
        distances = np.sum(np.square(valid_indices - invalid_index), axis=1)
        nearest_index = tuple(valid_indices[int(np.argmin(distances))])
        filled[tuple(invalid_index)] = filled[nearest_index]
    return filled


def black_call_price_from_forward(
    forward: np.ndarray,
    strike: np.ndarray,
    implied_vol: np.ndarray,
    maturity: np.ndarray,
    discount_factor: np.ndarray,
    *,
    min_std: float = 1e-12,
) -> np.ndarray:
    """
    Black (1976) call price on a forward/futures:
        C = DF * (F * N(d1) - K * N(d2))
        d1 = (ln(F/K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
    Shapes:
      - forward: [nT, 1]
      - strike: [1, nK]
      - implied_vol: [nT, nK]
      - maturity: [nT, 1] (year fraction)
      - discount_factor: [nT, 1]
    """
    forward = np.asarray(forward, dtype=np.float64)
    strike = np.asarray(strike, dtype=np.float64)
    implied_vol = np.asarray(implied_vol, dtype=np.float64)
    maturity = np.asarray(maturity, dtype=np.float64)
    discount_factor = np.asarray(discount_factor, dtype=np.float64)

    if implied_vol.ndim != 2:
        raise ValueError("implied_vol must be a 2D array [nT, nK].")

    if forward.shape[0] != implied_vol.shape[0] or strike.shape[1] != implied_vol.shape[1]:
        raise ValueError("Shape mismatch between forward/strike and implied_vol.")

    if np.any(forward <= 0) or np.any(strike <= 0):
        raise ValueError("forward and strike must be strictly positive.")
    if np.any(maturity <= 0):
        raise ValueError("maturity must be strictly positive.")
    if np.any(discount_factor <= 0):
        raise ValueError("discount_factor must be strictly positive.")

    std = np.maximum(implied_vol, 0.0) * np.sqrt(maturity)
    intrinsic = discount_factor * np.maximum(forward - strike, 0.0)
    price = intrinsic.copy()

    mask = std > float(min_std)
    if np.any(mask):
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(forward / strike) + 0.5 * (implied_vol * implied_vol) * maturity) / std
            d2 = d1 - std
            price = np.where(
                mask,
                discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2)),
                price,
            )
    return price


def dupire_local_vol_from_call_surface(
    call_prices: np.ndarray,
    strike_grid: Sequence[float],
    maturity_times: Sequence[float],
    *,
    r: Union[float, Sequence[float]] = 0.0,
    q: Union[float, Sequence[float]] = 0.0,
    min_d2c: float = 1e-12,
    max_local_var: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local volatility via the (extended) Dupire formula using call prices C(K, T).

    For a spot underlier with constant (or term-structured) rates r(T) and dividend yield q(T),
    the local variance can be written as:

        sigma_loc^2(K, T) = 2 * (dC/dT + (r-q) * K * dC/dK + q * C) / (K^2 * d2C/dK2)

    Notes:
    - For futures options, a common practical choice is to set q=r so that (r-q)=0, leading to:
        sigma_loc^2(K, T) = 2 * (dC/dT + r * C) / (K^2 * d2C/dK2)
      where C is the discounted call price surface.

    Inputs:
      - call_prices: [nT, nK]
      - strike_grid: [nK]
      - maturity_times: [nT] in year fractions
    Returns:
      - local_vol: [nT, nK]
      - local_var: [nT, nK]
    """
    c = np.asarray(call_prices, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("call_prices must be a 2D array [nT, nK].")

    k = _as_1d_float_array(strike_grid, name="strike_grid")
    t = _as_1d_float_array(maturity_times, name="maturity_times")

    n_terms, n_strikes = c.shape
    if n_terms != t.size or n_strikes != k.size:
        raise ValueError(
            f"Shape mismatch: call_prices is {c.shape}, strike_grid is {k.size}, maturity_times is {t.size}."
        )

    if not np.all(np.diff(k) > 0):
        raise ValueError("strike_grid must be strictly increasing.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("maturity_times must be strictly increasing.")
    if np.any(k <= 0):
        raise ValueError("strike_grid must be strictly positive.")
    if np.any(t <= 0):
        raise ValueError("maturity_times must be strictly positive.")

    r_col = _broadcast_term_vector(r, n_terms, name="r")
    q_col = _broadcast_term_vector(q, n_terms, name="q")

    # First derivative dC/dT, and dC/dK; second derivative d2C/dK2.
    dC_dT = np.gradient(c, t, axis=0, edge_order=2)
    dC_dK = np.gradient(c, k, axis=1, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, k, axis=1, edge_order=2)

    k_row = k.reshape(1, -1)
    denom = (k_row * k_row) * d2C_dK2

    num = dC_dT + (r_col - q_col) * k_row * dC_dK + q_col * c

    local_var = np.full_like(c, np.nan, dtype=np.float64)
    good = np.isfinite(num) & np.isfinite(denom) & (denom > float(min_d2c))
    if np.any(good):
        local_var[good] = 2.0 * num[good] / denom[good]

    local_var = np.where((local_var > 0.0) & np.isfinite(local_var), local_var, np.nan)
    if max_local_var is not None:
        local_var = np.minimum(local_var, float(max_local_var))

    local_vol = np.sqrt(local_var)
    return local_vol.astype(np.float64), local_var.astype(np.float64)


@dataclass
class LocalVolAlgo:
    strike_grid: Sequence[float]
    maturity_times: Sequence[float]
    r: Union[float, Sequence[float]] = 0.0
    q: Union[float, Sequence[float]] = 0.0
    min_d2c: float = 1e-12
    max_local_var: Optional[float] = None

    def call_surface_from_implied_vol(
        self,
        implied_vol_surface: np.ndarray,
        *,
        spot: Optional[float] = None,
        forward: Optional[Union[float, Sequence[float]]] = None,
        discount_factor: Optional[Union[float, Sequence[float]]] = None,
    ) -> np.ndarray:
        """
        Build a call price surface from an implied-vol surface using the Black formula on a forward.

        Provide either:
        - spot: uses forward(T) = spot * exp((r-q) * T), DF(T) = exp(-r * T), OR
        - forward: forward(T) directly (scalar or length nT)

        discount_factor:
        - if None: DF(T) = exp(-r(T) * T)
        - else: scalar or length nT
        """
        implied = np.asarray(implied_vol_surface, dtype=np.float64)
        k = _as_1d_float_array(self.strike_grid, name="strike_grid")
        t = _as_1d_float_array(self.maturity_times, name="maturity_times")
        n_terms = t.size

        if implied.shape != (n_terms, k.size):
            raise ValueError(
                f"implied_vol_surface must have shape {(n_terms, k.size)} (got {implied.shape})."
            )

        r_col = _broadcast_term_vector(self.r, n_terms, name="r")
        q_col = _broadcast_term_vector(self.q, n_terms, name="q")
        t_col = t.reshape(n_terms, 1)

        if discount_factor is None:
            df_col = np.exp(-r_col * t_col)
        else:
            df_col = _broadcast_term_vector(discount_factor, n_terms, name="discount_factor")

        if forward is not None and spot is not None:
            raise ValueError("Provide either spot or forward, not both.")

        if forward is None:
            if spot is None:
                raise ValueError("Either spot or forward must be provided.")
            if spot <= 0:
                raise ValueError("spot must be strictly positive.")
            fwd_col = float(spot) * np.exp((r_col - q_col) * t_col)
        else:
            fwd_col = _broadcast_term_vector(forward, n_terms, name="forward")

        strike_row = k.reshape(1, -1)
        return black_call_price_from_forward(
            forward=fwd_col,
            strike=strike_row,
            implied_vol=implied,
            maturity=t_col,
            discount_factor=df_col,
        )

    def local_vol_surface_from_call_surface(self, call_surface: np.ndarray) -> np.ndarray:
        local_vol, _ = dupire_local_vol_from_call_surface(
            call_prices=call_surface,
            strike_grid=self.strike_grid,
            maturity_times=self.maturity_times,
            r=self.r,
            q=self.q,
            min_d2c=self.min_d2c,
            max_local_var=self.max_local_var,
        )
        return local_vol

    def local_vol_surface_from_implied_vol(
        self,
        implied_vol_surface: np.ndarray,
        *,
        spot: Optional[float] = None,
        forward: Optional[Union[float, Sequence[float]]] = None,
        discount_factor: Optional[Union[float, Sequence[float]]] = None,
    ) -> np.ndarray:
        call_surface = self.call_surface_from_implied_vol(
            implied_vol_surface,
            spot=spot,
            forward=forward,
            discount_factor=discount_factor,
        )
        return self.local_vol_surface_from_call_surface(call_surface)


@dataclass
class LocalVolSurface:
    valuation_date: date
    local_vol_grid: np.ndarray
    percent_strikes: Sequence[float]
    business_days: Sequence[int]
    vol_daycount: DayCountBusN

    def __post_init__(self):
        self._local_vol = np.asarray(self.local_vol_grid, dtype=np.float64)
        self._percent_strikes = np.asarray(self.percent_strikes, dtype=np.float64).reshape(-1)
        self._days = np.asarray(self.business_days, dtype=np.float64).reshape(-1)
        self._days_in_year = float(self.vol_daycount.days_in_year)

        if self._local_vol.shape != (self._days.size, self._percent_strikes.size):
            raise ValueError(
                f"local_vol_grid must have shape {(self._days.size, self._percent_strikes.size)} "
                f"(got {self._local_vol.shape})."
            )
        if np.any(~np.isfinite(self._local_vol)) or np.any(self._local_vol < 0):
            raise ValueError("local_vol_grid must be finite and non-negative.")
        if np.any(self._days <= 0) or np.any(np.diff(self._days) <= 0):
            raise ValueError("business_days must be strictly positive and increasing.")
        if np.any(self._percent_strikes <= 0) or np.any(np.diff(self._percent_strikes) <= 0):
            raise ValueError("percent_strikes must be strictly positive and increasing.")

        self._local_var = np.square(self._local_vol)

    def _axis_weights(self, values: np.ndarray, x: float) -> Tuple[int, int, float]:
        if x <= values[0]:
            return 0, 0, 0.0
        if x >= values[-1]:
            last = values.size - 1
            return last, last, 0.0
        idx = int(np.searchsorted(values, x))
        lower = idx - 1
        upper = idx
        weight = float((x - values[lower]) / (values[upper] - values[lower]))
        return lower, upper, weight

    def _local_var_by_percent_strike_and_day(self, percent_strike: float, day: float) -> float:
        lower_day, upper_day, day_weight = self._axis_weights(self._days, max(float(day), 1.0))
        lower_strike, upper_strike, strike_weight = self._axis_weights(
            self._percent_strikes,
            max(float(percent_strike), 1e-12),
        )

        lower_lower = self._local_var[lower_day, lower_strike]
        lower_upper = self._local_var[lower_day, upper_strike]
        upper_lower = self._local_var[upper_day, lower_strike]
        upper_upper = self._local_var[upper_day, upper_strike]

        lower_interp = lower_lower + (lower_upper - lower_lower) * strike_weight
        upper_interp = upper_lower + (upper_upper - upper_lower) * strike_weight
        return float(lower_interp + (upper_interp - lower_interp) * day_weight)

    def local_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        day = self.vol_daycount.calendar.count_business_days(
            self.valuation_date,
            expiration_date,
            include_start=False,
            include_end=True,
        )
        percent_strike = max(float(strike) / max(float(forward), 1e-12), 1e-12)
        local_var = max(self._local_var_by_percent_strike_and_day(percent_strike=percent_strike, day=day), 0.0)
        return float(np.sqrt(local_var))

    def local_vol_surface(
        self,
        percent_strikes: Sequence[float],
        business_days: Sequence[int],
        forward: float = 1.0,
    ) -> List[List[float]]:
        del forward
        return [
            [
                float(
                    np.sqrt(
                        max(
                            self._local_var_by_percent_strike_and_day(
                                percent_strike=float(percent_strike),
                                day=float(day),
                            ),
                            0.0,
                        )
                    )
                )
                for percent_strike in percent_strikes
            ]
            for day in business_days
        ]


@dataclass
class LocalVolSurfaceBuilder:
    valuation_date: date
    vols: List[List[float]]
    percent_strikes: List[List[float]]
    business_days: List[int]
    target_percent_strikes: Sequence[float]
    target_business_days: Sequence[int]
    vol_daycount: DayCountBusN
    r: Union[float, Sequence[float]] = 0.0
    q: Union[float, Sequence[float]] = 0.0
    forward: Optional[Union[float, Sequence[float]]] = None
    discount_factor: Optional[Union[float, Sequence[float]]] = None
    min_d2c: float = 1e-12
    max_local_var: Optional[float] = None
    _days_in_year: float = field(init=False)
    _implied_vol_grid_cache: Optional[np.ndarray] = field(init=False, default=None)
    _surface_cache: Optional[LocalVolSurface] = field(init=False, default=None)

    def __post_init__(self):
        self._days_in_year = float(self.vol_daycount.days_in_year)
        self._target_percent_strikes = _as_1d_float_array(self.target_percent_strikes, name="target_percent_strikes")
        self._target_business_days = _as_1d_float_array(self.target_business_days, name="target_business_days")

        if not np.all(np.diff(self._target_percent_strikes) > 0):
            raise ValueError("target_percent_strikes must be strictly increasing.")
        if not np.all(np.diff(self._target_business_days) > 0):
            raise ValueError("target_business_days must be strictly increasing.")

        cleaned_slices = sorted(
            zip(self.business_days, self.percent_strikes, self.vols),
            key=lambda item: item[0],
        )
        self._source_business_days: List[int] = []
        self._source_percent_strikes: List[np.ndarray] = []
        self._source_vols: List[np.ndarray] = []
        for business_day, percent_strike_slice, vol_slice in cleaned_slices:
            if int(business_day) <= 0:
                continue
            strikes_arr, vols_arr = _sorted_unique_slice(percent_strikes=percent_strike_slice, vols=vol_slice)
            if strikes_arr.size == 0:
                continue
            self._source_business_days.append(int(business_day))
            self._source_percent_strikes.append(strikes_arr)
            self._source_vols.append(vols_arr)

        if not self._source_business_days:
            raise ValueError("No valid implied-vol slices available for LocalVolSurfaceBuilder.")

    def get_interpolated_implied_vol_grid(self) -> np.ndarray:
        if self._implied_vol_grid_cache is not None:
            return self._implied_vol_grid_cache.copy()

        source_days = np.asarray(self._source_business_days, dtype=np.float64)
        interpolated_by_source = np.vstack(
            [
                _interp_slice_with_clamp(
                    source_percent_strikes=source_strikes,
                    source_vols=source_vols,
                    target_percent_strikes=self._target_percent_strikes,
                )
                for source_strikes, source_vols in zip(self._source_percent_strikes, self._source_vols)
            ]
        )

        output = np.empty((self._target_business_days.size, self._target_percent_strikes.size), dtype=np.float64)
        for strike_idx in range(self._target_percent_strikes.size):
            source_vols = interpolated_by_source[:, strike_idx]
            source_vars = np.square(source_vols) * source_days / self._days_in_year
            interpolated_var = np.interp(
                self._target_business_days,
                source_days,
                source_vars,
                left=float(source_vars[0]),
                right=float(source_vars[-1]),
            )
            output[:, strike_idx] = np.sqrt(
                np.maximum(interpolated_var, 0.0) * self._days_in_year / self._target_business_days
            )

        self._implied_vol_grid_cache = output.astype(np.float64)
        return self._implied_vol_grid_cache.copy()

    def get_local_vol_surface(self) -> LocalVolSurface:
        if self._surface_cache is not None:
            return self._surface_cache

        implied_vol_grid = self.get_interpolated_implied_vol_grid()
        maturity_times = self._target_business_days / self._days_in_year
        algo = LocalVolAlgo(
            strike_grid=self._target_percent_strikes,
            maturity_times=maturity_times,
            r=self.r,
            q=self.q,
            min_d2c=self.min_d2c,
            max_local_var=self.max_local_var,
        )
        call_surface = algo.call_surface_from_implied_vol(
            implied_vol_grid,
            forward=1.0 if self.forward is None else self.forward,
            discount_factor=self.discount_factor,
        )
        _, local_var = dupire_local_vol_from_call_surface(
            call_prices=call_surface,
            strike_grid=self._target_percent_strikes,
            maturity_times=maturity_times,
            r=self.r,
            q=self.q,
            min_d2c=self.min_d2c,
            max_local_var=self.max_local_var,
        )
        cleaned_local_var = _fill_invalid_local_var_nearest(local_var=local_var)
        self._surface_cache = LocalVolSurface(
            valuation_date=self.valuation_date,
            local_vol_grid=np.sqrt(cleaned_local_var),
            percent_strikes=self._target_percent_strikes.tolist(),
            business_days=self._target_business_days.astype(int).tolist(),
            vol_daycount=self.vol_daycount,
        )
        return self._surface_cache

    def get_local_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        return self.get_local_vol_surface().local_vol(
            forward=forward,
            strike=strike,
            expiration_date=expiration_date,
        )
