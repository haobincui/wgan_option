import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import norm

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

