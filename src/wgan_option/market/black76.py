"""Black (1976) pricing and implied-volatility inversion for futures options."""

from __future__ import annotations

import math
from typing import Any, Tuple

from scipy.optimize import brentq
from scipy.special import ndtr


def _is_call(option_type: Any) -> bool:
    name = getattr(option_type, "name", option_type)
    normalized = str(name).strip().upper()
    if normalized in {"CALL", "C", "1"}:
        return True
    if normalized in {"PUT", "P", "0"}:
        return False
    raise ValueError(f"Unsupported option type: {option_type!r}")


def black76_no_arbitrage_bounds(
    *,
    futures_price: float,
    strike: float,
    discount_factor: float,
    option_type: Any,
) -> Tuple[float, float]:
    """Return European futures-option lower and upper price bounds."""

    futures_price = float(futures_price)
    strike = float(strike)
    discount_factor = float(discount_factor)
    if futures_price <= 0 or strike <= 0:
        raise ValueError("futures_price and strike must be positive")
    if not 0 < discount_factor <= 1.5:
        raise ValueError("discount_factor must be positive and plausible")

    if _is_call(option_type):
        return (
            discount_factor * max(futures_price - strike, 0.0),
            discount_factor * futures_price,
        )
    return (
        discount_factor * max(strike - futures_price, 0.0),
        discount_factor * strike,
    )


def black76_price(
    *,
    futures_price: float,
    strike: float,
    tau: float,
    volatility: float,
    discount_factor: float,
    option_type: Any,
) -> float:
    """Price one European option on a futures contract under Black (1976)."""

    futures_price = float(futures_price)
    strike = float(strike)
    tau = float(tau)
    volatility = float(volatility)
    discount_factor = float(discount_factor)
    if futures_price <= 0 or strike <= 0:
        raise ValueError("futures_price and strike must be positive")
    if tau < 0 or volatility < 0 or discount_factor <= 0:
        raise ValueError("tau, volatility and discount_factor must be non-negative/positive")

    is_call = _is_call(option_type)
    if tau == 0 or volatility == 0:
        intrinsic = max(futures_price - strike, 0.0) if is_call else max(
            strike - futures_price, 0.0
        )
        return discount_factor * intrinsic

    sigma_sqrt_tau = volatility * math.sqrt(tau)
    d1 = (
        math.log(futures_price / strike)
        + 0.5 * volatility * volatility * tau
    ) / sigma_sqrt_tau
    d2 = d1 - sigma_sqrt_tau
    if is_call:
        return discount_factor * (
            futures_price * float(ndtr(d1)) - strike * float(ndtr(d2))
        )
    return discount_factor * (
        strike * float(ndtr(-d2)) - futures_price * float(ndtr(-d1))
    )


def black76_implied_vol(
    *,
    price: float,
    futures_price: float,
    strike: float,
    tau: float,
    discount_factor: float,
    option_type: Any,
    lower_vol: float = 1.0e-8,
    upper_vol: float = 5.0,
    price_tolerance: float = 1.0e-10,
) -> float:
    """Invert Black (1976), rejecting observations outside no-arbitrage bounds."""

    price = float(price)
    tau = float(tau)
    if not math.isfinite(price) or price <= 0:
        raise ValueError("option price must be finite and positive")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")

    lower_price, upper_price = black76_no_arbitrage_bounds(
        futures_price=futures_price,
        strike=strike,
        discount_factor=discount_factor,
        option_type=option_type,
    )
    tolerance = max(
        float(price_tolerance),
        1.0e-12 * max(1.0, abs(price), abs(upper_price)),
    )
    if price < lower_price - tolerance or price > upper_price + tolerance:
        raise ValueError(
            "option price violates Black-76 bounds: "
            f"price={price}, bounds=({lower_price}, {upper_price})"
        )
    if abs(price - lower_price) <= tolerance:
        return float(lower_vol)

    def objective(volatility: float) -> float:
        return black76_price(
            futures_price=futures_price,
            strike=strike,
            tau=tau,
            volatility=volatility,
            discount_factor=discount_factor,
            option_type=option_type,
        ) - price

    low_value = objective(float(lower_vol))
    high_value = objective(float(upper_vol))
    if low_value > tolerance:
        raise ValueError("Black-76 lower volatility already overprices the option")
    if high_value < -tolerance:
        raise ValueError(
            f"implied volatility exceeds configured upper bound {upper_vol}"
        )
    return float(
        brentq(
            objective,
            float(lower_vol),
            float(upper_vol),
            xtol=1.0e-12,
            rtol=1.0e-12,
            maxiter=200,
        )
    )
