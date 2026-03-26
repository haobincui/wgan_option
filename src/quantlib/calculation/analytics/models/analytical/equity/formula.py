"""
formulas for equity derivative pricing
[the complete guide for option pricing]

"""
from typing import Union, List
import warnings

import numpy as np
import torch
from scipy import optimize
from scipy.stats import norm
from scipy.optimize import brentq
from torch import Tensor
from torch.autograd import Variable

from quantlib.calculation.analytics.models.math_tools.least_square import squared_error
from quantlib.calculation.analytics.models.math_tools.optimize_tool import optimize_tool
from quantlib.calculation.analytics.models.utils import double_is_zero
from quantlib.calculation.analytics.position.instruments.features import OptionType


def black_scholes_price(strike: float, option_type: OptionType,
                        spot: float, vol: float, tau: float, r: float, q: float) -> float:
    dfr = np.exp(-r * tau)
    dfq = np.exp(-q * tau)
    fwd = spot * dfq / dfr
    var_sqrt = vol * np.sqrt(tau)
    if double_is_zero(vol * vol * tau):
        return dfr * (max(fwd - strike, 0) if option_type == OptionType.CALL else max(strike - fwd, 0))
    d1 = np.log(fwd / strike) / var_sqrt + 0.5 * var_sqrt
    d2 = d1 - var_sqrt
    cp = dfr * (fwd * norm.cdf(d1) - strike * norm.cdf(d2))
    return cp if option_type == OptionType.CALL else cp - fwd * dfr + strike * dfr


def black_scholes_delta(strike: float, option_type: OptionType,
                        spot: float, vol: float, tau: float, r: float, q: float) -> float:
    dfr = np.exp(-r * tau)
    dfq = np.exp(-q * tau)
    fwd = spot * dfq / dfr
    var_sqrt = vol * np.sqrt(tau)
    if double_is_zero(vol * vol * tau):
        return (dfq if fwd >= strike else 0) if option_type == OptionType.CALL else (-dfq if fwd <= strike else 0)
    d1 = np.log(fwd / strike) / var_sqrt + 0.5 * var_sqrt
    delta = norm.cdf(d1)
    if option_type == OptionType.PUT:
        delta = delta - 1
    return delta * dfq


def black_scholes_gamma(strike: float, option_type: OptionType,
                        spot: float, vol: float, tau: float, r: float, q: float) -> float:
    dfr = np.exp(-r * tau)
    dfq = np.exp(-q * tau)
    fwd = spot * dfq / dfr
    var_sqrt = vol * np.sqrt(tau)
    if double_is_zero(vol * vol * tau):
        return 0
    d1 = np.log(fwd / strike) / var_sqrt + 0.5 * var_sqrt
    return dfq * norm.pdf(d1) / spot / var_sqrt


def black_scholes_vega(strike: float, option_type: OptionType,
                       spot: float, vol: float, tau: float, r: float, q: float) -> float:
    dfr = np.exp(-r * tau)
    dfq = np.exp(-q * tau)
    fwd = spot * dfq / dfr
    var_sqrt = vol * np.sqrt(tau)
    if double_is_zero(vol * vol * tau):
        return 0
    d1 = np.log(fwd / strike) / var_sqrt + 0.5 * var_sqrt
    return spot * dfq * norm.pdf(d1) * np.sqrt(tau)


def black_scholes_rho_r(strike: float, option_type: OptionType,
                        spot: float, vol: float, tau: float, r: float, q: float) -> float:
    dfr = np.exp(-r * tau)
    dfq = np.exp(-q * tau)
    fwd = spot * dfq / dfr
    var_sqrt = vol * np.sqrt(tau)
    if double_is_zero(vol * vol * tau):
        return (tau * strike * dfr if fwd >= strike else 0) if option_type == OptionType.CALL \
            else (-tau * strike * dfr if fwd <= strike else 0)
    dm = np.log(fwd / strike) / var_sqrt - 0.5 * var_sqrt
    return tau * strike * dfr * norm.cdf(dm) if option_type == OptionType.CALL else -tau * strike * dfr * norm.cdf(-dm)


def black_scholes_rho_q(strike: float, option_type: OptionType,
                        spot: float, vol: float, tau: float, r: float, q: float) -> float:
    dfr = np.exp(-r * tau)
    dfq = np.exp(-q * tau)
    fwd = spot * dfq / dfr
    var_sqrt = vol * np.sqrt(tau)
    if double_is_zero(vol * vol * tau):
        return (-tau * spot * dfq if fwd >= strike else 0) if option_type == OptionType.CALL \
            else (tau * spot * dfq if fwd <= strike else 0)
    dp = np.log(fwd / strike) / var_sqrt + 0.5 * var_sqrt
    return -tau * spot * dfq * norm.cdf(dp) if option_type == OptionType.CALL else tau * spot * dfq * norm.cdf(-dp)


def black_scholes_theta(strike: float, option_type: OptionType,
                        spot: float, vol: float, tau: float, r: float, q: float) -> float:
    dfr = np.exp(-r * tau)
    dfq = np.exp(-q * tau)
    fwd = spot * dfq / dfr
    var_sqrt = vol * np.sqrt(tau)
    if double_is_zero(vol * vol * tau):
        return (q * spot * dfq - r * strike * dfr if fwd >= strike else 0) if option_type == OptionType.CALL \
            else (strike * r * dfr - q * spot * dfr if fwd <= strike else 0)
    dp = np.log(fwd / strike) / var_sqrt + 0.5 * var_sqrt
    dm = dp - var_sqrt
    part1 = -0.5 * spot * dfq * vol * norm.pdf(dp) / np.sqrt(tau)
    return part1 + q * spot * dfq * norm.cdf(dp) - r * strike * dfr * norm.cdf(dm) if option_type == OptionType.CALL \
        else part1 - q * spot * dfq * norm.cdf(-dp) + r * strike * dfr * norm.cdf(-dm)


def black_scholes_vanna(strike: float, option_type: OptionType,
                        spot: float, vol: float, tau: float, r: float, q: float) -> float:
    dfr = np.exp(-r * tau)
    dfq = np.exp(-q * tau)
    fwd = spot * dfq / dfr
    var_sqrt = vol * np.sqrt(tau)
    if double_is_zero(vol * vol * tau):
        return 0
    dp = np.log(fwd / strike) / var_sqrt + 0.5 * var_sqrt
    dm = dp - var_sqrt
    return -dfq * dm * norm.pdf(dp) / vol


def black_scholes_implied_vol(price: float, strike: float, option_type: OptionType,
                              spot: float, tau: float, r: float, q: float) -> float:
    try:
        return brentq(lambda x: price - black_scholes_price(strike, option_type, spot, x, tau, r, q), 0.001, 10)
    except:
        def _target_func(param):
            current_vol = param
            current_price = black_scholes_price(strike, option_type, spot, current_vol, tau, r, q)
            return squared_error(price, current_price)

        def _gradian(param):
            current_vol = param
            return -2 * (price - black_scholes_price(strike, option_type, spot, current_vol, tau, r, q)) * \
                   black_scholes_vega(strike, option_type, spot, current_vol, tau, r, q)

        initial_guess = np.array([1])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The iteration is not making good progress.*",
                category=RuntimeWarning,
            )
            res = optimize.fsolve(_target_func, initial_guess,
                                  fprime=_gradian,
                                  xtol=1e-6)

        return float(*res)


def _norm_cdf(x, device):
    """
    standard normal cumulative distribution function:
    \frac{1}{2} (1 + \text{erf}(\frac{x}{\sqrt{2}}))
    """
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor([2.0], device=device))))


def _norm_pdf(x, device):
    """
    standard normal probability density function:
    \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}
    """
    prefactor = 1 / (torch.sqrt(torch.tensor(2 * torch.pi, device=device)))
    exponent = -0.5 * x ** 2
    return prefactor * torch.exp(exponent)


def black_scholes_price_torch(strike: Tensor, option_type: Union[Tensor, List[OptionType]],
                              spot: Tensor, vol: Tensor,
                              tau: Tensor, r: Tensor, q: Tensor, device) -> Tensor:
    if not isinstance(option_type, Tensor):
        option_type = torch.tensor([bool(i.value) for i in option_type], dtype=torch.bool, device=device)
    dfr = torch.exp(-r * tau)
    dfq = torch.exp(-q * tau)
    fwd = spot * dfq / dfr
    var_sqrt = vol * torch.sqrt(tau)
    if torch.max(vol * vol * tau) <= 1e-6:
        price = torch.where(option_type, dfr * (torch.max(fwd - strike, torch.zeros_like(fwd))),
                            dfr * (torch.max(strike - fwd, torch.zeros_like(fwd))))
        return price
    # if double_is_zero(vol * vol * tau):
    #     return dfr * (max(fwd - strike, 0) if option_type == OptionType.CALL else max(strike - fwd, 0))
    d1 = torch.log(fwd / strike) / var_sqrt + 0.5 * var_sqrt
    d2 = d1 - var_sqrt
    cp = dfr * (fwd * _norm_cdf(d1, device) - strike * _norm_cdf(d2, device))
    # CALL = 1, PUT = 0
    return torch.where(option_type, cp, cp - fwd * dfr + strike * dfr)


def black_scholes_vega_torch(strike: Tensor, option_type: Union[Tensor, List[OptionType]],
                             spot: Tensor, vol: Tensor,
                             tau: Tensor, r: Tensor, q: Tensor, device) -> Tensor:
    dfr = torch.exp(-r * tau)
    dfq = torch.exp(-q * tau)
    fwd = spot * dfq / dfr
    var_sqrt = vol * torch.sqrt(tau)
    if torch.max(vol * vol * tau) <= 1e-6:
        return torch.zeros_like(vol)
    # if double_is_zero(vol * vol * tau):
    #     return 0
    d1 = torch.log(fwd / strike) / var_sqrt + 0.5 * var_sqrt
    return spot * dfq * _norm_pdf(d1, device) * torch.sqrt(tau)


def black_scholes_implied_vol_torch(price: Tensor, strike: Tensor,
                                    option_type: Union[List[OptionType], Tensor],
                                    spot: Tensor, tau: Tensor,
                                    r: Tensor, q: Tensor, device, tol=1e-6, max_iter=1000) -> Tensor:
    if not isinstance(option_type, Tensor):
        option_type = torch.tensor([bool(i.value) for i in option_type], dtype=torch.bool, device=device)

    _min_vega = 0.0001
    vol_guess = [0.4] * len(price)
    vol = torch.tensor(vol_guess, device=device)

    # converged = torch.zeros_like(price, dtype=torch.bool, device=device)

    for _ in range(max_iter):
        current_price = black_scholes_price_torch(strike, option_type, spot, vol, tau, r, q, device)
        vega = black_scholes_vega_torch(strike, option_type, spot, vol, tau, r, q, device)

        vega = torch.where(vega.abs() < _min_vega, torch.full_like(vega, _min_vega), vega)

        # newton-raphson
        # x_next = x - f(x) / f'(x)
        # f(x) = (price(x) - target_price)^2
        # f'(x) = 2 * (price(x) - target_price) * vega(x)
        vol_update = (current_price - price) / (2 * vega)
        # vol_update = torch.where(converged, torch.zeros_like(vol), vol_update)

        vol -= vol_update
        # vol = torch.max(torch.zeros_like(vol), vol - vol_update)
        # converged |= torch.abs(vol_update) < tol
        # vol = torch.max(torch.zeros_like(vol), vol - vol_update)

        if torch.abs(vol_update).max() <= tol:
            break

        # if converged.all():
        #     break

    return vol
