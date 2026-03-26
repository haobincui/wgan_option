from typing import List, Union

from torch import Tensor

from quantlib.calculation.analytics.models.analytical.equity.formula import black_scholes_implied_vol_torch, \
    black_scholes_price_torch, black_scholes_implied_vol, black_scholes_price
from quantlib.calculation.analytics.position.instruments.features import OptionType


def get_density(price: float, strike: float, option_type: OptionType,
                spot: float, tau: float, r: float, q: float) -> float:
    vol = black_scholes_implied_vol(price, strike, option_type, spot, tau, r, q)
    option_price = black_scholes_price(strike, option_type, spot, vol, tau, r, q)
    kp = black_scholes_price(strike + strike * 0.01, option_type, spot, vol, tau, r, q)
    kd = black_scholes_price(strike - strike * 0.01, option_type, spot, vol, tau, r, q)
    return (kp - 2 * option_price + kd) / (strike * 0.01) ** 2


def get_density_torch(price: Tensor, strike: Tensor, option_type: Union[Tensor, List[OptionType]],
                      spot: Tensor, tau: Tensor, r: Tensor, q: Tensor, device, return_vol: bool = False) -> Tensor:
    vol = black_scholes_implied_vol_torch(price, strike, option_type, spot, tau, r, q, device)

    option_price = black_scholes_price_torch(strike, option_type, spot, vol, tau, r, q, device)
    kp = black_scholes_price_torch(strike + strike * 0.01, option_type, spot, vol, tau, r, q, device)
    kd = black_scholes_price_torch(strike - strike * 0.01, option_type, spot, vol, tau, r, q, device)
    if return_vol:
        return ((kp - 2 * option_price + kd) / (strike * 0.01) ** 2, vol)

    return (kp - 2 * option_price + kd) / (strike * 0.01) ** 2
