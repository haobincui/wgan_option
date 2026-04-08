from quantlib.calculation.analytics.models.analytical.equity import formula
from quantlib.calculation.analytics.position.instruments.features import OptionType
from quantlib.calculation.analytics.position.pricer.config import BlackScholesPricerConfig
from quantlib.calculation.analytics.position.pricer.utils import ValuationMethod


def vanilla_european_price(strike: float, option_type: OptionType,
                           spot: float, vol: float, tau: float, r: float, q: float,
                           config: BlackScholesPricerConfig) -> float:
    if config.valuation_method == ValuationMethod.Analytical:
        return formula.black_scholes_price(
            strike=strike,
            option_type=option_type,
            spot=spot,
            vol=vol,
            tau=tau,
            r=r,
            q=q,
        )


# def forward_price(strike: float, spot: float, vol: float, tau: float, r: float, q: float,
#                   config: BlackScholesPricerConfig) -> float:
#     if config.valuation_method == ValuationMethod.Analytical:
#         return formula.forward_price(strike, spot, vol, tau, r, q)

def vanilla_implied_vol(strike: float, option_type: OptionType, price: float,
                        spot: float, tau: float, r: float, q: float,
                        config: BlackScholesPricerConfig) -> float:
    if config.valuation_method == ValuationMethod.Analytical:
        return formula.black_scholes_implied_vol(
            price=price,
            strike=strike,
            option_type=option_type,
            spot=spot,
            tau=tau,
            r=r,
            q=q,
        )


