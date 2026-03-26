from functools import singledispatch
from datetime import date, time, datetime

from quantlib.calculation.analytics.models import analytic_lib
from quantlib.calculation.analytics.position.instruments.cash import CashPayment
from quantlib.calculation.analytics.position.instruments.features import Instrument
from quantlib.calculation.analytics.position.instruments.interest_rate.european import Forward, VanillaEuropean, \
    VanillaImpliedVolatility
from quantlib.calculation.analytics.position.pricer.config import BlackScholesPricerConfig, blackscholes_default_config

_seconds_per_day = 86400


def _vol_time(expiration_date: date, valuation_date: date, valuation_time: time,
              config: BlackScholesPricerConfig) -> float:
    # TODO: fix tau
    # tau = config.vol_calendar(valuation_date, expiration_date)
    if valuation_time:
        return (datetime.combine(expiration_date, time(15, 0, 0)) -
                datetime.combine(expiration_date, valuation_time)).total_seconds() / _seconds_per_day
    else:
        return config.vol_calendar(valuation_date, expiration_date)


@singledispatch
def blackscholes_const_params_pv_analytic(
        instrument: Instrument,
        spot: float,
        vol: float,
        r: float,
        q: float,
        valuation_date: date = date.today(),
        config: BlackScholesPricerConfig = blackscholes_default_config,
        valuation_time: time = None) -> float:
    """
    Generic BCT Black-Scholes constant parameter PV calculator.
    Implements basic component pricing. Decomposable position's pricing is generic and implemented
    as simple combination of basic pricing.
    Config is the place to adjust algo params. For example, if a product has multiple pricing algorithms
    under "Black-Scholes const params" category, one can distinguish them by adding an algo flag in the config.
    :param instrument: The position to be priced
    :param spot: Underlyer spot
    :param vol: Vol
    :param r: R
    :param q: Q
    :param valuation_date: Valuation date (default: today)
    :param config: Pricer config, includes calendars etc. For special pricers, derive from #BctBlackScholesPricerConfig
    :param valuation_time: Valuation time (default: None)
    :return: Instrument pv
    """
    raise ValueError(f'bct_blackscholes_const_params_pv_analytic: {instrument.__class__.__name__} not implemented')


@blackscholes_const_params_pv_analytic.register
def _(instrument: CashPayment, spot: float, vol: float, r: float, q: float,
      valuation_date: date = date.today(),
      config: BlackScholesPricerConfig = blackscholes_default_config, valuation_time: time = None) -> float:
    return instrument.amount


@blackscholes_const_params_pv_analytic.register
def _(instrument: Forward, spot: float, vol: float, r: float, q: float,
      valuation_date: date = date.today(),
      config: BlackScholesPricerConfig = blackscholes_default_config, valuation_time: time = None) -> float:
    tau = _vol_time(instrument.expiration_date, valuation_date, valuation_time, config)
    return analytic_lib.forward_price(instrument.strike, spot, vol, tau, r, q)


@blackscholes_const_params_pv_analytic.register
def _(instrument: VanillaEuropean, spot: float, vol: float, r: float, q: float,
      valuation_date: date = date.today(),
      config: BlackScholesPricerConfig = blackscholes_default_config, valuation_time: time = None) -> float:
    tau = _vol_time(instrument.expiration_date, valuation_date, valuation_time, config)
    return analytic_lib.vanilla_european_price(instrument.strike, instrument.option_type, spot, vol, tau, r, q, config)


@blackscholes_const_params_pv_analytic.register
def _(instrument: VanillaImpliedVolatility, spot: float, vol: float, r: float, q: float,
      valuation_date: date = date.today(),
      config: BlackScholesPricerConfig = blackscholes_default_config, valuation_time: time = None) -> float:
    tau = _vol_time(instrument.expiration_date, valuation_date, valuation_time, config)
    return analytic_lib.vanilla_implied_vol(instrument.strike, instrument.option_type, instrument.price, spot, tau, r,
                                            q, config)
