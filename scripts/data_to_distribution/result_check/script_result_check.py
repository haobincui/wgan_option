from datetime import datetime

import torch

from quantlib.calculation.analytics.models.analytical.equity.formula import black_scholes_implied_vol, \
    black_scholes_price, black_scholes_implied_vol_torch, black_scholes_price_torch
from quantlib.calculation.analytics.position.instruments.features import OptionType
from quantlib.calendar.daycount import bus_250_gbp, bus_250_usd, act_365
from quantlib.risk_engine.implied_distribution.get_density import get_density_torch, get_density
from scripts.data_to_distribution.data_to_distribution_algos.get_distribution import get_tau


def get_result(spot, strike, quote_time, maturity, price, option_type, daycount):
    _seconds_per_day = 86400

    # tau = get_tau(quote_time, maturity, daycount)
    #
    tau = (maturity - quote_time).total_seconds() / 86400 / 365

    print(price - black_scholes_price(strike, option_type, spot, 0.001, tau, 0.01, 0.01))
    print(price - black_scholes_price(strike, option_type, spot, 1, tau, 0.01, 0.01))

    vol = black_scholes_implied_vol(price, strike, option_type, spot, tau, 0.01, 0.01)
    print('vol', vol)
    price = black_scholes_price(strike, option_type, spot, vol, tau, 0.01, 0.01)
    print("price", price)

    density = get_density(price, strike, option_type, spot, tau, 0.01, 0.01)
    print('density', density)

    vol_torch = black_scholes_implied_vol_torch(torch.tensor([price], dtype=torch.float64),
                                                torch.tensor([strike], dtype=torch.float64),
                                                [option_type],
                                                torch.tensor([spot]),
                                                torch.tensor([tau], dtype=torch.float64),
                                                torch.tensor([0.01]), torch.tensor([0.01]), torch.device("cpu"))

    print('vol torch', vol_torch)

    price_torch = black_scholes_price_torch(
        torch.tensor([strike]), [option_type], torch.tensor([spot]),
        vol_torch, torch.tensor([tau]), torch.tensor([0.01]), torch.tensor([0.01]), torch.device("cpu"))
    print("torch price", price_torch)

    density_torch = get_density_torch(
        torch.tensor([price]), torch.tensor([strike]), [option_type], torch.tensor([spot]),
        torch.tensor([tau]), torch.tensor([0.01]), torch.tensor([0.01]), torch.device("cpu")
    )
    print('density torch', density_torch)

    kp = black_scholes_price_torch(
        torch.tensor([strike + strike * 0.01]), [option_type], torch.tensor([spot]),
        vol_torch, torch.tensor([tau]), torch.tensor([0.01]), torch.tensor([0.01]), torch.device("cpu"))
    kd = black_scholes_price_torch(
        torch.tensor([strike - strike * 0.01]), [option_type], torch.tensor([spot]),
        vol_torch, torch.tensor([tau]), torch.tensor([0.01]), torch.tensor([0.01]), torch.device("cpu"))

    density_torch = (kp - 2 * price_torch + kd) / (strike * 0.01) ** 2
    print('density torch 2 ', density_torch)


def case1():
    strike = 100.25
    spot = 115.25

    quote_time = datetime(2023, 5, 1, 0, 2, 46)
    maturity = datetime(2023, 6, 30, 20, 0, 0)
    price = 0.007
    option_type = OptionType.PUT
    daycount = bus_250_usd

    print("result for case 1")
    get_result(spot, strike, quote_time, maturity, price, option_type, daycount)
    print("end of case 1")
    print('=============================')


def case2():
    strike = 99
    spot = 112
    quote_time = datetime(2023, 5, 31, 20, 21, 29)
    maturity = datetime(2023, 9, 30, 20, 0, 0)
    # price = 10.640625
    price_ask = 0.03125
    price_bid = 0.046875
    option_type = OptionType.PUT
    daycount = bus_250_usd

    print("result ask for case 2")
    get_result(spot, strike, quote_time, maturity, price_ask, option_type, daycount)
    print("end of result ask for case 2")

    print("result bid for case 2")
    get_result(spot, strike, quote_time, maturity, price_bid, option_type, daycount)
    print("end of result bid for case 2")
    print('=============================')


def case3():
    strike = 100.25
    spot = 115.273438
    quote_time = datetime(2023, 5, 1, 0, 2, 47)
    maturity = datetime(2023, 6, 30, 23, 59, 59)
    # price = 10.640625
    price_ask = 0.007
    # price_bid = 0.046875
    option_type = OptionType.PUT
    daycount = bus_250_usd
    print("result ask for case 3")
    get_result(spot, strike, quote_time, maturity, price_ask, option_type, daycount)
    print("end of result ask for case 3")
    print('=============================')


def case4():
    strike = 101.5
    spot = 112.179688
    quote_time = datetime(2023, 6, 21, 14, 0, 2)
    maturity = datetime(2023, 7, 31, 23, 59, 59)
    # price = 10.640625
    price_ask = 0.007
    # price_bid = 0.046875
    option_type = OptionType.PUT
    daycount = bus_250_usd
    print("result ask for case 4")
    get_result(spot, strike, quote_time, maturity, price_ask, option_type, daycount)
    print("end of result ask for case 4")
    print('=============================')


def case5():
    strike = 100
    spot = 115.960938
    quote_time = datetime(2023, 5, 3, 18, 19, 1)
    maturity = datetime(2023, 7, 31, 23, 59, 59)
    price = 0.015625
    option_type = OptionType.PUT
    daycount = bus_250_usd

    print("result for case 5")
    get_result(spot, strike, quote_time, maturity, price, option_type, daycount)
    print("end of case 5")
    print('=============================')


def case6():
    strike = 99
    spot = 113.6796875
    quote_time = datetime(2023, 5, 31, 18, 24, 39)
    maturity = datetime(2023, 9, 29, 23, 59, 59)
    price = 0.03125
    option_type = OptionType.PUT
    daycount = bus_250_usd

    print("result for case 6")
    get_result(spot, strike, quote_time, maturity, price, option_type, daycount)
    print("end of case 6")
    print('=============================')


def case7():
    strike = 100.25
    spot = 113.2578125
    quote_time = datetime(2023, 6, 1, 7, 11, 18)
    maturity = datetime(2023, 7, 31, 23, 59, 59)
    price = 0.007
    option_type = OptionType.PUT
    daycount = bus_250_usd
    # daycount = act_365

    print("result for case 7")
    get_result(spot, strike, quote_time, maturity, price, option_type, daycount)
    print("end of case 7")
    print('=============================')
    # 0.114618085
    # 0.003036025

def case8():
    strike = 99
    spot = 112.3828125
    quote_time = datetime(2023, 6, 20, 4, 21, 21)
    maturity = datetime(2023, 9, 29, 23, 59, 59)
    price = 0.007
    option_type = OptionType.PUT
    daycount = bus_250_usd
    # daycount = act_365

    print("result for case 8")
    get_result(spot, strike, quote_time, maturity, price, option_type, daycount)
    print("end of case 8")
    print('=============================')
    # 0.09175101
    # 0.002903192




if __name__ == '__main__':
    # case1()
    # case2()
    # case3()
    # case4()
    # case5()
    # case6()
    # case7()
    case8()