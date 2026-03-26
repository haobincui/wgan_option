from datetime import datetime

import torch

from quantlib.calculation.analytics.models.analytical.equity.formula import black_scholes_implied_vol, \
    black_scholes_price, black_scholes_implied_vol_torch, black_scholes_price_torch
from quantlib.calculation.analytics.position.instruments.features import OptionType
from quantlib.calendar.daycount import bus_250_gbp, bus_250_usd
from quantlib.risk_engine.implied_distribution.get_density import get_density_torch, get_density
from scripts.data_to_distribution.data_to_distribution_algos.get_distribution import get_tau

"""
ContractId,ContractType,AskPrice,AskSize,Strike,AskQuoteTime,Maturity,AskVol,AskDensity,BidPrice,BidSize,BidQuoteTIme,BidVol,BidDensity
FLG10000A4,CALL,0.9,100.0,100.0,2023-11-14 09:19:44.543412,2034-01-31 20:00:00,0.733853716050297,0.17680806818631822,,,,,
FLG10000A4,CALL,0.9,100.0,100.0,2023-11-14 09:19:48.119036,2034-01-31 20:00:00,0.7339179830833937,0.17680806818631822,,,,,
FLG10000A4,CALL,0.91,100.0,100.0,2023-11-14 09:19:48.119036,2034-01-31 20:00:00,0.7420729790335545,0.1748634015363848,,,,,
FLG10000A4,CALL,0.91,100.0,100.0,2023-11-14 09:19:49.189322,2034-01-31 20:00:00,0.7420924330588133,0.17486199155314353,,,,,
FLG10000A4,CALL,0.9,100.0,100.0,2023-11-14 09:19:50.188949,2034-01-31 20:00:00,0.733955194704263,0.1768066471008467,,,,,
FLG10000A4,CALL,0.9,100.0,100.0,2023-11-14 09:19:58.954440,2034-01-31 20:00:00,0.7341128380696281,0.17680735764358246,,,,,
FLG10000A4,CALL,0.91,100.0,100.0,2023-11-14 09:19:58.954440,2034-01-31 20:00:00,0.7422699991685384,0.17486199155314353,,,,,
FLG10000A4,CALL,0.91,100.0,100.0,2023-11-14 09:20:00.124292,2034-01-31 20:00:00,0.7422912799741456,0.17486411207912056,,,,,
FLG10000A4,CALL,0.9,100.0,100.0,2023-11-14 09:20:01.438555,2034-01-31 20:00:00,0.7341575322252651,0.17680951147625024,,,,,
FLG10000A4,CALL,0.9,100.0,100.0,2023-11-14 09:20:21.723463,2034-01-31 20:00:00,0.734522803744577,0.17680737984804296,,,,,
"""

# strike = 100.25
# # strike = 1.0025
# spot = 110
# # spot = 1
# quote_time = datetime(2023, 7, 11, 10, 37, 34)
# maturity = datetime(2023, 8, 31, 20, 0, 0)
# price = 10.640625
# # price = 1.078125
# option_type = OptionType.CALL
# daycount = bus_250_usd



strike = 100.25
spot = 115.25

quote_time = datetime(2023, 5, 1, 0, 2, 46)
maturity = datetime(2023, 6, 30, 20, 0, 0)
price = 0.007
option_type = OptionType.PUT
daycount = bus_250_usd




# valuation_date = quote_time.date()
# valuation_time = quote_time.time()
expiration_date = maturity
_seconds_per_day = 86400

# tau = get_tau(quote_time, maturity, daycount)
#
tau = (maturity - quote_time).total_seconds() / 86400 / 365
print(price - black_scholes_price(strike, option_type, spot, 0.001, tau, 0.01, 0.01))
print(price - black_scholes_price(strike, option_type, spot, 1, tau, 0.01, 0.01))


vol = black_scholes_implied_vol(price, strike, option_type, spot, tau, 0.01, 0.01)
print(vol)
price = black_scholes_price(strike, option_type, spot, vol, tau, 0.01, 0.01)
print("price", price)

density = get_density(price, strike, option_type, spot, tau, 0.01, 0.01)
print('density', density)

vol_torch = black_scholes_implied_vol_torch(torch.tensor([price],dtype=torch.float64),
                                            torch.tensor([strike], dtype=torch.float64),
                                            [option_type],
                                            torch.tensor([spot]),
                                            torch.tensor([tau], dtype=torch.float64),
                                            torch.tensor([0.01]), torch.tensor([0.01]), torch.device("cpu"))

print(vol_torch)

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


