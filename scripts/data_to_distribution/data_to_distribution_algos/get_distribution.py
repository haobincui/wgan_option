import logging
import threading
from datetime import date, datetime, time

import numpy as np
import pandas as pd
import torch

from market_data.contract_handler.contract_type import ContractType
from market_data.contract_handler.option_contract import OptionContract
from market_data.contract_handler.utils import ContractTerminationRule
from quantlib.calendar.datetime_converter import DatetimeConverter
from quantlib.calendar.daycount import DayCountBusN
from quantlib.risk_engine.implied_distribution.get_density import get_density_torch
from scripts.data_to_distribution.data_to_distribution_algos.get_spot import get_spot_from_quote_time_df


def get_tau(quote_time: datetime, maturity: datetime, daycount: DayCountBusN):
    _seconds_per_day = 86400
    tau = daycount(quote_time.date(), maturity.date())
    if quote_time.time():
        tau = (datetime.combine(maturity.date(), time(20, 0, 0)) -
               datetime.combine(maturity.date(), quote_time.time())).total_seconds() / _seconds_per_day \
              / daycount.days_in_year
        return tau
    else:
        return tau


def get_distribution(input_dataframe: pd.DataFrame, daycount: DayCountBusN, data_date: date,
                     spot_price_df: pd.DataFrame, device: torch.device = None) -> pd.DataFrame:
    """
    Get the distribution of the option data.
    distribution = \partial^2 C / \partial K^2
    ContractId, ContractType, Maturity, Spot, Price, Size, Strike, QuoteTime, Vol, Density

    :param input_dataframe: #RIC,Date-Time,GMT Offset,Type,Bid Price,Bid Size,Ask Price,Ask Size,contract_type
    :param daycount:
    :param data_date:
    :param spot_price_df: get_spot_from_quote_time_df()
    :param device:
    :return: ContractId,ContractType,Maturity,AskSpot,AskPrice,AskSize,ASKStrike,AskQuoteTime,AskVol,AskDensity,
    BidSpot,BidPrice,BidSize,BidStrike,BidQuoteTIme,BidVol,BidDensity
    """
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(f'Using device: [{device}] in [{threading.current_thread().getName()}]')
    logging.info(f'Using device: {device} in [{threading.current_thread().getName()}]')

    bid_price = input_dataframe['Bid Price'].to_numpy(na_value=np.NaN)
    bid_size = input_dataframe['Bid Size'].to_numpy()

    ask_price = input_dataframe['Ask Price'].to_numpy(na_value=np.NaN)
    ask_size = input_dataframe['Ask Size'].to_numpy()

    datetime_converter = DatetimeConverter()
    quote_times = input_dataframe['Date-Time'].apply(lambda x: datetime_converter.from_string_to_datetime(x))

    contract_ids = input_dataframe['#RIC']

    cld = daycount.calendar

    terminal_rule = ContractTerminationRule.EndOfMonth
    contracts_obj = contract_ids.apply(lambda x: OptionContract(x, ContractType.Option))
    maturities = contracts_obj.apply(
        lambda x: x.get_contract_maturity_dates_by_contract_id(data_date, [cld], terminal_rule, time(23, 59, 59)))
    # maturities = maturities.apply(lambda x: datetime.combine(x, time(20, 0, 0)))
    raw_option_types = contracts_obj.apply(lambda x: x.get_option_type().name)
    option_types = contracts_obj.apply(lambda x: bool(x.get_option_type().value)).to_numpy()

    # taus = [get_tau(quote_time, maturity, daycount) for quote_time, maturity in zip(quote_times, maturities)]
    _seconds_per_day = 86400
    taus = (maturities - quote_times).dt.total_seconds() / _seconds_per_day / 365
    taus = taus.to_numpy()

    # spots = quote_times.apply(lambda x: get_spot(x, spot_price_map)).to_numpy()
    spots = get_spot_from_quote_time_df(quote_times, spot_price_df).to_numpy()

    strikes = contracts_obj.apply(lambda x: x.get_strike()).to_numpy()

    bid_price = torch.tensor(bid_price, dtype=torch.float64).to(device)
    bid_size = torch.tensor(bid_size, dtype=torch.float64).to(device)
    ask_price = torch.tensor(ask_price, dtype=torch.float64).to(device)
    ask_size = torch.tensor(ask_size, dtype=torch.float64).to(device)

    taus = torch.tensor(taus, dtype=torch.float64).to(device)
    strikes = torch.tensor(strikes, dtype=torch.float64).to(device)
    spots = torch.tensor(spots, dtype=torch.float64).to(device)
    option_types = torch.tensor(option_types, dtype=torch.bool).to(device)
    r = torch.tensor([0.05], dtype=torch.float64).to(device)

    # calc the density

    print('Start calculating density ... ')
    logging.info(f'Start calculating the density ...')
    bid_density_results, bid_vol = get_density_torch(bid_price, strikes, option_types, spots, taus, r, r, device, True)
    ask_density_results, ask_vol = get_density_torch(ask_price, strikes, option_types, spots, taus, r, r, device, True)
    print('Finished calculating the density ...')
    logging.info(f'Finished calculating the density ...')

    result_ask = pd.DataFrame()
    result_ask['ContractId'] = contract_ids
    result_ask['ContractType'] = raw_option_types
    result_ask['Maturity'] = maturities
    result_ask['AskSpot'] = spots.cpu()
    result_ask['AskPrice'] = ask_price.cpu()
    result_ask['AskSize'] = ask_size.cpu()
    result_ask['ASKStrike'] = strikes.cpu()
    result_ask['AskQuoteTime'] = quote_times

    result_ask['AskVol'] = ask_vol.cpu()
    result_ask['AskDensity'] = ask_density_results.cpu()

    result_bid = pd.DataFrame()
    result_bid['ContractId'] = contract_ids
    result_bid['ContractType'] = raw_option_types
    result_bid['Maturity'] = maturities
    result_bid['BidSpot'] = spots.cpu()
    result_bid['BidPrice'] = bid_price.cpu()
    result_bid['BidSize'] = bid_size.cpu()
    result_bid['BidStrike'] = strikes.cpu()
    result_bid['BidQuoteTIme'] = quote_times

    result_bid['BidVol'] = bid_vol.cpu()
    result_bid['BidDensity'] = bid_density_results.cpu()

    result = pd.concat([result_ask, result_bid], axis=0)

    return result
