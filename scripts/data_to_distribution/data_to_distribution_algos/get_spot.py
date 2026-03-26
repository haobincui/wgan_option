from datetime import datetime, timedelta, date, time
from typing import Dict

import pandas as pd

from market_data.contract_handler.contract_type import ContractType
from market_data.contract_handler.future_contract import FutureContract
from quantlib.calendar.datetime_converter import DatetimeConverter


# spot_price_map = {date1: {
#     time.hour1: {
#         time.minutes1:
#             {time.seconds1: spot1,
#              time.seconds2: spot2,
#              time.seconds3: spot3},
#         time.minutes2:
#             {time.seconds1: spot1,
#              time.seconds2: spot2,
#              time.seconds3: spot3}
#     },
#     time.hour2: {
#         time.minutes1:
#             {time.seconds1: spot1,
#              time.seconds2: spot2,
#              time.seconds3: spot3},
#         time.minutes2:
#             {time.seconds1: spot1,
#              time.seconds2: spot2,
#              time.seconds3: spot3}
#     }
# },
#     date2: {
#         time.hour1: {
#             time.minutes1:
#                 {time.seconds1: spot1,
#                  time.seconds2: spot2,
#                  time.seconds3: spot3},
#             time.minutes2:
#                 {time.seconds1: spot1,
#                  time.seconds2: spot2,
#                  time.seconds3: spot3}
#         },
#         time.hour2: {
#             time.minutes1:
#                 {time.seconds1: spot1,
#                  time.seconds2: spot2,
#                  time.seconds3: spot3},
#             time.minutes2:
#                 {time.seconds1: spot1,
#                  time.seconds2: spot2,
#                  time.seconds3: spot3}
#         }
#     }
# }


def get_spot_from_map(trading_datetime: datetime,
                      spot_price_map: dict) -> float:
    while trading_datetime.date() not in spot_price_map:
        trading_datetime -= timedelta(days=1)

    hour_map = spot_price_map[trading_datetime.date()].get(int(trading_datetime.hour), None)
    while hour_map is None:
        trading_datetime -= timedelta(hours=1)
        hour_map = spot_price_map[trading_datetime.date()].get(int(trading_datetime.hour), None)

    minutes_map = hour_map.get(int(trading_datetime.minute), None)
    while minutes_map is None:
        trading_datetime -= timedelta(minutes=1)
        minutes_map = hour_map.get(int(trading_datetime.minute), None)

    spot = minutes_map.get(int(trading_datetime.second), None)
    while spot is None:
        trading_datetime -= timedelta(seconds=1)
        spot = minutes_map.get(int(trading_datetime.second), None)
    return spot


def _get_target_month_code(current_month: int):
    if current_month <= 3:
        return 'H'
    elif current_month <= 6:
        return 'M'
    elif current_month <= 9:
        return 'U'
    else:
        return 'Z'


def get_spot_price_map(input_file: str, start_date: date) -> dict:
    spot_price_map = {}
    raw_data = pd.read_csv(input_file)
    maturity_month_code = raw_data['#RIC'].apply(
        lambda x: FutureContract(x, ContractType.Future).get_maturity_month_code())

    target_month_code = _get_target_month_code(start_date.month)

    raw_data = raw_data[maturity_month_code == target_month_code].drop(
        columns=['#RIC', 'Alias Underlying RIC', 'Domain', 'Type', 'Close Mid Price'])
    # Close Bid,Close Ask, Date-Time
    for _, row in raw_data.iterrows():
        current_datetime = DatetimeConverter.from_string_to_datetime(row['Date-Time'])
        current_date = current_datetime.date()
        # current_time = current_datetime.time()
        current_hour = int(current_datetime.hour)
        current_minute = int(current_datetime.minute)
        current_second = int(current_datetime.second)
        if current_date not in spot_price_map:
            spot_price_map[current_date] = {}
        if current_hour not in spot_price_map[current_date]:
            spot_price_map[current_date][current_hour] = {}
        if current_minute not in spot_price_map[current_date][current_hour]:
            spot_price_map[current_date][current_hour][current_minute] = {}
        spot_price_map[current_date][current_hour][current_minute][current_second] = (row['Close Bid'] + row[
            'Close Ask']) / 2
        # spot_price_map[current_date][current_hour] = (row['Close Bid'] + row['Close Ask']) / 2

    return spot_price_map


def get_spot_price_df(input_file: str, start_date) -> pd.DataFrame:
    raw_data = pd.read_csv(input_file)
    maturity_month_code = raw_data['#RIC'].apply(
        lambda x: FutureContract(x, ContractType.Future).get_maturity_month_code())

    target_month_code = _get_target_month_code(start_date.month)

    raw_data = raw_data[maturity_month_code == target_month_code].drop(
        columns=['#RIC', 'Alias Underlying RIC', 'Domain', 'Type', 'Close Mid Price'])
    raw_data['Spot Price'] = (raw_data['Close Bid'] + raw_data['Close Ask']) / 2
    raw_data = raw_data.drop(columns=['Close Bid', 'Close Ask'])
    raw_data['Date-Time'] = raw_data['Date-Time'].apply(lambda x: DatetimeConverter.from_string_to_datetime(x))

    return raw_data.sort_values(by='Date-Time')


def get_spot_from_quote_time_df(quote_time: pd.Series, spot_price_df: pd.DataFrame) -> pd.Series:
    quote_time_df = pd.DataFrame(quote_time, columns=['Date-Time'])
    quote_time_df['original_order'] = quote_time_df.index
    quote_time_df = quote_time_df.sort_values(by='Date-Time')
    merged_df = pd.merge_asof(quote_time_df, spot_price_df,
                              left_on='Date-Time',
                              right_on='Date-Time',
                              direction='backward')
    return merged_df.sort_values(by='original_order')['Spot Price']
