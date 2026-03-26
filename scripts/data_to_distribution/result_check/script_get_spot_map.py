from datetime import date, datetime

import pandas as pd

from quantlib.calendar.datetime_converter import DatetimeConverter
from scripts.data_to_distribution.data_to_distribution_algos.get_spot import get_spot_price_map, \
    get_spot_from_quote_time_df, get_spot_price_df

input_file = './../input/future_data/0#TY:/2023/future_sample_20230501.csv'
# spot_map = get_spot_price_map(input_file, date(2023, 5, 1))

spot_price_df = get_spot_price_df(input_file, date(2023, 5, 1))

trading_datetime = '2023-05-02T00:16:32.617343162Z'
trading_datetime = DatetimeConverter.from_string_to_datetime(trading_datetime)
quote_time_df = pd.DataFrame({'Date-Time': [trading_datetime]})['Date-Time']
spot = get_spot_from_quote_time_df(quote_time_df, spot_price_df)
print(spot)
