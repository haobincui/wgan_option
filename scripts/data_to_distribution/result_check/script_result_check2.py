from datetime import date

import pandas as pd

from quantlib.calendar.daycount import bus_250_gbp
from scripts.data_to_distribution.data_to_distribution_algos.get_distribution import get_distribution

input_file = pd.read_csv('../input/test_us_input.csv')
daycount = bus_250_gbp
data_date = date(2024, 3, 16)

res = get_distribution(input_file, daycount, data_date)
print(res)
