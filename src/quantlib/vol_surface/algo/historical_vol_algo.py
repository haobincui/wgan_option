import logging
import math

from typing import List, Dict, Union
from datetime import date

import pandas as pd
import numpy as np


class HistoricalVolAlgo:
    @staticmethod
    def calc_one_realized_vol(return_rates: List[float], days_in_year=245) -> Union[None, float]:
        """
        用 daily return 计算历史年化波动率
        """
        if return_rates is None or len(return_rates) == 0:
            logging.debug('收益率序列不能为空')
            return None
        realized_vol = 0
        for return_rate in return_rates:
            if return_rate is None:
                logging.debug('回报率是None，跳过该回报率')
                return None
            realized_vol = realized_vol + pow(return_rate, 2)
        realized_vol = math.sqrt(realized_vol) * math.sqrt(days_in_year / len(return_rates))
        return realized_vol

    @staticmethod
    def calc_one_rolling_vol(return_rate_dict: Dict[date, float], window: int) -> dict:
        """
        计算给定daily return 给定窗口的 rolling vol: sqrt((r1^2+r2^2+...rn^2)*245/n), n=window
        """
        if not return_rate_dict:
            return {}
        # 根据key进行排序
        dates = sorted(return_rate_dict.keys())
        new_return_rate_dict = {}
        for date in dates:
            new_return_rate_dict[date] = return_rate_dict[date]
        return_rate_dict = new_return_rate_dict
        trade_dates = list(return_rate_dict.keys())
        return_rates = list(return_rate_dict.values())
        df = pd.DataFrame(return_rates, index=trade_dates)
        rolling_vol_dict = df.rolling(window).apply(HistoricalVolAlgo.calc_one_realized_vol, raw=True)[0].dropna().to_dict()
        return rolling_vol_dict

    @staticmethod
    def calc_one_window_percentiles(start_date: date, end_date: date, return_rate_dict: Dict[date, float],
                                    window: int, percentiles: List[float]):
        """
        计算给定daily return 给定窗口的 rolling vol 相对应的 percentile
        """
        rolling_vol_dict = HistoricalVolAlgo.calc_one_rolling_vol(return_rate_dict, window)
        # 剔除不在前端请求时间段内的rolling_vol
        filtered_rolling_vol_dict = {}
        for date, vol in rolling_vol_dict.items():
            if start_date <= date <= end_date:
                filtered_rolling_vol_dict[date] = vol
        rolling_vol_dict = filtered_rolling_vol_dict
        if not rolling_vol_dict:
            return None
        vol_list = list(rolling_vol_dict.values())
        percentile_vol_dict = {}
        for percentile in percentiles:
            percentile_vol_dict[percentile] = np.percentile(vol_list, percentile)
        return percentile_vol_dict





