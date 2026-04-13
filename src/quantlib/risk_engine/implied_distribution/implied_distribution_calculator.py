from datetime import date
from typing import List

from quantlib.calculation.analytics.position.instruments.interest_rate.european import VanillaEuropean
from quantlib.calendar.holidays import HolidayCalendar
from quantlib.risk_engine.implied_distribution.implied_distribution import ImpliedDistributionByDate


class ImpliedDistributionCalculator:
    def __init__(self,
                 underlying_id: str,
                 model_date: date,
                 instruments: List[VanillaEuropean],
                 option_prices: List[float],
                 calendar: HolidayCalendar):
        self.underlying_id = underlying_id
        self.model_date = model_date
        self.instruments = instruments
        self.option_prices = option_prices
        self.calendar = calendar

    def calc(self) -> ImpliedDistributionByDate:
        pass


