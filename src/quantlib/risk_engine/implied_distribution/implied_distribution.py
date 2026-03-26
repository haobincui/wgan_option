from dataclasses import dataclass
from datetime import date
from typing import List

from quantlib.calendar.holidays import HolidayCalendar


@dataclass()
class ImpliedDistribution:
    """
    implied distribution = \partial^2 C / \partial K^2 for a \tau
    """
    underlying_id: str


@dataclass()
class ImpliedDistributionByDate(ImpliedDistribution):
    model_date: date
    percentage_strikes: List[float]
    probabilities: List[List[float]]
    calendar: HolidayCalendar

    def __post_init__(self):
        if len(self.percentage_strikes) != len(self.probabilities):
            raise ValueError('The length of Strikes should have the same length as Probabilities')

    def get_implied_distribution_for_a_date(self, expiration_date: date) -> (List[float], List[List[float]]):
        pass

    def get_implied_distribution_for_a_percentage_strike(self, percentage_strike: float) -> (
            List[date], List[List[float]]):
        pass

    def get_implied_distribution_for_a_spot(self, spot: float, strike: float) -> (List[date], List[List[float]]):
        pass

    def get_implied_probability_by_forward(self, forward: float, strike: float,
                                           expiration_date: date) -> float:
        pass

    def get_implied_probability_by_spot(self, spot: float, strike: float, r: float, q: float,
                                        expiration_date: date) -> float:
        pass



@dataclass()
class ImpliedDistributionByTenor(ImpliedDistribution):
    tenor: int
    percentage_strikes: List[float]
    probabilities: List[List[float]]
    calendar: HolidayCalendar

    def __post_init__(self):
        if len(self.percentage_strikes) != len(self.probabilities):
            raise ValueError('The length of Strikes should have the same length as Probabilities')

    def get_implied_distribution_for_a_date(self, expiration_date: date) -> (List[float], List[List[float]]):
        pass

    def get_implied_distribution_for_a_percentage_strike(self, percentage_strike: float) -> (
            List[date], List[List[float]]):
        pass

    def get_implied_distribution_for_a_spot(self, spot: float, strike: float) -> (List[date], List[List[float]]):
        pass

    def get_implied_probability_by_forward(self, forward: float, strike: float,
                                           expiration_date: date) -> float:
        pass

    def get_implied_probability_by_spot(self, spot: float, strike: float, r: float, q: float,
                                        expiration_date: date) -> float:
        pass







