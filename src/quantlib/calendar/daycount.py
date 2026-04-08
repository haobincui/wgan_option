from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from .holidays import HolidayCalendar, embedded_calendar, usd_calendar, gbp_calendar

@dataclass(frozen=True)
class DayCount(ABC):
    name: str
    days_in_year: int
    
    @abstractmethod
    def __call__(self, start: date, end: date) -> float:
        pass


@dataclass(frozen=True)
class DayCountAct365(DayCount):
    name: str = 'ACT365'
    days_in_year: int = 365

    def __call__(self, start: date, end: date) -> float:
        return (end - start).days / self.days_in_year


@dataclass(frozen=True, init=False)
class DayCountBusN(DayCount):
    calendar: HolidayCalendar
    include_start: bool = False
    include_end: bool = True

    def __init__(
        self,
        name: str,
        calendar: HolidayCalendar,
        days_in_year: int,
        include_start: bool = False,
        include_end: bool = True,
    ) -> None:
        if not hasattr(calendar, "count_business_days"):
            raise TypeError(
                "DayCountBusN expected (name, calendar, days_in_year[, include_start, include_end])"
            )

        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "calendar", calendar)
        object.__setattr__(self, "days_in_year", int(days_in_year))
        object.__setattr__(self, "include_start", bool(include_start))
        object.__setattr__(self, "include_end", bool(include_end))

    def __call__(self, start: date, end: date) -> float:
        return (
            self.calendar.count_business_days(
                start,
                end,
                include_start=self.include_start,
                include_end=self.include_end,
            )
            / self.days_in_year
        )


act_365 = DayCountAct365()

bus_250_embedded = DayCountBusN(
    'BUS250',
    embedded_calendar(),
    250,
    include_start=False,
    include_end=True,
)
bus_250_usd = DayCountBusN(
    'BUS250USD',
    usd_calendar(),
    250,
    include_start=False,
    include_end=True,
)
bus_250_gbp = DayCountBusN(
    'BUS250GBP',
    gbp_calendar(),
    250,
    include_start=False,
    include_end=True,
)
