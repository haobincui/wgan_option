from dataclasses import dataclass, field
from datetime import date

from quantlib.calculation.analytics.position.pricer.utils import ValuationMethod
from quantlib.calendar.daycount import DayCount, bus_250_embedded, act_365
from quantlib.calendar.holidays import HolidayCalendar, embedded_calendar
from quantlib.calendar.schedule import Period, TimeUnit, plus_period, EomConvention, BusinessDayConvention


@dataclass
class BlackScholesPricerConfig:
    vol_calendar: DayCount = bus_250_embedded
    rate_day_count: DayCount = bus_250_embedded
    calendar: HolidayCalendar = field(default_factory=embedded_calendar)
    days_in_year: float = 250
    # fd greeks params
    dspot_percent: float = 0.01
    dvol: float = 0.01
    dr: float = 0.0001
    dq: float = 0.0001
    dcorrelation: float = 0.01
    dk: float = 0.0001
    next_day: Period = field(default_factory=lambda: Period(1, TimeUnit.BUSINESS_DAY))
    # mc
    # rng_type: RngType = RngType.NP_PCG64
    seed: int = 12345
    num_paths: int = 10000
    valuation_method: ValuationMethod = ValuationMethod.Analytical

    def next_valuation_date(self, valuation_date: date):
        return plus_period(
            valuation_date,
            self.next_day,
            adj=BusinessDayConvention.FOLLOWING,
            calendar=self.calendar,
            eom=EomConvention.NONE,
        )


blackscholes_default_config = BlackScholesPricerConfig()
