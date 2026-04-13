from __future__ import annotations
from abc import ABC, abstractmethod
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import List, Any
from datetime import date, timedelta
import numpy as np
from scipy import interpolate

from quantlib.calendar.daycount import DayCount
from quantlib.calendar.holidays import HolidayCalendar
from quantlib.calendar.schedule import BusinessDayConvention, EomConvention, period_from_string, plus_period


@dataclass
class TermVolSurface:
    valuation_date: date
    expiration_dates: List[date]
    vols: List[float]
    vol_daycount: DayCount
    initial_vol: float = field(init=False)
    interp: Any = field(init=False)

    def __post_init__(self):
        ts = [0]
        variances = [0]
        for idx, t in enumerate(self.expiration_dates):
            if t > self.valuation_date:
                tau = self.vol_daycount(self.valuation_date, t)
                var = self.vols[idx] * self.vols[idx] * tau
                ts.append(tau)
                variances.append(var)
        if len(ts) < 2:
            raise ValueError('term vol requires at least 1 point')
        self.initial_vol = np.sqrt(variances[1] / ts[1])
        self.interp = interpolate.interp1d(ts, variances, fill_value='extrapolate')

    def implied_variance(self, expiration_date: date):
        tau = self.vol_daycount(self.valuation_date, expiration_date)
        return self.interp(tau)

    def implied_vol(self, expiration_date: date):
        if expiration_date <= self.valuation_date:
            return self.initial_vol
        tau = self.vol_daycount(self.valuation_date, expiration_date)
        var = self.interp(tau)
        return np.sqrt(var / tau)


@dataclass
class ImpliedVolSurface(ABC):
    valuation_date: date

    @abstractmethod
    def implied_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        pass

    @abstractmethod
    def implied_vol_by_spot(self, spot: float, strike: float, expiration_date: date) -> float:
        pass

    @abstractmethod
    def parallel_bump(self, amount: float) -> ImpliedVolSurface:
        pass

    @abstractmethod
    def roll(self, new_valuation_date: date) -> ImpliedVolSurface:
        pass

    @abstractmethod
    def shift_valuation_date(self, new_valuation_date: date) -> ImpliedVolSurface:
        """
        Copy the current vol surface to the new valuation date if there is no vol surface at that day
        """
        pass


@dataclass
class ConstVolSurface(ImpliedVolSurface):
    vol: float

    def implied_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        return self.vol

    def implied_vol_by_spot(self, spot: float, strike: float, expiration_date: date) -> float:
        return self.vol

    def parallel_bump(self, amount: float) -> ImpliedVolSurface:
        return ConstVolSurface(valuation_date=self.valuation_date, vol=self.vol + amount)

    def roll(self, new_valuation_date: date) -> ImpliedVolSurface:
        return ConstVolSurface(valuation_date=new_valuation_date, vol=self.vol)

    def shift_valuation_date(self, new_valuation_date: date) -> ImpliedVolSurface:
        return ConstVolSurface(valuation_date=new_valuation_date, vol=self.vol)


@dataclass
class InterpolatedImpliedVolSurface(ImpliedVolSurface):
    percent_strikes: List[float]
    business_days: List[int]
    vols: List[List[float]]
    calendar: HolidayCalendar
    days_in_year: int = 250  # need days in year, default = 250
    interp1d_var: Any = field(init=False)
    interps: Any = field(init=False)
    interp2d: Any = field(init=False)

    # boundary value,initialized as constant
    vol_lower_days_lower: float = field(init=False)
    vol_higher_days_lower: float = field(init=False)
    atm_vol: float = field(init=False)

    vol_lower_days_upper: float = field(init=False)
    vol_higher_days_upper: float = field(init=False)

    def __post_init__(self):

        if len(self.business_days) < 2:  # only one business day
            self.vols = [self.vols[0], self.vols[0]]
            self.business_days = [0, self.business_days[0]]

        if len(self.percent_strikes) < 2:  # only one percent strike
            self.percent_strikes = [self.percent_strikes[0],
                                    1000 * self.percent_strikes[0]]
            self.vols = [[vol[0], vol[0]] for vol in self.vols]

        self.interps = [interpolate.interp1d(self.percent_strikes, vol, fill_value='extrapolate') for vol in self.vols]

        vol_atm = [self.interps[i](1).tolist() for i in range(0, len(self.business_days))]
        var_atm = [0] * len(vol_atm)
        for i, t in enumerate(self.business_days):
            var_atm[i] = vol_atm[i] * vol_atm[i] * t / self.days_in_year

        self.interp1d_var = interpolate.interp1d(self.business_days, var_atm, fill_value='extrapolate')
        self.atm_vol = vol_atm[-1]

        # ini vol when days in [0, 5)
        self.vol_lower_days_lower = self.interps[0](self.percent_strikes[0])  # moneyness < 0.7
        self.vol_higher_days_lower = self.interps[0](self.percent_strikes[-1])  # monyness > 1.3

        # ini vol when days > 126
        self.vol_lower_days_upper = self.interps[-1](self.percent_strikes[0])  # moneyness < 0.7
        self.vol_higher_days_upper = self.interps[-1](self.percent_strikes[-1])  # monyness > 1.3

        def interp(moneyness, days):
            if days in self.business_days and moneyness in self.percent_strikes:
                days_idx = [idx for idx, ele in enumerate(self.business_days) if ele == days][0]
                vol = self.interps[days_idx](moneyness)
                return float(vol)
            if days == 0:
                return self.interps[0](moneyness)
            days_higher_idx = np.searchsorted(self.business_days, days, side='left')
            days_lower_idx = days_higher_idx - 1
            days_lower = self.business_days[days_lower_idx]
            days_higher = self.business_days[days_higher_idx]
            vol_low = self.interps[days_lower_idx](moneyness)
            vol_high = self.interps[days_higher_idx](moneyness)

            help1 = (days - days_lower) / (days_higher - days_lower)  # time diff
            var_high = (vol_high * vol_high * days_higher / self.days_in_year)
            var_low = (vol_low * vol_low * days_lower / self.days_in_year)
            vol = np.sqrt((help1 * (var_high - var_low) + var_low) * self.days_in_year / days)
            return float(vol)

        self.interp2d = interp

    def implied_vol(self, forward: float, strike: float, expiration_date: date) -> float:

        days = self.calendar.count_business_days(
            self.valuation_date,
            expiration_date,
            include_start=False,
            include_end=True,
        )
        if days < 0:
            raise ValueError(f'valuation date {self.valuation_date.isoformat()} '
                             f'is after the last expiration date of the original vol surface')

        moneyness = strike / forward
        if self.business_days[0] <= days <= self.business_days[-1]:  # inside the surface
            if self.percent_strikes[0] <= moneyness <= self.percent_strikes[-1]:
                vol = self.interp2d(moneyness, days)
            elif moneyness < self.percent_strikes[0]:  # moneyness < 0.7
                moneyness = self.percent_strikes[0]
                vol = self.interp2d(moneyness, days)
            else:  # moneyness > 1.3
                moneyness = self.percent_strikes[-1]
                vol = self.interp2d(moneyness, days)

        elif 0 <= days < self.business_days[0]:  # only vol interp
            if self.percent_strikes[0] <= moneyness <= self.percent_strikes[-1]:
                vol = self.interps[0](moneyness)
            elif moneyness < self.percent_strikes[0]:
                vol = self.vol_lower_days_lower
            else:  # moneyness > 1.3
                vol = self.vol_higher_days_lower

        else:  # days > 126 need vol shift
            var_atm = self.interp1d_var(days)
            if var_atm < 0:
                var_atm = 0
            shift = np.sqrt(var_atm * self.days_in_year / days) - self.atm_vol
            if self.percent_strikes[0] <= moneyness <= self.percent_strikes[-1]:
                vol = self.interps[-1](moneyness) + shift
            elif moneyness < self.percent_strikes[0]:
                vol = self.vol_lower_days_upper + shift
            else:  # moneyness > 1.3
                vol = self.vol_higher_days_upper

        if vol < 0:
            vol = 0
        return float(vol)

    def implied_vol_by_spot(self, spot: float, strike: float, expiration_date: date) -> float:
        return self.implied_vol(spot, strike, expiration_date)

    def parallel_bump(self, amount: float) -> ImpliedVolSurface:
        new_vols = [[v + amount for v in row] for row in self.vols]
        return InterpolatedImpliedVolSurface(valuation_date=self.valuation_date,
                                               percent_strikes=self.percent_strikes,
                                               business_days=self.business_days,
                                               vols=new_vols,
                                               calendar=self.calendar)

    def roll(self, new_valuation_date: date) -> ImpliedVolSurface:
        if new_valuation_date < self.valuation_date:
            raise ValueError(f'new valuation date {new_valuation_date.isoformat()}'
                             f'is before the original valuation date {self.valuation_date.isoformat()}')
        new_business_days = []
        new_vols = []
        n = self.calendar.count_business_days(
            self.valuation_date,
            new_valuation_date,
            include_start=False,
            include_end=True,
        )
        if n > self.business_days[-1]:
            raise ValueError(f'new valuation date {new_valuation_date.isoformat()} '
                             f'is after the last expiration date of the original vol surface')
        for idx, i in enumerate(self.business_days):
            if i >= n:
                new_business_days.append(self.business_days[idx] - n)
                new_vols.append(self.vols[idx])

        return InterpolatedImpliedVolSurface(valuation_date=new_valuation_date,
                                               percent_strikes=self.percent_strikes,
                                               business_days=new_business_days,
                                               vols=new_vols,
                                               calendar=self.calendar)

    def shift_valuation_date(self, new_valuation_date: date) -> ImpliedVolSurface:
        if new_valuation_date < self.valuation_date:
            raise ValueError(f'new valuation date {new_valuation_date.isoformat()}'
                             f'is before the original valuation date {self.valuation_date.isoformat()}')
        else:
            return InterpolatedImpliedVolSurface(valuation_date=new_valuation_date,
                                                   percent_strikes=self.percent_strikes,
                                                   business_days=self.business_days,
                                                   vols=self.vols,
                                                   calendar=self.calendar)


@dataclass
class InterpolatedPercentStrikeImpliedVolSurface(ImpliedVolSurface):
    percent_strikes: List[float]
    business_days: List[int]
    vols: List[List[float]]
    calendar: HolidayCalendar
    interp2d: Any = field(init=False)

    def __post_init__(self):
        var = [[self.vols[i][j] ** 2 * self.business_days[i] for j in range(len(self.percent_strikes))]
               for i in range(len(self.business_days))]
        n = len(var[0])
        if n > 3:
            kinds = 'cubic'
        elif n == 3:
            kinds = 'quadratic'
        elif n == 2:
            kinds = 'linear'
        else:
            kinds = 'zero'
        interp1d = [interpolate.interp1d(self.percent_strikes, var[i], kind=kinds)
                    for i in range(len(self.business_days))]
        min_percent_strikes = min(self.percent_strikes)
        max_percent_strikes = max(self.percent_strikes)

        def interp2d(percent, days):
            if days <= 0:
                days = 1
            if percent < min_percent_strikes:
                percent = min_percent_strikes
            if percent > max_percent_strikes:
                percent = max_percent_strikes
            if len(self.business_days) == 1:
                return np.sqrt(interp1d[0](percent)/self.business_days[0])
            ind = bisect_left(self.business_days, days)
            if ind == 0:
                ind = 1
            elif ind == len(self.business_days):
                ind = len(self.business_days) - 1
            lower_days = self.business_days[ind - 1]
            lower_vol = interp1d[ind - 1](percent)
            upper_days = self.business_days[ind]
            upper_vol = interp1d[ind](percent)
            if (upper_days - lower_days) < 0:
                return np.sqrt(lower_vol / days)
            implied_vol = np.sqrt((lower_vol + (upper_vol - lower_vol) /
                                   (upper_days - lower_days) * (days - lower_days)) / days)
            if implied_vol < 0:
                implied_vol = 0
            return implied_vol

        self.interp2d = interp2d

    def implied_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        days = self.calendar.count_business_days(
            self.valuation_date,
            expiration_date,
            include_start=False,
            include_end=True,
        )
        percent = strike / forward
        return self.interp2d(percent, days)

    def implied_vol_by_spot(self, spot: float, strike: float, expiration_date: date) -> float:
        days = self.calendar.count_business_days(
            self.valuation_date,
            expiration_date,
            include_start=False,
            include_end=True,
        )
        percent = strike / spot
        return self.interp2d(percent, days)

    def parallel_bump(self, amount: float) -> ImpliedVolSurface:
        new_vols = [[v + amount for v in row] for row in self.vols]
        return InterpolatedPercentStrikeImpliedVolSurface(valuation_date=self.valuation_date,
                                                          percent_strikes=self.percent_strikes,
                                                          business_days=self.business_days,
                                                          vols=new_vols,
                                                          calendar=self.calendar)

    def roll(self, new_valuation_date: date) -> ImpliedVolSurface:
        if new_valuation_date < self.valuation_date:
            raise ValueError(f'new valuation date {new_valuation_date.isoformat()}'
                             f'is before the original valuation date {self.valuation_date.isoformat()}')
        new_business_days = []
        new_vols = []
        n = self.calendar.count_business_days(
            self.valuation_date,
            new_valuation_date,
            include_start=False,
            include_end=True,
        )
        if n > self.business_days[-1]:
            raise ValueError(f'new valuation date {new_valuation_date.isoformat()} '
                             f'is after the last expiration date of the original vol surface')
        realized_var = [n * self.interp2d(percent, n) ** 2 for percent in self.percent_strikes]
        for idx, i in enumerate(self.business_days):
            if i > n:
                new_days = i - n
                new_business_days.append(new_days)
                new_vols.append([np.sqrt((self.vols[idx][j] ** 2 * i - realized_var[j]) / new_days)
                                 for j in range(len(realized_var))])
        if not new_vols:
            new_vols = self.vols[-1]
            new_business_days = [0]

        return InterpolatedPercentStrikeImpliedVolSurface(valuation_date=new_valuation_date,
                                                          percent_strikes=self.percent_strikes,
                                                          business_days=new_business_days,
                                                          vols=new_vols,
                                                          calendar=self.calendar)

    def shift_valuation_date(self, new_valuation_date: date) -> ImpliedVolSurface:
        return InterpolatedPercentStrikeImpliedVolSurface(valuation_date=new_valuation_date,
                                                          percent_strikes=self.percent_strikes,
                                                          business_days=self.business_days,
                                                          vols=self.vols,
                                                          calendar=self.calendar)


@dataclass
class InterpolatedPercentStrikeImpliedVolSurfaceByTenors(ImpliedVolSurface):
    tenors: List[str]
    percent_strikes: List[float]
    vols: List[List[float]]
    calendar: HolidayCalendar
    bus_adj: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    eom: EomConvention = EomConvention.NONE
    surface: InterpolatedPercentStrikeImpliedVolSurface = field(init=False)

    def __post_init__(self):
        periods = [period_from_string(s) for s in self.tenors]
        # TODO: type of all fields in surface/curve is string caused by __future__.annotation, need special conversion
        self.bus_adj = BusinessDayConvention.MODIFIED_FOLLOWING
        self.eom = EomConvention.NONE
        dates = [
            plus_period(
                self.valuation_date,
                p,
                adj=self.bus_adj,
                calendar=self.calendar,
                eom=self.eom,
            )
            for p in periods
        ]
        business_days = [
            self.calendar.count_business_days(
                self.valuation_date,
                d,
                include_start=False,
                include_end=True,
            )
            for d in dates
        ]
        for i in range(1, len(business_days)):
            if business_days[i] <= business_days[i - 1]:
                raise ValueError(f'期限必须递增[{self.tenors[i]}]<=[{self.tenors[i - 1]}]')
        self.surface = InterpolatedPercentStrikeImpliedVolSurface(valuation_date=self.valuation_date,
                                                                  percent_strikes=self.percent_strikes,
                                                                  business_days=business_days,
                                                                  vols=self.vols,
                                                                  calendar=self.calendar)

    def implied_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        return self.surface.implied_vol(forward, strike, expiration_date)

    def implied_vol_by_spot(self, spot: float, strike: float, expiration_date: date) -> float:
        return self.surface.implied_vol_by_spot(spot, strike, expiration_date)

    def parallel_bump(self, amount: float) -> ImpliedVolSurface:
        new_vols = [[v + amount for v in row] for row in self.vols]
        return InterpolatedPercentStrikeImpliedVolSurfaceByTenors(valuation_date=self.valuation_date,
                                                                  tenors=self.tenors,
                                                                  percent_strikes=self.percent_strikes,
                                                                  vols=new_vols,
                                                                  calendar=self.calendar,
                                                                  bus_adj=self.bus_adj,
                                                                  eom=self.eom)

    def roll(self, new_valuation_date: date) -> ImpliedVolSurface:
        return self.surface.roll(new_valuation_date)

    def shift_valuation_date(self, new_valuation_date: date) -> ImpliedVolSurface:
        return InterpolatedPercentStrikeImpliedVolSurfaceByTenors(valuation_date=new_valuation_date,
                                                                  tenors=self.tenors,
                                                                  percent_strikes=self.percent_strikes,
                                                                  vols=self.vols,
                                                                  calendar=self.calendar,
                                                                  bus_adj=self.bus_adj,
                                                                  eom=self.eom)


def create_term_vol_surface(valuation_date: date,
                            expiration_dates: List[date],
                            vols: List[float],
                            vol_calendar: DayCount) -> TermVolSurface:
    return TermVolSurface(valuation_date=valuation_date,
                          expiration_dates=expiration_dates,
                          vols=vols,
                          vol_daycount=vol_calendar)


def create_constant_vol_surface(valuation_date: date, vol: float,
                                vol_calendar: DayCount) -> TermVolSurface:
    return create_term_vol_surface(valuation_date, [valuation_date + timedelta(days=365)], [vol], vol_calendar)
