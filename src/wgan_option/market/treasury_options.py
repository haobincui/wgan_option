"""CBOT Treasury option expiration and underlying-futures conventions."""

from __future__ import annotations

import calendar as calendar_module
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Tuple
from zoneinfo import ZoneInfo

from quantlib.calendar.holidays import HolidayCalendar, create_calendar
from quantlib.calendar.utils import option_maturity_month_map, month_map


CHICAGO = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")
QUARTERLY_FUTURES_MONTHS = ((3, "H"), (6, "M"), (9, "U"), (12, "Z"))


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last_day = calendar_module.monthrange(year, month)[1]
    result = date(year, month, last_day)
    return result - timedelta(days=(result.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    # Gregorian computus, valid for the data years and deterministic offline rebuilds.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def cme_treasury_holidays(year: int) -> set[date]:
    """Return full-day CBOT Treasury closures used by the expiration rule."""

    holidays = {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Presidents Day
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed_fixed_holiday(year, 12, 25),
    }
    if year >= 2022:
        holidays.add(_observed_fixed_holiday(year, 6, 19))
    return holidays


def _holiday_span(years: Iterable[int]) -> set[date]:
    holidays: set[date] = set()
    for year in years:
        holidays.update(cme_treasury_holidays(int(year)))
    return holidays


_CME_TREASURY_CALENDAR_NAME = "CME_TREASURY_1990_2050"
_CME_TREASURY_CALENDAR: Optional[HolidayCalendar] = None


def cme_treasury_calendar() -> HolidayCalendar:
    global _CME_TREASURY_CALENDAR
    if _CME_TREASURY_CALENDAR is None:
        _CME_TREASURY_CALENDAR = create_calendar(
            _CME_TREASURY_CALENDAR_NAME,
            sorted(_holiday_span(range(1990, 2051))),
        )
    return _CME_TREASURY_CALENDAR


def cme_treasury_business_days(
    start: date,
    end: date,
    *,
    include_start: bool = False,
    include_end: bool = True,
) -> int:
    return cme_treasury_calendar().count_business_days(
        start,
        end,
        include_start=include_start,
        include_end=include_end,
    )


def _last_business_day(year: int, month: int) -> date:
    candidate = date(year, month, calendar_module.monthrange(year, month)[1])
    market_calendar = cme_treasury_calendar()
    while market_calendar.is_holiday(candidate):
        candidate -= timedelta(days=1)
    return candidate


def resolve_contract_year(
    year_code: int | str,
    *,
    trade_date: date,
    contract_month: int,
) -> int:
    """Resolve a one/two-digit contract year without consulting wall-clock time."""

    text = str(year_code).strip()
    if not text.isdigit() or len(text) not in {1, 2, 4}:
        raise ValueError(f"Unsupported contract year code: {year_code!r}")
    if len(text) == 4:
        year = int(text)
        if not 1900 <= year <= 2100:
            raise ValueError(f"Implausible contract year: {year}")
        return year

    modulus = 10 if len(text) == 1 else 100
    suffix = int(text)
    candidates = [
        year
        for year in range(trade_date.year - modulus, trade_date.year + modulus + 1)
        if year % modulus == suffix
    ]
    if not candidates:
        raise ValueError(f"Could not resolve contract year code: {year_code!r}")

    # Listed Treasury options should not be more than roughly two months past
    # their named month and should not be a decade into the future.
    lower_bound = trade_date - timedelta(days=75)
    plausible = [
        year
        for year in candidates
        if date(year, contract_month, calendar_module.monthrange(year, contract_month)[1])
        >= lower_bound
        and year <= trade_date.year + 3
    ]
    if not plausible:
        raise ValueError(
            "Contract year is inconsistent with trade date: "
            f"code={year_code}, trade_date={trade_date}, month={contract_month}"
        )
    return min(
        plausible,
        key=lambda year: abs(
            (
                date(year, contract_month, 15)
                - trade_date
            ).days
        ),
    )


def cme_treasury_option_last_trading_date(named_year: int, named_month: int) -> date:
    """Implement CBOT Rule 19A01.I for quarterly/serial Treasury options."""

    if not 1 <= int(named_month) <= 12:
        raise ValueError(f"Invalid named option month: {named_month}")
    if named_month == 1:
        prior_year, prior_month = named_year - 1, 12
    else:
        prior_year, prior_month = named_year, named_month - 1

    last_business_day = _last_business_day(prior_year, prior_month)
    candidate_friday = _last_weekday(prior_year, prior_month, 4)
    market_calendar = cme_treasury_calendar()

    def business_days_after(candidate: date) -> int:
        return market_calendar.count_business_days(
            candidate,
            last_business_day,
            include_start=False,
            include_end=True,
        )

    while business_days_after(candidate_friday) < 2:
        candidate_friday -= timedelta(days=7)

    last_trading_day = candidate_friday
    while market_calendar.is_holiday(last_trading_day):
        last_trading_day -= timedelta(days=1)
    return last_trading_day


def cme_treasury_option_last_trading_datetime(
    named_year: int,
    named_month: int,
) -> datetime:
    """Return the 4:00 p.m. Chicago close converted to UTC."""

    trading_date = cme_treasury_option_last_trading_date(named_year, named_month)
    chicago_close = datetime(
        trading_date.year,
        trading_date.month,
        trading_date.day,
        16,
        0,
        tzinfo=CHICAGO,
    )
    return chicago_close.astimezone(UTC)


def underlying_quarterly_future(named_year: int, named_month: int) -> Tuple[str, int]:
    for future_month, future_code in QUARTERLY_FUTURES_MONTHS:
        if named_month <= future_month:
            return future_code, named_year
    raise AssertionError("named_month validation should make this unreachable")


@dataclass(frozen=True)
class TyOptionContractDates:
    named_year: int
    named_month: int
    last_trading_date: date
    last_trading_datetime_utc: datetime
    underlying_future_month_code: str
    underlying_future_year: int
    contract_class: str = "quarterly_or_serial"


def resolve_ty_option_contract_dates(
    *,
    option_month_code: str,
    option_year_code: int | str,
    trade_date: date,
) -> TyOptionContractDates:
    normalized_code = str(option_month_code).strip().upper()
    month_name = option_maturity_month_map.get(normalized_code)
    if month_name is None:
        raise ValueError(
            "Unsupported TY option month code. Weekly product symbols require "
            f"an explicit parser and are not accepted: {option_month_code!r}"
        )
    named_month = int(month_map[month_name])
    named_year = resolve_contract_year(
        option_year_code,
        trade_date=trade_date,
        contract_month=named_month,
    )
    last_trading_datetime_utc = cme_treasury_option_last_trading_datetime(
        named_year,
        named_month,
    )
    future_month_code, future_year = underlying_quarterly_future(
        named_year,
        named_month,
    )
    return TyOptionContractDates(
        named_year=named_year,
        named_month=named_month,
        last_trading_date=last_trading_datetime_utc.date(),
        last_trading_datetime_utc=last_trading_datetime_utc,
        underlying_future_month_code=future_month_code,
        underlying_future_year=future_year,
    )
