import sys
import unittest
from datetime import datetime, time, timezone
from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").exists())
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlib.calendar.daycount import DayCountBusN
from quantlib.calendar.holidays import usd_calendar


def _to_utc(dt_value: datetime) -> datetime:
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=timezone.utc)
    return dt_value.astimezone(timezone.utc)


def get_tau(quote_time: datetime, maturity_time: datetime, daycount: DayCountBusN) -> float:
    quote_time = _to_utc(quote_time)
    maturity_time = _to_utc(maturity_time)
    base_tau = daycount(quote_time.date(), maturity_time.date())
    seconds_per_day = 86400.0

    quote_fraction = (
        quote_time - datetime.combine(quote_time.date(), time(0, 0, 0, tzinfo=timezone.utc))
    ).total_seconds() / seconds_per_day
    maturity_fraction = (
        maturity_time - datetime.combine(maturity_time.date(), time(0, 0, 0, tzinfo=timezone.utc))
    ).total_seconds() / seconds_per_day

    return float(base_tau + (maturity_fraction - quote_fraction) / float(daycount.days_in_year))


class TestTau(unittest.TestCase):
    def setUp(self):
        self.calendar = usd_calendar()
        self.bus_250_usd = DayCountBusN("BUS250USD", self.calendar, 250)
        self.bus_365_usd = DayCountBusN("BUS365USD", self.calendar, 365)

    def test_same_day_tau_uses_intraday_fraction_only(self):
        quote_time = datetime(2025, 11, 14, 9, 15, 0, tzinfo=timezone.utc)
        maturity_time = datetime(2025, 11, 14, 20, 0, 0, tzinfo=timezone.utc)

        tau = get_tau(quote_time, maturity_time, self.bus_250_usd)
        expected = ((20 * 3600) - (9 * 3600 + 15 * 60)) / 86400.0 / 250.0

        self.assertAlmostEqual(tau, expected, places=12)
        self.assertGreater(tau, 0.0)

    def test_cross_weekend_tau_counts_business_day_and_intraday_fraction(self):
        quote_time = datetime(2025, 11, 14, 18, 3, 33, tzinfo=timezone.utc)
        maturity_time = datetime(2025, 11, 17, 20, 0, 0, tzinfo=timezone.utc)

        tau = get_tau(quote_time, maturity_time, self.bus_250_usd)
        expected = 1.0 / 250.0 + (((20 * 3600) - (18 * 3600 + 3 * 60 + 33)) / 86400.0) / 250.0

        self.assertAlmostEqual(tau, expected, places=12)

    def test_busday_tau_differs_from_act365_seconds_formula(self):
        quote_time = datetime(2025, 11, 14, 18, 3, 33, tzinfo=timezone.utc)
        maturity_time = datetime(2025, 11, 17, 20, 0, 0, tzinfo=timezone.utc)

        tau_bus = get_tau(quote_time, maturity_time, self.bus_250_usd)
        tau_act365 = (maturity_time - quote_time).total_seconds() / 86400.0 / 365.0

        self.assertNotAlmostEqual(tau_bus, tau_act365, places=12)
        self.assertLess(tau_bus, tau_act365)

    def test_tau_is_non_positive_after_maturity(self):
        quote_time = datetime(2025, 11, 14, 20, 0, 1, tzinfo=timezone.utc)
        maturity_time = datetime(2025, 11, 14, 20, 0, 0, tzinfo=timezone.utc)

        tau = get_tau(quote_time, maturity_time, self.bus_250_usd)

        self.assertLessEqual(tau, 0.0)

    def test_days_in_year_parameter_changes_tau_scale(self):
        quote_time = datetime(2025, 11, 14, 9, 15, 0, tzinfo=timezone.utc)
        maturity_time = datetime(2025, 11, 14, 20, 0, 0, tzinfo=timezone.utc)

        tau_250 = get_tau(quote_time, maturity_time, self.bus_250_usd)
        tau_365 = get_tau(quote_time, maturity_time, self.bus_365_usd)

        self.assertGreater(tau_250, tau_365)
        self.assertAlmostEqual(tau_250 / tau_365, 365.0 / 250.0, places=12)


if __name__ == "__main__":
    unittest.main()
