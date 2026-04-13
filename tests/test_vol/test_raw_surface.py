import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlib.calendar.daycount import DayCountBusN
from quantlib.calendar.holidays import embedded_calendar
from quantlib.vol_surface.algo.raw_surface import RawVolSurface
from quantlib.vol_surface.surface import ImpliedVolSurface


def _add_business_days(calendar, start: date, business_days: int) -> date:
    current = start
    for _ in range(business_days):
        current = calendar.next(current)
    return current


class TestRawVolSurface(unittest.TestCase):
    def setUp(self):
        self.calendar = embedded_calendar()
        self.valuation_date = date(2023, 1, 2)
        self.daycount = DayCountBusN("BUS250", self.calendar, 250)

    def test_surface_recovers_input_points(self):
        surface = RawVolSurface(
            valuation_date=self.valuation_date,
            vols=[
                [0.24, 0.21, 0.2, 0.22],
                [0.29, 0.26, 0.25, 0.27],
            ],
            percent_strikes=[
                [0.8, 0.95, 1.05, 1.2],
                [0.8, 0.95, 1.05, 1.2],
            ],
            business_days=[21, 63],
            vol_daycount=self.daycount,
        )

        for business_day, percent_strikes, target_vols in zip(
            [21, 63],
            [[0.8, 0.95, 1.05, 1.2], [0.8, 0.95, 1.05, 1.2]],
            [[0.24, 0.21, 0.2, 0.22], [0.29, 0.26, 0.25, 0.27]],
        ):
            expiration_date = _add_business_days(self.calendar, self.valuation_date, business_day)
            for percent_strike, target_vol in zip(percent_strikes, target_vols):
                self.assertAlmostEqual(
                    surface.implied_vol(
                        forward=1.0,
                        strike=percent_strike,
                        expiration_date=expiration_date,
                    ),
                    target_vol,
                    delta=1e-10,
                )

    def test_surface_interpolates_linearly_and_clamps_boundaries(self):
        surface = RawVolSurface(
            valuation_date=self.valuation_date,
            vols=[[0.3, 0.2, 0.25]],
            percent_strikes=[[0.8, 1.0, 1.2]],
            business_days=[21],
            vol_daycount=self.daycount,
        )
        expiration_date = _add_business_days(self.calendar, self.valuation_date, 21)

        lower_var = 0.3 ** 2 * 21 / 250
        upper_var = 0.2 ** 2 * 21 / 250
        expected_mid_vol = np.sqrt((lower_var + (upper_var - lower_var) * 0.5) * 250 / 21)

        self.assertAlmostEqual(
            surface.implied_vol(forward=1.0, strike=0.9, expiration_date=expiration_date),
            expected_mid_vol,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            surface.implied_vol(forward=1.0, strike=0.7, expiration_date=expiration_date),
            0.3,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            surface.implied_vol(forward=1.0, strike=1.3, expiration_date=expiration_date),
            0.25,
            delta=1e-12,
        )

    def test_surface_interpolates_total_variance_between_terms(self):
        surface = RawVolSurface(
            valuation_date=self.valuation_date,
            vols=[
                [0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3],
            ],
            percent_strikes=[
                [0.9, 1.0, 1.1],
                [0.9, 1.0, 1.1],
            ],
            business_days=[10, 40],
            vol_daycount=self.daycount,
        )

        expiration_date = _add_business_days(self.calendar, self.valuation_date, 25)
        expected_var = 0.04 * 10 / 250 + (0.09 * 40 / 250 - 0.04 * 10 / 250) * (15 / 30)
        expected_vol = np.sqrt(expected_var * 250 / 25)

        self.assertAlmostEqual(
            surface.implied_vol(forward=1.0, strike=1.0, expiration_date=expiration_date),
            expected_vol,
            delta=1e-12,
        )

    def test_surface_is_implied_vol_surface_and_spot_alias_matches(self):
        surface = RawVolSurface(
            valuation_date=self.valuation_date,
            vols=[
                [0.24, 0.21, 0.2, 0.22],
                [0.29, 0.26, 0.25, 0.27],
            ],
            percent_strikes=[
                [0.8, 0.95, 1.05, 1.2],
                [0.8, 0.95, 1.05, 1.2],
            ],
            business_days=[21, 63],
            vol_daycount=self.daycount,
        )
        expiration_date = _add_business_days(self.calendar, self.valuation_date, 63)

        self.assertIsInstance(surface, ImpliedVolSurface)
        self.assertAlmostEqual(
            surface.implied_vol_by_spot(
                spot=1.0,
                strike=1.05,
                expiration_date=expiration_date,
            ),
            surface.implied_vol(
                forward=1.0,
                strike=1.05,
                expiration_date=expiration_date,
            ),
            delta=1e-12,
        )

    def test_surface_unsupported_surface_operations_raise(self):
        surface = RawVolSurface(
            valuation_date=self.valuation_date,
            vols=[[0.24, 0.21, 0.2, 0.22]],
            percent_strikes=[[0.8, 0.95, 1.05, 1.2]],
            business_days=[21],
            vol_daycount=self.daycount,
        )

        with self.assertRaises(NotImplementedError):
            surface.parallel_bump(0.01)
        with self.assertRaises(NotImplementedError):
            surface.roll(self.valuation_date)
        with self.assertRaises(NotImplementedError):
            surface.shift_valuation_date(self.valuation_date)
