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
from quantlib.vol_surface.surface import ImpliedVolSurface
from quantlib.vol_surface.algo.svi_surface import SviVolSurface, TermVolSurfaceByDays


def _add_business_days(calendar, start: date, business_days: int) -> date:
    current = start
    for _ in range(business_days):
        current = calendar.next(current)
    return current


class TestTermVolSurfaceByDays(unittest.TestCase):
    def setUp(self):
        self.calendar = embedded_calendar()
        self.valuation_date = date(2023, 1, 2)
        self.daycount = DayCountBusN("BUS250", self.calendar, 250)
        self.vols = [0.5, 0.4, 0.3, 0.2, 0.1]
        self.business_days = [5, 10, 21, 63, 125]
        self.variances = [
            vol * vol * business_day / 250
            for vol, business_day in zip(self.vols, self.business_days)
        ]
        self.surface = TermVolSurfaceByDays(
            valuation_date=self.valuation_date,
            vols=self.vols,
            business_days=self.business_days,
            vol_daycount=self.daycount,
        )

    def test_term_vol_surface_interpolates_variance_between_known_terms(self):
        expiration_date = _add_business_days(self.calendar, self.valuation_date, 7)
        var_target = self.variances[1] - (
            (self.variances[1] - self.variances[0]) / (self.business_days[1] - self.business_days[0])
        ) * (self.business_days[1] - 7)
        vol_target = np.sqrt(var_target * 250 / 7)

        vol = self.surface.implied_vol(expiration_date)

        self.assertAlmostEqual(vol, vol_target, delta=1e-14)

    def test_term_vol_surface_uses_front_slice_variance_below_first_term(self):
        expiration_date = _add_business_days(self.calendar, self.valuation_date, 3)
        vol_target = np.sqrt(self.variances[0] * 250 / 3)

        vol = self.surface.implied_vol(expiration_date)

        self.assertAlmostEqual(vol, vol_target, delta=1e-14)

    def test_term_vol_surface_extrapolates_past_last_term_with_last_slope(self):
        expiration_date = _add_business_days(self.calendar, self.valuation_date, 200)
        slope = (
            (self.variances[-1] - self.variances[-2])
            / (self.business_days[-1] - self.business_days[-2])
        )
        var_target = max(self.variances[-1] + slope * (200 - self.business_days[-1]), 0.0)
        vol_target = np.sqrt(var_target * 250 / 200)

        vol = self.surface.implied_vol(expiration_date)

        self.assertAlmostEqual(vol, vol_target, delta=1e-14)

    def test_term_vol_surface_validates_input_lengths(self):
        with self.assertRaises(ValueError):
            TermVolSurfaceByDays(
                valuation_date=self.valuation_date,
                vols=[0.2],
                business_days=[5, 10],
                vol_daycount=self.daycount,
            )


class TestSviVolSurfaceInterpolation(unittest.TestCase):
    def setUp(self):
        self.calendar = embedded_calendar()
        self.valuation_date = date(2023, 1, 2)
        self.daycount = DayCountBusN("BUS250", self.calendar, 250)
        self.vols = [0.5, 0.4, 0.3, 0.2, 0.1]
        self.business_days = [5, 10, 21, 63, 125]
        self.variances = [
            vol * vol * business_day / 250
            for vol, business_day in zip(self.vols, self.business_days)
        ]
        self.surface = SviVolSurface(
            valuation_date=self.valuation_date,
            svi_params={
                "a": self.variances,
                "b": [0.0] * len(self.business_days),
                "rho": [0.0] * len(self.business_days),
                "m": [0.0] * len(self.business_days),
                "sigma": [0.1] * len(self.business_days),
                "business_days": self.business_days,
            },
            vol_daycount=self.daycount,
        )

    def test_svi_surface_is_implied_vol_surface_and_spot_alias_matches(self):
        expiration_date = _add_business_days(self.calendar, self.valuation_date, 21)

        self.assertIsInstance(self.surface, ImpliedVolSurface)
        self.assertAlmostEqual(
            self.surface.implied_vol_by_spot(
                spot=1.0,
                strike=1.1,
                expiration_date=expiration_date,
            ),
            self.surface.implied_vol(
                forward=1.0,
                strike=1.1,
                expiration_date=expiration_date,
            ),
            delta=1e-14,
        )

    def test_svi_surface_unsupported_surface_operations_raise(self):
        with self.assertRaises(NotImplementedError):
            self.surface.parallel_bump(0.01)
        with self.assertRaises(NotImplementedError):
            self.surface.roll(self.valuation_date)
        with self.assertRaises(NotImplementedError):
            self.surface.shift_valuation_date(self.valuation_date)

    def test_svi_surface_interpolates_variance_between_terms(self):
        expiration_date = _add_business_days(self.calendar, self.valuation_date, 7)
        var_target = self.variances[1] - (
            (self.variances[1] - self.variances[0]) / (self.business_days[1] - self.business_days[0])
        ) * (self.business_days[1] - 7)
        vol_target = np.sqrt(var_target * 250 / 7)

        vol = self.surface.implied_vol(forward=1.0, strike=1.0, expiration_date=expiration_date)

        self.assertAlmostEqual(vol, vol_target, delta=1e-14)

    def test_svi_surface_clamps_to_front_slice_below_first_term(self):
        expiration_date = _add_business_days(self.calendar, self.valuation_date, 3)
        vol_target = np.sqrt(self.variances[0] * 250 / 3)

        vol = self.surface.implied_vol(forward=1.0, strike=1.0, expiration_date=expiration_date)

        self.assertAlmostEqual(vol, vol_target, delta=1e-14)

    def test_svi_surface_clamps_to_back_slice_above_last_term(self):
        expiration_date = _add_business_days(self.calendar, self.valuation_date, 200)
        vol_target = np.sqrt(self.variances[-1] * 250 / 200)

        vol = self.surface.implied_vol(forward=1.0, strike=1.0, expiration_date=expiration_date)

        self.assertAlmostEqual(vol, vol_target, delta=1e-14)

    def test_svi_surface_builds_expected_grid(self):
        surface = SviVolSurface(
            valuation_date=self.valuation_date,
            svi_params={
                "a": [
                    0.04 * self.business_days[0] / 250,
                    0.09 * self.business_days[1] / 250,
                ],
                "b": [0.0, 0.0],
                "rho": [0.0, 0.0],
                "m": [0.0, 0.0],
                "sigma": [0.1, 0.1],
                "business_days": self.business_days[:2],
            },
            vol_daycount=self.daycount,
        )

        vols = surface.implied_vol_surface(
            percent_strikes=[0.9, 1.0, 1.1],
            business_days=self.business_days[:2],
            forward=1.0,
        )

        self.assertEqual(len(vols), 2)
        self.assertEqual(len(vols[0]), 3)
        self.assertTrue(all(abs(v - 0.2) < 1e-14 for v in vols[0]))
        self.assertTrue(all(abs(v - 0.3) < 1e-14 for v in vols[1]))
