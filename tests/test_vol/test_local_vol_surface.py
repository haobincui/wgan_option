import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlib.calendar.daycount import DayCountBusN
from quantlib.calendar.holidays import embedded_calendar
from quantlib.vol_surface.algo.local_vol_algo import LocalVolSurfaceBuilder


def _add_business_days(calendar, start: date, business_days: int) -> date:
    current = start
    for _ in range(business_days):
        current = calendar.next(current)
    return current


class TestLocalVolSurfaceBuilder(unittest.TestCase):
    def setUp(self):
        self.calendar = embedded_calendar()
        self.valuation_date = date(2023, 1, 2)
        self.daycount = DayCountBusN("BUS250", self.calendar, 250)
        self.target_percent_strikes = np.linspace(0.8, 1.2, 9).tolist()
        self.target_business_days = [21, 42, 63, 84, 105, 126, 147]

    def _constant_builder(self) -> LocalVolSurfaceBuilder:
        vols = [[0.2, 0.2, 0.2, 0.2, 0.2] for _ in self.target_business_days]
        percent_strikes = [[0.8, 0.9, 1.0, 1.1, 1.2] for _ in self.target_business_days]
        business_days = list(self.target_business_days)
        return LocalVolSurfaceBuilder(
            valuation_date=self.valuation_date,
            vols=vols,
            percent_strikes=percent_strikes,
            business_days=business_days,
            target_percent_strikes=self.target_percent_strikes,
            target_business_days=self.target_business_days,
            vol_daycount=self.daycount,
        )

    def test_interpolated_implied_vol_grid_matches_slice_interpolation(self):
        target_percent_strikes = [0.8, 0.9, 1.0, 1.1, 1.2]
        target_business_days = [21, 63, 126]
        builder = LocalVolSurfaceBuilder(
            valuation_date=self.valuation_date,
            vols=[
                [0.21, 0.2, 0.22],
                [0.24, 0.23, 0.25],
                [0.28, 0.27, 0.29],
            ],
            percent_strikes=[
                [0.8, 1.0, 1.2],
                [0.85, 1.0, 1.2],
                [0.8, 0.95, 1.15],
            ],
            business_days=target_business_days,
            target_percent_strikes=target_percent_strikes,
            target_business_days=target_business_days,
            vol_daycount=self.daycount,
        )

        grid = builder.get_interpolated_implied_vol_grid()

        self.assertEqual(grid.shape, (3, 5))
        expected_front = np.interp(
            target_percent_strikes,
            [0.8, 1.0, 1.2],
            [0.21, 0.2, 0.22],
            left=0.21,
            right=0.22,
        )
        np.testing.assert_allclose(grid[0], expected_front, atol=1e-12)

    def test_constant_implied_vol_surface_produces_nearly_constant_interior_local_vol(self):
        builder = self._constant_builder()
        surface = builder.get_local_vol_surface()
        local_vol_grid = np.asarray(
            surface.local_vol_surface(
                percent_strikes=self.target_percent_strikes,
                business_days=self.target_business_days,
            ),
            dtype=np.float64,
        )

        interior = local_vol_grid[1:-1, 1:-1]
        self.assertTrue(np.all(np.isfinite(interior)))
        self.assertLess(float(np.mean(np.abs(interior - 0.2))), 2e-2)
        self.assertLess(float(np.max(np.abs(interior - 0.2))), 5e-2)

    def test_get_local_vol_matches_surface_output(self):
        builder = self._constant_builder()
        surface = builder.get_local_vol_surface()
        expiration_date = _add_business_days(self.calendar, self.valuation_date, 63)

        self.assertAlmostEqual(
            builder.get_local_vol(
                forward=1.0,
                strike=1.0,
                expiration_date=expiration_date,
            ),
            surface.local_vol(
                forward=1.0,
                strike=1.0,
                expiration_date=expiration_date,
            ),
            delta=1e-12,
        )

    def test_invalid_dupire_points_are_filled_from_nearest_valid_cells(self):
        builder = self._constant_builder()
        patched_local_var = np.asarray(
            [
                [0.04, np.nan, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
                [0.04, 0.04, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
                [0.04, 0.04, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
                [0.04, 0.04, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
                [0.04, 0.04, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
                [0.04, 0.04, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
                [0.04, 0.04, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
            ],
            dtype=np.float64,
        )
        patched_local_vol = np.sqrt(np.where(np.isfinite(patched_local_var), patched_local_var, np.nan))

        with patch(
            "quantlib.vol_surface.algo.local_vol_algo.dupire_local_vol_from_call_surface",
            return_value=(patched_local_vol, patched_local_var),
        ):
            surface = builder.get_local_vol_surface()

        self.assertTrue(np.all(np.isfinite(surface.local_vol_grid)))
        self.assertTrue(surface.local_vol_grid[0, 1] > 0)

    def test_fully_invalid_dupire_surface_raises(self):
        builder = self._constant_builder()
        invalid_local_var = np.full((7, 9), np.nan, dtype=np.float64)
        invalid_local_vol = np.full((7, 9), np.nan, dtype=np.float64)

        with patch(
            "quantlib.vol_surface.algo.local_vol_algo.dupire_local_vol_from_call_surface",
            return_value=(invalid_local_vol, invalid_local_var),
        ):
            with self.assertRaises(ValueError):
                builder.get_local_vol_surface()
