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
from quantlib.vol_surface.algo.svi_algo import SviCalibrationQuasiExplicit


def _add_business_days(calendar, start: date, business_days: int) -> date:
    current = start
    for _ in range(business_days):
        current = calendar.next(current)
    return current


class TestSviCalibration(unittest.TestCase):
    def setUp(self):
        self.calendar = embedded_calendar()
        self.valuation_date = date(2023, 1, 2)
        self.daycount = DayCountBusN("BUS250", self.calendar, 250)
        self.vols = [[0.22, 0.20, 0.23], [0.24, 0.21, 0.25]]
        self.percent_strikes = [[0.9, 1.0, 1.1], [0.9, 1.0, 1.1]]
        self.business_days = [5, 21]

    def test_quasi_explicit_calibration_recovers_input_slice_points(self):
        np.random.seed(0)
        calibration = SviCalibrationQuasiExplicit(
            valuation_date=self.valuation_date,
            vols=self.vols,
            percent_strikes=self.percent_strikes,
            business_days=self.business_days,
            vol_daycount=self.daycount,
        )
        surface = calibration.get_calibrated_vol_surface()

        for business_day, percent_strikes, target_vols in zip(
            self.business_days, self.percent_strikes, self.vols
        ):
            expiration_date = _add_business_days(
                self.calendar, self.valuation_date, business_day
            )
            for percent_strike, target_vol in zip(percent_strikes, target_vols):
                with self.subTest(
                    business_day=business_day,
                    percent_strike=percent_strike,
                ):
                    calibrated_vol = surface.implied_vol(
                        forward=1.0,
                        strike=percent_strike,
                        expiration_date=expiration_date,
                    )
                    self.assertAlmostEqual(calibrated_vol, target_vol, delta=1e-6)

    def test_get_calibrated_vol_matches_surface_output(self):
        np.random.seed(0)
        calibration = SviCalibrationQuasiExplicit(
            valuation_date=self.valuation_date,
            vols=self.vols,
            percent_strikes=self.percent_strikes,
            business_days=self.business_days,
            vol_daycount=self.daycount,
        )
        surface = calibration.get_calibrated_vol_surface()

        expiration_date = _add_business_days(self.calendar, self.valuation_date, 21)
        via_surface = surface.implied_vol(
            forward=1.0,
            strike=1.1,
            expiration_date=expiration_date,
        )
        via_calibration = calibration.get_calibrated_vol(
            forward=1.0,
            strike=1.1,
            expiration_date=expiration_date,
        )

        self.assertAlmostEqual(via_calibration, via_surface, delta=1e-10)

    def test_calibration_exposes_complete_param_keys(self):
        np.random.seed(0)
        calibration = SviCalibrationQuasiExplicit(
            valuation_date=self.valuation_date,
            vols=self.vols,
            percent_strikes=self.percent_strikes,
            business_days=self.business_days,
            vol_daycount=self.daycount,
        )

        self.assertEqual(
            set(calibration.params.keys()),
            {"a", "b", "rho", "m", "sigma", "business_days"},
        )
        self.assertEqual(calibration.params["business_days"], self.business_days)
