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
from quantlib.vol_surface.algo.sabr_algo import SabrCalibrationHagan
from quantlib.vol_surface.algo.sabr_surface import SabrVolSurface, hagan_lognormal_implied_vol


def _add_business_days(calendar, start: date, business_days: int) -> date:
    current = start
    for _ in range(business_days):
        current = calendar.next(current)
    return current


class TestSabrCalibration(unittest.TestCase):
    def setUp(self):
        self.calendar = embedded_calendar()
        self.valuation_date = date(2023, 1, 2)
        self.daycount = DayCountBusN("BUS250", self.calendar, 250)
        self.business_days = [21, 63]
        self.percent_strikes = [[0.8, 0.9, 1.0, 1.1, 1.2] for _ in self.business_days]

    def test_fixed_beta_calibration_recovers_input_slice_points(self):
        fixed_beta = 0.65
        vols = []
        slice_params = [(0.19, -0.25, 0.55), (0.24, -0.05, 0.8)]
        for business_day, (alpha, rho, nu) in zip(self.business_days, slice_params):
            maturity = business_day / 250.0
            vols.append(
                hagan_lognormal_implied_vol(
                    forward=1.0,
                    strike=np.asarray(self.percent_strikes[0], dtype=np.float64),
                    maturity=maturity,
                    alpha=alpha,
                    beta=fixed_beta,
                    rho=rho,
                    nu=nu,
                ).tolist()
            )

        calibration = SabrCalibrationHagan(
            valuation_date=self.valuation_date,
            vols=vols,
            percent_strikes=self.percent_strikes,
            business_days=self.business_days,
            vol_daycount=self.daycount,
            beta=fixed_beta,
        )
        surface = calibration.get_calibrated_vol_surface()

        for business_day, percent_strikes, target_vols in zip(self.business_days, self.percent_strikes, vols):
            expiration_date = _add_business_days(self.calendar, self.valuation_date, business_day)
            for percent_strike, target_vol in zip(percent_strikes, target_vols):
                calibrated_vol = surface.implied_vol(
                    forward=1.0,
                    strike=percent_strike,
                    expiration_date=expiration_date,
                )
                self.assertAlmostEqual(calibrated_vol, target_vol, delta=2e-3)

    def test_free_beta_calibration_recovers_input_slice_points(self):
        slice_params = [(0.18, 0.35, -0.2, 0.45), (0.23, 0.75, 0.15, 0.65)]
        vols = []
        for business_day, (alpha, beta, rho, nu) in zip(self.business_days, slice_params):
            maturity = business_day / 250.0
            vols.append(
                hagan_lognormal_implied_vol(
                    forward=1.0,
                    strike=np.asarray(self.percent_strikes[0], dtype=np.float64),
                    maturity=maturity,
                    alpha=alpha,
                    beta=beta,
                    rho=rho,
                    nu=nu,
                ).tolist()
            )

        calibration = SabrCalibrationHagan(
            valuation_date=self.valuation_date,
            vols=vols,
            percent_strikes=self.percent_strikes,
            business_days=self.business_days,
            vol_daycount=self.daycount,
            beta=None,
        )
        surface = calibration.get_calibrated_vol_surface()

        for business_day, percent_strikes, target_vols in zip(self.business_days, self.percent_strikes, vols):
            expiration_date = _add_business_days(self.calendar, self.valuation_date, business_day)
            for percent_strike, target_vol in zip(percent_strikes, target_vols):
                calibrated_vol = surface.implied_vol(
                    forward=1.0,
                    strike=percent_strike,
                    expiration_date=expiration_date,
                )
                self.assertAlmostEqual(calibrated_vol, target_vol, delta=5e-3)

    def test_get_calibrated_vol_matches_surface_output(self):
        calibration = SabrCalibrationHagan(
            valuation_date=self.valuation_date,
            vols=[
                [0.21, 0.2, 0.19, 0.2, 0.21],
                [0.24, 0.23, 0.22, 0.23, 0.24],
            ],
            percent_strikes=self.percent_strikes,
            business_days=self.business_days,
            vol_daycount=self.daycount,
            beta=1.0,
        )
        surface = calibration.get_calibrated_vol_surface()
        expiration_date = _add_business_days(self.calendar, self.valuation_date, 63)

        self.assertAlmostEqual(
            calibration.get_calibrated_vol(
                forward=1.0,
                strike=1.1,
                expiration_date=expiration_date,
            ),
            surface.implied_vol(
                forward=1.0,
                strike=1.1,
                expiration_date=expiration_date,
            ),
            delta=1e-12,
        )

    def test_sabr_surface_interpolates_and_clamps_in_time(self):
        surface = SabrVolSurface(
            valuation_date=self.valuation_date,
            sabr_params={
                "alpha": [0.2, 0.3],
                "beta": [1.0, 1.0],
                "rho": [0.0, 0.0],
                "nu": [0.0, 0.0],
                "business_days": [5, 20],
            },
            vol_daycount=self.daycount,
        )

        front_date = _add_business_days(self.calendar, self.valuation_date, 3)
        mid_date = _add_business_days(self.calendar, self.valuation_date, 10)
        back_date = _add_business_days(self.calendar, self.valuation_date, 40)

        self.assertAlmostEqual(
            surface.implied_vol(forward=1.0, strike=1.0, expiration_date=front_date),
            0.2,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            surface.implied_vol(forward=1.0, strike=1.0, expiration_date=mid_date),
            0.23333333333333334,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            surface.implied_vol(forward=1.0, strike=1.0, expiration_date=back_date),
            0.3,
            delta=1e-12,
        )

    def test_sabr_surface_is_implied_vol_surface_and_spot_alias_matches(self):
        surface = SabrVolSurface(
            valuation_date=self.valuation_date,
            sabr_params={
                "alpha": [0.2, 0.3],
                "beta": [1.0, 1.0],
                "rho": [0.0, 0.0],
                "nu": [0.0, 0.0],
                "business_days": [5, 20],
            },
            vol_daycount=self.daycount,
        )
        expiration_date = _add_business_days(self.calendar, self.valuation_date, 20)

        self.assertIsInstance(surface, ImpliedVolSurface)
        self.assertAlmostEqual(
            surface.implied_vol_by_spot(
                spot=1.0,
                strike=1.1,
                expiration_date=expiration_date,
            ),
            surface.implied_vol(
                forward=1.0,
                strike=1.1,
                expiration_date=expiration_date,
            ),
            delta=1e-12,
        )

    def test_sabr_surface_unsupported_surface_operations_raise(self):
        surface = SabrVolSurface(
            valuation_date=self.valuation_date,
            sabr_params={
                "alpha": [0.2, 0.3],
                "beta": [1.0, 1.0],
                "rho": [0.0, 0.0],
                "nu": [0.0, 0.0],
                "business_days": [5, 20],
            },
            vol_daycount=self.daycount,
        )

        with self.assertRaises(NotImplementedError):
            surface.parallel_bump(0.01)
        with self.assertRaises(NotImplementedError):
            surface.roll(self.valuation_date)
        with self.assertRaises(NotImplementedError):
            surface.shift_valuation_date(self.valuation_date)

    def test_calibration_is_deterministic(self):
        vols = [
            [0.21, 0.2, 0.19, 0.2, 0.21],
            [0.24, 0.23, 0.22, 0.23, 0.24],
        ]
        calibration_one = SabrCalibrationHagan(
            valuation_date=self.valuation_date,
            vols=vols,
            percent_strikes=self.percent_strikes,
            business_days=self.business_days,
            vol_daycount=self.daycount,
            beta=0.6,
        )
        calibration_two = SabrCalibrationHagan(
            valuation_date=self.valuation_date,
            vols=vols,
            percent_strikes=self.percent_strikes,
            business_days=self.business_days,
            vol_daycount=self.daycount,
            beta=0.6,
        )

        self.assertEqual(calibration_one.params, calibration_two.params)
