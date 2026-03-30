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

from quantlib.vol_surface.algo import svi_algo as svi_algo_module
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
        self.daycount = DayCountBusN(name="BUS250", calendar=self.calendar, days_in_year=250)
        self.vols = [[0.22, 0.20, 0.23], [0.24, 0.21, 0.25]]
        self.percent_strikes = [[0.9, 1.0, 1.1], [0.9, 1.0, 1.1]]
        self.business_days = [5, 21]
        self.boundary_fixture_vols = [[
            0.049948226660490036,
            0.057753950357437134,
            0.06100400164723396,
            0.060941558331251144,
            0.059292372316122055,
            0.056824296712875366,
            0.05666149780154228,
            0.06146609038114548,
        ]]
        self.boundary_fixture_percent_strikes = [[
            0.9548014455286857,
            0.9639383014667593,
            0.9867804413119431,
            0.9913488692809799,
            0.9959172972500167,
            1.0073383671726086,
            1.0164752231106822,
            1.0484542188939396,
        ]]
        self.boundary_fixture_business_days = [38]

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

    def test_quasi_explicit_calibration_is_deterministic_without_rng_seed(self):
        calibration_one = SviCalibrationQuasiExplicit(
            valuation_date=self.valuation_date,
            vols=self.vols,
            percent_strikes=self.percent_strikes,
            business_days=self.business_days,
            vol_daycount=self.daycount,
        )
        calibration_two = SviCalibrationQuasiExplicit(
            valuation_date=self.valuation_date,
            vols=self.vols,
            percent_strikes=self.percent_strikes,
            business_days=self.business_days,
            vol_daycount=self.daycount,
        )

        self.assertEqual(calibration_one.params, calibration_two.params)

    def test_boundary_fixture_no_longer_returns_exact_rho_or_sigma_floor(self):
        calibration = SviCalibrationQuasiExplicit(
            valuation_date=self.valuation_date,
            vols=self.boundary_fixture_vols,
            percent_strikes=self.boundary_fixture_percent_strikes,
            business_days=self.boundary_fixture_business_days,
            vol_daycount=self.daycount,
        )

        self.assertEqual(calibration.params["business_days"], self.boundary_fixture_business_days)
        self.assertTrue(all(abs(rho) < 0.999 for rho in calibration.params["rho"]))
        self.assertTrue(all(abs(sigma - 1e-4) > 1e-8 for sigma in calibration.params["sigma"]))

    def test_boundary_hit_uses_fallback_and_updates_stats(self):
        stage1 = svi_algo_module._CalibrationStageResult(
            params=[1e-3, 0.02, 1.0, 0.0, 1e-4],
            objective=1.0,
            vol_objective=1.0,
            hard_valid=True,
            boundary_hit=True,
            method="qls-stage1",
        )
        fallback = svi_algo_module._CalibrationStageResult(
            params=[1e-3, 0.02, 0.25, 0.0, 0.02],
            objective=1.02,
            vol_objective=1.00001,
            hard_valid=True,
            boundary_hit=False,
            method="qls-direct-fallback",
        )
        stats = {}

        with patch.object(svi_algo_module, "_calibration_function_qls", return_value=stage1):
            with patch.object(svi_algo_module, "_direct_svi_fallback_fit", return_value=fallback):
                calibration = SviCalibrationQuasiExplicit(
                    valuation_date=self.valuation_date,
                    vols=[self.vols[0]],
                    percent_strikes=[self.percent_strikes[0]],
                    business_days=[self.business_days[0]],
                    vol_daycount=self.daycount,
                    stats=stats,
                )

        self.assertEqual(calibration.params["rho"], [0.25])
        self.assertEqual(calibration.params["sigma"], [0.02])
        self.assertEqual(stats["qls_boundary_retry_slices"], 1)
        self.assertEqual(stats["qls_fallback_attempt_slices"], 1)
        self.assertEqual(stats["qls_fallback_success_slices"], 1)
        self.assertEqual(stats.get("qls_boundary_reject_slices", 0), 0)
        self.assertEqual(stats.get("qls_stage1_kept_slices", 0), 0)
