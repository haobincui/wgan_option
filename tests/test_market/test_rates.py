from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from wgan_option.market.rates import TreasuryParYieldCurve


class TreasuryParYieldCurveTests(unittest.TestCase):
    def _curve(self, directory: str, *, max_staleness_days: int = 7) -> TreasuryParYieldCurve:
        path = Path(directory) / "curve.csv"
        path.write_text(
            "\n".join(
                [
                    "Date,1 Mo,3 Mo,6 Mo,1 Yr",
                    "01/03/2022,0.10,0.20,0.30,0.40",
                    "01/05/2022,1.10,1.20,1.30,1.40",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return TreasuryParYieldCurve(path, max_staleness_days=max_staleness_days)

    def test_uses_latest_curve_on_or_before_valuation_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            curve = self._curve(directory)
            point = curve.point(date(2022, 1, 4), 0.25)
            self.assertEqual(point.curve_date, date(2022, 1, 3))
            self.assertAlmostEqual(point.par_yield_percent, 0.20)
            self.assertGreater(point.discount_factor, 0.0)
            self.assertLessEqual(point.discount_factor, 1.0)

    def test_does_not_read_future_curve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            curve = self._curve(directory)
            with self.assertRaisesRegex(ValueError, "No Treasury curve"):
                curve.point(date(2022, 1, 2), 0.25)

    def test_stale_curve_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            curve = self._curve(directory, max_staleness_days=1)
            with self.assertRaisesRegex(ValueError, "too stale"):
                curve.point(date(2022, 1, 7), 0.25)


if __name__ == "__main__":
    unittest.main()
