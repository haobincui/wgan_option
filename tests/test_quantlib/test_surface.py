import sys
import unittest
from datetime import date
from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").exists())
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlib.calendar.holidays import embedded_calendar
from quantlib.vol_surface.surface import InterpolatedImpliedVolSurface


class TestSurface(unittest.TestCase):
    def test_vol_surface(self):
        vols = [[0.3, 0.2, 0.5], [0.2, 0.1, 0.3]]
        moneynesses = [0.8, 1, 1.2]
        ttms = [5, 21]
        valuation_date = date(2023, 1,23)
        vol_surface = InterpolatedImpliedVolSurface(vols=vols,
                                                    percent_strikes=moneynesses,
                                                    business_days=ttms,
                                                    calendar=embedded_calendar(),
                                                    valuation_date=valuation_date)

        moneyness = 1
        ttm = 5
        target_vol = 0.2
        expiration_date = date(2023, 1, 27)
        res = vol_surface.implied_vol(1, 1, expiration_date)
        self.assertAlmostEqual(target_vol, res, delta = 1e-14)
