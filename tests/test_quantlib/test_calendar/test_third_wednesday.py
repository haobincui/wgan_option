import sys
import unittest
from datetime import date, datetime
from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").exists())
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlib.calendar.holidays import gbp_calendar, usd_calendar
from quantlib.calendar.schedule import get_third_wednesday_for_current_month


class TestThirdWednesday(unittest.TestCase):

    def test_third_wednesday(self):
        current_date_1 = date(2023, 10, 11)
        current_date_2 = date(2022, 5, 3)
        current_date_3 = datetime(2021, 1, 1, 0, 0, 0)
        current_date_4 = date(2022, 3, 1)
        current_date_5 = date(2022, 6, 1)
        current_dates = [current_date_1, current_date_2, current_date_3, current_date_4, current_date_5]


        target_date_1 = date(2023, 10, 18)
        target_date_2 = date(2022, 5, 18)
        target_date_3 = date(2021, 1, 20)
        target_date_4 = date(2022, 3, 16)
        target_date_5 = date(2022, 6, 15)
        target_dates = [target_date_1, target_date_2, target_date_3, target_date_4, target_date_5]

        calendars = [gbp_calendar(), usd_calendar()]

        res = [get_third_wednesday_for_current_month(day, calendars) for day in current_dates]

        for i, j in zip(res, target_dates):
            self.assertTrue(i == j)
