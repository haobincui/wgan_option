from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wgan_option.market.treasury_sessions import (  # noqa: E402
    TreasuryGlobexSessionCalendar,
)


class TreasuryGlobexSessionCalendarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calendar = TreasuryGlobexSessionCalendar.from_csv(
            ROOT_DIR
            / "data/reference/"
            "cme_treasury_globex_closures_2022_2023.csv"
        )

    def test_regular_week_and_daily_halt(self):
        self.assertTrue(self.calendar.is_open("2023-07-05T20:59:00Z"))
        self.assertFalse(self.calendar.is_open("2023-07-05T21:30:00Z"))
        self.assertEqual(
            self.calendar.closed_reason("2023-07-05T21:30:00Z"),
            "daily_halt",
        )
        self.assertEqual(
            self.calendar.next_open("2023-07-05T21:30:00Z"),
            pd.Timestamp("2023-07-05T22:00:00Z"),
        )
        self.assertFalse(self.calendar.is_open("2023-07-08T14:00:00Z"))

    def test_dst_changes_sunday_open_in_utc(self):
        self.assertFalse(self.calendar.is_open("2023-03-12T21:59:00Z"))
        self.assertTrue(self.calendar.is_open("2023-03-12T22:00:00Z"))
        self.assertFalse(self.calendar.is_open("2023-11-05T22:59:00Z"))
        self.assertTrue(self.calendar.is_open("2023-11-05T23:00:00Z"))

    def test_holiday_override_splits_regular_session(self):
        self.assertTrue(self.calendar.is_open("2023-11-23T17:59:00Z"))
        self.assertFalse(self.calendar.is_open("2023-11-23T18:00:00Z"))
        self.assertEqual(
            self.calendar.closed_reason("2023-11-23T18:00:00Z"),
            "holiday:Thanksgiving Day",
        )
        self.assertEqual(
            self.calendar.next_open("2023-11-23T18:00:00Z"),
            pd.Timestamp("2023-11-23T23:00:00Z"),
        )

    def test_full_surface_pair_must_remain_in_one_session(self):
        self.assertIsNone(
            self.calendar.session_for_interval(
                "2023-03-13T21:59:00Z",
                "2023-03-13T22:09:00Z",
            )
        )
        session = self.calendar.session_for_interval(
            "2023-03-13T22:00:00Z",
            "2023-03-13T22:10:00Z",
        )
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(
            session.open_utc,
            pd.Timestamp("2023-03-13T22:00:00Z"),
        )


if __name__ == "__main__":
    unittest.main()
