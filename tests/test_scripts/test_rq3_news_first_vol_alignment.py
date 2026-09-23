from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from scripts.rq3.news_first_vol_alignment import (  # noqa: E402
    DEFAULT_TOLERANCES_MINUTES,
    build_news_first_alignments,
    prepare_pair_universe,
    validate_news_first_alignments,
)
from wgan_option.market.treasury_sessions import (  # noqa: E402
    TreasuryGlobexSessionCalendar,
)


class NewsFirstVolAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.calendar = TreasuryGlobexSessionCalendar()

    def _pair(self, pair_id: str, origin: str, **extra):
        origin_at = pd.Timestamp(origin)
        target_at = origin_at + pd.Timedelta(minutes=5)
        session = self.calendar.session_for_interval(
            origin_at - pd.Timedelta(minutes=5), target_at
        )
        self.assertIsNotNone(session)
        return {
            "pair_id": pair_id,
            "origin_time_utc": origin_at,
            "target_time_utc": target_at,
            "session_id": session.session_id,
            "same_cme_continuous_session": True,
            "metric_status": "ok",
            "maturity_date": "2023-06-30",
            **extra,
        }

    @staticmethod
    def _news(times):
        return pd.DataFrame(
            {
                "news_row_id": range(1, len(times) + 1),
                "sample_id": [f"news_{index}" for index in range(1, len(times) + 1)],
                "news_available_time_utc": times,
                "publication_timestamp_utc": times,
                "timestamp_parse_status": ["ok"] * len(times),
            }
        )

    def test_cumulative_tolerances_use_earliest_pair_and_inclusive_boundary(self):
        pairs = pd.DataFrame(
            [
                self._pair("pair_3", "2023-03-14T14:03:00Z"),
                self._pair("pair_5", "2023-03-14T14:15:00Z"),
                self._pair("pair_8", "2023-03-14T14:26:00Z"),
                self._pair("pair_15", "2023-03-14T14:50:00Z"),
            ]
        )
        news = self._news(
            [
                "2023-03-14T14:00:00Z",  # +3
                "2023-03-14T14:10:00Z",  # +5, inclusive
                "2023-03-14T14:18:00Z",  # +8
                "2023-03-14T14:35:00Z",  # +15, inclusive
                "2023-03-14T14:30:00Z",  # +20, inclusive
                "2023-03-14T14:51:00Z",  # no later pair
            ]
        )

        alignments = build_news_first_alignments(
            news,
            pairs,
            session_calendar=self.calendar,
            tolerances=DEFAULT_TOLERANCES_MINUTES,
        )

        self.assertEqual(
            [
                int(alignments[value]["has_match"].sum())
                for value in DEFAULT_TOLERANCES_MINUTES
            ],
            [2, 3, 4, 5, 5],
        )
        five = alignments[5].set_index("news_row_id")
        self.assertEqual(five.at[1, "pair_id"], "pair_3")
        self.assertEqual(int(five.at[1, "origin_tolerance_minutes_used"]), 3)
        self.assertEqual(int(five.at[2, "origin_tolerance_minutes_used"]), 5)
        self.assertEqual(int(five.at[2, "first_included_tolerance_minutes"]), 5)
        self.assertEqual(int(five.at[3, "first_included_tolerance_minutes"]), 10)
        self.assertEqual(int(five.at[5, "first_included_tolerance_minutes"]), 20)
        self.assertEqual(five.at[1, "window_relation"], "current_partially_post_news")
        self.assertEqual(int(five.at[1, "post_news_current_overlap_minutes"]), 3)
        self.assertEqual(five.at[2, "window_relation"], "current_fully_post_news")
        self.assertEqual(int(five.at[2, "post_news_current_overlap_minutes"]), 5)

        ten = alignments[10].set_index("news_row_id")
        self.assertEqual(ten.at[3, "pair_id"], "pair_8")
        self.assertEqual(int(ten.at[3, "first_included_tolerance_minutes"]), 10)
        fifteen = alignments[15].set_index("news_row_id")
        self.assertEqual(int(fifteen.at[4, "first_included_tolerance_minutes"]), 15)
        twenty = alignments[20].set_index("news_row_id")
        self.assertEqual(int(twenty.at[5, "first_included_tolerance_minutes"]), 20)
        self.assertEqual(twenty.at[5, "pair_id"], "pair_15")
        thirty = alignments[30].set_index("news_row_id")
        self.assertEqual(int(thirty.at[5, "first_included_tolerance_minutes"]), 20)
        self.assertEqual(
            thirty.at[6, "unmatched_reason"],
            "no_valid_pair_within_origin_tolerance",
        )

        # Once a pair enters a cumulative sample, widening the tolerance cannot
        # change it to a later market outcome.
        for news_id in (1, 2):
            selections = {
                frame.set_index("news_row_id").at[news_id, "pair_id"]
                for frame in alignments.values()
            }
            self.assertEqual(len(selections), 1)

    def test_exact_open_news_has_clean_pre_post_window(self):
        pairs = pd.DataFrame([self._pair("pair_exact", "2023-03-14T14:00:00Z")])
        aligned = build_news_first_alignments(
            self._news(["2023-03-14T14:00:00Z"]),
            pairs,
            session_calendar=self.calendar,
        )[5].iloc[0]

        self.assertEqual(aligned["alignment_type"], "exact")
        self.assertEqual(
            aligned["window_relation"], "current_pre_news_target_post_news"
        )
        self.assertEqual(int(aligned["post_news_current_overlap_minutes"]), 0)
        self.assertEqual(int(aligned["training_eligible"]), 1)
        self.assertEqual(aligned["target_anchor_utc"], "2023-03-14T14:05:00Z")

    def test_closed_news_gets_auditable_candidate_but_is_never_training_eligible(self):
        # Chicago is on daylight time: the daily halt is 21:00-22:00 UTC.
        pairs = pd.DataFrame([self._pair("pair_open", "2023-03-13T22:05:00Z")])
        aligned = build_news_first_alignments(
            self._news(["2023-03-13T21:30:00Z"]),
            pairs,
            session_calendar=self.calendar,
        )[5].iloc[0]

        self.assertEqual(int(aligned["has_match"]), 1)
        self.assertEqual(int(aligned["training_eligible"]), 0)
        self.assertEqual(
            aligned["training_exclusion_reason"], "closed_market_publication"
        )
        self.assertEqual(aligned["publication_market_state"], "closed")
        self.assertEqual(aligned["scheduled_origin_utc"], "2023-03-13T22:00:00Z")
        self.assertEqual(aligned["alignment_type"], "closed_to_next_open")
        self.assertEqual(int(aligned["origin_tolerance_minutes_used"]), 5)
        self.assertEqual(int(aligned["origin_shift_minutes"]), 35)

    def test_pair_metric_quality_never_reselects_a_later_pair(self):
        first = self._pair("pair_first", "2023-03-14T14:02:00Z")
        duplicate_bad = dict(first)
        duplicate_bad.update(
            {"metric_status": "no_common_nominal_strike", "maturity_date": "2023-09-30"}
        )
        later = self._pair("pair_later", "2023-03-14T14:04:00Z")
        pairs = pd.DataFrame([duplicate_bad, first, later])

        universe = prepare_pair_universe(pairs, session_calendar=self.calendar)
        self.assertEqual(len(universe), 2)
        aligned = build_news_first_alignments(
            self._news(["2023-03-14T14:00:00Z"]),
            pairs,
            session_calendar=self.calendar,
        )[5].iloc[0]
        self.assertEqual(aligned["pair_id"], "pair_first")

    def test_invalid_timestamp_remains_at_news_grain(self):
        pairs = pd.DataFrame([self._pair("pair_1", "2023-03-14T14:00:00Z")])
        alignments = build_news_first_alignments(
            self._news(["not-a-time", "2023-03-14T14:00:00Z"]),
            pairs,
            session_calendar=self.calendar,
        )
        for frame in alignments.values():
            self.assertEqual(len(frame), 2)
            self.assertTrue(frame["news_row_id"].is_unique)
            invalid = frame.set_index("news_row_id").loc[1]
            self.assertEqual(invalid["unmatched_reason"], "invalid_news_timestamp")
            self.assertEqual(int(invalid["has_match"]), 0)

    def test_rejects_duplicate_news_and_invalid_pair_session(self):
        pairs = pd.DataFrame([self._pair("pair_1", "2023-03-14T14:00:00Z")])
        duplicate_news = self._news(["2023-03-14T14:00:00Z", "2023-03-14T14:01:00Z"])
        duplicate_news["news_row_id"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "duplicate news_row_id"):
            build_news_first_alignments(
                duplicate_news, pairs, session_calendar=self.calendar
            )

        invalid = pairs.copy()
        invalid.loc[0, "session_id"] = "wrong_session"
        with self.assertRaisesRegex(ValueError, "inconsistent session_id"):
            prepare_pair_universe(invalid, session_calendar=self.calendar)

    def test_validator_rejects_pre_news_and_non_nested_selection(self):
        pairs = pd.DataFrame(
            [
                self._pair("pair_3", "2023-03-14T14:03:00Z"),
                self._pair("pair_8", "2023-03-14T14:08:00Z"),
            ]
        )
        alignments = build_news_first_alignments(
            self._news(["2023-03-14T14:00:00Z"]),
            pairs,
            session_calendar=self.calendar,
        )

        pre_news = {key: value.copy() for key, value in alignments.items()}
        pre_news[5].loc[0, "effective_origin_utc"] = "2023-03-14T13:59:00Z"
        with self.assertRaisesRegex(ValueError, "pre-news origin"):
            validate_news_first_alignments(pre_news, session_calendar=self.calendar)

        changed = {key: value.copy() for key, value in alignments.items()}
        changed[10].loc[0, "pair_id"] = "pair_changed"
        with self.assertRaisesRegex(ValueError, "changed pair"):
            validate_news_first_alignments(changed, session_calendar=self.calendar)


if __name__ == "__main__":
    unittest.main()
