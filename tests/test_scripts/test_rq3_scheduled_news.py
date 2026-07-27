from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.rq3.scheduled_news_regime import (
    MATCH_COVARIATES,
    build_fomc_case_study,
    build_inference,
    label_information_regimes,
    load_event_calendar,
    match_scheduled_to_ordinary,
)


ROOT = Path(__file__).resolve().parents[2]


def _calendar() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "event_id": "event_1",
                "event_family": "FOMC",
                "release_name": "Test release",
                "release_time_local": "2023-01-03T09:00:00-05:00",
                "release_timezone": "America/New_York",
                "release_time_utc": "2023-01-03T14:00:00Z",
                "scheduled_or_unscheduled": "scheduled",
                "official_source": "https://example.com",
                "source_retrieval_date": "2026-07-25",
                "calendar_version": "test",
                "priority": 1,
            }
        ]
    )
    frame["release_time"] = pd.to_datetime(
        frame["release_time_utc"],
        utc=True,
    )
    frame["release_local_date"] = "2023-01-03"
    frame["release_trading_day"] = "2023-01-03"
    return frame


def _pair(pair_id: str, stamp: str, *, fold: str = "2023Q1") -> dict:
    timestamp = pd.Timestamp(stamp)
    minute = timestamp.hour * 60 + timestamp.minute
    weekday = timestamp.weekday()
    row = {
        "fold": fold,
        "surface_pair_id": pair_id,
        "current_snapshot_time_utc": stamp,
        "target_snapshot_time_utc": (
            timestamp + pd.Timedelta(minutes=5)
        ).isoformat(),
        "minute_of_day": float(minute),
        "minute_of_day_sin": float(
            np.sin(2.0 * np.pi * minute / 1440.0)
        ),
        "minute_of_day_cos": float(
            np.cos(2.0 * np.pi * minute / 1440.0)
        ),
        "weekday": weekday,
        "weekday_sin": float(np.sin(2.0 * np.pi * weekday / 7.0)),
        "weekday_cos": float(np.cos(2.0 * np.pi * weekday / 7.0)),
    }
    for index, column in enumerate(MATCH_COVARIATES):
        row.setdefault(column, float(index + 1))
    return row


class TestRQ3ScheduledNews(unittest.TestCase):
    def test_frozen_2023_calendar_schema_and_timezone(self):
        path = ROOT / "data/reference/rq3_scheduled_macro_events_2023.csv"
        calendar = load_event_calendar(path)
        self.assertEqual(len(calendar), 84)
        self.assertEqual(calendar["event_id"].nunique(), 84)
        self.assertEqual(calendar["event_family"].nunique(), 8)
        local_as_utc = pd.to_datetime(
            calendar["release_time_local"],
            utc=True,
        )
        pd.testing.assert_series_equal(
            local_as_utc.reset_index(drop=True),
            calendar["release_time"].reset_index(drop=True),
            check_names=False,
        )

    def test_regime_label_uses_post_release_window_and_buffer(self):
        pairs = pd.DataFrame(
            [
                _pair("event", "2023-01-03T14:02:00Z"),
                _pair("buffer", "2023-01-03T13:30:00Z"),
                _pair("ordinary", "2023-01-03T16:00:00Z"),
            ]
        )
        labeled = label_information_regimes(
            pairs,
            _calendar(),
            pre_window_minutes=0,
            post_window_minutes=5,
            ordinary_buffer_minutes=60,
        ).set_index("surface_pair_id")
        self.assertEqual(labeled.loc["event", "regime"], "scheduled_news")
        self.assertEqual(
            labeled.loc["buffer", "regime"],
            "event_buffer_excluded",
        )
        self.assertEqual(
            labeled.loc["ordinary", "regime"],
            "ordinary_news_candidate",
        )
        self.assertAlmostEqual(
            float(labeled.loc["event", "minutes_from_release"]),
            2.0,
        )

    def test_matching_is_fold_weekday_and_no_replacement_controlled(self):
        event = _pair("event", "2023-01-03T14:02:00Z")
        event.update(
            {
                "regime": "scheduled_news",
                "event_id": "event_1",
                "event_ids": '["event_1"]',
                "event_family": "FOMC",
                "release_name": "Test",
                "release_time_utc": "2023-01-03T14:00:00Z",
                "release_local_date": "2023-01-03",
                "release_trading_day": "2023-01-03",
                "minutes_from_release": 2.0,
            }
        )
        good = _pair("good", "2023-01-10T14:01:00Z")
        wrong_weekday = _pair("wrong_weekday", "2023-01-04T14:01:00Z")
        wrong_fold = _pair(
            "wrong_fold",
            "2023-01-10T14:01:00Z",
            fold="2023Q2",
        )
        for control in (good, wrong_weekday, wrong_fold):
            control.update(
                {
                    "regime": "ordinary_news_candidate",
                    "event_id": "",
                    "event_ids": "[]",
                    "event_family": "",
                    "release_name": "",
                    "release_time_utc": "",
                    "release_local_date": "",
                    "release_trading_day": "",
                    "minutes_from_release": np.nan,
                }
            )
        result = match_scheduled_to_ordinary(
            pd.DataFrame([event, good, wrong_weekday, wrong_fold]),
            time_caliper_minutes=15,
            exact_weekday=True,
        )
        self.assertEqual(len(result.manifest), 1)
        self.assertEqual(
            result.manifest.iloc[0]["control_surface_pair_id"],
            "good",
        )
        self.assertEqual(
            result.manifest["control_surface_pair_id"].nunique(),
            len(result.manifest),
        )

    def test_inference_uses_seed_level_direction_not_set_minimum(self):
        seed_rows = []
        average_rows = []
        manifest_rows = []
        for index, increment in enumerate([0.1, 0.2, 0.3, 0.4], start=1):
            set_id = f"set_{index}"
            release_day = f"2023-01-{index:02d}"
            for seed in (42, 202, 404):
                seed_rows.append(
                    {
                        "analysis_type": "scheduled_news",
                        "window": "primary_0_5",
                        "matched_set_id": set_id,
                        "fold": "2023Q1",
                        "seed": seed,
                        "event_id": f"event_{index}",
                        "event_family": "FOMC",
                        "release_time_utc": (
                            f"{release_day}T14:00:00Z"
                        ),
                        "release_local_date": release_day,
                        "release_trading_day": release_day,
                        "event_surface_pair_id": f"event_pair_{index}",
                        "control_surface_pair_id": f"control_pair_{index}",
                        "contrast": "lp_vs_continued_no_text",
                        "contrast_family": "primary",
                        "focal_model": "lp",
                        "baseline_model": "continued_no_text",
                        "metric": "surface_mae",
                        "event_text_advantage": increment,
                        "control_text_advantage": 0.0,
                        "scheduled_news_increment": increment,
                        "difference_direction": "event_minus_control",
                        "positive_means_focal_more_valuable_in_event": True,
                    }
                )
            average_rows.append(
                {
                    "analysis_type": "scheduled_news",
                    "window": "primary_0_5",
                    "matched_set_id": set_id,
                    "fold": "2023Q1",
                    "event_id": f"event_{index}",
                    "event_family": "FOMC",
                    "release_time_utc": f"{release_day}T14:00:00Z",
                    "release_local_date": release_day,
                    "release_trading_day": release_day,
                    "event_surface_pair_id": f"event_pair_{index}",
                    "control_surface_pair_id": f"control_pair_{index}",
                    "contrast": "lp_vs_continued_no_text",
                    "contrast_family": "primary",
                    "focal_model": "lp",
                    "baseline_model": "continued_no_text",
                    "metric": "surface_mae",
                    "difference_direction": "event_minus_control",
                    "positive_means_focal_more_valuable_in_event": True,
                    "event_text_advantage": increment,
                    "control_text_advantage": 0.0,
                    "scheduled_news_increment": increment,
                    "seed_count": 3,
                    "positive_seed_count": 3,
                }
            )
            manifest = {
                "matched_set_id": set_id,
                "release_trading_day": release_day,
            }
            for column in MATCH_COVARIATES:
                manifest[f"difference_{column}"] = 0.0
            manifest_rows.append(manifest)
        inference, conditional = build_inference(
            pd.DataFrame(average_rows),
            pd.DataFrame(seed_rows),
            {"primary_0_5": pd.DataFrame(manifest_rows)},
            bootstrap_iterations=500,
            bootstrap_seed=7,
        )
        self.assertEqual(
            int(inference.iloc[0]["positive_seed_direction_count"]),
            3,
        )
        self.assertAlmostEqual(
            float(inference.iloc[0]["mean_scheduled_news_increment"]),
            0.25,
        )
        self.assertAlmostEqual(
            float(conditional.iloc[0]["conditional_intercept"]),
            0.25,
        )

    def test_fomc_case_study_aggregates_multiple_pairs_per_meeting(self):
        seed_rows = []
        average_rows = []
        for set_id, value in (("a", 0.1), ("b", 0.3)):
            for seed in (42, 202, 404):
                seed_rows.append(
                    {
                        "window": "robustness_0_30",
                        "event_id": "fomc_1",
                        "event_family": "FOMC",
                        "release_time_utc": "2023-01-03T14:00:00Z",
                        "release_local_date": "2023-01-03",
                        "fold": "2023Q1",
                        "seed": seed,
                        "contrast": "lp_vs_continued_no_text",
                        "metric": "surface_mae",
                        "matched_set_id": set_id,
                        "scheduled_news_increment": value,
                    }
                )
            average_rows.append(
                {
                    "window": "robustness_0_30",
                    "event_id": "fomc_1",
                    "event_family": "FOMC",
                    "release_time_utc": "2023-01-03T14:00:00Z",
                    "release_local_date": "2023-01-03",
                    "fold": "2023Q1",
                    "contrast": "lp_vs_continued_no_text",
                    "metric": "surface_mae",
                    "matched_set_id": set_id,
                    "scheduled_news_increment": value,
                }
            )
        seed_frame, summary = build_fomc_case_study(
            pd.DataFrame(seed_rows),
            pd.DataFrame(average_rows),
        )
        self.assertEqual(len(seed_frame), 3)
        self.assertTrue(
            np.allclose(seed_frame["mean_scheduled_news_increment"], 0.2)
        )
        self.assertEqual(int(summary.iloc[0]["meeting_count"]), 1)
        self.assertAlmostEqual(
            float(summary.iloc[0]["mean_scheduled_news_increment"]),
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
