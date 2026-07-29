from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.raw_vol.relaxed_time_pipeline import _scan_source_files
from wgan_option.surface_generation.market_index import (
    ForwardAlignmentPolicy,
    align_news_to_valid_pairs,
    candidate_anchors_from_minutes,
    connect_market_index,
    find_unmatched_with_eligible_pair,
    insert_candidate_anchors,
    insert_surface_anchor,
    refresh_valid_pairs,
)


class MarketSurfaceIndexTest(unittest.TestCase):
    def test_candidate_anchors_cover_every_trailing_window_boundary(self):
        anchors = candidate_anchors_from_minutes(
            ["2023-01-03T13:56:41Z"],
            window_minutes=5,
        )
        self.assertEqual(
            anchors,
            [
                "2023-01-03T13:57:00Z",
                "2023-01-03T13:58:00Z",
                "2023-01-03T13:59:00Z",
                "2023-01-03T14:00:00Z",
                "2023-01-03T14:01:00Z",
            ],
        )

    def test_sqlite_resume_is_idempotent_and_pairs_require_both_anchors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            connection = connect_market_index(Path(tmpdir) / "index.sqlite")
            try:
                anchors = [
                    "2023-01-03T14:00:00Z",
                    "2023-01-03T14:05:00Z",
                    "2023-01-03T14:10:00Z",
                ]
                self.assertEqual(insert_candidate_anchors(connection, anchors), 3)
                self.assertEqual(insert_candidate_anchors(connection, anchors), 0)
                for anchor in anchors:
                    inserted = insert_surface_anchor(
                        connection,
                        anchor_time_utc=anchor,
                        surface_model="raw",
                        surface_params={
                            "business_days": [5, 10],
                            "percent_strikes": [[0.98, 1.02], [0.98, 1.02]],
                            "implied_vols": [[0.2, 0.21], [0.22, 0.23]],
                        },
                        source_target_utc=anchor,
                        source_direction="backward",
                    )
                    self.assertTrue(inserted)
                connection.commit()
                self.assertEqual(refresh_valid_pairs(connection), 2)
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM valid_pair"
                    ).fetchone()[0],
                    2,
                )

                duplicate = insert_surface_anchor(
                    connection,
                    anchor_time_utc=anchors[0],
                    surface_model="raw",
                    surface_params={
                        "business_days": [5, 10],
                        "percent_strikes": [[0.98, 1.02], [0.98, 1.02]],
                        "implied_vols": [[0.2, 0.21], [0.22, 0.23]],
                    },
                    source_target_utc=anchors[0],
                    source_direction="backward",
                )
                self.assertFalse(duplicate)
            finally:
                connection.close()

    def test_source_scan_builds_candidates_from_valid_option_rows_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.csv.gz"
            pd.DataFrame(
                [
                    {
                        "#RIC": "TYZ3",
                        "Date-Time": "2023-01-03T13:50:00Z",
                        "Price": 112.0,
                        "Volume": 10,
                    },
                    {
                        "#RIC": "TY110C3",
                        "Date-Time": "2023-01-03T14:00:00Z",
                        "Price": 0.25,
                        "Volume": 5,
                    },
                    {
                        "#RIC": "TY111C3",
                        "Date-Time": "2023-01-03T14:10:00Z",
                        "Price": "",
                        "Volume": 5,
                    },
                ]
            ).to_csv(raw_path, index=False, compression="gzip")

            connection = connect_market_index(root / "index.sqlite")
            try:
                summary = _scan_source_files(
                    connection,
                    input_glob=str(raw_path),
                    window_minutes=5,
                    chunk_size=2,
                )
                anchors = [
                    str(row["anchor_time_utc"])
                    for row in connection.execute(
                        "SELECT anchor_time_utc "
                        "FROM candidate_anchor ORDER BY anchor_time_utc"
                    ).fetchall()
                ]
            finally:
                connection.close()

            self.assertEqual(summary["newly_observed_minute_count"], 1)
            self.assertEqual(
                anchors,
                [
                    "2023-01-03T14:01:00Z",
                    "2023-01-03T14:02:00Z",
                    "2023-01-03T14:03:00Z",
                    "2023-01-03T14:04:00Z",
                    "2023-01-03T14:05:00Z",
                ],
            )


class ForwardNewsAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.origins = [
            "2023-03-24T14:00:00Z",
            "2023-03-24T14:05:00Z",
            "2023-03-27T13:30:00Z",
        ]

    @staticmethod
    def _news_frame(times):
        return pd.DataFrame(
            {
                "news_row_id": range(1, len(times) + 1),
                "timestamp_utc": times,
                "publication_timestamp_utc": times,
                "timestamp_parse_status": ["ok"] * len(times),
            }
        )

    def test_exact_intraday_session_and_unmatched_rules(self):
        news = self._news_frame(
            [
                "2023-03-24T14:00:00Z",
                "2023-03-24T13:59:00Z",
                "2023-03-24T13:45:00Z",
                "2023-03-24T14:01:00Z",
                "2023-03-24T21:01:00Z",
                "2023-03-20T10:00:00Z",
                "2023-03-27T13:31:00Z",
            ]
        )
        aligned = align_news_to_valid_pairs(
            news,
            self.origins,
            policy=ForwardAlignmentPolicy(
                intraday_tolerance_minutes=15,
                max_session_shift_minutes=4320,
                horizon_minutes=5,
            ),
        )
        self.assertEqual(
            aligned["alignment_type"].tolist(),
            [
                "exact",
                "intraday_shift",
                "intraday_shift",
                "intraday_shift",
                "session_shift",
                "unmatched",
                "unmatched",
            ],
        )
        self.assertEqual(
            aligned.loc[3, "effective_origin_utc"],
            "2023-03-24T14:05:00Z",
        )
        self.assertEqual(int(aligned.loc[3, "origin_shift_minutes"]), 4)
        self.assertEqual(
            aligned.loc[4, "effective_origin_utc"],
            "2023-03-27T13:30:00Z",
        )
        self.assertEqual(
            aligned.loc[0, "target_anchor_utc"],
            "2023-03-24T14:05:00Z",
        )
        self.assertEqual(
            aligned.loc[0, "current_window_start_utc"],
            "2023-03-24T13:55:00Z",
        )
        self.assertEqual(
            aligned.loc[0, "current_window_end_utc"],
            aligned.loc[0, "target_window_start_utc"],
        )
        self.assertEqual(
            find_unmatched_with_eligible_pair(
                aligned,
                self.origins,
                max_shift_minutes=4320,
            ),
            [],
        )

    def test_collision_pooling_keeps_one_pair_per_article(self):
        news = self._news_frame(
            [
                "2023-03-24T13:59:00Z",
                "2023-03-24T14:00:00Z",
            ]
        )
        aligned = align_news_to_valid_pairs(news, self.origins)
        self.assertEqual(aligned["effective_origin_utc"].nunique(), 1)
        self.assertEqual(aligned["collision_count"].tolist(), [2, 2])
        self.assertTrue(aligned["news_row_id"].is_unique)

    def test_session_shift_can_be_disabled(self):
        news = self._news_frame(["2023-03-24T21:01:00Z"])
        aligned = align_news_to_valid_pairs(
            news,
            self.origins,
            policy=ForwardAlignmentPolicy(include_session_shifted=False),
        )
        self.assertEqual(int(aligned.loc[0, "has_match"]), 0)
        self.assertEqual(
            aligned.loc[0, "unmatched_reason"],
            "session_shift_disabled",
        )

    def test_timezone_aware_dst_values_remain_ordered_in_utc(self):
        news = self._news_frame(
            [
                pd.Timestamp("2023-03-26T02:30:00+01:00").tz_convert("UTC")
            ]
        )
        origins = ["2023-03-26T01:35:00Z"]
        aligned = align_news_to_valid_pairs(news, origins)
        self.assertEqual(int(aligned.loc[0, "origin_shift_minutes"]), 5)
        self.assertEqual(
            aligned.loc[0, "alignment_type"],
            "intraday_shift",
        )


if __name__ == "__main__":
    unittest.main()
