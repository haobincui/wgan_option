import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for path in (str(ROOT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.rq3.market_jump_news import (  # noqa: E402
    build_episode_news_bridge,
    build_episode_official_bridge,
    parse_factiva_news,
)


class TestRQ3MarketJumpNews(unittest.TestCase):
    def _write_news(self, path: Path) -> None:
        pd.DataFrame(
            [
                {"SourceFile": "a", "ArticleID": "A", "PD": "2022-01-15", "ET": "12:00", "HD": "winter", "LP": "Same, text!"},
                {"SourceFile": "b", "ArticleID": "B", "PD": "2022-07-15", "ET": "12:00", "HD": "summer", "LP": "same text"},
                {"SourceFile": "c", "ArticleID": "C", "PD": "2022-07-15", "ET": "12:00", "HD": "collision", "LP": "other"},
            ]
        ).to_excel(path, index=False)

    def test_parse_factiva_news_uses_london_dst_and_stable_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "news.xlsx"
            self._write_news(path)
            parsed = parse_factiva_news(path)

            self.assertEqual(parsed["news_row_id"].tolist(), [1, 2, 3])
            self.assertEqual(parsed.loc[0, "publication_timestamp_utc"], "2022-01-15T12:00:00Z")
            self.assertEqual(parsed.loc[1, "publication_timestamp_utc"], "2022-07-15T11:00:00Z")
            self.assertEqual(parsed["source_utc_offset_minutes"].tolist(), [0.0, 60.0, 60.0])
            self.assertEqual(parsed.loc[1, "publication_group_size"], 2)
            self.assertEqual(parsed.loc[1, "publication_collision_count"], 1)
            self.assertEqual(parsed.loc[0, "lp_text_sha256"], parsed.loc[1, "lp_text_sha256"])
            expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(set(parsed["workbook_sha256"]), {expected_hash})
            self.assertNotIn("HD_embedding", parsed.columns)

    def test_parse_factiva_news_rejects_legacy_helper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "news_with_ty_plus_trade_counts.xlsx"
            self._write_news(path)
            with self.assertRaisesRegex(ValueError, "wrong source timezone"):
                parse_factiva_news(path)

    def test_parse_factiva_news_audits_london_dst_boundary_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "news.xlsx"
            pd.DataFrame(
                [
                    {"SourceFile": "a", "ArticleID": "A", "PD": "2022-03-27", "ET": "01:30", "HD": "spring", "LP": "spring"},
                    {"SourceFile": "b", "ArticleID": "B", "PD": "2022-10-30", "ET": "01:30", "HD": "autumn", "LP": "autumn"},
                ]
            ).to_excel(path, index=False)
            parsed = parse_factiva_news(path)
            self.assertEqual(
                parsed["timestamp_parse_status"].tolist(),
                ["dst_ambiguous_or_nonexistent", "dst_ambiguous_or_nonexistent"],
            )
            self.assertTrue(parsed["publication_timestamp_utc"].isna().all())

    def test_parse_factiva_news_requires_lineage_and_report_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "news.xlsx"
            pd.DataFrame([{"PD": "2022-01-01", "ET": "12:00"}]).to_excel(path, index=False)
            with self.assertRaisesRegex(ValueError, "ArticleID.*HD.*LP.*SourceFile"):
                parse_factiva_news(path)

    def test_news_bridge_respects_boundaries_collisions_and_alignment(self):
        episodes = pd.DataFrame(
            [{"episode_id": "ep1", "episode_start_utc": "2022-07-15T11:00:00Z", "episode_end_utc": "2022-07-15T11:05:00Z"}]
        )
        times = [
            "2022-07-15T10:55:00Z",  # inclusive pre start
            "2022-07-15T10:59:59Z",
            "2022-07-15T11:00:00Z",  # impact start
            "2022-07-15T11:04:59Z",
            "2022-07-15T11:05:00Z",  # post start
            "2022-07-15T11:20:00Z",  # exclusive post end
            "2022-07-15T11:20:01Z",  # excluded
        ]
        news = pd.DataFrame(
            {
                "news_row_id": range(1, 8),
                "sample_id": [f"news_{i}" for i in range(1, 8)],
                "news_available_time_utc": times,
                "publication_timestamp_utc": times,
                "workbook_sha256": "hash",
                "lp_text_sha256": ["same", "same", "a", "b", "c", "d", "e"],
                "SourceFile": "source",
                "ArticleID": ["A", "B", "C", "D", "E", "F", "G"],
                "HD": [f"headline {i}" for i in range(7)],
                "LP": "lead",
                "publication_group_size": [1, 1, 2, 2, 1, 1, 1],
                "publication_collision_count": [0, 0, 1, 1, 0, 0, 0],
                "article_id_group_size": 1,
                "lp_text_group_size": [2, 2, 1, 1, 1, 1, 1],
            }
        )
        alignment = pd.DataFrame(
            [
                {"news_row_id": 5, "has_match": 1, "effective_origin_utc": "2022-07-15T11:05:00Z", "target_anchor_utc": "2022-07-15T11:10:00Z", "match_method": "closed_to_next_open", "session_regime": "closed", "quality_status": "usable", "collision_count": 3},
                # A lower-priority duplicate must not duplicate bridge candidates.
                {"news_row_id": 5, "has_match": 0, "matching_rank": 9, "match_method": "unmatched"},
            ]
        )

        bridge = build_episode_news_bridge(episodes, news, alignment=alignment)
        self.assertEqual(bridge["news_row_id"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(
            bridge["window_relation"].tolist(),
            ["pre_context", "pre_context", "impact", "impact", "post_reporting"],
        )
        self.assertEqual(set(bridge["episode_candidate_count"]), {5})
        self.assertEqual(bridge.loc[bridge["news_row_id"] == 3, "episode_window_candidate_count"].iloc[0], 2)
        aligned = bridge[bridge["news_row_id"] == 5].iloc[0]
        self.assertEqual(aligned["alignment_row_count"], 2)
        self.assertEqual(aligned["alignment_match_method"], "closed_to_next_open")
        self.assertEqual(aligned["alignment_closed_to_next_open"], 1)
        self.assertEqual(aligned["alignment_collision_count"], 3)
        self.assertEqual(bridge["episode_news_bridge_id"].nunique(), 5)

    def test_official_bridge_uses_same_three_windows_and_legacy_time_alias(self):
        episodes = pd.DataFrame(
            [{"episode_id": "ep1", "start_utc": "2022-01-01T10:00:00Z", "end_utc": "2022-01-01T10:05:00Z"}]
        )
        calendar = pd.DataFrame(
            [
                {"event_id": "pre", "event_time_utc": "2022-01-01T09:55:00Z", "event_name": "pre"},
                {"event_id": "impact", "event_time_utc": "2022-01-01T10:00:00Z", "event_name": "impact"},
                {"event_id": "post", "event_time_utc": "2022-01-01T10:05:00Z", "event_name": "post"},
                {"event_id": "post_end", "event_time_utc": "2022-01-01T10:20:00Z", "event_name": "post end"},
                {"event_id": "outside", "event_time_utc": "2022-01-01T10:20:01Z", "event_name": "outside"},
            ]
        )
        bridge = build_episode_official_bridge(episodes, calendar)
        self.assertEqual(bridge["event_id"].tolist(), ["pre", "impact", "post"])
        self.assertEqual(
            bridge["window_relation"].tolist(),
            ["pre_context", "impact", "post_reporting"],
        )
        self.assertEqual(set(bridge["episode_candidate_count"]), {3})
        self.assertIn("event_name", bridge.columns)


if __name__ == "__main__":
    unittest.main()
