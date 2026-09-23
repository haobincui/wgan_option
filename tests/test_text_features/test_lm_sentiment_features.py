import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lm_sentiment import (  # noqa: E402
    load_lm_dictionary,
    score_lm_articles,
    score_lm_pairs,
    score_lm_text,
)


def _write_dictionary(path: Path) -> None:
    words = [
        ("GAIN", 0, 2009, 0, 0, 0, 0, 0),
        ("LOSS", 2009, 0, 0, 0, 0, 0, 0),
        ("RETIRED", -2020, 0, 0, 0, 0, 0, 0),
        ("MAY", 0, 0, 0, 0, 0, 2009, 0),
        ("NOT", 0, 0, 0, 0, 0, 0, 0),
        ("POLICY", 0, 0, 0, 0, 0, 0, 0),
        ("RISK", 0, 0, 2009, 0, 0, 0, 0),
        ("CLAIM", 0, 0, 0, 2009, 0, 0, 0),
        ("MUST", 0, 0, 0, 0, 2009, 0, 0),
        ("LIMIT", 0, 0, 0, 0, 0, 0, 2009),
    ]
    pd.DataFrame(
        words,
        columns=[
            "Word",
            "Negative",
            "Positive",
            "Uncertainty",
            "Litigious",
            "Strong_Modal",
            "Weak_Modal",
            "Constraining",
        ],
    ).to_csv(path, index=False)


class TestLMSentimentFeatures(unittest.TestCase):
    def test_loader_uses_only_active_positive_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lm.csv"
            _write_dictionary(path)
            dictionary = load_lm_dictionary(path)

        self.assertIn("LOSS", dictionary.categories["negative"])
        self.assertNotIn("RETIRED", dictionary.categories["negative"])
        self.assertIn("RETIRED", dictionary.words)
        self.assertEqual(dictionary.row_count, 10)
        self.assertEqual(len(dictionary.sha256), 64)

    def test_primary_score_uses_valid_word_denominator_and_negation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lm.csv"
            _write_dictionary(path)
            dictionary = load_lm_dictionary(path)

            regular = score_lm_text("gain loss policy unknown", dictionary)
            negated = score_lm_text("not policy policy policy gain", dictionary)
            outside_window = score_lm_text(
                "not policy policy policy policy gain", dictionary
            )

        self.assertEqual(regular["lm_raw_word_count"], 4)
        self.assertEqual(regular["lm_valid_word_count"], 3)
        self.assertAlmostEqual(regular["lm_sentiment_score"], 0.0)
        self.assertEqual(negated["lm_negated_positive_count"], 1)
        self.assertEqual(negated["lm_adjusted_positive_count"], 0)
        self.assertEqual(outside_window["lm_negated_positive_count"], 0)

    def test_title_case_month_may_is_excluded_but_lowercase_modal_is_kept(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lm.csv"
            _write_dictionary(path)
            dictionary = load_lm_dictionary(path)
            score = score_lm_text("May policy may risk", dictionary)

        self.assertEqual(score["lm_raw_word_count"], 3)
        self.assertEqual(score["lm_valid_word_count"], 3)
        self.assertEqual(score["lm_weak_modal_count"], 1)

    def test_pair_score_pools_counts_before_ratio_and_deduplicates_articles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lm.csv"
            _write_dictionary(path)
            dictionary = load_lm_dictionary(path)
            news = pd.DataFrame(
                {
                    "news_row_id": [1, 2, 3],
                    "ArticleID": ["a1", "a1", "a2"],
                    "LP": [
                        "gain policy policy policy policy policy policy policy policy policy",
                        "gain policy policy policy policy policy policy policy policy policy",
                        "loss policy",
                    ],
                }
            )
            article_scores = score_lm_articles(news, dictionary)
            pairs = pd.DataFrame(
                {
                    "pair_id": ["p1", "p1", "p1"],
                    "news_row_id": [1, 2, 3],
                    "article_id": ["a1", "a1", "a2"],
                    "sample_id": ["s1", "s2", "s3"],
                    "lp_text": news["LP"],
                    "session_id": ["session1"] * 3,
                }
            )
            result = score_lm_pairs(pairs, article_scores)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["lm_unique_article_count"], 2)
        self.assertEqual(row["lm_duplicate_article_row_count"], 1)
        self.assertEqual(row["lm_valid_word_count"], 12)
        self.assertAlmostEqual(row["lm_sentiment_score"], 0.0)

    def test_pair_score_rejects_conflicting_duplicate_article_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lm.csv"
            _write_dictionary(path)
            dictionary = load_lm_dictionary(path)
            news = pd.DataFrame(
                {
                    "news_row_id": [1, 2],
                    "ArticleID": ["a1", "a1"],
                    "LP": ["gain policy", "loss policy"],
                }
            )
            article_scores = score_lm_articles(news, dictionary)
            pairs = pd.DataFrame(
                {
                    "pair_id": ["p1", "p1"],
                    "news_row_id": [1, 2],
                    "article_id": ["a1", "a1"],
                    "sample_id": ["s1", "s2"],
                    "lp_text": news["LP"],
                }
            )

            with self.assertRaisesRegex(
                ValueError, "Duplicate article key maps to different LP text"
            ):
                score_lm_pairs(pairs, article_scores)

    def test_cli_writes_scores_and_auditable_manifest(self):
        from scripts.lm_sentiment.main import main as lm_main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dictionary_path = root / "lm.csv"
            news_path = root / "news.xlsx"
            pair_path = root / "pairs.xlsx"
            output_dir = root / "output"
            _write_dictionary(dictionary_path)
            pd.DataFrame(
                {
                    "ArticleID": ["a1", "a2"],
                    "SourceFile": ["source1", "source2"],
                    "LP": ["gain policy", "loss risk"],
                }
            ).to_excel(news_path, sheet_name="Sheet1", index=False)
            with pd.ExcelWriter(pair_path, engine="openpyxl") as writer:
                pd.DataFrame(
                    {
                        "pair_id": ["p1", "p1"],
                        "news_row_id": [1, 2],
                        "article_id": ["a1", "a2"],
                        "sample_id": ["s1", "s2"],
                        "session_id": ["session1", "session1"],
                        "lp_text": ["gain policy", "loss risk"],
                    }
                ).to_excel(writer, sheet_name="gan_input_ready", index=False)

            result_dir = lm_main(
                [
                    "--news-xlsx",
                    str(news_path),
                    "--dictionary-path",
                    str(dictionary_path),
                    "--pair-workbook",
                    str(pair_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            manifest = json.loads(
                (result_dir / "lm_sentiment_manifest.json").read_text()
            )
            pair_scores = pd.read_csv(result_dir / "lm_pair_scores_5m.csv")

        self.assertEqual(manifest["schema_version"], "lm_dictionary_sentiment_v1")
        self.assertEqual(manifest["primary_score"], "lm_net_tone")
        self.assertEqual(
            manifest["dictionary"]["active_flag_rule"],
            "strictly_greater_than_zero",
        )
        self.assertEqual(len(manifest["outputs"]["pair_scores"]["sha256"]), 64)
        self.assertGreater(manifest["outputs"]["workbook"]["bytes"], 0)
        self.assertEqual(pair_scores["pair_id"].tolist(), ["p1"])
        self.assertAlmostEqual(pair_scores.loc[0, "lm_sentiment_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
