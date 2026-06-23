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

from llm_sentiment.features import fit_sentiment_features, load_sentiment_lexicon  # noqa: E402
from scripts.rq2.enrich_merged_vol import enrich_workbook  # noqa: E402


def _parse_vector(value: str) -> list[float]:
    return [float(item) for item in json.loads(value)]


class TestSentimentFeatures(unittest.TestCase):
    def test_sentiment_features_use_dictionary_and_fixed_width(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dictionary_path = Path(tmpdir) / "lm.csv"
            pd.DataFrame(
                {
                    "Word": ["GAIN", "LOSS", "MAY"],
                    "Positive": [2009, 0, 0],
                    "Negative": [0, 2009, 0],
                    "Uncertainty": [0, 0, 2009],
                }
            ).to_csv(dictionary_path, index=False)
            news_df = pd.DataFrame(
                {
                    "ArticleID": ["a1", "a2"],
                    "SourceFile": ["s1", "s2"],
                    "LP": ["gain may follow policy news", ""],
                }
            )

            result = fit_sentiment_features(news_df, target_dim=20, dictionary_path=dictionary_path)

            self.assertEqual(result.frame["sentiment_dim"].tolist(), [20, 20])
            self.assertEqual(result.frame["sentiment_dictionary_source"].tolist(), [str(dictionary_path), str(dictionary_path)])
            first_vector = _parse_vector(result.frame.loc[0, "sentiment_embedding"])
            self.assertEqual(len(first_vector), 20)
            self.assertGreater(first_vector[0], 0.0)
            self.assertGreater(first_vector[2], 0.0)

    def test_missing_dictionary_can_fail_fast_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "missing.csv"
            with self.assertRaises(FileNotFoundError):
                load_sentiment_lexicon(missing_path, allow_builtin_fallback=False)

    def test_sentiment_cli_writes_feature_artifacts_with_builtin_fallback(self):
        from scripts.llm_sentiment.main import main as sentiment_main

        with tempfile.TemporaryDirectory() as tmpdir:
            news_path = Path(tmpdir) / "news.xlsx"
            output_dir = Path(tmpdir) / "features"
            pd.DataFrame(
                {
                    "ArticleID": ["a1", "a2"],
                    "SourceFile": ["s1", "s2"],
                    "LP": ["positive growth but uncertain risk", None],
                }
            ).to_excel(news_path, index=False)

            feature_path = sentiment_main(
                [
                    "--news-xlsx",
                    str(news_path),
                    "--output-dir",
                    str(output_dir),
                    "--target-dim",
                    "8",
                ]
            )

            self.assertEqual(feature_path, output_dir / "llm_sentiment_features.xlsx")
            self.assertTrue(feature_path.exists())
            self.assertTrue((output_dir / "llm_sentiment_manifest.json").exists())
            features = pd.read_excel(feature_path)
            self.assertEqual(features["sentiment_dictionary_source"].tolist(), ["fallback_builtin", "fallback_builtin"])


class TestRQ2WorkbookEnrichment(unittest.TestCase):
    def test_enrich_workbook_preserves_sheets_and_adds_rq2_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            merged_path = Path(tmpdir) / "merged_vol.xlsx"
            bow_path = Path(tmpdir) / "bow_features.xlsx"
            sentiment_path = Path(tmpdir) / "llm_sentiment_features.xlsx"
            output_path = Path(tmpdir) / "merged_vol_rq2_text.xlsx"

            with pd.ExcelWriter(merged_path, engine="openpyxl") as writer:
                pd.DataFrame(
                    {
                        "sample_id": ["news_1", "news_2"],
                        "news_timestamp_utc": ["2022-01-01T00:00:00Z", "2022-01-01T00:05:00Z"],
                        "existing_col": [1, 2],
                    }
                ).to_excel(writer, sheet_name="gan_input_ready", index=False)
                pd.DataFrame(
                    {
                        "news_row_id": [1, 2],
                        "sample_id": ["news_1", "news_2"],
                        "audit_col": ["a", "b"],
                    }
                ).to_excel(writer, sheet_name="news_surface_pair_audit", index=False)

            pd.DataFrame(
                {
                    "news_row_id": [1, 2],
                    "bow_embedding": [json.dumps([0.1, 0.2]), json.dumps([0.3, 0.4])],
                    "bow_dim": [2, 2],
                }
            ).to_excel(bow_path, index=False)
            pd.DataFrame(
                {
                    "news_row_id": [1, 2],
                    "sentiment_embedding": [json.dumps([1.0, 0.0]), json.dumps([0.0, 1.0])],
                    "sentiment_dim": [2, 2],
                    "sentiment_dictionary_source": ["fixture", "fixture"],
                }
            ).to_excel(sentiment_path, index=False)

            result_path = enrich_workbook(
                merged_vol_path=merged_path,
                bow_features_path=bow_path,
                sentiment_features_path=sentiment_path,
                output_path=output_path,
            )

            self.assertEqual(result_path, output_path)
            gan = pd.read_excel(output_path, sheet_name="gan_input_ready")
            audit = pd.read_excel(output_path, sheet_name="news_surface_pair_audit")
            for frame in (gan, audit):
                self.assertIn("bow_embedding", frame.columns)
                self.assertIn("sentiment_embedding", frame.columns)
                self.assertIn("bow_dim", frame.columns)
                self.assertIn("sentiment_dim", frame.columns)
            self.assertEqual(gan["existing_col"].tolist(), [1, 2])
            self.assertEqual(audit["audit_col"].tolist(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
