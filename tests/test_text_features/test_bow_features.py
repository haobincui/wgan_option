import json
import math
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

from bow import build_bow_features, fit_bow_features  # noqa: E402


def _parse_vector(value: str) -> list[float]:
    return [float(item) for item in json.loads(value)]


class TestBowFeatures(unittest.TestCase):
    def test_bow_features_are_fixed_width_and_deterministic(self):
        news_df = pd.DataFrame(
            {
                "ArticleID": ["a1", "a2", "a3"],
                "SourceFile": ["s1", "s2", "s3"],
                "LP": [
                    "bond volatility rises after policy surprise",
                    "policy news lowers market volatility risk",
                    "quiet market session",
                ],
            }
        )

        first = fit_bow_features(news_df, target_dim=4, max_features=20, random_state=7)
        second = fit_bow_features(news_df, target_dim=4, max_features=20, random_state=7)

        self.assertEqual(list(first.frame.columns), ["news_row_id", "article_id", "source_file", "bow_embedding", "bow_dim"])
        self.assertEqual(first.frame["bow_dim"].tolist(), [4, 4, 4])
        self.assertEqual(len(_parse_vector(first.frame.loc[0, "bow_embedding"])), 4)
        self.assertEqual(first.frame["bow_embedding"].tolist(), second.frame["bow_embedding"].tolist())
        self.assertTrue(first.manifest["fitted"])
        self.assertEqual(first.manifest["representation"], "ngram_frequency")
        self.assertEqual(first.manifest["weighting"], "log1p_count")
        self.assertEqual(first.manifest["reference_method"], "Manela_Moreira_2017_style_ngram_frequency")
        self.assertEqual(first.vocabulary, ["market", "policy", "volatility", "after"])
        expected_first = [0.0, math.log1p(1.0), math.log1p(1.0), math.log1p(1.0)]
        expected_second = [math.log1p(1.0), math.log1p(1.0), math.log1p(1.0), 0.0]
        for actual, expected in zip(_parse_vector(first.frame.loc[0, "bow_embedding"]), expected_first):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(_parse_vector(first.frame.loc[1, "bow_embedding"]), expected_second):
            self.assertAlmostEqual(actual, expected)

    def test_build_bow_features_handles_empty_text_rows(self):
        news_df = pd.DataFrame(
            {
                "news_row_id": [10, 11],
                "ArticleID": ["a10", "a11"],
                "SourceFile": ["s10", "s11"],
                "LP": ["", None],
            }
        )

        features = build_bow_features(news_df, target_dim=3, max_features=10)

        self.assertEqual(features["news_row_id"].tolist(), [10, 11])
        for raw_vector in features["bow_embedding"].tolist():
            self.assertEqual(_parse_vector(raw_vector), [0.0, 0.0, 0.0])

    def test_bow_cli_writes_feature_artifacts(self):
        from scripts.bow.main import main as bow_main

        with tempfile.TemporaryDirectory() as tmpdir:
            news_path = Path(tmpdir) / "news.xlsx"
            output_dir = Path(tmpdir) / "features"
            pd.DataFrame(
                {
                    "ArticleID": ["a1", "a2"],
                    "SourceFile": ["s1", "s2"],
                    "LP": ["growth improves volatility outlook", "risk and losses increase"],
                }
            ).to_excel(news_path, index=False)

            feature_path = bow_main(
                [
                    "--news-xlsx",
                    str(news_path),
                    "--output-dir",
                    str(output_dir),
                    "--target-dim",
                    "3",
                    "--max-features",
                    "20",
                ]
            )

            self.assertEqual(feature_path, output_dir / "bow_features.xlsx")
            self.assertTrue(feature_path.exists())
            self.assertTrue((output_dir / "bow_manifest.json").exists())
            self.assertTrue((output_dir / "bow_vocabulary.json").exists())
            self.assertFalse((output_dir / "tfidf_vectorizer.joblib").exists())
            self.assertFalse((output_dir / "svd_model.joblib").exists())
            manifest = json.loads((output_dir / "bow_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["representation"], "ngram_frequency")
            self.assertEqual(manifest["weighting"], "log1p_count")
            self.assertEqual(manifest["vocabulary_path"], str(output_dir / "bow_vocabulary.json"))
            vocabulary = json.loads((output_dir / "bow_vocabulary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(vocabulary), 3)


if __name__ == "__main__":
    unittest.main()
