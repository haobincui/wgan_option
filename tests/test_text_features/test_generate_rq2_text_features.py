import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.generate_rq2_text_features import main as generate_rq2_text_features  # noqa: E402


class TestGenerateRQ2TextFeaturesScript(unittest.TestCase):
    def test_script_generates_bow_and_sentiment_artifacts(self):
        class FakeBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def generate(self, prompt: str) -> str:
                return (
                    '{"macroeconomic_uncertainty": 0.2, '
                    '"institutional_action": 0.4, "risk_off_intensity": 0.6}'
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            news_path = Path(tmpdir) / "news.xlsx"
            output_dir = Path(tmpdir) / "rq2_features"
            pd.DataFrame(
                {
                    "ArticleID": ["a1", "a2"],
                    "SourceFile": ["s1", "s2"],
                    "LP": ["policy news improves growth outlook", "uncertain volatility risk and losses"],
                }
            ).to_excel(news_path, index=False)

            with patch("llm_sentiment.features.OpenAIChatGPTSentimentBackend", FakeBackend):
                result_dir = generate_rq2_text_features(
                    [
                        "--news-xlsx",
                        str(news_path),
                        "--output-dir",
                        str(output_dir),
                        "--target-dim",
                        "6",
                        "--max-features",
                        "20",
                        "--model",
                        "fixture-chatgpt",
                    ]
                )

            self.assertEqual(result_dir, output_dir)
            self.assertTrue((output_dir / "bow_features.xlsx").exists())
            self.assertTrue((output_dir / "bow_vocabulary.json").exists())
            self.assertTrue((output_dir / "llm_sentiment_features.xlsx").exists())
            self.assertTrue((output_dir / "bow_manifest.json").exists())
            self.assertTrue((output_dir / "llm_sentiment_manifest.json").exists())
            self.assertFalse((output_dir / "tfidf_vectorizer.joblib").exists())
            self.assertFalse((output_dir / "svd_model.joblib").exists())
            combined_manifest = json.loads((output_dir / "rq2_text_features_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(combined_manifest["target_dim"], 6)
            self.assertEqual(combined_manifest["bow_features"], str(output_dir / "bow_features.xlsx"))
            self.assertEqual(combined_manifest["llm_sentiment_features"], str(output_dir / "llm_sentiment_features.xlsx"))
            self.assertEqual(combined_manifest["bow_manifest"]["representation"], "ngram_frequency")
            self.assertEqual(combined_manifest["bow_manifest"]["vocabulary_path"], str(output_dir / "bow_vocabulary.json"))
            self.assertEqual(
                combined_manifest["llm_sentiment_manifest"]["representation"],
                "sun2026_style_openai_chatgpt_multidimensional_sentiment",
            )
            self.assertEqual(combined_manifest["llm_sentiment_manifest"]["model_id"], "fixture-chatgpt")


if __name__ == "__main__":
    unittest.main()
