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

from scripts.generate_rq2_text_features import main as generate_rq2_text_features  # noqa: E402


class TestGenerateRQ2TextFeaturesScript(unittest.TestCase):
    def test_script_generates_bow_and_sentiment_artifacts(self):
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
                ]
            )

            self.assertEqual(result_dir, output_dir)
            self.assertTrue((output_dir / "bow_features.xlsx").exists())
            self.assertTrue((output_dir / "llm_sentiment_features.xlsx").exists())
            self.assertTrue((output_dir / "bow_manifest.json").exists())
            self.assertTrue((output_dir / "llm_sentiment_manifest.json").exists())
            combined_manifest = json.loads((output_dir / "rq2_text_features_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(combined_manifest["target_dim"], 6)
            self.assertEqual(combined_manifest["bow_features"], str(output_dir / "bow_features.xlsx"))
            self.assertEqual(combined_manifest["llm_sentiment_features"], str(output_dir / "llm_sentiment_features.xlsx"))


if __name__ == "__main__":
    unittest.main()
