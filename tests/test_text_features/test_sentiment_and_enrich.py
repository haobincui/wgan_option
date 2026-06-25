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

from llm_sentiment.features import (  # noqa: E402
    SENTIMENT_DIMENSIONS,
    fit_sentiment_features,
    parse_chatgpt_sentiment_response,
)
from scripts.rq2.enrich_merged_vol import enrich_workbook  # noqa: E402


def _parse_vector(value: str) -> list[float]:
    return [float(item) for item in json.loads(value)]


class TestSentimentFeatures(unittest.TestCase):
    def test_chatgpt_sentiment_features_use_sun_dimensions_and_fixed_width(self):
        class FakeBackend:
            def __init__(self):
                self.prompts = []

            def generate(self, prompt: str) -> str:
                self.prompts.append(prompt)
                return json.dumps(
                    {
                        "macroeconomic_uncertainty": 0.7,
                        "institutional_action": 0.2,
                        "risk_off_intensity": 0.5,
                    }
                )

        backend = FakeBackend()
        news_df = pd.DataFrame(
            {
                "ArticleID": ["a1", "a2"],
                "SourceFile": ["s1", "s2"],
                "LP": ["central bank warns inflation risks may raise market volatility", ""],
            }
        )

        result = fit_sentiment_features(news_df, target_dim=8, model_id="fixture-chatgpt", backend=backend)

        self.assertEqual(result.manifest["sentiment_dimensions"], list(SENTIMENT_DIMENSIONS))
        self.assertEqual(result.frame["sentiment_dim"].tolist(), [8, 8])
        self.assertEqual(
            result.frame["sentiment_dictionary_source"].tolist(),
            ["openai_chatgpt_sun2026_style:fixture-chatgpt", "openai_chatgpt_sun2026_style:fixture-chatgpt"],
        )
        first_vector = _parse_vector(result.frame.loc[0, "sentiment_embedding"])
        second_vector = _parse_vector(result.frame.loc[1, "sentiment_embedding"])
        for actual, expected in zip(first_vector[:3], [0.7, 0.2, 0.5]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(first_vector[3:], [0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(second_vector, [0.0] * 8)
        self.assertEqual(result.frame["sentiment_parse_status"].tolist(), ["json", "empty_text"])
        self.assertEqual(len(backend.prompts), 1)
        self.assertIn("macroeconomic uncertainty", backend.prompts[0])
        self.assertIn("Do not mechanically", backend.prompts[0])

    def test_response_parser_handles_fenced_json_regex_and_malformed_text(self):
        fenced = parse_chatgpt_sentiment_response(
            """
            ```json
            {"macroeconomic_uncertainty": 0.6, "institutional_action": 0.1, "risk_off_intensity": 0.4}
            ```
            """
        )
        self.assertEqual(fenced.parse_status, "json")
        self.assertEqual(fenced.scores["macroeconomic_uncertainty"], 0.6)

        regex = parse_chatgpt_sentiment_response(
            "macroeconomic_uncertainty=75% institutional_action: 0.2 risk_off_intensity: 1.2"
        )
        self.assertEqual(regex.parse_status, "regex")
        self.assertEqual(regex.scores["macroeconomic_uncertainty"], 0.75)
        self.assertEqual(regex.scores["risk_off_intensity"], 1.0)

        malformed = parse_chatgpt_sentiment_response("not a score")
        self.assertEqual(malformed.parse_status, "parse_failed")
        self.assertEqual(malformed.scores, {dimension: 0.0 for dimension in SENTIMENT_DIMENSIONS})

    def test_cache_hit_does_not_call_backend_twice(self):
        class FakeBackend:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt: str) -> str:
                self.calls += 1
                return (
                    '{"macroeconomic_uncertainty": 0.9, '
                    '"institutional_action": 0.3, "risk_off_intensity": 0.4}'
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FakeBackend()
            cache_path = Path(tmpdir) / "cache.jsonl"
            news_df = pd.DataFrame({"LP": ["same text", "same text"]})

            result = fit_sentiment_features(
                news_df,
                target_dim=4,
                model_id="fixture-chatgpt",
                cache_path=cache_path,
                backend=backend,
            )

            self.assertEqual(backend.calls, 1)
            self.assertTrue(cache_path.exists())
            self.assertEqual(result.frame["sentiment_parse_status"].tolist(), ["json", "cache_json"])

    def test_sentiment_cli_writes_feature_artifacts_with_fake_openai_backend(self):
        from scripts.llm_sentiment.main import main as sentiment_main

        class FakeBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def generate(self, prompt: str) -> str:
                return (
                    '{"macroeconomic_uncertainty": 0.1, '
                    '"institutional_action": 0.8, "risk_off_intensity": 0.2}'
                )

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

            with patch("llm_sentiment.features.OpenAIChatGPTSentimentBackend", FakeBackend):
                feature_path = sentiment_main(
                    [
                        "--news-xlsx",
                        str(news_path),
                        "--output-dir",
                        str(output_dir),
                        "--target-dim",
                        "8",
                        "--model",
                        "fixture-chatgpt",
                    ]
                )

            self.assertEqual(feature_path, output_dir / "llm_sentiment_features.xlsx")
            self.assertTrue(feature_path.exists())
            self.assertTrue((output_dir / "llm_sentiment_manifest.json").exists())
            features = pd.read_excel(feature_path)
            self.assertEqual(
                features["sentiment_dictionary_source"].tolist(),
                ["openai_chatgpt_sun2026_style:fixture-chatgpt", "openai_chatgpt_sun2026_style:fixture-chatgpt"],
            )
            self.assertIn("sentiment_raw_response", features.columns)


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
