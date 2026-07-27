from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from scripts.rq123.audit_sentiment_scores import prepare
from scripts.rq123.corrected_pipeline import snapshot_inputs
from scripts.raw_vol.raw_vol_pipeline import (
    _precalibration_audit_summary,
)


class Rq123AuditTests(unittest.TestCase):
    def test_sentiment_audit_writes_deterministic_templates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            news_path = root / "news.xlsx"
            feature_path = root / "sentiment.xlsx"
            manifest_path = root / "sentiment_manifest.json"
            cache_path = root / "cache.jsonl"
            output_dir = root / "audit"
            pd.DataFrame(
                {
                    "ArticleID": ["A-1", "A-2", "A-3", "A-4"],
                    "SourceFile": ["one", "two", "three", "four"],
                    "LP": [
                        "Rates moved after inflation data.",
                        "The central bank held rates.",
                        "Investors reduced risky positions.",
                        "",
                    ],
                }
            ).to_excel(news_path, index=False)
            pd.DataFrame(
                {
                    "news_row_id": [1, 2, 3, 4],
                    "sentiment_embedding": [
                        "[0.1, 0.2, 0.3]",
                        "[0.2, 0.3, 0.4]",
                        "[0.8, 0.7, 0.9]",
                        "[0.0, 0.0, 0.0]",
                    ],
                }
            ).to_excel(feature_path, index=False)
            manifest_path.write_text(
                json.dumps(
                    {
                        "model_id": "gpt-5.4-mini",
                        "prompt_version": "sun2026_zero_shot_chatgpt_v1",
                        "max_input_chars": 6000,
                    }
                ),
                encoding="utf-8",
            )
            cache_path.write_text('{"cache_key":"one"}\n', encoding="utf-8")
            exit_code = prepare(
                Namespace(
                    news_workbook=str(news_path),
                    sentiment_features=str(feature_path),
                    sentiment_manifest=str(manifest_path),
                    sentiment_cache=str(cache_path),
                    output_dir=str(output_dir),
                    manual_sample_size=3,
                    repeat_sample_size=2,
                    seed=42,
                )
            )
            self.assertEqual(exit_code, 0)
            manual = pd.read_csv(
                output_dir / "sentiment_manual_annotation_sample.csv"
            )
            repeats = pd.read_csv(
                output_dir / "sentiment_repeat_scoring_results.csv"
            )
            manifest = json.loads(
                (
                    output_dir / "sentiment_audit_manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(len(manual), 3)
            self.assertFalse(manual["lp_text"].fillna("").eq("").any())
            self.assertEqual(len(repeats), 4)
            self.assertEqual(
                manifest["status"],
                "pending_human_annotation_and_independent_repeat_scoring",
            )
            self.assertEqual(
                manifest["empty_lp_rows_excluded_from_audit"],
                1,
            )

    def test_source_snapshot_hashes_and_preserves_directory_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "nested").mkdir()
            (source / "nested/input.csv").write_text(
                "a,b\n1,2\n",
                encoding="utf-8",
            )
            pipeline = root / "pipeline"
            exit_code = snapshot_inputs(
                Namespace(
                    pipeline_root=str(pipeline),
                    input=[f"raw_option_data={source}"],
                )
            )
            self.assertEqual(exit_code, 0)
            snapshot = (
                pipeline
                / "inputs/source_snapshot/raw_option_data/nested/input.csv"
            )
            self.assertEqual(
                snapshot.read_text(encoding="utf-8"),
                "a,b\n1,2\n",
            )
            manifest = pd.read_csv(
                pipeline / "inputs/source_manifest.csv"
            )
            self.assertEqual(len(manifest), 1)
            self.assertEqual(
                manifest.iloc[0]["category"],
                "raw_option_data",
            )
            self.assertEqual(len(manifest.iloc[0]["sha256"]), 64)

    def test_source_snapshot_glob_excludes_unmatched_aggregate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "daily_2022.csv.gz").write_bytes(b"daily")
            (source / "merged.csv.gz").write_bytes(b"aggregate")
            pipeline = root / "pipeline"
            snapshot_inputs(
                Namespace(
                    pipeline_root=str(pipeline),
                    input=[
                        f"raw_option_data={source}/daily_*.csv.gz"
                    ],
                )
            )
            snapshot_root = (
                pipeline / "inputs/source_snapshot/raw_option_data"
            )
            self.assertTrue((snapshot_root / "daily_2022.csv.gz").is_file())
            self.assertFalse((snapshot_root / "merged.csv.gz").exists())

    def test_precalibration_audit_requires_exact_frozen_curve_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            curve_sha = "a" * 64
            pd.DataFrame(
                {
                    "pricing_model": ["black76"],
                    "underlying_match_mode": ["last_prior_trade"],
                    "underlying_staleness_seconds": [12.0],
                    "is_otm": [True],
                    "passes_precalib_filter": [True],
                    "weight": [3.0],
                    "rate_curve_sha256": [curve_sha],
                }
            ).to_csv(
                root / "surface-raw-excel-precalib-points.csv",
                index=False,
            )
            matched = _precalibration_audit_summary(
                root,
                expected_rate_curve_sha256=curve_sha,
            )
            mismatched = _precalibration_audit_summary(
                root,
                expected_rate_curve_sha256="b" * 64,
            )
            self.assertTrue(matched["precalibration_corrected_inputs_ok"])
            self.assertFalse(
                mismatched["precalibration_corrected_inputs_ok"]
            )


if __name__ == "__main__":
    unittest.main()
