"""Tests for CNN WGAN post-hoc short-ATM evaluation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cnn_wgan.short_atm_eval import ShortATMEvalConfig, evaluate_many, evaluate_sample_payload


class CnnWGANShortATMEvalTests(unittest.TestCase):
    def _sample_payload(self):
        return {
            "sample_id": "sample_1",
            "global_index": 7,
            "news_timestamp_utc": "2023-01-01T00:00:00Z",
            "strike_grid": [0.95, 1.0, 1.1],
            "maturity_days_grid": [30.0, 90.0],
            "current_surface": [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]],
            "generated_surface": [[1.0, 3.0, 7.0], [7.0, 11.0, 13.0]],
            "target_surface": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        }

    def test_evaluate_sample_payload_uses_pure_and_weighted_masks(self):
        config = ShortATMEvalConfig(
            atm_short_range=0.05,
            atm_short_max_days=60.0,
            recon_atm_range=0.05,
            recon_atm_short_end_max_days=60.0,
            recon_atm_multiplier=3.0,
            label="unit",
        )

        row = evaluate_sample_payload(self._sample_payload(), config=config, source_path="sample.json")

        self.assertEqual(row["sample_id"], "sample_1")
        self.assertEqual(row["pure_mask_cell_count"], 2)
        self.assertEqual(row["weighted_mask_cell_count"], 2)
        self.assertAlmostEqual(row["mae"], 7.0)
        self.assertAlmostEqual(row["current_mae"], 7.0)
        self.assertAlmostEqual(row["atm_short_pure_mae"], 2.0)
        self.assertAlmostEqual(row["current_atm_short_pure_mae"], 3.0)
        self.assertAlmostEqual(row["atm_short_pure_mae_gap_vs_current"], -1.0)
        self.assertAlmostEqual(row["short_atm_weighted_mae"], 5.0)
        self.assertAlmostEqual(row["current_short_atm_weighted_mae"], 5.4)
        self.assertAlmostEqual(row["short_atm_mae_gap_vs_current"], -0.4)

    def test_evaluate_many_writes_summary_and_per_sample_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            samples_dir = root / "samples"
            samples_dir.mkdir()
            (samples_dir / "sample_1.json").write_text(json.dumps(self._sample_payload()), encoding="utf-8")

            _rows, summary, artifacts = evaluate_many(
                samples_dir=samples_dir,
                config_values={
                    "label": "unit",
                    "atm_short_range": 0.05,
                    "atm_short_max_days": 60.0,
                    "recon_atm_range": 0.05,
                    "recon_atm_short_end_max_days": 60.0,
                    "recon_atm_multiplier": 3.0,
                },
                output_dir=root / "short_atm_eval",
            )

            self.assertEqual(summary["sample_count"], 1)
            self.assertAlmostEqual(summary["atm_short_pure_mae_gap_vs_current"], -1.0)
            self.assertTrue(Path(artifacts["summary_json"]).exists())
            self.assertTrue(Path(artifacts["per_sample_csv"]).exists())


if __name__ == "__main__":
    unittest.main()
