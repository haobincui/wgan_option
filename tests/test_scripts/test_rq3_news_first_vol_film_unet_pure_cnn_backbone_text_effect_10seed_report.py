"""Report tests for the Pure-CNN parent -> FiLM text-effect experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_analysis as analysis,
)
from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_report as report,
)
from tests.test_scripts.test_rq3_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_analysis import (
    FOLDS,
    SEEDS,
    _intervention,
    _sparse_trajectory,
    _standard,
)


def _result() -> dict[str, object]:
    result = analysis.analyze_backbone_text_effect(
        _standard(),
        _intervention(),
        _sparse_trajectory(),
        expected_seeds=SEEDS,
        expected_folds=FOLDS,
        bootstrap_iterations=50,
        bootstrap_seed=17,
        minimum_nonworse_seeds=2,
        minimum_nonworse_folds=2,
    )
    for name in ("main_comparisons", "intervention_comparisons"):
        result[name]["bootstrap_iterations"] = 10_000
    result["conclusion"]["bootstrap_iterations_per_comparison"] = 10_000
    return result


class TextEffectReportTests(unittest.TestCase):
    def test_writes_matching_self_contained_reports_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = _result()
            analysis_paths = report.write_analysis_artifacts(root / "analysis", result)
            report_paths = report.write_text_effect_report(
                root / "report",
                result,
                metadata={
                    "interpretation": report.INTERPRETATION,
                    "checkpoint_allowlist_sha256": "a" * 64,
                },
            )
            markdown = report_paths["markdown"].read_text(encoding="utf-8")
            html = report_paths["html"].read_text(encoding="utf-8")
            artifact = json.loads(report_paths["artifact"].read_text(encoding="utf-8"))
            self.assertIn("stable_text_mae_increment", markdown)
            self.assertIn("Same-checkpoint text interventions", markdown)
            self.assertIn("stable_text_mae_increment", html)
            self.assertNotIn("<script", html.lower())
            self.assertNotIn("<link", html.lower())
            self.assertTrue(artifact["reports"]["html"]["self_contained"])
            for name in ("markdown", "html"):
                path = report_paths[name]
                self.assertEqual(
                    artifact["reports"][name]["sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            manifest = json.loads(
                analysis_paths["manifest"].read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["artifacts"]["main_comparisons"]["row_count"], 2)

    def test_report_rejects_nonformal_bootstrap_count(self) -> None:
        result = _result()
        result["main_comparisons"]["bootstrap_iterations"] = 9_999
        with self.assertRaisesRegex(report.TextEffectReportError, "10,000"):
            report.render_text_effect_markdown(result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
