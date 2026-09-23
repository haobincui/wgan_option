"""End-to-end archive and family checks for the analysis-only correction."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.rq123 import chapter3_shared_panel_bootstrap_v2 as analysis


class Chapter3BootstrapArchiveTests(unittest.TestCase):
    def test_holm_families_span_jobs_and_respect_alternative(self) -> None:
        frame = pd.DataFrame([
            dict(job_id="j1", contrast_id="a", family_id="joint", apply_holm=True,
                 alternative="one_sided", p_one=.02, p_two=.04),
            dict(job_id="j2", contrast_id="b", family_id="joint", apply_holm=True,
                 alternative="one_sided", p_one=.08, p_two=.16),
            dict(job_id="j3", contrast_id="c", family_id="other", apply_holm=True,
                 alternative="two_sided", p_one=.015, p_two=.03),
            dict(job_id="j4", contrast_id="d", family_id="diagnostic", apply_holm=False,
                 alternative="two_sided", p_one=.11, p_two=.22),
        ])
        result = analysis._holm(frame)
        np.testing.assert_allclose(result.reported_p, [.04, .08, .03, .22])
        self.assertEqual(result.family_size.tolist(), [2, 2, 1, 0])
        self.assertEqual(result.significance_stars.tolist(), ["**", "*", "**", ""])
        self.assertTrue(pd.isna(result.iloc[3].holm_p))

    def test_old_new_rejects_changed_estimand_and_marks_missing_legacy(self) -> None:
        current = pd.DataFrame([dict(job_id="j", contrast_id="c", point=.001,
                                    bootstrap_se=.002, ci_lower=-.003, ci_upper=.004,
                                    p_one=.6, holm_p=.6)])
        row = analysis._old_new(current, {}).iloc[0]
        self.assertFalse(row.legacy_available)
        with self.assertRaisesRegex(ValueError, "Observed estimand changed"):
            analysis._old_new(current, {("j", "c"): {"point": .002}})

    def test_analysis_replays_all_draws_and_cannot_overwrite_inputs_or_results(self) -> None:
        rows = []
        for seed in (11, 22):
            for fold in ("f1", "f2"):
                for session in range(3):
                    for pair in range(session + 1):
                        reference = 1 + session * .2 + pair * .03
                        for arm, ratio in (("a", .99 + session * .01), ("b", 1.0)):
                            rows.append(dict(condition=arm, seed=seed, fold=fold,
                                pair_id=f"{fold}-s{session}-p{pair}",
                                session_id=f"{fold}-s{session}", value=reference * ratio))
        panel = pd.DataFrame(rows)
        jobs = [dict(job_id="test", panel=panel, estimand="equal_cell", metadata={},
            contrasts=[dict(contrast_id="a_vs_b", focal="a", reference="b",
                family_id="one", alternative="one_sided", apply_holm=True)])]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config_path = base / "config.yaml"
            config_path.write_text("schema_version: 2\n", encoding="utf-8")
            source_path = base / "source.csv"
            panel.to_csv(source_path, index=False)
            original_source = source_path.read_bytes()
            manifest = [dict(path=str(source_path), sha256=analysis.sha256_file(source_path),
                             source_id="test", role="data")]
            config = {"bootstrap": {"iterations": 10000, "rng_seed": 123}}
            root = base / "corrected_v2"
            with patch.object(analysis, "load_sources", return_value=({}, manifest, config)), \
                    patch.object(analysis, "build_jobs", return_value=jobs):
                output = analysis.run_analysis(config_path, root)
                self.assertEqual(output, root)
                self.assertTrue(analysis.verify_analysis(root)["passed"])
                self.assertEqual(source_path.read_bytes(), original_source)
                original_manifest = (root / "analysis_manifest.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    analysis.run_analysis(config_path, root)
                self.assertEqual((root / "analysis_manifest.json").read_bytes(), original_manifest)
            recorded = json.loads((root / "analysis_manifest.json").read_text())
            self.assertIsInstance(recorded["rng_seed"], str)
            with np.load(root / recorded["jobs"][0]["draws_path"]) as draws:
                self.assertEqual(draws["log_ratio_draws"].shape, (10000, 1))
            with source_path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                analysis.verify_analysis(root, replay=False)

    def test_v1_result_cannot_be_claimed_as_corrected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis_manifest.json").write_text(json.dumps(
                {"schema_version": 1, "kind": "old_nested_bootstrap"}))
            with self.assertRaisesRegex(ValueError, "Not a corrected v2"):
                analysis.verify_analysis(root)


if __name__ == "__main__":
    unittest.main()
