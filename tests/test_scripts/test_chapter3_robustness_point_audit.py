"""Semantic checks for original-point, equal-fold robustness reporting."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.rq123 import chapter3_robustness_point_audit as audit
from scripts.rq123.shared_panel_bootstrap_core import make_schedule, prepare_panel, run_bootstrap


class RobustnessPointAuditTests(unittest.TestCase):
    def test_original_equal_fold_mae_differs_from_pooled_and_log_contrast(self) -> None:
        rows = []
        for seed in (42, 202, 404):
            for fold, count, focal, reference in (("f1", 1, 1., 2.), ("f2", 3, 9., 6.)):
                for pair in range(count):
                    for model, value in (("a", focal), ("b", reference)):
                        rows.append(dict(condition=model, seed=seed, fold=fold,
                            pair_id=f"{fold}-{pair}", session_id=f"{fold}-{pair}",
                            value=value, persistence_mae=reference))
        frame = pd.DataFrame(rows)
        summary, cells = audit.point_summary(frame)
        self.assertEqual(len(cells), 12)
        np.testing.assert_allclose(summary.observed_mean_mae, [5., 4.])
        self.assertEqual(frame[frame.condition.eq("a")].value.mean(), 7.)
        panel = prepare_panel(frame)
        result = run_bootstrap(panel, make_schedule(panel, iterations=100, rng_seed=19))
        contrast = result.contrast("a", "b")
        self.assertAlmostEqual(contrast["point"], np.log([.5, 1.5]).mean())
        self.assertNotAlmostEqual(contrast["point"], np.log(5. / 4.))

    def test_holm_does_not_mix_families_or_adjust_descriptive_contrasts(self) -> None:
        frame = pd.DataFrame([
            dict(contrast_id="a", family_id="primary", apply_holm=True, p_one=.02),
            dict(contrast_id="b", family_id="primary", apply_holm=True, p_one=.08),
            dict(contrast_id="c", family_id="text", apply_holm=True, p_one=.01),
            dict(contrast_id="d", family_id="diagnostic", apply_holm=False, p_one=.03),
        ])
        result = audit.apply_holm(frame)
        np.testing.assert_allclose(result.reported_p, [.04, .08, .01, .03])
        self.assertEqual(result.family_size.tolist(), [2, 2, 1, 0])
        self.assertTrue(pd.isna(result.iloc[3].holm_p))

    def test_unknown_historical_recipe_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Historical recipe mismatch"):
            audit.contrast_recipes(pd.DataFrame({"contrast_id": ["unknown"]}), alignment=True)

    def test_full_archive_replays_and_preserves_frozen_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            arch, align, capacity = (base / name for name in ("arch", "align", "capacity"))
            input_paths = []
            for root, is_alignment in ((arch, False), (align, True)):
                (root / "predictions/cells").mkdir(parents=True)
                (root / "analysis").mkdir()
                rows = []
                for tolerance in ((5, 10, 15, 20, 30) if is_alignment else (5,)):
                    for role in (("common_5m_primary", "own_tolerance_secondary")
                                 if is_alignment and tolerance > 5 else ("common_5m_primary",)):
                        for seed in (42, 202, 404):
                            for fold in audit.EXPECTED_FOLDS:
                                for pair in range(2):
                                    for model in ("a", "b"):
                                        rows.append(dict(model_id=model, text_condition="matched",
                                            train_tolerance_minutes=tolerance, panel_role=role,
                                            seed=seed, fold=fold, pair_id=f"{fold}-{pair}",
                                            session_id=f"{fold}-{pair}", persistence_mae=2. + pair,
                                            target_mae=(2. + pair) * (1 + seed / 100000) *
                                                (1.001 if model == "a" else 1.)))
                path = root / "predictions/cells/test.pair_metrics.csv"
                pd.DataFrame(rows).to_csv(path, index=False)
                input_paths.append(path)
                pd.DataFrame({"contrast_id": ["test"]}).to_csv(root / "analysis/bootstrap_results.csv", index=False)
            (capacity / "analysis").mkdir(parents=True)
            for quarter, count in (("q3", 135), ("q4", 143)):
                rows = [dict(tolerance_minutes=5, panel="core", stratum_type="overall",
                    capacity_profile="micro", seed=seed, pair_id=f"p{pair}",
                    model_mae=1. + pair / 100, persistence_mae=1.01 + pair / 100)
                    for seed in (42, 202, 404) for pair in range(count)]
                pd.DataFrame(rows).to_csv(capacity / f"analysis/film_nolp_capacity_{quarter}_pair_metrics.csv.gz", index=False)

            def recipes(_legacy: pd.DataFrame, *, alignment: bool) -> list[dict]:
                tolerance = 5 if alignment else None
                return [dict(contrast_id="a_vs_b", focal=audit.condition("a", "matched", tolerance),
                    reference=audit.condition("b", "matched", tolerance), family_id="one",
                    apply_holm=True, scope="primary")]

            original_bytes = {path: path.read_bytes() for path in input_paths}
            output = base / "audit"
            with patch.multiple(audit, ARCHITECTURE=arch, ALIGNMENT=align, CAPACITY=capacity,
                                EXPECTED_FOLDS={fold: 2 for fold in audit.EXPECTED_FOLDS}), \
                 patch.object(audit, "contrast_recipes", side_effect=recipes):
                audit.run(output)
                self.assertTrue(audit.verify(output)["passed"])
                with self.assertRaises(FileExistsError):
                    audit.run(output)
            for path, value in original_bytes.items():
                self.assertEqual(path.read_bytes(), value)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["iterations"], 10000)
            self.assertEqual({job["schedule"] for job in manifest["jobs"]}, {"shared_schedule.npz"})
            capacity_summary = pd.read_csv(output / "capacity_historical_summary.csv")
            self.assertTrue(capacity_summary.fold_count.eq(1).all())
            input_paths[0].write_bytes(original_bytes[input_paths[0]] + b"\n")
            with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                audit.verify(output)


if __name__ == "__main__":
    unittest.main()
