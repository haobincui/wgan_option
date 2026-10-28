"""Dual scoring estimands and frozen prediction provenance tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.rq3.rq12_vega_rescore import (
    _add_skills, checked_file, control_new_deltas, frozen_prediction_entries,
    score_arrays, summarize_pair_scores, summary_control_new_deltas, verify_shared_panel,
)
from wgan_option.utils.vega_weights import sha256_file


class DualRescoreTests(unittest.TestCase):
    def test_linear_vega_mae_and_same_mask_persistence(self):
        current = np.asarray([[0.10, 0.12, 100.]])
        truth = np.asarray([[0.11, 0.14, 900.]])
        prediction = np.asarray([[0.13, 0.14, -900.]])
        weights = np.asarray([[1., 9., 0.]])
        result = score_arrays(prediction, truth, current, np.asarray([[True, True, False]]), weights)
        self.assertAlmostEqual(result["plain_mae"], .01)
        self.assertAlmostEqual(result["vega_mae"], .002)
        self.assertAlmostEqual(result["persistence_plain_mae"], .015)
        self.assertAlmostEqual(result["persistence_vega_mae"], .019)
        scaled = score_arrays(prediction, truth, current, np.asarray([[True, True, False]]), weights * 100)
        self.assertAlmostEqual(result["vega_mae"], scaled["vega_mae"])
        with self.assertRaises(ValueError):
            score_arrays(prediction, truth, current, np.ones((1, 3), dtype=bool), weights)

    def test_equal_fold_equal_seed_not_pooled_pair_average(self):
        rows = []
        for seed, errors in [(42, (1., 3.)), (202, (5., 7.))]:
            for fold, (error, pair_count) in enumerate(zip(errors, (1, 9)), 1):
                for pair in range(pair_count):
                    rows.append({"experiment_version": "unweighted_control", "architecture": "film", "arm": "lp_matched", "tolerance_minutes": 5, "seed": seed, "fold": f"f{fold}", "pair_id": str(pair), "plain_mae": error, "vega_mae": error / 2, "persistence_plain_mae": 10., "persistence_vega_mae": 5.})
        frame = pd.DataFrame(rows)
        fold, seed, overall = summarize_pair_scores(frame)
        self.assertEqual(len(fold), 4)
        self.assertEqual(seed["plain_mae"].tolist(), [2., 6.])
        self.assertEqual(overall.iloc[0]["plain_mae"], 4.)
        self.assertAlmostEqual(overall.iloc[0]["plain_skill_vs_persistence"], .6)
        self.assertNotEqual(overall.iloc[0]["plain_mae"], frame["plain_mae"].mean())

    def test_pair_delta_requires_same_truth_support_and_weights(self):
        rows = []
        for version, error in [("unweighted_control", .02), ("vega_retrained", .01)]:
            rows.append({"experiment_version": version, "architecture": "film", "arm": "lp_matched", "fold": "f1", "seed": 42, "tolerance_minutes": 5, "pair_id": "p1", "pair_truth_fingerprint": "same_truth", "support_mask_fingerprint": "same_support", "cell_weight_fingerprint": "same_weights", "plain_mae": error, "vega_mae": error / 2, "persistence_plain_mae": .04, "persistence_vega_mae": .02})
        frame = _add_skills(pd.DataFrame(rows))
        delta = control_new_deltas(frame)
        self.assertAlmostEqual(delta.iloc[0]["delta_plain_mae_new_minus_control"], -.01)
        self.assertAlmostEqual(delta.iloc[0]["delta_vega_skill_vs_persistence_new_minus_control"], .25)
        changed = frame.copy()
        changed.loc[1, "cell_weight_fingerprint"] = "future_derived_weights"
        with self.assertRaisesRegex(ValueError, "cell_weight_fingerprint"):
            control_new_deltas(changed)
        changed = frame.copy()
        changed.loc[1, "pair_id"] = "p2"
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            control_new_deltas(changed)

    def test_shared_panel_rejects_mismatched_support_or_pair_universe(self):
        panel = {"p1": {"pair_truth_fingerprint": "a", "support_mask_fingerprint": "b", "cell_weight_fingerprint": "c"}}
        verify_shared_panel(panel, panel)
        with self.assertRaisesRegex(ValueError, "pair universes"):
            verify_shared_panel(panel, {})
        changed = {"p1": dict(panel["p1"], support_mask_fingerprint="new_mask")}
        with self.assertRaisesRegex(ValueError, "support_mask_fingerprint"):
            verify_shared_panel(panel, changed)

    def test_summary_skill_delta_uses_ratio_of_aggregates(self):
        rows = []
        for version, scale in [("unweighted_control", 1.), ("vega_retrained", .5)]:
            for pair, error, baseline in [("p1", .02, .04), ("p2", .2, .2)]:
                rows.append({"experiment_version": version, "architecture": "film", "arm": "lp_matched", "tolerance_minutes": 5, "seed": 42, "fold": "f1", "pair_id": pair, "plain_mae": error * scale, "vega_mae": error * scale, "persistence_plain_mae": baseline, "persistence_vega_mae": baseline})
        pairs = _add_skills(pd.DataFrame(rows))
        _, _, summary = summarize_pair_scores(pairs)
        delta = summary_control_new_deltas(summary, ["architecture", "arm", "tolerance_minutes"])
        self.assertAlmostEqual(delta.iloc[0]["delta_plain_skill_vs_persistence_new_minus_control"], .055 / .12)
        old = pairs[pairs["experiment_version"] == "unweighted_control"]
        new = pairs[pairs["experiment_version"] == "vega_retrained"]
        self.assertNotAlmostEqual(delta.iloc[0]["delta_plain_skill_vs_persistence_new_minus_control"], new["plain_skill_vs_persistence"].mean() - old["plain_skill_vs_persistence"].mean())

    def test_frozen_index_hash_and_completion_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "registry").mkdir()
            index = root / "predictions.csv"
            pd.DataFrame([{"job_id": "job", "arm": "lp_matched", "fold": "f1", "seed": 42, "tolerance_minutes": 5, "prediction_path": "p.csv", "prediction_sha256": "x", "prediction_manifest_path": "p.json", "prediction_manifest_sha256": "y"}]).to_csv(index, index=False)
            registry = {"predictions_frozen": True, "prediction_manifest_path": str(index), "prediction_manifest_sha256": sha256_file(index), "expected_prediction_cells": 1}
            (root / "registry/task_registry.json").write_text(json.dumps(registry))
            entries, _ = frozen_prediction_entries(root)
            self.assertEqual(len(entries), 1)
            checked_file(index, registry["prediction_manifest_sha256"])
            with index.open("a") as handle:
                handle.write("drift\n")
            with self.assertRaisesRegex(ValueError, "SHA drift"):
                frozen_prediction_entries(root)
            registry["predictions_frozen"] = False
            (root / "registry/task_registry.json").write_text(json.dumps(registry))
            with self.assertRaisesRegex(ValueError, "not frozen"):
                frozen_prediction_entries(root)


if __name__ == "__main__":
    unittest.main()
