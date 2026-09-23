"""Semantic regression checks for the read-only point-estimate audit."""

import unittest

import numpy as np
import pandas as pd

from scripts.rq123.audit_evaluation_estimands import (
    canonical_panel,
    contrast_points,
    require_same_predictions,
    summarize,
    verify_saved_summary,
)


def fixture():
    rows = []
    for condition in ["matched", "reference"]:
        for seed in [42, 202]:
            for fold, values in [("f1_2023q1", [1., 3.]), ("f2_2023q2", [10.])]:
                for index, value in enumerate(values):
                    rows.append({"arm": condition, "seed": seed, "fold": fold,
                                 "pair_id": f"{fold}_p{index}", "session_id": f"{fold}_s0",
                                 "target_mae": value if condition == "matched" else 2 * value,
                                 "persistence_mae": 3 * value})
    return pd.DataFrame(rows)


class EvaluationEstimandTests(unittest.TestCase):
    def test_equal_cell_is_not_pooled_pair(self):
        panel = canonical_panel(fixture(), "arm")
        rows = summarize(panel, by_fold=False, reference="reference")
        matched = next(row for row in rows if row["condition"] == "matched")
        self.assertEqual(matched["observed_mean_mae"], 6.)
        self.assertNotEqual(matched["observed_mean_mae"], panel[panel.condition.eq("matched")].target_mae.mean())
        self.assertEqual(matched["normalized_mae"], .5)
        self.assertAlmostEqual(matched["improvement_vs_persistence_percent"], 200 / 3)

    def test_pairs_not_sessions_get_equal_point_weights(self):
        frame = fixture()
        extra = frame[frame.fold.eq("f1_2023q1") & frame.pair_id.str.endswith("p1")].copy()
        extra["pair_id"] = "f1_2023q1_p2"
        extra["session_id"] = "f1_2023q1_s1"
        frame = pd.concat([frame, extra], ignore_index=True)
        rows = summarize(canonical_panel(frame, "arm"), by_fold=True, reference="matched")
        actual = next(row["observed_mean_mae"] for row in rows if row["condition"] == "matched" and row["fold"] == "f1_2023q1")
        self.assertAlmostEqual(actual, 7 / 3)
        self.assertNotEqual(actual, 2.5)  # Equal-session mean would be (2 + 3)/2.

    def test_duplicate_rows_rejected(self):
        frame = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            canonical_panel(pd.concat([frame, frame.iloc[:1]]), "arm")

    def test_missing_seed_condition_market_row_rejected(self):
        with self.assertRaisesRegex(ValueError, "lineage"):
            canonical_panel(fixture().iloc[1:], "arm")

    def test_persistence_drift_rejected(self):
        frame = fixture()
        frame.loc[0, "persistence_mae"] += .1
        with self.assertRaisesRegex(ValueError, "Persistence"):
            canonical_panel(frame, "arm")

    def test_nonfinite_and_nonpositive_values_rejected(self):
        for value in [np.nan, np.inf, -1, 0]:
            frame = fixture()
            frame.loc[0, "target_mae"] = value
            with self.assertRaises(ValueError):
                canonical_panel(frame, "arm")

    def test_matched_relabelled_input_reconciles_exactly(self):
        panel = canonical_panel(fixture(), "arm")
        other = panel.copy()
        other.loc[other.condition.eq("matched"), "condition"] = "matched_input"
        require_same_predictions(panel, "matched", other, "matched_input")
        other.loc[0, "target_mae"] += 1e-12
        with self.assertRaisesRegex(ValueError, "predictions"):
            require_same_predictions(panel, "matched", other, "matched_input")

    def test_geometric_and_arithmetic_contrasts_are_distinct(self):
        panel = canonical_panel(fixture(), "arm")
        selected = panel.condition.eq("reference") & panel.fold.eq("f2_2023q2")
        panel.loc[selected, "target_mae"] = 5.
        row = contrast_points(panel, "matched", ["reference"])[0]
        self.assertAlmostEqual(row["mean_cell_log_mae_ratio"], 0.)
        self.assertNotAlmostEqual(row["ratio_of_arithmetic_maes"], 1.)

    def test_saved_summary_checks_observed_not_bootstrap_mean(self):
        rows = [{"condition": "matched", "fold": "all_four_folds", "observed_mean_mae": 6.}]
        summary = pd.DataFrame([{"condition": "matched", "seed_count": 10, "fold_count": 4,
                                 "pair_count": 500, "observed_mean_mae": 6., "bootstrap_mean_mae": 7.}])
        self.assertEqual(verify_saved_summary(summary, rows), 1)
        summary.loc[0, "observed_mean_mae"] = 7.
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_saved_summary(summary, rows)


if __name__ == "__main__":
    unittest.main()
