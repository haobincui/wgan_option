"""Independent replay guards for frozen Chapter 3 main inference."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.rq123.chapter3_main_inference_audit import (
    _canonical_from_raw, _check_hashes, _check_point_estimates, _holm, sha256,
)


class MainInferenceAuditTests(unittest.TestCase):
    def test_raw_binding_preserves_pair_values_and_rejects_persistence_drift(self):
        raw = pd.DataFrame([
            {"arm": arm, "seed": 42, "fold": "f1", "pair_id": "p", "session_id": "s",
             "target_mae": error, "persistence_mae": 3.0}
            for arm, error in (("matched", 1.0), ("film_zero_text", 2.0))
        ])
        panel = pd.DataFrame({"condition": ["film_lp_matched", "film_zero_text", "persistence"],
                              "fold": ["f1"] * 3})
        result = _canonical_from_raw(raw, "rq3_branch", panel).set_index("condition")
        self.assertEqual(result.loc["film_lp_matched", "value"], 1.0)
        self.assertEqual(result.loc["persistence", "value"], 3.0)
        raw.loc[1, "persistence_mae"] = 4.0
        with self.assertRaisesRegex(ValueError, "Persistence differs"):
            _canonical_from_raw(raw, "rq3_branch", panel)

    def test_holm_family_spans_folds_and_preserves_unadjusted_row(self):
        frame = pd.DataFrame([
            {"job_id": "q1", "contrast_id": "a", "family_id": "rolling", "apply_holm": True,
             "alternative": "one_sided", "p_one": .02, "p_two": .04},
            {"job_id": "q2", "contrast_id": "b", "family_id": "rolling", "apply_holm": True,
             "alternative": "one_sided", "p_one": .20, "p_two": .40},
            {"job_id": "overall", "contrast_id": "c", "family_id": "overall", "apply_holm": False,
             "alternative": "two_sided", "p_one": .10, "p_two": .20},
        ])
        result = _holm(frame)
        np.testing.assert_allclose(result.reported_p, [.04, .20, .20])
        self.assertEqual(result.family_size.tolist(), [2, 2, 0])

    def test_point_crosscheck_rejects_pooled_fold_weighting(self):
        # The second fold has three pairs: pooling gives 16/6, equal folds 2.5.
        rows = [{"condition": "lp_matched", "seed": 42, "fold": fold, "value": value}
                for fold, values in (("f1", [1.]), ("f2", [3., 3., 3.]),
                                     ("f3", [3.]), ("f4", [3.])) for value in values]
        panel = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = pd.DataFrame({"dataset": ["rq12"] * 4, "arm": ["lp_matched"] * 4,
                                  "seed": [42] * 4, "fold": ["f1", "f2", "f3", "f4"],
                                  "observed_mean_mae": [1., 3., 3., 3.]})
            cells.to_csv(root / "cell_summary.csv", index=False)
            cells.drop(columns="seed").to_csv(root / "fold_summary.csv", index=False)
            overall = pd.DataFrame({"dataset": ["rq12"], "arm": ["lp_matched"],
                                    "observed_mean_mae": [2.5]})
            overall.to_csv(root / "overall_summary.csv", index=False)
            checks = _check_point_estimates("direct_overall", panel, root)
            self.assertEqual(len(checks), 9)
            overall.loc[0, "observed_mean_mae"] = 16 / 6
            overall.to_csv(root / "overall_summary.csv", index=False)
            with self.assertRaises(AssertionError):
                _check_point_estimates("direct_overall", panel, root)

    def test_frozen_hash_guard_detects_changed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            path.write_text("value\n1\n")
            records = [{"path": str(path), "sha256": sha256(path)}]
            _check_hashes(records)
            path.write_text("value\n2\n")
            with self.assertRaisesRegex(ValueError, "Frozen audit input changed"):
                _check_hashes(records)


if __name__ == "__main__":
    unittest.main()
