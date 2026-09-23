from __future__ import annotations

import unittest

import pandas as pd

from scripts.rq3.news_first_vol_label_reliability_analysis import (
    LabelReliabilityAnalysisError,
    build_arm_comparisons,
    holm_adjust,
    q3_checkpoint_allowlist,
    select_reliability_winner,
    validate_q3_prediction_frame,
)


def _pair_metrics() -> pd.DataFrame:
    multipliers = {"A": 1.0, "B": 0.97, "C": 0.90, "D": 0.94}
    rows = []
    for arm, multiplier in multipliers.items():
        for seed in (42, 202, 404):
            for fold_index, fold in enumerate(("F1", "F2", "F3", "F4"), 1):
                for pair_index in range(8):
                    rows.append(
                        {
                            "arm_id": arm,
                            "fold_id": fold,
                            "seed": seed,
                            "pair_id": f"{fold}_pair_{pair_index}",
                            "session_id": f"{fold}_session_{pair_index // 2}",
                            "model_mae": multiplier
                            * (1.0 + 0.01 * fold_index + 0.001 * pair_index),
                        }
                    )
    return pd.DataFrame(rows)


class LabelReliabilityAnalysisTests(unittest.TestCase):
    def test_hierarchical_bootstrap_and_holm_gate_are_deterministic(self) -> None:
        first = build_arm_comparisons(_pair_metrics(), iterations=250, seed=123)
        second = build_arm_comparisons(_pair_metrics(), iterations=250, seed=123)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["candidate_arm"]), {"B", "C", "D"})
        self.assertTrue(first["passes_gate"].all())
        self.assertTrue((first["ci_upper"] < 0.0).all())
        selected = select_reliability_winner(first)
        self.assertEqual(selected["status"], "selected")
        self.assertEqual(selected["selected_arm"], "C")
        self.assertFalse(selected["q3_used_for_selection"])
        self.assertFalse(selected["q4_accessed"])

    def test_one_se_priority_is_c_then_b_then_d(self) -> None:
        rows = []
        for arm, mean in (("B", -0.101), ("C", -0.100), ("D", -0.099)):
            rows.append(
                {
                    "candidate_arm": arm,
                    "mean_log_ratio": mean,
                    "bootstrap_se": 0.01,
                    "ci_upper": -0.01,
                    "holm_adjusted_p": 0.01,
                    "nonworse_fold_count": 4,
                    "nonworse_seed_count": 3,
                    "passes_gate": True,
                }
            )
        selected = select_reliability_winner(pd.DataFrame(rows))
        self.assertEqual(selected["best_arm"], "B")
        self.assertEqual(selected["selected_arm"], "C")
        self.assertEqual(selected["eligible_one_se_arms"], ["C", "B", "D"])

    def test_no_passing_arm_is_a_terminal_no_winner(self) -> None:
        rows = []
        for arm in ("B", "C", "D"):
            rows.append(
                {
                    "candidate_arm": arm,
                    "mean_log_ratio": 0.01,
                    "bootstrap_se": 0.02,
                    "ci_upper": 0.03,
                    "holm_adjusted_p": 1.0,
                    "nonworse_fold_count": 0,
                    "nonworse_seed_count": 0,
                    "passes_gate": False,
                }
            )
        selected = select_reliability_winner(pd.DataFrame(rows))
        self.assertEqual(selected["status"], "terminal_no_winner")
        self.assertIsNone(selected["selected_arm"])

    def test_unpaired_arm_panel_fails_closed(self) -> None:
        frame = _pair_metrics()
        bad = frame.drop(
            frame[
                (frame["arm_id"] == "B")
                & (frame["seed"] == 42)
                & (frame["fold_id"] == "F1")
            ].index[0]
        )
        with self.assertRaisesRegex(LabelReliabilityAnalysisError, "paired panel"):
            build_arm_comparisons(bad, iterations=20, seed=1)

    def test_holm_step_down(self) -> None:
        adjusted = holm_adjust({"B": 0.01, "C": 0.03, "D": 0.04})
        self.assertAlmostEqual(adjusted["B"], 0.03)
        self.assertAlmostEqual(adjusted["C"], 0.06)
        self.assertAlmostEqual(adjusted["D"], 0.06)

    def test_q3_time_guard_rejects_q4(self) -> None:
        good = pd.DataFrame(
            {"effective_origin_utc": ["2023-07-01T00:00:00Z", "2023-09-30T23:59:59Z"]}
        )
        self.assertEqual(len(validate_q3_prediction_frame(good)), 2)
        bad = pd.DataFrame({"effective_origin_utc": ["2023-10-01T00:00:00Z"]})
        with self.assertRaisesRegex(LabelReliabilityAnalysisError, "Q3-only"):
            validate_q3_prediction_frame(bad)

    def test_q3_checkpoint_allowlist_has_only_a_winner_f4(self) -> None:
        jobs = []
        for stage, family, tolerance in (
            ("stage1_regression_05m", "regression", 5),
            ("stage2_regression_30m", "regression", 30),
            ("stage3_wgan_05m", "wgan", 5),
        ):
            for arm in ("A", "C"):
                for seed in (42, 202, 404):
                    jobs.append(
                        {
                            "job_id": f"{stage}_{arm}_{seed}",
                            "stage_id": stage,
                            "model_family": family,
                            "tolerance_minutes": tolerance,
                            "arm_id": arm,
                            "fold_id": "F4",
                            "seed": seed,
                            "best_learned_checkpoint_path": f"/tmp/{stage}_{arm}_{seed}.pt",
                            "best_learned_checkpoint_sha256": "a" * 64,
                        }
                    )
        jobs.append(
            {
                **jobs[0],
                "job_id": "nonselected",
                "arm_id": "B",
            }
        )
        allowlist = q3_checkpoint_allowlist(jobs, "C")
        self.assertEqual(len(allowlist), 18)
        self.assertEqual({row["arm_id"] for row in allowlist}, {"A", "C"})


if __name__ == "__main__":
    unittest.main()
