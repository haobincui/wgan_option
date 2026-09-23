from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.rq3.news_first_vol_capacity_analysis import PROFILE_PARAMETER_COUNTS
from scripts.rq3.news_first_vol_fixed_lr_capacity_analysis import (
    EXPECTED_PROFILES,
    EXPECTED_SEEDS,
    FIXED_LEARNING_RATE,
    FixedLearningRateCapacityAnalysisError,
    build_bootstrap_tables,
    build_one_se_table,
    build_seed_run_metrics,
    build_selection_summary,
    run_fixed_lr_capacity_analysis,
    summarize_across_seeds,
    summarize_capacity_scores,
    validate_pair_metric_matrix,
)
from scripts.rq3.news_first_vol_fixed_lr_capacity_report import (
    FixedLearningRateCapacityReportError,
    render_fixed_lr_capacity_report,
)
from scripts.rq3.news_first_vol_fixed_lr_capacity_sweep import (
    _capacity_profile_sha256,
    _capacity_seed_profile_sha256,
)


REAL_RATIOS = {
    "micro": 0.9970,
    "tiny": 0.99385,
    "small": 0.99380,
    "medium": 0.99440,
    "large": 0.99580,
    "legacy": 1.00050,
}
CURRENT_RATIOS = {
    "micro": 0.9900,
    "tiny": 0.9950,
    "small": 0.9948,
    "medium": 0.9960,
    "large": 0.9970,
    "legacy": 1.0002,
}


def _synthetic_pair_metrics(*, all_fail: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile in EXPECTED_PROFILES:
        parameters = PROFILE_PARAMETER_COUNTS[profile]["regression"]
        for seed_index, seed in enumerate(EXPECTED_SEEDS):
            seed_offset = (seed_index - 1) * 0.00004
            for mode in ("current_only", "real_text"):
                base = (
                    1.0008
                    if all_fail
                    else (
                        REAL_RATIOS[profile]
                        if mode == "real_text"
                        else CURRENT_RATIOS[profile]
                    )
                )
                for tolerance in (5, 30):
                    tolerance_offset = -0.00003 if tolerance == 5 else 0.00003
                    for pair_index in range(123):
                        session_index = pair_index % 33
                        persistence = 0.00125 + pair_index * 1.0e-7
                        session_effect = 0.0015 * np.sin(
                            2.0 * np.pi * session_index / 33.0
                        )
                        ratio = base + seed_offset + tolerance_offset + session_effect
                        rows.append(
                            {
                                "capacity_profile": profile,
                                "capacity_profile_sha256": _capacity_profile_sha256(
                                    profile
                                ),
                                "capacity_seed_profile_sha256": _capacity_seed_profile_sha256(
                                    profile, seed
                                ),
                                "parameter_count": parameters,
                                "initial_learning_rate": FIXED_LEARNING_RATE,
                                "seed": seed,
                                "text_ablation_mode": mode,
                                "tolerance_minutes": tolerance,
                                "pair_id": f"pair_{pair_index:03d}",
                                "session_id": f"session_{session_index:02d}",
                                "model_mae": persistence * ratio,
                                "persistence_mae": persistence,
                            }
                        )
    return pd.DataFrame(rows)


class FixedLearningRateCapacityStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pairs = _synthetic_pair_metrics()

    def test_exact_matrix_seed_summary_and_real_text_lane(self) -> None:
        validated = validate_pair_metric_matrix(self.pairs)
        self.assertEqual(len(validated), 72 * 123)
        seed_runs = build_seed_run_metrics(validated)
        across = summarize_across_seeds(seed_runs)
        scores = summarize_capacity_scores(seed_runs)
        self.assertEqual(len(seed_runs), 72)
        self.assertEqual(len(across), 24)
        self.assertEqual(len(scores), 12)
        self.assertTrue((across["seed_count"] == 3).all())
        current_leader = scores[scores["text_ablation_mode"].eq("current_only")].iloc[0]
        real_leader = scores[scores["text_ablation_mode"].eq("real_text")].iloc[0]
        self.assertEqual(current_leader["capacity_profile"], "micro")
        self.assertEqual(real_leader["capacity_profile"], "small")

    def test_two_level_bootstrap_and_one_se_smallest_capacity(self) -> None:
        seed_runs = build_seed_run_metrics(self.pairs)
        scores = summarize_capacity_scores(seed_runs)
        one_se = build_one_se_table(self.pairs, scores, iterations=160, random_seed=101)
        selection = build_selection_summary(scores, one_se)
        self.assertEqual(selection["selection_mode"], "real_text")
        self.assertEqual(selection["ranked_leader_capacity_profile"], "small")
        self.assertTrue(selection["gate_passed"])
        self.assertEqual(selection["winner_capacity_profile"], "tiny")
        persistence, text, capacity = build_bootstrap_tables(
            self.pairs,
            ranked_leader_profile="small",
            iterations=80,
            random_seed=102,
        )
        self.assertEqual((len(persistence), len(text), len(capacity)), (36, 18, 15))
        for frame in (persistence, text, capacity):
            self.assertTrue(
                frame["resampling_method"]
                .eq("seed_then_paired_CME_session_cluster")
                .all()
            )
            self.assertTrue(np.isfinite(frame["ci_95_lower"]).all())

    def test_no_gate_does_not_force_a_winner(self) -> None:
        pairs = _synthetic_pair_metrics(all_fail=True)
        scores = summarize_capacity_scores(build_seed_run_metrics(pairs))
        one_se = build_one_se_table(pairs, scores, iterations=40, random_seed=103)
        selection = build_selection_summary(scores, one_se)
        self.assertFalse(selection["gate_passed"])
        self.assertEqual(selection["winner_capacity_profile"], "")
        self.assertEqual(selection["one_standard_error_rule"]["winner_profile"], "")

    def test_matrix_fails_closed_on_missing_pair(self) -> None:
        broken = self.pairs.drop(index=self.pairs.index[0])
        with self.assertRaises(FixedLearningRateCapacityAnalysisError):
            validate_pair_metric_matrix(broken)


class FixedLearningRateCapacityArtifactTests(unittest.TestCase):
    def test_injected_analysis_and_report_render_without_q4_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            validation_path = run_fixed_lr_capacity_analysis(
                root,
                q3_pair_metrics=_synthetic_pair_metrics(),
                bootstrap_iterations=60,
                bootstrap_seed=104,
            )
            self.assertTrue(validation_path.is_file())
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            self.assertEqual(validation["job_count"], 72)
            self.assertEqual(validation["pair_count_per_run"], 123)
            self.assertEqual(validation["session_count_per_run"], 33)
            self.assertFalse(validation["q4_used_for_selection"])
            self.assertFalse(validation["q4_predictions_generated"])
            report = render_fixed_lr_capacity_report(root)
            html = report.read_text(encoding="utf-8")
            self.assertIn("固定LR容量×三Seed", html)
            self.assertIn("Real-text", html)
            self.assertIn("Q4未预测、未评估、未参与选择", html)
            self.assertIn("<svg", html)
            self.assertNotIn("https://", html)

    def test_report_rejects_selection_hash_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run_fixed_lr_capacity_analysis(
                root,
                q3_pair_metrics=_synthetic_pair_metrics(),
                bootstrap_iterations=20,
                bootstrap_seed=105,
            )
            selection_path = root / "fixed_lr_capacity_selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["selection_mode"] = "current_only"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            with self.assertRaises(FixedLearningRateCapacityReportError):
                render_fixed_lr_capacity_report(root)


if __name__ == "__main__":
    unittest.main()
