from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.rq3.news_first_vol_local_lr_analysis import (
    EXPECTED_SEEDS,
    LocalLearningRateAnalysisError,
    build_bootstrap_tables,
    build_seed_run_metrics,
    run_local_lr_analysis,
    summarize_across_seeds,
    summarize_lr_scores,
    validate_pair_metric_matrix,
    _payload_sha256,
)
from scripts.rq3.news_first_vol_local_lr_report import (
    LocalLearningRateReportError,
    render_local_lr_report,
)
from scripts.rq3.news_first_vol_local_lr_sweep import (
    EXPERIMENT_KIND,
    FROZEN_LEARNING_RATES,
    FROZEN_SEEDS,
    FROZEN_TEXT_MODES,
    FROZEN_TOLERANCES,
    LR_PROFILE_IDS,
    _lr_profile_sha256,
    _lr_seed_profile_sha256,
)


def _synthetic_pair_metrics() -> pd.DataFrame:
    profile_ratio_effect = {
        "lr_5e_07": 0.0040,
        "lr_7_5e_07": 0.0020,
        "lr_1e_06": -0.0070,
        "lr_1_5e_06": -0.0040,
        "lr_2e_06": -0.0010,
    }
    seed_ratio_effect = {42: -0.0005, 202: 0.0, 404: 0.0004}
    rows: list[dict[str, object]] = []
    for profile in LR_PROFILE_IDS:
        for seed in FROZEN_SEEDS:
            for mode in FROZEN_TEXT_MODES:
                for tolerance in FROZEN_TOLERANCES:
                    for pair_index in range(6):
                        persistence = 0.010 + pair_index * 0.0004
                        ratio = (
                            1.0
                            + profile_ratio_effect[profile]
                            + seed_ratio_effect[seed]
                            + (0.0002 if tolerance == 30 else 0.0)
                            + (-0.0003 if mode == "real_text" else 0.0)
                        )
                        rows.append(
                            {
                                "lr_profile": profile,
                                "lr_profile_sha256": _lr_profile_sha256(profile),
                                "lr_seed_profile_sha256": _lr_seed_profile_sha256(
                                    profile, seed
                                ),
                                "initial_learning_rate": FROZEN_LEARNING_RATES[profile],
                                "seed": seed,
                                "text_ablation_mode": mode,
                                "tolerance_minutes": tolerance,
                                "pair_id": f"pair_{pair_index:02d}",
                                "session_id": f"session_{pair_index // 2:02d}",
                                "model_mae": persistence * ratio,
                                "persistence_mae": persistence,
                            }
                        )
    return pd.DataFrame(rows)


class LocalLearningRateStatisticsTests(unittest.TestCase):
    def test_seed_is_preserved_in_all_summary_keys(self) -> None:
        pairs = _synthetic_pair_metrics()
        seed_runs = build_seed_run_metrics(pairs)
        self.assertEqual(len(seed_runs), 60)
        self.assertEqual(set(seed_runs["seed"].astype(int)), set(EXPECTED_SEEDS))
        self.assertFalse(
            seed_runs.duplicated(
                [
                    "lr_profile",
                    "seed",
                    "text_ablation_mode",
                    "tolerance_minutes",
                ]
            ).any()
        )

        across = summarize_across_seeds(seed_runs)
        self.assertEqual(len(across), 20)
        self.assertTrue(across["seed_count"].eq(3).all())
        self.assertTrue((across["mae_ratio_sd"] > 0.0).all())
        self.assertEqual(set(across["seeds"]), {"42|202|404"})

        scores = summarize_lr_scores(seed_runs)
        self.assertEqual(len(scores), 10)
        leader = (
            scores[scores["text_ablation_mode"].eq("current_only")]
            .sort_values("mean_log_mae_ratio")
            .iloc[0]
        )
        self.assertEqual(leader["lr_profile"], "lr_1e_06")
        self.assertTrue(bool(leader["gate_passed"]))

    def test_matrix_and_lr_seed_hash_tampering_fail_closed(self) -> None:
        pairs = _synthetic_pair_metrics()
        missing = pairs[
            ~(
                pairs["seed"].eq(404)
                & pairs["lr_profile"].eq("lr_2e_06")
                & pairs["text_ablation_mode"].eq("real_text")
                & pairs["tolerance_minutes"].eq(30)
            )
        ]
        with self.assertRaisesRegex(LocalLearningRateAnalysisError, "60-job"):
            validate_pair_metric_matrix(missing)

        tampered = pairs.copy()
        tampered.loc[tampered.index[0], "lr_seed_profile_sha256"] = "tampered"
        with self.assertRaisesRegex(LocalLearningRateAnalysisError, r"LR\+seed hash"):
            validate_pair_metric_matrix(tampered)

    def test_two_level_bootstrap_is_deterministic_and_has_fixed_center(self) -> None:
        pairs = _synthetic_pair_metrics()
        first = build_bootstrap_tables(
            pairs,
            ranked_leader_profile="lr_1e_06",
            iterations=200,
            random_seed=123,
        )
        second = build_bootstrap_tables(
            pairs,
            ranked_leader_profile="lr_1e_06",
            iterations=200,
            random_seed=123,
        )
        for left, right in zip(first, second):
            pd.testing.assert_frame_equal(left, right)
            self.assertTrue(left["seed_count"].eq(3).all())
            self.assertTrue(
                left["resampling_method"]
                .eq("seed_then_paired_CME_session_cluster")
                .all()
            )

        persistence, text, pairwise = first
        self.assertEqual(len(persistence), 30)
        self.assertEqual(len(text), 15)
        self.assertEqual(len(pairwise), 24)
        self.assertEqual(
            set(pairwise["reference_kind"]),
            {"ranked_leader", "fixed_prior_center"},
        )
        fixed = pairwise[pairwise["reference_kind"].eq("fixed_prior_center")]
        self.assertEqual(len(fixed), 12)
        self.assertEqual(set(fixed["reference_lr_profile"]), {"lr_1e_06"})


class LocalLearningRateArtifactTests(unittest.TestCase):
    def test_analysis_and_report_are_q3_only_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation_path = run_local_lr_analysis(
                root,
                q3_pair_metrics=_synthetic_pair_metrics(),
                bootstrap_iterations=100,
                bootstrap_seed=321,
            )
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            selection_path = root / "local_lr_selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))

            self.assertEqual(selection["experiment_kind"], EXPERIMENT_KIND)
            self.assertEqual(
                selection["resolved_config_sha256"], "injected_pair_metrics"
            )
            self.assertEqual(
                validation["resolved_config_sha256"], "injected_pair_metrics"
            )
            self.assertFalse(validation["q4_used_for_selection"])
            self.assertFalse(validation["q4_predictions_generated"])
            self.assertEqual(validation["q4_rows_passed_to_evaluator"], 0)

            report_path = render_local_lr_report(root)
            html = report_path.read_text(encoding="utf-8")
            self.assertIn("两级配对Bootstrap", html)
            self.assertIn("相对Q3排名首位候选", html)
            self.assertIn("Q4未预测、未评估、未参与选择", html)
            self.assertIn("seed 42/202/404", html)
            self.assertIn("6个Q3 market pairs", html)

            selection["ranked_leader_learning_rate"] = 9.9e-7
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            with self.assertRaisesRegex(
                LocalLearningRateReportError, "selection payload hash"
            ):
                render_local_lr_report(root)

            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["ranked_leader_learning_rate"] = 1.0e-6
            selection["resolved_config_sha256"] = "a" * 64
            unhashed = {
                key: value
                for key, value in selection.items()
                if key != "selection_payload_sha256"
            }
            selection["selection_payload_sha256"] = _payload_sha256(unhashed)
            validation["resolved_config_sha256"] = "a" * 64
            validation["selection_payload_sha256"] = selection[
                "selection_payload_sha256"
            ]
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            validation_path.write_text(json.dumps(validation), encoding="utf-8")
            with self.assertRaisesRegex(
                LocalLearningRateReportError, "lineage files are missing"
            ):
                render_local_lr_report(root)


if __name__ == "__main__":
    unittest.main()
