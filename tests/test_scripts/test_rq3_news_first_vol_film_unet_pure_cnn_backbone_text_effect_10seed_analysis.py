"""Focused tests for the pure-CNN-backbone FiLM text-effect analysis."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_analysis as analysis,
)


SEEDS = (11, 22)
FOLDS = ("fold_a", "fold_b")


def _standard(*, supported: bool = True) -> pd.DataFrame:
    ratios = {
        "matched": 0.80 if supported else 1.02,
        "film_zero_text": 1.00,
        "film_lp_shuffle": 0.96,
    }
    rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(SEEDS):
        for fold_index, fold in enumerate(FOLDS):
            for pair_index in range(4):
                base = 1.0 + seed_index * 0.02 + fold_index * 0.01 + pair_index * 0.001
                for arm, ratio in ratios.items():
                    rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "fold": fold,
                            "pair_id": f"pair_{fold}_{pair_index}",
                            "session_id": f"session_{fold}_{pair_index // 2}",
                            "target_mae": base * ratio,
                            "persistence_mae": base * 1.10,
                        }
                    )
    return pd.DataFrame(rows)


def _intervention(*, supported: bool = True) -> pd.DataFrame:
    ratios = {
        "matched_input": 0.80 if supported else 1.01,
        "zero_input": 1.00,
        "wrong_input": 0.95,
    }
    rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(SEEDS):
        for fold_index, fold in enumerate(FOLDS):
            for pair_index in range(4):
                base = 1.0 + seed_index * 0.02 + fold_index * 0.01 + pair_index * 0.001
                for condition, ratio in ratios.items():
                    rows.append(
                        {
                            "input_condition": condition,
                            "seed": seed,
                            "fold": fold,
                            "pair_id": f"pair_{fold}_{pair_index}",
                            "session_id": f"session_{fold}_{pair_index // 2}",
                            "target_mae": base * ratio,
                        }
                    )
    return pd.DataFrame(rows)


def _trajectory() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arm, multiplier in (
        ("matched", 1.0),
        ("film_zero_text", 0.5),
        ("film_lp_shuffle", 0.4),
    ):
        for seed in SEEDS:
            for fold in FOLDS:
                for pair_index in range(2):
                    parent = 2.0 + pair_index * 0.01
                    for epoch in range(1, 31):
                        gain = multiplier * epoch / 100.0
                        rows.append(
                            {
                                "arm": arm,
                                "seed": seed,
                                "fold": fold,
                                "pair_id": f"pair_{fold}_{pair_index}",
                                "session_id": f"session_{fold}_{pair_index}",
                                "epoch": epoch,
                                "target_mae": parent - gain,
                                "parent_mae": parent,
                            }
                        )
    return pd.DataFrame(rows)


def _sparse_trajectory() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = (
        ("epoch_0", 0),
        ("epoch_1", 1),
        ("epoch_5", 5),
        ("epoch_10", 10),
        ("epoch_20", 20),
        ("epoch_30", 30),
        ("best", 20),
    )
    for arm, multiplier in (
        ("pure_cnn_continue_no_text", 0.6),
        ("film_lp_matched", 1.0),
        ("film_zero_text", 0.5),
        ("film_lp_shuffle", 0.4),
    ):
        for seed in SEEDS:
            for fold in FOLDS:
                for pair_index in range(2):
                    parent = 2.0 + pair_index * 0.01
                    for label, epoch in labels:
                        rows.append(
                            {
                                "arm": arm,
                                "seed": seed,
                                "fold": fold,
                                "pair_id": f"pair_{fold}_{pair_index}",
                                "session_id": f"session_{fold}_{pair_index}",
                                "checkpoint_label": label,
                                "epoch": epoch,
                                "target_mae": parent - multiplier * epoch / 100.0,
                            }
                        )
    return pd.DataFrame(rows)


class PureCnnBackboneTextEffectTests(unittest.TestCase):
    def test_full_analysis_is_deterministic_holm2_and_known_direction(self) -> None:
        kwargs = {
            "expected_seeds": SEEDS,
            "expected_folds": FOLDS,
            "bootstrap_iterations": 120,
            "bootstrap_seed": 123,
            "minimum_nonworse_seeds": 2,
            "minimum_nonworse_folds": 2,
        }
        first = analysis.analyze_backbone_text_effect(
            _standard(), _intervention(), _trajectory(), **kwargs
        )
        second = analysis.analyze_backbone_text_effect(
            _standard(), _intervention(), _trajectory(), **kwargs
        )
        for name in (
            "main_comparisons",
            "intervention_comparisons",
            "validation_trajectory",
            "validation_trajectory_cells",
            "validation_trajectory_arms",
            "validation_trajectory_comparisons",
            "validation_trajectory_snapshots",
        ):
            assert_frame_equal(first[name], second[name], check_exact=True)
        for family in ("main_comparisons", "intervention_comparisons"):
            result = first[family]
            self.assertEqual(len(result), 2)
            self.assertTrue(result["holm_family_size"].eq(2).all())
            self.assertTrue(result["mean_log_mae_ratio"].lt(0.0).all())
            self.assertTrue(result["ci_95_upper"].lt(0.0).all())
            self.assertTrue(result["passes_support_gate"].all())
        self.assertEqual(first["conclusion"]["status"], "stable_text_mae_increment")
        self.assertTrue(first["conclusion"]["all_three_evidence_layers_supported"])
        encoded = analysis.conclusion_json(first)
        self.assertEqual(json.loads(encoded)["status"], "stable_text_mae_increment")

    def test_csv_inputs_and_trajectory_gain_auc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            standard_path = root / "standard.csv"
            intervention_path = root / "intervention.csv"
            trajectory_path = root / "trajectory.csv"
            _standard().to_csv(standard_path, index=False)
            _intervention().to_csv(intervention_path, index=False)
            _trajectory().to_csv(trajectory_path, index=False)
            result = analysis.analyze_backbone_text_effect(
                standard_path,
                intervention_path,
                trajectory_path,
                expected_seeds=SEEDS,
                expected_folds=FOLDS,
                bootstrap_iterations=120,
                bootstrap_seed=9,
                minimum_nonworse_seeds=2,
                minimum_nonworse_folds=2,
            )
        matched = result["validation_trajectory_arms"].set_index("arm").loc["matched"]
        self.assertAlmostEqual(float(matched["epoch30_gain_equal_cell_mean"]), 0.30)
        self.assertAlmostEqual(
            float(matched["epoch1_30_gain_trapezoid_auc_equal_cell_mean"]),
            4.495,
        )
        self.assertTrue(
            np.allclose(
                result["validation_trajectory"]["gain"],
                result["validation_trajectory"]["parent_mae"]
                - result["validation_trajectory"]["target_mae"],
            )
        )

    def test_conclusion_status_exhausts_four_declared_states(self) -> None:
        expected = {
            (True, True): "stable_text_mae_increment",
            (True, False): "optimization_only",
            (False, True): "not_generalized",
            (False, False): "no_verified_text_reliance",
        }
        for (main, intervention), status in expected.items():
            with self.subTest(main=main, intervention=intervention):
                self.assertEqual(
                    analysis.conclusion_status(
                        main_all_supported=main,
                        intervention_all_supported=intervention,
                    ),
                    status,
                )

    def test_holm_is_step_down_monotone(self) -> None:
        adjusted = analysis.holm_adjust({"a": 0.01, "b": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.02)
        self.assertAlmostEqual(adjusted["b"], 0.03)

    def test_pair_or_session_lineage_drift_fails_closed(self) -> None:
        standard = _standard()
        mask = (
            standard["arm"].eq("film_zero_text")
            & standard["seed"].eq(SEEDS[0])
            & standard["fold"].eq(FOLDS[0])
            & standard["pair_id"].eq(f"pair_{FOLDS[0]}_0")
        )
        standard.loc[mask, "session_id"] = "wrong_session"
        with self.assertRaisesRegex(analysis.TextEffectAnalysisError, "session"):
            analysis.analyze_backbone_text_effect(
                standard,
                _intervention(),
                _trajectory(),
                expected_seeds=SEEDS,
                expected_folds=FOLDS,
                bootstrap_iterations=10,
                minimum_nonworse_seeds=2,
                minimum_nonworse_folds=2,
            )

    def test_trajectory_requires_every_epoch_one_through_thirty(self) -> None:
        trajectory = _trajectory()
        trajectory = trajectory.drop(
            trajectory[
                trajectory["arm"].eq("matched")
                & trajectory["seed"].eq(SEEDS[0])
                & trajectory["fold"].eq(FOLDS[0])
                & trajectory["pair_id"].eq(f"pair_{FOLDS[0]}_0")
                & trajectory["epoch"].eq(17)
            ].index
        )
        with self.assertRaisesRegex(analysis.TextEffectAnalysisError, "lacks epochs"):
            analysis.validation_gain_trajectory(trajectory)

    def test_sparse_trajectory_derives_parent_and_keeps_best_separate(self) -> None:
        rows, cells, arms = analysis.validation_gain_trajectory(_sparse_trajectory())
        self.assertEqual(set(rows["checkpoint_label"]), set(analysis.TRAJECTORY_LABELS))
        self.assertTrue(
            rows.loc[rows["checkpoint_label"].eq("epoch_0"), "gain"].eq(0).all()
        )
        matched = arms.set_index("arm").loc["matched"]
        self.assertAlmostEqual(float(matched["epoch30_gain_equal_cell_mean"]), 0.30)
        # Sparse piecewise-linear integration: gains .01/.05/.10/.20/.30.
        self.assertAlmostEqual(
            float(matched["epoch1_30_gain_trapezoid_auc_equal_cell_mean"]),
            4.495,
        )
        comparisons = analysis.validation_trajectory_comparisons(
            cells,
            minimum_nonworse_seeds=2,
            minimum_nonworse_folds=2,
        )
        self.assertTrue(comparisons["passes_optimization_gate"].all())

    def test_sparse_trajectory_rejects_cross_arm_epoch0_drift(self) -> None:
        trajectory = _sparse_trajectory()
        mask = (
            trajectory["arm"].eq("film_zero_text")
            & trajectory["seed"].eq(SEEDS[0])
            & trajectory["fold"].eq(FOLDS[0])
            & trajectory["pair_id"].eq(f"pair_{FOLDS[0]}_0")
            & trajectory["checkpoint_label"].eq("epoch_0")
        )
        trajectory.loc[mask, "target_mae"] += 1e-4
        with self.assertRaisesRegex(analysis.TextEffectAnalysisError, "not identical"):
            analysis.validation_gain_trajectory(trajectory)

    def test_sparse_trajectory_accepts_graft_tolerance_and_uses_pure_baseline(
        self,
    ) -> None:
        trajectory = _sparse_trajectory()
        mask = (
            trajectory["arm"].eq("film_lp_matched")
            & trajectory["seed"].eq(SEEDS[0])
            & trajectory["fold"].eq(FOLDS[0])
            & trajectory["pair_id"].eq(f"pair_{FOLDS[0]}_0")
            & trajectory["checkpoint_label"].eq("epoch_0")
        )
        trajectory.loc[mask, "target_mae"] += 5e-8
        rows, _cells, _arms = analysis.validation_gain_trajectory(trajectory)
        matched_epoch0 = rows.loc[
            rows["arm"].eq("matched")
            & rows["seed"].eq(SEEDS[0])
            & rows["fold"].eq(FOLDS[0])
            & rows["pair_id"].eq(f"pair_{FOLDS[0]}_0")
            & rows["checkpoint_label"].eq("epoch_0")
        ]
        # Gain is measured against the canonical pure-CNN epoch0, not this arm's
        # near-identical local output.
        self.assertAlmostEqual(float(matched_epoch0.iloc[0]["gain"]), -5e-8)

    def test_default_bootstrap_contract_is_ten_thousand(self) -> None:
        self.assertEqual(analysis.DEFAULT_BOOTSTRAP_ITERATIONS, 10_000)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
