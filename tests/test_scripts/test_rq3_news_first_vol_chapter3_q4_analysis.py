from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.rq3 import news_first_vol_chapter3_q4_analysis as analysis


class Chapter3Q4AnalysisTests(unittest.TestCase):
    def test_q4_bootstrap_is_deterministic_and_preserves_direction(self) -> None:
        rows = []
        for seed in analysis.SEEDS:
            for session in range(45):
                for pair_offset in range(4 if session < 8 else 3):
                    reference = 2.0 + 0.001 * pair_offset
                    rows.append(
                        {
                            "seed": seed,
                            "pair_id": f"p{session}_{pair_offset}",
                            "session_id": f"s{session}",
                            "focal_mae": reference * 0.99,
                            "reference_mae": reference,
                        }
                    )
        paired = pd.DataFrame(rows)
        self.assertEqual(len(paired), 143 * len(analysis.SEEDS))

        first, draws_first = analysis.bootstrap_q4_comparison(
            paired, iterations=100, rng_seed=7
        )
        second, draws_second = analysis.bootstrap_q4_comparison(
            paired, iterations=100, rng_seed=7
        )
        np.testing.assert_array_equal(draws_first, draws_second)
        self.assertLess(first["mean_log_mae_ratio"], 0.0)
        self.assertLess(first["ci_95_upper"], 0.0)
        self.assertEqual(first["consistent_seed_count"], 10)
        self.assertAlmostEqual(
            first["bootstrap_t"],
            first["mean_log_mae_ratio"] / first["bootstrap_se"],
        )

    def test_significance_stars_use_strict_conventional_thresholds(self) -> None:
        self.assertEqual(analysis.significance_stars(0.009), "***")
        self.assertEqual(analysis.significance_stars(0.01), "**")
        self.assertEqual(analysis.significance_stars(0.049), "**")
        self.assertEqual(analysis.significance_stars(0.05), "*")
        self.assertEqual(analysis.significance_stars(0.099), "*")
        self.assertEqual(analysis.significance_stars(0.10), "")
        with self.assertRaises(analysis.Chapter3Q4AnalysisError):
            analysis._bootstrap_t_statistic(-0.1, 0.0)
        with self.assertRaises(analysis.Chapter3Q4AnalysisError):
            analysis._bootstrap_t_statistic(float("nan"), 0.1)

    def test_q4_contrasts_have_frozen_holm_families(self) -> None:
        rows = []
        arm_factor = {
            "lp_matched": 0.98,
            "no_text": 1.00,
            "lp_shuffle": 0.995,
            "bow": 0.99,
            "sentiment": 1.01,
        }
        for arm in analysis.ARMS:
            for seed in analysis.SEEDS:
                for session in range(45):
                    pair_count = 4 if session < 8 else 3
                    for pair_offset in range(pair_count):
                        base = 2.0 + session * 1e-4 + pair_offset * 1e-5
                        rows.append(
                            {
                                "arm": arm,
                                "seed": seed,
                                "fold": analysis.Q4_FOLD,
                                "pair_id": f"p{session}_{pair_offset}",
                                "session_id": f"s{session}",
                                "target_mae": base * arm_factor[arm],
                                "persistence_mae": base * 1.02,
                            }
                        )
        pairs = pd.DataFrame(rows)
        self.assertEqual(len(pairs), len(analysis.ARMS) * len(analysis.SEEDS) * 143)
        contrasts, draws = analysis.q4_contrasts(pairs, iterations=50, rng_seed=11)
        self.assertEqual(len(contrasts), 5)
        self.assertEqual(len(draws), 250)
        family_sizes = contrasts.groupby("family")["holm_family_size"].first()
        self.assertEqual(family_sizes["rq1_primary"], 1)
        self.assertEqual(family_sizes["rq1_secondary_persistence"], 2)
        self.assertEqual(family_sizes["rq2_primary"], 2)
        self.assertTrue(
            (contrasts["holm_adjusted_p"] >= contrasts["p_value_one_sided"]).all()
        )
        self.assertTrue(np.isfinite(contrasts["bootstrap_t"]).all())
        self.assertTrue(
            contrasts["significance_stars"].isin(("", "*", "**", "***")).all()
        )

    def test_fold_persistence_contrasts_create_four_holm5_families(self) -> None:
        rows = []
        arm_factor = {
            "lp_matched": 0.980,
            "no_text": 0.985,
            "lp_shuffle": 0.990,
            "bow": 0.995,
            "sentiment": 1.005,
        }
        for fold in analysis.FOLDS:
            pair_count = analysis.EXPECTED_PAIRS[fold]
            session_count = analysis.EXPECTED_SESSIONS[fold]
            base_pairs, remainder = divmod(pair_count, session_count)
            for arm_index, arm in enumerate(analysis.ARMS):
                for seed in analysis.SEEDS:
                    pair_number = 0
                    for session in range(session_count):
                        count = base_pairs + int(session < remainder)
                        for pair_offset in range(count):
                            base = 2.0 + session * 1e-4 + pair_offset * 1e-5
                            variation = 1.0 + (arm_index + 1) * 1e-4 * (
                                (session % 3) - 1
                            )
                            rows.append(
                                {
                                    "arm": arm,
                                    "seed": seed,
                                    "fold": fold,
                                    "pair_id": f"{fold}_p{pair_number}",
                                    "session_id": f"{fold}_s{session}",
                                    "target_mae": base * arm_factor[arm] * variation,
                                    "persistence_mae": base,
                                }
                            )
                            pair_number += 1
        pairs = pd.DataFrame(rows)
        contrasts, draws = analysis.fold_persistence_contrasts(
            pairs, iterations=20, rng_seed=19
        )
        self.assertEqual(len(contrasts), len(analysis.FOLDS) * len(analysis.ARMS))
        self.assertEqual(len(draws), 20 * len(analysis.FOLDS) * len(analysis.ARMS))
        self.assertEqual(
            contrasts.groupby("fold")["holm_family_size"].first().to_dict(),
            {fold: len(analysis.ARMS) for fold in analysis.FOLDS},
        )
        self.assertTrue(np.isfinite(contrasts["bootstrap_t"]).all())
        for row in contrasts.itertuples(index=False):
            self.assertAlmostEqual(
                row.bootstrap_t, row.mean_log_mae_ratio / row.bootstrap_se
            )


if __name__ == "__main__":
    unittest.main()
