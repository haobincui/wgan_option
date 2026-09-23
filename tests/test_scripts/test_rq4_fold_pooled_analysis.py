"""Focused tests for the RQ4 fold-pooled MAE analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.rq3 import news_first_vol_rq4_fold_pooled_analysis as analysis


SEEDS = (11, 22)
FOLDS = ("fold_a", "fold_b")


def _metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = (
        ("train", "scheduled_only", True, False, True),
        ("train", "jump_only", False, True, True),
        ("validation", "both", True, True, False),
        ("validation", "scheduled_only", True, False, True),
        ("test", "scheduled_only", True, False, True),
        ("test", "jump_only", False, True, True),
        ("test", "both", True, True, False),
        ("test", "scheduled_only", True, False, True),
    )
    for seed_index, seed in enumerate(SEEDS):
        for fold_index, fold in enumerate(FOLDS):
            for pair_index, (
                split,
                regime,
                scheduled,
                jump,
                in_5m,
            ) in enumerate(specifications):
                base = 1.0 + fold_index * 0.01 + pair_index * 0.001
                seed_factor = 1.0 + seed_index * 0.02
                common = {
                    "seed": seed,
                    "checkpoint_fold": fold,
                    "source_split": split,
                    "pair_id": f"pair_{fold}_{split}_{pair_index}",
                    "session_id": f"session_{fold}_{split}_{pair_index // 2}",
                    "persistence_mae": base * 1.20,
                    "event_regime": regime,
                    "scheduled_event": scheduled,
                    "market_jump": jump,
                    "source_5m_membership": in_5m,
                }
                rows.append(
                    {
                        **common,
                        "model": "film_cnn",
                        "target_mae": base * seed_factor * 0.80,
                    }
                )
                rows.append(
                    {
                        **common,
                        "model": "pure_cnn",
                        "target_mae": base * seed_factor,
                    }
                )
    return pd.DataFrame(rows)


def _kwargs() -> dict[str, object]:
    return {
        "expected_seeds": SEEDS,
        "expected_folds": FOLDS,
        "bootstrap_iterations": 160,
        "bootstrap_seed": 123,
        "minimum_consistent_seeds": 2,
        "minimum_consistent_folds": 2,
        "strict_design_counts": False,
    }


class RQ4FoldPooledAnalysisTests(unittest.TestCase):
    def test_combined_and_test_only_results_have_known_direction(self) -> None:
        result = analysis.analyze_frame(_metrics(), **_kwargs())

        self.assertEqual(len(result["combined_cell_summary"]), 8)
        self.assertEqual(len(result["combined_model_summary"]), 2)
        self.assertEqual(len(result["combined_cell_comparisons"]), 4)
        comparisons = result["combined_cell_comparisons"]
        self.assertTrue(np.allclose(comparisons["film_to_pure_mae_ratio"], 0.8))
        self.assertTrue(comparisons["winning_model"].eq("film_cnn").all())
        combined = result["combined_comparison_summary"].iloc[0]
        self.assertEqual(
            combined["inference_role"], "descriptive_only_contains_train_and_validation"
        )
        self.assertAlmostEqual(
            float(combined["equal_cell_mean_log_mae_ratio"]), np.log(0.8)
        )

        strata = set(result["combined_strata_summary"]["stratum"])
        self.assertTrue(
            {
                "scheduled_any",
                "market_jump_any",
                "both",
                "source_30m_only",
                "train",
                "validation",
                "test",
            }.issubset(strata)
        )
        oos = result["test_only_oos_summary"]
        self.assertEqual(set(oos["model"]), set(analysis.CANONICAL_MODELS))
        self.assertTrue(oos["equal_cell_persistence_skill_percent"].gt(0.0).all())

        bootstrap = result["test_only_bootstrap"].set_index("comparison_id")
        primary = bootstrap.loc["film_cnn_vs_pure_cnn"]
        self.assertAlmostEqual(float(primary["mean_log_mae_ratio"]), np.log(0.8))
        self.assertLess(float(primary["ci_95_upper"]), 0.0)
        self.assertAlmostEqual(
            float(primary["geometric_mae_ratio_ci_95_lower"]),
            np.exp(float(primary["ci_95_lower"])),
        )
        self.assertAlmostEqual(
            float(primary["geometric_mae_ratio_ci_95_upper"]),
            np.exp(float(primary["ci_95_upper"])),
        )
        self.assertEqual(int(primary["consistent_seed_count"]), 2)
        self.assertEqual(int(primary["consistent_fold_count"]), 2)
        self.assertTrue(bool(primary["passes_evidence_gate"]))
        persistence = bootstrap.loc[
            ["film_cnn_vs_persistence", "pure_cnn_vs_persistence"]
        ]
        self.assertTrue(persistence["holm_adjusted_p"].notna().all())
        self.assertTrue(persistence["holm_family_size"].eq(2).all())
        self.assertTrue(persistence["passes_evidence_gate"].all())

        direction = result["test_only_direction_consistency"]
        self.assertEqual(len(direction), 3 * (len(SEEDS) + len(FOLDS)))
        film_pure = direction.loc[direction["comparison_id"].eq("film_cnn_vs_pure_cnn")]
        self.assertTrue(film_pure["direction"].eq("film_cnn_better").all())

    def test_bootstrap_is_exactly_deterministic(self) -> None:
        first = analysis.analyze_frame(_metrics(), **_kwargs())
        second = analysis.analyze_frame(_metrics(), **_kwargs())
        for key in (
            "combined_cell_summary",
            "combined_model_summary",
            "combined_cell_comparisons",
            "combined_comparison_summary",
            "combined_strata_summary",
            "test_only_oos_summary",
            "test_only_bootstrap",
            "test_only_direction_consistency",
        ):
            assert_frame_equal(first[key], second[key], check_exact=True)

    def test_csv_bundle_has_hashes_row_counts_and_required_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pair_metrics.csv.gz"
            destination = root / "analysis"
            _metrics().to_csv(source, index=False, compression="gzip")
            paths = analysis.analyze(source, destination, **_kwargs())
            self.assertEqual(
                set(paths), set(analysis.ARTIFACT_FILENAMES) | {"manifest", "report"}
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "rq4_fold_pooled_analysis_manifest_v1")
            self.assertTrue(manifest["paired_lineage_validated"])
            self.assertTrue(manifest["combined_results_are_descriptive_only"])
            self.assertEqual(manifest["models_vs_persistence_holm_family_size"], 2)
            for key, payload in manifest["artifacts"].items():
                path = Path(payload["path"])
                self.assertTrue(path.is_file(), key)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(payload["sha256"], digest)
                self.assertEqual(payload["rows"], len(pd.read_csv(path)))
            report = Path(manifest["report"]["path"])
            self.assertTrue(report.is_file())
            self.assertEqual(
                manifest["report"]["sha256"],
                hashlib.sha256(report.read_bytes()).hexdigest(),
            )
            self.assertIn("descriptive only", report.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["metadata"]["inference_scope"], "source_split_test_only"
            )

    def test_duplicate_missing_pair_and_lineage_drift_fail_closed(self) -> None:
        duplicate = pd.concat([_metrics(), _metrics().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(analysis.RQ4FoldPooledAnalysisError, "Duplicate"):
            analysis.analyze_frame(duplicate, **_kwargs())

        missing = _metrics().drop(index=0).reset_index(drop=True)
        with self.assertRaisesRegex(
            analysis.RQ4FoldPooledAnalysisError, "must occur once"
        ):
            analysis.analyze_frame(missing, **_kwargs())

        drift = _metrics()
        drift.loc[0, "session_id"] = "wrong_session"
        with self.assertRaisesRegex(
            analysis.RQ4FoldPooledAnalysisError, "session_id lineage drift"
        ):
            analysis.analyze_frame(drift, **_kwargs())

        persistence_drift = _metrics()
        persistence_drift.loc[0, "persistence_mae"] += 1.0e-6
        with self.assertRaisesRegex(
            analysis.RQ4FoldPooledAnalysisError, "persistence_mae lineage drift"
        ):
            analysis.analyze_frame(persistence_drift, **_kwargs())

    def test_event_label_schema_and_formal_counts_fail_closed(self) -> None:
        bad_label = _metrics()
        bad_label.loc[0, "event_regime"] = "both"
        with self.assertRaisesRegex(
            analysis.RQ4FoldPooledAnalysisError, "event_regime disagrees"
        ):
            analysis.analyze_frame(bad_label, **_kwargs())

        bad_split = _metrics()
        bad_split.loc[
            bad_split["source_split"].eq("validation"), "source_split"
        ] = "holdout"
        with self.assertRaisesRegex(
            analysis.RQ4FoldPooledAnalysisError, "source_split universe drift"
        ):
            analysis.analyze_frame(bad_split, **_kwargs())

        strict_kwargs = dict(_kwargs())
        strict_kwargs["strict_design_counts"] = True
        with self.assertRaisesRegex(
            analysis.RQ4FoldPooledAnalysisError, "canonical 10 seeds"
        ):
            analysis.analyze_frame(_metrics(), **strict_kwargs)

    def test_default_bootstrap_and_holm_contracts(self) -> None:
        self.assertEqual(analysis.DEFAULT_BOOTSTRAP_ITERATIONS, 10_000)
        adjusted = analysis.holm_adjust({"film": 0.01, "pure": 0.03})
        self.assertAlmostEqual(adjusted["film"], 0.02)
        self.assertAlmostEqual(adjusted["pure"], 0.03)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
