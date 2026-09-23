"""Focused contracts for the two-child 10-seed text composite."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd

from scripts.rq3 import news_first_vol_text_10seed_composite as composite


TEST_SEEDS = (11, 22)
TEST_FOLDS = ("fold_a", "fold_b")
TEST_COUNTS = {fold: 2 for fold in TEST_FOLDS}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pairs() -> pd.DataFrame:
    ratios = {
        "lp_matched": 0.90,
        "no_text": 1.00,
        "lp_shuffle": 1.02,
        "bow": 1.03,
        "sentiment": 1.04,
    }
    rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(TEST_SEEDS):
        for fold_index, fold in enumerate(TEST_FOLDS):
            noise = _sha(f"noise:{seed}:{fold}")
            for pair_index in range(2):
                base = 1.0 + seed_index * 0.02 + fold_index * 0.01 + pair_index * 0.001
                persistence = (1.0 + fold_index * 0.01 + pair_index * 0.001) * 1.10
                for arm, ratio in ratios.items():
                    job_id = f"job_{arm}_{seed}_{fold}"
                    rows.append(
                        {
                            "job_id": job_id,
                            "tolerance_minutes": 5,
                            "fold": fold,
                            "seed": seed,
                            "arm": arm,
                            "pair_id": f"pair_{fold}_{pair_index}",
                            "session_id": f"session_{fold}_{pair_index}",
                            "effective_origin_utc": (
                                f"2023-01-0{pair_index + 1}T12:00:00Z"
                            ),
                            "target_mae": base * ratio,
                            "persistence_mae": persistence,
                            "checkpoint_sha256": _sha(f"checkpoint:{job_id}"),
                            "prediction_sha256": _sha(f"prediction:{job_id}"),
                            "noise_bank_profile_sha256": noise,
                            "source_child": (
                                "pure_cnn_no_text" if arm == "no_text" else "film_text"
                            ),
                            "source_arm": (
                                "pure_cnn_no_text" if arm == "no_text" else arm
                            ),
                        }
                    )
    return pd.DataFrame(rows)


class TextTenSeedCompositeTests(unittest.TestCase):
    def test_config_binds_two_disjoint_direct_children(self) -> None:
        config = composite.load_config()
        self.assertEqual(tuple(config["children"]), ("film_text", "pure_cnn_no_text"))
        pure = config["children"]["pure_cnn_no_text"]
        self.assertEqual(pure["source_arms"], ["pure_cnn_no_text"])
        self.assertEqual(pure["arm_aliases"], {"pure_cnn_no_text": "no_text"})
        self.assertEqual(config["matrix"]["expected_training_jobs"], 200)
        self.assertEqual(config["matrix"]["expected_pair_metric_rows"], 25_000)
        self.assertFalse(config["analysis"]["run_bootstrap"])

    def test_missing_run_bootstrap_preserves_previous_enabled_behavior(self) -> None:
        config = composite.load_config()
        del config["analysis"]["run_bootstrap"]
        composite.validate_config(config)

    def test_equal_seed_summary_holm4_and_oracle_label(self) -> None:
        validated = composite.validate_combined_pairs(
            _pairs(),
            seeds=TEST_SEEDS,
            folds=TEST_FOLDS,
            arms=composite.ARMS,
            expected_pairs_by_fold=TEST_COUNTS,
            expected_sessions_by_fold=TEST_COUNTS,
            expected_rows=40,
        )
        result = composite.analyze_pairs(
            validated,
            seeds=TEST_SEEDS,
            folds=TEST_FOLDS,
            bootstrap_iterations=40,
            bootstrap_seed=123,
            minimum_nonworse_seeds=2,
            minimum_nonworse_folds=2,
        )
        self.assertEqual(len(result["seed_summary"]), 10)
        self.assertEqual(len(result["arm_summary"]), 5)
        self.assertEqual(len(result["contrasts"]), 4)
        leader = result["arm_summary"].iloc[0]
        self.assertEqual(leader["arm"], "lp_matched")
        self.assertEqual(
            leader["best_observed_seed_role"],
            "descriptive_oracle_only_not_primary_not_selection",
        )
        self.assertTrue((result["contrasts"]["holm_family_size"] == 4).all())
        self.assertTrue((result["contrasts"]["mean_log_mae_ratio"] < 0.0).all())
        markdown, html_report = composite._reports(
            result["arm_summary"],
            result["seed_summary"],
            result["contrasts"],
            bootstrap_iterations=40,
        )
        self.assertIn("descriptive oracle only", markdown)
        self.assertIn("parent jobs=0", html_report)
        self.assertNotIn("https://", html_report)

    def test_descriptive_only_analysis_skips_bootstrap_and_holm(self) -> None:
        validated = composite.validate_combined_pairs(
            _pairs(),
            seeds=TEST_SEEDS,
            folds=TEST_FOLDS,
            arms=composite.ARMS,
            expected_pairs_by_fold=TEST_COUNTS,
            expected_sessions_by_fold=TEST_COUNTS,
            expected_rows=40,
        )
        with mock.patch.object(
            composite,
            "_contrasts",
            side_effect=AssertionError("bootstrap must not run"),
        ):
            result = composite.analyze_pairs(
                validated,
                seeds=TEST_SEEDS,
                folds=TEST_FOLDS,
                run_bootstrap=False,
            )
        self.assertEqual(set(result), {"fold_summary", "seed_summary", "arm_summary"})
        self.assertEqual(len(result["seed_summary"]), 10)
        self.assertEqual(len(result["arm_summary"]), 5)
        markdown, html_report = composite._reports(
            result["arm_summary"],
            result["seed_summary"],
            None,
            bootstrap_iterations=10_000,
            run_bootstrap=False,
        )
        self.assertIn("Descriptive analysis only", markdown)
        self.assertIn("intentionally deferred", html_report)
        self.assertNotIn("Paired bootstrap", markdown)
        self.assertNotIn("Holm-4", html_report)

    def test_cross_child_noise_or_lineage_drift_fails_closed(self) -> None:
        frame = _pairs()
        mask = (
            frame["seed"].eq(TEST_SEEDS[0])
            & frame["fold"].eq(TEST_FOLDS[0])
            & frame["arm"].eq("no_text")
        )
        frame.loc[mask, "noise_bank_profile_sha256"] = _sha("wrong-noise")
        with self.assertRaisesRegex(composite.CompositeError, "lineage"):
            composite.validate_combined_pairs(
                frame,
                seeds=TEST_SEEDS,
                folds=TEST_FOLDS,
                arms=composite.ARMS,
                expected_pairs_by_fold=TEST_COUNTS,
                expected_sessions_by_fold=TEST_COUNTS,
                expected_rows=40,
            )

    def test_parent_or_continuation_lineage_is_rejected(self) -> None:
        direct_job = {
            "job_id": "direct",
            "stage": "direct_arms",
            "parent_state_path": "",
            "recipe_path": "",
            "parent_job_id": "",
            "continuation_job_id": "",
        }
        composite._validate_direct_job_contract(
            "child",
            [direct_job],
            {"parent_jobs": 0, "continuation_jobs": 0},
            expected_jobs=1,
        )
        parent_job = dict(direct_job, parent_state_path="parent.pt")
        with self.assertRaisesRegex(composite.CompositeError, "random-initialization"):
            composite._validate_direct_job_contract(
                "child",
                [parent_job],
                {"parent_jobs": 0, "continuation_jobs": 0},
                expected_jobs=1,
            )
        with self.assertRaisesRegex(composite.CompositeError, "zero parent"):
            composite._validate_direct_job_contract(
                "child",
                [direct_job],
                {"parent_jobs": 1, "continuation_jobs": 0},
                expected_jobs=1,
            )

    def test_film_child_runner_stays_inside_both_runtime_profiles(self) -> None:
        state = {"film": False, "multiseed": False, "runner_calls": 0}

        @contextmanager
        def film_profile():
            self.assertFalse(state["film"])
            state["film"] = True
            try:
                yield
            finally:
                state["film"] = False

        @contextmanager
        def multiseed_profile():
            self.assertTrue(state["film"])
            self.assertFalse(state["multiseed"])
            state["multiseed"] = True
            try:
                yield
            finally:
                state["multiseed"] = False

        def runner(*_args: object, **_kwargs: object) -> None:
            self.assertTrue(state["film"])
            self.assertTrue(state["multiseed"])
            state["runner_calls"] += 1

        module = SimpleNamespace(
            film_text_profile=film_profile,
            multiseed=SimpleNamespace(multiseed_profile=multiseed_profile),
            run_pipeline=runner,
        )
        child = {
            "module": "fake.film",
            "config": "fake-film.yaml",
            "output_root": "fake-film-root",
        }
        with (
            mock.patch.object(composite, "_child_status", return_value={}),
            mock.patch.object(
                composite.importlib, "import_module", return_value=module
            ),
            mock.patch.object(composite, "_child_artifacts", return_value={}),
        ):
            composite._run_child("film_text", child)
        self.assertEqual(state["runner_calls"], 1)
        self.assertFalse(state["film"])
        self.assertFalse(state["multiseed"])

    def test_pure_cnn_child_runner_does_not_require_film_profiles(self) -> None:
        runner = mock.Mock()
        module = SimpleNamespace(run_pipeline=runner)
        child = {
            "module": "fake.pure",
            "config": "fake-pure.yaml",
            "output_root": "fake-pure-root",
        }
        with (
            mock.patch.object(composite, "_child_status", return_value={}),
            mock.patch.object(
                composite.importlib, "import_module", return_value=module
            ),
            mock.patch.object(composite, "_child_artifacts", return_value={}),
        ):
            composite._run_child("pure_cnn_no_text", child)
        runner.assert_called_once()

    def test_absent_status_has_no_side_effects_in_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not-created"
            result = composite.status(root)
            self.assertEqual(result["status"], "absent")
            self.assertFalse(root.exists())
            self.assertEqual(result["combined_pair_rows"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
