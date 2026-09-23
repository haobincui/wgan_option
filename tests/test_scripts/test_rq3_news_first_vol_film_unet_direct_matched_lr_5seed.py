"""Contracts for the two-LR, five-seed direct matched-LP experiment."""

from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import tempfile
import unittest

from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_lr_5seed as experiment,
)


EXPECTED_SEEDS = (202, 404, 382624741, 1607127774, 1662128673)
EXPECTED_ARMS = ("film_lr_1e5", "film_lr_2p5e5")
EXPECTED_RATES = {"film_lr_1e5": 1.0e-5, "film_lr_2p5e5": 2.5e-5}


class DirectMatchedFilmLrFiveSeedOrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = experiment.load_config(experiment.DEFAULT_CONFIG)
        cls.specs = experiment.planned_specs(cls.config)

    def test_frozen_matrix_contains_exactly_five_fresh_seeds_and_two_lrs(
        self,
    ) -> None:
        self.assertEqual(tuple(experiment.SEEDS), EXPECTED_SEEDS)
        self.assertEqual(tuple(experiment.DIRECT_ARMS), EXPECTED_ARMS)
        self.assertEqual(dict(experiment.FILM_LEARNING_RATES), EXPECTED_RATES)
        self.assertNotIn(42, experiment.SEEDS)
        self.assertEqual(self.config["matrix"]["seeds"], list(EXPECTED_SEEDS))
        self.assertEqual(self.config["matrix"]["direct_arms"], list(EXPECTED_ARMS))
        self.assertEqual(
            self.config["matrix"]["arm_film_learning_rates"], EXPECTED_RATES
        )
        self.assertEqual(self.config["matrix"]["expected_training_jobs"], 40)
        self.assertEqual(self.config["matrix"]["expected_prediction_cells"], 40)
        self.assertEqual(self.config["matrix"]["expected_pair_metric_rows"], 5_000)
        self.assertFalse(self.config["analysis"]["test_based_lr_selection_permitted"])
        experiment.validate_config(copy.deepcopy(self.config))

    def test_forty_jobs_are_unique_complete_and_gpu_balanced(self) -> None:
        self.assertEqual(len(self.specs), 40)
        self.assertEqual(len({str(row["job_id"]) for row in self.specs}), 40)
        expected_cells = {
            (seed, fold, arm)
            for seed in EXPECTED_SEEDS
            for fold in experiment.direct.FOLDS
            for arm in EXPECTED_ARMS
        }
        observed_cells = {
            (int(row["seed"]), str(row["fold"]), str(row["arm"])) for row in self.specs
        }
        self.assertEqual(observed_cells, expected_cells)
        self.assertEqual(
            Counter(int(row["seed"]) for row in self.specs),
            {seed: 8 for seed in EXPECTED_SEEDS},
        )
        self.assertEqual(
            Counter(str(row["fold"]) for row in self.specs),
            {fold: 10 for fold in experiment.direct.FOLDS},
        )
        self.assertEqual(
            Counter(str(row["arm"]) for row in self.specs),
            {arm: 20 for arm in EXPECTED_ARMS},
        )
        self.assertEqual(
            Counter(int(row["gpu_id"]) for row in self.specs), {0: 20, 1: 20}
        )
        for seed in EXPECTED_SEEDS:
            for fold in experiment.direct.FOLDS:
                block = [
                    row
                    for row in self.specs
                    if int(row["seed"]) == seed and str(row["fold"]) == fold
                ]
                self.assertEqual(len(block), 2)
                self.assertEqual(len({int(row["gpu_id"]) for row in block}), 1)
        experiment.validate_gpu_balance(self.specs)

    def test_initial_states_are_common_within_seed_and_distinct_across_seeds(
        self,
    ) -> None:
        generator_by_seed: dict[int, str] = {}
        critic_by_seed: dict[int, str] = {}
        for seed in EXPECTED_SEEDS:
            block = [row for row in self.specs if int(row["seed"]) == seed]
            generator_hashes = {
                str(row["initial_generator_state_sha256"]) for row in block
            }
            critic_hashes = {str(row["initial_critic_state_sha256"]) for row in block}
            self.assertEqual(len(generator_hashes), 1)
            self.assertEqual(len(critic_hashes), 1)
            generator_by_seed[seed] = generator_hashes.pop()
            critic_by_seed[seed] = critic_hashes.pop()
        self.assertEqual(len(set(generator_by_seed.values())), len(EXPECTED_SEEDS))
        self.assertEqual(len(set(critic_by_seed.values())), len(EXPECTED_SEEDS))

    def test_seed_scoped_run_and_prediction_paths_do_not_collide(self) -> None:
        root = Path("/tmp/five-seed-path-contract")
        representatives = [
            next(
                row
                for row in self.specs
                if int(row["seed"]) == seed
                and str(row["fold"]) == experiment.direct.FOLDS[0]
                and str(row["arm"]) == EXPECTED_ARMS[0]
            )
            for seed in EXPECTED_SEEDS
        ]
        run_paths = {experiment._run_directory(root, row) for row in representatives}
        prediction_paths = {
            experiment._prediction_path(root, row) for row in representatives
        }
        self.assertEqual(len(run_paths), len(EXPECTED_SEEDS))
        self.assertEqual(len(prediction_paths), len(EXPECTED_SEEDS))
        for seed, row in zip(EXPECTED_SEEDS, representatives, strict=True):
            self.assertIn(f"seed_{seed}", str(experiment._run_directory(root, row)))
            self.assertIn(f"seed_{seed}", str(experiment._prediction_path(root, row)))

    def test_training_payload_preserves_seed_and_changes_only_film_lr(self) -> None:
        for spec in self.specs:
            payload = experiment._training_payload(self.config, Path("/unused"), spec)
            arm = str(spec["arm"])
            self.assertEqual(payload["seed"], int(spec["seed"]))
            self.assertEqual(
                payload["generator_optimizer_profile"], "film_unet_split_lr_v1"
            )
            self.assertEqual(payload["generator_learning_rate"], 5.0e-7)
            self.assertEqual(payload["generator_text_learning_rate"], 2.5e-6)
            self.assertEqual(payload["generator_text_min_learning_rate"], 2.5e-7)
            self.assertEqual(payload["discriminator_learning_rate"], 5.0e-7)
            self.assertEqual(
                payload["generator_film_learning_rate"], EXPECTED_RATES[arm]
            )
            self.assertEqual(
                payload["generator_film_min_learning_rate"],
                EXPECTED_RATES[arm] * 0.1,
            )
            self.assertEqual(payload["news_first_pair_text_overlay_mode"], "lp_mean_l2")
            self.assertTrue(payload["news_first_materialize_validation_loader"])
            self.assertFalse(payload["news_first_materialize_test_loader"])
            self.assertEqual(payload["news_first_refit_mode"], "none")

    def test_selection_provenance_and_no_test_selection_drift_fail_closed(
        self,
    ) -> None:
        cases: list[dict[str, object]] = []

        report_hash = copy.deepcopy(self.config)
        report_hash["analysis"]["selection_provenance"]["report_sha256"] = "0" * 64
        cases.append(report_hash)

        manifest_hash = copy.deepcopy(self.config)
        manifest_hash["analysis"]["selection_provenance"][
            "output_hash_manifest_sha256"
        ] = "0" * 64
        cases.append(manifest_hash)

        included_discovery = copy.deepcopy(self.config)
        included_discovery["analysis"]["selection_provenance"][
            "excluded_from_validation"
        ] = False
        cases.append(included_discovery)

        changed_seed = copy.deepcopy(self.config)
        changed_seed["analysis"]["selection_provenance"]["discovery_seed"] = 202
        cases.append(changed_seed)

        test_selection = copy.deepcopy(self.config)
        test_selection["analysis"]["test_based_lr_selection_permitted"] = True
        cases.append(test_selection)

        for config in cases:
            with self.subTest(provenance=config["analysis"]):
                with self.assertRaises(ValueError):
                    experiment.validate_config(config)

    def test_selection_provenance_is_bound_to_existing_immutable_files(self) -> None:
        provenance = self.config["analysis"]["selection_provenance"]
        report = experiment.direct.resolve_path(provenance["report_path"])
        manifest = experiment.direct.resolve_path(
            provenance["output_hash_manifest_path"]
        )
        self.assertTrue(report.is_file())
        self.assertTrue(manifest.is_file())
        self.assertEqual(
            experiment.direct.sha256_file(report), provenance["report_sha256"]
        )
        self.assertEqual(
            experiment.direct.sha256_file(manifest),
            provenance["output_hash_manifest_sha256"],
        )

    def test_completed_source_config_load_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unused"
            first = experiment.planned_specs(self.config)
            second = experiment.planned_specs(self.config)
            self.assertEqual(first, second)
            self.assertFalse(path.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
