"""Contracts for the ten-seed Pure-CNN no-text rolling experiment."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from scripts.rq3 import news_first_vol_cnn_unet_pure_no_text_10seed as experiment


class PureCnnNoTextTenSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = experiment.load_config()
        cls.specs = experiment.planned_specs(cls.config)

    def test_frozen_matrix_model_and_learning_rates(self) -> None:
        self.assertEqual(len(experiment.SEEDS), 10)
        self.assertEqual(experiment.DIRECT_ARMS, ("pure_cnn_no_text",))
        self.assertEqual(experiment.EXPECTED_TRAINING_JOBS, 40)
        self.assertEqual(experiment.EXPECTED_PREDICTION_CELLS, 40)
        self.assertEqual(experiment.EXPECTED_PAIR_METRIC_ROWS, 5_000)
        contract = experiment.model_contract(self.config)
        self.assertEqual(
            contract["generator_conditioning_mode"], "cnn_unet_mask_coords_v1"
        )
        self.assertEqual(
            contract["critic_conditioning_mode"], "lp_disabled_same_shape_v1"
        )
        self.assertEqual(contract["generator_parameters"], 416_353)
        self.assertEqual(contract["critic_parameters"], 729_157)
        self.assertEqual(contract["total_parameters"], 1_145_510)
        training = self.config["training"]
        self.assertEqual(training["generator_optimizer_profile"], "uniform_v1")
        self.assertEqual(training["generator_learning_rate"], 5e-7)
        self.assertEqual(training["discriminator_learning_rate"], 5e-7)

    def test_forty_unique_direct_jobs_are_balanced(self) -> None:
        self.assertEqual(len(self.specs), 40)
        self.assertEqual(len({row["job_id"] for row in self.specs}), 40)
        self.assertEqual(
            {gpu: sum(row["gpu_id"] == gpu for row in self.specs) for gpu in (0, 1)},
            {0: 20, 1: 20},
        )
        for seed in experiment.SEEDS:
            selected = [row for row in self.specs if row["seed"] == seed]
            self.assertEqual(len(selected), 4)
            self.assertEqual(
                len({row["initial_generator_state_sha256"] for row in selected}), 1
            )
            self.assertEqual(
                len({row["initial_critic_state_sha256"] for row in selected}), 1
            )
        experiment.validate_gpu_balance(self.specs)

    def test_profile_installs_and_restores_shared_orchestrators(self) -> None:
        multi_names = ("GENERATOR_MODE", "SEEDS", "DIRECT_ARMS", "WORKER_MODULE")
        direct_names = ("GENERATOR_MODE", "EXPECTED_PARAMETER_COUNTS", "WORKER_MODULE")
        before_multi = {name: getattr(experiment.multi, name) for name in multi_names}
        before_direct = {
            name: getattr(experiment.direct, name) for name in direct_names
        }
        with experiment.pure_multiseed_profile():
            with experiment.multi.multiseed_profile():
                self.assertEqual(
                    experiment.direct.GENERATOR_MODE, "cnn_unet_mask_coords_v1"
                )
                self.assertEqual(
                    experiment.direct.EXPECTED_PARAMETER_COUNTS["generator"], 416_353
                )
                self.assertEqual(
                    experiment.direct.WORKER_MODULE, experiment.WORKER_MODULE
                )
                self.assertEqual(len(experiment.direct.planned_specs(self.config)), 40)
        for name, expected in before_multi.items():
            self.assertIs(getattr(experiment.multi, name), expected)
        for name, expected in before_direct.items():
            self.assertIs(getattr(experiment.direct, name), expected)

    def test_payload_is_direct_random_init_and_never_opens_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = experiment._training_payload(
                self.config, Path(directory), self.specs[0]
            )
        self.assertEqual(
            payload["generator_conditioning_mode"], "cnn_unet_mask_coords_v1"
        )
        self.assertEqual(
            payload["critic_conditioning_mode"], "lp_disabled_same_shape_v1"
        )
        self.assertEqual(payload["generator_optimizer_profile"], "uniform_v1")
        self.assertEqual(payload["generator_learning_rate"], 5e-7)
        self.assertEqual(payload["discriminator_learning_rate"], 5e-7)
        self.assertEqual(payload["news_first_pair_text_overlay_mode"], "current_only")
        self.assertEqual(
            payload["news_first_full_training_state_mode"], "save_dynamic_v1"
        )
        self.assertEqual(payload["news_first_refit_mode"], "none")
        self.assertTrue(payload["news_first_materialize_validation_loader"])
        self.assertFalse(payload["news_first_materialize_test_loader"])
        self.assertTrue(payload["use_early_stopping"])
        self.assertEqual(payload["num_epochs"], 240)
        self.assertNotIn("parent_state_path", payload)
        self.assertNotIn("recipe_path", payload)

    def test_config_rejects_parent_style_or_high_backbone_lr_drift(self) -> None:
        parent = copy.deepcopy(self.config)
        parent["training"]["protocol"] = "parent_continuation_v1"
        with self.assertRaisesRegex(ValueError, "training.protocol"):
            experiment.validate_config(parent)
        high_lr = copy.deepcopy(self.config)
        high_lr["training"]["generator_learning_rate"] = 2.5e-5
        with self.assertRaisesRegex(ValueError, "generator_learning_rate"):
            experiment.validate_config(high_lr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
