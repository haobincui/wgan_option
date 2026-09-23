"""Contracts for the seed-42 pure-CNN no-text rolling experiment."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.rq3 import news_first_vol_cnn_unet_pure_no_text_seed42 as experiment
from wgan_option.models.generator import Generator


class PureCnnNoTextOrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = experiment.load_config(experiment.DEFAULT_CONFIG)
        cls.specs = experiment.planned_specs(cls.config)

    def test_frozen_matrix_and_model_contract(self) -> None:
        self.assertEqual(experiment.DIRECT_ARMS, ("pure_cnn_no_text",))
        self.assertEqual(experiment.EXPECTED_TRAINING_JOBS, 4)
        self.assertEqual(experiment.EXPECTED_PREDICTION_CELLS, 4)
        self.assertEqual(experiment.EXPECTED_PAIR_METRIC_ROWS, 500)
        self.assertEqual(self.config["matrix"]["seeds"], [42])
        self.assertEqual(self.config["matrix"]["direct_arms"], ["pure_cnn_no_text"])
        self.assertEqual(
            self.config["model"]["generator_conditioning_mode"],
            "cnn_unet_mask_coords_v1",
        )
        self.assertEqual(
            self.config["model"]["critic_conditioning_mode"],
            "lp_disabled_same_shape_v1",
        )
        contract = experiment.model_contract(self.config)
        self.assertEqual(contract["generator_parameters"], 416_353)
        self.assertEqual(contract["critic_parameters"], 729_157)
        self.assertEqual(contract["total_parameters"], 1_145_510)

    def test_four_unique_jobs_are_balanced_two_per_gpu(self) -> None:
        self.assertEqual(len(self.specs), 4)
        self.assertEqual(len({row["job_id"] for row in self.specs}), 4)
        self.assertEqual(
            {gpu: sum(row["gpu_id"] == gpu for row in self.specs) for gpu in (0, 1)},
            {0: 2, 1: 2},
        )
        experiment.validate_gpu_balance(self.specs)

    def test_profile_restores_shared_direct_orchestrator(self) -> None:
        names = (
            "DIRECT_ARMS",
            "NO_TEXT_ARM",
            "EXPECTED_TRAINING_JOBS",
            "GENERATOR_MODE",
            "EXPECTED_PARAMETER_COUNTS",
            "WORKER_MODULE",
        )
        before = {name: getattr(experiment.direct, name) for name in names}
        with experiment.pure_cnn_profile():
            self.assertEqual(experiment.direct.DIRECT_ARMS, ("pure_cnn_no_text",))
            self.assertEqual(experiment.direct.EXPECTED_TRAINING_JOBS, 4)
            self.assertEqual(
                experiment.direct.GENERATOR_MODE, "cnn_unet_mask_coords_v1"
            )
        for name, expected in before.items():
            self.assertIs(getattr(experiment.direct, name), expected)

    def test_training_payload_is_current_only_and_never_opens_test(self) -> None:
        with experiment.pure_cnn_profile(), tempfile.TemporaryDirectory() as directory:
            payload = experiment.direct._training_payload(
                self.config, Path(directory), self.specs[0]
            )
        self.assertEqual(
            payload["generator_conditioning_mode"], "cnn_unet_mask_coords_v1"
        )
        self.assertEqual(
            payload["critic_conditioning_mode"], "lp_disabled_same_shape_v1"
        )
        self.assertEqual(payload["news_first_pair_text_overlay_mode"], "current_only")
        self.assertTrue(payload["news_first_materialize_validation_loader"])
        self.assertFalse(payload["news_first_materialize_test_loader"])
        self.assertEqual(payload["num_epochs"], 240)

    def test_config_rejects_film_or_non_four_cell_drift(self) -> None:
        film = copy.deepcopy(self.config)
        film["model"]["generator_conditioning_mode"] = "film_unet_mask_coords_v1"
        with self.assertRaisesRegex(ValueError, "Generator"):
            experiment.validate_config(film)
        count = copy.deepcopy(self.config)
        count["matrix"]["expected_training_jobs"] = 20
        with self.assertRaisesRegex(ValueError, "training jobs"):
            experiment.validate_config(count)

    def test_generator_is_text_free_and_text_invariant(self) -> None:
        model = self.config["model"]
        generator = Generator(
            channels=1,
            embedding_dim=1024,
            noise_dim=32,
            surface_height=16,
            surface_width=16,
            base_channels=32,
            res_blocks=0,
            text_hidden_dim=256,
            text_out_dim=128,
            hidden_dim=1024,
            residual_output_mode="identity_softplus_residual",
            generator_noise_mode="gaussian",
            generator_current_input_mode="current_support_masked",
            generator_conditioning_mode="cnn_unet_mask_coords_v1",
            strike_grid=np.asarray(
                self.config["data"]["strike_grid"], dtype=np.float32
            ),
            maturity_grid_days=np.asarray(
                self.config["data"]["maturity_days_grid"], dtype=np.float32
            ),
        ).eval()
        self.assertEqual(
            sum(parameter.numel() for parameter in generator.parameters()), 416_353
        )
        self.assertFalse(hasattr(generator, "text_encoder"))
        self.assertFalse(
            any("film" in name.lower() for name, _ in generator.named_modules())
        )
        current = torch.rand(2, 1, 16, 16) + 0.1
        support = torch.ones_like(current)
        noise = torch.randn(2, 32)
        with torch.no_grad():
            first = generator(
                current,
                torch.zeros(2, 1024),
                noise,
                current_support_mask=support,
            )
            second = generator(
                current,
                torch.randn(2, 1024),
                noise,
                current_support_mask=support,
            )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(model["expected_generator_parameters"], 416_353)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
