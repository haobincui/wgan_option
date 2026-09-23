"""Contracts for the opt-in FiLM U-Net split Generator optimizer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from wgan_option.config import Config
from wgan_option.models.common import (
    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    IDENTITY_RESIDUAL_OUTPUT_MODE,
)
from wgan_option.models.gan_model import WGAN_GP


class GroupedGeneratorOptimizerTest(unittest.TestCase):
    def _config(self, root: Path, **overrides: object) -> Config:
        values: dict[str, object] = {
            "cuda": False,
            "num_epochs": 3,
            "learning_rate": 5e-7,
            "generator_learning_rate": 5e-7,
            "generator_optimizer_profile": "film_unet_split_lr_v1",
            "generator_text_learning_rate": 2.5e-6,
            "generator_film_learning_rate": 1e-6,
            "generator_text_min_learning_rate": 2.5e-7,
            "generator_film_min_learning_rate": 1e-7,
            "reduce_lr_min_lr": 5e-8,
            "generator_conditioning_mode": (
                FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
            ),
            "generator_current_input_mode": (
                CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
            ),
            "support_mask_mode": "raw_joint",
            "residual_output_mode": IDENTITY_RESIDUAL_OUTPUT_MODE,
            "gen_base_channels": 32,
            "gen_text_hidden_dim": 256,
            "gen_text_out_dim": 128,
            "disc_base_channels": 2,
            "disc_res_blocks": 0,
            "disc_text_hidden_dim": 2,
            "disc_hidden_dim": 4,
            "noise_dim": 32,
            "models_path": str(root / "models"),
            "outputs_path": str(root / "models"),
            "samples_path": str(root / "samples"),
            "metrics_path": str(root / "metrics"),
        }
        values.update(overrides)
        return Config(**values)

    def _model(self, config: Config) -> WGAN_GP:
        return WGAN_GP(
            config,
            embedding_dim=1024,
            strike_grid=np.linspace(0.97, 1.03, 16, dtype=np.float32),
            maturity_grid_days=np.linspace(1.0, 38.0, 16, dtype=np.float32),
        )

    def test_split_groups_are_complete_disjoint_and_named(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self._model(self._config(Path(directory)))
        groups = model.g_optimizer.param_groups
        self.assertEqual(
            [group["group_name"] for group in groups],
            ["backbone", "text_encoder", "film_projection"],
        )
        self.assertEqual([group["lr"] for group in groups], [5e-7, 2.5e-6, 1e-6])
        grouped_ids = [
            id(parameter) for group in groups for parameter in group["params"]
        ]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(
            set(grouped_ids), {id(parameter) for parameter in model.G.parameters()}
        )
        counts = {
            group["group_name"]: sum(parameter.numel() for parameter in group["params"])
            for group in groups
        }
        self.assertEqual(
            counts,
            {"backbone": 416353, "text_encoder": 295808, "film_projection": 115584},
        )
        self.assertEqual(sum(counts.values()), 827745)

    def test_uniform_default_preserves_one_unnamed_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(
                Path(directory),
                generator_optimizer_profile="uniform_v1",
                generator_text_learning_rate=0.0,
                generator_film_learning_rate=0.0,
                generator_text_min_learning_rate=0.0,
                generator_film_min_learning_rate=0.0,
            )
            model = self._model(config)
        self.assertEqual(len(model.g_optimizer.param_groups), 1)
        self.assertNotIn("group_name", model.g_optimizer.param_groups[0])
        self.assertEqual(model._generator_optimizer_group_learning_rates(), {})
        self.assertNotIn("generator_optimizer_profile", model._learning_rate_contract())

    def test_warmup_preserves_each_group_target_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self._model(
                self._config(
                    Path(directory), lr_warmup_epochs=3, lr_warmup_start_factor=0.1
                )
            )
        model._apply_learning_rate_warmup(1)
        np.testing.assert_allclose(
            list(model._generator_optimizer_group_learning_rates().values()),
            [5e-8, 2.5e-7, 1e-7],
            rtol=0.0,
            atol=1e-15,
        )
        model._apply_learning_rate_warmup(2)
        np.testing.assert_allclose(
            list(model._generator_optimizer_group_learning_rates().values()),
            [2.75e-7, 1.375e-6, 5.5e-7],
            rtol=0.0,
            atol=1e-15,
        )
        model._apply_learning_rate_warmup(3)
        np.testing.assert_allclose(
            list(model._generator_optimizer_group_learning_rates().values()),
            [5e-7, 2.5e-6, 1e-6],
            rtol=0.0,
            atol=1e-15,
        )

    def test_plateau_uses_per_group_floors_and_contract_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self._model(
                self._config(
                    Path(directory), reduce_lr_factor=0.5, reduce_lr_patience=0
                )
            )
        scheduler = model._create_plateau_scheduler(model.g_optimizer)
        self.assertEqual(scheduler.min_lrs, [5e-8, 2.5e-7, 1e-7])
        for _ in range(10):
            scheduler.step(1.0)
        np.testing.assert_allclose(
            list(model._generator_optimizer_group_learning_rates().values()),
            [5e-8, 2.5e-7, 1e-7],
            rtol=0.0,
            atol=1e-15,
        )
        row = {"epoch": 1, "g_lr": 5e-7, "d_lr": 5e-7}
        row.update(
            {
                "g_lr_backbone": 5e-7,
                "g_lr_text_encoder": 2.5e-6,
                "g_lr_film_projection": 1e-6,
            }
        )
        model._metrics_rows = [row]
        contract = model._learning_rate_contract()
        self.assertEqual(
            contract["generator_group_scheduler_min_learning_rates"],
            {"backbone": 5e-8, "text_encoder": 2.5e-7, "film_projection": 1e-7},
        )
        self.assertEqual(
            contract["generator_group_lr_trace"][0]["text_encoder"], 2.5e-6
        )

    def test_split_profile_requires_positive_special_learning_rates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires positive"):
                self._config(Path(directory), generator_text_learning_rate=0.0)


if __name__ == "__main__":
    unittest.main()
