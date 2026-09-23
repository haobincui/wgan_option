from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from scripts.rq3.news_first_vol_training import _validate_frozen_config
from wgan_option.config import Config, parse_cli_overrides
from wgan_option.models.common import (
    IDENTITY_RESIDUAL_OUTPUT_MODE,
    LEGACY_RESIDUAL_OUTPUT_MODE,
    RESIDUAL_OUTPUT_EPSILON,
    apply_residual_surface_output,
    residual_output_fingerprint,
)
from wgan_option.models.gan_model import WGAN_GP
from wgan_option.models.generator import Generator
from wgan_option.models.vol_regressor import VolSurfaceRegressor
from wgan_option.train_vol_regression_xlsx import VolSurfaceRegressionTrainer
from wgan_option.utils.inference_helpers import (
    load_vol_generator,
    load_vol_regressor,
    resolve_checkpoint_residual_output_contract,
)


class TestResidualOutputContract(unittest.TestCase):
    @staticmethod
    def _current_surface() -> torch.Tensor:
        return torch.tensor(
            [
                [
                    [
                        [0.05, 0.08, 0.12, 0.20],
                        [0.06, 0.09, 0.13, 0.21],
                        [0.07, 0.10, 0.14, 0.22],
                        [0.08, 0.11, 0.15, 0.23],
                    ]
                ]
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _generator(*, mode: str) -> Generator:
        return Generator(
            channels=1,
            embedding_dim=3,
            noise_dim=2,
            surface_height=4,
            surface_width=4,
            base_channels=2,
            text_hidden_dim=4,
            text_out_dim=3,
            hidden_dim=8,
            residual_output_mode=mode,
        )

    @staticmethod
    def _regressor(*, mode: str) -> VolSurfaceRegressor:
        return VolSurfaceRegressor(
            channels=1,
            embedding_dim=3,
            surface_height=4,
            surface_width=4,
            base_channels=2,
            text_hidden_dim=4,
            text_out_dim=3,
            hidden_dim=8,
            residual_output_mode=mode,
        )

    @staticmethod
    def _checkpoint_config(*, mode: str) -> Config:
        return Config(
            cuda=False,
            channels=1,
            embedding_dim=3,
            noise_dim=2,
            strike_bins=4,
            maturity_bins=4,
            gen_base_channels=2,
            gen_text_hidden_dim=4,
            gen_text_out_dim=3,
            gen_hidden_dim=8,
            residual_output_mode=mode,
        )

    def test_identity_transform_delta_zero_is_exact_identity_and_positive(self):
        current = self._current_surface()
        output = apply_residual_surface_output(
            current,
            torch.zeros_like(current),
            mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
        )
        self.assertTrue(torch.equal(output, current))

        lower_limit = apply_residual_surface_output(
            current,
            torch.full_like(current, -100.0),
            mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
        )
        self.assertTrue(torch.isfinite(lower_limit).all())
        self.assertTrue((lower_limit > 0.0).all())
        self.assertTrue(
            torch.allclose(
                lower_limit,
                torch.full_like(lower_limit, RESIDUAL_OUTPUT_EPSILON),
                atol=5e-8,
                rtol=0.0,
            )
        )

    def test_identity_models_zero_initialize_head_and_start_at_persistence(self):
        current = self._current_surface()
        text = torch.randn(1, 3)

        generator = self._generator(mode=IDENTITY_RESIDUAL_OUTPUT_MODE)
        self.assertEqual(torch.count_nonzero(generator.fusion[-1].weight).item(), 0)
        self.assertEqual(torch.count_nonzero(generator.fusion[-1].bias).item(), 0)
        generated = generator(current, text, noise=torch.randn(1, 2))
        self.assertTrue(torch.equal(generated, current))

        regressor = self._regressor(mode=IDENTITY_RESIDUAL_OUTPUT_MODE)
        self.assertEqual(torch.count_nonzero(regressor.fusion[-1].weight).item(), 0)
        self.assertEqual(torch.count_nonzero(regressor.fusion[-1].bias).item(), 0)
        predicted = regressor(current, text)
        self.assertTrue(torch.equal(predicted, current))

    def test_legacy_mode_preserves_historical_non_identity_formula(self):
        current = self._current_surface()
        zero = torch.zeros_like(current)
        actual = apply_residual_surface_output(
            current,
            zero,
            mode=LEGACY_RESIDUAL_OUTPUT_MODE,
        )
        expected = torch.nn.functional.softplus(current) + RESIDUAL_OUTPUT_EPSILON
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, current))

        legacy = self._generator(mode=LEGACY_RESIDUAL_OUTPUT_MODE)
        self.assertGreater(torch.count_nonzero(legacy.fusion[-1].weight).item(), 0)

    def test_config_and_wgan_construction_propagate_identity_mode(self):
        overrides = parse_cli_overrides(
            [f"residual_output_mode={IDENTITY_RESIDUAL_OUTPUT_MODE}"]
        )
        self.assertEqual(
            overrides["residual_output_mode"],
            IDENTITY_RESIDUAL_OUTPUT_MODE,
        )
        config = self._checkpoint_config(mode=IDENTITY_RESIDUAL_OUTPUT_MODE)
        model = WGAN_GP(
            config,
            strike_grid=np.linspace(0.97, 1.03, 4),
            maturity_grid_days=np.asarray([7, 17, 28, 38]),
            embedding_dim=3,
        )
        self.assertEqual(model.G.residual_output_mode, IDENTITY_RESIDUAL_OUTPUT_MODE)
        self.assertEqual(
            model.G.residual_output_fingerprint,
            residual_output_fingerprint(IDENTITY_RESIDUAL_OUTPUT_MODE),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported residual_output_mode"):
            Config(residual_output_mode="unknown")

    def test_old_checkpoints_without_contract_metadata_load_as_legacy(self):
        config = self._checkpoint_config(mode=LEGACY_RESIDUAL_OUTPUT_MODE)
        old_config = asdict(config)
        old_config.pop("residual_output_mode")
        sample = SimpleNamespace(current_surface=np.ones((1, 4, 4), dtype=np.float32))
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            generator_path = Path(tmpdir) / "generator.pt"
            torch.save(
                {
                    "state_dict": self._generator(
                        mode=LEGACY_RESIDUAL_OUTPUT_MODE
                    ).state_dict(),
                    "config": old_config,
                    "embedding_dim": 3,
                },
                generator_path,
            )
            generator, generator_config, _ = load_vol_generator(
                generator_path, sample, device
            )
            self.assertEqual(
                generator.residual_output_mode, LEGACY_RESIDUAL_OUTPUT_MODE
            )
            self.assertEqual(
                generator_config.residual_output_mode,
                LEGACY_RESIDUAL_OUTPUT_MODE,
            )

            regressor_path = Path(tmpdir) / "regressor.pt"
            torch.save(
                {
                    "state_dict": self._regressor(
                        mode=LEGACY_RESIDUAL_OUTPUT_MODE
                    ).state_dict(),
                    "config": old_config,
                    "embedding_dim": 3,
                },
                regressor_path,
            )
            regressor, regressor_config, _ = load_vol_regressor(
                regressor_path, sample, device
            )
            self.assertEqual(
                regressor.residual_output_mode, LEGACY_RESIDUAL_OUTPUT_MODE
            )
            self.assertEqual(
                regressor_config.residual_output_mode,
                LEGACY_RESIDUAL_OUTPUT_MODE,
            )

    def test_identity_checkpoint_requires_matching_formula_fingerprint(self):
        config = asdict(self._checkpoint_config(mode=IDENTITY_RESIDUAL_OUTPUT_MODE))
        fingerprint = residual_output_fingerprint(IDENTITY_RESIDUAL_OUTPUT_MODE)
        checkpoint = {
            "config": config,
            "residual_output_mode": IDENTITY_RESIDUAL_OUTPUT_MODE,
            "residual_output_fingerprint": fingerprint,
        }
        self.assertEqual(
            resolve_checkpoint_residual_output_contract(checkpoint),
            (IDENTITY_RESIDUAL_OUTPUT_MODE, fingerprint),
        )

        missing = dict(checkpoint)
        missing.pop("residual_output_fingerprint")
        with self.assertRaisesRegex(ValueError, "missing residual_output_fingerprint"):
            resolve_checkpoint_residual_output_contract(missing)

        corrupted = dict(checkpoint, residual_output_fingerprint="0" * 64)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            resolve_checkpoint_residual_output_contract(corrupted)

        conflicting = dict(checkpoint, residual_output_mode=LEGACY_RESIDUAL_OUTPUT_MODE)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            resolve_checkpoint_residual_output_contract(conflicting)

    def test_trainers_save_and_load_identity_contract_for_both_model_families(self):
        sample = SimpleNamespace(current_surface=np.ones((1, 4, 4), dtype=np.float32))
        device = torch.device("cpu")
        expected_fingerprint = residual_output_fingerprint(
            IDENTITY_RESIDUAL_OUTPUT_MODE
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._checkpoint_config(mode=IDENTITY_RESIDUAL_OUTPUT_MODE)
            config.models_path = str(Path(tmpdir) / "checkpoints")

            wgan = WGAN_GP(
                config,
                strike_grid=np.linspace(0.97, 1.03, 4),
                maturity_grid_days=np.asarray([7, 17, 28, 38]),
                embedding_dim=3,
            )
            wgan_paths = wgan.save_model(label="best")
            wgan_payload = torch.load(
                wgan_paths["generator"], map_location="cpu", weights_only=False
            )
            self.assertEqual(
                wgan_payload["residual_output_mode"],
                IDENTITY_RESIDUAL_OUTPUT_MODE,
            )
            self.assertEqual(
                wgan_payload["residual_output_fingerprint"],
                expected_fingerprint,
            )
            loaded_generator, loaded_generator_config, _ = load_vol_generator(
                wgan_paths["generator"], sample, device
            )
            self.assertEqual(
                loaded_generator.residual_output_mode,
                IDENTITY_RESIDUAL_OUTPUT_MODE,
            )
            self.assertEqual(
                loaded_generator_config.residual_output_mode,
                IDENTITY_RESIDUAL_OUTPUT_MODE,
            )

            trainer = VolSurfaceRegressionTrainer(config)
            trainer.model = self._regressor(mode=IDENTITY_RESIDUAL_OUTPUT_MODE)
            trainer.bundle = SimpleNamespace(embedding_dim=3)
            regression_path = trainer.save_model(label="best")["model"]
            regression_payload = torch.load(
                regression_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(
                regression_payload["residual_output_mode"],
                IDENTITY_RESIDUAL_OUTPUT_MODE,
            )
            self.assertEqual(
                regression_payload["residual_output_fingerprint"],
                expected_fingerprint,
            )
            loaded_regressor, loaded_regressor_config, _ = load_vol_regressor(
                regression_path, sample, device
            )
            self.assertEqual(
                loaded_regressor.residual_output_mode,
                IDENTITY_RESIDUAL_OUTPUT_MODE,
            )
            self.assertEqual(
                loaded_regressor_config.residual_output_mode,
                IDENTITY_RESIDUAL_OUTPUT_MODE,
            )

    def test_orchestrator_accepts_shared_identity_mode_and_rejects_mismatch(self):
        path = Path("configs/rq3/news_first_vol_training_narrow_grid.yaml")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))[
            "news_first_vol_training"
        ]
        _validate_frozen_config(config)

        for family in ("wgan", "regression"):
            config["models"][family]["training"][
                "residual_output_mode"
            ] = IDENTITY_RESIDUAL_OUTPUT_MODE
        _validate_frozen_config(config)

        config["models"]["regression"]["training"][
            "residual_output_mode"
        ] = LEGACY_RESIDUAL_OUTPUT_MODE
        with self.assertRaisesRegex(ValueError, "same residual_output_mode"):
            _validate_frozen_config(config)


if __name__ == "__main__":
    unittest.main()
