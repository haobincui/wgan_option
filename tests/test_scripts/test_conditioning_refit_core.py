"""Core contracts for versioned conditioning and validation-free Stage-B refits."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wgan_option.config import (  # noqa: E402
    FROZEN_EPOCH_LR_REPLAY_REFIT_MODE,
    Config,
    load_refit_recipe,
)
from wgan_option.models.common import (  # noqa: E402
    BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    LP_CONCAT_CRITIC_CONDITIONING_MODE,
    LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE,
    generator_conditioning_fingerprint,
)
from wgan_option.models.discriminator import Discriminator  # noqa: E402
from wgan_option.models.gan_model import WGAN_GP  # noqa: E402
from wgan_option.models.generator import Generator  # noqa: E402
from wgan_option.utils.inference_helpers import (  # noqa: E402
    resolve_checkpoint_critic_conditioning_contract,
    resolve_checkpoint_generator_conditioning_contract,
    resolve_checkpoint_refit_contract,
    load_vol_generator,
)
from wgan_option.utils.merged_xlsx_types import VolSurfaceSample  # noqa: E402
from wgan_option.utils.news_first_dataloaders import (  # noqa: E402
    create_configured_vol_surface_dataloaders,
)


def _generator(mode: str) -> Generator:
    return Generator(
        channels=1,
        embedding_dim=5,
        noise_dim=3,
        surface_height=8,
        surface_width=8,
        base_channels=2,
        res_blocks=0,
        text_hidden_dim=4,
        text_out_dim=3,
        hidden_dim=7,
        generator_conditioning_mode=mode,
    )


def _critic(mode: str) -> Discriminator:
    return Discriminator(
        channels=1,
        embedding_dim=5,
        surface_height=16,
        surface_width=16,
        base_channels=2,
        text_hidden_dim=3,
        hidden_dim=7,
        critic_conditioning_mode=mode,
    )


def _sample() -> VolSurfaceSample:
    current = np.full((1, 2, 2), 0.2, dtype=np.float32)
    return VolSurfaceSample(
        sample_id="train-1",
        timestamp="2023-09-01T12:00:00Z",
        current_snapshot_time_utc="2023-09-01T12:00:00Z",
        target_snapshot_time_utc="2023-09-01T12:05:00Z",
        current_surface=current,
        target_surface=current + 0.01,
        text_embedding=np.asarray([0.1, 0.2], dtype=np.float32),
        strike_grid=np.asarray([0.9, 1.1], dtype=np.float32),
        maturity_grid_days=np.asarray([10.0, 20.0], dtype=np.float32),
        surface_shape=(2, 2),
        global_index=1,
        metadata={},
        pair_id="pair-1",
        session_id="session-1",
        effective_origin_utc="2023-09-01T12:00:00Z",
        stable_sample_key="stable-1",
        support_grid_fingerprint="grid-1",
    )


class TestConditioningContracts(unittest.TestCase):
    def test_defaults_and_film_residual_block_boundary(self):
        config = Config(cuda=False)
        self.assertEqual(
            config.generator_conditioning_mode,
            BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
        )
        self.assertEqual(
            config.critic_conditioning_mode,
            LP_CONCAT_CRITIC_CONDITIONING_MODE,
        )
        with self.assertRaisesRegex(ValueError, "gen_res_blocks=0"):
            Config(
                cuda=False,
                generator_conditioning_mode=(
                    FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE
                ),
                gen_res_blocks=1,
            )

    def test_film_is_identity_initialized_without_advancing_rng(self):
        torch.manual_seed(123)
        concat = _generator(BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE).eval()
        concat_tail = torch.randn(4)
        torch.manual_seed(123)
        film = _generator(
            FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE
        ).eval()
        film_tail = torch.randn(4)
        torch.testing.assert_close(concat_tail, film_tail)

        film_state = film.state_dict()
        for name, value in concat.state_dict().items():
            torch.testing.assert_close(value, film_state[name])
        expected_extra = 2 * (2 + 4 + 8) * (3 + 1)
        self.assertEqual(
            sum(parameter.numel() for parameter in film.parameters())
            - sum(parameter.numel() for parameter in concat.parameters()),
            expected_extra,
        )
        current = torch.randn(2, 1, 8, 8)
        text = torch.randn(2, 5)
        noise = torch.randn(2, 3)
        torch.testing.assert_close(
            concat(current, text, noise=noise),
            film(current, text, noise=noise),
        )
        film(current, text, noise=noise).sum().backward()
        self.assertTrue(
            any(
                layer.projection.weight.grad is not None
                and int(torch.count_nonzero(layer.projection.weight.grad)) > 0
                for layer in film.encoder_film_layers
            )
        )

    def test_disabled_critic_preserves_state_and_zeros_lp_gradients(self):
        torch.manual_seed(456)
        concat = _critic(LP_CONCAT_CRITIC_CONDITIONING_MODE)
        torch.manual_seed(456)
        disabled = _critic(LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE)
        self.assertEqual(concat.state_dict().keys(), disabled.state_dict().keys())
        self.assertEqual(
            sum(parameter.numel() for parameter in concat.parameters()),
            sum(parameter.numel() for parameter in disabled.parameters()),
        )
        for name, value in concat.state_dict().items():
            torch.testing.assert_close(value, disabled.state_dict()[name])

        current = torch.randn(2, 1, 16, 16)
        future = torch.randn(2, 1, 16, 16)
        first_text = torch.randn(2, 5, requires_grad=True)
        second_text = torch.randn(2, 5)
        first = disabled(future, current, first_text)
        second = disabled(future, current, second_text)
        torch.testing.assert_close(first, second)
        first.sum().backward()
        self.assertEqual(int(torch.count_nonzero(first_text.grad)), 0)
        for parameter in disabled.text_encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(int(torch.count_nonzero(parameter.grad)), 0)
        classifier_gradient = disabled.classifier[0].weight.grad
        self.assertIsNotNone(classifier_gradient)
        self.assertEqual(int(torch.count_nonzero(classifier_gradient[:, -3:])), 0)

    def test_checkpoint_conditioning_contract_defaults_and_fails_closed(self):
        legacy = {"config": {}}
        self.assertEqual(
            resolve_checkpoint_generator_conditioning_contract(legacy)[0],
            BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
        )
        self.assertEqual(
            resolve_checkpoint_critic_conditioning_contract(legacy)[0],
            LP_CONCAT_CRITIC_CONDITIONING_MODE,
        )
        film_mode = FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE
        checkpoint = {
            "config": {"generator_conditioning_mode": film_mode},
            "generator_conditioning_mode": film_mode,
            "generator_conditioning_fingerprint": (
                generator_conditioning_fingerprint(film_mode)
            ),
        }
        self.assertEqual(
            resolve_checkpoint_generator_conditioning_contract(checkpoint)[0],
            film_mode,
        )
        del checkpoint["generator_conditioning_fingerprint"]
        with self.assertRaisesRegex(ValueError, "missing metadata"):
            resolve_checkpoint_generator_conditioning_contract(checkpoint)

    def test_nondefault_generator_checkpoint_round_trips_through_inference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = Config(
                cuda=False,
                generator_conditioning_mode=(
                    FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE
                ),
                critic_conditioning_mode=(
                    LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE
                ),
                channels=1,
                embedding_dim=5,
                noise_dim=3,
                gen_base_channels=2,
                gen_res_blocks=0,
                gen_text_hidden_dim=4,
                gen_text_out_dim=3,
                gen_hidden_dim=7,
                disc_base_channels=2,
                disc_text_hidden_dim=3,
                disc_hidden_dim=7,
                models_path=str(root / "checkpoints"),
                metrics_path=str(root / "metrics"),
            )
            strike_grid = np.linspace(0.97, 1.03, 8, dtype=np.float32)
            maturity_grid = np.linspace(1, 38, 8, dtype=np.float32)
            model = WGAN_GP(
                config=config,
                strike_grid=strike_grid,
                maturity_grid_days=maturity_grid,
                embedding_dim=5,
            )
            checkpoint_path = Path(model.save_model()["generator"])
            sample = SimpleNamespace(
                current_surface=np.zeros((1, 8, 8), dtype=np.float32),
                strike_grid=strike_grid,
                maturity_grid_days=maturity_grid,
            )
            loaded, loaded_config, _ = load_vol_generator(
                checkpoint_path,
                sample,
                torch.device("cpu"),
            )
            self.assertEqual(
                loaded.generator_conditioning_mode,
                FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
            )
            self.assertEqual(
                loaded_config.critic_conditioning_mode,
                LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE,
            )


class TestRefitContract(unittest.TestCase):
    @staticmethod
    def _recipe(path: Path) -> str:
        payload = {
            "schema_version": 1,
            "refit_mode": FROZEN_EPOCH_LR_REPLAY_REFIT_MODE,
            "num_epochs": 2,
            "generator_lr_trace": [
                {"epoch": 1, "lr": 5e-7},
                {"epoch": 2, "lr": 2.5e-7},
            ],
            "discriminator_lr_trace": [
                {"epoch": 1, "lr": 4e-7},
                {"epoch": 2, "lr": 2e-7},
            ],
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        path.write_bytes(encoded)
        return hashlib.sha256(encoded).hexdigest()

    def test_refit_config_recipe_and_final_lr_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recipe_path = root / "recipe.json"
            recipe_sha = self._recipe(recipe_path)
            recipe = load_refit_recipe(str(recipe_path), recipe_sha)
            self.assertEqual(recipe["num_epochs"], 2)
            config = Config(
                cuda=False,
                num_epochs=2,
                batch_size=1,
                num_workers=0,
                channels=1,
                embedding_dim=2,
                noise_dim=2,
                gen_base_channels=1,
                gen_text_hidden_dim=2,
                gen_text_out_dim=2,
                gen_hidden_dim=4,
                disc_base_channels=1,
                disc_text_hidden_dim=2,
                disc_hidden_dim=4,
                discriminator_iter=1,
                use_calendar_constraint=False,
                use_butterfly_constraint=False,
                use_smooth_constraint=False,
                news_first_refit_mode=FROZEN_EPOCH_LR_REPLAY_REFIT_MODE,
                news_first_refit_recipe_path=str(recipe_path),
                news_first_refit_recipe_sha256=recipe_sha,
                news_first_materialize_validation_loader=False,
                news_first_materialize_test_loader=False,
                models_path=str(root / "checkpoints"),
                metrics_path=str(root / "metrics"),
                save_every=99,
            )
            model = WGAN_GP(
                config=config,
                strike_grid=np.linspace(0.97, 1.03, 16, dtype=np.float32),
                maturity_grid_days=np.linspace(1, 38, 16, dtype=np.float32),
                embedding_dim=2,
            )
            current = torch.full((1, 1, 16, 16), 0.2)
            text = torch.full((1, 2), 0.1)
            target = current + 0.01
            loader = DataLoader(
                TensorDataset(
                    current,
                    text,
                    target,
                    torch.ones(1),
                    torch.ones(1, dtype=torch.int64),
                ),
                batch_size=1,
            )
            model.train(loader, None, val_samples=[])
            checkpoint = torch.load(
                root / "checkpoints" / "generator.pt",
                map_location="cpu",
                weights_only=False,
            )
            lineage = resolve_checkpoint_refit_contract(checkpoint)
            self.assertEqual(
                lineage["refit_generator_lr_trace"],
                recipe["generator_lr_trace"],
            )
            self.assertEqual(
                lineage["refit_discriminator_lr_trace"],
                recipe["discriminator_lr_trace"],
            )

    def test_training_only_loader_never_materializes_common_or_validation(self):
        training = [_sample()]
        support = {
            "support_mask_applied": False,
            "time_partitions": {"train": {"input_rows": 1}},
        }
        config = Config(
            cuda=False,
            data_path="train-only.xlsx",
            batch_size=1,
            num_workers=0,
            news_first_train_end_utc="2023-10-01T00:00:00Z",
            news_first_validation_end_utc="2023-10-01T00:00:00Z",
            news_first_common_eval_data_path="must-not-be-read.xlsx",
            news_first_materialize_validation_loader=False,
            news_first_materialize_test_loader=False,
        )

        def text_transform(items, *, mode, seed, namespace):
            del mode, seed
            return list(items), {"namespace": namespace}

        with (
            patch(
                "wgan_option.utils.news_first_dataloaders._load_surface_items",
                return_value=(training, support),
            ) as load_items,
            patch(
                "wgan_option.utils.news_first_dataloaders._apply_text_ablation",
                side_effect=text_transform,
            ) as transform_text,
            patch(
                "wgan_option.utils.news_first_dataloaders._validate_shared_grid"
            ) as validate_grid,
            patch(
                "wgan_option.utils.news_first_dataloaders._validate_label_reliability_data_window_contract"
            ) as validate_common,
        ):
            bundle = create_configured_vol_surface_dataloaders(config)

        load_items.assert_called_once()
        transform_text.assert_called_once()
        validate_grid.assert_not_called()
        validate_common.assert_not_called()
        self.assertIsNone(bundle.val_loader)
        self.assertEqual(bundle.val_items, [])
        self.assertEqual(bundle.val_samples, 0)
        self.assertEqual(bundle.split_metadata["validation_materialized"], False)
        self.assertEqual(
            bundle.split_metadata["validation_forbidden_from_utc_inclusive"],
            "2023-10-01T00:00:00+00:00",
        )


if __name__ == "__main__":
    unittest.main()
