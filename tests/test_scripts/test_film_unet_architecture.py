"""Contracts for the branch-local FiLM-only and fully convolutional ablations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from wgan_option.config import Config
from wgan_option.models.common import (
    CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    FILM_UNET_GENERATOR_CONDITIONING_MODE,
    FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    IDENTITY_RESIDUAL_OUTPUT_MODE,
    LP_CONCAT_CRITIC_CONDITIONING_MODE,
    LP_PROJECTION_CRITIC_CONDITIONING_MODE,
    critic_conditioning_contract,
    generator_conditioning_contract,
)
from wgan_option.models.discriminator import Discriminator
from wgan_option.models.gan_model import WGAN_GP
from wgan_option.models.generator import Generator
from wgan_option.utils.inference_helpers import load_vol_generator


EXACT_TTM = np.asarray(
    [1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38],
    dtype=np.float32,
)
EXACT_MONEYNESS = np.linspace(0.97, 1.03, 16, dtype=np.float32)


def _legacy_sized_generator(mode: str, *, text_dim: int = 128) -> Generator:
    kwargs: dict[str, object] = {}
    if mode in {
        FILM_UNET_GENERATOR_CONDITIONING_MODE,
        FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    }:
        kwargs.update(
            generator_current_input_mode=(CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE),
        )
    if mode in {
        FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    }:
        kwargs.update(
            strike_grid=EXACT_MONEYNESS,
            maturity_grid_days=EXACT_TTM,
        )
    return Generator(
        channels=1,
        embedding_dim=1024,
        noise_dim=32,
        surface_height=16,
        surface_width=16,
        base_channels=32,
        res_blocks=0,
        text_hidden_dim=256 if text_dim == 128 else 64,
        text_out_dim=text_dim,
        hidden_dim=1024,
        residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
        generator_conditioning_mode=mode,
        **kwargs,
    )


class TestFiLMOnlyGenerator(unittest.TestCase):
    def test_no_bottleneck_concat_has_expected_shape_and_parameter_count(self):
        baseline = _legacy_sized_generator(
            FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE
        )
        model = _legacy_sized_generator(
            FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE
        ).eval()
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), 4_020_288)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3_889_216)
        self.assertEqual(
            baseline.fusion[0].in_features - model.fusion[0].in_features,
            128,
        )
        self.assertFalse(
            generator_conditioning_contract(
                FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE
            )["bottleneck_text_concat"]
        )

        current = torch.rand(2, 1, 16, 16) + 0.1
        text = torch.randn(2, 1024)
        result = model(current, text, noise=torch.zeros(2, 32))
        self.assertEqual(tuple(result.shape), (2, 1, 16, 16))
        torch.testing.assert_close(result, current, rtol=0.0, atol=0.0)

    def test_no_bottleneck_concat_text_reaches_output_only_through_film(self):
        torch.manual_seed(11)
        model = _legacy_sized_generator(
            FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE
        ).eval()
        with torch.no_grad():
            model.fusion[-1].weight.normal_(std=0.01)
            for layer in model.encoder_film_layers:
                layer.projection.weight.normal_(std=0.01)

        current = torch.rand(2, 1, 16, 16) + 0.1
        text = torch.randn(2, 1024, requires_grad=True)
        noise = torch.zeros(2, 32)
        output = model(current, text, noise=noise)
        output.sum().backward()
        self.assertIsNotNone(text.grad)
        self.assertTrue(bool(torch.isfinite(text.grad).all()))
        self.assertGreater(int(torch.count_nonzero(text.grad)), 0)
        with torch.no_grad():
            changed = model(current, text.detach().flip(0), noise=noise)
        self.assertGreater(float((output.detach() - changed).abs().max()), 0.0)


class TestFiLMUNetGenerator(unittest.TestCase):
    def test_four_channel_input_coordinates_identity_and_mask_validation(self):
        model = _legacy_sized_generator(
            FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
        ).eval()
        self.assertNotIn("moneyness_coordinate", model.state_dict())
        self.assertNotIn("log_ttm_coordinate", model.state_dict())
        captured: list[torch.Tensor] = []
        handle = model.surface_encoder[0].register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        current = torch.rand(2, 1, 16, 16) + 0.1
        mask = torch.ones(2, 1, 16, 16)
        mask[:, :, 0, 0] = 0.0
        text = torch.randn(2, 1024)
        result = model(
            current,
            text,
            noise=torch.zeros(2, 32),
            current_support_mask=mask,
        )
        handle.remove()
        torch.testing.assert_close(result, current, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(captured[0].shape), (2, 4, 16, 16))
        torch.testing.assert_close(captured[0][:, 0:1], current * mask)
        torch.testing.assert_close(captured[0][:, 1:2], mask)
        torch.testing.assert_close(
            captured[0][:, 2:3], model.moneyness_coordinate.expand(2, -1, -1, -1)
        )
        torch.testing.assert_close(
            captured[0][:, 3:4], model.log_ttm_coordinate.expand(2, -1, -1, -1)
        )
        self.assertAlmostEqual(float(model.moneyness_coordinate.min()), -1.0)
        self.assertAlmostEqual(float(model.moneyness_coordinate.max()), 1.0)
        self.assertAlmostEqual(float(model.log_ttm_coordinate.min()), -1.0)
        self.assertAlmostEqual(float(model.log_ttm_coordinate.max()), 1.0)

        with self.assertRaisesRegex(ValueError, "explicit current_support_mask"):
            model(current, text, noise=torch.zeros(2, 32))
        bad_mask = mask.clone()
        bad_mask[:, :, 1, 1] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            model(
                current,
                text,
                noise=torch.zeros(2, 32),
                current_support_mask=bad_mask,
            )

    def test_unet_parameter_counts_and_text_gradient(self):
        model128 = _legacy_sized_generator(
            FILM_UNET_GENERATOR_CONDITIONING_MODE,
            text_dim=128,
        )
        model64 = _legacy_sized_generator(
            FILM_UNET_GENERATOR_CONDITIONING_MODE,
            text_dim=64,
        )
        coords128 = _legacy_sized_generator(
            FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            text_dim=128,
        )
        self.assertEqual(sum(p.numel() for p in model128.parameters()), 826_881)
        self.assertEqual(sum(p.numel() for p in model64.parameters()), 543_617)
        self.assertEqual(sum(p.numel() for p in coords128.parameters()), 827_745)
        self.assertEqual(model128.surface_encoder[0].in_channels, 1)
        self.assertEqual(coords128.surface_encoder[0].in_channels, 4)

        torch.manual_seed(13)
        model = model64.eval()
        with torch.no_grad():
            model.residual_head.weight.normal_(std=0.01)
            film_layers = [
                *model.encoder_film_layers,
                model.bottleneck_film_layer,
                *model.decoder_film_layers,
            ]
            for layer in film_layers:
                layer.projection.weight.normal_(std=0.01)
        current = torch.rand(2, 1, 16, 16) + 0.1
        mask = torch.ones_like(current)
        text = torch.randn(2, 1024, requires_grad=True)
        noise = torch.randn(2, 32)
        first = model(
            current,
            text,
            noise=noise,
            current_support_mask=mask,
        )
        first.sum().backward()
        self.assertIsNotNone(text.grad)
        self.assertTrue(bool(torch.isfinite(text.grad).all()))
        self.assertGreater(int(torch.count_nonzero(text.grad)), 0)
        with torch.no_grad():
            second = model(
                current,
                text.detach().flip(0),
                noise=noise,
                current_support_mask=mask,
            )
        self.assertGreater(float((first.detach() - second).abs().max()), 0.0)

    def test_config_requires_masked_current_contract(self):
        with self.assertRaisesRegex(ValueError, "current_support_masked"):
            Config(
                cuda=False,
                generator_conditioning_mode=(
                    FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
                ),
            )
        config = Config(
            cuda=False,
            generator_conditioning_mode=FILM_UNET_GENERATOR_CONDITIONING_MODE,
            generator_current_input_mode=(CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE),
            support_mask_mode="raw_joint",
        )
        self.assertEqual(
            config.generator_conditioning_mode,
            FILM_UNET_GENERATOR_CONDITIONING_MODE,
        )


class TestPureCNNUNetGenerator(unittest.TestCase):
    def test_has_no_text_or_film_modules_and_is_text_invariant(self):
        model = _legacy_sized_generator(
            CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
        ).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 416_353)
        self.assertFalse(hasattr(model, "text_encoder"))
        self.assertFalse(hasattr(model, "encoder_film_layers"))
        self.assertFalse(hasattr(model, "bottleneck_film_layer"))
        self.assertFalse(hasattr(model, "decoder_film_layers"))
        self.assertFalse(
            any(isinstance(module, torch.nn.Linear) for module in model.modules())
        )
        contract = generator_conditioning_contract(
            CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
        )
        self.assertEqual(contract["text_embedding_treatment"], "ignored")
        self.assertFalse(contract["text_encoder_registered"])
        self.assertEqual(contract["film_injection_points"], [])

        current = torch.rand(2, 1, 16, 16) + 0.1
        mask = torch.ones_like(current)
        mask[:, :, 0, 0] = 0.0
        noise = torch.randn(2, 32)
        first_text = torch.randn(2, 1024, requires_grad=True)
        with torch.no_grad():
            identity = model(
                current,
                first_text,
                noise=noise,
                current_support_mask=mask,
            )
        torch.testing.assert_close(identity, current, rtol=0.0, atol=0.0)
        with torch.no_grad():
            model.residual_head.weight.normal_(std=0.01)
        first = model(
            current,
            first_text,
            noise=noise,
            current_support_mask=mask,
        )
        first.sum().backward()
        self.assertIsNone(first_text.grad)
        self.assertIsNotNone(model.surface_encoder[0].weight.grad)
        self.assertGreater(
            int(torch.count_nonzero(model.surface_encoder[0].weight.grad)), 0
        )
        with torch.no_grad():
            second = model(
                current,
                torch.full((2, 7), float("nan")),
                noise=noise,
                current_support_mask=mask,
            )
        torch.testing.assert_close(first.detach(), second, rtol=0.0, atol=0.0)

    def test_shared_convolutions_and_following_critic_match_film_seed(self):
        torch.manual_seed(37)
        film = _legacy_sized_generator(
            FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
        )
        film_critic = Discriminator(
            channels=1,
            embedding_dim=1024,
            surface_height=16,
            surface_width=16,
            base_channels=32,
            text_hidden_dim=128,
            hidden_dim=786,
        )

        torch.manual_seed(37)
        cnn = _legacy_sized_generator(CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE)
        cnn_critic = Discriminator(
            channels=1,
            embedding_dim=1024,
            surface_height=16,
            surface_width=16,
            base_channels=32,
            text_hidden_dim=128,
            hidden_dim=786,
        )

        shared_prefixes = (
            "surface_encoder.",
            "bottleneck_conv.",
            "decoder_convs.",
            "residual_head.",
        )
        film_state = film.state_dict()
        cnn_state = cnn.state_dict()
        for name, value in cnn_state.items():
            if name.startswith(shared_prefixes):
                torch.testing.assert_close(value, film_state[name], rtol=0.0, atol=0.0)
        for name, value in cnn_critic.state_dict().items():
            torch.testing.assert_close(
                value,
                film_critic.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )

    def test_requires_masked_current_contract(self):
        with self.assertRaisesRegex(ValueError, "current_support_masked"):
            Config(
                cuda=False,
                generator_conditioning_mode=(
                    CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
                ),
            )
        config = Config(
            cuda=False,
            generator_conditioning_mode=(
                CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
            ),
            generator_current_input_mode=(CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE),
            support_mask_mode="raw_joint",
        )
        self.assertEqual(
            config.generator_conditioning_mode,
            CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        )


class TestProjectionCritic(unittest.TestCase):
    def test_projection_is_parameter_matched_and_text_sensitive(self):
        kwargs = dict(
            channels=1,
            embedding_dim=1024,
            surface_height=16,
            surface_width=16,
            base_channels=32,
            text_hidden_dim=128,
            hidden_dim=786,
        )
        concat = Discriminator(
            **kwargs, critic_conditioning_mode=LP_CONCAT_CRITIC_CONDITIONING_MODE
        )
        projection = Discriminator(
            **kwargs,
            critic_conditioning_mode=LP_PROJECTION_CRITIC_CONDITIONING_MODE,
        ).eval()
        self.assertEqual(sum(p.numel() for p in concat.parameters()), 729_157)
        self.assertEqual(sum(p.numel() for p in projection.parameters()), 729_157)
        self.assertTrue(
            critic_conditioning_contract(LP_PROJECTION_CRITIC_CONDITIONING_MODE)[
                "parameter_count_preserved"
            ]
        )

        current = torch.rand(2, 1, 16, 16)
        future = torch.rand(2, 1, 16, 16)
        text = torch.randn(2, 1024, requires_grad=True)
        first = projection(future, current, text)
        self.assertEqual(tuple(first.shape), (2, 1))
        first.sum().backward()
        self.assertIsNotNone(text.grad)
        self.assertGreater(int(torch.count_nonzero(text.grad)), 0)
        with torch.no_grad():
            second = projection(future, current, text.detach().flip(0))
        self.assertGreater(float((first.detach() - second).abs().max()), 0.0)


class TestFiLMUNetInferenceRoundTrip(unittest.TestCase):
    def test_loader_reconstructs_nonuniform_coordinates_from_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = Config(
                cuda=False,
                channels=1,
                embedding_dim=5,
                noise_dim=3,
                generator_conditioning_mode=(
                    FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
                ),
                generator_current_input_mode=(
                    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
                ),
                support_mask_mode="raw_joint",
                residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
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
            strike = np.asarray([0.91, 0.94, 0.99, 1.0, 1.01, 1.04, 1.08, 1.2])
            maturity = np.asarray([1, 2, 4, 8, 16, 30, 60, 120])
            wgan = WGAN_GP(config, strike, maturity, embedding_dim=5)
            checkpoint = Path(wgan.save_model()["generator"])
            sample = SimpleNamespace(
                current_surface=np.full((1, 8, 8), 0.2, dtype=np.float32),
                strike_grid=strike,
                maturity_grid_days=maturity,
            )
            loaded, _, _ = load_vol_generator(checkpoint, sample, torch.device("cpu"))
            torch.testing.assert_close(
                loaded.moneyness_coordinate, wgan.G.moneyness_coordinate.cpu()
            )
            torch.testing.assert_close(
                loaded.log_ttm_coordinate, wgan.G.log_ttm_coordinate.cpu()
            )

    def test_loader_reconstructs_pure_cnn_without_text_modules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = Config(
                cuda=False,
                channels=1,
                embedding_dim=5,
                noise_dim=3,
                generator_conditioning_mode=(
                    CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
                ),
                generator_current_input_mode=(
                    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
                ),
                support_mask_mode="raw_joint",
                residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
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
            strike = np.asarray([0.91, 0.94, 0.99, 1.0, 1.01, 1.04, 1.08, 1.2])
            maturity = np.asarray([1, 2, 4, 8, 16, 30, 60, 120])
            wgan = WGAN_GP(config, strike, maturity, embedding_dim=5)
            checkpoint = Path(wgan.save_model()["generator"])
            sample = SimpleNamespace(
                current_surface=np.full((1, 8, 8), 0.2, dtype=np.float32),
                strike_grid=strike,
                maturity_grid_days=maturity,
            )
            loaded, loaded_config, embedding_dim = load_vol_generator(
                checkpoint, sample, torch.device("cpu")
            )
            self.assertEqual(embedding_dim, 5)
            self.assertEqual(
                loaded_config.generator_conditioning_mode,
                CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            )
            self.assertFalse(hasattr(loaded, "text_encoder"))
            self.assertFalse(hasattr(loaded, "encoder_film_layers"))
            torch.testing.assert_close(
                loaded.moneyness_coordinate, wgan.G.moneyness_coordinate.cpu()
            )
            torch.testing.assert_close(
                loaded.log_ttm_coordinate, wgan.G.log_ttm_coordinate.cpu()
            )


if __name__ == "__main__":
    unittest.main()
