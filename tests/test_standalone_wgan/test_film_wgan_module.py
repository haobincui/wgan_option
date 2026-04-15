import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from film_wgan.config import load_train_config  # noqa: E402
from film_wgan.data import create_train_val_bundle, denormalize_tensor, normalize_surface_tensor  # noqa: E402
from film_wgan.inference import normalization_stats_to_tensors  # noqa: E402
from film_wgan.losses import gradient_penalty  # noqa: E402
from film_wgan.models import (  # noqa: E402
    FiLMLayer as FilmFiLMLayer,
    FilmWGANCritic,
    FilmWGANGenerator,
    reconstruct_future_surface,
)
from stylemod_wgan.models import FiLMLayer as StyleModFiLMLayer  # noqa: E402

FILM_TRAIN_CONFIG_PATH = ROOT_DIR / "configs/film_wgan/train_lp_gen128_disc128.yaml"


class TestFiLMLayerInitialization(unittest.TestCase):
    def test_film_layers_start_as_identity_maps(self):
        torch.manual_seed(0)
        x = torch.randn(3, 4, 5, 5)
        conditioning = torch.randn(3, 7)

        for layer_cls in (FilmFiLMLayer, StyleModFiLMLayer):
            layer = layer_cls(7, 4)
            params = layer.projection(conditioning)
            gamma, beta = params.chunk(2, dim=-1)
            out = layer(x, conditioning)

            self.assertTrue(torch.equal(gamma, torch.zeros_like(gamma)))
            self.assertTrue(torch.equal(beta, torch.zeros_like(beta)))
            self.assertTrue(torch.equal(out, x))


class TestFilmWGANInitializationSmoke(unittest.TestCase):
    def test_train_lp_gen128_disc128_first_batch_stays_finite_and_unsaturated(self):
        if not FILM_TRAIN_CONFIG_PATH.exists():
            self.skipTest(f"Missing training config: {FILM_TRAIN_CONFIG_PATH}")

        config = load_train_config(FILM_TRAIN_CONFIG_PATH)
        workbook_path = ROOT_DIR / str(config.data_path)
        if not workbook_path.exists():
            self.skipTest(f"Missing merged-vol workbook required for smoke check: {workbook_path}")

        config = replace(config, data_path=str(workbook_path), cuda=False, num_workers=0)
        bundle = create_train_val_bundle(config)
        normalization = normalization_stats_to_tensors(bundle.normalization_stats, torch.device("cpu"))
        surface_height, surface_width = bundle.surface_shape

        generator = FilmWGANGenerator(
            surface_height=surface_height,
            surface_width=surface_width,
            embedding_dim=bundle.embedding_dim,
            noise_dim=config.noise_dim,
            base_channels=config.gen_base_channels,
            res_blocks=config.gen_res_blocks,
            text_hidden_dim=config.text_hidden_dim,
            text_out_dim=config.text_out_dim,
            fusion_hidden_dim=config.fusion_hidden_dim,
        )
        critic = FilmWGANCritic(
            surface_height=surface_height,
            surface_width=surface_width,
            embedding_dim=bundle.embedding_dim,
            base_channels=config.disc_base_channels,
            res_blocks=config.disc_res_blocks,
            text_hidden_dim=config.text_hidden_dim,
            text_out_dim=config.text_out_dim,
            fusion_hidden_dim=config.fusion_hidden_dim,
        )

        current_features, text_features, _real_delta_norm, current_flat, target_flat = next(iter(bundle.train_loader))
        batch_size = min(8, current_features.size(0))
        current_features = current_features[:batch_size]
        text_features = text_features[:batch_size]
        current_flat = current_flat[:batch_size]
        target_flat = target_flat[:batch_size]
        noise = torch.randn(batch_size, int(config.noise_dim), dtype=torch.float32)

        with torch.no_grad():
            encoded_text = generator.text_encoder(text_features)
            gamma, beta = generator.film1.projection(encoded_text).chunk(2, dim=-1)
            fake_delta_norm = generator(current_features, text_features, noise=noise)
            fake_delta = denormalize_tensor(fake_delta_norm, normalization.delta_mean, normalization.delta_std)
            fake_future_flat = reconstruct_future_surface(current_flat, fake_delta)
            fake_future_surface = normalize_surface_tensor(
                fake_future_flat,
                normalization.current_log_mean,
                normalization.current_log_std,
            ).view(batch_size, 1, surface_height, surface_width)
            real_future_surface = normalize_surface_tensor(
                target_flat,
                normalization.current_log_mean,
                normalization.current_log_std,
            ).view(batch_size, 1, surface_height, surface_width)

        gp = gradient_penalty(
            critic=critic,
            real_future_surface=real_future_surface,
            fake_future_surface=fake_future_surface,
            current_surface=current_features,
            text_embedding=text_features,
            lambda_gp=10.0,
        )

        frac_floor = (fake_future_flat <= 1.00001e-4).float().mean().item()
        frac_ceil = (fake_future_flat >= 4.99999).float().mean().item()

        self.assertTrue(torch.equal(gamma, torch.zeros_like(gamma)))
        self.assertTrue(torch.equal(beta, torch.zeros_like(beta)))
        self.assertTrue(torch.isfinite(fake_delta_norm).all())
        self.assertTrue(torch.isfinite(fake_delta).all())
        self.assertTrue(torch.isfinite(fake_future_flat).all())
        self.assertTrue(torch.isfinite(gp))
        self.assertLess(frac_floor + frac_ceil, 1.0)

