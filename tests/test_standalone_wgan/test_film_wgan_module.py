import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from film_wgan.config import load_train_config  # noqa: E402
from film_wgan.data import create_train_val_bundle, denormalize_tensor, normalize_surface_tensor  # noqa: E402
from film_wgan.inference import normalization_stats_to_tensors  # noqa: E402
from film_wgan.losses import (  # noqa: E402
    build_reconstruction_weight_template,
    gradient_penalty,
    weighted_surface_mae,
)
from film_wgan.models import (  # noqa: E402
    FiLMLayer as FilmFiLMLayer,
    FilmWGANCritic,
    FilmWGANGenerator,
    reconstruct_future_surface,
)
from film_wgan.trainer import FilmWGANTrainer  # noqa: E402
from stylemod_wgan.models import FiLMLayer as StyleModFiLMLayer  # noqa: E402

FILM_TRAIN_CONFIG_PATH = ROOT_DIR / "configs/film_wgan/train_lp_gen128_disc128.yaml"


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return path


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


class TestFilmWGANLossWeighting(unittest.TestCase):
    @staticmethod
    def _grids() -> tuple[torch.Tensor, torch.Tensor]:
        strike_grid = torch.tensor([0.80, 0.92, 1.00, 1.08, 1.20], dtype=torch.float32)
        maturity_days_grid = torch.tensor([7.0, 30.0, 120.0], dtype=torch.float32)
        return strike_grid, maturity_days_grid

    def test_uniform_weight_template_is_all_ones(self):
        strike_grid, maturity_days_grid = self._grids()
        weights = build_reconstruction_weight_template(
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
            mode="uniform",
            atm_range=0.08,
            short_end_max_days=90.0,
            atm_multiplier=3.0,
        )

        self.assertEqual(tuple(weights.shape), (3, 5))
        self.assertTrue(torch.allclose(weights, torch.ones_like(weights)))

    def test_short_atm_band_weights_emphasize_short_end_atm_and_are_mean_normalized(self):
        strike_grid, maturity_days_grid = self._grids()
        weights = build_reconstruction_weight_template(
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
            mode="short_atm_band",
            atm_range=0.08,
            short_end_max_days=30.0,
            atm_multiplier=3.0,
        )

        self.assertAlmostEqual(float(weights.mean().item()), 1.0, places=6)
        self.assertEqual(int(torch.count_nonzero(weights > 1.0).item()), 6)
        self.assertTrue(torch.all(weights[:2, 1:4] > 1.0))
        self.assertTrue(torch.all(weights[2, :] < 1.0))
        self.assertTrue(torch.all(weights[:, 0] < 1.0))
        self.assertTrue(torch.all(weights[:, 4] < 1.0))

    def test_weighted_mae_matches_plain_l1_under_uniform_weights(self):
        predicted = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float32)
        target = torch.tensor([[0.0, 1.0], [4.0, 7.0]], dtype=torch.float32)
        weights = torch.ones_like(predicted)

        weighted = weighted_surface_mae(predicted, target, weights)
        plain = torch.nn.functional.l1_loss(predicted, target)

        self.assertAlmostEqual(float(weighted.item()), float(plain.item()), places=6)

    def test_weighted_mae_emphasizes_band_local_errors(self):
        strike_grid, maturity_days_grid = self._grids()
        weights = build_reconstruction_weight_template(
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
            mode="short_atm_band",
            atm_range=0.08,
            short_end_max_days=30.0,
            atm_multiplier=3.0,
        )
        predicted = torch.zeros((3, 5), dtype=torch.float32)
        target = torch.zeros((3, 5), dtype=torch.float32)
        target[0, 2] = 1.0

        weighted = weighted_surface_mae(predicted, target, weights)
        plain = torch.nn.functional.l1_loss(predicted, target)

        self.assertGreater(float(weighted.item()), float(plain.item()))

    def test_weighted_mae_deemphasizes_out_of_band_errors(self):
        strike_grid, maturity_days_grid = self._grids()
        weights = build_reconstruction_weight_template(
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
            mode="short_atm_band",
            atm_range=0.08,
            short_end_max_days=30.0,
            atm_multiplier=3.0,
        )
        predicted = torch.zeros((3, 5), dtype=torch.float32)
        target = torch.zeros((3, 5), dtype=torch.float32)
        target[2, 4] = 1.0

        weighted = weighted_surface_mae(predicted, target, weights)
        plain = torch.nn.functional.l1_loss(predicted, target)

        self.assertLess(float(weighted.item()), float(plain.item()))


class TestFilmWGANConfiguration(unittest.TestCase):
    def test_active_config_enables_short_end_atm_weighting(self):
        config = load_train_config(FILM_TRAIN_CONFIG_PATH)

        self.assertEqual(config.recon_weight_mode, "short_atm_band")
        self.assertAlmostEqual(float(config.recon_atm_range), 0.08, places=6)
        self.assertAlmostEqual(float(config.recon_atm_short_end_max_days), 90.0, places=6)
        self.assertAlmostEqual(float(config.recon_atm_multiplier), 3.0, places=6)


class TestFilmWGANTrainerMetrics(unittest.TestCase):
    def test_training_metrics_json_includes_short_end_atm_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "training" / "film_wgan"
            config_path = _write_yaml(
                Path(tmpdir) / "film_metrics.yaml",
                {
                    "training": {
                        "data_path": str(Path(tmpdir) / "dummy.xlsx"),
                        "sheet_name": "gan_input_ready",
                        "text_embedding_mode": "lp",
                        "train_ratio": 0.67,
                        "min_samples_for_training": 2,
                        "noise_dim": 4,
                        "gen_base_channels": 8,
                        "disc_base_channels": 8,
                        "gen_res_blocks": 1,
                        "disc_res_blocks": 1,
                        "text_hidden_dim": 16,
                        "text_out_dim": 8,
                        "fusion_hidden_dim": 32,
                        "num_epochs": 1,
                        "batch_size": 1,
                        "eval_mc_samples": 2,
                        "cuda": False,
                        "num_workers": 0,
                        "output_root": str(output_root),
                        "save_every": 20,
                    }
                },
            )

            trainer = FilmWGANTrainer(load_train_config(config_path), config_path=str(config_path))

            def fake_setup():
                trainer.checkpoints_dir.mkdir(parents=True, exist_ok=True)
                trainer.metrics_dir.mkdir(parents=True, exist_ok=True)
                trainer.bundle = type(
                    "Bundle",
                    (),
                    {
                        "train_loader": [
                            (
                                torch.zeros((1, 1, 4, 4), dtype=torch.float32),
                                torch.zeros((1, 3), dtype=torch.float32),
                                torch.zeros((1, 16), dtype=torch.float32),
                                torch.ones((1, 16), dtype=torch.float32),
                                torch.ones((1, 16), dtype=torch.float32),
                            )
                        ]
                    },
                )()

            eval_row = {
                "val_mae": 0.10,
                "val_rmse": 0.12,
                "val_current_mae": 0.11,
                "val_current_rmse": 0.13,
                "val_short_atm_weighted_mae": 0.09,
                "val_current_short_atm_weighted_mae": 0.12,
                "val_short_atm_mae_gap_vs_current": -0.03,
                "val_mae_gap_vs_current": -0.01,
                "val_win_rate_vs_current": 1.0,
                "val_generated_current_mae": 0.02,
                "val_real_current_mae": 0.11,
                "val_calendar": 0.0,
                "val_butterfly": 0.0,
                "val_penalty_mean": 0.0,
                "val_penalty_std": 0.0,
                "val_weight_entropy": 1.0,
            }

            with patch.object(trainer, "setup", side_effect=fake_setup), \
                patch.object(
                    trainer,
                    "_discriminator_step",
                    return_value={"d_total": 0.0, "d_real": 0.0, "d_fake": 0.0, "gp": 0.0},
                ), \
                patch.object(
                    trainer,
                    "_generator_step",
                    return_value={
                        "g_total": 0.0,
                        "g_adv": 0.0,
                        "g_calendar": 0.0,
                        "g_butterfly": 0.0,
                        "g_smooth": 0.0,
                        "g_recon": 0.0,
                        "g_recon_weighted": 0.0,
                    },
                ), \
                patch.object(trainer, "_evaluate", return_value=eval_row), \
                patch.object(trainer, "_save_loss_curves", return_value=None), \
                patch.object(trainer, "_checkpoint_payload", return_value={"state": "ok"}):
                trainer.train()

            run_dir = next(path for path in output_root.iterdir() if path.is_dir())
            metrics_rows = json.loads((run_dir / "metrics" / "training_metrics.json").read_text(encoding="utf-8"))
            row = metrics_rows[-1]
            self.assertIn("g_recon_weighted", row)
            self.assertIn("val_short_atm_weighted_mae", row)
            self.assertIn("val_current_short_atm_weighted_mae", row)
            self.assertIn("val_short_atm_mae_gap_vs_current", row)
            self.assertAlmostEqual(float(row["val_short_atm_mae_gap_vs_current"]), -0.03, places=6)
            self.assertTrue((run_dir / "checkpoints" / "film_wgan_best.pt").exists())


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
