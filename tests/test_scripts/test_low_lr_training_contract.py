"""Low-learning-rate lineage and scheduler safety contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wgan_option.config import Config  # noqa: E402
from wgan_option.models.common import IDENTITY_RESIDUAL_OUTPUT_MODE  # noqa: E402
from wgan_option.models.gan_model import WGAN_GP  # noqa: E402
from wgan_option.models.vol_regressor import VolSurfaceRegressor  # noqa: E402
from wgan_option.train_vol_regression_xlsx import (  # noqa: E402
    VolSurfaceRegressionTrainer,
)
from wgan_option.utils.merged_xlsx_types import VolSurfaceXlsxBundle  # noqa: E402


LR_PROFILE_SHA = "b" * 64


def _loader() -> DataLoader:
    current = torch.full((2, 1, 16, 16), 0.20)
    target = torch.full((2, 1, 16, 16), 0.21)
    text = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    weight = torch.ones(2)
    key = torch.tensor([101, 102], dtype=torch.int64)
    mask = torch.ones((2, 1, 16, 16))
    return DataLoader(
        TensorDataset(current, text, target, weight, key, mask),
        batch_size=2,
        shuffle=False,
    )


def _config(root: Path, *, profile: str = "micro") -> Config:
    base_channels = 1 if profile == "micro" else 12
    text_hidden = 3 if profile == "micro" else 96
    text_out = 2 if profile == "micro" else 48
    hidden = 4 if profile == "micro" else 384
    disc_hidden = 4 if profile == "micro" else 288
    disc_text_hidden = 2 if profile == "micro" else 48
    return Config(
        cuda=False,
        residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
        evaluate_initial_checkpoint=True,
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-6,
        save_every=999,
        use_early_stopping=True,
        early_stopping_patience=1,
        early_stopping_min_epochs=30,
        best_checkpoint_metric="val_recon",
        use_reduce_lr_on_plateau=True,
        lr_scheduler_type="plateau",
        reduce_lr_patience=1,
        reduce_lr_min_lr=1e-7,
        validation_mc_samples=1,
        discriminator_iter=1,
        noise_dim=2,
        gen_base_channels=base_channels,
        disc_base_channels=base_channels,
        gen_text_hidden_dim=text_hidden,
        gen_text_out_dim=text_out,
        disc_text_hidden_dim=disc_text_hidden,
        gen_hidden_dim=hidden,
        disc_hidden_dim=disc_hidden,
        use_calendar_constraint=False,
        use_butterfly_constraint=False,
        use_smooth_constraint=False,
        news_first_capacity_profile=profile,
        news_first_capacity_profile_sha256="a" * 64,
        news_first_lr_profile="lr_1e-6",
        news_first_lr_profile_sha256=LR_PROFILE_SHA,
        models_path=str(root / "models"),
        outputs_path=str(root / "models"),
        samples_path=str(root / "samples"),
        metrics_path=str(root / "metrics"),
    )


def _bundle(loader: DataLoader) -> VolSurfaceXlsxBundle:
    return VolSurfaceXlsxBundle(
        train_loader=loader,
        val_loader=loader,
        strike_grid=np.linspace(0.97, 1.03, 16, dtype=np.float32),
        maturity_grid_days=np.linspace(7.0, 38.0, 16, dtype=np.float32),
        embedding_dim=2,
        train_samples=2,
        val_samples=2,
        timestamps=[],
        train_timestamps=[],
        val_timestamps=[],
        all_items=[],
        train_items=[],
        val_items=[],
    )


def _state_changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(before[name], value.detach().cpu())
        for name, value in module.state_dict().items()
    )


class TestLowLearningRateContract(unittest.TestCase):
    def test_config_defaults_keep_old_checkpoint_payloads_readable(self):
        current = asdict(Config(cuda=False))
        current.pop("news_first_lr_profile")
        current.pop("news_first_lr_profile_sha256")
        current.pop("lr_warmup_epochs")
        current.pop("lr_warmup_start_factor")
        restored = Config(**current)
        self.assertEqual(restored.news_first_lr_profile, "default")
        self.assertEqual(restored.news_first_lr_profile_sha256, "")
        self.assertEqual(restored.lr_warmup_epochs, 0)
        self.assertEqual(restored.lr_warmup_start_factor, 0.1)
        for learning_rate in (1e-6, 1e-5, 1e-4):
            config = Config(
                cuda=False,
                learning_rate=learning_rate,
                reduce_lr_min_lr=learning_rate / 10.0,
            )
            self.assertEqual(config.learning_rate, learning_rate)
            self.assertEqual(config.reduce_lr_min_lr, learning_rate / 10.0)

    def test_lr_warmup_config_fails_closed_for_invalid_contracts(self):
        with self.assertRaisesRegex(ValueError, "lr_warmup_epochs"):
            Config(cuda=False, num_epochs=3, lr_warmup_epochs=4)
        with self.assertRaisesRegex(ValueError, "lr_warmup_start_factor"):
            Config(cuda=False, lr_warmup_start_factor=0.0)
        with self.assertRaisesRegex(ValueError, "not cosine"):
            Config(
                cuda=False,
                num_epochs=3,
                lr_warmup_epochs=2,
                lr_scheduler_type="cosine",
            )

    def test_wgan_linear_lr_warmup_reaches_target_before_plateau_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root)
            config.num_epochs = 3
            config.lr_warmup_epochs = 2
            config.lr_warmup_start_factor = 0.1
            config.use_early_stopping = False
            loader = _loader()
            model = WGAN_GP(
                config=config,
                strike_grid=np.linspace(0.97, 1.03, 16, dtype=np.float32),
                maturity_grid_days=np.linspace(7.0, 38.0, 16, dtype=np.float32),
                embedding_dim=2,
            )
            with (
                patch.object(model, "_save_loss_curves"),
                patch.object(
                    model,
                    "_step_plateau_scheduler",
                    wraps=model._step_plateau_scheduler,
                ) as scheduler_step,
            ):
                model.train(loader, loader)

            rows = json.loads((root / "metrics" / "training_metrics.json").read_text())
            self.assertEqual([row["epoch"] for row in rows], [0, 1, 2, 3])
            self.assertEqual(
                [row["g_lr"] for row in rows],
                [1e-7, 1e-7, 1e-6, 1e-6],
            )
            self.assertEqual(
                [row["d_lr"] for row in rows],
                [1e-7, 1e-7, 1e-6, 1e-6],
            )
            self.assertEqual(scheduler_step.call_count, 2)
            final = torch.load(
                root / "models" / "generator.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(final["lr_warmup_epochs"], 2)
            self.assertEqual(final["lr_warmup_start_factor"], 0.1)
            self.assertEqual(final["generator_warmup_start_learning_rate"], 1e-7)
            self.assertEqual(final["discriminator_warmup_start_learning_rate"], 1e-7)

    def test_scheduler_floor_above_initial_lr_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root)
            config.reduce_lr_min_lr = 1e-5
            loader = _loader()
            trainer = VolSurfaceRegressionTrainer(config)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                trainer._create_plateau_scheduler(optimizer)

            wgan = WGAN_GP(
                config=config,
                strike_grid=np.linspace(0.97, 1.03, 8, dtype=np.float32),
                maturity_grid_days=np.linspace(7.0, 38.0, 8, dtype=np.float32),
                embedding_dim=2,
            )
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                wgan.train(loader, loader)

    def test_regression_updates_at_1e6_and_persists_exact_lr_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root)
            loader = _loader()
            trainer = VolSurfaceRegressionTrainer(config)
            trainer.bundle = _bundle(loader)
            trainer.run_dir = root
            trainer.model = VolSurfaceRegressor(
                channels=1,
                embedding_dim=2,
                surface_height=16,
                surface_width=16,
                base_channels=1,
                text_hidden_dim=3,
                text_out_dim=2,
                hidden_dim=4,
                residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
            )
            trainer.optimizer = torch.optim.Adam(
                trainer.model.parameters(),
                lr=config.learning_rate,
            )
            trainer.strike_grid = torch.tensor(
                trainer.bundle.strike_grid,
                dtype=torch.float32,
            )
            trainer.tau_years = torch.tensor(
                trainer.bundle.maturity_grid_days / 365.0,
                dtype=torch.float32,
            )
            before = {
                name: value.detach().cpu().clone()
                for name, value in trainer.model.state_dict().items()
            }
            with patch.object(trainer, "_save_loss_curves"):
                trainer._train_impl()

            self.assertTrue(_state_changed(before, trainer.model))
            initial = json.loads(
                (root / "metrics" / "initial_checkpoint.json").read_text()
            )
            learned = json.loads(
                (root / "metrics" / "best_learned_checkpoint.json").read_text()
            )
            final = torch.load(
                root / "models" / "vol_regressor.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(learned["best_learned_epoch_ge_1"], 1)
            for payload in (initial, learned, final):
                self.assertEqual(payload["seed"], config.seed)
                self.assertEqual(payload["lr_profile"], "lr_1e-6")
                self.assertEqual(payload["lr_profile_sha256"], LR_PROFILE_SHA)
                self.assertEqual(payload["initial_learning_rate"], 1e-6)
                self.assertEqual(payload["scheduler_min_lr"], 1e-7)
                self.assertGreaterEqual(
                    min(row["lr"] for row in payload["lr_trace"]), 1e-7
                )
                self.assertLessEqual(
                    max(row["lr"] for row in payload["lr_trace"]), 1e-6
                )
            self.assertEqual(initial["lr_trace"], [{"epoch": 0, "lr": 1e-6}])
            self.assertEqual([row["epoch"] for row in final["lr_trace"]], [0, 1])

    def test_wgan_updates_at_1e6_and_persists_g_d_lr_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root)
            loader = _loader()
            model = WGAN_GP(
                config=config,
                strike_grid=np.linspace(0.97, 1.03, 16, dtype=np.float32),
                maturity_grid_days=np.linspace(7.0, 38.0, 16, dtype=np.float32),
                embedding_dim=2,
            )
            g_before = {
                name: value.detach().cpu().clone()
                for name, value in model.G.state_dict().items()
            }
            d_before = {
                name: value.detach().cpu().clone()
                for name, value in model.D.state_dict().items()
            }
            with patch.object(model, "_save_loss_curves"):
                model.train(loader, loader)

            self.assertTrue(_state_changed(g_before, model.G))
            self.assertTrue(_state_changed(d_before, model.D))
            learned = json.loads(
                (root / "metrics" / "best_learned_checkpoint.json").read_text()
            )
            final = torch.load(
                root / "models" / "generator.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(learned["best_learned_epoch_ge_1"], 1)
            for payload in (learned, final):
                self.assertEqual(payload["lr_profile"], "lr_1e-6")
                self.assertEqual(payload["lr_profile_sha256"], LR_PROFILE_SHA)
                self.assertEqual(payload["initial_learning_rate"], 1e-6)
                self.assertEqual(payload["generator_initial_learning_rate"], 1e-6)
                self.assertEqual(payload["discriminator_initial_learning_rate"], 1e-6)
                self.assertEqual(payload["scheduler_min_lr"], 1e-7)
                self.assertTrue(payload["generator_lr_trace"])
                self.assertTrue(payload["discriminator_lr_trace"])
                self.assertGreaterEqual(
                    min(row["lr"] for row in payload["generator_lr_trace"]),
                    1e-7,
                )
                self.assertLessEqual(
                    max(row["lr"] for row in payload["generator_lr_trace"]),
                    1e-6,
                )

    def test_early_stopping_cannot_trigger_before_epoch_30(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root)
            config.num_epochs = 31
            loader = _loader()
            trainer = VolSurfaceRegressionTrainer(config)
            trainer.bundle = _bundle(loader)
            trainer.run_dir = root
            trainer.model = VolSurfaceRegressor(
                channels=1,
                embedding_dim=2,
                surface_height=16,
                surface_width=16,
                base_channels=1,
                text_hidden_dim=3,
                text_out_dim=2,
                hidden_dim=4,
                residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
            )
            trainer.optimizer = torch.optim.Adam(
                trainer.model.parameters(),
                lr=config.learning_rate,
            )

            def run_epoch(_loader, *, train: bool, epoch: int):
                if train:
                    return {"train_total": 0.2, "train_recon": 0.2}
                value = 0.1 + epoch * 0.01
                return {
                    "val_recon": value,
                    "val_current_recon": 0.1,
                    "val_baseline_gap": value - 0.1,
                    "val_hybrid_score": value,
                    "val_calendar": 0.0,
                    "val_butterfly": 0.0,
                    "val_delta_shrink": 0.0,
                }

            with (
                patch.object(trainer, "_run_epoch", side_effect=run_epoch),
                patch.object(trainer, "_save_loss_curves"),
            ):
                trainer._train_impl()
            rows = json.loads((root / "metrics" / "training_metrics.json").read_text())
            self.assertEqual(rows[-1]["epoch"], 30)
            self.assertEqual(
                json.loads(
                    (root / "metrics" / "best_learned_checkpoint.json").read_text()
                )["best_learned_epoch_ge_1"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
