"""Tests for optional epoch-zero validation and checkpoint selection."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
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


class _ForbiddenLoader:
    def __init__(self) -> None:
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        raise AssertionError("test_loader must not be used during training")


def _loader() -> DataLoader:
    current = torch.full((1, 1, 8, 8), 0.2)
    target = torch.full((1, 1, 8, 8), 0.3)
    text = torch.tensor([[1.0, 2.0]])
    weight = torch.ones(1)
    key = torch.tensor([101], dtype=torch.int64)
    return DataLoader(
        TensorDataset(current, text, target, weight, key),
        batch_size=1,
        shuffle=False,
    )


def _config(root: Path) -> Config:
    return Config(
        cuda=False,
        residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
        evaluate_initial_checkpoint=True,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.0,
        save_every=999,
        use_early_stopping=True,
        early_stopping_patience=1,
        best_checkpoint_metric="val_hybrid_score",
        validation_mc_samples=1,
        discriminator_iter=1,
        noise_dim=2,
        gen_base_channels=1,
        disc_base_channels=1,
        gen_text_hidden_dim=3,
        gen_text_out_dim=2,
        disc_text_hidden_dim=2,
        gen_hidden_dim=4,
        disc_hidden_dim=4,
        models_path=str(root / "models"),
        outputs_path=str(root / "models"),
        samples_path=str(root / "samples"),
        metrics_path=str(root / "metrics"),
    )


class TestInitialCheckpointEvaluation(unittest.TestCase):
    def test_config_default_preserves_historical_training_flow(self):
        self.assertFalse(Config(cuda=False).evaluate_initial_checkpoint)

    def test_wgan_identity_initialization_can_remain_best_at_epoch_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root)
            loader = _loader()
            model = WGAN_GP(
                config=config,
                strike_grid=np.linspace(0.9, 1.1, 8, dtype=np.float32),
                maturity_grid_days=np.linspace(7.0, 90.0, 8, dtype=np.float32),
                embedding_dim=2,
            )
            self.assertEqual(
                int(torch.count_nonzero(model.G.fusion[-1].weight)),
                0,
            )

            events: list[str] = []
            original_evaluate = model._evaluate

            def tracked_evaluate(val_loader):
                events.append("validation")
                return original_evaluate(val_loader)

            def discriminator_step(*args, **kwargs):
                del args, kwargs
                events.append("training")
                return {"d_total": 0.0}

            with (
                patch.object(model, "_evaluate", side_effect=tracked_evaluate),
                patch.object(
                    model,
                    "_discriminator_step",
                    side_effect=discriminator_step,
                ),
                patch.object(model, "_generator_step", return_value={"g_total": 0.0}),
                patch.object(model, "_save_loss_curves"),
            ):
                model.train(loader, loader)

            self.assertEqual(events, ["validation", "training", "validation"])
            rows = json.loads(
                (root / "metrics" / "training_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual([row["epoch"] for row in rows], [0, 1])
            self.assertAlmostEqual(rows[0]["val_recon"], 0.1, places=6)
            self.assertAlmostEqual(
                rows[0]["val_recon"],
                rows[0]["val_current_recon"],
                places=6,
            )
            best = json.loads(
                (root / "metrics" / "best_checkpoint.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(best["best_epoch"], 0)
            self.assertTrue((root / "models" / "generator_best.pt").is_file())
            self.assertTrue((root / "models" / "discriminator_best.pt").is_file())

    def test_regression_uses_only_validation_for_epoch_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root)
            loader = _loader()
            forbidden_test_loader = _ForbiddenLoader()
            bundle = VolSurfaceXlsxBundle(
                train_loader=loader,
                val_loader=loader,
                test_loader=forbidden_test_loader,  # type: ignore[arg-type]
                strike_grid=np.linspace(0.9, 1.1, 8, dtype=np.float32),
                maturity_grid_days=np.linspace(7.0, 90.0, 8, dtype=np.float32),
                embedding_dim=2,
                train_samples=1,
                val_samples=1,
                test_samples=1,
                timestamps=[],
                train_timestamps=[],
                val_timestamps=[],
                all_items=[],
                train_items=[],
                val_items=[],
                test_items=[],
            )
            trainer = VolSurfaceRegressionTrainer(config)
            trainer.bundle = bundle
            trainer.run_dir = root
            trainer.model = VolSurfaceRegressor(
                channels=1,
                embedding_dim=2,
                surface_height=8,
                surface_width=8,
                base_channels=1,
                text_hidden_dim=3,
                text_out_dim=2,
                hidden_dim=4,
                residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
            )
            trainer.optimizer = torch.optim.Adam(
                trainer.model.parameters(),
                lr=0.0,
            )
            trainer.strike_grid = torch.linspace(0.9, 1.1, 8)
            trainer.tau_years = torch.linspace(7.0, 90.0, 8) / 365.0
            self.assertEqual(
                int(torch.count_nonzero(trainer.model.fusion[-1].weight)),
                0,
            )

            calls: list[tuple[bool, int]] = []
            original_run_epoch = trainer._run_epoch

            def tracked_run_epoch(epoch_loader, *, train: bool, epoch: int):
                calls.append((train, epoch))
                return original_run_epoch(epoch_loader, train=train, epoch=epoch)

            with (
                patch.object(trainer, "_run_epoch", side_effect=tracked_run_epoch),
                patch.object(trainer, "_save_loss_curves"),
            ):
                trainer._train_impl()

            self.assertEqual(calls, [(False, 0), (True, 1), (False, 1)])
            self.assertEqual(forbidden_test_loader.iterations, 0)
            rows = json.loads(
                (root / "metrics" / "training_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual([row["epoch"] for row in rows], [0, 1])
            self.assertAlmostEqual(rows[0]["val_recon"], 0.1, places=6)
            self.assertAlmostEqual(
                rows[0]["val_recon"],
                rows[0]["val_current_recon"],
                places=6,
            )
            best = json.loads(
                (root / "metrics" / "best_checkpoint.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(best["best_epoch"], 0)
            self.assertTrue((root / "models" / "vol_regressor_best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
