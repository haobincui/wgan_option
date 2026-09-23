"""Training-core contracts used by the news-first capacity sweep."""

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

from wgan_option.config import Config, config_to_dict  # noqa: E402
from wgan_option.models.common import IDENTITY_RESIDUAL_OUTPUT_MODE  # noqa: E402
from wgan_option.models.gan_model import WGAN_GP  # noqa: E402
from wgan_option.models.vol_regressor import VolSurfaceRegressor  # noqa: E402
from wgan_option.train_vol_regression_xlsx import (  # noqa: E402
    VolSurfaceRegressionTrainer,
)
from wgan_option.utils.merged_xlsx_types import VolSurfaceXlsxBundle  # noqa: E402


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
        num_epochs=5,
        batch_size=1,
        learning_rate=1e-4,
        save_every=999,
        use_early_stopping=True,
        early_stopping_patience=1,
        early_stopping_min_epochs=3,
        best_checkpoint_metric="val_recon",
        use_reduce_lr_on_plateau=True,
        lr_scheduler_type="plateau",
        reduce_lr_patience=1,
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
        news_first_capacity_profile="micro",
        news_first_capacity_profile_sha256="a" * 64,
        models_path=str(root / "models"),
        outputs_path=str(root / "models"),
        samples_path=str(root / "samples"),
        metrics_path=str(root / "metrics"),
    )


def _validation_stats(value: float) -> dict[str, float]:
    return {
        "val_recon": value,
        "val_current_recon": 0.1,
        "val_baseline_gap": value - 0.1,
        "val_hybrid_score": value + 2.0 * max(0.0, value - 0.1),
        "val_calendar": 0.0,
        "val_butterfly": 0.0,
        "val_delta_shrink": 0.0,
    }


class TestCapacityTrainingCore(unittest.TestCase):
    def test_capacity_config_defaults_are_backward_compatible(self):
        config = Config(cuda=False)
        self.assertEqual(config.news_first_capacity_profile, "default")
        self.assertEqual(config.news_first_capacity_profile_sha256, "")
        self.assertEqual(config.early_stopping_min_epochs, 0)
        self.assertEqual(
            config_to_dict(config)["news_first_capacity_profile"],
            "default",
        )
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            Config(cuda=False, early_stopping_min_epochs=-1)

    def test_regression_tracks_epoch_zero_and_best_learned_independently(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root)
            loader = _loader()
            bundle = VolSurfaceXlsxBundle(
                train_loader=loader,
                val_loader=loader,
                strike_grid=np.linspace(0.9, 1.1, 8, dtype=np.float32),
                maturity_grid_days=np.linspace(7.0, 90.0, 8, dtype=np.float32),
                embedding_dim=2,
                train_samples=1,
                val_samples=1,
                timestamps=[],
                train_timestamps=[],
                val_timestamps=[],
                all_items=[],
                train_items=[],
                val_items=[],
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
                lr=config.learning_rate,
            )

            def run_epoch(_loader, *, train: bool, epoch: int):
                if train:
                    return {"train_total": 0.2, "train_recon": 0.2}
                return _validation_stats(0.1 + 0.1 * epoch)

            original_scheduler_step = trainer._step_plateau_scheduler
            with (
                patch.object(trainer, "_run_epoch", side_effect=run_epoch),
                patch.object(
                    trainer,
                    "_step_plateau_scheduler",
                    wraps=original_scheduler_step,
                ) as scheduler_step,
                patch.object(trainer, "_save_loss_curves"),
            ):
                trainer._train_impl()

            rows = json.loads(
                (root / "metrics" / "training_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual([row["epoch"] for row in rows], [0, 1, 2, 3])
            self.assertEqual(
                [
                    round(call.kwargs["metric_value"], 6)
                    for call in scheduler_step.call_args_list
                ],
                [0.1, 0.2, 0.3, 0.4],
            )

            selected = json.loads(
                (root / "metrics" / "best_checkpoint.json").read_text(encoding="utf-8")
            )
            initial = json.loads(
                (root / "metrics" / "initial_checkpoint.json").read_text(
                    encoding="utf-8"
                )
            )
            learned = json.loads(
                (root / "metrics" / "best_learned_checkpoint.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(selected["best_epoch"], 0)
            self.assertEqual(selected["selection_scope"], "baseline_inclusive")
            self.assertEqual(initial["initial_epoch"], 0)
            self.assertEqual(initial["selection_scope"], "initial_epoch0")
            self.assertEqual(initial["capacity_profile"], "micro")
            self.assertEqual(learned["best_epoch"], 1)
            self.assertEqual(learned["best_learned_epoch_ge_1"], 1)
            self.assertEqual(learned["selection_scope"], "trained_epochs_only")
            self.assertEqual(learned["capacity_profile"], "micro")
            self.assertEqual(learned["capacity_profile_sha256"], "a" * 64)
            self.assertTrue((root / "models" / "vol_regressor_best.pt").is_file())
            self.assertTrue(
                (root / "models" / "vol_regressor_initial_epoch0.pt").is_file()
            )
            self.assertTrue(
                (root / "models" / "vol_regressor_best_learned.pt").is_file()
            )
            learned_model = torch.load(
                root / "models" / "vol_regressor_best_learned.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(learned_model["capacity_profile"], "micro")
            self.assertEqual(learned_model["capacity_profile_sha256"], "a" * 64)

    def test_wgan_matches_regression_checkpoint_and_scheduler_contract(self):
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
            validation_values = iter((0.1, 0.2, 0.3, 0.4))

            def evaluate(_loader):
                return _validation_stats(next(validation_values))

            original_scheduler_step = model._step_plateau_scheduler
            with (
                patch.object(model, "_evaluate", side_effect=evaluate),
                patch.object(
                    model,
                    "_discriminator_step",
                    return_value={"d_total": 0.0},
                ),
                patch.object(
                    model,
                    "_generator_step",
                    return_value={"g_total": 0.0},
                ),
                patch.object(
                    model,
                    "_step_plateau_scheduler",
                    wraps=original_scheduler_step,
                ) as scheduler_step,
                patch.object(model, "_save_loss_curves"),
            ):
                model.train(loader, loader)

            rows = json.loads(
                (root / "metrics" / "training_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual([row["epoch"] for row in rows], [0, 1, 2, 3])
            self.assertEqual(
                [call.kwargs["metric_value"] for call in scheduler_step.call_args_list],
                [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4],
            )

            selected = json.loads(
                (root / "metrics" / "best_checkpoint.json").read_text(encoding="utf-8")
            )
            initial = json.loads(
                (root / "metrics" / "initial_checkpoint.json").read_text(
                    encoding="utf-8"
                )
            )
            learned = json.loads(
                (root / "metrics" / "best_learned_checkpoint.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(selected["best_epoch"], 0)
            self.assertEqual(initial["initial_epoch"], 0)
            self.assertEqual(initial["selection_scope"], "initial_epoch0")
            self.assertEqual(initial["capacity_profile_sha256"], "a" * 64)
            self.assertEqual(learned["best_epoch"], 1)
            self.assertEqual(learned["best_learned_epoch_ge_1"], 1)
            self.assertEqual(learned["capacity_profile"], "micro")
            self.assertTrue((root / "models" / "generator_best.pt").is_file())
            self.assertTrue((root / "models" / "discriminator_best.pt").is_file())
            self.assertTrue((root / "models" / "generator_initial_epoch0.pt").is_file())
            self.assertTrue(
                (root / "models" / "discriminator_initial_epoch0.pt").is_file()
            )
            self.assertTrue((root / "models" / "generator_best_learned.pt").is_file())
            self.assertTrue(
                (root / "models" / "discriminator_best_learned.pt").is_file()
            )
            learned_generator = torch.load(
                root / "models" / "generator_best_learned.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(learned_generator["capacity_profile"], "micro")
            self.assertEqual(
                learned_generator["capacity_profile_sha256"],
                "a" * 64,
            )


if __name__ == "__main__":
    unittest.main()
