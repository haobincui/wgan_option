"""Tests for explicit news-first splits and pair-balanced training semantics."""

from __future__ import annotations

import sys
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
from wgan_option.models.gan_model import WGAN_GP  # noqa: E402
from wgan_option.train_vol_regression_xlsx import VolSurfaceRegressionTrainer  # noqa: E402
from wgan_option.utils.merged_xlsx_types import VolSurfaceSample  # noqa: E402
from wgan_option.utils.news_first_dataloaders import (  # noqa: E402
    create_news_first_vol_surface_dataloaders,
    select_news_first_vol_splits,
)
from wgan_option.utils.weighted_training import (  # noqa: E402
    stable_noise_for_keys,
    training_weighted_mean,
    weighted_mean,
)


def _sample(
    sample_id: str,
    origin: str,
    pair_id: str,
    session_id: str,
    value: float,
) -> VolSurfaceSample:
    current = np.full((1, 2, 2), value, dtype=np.float32)
    return VolSurfaceSample(
        sample_id=sample_id,
        timestamp=origin,
        current_snapshot_time_utc=origin,
        target_snapshot_time_utc=origin,
        current_surface=current,
        target_surface=current + 0.01,
        text_embedding=np.asarray([value, value + 1.0], dtype=np.float32),
        strike_grid=np.asarray([0.9, 1.1], dtype=np.float32),
        maturity_grid_days=np.asarray([30.0, 60.0], dtype=np.float32),
        surface_shape=(2, 2),
        global_index=999,
        metadata={},
        news_row_id=int(value * 100),
        pair_id=pair_id,
        session_id=session_id,
        effective_origin_utc=origin,
        stable_sample_key=f"stable:{sample_id}",
    )


class _BiasRegressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, current_surface, text_embedding):
        del text_embedding
        return current_surface + self.bias


class TestNewsFirstTrainingCore(unittest.TestCase):
    def _split_rows(self):
        training = [
            _sample("t1", "2023-06-01T12:00:00Z", "p1", "s1", 1.0),
            _sample("t2", "2023-06-01T12:00:00Z", "p1", "s1", 2.0),
            _sample("t3", "2023-06-02T12:00:00Z", "p2", "s2", 3.0),
            _sample("ignored", "2023-08-01T12:00:00Z", "p3", "s3", 4.0),
        ]
        common = [
            _sample("v1", "2023-08-01T12:00:00Z", "pv", "sv", 5.0),
            _sample("e1", "2023-11-01T12:00:00Z", "pe", "se", 6.0),
        ]
        return training, common

    def test_split_is_pair_balanced_and_builder_emits_five_item_batches(self):
        training, common = self._split_rows()
        selection = select_news_first_vol_splits(training, common)

        self.assertEqual([item.sample_id for item in selection.train_items], ["t1", "t2", "t3"])
        self.assertEqual(
            [item.metadata["raw_pair_sample_weight"] for item in selection.train_items],
            [0.5, 0.5, 1.0],
        )
        self.assertEqual(
            [item.metadata["scaled_training_sample_weight"] for item in selection.train_items],
            [0.75, 0.75, 1.5],
        )

        config = Config(data_path="train.xlsx", batch_size=3, num_workers=0, seed=17)
        with patch(
            "wgan_option.utils.news_first_dataloaders.load_vol_surface_samples",
            side_effect=[training, common, training, common],
        ):
            bundle = create_news_first_vol_surface_dataloaders(config, "5m.xlsx")
            repeated_bundle = create_news_first_vol_surface_dataloaders(
                config,
                "5m.xlsx",
            )

        first_batch = next(iter(bundle.train_loader))
        repeated_batch = next(iter(repeated_bundle.train_loader))
        self.assertEqual(len(first_batch), 5)
        torch.testing.assert_close(first_batch[4], repeated_batch[4])
        self.assertEqual(bundle.test_samples, 1)
        self.assertEqual(bundle.split_metadata["train_pairs"], 2)
        self.assertEqual(bundle.split_metadata["train_sessions"], 2)
        self.assertEqual(bundle.split_metadata["validation_sessions"], 1)
        self.assertEqual(bundle.split_metadata["test_sessions"], 1)

    def test_split_rejects_session_leakage_even_when_pairs_are_distinct(self):
        training, common = self._split_rows()
        common[0].session_id = "s1"
        with self.assertRaisesRegex(ValueError, "session_id leakage"):
            select_news_first_vol_splits(training, common)

    def test_split_rejects_missing_session_ids(self):
        training, common = self._split_rows()
        training[0].session_id = ""
        with self.assertRaisesRegex(ValueError, "session_id is required"):
            select_news_first_vol_splits(training, common)

    def test_pair_balanced_objective_is_invariant_to_duplicate_articles(self):
        original_losses = torch.tensor([2.0, 8.0])
        original_scaled_weights = torch.tensor([1.0, 1.0])
        duplicated_losses = torch.tensor([2.0, 2.0, 8.0])
        duplicated_scaled_weights = torch.tensor([0.75, 0.75, 1.5])

        self.assertAlmostEqual(
            float(training_weighted_mean(original_losses, original_scaled_weights)),
            float(training_weighted_mean(duplicated_losses, duplicated_scaled_weights)),
        )
        self.assertAlmostEqual(
            float(weighted_mean(original_losses, torch.tensor([1.0, 1.0]))),
            float(weighted_mean(duplicated_losses, torch.tensor([0.5, 0.5, 1.0]))),
        )

    def test_stable_noise_depends_on_sample_key_not_batch_order(self):
        first = stable_noise_for_keys(
            [11, 22],
            noise_dim=4,
            base_seed=7,
            draw_index=3,
            device=torch.device("cpu"),
        )
        reordered = stable_noise_for_keys(
            [22, 11],
            noise_dim=4,
            base_seed=7,
            draw_index=3,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(first[0], reordered[1])
        torch.testing.assert_close(first[1], reordered[0])

    def test_wgan_constructor_seeds_before_parameter_initialization(self):
        config = Config(
            cuda=False,
            seed=73,
            noise_dim=2,
            gen_base_channels=1,
            disc_base_channels=1,
            gen_text_hidden_dim=3,
            gen_text_out_dim=2,
            disc_text_hidden_dim=2,
            gen_hidden_dim=4,
            disc_hidden_dim=4,
        )
        kwargs = {
            "config": config,
            "strike_grid": np.asarray([0.9, 1.1], dtype=np.float32),
            "maturity_grid_days": np.asarray([30.0, 60.0], dtype=np.float32),
            "embedding_dim": 2,
        }
        first = WGAN_GP(**kwargs)
        first_parameters = {
            name: value.detach().clone() for name, value in first.G.state_dict().items()
        }
        torch.randn(100)
        second = WGAN_GP(**kwargs)

        for name, value in second.G.state_dict().items():
            torch.testing.assert_close(first_parameters[name], value)

    def test_wgan_validation_is_fixed_across_batching_and_order(self):
        config = Config(
            cuda=False,
            seed=31,
            validation_mc_samples=3,
            noise_dim=3,
            gen_base_channels=2,
            disc_base_channels=2,
            gen_text_hidden_dim=4,
            gen_text_out_dim=3,
            disc_text_hidden_dim=3,
            gen_hidden_dim=8,
            disc_hidden_dim=8,
        )
        model = WGAN_GP(
            config=config,
            strike_grid=np.linspace(0.8, 1.2, 8, dtype=np.float32),
            maturity_grid_days=np.linspace(7, 90, 8, dtype=np.float32),
            embedding_dim=2,
        )
        current = torch.stack(
            [torch.full((1, 8, 8), 0.2), torch.full((1, 8, 8), 0.3)]
        )
        text = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = current + 0.01
        weights = torch.tensor([0.5, 1.0])
        keys = torch.tensor([101, 202], dtype=torch.int64)
        loader = DataLoader(
            TensorDataset(current, text, target, weights, keys),
            batch_size=2,
            shuffle=False,
        )
        order = torch.tensor([1, 0])
        reordered_loader = DataLoader(
            TensorDataset(
                current[order],
                text[order],
                target[order],
                weights[order],
                keys[order],
            ),
            batch_size=1,
            shuffle=False,
        )

        expected = model._evaluate(loader)
        actual = model._evaluate(reordered_loader)
        for name in expected:
            self.assertAlmostEqual(expected[name], actual[name], places=6)

    def test_regression_epoch_accepts_weighted_five_item_batch(self):
        config = Config(
            cuda=False,
            learning_rate=0.0,
            lambda_recon=1.0,
            use_calendar_constraint=False,
            use_butterfly_constraint=False,
            use_smooth_constraint=False,
        )
        trainer = VolSurfaceRegressionTrainer(config)
        trainer.model = _BiasRegressor()
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.0)
        trainer.strike_grid = torch.tensor([0.9, 1.0, 1.1])
        trainer.tau_years = torch.tensor([30.0 / 365.0, 60.0 / 365.0])
        current = torch.zeros((2, 1, 2, 3))
        text = torch.zeros((2, 2))
        target = torch.tensor([0.0, 0.2]).view(2, 1, 1, 1).expand(-1, 1, 2, 3)
        loader = DataLoader(
            TensorDataset(
                current,
                text,
                target,
                torch.tensor([0.5, 1.0]),
                torch.tensor([101, 202], dtype=torch.int64),
            ),
            batch_size=2,
        )

        metrics = trainer._run_epoch(loader, train=False, epoch=1)
        self.assertAlmostEqual(metrics["val_recon"], 0.1, places=6)
        self.assertAlmostEqual(metrics["val_current_recon"], 2.0 / 15.0, places=6)


if __name__ == "__main__":
    unittest.main()
