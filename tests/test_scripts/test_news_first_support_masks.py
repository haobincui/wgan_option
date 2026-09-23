"""Focused coverage for raw joint-support masked vol training."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for value in (str(ROOT), str(SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from wgan_option.config import Config  # noqa: E402
from wgan_option.models.gan_model import WGAN_GP  # noqa: E402
from wgan_option.train_vol_regression_xlsx import (  # noqa: E402
    VolSurfaceRegressionTrainer,
)
from wgan_option.utils.merged_xlsx_samples import (  # noqa: E402
    load_vol_surface_samples_with_diagnostics,
)
from wgan_option.utils.merged_xlsx_dataloaders import (  # noqa: E402
    create_vol_surface_xlsx_dataloaders,
)
from wgan_option.utils.news_first_dataloaders import (  # noqa: E402
    create_news_first_vol_surface_dataloaders,
)
from wgan_option.utils.weighted_training import (  # noqa: E402
    masked_mean_per_sample,
    unpack_vol_training_batch,
)


def _raw_params(lower: float, upper: float) -> str:
    return json.dumps(
        {
            "business_days": [10, 20],
            "percent_strikes": [[lower, upper], [lower, upper]],
            "implied_vols": [[0.20, 0.21], [0.22, 0.23]],
        }
    )


def _row(
    sample_id: str,
    origin: str,
    pair_id: str,
    session_id: str,
    *,
    zero_joint: bool = False,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "news_row_id": int(sum(ord(value) for value in sample_id)),
        "news_timestamp_utc": origin,
        "effective_origin_utc": origin,
        "current_snapshot_time_utc": origin,
        "target_snapshot_time_utc": origin,
        "pair_id": pair_id,
        "session_id": session_id,
        "hd_embedding": "[1.0, 2.0]",
        "lp_embedding": "[3.0, 4.0]",
        "current_surface_flat": "[0.20, 0.21, 0.22, 0.23, 0.24, 0.25]",
        "target_surface_flat": "[0.21, 0.22, 0.23, 0.24, 0.25, 0.26]",
        "surface_shape": "[2, 3]",
        "strike_grid": "[0.9, 1.0, 1.1]",
        "maturity_days_grid": "[10, 20]",
        "surface_model": "raw",
        "current_surface_param_json": _raw_params(0.9, 1.05),
        "target_surface_param_json": (
            _raw_params(1.2, 1.3) if zero_joint else _raw_params(0.95, 1.1)
        ),
        "training_candidate_flag": 1,
        "sample_weight": 1.0,
    }


class _ConstantGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, current_surface, text_embedding):
        del text_embedding
        return torch.zeros_like(current_surface) + self.bias


class _RecordingSumCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, next_surface, current_surface, text_embedding):
        del text_embedding
        self.calls.append(
            (next_surface.detach().clone(), current_surface.detach().clone())
        )
        return (
            next_surface.reshape(next_surface.shape[0], -1).sum(dim=1, keepdim=True)
            * self.scale
        )


class _UnitBiasRegressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, current_surface, text_embedding):
        del text_embedding
        return current_surface + self.bias


class TestNewsFirstSupportMasks(unittest.TestCase):
    def test_raw_joint_loader_emits_separate_joint_and_current_masks(self):
        training = pd.DataFrame(
            [
                _row("t1", "2023-06-01T12:00:00Z", "p1", "s1"),
                _row("t2", "2023-06-01T12:01:00Z", "p1", "s1"),
                _row(
                    "tz",
                    "2023-06-02T12:00:00Z",
                    "pz",
                    "sz",
                    zero_joint=True,
                ),
            ]
        )
        common = pd.DataFrame(
            [
                _row("v1", "2023-08-01T12:00:00Z", "pv", "sv"),
                _row("e1", "2023-11-01T12:00:00Z", "pe", "se"),
            ]
        )
        config = Config(
            data_path="train.xlsx",
            text_embedding_mode="lp",
            support_mask_mode="raw_joint",
            batch_size=8,
            num_workers=0,
        )
        with patch(
            "wgan_option.utils.merged_xlsx_samples._read_sheet",
            side_effect=[training, common],
        ):
            bundle = create_news_first_vol_surface_dataloaders(config, "5m.xlsx")

        batch = next(iter(bundle.train_loader))
        self.assertEqual(len(batch), 7)
        self.assertTrue(bundle.uses_support_masks)
        self.assertTrue(bundle.uses_current_support_masks)
        self.assertEqual(bundle.train_samples, 2)
        torch.testing.assert_close(batch[3].sum(), torch.tensor(2.0))
        expected_mask = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]])
        for mask in batch[5]:
            torch.testing.assert_close(mask.unsqueeze(0), expected_mask)
        expected_current_mask = torch.tensor([[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]])
        for mask in batch[6]:
            torch.testing.assert_close(mask.unsqueeze(0), expected_current_mask)
        self.assertEqual(
            bundle.split_metadata["support_filter_splits"]["train"]["input_rows"],
            3,
        )
        self.assertEqual(
            bundle.split_metadata["support_filter_splits"]["train"][
                "excluded_zero_joint_support_rows"
            ],
            1,
        )
        sample = bundle.train_items[0]
        self.assertEqual(len(sample.support_grid_fingerprint), 64)
        self.assertEqual(len(sample.support_mask_fingerprint), 64)
        self.assertEqual(len(sample.current_support_mask_fingerprint), 64)
        self.assertNotEqual(
            sample.support_mask_fingerprint,
            sample.current_support_mask_fingerprint,
        )
        np.testing.assert_array_equal(
            sample.current_support_mask,
            expected_current_mask.numpy()[0],
        )
        self.assertEqual(sample.metadata["current_raw_support_cell_count"], 4)
        self.assertEqual(sample.metadata["joint_raw_support_cell_count"], 2)

    def test_current_support_mask_is_independent_of_target_params(self):
        first = _row("first", "2023-06-01T12:00:00Z", "p1", "s1")
        second = _row("second", "2023-06-01T12:01:00Z", "p2", "s2")
        second["target_surface_param_json"] = _raw_params(0.9, 0.95)
        config = Config(
            data_path="fixture.xlsx",
            text_embedding_mode="lp",
            support_mask_mode="raw_joint",
        )
        with patch(
            "wgan_option.utils.merged_xlsx_samples._read_sheet",
            return_value=pd.DataFrame([first, second]),
        ):
            samples, _ = load_vol_surface_samples_with_diagnostics(config)

        self.assertEqual(len(samples), 2)
        np.testing.assert_array_equal(
            samples[0].current_support_mask,
            samples[1].current_support_mask,
        )
        self.assertEqual(
            samples[0].current_support_mask_fingerprint,
            samples[1].current_support_mask_fingerprint,
        )
        self.assertFalse(
            np.array_equal(samples[0].support_mask, samples[1].support_mask)
        )
        self.assertNotEqual(
            samples[0].support_mask_fingerprint,
            samples[1].support_mask_fingerprint,
        )

    def test_chronological_xlsx_loader_uses_same_seven_item_contract(self):
        frame = pd.DataFrame(
            [
                _row("first", "2023-06-01T12:00:00Z", "p1", "s1"),
                _row("second", "2023-06-02T12:00:00Z", "p2", "s2"),
            ]
        )
        config = Config(
            data_path="fixture.xlsx",
            text_embedding_mode="lp",
            support_mask_mode="raw_joint",
            train_ratio=0.5,
            batch_size=8,
            num_workers=0,
        )
        with patch(
            "wgan_option.utils.merged_xlsx_samples._read_sheet",
            return_value=frame,
        ):
            bundle = create_vol_surface_xlsx_dataloaders(config)

        self.assertEqual(len(next(iter(bundle.train_loader))), 7)
        self.assertEqual(len(next(iter(bundle.val_loader))), 7)
        self.assertTrue(bundle.uses_support_masks)
        self.assertTrue(bundle.uses_current_support_masks)

    def test_raw_support_fails_closed_when_row_grid_differs(self):
        first = _row("first", "2023-06-01T12:00:00Z", "p1", "s1")
        second = _row("second", "2023-06-02T12:00:00Z", "p2", "s2")
        second["maturity_days_grid"] = "[10, 20.4]"
        config = Config(
            data_path="fixture.xlsx",
            text_embedding_mode="lp",
            support_mask_mode="raw_joint",
        )
        with patch(
            "wgan_option.utils.merged_xlsx_samples._read_sheet",
            return_value=pd.DataFrame([first, second]),
        ):
            with self.assertRaisesRegex(ValueError, "Support grid differs"):
                load_vol_surface_samples_with_diagnostics(config)

    def test_loader_diagnostics_and_legacy_batch_contract(self):
        frame = pd.DataFrame(
            [
                _row("kept", "2023-06-01T12:00:00Z", "pk", "sk"),
                _row(
                    "dropped",
                    "2023-06-02T12:00:00Z",
                    "pd",
                    "sd",
                    zero_joint=True,
                ),
            ]
        )
        config = Config(
            data_path="fixture.xlsx",
            text_embedding_mode="lp",
            support_mask_mode="raw_joint",
        )
        with patch(
            "wgan_option.utils.merged_xlsx_samples._read_sheet",
            return_value=frame,
        ):
            samples, diagnostics = load_vol_surface_samples_with_diagnostics(config)
        self.assertEqual([sample.sample_id for sample in samples], ["kept"])
        self.assertEqual(diagnostics["loaded_rows"], 1)
        self.assertEqual(diagnostics["excluded_zero_joint_support_rows"], 1)
        self.assertEqual(diagnostics["time_partitions"]["train"]["kept_pairs"], 1)

        current = torch.zeros(2, 1, 2, 2)
        text = torch.zeros(2, 3)
        target = torch.ones(2, 1, 2, 2)
        legacy = unpack_vol_training_batch((current, text, target))
        self.assertIsNone(legacy.support_mask)
        self.assertIsNone(legacy.current_support_mask)
        torch.testing.assert_close(legacy.sample_weight, torch.ones(2))
        support = torch.ones(2, 1, 2, 2)
        masked = unpack_vol_training_batch(
            (current, text, target, torch.ones(2), torch.arange(2), support)
        )
        torch.testing.assert_close(masked.support_mask, support)
        self.assertIsNone(masked.current_support_mask)
        current_support = torch.tensor(
            [
                [[[1.0, 1.0], [0.0, 0.0]]],
                [[[1.0, 0.0], [1.0, 0.0]]],
            ]
        )
        current_masked = unpack_vol_training_batch(
            (
                current,
                text,
                target,
                torch.ones(2),
                torch.arange(2),
                support,
                current_support,
            )
        )
        torch.testing.assert_close(current_masked.support_mask, support)
        torch.testing.assert_close(
            current_masked.current_support_mask,
            current_support,
        )

    def test_masked_reduction_critic_and_gradient_penalty_ignore_unsupported_cells(
        self,
    ):
        values = torch.tensor([[[[2.0, 100.0], [100.0, 100.0]]]])
        support = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        torch.testing.assert_close(
            masked_mean_per_sample(values, support), torch.tensor([2.0])
        )

        config = Config(
            cuda=False,
            lambda_gp=10.0,
            lambda_recon=1.0,
            lambda_calendar=0.0,
            lambda_butterfly=0.0,
            lambda_smooth=0.0,
            use_calendar_constraint=False,
            use_butterfly_constraint=False,
            use_smooth_constraint=False,
        )
        model = WGAN_GP(
            config=config,
            strike_grid=np.asarray([0.9, 1.1], dtype=np.float32),
            maturity_grid_days=np.asarray([10.0, 20.0], dtype=np.float32),
            embedding_dim=2,
        )
        model.G = _ConstantGenerator()
        model.D = _RecordingSumCritic()
        model.g_optimizer = torch.optim.SGD(model.G.parameters(), lr=0.01)
        model.d_optimizer = torch.optim.SGD(model.D.parameters(), lr=0.01)

        current = torch.tensor([[[[0.0, 7.0], [8.0, 9.0]]]])
        real = torch.tensor([[[[2.0, 100.0], [100.0, 100.0]]]])
        text = torch.zeros(1, 2)
        stats = model._generator_step(
            current,
            text,
            real,
            sample_weight=torch.ones(1),
            support_mask=support,
        )
        self.assertAlmostEqual(stats["g_recon"], 2.0, places=6)
        critic_next, critic_current = model.D.calls[0]
        self.assertEqual(int(torch.count_nonzero(critic_next[:, :, :, 1:])), 0)
        self.assertEqual(float(critic_current[0, 0, 1, 1]), 0.0)

        masked_gp = model.calculate_gradient_penalty(
            real,
            torch.zeros_like(real),
            current,
            text,
            torch.ones(1),
            support,
        )
        unmasked_gp = model.calculate_gradient_penalty(
            real,
            torch.zeros_like(real),
            current,
            text,
            torch.ones(1),
            None,
        )
        self.assertAlmostEqual(float(masked_gp), 0.0, places=6)
        self.assertAlmostEqual(float(unmasked_gp), 10.0, places=6)

    def test_regression_validation_metrics_are_cell_masked(self):
        trainer = object.__new__(VolSurfaceRegressionTrainer)
        trainer.config = Config(
            cuda=False,
            lambda_recon=1.0,
            lambda_calendar=0.0,
            lambda_butterfly=0.0,
            lambda_smooth=0.0,
            use_calendar_constraint=False,
            use_butterfly_constraint=False,
            use_smooth_constraint=False,
        )
        trainer.device = torch.device("cpu")
        trainer.model = _UnitBiasRegressor()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        trainer.strike_grid = torch.tensor([0.9, 1.1])
        trainer.tau_years = torch.tensor([10.0 / 365.0, 20.0 / 365.0])

        current = torch.zeros(1, 1, 2, 2)
        target = torch.tensor([[[[0.0, 100.0], [100.0, 100.0]]]])
        support = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        loader = DataLoader(
            TensorDataset(
                current,
                torch.zeros(1, 2),
                target,
                torch.ones(1),
                torch.ones(1, dtype=torch.int64),
                support,
            ),
            batch_size=1,
        )
        metrics = trainer._run_epoch(loader, train=False, epoch=1)
        self.assertAlmostEqual(metrics["val_recon"], 1.0, places=6)
        self.assertAlmostEqual(metrics["val_current_recon"], 0.0, places=6)

    def test_constraints_require_fully_supported_neighbors(self):
        model = WGAN_GP(
            config=Config(cuda=False),
            strike_grid=np.asarray([0.9, 1.0, 1.1], dtype=np.float32),
            maturity_grid_days=np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
            embedding_dim=2,
        )
        support = torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]])
        baseline = torch.tensor(
            [[[[0.20, 0.21, 0.22], [0.23, 0.24, 0.25], [0.26, 0.27, 0.28]]]]
        )
        changed_only_outside = baseline.clone()
        changed_only_outside[:, :, 2, :] = torch.tensor([10.0, 0.001, 20.0])

        for penalty in (
            model._calendar_penalty_per_sample,
            model._butterfly_penalty_per_sample,
            model._smoothness_penalty_per_sample,
        ):
            torch.testing.assert_close(
                penalty(baseline, support),
                penalty(changed_only_outside, support),
            )

    def test_wgan_validation_recon_and_persistence_are_cell_masked(self):
        model = WGAN_GP(
            config=Config(cuda=False, baseline_penalty_weight=2.0),
            strike_grid=np.asarray([0.9, 1.1], dtype=np.float32),
            maturity_grid_days=np.asarray([10.0, 20.0], dtype=np.float32),
            embedding_dim=2,
        )
        model.G = _UnitBiasRegressor()
        current = torch.zeros(1, 1, 2, 2)
        target = torch.tensor([[[[0.0, 100.0], [100.0, 100.0]]]])
        support = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        loader = DataLoader(
            TensorDataset(
                current,
                torch.zeros(1, 2),
                target,
                torch.ones(1),
                torch.ones(1, dtype=torch.int64),
                support,
            ),
            batch_size=1,
        )

        metrics = model._evaluate(loader)
        self.assertAlmostEqual(metrics["val_recon"], 1.0, places=6)
        self.assertAlmostEqual(metrics["val_current_recon"], 0.0, places=6)
        self.assertAlmostEqual(metrics["val_hybrid_score"], 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
