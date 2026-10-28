"""Protect Vega loss units, independent masks, and train/evaluation parity."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.rq3.news_first_vol_comparison_analysis import (  # noqa: E402
    RunSpec, aggregate_pair_metrics, compute_sample_metrics, _prediction_export_frame,
)
from wgan_option.config import Config  # noqa: E402
from wgan_option.models.gan_model import WGAN_GP  # noqa: E402
from wgan_option.train_vol_regression_xlsx import VolSurfaceRegressionTrainer  # noqa: E402
from wgan_option.utils.merged_xlsx_types import VolSurfaceSample  # noqa: E402
from wgan_option.utils.news_first_dataloaders import _weighted_loader, _with_cell_weights  # noqa: E402
from wgan_option.utils.vol_forecast_metrics import cell_weighted_mean_abs_error  # noqa: E402
from wgan_option.utils.weighted_training import (  # noqa: E402
    masked_cell_weighted_mean_per_sample, masked_mean_per_sample,
    unpack_vol_training_batch, validated_surface_mask,
)


class _MatrixGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.level = torch.nn.Parameter(torch.zeros(1, 2, 2))

    def forward(self, current, text):
        return self.level.unsqueeze(0).expand_as(current)


class _RecordingCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, future, current, text):
        self.calls.append((future.detach().clone(), current.detach().clone()))
        return future.reshape(len(future), -1).sum(dim=1, keepdim=True) * 0.0


def _sample():
    current = np.asarray([[[0.0, 2.0], [4.0, 8.0]]], dtype=np.float32)
    return VolSurfaceSample(
        sample_id="news_1", timestamp="2023-06-01T12:00:00Z",
        current_snapshot_time_utc="2023-06-01T12:00:00Z",
        target_snapshot_time_utc="2023-06-01T12:05:00Z",
        current_surface=current, target_surface=np.asarray([[[1, 3], [999, 999]]], dtype=np.float32),
        text_embedding=np.zeros(2, dtype=np.float32),
        strike_grid=np.asarray([0.9, 1.1], dtype=np.float32),
        maturity_grid_days=np.asarray([10, 20], dtype=np.float32),
        surface_shape=(2, 2), global_index=0,
        metadata={"raw_pair_sample_weight": 1.0, "scaled_training_sample_weight": 1.0},
        pair_id="pair_1", session_id="session_1", stable_sample_key="pair_1",
        support_mask=np.asarray([[[1, 1], [0, 0]]], dtype=np.float32),
        current_support_mask=np.ones((1, 2, 2), dtype=np.float32),
    )


class TestVegaTrainingContract(unittest.TestCase):
    def test_mae_only_scope_enforces_selection_and_preserves_legacy_monitor(self):
        from scripts.rq3.vega_experiment_support import validate_weighted_experiment

        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"mode": "black76_current_vega_v1"}))
            config = {
                "data": {
                    "cell_weight_mode": "black76_current_vega_v1",
                    "cell_weight_manifest_path": str(manifest),
                    "cell_weight_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "support_mask_mode": "raw_joint",
                },
                "training": {"best_checkpoint_metric": "val_vega_hybrid_score", "baseline_penalty_weight": 2.0},
            }
            validate_weighted_experiment(config)
            config["training"].update(best_checkpoint_metric="val_vega_recon", baseline_penalty_weight=0.0, vega_weight_scope="mae_only")
            validate_weighted_experiment(config)
            wrong_monitor = deepcopy(config)
            wrong_monitor["training"]["best_checkpoint_metric"] = "val_vega_hybrid_score"
            with self.assertRaisesRegex(ValueError, "mae_only scope must select"):
                validate_weighted_experiment(wrong_monitor)
            for penalty in (2.0, -1.0, None, float("nan"), float("inf")):
                wrong_penalty = deepcopy(config)
                wrong_penalty["training"]["baseline_penalty_weight"] = penalty
                with self.subTest(penalty=penalty), self.assertRaisesRegex(ValueError, "baseline_penalty_weight=0"):
                    validate_weighted_experiment(wrong_penalty)
            unweighted = deepcopy(config)
            unweighted["data"]["cell_weight_mode"] = "none"
            with self.assertRaisesRegex(ValueError, "requires Vega cell weighting"):
                validate_weighted_experiment(unweighted)

    def test_mae_only_scope_is_experiment_metadata_not_unknown_model_config(self):
        from scripts.rq3 import news_first_vol_film_unet_text_10seed_lr2p5e5 as film
        from scripts.rq3 import news_first_vol_cnn_unet_pure_no_text_10seed as pure

        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"mode": "black76_current_vega_v1"}))
            for module, arm in ((film, "lp_matched"), (pure, "pure_cnn_no_text")):
                config = module.load_config()
                config["data"].update(
                    cell_weight_mode="black76_current_vega_v1",
                    cell_weight_manifest_path=str(manifest),
                    cell_weight_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                )
                config["training"].update(best_checkpoint_metric="val_vega_recon", baseline_penalty_weight=0.0, vega_weight_scope="mae_only")
                with self.subTest(module=module.__name__):
                    payload = module._training_payload(config, Path(directory), {"arm": arm, "seed": 42})
                    self.assertEqual(payload["best_checkpoint_metric"], "val_vega_recon")
                    self.assertNotIn("vega_weight_scope", payload)
                    self.assertTrue(set(payload).issubset(Config.__dataclass_fields__))

    def test_formula_scaling_gradient_and_numpy_parity(self):
        errors = torch.tensor([[[[1.0, 3.0], [999.0, 999.0]]]], requires_grad=True)
        mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        weights = torch.tensor([[[[1.0, 3.0], [1e8, 1e8]]]], requires_grad=True)
        actual = masked_cell_weighted_mean_per_sample(errors, mask, weights)
        self.assertAlmostEqual(float(actual.item()), 2.5)
        torch.testing.assert_close(actual, masked_cell_weighted_mean_per_sample(errors, mask, 100 * weights))
        torch.testing.assert_close(masked_cell_weighted_mean_per_sample(errors, mask), masked_mean_per_sample(errors, mask))
        actual.sum().backward()
        torch.testing.assert_close(errors.grad, torch.tensor([[[[0.25, 0.75], [0.0, 0.0]]]]))
        self.assertIsNone(weights.grad)
        value = cell_weighted_mean_abs_error(
            errors.detach().numpy()[0, 0], np.zeros((2, 2)),
            weights.detach().numpy()[0, 0], mask.numpy()[0, 0],
        )
        self.assertAlmostEqual(value, 2.5)

    def test_bad_weights_rejected_and_mask_stays_binary(self):
        surface = torch.ones(1, 1, 2, 2)
        for value in (-1.0, float("nan"), 0.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                masked_cell_weighted_mean_per_sample(surface, surface, torch.full_like(surface, value))
        with self.assertRaisesRegex(ValueError, "binary"):
            validated_surface_mask(torch.full_like(surface, 2.0), reference_surface=surface)

    def test_loader_preserves_ivs_and_old_batch_positions(self):
        sample = _sample()
        weights = np.asarray([[1, 3], [0, 0]], dtype=np.float32)
        metadata = {"cell_weight_fingerprint": "a" * 64, "manifest_sha256": "b" * 64}
        config = SimpleNamespace(cell_weight_mode="black76_current_vega_v1", cell_weight_manifest_path="manifest.json", cell_weight_manifest_sha256="b" * 64)
        with patch("wgan_option.utils.vega_weights.load_vega_weight_grid", return_value=(weights, metadata)) as load:
            [weighted] = _with_cell_weights([sample], config)
        self.assertEqual(load.call_args.args[1], sample.current_snapshot_time_utc)
        np.testing.assert_array_equal(weighted.current_surface, sample.current_surface)
        np.testing.assert_array_equal(weighted.target_surface, sample.target_surface)
        loader, _ = _weighted_loader([weighted], batch_size=1, num_workers=0, seed=3, shuffle=False, training_weights=True)
        raw = next(iter(loader))
        self.assertEqual(len(raw), 9)
        batch = unpack_vol_training_batch(raw)
        torch.testing.assert_close(batch.label_reliability_weight, torch.ones(1))
        torch.testing.assert_close(batch.cell_weight, torch.tensor(weights).reshape(1, 1, 2, 2))
        self.assertIsNone(unpack_vol_training_batch(raw[:8]).cell_weight)
        self.assertIsNone(unpack_vol_training_batch(raw[:7]).label_reliability_weight)

    def test_wgan_optimizes_vega_mae_but_critic_sees_original_ivs(self):
        config = Config(cuda=False, gen_base_channels=2, disc_base_channels=2, gen_hidden_dim=8, disc_hidden_dim=8, gen_text_hidden_dim=4, gen_text_out_dim=2, disc_text_hidden_dim=2, lambda_recon=1.0, use_calendar_constraint=False, use_butterfly_constraint=False, use_smooth_constraint=False)
        model = WGAN_GP(config, np.asarray([0.9, 1.1]), np.asarray([10, 20]), embedding_dim=2)
        model.G, model.D = _MatrixGenerator(), _RecordingCritic()
        model.g_optimizer = torch.optim.SGD(model.G.parameters(), lr=0.0)
        sample = replace(_sample(), cell_weight=np.asarray([[[1, 3], [0, 0]]], dtype=np.float32))
        loader, _ = _weighted_loader([sample], batch_size=1, num_workers=0, seed=3, shuffle=False, training_weights=False)
        batch = unpack_vol_training_batch(next(iter(loader)))
        stats = model._generator_step(batch.current_surface, batch.text_embedding, batch.target_surface, support_mask=batch.support_mask, cell_weight=batch.cell_weight)
        self.assertAlmostEqual(stats["g_recon"], 2.0)
        self.assertAlmostEqual(stats["g_vega_recon"], 2.5)
        self.assertAlmostEqual(stats["g_total"], 2.5)
        torch.testing.assert_close(model.D.calls[0][1], batch.current_surface * batch.support_mask)
        metrics = model._evaluate(loader)
        self.assertAlmostEqual(metrics["val_recon"], 2.0)
        self.assertAlmostEqual(metrics["val_vega_recon"], 2.5)
        self.assertAlmostEqual(metrics["val_vega_current_recon"], 1.0)
        self.assertAlmostEqual(metrics["val_vega_hybrid_score"], 5.5)

        trainer = VolSurfaceRegressionTrainer(config)
        trainer.model = _MatrixGenerator()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
        trainer.strike_grid, trainer.tau_years = torch.tensor([0.9, 1.1]), torch.tensor([10 / 365, 20 / 365])
        regression = trainer._run_epoch(loader, train=False, epoch=1)
        for field in ("val_recon", "val_vega_recon", "val_vega_current_recon", "val_vega_hybrid_score"):
            self.assertAlmostEqual(metrics[field], regression[field])

        # The optional uncertainty fallback must select and report the same
        # error definition used by weighted validation checkpoint selection.
        with tempfile.TemporaryDirectory() as directory:
            model.model_path = directory
            model._fallback_calibration_file = str(Path(directory) / "fallback.json")
            Path(directory, "generator_best.pt").touch()
            model.config.cell_weight_mode = "black76_current_vega_v1"
            with patch("wgan_option.models.gan_model.load_vol_generator", return_value=(model.G, config, 2)), patch("wgan_option.models.gan_model.infer_vol_surface_mc", side_effect=[(np.zeros((2, 2)), 1.0, {}), (np.zeros((2, 2)), 2.0, {})]):
                model._calibrate_fallback_threshold([sample, sample])
            calibration = json.loads(Path(model._fallback_calibration_file).read_text())
            self.assertAlmostEqual(calibration["selected_metrics"]["val_recon"], 1.5)
            self.assertAlmostEqual(calibration["selected_metrics"]["val_vega_recon"], 1.75)
            self.assertAlmostEqual(calibration["selected_metrics"]["val_vega_hybrid_score"], 3.25)

    def test_prediction_keeps_unweighted_metrics_and_weight_lineage(self):
        sample = _sample()
        serial = lambda a: json.dumps(np.asarray(a).reshape(-1).tolist())
        panel = pd.DataFrame([{
            "sample_id": "news_1", "news_row_id": 1, "pair_id": "pair_1", "session_id": "session_1",
            "effective_origin_utc": "2023-11-01T12:00:00Z", "current_snapshot_time_utc": sample.current_snapshot_time_utc,
            "current_surface_flat": serial(sample.current_surface), "target_surface_flat": serial(sample.target_surface),
            "surface_shape": "[2,2]", "strike_grid": "[0.9,1.1]", "maturity_days_grid": "[10,20]",
            "sample_weight": 1.0, "lp_embedding": "[0,0]", "hd_embedding": "[0,0]",
        }])
        predictions = pd.DataFrame([{"sample_id": "news_1", "predicted_surface_flat": "[0,0,0,0]"}])
        run = RunSpec("run", Path("/tmp"), "wgan", 5, 42, Path("/tmp/model.pt"), support_mask_mode="raw_joint", cell_weight_mode="black76_current_vega_v1", cell_weight_manifest_path="/tmp/weights.json", cell_weight_manifest_sha256="b" * 64)
        metadata = {"cell_weight_fingerprint": "a" * 64, "manifest_sha256": "b" * 64, "source_method": "current_quotes"}
        with patch("scripts.rq3.news_first_vol_comparison_analysis._joint_support_mask", return_value=(sample.support_mask[0].astype(bool), "raw_joint", "support")), patch("wgan_option.utils.vega_weights.load_vega_weight_grid", return_value=(np.asarray([[1,3],[0,0]], dtype=np.float32), metadata)):
            samples, exclusions, _ = compute_sample_metrics(run, "core", panel, predictions, evaluate_embedded_atm_skew=False)
            export = _prediction_export_frame(run, "core", panel, predictions, samples)
            for column in ("cell_weight_mode", "cell_weight_manifest_sha256", "cell_weight_fingerprint"):
                self.assertEqual(export.iloc[0][column], samples.iloc[0][column])
        self.assertTrue(exclusions.empty)
        self.assertAlmostEqual(samples.iloc[0]["model_mae"], 2.0)
        self.assertAlmostEqual(samples.iloc[0]["vega_model_mae"], 2.5)
        self.assertAlmostEqual(samples.iloc[0]["vega_persistence_mae"], 1.0)
        pairs = aggregate_pair_metrics(samples)
        self.assertAlmostEqual(pairs.iloc[0]["vega_model_mae"], 2.5)
        self.assertEqual(pairs.iloc[0]["cell_weight_fingerprint"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
