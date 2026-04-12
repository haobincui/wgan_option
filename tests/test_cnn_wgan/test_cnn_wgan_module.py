import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cnn_wgan  # noqa: E402
from cnn_wgan.arbitrage import butterfly_arbitrage_penalty, calendar_arbitrage_penalty, reweight_scenarios  # noqa: E402
from cnn_wgan.config import CnnWGANSampleConfig, CnnWGANTrainConfig, load_sample_config, load_train_config  # noqa: E402
from cnn_wgan.data import create_train_val_bundle, denormalize_tensor, load_cnn_wgan_samples, normalize_surface_tensor  # noqa: E402
from cnn_wgan.io import ensure_dir, load_checkpoint  # noqa: E402
from cnn_wgan.losses import gradient_penalty, maturity_smoothness_penalty, strike_smoothness_penalty  # noqa: E402
from cnn_wgan.models import CnnWGANCritic, CnnWGANGenerator, reconstruct_future_surface  # noqa: E402
from cnn_wgan.training_plots import plot_training_curves  # noqa: E402


def _load_script_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _json_text(values):
    return json.dumps(list(values), ensure_ascii=False)


def _surface_values(base: float) -> list[float]:
    return [float(base + idx) / 1000.0 for idx in range(16)]


def _write_vol_workbook(tmpdir: str) -> Path:
    path = Path(tmpdir) / "merged_vol.xlsx"
    strike_grid = [0.8, 1.0, 1.2, 1.4]
    maturity_grid = [30, 60, 90, 120]
    dataframe = pd.DataFrame(
        [
            {
                "sample_id": "news_2",
                "news_timestamp_utc": "2022-12-30T13:40:00Z",
                "current_snapshot_time_utc": "2022-12-30T13:40:00Z",
                "target_snapshot_time_utc": "2022-12-30T13:45:00Z",
                "hd_embedding": _json_text([1.0, 2.0]),
                "lp_embedding": _json_text([10.0, 20.0, 30.0]),
                "strike_grid": _json_text(strike_grid),
                "maturity_days_grid": _json_text(maturity_grid),
                "current_surface_flat": _json_text(_surface_values(1.0)),
                "target_surface_flat": _json_text(_surface_values(2.0)),
                "pair_quality_label": "usable",
                "training_candidate_flag": 1,
            },
            {
                "sample_id": "news_1",
                "news_timestamp_utc": "2022-12-30T13:30:00Z",
                "current_snapshot_time_utc": "2022-12-30T13:30:00Z",
                "target_snapshot_time_utc": "2022-12-30T13:35:00Z",
                "hd_embedding": _json_text([3.0, 4.0]),
                "lp_embedding": _json_text([40.0, 50.0, 60.0]),
                "strike_grid": _json_text(strike_grid),
                "maturity_days_grid": _json_text(maturity_grid),
                "current_surface_flat": _json_text(_surface_values(3.0)),
                "target_surface_flat": _json_text(_surface_values(4.0)),
                "pair_quality_label": "usable",
                "training_candidate_flag": 1,
            },
            {
                "sample_id": "news_3",
                "news_timestamp_utc": "2022-12-30T13:50:00Z",
                "current_snapshot_time_utc": "2022-12-30T13:50:00Z",
                "target_snapshot_time_utc": "2022-12-30T13:55:00Z",
                "hd_embedding": _json_text([5.0, 6.0]),
                "lp_embedding": _json_text([70.0, 80.0, 90.0]),
                "strike_grid": _json_text(strike_grid),
                "maturity_days_grid": _json_text(maturity_grid),
                "current_surface_flat": _json_text(_surface_values(5.0)),
                "target_surface_flat": _json_text(_surface_values(6.0)),
                "pair_quality_label": "usable",
                "training_candidate_flag": 1,
            },
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name="gan_input_ready", index=False)
    return path


class TestStandaloneCnnWgan(unittest.TestCase):
    def test_package_is_importable_and_pyproject_includes_cnn_wgan(self):
        self.assertTrue(hasattr(cnn_wgan, "CnnWGANTrainer"))
        pyproject_text = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("cnn_wgan*", pyproject_text)

    def test_config_loading_from_default_yaml(self):
        train_config = load_train_config(ROOT_DIR / "configs/cnn_wgan/train_default.yaml")
        sample_config = load_sample_config(ROOT_DIR / "configs/cnn_wgan/sample_default.yaml")
        self.assertIsInstance(train_config, CnnWGANTrainConfig)
        self.assertIsInstance(sample_config, CnnWGANSampleConfig)
        self.assertEqual(train_config.text_embedding_mode, "hd")
        self.assertEqual(sample_config.split, "val")

    def test_workbook_parsing_is_chronological_and_embedding_modes_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            config = CnnWGANTrainConfig(
                data_path=str(workbook_path),
                text_embedding_mode="lp",
                train_ratio=0.67,
                batch_size=2,
                num_workers=0,
                cuda=False,
            )
            samples = load_cnn_wgan_samples(config)
            self.assertEqual([sample.sample_id for sample in samples], ["news_1", "news_2", "news_3"])
            bundle = create_train_val_bundle(config)
            self.assertEqual(bundle.train_samples, 2)
            self.assertEqual(bundle.val_samples, 1)
            batch = next(iter(bundle.train_loader))
            self.assertEqual(tuple(batch[0].shape[1:]), (1, 4, 4))

    def test_models_and_losses_behave_as_expected(self):
        generator = CnnWGANGenerator(
            surface_height=4,
            surface_width=4,
            embedding_dim=2,
            noise_dim=4,
            base_channels=8,
            res_blocks=1,
            text_hidden_dim=8,
            text_out_dim=4,
            fusion_hidden_dim=16,
        )
        critic = CnnWGANCritic(
            surface_height=4,
            surface_width=4,
            embedding_dim=2,
            base_channels=8,
            res_blocks=1,
            text_hidden_dim=8,
            text_out_dim=4,
            fusion_hidden_dim=16,
        )
        current = torch.full((3, 1, 4, 4), 0.2, dtype=torch.float32)
        text = torch.randn(3, 2, dtype=torch.float32)
        delta = generator(current, text)
        self.assertEqual(tuple(delta.shape), (3, 16))
        current_flat = torch.full((3, 16), 0.2, dtype=torch.float32)
        future_flat = reconstruct_future_surface(current_flat, delta)
        self.assertTrue(torch.all(future_flat > 0).item())
        future = future_flat.view(3, 1, 4, 4)
        scores = critic(future, current, text)
        self.assertEqual(tuple(scores.shape), (3, 1))

        gp = gradient_penalty(
            critic=critic,
            real_future_surface=future,
            fake_future_surface=future + 0.01,
            current_surface=current,
            text_embedding=text,
            lambda_gp=10.0,
        )
        self.assertGreaterEqual(float(gp.item()), 0.0)

        constant_log_surface = torch.zeros((2, 4, 4), dtype=torch.float32)
        strike_grid = torch.tensor([0.8, 1.0, 1.2, 1.4], dtype=torch.float32)
        maturity_days = torch.tensor([30.0, 60.0, 90.0, 120.0], dtype=torch.float32)
        self.assertAlmostEqual(float(strike_smoothness_penalty(constant_log_surface, strike_grid).item()), 0.0, places=6)
        self.assertAlmostEqual(float(maturity_smoothness_penalty(constant_log_surface, maturity_days).item()), 0.0, places=6)

        cal_pen = calendar_arbitrage_penalty(
            torch.tensor([[[0.5, 0.5, 0.5], [0.01, 0.01, 0.01]]], dtype=torch.float32),
            torch.tensor([0.8, 1.0, 1.2], dtype=torch.float32),
            torch.tensor([30.0, 365.0], dtype=torch.float32),
        )
        bfly_pen = butterfly_arbitrage_penalty(
            torch.tensor([[[0.01, 2.0, 0.01], [0.01, 2.0, 0.01]]], dtype=torch.float32),
            torch.tensor([0.8, 1.0, 1.2], dtype=torch.float32),
            torch.tensor([30.0, 60.0], dtype=torch.float32),
        )
        self.assertGreater(float(cal_pen.item()), 0.0)
        self.assertGreaterEqual(float(bfly_pen.item()), 0.0)
        weights, beta = reweight_scenarios(np.asarray([0.1, 1.0, 2.0]), beta_mode="fixed", beta_value=10.0)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=6)
        self.assertEqual(beta, 10.0)

    def test_normalization_helpers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            config = CnnWGANTrainConfig(
                data_path=str(workbook_path),
                text_embedding_mode="hd",
                train_ratio=0.67,
                batch_size=2,
                num_workers=0,
                cuda=False,
            )
            bundle = create_train_val_bundle(config)
            stats = bundle.normalization_stats
            current_flat = torch.tensor(bundle.train_items[0].current_surface.reshape(1, -1), dtype=torch.float32)
            norm = normalize_surface_tensor(
                current_flat,
                torch.tensor(stats.current_log_mean.reshape(1, -1), dtype=torch.float32),
                torch.tensor(stats.current_log_std.reshape(1, -1), dtype=torch.float32),
            )
            restored = denormalize_tensor(
                torch.zeros_like(norm),
                torch.tensor(stats.delta_mean.reshape(1, -1), dtype=torch.float32),
                torch.tensor(stats.delta_std.reshape(1, -1), dtype=torch.float32),
            )
            self.assertEqual(tuple(norm.shape), (1, 16))
            self.assertEqual(tuple(restored.shape), (1, 16))

    def test_training_plot_helper_writes_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "loss_curves.png"
            plot_training_curves(
                [
                    {
                        "epoch": 1,
                        "g_total": 1.0,
                        "g_adv": 0.8,
                        "g_calendar": 0.1,
                        "g_butterfly": 0.1,
                        "g_smooth": 0.1,
                        "d_total": 0.7,
                        "gp": 0.2,
                        "val_mae": 0.2,
                        "val_current_mae": 0.25,
                        "val_rmse": 0.3,
                        "val_current_rmse": 0.32,
                        "val_mae_gap_vs_current": -0.05,
                        "val_win_rate_vs_current": 0.6,
                        "val_generated_current_mae": 0.1,
                        "val_real_current_mae": 0.25,
                        "val_calendar": 0.01,
                        "val_butterfly": 0.02,
                        "val_penalty_mean": 0.01,
                        "val_penalty_std": 0.001,
                        "val_weight_entropy": 0.9,
                    }
                ],
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())

    def test_train_and_sample_smoke_via_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            train_output_root = Path(tmpdir) / "train_outputs"
            sample_output_root = Path(tmpdir) / "sample_outputs"
            ensure_dir(train_output_root)
            ensure_dir(sample_output_root)

            train_config_path = Path(tmpdir) / "train.yaml"
            sample_config_path = Path(tmpdir) / "sample.yaml"
            train_config_path.write_text(
                "\n".join(
                    [
                        f"data_path: {workbook_path}",
                        "sheet_name: gan_input_ready",
                        "text_embedding_mode: hd",
                        "train_ratio: 0.67",
                        "noise_dim: 4",
                        "gen_base_channels: 8",
                        "disc_base_channels: 8",
                        "gen_res_blocks: 0",
                        "disc_res_blocks: 0",
                        "text_hidden_dim: 8",
                        "text_out_dim: 4",
                        "fusion_hidden_dim: 16",
                        "learning_rate: 0.0001",
                        "generator_learning_rate: 0.0001",
                        "discriminator_learning_rate: 0.0001",
                        "beta_1: 0.5",
                        "beta_2: 0.9",
                        "critic_iter: 1",
                        "lambda_gp: 1.0",
                        "num_epochs: 1",
                        "batch_size: 2",
                        "lambda_calendar: 1.0",
                        "lambda_butterfly: 1.0",
                        "lambda_smooth: 0.1",
                        "use_calendar_constraint: true",
                        "use_butterfly_constraint: true",
                        "use_smooth_constraint: true",
                        "eval_mc_samples: 4",
                        "eval_reweight_beta_mode: fixed",
                        "eval_reweight_beta: 10.0",
                        "checkpoint_metric: val_mae_gap_vs_current",
                        "seed: 7",
                        "cuda: false",
                        "num_workers: 0",
                        f"output_root: {train_output_root}",
                        "save_every: 1",
                    ]
                ),
                encoding="utf-8",
            )

            main_module = _load_script_module(ROOT_DIR / "scripts/cnn_wgan/main.py", "cnn_wgan_main_script")
            train_run_dir = Path(main_module.main(["train", "--config", str(train_config_path)]))
            checkpoint_path = train_run_dir / "checkpoints" / "cnn_wgan_best.pt"
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue((train_run_dir / "metrics" / "training_metrics.json").exists())
            self.assertTrue((train_run_dir / "metrics" / "loss_curves.png").exists())
            checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
            self.assertIn("normalization_stats", checkpoint)

            sample_config_path.write_text(
                "\n".join(
                    [
                        f"data_path: {workbook_path}",
                        "sheet_name: gan_input_ready",
                        "text_embedding_mode: hd",
                        "train_ratio: 0.67",
                        f"checkpoint_path: {checkpoint_path}",
                        "seed: 7",
                        "cuda: false",
                        "mc_samples: 4",
                        "reweight_beta_mode: fixed",
                        "reweight_beta: 10.0",
                        "quantiles:",
                        "  - 0.5",
                        "split: val",
                        "selection_mode: all",
                        "selection_count: 0",
                        f"output_dir: {sample_output_root}",
                    ]
                ),
                encoding="utf-8",
            )

            sample_run_dir = Path(main_module.main(["sample", "--config", str(sample_config_path)]))
            self.assertTrue((sample_run_dir / "summary.csv").exists())
            sample_jsons = sorted((sample_run_dir / "samples").glob("*.json"))
            self.assertEqual(len(sample_jsons), 1)
            sample_pngs = sorted((sample_run_dir / "plots").glob("*.png"))
            self.assertGreaterEqual(len(sample_pngs), 1)
            payload = json.loads(sample_jsons[0].read_text(encoding="utf-8"))
            self.assertIn("generated_surface", payload)
            self.assertIn("weights", payload)

            metrics_rows = json.loads((train_run_dir / "metrics" / "training_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("val_mae_gap_vs_current", metrics_rows[-1])
            self.assertIn("val_penalty_mean", metrics_rows[-1])
            self.assertIn("val_weight_entropy", metrics_rows[-1])
            self.assertAlmostEqual(float(metrics_rows[-1]["val_mae"]), float(payload["metrics"]["mae"]), places=6)
            self.assertAlmostEqual(float(metrics_rows[-1]["val_current_mae"]), float(payload["current_metrics"]["mae"]), places=6)
            best_payload = json.loads((train_run_dir / "metrics" / "best_checkpoint.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(float(best_payload["best_metric"]), float(metrics_rows[-1]["val_mae_gap_vs_current"]), places=6)


if __name__ == "__main__":
    unittest.main()
