import importlib.util
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wgan_option.config import Config, load_config, parse_cli_overrides  # noqa: E402
from wgan_option.models.gan_model import WGAN_GP  # noqa: E402
from wgan_option.train_svi_xlsx import SviXlsxTrainer  # noqa: E402
from wgan_option.utils.merged_xlsx import (  # noqa: E402
    create_svi_xlsx_dataloaders,
    create_vol_surface_xlsx_dataloaders,
)
from wgan_option.utils.visualization import plot_training_curves  # noqa: E402


def _load_script_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _surface_values(base: float):
    return [float(base + idx) / 1000.0 for idx in range(256)]


def _json_text(values):
    return json.dumps(list(values), ensure_ascii=False)


class _DummyGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, current_surface, text_embedding):
        return current_surface + self.bias


class _DummyDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, future_surface, current_surface, text_embedding):
        return future_surface.mean(dim=(1, 2, 3)) * self.scale


def _build_test_wgan(config: Config) -> WGAN_GP:
    model = WGAN_GP(
        config=config,
        strike_grid=np.asarray([0.9, 1.1], dtype=np.float32),
        maturity_grid_days=np.asarray([30.0, 60.0], dtype=np.float32),
        embedding_dim=2,
    )
    model.G = _DummyGenerator().to(model.device)
    model.D = _DummyDiscriminator().to(model.device)
    model.g_optimizer = torch.optim.Adam(model.G.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2))
    model.d_optimizer = torch.optim.Adam(model.D.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2))
    return model


def _build_small_wgan_loader():
    dataset = torch.utils.data.TensorDataset(
        torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((1, 1, 2, 2), dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


class TestTrainMergedXlsx(unittest.TestCase):
    def _write_vol_workbook(self, tmpdir: str) -> Path:
        path = Path(tmpdir) / "merged_vol.xlsx"
        strike_grid = [0.7 + 0.04 * idx for idx in range(16)]
        maturity_grid = [7 + int((365 - 7) * idx / 15) for idx in range(16)]
        df = pd.DataFrame(
            [
                {
                    "sample_id": "news_2",
                    "news_timestamp_utc": "2022-12-30T13:40:00Z",
                    "current_snapshot_time_utc": "2022-12-30T13:40:00Z",
                    "target_snapshot_time_utc": "2022-12-30T13:45:00Z",
                    "hd_embedding": _json_text([1.0, 2.0]),
                    "lp_embedding": _json_text([10.0, 20.0, 30.0]),
                    "hd_dim": 2,
                    "lp_dim": 3,
                    "strike_grid": _json_text(strike_grid),
                    "maturity_days_grid": _json_text(maturity_grid),
                    "surface_shape": _json_text([16, 16]),
                    "current_surface_flat": _json_text(_surface_values(1.0)),
                    "target_surface_flat": _json_text(_surface_values(2.0)),
                    "current_weighted_iv_rmse": 0.01,
                    "target_weighted_iv_rmse": 0.02,
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
                    "hd_dim": 2,
                    "lp_dim": 3,
                    "strike_grid": _json_text(strike_grid),
                    "maturity_days_grid": _json_text(maturity_grid),
                    "surface_shape": _json_text([16, 16]),
                    "current_surface_flat": _json_text(_surface_values(3.0)),
                    "target_surface_flat": _json_text(_surface_values(4.0)),
                    "current_weighted_iv_rmse": 0.01,
                    "target_weighted_iv_rmse": 0.02,
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
                    "hd_dim": 2,
                    "lp_dim": 3,
                    "strike_grid": _json_text(strike_grid),
                    "maturity_days_grid": _json_text(maturity_grid),
                    "surface_shape": _json_text([16, 16]),
                    "current_surface_flat": _json_text(_surface_values(5.0)),
                    "target_surface_flat": _json_text(_surface_values(6.0)),
                    "current_weighted_iv_rmse": 0.01,
                    "target_weighted_iv_rmse": 0.02,
                    "pair_quality_label": "usable",
                    "training_candidate_flag": 1,
                },
            ]
        )
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="gan_input_ready", index=False)
        return path

    def _write_svi_workbook(self, tmpdir: str) -> Path:
        path = Path(tmpdir) / "merged_svi.xlsx"

        def row(news_row_id, direction, timestamp, bd_list, a_list, b_list, rho_list, m_list, sigma_list):
            return {
                "sample_id": f"news_{news_row_id}_{direction}",
                "news_row_id": news_row_id,
                "direction": direction,
                "news_timestamp_utc": timestamp,
                "matched_snapshot_time_utc": timestamp,
                "json_target_timestamp_utc": timestamp,
                "hd_embedding": _json_text([1.0 * news_row_id, 2.0 * news_row_id]),
                "lp_embedding": _json_text([10.0 * news_row_id, 20.0 * news_row_id, 30.0 * news_row_id]),
                "hd_dim": 2,
                "lp_dim": 3,
                "svi_business_days_list": _json_text(bd_list),
                "svi_a_list": _json_text(a_list),
                "svi_b_list": _json_text(b_list),
                "svi_rho_list": _json_text(rho_list),
                "svi_m_list": _json_text(m_list),
                "svi_sigma_list": _json_text(sigma_list),
                "fit_quality_label": "usable",
                "training_candidate_flag": 1,
            }

        df = pd.DataFrame(
            [
                row(2, "backward", "2022-12-30T13:40:00Z", [30, 31], [0.3, 0.31], [0.4, 0.41], [0.1, 0.11], [0.0, 0.01], [0.2, 0.21]),
                row(1, "forward", "2022-12-30T13:35:00Z", [20], [0.2], [0.3], [0.1], [0.0], [0.2]),
                row(1, "backward", "2022-12-30T13:30:00Z", [10], [0.1], [0.2], [0.1], [0.0], [0.2]),
                row(2, "forward", "2022-12-30T13:45:00Z", [40, 41], [0.4, 0.41], [0.5, 0.51], [0.1, 0.11], [0.0, 0.01], [0.2, 0.21]),
                row(3, "backward", "2022-12-30T13:50:00Z", [1000], [1.0], [1.1], [0.1], [0.0], [0.2]),
                row(3, "forward", "2022-12-30T13:55:00Z", [1001], [1.1], [1.2], [0.1], [0.0], [0.2]),
                row(4, "backward", "2022-12-30T14:00:00Z", [50], [0.5], [0.6], [0.1], [0.0], [0.2]),
            ]
        )
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="news_direction_audit", index=False)
        return path

    def test_vol_loader_parses_embeddings_and_uses_chronological_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_vol_workbook(tmpdir)

            hd_bundle = create_vol_surface_xlsx_dataloaders(
                Config(
                    data_path=str(workbook_path),
                    sheet_name="gan_input_ready",
                    text_embedding_mode="hd",
                    batch_size=8,
                    train_ratio=2 / 3,
                    num_workers=0,
                    cuda=False,
                )
            )
            self.assertEqual(hd_bundle.embedding_dim, 2)
            self.assertEqual(hd_bundle.train_samples, 2)
            self.assertEqual(hd_bundle.val_samples, 1)
            self.assertEqual(
                hd_bundle.train_timestamps,
                ["2022-12-30T13:30:00Z", "2022-12-30T13:40:00Z"],
            )
            self.assertEqual(hd_bundle.val_timestamps, ["2022-12-30T13:50:00Z"])
            current_surface, text_embedding, target_surface = next(iter(hd_bundle.train_loader))
            self.assertEqual(tuple(current_surface.shape[1:]), (1, 16, 16))
            self.assertEqual(tuple(target_surface.shape[1:]), (1, 16, 16))
            self.assertEqual(int(text_embedding.shape[1]), 2)

            lp_bundle = create_vol_surface_xlsx_dataloaders(
                Config(
                    data_path=str(workbook_path),
                    sheet_name="gan_input_ready",
                    text_embedding_mode="lp",
                    batch_size=8,
                    train_ratio=2 / 3,
                    num_workers=0,
                    cuda=False,
                )
            )
            self.assertEqual(lp_bundle.embedding_dim, 3)

            concat_bundle = create_vol_surface_xlsx_dataloaders(
                Config(
                    data_path=str(workbook_path),
                    sheet_name="gan_input_ready",
                    text_embedding_mode="concat",
                    batch_size=8,
                    train_ratio=2 / 3,
                    num_workers=0,
                    cuda=False,
                )
            )
            self.assertEqual(concat_bundle.embedding_dim, 5)

    def test_svi_loader_pairs_backward_forward_and_fits_train_only_normalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_svi_workbook(tmpdir)

            bundle = create_svi_xlsx_dataloaders(
                Config(
                    data_path=str(workbook_path),
                    sheet_name="news_direction_audit",
                    text_embedding_mode="concat",
                    batch_size=8,
                    train_ratio=2 / 3,
                    num_workers=0,
                    cuda=False,
                    max_slices=4,
                )
            )

            self.assertEqual(bundle.embedding_dim, 5)
            self.assertEqual(bundle.train_samples, 2)
            self.assertEqual(bundle.val_samples, 1)
            self.assertEqual(bundle.current_input_dim, 29)
            self.assertEqual(bundle.regression_dim, 24)
            self.assertEqual(
                bundle.train_timestamps,
                ["2022-12-30T13:30:00Z", "2022-12-30T13:40:00Z"],
            )
            self.assertEqual(bundle.val_timestamps, ["2022-12-30T13:50:00Z"])
            self.assertAlmostEqual(bundle.normalization_stats["mean"][0], 28.666666, places=4)

            current_features, text_embedding, future_regression, future_mask, future_count = next(iter(bundle.train_loader))
            self.assertEqual(int(current_features.shape[1]), 29)
            self.assertEqual(int(text_embedding.shape[1]), 5)
            self.assertEqual(int(future_regression.shape[1]), 24)
            self.assertEqual(int(future_mask.shape[1]), 4)
            self.assertEqual(future_count.dtype, torch.int64)

    def test_shared_wgan_config_loads_constraint_switches_and_cli_overrides(self):
        config = load_config(config_path="configs/wgan/train_default.yaml")
        self.assertTrue(config.use_calendar_constraint)
        self.assertTrue(config.use_butterfly_constraint)
        self.assertTrue(config.use_smooth_constraint)
        self.assertFalse(config.use_early_stopping)
        self.assertEqual(config.early_stopping_patience, 10)
        self.assertEqual(config.early_stopping_min_delta, 0.0)
        self.assertFalse(config.use_reduce_lr_on_plateau)
        self.assertEqual(config.reduce_lr_factor, 0.5)
        self.assertEqual(config.reduce_lr_patience, 8)
        self.assertEqual(config.reduce_lr_min_lr, 1e-5)

        overrides = parse_cli_overrides(
            [
                "use_calendar_constraint=false",
                "use_butterfly_constraint=true",
                "use_smooth_constraint=off",
                "use_early_stopping=true",
                "early_stopping_patience=15",
                "early_stopping_min_delta=0.05",
                "use_reduce_lr_on_plateau=true",
                "reduce_lr_factor=0.25",
                "reduce_lr_patience=3",
                "reduce_lr_min_lr=1e-6",
            ]
        )
        overridden = load_config(config_path="configs/wgan/train_default.yaml", overrides=overrides)
        self.assertFalse(overridden.use_calendar_constraint)
        self.assertTrue(overridden.use_butterfly_constraint)
        self.assertFalse(overridden.use_smooth_constraint)
        self.assertTrue(overridden.use_early_stopping)
        self.assertEqual(overridden.early_stopping_patience, 15)
        self.assertEqual(overridden.early_stopping_min_delta, 0.05)
        self.assertTrue(overridden.use_reduce_lr_on_plateau)
        self.assertEqual(overridden.reduce_lr_factor, 0.25)
        self.assertEqual(overridden.reduce_lr_patience, 3)
        self.assertEqual(overridden.reduce_lr_min_lr, 1e-6)

        with self.assertRaises(ValueError):
            parse_cli_overrides(["use_unknown_constraint=false"])

    def test_wgan_generator_loss_switches_disable_selected_constraints_only(self):
        config = Config(
            cuda=False,
            learning_rate=0.0,
            lambda_recon=0.0,
            lambda_calendar=2.0,
            lambda_butterfly=1.5,
            lambda_smooth=0.25,
            use_calendar_constraint=False,
            use_butterfly_constraint=True,
            use_smooth_constraint=False,
            noise_dim=4,
            gen_hidden_dim=8,
            disc_hidden_dim=8,
        )
        model = _build_test_wgan(config)
        model.calendar_arbitrage_penalty = Mock(return_value=torch.tensor(1.5, device=model.device))
        model.butterfly_arbitrage_penalty = Mock(return_value=torch.tensor(2.0, device=model.device))
        model.smoothness_penalty = Mock(return_value=torch.tensor(3.0, device=model.device))

        stats = model._generator_step(
            current_surface=torch.zeros((1, 1, 2, 2), device=model.device),
            text_embedding=torch.zeros((1, 2), device=model.device),
            real_future=torch.zeros((1, 1, 2, 2), device=model.device),
        )

        self.assertAlmostEqual(stats["g_calendar"], 1.5, places=6)
        self.assertAlmostEqual(stats["g_butterfly"], 2.0, places=6)
        self.assertAlmostEqual(stats["g_smooth"], 3.0, places=6)
        expected_total = stats["g_adv"] + config.lambda_butterfly * stats["g_butterfly"]
        self.assertAlmostEqual(stats["g_total"], expected_total, places=6)

    def test_train_vol_script_dry_run_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_vol_workbook(tmpdir)
            metrics_dir = Path(tmpdir) / "vol_metrics"
            models_dir = Path(tmpdir) / "vol_models"
            config_path = Path(tmpdir) / "train_vol.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_path: {workbook_path}",
                        "sheet_name: gan_input_ready",
                        "text_embedding_mode: concat",
                        "train_ratio: 0.67",
                        "batch_size: 2",
                        "num_workers: 0",
                        "cuda: false",
                        f"models_path: {models_dir}",
                        f"outputs_path: {models_dir}",
                        f"samples_path: {Path(tmpdir) / 'vol_samples'}",
                        f"metrics_path: {metrics_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            module = _load_script_module(ROOT_DIR / "scripts/train/train_vol.py", "train_vol_script")
            module.main(["--config", str(config_path), "--dry-run"])

            self.assertTrue(any(path.name.startswith("run_config_") for path in metrics_dir.iterdir()))
            self.assertFalse((metrics_dir / "loss_curves.png").exists())
            self.assertFalse((metrics_dir / "training_metrics.csv").exists())
            self.assertFalse((metrics_dir / "best_checkpoint.json").exists())
            self.assertFalse((models_dir / "generator_best.pt").exists())
            self.assertFalse((models_dir / "discriminator_best.pt").exists())

    def test_train_vol_script_full_run_saves_loss_curves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_vol_workbook(tmpdir)
            metrics_dir = Path(tmpdir) / "vol_metrics"
            models_dir = Path(tmpdir) / "vol_models"
            config_path = Path(tmpdir) / "train_vol.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_path: {workbook_path}",
                        "sheet_name: gan_input_ready",
                        "text_embedding_mode: concat",
                        "train_ratio: 0.67",
                        "batch_size: 2",
                        "num_epochs: 1",
                        "save_every: 1",
                        "discriminator_iter: 1",
                        "noise_dim: 8",
                        "gen_hidden_dim: 32",
                        "disc_hidden_dim: 16",
                        "num_workers: 0",
                        "cuda: false",
                        "use_calendar_constraint: false",
                        f"models_path: {models_dir}",
                        f"outputs_path: {models_dir}",
                        f"samples_path: {Path(tmpdir) / 'vol_samples'}",
                        f"metrics_path: {metrics_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            module = _load_script_module(ROOT_DIR / "scripts/train/train_vol.py", "train_vol_full_script")
            module.main(["--config", str(config_path)])

            plot_path = metrics_dir / "loss_curves.png"
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)
            metrics_rows = json.loads((metrics_dir / "training_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics_rows)
            self.assertIn("g_calendar", metrics_rows[0])
            self.assertIn("val_calendar", metrics_rows[0])
            self.assertIn("g_lr", metrics_rows[0])
            self.assertIn("d_lr", metrics_rows[0])
            csv_path = metrics_dir / "training_metrics.csv"
            self.assertTrue(csv_path.exists())
            with csv_path.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 1)
            self.assertIn("g_lr", csv_rows[0])
            self.assertIn("d_lr", csv_rows[0])
            best_path = metrics_dir / "best_checkpoint.json"
            self.assertTrue(best_path.exists())
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))
            self.assertEqual(best_payload["monitor_metric"], "val_recon")
            self.assertEqual(best_payload["best_epoch"], 1)
            self.assertTrue((models_dir / "generator_best.pt").exists())
            self.assertTrue((models_dir / "discriminator_best.pt").exists())

    def test_train_svi_script_dry_run_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_svi_workbook(tmpdir)
            metrics_dir = Path(tmpdir) / "svi_metrics"
            models_dir = Path(tmpdir) / "svi_models"
            stats_path = metrics_dir / "normalization_stats.json"
            config_path = Path(tmpdir) / "train_svi.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_path: {workbook_path}",
                        "sheet_name: news_direction_audit",
                        "text_embedding_mode: hd",
                        "train_ratio: 0.67",
                        "batch_size: 2",
                        "num_workers: 0",
                        "cuda: false",
                        "max_slices: 4",
                        f"models_path: {models_dir}",
                        f"outputs_path: {models_dir}",
                        f"samples_path: {Path(tmpdir) / 'svi_samples'}",
                        f"metrics_path: {metrics_dir}",
                        f"normalization_stats_path: {stats_path}",
                    ]
                ),
                encoding="utf-8",
            )

            module = _load_script_module(ROOT_DIR / "scripts/train/train_svi.py", "train_svi_script")
            module.main(["--config", str(config_path), "--dry-run"])

            self.assertTrue(stats_path.exists())
            self.assertTrue(any(path.name.startswith("run_config_") for path in metrics_dir.iterdir()))
            self.assertFalse((metrics_dir / "loss_curves.png").exists())
            self.assertFalse((metrics_dir / "training_metrics.csv").exists())
            self.assertFalse((metrics_dir / "best_checkpoint.json").exists())
            self.assertFalse((models_dir / "svi_regressor_best.pt").exists())

    def test_train_svi_script_full_run_saves_loss_curves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_svi_workbook(tmpdir)
            metrics_dir = Path(tmpdir) / "svi_metrics"
            models_dir = Path(tmpdir) / "svi_models"
            stats_path = metrics_dir / "normalization_stats.json"
            config_path = Path(tmpdir) / "train_svi.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_path: {workbook_path}",
                        "sheet_name: news_direction_audit",
                        "text_embedding_mode: hd",
                        "train_ratio: 0.67",
                        "batch_size: 2",
                        "num_epochs: 1",
                        "save_every: 1",
                        "svi_hidden_dim: 16",
                        "num_workers: 0",
                        "cuda: false",
                        "max_slices: 4",
                        f"models_path: {models_dir}",
                        f"outputs_path: {models_dir}",
                        f"samples_path: {Path(tmpdir) / 'svi_samples'}",
                        f"metrics_path: {metrics_dir}",
                        f"normalization_stats_path: {stats_path}",
                    ]
                ),
                encoding="utf-8",
            )

            module = _load_script_module(ROOT_DIR / "scripts/train/train_svi.py", "train_svi_full_script")
            module.main(["--config", str(config_path)])

            plot_path = metrics_dir / "loss_curves.png"
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)
            metrics_rows = json.loads((metrics_dir / "training_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics_rows)
            self.assertIn("lr", metrics_rows[0])
            csv_path = metrics_dir / "training_metrics.csv"
            self.assertTrue(csv_path.exists())
            with csv_path.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 1)
            self.assertIn("lr", csv_rows[0])
            best_path = metrics_dir / "best_checkpoint.json"
            self.assertTrue(best_path.exists())
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))
            self.assertEqual(best_payload["monitor_metric"], "val_regression")
            self.assertEqual(best_payload["best_epoch"], 1)
            self.assertTrue((models_dir / "svi_regressor_best.pt").exists())

    def test_wgan_early_stopping_saves_best_checkpoint_before_last_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir) / "metrics"
            models_dir = Path(tmpdir) / "models"
            config = Config(
                cuda=False,
                learning_rate=0.1,
                num_epochs=5,
                batch_size=1,
                discriminator_iter=1,
                noise_dim=4,
                gen_hidden_dim=8,
                disc_hidden_dim=8,
                models_path=str(models_dir),
                outputs_path=str(models_dir),
                metrics_path=str(metrics_dir),
                save_every=10,
                use_early_stopping=True,
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
                use_reduce_lr_on_plateau=True,
                reduce_lr_factor=0.5,
                reduce_lr_patience=0,
                reduce_lr_min_lr=0.01,
            )
            model = _build_test_wgan(config)
            train_loader = _build_small_wgan_loader()
            val_loader = _build_small_wgan_loader()
            model._evaluate = Mock(
                side_effect=[
                    {"val_recon": 0.20, "val_calendar": 0.01, "val_butterfly": 0.01},
                    {"val_recon": 0.25, "val_calendar": 0.01, "val_butterfly": 0.01},
                    {"val_recon": 0.30, "val_calendar": 0.01, "val_butterfly": 0.01},
                ]
            )

            model.train(train_loader, val_loader)

            metrics_rows = json.loads((metrics_dir / "training_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metrics_rows), 3)
            self.assertAlmostEqual(metrics_rows[0]["g_lr"], 0.1, places=6)
            self.assertAlmostEqual(metrics_rows[0]["d_lr"], 0.1, places=6)
            self.assertAlmostEqual(metrics_rows[2]["g_lr"], 0.05, places=6)
            self.assertAlmostEqual(metrics_rows[2]["d_lr"], 0.05, places=6)
            with (metrics_dir / "training_metrics.csv").open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(csv_rows[2]["g_lr"]), 0.05, places=6)
            self.assertAlmostEqual(float(csv_rows[2]["d_lr"]), 0.05, places=6)
            best_payload = json.loads((metrics_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(best_payload["monitor_metric"], "val_recon")
            self.assertEqual(best_payload["best_epoch"], 1)
            self.assertAlmostEqual(best_payload["best_metric"], 0.20, places=6)
            self.assertTrue((models_dir / "generator_best.pt").exists())
            self.assertTrue((models_dir / "discriminator_best.pt").exists())
            self.assertTrue((models_dir / "generator.pt").exists())
            self.assertTrue((models_dir / "discriminator.pt").exists())
            self.assertAlmostEqual(model.g_optimizer.param_groups[0]["lr"], 0.025, places=6)
            self.assertAlmostEqual(model.d_optimizer.param_groups[0]["lr"], 0.025, places=6)

    def test_wgan_training_without_validation_skips_best_checkpoint_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir) / "metrics"
            models_dir = Path(tmpdir) / "models"
            config = Config(
                cuda=False,
                learning_rate=0.1,
                num_epochs=2,
                batch_size=1,
                discriminator_iter=1,
                noise_dim=4,
                gen_hidden_dim=8,
                disc_hidden_dim=8,
                models_path=str(models_dir),
                outputs_path=str(models_dir),
                metrics_path=str(metrics_dir),
                save_every=10,
                use_early_stopping=True,
                early_stopping_patience=1,
                use_reduce_lr_on_plateau=True,
                reduce_lr_factor=0.5,
                reduce_lr_patience=0,
                reduce_lr_min_lr=0.01,
            )
            model = _build_test_wgan(config)

            model.train(_build_small_wgan_loader(), None)

            metrics_rows = json.loads((metrics_dir / "training_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metrics_rows), 2)
            self.assertTrue((metrics_dir / "training_metrics.csv").exists())
            self.assertTrue(all(abs(row["g_lr"] - 0.1) < 1e-9 for row in metrics_rows))
            self.assertTrue(all(abs(row["d_lr"] - 0.1) < 1e-9 for row in metrics_rows))
            self.assertFalse((metrics_dir / "best_checkpoint.json").exists())
            self.assertFalse((models_dir / "generator_best.pt").exists())
            self.assertFalse((models_dir / "discriminator_best.pt").exists())
            self.assertAlmostEqual(model.g_optimizer.param_groups[0]["lr"], 0.1, places=6)
            self.assertAlmostEqual(model.d_optimizer.param_groups[0]["lr"], 0.1, places=6)

    def test_svi_early_stopping_saves_best_checkpoint_before_last_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_svi_workbook(tmpdir)
            metrics_dir = Path(tmpdir) / "metrics"
            models_dir = Path(tmpdir) / "models"
            config = Config(
                data_path=str(workbook_path),
                sheet_name="news_direction_audit",
                text_embedding_mode="hd",
                train_ratio=0.67,
                batch_size=2,
                num_epochs=5,
                learning_rate=0.1,
                num_workers=0,
                cuda=False,
                max_slices=4,
                svi_hidden_dim=16,
                models_path=str(models_dir),
                outputs_path=str(models_dir),
                metrics_path=str(metrics_dir),
                normalization_stats_path=str(metrics_dir / "normalization_stats.json"),
                use_early_stopping=True,
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
                use_reduce_lr_on_plateau=True,
                reduce_lr_factor=0.5,
                reduce_lr_patience=0,
                reduce_lr_min_lr=0.01,
            )
            trainer = SviXlsxTrainer(config)
            trainer._run_epoch = Mock(
                side_effect=[
                    {"train_total": 1.0, "train_regression": 0.8, "train_count": 0.2},
                    {"val_total": 0.6, "val_regression": 0.5, "val_count": 0.1},
                    {"train_total": 0.9, "train_regression": 0.7, "train_count": 0.2},
                    {"val_total": 0.7, "val_regression": 0.6, "val_count": 0.1},
                    {"train_total": 0.85, "train_regression": 0.65, "train_count": 0.2},
                    {"val_total": 0.8, "val_regression": 0.7, "val_count": 0.1},
                ]
            )

            trainer.start_train()

            metrics_rows = json.loads((metrics_dir / "training_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metrics_rows), 3)
            self.assertAlmostEqual(metrics_rows[0]["lr"], 0.1, places=6)
            self.assertAlmostEqual(metrics_rows[2]["lr"], 0.05, places=6)
            with (metrics_dir / "training_metrics.csv").open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(csv_rows[2]["lr"]), 0.05, places=6)
            best_payload = json.loads((metrics_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(best_payload["monitor_metric"], "val_regression")
            self.assertEqual(best_payload["best_epoch"], 1)
            self.assertAlmostEqual(best_payload["best_metric"], 0.5, places=6)
            self.assertTrue((models_dir / "svi_regressor_best.pt").exists())
            self.assertTrue((models_dir / "svi_regressor.pt").exists())
            self.assertIsNotNone(trainer.optimizer)
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.025, places=6)

    def test_svi_training_without_validation_skips_best_checkpoint_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_svi_workbook(tmpdir)
            metrics_dir = Path(tmpdir) / "metrics"
            models_dir = Path(tmpdir) / "models"
            config = Config(
                data_path=str(workbook_path),
                sheet_name="news_direction_audit",
                text_embedding_mode="hd",
                train_ratio=1.0,
                batch_size=2,
                num_epochs=2,
                learning_rate=0.1,
                num_workers=0,
                cuda=False,
                max_slices=4,
                svi_hidden_dim=16,
                models_path=str(models_dir),
                outputs_path=str(models_dir),
                metrics_path=str(metrics_dir),
                normalization_stats_path=str(metrics_dir / "normalization_stats.json"),
                use_early_stopping=True,
                early_stopping_patience=1,
                use_reduce_lr_on_plateau=True,
                reduce_lr_factor=0.5,
                reduce_lr_patience=0,
                reduce_lr_min_lr=0.01,
            )
            trainer = SviXlsxTrainer(config)

            trainer.start_train()

            metrics_rows = json.loads((metrics_dir / "training_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metrics_rows), 2)
            self.assertTrue((metrics_dir / "training_metrics.csv").exists())
            self.assertTrue(all(abs(row["lr"] - 0.1) < 1e-9 for row in metrics_rows))
            self.assertFalse((metrics_dir / "best_checkpoint.json").exists())
            self.assertFalse((models_dir / "svi_regressor_best.pt").exists())
            self.assertIsNotNone(trainer.optimizer)
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.1, places=6)

    def test_plot_training_curves_skips_missing_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / "metrics" / "loss_curves.png"
            output_path = plot_training_curves(
                [
                    {"epoch": 1, "train_total": 1.2, "train_count": 0.8},
                    {"epoch": 2, "train_total": 0.9, "train_count": 0.6},
                ],
                title="Train-only Loss Curves",
                metric_groups=(
                    ("Primary losses", ("train_total", "val_total")),
                    ("Count losses", ("train_count", "val_count")),
                ),
                output_path=plot_path,
            )

            self.assertEqual(output_path, plot_path)
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_unified_train_main_dispatches_to_expected_subcommand(self):
        module = _load_script_module(ROOT_DIR / "scripts/train/main.py", "train_main_script")
        vol_mock = Mock()
        svi_mock = Mock()
        module.COMMANDS["vol-xlsx"] = vol_mock
        module.COMMANDS["svi-xlsx"] = svi_mock

        module.main(["vol-xlsx", "--dry-run"])
        vol_mock.assert_called_once_with(["--dry-run"])
        svi_mock.assert_not_called()

    def test_unified_train_main_prints_help_without_args(self):
        module = _load_script_module(ROOT_DIR / "scripts/train/main.py", "train_main_help_script")
        with patch("builtins.print") as mock_print:
            module.main([])
        self.assertTrue(mock_print.called)


if __name__ == "__main__":
    unittest.main()
