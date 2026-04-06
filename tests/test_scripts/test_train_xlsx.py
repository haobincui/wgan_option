import importlib.util
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

        overrides = parse_cli_overrides(
            [
                "use_calendar_constraint=false",
                "use_butterfly_constraint=true",
                "use_smooth_constraint=off",
            ]
        )
        overridden = load_config(config_path="configs/wgan/train_default.yaml", overrides=overrides)
        self.assertFalse(overridden.use_calendar_constraint)
        self.assertTrue(overridden.use_butterfly_constraint)
        self.assertFalse(overridden.use_smooth_constraint)

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
