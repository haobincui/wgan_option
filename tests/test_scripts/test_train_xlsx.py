import importlib.util
import csv
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wgan_option.config import Config, load_config, parse_cli_overrides  # noqa: E402
from wgan_option.models.gan_model import WGAN_GP  # noqa: E402
from wgan_option.train_vol_regression_xlsx import VolSurfaceRegressionTrainer  # noqa: E402
from wgan_option.train_svi_xlsx import SviXlsxTrainer  # noqa: E402
from wgan_option.utils.merged_xlsx import (  # noqa: E402
    create_svi_xlsx_dataloaders,
    create_vol_surface_xlsx_dataloaders,
)
from wgan_option.utils.training_run_paths import prepare_timestamped_training_config  # noqa: E402
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
    def _find_only_run_dir(self, root: Path) -> Path:
        run_dirs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and re.fullmatch(r"\d{8}_\d{6}", path.name)
        )
        self.assertEqual(len(run_dirs), 1)
        return run_dirs[0]

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

    def _write_legacy_training_dir(
        self,
        outputs_root: Path,
        *,
        family: str,
        legacy_suffix: str,
        run_ts: str,
    ) -> Path:
        legacy_dir = outputs_root / f"{family}_{legacy_suffix}"
        checkpoints_dir = legacy_dir / "checkpoints"
        metrics_dir = legacy_dir / "metrics"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (checkpoints_dir / "artifact.bin").write_bytes(b"checkpoint")
        (metrics_dir / f"run_config_{run_ts}.yaml").write_text("seed: 42\n", encoding="utf-8")
        (metrics_dir / "training_metrics.json").write_text("[]\n", encoding="utf-8")
        return legacy_dir

    def _write_nested_training_run_dir(
        self,
        outputs_root: Path,
        *,
        family: str,
        run_ts: str,
    ) -> Path:
        run_dir = outputs_root / family / run_ts
        checkpoints_dir = run_dir / "checkpoints"
        metrics_dir = run_dir / "metrics"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (checkpoints_dir / "artifact.bin").write_bytes(b"checkpoint")
        (metrics_dir / f"run_config_{run_ts}.yaml").write_text("seed: 42\n", encoding="utf-8")
        (metrics_dir / "training_metrics.json").write_text("[]\n", encoding="utf-8")
        return run_dir

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
        config = load_config(config_path="configs/wgan/train_vol_xlsx.yaml")
        self.assertTrue(config.use_calendar_constraint)
        self.assertTrue(config.use_butterfly_constraint)
        self.assertTrue(config.use_smooth_constraint)
        self.assertEqual(config.constraint_warmup_epochs, 0)
        self.assertEqual(config.lambda_delta_shrink, 0.0)
        self.assertEqual(config.best_checkpoint_metric, "val_recon")
        self.assertEqual(config.baseline_penalty_weight, 2.0)
        self.assertFalse(config.use_early_stopping)
        self.assertEqual(config.early_stopping_patience, 10)
        self.assertEqual(config.early_stopping_min_delta, 0.0)
        self.assertTrue(config.use_reduce_lr_on_plateau)
        self.assertEqual(config.reduce_lr_factor, 0.5)
        self.assertEqual(config.reduce_lr_patience, 8)
        self.assertEqual(config.reduce_lr_min_lr, 1e-5)

        overrides = parse_cli_overrides(
            [
                "use_calendar_constraint=false",
                "use_butterfly_constraint=true",
                "use_smooth_constraint=false",
                "constraint_warmup_epochs=20",
                "lambda_delta_shrink=0.05",
                "best_checkpoint_metric=val_hybrid_score",
                "baseline_penalty_weight=3.0",
                "use_early_stopping=true",
                "early_stopping_patience=15",
                "early_stopping_min_delta=0.05",
                "use_reduce_lr_on_plateau=true",
                "reduce_lr_factor=0.25",
                "reduce_lr_patience=3",
                "reduce_lr_min_lr=1e-6",
            ]
        )
        overridden = load_config(config_path="configs/wgan/train_vol_xlsx.yaml", overrides=overrides)
        self.assertFalse(overridden.use_calendar_constraint)
        self.assertTrue(overridden.use_butterfly_constraint)
        self.assertFalse(overridden.use_smooth_constraint)
        self.assertEqual(overridden.constraint_warmup_epochs, 20)
        self.assertEqual(overridden.lambda_delta_shrink, 0.05)
        self.assertEqual(overridden.best_checkpoint_metric, "val_hybrid_score")
        self.assertEqual(overridden.baseline_penalty_weight, 3.0)
        self.assertTrue(overridden.use_early_stopping)
        self.assertEqual(overridden.early_stopping_patience, 15)
        self.assertEqual(overridden.early_stopping_min_delta, 0.05)
        self.assertTrue(overridden.use_reduce_lr_on_plateau)
        self.assertEqual(overridden.reduce_lr_factor, 0.25)
        self.assertEqual(overridden.reduce_lr_patience, 3)
        self.assertEqual(overridden.reduce_lr_min_lr, 1e-6)

        with self.assertRaises(ValueError):
            parse_cli_overrides(["use_unknown_constraint=false"])

        for invalid_bool in [
            "use_smooth_constraint=off",
            "use_smooth_constraint=yes",
            "use_smooth_constraint=1",
        ]:
            with self.subTest(invalid_bool=invalid_bool):
                with self.assertRaises(ValueError):
                    parse_cli_overrides([invalid_bool])

    def test_prepare_timestamped_training_config_infers_output_root_from_data_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auto_root = Path(tmpdir) / "outputs" / "training"
            config = Config(
                data_path="data/processed/cubic-excel/20260330-01/merged_vol.xlsx",
                sheet_name="gan_input_ready",
                output_root="",
            )

            with patch("wgan_option.utils.training_run_paths._AUTO_TRAINING_ROOT", auto_root):
                resolved_config, run_dir = prepare_timestamped_training_config(config)

            self.assertEqual(run_dir.parent, auto_root / "cubic-excel")
            self.assertEqual(resolved_config.output_root, str(auto_root / "cubic-excel"))
            self.assertEqual(resolved_config.models_path, str(run_dir / "checkpoints"))
            self.assertEqual(resolved_config.samples_path, str(run_dir / "samples"))
            self.assertEqual(resolved_config.metrics_path, str(run_dir / "metrics"))

    def test_prepare_timestamped_training_config_treats_legacy_processed_layout_as_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auto_root = Path(tmpdir) / "outputs" / "training"
            config = Config(
                data_path="data/processed/svi/20260330-01/merged_svi.xlsx",
                sheet_name="news_direction_audit",
                output_root="",
            )

            with patch("wgan_option.utils.training_run_paths._AUTO_TRAINING_ROOT", auto_root):
                resolved_config, run_dir = prepare_timestamped_training_config(config)

            self.assertEqual(run_dir.parent, auto_root / "svi-all")
            self.assertEqual(resolved_config.output_root, str(auto_root / "svi-all"))
            self.assertEqual(resolved_config.models_path, str(run_dir / "checkpoints"))
            self.assertEqual(resolved_config.metrics_path, str(run_dir / "metrics"))

    def test_prepare_timestamped_training_config_does_not_eagerly_create_artifact_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "training" / "vol_xlsx"
            config = Config(
                data_path="data/example.xlsx",
                sheet_name="gan_input_ready",
                output_root=str(output_root),
            )

            resolved_config, run_dir = prepare_timestamped_training_config(config)

            self.assertFalse(run_dir.exists())
            self.assertFalse(Path(resolved_config.models_path).exists())
            self.assertFalse(Path(resolved_config.metrics_path).exists())
            self.assertFalse(Path(resolved_config.samples_path).exists())

    def test_load_config_derives_training_paths_from_output_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train_vol_output_root.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "data_path: data/example.xlsx",
                        "sheet_name: gan_input_ready",
                        "output_root: outputs/training/svi-all",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path=str(config_path))

            self.assertEqual(config.output_root, "outputs/training/svi-all")
            self.assertEqual(config.models_path, "outputs/training/svi-all/checkpoints")
            self.assertEqual(config.outputs_path, "outputs/training/svi-all/checkpoints")
            self.assertEqual(config.samples_path, "outputs/training/svi-all/samples")
            self.assertEqual(config.metrics_path, "outputs/training/svi-all/metrics")
            self.assertEqual(
                config.normalization_stats_path,
                "outputs/training/svi-all/metrics/normalization_stats.json",
            )

    def test_load_config_rejects_output_root_with_explicit_legacy_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train_conflict.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "data_path: data/example.xlsx",
                        "sheet_name: gan_input_ready",
                        "metrics_path: outputs/custom_metrics",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_config(
                    config_path=str(config_path),
                    overrides={"output_root": "outputs/training/svi-all"},
                )

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

    def test_wgan_generator_loss_adds_delta_shrink_when_enabled(self):
        config = Config(
            cuda=False,
            learning_rate=0.0,
            lambda_recon=0.0,
            lambda_delta_shrink=0.5,
            use_calendar_constraint=False,
            use_butterfly_constraint=False,
            use_smooth_constraint=False,
            noise_dim=4,
            gen_hidden_dim=8,
            disc_hidden_dim=8,
        )
        model = _build_test_wgan(config)

        stats = model._generator_step(
            current_surface=torch.zeros((1, 1, 2, 2), device=model.device),
            text_embedding=torch.zeros((1, 2), device=model.device),
            real_future=torch.zeros((1, 1, 2, 2), device=model.device),
        )

        expected_total = stats["g_adv"] + config.lambda_delta_shrink * stats["g_delta_shrink"]
        self.assertGreater(stats["g_delta_shrink"], 0.0)
        self.assertAlmostEqual(stats["g_total"], expected_total, places=6)

    def test_wgan_generator_loss_constraint_warmup_skips_penalties_until_epoch_threshold(self):
        config = Config(
            cuda=False,
            learning_rate=0.0,
            lambda_recon=0.0,
            lambda_calendar=2.0,
            lambda_butterfly=1.5,
            lambda_smooth=0.25,
            use_calendar_constraint=True,
            use_butterfly_constraint=True,
            use_smooth_constraint=True,
            constraint_warmup_epochs=2,
            noise_dim=4,
            gen_hidden_dim=8,
            disc_hidden_dim=8,
        )
        model = _build_test_wgan(config)
        model.calendar_arbitrage_penalty = Mock(return_value=torch.tensor(1.5, device=model.device))
        model.butterfly_arbitrage_penalty = Mock(return_value=torch.tensor(2.0, device=model.device))
        model.smoothness_penalty = Mock(return_value=torch.tensor(3.0, device=model.device))

        warmup_stats = model._generator_step(
            current_surface=torch.zeros((1, 1, 2, 2), device=model.device),
            text_embedding=torch.zeros((1, 2), device=model.device),
            real_future=torch.zeros((1, 1, 2, 2), device=model.device),
            epoch=1,
        )
        post_warmup_stats = model._generator_step(
            current_surface=torch.zeros((1, 1, 2, 2), device=model.device),
            text_embedding=torch.zeros((1, 2), device=model.device),
            real_future=torch.zeros((1, 1, 2, 2), device=model.device),
            epoch=3,
        )

        self.assertAlmostEqual(warmup_stats["g_total"], warmup_stats["g_adv"], places=6)
        expected_post_warmup = (
            post_warmup_stats["g_adv"]
            + config.lambda_calendar * post_warmup_stats["g_calendar"]
            + config.lambda_butterfly * post_warmup_stats["g_butterfly"]
            + config.lambda_smooth * post_warmup_stats["g_smooth"]
        )
        self.assertAlmostEqual(post_warmup_stats["g_total"], expected_post_warmup, places=6)

    def test_wgan_initialization_does_not_eagerly_create_artifact_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints_dir = Path(tmpdir) / "training" / "checkpoints"
            metrics_dir = Path(tmpdir) / "training" / "metrics"
            config = Config(
                cuda=False,
                learning_rate=0.0,
                noise_dim=4,
                gen_hidden_dim=8,
                disc_hidden_dim=8,
                models_path=str(checkpoints_dir),
                outputs_path=str(checkpoints_dir),
                metrics_path=str(metrics_dir),
            )

            model = _build_test_wgan(config)

            self.assertFalse(checkpoints_dir.exists())
            self.assertFalse(metrics_dir.exists())

            model._generator_step(
                current_surface=torch.zeros((1, 1, 2, 2), device=model.device),
                text_embedding=torch.zeros((1, 2), device=model.device),
                real_future=torch.zeros((1, 1, 2, 2), device=model.device),
            )

            self.assertFalse(checkpoints_dir.exists())
            self.assertFalse(metrics_dir.exists())

    def test_train_vol_script_dry_run_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_vol_workbook(tmpdir)
            output_root = Path(tmpdir) / "training" / "vol_xlsx"
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
                        f"output_root: {output_root}",
                    ]
                ),
                encoding="utf-8",
            )

            module = _load_script_module(ROOT_DIR / "scripts/train/train_vol.py", "train_vol_script")
            module.main(["--config", str(config_path), "--dry-run"])

            run_dir = self._find_only_run_dir(output_root)
            metrics_dir = run_dir / "metrics"
            models_dir = run_dir / "checkpoints"
            samples_dir = run_dir / "samples"
            self.assertTrue(samples_dir.exists())
            self.assertTrue((metrics_dir / f"run_config_{run_dir.name}.yaml").exists())
            self.assertFalse((metrics_dir / "loss_curves.png").exists())
            self.assertFalse((metrics_dir / "training_metrics.csv").exists())
            self.assertFalse((metrics_dir / "best_checkpoint.json").exists())
            self.assertFalse((models_dir / "generator_best.pt").exists())
            self.assertFalse((models_dir / "discriminator_best.pt").exists())

    def test_train_vol_script_full_run_saves_loss_curves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_vol_workbook(tmpdir)
            output_root = Path(tmpdir) / "training" / "vol_xlsx"
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
                        f"output_root: {output_root}",
                    ]
                ),
                encoding="utf-8",
            )

            module = _load_script_module(ROOT_DIR / "scripts/train/train_vol.py", "train_vol_full_script")
            module.main(["--config", str(config_path)])

            run_dir = self._find_only_run_dir(output_root)
            metrics_dir = run_dir / "metrics"
            models_dir = run_dir / "checkpoints"
            samples_dir = run_dir / "samples"
            self.assertTrue(samples_dir.exists())
            self.assertTrue((metrics_dir / f"run_config_{run_dir.name}.yaml").exists())
            plot_path = metrics_dir / "loss_curves.png"
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)
            metrics_rows = json.loads((metrics_dir / "training_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics_rows)
            self.assertIn("g_calendar", metrics_rows[0])
            self.assertIn("val_calendar", metrics_rows[0])
            self.assertIn("g_delta_shrink", metrics_rows[0])
            self.assertIn("val_current_recon", metrics_rows[0])
            self.assertIn("val_hybrid_score", metrics_rows[0])
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
            self.assertIn("val_current_recon", best_payload["metrics"])
            self.assertTrue((models_dir / "generator_best.pt").exists())
            self.assertTrue((models_dir / "discriminator_best.pt").exists())
            fallback_path = metrics_dir / "fallback_calibration.json"
            self.assertTrue(fallback_path.exists())
            fallback_payload = json.loads(fallback_path.read_text(encoding="utf-8"))
            self.assertIn("uncertainty_threshold", fallback_payload)
            self.assertEqual(fallback_payload["mc_samples"], 5)

    def test_train_vol_regression_script_dry_run_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_vol_workbook(tmpdir)
            output_root = Path(tmpdir) / "training" / "vol_regression_xlsx"
            config_path = Path(tmpdir) / "train_vol_regression.yaml"
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
                        "best_checkpoint_metric: val_hybrid_score",
                        f"output_root: {output_root}",
                    ]
                ),
                encoding="utf-8",
            )

            module = _load_script_module(ROOT_DIR / "scripts/train/train_vol_regression.py", "train_vol_regression_script")
            module.main(["--config", str(config_path), "--dry-run"])

            run_dir = self._find_only_run_dir(output_root)
            metrics_dir = run_dir / "metrics"
            models_dir = run_dir / "checkpoints"
            samples_dir = run_dir / "samples"
            self.assertTrue(samples_dir.exists())
            self.assertTrue((metrics_dir / f"run_config_{run_dir.name}.yaml").exists())
            self.assertFalse((metrics_dir / "training_metrics.csv").exists())
            self.assertFalse((metrics_dir / "best_checkpoint.json").exists())
            self.assertFalse((models_dir / "vol_regressor_best.pt").exists())

    def test_train_vol_regression_script_full_run_saves_loss_curves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_vol_workbook(tmpdir)
            output_root = Path(tmpdir) / "training" / "vol_regression_xlsx"
            config_path = Path(tmpdir) / "train_vol_regression.yaml"
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
                        "gen_hidden_dim: 32",
                        "num_workers: 0",
                        "cuda: false",
                        "lambda_delta_shrink: 0.05",
                        "best_checkpoint_metric: val_hybrid_score",
                        f"output_root: {output_root}",
                    ]
                ),
                encoding="utf-8",
            )

            module = _load_script_module(
                ROOT_DIR / "scripts/train/train_vol_regression.py",
                "train_vol_regression_full_script",
            )
            module.main(["--config", str(config_path)])

            run_dir = self._find_only_run_dir(output_root)
            metrics_dir = run_dir / "metrics"
            models_dir = run_dir / "checkpoints"
            self.assertTrue((metrics_dir / "loss_curves.png").exists())
            metrics_rows = json.loads((metrics_dir / "training_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics_rows)
            self.assertIn("train_delta_shrink", metrics_rows[0])
            self.assertIn("val_current_recon", metrics_rows[0])
            self.assertIn("val_hybrid_score", metrics_rows[0])
            best_payload = json.loads((metrics_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(best_payload["monitor_metric"], "val_hybrid_score")
            self.assertTrue((models_dir / "vol_regressor_best.pt").exists())

    def test_train_svi_script_dry_run_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_svi_workbook(tmpdir)
            output_root = Path(tmpdir) / "training" / "svi_xlsx"
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
                        f"output_root: {output_root}",
                    ]
                ),
                encoding="utf-8",
            )

            module = _load_script_module(ROOT_DIR / "scripts/train/train_svi.py", "train_svi_script")
            module.main(["--config", str(config_path), "--dry-run"])

            run_dir = self._find_only_run_dir(output_root)
            metrics_dir = run_dir / "metrics"
            models_dir = run_dir / "checkpoints"
            samples_dir = run_dir / "samples"
            stats_path = metrics_dir / "normalization_stats.json"
            self.assertTrue(samples_dir.exists())
            self.assertTrue(stats_path.exists())
            self.assertTrue((metrics_dir / f"run_config_{run_dir.name}.yaml").exists())
            self.assertFalse((metrics_dir / "loss_curves.png").exists())
            self.assertFalse((metrics_dir / "training_metrics.csv").exists())
            self.assertFalse((metrics_dir / "best_checkpoint.json").exists())
            self.assertFalse((models_dir / "svi_regressor_best.pt").exists())

    def test_train_svi_script_full_run_saves_loss_curves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_svi_workbook(tmpdir)
            output_root = Path(tmpdir) / "training" / "svi_xlsx"
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
                        f"output_root: {output_root}",
                    ]
                ),
                encoding="utf-8",
            )

            module = _load_script_module(ROOT_DIR / "scripts/train/train_svi.py", "train_svi_full_script")
            module.main(["--config", str(config_path)])

            run_dir = self._find_only_run_dir(output_root)
            metrics_dir = run_dir / "metrics"
            models_dir = run_dir / "checkpoints"
            samples_dir = run_dir / "samples"
            self.assertTrue(samples_dir.exists())
            self.assertTrue((metrics_dir / f"run_config_{run_dir.name}.yaml").exists())
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
            run_dir = Path(tmpdir) / "training" / "vol_xlsx" / "20260410_010101"
            metrics_dir = run_dir / "metrics"
            models_dir = run_dir / "checkpoints"
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

    def test_wgan_best_checkpoint_can_monitor_hybrid_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "training" / "vol_xlsx" / "20260410_010103"
            metrics_dir = run_dir / "metrics"
            models_dir = run_dir / "checkpoints"
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
                best_checkpoint_metric="val_hybrid_score",
            )
            model = _build_test_wgan(config)
            model._evaluate = Mock(
                side_effect=[
                    {
                        "val_recon": 0.20,
                        "val_current_recon": 0.18,
                        "val_baseline_gap": 0.02,
                        "val_hybrid_score": 0.24,
                        "val_calendar": 0.01,
                        "val_butterfly": 0.01,
                        "val_delta_shrink": 0.10,
                    },
                    {
                        "val_recon": 0.19,
                        "val_current_recon": 0.19,
                        "val_baseline_gap": 0.0,
                        "val_hybrid_score": 0.19,
                        "val_calendar": 0.01,
                        "val_butterfly": 0.01,
                        "val_delta_shrink": 0.08,
                    },
                ]
            )

            model.train(_build_small_wgan_loader(), _build_small_wgan_loader())

            best_payload = json.loads((metrics_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(best_payload["monitor_metric"], "val_hybrid_score")
            self.assertEqual(best_payload["best_epoch"], 2)
            self.assertAlmostEqual(best_payload["best_metric"], 0.19, places=6)

    def test_wgan_training_without_validation_skips_best_checkpoint_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "training" / "vol_xlsx" / "20260410_010102"
            metrics_dir = run_dir / "metrics"
            models_dir = run_dir / "checkpoints"
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
            output_root = Path(tmpdir) / "training" / "svi_xlsx"
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
                output_root=str(output_root),
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

            run_dir = self._find_only_run_dir(output_root)
            metrics_dir = run_dir / "metrics"
            models_dir = run_dir / "checkpoints"
            samples_dir = run_dir / "samples"
            metrics_rows = json.loads((metrics_dir / "training_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metrics_rows), 3)
            self.assertTrue(samples_dir.exists())
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
            output_root = Path(tmpdir) / "training" / "svi_xlsx"
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
                output_root=str(output_root),
                use_early_stopping=True,
                early_stopping_patience=1,
                use_reduce_lr_on_plateau=True,
                reduce_lr_factor=0.5,
                reduce_lr_patience=0,
                reduce_lr_min_lr=0.01,
            )
            trainer = SviXlsxTrainer(config)

            trainer.start_train()

            run_dir = self._find_only_run_dir(output_root)
            metrics_dir = run_dir / "metrics"
            models_dir = run_dir / "checkpoints"
            samples_dir = run_dir / "samples"
            metrics_rows = json.loads((metrics_dir / "training_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metrics_rows), 2)
            self.assertTrue(samples_dir.exists())
            self.assertTrue((metrics_dir / "training_metrics.csv").exists())
            self.assertTrue(all(abs(row["lr"] - 0.1) < 1e-9 for row in metrics_rows))
            self.assertFalse((metrics_dir / "best_checkpoint.json").exists())
            self.assertFalse((models_dir / "svi_regressor_best.pt").exists())
            self.assertIsNotNone(trainer.optimizer)
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.1, places=6)

    def test_migrate_training_outputs_moves_legacy_runs_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_root = Path(tmpdir) / "outputs"
            legacy_vol = self._write_legacy_training_dir(
                outputs_root,
                family="vol_xlsx",
                legacy_suffix="20260406-04",
                run_ts="20260406_151145",
            )
            nested_svi = self._write_nested_training_run_dir(
                outputs_root,
                family="svi_xlsx",
                run_ts="20260406_131051",
            )
            skipped_vol = self._write_nested_training_run_dir(
                outputs_root,
                family="vol_xlsx",
                run_ts="20260406_123402",
            )
            existing_destination = outputs_root / "training" / "vol_xlsx" / "20260406_123402"
            existing_destination.mkdir(parents=True, exist_ok=True)

            module = _load_script_module(
                ROOT_DIR / "scripts/train/migrate_training_outputs.py",
                "migrate_training_outputs_script",
            )
            migrated = module.main(["--outputs-root", str(outputs_root)])

            vol_target = outputs_root / "training" / "vol_xlsx" / "20260406_151145"
            svi_target = outputs_root / "training" / "svi_xlsx" / "20260406_131051"
            self.assertEqual(sorted(path.name for path in migrated), ["20260406_131051", "20260406_151145"])
            self.assertFalse(legacy_vol.exists())
            self.assertFalse(nested_svi.exists())
            self.assertTrue((vol_target / "checkpoints" / "artifact.bin").exists())
            self.assertTrue((svi_target / "metrics" / "training_metrics.json").exists())
            self.assertTrue((vol_target / "samples").exists())
            self.assertTrue((svi_target / "samples").exists())
            self.assertTrue(skipped_vol.exists())

            rerun = module.main(["--outputs-root", str(outputs_root)])
            self.assertEqual(rerun, [])

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
