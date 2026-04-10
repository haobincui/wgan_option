import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wgan_option.generate_result_runtime import (  # noqa: E402
    resolve_checkpoint_path,
    select_samples_from_split,
)
from wgan_option.config import Config  # noqa: E402
from wgan_option.models.generator import Generator  # noqa: E402
from wgan_option.models.svi_regressor import SviRegressor  # noqa: E402
from wgan_option.result_config import (  # noqa: E402
    GenerateResultConfig,
    load_generate_result_config,
    parse_generate_result_overrides,
)
from wgan_option.utils.merged_xlsx import load_vol_surface_samples, select_ordered_split  # noqa: E402


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


def _write_yaml(path: Path, payload: dict):
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return path


class TestGenerateResultScripts(unittest.TestCase):
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
                row(3, "backward", "2022-12-30T13:50:00Z", [50], [0.5], [0.6], [0.1], [0.0], [0.2]),
                row(3, "forward", "2022-12-30T13:55:00Z", [55], [0.55], [0.65], [0.1], [0.0], [0.2]),
            ]
        )
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="news_direction_audit", index=False)
        return path

    def _write_generator_checkpoint(
        self,
        tmpdir: str,
        *,
        embedding_dim: int = 2,
        text_embedding_mode: str = "hd",
    ) -> tuple[Path, Path, Path]:
        models_dir = Path(tmpdir) / "vol_models"
        metrics_dir = Path(tmpdir) / "vol_metrics"
        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(7)
        config = Config(
            cuda=False,
            channels=1,
            embedding_dim=embedding_dim,
            text_embedding_mode=text_embedding_mode,
            noise_dim=4,
            gen_hidden_dim=16,
            disc_hidden_dim=8,
            models_path=str(models_dir),
            metrics_path=str(metrics_dir),
        )
        generator = Generator(
            channels=1,
            embedding_dim=embedding_dim,
            noise_dim=config.noise_dim,
            surface_height=16,
            surface_width=16,
            hidden_dim=config.gen_hidden_dim,
        )
        generator_path = models_dir / "generator.pt"
        generator_best_path = models_dir / "generator_best.pt"
        payload = {"state_dict": generator.state_dict(), "config": asdict(config), "embedding_dim": embedding_dim}
        torch.save(payload, generator_path)
        torch.save(payload, generator_best_path)
        (metrics_dir / "best_checkpoint.json").write_text(
            json.dumps(
                {
                    "monitor_metric": "val_recon",
                    "best_epoch": 1,
                    "best_metric": 0.1,
                    "artifacts": {"generator": str(generator_best_path)},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return models_dir, metrics_dir, generator_path

    def _write_generator_checkpoint_run_root(
        self,
        tmpdir: str,
        *,
        run_ts: str,
        embedding_dim: int = 2,
        text_embedding_mode: str = "hd",
    ) -> tuple[Path, Path]:
        run_root = Path(tmpdir) / "vol_xlsx"
        run_dir = run_root / run_ts
        checkpoints_dir = run_dir / "checkpoints"
        metrics_dir = run_dir / "metrics"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(7)
        config = Config(
            cuda=False,
            channels=1,
            embedding_dim=embedding_dim,
            text_embedding_mode=text_embedding_mode,
            noise_dim=4,
            gen_hidden_dim=16,
            disc_hidden_dim=8,
            models_path=str(checkpoints_dir),
            metrics_path=str(metrics_dir),
        )
        generator = Generator(
            channels=1,
            embedding_dim=embedding_dim,
            noise_dim=config.noise_dim,
            surface_height=16,
            surface_width=16,
            hidden_dim=config.gen_hidden_dim,
        )
        generator_best_path = checkpoints_dir / "generator_best.pt"
        payload = {"state_dict": generator.state_dict(), "config": asdict(config), "embedding_dim": embedding_dim}
        torch.save(payload, generator_best_path)
        (metrics_dir / "best_checkpoint.json").write_text(
            json.dumps(
                {
                    "monitor_metric": "val_recon",
                    "best_epoch": 1,
                    "best_metric": 0.1,
                    "artifacts": {"generator": str(generator_best_path)},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return run_root, run_dir

    def _write_svi_checkpoint(self, tmpdir: str, *, embedding_dim: int = 2) -> tuple[Path, Path]:
        models_dir = Path(tmpdir) / "svi_models"
        metrics_dir = Path(tmpdir) / "svi_metrics"
        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(11)
        model = SviRegressor(
            current_input_dim=29,
            embedding_dim=embedding_dim,
            regression_dim=24,
            count_classes=4,
            hidden_dim=16,
            dropout=0.0,
        )
        for parameter in model.parameters():
            torch.nn.init.constant_(parameter, 0.0)

        checkpoint_path = models_dir / "svi_regressor_best.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {"svi_hidden_dim": 16, "svi_dropout": 0.0},
                "embedding_dim": embedding_dim,
                "current_input_dim": 29,
                "regression_dim": 24,
                "max_slices": 4,
                "normalization_stats": {
                    "feature_order": ["business_days", "a", "b", "rho", "m", "sigma"],
                    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                },
            },
            checkpoint_path,
        )
        (metrics_dir / "best_checkpoint.json").write_text(
            json.dumps(
                {
                    "monitor_metric": "val_regression",
                    "best_epoch": 1,
                    "best_metric": 0.1,
                    "artifacts": {"model": str(checkpoint_path)},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return models_dir, metrics_dir

    def _write_svi_checkpoint_run_root(
        self,
        tmpdir: str,
        *,
        run_ts: str,
        embedding_dim: int = 2,
    ) -> tuple[Path, Path]:
        run_root = Path(tmpdir) / "svi_xlsx"
        run_dir = run_root / run_ts
        checkpoints_dir = run_dir / "checkpoints"
        metrics_dir = run_dir / "metrics"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(11)
        model = SviRegressor(
            current_input_dim=29,
            embedding_dim=embedding_dim,
            regression_dim=24,
            count_classes=4,
            hidden_dim=16,
            dropout=0.0,
        )
        for parameter in model.parameters():
            torch.nn.init.constant_(parameter, 0.0)

        checkpoint_path = checkpoints_dir / "svi_regressor_best.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {"svi_hidden_dim": 16, "svi_dropout": 0.0},
                "embedding_dim": embedding_dim,
                "current_input_dim": 29,
                "regression_dim": 24,
                "max_slices": 4,
                "normalization_stats": {
                    "feature_order": ["business_days", "a", "b", "rho", "m", "sigma"],
                    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                },
            },
            checkpoint_path,
        )
        (metrics_dir / "best_checkpoint.json").write_text(
            json.dumps(
                {
                    "monitor_metric": "val_regression",
                    "best_epoch": 1,
                    "best_metric": 0.1,
                    "artifacts": {"model": str(checkpoint_path)},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return run_root, run_dir

    def test_generate_result_config_loads_yaml_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "vol.yaml"
            _write_yaml(
                config_path,
                {
                    "data_path": "data/example.xlsx",
                    "sheet_name": "gan_input_ready",
                    "text_embedding_mode": "hd",
                    "train_ratio": 0.8,
                    "split": "val",
                    "selection_mode": "first_n",
                    "limit": 3,
                    "save_plots": True,
                    "save_json": True,
                    "output_dir": "outputs/generate_result/vol",
                },
            )

            overrides = parse_generate_result_overrides(["selection_mode=row_index", "row_index=2", "save_json=false"])
            config = load_generate_result_config(str(config_path), overrides=overrides)

            self.assertEqual(config.selection_mode, "row_index")
            self.assertEqual(config.row_index, 2)
            self.assertFalse(config.save_json)

            for invalid_bool in ["save_json=off", "save_json=yes", "save_json=1"]:
                with self.subTest(invalid_bool=invalid_bool):
                    with self.assertRaises(ValueError):
                        parse_generate_result_overrides([invalid_bool])

    def test_vol_split_and_selection_helpers_follow_chronological_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_vol_workbook(tmpdir)
            config = GenerateResultConfig(
                data_path=str(workbook_path),
                sheet_name="gan_input_ready",
                text_embedding_mode="hd",
                train_ratio=2 / 3,
                split="val",
                selection_mode="row_index",
                row_index=0,
            )
            samples = load_vol_surface_samples(config)
            split_selection = select_ordered_split(samples, train_ratio=config.train_ratio, split=config.split)
            selected = select_samples_from_split(split_selection, config)
            self.assertEqual([sample.sample_id for sample in split_selection.train_items], ["news_1", "news_2"])
            self.assertEqual([sample.sample_id for sample in split_selection.val_items], ["news_3"])
            self.assertEqual(selected[0].sample_id, "news_3")

            config = GenerateResultConfig(
                data_path=str(workbook_path),
                sheet_name="gan_input_ready",
                text_embedding_mode="hd",
                train_ratio=2 / 3,
                split="all",
                selection_mode="sample_id",
                sample_id="news_2",
            )
            selected = select_samples_from_split(
                select_ordered_split(load_vol_surface_samples(config), train_ratio=config.train_ratio, split=config.split),
                config,
            )
            self.assertEqual(selected[0].sample_id, "news_2")

    def test_resolve_checkpoint_path_prefers_explicit_then_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir, metrics_dir, generator_path = self._write_generator_checkpoint(tmpdir)
            explicit_path = Path(tmpdir) / "explicit_generator.pt"
            explicit_path.write_bytes(generator_path.read_bytes())

            config = GenerateResultConfig(
                checkpoint_path=str(explicit_path),
                models_path=str(models_dir),
                metrics_path=str(metrics_dir),
            )
            resolved = resolve_checkpoint_path(
                config,
                artifact_key="generator",
                fallback_filenames=("generator_best.pt", "generator.pt"),
            )
            self.assertEqual(resolved, explicit_path)

            config = GenerateResultConfig(
                checkpoint_path="",
                models_path=str(models_dir),
                metrics_path=str(metrics_dir),
            )
            resolved = resolve_checkpoint_path(
                config,
                artifact_key="generator",
                fallback_filenames=("generator_best.pt", "generator.pt"),
            )
            self.assertEqual(resolved.name, "generator_best.pt")

    def test_resolve_checkpoint_path_from_run_root_picks_latest_timestamped_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root, _ = self._write_generator_checkpoint_run_root(tmpdir, run_ts="20260407_101010")
            _, latest_run_dir = self._write_generator_checkpoint_run_root(tmpdir, run_ts="20260407_111111")

            config = GenerateResultConfig(
                checkpoint_path="",
                models_path=str(run_root),
                metrics_path=str(run_root),
            )
            resolved = resolve_checkpoint_path(
                config,
                artifact_key="generator",
                fallback_filenames=("generator_best.pt", "generator.pt"),
            )
            self.assertEqual(resolved, latest_run_dir / "checkpoints" / "generator_best.pt")

    def test_generate_vol_script_outputs_json_png_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_vol_workbook(tmpdir)
            run_root, _ = self._write_generator_checkpoint_run_root(tmpdir, run_ts="20260407_121212")
            output_dir = Path(tmpdir) / "generate_vol"
            config_path = Path(tmpdir) / "generate_vol.yaml"
            _write_yaml(
                config_path,
                {
                    "data_path": str(workbook_path),
                    "sheet_name": "gan_input_ready",
                    "text_embedding_mode": "hd",
                    "train_ratio": 2 / 3,
                    "cuda": False,
                    "seed": 123,
                    "checkpoint_path": "",
                    "models_path": str(run_root),
                    "metrics_path": str(run_root),
                    "split": "val",
                    "selection_mode": "row_index",
                    "row_index": 0,
                    "limit": 5,
                    "output_dir": str(output_dir),
                    "save_plots": True,
                    "save_json": True,
                    "plot_style": "heatmap_diff",
                },
            )

            module = _load_script_module(ROOT_DIR / "scripts/generate_result/generate_vol.py", "generate_vol_script")
            run_dir = module.main(["--config", str(config_path)])

            self.assertTrue((run_dir / "resolved_config.yaml").exists())
            self.assertTrue((run_dir / "summary.csv").exists())
            json_files = sorted((run_dir / "samples").glob("*.json"))
            png_files = sorted((run_dir / "plots").glob("*.png"))
            self.assertEqual(len(json_files), 1)
            self.assertEqual(len(png_files), 3)
            self.assertTrue(any(path.name.endswith("_lines.png") for path in png_files))
            self.assertTrue(any(path.name.endswith("_atm.png") for path in png_files))
            payload = json.loads(json_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["sample_id"], "news_3")
            self.assertEqual(payload["mode"], "vol")
            self.assertIn("mae", payload["metrics"])
            with (run_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_id"], "news_3")

    def test_generate_vol_script_supports_none_text_mode_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_vol_workbook(tmpdir)
            run_root, _ = self._write_generator_checkpoint_run_root(
                tmpdir,
                run_ts="20260407_141414",
                embedding_dim=1,
                text_embedding_mode="none",
            )
            output_dir = Path(tmpdir) / "generate_vol_none"
            config_path = Path(tmpdir) / "generate_vol_none.yaml"
            _write_yaml(
                config_path,
                {
                    "data_path": str(workbook_path),
                    "sheet_name": "gan_input_ready",
                    "text_embedding_mode": "none",
                    "train_ratio": 2 / 3,
                    "cuda": False,
                    "seed": 123,
                    "checkpoint_path": "",
                    "models_path": str(run_root),
                    "metrics_path": str(run_root),
                    "split": "val",
                    "selection_mode": "row_index",
                    "row_index": 0,
                    "limit": 5,
                    "output_dir": str(output_dir),
                    "save_plots": False,
                    "save_json": True,
                    "plot_style": "heatmap_diff",
                },
            )

            module = _load_script_module(ROOT_DIR / "scripts/generate_result/generate_vol.py", "generate_vol_none_script")
            run_dir = module.main(["--config", str(config_path)])

            self.assertTrue((run_dir / "summary.csv").exists())
            json_files = sorted((run_dir / "samples").glob("*.json"))
            self.assertEqual(len(json_files), 1)
            payload = json.loads(json_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["sample_id"], "news_3")

    def test_generate_svi_script_outputs_predicted_and_real_surfaces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_svi_workbook(tmpdir)
            run_root, _ = self._write_svi_checkpoint_run_root(tmpdir, run_ts="20260407_131313")
            output_dir = Path(tmpdir) / "generate_svi"
            config_path = Path(tmpdir) / "generate_svi.yaml"
            _write_yaml(
                config_path,
                {
                    "data_path": str(workbook_path),
                    "sheet_name": "news_direction_audit",
                    "text_embedding_mode": "hd",
                    "train_ratio": 2 / 3,
                    "cuda": False,
                    "seed": 123,
                    "checkpoint_path": "",
                    "models_path": str(run_root),
                    "metrics_path": str(run_root),
                    "split": "val",
                    "selection_mode": "row_index",
                    "row_index": 0,
                    "limit": 5,
                    "output_dir": str(output_dir),
                    "save_plots": True,
                    "save_json": True,
                    "plot_style": "heatmap_diff",
                    "strike_bins": 16,
                    "maturity_bins": 16,
                    "moneyness_min": 0.7,
                    "moneyness_max": 1.3,
                    "maturity_min_days": 7,
                    "maturity_max_days": 365,
                },
            )

            module = _load_script_module(ROOT_DIR / "scripts/generate_result/generate_svi.py", "generate_svi_script")
            run_dir = module.main(["--config", str(config_path)])

            json_files = sorted((run_dir / "samples").glob("*.json"))
            png_files = sorted((run_dir / "plots").glob("*.png"))
            self.assertEqual(len(json_files), 1)
            self.assertEqual(len(png_files), 3)
            self.assertTrue(any(path.name.endswith("_lines.png") for path in png_files))
            self.assertTrue(any(path.name.endswith("_atm.png") for path in png_files))
            payload = json.loads(json_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["sample_id"], "news_3")
            self.assertEqual(payload["mode"], "svi")
            self.assertEqual(len(payload["strike_grid"]), 16)
            self.assertIn("predicted_future_svi", payload["metadata"])
            self.assertIn("real_future_svi", payload["metadata"])
            with (run_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_id"], "news_3")

    def test_plot_surface_script_renders_png_from_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "payload.json"
            payload = {
                "sample_id": "news_1",
                "mode": "vol",
                "strike_grid": [0.9, 1.0, 1.1],
                "maturity_days_grid": [10.0, 20.0],
                "current_surface": [[0.2, 0.21, 0.22], [0.23, 0.24, 0.25]],
                "generated_surface": [[0.22, 0.23, 0.24], [0.25, 0.26, 0.27]],
                "real_surface": [[0.21, 0.22, 0.23], [0.24, 0.25, 0.26]],
                "metrics": {"mae": 0.01, "rmse": 0.02, "max_abs": 0.03},
                "metadata": {},
            }
            input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            output_path = Path(tmpdir) / "plot.png"

            module = _load_script_module(ROOT_DIR / "scripts/generate_result/plot_surface.py", "plot_surface_script")
            returned_path = module.main(["--input-json", str(input_path), "--output", str(output_path)])

            self.assertEqual(returned_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            sidecar_lines = output_path.with_name(f"{output_path.stem}_lines{output_path.suffix}")
            sidecar_atm = output_path.with_name(f"{output_path.stem}_atm{output_path.suffix}")
            self.assertTrue(sidecar_lines.exists())
            self.assertTrue(sidecar_atm.exists())
            self.assertGreater(sidecar_lines.stat().st_size, 0)
            self.assertGreater(sidecar_atm.stat().st_size, 0)

    def test_plot_surface_helper_uses_nearest_atm_and_short_maturity(self):
        module = _load_script_module(ROOT_DIR / "scripts/generate_result/plot_surface.py", "plot_surface_helper_script")

        atm_idx = module._nearest_atm_index([0.7, 0.91, 1.03, 1.25])
        short_idx = module._short_maturity_index([7.0, 30.0, 60.0, 120.0, 240.0])

        self.assertEqual(atm_idx, 2)
        self.assertEqual(short_idx, 0)

    def test_generate_result_main_dispatches_subcommands(self):
        module = _load_script_module(ROOT_DIR / "scripts/generate_result/main.py", "generate_result_main_script")
        calls = []

        def _record(argv):
            calls.append(list(argv))

        with patch.dict(module.COMMANDS, {"vol": _record, "svi": _record, "plot": _record}, clear=False):
            module.main(["vol", "--config", "configs/generate_result/vol.yaml"])

        self.assertEqual(calls, [["--config", "configs/generate_result/vol.yaml"]])


if __name__ == "__main__":
    unittest.main()
