import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

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

from scripts.analyze_error.bootstrap import bootstrap_from_error_rows, bootstrap_mean_mse  # noqa: E402
from scripts.analyze_error.plotting import plot_bootstrap_mean_histogram, plot_mse_histogram  # noqa: E402
from wgan_option.analysis_config import AnalysisConfig, load_analysis_config, parse_analysis_overrides  # noqa: E402
from wgan_option.config import Config  # noqa: E402
from wgan_option.models.generator import Generator  # noqa: E402
from wgan_option.models.svi_regressor import SviRegressor  # noqa: E402
from wgan_option.utils.inference_helpers import (  # noqa: E402
    build_inference_device,
    infer_future_svi,
    infer_vol_surface,
    load_svi_regressor,
    load_vol_generator,
    reconstruct_svi_surface,
)
from wgan_option.utils.merged_xlsx import load_svi_paired_samples, load_vol_surface_samples  # noqa: E402


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


class TestAnalyzeErrorScripts(unittest.TestCase):
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
                    "current_weighted_iv_rmse": 0.03,
                    "target_weighted_iv_rmse": 0.04,
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
                    "current_weighted_iv_rmse": 0.05,
                    "target_weighted_iv_rmse": 0.06,
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

    def _write_generator_checkpoint_run_root(
        self,
        tmpdir: str,
        *,
        run_ts: str,
        embedding_dim: int = 2,
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

    def test_analysis_config_loads_yaml_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "vol.yaml"
            _write_yaml(
                config_path,
                {
                    "data_path": "data/example.xlsx",
                    "sheet_name": "gan_input_ready",
                    "text_embedding_mode": "hd",
                    "train_ratio": 0.8,
                    "split": "all",
                    "selection_mode": "all",
                    "histogram_bins": 40,
                    "save_mse_histogram": True,
                    "save_bootstrap_histogram": True,
                    "bootstrap_samples": 1000,
                    "confidence_level": 0.95,
                    "output_dir": "outputs/analyze_error/vol",
                },
            )

            overrides = parse_analysis_overrides(
                [
                    "selection_mode=row_index",
                    "row_index=2",
                    "histogram_bins=12",
                    "save_mse_histogram=false",
                    "save_bootstrap_histogram=false",
                    "bootstrap_samples=250",
                    "save_bootstrap_distribution=false",
                ]
            )
            config = load_analysis_config(str(config_path), overrides=overrides)

            self.assertEqual(config.selection_mode, "row_index")
            self.assertEqual(config.row_index, 2)
            self.assertEqual(config.histogram_bins, 12)
            self.assertFalse(config.save_mse_histogram)
            self.assertFalse(config.save_bootstrap_histogram)
            self.assertEqual(config.bootstrap_samples, 250)
            self.assertFalse(config.save_bootstrap_distribution)

            for invalid_bool in [
                "save_mse_histogram=off",
                "save_bootstrap_histogram=yes",
                "save_bootstrap_distribution=1",
            ]:
                with self.subTest(invalid_bool=invalid_bool):
                    with self.assertRaises(ValueError):
                        parse_analysis_overrides([invalid_bool])

    def test_bootstrap_mean_mse_is_repeatable_for_fixed_seed(self):
        summary_a, distribution_a = bootstrap_mean_mse(
            [0.1, 0.2, 0.3],
            bootstrap_samples=64,
            confidence_level=0.95,
            seed=17,
        )
        summary_b, distribution_b = bootstrap_mean_mse(
            [0.1, 0.2, 0.3],
            bootstrap_samples=64,
            confidence_level=0.95,
            seed=17,
        )

        self.assertEqual(summary_a, summary_b)
        np.testing.assert_allclose(distribution_a, distribution_b)

    def test_plotting_helpers_render_histograms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            mse_histogram = output_dir / "mse_histogram.png"
            bootstrap_histogram = output_dir / "bootstrap_mean_mse_histogram.png"
            error_rows = [
                {"mse": 0.1},
                {"mse": 0.2},
                {"mse": 0.3},
            ]
            bootstrap_summary = {
                "observed_mean_mse": 0.2,
                "ci_lower": 0.15,
                "ci_upper": 0.25,
            }
            bootstrap_distribution = np.asarray([0.15, 0.18, 0.2, 0.23, 0.25], dtype=np.float64)

            plot_mse_histogram(error_rows, mse_histogram, bins=5)
            plot_bootstrap_mean_histogram(
                bootstrap_distribution,
                bootstrap_summary,
                bootstrap_histogram,
                bins=5,
            )

            self.assertTrue(mse_histogram.exists())
            self.assertTrue(bootstrap_histogram.exists())
            self.assertGreater(mse_histogram.stat().st_size, 0)
            self.assertGreater(bootstrap_histogram.stat().st_size, 0)

    def test_analyze_vol_script_outputs_errors_and_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_vol_workbook(tmpdir)
            run_root, _ = self._write_generator_checkpoint_run_root(tmpdir, run_ts="20260407_121212")
            output_dir = Path(tmpdir) / "analyze_vol"
            config_path = Path(tmpdir) / "analyze_vol.yaml"
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
                    "split": "all",
                    "selection_mode": "all",
                    "output_dir": str(output_dir),
                    "histogram_bins": 10,
                    "save_mse_histogram": True,
                    "save_bootstrap_histogram": True,
                    "bootstrap_samples": 128,
                    "confidence_level": 0.95,
                    "bootstrap_seed": 11,
                    "save_bootstrap_distribution": True,
                },
            )

            module = _load_script_module(ROOT_DIR / "scripts/analyze_error/analyze_vol.py", "analyze_vol_script")
            run_dir = module.main(["--config", str(config_path)])

            self.assertTrue((run_dir / "resolved_config.yaml").exists())
            self.assertTrue((run_dir / "errors.csv").exists())
            self.assertTrue((run_dir / "bootstrap_summary.csv").exists())
            self.assertTrue((run_dir / "bootstrap_summary.json").exists())
            self.assertTrue((run_dir / "bootstrap_distribution.csv").exists())
            self.assertTrue((run_dir / "mse_histogram.png").exists())
            self.assertTrue((run_dir / "bootstrap_mean_mse_histogram.png").exists())
            self.assertGreater((run_dir / "mse_histogram.png").stat().st_size, 0)
            self.assertGreater((run_dir / "bootstrap_mean_mse_histogram.png").stat().st_size, 0)

            with (run_dir / "errors.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["sample_id"] for row in rows], ["news_1", "news_2", "news_3"])
            self.assertIn("mse", rows[0])
            self.assertIn("error_mean", rows[0])
            self.assertIn("pair_quality_label", rows[0])

            config = AnalysisConfig(
                data_path=str(workbook_path),
                sheet_name="gan_input_ready",
                text_embedding_mode="hd",
                train_ratio=2 / 3,
                cuda=False,
                seed=123,
            )
            samples = load_vol_surface_samples(config)
            device = build_inference_device(False)
            model, train_config, _ = load_vol_generator(run_root / "20260407_121212" / "checkpoints" / "generator_best.pt", samples[0], device)
            generated_surface = infer_vol_surface(
                model,
                samples[0],
                noise_dim=int(train_config.noise_dim),
                seed=int(config.seed),
                device=device,
            )
            target_surface = samples[0].target_surface[0]
            expected_mse = float(np.mean((target_surface - generated_surface) ** 2))
            self.assertAlmostEqual(float(rows[0]["mse"]), expected_mse, places=7)

            bootstrap_summary, bootstrap_distribution = bootstrap_from_error_rows(
                rows,
                bootstrap_samples=128,
                confidence_level=0.95,
                seed=11,
            )
            with (run_dir / "bootstrap_summary.json").open(encoding="utf-8") as handle:
                saved_summary = json.load(handle)
            self.assertEqual(saved_summary, bootstrap_summary)
            with (run_dir / "bootstrap_distribution.csv").open(encoding="utf-8", newline="") as handle:
                distribution_rows = list(csv.DictReader(handle))
            self.assertEqual(len(distribution_rows), 128)
            self.assertAlmostEqual(float(distribution_rows[0]["mean_mse"]), float(bootstrap_distribution[0]), places=12)

    def test_analyze_svi_script_outputs_errors_and_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = self._write_svi_workbook(tmpdir)
            run_root, _ = self._write_svi_checkpoint_run_root(tmpdir, run_ts="20260407_131313")
            output_dir = Path(tmpdir) / "analyze_svi"
            config_path = Path(tmpdir) / "analyze_svi.yaml"
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
                    "split": "all",
                    "selection_mode": "all",
                    "output_dir": str(output_dir),
                    "histogram_bins": 10,
                    "save_mse_histogram": True,
                    "save_bootstrap_histogram": True,
                    "bootstrap_samples": 128,
                    "confidence_level": 0.95,
                    "bootstrap_seed": 19,
                    "save_bootstrap_distribution": True,
                    "strike_bins": 16,
                    "maturity_bins": 16,
                    "moneyness_min": 0.7,
                    "moneyness_max": 1.3,
                    "maturity_min_days": 7,
                    "maturity_max_days": 365,
                },
            )

            module = _load_script_module(ROOT_DIR / "scripts/analyze_error/analyze_svi.py", "analyze_svi_script")
            run_dir = module.main(["--config", str(config_path)])

            self.assertTrue((run_dir / "mse_histogram.png").exists())
            self.assertTrue((run_dir / "bootstrap_mean_mse_histogram.png").exists())
            self.assertGreater((run_dir / "mse_histogram.png").stat().st_size, 0)
            self.assertGreater((run_dir / "bootstrap_mean_mse_histogram.png").stat().st_size, 0)

            with (run_dir / "errors.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["sample_id"] for row in rows], ["news_1", "news_2", "news_3"])
            self.assertIn("news_row_id", rows[0])
            self.assertIn("current_slice_count", rows[0])
            self.assertIn("predicted_slice_count", rows[0])
            self.assertIn("real_future_slice_count", rows[0])

            config = AnalysisConfig(
                data_path=str(workbook_path),
                sheet_name="news_direction_audit",
                text_embedding_mode="hd",
                train_ratio=2 / 3,
                cuda=False,
                seed=123,
                strike_bins=16,
                maturity_bins=16,
                moneyness_min=0.7,
                moneyness_max=1.3,
                maturity_min_days=7,
                maturity_max_days=365,
            )
            samples = load_svi_paired_samples(config)
            device = build_inference_device(False)
            model, checkpoint = load_svi_regressor(run_root / "20260407_131313" / "checkpoints" / "svi_regressor_best.pt", device)
            strike_grid = np.linspace(0.7, 1.3, 16, dtype=np.float32)
            maturity_days_grid = np.linspace(7, 365, 16, dtype=np.float32)
            predicted_svi, _, _ = infer_future_svi(
                model,
                samples[0],
                embedding_dim=int(checkpoint["embedding_dim"]),
                max_slices=int(checkpoint["max_slices"]),
                normalization_stats=checkpoint["normalization_stats"],
                device=device,
            )
            generated_surface = reconstruct_svi_surface(
                predicted_svi,
                strike_grid=strike_grid,
                maturity_days_grid=maturity_days_grid,
            )
            target_surface = reconstruct_svi_surface(
                samples[0].future_svi,
                strike_grid=strike_grid,
                maturity_days_grid=maturity_days_grid,
            )
            expected_mse = float(np.mean((target_surface - generated_surface) ** 2))
            self.assertAlmostEqual(float(rows[0]["mse"]), expected_mse, places=7)

            with (run_dir / "bootstrap_summary.json").open(encoding="utf-8") as handle:
                saved_summary = json.load(handle)
            expected_summary, _ = bootstrap_from_error_rows(
                rows,
                bootstrap_samples=128,
                confidence_level=0.95,
                seed=19,
            )
            self.assertEqual(saved_summary, expected_summary)

    def test_analyze_error_main_dispatches_subcommands(self):
        module = _load_script_module(ROOT_DIR / "scripts/analyze_error/main.py", "analyze_error_main_script")
        received = []

        def _fake(argv):
            received.append(list(argv))

        module.COMMANDS["vol"] = _fake
        module.main(["vol", "--config", "demo.yaml"])

        self.assertEqual(received, [["--config", "demo.yaml"]])


if __name__ == "__main__":
    unittest.main()
