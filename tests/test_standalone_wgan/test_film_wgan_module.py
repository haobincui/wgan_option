import csv
import json
import math
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from film_wgan.config import FilmWGANSampleConfig, FilmWGANTrainConfig, load_train_config  # noqa: E402
from film_wgan.data import create_train_val_bundle, denormalize_tensor, normalize_surface_tensor  # noqa: E402
from film_wgan.inference import FilmWGANSampler, normalization_stats_to_tensors  # noqa: E402
from film_wgan.losses import (  # noqa: E402
    build_reconstruction_weight_template,
    gradient_penalty,
    weighted_surface_mae,
)
from film_wgan.models import (  # noqa: E402
    FiLMLayer as FilmFiLMLayer,
    FilmWGANCritic,
    FilmWGANGenerator,
    reconstruct_future_surface,
)
from film_wgan.short_atm_study import (  # noqa: E402
    build_short_atm_grid_specs,
    choose_recommended_config,
    select_blend_scan_config_ids,
)
from film_wgan.trainer import FilmWGANTrainer  # noqa: E402
from stylemod_wgan.models import FiLMLayer as StyleModFiLMLayer  # noqa: E402

FILM_TRAIN_CONFIG_PATH = ROOT_DIR / "configs/film_wgan/train_lp_gen128_disc128.yaml"


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return path


def _json_text(values) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def _surface_values(base: float, cells: int) -> list[float]:
    return [float(base + idx) / 1000.0 for idx in range(cells)]


def _write_vol_workbook(tmpdir: str) -> Path:
    path = Path(tmpdir) / "merged_vol.xlsx"
    strike_grid = [0.80, 1.02, 1.20]
    maturity_grid = [7, 30]
    cells = len(strike_grid) * len(maturity_grid)
    dataframe = pd.DataFrame(
        [
            {
                "sample_id": "news_2",
                "news_timestamp_utc": "2022-12-30T13:40:00Z",
                "current_snapshot_time_utc": "2022-12-30T13:40:00Z",
                "target_snapshot_time_utc": "2022-12-30T13:45:00Z",
                "hd_embedding": _json_text([1.0, 2.0]),
                "lp_embedding": _json_text([10.0, 20.0, 30.0]),
                "bow_embedding": _json_text([0.10, 0.20, 0.30, 0.40]),
                "sentiment_embedding": _json_text([1.0, 0.0, 0.5, 0.0, 0.25]),
                "strike_grid": _json_text(strike_grid),
                "maturity_days_grid": _json_text(maturity_grid),
                "current_surface_flat": _json_text(_surface_values(1.0, cells)),
                "target_surface_flat": _json_text(_surface_values(2.0, cells)),
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
                "bow_embedding": _json_text([0.50, 0.60, 0.70, 0.80]),
                "sentiment_embedding": _json_text([0.0, 1.0, 0.25, 0.0, 0.50]),
                "strike_grid": _json_text(strike_grid),
                "maturity_days_grid": _json_text(maturity_grid),
                "current_surface_flat": _json_text(_surface_values(3.0, cells)),
                "target_surface_flat": _json_text(_surface_values(4.0, cells)),
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
                "bow_embedding": _json_text([0.90, 1.00, 1.10, 1.20]),
                "sentiment_embedding": _json_text([0.0, 0.0, 0.75, 1.0, 0.00]),
                "strike_grid": _json_text(strike_grid),
                "maturity_days_grid": _json_text(maturity_grid),
                "current_surface_flat": _json_text(_surface_values(5.0, cells)),
                "target_surface_flat": _json_text(_surface_values(6.0, cells)),
                "pair_quality_label": "usable",
                "training_candidate_flag": 1,
            },
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name="gan_input_ready", index=False)
    return path


def _write_film_checkpoint(
    tmpdir: str,
    *,
    workbook_path: Path,
    run_ts: str = "20260416_010101",
) -> tuple[Path, Path, Path]:
    output_root = Path(tmpdir) / "training" / "film_wgan"
    run_dir = output_root / run_ts
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config = FilmWGANTrainConfig(
        data_path=str(workbook_path),
        sheet_name="gan_input_ready",
        text_embedding_mode="lp",
        train_ratio=2 / 3,
        normalize_current_surface=True,
        normalize_target_delta=True,
        normalize_text_embedding=True,
        noise_dim=4,
        gen_base_channels=8,
        disc_base_channels=8,
        gen_res_blocks=1,
        disc_res_blocks=1,
        text_hidden_dim=16,
        text_out_dim=8,
        fusion_hidden_dim=32,
        num_epochs=1,
        batch_size=2,
        cuda=False,
        num_workers=0,
        output_root=str(output_root),
        checkpoints_path=str(checkpoints_dir),
        metrics_path=str(run_dir / "metrics"),
        save_every=1,
    )
    generator = FilmWGANGenerator(
        surface_height=2,
        surface_width=3,
        embedding_dim=3,
        noise_dim=config.noise_dim,
        base_channels=config.gen_base_channels,
        res_blocks=config.gen_res_blocks,
        text_hidden_dim=config.text_hidden_dim,
        text_out_dim=config.text_out_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
    )
    checkpoint_path = checkpoints_dir / "film_wgan_best.pt"
    torch.save(
        {
            "config": asdict(config),
            "surface_shape": [2, 3],
            "strike_grid": [0.80, 1.02, 1.20],
            "maturity_days_grid": [7.0, 30.0],
            "embedding_dim": 3,
            "generator_state_dict": generator.state_dict(),
            "normalization_stats": {
                "current_log_mean": [0.0] * 6,
                "current_log_std": [1.0] * 6,
                "delta_mean": [0.0] * 6,
                "delta_std": [1.0] * 6,
                "text_mean": [0.0] * 3,
                "text_std": [1.0] * 3,
            },
        },
        checkpoint_path,
    )
    return output_root, run_dir, checkpoint_path


class TestFiLMLayerInitialization(unittest.TestCase):
    def test_film_layers_start_as_identity_maps(self):
        torch.manual_seed(0)
        x = torch.randn(3, 4, 5, 5)
        conditioning = torch.randn(3, 7)

        for layer_cls in (FilmFiLMLayer, StyleModFiLMLayer):
            layer = layer_cls(7, 4)
            params = layer.projection(conditioning)
            gamma, beta = params.chunk(2, dim=-1)
            out = layer(x, conditioning)

            self.assertTrue(torch.equal(gamma, torch.zeros_like(gamma)))
            self.assertTrue(torch.equal(beta, torch.zeros_like(beta)))
            self.assertTrue(torch.equal(out, x))


class TestFilmWGANLossWeighting(unittest.TestCase):
    @staticmethod
    def _grids() -> tuple[torch.Tensor, torch.Tensor]:
        strike_grid = torch.tensor([0.80, 0.92, 1.00, 1.08, 1.20], dtype=torch.float32)
        maturity_days_grid = torch.tensor([7.0, 30.0, 120.0], dtype=torch.float32)
        return strike_grid, maturity_days_grid

    def test_uniform_weight_template_is_all_ones(self):
        strike_grid, maturity_days_grid = self._grids()
        weights = build_reconstruction_weight_template(
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
            mode="uniform",
            atm_range=0.08,
            short_end_max_days=90.0,
            atm_multiplier=3.0,
        )

        self.assertEqual(tuple(weights.shape), (3, 5))
        self.assertTrue(torch.allclose(weights, torch.ones_like(weights)))

    def test_short_atm_band_weights_emphasize_short_end_atm_and_are_mean_normalized(self):
        strike_grid, maturity_days_grid = self._grids()
        weights = build_reconstruction_weight_template(
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
            mode="short_atm_band",
            atm_range=0.08,
            short_end_max_days=30.0,
            atm_multiplier=3.0,
        )

        self.assertAlmostEqual(float(weights.mean().item()), 1.0, places=6)
        self.assertEqual(int(torch.count_nonzero(weights > 1.0).item()), 6)
        self.assertTrue(torch.all(weights[:2, 1:4] > 1.0))
        self.assertTrue(torch.all(weights[2, :] < 1.0))
        self.assertTrue(torch.all(weights[:, 0] < 1.0))
        self.assertTrue(torch.all(weights[:, 4] < 1.0))

    def test_weighted_mae_matches_plain_l1_under_uniform_weights(self):
        predicted = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float32)
        target = torch.tensor([[0.0, 1.0], [4.0, 7.0]], dtype=torch.float32)
        weights = torch.ones_like(predicted)

        weighted = weighted_surface_mae(predicted, target, weights)
        plain = torch.nn.functional.l1_loss(predicted, target)

        self.assertAlmostEqual(float(weighted.item()), float(plain.item()), places=6)

    def test_weighted_mae_emphasizes_band_local_errors(self):
        strike_grid, maturity_days_grid = self._grids()
        weights = build_reconstruction_weight_template(
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
            mode="short_atm_band",
            atm_range=0.08,
            short_end_max_days=30.0,
            atm_multiplier=3.0,
        )
        predicted = torch.zeros((3, 5), dtype=torch.float32)
        target = torch.zeros((3, 5), dtype=torch.float32)
        target[0, 2] = 1.0

        weighted = weighted_surface_mae(predicted, target, weights)
        plain = torch.nn.functional.l1_loss(predicted, target)

        self.assertGreater(float(weighted.item()), float(plain.item()))

    def test_weighted_mae_deemphasizes_out_of_band_errors(self):
        strike_grid, maturity_days_grid = self._grids()
        weights = build_reconstruction_weight_template(
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
            mode="short_atm_band",
            atm_range=0.08,
            short_end_max_days=30.0,
            atm_multiplier=3.0,
        )
        predicted = torch.zeros((3, 5), dtype=torch.float32)
        target = torch.zeros((3, 5), dtype=torch.float32)
        target[2, 4] = 1.0

        weighted = weighted_surface_mae(predicted, target, weights)
        plain = torch.nn.functional.l1_loss(predicted, target)

        self.assertLess(float(weighted.item()), float(plain.item()))


class TestFilmWGANConfiguration(unittest.TestCase):
    def test_active_config_enables_short_end_atm_weighting(self):
        config = load_train_config(FILM_TRAIN_CONFIG_PATH)

        self.assertEqual(config.recon_weight_mode, "short_atm_band")
        self.assertAlmostEqual(float(config.recon_atm_range), 0.08, places=6)
        self.assertAlmostEqual(float(config.recon_atm_short_end_max_days), 90.0, places=6)
        self.assertAlmostEqual(float(config.recon_atm_multiplier), 3.0, places=6)

    def test_extra_checkpoint_metrics_default_to_empty_and_load_from_yaml(self):
        self.assertEqual(FilmWGANTrainConfig().extra_checkpoint_metrics, [])

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _write_yaml(
                Path(tmpdir) / "film_extra_metrics.yaml",
                {
                    "training": {
                        "extra_checkpoint_metrics": [
                            "val_mae_gap_vs_current",
                            "val_short_atm_mae_gap_vs_current",
                        ]
                    }
                },
            )
            config = load_train_config(config_path)

        self.assertEqual(
            config.extra_checkpoint_metrics,
            ["val_mae_gap_vs_current", "val_short_atm_mae_gap_vs_current"],
        )

    def test_none_text_mode_uses_single_zero_feature_with_or_without_normalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)

            for normalize_text_embedding in (True, False):
                config = FilmWGANTrainConfig(
                    data_path=str(workbook_path),
                    sheet_name="gan_input_ready",
                    text_embedding_mode="none",
                    train_ratio=2 / 3,
                    normalize_text_embedding=normalize_text_embedding,
                    batch_size=8,
                    cuda=False,
                    num_workers=0,
                )
                bundle = create_train_val_bundle(config)

                self.assertEqual(bundle.embedding_dim, 1)
                current_features, text_features, target_delta, current_flat, target_flat = next(iter(bundle.train_loader))
                self.assertEqual(int(text_features.shape[1]), 1)
                self.assertTrue(torch.all(text_features == 0.0))
                self.assertFalse(torch.isnan(text_features).any())
                self.assertFalse(torch.isnan(current_features).any())
                self.assertFalse(torch.isnan(target_delta).any())
                self.assertFalse(torch.isnan(current_flat).any())
                self.assertFalse(torch.isnan(target_flat).any())

    def test_rq2_text_modes_read_bow_and_sentiment_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)

            expected_dims = {
                "bow": 4,
                "sentiment": 5,
                "llm-sentiment": 5,
            }
            for mode, expected_dim in expected_dims.items():
                config = FilmWGANTrainConfig(
                    data_path=str(workbook_path),
                    sheet_name="gan_input_ready",
                    text_embedding_mode=mode,
                    train_ratio=2 / 3,
                    normalize_text_embedding=False,
                    batch_size=8,
                    cuda=False,
                    num_workers=0,
                )
                bundle = create_train_val_bundle(config)
                _current_features, text_features, _target_delta, _current_flat, _target_flat = next(iter(bundle.train_loader))

                self.assertEqual(bundle.embedding_dim, expected_dim)
                self.assertEqual(int(text_features.shape[1]), expected_dim)
                self.assertFalse(torch.isnan(text_features).any())


class TestFilmWGANGenerateResultATMOutputs(unittest.TestCase):
    def test_extract_atm_short_value_uses_nearest_atm_and_shortest_maturity(self):
        from film_wgan.plotting import extract_atm_short_value

        stats = extract_atm_short_value(
            [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]],
            strike_grid=[0.75, 1.05, 1.30],
            maturity_days_grid=[7.0, 30.0],
        )

        self.assertAlmostEqual(float(stats["value"]), 0.20, places=6)
        self.assertAlmostEqual(float(stats["atm_strike"]), 1.05, places=6)
        self.assertAlmostEqual(float(stats["short_maturity_days"]), 7.0, places=6)
        self.assertEqual(int(stats["atm_index"]), 1)
        self.assertEqual(int(stats["short_index"]), 0)

    def test_plot_atm_vol_timeseries_uses_line_only_and_dashed_target(self):
        from film_wgan import plotting as plotting_module

        rows = [
            {
                "sample_id": "news_1",
                "global_index": 0,
                "news_timestamp_utc": "2022-12-30T13:30:00Z",
                "current_atm_vol": 0.004,
                "generated_atm_vol": 0.0045,
                "target_atm_vol": 0.005,
                "atm_strike": 1.02,
                "short_maturity_days": 7.0,
            },
            {
                "sample_id": "news_2",
                "global_index": 1,
                "news_timestamp_utc": "2022-12-30T13:35:00Z",
                "current_atm_vol": 0.006,
                "generated_atm_vol": 0.0065,
                "target_atm_vol": 0.007,
                "atm_strike": 1.02,
                "short_maturity_days": 7.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "atm_vol.png"
            fig = plotting_module.plt.figure(figsize=(10.0, 4.8))
            ax = fig.subplots(1, 1)

            with patch.object(plotting_module.plt, "subplots", return_value=(fig, ax)):
                plotting_module.plot_atm_vol_timeseries(rows, output_path, series_scope="sample")

            self.assertTrue(output_path.exists())
            lines = ax.get_lines()
            self.assertEqual([line.get_label() for line in lines], ["Current", "Generated", "Target"])
            self.assertEqual([line.get_linestyle() for line in lines], ["-", "-", "--"])
            for line in lines:
                self.assertIn(str(line.get_marker()).lower(), {"none", ""})

            plotting_module.plt.close(fig)

    def test_sampler_writes_sample_and_full_atm_vol_outputs_alongside_original_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            _output_root, training_run_dir, checkpoint_path = _write_film_checkpoint(tmpdir, workbook_path=workbook_path)
            sample_run_dir = training_run_dir / "generate_result" / "custom_samples"
            config = FilmWGANSampleConfig(
                data_path=str(workbook_path),
                sheet_name="gan_input_ready",
                text_embedding_mode="lp",
                train_ratio=2 / 3,
                checkpoint_path=str(checkpoint_path),
                seed=123,
                cuda=False,
                mc_samples=2,
                split="val",
                selection_mode="all",
                selection_count=0,
                aggregation_mode="weighted_mean",
                output_dir=str(sample_run_dir),
                save_json=True,
                save_plots=True,
            )

            run_dir = FilmWGANSampler(config).sample()

            self.assertEqual(run_dir, sample_run_dir)
            self.assertTrue((run_dir / "summary.csv").exists())
            self.assertTrue((run_dir / "run_metadata.json").exists())
            self.assertEqual(len(list((run_dir / "samples").glob("*.json"))), 1)
            self.assertEqual(len(list((run_dir / "plots").glob("*.png"))), 2)

            atm_vol_dir = training_run_dir / "generate_result" / "atm_vol"
            sample_csv = atm_vol_dir / "film_wgan_best_atm_vol_timeseries_sample.csv"
            sample_png = atm_vol_dir / "film_wgan_best_atm_vol_timeseries_sample.png"
            full_csv = atm_vol_dir / "film_wgan_best_atm_vol_timeseries_full.csv"
            full_png = atm_vol_dir / "film_wgan_best_atm_vol_timeseries_full.png"
            for path in (sample_csv, sample_png, full_csv, full_png):
                self.assertTrue(path.exists())
            self.assertGreater(sample_png.stat().st_size, 0)
            self.assertGreater(full_png.stat().st_size, 0)

            with sample_csv.open(encoding="utf-8", newline="") as handle:
                sample_rows = list(csv.DictReader(handle))
            with full_csv.open(encoding="utf-8", newline="") as handle:
                full_rows = list(csv.DictReader(handle))

            self.assertEqual([row["sample_id"] for row in sample_rows], ["news_3"])
            self.assertEqual([row["sample_id"] for row in full_rows], ["news_1", "news_2", "news_3"])
            self.assertEqual(len(sample_rows), 1)
            self.assertEqual(len(full_rows), 3)
            self.assertLess(len(sample_rows), len(full_rows))

            row = sample_rows[0]
            self.assertAlmostEqual(float(row["atm_strike"]), 1.02, places=6)
            self.assertAlmostEqual(float(row["short_maturity_days"]), 7.0, places=6)
            self.assertAlmostEqual(float(row["current_atm_vol"]), 0.006, places=6)
            self.assertAlmostEqual(float(row["target_atm_vol"]), 0.007, places=6)
            self.assertAlmostEqual(float(row["current_target_abs_error"]), 0.001, places=6)
            self.assertEqual(row["checkpoint_path"], str(checkpoint_path))
            self.assertTrue(math.isfinite(float(row["generated_atm_vol"])))
            self.assertTrue(math.isfinite(float(row["generated_target_abs_error"])))

            for row, expected_current, expected_target in [
                (full_rows[0], 0.004, 0.005),
                (full_rows[1], 0.002, 0.003),
                (full_rows[2], 0.006, 0.007),
            ]:
                self.assertAlmostEqual(float(row["atm_strike"]), 1.02, places=6)
                self.assertAlmostEqual(float(row["short_maturity_days"]), 7.0, places=6)
                self.assertAlmostEqual(float(row["current_atm_vol"]), expected_current, places=6)
                self.assertAlmostEqual(float(row["target_atm_vol"]), expected_target, places=6)
                self.assertAlmostEqual(float(row["current_target_abs_error"]), 0.001, places=6)
                self.assertEqual(row["checkpoint_path"], str(checkpoint_path))
                self.assertTrue(math.isfinite(float(row["generated_atm_vol"])))
                self.assertTrue(math.isfinite(float(row["generated_target_abs_error"])))

    def test_sampler_writes_atm_csv_without_any_png_when_save_plots_is_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            _output_root, training_run_dir, checkpoint_path = _write_film_checkpoint(tmpdir, workbook_path=workbook_path)
            sample_run_dir = training_run_dir / "generate_result" / "custom_no_plot"
            config = FilmWGANSampleConfig(
                data_path=str(workbook_path),
                sheet_name="gan_input_ready",
                text_embedding_mode="lp",
                train_ratio=2 / 3,
                checkpoint_path=str(checkpoint_path),
                seed=123,
                cuda=False,
                mc_samples=2,
                split="all",
                selection_mode="all",
                selection_count=0,
                aggregation_mode="weighted_mean",
                output_dir=str(sample_run_dir),
                save_json=True,
                save_plots=False,
            )

            run_dir = FilmWGANSampler(config).sample()

            self.assertEqual(run_dir, sample_run_dir)
            self.assertTrue((run_dir / "summary.csv").exists())
            self.assertEqual(len(list((run_dir / "samples").glob("*.json"))), 3)
            self.assertFalse((run_dir / "plots").exists())

            atm_vol_dir = training_run_dir / "generate_result" / "atm_vol"
            sample_csv = atm_vol_dir / "film_wgan_best_atm_vol_timeseries_sample.csv"
            sample_png = atm_vol_dir / "film_wgan_best_atm_vol_timeseries_sample.png"
            full_csv = atm_vol_dir / "film_wgan_best_atm_vol_timeseries_full.csv"
            full_png = atm_vol_dir / "film_wgan_best_atm_vol_timeseries_full.png"
            self.assertTrue(sample_csv.exists())
            self.assertTrue(full_csv.exists())
            self.assertFalse(sample_png.exists())
            self.assertFalse(full_png.exists())

    def test_sampler_summary_tracks_short_end_metrics_and_residual_blend_alpha(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            _output_root, training_run_dir, checkpoint_path = _write_film_checkpoint(tmpdir, workbook_path=workbook_path)
            sample_run_dir = training_run_dir / "generate_result" / "blended_current"
            config = FilmWGANSampleConfig(
                data_path=str(workbook_path),
                sheet_name="gan_input_ready",
                text_embedding_mode="lp",
                train_ratio=2 / 3,
                checkpoint_path=str(checkpoint_path),
                seed=123,
                cuda=False,
                mc_samples=2,
                split="val",
                selection_mode="all",
                selection_count=0,
                aggregation_mode="weighted_mean",
                residual_blend_alpha=0.0,
                output_dir=str(sample_run_dir),
                save_json=False,
                save_plots=False,
            )

            run_dir = FilmWGANSampler(config).sample()

            with (run_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertAlmostEqual(float(row["residual_blend_alpha"]), 0.0, places=6)
            self.assertAlmostEqual(float(row["mae"]), float(row["current_mae"]), places=6)
            self.assertAlmostEqual(float(row["rmse"]), float(row["current_rmse"]), places=6)
            self.assertAlmostEqual(float(row["max_abs"]), float(row["current_max_abs"]), places=6)
            self.assertAlmostEqual(float(row["generated_current_mae"]), 0.0, places=6)
            self.assertAlmostEqual(float(row["mae_gap_vs_current"]), 0.0, places=6)
            self.assertAlmostEqual(float(row["short_atm_weighted_mae"]), float(row["current_short_atm_weighted_mae"]), places=6)
            self.assertAlmostEqual(float(row["short_atm_mae_gap_vs_current"]), 0.0, places=6)
            self.assertAlmostEqual(float(row["atm_short_pure_mae"]), float(row["current_atm_short_pure_mae"]), places=6)
            self.assertAlmostEqual(float(row["atm_short_pure_mae_gap_vs_current"]), 0.0, places=6)
            self.assertEqual(float(row["win_flag_vs_current"]), 0.0)
            self.assertEqual(float(row["short_atm_weighted_win_flag_vs_current"]), 0.0)
            self.assertEqual(float(row["atm_short_pure_win_flag_vs_current"]), 0.0)


class TestFilmWGANTrainerMetrics(unittest.TestCase):
    def test_training_metrics_json_includes_short_end_atm_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "training" / "film_wgan"
            config_path = _write_yaml(
                Path(tmpdir) / "film_metrics.yaml",
                {
                    "training": {
                        "data_path": str(Path(tmpdir) / "dummy.xlsx"),
                        "sheet_name": "gan_input_ready",
                        "text_embedding_mode": "lp",
                        "train_ratio": 0.67,
                        "min_samples_for_training": 2,
                        "noise_dim": 4,
                        "gen_base_channels": 8,
                        "disc_base_channels": 8,
                        "gen_res_blocks": 1,
                        "disc_res_blocks": 1,
                        "text_hidden_dim": 16,
                        "text_out_dim": 8,
                        "fusion_hidden_dim": 32,
                        "num_epochs": 1,
                        "batch_size": 1,
                        "eval_mc_samples": 2,
                        "cuda": False,
                        "num_workers": 0,
                        "output_root": str(output_root),
                        "save_every": 20,
                    }
                },
            )

            trainer = FilmWGANTrainer(load_train_config(config_path), config_path=str(config_path))

            def fake_setup():
                trainer.checkpoints_dir.mkdir(parents=True, exist_ok=True)
                trainer.metrics_dir.mkdir(parents=True, exist_ok=True)
                trainer.bundle = type(
                    "Bundle",
                    (),
                    {
                        "train_loader": [
                            (
                                torch.zeros((1, 1, 4, 4), dtype=torch.float32),
                                torch.zeros((1, 3), dtype=torch.float32),
                                torch.zeros((1, 16), dtype=torch.float32),
                                torch.ones((1, 16), dtype=torch.float32),
                                torch.ones((1, 16), dtype=torch.float32),
                            )
                        ]
                    },
                )()

            eval_row = {
                "val_mae": 0.10,
                "val_rmse": 0.12,
                "val_current_mae": 0.11,
                "val_current_rmse": 0.13,
                "val_short_atm_weighted_mae": 0.09,
                "val_current_short_atm_weighted_mae": 0.12,
                "val_short_atm_mae_gap_vs_current": -0.03,
                "val_mae_gap_vs_current": -0.01,
                "val_win_rate_vs_current": 1.0,
                "val_generated_current_mae": 0.02,
                "val_real_current_mae": 0.11,
                "val_calendar": 0.0,
                "val_butterfly": 0.0,
                "val_penalty_mean": 0.0,
                "val_penalty_std": 0.0,
                "val_weight_entropy": 1.0,
            }

            with patch.object(trainer, "setup", side_effect=fake_setup), \
                patch.object(
                    trainer,
                    "_discriminator_step",
                    return_value={"d_total": 0.0, "d_real": 0.0, "d_fake": 0.0, "gp": 0.0},
                ), \
                patch.object(
                    trainer,
                    "_generator_step",
                    return_value={
                        "g_total": 0.0,
                        "g_adv": 0.0,
                        "g_calendar": 0.0,
                        "g_butterfly": 0.0,
                        "g_smooth": 0.0,
                        "g_recon": 0.0,
                        "g_recon_weighted": 0.0,
                    },
                ), \
                patch.object(trainer, "_evaluate", return_value=eval_row), \
                patch.object(trainer, "_save_loss_curves", return_value=None), \
                patch.object(trainer, "_checkpoint_payload", return_value={"state": "ok"}):
                trainer.train()

            run_dir = next(path for path in output_root.iterdir() if path.is_dir())
            metrics_rows = json.loads((run_dir / "metrics" / "training_metrics.json").read_text(encoding="utf-8"))
            row = metrics_rows[-1]
            self.assertIn("g_recon_weighted", row)
            self.assertIn("val_short_atm_weighted_mae", row)
            self.assertIn("val_current_short_atm_weighted_mae", row)
            self.assertIn("val_short_atm_mae_gap_vs_current", row)
            self.assertAlmostEqual(float(row["val_short_atm_mae_gap_vs_current"]), -0.03, places=6)
            self.assertTrue((run_dir / "checkpoints" / "film_wgan_best.pt").exists())

    def test_trainer_tracks_extra_checkpoint_metrics_without_changing_primary_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "training" / "film_wgan"
            config_path = _write_yaml(
                Path(tmpdir) / "film_multi_metric.yaml",
                {
                    "training": {
                        "data_path": str(Path(tmpdir) / "dummy.xlsx"),
                        "sheet_name": "gan_input_ready",
                        "text_embedding_mode": "lp",
                        "train_ratio": 0.67,
                        "min_samples_for_training": 2,
                        "noise_dim": 4,
                        "gen_base_channels": 8,
                        "disc_base_channels": 8,
                        "gen_res_blocks": 1,
                        "disc_res_blocks": 1,
                        "text_hidden_dim": 16,
                        "text_out_dim": 8,
                        "fusion_hidden_dim": 32,
                        "num_epochs": 5,
                        "batch_size": 1,
                        "eval_mc_samples": 2,
                        "cuda": False,
                        "num_workers": 0,
                        "output_root": str(output_root),
                        "save_every": 20,
                        "checkpoint_metric": "val_atm_short_pure_mae_gap_vs_current",
                        "extra_checkpoint_metrics": [
                            "val_mae_gap_vs_current",
                            "val_short_atm_mae_gap_vs_current",
                        ],
                        "checkpoint_warmup_epochs": 0,
                        "use_early_stopping": True,
                        "early_stopping_patience": 1,
                        "early_stopping_min_delta": 0.0,
                    }
                },
            )

            trainer = FilmWGANTrainer(load_train_config(config_path), config_path=str(config_path))

            def fake_setup():
                trainer.checkpoints_dir.mkdir(parents=True, exist_ok=True)
                trainer.metrics_dir.mkdir(parents=True, exist_ok=True)
                trainer.bundle = type(
                    "Bundle",
                    (),
                    {
                        "train_loader": [
                            (
                                torch.zeros((1, 1, 4, 4), dtype=torch.float32),
                                torch.zeros((1, 3), dtype=torch.float32),
                                torch.zeros((1, 16), dtype=torch.float32),
                                torch.ones((1, 16), dtype=torch.float32),
                                torch.ones((1, 16), dtype=torch.float32),
                            )
                        ]
                    },
                )()

            eval_rows = [
                {
                    "val_mae": 0.20,
                    "val_rmse": 0.21,
                    "val_current_mae": 0.30,
                    "val_current_rmse": 0.31,
                    "val_short_atm_weighted_mae": 0.18,
                    "val_current_short_atm_weighted_mae": 0.38,
                    "val_short_atm_mae_gap_vs_current": -0.20,
                    "val_atm_short_pure_mae": 0.15,
                    "val_current_atm_short_pure_mae": 0.55,
                    "val_atm_short_pure_mae_gap_vs_current": -0.40,
                    "val_atm_short_win_rate_vs_current": 0.60,
                    "val_mae_gap_vs_current": -0.10,
                    "val_win_rate_vs_current": 0.60,
                    "val_generated_current_mae": 0.10,
                    "val_real_current_mae": 0.30,
                    "val_calendar": 0.0,
                    "val_butterfly": 0.0,
                    "val_penalty_mean": 0.0,
                    "val_penalty_std": 0.0,
                    "val_weight_entropy": 1.0,
                },
                {
                    "val_mae": 0.10,
                    "val_rmse": 0.12,
                    "val_current_mae": 0.60,
                    "val_current_rmse": 0.61,
                    "val_short_atm_weighted_mae": 0.12,
                    "val_current_short_atm_weighted_mae": 0.72,
                    "val_short_atm_mae_gap_vs_current": -0.60,
                    "val_atm_short_pure_mae": 0.25,
                    "val_current_atm_short_pure_mae": 0.55,
                    "val_atm_short_pure_mae_gap_vs_current": -0.30,
                    "val_atm_short_win_rate_vs_current": 0.50,
                    "val_mae_gap_vs_current": -0.50,
                    "val_win_rate_vs_current": 0.70,
                    "val_generated_current_mae": 0.12,
                    "val_real_current_mae": 0.60,
                    "val_calendar": 0.0,
                    "val_butterfly": 0.0,
                    "val_penalty_mean": 0.0,
                    "val_penalty_std": 0.0,
                    "val_weight_entropy": 1.0,
                },
            ]

            with patch.object(trainer, "setup", side_effect=fake_setup), \
                patch.object(
                    trainer,
                    "_discriminator_step",
                    return_value={"d_total": 0.0, "d_real": 0.0, "d_fake": 0.0, "gp": 0.0},
                ), \
                patch.object(
                    trainer,
                    "_generator_step",
                    return_value={
                        "g_total": 0.0,
                        "g_adv": 0.0,
                        "g_adv_effective_lambda": 0.0,
                        "g_calendar": 0.0,
                        "g_butterfly": 0.0,
                        "g_smooth": 0.0,
                        "g_recon": 0.0,
                        "g_recon_weighted": 0.0,
                        "g_atm_short": 0.0,
                    },
                ), \
                patch.object(trainer, "_evaluate", side_effect=eval_rows), \
                patch.object(trainer, "_save_loss_curves", return_value=None), \
                patch.object(trainer, "_checkpoint_payload", return_value={"state": "ok"}):
                trainer.train()

            run_dir = next(path for path in output_root.iterdir() if path.is_dir())
            checkpoints_dir = run_dir / "checkpoints"
            metrics_dir = run_dir / "metrics"

            best_checkpoint = json.loads((metrics_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(best_checkpoint["checkpoint_metric"], "val_atm_short_pure_mae_gap_vs_current")
            self.assertEqual(int(best_checkpoint["best_epoch"]), 1)
            self.assertAlmostEqual(float(best_checkpoint["best_metric"]), -0.40, places=6)
            self.assertTrue(bool(best_checkpoint["early_stopped"]))

            best_metrics_summary = json.loads((metrics_dir / "best_metrics_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(best_metrics_summary["primary_metric"], "val_atm_short_pure_mae_gap_vs_current")
            primary_summary = best_metrics_summary["metrics"]["val_atm_short_pure_mae_gap_vs_current"]
            overall_summary = best_metrics_summary["metrics"]["val_mae_gap_vs_current"]
            short_atm_summary = best_metrics_summary["metrics"]["val_short_atm_mae_gap_vs_current"]

            self.assertEqual(int(primary_summary["best_epoch"]), 1)
            self.assertAlmostEqual(float(primary_summary["best_value"]), -0.40, places=6)
            self.assertEqual(primary_summary["tracking_mode"], "primary")
            self.assertTrue(bool(primary_summary["is_primary"]))
            self.assertFalse(bool(primary_summary["is_extra"]))

            self.assertEqual(int(overall_summary["best_epoch"]), 2)
            self.assertAlmostEqual(float(overall_summary["best_value"]), -0.50, places=6)
            self.assertEqual(overall_summary["tracking_mode"], "extra")
            self.assertFalse(bool(overall_summary["is_primary"]))
            self.assertTrue(bool(overall_summary["is_extra"]))

            self.assertEqual(int(short_atm_summary["best_epoch"]), 2)
            self.assertAlmostEqual(float(short_atm_summary["best_value"]), -0.60, places=6)
            self.assertEqual(short_atm_summary["tracking_mode"], "extra")

            self.assertTrue((checkpoints_dir / "film_wgan_best.pt").exists())
            self.assertTrue((checkpoints_dir / "film_wgan_best_val_mae_gap_vs_current.pt").exists())
            self.assertTrue((checkpoints_dir / "film_wgan_best_val_short_atm_mae_gap_vs_current.pt").exists())


class TestFilmWGANInitializationSmoke(unittest.TestCase):
    def test_train_lp_gen128_disc128_first_batch_stays_finite_and_unsaturated(self):
        if not FILM_TRAIN_CONFIG_PATH.exists():
            self.skipTest(f"Missing training config: {FILM_TRAIN_CONFIG_PATH}")

        config = load_train_config(FILM_TRAIN_CONFIG_PATH)
        workbook_path = ROOT_DIR / str(config.data_path)
        if not workbook_path.exists():
            self.skipTest(f"Missing merged-vol workbook required for smoke check: {workbook_path}")

        config = replace(config, data_path=str(workbook_path), cuda=False, num_workers=0)
        bundle = create_train_val_bundle(config)
        normalization = normalization_stats_to_tensors(bundle.normalization_stats, torch.device("cpu"))
        surface_height, surface_width = bundle.surface_shape

        generator = FilmWGANGenerator(
            surface_height=surface_height,
            surface_width=surface_width,
            embedding_dim=bundle.embedding_dim,
            noise_dim=config.noise_dim,
            base_channels=config.gen_base_channels,
            res_blocks=config.gen_res_blocks,
            text_hidden_dim=config.text_hidden_dim,
            text_out_dim=config.text_out_dim,
            fusion_hidden_dim=config.fusion_hidden_dim,
        )
        critic = FilmWGANCritic(
            surface_height=surface_height,
            surface_width=surface_width,
            embedding_dim=bundle.embedding_dim,
            base_channels=config.disc_base_channels,
            res_blocks=config.disc_res_blocks,
            text_hidden_dim=config.text_hidden_dim,
            text_out_dim=config.text_out_dim,
            fusion_hidden_dim=config.fusion_hidden_dim,
        )

        current_features, text_features, _real_delta_norm, current_flat, target_flat = next(iter(bundle.train_loader))
        batch_size = min(8, current_features.size(0))
        current_features = current_features[:batch_size]
        text_features = text_features[:batch_size]
        current_flat = current_flat[:batch_size]
        target_flat = target_flat[:batch_size]
        noise = torch.randn(batch_size, int(config.noise_dim), dtype=torch.float32)

        with torch.no_grad():
            encoded_text = generator.text_encoder(text_features)
            gamma, beta = generator.film1.projection(encoded_text).chunk(2, dim=-1)
            fake_delta_norm = generator(current_features, text_features, noise=noise)
            fake_delta = denormalize_tensor(fake_delta_norm, normalization.delta_mean, normalization.delta_std)
            fake_future_flat = reconstruct_future_surface(current_flat, fake_delta)
            fake_future_surface = normalize_surface_tensor(
                fake_future_flat,
                normalization.current_log_mean,
                normalization.current_log_std,
            ).view(batch_size, 1, surface_height, surface_width)
            real_future_surface = normalize_surface_tensor(
                target_flat,
                normalization.current_log_mean,
                normalization.current_log_std,
            ).view(batch_size, 1, surface_height, surface_width)

        gp = gradient_penalty(
            critic=critic,
            real_future_surface=real_future_surface,
            fake_future_surface=fake_future_surface,
            current_surface=current_features,
            text_embedding=text_features,
            lambda_gp=10.0,
        )

        frac_floor = (fake_future_flat <= 1.00001e-4).float().mean().item()
        frac_ceil = (fake_future_flat >= 4.99999).float().mean().item()

        self.assertTrue(torch.equal(gamma, torch.zeros_like(gamma)))
        self.assertTrue(torch.equal(beta, torch.zeros_like(beta)))
        self.assertTrue(torch.isfinite(fake_delta_norm).all())
        self.assertTrue(torch.isfinite(fake_delta).all())
        self.assertTrue(torch.isfinite(fake_future_flat).all())
        self.assertTrue(torch.isfinite(gp))
        self.assertLess(frac_floor + frac_ceil, 1.0)


class TestFilmWGANShortATMStudy(unittest.TestCase):
    def test_short_atm_study_grid_matches_expected_2x2_matrix(self):
        specs = build_short_atm_grid_specs()

        self.assertEqual(
            [
                (
                    spec.config_id,
                    spec.recon_atm_range,
                    spec.recon_atm_short_end_max_days,
                    spec.recon_atm_multiplier,
                    spec.lambda_atm_short,
                )
                for spec in specs
            ],
            [
                ("A", 0.04, 60.0, 16.0, 50.0),
                ("B", 0.04, 60.0, 16.0, 75.0),
                ("C", 0.06, 90.0, 8.0, 50.0),
                ("D", 0.06, 90.0, 8.0, 75.0),
            ],
        )

    def test_short_atm_selection_rule_prefers_qualified_config(self):
        primary_rows = [
            {
                "config_id": "A",
                "val_atm_short_pure_mae_gap_vs_current_mean": -0.00118,
                "val_mae_gap_vs_current_mean": -0.00210,
                "val_calendar_mean": 1.3e-5,
                "val_butterfly_mean": 9.0e-7,
            },
            {
                "config_id": "B",
                "val_atm_short_pure_mae_gap_vs_current_mean": -0.00150,
                "val_mae_gap_vs_current_mean": -0.00205,
                "val_calendar_mean": 1.2e-5,
                "val_butterfly_mean": 8.5e-7,
            },
            {
                "config_id": "C",
                "val_atm_short_pure_mae_gap_vs_current_mean": -0.00160,
                "val_mae_gap_vs_current_mean": -0.00120,
                "val_calendar_mean": 1.1e-5,
                "val_butterfly_mean": 7.5e-7,
            },
            {
                "config_id": "D",
                "val_atm_short_pure_mae_gap_vs_current_mean": -0.00125,
                "val_mae_gap_vs_current_mean": -0.00195,
                "val_calendar_mean": 2.5e-5,
                "val_butterfly_mean": 8.0e-7,
            },
        ]

        recommended = choose_recommended_config(primary_rows)
        top_two = select_blend_scan_config_ids(primary_rows, top_k=2)

        self.assertEqual(recommended["recommended_config_id"], "B")
        self.assertEqual(recommended["qualified_config_ids"], ["B"])
        self.assertEqual(top_two, ["C", "B"])
