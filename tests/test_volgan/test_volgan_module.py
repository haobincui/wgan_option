import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import volgan  # noqa: E402
from utils.output_paths import default_output_root, find_best_checkpoint, find_latest_run_dir  # noqa: E402
from volgan.arbitrage import (  # noqa: E402
    butterfly_arbitrage_penalty,
    calendar_arbitrage_penalty,
    reweight_scenarios,
)
from volgan.config import (  # noqa: E402
    VolGANSampleConfig,
    VolGANTrainConfig,
    build_sample_config_from_train_config,
    load_sample_config,
    load_train_config,
)
from volgan.data import (  # noqa: E402
    create_train_val_bundle,
    denormalize_tensor,
    load_vol_surface_samples,
    normalize_current_surface_tensor,
)
from volgan.io import ensure_dir, load_checkpoint  # noqa: E402
from volgan.losses import (  # noqa: E402
    estimate_gradient_matching,
    maturity_smoothness_penalty,
    strike_smoothness_penalty,
)
from volgan.models import VolGANDiscriminator, VolGANGenerator, reconstruct_future_surface  # noqa: E402
from volgan.training_plots import plot_training_curves  # noqa: E402


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
            {
                "sample_id": "news_skip",
                "news_timestamp_utc": "2022-12-30T13:20:00Z",
                "current_snapshot_time_utc": "2022-12-30T13:20:00Z",
                "target_snapshot_time_utc": "2022-12-30T13:25:00Z",
                "hd_embedding": _json_text([7.0, 8.0]),
                "lp_embedding": _json_text([90.0, 91.0, 92.0]),
                "strike_grid": _json_text(strike_grid),
                "maturity_days_grid": _json_text(maturity_grid),
                "current_surface_flat": _json_text(_surface_values(7.0)),
                "target_surface_flat": _json_text(_surface_values(8.0)),
                "pair_quality_label": "filtered",
                "training_candidate_flag": 0,
            },
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name="gan_input_ready", index=False)
    return path


class TestStandaloneVolgan(unittest.TestCase):
    def test_package_is_importable_and_pyproject_includes_volgan(self):
        self.assertTrue(hasattr(volgan, "VolGANTrainer"))
        pyproject_text = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("volgan*", pyproject_text)
        self.assertIn("utils*", pyproject_text)

    def test_config_loading_from_default_yaml(self):
        train_config = load_train_config(ROOT_DIR / "configs/volgan/train_default.yaml")
        sample_config = load_sample_config(ROOT_DIR / "configs/volgan/sample_default.yaml")
        self.assertIsInstance(train_config, VolGANTrainConfig)
        self.assertIsInstance(sample_config, VolGANSampleConfig)
        self.assertEqual(train_config.text_embedding_mode, "hd")
        self.assertEqual(sample_config.split, "val")

    def test_workbook_parsing_is_chronological_and_embedding_modes_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            hd_config = VolGANTrainConfig(
                data_path=str(workbook_path),
                text_embedding_mode="hd",
                train_ratio=0.67,
                batch_size=2,
                num_workers=0,
                cuda=False,
            )
            lp_config = VolGANTrainConfig(
                data_path=str(workbook_path),
                text_embedding_mode="lp",
                train_ratio=0.67,
                batch_size=2,
                num_workers=0,
                cuda=False,
            )
            concat_config = VolGANTrainConfig(
                data_path=str(workbook_path),
                text_embedding_mode="concat",
                train_ratio=0.67,
                batch_size=2,
                num_workers=0,
                cuda=False,
            )

            hd_samples = load_vol_surface_samples(hd_config)
            lp_samples = load_vol_surface_samples(lp_config)
            concat_samples = load_vol_surface_samples(concat_config)
            self.assertEqual([sample.sample_id for sample in hd_samples], ["news_1", "news_2", "news_3"])
            self.assertEqual(int(hd_samples[0].text_embedding.size), 2)
            self.assertEqual(int(lp_samples[0].text_embedding.size), 3)
            self.assertEqual(int(concat_samples[0].text_embedding.size), 5)

            bundle = create_train_val_bundle(hd_config)
            self.assertEqual(bundle.train_samples, 2)
            self.assertEqual(bundle.val_samples, 1)
            self.assertEqual(bundle.surface_shape, (4, 4))
            self.assertEqual(tuple(bundle.normalization_stats.current_log_mean.shape), (16,))
            first_batch = next(iter(bundle.train_loader))
            self.assertEqual(len(first_batch), 5)
            self.assertEqual(tuple(first_batch[0].shape[1:]), (16,))

    def test_models_and_penalties_behave_as_expected(self):
        generator = VolGANGenerator(surface_dim=16, embedding_dim=2, noise_dim=4, hidden_dim=8)
        discriminator = VolGANDiscriminator(surface_dim=16, embedding_dim=2, hidden_dim=8)
        current_flat = torch.full((3, 16), 0.2, dtype=torch.float32)
        text_embedding = torch.randn(3, 2, dtype=torch.float32)
        delta = generator(current_flat, text_embedding)
        self.assertEqual(tuple(delta.shape), (3, 16))
        scores = discriminator(current_flat, text_embedding, delta)
        self.assertEqual(tuple(scores.shape), (3, 1))

        future_flat = reconstruct_future_surface(current_flat, delta)
        self.assertTrue(torch.all(future_flat > 0.0).item())

        constant_log_surface = torch.zeros((2, 4, 4), dtype=torch.float32)
        strike_grid = torch.tensor([0.8, 1.0, 1.2, 1.4], dtype=torch.float32)
        maturity_days = torch.tensor([30.0, 60.0, 90.0, 120.0], dtype=torch.float32)
        self.assertAlmostEqual(
            float(strike_smoothness_penalty(constant_log_surface, strike_grid).item()),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(maturity_smoothness_penalty(constant_log_surface, maturity_days).item()),
            0.0,
            places=6,
        )

        violating_surface = torch.tensor(
            [[[0.5, 0.5, 0.5], [0.01, 0.01, 0.01]]],
            dtype=torch.float32,
        )
        cal_pen = calendar_arbitrage_penalty(
            violating_surface,
            torch.tensor([0.8, 1.0, 1.2], dtype=torch.float32),
            torch.tensor([30.0, 365.0], dtype=torch.float32),
        )
        bfly_pen = butterfly_arbitrage_penalty(
            torch.tensor(
                [[[0.01, 2.0, 0.01], [0.01, 2.0, 0.01]]],
                dtype=torch.float32,
            ),
            torch.tensor([0.8, 1.0, 1.2], dtype=torch.float32),
            torch.tensor([30.0, 60.0], dtype=torch.float32),
        )
        self.assertGreater(float(cal_pen.item()), 0.0)
        self.assertGreaterEqual(float(bfly_pen.item()), 0.0)

        weights, beta = reweight_scenarios(np.asarray([0.1, 1.0, 2.0]), beta_mode="fixed", beta_value=10.0)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=6)
        self.assertGreater(weights[0], weights[-1])
        self.assertEqual(beta, 10.0)

    def test_normalization_and_gradient_matching_helpers_behave_as_expected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            config = VolGANTrainConfig(
                data_path=str(workbook_path),
                text_embedding_mode="hd",
                train_ratio=0.67,
                batch_size=2,
                num_workers=0,
                cuda=False,
                use_gradient_matching=False,
            )
            bundle = create_train_val_bundle(config)
            stats = bundle.normalization_stats
            current_flat = torch.tensor(bundle.train_items[0].current_surface.reshape(1, -1), dtype=torch.float32)
            current_norm = normalize_current_surface_tensor(
                current_flat,
                torch.tensor(stats.current_log_mean.reshape(1, -1), dtype=torch.float32),
                torch.tensor(stats.current_log_std.reshape(1, -1), dtype=torch.float32),
            )
            delta_norm = torch.zeros_like(current_norm)
            delta = denormalize_tensor(
                delta_norm,
                torch.tensor(stats.delta_mean.reshape(1, -1), dtype=torch.float32),
                torch.tensor(stats.delta_std.reshape(1, -1), dtype=torch.float32),
            )
            self.assertEqual(tuple(current_norm.shape), (1, 16))
            self.assertEqual(tuple(delta.shape), (1, 16))

            generator = VolGANGenerator(surface_dim=16, embedding_dim=2, noise_dim=4, hidden_dim=8)
            discriminator = VolGANDiscriminator(surface_dim=16, embedding_dim=2, hidden_dim=8)
            alpha_m, alpha_tau = estimate_gradient_matching(
                generator=generator,
                discriminator=discriminator,
                train_loader=bundle.train_loader,
                device=torch.device("cpu"),
                strike_grid=torch.tensor(bundle.strike_grid, dtype=torch.float32),
                maturity_days_grid=torch.tensor(bundle.maturity_days_grid, dtype=torch.float32),
                delta_mean=torch.tensor(stats.delta_mean.reshape(1, -1), dtype=torch.float32),
                delta_std=torch.tensor(stats.delta_std.reshape(1, -1), dtype=torch.float32),
                noise_dim=4,
                epochs=1,
                real_label_value=0.9,
                alpha_clip_min=0.2,
                alpha_clip_max=0.3,
                normalize_target_delta=True,
            )
            self.assertGreaterEqual(alpha_m, 0.2)
            self.assertLessEqual(alpha_m, 0.3)
            self.assertGreaterEqual(alpha_tau, 0.2)
            self.assertLessEqual(alpha_tau, 0.3)

    def test_training_plot_helper_writes_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "loss_curves.png"
            plot_training_curves(
                [
                    {
                        "epoch": 1,
                        "g_total": 1.0,
                        "g_adv": 0.8,
                        "g_smooth_m": 0.1,
                        "g_smooth_t": 0.1,
                        "d_total": 0.7,
                        "d_real": 0.6,
                        "d_fake": 0.4,
                        "val_mae": 0.2,
                        "val_rmse": 0.3,
                        "val_calendar": 0.01,
                        "val_butterfly": 0.02,
                        "alpha_m": 1.0,
                        "alpha_tau": 2.0,
                    },
                    {
                        "epoch": 2,
                        "g_total": 0.9,
                        "g_adv": 0.7,
                        "g_smooth_m": 0.1,
                        "g_smooth_t": 0.1,
                        "d_total": 0.6,
                        "d_real": 0.7,
                        "d_fake": 0.3,
                        "val_mae": 0.18,
                        "val_rmse": 0.28,
                        "val_calendar": 0.009,
                        "val_butterfly": 0.018,
                        "alpha_m": 1.0,
                        "alpha_tau": 2.0,
                    },
                ],
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())

    def test_output_helpers_and_sample_config_can_reuse_train_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_dir = Path(tmpdir) / "data" / "processed" / "svi-excel" / "20260410-174929"
            workbook_dir.mkdir(parents=True, exist_ok=True)
            workbook_path = _write_vol_workbook(str(workbook_dir))
            train_output_root = Path(tmpdir) / "train_outputs"

            older_run_dir = train_output_root / "20260410_120000"
            newer_run_dir = train_output_root / "20260410_130000"
            older_run_dir.mkdir(parents=True, exist_ok=True)
            newer_run_dir.mkdir(parents=True, exist_ok=True)
            (older_run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (newer_run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (older_run_dir / "checkpoints" / "volgan_best.pt").write_bytes(b"older")
            (newer_run_dir / "checkpoints" / "volgan_best.pt").write_bytes(b"newer")

            train_config_path = Path(tmpdir) / "train.yaml"
            train_config_path.write_text(
                "\n".join(
                    [
                        f"data_path: {workbook_path}",
                        "sheet_name: gan_input_ready",
                        "text_embedding_mode: lp",
                        "train_ratio: 0.67",
                        "eval_mc_samples: 9",
                        "eval_reweight_beta_mode: adaptive",
                        "eval_reweight_beta: 12.5",
                        "eval_aggregation_mode: weighted_mean",
                        "seed: 11",
                        "cuda: false",
                        f"output_root: {train_output_root}",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(find_latest_run_dir(train_output_root), newer_run_dir)
            self.assertEqual(find_best_checkpoint(newer_run_dir), newer_run_dir / "checkpoints" / "volgan_best.pt")
            self.assertEqual(
                default_output_root("outputs/training/volgan", workbook_path),
                Path("outputs/training/volgan/svi-excel"),
            )

            sample_config = build_sample_config_from_train_config(train_config_path)
            self.assertEqual(sample_config.data_path, str(workbook_path))
            self.assertEqual(sample_config.text_embedding_mode, "lp")
            self.assertEqual(sample_config.seed, 11)
            self.assertFalse(sample_config.cuda)
            self.assertEqual(sample_config.mc_samples, 9)
            self.assertEqual(sample_config.reweight_beta_mode, "adaptive")
            self.assertEqual(sample_config.reweight_beta, 12.5)
            self.assertEqual(
                sample_config.checkpoint_path,
                str(newer_run_dir / "checkpoints" / "volgan_best.pt"),
            )
            self.assertEqual(sample_config.output_dir, str(newer_run_dir / "generate_result"))

    def test_volgan_main_routes_pipeline_command(self):
        main_module = _load_script_module(ROOT_DIR / "scripts/volgan/main.py", "volgan_main_router")
        with patch.object(main_module, "pipeline_main", return_value=Path("/tmp/pipeline")) as mock_pipeline:
            result = main_module.main(["pipeline", "--config", "demo.yaml"])

        mock_pipeline.assert_called_once_with(["--config", "demo.yaml"])
        self.assertEqual(result, Path("/tmp/pipeline"))

    def test_train_and_sample_smoke_via_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            train_output_root = Path(tmpdir) / "train_outputs"
            ensure_dir(train_output_root)

            train_config_path = Path(tmpdir) / "train.yaml"
            train_config_path.write_text(
                "\n".join(
                    [
                        f"data_path: {workbook_path}",
                        "sheet_name: gan_input_ready",
                        "text_embedding_mode: hd",
                        "train_ratio: 0.67",
                        "noise_dim: 4",
                        "hidden_dim: 8",
                        "num_epochs: 1",
                        "batch_size: 2",
                        "learning_rate: 0.0001",
                        "use_gradient_matching: false",
                        "alpha_m: 1.0",
                        "alpha_tau: 1.0",
                        "eval_mc_samples: 4",
                        "disc_steps_per_batch: 1",
                        "gen_steps_per_batch: 1",
                        "seed: 7",
                        "cuda: false",
                        "num_workers: 0",
                        f"output_root: {train_output_root}",
                        "save_every: 1",
                    ]
                ),
                encoding="utf-8",
            )

            main_module = _load_script_module(ROOT_DIR / "scripts/volgan/main.py", "volgan_main_script")
            train_run_dir = Path(main_module.main(["train", "--config", str(train_config_path)]))
            checkpoint_path = train_run_dir / "checkpoints" / "volgan_best.pt"
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue((train_run_dir / "metrics" / "training_metrics.json").exists())
            self.assertTrue((train_run_dir / "metrics" / "loss_curves.png").exists())
            checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
            self.assertIn("normalization_stats", checkpoint)

            sample_run_dir = Path(main_module.main(["sample", "--config", str(train_config_path)]))
            self.assertEqual(sample_run_dir, train_run_dir / "generate_result")
            self.assertTrue((sample_run_dir / "summary.csv").exists())
            sample_jsons = sorted((sample_run_dir / "samples").glob("*.json"))
            self.assertEqual(len(sample_jsons), 1)
            sample_pngs = sorted((sample_run_dir / "plots").glob("*.png"))
            self.assertGreaterEqual(len(sample_pngs), 1)
            payload = json.loads(sample_jsons[0].read_text(encoding="utf-8"))
            self.assertIn("generated_surface", payload)
            self.assertIn("weights", payload)
            self.assertIn("generated_current_metrics", payload)
            self.assertIn("weight_entropy", payload)

            metrics_rows = json.loads((train_run_dir / "metrics" / "training_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("val_mae_gap_vs_current", metrics_rows[-1])
            self.assertIn("val_penalty_mean", metrics_rows[-1])
            self.assertIn("val_weight_entropy", metrics_rows[-1])
            self.assertAlmostEqual(float(metrics_rows[-1]["val_mae"]), float(payload["metrics"]["mae"]), places=6)
            self.assertAlmostEqual(
                float(metrics_rows[-1]["val_current_mae"]),
                float(payload["current_metrics"]["mae"]),
                places=6,
            )
            best_payload = json.loads((train_run_dir / "metrics" / "best_checkpoint.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                float(best_payload["best_metric"]),
                float(metrics_rows[-1]["val_mae_gap_vs_current"]),
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
