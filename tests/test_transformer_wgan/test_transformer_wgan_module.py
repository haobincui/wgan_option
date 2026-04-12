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

import transformer_wgan  # noqa: E402
from transformer_wgan.config import (  # noqa: E402
    TransformerWGANSampleConfig,
    TransformerWGANTrainConfig,
    load_sample_config,
    load_train_config,
)
from transformer_wgan.data import create_train_val_bundle, denormalize_tensor, load_transformer_wgan_samples, normalize_tensor  # noqa: E402
from transformer_wgan.io import ensure_dir, load_checkpoint  # noqa: E402
from transformer_wgan.losses import (  # noqa: E402
    assemble_generator_loss,
    butterfly_arbitrage_penalty,
    calendar_arbitrage_penalty,
    gradient_penalty,
    reweight_scenarios,
    resolve_monitor_metric,
    smoothness_penalty,
)
from transformer_wgan.models import TransformerWGANCritic, TransformerWGANGenerator, reconstruct_future_surface  # noqa: E402
from transformer_wgan.training_plots import plot_training_curves  # noqa: E402


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


class TestStandaloneTransformerWgan(unittest.TestCase):
    def test_package_is_importable_and_pyproject_includes_transformer_wgan(self):
        self.assertTrue(hasattr(transformer_wgan, "TransformerWGANTrainer"))
        pyproject_text = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("transformer_wgan*", pyproject_text)

    def test_config_loading_from_default_yaml(self):
        train_config = load_train_config(ROOT_DIR / "configs/transformer_wgan/train_default.yaml")
        sample_config = load_sample_config(ROOT_DIR / "configs/transformer_wgan/sample_default.yaml")
        self.assertIsInstance(train_config, TransformerWGANTrainConfig)
        self.assertIsInstance(sample_config, TransformerWGANSampleConfig)
        self.assertEqual(train_config.text_embedding_mode, "hd")
        self.assertEqual(sample_config.split, "val")

    def test_workbook_parsing_is_chronological_and_embedding_modes_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            config = TransformerWGANTrainConfig(
                data_path=str(workbook_path),
                text_embedding_mode="lp",
                train_ratio=0.67,
                batch_size=2,
                num_workers=0,
                cuda=False,
            )
            samples = load_transformer_wgan_samples(config)
            self.assertEqual([sample.sample_id for sample in samples], ["news_1", "news_2", "news_3"])
            bundle = create_train_val_bundle(config)
            self.assertEqual(bundle.train_samples, 2)
            self.assertEqual(bundle.val_samples, 1)
            batch = next(iter(bundle.train_loader))
            self.assertEqual(tuple(batch[0].shape[1:]), (1, 4, 4))

    def test_models_and_losses_behave_as_expected(self):
        generator = TransformerWGANGenerator(
            surface_height=4,
            surface_width=4,
            embedding_dim=2,
            noise_dim=4,
            model_dim=16,
            layers=2,
            num_heads=4,
            ffn_dim=32,
            dropout=0.1,
            text_hidden_dim=8,
            text_token_dim=8,
            noise_hidden_dim=8,
        )
        critic = TransformerWGANCritic(
            surface_height=4,
            surface_width=4,
            embedding_dim=2,
            model_dim=16,
            layers=2,
            num_heads=4,
            ffn_dim=32,
            dropout=0.1,
            text_hidden_dim=8,
            text_token_dim=8,
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

        smooth = smoothness_penalty(torch.zeros((2, 4, 4), dtype=torch.float32))
        self.assertAlmostEqual(float(smooth.item()), 0.0, places=6)

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
        self.assertGreater(float(cal_pen.mean().item()), 0.0)
        self.assertGreaterEqual(float(bfly_pen.mean().item()), 0.0)
        weights, beta = reweight_scenarios(np.asarray([0.1, 1.0, 2.0]), beta_mode="fixed", beta_value=10.0)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=6)
        self.assertEqual(beta, 10.0)

    def test_normalization_and_loss_switch_helpers(self):
        base = torch.tensor(1.0)
        total_supervised = assemble_generator_loss(
            adv_loss=base,
            recon_loss=base,
            calendar_penalty=base,
            butterfly_penalty=base,
            smooth_penalty_value=base,
            delta_shrink=base,
            pure_adversarial=False,
            lambda_recon=10.0,
            lambda_calendar=2.0,
            lambda_butterfly=2.0,
            lambda_smooth=0.1,
            lambda_delta_shrink=0.5,
            use_calendar_constraint=True,
            use_butterfly_constraint=True,
            use_smooth_constraint=True,
        )
        total_pure_adv = assemble_generator_loss(
            adv_loss=base,
            recon_loss=base,
            calendar_penalty=base,
            butterfly_penalty=base,
            smooth_penalty_value=base,
            delta_shrink=base,
            pure_adversarial=True,
            lambda_recon=10.0,
            lambda_calendar=2.0,
            lambda_butterfly=2.0,
            lambda_smooth=0.1,
            lambda_delta_shrink=0.5,
            use_calendar_constraint=True,
            use_butterfly_constraint=True,
            use_smooth_constraint=True,
        )
        self.assertAlmostEqual(float(total_supervised.item()), 15.6, places=5)
        self.assertAlmostEqual(float(total_pure_adv.item()), 5.1, places=5)
        self.assertAlmostEqual(float(resolve_monitor_metric({"val_recon": 0.1}, "val_recon")), 0.1, places=6)

        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            config = TransformerWGANTrainConfig(
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
            norm = normalize_tensor(
                current_flat,
                torch.tensor(stats.current_mean.reshape(1, -1), dtype=torch.float32),
                torch.tensor(stats.current_std.reshape(1, -1), dtype=torch.float32),
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
                        "g_recon": 0.2,
                        "g_delta_shrink": 0.1,
                        "g_calendar": 0.1,
                        "g_butterfly": 0.1,
                        "g_smooth": 0.1,
                        "d_total": 0.7,
                        "gp": 0.2,
                        "val_recon": 0.2,
                        "val_current_recon": 0.25,
                        "val_baseline_gap": -0.05,
                        "val_hybrid_score": 0.2,
                        "val_calendar": 0.01,
                        "val_butterfly": 0.02,
                        "val_delta_shrink": 0.03,
                    }
                ],
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())

    def test_train_and_sample_smoke_via_cli_for_both_modes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            train_output_root = Path(tmpdir) / "train_outputs"
            sample_output_root = Path(tmpdir) / "sample_outputs"
            pure_output_root = Path(tmpdir) / "pure_train_outputs"
            ensure_dir(train_output_root)
            ensure_dir(sample_output_root)
            ensure_dir(pure_output_root)

            train_config_path = Path(tmpdir) / "train.yaml"
            pure_train_config_path = Path(tmpdir) / "train_pure.yaml"
            sample_config_path = Path(tmpdir) / "sample.yaml"

            train_config_path.write_text(
                "\n".join(
                    [
                        f"data_path: {workbook_path}",
                        "sheet_name: gan_input_ready",
                        "text_embedding_mode: hd",
                        "train_ratio: 0.67",
                        "noise_dim: 4",
                        "model_dim: 16",
                        "gen_layers: 2",
                        "disc_layers: 2",
                        "num_heads: 4",
                        "ffn_dim: 32",
                        "dropout: 0.1",
                        "text_hidden_dim: 8",
                        "text_token_dim: 8",
                        "noise_hidden_dim: 8",
                        "learning_rate: 0.0001",
                        "generator_learning_rate: 0.0001",
                        "discriminator_learning_rate: 0.0001",
                        "num_epochs: 1",
                        "batch_size: 2",
                        "beta_1: 0.5",
                        "beta_2: 0.9",
                        "critic_iter: 1",
                        "lambda_gp: 1.0",
                        "pure_adversarial: false",
                        "lambda_recon: 1.0",
                        "lambda_calendar: 1.0",
                        "lambda_butterfly: 1.0",
                        "lambda_smooth: 0.1",
                        "lambda_delta_shrink: 0.1",
                        "use_calendar_constraint: true",
                        "use_butterfly_constraint: true",
                        "use_smooth_constraint: true",
                        "best_checkpoint_metric: val_recon",
                        "baseline_penalty_weight: 2.0",
                        "seed: 7",
                        "cuda: false",
                        "num_workers: 0",
                        f"output_root: {train_output_root}",
                        "save_every: 1",
                    ]
                ),
                encoding="utf-8",
            )
            pure_train_config_path.write_text(
                train_config_path.read_text(encoding="utf-8").replace("pure_adversarial: false", "pure_adversarial: true").replace(
                    f"output_root: {train_output_root}",
                    f"output_root: {pure_output_root}",
                ),
                encoding="utf-8",
            )

            main_module = _load_script_module(ROOT_DIR / "scripts/transformer_wgan/main.py", "transformer_wgan_main_script")
            train_run_dir = Path(main_module.main(["train", "--config", str(train_config_path)]))
            pure_train_run_dir = Path(main_module.main(["train", "--config", str(pure_train_config_path)]))
            checkpoint_path = train_run_dir / "checkpoints" / "transformer_wgan_best.pt"
            pure_checkpoint_path = pure_train_run_dir / "checkpoints" / "transformer_wgan_best.pt"
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(pure_checkpoint_path.exists())
            self.assertTrue((train_run_dir / "metrics" / "training_metrics.json").exists())
            self.assertTrue((train_run_dir / "metrics" / "loss_curves.png").exists())
            checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
            pure_checkpoint = load_checkpoint(pure_checkpoint_path, torch.device("cpu"))
            self.assertIn("normalization_stats", checkpoint)
            self.assertFalse(bool(checkpoint["config"]["pure_adversarial"]))
            self.assertTrue(bool(pure_checkpoint["config"]["pure_adversarial"]))

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
            self.assertIn("val_recon", metrics_rows[-1])
            self.assertIn("val_baseline_gap", metrics_rows[-1])
            best_payload = json.loads((train_run_dir / "metrics" / "best_checkpoint.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(float(best_payload["best_metric"]), float(metrics_rows[-1]["val_recon"]), places=6)


if __name__ == "__main__":
    unittest.main()
