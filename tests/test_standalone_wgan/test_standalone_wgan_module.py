import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trainer import BaseTrainer  # noqa: E402
from cnn_wgan.config import build_sample_config_from_train_config, load_train_config as load_cnn_train_config  # noqa: E402
from cnn_wgan.trainer import CnnWGANTrainer  # noqa: E402
from transformer_wgan.config import (  # noqa: E402
    build_sample_config_from_train_config as build_transformer_sample_config_from_train_config,
)
from transformer_wgan.config import load_train_config as load_transformer_train_config  # noqa: E402
from transformer_wgan.trainer import TransformerWGANTrainer  # noqa: E402


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


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return path


class TestStandaloneWGAN(unittest.TestCase):
    def test_package_exports_and_pyproject_include_new_packages(self):
        pyproject_text = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("cnn_wgan*", pyproject_text)
        self.assertIn("transformer_wgan*", pyproject_text)
        self.assertTrue(issubclass(CnnWGANTrainer, BaseTrainer))
        self.assertTrue(issubclass(TransformerWGANTrainer, BaseTrainer))

    def test_merged_config_loading_and_generate_result_derivation(self):
        cnn_config = load_cnn_train_config(ROOT_DIR / "configs/cnn_wgan/train_lp_gen128_disc128.yaml")
        transformer_config = load_transformer_train_config(ROOT_DIR / "configs/transformer_wgan/train_lp.yaml")
        self.assertEqual(cnn_config.text_embedding_mode, "lp")
        self.assertEqual(transformer_config.text_embedding_mode, "lp")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run" / "20260413_010101"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints" / "cnn_wgan_best.pt").write_bytes(b"x")
            (run_dir / "checkpoints" / "transformer_wgan_best.pt").write_bytes(b"y")

            cnn_sample = build_sample_config_from_train_config(
                ROOT_DIR / "configs/cnn_wgan/train_lp_gen128_disc128.yaml",
                run_dir=run_dir,
            )
            transformer_sample = build_transformer_sample_config_from_train_config(
                ROOT_DIR / "configs/transformer_wgan/train_lp.yaml",
                run_dir=run_dir,
            )

            self.assertEqual(Path(cnn_sample.output_dir), run_dir / "generate_result" / "cnn_wgan_best")
            self.assertEqual(Path(transformer_sample.output_dir), run_dir / "generate_result" / "transformer_wgan_best")
            self.assertEqual(Path(cnn_sample.checkpoint_path), run_dir / "checkpoints" / "cnn_wgan_best.pt")
            self.assertEqual(
                Path(transformer_sample.checkpoint_path),
                run_dir / "checkpoints" / "transformer_wgan_best.pt",
            )

    def test_cnn_explicit_generate_result_output_dir_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            run_dir = Path(tmpdir) / "training" / "cnn_wgan" / "20260413_010101"
            checkpoint_path = run_dir / "checkpoints" / "cnn_wgan_epoch_0130.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"x")

            config_path = _write_yaml(
                Path(tmpdir) / "cnn_generate_epoch.yaml",
                {
                    "training": {
                        "data_path": str(workbook_path),
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
                        "batch_size": 2,
                        "eval_mc_samples": 2,
                        "cuda": False,
                        "num_workers": 0,
                        "output_root": str(run_dir.parent),
                        "save_every": 1,
                    },
                    "generate_result": {
                        "checkpoint_path": str(checkpoint_path),
                        "mc_samples": 2,
                        "output_dir": "generate_result/epoch_0130",
                        "save_plots": False,
                        "save_json": True,
                    },
                },
            )

            trainer = CnnWGANTrainer(load_cnn_train_config(config_path), config_path=str(config_path))
            resolved_config, generate_dir = trainer._prepare_generate_result(config_path=str(config_path))

            self.assertEqual(generate_dir, run_dir / "generate_result" / "epoch_0130")
            self.assertEqual(Path(resolved_config.checkpoint_path), checkpoint_path)
            self.assertEqual(Path(resolved_config.output_dir), generate_dir)

    def test_cnn_pipeline_writes_generate_result_under_training_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            output_root = Path(tmpdir) / "training" / "cnn_wgan"
            config_path = _write_yaml(
                Path(tmpdir) / "cnn_pipeline.yaml",
                {
                    "training": {
                        "data_path": str(workbook_path),
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
                        "batch_size": 2,
                        "eval_mc_samples": 2,
                        "cuda": False,
                        "num_workers": 0,
                        "output_root": str(output_root),
                        "save_every": 1,
                    },
                    "generate_result": {
                        "mc_samples": 2,
                        "save_plots": False,
                        "save_json": True,
                    },
                },
            )

            trainer = CnnWGANTrainer(load_cnn_train_config(config_path), config_path=str(config_path))
            generate_dir = trainer.run_pipeline(config_path=str(config_path))

            run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            training_run_dir = run_dirs[0]
            self.assertEqual(generate_dir, training_run_dir / "generate_result" / "cnn_wgan_best")
            self.assertTrue((training_run_dir / "run.log").exists())
            self.assertTrue((training_run_dir / "metrics" / "training_resolved_config.yaml").exists())
            self.assertTrue((generate_dir / "generate_resolved_config.yaml").exists())
            self.assertTrue((training_run_dir / "checkpoints" / "cnn_wgan_best.pt").exists())
            self.assertTrue((generate_dir / "summary.csv").exists())

    def test_transformer_pipeline_writes_generate_result_under_training_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            output_root = Path(tmpdir) / "training" / "transformer_wgan"
            config_path = _write_yaml(
                Path(tmpdir) / "transformer_pipeline.yaml",
                {
                    "training": {
                        "data_path": str(workbook_path),
                        "sheet_name": "gan_input_ready",
                        "text_embedding_mode": "lp",
                        "train_ratio": 0.67,
                        "min_samples_for_training": 2,
                        "noise_dim": 4,
                        "model_dim": 8,
                        "gen_layers": 1,
                        "disc_layers": 1,
                        "num_heads": 2,
                        "ffn_dim": 16,
                        "dropout": 0.0,
                        "text_hidden_dim": 8,
                        "text_token_dim": 8,
                        "noise_hidden_dim": 8,
                        "num_epochs": 1,
                        "batch_size": 2,
                        "eval_mc_samples": 2,
                        "cuda": False,
                        "num_workers": 0,
                        "output_root": str(output_root),
                        "save_every": 1,
                    },
                    "generate_result": {
                        "mc_samples": 2,
                        "save_plots": False,
                        "save_json": True,
                    },
                },
            )

            trainer = TransformerWGANTrainer(
                load_transformer_train_config(config_path),
                config_path=str(config_path),
            )
            generate_dir = trainer.run_pipeline(config_path=str(config_path))

            run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            training_run_dir = run_dirs[0]
            self.assertEqual(generate_dir, training_run_dir / "generate_result" / "transformer_wgan_best")
            self.assertTrue((training_run_dir / "run.log").exists())
            self.assertTrue((training_run_dir / "metrics" / "training_resolved_config.yaml").exists())
            self.assertTrue((generate_dir / "generate_resolved_config.yaml").exists())
            self.assertTrue((training_run_dir / "checkpoints" / "transformer_wgan_best.pt").exists())
            self.assertTrue((generate_dir / "summary.csv").exists())

    def test_main_scripts_dispatch_subcommands(self):
        cnn_main = _load_script_module(ROOT_DIR / "scripts/cnn_wgan/main.py", "cnn_wgan_main_router")
        transformer_main = _load_script_module(
            ROOT_DIR / "scripts/transformer_wgan/main.py",
            "transformer_wgan_main_router",
        )

        with patch.object(cnn_main, "train_main", return_value="cnn-train") as cnn_train:
            self.assertEqual(cnn_main.main(["train", "--config", "cnn.yaml"]), "cnn-train")
            cnn_train.assert_called_once_with(["--config", "cnn.yaml"])

        with patch.object(transformer_main, "generate_result_main", return_value="transformer-gen") as transformer_gen:
            self.assertEqual(
                transformer_main.main(["generate-result", "--config", "transformer.yaml"]),
                "transformer-gen",
            )
            transformer_gen.assert_called_once_with(["--config", "transformer.yaml"])

    def test_shell_wrappers_parse_cleanly(self):
        subprocess.run(["bash", "-n", str(ROOT_DIR / "run_cnn_wgan_svi_excel.sh")], check=True)
        subprocess.run(["bash", "-n", str(ROOT_DIR / "run_transformer_wgan_svi_excel.sh")], check=True)
