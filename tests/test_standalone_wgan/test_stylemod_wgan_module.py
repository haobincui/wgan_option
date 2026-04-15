import importlib.util
import json
import sys
import tempfile
import unittest
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

from trainer import BaseTrainer  # noqa: E402
from stylemod_wgan.config import (  # noqa: E402
    build_sample_config_from_train_config,
    load_train_config,
)
from stylemod_wgan.layers import ModulatedConv2d  # noqa: E402
from stylemod_wgan.models import StyleModWGANGenerator  # noqa: E402
from stylemod_wgan.trainer import StyleModWGANTrainer  # noqa: E402


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


class TestStyleModWGAN(unittest.TestCase):
    def test_package_exports_and_pyproject_include_stylemod_package(self):
        pyproject_text = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("stylemod_wgan*", pyproject_text)
        self.assertTrue(issubclass(StyleModWGANTrainer, BaseTrainer))

    def test_generate_result_derivation_uses_stylemod_checkpoint_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run" / "20260413_010101"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints" / "stylemod_wgan_best.pt").write_bytes(b"x")

            sample_config = build_sample_config_from_train_config(
                ROOT_DIR / "configs/stylemod_wgan/train_lp_gen128_disc128.yaml",
                run_dir=run_dir,
            )

            self.assertEqual(Path(sample_config.output_dir), run_dir / "generate_result" / "stylemod_wgan_best")
            self.assertEqual(Path(sample_config.checkpoint_path), run_dir / "checkpoints" / "stylemod_wgan_best.pt")

    def test_stylemod_pipeline_writes_generate_result_under_training_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_vol_workbook(tmpdir)
            output_root = Path(tmpdir) / "training" / "stylemod_wgan"
            config_path = _write_yaml(
                Path(tmpdir) / "stylemod_pipeline.yaml",
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
                        "style_dim": 8,
                        "style_noise_scale": 0.2,
                        "style_demodulate": True,
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

            trainer = StyleModWGANTrainer(load_train_config(config_path), config_path=str(config_path))
            generate_dir = trainer.run_pipeline(config_path=str(config_path))

            run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            training_run_dir = run_dirs[0]
            self.assertEqual(generate_dir, training_run_dir / "generate_result" / "stylemod_wgan_best")
            self.assertTrue((training_run_dir / "run.log").exists())
            self.assertTrue((training_run_dir / "metrics" / "training_resolved_config.yaml").exists())
            self.assertTrue((generate_dir / "generate_resolved_config.yaml").exists())
            self.assertTrue((training_run_dir / "checkpoints" / "stylemod_wgan_best.pt").exists())
            self.assertTrue((generate_dir / "summary.csv").exists())
            sample_json = next((generate_dir / "samples").glob("*.json"))
            payload = json.loads(sample_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "stylemod_wgan")

    def test_main_script_dispatches_subcommands(self):
        module = _load_script_module(ROOT_DIR / "scripts/stylemod_wgan/main.py", "stylemod_wgan_main_router")

        with patch.object(module, "train_main", return_value="stylemod-train") as train_main:
            self.assertEqual(module.main(["train", "--config", "stylemod.yaml"]), "stylemod-train")
            train_main.assert_called_once_with(["--config", "stylemod.yaml"])

        with patch.object(module, "generate_result_main", return_value="stylemod-gen") as generate_main:
            self.assertEqual(module.main(["generate-result", "--config", "stylemod.yaml"]), "stylemod-gen")
            generate_main.assert_called_once_with(["--config", "stylemod.yaml"])

    def test_shared_generate_result_router_supports_stylemod_model_flag(self):
        module = _load_script_module(ROOT_DIR / "scripts/generate_result/main.py", "generate_result_stylemod_switch_script")
        calls = []

        def _record(argv):
            calls.append(list(argv))

        with patch.dict(module.MODEL_COMMANDS, {"stylemod-wgan": _record}, clear=False):
            module.main(["--model", "stylemod-wgan", "--config", "configs/stylemod_wgan/train_default.yaml"])

        self.assertEqual(calls, [["--config", "configs/stylemod_wgan/train_default.yaml"]])

    def test_modulated_conv_changes_with_style_and_demod_stays_finite(self):
        torch.manual_seed(0)
        layer = ModulatedConv2d(3, 5, 3, 7, stride=1, demodulate=True)
        x = torch.randn(2, 3, 8, 8)
        style_a = torch.randn(2, 7)
        style_b = style_a + 3.0
        out_a = layer(x, style_a)
        out_b = layer(x, style_b)
        self.assertEqual(out_a.shape, (2, 5, 8, 8))
        self.assertTrue(torch.isfinite(out_a).all())
        self.assertTrue(torch.isfinite(out_b).all())
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_generator_noise_is_reproducible_and_optional(self):
        torch.manual_seed(0)
        generator = StyleModWGANGenerator(
            surface_height=8,
            surface_width=8,
            embedding_dim=6,
            noise_dim=4,
            base_channels=4,
            res_blocks=1,
            text_hidden_dim=8,
            text_out_dim=6,
            fusion_hidden_dim=16,
            style_dim=6,
            style_noise_scale=0.5,
            style_demodulate=True,
        )
        current_surface = torch.rand(2, 1, 8, 8)
        text_embedding = torch.rand(2, 6)
        noise = torch.randn(2, 4)
        out1 = generator(current_surface, text_embedding, noise=noise)
        out2 = generator(current_surface, text_embedding, noise=noise)
        out3 = generator(current_surface, text_embedding, noise=noise + 1.0)
        self.assertTrue(torch.allclose(out1, out2))
        self.assertFalse(torch.allclose(out1, out3))

        deterministic_generator = StyleModWGANGenerator(
            surface_height=8,
            surface_width=8,
            embedding_dim=6,
            noise_dim=4,
            base_channels=4,
            res_blocks=1,
            text_hidden_dim=8,
            text_out_dim=6,
            fusion_hidden_dim=16,
            style_dim=6,
            style_noise_scale=0.0,
            style_demodulate=True,
        )
        fixed_a = deterministic_generator(current_surface, text_embedding, noise=torch.randn(2, 4))
        fixed_b = deterministic_generator(current_surface, text_embedding, noise=torch.randn(2, 4))
        self.assertTrue(torch.allclose(fixed_a, fixed_b, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
