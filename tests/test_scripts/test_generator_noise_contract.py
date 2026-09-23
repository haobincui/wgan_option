"""Contracts for the same-width deterministic WGAN latent ablation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.rq3.news_first_vol_comparison_analysis import (  # noqa: E402
    RunSpec,
    TrainedRunEvaluator,
)
from wgan_option.config import Config  # noqa: E402
from wgan_option.models.common import (  # noqa: E402
    GAUSSIAN_GENERATOR_NOISE_MODE,
    ZERO_GENERATOR_NOISE_MODE,
    generator_noise_contract,
    generator_noise_fingerprint,
    generator_noise_tensor,
)
from wgan_option.models.gan_model import WGAN_GP  # noqa: E402
from wgan_option.models.generator import Generator  # noqa: E402
from wgan_option.utils.inference_helpers import load_vol_generator  # noqa: E402


def _config(*, mode: str, root: Path | None = None) -> Config:
    output = root or Path("/tmp/noise-contract")
    return Config(
        cuda=False,
        text_embedding_mode="lp",
        generator_noise_mode=mode,
        channels=1,
        embedding_dim=2,
        noise_dim=3,
        validation_mc_samples=7,
        gen_base_channels=2,
        gen_res_blocks=0,
        gen_text_hidden_dim=4,
        gen_text_out_dim=3,
        gen_hidden_dim=8,
        disc_base_channels=2,
        disc_res_blocks=0,
        disc_text_hidden_dim=3,
        disc_hidden_dim=8,
        discriminator_iter=1,
        use_calendar_constraint=False,
        use_butterfly_constraint=False,
        use_smooth_constraint=False,
        models_path=str(output / "checkpoints"),
        metrics_path=str(output / "metrics"),
    )


def _generator(config: Config, *, size: int = 2) -> Generator:
    return Generator(
        channels=1,
        embedding_dim=2,
        noise_dim=config.noise_dim,
        surface_height=size,
        surface_width=size,
        base_channels=config.gen_base_channels,
        res_blocks=config.gen_res_blocks,
        text_hidden_dim=config.gen_text_hidden_dim,
        text_out_dim=config.gen_text_out_dim,
        hidden_dim=config.gen_hidden_dim,
        residual_output_mode=config.residual_output_mode,
        generator_noise_mode=config.generator_noise_mode,
    )


class TestGeneratorNoiseContract(unittest.TestCase):
    def test_config_default_and_contract_are_backward_compatible(self):
        self.assertEqual(
            Config(cuda=False).generator_noise_mode,
            GAUSSIAN_GENERATOR_NOISE_MODE,
        )
        self.assertEqual(
            Config(cuda=False, generator_noise_mode=" ZERO ").generator_noise_mode,
            "zero",
        )
        with self.assertRaisesRegex(ValueError, "generator_noise_mode"):
            Config(cuda=False, generator_noise_mode="uniform")

        contract = generator_noise_contract(ZERO_GENERATOR_NOISE_MODE, 32)
        self.assertEqual(contract["noise_dim"], 32)
        self.assertEqual(
            contract["training_rng_policy"], "burn_standard_normal_then_zero"
        )
        self.assertEqual(contract["evaluation_rng_policy"], "literal_zero_single_pass")
        self.assertNotEqual(
            generator_noise_fingerprint(ZERO_GENERATOR_NOISE_MODE, 32),
            generator_noise_fingerprint(GAUSSIAN_GENERATOR_NOISE_MODE, 32),
        )

    def test_zero_training_input_burns_same_rng_as_gaussian(self):
        reference = torch.empty(2, 1, 2, 2)
        torch.manual_seed(1234)
        gaussian = generator_noise_tensor(
            reference,
            batch_size=2,
            noise_dim=3,
            mode=GAUSSIAN_GENERATOR_NOISE_MODE,
            preserve_gaussian_rng_progression=True,
        )
        gaussian_tail = torch.randn(5)

        torch.manual_seed(1234)
        zero = generator_noise_tensor(
            reference,
            batch_size=2,
            noise_dim=3,
            mode=ZERO_GENERATOR_NOISE_MODE,
            preserve_gaussian_rng_progression=True,
        )
        zero_tail = torch.randn(5)

        self.assertGreater(int(torch.count_nonzero(gaussian)), 0)
        self.assertEqual(int(torch.count_nonzero(zero)), 0)
        torch.testing.assert_close(gaussian_tail, zero_tail)

    def test_zero_mode_preserves_architecture_and_seeded_initialization(self):
        gaussian_config = _config(mode=GAUSSIAN_GENERATOR_NOISE_MODE)
        zero_config = _config(mode=ZERO_GENERATOR_NOISE_MODE)
        gaussian_config.seed = zero_config.seed = 202
        kwargs = {
            "strike_grid": np.linspace(0.9, 1.1, 16, dtype=np.float32),
            "maturity_grid_days": np.linspace(7.0, 38.0, 16, dtype=np.float32),
            "embedding_dim": 2,
        }
        gaussian = WGAN_GP(config=gaussian_config, **kwargs)
        zero = WGAN_GP(config=zero_config, **kwargs)
        self.assertEqual(
            sum(parameter.numel() for parameter in gaussian.G.parameters()),
            sum(parameter.numel() for parameter in zero.G.parameters()),
        )
        for name, value in gaussian.G.state_dict().items():
            torch.testing.assert_close(value, zero.G.state_dict()[name])

    def test_zero_generator_is_authoritative_for_explicit_noise(self):
        config = _config(mode=ZERO_GENERATOR_NOISE_MODE)
        model = _generator(config).eval()
        with torch.no_grad():
            model.fusion[-1].weight.fill_(0.02)
        current = torch.full((2, 1, 2, 2), 0.2)
        text = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        first = model(current, text, noise=torch.randn(2, 3))
        second = model(current, text, noise=torch.randn(2, 3) * 100.0)
        torch.testing.assert_close(first, second)

    def test_all_wgan_train_and_validation_paths_receive_zero_noise(self):
        config = _config(mode=ZERO_GENERATOR_NOISE_MODE)
        model = WGAN_GP(
            config=config,
            strike_grid=np.linspace(0.9, 1.1, 16, dtype=np.float32),
            maturity_grid_days=np.linspace(7.0, 38.0, 16, dtype=np.float32),
            embedding_dim=2,
        )
        current = torch.full((1, 1, 16, 16), 0.2)
        text = torch.tensor([[0.1, 0.2]])
        target = current + 0.01
        weight = torch.ones(1)
        key = torch.tensor([101], dtype=torch.int64)
        loader = DataLoader(
            TensorDataset(current, text, target, weight, key),
            batch_size=1,
            shuffle=False,
        )
        observed: list[torch.Tensor] = []
        original_forward = model.G.forward

        def recording_forward(current_surface, text_embedding, noise=None):
            self.assertIsNotNone(noise)
            observed.append(noise.detach().clone())
            return original_forward(current_surface, text_embedding, noise=noise)

        with patch.object(model.G, "forward", side_effect=recording_forward):
            model._discriminator_step(current, text, target, weight)
            model._generator_step(current, text, target, sample_weight=weight)
            model._evaluate(loader)

        # One critic call, one generator call and one validation call.  The
        # configured seven validation MC draws collapse to a single pass.
        self.assertEqual(len(observed), 3)
        self.assertTrue(all(int(torch.count_nonzero(item)) == 0 for item in observed))

    def test_zero_checkpoint_requires_fingerprint_and_best_json_records_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(mode=ZERO_GENERATOR_NOISE_MODE, root=root)
            model = WGAN_GP(
                config=config,
                strike_grid=np.linspace(0.9, 1.1, 16, dtype=np.float32),
                maturity_grid_days=np.linspace(7.0, 38.0, 16, dtype=np.float32),
                embedding_dim=2,
            )
            model._init_metrics_file()
            metrics = {
                "val_recon": 0.1,
                "val_current_recon": 0.1,
                "val_baseline_gap": 0.0,
                "val_hybrid_score": 0.1,
            }
            model._save_best_learned_validation_checkpoint(
                epoch=1,
                monitor_metric="val_hybrid_score",
                current_metric=0.1,
                epoch_stats=metrics,
            )
            expected = generator_noise_fingerprint(ZERO_GENERATOR_NOISE_MODE, 3)
            metadata = json.loads(
                (root / "metrics" / "best_learned_checkpoint.json").read_text()
            )
            self.assertEqual(metadata["generator_noise_mode"], "zero")
            self.assertEqual(metadata["generator_noise_fingerprint"], expected)
            checkpoint_path = root / "checkpoints" / "generator_best_learned.pt"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["generator_noise_mode"], "zero")
            self.assertEqual(checkpoint["generator_noise_fingerprint"], expected)

            sample = SimpleNamespace(
                current_surface=np.zeros((1, 16, 16), dtype=np.float32)
            )
            loaded, loaded_config, _ = load_vol_generator(
                checkpoint_path, sample, torch.device("cpu")
            )
            self.assertEqual(loaded.generator_noise_mode, "zero")
            self.assertEqual(loaded_config.generator_noise_mode, "zero")

            missing = dict(checkpoint)
            missing.pop("generator_noise_fingerprint")
            missing_path = root / "missing.pt"
            torch.save(missing, missing_path)
            with self.assertRaisesRegex(
                ValueError, "missing generator_noise_fingerprint"
            ):
                load_vol_generator(missing_path, sample, torch.device("cpu"))

            corrupted = dict(checkpoint, generator_noise_fingerprint="0" * 64)
            corrupted_path = root / "corrupted.pt"
            torch.save(corrupted, corrupted_path)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_vol_generator(corrupted_path, sample, torch.device("cpu"))

    def test_historical_checkpoint_without_noise_metadata_remains_gaussian(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _config(mode=GAUSSIAN_GENERATOR_NOISE_MODE)
            generator = _generator(config, size=2)
            historical_config = asdict(config)
            historical_config.pop("generator_noise_mode")
            path = Path(tmpdir) / "historical.pt"
            torch.save(
                {
                    "state_dict": generator.state_dict(),
                    "config": historical_config,
                    "embedding_dim": 2,
                },
                path,
            )
            sample = SimpleNamespace(
                current_surface=np.zeros((1, 2, 2), dtype=np.float32)
            )
            loaded, loaded_config, _ = load_vol_generator(
                path, sample, torch.device("cpu")
            )
            self.assertEqual(loaded.generator_noise_mode, "gaussian")
            self.assertEqual(loaded_config.generator_noise_mode, "gaussian")

    def test_production_evaluator_uses_one_zero_noise_pass_and_exports_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(mode=ZERO_GENERATOR_NOISE_MODE)
            generator = _generator(config, size=2)
            fingerprint = generator_noise_fingerprint(ZERO_GENERATOR_NOISE_MODE, 3)
            checkpoint = root / "generator.pt"
            torch.save(
                {
                    "state_dict": generator.state_dict(),
                    "config": asdict(config),
                    "embedding_dim": 2,
                    "generator_noise_mode": ZERO_GENERATOR_NOISE_MODE,
                    "generator_noise_fingerprint": fingerprint,
                },
                checkpoint,
            )
            panel = pd.DataFrame(
                [
                    {
                        "sample_id": "news_1",
                        "news_row_id": 1,
                        "pair_id": "pair_1",
                        "session_id": "session_1",
                        "effective_origin_utc": "2023-08-01T14:00:00Z",
                        "current_surface_flat": "[0.2, 0.2, 0.2, 0.2]",
                        "target_surface_flat": "[0.21, 0.21, 0.21, 0.21]",
                        "surface_shape": "[2, 2]",
                        "hd_embedding": "[0.1, 0.2]",
                        "lp_embedding": "[0.3, 0.4]",
                    }
                ]
            )
            run = RunSpec("zero", root, "wgan", 5, 42, checkpoint)
            evaluator = TrainedRunEvaluator(mc_samples=64, device="cpu")
            prediction = evaluator(run, "q3", panel)
            self.assertEqual(int(prediction.loc[0, "prediction_mc_samples"]), 1)
            self.assertEqual(prediction.loc[0, "generator_noise_mode"], "zero")
            self.assertEqual(
                prediction.loc[0, "generator_noise_fingerprint"], fingerprint
            )
            self.assertFalse(bool(prediction.loc[0, "prediction_fallback"]))


if __name__ == "__main__":
    unittest.main()
