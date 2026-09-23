"""Contracts for current-only raw-support Generator conditioning."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

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

from film_wgan.support import SUPPORT_METHOD  # noqa: E402
from scripts.rq3.news_first_vol_comparison_analysis import (  # noqa: E402
    RunSpec,
    TrainedRunEvaluator,
)
from wgan_option.config import Config  # noqa: E402
from wgan_option.models.common import (  # noqa: E402
    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    FULL_CURRENT_GENERATOR_INPUT_MODE,
    GENERATOR_CURRENT_SUPPORT_METHOD,
    IDENTITY_RESIDUAL_OUTPUT_MODE,
    generator_current_input_contract,
    generator_current_input_fingerprint,
    residual_output_fingerprint,
)
from wgan_option.models.gan_model import WGAN_GP  # noqa: E402
from wgan_option.models.generator import Generator  # noqa: E402
from wgan_option.utils.inference_helpers import (  # noqa: E402
    resolve_checkpoint_generator_current_input_contract,
)


def _config(
    *,
    current_input_mode: str = CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    root: Path | None = None,
) -> Config:
    output = root or Path("/tmp/current-input-contract")
    return Config(
        cuda=False,
        text_embedding_mode="lp",
        support_mask_mode=(
            "raw_joint"
            if current_input_mode == CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
            else "none"
        ),
        generator_current_input_mode=current_input_mode,
        channels=1,
        embedding_dim=2,
        noise_dim=3,
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
        residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
        models_path=str(output / "checkpoints"),
        metrics_path=str(output / "metrics"),
    )


def _generator(config: Config, *, size: int = 2) -> Generator:
    return Generator(
        channels=1,
        embedding_dim=2,
        noise_dim=3,
        surface_height=size,
        surface_width=size,
        base_channels=2,
        res_blocks=0,
        text_hidden_dim=4,
        text_out_dim=3,
        hidden_dim=8,
        residual_output_mode=config.residual_output_mode,
        generator_current_input_mode=config.generator_current_input_mode,
    )


def _raw_params(lower: float, upper: float) -> str:
    return json.dumps(
        {
            "business_days": [10, 20],
            "percent_strikes": [[lower, upper], [lower, upper]],
            "implied_vols": [[0.20, 0.21], [0.22, 0.23]],
        }
    )


def _panel(target_params: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "news_1",
                "news_row_id": 1,
                "pair_id": "pair_1",
                "session_id": "session_1",
                "effective_origin_utc": "2023-08-01T14:00:00Z",
                "current_surface_flat": "[0.2, 0.3, 0.4, 0.5]",
                "target_surface_flat": "[0.21, 0.31, 0.41, 0.51]",
                "surface_shape": "[2, 2]",
                "strike_grid": "[0.9, 1.1]",
                "maturity_days_grid": "[10, 20]",
                "hd_embedding": "[0.1, 0.2]",
                "lp_embedding": "[0.3, 0.4]",
                # Only q=0.9 is current-supported.
                "current_surface_param_json": _raw_params(0.85, 1.0),
                "target_surface_param_json": target_params,
            }
        ]
    )


class TestGeneratorCurrentInputContract(unittest.TestCase):
    def test_config_defaults_to_full_current_and_masked_requires_raw_joint(self):
        self.assertEqual(
            Config(cuda=False).generator_current_input_mode,
            FULL_CURRENT_GENERATOR_INPUT_MODE,
        )
        masked = _config()
        self.assertEqual(
            masked.generator_current_input_mode,
            CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
        )
        with self.assertRaisesRegex(
            ValueError, "requires support_mask_mode='raw_joint'"
        ):
            Config(
                cuda=False,
                generator_current_input_mode=CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
            )
        with self.assertRaisesRegex(ValueError, "generator_current_input_mode"):
            Config(cuda=False, generator_current_input_mode="joint_masked")

    def test_contract_pins_support_algorithm_without_circular_import(self):
        contract = generator_current_input_contract(
            CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
        )
        self.assertEqual(GENERATOR_CURRENT_SUPPORT_METHOD, SUPPORT_METHOD)
        self.assertEqual(contract["support_method"], SUPPORT_METHOD)
        self.assertEqual(contract["unsupported_cell_fill_value"], 0.0)
        self.assertEqual(
            contract["residual_anchor"], "original_unmasked_current_surface"
        )
        self.assertNotEqual(
            generator_current_input_fingerprint(
                CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
            ),
            generator_current_input_fingerprint(FULL_CURRENT_GENERATOR_INPUT_MODE),
        )

        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(SRC_DIR)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from wgan_option.config import Config; "
                "print(Config(cuda=False).generator_current_input_mode)",
            ],
            cwd=ROOT_DIR,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), FULL_CURRENT_GENERATOR_INPUT_MODE)

    def test_masked_encoder_zeros_unsupported_cells_but_anchor_stays_original(self):
        model = _generator(_config()).eval()
        current = torch.tensor([[[[0.2, 9.0], [0.3, 8.0]]]])
        mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        text = torch.tensor([[0.1, 0.2]])
        observed: list[torch.Tensor] = []
        handle = model.surface_encoder.register_forward_pre_hook(
            lambda _module, inputs: observed.append(inputs[0].detach().clone())
        )
        try:
            output = model(
                current,
                text,
                noise=torch.zeros(1, 3),
                current_support_mask=mask,
            )
        finally:
            handle.remove()

        torch.testing.assert_close(observed[0], current * mask)
        # The identity residual head starts at persistence over the full grid,
        # including cells excluded from the encoder.
        self.assertTrue(torch.equal(output, current))
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            sum(
                parameter.numel()
                for parameter in _generator(
                    _config(current_input_mode=FULL_CURRENT_GENERATOR_INPUT_MODE)
                ).parameters()
            ),
        )

        with self.assertRaisesRegex(ValueError, "explicit current_support_mask"):
            model(current, text, noise=torch.zeros(1, 3))
        with self.assertRaisesRegex(ValueError, "binary"):
            model(
                current,
                text,
                noise=torch.zeros(1, 3),
                current_support_mask=torch.full_like(mask, 0.5),
            )

    def test_checkpoint_contract_is_backward_compatible_and_masked_fails_closed(self):
        full_mode, full_fingerprint = (
            resolve_checkpoint_generator_current_input_contract({"config": {}})
        )
        self.assertEqual(full_mode, FULL_CURRENT_GENERATOR_INPUT_MODE)
        self.assertEqual(
            full_fingerprint,
            generator_current_input_fingerprint(FULL_CURRENT_GENERATOR_INPUT_MODE),
        )

        config = asdict(_config())
        fingerprint = generator_current_input_fingerprint(
            CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
        )
        checkpoint = {
            "config": config,
            "generator_current_input_mode": CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
            "generator_current_input_fingerprint": fingerprint,
        }
        self.assertEqual(
            resolve_checkpoint_generator_current_input_contract(checkpoint),
            (CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE, fingerprint),
        )
        missing_fingerprint = dict(checkpoint)
        missing_fingerprint.pop("generator_current_input_fingerprint")
        with self.assertRaisesRegex(ValueError, "missing"):
            resolve_checkpoint_generator_current_input_contract(missing_fingerprint)

        metadata_only = dict(checkpoint)
        metadata_only["config"] = dict(config)
        metadata_only["config"].pop("generator_current_input_mode")
        with self.assertRaisesRegex(ValueError, "must explicitly save"):
            resolve_checkpoint_generator_current_input_contract(metadata_only)

        wrong_support = dict(checkpoint)
        wrong_support["config"] = dict(config, support_mask_mode="none")
        with self.assertRaisesRegex(ValueError, "support_mask_mode='raw_joint'"):
            resolve_checkpoint_generator_current_input_contract(wrong_support)

        corrupted = dict(
            checkpoint,
            generator_current_input_fingerprint="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            resolve_checkpoint_generator_current_input_contract(corrupted)

    def test_wgan_checkpoint_saves_current_input_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _config(root=Path(tmpdir))
            model = WGAN_GP(
                config,
                strike_grid=np.asarray([0.9, 1.1]),
                maturity_grid_days=np.asarray([10, 20]),
                embedding_dim=2,
            )
            checkpoint_path = Path(model.save_model(label="best")["generator"])
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                payload["generator_current_input_mode"],
                CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
            )
            self.assertEqual(
                payload["generator_current_input_fingerprint"],
                generator_current_input_fingerprint(
                    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
                ),
            )

    def test_wgan_paths_require_the_separate_current_mask(self):
        model = WGAN_GP(
            _config(),
            strike_grid=np.linspace(0.9, 1.1, 16),
            maturity_grid_days=np.linspace(10, 20, 16),
            embedding_dim=2,
        )
        current = torch.linspace(0.2, 0.35, 256).reshape(1, 1, 16, 16)
        target = current + 0.01
        text = torch.tensor([[0.1, 0.2]])
        weight = torch.ones(1)
        key = torch.tensor([101], dtype=torch.int64)
        joint_mask = torch.zeros_like(current)
        joint_mask[:, :, 0, 0] = 1.0
        current_mask = torch.zeros_like(current)
        current_mask[:, :, :, :2] = 1.0

        model._discriminator_step(
            current,
            text,
            target,
            weight,
            joint_mask,
            current_mask,
        )
        model._generator_step(
            current,
            text,
            target,
            sample_weight=weight,
            support_mask=joint_mask,
            current_support_mask=current_mask,
        )
        loader = DataLoader(
            TensorDataset(
                current,
                text,
                target,
                weight,
                key,
                joint_mask,
                current_mask,
            ),
            batch_size=1,
            shuffle=False,
        )
        self.assertIn("val_recon", model._evaluate(loader))

        with self.assertRaisesRegex(ValueError, "explicit current_support_mask"):
            model._generator_step(
                current,
                text,
                target,
                sample_weight=weight,
                support_mask=joint_mask,
            )

    def test_evaluator_conditioning_mask_does_not_depend_on_target_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root=root)
            generator = _generator(config)
            checkpoint = root / "generator.pt"
            torch.save(
                {
                    "state_dict": generator.state_dict(),
                    "config": asdict(config),
                    "embedding_dim": 2,
                    "residual_output_mode": config.residual_output_mode,
                    "residual_output_fingerprint": residual_output_fingerprint(
                        config.residual_output_mode
                    ),
                    "generator_current_input_mode": (
                        CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
                    ),
                    "generator_current_input_fingerprint": (
                        generator_current_input_fingerprint(
                            CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
                        )
                    ),
                },
                checkpoint,
            )
            run = RunSpec(
                "masked",
                root,
                "wgan",
                5,
                42,
                checkpoint,
                support_mask_mode="raw_joint",
                generator_current_input_mode=(
                    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
                ),
            )
            evaluator = TrainedRunEvaluator(
                mc_samples=3,
                sample_batch_size=1,
                draw_batch_size=2,
                device="cpu",
            )
            first = evaluator(run, "q3", _panel(_raw_params(0.85, 1.2))).iloc[0]
            second = evaluator(run, "q3", _panel(_raw_params(1.05, 1.2))).iloc[0]

            np.testing.assert_allclose(
                first["predicted_surface_flat"],
                second["predicted_surface_flat"],
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(first["current_support_cell_count"], 2)
            self.assertEqual(
                first["current_support_mask_fingerprint"],
                second["current_support_mask_fingerprint"],
            )
            self.assertEqual(
                first["generator_current_input_mode"],
                CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
            )


if __name__ == "__main__":
    unittest.main()
