from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for value in (str(ROOT), str(SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.rq3 import news_first_vol_wgan_grid08_analysis as analysis  # noqa: E402
from scripts.rq3 import news_first_vol_wgan_grid08_sweep as sweep  # noqa: E402
from wgan_option.models.common import (  # noqa: E402
    INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
    LEGACY_CRITIC_NORMALIZATION_MODE,
    critic_normalization_fingerprint,
)
from wgan_option.models.discriminator import Discriminator  # noqa: E402
from wgan_option.models.generator import Generator  # noqa: E402
from wgan_option.utils.inference_helpers import (  # noqa: E402
    resolve_checkpoint_surface_grid_contract,
)


CONFIG = ROOT / "configs/rq3/news_first_vol_wgan_grid08_capacity_lr.yaml"


class Grid08ModelContractTests(unittest.TestCase):
    def test_exact_axes_profiles_and_parameter_counts(self) -> None:
        resolved = sweep.resolve_config(CONFIG)
        grid = sweep._grid_contract(resolved)
        self.assertEqual(grid["surface_shape"], [8, 8])
        self.assertEqual(grid["maturity_grid_days"], [7, 11, 16, 20, 25, 29, 34, 38])
        np.testing.assert_allclose(
            grid["strike_grid"],
            np.linspace(0.97, 1.03, 8),
            rtol=0,
            atol=1e-15,
        )
        contracts = sweep._instantiate_profile_contracts(resolved, grid)
        self.assertEqual(
            [contracts[name]["expected_wgan_parameters"] for name in sweep.PROFILES],
            [17690, 39451, 95381, 256873, 484541, 2620165],
        )

    def test_group_tail_critic_accepts_batch_one_and_two(self) -> None:
        model = Discriminator(
            channels=1,
            embedding_dim=1024,
            surface_height=8,
            surface_width=8,
            base_channels=4,
            text_hidden_dim=16,
            hidden_dim=96,
            critic_normalization_mode=INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
        )
        for training in (True, False):
            model.train(training)
            for batch in (1, 2):
                result = model(
                    torch.rand(batch, 1, 8, 8),
                    torch.rand(batch, 1, 8, 8),
                    torch.rand(batch, 1024),
                )
                self.assertEqual(tuple(result.shape), (batch, 1))
                self.assertTrue(torch.isfinite(result).all())

    def test_legacy_critic_contract_remains_16x16(self) -> None:
        model = Discriminator(
            channels=1,
            embedding_dim=1024,
            surface_height=16,
            surface_width=16,
            critic_normalization_mode=LEGACY_CRITIC_NORMALIZATION_MODE,
        )
        result = model(
            torch.rand(2, 1, 16, 16),
            torch.rand(2, 1, 16, 16),
            torch.rand(2, 1024),
        )
        self.assertEqual(tuple(result.shape), (2, 1))

    def test_identity_epoch_zero_still_returns_current(self) -> None:
        generator = Generator(
            channels=1,
            embedding_dim=1024,
            noise_dim=32,
            surface_height=8,
            surface_width=8,
            base_channels=4,
            text_hidden_dim=32,
            text_out_dim=16,
            hidden_dim=128,
            residual_output_mode="identity_softplus_residual",
            generator_noise_mode="gaussian",
            generator_current_input_mode="current_support_masked",
        ).eval()
        current = torch.rand(2, 1, 8, 8) * 0.1 + 0.04
        with torch.no_grad():
            output = generator(
                current,
                torch.rand(2, 1024),
                noise=torch.zeros(2, 32),
                current_support_mask=torch.ones_like(current),
            )
        self.assertTrue(torch.equal(output, current))

    def test_checkpoint_grid_and_normalization_fail_closed(self) -> None:
        resolved = sweep.resolve_config(CONFIG)
        grid = sweep._grid_contract(resolved)
        raw_config = {
            "news_first_surface_grid_profile": sweep.GRID_PROFILE,
            "news_first_surface_grid_sha256": grid["grid_sha256"],
            "news_first_architecture_profile_sha256": "a" * 64,
            "news_first_model_contract_sha256": "b" * 64,
            "critic_normalization_mode": INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
        }
        checkpoint = {
            "config": raw_config,
            "surface_grid_profile": sweep.GRID_PROFILE,
            "surface_grid_sha256": grid["grid_sha256"],
            "architecture_profile_sha256": "a" * 64,
            "model_contract_sha256": "b" * 64,
            "critic_normalization_mode": INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
            "critic_normalization_fingerprint": critic_normalization_fingerprint(
                INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE
            ),
            "surface_shape": [8, 8],
            "strike_grid": grid["strike_grid"],
            "maturity_days_grid": grid["maturity_grid_days"],
        }
        sample = SimpleNamespace(
            current_surface=np.zeros((1, 8, 8)),
            strike_grid=np.asarray(grid["strike_grid"]),
            maturity_grid_days=np.asarray(grid["maturity_grid_days"]),
        )
        contract = resolve_checkpoint_surface_grid_contract(checkpoint, sample)
        self.assertEqual(contract["surface_grid_profile"], sweep.GRID_PROFILE)
        wrong = SimpleNamespace(
            current_surface=np.zeros((1, 8, 8)),
            strike_grid=np.linspace(0.90, 1.10, 8),
            maturity_grid_days=np.asarray(grid["maturity_grid_days"]),
        )
        with self.assertRaises(ValueError):
            resolve_checkpoint_surface_grid_contract(checkpoint, wrong)
        changed = json.loads(json.dumps(checkpoint))
        changed["critic_normalization_mode"] = LEGACY_CRITIC_NORMALIZATION_MODE
        with self.assertRaises(ValueError):
            resolve_checkpoint_surface_grid_contract(changed, sample)


class Grid08OrchestrationTests(unittest.TestCase):
    def test_primary_matrix_and_gpu_balance(self) -> None:
        specs = sweep.primary_specs()
        self.assertEqual(len(specs), 180)
        assigned = sweep._balanced_assignments(specs, gpu_ids=(0, 1), slots_per_gpu=24)
        self.assertEqual(max(row["wave"] for row in assigned), 4)
        frame = pd.DataFrame(assigned)
        for factor in ("capacity_profile", "lr_profile", "seed", "tolerance_minutes"):
            table = frame.groupby([factor, "gpu_id"]).size().unstack(fill_value=0)
            self.assertTrue((table[0] == table[1]).all(), factor)
        keys = [
            "capacity_profile",
            "lr_profile",
            "seed",
            "text_ablation_mode",
            "tolerance_minutes",
        ]
        self.assertEqual(len(frame[keys].drop_duplicates()), 180)

    def test_twelve_slot_schedule_has_eight_waves(self) -> None:
        assigned = sweep._balanced_assignments(
            sweep.primary_specs(), gpu_ids=(0, 1), slots_per_gpu=12
        )
        self.assertEqual(max(row["wave"] for row in assigned), 8)

    def test_one_se_prefers_smallest_parameters_then_lower_lr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "analysis").mkdir()
            (root / "analysis" / "grid08_capacity_lr_scores.csv").write_text("x\n1\n")
            scores = pd.DataFrame(
                [
                    {
                        "capacity_profile": "large",
                        "lr_profile": "lr_5e_07",
                        "score": -0.01,
                        "score_bootstrap_se": 0.002,
                        "expected_wgan_parameters": 484541,
                        "initial_learning_rate": 5e-7,
                        "ci_95_upper": 0.1,
                        "holm_p": 1.0,
                        "non_worse_seed_count": 3,
                    },
                    {
                        "capacity_profile": "micro",
                        "lr_profile": "lr_5e_07",
                        "score": -0.009,
                        "score_bootstrap_se": 0.003,
                        "expected_wgan_parameters": 17690,
                        "initial_learning_rate": 5e-7,
                        "ci_95_upper": 0.1,
                        "holm_p": 1.0,
                        "non_worse_seed_count": 3,
                    },
                    {
                        "capacity_profile": "micro",
                        "lr_profile": "lr_2_5e_07",
                        "score": -0.009,
                        "score_bootstrap_se": 0.003,
                        "expected_wgan_parameters": 17690,
                        "initial_learning_rate": 2.5e-7,
                        "ci_95_upper": 0.1,
                        "holm_p": 1.0,
                        "non_worse_seed_count": 3,
                    },
                ]
            )
            selected = analysis._selection_from_scores(scores, root)
            self.assertEqual(selected["point_leader_capacity_profile"], "large")
            self.assertEqual(selected["one_se_candidate_capacity_profile"], "micro")
            self.assertEqual(selected["one_se_candidate_lr_profile"], "lr_2_5e_07")
            self.assertEqual(selected["selection_label"], "descriptive_only")

    def test_frozen_selection_distinguishes_payload_and_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            payload = {"schema_version": 1, "candidate": "micro"}
            payload["selection_sha256"] = sweep._payload_sha256(payload)
            sweep._write_json(path, payload)
            frozen, file_sha = sweep._validate_frozen_selection(path, payload)
            self.assertEqual(frozen, payload)
            self.assertEqual(file_sha, sweep._sha256_file(path))
            self.assertNotEqual(file_sha, payload["selection_sha256"])
            tampered = dict(payload)
            tampered["candidate"] = "large"
            sweep._write_json(path, tampered)
            with self.assertRaisesRegex(ValueError, "self-hash"):
                sweep._validate_frozen_selection(path, payload)

    def test_two_level_bootstrap_preserves_seed_session_clusters(self) -> None:
        rows = []
        for seed in sweep.SEEDS:
            for session in range(3):
                for pair in range(2):
                    rows.append(
                        {
                            "seed": seed,
                            "session_id": f"s{session}",
                            "model_mae": 0.9,
                            "persistence_mae": 1.0,
                        }
                    )
        frame = pd.DataFrame(rows)
        observed, draws = analysis._two_level_draws(
            frame,
            columns=("model_mae", "persistence_mae"),
            statistic=lambda values: values[:, 0] - values[:, 1],
            seed=123,
            iterations=500,
        )
        self.assertAlmostEqual(observed, -0.1)
        np.testing.assert_allclose(draws, -0.1, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
