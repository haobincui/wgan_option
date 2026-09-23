from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_text128_c32_epoch240_5seed_q4 as q4,
)


class FilmUnetText128C32Epoch240FiveSeedQ4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolved = q4.resolve_config(q4.DEFAULT_CONFIG)

    def test_q4_contract_is_frozen_and_retrospective(self) -> None:
        self.assertEqual(self.resolved["experiment_kind"], q4.EXPERIMENT_KIND)
        self.assertEqual(
            self.resolved["confirmation_label"],
            "retrospective_frozen_exploratory",
        )
        self.assertEqual(tuple(self.resolved["seeds"]), q4.SEEDS)
        self.assertEqual(self.resolved["checkpoint_role"], "best_learned")
        self.assertEqual(self.resolved["q4_mc_samples"], 64)
        self.assertEqual(self.resolved["bootstrap_iterations"], 10_000)
        panel = self.resolved["q4_panel"]
        self.assertEqual(panel["expected_counts"], q4.EXPECTED_COUNTS)
        self.assertEqual(panel["support_mask_mode"], "raw_joint")
        self.assertEqual(panel["surface_grid_profile"], "exact_ttm_16x16_v1")

    def test_prepare_freezes_five_checkpoints_before_reading_q4(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "q4"
            with mock.patch.object(
                q4.pd,
                "read_excel",
                side_effect=AssertionError("prepare must read zero Q4 rows"),
            ):
                registry = q4.prepare(self.resolved, root, resume=False)
            self.assertEqual(registry["state"], "q4_locked")
            self.assertEqual(registry["checkpoint_count"], 5)
            self.assertEqual(registry["q4_rows_read"], 0)
            self.assertFalse(registry["q4_gate_open"])
            self.assertFalse(registry["q4_window_materialized"])
            self.assertFalse(registry["q4_predictions_generated"])
            self.assertFalse(registry["q4_evaluated"])
            allowlist = q4._read_json(q4._allowlist_path(root))
            q4._verify_self_hash(
                allowlist, "allowlist_payload_sha256", "checkpoint allowlist"
            )
            rows = allowlist["rows"]
            self.assertEqual(tuple(row["seed"] for row in rows), q4.SEEDS)
            self.assertEqual(
                len({row["checkpoint_metadata_sha256"] for row in rows}), 5
            )
            self.assertEqual(
                len({row["generator_checkpoint_sha256"] for row in rows}), 5
            )
            self.assertEqual({row["capacity_id"] for row in rows}, {"c32"})
            self.assertEqual(
                {row["generator_conditioning_mode"] for row in rows},
                {"film_unet_mask_coords_v1"},
            )
            self.assertEqual(
                {row["critic_conditioning_mode"] for row in rows},
                {"lp_concat_v1"},
            )
            resumed = q4.prepare(self.resolved, root, resume=True)
            self.assertEqual(
                resumed["allowlist_rows_sha256"], registry["allowlist_rows_sha256"]
            )

    def test_q4_materializer_refuses_to_read_before_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(
                q4.pd,
                "read_excel",
                side_effect=AssertionError("Q4 source must remain unread"),
            ):
                with self.assertRaisesRegex(RuntimeError, "before the explicit gate"):
                    q4._materialize_q4(
                        self.resolved,
                        root,
                        {"q4_gate_open": False},
                    )

    def test_production_run_spec_loads_frozen_generator_not_metadata(self) -> None:
        row = {
            "job_id": "seed_042",
            "run_dir": "/tmp/run",
            "seed": 42,
            "checkpoint_metadata_path": "/tmp/best_learned_checkpoint.json",
            "generator_checkpoint_path": "/tmp/generator_best_learned.pt",
            "generator_conditioning_mode": "film_unet_mask_coords_v1",
            "critic_conditioning_mode": "lp_concat_v1",
        }
        spec = q4._run_spec_from_allowlist_row(row)
        self.assertEqual(spec.checkpoint_path, Path(row["generator_checkpoint_path"]))
        self.assertNotEqual(spec.checkpoint_path, Path(row["checkpoint_metadata_path"]))

    def test_two_level_bootstrap_is_deterministic_and_negative(self) -> None:
        rows = []
        for seed in q4.SEEDS:
            for pair_index in range(143):
                rows.append(
                    {
                        "seed": seed,
                        "session_id": f"session_{pair_index % 45:02d}",
                        "pair_id": f"pair_{pair_index:03d}",
                        "model_mae": 0.9 + seed % 7 * 1e-4,
                        "persistence_mae": 1.0,
                    }
                )
        frame = pd.DataFrame(rows)
        first = q4.seed_session_bootstrap(frame, iterations=200, random_seed=7)
        second = q4.seed_session_bootstrap(frame, iterations=200, random_seed=7)
        self.assertEqual(first, second)
        self.assertLess(first["mean_difference_model_minus_persistence"], 0.0)
        self.assertLess(first["ci_95_upper"], 0.0)
        self.assertEqual(first["seed_count"], 5)
        self.assertEqual(first["sessions_per_seed"], 45)
        self.assertEqual(first["pairs_per_seed"], 143)
        self.assertEqual(
            first["resampling_method"], "seed_then_paired_CME_session_cluster"
        )

    def test_output_root_is_independent_from_training(self) -> None:
        self.assertNotEqual(
            q4._resolve_repo_path(q4.DEFAULT_OUTPUT_DIR),
            Path(self.resolved["training_root"]),
        )
        self.assertEqual(q4.MC_SAMPLES, 64)
        self.assertEqual(q4.EVALUATOR_PANEL_NAMESPACE, "core")


if __name__ == "__main__":
    unittest.main()
