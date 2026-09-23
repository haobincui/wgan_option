from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from scripts.rq3 import (
    news_first_vol_film_unet_text128_c32_epoch240_nolp_5seed as sweep,
)


class FilmUnetText128C32Epoch240NoLPFiveSeedTests(unittest.TestCase):
    """Freeze the exact NoLP-Critic counterpart to the completed LP run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.resolved = sweep.resolve_config(sweep.DEFAULT_CONFIG)
        cls.specs = sweep.experiment_specs(cls.resolved)

    def test_protocol_changes_only_the_critic_conditioning_mode(self) -> None:
        self.assertEqual(self.resolved["experiment_kind"], sweep.EXPERIMENT_KIND)
        self.assertEqual(tuple(self.resolved["seeds"]), sweep.SEEDS)
        self.assertEqual(tuple(self.resolved["capacity_profiles"]), ("c32",))
        self.assertEqual(tuple(self.resolved["arms"]), (sweep.ARM_ID,))
        arm = self.resolved["arms"][sweep.ARM_ID]
        self.assertEqual(arm["generator_conditioning_mode"], "film_unet_mask_coords_v1")
        self.assertEqual(arm["critic_conditioning_mode"], "lp_disabled_same_shape_v1")
        self.assertEqual(arm["gen_text_hidden_dim"], 256)
        self.assertEqual(arm["gen_text_out_dim"], 128)

    def test_training_budget_and_q3_only_contract_are_frozen(self) -> None:
        self.assertEqual(self.resolved["learning_rate"], 5e-7)
        self.assertEqual(self.resolved["max_epochs"], 240)
        self.assertEqual(self.resolved["lr_warmup_epochs"], 0)
        self.assertEqual(self.resolved["early_stopping_min_epochs"], 30)
        self.assertEqual(self.resolved["early_stopping_patience"], 20)
        self.assertEqual(self.resolved["validation_mc_samples"], 16)

    def test_five_fresh_jobs_have_frozen_gpu_mapping_and_parameter_counts(self) -> None:
        self.assertEqual(len(self.specs), 5)
        self.assertEqual(len({spec["job_id"] for spec in self.specs}), 5)
        self.assertEqual(tuple(spec["seed"] for spec in self.specs), sweep.SEEDS)
        self.assertEqual(tuple(spec["gpu_id"] for spec in self.specs), (0, 1, 0, 1, 0))
        self.assertEqual({spec["arm_id"] for spec in self.specs}, {sweep.ARM_ID})
        self.assertEqual(
            {spec["critic_conditioning_mode"] for spec in self.specs},
            {"lp_disabled_same_shape_v1"},
        )
        self.assertEqual(
            {
                (
                    spec["generator_parameters"],
                    spec["critic_parameters"],
                    spec["wgan_parameters"],
                )
                for spec in self.specs
            },
            {(827_745, 729_157, 1_556_902)},
        )

    def test_initial_states_exactly_match_completed_lp_reference_by_seed(self) -> None:
        comparison = self.resolved["comparison"]
        registry = json.loads(
            Path(comparison["lp_registry"]).read_text(encoding="utf-8")
        )
        lp_by_seed = {int(job["seed"]): job for job in registry["jobs"]}
        for spec in self.specs:
            seed = int(spec["seed"])
            lp_job = lp_by_seed[seed]
            self.assertEqual(
                spec["initial_generator_state_sha256"],
                lp_job["initial_generator_state_sha256"],
            )
            self.assertEqual(
                spec["initial_critic_state_sha256"],
                lp_job["initial_critic_state_sha256"],
            )
            self.assertNotEqual(
                spec["critic_conditioning_fingerprint"],
                lp_job["critic_conditioning_fingerprint"],
            )
            self.assertNotEqual(
                spec["architecture_profile_sha256"],
                lp_job["architecture_profile_sha256"],
            )
            self.assertNotEqual(
                spec["model_contract_sha256"], lp_job["model_contract_sha256"]
            )

    def test_training_payload_keeps_text128_c32_and_disables_critic_lp(self) -> None:
        spec = next(spec for spec in self.specs if int(spec["seed"]) == 404)
        payload = sweep._training_payload(
            self.resolved, Path("/tmp/film-unet-text128-nolp-epoch240"), spec
        )
        self.assertEqual(
            payload["generator_conditioning_mode"], "film_unet_mask_coords_v1"
        )
        self.assertEqual(
            payload["critic_conditioning_mode"], "lp_disabled_same_shape_v1"
        )
        self.assertEqual(payload["gen_base_channels"], 32)
        self.assertEqual(payload["disc_base_channels"], 32)
        self.assertEqual(payload["gen_text_hidden_dim"], 256)
        self.assertEqual(payload["gen_text_out_dim"], 128)
        self.assertEqual(payload["num_epochs"], 240)
        self.assertEqual(payload["lr_warmup_epochs"], 0)
        self.assertTrue(payload["news_first_materialize_validation_loader"])
        self.assertFalse(payload["news_first_materialize_test_loader"])
        self.assertIn("nolp", payload["news_first_lr_profile"])
        for forbidden in (
            "resume_checkpoint",
            "generator_checkpoint",
            "discriminator_checkpoint",
            "parent_checkpoint",
        ):
            self.assertNotIn(forbidden, payload)

    def test_prepare_writes_five_isolated_nolp_configs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            registry = sweep.prepare(self.resolved, root, resume=False)
            self.assertEqual(registry["experiment_kind"], sweep.EXPERIMENT_KIND)
            self.assertEqual(registry["expected_job_count"], 5)
            self.assertEqual(len(registry["jobs"]), 5)
            for job in registry["jobs"]:
                payload = sweep._require_mapping(
                    yaml.safe_load(
                        Path(job["config_path"]).read_text(encoding="utf-8")
                    ),
                    "generated training config",
                )
                expected = (
                    root
                    / "runs"
                    / sweep.ARM_ID
                    / "c32"
                    / f"seed_{int(job['seed']):03d}"
                ).resolve()
                self.assertEqual(Path(job["run_root"]), expected)
                self.assertEqual(Path(payload["output_root"]), expected)
                self.assertEqual(
                    payload["critic_conditioning_mode"],
                    "lp_disabled_same_shape_v1",
                )
                self.assertFalse(payload["news_first_materialize_test_loader"])
            resumed = sweep.prepare(self.resolved, root, resume=True)
            self.assertEqual(
                resumed["jobs_payload_sha256"], registry["jobs_payload_sha256"]
            )

    def test_benchmark_has_five_one_epoch_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "benchmark"
            registry = sweep._prepare_benchmark(self.resolved, root, resume=False)
            self.assertEqual(
                registry["experiment_kind"], sweep.BENCHMARK_EXPERIMENT_KIND
            )
            self.assertEqual(len(registry["jobs"]), 5)
            self.assertEqual({job["num_epochs"] for job in registry["jobs"]}, {1})
            for job in registry["jobs"]:
                payload = sweep._require_mapping(
                    yaml.safe_load(
                        Path(job["config_path"]).read_text(encoding="utf-8")
                    ),
                    "benchmark config",
                )
                self.assertEqual(payload["num_epochs"], 1)
                self.assertFalse(payload["use_early_stopping"])
                self.assertFalse(payload["news_first_materialize_test_loader"])
                self.assertIn("nolp", payload["news_first_lr_profile"])
                self.assertIn("epoch1_benchmark", payload["news_first_lr_profile"])
            sweep._prepare_benchmark(self.resolved, root, resume=True)

    def test_default_locations_are_independent_from_the_lp_root(self) -> None:
        self.assertIn("nolp", sweep.DEFAULT_CONFIG)
        self.assertIn("nolp", sweep.DEFAULT_OUTPUT_DIR)
        self.assertIn("nolp", sweep.DEFAULT_BENCHMARK_DIR)
        self.assertNotEqual(sweep.DEFAULT_OUTPUT_DIR, sweep.lp.DEFAULT_OUTPUT_DIR)
        self.assertNotEqual(sweep.DEFAULT_OUTPUT_DIR, sweep.DEFAULT_BENCHMARK_DIR)


if __name__ == "__main__":
    unittest.main()
