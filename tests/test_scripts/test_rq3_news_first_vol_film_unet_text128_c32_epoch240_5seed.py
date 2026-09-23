from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from scripts.rq3 import (
    news_first_vol_film_unet_text128_c32_epoch240_5seed as sweep,
)


class FilmUnetText128C32Epoch240FiveSeedTests(unittest.TestCase):
    """Freeze the independent five-seed Text128 + LP critic extension."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.resolved = sweep.resolve_config(sweep.DEFAULT_CONFIG)
        cls.specs = sweep.experiment_specs(cls.resolved)

    def test_protocol_is_one_text128_lp_arm_c32_and_five_frozen_seeds(self) -> None:
        self.assertEqual(self.resolved["experiment_kind"], sweep.EXPERIMENT_KIND)
        self.assertEqual(
            tuple(self.resolved["seeds"]),
            (42, 202, 404, 382624741, 1607127774),
        )
        self.assertEqual(tuple(self.resolved["capacity_profiles"]), ("c32",))
        self.assertEqual(
            tuple(self.resolved["arms"]),
            ("film_unet_mask_coords_text128",),
        )
        arm = self.resolved["arms"]["film_unet_mask_coords_text128"]
        self.assertEqual(arm["generator_conditioning_mode"], "film_unet_mask_coords_v1")
        self.assertEqual(arm["critic_conditioning_mode"], "lp_concat_v1")
        self.assertEqual(arm["gen_text_hidden_dim"], 256)
        self.assertEqual(arm["gen_text_out_dim"], 128)

    def test_training_budget_lr_and_stopping_contract_are_frozen(self) -> None:
        self.assertEqual(self.resolved["learning_rate"], 5e-7)
        self.assertEqual(self.resolved["max_epochs"], 240)
        self.assertEqual(self.resolved["lr_warmup_epochs"], 0)
        self.assertEqual(self.resolved["early_stopping_min_epochs"], 30)
        self.assertEqual(self.resolved["early_stopping_patience"], 20)
        self.assertEqual(self.resolved["validation_mc_samples"], 16)

    def test_five_unique_fresh_jobs_alternate_gpus_three_to_two(self) -> None:
        expected_seeds = (42, 202, 404, 382624741, 1607127774)
        self.assertEqual(len(self.specs), 5)
        self.assertEqual(len({spec["job_id"] for spec in self.specs}), 5)
        self.assertEqual(tuple(spec["seed"] for spec in self.specs), expected_seeds)
        self.assertEqual(
            tuple(spec["gpu_id"] for spec in self.specs),
            (0, 1, 0, 1, 0),
        )
        self.assertEqual(
            {gpu: sum(spec["gpu_id"] == gpu for spec in self.specs) for gpu in (0, 1)},
            {0: 3, 1: 2},
        )
        self.assertEqual(
            {spec["arm_id"] for spec in self.specs},
            {"film_unet_mask_coords_text128"},
        )
        self.assertEqual({spec["capacity_id"] for spec in self.specs}, {"c32"})
        self.assertEqual(
            {spec["training_state_inherited"] for spec in self.specs}, {False}
        )
        self.assertEqual(
            len({spec["initial_generator_state_sha256"] for spec in self.specs}),
            5,
        )
        self.assertEqual(
            len({spec["initial_critic_state_sha256"] for spec in self.specs}),
            5,
        )

    def test_real_c32_parameter_counts_are_frozen(self) -> None:
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

    def test_training_payload_is_fresh_q3_only_and_has_no_warmup(self) -> None:
        spec = next(spec for spec in self.specs if spec["seed"] == 404)
        root = Path("/tmp/film-unet-text128-epoch240")
        payload = sweep._training_payload(self.resolved, root, spec)
        self.assertEqual(
            payload["generator_conditioning_mode"], "film_unet_mask_coords_v1"
        )
        self.assertEqual(payload["critic_conditioning_mode"], "lp_concat_v1")
        self.assertEqual(payload["gen_base_channels"], 32)
        self.assertEqual(payload["disc_base_channels"], 32)
        self.assertEqual(payload["gen_text_hidden_dim"], 256)
        self.assertEqual(payload["gen_text_out_dim"], 128)
        self.assertEqual(payload["generator_learning_rate"], 5e-7)
        self.assertEqual(payload["discriminator_learning_rate"], 5e-7)
        self.assertEqual(payload["num_epochs"], 240)
        self.assertEqual(payload["lr_warmup_epochs"], 0)
        self.assertEqual(payload["early_stopping_min_epochs"], 30)
        self.assertEqual(payload["early_stopping_patience"], 20)
        self.assertTrue(payload["news_first_materialize_validation_loader"])
        self.assertFalse(payload["news_first_materialize_test_loader"])
        self.assertEqual(payload["news_first_train_end_utc"], "2023-07-01T00:00:00Z")
        self.assertEqual(
            payload["news_first_validation_end_utc"], "2023-10-01T00:00:00Z"
        )
        self.assertEqual(payload["news_first_capacity_profile"], "c32")
        self.assertIn("epoch240", payload["news_first_lr_profile"])
        for forbidden in (
            "resume_checkpoint",
            "generator_checkpoint",
            "discriminator_checkpoint",
            "parent_checkpoint",
        ):
            self.assertNotIn(forbidden, payload)

    def test_prepare_writes_five_isolated_configs_and_output_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            registry = sweep.prepare(self.resolved, root, resume=False)
            self.assertEqual(registry["experiment_kind"], sweep.EXPERIMENT_KIND)
            self.assertEqual(registry["expected_job_count"], 5)
            self.assertEqual(len(registry["jobs"]), 5)
            self.assertEqual(len({job["config_path"] for job in registry["jobs"]}), 5)
            self.assertEqual(len({job["run_root"] for job in registry["jobs"]}), 5)

            for job in registry["jobs"]:
                config_path = Path(job["config_path"])
                self.assertTrue(config_path.is_relative_to(root.resolve()))
                payload = sweep._require_mapping(
                    yaml.safe_load(config_path.read_text(encoding="utf-8")),
                    "generated training config",
                )
                expected_run_root = (
                    root
                    / "runs"
                    / "film_unet_mask_coords_text128"
                    / "c32"
                    / f"seed_{int(job['seed']):03d}"
                ).resolve()
                self.assertEqual(Path(job["run_root"]), expected_run_root)
                self.assertEqual(Path(payload["output_root"]), expected_run_root)
                self.assertEqual(payload["seed"], job["seed"])
                self.assertEqual(payload["num_epochs"], 240)
                self.assertFalse(payload["news_first_materialize_test_loader"])

            resumed = sweep.prepare(self.resolved, root, resume=True)
            self.assertEqual(
                resumed["jobs_payload_sha256"], registry["jobs_payload_sha256"]
            )

    def test_benchmark_registry_has_five_one_epoch_q3_only_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "benchmark"
            registry = sweep._prepare_benchmark(self.resolved, root, resume=False)
            self.assertEqual(
                registry["experiment_kind"], sweep.BENCHMARK_EXPERIMENT_KIND
            )
            self.assertEqual(registry["expected_job_count"], 5)
            self.assertEqual({job["num_epochs"] for job in registry["jobs"]}, {1})
            self.assertEqual(
                tuple(job["gpu_id"] for job in registry["jobs"]),
                (0, 1, 0, 1, 0),
            )
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
                self.assertIn("epoch1_benchmark", payload["news_first_lr_profile"])

            sweep._prepare_benchmark(self.resolved, root, resume=True)

    def test_default_config_and_output_locations_are_branch_local(self) -> None:
        self.assertEqual(
            sweep.DEFAULT_CONFIG,
            "configs/rq3/news_first_vol_film_unet_text128_c32_epoch240_5seed.yaml",
        )
        self.assertEqual(
            sweep.DEFAULT_OUTPUT_DIR,
            "outputs/experiments/"
            "rq3_news_first_vol_film_unet_text128_c32_epoch240_5seed_exact_ttm_v1",
        )
        self.assertEqual(
            sweep.DEFAULT_BENCHMARK_DIR,
            "outputs/benchmarks/"
            "rq3_news_first_vol_film_unet_text128_c32_epoch240_5seed_epoch1_v1",
        )
        self.assertNotEqual(sweep.DEFAULT_OUTPUT_DIR, sweep.DEFAULT_BENCHMARK_DIR)


if __name__ == "__main__":
    unittest.main()
