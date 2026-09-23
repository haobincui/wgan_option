from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.rq3 import news_first_vol_film_unet_capacity_epoch_3seed as sweep


class FilmUnetCapacityEpoch3SeedSweepTests(unittest.TestCase):
    """Freeze the branch-local 60-epoch Q3 capacity protocol."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.resolved = sweep.resolve_config(sweep.DEFAULT_CONFIG)
        cls.specs = sweep.experiment_specs(cls.resolved)

    def test_config_parses_the_four_selected_arms_and_six_profiles(self) -> None:
        self.assertEqual(self.resolved["experiment_kind"], sweep.EXPERIMENT_KIND)
        self.assertEqual(tuple(self.resolved["seeds"]), sweep.SEEDS)
        self.assertEqual(tuple(self.resolved["capacity_profiles"]), sweep.CAPACITY_IDS)
        self.assertEqual(
            tuple(self.resolved["arms"]),
            (
                "film_unet_mask_coords_text128",
                "film_unet_mask_coords_text64",
                "film_unet_mask_coords_text64_nolp",
                "film_unet_mask_coords_text64_projection",
            ),
        )
        self.assertEqual(self.resolved["learning_rate"], 5e-7)
        self.assertEqual(self.resolved["max_epochs"], 60)
        self.assertEqual(self.resolved["early_stopping_min_epochs"], 30)
        self.assertEqual(self.resolved["early_stopping_patience"], 20)
        self.assertEqual(self.resolved["runtime"]["workers_per_gpu"], 18)
        self.assertEqual(self.resolved["runtime"]["benchmark_workers_per_gpu"], 18)

    def test_matrix_is_exact_unique_and_balanced_by_capacity(self) -> None:
        self.assertEqual(len(self.specs), 72)
        self.assertEqual(len({spec["job_id"] for spec in self.specs}), 72)
        self.assertEqual(
            {gpu: sum(spec["gpu_id"] == gpu for spec in self.specs) for gpu in (0, 1)},
            {0: 36, 1: 36},
        )
        for capacity_id in sweep.CAPACITY_IDS:
            self.assertEqual(
                {
                    gpu: sum(
                        spec["capacity_id"] == capacity_id and spec["gpu_id"] == gpu
                        for spec in self.specs
                    )
                    for gpu in sweep.GPU_IDS
                },
                {0: 6, 1: 6},
            )
        self.assertEqual(
            {(spec["arm_id"], spec["capacity_id"]) for spec in self.specs},
            {
                (arm_id, capacity_id)
                for arm_id in sweep.ARM_IDS
                for capacity_id in sweep.CAPACITY_IDS
            },
        )

    def test_real_executable_parameter_counts_are_frozen(self) -> None:
        expected = {
            "c04": (320_973, 87_885, 151_413),
            "c08": (357_945, 117_689, 186_793),
            "c12": (406_725, 159_301, 237_341),
            "c16": (467_313, 212_721, 303_057),
            "c24": (623_913, 354_985, 479_993),
            "c32": (827_745, 544_481, 729_157),
        }
        by_key = {(spec["arm_id"], spec["capacity_id"]): spec for spec in self.specs}
        for capacity_id, (generator_128, generator_64, critic) in expected.items():
            self.assertEqual(
                (
                    by_key[("film_unet_mask_coords_text128", capacity_id)][
                        "generator_parameters"
                    ],
                    by_key[("film_unet_mask_coords_text128", capacity_id)][
                        "critic_parameters"
                    ],
                ),
                (generator_128, critic),
            )
            for arm_id in (
                "film_unet_mask_coords_text64",
                "film_unet_mask_coords_text64_nolp",
                "film_unet_mask_coords_text64_projection",
            ):
                spec = by_key[(arm_id, capacity_id)]
                self.assertEqual(spec["generator_parameters"], generator_64)
                self.assertEqual(spec["critic_parameters"], critic)
                self.assertEqual(spec["wgan_parameters"], generator_64 + critic)

    def test_fair_initialization_contract_is_capacity_and_seed_local(self) -> None:
        by_key = {
            (spec["arm_id"], spec["capacity_id"], spec["seed"]): spec
            for spec in self.specs
        }
        compact_arms = (
            "film_unet_mask_coords_text64",
            "film_unet_mask_coords_text64_nolp",
            "film_unet_mask_coords_text64_projection",
        )
        for capacity_id in sweep.CAPACITY_IDS:
            for seed in sweep.SEEDS:
                self.assertEqual(
                    len(
                        {
                            by_key[(arm_id, capacity_id, seed)][
                                "initial_generator_state_sha256"
                            ]
                            for arm_id in compact_arms
                        }
                    ),
                    1,
                )
                self.assertEqual(
                    len(
                        {
                            by_key[(arm_id, capacity_id, seed)][
                                "initial_critic_state_sha256"
                            ]
                            for arm_id in (
                                "film_unet_mask_coords_text64",
                                "film_unet_mask_coords_text64_nolp",
                            )
                        }
                    ),
                    1,
                )
                self.assertNotEqual(
                    by_key[
                        (
                            "film_unet_mask_coords_text64",
                            capacity_id,
                            seed,
                        )
                    ]["initial_critic_state_sha256"],
                    by_key[
                        (
                            "film_unet_mask_coords_text64_projection",
                            capacity_id,
                            seed,
                        )
                    ]["initial_critic_state_sha256"],
                )

    def test_training_payload_freezes_epoch_lr_and_q3_only_contract(self) -> None:
        spec = next(
            spec
            for spec in self.specs
            if spec["arm_id"] == "film_unet_mask_coords_text64_projection"
            and spec["capacity_id"] == "c24"
            and spec["seed"] == 404
        )
        payload = sweep._training_payload(self.resolved, Path("/tmp/unet-sweep"), spec)
        self.assertEqual(
            payload["generator_conditioning_mode"], "film_unet_mask_coords_v1"
        )
        self.assertEqual(payload["critic_conditioning_mode"], "lp_projection_v1")
        self.assertEqual(payload["gen_base_channels"], 24)
        self.assertEqual(payload["disc_base_channels"], 24)
        self.assertEqual(payload["gen_text_hidden_dim"], 64)
        self.assertEqual(payload["gen_text_out_dim"], 64)
        self.assertEqual(payload["generator_learning_rate"], 5e-7)
        self.assertEqual(payload["discriminator_learning_rate"], 5e-7)
        self.assertEqual(payload["num_epochs"], 60)
        self.assertEqual(payload["early_stopping_min_epochs"], 30)
        self.assertEqual(payload["early_stopping_patience"], 20)
        self.assertEqual(payload["lr_warmup_epochs"], 0)
        self.assertEqual(payload["validation_mc_samples"], 16)
        self.assertTrue(payload["news_first_materialize_validation_loader"])
        self.assertFalse(payload["news_first_materialize_test_loader"])
        self.assertEqual(payload["news_first_capacity_profile"], "c24")

    def test_benchmark_registry_is_separate_and_exactly_one_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "benchmark"
            registry = sweep._prepare_benchmark(self.resolved, root, resume=False)
            self.assertEqual(
                registry["experiment_kind"], sweep.BENCHMARK_EXPERIMENT_KIND
            )
            self.assertEqual(registry["expected_job_count"], 72)
            self.assertEqual({job["num_epochs"] for job in registry["jobs"]}, {1})
            self.assertEqual(
                {
                    gpu: sum(job["gpu_id"] == gpu for job in registry["jobs"])
                    for gpu in sweep.GPU_IDS
                },
                {0: 36, 1: 36},
            )
            payload = sweep._require_mapping(
                sweep.yaml.safe_load(
                    Path(registry["jobs"][0]["config_path"]).read_text(encoding="utf-8")
                ),
                "benchmark config",
            )
            self.assertEqual(payload["num_epochs"], 1)
            self.assertFalse(payload["use_early_stopping"])
            self.assertFalse(payload["news_first_materialize_test_loader"])
            self.assertEqual(
                payload["news_first_lr_profile"],
                "film_unet_capacity_epoch1_benchmark",
            )

    def test_resource_peaks_and_exclusive_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control"
            control.mkdir()
            snapshots = [
                {
                    "rows": [
                        "0, NVIDIA A30, 1024, 23552, 1",
                        "1, NVIDIA A30, 2048, 22528, 1",
                    ],
                    "MemTotal_kib": 1_000,
                    "MemAvailable_kib": 250,
                    "returncode": 0,
                },
                {
                    "rows": [
                        "0, NVIDIA A30, 3072, 21504, 2",
                        "1, NVIDIA A30, 1024, 23552, 2",
                    ],
                    "MemTotal_kib": 1_000,
                    "MemAvailable_kib": 200,
                    "returncode": 0,
                },
            ]
            path = control / "resource_snapshots.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in snapshots),
                encoding="utf-8",
            )
            peaks = sweep._resource_peaks(root)
            self.assertEqual(
                peaks["peak_memory_mib_by_gpu"], {"0": 3072.0, "1": 2048.0}
            )
            self.assertEqual(peaks["telemetry_rows"], 4)
            self.assertAlmostEqual(peaks["peak_host_ram_fraction"], 0.8)
            self.assertTrue(
                sweep._benchmark_resource_gate(peaks, self.resolved["runtime"])[
                    "passed"
                ]
            )

            at_limit = dict(peaks)
            at_limit["peak_memory_gib_by_gpu"] = {"0": 20.0, "1": 1.0}
            self.assertFalse(
                sweep._benchmark_resource_gate(at_limit, self.resolved["runtime"])[
                    "passed"
                ]
            )


if __name__ == "__main__":
    unittest.main()
