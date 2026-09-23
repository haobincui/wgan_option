from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import yaml

from scripts.rq3 import news_first_vol_film_unet_c32_epoch120_seed42 as sweep
from scripts.rq3 import news_first_vol_film_unet_capacity_epoch_3seed as predecessor


class FilmUnetC32Epoch120Seed42Tests(unittest.TestCase):
    """Freeze the independent single-factor epoch-budget extension."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.resolved = sweep.resolve_config(sweep.DEFAULT_CONFIG)
        cls.specs = sweep.experiment_specs(cls.resolved)

    def test_protocol_is_c32_seed42_max120_with_predecessor_stopping_rule(self) -> None:
        self.assertEqual(self.resolved["experiment_kind"], sweep.EXPERIMENT_KIND)
        self.assertEqual(tuple(self.resolved["seeds"]), (42,))
        self.assertEqual(tuple(self.resolved["capacity_profiles"]), ("c32",))
        self.assertEqual(self.resolved["learning_rate"], 5e-7)
        self.assertEqual(self.resolved["max_epochs"], 120)
        self.assertEqual(self.resolved["lr_warmup_epochs"], 0)
        self.assertEqual(self.resolved["early_stopping_min_epochs"], 30)
        self.assertEqual(self.resolved["early_stopping_patience"], 20)
        self.assertEqual(self.resolved["validation_mc_samples"], 16)
        self.assertEqual(self.resolved["runtime"]["workers_per_gpu"], 2)
        self.assertEqual(self.resolved["runtime"]["benchmark_workers_per_gpu"], 2)

    def test_four_unique_jobs_have_frozen_two_by_two_gpu_mapping(self) -> None:
        self.assertEqual(len(self.specs), 4)
        self.assertEqual(len({spec["job_id"] for spec in self.specs}), 4)
        self.assertEqual({spec["seed"] for spec in self.specs}, {42})
        self.assertEqual({spec["capacity_id"] for spec in self.specs}, {"c32"})
        self.assertEqual(
            {spec["training_state_inherited"] for spec in self.specs}, {False}
        )
        self.assertEqual(
            {spec["predecessor_job_summary_sha256"] for spec in self.specs},
            {self.resolved["comparison"]["epoch60_job_summary_sha256"]},
        )
        self.assertEqual(
            {gpu: sum(spec["gpu_id"] == gpu for spec in self.specs) for gpu in (0, 1)},
            {0: 2, 1: 2},
        )
        by_arm = {spec["arm_id"]: spec for spec in self.specs}
        self.assertEqual(by_arm["film_unet_mask_coords_text64"]["gpu_id"], 0)
        self.assertEqual(by_arm["film_unet_mask_coords_text64_nolp"]["gpu_id"], 1)
        self.assertEqual(by_arm["film_unet_mask_coords_text128"]["gpu_id"], 1)
        self.assertEqual(by_arm["film_unet_mask_coords_text64_projection"]["gpu_id"], 0)

    def test_c32_parameter_counts_and_initialization_fairness_are_real(self) -> None:
        by_arm = {spec["arm_id"]: spec for spec in self.specs}
        self.assertEqual(
            (
                by_arm["film_unet_mask_coords_text128"]["generator_parameters"],
                by_arm["film_unet_mask_coords_text128"]["critic_parameters"],
            ),
            (827_745, 729_157),
        )
        for arm_id in (
            "film_unet_mask_coords_text64",
            "film_unet_mask_coords_text64_nolp",
            "film_unet_mask_coords_text64_projection",
        ):
            self.assertEqual(by_arm[arm_id]["generator_parameters"], 544_481)
            self.assertEqual(by_arm[arm_id]["critic_parameters"], 729_157)
        compact_generator_hashes = {
            by_arm[arm_id]["initial_generator_state_sha256"]
            for arm_id in (
                "film_unet_mask_coords_text64",
                "film_unet_mask_coords_text64_nolp",
                "film_unet_mask_coords_text64_projection",
            )
        }
        self.assertEqual(len(compact_generator_hashes), 1)
        self.assertEqual(
            by_arm["film_unet_mask_coords_text64"]["initial_critic_state_sha256"],
            by_arm["film_unet_mask_coords_text64_nolp"]["initial_critic_state_sha256"],
        )

    def test_training_payload_is_fresh_q3_only_and_changes_only_epoch_budget(
        self,
    ) -> None:
        spec = next(
            spec
            for spec in self.specs
            if spec["arm_id"] == "film_unet_mask_coords_text64_projection"
        )
        payload = sweep._training_payload(self.resolved, Path("/tmp/epoch120"), spec)
        self.assertEqual(
            payload["generator_conditioning_mode"], "film_unet_mask_coords_v1"
        )
        self.assertEqual(payload["critic_conditioning_mode"], "lp_projection_v1")
        self.assertEqual(payload["gen_base_channels"], 32)
        self.assertEqual(payload["disc_base_channels"], 32)
        self.assertEqual(payload["generator_learning_rate"], 5e-7)
        self.assertEqual(payload["discriminator_learning_rate"], 5e-7)
        self.assertEqual(payload["num_epochs"], 120)
        self.assertEqual(payload["early_stopping_min_epochs"], 30)
        self.assertEqual(payload["early_stopping_patience"], 20)
        self.assertEqual(payload["lr_warmup_epochs"], 0)
        self.assertTrue(payload["news_first_materialize_validation_loader"])
        self.assertFalse(payload["news_first_materialize_test_loader"])
        self.assertEqual(payload["news_first_capacity_profile"], "c32")
        self.assertEqual(
            payload["news_first_lr_profile"],
            "film_unet_c32_epoch120_lr_5e_07_no_warmup",
        )
        for forbidden in (
            "resume_checkpoint",
            "generator_checkpoint",
            "discriminator_checkpoint",
            "parent_checkpoint",
        ):
            self.assertNotIn(forbidden, payload)

    def test_reuse_does_not_permanently_mutate_predecessor_module(self) -> None:
        frozen = {
            "experiment_kind": predecessor.EXPERIMENT_KIND,
            "seeds": predecessor.SEEDS,
            "capacity_ids": predecessor.CAPACITY_IDS,
            "expected_jobs": predecessor.EXPECTED_JOB_COUNT,
            "training_payload": predecessor._training_payload,
            "experiment_specs": predecessor.experiment_specs,
        }
        sweep.experiment_specs(self.resolved)
        self.assertEqual(predecessor.EXPERIMENT_KIND, frozen["experiment_kind"])
        self.assertEqual(predecessor.SEEDS, frozen["seeds"])
        self.assertEqual(predecessor.CAPACITY_IDS, frozen["capacity_ids"])
        self.assertEqual(predecessor.EXPECTED_JOB_COUNT, frozen["expected_jobs"])
        self.assertIs(predecessor._training_payload, frozen["training_payload"])
        self.assertIs(predecessor.experiment_specs, frozen["experiment_specs"])

    def test_source_contract_includes_wrapper_core_config_and_predecessor_metrics(
        self,
    ) -> None:
        with sweep._configured_core():
            hashes = sweep._source_hashes_impl(self.resolved)
        names = set(hashes)
        self.assertTrue(
            any(
                "news_first_vol_film_unet_c32_epoch120_seed42.py" in name
                for name in names
            )
        )
        self.assertTrue(
            any(
                "news_first_vol_film_unet_capacity_epoch_3seed.py" in name
                for name in names
            )
        )
        self.assertTrue(
            any(
                "news_first_vol_film_unet_c32_epoch120_seed42.yaml" in name
                for name in names
            )
        )
        predecessor_key = next(
            name for name in names if name.startswith("input_sha256::")
        )
        self.assertEqual(
            hashes[predecessor_key],
            self.resolved["comparison"]["epoch60_job_summary_sha256"],
        )

    def test_benchmark_registry_has_four_one_epoch_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "benchmark"
            registry = sweep._prepare_benchmark(self.resolved, root, resume=False)
            self.assertEqual(
                registry["experiment_kind"], sweep.BENCHMARK_EXPERIMENT_KIND
            )
            self.assertEqual(registry["expected_job_count"], 4)
            self.assertEqual({job["num_epochs"] for job in registry["jobs"]}, {1})
            self.assertEqual(
                {
                    gpu: sum(job["gpu_id"] == gpu for job in registry["jobs"])
                    for gpu in (0, 1)
                },
                {0: 2, 1: 2},
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
                self.assertEqual(
                    payload["news_first_lr_profile"],
                    "film_unet_c32_epoch120_seed42_epoch1_benchmark",
                )
            sweep._prepare_benchmark(self.resolved, root, resume=True)

    def test_prepare_resume_rejects_generated_config_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            registry = sweep.prepare(self.resolved, root, resume=False)
            self.assertEqual(len(registry["jobs"]), 4)
            config_path = Path(registry["jobs"][0]["config_path"])
            config_path.write_text(
                config_path.read_text(encoding="utf-8") + "\n# tampered\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Generated job config drift"):
                sweep.prepare(self.resolved, root, resume=True)

    def test_prepare_resume_rejects_registry_and_status_binding_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            registry = sweep.prepare(self.resolved, root, resume=False)
            registry_path = sweep._registry_path(root)
            tampered = sweep._read_json(registry_path)
            tampered["jobs"][0]["gpu_id"] = 1 - int(tampered["jobs"][0]["gpu_id"])
            sweep._write_json(registry_path, tampered)
            with self.assertRaisesRegex(ValueError, "Registry job_spec SHA drift"):
                sweep.prepare(self.resolved, root, resume=True)

            sweep._write_json(registry_path, registry)
            job = registry["jobs"][0]
            status_path = sweep._job_status_path(root, job["job_id"])
            status_payload = sweep._read_json(status_path)
            status_payload["job_spec_sha256"] = "0" * 64
            sweep._write_json(status_path, status_payload)
            with self.assertRaisesRegex(ValueError, "Status job_spec_sha256"):
                sweep.prepare(self.resolved, root, resume=True)

    def test_single_seed_postprocess_is_descriptive_and_compares_epoch60(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            registry = sweep.prepare(self.resolved, root, resume=False)
            old = sweep._epoch60_rows(self.resolved)
            for index, job in enumerate(registry["jobs"], start=1):
                run_dir = Path(job["run_root"]) / "fixture_run"
                run_dir.mkdir(parents=True)
                artifacts = []
                for role in sorted(sweep.REQUIRED_ARTIFACT_ROLES):
                    artifact_path = run_dir / f"{role}.txt"
                    artifact_path.write_text(
                        f"artifact-{index}-{role}\n", encoding="utf-8"
                    )
                    artifacts.append(sweep._artifact(artifact_path, role))
                old_mae = float(old[job["arm_id"]]["best_q3_mae"])
                sweep._write_status(
                    root,
                    job["job_id"],
                    {
                        "job_id": job["job_id"],
                        "job_spec_sha256": job["job_spec_sha256"],
                        "state": "complete",
                        "best_epoch": 80 + index,
                        "completed_epochs": 120,
                        "best_mae": old_mae - index * 1e-8,
                        "persistence_mae": float(
                            old[job["arm_id"]]["persistence_q3_mae"]
                        ),
                        "run_dir": str(run_dir),
                        "artifacts": artifacts,
                    },
                )
            result = sweep.postprocess(root)
            self.assertEqual(result["completed_jobs"], 4)
            self.assertEqual(
                result["inference_scope"], "single-seed descriptive epoch extension"
            )
            self.assertTrue(result["fresh_initialization"])
            self.assertFalse(result["training_state_inherited"])
            rows_path = root / "analysis/architecture_ranking.csv"
            with rows_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertTrue(
                all(float(row["mae_delta_120_minus_60"]) < 0 for row in rows)
            )
            self.assertTrue(
                all(row["training_state_inherited"] == "False" for row in rows)
            )
            self.assertEqual(
                {int(row["descriptive_rank"]) for row in rows}, {1, 2, 3, 4}
            )

    def test_resource_gate_remains_strictly_exclusive(self) -> None:
        peaks = {
            "peak_memory_gib_by_gpu": {"0": 19.9, "1": 19.9},
            "peak_host_ram_fraction": 0.84,
            "telemetry_rows": 2,
        }
        self.assertTrue(
            sweep._benchmark_resource_gate(peaks, self.resolved["runtime"])["passed"]
        )
        at_limit = dict(peaks)
        at_limit["peak_memory_gib_by_gpu"] = {"0": 20.0, "1": 19.9}
        self.assertFalse(
            sweep._benchmark_resource_gate(at_limit, self.resolved["runtime"])["passed"]
        )

    def test_pipeline_rejects_same_custom_benchmark_and_formal_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(ValueError, "Benchmark root must differ"):
                sweep.run_pipeline(
                    self.resolved,
                    root,
                    resume=False,
                    benchmark_root=root,
                )


if __name__ == "__main__":
    unittest.main()
