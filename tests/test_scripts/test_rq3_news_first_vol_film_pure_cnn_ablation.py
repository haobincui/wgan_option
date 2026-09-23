from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts.rq3 import news_first_vol_film_pure_cnn_ablation as sweep


class FilmPureCnnAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolved = sweep.resolve_config(sweep.DEFAULT_CONFIG)
        cls.specs = sweep.experiment_specs(cls.resolved)

    def test_frozen_single_factor_arm_order(self) -> None:
        self.assertEqual(
            sweep.ARM_IDS,
            (
                "baseline",
                "film_dense_no_text_concat",
                "film_unet_text128",
                "film_unet_mask_coords_text128",
                "film_unet_mask_coords_text64",
                "film_unet_mask_coords_text64_nolp",
                "film_unet_mask_coords_text64_projection",
            ),
        )
        modes = {
            arm_id: (
                sweep.ARM_CONTRACTS[arm_id]["generator_conditioning_mode"],
                sweep.ARM_CONTRACTS[arm_id]["critic_conditioning_mode"],
                sweep.ARM_CONTRACTS[arm_id]["gen_text_hidden_dim"],
                sweep.ARM_CONTRACTS[arm_id]["gen_text_out_dim"],
            )
            for arm_id in sweep.ARM_IDS
        }
        self.assertEqual(
            modes["baseline"],
            ("film_conv_bottleneck_concat_v1", "lp_concat_v1", 256, 128),
        )
        self.assertEqual(
            modes["film_dense_no_text_concat"],
            ("film_conv_no_bottleneck_concat_v1", "lp_concat_v1", 256, 128),
        )
        self.assertEqual(
            modes["film_unet_text128"],
            ("film_unet_v1", "lp_concat_v1", 256, 128),
        )
        self.assertEqual(
            modes["film_unet_mask_coords_text128"],
            ("film_unet_mask_coords_v1", "lp_concat_v1", 256, 128),
        )
        self.assertEqual(
            modes["film_unet_mask_coords_text64_nolp"][1],
            "lp_disabled_same_shape_v1",
        )
        self.assertEqual(
            modes["film_unet_mask_coords_text64_projection"][1],
            "lp_projection_v1",
        )

    def test_matrix_is_exact_unique_and_gpu_balanced(self) -> None:
        self.assertEqual(len(self.specs), 21)
        self.assertEqual(len({spec["job_id"] for spec in self.specs}), 21)
        self.assertEqual({spec["seed"] for spec in self.specs}, {42, 202, 404})
        self.assertEqual({spec["learning_rate"] for spec in self.specs}, {5e-7})
        self.assertEqual(
            {gpu: sum(spec["gpu_id"] == gpu for spec in self.specs) for gpu in (0, 1)},
            {0: 11, 1: 10},
        )
        for seed in sweep.SEEDS:
            counts = {
                gpu: sum(
                    spec["seed"] == seed and spec["gpu_id"] == gpu
                    for spec in self.specs
                )
                for gpu in sweep.GPU_IDS
            }
            self.assertLessEqual(abs(counts[0] - counts[1]), 1)

    def test_executable_parameter_counts_are_frozen(self) -> None:
        counts = {
            spec["arm_id"]: (
                spec["generator_parameters"],
                spec["critic_parameters"],
                spec["wgan_parameters"],
            )
            for spec in self.specs
        }
        self.assertEqual(counts["baseline"], (4_020_288, 729_157, 4_749_445))
        self.assertEqual(
            counts["film_dense_no_text_concat"],
            (3_889_216, 729_157, 4_618_373),
        )
        self.assertEqual(
            counts["film_unet_text128"],
            (826_881, 729_157, 1_556_038),
        )
        self.assertEqual(
            counts["film_unet_mask_coords_text128"],
            (827_745, 729_157, 1_556_902),
        )
        for arm_id in (
            "film_unet_mask_coords_text64",
            "film_unet_mask_coords_text64_nolp",
            "film_unet_mask_coords_text64_projection",
        ):
            self.assertEqual(counts[arm_id], (544_481, 729_157, 1_273_638))

    def test_identical_graph_arms_share_seeded_initial_states(self) -> None:
        by_key = {(spec["arm_id"], spec["seed"]): spec for spec in self.specs}
        for seed in sweep.SEEDS:
            generator_hashes = {
                by_key[(arm_id, seed)]["initial_generator_state_sha256"]
                for arm_id in (
                    "film_unet_mask_coords_text64",
                    "film_unet_mask_coords_text64_nolp",
                    "film_unet_mask_coords_text64_projection",
                )
            }
            self.assertEqual(len(generator_hashes), 1)
            critic_hashes = {
                by_key[(arm_id, seed)]["initial_critic_state_sha256"]
                for arm_id in (
                    "film_unet_mask_coords_text64",
                    "film_unet_mask_coords_text64_nolp",
                )
            }
            self.assertEqual(len(critic_hashes), 1)
            self.assertNotEqual(
                by_key[("film_unet_mask_coords_text64", seed)][
                    "initial_critic_state_sha256"
                ],
                by_key[("film_unet_mask_coords_text64_projection", seed)][
                    "initial_critic_state_sha256"
                ],
            )

    def test_training_payload_freezes_q3_only_no_warmup_contract(self) -> None:
        spec = next(
            spec
            for spec in self.specs
            if spec["arm_id"] == "film_unet_mask_coords_text64_projection"
            and spec["seed"] == 404
        )
        payload = sweep._training_payload(
            self.resolved,
            Path("/tmp/pure-cnn-ablation"),
            spec,
        )
        self.assertEqual(
            payload["generator_conditioning_mode"], "film_unet_mask_coords_v1"
        )
        self.assertEqual(payload["critic_conditioning_mode"], "lp_projection_v1")
        self.assertEqual(payload["gen_base_channels"], 32)
        self.assertEqual(payload["disc_base_channels"], 32)
        self.assertEqual(payload["gen_text_hidden_dim"], 64)
        self.assertEqual(payload["gen_text_out_dim"], 64)
        self.assertEqual(payload["generator_learning_rate"], 5e-7)
        self.assertEqual(payload["discriminator_learning_rate"], 5e-7)
        self.assertEqual(payload["lr_warmup_epochs"], 0)
        self.assertEqual(payload["num_epochs"], 30)
        self.assertEqual(payload["early_stopping_min_epochs"], 15)
        self.assertEqual(payload["early_stopping_patience"], 16)
        self.assertEqual(payload["validation_mc_samples"], 16)
        self.assertEqual(payload["news_first_train_end_utc"], "2023-07-01T00:00:00Z")
        self.assertEqual(
            payload["news_first_validation_end_utc"],
            "2023-10-01T00:00:00Z",
        )
        self.assertTrue(payload["news_first_materialize_validation_loader"])
        self.assertFalse(payload["news_first_materialize_test_loader"])

    def test_prepare_is_idempotent_and_rejects_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            registry = sweep.prepare(self.resolved, root, resume=False)
            self.assertEqual(registry["expected_job_count"], 21)
            self.assertEqual(len(list((root / "configs").glob("*.yaml"))), 21)
            fairness_path = Path(registry["initial_state_fairness_path"])
            self.assertTrue(fairness_path.is_file())
            fairness = json.loads(fairness_path.read_text(encoding="utf-8"))
            self.assertEqual(len(fairness["seeds"]), 3)
            resumed = sweep.prepare(self.resolved, root, resume=True)
            self.assertEqual(
                resumed["jobs_payload_sha256"], registry["jobs_payload_sha256"]
            )
            first_config = Path(registry["jobs"][0]["config_path"])
            first_config.write_text(
                first_config.read_text(encoding="utf-8") + "\n# drift\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "config drift"):
                sweep.prepare(self.resolved, root, resume=True)

    def test_dry_run_checks_one_cell_per_arm_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            sweep.prepare(self.resolved, root, resume=False)
            completed = subprocess.CompletedProcess(args=[], returncode=0)
            with mock.patch.object(
                sweep.subprocess, "run", return_value=completed
            ) as run:
                manifest = sweep.dry_run(self.resolved, root, resume=True)
                self.assertEqual(run.call_count, len(sweep.ARM_IDS))
            self.assertEqual(len(manifest["artifacts"]), len(sweep.ARM_IDS))
            with mock.patch.object(sweep.subprocess, "run") as run:
                resumed = sweep.dry_run(self.resolved, root, resume=True)
                run.assert_not_called()
            self.assertEqual(resumed["contract_sha256"], manifest["contract_sha256"])

    def test_benchmark_registry_is_isolated_and_one_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "benchmark"
            registry = sweep._prepare_benchmark(self.resolved, root, resume=False)
            self.assertEqual(
                registry["experiment_kind"], sweep.BENCHMARK_EXPERIMENT_KIND
            )
            self.assertEqual(registry["expected_job_count"], 21)
            self.assertEqual({job["num_epochs"] for job in registry["jobs"]}, {1})
            first_config = Path(registry["jobs"][0]["config_path"])
            payload = sweep._require_mapping(
                sweep.yaml.safe_load(first_config.read_text(encoding="utf-8")),
                "benchmark config",
            )
            self.assertEqual(payload["num_epochs"], 1)
            self.assertFalse(payload["use_early_stopping"])
            self.assertFalse(payload["news_first_materialize_test_loader"])
            self.assertNotEqual(
                root.resolve(), sweep._resolve_repo_path(sweep.DEFAULT_OUTPUT_DIR)
            )

    def test_postprocess_ranks_all_complete_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            registry = sweep.prepare(self.resolved, root, resume=False)
            arm_factor = {
                arm_id: 0.999 - index * 0.001
                for index, arm_id in enumerate(sweep.ARM_IDS)
            }
            persistence_by_seed = {42: 0.0018, 202: 0.0019, 404: 0.0020}
            for job in registry["jobs"]:
                artifact_path = root / f"registry/jobs/{job['job_id']}.artifact"
                artifact_path.write_text("frozen\n", encoding="utf-8")
                persistence = persistence_by_seed[int(job["seed"])]
                sweep._write_status(
                    root,
                    str(job["job_id"]),
                    {
                        "job_id": job["job_id"],
                        "state": "complete",
                        "attempt": 1,
                        "best_epoch": 7,
                        "completed_epochs": 23,
                        "best_mae": persistence * arm_factor[str(job["arm_id"])],
                        "persistence_mae": persistence,
                        "run_dir": str(root / "runs" / str(job["job_id"])),
                        "artifacts": [sweep._artifact(artifact_path, "fixture")],
                    },
                )
            result = sweep.postprocess(root)
            self.assertEqual(result["completed_jobs"], 21)
            self.assertEqual(
                result["point_leader"],
                "film_unet_mask_coords_text64_projection",
            )
            self.assertEqual([row["rank"] for row in result["arms"]], list(range(1, 8)))
            self.assertTrue((root / "analysis/job_summary.csv").is_file())
            ranking = json.loads(
                (root / "analysis/arm_ranking.json").read_text(encoding="utf-8")
            )
            self.assertIn("Q3 development only", ranking["selection_scope"])

    def test_cli_exposes_all_lifecycle_actions(self) -> None:
        parser = sweep.build_parser()
        for action in (
            "benchmark",
            "prepare",
            "dry-run",
            "worker",
            "launch",
            "status",
            "postprocess",
            "run-pipeline",
        ):
            argv = [action]
            if action == "worker":
                argv.extend(["--job-id", "baseline_seed_042"])
            self.assertEqual(parser.parse_args(argv).action, action)


if __name__ == "__main__":
    unittest.main()
