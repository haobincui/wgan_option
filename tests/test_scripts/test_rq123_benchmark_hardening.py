from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.rq123 import news_first_vol_film_nolp_10seed as experiment


CONFIG = Path("configs/rq123/news_first_vol_film_nolp_legacy_10seed_v2.yaml")


class Rq123BenchmarkHardeningTests(unittest.TestCase):
    def test_selected_concurrency_is_frozen_in_config_and_runtime_contract(
        self,
    ) -> None:
        config = experiment.load_config(CONFIG)
        frozen = experiment._with_selected_workers(config, 12)
        self.assertIsNone(config["runtime"]["formal_workers_per_gpu"])
        self.assertEqual(frozen["runtime"]["formal_workers_per_gpu"], 12)
        contract = experiment._runtime_contract(
            frozen, workers_per_gpu=12, mode="benchmark_candidate"
        )
        unsigned = {
            key: value for key, value in contract.items() if key != "payload_sha256"
        }
        self.assertEqual(
            contract["payload_sha256"], experiment.payload_sha256(unsigned)
        )
        self.assertEqual(contract["selected_workers_per_gpu"], 12)

    def test_prepare_contract_is_hash_bound_and_detects_tampering(self) -> None:
        source = experiment.load_config(CONFIG)
        config = experiment._with_selected_workers(source, 18)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal"
            root.mkdir()
            experiment._write_prepare_contract(
                root,
                config,
                workers_per_gpu=18,
                mode="formal",
                status="preparing",
            )
            observed = experiment._validate_prepare_contract(
                root,
                config,
                workers_per_gpu=18,
                mode="formal",
            )
            self.assertEqual(observed["status"], "preparing")
            path = experiment._prepare_contract_path(root)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["selected_workers_per_gpu"] = 12
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Preparation contract drift"):
                experiment._validate_prepare_contract(
                    root,
                    config,
                    workers_per_gpu=18,
                    mode="formal",
                )

    def test_non_capacity_benchmark_failure_does_not_try_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal_root = Path(temporary) / "formal"
            config = {
                "source_config_sha256": "a" * 64,
                "experiment": {"output_root": str(formal_root)},
            }
            with (
                patch.object(experiment, "load_config", return_value=config),
                patch.object(
                    experiment,
                    "_run_one_benchmark_candidate",
                    side_effect=ValueError("dataset hash drift"),
                ) as candidate,
            ):
                with self.assertRaisesRegex(ValueError, "dataset hash drift"):
                    experiment.run_benchmark(CONFIG, formal_root)
            self.assertEqual(candidate.call_count, 1)

    def test_capacity_gate_is_the_only_path_to_twelve_worker_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal_root = Path(temporary) / "formal"
            config = {
                "source_config_sha256": "a" * 64,
                "experiment": {"output_root": str(formal_root)},
            }
            passed = {
                "status": "passed",
                "config_sha256": "a" * 64,
                "job_count": 36,
                "selected_workers_per_gpu": 12,
            }
            with (
                patch.object(experiment, "load_config", return_value=config),
                patch.object(
                    experiment,
                    "_run_one_benchmark_candidate",
                    side_effect=[
                        experiment.BenchmarkConcurrencyGateError("GPU gate"),
                        passed,
                    ],
                ) as candidate,
                patch.object(experiment, "_require_benchmark", return_value=passed),
            ):
                result_path = experiment.run_benchmark(CONFIG, formal_root)
            self.assertTrue(result_path.is_file())
            self.assertEqual(
                [call.kwargs["workers_per_gpu"] for call in candidate.call_args_list],
                [18, 12],
            )
            result = experiment.read_json(result_path)
            self.assertEqual(result["selected_workers_per_gpu"], 12)
            self.assertEqual(len(result["candidate_failures"]), 1)

    def test_capacity_detection_reads_failed_worker_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "registry/job_status").mkdir(parents=True)
            (root / "logs").mkdir()
            experiment.write_json(
                root / "registry/job_status/job.json",
                {"status": "failed", "error": "worker exited"},
            )
            (root / "logs/job.log").write_text(
                "RuntimeError: CUDA out of memory\n", encoding="utf-8"
            )
            self.assertTrue(
                experiment._benchmark_failure_is_capacity_related(
                    root, RuntimeError("wave failed")
                )
            )
            (root / "logs/job.log").write_text(
                "ValueError: dataset SHA drift\n", encoding="utf-8"
            )
            self.assertFalse(
                experiment._benchmark_failure_is_capacity_related(
                    root, RuntimeError("wave failed")
                )
            )

    def test_capacity_gate_journal_survives_resume_and_is_hash_bound(self) -> None:
        config = experiment.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "benchmark"
            root.mkdir()
            path = experiment._record_benchmark_capacity_gate(
                root,
                config,
                workers_per_gpu=18,
                reason="host RAM gate",
            )
            observed = experiment._read_benchmark_capacity_gate(
                root, config, workers_per_gpu=18
            )
            self.assertEqual(observed["reason"], "host RAM gate")
            payload = experiment.read_json(path)
            payload["reason"] = "forged"
            experiment.write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "journal drift"):
                experiment._read_benchmark_capacity_gate(
                    root, config, workers_per_gpu=18
                )

    def test_benchmark_resume_reuses_a_valid_interrupted_root(self) -> None:
        source = experiment.load_config(CONFIG)
        frozen = experiment._with_selected_workers(source, 18)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "benchmark"
            root.mkdir()
            with (
                patch.object(experiment, "load_config", return_value=source),
                patch.object(experiment, "validate_root", return_value=frozen),
                patch.object(
                    experiment,
                    "read_registry",
                    return_value={"formal_workers_per_gpu": 18},
                ),
            ):
                observed = experiment._prepare_benchmark_root(
                    CONFIG, root, workers_per_gpu=18, resume=True
                )
        self.assertEqual(observed, root)

    def test_invalid_prepared_partial_root_fails_closed(self) -> None:
        source = experiment.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "benchmark"
            root.mkdir()
            with (
                patch.object(experiment, "load_config", return_value=source),
                patch.object(
                    experiment, "validate_root", side_effect=ValueError("drift")
                ),
                patch.object(
                    experiment,
                    "_validate_prepare_contract",
                    return_value={"status": "prepared"},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "not safely resumable"):
                    experiment._prepare_benchmark_root(
                        CONFIG, root, workers_per_gpu=18, resume=True
                    )

    def test_current_manifest_comparison_rejects_rebound_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("same", encoding="utf-8")
            second.write_text("same", encoding="utf-8")
            manifest = root / "hashes.csv"
            experiment._manifest_csv(
                manifest, [experiment.manifest_row("source", first)]
            )
            experiment._assert_manifest_equals_current(
                manifest, [experiment.manifest_row("source", first)]
            )
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                experiment._assert_manifest_equals_current(
                    manifest, [experiment.manifest_row("source", second)]
                )

    def test_formal_prepare_safely_resumes_owned_partial_and_freezes_slots(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal"
            loaded = experiment.load_config(CONFIG)
            source = {
                **loaded,
                "experiment": {**loaded["experiment"], "output_root": str(root)},
            }
            frozen = experiment._with_selected_workers(source, 12)
            result_path = experiment._benchmark_result_path(root)
            result_path.parent.mkdir(parents=True)
            result_path.write_text("{}\n", encoding="utf-8")
            experiment._recovery_canary_result_path(root).write_text(
                "{}\n", encoding="utf-8"
            )
            experiment._matrix_smoke_result_path(root).write_text(
                "{}\n", encoding="utf-8"
            )
            root.mkdir()
            experiment._write_prepare_contract(
                root,
                frozen,
                workers_per_gpu=12,
                mode="formal",
                status="preparing",
                benchmark_result_path=result_path,
            )

            def fake_build_job(config, build_root, spec, **_kwargs):
                identifier = experiment.job_id(spec)
                job = {
                    **dict(spec),
                    "job_id": identifier,
                    "job_spec_sha256": f"sha-{identifier}",
                }
                return job, {}

            with (
                patch.object(experiment, "load_config", return_value=source),
                patch.object(
                    experiment,
                    "_require_benchmark",
                    return_value={"selected_workers_per_gpu": 12},
                ),
                patch.object(
                    experiment,
                    "_require_matrix_smoke",
                    return_value={"selected_workers_per_gpu": 12},
                ),
                patch.object(
                    experiment,
                    "_require_recovery_canary",
                    return_value={"payload_sha256": "canary-payload"},
                ),
                patch.object(
                    experiment,
                    "validate_root",
                    side_effect=[ValueError("partial"), frozen],
                ),
                patch.object(experiment, "materialize_pair_universes"),
                patch.object(experiment, "materialize_pair_text_overlays"),
                patch.object(experiment, "build_job", side_effect=fake_build_job),
                patch.object(experiment, "_prepare_hash_manifests"),
            ):
                observed = experiment.prepare_experiment(CONFIG, root, resume=True)
            self.assertEqual(observed, root)
            resolved = experiment.yaml.safe_load(
                (root / "resolved_config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(resolved["runtime"]["formal_workers_per_gpu"], 12)
            runtime_contract = experiment.read_json(root / "runtime_contract.json")
            self.assertEqual(runtime_contract["selected_workers_per_gpu"], 12)
            self.assertEqual(
                experiment.read_json(experiment._prepare_contract_path(root))["status"],
                "prepared",
            )

    def test_storage_projection_uses_role_specific_stage_counts(self) -> None:
        roles = {
            "generator_initial_epoch0": 11,
            "discriminator_initial_epoch0": 5,
            "generator_best_learned": 11,
            "discriminator_best_learned": 5,
            "generator_final": 11,
            "discriminator_final": 5,
            "best_learned_checkpoint": 2,
            "full_training_state": 31,
            "resolved_training_config": 3,
            "full_state_contract": 4,
            "training_metrics_csv": 7,
            "training_metrics_json": 8,
            "run_log": 9,
        }
        status = {
            "artifacts": [
                {"artifact_role": role, "size_bytes": size}
                for role, size in roles.items()
            ]
        }
        jobs = [{"job_id": f"job-{index}"} for index in range(36)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "inputs").mkdir()
            (root / "inputs/overlay.json").write_bytes(b"x" * 13)
            with (
                patch.object(experiment, "_stage_jobs", return_value=jobs),
                patch.object(experiment, "read_json", return_value=status),
                patch.object(experiment, "_completed_valid", return_value=True),
                patch.object(
                    experiment,
                    "validate_root",
                    return_value={"training": {"num_epochs": 10}},
                ),
            ):
                result = experiment._project_formal_storage_breakdown(root)
        dynamic = 160
        branches = 240
        self.assertEqual(result["selected_full_training_states"], 31 * dynamic)
        self.assertEqual(
            result["dynamic_checkpoints"],
            (11 + 5) * dynamic,
        )
        self.assertEqual(result["branch_checkpoints"], (11 + 5) * branches)
        self.assertEqual(
            result["epoch_scaled_metrics_and_logs"],
            (7 + 8 + 9) * 400 * 10,
        )
        self.assertEqual(result["development_and_test_inputs"], 26)
        self.assertEqual(
            result["total"],
            sum(value for key, value in result.items() if key != "total"),
        )


if __name__ == "__main__":
    unittest.main()
