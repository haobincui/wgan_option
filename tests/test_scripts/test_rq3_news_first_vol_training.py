from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for value in (str(ROOT), str(SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.rq3.main import main as rq3_main  # noqa: E402
from scripts.rq3.news_first_vol_training import (  # noqa: E402
    DIRECT_CODE_PATHS,
    _job_status_path,
    _query_gpu_resources,
    _read_json,
    _sha256_file,
    _write_json,
    build_worker_command,
    launch_experiment,
    prepare_experiment,
    refresh_experiment_lineage,
    run_worker,
)


class TestNewsFirstVolTrainingOrchestration(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        summary_rows = []
        for tolerance in (5, 10, 15, 30):
            directory = self.dataset / f"tolerance_{tolerance:02d}m"
            directory.mkdir()
            (directory / "merged_vol.xlsx").write_bytes(f"workbook-{tolerance}".encode())
            summary_rows.append(
                {
                    "tolerance_minutes": tolerance,
                    "surface_usable_news": tolerance * 10,
                    "status": "pass",
                }
            )
        with (self.dataset / "dataset_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("tolerance_minutes", "surface_usable_news", "status"),
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        payload = yaml.safe_load(
            (ROOT / "configs/rq3/news_first_vol_training.yaml").read_text(encoding="utf-8")
        )
        config = payload["news_first_vol_training"]
        config["datasets"]["root"] = str(self.dataset)
        config["runtime"]["python_executable"] = sys.executable
        config["runtime"]["numactl_executable"] = "numactl"
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        self.experiment = self.root / "experiment"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _prepare(self) -> Path:
        def fake_split_manifest(resolved, experiment_root):
            del resolved
            path = experiment_root / "split_manifest.csv"
            path.write_text(
                "tolerance_minutes,train_rows,train_pairs,status\n"
                "5,1234,952,pass\n10,1506,1115,pass\n"
                "15,1748,1230,pass\n30,2263,1452,pass\n",
                encoding="utf-8",
            )
            return path

        with patch(
            "scripts.rq3.news_first_vol_training._build_split_manifest",
            side_effect=fake_split_manifest,
        ):
            return prepare_experiment(self.config_path, self.experiment)

    def test_narrow_grid_config_changes_only_the_dataset_root(self):
        formal = yaml.safe_load(
            (ROOT / "configs/rq3/news_first_vol_training.yaml").read_text(encoding="utf-8")
        )["news_first_vol_training"]
        narrow = yaml.safe_load(
            (
                ROOT / "configs/rq3/news_first_vol_training_narrow_grid.yaml"
            ).read_text(encoding="utf-8")
        )["news_first_vol_training"]

        self.assertEqual(
            narrow["datasets"]["root"],
            "data/processed/rq3/news_first_vol_surfaces_q097_103_ttm07_38_v1",
        )
        self.assertNotEqual(narrow["datasets"]["root"], formal["datasets"]["root"])
        narrow["datasets"]["root"] = formal["datasets"]["root"]
        self.assertEqual(narrow, formal)

    def test_prepare_materializes_frozen_two_wave_eight_job_registry(self):
        root = self._prepare()
        registry = _read_json(root / "registry/jobs.json")
        jobs = registry["jobs"]

        self.assertEqual(len(jobs), 8)
        self.assertEqual([sum(job["wave"] == wave for job in jobs) for wave in (1, 2)], [4, 4])
        self.assertEqual({job["model_family"] for job in jobs}, {"wgan", "regression"})
        self.assertEqual({job["tolerance_minutes"] for job in jobs}, {5, 10, 15, 30})
        assignment = {
            job["tolerance_minutes"]: (job["gpu_id"], job["gpu_slot"], job["numa_node"])
            for job in jobs
            if job["model_family"] == "wgan"
        }
        self.assertEqual(
            assignment,
            {5: (0, 0, 0), 30: (0, 1, 0), 10: (1, 0, 1), 15: (1, 1, 1)},
        )

        for job in jobs:
            training = yaml.safe_load(Path(job["training_config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(training["text_embedding_mode"], "lp")
            self.assertEqual(training["seed"], 42)
            self.assertEqual(training["news_first_train_end_utc"], "2023-07-01T00:00:00Z")
            self.assertEqual(training["news_first_validation_end_utc"], "2023-10-01T00:00:00Z")
            self.assertEqual(training["validation_mc_samples"], 16)
            self.assertEqual(training["best_checkpoint_metric"], "val_hybrid_score")
            self.assertEqual(training["num_epochs"], 100)
            self.assertGreater(training["save_every"], training["num_epochs"])
            self.assertEqual(_sha256_file(job["training_config_path"]), job["config_sha256"])

        for name in (
            "task_registry.csv",
            "resource_usage.csv",
            "config_hashes.csv",
            "source_hashes.csv",
            "code_hashes.csv",
            "run_manifest.json",
            "output_hashes.csv",
            "split_manifest.csv",
        ):
            if name == "resource_usage.csv":
                self.assertFalse((root / name).exists())
            else:
                self.assertTrue((root / name).is_file(), name)

        with (root / "code_hashes.csv").open("r", encoding="utf-8", newline="") as handle:
            code_rows = list(csv.DictReader(handle))
        self.assertEqual({row["relative_path"] for row in code_rows}, set(DIRECT_CODE_PATHS))
        for row in code_rows:
            code_path = Path(row["path"])
            self.assertEqual(row["sha256"], _sha256_file(code_path))
            self.assertEqual(int(row["size_bytes"]), code_path.stat().st_size)

        manifest = _read_json(root / "run_manifest.json")
        initial_git = manifest["initial_git_snapshot"]
        self.assertIn(initial_git["git_commit_status"], {"ok", "unavailable"})
        self.assertIsInstance(initial_git["git_dirty"], bool)
        self.assertIn("base commit only", initial_git["commit_scope_note"])
        self.assertEqual(manifest["lineage_capture_timing"], "before_first_worker_attempt")
        self.assertEqual(
            manifest["resource_telemetry"]["telemetry_scope"],
            "assigned_gpu_wave_aggregate",
        )
        self.assertEqual(manifest["resource_telemetry"]["concurrent_slots_on_gpu"], 2)

        with (root / "output_hashes.csv").open("r", encoding="utf-8", newline="") as handle:
            output_roles = {row["artifact_role"] for row in csv.DictReader(handle)}
        self.assertIn("experiment:code_hashes.csv", output_roles)
        self.assertIn("experiment:run_manifest.json", output_roles)

    def test_masked_text_ablation_prepare_materializes_24_jobs_in_six_waves(self):
        payload = yaml.safe_load(
            (
                ROOT
                / "configs/rq3/news_first_vol_training_narrow_grid_masked_ablation.yaml"
            ).read_text(encoding="utf-8")
        )
        config = payload["news_first_vol_training"]
        config["datasets"]["root"] = str(self.dataset)
        config["runtime"]["python_executable"] = sys.executable
        for tolerance in (5, 10, 15, 30):
            (
                self.dataset
                / f"tolerance_{tolerance:02d}m"
                / "surface_support_audit.csv.gz"
            ).write_bytes(b"fixture")
        config_path = self.root / "ablation.yaml"
        config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
        experiment = self.root / "ablation_experiment"

        def fake_split_manifest(resolved, experiment_root):
            del resolved
            path = experiment_root / "split_manifest.csv"
            path.write_text("tolerance_minutes,status\n5,pass\n", encoding="utf-8")
            return path

        with patch(
            "scripts.rq3.news_first_vol_training._build_split_manifest",
            side_effect=fake_split_manifest,
        ):
            root = prepare_experiment(config_path, experiment)
        jobs = _read_json(root / "registry/jobs.json")["jobs"]
        self.assertEqual(len(jobs), 24)
        self.assertEqual(
            [sum(int(job["wave"]) == wave for job in jobs) for wave in range(1, 7)],
            [4, 4, 4, 4, 4, 4],
        )
        self.assertEqual(
            {job["text_ablation_mode"] for job in jobs},
            {"real_text", "current_only", "text_shuffle"},
        )
        self.assertEqual({job["support_mask_mode"] for job in jobs}, {"raw_joint"})
        for job in jobs:
            training = yaml.safe_load(
                Path(job["training_config_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(training["support_mask_mode"], "raw_joint")
            self.assertEqual(
                training["news_first_text_ablation_mode"],
                job["text_ablation_mode"],
            )
            self.assertTrue(training["evaluate_initial_checkpoint"])
            self.assertEqual(
                training["residual_output_mode"], "identity_softplus_residual"
            )

    def test_prepare_reuse_is_idempotent_and_hash_mismatch_fails_closed(self):
        first = self._prepare()
        jobs_hash = _sha256_file(first / "registry/jobs.json")
        second = prepare_experiment(self.config_path, self.experiment, reuse=True)
        self.assertEqual(first, second)
        self.assertEqual(_sha256_file(second / "registry/jobs.json"), jobs_hash)

        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        payload["news_first_vol_training"]["runtime"]["resource_sample_interval_seconds"] = 3
        changed = self.root / "changed.yaml"
        changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "config hash differs"):
            prepare_experiment(changed, self.experiment, reuse=True)

    def test_prepare_rejects_dataset_file_that_disagrees_with_declared_hash(self):
        declared_lines = []
        for path in sorted(self.dataset.rglob("*")):
            if path.is_file() and path.name != "dataset_output_sha256.txt":
                relative = path.relative_to(self.dataset).as_posix()
                declared_lines.append(f"{_sha256_file(path)}  {relative}")
        (self.dataset / "dataset_output_sha256.txt").write_text(
            "\n".join(declared_lines) + "\n", encoding="utf-8"
        )
        (self.dataset / "tolerance_05m" / "merged_vol.xlsx").write_bytes(
            b"tampered-after-declaration"
        )

        with self.assertRaisesRegex(ValueError, "Dataset source hash mismatch"):
            self._prepare()

    def test_worker_rejects_dataset_tampering_before_training(self):
        root = self._prepare()
        (self.dataset / "tolerance_05m" / "merged_vol.xlsx").write_bytes(
            b"tampered-after-prepare"
        )

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False),
            patch(
                "scripts.rq3.news_first_vol_training._execute_training_job"
            ) as execute,
        ):
            with self.assertRaisesRegex(ValueError, "Dataset hash mismatch"):
                run_worker(root, "wgan_05m", dry_run=True)
        execute.assert_not_called()

    def test_lineage_backfill_after_attempt_preserves_every_job_status(self):
        root = self._prepare()
        first_job = _read_json(root / "registry/jobs.json")["jobs"][0]
        status_path = _job_status_path(root, first_job["job_id"])
        attempted_status = _read_json(status_path)
        attempted_status["attempt"] = 1
        attempted_status["status"] = "failed"
        attempted_status["error"] = "fixture"
        _write_json(status_path, attempted_status)
        for name in ("code_hashes.csv", "run_manifest.json"):
            (root / name).unlink()

        status_paths = sorted((root / "registry/jobs").glob("*.status.json"))
        before = {path.name: path.read_bytes() for path in status_paths}
        manifest_path = refresh_experiment_lineage(self.config_path, root)
        after = {path.name: path.read_bytes() for path in status_paths}

        self.assertEqual(after, before)
        self.assertTrue((root / "code_hashes.csv").is_file())
        manifest = _read_json(manifest_path)
        self.assertEqual(manifest["first_capture_max_job_attempt"], 1)
        self.assertEqual(
            manifest["lineage_capture_timing"],
            "backfilled_after_worker_attempts_started",
        )
        self.assertTrue(manifest["lineage_limitation"])

        # The public prepare --reuse path also refreshes the root output hash manifest.
        prepare_experiment(self.config_path, root, reuse=True)
        after_reuse = {path.name: path.read_bytes() for path in status_paths}
        self.assertEqual(after_reuse, before)
        with (root / "output_hashes.csv").open("r", encoding="utf-8", newline="") as handle:
            output_roles = {row["artifact_role"] for row in csv.DictReader(handle)}
        self.assertIn("experiment:code_hashes.csv", output_roles)
        self.assertIn("experiment:run_manifest.json", output_roles)

    def test_worker_command_contains_isolated_gpu_numa_and_hidden_worker_mode(self):
        root = self._prepare()
        job = next(job for job in _read_json(root / "registry/jobs.json")["jobs"] if job["job_id"] == "wgan_30m")
        command = build_worker_command(
            self.config_path,
            root,
            job,
            dry_run=True,
            resume=True,
        )
        self.assertEqual(command[:3], ["numactl", "--cpunodebind=0", "--membind=0"])
        self.assertIn("train-news-first-vol-comparison", command)
        self.assertIn("worker", command)
        self.assertIn("wgan_30m", command)
        self.assertIn("--worker-dry-run", command)
        self.assertIn("--resume", command)

    def test_worker_dry_run_transitions_status_without_importing_training_stack(self):
        root = self._prepare()
        fake_run = root / "runs/fake"
        fake_run.mkdir(parents=True)
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False),
            patch(
                "scripts.rq3.news_first_vol_training._execute_training_job",
                return_value=(fake_run, []),
            ) as execute,
        ):
            result = run_worker(root, "wgan_05m", dry_run=True)
        self.assertEqual(result, fake_run)
        execute.assert_called_once()
        status = _read_json(_job_status_path(root, "wgan_05m"))
        self.assertEqual(status["status"], "dry_run_passed")
        self.assertEqual(status["attempt"], 1)
        self.assertEqual(status["exit_code"], 0)

    def test_worker_rejects_wrong_visible_gpu_before_torch_import(self):
        root = self._prepare()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "GPU mismatch"):
                run_worker(root, "wgan_05m", dry_run=True)

    def test_launcher_sequences_two_four_job_waves_and_exports_terminal_registry(self):
        root = self._prepare()
        calls = []

        def complete_wave(experiment_root, config_path, wave, jobs, **kwargs):
            del config_path, kwargs
            calls.append((wave, [job["job_id"] for job in jobs]))
            for job in jobs:
                artifact = experiment_root / "fake_artifacts" / f"{job['job_id']}.bin"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(job["job_id"].encode())
                _write_json(
                    _job_status_path(experiment_root, job["job_id"]),
                    {
                        "job_id": job["job_id"],
                        "status": "completed",
                        "attempt": 1,
                        "config_sha256": job["config_sha256"],
                        "run_dir": str(experiment_root / "runs" / job["job_id"]),
                        "artifacts": [
                            {
                                "artifact_role": "fixture",
                                "path": str(artifact),
                                "size_bytes": artifact.stat().st_size,
                                "sha256": _sha256_file(artifact),
                            }
                        ],
                        "exit_code": 0,
                    },
                )

        with (
            patch("scripts.rq3.news_first_vol_training._run_wave", side_effect=complete_wave),
            patch("scripts.rq3.news_first_vol_training.run_default_postprocess") as postprocess,
        ):
            result = launch_experiment(self.config_path, root)
        self.assertEqual(result, root)
        self.assertEqual([wave for wave, _ in calls], [1, 2])
        self.assertTrue(all(len(jobs) == 4 for _, jobs in calls))
        postprocess.assert_called_once_with(root)
        status = _read_json(root / "registry/experiment_status.json")
        self.assertEqual(status["status"], "completed")
        with (root / "task_registry.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["status"] for row in rows}, {"completed"})
        with (root / "output_hashes.csv").open("r", encoding="utf-8", newline="") as handle:
            output_rows = list(csv.DictReader(handle))
        self.assertEqual(
            {row["job_id"] for row in output_rows if row["job_id"]},
            {
                "wgan_05m", "wgan_10m", "wgan_15m", "wgan_30m",
                "regression_05m", "regression_10m", "regression_15m", "regression_30m",
            },
        )
        root_roles = {row["artifact_role"] for row in output_rows if not row["job_id"]}
        self.assertIn("experiment:task_registry.csv", root_roles)
        self.assertIn("experiment:split_manifest.csv", root_roles)
        self.assertIn("experiment:resource_summary.csv", root_roles)
        self.assertIn("experiment:code_hashes.csv", root_roles)
        self.assertIn("experiment:run_manifest.json", root_roles)
        with (root / "resource_summary.csv").open("r", encoding="utf-8", newline="") as handle:
            resources = list(csv.DictReader(handle))
        self.assertEqual(len(resources), 8)
        self.assertEqual({row["model"] for row in resources}, {"wgan", "regression"})
        self.assertEqual(
            {row["telemetry_scope"] for row in resources},
            {"assigned_gpu_wave_aggregate"},
        )
        self.assertEqual({row["concurrent_slots_on_gpu"] for row in resources}, {"2"})

    def test_resume_skips_hash_valid_completed_jobs(self):
        root = self._prepare()
        job = _read_json(root / "registry/jobs.json")["jobs"][0]
        artifact = root / "artifact.bin"
        artifact.write_bytes(b"valid")
        status = {
            "job_id": job["job_id"],
            "status": "completed",
            "attempt": 1,
            "config_sha256": job["config_sha256"],
            "run_dir": str(root / "existing_run"),
            "artifacts": [
                {
                    "artifact_role": "fixture",
                    "path": str(artifact),
                    "size_bytes": artifact.stat().st_size,
                    "sha256": _sha256_file(artifact),
                }
            ],
        }
        _write_json(_job_status_path(root, job["job_id"]), status)
        with patch("scripts.rq3.news_first_vol_training._execute_training_job") as execute:
            result = run_worker(root, job["job_id"], resume=True)
        self.assertEqual(result, root / "existing_run")
        execute.assert_not_called()

    def test_gpu_resource_parser_keeps_stable_schema_and_errors(self):
        success = type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": "2026/08/19 12:00:00.000, 0, GPU-abc, NVIDIA A30, 97, 29, 1923, 24576, 122.4, 38\n",
                "stderr": "",
            },
        )()
        with patch("scripts.rq3.news_first_vol_training.subprocess.run", return_value=success):
            rows = _query_gpu_resources("nvidia-smi", 1)
        self.assertEqual(rows[0]["gpu_index"], "0")
        self.assertEqual(rows[0]["utilization_gpu_pct"], "97")
        self.assertEqual(rows[0]["sample_status"], "ok")

        failure = type("Completed", (), {"returncode": 1, "stdout": "", "stderr": "no gpu"})()
        with patch("scripts.rq3.news_first_vol_training.subprocess.run", return_value=failure):
            rows = _query_gpu_resources("nvidia-smi", 2)
        self.assertEqual(rows[0]["sample_status"], "error")
        self.assertIn("no gpu", rows[0]["error"])

    def test_rq3_main_dispatches_new_public_command_lazily(self):
        expected = self.root / "result"
        with patch(
            "scripts.rq3.news_first_vol_training.run_news_first_vol_training",
            return_value=expected,
        ) as runner:
            result = rq3_main(
                [
                    "train-news-first-vol-comparison",
                    "prepare",
                    "--config",
                    str(self.config_path),
                    "--output-dir",
                    str(self.experiment),
                ]
            )
        self.assertEqual(result, expected)
        runner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
