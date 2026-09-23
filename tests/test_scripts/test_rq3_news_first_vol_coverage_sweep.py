from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for value in (str(ROOT), str(SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.rq3 import news_first_vol_coverage_sweep as sweep  # noqa: E402


CONFIG_PATH = ROOT / "configs/rq3/news_first_vol_coverage_completion_sweep.yaml"


class CoverageSweepTests(unittest.TestCase):
    def _payload(self) -> dict:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    @contextmanager
    def _fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            temporary = Path(tmp)
            payload = self._payload()
            config = payload[sweep.ROOT_KEY]
            dataset_root = temporary / "dataset"
            for tolerance in (5, 10, 15, 30):
                directory = dataset_root / f"tolerance_{tolerance:02d}m"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "merged_vol.xlsx").write_bytes(
                    f"workbook-{tolerance}".encode()
                )
                (directory / "surface_support_audit.csv.gz").write_bytes(
                    f"support-{tolerance}".encode()
                )
            (dataset_root / "dataset_summary.csv").write_text(
                "tolerance_minutes,status\n5,pass\n10,pass\n15,pass\n30,pass\n",
                encoding="utf-8",
            )
            config["datasets"]["root"] = str(dataset_root)

            references = config["coverage_completion_sweep"]["reference_experiments"]
            for name, reference in references.items():
                reference_root = temporary / f"reference_{name}"
                pair_metrics = reference_root / "analysis" / "pair_metrics.csv.gz"
                registry = reference_root / "registry" / "jobs.json"
                resolved_hash = reference_root / "registry" / "resolved_config.sha256"
                pair_metrics.parent.mkdir(parents=True, exist_ok=True)
                registry.parent.mkdir(parents=True, exist_ok=True)
                pair_metrics.write_bytes(f"pairs-{name}".encode())
                registry.write_text('{"jobs": []}\n', encoding="utf-8")
                resolved_hash.write_text(f"hash-{name}\n", encoding="utf-8")
                reference.update(
                    {
                        "root": str(reference_root),
                        "pair_metrics": str(pair_metrics),
                        "registry": str(registry),
                        "resolved_config_hash": str(resolved_hash),
                    }
                )
            config_path = temporary / "coverage.yaml"
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            experiment_root = temporary / "experiment"

            def fake_split(_resolved, root):
                (root / "split_manifest.csv").write_text(
                    "tolerance_minutes,status\n5,pass\n", encoding="utf-8"
                )
                (root / "text_ablation_manifest.csv").write_text(
                    "mode,status\nreal_text,pass\n", encoding="utf-8"
                )
                return root / "split_manifest.csv"

            with (
                patch.object(sweep, "_build_split_manifest", side_effect=fake_split),
                patch.object(sweep.training, "_validate_dataset_summary"),
            ):
                yield config_path, experiment_root

    def test_frozen_456_job_matrix_and_balanced_dynamic_waves(self):
        resolved = sweep._resolved_config(CONFIG_PATH)
        jobs = sweep._job_specs(resolved)
        self.assertEqual(len(jobs), 456)
        self.assertEqual(
            Counter(job["stage_id"] for job in jobs),
            Counter(sweep.EXPECTED_STAGE_JOBS),
        )
        self.assertEqual(len({sweep._job_id(job) for job in jobs}), 456)

        stage_one = [job for job in jobs if job["stage_id"] == sweep.STAGE_IDS[0]]
        by_wave = defaultdict(list)
        for job in stage_one:
            by_wave[int(job["stage_wave"])].append(job)
        self.assertEqual([len(by_wave[index]) for index in sorted(by_wave)], [48, 24])
        for rows in by_wave.values():
            gpu_counts = Counter(int(row["gpu_id"]) for row in rows)
            self.assertEqual(len(set(gpu_counts.values())), 1)
            profile_counts = Counter(row["capacity_profile"] for row in rows)
            seed_counts = Counter(int(row["seed"]) for row in rows)
            self.assertEqual(len(set(profile_counts.values())), 1)
            self.assertEqual(len(set(seed_counts.values())), 1)

        stage_four = [job for job in jobs if job["stage_id"] == sweep.STAGE_IDS[3]]
        self.assertEqual(
            Counter(job["stage_wave"] for job in stage_four),
            Counter({1: 48, 2: 48, 3: 48, 4: 48, 5: 48}),
        )
        self.assertEqual(
            {
                (
                    job["capacity_profile"],
                    job["lr_profile"],
                    job["seed"],
                    job["tolerance_minutes"],
                    job["text_ablation_mode"],
                )
                for job in stage_four
            },
            {
                (profile, lr, seed, tolerance, mode)
                for profile in ("micro", "tiny", "small", "medium", "legacy")
                for lr in ("lr_7_5e_07", "lr_1e_06", "lr_1_5e_06", "lr_2e_06")
                for seed in (42, 202, 404)
                for tolerance in (5, 30)
                for mode in ("current_only", "real_text")
            },
        )

    def test_refresh_exports_hashes_final_analysis_report_and_candidate(self):
        with self._fixture() as (config_path, experiment_root):
            root = sweep.prepare_coverage_sweep_experiment(config_path, experiment_root)
            analysis = root / "analysis" / "summary.csv"
            report = root / "report" / "report.html"
            candidate = root / "coverage_final_candidate.json"
            analysis.parent.mkdir(parents=True, exist_ok=True)
            report.parent.mkdir(parents=True, exist_ok=True)
            analysis.write_text("metric,value\nmae,1\n", encoding="utf-8")
            report.write_text("<html></html>\n", encoding="utf-8")
            candidate.write_text('{"candidate":"small"}\n', encoding="utf-8")

            sweep._refresh_exports(root)
            with (root / "output_hashes.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            paths = {Path(str(row["path"])) for row in rows}
            self.assertIn(analysis, paths)
            self.assertIn(report, paths)
            self.assertIn(candidate, paths)

    def test_stage_specific_slots_and_preferred_numa_worker_command(self):
        with self._fixture() as (config_path, root):
            sweep.prepare_coverage_sweep_experiment(config_path, root)
            registry = sweep._load_registry(root)
            stage_one = [
                job for job in registry["jobs"] if job["stage_id"] == sweep.STAGE_IDS[0]
            ]
            stage_three = [
                job for job in registry["jobs"] if job["stage_id"] == sweep.STAGE_IDS[2]
            ]
            self.assertEqual(max(int(job["gpu_slot"]) for job in stage_one), 23)
            self.assertEqual(max(int(job["gpu_slot"]) for job in stage_three), 23)
            command = sweep.build_coverage_worker_command(
                config_path, root, stage_one[0], dry_run=True, resume=True
            )
            self.assertIn("--cpunodebind=0", command)
            self.assertIn("--preferred=0", command)
            self.assertNotIn("--membind=0", command)
            self.assertIn("scripts.rq3.news_first_vol_coverage_sweep", command)
            self.assertIn("--worker-dry-run", command)

    def test_wgan_run_contract_validates_dual_lr_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metrics = run_dir / "metrics"
            metrics.mkdir(parents=True)
            job = {
                "job_id": "wgan-cell",
                "model_family": "wgan",
                "capacity_profile": "micro",
                "capacity_profile_sha256": "a" * 64,
                "lr_profile": "lr_5e_07",
                "lr_profile_sha256": "b" * 64,
                "initial_learning_rate": 5.0e-7,
                "scheduler_min_lr": 5.0e-8,
                "seed": 42,
            }
            (metrics / "training_metrics.json").write_text(
                json.dumps(
                    [
                        {"epoch": 0, "g_lr": 5.0e-7, "d_lr": 5.0e-7},
                        {"epoch": 1, "g_lr": 2.5e-7, "d_lr": 2.5e-7},
                    ]
                ),
                encoding="utf-8",
            )
            (metrics / "best_learned_checkpoint.json").write_text(
                json.dumps(
                    {
                        "capacity_profile": "micro",
                        "capacity_profile_sha256": "a" * 64,
                        "lr_profile": "lr_5e_07",
                        "lr_profile_sha256": "b" * 64,
                        "initial_learning_rate": 5.0e-7,
                        "scheduler_min_lr": 5.0e-8,
                    }
                ),
                encoding="utf-8",
            )
            (metrics / "training_resolved_config.yaml").write_text(
                "seed: 42\n", encoding="utf-8"
            )
            trace = sweep._validate_run_contract(job, run_dir, dry_run=False)
            self.assertEqual(trace[-1]["g_lr"], 2.5e-7)
            self.assertEqual(trace[-1]["d_lr"], 2.5e-7)
            rows = json.loads((metrics / "training_metrics.json").read_text())
            rows[-1]["g_lr"] = 1.0e-8
            (metrics / "training_metrics.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "floor contract"):
                sweep._validate_run_contract(job, run_dir, dry_run=False)

    def test_prepare_hashes_every_job_and_fails_closed_after_config_tamper(self):
        with self._fixture() as (config_path, root):
            sweep.prepare_coverage_sweep_experiment(config_path, root)
            registry = sweep._load_registry(root)
            self.assertEqual(registry["expected_total_jobs"], 456)
            self.assertEqual(len(registry["jobs"]), 456)
            self.assertEqual(
                set(registry["reference_roots"]), {"fixed_lr_capacity", "local_lr"}
            )
            self.assertTrue((root / "coverage_stage_status.json").is_file())
            self.assertTrue((root / "stage_manifest.csv").is_file())
            self.assertTrue((root / "capacity_lr_profile_manifest.csv").is_file())
            sweep._validate_registry(root, config_path)

            first = registry["jobs"][0]
            job_config = Path(first["training_config_path"])
            job_config.write_text(
                job_config.read_text(encoding="utf-8") + "# tamper\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Training config hash mismatch"):
                sweep._validate_registry(root, config_path)

    def test_stage_prerequisite_requires_hash_verified_qa(self):
        with self._fixture() as (config_path, root):
            sweep.prepare_coverage_sweep_experiment(config_path, root)
            with self.assertRaisesRegex(RuntimeError, "Cannot start"):
                sweep._assert_stage_prerequisite(root, 2, dry_run=True)

            stage_id = sweep.STAGE_IDS[0]
            jobs = [
                job
                for job in sweep._load_registry(root)["jobs"]
                if job["stage_id"] == stage_id
            ]
            for job in jobs:
                status_path = sweep._job_status_path(root, job["job_id"])
                status = sweep._read_json(status_path)
                status.update({"status": "dry_run_passed", "artifacts": []})
                sweep._write_json(status_path, status)
            qa_path = sweep.run_stage_qa(root, stage_id, dry_run=True)
            qa = sweep._read_json(qa_path)
            sweep._mark_stage(
                root,
                stage_id,
                "dry_run_passed",
                qa_path=str(qa_path),
                qa_sha256=qa["qa_sha256"],
                completed_jobs=72,
            )
            sweep._assert_stage_prerequisite(root, 2, dry_run=True)
            qa["observed_jobs"] = 71
            sweep._write_json(qa_path, qa)
            with self.assertRaisesRegex(RuntimeError, "QA hash mismatch"):
                sweep._assert_stage_prerequisite(root, 2, dry_run=True)

    def test_dry_run_launcher_advances_all_stages_in_order(self):
        with self._fixture() as (config_path, root):
            observed_stages = []

            def fake_wave(experiment_root, _config, _wave, jobs, **_kwargs):
                if jobs:
                    observed_stages.append(jobs[0]["stage_id"])
                for job in jobs:
                    status_path = sweep._job_status_path(experiment_root, job["job_id"])
                    status = sweep._read_json(status_path)
                    status.update(
                        {
                            "status": "dry_run_passed",
                            "attempt": int(status.get("attempt", 0)) + 1,
                            "dry_run": True,
                            "artifacts": [],
                        }
                    )
                    sweep._write_json(status_path, status)

            with patch.object(sweep, "_run_wave", side_effect=fake_wave):
                sweep.launch_coverage_sweep_experiment(config_path, root, dry_run=True)
            status = sweep._read_json(root / "coverage_stage_status.json")
            self.assertEqual(status["status"], "dry_run_passed")
            self.assertEqual(
                [status["stages"][stage]["status"] for stage in sweep.STAGE_IDS],
                ["dry_run_passed"] * 4,
            )
            first_occurrence = []
            for stage in observed_stages:
                if not first_occurrence or first_occurrence[-1] != stage:
                    first_occurrence.append(stage)
            self.assertEqual(first_occurrence, list(sweep.STAGE_IDS))


if __name__ == "__main__":
    unittest.main()
