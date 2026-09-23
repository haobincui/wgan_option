from __future__ import annotations

import csv
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
from scripts.rq3.news_first_vol_fixed_lr_capacity_sweep import (  # noqa: E402
    EXPERIMENT_KIND,
    EXPERIMENT_STAGE,
    FIXED_LEARNING_RATE,
    FIXED_LR_PROFILE,
    FIXED_SCHEDULER_MIN_LR,
    FROZEN_CAPACITY_PROFILES,
    FROZEN_SEEDS,
    _capacity_profile_sha256,
    _capacity_seed_profile_sha256,
    _job_status_path,
    _read_json,
    _resolved_config,
    _validate_job_lineage,
    _write_json,
    build_fixed_lr_capacity_worker_command,
    launch_fixed_lr_capacity_experiment,
    prepare_fixed_lr_capacity_experiment,
)
from wgan_option.config import load_config  # noqa: E402


class TestFixedLearningRateCapacitySeedSweep(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        summaries = []
        for tolerance in (5, 10, 15, 30):
            directory = self.dataset / f"tolerance_{tolerance:02d}m"
            directory.mkdir()
            (directory / "merged_vol.xlsx").write_bytes(
                f"workbook-{tolerance}".encode()
            )
            (directory / "surface_support_audit.csv.gz").write_bytes(
                f"support-{tolerance}".encode()
            )
            summaries.append({"tolerance_minutes": tolerance, "status": "pass"})
        with (self.dataset / "dataset_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("tolerance_minutes", "status"))
            writer.writeheader()
            writer.writerows(summaries)
        payload = yaml.safe_load(
            (
                ROOT / "configs/rq3/news_first_vol_fixed_lr_capacity_seed_sweep.yaml"
            ).read_text(encoding="utf-8")
        )
        config = payload["news_first_vol_training"]
        config["datasets"]["root"] = str(self.dataset)
        config["runtime"]["python_executable"] = sys.executable
        self.config_path = self.root / "fixed_lr_capacity.yaml"
        self.config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
        self.experiment = self.root / "experiment"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _prepare(self) -> Path:
        def fake_split_manifest(resolved, root):
            del resolved
            path = root / "split_manifest.csv"
            path.write_text(
                "tolerance_minutes,status\n5,pass\n10,pass\n15,pass\n30,pass\n",
                encoding="utf-8",
            )
            (root / "text_ablation_manifest.csv").write_text(
                "mode,status\ncurrent_only,pass\nreal_text,pass\n",
                encoding="utf-8",
            )
            return path

        with patch(
            "scripts.rq3.news_first_vol_fixed_lr_capacity_sweep._build_split_manifest",
            side_effect=fake_split_manifest,
        ):
            return prepare_fixed_lr_capacity_experiment(
                self.config_path, self.experiment
            )

    def test_prepare_materializes_72_jobs_in_18_four_job_waves(self):
        root = self._prepare()
        registry = _read_json(root / "registry/jobs.json")
        self.assertEqual(registry["experiment_kind"], EXPERIMENT_KIND)
        jobs = registry["jobs"]
        self.assertEqual(len(jobs), 72)
        self.assertEqual(len({job["job_id"] for job in jobs}), 72)
        self.assertEqual(
            {job["capacity_profile"] for job in jobs},
            set(FROZEN_CAPACITY_PROFILES),
        )
        self.assertEqual({int(job["seed"]) for job in jobs}, set(FROZEN_SEEDS))
        self.assertEqual({int(job["tolerance_minutes"]) for job in jobs}, {5, 30})
        self.assertEqual(
            {job["text_ablation_mode"] for job in jobs},
            {"current_only", "real_text"},
        )
        self.assertEqual({job["experiment_stage"] for job in jobs}, {EXPERIMENT_STAGE})
        self.assertEqual(
            [sum(int(job["wave"]) == wave for job in jobs) for wave in range(1, 19)],
            [4] * 18,
        )
        for wave in range(1, 19):
            group = [job for job in jobs if int(job["wave"]) == wave]
            self.assertEqual(len({job["capacity_profile"] for job in group}), 1)
            self.assertEqual(len({int(job["seed"]) for job in group}), 1)
            self.assertEqual(
                {(int(job["gpu_id"]), int(job["gpu_slot"])) for job in group},
                {(0, 0), (0, 1), (1, 0), (1, 1)},
            )
            for gpu_id in (0, 1):
                gpu_group = [job for job in group if int(job["gpu_id"]) == gpu_id]
                self.assertEqual(
                    {job["text_ablation_mode"] for job in gpu_group},
                    {"current_only", "real_text"},
                )
                self.assertEqual(
                    {int(job["tolerance_minutes"]) for job in gpu_group},
                    {5, 30},
                )

    def test_capacity_seed_hash_and_training_config_lineage(self):
        root = self._prepare()
        jobs = _read_json(root / "registry/jobs.json")["jobs"]
        combined_hashes = set()
        for job in jobs:
            profile = job["capacity_profile"]
            seed = int(job["seed"])
            combined_hashes.add(job["capacity_seed_profile_sha256"])
            self.assertEqual(
                job["capacity_profile_sha256"], _capacity_profile_sha256(profile)
            )
            self.assertEqual(
                job["capacity_seed_profile_sha256"],
                _capacity_seed_profile_sha256(profile, seed),
            )
            self.assertEqual(job["fixed_lr_profile"], FIXED_LR_PROFILE)
            self.assertEqual(job["initial_learning_rate"], FIXED_LEARNING_RATE)
            self.assertEqual(job["scheduler_min_lr"], FIXED_SCHEDULER_MIN_LR)
            self.assertIn(f"/{profile}/seed_{seed:03d}/", job["output_root"])
            payload = yaml.safe_load(
                Path(job["training_config_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(int(payload["seed"]), seed)
            self.assertEqual(payload["news_first_capacity_profile"], profile)
            self.assertEqual(payload["learning_rate"], FIXED_LEARNING_RATE)
        self.assertEqual(len(combined_hashes), 18)

    def test_profile_manifest_is_one_row_per_capacity_seed(self):
        root = self._prepare()
        with (root / "capacity_seed_profile_manifest.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 18)
        self.assertEqual(
            {(row["capacity_profile"], int(row["seed"])) for row in rows},
            {
                (profile, seed)
                for profile in FROZEN_CAPACITY_PROFILES
                for seed in FROZEN_SEEDS
            },
        )

    def test_generated_training_config_preserves_new_lineage_fields(self):
        root = self._prepare()
        job = _read_json(root / "registry/jobs.json")["jobs"][0]
        loaded = load_config(job["training_config_path"])
        self.assertEqual(loaded.news_first_capacity_profile, "micro")
        self.assertEqual(
            loaded.news_first_capacity_seed_profile_sha256,
            job["capacity_seed_profile_sha256"],
        )
        self.assertEqual(
            loaded.news_first_fixed_learning_rate_profile, FIXED_LR_PROFILE
        )
        self.assertEqual(loaded.learning_rate, FIXED_LEARNING_RATE)

    def test_selection_lane_and_frozen_lr_fail_closed(self):
        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        sweep = payload["news_first_vol_training"]["fixed_lr_capacity_seed_sweep"]
        sweep["analysis"]["primary_text_ablation_mode"] = "current_only"
        changed = self.root / "changed.yaml"
        changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "primary_text_ablation_mode"):
            _resolved_config(changed)

        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        payload["news_first_vol_training"]["fixed_lr_capacity_seed_sweep"][
            "initial_learning_rate"
        ] = 7.5e-7
        changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "initial LR"):
            _resolved_config(changed)

        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        payload["news_first_vol_training"]["models"]["regression"]["training"][
            "lambda_recon"
        ] = 9.0
        changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lambda_recon"):
            _resolved_config(changed)

    def test_tampering_with_job_or_config_fails_closed(self):
        root = self._prepare()
        job = _read_json(root / "registry/jobs.json")["jobs"][0]
        job["seed"] = 202
        with self.assertRaisesRegex(ValueError, "job-spec hash mismatch"):
            _validate_job_lineage(root, job)

        original = _read_json(root / "registry/jobs.json")["jobs"][0]
        config_path = Path(original["training_config_path"])
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["gen_hidden_dim"] += 1
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Training config hash mismatch"):
            _validate_job_lineage(root, original)

    def test_reuse_requires_complete_immutable_matrix(self):
        root = self._prepare()
        self.assertEqual(
            prepare_fixed_lr_capacity_experiment(self.config_path, root, reuse=True),
            root,
        )
        registry_path = root / "registry/jobs.json"
        registry = _read_json(registry_path)
        registry["jobs"].pop()
        _write_json(registry_path, registry)
        with self.assertRaisesRegex(ValueError, "exactly the frozen 72-job matrix"):
            prepare_fixed_lr_capacity_experiment(self.config_path, root, reuse=True)

    def test_worker_command_uses_independent_cli(self):
        root = self._prepare()
        job = _read_json(root / "registry/jobs.json")["jobs"][0]
        command = build_fixed_lr_capacity_worker_command(
            self.config_path, root, job, dry_run=True, resume=True
        )
        self.assertIn("train-news-first-vol-fixed-lr-capacity-seed-sweep", command)
        self.assertIn(job["job_id"], command)
        self.assertIn("--worker-dry-run", command)
        self.assertIn("--resume", command)

    def test_dry_run_executes_all_waves_without_postprocessing(self):
        root = self._prepare()
        seen = []

        def fake_wave(experiment_root, config_path, wave, jobs, **kwargs):
            del config_path
            seen.append((wave, len(jobs)))
            for job in jobs:
                _write_json(
                    _job_status_path(experiment_root, str(job["job_id"])),
                    {
                        "job_id": job["job_id"],
                        "status": "dry_run_passed",
                        "attempt": 1,
                        "config_sha256": job["config_sha256"],
                        "dry_run": True,
                    },
                )

        with (
            patch(
                "scripts.rq3.news_first_vol_fixed_lr_capacity_sweep._run_wave",
                side_effect=fake_wave,
            ),
            patch(
                "scripts.rq3.news_first_vol_fixed_lr_capacity_sweep._write_resource_summary"
            ),
        ):
            launch_fixed_lr_capacity_experiment(self.config_path, root, dry_run=True)
        self.assertEqual(seen, [(wave, 4) for wave in range(1, 19)])
        status = _read_json(root / "registry/experiment_status.json")
        self.assertEqual(status["status"], "dry_run_passed")
        self.assertFalse(status["q4_predictions_generated"])
        self.assertFalse(status["q4_evaluated"])

    def test_cli_dispatches_new_command_without_replacing_existing_commands(self):
        expected = self.root / "cli-result"
        with patch(
            "scripts.rq3.news_first_vol_fixed_lr_capacity_sweep."
            "run_news_first_vol_fixed_lr_capacity_sweep",
            return_value=expected,
        ) as run:
            result = rq3_main(
                [
                    "train-news-first-vol-fixed-lr-capacity-seed-sweep",
                    "prepare",
                    "--config",
                    str(self.config_path),
                    "--output-dir",
                    str(expected),
                    "--reuse",
                ]
            )
        self.assertEqual(result, expected)
        self.assertEqual(run.call_args.kwargs["action"], "prepare")
        self.assertTrue(run.call_args.kwargs["reuse"])


if __name__ == "__main__":
    unittest.main()
