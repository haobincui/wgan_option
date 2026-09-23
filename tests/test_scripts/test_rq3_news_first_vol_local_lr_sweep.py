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
from scripts.rq3.news_first_vol_local_lr_sweep import (  # noqa: E402
    EXPERIMENT_KIND,
    EXPERIMENT_STAGE,
    FROZEN_LEARNING_RATES,
    FROZEN_SCHEDULER_MIN_LRS,
    FROZEN_SEEDS,
    LR_PROFILE_IDS,
    _job_status_path,
    _lr_profile_sha256,
    _lr_seed_profile_sha256,
    _read_json,
    _resolved_config,
    _validate_job_lineage,
    _write_json,
    build_local_lr_worker_command,
    launch_local_lr_experiment,
    prepare_local_lr_experiment,
)


class TestLocalLearningRateSeedSweep(unittest.TestCase):
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
            (ROOT / "configs/rq3/news_first_vol_lr_seed_sweep.yaml").read_text(
                encoding="utf-8"
            )
        )
        payload["news_first_vol_training"]["datasets"]["root"] = str(self.dataset)
        payload["news_first_vol_training"]["runtime"]["python_executable"] = (
            sys.executable
        )
        self.config_path = self.root / "local_lr.yaml"
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
            "scripts.rq3.news_first_vol_local_lr_sweep._build_split_manifest",
            side_effect=fake_split_manifest,
        ):
            return prepare_local_lr_experiment(self.config_path, self.experiment)

    def test_prepare_materializes_exact_60_job_matrix_in_15_waves(self):
        root = self._prepare()
        registry = _read_json(root / "registry/jobs.json")
        self.assertEqual(registry["experiment_kind"], EXPERIMENT_KIND)
        jobs = registry["jobs"]
        self.assertEqual(len(jobs), 60)
        self.assertEqual(len({job["job_id"] for job in jobs}), 60)
        self.assertEqual({job["lr_profile"] for job in jobs}, set(LR_PROFILE_IDS))
        self.assertEqual({int(job["seed"]) for job in jobs}, set(FROZEN_SEEDS))
        self.assertEqual({int(job["tolerance_minutes"]) for job in jobs}, {5, 30})
        self.assertEqual(
            {job["text_ablation_mode"] for job in jobs},
            {"current_only", "real_text"},
        )
        self.assertEqual({job["experiment_stage"] for job in jobs}, {EXPERIMENT_STAGE})
        self.assertEqual(
            [sum(int(job["wave"]) == wave for job in jobs) for wave in range(1, 16)],
            [4] * 15,
        )
        for wave in range(1, 16):
            group = [job for job in jobs if int(job["wave"]) == wave]
            self.assertEqual(len({job["lr_profile"] for job in group}), 1)
            self.assertEqual(len({int(job["seed"]) for job in group}), 1)
            self.assertEqual(
                {(int(job["gpu_id"]), int(job["gpu_slot"])) for job in group},
                {(0, 0), (0, 1), (1, 0), (1, 1)},
            )

    def test_seed_and_lr_have_separate_hash_and_output_lineage(self):
        root = self._prepare()
        jobs = _read_json(root / "registry/jobs.json")["jobs"]
        lr_hashes = {}
        seed_hashes = set()
        for job in jobs:
            profile = job["lr_profile"]
            seed = int(job["seed"])
            lr_hashes.setdefault(profile, set()).add(job["lr_profile_sha256"])
            seed_hashes.add(job["lr_seed_profile_sha256"])
            self.assertEqual(
                job["initial_learning_rate"], FROZEN_LEARNING_RATES[profile]
            )
            self.assertEqual(job["scheduler_min_lr"], FROZEN_SCHEDULER_MIN_LRS[profile])
            self.assertEqual(job["lr_profile_sha256"], _lr_profile_sha256(profile))
            self.assertEqual(
                job["lr_seed_profile_sha256"],
                _lr_seed_profile_sha256(profile, seed),
            )
            self.assertIn(f"/{profile}/seed_{seed:03d}/", job["output_root"])
            payload = yaml.safe_load(
                Path(job["training_config_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(int(payload["seed"]), seed)
            self.assertEqual(payload["news_first_lr_profile"], profile)
            self.assertEqual(payload["learning_rate"], FROZEN_LEARNING_RATES[profile])
        self.assertTrue(all(len(values) == 1 for values in lr_hashes.values()))
        self.assertEqual(len(seed_hashes), 15)

    def test_profile_seed_order_and_values_fail_closed(self):
        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        sweep = payload["news_first_vol_training"]["local_lr_seed_sweep"]
        sweep["seeds"] = [42, 404, 202]
        changed = self.root / "changed.yaml"
        changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "seeds are frozen"):
            _resolved_config(changed)

        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        payload["news_first_vol_training"]["local_lr_seed_sweep"]["profiles"][
            "lr_1e_06"
        ] = 1.1e-6
        changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen"):
            _resolved_config(changed)

    def test_seed_or_job_tamper_fails_closed(self):
        root = self._prepare()
        job = _read_json(root / "registry/jobs.json")["jobs"][0]
        job["seed"] = 202
        with self.assertRaisesRegex(ValueError, "job-spec hash mismatch"):
            _validate_job_lineage(root, job)

        original = _read_json(root / "registry/jobs.json")["jobs"][0]
        config_path = Path(original["training_config_path"])
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["seed"] = 202
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Training config hash mismatch"):
            _validate_job_lineage(root, original)

    def test_reuse_is_idempotent_and_requires_complete_matrix(self):
        root = self._prepare()
        self.assertEqual(
            prepare_local_lr_experiment(self.config_path, root, reuse=True), root
        )
        registry_path = root / "registry/jobs.json"
        registry = _read_json(registry_path)
        registry["jobs"].pop()
        _write_json(registry_path, registry)
        with self.assertRaisesRegex(ValueError, "exactly the frozen 60-job matrix"):
            prepare_local_lr_experiment(self.config_path, root, reuse=True)

    def test_worker_command_uses_independent_cli_and_unique_job(self):
        root = self._prepare()
        job = _read_json(root / "registry/jobs.json")["jobs"][0]
        command = build_local_lr_worker_command(
            self.config_path, root, job, dry_run=True, resume=True
        )
        self.assertIn("train-news-first-vol-local-lr-seed-sweep", command)
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
                "scripts.rq3.news_first_vol_local_lr_sweep._run_wave",
                side_effect=fake_wave,
            ),
            patch("scripts.rq3.news_first_vol_local_lr_sweep._write_resource_summary"),
        ):
            launch_local_lr_experiment(self.config_path, root, dry_run=True)
        self.assertEqual(seen, [(wave, 4) for wave in range(1, 16)])
        status = _read_json(root / "registry/experiment_status.json")
        self.assertEqual(status["status"], "dry_run_passed")
        self.assertFalse(status["q4_predictions_generated"])
        self.assertFalse(status["q4_evaluated"])

    def test_cli_dispatches_new_command_without_replacing_old_one(self):
        expected = self.root / "cli-result"
        with patch(
            "scripts.rq3.news_first_vol_local_lr_sweep.run_news_first_vol_local_lr_sweep",
            return_value=expected,
        ) as run:
            result = rq3_main(
                [
                    "train-news-first-vol-local-lr-seed-sweep",
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
