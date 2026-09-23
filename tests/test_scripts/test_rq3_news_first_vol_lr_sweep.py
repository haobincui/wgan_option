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
from scripts.rq3.news_first_vol_lr_sweep import (  # noqa: E402
    FROZEN_LEARNING_RATES,
    FROZEN_SCHEDULER_MIN_LRS,
    LR_PROFILE_IDS,
    _append_lr_stage_jobs,
    _job_status_path,
    _lr_profile_sha256,
    _read_json,
    _resolved_config,
    _sha256_file,
    _validate_lr_job_lineage,
    _write_json,
    build_lr_worker_command,
    launch_lr_sweep_experiment,
    prepare_lr_sweep_experiment,
)


class TestNewsFirstVolLearningRateSweep(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        summary_rows = []
        for tolerance in (5, 10, 15, 30):
            directory = self.dataset / f"tolerance_{tolerance:02d}m"
            directory.mkdir()
            (directory / "merged_vol.xlsx").write_bytes(
                f"workbook-{tolerance}".encode()
            )
            (directory / "surface_support_audit.csv.gz").write_bytes(
                f"support-{tolerance}".encode()
            )
            summary_rows.append({"tolerance_minutes": tolerance, "status": "pass"})
        with (self.dataset / "dataset_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("tolerance_minutes", "status")
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        payload = yaml.safe_load(
            (ROOT / "configs/rq3/news_first_vol_lr_sweep.yaml").read_text(
                encoding="utf-8"
            )
        )
        config = payload["news_first_vol_training"]
        config["datasets"]["root"] = str(self.dataset)
        config["runtime"]["python_executable"] = sys.executable
        self.config_path = self.root / "lr.yaml"
        self.config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
        self.experiment = self.root / "experiment"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _prepare(self) -> Path:
        def fake_split_manifest(resolved, experiment_root):
            del resolved
            path = experiment_root / "split_manifest.csv"
            path.write_text(
                "tolerance_minutes,status\n5,pass\n10,pass\n15,pass\n30,pass\n",
                encoding="utf-8",
            )
            (experiment_root / "text_ablation_manifest.csv").write_text(
                "mode,status\ncurrent_only,pass\nreal_text,pass\n",
                encoding="utf-8",
            )
            return path

        with patch(
            "scripts.rq3.news_first_vol_lr_sweep._build_split_manifest",
            side_effect=fake_split_manifest,
        ):
            return prepare_lr_sweep_experiment(self.config_path, self.experiment)

    @staticmethod
    def _complete_wave(root, config_path, wave, jobs, **kwargs):
        del config_path, wave, kwargs
        for job in jobs:
            artifact = root / "fake_artifacts" / f"{job['job_id']}.bin"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(str(job["job_id"]).encode())
            _write_json(
                _job_status_path(root, str(job["job_id"])),
                {
                    "job_id": job["job_id"],
                    "status": "completed",
                    "attempt": 1,
                    "config_sha256": job["config_sha256"],
                    "run_dir": str(root / "runs" / str(job["job_id"])),
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

    @staticmethod
    def _render_report(root):
        output = Path(root) / "report" / "lr_sweep_report.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("q3-only", encoding="utf-8")
        return output

    def test_prepare_materializes_exact_20_jobs_in_five_four_job_waves(self):
        root = self._prepare()
        registry = _read_json(root / "registry/jobs.json")
        self.assertEqual(registry["experiment_kind"], "learning_rate_sweep")
        jobs = registry["jobs"]
        self.assertEqual(len(jobs), 20)
        self.assertEqual(
            [sum(int(job["wave"]) == wave for job in jobs) for wave in range(1, 6)],
            [4, 4, 4, 4, 4],
        )
        self.assertEqual({job["lr_profile"] for job in jobs}, set(LR_PROFILE_IDS))
        self.assertEqual({job["tolerance_minutes"] for job in jobs}, {5, 30})
        self.assertEqual(
            {job["text_ablation_mode"] for job in jobs},
            {"current_only", "real_text"},
        )
        self.assertEqual({job["capacity_profile"] for job in jobs}, {"large"})
        self.assertEqual({job["lr_stage"] for job in jobs}, {"lr_screen"})
        for wave in range(1, 6):
            assignments = {
                (int(job["gpu_id"]), int(job["gpu_slot"]))
                for job in jobs
                if int(job["wave"]) == wave
            }
            self.assertEqual(assignments, {(0, 0), (0, 1), (1, 0), (1, 1)})
        for job in jobs:
            value = FROZEN_LEARNING_RATES[job["lr_profile"]]
            self.assertEqual(job["initial_learning_rate"], value)
            self.assertEqual(
                job["scheduler_min_lr"],
                FROZEN_SCHEDULER_MIN_LRS[job["lr_profile"]],
            )
            self.assertEqual(job["lr_trace"], [value])
            self.assertEqual(
                job["lr_profile_sha256"], _lr_profile_sha256(job["lr_profile"])
            )
            self.assertIn(f"/{job['lr_profile']}/", job["output_root"])
            training = yaml.safe_load(
                Path(job["training_config_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(training["news_first_lr_profile"], job["lr_profile"])
            self.assertEqual(
                training["news_first_lr_profile_sha256"],
                job["lr_profile_sha256"],
            )
            self.assertEqual(training["learning_rate"], value)
            self.assertEqual(
                training["reduce_lr_min_lr"],
                FROZEN_SCHEDULER_MIN_LRS[job["lr_profile"]],
            )
            self.assertEqual(training["early_stopping_min_epochs"], 30)
            self.assertEqual(training["early_stopping_patience"], 12)

    def test_reuse_requires_same_config_and_complete_screen_registry(self):
        root = self._prepare()
        reused = prepare_lr_sweep_experiment(
            self.config_path, root, reuse=True
        )
        self.assertEqual(reused, root)

        registry_path = root / "registry/jobs.json"
        registry = _read_json(registry_path)
        registry["jobs"].pop()
        _write_json(registry_path, registry)
        with self.assertRaisesRegex(ValueError, "missing one or more frozen screen"):
            prepare_lr_sweep_experiment(self.config_path, root, reuse=True)

    def test_frozen_profile_order_value_and_selection_rule_fail_closed(self):
        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        sweep = payload["news_first_vol_training"]["lr_sweep"]
        sweep["profiles"]["lr_1e_06"] = 2.0e-6
        changed = self.root / "changed.yaml"
        changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen to"):
            _resolved_config(changed)

        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        payload["news_first_vol_training"]["lr_sweep"]["selection"][
            "selection_rule"
        ] = "best_only"
        changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "selection_rule"):
            _resolved_config(changed)

    def test_worker_command_uses_independent_lr_cli(self):
        root = self._prepare()
        job = _read_json(root / "registry/jobs.json")["jobs"][0]
        command = build_lr_worker_command(
            self.config_path, root, job, dry_run=True, resume=True
        )
        self.assertIn("train-news-first-vol-lr-sweep", command)
        self.assertIn(job["job_id"], command)
        self.assertIn("--worker-dry-run", command)
        self.assertIn("--resume", command)

    def test_job_config_or_job_spec_tamper_fails_closed(self):
        root = self._prepare()
        job = _read_json(root / "registry/jobs.json")["jobs"][0]
        training_path = Path(job["training_config_path"])
        payload = yaml.safe_load(training_path.read_text(encoding="utf-8"))
        payload["learning_rate"] *= 10
        training_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Training config hash mismatch"):
            _validate_lr_job_lineage(root, job)

        # Restore the config, then prove registry mutations are independently caught.
        payload["learning_rate"] /= 10
        training_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        job["config_sha256"] = _sha256_file(training_path)
        job["gpu_slot"] = 99
        with self.assertRaisesRegex(ValueError, "job-spec hash mismatch"):
            _validate_lr_job_lineage(root, job)

    def test_gate_pass_appends_only_four_confirm_jobs_and_stays_q3_only(self):
        root = self._prepare()

        def select(experiment_root, stage):
            del experiment_root
            return {
                "gate_passed": True,
                "selected_lr_profile": "lr_1e_05",
                "lr_profile_sha256": _lr_profile_sha256("lr_1e_05"),
                "initial_learning_rate": 1.0e-5,
                "scheduler_min_lr": 1.0e-6,
                "selection_panel": "common_validation_05m",
                "q4_used_for_selection": False,
                "stage_echo": stage,
            }

        with (
            patch(
                "scripts.rq3.news_first_vol_lr_sweep._run_lr_wave",
                side_effect=self._complete_wave,
            ),
            patch(
                "scripts.rq3.news_first_vol_lr_report.render_lr_sweep_report",
                side_effect=self._render_report,
            ),
        ):
            result = launch_lr_sweep_experiment(
                self.config_path, root, selection_hook=select
            )
        self.assertEqual(result, root)
        jobs = _read_json(root / "registry/jobs.json")["jobs"]
        self.assertEqual(len(jobs), 24)
        self.assertEqual(max(int(job["wave"]) for job in jobs), 6)
        confirm = [job for job in jobs if job["lr_stage"] == "lr_confirm"]
        self.assertEqual(len(confirm), 4)
        self.assertEqual({job["lr_profile"] for job in confirm}, {"lr_1e_05"})
        self.assertEqual({job["tolerance_minutes"] for job in confirm}, {10, 15})
        self.assertEqual(
            _read_json(root / "registry/experiment_status.json")["status"],
            "completed_q3_only",
        )
        self.assertFalse((root / "analysis" / "q4_pair_metrics.csv.gz").exists())
        with (root / "output_hashes.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            roles = {row["artifact_role"] for row in csv.DictReader(handle)}
        self.assertIn("experiment:report/lr_sweep_report.html", roles)

    def test_gate_failure_stops_at_20_jobs_without_q4(self):
        root = self._prepare()

        def select(experiment_root, stage):
            del experiment_root, stage
            return {
                "gate_passed": False,
                "selection_panel": "common_validation_05m",
                "q4_used_for_selection": False,
            }

        with (
            patch(
                "scripts.rq3.news_first_vol_lr_sweep._run_lr_wave",
                side_effect=self._complete_wave,
            ),
            patch(
                "scripts.rq3.news_first_vol_lr_report.render_lr_sweep_report",
                side_effect=self._render_report,
            ),
        ):
            launch_lr_sweep_experiment(self.config_path, root, selection_hook=select)
        self.assertEqual(len(_read_json(root / "registry/jobs.json")["jobs"]), 20)
        self.assertEqual(
            _read_json(root / "registry/experiment_status.json")["status"],
            "completed_no_learned_lr",
        )
        self.assertFalse((root / "registry/selections/lr_confirm.json").exists())

    def test_selection_must_explicitly_disavow_q4(self):
        root = self._prepare()

        def select(experiment_root, stage):
            del experiment_root, stage
            return {
                "gate_passed": True,
                "selected_lr_profile": "lr_1e_05",
                "selection_panel": "common_validation_05m",
            }

        with (
            patch(
                "scripts.rq3.news_first_vol_lr_sweep._run_lr_wave",
                side_effect=self._complete_wave,
            ),
            self.assertRaisesRegex(ValueError, "q4_used_for_selection=false"),
        ):
            launch_lr_sweep_experiment(self.config_path, root, selection_hook=select)

    def test_failed_gate_cannot_retain_a_selected_lr(self):
        root = self._prepare()

        def select(experiment_root, stage):
            del experiment_root, stage
            return {
                "gate_passed": False,
                "selected_lr_profile": "lr_1e_05",
                "selection_panel": "common_validation_05m",
                "q4_used_for_selection": False,
            }

        with (
            patch(
                "scripts.rq3.news_first_vol_lr_sweep._run_lr_wave",
                side_effect=self._complete_wave,
            ),
            self.assertRaisesRegex(ValueError, "must not retain"),
        ):
            launch_lr_sweep_experiment(self.config_path, root, selection_hook=select)

    def test_terminal_resume_rejects_cumulative_selection_tamper(self):
        root = self._prepare()

        def select(experiment_root, stage):
            del experiment_root, stage
            return {
                "gate_passed": False,
                "selection_panel": "common_validation_05m",
                "q4_used_for_selection": False,
            }

        with (
            patch(
                "scripts.rq3.news_first_vol_lr_sweep._run_lr_wave",
                side_effect=self._complete_wave,
            ),
            patch(
                "scripts.rq3.news_first_vol_lr_report.render_lr_sweep_report",
                side_effect=self._render_report,
            ),
        ):
            launch_lr_sweep_experiment(self.config_path, root, selection_hook=select)
        cumulative_path = root / "lr_selection.json"
        cumulative = _read_json(cumulative_path)
        cumulative["stages"]["lr_screen"]["gate_passed"] = True
        _write_json(cumulative_path, cumulative)
        with self.assertRaisesRegex(ValueError, "disagrees with frozen snapshot"):
            launch_lr_sweep_experiment(
                self.config_path,
                root,
                resume=True,
                selection_hook=select,
            )

    def test_confirm_lineage_rejects_selection_snapshot_tamper(self):
        root = self._prepare()
        selection_path = root / "registry/selections/lr_screen.json"
        _write_json(
            selection_path,
            {
                "schema_version": 1,
                "stage": "lr_screen",
                "gate_passed": True,
                "selected_lr_profile": "lr_1e_05",
                "selection_panel": "common_validation_05m",
                "q4_used_for_selection": False,
            },
        )
        resolved = yaml.safe_load(
            (root / "resolved_config.yaml").read_text(encoding="utf-8")
        )["news_first_vol_training"]
        _append_lr_stage_jobs(
            root,
            resolved,
            stage="lr_confirm",
            profiles=("lr_1e_05",),
            selection_sha256=_sha256_file(selection_path),
        )
        job = next(
            job
            for job in _read_json(root / "registry/jobs.json")["jobs"]
            if job["lr_stage"] == "lr_confirm"
        )
        _validate_lr_job_lineage(root, job)
        payload = _read_json(selection_path)
        payload["selected_lr_profile"] = "lr_3e_05"
        _write_json(selection_path, payload)
        with self.assertRaisesRegex(ValueError, "selection hash mismatch"):
            _validate_lr_job_lineage(root, job)

    def test_cli_dispatches_without_changing_older_commands(self):
        expected = self.root / "cli-result"
        with patch(
            "scripts.rq3.news_first_vol_lr_sweep.run_news_first_vol_lr_sweep",
            return_value=expected,
        ) as run:
            result = rq3_main(
                [
                    "train-news-first-vol-lr-sweep",
                    "prepare",
                    "--config",
                    str(self.config_path),
                    "--output-dir",
                    str(expected),
                    "--reuse",
                ]
            )
        self.assertEqual(result, expected)
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["action"], "prepare")
        self.assertTrue(run.call_args.kwargs["reuse"])


if __name__ == "__main__":
    unittest.main()
