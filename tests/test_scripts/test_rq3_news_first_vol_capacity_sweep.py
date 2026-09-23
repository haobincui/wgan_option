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
from scripts.rq3.news_first_vol_training import (  # noqa: E402
    CAPACITY_PROFILE_NAMES,
    CAPACITY_PROFILE_SHAPE_FIELDS,
    FROZEN_CAPACITY_PROFILES,
    _append_capacity_stage_jobs,
    _capacity_profile_sha256,
    _capacity_selection_snapshot_path,
    _job_status_path,
    _read_json,
    _resolved_config,
    _sha256_file,
    _validate_capacity_job_lineage,
    _write_json,
    build_worker_command,
    launch_capacity_experiment,
    prepare_capacity_experiment,
    run_capacity_postprocess,
)


class TestNewsFirstVolCapacitySweep(unittest.TestCase):
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
                handle,
                fieldnames=("tolerance_minutes", "status"),
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        payload = yaml.safe_load(
            (ROOT / "configs/rq3/news_first_vol_capacity_sweep.yaml").read_text(
                encoding="utf-8"
            )
        )
        config = payload["news_first_vol_training"]
        config["datasets"]["root"] = str(self.dataset)
        config["runtime"]["python_executable"] = sys.executable
        self.config_path = self.root / "capacity.yaml"
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
            return path

        with patch(
            "scripts.rq3.news_first_vol_training._build_split_manifest",
            side_effect=fake_split_manifest,
        ):
            return prepare_capacity_experiment(
                self.config_path,
                self.experiment,
            )

    @staticmethod
    def _complete_wave(experiment_root, config_path, wave, jobs, **kwargs):
        del config_path, wave, kwargs
        for job in jobs:
            artifact = experiment_root / "fake_artifacts" / f"{job['job_id']}.bin"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(str(job["job_id"]).encode())
            _write_json(
                _job_status_path(experiment_root, str(job["job_id"])),
                {
                    "job_id": job["job_id"],
                    "status": "completed",
                    "attempt": 1,
                    "config_sha256": job["config_sha256"],
                    "run_dir": str(experiment_root / "runs" / str(job["job_id"])),
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

    def test_prepare_materializes_exact_24_job_screen_and_profile_manifest(self):
        root = self._prepare()
        jobs = _read_json(root / "registry/jobs.json")["jobs"]
        self.assertEqual(len(jobs), 24)
        self.assertEqual(
            [sum(int(job["wave"]) == wave for job in jobs) for wave in range(1, 7)],
            [4, 4, 4, 4, 4, 4],
        )
        self.assertEqual({job["model_family"] for job in jobs}, {"regression"})
        self.assertEqual(
            {job["capacity_profile"] for job in jobs}, set(CAPACITY_PROFILE_NAMES)
        )
        self.assertEqual(
            {job["text_ablation_mode"] for job in jobs},
            {"current_only", "real_text"},
        )
        self.assertEqual({job["tolerance_minutes"] for job in jobs}, {5, 30})
        self.assertEqual({job["capacity_stage"] for job in jobs}, {"regression_screen"})

        with (root / "capacity_profile_manifest.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            profiles = list(csv.DictReader(handle))
        self.assertEqual(
            [row["capacity_profile"] for row in profiles],
            list(CAPACITY_PROFILE_NAMES),
        )
        for row in profiles:
            name = row["capacity_profile"]
            expected = FROZEN_CAPACITY_PROFILES[name]
            self.assertEqual(
                row["capacity_profile_sha256"],
                _capacity_profile_sha256(name, expected),
            )
            for field in CAPACITY_PROFILE_SHAPE_FIELDS:
                self.assertEqual(int(row[field]), expected[field])

        for job in jobs:
            training = yaml.safe_load(
                Path(job["training_config_path"]).read_text(encoding="utf-8")
            )
            profile = FROZEN_CAPACITY_PROFILES[job["capacity_profile"]]
            for field in CAPACITY_PROFILE_SHAPE_FIELDS:
                self.assertEqual(int(training[field]), profile[field])
            self.assertEqual(
                training["news_first_capacity_profile"], job["capacity_profile"]
            )
            self.assertEqual(
                training["news_first_capacity_profile_sha256"],
                job["capacity_profile_sha256"],
            )
            self.assertEqual(training["reduce_lr_patience"], 3)
            self.assertEqual(training["early_stopping_min_epochs"], 16)
            self.assertEqual(training["early_stopping_patience"], 12)

    def test_profile_shape_mutation_fails_closed(self):
        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        payload["news_first_vol_training"]["capacity_sweep"]["profiles"]["micro"][
            "gen_hidden_dim"
        ] = 33
        changed = self.root / "changed.yaml"
        changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from the frozen"):
            _resolved_config(changed)

    def test_worker_command_uses_capacity_cli(self):
        root = self._prepare()
        job = _read_json(root / "registry/jobs.json")["jobs"][0]
        command = build_worker_command(
            self.config_path,
            root,
            job,
            dry_run=True,
        )
        self.assertIn("train-news-first-vol-capacity-sweep", command)
        self.assertIn(job["job_id"], command)

    def test_downstream_job_fails_closed_if_frozen_selection_is_tampered(self):
        root = self._prepare()
        selection_path = _capacity_selection_snapshot_path(root, "regression_screen")
        _write_json(
            selection_path,
            {
                "schema_version": 1,
                "stage": "regression_screen",
                "selected_profiles": ["micro", "tiny"],
            },
        )
        resolved = yaml.safe_load(
            (root / "resolved_config.yaml").read_text(encoding="utf-8")
        )["news_first_vol_training"]
        _append_capacity_stage_jobs(
            root,
            resolved,
            stage="regression_confirm",
            profiles=("micro", "tiny"),
            selection_sha256=_sha256_file(selection_path),
        )
        job = next(
            job
            for job in _read_json(root / "registry/jobs.json")["jobs"]
            if job["capacity_stage"] == "regression_confirm"
        )
        _validate_capacity_job_lineage(root, job)
        _write_json(
            selection_path,
            {
                "schema_version": 1,
                "stage": "regression_screen",
                "selected_profiles": ["small", "medium"],
            },
        )
        with self.assertRaisesRegex(ValueError, "selection hash mismatch"):
            _validate_capacity_job_lineage(root, job)

    def test_full_state_machine_appends_56_unique_jobs_in_15_waves(self):
        root = self._prepare()

        def select(experiment_root, stage):
            del experiment_root
            if stage == "regression_screen":
                return {
                    "gate_passed": True,
                    "selected_profiles": ["micro", "tiny"],
                    "winner_profile": "micro",
                }
            if stage == "regression_confirm":
                return {"gate_passed": True, "winner_profile": "micro"}
            if stage == "wgan_screen":
                return {"gate_passed": True, "winner_profile": "tiny"}
            if stage == "wgan_confirm":
                return {"gate_passed": True, "winner_profile": "tiny"}
            raise AssertionError(stage)

        with patch(
            "scripts.rq3.news_first_vol_training._run_wave",
            side_effect=self._complete_wave,
        ):
            result = launch_capacity_experiment(
                self.config_path,
                root,
                selection_hook=select,
                postprocess_hook=lambda value: value,
            )
        self.assertEqual(result, root)
        jobs = _read_json(root / "registry/jobs.json")["jobs"]
        self.assertEqual(len(jobs), 56)
        self.assertEqual(len({job["job_id"] for job in jobs}), 56)
        self.assertEqual(max(int(job["wave"]) for job in jobs), 15)
        self.assertLessEqual(
            max(sum(int(job["wave"]) == wave for job in jobs) for wave in range(1, 16)),
            4,
        )
        by_stage = {}
        for job in jobs:
            by_stage[job["capacity_stage"]] = by_stage.get(job["capacity_stage"], 0) + 1
        self.assertEqual(
            by_stage,
            {
                "regression_screen": 24,
                "regression_confirm": 16,
                "wgan_screen": 6,
                "wgan_confirm": 10,
            },
        )
        self.assertEqual(
            _read_json(root / "registry/experiment_status.json")["status"],
            "completed",
        )
        self.assertEqual(
            set(path.stem for path in (root / "registry/selections").glob("*.json")),
            {
                "regression_screen",
                "regression_confirm",
                "wgan_screen",
                "wgan_confirm",
            },
        )

    def test_regression_gate_failure_is_valid_terminal_state(self):
        root = self._prepare()

        def select(experiment_root, stage):
            del experiment_root
            if stage == "regression_screen":
                return {
                    "gate_passed": True,
                    "selected_profiles": ["micro", "tiny"],
                    "winner_profile": "micro",
                }
            return {"gate_passed": False, "winner_profile": ""}

        def render_stage_only(experiment_root):
            report = Path(experiment_root) / "report" / "capacity_report.html"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("stage-only", encoding="utf-8")
            return report

        with (
            patch(
                "scripts.rq3.news_first_vol_training._run_wave",
                side_effect=self._complete_wave,
            ),
            patch(
                "scripts.rq3.news_first_vol_capacity_report.render_capacity_report",
                side_effect=render_stage_only,
            ) as render,
            patch(
                "scripts.rq3.news_first_vol_capacity_analysis.run_final_q4_analysis"
            ) as q4,
        ):
            launch_capacity_experiment(
                self.config_path,
                root,
                selection_hook=select,
                postprocess_hook=lambda value: value,
            )
        jobs = _read_json(root / "registry/jobs.json")["jobs"]
        self.assertEqual(len(jobs), 40)
        self.assertEqual({job["model_family"] for job in jobs}, {"regression"})
        self.assertEqual(
            _read_json(root / "registry/experiment_status.json")["status"],
            "completed_no_learned_capacity",
        )
        render.assert_called_once_with(root)
        q4.assert_not_called()
        with (root / "output_hashes.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            roles = {row["artifact_role"] for row in csv.DictReader(handle)}
        self.assertIn("experiment:report/capacity_report.html", roles)
        self.assertIn("experiment:registry/jobs.json", roles)
        self.assertIn("experiment:registry/experiment_status.json", roles)
        self.assertIn("experiment:registry/resolved_config.sha256", roles)
        self.assertTrue(
            any(
                role.startswith("experiment:registry/jobs/")
                and role.endswith(".status.json")
                for role in roles
            )
        )

    def test_regression_screen_gate_failure_renders_stage_only_without_q4(self):
        root = self._prepare()

        def select(experiment_root, stage):
            del experiment_root
            self.assertEqual(stage, "regression_screen")
            return {
                "gate_passed": False,
                "selected_profiles": [],
                "winner_profile": "",
                "ranked_top2_profiles": ["micro", "tiny"],
                "ranked_leader_profile": "micro",
            }

        def render_stage_only(experiment_root):
            report = Path(experiment_root) / "report" / "capacity_report.html"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("screen-stop", encoding="utf-8")
            return report

        with (
            patch(
                "scripts.rq3.news_first_vol_training._run_wave",
                side_effect=self._complete_wave,
            ),
            patch(
                "scripts.rq3.news_first_vol_capacity_report.render_capacity_report",
                side_effect=render_stage_only,
            ) as render,
            patch(
                "scripts.rq3.news_first_vol_capacity_analysis.run_final_q4_analysis"
            ) as q4,
        ):
            launch_capacity_experiment(
                self.config_path,
                root,
                selection_hook=select,
            )
        self.assertEqual(len(_read_json(root / "registry/jobs.json")["jobs"]), 24)
        self.assertEqual(
            _read_json(root / "registry/experiment_status.json")["status"],
            "completed_no_learned_capacity",
        )
        render.assert_called_once_with(root)
        q4.assert_not_called()

    def test_q4_postprocess_requires_every_conditionally_required_snapshot(self):
        root = self.root / "postprocess"
        (root / "registry" / "selections").mkdir(parents=True)
        regression_screen = {
            "schema_version": 1,
            "capacity_stage": "regression_screen",
            "gate_passed": True,
            "selected_profiles": ["micro", "tiny"],
            "winner_profile": "micro",
        }
        regression_confirm = {
            "schema_version": 1,
            "capacity_stage": "regression_confirm",
            "gate_passed": True,
            "selected_profiles": ["micro", "tiny"],
            "winner_profile": "micro",
        }
        screen_snapshot = {
            "stage": "regression_screen",
            **regression_screen,
        }
        screen_path = _capacity_selection_snapshot_path(root, "regression_screen")
        _write_json(screen_path, screen_snapshot)
        _write_json(
            root / "capacity_selection.json",
            {
                "schema_version": 1,
                "stages": {
                    "regression_screen": regression_screen,
                    "regression_confirm": regression_confirm,
                },
            },
        )
        _write_json(
            root / "capacity_stage_status.json",
            {
                "schema_version": 1,
                "current_stage": "regression_confirm",
                "stages": {
                    "regression_screen": {
                        "status": "selected",
                        "selection_sha256": _sha256_file(screen_path),
                    },
                    "regression_confirm": {
                        "status": "selected",
                        "selection_sha256": "missing-snapshot",
                    },
                },
            },
        )
        with self.assertRaisesRegex(
            ValueError,
            "Missing required frozen capacity selection snapshot: regression_confirm",
        ):
            run_capacity_postprocess(root)

        confirm_path = _capacity_selection_snapshot_path(root, "regression_confirm")
        _write_json(
            confirm_path,
            {"stage": "regression_confirm", **regression_confirm},
        )
        wgan_screen = {
            "schema_version": 1,
            "capacity_stage": "wgan_screen",
            "gate_passed": True,
            "selected_profiles": ["micro", "legacy"],
            "winner_profile": "micro",
        }
        wgan_screen_path = _capacity_selection_snapshot_path(root, "wgan_screen")
        _write_json(
            wgan_screen_path,
            {"stage": "wgan_screen", **wgan_screen},
        )
        _write_json(
            root / "capacity_selection.json",
            {
                "schema_version": 1,
                "stages": {
                    "regression_screen": regression_screen,
                    "regression_confirm": regression_confirm,
                    "wgan_screen": wgan_screen,
                },
            },
        )
        _write_json(
            root / "capacity_stage_status.json",
            {
                "schema_version": 1,
                "current_stage": "wgan_confirm",
                "stages": {
                    "regression_screen": {
                        "status": "selected",
                        "selection_sha256": _sha256_file(screen_path),
                    },
                    "regression_confirm": {
                        "status": "selected",
                        "selection_sha256": _sha256_file(confirm_path),
                    },
                    "wgan_screen": {
                        "status": "selected",
                        "selection_sha256": _sha256_file(wgan_screen_path),
                    },
                    "wgan_confirm": {
                        "status": "completed",
                        "selection_sha256": "missing-snapshot",
                    },
                },
            },
        )
        with self.assertRaisesRegex(
            ValueError,
            "Missing required frozen capacity selection snapshot: wgan_confirm",
        ):
            run_capacity_postprocess(root)

    def test_rq3_main_dispatches_capacity_command_lazily(self):
        expected = self.root / "result"
        with patch(
            "scripts.rq3.news_first_vol_training.run_news_first_vol_capacity_sweep",
            return_value=expected,
        ) as runner:
            result = rq3_main(
                [
                    "train-news-first-vol-capacity-sweep",
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
