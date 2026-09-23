from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import yaml

from scripts.rq3 import news_first_vol_current_input_sweep as experiment
from scripts.rq3 import main as rq3_main
from wgan_option.models.common import generator_current_input_fingerprint


CONFIG_PATH = Path("configs/rq3/news_first_vol_current_input_sweep.yaml")


class CurrentSupportMaskedSweepContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name) / "experiment"
        cls.resolved = experiment._resolved_config(CONFIG_PATH)
        cls.reference_rows = experiment._reference_rows(cls.resolved)
        experiment.prepare_current_input_experiment(CONFIG_PATH, cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _failed_postprocess_root(self, temporary_root: Path) -> Path:
        root = temporary_root / "experiment"
        shutil.copytree(self.root, root)
        registry = json.loads((root / "registry/jobs.json").read_text())
        for job in registry["jobs"]:
            status_path = experiment._job_status_path(root, job["job_id"])
            status = json.loads(status_path.read_text())
            status.update(
                {
                    "status": "completed",
                    "attempt": 2,
                    "dry_run": False,
                    "job_spec_sha256": job["job_spec_sha256"],
                    "config_sha256": job["config_sha256"],
                    "q4_evaluator_calls": 0,
                    "q4_predictions_generated": False,
                    "q4_evaluated": False,
                    "artifacts": [],
                }
            )
            experiment._write_json(status_path, status)
        experiment._mark_status(
            root,
            "failed",
            error="CurrentInputAnalysisError: precision mismatch",
        )
        experiment._refresh_exports(root)
        experiment.refresh_current_input_lineage(root)
        return root

    @staticmethod
    def _write_minimal_postprocess_artifacts(root: Path) -> tuple[Path, Path]:
        analysis_dir = root / "analysis/current_input_ablation"
        report_dir = root / "report"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        summary = {"schema_version": 1, "q4_evaluated": False}
        summary["analysis_sha256"] = experiment._payload_sha256(summary)
        summary_path = experiment._write_json(
            analysis_dir / "current_input_analysis_summary.json", summary
        )
        validation = {
            "schema_version": 1,
            "status": "pass",
            "analysis_sha256": summary["analysis_sha256"],
        }
        validation["validation_sha256"] = experiment._payload_sha256(validation)
        experiment._write_json(
            analysis_dir / "current_input_validation_summary.json", validation
        )
        report_path = report_dir / "current_input_ablation_q3_report.html"
        report_path.write_text("<html>valid</html>", encoding="utf-8")
        report_manifest = {
            "schema_version": 1,
            "analysis_sha256": summary["analysis_sha256"],
            "report_path": str(report_path),
            "report_sha256": experiment._sha256_file(report_path),
            "q4_predictions_generated": False,
            "q4_evaluated": False,
        }
        report_manifest["report_manifest_sha256"] = experiment._payload_sha256(
            report_manifest
        )
        experiment._write_json(
            report_dir / "current_input_ablation_report_manifest.json",
            report_manifest,
        )
        return summary_path, report_path

    def test_six_jobs_preserve_reference_gpu_lanes_in_four_plus_two_waves(self) -> None:
        specs = experiment._job_specs()
        self.assertEqual(len(specs), 6)
        self.assertEqual(
            {(row["seed"], row["tolerance_minutes"]) for row in specs},
            {(seed, tolerance) for seed in (42, 202, 404) for tolerance in (5, 30)},
        )
        self.assertEqual(
            [sum(row["wave"] == wave for row in specs) for wave in (1, 2)], [4, 2]
        )
        self.assertEqual({row["seed"] for row in specs if row["wave"] == 1}, {42, 404})
        self.assertEqual({row["seed"] for row in specs if row["wave"] == 2}, {202})
        for row in specs:
            self.assertEqual(
                row["gpu_index"], 0 if row["tolerance_minutes"] == 5 else 1
            )

    def test_completed_resume_rejects_tampered_report(self) -> None:
        experiment_status_path = self.root / "registry/experiment_status.json"
        original_status = experiment_status_path.read_bytes()
        analysis_dir = self.root / "analysis/current_input_ablation"
        report_dir = self.root / "report"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        summary = {"schema_version": 1, "q4_evaluated": False}
        summary["analysis_sha256"] = experiment._payload_sha256(summary)
        summary_path = experiment._write_json(
            analysis_dir / "current_input_analysis_summary.json", summary
        )
        validation = {
            "schema_version": 1,
            "status": "pass",
            "analysis_sha256": summary["analysis_sha256"],
        }
        validation["validation_sha256"] = experiment._payload_sha256(validation)
        experiment._write_json(
            analysis_dir / "current_input_validation_summary.json", validation
        )
        report_path = report_dir / "current_input_ablation_q3_report.html"
        report_path.write_text("<html>valid</html>", encoding="utf-8")
        report_manifest = {
            "schema_version": 1,
            "analysis_sha256": summary["analysis_sha256"],
            "report_path": str(report_path),
            "report_sha256": experiment._sha256_file(report_path),
            "q4_predictions_generated": False,
            "q4_evaluated": False,
        }
        report_manifest["report_manifest_sha256"] = experiment._payload_sha256(
            report_manifest
        )
        experiment._write_json(
            report_dir / "current_input_ablation_report_manifest.json",
            report_manifest,
        )
        experiment._write_json(
            experiment_status_path,
            {
                "status": "completed_q3_only",
                "q4_evaluator_calls": 0,
                "q4_predictions_generated": False,
                "q4_evaluated": False,
                "analysis_path": str(summary_path),
                "analysis_sha256": experiment._sha256_file(summary_path),
                "report_path": str(report_path),
                "report_sha256": experiment._sha256_file(report_path),
            },
        )
        experiment.refresh_current_input_lineage(self.root)
        original_report = report_path.read_bytes()
        try:
            with mock.patch.object(
                experiment, "_completed_job_is_valid", return_value=True
            ):
                experiment._validate_completed_experiment(self.root)
                report_path.write_text("<html>tampered</html>", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                    experiment._validate_completed_experiment(self.root)
        finally:
            report_path.write_bytes(original_report)
            experiment_status_path.write_bytes(original_status)
            experiment.refresh_current_input_lineage(self.root)

    def test_public_cli_defaults_to_safe_prepare_and_fixed_root(self) -> None:
        args = rq3_main.build_parser().parse_args(
            ["train-news-first-vol-current-input-sweep"]
        )
        self.assertEqual(args.action, "prepare")
        self.assertEqual(args.config, str(CONFIG_PATH))
        self.assertEqual(
            args.output_dir,
            "outputs/experiments/"
            "rq3_news_first_vol_current_support_masked_seed_sweep_"
            "q097_103_ttm07_38_v1",
        )
        expected = Path("/tmp/current-input-public-cli")
        with mock.patch.object(
            experiment,
            "run_news_first_vol_current_input_sweep",
            return_value=expected,
        ) as run:
            result = rq3_main.main(
                [
                    "train-news-first-vol-current-input-sweep",
                    "prepare",
                    "--output-dir",
                    str(expected),
                ]
            )
        self.assertEqual(result, expected)
        run.assert_called_once()

    def test_reference_manifest_freezes_six_full_current_cells(self) -> None:
        self.assertEqual(len(self.reference_rows), 6)
        fingerprint = generator_current_input_fingerprint("full_current")
        for row in self.reference_rows:
            expected = experiment.EXPECTED_REFERENCE_HASHES[row["reference_job_id"]]
            self.assertEqual(
                row["reference_training_config_sha256"], expected["config"]
            )
            self.assertEqual(row["reference_status_sha256"], expected["status"])
            self.assertEqual(row["generator_checkpoint_sha256"], expected["generator"])
            self.assertEqual(
                row["discriminator_checkpoint_sha256"], expected["discriminator"]
            )
            self.assertEqual(row["best_learned_metadata_sha256"], expected["metadata"])
            self.assertEqual(
                row["reference_q3_pair_panel_sha256"], expected["pair_panel"]
            )
            self.assertEqual(row["generator_current_input_mode"], "full_current")
            self.assertEqual(row["generator_current_input_fingerprint"], fingerprint)
            self.assertEqual(row["reference_q3_pair_count"], 123)
            self.assertEqual(row["reference_q3_session_count"], 33)

        registry = json.loads((self.root / "registry/jobs.json").read_text())
        manifest = self.root / "full_current_reference_manifest.csv"
        self.assertEqual(
            registry["full_current_reference_manifest_sha256"],
            experiment._sha256_file(manifest),
        )
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)

    def test_generated_configs_change_only_explicit_input_semantics_and_output(
        self,
    ) -> None:
        registry = json.loads((self.root / "registry/jobs.json").read_text())
        references = {
            (int(row["seed"]), int(row["tolerance_minutes"])): row
            for row in self.reference_rows
        }
        for job in registry["jobs"]:
            new = yaml.safe_load(Path(job["training_config_path"]).read_text())
            reference = references[(job["seed"], job["tolerance_minutes"])]
            old = yaml.safe_load(
                Path(reference["reference_training_config_path"]).read_text()
            )
            differing = {
                key
                for key in set(old) | set(new)
                if old.get(key, "<missing>") != new.get(key, "<missing>")
            }
            self.assertEqual(
                differing,
                {
                    "generator_current_input_mode",
                    "generator_noise_mode",
                    "output_root",
                },
            )
            self.assertEqual(new["generator_noise_mode"], "gaussian")
            self.assertEqual(
                new["generator_current_input_mode"], "current_support_masked"
            )
            self.assertEqual(new["support_mask_mode"], "raw_joint")
            self.assertEqual(new["validation_mc_samples"], 16)
            self.assertEqual(
                experiment._shared_training_contract_sha256(new),
                experiment._shared_training_contract_sha256(old),
            )

    def test_registry_has_q3_only_lineage_and_balanced_gpu_utilization(self) -> None:
        registry = json.loads((self.root / "registry/jobs.json").read_text())
        self.assertEqual(registry["experiment_kind"], experiment.EXPERIMENT_KIND)
        self.assertEqual(registry["q4_evaluator_calls"], 0)
        self.assertFalse(registry["q4_predictions_generated"])
        self.assertFalse(registry["q4_evaluated"])
        self.assertEqual(len(registry["jobs"]), 6)
        gpu_counts = {
            gpu_id: sum(job["gpu_id"] == gpu_id for job in registry["jobs"])
            for gpu_id in (0, 1)
        }
        self.assertEqual(gpu_counts, {0: 3, 1: 3})
        for job in registry["jobs"]:
            self.assertEqual(job["q4_evaluator_calls"], 0)
            self.assertEqual(
                job["generator_current_input_fingerprint"],
                generator_current_input_fingerprint("current_support_masked"),
            )
            self.assertEqual(
                job["full_current_reference_manifest_sha256"],
                registry["full_current_reference_manifest_sha256"],
            )
        lineage = json.loads((self.root / "experiment_lineage.json").read_text())
        current_status = json.loads(
            (self.root / "registry/experiment_status.json").read_text()
        )
        self.assertEqual(lineage["experiment_status"], current_status["status"])
        self.assertEqual(lineage["q4_evaluator_calls"], 0)
        with (self.root / "output_hashes.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            roles = {row["artifact_role"] for row in csv.DictReader(handle)}
        self.assertIn("experiment:experiment_lineage.json", roles)
        self.assertIn("experiment:full_current_reference_manifest.csv", roles)

    def test_tampered_job_config_or_reference_manifest_fails_closed(self) -> None:
        registry_path = self.root / "registry/jobs.json"
        registry_bytes = registry_path.read_bytes()
        registry = json.loads(registry_bytes)
        registry["jobs"][0]["generator_current_input_mode"] = "full_current"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Job-spec hash mismatch"):
            experiment._validate_registry(self.root)
        registry_path.write_bytes(registry_bytes)

        manifest_path = self.root / "full_current_reference_manifest.csv"
        manifest_bytes = manifest_path.read_bytes()
        manifest_path.write_bytes(manifest_bytes + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            experiment._validate_registry(self.root)
        manifest_path.write_bytes(manifest_bytes)

        registry = json.loads(registry_path.read_text())
        config_path = Path(registry["jobs"][0]["training_config_path"])
        config_bytes = config_path.read_bytes()
        payload = yaml.safe_load(config_bytes)
        payload["generator_current_input_mode"] = "full_current"
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Training config hash mismatch"):
            experiment._validate_registry(self.root)
        config_path.write_bytes(config_bytes)
        experiment._validate_registry(self.root)

    def test_worker_command_uses_independent_module_cli(self) -> None:
        job = json.loads((self.root / "registry/jobs.json").read_text())["jobs"][0]
        command = experiment.build_current_input_worker_command(
            CONFIG_PATH, self.root, job, dry_run=True, resume=True
        )
        self.assertIn("scripts.rq3.news_first_vol_current_input_sweep", command)
        self.assertIn("--worker-dry-run", command)
        self.assertIn("--resume", command)

    def test_dry_run_executes_both_waves_without_formal_completion(self) -> None:
        calls: list[tuple[int, int]] = []

        def fake_wave(root, config_path, wave, jobs, *, dry_run, resume):
            del config_path, resume
            self.assertTrue(dry_run)
            calls.append((wave, len(jobs)))
            for job in jobs:
                status_path = experiment._job_status_path(root, job["job_id"])
                status = json.loads(status_path.read_text())
                status.update(
                    {
                        "status": "dry_run_passed",
                        "q4_evaluator_calls": 0,
                        "run_dir": str(root / "dry_runs" / job["job_id"]),
                    }
                )
                status_path.write_text(json.dumps(status), encoding="utf-8")

        with (
            mock.patch.object(experiment, "_run_wave", side_effect=fake_wave),
            mock.patch.object(experiment.training, "_write_resource_summary"),
            mock.patch.object(experiment, "_run_postprocess") as postprocess,
        ):
            result = experiment.launch_current_input_experiment(
                CONFIG_PATH, self.root, dry_run=True
            )
        postprocess.assert_not_called()
        self.assertEqual(result, self.root.resolve())
        self.assertEqual(calls, [(1, 4), (2, 2)])
        status = json.loads((self.root / "registry/experiment_status.json").read_text())
        self.assertEqual(status["status"], "ready_after_dry_run")
        self.assertEqual(status["q4_evaluator_calls"], 0)

    def test_postprocess_failure_cannot_mark_formal_experiment_completed(self) -> None:
        status_path = self.root / "registry/experiment_status.json"
        original = status_path.read_bytes()
        try:
            with (
                mock.patch.object(experiment, "_jobs_for_wave", return_value=[]),
                mock.patch.object(experiment, "_run_wave"),
                mock.patch.object(experiment, "_validate_wave_completion"),
                mock.patch.object(
                    experiment, "_completed_job_is_valid", return_value=True
                ),
                mock.patch.object(
                    experiment,
                    "_run_postprocess",
                    side_effect=RuntimeError("analysis failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "analysis failed"):
                    experiment.launch_current_input_experiment(
                        CONFIG_PATH, self.root, resume=True, dry_run=False
                    )
            failed = json.loads(status_path.read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["q4_evaluator_calls"], 0)
        finally:
            status_path.write_bytes(original)
            experiment.refresh_current_input_lineage(self.root)

    def test_postprocess_only_permitted_analysis_drift_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._failed_postprocess_root(Path(temporary))
            original_code_ledger = (root / "code_hashes.csv").read_bytes()
            current_rows = experiment._read_code_manifest(root / "code_hashes.csv")
            analysis_relative = "scripts/rq3/news_first_vol_current_input_analysis.py"
            for row in current_rows:
                if row["relative_path"] == analysis_relative:
                    row["sha256"] = "a" * 64
                    row["size_bytes"] = int(row["size_bytes"]) + 1
                    break
            else:  # pragma: no cover - manifest construction already tests this
                self.fail("analysis module is missing from the code manifest")

            with (
                mock.patch.object(experiment, "_validate_registry"),
                mock.patch.object(experiment, "_validate_split_manifest"),
                mock.patch.object(
                    experiment, "_completed_job_is_valid", return_value=True
                ),
                mock.patch.object(experiment, "_code_rows", return_value=current_rows),
                mock.patch.object(
                    experiment,
                    "_run_postprocess",
                    side_effect=self._write_minimal_postprocess_artifacts,
                ),
                mock.patch.object(
                    experiment, "_run_wave", side_effect=AssertionError("retrained")
                ) as run_wave,
            ):
                result = experiment.postprocess_current_input_experiment(root)

            self.assertEqual(result, root.resolve())
            run_wave.assert_not_called()
            self.assertEqual(
                (root / "code_hashes.csv").read_bytes(), original_code_ledger
            )
            status = json.loads((root / "registry/experiment_status.json").read_text())
            self.assertEqual(status["status"], "completed_q3_only")
            self.assertTrue(status["postprocess_only"])
            self.assertTrue(status["code_changed_since_prepared_snapshot"])
            self.assertEqual(
                status["postprocess_code_changed_paths"], [analysis_relative]
            )
            for filename in (
                experiment.CURRENT_CODE_MANIFEST_FILENAME,
                experiment.POSTPROCESS_CODE_DRIFT_LEDGER_FILENAME,
                experiment.PRE_POSTPROCESS_OUTPUT_HASH_MANIFEST_FILENAME,
            ):
                self.assertTrue((root / filename).is_file())
            lineage = json.loads((root / "experiment_lineage.json").read_text())
            self.assertEqual(
                lineage["prepared_code_manifest_sha256"],
                status["prepared_code_manifest_sha256"],
            )
            self.assertEqual(
                lineage["current_code_manifest_sha256"],
                status["current_code_manifest_sha256"],
            )

    def test_postprocess_only_rejects_training_code_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._failed_postprocess_root(Path(temporary))
            original_status = (root / "registry/experiment_status.json").read_bytes()
            current_rows = experiment._read_code_manifest(root / "code_hashes.csv")
            training_relative = "src/wgan_option/models/generator.py"
            for row in current_rows:
                if row["relative_path"] == training_relative:
                    row["sha256"] = "b" * 64
                    row["size_bytes"] = int(row["size_bytes"]) + 1
                    break
            else:  # pragma: no cover - base training manifest is separately tested
                self.fail("Generator module is missing from the code manifest")

            with (
                mock.patch.object(experiment, "_validate_registry"),
                mock.patch.object(experiment, "_validate_split_manifest"),
                mock.patch.object(
                    experiment, "_completed_job_is_valid", return_value=True
                ),
                mock.patch.object(experiment, "_code_rows", return_value=current_rows),
                mock.patch.object(experiment, "_run_postprocess") as postprocess,
            ):
                with self.assertRaisesRegex(
                    ValueError, f"Forbidden postprocess code drift: {training_relative}"
                ):
                    experiment.postprocess_current_input_experiment(root)
            postprocess.assert_not_called()
            self.assertEqual(
                (root / "registry/experiment_status.json").read_bytes(), original_status
            )
            self.assertFalse(
                (root / experiment.CURRENT_CODE_MANIFEST_FILENAME).exists()
            )

    def test_postprocess_only_rejects_incomplete_job_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._failed_postprocess_root(Path(temporary))
            original_status = (root / "registry/experiment_status.json").read_bytes()
            with (
                mock.patch.object(experiment, "_validate_registry"),
                mock.patch.object(experiment, "_validate_split_manifest"),
                mock.patch.object(
                    experiment, "_completed_job_is_valid", return_value=False
                ),
                mock.patch.object(experiment, "_postprocess_code_snapshot") as snapshot,
            ):
                with self.assertRaisesRegex(RuntimeError, "six-job matrix"):
                    experiment.postprocess_current_input_experiment(root)
            snapshot.assert_not_called()
            self.assertEqual(
                (root / "registry/experiment_status.json").read_bytes(), original_status
            )

    def test_module_cli_exposes_postprocess_only_action(self) -> None:
        expected = Path("/tmp/current-input-postprocess-only")
        parsed = experiment._parser().parse_args(
            ["postprocess", "--output-dir", str(expected)]
        )
        self.assertEqual(parsed.action, "postprocess")
        with mock.patch.object(
            experiment, "postprocess_current_input_experiment", return_value=expected
        ) as postprocess:
            result = experiment.run_news_first_vol_current_input_sweep(
                CONFIG_PATH,
                expected,
                action="postprocess",
            )
        self.assertEqual(result, expected)
        postprocess.assert_called_once_with(expected)


if __name__ == "__main__":
    unittest.main()
