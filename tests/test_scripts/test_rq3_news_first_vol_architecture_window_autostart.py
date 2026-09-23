from __future__ import annotations

from collections import namedtuple
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.rq3 import news_first_vol_architecture_window_autostart as autostart


class ArchitectureWindowAutostartTests(unittest.TestCase):
    def _write_signed(self, path: Path, payload: dict[str, object]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(autostart._canonical_bytes(autostart._signed(payload)))
        return path

    def _write_legacy_signed(self, path: Path, payload: dict[str, object]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        value = dict(payload)
        value["payload_sha256"] = autostart._legacy_payload_sha256(value)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    def test_validate_old_terminal_binds_280_jobs_and_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [
                {"job_id": f"job-{index}", "status": "completed"}
                for index in range(280)
            ]
            snapshot = self._write_legacy_signed(
                root / "registry/final_registry_snapshot.json",
                {
                    "schema_version": 1,
                    "kind": "pure_cnn_parent_film_text_effect_final_registry_snapshot_v1",
                },
            )
            qa = self._write_legacy_signed(
                root / "qa.json",
                {
                    "schema_version": 1,
                    "kind": "pure_cnn_parent_film_text_effect_terminal_qa_v1",
                    "status": "passed",
                    "training_jobs_completed": 280,
                },
            )
            output = root / "output_hashes.csv"
            output.write_text("relative_path,size_bytes,sha256\n", encoding="utf-8")
            code = root / "stages/backbones/code_hashes.csv"
            code.parent.mkdir(parents=True)
            with code.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("artifact_role", "path", "size_bytes", "sha256"),
                )
                writer.writeheader()
                for index in range(288):
                    writer.writerow(
                        {
                            "artifact_role": f"code:{index}",
                            "path": f"source-{index}",
                            "size_bytes": 1,
                            "sha256": f"{index:064x}"[-64:],
                        }
                    )
            registry = {
                "kind": "pure_cnn_parent_film_text_effectiveness_registry_v1",
                "status": "completed",
                "terminal_complete": True,
                "jobs": jobs,
                "jobs_sha256": autostart._legacy_payload_sha256(jobs),
                "terminal_qa_path": str(qa),
                "terminal_qa_sha256": autostart.sha256_file(qa),
                "output_manifest_path": str(output),
                "output_manifest_sha256": autostart.sha256_file(output),
                "final_registry_snapshot_path": str(snapshot),
                "final_registry_snapshot_sha256": autostart.sha256_file(snapshot),
            }
            registry_path = root / "registry/task_registry.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")

            evidence = autostart.validate_old_terminal(root)
            self.assertEqual(evidence["training_jobs_completed"], 280)
            self.assertEqual(evidence["source_manifest_rows"], 288)

            jobs[-1]["status"] = "failed"
            registry["jobs"] = jobs
            registry["jobs_sha256"] = autostart._legacy_payload_sha256(jobs)
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaisesRegex(autostart.AutostartError, "280/280"):
                autostart.validate_old_terminal(root)

    def test_validate_core_ready_requires_exact_five_current_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            rows = []
            for relative in autostart.CORE_FILES:
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
                rows.append(
                    {
                        "path": relative,
                        "size_bytes": path.stat().st_size,
                        "sha256": autostart.sha256_file(path),
                    }
                )
            ready = self._write_legacy_signed(
                repo / "core_ready.json",
                {
                    "schema_version": 1,
                    "kind": "architecture_window_core_ready_v1",
                    "status": "ready",
                    "files": rows,
                    "staging_removed": True,
                    "gpu_started": False,
                },
            )
            with mock.patch.object(autostart, "REPO_ROOT", repo):
                evidence = autostart.validate_core_ready(ready)
                self.assertEqual(len(evidence["files"]), 5)
                (repo / autostart.CORE_FILES[0]).write_text("drift", encoding="utf-8")
                with self.assertRaisesRegex(autostart.AutostartError, "hash drift"):
                    autostart.validate_core_ready(ready)

    def test_signed_reader_accepts_only_the_two_frozen_canonical_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {"schema_version": 1, "kind": "fixture", "value": "x"}
            pretty = self._write_signed(root / "pretty.json", payload)
            legacy = self._write_legacy_signed(root / "legacy.json", payload)
            self.assertEqual(autostart._read_signed(pretty)["value"], "x")
            self.assertEqual(autostart._read_signed(legacy)["value"], "x")
            drift = json.loads(legacy.read_text(encoding="utf-8"))
            drift["value"] = "changed"
            legacy.write_text(json.dumps(drift), encoding="utf-8")
            with self.assertRaisesRegex(autostart.AutostartError, "drift"):
                autostart._read_signed(legacy)

    def test_stable_resource_gate_resets_after_busy_sample(self) -> None:
        idle = {
            "gpus": [
                {
                    "gpu_index": index,
                    "gpu_uuid": f"gpu-{index}",
                    "memory_used_mib": 10.0,
                    "memory_total_mib": 24576.0,
                    "utilization_gpu_pct": 0.0,
                }
                for index in (0, 1)
            ],
            "compute_processes": [],
        }
        busy = json.loads(json.dumps(idle))
        busy["gpus"][0]["utilization_gpu_pct"] = 20.0
        values = iter((idle, busy, idle, idle))
        Usage = namedtuple("Usage", "total used free")
        sleeps: list[float] = []
        evidence = autostart.wait_for_stable_resources(
            samples=2,
            sample_seconds=1.0,
            snapshot_provider=lambda: next(values),
            disk_usage_provider=lambda _path: Usage(100 << 30, 20 << 30, 80 << 30),
            sleeper=sleeps.append,
        )
        self.assertEqual(evidence["stable_samples"], 2)
        self.assertEqual(len(evidence["samples"]), 2)
        self.assertEqual(len(sleeps), 3)

    def test_dry_run_gate_requires_passed_core_inputs_and_exact_jobs(self) -> None:
        payload = {
            "line": "architecture",
            "new_training_jobs": 69,
            "formal_root_created": False,
            "test_metric_files_read": 0,
            "baseline_status": "passed",
            "input_preflight": {"status": "passed"},
            "core_runtime_preflight": {"status": "passed"},
        }
        self.assertEqual(
            autostart.validate_dry_run(payload, line="architecture")[
                "new_training_jobs"
            ],
            69,
        )
        payload["core_runtime_preflight"] = {"status": "blocked"}
        with self.assertRaisesRegex(autostart.AutostartError, "dry-run"):
            autostart.validate_dry_run(payload, line="architecture")

    def test_shared_benchmark_suite_binds_both_external_evidence_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            architecture = base / autostart.ARCHITECTURE_ROOT.name
            window = base / autostart.WINDOW_ROOT.name
            benchmark_paths = {}
            rows = []
            for root in (architecture, window):
                path = root.with_name(root.name + "_control") / "benchmark.json"
                self._write_signed(
                    path,
                    {
                        "schema_version": 1,
                        "kind": "architecture_window_gpu_epoch1_benchmark_v1",
                        "status": "passed",
                        "formal_output_root": str(root.resolve()),
                        "selected_workers_per_gpu": 8,
                        "formal_root_created_before_benchmark_pass": False,
                    },
                )
                benchmark_paths[root.name] = path
                rows.append(
                    {
                        "formal_root": str(root.resolve()),
                        "benchmark_path": str(path.resolve()),
                        "benchmark_sha256": autostart.sha256_file(path),
                    }
                )
            suite = self._write_signed(
                base / "benchmark_suite.json",
                {
                    "schema_version": 1,
                    "kind": "architecture_window_dual_benchmark_suite_v1",
                    "benchmarks": rows,
                },
            )
            with (
                mock.patch.object(autostart, "ARCHITECTURE_ROOT", architecture),
                mock.patch.object(autostart, "WINDOW_ROOT", window),
            ):
                observed = autostart.validate_shared_benchmark_suite(suite)
                self.assertEqual(len(observed["benchmarks"]), 2)
                benchmark_paths[architecture.name].write_text("{}", encoding="utf-8")
                with self.assertRaises(autostart.AutostartError):
                    autostart.validate_shared_benchmark_suite(suite)

    def test_existing_live_launch_is_monitor_only_and_dead_partial_needs_resume(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory)
            self._write_signed(
                control / "launch_manifest.json",
                {
                    "schema_version": 1,
                    "kind": "architecture_window_autostart_launch_v1",
                    "pipelines": [
                        {"pid": 123, "process_start_ticks": 456, "line": "architecture"}
                    ],
                },
            )
            with (
                mock.patch.object(autostart, "pid_alive", return_value=True),
                mock.patch.object(autostart, "_terminal_new_root", return_value=False),
            ):
                self.assertEqual(
                    autostart.classify_existing(control), "live_monitor_only"
                )
            with (
                mock.patch.object(autostart, "pid_alive", return_value=False),
                mock.patch.object(autostart, "_terminal_new_root", return_value=False),
            ):
                self.assertEqual(autostart.classify_existing(control), "dead_partial")

    def test_dead_partial_requires_resume_and_skips_initial_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory)
            with mock.patch.object(
                autostart, "classify_existing", return_value="dead_partial"
            ):
                with self.assertRaisesRegex(autostart.AutostartError, "--resume"):
                    autostart.run(control_root=control, resume=False, poll_seconds=0.1)
            with (
                mock.patch.object(
                    autostart, "classify_existing", return_value="dead_partial"
                ),
                mock.patch.object(autostart, "validate_resume_gates") as validate,
                mock.patch.object(autostart, "_wait_no_processes") as drain,
                mock.patch.object(
                    autostart,
                    "wait_for_stable_resources",
                    return_value={"status": "passed"},
                ),
                mock.patch.object(
                    autostart,
                    "_write_signed",
                    return_value=control / "gate.json",
                ),
                mock.patch.object(
                    autostart,
                    "launch_pipelines",
                    return_value=control / "launch_manifest.json",
                ) as launch,
                mock.patch.object(
                    autostart,
                    "status",
                    return_value={"state": "live_monitor_only"},
                ),
            ):
                result = autostart.run(
                    control_root=control,
                    resume=True,
                    poll_seconds=0.1,
                    gpu_stable_samples=1,
                )
            self.assertEqual(result["state"], "live_monitor_only")
            validate.assert_called_once()
            drain.assert_called_once()
            launch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
