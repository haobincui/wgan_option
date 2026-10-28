"""Scheduling, resource gates, evidence binding and inherited-slot safety."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.rq3 import rq12_direct_vega_parallel as subject


class ParallelDirectTests(unittest.TestCase):
    @staticmethod
    def _override_reference(root):
        workspace = root / "prior_trials"
        legacy_children = []
        configs = []
        for family in ("film", "pure"):
            config = root / f"new_{family}.yaml"
            formal_root = root / "fresh_formal" / family
            config.write_text(
                f"experiment:\n  output_root: {formal_root}\n"
                f"runtime:\n  parallel_scheduler_mode: {subject.SCHEDULER_MODE}\n")
            configs.append(config)
            legacy_root = workspace / "workers_08" / family
            legacy_root.mkdir(parents=True)
            (legacy_root / "resolved_config.yaml").write_text("prior_epoch_count: 1\n")
            for name in ("code_hashes.csv", "source_hashes.csv", "config_hashes.csv"):
                (legacy_root / name).write_text("path,sha256\n")
            legacy_children.append(subject.Child(family, legacy_root, config, subject.MODULES[family]))
        subject.write_json(workspace / "workers_08/attempt.json", subject.signed({
            "kind": "rq12_direct_vega_candidate_v1", "status": "passed",
            "completed": 200, "failed": 0, "workers_per_gpu": 8,
            "supervisor_code_sha256": "prior_scheduler_used_only_for_disk_sizing",
            "children": [subject.child_binding(child) for child in legacy_children],
        }))
        return configs, workspace

    def test_explicit_forty_worker_override_is_signed_without_claiming_benchmark_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs, workspace = self._override_reference(root)
            main = root / "fresh_formal"
            path = subject.user_concurrency_override(*configs, main, workers_per_gpu=40,
                                                     workspace=workspace)
            evidence = subject.read_signed(path, kind=subject.OVERRIDE_KIND)
            self.assertEqual(evidence["status"], "authorized_by_user")
            self.assertEqual(evidence["authorization"], "explicit_user_instruction")
            self.assertEqual(evidence["user_instruction"], subject.OVERRIDE_INSTRUCTIONS[40])
            self.assertEqual(evidence["selected_workers_per_gpu"], 40)
            self.assertFalse(evidence["performance_benchmark_passed"])
            self.assertTrue(evidence["benchmark_skipped"])
            self.assertFalse(evidence["disk_projection_reference"]["training_code_reused"])
            self.assertEqual(evidence["supervisor_code_sha256"], subject.sha256_file(subject.__file__))
            for family, config in zip(("film", "pure"), configs):
                self.assertEqual(evidence["source_configs"][family], {
                    "path": str(config.resolve()), "sha256": subject.sha256_file(config)})
            with patch.object(subject, "formal_disk_gate") as disk_gate, \
                    patch.object(subject, "_prepare_children") as prepare, \
                    patch.object(subject, "verify_bindings", side_effect=AssertionError("legacy code cannot gate new training")):
                children = subject.prepare_global(*configs, main)
            self.assertEqual([child.root for child in children], [main / "film", main / "pure"])
            disk_gate.assert_called_once_with(children, evidence, main)
            prepare.assert_called_once_with(children, 40, benchmark=False, evidence=evidence, resume=False)

    def test_override_rejects_unapproved_worker_counts_before_any_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(subject, "_children_from_configs") as children, \
                    patch.object(subject, "benchmark") as benchmark, \
                    patch.object(subject, "prepare_global") as prepare:
                for workers in (0, -1, 21, 39, 41, 56):
                    with self.subTest(workers=workers):
                        with self.assertRaisesRegex(ValueError, "20 or 40"):
                            subject.user_concurrency_override(root / "film.yaml", root / "pure.yaml",
                                                              root / "formal", workers_per_gpu=workers)
                        with self.assertRaisesRegex(ValueError, "20 or 40"):
                            subject.main(["run", "--workers-per-gpu", str(workers), "--skip-benchmark",
                                          "--output-dir", str(root / "formal")])
                children.assert_not_called()
                benchmark.assert_not_called()
                prepare.assert_not_called()

    def test_even_resigned_override_requires_matching_instruction_and_source_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs, workspace = self._override_reference(root)
            main = root / "fresh_formal"
            path = subject.user_concurrency_override(*configs, main, workers_per_gpu=40,
                                                     workspace=workspace)
            original = subject.read_signed(path)
            for alteration in ({"authorization": "implicit"},
                               {"user_instruction": subject.OVERRIDE_INSTRUCTIONS[20]},
                               {"selected_workers_per_gpu": 39}):
                with self.subTest(alteration=alteration), patch.object(subject, "_prepare_children") as prepare:
                    subject.write_json(path, subject.signed({**original, **alteration}))
                    with self.assertRaisesRegex(ValueError, "authorization drift"):
                        subject.prepare_global(*configs, main)
                    prepare.assert_not_called()
            subject.write_json(path, original)
            configs[0].write_text(configs[0].read_text() + "changed_training_scope: true\n")
            with self.assertRaisesRegex(ValueError, "override drift"):
                subject.user_concurrency_override(*configs, main, workers_per_gpu=40, workspace=workspace)

    def test_public_shared_and_direct_helpers_use_both_family_profiles(self):
        for family in ("film", "pure"):
            stack = []

            @contextmanager
            def outer():
                stack.append(family)
                try:
                    yield
                finally:
                    stack.pop()

            @contextmanager
            def inner():
                self.assertEqual(stack, [family])
                stack.append("shared")
                try:
                    yield
                finally:
                    stack.pop()

            def helper(*args, **kwargs):
                self.assertEqual(stack, [family, "shared"])
                return "correct family profile"

            def direct_call(name, *args, **kwargs):
                self.assertEqual(name, "direct_helper")
                return helper(*args, **kwargs)

            shared = SimpleNamespace(multiseed_profile=inner, shared_helper=helper,
                                     _call=direct_call)
            module = SimpleNamespace(public_helper=helper)
            if family == "film":
                module.film_text_profile = outer
                module.multiseed = shared
            else:
                module.pure_multiseed_profile = outer
                module.multi = shared
            child = subject.Child(family, Path("/unused"), Path("/unused/config"), "mock.module")
            with patch.object(subject.importlib, "import_module", return_value=module):
                for name in ("public_helper", "shared_helper", "direct_helper"):
                    self.assertEqual(subject._child_call(child, name), "correct family profile")
                    self.assertEqual(stack, [])

    def test_candidate_selection_uses_throughput_and_two_small_gains_stop(self):
        rows = [{"status": "passed", "workers_per_gpu": workers,
                 "throughput_jobs_per_second": rate}
                for workers, rate in ((8, 1.), (10, 1.02), (12, 1.04))]
        self.assertFalse(subject.plateau_reached(rows[:2]))
        self.assertTrue(subject.plateau_reached(rows))
        self.assertEqual(subject.select_candidate(rows)["workers_per_gpu"], 12)
        rows.append({"status": "passed", "workers_per_gpu": 16,
                     "throughput_jobs_per_second": .95})
        self.assertEqual(subject.select_candidate(rows)["workers_per_gpu"], 12)
        rows.append({"status": "resource_failed", "workers_per_gpu": 24})
        self.assertEqual(subject.select_candidate(rows)["workers_per_gpu"], 12)

    def test_resource_gate_checks_each_card_and_host_ram(self):
        snapshot = {"gpu": {"0": {"memory_used_mib": 100},
                            "1": {"memory_used_mib": 20 * 1024}},
                    "host_ram_fraction": .2}
        with self.assertRaises(subject.ResourceLimitError):
            subject.enforce_resources(snapshot, 20, .85)
        snapshot["gpu"]["1"]["memory_used_mib"] = 100
        snapshot["host_ram_fraction"] = .85
        with self.assertRaises(subject.ResourceLimitError):
            subject.enforce_resources(snapshot, 20, .85)
        snapshot["host_ram_fraction"] = .1
        subject.enforce_resources(snapshot, 20, .85)

    def test_worker_keeps_slot_after_parent_closes_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor, slot = subject.acquire_gpu_slot(root, 0, 1)
            self.assertEqual(slot, 0)
            process = subprocess.Popen(
                [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(0.3)"],
                pass_fds=(descriptor,), stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), "ready")
                subject.close_gpu_slot(descriptor)
                self.assertIsNone(subject.acquire_gpu_slot(root, 0, 1))
                process.wait(timeout=3)
                acquired = subject.acquire_gpu_slot(root, 0, 1)
                self.assertIsNotNone(acquired)
                subject.close_gpu_slot(acquired[0])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()

    def test_signed_bindings_reject_source_manifest_member_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "source.yaml"
            config.write_text("sample: 1\n")
            source = root / "weights.npy"
            source.write_bytes(b"original weights")
            manifest = root / "source_hashes.csv"
            manifest.write_text("path,sha256\n" + str(source) + "," + subject.sha256_file(source) + "\n")
            binding = {"config_path": str(config), "source_config_sha256": subject.sha256_file(config),
                       "files": {"source_manifest": {"path": str(manifest),
                                                      "sha256": subject.sha256_file(manifest)}}}
            subject.verify_bindings([binding])
            source.write_bytes(b"changed weights")
            with self.assertRaisesRegex(ValueError, "Manifest member drift"):
                subject.verify_bindings([binding])

    def test_all_mandatory_candidates_run_before_extension_plateau(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            film, pure = root / "film.yaml", root / "pure.yaml"
            film.write_text("film configuration\n")
            pure.write_text("pure configuration\n")
            children = [subject.Child("film", root / "film", film, subject.MODULES["film"]),
                        subject.Child("pure", root / "pure", pure, subject.MODULES["pure"])]
            observed = []
            rates = {8: 1, 10: 1.01, 12: 1.02, 16: 1.03, 18: 1.04,
                     24: 1.05, 32: 1.06, 48: 1.2, 56: 1.21, 64: 1.22}

            def run(children, root, workers, **kwargs):
                observed.append(workers)
                return {"status": "completed", "completed": 200,
                        "throughput_jobs_per_second": rates[workers]}

            def binding(child):
                return {**child.as_dict(), "source_config_sha256": subject.sha256_file(child.config_path),
                        "files": {}}

            with patch.object(subject, "_children_from_configs", return_value=children), \
                    patch.object(subject, "_prepare_children"), \
                    patch.object(subject, "run_queue", side_effect=run), \
                    patch.object(subject, "child_binding", side_effect=binding), \
                    patch.object(subject, "_validate_real_updates"):
                result = subject.benchmark(film, pure, root / "global", workspace=root / "benchmarks")
            self.assertEqual(observed, [8, 10, 12, 16, 18, 24, 32, 48, 56, 64])
            self.assertEqual(subject.read_signed(result)["selected_workers_per_gpu"], 64)

    def test_disk_gate_reserves_thirty_gib_after_projected_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            children, jobs = self._fixtures(root)
            formal = []
            for child in children:
                specs = [job.spec for job in jobs if job.child.family == child.family]
                subject.write_json(child.root / "registry/task_registry.json", {"jobs": specs})
                formal.append(subject.Child(child.family, root / "formal" / child.family,
                                            child.config_path, child.module))
            evidence = {"children": [child.as_dict() for child in children]}
            with patch.object(subject.shutil, "disk_usage", return_value=SimpleNamespace(free=30 * 1024**3)):
                with self.assertRaises(subject.ResourceLimitError):
                    subject.formal_disk_gate(formal, evidence, root / "global")

    @staticmethod
    def _fixtures(root):
        children, jobs = [], []
        for family, total in (("film", 160), ("pure", 40)):
            child_root = root / family
            (child_root / "registry/job_status").mkdir(parents=True)
            (child_root / "logs").mkdir()
            config_path = root / f"{family}.yaml"
            config_path.write_text("runtime:\n  python_executable: " + sys.executable + "\n")
            (child_root / "resolved_config.yaml").write_text(config_path.read_text())
            for name in ("code_hashes.csv", "source_hashes.csv", "config_hashes.csv"):
                (child_root / name).write_text("path,sha256\n")
            child = subject.Child(family, child_root, config_path, subject.MODULES[family])
            children.append(child)
            for index in range(total):
                gpu = index % 2
                fold = "f3_2023q3" if gpu == 0 else "f4_2023q4"
                spec = {"job_id": f"{family}_{index}", "gpu_id": gpu, "fold": fold,
                        "seed": index, "arm": family, "job_spec_sha256": f"job{index}",
                        "training_config_sha256": f"config{index}"}
                job = subject.Job(child, spec)
                subject.write_json(job.status_path, {"status": "pending", "attempt": 0})
                jobs.append(job)
            subject.write_json(child_root / "registry/task_registry.json", {"root_mode": "formal"})
        return children, jobs

    def test_shared_queue_never_adds_child_limits_and_replaces_finished_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            children, jobs = self._fixtures(root)
            by_id = {str(job.spec["job_id"]): job for job in jobs}
            processes = []
            observed_max = {0: 0, 1: 0}
            stragglers = set()
            replenished_during_straggler = []

            class FakeProcess:
                def __init__(self, command, **kwargs):
                    job_id = command[command.index("--job-id") + 1]
                    self.job = by_id[job_id]
                    self.pid = 10000 + len(processes)
                    self.returncode = None
                    self.polls = 0
                    if not any(process.job.gpu == self.job.gpu for process in processes):
                        stragglers.add(job_id)
                    elif any(str(process.job.spec["job_id"]) in stragglers
                             and process.returncode is None for process in processes):
                        if sum(process.job.gpu == self.job.gpu for process in processes) >= 2:
                            replenished_during_straggler.append(job_id)
                    processes.append(self)
                    alive = sum(process.job.gpu == self.job.gpu and process.returncode is None
                                for process in processes)
                    observed_max[self.job.gpu] = max(observed_max[self.job.gpu], alive)
                    self.env = kwargs["env"]

                def poll(self):
                    self.polls += 1
                    delay = 5 if str(self.job.spec["job_id"]) in stragglers else 1
                    if self.returncode is None and self.polls >= delay:
                        artifact = self.job.child.root / f"{self.job.spec['job_id']}.checkpoint"
                        artifact.write_bytes(b"updated checkpoint")
                        subject.write_json(self.job.status_path, {
                            **self.job.spec, "status": "completed",
                            "artifacts": [{"artifact_role": "best_checkpoint", "path": str(artifact),
                                           "sha256": subject.sha256_file(artifact)}]})
                        self.returncode = 0
                    return self.returncode

                def wait(self, **kwargs):
                    return self.poll()

            snapshot = {"at_utc": "now", "gpu": {"0": {"memory_used_mib": 100},
                                                   "1": {"memory_used_mib": 100}},
                        "host_ram_fraction": .1}
            with patch.object(subject, "load_jobs", return_value=jobs), \
                    patch.object(subject, "resource_snapshot", return_value=snapshot), \
                    patch.object(subject.subprocess, "Popen", FakeProcess), \
                    patch.object(subject.time, "sleep"):
                result = subject.run_queue(children, root / "global", 2)
            self.assertEqual(result["completed"], 200)
            self.assertEqual(observed_max, {0: 2, 1: 2})
            self.assertTrue(replenished_during_straggler)
            self.assertEqual(len(processes), 200)
            for process in processes:
                self.assertEqual(process.env["CUDA_VISIBLE_DEVICES"], str(process.job.gpu))
            first_five = [process.job.child.family for process in processes if process.job.gpu == 0][:5]
            self.assertEqual(first_five, ["film", "film", "film", "film", "pure"])
            with patch.object(subject, "load_jobs", return_value=jobs):
                resumed = subject.run_queue(children, root / "global", 2, resume=True)
            self.assertEqual(resumed["completed"], 200)


if __name__ == "__main__":
    unittest.main()
