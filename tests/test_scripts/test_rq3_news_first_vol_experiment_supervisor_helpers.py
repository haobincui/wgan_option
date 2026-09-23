"""Focused tests for reusable RQ3 supervisor and resource helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.rq3 import news_first_vol_experiment_supervisor_helpers as helpers


def _snapshot(*, busy: bool = False, vllm: bool = False) -> dict[str, object]:
    used = 5000.0 if busy else 100.0
    utilization = 80.0 if busy else 0.0
    processes: list[dict[str, object]] = []
    if vllm:
        processes.append(
            {
                "gpu_uuid": "GPU-a",
                "pid": 987654,
                "process_name": "python",
                "used_memory_mib": 4096.0,
                "command": "python -m vllm.entrypoints.openai.api_server",
                "protected_vllm": True,
            }
        )
    return {
        "gpus": [
            {
                "gpu_index": 0,
                "gpu_uuid": "GPU-a",
                "gpu_name": "A30",
                "memory_used_mib": used,
                "memory_total_mib": 24576.0,
                "utilization_gpu_pct": utilization,
            },
            {
                "gpu_index": 1,
                "gpu_uuid": "GPU-b",
                "gpu_name": "A30",
                "memory_used_mib": 100.0,
                "memory_total_mib": 24576.0,
                "utilization_gpu_pct": 0.0,
            },
        ],
        "compute_processes": processes,
    }


class SupervisorHelperTests(unittest.TestCase):
    def test_parse_telemetry_and_protect_external_vllm(self) -> None:
        gpus = helpers.parse_gpu_rows(
            "0, GPU-a, NVIDIA A30, 512, 24576, 0\n1, GPU-b, NVIDIA A30, 256, 24576, 1\n"
        )
        processes = helpers.parse_compute_process_rows(
            "GPU-a, 77, python, 4096\n",
            command_lookup=lambda pid: (
                "python -m vllm.entrypoints.openai.api_server" if pid == 77 else ""
            ),
        )
        decision = helpers.gpu_availability_decision(
            {"gpus": gpus, "compute_processes": processes}
        )
        self.assertFalse(decision["ready"])
        self.assertIn("protected_external_vllm", decision["reasons"])
        self.assertEqual(decision["protected_vllm_processes"][0]["pid"], 77)
        self.assertEqual(
            decision["policy"], "wait_only_never_signal_or_preempt_external_processes"
        )

    def test_dual_gpu_wait_observes_vllm_then_returns_when_both_are_idle(self) -> None:
        snapshots = iter((_snapshot(busy=True, vllm=True), _snapshot()))
        clock = iter((0.0, 0.0, 5.0, 5.0))
        sleeps: list[float] = []
        result = helpers.wait_for_dual_gpu_availability(
            lambda: next(snapshots),
            timeout_seconds=30.0,
            poll_interval_seconds=5.0,
            monotonic=lambda: next(clock),
            sleeper=sleeps.append,
        )
        self.assertTrue(result["ready"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(sleeps, [5.0])

    def test_gpu_wait_times_out_without_preempting(self) -> None:
        clock = iter((0.0, 10.0))
        with self.assertRaisesRegex(TimeoutError, "protected_vllm=1"):
            helpers.wait_for_dual_gpu_availability(
                lambda: _snapshot(busy=True, vllm=True),
                timeout_seconds=10.0,
                poll_interval_seconds=5.0,
                monotonic=lambda: next(clock),
                sleeper=lambda _seconds: None,
            )

    def test_canary_gate_separates_capacity_from_correctness(self) -> None:
        base = {
            "completed_jobs": 4,
            "failed_jobs": 0,
            "oom_count": 0,
            "nan_count": 0,
            "generator_updated_jobs": 4,
            "critic_updated_jobs": 4,
            "peak_gpu_memory_gib_by_gpu": {0: 12.0, 1: 13.0},
            "peak_host_ram_fraction": 0.50,
        }
        passed = helpers.evaluate_canary_or_benchmark(base, expected_jobs=4)
        self.assertTrue(passed["passed"])
        capacity = helpers.evaluate_canary_or_benchmark(
            {**base, "peak_gpu_memory_gib_by_gpu": {0: 20.0, 1: 13.0}},
            expected_jobs=4,
        )
        self.assertEqual(capacity["failure_category"], "capacity")
        oom_capacity = helpers.evaluate_canary_or_benchmark(
            {
                **base,
                "completed_jobs": 3,
                "failed_jobs": 1,
                "oom_count": 1,
                "generator_updated_jobs": 3,
                "critic_updated_jobs": 3,
            },
            expected_jobs=4,
        )
        self.assertEqual(oom_capacity["failure_category"], "capacity")
        self.assertIn("oom", oom_capacity["capacity_reasons"])
        correctness = helpers.evaluate_canary_or_benchmark(
            {**base, "nan_count": 1}, expected_jobs=4
        )
        self.assertEqual(correctness["failure_category"], "correctness")

    def test_only_capacity_failure_can_enter_fallback_benchmark(self) -> None:
        passed = {"passed": True, "failure_category": "none"}
        capacity = {"passed": False, "failure_category": "capacity"}
        correctness = {"passed": False, "failure_category": "correctness"}
        self.assertEqual(
            helpers.benchmark_concurrency_decision(
                passed, primary_workers_per_gpu=8, fallback_workers_per_gpu=4
            )["selected_workers_per_gpu"],
            8,
        )
        self.assertEqual(
            helpers.benchmark_concurrency_decision(
                capacity, primary_workers_per_gpu=8, fallback_workers_per_gpu=4
            )["action"],
            "run_fallback_benchmark",
        )
        self.assertEqual(
            helpers.benchmark_concurrency_decision(
                capacity,
                primary_workers_per_gpu=8,
                fallback_workers_per_gpu=4,
                fallback=passed,
            )["selected_workers_per_gpu"],
            4,
        )
        self.assertEqual(
            helpers.benchmark_concurrency_decision(
                correctness, primary_workers_per_gpu=8, fallback_workers_per_gpu=4
            )["action"],
            "abort",
        )

    def test_exclusive_lock_pid_identity_and_journal_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "control"
            first = helpers.SupervisorLock(control)
            second = helpers.SupervisorLock(control)
            with first:
                with self.assertRaisesRegex(helpers.SupervisorHelperError, "lock"):
                    second.acquire()
                identity = helpers._read_signed_json(first.pid_path)
                self.assertEqual(identity["pid"], os.getpid())
                self.assertTrue(
                    helpers.pid_alive(
                        identity["pid"],
                        expected_start_ticks=identity["process_start_ticks"],
                    )
                )
                journal = helpers.append_stage_journal(
                    control,
                    experiment_kind="fixture",
                    stage="benchmark",
                    status="running",
                )
                helpers.append_stage_journal(
                    control,
                    experiment_kind="fixture",
                    stage="benchmark",
                    status="completed",
                )
                self.assertEqual(len(helpers._read_signed_json(journal)["events"]), 2)
            self.assertFalse(helpers.advisory_lock_is_held(first.lock_path))
            payload = json.loads(journal.read_text(encoding="utf-8"))
            payload["latest"]["status"] = "forged"
            journal.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(helpers.SupervisorHelperError, "hash drift"):
                helpers.append_stage_journal(
                    control,
                    experiment_kind="fixture",
                    stage="formal",
                    status="running",
                )

    def test_resume_state_is_fail_closed_and_terminal_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "experiment"
            control = base / "control"
            self.assertEqual(
                helpers.assess_resume_state(root, control, terminal_marker="qa.json")[
                    "state"
                ],
                "new",
            )
            root.mkdir()
            invalid = helpers.assess_resume_state(
                root, control, terminal_marker="qa.json"
            )
            self.assertEqual(invalid["state"], "invalid_partial")
            resumable = helpers.assess_resume_state(
                root,
                control,
                terminal_marker="qa.json",
                partial_validator=lambda path: self.assertEqual(path, root.resolve()),
            )
            self.assertEqual(resumable["state"], "resumable_partial")
            (root / "qa.json").write_text("{}\n", encoding="utf-8")
            terminal = helpers.assess_resume_state(
                root,
                control,
                terminal_marker="qa.json",
                terminal_validator=lambda path: self.assertEqual(path, root.resolve()),
            )
            self.assertEqual(terminal["state"], "terminal_read_only")
            self.assertFalse(terminal["may_write"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
