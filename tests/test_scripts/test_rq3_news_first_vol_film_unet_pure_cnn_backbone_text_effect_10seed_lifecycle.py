"""Focused fail-closed tests for the text-effect experiment lifecycle."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed as experiment,
)
from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle as lifecycle,
)
from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_analysis as analysis_module,
)


def _idle_snapshot() -> dict[str, object]:
    return {
        "gpus": [
            {
                "gpu_index": gpu,
                "gpu_uuid": f"GPU-{gpu}",
                "gpu_name": "A30",
                "memory_used_mib": 100.0,
                "memory_total_mib": 24_576.0,
                "utilization_gpu_pct": 0.0,
            }
            for gpu in (0, 1)
        ],
        "compute_processes": [],
        "host_memory": {"memory_used_fraction": 0.10},
        "cpu": {},
    }


def _busy_vllm_snapshot() -> dict[str, object]:
    value = _idle_snapshot()
    value["gpus"][0]["memory_used_mib"] = 21_000.0  # type: ignore[index]
    value["gpus"][0]["utilization_gpu_pct"] = 80.0  # type: ignore[index]
    value["compute_processes"] = [
        {
            "gpu_uuid": "GPU-0",
            "pid": 123_456,
            "process_name": "python",
            "used_memory_mib": 20_000.0,
            "command": "python -m vllm.entrypoints.openai.api_server",
            "protected_vllm": True,
        }
    ]
    return value


class TextEffectLifecycleTests(unittest.TestCase):
    def test_benchmark_manifest_uses_registry_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            benchmark_root = Path(directory) / "benchmark"
            for stage_root in experiment._stage_roots(benchmark_root).values():
                registry_path = stage_root / "registry/task_registry.json"
                registry_path.parent.mkdir(parents=True)
                registry_path.write_text('{"jobs": []}\n', encoding="utf-8")

            evidence = lifecycle._benchmark_stage_registry_evidence(benchmark_root)

            self.assertEqual(
                set(evidence),
                {"backbones", "pure_continuation", "film_continuations"},
            )
            for key, stage_root in experiment._stage_roots(benchmark_root).items():
                registry_path = stage_root / "registry/task_registry.json"
                self.assertEqual(evidence[key]["path"], str(registry_path.resolve()))
                self.assertEqual(
                    evidence[key]["sha256"],
                    experiment.sha256_file(registry_path),
                )

    def test_completed_benchmark_is_idempotent_after_formal_root_exists(self) -> None:
        """A partial formal pipeline must be able to revisit its benchmark stage."""

        with tempfile.TemporaryDirectory() as directory:
            formal_root = Path(directory) / "formal"
            formal_root.mkdir()
            manifest = Path(directory) / "benchmark" / "benchmark_manifest.json"
            lifecycle._write_signed(
                manifest,
                {
                    "schema_version": 1,
                    "kind": lifecycle.BENCHMARK_ROOT_KIND,
                    "status": "passed",
                },
            )
            result_path = (
                experiment._control_root(formal_root) / "benchmark_result.json"
            )
            lifecycle._write_signed(
                result_path,
                {
                    "schema_version": 1,
                    "kind": experiment.BENCHMARK_KIND,
                    "status": "passed",
                    "source_config_sha256": "a" * 64,
                    "benchmark_manifest_path": str(manifest.resolve()),
                    "benchmark_manifest_sha256": experiment.sha256_file(manifest),
                },
            )
            config = {
                "source_config_sha256": "a" * 64,
                "source_config_path": str(Path(directory) / "config.yaml"),
            }
            with (
                mock.patch.object(experiment, "load_config", return_value=config),
                mock.patch.object(experiment, "validate_config"),
            ):
                observed = lifecycle.benchmark("unused.yaml", formal_root, resume=True)
            self.assertEqual(observed, result_path.resolve())

    def test_capacity_failure_alone_enters_fallback_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            formal_root = Path(directory) / "formal"
            source = Path(directory) / "config.yaml"
            source.write_text("fixture: true\n", encoding="utf-8")
            config = {
                "source_config_path": str(source),
                "source_config_sha256": experiment.sha256_file(source),
                "runtime": {
                    "benchmark_workers_per_gpu": 10,
                    "fallback_workers_per_gpu": 6,
                    "disk_projection_safety_factor": 1.30,
                    "minimum_free_disk_after_projected_bytes_gib": 0,
                },
            }

            def candidate(
                _config: object,
                root: Path,
                *,
                workers: int,
                resume: bool,
            ) -> tuple[Path, dict[str, object]]:
                del resume
                candidate_root = root.with_name(f"{root.name}_{workers}")
                manifest = candidate_root / "benchmark_manifest.json"
                lifecycle._write_signed(
                    manifest,
                    {
                        "schema_version": 1,
                        "kind": lifecycle.BENCHMARK_ROOT_KIND,
                        "status": "failed" if workers == 10 else "passed",
                    },
                )
                if workers == 10:
                    return manifest, {
                        "passed": False,
                        "failure_category": "capacity",
                    }
                return manifest, {"passed": True, "failure_category": "none"}

            with (
                mock.patch.object(experiment, "load_config", return_value=config),
                mock.patch.object(experiment, "validate_config"),
                mock.patch.object(
                    lifecycle, "_run_benchmark_candidate", side_effect=candidate
                ) as run_candidate,
            ):
                result_path = lifecycle.benchmark("unused.yaml", formal_root)
            self.assertEqual(run_candidate.call_count, 2)
            payload = lifecycle._read_signed(
                result_path, kind=experiment.BENCHMARK_KIND
            )
            self.assertEqual(payload["selected_workers_per_gpu"], 6)

    def test_pipeline_resume_uses_partial_root_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            with mock.patch.object(
                lifecycle,
                "assess_resume_state",
                return_value={"state": "terminal_read_only"},
            ) as assess:
                self.assertEqual(lifecycle.run_pipeline("unused.yaml", root), root)
            validator = assess.call_args.kwargs["partial_validator"]
            with (
                mock.patch.object(
                    experiment, "validate_partial_root", return_value={}
                ) as validate_partial,
                mock.patch.object(
                    experiment,
                    "validate_root",
                    side_effect=AssertionError("full-root validator used"),
                ),
            ):
                validator(root)
            validate_partial.assert_called_once_with(root, verify_stage_roots=True)

    def test_pipeline_rejects_invalid_terminal_before_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            with (
                mock.patch.object(
                    lifecycle,
                    "assess_resume_state",
                    return_value={
                        "state": "invalid_terminal",
                        "reason": "qa hash drift",
                    },
                ),
                mock.patch.object(experiment, "load_config", return_value={}),
                mock.patch.object(lifecycle, "_wait_for_resources") as wait,
                self.assertRaisesRegex(RuntimeError, "invalid_terminal"),
            ):
                lifecycle.run_pipeline("unused.yaml", root, resume=True)
            wait.assert_not_called()

    def test_analysis_input_resume_repairs_registry_commit(self) -> None:
        """A crash after the signed input manifest must remain resumable."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            analysis_dir = root / "analysis"
            analysis_dir.mkdir(parents=True)
            sources: dict[str, Path] = {}
            for name in ("standard", "intervention", "trajectory"):
                path = analysis_dir / f"{name}.csv"
                pd.DataFrame({"value": [1]}).to_csv(path, index=False)
                sources[name] = path
            manifest_path = analysis_dir / "analysis_inputs.json"
            lifecycle._write_signed(
                manifest_path,
                {
                    "schema_version": 1,
                    "kind": lifecycle.ANALYSIS_INPUT_KIND,
                    "interpretation": experiment.INTERPRETATION,
                    "standard_panel_path": str(sources["standard"].resolve()),
                    "standard_panel_sha256": experiment.sha256_file(
                        sources["standard"]
                    ),
                    "standard_panel_rows": 1,
                    "intervention_panel_path": str(sources["intervention"].resolve()),
                    "intervention_panel_sha256": experiment.sha256_file(
                        sources["intervention"]
                    ),
                    "intervention_panel_rows": 1,
                    "validation_trajectory_path": str(sources["trajectory"].resolve()),
                    "validation_trajectory_sha256": experiment.sha256_file(
                        sources["trajectory"]
                    ),
                    "validation_trajectory_rows": 1,
                },
            )
            registry = {
                "analysis_inputs_frozen": False,
                "standard_predictions_frozen": True,
                "interventions_frozen": True,
                "validation_trajectories_frozen": True,
                "standard_pair_metrics_path": str(sources["standard"]),
                "intervention_pair_metrics_path": str(sources["intervention"]),
                "validation_trajectory_pair_metrics_path": str(sources["trajectory"]),
            }
            with (
                mock.patch.object(experiment, "validate_root"),
                mock.patch.object(experiment, "_read_registry", return_value=registry),
                mock.patch.object(experiment, "_write_registry") as write_registry,
            ):
                observed = lifecycle.analyze(root, resume=True)
            self.assertEqual(observed, manifest_path)
            write_registry.assert_called_once()
            committed = write_registry.call_args.args[1]
            self.assertTrue(committed["analysis_inputs_frozen"])
            self.assertEqual(
                committed["analysis_inputs_sha256"],
                experiment.sha256_file(manifest_path),
            )

    def test_completed_report_revalidates_nested_report_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            report_dir = root / "report"
            report_dir.mkdir(parents=True)
            markdown = report_dir / "report.md"
            html = report_dir / "report.html"
            markdown.write_text("original\n", encoding="utf-8")
            html.write_text("<p>original</p>\n", encoding="utf-8")
            artifact = report_dir / "report_artifact.json"
            experiment._write_json(
                artifact,
                {
                    "schema_version": 1,
                    "kind": (
                        "film_unet_pure_cnn_backbone_text_effect_report_artifact_v1"
                    ),
                    "interpretation": experiment.INTERPRETATION,
                    "conclusion_status": "fixture",
                    "reports": {
                        "markdown": {
                            "path": str(markdown.resolve()),
                            "size_bytes": markdown.stat().st_size,
                            "sha256": experiment.sha256_file(markdown),
                        },
                        "html": {
                            "path": str(html.resolve()),
                            "size_bytes": html.stat().st_size,
                            "sha256": experiment.sha256_file(html),
                            "self_contained": True,
                        },
                    },
                },
            )
            markdown.write_text("tampered\n", encoding="utf-8")
            analysis_manifest = root / "analysis/results/analysis_manifest.json"
            analysis_manifest.parent.mkdir(parents=True)
            analysis_manifest.write_text("{}\n", encoding="utf-8")
            registry = {
                "analysis_complete": True,
                "analysis_manifest_path": str(analysis_manifest),
                "analysis_manifest_sha256": experiment.sha256_file(analysis_manifest),
                "report_complete": True,
                "report_artifact_sha256": experiment.sha256_file(artifact),
            }
            with (
                mock.patch.object(experiment, "validate_root"),
                mock.patch.object(experiment, "_read_registry", return_value=registry),
                self.assertRaisesRegex(ValueError, "drift"),
            ):
                lifecycle.report(root, resume=True)

    def test_bootstrap_rejects_analysis_input_registry_sha_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            inputs_dir = root / "analysis"
            inputs_dir.mkdir(parents=True)
            panel = inputs_dir / "panel.csv"
            pd.DataFrame({"value": [1]}).to_csv(panel, index=False)
            inputs = lifecycle._write_signed(
                inputs_dir / "analysis_inputs.json",
                {
                    "schema_version": 1,
                    "kind": lifecycle.ANALYSIS_INPUT_KIND,
                    "standard_panel_path": str(panel),
                    "standard_panel_sha256": experiment.sha256_file(panel),
                    "intervention_panel_path": str(panel),
                    "intervention_panel_sha256": experiment.sha256_file(panel),
                    "validation_trajectory_path": str(panel),
                    "validation_trajectory_sha256": experiment.sha256_file(panel),
                },
            )
            registry = {
                "analysis_inputs_frozen": True,
                "analysis_inputs_path": str(inputs),
                "analysis_inputs_sha256": "0" * 64,
                "analysis_complete": False,
            }
            config = {
                "analysis": {
                    "bootstrap_replicates": 10_000,
                    "bootstrap_seed": 42,
                    "confidence_level": 0.95,
                    "primary_support_gate": {
                        "minimum_nonworse_seeds": 7,
                        "minimum_nonworse_folds": 3,
                    },
                }
            }
            with (
                mock.patch.object(experiment, "validate_root", return_value=config),
                mock.patch.object(experiment, "_read_registry", return_value=registry),
                mock.patch.object(
                    analysis_module,
                    "analyze_backbone_text_effect",
                    side_effect=AssertionError("statistics ran before SHA validation"),
                ) as analyze_result,
                self.assertRaisesRegex(ValueError, "Frozen file drift|analysis.*input"),
            ):
                lifecycle.bootstrap(root, resume=True)
            analyze_result.assert_not_called()

    def test_report_rejects_analysis_manifest_registry_sha_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            manifest = root / "analysis/results/analysis_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}\n", encoding="utf-8")
            registry = {
                "analysis_complete": True,
                "analysis_manifest_path": str(manifest),
                "analysis_manifest_sha256": "0" * 64,
                "report_complete": False,
            }
            with (
                mock.patch.object(experiment, "validate_root"),
                mock.patch.object(experiment, "_read_registry", return_value=registry),
                mock.patch.object(
                    lifecycle,
                    "_load_analysis_result",
                    side_effect=AssertionError("unbound analysis was loaded"),
                ) as load_result,
                self.assertRaisesRegex(
                    ValueError, "Frozen file drift|analysis.*manifest"
                ),
            ):
                lifecycle.report(root, resume=True)
            load_result.assert_not_called()

    def test_output_manifest_rejects_unlisted_root_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            listed = root / "listed.txt"
            listed.write_text("listed\n", encoding="utf-8")
            (root / "unlisted.txt").write_text("unlisted\n", encoding="utf-8")
            manifest = root / "output_hashes.csv"
            pd.DataFrame(
                [
                    {
                        "relative_path": "listed.txt",
                        "size_bytes": listed.stat().st_size,
                        "sha256": experiment.sha256_file(listed),
                    }
                ]
            ).to_csv(manifest, index=False)
            with self.assertRaisesRegex(ValueError, "unlisted|complete"):
                lifecycle._validate_output_manifest(root, manifest)

    def test_terminal_validation_rejects_false_qa_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            root.mkdir()
            snapshot = lifecycle._write_signed(
                root / "registry/final_registry_snapshot.json",
                {
                    "schema_version": 1,
                    "kind": "pure_cnn_parent_film_text_effect_final_registry_snapshot_v1",
                    "registry": {},
                },
            )
            qa_path = lifecycle._write_signed(
                root / "qa.json",
                {
                    "schema_version": 1,
                    "kind": lifecycle.TERMINAL_QA_KIND,
                    "status": "passed",
                    "training_jobs_completed": 279,
                    "standard_prediction_cells": 280,
                    "intervention_prediction_cells": 80,
                    "validation_trajectory_cells": 1_680,
                    "validation_trajectory_pair_rows": 210_420,
                    "standard_pair_rows": 35_000,
                    "intervention_pair_rows": 10_000,
                    "bootstrap_draws_per_comparison": 10_000,
                },
            )
            output = root / "output_hashes.csv"
            output.write_text("relative_path,size_bytes,sha256\n", encoding="utf-8")
            registry = {
                "terminal_complete": True,
                "status": "completed",
                "terminal_qa_path": str(qa_path),
                "terminal_qa_sha256": experiment.sha256_file(qa_path),
                "output_manifest_path": str(output),
                "output_manifest_sha256": experiment.sha256_file(output),
                "output_manifest_rows": 0,
                "final_registry_snapshot_path": str(snapshot),
                "final_registry_snapshot_sha256": experiment.sha256_file(snapshot),
            }
            with (
                mock.patch.object(experiment, "validate_root"),
                mock.patch.object(experiment, "_read_registry", return_value=registry),
                mock.patch.object(
                    lifecycle, "_validate_output_manifest", return_value=0
                ),
                self.assertRaisesRegex(ValueError, "QA.*count"),
            ):
                lifecycle.validate_terminal(root)

    def test_resource_wait_records_vllm_wait_then_idle_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            config = {"runtime": {"nvidia_smi_executable": "nvidia-smi"}}
            events: list[tuple[str, str]] = []

            def journal(_control: object, **kwargs: object) -> Path:
                events.append((str(kwargs["stage"]), str(kwargs["status"])))
                return Path(directory) / "journal.json"

            with (
                mock.patch.object(
                    lifecycle,
                    "system_resource_snapshot",
                    side_effect=(_busy_vllm_snapshot(), _idle_snapshot()),
                ),
                mock.patch.object(lifecycle, "atomic_write_json"),
                mock.patch.object(
                    lifecycle, "append_stage_journal", side_effect=journal
                ),
                mock.patch.object(lifecycle.time, "sleep"),
            ):
                lifecycle._wait_for_resources(config, root)
            self.assertEqual(
                events,
                [("resource_gate", "waiting"), ("resource_gate", "completed")],
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
