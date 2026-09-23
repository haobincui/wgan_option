"""Contracts for the ungated low-projection-LR FiLM supplement."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.rq3 import news_first_vol_film_unet_text_signal_probe as base
from scripts.rq3 import news_first_vol_film_unet_ungated_proj_lr5e7_probe as probe


class UngatedProjectionLRProbeTests(unittest.TestCase):
    def test_frozen_config_has_exact_one_cell_design(self) -> None:
        config = probe.load_config()
        self.assertEqual(config["training"]["job_id"], probe.JOB_ID)
        self.assertEqual(config["training"]["epochs"], 60)
        self.assertEqual(config["training"]["physical_gpu_id"], 0)
        self.assertEqual(config["training"]["text_encoder_learning_rate"], 2.5e-6)
        self.assertEqual(config["training"]["global_film_learning_rate"], 5.0e-7)
        self.assertFalse(config["model_contract"]["gate_enabled"])
        self.assertEqual(config["model_contract"]["generator_parameters"], 827_745)
        self.assertEqual(
            config["data_contract"]["allowed_partitions"], ["train", "validation"]
        )
        self.assertEqual(config["data_contract"]["test_loader_count"], 0)
        self.assertEqual(config["data_contract"]["q3_q4_input_rows"], 0)

    def test_materialized_spec_uses_grouped_optimizer_without_gate(self) -> None:
        config = probe.load_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = probe._build_record(root, config)
            spec = base._verify_job_spec(
                record["job_spec_path"], record["job_spec_sha256"]
            )
        self.assertEqual(spec["job_id"], probe.JOB_ID)
        self.assertEqual(spec["factor_code"], "1000")
        self.assertEqual(spec["max_epochs"], 60)
        self.assertEqual(spec["text_encoder_learning_rate"], 2.5e-6)
        self.assertEqual(spec["global_film_learning_rate"], 5.0e-7)
        self.assertEqual(spec["expected_generator_parameters"], 827_745)
        self.assertEqual(spec["allowed_partitions"], ["train", "validation"])
        self.assertNotIn("global_residual_gate_initial", spec)
        self.assertNotIn("global_residual_gate_max", spec)
        self.assertNotIn("global_residual_gate_freeze_epochs", spec)
        self.assertNotIn("global_gate_learning_rate", spec)

    def test_descriptive_retention_classifier_has_frozen_boundaries(self) -> None:
        classify = probe._classify_retention
        kwargs = {"low_lr_maximum": 0.1, "gate_minimum": 0.5}
        self.assertEqual(
            classify(0.0, **kwargs),
            "low_projection_lr_is_primary_text_suppressor",
        )
        self.assertEqual(
            classify(0.1, **kwargs),
            "low_projection_lr_is_primary_text_suppressor",
        )
        self.assertEqual(classify(0.25, **kwargs), "mixed_or_ambiguous")
        self.assertEqual(classify(0.5, **kwargs), "gate_is_primary_text_suppressor")
        with self.assertRaises(ValueError):
            classify(math.nan, **kwargs)

    def test_status_before_prepare_has_no_evaluation_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            value = probe.status(Path(temporary) / "not-created")
        self.assertEqual(value["status"], "not_prepared")
        self.assertEqual(value["completed_jobs"], 0)
        self.assertEqual(value["total_jobs"], 1)

    def test_completed_registry_resume_is_strictly_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            root.mkdir()
            probe._write_registry(
                root,
                {
                    "schema_version": 1,
                    "kind": probe.REGISTRY_KIND,
                    "status": "completed",
                },
            )
            base._output_hash_manifest(root)
            before = {
                path.relative_to(root): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            completed = probe._completed_registry(root, resume=True, action="unit-test")
            after = {
                path.relative_to(root): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(before, after)
            with self.assertRaises(FileExistsError):
                probe._completed_registry(root, resume=False, action="unit-test")

    def test_canary_rejects_a_stale_nonmatching_formal_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            probe.prepare(output_dir=root)
            registry = probe._read_registry(root)
            formal = base._verify_job_spec(
                registry["job"]["job_spec_path"],
                registry["job"]["job_spec_sha256"],
            )
            canary_root = base._control_dir(root) / "canary"
            output_dir = canary_root / "runs" / f"canary_{probe.JOB_ID}"
            stale = {
                **formal,
                "job_id": f"canary_{probe.JOB_ID}",
                "stage": "canary",
                "max_epochs": 1,
                "output_dir": str(output_dir.resolve()),
                "global_film_learning_rate": 1.0e-7,
            }
            stale.pop("job_spec_sha256", None)
            spec_path, spec_sha = base._write_job_spec(
                canary_root / "job_specs/stale.json", stale
            )
            canary_record = {
                **registry["job"],
                "job_id": f"canary_{probe.JOB_ID}",
                "stage": "canary",
                "job_spec_path": str(spec_path.resolve()),
                "job_spec_sha256": spec_sha,
                "output_dir": str(output_dir.resolve()),
            }
            result_path = base._write_json(
                base._control_dir(root) / "canary_result.json",
                base._signed_payload(
                    {
                        "status": "passed",
                        "canary_root": str(canary_root.resolve()),
                        "job": canary_record,
                    }
                ),
            )
            with mock.patch.object(base, "_completed_valid", return_value=True):
                with self.assertRaisesRegex(ValueError, "one-epoch formal-spec"):
                    probe._verify_canary(root, result_path)

    def test_report_rejects_registry_analysis_sha_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            probe.prepare(output_dir=root)
            analysis_path = base._write_json(
                root / "analysis/ungated_proj_lr5e7_analysis.json",
                base._signed_payload({"kind": "synthetic_analysis"}),
            )
            registry = probe._read_registry(root)
            registry.update(
                status="analyzed",
                analysis_path=str(analysis_path.resolve()),
                analysis_sha256="0" * 64,
            )
            probe._write_registry(root, registry)
            with self.assertRaisesRegex(ValueError, "analysis binding drift"):
                probe.report(output_dir=root, resume=True)


if __name__ == "__main__":
    unittest.main()
