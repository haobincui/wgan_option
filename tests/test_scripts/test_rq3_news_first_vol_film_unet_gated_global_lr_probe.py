"""Contracts for the three-cell gated Global-FiLM LR probe."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.rq3 import news_first_vol_film_unet_gated_global_lr_probe as probe
from scripts.rq3 import news_first_vol_film_unet_text_signal_probe as base


class GatedGlobalLRProbeTests(unittest.TestCase):
    def test_frozen_config_has_exact_three_cell_design(self) -> None:
        config = probe.load_config()
        jobs = config["training"]["jobs"]
        self.assertEqual(
            [row["job_id"] for row in jobs],
            [
                "control_1000_uniform_2p5e6",
                "gated_global_proj_lr5e7",
                "gated_global_proj_lr2p5e7",
            ],
        )
        self.assertEqual([row["physical_gpu_id"] for row in jobs], [0, 1, 0])
        self.assertEqual(config["model_contract"]["gate_freeze_epochs"], 10)
        self.assertEqual(
            config["data_contract"]["allowed_partitions"], ["train", "validation"]
        )
        self.assertEqual(config["data_contract"]["test_loader_count"], 0)
        self.assertEqual(config["data_contract"]["q3_q4_input_rows"], 0)

    def test_materialized_specs_are_unique_and_bind_frozen_inputs(self) -> None:
        config = probe.load_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = probe._build_job_records(root, config)
            self.assertEqual(len(records), 3)
            self.assertEqual(len({row["job_id"] for row in records}), 3)
            specs = [
                base._verify_job_spec(row["job_spec_path"], row["job_spec_sha256"])
                for row in records
            ]
        self.assertTrue(all(spec["factor_code"] == "1000" for spec in specs))
        self.assertTrue(all(spec["max_epochs"] == 60 for spec in specs))
        self.assertTrue(
            all(spec["allowed_partitions"] == ["train", "validation"] for spec in specs)
        )
        control, gated_5, gated_2_5 = specs
        self.assertNotIn("global_residual_gate_initial", control)
        self.assertEqual(control["expected_generator_parameters"], 827_745)
        for spec in (gated_5, gated_2_5):
            self.assertEqual(spec["global_residual_gate_initial"], 0.01)
            self.assertEqual(spec["global_residual_gate_max"], 0.1)
            self.assertEqual(spec["global_residual_gate_freeze_epochs"], 10)
            self.assertEqual(spec["expected_generator_parameters"], 827_751)
        self.assertEqual(gated_5["global_film_learning_rate"], 5.0e-7)
        self.assertEqual(gated_2_5["global_film_learning_rate"], 2.5e-7)

    def test_status_before_prepare_has_no_evaluation_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            value = probe.status(Path(temporary) / "not-created")
        self.assertEqual(value["status"], "not_prepared")
        self.assertEqual(value["completed_jobs"], 0)
        self.assertEqual(value["total_jobs"], 3)


if __name__ == "__main__":
    unittest.main()
