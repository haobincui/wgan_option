from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts.rq3 import news_first_vol_film_unet_text_signal_probe as probe


class TextSignalProbeOrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = probe.load_config()

    def test_cli_action_universe_is_narrow_and_has_no_prediction_action(self) -> None:
        self.assertEqual(
            probe.ACTION_NAMES,
            (
                "prepare",
                "canary",
                "launch-screen",
                "analyze-screen",
                "launch-confirmation",
                "report",
                "qa",
                "status",
                "run-pipeline",
            ),
        )
        self.assertNotIn("predict", probe.ACTION_NAMES)

    def test_screen_is_complete_binary_factorial_and_gpu_balanced(self) -> None:
        rows = probe.build_screen_designs(self.config)
        self.assertEqual(len(rows), 16)
        self.assertEqual(
            {str(row["factor_code"]) for row in rows},
            set(probe.SCREEN_FACTOR_CODES),
        )
        self.assertEqual(len({str(row["job_id"]) for row in rows}), 16)
        self.assertEqual(
            {
                gpu: sum(int(row["physical_gpu_id"]) == gpu for row in rows)
                for gpu in (0, 1)
            },
            {0: 8, 1: 8},
        )
        self.assertTrue(all(int(row["max_epochs"]) == 60 for row in rows))
        self.assertTrue(all(row["text_assignment"] == "matched" for row in rows))

    def test_confirmation_matrix_is_conditional_on_winner_spatial_factor(self) -> None:
        nonspatial = probe.build_confirmation_designs(
            self.config,
            {"confirmation_eligible": True, "winner_factor_code": "1110"},
        )
        spatial = probe.build_confirmation_designs(
            self.config,
            {"confirmation_eligible": True, "winner_factor_code": "1111"},
        )
        self.assertEqual(len(nonspatial), 3)
        self.assertEqual(len(spatial), 4)
        self.assertEqual(
            [row["confirmation_role"] for row in spatial],
            [
                "baseline_0000_matched",
                "winner_matched",
                "winner_independent_shuffle",
                "winner_text328_global_capacity_control",
            ],
        )
        capacity = spatial[-1]
        self.assertEqual(capacity["factor_code"], "1110")
        self.assertEqual(capacity["text_output_dimension"], 328)
        self.assertTrue(capacity["capacity_matched_control"])
        self.assertTrue(all(int(row["max_epochs"]) == 240 for row in spatial))

    def test_no_winner_creates_no_confirmation_jobs(self) -> None:
        rows = probe.build_confirmation_designs(
            self.config,
            {"confirmation_eligible": False, "winner_factor_code": None},
        )
        self.assertEqual(rows, [])

    def test_config_rejects_any_test_partition_bound(self) -> None:
        invalid = deepcopy(self.config)
        invalid["data"]["split"]["test_start_utc"] = "2023-04-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "train/validation bounds only"):
            probe.validate_config(invalid)

    def test_result_verifier_binds_epoch_factor_assignment_and_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "metric.csv"
            artifact.write_text("value\n1\n", encoding="utf-8")
            spec = {
                "job_id": "job",
                "job_spec_sha256": "spec-sha",
                "max_epochs": 60,
                "factor_code": "1010",
                "text_assignment": "matched",
                "input_manifest_sha256": "input-sha",
            }
            result = {
                "schema_version": 1,
                "kind": probe.RESULT_KIND,
                "status": "completed",
                "job_id": "job",
                "job_spec_sha256": "spec-sha",
                "epochs": 60,
                "factor_code": "1010",
                "assignment": "matched",
                "input_manifest_sha256": "input-sha",
                "allowed_partitions": ["train", "validation"],
                "test_loader_count": 0,
                "prediction_count": 0,
                "artifacts": [probe._file_row("metric", artifact)],
            }
            probe._write_json(root / "result.json", result)
            self.assertEqual(probe._verify_result(spec, root), result)
            result["epochs"] = 59
            probe._write_json(root / "result.json", result)
            with self.assertRaisesRegex(ValueError, "epochs drift"):
                probe._verify_result(spec, root)


if __name__ == "__main__":
    unittest.main()
