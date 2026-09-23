"""CPU/mock tests for the spawn-safe text-effect prediction callback."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_prediction_supervisor as runner,
)
from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_prediction_worker as worker,
)


def _unit(root: Path, checkpoint: Path, kind: str) -> dict[str, object]:
    arm = (
        "pure_cnn_continue_no_text"
        if kind == "validation_trajectory"
        else "film_lp_matched"
    )
    payload: dict[str, object] = {
        "prediction_unit_id": f"fixture::{kind}",
        "prediction_kind": kind,
        "job_id": f"fixture-job::{kind}",
        "experiment_root": str(root),
        "seed": 42,
        "fold": "f2_2023q2",
        "arm": arm,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": worker.sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "noise_bank_namespace": "fixture/f2/42",
        "noise_bank_profile_sha256": "a" * 64,
        "expected_artifact_roles": ["pair_metrics"],
    }
    if kind == "matched_checkpoint_intervention":
        payload.update(input_condition="zero_input", input_overlay_arm="film_zero_text")
    if kind == "validation_trajectory":
        payload.update(
            job_id="continuation-fixture",
            parent_job_id="parent-fixture",
            checkpoint_label="epoch_1",
            epoch=1,
            parent_checkpoint_path=str(checkpoint),
            parent_checkpoint_sha256=worker.sha256_file(checkpoint),
        )
    return payload


class PredictionWorkerTests(unittest.TestCase):
    def test_all_three_prediction_kinds_dispatch_and_return_bound_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "generator.pt"
            checkpoint.write_bytes(b"checkpoint")
            artifact = root / "cell.csv"
            artifact.write_text("pair_id,target_mae\na,0.1\n", encoding="utf-8")
            dispatch = {
                "standard_test": "_run_standard_test",
                "matched_checkpoint_intervention": "_run_matched_intervention",
                "validation_trajectory": "_run_validation_trajectory",
            }
            for kind, function_name in dispatch.items():
                with self.subTest(kind=kind):
                    fake = ({"pair_metrics": artifact}, {"fixture": kind})
                    with (
                        patch.dict(
                            os.environ,
                            {
                                "CUDA_VISIBLE_DEVICES": "7",
                                "RQ3_LOGICAL_CUDA_DEVICE": "0",
                            },
                            clear=False,
                        ),
                        patch.object(worker, function_name, return_value=fake) as call,
                    ):
                        result = worker.run_prediction_unit(
                            unit=_unit(root, checkpoint, kind),
                            logical_device=0,
                            resume=True,
                        )
                    call.assert_called_once()
                    self.assertEqual(result["noise_bank_profile_sha256"], "a" * 64)
                    self.assertEqual(result["metadata"]["logical_cuda_device"], 0)
                    self.assertEqual(result["metadata"]["fixture"], kind)
                    self.assertEqual(len(result["artifacts"]), 1)
                    self.assertEqual(result["artifacts"][0]["role"], "pair_metrics")
                    self.assertEqual(
                        result["artifacts"][0]["sha256"], worker.sha256_file(artifact)
                    )

    def test_nonzero_logical_device_or_multiple_visible_devices_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "generator.pt"
            checkpoint.write_bytes(b"checkpoint")
            unit = _unit(root, checkpoint, "standard_test")
            with self.assertRaisesRegex(
                worker.TextEffectPredictionWorkerError, "logical cuda:0"
            ):
                worker.run_prediction_unit(unit=unit, logical_device=1, resume=False)
            with (
                patch.dict(
                    os.environ,
                    {
                        "CUDA_VISIBLE_DEVICES": "0,1",
                        "RQ3_LOGICAL_CUDA_DEVICE": "0",
                    },
                    clear=False,
                ),
                self.assertRaisesRegex(
                    worker.TextEffectPredictionWorkerError, "exactly one"
                ),
            ):
                worker.run_prediction_unit(unit=unit, logical_device=0, resume=False)

    def test_partial_bundle_state_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.write_text("present", encoding="utf-8")
            self.assertEqual(worker._existing_bundle_state((first, second)), "partial")
            second.write_text("present", encoding="utf-8")
            self.assertEqual(worker._existing_bundle_state((first, second)), "complete")

    def test_runner_loads_json_and_csv_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "units.json"
            json_path.write_text(
                json.dumps({"units": [{"prediction_unit_id": "json-cell"}]}),
                encoding="utf-8",
            )
            self.assertEqual(
                runner.load_units(json_path)[0]["prediction_unit_id"], "json-cell"
            )
            csv_path = root / "units.csv"
            csv_path.write_text(
                "prediction_unit_id,seed,expected_artifact_roles\n"
                'csv-cell,42,"[""pair_metrics""]"\n',
                encoding="utf-8",
            )
            row = runner.load_units(csv_path)[0]
            self.assertEqual(row["seed"], 42)
            self.assertEqual(row["expected_artifact_roles"], ["pair_metrics"])

    def test_thin_runner_and_callback_imports_do_not_import_torch(self) -> None:
        modules = (
            "scripts.rq3."
            "news_first_vol_film_unet_pure_cnn_backbone_text_effect_prediction_supervisor",
            "scripts.rq3."
            "news_first_vol_film_unet_pure_cnn_backbone_text_effect_prediction_worker",
        )
        code = (
            "import importlib,sys; "
            f"[importlib.import_module(x) for x in {modules!r}]; "
            "print(int('torch' in sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "0")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
