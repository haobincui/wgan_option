"""Contract tests for the RQ4 fold-pooled prediction worker."""

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
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_parallel_prediction as parallel,
)
from scripts.rq3 import news_first_vol_rq4_fold_pooled_prediction_worker as worker


class RQ4FoldPooledPredictionWorkerTests(unittest.TestCase):
    def test_noise_profile_is_order_independent_and_model_independent(self) -> None:
        first = worker.noise_bank_profile_sha256(
            seed=42,
            fold="f1_2023q1",
            sample_ids=["pair::b", "pair::a"],
        )
        second = worker.noise_bank_profile_sha256(
            seed=42,
            fold="f1_2023q1",
            sample_ids=["pair::a", "pair::b"],
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_noise_profile_rejects_duplicates_or_nonformal_draw_count(self) -> None:
        with self.assertRaises(worker.RQ4FoldPooledPredictionError):
            worker.noise_bank_profile_sha256(
                seed=42,
                fold="f1_2023q1",
                sample_ids=["pair::a", "pair::a"],
            )
        with self.assertRaises(worker.RQ4FoldPooledPredictionError):
            worker.noise_bank_profile_sha256(
                seed=42,
                fold="f1_2023q1",
                sample_ids=["pair::a"],
                draws=16,
            )

    def test_new_prediction_kind_is_accepted_by_generic_scheduler(self) -> None:
        unit = {
            "prediction_unit_id": "rq4::film::42::f1",
            "prediction_kind": worker.PREDICTION_KIND,
            "seed": 42,
            "fold": "f1_2023q1",
            "noise_bank_namespace": "rq4/f1/42",
            "noise_bank_profile_sha256": "a" * 64,
        }
        contract = parallel._unit_contract(unit)
        self.assertEqual(
            contract["scientific_unit"]["prediction_kind"], "rq4_fold_pooled"
        )

    def test_device_isolation_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            worker.RQ4FoldPooledPredictionError, "logical cuda:0"
        ):
            worker._validate_device(1)
        with (
            patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "0,1", "RQ3_LOGICAL_CUDA_DEVICE": "0"},
                clear=False,
            ),
            self.assertRaisesRegex(worker.RQ4FoldPooledPredictionError, "exactly one"),
        ):
            worker._validate_device(0)

    def test_import_does_not_import_torch(self) -> None:
        module = "scripts.rq3.news_first_vol_rq4_fold_pooled_prediction_worker"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib,sys; "
                f"importlib.import_module({module!r}); "
                "print(int('torch' in sys.modules))",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "0")

    def test_complete_manifest_detects_metric_tampering(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            panel = root / "panel.csv.gz"
            checkpoint.write_bytes(b"checkpoint")
            pd.DataFrame({"sample_id": ["pair::a"]}).to_csv(
                panel, index=False, compression="gzip"
            )
            cell = root / "cell"
            cell.mkdir()
            unit = {
                "prediction_unit_id": "fixture",
                "prediction_kind": worker.PREDICTION_KIND,
                "model_name": "film_cnn",
                "source_arm": "lp_matched",
                "seed": 42,
                "fold": "f1_2023q1",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": worker.sha256_file(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "panel_path": str(panel),
                "panel_sha256": worker.sha256_file(panel),
                "panel_row_count": 1,
                "panel_pair_count": 1,
                "panel_session_count": 1,
                "artifact_root": str(root),
                "cell_output_dir": str(cell),
                "noise_bank_profile_sha256": "a" * 64,
                "mc_samples": 64,
            }
            prediction_path, metrics_path, manifest_path = worker._artifact_paths(unit)
            pd.DataFrame(
                {"sample_id": ["pair::a"], "predicted_surface_flat": ["[0]"]}
            ).to_csv(prediction_path, index=False, compression="gzip")
            pd.DataFrame(
                {"pair_id": ["a"], "target_mae": [0.1], "persistence_mae": [0.2]}
            ).to_csv(metrics_path, index=False, compression="gzip")
            manifest = worker._manifest_payload(unit, prediction_path, metrics_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            worker._validate_complete_bundle(
                unit, prediction_path, metrics_path, manifest_path
            )
            pd.DataFrame(
                {"pair_id": ["a"], "target_mae": [9.9], "persistence_mae": [0.2]}
            ).to_csv(metrics_path, index=False, compression="gzip")
            with self.assertRaisesRegex(
                worker.RQ4FoldPooledPredictionError, "lineage drift"
            ):
                worker._validate_complete_bundle(
                    unit, prediction_path, metrics_path, manifest_path
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
