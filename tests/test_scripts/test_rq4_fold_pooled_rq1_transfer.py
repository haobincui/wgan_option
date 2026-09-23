"""Tests for the RQ4 fold-pooled RQ1 transfer orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.rq3 import news_first_vol_rq4_fold_pooled_rq1_transfer as transfer


class RQ4FoldPooledTransferTests(unittest.TestCase):
    def test_formal_config_locks_models_seeds_folds_and_runtime(self) -> None:
        config, path = transfer.load_config()
        self.assertTrue(path.is_file())
        self.assertEqual(tuple(config["models"]), transfer.CANONICAL_MODELS)
        self.assertEqual(tuple(config["matrix"]["seeds"]), transfer.CANONICAL_SEEDS)
        self.assertEqual(tuple(config["matrix"]["folds"]), transfer.CANONICAL_FOLDS)
        self.assertEqual(config["prediction"]["mc_samples"], 64)
        self.assertEqual(config["runtime"]["gpu_ids"], [0, 1])
        self.assertEqual(config["runtime"]["workers_per_gpu"], 4)

    def test_prediction_units_are_exact_and_share_noise_by_seed_fold(self) -> None:
        config, _ = transfer.load_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            panels: dict[str, pd.DataFrame] = {}
            paths: dict[str, Path] = {}
            for fold in transfer.CANONICAL_FOLDS:
                panel = pd.DataFrame(
                    {
                        "sample_id": [f"pair::{fold}::a", f"pair::{fold}::b"],
                        "pair_id": [f"{fold}::a", f"{fold}::b"],
                        "session_id": [f"{fold}::s1", f"{fold}::s2"],
                    }
                )
                path = root / f"{fold}.csv"
                panel.to_csv(path, index=False)
                panels[fold] = panel
                paths[fold] = path
            rows = []
            for model in transfer.CANONICAL_MODELS:
                for seed in transfer.CANONICAL_SEEDS:
                    for fold in transfer.CANONICAL_FOLDS:
                        rows.append(
                            {
                                "model": model,
                                "effective_text_use": model == "film_cnn",
                                "source_arm": (
                                    "lp_matched"
                                    if model == "film_cnn"
                                    else "pure_cnn_no_text"
                                ),
                                "source_job_id": f"source::{model}::{seed}::{fold}",
                                "seed": seed,
                                "checkpoint_fold": fold,
                                "checkpoint_path": str(root / "unused.pt"),
                                "checkpoint_sha256": "a" * 64,
                                "checkpoint_size_bytes": 1,
                                "source_root": str(root),
                                "source_qa_path": str(root / "qa.json"),
                                "source_qa_sha256": "b" * 64,
                            }
                        )
            units = transfer._prediction_units(
                config,
                root,
                pd.DataFrame(rows),
                {"panels": panels, "paths": paths},
            )
            self.assertEqual(len(units), 80)
            profiles = {}
            for unit in units:
                key = (unit["seed"], unit["fold"])
                profiles.setdefault(key, set()).add(unit["noise_bank_profile_sha256"])
            self.assertEqual(len(profiles), 40)
            self.assertTrue(all(len(values) == 1 for values in profiles.values()))
            self.assertEqual(
                {unit["prediction_kind"] for unit in units}, {"rq4_fold_pooled"}
            )

    def test_signed_unit_manifest_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "units.json"
            units = [
                {
                    "prediction_unit_id": f"cell-{index}",
                    "seed": transfer.CANONICAL_SEEDS[index // 8],
                    "fold": transfer.CANONICAL_FOLDS[(index // 2) % 4],
                }
                for index in range(80)
            ]
            transfer._unit_manifest(path, units)
            self.assertEqual(len(transfer._read_units(path)), 80)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["units"][0]["seed"] = -1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                transfer.RQ4FoldPooledTransferError, "payload SHA drift"
            ):
                transfer._read_units(path)

    def test_status_is_pending_for_an_unstarted_override_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = transfer.status(output_dir=temporary)
        self.assertFalse(result["prepared"])
        self.assertEqual(result["completed_prediction_tasks"], 0)
        self.assertEqual(result["qa_status"], "pending")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
