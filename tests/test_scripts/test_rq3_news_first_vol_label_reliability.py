from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from scripts.rq3.news_first_vol_label_reliability import (
    DEFAULT_CONFIG,
    FOLDS,
    PROFILE_FILE_COLUMNS,
    LabelReliabilityExperimentError,
    _prepare_bootstrap_pair_manifests,
    _resolved_config,
    _training_payload,
    _validate_frozen_config,
    build_full_job_matrix,
    canonical_profile_sha256,
    materialize_reliability_profiles,
    train_pair_universe_sha256,
)


class LabelReliabilityOrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = Path(DEFAULT_CONFIG).resolve()
        cls.raw_config = yaml.safe_load(cls.config_path.read_text(encoding="utf-8"))[
            "news_first_vol_training"
        ]

    def test_frozen_config_and_drift_rejection(self) -> None:
        _validate_frozen_config(self.raw_config)
        bad = deepcopy(self.raw_config)
        bad["label_reliability_experiment"]["initial_learning_rate"] = 1.0e-6
        with self.assertRaisesRegex(LabelReliabilityExperimentError, "LR is frozen"):
            _validate_frozen_config(bad)
        bad = deepcopy(self.raw_config)
        bad["label_reliability_experiment"]["folds"][3][
            "validation_end_utc"
        ] = "2023-10-01T00:00:00Z"
        with self.assertRaisesRegex(LabelReliabilityExperimentError, "folds"):
            _validate_frozen_config(bad)

    def test_conditional_matrix_has_48_plus_24_plus_24_unique_jobs(self) -> None:
        jobs = build_full_job_matrix("C", gpu_ids=(0, 1), slots_per_gpu=24)
        self.assertEqual(len(jobs), 96)
        self.assertEqual(len({row["job_id"] for row in jobs}), 96)
        counts = pd.Series([row["stage_id"] for row in jobs]).value_counts().to_dict()
        self.assertEqual(
            counts,
            {
                "stage1_regression_05m": 48,
                "stage2_regression_30m": 24,
                "stage3_wgan_05m": 24,
            },
        )
        for stage in counts:
            stage_jobs = [row for row in jobs if row["stage_id"] == stage]
            gpu_counts = pd.Series([row["gpu_id"] for row in stage_jobs]).value_counts()
            self.assertLessEqual(abs(int(gpu_counts[0]) - int(gpu_counts[1])), 1)
        jobs_12 = build_full_job_matrix("B", gpu_ids=(2, 3), slots_per_gpu=12)
        stage1_waves = {
            row["wave"] for row in jobs_12 if row["stage_id"] == "stage1_regression_05m"
        }
        self.assertEqual(len(stage1_waves), 2)

    def test_profile_hash_and_train_universe_hash_are_order_stable(self) -> None:
        frame = pd.DataFrame(
            {
                "tolerance_minutes": [5, 5],
                "fold_id": ["F1", "F1"],
                "pair_id": ["p2", "p1"],
                "included": [False, True],
                "normalized_label_weight": [0.0, 1.0],
                "reliability_score": [0.2, 0.8],
            }
        )
        self.assertEqual(
            canonical_profile_sha256(frame),
            canonical_profile_sha256(frame.iloc[::-1].reset_index(drop=True)),
        )
        self.assertEqual(
            train_pair_universe_sha256(["p2", "p1", "p1"], fold_id="F1", tolerance_minutes=5),
            train_pair_universe_sha256(["p1", "p2"], fold_id="F1", tolerance_minutes=5),
        )

    def test_formula_profiles_are_bounded_mean_one_and_retention_gated(self) -> None:
        metrics = []
        for tolerance in (5, 30):
            for index in range(20):
                metrics.append(
                    {
                        "pair_id": f"p{index:02d}",
                        "tolerance_minutes": tolerance,
                        "session_id": f"s{index:02d}",
                        "effective_origin_utc": "2022-01-03T14:00:00Z",
                        "c_i": 25 if index % 2 else 9,
                        "q_i": 8,
                        "m_i": 4,
                        "u_i": 1.0e-5 * (index + 1),
                        "u_estimable": True,
                        "v_i": 0.9,
                        "h_i": 0.8,
                    }
                )
        pair_metrics = pd.DataFrame(metrics)

        def universe(*_args, **_kwargs):
            return pd.DataFrame(
                {
                    "pair_id": [f"p{index:02d}" for index in range(20)],
                    "session_id": [f"s{index:02d}" for index in range(20)],
                    "row_count": 1,
                }
            )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.rq3.news_first_vol_label_reliability._read_training_pair_universe",
            side_effect=universe,
        ):
            manifest = materialize_reliability_profiles(pair_metrics, {}, tmp)
            self.assertEqual(len(manifest), 32)
            b_rows = manifest[manifest["arm_id"] == "B"]
            self.assertFalse(b_rows["retention_gate_valid"].all())
            c_row = manifest[
                (manifest["arm_id"] == "C")
                & (manifest["tolerance_minutes"] == 5)
                & (manifest["fold_id"] == "F1")
            ].iloc[0]
            profile = pd.read_csv(c_row["profile_path"])
            self.assertEqual(tuple(profile.columns), PROFILE_FILE_COLUMNS)
            included = profile[profile["included"].astype(bool)]
            self.assertAlmostEqual(included["normalized_label_weight"].mean(), 1.0)
            self.assertGreaterEqual(included["normalized_label_weight"].min(), 0.5)
            self.assertLessEqual(included["normalized_label_weight"].max(), 2.0)
            self.assertEqual(
                set(profile["train_pair_universe_sha256"]),
                {c_row["train_pair_universe_sha256"]},
            )

    def test_training_payload_forbids_test_loader_and_binds_three_hashes(self) -> None:
        resolved = _resolved_config(self.config_path)
        profile = SimpleNamespace(
            profile_path="/tmp/profile.csv",
            manifest_sha256="a" * 64,
            profile_sha256="b" * 64,
            train_pair_universe_sha256="c" * 64,
        )
        job = {
            "job_id": "stage1_regression_05m_regression_a_f1_seed_042_05m",
            "stage_id": "stage1_regression_05m",
            "model_family": "regression",
            "arm_id": "A",
            "fold_id": "F1",
            "seed": 42,
            "tolerance_minutes": 5,
        }
        payload = _training_payload(resolved, Path("/tmp/label-root"), job, profile)
        self.assertIs(payload["news_first_materialize_test_loader"], False)
        self.assertEqual(payload["news_first_validation_end_utc"], FOLDS[0]["validation_end_utc"])
        self.assertEqual(payload["news_first_label_reliability_manifest_sha256"], "a" * 64)
        self.assertEqual(payload["news_first_label_reliability_profile_sha256"], "b" * 64)
        self.assertEqual(
            payload["news_first_label_reliability_train_pair_universe_sha256"],
            "c" * 64,
        )

    def test_compact_bootstrap_manifest_deduplicates_and_stays_pre_q3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datasets = base / "data"
            for tolerance in (5, 30):
                folder = datasets / f"tolerance_{tolerance:02d}m"
                folder.mkdir(parents=True)
                common = {
                    "pair_id": "p1",
                    "session_id": "s1",
                    "effective_origin_utc": "2023-06-30T12:00:00Z",
                    "current_snapshot_time_utc": "2023-06-30T12:00:00Z",
                    "target_snapshot_time_utc": "2023-06-30T12:05:00Z",
                    "current_surface_param_json": json.dumps({"x": 1}),
                    "target_surface_param_json": json.dumps({"x": 2}),
                    "current_surface_flat": "[0.1]",
                    "target_surface_flat": "[0.2]",
                    "strike_grid": "[1.0]",
                    "maturity_days_grid": "[7]",
                }
                q4 = {**common, "pair_id": "q4", "effective_origin_utc": "2023-10-01T00:00:00Z"}
                frame = pd.DataFrame([common, common, q4])
                with pd.ExcelWriter(folder / "merged_vol.xlsx", engine="openpyxl") as writer:
                    frame.to_excel(writer, sheet_name="gan_input_ready", index=False)
                pd.DataFrame(
                    {
                        "pair_id": ["p1", "q4"],
                        "joint_strict_support_cell_count": [16, 16],
                        "surface_training_eligible": [True, True],
                        "joint_zero_support": [False, False],
                        "grid_fingerprint": ["g", "g"],
                    }
                ).to_csv(folder / "surface_support_audit.csv.gz", index=False)
            resolved = {
                "datasets": {
                    "root": str(datasets),
                    "workbook_template": "tolerance_{tolerance02}m/merged_vol.xlsx",
                }
            }
            output = base / "experiment"
            (output / "bootstrap").mkdir(parents=True)
            manifests = _prepare_bootstrap_pair_manifests(output, resolved)
            self.assertEqual(set(manifests), {"5", "30"})
            compact = pd.read_csv(manifests["5"])
            self.assertEqual(compact["pair_id"].tolist(), ["p1"])
            self.assertIn("current_surface_param_json", compact.columns)
            self.assertIn("joint_strict_support_cell_count", compact.columns)


if __name__ == "__main__":
    unittest.main()
