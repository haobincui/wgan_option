from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.rq3 import news_first_vol_zero_noise_ablation as experiment
from scripts.rq3.news_first_vol_zero_noise_analysis import (
    run_zero_noise_analysis,
)
from scripts.rq3.news_first_vol_zero_noise_report import (
    ZeroNoiseReportError,
    render_zero_noise_report,
)
from wgan_option.models.common import generator_noise_fingerprint


class ZeroNoiseAblationContractTest(unittest.TestCase):
    def test_frozen_job_matrix_and_shared_training_profile_hashes(self) -> None:
        specs = experiment._job_specs()
        self.assertEqual(len(specs), 4)
        self.assertEqual(
            {(row["text_ablation_mode"], row["tolerance_minutes"]) for row in specs},
            {
                ("current_only", 5),
                ("current_only", 30),
                ("real_text", 5),
                ("real_text", 30),
            },
        )
        self.assertEqual({row["gpu_index"] for row in specs}, {0, 1})
        for gpu_index in (0, 1):
            cells = [row for row in specs if row["gpu_index"] == gpu_index]
            self.assertEqual(len(cells), 2)
            self.assertEqual({row["tolerance_minutes"] for row in cells}, {5, 30})
            self.assertEqual(
                {row["text_ablation_mode"] for row in cells},
                {"current_only", "real_text"},
            )
        self.assertEqual(
            experiment._lr_profile_sha256(),
            "51d9ec40a646a92b6d759186d6b74da389246e6af15def4ab2c7275aee6a4264",
        )
        self.assertEqual(
            experiment._capacity_seed_profile_sha256(),
            "7affab6991de8905f7be313ae82cd7c42242300553489addc926c0d1ab116a90",
        )
        for mode in experiment.FROZEN_TEXT_MODES:
            for tolerance in experiment.FROZEN_TOLERANCES:
                self.assertIn("_zero_", experiment._job_id(mode, tolerance))

    def test_exact_immutable_gaussian_reference_contract(self) -> None:
        config = experiment._resolved_config(
            "configs/rq3/news_first_vol_zero_noise_ablation.yaml"
        )
        rows = experiment._gaussian_reference_rows(config)
        self.assertEqual(len(rows), 4)
        for row in rows:
            expected = experiment.EXPECTED_REFERENCE_HASHES[row["reference_job_id"]]
            self.assertEqual(row["config_sha256"], expected["config"])
            self.assertEqual(row["generator_checkpoint_sha256"], expected["generator"])
            self.assertEqual(
                row["discriminator_checkpoint_sha256"],
                expected["discriminator"],
            )
            self.assertEqual(row["best_learned_metadata_sha256"], expected["metadata"])
            self.assertEqual(
                row["initial_generator_state_sha256"],
                experiment.EXPECTED_INITIAL_GENERATOR_STATE_SHA256,
            )
            self.assertEqual(
                row["initial_discriminator_state_sha256"],
                experiment.EXPECTED_INITIAL_DISCRIMINATOR_STATE_SHA256,
            )

    def test_wgan_lr_trace_checks_epoch_zero_floor_and_minimum_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            metrics = run_dir / "metrics"
            metrics.mkdir()
            rows = []
            for epoch in range(31):
                row = {
                    "epoch": epoch,
                    "val_recon": 0.1,
                    "val_current_recon": 0.1,
                    "val_baseline_gap": 0.0,
                    "val_hybrid_score": 0.1,
                    "val_calendar": 0.0,
                    "val_butterfly": 0.0,
                    "val_delta_shrink": 0.0,
                    "g_lr": 5e-7,
                    "d_lr": 5e-7,
                }
                if epoch:
                    row.update(
                        {
                            "d_total": 1.0,
                            "d_real": 0.1,
                            "d_fake": 0.2,
                            "gp": 1.0 + epoch / 100.0,
                            "g_total": 1.0,
                            "g_adv": 0.1,
                            "g_recon": 0.1,
                            "g_calendar": 0.0,
                            "g_butterfly": 0.0,
                            "g_smooth": 0.1,
                            "g_delta_shrink": 0.0,
                        }
                    )
                rows.append(row)
            (metrics / "training_metrics.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )
            job = {
                "job_id": "fixture",
                "initial_learning_rate": 5e-7,
                "scheduler_min_lr": 5e-8,
            }
            trace = experiment._validated_wgan_lr_trace(job, run_dir, dry_run=False)
            self.assertEqual(len(trace), 31)
            rows.pop()
            (metrics / "training_metrics.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "minimum 30 epochs"):
                experiment._validated_wgan_lr_trace(job, run_dir, dry_run=False)

    def test_split_manifest_exact_counts_and_leakage_gate(self) -> None:
        fields = {
            "validation_rows": 133,
            "validation_pairs": 123,
            "validation_sessions": 33,
            "test_rows": 152,
            "test_pairs": 130,
            "test_sessions": 45,
            "support_mask_mode": "raw_joint",
            "status": "pass",
            "train_validation_pair_overlap": 0,
            "train_test_pair_overlap": 0,
            "validation_test_pair_overlap": 0,
            "train_validation_session_overlap": 0,
            "train_test_session_overlap": 0,
            "validation_test_session_overlap": 0,
        }
        rows = [
            {
                "tolerance_minutes": 5,
                "train_rows": 936,
                "train_pairs": 720,
                "train_sessions": 210,
                **fields,
            },
            {
                "tolerance_minutes": 30,
                "train_rows": 1591,
                "train_pairs": 1050,
                "train_sessions": 237,
                **fields,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(rows).to_csv(root / "split_manifest.csv", index=False)
            experiment._validate_split_manifest(root)
            rows[1]["train_test_pair_overlap"] = 1
            pd.DataFrame(rows).to_csv(root / "split_manifest.csv", index=False)
            with self.assertRaisesRegex(ValueError, "leakage"):
                experiment._validate_split_manifest(root)


class ZeroNoiseAnalysisFixtureTest(unittest.TestCase):
    @staticmethod
    def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
        sessions = [f"session_{index % 33:02d}" for index in range(123)]
        zero_rows = []
        gaussian_rows = []
        ref_by_cell = {
            ("current_only", 5): "s3_wgan_small_lr_5e_07_seed_042_current_only_05m",
            ("current_only", 30): "s3_wgan_small_lr_5e_07_seed_042_current_only_30m",
            ("real_text", 5): "s3_wgan_small_lr_5e_07_seed_042_real_text_05m",
            ("real_text", 30): "s3_wgan_small_lr_5e_07_seed_042_real_text_30m",
        }
        effects = {
            ("current_only", 5): -0.002,
            ("current_only", 30): -0.001,
            ("real_text", 5): 0.001,
            ("real_text", 30): 0.002,
        }
        fingerprint = generator_noise_fingerprint("zero", 32)
        for mode in experiment.FROZEN_TEXT_MODES:
            for tolerance in experiment.FROZEN_TOLERANCES:
                ref = ref_by_cell[(mode, tolerance)]
                for index in range(123):
                    pair_id = f"pair_{index:03d}"
                    gaussian_mae = 0.02 + index / 1_000_000.0
                    persistence = 0.03 + index / 1_000_000.0
                    zero_rows.append(
                        {
                            "zero_job_id": experiment._job_id(mode, tolerance),
                            "gaussian_reference_job_id": ref,
                            "generator_noise_mode": "zero",
                            "generator_noise_fingerprint": fingerprint,
                            "prediction_mc_samples": 1,
                            "text_ablation_mode": mode,
                            "tolerance_minutes": tolerance,
                            "pair_id": pair_id,
                            "session_id": sessions[index],
                            "model_mae": gaussian_mae + effects[(mode, tolerance)],
                            "persistence_mae": persistence,
                        }
                    )
                    gaussian_rows.append(
                        {
                            "run_id": ref,
                            "text_ablation_mode": mode,
                            "tolerance_minutes": tolerance,
                            "pair_id": pair_id,
                            "session_id": sessions[index],
                            "model_mae": gaussian_mae,
                            "persistence_mae": persistence,
                        }
                    )
        return pd.DataFrame(zero_rows), pd.DataFrame(gaussian_rows)

    def test_full_injected_analysis_and_report(self) -> None:
        zero, gaussian = self._frames()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = run_zero_noise_analysis(
                root,
                zero_pair_metrics=zero,
                gaussian_pair_metrics=gaussian,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertFalse(summary["q4_predictions_generated"])
            self.assertFalse(summary["q4_evaluated"])
            comparisons = pd.read_csv(
                root / "analysis" / "zero_noise_vs_gaussian_bootstrap.csv"
            )
            primary = comparisons[comparisons["analysis_role"].eq("primary")]
            self.assertEqual(len(primary), 2)
            self.assertEqual(set(primary["holm_family"]), {"primary_combined_modes"})
            secondary = comparisons[
                comparisons["analysis_role"].eq("secondary_sensitivity")
            ]
            self.assertEqual(len(secondary), 4)
            report = render_zero_noise_report(root)
            body = report.read_text(encoding="utf-8")
            self.assertIn("Q3 validation", body)
            self.assertIn("145,237 active", body)
            self.assertIn("没有生成 Q4 预测", body)

    def test_report_rejects_rehashed_q4_claim(self) -> None:
        zero, gaussian = self._frames()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = run_zero_noise_analysis(
                root,
                zero_pair_metrics=zero,
                gaussian_pair_metrics=gaussian,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["q4_evaluated"] = True
            summary.pop("analysis_sha256")
            summary["analysis_sha256"] = experiment._payload_sha256(summary)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ZeroNoiseReportError, "Q4 isolation"):
                render_zero_noise_report(root)


if __name__ == "__main__":
    unittest.main()
