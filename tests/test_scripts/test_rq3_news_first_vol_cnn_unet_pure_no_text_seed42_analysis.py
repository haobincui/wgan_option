from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.rq3 import news_first_vol_cnn_unet_pure_no_text_seed42_analysis as analysis


FOLDS = tuple(analysis.DIRECT_FOLDS)
FOLD_COUNTS = {fold: (4, 2) for fold in FOLDS}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _synthetic_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    historical_rows: list[dict[str, object]] = []
    pure_rows: list[dict[str, object]] = []
    multipliers = {
        "no_text": 1.00,
        "lp_matched": 0.90,
        "lp_shuffle": 0.85,
        "bow": 1.10,
        "sentiment": 1.20,
    }
    for fold_index, fold in enumerate(FOLDS):
        noise_sha = _digest(f"noise-{fold}")
        for pair_index in range(4):
            pair_id = f"{fold}-pair-{pair_index}"
            session_id = f"{fold}-session-{pair_index // 2}"
            origin = f"2023-0{fold_index + 1}-0{pair_index + 1}T14:00:00Z"
            base = 1.0 + 0.1 * fold_index + 0.01 * pair_index
            persistence = 2.0 * base
            common = {
                "tolerance_minutes": 5,
                "fold": fold,
                "seed": 42,
                "pair_id": pair_id,
                "session_id": session_id,
                "effective_origin_utc": origin,
                "persistence_mae": persistence,
                "noise_bank_profile_sha256": noise_sha,
            }
            for arm, multiplier in multipliers.items():
                job_id = f"historical-{fold}-{arm}"
                historical_rows.append(
                    {
                        **common,
                        "job_id": job_id,
                        "arm": arm,
                        "target_mae": base * multiplier,
                        "checkpoint_sha256": _digest(f"checkpoint-{job_id}"),
                        "prediction_sha256": _digest(f"prediction-{job_id}"),
                    }
                )
            pure_job_id = f"pure-{fold}"
            pure_rows.append(
                {
                    **common,
                    "job_id": pure_job_id,
                    "arm": analysis.PURE_ARM,
                    "target_mae": base * 0.80,
                    "checkpoint_sha256": _digest(f"checkpoint-{pure_job_id}"),
                    "prediction_sha256": _digest(f"prediction-{pure_job_id}"),
                }
            )
    return pd.DataFrame(pure_rows), pd.DataFrame(historical_rows)


def _training_summary(pure: pd.DataFrame) -> pd.DataFrame:
    jobs = pure[["job_id", "fold", "arm", "checkpoint_sha256"]].drop_duplicates()
    return jobs.assign(
        best_epoch=40,
        epochs_ran=60,
        final_generator_lr=5e-8,
        final_discriminator_lr=5e-8,
        best_validation_score=0.01,
    )


class PureCnnAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pure, self.historical = _synthetic_evidence()

    def analyze(self, *, iterations: int = 256) -> analysis.PureCnnAnalysis:
        return analysis.analyze_pure_cnn(
            self.pure,
            self.historical,
            training_summary=_training_summary(self.pure),
            bootstrap_iterations=iterations,
            bootstrap_seed=1234,
            expected_fold_counts=FOLD_COUNTS,
            expected_pure_row_count=16,
            expected_historical_row_count=80,
        )

    def test_primary_comparison_is_paired_fold_then_session_and_descriptive(
        self,
    ) -> None:
        result = self.analyze()
        primary = result.comparisons[
            result.comparisons["reference_arm"].eq("no_text")
        ].iloc[0]
        self.assertAlmostEqual(
            float(primary["mean_log_mae_ratio"]), math.log(0.8), places=14
        )
        self.assertLess(float(primary["ci_95_upper"]), 0.0)
        self.assertEqual(int(primary["focal_nonworse_fold_count"]), 4)
        self.assertEqual(int(primary["pair_count"]), 16)
        self.assertEqual(int(primary["session_count"]), 8)
        self.assertEqual(int(primary["bootstrap_iterations"]), 256)
        self.assertFalse(bool(primary["capacity_matched"]))
        self.assertFalse(bool(primary["inference_permitted"]))
        self.assertEqual(
            primary["resampling_method"],
            "fold_then_paired_cme_session_cluster_recompute_fold_log_mae_ratio",
        )
        self.assertEqual(
            set(result.comparisons["reference_arm"]),
            {"no_text", "lp_matched", "lp_shuffle"},
        )

        self.assertAlmostEqual(
            result.summary["pure_equal_fold_mae"],
            result.summary["film_no_text_equal_fold_mae"] * 0.8,
            places=14,
        )
        self.assertEqual(result.summary["pure_job_count"], 4)
        self.assertEqual(result.summary["pure_pair_metric_row_count"], 16)
        self.assertEqual(result.summary["pair_count"], 16)
        self.assertEqual(result.summary["session_count"], 8)
        self.assertFalse(result.summary["capacity_matched_comparison"])
        self.assertEqual(result.summary["generator_parameter_delta"], -411_392)
        self.assertEqual(result.summary["total_parameter_delta"], -411_392)
        self.assertEqual(result.summary["share_readiness"], "share_with_caveats")

    def test_bootstrap_and_tables_are_deterministic(self) -> None:
        first = self.analyze(iterations=128)
        second = self.analyze(iterations=128)
        assert_frame_equal(first.comparisons, second.comparisons, check_exact=True)
        assert_frame_equal(first.persistence, second.persistence, check_exact=True)
        assert_frame_equal(first.arm_summary, second.arm_summary, check_exact=True)

    def test_cross_experiment_pair_time_persistence_and_noise_drift_fail_closed(
        self,
    ) -> None:
        pure = analysis.validate_pure_pair_metrics(
            self.pure,
            expected_fold_counts=FOLD_COUNTS,
            expected_row_count=16,
        )
        historical = analysis.validate_historical_pair_metrics(
            self.historical,
            expected_fold_counts=FOLD_COUNTS,
            expected_row_count=80,
        )
        analysis.validate_cross_experiment_lineage(pure, historical)

        for column, replacement, message in (
            ("session_id", "wrong-session", "session_id"),
            ("effective_origin_utc", "2020-01-01T00:00:00Z", "effective_origin"),
            ("persistence_mae", 999.0, "persistence"),
        ):
            changed = pure.copy()
            changed.loc[0, column] = replacement
            with self.subTest(column=column):
                with self.assertRaisesRegex(analysis.PureCnnAnalysisError, message):
                    analysis.validate_cross_experiment_lineage(changed, historical)

        changed_noise = pure.copy()
        changed_noise.loc[
            changed_noise["fold"].eq(FOLDS[0]), "noise_bank_profile_sha256"
        ] = _digest("different-noise")
        with self.assertRaisesRegex(analysis.PureCnnAnalysisError, "noise_bank"):
            analysis.validate_cross_experiment_lineage(changed_noise, historical)

    def test_training_summary_must_match_all_four_frozen_checkpoints(self) -> None:
        pure = analysis.validate_pure_pair_metrics(
            self.pure,
            expected_fold_counts=FOLD_COUNTS,
            expected_row_count=16,
        )
        training = _training_summary(self.pure)
        validated = analysis.validate_training_summary(training, pure)
        self.assertEqual(len(validated), 4)
        changed = training.copy()
        changed.loc[0, "checkpoint_sha256"] = _digest("wrong-checkpoint")
        with self.assertRaisesRegex(analysis.PureCnnAnalysisError, "checkpoint"):
            analysis.validate_training_summary(changed, pure)
        too_long = training.copy()
        too_long.loc[0, "epochs_ran"] = 241
        with self.assertRaisesRegex(analysis.PureCnnAnalysisError, "epochs"):
            analysis.validate_training_summary(too_long, pure)

    def test_report_is_answer_first_and_states_all_interpretation_boundaries(
        self,
    ) -> None:
        result = self.analyze(iterations=64)
        markdown, html_report = analysis.render_reports(result)
        self.assertIn("## 技术摘要", markdown)
        self.assertIn("不是容量匹配消融", markdown)
        self.assertIn("seed 42", markdown)
        self.assertIn(
            "retrospective_rolling_development_single_seed_descriptive", markdown
        )
        self.assertIn("纯CNN四fold等权MAE", markdown)
        self.assertIn("建议的下一步", markdown)
        self.assertIn('<meta name="color-scheme" content="light dark">', html_report)
        self.assertIn("限制与稳健性", html_report)
        self.assertIn("配对bootstrap不能补偿", html_report)

    def test_bundle_freezes_both_sources_and_is_idempotent(self) -> None:
        result = self.analyze(iterations=64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pure_path = root / "rq12_pair_metrics.csv.gz"
            historical_path = root / "historical_pair_metrics.csv.gz"
            training_path = root / "training_summary.csv"
            self.pure.to_csv(pure_path, index=False)
            self.historical.to_csv(historical_path, index=False)
            _training_summary(self.pure).to_csv(training_path, index=False)
            kwargs = {
                "analysis": result,
                "pure_pair_metrics_path": pure_path,
                "pure_pair_metrics_sha256": analysis.sha256_file(pure_path),
                "historical_pair_metrics_path": historical_path,
                "historical_pair_metrics_sha256": analysis.sha256_file(historical_path),
                "training_summary_path": training_path,
                "training_summary_sha256": analysis.sha256_file(training_path),
                "output_dir": root / "bundle",
            }
            first = analysis.write_analysis_bundle(**kwargs)
            before = {
                path.name: analysis.sha256_file(path)
                for path in (root / "bundle").iterdir()
                if path.is_file()
            }
            second = analysis.write_analysis_bundle(**kwargs)
            after = {
                path.name: analysis.sha256_file(path)
                for path in (root / "bundle").iterdir()
                if path.is_file()
            }
            self.assertEqual(first, second)
            self.assertEqual(before, after)
            manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["inputs"]), 3)
            self.assertEqual(len(manifest["artifacts"]), 7)
            self.assertEqual(len({row["role"] for row in manifest["artifacts"]}), 7)
            self.assertFalse(manifest["capacity_matched_comparison"])
            self.assertEqual(manifest["audience"], "technical")

            historical_path.write_bytes(historical_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(analysis.PureCnnAnalysisError, "SHA"):
                analysis.write_analysis_bundle(**kwargs)


if __name__ == "__main__":
    unittest.main()
