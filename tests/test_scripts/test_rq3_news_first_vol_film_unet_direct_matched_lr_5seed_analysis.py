"""Synthetic contracts for the two-LR, five-seed analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_lr_5seed_analysis as analysis,
)


FOLD_COUNTS = {fold: (1, 1) for fold in analysis.FOLDS}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _pair_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(analysis.SEEDS):
        for fold_index, fold in enumerate(analysis.FOLDS):
            pair_id = f"pair_{fold_index}"
            session_id = f"session_{fold_index}"
            origin = f"2023-0{fold_index + 1}-01T12:00:00Z"
            noise_sha = _sha(f"noise:{seed}:{fold}")
            persistence = 1.2 + 0.01 * fold_index
            reference_mae = 1.0 + 0.01 * fold_index + 0.001 * seed_index
            for arm, ratio in (
                ("film_lr_1e5", 1.0),
                ("film_lr_2p5e5", 0.96),
            ):
                job_id = f"{arm}_{seed}_{fold}"
                rows.append(
                    {
                        "job_id": job_id,
                        "tolerance_minutes": 5,
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "pair_id": pair_id,
                        "session_id": session_id,
                        "effective_origin_utc": origin,
                        "target_mae": reference_mae * ratio,
                        "persistence_mae": persistence,
                        "checkpoint_sha256": _sha(f"checkpoint:{job_id}"),
                        "prediction_sha256": _sha(f"prediction:{job_id}"),
                        "noise_bank_profile_sha256": noise_sha,
                    }
                )
    return pd.DataFrame(rows)


def _optimizer_contract(film_lr: float, epochs: int = 2) -> dict[str, object]:
    groups = {}
    for name, (parameter_count, fixed_lr) in analysis.EXPECTED_GROUPS.items():
        learning_rate = film_lr if name == "film" else fixed_lr
        assert learning_rate is not None
        groups[name] = {
            "parameter_count": parameter_count,
            "configured_learning_rate": learning_rate,
            "initial_learning_rate": learning_rate,
            "final_learning_rate": learning_rate,
            "lr_trace": [
                {"epoch": epoch, "lr": learning_rate} for epoch in range(epochs + 1)
            ],
        }
    return {"schema_version": 1, "groups": groups}


def _training_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    jobs = pairs[
        ["job_id", "seed", "fold", "arm", "checkpoint_sha256"]
    ].drop_duplicates()
    for row in jobs.itertuples(index=False):
        rows.append(
            {
                "job_id": row.job_id,
                "seed": row.seed,
                "fold": row.fold,
                "arm": row.arm,
                "best_epoch": 2,
                "epochs_ran": 2,
                "best_validation_score": 0.01,
                "checkpoint_sha256": row.checkpoint_sha256,
                "optimizer_contract_json": json.dumps(
                    _optimizer_contract(analysis.FILM_LR_ARMS[row.arm]),
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


class FiveSeedFilmLrAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs = _pair_metrics()
        self.training = _training_summary(self.pairs)

    def _analyze(self) -> analysis.FilmLrFiveSeedAnalysis:
        return analysis.analyze_film_lr(
            self.pairs,
            self.training,
            bootstrap_iterations=100,
            bootstrap_seed=123,
            expected_fold_counts=FOLD_COUNTS,
            expected_pair_rows=40,
        )

    def test_fixed_primary_holm2_and_grouped_optimizer_evidence(self) -> None:
        result = self._analyze()
        self.assertEqual(len(result.pair_metrics), 40)
        self.assertEqual(len(result.seed_fold_summary), 40)
        self.assertEqual(len(result.arm_summary), 2)
        self.assertEqual(len(result.primary_comparison), 1)
        self.assertEqual(len(result.persistence_comparisons), 2)
        self.assertEqual(len(result.training_diagnostics), 40)
        self.assertEqual(len(result.lr_trace), 40 * 4 * 3)
        primary = result.primary_comparison.iloc[0]
        self.assertEqual(primary["focal_arm"], "film_lr_2p5e5")
        self.assertEqual(primary["reference_arm"], "film_lr_1e5")
        self.assertLess(float(primary["mean_log_mae_ratio"]), 0.0)
        self.assertEqual(int(primary["consistent_seed_count"]), 5)
        self.assertEqual(int(primary["consistent_fold_count"]), 4)
        self.assertTrue((result.persistence_comparisons["holm_family_size"] == 2).all())
        self.assertEqual(
            set(result.training_diagnostics["film_configured"]), {1e-5, 2.5e-5}
        )
        self.assertEqual(
            set(result.lr_trace["parameter_group"]),
            {"backbone", "text_encoder", "film", "critic"},
        )
        self.assertFalse(result.summary["test_based_lr_selection_permitted"])
        self.assertEqual(
            result.summary["interpretation"],
            "retrospective_rolling_development_post_selection_seed_robustness",
        )

    def test_lineage_seed_and_optimizer_drift_fail_closed(self) -> None:
        noise_drift = self.pairs.copy()
        mask = (
            noise_drift["seed"].eq(analysis.SEEDS[0])
            & noise_drift["fold"].eq(analysis.FOLDS[0])
            & noise_drift["arm"].eq("film_lr_2p5e5")
        )
        noise_drift.loc[mask, "noise_bank_profile_sha256"] = _sha("wrong-noise")
        with self.assertRaisesRegex(analysis.FilmLrFiveSeedAnalysisError, "lineage"):
            analysis.validate_pair_metrics(
                noise_drift,
                expected_fold_counts=FOLD_COUNTS,
                expected_row_count=40,
            )

        seed_drift = self.pairs.copy()
        seed_drift.loc[seed_drift["seed"].eq(analysis.SEEDS[-1]), "seed"] = 999
        with self.assertRaisesRegex(
            analysis.FilmLrFiveSeedAnalysisError, "seed universe"
        ):
            analysis.validate_pair_metrics(
                seed_drift,
                expected_fold_counts=FOLD_COUNTS,
                expected_row_count=40,
            )

        wrong_lr = self.training.copy()
        payload = json.loads(wrong_lr.loc[0, "optimizer_contract_json"])
        payload["groups"]["film"]["configured_learning_rate"] = 9e-4
        wrong_lr.loc[0, "optimizer_contract_json"] = json.dumps(payload)
        validated = analysis.validate_pair_metrics(
            self.pairs,
            expected_fold_counts=FOLD_COUNTS,
            expected_row_count=40,
        )
        with self.assertRaisesRegex(
            analysis.FilmLrFiveSeedAnalysisError, "configured LR drift"
        ):
            analysis.validate_training_evidence(wrong_lr, validated)

    def test_sha_bound_bundle_is_idempotent_and_self_contained(self) -> None:
        result = self._analyze()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_path = root / "pairs.csv"
            training_path = root / "training.csv"
            self.pairs.to_csv(pair_path, index=False)
            self.training.to_csv(training_path, index=False)
            kwargs = {
                "analysis": result,
                "pair_metrics_path": pair_path,
                "pair_metrics_sha256": analysis.sha256_file(pair_path),
                "training_summary_path": training_path,
                "training_summary_sha256": analysis.sha256_file(training_path),
                "output_dir": root / "out",
            }
            first = analysis.write_analysis_bundle(**kwargs)
            second = analysis.write_analysis_bundle(**kwargs)
            self.assertEqual(first, second)
            report = first["report_html"].read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", report)
            self.assertNotIn("https://", report)
            manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["kind"],
                "film_unet_direct_matched_lr_5seed_analysis_manifest_v1",
            )
            self.assertFalse(manifest["confirmatory"])
            self.assertFalse(manifest["test_based_lr_selection_permitted"])
            self.assertEqual(len(manifest["inputs"]), 2)
            self.assertEqual(len(manifest["artifacts"]), 9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
