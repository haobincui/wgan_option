"""Synthetic contracts for the direct matched-text high FiLM-LR analysis."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_high_lr_seed42_analysis as analysis,
)
from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_lr_seed42_analysis as low_analysis,
)


FOLD_COUNTS = {fold: (1, 1) for fold in analysis.FOLDS}
EXPECTED_RATES = {
    "film_lr_5e6": 5.0e-6,
    "film_lr_1e5": 1.0e-5,
    "film_lr_2p5e5": 2.5e-5,
    "film_lr_5e5": 5.0e-5,
    "film_lr_1e4": 1.0e-4,
}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _pair_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    film_rows: list[dict[str, object]] = []
    pure_rows: list[dict[str, object]] = []
    ratios = {
        "film_lr_5e6": 0.990,
        "film_lr_1e5": 0.980,
        "film_lr_2p5e5": 0.960,
        "film_lr_5e5": 0.970,
        "film_lr_1e4": 0.995,
    }
    for fold_index, fold in enumerate(analysis.FOLDS):
        pair_id = f"pair_{fold_index}"
        session_id = f"session_{fold_index}"
        origin = f"2023-0{fold_index + 1}-01T12:00:00Z"
        noise_sha = _sha(f"noise:{fold}")
        persistence = 1.1 + fold_index * 0.01
        pure_mae = 1.0 + fold_index * 0.01
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
        for arm, ratio in ratios.items():
            job_id = f"{arm}_{fold}"
            film_rows.append(
                {
                    **common,
                    "job_id": job_id,
                    "arm": arm,
                    "target_mae": pure_mae * ratio,
                    "checkpoint_sha256": _sha(f"checkpoint:{job_id}"),
                    "prediction_sha256": _sha(f"prediction:{job_id}"),
                }
            )
        pure_job = f"pure_{fold}"
        pure_rows.append(
            {
                **common,
                "job_id": pure_job,
                "arm": analysis.PURE_ARM,
                "target_mae": pure_mae,
                "checkpoint_sha256": _sha(f"checkpoint:{pure_job}"),
                "prediction_sha256": _sha(f"prediction:{pure_job}"),
            }
        )
    return pd.DataFrame(film_rows), pd.DataFrame(pure_rows)


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


def _training_summary(film: pd.DataFrame) -> pd.DataFrame:
    rows = []
    jobs = film[["job_id", "fold", "arm", "checkpoint_sha256"]].drop_duplicates()
    for row in jobs.itertuples(index=False):
        rows.append(
            {
                "job_id": row.job_id,
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


class HighFilmLrAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.film, self.pure = _pair_metrics()
        self.training = _training_summary(self.film)

    def _analyze(self) -> analysis.FilmLrAnalysis:
        return analysis.analyze_film_lr(
            self.film,
            self.pure,
            self.training,
            bootstrap_iterations=100,
            bootstrap_seed=123,
            expected_fold_counts=FOLD_COUNTS,
            expected_film_rows=20,
            expected_pure_rows=4,
        )

    def test_explicit_profile_holm_families_and_diagnostics(self) -> None:
        low_before = dict(low_analysis.FILM_LR_ARMS)
        result = self._analyze()
        self.assertEqual(dict(analysis.FILM_LR_ARMS), EXPECTED_RATES)
        self.assertEqual(dict(low_analysis.FILM_LR_ARMS), low_before)
        self.assertEqual(len(result.ranking), 5)
        self.assertEqual(len(result.comparisons), 5)
        self.assertEqual(len(result.pairwise_comparisons), 10)
        self.assertEqual(
            int(result.pairwise_comparisons["adjacent_learning_rates"].sum()), 4
        )
        self.assertEqual(len(result.training_diagnostics), 20)
        self.assertEqual(len(result.lr_trace), 20 * 4 * 3)
        self.assertEqual(result.summary["point_leader_arm"], "film_lr_2p5e5")
        self.assertEqual(
            result.summary["experiment"],
            "film_unet_direct_matched_high_lr_seed42_vs_frozen_pure_cnn",
        )
        self.assertFalse(result.summary["point_leader_is_model_selection"])
        self.assertFalse(result.summary["test_based_lr_selection_permitted"])
        self.assertTrue((result.comparisons["holm_family_size"] == 5).all())
        self.assertTrue((result.pairwise_comparisons["holm_family_size"] == 10).all())
        self.assertTrue(result.pairwise_comparisons["comparison_id"].is_unique)
        self.assertEqual(
            set(result.training_diagnostics["film_configured"]),
            set(EXPECTED_RATES.values()),
        )
        self.assertEqual(
            set(result.lr_trace["parameter_group"]),
            {"backbone", "text_encoder", "film", "critic"},
        )

    def test_lineage_and_wrong_optimizer_lr_fail_closed(self) -> None:
        noise_drift = self.pure.copy()
        noise_drift.loc[0, "noise_bank_profile_sha256"] = _sha("wrong-noise")
        with self.assertRaisesRegex(analysis.FilmLrAnalysisError, "noise"):
            analysis.analyze_film_lr(
                self.film,
                noise_drift,
                self.training,
                bootstrap_iterations=10,
                expected_fold_counts=FOLD_COUNTS,
                expected_film_rows=20,
                expected_pure_rows=4,
            )

        wrong_lr = self.training.copy()
        payload = json.loads(wrong_lr.loc[0, "optimizer_contract_json"])
        payload["groups"]["film"]["configured_learning_rate"] = 9e-4
        payload["groups"]["film"]["initial_learning_rate"] = 9e-4
        payload["groups"]["film"]["final_learning_rate"] = 9e-4
        payload["groups"]["film"]["lr_trace"] = [9e-4, 9e-4, 9e-4]
        wrong_lr.loc[0, "optimizer_contract_json"] = json.dumps(payload)
        with self.assertRaisesRegex(
            analysis.FilmLrAnalysisError, "configured LR drift"
        ):
            analysis.validate_training_evidence(wrong_lr, self.film)

        incomplete = self.training.copy()
        payload = json.loads(incomplete.loc[0, "optimizer_contract_json"])
        payload["groups"]["film"]["lr_trace"] = payload["groups"]["film"]["lr_trace"][
            :-1
        ]
        incomplete.loc[0, "optimizer_contract_json"] = json.dumps(payload)
        with self.assertRaisesRegex(analysis.FilmLrAnalysisError, "contiguous"):
            analysis.validate_training_evidence(incomplete, self.film)

    def test_sha_bound_bundle_is_idempotent_and_uses_shared_schema_kind(self) -> None:
        result = self._analyze()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            film_path = root / "film.csv"
            pure_path = root / "pure.csv"
            training_path = root / "training.csv"
            self.film.to_csv(film_path, index=False)
            self.pure.to_csv(pure_path, index=False)
            self.training.to_csv(training_path, index=False)
            kwargs = {
                "analysis": result,
                "film_pair_metrics_path": film_path,
                "film_pair_metrics_sha256": analysis.sha256_file(film_path),
                "pure_pair_metrics_path": pure_path,
                "pure_pair_metrics_sha256": analysis.sha256_file(pure_path),
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
                "film_unet_direct_matched_lr_seed42_analysis_manifest_v1",
            )
            self.assertFalse(manifest["confirmatory"])
            self.assertFalse(manifest["test_based_lr_selection_permitted"])
            self.assertEqual(len(manifest["inputs"]), 3)
            self.assertEqual(len(manifest["artifacts"]), 9)

    def test_low_range_arm_is_rejected_without_global_profile_patching(self) -> None:
        drift = self.film.copy()
        drift.loc[drift["arm"].eq("film_lr_1e5"), "arm"] = "film_lr_1e6"
        low_before = copy.deepcopy(dict(low_analysis.FILM_LR_ARMS))
        with self.assertRaises(analysis.FilmLrAnalysisError):
            analysis.validate_film_pair_metrics(
                drift,
                expected_fold_counts=FOLD_COUNTS,
                expected_row_count=20,
            )
        self.assertEqual(dict(low_analysis.FILM_LR_ARMS), low_before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
