"""Synthetic contracts for the direct FiLM projection-LR analysis."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_lr_seed42_analysis as analysis,
)


FOLD_COUNTS = {fold: (1, 1) for fold in analysis.FOLDS}


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _pair_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    film_rows: list[dict[str, object]] = []
    pure_rows: list[dict[str, object]] = []
    ratios = {
        "film_lr_2p5e7": 0.995,
        "film_lr_5e7": 0.990,
        "film_lr_1e6": 0.985,
        "film_lr_2p5e6": 0.980,
        "film_lr_5e6": 0.975,
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


class FilmLrAnalysisTests(unittest.TestCase):
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

    def test_full_matrix_ranking_holm5_and_all_pairwise_holm10(self) -> None:
        result = self._analyze()
        self.assertEqual(len(result.ranking), 5)
        self.assertEqual(len(result.comparisons), 5)
        self.assertEqual(len(result.pairwise_comparisons), 10)
        self.assertEqual(
            int(result.pairwise_comparisons["adjacent_learning_rates"].sum()), 4
        )
        self.assertEqual(len(result.training_diagnostics), 20)
        self.assertEqual(len(result.lr_trace), 20 * 4 * 3)
        self.assertEqual(result.summary["point_leader_arm"], "film_lr_5e6")
        self.assertFalse(result.summary["point_leader_is_model_selection"])
        self.assertFalse(result.summary["test_based_lr_selection_permitted"])
        self.assertTrue((result.comparisons["holm_family_size"] == 5).all())
        self.assertTrue((result.pairwise_comparisons["holm_family_size"] == 10).all())
        self.assertTrue(result.pairwise_comparisons["comparison_id"].is_unique)
        self.assertTrue((result.comparisons["mean_log_mae_ratio"] < 0.0).all())
        self.assertTrue((result.ranking["selection_permitted"] == False).all())  # noqa: E712

    def test_cross_experiment_noise_and_persistence_drift_fail_closed(self) -> None:
        film_noise_drift = self.film.copy()
        film_noise_drift.loc[
            film_noise_drift["arm"].eq("film_lr_5e6")
            & film_noise_drift["fold"].eq(analysis.FOLDS[0]),
            "noise_bank_profile_sha256",
        ] = _sha("wrong-film-noise")
        with self.assertRaisesRegex(analysis.FilmLrAnalysisError, "MC-noise|noise"):
            analysis.analyze_film_lr(
                film_noise_drift,
                self.pure,
                self.training,
                bootstrap_iterations=10,
                expected_fold_counts=FOLD_COUNTS,
                expected_film_rows=20,
                expected_pure_rows=4,
            )
        noise_drift = self.pure.copy()
        noise_drift.loc[0, "noise_bank_profile_sha256"] = _sha("wrong-noise")
        with self.assertRaisesRegex(analysis.FilmLrAnalysisError, "MC-noise|noise"):
            analysis.analyze_film_lr(
                self.film,
                noise_drift,
                self.training,
                bootstrap_iterations=10,
                expected_fold_counts=FOLD_COUNTS,
                expected_film_rows=20,
                expected_pure_rows=4,
            )
        persistence_drift = self.pure.copy()
        persistence_drift.loc[0, "persistence_mae"] += 0.1
        with self.assertRaisesRegex(analysis.FilmLrAnalysisError, "persistence"):
            analysis.analyze_film_lr(
                self.film,
                persistence_drift,
                self.training,
                bootstrap_iterations=10,
                expected_fold_counts=FOLD_COUNTS,
                expected_film_rows=20,
                expected_pure_rows=4,
            )

    def test_training_contract_rejects_wrong_film_lr_and_incomplete_trace(self) -> None:
        wrong_lr = self.training.copy()
        payload = json.loads(wrong_lr.loc[0, "optimizer_contract_json"])
        payload["groups"]["film"]["configured_learning_rate"] = 9e-6
        payload["groups"]["film"]["initial_learning_rate"] = 9e-6
        payload["groups"]["film"]["final_learning_rate"] = 9e-6
        payload["groups"]["film"]["lr_trace"] = [9e-6, 9e-6, 9e-6]
        wrong_lr.loc[0, "optimizer_contract_json"] = json.dumps(payload)
        with self.assertRaisesRegex(
            analysis.FilmLrAnalysisError, "configured LR drift"
        ):
            analysis.validate_training_evidence(wrong_lr, self.film)

        missing_epoch = self.training.copy()
        payload = json.loads(missing_epoch.loc[0, "optimizer_contract_json"])
        payload["groups"]["film"]["lr_trace"] = payload["groups"]["film"]["lr_trace"][
            :-1
        ]
        missing_epoch.loc[0, "optimizer_contract_json"] = json.dumps(payload)
        with self.assertRaisesRegex(analysis.FilmLrAnalysisError, "contiguous"):
            analysis.validate_training_evidence(missing_epoch, self.film)

    def test_sha_bound_bundle_is_idempotent_and_html_is_self_contained(self) -> None:
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
            html_text = first["report_html"].read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", html_text)
            self.assertNotIn("http://", html_text)
            self.assertNotIn("https://", html_text)
            manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
            self.assertFalse(manifest["confirmatory"])
            self.assertFalse(manifest["test_based_lr_selection_permitted"])
            self.assertEqual(len(manifest["inputs"]), 3)
            self.assertEqual(len(manifest["artifacts"]), 9)

    def test_external_optimizer_contract_is_sha_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = self.training.copy()
            payload = external.loc[0, "optimizer_contract_json"]
            contract_path = root / "optimizer.json"
            contract_path.write_text(payload, encoding="utf-8")
            external.loc[0, "optimizer_contract_json"] = ""
            external.loc[0, "optimizer_contract_path"] = str(contract_path)
            external.loc[0, "optimizer_contract_sha256"] = analysis.sha256_file(
                contract_path
            )
            evidence = analysis.validate_training_evidence(external, self.film)
            self.assertEqual(len(evidence.external_inputs), 1)
            tampered = copy.deepcopy(external)
            tampered.loc[0, "optimizer_contract_sha256"] = _sha("wrong")
            with self.assertRaisesRegex(analysis.FilmLrAnalysisError, "SHA drift"):
                analysis.validate_training_evidence(tampered, self.film)

    def test_native_direct_artifacts_supply_real_split_lr_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            analysis_dir = root / "analysis"
            status_dir = root / "registry/job_status"
            analysis_dir.mkdir(parents=True)
            status_dir.mkdir(parents=True)
            film = self.film.copy()
            summary = self.training.drop(columns="optimizer_contract_json").copy()
            summary["best_epoch"] = 1
            registry_jobs = []
            for index, row in enumerate(summary.to_dict(orient="records")):
                job_id = str(row["job_id"])
                arm = str(row["arm"])
                film_lr = analysis.FILM_LR_ARMS[arm]
                run = root / "runs" / job_id
                run.mkdir(parents=True)
                generator_path = run / "generator_best_learned.pt"
                generator_path.write_bytes(f"generator:{job_id}".encode())
                checkpoint_sha = analysis.sha256_file(generator_path)
                summary.loc[index, "checkpoint_sha256"] = checkpoint_sha
                film.loc[film["job_id"].eq(job_id), "checkpoint_sha256"] = (
                    checkpoint_sha
                )

                configured = {
                    "backbone": 5e-7,
                    "text_encoder": 2.5e-6,
                    "film_projection": film_lr,
                }
                group_trace = [{"epoch": epoch, **configured} for epoch in range(2)]
                best_path = run / "best_learned_checkpoint.json"
                best_path.write_text(
                    json.dumps(
                        {
                            "best_epoch": 1,
                            "best_metric": 0.01,
                            "generator_group_initial_learning_rates": configured,
                            "generator_group_lr_trace": group_trace,
                            "discriminator_initial_learning_rate": 5e-7,
                            "discriminator_lr_trace": [
                                {"epoch": epoch, "lr": 5e-7} for epoch in range(2)
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                metrics_path = run / "training_metrics.csv"
                pd.DataFrame(
                    {
                        "epoch": range(3),
                        "g_lr_backbone": [5e-7] * 3,
                        "g_lr_text_encoder": [2.5e-6] * 3,
                        "g_lr_film_projection": [film_lr] * 3,
                        "d_lr": [5e-7] * 3,
                    }
                ).to_csv(metrics_path, index=False)
                config_path = run / "training_resolved_config.yaml"
                config_path.write_text(
                    json.dumps(
                        {
                            "generator_optimizer_profile": "film_unet_split_lr_v1",
                            "generator_learning_rate": 5e-7,
                            "generator_text_learning_rate": 2.5e-6,
                            "generator_film_learning_rate": film_lr,
                            "discriminator_learning_rate": 5e-7,
                        }
                    ),
                    encoding="utf-8",
                )
                job_spec_sha = _sha(f"job-spec:{job_id}")
                training_config_sha = _sha(f"training-config:{job_id}")
                artifacts = []
                for role, path in (
                    ("best_learned_checkpoint", best_path),
                    ("training_metrics_csv", metrics_path),
                    ("resolved_training_config", config_path),
                    ("generator_best_learned", generator_path),
                ):
                    artifacts.append(
                        {
                            "artifact_role": role,
                            "path": str(path.resolve()),
                            "sha256": analysis.sha256_file(path),
                            "size_bytes": path.stat().st_size,
                        }
                    )
                status_path = status_dir / f"{job_id}.json"
                status_path.write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "job_id": job_id,
                            "job_spec_sha256": job_spec_sha,
                            "training_config_sha256": training_config_sha,
                            "artifacts": artifacts,
                        }
                    ),
                    encoding="utf-8",
                )
                registry_jobs.append(
                    {
                        "job_id": job_id,
                        "job_spec_sha256": job_spec_sha,
                        "training_config_sha256": training_config_sha,
                    }
                )
            (root / "model_contract.json").write_text(
                json.dumps(
                    {
                        "generator_parameters": 827745,
                        "critic_parameters": 729157,
                    }
                ),
                encoding="utf-8",
            )
            (root / "registry/task_registry.json").write_text(
                json.dumps({"jobs": registry_jobs}), encoding="utf-8"
            )
            summary_path = analysis_dir / "training_summary.csv"
            summary.to_csv(summary_path, index=False)

            evidence = analysis.validate_training_evidence(summary_path, film)
            self.assertEqual(len(evidence.diagnostics), 20)
            self.assertEqual(len(evidence.lr_trace), 20 * 4 * 3)
            self.assertEqual(len(evidence.external_inputs), 20 * 5 + 2)
            self.assertEqual(
                set(evidence.lr_trace["parameter_group"]),
                {"backbone", "text_encoder", "film", "critic"},
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
