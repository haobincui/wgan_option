import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bow import transform_bow_counts  # noqa: E402
from film_wgan.text_transform import FilmWGANTextTransform, fit_text_transform  # noqa: E402
from scripts.rq2_pair.pair_features import (  # noqa: E402
    PAIR_FEATURE_SCHEMA_VERSION,
    build_fold_pair_features,
    build_pair_feature_coverage_lineage,
)
from scripts.rq2_pair import rq2_pair_experiment  # noqa: E402
from scripts.rq2_pair.rq2_pair_experiment import (  # noqa: E402
    _build_pairwise_differences,
    _cluster_bootstrap,
    _dm_hac_daily,
    _holm_adjust,
    _assert_loader_pair_ids_match_lineage,
    _existing_pair_feature_artifacts,
    _freeze_pair_feature_split_manifest,
    _link_or_copy,
    _preflight_reuse_root,
    _validate_model_matching,
)


def _serialized(values):
    return json.dumps([float(value) for value in values])


class TestRQ2PairFeatures(unittest.TestCase):
    def _write_fixture(self, root: Path):
        texts = [
            "alpha beta alpha",
            "beta gamma",
            "delta policy",
            "market rates",
            "credit risk",
            "validationonly phrase",
            "testonlytoken surprise",
            "late test article",
        ]
        news = pd.DataFrame(
            {
                "ArticleID": [f"article-{index}" for index in range(1, 9)],
                "SourceFile": [f"source-{index}.html" for index in range(1, 9)],
                "LP": texts,
                "LP_embedding": [
                    _serialized([index, index + 1.0, 1.0]) for index in range(1, 9)
                ],
            }
        )
        news_path = root / "news.xlsx"
        news.to_excel(news_path, index=False)

        scores = [
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.5],
            [0.5, 0.6, 0.7],
            [0.7, 0.8, 0.9],
            [0.2, 0.1, 0.0],
            [0.8, 0.2, 0.4],
            [0.9, 0.9, 0.9],
            [0.0, 0.0, 0.0],
        ]
        sentiment = pd.DataFrame(
            {
                "news_row_id": range(1, 9),
                "article_id": news["ArticleID"],
                "source_file": news["SourceFile"],
                "sentiment_embedding": [
                    _serialized(values + [0.0] * 5) for values in scores
                ],
                "sentiment_model_id": ["gpt-5.4-mini"] * 8,
                "sentiment_prompt_version": ["sun2026_zero_shot_chatgpt_v1"] * 8,
                "sentiment_parse_status": [
                    "cache_json",
                    "cache_json",
                    "cache_json",
                    "cache_json",
                    "cache_json",
                    "cache_json",
                    "cache_json",
                    "cache_json",
                ],
            }
        )
        sentiment_path = root / "sentiment.xlsx"
        sentiment.to_excel(sentiment_path, index=False)

        lineage = pd.DataFrame(
            [
                {
                    "surface_pair_id": "p0",
                    "split": "train",
                    "source_sample_ids": json.dumps(["news_1", "news_2"]),
                },
                {
                    "surface_pair_id": "p1",
                    "split": "train",
                    "source_sample_ids": json.dumps(["news_3"]),
                },
                {
                    "surface_pair_id": "p2",
                    "split": "train",
                    "source_sample_ids": json.dumps(["news_4"]),
                },
                {
                    "surface_pair_id": "p3",
                    "split": "train",
                    "source_sample_ids": json.dumps(["news_5"]),
                },
                {
                    "surface_pair_id": "p4",
                    "split": "val",
                    "source_sample_ids": json.dumps(["news_6"]),
                },
                {
                    "surface_pair_id": "p5",
                    "split": "test",
                    "source_sample_ids": json.dumps(["news_7"]),
                },
                {
                    "surface_pair_id": "p6",
                    "split": "train",
                    "source_sample_ids": json.dumps(["news_8"]),
                },
            ]
        )
        coverage_lineage_path = root / "coverage_lineage.csv"
        lineage.to_csv(coverage_lineage_path, index=False)
        fit_lineage_path = root / "fit_lineage.csv"
        lineage[lineage["surface_pair_id"] != "p6"].to_csv(
            fit_lineage_path,
            index=False,
        )
        workbook = root / "raw_vol_workbook.bin"
        workbook.write_bytes(b"raw-vol-fixture")
        return (
            coverage_lineage_path,
            fit_lineage_path,
            news_path,
            sentiment_path,
            workbook,
            scores,
        )

    def test_train_only_bow_and_sentiment_transforms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            coverage, fit, news, sentiment, workbook, scores = self._write_fixture(root)
            artifacts = build_fold_pair_features(
                fold="fixture",
                feature_coverage_lineage_path=coverage,
                fit_lineage_path=fit,
                news_workbook_path=news,
                sentiment_feature_path=sentiment,
                output_dir=root / "features",
                input_workbook_path=workbook,
                vocabulary_size=6,
                output_dim=3,
            )

            vocabulary_payload = json.loads(
                artifacts.bow_vocabulary_path.read_text(encoding="utf-8")
            )
            vocabulary = vocabulary_payload["terms"]
            self.assertEqual(len(vocabulary), 6)
            self.assertNotIn("testonlytoken", vocabulary)
            self.assertNotIn("validationonly", vocabulary)
            self.assertNotIn("late", vocabulary)

            bow = pd.read_csv(artifacts.bow_feature_path).set_index("surface_pair_id")
            expected_counts = transform_bow_counts(
                ["alpha beta alpha", "beta gamma"],
                vocabulary,
                ngram_range=(1, 2),
            ).sum(axis=0)
            expected = np.log1p(expected_counts)
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(
                np.asarray(
                    json.loads(bow.loc["p0", "text_embedding"]),
                    dtype=np.float32,
                ),
                expected,
                atol=1e-7,
            )
            self.assertEqual(
                sorted(json.loads(bow.loc["p0", "source_sample_ids"])),
                ["news_1", "news_2"],
            )

            sentiment_frame = pd.read_csv(artifacts.sentiment_feature_path).set_index(
                "surface_pair_id"
            )
            np.testing.assert_allclose(
                np.asarray(
                    json.loads(sentiment_frame.loc["p0", "text_embedding"]),
                    dtype=np.float32,
                ),
                np.mean(np.asarray(scores[:2]), axis=0),
                atol=1e-7,
            )
            train_pair_scores = np.asarray(
                [
                    np.mean(np.asarray(scores[:2]), axis=0),
                    scores[2],
                    scores[3],
                    scores[4],
                ],
                dtype=np.float32,
            )
            transform = FilmWGANTextTransform.load(artifacts.sentiment_transform_path)
            np.testing.assert_allclose(
                transform.mean,
                train_pair_scores.mean(axis=0),
                atol=1e-7,
            )
            self.assertEqual(transform.output_dim, 3)
            audit = pd.read_csv(artifacts.audit_path).set_index("surface_pair_id")
            self.assertEqual(int(audit.loc["p6", "empty_text_count"]), 0)
            self.assertTrue(bool(audit.loc["p6", "sentiment_all_zero"]))
            self.assertEqual(
                pd.read_csv(artifacts.bow_feature_path)["surface_pair_id"].tolist(),
                [f"p{index}" for index in range(7)],
            )
            manifest = json.loads(
                (root / "features/feature_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["schema_version"],
                PAIR_FEATURE_SCHEMA_VERSION,
            )
            self.assertEqual(manifest["feature_coverage_pair_count"], 7)
            self.assertEqual(manifest["fit_pair_count"], 6)
            self.assertEqual(manifest["train_pair_count"], 4)

    def test_coverage_lineage_includes_support_zero_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            coverage, fit, news, _sentiment, _workbook, _scores = self._write_fixture(
                root
            )
            source = pd.read_csv(coverage)
            rows = []
            global_index = 0
            for record in source.itertuples(index=False):
                for sample_id in json.loads(record.source_sample_ids):
                    rows.append(
                        {
                            "global_index": global_index,
                            "sample_id": sample_id,
                            "surface_pair_id": record.surface_pair_id,
                            "split": record.split,
                            "current_snapshot_time_utc": "2023-01-01T14:00:00Z",
                            "target_snapshot_time_utc": "2023-01-01T14:05:00Z",
                        }
                    )
                    global_index += 1
            split_manifest = root / "split_manifest.csv"
            pd.DataFrame(rows).to_csv(split_manifest, index=False)
            target = root / "pair_feature_coverage_lineage.csv"
            build_pair_feature_coverage_lineage(
                fold="fixture",
                split_manifest_path=split_manifest,
                news_workbook_path=news,
                fit_lineage_path=fit,
                output_path=target,
            )
            frame = pd.read_csv(target)
            self.assertEqual(
                frame["surface_pair_id"].tolist(), [f"p{i}" for i in range(7)]
            )
            self.assertIn("p6", set(frame["surface_pair_id"]))
            self.assertNotIn("p6", set(pd.read_csv(fit)["surface_pair_id"]))
            before = target.read_bytes()
            build_pair_feature_coverage_lineage(
                fold="fixture",
                split_manifest_path=split_manifest,
                news_workbook_path=news,
                fit_lineage_path=fit,
                output_path=target,
            )
            self.assertEqual(target.read_bytes(), before)

    def test_feature_publish_is_atomic_on_transform_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            coverage, fit, news, sentiment, workbook, _scores = self._write_fixture(
                root
            )
            output = root / "features"
            with self.assertRaisesRegex(ValueError, "exceeds"):
                build_fold_pair_features(
                    fold="fixture",
                    feature_coverage_lineage_path=coverage,
                    fit_lineage_path=fit,
                    news_workbook_path=news,
                    sentiment_feature_path=sentiment,
                    output_dir=output,
                    input_workbook_path=workbook,
                    vocabulary_size=6,
                    output_dim=5,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".features.staging-*")), [])

    def test_reuse_validates_dual_lineages_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            coverage, fit, news, sentiment, workbook, _scores = self._write_fixture(
                root
            )
            output = root / "features"
            artifacts = build_fold_pair_features(
                fold="fixture",
                feature_coverage_lineage_path=coverage,
                fit_lineage_path=fit,
                news_workbook_path=news,
                sentiment_feature_path=sentiment,
                output_dir=output,
                input_workbook_path=workbook,
                vocabulary_size=6,
                output_dim=3,
            )
            reused = _existing_pair_feature_artifacts(
                output,
                feature_coverage_lineage_path=coverage,
                fit_lineage_path=fit,
            )
            self.assertIsNotNone(reused)
            artifacts.bow_feature_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA validation"):
                _existing_pair_feature_artifacts(
                    output,
                    feature_coverage_lineage_path=coverage,
                    fit_lineage_path=fit,
                )

    def test_zscore_pad_reaches_shared_dimension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "input.bin"
            source.write_bytes(b"input")
            values = np.asarray(
                [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                dtype=np.float32,
            )
            transform = fit_text_transform(
                values,
                mode="zscore_pad",
                components=128,
                whiten=False,
                train_pair_ids=["a", "b", "c"],
                input_workbook_path=source,
                output_dim=128,
            )
            output = transform.transform(values)
            self.assertEqual(output.shape, (3, 128))
            np.testing.assert_array_equal(
                output[:, 3:],
                np.zeros((3, 125), dtype=np.float32),
            )


class TestRQ2StatisticsAndImport(unittest.TestCase):
    @staticmethod
    def _matched_model_frame():
        model_offsets = {
            "continued_no_text": 0.04,
            "lp": 0.01,
            "bow": 0.03,
            "llm_sentiment": 0.05,
        }
        rows = []
        for model, offset in model_offsets.items():
            for pair_index in range(2):
                rows.append(
                    {
                        "fold": "2023Q1",
                        "seed": 42,
                        "model": model,
                        "surface_pair_id": f"pair-{pair_index}",
                        "current_snapshot_time_utc": (
                            f"2023-01-0{pair_index + 1}T14:00:00Z"
                        ),
                        "target_snapshot_time_utc": (
                            f"2023-01-0{pair_index + 1}T14:05:00Z"
                        ),
                        "surface_mae": offset + pair_index * 0.001,
                        "short_atm_mae": offset + 0.01,
                        "supported_shortest_atm_abs_err": offset + 0.02,
                        "current_mae": 0.10,
                        "current_atm_short_pure_mae": 0.11,
                        "current_supported_shortest_atm_abs_err": 0.12,
                    }
                )
        return pd.DataFrame(rows)

    def test_pair_matching_and_difference_direction(self):
        samples = self._matched_model_frame()
        _validate_model_matching(samples)
        differences = _build_pairwise_differences(samples)
        primary = differences[
            (differences["contrast"] == "lp_vs_bow")
            & (differences["metric"] == "surface_mae")
        ]
        np.testing.assert_allclose(
            primary["difference"].to_numpy(dtype=np.float64),
            [0.02, 0.02],
            atol=1e-12,
        )
        self.assertTrue(primary["positive_means_focal_better"].all())

        broken = samples.copy()
        index = broken.index[broken["model"] == "bow"][0]
        broken.loc[index, "current_mae"] = 0.2
        with self.assertRaisesRegex(ValueError, "Current baseline mismatch"):
            _validate_model_matching(broken)

    def test_holm_cluster_bootstrap_and_hac_direction(self):
        self.assertEqual(
            _holm_adjust([0.01, 0.04, 0.03]),
            [0.03, 0.06, 0.06],
        )
        frame = pd.DataFrame(
            {
                "trading_day": [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                ],
                "difference": [0.2, 0.1, 0.3, 0.2, 0.4],
            }
        )
        bootstrap = _cluster_bootstrap(
            frame,
            iterations=1000,
            seed=20260722,
        )
        self.assertGreater(bootstrap["mean_difference"], 0.0)
        self.assertGreater(bootstrap["ci_95_lower"], 0.0)
        self.assertLess(bootstrap["p_one_sided_focal_better"], 0.05)
        hac = _dm_hac_daily(frame, max_lag=2)
        self.assertGreater(hac["dm_hac_statistic"], 0.0)
        self.assertLess(hac["p_one_sided_focal_better"], 0.05)

    def test_import_prefers_hardlink_and_verifies_sha(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "experiment"
            source = Path(tmpdir) / "source.bin"
            target = root / "inputs/source.bin"
            source.write_bytes(b"immutable-artifact")
            rows = []
            method = _link_or_copy(
                source,
                target,
                manifest_rows=rows,
                experiment_root=root,
                category="test",
            )
            self.assertIn(method, {"hardlink", "copy"})
            self.assertEqual(source.read_bytes(), target.read_bytes())
            self.assertEqual(rows[0]["source_path"], str(source))
            self.assertEqual(len(rows[0]["sha256"]), 64)
            if method == "hardlink":
                self.assertEqual(source.stat().st_ino, target.stat().st_ino)

    def test_loader_pair_ids_must_equal_post_support_lineage(self):
        lineage = pd.DataFrame(
            [
                {"split": "train", "surface_pair_id": "train-a"},
                {"split": "train", "surface_pair_id": "train-b"},
                {"split": "val", "surface_pair_id": "val-a"},
                {"split": "test", "surface_pair_id": "test-a"},
            ]
        )
        bundle = SimpleNamespace(
            train_items=[
                SimpleNamespace(surface_pair_id="train-a"),
                SimpleNamespace(surface_pair_id="train-b"),
            ],
            val_items=[SimpleNamespace(surface_pair_id="val-a")],
            test_items=[SimpleNamespace(surface_pair_id="test-a")],
        )
        _assert_loader_pair_ids_match_lineage(
            fold="fixture",
            variant="bow",
            bundle=bundle,
            fit_lineage=lineage,
        )
        bundle.train_items.reverse()
        with self.assertRaisesRegex(ValueError, "pair ID order"):
            _assert_loader_pair_ids_match_lineage(
                fold="fixture",
                variant="bow",
                bundle=bundle,
                fit_lineage=lineage,
            )

    def test_legacy_manifest_reuse_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "rq2"
            feature_dir = root / "inputs/folds/2023Q1/representations"
            feature_dir.mkdir(parents=True)
            manifest = feature_dir / "feature_manifest.json"
            manifest.write_text(
                json.dumps({"fold": "2023Q1"}) + "\n",
                encoding="utf-8",
            )
            before = manifest.read_bytes()
            with patch.object(
                rq2_pair_experiment,
                "FOLDS",
                {"2023Q1": {"counts": (1, 1, 1)}},
            ):
                with self.assertRaisesRegex(ValueError, "pre-dual-universe"):
                    _preflight_reuse_root(root)
            self.assertEqual(manifest.read_bytes(), before)

    def test_derived_split_excludes_only_pairs_absent_from_coverage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = pd.DataFrame(
                [
                    {
                        "global_index": 0,
                        "sample_id": "news_1",
                        "surface_pair_id": "kept",
                        "split": "train",
                    },
                    {
                        "global_index": 1,
                        "sample_id": "news_2",
                        "surface_pair_id": "invalid-text",
                        "split": "val",
                    },
                    {
                        "global_index": 2,
                        "sample_id": "news_3",
                        "surface_pair_id": "outside-fold",
                        "split": "excluded",
                    },
                ]
            )
            coverage = pd.DataFrame(
                [
                    {
                        "surface_pair_id": "kept",
                        "split": "train",
                        "source_sample_ids": json.dumps(["news_1"]),
                    }
                ]
            )
            source_path = root / "source.csv"
            coverage_path = root / "coverage.csv"
            output_path = root / "derived.csv"
            source.to_csv(source_path, index=False)
            coverage.to_csv(coverage_path, index=False)
            _freeze_pair_feature_split_manifest(
                source_split_manifest_path=source_path,
                feature_coverage_lineage_path=coverage_path,
                output_path=output_path,
            )
            derived = pd.read_csv(output_path).set_index("surface_pair_id")
            self.assertEqual(derived.loc["kept", "split"], "train")
            self.assertEqual(derived.loc["invalid-text", "split"], "excluded")
            self.assertEqual(derived.loc["outside-fold", "split"], "excluded")
            self.assertEqual(derived.loc["invalid-text", "source_split"], "val")
            self.assertEqual(
                derived.loc["kept", "pair_feature_coverage_schema_version"],
                PAIR_FEATURE_SCHEMA_VERSION,
            )

    def test_comparison_pipeline_writes_complete_primary_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "rq2"
            (root / "registry").mkdir(parents=True)
            (root / "checkpoint_selection").mkdir(parents=True)
            rows = []
            model_offsets = {
                "continued_no_text": 0.04,
                "lp": 0.01,
                "bow": 0.03,
                "llm_sentiment": 0.05,
            }
            variant_by_model = {
                "continued_no_text": "pair_pca_no_text_continued",
                "lp": "pair_pca_text_residual_pretrained",
                "bow": "pair_pca_bow_residual_pretrained",
                "llm_sentiment": "pair_sentiment_residual_pretrained",
            }
            for seed_index, seed in enumerate((1, 2, 3)):
                for model, offset in model_offsets.items():
                    summary = root / "summaries" / model / f"seed_{seed}.csv"
                    summary.parent.mkdir(parents=True, exist_ok=True)
                    frame = self._matched_model_frame()
                    frame = frame[frame["model"] == model].copy()
                    frame["seed"] = seed
                    seed_effect = (
                        seed_index * 0.0001 * (1.0 + list(model_offsets).index(model))
                    )
                    for metric in (
                        "surface_mae",
                        "short_atm_mae",
                        "supported_shortest_atm_abs_err",
                    ):
                        frame[metric] = frame[metric] + seed_effect
                    frame.to_csv(summary, index=False)
                    rows.append(
                        {
                            "fold": "2023Q1",
                            "seed": seed,
                            "model": model,
                            "variant": variant_by_model[model],
                            "status": "fixture",
                            "summary_path": str(summary),
                            "sample_count": 2,
                        }
                    )
            pd.DataFrame(rows).to_csv(
                root / "registry/generate_registry.csv",
                index=False,
            )
            pd.DataFrame([{"checkpoint": index} for index in range(6)]).to_csv(
                root / "checkpoint_selection/selected_checkpoints.csv",
                index=False,
            )

            with (
                patch.object(
                    rq2_pair_experiment,
                    "FOLDS",
                    {"2023Q1": {"counts": (2, 1, 2)}},
                ),
                patch.object(rq2_pair_experiment, "SEEDS", (1, 2, 3)),
            ):
                rq2_pair_experiment.build_comparison(
                    argparse.Namespace(
                        experiment_root=str(root),
                        bootstrap_iterations=100,
                        bootstrap_seed=7,
                    )
                )

            samples = pd.read_csv(
                root / "comparisons/development_rq2_test_sample_metrics.csv"
            )
            bootstrap = pd.read_csv(
                root / "comparisons/development_rq2_cluster_bootstrap_ci.csv"
            )
            dm = pd.read_csv(root / "comparisons/development_rq2_dm_hac_tests.csv")
            primary = pd.read_csv(
                root / "final_tables/development_rq2_primary_lp_vs_baselines.csv"
            )
            result = json.loads(
                (root / "final_tables/development_rq2_result_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(samples), 24)
            self.assertEqual(len(bootstrap), 18)
            self.assertEqual(len(dm), 18)
            self.assertEqual(len(primary), 2)
            self.assertTrue((primary["mean_difference"] > 0.0).all())
            self.assertEqual(
                set(primary["holm_family"]),
                {"primary_surface_lp_vs_two_representations"},
            )
            self.assertEqual(result["status"], "ok")

    def test_results_pipeline_records_completed_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "rq2"
            (root / "registry").mkdir(parents=True)
            args = argparse.Namespace(
                experiment_root=str(root),
                bootstrap_iterations=100,
                bootstrap_seed=7,
            )
            with (
                patch.object(
                    rq2_pair_experiment,
                    "collect_checkpoints",
                ) as collect,
                patch.object(
                    rq2_pair_experiment,
                    "generate_matrix",
                ) as generate,
                patch.object(
                    rq2_pair_experiment,
                    "build_comparison",
                    return_value=root,
                ) as compare,
            ):
                result = rq2_pair_experiment.results_pipeline(args)
            self.assertEqual(result, root)
            collect.assert_called_once()
            generate.assert_called_once()
            compare.assert_called_once()
            status = json.loads(
                (root / "registry/results_pipeline_status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["phase"], "completed")
            self.assertTrue(status["finished_at_utc"])
            self.assertEqual(status["error"], "")

    def test_results_pipeline_records_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "rq2"
            (root / "registry").mkdir(parents=True)
            args = argparse.Namespace(
                experiment_root=str(root),
                bootstrap_iterations=100,
                bootstrap_seed=7,
            )
            with patch.object(
                rq2_pair_experiment,
                "collect_checkpoints",
                side_effect=RuntimeError("fixture failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "fixture failure",
                ):
                    rq2_pair_experiment.results_pipeline(args)
            status = json.loads(
                (root / "registry/results_pipeline_status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["phase"], "collect_checkpoints")
            self.assertEqual(status["error_type"], "RuntimeError")
            self.assertEqual(status["error"], "fixture failure")


if __name__ == "__main__":
    unittest.main()
