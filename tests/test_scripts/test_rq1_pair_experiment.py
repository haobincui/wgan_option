import argparse
import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.config import FilmWGANTrainConfig  # noqa: E402
from film_wgan.data import create_train_val_bundle, write_split_manifest  # noqa: E402
from film_wgan.models import FilmWGANCritic, FilmWGANGenerator  # noqa: E402
from film_wgan.text_transform import FilmWGANTextTransform, fit_text_transform  # noqa: E402
from film_wgan.trainer import FilmWGANTrainer  # noqa: E402
from scripts.rq1_pair import rq1_pair_experiment  # noqa: E402


def _serialized(values):
    return json.dumps([float(value) for value in values])


def _write_pair_fixture(tmpdir: str) -> tuple[Path, Path]:
    workbook = Path(tmpdir) / "merged_vol.xlsx"
    news_workbook = Path(tmpdir) / "news.xlsx"
    strikes = [0.9, 1.1]
    maturities = [7.0, 30.0]
    merged_rows = []
    news_rows = []
    news_row_id = 1
    for pair_index in range(8):
        timestamp = pd.Timestamp("2022-01-03T14:00:00Z") + pd.Timedelta(days=pair_index)
        repeats = 2 if pair_index == 2 else 1
        for repeat in range(repeats):
            embedding = np.asarray(
                [pair_index + 1.0, repeat + 1.0, pair_index + 2.0, 0.5],
                dtype=np.float32,
            )
            article_id = f"article-{pair_index}-{repeat}"
            if pair_index == 2 and repeat == 1:
                article_id = "article-2-0"
                embedding = np.asarray([3.0, 1.0, 4.0, 0.5], dtype=np.float32)
            current = np.full(4, 0.20 + pair_index * 0.001, dtype=np.float32)
            target = current + 0.002
            merged_rows.append(
                {
                    "sample_id": f"news_{news_row_id}",
                    "news_timestamp_utc": timestamp.isoformat(),
                    "current_snapshot_time_utc": timestamp.isoformat(),
                    "target_snapshot_time_utc": (timestamp + pd.Timedelta(minutes=5)).isoformat(),
                    "hd_embedding": _serialized([0.0, 0.0]),
                    "lp_embedding": _serialized(embedding),
                    "strike_grid": _serialized(strikes),
                    "maturity_days_grid": _serialized(maturities),
                    "current_surface_flat": _serialized(current),
                    "target_surface_flat": _serialized(target),
                    "training_candidate_flag": 1,
                    "pair_quality_label": "usable",
                }
            )
            news_rows.append(
                {
                    "SourceFile": f"source-{news_row_id}.txt",
                    "ArticleID": article_id,
                    "LP_embedding": _serialized(embedding),
                }
            )
            news_row_id += 1
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(merged_rows).to_excel(writer, sheet_name="gan_input_ready", index=False)
    pd.DataFrame(news_rows).to_excel(news_workbook, index=False)
    return workbook, news_workbook


def _pair_config(
    workbook: Path,
    news_workbook: Path,
    transform_path: Path,
    *,
    text_mode: str = "lp",
) -> FilmWGANTrainConfig:
    return FilmWGANTrainConfig(
        data_path=str(workbook),
        news_workbook_path=str(news_workbook),
        sample_unit="surface_pair",
        text_embedding_mode=text_mode,
        text_preprocessing_mode="pca",
        text_transform_path=str(transform_path),
        text_pca_components=2,
        normalize_text_embedding=False,
        split_strategy="grouped_chronological",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        conditioning_mode="residual_film",
        critic_conditioning_mode="projection",
        text_hidden_dim=8,
        text_out_dim=4,
        gen_base_channels=4,
        disc_base_channels=4,
        fusion_hidden_dim=16,
        noise_dim=2,
        batch_size=8,
        cuda=False,
    )


class TestPairTextData(unittest.TestCase):
    def test_pair_pooling_lineage_dedup_and_shared_pca_zero_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook, news_workbook = _write_pair_fixture(tmpdir)
            transform_path = Path(tmpdir) / "text_transform.npz"
            manifest_path = Path(tmpdir) / "split_manifest.csv"
            config = _pair_config(workbook, news_workbook, transform_path)
            write_split_manifest(config, manifest_path)
            text_bundle = create_train_val_bundle(
                replace(config, split_manifest_path=str(manifest_path))
            )

            self.assertEqual(
                (text_bundle.train_samples, text_bundle.val_samples, text_bundle.test_samples),
                (4, 2, 2),
            )
            self.assertEqual(text_bundle.embedding_dim, 2)
            self.assertTrue(transform_path.is_file())
            duplicate_pair = next(
                sample for sample in text_bundle.train_items if sample.metadata["news_count"] == 2
            )
            self.assertEqual(duplicate_pair.metadata["unique_embedding_count"], 1)
            self.assertEqual(len(duplicate_pair.metadata["article_ids"]), 1)
            self.assertAlmostEqual(
                float(np.linalg.norm(duplicate_pair.raw_text_embedding)),
                1.0,
                places=6,
            )

            no_text_bundle = create_train_val_bundle(
                replace(
                    config,
                    split_manifest_path=str(manifest_path),
                    text_embedding_mode="zero_lp",
                )
            )
            self.assertEqual(no_text_bundle.embedding_dim, text_bundle.embedding_dim)
            self.assertTrue(
                all(np.array_equal(sample.text_embedding, np.zeros(2, dtype=np.float32)) for sample in no_text_bundle.all_items)
            )
            batch = next(iter(no_text_bundle.train_loader))
            self.assertEqual(len(batch), 6)
            self.assertTrue(torch.equal(batch[1], torch.zeros_like(batch[1])))
            self.assertTrue(torch.equal(batch[5], torch.zeros_like(batch[5])))

    def test_pair_surface_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook, news_workbook = _write_pair_fixture(tmpdir)
            frame = pd.read_excel(workbook, sheet_name="gan_input_ready")
            duplicate_indexes = frame.index[
                frame["current_snapshot_time_utc"] == frame.loc[2, "current_snapshot_time_utc"]
            ].tolist()
            self.assertEqual(len(duplicate_indexes), 2)
            frame.loc[duplicate_indexes[1], "current_surface_flat"] = _serialized([0.9] * 4)
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                frame.to_excel(writer, sheet_name="gan_input_ready", index=False)
            config = _pair_config(workbook, news_workbook, Path(tmpdir) / "transform.npz")
            with self.assertRaisesRegex(ValueError, "inconsistent current_surface"):
                create_train_val_bundle(config)

    def test_transform_round_trip_uses_only_supplied_train_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            values = np.asarray(
                [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
                dtype=np.float32,
            )
            input_path = Path(tmpdir) / "input.bin"
            input_path.write_bytes(b"fixed-input")
            transform = fit_text_transform(
                values,
                mode="pca",
                components=2,
                whiten=False,
                train_pair_ids=["a", "b", "c"],
                input_workbook_path=input_path,
            )
            artifact, _metadata = transform.save(Path(tmpdir) / "transform.npz")
            restored = FilmWGANTextTransform.load(artifact)
            np.testing.assert_allclose(
                transform.transform(values),
                restored.transform(values),
                atol=1e-7,
            )
            self.assertEqual(restored.metadata["train_pair_count"], 3)


class TestResidualFiLMArchitecture(unittest.TestCase):
    @staticmethod
    def _generator() -> FilmWGANGenerator:
        return FilmWGANGenerator(
            surface_height=4,
            surface_width=4,
            embedding_dim=2,
            noise_dim=3,
            base_channels=4,
            res_blocks=1,
            text_hidden_dim=8,
            text_out_dim=4,
            fusion_hidden_dim=16,
            conditioning_mode="residual_film",
            text_dropout=0.3,
            text_gate_initial_value=0.0,
        )

    def test_zero_gate_is_exact_nested_no_text_prediction(self):
        torch.manual_seed(7)
        generator = self._generator().eval()
        current = torch.randn(3, 1, 4, 4)
        noise = torch.randn(3, 3)
        first_text = torch.randn(3, 2)
        second_text = torch.randn(3, 2)
        with torch.no_grad():
            first = generator(current, first_text, noise=noise, has_text=torch.ones(3))
            second = generator(current, second_text, noise=noise, has_text=torch.ones(3))
            no_text = generator(current, second_text, noise=noise, has_text=torch.zeros(3))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, no_text))

    def test_mask_blocks_adapter_after_gate_opens(self):
        torch.manual_seed(11)
        generator = self._generator().eval()
        generator.text_gate.data.fill_(0.7)
        current = torch.randn(2, 1, 4, 4)
        noise = torch.randn(2, 3)
        first_text = torch.randn(2, 2)
        second_text = torch.randn(2, 2)
        with torch.no_grad():
            masked_first = generator(current, first_text, noise=noise, has_text=torch.zeros(2))
            masked_second = generator(current, second_text, noise=noise, has_text=torch.zeros(2))
            conditioned_first = generator(current, first_text, noise=noise, has_text=torch.ones(2))
            conditioned_second = generator(current, second_text, noise=noise, has_text=torch.ones(2))
        self.assertTrue(torch.equal(masked_first, masked_second))
        self.assertFalse(torch.equal(conditioned_first, conditioned_second))

    def test_projection_critic_mask_removes_text_bias(self):
        torch.manual_seed(13)
        critic = FilmWGANCritic(
            surface_height=4,
            surface_width=4,
            embedding_dim=2,
            base_channels=4,
            res_blocks=1,
            text_hidden_dim=8,
            text_out_dim=4,
            fusion_hidden_dim=16,
            conditioning_mode="residual_film",
            critic_conditioning_mode="projection",
            text_dropout=0.0,
        ).eval()
        future = torch.randn(2, 1, 4, 4)
        current = torch.randn(2, 1, 4, 4)
        with torch.no_grad():
            first = critic(future, current, torch.randn(2, 2), has_text=torch.zeros(2))
            second = critic(future, current, torch.randn(2, 2), has_text=torch.zeros(2))
        self.assertTrue(torch.equal(first, second))

    def test_parent_checkpoint_freezes_then_unfreezes_backbone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook, news_workbook = _write_pair_fixture(tmpdir)
            transform_path = Path(tmpdir) / "transform.npz"
            manifest_path = Path(tmpdir) / "manifest.csv"
            base = _pair_config(workbook, news_workbook, transform_path, text_mode="zero_lp")
            write_split_manifest(base, manifest_path)
            no_text_bundle = create_train_val_bundle(
                replace(base, split_manifest_path=str(manifest_path))
            )
            parent_generator = FilmWGANGenerator(
                surface_height=2,
                surface_width=2,
                embedding_dim=no_text_bundle.embedding_dim,
                noise_dim=base.noise_dim,
                base_channels=base.gen_base_channels,
                res_blocks=base.gen_res_blocks,
                text_hidden_dim=base.text_hidden_dim,
                text_out_dim=base.text_out_dim,
                fusion_hidden_dim=base.fusion_hidden_dim,
                conditioning_mode="residual_film",
                text_dropout=base.text_dropout,
                text_gate_initial_value=base.text_gate_initial_value,
            )
            parent_path = Path(tmpdir) / "parent.pt"
            torch.save(
                {
                    "config": asdict(base),
                    "surface_shape": [2, 2],
                    "embedding_dim": no_text_bundle.embedding_dim,
                    "conditioning_mode": "residual_film",
                    "text_transform_sha256": no_text_bundle.text_transform_sha256,
                    "generator_state_dict": parent_generator.state_dict(),
                },
                parent_path,
            )
            text_config = replace(
                base,
                split_manifest_path=str(manifest_path),
                text_embedding_mode="lp",
                initial_generator_checkpoint_path=str(parent_path),
                freeze_backbone_epochs=5,
                output_root=str(Path(tmpdir) / "outputs"),
            )
            trainer = FilmWGANTrainer(text_config)
            trainer._ensure_runtime_prepared()
            trainer.setup()
            self.assertTrue(trainer._backbone_frozen)
            self.assertFalse(any(parameter.requires_grad for parameter in trainer.generator.backbone_parameters()))

            continued_config = replace(
                base,
                split_manifest_path=str(manifest_path),
                initial_generator_checkpoint_path=str(parent_path),
                freeze_backbone_epochs=5,
                output_root=str(Path(tmpdir) / "continued-outputs"),
            )
            continued_trainer = FilmWGANTrainer(continued_config)
            continued_trainer._ensure_runtime_prepared()
            continued_trainer.setup()
            self.assertEqual(
                trainer._parent_checkpoint_sha256,
                continued_trainer._parent_checkpoint_sha256,
            )
            for name, value in trainer.generator.state_dict().items():
                self.assertTrue(torch.equal(value, continued_trainer.generator.state_dict()[name]))
            for name, value in trainer.critic.state_dict().items():
                self.assertTrue(torch.equal(value, continued_trainer.critic.state_dict()[name]))

            trainer._update_backbone_freeze_state(6)
            self.assertFalse(trainer._backbone_frozen)
            self.assertTrue(all(parameter.requires_grad for parameter in trainer.generator.backbone_parameters()))


class TestPairRollingComparison(unittest.TestCase):
    def test_no_text_continuation_uses_the_paired_stage_b_schedule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir) / "parent.pt"
            parent.write_bytes(b"parent")
            transform = Path(tmpdir) / "transform.npz"
            transform.write_bytes(b"transform")
            fold_config = {"training": {"text_transform_path": str(transform)}}

            continued = rq1_pair_experiment._variant_overrides(
                rq1_pair_experiment.CONTINUATION_VARIANT,
                fold_config=fold_config,
                output_root=Path(tmpdir) / "continued",
                seed=42,
                parent_checkpoint=parent,
            )
            matched_text = rq1_pair_experiment._variant_overrides(
                rq1_pair_experiment.TEXT_RESIDUAL_VARIANT,
                fold_config=fold_config,
                output_root=Path(tmpdir) / "text",
                seed=42,
                parent_checkpoint=parent,
            )

            self.assertEqual(len(rq1_pair_experiment.VARIANTS), 7)
            self.assertEqual(rq1_pair_experiment.VARIANTS[1], rq1_pair_experiment.CONTINUATION_VARIANT)
            self.assertEqual(continued["text_embedding_mode"], "zero_lp")
            self.assertEqual(continued["initial_generator_checkpoint_path"], str(parent))
            self.assertEqual(continued["freeze_backbone_epochs"], 5)
            self.assertEqual(continued["lambda_film"], 0.0)
            self.assertEqual(continued["lambda_mismatch"], 0.0)
            for field in (
                "seed",
                "text_preprocessing_mode",
                "text_transform_path",
                "conditioning_mode",
                "critic_conditioning_mode",
                "freeze_backbone_epochs",
                "initial_generator_checkpoint_path",
            ):
                self.assertEqual(continued[field], matched_text[field])

            with self.assertRaisesRegex(FileNotFoundError, "No-text continuation"):
                rq1_pair_experiment._variant_overrides(
                    rq1_pair_experiment.CONTINUATION_VARIANT,
                    fold_config=fold_config,
                    output_root=Path(tmpdir) / "missing-parent",
                    seed=42,
                    parent_checkpoint=None,
                )

    def test_paired_stage_audit_requires_one_parent_and_transform(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir) / "parent.pt"
            parent.write_bytes(b"parent")
            transform = Path(tmpdir) / "transform.npz"
            transform.write_bytes(b"transform")
            selected = pd.DataFrame(
                [
                    {
                        "fold": "2023Q1",
                        "seed": 42,
                        "variant": rq1_pair_experiment.PARENT_VARIANT,
                        "checkpoint_path": str(parent),
                        "checkpoint_sha256": rq1_pair_experiment.sha256_file(parent),
                    }
                ]
            )
            parent_training = {"text_transform_path": str(transform)}
            resolved = {
                ("2023Q1", 42, rq1_pair_experiment.PARENT_VARIANT): {
                    "training": parent_training
                }
            }
            for variant in rq1_pair_experiment.PAIRED_STAGE_B_VARIANTS:
                overrides = rq1_pair_experiment._variant_overrides(
                    variant,
                    fold_config={"training": parent_training},
                    output_root=Path(tmpdir) / variant,
                    seed=42,
                    parent_checkpoint=parent,
                )
                resolved[("2023Q1", 42, variant)] = {"training": overrides}

            with (
                patch.object(rq1_pair_experiment, "FOLDS", {"2023Q1": {}}),
                patch.object(rq1_pair_experiment, "SEEDS", (42,)),
            ):
                audit, failures = rq1_pair_experiment._paired_stage_audit(selected, resolved)
                self.assertEqual(len(audit), 3)
                self.assertFalse(failures)
                self.assertEqual(set(audit["status"]), {"ok"})

                wrong_parent = Path(tmpdir) / "wrong-parent.pt"
                wrong_parent.write_bytes(b"wrong")
                text_key = ("2023Q1", 42, rq1_pair_experiment.TEXT_RESIDUAL_VARIANT)
                bad_training = dict(resolved[text_key]["training"])
                bad_training["initial_generator_checkpoint_path"] = str(wrong_parent)
                bad_resolved = dict(resolved)
                bad_resolved[text_key] = {"training": bad_training}
                failed_audit, failures = rq1_pair_experiment._paired_stage_audit(
                    selected,
                    bad_resolved,
                )
                failed_text = failed_audit[
                    failed_audit["variant"] == rq1_pair_experiment.TEXT_RESIDUAL_VARIANT
                ].iloc[0]
                self.assertEqual(failed_text["status"], "failed")
                self.assertTrue(failures)

    def test_raw_workbook_validation_rejects_svi_surface(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook = Path(tmpdir) / "surface_models.xlsx"
            pd.DataFrame(
                {
                    "surface_model": ["raw", "raw"],
                    "source_timezone": ["Europe/London", "Europe/London"],
                    "timestamp_parse_status": ["ok", "ok"],
                }
            ).to_excel(
                workbook,
                sheet_name="gan_input_ready",
                index=False,
            )
            self.assertEqual(
                rq1_pair_experiment._validate_raw_surface_workbook(
                    workbook,
                    sheet_name="gan_input_ready",
                ),
                "raw",
            )
            pd.DataFrame({"surface_model": ["svi"]}).to_excel(
                workbook,
                sheet_name="gan_input_ready",
                index=False,
            )
            with self.assertRaisesRegex(ValueError, "requires surface_model='raw'"):
                rq1_pair_experiment._validate_raw_surface_workbook(
                    workbook,
                    sheet_name="gan_input_ready",
                )

    def test_cli_overrides_do_not_quote_string_values(self):
        self.assertEqual(
            rq1_pair_experiment._cli_override("text_alignment_mode", "matched"),
            "text_alignment_mode=matched",
        )
        self.assertEqual(
            rq1_pair_experiment._cli_override("normalize_text_embedding", False),
            "normalize_text_embedding=false",
        )
        self.assertEqual(
            rq1_pair_experiment._cli_override("checkpoints_path", ""),
            "checkpoints_path=",
        )

    def test_comparison_archive_uses_all_fold_pairs_and_development_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "rq1_pair_text_raw_vol_rolling_test"
            registry_rows = []
            variant_offsets = {
                variant: index * 0.0001
                for index, variant in enumerate(rq1_pair_experiment.VARIANTS)
            }
            for fold, specification in rq1_pair_experiment.FOLDS.items():
                pair_count = int(specification["counts"][2])
                for seed in rq1_pair_experiment.SEEDS:
                    for variant in rq1_pair_experiment.VARIANTS:
                        summary_path = (
                            root
                            / "synthetic"
                            / variant
                            / fold
                            / f"seed_{seed}"
                            / "summary.csv"
                        )
                        summary_path.parent.mkdir(parents=True, exist_ok=True)
                        rows = []
                        for index in range(pair_count):
                            base = 0.02 + index * 1e-7 + seed * 1e-8
                            row = {
                                "surface_pair_id": f"{fold}-{index}",
                                "current_snapshot_time_utc": (
                                    pd.Timestamp("2023-01-03T14:00:00Z")
                                    + pd.Timedelta(days=index % 40)
                                ).isoformat(),
                                "surface_mae": base + variant_offsets[variant],
                                "short_atm_mae": base * 0.8 + variant_offsets[variant],
                                "supported_shortest_atm_abs_err": (
                                    base * 0.9 + variant_offsets[variant]
                                ),
                                "energy_score": base,
                                "variogram_score": base * 0.5,
                                "coverage_50": 0.5,
                                "coverage_80": 0.8,
                                "coverage_90": 0.9,
                                "interval_width_50": 0.01,
                                "interval_width_80": 0.02,
                                "interval_width_90": 0.03,
                                "calibration_error": 0.0,
                                "scenario_spread": 0.01,
                                "mc_surface_mae_se": 0.0001,
                                "calendar_violation_rate": 0.0,
                                "butterfly_violation_rate": 0.0,
                            }
                            rows.append(row)
                        pd.DataFrame(rows).to_csv(summary_path, index=False)
                        registry_rows.append(
                            {
                                "fold": fold,
                                "seed": seed,
                                "variant": variant,
                                "status": "complete",
                                "summary_path": str(summary_path),
                                "sample_count": pair_count,
                            }
                        )
            for fold, specification in rq1_pair_experiment.FOLDS.items():
                pair_count = int(specification["counts"][2])
                lineage_path = (
                    root
                    / "inputs"
                    / "folds"
                    / fold
                    / "pair_lineage_audit.csv"
                )
                lineage_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "fold": [fold] * pair_count,
                        "split": ["test"] * pair_count,
                        "surface_pair_id": [
                            f"{fold}-{index}" for index in range(pair_count)
                        ],
                        "exact_embedding_duplicate_with_train": np.zeros(
                            pair_count,
                            dtype=int,
                        ),
                        "exact_text_duplicate_with_train": np.zeros(
                            pair_count,
                            dtype=int,
                        ),
                        "near_text_candidate_duplicate_with_train": np.zeros(
                            pair_count,
                            dtype=int,
                        ),
                    }
                ).to_csv(lineage_path, index=False)
            registry_path = root / "registry/generate_registry.csv"
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(registry_rows).to_csv(registry_path, index=False)

            rq1_pair_experiment.build_comparison(
                argparse.Namespace(
                    experiment_root=str(root),
                    bootstrap_iterations=100,
                    bootstrap_seed=20260722,
                )
            )
            primary = pd.read_csv(root / "final_tables/development_rq1_primary_test.csv")
            controlled_primary = pd.read_csv(
                root / "final_tables/development_rq1_primary_controlled_incremental_text.csv"
            )
            diagnostics = pd.read_csv(
                root / "final_tables/development_rq1_parent_continuation_diagnostics.csv"
            )
            primary_all = pd.read_csv(
                root / "final_tables/development_rq1_primary_all_metrics.csv"
            )
            overall = pd.read_csv(
                root / "final_tables/development_rq1_model_overall_metrics.csv"
            )
            seed_tests = pd.read_csv(root / "comparisons/development_seed_level_tests.csv")
            validation = json.loads((root / "validation_summary.json").read_text(encoding="utf-8"))
            result_summary = json.loads(
                (root / "final_tables/development_rq1_result_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(primary), 1)
            pd.testing.assert_frame_equal(primary, controlled_primary)
            self.assertEqual(primary.loc[0, "focal_variant"], rq1_pair_experiment.TEXT_RESIDUAL_VARIANT)
            self.assertEqual(primary.loc[0, "baseline_variant"], rq1_pair_experiment.CONTINUATION_VARIANT)
            self.assertEqual(
                set(diagnostics["contrast"]),
                {"text_vs_parent", "continuation_effect"},
            )
            self.assertEqual(len(seed_tests), len(rq1_pair_experiment.CONTRASTS) * 3)
            self.assertEqual(validation["status"], "ok")
            self.assertTrue(validation["development_only"])
            self.assertEqual(
                validation["primary_baseline_variant"],
                rq1_pair_experiment.CONTINUATION_VARIANT,
            )
            self.assertEqual(
                validation["primary_difference_direction"],
                "continued_no_text_error_minus_text_error",
            )
            self.assertEqual(set(primary_all["metric"]), set(rq1_pair_experiment.POINT_METRICS))
            self.assertEqual(set(overall["variant"]), set(rq1_pair_experiment.VARIANTS))
            self.assertEqual(result_summary["status"], "ok")
            self.assertEqual(
                result_summary["difference_direction"],
                "no_text_error_minus_text_error",
            )
            expected_rows = (
                sum(int(specification["counts"][2]) for specification in rq1_pair_experiment.FOLDS.values())
                * len(rq1_pair_experiment.SEEDS)
                * len(rq1_pair_experiment.VARIANTS)
            )
            self.assertEqual(validation["sample_metric_rows"], expected_rows)

    def test_results_pipeline_records_completed_stages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "rq1_pair_text_raw_vol_rolling_test"
            (root / "checkpoint_selection").mkdir(parents=True)
            (root / "registry").mkdir(parents=True)
            (root / "final_tables").mkdir(parents=True)
            expected_runs = (
                len(rq1_pair_experiment.FOLDS)
                * len(rq1_pair_experiment.SEEDS)
                * len(rq1_pair_experiment.VARIANTS)
            )

            def fake_collect(_args):
                pd.DataFrame({"run": range(expected_runs)}).to_csv(
                    root / "checkpoint_selection/selected_checkpoints.csv",
                    index=False,
                )
                return root

            def fake_generate(_args):
                pd.DataFrame(
                    {
                        "run": range(expected_runs),
                        "sample_count": np.ones(expected_runs, dtype=int),
                    }
                ).to_csv(root / "registry/generate_registry.csv", index=False)
                return root

            def fake_compare(_args):
                (root / "final_tables/development_rq1_result_summary.json").write_text(
                    '{"status": "ok"}\n',
                    encoding="utf-8",
                )
                return root

            args = argparse.Namespace(
                experiment_root=str(root),
                bootstrap_iterations=100,
                bootstrap_seed=20260722,
            )
            with (
                mock.patch.object(rq1_pair_experiment, "_assert_py312"),
                mock.patch.object(
                    rq1_pair_experiment,
                    "collect_checkpoints",
                    side_effect=fake_collect,
                ),
                mock.patch.object(
                    rq1_pair_experiment,
                    "generate_matrix",
                    side_effect=fake_generate,
                ),
                mock.patch.object(
                    rq1_pair_experiment,
                    "build_comparison",
                    side_effect=fake_compare,
                ),
            ):
                result = rq1_pair_experiment.run_results_pipeline(args)

            self.assertEqual(result, root)
            status = json.loads(
                (root / "registry/results_pipeline_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["current_stage"], "completed")
            self.assertEqual(
                status["stages"]["collect_checkpoints"]["selected_checkpoint_count"],
                expected_runs,
            )
            self.assertEqual(
                status["stages"]["generate_test_results"]["generated_run_count"],
                expected_runs,
            )
            self.assertEqual(status["stages"]["build_comparison"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
