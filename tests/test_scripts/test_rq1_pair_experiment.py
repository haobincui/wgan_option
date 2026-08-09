import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.config import (  # noqa: E402
    FilmWGANTrainConfig,
    load_sample_config,
    parse_sample_overrides,
)
from film_wgan.data import create_train_val_bundle, write_split_manifest  # noqa: E402
from film_wgan.models import FilmWGANCritic, FilmWGANGenerator  # noqa: E402
from film_wgan.text_transform import FilmWGANTextTransform, fit_text_transform  # noqa: E402
from film_wgan.trainer import FilmWGANTrainer, module_state_sha256  # noqa: E402
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

    def test_strict_timing_still_rejects_news_surface_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook, news_workbook = _write_pair_fixture(tmpdir)
            frame = pd.read_excel(workbook, sheet_name="gan_input_ready")
            news_time = pd.Timestamp(frame.loc[0, "news_timestamp_utc"])
            frame.loc[0, "current_snapshot_time_utc"] = (
                news_time + pd.Timedelta(minutes=1)
            ).isoformat()
            frame.loc[0, "target_snapshot_time_utc"] = (
                news_time + pd.Timedelta(minutes=6)
            ).isoformat()
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                frame.to_excel(writer, sheet_name="gan_input_ready", index=False)

            config = _pair_config(
                workbook,
                news_workbook,
                Path(tmpdir) / "transform.npz",
            )
            with self.assertRaisesRegex(
                ValueError,
                "violates news=current timestamp",
            ):
                create_train_val_bundle(config)

    def test_forward_aligned_timing_accepts_valid_shift_and_rejects_leakage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook, news_workbook = _write_pair_fixture(tmpdir)
            frame = pd.read_excel(workbook, sheet_name="gan_input_ready")
            news_time = pd.Timestamp(frame.loc[0, "news_timestamp_utc"])
            origin = news_time + pd.Timedelta(minutes=20)
            frame.loc[0, "news_alignment_mode"] = "forward_valid_pair"
            frame.loc[0, "news_available_time_utc"] = news_time.isoformat()
            frame.loc[0, "effective_origin_utc"] = origin.isoformat()
            frame.loc[0, "origin_shift_minutes"] = 20
            frame.loc[0, "alignment_type"] = "session_shift"
            frame.loc[0, "current_snapshot_time_utc"] = origin.isoformat()
            frame.loc[0, "target_snapshot_time_utc"] = (
                origin + pd.Timedelta(minutes=5)
            ).isoformat()
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                frame.to_excel(writer, sheet_name="gan_input_ready", index=False)

            config = _pair_config(
                workbook,
                news_workbook,
                Path(tmpdir) / "transform.npz",
            )
            bundle = create_train_val_bundle(config)
            self.assertEqual(len(bundle.all_items), 8)

            frame.loc[0, "effective_origin_utc"] = (
                news_time - pd.Timedelta(minutes=1)
            ).isoformat()
            frame.loc[0, "origin_shift_minutes"] = -1
            frame.loc[0, "alignment_type"] = "intraday_shift"
            frame.loc[0, "current_snapshot_time_utc"] = frame.loc[
                0,
                "effective_origin_utc",
            ]
            frame.loc[0, "target_snapshot_time_utc"] = (
                news_time + pd.Timedelta(minutes=4)
            ).isoformat()
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                frame.to_excel(writer, sheet_name="gan_input_ready", index=False)
            with self.assertRaisesRegex(
                ValueError,
                "origin before news availability",
            ):
                create_train_val_bundle(
                    replace(
                        config,
                        text_transform_path=str(
                            Path(tmpdir) / "invalid_transform.npz"
                        ),
                    )
                )

    def test_exchange_session_timing_accepts_next_open_plus_tolerance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook, news_workbook = _write_pair_fixture(tmpdir)
            frame = pd.read_excel(workbook, sheet_name="gan_input_ready")
            news_time = pd.Timestamp(frame.loc[0, "news_timestamp_utc"])
            scheduled_origin = news_time + pd.Timedelta(minutes=30)
            origin = scheduled_origin + pd.Timedelta(minutes=5)
            target = origin + pd.Timedelta(minutes=5)
            frame.loc[0, "news_alignment_mode"] = "exchange_session"
            frame.loc[0, "news_available_time_utc"] = news_time.isoformat()
            frame.loc[0, "effective_origin_utc"] = origin.isoformat()
            frame.loc[0, "origin_shift_minutes"] = 35
            frame.loc[0, "alignment_type"] = "closed_to_next_open"
            frame.loc[0, "publication_market_state"] = "closed"
            frame.loc[0, "scheduled_origin_utc"] = (
                scheduled_origin.isoformat()
            )
            frame.loc[0, "origin_tolerance_minutes_used"] = 5
            frame.loc[0, "session_shift_minutes"] = 30
            frame.loc[0, "session_shift_reason"] = "daily_halt"
            frame.loc[0, "session_id"] = "test_session"
            frame.loc[0, "session_open_utc"] = (
                scheduled_origin.isoformat()
            )
            frame.loc[0, "session_close_utc"] = (
                scheduled_origin + pd.Timedelta(hours=23)
            ).isoformat()
            frame.loc[0, "current_window_start_utc"] = (
                origin - pd.Timedelta(minutes=5)
            ).isoformat()
            frame.loc[0, "current_window_end_utc"] = origin.isoformat()
            frame.loc[0, "target_window_start_utc"] = origin.isoformat()
            frame.loc[0, "target_window_end_utc"] = target.isoformat()
            frame.loc[0, "current_snapshot_time_utc"] = origin.isoformat()
            frame.loc[0, "target_snapshot_time_utc"] = target.isoformat()
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                frame.to_excel(
                    writer,
                    sheet_name="gan_input_ready",
                    index=False,
                )

            config = _pair_config(
                workbook,
                news_workbook,
                Path(tmpdir) / "transform.npz",
            )
            bundle = create_train_val_bundle(config)
            self.assertEqual(len(bundle.all_items), 8)

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

    def test_transition_matching_trainer_steps_are_finite_and_keep_critic_frozen_for_g(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook, news_workbook = _write_pair_fixture(tmpdir)
            transform_path = Path(tmpdir) / "transform.npz"
            manifest_path = Path(tmpdir) / "manifest.csv"
            config = replace(
                _pair_config(workbook, news_workbook, transform_path),
                critic_conditioning_mode="transition_matching",
                critic_text_dropout=0.0,
                gradient_penalty_mode="support_masked",
                lambda_mismatch=0.0,
                lambda_critic_matching=0.1,
                lambda_generator_matching=0.01,
                matching_min_supported_cells=1,
                matching_negative_count=2,
                adv_warmup_epochs=0,
                adv_ramp_epochs=0,
                use_calendar_constraint=False,
                use_butterfly_constraint=False,
                use_smooth_constraint=False,
                output_root=str(Path(tmpdir) / "training"),
            )
            write_split_manifest(config, manifest_path)
            config = replace(config, split_manifest_path=str(manifest_path))
            trainer = FilmWGANTrainer(config)
            trainer._ensure_runtime_prepared()
            trainer.setup()
            trainer.config = replace(
                trainer.config,
                adv_warmup_epochs=2,
                adv_ramp_epochs=5,
            )
            trainer._current_epoch = 2
            self.assertEqual(trainer._adversarial_ramp_factor(), 0.0)
            trainer._current_epoch = 3
            self.assertAlmostEqual(trainer._adversarial_ramp_factor(), 0.2)
            trainer._current_epoch = 7
            self.assertEqual(trainer._adversarial_ramp_factor(), 1.0)
            trainer.config = replace(
                trainer.config,
                adv_warmup_epochs=0,
                adv_ramp_epochs=0,
            )
            trainer._current_epoch = 1
            batch = next(iter(trainer.bundle.train_loader))
            (
                current_features,
                text_features,
                real_delta_norm,
                current_flat,
                target_flat,
                has_text,
                sample_indices,
            ) = [tensor.to(trainer.device) for tensor in batch]
            support = torch.ones_like(current_flat)
            torch.testing.assert_close(
                trainer._matching_text_bank[sample_indices],
                text_features,
            )

            d_metrics = trainer._discriminator_step(
                current_features,
                text_features,
                real_delta_norm,
                current_flat,
                target_flat,
                has_text,
                support,
                sample_indices,
            )
            self.assertTrue(all(math.isfinite(float(value)) for value in d_metrics.values()))
            self.assertGreater(d_metrics["d_matching"], 0.0)
            self.assertEqual(d_metrics["d_matching_eligible_fraction"], 1.0)
            self.assertAlmostEqual(
                d_metrics["d_total"],
                d_metrics["d_wgan"]
                + d_metrics["gp"]
                + config.lambda_critic_matching * d_metrics["d_matching"],
                places=5,
            )
            self.assertTrue(
                any(parameter.grad is not None for parameter in trainer.critic.parameters())
            )

            critic_hash_before_g = module_state_sha256(trainer.critic)
            generator_hash_before_g = module_state_sha256(trainer.generator)
            g_metrics = trainer._generator_step(
                current_features,
                text_features,
                current_flat,
                target_flat,
                has_text,
                support,
                sample_indices,
            )
            self.assertTrue(all(math.isfinite(float(value)) for value in g_metrics.values()))
            self.assertGreater(g_metrics["g_matching"], 0.0)
            self.assertAlmostEqual(
                g_metrics["g_total"],
                g_metrics["g_adv_effective_lambda"] * g_metrics["g_adv"]
                + g_metrics["g_matching_effective_lambda"]
                * g_metrics["g_matching"],
                places=5,
            )
            self.assertTrue(all(parameter.grad is None for parameter in trainer.critic.parameters()))
            self.assertEqual(module_state_sha256(trainer.critic), critic_hash_before_g)
            self.assertNotEqual(module_state_sha256(trainer.generator), generator_hash_before_g)

    def test_generator_matching_uses_the_delivered_clamped_transition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook, news_workbook = _write_pair_fixture(tmpdir)
            transform_path = Path(tmpdir) / "transform.npz"
            manifest_path = Path(tmpdir) / "manifest.csv"
            config = replace(
                _pair_config(workbook, news_workbook, transform_path),
                critic_conditioning_mode="transition_matching",
                critic_text_dropout=0.0,
                gradient_penalty_mode="support_masked",
                lambda_mismatch=0.0,
                lambda_critic_matching=0.1,
                lambda_generator_matching=0.01,
                matching_min_supported_cells=1,
                matching_negative_count=2,
                adv_warmup_epochs=0,
                adv_ramp_epochs=0,
                use_calendar_constraint=False,
                use_butterfly_constraint=False,
                use_smooth_constraint=False,
                output_root=str(Path(tmpdir) / "training"),
            )
            write_split_manifest(config, manifest_path)
            trainer = FilmWGANTrainer(
                replace(config, split_manifest_path=str(manifest_path))
            )
            trainer._ensure_runtime_prepared()
            trainer.setup()
            trainer._current_epoch = 1
            batch = next(iter(trainer.bundle.train_loader))
            (
                current_features,
                text_features,
                _real_delta_norm,
                current_flat,
                target_flat,
                has_text,
                sample_indices,
            ) = [tensor.to(trainer.device) for tensor in batch]
            support = torch.ones_like(current_flat)
            extreme = torch.full_like(current_flat, 1.0e6)
            captured_transitions = []
            original_matching_logits = trainer.critic.matching_logits

            def extreme_forward(*_args, **_kwargs):
                anchor = next(trainer.generator.parameters()).sum() * 0.0
                return extreme + anchor

            def capture_matching_logits(transition_surface, *args, **kwargs):
                captured_transitions.append(transition_surface.detach().clone())
                return original_matching_logits(transition_surface, *args, **kwargs)

            with (
                patch.object(trainer.generator, "forward", side_effect=extreme_forward),
                patch.object(
                    trainer.critic,
                    "matching_logits",
                    side_effect=capture_matching_logits,
                ),
            ):
                metrics = trainer._generator_step(
                    current_features,
                    text_features,
                    current_flat,
                    target_flat,
                    has_text,
                    support,
                    sample_indices,
                )

            delivered = torch.full_like(current_flat, 5.0)
            delivered_delta = torch.log(delivered) - torch.log(current_flat)
            expected = (
                delivered_delta - trainer.normalization.delta_mean
            ) / trainer.normalization.delta_std.clamp_min(1.0e-6)
            self.assertTrue(captured_transitions)
            torch.testing.assert_close(
                captured_transitions[0],
                expected.view_as(captured_transitions[0]),
                rtol=0.0,
                atol=1.0e-5,
            )
            self.assertEqual(metrics["g_saturation_rate"], 1.0)
            self.assertGreater(metrics["g_transition_delivery_max_abs_gap"], 1.0)

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

            transition_config = replace(
                text_config,
                critic_conditioning_mode="transition_matching",
                critic_text_dropout=0.0,
                gradient_penalty_mode="support_masked",
                lambda_mismatch=0.0,
                lambda_critic_matching=0.1,
                lambda_generator_matching=0.01,
                matching_min_supported_cells=1,
                output_root=str(Path(tmpdir) / "v2-parent-rejection"),
            )
            transition_trainer = FilmWGANTrainer(transition_config)
            transition_trainer._ensure_runtime_prepared()
            with self.assertRaisesRegex(ValueError, "schema-5 v2 parent"):
                transition_trainer.setup()

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
    def test_v2_training_rejects_a_different_clean_commit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "inputs/configs/train_rq1_pair_textbase.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "training": {
                            "critic_conditioning_mode": "transition_matching",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "inputs/git_state.txt").write_text(
                "commit=frozen-commit\n## rq1-film-wgan-v2\n",
                encoding="utf-8",
            )
            clean_status = SimpleNamespace(stdout="")
            different_head = SimpleNamespace(stdout="different-commit\n")
            with patch.object(
                rq1_pair_experiment.subprocess,
                "run",
                side_effect=[clean_status, different_head],
            ):
                with self.assertRaisesRegex(RuntimeError, "across commits"):
                    rq1_pair_experiment._assert_frozen_v2_worktree(root)

    def test_launch_registry_upsert_preserves_prior_runs_and_replaces_same_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = {
                "fold": "2023Q1",
                "seed": 42,
                "variant": rq1_pair_experiment.PARENT_VARIANT,
                "status": "completed",
                "run_dir": "first-parent",
            }
            rq1_pair_experiment._upsert_launch_registry(root, [first])

            second = {
                "fold": "2023Q1",
                "seed": 42,
                "variant": rq1_pair_experiment.TEXT_RESIDUAL_VARIANT,
                "status": "completed",
                "run_dir": "first-text",
            }
            updated_first = {
                **first,
                "status": "reused",
                "run_dir": "reused-parent",
                "checkpoint": "parent.pt",
            }
            registry_path = rq1_pair_experiment._upsert_launch_registry(
                root,
                [second, updated_first],
            )

            registry = pd.read_csv(registry_path).sort_values("variant")
            self.assertEqual(len(registry), 2)
            parent = registry[
                registry["variant"] == rq1_pair_experiment.PARENT_VARIANT
            ].iloc[0]
            text = registry[
                registry["variant"]
                == rq1_pair_experiment.TEXT_RESIDUAL_VARIANT
            ].iloc[0]
            self.assertEqual(parent["status"], "reused")
            self.assertEqual(parent["run_dir"], "reused-parent")
            self.assertEqual(parent["checkpoint"], "parent.pt")
            self.assertEqual(text["run_dir"], "first-text")

    def test_transition_matching_run_reuse_requires_v2_checkpoint_protocol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            checkpoint = run_dir / "checkpoints/film_wgan_best.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(
                {
                    "checkpoint_schema_version": 4,
                    "critic_architecture_version": "legacy_v1",
                },
                checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "non-v2 completed run"):
                rq1_pair_experiment._validate_run_protocol(
                    run_dir,
                    critic_mode="transition_matching",
                )

            torch.save(
                {
                    "checkpoint_schema_version": 5,
                    "training_protocol_version": "film_wgan_transition_matching_v2",
                    "critic_architecture_version": "transition_matching_v2",
                },
                checkpoint,
            )
            rq1_pair_experiment._validate_run_protocol(
                run_dir,
                critic_mode="transition_matching",
            )

            torch.save(
                {
                    "checkpoint_schema_version": 5,
                    "training_protocol_version": "film_wgan_v1_compatible",
                    "critic_architecture_version": "legacy_v1",
                },
                checkpoint,
            )
            rq1_pair_experiment._validate_run_protocol(
                run_dir,
                critic_mode="inherit",
                require_schema5=True,
            )

    def test_generate_matrix_respects_frozen_validation_only_protocol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "inputs/configs/train_rq1_pair_textbase.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "training": {},
                        "generate_result": {
                            "split": "val",
                            "output_dir": "validation_pilot_json",
                            "selection_mode": "all",
                            "selection_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            run_dir = root / "run"
            checkpoint = run_dir / "checkpoints/model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "training_resolved_config.yaml").write_text(
                yaml.safe_dump({"training": {}}, sort_keys=False),
                encoding="utf-8",
            )
            selected_path = root / "checkpoint_selection/selected_checkpoints.csv"
            selected_path.parent.mkdir(parents=True)
            (root / "registry").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "run_dir": str(run_dir),
                        "checkpoint_path": str(checkpoint),
                        "fold": "2023Q1",
                        "seed": 42,
                        "variant": rq1_pair_experiment.TEXT_RESIDUAL_VARIANT,
                    }
                ]
            ).to_csv(selected_path, index=False)

            observed_commands = []

            def fake_run(command, *, log_path):
                del log_path
                observed_commands.append(command)
                output_dir = run_dir / "validation_pilot_json"
                output_dir.mkdir(parents=True)
                pd.DataFrame({"sample": [0, 1]}).to_csv(
                    output_dir / "summary.csv",
                    index=False,
                )

            with (
                patch.object(rq1_pair_experiment, "_assert_py312"),
                patch.object(
                    rq1_pair_experiment,
                    "_fold_counts",
                    return_value=(4, 2, 3),
                ),
                patch.object(rq1_pair_experiment, "_run", side_effect=fake_run),
            ):
                rq1_pair_experiment.generate_matrix(
                    argparse.Namespace(experiment_root=str(root))
                )

            self.assertEqual(len(observed_commands), 1)
            command = observed_commands[0]
            merged_generation = yaml.safe_load(
                Path(command[command.index("--config") + 1]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(merged_generation["generate_result"]["split"], "val")
            self.assertEqual(
                merged_generation["generate_result"]["output_dir"],
                "validation_pilot_json",
            )
            registry = pd.read_csv(root / "registry/generate_registry.csv")
            self.assertEqual(int(registry.loc[0, "sample_count"]), 2)
            self.assertEqual(registry.loc[0, "split"], "val")
            with self.assertRaisesRegex(RuntimeError, "Validation-only pilot"):
                rq1_pair_experiment.build_comparison(
                    argparse.Namespace(
                        experiment_root=str(root),
                        bootstrap_iterations=10,
                        bootstrap_seed=123,
                    )
                )

    def test_default_seed_set_is_frozen_to_fifteen_unique_seeds(self):
        self.assertEqual(len(rq1_pair_experiment.SEEDS), 15)
        self.assertEqual(len(set(rq1_pair_experiment.SEEDS)), 15)
        self.assertEqual(rq1_pair_experiment.SEEDS[:3], (42, 202, 404))

    def test_experiment_seed_manifest_preserves_legacy_seed_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            design = root / "inputs/experiment_design.json"
            design.parent.mkdir(parents=True)
            design.write_text(
                json.dumps({"seeds": [42, 202, 404]}),
                encoding="utf-8",
            )
            self.assertEqual(
                rq1_pair_experiment._experiment_seeds(root),
                (42, 202, 404),
            )

    def test_compact_fifteen_seed_config_keeps_only_primary_checkpoints(self):
        training = rq1_pair_experiment._read_yaml(
            ROOT / "configs/film_wgan/train_rq1_pair_textbase_15seed.yaml"
        )["training"]
        self.assertEqual(int(training["save_every"]), 0)
        self.assertEqual(training["extra_checkpoint_metrics"], [])
        self.assertEqual(training["checkpoint_metric"], "val_mae")

    def test_short_atm_support_audit_uses_distinct_train_supported_region(self):
        params = json.dumps(
            {
                "business_days": [7, 45],
                "percent_strikes": [
                    [0.98, 1.0, 1.02, 1.04],
                    [0.98, 1.0, 1.02, 1.04],
                ],
                "implied_vols": [
                    [0.20, 0.19, 0.20, 0.21],
                    [0.22, 0.21, 0.22, 0.23],
                ],
            }
        )
        samples = [
            SimpleNamespace(
                strike_grid=np.asarray([0.98, 1.0, 1.02, 1.04]),
                maturity_days_grid=np.asarray([7.0, 14.0, 21.0, 30.0, 45.0]),
                evaluation_support_mask=np.ones((5, 4), dtype=bool),
                metadata={
                    "current_surface_param_json": params,
                    "target_surface_param_json": params,
                },
            )
            for _ in range(4)
        ]
        split_items = (("train", samples), ("val", samples), ("test", samples))
        support_rows = rq1_pair_experiment._short_atm_support_rows(
            fold="2023Q1",
            split_items=split_items,
            atm_range=0.02,
            selected_max_days=21.0,
        )
        rq1_pair_experiment._validate_selected_short_atm_rows(support_rows)
        selected_train = next(
            row
            for row in support_rows
            if row["split"] == "train" and row["selected"] == 1
        )
        self.assertEqual(selected_train["grid_atm_strike_count"], 3)
        self.assertEqual(selected_train["grid_short_maturity_count"], 3)
        self.assertAlmostEqual(
            selected_train["local_supported_cell_share"],
            9.0 / 20.0,
        )
        self.assertEqual(selected_train["eligible_pair_ratio"], 1.0)

        maturity_rows = rq1_pair_experiment._raw_maturity_distribution_rows(
            fold="2023Q1",
            split_items=split_items,
        )
        train_combined = next(
            row
            for row in maturity_rows
            if row["split"] == "train" and row["surface_side"] == "combined"
        )
        self.assertEqual(train_combined["raw_slice_count"], 16)
        self.assertAlmostEqual(train_combined["p50_days"], 26.0)

    def test_short_atm_support_audit_rejects_full_support_mask(self):
        sample = SimpleNamespace(
            strike_grid=np.asarray([0.98, 1.0, 1.02]),
            maturity_days_grid=np.asarray([7.0, 21.0]),
            evaluation_support_mask=np.ones((2, 3), dtype=bool),
        )
        rows = rq1_pair_experiment._short_atm_support_rows(
            fold="2023Q1",
            split_items=(("train", [sample] * 4),),
            atm_range=0.10,
            selected_max_days=45.0,
            maturity_candidates=(45.0,),
        )
        with self.assertRaisesRegex(
            ValueError,
            "selected short-ATM support equals full support",
        ):
            rq1_pair_experiment._validate_selected_short_atm_rows(rows)

    def test_rq1_and_rq2_configs_share_audited_short_atm_definition(self):
        rq1_training = rq1_pair_experiment._read_yaml(
            rq1_pair_experiment.DEFAULT_CONFIG
        )["training"]
        rq2_training = rq1_pair_experiment._read_yaml(
            ROOT / "configs/film_wgan/train_rq2_pair_textbase.yaml"
        )["training"]
        for training in (rq1_training, rq2_training):
            self.assertAlmostEqual(float(training["atm_short_range"]), 0.02)
            self.assertAlmostEqual(float(training["atm_short_max_days"]), 21.0)
            self.assertAlmostEqual(float(training["recon_atm_range"]), 0.02)
            self.assertAlmostEqual(
                float(training["recon_atm_short_end_max_days"]),
                21.0,
            )

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

    def test_v2_variant_overrides_keep_matching_only_for_text_residual_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir) / "parent.pt"
            parent.write_bytes(b"parent")
            transform = Path(tmpdir) / "transform.npz"
            transform.write_bytes(b"transform")
            fold_config = {
                "training": {
                    "text_transform_path": str(transform),
                    "critic_conditioning_mode": "transition_matching",
                    "lambda_mismatch": 0.0,
                    "lambda_critic_matching": 0.1,
                    "lambda_generator_matching": 0.01,
                }
            }

            parent_overrides = rq1_pair_experiment._variant_overrides(
                rq1_pair_experiment.PARENT_VARIANT,
                fold_config=fold_config,
                output_root=Path(tmpdir) / "parent",
                seed=42,
                parent_checkpoint=None,
            )
            continued = rq1_pair_experiment._variant_overrides(
                rq1_pair_experiment.CONTINUATION_VARIANT,
                fold_config=fold_config,
                output_root=Path(tmpdir) / "continued",
                seed=42,
                parent_checkpoint=parent,
            )
            matched = rq1_pair_experiment._variant_overrides(
                rq1_pair_experiment.TEXT_RESIDUAL_VARIANT,
                fold_config=fold_config,
                output_root=Path(tmpdir) / "matched",
                seed=42,
                parent_checkpoint=parent,
            )
            shuffled = rq1_pair_experiment._variant_overrides(
                rq1_pair_experiment.SHUFFLED_RESIDUAL_VARIANT,
                fold_config=fold_config,
                output_root=Path(tmpdir) / "shuffled",
                seed=42,
                parent_checkpoint=parent,
            )

            for overrides in (parent_overrides, continued):
                self.assertEqual(overrides["critic_conditioning_mode"], "transition_matching")
                self.assertEqual(overrides["lambda_critic_matching"], 0.0)
                self.assertEqual(overrides["lambda_generator_matching"], 0.0)
            for overrides in (matched, shuffled):
                self.assertEqual(overrides["critic_conditioning_mode"], "transition_matching")
                self.assertEqual(overrides["lambda_mismatch"], 0.0)
                self.assertEqual(overrides["lambda_critic_matching"], 0.1)
                self.assertEqual(overrides["lambda_generator_matching"], 0.01)

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

                legacy_resolved = dict(resolved)
                for variant in rq1_pair_experiment.PAIRED_STAGE_B_VARIANTS:
                    key = ("2023Q1", 42, variant)
                    legacy_training = dict(legacy_resolved[key]["training"])
                    legacy_training.pop("lambda_critic_matching", None)
                    legacy_training.pop("lambda_generator_matching", None)
                    legacy_resolved[key] = {"training": legacy_training}
                legacy_audit, legacy_failures = (
                    rq1_pair_experiment._paired_stage_audit(
                        selected,
                        legacy_resolved,
                    )
                )
                self.assertFalse(legacy_failures)
                self.assertEqual(set(legacy_audit["status"]), {"ok"})

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
                                "short_atm_mae": (
                                    float("nan")
                                    if index % 5 == 0
                                    else base * 0.8 + variant_offsets[variant]
                                ),
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
            short_primary = primary_all[
                primary_all["metric"] == "short_atm_mae"
            ].iloc[0]
            expected_short_pairs = sum(
                int(specification["counts"][2])
                - (int(specification["counts"][2]) + 4) // 5
                for specification in rq1_pair_experiment.FOLDS.values()
            )
            self.assertEqual(
                int(short_primary["pair_count"]),
                expected_short_pairs,
            )
            self.assertTrue(
                np.isfinite(float(short_primary["mean_difference"]))
            )
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


class TestRQ1ProtocolV3Orchestration(unittest.TestCase):
    def test_v3_config_and_gpu0_launcher_are_fail_closed(self):
        payload = rq1_pair_experiment._read_yaml(
            ROOT
            / "configs/film_wgan/train_rq1_pair_textbase_v3_pilot.yaml"
        )
        training = payload["training"]
        generate = payload["generate_result"]
        self.assertEqual(
            training["training_protocol_version"],
            rq1_pair_experiment.TRAINING_PROTOCOL_VERSION_V3,
        )
        self.assertEqual(training["text_alignment_plan_path"], "")
        self.assertEqual(training["matching_negative_source_plan_path"], "")
        self.assertEqual(int(training["scheduler_horizon_epochs"]), 60)
        self.assertEqual(int(training["diagnostics_schema_version"]), 1)
        self.assertEqual(int(training["checkpoint_warmup_epochs"]), 0)
        self.assertEqual(generate["split"], "val")

        launcher = (
            ROOT / "scripts/rq1_pair/start_v3_pilot_gpu0.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("CUDA_VISIBLE_DEVICES=0", launcher)
        self.assertIn("FILM_WGAN_MAX_PARALLEL:-5", launcher)
        self.assertIn("Phase 1/3", launcher)
        self.assertIn("Phase 2/3", launcher)
        self.assertIn("Phase 3/3", launcher)
        self.assertIn("wait -n -p", launcher)
        self.assertIn("setsid --wait", launcher)
        self.assertIn("terminate_and_reap_children", launcher)
        self.assertIn("summarize-validation-pilot", launcher)

        result = subprocess.run(
            [str(ROOT / "scripts/rq1_pair/start_v3_pilot_gpu0.sh")],
            cwd=ROOT,
            env={
                **dict(__import__("os").environ),
                "FILM_WGAN_LAUNCHER_WAIT_SELF_TEST": "1",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("fast-failure", result.stderr)
        self.assertNotIn("unexpectedly continued", result.stderr)

    def test_v3_gpu0_launcher_signal_cleanup_reaps_child(self):
        process = subprocess.Popen(
            [str(ROOT / "scripts/rq1_pair/start_v3_pilot_gpu0.sh")],
            cwd=ROOT,
            env={
                **dict(os.environ),
                "FILM_WGAN_LAUNCHER_SIGNAL_SELF_TEST": "1",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertIsNotNone(process.stdout)
        ready_line = process.stdout.readline().strip()
        self.assertTrue(
            ready_line.startswith("signal self-test ready: "),
            ready_line,
        )
        child_pid = int(ready_line.rsplit(" ", maxsplit=1)[1])
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)

        self.assertEqual(process.returncode, 130, (stdout, stderr))
        self.assertIn("received INT", stderr)
        self.assertNotIn("unbound variable", stderr)
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)

    def test_v3_epoch_policy_uses_continuation_anchor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transform = Path(tmpdir) / "transform.npz"
            transform.write_bytes(b"transform")
            parent = Path(tmpdir) / "parent.pt"
            parent.write_bytes(b"parent")
            fold_config = {
                "training": {
                    "text_transform_path": str(transform),
                    "critic_conditioning_mode": "transition_matching",
                    "lambda_critic_matching": 0.1,
                    "lambda_generator_matching": 0.01,
                    # Exercise the bug directly: Stage-A must not inherit the
                    # continuation-only selection warmup from a fold config.
                    "checkpoint_warmup_epochs": 10,
                    "training_protocol_version": (
                        rq1_pair_experiment.TRAINING_PROTOCOL_VERSION_V3
                    ),
                }
            }
            parent_overrides = rq1_pair_experiment._variant_overrides(
                rq1_pair_experiment.PARENT_VARIANT,
                fold_config=fold_config,
                output_root=Path(tmpdir) / "parent",
                seed=42,
                parent_checkpoint=None,
            )
            continued = rq1_pair_experiment._variant_overrides(
                rq1_pair_experiment.CONTINUATION_VARIANT,
                fold_config=fold_config,
                output_root=Path(tmpdir) / "continued",
                seed=42,
                parent_checkpoint=parent,
            )
            matched = rq1_pair_experiment._variant_overrides(
                rq1_pair_experiment.TEXT_RESIDUAL_VARIANT,
                fold_config=fold_config,
                output_root=Path(tmpdir) / "matched",
                seed=42,
                parent_checkpoint=parent,
                continuation_anchor_epoch=37,
            )
            shuffled = rq1_pair_experiment._variant_overrides(
                rq1_pair_experiment.SHUFFLED_RESIDUAL_VARIANT,
                fold_config=fold_config,
                output_root=Path(tmpdir) / "shuffled",
                seed=42,
                parent_checkpoint=parent,
                continuation_anchor_epoch=37,
            )
            resolved_parent = {**fold_config["training"], **parent_overrides}
            resolved_continued = {**fold_config["training"], **continued}
            resolved_matched = {**fold_config["training"], **matched}
            resolved_shuffled = {**fold_config["training"], **shuffled}

            self.assertEqual(resolved_parent["num_epochs"], 60)
            self.assertEqual(resolved_parent["scheduler_horizon_epochs"], 60)
            self.assertEqual(resolved_parent["checkpoint_warmup_epochs"], 0)
            self.assertEqual(resolved_continued["num_epochs"], 100)
            self.assertEqual(resolved_continued["early_stopping_patience"], 15)
            self.assertEqual(resolved_continued["scheduler_horizon_epochs"], 100)
            self.assertEqual(resolved_continued["checkpoint_warmup_epochs"], 10)
            for resolved in (resolved_matched, resolved_shuffled):
                self.assertEqual(resolved["num_epochs"], 37)
                self.assertFalse(resolved["use_early_stopping"])
                self.assertEqual(resolved["scheduler_horizon_epochs"], 100)
                self.assertEqual(resolved["checkpoint_warmup_epochs"], 0)

            for variant in (
                rq1_pair_experiment.TEXT_RESIDUAL_VARIANT,
                rq1_pair_experiment.SHUFFLED_RESIDUAL_VARIANT,
            ):
                run_dir = Path(tmpdir) / f"comparison-{variant}"
                (run_dir / "checkpoints").mkdir(parents=True)
                (run_dir / "metrics").mkdir()
                torch.save(
                    {"epoch": 37},
                    run_dir / "checkpoints/film_wgan_final.pt",
                )
                (run_dir / "metrics/best_checkpoint.json").write_text(
                    json.dumps({"best_epoch": 12}),
                    encoding="utf-8",
                )
                checkpoint, epoch = rq1_pair_experiment._comparison_checkpoint(
                    run_dir,
                    variant=variant,
                    continuation_anchor_epoch=37,
                )
                self.assertEqual(checkpoint.name, "film_wgan_final.pt")
                self.assertEqual(epoch, 37)

    def test_v3_paired_stage_audit_requires_common_scheduler_trace_and_checkpoint_roles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transform = root / "transform.npz"
            alignment = root / "text_alignment_plan.csv"
            negatives = root / "transition_matching_negative_source_plan.csv"
            transform.write_bytes(b"transform")
            alignment.write_bytes(b"alignment")
            negatives.write_bytes(b"negatives")
            alignment_sha = rq1_pair_experiment.sha256_file(alignment)
            negative_sha = rq1_pair_experiment.sha256_file(negatives)
            parent = root / "parent.pt"
            parent.write_bytes(b"parent")
            base_training = {
                "training_protocol_version": (
                    rq1_pair_experiment.TRAINING_PROTOCOL_VERSION_V3
                ),
                "critic_conditioning_mode": "transition_matching",
                "text_transform_path": str(transform),
                "text_alignment_plan_path": str(alignment),
                "matching_negative_source_plan_path": str(negatives),
                "lambda_critic_matching": 0.1,
                "lambda_generator_matching": 0.01,
            }
            fold_config = {"training": base_training}
            selected_rows = [
                {
                    "fold": "2023Q1",
                    "seed": 42,
                    "variant": rq1_pair_experiment.PARENT_VARIANT,
                    "checkpoint_path": str(parent),
                    "checkpoint_sha256": rq1_pair_experiment.sha256_file(parent),
                }
            ]
            resolved = {
                ("2023Q1", 42, rq1_pair_experiment.PARENT_VARIANT): {
                    "training": {"text_transform_path": str(transform)}
                }
            }
            comparison_epoch = 20
            common_trace = pd.DataFrame(
                {
                    "epoch": np.arange(1, comparison_epoch + 1),
                    "lr_generator": np.linspace(9.9e-5, 8.0e-5, comparison_epoch),
                    "lr_critic": np.linspace(1.98e-4, 1.6e-4, comparison_epoch),
                }
            )
            for variant in rq1_pair_experiment.PAIRED_STAGE_B_VARIANTS:
                output_root = root / variant
                run_dir = output_root / "run_001"
                checkpoints = run_dir / "checkpoints"
                metrics = run_dir / "metrics"
                checkpoints.mkdir(parents=True)
                metrics.mkdir(parents=True)
                overrides = rq1_pair_experiment._variant_overrides(
                    variant,
                    fold_config=fold_config,
                    output_root=output_root,
                    seed=42,
                    parent_checkpoint=parent,
                    continuation_anchor_epoch=(
                        comparison_epoch
                        if variant != rq1_pair_experiment.CONTINUATION_VARIANT
                        else None
                    ),
                )
                resolved[("2023Q1", 42, variant)] = {
                    "training": {**base_training, **overrides}
                }
                checkpoint_payload = {
                    "checkpoint_schema_version": 6,
                    "training_protocol_version": (
                        rq1_pair_experiment.TRAINING_PROTOCOL_VERSION_V3
                    ),
                    "critic_architecture_version": "transition_matching_v2",
                    "run_fingerprint_sha256": "a" * 64,
                    "epoch": comparison_epoch,
                    "initial_generator_state_sha256": "g" * 64,
                    "initial_critic_state_sha256": "c" * 64,
                    "text_alignment_plan_sha256": alignment_sha,
                    "matching_negative_source_plan_sha256": negative_sha,
                    "scheduler_horizon_epochs": 100,
                }
                torch.save(checkpoint_payload, checkpoints / "film_wgan_best.pt")
                torch.save(checkpoint_payload, checkpoints / "film_wgan_final.pt")
                (metrics / "best_checkpoint.json").write_text(
                    json.dumps(
                        {
                            "best_epoch": comparison_epoch,
                            "checkpoint_metric": "val_mae",
                        }
                    ),
                    encoding="utf-8",
                )
                common_trace.to_csv(metrics / "training_metrics.csv", index=False)
                selected_checkpoint = checkpoints / (
                    "film_wgan_best.pt"
                    if variant == rq1_pair_experiment.CONTINUATION_VARIANT
                    else "film_wgan_final.pt"
                )
                selected_rows.append(
                    {
                        "fold": "2023Q1",
                        "seed": 42,
                        "variant": variant,
                        "checkpoint_path": str(selected_checkpoint),
                        "checkpoint_sha256": rq1_pair_experiment.sha256_file(
                            selected_checkpoint
                        ),
                        "text_alignment_plan_sha256": alignment_sha,
                        "matching_negative_source_plan_sha256": negative_sha,
                        "selected_epoch": comparison_epoch,
                        "run_dir": str(run_dir),
                    }
                )

            selected = pd.DataFrame(selected_rows)
            audit, failures = rq1_pair_experiment._paired_stage_audit(
                selected,
                resolved,
                seeds=(42,),
                folds=("2023Q1",),
            )
            self.assertFalse(failures)
            self.assertEqual(set(audit["status"]), {"ok"})
            self.assertEqual(audit["scheduler_trace_sha256"].nunique(), 1)

            shuffled_metrics = (
                root
                / rq1_pair_experiment.SHUFFLED_RESIDUAL_VARIANT
                / "run_001/metrics/training_metrics.csv"
            )
            changed = pd.read_csv(shuffled_metrics)
            changed.loc[changed.index[-1], "lr_generator"] *= 0.5
            changed.to_csv(shuffled_metrics, index=False)
            _audit, failures = rq1_pair_experiment._paired_stage_audit(
                selected,
                resolved,
                seeds=(42,),
                folds=("2023Q1",),
            )
            shuffled_failure = next(
                failure
                for failure in failures
                if failure["variant"]
                == rq1_pair_experiment.SHUFFLED_RESIDUAL_VARIANT
            )
            self.assertIn("scheduler_trace_mismatch", shuffled_failure["errors"])

    def test_v3_collects_and_generates_exact_twelve_validation_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "inputs/configs").mkdir(parents=True)
            (root / "inputs/folds/2023Q1").mkdir(parents=True)
            (root / "registry").mkdir(parents=True)
            (root / "checkpoint_selection").mkdir(parents=True)
            design = {
                "experiment_design_schema_version": 2,
                "training_protocol_version": (
                    rq1_pair_experiment.TRAINING_PROTOCOL_VERSION_V3
                ),
                "run_folds": list(rq1_pair_experiment.V3_PILOT_FOLDS),
                "run_seeds": list(rq1_pair_experiment.V3_PILOT_SEEDS),
                "run_variants": list(rq1_pair_experiment.V3_PILOT_VARIANTS),
                "expected_training_runs": 12,
            }
            (root / "inputs/experiment_design.json").write_text(
                json.dumps(design), encoding="utf-8"
            )
            transform = root / "inputs/folds/2023Q1/text_transform.npz"
            alignment = root / "inputs/folds/2023Q1/text_alignment_plan.csv"
            negatives = (
                root
                / "inputs/folds/2023Q1/transition_matching_negative_source_plan.csv"
            )
            transform.write_bytes(b"transform")
            alignment.write_bytes(b"alignment")
            negatives.write_bytes(b"negatives")
            alignment_sha = rq1_pair_experiment.sha256_file(alignment)
            negative_sha = rq1_pair_experiment.sha256_file(negatives)
            base_training = {
                "training_protocol_version": (
                    rq1_pair_experiment.TRAINING_PROTOCOL_VERSION_V3
                ),
                "critic_conditioning_mode": "transition_matching",
                "text_transform_path": str(transform),
                "text_alignment_plan_path": str(alignment),
                "matching_negative_source_plan_path": str(negatives),
                "lambda_critic_matching": 0.1,
                "lambda_generator_matching": 0.01,
            }
            fold_payload = {"training": base_training}
            fold_config_path = (
                root / "inputs/folds/2023Q1/train_rq1_pair_textbase.yaml"
            )
            fold_config_path.write_text(
                yaml.safe_dump(fold_payload, sort_keys=False), encoding="utf-8"
            )
            frozen_config_path = (
                root / "inputs/configs/train_rq1_pair_textbase.yaml"
            )
            frozen_config_path.write_text(
                yaml.safe_dump(
                    {
                        "training": base_training,
                        "generate_result": {
                            "evaluation_noise_seed": 20260809,
                            "mc_samples": 64,
                            "reweight_beta_mode": "fixed",
                            "reweight_beta": 0.0,
                            "quantiles": [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
                            "calibration_levels": [0.5, 0.8, 0.9],
                            "arbitrage_violation_tolerance": 1.0e-8,
                            "split": "val",
                            "output_dir": "validation_pilot_json",
                            "selection_mode": "all",
                            "selection_count": 0,
                            "aggregation_mode": "weighted_mean",
                            "residual_blend_alpha": 1.0,
                            "save_json": True,
                            "save_plots": False,
                            "save_full_atm_timeseries": False,
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            def fake_fingerprint(
                _root,
                *,
                fold,
                seed,
                variant,
                training,
                overrides,
                parent_checkpoint_sha256,
            ):
                del training, overrides, parent_checkpoint_sha256
                return rq1_pair_experiment.canonical_payload_sha256(
                    {"fold": fold, "seed": int(seed), "variant": variant}
                )

            comparison_epoch = 20

            def write_run(variant, seed, parent_checkpoint=None):
                output_root = (
                    root
                    / "training_runs"
                    / variant
                    / "2023Q1"
                    / f"seed_{seed}"
                )
                run_dir = output_root / "run_001"
                checkpoints = run_dir / "checkpoints"
                metrics_dir = run_dir / "metrics"
                checkpoints.mkdir(parents=True)
                metrics_dir.mkdir(parents=True)
                overrides = rq1_pair_experiment._variant_overrides(
                    variant,
                    fold_config=fold_payload,
                    output_root=output_root,
                    seed=seed,
                    parent_checkpoint=parent_checkpoint,
                    continuation_anchor_epoch=(
                        comparison_epoch
                        if variant
                        in {
                            rq1_pair_experiment.TEXT_RESIDUAL_VARIANT,
                            rq1_pair_experiment.SHUFFLED_RESIDUAL_VARIANT,
                        }
                        else None
                    ),
                )
                fingerprint = fake_fingerprint(
                    root,
                    fold="2023Q1",
                    seed=seed,
                    variant=variant,
                    training=base_training,
                    overrides=overrides,
                    parent_checkpoint_sha256="",
                )
                overrides["run_fingerprint_sha256"] = fingerprint
                resolved_training = {**base_training, **overrides}
                (metrics_dir / "training_resolved_config.yaml").write_text(
                    yaml.safe_dump(
                        {"training": resolved_training}, sort_keys=False
                    ),
                    encoding="utf-8",
                )
                is_stage_b = variant in rq1_pair_experiment.PAIRED_STAGE_B_VARIANTS
                selected_epoch = (
                    1
                    if variant == rq1_pair_experiment.PARENT_VARIANT
                    else comparison_epoch
                )
                checkpoint_payload = {
                    "checkpoint_schema_version": 6,
                    "training_protocol_version": (
                        rq1_pair_experiment.TRAINING_PROTOCOL_VERSION_V3
                    ),
                    "critic_architecture_version": "transition_matching_v2",
                    "run_fingerprint_sha256": fingerprint,
                    "epoch": selected_epoch,
                    "initial_generator_state_sha256": (
                        f"{seed:064x}" if is_stage_b else ""
                    ),
                    "initial_critic_state_sha256": (
                        f"{seed + 1:064x}" if is_stage_b else ""
                    ),
                    "text_alignment_plan_sha256": alignment_sha,
                    "matching_negative_source_plan_sha256": negative_sha,
                    "scheduler_horizon_epochs": 100 if is_stage_b else 60,
                }
                torch.save(
                    checkpoint_payload, checkpoints / "film_wgan_best.pt"
                )
                torch.save(
                    checkpoint_payload, checkpoints / "film_wgan_final.pt"
                )
                (metrics_dir / "best_checkpoint.json").write_text(
                    json.dumps(
                        {
                            "best_epoch": selected_epoch,
                            "checkpoint_metric": "val_mae",
                        }
                    ),
                    encoding="utf-8",
                )
                pd.DataFrame(
                    {
                        "epoch": np.arange(1, comparison_epoch + 1),
                        "lr_generator": np.linspace(
                            9.9e-5, 8.0e-5, comparison_epoch
                        ),
                        "lr_critic": np.linspace(
                            1.98e-4, 1.6e-4, comparison_epoch
                        ),
                        "val_mae": np.linspace(
                            0.03, 0.01, comparison_epoch
                        ),
                    }
                ).to_csv(metrics_dir / "training_metrics.csv", index=False)
                return run_dir

            for seed in rq1_pair_experiment.V3_PILOT_SEEDS:
                parent_run = write_run(
                    rq1_pair_experiment.PARENT_VARIANT, seed
                )
                parent_checkpoint = (
                    parent_run / "checkpoints/film_wgan_best.pt"
                )
                for variant in rq1_pair_experiment.PAIRED_STAGE_B_VARIANTS:
                    write_run(variant, seed, parent_checkpoint)

            with (
                patch.object(rq1_pair_experiment, "verify_existing"),
                patch.object(
                    rq1_pair_experiment,
                    "_run_fingerprint",
                    side_effect=fake_fingerprint,
                ),
            ):
                rq1_pair_experiment.collect_checkpoints(
                    argparse.Namespace(experiment_root=str(root))
                )
            selected = pd.read_csv(
                root / "checkpoint_selection/selected_checkpoints.csv"
            )
            self.assertEqual(len(selected), 12)
            self.assertEqual(
                set(zip(selected["fold"], selected["seed"], selected["variant"])),
                {
                    (fold, seed, variant)
                    for fold in rq1_pair_experiment.V3_PILOT_FOLDS
                    for seed in rq1_pair_experiment.V3_PILOT_SEEDS
                    for variant in rq1_pair_experiment.V3_PILOT_VARIANTS
                },
            )
            for row in selected.itertuples(index=False):
                expected_name = (
                    "film_wgan_final.pt"
                    if row.variant
                    in {
                        rq1_pair_experiment.TEXT_RESIDUAL_VARIANT,
                        rq1_pair_experiment.SHUFFLED_RESIDUAL_VARIANT,
                    }
                    else "film_wgan_best.pt"
                )
                self.assertEqual(Path(row.checkpoint_path).name, expected_name)
                expected_epoch = (
                    1
                    if row.variant == rq1_pair_experiment.PARENT_VARIANT
                    else comparison_epoch
                )
                self.assertEqual(int(row.comparison_epoch), expected_epoch)

            generated_commands = []
            parsed_sample_configs = []

            def parsed_sample_config(command):
                config_path = Path(command[command.index("--config") + 1])
                set_items = [
                    command[index + 1]
                    for index, value in enumerate(command[:-1])
                    if value == "--set"
                ]
                overrides = parse_sample_overrides(set_items)
                for flag, field in (
                    ("--output-dir", "output_dir"),
                    ("--split", "split"),
                    ("--selection-mode", "selection_mode"),
                ):
                    if flag in command:
                        overrides[field] = command[command.index(flag) + 1]
                if "--selection-count" in command:
                    overrides["selection_count"] = int(
                        command[command.index("--selection-count") + 1]
                    )
                if "--no-plot" in command:
                    overrides["save_plots"] = False
                if "--no-json" in command:
                    overrides["save_json"] = False
                checkpoint = command[command.index("--checkpoint") + 1]
                return load_sample_config(
                    config_path,
                    overrides=overrides,
                    checkpoint_path=checkpoint,
                )

            def fake_generate(command, *, log_path):
                del log_path
                generated_commands.append(command)
                config_path = Path(command[command.index("--config") + 1])
                run_dir = config_path.parents[1]
                sample_config = parsed_sample_config(command)
                parsed_sample_configs.append(sample_config)
                output_dir = run_dir / sample_config.output_dir
                output_dir.mkdir(parents=True)
                pd.DataFrame({"surface_pair_id": ["p0", "p1"]}).to_csv(
                    output_dir / "summary.csv", index=False
                )
                samples_dir = output_dir / "samples"
                samples_dir.mkdir(parents=True, exist_ok=True)
                for pair_id in ("p0", "p1"):
                    (samples_dir / f"{pair_id}.json").write_text(
                        json.dumps(
                            {
                                "surface_pair_id": pair_id,
                                "generated_surface": [[0.21]],
                                "current_surface": [[0.20]],
                                "evaluation_support_mask": [[1]],
                                "effective_beta": float(sample_config.reweight_beta),
                                "metadata": {
                                    "mc_samples": int(sample_config.mc_samples),
                                    "split": sample_config.split,
                                    "aggregation_mode": sample_config.aggregation_mode,
                                    "residual_blend_alpha": float(
                                        sample_config.residual_blend_alpha
                                    ),
                                    "text_alignment_mode": (
                                        sample_config.text_alignment_mode
                                    )
                                },
                            }
                        ),
                        encoding="utf-8",
                    )

            with (
                patch.object(rq1_pair_experiment, "_assert_py312"),
                patch.object(rq1_pair_experiment, "verify_existing"),
                patch.object(
                    rq1_pair_experiment, "_fold_counts", return_value=(4, 2, 3)
                ),
                patch.object(
                    rq1_pair_experiment, "_run", side_effect=fake_generate
                ),
            ):
                rq1_pair_experiment.generate_matrix(
                    argparse.Namespace(experiment_root=str(root))
                )
            self.assertEqual(len(generated_commands), 12)
            generate_registry = pd.read_csv(
                root / "registry/generate_registry.csv"
            )
            self.assertEqual(len(generate_registry), 12)
            self.assertEqual(set(generate_registry["split"]), {"val"})
            self.assertEqual(
                set(generate_registry["output_dir"]), {"validation_pilot_json"}
            )
            self.assertTrue(
                generate_registry["summary_path"]
                .astype(str)
                .str.contains("validation_pilot_json/summary.csv", regex=False)
                .all()
            )
            self.assertEqual(len(parsed_sample_configs), 12)
            for command, sample_config in zip(
                generated_commands, parsed_sample_configs
            ):
                self.assertEqual(
                    Path(command[command.index("--config") + 1]).name,
                    "validation_pilot_generate_config.yaml",
                )
                self.assertEqual(sample_config.reweight_beta_mode, "fixed")
                self.assertEqual(float(sample_config.reweight_beta), 0.0)
                self.assertEqual(int(sample_config.mc_samples), 64)
                self.assertEqual(sample_config.split, "val")
                self.assertEqual(sample_config.aggregation_mode, "weighted_mean")
                self.assertEqual(
                    list(sample_config.quantiles),
                    [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
                )
                self.assertEqual(
                    list(sample_config.calibration_levels), [0.5, 0.8, 0.9]
                )

            with (
                patch.object(
                    rq1_pair_experiment, "_fold_counts", return_value=(4, 2, 3)
                ),
                patch.object(
                    rq1_pair_experiment, "_run", side_effect=fake_generate
                ),
            ):
                text_swap = rq1_pair_experiment._collect_text_swap_sensitivity(
                    root=root,
                    selected=selected,
                    registry=generate_registry,
                )
            self.assertEqual(len(text_swap), 12)
            self.assertEqual(len(generated_commands), 18)
            for command, sample_config in zip(
                generated_commands[-6:], parsed_sample_configs[-6:]
            ):
                self.assertEqual(
                    Path(command[command.index("--config") + 1]).name,
                    "validation_pilot_generate_config.yaml",
                )
                self.assertEqual(float(sample_config.reweight_beta), 0.0)
                self.assertEqual(int(sample_config.mc_samples), 64)
                self.assertEqual(sample_config.split, "val")
                self.assertEqual(sample_config.aggregation_mode, "weighted_mean")
                self.assertIn(
                    sample_config.text_alignment_mode, {"matched", "permuted"}
                )
            with self.assertRaisesRegex(RuntimeError, "Validation-only pilot"):
                rq1_pair_experiment.build_comparison(
                    argparse.Namespace(
                        experiment_root=str(root),
                        bootstrap_iterations=10,
                        bootstrap_seed=123,
                    )
                )
            with (
                patch.object(rq1_pair_experiment, "_assert_py312"),
                self.assertRaisesRegex(RuntimeError, "validation-only pilot"),
            ):
                rq1_pair_experiment.run_results_pipeline(
                    argparse.Namespace(
                        experiment_root=str(root),
                        bootstrap_iterations=10,
                        bootstrap_seed=123,
                    )
                )

            frozen = yaml.safe_load(frozen_config_path.read_text(encoding="utf-8"))
            frozen["generate_result"]["split"] = "test"
            frozen_config_path.write_text(
                yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8"
            )
            with (
                patch.object(rq1_pair_experiment, "_assert_py312"),
                patch.object(rq1_pair_experiment, "verify_existing"),
                self.assertRaisesRegex(RuntimeError, "locked to split=val"),
            ):
                rq1_pair_experiment.generate_matrix(
                    argparse.Namespace(experiment_root=str(root))
                )

    def test_v3_checkpoint_reuse_requires_schema6_and_exact_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            checkpoint = run_dir / "checkpoints/film_wgan_best.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(
                {
                    "checkpoint_schema_version": 6,
                    "training_protocol_version": (
                        rq1_pair_experiment.TRAINING_PROTOCOL_VERSION_V3
                    ),
                    "critic_architecture_version": "transition_matching_v2",
                    "run_fingerprint_sha256": "a" * 64,
                    "epoch": 31,
                },
                checkpoint,
            )
            rq1_pair_experiment._validate_run_protocol(
                run_dir,
                critic_mode="transition_matching",
                require_v3=True,
                expected_fingerprint="a" * 64,
            )
            with self.assertRaisesRegex(ValueError, "run_fingerprint_sha256"):
                rq1_pair_experiment._validate_run_protocol(
                    run_dir,
                    critic_mode="transition_matching",
                    require_v3=True,
                    expected_fingerprint="b" * 64,
                )

    def test_validation_admission_gates_apply_all_thresholds_fail_closed(self):
        diagnostic_rows = []
        for variant in (
            rq1_pair_experiment.TEXT_RESIDUAL_VARIANT,
            rq1_pair_experiment.SHUFFLED_RESIDUAL_VARIANT,
        ):
            for seed in rq1_pair_experiment.V3_PILOT_SEEDS:
                for epoch in range(16, 21):
                    matched = variant == rq1_pair_experiment.TEXT_RESIDUAL_VARIANT
                    diagnostic_rows.append(
                        {
                            "fold": "2023Q1",
                            "seed": seed,
                            "variant": variant,
                            "epoch": epoch,
                            "comparison_epoch": 20,
                            "is_comparison_epoch": int(epoch == 20),
                            "val_matching_real_pairwise_accuracy": 0.65 if matched else 0.50,
                            "val_matching_real_accuracy_ci95_low": 0.53 if matched else 0.44,
                            "val_matching_real_accuracy_ci95_high": 0.75 if matched else 0.56,
                            "val_matching_real_margin_mean": 0.1 if matched else 0.0,
                            "gp_raw_norm_mean": 1.0,
                            "gp_unscaled_penalty": 0.01,
                            "gp_raw_norm_outside_0p5_1p5_rate": 0.02,
                            "gp_unsupported_max_abs_gradient": 0.0,
                            "diag_g_probe_active": 1.0,
                            "diag_g_matching_output_grad_ratio_median": 0.10,
                            "diag_g_matching_output_grad_ratio_p95": 0.30,
                            "g_transition_unclipped_raw_log_max_abs_error": 1.0e-8,
                            "g_transition_unclipped_normalized_max_abs_error": 1.0e-5,
                            "g_transition_clipped_fraction": 1.0e-4,
                        }
                    )
        seed_rows = []
        for contrast in ("matched_vs_continuation", "matched_vs_shuffled"):
            for seed in rq1_pair_experiment.V3_PILOT_SEEDS:
                seed_rows.append(
                    {
                        "contrast": contrast,
                        "metric": "surface_mae",
                        "fold": "2023Q1",
                        "seed": seed,
                        "mean_baseline_minus_focal": 0.01,
                    }
                )
        duplicate = pd.DataFrame(
            [
                {
                    "contrast": contrast,
                    "metric": "surface_mae",
                    "policy": "exclude_any_exact_or_near_duplicate_seen_in_train",
                    "unique_pairs": 100,
                    "seed_mean_baseline_minus_focal": 0.01,
                    "positive_seed_count": 3,
                }
                for contrast in ("matched_vs_continuation", "matched_vs_shuffled")
            ]
        )
        samples = pd.DataFrame(
            [
                {
                    "fold": "2023Q1",
                    "seed": seed,
                    "variant": rq1_pair_experiment.TEXT_RESIDUAL_VARIANT,
                    "surface_mae": 0.10,
                    "current_mae": 0.20,
                }
                for seed in rq1_pair_experiment.V3_PILOT_SEEDS
            ]
        )
        gates = rq1_pair_experiment._validation_admission_gates(
            diagnostics=pd.DataFrame(diagnostic_rows),
            seed_summary=pd.DataFrame(seed_rows),
            duplicate_sensitivity=duplicate,
            samples=samples,
        )
        self.assertTrue(gates["passed"])
        self.assertEqual(
            set(gates["gates"]),
            {
                "matched_heldout_matcher",
                "matched_vs_shuffled_matcher_accuracy",
                "shuffled_matcher_at_chance",
                "support_aware_gradient_penalty",
                "unsupported_gradient_zero",
                "generator_matching_output_gradient_ratio",
                "transition_roundtrip",
                "transition_clipping",
                "matched_surface_mae",
                "duplicate_free_direction",
            },
        )
        missing = pd.DataFrame(diagnostic_rows).drop(
            columns=["gp_unsupported_max_abs_gradient"]
        )
        failed = rq1_pair_experiment._validation_admission_gates(
            diagnostics=missing,
            seed_summary=pd.DataFrame(seed_rows),
            duplicate_sensitivity=duplicate,
            samples=samples,
        )
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["gates"]["unsupported_gradient_zero"]["passed"])
        self.assertIn(
            "missing required fields",
            failed["gates"]["unsupported_gradient_zero"]["reason"],
        )

        partial_nan = pd.DataFrame(diagnostic_rows)
        partial_nan.loc[
            partial_nan.index[0],
            "g_transition_unclipped_raw_log_max_abs_error",
        ] = np.nan
        partial_nan.loc[
            partial_nan.index[1], "g_transition_clipped_fraction"
        ] = np.nan
        failed_nan = rq1_pair_experiment._validation_admission_gates(
            diagnostics=partial_nan,
            seed_summary=pd.DataFrame(seed_rows),
            duplicate_sensitivity=duplicate,
            samples=samples,
        )
        self.assertFalse(failed_nan["passed"])
        self.assertFalse(
            failed_nan["gates"]["transition_roundtrip"]["passed"]
        )
        self.assertFalse(
            failed_nan["gates"]["transition_clipping"]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
