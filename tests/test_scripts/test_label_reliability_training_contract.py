"""Focused contracts for train-only label reliability filtering and weighting."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for value in (str(ROOT), str(SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from wgan_option.config import Config, label_reliability_lineage  # noqa: E402
from wgan_option.models.gan_model import WGAN_GP  # noqa: E402
from wgan_option.train_vol_regression_xlsx import (  # noqa: E402
    VolSurfaceRegressionTrainer,
)
from wgan_option.utils.inference_helpers import (  # noqa: E402
    resolve_checkpoint_label_reliability_contract,
)
from wgan_option.utils.merged_xlsx_types import (  # noqa: E402
    VolSurfaceSample,
    VolSurfaceXlsxBundle,
)
from wgan_option.utils.news_first_dataloaders import (  # noqa: E402
    NewsFirstVolSplitSelection,
    _apply_label_reliability_contract,
    _weighted_loader,
    create_news_first_vol_surface_dataloaders,
    label_reliability_profile_sha256,
    label_reliability_train_pair_universe_sha256,
)
from wgan_option.utils.weighted_training import (  # noqa: E402
    unpack_vol_training_batch,
    validated_label_reliability_weights,
)


def _sample(
    sample_id: str,
    pair_id: str,
    session_id: str,
    origin: str,
    *,
    target_value: float = 1.0,
) -> VolSurfaceSample:
    grid = np.asarray([0.9, 1.1], dtype=np.float32)
    maturities = np.asarray([10.0, 20.0], dtype=np.float32)
    mask = np.ones((1, 2, 2), dtype=np.float32)
    return VolSurfaceSample(
        sample_id=sample_id,
        timestamp=origin,
        current_snapshot_time_utc=origin,
        target_snapshot_time_utc=origin,
        current_surface=np.zeros((1, 2, 2), dtype=np.float32),
        target_surface=np.full((1, 2, 2), target_value, dtype=np.float32),
        text_embedding=np.asarray([1.0, 2.0], dtype=np.float32),
        strike_grid=grid,
        maturity_grid_days=maturities,
        surface_shape=(2, 2),
        global_index=0,
        metadata={},
        pair_id=pair_id,
        session_id=session_id,
        effective_origin_utc=origin,
        stable_sample_key=sample_id,
        support_mask=mask,
        current_support_mask=mask.copy(),
        support_grid_fingerprint="g" * 64,
        support_mask_fingerprint="j" * 64,
        current_support_mask_fingerprint="c" * 64,
    )


def _manifest_frame(
    *,
    included: tuple[bool, bool, bool] = (True, True, False),
    weights: tuple[float, float, float] = (0.5, 1.5, 0.0),
) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "tolerance_minutes": 5,
                "fold_id": "train_bootstrap_v1",
                "pair_id": pair_id,
                "included": include,
                "normalized_label_weight": weight,
                "reliability_score": score,
            }
            for pair_id, include, weight, score in zip(
                ("p1", "p2", "p3"),
                included,
                weights,
                (0.9, 0.8, 0.1),
                strict=True,
            )
        ]
    )
    frame["train_pair_universe_sha256"] = (
        label_reliability_train_pair_universe_sha256(
            frame["pair_id"].tolist(),
            fold_id="train_bootstrap_v1",
            tolerance_minutes=5,
        )
    )
    return frame


def _manifest_config(
    path: Path,
    frame: pd.DataFrame,
    *,
    mode: str,
) -> Config:
    manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return Config(
        cuda=False,
        support_mask_mode="raw_joint",
        news_first_dataset_tolerance_minutes=5,
        news_first_label_reliability_mode=mode,
        news_first_label_reliability_manifest_path=str(path),
        news_first_label_reliability_manifest_sha256=manifest_sha,
        news_first_label_reliability_profile_sha256=(
            label_reliability_profile_sha256(frame)
        ),
        news_first_label_reliability_fold_id="train_bootstrap_v1",
        news_first_label_reliability_train_pair_universe_sha256=(
            str(frame["train_pair_universe_sha256"].iloc[0])
        ),
        news_first_materialize_test_loader=False,
        batch_size=16,
        num_workers=0,
    )


def _selection() -> NewsFirstVolSplitSelection:
    return NewsFirstVolSplitSelection(
        train_items=[
            _sample("p1-a", "p1", "train-s1", "2023-06-01T12:00:00Z"),
            _sample("p1-b", "p1", "train-s1", "2023-06-01T12:01:00Z"),
            _sample("p2-a", "p2", "train-s2", "2023-06-02T12:00:00Z"),
            _sample("p3-a", "p3", "train-s3", "2023-06-03T12:00:00Z"),
        ],
        val_items=[
            _sample("val", "pv", "val-s", "2023-08-01T12:00:00Z"),
        ],
        test_items=[
            _sample("test", "pt", "test-s", "2023-11-01T12:00:00Z"),
        ],
        train_end_utc="2023-07-01T00:00:00+00:00",
        validation_end_utc="2023-10-01T00:00:00+00:00",
    )


def _support_diagnostics() -> dict[str, object]:
    split = {
        "input_rows": 1,
        "input_pairs": 1,
        "input_sessions": 1,
        "excluded_zero_joint_support_rows": 0,
        "excluded_zero_joint_support_pairs": 0,
        "excluded_zero_joint_support_sessions": 0,
        "kept_rows": 1,
        "kept_pairs": 1,
        "kept_sessions": 1,
        "joint_support_cell_count": {"min": 4},
    }
    return {
        "support_mask_mode": "raw_joint",
        "support_mask_applied": True,
        "time_partitions": {
            "train": dict(split),
            "validation": dict(split),
            "test": dict(split),
        },
    }


class _ZeroRegressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, current_surface, text_embedding):
        del text_embedding
        return torch.zeros_like(current_surface) + self.bias


class _TextCritic(torch.nn.Module):
    def forward(self, next_surface, current_surface, text_embedding):
        del current_surface
        return text_embedding[:, :1] + next_surface.reshape(
            next_surface.shape[0], -1
        ).mean(dim=1, keepdim=True) * 0.0


class TestLabelReliabilityTrainingContract(unittest.TestCase):
    def test_config_defaults_and_old_checkpoint_config_resolve_to_none(self):
        config = Config(cuda=False)
        self.assertEqual(config.news_first_label_reliability_mode, "none")
        self.assertEqual(config.news_first_label_reliability_manifest_path, "")
        self.assertTrue(config.news_first_materialize_test_loader)

        old_payload = asdict(config)
        for field in tuple(old_payload):
            if field.startswith("news_first_label_reliability"):
                old_payload.pop(field)
        old_payload.pop("news_first_materialize_test_loader")
        restored = Config(**old_payload)
        self.assertEqual(
            label_reliability_lineage(restored),
            {
                "news_first_label_reliability_mode": "none",
                "news_first_label_reliability_manifest_path": "",
                "news_first_label_reliability_manifest_sha256": "",
                "news_first_label_reliability_profile_sha256": "",
                "news_first_label_reliability_fold_id": "",
                "news_first_label_reliability_train_pair_universe_sha256": "",
                "news_first_materialize_test_loader": True,
            },
        )
        self.assertEqual(
            resolve_checkpoint_label_reliability_contract({"config": {}})[
                "news_first_label_reliability_mode"
            ],
            "none",
        )
        with self.assertRaisesRegex(ValueError, "complete manifest lineage"):
            Config(news_first_label_reliability_mode="soft_weight")
        with self.assertRaisesRegex(ValueError, "support_mask_mode='raw_joint'"):
            Config(
                news_first_dataset_tolerance_minutes=5,
                news_first_label_reliability_manifest_path="manifest.csv",
                news_first_label_reliability_manifest_sha256="a" * 64,
                news_first_label_reliability_profile_sha256="b" * 64,
                news_first_label_reliability_fold_id="fold",
                news_first_label_reliability_train_pair_universe_sha256="c" * 64,
                news_first_materialize_test_loader=False,
            )

    def test_eighth_batch_item_preserves_historical_three_through_seven(self):
        current = torch.zeros(2, 1, 2, 2)
        text = torch.zeros(2, 3)
        target = torch.ones(2, 1, 2, 2)
        pair_weight = torch.tensor([0.75, 1.25])
        stable_key = torch.tensor([1, 2])
        joint = torch.ones_like(current)
        current_mask = torch.ones_like(current)
        label_weight = torch.tensor([0.5, 1.5])
        values = (
            current,
            text,
            target,
            pair_weight,
            stable_key,
            joint,
            current_mask,
            label_weight,
        )
        for length in range(3, 8):
            unpacked = unpack_vol_training_batch(values[:length])
            self.assertIsNone(unpacked.label_reliability_weight)
            if length == 7:
                torch.testing.assert_close(
                    unpacked.current_support_mask,
                    current_mask,
                )
        unpacked = unpack_vol_training_batch(values)
        torch.testing.assert_close(unpacked.label_reliability_weight, label_weight)
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            validated_label_reliability_weights(
                label_weight,
                batch_size=2,
                device=torch.device("cpu"),
                dtype=torch.float32,
                require_ones=True,
            )

    def test_profile_hash_uses_canonical_six_column_records(self):
        frame = _manifest_frame().iloc[[2, 0, 1]].reset_index(drop=True)
        records = (
            _manifest_frame()
            .sort_values("pair_id")
            .loc[
                :,
                [
                    "tolerance_minutes",
                    "fold_id",
                    "pair_id",
                    "included",
                    "normalized_label_weight",
                    "reliability_score",
                ],
            ]
            .to_dict("records")
        )
        expected = hashlib.sha256(
            json.dumps(
                records,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(label_reliability_profile_sha256(frame), expected)

    def test_support_filter_soft_weight_filters_whole_pairs_and_rebalances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.csv"
            frame = _manifest_frame()
            frame.to_csv(path, index=False)
            config = _manifest_config(
                path,
                frame,
                mode="support_filter_soft_weight",
            )
            selected, audit = _apply_label_reliability_contract(
                _selection(),
                config,
            )

        self.assertEqual([item.pair_id for item in selected.train_items], ["p1", "p1", "p2"])
        self.assertEqual(
            [item.label_reliability_weight for item in selected.train_items],
            [0.5, 0.5, 1.5],
        )
        self.assertEqual(audit["filtered_train_pairs"], 1)
        self.assertAlmostEqual(audit["effective_pair_weight_mean"], 1.0)
        scaled_pair_weights = [
            item.metadata["scaled_training_sample_weight"]
            for item in selected.train_items
        ]
        self.assertEqual(scaled_pair_weights, [0.75, 0.75, 1.5])
        self.assertAlmostEqual(float(np.mean(scaled_pair_weights)), 1.0)
        self.assertAlmostEqual(
            float(
                np.mean(
                    [
                        base * item.label_reliability_weight
                        for base, item in zip(
                            scaled_pair_weights,
                            selected.train_items,
                            strict=True,
                        )
                    ]
                )
            ),
            1.0,
        )
        self.assertEqual(selected.val_items[0].label_reliability_weight, 1.0)
        self.assertEqual(selected.test_items[0].label_reliability_weight, 1.0)

    def test_arm_a_validates_lineage_but_does_not_filter_or_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.csv"
            frame = _manifest_frame()
            frame.to_csv(path, index=False)
            config = _manifest_config(path, frame, mode="none")
            selected, audit = _apply_label_reliability_contract(
                _selection(),
                config,
            )
        self.assertEqual(len({item.pair_id for item in selected.train_items}), 3)
        self.assertEqual(
            {item.label_reliability_weight for item in selected.train_items},
            {1.0},
        )
        self.assertTrue(audit["lineage_validated"])
        self.assertEqual(audit["filtered_train_pairs"], 0)

    def test_manifest_drift_and_invalid_soft_normalization_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.csv"
            frame = _manifest_frame(
                included=(True, True, True),
                weights=(0.5, 1.0, 1.0),
            )
            frame.to_csv(path, index=False)
            config = _manifest_config(path, frame, mode="soft_weight")
            with self.assertRaisesRegex(ValueError, "pair-level mean 1"):
                _apply_label_reliability_contract(_selection(), config)

            valid = _manifest_frame()
            wrong_universe = valid.copy()
            wrong_universe["train_pair_universe_sha256"] = "0" * 64
            wrong_universe.to_csv(path, index=False)
            config = _manifest_config(path, wrong_universe, mode="none")
            with self.assertRaisesRegex(ValueError, "universe SHA256 mismatch"):
                _apply_label_reliability_contract(_selection(), config)

            valid.to_csv(path, index=False)
            config = _manifest_config(path, valid, mode="none")
            path.write_text(path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest SHA256 mismatch"):
                _apply_label_reliability_contract(_selection(), config)

    def test_formal_loader_emits_eighth_weight_and_never_materializes_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.csv"
            frame = _manifest_frame()
            frame.to_csv(path, index=False)
            config = _manifest_config(
                path,
                frame,
                mode="support_filter_soft_weight",
            )
            config.data_path = "train.xlsx"
            training = _selection().train_items
            common = [*_selection().val_items, *_selection().test_items]
            diagnostics = _support_diagnostics()
            with (
                patch(
                    "wgan_option.utils.news_first_dataloaders._load_surface_items",
                    side_effect=[
                        (training, diagnostics),
                        (common, diagnostics),
                    ],
                ),
                patch(
                    "wgan_option.utils.news_first_dataloaders._weighted_loader",
                    wraps=_weighted_loader,
                ) as weighted_loader,
            ):
                bundle = create_news_first_vol_surface_dataloaders(
                    config,
                    "common-5m.xlsx",
                )

        train_batch = next(iter(bundle.train_loader))
        val_batch = next(iter(bundle.val_loader))
        self.assertEqual(len(train_batch), 8)
        self.assertEqual(len(val_batch), 8)
        self.assertEqual(sorted(train_batch[7].tolist()), [0.5, 0.5, 1.5])
        torch.testing.assert_close(val_batch[7], torch.ones_like(val_batch[7]))
        self.assertEqual(weighted_loader.call_count, 2)
        self.assertIsNone(bundle.test_loader)
        self.assertEqual(bundle.test_items, [])
        self.assertEqual(bundle.test_samples, 0)
        self.assertEqual(bundle.split_metadata["test_rows"], 1)
        self.assertEqual(bundle.split_metadata["materialized_test_rows"], 0)

    def test_regression_and_wgan_apply_label_weight_only_to_reconstruction(self):
        current = torch.zeros(2, 1, 2, 2)
        target = torch.stack(
            [torch.ones(1, 2, 2), torch.full((1, 2, 2), 3.0)],
        )
        text = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
        pair_weight = torch.ones(2)
        key = torch.tensor([1, 2])
        mask = torch.ones_like(current)
        label_weight = torch.tensor([0.5, 1.5])
        loader = DataLoader(
            TensorDataset(
                current,
                text,
                target,
                pair_weight,
                key,
                mask,
                mask,
                label_weight,
            ),
            batch_size=2,
            shuffle=False,
        )
        config = Config(
            cuda=False,
            lambda_recon=1.0,
            lambda_calendar=1.0,
            lambda_butterfly=0.0,
            lambda_smooth=0.0,
            lambda_delta_shrink=0.0,
            use_calendar_constraint=True,
            use_butterfly_constraint=False,
            use_smooth_constraint=False,
            learning_rate=0.0,
            reduce_lr_min_lr=0.0,
            noise_dim=2,
            gen_base_channels=1,
            disc_base_channels=1,
            gen_text_hidden_dim=3,
            gen_text_out_dim=2,
            disc_text_hidden_dim=2,
            gen_hidden_dim=4,
            disc_hidden_dim=4,
        )
        trainer = VolSurfaceRegressionTrainer(config)
        trainer.model = _ZeroRegressor()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
        trainer.strike_grid = torch.tensor([0.9, 1.1])
        trainer.tau_years = torch.tensor([10.0, 20.0]) / 365.0
        with patch.object(
            trainer,
            "_calendar_penalty_per_sample",
            return_value=torch.tensor([1.0, 3.0]),
        ):
            train_metrics = trainer._run_epoch(loader, train=True, epoch=1)
        self.assertAlmostEqual(train_metrics["train_recon"], 2.5, places=6)
        self.assertAlmostEqual(train_metrics["train_calendar"], 2.0, places=6)
        self.assertAlmostEqual(train_metrics["train_total"], 4.5, places=6)

        model = WGAN_GP(
            config,
            strike_grid=np.asarray([0.9, 1.1], dtype=np.float32),
            maturity_grid_days=np.asarray([10.0, 20.0], dtype=np.float32),
            embedding_dim=2,
        )
        model.G = _ZeroRegressor()
        model.D = _TextCritic()
        model.g_optimizer = torch.optim.SGD(model.G.parameters(), lr=0.0)
        with patch.object(
            model,
            "_calendar_penalty_per_sample",
            return_value=torch.tensor([1.0, 3.0]),
        ):
            stats = model._generator_step(
                current,
                text,
                target,
                sample_weight=pair_weight,
                label_reliability_weight=label_weight,
                support_mask=mask,
            )
        self.assertAlmostEqual(stats["g_recon"], 2.5, places=6)
        self.assertAlmostEqual(stats["g_adv"], -2.0, places=6)
        self.assertAlmostEqual(stats["g_calendar"], 2.0, places=6)
        self.assertAlmostEqual(stats["g_total"], 2.5, places=6)

        invalid_val_loader = DataLoader(
            TensorDataset(
                current,
                text,
                target,
                pair_weight,
                key,
                mask,
                mask,
                label_weight,
            ),
            batch_size=2,
        )
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            trainer._run_epoch(invalid_val_loader, train=False, epoch=0)

    def test_regression_and_wgan_checkpoints_persist_exact_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.csv"
            frame = _manifest_frame()
            frame.to_csv(manifest_path, index=False)
            config = _manifest_config(
                manifest_path,
                frame,
                mode="support_filter_soft_weight",
            )
            config.models_path = str(root / "models")
            config.metrics_path = str(root / "metrics")
            loader = DataLoader(
                TensorDataset(
                    torch.zeros(1, 1, 2, 2),
                    torch.zeros(1, 2),
                    torch.ones(1, 1, 2, 2),
                ),
                batch_size=1,
            )
            bundle = VolSurfaceXlsxBundle(
                train_loader=loader,
                val_loader=loader,
                strike_grid=np.asarray([0.9, 1.1], dtype=np.float32),
                maturity_grid_days=np.asarray([10.0, 20.0], dtype=np.float32),
                embedding_dim=2,
                train_samples=1,
                val_samples=1,
                timestamps=[],
                train_timestamps=[],
                val_timestamps=[],
                all_items=[],
                train_items=[],
                val_items=[],
            )
            trainer = VolSurfaceRegressionTrainer(config)
            trainer.bundle = bundle
            trainer.model = _ZeroRegressor()
            trainer.model.residual_output_mode = "identity_residual"  # type: ignore[attr-defined]
            trainer.model.residual_output_fingerprint = "test"  # type: ignore[attr-defined]
            regression_path = trainer.save_model()["model"]

            wgan = WGAN_GP(
                config,
                strike_grid=np.linspace(0.9, 1.1, 8, dtype=np.float32),
                maturity_grid_days=np.linspace(10.0, 20.0, 8, dtype=np.float32),
                embedding_dim=2,
            )
            wgan_path = wgan.save_model()["generator"]

            expected = label_reliability_lineage(config)
            for checkpoint_path in (regression_path, wgan_path):
                payload = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                for key, value in expected.items():
                    self.assertEqual(payload[key], value)
                for key, value in expected.items():
                    self.assertEqual(payload["config"][key], value)
                self.assertEqual(
                    resolve_checkpoint_label_reliability_contract(payload),
                    expected,
                )
                payload["news_first_label_reliability_manifest_sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    resolve_checkpoint_label_reliability_contract(payload)


if __name__ == "__main__":
    unittest.main()
