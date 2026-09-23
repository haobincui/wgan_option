"""Pair-level text and exact continuation contracts for unified RQ1--RQ3."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wgan_option.config import Config  # noqa: E402
from wgan_option.utils.merged_xlsx_types import VolSurfaceSample  # noqa: E402
from wgan_option.utils.news_first_experiment_core import (  # noqa: E402
    RESUME_DYNAMIC_FULL_TRAINING_STATE,
    RESUME_FROZEN_LR_FULL_TRAINING_STATE,
    SAVE_DYNAMIC_FULL_TRAINING_STATE,
    apply_pair_text_overlay,
    build_pair_text_overlay_manifests,
    load_full_training_state,
    load_full_training_state_contract,
    load_pair_text_overlay_manifest,
    pair_universe_sha256,
    save_full_training_state,
    sha256_file,
    training_config_payload_sha256,
    write_full_training_state_contract,
    write_pair_text_overlay_manifest,
)


def _sample(
    pair_id: str,
    session_id: str,
    news_id: int,
    *,
    value: float,
) -> VolSurfaceSample:
    current = np.full((1, 2, 2), value, dtype=np.float32)
    support = np.ones_like(current)
    return VolSurfaceSample(
        sample_id=f"row-{news_id}",
        timestamp="2023-01-03T14:00:00Z",
        current_snapshot_time_utc="2023-01-03T14:00:00Z",
        target_snapshot_time_utc="2023-01-03T14:05:00Z",
        current_surface=current,
        target_surface=current + 0.01,
        text_embedding=np.full(1024, news_id / 100.0, dtype=np.float32),
        strike_grid=np.asarray([0.97, 1.03], dtype=np.float32),
        maturity_grid_days=np.asarray([1, 2], dtype=np.float32),
        surface_shape=(2, 2),
        global_index=news_id,
        metadata={"article": f"article-{news_id}"},
        news_row_id=news_id,
        pair_id=pair_id,
        session_id=session_id,
        effective_origin_utc="2023-01-03T14:00:00Z",
        stable_sample_key=f"stable-{news_id}",
        support_mask=support,
        current_support_mask=support,
        support_grid_fingerprint="grid",
        support_mask_fingerprint=f"support-{pair_id}",
        current_support_mask_fingerprint=f"current-support-{pair_id}",
    )


def _lineage(*, arm: str = "parent") -> dict[str, object]:
    digest = hashlib.sha256(b"lineage").hexdigest()
    return {
        "fold_id": "f1_2023q1",
        "seed": 42,
        "arm": arm,
        "model_contract_sha256": digest,
        "grid_sha256": digest,
        "training_config_payload_sha256": digest,
        "code_sha256": digest,
        "dataset_sha256": digest,
        "support_sha256": digest,
        "pair_universe_sha256": digest,
        "text_manifest_sha256": digest,
        "job_sha256": digest,
    }


class TestPairTextOverlay(unittest.TestCase):
    def test_pair_rows_collapse_and_manifest_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lp.json"
            first = np.zeros(1024, dtype=np.float32)
            first[0] = 1.0
            second = np.zeros(1024, dtype=np.float32)
            second[1] = 1.0
            written = write_pair_text_overlay_manifest(
                path,
                mode="lp_mean_l2",
                namespace="f1_train_05m",
                records=[
                    {
                        "pair_id": "pair-a",
                        "session_id": "session-a",
                        "embedding": first,
                    },
                    {
                        "pair_id": "pair-b",
                        "session_id": "session-b",
                        "embedding": second,
                    },
                ],
            )
            manifest = load_pair_text_overlay_manifest(
                path,
                written["manifest_sha256"],
                written["profile_sha256"],
                expected_mode="lp_mean_l2",
            )
            items = [
                _sample("pair-a", "session-a", 2, value=0.2),
                _sample("pair-a", "session-a", 1, value=0.2),
                _sample("pair-b", "session-b", 3, value=0.3),
            ]
            collapsed, audit = apply_pair_text_overlay(items, manifest)
            self.assertEqual([item.pair_id for item in collapsed], ["pair-a", "pair-b"])
            self.assertEqual(collapsed[0].sample_id, "pair::pair-a")
            self.assertEqual(collapsed[0].metadata["source_news_row_count"], 2)
            self.assertEqual(collapsed[0].metadata["source_news_row_ids"], [1, 2])
            np.testing.assert_array_equal(collapsed[0].text_embedding, first)
            self.assertEqual(audit["pair_count"], 2)
            self.assertEqual(
                audit["pair_universe_sha256"],
                pair_universe_sha256(["pair-a", "pair-b"]),
            )

            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_pair_text_overlay_manifest(
                    path,
                    written["manifest_sha256"],
                    written["profile_sha256"],
                    expected_mode="lp_mean_l2",
                )

    def test_current_only_and_shuffle_rules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current = write_pair_text_overlay_manifest(
                root / "current.json",
                mode="current_only",
                namespace="train",
                records=[
                    {"pair_id": "a", "session_id": "s1"},
                    {"pair_id": "b", "session_id": "s2"},
                ],
            )
            loaded = load_pair_text_overlay_manifest(
                root / "current.json",
                current["manifest_sha256"],
                current["profile_sha256"],
                expected_mode="current_only",
            )
            self.assertEqual(int(np.count_nonzero(loaded.embeddings["a"])), 0)

            vector_a = np.zeros(1024, dtype=np.float32)
            vector_a[0] = 1.0
            vector_b = np.zeros(1024, dtype=np.float32)
            vector_b[1] = 1.0
            with self.assertRaisesRegex(ValueError, "fixed-point-free"):
                write_pair_text_overlay_manifest(
                    root / "bad-shuffle.json",
                    mode="lp_shuffle",
                    namespace="train",
                    records=[
                        {
                            "pair_id": "a",
                            "session_id": "s1",
                            "embedding": vector_a,
                            "donor_pair_id": "a",
                        },
                        {
                            "pair_id": "b",
                            "session_id": "s2",
                            "embedding": vector_b,
                            "donor_pair_id": "b",
                        },
                    ],
                )

    def test_legacy_config_defaults_and_new_lineage_is_strict(self):
        self.assertEqual(Config(cuda=False).news_first_pair_text_overlay_mode, "none")
        with self.assertRaisesRegex(ValueError, "complete manifest lineage"):
            Config(
                cuda=False,
                embedding_dim=1024,
                news_first_pair_text_overlay_mode="lp_mean_l2",
            )

    def test_prepare_ignores_malformed_test_only_text_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "data"
            data_root.mkdir()
            train_pairs = ["train-a", "train-b"]
            validation_pairs = ["validation-a", "validation-b"]
            test_pairs = ["test-only"]
            all_pairs = [*train_pairs, *validation_pairs, *test_pairs]

            lp_vector = json.dumps([1.0, *([0.0] * 1023)])
            workbook_rows = []
            sentiment_rows = []
            for news_row_id, pair_id in enumerate(all_pairs, start=1):
                test_only = pair_id in test_pairs
                workbook_rows.append(
                    {
                        "pair_id": pair_id,
                        "session_id": f"session-{pair_id}",
                        "article_id": f"article-{pair_id}",
                        "sample_id": f"sample-{pair_id}",
                        "news_row_id": news_row_id,
                        "lp_text": (
                            np.nan if pair_id == "train-a" else f"text for {pair_id}"
                        ),
                        "lp_embedding": "malformed-test-lp" if test_only else lp_vector,
                    }
                )
                sentiment_rows.append(
                    {
                        "news_row_id": news_row_id,
                        "sentiment_embedding": (
                            "malformed-test-sentiment" if test_only else lp_vector
                        ),
                    }
                )
            duplicate_lp_vector = json.dumps([0.5, 0.5, *([0.0] * 1022)])
            duplicate_news_row_id = len(all_pairs) + 1
            workbook_rows.append(
                {
                    "pair_id": "train-b",
                    "session_id": "session-train-b",
                    "article_id": "article-train-b",
                    "sample_id": "sample-train-b-duplicate",
                    "news_row_id": duplicate_news_row_id,
                    "lp_text": "text for train-b",
                    "lp_embedding": duplicate_lp_vector,
                }
            )
            sentiment_rows.append(
                {
                    "news_row_id": duplicate_news_row_id,
                    "sentiment_embedding": lp_vector,
                }
            )
            workbook_path = data_root / "tolerance_05m.xlsx"
            pd.DataFrame(workbook_rows).to_excel(
                workbook_path, sheet_name="gan_input_ready", index=False
            )
            sentiment_path = root / "sentiment.xlsx"
            pd.DataFrame(sentiment_rows).to_excel(
                sentiment_path, sheet_name="features", index=False
            )

            universe_rows = []
            for partition, pair_ids in (
                ("train", train_pairs),
                ("validation", validation_pairs),
                ("test", test_pairs),
            ):
                universe_sha = pair_universe_sha256(pair_ids)
                universe_rows.extend(
                    {
                        "tolerance_minutes": 5,
                        "fold": "f1",
                        "partition": partition,
                        "pair_id": pair_id,
                        "session_id": f"session-{pair_id}",
                        "pair_universe_sha256": universe_sha,
                    }
                    for pair_id in pair_ids
                )
            universe_path = root / "pair_universes.csv"
            pd.DataFrame(universe_rows).to_csv(universe_path, index=False)
            generated = build_pair_text_overlay_manifests(
                config={
                    "data": {
                        "root": str(data_root),
                        "workbook_template": "tolerance_{tolerance02}m.xlsx",
                        "sheet_name": "gan_input_ready",
                        "sentiment_workbook_path": str(sentiment_path),
                    },
                    "matrix": {"tolerances_minutes": [5], "shuffle_seed": 17},
                    "folds": [{"id": "f1"}],
                },
                pair_universe_path=universe_path,
                output_dir=root / "overlays",
            )
            self.assertEqual(len(generated), 6)
            expected_pairs = set(train_pairs + validation_pairs)
            for path in generated:
                payload = json.loads(path.read_text(encoding="utf-8"))
                observed_pairs = {row["pair_id"] for row in payload["records"]}
                self.assertEqual(observed_pairs, expected_pairs)
                self.assertNotIn("test-only", observed_pairs)
                self.assertEqual(
                    payload["pair_universe_sha256"],
                    pair_universe_sha256(sorted(expected_pairs)),
                )
                if payload["mode"] == "bow1024":
                    transform = payload["transform"]
                    self.assertNotIn("nan", transform["vocabulary"])
                    self.assertEqual(transform["missing_text_article_count_train"], 1)
                    self.assertEqual(transform["missing_text_pair_count_train"], 1)
                    record = next(
                        row for row in payload["records"] if row["pair_id"] == "train-a"
                    )
                    self.assertEqual(int(np.count_nonzero(record["embedding"])), 0)
                transform = payload["transform"]
                self.assertEqual(
                    transform["article_deduplication_selection_rule"],
                    "minimum_numeric_news_row_id_then_sample_id_v1",
                )
                self.assertEqual(transform["duplicate_article_row_count_train"], 1)
                self.assertEqual(
                    transform["duplicate_article_row_count_train_validation"], 1
                )
                self.assertEqual(
                    transform["duplicate_lp_embedding_conflict_count_train"], 1
                )
                self.assertEqual(
                    transform["duplicate_lp_embedding_conflict_count_train_validation"],
                    1,
                )
                self.assertGreater(
                    transform[
                        "duplicate_lp_embedding_conflict_max_abs_train_validation"
                    ],
                    0.0,
                )
                self.assertEqual(
                    transform["duplicate_sentiment_conflict_count_train_validation"],
                    0,
                )


class TestFullTrainingState(unittest.TestCase):
    @staticmethod
    def _components(device: torch.device | str = "cpu"):
        generator = torch.nn.Linear(3, 2).to(device)
        discriminator = torch.nn.Linear(2, 1).to(device)
        g_optimizer = Adam(generator.parameters(), lr=5e-4)
        d_optimizer = Adam(discriminator.parameters(), lr=4e-4)
        g_scheduler = ReduceLROnPlateau(
            g_optimizer, mode="min", factor=0.5, patience=0, min_lr=5e-5
        )
        d_scheduler = ReduceLROnPlateau(
            d_optimizer, mode="min", factor=0.5, patience=0, min_lr=4e-5
        )
        return (
            generator,
            discriminator,
            g_optimizer,
            d_optimizer,
            g_scheduler,
            d_scheduler,
        )

    @staticmethod
    def _optimizer_step(model, optimizer, input_tensor):
        optimizer.zero_grad(set_to_none=True)
        model(input_tensor).sum().backward()
        optimizer.step()

    def _assert_nested_equal(self, first, second):
        self.assertEqual(type(first), type(second))
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self._assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self._assert_nested_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_atomic_full_state_round_trip_restores_training_and_rng(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "parent.pt"
            components = self._components()
            generator, discriminator, g_opt, d_opt, g_sched, d_sched = components
            self._optimizer_step(generator, g_opt, torch.ones(2, 3))
            self._optimizer_step(discriminator, d_opt, torch.ones(2, 2))
            g_sched.step(1.0)
            d_sched.step(1.0)
            loader_generator = torch.Generator().manual_seed(777)
            random.seed(10)
            np.random.seed(11)
            torch.manual_seed(12)
            digest = save_full_training_state(
                path,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=g_opt,
                discriminator_optimizer=d_opt,
                generator_scheduler=g_sched,
                discriminator_scheduler=d_sched,
                loader_generator=loader_generator,
                completed_epoch=7,
                lineage=_lineage(),
                contract_sha256=hashlib.sha256(b"contract").hexdigest(),
            )
            saved_generator = {
                key: value.detach().clone()
                for key, value in generator.state_dict().items()
            }
            expected_rng = (
                random.random(),
                float(np.random.random()),
                torch.rand(3),
                torch.randperm(8, generator=loader_generator),
            )
            self._optimizer_step(generator, g_opt, torch.full((2, 3), 9.0))
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            loader_generator.manual_seed(999)

            restored = load_full_training_state(
                path,
                digest,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=g_opt,
                discriminator_optimizer=d_opt,
                generator_scheduler=g_sched,
                discriminator_scheduler=d_sched,
                loader_generator=loader_generator,
                expected_lineage=_lineage(),
                restore_schedulers=True,
                map_location="cpu",
            )
            self.assertEqual(restored["completed_epoch"], 7)
            for key, value in saved_generator.items():
                torch.testing.assert_close(generator.state_dict()[key], value)
            actual_rng = (
                random.random(),
                float(np.random.random()),
                torch.rand(3),
                torch.randperm(8, generator=loader_generator),
            )
            self.assertEqual(actual_rng[0], expected_rng[0])
            self.assertEqual(actual_rng[1], expected_rng[1])
            torch.testing.assert_close(actual_rng[2], expected_rng[2])
            torch.testing.assert_close(actual_rng[3], expected_rng[3])

    def test_restored_next_step_matches_uninterrupted_control(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "parent.pt"
            torch.manual_seed(101)
            control = self._components()
            (
                control_g,
                control_d,
                control_g_opt,
                control_d_opt,
                control_g_sched,
                control_d_sched,
            ) = control
            self._optimizer_step(control_g, control_g_opt, torch.ones(2, 3))
            self._optimizer_step(control_d, control_d_opt, torch.ones(2, 2))
            control_g_sched.step(1.0)
            control_d_sched.step(1.0)
            loader_generator = torch.Generator().manual_seed(777)
            digest = save_full_training_state(
                path,
                generator=control_g,
                discriminator=control_d,
                generator_optimizer=control_g_opt,
                discriminator_optimizer=control_d_opt,
                generator_scheduler=control_g_sched,
                discriminator_scheduler=control_d_sched,
                loader_generator=loader_generator,
                completed_epoch=1,
                lineage=_lineage(),
                contract_sha256=hashlib.sha256(b"contract").hexdigest(),
            )
            saved_g_scheduler = copy.deepcopy(control_g_sched.state_dict())
            saved_d_scheduler = copy.deepcopy(control_d_sched.state_dict())

            next_g_input = torch.full((2, 3), 0.25)
            next_d_input = torch.full((2, 2), -0.5)
            self._optimizer_step(control_g, control_g_opt, next_g_input)
            self._optimizer_step(control_d, control_d_opt, next_d_input)
            control_g_sched.step(2.0)
            control_d_sched.step(2.0)

            torch.manual_seed(999)
            restored = self._components()
            (
                restored_g,
                restored_d,
                restored_g_opt,
                restored_d_opt,
                restored_g_sched,
                restored_d_sched,
            ) = restored
            load_full_training_state(
                path,
                digest,
                generator=restored_g,
                discriminator=restored_d,
                generator_optimizer=restored_g_opt,
                discriminator_optimizer=restored_d_opt,
                generator_scheduler=restored_g_sched,
                discriminator_scheduler=restored_d_sched,
                loader_generator=torch.Generator(),
                expected_lineage=_lineage(),
                restore_schedulers=True,
                map_location="cpu",
            )
            self._assert_nested_equal(restored_g_sched.state_dict(), saved_g_scheduler)
            self._assert_nested_equal(restored_d_sched.state_dict(), saved_d_scheduler)
            self.assertEqual(
                restored_g_sched.last_epoch, saved_g_scheduler["last_epoch"]
            )
            self.assertEqual(
                restored_d_sched.last_epoch, saved_d_scheduler["last_epoch"]
            )

            self._optimizer_step(restored_g, restored_g_opt, next_g_input)
            self._optimizer_step(restored_d, restored_d_opt, next_d_input)
            restored_g_sched.step(2.0)
            restored_d_sched.step(2.0)

            self._assert_nested_equal(control_g.state_dict(), restored_g.state_dict())
            self._assert_nested_equal(control_d.state_dict(), restored_d.state_dict())
            self._assert_nested_equal(
                control_g_opt.state_dict(), restored_g_opt.state_dict()
            )
            self._assert_nested_equal(
                control_d_opt.state_dict(), restored_d_opt.state_dict()
            )
            self._assert_nested_equal(
                control_g_sched.state_dict(), restored_g_sched.state_dict()
            )
            self._assert_nested_equal(
                control_d_sched.state_dict(), restored_d_sched.state_dict()
            )
            self.assertEqual(
                restored_g_sched.last_epoch, saved_g_scheduler["last_epoch"] + 1
            )
            self.assertEqual(
                restored_d_sched.last_epoch, saved_d_scheduler["last_epoch"] + 1
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_map_location_restores_rng_and_exact_next_update(self):
        device = torch.device("cuda:0")
        original_python_rng = random.getstate()
        original_numpy_rng = np.random.get_state()
        original_torch_cpu_rng = torch.get_rng_state()
        original_torch_cuda_rng = torch.cuda.get_rng_state_all()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "parent-cuda.pt"
                torch.manual_seed(101)
                torch.cuda.manual_seed_all(102)
                control = self._components(device)
                (
                    control_g,
                    control_d,
                    control_g_opt,
                    control_d_opt,
                    control_g_sched,
                    control_d_sched,
                ) = control
                self._optimizer_step(
                    control_g, control_g_opt, torch.ones(2, 3, device=device)
                )
                self._optimizer_step(
                    control_d, control_d_opt, torch.ones(2, 2, device=device)
                )
                control_g_sched.step(1.0)
                control_d_sched.step(1.0)

                loader_generator = torch.Generator().manual_seed(777)
                random.seed(10)
                np.random.seed(11)
                torch.manual_seed(12)
                torch.cuda.manual_seed_all(13)
                digest = save_full_training_state(
                    path,
                    generator=control_g,
                    discriminator=control_d,
                    generator_optimizer=control_g_opt,
                    discriminator_optimizer=control_d_opt,
                    generator_scheduler=control_g_sched,
                    discriminator_scheduler=control_d_sched,
                    loader_generator=loader_generator,
                    completed_epoch=1,
                    lineage=_lineage(),
                    contract_sha256=hashlib.sha256(b"contract").hexdigest(),
                )
                saved_g_scheduler = copy.deepcopy(control_g_sched.state_dict())
                saved_d_scheduler = copy.deepcopy(control_d_sched.state_dict())
                expected_rng = (
                    random.random(),
                    float(np.random.random()),
                    torch.rand(3),
                    [
                        torch.rand(3, device=f"cuda:{index}").cpu()
                        for index in range(torch.cuda.device_count())
                    ],
                    torch.randperm(8, generator=loader_generator),
                )

                next_g_input = torch.full((2, 3), 0.25, device=device)
                next_d_input = torch.full((2, 2), -0.5, device=device)
                self._optimizer_step(control_g, control_g_opt, next_g_input)
                self._optimizer_step(control_d, control_d_opt, next_d_input)
                control_g_sched.step(2.0)
                control_d_sched.step(2.0)

                restored = self._components(device)
                (
                    restored_g,
                    restored_d,
                    restored_g_opt,
                    restored_d_opt,
                    restored_g_sched,
                    restored_d_sched,
                ) = restored
                random.seed(999)
                np.random.seed(999)
                torch.manual_seed(999)
                torch.cuda.manual_seed_all(999)
                loader_generator.manual_seed(999)
                load_full_training_state(
                    path,
                    digest,
                    generator=restored_g,
                    discriminator=restored_d,
                    generator_optimizer=restored_g_opt,
                    discriminator_optimizer=restored_d_opt,
                    generator_scheduler=restored_g_sched,
                    discriminator_scheduler=restored_d_sched,
                    loader_generator=loader_generator,
                    expected_lineage=_lineage(),
                    restore_schedulers=True,
                    map_location=device,
                )

                self._assert_nested_equal(
                    restored_g_sched.state_dict(), saved_g_scheduler
                )
                self._assert_nested_equal(
                    restored_d_sched.state_dict(), saved_d_scheduler
                )
                actual_rng = (
                    random.random(),
                    float(np.random.random()),
                    torch.rand(3),
                    [
                        torch.rand(3, device=f"cuda:{index}").cpu()
                        for index in range(torch.cuda.device_count())
                    ],
                    torch.randperm(8, generator=loader_generator),
                )
                self.assertEqual(actual_rng[0], expected_rng[0])
                self.assertEqual(actual_rng[1], expected_rng[1])
                torch.testing.assert_close(actual_rng[2], expected_rng[2])
                self._assert_nested_equal(actual_rng[3], expected_rng[3])
                torch.testing.assert_close(actual_rng[4], expected_rng[4])

                self._optimizer_step(restored_g, restored_g_opt, next_g_input)
                self._optimizer_step(restored_d, restored_d_opt, next_d_input)
                restored_g_sched.step(2.0)
                restored_d_sched.step(2.0)

                self._assert_nested_equal(
                    control_g.state_dict(), restored_g.state_dict()
                )
                self._assert_nested_equal(
                    control_d.state_dict(), restored_d.state_dict()
                )
                self._assert_nested_equal(
                    control_g_opt.state_dict(), restored_g_opt.state_dict()
                )
                self._assert_nested_equal(
                    control_d_opt.state_dict(), restored_d_opt.state_dict()
                )
                self._assert_nested_equal(
                    control_g_sched.state_dict(), restored_g_sched.state_dict()
                )
                self._assert_nested_equal(
                    control_d_sched.state_dict(), restored_d_sched.state_dict()
                )
                self.assertEqual(
                    restored_g_sched.last_epoch,
                    saved_g_scheduler["last_epoch"] + 1,
                )
                self.assertEqual(
                    restored_d_sched.last_epoch,
                    saved_d_scheduler["last_epoch"] + 1,
                )
        finally:
            random.setstate(original_python_rng)
            np.random.set_state(original_numpy_rng)
            torch.set_rng_state(original_torch_cpu_rng)
            torch.cuda.set_rng_state_all(original_torch_cuda_rng)

    def test_weights_only_rejected_and_contracts_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights_path = root / "weights.pt"
            torch.save({"state_dict": {}}, weights_path)
            components = self._components()
            with self.assertRaisesRegex(ValueError, "full_training_state_v1"):
                load_full_training_state(
                    weights_path,
                    sha256_file(weights_path),
                    generator=components[0],
                    discriminator=components[1],
                    generator_optimizer=components[2],
                    discriminator_optimizer=components[3],
                    generator_scheduler=components[4],
                    discriminator_scheduler=components[5],
                    loader_generator=torch.Generator(),
                    expected_lineage=_lineage(),
                    restore_schedulers=True,
                    map_location="cpu",
                )

            parent_contract = write_full_training_state_contract(
                root / "parent-contract.json",
                mode=SAVE_DYNAMIC_FULL_TRAINING_STATE,
                output_path=root / "parent.pt",
                output_lineage=_lineage(),
            )
            loaded = load_full_training_state_contract(
                parent_contract["contract_path"],
                parent_contract["contract_sha256"],
                expected_mode=SAVE_DYNAMIC_FULL_TRAINING_STATE,
            )
            self.assertEqual(loaded.output_lineage, _lineage())

            frozen_contract = write_full_training_state_contract(
                root / "frozen-contract.json",
                mode=RESUME_FROZEN_LR_FULL_TRAINING_STATE,
                output_path=None,
                output_lineage=_lineage(arm="lp_matched"),
                input_path=root / "parent.pt",
                input_sha256=hashlib.sha256(b"parent").hexdigest(),
                expected_input_lineage=_lineage(),
            )
            frozen = load_full_training_state_contract(
                frozen_contract["contract_path"],
                frozen_contract["contract_sha256"],
                expected_mode=RESUME_FROZEN_LR_FULL_TRAINING_STATE,
            )
            self.assertIsNone(frozen.output_path)

    def test_config_payload_hash_breaks_contract_cycle(self):
        first = {
            "seed": 42,
            "news_first_full_training_state_contract_path": "first.json",
            "news_first_full_training_state_contract_sha256": "a" * 64,
        }
        second = dict(first)
        second["news_first_full_training_state_contract_path"] = "second.json"
        second["news_first_full_training_state_contract_sha256"] = "b" * 64
        self.assertEqual(
            training_config_payload_sha256(first),
            training_config_payload_sha256(second),
        )
        second["seed"] = 202
        self.assertNotEqual(
            training_config_payload_sha256(first),
            training_config_payload_sha256(second),
        )

    def test_dynamic_resume_config_requires_plateau(self):
        manifest_sha = hashlib.sha256(b"manifest").hexdigest()
        contract_sha = hashlib.sha256(b"contract").hexdigest()
        with self.assertRaisesRegex(ValueError, "ReduceLROnPlateau"):
            Config(
                cuda=False,
                embedding_dim=1024,
                news_first_pair_text_overlay_mode="current_only",
                news_first_pair_text_manifest_path="manifest.json",
                news_first_pair_text_manifest_sha256=manifest_sha,
                news_first_pair_text_profile_sha256=manifest_sha,
                news_first_full_training_state_mode=(
                    RESUME_DYNAMIC_FULL_TRAINING_STATE
                ),
                news_first_full_training_state_contract_path="contract.json",
                news_first_full_training_state_contract_sha256=contract_sha,
            )


if __name__ == "__main__":
    unittest.main()
