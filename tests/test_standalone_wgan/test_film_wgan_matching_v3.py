import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.matching import (  # noqa: E402
    TextAlignmentPlan,
    TransitionMatchingNegativePlan,
    build_text_alignment_plan,
    build_transition_matching_negative_source_plan,
)
from film_wgan.config import FilmWGANSampleConfig, FilmWGANTrainConfig  # noqa: E402
from film_wgan.data import (  # noqa: E402
    _permutation_for_pair_ids,
    create_train_val_bundle,
    load_film_wgan_samples,
    split_samples,
)
from tests.test_standalone_wgan.test_film_wgan_module import (  # noqa: E402
    _write_grouped_vol_workbook,
)


def _sample(
    index: int,
    embedding: np.ndarray,
    *,
    has_text: float = 1.0,
    supported_cells: int = 4,
    article_ids: list[str] | None = None,
    source_files: list[str] | None = None,
):
    support = np.zeros((2, 2), dtype=bool)
    support.reshape(-1)[:supported_cells] = True
    return SimpleNamespace(
        sample_id=f"sample-{index}",
        surface_pair_id=f"pair-{index}",
        current_snapshot_time_utc=(
            f"2023-01-{index + 1:02d}T{8 + index % 8:02d}:00:00Z"
        ),
        current_surface=np.full((2, 2), 0.20 + index * 0.001, dtype=np.float32),
        current_support_mask=np.ones((2, 2), dtype=bool),
        evaluation_support_mask=support,
        text_embedding=np.asarray(embedding, dtype=np.float32),
        metadata={
            "has_text": has_text,
            "article_ids": article_ids or [f"article-{index}"],
            "source_files": source_files or [f"source-{index}"],
            "source_ids": [f"source-id-{index}"],
            "event_group": f"event-{index}",
            "news_cluster_id": f"cluster-{index}",
            "pair_text_feature_sha256": f"hash-{index}",
            "publication_market_state": "open" if index % 2 == 0 else "closed",
            "news_count": 1,
        },
    )


def _orthogonal_samples(count: int) -> list[SimpleNamespace]:
    return [_sample(index, np.eye(count, dtype=np.float32)[index]) for index in range(count)]


class TestTextAlignmentPlanV3(unittest.TestCase):
    def test_schema5_implicit_permutation_keeps_historical_numpy_mapping(self):
        np.testing.assert_array_equal(
            _permutation_for_pair_ids(
                ["a", "b", "c", "d", "e"],
                seed=134,
            ),
            np.asarray([1, 4, 0, 2, 3], dtype=np.int64),
        )

    def test_plan_is_a_reproducible_pair_derangement_and_round_trips(self):
        samples = _orthogonal_samples(8)
        first = build_text_alignment_plan(samples, split="train", seed=123)
        second = build_text_alignment_plan(samples, split="train", seed=123)

        np.testing.assert_array_equal(
            first.placebo_source_indices,
            second.placebo_source_indices,
        )
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(
            sorted(first.placebo_source_indices.tolist()),
            list(range(len(samples))),
        )
        for target_index, source_index in enumerate(first.placebo_source_indices):
            self.assertNotEqual(
                samples[target_index].surface_pair_id,
                samples[int(source_index)].surface_pair_id,
            )
        restored = TextAlignmentPlan.from_frame(first.to_frame(), split="train")
        self.assertEqual(restored.sha256, first.sha256)
        np.testing.assert_array_equal(
            restored.positive_source_indices("matched"),
            np.arange(len(samples)),
        )
        np.testing.assert_array_equal(
            restored.positive_source_indices("shuffled"),
            first.placebo_source_indices,
        )

    def test_plan_rejects_non_bijection_and_same_pair_source(self):
        samples = _orthogonal_samples(4)
        with self.assertRaisesRegex(ValueError, "bijection"):
            build_text_alignment_plan(
                samples,
                split="val",
                seed=7,
                placebo_source_indices=[1, 1, 3, 0],
            )
        with self.assertRaisesRegex(ValueError, "derangement"):
            build_text_alignment_plan(
                samples,
                split="val",
                seed=7,
                placebo_source_indices=[0, 2, 3, 1],
            )

    def test_frozen_alignment_frame_rejects_schema_hash_order_and_source_tampering(self):
        plan = build_text_alignment_plan(
            _orthogonal_samples(6),
            split="train",
            seed=123,
        )
        frame = plan.to_frame()

        missing_sha = frame.drop(columns=["text_alignment_plan_sha256"])
        with self.assertRaisesRegex(ValueError, "missing fields.*sha256"):
            TextAlignmentPlan.from_frame(missing_sha, split="train")

        bad_version = frame.copy()
        bad_version["mapping_version"] = "tampered_alignment_version"
        with self.assertRaisesRegex(ValueError, "Unsupported text alignment"):
            TextAlignmentPlan.from_frame(bad_version, split="train")

        bad_sha = frame.copy()
        bad_sha["text_alignment_plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            TextAlignmentPlan.from_frame(bad_sha, split="train")

        bad_order = frame.copy()
        bad_order.loc[0, "target_dataset_index"] = len(frame) + 1
        with self.assertRaisesRegex(ValueError, "canonical order"):
            TextAlignmentPlan.from_frame(bad_order, split="train")

        bad_native = frame.copy()
        bad_native.loc[0, "native_source_sample_id"] = "wrong-native"
        with self.assertRaisesRegex(ValueError, "source identity"):
            TextAlignmentPlan.from_frame(bad_native, split="train")

        bad_native_index = frame.copy()
        bad_native_index.loc[0, "native_source_index"] = 1
        with self.assertRaisesRegex(ValueError, "native source index"):
            TextAlignmentPlan.from_frame(bad_native_index, split="train")

        bad_placebo = frame.copy()
        bad_placebo.loc[0, "placebo_source_surface_pair_id"] = "wrong-placebo"
        with self.assertRaisesRegex(ValueError, "source identity"):
            TextAlignmentPlan.from_frame(bad_placebo, split="train")

        bad_placebo_index = frame.copy()
        bad_placebo_index.loc[0, "placebo_source_index"] = (
            int(frame.loc[0, "placebo_source_index"]) + 1
        ) % len(frame)
        with self.assertRaisesRegex(
            ValueError,
            "source identity|derangement|bijection",
        ):
            TextAlignmentPlan.from_frame(bad_placebo_index, split="train")

    def test_data_bundle_reloads_train_val_frozen_plan_and_leaves_test_native(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = _write_grouped_vol_workbook(tmpdir)
            legacy_config = FilmWGANTrainConfig(
                data_path=str(workbook_path),
                split_strategy="grouped_chronological",
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                text_alignment_mode="permuted",
                text_permutation_seed=123,
                batch_size=8,
                cuda=False,
            )
            prepared = create_train_val_bundle(legacy_config)
            plan_path = Path(tmpdir) / "text_alignment_plan.csv"
            pd.concat(
                [
                    prepared.text_alignment_plans["train"].to_frame(),
                    prepared.text_alignment_plans["val"].to_frame(),
                ],
                ignore_index=True,
            ).to_csv(plan_path, index=False)

            frozen = create_train_val_bundle(
                replace(legacy_config, text_alignment_plan_path=str(plan_path))
            )
            matched = create_train_val_bundle(
                replace(
                    legacy_config,
                    text_alignment_mode="matched",
                    text_alignment_plan_path=str(plan_path),
                )
            )
            self.assertEqual(
                frozen.text_alignment_plans["train"].sha256,
                prepared.text_alignment_plans["train"].sha256,
            )
            self.assertEqual(
                frozen.text_alignment_plans["val"].sha256,
                prepared.text_alignment_plans["val"].sha256,
            )
            self.assertNotIn("test", frozen.text_alignment_plans)
            for split_name in ("train", "val"):
                plan = frozen.text_alignment_plans[split_name]
                permuted_items = getattr(frozen, f"{split_name}_items")
                matched_items = getattr(matched, f"{split_name}_items")
                native_items = getattr(frozen, f"native_{split_name}_items")
                for index, (permuted_item, matched_item) in enumerate(
                    zip(permuted_items, matched_items)
                ):
                    placebo_index = int(plan.placebo_source_indices[index])
                    self.assertEqual(
                        matched_item.metadata["text_positive_source_index"],
                        index,
                    )
                    self.assertEqual(
                        permuted_item.metadata["text_positive_source_index"],
                        placebo_index,
                    )
                    np.testing.assert_array_equal(
                        matched_item.text_embedding,
                        native_items[index].text_embedding,
                    )
                    np.testing.assert_array_equal(
                        permuted_item.text_embedding,
                        native_items[placebo_index].text_embedding,
                    )
            for aligned, native in zip(frozen.test_items, frozen.native_test_items):
                np.testing.assert_array_equal(aligned.text_embedding, native.text_embedding)
                self.assertNotIn("text_alignment_plan_sha256", aligned.metadata)

            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                create_train_val_bundle(
                    replace(
                        legacy_config,
                        text_alignment_plan_path=str(plan_path),
                        text_permutation_seed=124,
                    )
                )

            tampered_path = Path(tmpdir) / "tampered_alignment_plan.csv"
            tampered = pd.read_csv(plan_path)
            tampered.loc[0, "target_sample_id"] = "wrong-item-order"
            tampered.to_csv(tampered_path, index=False)
            with self.assertRaisesRegex(
                ValueError,
                "SHA256|item order|source identity",
            ):
                create_train_val_bundle(
                    replace(
                        legacy_config,
                        text_alignment_plan_path=str(tampered_path),
                    )
                )

            sample_common = {
                "data_path": str(workbook_path),
                "split_strategy": "grouped_chronological",
                "train_ratio": 0.5,
                "val_ratio": 0.25,
                "test_ratio": 0.25,
                "text_alignment_mode": "permuted",
                "text_permutation_seed": 123,
                "text_alignment_plan_path": str(plan_path),
                "cuda": False,
            }
            val_config = FilmWGANSampleConfig(**sample_common, split="val")
            val_items = split_samples(
                val_config,
                load_film_wgan_samples(val_config),
            )
            self.assertEqual(len(val_items), len(frozen.val_items))
            self.assertTrue(
                all("text_alignment_plan_sha256" in item.metadata for item in val_items)
            )
            test_config = FilmWGANSampleConfig(**sample_common, split="test")
            with self.assertRaisesRegex(ValueError, "empty for the requested split"):
                split_samples(
                    test_config,
                    load_film_wgan_samples(test_config),
                )


class TestSymmetricNegativeSourcePlanV3(unittest.TestCase):
    def test_native_and_placebo_union_is_protected_and_plan_round_trips(self):
        samples = _orthogonal_samples(18)
        samples[2].metadata["article_ids"] = ["article-0"]
        samples[3].metadata["source_files"] = ["source-1"]
        samples[4].metadata["source_ids"] = ["source-id-0"]
        samples[5].metadata["event_group"] = "event-1"
        samples[6].metadata["news_cluster_id"] = "cluster-0"
        samples[7].metadata["pair_text_feature_sha256"] = "hash-1"
        # Sources 8/9 are distinct but above the near-duplicate cosine
        # threshold for the native/placebo positives of target 0.
        samples[8].text_embedding = (
            samples[0].text_embedding + 0.04 * samples[9].text_embedding
        )
        samples[9].text_embedding = (
            samples[1].text_embedding + 0.04 * samples[10].text_embedding
        )
        for near_index, positive_index in ((8, 0), (9, 1)):
            cosine = float(
                np.dot(
                    samples[near_index].text_embedding,
                    samples[positive_index].text_embedding,
                )
                / np.linalg.norm(samples[near_index].text_embedding)
            )
            self.assertGreater(cosine, 0.995)
            self.assertLess(cosine, 1.0)
        placebo = np.roll(np.arange(len(samples), dtype=np.int64), -1)
        alignment = build_text_alignment_plan(
            samples,
            split="train",
            seed=123,
            placebo_source_indices=placebo,
        )

        plan = build_transition_matching_negative_source_plan(
            samples,
            text_alignment_plan=alignment,
            split="train",
            negative_count=2,
            minimum_supported_cells=2,
            seed=456,
            duplicate_cosine_threshold=0.995,
        )
        target_zero_negatives = set(plan.negative_source_indices[0].tolist())
        self.assertTrue(
            target_zero_negatives.isdisjoint({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        )
        self.assertEqual(plan.summary["native_positive_as_negative_count"], 0)
        self.assertEqual(plan.summary["placebo_positive_as_negative_count"], 0)
        self.assertEqual(plan.summary["rank_reuse_count"], 0)
        self.assertEqual(plan.summary["fallback_count"], 0)
        self.assertGreaterEqual(plan.summary["safe_candidate_count_p05"], 2)
        self.assertGreaterEqual(plan.summary["safe_candidate_count_median"], 2)
        for rank in range(plan.negative_count):
            values = plan.negative_source_indices[plan.eligible_mask, rank].tolist()
            self.assertEqual(len(values), len(set(values)))

        restored = TransitionMatchingNegativePlan.from_frame(
            plan.to_frame(),
            split="train",
        )
        self.assertEqual(restored.sha256, plan.sha256)
        np.testing.assert_array_equal(
            restored.negative_source_indices,
            plan.negative_source_indices,
        )
        np.testing.assert_array_equal(restored.eligible_mask, plan.eligible_mask)

    def test_frozen_negative_frame_rejects_schema_hash_order_and_identity_tampering(self):
        samples = _orthogonal_samples(10)
        alignment = build_text_alignment_plan(
            samples,
            split="val",
            seed=123,
        )
        plan = build_transition_matching_negative_source_plan(
            samples,
            text_alignment_plan=alignment,
            split="val",
            negative_count=2,
            minimum_supported_cells=2,
            seed=456,
        )
        frame = plan.to_frame()

        missing_sha = frame.drop(columns=["matching_negative_source_plan_sha256"])
        with self.assertRaisesRegex(ValueError, "missing fields.*sha256"):
            TransitionMatchingNegativePlan.from_frame(missing_sha, split="val")

        bad_version = frame.copy()
        bad_version["plan_version"] = "tampered_negative_version"
        with self.assertRaisesRegex(ValueError, "Unsupported transition-matching"):
            TransitionMatchingNegativePlan.from_frame(bad_version, split="val")

        bad_sha = frame.copy()
        bad_sha["matching_negative_source_plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            TransitionMatchingNegativePlan.from_frame(bad_sha, split="val")

        bad_order = frame.copy()
        bad_order.loc[0, "target_dataset_index"] = len(samples) + 1
        with self.assertRaisesRegex(ValueError, "canonical order"):
            TransitionMatchingNegativePlan.from_frame(bad_order, split="val")

        bad_source_sample = frame.copy()
        bad_source_sample.loc[0, "negative_source_sample_id"] = "wrong-negative"
        with self.assertRaisesRegex(ValueError, "Negative-source sample ID"):
            TransitionMatchingNegativePlan.from_frame(bad_source_sample, split="val")

        bad_source_pair = frame.copy()
        bad_source_pair.loc[0, "negative_source_surface_pair_id"] = "wrong-negative"
        with self.assertRaisesRegex(ValueError, "surface-pair ID"):
            TransitionMatchingNegativePlan.from_frame(bad_source_pair, split="val")

        bad_source_index = frame.copy()
        bad_source_index.loc[0, "negative_source_index"] = (
            int(frame.loc[0, "negative_source_index"]) + 1
        ) % len(samples)
        with self.assertRaisesRegex(
            ValueError,
            "Negative-source sample ID|reuse|positive",
        ):
            TransitionMatchingNegativePlan.from_frame(bad_source_index, split="val")

        bad_native = frame.copy()
        bad_native.loc[0, "native_positive_sample_id"] = "wrong-native"
        with self.assertRaisesRegex(ValueError, "native-positive sample ID"):
            TransitionMatchingNegativePlan.from_frame(bad_native, split="val")

        bad_placebo = frame.copy()
        bad_placebo.loc[0, "placebo_positive_sample_id"] = "wrong-placebo"
        with self.assertRaisesRegex(ValueError, "placebo-positive sample ID"):
            TransitionMatchingNegativePlan.from_frame(bad_placebo, split="val")

    def test_mixed_has_text_uses_shared_arm_eligibility(self):
        samples = _orthogonal_samples(8)
        samples[1].metadata["has_text"] = 0.0
        samples[1].text_embedding = np.zeros(8, dtype=np.float32)
        placebo = np.roll(np.arange(len(samples), dtype=np.int64), -1)
        alignment = build_text_alignment_plan(
            samples,
            split="val",
            seed=99,
            placebo_source_indices=placebo,
        )
        plan = build_transition_matching_negative_source_plan(
            samples,
            text_alignment_plan=alignment,
            split="val",
            negative_count=1,
            minimum_supported_cells=2,
            seed=101,
        )

        # Target 0 has text but its placebo source (1) does not. Target 1 has
        # no native text. Both are excluded symmetrically from both arms.
        self.assertFalse(bool(plan.eligible_mask[0]))
        self.assertFalse(bool(plan.eligible_mask[1]))
        self.assertTrue(bool(np.all(plan.eligible_mask[2:])))
        self.assertTrue(bool(np.all(plan.negative_source_indices[:2] == -1)))

    def test_candidate_shortage_hard_fails_without_fallback(self):
        samples = [
            _sample(index, np.asarray([1.0, 0.0], dtype=np.float32))
            for index in range(4)
        ]
        alignment = build_text_alignment_plan(
            samples,
            split="train",
            seed=5,
            placebo_source_indices=[1, 2, 3, 0],
        )
        with self.assertRaisesRegex(ValueError, "Insufficient.*union-safe"):
            build_transition_matching_negative_source_plan(
                samples,
                text_alignment_plan=alignment,
                split="train",
                negative_count=1,
                minimum_supported_cells=1,
                seed=6,
            )


if __name__ == "__main__":
    unittest.main()
