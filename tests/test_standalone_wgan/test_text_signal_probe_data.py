from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from wgan_option.utils.text_signal_probe_data import (
    baseline_unique_article_mean_l2,
    build_wrong_text_donor_plan,
    deduplicate_lp_articles,
    fit_improved_lp_transform,
    load_improved_lp_transform,
    load_wrong_text_donor_manifest,
    recency_weighted_pair_means,
    save_improved_lp_transform,
    transform_improved_lp,
    transform_then_derange,
    validate_wrong_text_donor_mapping,
    write_wrong_text_donor_manifest,
)


def _json_vector(values: list[float]) -> str:
    return json.dumps(values, separators=(",", ":"))


def _article_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    embeddings = {
        "p1": ([3.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]),
        "p2": ([0.0, 2.0, 1.0, 0.0], [0.0, 0.0, 3.0, 0.0]),
        "p3": ([1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 3.0]),
        "v1": ([1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]),
        "v2": ([0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
    }
    news_row_id = 1
    for pair_index, (pair_id, pair_vectors) in enumerate(embeddings.items()):
        pair_timestamp = pd.Timestamp("2023-01-10T15:00:00Z") + pd.Timedelta(
            days=pair_index
        )
        for article_index, vector in enumerate(pair_vectors):
            age = 10 if article_index == 0 else 0
            rows.append(
                {
                    "pair_id": pair_id,
                    "article_id": f"{pair_id}-article-{article_index}",
                    "sample_id": f"{pair_id}-sample-{article_index}",
                    "news_row_id": news_row_id,
                    "lp_embedding": _json_vector(list(vector)),
                    "news_available_time_utc": pair_timestamp
                    - pd.Timedelta(minutes=age),
                    "current_snapshot_time_utc": pair_timestamp,
                }
            )
            news_row_id += 1
    # The historical rule keeps the lower news_row_id for a duplicate article.
    duplicate = dict(rows[0])
    duplicate["news_row_id"] = 999
    duplicate["sample_id"] = "later-duplicate"
    duplicate["lp_embedding"] = _json_vector([99.0, 99.0, 99.0, 99.0])
    rows.append(duplicate)
    return pd.DataFrame(rows)


def _pair_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    definitions = [
        ("t1", "train", "session-a", 10, [1.0, 2.0, 3.0, 4.0]),
        ("t2", "train", "session-a", 20, [1.0, 2.0, 4.0, 5.0]),
        ("t3", "train", "session-b", 11, [2.0, 2.0, 3.0, 4.0]),
        ("t4", "train", "session-b", 21, [2.0, 3.0, 4.0, 5.0]),
        ("v1", "validation", "session-c", 12, [3.0, 2.0, 3.0, 4.0]),
        ("v2", "validation", "session-c", 22, [3.0, 3.0, 4.0, 5.0]),
        ("v3", "validation", "session-d", 13, [4.0, 2.0, 3.0, 4.0]),
        ("v4", "validation", "session-d", 23, [4.0, 3.0, 4.0, 5.0]),
    ]
    for pair_id, partition, session_id, minute, surface in definitions:
        session_open = pd.Timestamp("2023-01-10T00:00:00Z")
        rows.append(
            {
                "pair_id": pair_id,
                "partition": partition,
                "session_id": session_id,
                "session_open_utc": session_open,
                "current_snapshot_time_utc": session_open
                + pd.Timedelta(minutes=minute),
                "current_surface_flat": _json_vector(surface),
                # These prohibited inputs must be invisible to donor planning.
                "target_surface_flat": _json_vector([100.0 + minute]),
                "validation_error": float(minute),
                "lp_text": f"secret text {pair_id}",
            }
        )
    return pd.DataFrame(rows)


class ImprovedLPFeatureTests(unittest.TestCase):
    def test_baseline_and_recency_aggregation_preserve_declared_semantics(self):
        articles = deduplicate_lp_articles(_article_frame(), embedding_dim=4)
        self.assertEqual(len(articles["p1"]), 2)

        baseline = baseline_unique_article_mean_l2(articles)
        expected_baseline = np.asarray([1.5, 2.0, 0.0, 0.0], dtype=np.float32)
        expected_baseline /= np.linalg.norm(expected_baseline)
        np.testing.assert_allclose(baseline["p1"], expected_baseline, atol=1e-7)

        recency = recency_weighted_pair_means(articles, half_life_minutes=5.0)
        # Ages are 10m and 0m, so the unnormalised weights are 0.25 and 1.
        np.testing.assert_allclose(
            recency["p1"],
            np.asarray([0.2, 0.8, 0.0, 0.0], dtype=np.float32),
            atol=1e-7,
        )

    def test_validation_changes_do_not_change_train_only_transform_sha(self):
        frame = _article_frame()
        first_articles = deduplicate_lp_articles(frame, embedding_dim=4)
        first = fit_improved_lp_transform(
            first_articles, ["p1", "p2", "p3"], half_life_minutes=5.0
        )

        changed = frame.copy()
        changed.loc[changed["pair_id"].str.startswith("v"), "lp_embedding"] = (
            _json_vector([50.0, -20.0, 10.0, 3.0])
        )
        second_articles = deduplicate_lp_articles(changed, embedding_dim=4)
        second = fit_improved_lp_transform(
            second_articles, ["p1", "p2", "p3"], half_life_minutes=5.0
        )

        self.assertEqual(first.transform_sha256, second.transform_sha256)
        self.assertEqual(first.train_source_sha256, second.train_source_sha256)
        np.testing.assert_array_equal(first.train_mean, second.train_mean)
        np.testing.assert_array_equal(first.top_pc1, second.top_pc1)
        transformed = transform_improved_lp(first_articles, first)
        self.assertEqual(set(transformed), set(first_articles))
        for vector in transformed.values():
            self.assertTrue(
                np.isclose(np.linalg.norm(vector), 0.0, atol=1e-7)
                or np.isclose(np.linalg.norm(vector), 1.0, atol=1e-6)
            )

    def test_transform_artifact_round_trips_and_rejects_array_tampering(self):
        articles = deduplicate_lp_articles(_article_frame(), embedding_dim=4)
        transform = fit_improved_lp_transform(articles, ["p1", "p2", "p3"])
        with tempfile.TemporaryDirectory() as temporary:
            artifact = save_improved_lp_transform(transform, temporary)
            restored = load_improved_lp_transform(
                artifact["manifest_path"],
                expected_manifest_sha256=artifact["manifest_sha256"],
                expected_transform_sha256=transform.transform_sha256,
            )
            np.testing.assert_array_equal(restored.train_mean, transform.train_mean)
            np.testing.assert_array_equal(restored.top_pc1, transform.top_pc1)

            np.save(
                Path(temporary) / "train_mean.npy",
                np.zeros_like(transform.train_mean),
                allow_pickle=False,
            )
            with self.assertRaisesRegex(ValueError, "file SHA mismatch"):
                load_improved_lp_transform(artifact["manifest_path"])


class WrongTextDonorTests(unittest.TestCase):
    def test_plan_is_deterministic_ignores_prohibited_inputs_and_round_trips(self):
        frame = _pair_frame()
        first = build_wrong_text_donor_plan(
            frame,
            master_seed=314159,
            namespace="probe/wrong-text",
            surface_dim=4,
        )
        changed = frame.copy()
        changed["target_surface_flat"] = _json_vector([-999.0])
        changed["validation_error"] = -999.0
        changed["lp_text"] = "completely different prohibited content"
        second = build_wrong_text_donor_plan(
            changed,
            master_seed=314159,
            namespace="probe/wrong-text",
            surface_dim=4,
        )
        self.assertEqual(first.mapping, second.mapping)
        self.assertEqual(first.profile_sha256, second.profile_sha256)
        validate_wrong_text_donor_mapping(
            first.mapping, first.partition_by_pair, first.session_by_pair
        )
        for receiver, donor in first.mapping.items():
            self.assertNotEqual(receiver, donor)
            self.assertEqual(
                first.partition_by_pair[receiver], first.partition_by_pair[donor]
            )
            self.assertNotEqual(
                first.session_by_pair[receiver], first.session_by_pair[donor]
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wrong_text_donors.json"
            artifact = write_wrong_text_donor_manifest(first, path)
            restored = load_wrong_text_donor_manifest(
                path,
                expected_manifest_sha256=artifact["manifest_sha256"],
                expected_profile_sha256=first.profile_sha256,
            )
            self.assertEqual(restored.mapping, first.mapping)
            self.assertEqual(restored.cost_by_pair, first.cost_by_pair)

    def test_validator_rejects_same_session_mapping(self):
        plan = build_wrong_text_donor_plan(
            _pair_frame(),
            master_seed=7,
            namespace="probe/wrong-text-invalid",
            surface_dim=4,
        )
        invalid = dict(plan.mapping)
        invalid["t1"] = "t2"
        with self.assertRaisesRegex(ValueError, "bijection|receiver session"):
            validate_wrong_text_donor_mapping(
                invalid, plan.partition_by_pair, plan.session_by_pair
            )


class IndependentShuffleTests(unittest.TestCase):
    def test_transform_then_derange_is_split_local_and_preserves_multisets(self):
        pair_ids = ["t1", "t2", "t3", "t4", "v1", "v2", "v3", "v4"]
        embeddings = {
            pair_id: np.asarray([index, index + 0.5], dtype=np.float32)
            for index, pair_id in enumerate(pair_ids)
        }
        partitions = {
            pair_id: "train" if pair_id.startswith("t") else "validation"
            for pair_id in pair_ids
        }
        shuffled, donors = transform_then_derange(
            embeddings,
            partitions,
            master_seed=271828,
            namespace="probe/independent-shuffle",
        )
        repeated, repeated_donors = transform_then_derange(
            embeddings,
            partitions,
            master_seed=271828,
            namespace="probe/independent-shuffle",
        )
        self.assertEqual(donors, repeated_donors)
        for partition in ("train", "validation"):
            ids = sorted(
                pair_id for pair_id, value in partitions.items() if value == partition
            )
            self.assertEqual({donors[pair_id] for pair_id in ids}, set(ids))
            self.assertTrue(all(donors[pair_id] != pair_id for pair_id in ids))
            before = sorted(tuple(embeddings[pair_id]) for pair_id in ids)
            after = sorted(tuple(shuffled[pair_id]) for pair_id in ids)
            self.assertEqual(before, after)
            for pair_id in ids:
                np.testing.assert_array_equal(shuffled[pair_id], repeated[pair_id])


if __name__ == "__main__":
    unittest.main()
