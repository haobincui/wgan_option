from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wgan_option.utils.text_ablation import (  # noqa: E402
    CURRENT_ONLY,
    REAL_TEXT,
    TEXT_SHUFFLE,
    fixed_text_shuffle,
    transform_embedding_matrix,
)


class TestTextAblation(unittest.TestCase):
    def setUp(self) -> None:
        self.keys = ["s3", "s1", "s4", "s2", "s6", "s5"]
        self.pairs = ["p2", "p1", "p3", "p1", "p4", "p2"]
        self.sessions = ["d1", "d1", "d2", "d1", "d3", "d2"]
        self.embeddings = np.arange(24, dtype=np.float32).reshape(6, 4)

    def test_shuffle_is_row_order_invariant_pair_breaking_and_auditable(self):
        first = fixed_text_shuffle(
            self.keys,
            seed=42,
            namespace="common_test_core_05m",
            pair_ids=self.pairs,
            session_ids=self.sessions,
        )
        order = [5, 2, 0, 4, 1, 3]
        second = fixed_text_shuffle(
            [self.keys[index] for index in order],
            seed=42,
            namespace="common_test_core_05m",
            pair_ids=[self.pairs[index] for index in order],
            session_ids=[self.sessions[index] for index in order],
        )
        self.assertEqual(first.donor_by_receiver(), second.donor_by_receiver())
        self.assertEqual(first.mapping_sha256, second.mapping_sha256)
        self.assertEqual(first.fixed_point_count, 0)
        self.assertEqual(first.same_pair_count, 0)
        self.assertGreaterEqual(first.same_session_count, 0)
        self.assertEqual(set(first.receiver_keys), set(first.donor_keys))

    def test_shuffle_preserves_embedding_multiset_and_current_only_is_exact_zero(self):
        shuffled, audit = transform_embedding_matrix(
            self.embeddings,
            self.keys,
            mode=TEXT_SHUFFLE,
            seed=42,
            namespace="train_05m",
            pair_ids=self.pairs,
            session_ids=self.sessions,
        )
        self.assertEqual(
            sorted(map(tuple, shuffled.tolist())),
            sorted(map(tuple, self.embeddings.tolist())),
        )
        self.assertEqual(audit["text_shuffle_fixed_point_count"], 0)
        self.assertEqual(audit["text_shuffle_same_pair_count"], 0)
        zero, zero_audit = transform_embedding_matrix(
            self.embeddings,
            self.keys,
            mode=CURRENT_ONLY,
            seed=42,
            namespace="train_05m",
            pair_ids=self.pairs,
            session_ids=self.sessions,
        )
        self.assertTrue(np.array_equal(zero, np.zeros_like(self.embeddings)))
        self.assertTrue(zero_audit["zero_text_verified"])
        real, _ = transform_embedding_matrix(
            self.embeddings,
            self.keys,
            mode=REAL_TEXT,
            seed=42,
            namespace="train_05m",
        )
        self.assertTrue(np.array_equal(real, self.embeddings))

    def test_shuffle_fails_closed_when_pair_breaking_permutation_is_impossible(self):
        with self.assertRaisesRegex(ValueError, "avoid same-pair donors"):
            fixed_text_shuffle(
                ["a", "b", "c"],
                seed=42,
                namespace="train_05m",
                pair_ids=["only", "only", "only"],
                session_ids=["d1", "d1", "d1"],
            )


if __name__ == "__main__":
    unittest.main()
