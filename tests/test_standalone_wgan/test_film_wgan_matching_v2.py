import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.matching import build_transition_matching_donor_mapping  # noqa: E402


def _sample(
    index: int,
    embedding: list[float],
    *,
    supported_cells: int = 4,
    has_text: float = 1.0,
):
    support = np.zeros((2, 2), dtype=bool)
    support.reshape(-1)[:supported_cells] = True
    return SimpleNamespace(
        sample_id=f"sample-{index}",
        surface_pair_id=f"pair-{index}",
        current_snapshot_time_utc=f"2023-01-{index + 1:02d}T{8 + index:02d}:00:00Z",
        current_surface=np.full((2, 2), 0.20 + 0.001 * index, dtype=np.float32),
        current_support_mask=np.ones((2, 2), dtype=bool),
        evaluation_support_mask=support,
        text_embedding=np.asarray(embedding, dtype=np.float32),
        metadata={
            "has_text": has_text,
            "article_ids": [f"article-{index}"],
            "source_files": [f"source-{index}"],
            "event_group": f"event-{index}",
            "news_cluster_id": f"cluster-{index}",
            "pair_text_feature_sha256": f"hash-{index}",
            "publication_market_state": "open" if index < 5 else "closed",
            "news_count": 1,
        },
    )


class TestTransitionMatchingDonorMapping(unittest.TestCase):
    def test_mapping_is_deterministic_duplicate_safe_and_eligibility_aware(self):
        samples = [
            _sample(0, [1.0, 0.0, 0.0]),
            # Exact near duplicate of sample 0; the pair must never be selected
            # in either direction despite distinct IDs and lineage strings.
            _sample(1, [1.0, 0.0, 0.0]),
            _sample(2, [0.0, 1.0, 0.0]),
            _sample(3, [0.0, 0.0, 1.0]),
            _sample(4, [-1.0, 0.0, 0.0]),
            _sample(5, [0.0, -1.0, 0.0]),
            _sample(6, [0.0, 0.0, -1.0]),
            _sample(7, [1.0, 1.0, 0.0], supported_cells=1),
            _sample(8, [1.0, 0.0, 1.0], has_text=0.0),
        ]

        first, rows = build_transition_matching_donor_mapping(
            samples,
            negative_count=2,
            minimum_supported_cells=2,
            seed=123,
            duplicate_cosine_threshold=0.995,
        )
        second, second_rows = build_transition_matching_donor_mapping(
            samples,
            negative_count=2,
            minimum_supported_cells=2,
            seed=123,
            duplicate_cosine_threshold=0.995,
        )

        np.testing.assert_array_equal(first, second)
        self.assertEqual(rows, second_rows)
        self.assertTrue(np.all(first[:7] >= 0))
        self.assertTrue(np.all(first[7:] == -1))
        for target_index in range(7):
            self.assertNotIn(target_index, set(first[target_index].tolist()))
            self.assertEqual(len(set(first[target_index].tolist())), 2)
            self.assertNotIn(8, set(first[target_index].tolist()))
        self.assertNotIn(1, set(first[0].tolist()))
        self.assertNotIn(0, set(first[1].tolist()))
        self.assertTrue(all(row["target_surface_pair_id"] != row["donor_surface_pair_id"] for row in rows))

    def test_mapping_rejects_an_insufficient_duplicate_safe_pool(self):
        samples = [
            _sample(0, [1.0, 0.0]),
            _sample(1, [1.0, 0.0]),
            _sample(2, [1.0, 0.0]),
        ]
        with self.assertRaisesRegex(ValueError, "Insufficient duplicate-safe"):
            build_transition_matching_donor_mapping(
                samples,
                negative_count=1,
                minimum_supported_cells=1,
                seed=123,
                duplicate_cosine_threshold=0.995,
            )


if __name__ == "__main__":
    unittest.main()
