from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.rq1_event30.event30_dataset import (
    NewsSet,
    _news_for_origin,
    _news_mode,
)


def _minute_news(
    timestamp: str,
    article_id: str,
    embedding_hash: str,
    embedding: list[float],
) -> NewsSet:
    value = np.asarray(embedding, dtype=np.float32)
    return NewsSet(
        article_ids=(article_id,),
        embedding_hashes=(embedding_hash,),
        embeddings=(value,),
        publication_times_utc=(timestamp,),
        pooled_embedding=value / np.linalg.norm(value),
        latest_recency_minutes=0.0,
    )


class Event30NewsWindowTest(unittest.TestCase):
    def test_prior_five_minutes_includes_lower_bound_and_excludes_origin(self) -> None:
        origin = pd.Timestamp("2023-01-03T14:30:00Z")
        news = {
            "2023-01-03T14:25:00Z": _minute_news(
                "2023-01-03T14:25:00Z", "lower", "hash-lower", [1.0, 0.0]
            ),
            "2023-01-03T14:29:00Z": _minute_news(
                "2023-01-03T14:29:00Z", "latest", "hash-latest", [0.0, 1.0]
            ),
            "2023-01-03T14:30:00Z": _minute_news(
                "2023-01-03T14:30:00Z", "origin", "hash-origin", [1.0, 1.0]
            ),
        }

        result = _news_for_origin(
            origin,
            news_by_minute=news,
            embedding_dim=2,
            lookback_minutes=5,
            exclude_origin_minute=True,
        )

        self.assertEqual(result.article_ids, ("lower", "latest"))
        self.assertEqual(result.publication_times_utc[0], "2023-01-03T14:25:00Z")
        self.assertEqual(result.latest_recency_minutes, 1.0)
        self.assertAlmostEqual(
            float(np.linalg.norm(result.pooled_embedding)),
            1.0,
            delta=1.0e-6,
        )

    def test_window_deduplicates_article_and_embedding_hash(self) -> None:
        origin = pd.Timestamp("2023-01-03T14:30:00Z")
        news = {
            "2023-01-03T14:27:00Z": _minute_news(
                "2023-01-03T14:27:00Z", "article-a", "same-hash", [1.0, 0.0]
            ),
            "2023-01-03T14:29:00Z": _minute_news(
                "2023-01-03T14:29:00Z", "article-b", "same-hash", [1.0, 0.0]
            ),
        }

        result = _news_for_origin(
            origin,
            news_by_minute=news,
            embedding_dim=2,
            lookback_minutes=5,
            exclude_origin_minute=True,
        )

        self.assertEqual(result.article_ids, ("article-a",))

    def test_news_mode_names_exclusive_prior_window(self) -> None:
        self.assertEqual(
            _news_mode(5, True),
            ("prior_5m_news", "prior_5m_exclusive"),
        )


if __name__ == "__main__":
    unittest.main()
