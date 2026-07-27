from __future__ import annotations

import unittest

import pandas as pd

from film_wgan.text_lineage import build_text_lineage_audit


class TextLineageAuditTests(unittest.TestCase):
    def test_missing_ids_are_row_specific_and_empty_text_is_excluded(self):
        frame = pd.DataFrame(
            {
                "ArticleID": [float("nan"), float("nan"), "A-3"],
                "SourceFile": ["one", "two", "three"],
                "LP": ["same market text", "same market text", ""],
                "LP_embedding": [
                    "[0.1, 0.2]",
                    "[0.1, 0.2]",
                    "[0.1, 0.2]",
                ],
            }
        )
        rows, groups, summary = build_text_lineage_audit(frame)
        self.assertEqual(rows.loc[0, "article_id_effective"], "news_row_1")
        self.assertEqual(rows.loc[1, "article_id_effective"], "news_row_2")
        self.assertEqual(
            rows.loc[2, "lineage_status"],
            "exclude_empty_lp_nonzero_embedding",
        )
        self.assertEqual(summary["missing_article_id_count"], 2)
        exact_embedding = groups[groups["group_type"] == "exact_embedding"]
        self.assertEqual(int(exact_embedding.iloc[0]["row_count"]), 3)

    def test_reordered_tokens_share_a_near_duplicate_cluster(self):
        frame = pd.DataFrame(
            {
                "ArticleID": ["A-1", "A-2", "A-3"],
                "SourceFile": ["one", "two", "three"],
                "LP": [
                    "rates inflation treasury market",
                    "market treasury inflation rates",
                    "unrelated earnings guidance",
                ],
                "LP_embedding": [
                    "[0.1, 0.2]",
                    "[0.3, 0.4]",
                    "[0.5, 0.6]",
                ],
            }
        )
        rows, groups, _summary = build_text_lineage_audit(frame)
        self.assertNotEqual(
            rows.loc[0, "lp_text_sha256"],
            rows.loc[1, "lp_text_sha256"],
        )
        self.assertEqual(
            rows.loc[0, "near_duplicate_cluster_id"],
            rows.loc[1, "near_duplicate_cluster_id"],
        )
        self.assertTrue(rows.loc[0, "near_duplicate_cluster_id"])
        near_groups = groups[
            groups["group_type"] == "near_text_hamming_le_3"
        ]
        self.assertEqual(int(near_groups.iloc[0]["row_count"]), 2)


if __name__ == "__main__":
    unittest.main()
