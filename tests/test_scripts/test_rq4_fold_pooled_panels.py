from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.rq3.news_first_vol_rq4_fold_pooled_panels import (  # noqa: E402
    FoldPooledPanelError,
    _payload_sha256,
    build_fold_pooled_panels,
    materialize_fold_pooled_panels,
    sha256_file,
    validate_fold_pooled_bundle,
)


FOLD = "fixture_fold"
EXPECTED = {
    FOLD: {
        "splits": {
            "train": {"pairs": 2, "sessions": 2},
            "validation": {"pairs": 1, "sessions": 1},
            "test": {"pairs": 1, "sessions": 1},
        },
        "pairs": 4,
        "sessions": 4,
        "scheduled": 3,
        "jump": 2,
        "both": 1,
    }
}


def _vector(length: int, first: float = 1.0, second: float = 0.0) -> str:
    values = np.zeros(length, dtype=float)
    values[0] = first
    values[1] = second
    return json.dumps(values.tolist(), separators=(",", ":"))


def _fixtures() -> tuple[pd.DataFrame, ...]:
    pairs = [
        ("pair_train_a", "session_a", "train", "2023-01-03T10:00:00Z"),
        ("pair_train_b", "session_b", "train", "2023-01-03T10:10:00Z"),
        ("pair_val", "session_c", "validation", "2023-01-04T11:00:00Z"),
        ("pair_test", "session_d", "test", "2023-01-05T12:00:00Z"),
    ]
    universe_rows = []
    five_members = {"pair_train_a", "pair_val", "pair_test"}
    for tolerance in (30, 5):
        for pair_id, session_id, split, origin in pairs:
            if tolerance == 5 and pair_id not in five_members:
                continue
            current = pd.Timestamp(origin)
            universe_rows.append(
                {
                    "tolerance_minutes": tolerance,
                    "fold": FOLD,
                    "partition": split,
                    "pair_id": pair_id,
                    "session_id": session_id,
                    "effective_origin_utc": origin,
                    "current_snapshot_time_utc": origin,
                    "target_snapshot_time_utc": (current + pd.Timedelta(minutes=5))
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "joint_support_cells": 8,
                }
            )
    universes = pd.DataFrame(universe_rows)

    support_rows = []
    merged_rows = []
    for index, (pair_id, session_id, _, origin) in enumerate(pairs, start=1):
        target = (
            (pd.Timestamp(origin) + pd.Timedelta(minutes=5))
            .isoformat()
            .replace("+00:00", "Z")
        )
        support_rows.append(
            {
                "tolerance_minutes": 30,
                "pair_id": pair_id,
                "effective_origin_utc": origin,
                "target_snapshot_time_utc": target,
                "surface_training_eligible": True,
                "grid_cell_count": 256,
                "joint_strict_support_cell_count": 8,
                "joint_zero_support": False,
                "support_method": "fixture_raw_joint",
                "grid_fingerprint": "fixture-grid",
            }
        )
        merged_rows.append(
            {
                "pair_id": pair_id,
                "session_id": session_id,
                "effective_origin_utc": origin,
                "current_snapshot_time_utc": origin,
                "target_snapshot_time_utc": target,
                "current_surface_flat": _vector(256, 0.2 + index / 100),
                "target_surface_flat": _vector(256, 0.3 + index / 100),
                "current_surface_param_json": "{}",
                "target_surface_param_json": "{}",
                "strike_grid": json.dumps(list(range(16))),
                "maturity_days_grid": json.dumps(list(range(16))),
                "surface_shape": "[16,16]",
                "sample_id": f"source_{index}",
                "news_row_id": index * 10,
                "article_id": f"article_{index}",
                "lp_text": f"article {index}",
                "lp_embedding": _vector(1024, 1.0, 0.0),
                "lp_dim": 1024,
                "sample_weight": 0.5,
                "dataset_tolerance_minutes": 30,
            }
        )
    # Two unique articles and one repeated article.  The repeated article has a
    # conflicting vector deliberately: RQ1 semantics retain its first stable
    # occurrence and include the second unique article exactly once.
    first = dict(merged_rows[0])
    first.update(
        sample_id="source_second_article",
        news_row_id=11,
        article_id="article_train_a_2",
        lp_text="second article",
        lp_embedding=_vector(1024, 0.0, 1.0),
    )
    duplicate = dict(merged_rows[0])
    duplicate.update(
        sample_id="source_duplicate",
        news_row_id=12,
        lp_embedding=_vector(1024, 0.0, 1.0),
    )
    merged_rows.extend([first, duplicate])
    merged = pd.DataFrame(merged_rows)
    support = pd.DataFrame(support_rows)
    events = pd.DataFrame(
        [
            {
                "event_id": "event_train",
                "release_time_utc": "2023-01-03T10:00:00Z",
                "scheduled_or_unscheduled": "scheduled",
            },
            {
                "event_id": "event_val",
                "release_time_utc": "2023-01-04T11:00:00Z",
                "scheduled_or_unscheduled": "scheduled",
            },
        ]
    )
    jumps = pd.DataFrame(
        [
            {
                "pair_id": "pair_train_b",
                "origin_time_utc": "2023-01-03T10:10:00Z",
                "target_time_utc": "2023-01-03T10:15:00Z",
                "session_id": "session_b",
                "horizon_minutes": 5,
                "anomaly_tier": "broad",
            },
            {
                "pair_id": "outside_primary",
                "origin_time_utc": "2022-12-01T13:00:00Z",
                "target_time_utc": "2022-12-01T13:05:00Z",
                "session_id": "outside_session_primary",
                "horizon_minutes": 5,
                "anomaly_tier": "primary",
            },
            {
                "pair_id": "pair_test",
                "origin_time_utc": "2023-01-05T12:00:00Z",
                "target_time_utc": "2023-01-05T12:05:00Z",
                "session_id": "session_d",
                "horizon_minutes": 5,
                "anomaly_tier": "high",
            },
        ]
    )
    return merged, support, universes, events, jumps


class FoldPooledPanelTests(unittest.TestCase):
    def test_build_pools_splits_and_reuses_rq1_lp_collapse(self) -> None:
        merged, support, universes, events, jumps = _fixtures()
        panels, summary, audit = build_fold_pooled_panels(
            merged_vol=merged,
            support_audit=support,
            pair_universes=universes,
            frozen_scheduled_events=events,
            market_jump_pairs=jumps,
            expected_counts=EXPECTED,
        )
        panel = panels[FOLD].set_index("pair_id")
        self.assertEqual(len(panel), 4)
        self.assertEqual(set(panel["source_split"]), {"train", "validation", "test"})
        self.assertEqual(panel.loc["pair_train_b", "event_regime"], "both")
        self.assertEqual(panel.loc["pair_test", "event_regime"], "jump_only")
        self.assertEqual(panel.loc["pair_val", "event_regime"], "scheduled_only")
        self.assertFalse(bool(panel.loc["pair_train_b", "source_5m_membership"]))
        self.assertTrue(bool(panel.loc["pair_test", "source_5m_membership"]))
        self.assertEqual(panel.loc["pair_train_a", "sample_id"], "pair::pair_train_a")
        self.assertEqual(float(panel.loc["pair_train_a", "sample_weight"]), 1.0)
        self.assertEqual(
            panel.loc["pair_train_a", "pair_lp_transform"],
            "unique_article_lp_mean_l2_v1",
        )
        self.assertEqual(int(panel.loc["pair_train_a", "pair_unique_article_count"]), 2)
        embedding = np.asarray(
            json.loads(panel.loc["pair_train_a", "lp_embedding"]), dtype=float
        )
        self.assertAlmostEqual(embedding[0], 1.0 / np.sqrt(2.0), places=6)
        self.assertAlmostEqual(embedding[1], 1.0 / np.sqrt(2.0), places=6)
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=6)
        self.assertEqual(int(summary.iloc[0]["pair_count"]), 4)
        self.assertEqual(audit["routed_row_count"], 4)

    def test_count_drift_fails_closed(self) -> None:
        merged, support, universes, events, jumps = _fixtures()
        wrong = json.loads(json.dumps(EXPECTED))
        wrong[FOLD]["pairs"] = 5
        with self.assertRaisesRegex(FoldPooledPanelError, "Count drift"):
            build_fold_pooled_panels(
                merged_vol=merged,
                support_audit=support,
                pair_universes=universes,
                frozen_scheduled_events=events,
                market_jump_pairs=jumps,
                expected_counts=wrong,
            )

    def test_materialized_bundle_is_signed_idempotent_and_source_bound(self) -> None:
        merged, support, universes, events, jumps = _fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            paths = {
                "merged_vol": source / "merged.csv",
                "support_audit": source / "support.csv",
                "pair_universes": source / "pair_universes.csv",
                "frozen_scheduled_events": source / "frozen_scheduled_events.csv",
                "market_jump_pairs": source / "candidate_pairs.csv",
            }
            for role, frame in zip(
                paths,
                (merged, support, universes, events, jumps),
                strict=True,
            ):
                frame.to_csv(paths[role], index=False)

            pair_manifest = {
                "schema_version": 1,
                "kind": "rq123_pair_universe_manifest_v1",
                "pair_universes": {
                    "path": str(paths["pair_universes"].resolve()),
                    "sha256": sha256_file(paths["pair_universes"]),
                    "size_bytes": paths["pair_universes"].stat().st_size,
                },
            }
            pair_manifest["manifest_sha256"] = _payload_sha256(pair_manifest)
            pair_manifest_path = source / "pair_universe_manifest.json"
            pair_manifest_path.write_text(json.dumps(pair_manifest), encoding="utf-8")

            event_manifest = {
                "schema_version": 1,
                "kind": "rq123_frozen_event_sources_v1",
                "scheduled_events_path": str(
                    paths["frozen_scheduled_events"].resolve()
                ),
                "scheduled_events_sha256": sha256_file(
                    paths["frozen_scheduled_events"]
                ),
                "market_jump_pairs_path": str(paths["market_jump_pairs"].resolve()),
                "market_jump_pairs_sha256": sha256_file(paths["market_jump_pairs"]),
            }
            event_manifest["payload_sha256"] = _payload_sha256(event_manifest)
            event_manifest_path = source / "frozen_event_sources.json"
            event_manifest_path.write_text(json.dumps(event_manifest), encoding="utf-8")

            validation_path = source / "validation_summary.json"
            validation_path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "tolerance_minutes": 30,
                        "workbook_path": str(paths["merged_vol"].resolve()),
                        "workbook_size_bytes": paths["merged_vol"].stat().st_size,
                        "joint_strict_support_pairs": 4,
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "rq4_fold_pooled": {
                    "data": {
                        "merged_vol_path": str(paths["merged_vol"]),
                        "support_audit_path": str(paths["support_audit"]),
                        "pair_universes_path": str(paths["pair_universes"]),
                        "frozen_scheduled_events_path": str(
                            paths["frozen_scheduled_events"]
                        ),
                        "market_jump_pairs_path": str(paths["market_jump_pairs"]),
                        "pair_universe_manifest_path": str(pair_manifest_path),
                        "frozen_event_sources_manifest_path": str(event_manifest_path),
                        "dataset_validation_path": str(validation_path),
                    },
                    "expected_fold_counts": EXPECTED,
                }
            }
            output = root / "bundle"
            first = materialize_fold_pooled_panels(config, output)
            first_hash = sha256_file(first["manifest"])
            second = materialize_fold_pooled_panels(config, output)
            self.assertEqual(first_hash, sha256_file(second["manifest"]))
            self.assertEqual(validate_fold_pooled_bundle(output)[FOLD], first[FOLD])

            paths["support_audit"].write_text(
                paths["support_audit"].read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FoldPooledPanelError, "Source artifact drift"):
                validate_fold_pooled_bundle(output)


if __name__ == "__main__":
    unittest.main()
