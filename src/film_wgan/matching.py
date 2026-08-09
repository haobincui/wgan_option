"""Leakage-safe donor selection for transition--text matching losses."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd


_EMPTY_TOKENS = {"", "nan", "none", "null"}


def _clean_token(value: Any) -> str:
    rendered = str(value).strip()
    return "" if rendered.lower() in _EMPTY_TOKENS else rendered


def _token_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {
            token
            for item in value
            if (token := _clean_token(item))
        }
    token = _clean_token(value)
    return {token} if token else set()


def _metadata_value(sample: Any, source_key: str, fallback_key: str) -> Any:
    metadata = dict(getattr(sample, "metadata", {}) or {})
    source_value = metadata.get(source_key)
    if source_value not in (None, "", [], ()):
        return source_value
    return metadata.get(fallback_key)


def _text_identity(sample: Any) -> dict[str, Any]:
    metadata = dict(getattr(sample, "metadata", {}) or {})
    return {
        "surface_pair_id": _clean_token(
            metadata.get("text_source_surface_pair_id")
            or getattr(sample, "surface_pair_id")
        ),
        "article_ids": _token_set(
            _metadata_value(sample, "text_source_article_ids", "article_ids")
        ),
        "source_files": _token_set(
            _metadata_value(sample, "text_source_files", "source_files")
        ),
        "event_group": _clean_token(
            _metadata_value(sample, "text_source_event_group", "event_group")
        ),
        "news_cluster_id": _clean_token(
            _metadata_value(
                sample,
                "text_source_news_cluster_id",
                "news_cluster_id",
            )
        ),
        "feature_sha256": _clean_token(
            _metadata_value(
                sample,
                "text_source_feature_sha256",
                "pair_text_feature_sha256",
            )
        ),
    }


def _stable_fraction(*parts: Any) -> float:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big") / float(2**64)


def _current_regime(sample: Any) -> tuple[float, float, int, str, float]:
    current = np.asarray(sample.current_surface, dtype=np.float64).reshape(-1)
    support = (
        np.asarray(sample.current_support_mask, dtype=bool).reshape(-1)
        if sample.current_support_mask is not None
        else np.ones(current.size, dtype=bool)
    )
    if not bool(np.any(support)):
        raise ValueError(f"Sample {sample.sample_id} has empty current support.")
    mean_log_iv = float(np.mean(np.log(np.clip(current[support], 1.0e-4, None))))
    support_fraction = float(np.mean(support))
    timestamp = pd.Timestamp(sample.current_snapshot_time_utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    session_bucket = int(timestamp.tz_convert("Europe/London").hour // 4)
    metadata = dict(sample.metadata or {})
    market_state = _clean_token(metadata.get("publication_market_state", ""))
    news_count = float(metadata.get("news_count", 1) or 1)
    if not np.isfinite(news_count):
        news_count = 1.0
    return mean_log_iv, support_fraction, session_bucket, market_state, news_count


def build_transition_matching_donor_mapping(
    samples: Sequence[Any],
    *,
    negative_count: int,
    minimum_supported_cells: int,
    seed: int,
    duplicate_cosine_threshold: float = 0.995,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build a deterministic, split-local current-regime donor mapping.

    Pair support determines whether a target transition contributes to the
    auxiliary loss.  Donor eligibility and ranking use only origin-time
    information: future surfaces, donor target support, fit errors, and model
    scores never enter donor selection. Exact lineage overlaps and
    near-duplicate text embeddings are excluded.
    """

    items = list(samples)
    k = int(negative_count)
    minimum_cells = int(minimum_supported_cells)
    threshold = float(duplicate_cosine_threshold)
    if k < 1:
        raise ValueError("negative_count must be positive.")
    if minimum_cells < 1:
        raise ValueError("minimum_supported_cells must be positive.")
    if not -1.0 < threshold < 1.0:
        raise ValueError("duplicate_cosine_threshold must be strictly between -1 and 1.")

    mapping = np.full((len(items), k), -1, dtype=np.int64)
    support_counts = np.asarray(
        [int(np.asarray(item.evaluation_support_mask, dtype=bool).sum()) for item in items],
        dtype=np.int64,
    )
    target_eligible = [
        index
        for index, item in enumerate(items)
        if float(item.metadata.get("has_text", 1.0)) > 0.0
        and int(support_counts[index]) >= minimum_cells
    ]
    current_support_counts = np.asarray(
        [
            int(
                np.asarray(
                    item.current_support_mask
                    if item.current_support_mask is not None
                    else np.ones_like(item.current_surface, dtype=bool),
                    dtype=bool,
                ).sum()
            )
            for item in items
        ],
        dtype=np.int64,
    )
    # Donor texts need not have a sufficiently observed target transition.
    # Restricting the donor pool by target availability would leak a
    # post-origin quantity into hard-negative selection.  Only text presence
    # and current-time support are required for donors.
    donor_pool = [
        index
        for index, item in enumerate(items)
        if float(item.metadata.get("has_text", 1.0)) > 0.0
        and int(current_support_counts[index]) > 0
    ]
    if not target_eligible:
        raise ValueError(
            "No training samples are eligible for transition matching after the "
            "text/support filters."
        )
    if len(donor_pool) <= k:
        raise ValueError(
            "Transition matching requires at least negative_count + 1 eligible "
            f"donor texts; got donors={len(donor_pool)} negative_count={k}."
        )

    text = np.stack(
        [
            np.asarray(items[index].text_embedding, dtype=np.float64)
            for index in donor_pool
        ],
        axis=0,
    )
    text_norms = np.linalg.norm(text, axis=1)
    if bool(np.any(~np.isfinite(text_norms))) or bool(np.any(text_norms <= 1.0e-12)):
        raise ValueError("Every transition-matching text embedding must be finite and non-zero.")
    text_unit = text / text_norms[:, None]
    position = {
        sample_index: offset for offset, sample_index in enumerate(donor_pool)
    }
    identities = {index: _text_identity(items[index]) for index in donor_pool}
    transition_regimes = {
        index: _current_regime(items[index]) for index in donor_pool
    }
    pair_to_index = {
        str(item.surface_pair_id): index for index, item in enumerate(items)
    }
    # In the shuffled-text control an item carries text from another pair.
    # Rank that text by its true origin-time regime rather than the current
    # surface of the row onto which it was permuted.
    donor_regimes = {
        index: transition_regimes[
            pair_to_index.get(identities[index]["surface_pair_id"], index)
        ]
        for index in donor_pool
    }
    log_scale = float(np.std([donor_regimes[index][0] for index in donor_pool]))
    support_scale = float(
        np.std([donor_regimes[index][1] for index in donor_pool])
    )
    news_scale = float(np.std([donor_regimes[index][4] for index in donor_pool]))
    log_scale = max(log_scale, 1.0e-6)
    support_scale = max(support_scale, 1.0e-6)
    news_scale = max(news_scale, 1.0)

    candidates: dict[int, list[dict[str, Any]]] = {}
    for target_index in target_eligible:
        target = items[target_index]
        target_identity = identities[target_index]
        target_regime = transition_regimes[target_index]
        target_rows: list[dict[str, Any]] = []
        for donor_index in donor_pool:
            if donor_index == target_index:
                continue
            donor = items[donor_index]
            donor_identity = identities[donor_index]
            if donor.surface_pair_id == target.surface_pair_id:
                continue
            if donor_identity["surface_pair_id"] == target_identity["surface_pair_id"]:
                continue
            if target_identity["article_ids"] & donor_identity["article_ids"]:
                continue
            if target_identity["source_files"] & donor_identity["source_files"]:
                continue
            if (
                target_identity["event_group"]
                and target_identity["event_group"] == donor_identity["event_group"]
            ):
                continue
            if (
                target_identity["news_cluster_id"]
                and target_identity["news_cluster_id"]
                == donor_identity["news_cluster_id"]
            ):
                continue
            if (
                target_identity["feature_sha256"]
                and target_identity["feature_sha256"]
                == donor_identity["feature_sha256"]
            ):
                continue
            cosine = float(
                np.dot(text_unit[position[target_index]], text_unit[position[donor_index]])
            )
            if cosine > threshold:
                continue

            donor_regime = donor_regimes[donor_index]
            same_market_state = bool(
                target_regime[3]
                and donor_regime[3]
                and target_regime[3] == donor_regime[3]
            )
            same_session_bucket = target_regime[2] == donor_regime[2]
            tier = 0 if same_market_state and same_session_bucket else 1 if same_market_state else 2
            regime_distance = (
                abs(target_regime[0] - donor_regime[0]) / log_scale
                + abs(target_regime[1] - donor_regime[1]) / support_scale
                + 0.25 * abs(target_regime[2] - donor_regime[2])
                + 0.10 * abs(target_regime[4] - donor_regime[4]) / news_scale
            )
            target_rows.append(
                {
                    "donor_index": donor_index,
                    "tier": tier,
                    "regime_distance": float(regime_distance),
                    "embedding_cosine": cosine,
                    "tie_break": _stable_fraction(
                        int(seed),
                        target.surface_pair_id,
                        donor.surface_pair_id,
                    ),
                }
            )
        target_rows.sort(
            key=lambda row: (
                int(row["tier"]),
                float(row["regime_distance"]),
                float(row["tie_break"]),
                int(row["donor_index"]),
            )
        )
        if len(target_rows) < k:
            raise ValueError(
                "Insufficient duplicate-safe transition-matching donors for "
                f"sample={target.sample_id}: required={k}, available={len(target_rows)}."
            )
        candidates[target_index] = target_rows

    selected_rows: list[dict[str, Any]] = []
    selected_by_target: dict[int, set[int]] = {
        index: set() for index in target_eligible
    }
    for negative_rank in range(k):
        used_this_round: set[int] = set()
        target_order = sorted(
            target_eligible,
            key=lambda index: (
                _stable_fraction(int(seed), negative_rank, items[index].surface_pair_id),
                index,
            ),
        )
        for target_index in target_order:
            available = [
                row
                for row in candidates[target_index]
                if int(row["donor_index"]) not in selected_by_target[target_index]
            ]
            preferred = [
                row
                for row in available
                if int(row["donor_index"]) not in used_this_round
            ]
            reuse_fallback = not bool(preferred)
            chosen = (preferred or available)[0]
            donor_index = int(chosen["donor_index"])
            mapping[target_index, negative_rank] = donor_index
            selected_by_target[target_index].add(donor_index)
            used_this_round.add(donor_index)
            target = items[target_index]
            donor = items[donor_index]
            selected_rows.append(
                {
                    "target_dataset_index": target_index,
                    "target_sample_id": str(target.sample_id),
                    "target_surface_pair_id": str(target.surface_pair_id),
                    "target_text_source_surface_pair_id": identities[target_index][
                        "surface_pair_id"
                    ],
                    "negative_rank": negative_rank,
                    "donor_dataset_index": donor_index,
                    "donor_sample_id": str(donor.sample_id),
                    "donor_surface_pair_id": str(donor.surface_pair_id),
                    "donor_text_source_surface_pair_id": identities[donor_index][
                        "surface_pair_id"
                    ],
                    "selection_tier": int(chosen["tier"]),
                    "regime_distance": float(chosen["regime_distance"]),
                    "embedding_cosine": float(chosen["embedding_cosine"]),
                    "round_reuse_fallback": int(reuse_fallback),
                    "mapping_seed": int(seed),
                    "minimum_supported_cells": minimum_cells,
                    "target_supported_cells": int(support_counts[target_index]),
                    "donor_current_supported_cells": int(
                        current_support_counts[donor_index]
                    ),
                }
            )

    selected_rows.sort(
        key=lambda row: (
            int(row["target_dataset_index"]),
            int(row["negative_rank"]),
        )
    )
    return mapping, selected_rows
