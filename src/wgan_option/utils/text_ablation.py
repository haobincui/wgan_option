"""Deterministic text controls for news-first volatility experiments.

The helpers in this module deliberately operate on stable sample keys rather
than dataframe positions.  Consequently a workbook row reorder cannot change
the shuffled-text donor assigned to a sample.  Shuffling is always scoped by
an explicit split namespace, so text is never borrowed across train,
validation, core-test, or broad-test boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np


REAL_TEXT = "real_text"
CURRENT_ONLY = "current_only"
TEXT_SHUFFLE = "text_shuffle"
TEXT_ABLATION_MODES = (REAL_TEXT, CURRENT_ONLY, TEXT_SHUFFLE)
TEXT_SHUFFLE_METHOD = "stable_key_pair_breaking_rotation_v1"


def normalize_text_ablation_mode(value: str | None) -> str:
    """Normalize and validate one information-ablation mode."""

    mode = str(value or REAL_TEXT).strip().lower().replace("-", "_")
    aliases = {
        "real": REAL_TEXT,
        "lp": REAL_TEXT,
        "zero": CURRENT_ONLY,
        "zero_text": CURRENT_ONLY,
        "shuffle": TEXT_SHUFFLE,
        "shuffled_text": TEXT_SHUFFLE,
    }
    mode = aliases.get(mode, mode)
    if mode not in TEXT_ABLATION_MODES:
        raise ValueError(
            f"text ablation mode must be one of {list(TEXT_ABLATION_MODES)}, got: {value!r}"
        )
    return mode


def text_information_path(mode: str | None) -> str:
    """Return the explicit model-input path represented by ``mode``."""

    normalized = normalize_text_ablation_mode(mode)
    return {
        REAL_TEXT: "current_surface_plus_real_lp_embedding",
        CURRENT_ONLY: "current_surface_plus_zero_lp_embedding",
        TEXT_SHUFFLE: "current_surface_plus_fixed_shuffled_lp_embedding",
    }[normalized]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalized_unique_keys(stable_keys: Sequence[str]) -> list[str]:
    keys = [str(value).strip() for value in stable_keys]
    if any(not value for value in keys):
        raise ValueError("Fixed text shuffle requires non-empty stable sample keys")
    if len(set(keys)) != len(keys):
        duplicates = sorted({value for value in keys if keys.count(value) > 1})
        raise ValueError(
            "Fixed text shuffle requires unique stable sample keys; duplicates="
            f"{duplicates[:5]}"
        )
    return keys


@dataclass(frozen=True)
class FixedTextShuffle:
    """One row-order-invariant receiver-to-donor permutation."""

    namespace: str
    seed: int
    method: str
    receiver_keys: tuple[str, ...]
    donor_keys: tuple[str, ...]
    mapping_sha256: str
    universe_sha256: str
    same_pair_count: int = 0
    same_session_count: int = 0

    @property
    def fixed_point_count(self) -> int:
        return sum(
            receiver == donor
            for receiver, donor in zip(self.receiver_keys, self.donor_keys)
        )

    def donor_by_receiver(self) -> dict[str, str]:
        return dict(zip(self.receiver_keys, self.donor_keys))


def fixed_text_shuffle(
    stable_keys: Sequence[str],
    *,
    seed: int,
    namespace: str,
    pair_ids: Sequence[str] | None = None,
    session_ids: Sequence[str] | None = None,
) -> FixedTextShuffle:
    """Build a deterministic, pair-breaking donor permutation for one split.

    Keys are sorted before the RNG is seeded.  A seeded donor permutation is
    cyclically rotated until it has neither fixed samples nor same-pair
    assignments.  Donors from the same CME session are allowed intentionally
    because preserving intraday-regime composition is useful for this null;
    their count is nevertheless exported for audit.
    """

    original = _normalized_unique_keys(stable_keys)
    if len(original) < 2:
        raise ValueError("Fixed text shuffle requires at least two samples per split")
    normalized_namespace = str(namespace).strip()
    if not normalized_namespace:
        raise ValueError("Fixed text shuffle namespace must be non-empty")

    def keyed(values: Sequence[str] | None, *, label: str) -> dict[str, str] | None:
        if values is None:
            return None
        if len(values) != len(original):
            raise ValueError(f"{label} must contain one value per stable key")
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError(f"{label} values must be non-empty")
        return dict(zip(original, normalized))

    pair_by_key = keyed(pair_ids, label="pair_ids")
    session_by_key = keyed(session_ids, label="session_ids")

    receivers = sorted(original)
    universe_sha256 = hashlib.sha256(
        _canonical_json(receivers).encode("utf-8")
    ).hexdigest()
    seed_payload = {
        "method": TEXT_SHUFFLE_METHOD,
        "namespace": normalized_namespace,
        "seed": int(seed),
        "universe_sha256": universe_sha256,
        "pair_constraints_sha256": (
            hashlib.sha256(
                _canonical_json(sorted(pair_by_key.items())).encode("utf-8")
            ).hexdigest()
            if pair_by_key is not None
            else ""
        ),
    }
    derived_seed = int.from_bytes(
        hashlib.sha256(_canonical_json(seed_payload).encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    rng = np.random.default_rng(derived_seed)
    randomized = [receivers[int(index)] for index in rng.permutation(len(receivers))]
    donors: list[str] | None = None
    for offset in range(len(randomized)):
        candidate = randomized[offset:] + randomized[:offset]
        if any(receiver == donor for receiver, donor in zip(receivers, candidate)):
            continue
        if pair_by_key is not None and any(
            pair_by_key[receiver] == pair_by_key[donor]
            for receiver, donor in zip(receivers, candidate)
        ):
            continue
        donors = candidate
        break
    if donors is None:
        raise ValueError(
            "No fixed split-internal text-shuffle permutation can avoid same-pair donors; "
            f"namespace={normalized_namespace!r}, samples={len(receivers)}"
        )
    same_pair_count = (
        sum(
            pair_by_key[receiver] == pair_by_key[donor]
            for receiver, donor in zip(receivers, donors)
        )
        if pair_by_key is not None
        else 0
    )
    same_session_count = (
        sum(
            session_by_key[receiver] == session_by_key[donor]
            for receiver, donor in zip(receivers, donors)
        )
        if session_by_key is not None
        else 0
    )
    mapping_payload = {
        **seed_payload,
        "assignments": list(zip(receivers, donors)),
    }
    mapping_sha256 = hashlib.sha256(
        _canonical_json(mapping_payload).encode("utf-8")
    ).hexdigest()
    result = FixedTextShuffle(
        namespace=normalized_namespace,
        seed=int(seed),
        method=TEXT_SHUFFLE_METHOD,
        receiver_keys=tuple(receivers),
        donor_keys=tuple(donors),
        mapping_sha256=mapping_sha256,
        universe_sha256=universe_sha256,
        same_pair_count=int(same_pair_count),
        same_session_count=int(same_session_count),
    )
    if result.fixed_point_count or result.same_pair_count:
        raise RuntimeError("Fixed text shuffle violated sample/pair donor constraints")
    return result


def transform_embedding_matrix(
    embeddings: np.ndarray,
    stable_keys: Sequence[str],
    *,
    mode: str,
    seed: int,
    namespace: str,
    pair_ids: Sequence[str] | None = None,
    session_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply real, zero-text, or fixed-shuffle treatment to an embedding matrix."""

    normalized = normalize_text_ablation_mode(mode)
    matrix = np.asarray(embeddings, dtype=np.float32)
    keys = _normalized_unique_keys(stable_keys)
    if matrix.ndim != 2 or int(matrix.shape[0]) != len(keys):
        raise ValueError(
            "Embedding matrix must be two-dimensional with one row per stable key: "
            f"shape={matrix.shape}, keys={len(keys)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains non-finite values")

    metadata: dict[str, object] = {
        "text_ablation_mode": normalized,
        "text_information_path": text_information_path(normalized),
        "text_shuffle_seed": int(seed),
        "text_shuffle_namespace": str(namespace),
        "text_shuffle_method": TEXT_SHUFFLE_METHOD if normalized == TEXT_SHUFFLE else "none",
        "text_shuffle_mapping_sha256": "",
        "text_shuffle_universe_sha256": hashlib.sha256(
            _canonical_json(sorted(keys)).encode("utf-8")
        ).hexdigest(),
        "text_shuffle_fixed_point_count": 0,
        "text_shuffle_same_pair_count": 0,
        "text_shuffle_same_session_count": 0,
        "embedding_dim": int(matrix.shape[1]),
    }
    if normalized == REAL_TEXT:
        return matrix.copy(), metadata
    if normalized == CURRENT_ONLY:
        output = np.zeros_like(matrix, dtype=np.float32)
        metadata["zero_text_verified"] = bool(np.count_nonzero(output) == 0)
        return output, metadata

    mapping = fixed_text_shuffle(
        keys,
        seed=int(seed),
        namespace=namespace,
        pair_ids=pair_ids,
        session_ids=session_ids,
    )
    original_position = {key: index for index, key in enumerate(keys)}
    donor_by_receiver = mapping.donor_by_receiver()
    donor_positions = [original_position[donor_by_receiver[key]] for key in keys]
    output = matrix[np.asarray(donor_positions, dtype=np.int64)].copy()
    metadata.update(
        {
            "text_shuffle_mapping_sha256": mapping.mapping_sha256,
            "text_shuffle_universe_sha256": mapping.universe_sha256,
            "text_shuffle_fixed_point_count": mapping.fixed_point_count,
            "text_shuffle_same_pair_count": mapping.same_pair_count,
            "text_shuffle_same_session_count": mapping.same_session_count,
            "text_shuffle_donor_by_receiver": donor_by_receiver,
        }
    )
    return output, metadata


__all__ = [
    "CURRENT_ONLY",
    "FixedTextShuffle",
    "REAL_TEXT",
    "TEXT_ABLATION_MODES",
    "TEXT_SHUFFLE",
    "TEXT_SHUFFLE_METHOD",
    "fixed_text_shuffle",
    "normalize_text_ablation_mode",
    "text_information_path",
    "transform_embedding_matrix",
]
