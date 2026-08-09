"""Versioned protocol constants and canonical hashing for FiLM-WGAN studies.

The transition-matching v3 protocol is intentionally incompatible with the
schema-5 pilot.  The critic parameter layout is unchanged, but the persisted
text-alignment and negative-source semantics are different and must never be
silently reused across experiment roots.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_SCHEMA_VERSION_V3 = 6
TRAINING_PROTOCOL_VERSION_V3 = (
    "film_wgan_transition_matching_symmetric_negative_v3"
)
CRITIC_ARCHITECTURE_VERSION_TRANSITION_MATCHING = "transition_matching_v2"
MATCHING_NEGATIVE_SOURCE_PLAN_VERSION = "paired_text_source_v3"
TEXT_ALIGNMENT_PLAN_VERSION = "paired_text_alignment_v3"
DIAGNOSTICS_SCHEMA_VERSION = 1
EXPERIMENT_DESIGN_SCHEMA_VERSION = 2
EPOCH_POLICY_CONTINUATION_ANCHOR = "continuation_anchor_v1"


def _canonical_value(value: Any) -> Any:
    """Convert supported research metadata into deterministic JSON values."""

    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonical_value(item) for item in value), key=repr)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _canonical_value(value.item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "Unsupported value in canonical protocol payload: "
        f"{type(value).__name__}."
    )


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a payload with stable ordering and strict finite JSON."""

    return json.dumps(
        _canonical_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_payload_sha256(payload: Any) -> str:
    """Return the SHA256 of a canonical JSON representation."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
