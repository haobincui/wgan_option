from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.protocol import (
    CHECKPOINT_SCHEMA_VERSION_V3,
    MATCHING_NEGATIVE_SOURCE_PLAN_VERSION,
    TRAINING_PROTOCOL_VERSION_V3,
    canonical_json_bytes,
    canonical_payload_sha256,
)


class TestFilmWGANProtocolV3(unittest.TestCase):
    def test_protocol_versions_define_the_incompatible_v3_boundary(self) -> None:
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION_V3, 6)
        self.assertEqual(
            TRAINING_PROTOCOL_VERSION_V3,
            "film_wgan_transition_matching_symmetric_negative_v3",
        )
        self.assertEqual(
            MATCHING_NEGATIVE_SOURCE_PLAN_VERSION,
            "paired_text_source_v3",
        )

    def test_canonical_hash_ignores_mapping_and_set_order(self) -> None:
        first = {"b": {"y", "x"}, "a": [Path("fold/plan.csv"), 2]}
        second = {"a": ["fold/plan.csv", 2], "b": {"x", "y"}}
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(
            canonical_payload_sha256(first),
            canonical_payload_sha256(second),
        )

    def test_canonical_hash_supports_dataclasses_and_rejects_nonfinite_json(self) -> None:
        @dataclass(frozen=True)
        class Payload:
            fold: str
            seed: int

        self.assertEqual(
            canonical_json_bytes(Payload("2023Q1", 42)),
            b'{"fold":"2023Q1","seed":42}',
        )
        with self.assertRaises(ValueError):
            canonical_json_bytes({"bad": math.nan})


if __name__ == "__main__":
    unittest.main()
