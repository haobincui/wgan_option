from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wgan_option.merge_support import (
    DEFAULT_SOURCE_TIMEZONE,
    resolve_news_source_timezone,
)
from wgan_option.news_time import parse_news_timestamps


class TestFactivaNewsTime(unittest.TestCase):
    def test_default_timezone_is_europe_london(self):
        self.assertEqual(DEFAULT_SOURCE_TIMEZONE, "Europe/London")

    def test_london_winter_and_summer_offsets(self):
        parsed = parse_news_timestamps(
            pd.Series(["30 December 2023", "1 July 2023"]),
            pd.Series(["07:35", "08:30"]),
            source_timezone="Europe/London",
        )
        self.assertEqual(
            parsed.timestamp_utc.tolist(),
            ["2023-12-30T07:35:00Z", "2023-07-01T07:30:00Z"],
        )
        self.assertEqual(parsed.utc_offset_minutes.tolist(), [0.0, 60.0])
        self.assertEqual(parsed.parse_status.tolist(), ["ok", "ok"])

    def test_london_dst_nonexistent_time_is_audited(self):
        parsed = parse_news_timestamps(
            pd.Series(["26 March 2023"]),
            pd.Series(["01:30"]),
            source_timezone="Europe/London",
        )
        self.assertEqual(
            parsed.parse_status.tolist(),
            ["dst_ambiguous_or_nonexistent"],
        )
        self.assertIsNone(parsed.timestamp_utc.iloc[0])

    def test_merge_rejects_timezone_mismatch_with_surface_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "surface-resolved_config.yaml").write_text(
                "surface_builder:\n"
                "  generate_surface:\n"
                "    source_timezone: Europe/London\n",
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_news_source_timezone(root, None),
                "Europe/London",
            )
            with self.assertRaisesRegex(ValueError, "timezone mismatch"):
                resolve_news_source_timezone(root, "America/New_York")


if __name__ == "__main__":
    unittest.main()
