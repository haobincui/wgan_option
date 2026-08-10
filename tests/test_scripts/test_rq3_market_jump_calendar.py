from __future__ import annotations

import csv
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SCHEMA = [
    "event_id",
    "event_family",
    "release_name",
    "release_time_local",
    "release_timezone",
    "release_time_utc",
    "scheduled_or_unscheduled",
    "official_source",
    "source_retrieval_date",
    "calendar_version",
    "priority",
]
EXPECTED_FAMILY_COUNTS = {
    "NFP": 12,
    "CPI": 12,
    "PPI": 12,
    "RETAIL_SALES": 12,
    "GDP_ADVANCE": 4,
    "FOMC": 8,
    "ISM_MANUFACTURING": 12,
    "ISM_SERVICES": 12,
}
EXPECTED_PRIORITIES = {
    "FOMC": 10,
    "NFP": 20,
    "CPI": 30,
    "PPI": 40,
    "GDP_ADVANCE": 50,
    "RETAIL_SALES": 60,
    "ISM_MANUFACTURING": 70,
    "ISM_SERVICES": 80,
}
EXPECTED_FOMC_DATES = {
    2022: {
        "2022-01-26",
        "2022-03-16",
        "2022-05-04",
        "2022-06-15",
        "2022-07-27",
        "2022-09-21",
        "2022-11-02",
        "2022-12-14",
    },
    2023: {
        "2023-02-01",
        "2023-03-22",
        "2023-05-03",
        "2023-06-14",
        "2023-07-26",
        "2023-09-20",
        "2023-11-01",
        "2023-12-13",
    },
}
EXPECTED_GDP_ADVANCE_DATES = {
    2022: {
        "2022-01-27",
        "2022-04-28",
        "2022-07-28",
        "2022-10-27",
    },
    2023: {
        "2023-01-26",
        "2023-04-27",
        "2023-07-27",
        "2023-10-26",
    },
}


def _read_calendar(year: int) -> tuple[list[str], list[dict[str, str]]]:
    path = ROOT / f"data/reference/rq3_scheduled_macro_events_{year}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


class TestRQ3MarketJumpCalendar(unittest.TestCase):
    def test_frozen_calendars_have_expected_schema_counts_and_keys(self):
        all_event_ids: list[str] = []
        for year in (2022, 2023):
            schema, rows = _read_calendar(year)
            self.assertEqual(schema, EXPECTED_SCHEMA)
            self.assertEqual(len(rows), 84)
            self.assertEqual(
                Counter(row["event_family"] for row in rows),
                Counter(EXPECTED_FAMILY_COUNTS),
            )

            event_ids = [row["event_id"] for row in rows]
            family_release_keys = [
                (row["event_family"], row["release_time_utc"])
                for row in rows
            ]
            self.assertEqual(len(event_ids), len(set(event_ids)))
            self.assertEqual(
                len(family_release_keys),
                len(set(family_release_keys)),
            )
            self.assertEqual(
                {row["scheduled_or_unscheduled"] for row in rows},
                {"scheduled"},
            )
            self.assertEqual(
                {
                    row["event_family"]: int(row["priority"])
                    for row in rows
                },
                EXPECTED_PRIORITIES,
            )
            all_event_ids.extend(event_ids)

        self.assertEqual(len(all_event_ids), len(set(all_event_ids)))

    def test_local_offsets_and_utc_follow_new_york_zoneinfo(self):
        observed_offsets = set()
        for year in (2022, 2023):
            _, rows = _read_calendar(year)
            for row in rows:
                self.assertEqual(row["release_timezone"], "America/New_York")
                encoded_local = datetime.fromisoformat(
                    row["release_time_local"]
                )
                zone_local = encoded_local.replace(tzinfo=None).replace(
                    tzinfo=ZoneInfo(row["release_timezone"])
                )
                self.assertEqual(encoded_local.utcoffset(), zone_local.utcoffset())
                observed_offsets.add(encoded_local.utcoffset())

                expected_utc = (
                    zone_local.astimezone(timezone.utc)
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z")
                )
                self.assertEqual(row["release_time_utc"], expected_utc)

        self.assertEqual(
            observed_offsets,
            {
                datetime.fromisoformat("2022-01-01T00:00:00-05:00").utcoffset(),
                datetime.fromisoformat("2022-07-01T00:00:00-04:00").utcoffset(),
            },
        )

    def test_fomc_rows_are_policy_statements_at_decision_time(self):
        for year in (2022, 2023):
            _, rows = _read_calendar(year)
            fomc = [row for row in rows if row["event_family"] == "FOMC"]
            self.assertEqual(
                {row["release_time_local"][:10] for row in fomc},
                EXPECTED_FOMC_DATES[year],
            )
            self.assertEqual(
                {row["release_name"] for row in fomc},
                {"FOMC Policy Statement"},
            )
            self.assertTrue(
                all("T14:00:00" in row["release_time_local"] for row in fomc)
            )
            self.assertTrue(
                all(
                    urlparse(row["official_source"]).hostname
                    == "www.federalreserve.gov"
                    for row in fomc
                )
            )

    def test_gdp_rows_are_advance_estimates_only(self):
        for year in (2022, 2023):
            _, rows = _read_calendar(year)
            gdp = [
                row for row in rows if row["event_family"] == "GDP_ADVANCE"
            ]
            self.assertEqual(
                {row["release_time_local"][:10] for row in gdp},
                EXPECTED_GDP_ADVANCE_DATES[year],
            )
            self.assertEqual(
                {row["release_name"] for row in gdp},
                {"GDP Advance Estimate"},
            )
            self.assertTrue(
                all("T08:30:00" in row["release_time_local"] for row in gdp)
            )

    def test_2022_calendar_has_frozen_official_provenance(self):
        _, rows = _read_calendar(2022)
        expected_hosts = {
            "NFP": "www.bls.gov",
            "CPI": "www.bls.gov",
            "PPI": "www.bls.gov",
            "RETAIL_SALES": "www.census.gov",
            "GDP_ADVANCE": "www.bea.gov",
            "FOMC": "www.federalreserve.gov",
            "ISM_MANUFACTURING": "www.ismworld.org",
            "ISM_SERVICES": "www.ismworld.org",
        }
        self.assertEqual(
            {row["source_retrieval_date"] for row in rows},
            {"2026-08-10"},
        )
        self.assertEqual(
            {row["calendar_version"] for row in rows},
            {"2022_v1"},
        )
        for row in rows:
            self.assertEqual(
                urlparse(row["official_source"]).hostname,
                expected_hosts[row["event_family"]],
            )

    def test_same_day_releases_at_different_times_are_preserved(self):
        for year, release_date in ((2022, "2022-06-15"), (2023, "2023-06-14")):
            _, rows = _read_calendar(year)
            same_day = [
                row
                for row in rows
                if row["release_time_local"].startswith(release_date)
            ]
            self.assertGreaterEqual(len(same_day), 2)
            self.assertGreater(
                len({row["release_time_local"] for row in same_day}),
                1,
            )


if __name__ == "__main__":
    unittest.main()
