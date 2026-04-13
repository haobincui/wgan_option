import csv
import gzip
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import build_ty_plus_trade_tables as build_ty_plus_trade_tables  # noqa: E402


def _write_gzip_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)


class TestBuildTyPlusTradeTables(unittest.TestCase):
    def test_run_pipeline_filters_invalid_rows_and_drops_embedding_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "0#TY+"
            merged_output = root / "ty_plus_merged.csv.gz"
            minute_counts_output = root / "ty_plus_minute_trade_counts.csv"
            news_xlsx = root / "news.xlsx"
            news_output = root / "news_with_trade_counts.xlsx"

            _write_gzip_csv(
                input_dir / "source.csv.gz",
                [
                    build_ty_plus_trade_tables.EXPECTED_HEADER,
                    ["TYA", "", "Market Price", "2022-01-01T00:00:01Z", "Trade", "1.0", "1"],
                    ["TYB", "", "Market Price", "2022-01-01T00:00:10Z", "Trade", "0", "1"],
                    ["TYC", "", "Market Price", "2022-01-01T00:00:20Z", "Trade", "-1.0", "1"],
                    ["TYD", "", "Market Price", "2022-01-01T00:00:30Z", "Trade", "2.0", "0"],
                    ["TYE", "", "Market Price", "2022-01-01T00:00:40Z", "Trade", "2.0", "-3"],
                    ["TYF", "", "Market Price", "2022-01-01T00:05:00Z", "Trade", "", "1"],
                    ["TYG", "", "Market Price", "2022-01-01T00:05:01Z", "Trade", "3.0", "oops"],
                    ["TYC", "", "Market Price", "2022-01-01T00:05:05Z", "Trade", "3.0", "1"],
                ],
            )

            pd.DataFrame(
                [
                    {
                        "SourceFile": "a.txt",
                        "ArticleID": 1,
                        "PD": "2022-01-01",
                        "ET": "00:00:00",
                        "HD": "headline-a",
                        "LP": "lead-a",
                        "HD_embedding": "[0.1, 0.2]",
                        "LP_embedding": "[0.3, 0.4]",
                        "HD_dim": 2,
                        "LP_dim": 2,
                    },
                    {
                        "SourceFile": "b.txt",
                        "ArticleID": 2,
                        "PD": "2022-01-01",
                        "ET": "00:05:00",
                        "HD": "headline-b",
                        "LP": "lead-b",
                        "HD_embedding": "[0.5, 0.6]",
                        "LP_embedding": "[0.7, 0.8]",
                        "HD_dim": 2,
                        "LP_dim": 2,
                    },
                ]
            ).to_excel(news_xlsx, index=False, engine="openpyxl")

            stats = build_ty_plus_trade_tables.run_pipeline(
                input_dir=input_dir,
                merged_output=merged_output,
                minute_counts_output=minute_counts_output,
                news_xlsx=news_xlsx,
                news_output=news_output,
                source_timezone="UTC",
                offset_minutes=5,
                sheet_name="Sheet1",
                date_column="PD",
                time_column="ET",
                fail_on_invalid=False,
            )

            self.assertEqual(stats.merge_stats.files_merged, 1)
            self.assertEqual(stats.merge_stats.rows_written, 8)
            self.assertEqual(stats.count_summary.counted_rows, 2)
            self.assertEqual(stats.count_summary.filtered_non_positive_price_rows, 2)
            self.assertEqual(stats.count_summary.filtered_non_positive_volume_rows, 2)
            self.assertEqual(stats.count_summary.filtered_invalid_numeric_rows, 2)
            self.assertEqual(stats.minute_count_rows, 2)
            self.assertEqual(stats.news_rows, 2)

            minute_df = pd.read_csv(minute_counts_output)
            self.assertEqual(
                minute_df.to_dict(orient="records"),
                [
                    {"timestamp_utc": "2022-01-01T00:00:00Z", "trade_count": 1},
                    {"timestamp_utc": "2022-01-01T00:05:00Z", "trade_count": 1},
                ],
            )

            news_df = pd.read_excel(
                news_output,
                sheet_name="news_with_trade_counts",
                engine="openpyxl",
            )
            self.assertEqual(
                news_df[
                    [
                        "timestamp_utc",
                        "trade_count_at_timestamp_utc",
                        "timestamp_utc_plus_5m",
                        "trade_count_at_timestamp_utc_plus_5m",
                    ]
                ].to_dict(orient="records"),
                [
                    {
                        "timestamp_utc": "2022-01-01T00:00:00Z",
                        "trade_count_at_timestamp_utc": 1,
                        "timestamp_utc_plus_5m": "2022-01-01T00:05:00Z",
                        "trade_count_at_timestamp_utc_plus_5m": 1,
                    },
                    {
                        "timestamp_utc": "2022-01-01T00:05:00Z",
                        "trade_count_at_timestamp_utc": 1,
                        "timestamp_utc_plus_5m": "2022-01-01T00:10:00Z",
                        "trade_count_at_timestamp_utc_plus_5m": 0,
                    },
                ],
            )
            self.assertNotIn("HD_embedding", news_df.columns)
            self.assertNotIn("LP_embedding", news_df.columns)
            self.assertIn("HD", news_df.columns)
            self.assertIn("LP", news_df.columns)
            self.assertIn("HD_dim", news_df.columns)
            self.assertIn("LP_dim", news_df.columns)

    def test_run_pipeline_skips_invalid_source_files_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "0#TY+"
            merged_output = root / "ty_plus_merged.csv.gz"
            minute_counts_output = root / "ty_plus_minute_trade_counts.csv"
            news_xlsx = root / "news.xlsx"
            news_output = root / "news_with_trade_counts.xlsx"

            _write_gzip_csv(
                input_dir / "good.csv.gz",
                [
                    build_ty_plus_trade_tables.EXPECTED_HEADER,
                    ["TYA", "", "Market Price", "2022-01-01T00:00:01Z", "Trade", "1.0", "1"],
                ],
            )
            bad_path = input_dir / "bad.csv.gz"
            bad_path.write_text('{"error":"not gzip"}', encoding="utf-8")

            pd.DataFrame(
                [
                    {"PD": "2022-01-01", "ET": "00:00:00", "HD": "headline-a"},
                ]
            ).to_excel(news_xlsx, index=False, engine="openpyxl")

            stats = build_ty_plus_trade_tables.run_pipeline(
                input_dir=input_dir,
                merged_output=merged_output,
                minute_counts_output=minute_counts_output,
                news_xlsx=news_xlsx,
                news_output=news_output,
                source_timezone="UTC",
                offset_minutes=5,
                sheet_name="Sheet1",
                date_column="PD",
                time_column="ET",
                fail_on_invalid=False,
            )

            self.assertEqual(stats.merge_stats.files_merged, 1)
            self.assertEqual(stats.merge_stats.skipped_files, (bad_path,))
            self.assertEqual(stats.count_summary.counted_rows, 1)
            minute_df = pd.read_csv(minute_counts_output)
            self.assertEqual(
                minute_df.to_dict(orient="records"),
                [{"timestamp_utc": "2022-01-01T00:00:00Z", "trade_count": 1}],
            )


if __name__ == "__main__":
    unittest.main()
