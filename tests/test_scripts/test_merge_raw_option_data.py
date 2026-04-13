import csv
import gzip
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.merge_raw_option_data as merge_raw_option_data  # noqa: E402


def _write_gzip_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)


class TestMergeRawOptionData(unittest.TestCase):
    def test_merge_preserves_header_and_row_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "0#TY+"
            output_path = root / "ty_plus_merged.csv.gz"

            _write_gzip_csv(
                input_dir / "b.csv.gz",
                [
                    merge_raw_option_data.EXPECTED_HEADER,
                    ["TYB", "", "Market Price", "2022-01-01T00:00:00Z", "Trade", "1.0", "2"],
                ],
            )
            _write_gzip_csv(
                input_dir / "a.csv.gz",
                [
                    merge_raw_option_data.EXPECTED_HEADER,
                    ["TYA", "", "Market Price", "2022-01-01T00:01:00Z", "Trade", "2.0", "3"],
                ],
            )
            _write_gzip_csv(
                input_dir / "nested" / "c.csv.gz",
                [
                    merge_raw_option_data.EXPECTED_HEADER,
                    ["TYC", "", "Market Price", "2022-01-01T00:02:00Z", "Trade", "", "4"],
                ],
            )

            stats = merge_raw_option_data.merge_raw_option_data(input_dir, output_path)

            self.assertEqual(stats.files_merged, 3)
            self.assertEqual(stats.rows_written, 3)
            self.assertEqual(stats.skipped_files, ())
            self.assertEqual(stats.output_path, output_path)

            with gzip.open(output_path, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))

            self.assertEqual(rows[0], merge_raw_option_data.EXPECTED_HEADER)
            self.assertEqual(
                rows[1:],
                [
                    ["TYA", "", "Market Price", "2022-01-01T00:01:00Z", "Trade", "2.0", "3"],
                    ["TYB", "", "Market Price", "2022-01-01T00:00:00Z", "Trade", "1.0", "2"],
                    ["TYC", "", "Market Price", "2022-01-01T00:02:00Z", "Trade", "", "4"],
                ],
            )

    def test_merge_fails_fast_on_header_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "0#TY+"
            output_path = root / "ty_plus_merged.csv.gz"

            _write_gzip_csv(
                input_dir / "good.csv.gz",
                [
                    merge_raw_option_data.EXPECTED_HEADER,
                    ["TYA", "", "Market Price", "2022-01-01T00:01:00Z", "Trade", "2.0", "3"],
                ],
            )
            _write_gzip_csv(
                input_dir / "bad.csv.gz",
                [
                    ["bad", "header"],
                    ["oops"],
                ],
            )

            with self.assertRaises(RuntimeError):
                merge_raw_option_data.merge_raw_option_data(input_dir, output_path)
            self.assertFalse(output_path.exists())

    def test_merge_can_skip_invalid_inputs_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "0#TY+"
            output_path = root / "ty_plus_merged.csv.gz"

            _write_gzip_csv(
                input_dir / "good.csv.gz",
                [
                    merge_raw_option_data.EXPECTED_HEADER,
                    ["TYA", "", "Market Price", "2022-01-01T00:01:00Z", "Trade", "2.0", "3"],
                ],
            )
            bad_path = input_dir / "bad.csv.gz"
            bad_path.write_text('{"error":"not gzip"}', encoding="utf-8")

            stats = merge_raw_option_data.merge_raw_option_data(
                input_dir,
                output_path,
                skip_invalid=True,
            )

            self.assertEqual(stats.files_merged, 1)
            self.assertEqual(stats.rows_written, 1)
            self.assertEqual(stats.skipped_files, (bad_path,))
            with gzip.open(output_path, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], merge_raw_option_data.EXPECTED_HEADER)
            self.assertEqual(
                rows[1:],
                [["TYA", "", "Market Price", "2022-01-01T00:01:00Z", "Trade", "2.0", "3"]],
            )


if __name__ == "__main__":
    unittest.main()
