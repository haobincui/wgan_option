"""Merge raw option-data gzip CSV files into one inspection CSV."""

from __future__ import annotations

import argparse
import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


EXPECTED_HEADER = [
    "#RIC",
    "Alias Underlying RIC",
    "Domain",
    "Date-Time",
    "Type",
    "Price",
    "Volume",
]

DEFAULT_INPUT_DIR = Path("data/raw/option_data/0#TY+")
DEFAULT_OUTPUT_PATH = Path("data/raw/option_data/ty_plus_merged.csv.gz")


@dataclass(frozen=True)
class MergeStats:
    files_merged: int
    rows_written: int
    skipped_files: tuple[Path, ...]
    output_path: Path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge raw option-data *.csv.gz files into one gzip-compressed inspection CSV.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing source *.csv.gz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output gzip CSV path.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip unreadable or schema-mismatched source files instead of failing fast.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _discover_input_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_dir}")

    files = sorted(path for path in input_dir.rglob("*.csv.gz") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No *.csv.gz files found under: {input_dir}")
    return files


def _validate_header(header: Sequence[str], source_path: Path) -> None:
    if list(header) != EXPECTED_HEADER:
        raise ValueError(
            "Header mismatch in "
            f"{source_path}: expected {EXPECTED_HEADER}, got {list(header)}"
        )


def merge_raw_option_data(
    input_dir: Path,
    output_path: Path,
    *,
    skip_invalid: bool = False,
) -> MergeStats:
    files = _discover_input_files(Path(input_dir))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    skipped_files: list[Path] = []
    files_merged = 0

    try:
        with gzip.open(output_path, "wt", encoding="utf-8", newline="") as out_handle:
            writer = csv.writer(out_handle)
            writer.writerow(EXPECTED_HEADER)

            for source_path in files:
                try:
                    with gzip.open(source_path, "rt", encoding="utf-8", newline="") as in_handle:
                        reader = csv.reader(in_handle)
                        header = next(reader, None)
                        if header is None:
                            raise ValueError(f"Input file has no header row: {source_path}")
                        _validate_header(header, source_path)

                        for row in reader:
                            writer.writerow(row)
                            rows_written += 1
                    files_merged += 1
                except (OSError, EOFError, csv.Error, UnicodeDecodeError, ValueError) as exc:
                    if skip_invalid:
                        skipped_files.append(source_path)
                        continue
                    raise RuntimeError(f"Failed while merging {source_path}: {exc}") from exc
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    return MergeStats(
        files_merged=files_merged,
        rows_written=rows_written,
        skipped_files=tuple(skipped_files),
        output_path=output_path,
    )


def main(argv: Iterable[str] | None = None) -> MergeStats:
    args = _parse_args(argv)
    stats = merge_raw_option_data(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output),
        skip_invalid=bool(args.skip_invalid),
    )
    print(f"Merged {stats.files_merged} files into {stats.output_path}")
    print(f"Rows written: {stats.rows_written}")
    if stats.skipped_files:
        print(f"Skipped invalid files: {len(stats.skipped_files)}")
        for path in stats.skipped_files:
            print(f"  {path}")
    return stats


if __name__ == "__main__":
    main()
