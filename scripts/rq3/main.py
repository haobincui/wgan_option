"""CLI for RQ3 event/no-event volatility analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from scripts.rq3.event_study import (  # noqa: E402
    GAN_SHEET,
    analyze_results,
    analyze_workbook,
    timestamp_string,
    write_event_template,
)


def _default_output_dir(prefix: str) -> str:
    return str(Path("outputs/rq3") / f"{prefix}_{timestamp_string()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ3 event/no-event volatility analysis.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("write-event-template", help="Write an empty event calendar CSV template.")
    template.add_argument("--output", required=True, help="Output CSV path for the event template.")

    workbook = subparsers.add_parser("workbook", help="Split merged-vol workbook samples into event/quiet groups.")
    workbook.add_argument("--merged-vol", required=True, help="Input merged_vol.xlsx or merged_vol_rq2_text.xlsx path.")
    workbook.add_argument("--events-csv", required=True, help="Event calendar CSV path.")
    workbook.add_argument("--output-dir", default="", help="Output directory. Defaults to outputs/rq3/rq3_<timestamp>.")
    workbook.add_argument("--sheet-name", default=GAN_SHEET, help=f"Workbook sheet name (default: {GAN_SHEET}).")
    workbook.add_argument("--window-minutes", type=float, default=30.0, help="Symmetric event-window half-width in minutes.")
    workbook.add_argument(
        "--pre-window-minutes",
        type=float,
        default=None,
        help="Asymmetric pre-event window in minutes. Provide with --post-window-minutes.",
    )
    workbook.add_argument(
        "--post-window-minutes",
        type=float,
        default=None,
        help="Asymmetric post-event window in minutes. Vergote-style main RQ3 uses 0 and 10.",
    )
    workbook.add_argument("--split", choices=["train", "val", "all"], default="val", help="Chronological split to analyze.")
    workbook.add_argument("--train-ratio", type=float, default=0.8, help="Chronological train split ratio.")
    workbook.add_argument("--max-case-events", type=int, default=3, help="Maximum event case plots to write.")
    workbook.add_argument("--no-plots", action="store_true", help="Disable event case plot generation.")
    workbook.add_argument(
        "--allow-zero-announcement",
        action="store_true",
        help="Allow quiet-only diagnostic runs when no sample falls inside an event window.",
    )

    result = subparsers.add_parser("result", help="Split generate_result summary CSV rows into event/quiet groups.")
    result.add_argument("--events-csv", required=True, help="Event calendar CSV path.")
    result.add_argument("--output-dir", default="", help="Output directory. Defaults to outputs/rq3/rq3_results_<timestamp>.")
    result.add_argument("--window-minutes", type=float, default=30.0, help="Symmetric event-window half-width in minutes.")
    result.add_argument(
        "--pre-window-minutes",
        type=float,
        default=None,
        help="Asymmetric pre-event window in minutes. Provide with --post-window-minutes.",
    )
    result.add_argument(
        "--post-window-minutes",
        type=float,
        default=None,
        help="Asymmetric post-event window in minutes. Vergote-style main RQ3 uses 0 and 10.",
    )
    result.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Result summary CSV with a model label. Repeat for multiple models.",
    )
    result.add_argument(
        "--allow-zero-announcement",
        action="store_true",
        help="Allow quiet-only diagnostic runs when no sample falls inside an event window.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "write-event-template":
        output = write_event_template(args.output)
        print(f"RQ3 event template written to {output}")
        return output
    if args.command == "workbook":
        output_dir = args.output_dir or _default_output_dir("rq3")
        output = analyze_workbook(
            merged_vol_path=args.merged_vol,
            events_csv=args.events_csv,
            output_dir=output_dir,
            sheet_name=args.sheet_name,
            window_minutes=float(args.window_minutes),
            pre_window_minutes=args.pre_window_minutes,
            post_window_minutes=args.post_window_minutes,
            split=args.split,
            train_ratio=float(args.train_ratio),
            save_plots=not bool(args.no_plots),
            max_case_events=int(args.max_case_events),
            allow_zero_announcement=bool(args.allow_zero_announcement),
        )
        print(f"RQ3 workbook analysis written to {output}")
        return output
    if args.command == "result":
        output_dir = args.output_dir or _default_output_dir("rq3_results")
        output = analyze_results(
            result_specs=args.result,
            events_csv=args.events_csv,
            output_dir=output_dir,
            window_minutes=float(args.window_minutes),
            pre_window_minutes=args.pre_window_minutes,
            post_window_minutes=args.post_window_minutes,
            allow_zero_announcement=bool(args.allow_zero_announcement),
        )
        print(f"RQ3 result analysis written to {output}")
        return output
    raise ValueError(f"Unknown RQ3 command: {args.command}")


if __name__ == "__main__":
    main()
