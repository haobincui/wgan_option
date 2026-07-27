"""RQ3 scheduled-news robustness CLI with legacy diagnostic compatibility."""

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
from scripts.rq3.news_quiet import (  # noqa: E402
    analyze_news_quiet_results,
    analyze_news_quiet_workbook,
    build_news_quiet_workbook_from_window,
    build_news_quiet_workbook,
    prepare_news_quiet_targets,
)
from scripts.rq3.scheduled_news_regime import (  # noqa: E402
    run_scheduled_news_regime,
)
from wgan_option.merge_support import DEFAULT_SOURCE_TIMEZONE  # noqa: E402


def _default_output_dir(prefix: str) -> str:
    return str(Path("outputs/rq3") / f"{prefix}_{timestamp_string()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RQ3 scheduled-news conditional predictive robustness. Older "
            "event/quiet commands are retained only for audit compatibility."
        )
    )
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

    build_news_quiet = subparsers.add_parser(
        "build-news-quiet-workbook",
        help="Build a news-event vs no-news quiet evaluation workbook.",
    )
    build_news_quiet.add_argument("--source-merged-vol", required=True, help="Source merged_vol_rq2_text.xlsx path.")
    build_news_quiet.add_argument("--surface-all-json", required=True, help="surface-*-all.json path.")
    build_news_quiet.add_argument("--news-xlsx", required=True, help="Raw news workbook used to define news buffers.")
    build_news_quiet.add_argument(
        "--source-timezone",
        default=DEFAULT_SOURCE_TIMEZONE,
        help="Timezone for Factiva PD/ET publication timestamps.",
    )
    build_news_quiet.add_argument("--output-workbook", required=True, help="Output news/quiet workbook path.")
    build_news_quiet.add_argument("--sheet-name", default=GAN_SHEET, help=f"Source sheet name (default: {GAN_SHEET}).")
    build_news_quiet.add_argument("--horizon-minutes", type=int, default=5, help="Target horizon in minutes.")
    build_news_quiet.add_argument("--quiet-grid-minutes", type=int, default=5, help="Quiet candidate grid size.")
    build_news_quiet.add_argument(
        "--quiet-buffer-minutes",
        type=int,
        default=60,
        help="Minimum distance from any news timestamp for quiet samples.",
    )

    prepare_news_quiet = subparsers.add_parser(
        "prepare-news-quiet-targets",
        help="Prepare reproducible no-news quiet target timestamps for the fast RQ3 workflow.",
    )
    prepare_news_quiet.add_argument("--source-merged-vol", required=True, help="Source merged_vol_rq2_text.xlsx path.")
    prepare_news_quiet.add_argument("--news-xlsx", required=True, help="Raw news workbook used to define news buffers.")
    prepare_news_quiet.add_argument(
        "--source-timezone",
        default=DEFAULT_SOURCE_TIMEZONE,
        help="Timezone for Factiva PD/ET publication timestamps.",
    )
    prepare_news_quiet.add_argument("--output-dir", required=True, help="Output directory for quiet_targets.txt and audit CSV.")
    prepare_news_quiet.add_argument("--sheet-name", default=GAN_SHEET, help=f"Source sheet name (default: {GAN_SHEET}).")
    prepare_news_quiet.add_argument("--horizon-minutes", type=int, default=5, help="Target horizon in minutes.")
    prepare_news_quiet.add_argument("--quiet-grid-minutes", type=int, default=5, help="Quiet candidate grid size.")
    prepare_news_quiet.add_argument("--quiet-buffer-minutes", type=int, default=60, help="Minimum distance from any news timestamp.")
    prepare_news_quiet.add_argument("--candidate-count", type=int, default=12000, help="Number of eligible quiet targets to sample.")
    prepare_news_quiet.add_argument("--sample-seed", type=int, default=20260625, help="Random seed for target sampling.")

    build_news_quiet_window = subparsers.add_parser(
        "build-news-quiet-workbook-from-window",
        help="Build a news-event vs quiet workbook from window surface JSON.",
    )
    build_news_quiet_window.add_argument("--source-merged-vol", required=True, help="Source merged_vol_rq2_text.xlsx path.")
    build_news_quiet_window.add_argument("--window-surface-json", required=True, help="surface-*-window.json path.")
    build_news_quiet_window.add_argument("--output-workbook", required=True, help="Output news/quiet workbook path.")
    build_news_quiet_window.add_argument("--sheet-name", default=GAN_SHEET, help=f"Source sheet name (default: {GAN_SHEET}).")
    build_news_quiet_window.add_argument("--horizon-minutes", type=int, default=5, help="Target horizon in minutes.")
    build_news_quiet_window.add_argument("--quiet-grid-minutes", type=int, default=5, help="Quiet candidate grid size.")
    build_news_quiet_window.add_argument("--quiet-buffer-minutes", type=int, default=60, help="Minimum distance from any news timestamp.")
    build_news_quiet_window.add_argument("--quiet-max-samples", type=int, default=3711, help="Final quiet rows to keep.")
    build_news_quiet_window.add_argument("--quiet-sample-seed", type=int, default=20260625, help="Random seed for final quiet row sampling.")

    news_quiet_workbook = subparsers.add_parser(
        "news-quiet-workbook",
        help="Analyze current-to-target IVS jumps in a news-event vs quiet workbook.",
    )
    news_quiet_workbook.add_argument("--workbook", required=True, help="Input news/quiet workbook path.")
    news_quiet_workbook.add_argument("--output-dir", default="", help="Output directory.")
    news_quiet_workbook.add_argument("--sheet-name", default=GAN_SHEET, help=f"Workbook sheet name (default: {GAN_SHEET}).")
    news_quiet_workbook.add_argument("--split", choices=["train", "val", "all"], default="all")
    news_quiet_workbook.add_argument("--train-ratio", type=float, default=0.8)

    news_quiet_result = subparsers.add_parser(
        "news-quiet-result",
        help="Analyze generate-result summaries with news_event/quiet metadata.",
    )
    news_quiet_result.add_argument("--output-dir", default="", help="Output directory.")
    news_quiet_result.add_argument("--text-label", default="", help="Model label for LP text result.")
    news_quiet_result.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Result summary CSV with a model label. Repeat for multiple models.",
    )

    scheduled_news = subparsers.add_parser(
        "scheduled-news-regime",
        help=(
            "Run frozen-model all-OOS scheduled-news conditional robustness "
            "with matched ordinary-news analysis as a secondary diagnostic."
        ),
    )
    scheduled_news.add_argument(
        "--config",
        required=True,
        help="RQ3 scheduled-news YAML config.",
    )
    scheduled_news.add_argument(
        "--output-dir",
        default="",
        help=(
            "Output archive directory. Defaults to "
            "outputs/experiments/rq3_scheduled_news_regime_raw_vol_<timestamp>."
        ),
    )
    scheduled_news.add_argument(
        "--rq1-experiment",
        default="",
        help="Override the RQ1 frozen-prediction experiment path.",
    )
    scheduled_news.add_argument(
        "--rq2-experiment",
        default="",
        help="Override the RQ2 frozen-prediction experiment path.",
    )
    scheduled_news.add_argument(
        "--event-calendar",
        default="",
        help="Override the frozen scheduled-event calendar path.",
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
    if args.command == "build-news-quiet-workbook":
        output = build_news_quiet_workbook(
            source_merged_vol=args.source_merged_vol,
            surface_all_json=args.surface_all_json,
            news_xlsx=args.news_xlsx,
            output_workbook=args.output_workbook,
            horizon_minutes=int(args.horizon_minutes),
            quiet_grid_minutes=int(args.quiet_grid_minutes),
            quiet_buffer_minutes=int(args.quiet_buffer_minutes),
            sheet_name=args.sheet_name,
            source_timezone=args.source_timezone,
        )
        print(f"RQ3 news/quiet workbook written to {output}")
        return output
    if args.command == "prepare-news-quiet-targets":
        output = prepare_news_quiet_targets(
            source_merged_vol=args.source_merged_vol,
            news_xlsx=args.news_xlsx,
            output_dir=args.output_dir,
            horizon_minutes=int(args.horizon_minutes),
            quiet_grid_minutes=int(args.quiet_grid_minutes),
            quiet_buffer_minutes=int(args.quiet_buffer_minutes),
            candidate_count=int(args.candidate_count),
            sample_seed=int(args.sample_seed),
            sheet_name=args.sheet_name,
            source_timezone=args.source_timezone,
        )
        print(f"RQ3 quiet target files written to {output}")
        return output
    if args.command == "build-news-quiet-workbook-from-window":
        output = build_news_quiet_workbook_from_window(
            source_merged_vol=args.source_merged_vol,
            window_surface_json=args.window_surface_json,
            output_workbook=args.output_workbook,
            horizon_minutes=int(args.horizon_minutes),
            quiet_grid_minutes=int(args.quiet_grid_minutes),
            quiet_buffer_minutes=int(args.quiet_buffer_minutes),
            quiet_max_samples=int(args.quiet_max_samples),
            quiet_sample_seed=int(args.quiet_sample_seed),
            sheet_name=args.sheet_name,
        )
        print(f"RQ3 news/quiet workbook written to {output}")
        return output
    if args.command == "news-quiet-workbook":
        output_dir = args.output_dir or _default_output_dir("rq3_news_quiet")
        output = analyze_news_quiet_workbook(
            workbook_path=args.workbook,
            output_dir=output_dir,
            sheet_name=args.sheet_name,
            split=args.split,
            train_ratio=float(args.train_ratio),
        )
        print(f"RQ3 news/quiet workbook analysis written to {output}")
        return output
    if args.command == "news-quiet-result":
        output_dir = args.output_dir or _default_output_dir("rq3_news_quiet_results")
        output = analyze_news_quiet_results(
            result_specs=args.result,
            output_dir=output_dir,
            text_label=args.text_label or None,
        )
        print(f"RQ3 news/quiet result analysis written to {output}")
        return output
    if args.command == "scheduled-news-regime":
        output = run_scheduled_news_regime(
            args.config,
            output_dir=args.output_dir or None,
            rq1_experiment_override=args.rq1_experiment or None,
            rq2_experiment_override=args.rq2_experiment or None,
            event_calendar_override=args.event_calendar or None,
        )
        print(f"RQ3 scheduled-news regime archive written to {output}")
        return output
    raise ValueError(f"Unknown RQ3 command: {args.command}")


if __name__ == "__main__":
    main()
