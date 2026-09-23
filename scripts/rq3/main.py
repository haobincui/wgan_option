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
from scripts.rq3.market_jump_detection import (  # noqa: E402
    run_market_jump_detection,
)
from wgan_option.merge_support import DEFAULT_SOURCE_TIMEZONE  # noqa: E402


def _default_output_dir(prefix: str) -> str:
    return str(Path("outputs/rq3") / f"{prefix}_{timestamp_string()}")


def _default_processed_output_dir(prefix: str) -> str:
    return str(Path("data/processed/rq3") / f"{prefix}_{timestamp_string()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RQ3 scheduled-news conditional predictive robustness. Older "
            "event/quiet commands are retained only for audit compatibility."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser(
        "write-event-template", help="Write an empty event calendar CSV template."
    )
    template.add_argument(
        "--output", required=True, help="Output CSV path for the event template."
    )

    workbook = subparsers.add_parser(
        "workbook", help="Split merged-vol workbook samples into event/quiet groups."
    )
    workbook.add_argument(
        "--merged-vol",
        required=True,
        help="Input merged_vol.xlsx or merged_vol_rq2_text.xlsx path.",
    )
    workbook.add_argument(
        "--events-csv", required=True, help="Event calendar CSV path."
    )
    workbook.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to outputs/rq3/rq3_<timestamp>.",
    )
    workbook.add_argument(
        "--sheet-name",
        default=GAN_SHEET,
        help=f"Workbook sheet name (default: {GAN_SHEET}).",
    )
    workbook.add_argument(
        "--window-minutes",
        type=float,
        default=30.0,
        help="Symmetric event-window half-width in minutes.",
    )
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
    workbook.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="val",
        help="Chronological split to analyze.",
    )
    workbook.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Chronological train split ratio.",
    )
    workbook.add_argument(
        "--max-case-events",
        type=int,
        default=3,
        help="Maximum event case plots to write.",
    )
    workbook.add_argument(
        "--no-plots", action="store_true", help="Disable event case plot generation."
    )
    workbook.add_argument(
        "--allow-zero-announcement",
        action="store_true",
        help="Allow quiet-only diagnostic runs when no sample falls inside an event window.",
    )

    result = subparsers.add_parser(
        "result", help="Split generate_result summary CSV rows into event/quiet groups."
    )
    result.add_argument("--events-csv", required=True, help="Event calendar CSV path.")
    result.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to outputs/rq3/rq3_results_<timestamp>.",
    )
    result.add_argument(
        "--window-minutes",
        type=float,
        default=30.0,
        help="Symmetric event-window half-width in minutes.",
    )
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
    build_news_quiet.add_argument(
        "--source-merged-vol",
        required=True,
        help="Source merged_vol_rq2_text.xlsx path.",
    )
    build_news_quiet.add_argument(
        "--surface-all-json", required=True, help="surface-*-all.json path."
    )
    build_news_quiet.add_argument(
        "--news-xlsx",
        required=True,
        help="Raw news workbook used to define news buffers.",
    )
    build_news_quiet.add_argument(
        "--source-timezone",
        default=DEFAULT_SOURCE_TIMEZONE,
        help="Timezone for Factiva PD/ET publication timestamps.",
    )
    build_news_quiet.add_argument(
        "--output-workbook", required=True, help="Output news/quiet workbook path."
    )
    build_news_quiet.add_argument(
        "--sheet-name",
        default=GAN_SHEET,
        help=f"Source sheet name (default: {GAN_SHEET}).",
    )
    build_news_quiet.add_argument(
        "--horizon-minutes", type=int, default=5, help="Target horizon in minutes."
    )
    build_news_quiet.add_argument(
        "--quiet-grid-minutes", type=int, default=5, help="Quiet candidate grid size."
    )
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
    prepare_news_quiet.add_argument(
        "--source-merged-vol",
        required=True,
        help="Source merged_vol_rq2_text.xlsx path.",
    )
    prepare_news_quiet.add_argument(
        "--news-xlsx",
        required=True,
        help="Raw news workbook used to define news buffers.",
    )
    prepare_news_quiet.add_argument(
        "--source-timezone",
        default=DEFAULT_SOURCE_TIMEZONE,
        help="Timezone for Factiva PD/ET publication timestamps.",
    )
    prepare_news_quiet.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for quiet_targets.txt and audit CSV.",
    )
    prepare_news_quiet.add_argument(
        "--sheet-name",
        default=GAN_SHEET,
        help=f"Source sheet name (default: {GAN_SHEET}).",
    )
    prepare_news_quiet.add_argument(
        "--horizon-minutes", type=int, default=5, help="Target horizon in minutes."
    )
    prepare_news_quiet.add_argument(
        "--quiet-grid-minutes", type=int, default=5, help="Quiet candidate grid size."
    )
    prepare_news_quiet.add_argument(
        "--quiet-buffer-minutes",
        type=int,
        default=60,
        help="Minimum distance from any news timestamp.",
    )
    prepare_news_quiet.add_argument(
        "--candidate-count",
        type=int,
        default=12000,
        help="Number of eligible quiet targets to sample.",
    )
    prepare_news_quiet.add_argument(
        "--sample-seed",
        type=int,
        default=20260625,
        help="Random seed for target sampling.",
    )

    build_news_quiet_window = subparsers.add_parser(
        "build-news-quiet-workbook-from-window",
        help="Build a news-event vs quiet workbook from window surface JSON.",
    )
    build_news_quiet_window.add_argument(
        "--source-merged-vol",
        required=True,
        help="Source merged_vol_rq2_text.xlsx path.",
    )
    build_news_quiet_window.add_argument(
        "--window-surface-json", required=True, help="surface-*-window.json path."
    )
    build_news_quiet_window.add_argument(
        "--output-workbook", required=True, help="Output news/quiet workbook path."
    )
    build_news_quiet_window.add_argument(
        "--sheet-name",
        default=GAN_SHEET,
        help=f"Source sheet name (default: {GAN_SHEET}).",
    )
    build_news_quiet_window.add_argument(
        "--horizon-minutes", type=int, default=5, help="Target horizon in minutes."
    )
    build_news_quiet_window.add_argument(
        "--quiet-grid-minutes", type=int, default=5, help="Quiet candidate grid size."
    )
    build_news_quiet_window.add_argument(
        "--quiet-buffer-minutes",
        type=int,
        default=60,
        help="Minimum distance from any news timestamp.",
    )
    build_news_quiet_window.add_argument(
        "--quiet-max-samples", type=int, default=3711, help="Final quiet rows to keep."
    )
    build_news_quiet_window.add_argument(
        "--quiet-sample-seed",
        type=int,
        default=20260625,
        help="Random seed for final quiet row sampling.",
    )

    news_quiet_workbook = subparsers.add_parser(
        "news-quiet-workbook",
        help="Analyze current-to-target IVS jumps in a news-event vs quiet workbook.",
    )
    news_quiet_workbook.add_argument(
        "--workbook", required=True, help="Input news/quiet workbook path."
    )
    news_quiet_workbook.add_argument(
        "--output-dir", default="", help="Output directory."
    )
    news_quiet_workbook.add_argument(
        "--sheet-name",
        default=GAN_SHEET,
        help=f"Workbook sheet name (default: {GAN_SHEET}).",
    )
    news_quiet_workbook.add_argument(
        "--split", choices=["train", "val", "all"], default="all"
    )
    news_quiet_workbook.add_argument("--train-ratio", type=float, default=0.8)

    news_quiet_result = subparsers.add_parser(
        "news-quiet-result",
        help="Analyze generate-result summaries with news_event/quiet metadata.",
    )
    news_quiet_result.add_argument("--output-dir", default="", help="Output directory.")
    news_quiet_result.add_argument(
        "--text-label", default="", help="Model label for LP text result."
    )
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

    market_jumps = subparsers.add_parser(
        "detect-market-jumps",
        help=(
            "Detect approximate-ATM and volatility-skew jumps from the full "
            "raw-IV market index, then join scheduled releases and Factiva news."
        ),
    )
    market_jumps.add_argument(
        "--config",
        required=True,
        help="Market-jump detection YAML config.",
    )
    market_jumps.add_argument(
        "--output-dir",
        default="",
        help="Output archive. Defaults to outputs/rq3/atm_skew_jumps_<timestamp>.",
    )

    news_first_vol = subparsers.add_parser(
        "build-news-first-vol-surfaces",
        help=(
            "Build cumulative 5/10/15/30-minute news-first raw-vol surface "
            "datasets from the full Factiva news population."
        ),
    )
    news_first_vol.add_argument(
        "--config",
        required=True,
        help="News-first vol-surface YAML config.",
    )
    news_first_vol.add_argument(
        "--output-dir",
        default="",
        help=(
            "Output dataset root. Defaults to "
            "data/processed/rq3/news_first_vol_surfaces_<timestamp>."
        ),
    )

    news_first_training = subparsers.add_parser(
        "train-news-first-vol-comparison",
        help=(
            "Prepare, dry-run, or launch the frozen two-model comparison over "
            "the cumulative 5/10/15/30-minute news-first datasets. A successful "
            "formal launch also runs comparison analysis and the portable report."
        ),
    )
    news_first_training.add_argument(
        "action",
        nargs="?",
        choices=("prepare", "dry-run", "worker", "launch"),
        default="launch",
        help="Orchestration action (default: launch).",
    )
    news_first_training.add_argument(
        "--config",
        required=True,
        help="Frozen news-first training-comparison YAML config.",
    )
    news_first_training.add_argument(
        "--output-dir",
        default="",
        help=(
            "Experiment root. Defaults to "
            "outputs/rq3/news_first_vol_training_<timestamp>."
        ),
    )
    news_first_training.add_argument(
        "--job-id",
        default="",
        help="Registry job identifier; required only for the worker action.",
    )
    news_first_training.add_argument(
        "--resume",
        action="store_true",
        help="Skip hash-valid completed jobs and restart failed/interrupted jobs safely.",
    )
    news_first_training.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse an identically configured prepared experiment root.",
    )
    news_first_training.add_argument(
        "--worker-dry-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    capacity_sweep = subparsers.add_parser(
        "train-news-first-vol-capacity-sweep",
        help=(
            "Run the staged Regression-first model-capacity sweep, then test "
            "the shortlisted capacities with WGAN without exposing Q4 early."
        ),
    )
    capacity_sweep.add_argument(
        "action",
        nargs="?",
        choices=("prepare", "dry-run", "worker", "launch"),
        default="launch",
        help="Capacity orchestration action (default: launch).",
    )
    capacity_sweep.add_argument(
        "--config",
        required=True,
        help="Frozen news-first capacity-sweep YAML config.",
    )
    capacity_sweep.add_argument(
        "--output-dir",
        default="",
        help=(
            "Experiment root. Defaults to outputs/experiments/"
            "rq3_news_first_vol_capacity_sweep_<timestamp>."
        ),
    )
    capacity_sweep.add_argument(
        "--job-id",
        default="",
        help="Registry job identifier; required only for the worker action.",
    )
    capacity_sweep.add_argument(
        "--resume",
        action="store_true",
        help="Resume hash-valid completed stages and failed/interrupted jobs safely.",
    )
    capacity_sweep.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse an identically configured prepared capacity experiment root.",
    )
    capacity_sweep.add_argument(
        "--worker-dry-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    lr_sweep = subparsers.add_parser(
        "train-news-first-vol-lr-sweep",
        help=(
            "Run the Q3-only Regression large-profile learning-rate screen "
            "with immutable LR/scheduler lineage and no Q4 access."
        ),
    )
    lr_sweep.add_argument(
        "action",
        nargs="?",
        choices=("prepare", "dry-run", "worker", "launch"),
        default="launch",
        help="Learning-rate orchestration action (default: launch).",
    )
    lr_sweep.add_argument(
        "--config",
        required=True,
        help="Frozen news-first learning-rate-sweep YAML config.",
    )
    lr_sweep.add_argument(
        "--output-dir",
        default="",
        help=(
            "Experiment root. Defaults to outputs/experiments/"
            "rq3_news_first_vol_lr_sweep_<timestamp>."
        ),
    )
    lr_sweep.add_argument(
        "--job-id",
        default="",
        help="Registry job identifier; required only for the worker action.",
    )
    lr_sweep.add_argument(
        "--resume",
        action="store_true",
        help="Resume only after immutable LR/config/source lineage validates.",
    )
    lr_sweep.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse an identically configured prepared LR-sweep root.",
    )
    lr_sweep.add_argument(
        "--worker-dry-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    local_lr_sweep = subparsers.add_parser(
        "train-news-first-vol-local-lr-seed-sweep",
        help=(
            "Run the Q3-only local learning-rate sweep around 1e-6 with "
            "three independent training seeds per learning rate."
        ),
    )
    local_lr_sweep.add_argument(
        "action",
        nargs="?",
        choices=("prepare", "dry-run", "worker", "launch"),
        default="launch",
        help="Local LR/seed orchestration action (default: launch).",
    )
    local_lr_sweep.add_argument(
        "--config",
        required=True,
        help="Frozen local learning-rate/seed sweep YAML config.",
    )
    local_lr_sweep.add_argument(
        "--output-dir",
        default="",
        help=(
            "Experiment root. Defaults to outputs/experiments/"
            "rq3_news_first_vol_local_lr_seed_sweep_<timestamp>."
        ),
    )
    local_lr_sweep.add_argument(
        "--job-id",
        default="",
        help="Registry job identifier; required only for worker.",
    )
    local_lr_sweep.add_argument(
        "--resume",
        action="store_true",
        help="Resume only after immutable LR/seed/config/source lineage validates.",
    )
    local_lr_sweep.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse an identically configured prepared local LR experiment.",
    )
    local_lr_sweep.add_argument(
        "--worker-dry-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    fixed_lr_capacity = subparsers.add_parser(
        "train-news-first-vol-fixed-lr-capacity-seed-sweep",
        help=(
            "Run the Q3-only six-capacity Regression screen at LR 5e-7 "
            "with three independent seeds per capacity."
        ),
    )
    fixed_lr_capacity.add_argument(
        "action",
        nargs="?",
        choices=("prepare", "dry-run", "worker", "launch"),
        default="launch",
        help="Fixed-LR capacity/seed orchestration action (default: launch).",
    )
    fixed_lr_capacity.add_argument(
        "--config",
        required=True,
        help="Frozen fixed-LR capacity/seed sweep YAML config.",
    )
    fixed_lr_capacity.add_argument(
        "--output-dir",
        default="",
        help=(
            "Experiment root. Defaults to outputs/experiments/"
            "rq3_news_first_vol_fixed_lr_capacity_seed_sweep_<timestamp>."
        ),
    )
    fixed_lr_capacity.add_argument(
        "--job-id",
        default="",
        help="Registry job identifier; required only for worker.",
    )
    fixed_lr_capacity.add_argument(
        "--resume",
        action="store_true",
        help="Resume only after immutable capacity/seed/source lineage validates.",
    )
    fixed_lr_capacity.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse an identically configured prepared fixed-LR experiment.",
    )
    fixed_lr_capacity.add_argument(
        "--worker-dry-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    current_input_sweep = subparsers.add_parser(
        "train-news-first-vol-current-input-sweep",
        help=(
            "Run the Q3-only three-seed current-support-masked Generator "
            "input comparison against frozen full-current references."
        ),
    )
    current_input_sweep.add_argument(
        "action",
        nargs="?",
        choices=("prepare", "dry-run", "worker", "launch"),
        default="prepare",
        help="Current-input orchestration action (safe default: prepare).",
    )
    current_input_sweep.add_argument(
        "--config",
        default="configs/rq3/news_first_vol_current_input_sweep.yaml",
        help="Frozen current-input sweep YAML config.",
    )
    current_input_sweep.add_argument(
        "--output-dir",
        default=(
            "outputs/experiments/"
            "rq3_news_first_vol_current_support_masked_seed_sweep_"
            "q097_103_ttm07_38_v1"
        ),
        help="Independent immutable experiment root.",
    )
    current_input_sweep.add_argument(
        "--job-id",
        default="",
        help="Registry job identifier; required only for worker.",
    )
    current_input_sweep.add_argument(
        "--resume",
        action="store_true",
        help="Resume only after config/job/source/reference hashes validate.",
    )
    current_input_sweep.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse an identically configured prepared experiment root.",
    )
    current_input_sweep.add_argument(
        "--worker-dry-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    coverage_sweep = subparsers.add_parser(
        "train-news-first-vol-coverage-sweep",
        help=(
            "Run the four-stage Q3-only coverage completion sweep across "
            "tolerances, shuffled text, low-LR WGAN capacity, and the "
            "Regression capacity-by-learning-rate interaction grid."
        ),
    )
    coverage_sweep.add_argument(
        "action",
        nargs="?",
        choices=("prepare", "dry-run", "worker", "qa", "launch"),
        default="launch",
        help="Coverage-sweep orchestration action (default: launch).",
    )
    coverage_sweep.add_argument(
        "--config",
        required=True,
        help="Frozen four-stage coverage-completion YAML config.",
    )
    coverage_sweep.add_argument(
        "--output-dir",
        default="",
        help=(
            "Experiment root. Defaults to outputs/experiments/"
            "rq3_news_first_vol_coverage_completion_<timestamp>."
        ),
    )
    coverage_sweep.add_argument(
        "--job-id",
        default="",
        help="Registry job identifier; required only for worker.",
    )
    coverage_sweep.add_argument(
        "--stage-id",
        default="",
        help="Stage identifier; required only for the explicit QA action.",
    )
    coverage_sweep.add_argument(
        "--resume",
        action="store_true",
        help="Resume only after stage, job, source, config, and artifact hashes validate.",
    )
    coverage_sweep.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse an identically configured prepared coverage-sweep root.",
    )
    coverage_sweep.add_argument(
        "--worker-dry-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    grid08_wgan_sweep = subparsers.add_parser(
        "train-news-first-vol-wgan-grid08-sweep",
        help=(
            "Run the independent 8x8 WGAN capacity-by-learning-rate sweep, "
            "Q2 selection, current-only diagnostics, and frozen Q3 evaluation."
        ),
    )
    grid08_wgan_sweep.add_argument(
        "action",
        nargs="?",
        choices=("prepare", "benchmark", "dry-run", "worker", "launch", "postprocess"),
        default="launch",
        help="Grid08 sweep action (default: launch).",
    )
    grid08_wgan_sweep.add_argument(
        "--config",
        default="configs/rq3/news_first_vol_wgan_grid08_capacity_lr.yaml",
        help="Frozen 8x8 WGAN capacity/LR config.",
    )
    grid08_wgan_sweep.add_argument(
        "--output-dir",
        default=(
            "outputs/experiments/"
            "rq3_news_first_vol_wgan_grid08_capacity_lr_q097_103_ttm07_38_v1"
        ),
        help="Independent immutable experiment root.",
    )
    grid08_wgan_sweep.add_argument(
        "--job-id", default="", help="Registry job identifier for worker."
    )
    grid08_wgan_sweep.add_argument(
        "--resume", action="store_true", help="Resume after all hashes validate."
    )
    grid08_wgan_sweep.add_argument(
        "--reuse", action="store_true", help="Reuse an identical prepared root."
    )
    grid08_wgan_sweep.add_argument(
        "--worker-dry-run", action="store_true", help=argparse.SUPPRESS
    )

    film_critic_factorial = subparsers.add_parser(
        "train-news-first-vol-generator-film-critic-factorial",
        help=(
            "Run the single-seed Generator-FiLM by Critic-LP factorial, freeze "
            "Q3 epoch/LR recipes, refit through Q3, then explicitly evaluate Q4."
        ),
    )
    film_critic_factorial.add_argument(
        "action",
        nargs="?",
        choices=(
            "prepare",
            "benchmark",
            "dry-run",
            "launch-development",
            "freeze-selection",
            "launch-refit",
            "evaluate-q4",
            "postprocess",
            "worker",
            "qa",
        ),
        default="prepare",
        help="Factorial action (safe default: prepare).",
    )
    film_critic_factorial.add_argument(
        "--config",
        default="configs/rq3/news_first_vol_generator_film_critic_factorial.yaml",
        help="Frozen exact-TTM factorial config.",
    )
    film_critic_factorial.add_argument(
        "--output-dir",
        default=(
            "outputs/experiments/"
            "rq3_news_first_vol_generator_film_critic_factorial_exact_ttm_seed42_v1"
        ),
        help="Independent two-stage experiment root.",
    )
    film_critic_factorial.add_argument(
        "--job-id", default="", help="Registry job identifier for worker."
    )
    film_critic_factorial.add_argument("--resume", action="store_true")
    film_critic_factorial.add_argument("--reuse", action="store_true")
    film_critic_factorial.add_argument(
        "--worker-dry-run", action="store_true", help=argparse.SUPPRESS
    )

    film_nolp_capacity = subparsers.add_parser(
        "train-news-first-vol-film-nolp-capacity-seed-sweep",
        help=(
            "Run the three-seed, six-capacity FiLM-Generator + LP-disabled "
            "Critic development/refit/Q4 experiment."
        ),
    )
    film_nolp_capacity.add_argument(
        "action",
        nargs="?",
        choices=(
            "prepare",
            "benchmark",
            "dry-run",
            "launch-development",
            "freeze-selection",
            "launch-refit",
            "evaluate-q4",
            "postprocess",
            "worker",
            "qa",
            "status",
            "run-pipeline",
        ),
        default="prepare",
        help="Capacity experiment action (safe default: prepare).",
    )
    film_nolp_capacity.add_argument(
        "--config",
        default="configs/rq3/news_first_vol_film_nolp_capacity_seed.yaml",
        help="Frozen exact-TTM FiLM/NoLP capacity config.",
    )
    film_nolp_capacity.add_argument(
        "--output-dir",
        default=(
            "outputs/experiments/"
            "rq3_news_first_vol_film_nolp_capacity_seed_exact_ttm_v1"
        ),
        help="Independent development/refit/Q4 experiment root.",
    )
    film_nolp_capacity.add_argument(
        "--job-id", default="", help="Registry job identifier for worker."
    )
    film_nolp_capacity.add_argument("--resume", action="store_true")
    film_nolp_capacity.add_argument("--reuse", action="store_true")
    film_nolp_capacity.add_argument(
        "--worker-dry-run", action="store_true", help=argparse.SUPPRESS
    )

    legacy_architecture_seed = subparsers.add_parser(
        "train-news-first-vol-legacy-architecture-seed-sweep",
        help=(
            "Run the legacy-width, three-seed 2x2 Generator/Critic architecture "
            "comparison on the common 5m Q3 panel without reading Q4."
        ),
    )
    legacy_architecture_seed.add_argument(
        "action",
        nargs="?",
        choices=(
            "prepare",
            "dry-run",
            "launch",
            "worker",
            "analyze",
            "postprocess",
            "qa",
            "status",
            "run-pipeline",
        ),
        default="prepare",
        help="Legacy architecture experiment action (safe default: prepare).",
    )
    legacy_architecture_seed.add_argument(
        "--config",
        default="configs/rq3/news_first_vol_legacy_architecture_seed.yaml",
        help="Frozen exact-TTM legacy architecture config.",
    )
    legacy_architecture_seed.add_argument(
        "--output-dir",
        default=(
            "outputs/experiments/"
            "rq3_news_first_vol_legacy_architecture_seed_exact_ttm_v1"
        ),
        help="Independent Q3-only architecture experiment root.",
    )
    legacy_architecture_seed.add_argument(
        "--job-id", default="", help="Registry job identifier for worker."
    )
    legacy_architecture_seed.add_argument("--resume", action="store_true")
    legacy_architecture_seed.add_argument("--reuse", action="store_true")
    legacy_architecture_seed.add_argument(
        "--worker-dry-run", action="store_true", help=argparse.SUPPRESS
    )

    label_reliability = subparsers.add_parser(
        "train-news-first-vol-label-reliability",
        help=(
            "Audit target-label reliability, then run the conditional "
            "A/B/C/D rolling-fold Regression and WGAN experiment."
        ),
    )
    label_reliability.add_argument(
        "action",
        nargs="?",
        choices=("prepare", "bootstrap", "dry-run", "worker", "launch", "postprocess"),
        default="prepare",
        help="Label-reliability action (safe default: prepare).",
    )
    label_reliability.add_argument(
        "--config",
        default="configs/rq3/news_first_vol_label_reliability.yaml",
        help="Frozen label-reliability experiment YAML config.",
    )
    label_reliability.add_argument(
        "--output-dir",
        default=(
            "outputs/experiments/"
            "rq3_news_first_vol_label_reliability_q097_103_ttm07_38_v1"
        ),
        help="Independent immutable label-reliability experiment root.",
    )
    label_reliability.add_argument("--job-id", default="")
    label_reliability.add_argument("--resume", action="store_true")
    label_reliability.add_argument("--reuse", action="store_true")
    label_reliability.add_argument(
        "--worker-dry-run", action="store_true", help=argparse.SUPPRESS
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
        # Keep the frozen-model training stack out of lightweight analytical
        # commands such as detect-market-jumps.  Some research environments
        # intentionally install only the pandas/scipy analysis dependencies.
        from scripts.rq3.scheduled_news_regime import run_scheduled_news_regime

        output = run_scheduled_news_regime(
            args.config,
            output_dir=args.output_dir or None,
            rq1_experiment_override=args.rq1_experiment or None,
            rq2_experiment_override=args.rq2_experiment or None,
            event_calendar_override=args.event_calendar or None,
        )
        print(f"RQ3 scheduled-news regime archive written to {output}")
        return output
    if args.command == "detect-market-jumps":
        output_dir = args.output_dir or _default_output_dir("atm_skew_jumps")
        output = run_market_jump_detection(
            args.config,
            output_dir=output_dir,
        )
        print(f"RQ3 ATM/skew market-jump analysis written to {output}")
        return output
    if args.command == "build-news-first-vol-surfaces":
        # Keep the large raw-surface reconstruction stack out of lightweight
        # RQ3 analytical commands until this builder is explicitly requested.
        from scripts.rq3.news_first_vol_surfaces import run_news_first_vol_surfaces

        output_dir = args.output_dir or _default_processed_output_dir(
            "news_first_vol_surfaces"
        )
        output = run_news_first_vol_surfaces(
            args.config,
            output_dir=output_dir,
        )
        print(f"RQ3 news-first vol-surface datasets written to {output}")
        return output
    if args.command == "train-news-first-vol-comparison":
        # Keep torch/CUDA imports out of analytical RQ3 commands.  The worker
        # also sets CUDA visibility and CPU thread caps before importing the
        # existing training stack.
        from scripts.rq3.news_first_vol_training import run_news_first_vol_training

        output_dir = args.output_dir or _default_output_dir("news_first_vol_training")
        output = run_news_first_vol_training(
            args.config,
            output_dir,
            action=args.action,
            job_id=args.job_id,
            resume=bool(args.resume),
            reuse=bool(args.reuse),
            worker_dry_run=bool(args.worker_dry_run),
        )
        print(f"RQ3 news-first vol training comparison written to {output}")
        return output
    if args.command == "train-news-first-vol-capacity-sweep":
        from scripts.rq3.news_first_vol_training import (
            run_news_first_vol_capacity_sweep,
        )

        output_dir = args.output_dir or str(
            Path("outputs/experiments")
            / f"rq3_news_first_vol_capacity_sweep_{timestamp_string()}"
        )
        output = run_news_first_vol_capacity_sweep(
            args.config,
            output_dir,
            action=args.action,
            job_id=args.job_id,
            resume=bool(args.resume),
            reuse=bool(args.reuse),
            worker_dry_run=bool(args.worker_dry_run),
        )
        print(f"RQ3 news-first vol capacity sweep written to {output}")
        return output
    if args.command == "train-news-first-vol-lr-sweep":
        from scripts.rq3.news_first_vol_lr_sweep import (
            run_news_first_vol_lr_sweep,
        )

        output_dir = args.output_dir or str(
            Path("outputs/experiments")
            / f"rq3_news_first_vol_lr_sweep_{timestamp_string()}"
        )
        output = run_news_first_vol_lr_sweep(
            args.config,
            output_dir,
            action=args.action,
            job_id=args.job_id,
            resume=bool(args.resume),
            reuse=bool(args.reuse),
            worker_dry_run=bool(args.worker_dry_run),
        )
        print(f"RQ3 news-first vol LR sweep written to {output}")
        return output
    if args.command == "train-news-first-vol-local-lr-seed-sweep":
        from scripts.rq3.news_first_vol_local_lr_sweep import (
            run_news_first_vol_local_lr_sweep,
        )

        output_dir = args.output_dir or str(
            Path("outputs/experiments")
            / f"rq3_news_first_vol_local_lr_seed_sweep_{timestamp_string()}"
        )
        output = run_news_first_vol_local_lr_sweep(
            args.config,
            output_dir,
            action=args.action,
            job_id=args.job_id,
            resume=bool(args.resume),
            reuse=bool(args.reuse),
            worker_dry_run=bool(args.worker_dry_run),
        )
        print(f"RQ3 local LR/seed sweep written to {output}")
        return output
    if args.command == "train-news-first-vol-fixed-lr-capacity-seed-sweep":
        from scripts.rq3.news_first_vol_fixed_lr_capacity_sweep import (
            run_news_first_vol_fixed_lr_capacity_sweep,
        )

        output_dir = args.output_dir or str(
            Path("outputs/experiments")
            / (f"rq3_news_first_vol_fixed_lr_capacity_seed_sweep_{timestamp_string()}")
        )
        output = run_news_first_vol_fixed_lr_capacity_sweep(
            args.config,
            output_dir,
            action=args.action,
            job_id=args.job_id,
            resume=bool(args.resume),
            reuse=bool(args.reuse),
            worker_dry_run=bool(args.worker_dry_run),
        )
        print(f"RQ3 fixed-LR capacity/seed sweep written to {output}")
        return output
    if args.command == "train-news-first-vol-current-input-sweep":
        from scripts.rq3.news_first_vol_current_input_sweep import (
            run_news_first_vol_current_input_sweep,
        )

        output = run_news_first_vol_current_input_sweep(
            args.config,
            args.output_dir,
            action=args.action,
            job_id=args.job_id,
            resume=bool(args.resume),
            reuse=bool(args.reuse),
            worker_dry_run=bool(args.worker_dry_run),
        )
        print(f"RQ3 current-input sweep written to {output}")
        return output
    if args.command == "train-news-first-vol-coverage-sweep":
        from scripts.rq3.news_first_vol_coverage_sweep import (
            run_news_first_vol_coverage_sweep,
        )

        output_dir = args.output_dir or str(
            Path("outputs/experiments")
            / f"rq3_news_first_vol_coverage_completion_{timestamp_string()}"
        )

        stage_hook = None
        final_hook = None
        if args.action == "launch":
            from scripts.rq3.news_first_vol_coverage_sweep_analysis import (
                run_coverage_sweep_analysis,
            )
            from scripts.rq3.news_first_vol_coverage_sweep_report import (
                render_coverage_sweep_report,
            )

            def stage_hook(root: Path, stage_id: str) -> None:
                run_coverage_sweep_analysis(root, stages=(stage_id,))

            def final_hook(root: Path) -> None:
                run_coverage_sweep_analysis(root)
                render_coverage_sweep_report(root)

        output = run_news_first_vol_coverage_sweep(
            args.config,
            output_dir,
            action=args.action,
            job_id=args.job_id,
            stage_id=args.stage_id,
            resume=bool(args.resume),
            reuse=bool(args.reuse),
            worker_dry_run=bool(args.worker_dry_run),
            stage_postprocess_hook=stage_hook,
            postprocess_hook=final_hook,
        )
        print(f"RQ3 coverage-completion sweep written to {output}")
        return output
    if args.command == "train-news-first-vol-wgan-grid08-sweep":
        from scripts.rq3.news_first_vol_wgan_grid08_sweep import (
            run_news_first_vol_wgan_grid08_sweep,
        )

        output = run_news_first_vol_wgan_grid08_sweep(
            args.config,
            args.output_dir,
            action=args.action,
            job_id=args.job_id,
            resume=bool(args.resume),
            reuse=bool(args.reuse),
            worker_dry_run=bool(args.worker_dry_run),
        )
        print(f"RQ3 8x8 WGAN capacity/LR sweep written to {output}")
        return output
    if args.command == "train-news-first-vol-generator-film-critic-factorial":
        from scripts.rq3.news_first_vol_generator_film_critic_factorial import (
            run_news_first_vol_generator_film_critic_factorial,
        )

        output = run_news_first_vol_generator_film_critic_factorial(
            args.config,
            args.output_dir,
            action=args.action,
            job_id=args.job_id,
            resume=bool(args.resume),
            reuse=bool(args.reuse),
            worker_dry_run=bool(args.worker_dry_run),
        )
        print(f"RQ3 Generator/Critic factorial experiment written to {output}")
        return output
    if args.command == "train-news-first-vol-film-nolp-capacity-seed-sweep":
        from scripts.rq3.news_first_vol_film_nolp_capacity_seed import (
            run_news_first_vol_film_nolp_capacity_seed,
        )

        output = run_news_first_vol_film_nolp_capacity_seed(
            args.config,
            args.output_dir,
            action=args.action,
            job_id=args.job_id,
            resume=bool(args.resume),
            reuse=bool(args.reuse),
            worker_dry_run=bool(args.worker_dry_run),
        )
        print(f"RQ3 FiLM/NoLP capacity-seed experiment: {output}")
        return output
    if args.command == "train-news-first-vol-legacy-architecture-seed-sweep":
        from scripts.rq3.news_first_vol_legacy_architecture_seed import (
            run_news_first_vol_legacy_architecture_seed,
        )

        output = run_news_first_vol_legacy_architecture_seed(
            args.config,
            args.output_dir,
            action=args.action,
            job_id=args.job_id,
            resume=bool(args.resume),
            reuse=bool(args.reuse),
            worker_dry_run=bool(args.worker_dry_run),
        )
        print(f"RQ3 legacy architecture-seed experiment: {output}")
        return output
    if args.command == "train-news-first-vol-label-reliability":
        from scripts.rq3.news_first_vol_label_reliability import (
            run_news_first_vol_label_reliability,
        )

        output = run_news_first_vol_label_reliability(
            args.config,
            args.output_dir,
            action=args.action,
            job_id=args.job_id,
            resume=bool(args.resume),
            reuse=bool(args.reuse),
            worker_dry_run=bool(args.worker_dry_run),
        )
        print(f"RQ3 label-reliability experiment written to {output}")
        return output
    raise ValueError(f"Unknown RQ3 command: {args.command}")


if __name__ == "__main__":
    main()
