"""CLI entrypoint for the FiLM WGAN short-end ATM study workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    SRC_DIR = ROOT_DIR / "src"
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

from film_wgan.short_atm_study import (  # noqa: E402
    DEFAULT_BASE_CONFIG_PATH,
    DEFAULT_BLEND_ALPHAS,
    DEFAULT_SEEDS,
    run_blend_scan,
    run_short_atm_study,
    summarize_short_atm_study,
    train_short_atm_grid,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the FiLM WGAN short-end ATM study grid, summaries, and residual blend scan.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_BASE_CONFIG_PATH,
        help=f"Base FiLM WGAN training config used for the study (default: {DEFAULT_BASE_CONFIG_PATH})",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "train-grid", "summarize", "blend-scan"),
        default="all",
        help="Which part of the study workflow to run.",
    )
    parser.add_argument(
        "--study-dir",
        type=str,
        default="",
        help="Existing study directory to reuse, or empty to create a fresh one for train-grid/all.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds used for the training grid.",
    )
    parser.add_argument(
        "--blend-alphas",
        nargs="*",
        type=float,
        default=list(DEFAULT_BLEND_ALPHAS),
        help="Residual blend alphas used during blend-scan.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Keep per-sample JSON payloads during study checkpoint exports.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Keep PNG plots during study checkpoint exports.",
    )
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    study_dir = Path(args.study_dir) if str(args.study_dir).strip() else None
    if args.phase in {"summarize", "blend-scan"} and study_dir is None:
        raise ValueError("--study-dir is required for phase=summarize and phase=blend-scan.")

    if args.phase == "train-grid":
        return train_short_atm_grid(
            base_config_path=args.config,
            study_dir=study_dir,
            seeds=args.seeds,
            save_json=bool(args.save_json),
            save_plots=bool(args.save_plots),
        )
    if args.phase == "summarize":
        return summarize_short_atm_study(study_dir)
    if args.phase == "blend-scan":
        return run_blend_scan(
            study_dir,
            base_config_path=args.config,
            blend_alphas=args.blend_alphas,
            save_json=bool(args.save_json),
            save_plots=bool(args.save_plots),
        )
    return run_short_atm_study(
        base_config_path=args.config,
        study_dir=study_dir,
        seeds=args.seeds,
        blend_alphas=args.blend_alphas,
        save_json=bool(args.save_json),
        save_plots=bool(args.save_plots),
    )


if __name__ == "__main__":
    main()
