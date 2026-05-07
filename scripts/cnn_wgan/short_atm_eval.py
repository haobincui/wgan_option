"""CLI for post-hoc short-end ATM evaluation of CNN WGAN sample JSON outputs."""

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

from cnn_wgan.short_atm_eval import evaluate_many, format_summary  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute post-hoc short-end ATM metrics from CNN WGAN generate-result sample JSON files.",
    )
    parser.add_argument(
        "--samples-dir",
        required=True,
        help="Directory containing generate-result sample JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for summary.json and per_sample.csv. If empty, metrics are only printed.",
    )
    parser.add_argument("--label", default="short_atm_posthoc", help="Label stored in output artifacts.")
    parser.add_argument("--atm-short-range", type=float, default=0.04, help="Pure ATM mask strike half-width.")
    parser.add_argument("--atm-short-max-days", type=float, default=60.0, help="Pure ATM mask max maturity in days.")
    parser.add_argument("--recon-atm-range", type=float, default=0.04, help="Weighted band strike half-width.")
    parser.add_argument(
        "--recon-atm-short-end-max-days",
        type=float,
        default=60.0,
        help="Weighted band max maturity in days.",
    )
    parser.add_argument("--recon-atm-multiplier", type=float, default=16.0, help="Weighted band multiplier.")
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_values = {
        "label": args.label,
        "atm_short_range": args.atm_short_range,
        "atm_short_max_days": args.atm_short_max_days,
        "recon_atm_range": args.recon_atm_range,
        "recon_atm_short_end_max_days": args.recon_atm_short_end_max_days,
        "recon_atm_multiplier": args.recon_atm_multiplier,
    }
    _rows, summary, artifacts = evaluate_many(
        samples_dir=args.samples_dir,
        config_values=config_values,
        output_dir=args.output_dir or None,
    )
    print(format_summary(summary))
    if artifacts:
        print("summary_json:", artifacts["summary_json"])
        print("per_sample_csv:", artifacts["per_sample_csv"])
    return summary


if __name__ == "__main__":
    main()

