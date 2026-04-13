"""Standalone generate-result entrypoint for the independent VolGAN module."""

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

from volgan.config import load_train_config  # noqa: E402
from volgan.trainer import VolGANTrainer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate results from a standalone VolGAN training YAML.")
    parser.add_argument("--config", required=True, help="Path to a VolGAN training YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override.")
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    config = load_train_config(args.config)
    trainer = VolGANTrainer(config, config_path=args.config)
    overrides = {}
    if args.checkpoint is not None:
        overrides["checkpoint_path"] = args.checkpoint
    return trainer.generate_result(
        overrides=overrides or None,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
