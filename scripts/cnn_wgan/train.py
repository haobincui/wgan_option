"""Standalone training entrypoint for the independent CNN WGAN module."""

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

from cnn_wgan.config import load_train_config  # noqa: E402
from cnn_wgan.trainer import CnnWGANTrainer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the standalone independent CNN WGAN module.")
    parser.add_argument("--config", required=True, help="Path to a CNN WGAN training YAML config.")
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    config = load_train_config(args.config)
    trainer = CnnWGANTrainer(config)
    return trainer.train()


if __name__ == "__main__":
    main()
