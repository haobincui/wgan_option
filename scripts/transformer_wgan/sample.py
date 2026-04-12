"""Standalone sampling entrypoint for the independent Transformer WGAN module."""

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

from transformer_wgan.config import load_sample_config  # noqa: E402
from transformer_wgan.inference import TransformerWGANSampler  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample arbitrage-weighted scenarios from standalone Transformer WGAN.")
    parser.add_argument("--config", required=True, help="Path to a Transformer WGAN sampling YAML config.")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    config = load_sample_config(args.config)
    sampler = TransformerWGANSampler(config)
    return sampler.sample()


if __name__ == "__main__":
    main()
