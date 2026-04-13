"""Standalone training entrypoint for the independent Transformer WGAN module."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    SRC_DIR = ROOT_DIR / "src"
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

from transformer_wgan.config import config_to_dict, load_train_config, parse_train_overrides  # noqa: E402
from transformer_wgan.trainer import TransformerWGANTrainer  # noqa: E402
from utils.standalone_cli import run_training_cli  # noqa: E402


def main(argv: list[str] | None = None):
    return run_training_cli(
        argv=argv,
        description="Train or pipeline-run the standalone Transformer WGAN module.",
        default_config_path="configs/transformer_wgan/train_default.yaml",
        load_config=load_train_config,
        parse_overrides=parse_train_overrides,
        config_to_dict=config_to_dict,
        trainer_cls=TransformerWGANTrainer,
    )


if __name__ == "__main__":
    main()
