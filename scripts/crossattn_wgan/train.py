"""Standalone training entrypoint for the independent Cross-Attention WGAN module."""

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

from crossattn_wgan.config import config_to_dict, load_train_config, parse_train_overrides  # noqa: E402
from crossattn_wgan.trainer import CrossAttnWGANTrainer  # noqa: E402
from utils.standalone_cli import run_training_cli  # noqa: E402


def main(argv: list[str] | None = None):
    return run_training_cli(
        argv=argv,
        description="Train or pipeline-run the standalone Cross-Attention WGAN module.",
        default_config_path="configs/crossattn_wgan/train_default.yaml",
        load_config=load_train_config,
        parse_overrides=parse_train_overrides,
        config_to_dict=config_to_dict,
        trainer_cls=CrossAttnWGANTrainer,
    )


if __name__ == "__main__":
    main()
