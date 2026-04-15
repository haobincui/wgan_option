"""Standalone generate-result entrypoint for the independent StyleMod WGAN module."""

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

from stylemod_wgan.config import config_to_dict, load_train_config, parse_sample_overrides  # noqa: E402
from stylemod_wgan.trainer import StyleModWGANTrainer  # noqa: E402
from utils.standalone_cli import build_generate_arg_parser, run_generate_cli  # noqa: E402


def build_parser():
    return build_generate_arg_parser(
        description="Generate arbitrage-weighted future surfaces from a trained StyleMod WGAN run.",
        default_config_path="configs/stylemod_wgan/train_default.yaml",
    )


def main(argv: list[str] | None = None):
    return run_generate_cli(
        argv=argv,
        description="Generate arbitrage-weighted future surfaces from a trained StyleMod WGAN run.",
        default_config_path="configs/stylemod_wgan/train_default.yaml",
        load_train_config=load_train_config,
        parse_overrides=parse_sample_overrides,
        config_to_dict=config_to_dict,
        trainer_cls=StyleModWGANTrainer,
    )


if __name__ == "__main__":
    main()
