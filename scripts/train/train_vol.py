"""CLI entrypoint for merged vol-surface xlsx training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from scripts.train.common import DEFAULT_VOL_CONFIG_PATH, run_training_cli
from wgan_option.train_vol_xlsx import VolSurfaceXlsxTrainer


def main(argv: Optional[Iterable[str]] = None):
    run_training_cli(
        argv=argv,
        description="Train WGAN-GP on merged vol-surface xlsx data.",
        default_config_path=DEFAULT_VOL_CONFIG_PATH,
        trainer_cls=VolSurfaceXlsxTrainer,
    )


if __name__ == "__main__":
    main()
