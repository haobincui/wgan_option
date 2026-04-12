"""CLI entrypoint for deterministic merged vol-surface regression."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from scripts.train.common import DEFAULT_VOL_REGRESSION_CONFIG_PATH, run_training_cli
from wgan_option.train_vol_regression_xlsx import VolSurfaceRegressionTrainer


def main(argv: Optional[Iterable[str]] = None):
    run_training_cli(
        argv=argv,
        description="Train deterministic residual regression on merged vol-surface xlsx data.",
        default_config_path=DEFAULT_VOL_REGRESSION_CONFIG_PATH,
        trainer_cls=VolSurfaceRegressionTrainer,
    )


if __name__ == "__main__":
    main()
