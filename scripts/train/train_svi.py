"""CLI entrypoint for merged SVI xlsx training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from scripts.train.common import DEFAULT_SVI_CONFIG_PATH, run_training_cli
from wgan_option.train_svi_xlsx import SviXlsxTrainer


def main(argv: Optional[Iterable[str]] = None):
    run_training_cli(
        argv=argv,
        description="Train an SVI regressor on merged SVI xlsx data.",
        default_config_path=DEFAULT_SVI_CONFIG_PATH,
        trainer_cls=SviXlsxTrainer,
    )


if __name__ == "__main__":
    main()
