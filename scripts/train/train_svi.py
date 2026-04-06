"""CLI entrypoint for merged SVI xlsx training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
