"""Thin CLI wrapper — generate future vol surfaces from a trained WGAN checkpoint."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, Optional

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from wgan_option.generate_result_runtime import run_vol_generation  # noqa: E402


def main(argv: Optional[Iterable[str]] = None) -> Path:
    return run_vol_generation(argv)


if __name__ == "__main__":
    main()
