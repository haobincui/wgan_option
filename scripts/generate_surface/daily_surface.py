"""Thin daily-surface CLI wrapper used by the unified surface CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401
from scripts.generate_surface.common.daily_surface import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_daily_surface_config,
    parse_daily_surface_args,
    run_daily_surface_job,
)

_load_daily_surface_config = load_daily_surface_config
_parse_args = parse_daily_surface_args


def run(args) -> Path:
    return run_daily_surface_job(args)


def main(argv=None) -> None:
    args = _parse_args(argv)
    out_file = run(args)
    print(f"Surface file generated: {out_file}")


if __name__ == "__main__":
    main()
