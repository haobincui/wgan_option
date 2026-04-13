"""Thin wrapper around the shared generate-result CLI."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from generate_result.cli import COMMANDS, HELP_TEXT, main as generate_result_main  # noqa: E402


def main(argv=None) -> None:
    return generate_result_main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    main()
