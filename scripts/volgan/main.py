"""Standalone CLI router for the independent VolGAN module."""

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

from scripts.volgan.generate_result import main as generate_result_main  # noqa: E402
from scripts.volgan.pipeline import main as pipeline_main  # noqa: E402
from scripts.volgan.sample import main as sample_main  # noqa: E402
from scripts.volgan.train import main as train_main  # noqa: E402

HELP_TEXT = """Usage:
  python scripts/volgan/main.py train --config configs/volgan/train_default.yaml
  python scripts/volgan/main.py sample --config configs/volgan/train_default.yaml
  python scripts/volgan/main.py generate-result --config configs/volgan/train_default.yaml
  python scripts/volgan/main.py pipeline --config configs/volgan/train_default.yaml
"""


def main(argv: list[str] | None = None):
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print(HELP_TEXT)
        return
    command, remainder = args[0], args[1:]
    if command == "train":
        return train_main(remainder)
    if command == "sample":
        return sample_main(remainder)
    if command in {"generate-result", "generate_result"}:
        return generate_result_main(remainder)
    if command == "pipeline":
        return pipeline_main(remainder)
    raise SystemExit(f"Unsupported standalone VolGAN command: {command}\n\n{HELP_TEXT}")


if __name__ == "__main__":
    main()
