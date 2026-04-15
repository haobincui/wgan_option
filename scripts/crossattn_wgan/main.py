"""Standalone CLI router for the independent Cross-Attention WGAN module."""

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

from scripts.crossattn_wgan.generate_result import main as generate_result_main  # noqa: E402
from scripts.crossattn_wgan.sample import main as sample_main  # noqa: E402
from scripts.crossattn_wgan.train import main as train_main  # noqa: E402

HELP_TEXT = """Usage:
  python scripts/crossattn_wgan/main.py train --config configs/crossattn_wgan/train_default.yaml
  python scripts/crossattn_wgan/main.py sample --config configs/crossattn_wgan/train_default.yaml
  python scripts/crossattn_wgan/main.py generate-result --config configs/crossattn_wgan/train_default.yaml

Notes:
  `train` defaults to the full pipeline: training plus generate_result.
  `sample` is a thin alias for `generate-result` and also consumes the training YAML.
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
    raise SystemExit(f"Unsupported standalone Cross-Attention WGAN command: {command}\n\n{HELP_TEXT}")


if __name__ == "__main__":
    main()
