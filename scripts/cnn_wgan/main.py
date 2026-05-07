"""Standalone CLI router for the independent CNN WGAN module."""

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

HELP_TEXT = """Usage:
  python scripts/cnn_wgan/main.py train --config configs/cnn_wgan/train_default.yaml
  python scripts/cnn_wgan/main.py sample --config configs/cnn_wgan/train_default.yaml
  python scripts/cnn_wgan/main.py generate-result --config configs/cnn_wgan/train_default.yaml
  python scripts/cnn_wgan/main.py short-atm-eval --samples-dir outputs/training/cnn_wgan/svi-excel/<run>/generate_result/cnn_wgan_best/samples

Notes:
  `train` defaults to the full pipeline: training plus generate_result.
  `sample` is a thin alias for `generate-result` and also consumes the training YAML.
  `short-atm-eval` computes post-hoc short-end ATM metrics from saved sample JSON files.
"""


def main(argv: list[str] | None = None):
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print(HELP_TEXT)
        return
    command, remainder = args[0], args[1:]
    if command == "train":
        from scripts.cnn_wgan.train import main as train_main

        return train_main(remainder)
    if command == "sample":
        from scripts.cnn_wgan.sample import main as sample_main

        return sample_main(remainder)
    if command in {"generate-result", "generate_result"}:
        from scripts.cnn_wgan.generate_result import main as generate_result_main

        return generate_result_main(remainder)
    if command in {"short-atm-eval", "short_atm_eval"}:
        from scripts.cnn_wgan.short_atm_eval import main as short_atm_eval_main

        return short_atm_eval_main(remainder)
    raise SystemExit(f"Unsupported standalone CNN WGAN command: {command}\n\n{HELP_TEXT}")


if __name__ == "__main__":
    main()
