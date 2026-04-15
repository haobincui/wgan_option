"""Thin wrapper around the shared generate-result CLI plus model dispatch."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from generate_result.cli import COMMANDS, main as generate_result_main  # noqa: E402
from scripts.cnn_wgan.sample import main as cnn_wgan_generate_main  # noqa: E402
from scripts.stylemod_wgan.sample import main as stylemod_wgan_generate_main  # noqa: E402
from scripts.transformer_wgan.sample import main as transformer_wgan_generate_main  # noqa: E402

HELP_TEXT = dedent(
    """
    Usage:
      python scripts/generate_result/main.py vol --config configs/wgan/train_vol_xlsx.yaml
      python scripts/generate_result/main.py vol-regression --config configs/wgan/train_vol_regression_xlsx.yaml
      python scripts/generate_result/main.py svi --config configs/wgan/train_svi_xlsx.yaml
      python scripts/generate_result/main.py --model cnn-wgan --config configs/generate_result/cnn_wgan_epoch0130.yaml
      python scripts/generate_result/main.py --model stylemod-wgan --config configs/stylemod_wgan/train_default.yaml
      python scripts/generate_result/main.py --model transformer-wgan --config configs/transformer_wgan/train_default.yaml
      python scripts/generate_result/main.py plot --input-json <payload.json> [--output <file.png>]

    Notes:
      `--model` only adds script-layer dispatch and keeps src-side trainer logic unchanged.
      CNN/StyleMod/Transformer routes can consume merged YAML configs such as files under configs/generate_result/.
    """
).strip()

MODEL_COMMANDS = {
    "vol": lambda argv: COMMANDS["vol"](argv),
    "vol-regression": lambda argv: COMMANDS["vol-regression"](argv),
    "vol_regression": lambda argv: COMMANDS["vol-regression"](argv),
    "svi": lambda argv: COMMANDS["svi"](argv),
    "cnn-wgan": cnn_wgan_generate_main,
    "cnn_wgan": cnn_wgan_generate_main,
    "stylemod-wgan": stylemod_wgan_generate_main,
    "stylemod_wgan": stylemod_wgan_generate_main,
    "transformer-wgan": transformer_wgan_generate_main,
    "transformer_wgan": transformer_wgan_generate_main,
}


def _extract_model_flag(argv_list: list[str]) -> tuple[str | None, list[str]]:
    model_name = None
    remainder: list[str] = []
    idx = 0

    while idx < len(argv_list):
        arg = argv_list[idx]
        if arg in {"--model", "-m"}:
            if idx + 1 >= len(argv_list):
                raise SystemExit("--model requires a value.")
            if model_name is not None:
                raise SystemExit("--model may only be provided once.")
            model_name = argv_list[idx + 1]
            idx += 2
            continue
        if arg.startswith("--model="):
            if model_name is not None:
                raise SystemExit("--model may only be provided once.")
            model_name = arg.split("=", 1)[1]
            idx += 1
            continue
        remainder.append(arg)
        idx += 1

    return model_name, remainder


def main(argv=None) -> None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    if not argv_list or argv_list[0] in {"-h", "--help"}:
        print(HELP_TEXT)
        return None

    model_name, remainder = _extract_model_flag(argv_list)
    if model_name is not None:
        if remainder and remainder[0] in COMMANDS:
            raise SystemExit("Do not combine --model with subcommands. Use one style only.")
        normalized_model = str(model_name).strip().lower()
        if normalized_model not in MODEL_COMMANDS:
            raise SystemExit(f"Unsupported --model value: {model_name}\n\n{HELP_TEXT}")
        return MODEL_COMMANDS[normalized_model](remainder)

    return generate_result_main(argv_list)


if __name__ == "__main__":
    main()
