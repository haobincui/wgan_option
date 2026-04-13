"""Shared CLI dispatch helpers for generate-surface entrypoints."""

from __future__ import annotations

import argparse
from typing import Iterable, List, Tuple

import torch


VALID_DEVICES = {"cpu", "gpu"}


def extract_device_arg(argv: Iterable[str], default: str = "cpu") -> Tuple[str, List[str]]:
    argv_list = list(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=sorted(VALID_DEVICES), default=default)
    parsed, remaining = parser.parse_known_args(argv_list)
    return str(parsed.device), remaining


def ensure_cuda_available(device: str) -> None:
    if device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("`--device gpu` was requested, but CUDA is not available on this machine.")
