"""Reproducibility helpers shared by merged-surface trainers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch before loaders or models are constructed."""

    resolved_seed = int(seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seeded_torch_generator(seed: int) -> torch.Generator:
    """Return an explicit CPU generator for deterministic DataLoader shuffling."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator
