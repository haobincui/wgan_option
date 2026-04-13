"""Backward-compatible shim for the canonical ``wgan_option.trainer`` module."""

from .trainer import WGANTrainer, main

__all__ = ["WGANTrainer", "main"]
