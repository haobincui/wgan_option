"""Helpers for script-side compatibility shims in ``scripts.generate_surface``."""

from __future__ import annotations

import sys
from importlib import import_module

import scripts._path_setup  # noqa: F401


def alias_module(script_module_name: str, target_module_name: str):
    """Point a legacy script module path at the canonical implementation module."""

    module = import_module(target_module_name)
    sys.modules[script_module_name] = module
    return module
