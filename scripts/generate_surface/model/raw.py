"""Compatibility shim for raw surface builder."""

from scripts.generate_surface._compat import alias_module

alias_module(__name__, "wgan_option.surface_generation.model.raw")
