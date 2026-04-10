"""Compatibility shim for cubic-spline surface builder."""

from scripts.generate_surface._compat import alias_module

alias_module(__name__, "wgan_option.surface_generation.model.cubic")
