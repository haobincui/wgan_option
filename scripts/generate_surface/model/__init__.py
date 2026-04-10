"""Compatibility shim for surface-generation model builders now owned by ``wgan_option``."""

from scripts.generate_surface._compat import alias_module

alias_module(__name__, "wgan_option.surface_generation.model")
