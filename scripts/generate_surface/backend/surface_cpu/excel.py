"""Compatibility shim for CPU Excel surface generation backend."""

from scripts.generate_surface._compat import alias_module

_module = alias_module(__name__, "wgan_option.surface_generation.backend.surface_cpu.excel")


if __name__ == "__main__":
    _module.main()
