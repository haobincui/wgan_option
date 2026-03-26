"""Utilities for expanding small config variables inside surface-builder YAML."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping

_BRACED_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_BARE_VAR_PATH_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)([/\\].+)?$")


def _is_expandable_mapping_value(value: Any) -> bool:
    return not isinstance(value, dict)


def build_config_scope(config_root: Mapping[str, Any]) -> Dict[str, Any]:
    """Expose top-level scalar/list values as variables for section expansion."""

    return {
        str(key): value
        for key, value in config_root.items()
        if isinstance(key, str) and _is_expandable_mapping_value(value)
    }


def resolve_config_variables(
    values: Mapping[str, Any],
    extra_scope: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Resolve `${name}` and `name/...` references within one config section."""

    raw_values: Dict[str, Any] = {}
    if extra_scope:
        raw_values.update({str(key): value for key, value in extra_scope.items() if isinstance(key, str)})
    raw_values.update({str(key): value for key, value in values.items() if isinstance(key, str)})

    resolved_cache: Dict[str, Any] = {}
    resolving: set[str] = set()

    def _resolve_name(name: str) -> Any:
        if name in resolved_cache:
            return resolved_cache[name]
        if name not in raw_values:
            raise ValueError(f"Unknown config variable reference: {name}")
        if name in resolving:
            cycle = " -> ".join([*sorted(resolving), name])
            raise ValueError(f"Circular config variable reference detected: {cycle}")

        resolving.add(name)
        try:
            resolved_value = _resolve_value(raw_values[name])
        finally:
            resolving.remove(name)
        resolved_cache[name] = resolved_value
        return resolved_value

    def _resolve_string(value: str) -> Any:
        if not value:
            return value

        expanded = _BRACED_VAR_RE.sub(lambda match: str(_resolve_name(match.group(1))), value)
        bare_match = _BARE_VAR_PATH_RE.fullmatch(expanded)
        if bare_match:
            var_name = bare_match.group(1)
            suffix = bare_match.group(2) or ""
            if var_name in raw_values:
                base_value = _resolve_name(var_name)
                if suffix:
                    return f"{base_value}{suffix}"
                return base_value
        return expanded

    def _resolve_value(value: Any) -> Any:
        if isinstance(value, str):
            return _resolve_string(value)
        if isinstance(value, list):
            return [_resolve_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_resolve_value(item) for item in value)
        if isinstance(value, dict):
            return {key: _resolve_value(item) for key, item in value.items()}
        return value

    return {
        key: _resolve_name(key)
        for key in values.keys()
        if isinstance(key, str)
    }
