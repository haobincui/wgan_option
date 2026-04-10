"""Utilities for expanding small config variables inside surface-builder YAML."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


ROOT_DIR = Path(__file__).resolve().parents[4]


def resolve_config_path(path_value: str) -> Path:
    """Resolve a config path: absolute -> cwd-relative -> project-root-relative."""

    path = Path(path_value)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return ROOT_DIR / path


def parse_bool_from_config(value: Any, key: str) -> bool:
    """Require a real YAML boolean value in config payloads."""

    if isinstance(value, bool):
        return value
    raise ValueError(f"Invalid boolean value for `{key}` in config: {value!r}. Expected a YAML boolean.")


def load_yaml_mapping(config_path_value: str) -> tuple[Path, Dict[str, Any]]:
    """Load one YAML file and require its root to be a mapping."""

    config_path = resolve_config_path(config_path_value)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle) or {}
    if not isinstance(raw_data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return config_path, raw_data


def load_surface_builder_root(config_path_value: str) -> tuple[Path, Dict[str, Any]]:
    """Load one config file and return the `surface_builder` root mapping."""

    config_path, raw_data = load_yaml_mapping(config_path_value)
    if "surface_builder" not in raw_data:
        raise ValueError(f"Config file must contain a top-level `surface_builder` mapping: {config_path}")
    config_root = raw_data["surface_builder"]
    if not isinstance(config_root, dict):
        raise ValueError(f"`surface_builder` must be a mapping in config file: {config_path}")
    return config_path, config_root


def load_surface_builder_section(
    config_path_value: str,
    *,
    section_key: str,
    supported_keys: set[str],
    merge_root_supported_defaults: bool = False,
) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
    """Load one required `surface_builder` subsection."""

    config_path, config_root = load_surface_builder_root(config_path_value)
    root_supported = {key: value for key, value in config_root.items() if key in supported_keys}

    section = config_root.get(section_key)
    if section is None:
        raise ValueError(f"Missing `surface_builder.{section_key}` mapping in config file: {config_path}")
    if not isinstance(section, dict):
        raise ValueError(f"`{section_key}` must be a mapping in config file: {config_path}")
    unknown_keys = sorted(set(section.keys()) - supported_keys)
    if unknown_keys:
        raise ValueError(f"Unknown {section_key} config keys in {config_path}: {unknown_keys}")
    defaults = dict(section)
    if merge_root_supported_defaults:
        merged_defaults = dict(root_supported)
        merged_defaults.update(defaults)
        defaults = merged_defaults

    return config_path, config_root, defaults


def write_yaml_mapping(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    """Persist one YAML mapping to disk."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False, allow_unicode=False)
    return output

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
