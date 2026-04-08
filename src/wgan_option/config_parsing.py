"""Shared parsing helpers for script-facing config modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


def validate_config_keys(data: Mapping[str, Any], *, valid_keys: set[str], source: str) -> None:
    """Fail fast when unknown keys appear in yaml payloads or CLI overrides."""

    unknown_keys = sorted(set(data.keys()) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown config keys in {source}: {unknown_keys}")


def parse_cli_bool(raw: str) -> bool:
    """Parse strict CLI boolean formats."""

    value = raw.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Cannot parse boolean value from '{raw}'. Expected true or false.")


def cast_override(raw_value: str, default_value: Any) -> Any:
    """Cast a `--set key=value` payload using the default field type."""

    if isinstance(default_value, bool):
        return parse_cli_bool(raw_value)
    if isinstance(default_value, int):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    if isinstance(default_value, str):
        return raw_value
    raise TypeError(f"Unsupported override type: {type(default_value)}")


def parse_typed_overrides(
    override_items: Iterable[str],
    *,
    defaults: Mapping[str, Any],
    example: str,
) -> Dict[str, Any]:
    """Parse repeated `--set key=value` items into a typed override dict."""

    overrides: Dict[str, Any] = {}
    for item in override_items:
        if "=" not in item:
            raise ValueError(
                f"Override '{item}' is invalid. Expected KEY=VALUE, "
                f"example: {example}"
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key not in defaults:
            raise ValueError(f"Unknown override key: '{key}'")
        overrides[key] = cast_override(raw_value, defaults[key])
    return overrides


def load_yaml_config_values(
    config_path: str | Path,
    *,
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
    """Load one YAML config file and merge validated overrides onto defaults."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    loaded_values = dict(defaults)
    with path.open("r", encoding="utf-8") as handle:
        yaml_values = yaml.safe_load(handle) or {}
    if not isinstance(yaml_values, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    valid_keys = set(defaults.keys())
    validate_config_keys(yaml_values, valid_keys=valid_keys, source=str(path))
    loaded_values.update(yaml_values)

    if overrides:
        validate_config_keys(overrides, valid_keys=valid_keys, source="cli overrides")
        loaded_values.update(overrides)

    return path, yaml_values, loaded_values
