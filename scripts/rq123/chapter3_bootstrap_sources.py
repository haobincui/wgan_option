"""Frozen Chapter 3 sources and shared-market-panel bootstrap recipes.

This module is deliberately analysis-only.  It reads immutable pair-level
artifacts, converts their heterogeneous schemas to one canonical panel, and
describes the RQ1--RQ3/robustness jobs consumed by the v2 bootstrap CLI.  It
does not import any experiment lifecycle module and never writes to an old
experiment root.  RQ4 is intentionally outside the configured scope.

The canonical row grain is::

    condition, seed, fold, pair_id, session_id, value [, persistence_mae]

``value`` is pair-level masked MAE.  Persistence is also materialized as an
ordinary condition before it is used in a contrast.  Raw ``persistence_mae``
is retained as an audit field so crossed seed-by-market panels can be checked
before any bootstrap draw is made.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "configs/rq123/chapter3_shared_panel_bootstrap_v2.yaml"
)
CANONICAL_COLUMNS = (
    "condition",
    "seed",
    "fold",
    "pair_id",
    "session_id",
    "value",
)
OPTIONAL_CANONICAL_COLUMNS = ("persistence_mae",)
MARKET_KEY = ("fold", "pair_id", "session_id")
ROW_KEY = ("condition", "seed", *MARKET_KEY)
CROSS_SOURCE_ABS_TOL = 5e-15


class Chapter3BootstrapSourceError(RuntimeError):
    """Raised when frozen evidence, its schema, or a recipe drifts."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Chapter3BootstrapSourceError(f"{label} must be a mapping")
    return dict(value)


def _strict_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise Chapter3BootstrapSourceError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise Chapter3BootstrapSourceError(f"{label} must be non-negative")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _verify_hash(path: Path, expected: Any, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_text = str(expected or "").strip().lower()
    if len(expected_text) != 64:
        raise Chapter3BootstrapSourceError(f"{label} has no frozen SHA-256")
    observed = _sha256_file(path)
    if observed != expected_text:
        raise Chapter3BootstrapSourceError(
            f"Frozen SHA-256 drift for {label}: {observed} != {expected_text}"
        )
    return observed


def _csv_row_count(path: Path) -> int:
    try:
        return int(
            sum(
                len(chunk)
                for chunk in pd.read_csv(
                    path,
                    usecols=[0],
                    chunksize=100_000,
                    low_memory=False,
                )
            )
        )
    except (ValueError, pd.errors.EmptyDataError) as exc:
        raise Chapter3BootstrapSourceError(
            f"Could not count rows in frozen table: {path}"
        ) from exc


def _verify_expected_raw_values(
    frame: pd.DataFrame, expected: Mapping[str, Any], source_id: str
) -> None:
    for column, wanted_raw in expected.items():
        if column not in frame:
            raise Chapter3BootstrapSourceError(
                f"{source_id} is missing frozen enum column {column!r}"
            )
        wanted = list(wanted_raw or [])
        if wanted and all(isinstance(value, (int, float)) for value in wanted):
            observed_values = pd.to_numeric(frame[column], errors="raise")
            observed = set(observed_values.tolist())
            target = set(wanted)
        else:
            observed = set(frame[column].astype(str).tolist())
            target = {str(value) for value in wanted}
        if observed != target:
            raise Chapter3BootstrapSourceError(
                f"{source_id}.{column} drift: {sorted(observed, key=str)} "
                f"!= {sorted(target, key=str)}"
            )


def _filter_frame(
    frame: pd.DataFrame, filters: Mapping[str, Any], source_id: str
) -> pd.DataFrame:
    selected = frame
    for column, allowed_raw in filters.items():
        if column not in selected:
            raise Chapter3BootstrapSourceError(
                f"{source_id} filter column is missing: {column}"
            )
        allowed = list(allowed_raw or [])
        if not allowed:
            raise Chapter3BootstrapSourceError(
                f"{source_id} filter is empty: {column}"
            )
        if all(isinstance(value, (int, float)) for value in allowed):
            mask = pd.to_numeric(selected[column], errors="raise").isin(allowed)
        else:
            mask = selected[column].astype(str).isin(map(str, allowed))
        selected = selected.loc[mask]
    if selected.empty:
        raise Chapter3BootstrapSourceError(f"{source_id} adapter selected no rows")
    return selected.copy()


def _study_condition(
    model: Any, train_tolerance: Any, panel_role: Any, text_condition: Any
) -> str:
    tolerance = int(train_tolerance)
    return (
        f"{model}::train_{tolerance:02d}m::"
        f"{panel_role}::text_{text_condition}"
    )


def _validation_condition(arm: Any, checkpoint_label: Any) -> str:
    return f"{arm}::{checkpoint_label}"


def _adapt_source(
    source_id: str, frame: pd.DataFrame, spec: Mapping[str, Any]
) -> pd.DataFrame:
    adapter = str(spec.get("adapter", ""))
    selected = frame
    if spec.get("include"):
        selected = _filter_frame(
            selected, _mapping(spec["include"], f"{source_id}.include"), source_id
        )

    if adapter == "simple_condition":
        condition_column = str(spec.get("condition_column", ""))
        if condition_column not in selected:
            raise Chapter3BootstrapSourceError(
                f"{source_id} condition column is missing: {condition_column}"
            )
        conditions = selected[condition_column].astype(str)
        condition_map = {
            str(key): str(value)
            for key, value in _mapping(
                spec.get("condition_map", {}), f"{source_id}.condition_map"
            ).items()
        }
        conditions = conditions.replace(condition_map)
    elif adapter == "validation_checkpoint":
        required = {"arm", "checkpoint_label"}
        if required - set(selected):
            raise Chapter3BootstrapSourceError(
                f"{source_id} lacks validation selector columns"
            )
        conditions = pd.Series(
            (
                _validation_condition(arm, label)
                for arm, label in zip(
                    selected["arm"], selected["checkpoint_label"], strict=True
                )
            ),
            index=selected.index,
            dtype="object",
        )
    elif adapter == "study_selector":
        required = {
            "model_id",
            "train_tolerance_minutes",
            "panel_role",
            "text_condition",
        }
        if required - set(selected):
            raise Chapter3BootstrapSourceError(
                f"{source_id} lacks architecture/window selector columns"
            )
        conditions = pd.Series(
            (
                _study_condition(model, tolerance, role, condition)
                for model, tolerance, role, condition in zip(
                    selected["model_id"],
                    selected["train_tolerance_minutes"],
                    selected["panel_role"],
                    selected["text_condition"],
                    strict=True,
                )
            ),
            index=selected.index,
            dtype="object",
        )
    else:
        raise Chapter3BootstrapSourceError(
            f"Unsupported adapter for {source_id}: {adapter!r}"
        )

    required_raw = {"seed", "fold", "pair_id", "session_id"}
    value_column = str(spec.get("value_column", ""))
    persistence_column = str(spec.get("persistence_column", ""))
    required_raw.add(value_column)
    if persistence_column:
        required_raw.add(persistence_column)
    missing = sorted(required_raw - set(selected))
    if missing:
        raise Chapter3BootstrapSourceError(
            f"{source_id} is missing required raw columns: {missing}"
        )

    seed_numeric = pd.to_numeric(selected["seed"], errors="raise")
    if not np.isfinite(seed_numeric.to_numpy(dtype=float)).all() or not np.equal(
        seed_numeric.to_numpy(dtype=float),
        seed_numeric.to_numpy(dtype=np.int64).astype(float),
    ).all():
        raise Chapter3BootstrapSourceError(f"{source_id} seed values are not integers")

    canonical = pd.DataFrame(
        {
            "condition": conditions.to_numpy(dtype=object),
            "seed": seed_numeric.to_numpy(dtype=np.int64),
            "fold": selected["fold"].astype(str).to_numpy(),
            "pair_id": selected["pair_id"].astype(str).to_numpy(),
            "session_id": selected["session_id"].astype(str).to_numpy(),
            "value": pd.to_numeric(
                selected[value_column], errors="raise"
            ).to_numpy(dtype=float),
        }
    )
    if persistence_column:
        canonical["persistence_mae"] = pd.to_numeric(
            selected[persistence_column], errors="raise"
        ).to_numpy(dtype=float)
    return canonical


def _validate_canonical_rows(frame: pd.DataFrame, label: str) -> None:
    missing = sorted(set(CANONICAL_COLUMNS) - set(frame))
    if missing:
        raise Chapter3BootstrapSourceError(
            f"{label} lacks canonical columns: {missing}"
        )
    if frame.empty:
        raise Chapter3BootstrapSourceError(f"{label} canonical panel is empty")
    for column in ("condition", "fold", "pair_id", "session_id"):
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise Chapter3BootstrapSourceError(
                f"{label}.{column} contains missing/empty identifiers"
            )
    seed = pd.to_numeric(frame["seed"], errors="raise")
    if not np.equal(
        seed.to_numpy(dtype=float), seed.to_numpy(dtype=np.int64).astype(float)
    ).all():
        raise Chapter3BootstrapSourceError(f"{label}.seed is not integral")
    frame["seed"] = seed.astype(np.int64)
    values = pd.to_numeric(frame["value"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise Chapter3BootstrapSourceError(
            f"{label}.value must be finite and strictly positive"
        )
    frame["value"] = values
    if "persistence_mae" in frame:
        persistence = pd.to_numeric(
            frame["persistence_mae"], errors="raise"
        ).to_numpy(dtype=float)
        if not np.isfinite(persistence).all() or (persistence <= 0.0).any():
            raise Chapter3BootstrapSourceError(
                f"{label}.persistence_mae must be finite and strictly positive"
            )
        frame["persistence_mae"] = persistence
    if frame.duplicated(list(ROW_KEY)).any():
        examples = frame.loc[frame.duplicated(list(ROW_KEY), keep=False), list(ROW_KEY)]
        raise Chapter3BootstrapSourceError(
            f"{label} has duplicate canonical rows; first examples: "
            f"{examples.head(3).to_dict(orient='records')}"
        )


def _materialize_persistence(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "persistence" in set(frame["condition"].astype(str)):
        return frame.copy()
    if "persistence_mae" not in frame:
        raise Chapter3BootstrapSourceError(
            f"{label} cannot materialize persistence without persistence_mae"
        )
    keys = ["seed", *MARKET_KEY]
    grouped = frame.groupby(keys, sort=False, dropna=False)["persistence_mae"]
    if grouped.nunique(dropna=False).gt(1).any():
        raise Chapter3BootstrapSourceError(
            f"{label} persistence differs across paired conditions"
        )
    persistence = (
        grouped.first().reset_index().rename(columns={"persistence_mae": "value"})
    )
    persistence.insert(0, "condition", "persistence")
    persistence["persistence_mae"] = persistence["value"]
    result = pd.concat([frame, persistence], ignore_index=True, sort=False)
    _validate_canonical_rows(result, label)
    return result


def _assert_shared_panel(frame: pd.DataFrame, label: str) -> None:
    """Require one identical market panel (and persistence) for every seed/arm."""

    audit_columns = list(MARKET_KEY)
    if "persistence_mae" in frame:
        audit_columns.append("persistence_mae")
    reference: pd.DataFrame | None = None
    reference_label = ""
    for (condition, seed), group in frame.groupby(
        ["condition", "seed"], sort=True, dropna=False
    ):
        panel = group[audit_columns].sort_values(list(MARKET_KEY), kind="stable")
        panel = panel.reset_index(drop=True)
        if reference is None:
            reference = panel
            reference_label = f"{condition}/{seed}"
        elif not panel.equals(reference):
            raise Chapter3BootstrapSourceError(
                f"{label} market panel differs for {condition}/{seed}; "
                f"reference={reference_label}"
            )


def _assert_cross_source_equivalent(
    sources: Mapping[str, pd.DataFrame],
    left_source: str,
    left_condition: str,
    right_source: str,
    right_condition: str,
    *,
    atol: float = CROSS_SOURCE_ABS_TOL,
) -> None:
    """Bind duplicated frozen predictions to one market lineage and value.

    Some chapter tables persist the same prediction rows in more than one
    artifact.  This check prevents those copies from silently diverging while
    allowing only sub-CSV-roundoff noise in numeric values.
    """

    columns = ["seed", *MARKET_KEY, "value", "persistence_mae"]

    def select(source_id: str, condition: str) -> pd.DataFrame:
        if source_id not in sources:
            raise Chapter3BootstrapSourceError(
                f"Cross-source audit is missing source: {source_id}"
            )
        frame = sources[source_id]
        missing = sorted(set(columns) - set(frame))
        if missing:
            raise Chapter3BootstrapSourceError(
                f"Cross-source audit {source_id} is missing columns: {missing}"
            )
        selected = frame.loc[frame["condition"].astype(str).eq(condition), columns]
        if selected.empty:
            raise Chapter3BootstrapSourceError(
                f"Cross-source audit selected no rows: {source_id}/{condition}"
            )
        return selected.sort_values(columns[:4], kind="stable").reset_index(drop=True)

    left = select(left_source, left_condition)
    right = select(right_source, right_condition)
    label = (
        f"{left_source}/{left_condition} vs "
        f"{right_source}/{right_condition}"
    )
    if len(left) != len(right) or not left[columns[:4]].equals(right[columns[:4]]):
        raise Chapter3BootstrapSourceError(
            f"Cross-source market lineage drift: {label}"
        )
    for column in ("value", "persistence_mae"):
        left_values = left[column].to_numpy(dtype=float)
        right_values = right[column].to_numpy(dtype=float)
        if not np.allclose(left_values, right_values, rtol=0.0, atol=atol):
            maximum = float(np.max(np.abs(left_values - right_values)))
            raise Chapter3BootstrapSourceError(
                f"Cross-source {column} drift: {label}; "
                f"max_abs_difference={maximum:.17g}, tolerance={atol:.17g}"
            )


def _audit_cross_source_lineage(sources: Mapping[str, pd.DataFrame]) -> None:
    # The direct and RQ3 analyses evaluate the same test market; CSV writers
    # differ only at roughly 1e-16 for the duplicated persistence values.
    _assert_cross_source_equivalent(
        sources, "direct", "persistence", "rq3_full", "persistence"
    )
    # The branch table is an exact subset of the full seven-arm evaluation.
    for condition in ("film_lp_matched", "film_lp_shuffle", "film_zero_text"):
        _assert_cross_source_equivalent(
            sources, "rq3_full", condition, "rq3_branch", condition
        )
    # matched_input is the unmodified prediction copied into the intervention
    # table; wrong/zero input are the actual interventions.
    _assert_cross_source_equivalent(
        sources,
        "rq3_full",
        "film_lp_matched",
        "rq3_intervention",
        "matched_input",
    )
    # Architecture and alignment studies reuse the same three-seed reference
    # predictions on the common five-minute panel under different labels.
    _assert_cross_source_equivalent(
        sources,
        "architecture",
        _study_condition("film_reference", 5, "common_5m_primary", "matched"),
        "alignment",
        _study_condition("film_lp_matched", 5, "common_5m_primary", "matched"),
    )
    _assert_cross_source_equivalent(
        sources,
        "architecture",
        _study_condition("pure_cnn_reference", 5, "common_5m_primary", "zero"),
        "alignment",
        _study_condition("pure_cnn_no_text", 5, "common_5m_primary", "zero"),
    )


def _manifest_entry(
    source_id: str,
    role: str,
    path: Path,
    sha256: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "role": role,
        "path": _display_path(path),
        "sha256": sha256,
        "size_bytes": int(path.stat().st_size),
        **extra,
    }


def _verify_support_artifact(
    source_id: str, spec: Mapping[str, Any]
) -> dict[str, Any]:
    path = _resolve_path(str(spec.get("path", "")))
    observed_hash = _verify_hash(path, spec.get("sha256"), source_id)
    expected_rows = spec.get("expected_rows")
    extra: dict[str, Any] = {}
    if expected_rows is not None:
        observed_rows = _csv_row_count(path)
        if observed_rows != int(expected_rows):
            raise Chapter3BootstrapSourceError(
                f"{source_id} row-count drift: {observed_rows} != {expected_rows}"
            )
        extra["row_count"] = observed_rows
    if spec.get("json_require") is not None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Chapter3BootstrapSourceError(
                f"{source_id} is not valid frozen JSON"
            ) from exc
        required = _mapping(spec["json_require"], f"{source_id}.json_require")
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise Chapter3BootstrapSourceError(
                    f"{source_id} JSON assertion failed: {key}="
                    f"{payload.get(key)!r}, expected {expected!r}"
                )
        extra["json_require"] = required
    return _manifest_entry(
        source_id,
        str(spec.get("role", "support")),
        path,
        observed_hash,
        **extra,
    )


def load_sources(
    config_path: str | Path = DEFAULT_CONFIG,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], dict[str, Any]]:
    """Load, hash-check, adapt, and audit all frozen pair-level sources.

    Returns ``(sources, input_manifest_entries, config)``.  Data-frame values
    are canonical panels; manifest entries also cover the legacy comparison,
    QA, and descriptive pass-through artifacts.  No output directory is
    created and no file is modified.
    """

    config_file = _resolve_path(config_path)
    config_hash = _verify_hash(config_file, _sha256_file(config_file), "config")
    raw_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config = _mapping(raw_config, "Chapter 3 bootstrap config")
    if int(config.get("schema_version", -1)) != 2 or config.get("kind") != (
        "chapter3_shared_market_panel_bootstrap_v2"
    ):
        raise Chapter3BootstrapSourceError("Unsupported Chapter 3 config contract")
    bootstrap = _mapping(config.get("bootstrap"), "bootstrap")
    iterations = _strict_nonnegative_integer(
        bootstrap.get("iterations"), "bootstrap.iterations"
    )
    if iterations != 10_000:
        raise Chapter3BootstrapSourceError("Formal correction requires 10,000 draws")
    _strict_nonnegative_integer(bootstrap.get("rng_seed"), "bootstrap.rng_seed")
    confidence_level = bootstrap.get("confidence_level")
    if (
        isinstance(confidence_level, (bool, np.bool_))
        or not isinstance(
            confidence_level, (int, float, np.integer, np.floating)
        )
        or not math.isfinite(float(confidence_level))
        or float(confidence_level) != 0.95
    ):
        raise Chapter3BootstrapSourceError(
            "bootstrap.confidence_level must be exactly 0.95"
        )
    if str(bootstrap.get("point_estimate")) != "original_sample_plugin":
        raise Chapter3BootstrapSourceError("Bootstrap point-estimate contract drift")
    for dimension in ("seed", "fold", "session"):
        if str(bootstrap.get(f"{dimension}_resampling")) != "with_replacement":
            raise Chapter3BootstrapSourceError(
                f"bootstrap.{dimension}_resampling must be with_replacement"
            )
    if str(bootstrap.get("market_weights")) != (
        "shared_across_all_sampled_seeds_and_paired_conditions"
    ):
        raise Chapter3BootstrapSourceError("Shared-market bootstrap contract drift")
    if str(bootstrap.get("resampling_contract")) != (
        "crossed_seed_by_market_panel_shared_fold_session_weights_v2"
    ):
        raise Chapter3BootstrapSourceError("Crossed resampling contract drift")
    excluded = set(_mapping(config.get("scope"), "scope").get("excluded") or [])
    if "rq4" not in excluded:
        raise Chapter3BootstrapSourceError("RQ4 must remain excluded from this run")

    manifest: list[dict[str, Any]] = [
        _manifest_entry("config", "config", config_file, config_hash)
    ]
    sources: dict[str, pd.DataFrame] = {}
    source_specs = _mapping(config.get("sources"), "sources")
    for source_id, raw_spec in source_specs.items():
        spec = _mapping(raw_spec, f"sources.{source_id}")
        path = _resolve_path(str(spec.get("path", "")))
        observed_hash = _verify_hash(path, spec.get("sha256"), source_id)
        raw = pd.read_csv(
            path,
            dtype={"pair_id": "string", "session_id": "string"},
            low_memory=False,
        )
        if len(raw) != int(spec.get("expected_rows", -1)):
            raise Chapter3BootstrapSourceError(
                f"{source_id} raw row-count drift: {len(raw)} != "
                f"{spec.get('expected_rows')}"
            )
        _verify_expected_raw_values(
            raw,
            _mapping(
                spec.get("expected_raw_values", {}),
                f"sources.{source_id}.expected_raw_values",
            ),
            str(source_id),
        )
        adapted = _adapt_source(str(source_id), raw, spec)
        expected_adapted = spec.get("expected_adapted_data_rows")
        if expected_adapted is not None and len(adapted) != int(expected_adapted):
            raise Chapter3BootstrapSourceError(
                f"{source_id} adapted row-count drift: {len(adapted)} != "
                f"{expected_adapted}"
            )
        _validate_canonical_rows(adapted, str(source_id))
        if bool(spec.get("add_persistence_condition", False)):
            adapted = _materialize_persistence(adapted, str(source_id))
        if len(adapted) != int(spec.get("expected_canonical_rows", -1)):
            raise Chapter3BootstrapSourceError(
                f"{source_id} canonical row-count drift: {len(adapted)} != "
                f"{spec.get('expected_canonical_rows')}"
            )
        if adapted["condition"].nunique() != int(
            spec.get("expected_condition_count", -1)
        ):
            raise Chapter3BootstrapSourceError(
                f"{source_id} canonical condition-count drift"
            )
        if adapted["seed"].nunique() != int(spec.get("expected_seed_count", -1)):
            raise Chapter3BootstrapSourceError(f"{source_id} seed-count drift")
        if adapted["fold"].nunique() != int(spec.get("expected_fold_count", -1)):
            raise Chapter3BootstrapSourceError(f"{source_id} fold-count drift")
        if str(spec.get("panel_contract")) == "shared":
            _assert_shared_panel(adapted, str(source_id))
        elif str(spec.get("panel_contract")) != "mixed":
            raise Chapter3BootstrapSourceError(
                f"{source_id} has unsupported panel contract"
            )
        adapted = adapted.sort_values(list(ROW_KEY), kind="stable").reset_index(
            drop=True
        )
        sources[str(source_id)] = adapted
        manifest.append(
            _manifest_entry(
                str(source_id),
                str(spec.get("role", "data")),
                path,
                observed_hash,
                raw_row_count=int(len(raw)),
                canonical_row_count=int(len(adapted)),
                condition_count=int(adapted["condition"].nunique()),
                seed_count=int(adapted["seed"].nunique()),
                fold_count=int(adapted["fold"].nunique()),
                panel_contract=str(spec.get("panel_contract")),
            )
        )

    _audit_cross_source_lineage(sources)

    for section in ("legacy_sources", "qa_artifacts", "passthrough_artifacts"):
        for source_id, raw_spec in _mapping(config.get(section, {}), section).items():
            manifest.append(
                _verify_support_artifact(
                    f"{section}:{source_id}", _mapping(raw_spec, f"{section}.{source_id}")
                )
            )

    config["_config_path"] = _display_path(config_file)
    config["_config_sha256"] = config_hash
    config["_repo_root"] = str(REPO_ROOT)
    return sources, manifest, config


def _legacy_tables(config: Mapping[str, Any]) -> dict[str, tuple[pd.DataFrame, str]]:
    tables: dict[str, tuple[pd.DataFrame, str]] = {}
    for source_id, raw_spec in _mapping(
        config.get("legacy_sources", {}), "legacy_sources"
    ).items():
        spec = _mapping(raw_spec, f"legacy_sources.{source_id}")
        path = _resolve_path(str(spec.get("path", "")))
        _verify_hash(path, spec.get("sha256"), f"legacy:{source_id}")
        frame = pd.read_csv(path, low_memory=False)
        if len(frame) != int(spec.get("expected_rows", -1)):
            raise Chapter3BootstrapSourceError(
                f"legacy:{source_id} row-count drift"
            )
        tables[str(source_id)] = (frame, _display_path(path))
    return tables


def _nullable_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _legacy_lookup(
    tables: Mapping[str, tuple[pd.DataFrame, str]],
    table_id: str,
    filters: Mapping[str, Any],
    *,
    point: str,
    se: str | None,
    ci_lower: str | None,
    ci_upper: str | None,
    p_one: str | None,
    p_two: str | None,
    holm_p: str | None,
) -> dict[str, Any]:
    if table_id not in tables:
        raise Chapter3BootstrapSourceError(f"Unknown legacy table: {table_id}")
    frame, path = tables[table_id]
    selected = frame
    fragments: list[str] = []
    for column, expected in filters.items():
        if column not in selected:
            raise Chapter3BootstrapSourceError(
                f"Legacy selector column is missing: {table_id}.{column}"
            )
        if isinstance(expected, (int, float)):
            mask = pd.to_numeric(selected[column], errors="coerce").eq(expected)
        else:
            mask = selected[column].astype(str).eq(str(expected))
        selected = selected.loc[mask]
        fragments.append(f"{column}={expected}")
    if len(selected) != 1:
        raise Chapter3BootstrapSourceError(
            f"Legacy selector is not one-to-one: {table_id}/{filters}; rows={len(selected)}"
        )
    row = selected.iloc[0]

    def get(column: str | None) -> float | None:
        if column is None:
            return None
        if column not in row.index:
            raise Chapter3BootstrapSourceError(
                f"Legacy value column is missing: {table_id}.{column}"
            )
        return _nullable_float(row[column])

    return {
        "point": get(point),
        "se": get(se),
        "ci_lower": get(ci_lower),
        "ci_upper": get(ci_upper),
        "p_one": get(p_one),
        "p_two": get(p_two),
        "holm_p": get(holm_p),
        "source": path + "#" + "&".join(fragments),
    }


def _standard_legacy(
    tables: Mapping[str, tuple[pd.DataFrame, str]],
    table_id: str,
    filters: Mapping[str, Any],
    *,
    holm_p: str | None = "holm_adjusted_p",
) -> dict[str, Any]:
    return _legacy_lookup(
        tables,
        table_id,
        filters,
        point="mean_log_mae_ratio",
        se="bootstrap_se",
        ci_lower="ci_95_lower",
        ci_upper="ci_95_upper",
        p_one="p_value_one_sided",
        p_two="p_value_two_sided",
        holm_p=holm_p,
    )


def _architecture_legacy(
    tables: Mapping[str, tuple[pd.DataFrame, str]], table_id: str, contrast_id: str
) -> dict[str, Any]:
    return _legacy_lookup(
        tables,
        table_id,
        {"contrast_id": contrast_id},
        point="log_mae_ratio",
        se=None,
        ci_lower="ci_lower",
        ci_upper="ci_upper",
        p_one="p_value_one_sided",
        p_two=None,
        holm_p="holm_p_value",
    )


def _contrast(
    contrast_id: str,
    focal: str,
    reference: str,
    family_id: str,
    *,
    apply_holm: bool,
    legacy: dict[str, Any] | None = None,
    support_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "contrast_id": str(contrast_id),
        "focal": str(focal),
        "reference": str(reference),
        "family_id": str(family_id),
        "alternative": "one_sided",
        "apply_holm": bool(apply_holm),
        "legacy": legacy,
    }
    if support_gate is not None:
        result["support_gate"] = dict(support_gate)
    return result


def _gate(*, seeds: int, folds: int) -> dict[str, Any]:
    return {
        "alpha": 0.05,
        "minimum_nonworse_seeds": int(seeds),
        "minimum_nonworse_folds": int(folds),
        "require_ci_below_zero": True,
    }


def _ensure_persistence(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    result = panel.copy()
    _validate_canonical_rows(result, label)
    if "persistence" not in set(result["condition"].astype(str)):
        result = _materialize_persistence(result, label)
    _assert_shared_panel(result, label)
    columns = [*CANONICAL_COLUMNS]
    if "persistence_mae" in result:
        columns.append("persistence_mae")
    return result[columns].sort_values(list(ROW_KEY), kind="stable").reset_index(
        drop=True
    )


def _job(
    job_id: str,
    panel: pd.DataFrame,
    estimand: str,
    contrasts: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if estimand not in {"equal_cell", "pooled_pair"}:
        raise Chapter3BootstrapSourceError(f"Unsupported estimand: {estimand}")
    canonical = _ensure_persistence(panel, job_id)
    conditions = set(canonical["condition"].astype(str))
    normalized = [dict(contrast) for contrast in contrasts]
    for contrast in normalized:
        if contrast["focal"] not in conditions or contrast["reference"] not in conditions:
            raise Chapter3BootstrapSourceError(
                f"{job_id}/{contrast['contrast_id']} references absent conditions"
            )
    return {
        "job_id": str(job_id),
        "panel": canonical,
        "estimand": estimand,
        "contrasts": normalized,
        "metadata": dict(metadata),
    }


def _direct_jobs(
    sources: Mapping[str, pd.DataFrame],
    recipes: Mapping[str, Any],
    legacy: Mapping[str, tuple[pd.DataFrame, str]],
) -> list[dict[str, Any]]:
    frame = sources["direct"]
    direct = _mapping(recipes["direct"], "recipes.direct")
    folds = [str(value) for value in recipes["folds"]]
    arms = [str(value) for value in direct["model_arms"]]
    jobs: list[dict[str, Any]] = []
    for fold in folds:
        contrasts: list[dict[str, Any]] = []
        for arm in arms:
            contrasts.append(
                _contrast(
                    f"direct_{fold}_{arm}_vs_persistence",
                    arm,
                    "persistence",
                    f"direct_{fold}_model_vs_persistence_holm5",
                    apply_holm=True,
                    legacy=_standard_legacy(
                        legacy,
                        "direct_fold_persistence",
                        {"fold": fold, "comparison_id": f"{arm}_vs_persistence"},
                    ),
                    support_gate=_gate(seeds=7, folds=1),
                )
            )
        rq1_legacy = None
        if fold == "f4_2023q4":
            # The old artifact used a Holm-1 RQ1 family; retain its raw fields
            # but do not relabel that adjustment as the new rolling Holm-4.
            rq1_legacy = _standard_legacy(
                legacy,
                "direct_q4_primary",
                {"fold": fold, "comparison_id": "lp_matched_vs_no_text"},
                holm_p=None,
            )
        contrasts.append(
            _contrast(
                f"rq1_{fold}_lp_matched_vs_no_text",
                str(direct["rq1_focal"]),
                str(direct["rq1_reference"]),
                "rq1_lp_matched_vs_no_text_rolling_holm4",
                apply_holm=True,
                legacy=rq1_legacy,
                support_gate=_gate(seeds=7, folds=1) if rq1_legacy else None,
            )
        )
        for reference in map(str, direct["rq2_references"]):
            rq2_legacy = None
            if fold == "f4_2023q4":
                rq2_legacy = _standard_legacy(
                    legacy,
                    "direct_q4_primary",
                    {
                        "fold": fold,
                        "comparison_id": f"lp_matched_vs_{reference}",
                    },
                )
            contrasts.append(
                _contrast(
                    f"rq2_{fold}_lp_matched_vs_{reference}",
                    str(direct["rq1_focal"]),
                    reference,
                    f"rq2_representations_{fold}_holm2",
                    apply_holm=True,
                    legacy=rq2_legacy,
                    support_gate=_gate(seeds=7, folds=1) if rq2_legacy else None,
                )
            )
        jobs.append(
            _job(
                f"direct_{fold}",
                frame.loc[frame["fold"].astype(str).eq(fold)],
                "equal_cell",
                contrasts,
                {
                    "scope": "rq1_rq2",
                    "section": "direct_rolling_fold",
                    "fold": fold,
                    "source_id": "direct",
                },
            )
        )

    overall: list[dict[str, Any]] = []
    for arm in arms:
        overall.append(
            _contrast(
                f"direct_overall_{arm}_vs_persistence",
                arm,
                "persistence",
                "direct_overall_model_vs_persistence_holm5",
                apply_holm=True,
                legacy=_legacy_lookup(
                    legacy,
                    "direct_overall_persistence",
                    {"comparison_id": f"{arm}_vs_persistence"},
                    point="mean_log_mae_ratio",
                    se="log_ratio_bootstrap_se",
                    ci_lower="log_ratio_ci_95_lower",
                    ci_upper="log_ratio_ci_95_upper",
                    p_one="p_value_one_sided_sign_tail",
                    p_two="p_value_two_sided_sign_tail",
                    holm_p="holm5_adjusted_p",
                ),
                support_gate=_gate(seeds=7, folds=3),
            )
        )
    overall.append(
        _contrast(
            "rq1_overall_lp_matched_vs_no_text",
            str(direct["rq1_focal"]),
            str(direct["rq1_reference"]),
            "rq1_lp_matched_vs_no_text_overall_unadjusted",
            apply_holm=False,
        )
    )
    for reference in map(str, direct["rq2_references"]):
        overall.append(
            _contrast(
                f"rq2_overall_lp_matched_vs_{reference}",
                str(direct["rq1_focal"]),
                reference,
                "rq2_representations_overall_holm2",
                apply_holm=True,
            )
        )
    jobs.append(
        _job(
            "direct_overall",
            frame,
            "equal_cell",
            overall,
            {
                "scope": "rq1_rq2",
                "section": "direct_four_fold_overall",
                "fold": "overall",
                "source_id": "direct",
            },
        )
    )
    return jobs


def _rq3_jobs(
    sources: Mapping[str, pd.DataFrame],
    recipes: Mapping[str, Any],
    legacy: Mapping[str, tuple[pd.DataFrame, str]],
) -> list[dict[str, Any]]:
    rq3 = _mapping(recipes["rq3"], "recipes.rq3")
    jobs = [
        _job(
            "rq3_full_absolute_mae",
            sources["rq3_full"],
            "equal_cell",
            [],
            {
                "scope": "rq3",
                "section": "full_seven_arm_absolute_mae",
                "fold": "overall",
                "source_id": "rq3_full",
                "inference": "descriptive_no_new_family",
            },
        )
    ]
    branch: list[dict[str, Any]] = []
    focal = str(rq3["branch_focal"])
    for reference in map(str, rq3["branch_references"]):
        old_id = (
            "matched_vs_film_lp_shuffle"
            if reference == "film_lp_shuffle"
            else "matched_vs_film_zero_text"
        )
        branch.append(
            _contrast(
                old_id,
                focal,
                reference,
                "rq3_branch_holm2",
                apply_holm=True,
                legacy=_standard_legacy(
                    legacy, "rq3_main", {"comparison_id": old_id}
                ),
                support_gate=_gate(seeds=7, folds=3),
            )
        )
    jobs.append(
        _job(
            "rq3_branch",
            sources["rq3_branch"],
            "equal_cell",
            branch,
            {
                "scope": "rq3",
                "section": "common_parent_branch_inference",
                "fold": "overall",
                "source_id": "rq3_branch",
            },
        )
    )
    intervention: list[dict[str, Any]] = []
    focal = str(rq3["intervention_focal"])
    for reference in map(str, rq3["intervention_references"]):
        old_id = f"matched_input_vs_{reference}"
        intervention.append(
            _contrast(
                old_id,
                focal,
                reference,
                "rq3_intervention_holm2",
                apply_holm=True,
                legacy=_standard_legacy(
                    legacy, "rq3_intervention", {"comparison_id": old_id}
                ),
                support_gate=_gate(seeds=7, folds=3),
            )
        )
    jobs.append(
        _job(
            "rq3_intervention",
            sources["rq3_intervention"],
            "equal_cell",
            intervention,
            {
                "scope": "rq3",
                "section": "same_checkpoint_intervention_inference",
                "fold": "overall",
                "source_id": "rq3_intervention",
            },
        )
    )

    validation: list[dict[str, Any]] = []
    for arm in map(str, rq3["validation_arms"]):
        validation.append(
            _contrast(
                f"validation_{arm}_epoch0_to_epoch30",
                _validation_condition(arm, "epoch_30"),
                _validation_condition(arm, "epoch_0"),
                "rq3_validation_epoch0_to_epoch30_holm4",
                apply_holm=True,
            )
        )
        validation.append(
            _contrast(
                f"validation_{arm}_epoch30_to_best",
                _validation_condition(arm, "best"),
                _validation_condition(arm, "epoch_30"),
                "rq3_validation_epoch30_to_best_holm4",
                apply_holm=True,
            )
        )
    jobs.append(
        _job(
            "rq3_validation_epoch0_epoch30_best",
            sources["rq3_validation"],
            "equal_cell",
            validation,
            {
                "scope": "rq3",
                "section": "validation_trajectory_diagnostic",
                "fold": "overall",
                "source_id": "rq3_validation",
                "post_selection": True,
            },
        )
    )
    return jobs


def _architecture_job(
    sources: Mapping[str, pd.DataFrame],
    recipes: Mapping[str, Any],
    legacy: Mapping[str, tuple[pd.DataFrame, str]],
) -> dict[str, Any]:
    recipe = _mapping(recipes["architecture"], "recipes.architecture")
    tolerance = int(recipe["train_tolerance_minutes"])
    role = str(recipe["panel_role"])

    def condition(model: str, text: str) -> str:
        return _study_condition(model, tolerance, role, text)

    contrasts: list[dict[str, Any]] = []
    for mode in map(str, recipe["formal_modes"]):
        old_id = f"{mode}_vs_film_reference"
        contrasts.append(
            _contrast(
                old_id,
                condition(f"formal:{mode}", "matched"),
                condition("film_reference", "matched"),
                "architecture_primary_new_modes_vs_film_holm3",
                apply_holm=True,
                legacy=_architecture_legacy(legacy, "architecture", old_id),
            )
        )
        old_id = f"{mode}_vs_pure_cnn_reference"
        contrasts.append(
            _contrast(
                old_id,
                condition(f"formal:{mode}", "matched"),
                condition("pure_cnn_reference", "zero"),
                "architecture_vs_pure_cnn_descriptive",
                apply_holm=False,
                legacy=_architecture_legacy(legacy, "architecture", old_id),
            )
        )
    old_id = "scaled_point_leader_vs_scaled_film"
    contrasts.append(
        _contrast(
            old_id,
            condition("scaled:validation_point_leader_scaled", "matched"),
            condition("scaled:film_reference_scaled", "matched"),
            "architecture_scaled_descriptive",
            apply_holm=False,
            legacy=_architecture_legacy(legacy, "architecture", old_id),
        )
    )
    for model in sorted(map(str, recipe["conditional_models"])):
        for reference_text in ("zero", "shuffle"):
            old_id = f"{model}_matched_vs_{reference_text}"
            contrasts.append(
                _contrast(
                    old_id,
                    condition(model, "matched"),
                    condition(model, reference_text),
                    f"architecture_text_reliance_{model}_holm2",
                    apply_holm=True,
                    legacy=_architecture_legacy(legacy, "architecture", old_id),
                )
            )
    return _job(
        "generator_architecture_robustness",
        sources["architecture"],
        "pooled_pair",
        contrasts,
        {
            "scope": "robustness",
            "section": "generator_architecture",
            "fold": "overall",
            "source_id": "architecture",
            "legacy_selector_contract": "architecture_window_study_analysis_v1",
        },
    )


def _alignment_jobs(
    sources: Mapping[str, pd.DataFrame],
    recipes: Mapping[str, Any],
    legacy: Mapping[str, tuple[pd.DataFrame, str]],
) -> list[dict[str, Any]]:
    recipe = _mapping(recipes["alignment"], "recipes.alignment")
    common_role = str(recipe["common_panel_role"])
    own_role = str(recipe["own_panel_role"])
    tolerances = [int(value) for value in recipe["tolerances_minutes"]]

    def condition(model: str, tolerance: int, role: str, text: str) -> str:
        return _study_condition(model, tolerance, role, text)

    common_conditions: set[str] = set()
    contrasts: list[dict[str, Any]] = []
    for tolerance in tolerances:
        film = condition("film_lp_matched", tolerance, common_role, "matched")
        film_zero = condition("film_lp_matched", tolerance, common_role, "zero")
        pure = condition("pure_cnn_no_text", tolerance, common_role, "zero")
        common_conditions.update((film, film_zero, pure))
        old_id = f"common5_train{tolerance:02d}_film_vs_pure_cnn"
        contrasts.append(
            _contrast(
                old_id,
                film,
                pure,
                "window_between_models_common5_holm5",
                apply_holm=True,
                legacy=_architecture_legacy(legacy, "alignment", old_id),
            )
        )
        old_id = f"common5_train{tolerance:02d}_film_matched_vs_zero"
        contrasts.append(
            _contrast(
                old_id,
                film,
                film_zero,
                "window_matched_vs_zero_descriptive",
                apply_holm=False,
                legacy=_architecture_legacy(legacy, "alignment", old_id),
            )
        )
    for model, text in (
        ("film_lp_matched", "matched"),
        ("pure_cnn_no_text", "zero"),
    ):
        for tolerance in (10, 15, 20, 30):
            old_id = f"common5_{model}_train{tolerance:02d}_vs_train05"
            contrasts.append(
                _contrast(
                    old_id,
                    condition(model, tolerance, common_role, text),
                    condition(model, 5, common_role, text),
                    f"window_within_{model}_common5_holm4",
                    apply_holm=True,
                    legacy=_architecture_legacy(legacy, "alignment", old_id),
                )
            )
    source = sources["alignment"]
    common_panel = source.loc[source["condition"].astype(str).isin(common_conditions)]
    jobs = [
        _job(
            "alignment_common_5m_panel",
            common_panel,
            "pooled_pair",
            contrasts,
            {
                "scope": "robustness",
                "section": "alignment_common_five_minute_panel",
                "fold": "overall",
                "source_id": "alignment",
                "legacy_selector_contract": "architecture_window_study_analysis_v1",
            },
        )
    ]
    for tolerance in map(int, recipe["own_panel_tolerances_minutes"]):
        wanted = {
            condition("film_lp_matched", tolerance, own_role, "matched"),
            condition("film_lp_matched", tolerance, own_role, "zero"),
            condition("pure_cnn_no_text", tolerance, own_role, "zero"),
        }
        panel = source.loc[source["condition"].astype(str).isin(wanted)]
        old_id = f"own_panel{tolerance:02d}_film_vs_pure_cnn"
        jobs.append(
            _job(
                f"alignment_own_panel_{tolerance:02d}m",
                panel,
                "pooled_pair",
                [
                    _contrast(
                        old_id,
                        condition("film_lp_matched", tolerance, own_role, "matched"),
                        condition("pure_cnn_no_text", tolerance, own_role, "zero"),
                        "window_own_panel_between_models_descriptive",
                        apply_holm=False,
                        legacy=_architecture_legacy(legacy, "alignment", old_id),
                    )
                ],
                {
                    "scope": "robustness",
                    "section": "alignment_own_tolerance_panel",
                    "fold": "overall",
                    "source_id": "alignment",
                    "training_tolerance_minutes": tolerance,
                    "descriptive_coverage_panel": True,
                },
            )
        )
    return jobs


def _audit_jobs(jobs: Sequence[Mapping[str, Any]]) -> None:
    job_ids = [str(job["job_id"]) for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise Chapter3BootstrapSourceError("Duplicate Chapter 3 bootstrap job_id")
    contrasts = [
        dict(contrast) for job in jobs for contrast in list(job["contrasts"])
    ]
    contrast_ids = [str(row["contrast_id"]) for row in contrasts]
    if len(contrast_ids) != len(set(contrast_ids)):
        raise Chapter3BootstrapSourceError("Duplicate Chapter 3 contrast_id")
    counts: dict[str, int] = {}
    for row in contrasts:
        if bool(row["apply_holm"]):
            family = str(row["family_id"])
            counts[family] = counts.get(family, 0) + 1
    for family, count in counts.items():
        declared = re.search(r"(?:^|_)holm(\d+)(?:_|$)", family)
        if declared is None:
            raise Chapter3BootstrapSourceError(
                f"Holm family does not declare its complete size: {family}"
            )
        expected = int(declared.group(1))
        if count != expected:
            raise Chapter3BootstrapSourceError(
                f"Holm family-size drift: {family}={count} != {expected}"
            )
    required = {
        "rq1_lp_matched_vs_no_text_rolling_holm4": 4,
        "rq2_representations_overall_holm2": 2,
        "rq3_branch_holm2": 2,
        "rq3_intervention_holm2": 2,
        "rq3_validation_epoch0_to_epoch30_holm4": 4,
        "rq3_validation_epoch30_to_best_holm4": 4,
        "architecture_primary_new_modes_vs_film_holm3": 3,
        "window_between_models_common5_holm5": 5,
        "window_within_film_lp_matched_common5_holm4": 4,
        "window_within_pure_cnn_no_text_common5_holm4": 4,
    }
    for family, expected in required.items():
        if counts.get(family) != expected:
            raise Chapter3BootstrapSourceError(
                f"Holm family-size drift: {family}={counts.get(family)} != {expected}"
            )
    architecture_text = {
        family: count
        for family, count in counts.items()
        if family.startswith("architecture_text_reliance_")
    }
    if len(architecture_text) != 6 or set(architecture_text.values()) != {2}:
        raise Chapter3BootstrapSourceError(
            f"Architecture Holm-2 family drift: {architecture_text}"
        )
    direct_persistence = {
        family: count
        for family, count in counts.items()
        if family.startswith("direct_")
        and family.endswith("_model_vs_persistence_holm5")
    }
    if len(direct_persistence) != 5 or set(direct_persistence.values()) != {5}:
        raise Chapter3BootstrapSourceError(
            f"Direct persistence Holm-5 family drift: {direct_persistence}"
        )
    rq2_folds = {
        family: count
        for family, count in counts.items()
        if family.startswith("rq2_representations_f")
    }
    if len(rq2_folds) != 4 or set(rq2_folds.values()) != {2}:
        raise Chapter3BootstrapSourceError(
            f"Quarterly RQ2 Holm-2 family drift: {rq2_folds}"
        )


def build_jobs(
    sources: Mapping[str, pd.DataFrame], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build all RQ1--RQ3 and displayed inferential robustness recipes.

    Holm family identifiers are global across jobs.  In particular, the four
    quarterly RQ1 contrasts share one Holm-4 family even though each quarter is
    represented by a separate single-fold job.  Capacity and coverage lineage
    remain in the input manifest as descriptive pass-throughs and do not create
    inferential jobs.  RQ4 never enters this function.
    """

    required_sources = {
        "direct",
        "rq3_full",
        "rq3_branch",
        "rq3_intervention",
        "rq3_validation",
        "architecture",
        "alignment",
    }
    missing = sorted(required_sources - set(sources))
    if missing:
        raise Chapter3BootstrapSourceError(f"Missing canonical sources: {missing}")
    recipes = _mapping(config.get("recipes"), "recipes")
    legacy = _legacy_tables(config)
    jobs: list[dict[str, Any]] = []
    jobs.extend(_direct_jobs(sources, recipes, legacy))
    jobs.extend(_rq3_jobs(sources, recipes, legacy))
    jobs.append(_architecture_job(sources, recipes, legacy))
    jobs.extend(_alignment_jobs(sources, recipes, legacy))
    _audit_jobs(jobs)
    return jobs


__all__ = [
    "CANONICAL_COLUMNS",
    "Chapter3BootstrapSourceError",
    "DEFAULT_CONFIG",
    "build_jobs",
    "load_sources",
]
