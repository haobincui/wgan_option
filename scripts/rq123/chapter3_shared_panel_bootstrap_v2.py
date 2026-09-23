"""Reanalyse frozen Chapter 3 predictions with shared market bootstrap weights.

Run with ``python -m scripts.rq123.chapter3_shared_panel_bootstrap_v2``.
This entrypoint has no training or prediction stage. Historical experiment
roots are read-only inputs; corrected results have a separate versioned root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.rq123.chapter3_bootstrap_sources import build_jobs, load_sources
from scripts.rq123.shared_panel_bootstrap_core import (
    METHOD_VERSION,
    load_schedule,
    make_schedule,
    prepare_panel,
    run_bootstrap,
    save_schedule,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/rq123/chapter3_shared_panel_bootstrap_v2.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/analysis/chapter3_shared_market_panel_bootstrap_10000_v2"
KIND = "chapter3_shared_market_panel_bootstrap_v2"
HASH_MANIFEST = "output_hashes.csv"
REQUIRED_TOP_LEVEL_FILES = {
    "analysis_manifest.json",
    "chapter3_values.json",
    "input_manifest.json",
    "qa.json",
    "report.md",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, float_format="%.17g", compression=compression)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _archive_path(root: Path, relative_path: Any, *, label: str) -> Path:
    """Resolve one archive member and reject absolute/traversing/symlink paths."""

    text = str(relative_path)
    relative = Path(text)
    if not text or text == "." or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe {label} archive path: {text!r}")
    if text != relative.as_posix() or "." in relative.parts:
        raise ValueError(f"Non-canonical {label} archive path: {text!r}")
    root_resolved = root.resolve()
    candidate = root / relative
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"Escaping {label} archive path: {text!r}") from error
    if candidate.is_symlink():
        raise ValueError(f"Symlinked {label} archive path is not permitted: {text!r}")
    return candidate


def _schedule_key(panel: Any, *, iterations: int, rng_seed: int) -> str:
    payload = {
        "method_version": METHOD_VERSION,
        "market": panel.market_fingerprint,
        "seeds": list(panel.seeds),
        "iterations": int(iterations),
        "rng_seed": str(int(rng_seed)),
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _copy_provenance(staging: Path, config_path: Path) -> tuple[str, list[dict[str, Any]]]:
    provenance = staging / "provenance"
    provenance.mkdir()
    config_archive = "provenance/config.yaml"
    shutil.copyfile(config_path, staging / config_archive)
    implementations = (
        Path(__file__).resolve(),
        REPO_ROOT / "scripts/rq123/shared_panel_bootstrap_core.py",
        REPO_ROOT / "scripts/rq123/chapter3_bootstrap_sources.py",
    )
    records: list[dict[str, Any]] = []
    for source in implementations:
        relative = f"provenance/{source.name}"
        shutil.copyfile(source, staging / relative)
        source_hash = sha256_file(source)
        if sha256_file(staging / relative) != source_hash:
            raise ValueError(f"Provenance copy drift: {source}")
        records.append(
            {"path": str(source), "sha256": source_hash, "archived_path": relative}
        )
    return config_archive, records


def _verify_output_hashes(root: Path) -> pd.DataFrame:
    manifest_path = root / HASH_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Output hash manifest is missing or symlinked")
    hashes = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if list(hashes.columns) != ["relative_path", "sha256"] or hashes.empty:
        raise ValueError("Output hash manifest schema is invalid")
    if hashes["relative_path"].duplicated().any():
        raise ValueError("Output hash manifest contains duplicate paths")
    listed: set[str] = set()
    for row in hashes.itertuples(index=False):
        path = _archive_path(root, row.relative_path, label="output")
        relative = Path(str(row.relative_path)).as_posix()
        if relative == HASH_MANIFEST:
            raise ValueError("Output hash manifest must not hash itself")
        expected = str(row.sha256).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid output SHA-256 for {relative}")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Output hash mismatch: {path}")
        listed.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != HASH_MANIFEST
    }
    if listed != actual:
        raise ValueError(
            "Output hash file-set drift: "
            f"unlisted={sorted(actual - listed)}, missing={sorted(listed - actual)}"
        )
    missing_required = sorted(REQUIRED_TOP_LEVEL_FILES - listed)
    if missing_required:
        raise ValueError(f"Output hash manifest lacks required files: {missing_required}")
    return hashes


def _holm(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the declared families jointly, including families spanning jobs."""

    required = {
        "job_id",
        "contrast_id",
        "family_id",
        "apply_holm",
        "alternative",
        "p_one",
        "p_two",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"Holm input is empty or missing columns: {missing}")
    result = frame.copy()
    for column in ("job_id", "contrast_id", "family_id", "alternative"):
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Holm input has empty {column}")
        result[column] = result[column].astype(str)
    if result.duplicated(["job_id", "contrast_id"]).any():
        raise ValueError("Holm input has duplicate job/contrast identifiers")
    if not result["alternative"].isin({"one_sided", "two_sided"}).all():
        raise ValueError("alternative must be one_sided or two_sided")
    if not all(isinstance(value, (bool, np.bool_)) for value in result["apply_holm"]):
        raise ValueError("apply_holm must contain booleans")
    result["apply_holm"] = result["apply_holm"].astype(bool)
    for column in ("p_one", "p_two"):
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"{column} must contain finite probabilities")
        result[column] = values
    result["holm_p"] = np.nan
    result["family_size"] = 0
    result["p_selected"] = np.where(
        result["alternative"].eq("two_sided"), result["p_two"], result["p_one"]
    )
    adjusted_rows = result[result["apply_holm"]]
    for family_id, group in adjusted_rows.groupby("family_id", sort=True, dropna=False):
        declared = re.search(r"holm[_-]?(\d+)(?:_|$)", str(family_id), re.IGNORECASE)
        if declared and int(declared.group(1)) != len(group):
            raise ValueError(f"Incomplete Holm family: {family_id}, observed={len(group)}")
        ordered = group.sort_values(["p_selected", "job_id", "contrast_id"], kind="stable")
        adjusted = np.minimum(
            1.0,
            np.maximum.accumulate(
                ordered["p_selected"].to_numpy(float)
                * np.arange(len(ordered), 0, -1)
            ),
        )
        result.loc[ordered.index, "holm_p"] = adjusted
        result.loc[ordered.index, "family_size"] = len(ordered)
    result["reported_p"] = result["holm_p"].where(result["apply_holm"], result["p_selected"])
    if not np.isfinite(result["reported_p"].to_numpy(float)).all():
        raise ValueError("Holm adjustment left a missing reported p-value")
    result["significance_stars"] = [
        "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
        for p in result["reported_p"]
    ]
    result["passes_existing_support_gate"] = None
    if "gate_alpha" in result:
        for index, row in result[result["gate_alpha"].notna()].iterrows():
            required_gate = (
                "point",
                "ci_upper",
                "seed_consistent",
                "fold_consistent",
                "gate_alpha",
                "gate_minimum_seeds",
                "gate_minimum_folds",
                "gate_require_ci",
            )
            if any(column not in result for column in required_gate):
                raise ValueError("Support-gate columns are incomplete")
            gate_values = np.asarray(
                [
                    row["point"],
                    row["ci_upper"],
                    row["seed_consistent"],
                    row["fold_consistent"],
                    row["gate_alpha"],
                    row["gate_minimum_seeds"],
                    row["gate_minimum_folds"],
                ],
                dtype=float,
            )
            if not np.isfinite(gate_values).all():
                raise ValueError("Support-gate inputs must be finite")
            require_ci = row["gate_require_ci"]
            if not isinstance(require_ci, (bool, np.bool_)):
                raise ValueError("gate_require_ci must be boolean for gated contrasts")
            result.at[index, "passes_existing_support_gate"] = bool(
                row["point"] < 0 and row["reported_p"] < row["gate_alpha"]
                and (not require_ci or row["ci_upper"] < 0)
                and row["seed_consistent"] >= row["gate_minimum_seeds"]
                and row["fold_consistent"] >= row["gate_minimum_folds"])
    return result


def _verify_inputs(manifest: list[dict[str, Any]]) -> None:
    for row in manifest:
        if "path" not in row or "sha256" not in row:
            raise ValueError("Frozen input manifest row lacks path/SHA-256")
        path = _resolve(row["path"])
        expected = str(row["sha256"]).lower()
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected)
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise ValueError(f"Frozen input changed: {path}")


def _legacy_numeric(
    previous: Mapping[str, Any], candidates: tuple[str, ...], *, label: str
) -> float | None:
    observed: list[tuple[str, float]] = []
    for candidate in candidates:
        if candidate not in previous or previous[candidate] is None:
            continue
        try:
            if pd.isna(previous[candidate]):
                continue
            value = float(previous[candidate])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Legacy {label}.{candidate} is not numeric") from error
        if math.isfinite(value):
            observed.append((candidate, value))
    if not observed:
        return None
    reference_name, reference = observed[0]
    for name, value in observed[1:]:
        if not math.isclose(reference, value, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"Conflicting legacy aliases for {label}: "
                f"{reference_name}={reference}, {name}={value}"
            )
    return reference


def _old_new(contrasts: pd.DataFrame, legacy: dict[tuple[str, str], dict]) -> pd.DataFrame:
    rows = []
    aliases = {"point": ("point", "mean_log_mae_ratio", "log_mae_ratio"),
               "bootstrap_se": ("se", "bootstrap_se", "log_ratio_bootstrap_se"),
               "ci_lower": ("ci_lower", "ci_95_lower", "log_ratio_ci_95_lower"),
               "ci_upper": ("ci_upper", "ci_95_upper", "log_ratio_ci_95_upper"),
               "p_one": ("p_one", "p_value_one_sided", "p_value_one_sided_sign_tail"),
               "holm_p": ("holm_p", "holm_adjusted_p", "holm_p_value", "holm5_adjusted_p")}
    for new in contrasts.to_dict("records"):
        previous = legacy.get((new["job_id"], new["contrast_id"])) or {}
        row = {"job_id": new["job_id"], "contrast_id": new["contrast_id"],
               "legacy_source": previous.get("source", ""),
               "legacy_available": bool(previous)}
        for field, candidates in aliases.items():
            old_value = _legacy_numeric(
                previous,
                candidates,
                label=f"{new['job_id']}/{new['contrast_id']}/{field}",
            )
            row[f"old_{field}"] = old_value
            row[f"new_{field}"] = new.get(field)
        old_point = row["old_point"]
        if previous and old_point is None:
            raise ValueError(
                f"Archived legacy comparison has no finite point for "
                f"{new['job_id']}/{new['contrast_id']}"
            )
        row["point_preservation_checked"] = old_point is not None
        if old_point is not None:
            row["point_difference"] = new["point"] - float(old_point)
            if not math.isclose(new["point"], float(old_point), rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError(f"Observed estimand changed for {new['job_id']}/{new['contrast_id']}")
        old_se = row["old_bootstrap_se"]
        row["se_ratio_new_over_old"] = (new["bootstrap_se"] / float(old_se)
                                        if old_se is not None and float(old_se) > 0 else None)
        rows.append(row)
    return pd.DataFrame(rows)


def _normalise_recipe(
    raw: Mapping[str, Any], *, job_id: str, conditions: tuple[str, ...]
) -> dict[str, Any]:
    recipe = dict(raw)
    required = ("contrast_id", "focal", "reference", "family_id")
    missing = [key for key in required if key not in recipe]
    if missing:
        raise ValueError(f"{job_id} contrast recipe lacks keys: {missing}")
    result = {key: str(recipe[key]).strip() for key in required}
    if any(not result[key] for key in required):
        raise ValueError(f"{job_id} contrast recipe contains an empty identifier")
    if result["focal"] not in conditions or result["reference"] not in conditions:
        raise ValueError(
            f"{job_id}/{result['contrast_id']} references an absent condition"
        )
    alternative = str(recipe.get("alternative", "one_sided")).strip()
    if alternative not in {"one_sided", "two_sided"}:
        raise ValueError(
            f"{job_id}/{result['contrast_id']} has invalid alternative={alternative!r}"
        )
    apply_holm = recipe.get("apply_holm", True)
    if not isinstance(apply_holm, (bool, np.bool_)):
        raise ValueError(f"{job_id}/{result['contrast_id']} apply_holm must be boolean")
    result.update(alternative=alternative, apply_holm=bool(apply_holm))

    gate_raw = recipe.get("support_gate")
    if gate_raw is not None:
        if not isinstance(gate_raw, Mapping):
            raise ValueError(f"{job_id}/{result['contrast_id']} support_gate is invalid")
        gate = dict(gate_raw)
        required_gate = ("alpha", "minimum_nonworse_seeds", "minimum_nonworse_folds")
        if any(key not in gate for key in required_gate):
            raise ValueError(f"{job_id}/{result['contrast_id']} support_gate is incomplete")
        alpha = float(gate["alpha"])
        minimum_seeds = int(gate["minimum_nonworse_seeds"])
        minimum_folds = int(gate["minimum_nonworse_folds"])
        require_ci = gate.get("require_ci_below_zero", True)
        if (
            not 0.0 < alpha < 1.0
            or not 1 <= minimum_seeds
            or not 1 <= minimum_folds
            or not isinstance(require_ci, (bool, np.bool_))
        ):
            raise ValueError(f"{job_id}/{result['contrast_id']} support_gate is invalid")
        result["support_gate"] = {
            "alpha": alpha,
            "minimum_nonworse_seeds": minimum_seeds,
            "minimum_nonworse_folds": minimum_folds,
            "require_ci_below_zero": bool(require_ci),
        }
    else:
        result["support_gate"] = None

    legacy_raw = recipe.get("legacy")
    if legacy_raw is not None and not isinstance(legacy_raw, Mapping):
        raise ValueError(f"{job_id}/{result['contrast_id']} legacy recipe is invalid")
    result["legacy"] = _clean(dict(legacy_raw)) if legacy_raw else None
    return result


def _common_fields(job: Mapping[str, Any], panel: Any, schedule_id: str) -> dict[str, Any]:
    metadata_raw = job.get("metadata", {})
    if not isinstance(metadata_raw, Mapping):
        raise ValueError(f"{job['job_id']} metadata must be a mapping")
    metadata = _clean(dict(metadata_raw))
    reserved = {
        "job_id",
        "estimand",
        "seed_count",
        "fold_count",
        "pair_count",
        "session_count",
        "schedule_id",
        "condition",
        "contrast_id",
    }
    collision = sorted(reserved & set(metadata))
    if collision:
        raise ValueError(f"{job['job_id']} metadata collides with core fields: {collision}")
    canonical = panel.frame
    return {
        "job_id": str(job["job_id"]),
        "estimand": str(job["estimand"]),
        "seed_count": len(panel.seeds),
        "fold_count": len(panel.folds),
        "pair_count": int(canonical[["fold", "pair_id"]].drop_duplicates().shape[0]),
        "session_count": int(
            canonical[["fold", "session_id"]].drop_duplicates().shape[0]
        ),
        "schedule_id": schedule_id,
        **metadata,
    }


def _arm_rows(result: Any, common: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, condition in enumerate(result.conditions):
        values = result.mean_draws[:, index]
        lower, upper = np.quantile(values, (0.025, 0.975))
        rows.append(
            {
                **common,
                "condition": condition,
                "observed_mean_mae": float(result.observed_means[index]),
                "bootstrap_mean_mae": float(values.mean()),
                "bootstrap_mae_se": float(values.std(ddof=1)),
                "mae_ci_lower": float(lower),
                "mae_ci_upper": float(upper),
            }
        )
    return rows


def _contrast_rows(
    result: Any,
    common: Mapping[str, Any],
    recipes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[tuple[str, str], dict]]:
    rows: list[dict[str, Any]] = []
    draws: list[np.ndarray] = []
    legacy: dict[tuple[str, str], dict] = {}
    job_id = str(common["job_id"])
    for recipe in recipes:
        full_stats = result.contrast(recipe["focal"], recipe["reference"])
        contrast_draws = full_stats["draws"]
        stats = {key: value for key, value in full_stats.items() if key != "draws"}
        contrast_id = recipe["contrast_id"]
        legacy[(job_id, contrast_id)] = dict(recipe.get("legacy") or {})
        row = {
            **common,
            **stats,
            "contrast_id": contrast_id,
            "focal": recipe["focal"],
            "reference": recipe["reference"],
            "family_id": recipe["family_id"],
            "alternative": recipe["alternative"],
            "apply_holm": recipe["apply_holm"],
            "geometric_gain_percent": 100.0 * (1.0 - math.exp(stats["point"])),
            "gain_ci_lower_percent": 100.0 * (1.0 - math.exp(stats["ci_upper"])),
            "gain_ci_upper_percent": 100.0 * (1.0 - math.exp(stats["ci_lower"])),
            "consistent_seed_count": stats["seed_consistent"],
            "consistent_fold_count": stats["fold_consistent"],
        }
        gate = recipe.get("support_gate")
        if gate is not None:
            row.update(
                gate_alpha=float(gate["alpha"]),
                gate_minimum_seeds=int(gate["minimum_nonworse_seeds"]),
                gate_minimum_folds=int(gate["minimum_nonworse_folds"]),
                gate_require_ci=bool(gate["require_ci_below_zero"]),
            )
        rows.append(row)
        draws.append(contrast_draws)
    return rows, draws, legacy


def _build_values(
    job_ids: list[str], arms: pd.DataFrame, contrasts: pd.DataFrame
) -> dict[str, Any]:
    values = {job_id: {"arms": {}, "contrasts": {}} for job_id in job_ids}
    for row in arms.to_dict("records"):
        values[row["job_id"]]["arms"][row["condition"]] = row
    for row in contrasts.to_dict("records"):
        values[row["job_id"]]["contrasts"][row["contrast_id"]] = row
    for job_values in values.values():
        job_values["ratios"] = {}
        for arm, left in job_values["arms"].items():
            for reference, right in job_values["arms"].items():
                observed_ratio = left["observed_mean_mae"] / right["observed_mean_mae"]
                bootstrap_ratio = (
                    left["bootstrap_mean_mae"] / right["bootstrap_mean_mae"]
                )
                job_values["ratios"][f"{arm}__over__{reference}"] = {
                    "observed_mae_ratio": observed_ratio,
                    "bootstrap_mae_ratio": bootstrap_ratio,
                    "observed_improvement_percent": 100.0 * (1.0 - observed_ratio),
                    "bootstrap_improvement_percent": 100.0 * (1.0 - bootstrap_ratio),
                    "observed_mae_difference": left["observed_mean_mae"]
                    - right["observed_mean_mae"],
                    "bootstrap_mae_difference": left["bootstrap_mean_mae"]
                    - right["bootstrap_mean_mae"],
                }
    return {"kind": KIND, "jobs": values}


def _qa_payload(
    *,
    jobs: list[Mapping[str, Any]],
    arms: pd.DataFrame,
    contrasts: pd.DataFrame,
    comparison: pd.DataFrame,
) -> dict[str, Any]:
    checked = int(comparison.get("point_preservation_checked", pd.Series(dtype=bool)).sum())
    legacy_available = int(comparison["legacy_available"].sum())
    return {
        "kind": KIND,
        "schema_version": 2,
        "passed": True,
        "input_hashes_unchanged": True,
        "iterations_per_comparison": 10_000,
        "job_count": len(jobs),
        "contrast_count": len(contrasts),
        "arm_summary_count": len(arms),
        "legacy_comparison_count": legacy_available,
        "point_preservation_checked_count": checked,
        "legacy_without_reproducible_point_count": legacy_available - checked,
        "observed_estimands_preserved": checked == legacy_available,
        "rq4_recomputed": False,
    }


def _report_text(
    arms: pd.DataFrame, contrasts: pd.DataFrame, comparison: pd.DataFrame
) -> str:
    lines = ["# Chapter 3 shared market bootstrap v2", "",
             "Frozen predictions; 10,000 crossed seed × market draws per comparison. "
             "Each fold occurrence has one session sample shared by every seed and paired arm. "
             "Repeated occurrences of a quarter have independent session samples.", "",
             "The original equal-cell and pooled-pair estimands, hypothesis directions and "
             "Holm families are retained. Percentile intervals and +1 sign-tail probabilities "
             "are approximate diagnostics, not exact or bootstrap-t tests. "
             "RQ4 is excluded. Test-panel development reuse, few quarters and dependence "
             "across sessions remain limitations.", "",
             "Method reference: [Owen and Eckles (2012), Bootstrapping data arrays of arbitrary order]"
             "(https://arxiv.org/abs/1106.2125). The crossed-factor motivation does not provide "
             "an exact finite-sample guarantee for these nonlinear log-ratio statistics.", "",
             f"{len(arms)} arm summaries; {len(contrasts)} contrasts; "
             f"{int(comparison['legacy_available'].sum())} contrasts with archived legacy inference.", "",
             "Observed point estimates are distinct from bootstrap means. Missing legacy "
             "inference means no reproducible archived result was found; no old values were invented.", "",
             "| Job | Contrast | Point log ratio | SE | 95% CI | Reported p | Family |",
             "|---|---|---:|---:|---|---:|---|"]
    for row in contrasts.itertuples(index=False):
        lines.append(
            f"| {row.job_id} | {row.contrast_id} | {row.point:.8g} | "
            f"{row.bootstrap_se:.8g} | [{row.ci_lower:.8g}, {row.ci_upper:.8g}] | "
            f"{row.reported_p:.6g} | {row.family_id} |"
        )
    lines += ["", "The input manifest binds original data and historical results by SHA-256. "
              "Saved schedules, canonical panels and draws permit full replay with `--verify-only`. "
              "Purely descriptive training, capacity and coverage evidence retains its original source."]
    return "\n".join(lines) + "\n"


def _report(root: Path, arms: pd.DataFrame, contrasts: pd.DataFrame, comparison: pd.DataFrame) -> None:
    (root / "report.md").write_text(
        _report_text(arms, contrasts, comparison), encoding="utf-8"
    )


def _assert_frame_matches(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    label: str,
    atol: float = 1e-12,
) -> None:
    """Compare a CSV round-trip with a freshly reconstructed data frame."""

    if (actual.columns.duplicated().any() or expected.columns.duplicated().any()
            or set(actual.columns) != set(expected.columns)):
        raise ValueError(
            f"{label} column drift: observed={list(actual.columns)}, "
            f"expected={list(expected.columns)}"
        )
    missing_keys = [key for key in keys if key not in actual]
    if missing_keys:
        raise ValueError(f"{label} lacks comparison keys: {missing_keys}")
    observed = actual.reindex(columns=expected.columns).sort_values(
        list(keys), kind="stable"
    ).reset_index(drop=True)
    reference = expected.sort_values(list(keys), kind="stable").reset_index(drop=True)
    # CSV has no lossless distinction between an empty string and a missing
    # field.  Normalise only columns whose reconstructed, non-null values are
    # all strings; numeric and nullable-boolean columns remain fail-closed.
    for column in reference.columns:
        non_null = reference[column].dropna()
        if len(non_null) and all(isinstance(value, str) for value in non_null):
            observed[column] = observed[column].fillna("")
            reference[column] = reference[column].fillna("")
        elif observed[column].dtype == object or reference[column].dtype == object:
            sentinel = "<ARCHIVE_NULL>"
            observed[column] = observed[column].map(
                lambda value: sentinel if pd.isna(value) else value
            )
            reference[column] = reference[column].map(
                lambda value: sentinel if pd.isna(value) else value
            )
    try:
        pd.testing.assert_frame_equal(
            observed,
            reference,
            check_dtype=False,
            check_exact=False,
            check_categorical=False,
            rtol=0.0,
            atol=atol,
        )
    except AssertionError as error:
        raise ValueError(f"{label} drift: {str(error)[:1200]}") from error


def _assert_json_matches(actual: Any, expected: Any, *, label: str) -> None:
    """Recursively compare JSON-compatible values with tight numeric tolerance."""

    def compare(left: Any, right: Any, location: str) -> None:
        left = _clean(left)
        right = _clean(right)
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                raise ValueError(f"{label} type drift at {location}")
            if set(left) != set(right):
                raise ValueError(
                    f"{label} key drift at {location}: "
                    f"observed={sorted(left)}, expected={sorted(right)}"
                )
            for key in sorted(left):
                compare(left[key], right[key], f"{location}.{key}")
            return
        if isinstance(left, list) or isinstance(right, list):
            if not isinstance(left, list) or not isinstance(right, list):
                raise ValueError(f"{label} type drift at {location}")
            if len(left) != len(right):
                raise ValueError(f"{label} length drift at {location}")
            for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
                compare(left_value, right_value, f"{location}[{index}]")
            return
        if isinstance(left, bool) or isinstance(right, bool):
            if not isinstance(left, bool) or not isinstance(right, bool) or left != right:
                raise ValueError(f"{label} boolean drift at {location}")
            return
        numeric = (int, float)
        if isinstance(left, numeric) or isinstance(right, numeric):
            if not isinstance(left, numeric) or not isinstance(right, numeric):
                raise ValueError(f"{label} type drift at {location}")
            if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"{label} numeric drift at {location}: {left!r} != {right!r}"
                )
            return
        if left != right:
            raise ValueError(f"{label} value drift at {location}: {left!r} != {right!r}")

    compare(actual, expected, "$")


def _npz_scalar_string(archive: Any, key: str) -> str:
    value = archive[key]
    if value.size != 1:
        raise ValueError(f"Draw archive {key} must be a scalar")
    return str(value.reshape(()).item())


def _assert_array_matches(
    actual: Any, expected: Any, *, label: str, atol: float = 0.0
) -> None:
    try:
        if atol == 0.0:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)
    except AssertionError as error:
        raise ValueError(f"{label} drift: {str(error)[:1200]}") from error


def run_analysis(config_path: Path = DEFAULT_CONFIG, output_root: Path | None = None) -> Path:
    sources, inputs, config = load_sources(config_path)
    iterations = int(config["bootstrap"]["iterations"])
    if iterations != 10_000:
        raise ValueError("Formal v2 inference requires exactly 10,000 draws")
    rng_seed = int(config["bootstrap"]["rng_seed"])
    if rng_seed < 0:
        raise ValueError("Bootstrap RNG seed must be non-negative")
    config_file = _resolve(config.get("_config_path", config_path))
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    config_hash = sha256_file(config_file)
    configured_hash = config.get("_config_sha256")
    if configured_hash is not None and str(configured_hash).lower() != config_hash:
        raise ValueError("Loaded configuration hash drift")
    target = _resolve(output_root or config.get("output_root", DEFAULT_OUTPUT))
    if target.exists():
        raise FileExistsError(f"Output exists; use --verify-only: {target}")
    jobs = build_jobs(sources, config)
    if not jobs:
        raise ValueError("No Chapter 3 bootstrap jobs were constructed")
    ids = [str(job["job_id"]) for job in jobs]
    if len(ids) != len(set(ids)) or len(ids) != len({_safe_name(v) for v in ids}):
        raise ValueError("Duplicate or filename-colliding job identifiers")
    _verify_inputs(inputs)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.attempt-", dir=target.parent))
    for folder in ("analysis", "panels", "schedules", "draws"):
        (staging / folder).mkdir()
    config_archive, implementation = _copy_provenance(staging, config_file)
    _write_json(
        staging / "input_manifest.json",
        {
            "kind": KIND,
            "schema_version": 2,
            "bootstrap_method_version": METHOD_VERSION,
            "inputs": inputs,
            "config_path": str(config_file),
            "config_sha256": config_hash,
            "config_archive_path": config_archive,
            "config": config,
            "implementation": implementation,
        },
    )
    schedules: dict[str, Any] = {}
    arm_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    job_records: list[dict[str, Any]] = []
    legacy: dict[tuple[str, str], dict[str, Any]] = {}
    for job in jobs:
        job_id = str(job["job_id"])
        filename = _safe_name(job_id)
        panel = prepare_panel(job["panel"])
        recipes = [
            _normalise_recipe(recipe, job_id=job_id, conditions=panel.conditions)
            for recipe in job["contrasts"]
        ]
        contrast_ids = [recipe["contrast_id"] for recipe in recipes]
        if len(contrast_ids) != len(set(contrast_ids)):
            raise ValueError(f"{job_id} has duplicate contrast identifiers")
        schedule_key = _schedule_key(
            panel, iterations=iterations, rng_seed=rng_seed
        )
        schedule_path = f"schedules/{schedule_key}.npz"
        if schedule_key not in schedules:
            schedules[schedule_key] = make_schedule(
                panel, iterations=iterations, rng_seed=rng_seed
            )
            save_schedule(schedules[schedule_key], staging / schedule_path)
        schedule = schedules[schedule_key]
        result = run_bootstrap(panel, schedule, estimand=job["estimand"])
        panel_path = f"panels/{filename}.csv.gz"
        draws_path = f"draws/{filename}.npz"
        canonical = panel.frame
        _write_csv(staging / panel_path, canonical)
        common = _common_fields(job, panel, schedule_key)
        arm_rows.extend(_arm_rows(result, common))
        new_contrasts, log_draws, job_legacy = _contrast_rows(
            result, common, recipes
        )
        contrast_rows.extend(new_contrasts)
        legacy.update(job_legacy)
        np.savez_compressed(
            staging / draws_path,
            draw_id=schedule.draw_id,
            condition_names=np.asarray(result.conditions, dtype=str),
            observed_means=result.observed_means,
            mean_draws=result.mean_draws,
            contrast_ids=np.asarray(contrast_ids, dtype=str),
            log_ratio_draws=(np.column_stack(log_draws) if log_draws else
                             np.empty((iterations, 0), dtype=float)),
            schedule_id=np.asarray(schedule_key),
            method_version=np.asarray(METHOD_VERSION),
            panel_fingerprint=np.asarray(panel.panel_fingerprint),
            estimand=np.asarray(str(job["estimand"])),
        )
        metadata = _clean(dict(job.get("metadata", {})))
        job_records.append(
            {
                "job_id": job_id,
                "estimand": str(job["estimand"]),
                "metadata": metadata,
                "panel_path": panel_path,
                "schedule_path": schedule_path,
                "draws_path": draws_path,
                "schedule_id": schedule_key,
                "bootstrap_method_version": METHOD_VERSION,
                "market_fingerprint": panel.market_fingerprint,
                "panel_fingerprint": panel.panel_fingerprint,
                "conditions": list(panel.conditions),
                "seeds": list(panel.seeds),
                "folds": list(panel.folds),
                "sessions_by_fold": [list(values) for values in panel.sessions_by_fold],
                "pair_count": common["pair_count"],
                "session_count": common["session_count"],
                "contrasts": recipes,
            }
        )
        print(f"Completed {job_id}: {len(result.conditions)} conditions, {len(recipes)} contrasts", flush=True)
    arms = pd.DataFrame(arm_rows)
    contrasts = _holm(pd.DataFrame(contrast_rows))
    comparison = _old_new(contrasts, legacy)
    _write_csv(staging / "analysis/all_arm_summary.csv", arms)
    _write_csv(staging / "analysis/all_contrasts.csv", contrasts)
    _write_csv(staging / "analysis/new_old_comparison.csv", comparison)
    _write_json(staging / "chapter3_values.json", _build_values(ids, arms, contrasts))
    _write_json(
        staging / "analysis_manifest.json",
        {
            "kind": KIND,
            "schema_version": 2,
            "bootstrap_method_version": METHOD_VERSION,
            "iterations": iterations,
            "rng_seed": str(rng_seed),
            "schedule_identity_fields": [
                "bootstrap_method_version",
                "market_fingerprint",
                "seed_universe",
                "iterations",
                "rng_seed",
            ],
            "jobs": job_records,
            "shared_market_schedules": len(schedules),
            "scope": "RQ1-RQ3 and architecture/alignment; excludes RQ4",
            "fold_treatment": (
                "globally_resampled_occurrences_with_independent_child_session_samples"
            ),
            "p_value_interpretation": (
                "approximate_uncentered_plus_one_bootstrap_sign_tail"
            ),
            "ci_method": "percentile_95",
            "confirmatory": False,
        },
    )
    _report(staging, arms, contrasts, comparison)
    _verify_inputs(inputs)
    _write_json(
        staging / "qa.json",
        _qa_payload(jobs=jobs, arms=arms, contrasts=contrasts, comparison=comparison),
    )
    outputs = [
        {
            "relative_path": path.relative_to(staging).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != HASH_MANIFEST
    ]
    _write_csv(staging / HASH_MANIFEST, pd.DataFrame(outputs))
    verify_analysis(staging, replay=True)
    if target.exists():
        raise FileExistsError(f"Output appeared while computing; attempt retained at {staging}")
    os.rename(staging, target)
    return target


def verify_analysis(root: Path, *, replay: bool = True) -> dict[str, Any]:
    root = _resolve(root)
    manifest_path = root / "analysis_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Analysis manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != KIND or manifest.get("schema_version") != 2:
        raise ValueError("Not a corrected v2 result")
    if (
        manifest.get("bootstrap_method_version") != METHOD_VERSION
        or manifest.get("iterations") != 10_000
    ):
        raise ValueError("Formal draw count is not 10,000")
    iterations = int(manifest["iterations"])
    try:
        rng_seed = int(manifest["rng_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Analysis manifest has an invalid RNG seed") from error
    if rng_seed < 0 or str(rng_seed) != str(manifest["rng_seed"]):
        raise ValueError("Analysis manifest has a non-canonical RNG seed")
    _verify_output_hashes(root)

    input_manifest = json.loads(
        (root / "input_manifest.json").read_text(encoding="utf-8")
    )
    if (
        input_manifest.get("kind") != KIND
        or input_manifest.get("schema_version") != 2
        or input_manifest.get("bootstrap_method_version") != METHOD_VERSION
    ):
        raise ValueError("Input manifest contract drift")
    frozen_inputs = input_manifest.get("inputs")
    implementation = input_manifest.get("implementation")
    if not isinstance(frozen_inputs, list) or not isinstance(implementation, list):
        raise ValueError("Input manifest lacks frozen input/provenance lists")
    _verify_inputs(frozen_inputs)
    _verify_inputs(implementation)
    source_config = _resolve(input_manifest["config_path"])
    config_sha = str(input_manifest["config_sha256"]).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", config_sha):
        raise ValueError("Source configuration hash is invalid")
    if not source_config.is_file() or sha256_file(source_config) != config_sha:
        raise ValueError("Source configuration changed")
    archived_config = _archive_path(
        root, input_manifest["config_archive_path"], label="config provenance"
    )
    if not archived_config.is_file() or sha256_file(archived_config) != config_sha:
        raise ValueError("Archived configuration provenance drift")
    archived_paths = {str(input_manifest["config_archive_path"])}
    for record in implementation:
        archived = _archive_path(
            root, record.get("archived_path"), label="implementation provenance"
        )
        expected = str(record.get("sha256", "")).lower()
        if not archived.is_file() or sha256_file(archived) != expected:
            raise ValueError(f"Archived implementation provenance drift: {archived}")
        archived_paths.add(str(record["archived_path"]))
    if len(archived_paths) != len(implementation) + 1:
        raise ValueError("Provenance archive paths collide")
    config = input_manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Archived configuration object is invalid")
    bootstrap = config.get("bootstrap")
    if (
        not isinstance(bootstrap, Mapping)
        or int(bootstrap.get("iterations", -1)) != 10_000
        or int(bootstrap.get("rng_seed", -1)) != rng_seed
    ):
        raise ValueError("Archived configuration bootstrap contract drift")

    job_records = manifest.get("jobs")
    if not isinstance(job_records, list) or not job_records:
        raise ValueError("Analysis manifest has no jobs")
    job_ids = [str(job.get("job_id", "")) for job in job_records]
    if (
        any(not value for value in job_ids)
        or len(job_ids) != len(set(job_ids))
        or len(job_ids) != len({_safe_name(value) for value in job_ids})
    ):
        raise ValueError("Analysis manifest job identifiers are invalid")

    expected_files = {
        HASH_MANIFEST,
        *REQUIRED_TOP_LEVEL_FILES,
        "analysis/all_arm_summary.csv",
        "analysis/all_contrasts.csv",
        "analysis/new_old_comparison.csv",
        *archived_paths,
    }
    schedule_ids: set[str] = set()
    for job in job_records:
        job_id = str(job["job_id"])
        filename = _safe_name(job_id)
        expected_panel = f"panels/{filename}.csv.gz"
        expected_draws = f"draws/{filename}.npz"
        if job.get("panel_path") != expected_panel or job.get("draws_path") != expected_draws:
            raise ValueError(f"Non-canonical archive paths for job {job_id}")
        schedule_id = str(job.get("schedule_id", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", schedule_id):
            raise ValueError(f"Invalid schedule identity for job {job_id}")
        expected_schedule = f"schedules/{schedule_id}.npz"
        if job.get("schedule_path") != expected_schedule:
            raise ValueError(f"Schedule path/identity mismatch for job {job_id}")
        if job.get("bootstrap_method_version") != METHOD_VERSION:
            raise ValueError(f"Bootstrap method drift for job {job_id}")
        schedule_ids.add(schedule_id)
        expected_files.update((expected_panel, expected_draws, expected_schedule))
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(
            "Archive contract file-set drift: "
            f"extra={sorted(actual_files - expected_files)}, "
            f"missing={sorted(expected_files - actual_files)}"
        )
    if int(manifest.get("shared_market_schedules", -1)) != len(schedule_ids):
        raise ValueError("Shared schedule count drift")

    summary = pd.read_csv(
        root / "analysis/all_contrasts.csv", float_precision="round_trip"
    )
    if "significance_stars" in summary:
        summary["significance_stars"] = summary["significance_stars"].fillna("")
    arms = pd.read_csv(
        root / "analysis/all_arm_summary.csv", float_precision="round_trip"
    )
    comparison = pd.read_csv(
        root / "analysis/new_old_comparison.csv", float_precision="round_trip"
    )
    if summary.empty or arms.empty or comparison.empty:
        raise ValueError("Archived analysis tables must not be empty")
    if summary.duplicated(["job_id", "contrast_id"]).any() or arms.duplicated(
        ["job_id", "condition"]
    ).any():
        raise ValueError("Duplicate archived summaries")
    contrast_job_ids = {str(job["job_id"]) for job in job_records if job.get("contrasts")}
    if set(summary["job_id"].astype(str)) != contrast_job_ids or set(
        arms["job_id"].astype(str)
    ) != set(job_ids):
        raise ValueError("Summary job universe drift")

    recipe_keys: set[tuple[str, str]] = set()
    legacy: dict[tuple[str, str], dict[str, Any]] = {}
    normalised_recipes: dict[str, list[dict[str, Any]]] = {}
    for job in job_records:
        job_id = str(job["job_id"])
        conditions_raw = job.get("conditions")
        if not isinstance(conditions_raw, list) or not conditions_raw:
            raise ValueError(f"{job_id} has no archived condition universe")
        conditions = tuple(str(value) for value in conditions_raw)
        if len(conditions) != len(set(conditions)):
            raise ValueError(f"{job_id} has duplicate archived conditions")
        raw_recipes = job.get("contrasts")
        if not isinstance(raw_recipes, list):
            raise ValueError(f"{job_id} has invalid archived contrast recipes")
        recipes = [
            _normalise_recipe(recipe, job_id=job_id, conditions=conditions)
            for recipe in raw_recipes
        ]
        for raw, recipe in zip(raw_recipes, recipes, strict=True):
            _assert_json_matches(raw, recipe, label=f"{job_id} contrast recipe")
            key = (job_id, recipe["contrast_id"])
            if key in recipe_keys:
                raise ValueError(f"Duplicate archived contrast recipe: {key}")
            recipe_keys.add(key)
            legacy[key] = dict(recipe.get("legacy") or {})
            selected = summary[
                summary["job_id"].astype(str).eq(job_id)
                & summary["contrast_id"].astype(str).eq(recipe["contrast_id"])
            ]
            if len(selected) != 1:
                raise ValueError(f"Summary comparison coverage drift: {key}")
            row = selected.iloc[0]
            for field in ("focal", "reference", "family_id", "alternative"):
                if str(row[field]) != recipe[field]:
                    raise ValueError(f"Archived recipe/summary {field} drift: {key}")
            if not isinstance(row["apply_holm"], (bool, np.bool_)) or bool(
                row["apply_holm"]
            ) != recipe["apply_holm"]:
                raise ValueError(f"Archived recipe/summary apply_holm drift: {key}")
            gate = recipe.get("support_gate")
            gate_fields = (
                "gate_alpha",
                "gate_minimum_seeds",
                "gate_minimum_folds",
                "gate_require_ci",
            )
            if gate is None:
                if any(field in summary and not pd.isna(row[field]) for field in gate_fields):
                    raise ValueError(f"Unexpected support gate in summary: {key}")
            else:
                expected_gate = (
                    gate["alpha"],
                    gate["minimum_nonworse_seeds"],
                    gate["minimum_nonworse_folds"],
                )
                observed_gate = tuple(float(row[field]) for field in gate_fields[:3])
                if not np.allclose(observed_gate, expected_gate, rtol=0.0, atol=1e-15):
                    raise ValueError(f"Support-gate numeric drift: {key}")
                if not isinstance(row["gate_require_ci"], (bool, np.bool_)) or bool(
                    row["gate_require_ci"]
                ) != gate["require_ci_below_zero"]:
                    raise ValueError(f"Support-gate CI rule drift: {key}")
        normalised_recipes[job_id] = recipes
        observed_conditions = set(
            arms.loc[arms["job_id"].astype(str).eq(job_id), "condition"].astype(str)
        )
        if observed_conditions != set(conditions):
            raise ValueError(f"Arm condition universe drift for job {job_id}")
        metadata = job.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{job_id} metadata is invalid")
        expected_common = {
            "job_id": job_id,
            "estimand": job.get("estimand"),
            "seed_count": len(job.get("seeds", [])),
            "fold_count": len(job.get("folds", [])),
            "pair_count": job.get("pair_count"),
            "session_count": job.get("session_count"),
            "schedule_id": job.get("schedule_id"),
            **dict(metadata),
        }
        combined = pd.concat(
            [
                arms[arms["job_id"].astype(str).eq(job_id)],
                summary[summary["job_id"].astype(str).eq(job_id)],
            ],
            ignore_index=True,
            sort=False,
        )
        for field, value in expected_common.items():
            if field not in combined:
                raise ValueError(f"{job_id} summaries lack common field {field}")
            observed = combined[field].drop_duplicates().tolist()
            if len(observed) != 1:
                raise ValueError(f"{job_id} common field {field} is inconsistent")
            _assert_json_matches(
                _clean(observed[0]), _clean(value), label=f"{job_id}.{field}"
            )
    archived_keys = set(
        zip(
            summary["job_id"].astype(str),
            summary["contrast_id"].astype(str),
            strict=True,
        )
    )
    if archived_keys != recipe_keys:
        raise ValueError("Manifest/summary contrast universe drift")

    recalculated = _holm(summary)
    derived_columns = (
        "job_id",
        "contrast_id",
        "holm_p",
        "family_size",
        "p_selected",
        "reported_p",
        "significance_stars",
        "passes_existing_support_gate",
    )
    _assert_frame_matches(
        summary.loc[:, derived_columns],
        recalculated.loc[:, derived_columns],
        keys=("job_id", "contrast_id"),
        label="Holm/support-gate results",
        atol=1e-15,
    )
    expected_comparison = _old_new(summary, legacy)
    _assert_frame_matches(
        comparison,
        expected_comparison,
        keys=("job_id", "contrast_id"),
        label="legacy comparison",
    )
    values = json.loads((root / "chapter3_values.json").read_text(encoding="utf-8"))
    _assert_json_matches(
        values,
        _build_values(job_ids, arms, summary),
        label="chapter3 values",
    )
    qa = json.loads((root / "qa.json").read_text(encoding="utf-8"))
    expected_qa = _qa_payload(
        jobs=job_records, arms=arms, contrasts=summary, comparison=comparison
    )
    _assert_json_matches(qa, expected_qa, label="terminal QA")
    report = (root / "report.md").read_text(encoding="utf-8")
    if report != _report_text(arms, summary, comparison):
        raise ValueError("Human-readable report drift")

    if replay:
        replay_arms: list[dict[str, Any]] = []
        replay_contrasts: list[dict[str, Any]] = []
        replay_legacy: dict[tuple[str, str], dict[str, Any]] = {}
        replay_schedule_paths: dict[str, str] = {}
        for job in manifest["jobs"]:
            job_id = str(job["job_id"])
            frame = pd.read_csv(
                _archive_path(root, job["panel_path"], label="panel"),
                float_precision="round_trip",
            )
            panel = prepare_panel(frame)
            if (
                panel.market_fingerprint != job.get("market_fingerprint")
                or panel.panel_fingerprint != job.get("panel_fingerprint")
                or list(panel.conditions) != job.get("conditions")
                or list(panel.seeds) != job.get("seeds")
                or list(panel.folds) != job.get("folds")
                or [list(values) for values in panel.sessions_by_fold]
                != job.get("sessions_by_fold")
            ):
                raise ValueError(f"Archived panel identity drift for job {job_id}")
            schedule_id = _schedule_key(
                panel, iterations=10_000, rng_seed=rng_seed
            )
            if schedule_id != job.get("schedule_id"):
                raise ValueError(f"Recomputed schedule identity drift for job {job_id}")
            schedule_path = str(job["schedule_path"])
            previous_path = replay_schedule_paths.setdefault(schedule_id, schedule_path)
            if previous_path != schedule_path:
                raise ValueError(f"One schedule identity has multiple paths: {schedule_id}")
            schedule = load_schedule(
                _archive_path(root, schedule_path, label="schedule")
            )
            if (
                schedule.metadata.get("method_version") != METHOD_VERSION
                or int(schedule.metadata.get("rngseed", -1)) != rng_seed
                or int(schedule.metadata.get("iterations", -1)) != 10_000
                or schedule.metadata.get("market_fingerprint")
                != panel.market_fingerprint
                or tuple(schedule.seeds) != panel.seeds
            ):
                raise ValueError(f"Schedule metadata drift for job {job_id}")
            result = run_bootstrap(panel, schedule, estimand=job["estimand"])
            common = _common_fields(job, panel, schedule_id)
            if (
                common["pair_count"] != job.get("pair_count")
                or common["session_count"] != job.get("session_count")
            ):
                raise ValueError(f"Archived panel counts drift for job {job_id}")
            replay_arms.extend(_arm_rows(result, common))
            rows, log_draws, job_legacy = _contrast_rows(
                result, common, normalised_recipes[job_id]
            )
            replay_contrasts.extend(rows)
            replay_legacy.update(job_legacy)
            draw_path = _archive_path(root, job["draws_path"], label="draw")
            with np.load(draw_path, allow_pickle=False) as saved:
                required_members = {
                    "draw_id",
                    "condition_names",
                    "observed_means",
                    "mean_draws",
                    "contrast_ids",
                    "log_ratio_draws",
                    "schedule_id",
                    "method_version",
                    "panel_fingerprint",
                    "estimand",
                }
                if set(saved.files) != required_members:
                    raise ValueError(f"Draw archive member drift for job {job_id}")
                if (
                    _npz_scalar_string(saved, "schedule_id") != schedule_id
                    or _npz_scalar_string(saved, "method_version") != METHOD_VERSION
                    or _npz_scalar_string(saved, "panel_fingerprint")
                    != panel.panel_fingerprint
                    or _npz_scalar_string(saved, "estimand") != str(job["estimand"])
                ):
                    raise ValueError(f"Draw archive metadata drift for job {job_id}")
                _assert_array_matches(
                    saved["draw_id"], schedule.draw_id, label=f"{job_id} draw IDs"
                )
                _assert_array_matches(
                    saved["condition_names"],
                    np.asarray(result.conditions),
                    label=f"{job_id} draw conditions",
                )
                _assert_array_matches(
                    saved["observed_means"],
                    result.observed_means,
                    label=f"{job_id} observed means",
                    atol=1e-14,
                )
                _assert_array_matches(
                    saved["mean_draws"],
                    result.mean_draws,
                    label=f"{job_id} arm draws",
                    atol=1e-14,
                )
                contrast_ids = [
                    recipe["contrast_id"] for recipe in normalised_recipes[job_id]
                ]
                _assert_array_matches(
                    saved["contrast_ids"],
                    np.asarray(contrast_ids, dtype=str),
                    label=f"{job_id} draw contrasts",
                )
                _assert_array_matches(
                    saved["log_ratio_draws"],
                    (np.column_stack(log_draws) if log_draws else
                     np.empty((iterations, 0), dtype=float)),
                    label=f"{job_id} contrast draws",
                    atol=1e-13,
                )

        expected_arms = pd.DataFrame(replay_arms)
        expected_summary = _holm(pd.DataFrame(replay_contrasts))
        expected_comparison = _old_new(expected_summary, replay_legacy)
        _assert_frame_matches(
            arms,
            expected_arms,
            keys=("job_id", "condition"),
            label="replayed arm summary",
        )
        _assert_frame_matches(
            summary,
            expected_summary,
            keys=("job_id", "contrast_id"),
            label="replayed contrast summary",
        )
        _assert_frame_matches(
            comparison,
            expected_comparison,
            keys=("job_id", "contrast_id"),
            label="replayed legacy comparison",
        )
        _assert_json_matches(
            values,
            _build_values(job_ids, expected_arms, expected_summary),
            label="replayed chapter3 values",
        )
        _assert_json_matches(
            qa,
            _qa_payload(
                jobs=job_records,
                arms=expected_arms,
                contrasts=expected_summary,
                comparison=expected_comparison,
            ),
            label="replayed terminal QA",
        )
        if report != _report_text(expected_arms, expected_summary, expected_comparison):
            raise ValueError("Replayed human-readable report drift")
    return {"kind": KIND, "passed": True, "replayed": replay,
            "jobs": len(manifest["jobs"]), "contrasts": len(summary)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        result = verify_analysis(args.output_root or DEFAULT_OUTPUT)
        print(json.dumps(result, sort_keys=True))
    else:
        print(run_analysis(args.config, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
