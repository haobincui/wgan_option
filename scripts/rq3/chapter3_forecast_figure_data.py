"""Read-only, hash-bound data preparation for Chapter 3 forecast figures.

The frozen CSVs already contain physical ACT/365 implied volatilities. This
module neither standardises nor inversely standardises them. Display code may
multiply IV by 100 to express it as a percentage.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


FOLDS = ("f1_2023q1", "f2_2023q2", "f3_2023q3", "f4_2023q4")
MONEYNESS_GRID = np.array(
    [0.970, 0.974, 0.978, 0.982, 0.986, 0.990, 0.994, 0.998,
     1.002, 1.006, 1.010, 1.014, 1.018, 1.022, 1.026, 1.030],
    dtype=np.float64,
)
MATURITY_GRID = np.array(
    [1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38],
    dtype=np.float64,
)
SURFACE_ELIGIBILITY = {
    "minimum_supported_cells": 32,
    "minimum_supported_maturity_rows": 4,
    "minimum_supported_moneyness_columns": 4,
    "minimum_complete_adjacent_quads": 12,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _check_hash(path: Path, expected: Any, label: str) -> str:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Missing or malformed {label} SHA256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {path}")
    return actual


@lru_cache(maxsize=1)
def _raw_support_module() -> Any:
    """Load support.py without importing the film_wgan trainer package."""
    repo_root = Path(__file__).resolve().parents[2]
    source_dir = str(repo_root / "src")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    name = "_chapter3_forecast_raw_support"
    spec = importlib.util.spec_from_file_location(name, repo_root / "src/film_wgan/support.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load frozen raw support semantics")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass annotations resolve their module here.
    spec.loader.exec_module(module)
    return module


def _json_array(value: Any, label: str) -> np.ndarray:
    try:
        payload = json.loads(value) if isinstance(value, str) else value
        return np.asarray(payload, dtype=np.float64)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid numeric JSON array for {label}") from exc


def _surface_array(value: Any, label: str, *, mask: bool = False) -> np.ndarray:
    array = _json_array(value, label)
    if array.shape not in {(256,), (16, 16)}:
        raise ValueError(f"{label} must have shape (256,) or (16, 16); got {array.shape}")
    array = array.reshape(16, 16)
    if mask:
        if not np.isfinite(array).all() or not np.isin(array, [0.0, 1.0]).all():
            raise ValueError(f"{label} must contain only finite binary mask values")
        return array.astype(bool)
    return array


def _validate_grids(moneyness: Any, maturities: Any) -> tuple[np.ndarray, np.ndarray]:
    m_grid = _json_array(moneyness, "strike_grid")
    q_grid = _json_array(maturities, "maturity_days_grid")
    if (m_grid.shape != (16,) or
            not np.allclose(m_grid, MONEYNESS_GRID, atol=1e-12, rtol=0.0)):
        raise ValueError("Moneyness grid differs from the frozen 16-column grid")
    if (q_grid.shape != (16,) or
            not np.array_equal(q_grid, MATURITY_GRID)):
        raise ValueError("Maturity grid differs from the frozen 16-row grid")
    return m_grid, q_grid


def support_statistics(mask: np.ndarray) -> dict[str, Any]:
    """Count supported vertices and complete adjacent 2x2 mesh faces."""
    mask = _surface_array(mask, "joint_mask", mask=True)
    cells = int(mask.sum())
    rows = int(mask.any(axis=1).sum())
    columns = int(mask.any(axis=0).sum())
    quads = int((mask[:-1, :-1] & mask[:-1, 1:] &
                 mask[1:, :-1] & mask[1:, 1:]).sum())
    return {
        "supported_cell_count": cells,
        "supported_maturity_rows": rows,
        "supported_moneyness_columns": columns,
        "supported_quad_count": quads,
        "surface_eligible": cells >= 32 and rows >= 4 and columns >= 4 and quads >= 12,
    }


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


def _unique_pairs(frame: pd.DataFrame, label: str) -> None:
    if frame["pair_id"].isna().any() or frame["pair_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"Missing pair_id in {label}")
    if frame["pair_id"].duplicated().any():
        raise ValueError(f"Duplicate pair_id in {label}; join must be one-to-one")


def load_frozen_inputs(
    source_root: Path, seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Validate and join the four original matched-LP FiLM prediction folds."""
    if seed != 42:
        raise ValueError("Chapter 3 figures use the frozen seed 42 predictions")
    root = Path(source_root).expanduser().resolve()
    support = _raw_support_module()
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    moneyness = MONEYNESS_GRID.copy()
    maturities = MATURITY_GRID.copy()
    pred_columns = [
        "pair_id", "session_id", "predicted_surface_flat", "support_mask_flat",
        "model", "arm", "fold", "seed", "tolerance_minutes", "prediction_status",
        "prediction_mc_samples", "prediction_fallback", "support_mask_mode",
        "text_ablation_mode", "text_information_path", "checkpoint_sha256",
        "noise_bank_profile_sha256", "supported_cell_count",
    ]
    panel_columns = [
        "pair_id", "session_id", "effective_origin_utc", "current_snapshot_time_utc",
        "target_snapshot_time_utc", "strike_grid", "maturity_days_grid",
        "current_surface_flat", "target_surface_flat", "current_surface_param_json",
        "target_surface_param_json",
    ]
    for fold in FOLDS:
        prediction_path = root / "evaluation/predictions/tolerance_05m" / fold / "seed_42/lp_matched.csv.gz"
        manifest_path = prediction_path.with_name("lp_matched.csv.manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in manifest.items() if key != "payload_sha256"}
        if _payload_sha256(unsigned) != manifest.get("payload_sha256"):
            raise ValueError(f"Prediction manifest payload SHA256 mismatch: {manifest_path}")
        for key, expected in {
            "kind": "rq123_prediction_manifest_v1", "arm": "lp_matched", "fold": fold,
            "seed": seed, "tolerance_minutes": 5, "prediction_mc_samples": 64,
        }.items():
            if manifest.get(key) != expected:
                raise ValueError(f"Frozen manifest {key} mismatch for {fold}")
        panel_path = Path(manifest["panel_path"])
        if not panel_path.is_absolute():
            panel_path = root / panel_path
        expected_panel = root / "evaluation/test_panels/tolerance_05m" / f"{fold}.csv.gz"
        if panel_path.resolve() != expected_panel.resolve():
            raise ValueError(f"Frozen manifest panel_path mismatch for {fold}")
        declared_prediction = Path(manifest["prediction_path"])
        if not declared_prediction.is_absolute():
            declared_prediction = root / declared_prediction
        if declared_prediction.resolve() != prediction_path.resolve():
            raise ValueError(f"Frozen manifest prediction_path mismatch for {fold}")
        prediction_hash = _check_hash(prediction_path, manifest.get("prediction_sha256"), "Prediction")
        panel_hash = _check_hash(panel_path, manifest.get("panel_sha256"), "Panel")
        checkpoint_path = Path(manifest["checkpoint_path"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        checkpoint_hash = _check_hash(checkpoint_path, manifest.get("checkpoint_sha256"), "Checkpoint")
        noise_hash = manifest.get("noise_bank_profile_sha256")
        if not isinstance(noise_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", noise_hash):
            raise ValueError(f"Missing or malformed noise bank profile SHA256 for {fold}")
        predictions = pd.read_csv(prediction_path)
        panel = pd.read_csv(panel_path)
        _require_columns(predictions, pred_columns, "prediction")
        _require_columns(panel, panel_columns, "panel")
        _unique_pairs(predictions, "predictions")
        _unique_pairs(panel, "panel")
        if set(predictions.pair_id) != set(panel.pair_id):
            raise ValueError(f"Unmatched pair_id values in prediction/panel join for {fold}")
        if len(predictions) != manifest.get("row_count"):
            raise ValueError(f"Prediction row count mismatch for {fold}")
        expected_fields = {
            "model": "wgan", "arm": "lp_matched", "fold": fold, "seed": seed,
            "tolerance_minutes": 5, "prediction_status": "ok", "prediction_mc_samples": 64,
            "support_mask_mode": "raw_joint", "text_ablation_mode": "real_text",
            "text_information_path": "current_surface_plus_real_lp_embedding",
            "checkpoint_sha256": checkpoint_hash, "noise_bank_profile_sha256": noise_hash,
        }
        for key, expected in expected_fields.items():
            if not predictions[key].eq(expected).all():
                raise ValueError(f"Frozen prediction {key} mismatch for {fold}")
        if not predictions.prediction_fallback.eq(False).all():
            raise ValueError(f"Prediction fallback present for {fold}")
        joined = predictions[pred_columns].merge(
            panel[panel_columns], on="pair_id", how="inner", validate="one_to_one",
            suffixes=("_prediction", "_panel"),
        )
        if not joined.session_id_prediction.eq(joined.session_id_panel).all() or joined.session_id_panel.isna().any():
            raise ValueError(f"Prediction/panel session_id mismatch for {fold}")
        for item in joined.to_dict("records"):
            pair_id = str(item["pair_id"])
            m_grid, q_grid = _validate_grids(item["strike_grid"], item["maturity_days_grid"])
            origin = pd.to_datetime(item["effective_origin_utc"], utc=True, errors="raise")
            current_time = pd.to_datetime(item["current_snapshot_time_utc"], utc=True, errors="raise")
            target = pd.to_datetime(item["target_snapshot_time_utc"], utc=True, errors="raise")
            if pd.isna(origin) or current_time != origin or target != origin + pd.Timedelta(minutes=5):
                raise ValueError(f"Current/origin/+5-minute target mismatch for {pair_id}")
            realised = _surface_array(item["target_surface_flat"], f"realised {pair_id}")
            current = _surface_array(item["current_surface_flat"], f"current {pair_id}")
            predicted = _surface_array(item["predicted_surface_flat"], f"predicted {pair_id}")
            mask = _surface_array(item["support_mask_flat"], f"joint mask {pair_id}", mask=True)
            current_mask = support.raw_support_mask(
                support.parse_raw_surface_params(item["current_surface_param_json"]),
                strike_grid=m_grid, maturity_days_grid=q_grid,
            )
            target_mask = support.raw_support_mask(
                support.parse_raw_surface_params(item["target_surface_param_json"]),
                strike_grid=m_grid, maturity_days_grid=q_grid,
            )
            if not np.array_equal(mask, current_mask & target_mask):
                raise ValueError(f"Raw joint support mask mismatch for {pair_id}")
            statistics = support_statistics(mask)
            if item["supported_cell_count"] != statistics["supported_cell_count"]:
                raise ValueError(f"Supported cell count mismatch for {pair_id}")
            for label, values in (("current", current), ("realised", realised), ("predicted", predicted)):
                if not np.isfinite(values[mask]).all() or np.any(values[mask] <= 0.0):
                    raise ValueError(f"Non-finite or non-positive supported {label} IV for {pair_id}")
            rows.append({
                "fold": fold, "pair_id": pair_id, "session_id": item["session_id_panel"],
                "origin_time_utc": origin, "target_time_utc": target,
                "realised": realised, "predicted": predicted, "joint_mask": mask,
                "target_mask": target_mask,
                **statistics,
            })
        sources.append({
            "fold": fold, "row_count": len(predictions),
            "prediction_path": str(prediction_path), "prediction_sha256": prediction_hash,
            "panel_path": str(panel_path), "panel_sha256": panel_hash,
            "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "manifest_payload_sha256": manifest["payload_sha256"],
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
            "noise_bank_profile_sha256": noise_hash,
            "inference_determinism_contract_sha256": manifest.get("inference_determinism_contract_sha256"),
            "test_overlay_sha256": manifest.get("test_overlay_sha256"),
        })
    pairs = pd.DataFrame(rows).sort_values(["fold", "pair_id"], kind="stable").reset_index(drop=True)
    _unique_pairs(pairs, "all four folds")
    provenance = {
        "source_root": str(root), "arm": "lp_matched", "seed": seed,
        "prediction_mc_samples": 64, "iv_units": "physical ACT/365 decimal IV",
        "support_method": "raw_bracket_intersection_v1", "support_masks_recomputed": True,
        "sources": sources, "pair_count": len(pairs),
        "pair_count_by_fold": {fold: int(pairs.fold.eq(fold).sum()) for fold in FOLDS},
        "surface_eligible_count": int(pairs.surface_eligible.sum()),
        "surface_eligibility": dict(SURFACE_ELIGIBILITY),
        "moneyness_grid": moneyness.tolist(), "maturity_days_grid": maturities.tolist(),
    }
    return pairs, moneyness, maturities, provenance


def select_surface_pair(pairs: pd.DataFrame, selection_seed: int = 20261008) -> pd.Series:
    """Uniformly select from eligible pairs in a stable, outcome-blind order."""
    eligible = pairs.loc[pairs.surface_eligible].sort_values(["fold", "pair_id"], kind="stable")
    if eligible.empty:
        raise ValueError("No surface pair passes the predeclared support eligibility rule")
    position = int(np.random.default_rng(selection_seed).choice(len(eligible)))
    return eligible.iloc[position].copy()


def with_atm_values(
    pairs: pd.DataFrame, moneyness: np.ndarray, maturities: np.ndarray,
    maturity_days: int = 21,
) -> pd.DataFrame:
    """Add supported ATM IV using variance interpolation; retain every pair."""
    m_grid, q_grid = _validate_grids(moneyness, maturities)
    exact = np.flatnonzero(q_grid == maturity_days)
    if len(exact) != 1:
        raise ValueError("ATM maturity must be an exact frozen grid row")
    row = int(exact[0])
    right = int(np.searchsorted(m_grid, 1.0))
    left = right - 1
    weight = float((1.0 - m_grid[left]) / (m_grid[right] - m_grid[left]))
    valid: list[bool] = []
    realised: list[float] = []
    predicted: list[float] = []
    for item in pairs.to_dict("records"):
        mask = _surface_array(item["joint_mask"], "ATM joint mask", mask=True)
        is_valid = bool(mask[row, left] and mask[row, right])
        valid.append(is_valid)
        if not is_valid:
            realised.append(float("nan"))
            predicted.append(float("nan"))
            continue
        for key, values in (("realised", realised), ("predicted", predicted)):
            surface = _surface_array(item[key], f"ATM {key}")
            endpoints = surface[row, [left, right]]
            if not np.isfinite(endpoints).all() or np.any(endpoints <= 0.0):
                raise ValueError(f"Invalid supported ATM {key} endpoints")
            values.append(float(np.sqrt((1.0 - weight) * endpoints[0] ** 2 + weight * endpoints[1] ** 2)))
    result = pairs.copy()
    result["atm_valid"] = valid
    result["atm_realised_iv"] = realised
    result["atm_predicted_iv"] = predicted
    return result


def detail_link_indices(detail: pd.DataFrame, max_gap_minutes: float = 5) -> list[tuple[int, int]]:
    """Link adjacent original observations only, retaining unsupported breaks.

    Returned positions refer to ``detail.sort_values(["target_time_utc",
    "pair_id"]).reset_index(drop=True)``. Invalid rows are never removed.
    """
    if max_gap_minutes <= 0:
        raise ValueError("Maximum detail gap must be positive")
    ordered = detail.sort_values(["target_time_utc", "pair_id"], kind="stable").reset_index(drop=True)
    times = pd.to_datetime(ordered.target_time_utc, utc=True, errors="raise")
    valid = ordered.atm_valid.eq(True).fillna(False)
    links: list[tuple[int, int]] = []
    for position in range(1, len(ordered)):
        previous = position - 1
        gap = times.iloc[position] - times.iloc[previous]
        if (valid.iloc[previous] and valid.iloc[position]
                and ordered.session_id.iloc[previous] == ordered.session_id.iloc[position]
                and pd.Timedelta(0) < gap <= pd.Timedelta(minutes=max_gap_minutes)):
            links.append((previous, position))
    return links
