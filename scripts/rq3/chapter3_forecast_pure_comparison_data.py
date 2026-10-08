"""Add original Pure-CNN frozen forecasts to the Chapter 3 figure panel.

The loader keeps the FiLM panel and its support semantics unchanged. Pure-CNN
identity is established from its saved architecture configuration, because the
generic frozen prediction schema also contains text-related bookkeeping fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scripts.rq3.chapter3_forecast_figure_data import (
    FOLDS,
    _check_hash,
    _payload_sha256,
    _require_columns,
    _surface_array,
    _unique_pairs,
    _validate_grids,
    load_frozen_inputs,
    sha256_file,
    with_atm_values,
)


DEFAULT_PURE_SOURCE_ROOT = (
    Path(__file__).resolve().parents[2] / "outputs/experiments"
    / "rq12_news_first_vol_cnn_unet_c32_nolp_pure_no_text_10seed_exact_ttm_rolling_v1"
)


def _source_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _architecture_source(checkpoint: Path, seed: int) -> dict[str, Any]:
    """Read the configuration and selection record next to this checkpoint."""
    run_root = checkpoint.parents[1]
    config_path = run_root / "metrics/training_resolved_config.yaml"
    selection_path = run_root / "metrics/best_learned_checkpoint.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid Pure-CNN training configuration: {config_path}")
    expected = {
        "generator_conditioning_mode": "cnn_unet_mask_coords_v1",
        "critic_conditioning_mode": "lp_disabled_same_shape_v1",
    }
    for name, value in expected.items():
        if config.get(name) != value or selection.get(name) != value:
            raise ValueError(f"Pure-CNN architecture {name} mismatch: {run_root}")
    if config.get("seed") != seed:
        raise ValueError(f"Pure-CNN configuration seed mismatch: {config_path}")
    if config.get("news_first_text_information_path") != "pure_cnn_no_text":
        raise ValueError(f"Pure-CNN training information path mismatch: {config_path}")
    selected = selection.get("artifacts", {}).get("generator")
    if not selected or _source_path(selected, run_root) != checkpoint:
        raise ValueError(f"Pure-CNN selection record checkpoint mismatch: {selection_path}")
    architecture_hash = config.get("news_first_architecture_profile_sha256")
    if architecture_hash != selection.get("architecture_profile_sha256"):
        raise ValueError(f"Pure-CNN architecture profile mismatch: {run_root}")
    return {
        "training_configuration_path": str(config_path),
        "training_configuration_sha256": sha256_file(config_path),
        "checkpoint_selection_path": str(selection_path),
        "checkpoint_selection_sha256": sha256_file(selection_path),
        "generator_conditioning_mode": expected["generator_conditioning_mode"],
        "critic_conditioning_mode": expected["critic_conditioning_mode"],
        "training_text_information_path": "pure_cnn_no_text",
        "architecture_profile_sha256": architecture_hash,
        "identity_basis": "saved training configuration and checkpoint selection record",
    }


def load_pure_comparison_inputs(
    film_source_root: Path,
    pure_source_root: Path = DEFAULT_PURE_SOURCE_ROOT,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the validated FiLM panel with a physical-IV Pure-CNN surface.

    ``pure_predicted`` uses the same 16 maturity rows and 16 moneyness columns
    as ``realised`` and ``predicted``. No rescaling or interpolation occurs.
    """
    pairs, moneyness, maturities, film_provenance = load_frozen_inputs(
        film_source_root, seed=seed,
    )
    pure_root = Path(pure_source_root).expanduser().resolve()
    film_sources = {item["fold"]: item for item in film_provenance["sources"]}
    pure_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    required = [
        "pair_id", "session_id", "model", "arm", "fold", "seed",
        "tolerance_minutes", "prediction_mc_samples", "prediction_status",
        "prediction_fallback", "support_mask_mode", "support_mask_flat",
        "supported_cell_count", "predicted_surface_flat", "checkpoint_sha256",
        "noise_bank_profile_sha256",
    ]
    for fold in FOLDS:
        prediction_path = (pure_root / "evaluation/predictions/tolerance_05m"
                           / fold / "seed_42/pure_cnn_no_text.csv.gz")
        manifest_path = prediction_path.with_name("pure_cnn_no_text.csv.manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in manifest.items() if key != "payload_sha256"}
        if _payload_sha256(unsigned) != manifest.get("payload_sha256"):
            raise ValueError(f"Pure-CNN manifest payload SHA256 mismatch: {manifest_path}")
        for key, expected in {
            "kind": "rq123_prediction_manifest_v1", "arm": "pure_cnn_no_text",
            "fold": fold, "seed": seed, "tolerance_minutes": 5,
            "prediction_mc_samples": 64,
        }.items():
            if manifest.get(key) != expected:
                raise ValueError(f"Pure-CNN manifest {key} mismatch for {fold}")
        if _source_path(manifest["prediction_path"], pure_root) != prediction_path:
            raise ValueError(f"Pure-CNN manifest prediction path mismatch for {fold}")
        panel_path = _source_path(manifest["panel_path"], pure_root)
        expected_panel = pure_root / "evaluation/test_panels/tolerance_05m" / f"{fold}.csv.gz"
        if panel_path != expected_panel:
            raise ValueError(f"Pure-CNN manifest panel path mismatch for {fold}")
        checkpoint = _source_path(manifest["checkpoint_path"], pure_root)
        prediction_hash = _check_hash(prediction_path, manifest.get("prediction_sha256"), "Pure-CNN prediction")
        panel_hash = _check_hash(panel_path, manifest.get("panel_sha256"), "Pure-CNN panel")
        checkpoint_hash = _check_hash(checkpoint, manifest.get("checkpoint_sha256"), "Pure-CNN checkpoint")
        film_source = film_sources[fold]
        if panel_hash != film_source["panel_sha256"]:
            raise ValueError(f"FiLM/Pure-CNN frozen panel SHA256 mismatch for {fold}")
        noise_hash = manifest.get("noise_bank_profile_sha256")
        if noise_hash != film_source["noise_bank_profile_sha256"]:
            raise ValueError(f"FiLM/Pure-CNN frozen noise-bank profile mismatch for {fold}")
        architecture = _architecture_source(checkpoint, seed)
        predictions = pd.read_csv(prediction_path)
        panel = pd.read_csv(panel_path, usecols=["pair_id", "session_id"])
        _require_columns(predictions, required, "Pure-CNN prediction")
        _unique_pairs(predictions, "Pure-CNN predictions")
        _unique_pairs(panel, "Pure-CNN panel")
        reference = pairs.loc[pairs.fold.eq(fold)]
        if (set(predictions.pair_id) != set(panel.pair_id)
                or set(predictions.pair_id) != set(reference.pair_id)):
            raise ValueError(f"Unmatched FiLM/Pure-CNN pair_id values for {fold}")
        if len(predictions) != manifest.get("row_count"):
            raise ValueError(f"Pure-CNN prediction row count mismatch for {fold}")
        for key, expected in {
            "model": "wgan", "arm": "pure_cnn_no_text", "fold": fold,
            "seed": seed, "tolerance_minutes": 5, "prediction_mc_samples": 64,
            "prediction_status": "ok", "support_mask_mode": "raw_joint",
            "checkpoint_sha256": checkpoint_hash, "noise_bank_profile_sha256": noise_hash,
        }.items():
            if not predictions[key].eq(expected).all():
                raise ValueError(f"Pure-CNN prediction {key} mismatch for {fold}")
        if not predictions.prediction_fallback.eq(False).all():
            raise ValueError(f"Pure-CNN prediction fallback present for {fold}")
        joined = predictions[required].merge(
            panel, on="pair_id", validate="one_to_one", suffixes=("", "_panel"),
        ).merge(
            reference[["pair_id", "session_id", "joint_mask"]],
            on="pair_id", validate="one_to_one", suffixes=("", "_film"),
        )
        if (not joined.session_id.eq(joined.session_id_panel).all()
                or not joined.session_id.eq(joined.session_id_film).all()
                or joined.session_id.isna().any()):
            raise ValueError(f"FiLM/Pure-CNN prediction/panel session mismatch for {fold}")
        for item in joined.to_dict("records"):
            mask = _surface_array(item["support_mask_flat"], "Pure-CNN support", mask=True)
            if not np.array_equal(mask, item["joint_mask"]):
                raise ValueError(f"FiLM/Pure-CNN joint-support mismatch for {item['pair_id']}")
            if item["supported_cell_count"] != int(mask.sum()):
                raise ValueError(f"Pure-CNN supported cell count mismatch for {item['pair_id']}")
            surface = _surface_array(item["predicted_surface_flat"], "Pure-CNN predicted IV")
            if not np.isfinite(surface[mask]).all() or np.any(surface[mask] <= 0.0):
                raise ValueError(f"Invalid supported Pure-CNN predicted IV for {item['pair_id']}")
            pure_rows.append({"fold": fold, "pair_id": item["pair_id"], "pure_predicted": surface})
        sources.append({
            "fold": fold, "row_count": len(predictions),
            "prediction_path": str(prediction_path), "prediction_sha256": prediction_hash,
            "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "manifest_payload_sha256": manifest["payload_sha256"],
            "panel_path": str(panel_path), "panel_sha256": panel_hash,
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "noise_bank_profile_sha256": noise_hash,
            "identical_film_panel_sha256": film_source["panel_sha256"],
            "architecture_source": architecture,
        })
    pure_pairs = pd.DataFrame(pure_rows)
    _unique_pairs(pure_pairs, "Pure-CNN all four folds")
    result = pairs.merge(pure_pairs, on=["fold", "pair_id"], validate="one_to_one", how="left")
    if len(result) != len(pairs) or result.pure_predicted.isna().any():
        raise ValueError("Incomplete FiLM/Pure-CNN frozen prediction merge")
    provenance = {
        **film_provenance,
        "pure_cnn": {
            "source_root": str(pure_root), "arm": "pure_cnn_no_text", "seed": seed,
            "prediction_mc_samples": 64, "pair_count": len(pure_pairs),
            "iv_units": "physical ACT/365 decimal IV", "sources": sources,
            "panels_identical_to_film": True, "joint_masks_identical_to_film": True,
            "noise_bank_profiles_identical_to_film": True,
        },
    }
    return result, moneyness, maturities, provenance


def with_pure_atm_values(
    pairs: pd.DataFrame, moneyness: np.ndarray, maturities: np.ndarray,
    maturity_days: int = 21,
) -> pd.DataFrame:
    """Add Pure-CNN ATM variance interpolation under the unchanged joint mask."""
    result = with_atm_values(pairs, moneyness, maturities, maturity_days=maturity_days)
    m_grid, q_grid = _validate_grids(moneyness, maturities)
    row = int(np.flatnonzero(q_grid == maturity_days)[0])
    right = int(np.searchsorted(m_grid, 1.0))
    left = right - 1
    weight = float((1.0 - m_grid[left]) / (m_grid[right] - m_grid[left]))
    values: list[float] = []
    for item in result.to_dict("records"):
        if not item["atm_valid"]:
            values.append(float("nan"))
            continue
        surface = _surface_array(item["pure_predicted"], "Pure-CNN ATM surface")
        endpoints = surface[row, [left, right]]
        if not np.isfinite(endpoints).all() or np.any(endpoints <= 0.0):
            raise ValueError(f"Invalid supported Pure-CNN ATM endpoints for {item['pair_id']}")
        values.append(float(np.sqrt((1.0 - weight) * endpoints[0] ** 2 + weight * endpoints[1] ** 2)))
    result["atm_pure_iv"] = values
    return result
