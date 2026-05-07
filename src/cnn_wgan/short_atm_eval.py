"""Post-hoc short-end ATM evaluation for CNN WGAN sample JSON outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ShortATMEvalConfig:
    """Mask and weighting parameters for post-hoc short-ATM evaluation."""

    atm_short_range: float = 0.04
    atm_short_max_days: float = 60.0
    recon_atm_range: float = 0.04
    recon_atm_short_end_max_days: float = 60.0
    recon_atm_multiplier: float = 16.0
    label: str = "short_atm_posthoc"


def _as_surface(payload: Mapping[str, Any], key: str) -> np.ndarray:
    surface = np.asarray(payload[key], dtype=np.float64)
    if surface.ndim != 2:
        raise ValueError(f"{key} must be a 2D surface, got shape {surface.shape}")
    return surface


def _target_surface(payload: Mapping[str, Any]) -> np.ndarray:
    if "target_surface" in payload:
        return _as_surface(payload, "target_surface")
    if "real_surface" in payload:
        return _as_surface(payload, "real_surface")
    raise KeyError("Sample payload must include target_surface or real_surface.")


def _mask(
    *,
    strike_grid: np.ndarray,
    maturity_days_grid: np.ndarray,
    atm_range: float,
    max_days: float,
) -> np.ndarray:
    if float(atm_range) < 0.0:
        raise ValueError(f"atm_range must be non-negative, got {atm_range}")
    if float(max_days) <= 0.0:
        raise ValueError(f"max_days must be positive, got {max_days}")
    strike_mask = np.abs(strike_grid.reshape(1, -1) - 1.0) <= float(atm_range) + 1e-6
    maturity_mask = maturity_days_grid.reshape(-1, 1) <= float(max_days) + 1e-6
    output = strike_mask & maturity_mask
    if not bool(output.any()):
        raise ValueError(
            "Short-ATM mask selected no cells. "
            f"atm_range={atm_range} max_days={max_days} "
            f"strike_grid={strike_grid.tolist()} maturity_days_grid={maturity_days_grid.tolist()}"
        )
    return output


def _weighted_mae(predicted: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    if predicted.shape != target.shape:
        raise ValueError(f"predicted and target shapes differ: {predicted.shape} vs {target.shape}")
    if predicted.shape != weights.shape:
        raise ValueError(f"weights shape differs from surface shape: {weights.shape} vs {predicted.shape}")
    weighted_abs = np.abs(predicted - target) * weights
    return float(weighted_abs.sum() / max(float(weights.sum()), 1e-12))


def _surface_mae(predicted: np.ndarray, target: np.ndarray) -> float:
    if predicted.shape != target.shape:
        raise ValueError(f"predicted and target shapes differ: {predicted.shape} vs {target.shape}")
    return float(np.mean(np.abs(predicted - target)))


def _sample_identity(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    return {
        "sample_id": str(payload.get("sample_id", path.stem)),
        "global_index": int(payload.get("global_index", -1)),
        "news_timestamp_utc": str(payload.get("news_timestamp_utc", "")),
    }


def evaluate_sample_payload(
    payload: Mapping[str, Any],
    *,
    config: ShortATMEvalConfig,
    source_path: str | Path = "",
) -> dict[str, Any]:
    """Evaluate one generated sample payload with the configured short-ATM masks."""

    strike_grid = np.asarray(payload["strike_grid"], dtype=np.float64).reshape(-1)
    maturity_days_grid = np.asarray(payload["maturity_days_grid"], dtype=np.float64).reshape(-1)
    generated = _as_surface(payload, "generated_surface")
    current = _as_surface(payload, "current_surface")
    target = _target_surface(payload)

    expected_shape = (maturity_days_grid.size, strike_grid.size)
    for name, surface in (("generated_surface", generated), ("current_surface", current), ("target_surface", target)):
        if surface.shape != expected_shape:
            raise ValueError(f"{name} shape {surface.shape} does not match grid shape {expected_shape}")

    pure_mask = _mask(
        strike_grid=strike_grid,
        maturity_days_grid=maturity_days_grid,
        atm_range=float(config.atm_short_range),
        max_days=float(config.atm_short_max_days),
    )
    recon_mask = _mask(
        strike_grid=strike_grid,
        maturity_days_grid=maturity_days_grid,
        atm_range=float(config.recon_atm_range),
        max_days=float(config.recon_atm_short_end_max_days),
    )
    weights = np.ones(expected_shape, dtype=np.float64)
    weights[recon_mask] = float(config.recon_atm_multiplier)
    if float(config.recon_atm_multiplier) <= 0.0:
        raise ValueError(f"recon_atm_multiplier must be positive, got {config.recon_atm_multiplier}")

    mae = _surface_mae(generated, target)
    current_mae = _surface_mae(current, target)
    short_atm_weighted_mae = _weighted_mae(generated, target, weights)
    current_short_atm_weighted_mae = _weighted_mae(current, target, weights)
    atm_short_pure_mae = _surface_mae(generated[pure_mask], target[pure_mask])
    current_atm_short_pure_mae = _surface_mae(current[pure_mask], target[pure_mask])

    return {
        **_sample_identity(payload, Path(source_path)),
        "source_path": str(source_path),
        "label": str(config.label),
        "mae": mae,
        "current_mae": current_mae,
        "mae_gap_vs_current": mae - current_mae,
        "win_flag_vs_current": 1.0 if mae < current_mae else 0.0,
        "short_atm_weighted_mae": short_atm_weighted_mae,
        "current_short_atm_weighted_mae": current_short_atm_weighted_mae,
        "short_atm_mae_gap_vs_current": short_atm_weighted_mae - current_short_atm_weighted_mae,
        "short_atm_weighted_win_flag_vs_current": 1.0
        if short_atm_weighted_mae < current_short_atm_weighted_mae
        else 0.0,
        "atm_short_pure_mae": atm_short_pure_mae,
        "current_atm_short_pure_mae": current_atm_short_pure_mae,
        "atm_short_pure_mae_gap_vs_current": atm_short_pure_mae - current_atm_short_pure_mae,
        "atm_short_pure_win_flag_vs_current": 1.0 if atm_short_pure_mae < current_atm_short_pure_mae else 0.0,
        "atm_short_range": float(config.atm_short_range),
        "atm_short_max_days": float(config.atm_short_max_days),
        "recon_atm_range": float(config.recon_atm_range),
        "recon_atm_short_end_max_days": float(config.recon_atm_short_end_max_days),
        "recon_atm_multiplier": float(config.recon_atm_multiplier),
        "pure_mask_cell_count": int(pure_mask.sum()),
        "weighted_mask_cell_count": int(recon_mask.sum()),
    }


def evaluate_sample_file(path: str | Path, *, config: ShortATMEvalConfig) -> dict[str, Any]:
    sample_path = Path(path)
    with sample_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return evaluate_sample_payload(payload, config=config, source_path=sample_path)


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return float(np.mean(values)) if values else 0.0


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, config: ShortATMEvalConfig, samples_dir: str | Path) -> dict[str, Any]:
    """Aggregate per-sample post-hoc metrics."""

    metric_keys = (
        "mae",
        "current_mae",
        "mae_gap_vs_current",
        "win_flag_vs_current",
        "short_atm_weighted_mae",
        "current_short_atm_weighted_mae",
        "short_atm_mae_gap_vs_current",
        "short_atm_weighted_win_flag_vs_current",
        "atm_short_pure_mae",
        "current_atm_short_pure_mae",
        "atm_short_pure_mae_gap_vs_current",
        "atm_short_pure_win_flag_vs_current",
    )
    return {
        "label": str(config.label),
        "samples_dir": str(samples_dir),
        "sample_count": int(len(rows)),
        **asdict(config),
        **{key: _mean(rows, key) for key in metric_keys},
    }


def _json_files(samples_dir: str | Path) -> list[Path]:
    root = Path(samples_dir)
    if not root.exists():
        raise FileNotFoundError(f"Samples directory does not exist: {root}")
    files = sorted(path for path in root.glob("*.json") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No sample JSON files found in {root}")
    return files


def evaluate_samples_dir(
    samples_dir: str | Path,
    *,
    config: ShortATMEvalConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate every sample JSON in a generate-result samples directory."""

    rows = [evaluate_sample_file(path, config=config) for path in _json_files(samples_dir)]
    return rows, summarize_rows(rows, config=config, samples_dir=samples_dir)


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=False)
        handle.write("\n")
    return output_path


def write_evaluation_outputs(
    *,
    output_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    """Write per-sample and aggregate post-hoc evaluation artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(output_path / "summary.json", summary)
    per_sample_path = write_csv_rows(output_path / "per_sample.csv", rows)
    return {
        "summary_json": str(summary_path),
        "per_sample_csv": str(per_sample_path),
    }


def format_summary(summary: Mapping[str, Any]) -> str:
    """Return a compact human-readable summary for CLI output."""

    keys = (
        "label",
        "sample_count",
        "mae",
        "current_mae",
        "mae_gap_vs_current",
        "short_atm_weighted_mae",
        "current_short_atm_weighted_mae",
        "short_atm_mae_gap_vs_current",
        "atm_short_pure_mae",
        "current_atm_short_pure_mae",
        "atm_short_pure_mae_gap_vs_current",
        "atm_short_pure_win_flag_vs_current",
    )
    lines: list[str] = []
    for key in keys:
        if key not in summary:
            continue
        value = summary[key]
        if isinstance(value, float):
            lines.append(f"{key}: {value:.12f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def config_from_mapping(values: Mapping[str, Any]) -> ShortATMEvalConfig:
    return ShortATMEvalConfig(
        atm_short_range=float(values.get("atm_short_range", ShortATMEvalConfig.atm_short_range)),
        atm_short_max_days=float(values.get("atm_short_max_days", ShortATMEvalConfig.atm_short_max_days)),
        recon_atm_range=float(values.get("recon_atm_range", ShortATMEvalConfig.recon_atm_range)),
        recon_atm_short_end_max_days=float(
            values.get("recon_atm_short_end_max_days", ShortATMEvalConfig.recon_atm_short_end_max_days)
        ),
        recon_atm_multiplier=float(values.get("recon_atm_multiplier", ShortATMEvalConfig.recon_atm_multiplier)),
        label=str(values.get("label", ShortATMEvalConfig.label)),
    )


def evaluate_many(
    *,
    samples_dir: str | Path,
    config_values: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    """Convenience wrapper used by the CLI and tests."""

    config = config_from_mapping(config_values)
    rows, summary = evaluate_samples_dir(samples_dir, config=config)
    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = write_evaluation_outputs(output_dir=output_dir, rows=rows, summary=summary)
    return rows, summary, artifacts
