"""Generate future SVI params, reconstruct surfaces, and compare them."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlib.calendar.daycount import DayCountBusN  # noqa: E402
from quantlib.calendar.holidays import embedded_calendar  # noqa: E402
from quantlib.vol_surface.algo.svi_surface import SviVolSurface  # noqa: E402
from scripts.generate_result.common import (  # noqa: E402
    build_surface_grids,
    compute_surface_metrics,
    prepare_run_output_dir,
    resolve_checkpoint_path,
    resolve_result_config,
    safe_sample_filename,
    save_payload_json,
    select_samples_from_split,
    split_metadata,
    write_resolved_config,
    write_summary_csv,
)
from scripts.generate_result.plot_surface import plot_surface_payload  # noqa: E402
from wgan_option.models.svi_regressor import SviRegressor  # noqa: E402
from wgan_option.utils.merged_xlsx import (  # noqa: E402
    SVI_FEATURE_ORDER,
    _build_svi_matrix_from_params,
    _normalize_svi_matrix,
    load_svi_paired_samples,
    select_ordered_split,
)

_RECONSTRUCTION_DAYCOUNT = DayCountBusN("BUS250", 250, embedded_calendar())
_RECONSTRUCTION_VALUATION_DATE = date(2023, 1, 2)


def _load_svi_regressor(checkpoint_path: Path, device: torch.device) -> tuple[SviRegressor, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SviRegressor(
        current_input_dim=int(checkpoint["current_input_dim"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        regression_dim=int(checkpoint["regression_dim"]),
        count_classes=int(checkpoint["max_slices"]),
        hidden_dim=int(checkpoint.get("config", {}).get("svi_hidden_dim", 256)),
        dropout=float(checkpoint.get("config", {}).get("svi_dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _build_current_feature_vector(
    svi_params: Dict[str, List[float]],
    *,
    max_slices: int,
    normalization_stats: Dict[str, object],
    sample_label: str,
) -> Tuple[np.ndarray, int]:
    matrix, mask, count = _build_svi_matrix_from_params(svi_params, max_slices, sample_label=sample_label)
    normalized = _normalize_svi_matrix(matrix, mask, normalization_stats)
    feature_vector = np.concatenate(
        [
            normalized.reshape(-1),
            mask.astype(np.float32),
            np.asarray([float(count) / float(max_slices)], dtype=np.float32),
        ],
        axis=0,
    )
    return feature_vector.astype(np.float32), count


def _sanitize_svi_params(svi_params: Dict[str, List[float]]) -> Dict[str, List[float]]:
    rows: List[Dict[str, float]] = []
    slice_count = len(svi_params["business_days"])
    for idx in range(slice_count):
        rows.append({feature: float(svi_params[feature][idx]) for feature in SVI_FEATURE_ORDER})

    rows.sort(key=lambda row: row["business_days"])
    previous_day = 0
    sanitized: List[Dict[str, float]] = []
    for row in rows:
        business_day = max(1, int(round(row["business_days"])))
        if business_day <= previous_day:
            business_day = previous_day + 1
        previous_day = business_day
        sanitized.append(
            {
                "business_days": float(business_day),
                "a": float(row["a"]),
                "b": float(row["b"]),
                "rho": float(row["rho"]),
                "m": float(row["m"]),
                "sigma": max(float(row["sigma"]), 1e-8),
            }
        )

    return {
        feature: [float(row[feature]) for row in sanitized]
        for feature in SVI_FEATURE_ORDER
    }


def _denormalize_prediction(
    predicted_regression: np.ndarray,
    *,
    predicted_count: int,
    normalization_stats: Dict[str, object],
    max_slices: int,
) -> Dict[str, List[float]]:
    feature_dim = len(SVI_FEATURE_ORDER)
    mean = np.asarray(normalization_stats["mean"], dtype=np.float32)
    std = np.asarray(normalization_stats["std"], dtype=np.float32)
    regression_matrix = predicted_regression.reshape(int(max_slices), feature_dim)
    denormalized = regression_matrix * std.reshape(1, feature_dim) + mean.reshape(1, feature_dim)

    raw_params = {
        feature: denormalized[:predicted_count, feature_idx].astype(float).tolist()
        for feature_idx, feature in enumerate(SVI_FEATURE_ORDER)
    }
    return _sanitize_svi_params(raw_params)


def _reconstruct_surface(
    svi_params: Dict[str, List[float]],
    *,
    strike_grid: np.ndarray,
    maturity_days_grid: np.ndarray,
) -> np.ndarray:
    surface = SviVolSurface(
        valuation_date=_RECONSTRUCTION_VALUATION_DATE,
        svi_params=_sanitize_svi_params(svi_params),
        vol_daycount=_RECONSTRUCTION_DAYCOUNT,
    )
    grid = surface.implied_vol_surface(
        percent_strikes=[float(value) for value in strike_grid.tolist()],
        business_days=[int(round(value)) for value in maturity_days_grid.tolist()],
        forward=1.0,
    )
    return np.asarray(grid, dtype=np.float32)


def main(argv: Optional[Iterable[str]] = None) -> Path:
    config = resolve_result_config(
        argv=argv,
        description="Generate future SVI params, reconstruct surfaces, and compare them.",
        default_config_path="configs/generate_result/svi.yaml",
    )

    all_samples = load_svi_paired_samples(config)
    split_selection = select_ordered_split(all_samples, train_ratio=config.train_ratio, split=config.split)
    selected_samples = select_samples_from_split(split_selection, config)

    run_dir = prepare_run_output_dir(config.output_dir)
    write_resolved_config(config, run_dir)
    checkpoint_path = resolve_checkpoint_path(
        config,
        artifact_key="model",
        fallback_filenames=("svi_regressor_best.pt", "svi_regressor.pt"),
    )

    device = torch.device("cuda:0" if (config.cuda and torch.cuda.is_available()) else "cpu")
    model, checkpoint = _load_svi_regressor(checkpoint_path, device)
    embedding_dim = int(checkpoint["embedding_dim"])
    max_slices = int(checkpoint["max_slices"])
    normalization_stats = checkpoint["normalization_stats"]

    strike_grid, maturity_days_grid = build_surface_grids(config)
    sample_json_dir = run_dir / "samples"
    plot_dir = run_dir / "plots"
    summary_rows = []
    split_meta = split_metadata(split_selection)

    for sample in selected_samples:
        if int(sample.text_embedding.size) != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch for sample {sample.sample_id}: "
                f"expected {embedding_dim}, got {sample.text_embedding.size}"
            )

        current_features, current_count = _build_current_feature_vector(
            sample.current_svi,
            max_slices=max_slices,
            normalization_stats=normalization_stats,
            sample_label=f"{sample.sample_id}:current",
        )
        current_tensor = torch.tensor(current_features, dtype=torch.float32, device=device).unsqueeze(0)
        text_tensor = torch.tensor(sample.text_embedding, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            predicted_regression, predicted_count_logits = model(current_tensor, text_tensor)

        predicted_regression_np = predicted_regression.detach().cpu().numpy()[0]
        predicted_count = int(torch.argmax(predicted_count_logits, dim=1).item()) + 1
        predicted_svi = _denormalize_prediction(
            predicted_regression_np,
            predicted_count=predicted_count,
            normalization_stats=normalization_stats,
            max_slices=max_slices,
        )

        current_surface = _reconstruct_surface(sample.current_svi, strike_grid=strike_grid, maturity_days_grid=maturity_days_grid)
        generated_surface = _reconstruct_surface(predicted_svi, strike_grid=strike_grid, maturity_days_grid=maturity_days_grid)
        real_surface = _reconstruct_surface(sample.future_svi, strike_grid=strike_grid, maturity_days_grid=maturity_days_grid)
        metrics = compute_surface_metrics(generated_surface, real_surface)

        payload = {
            "sample_id": sample.sample_id,
            "mode": "svi",
            "strike_grid": strike_grid.astype(float).tolist(),
            "maturity_days_grid": maturity_days_grid.astype(float).tolist(),
            "current_surface": current_surface.astype(float).tolist(),
            "generated_surface": generated_surface.astype(float).tolist(),
            "real_surface": real_surface.astype(float).tolist(),
            "metrics": metrics,
            "metadata": {
                "checkpoint_path": str(checkpoint_path),
                "global_index": int(sample.global_index),
                "news_row_id": int(sample.news_row_id),
                "news_timestamp_utc": sample.timestamp,
                "current_timestamp_utc": sample.current_timestamp_utc,
                "future_timestamp_utc": sample.future_timestamp_utc,
                "current_slice_count": int(current_count),
                "predicted_slice_count": int(predicted_count),
                "real_future_slice_count": int(len(sample.future_svi["business_days"])),
                "current_svi": sample.current_svi,
                "predicted_future_svi": predicted_svi,
                "real_future_svi": sample.future_svi,
                "selection_mode": config.selection_mode,
                **split_meta,
            },
        }

        output_stem = f"{sample.global_index:04d}_{safe_sample_filename(sample.sample_id)}"
        if config.save_json:
            save_payload_json(payload, sample_json_dir / f"{output_stem}.json")
        if config.save_plots:
            plot_surface_payload(payload, plot_dir / f"{output_stem}.png")

        summary_rows.append(
            {
                "sample_id": sample.sample_id,
                "mode": "svi",
                "global_index": int(sample.global_index),
                "news_row_id": int(sample.news_row_id),
                "news_timestamp_utc": sample.timestamp,
                "current_timestamp_utc": sample.current_timestamp_utc,
                "future_timestamp_utc": sample.future_timestamp_utc,
                "checkpoint_path": str(checkpoint_path),
                "predicted_slice_count": int(predicted_count),
                "real_future_slice_count": int(len(sample.future_svi["business_days"])),
                **split_meta,
                **metrics,
            }
        )

    write_summary_csv(summary_rows, run_dir / "summary.csv")
    return run_dir


if __name__ == "__main__":
    main()
