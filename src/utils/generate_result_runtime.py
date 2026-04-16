"""Result-generation CLI helpers that used to live under ``scripts/``."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.postprocess_runtime import (
    VALID_SELECTION_MODES,
    VALID_SPLITS,
    add_shared_sample_selection_args,
    build_surface_grids,
    compute_surface_metrics,
    resolve_checkpoint_path,
    safe_sample_filename,
    save_payload_json,
    select_samples_from_split,
    split_metadata,
    write_json,
    write_summary_csv,
)
from utils.result_config import (
    GenerateResultConfig,
    save_generate_result_config_yaml,
)
from utils.training_paths import generate_result_config_path

VALID_FALLBACK_MODES = {"none", "mc_uncertainty_to_current"}


def build_result_arg_parser(*, description: str, default_config_path: str) -> argparse.ArgumentParser:
    """Create the standard result-generation parser."""

    parser = argparse.ArgumentParser(description=description)
    add_shared_sample_selection_args(parser, default_config_path=default_config_path)
    parser.add_argument("--no-plot", action="store_true", help="Disable PNG plot generation.")
    parser.add_argument("--no-json", action="store_true", help="Disable per-sample JSON payload output.")
    return parser


def validate_result_config(config: GenerateResultConfig) -> None:
    """Fail fast on invalid high-level generation config values."""

    if config.split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got: {config.split}")
    if config.selection_mode not in VALID_SELECTION_MODES:
        raise ValueError(
            f"selection_mode must be one of {sorted(VALID_SELECTION_MODES)}, got: {config.selection_mode}"
        )
    if str(config.plot_style).strip().lower() != "heatmap_diff":
        raise ValueError(f"Only plot_style=heatmap_diff is currently supported, got: {config.plot_style}")
    if config.selection_mode == "sample_id" and not str(config.sample_id).strip():
        raise ValueError("sample_id must be provided when selection_mode=sample_id.")
    if config.selection_mode == "row_index" and int(config.row_index) < 0:
        raise ValueError("row_index must be >= 0 when selection_mode=row_index.")
    if config.selection_mode == "first_n" and int(config.limit) <= 0:
        raise ValueError("limit must be > 0 when selection_mode=first_n.")
    if config.fallback_mode not in VALID_FALLBACK_MODES:
        raise ValueError(
            f"fallback_mode must be one of {sorted(VALID_FALLBACK_MODES)}, got: {config.fallback_mode}"
        )
    if float(config.timeseries_atm_range) < 0.0:
        raise ValueError("timeseries_atm_range must be >= 0.")
    if float(config.timeseries_short_end_max_days) < 0.0:
        raise ValueError("timeseries_short_end_max_days must be >= 0.")
    if int(config.mc_samples) <= 0:
        raise ValueError("mc_samples must be > 0.")
    if config.fallback_mode != "none" and int(config.mc_samples) < 2:
        raise ValueError("mc_samples must be >= 2 when fallback_mode is enabled.")


def write_resolved_config(config: GenerateResultConfig, run_dir: str | Path) -> Path:
    """Persist the resolved runtime config into the run directory."""

    return save_generate_result_config_yaml(config, generate_result_config_path(run_dir))


# ---------------------------------------------------------------------------
# High-level generation runners used by trainer-led generate_result flows.
# ---------------------------------------------------------------------------


def generate_vol_result(config: GenerateResultConfig) -> Path:
    """Generate future vol surfaces from one resolved runtime config."""

    import json as _json
    from typing import Optional as _Opt

    from utils.postprocess_runtime import resolve_metrics_artifact_path
    from wgan_option.utils.inference_helpers import (
        build_inference_device,
        ensure_matching_embedding_dim,
        infer_vol_surface,
        infer_vol_surface_mc,
        load_vol_generator,
    )
    from wgan_option.utils.merged_xlsx import load_vol_surface_samples, select_ordered_split
    from wgan_option.visualization.surface_plot import (
        extract_short_end_atm_band_value,
        plot_short_end_atm_band_timeseries,
        plot_surface_payload,
    )

    all_samples = load_vol_surface_samples(config)
    split_selection = select_ordered_split(all_samples, train_ratio=config.train_ratio, split=config.split)
    selected_samples = select_samples_from_split(split_selection, config)

    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, run_dir)
    checkpoint_path = resolve_checkpoint_path(
        config,
        artifact_key="generator",
        fallback_filenames=("generator_best.pt", "generator.pt"),
    )

    device = build_inference_device(config.cuda)
    model, train_config, embedding_dim = load_vol_generator(checkpoint_path, selected_samples[0], device)

    fallback_threshold: _Opt[float] = None
    fallback_calibration_path: _Opt[Path] = None
    effective_mc_samples = int(config.mc_samples)
    if str(config.fallback_mode).strip().lower() == "mc_uncertainty_to_current":
        if float(config.uncertainty_threshold) >= 0.0:
            fallback_threshold = float(config.uncertainty_threshold)
            effective_mc_samples = int(config.mc_samples)
        else:
            calibration_path = resolve_metrics_artifact_path(
                config, filename="fallback_calibration.json", checkpoint_path=checkpoint_path,
            )
            with calibration_path.open("r", encoding="utf-8") as handle:
                cal_payload = _json.load(handle)
            fallback_threshold = float(cal_payload["uncertainty_threshold"])
            effective_mc_samples = int(cal_payload.get("mc_samples", config.mc_samples))
            fallback_calibration_path = calibration_path

    sample_json_dir = run_dir / "samples"
    plot_dir = run_dir / "plots"
    summary_rows = []
    split_meta = split_metadata(split_selection)

    for sample in selected_samples:
        ensure_matching_embedding_dim(
            sample_id=str(sample.sample_id),
            embedding=sample.text_embedding,
            embedding_dim=embedding_dim,
        )

        current_surface = sample.current_surface[0]
        real_surface = sample.target_surface[0]
        current_metrics = compute_surface_metrics(current_surface, real_surface)
        uncertainty_score = 0.0
        used_fallback = False
        prediction_source = "model_mean"

        if str(config.fallback_mode).strip().lower() == "mc_uncertainty_to_current":
            model_mean_surface, uncertainty_score, _ = infer_vol_surface_mc(
                model, sample,
                noise_dim=int(train_config.noise_dim),
                seed=int(config.seed), device=device,
                mc_samples=effective_mc_samples,
            )
            used_fallback = bool(
                fallback_threshold is not None and float(uncertainty_score) > float(fallback_threshold)
            )
            generated_surface = current_surface.copy() if used_fallback else model_mean_surface
            prediction_source = "current_fallback" if used_fallback else "model_mean"
        else:
            generated_surface = infer_vol_surface(
                model, sample,
                noise_dim=int(train_config.noise_dim),
                seed=int(config.seed), device=device,
            )

        metrics = compute_surface_metrics(generated_surface, real_surface)
        current_band = extract_short_end_atm_band_value(
            current_surface,
            strike_grid=sample.strike_grid,
            maturity_days_grid=sample.maturity_grid_days,
            atm_range=float(config.timeseries_atm_range),
            short_end_max_days=float(config.timeseries_short_end_max_days),
        )
        generated_band = extract_short_end_atm_band_value(
            generated_surface,
            strike_grid=sample.strike_grid,
            maturity_days_grid=sample.maturity_grid_days,
            atm_range=float(config.timeseries_atm_range),
            short_end_max_days=float(config.timeseries_short_end_max_days),
        )
        real_band = extract_short_end_atm_band_value(
            real_surface,
            strike_grid=sample.strike_grid,
            maturity_days_grid=sample.maturity_grid_days,
            atm_range=float(config.timeseries_atm_range),
            short_end_max_days=float(config.timeseries_short_end_max_days),
        )

        payload = {
            "sample_id": sample.sample_id,
            "mode": "vol",
            "strike_grid": sample.strike_grid.astype(float).tolist(),
            "maturity_days_grid": sample.maturity_grid_days.astype(float).tolist(),
            "current_surface": current_surface.astype(float).tolist(),
            "generated_surface": generated_surface.astype(float).tolist(),
            "real_surface": real_surface.astype(float).tolist(),
            "metrics": metrics,
            "current_metrics": current_metrics,
            "metadata": {
                "checkpoint_path": str(checkpoint_path),
                "global_index": int(sample.global_index),
                "news_timestamp_utc": sample.timestamp,
                "current_snapshot_time_utc": sample.current_snapshot_time_utc,
                "target_snapshot_time_utc": sample.target_snapshot_time_utc,
                "pair_quality_label": str(sample.metadata.get("pair_quality_label", "")),
                "current_weighted_iv_rmse": (
                    None if sample.metadata.get("current_weighted_iv_rmse") is None
                    else float(sample.metadata["current_weighted_iv_rmse"])
                ),
                "target_weighted_iv_rmse": (
                    None if sample.metadata.get("target_weighted_iv_rmse") is None
                    else float(sample.metadata["target_weighted_iv_rmse"])
                ),
                "selection_mode": config.selection_mode,
                "fallback_mode": str(config.fallback_mode),
                "mc_samples": int(effective_mc_samples),
                "uncertainty_score": float(uncertainty_score),
                "fallback_threshold": None if fallback_threshold is None else float(fallback_threshold),
                "used_fallback": bool(used_fallback),
                "prediction_source": prediction_source,
                "fallback_calibration_path": (
                    None if fallback_calibration_path is None else str(fallback_calibration_path)
                ),
                **split_meta,
            },
        }

        output_stem = f"{sample.global_index:04d}_{safe_sample_filename(sample.sample_id)}"
        if config.save_json:
            save_payload_json(payload, sample_json_dir / f"{output_stem}.json")
        if config.save_plots:
            plot_surface_payload(payload, plot_dir / f"{output_stem}.png")

        summary_rows.append({
            "sample_id": sample.sample_id, "mode": "vol",
            "global_index": int(sample.global_index),
            "news_timestamp_utc": sample.timestamp,
            "current_snapshot_time_utc": sample.current_snapshot_time_utc,
            "target_snapshot_time_utc": sample.target_snapshot_time_utc,
            "checkpoint_path": str(checkpoint_path),
            "current_mae": float(current_metrics["mae"]),
            "current_rmse": float(current_metrics["rmse"]),
            "current_max_abs": float(current_metrics["max_abs"]),
            "fallback_mode": str(config.fallback_mode),
            "mc_samples": int(effective_mc_samples),
            "uncertainty_score": float(uncertainty_score),
            "fallback_threshold": "" if fallback_threshold is None else float(fallback_threshold),
            "used_fallback": bool(used_fallback),
            "prediction_source": prediction_source,
            "short_atm_band_current_vol": float(current_band["value"]),
            "short_atm_band_generated_future_vol": float(generated_band["value"]),
            "short_atm_band_real_future_vol": float(real_band["value"]),
            "short_atm_band_generated_abs_error": float(abs(generated_band["value"] - real_band["value"])),
            "short_atm_band_current_abs_error": float(abs(current_band["value"] - real_band["value"])),
            "short_atm_band_point_count": int(current_band["point_count"]),
            "short_atm_band_selection": str(current_band["selection"]),
            "short_atm_band_atm_range": float(config.timeseries_atm_range),
            "short_atm_band_max_days": float(config.timeseries_short_end_max_days),
            **split_meta, **metrics,
        })

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (str(row.get("news_timestamp_utc", "")), int(row.get("global_index", -1))),
    )
    write_summary_csv(summary_rows, run_dir / "summary.csv")
    if config.save_plots and len(summary_rows) >= 2:
        plot_short_end_atm_band_timeseries(
            summary_rows,
            plot_dir / "short_atm_band_timeseries.png",
            atm_range=float(config.timeseries_atm_range),
            short_end_max_days=float(config.timeseries_short_end_max_days),
        )
    return run_dir

def generate_svi_result(config: GenerateResultConfig) -> Path:
    """Generate future SVI params, reconstruct surfaces, and compare them."""

    from wgan_option.utils.inference_helpers import (
        build_inference_device,
        infer_future_svi,
        load_svi_regressor,
        reconstruct_svi_surface,
    )
    from wgan_option.utils.merged_xlsx import load_svi_paired_samples, select_ordered_split
    from wgan_option.visualization.surface_plot import plot_surface_payload

    all_samples = load_svi_paired_samples(config)
    split_selection = select_ordered_split(all_samples, train_ratio=config.train_ratio, split=config.split)
    selected_samples = select_samples_from_split(split_selection, config)

    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, run_dir)
    checkpoint_path = resolve_checkpoint_path(
        config, artifact_key="model",
        fallback_filenames=("svi_regressor_best.pt", "svi_regressor.pt"),
    )

    device = build_inference_device(config.cuda)
    model, checkpoint = load_svi_regressor(checkpoint_path, device)
    embedding_dim = int(checkpoint["embedding_dim"])
    max_slices = int(checkpoint["max_slices"])
    normalization_stats = checkpoint["normalization_stats"]

    strike_grid, maturity_days_grid = build_surface_grids(config)
    sample_json_dir = run_dir / "samples"
    plot_dir = run_dir / "plots"
    summary_rows = []
    split_meta = split_metadata(split_selection)

    for sample in selected_samples:
        predicted_svi, current_count, predicted_count = infer_future_svi(
            model, sample,
            embedding_dim=embedding_dim, max_slices=max_slices,
            normalization_stats=normalization_stats, device=device,
        )

        current_surface = reconstruct_svi_surface(sample.current_svi, strike_grid=strike_grid, maturity_days_grid=maturity_days_grid)
        generated_surface = reconstruct_svi_surface(predicted_svi, strike_grid=strike_grid, maturity_days_grid=maturity_days_grid)
        real_surface = reconstruct_svi_surface(sample.future_svi, strike_grid=strike_grid, maturity_days_grid=maturity_days_grid)
        metrics = compute_surface_metrics(generated_surface, real_surface)

        payload = {
            "sample_id": sample.sample_id, "mode": "svi",
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

        summary_rows.append({
            "sample_id": sample.sample_id, "mode": "svi",
            "global_index": int(sample.global_index),
            "news_row_id": int(sample.news_row_id),
            "news_timestamp_utc": sample.timestamp,
            "current_timestamp_utc": sample.current_timestamp_utc,
            "future_timestamp_utc": sample.future_timestamp_utc,
            "checkpoint_path": str(checkpoint_path),
            "predicted_slice_count": int(predicted_count),
            "real_future_slice_count": int(len(sample.future_svi["business_days"])),
            **split_meta, **metrics,
        })

    write_summary_csv(summary_rows, run_dir / "summary.csv")
    return run_dir

def generate_vol_regression_result(config: GenerateResultConfig) -> Path:
    """Generate future vol surfaces from a deterministic regression checkpoint."""

    from wgan_option.utils.inference_helpers import (
        build_inference_device,
        ensure_matching_embedding_dim,
        infer_vol_regression_surface,
        load_vol_regressor,
    )
    from wgan_option.utils.merged_xlsx import load_vol_surface_samples, select_ordered_split
    from wgan_option.visualization.surface_plot import plot_surface_payload

    all_samples = load_vol_surface_samples(config)
    split_selection = select_ordered_split(all_samples, train_ratio=config.train_ratio, split=config.split)
    selected_samples = select_samples_from_split(split_selection, config)

    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, run_dir)
    checkpoint_path = resolve_checkpoint_path(
        config, artifact_key="model",
        fallback_filenames=("vol_regressor_best.pt", "vol_regressor.pt"),
    )

    device = build_inference_device(config.cuda)
    model, _, embedding_dim = load_vol_regressor(checkpoint_path, selected_samples[0], device)

    sample_json_dir = run_dir / "samples"
    plot_dir = run_dir / "plots"
    summary_rows = []
    split_meta = split_metadata(split_selection)

    for sample in selected_samples:
        ensure_matching_embedding_dim(
            sample_id=str(sample.sample_id),
            embedding=sample.text_embedding,
            embedding_dim=embedding_dim,
        )
        current_surface = sample.current_surface[0]
        generated_surface = infer_vol_regression_surface(model, sample, device=device)
        real_surface = sample.target_surface[0]
        metrics = compute_surface_metrics(generated_surface, real_surface)
        current_metrics = compute_surface_metrics(current_surface, real_surface)

        payload = {
            "sample_id": sample.sample_id, "mode": "vol-regression",
            "strike_grid": sample.strike_grid.astype(float).tolist(),
            "maturity_days_grid": sample.maturity_grid_days.astype(float).tolist(),
            "current_surface": current_surface.astype(float).tolist(),
            "generated_surface": generated_surface.astype(float).tolist(),
            "real_surface": real_surface.astype(float).tolist(),
            "metrics": metrics, "current_metrics": current_metrics,
            "metadata": {
                "checkpoint_path": str(checkpoint_path),
                "global_index": int(sample.global_index),
                "news_timestamp_utc": sample.timestamp,
                "current_snapshot_time_utc": sample.current_snapshot_time_utc,
                "target_snapshot_time_utc": sample.target_snapshot_time_utc,
                "pair_quality_label": str(sample.metadata.get("pair_quality_label", "")),
                "selection_mode": config.selection_mode,
                **split_meta,
            },
        }

        output_stem = f"{sample.global_index:04d}_{safe_sample_filename(sample.sample_id)}"
        if config.save_json:
            save_payload_json(payload, sample_json_dir / f"{output_stem}.json")
        if config.save_plots:
            plot_surface_payload(payload, plot_dir / f"{output_stem}.png")

        summary_rows.append({
            "sample_id": sample.sample_id, "mode": "vol-regression",
            "global_index": int(sample.global_index),
            "news_timestamp_utc": sample.timestamp,
            "current_snapshot_time_utc": sample.current_snapshot_time_utc,
            "target_snapshot_time_utc": sample.target_snapshot_time_utc,
            "checkpoint_path": str(checkpoint_path),
            "current_mae": float(current_metrics["mae"]),
            "current_rmse": float(current_metrics["rmse"]),
            "current_max_abs": float(current_metrics["max_abs"]),
            **split_meta, **metrics,
        })

    write_summary_csv(summary_rows, run_dir / "summary.csv")
    return run_dir
