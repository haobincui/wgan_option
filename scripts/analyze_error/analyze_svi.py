"""Analyze SVI-model surface errors with bootstrap MSE statistics."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, Optional

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from scripts.analyze_error.bootstrap import bootstrap_from_error_rows  # noqa: E402
from scripts.analyze_error.common import (  # noqa: E402
    build_surface_grids,
    compute_error_metrics,
    prepare_run_output_dir,
    resolve_analysis_config,
    resolve_checkpoint_path,
    select_samples_from_split,
    split_metadata,
    write_bootstrap_outputs,
    write_resolved_config,
    write_summary_csv,
)
from scripts.analyze_error.plotting import (  # noqa: E402
    plot_bootstrap_mean_histogram,
    plot_mse_histogram,
)
from wgan_option.analysis_config import DEFAULT_SVI_ANALYSIS_CONFIG_PATH  # noqa: E402
from wgan_option.utils.inference_helpers import (  # noqa: E402
    build_inference_device,
    infer_future_svi,
    load_svi_regressor,
    reconstruct_svi_surface,
)
from wgan_option.utils.merged_xlsx import load_svi_paired_samples, select_ordered_split  # noqa: E402


def main(argv: Optional[Iterable[str]] = None) -> Path:
    config = resolve_analysis_config(
        argv=argv,
        description="Analyze target-vs-generated SVI-reconstructed surface errors and bootstrap mean MSE.",
        default_config_path=DEFAULT_SVI_ANALYSIS_CONFIG_PATH,
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

    device = build_inference_device(config.cuda)
    model, checkpoint = load_svi_regressor(checkpoint_path, device)
    embedding_dim = int(checkpoint["embedding_dim"])
    max_slices = int(checkpoint["max_slices"])
    normalization_stats = checkpoint["normalization_stats"]
    strike_grid, maturity_days_grid = build_surface_grids(config)
    split_meta = split_metadata(split_selection)
    error_rows = []

    for sample in selected_samples:
        predicted_svi, current_count, predicted_count = infer_future_svi(
            model,
            sample,
            embedding_dim=embedding_dim,
            max_slices=max_slices,
            normalization_stats=normalization_stats,
            device=device,
        )
        generated_surface = reconstruct_svi_surface(
            predicted_svi,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
        )
        target_surface = reconstruct_svi_surface(
            sample.future_svi,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
        )
        error_surface = target_surface - generated_surface
        metrics = compute_error_metrics(error_surface)
        error_rows.append(
            {
                "mode": "svi",
                "sample_id": sample.sample_id,
                "global_index": int(sample.global_index),
                "news_row_id": int(sample.news_row_id),
                "news_timestamp_utc": sample.timestamp,
                "current_timestamp_utc": sample.current_timestamp_utc,
                "future_timestamp_utc": sample.future_timestamp_utc,
                "checkpoint_path": str(checkpoint_path),
                "current_slice_count": int(current_count),
                "predicted_slice_count": int(predicted_count),
                "real_future_slice_count": int(len(sample.future_svi["business_days"])),
                **split_meta,
                **metrics,
            }
        )

    write_summary_csv(error_rows, run_dir / "errors.csv")
    bootstrap_summary, bootstrap_distribution = bootstrap_from_error_rows(
        error_rows,
        bootstrap_samples=int(config.bootstrap_samples),
        confidence_level=float(config.confidence_level),
        seed=int(config.bootstrap_seed),
    )
    write_bootstrap_outputs(
        summary=bootstrap_summary,
        distribution=bootstrap_distribution,
        run_dir=run_dir,
        save_distribution=bool(config.save_bootstrap_distribution),
    )
    if config.save_mse_histogram:
        plot_mse_histogram(
            error_rows,
            run_dir / "mse_histogram.png",
            bins=int(config.histogram_bins),
        )
    if config.save_bootstrap_histogram:
        plot_bootstrap_mean_histogram(
            bootstrap_distribution,
            bootstrap_summary,
            run_dir / "bootstrap_mean_mse_histogram.png",
            bins=int(config.histogram_bins),
        )
    return run_dir


if __name__ == "__main__":
    main()
