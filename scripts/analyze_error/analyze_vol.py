"""Analyze generated-vs-target vol surface errors with bootstrap MSE statistics."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, Optional

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from wgan_option.analysis_runtime import (  # noqa: E402
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
from scripts.analyze_error.bootstrap import bootstrap_from_error_rows  # noqa: E402
from scripts.analyze_error.plotting import (  # noqa: E402
    plot_bootstrap_mean_histogram,
    plot_mse_histogram,
)
from wgan_option.analysis_config import DEFAULT_VOL_ANALYSIS_CONFIG_PATH  # noqa: E402
from wgan_option.utils.inference_helpers import (  # noqa: E402
    build_inference_device,
    ensure_matching_embedding_dim,
    infer_vol_surface,
    load_vol_generator,
)
from wgan_option.utils.merged_xlsx import load_vol_surface_samples, select_ordered_split  # noqa: E402


def main(argv: Optional[Iterable[str]] = None) -> Path:
    config = resolve_analysis_config(
        argv=argv,
        description="Analyze target-vs-generated vol surface errors and bootstrap mean MSE.",
        default_config_path=DEFAULT_VOL_ANALYSIS_CONFIG_PATH,
    )

    all_samples = load_vol_surface_samples(config)
    split_selection = select_ordered_split(all_samples, train_ratio=config.train_ratio, split=config.split)
    selected_samples = select_samples_from_split(split_selection, config)

    run_dir = prepare_run_output_dir(config.output_dir)
    write_resolved_config(config, run_dir)
    checkpoint_path = resolve_checkpoint_path(
        config,
        artifact_key="generator",
        fallback_filenames=("generator_best.pt", "generator.pt"),
    )

    device = build_inference_device(config.cuda)
    model, train_config, embedding_dim = load_vol_generator(checkpoint_path, selected_samples[0], device)
    split_meta = split_metadata(split_selection)
    error_rows = []

    for sample in selected_samples:
        ensure_matching_embedding_dim(
            sample_id=str(sample.sample_id),
            embedding=sample.text_embedding,
            embedding_dim=embedding_dim,
        )
        generated_surface = infer_vol_surface(
            model,
            sample,
            noise_dim=int(train_config.noise_dim),
            seed=int(config.seed),
            device=device,
        )
        target_surface = sample.target_surface[0]
        error_surface = target_surface - generated_surface
        metrics = compute_error_metrics(error_surface)
        error_rows.append(
            {
                "mode": "vol",
                "sample_id": sample.sample_id,
                "global_index": int(sample.global_index),
                "news_timestamp_utc": sample.timestamp,
                "current_snapshot_time_utc": sample.current_snapshot_time_utc,
                "target_snapshot_time_utc": sample.target_snapshot_time_utc,
                "checkpoint_path": str(checkpoint_path),
                "pair_quality_label": str(sample.metadata.get("pair_quality_label", "")),
                "current_weighted_iv_rmse": (
                    None
                    if sample.metadata.get("current_weighted_iv_rmse") is None
                    else float(sample.metadata["current_weighted_iv_rmse"])
                ),
                "target_weighted_iv_rmse": (
                    None
                    if sample.metadata.get("target_weighted_iv_rmse") is None
                    else float(sample.metadata["target_weighted_iv_rmse"])
                ),
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
