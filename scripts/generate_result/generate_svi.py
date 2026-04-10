"""Generate future SVI params, reconstruct surfaces, and compare them."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, Optional

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from wgan_option.generate_result_runtime import (  # noqa: E402
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
from wgan_option.utils.inference_helpers import (  # noqa: E402
    build_inference_device,
    infer_future_svi,
    load_svi_regressor,
    reconstruct_svi_surface,
)
from wgan_option.utils.merged_xlsx import load_svi_paired_samples, select_ordered_split  # noqa: E402


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
            model,
            sample,
            embedding_dim=embedding_dim,
            max_slices=max_slices,
            normalization_stats=normalization_stats,
            device=device,
        )

        current_surface = reconstruct_svi_surface(
            sample.current_svi,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
        )
        generated_surface = reconstruct_svi_surface(
            predicted_svi,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
        )
        real_surface = reconstruct_svi_surface(
            sample.future_svi,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
        )
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
