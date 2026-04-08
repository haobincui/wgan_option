"""Generate future vol surfaces from a trained WGAN checkpoint."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, Optional

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from scripts.generate_result.common import (  # noqa: E402
    compute_surface_metrics,
    prepare_run_output_dir,
    resolve_checkpoint_path,
    resolve_result_config,
    safe_sample_filename,
    select_samples_from_split,
    split_metadata,
    write_resolved_config,
    write_summary_csv,
    save_payload_json,
)
from scripts.generate_result.plot_surface import plot_surface_payload  # noqa: E402
from wgan_option.utils.inference_helpers import (  # noqa: E402
    build_inference_device,
    ensure_matching_embedding_dim,
    infer_vol_surface,
    load_vol_generator,
)
from wgan_option.utils.merged_xlsx import load_vol_surface_samples, select_ordered_split  # noqa: E402


def main(argv: Optional[Iterable[str]] = None) -> Path:
    config = resolve_result_config(
        argv=argv,
        description="Generate future vol surfaces from a trained WGAN checkpoint.",
        default_config_path="configs/generate_result/vol.yaml",
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
        generated_surface = infer_vol_surface(
            model,
            sample,
            noise_dim=int(train_config.noise_dim),
            seed=int(config.seed),
            device=device,
        )

        current_surface = sample.current_surface[0]
        real_surface = sample.target_surface[0]
        metrics = compute_surface_metrics(generated_surface, real_surface)

        payload = {
            "sample_id": sample.sample_id,
            "mode": "vol",
            "strike_grid": sample.strike_grid.astype(float).tolist(),
            "maturity_days_grid": sample.maturity_grid_days.astype(float).tolist(),
            "current_surface": current_surface.astype(float).tolist(),
            "generated_surface": generated_surface.astype(float).tolist(),
            "real_surface": real_surface.astype(float).tolist(),
            "metrics": metrics,
            "metadata": {
                "checkpoint_path": str(checkpoint_path),
                "global_index": int(sample.global_index),
                "news_timestamp_utc": sample.timestamp,
                "current_snapshot_time_utc": sample.current_snapshot_time_utc,
                "target_snapshot_time_utc": sample.target_snapshot_time_utc,
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
                "mode": "vol",
                "global_index": int(sample.global_index),
                "news_timestamp_utc": sample.timestamp,
                "current_snapshot_time_utc": sample.current_snapshot_time_utc,
                "target_snapshot_time_utc": sample.target_snapshot_time_utc,
                "checkpoint_path": str(checkpoint_path),
                **split_meta,
                **metrics,
            }
        )

    write_summary_csv(summary_rows, run_dir / "summary.csv")
    return run_dir


if __name__ == "__main__":
    main()
