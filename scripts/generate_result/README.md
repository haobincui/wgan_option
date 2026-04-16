# generate_result

This directory contains the post-training inference workflow for generating future model outputs from saved checkpoints and comparing them against real targets.

It supports three subcommands:

- `vol`: generate future vol surfaces from a trained WGAN checkpoint
- `svi`: generate future SVI parameters from a trained SVI regressor, reconstruct them into vol surfaces, and compare them against real future SVI-reconstructed surfaces
- `plot`: render a previously saved payload JSON into PNG comparison plots

## Entry Point

Unified CLI:

```bash
python scripts/generate_result/main.py vol --config configs/wgan/train_vol_xlsx.yaml
python scripts/generate_result/main.py svi --config configs/wgan/train_svi_xlsx.yaml
python scripts/generate_result/main.py plot --input-json path/to/sample.json
```

## Shared Runtime Logic

Shared CLI and runtime helpers live in `src/generate_result/cli.py` and `src/utils/generate_result_runtime.py`.

Common responsibilities:

- parse merged training YAML plus `--set` overrides
- validate split and sample-selection choices
- resolve the effective checkpoint
- derive the final output directory under the training run
- save `generate_resolved_config.yaml`
- write `summary.csv`
- write per-sample payload JSON files

For `vol` mode, the resolved config also controls one run-level short-end ATM time-series view through:

- `timeseries_atm_range` (default `0.08`)
- `timeseries_short_end_max_days` (default `10`)

Supported split selection:

- `train`
- `val`
- `all`

Supported sample selection:

- `all`
- `first_n`
- `row_index`
- `sample_id`

## vol Mode

Implementation:

- `scripts/generate_result/generate_vol.py`

Data source:

- `merged_vol.xlsx`

Flow:

1. Load ordered paired vol-surface samples from the workbook.
2. Apply chronological train/validation splitting.
3. Select samples according to the configured split and selection mode.
4. Resolve the generator checkpoint.
5. Rebuild the generator from the saved training config.
6. Re-run inference for each selected sample.
7. Compare:
   - `generated_surface`
   - `real_surface` from the workbook
8. Save JSON payloads, per-sample plots, a run-level short-end ATM time-series plot when multiple samples are selected, and a run summary.

The per-sample payload includes:

- `current_surface`
- `generated_surface`
- `real_surface`
- surface grids
- simple comparison metrics
- metadata such as timestamps and checkpoint path

The vol `summary.csv` also includes one short-end near-ATM scalar per selected sample:

- `short_atm_band_current_vol`
- `short_atm_band_generated_future_vol`
- `short_atm_band_real_future_vol`
- absolute-error columns against the real future band
- the band definition metadata (`short_atm_band_atm_range`, `short_atm_band_max_days`)

The band is defined on the surface grid as:

- `|strike - 1.0| <= timeseries_atm_range`
- `maturity_days <= timeseries_short_end_max_days`

If no grid cells satisfy that mask, the runtime falls back to the nearest ATM column at the shortest maturity row.

## svi Mode

Implementation:

- `scripts/generate_result/generate_svi.py`

Data source:

- `merged_svi.xlsx`

Flow:

1. Load usable SVI rows.
2. Pair `backward/current` and `forward/future` rows by `news_row_id`.
3. Apply chronological splitting and sample selection.
4. Resolve the SVI regressor checkpoint.
5. Rebuild the regressor and recover normalization statistics.
6. Predict future SVI parameters and future slice count.
7. Reconstruct:
   - current surface
   - generated future surface
   - real future surface
8. Compare generated and real reconstructed surfaces.
9. Save JSON payloads, plots, and a run summary.

The payload also stores:

- `current_svi`
- `predicted_future_svi`
- `real_future_svi`
- current/predicted/real slice counts

## Metrics

The shared metric helper computes:

- `mae`
- `rmse`
- `max_abs`

These are computed from:

```text
generated_surface - real_surface
```

The goal of `generate_result` is qualitative and sample-level inspection rather than bootstrap-style statistical analysis.

## Plot Rendering

Plotting logic lives in `scripts/generate_result/plot_surface.py`.

For a standard payload JSON, it generates:

- a main heatmap comparison figure
- a `_lines` sidecar figure
- an `_atm` sidecar figure

For `vol` mode runs with `save_plots=true` and at least two selected samples, the runtime also generates:

- `plots/short_atm_band_timeseries.png`

The main figure includes:

- current surface, if present
- generated surface
- real surface, if present
- generated-minus-real difference heatmap, if real surface is present

The sidecar line plots focus on:

- an ATM term structure view
- a short-maturity smile view

The run-level time-series plot uses:

- x-axis = `news_timestamp_utc`
- lines = `Current`, `Generated Future`, `Real Future`
- title text that records the active short-end ATM band definition

## Output Layout

Each run writes into a checkpoint-named directory under the training run's `generate_result/` root.

Typical contents:

- `generate_resolved_config.yaml`
- `summary.csv`
- `samples/*.json`
- `plots/*.png`

For each selected sample, the plot output usually includes:

- the main comparison PNG
- a `_lines.png` sidecar
- an `_atm.png` sidecar

Additionally, vol runs that select multiple samples usually include:

- `plots/short_atm_band_timeseries.png`

## Typical Commands

Generate several validation-set vol examples:

```bash
python scripts/generate_result/main.py vol \
  --config configs/wgan/train_vol_xlsx.yaml \
  --set split=val \
  --set selection_mode=first_n \
  --set limit=5
```

Generate one SVI sample by row index:

```bash
python scripts/generate_result/main.py svi \
  --config configs/wgan/train_svi_xlsx.yaml \
  --set selection_mode=row_index \
  --set row_index=0
```

Render a saved payload JSON into a plot:

```bash
python scripts/generate_result/main.py plot \
  --input-json outputs/training/<model>/<data_range>/<run_ts>/generate_result/<checkpoint_name>/samples/sample.json
```
