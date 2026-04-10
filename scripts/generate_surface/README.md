# generate_surface

This directory contains the unified minute-level surface generation workflow.

The entrypoint is:

```bash
python scripts/generate_surface/main.py generate_surface ...
```

## Layout

- `main.py`
  - top-level CLI and `data_range` dispatch
- `dispatch.py`
  - shared `--device` parsing and CUDA checks
- `backend/`
  - thin CPU and GPU runtime entrypoints
- `data_helperd/`
  - shared minute data helpers
  - `all.py` for `data_range=all`
  - `window.py` for `data_range=window`
  - `excel.py` for `data_range=excel`
- `model/`
  - model-specific surface serializers for `svi`, `sabr`, `cubic`, and `raw`
- `common/`
  - shared config utilities

## CLI

`main.py` now supports one generation job:

- `generate_surface`

Required minute selectors:

- `--model {svi,sabr,cubic,raw}`
- `--data_range {all,window,excel}`

Common examples:

```bash
python scripts/generate_surface/main.py generate_surface \
  --device cpu \
  --model svi \
  --data_range all \
  --config configs/surface_builder/svi/generate_surface-svi-all.yaml

python scripts/generate_surface/main.py generate_surface \
  --device gpu \
  --model sabr \
  --data_range excel \
  --config configs/surface_builder/sabr/generate_surface-sabr-excel.yaml

python scripts/generate_surface/main.py generate_surface \
  --device cpu \
  --model raw \
  --data_range window \
  --config configs/surface_builder/raw/generate_surface-raw-window.yaml
```

`data_range` means:

- `all`
  - scan all eligible minutes in the input trade files
- `window`
  - only build surfaces within `+/- window_minutes` around target UTC timestamps
- `excel`
  - load target timestamps from the news workbook and then run the window workflow

## Config

Configs live under `configs/surface_builder/<model>/`.

Examples:

- `configs/surface_builder/svi/generate_surface-svi-all.yaml`
- `configs/surface_builder/sabr/generate_surface-sabr-window.yaml`
- `configs/surface_builder/cubic/generate_surface-cubic-excel.yaml`
- `configs/surface_builder/raw/generate_surface-raw-all.yaml`

All minute configs now use one section:

- `surface_builder.job = generate_surface`
- `surface_builder.generate_surface`

The shared `generate_surface` section contains:

- common runtime fields such as `model`, `data_range`, `run_ts`, `input_glob`, `output_dir`, `output_json`, `log_file`, `precalib_csv`
- quality-filter and chunking fields such as `days_in_year`, `min_strikes_per_expiry`, `min_expiries_per_minute`, `max_precalib_iv`, `max_files`, `max_minutes`, `chunk_size`, `calibration_workers`
- `window` fields such as `target_datetimes`, `target_datetimes_file`, `window_minutes`
- `excel` fields such as `target_xlsx`, `sheet_name`, `date_column`, `time_column`, `source_timezone`, `max_target_datetimes`, `window_minutes`

`calibration_workers` is currently used only by CPU `svi` jobs with `data_range=window` or `data_range=excel`.
Set it to `0` to keep serial behavior, or a positive value such as `32` to parallelize minute SVI calibration across CPU processes.
GPU jobs still run serially for this stage.

## Models

Minute generation currently supports four models:

- `svi`
  - fit SVI parameters slice by slice
- `sabr`
  - fit SABR parameters slice by slice
- `cubic`
  - store filtered discrete IV slices and reconstruct with cubic-spline interpolation
- `raw`
  - store filtered discrete IV slices with no calibration
  - downstream reconstruction uses linear interpolation in strike and total variance across maturities

## Output Directory

Each run writes to:

```text
data/processed/<model>-<data_range>/<run_ts>/
```

Example:

```text
data/processed/svi-all/20260330-01/
```

## Output Files

Main run artifacts use the new `surface-*` naming convention:

- `surface-<model>-<data_range>.json`
  - primary model-aware payload
  - `all`: `minute_ts -> {"surface_model", "surface_params"}`
  - `window` / `excel`: `target_ts -> {"backward": {...}, "forward": {...}}`
- `surface-<model>-<data_range>.log`
  - runtime log for the generation job
- `surface-<model>-<data_range>-precalib-points.csv`
  - one row per option observation that reached the minute implied-vol stage
  - includes timestamp, strike, spot, percent strike, implied vol, filter pass/fail, reason, and weight
- `surface-resolved_config.yaml`
  - resolved config snapshot after variable expansion and CLI overrides
  - records `model`, `data_range`, `run_ts`, and expanded output paths

`surface_params` depends on `surface_model`:

- `svi`
  - `business_days`, `a`, `b`, `rho`, `m`, `sigma`
- `sabr`
  - `business_days`, `alpha`, `beta`, `rho`, `nu`
- `cubic`
  - `business_days`, `percent_strikes`, `implied_vols`
- `raw`
  - `business_days`, `percent_strikes`, `implied_vols`

These outputs are the direct inputs to `scripts/merge_file`.

## Merge Outputs

After generation, the merge scripts write workbook outputs into the same run directory:

- `merged_svi.xlsx`
  - direction-level surface parameter audit workbook
- `merged_vol.xlsx`
  - paired current/target surface workbook for GAN training
- `merged_params.xlsx`
  - model-neutral parameter audit workbook

## Direction Semantics

For `data_range=window` and `data_range=excel`:

- `backward`
  - aligned with the original news timestamp
- `forward`
  - aligned with the offset future timestamp window

Those semantics are preserved later by `merge_svi.py` and `merge_vol.py`.
