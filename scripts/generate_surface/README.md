# generate_surface

This directory contains the raw surface-generation workflows.

It covers two distinct tasks:

- daily-surface generation from raw option trade files
- minute-level surface generation from raw option trade files

The unified entrypoint is:

```bash
python scripts/generate_surface/main.py ...
```

## Entry Point and Job Dispatch

`scripts/generate_surface/main.py` is the top-level CLI.

Supported jobs:

- `daily-surface`
- `minute-svi`
- `minute-svi-window`
- `minute-svi-excel`

You can either:

- provide the subcommand explicitly, or
- omit the subcommand and let the script read `surface_builder.job` from the YAML config

Examples:

```bash
python scripts/generate_surface/main.py daily-surface --config configs/surface_builder/default.yaml
python scripts/generate_surface/main.py minute-svi --device cpu --model svi --config configs/surface_builder/svi/minute-svi-all.yaml
python scripts/generate_surface/main.py minute-svi-excel --device gpu --model sabr --config configs/surface_builder/sabr/minute-sabr-excel.yaml
python scripts/generate_surface/main.py minute-svi --device cpu --model raw --config configs/surface_builder/raw/minute-raw-all.yaml
```

Device dispatch is handled by `scripts/generate_surface/dispatch.py`:

- minute-SVI jobs support `--device cpu` and `--device gpu`
- minute jobs support `--model {svi,sabr,cubic,raw}`
- GPU mode requires CUDA
- daily-surface ignores the device choice and always runs its own path

Internal layout now follows this split:

- `scripts/generate_surface/model/`
  - model-specific SVI / SABR / cubic / raw surface parameter construction
- `scripts/generate_surface/data_helperd/`
  - all-minute, datetime-window, and Excel-driven data-flow helpers
- `scripts/generate_surface/common/`
  - shared config and reusable support utilities

## daily-surface

Implementation:

- `scripts/generate_surface/daily_surface.py`

Purpose:

- build daily volatility-like surface tensors from raw option trade files

Main flow:

1. load the daily-surface config section
2. resolve config variables from the YAML file
3. build `OptionSurfaceBuilderConfig`
4. run `OptionSurfaceBuilder`
5. save the output stack to disk

Typical outputs:

- a compressed NPZ stack
- optional per-day CSV snapshots

This is the historical daily-surface path used by the older training workflow.

It is separate from minute `--model raw`:

- `daily-surface` writes raw daily grids / NPZ snapshots
- minute `raw` writes model-aware implied-vol JSON payloads for downstream merge workflows

## minute-svi

Main CPU entrypoint:

- `scripts/generate_surface/backend/surface_cpu/all.py`

Related GPU entrypoint:

- `scripts/generate_surface/backend/surface_gpu/all.py`

Purpose:

- build minute-level surface parameters from raw option trade files for all eligible minutes

Core shared logic:

- `scripts/generate_surface/data_helperd/all.py`

### High-Level Flow

1. load the minute-SVI config
2. scan raw trade files with chunked CSV reading
3. normalize timestamps to UTC minutes
4. separate futures rows from option rows
5. infer the minute spot from the relevant future contract
6. build option candidates for that minute
7. compute implied vols
8. apply a pre-calibration filter
9. fit or serialize the configured surface model if the minute has enough valid expiries and strike points
10. write JSON and optional pre-calibration CSV outputs

### Main Outputs

The standard minute run writes:

- `minute_svi_params.json`
- `minute_svi_precalib_points.csv` if enabled
- `minute_svi_params.log`
- `resolved_config.yaml`

By default those files now live under:

- `data/processed/<model>/<run_ts>/`

The JSON filename remains legacy, but the payload is model-aware through:

- `surface_model`
- `surface_params`

Supported minute models:

- `svi`
- `sabr`
- `cubic`
- `raw`

`raw` means the minute job stops after implied-vol filtering and expiry/strike aggregation, then stores the resulting discrete implied-vol slices as a non-parametric surface. It does not run SVI/SABR/cubic calibration.

These files become the direct inputs for `scripts/merge_file`.

When `surface_model = raw`, downstream `merge_vol.py` reconstructs the regular training grid by:

- linearly interpolating along strike inside each expiry slice
- linearly interpolating total variance across business-day terms
- clamping only outside the observed strike/time range

So if a raw surface does not contain the exact grid node needed by `merged_vol.xlsx`, that node is filled by linear interpolation rather than left missing.

### CPU vs GPU

The CPU and GPU backends share the same overall pipeline.

The main difference is the implied-vol calculation engine:

- CPU path uses the CPU-side Black-Scholes implied vol routine
- GPU path uses the GPU implementation while keeping the same surrounding workflow

### Output Files

Minute jobs write these files under `data/processed/<model>/<run_ts>/` by default:

- `minute_svi_params.json`
  - The primary model-aware result payload.
  - `minute-svi`: `minute_ts -> {"surface_model", "surface_params"}`
  - `minute-svi-window` / `minute-svi-excel`: `target_ts -> {"backward": {...}, "forward": {...}}`
  - For `svi`, `surface_params` stores fitted SVI slice parameters.
  - For `sabr`, `surface_params` stores fitted SABR slice parameters.
  - For `cubic`, `surface_params` stores the discrete implied-vol slices used by the spline surface.
  - For `raw`, `surface_params` stores the filtered discrete implied-vol slices used by the linear-interpolation surface.

- `minute_svi_precalib_points.csv`
  - One row per raw option point that reached the minute implied-vol stage.
  - Includes raw trade timestamp, minute calibration timestamp, strike, spot, percent strike, implied vol, filter pass/fail flag, filter reason, and weight.
  - This is the main audit file for checking which raw observations entered or failed the minute surface build.

- `minute_svi_params.log`
  - Runtime log for that generation job.
  - Records CLI arguments, progress, skip counters, and surface-fit failures.

- `resolved_config.yaml`
  - Snapshot of the effective config used for the run after CLI overrides and variable expansion.
  - Includes the resolved `model`, `run_ts`, output paths, and any window / Excel target settings.

Daily `daily-surface` writes a different output family:

- `vol_surface_stack.npz` by default
  - Historical daily raw-grid stack used by the older workflow.

- optional per-day CSV snapshots
  - Enabled by `--save-daily-csv`.
  - These are daily raw-grid exports, not minute model-aware surface payloads.

## minute-svi-window

CPU entrypoint:

- `scripts/generate_surface/backend/surface_cpu/window.py`

Purpose:

- generate minute-SVI surfaces only inside windows around explicitly supplied target datetimes

Shared logic:

- `scripts/generate_surface/data_helperd/window.py`

Key idea:

- instead of calibrating every eligible minute, this mode only collects rows inside `+/- window_minutes` around selected UTC target datetimes

This is useful when the workflow needs a focused set of timestamps rather than a full-minute sweep.

## minute-svi-excel

CPU entrypoint:

- `scripts/generate_surface/backend/surface_cpu/excel.py`

GPU entrypoint:

- `scripts/generate_surface/backend/surface_gpu/excel.py`

Shared logic:

- `scripts/generate_surface/data_helperd/excel.py`

Purpose:

- extract target timestamps from the news embedding Excel workbook
- convert `PD` + `ET` into UTC timestamps
- run the datetime-window minute-SVI workflow around those news timestamps

Typical use case:

- build minute-SVI outputs aligned with the timestamps that will later be merged with text embeddings

## Config Structure

Config helpers live in:

- `scripts/generate_surface/common/config_utils.py`

The surface-builder configs support small variable expansion, including:

- `${name}`
- `name/...`

This allows a config to define shared paths at the top level and reuse them in job-specific sections.

Common sections include:

- `surface_builder.daily_surface`
- `surface_builder.minute_svi`
- `surface_builder.minute_svi_window`
- `surface_builder.minute_svi_excel`

## minute-SVI Output Semantics

The minute-SVI JSON output is structured by target timestamp and direction.

For the downstream workflow, the important direction semantics are:

- `backward`: aligned with the original news timestamp
- `forward`: aligned with the future offset window

That convention is later preserved by `merge_svi.py` and `merge_vol.py`.

## Typical Commands

Daily surfaces:

```bash
python scripts/generate_surface/main.py daily-surface --config configs/surface_builder/default.yaml
```

All-minute CPU SVI generation:

```bash
python scripts/generate_surface/main.py minute-svi --device cpu --config configs/surface_builder/svi/minute-svi-all.yaml
```

All-minute GPU SVI generation:

```bash
python scripts/generate_surface/main.py minute-svi --device gpu --config configs/surface_builder/svi/minute-svi-all.yaml
```

Excel-driven GPU generation:

```bash
python scripts/generate_surface/main.py minute-svi-excel --device gpu --model cubic --config configs/surface_builder/cubic/minute-cubic-excel.yaml
```

All-minute raw generation:

```bash
python scripts/generate_surface/main.py minute-svi --device cpu --model raw --config configs/surface_builder/raw/minute-raw-all.yaml
```
