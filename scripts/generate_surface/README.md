# generate_surface

This directory contains the raw surface-generation workflows.

It covers two distinct tasks:

- daily-surface generation from raw option trade files
- minute-level SVI generation from raw option trade files

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
python scripts/generate_surface/main.py minute-svi --device cpu --config configs/surface_builder/default.yaml
python scripts/generate_surface/main.py minute-svi-excel --device gpu --config configs/surface_builder/minute-svi-excel.yaml
```

Device dispatch is handled by `scripts/generate_surface/dispatch.py`:

- minute-SVI jobs support `--device cpu` and `--device gpu`
- GPU mode requires CUDA
- daily-surface ignores the device choice and always runs its own path

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

## minute-svi

Main CPU entrypoint:

- `scripts/generate_surface/surface_cpu/generate_minute_svi_params.py`

Related GPU entrypoint:

- `scripts/generate_surface/surface_gpu/generate_minute_svi_params.py`

Purpose:

- calibrate minute-level SVI parameters from raw option trade files for all eligible minutes

Core shared logic:

- `scripts/generate_surface/common/minute_svi_common.py`

### High-Level Flow

1. load the minute-SVI config
2. scan raw trade files with chunked CSV reading
3. normalize timestamps to UTC minutes
4. separate futures rows from option rows
5. infer the minute spot from the relevant future contract
6. build option candidates for that minute
7. compute implied vols
8. apply a pre-calibration filter
9. calibrate SVI if the minute has enough valid expiries and strike points
10. write JSON and optional pre-calibration CSV outputs

### Main Outputs

The standard minute-SVI run writes:

- `minute_svi_params.json`
- `minute_svi_precalib_points.csv` if enabled
- `minute_svi_params.log`

These files become the direct inputs for `scripts/merge_file`.

### CPU vs GPU

The CPU and GPU backends share the same overall pipeline.

The main difference is the implied-vol calculation engine:

- CPU path uses the CPU-side Black-Scholes implied vol routine
- GPU path uses the GPU implementation while keeping the same surrounding workflow

## minute-svi-window

CPU entrypoint:

- `scripts/generate_surface/surface_cpu/generate_minute_svi_params_for_datetimes.py`

Purpose:

- generate minute-SVI surfaces only inside windows around explicitly supplied target datetimes

Shared logic:

- `scripts/generate_surface/common/minute_svi_window_common.py`

Key idea:

- instead of calibrating every eligible minute, this mode only collects rows inside `+/- window_minutes` around selected UTC target datetimes

This is useful when the workflow needs a focused set of timestamps rather than a full-minute sweep.

## minute-svi-excel

CPU entrypoint:

- `scripts/generate_surface/surface_cpu/generate_minute_svi_params_from_excel_pd_et.py`

GPU entrypoint:

- `scripts/generate_surface/surface_gpu/generate_minute_svi_params_from_excel_pd_et.py`

Shared logic:

- `scripts/generate_surface/common/minute_svi_excel_common.py`

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
python scripts/generate_surface/main.py minute-svi --device cpu --config configs/surface_builder/default.yaml
```

All-minute GPU SVI generation:

```bash
python scripts/generate_surface/main.py minute-svi --device gpu --config configs/surface_builder/default.yaml
```

Excel-driven GPU generation:

```bash
python scripts/generate_surface/main.py minute-svi-excel --device gpu --config configs/surface_builder/minute-svi-excel.yaml
```
