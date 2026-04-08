# scripts

This directory contains the runnable research workflows for the repository.

At a high level, the scripts layer is organized into four main stages:

1. `generate_surface`
   - Build daily surfaces or minute-level SVI calibration outputs from raw option trades.
2. `merge_file`
   - Join generated SVI outputs with the news embedding workbook and write training-ready Excel files.
3. `train`
   - Train models on the merged Excel datasets.
4. `generate_result` / `analyze_error`
   - Run trained models on selected samples and inspect the generated outputs.

Existing script-level READMEs:

- `scripts/generate_surface/README.md`
- `scripts/merge_file/README.md`
- `scripts/train/README.md`
- `scripts/generate_result/README.md`
- `scripts/analyze_error/README.md`

## Directory Map

- `scripts/generate_surface`
  - Unified surface-generation CLI.
  - Supports daily-surface generation and several minute-SVI jobs.
- `scripts/merge_file`
  - Builds `merged_svi.xlsx` and `merged_vol.xlsx`.
- `scripts/train`
  - Unified CLI for merged-xlsx training.
- `scripts/train.py`
  - Legacy training entrypoint for the older daily-surface WGAN path.
- `scripts/generate_result`
  - Re-run inference from trained checkpoints and save generated-vs-real comparisons.
- `scripts/analyze_error`
  - Re-run inference, compute per-sample MSE-style error metrics, bootstrap them, and save histograms.

## Recommended Workflow

For the current merged-xlsx pipeline, the typical order is:

```text
raw option files
-> scripts/generate_surface
-> minute_svi_params.json + minute_svi_precalib_points.csv
-> scripts/merge_file
-> merged_svi.xlsx / merged_vol.xlsx
-> scripts/train
-> trained checkpoints and metrics
-> scripts/generate_result or scripts/analyze_error
```

For the legacy daily-surface workflow, `scripts/train.py` still exists and trains directly from the older data path.

## Notes

- `merged_vol.xlsx` is already pair-based for current -> future surface training.
- `merged_svi.xlsx` is primarily an audit workbook; paired SVI forecasting is assembled during training and inference.
- The newer preferred training CLI is `scripts/train/main.py`.
- The newer preferred post-training inspection CLIs are:
  - `scripts/generate_result/main.py`
  - `scripts/analyze_error/main.py`
