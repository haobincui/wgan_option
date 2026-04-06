# Version Information

## Review Target

- Output directory: `outputs/vol_xlsx`
- Actual latest config file in this directory: `outputs/vol_xlsx/metrics/run_config_20260406_125835.yaml`
- Note: the current `outputs/vol_xlsx` artifacts correspond to the `125835` run, not the older `123402` snapshot

## Training Configuration Summary

- Data file: `data/processed_excel_20260330-01/merged_vol.xlsx`
- Sheet name: `gan_input_ready`
- Text mode: `hd`
- Text embedding dimension: `1024`
- Surface grid: `16 x 16`
- Epochs: `100`
- Batch size: `16`
- Learning rate: `0.0002`
- `discriminator_iter`: `5`
- `lambda_gp`: `10.0`
- `lambda_recon`: `10.0`
- `lambda_calendar`: `2.0`
- `lambda_butterfly`: `2.0`
- `lambda_smooth`: `0.1`
- Checkpoint save interval: every `10` epochs

## Dataset Summary

- Total rows in `merged_vol.xlsx`: `2734`
- Trainable rows with `training_candidate_flag = 1`: `2734`
- `pair_quality_label`: all rows are `usable`
- Time span: `2022-01-27 12:25:00+00:00` to `2023-12-27 18:02:00+00:00`
- Split policy: time-ordered split
- Train rows: `2187`
- Validation rows: `547`
- Train end timestamp: `2023-07-06 12:55:00+00:00`
- Validation start timestamp: `2023-07-06 13:50:00+00:00`

## Data Quality Summary

- `current_weighted_iv_rmse`
  - mean: `0.00647`
  - median: `0.00303`
  - p90: `0.01790`
  - max: `0.04987`
- `target_weighted_iv_rmse`
  - mean: `0.00646`
  - median: `0.00302`
  - p90: `0.01802`
  - max: `0.04969`

## Key Training Results

- `g_recon`: `0.10206 -> 0.01888`
- `val_recon`: `0.04828 -> 0.03783`
- `g_calendar`: `0.002608 -> 0.000287`
- `g_butterfly`: `0.002322 -> 0.000148`
- `g_smooth`: `0.013169 -> 0.004265`
- `gp`: `0.134464 -> 0.016879`

## Best Validation Results

- Lowest `val_recon` across all epochs: epoch `3`, `0.03683`
- Best late-stage validation result: epoch `99`, `0.03709`
- Best saved 10-epoch checkpoint: epoch `90`, `val_recon = 0.03761`

## Artifact Status

- Present:
  - `outputs/vol_xlsx/checkpoints/generator_epoch_0010.pt` through `generator_epoch_0100.pt`
  - `outputs/vol_xlsx/checkpoints/generator.pt`
  - `outputs/vol_xlsx/checkpoints/discriminator_epoch_0010.pt` through `discriminator_epoch_0100.pt`
  - `outputs/vol_xlsx/checkpoints/discriminator.pt`
  - `outputs/vol_xlsx/metrics/training_metrics.json`
  - `outputs/vol_xlsx/metrics/loss_curves.png`
- Missing:
  - `outputs/vol_xlsx/metrics/training_metrics.csv`

