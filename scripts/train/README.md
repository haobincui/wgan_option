# train

This directory contains the current training CLIs for the merged Excel workflows.

There are now two training surfaces in the repository:

- the legacy training path in `scripts/train.py`
- the newer merged-xlsx training path in `scripts/train/`

This README focuses on the newer merged-xlsx path and also explains how it relates to the legacy entrypoint.

## Entry Points

Unified CLI:

```bash
python scripts/train/main.py vol-xlsx --config configs/wgan/train_vol_xlsx.yaml
python scripts/train/main.py svi-xlsx --config configs/wgan/train_svi_xlsx.yaml
```

Accepted aliases:

```bash
python scripts/train/main.py train-vol ...
python scripts/train/main.py train-svi ...
```

Individual wrappers:

- `scripts/train/train_vol.py`
- `scripts/train/train_svi.py`

Legacy path:

```bash
python scripts/train.py --config configs/wgan/train_default.yaml
```

## Unified CLI Logic

`scripts/train/main.py` is a thin dispatcher:

- `vol-xlsx` -> `scripts/train/train_vol.py`
- `svi-xlsx` -> `scripts/train/train_svi.py`

The shared CLI behavior lives in `scripts/train/common.py`.

Common flags:

- `--config`
- `--set KEY=VALUE`
- `--dry-run`
- `--print-config`

Shared flow:

1. Parse CLI flags.
2. Load the YAML config.
3. Apply repeated `--set` overrides.
4. Build the trainer class for the selected mode.
5. Either run `dry_run()` or `start_train()`.

## vol-xlsx Training

Implementation wrapper:

- `scripts/train/train_vol.py`

Runtime trainer:

- `wgan_option.train_vol_xlsx.VolSurfaceXlsxTrainer`

Purpose:

- train the WGAN-GP model on paired current/target vol surfaces from `merged_vol.xlsx`

Input semantics:

- `merged_vol.xlsx` already stores paired samples
- one usable row corresponds to:
  - current surface
  - target future surface
  - text embedding

The trainer:

1. loads the workbook with `create_vol_surface_xlsx_dataloaders()`
2. chronologically splits the rows into train/validation partitions
3. initializes the WGAN model with the workbook grid shape and embedding width
4. writes artifacts under a timestamped run directory

## svi-xlsx Training

Implementation wrapper:

- `scripts/train/train_svi.py`

Runtime trainer:

- `wgan_option.train_svi_xlsx.SviXlsxTrainer`

Purpose:

- train a supervised SVI regressor on paired SVI samples derived from `merged_svi.xlsx`

Input semantics:

- `merged_svi.xlsx` is not directly pair-shaped in its `gan_input_ready` sheet
- the runtime logic pairs `backward/current` and `forward/future` rows by `news_row_id`

The trainer:

1. loads usable rows from the SVI workbook
2. pairs `backward` with `forward`
3. pads slice data up to `max_slices`
4. predicts:
  - future SVI regression outputs
  - future slice count
5. tracks train and validation losses
6. writes checkpoints, metrics, normalization stats, and loss curves

## Output Structure

The merged-xlsx training path writes timestamped run directories under:

- `outputs/training/vol_xlsx/<run_ts>/`
- `outputs/training/svi_xlsx/<run_ts>/`

Typical subdirectories and files include:

- `checkpoints/`
- `metrics/`
- `metrics/run_config_<run_ts>.yaml`
- `training_metrics.csv`
- `training_metrics.json`
- `best_checkpoint.json`

The SVI trainer also writes normalization statistics used later by inference and analysis.

## Legacy Training Path

The root-level `scripts/train.py` is still the legacy training entrypoint.

It:

- uses the older `wgan_option.cli.parse_train_args`
- loads the older config surface
- launches `wgan_option.train.WGANTrainer`

This path belongs to the older daily-surface workflow and does not understand the merged-xlsx `vol-xlsx` / `svi-xlsx` split.

Use it only when you intentionally want the historical daily-surface training path.

## Migration Helper

`scripts/train/migrate_training_outputs.py` is a one-time migration utility for older merged-xlsx output directories.

Purpose:

- move old directories like `vol_xlsx_<legacy_suffix>` and `svi_xlsx_<legacy_suffix>`
- move existing top-level run folders like `outputs/vol_xlsx/<run_ts>/`
- into the newer layout:

```text
outputs/training/vol_xlsx/<run_ts>/
outputs/training/svi_xlsx/<run_ts>/
```

It tries to infer the run timestamp from `metrics/run_config_<run_ts>.yaml`, and falls back to the source directory name when needed.

## Typical Commands

Print the resolved training config:

```bash
python scripts/train/main.py vol-xlsx --config configs/wgan/train_vol_xlsx.yaml --print-config
```

Run a dry-run:

```bash
python scripts/train/main.py svi-xlsx --config configs/wgan/train_svi_xlsx.yaml --dry-run
```

Override config fields from the CLI:

```bash
python scripts/train/main.py vol-xlsx \
  --config configs/wgan/train_vol_xlsx.yaml \
  --set batch_size=16 \
  --set learning_rate=0.0001
```
