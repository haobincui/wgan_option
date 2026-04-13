# train

This directory contains the current merged-workbook training CLIs.

Unified entrypoint:

```bash
python scripts/train/main.py vol-xlsx --config configs/wgan/train_vol_xlsx.yaml
python scripts/train/main.py svi-xlsx --config configs/wgan/train_svi_xlsx.yaml
```

Accepted aliases:

```bash
python scripts/train/main.py train-vol ...
python scripts/train/main.py train-svi ...
```

## Entry Points

- `main.py`
  - thin dispatcher for `vol-xlsx` and `svi-xlsx`
- `common.py`
  - shared `--config`, `--set`, `--dry-run`, `--print-config`
- `train_vol.py`
  - wrapper for merged vol-surface training
- `train_svi.py`
  - wrapper for merged SVI training

## Training Modes

- `vol-xlsx`
  - reads `merged_vol.xlsx`
  - trains the WGAN-GP surface model on paired current/target surfaces
- `svi-xlsx`
  - reads `merged_svi.xlsx`
  - pairs `backward` and `forward` rows by `news_row_id`
  - trains the supervised SVI regressor

## Output Structure

If `output_root` is left empty in the training YAML, the trainer now infers the output family from `data_path`.

Default runtime layout:

```text
outputs/training/<model>-<data_range>/<run_ts>/
```

Example:

```text
outputs/training/svi-all/20260409_010203/
```

Each run directory contains:

- `checkpoints/`
- `samples/`
- `metrics/`
- `metrics/run_config_<run_ts>.yaml`
- `metrics/training_metrics.json`
- `metrics/training_metrics.csv`
- `metrics/loss_curves.png`
- `metrics/best_checkpoint.json`

The SVI trainer also writes:

- `metrics/normalization_stats.json`

If you want a custom location, you can still set `output_root` explicitly in the YAML or via `--set output_root=...`.

## Typical Commands

Print the resolved training config:

```bash
python scripts/train/main.py vol-xlsx --config configs/wgan/train_vol_xlsx.yaml --print-config
```

Run a dry-run:

```bash
python scripts/train/main.py svi-xlsx --config configs/wgan/train_svi_xlsx.yaml --dry-run
```

Override config fields from CLI:

```bash
python scripts/train/main.py vol-xlsx \
  --config configs/wgan/train_vol_xlsx.yaml \
  --set batch_size=16 \
  --set learning_rate=0.0001
```
