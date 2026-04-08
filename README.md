# wgan_option

`wgan_option` is a research codebase for building bond-option volatility representations and training text-conditioned forecasting models on them.

The repo currently supports three closely related workflows:

- surface generation from raw option data
- merged workbook construction that joins option-side artifacts with news embeddings
- model training on either vol surfaces or SVI parameter representations

The project is being developed as part of a PhD thesis chapter, so reproducibility, data lineage, and experiment comparability matter as much as model code.

## Main Workflows

### 1. Surface generation

Build daily surfaces or minute-level SVI outputs from raw option data.

Main entrypoint:

```bash
python scripts/generate_surface/main.py --help
```

Common commands:

```bash
python scripts/generate_surface/main.py minute-svi-excel --device gpu --config configs/surface_builder/minute-svi-excel.yaml
python scripts/generate_surface/main.py minute-svi --device cpu --input-glob "data/raw/option_data/**/*.csv.gz"
python scripts/generate_surface/main.py daily-surface --config configs/surface_builder/default.yaml
```

### 2. Merge generated outputs with news embeddings

Build training-ready Excel workbooks from minute-SVI outputs.

```bash
python scripts/merge_file/merge_svi.py --input-dir data/processed_excel_20260330-01
python scripts/merge_file/merge_vol.py --input-dir data/processed_excel_20260330-01
```

Outputs:

- `merged_svi.xlsx`
- `merged_vol.xlsx`

Important semantics:

- `merged_vol.xlsx` is already pair-based:
  - `current_surface` corresponds to the backward snapshot
  - `target_surface` corresponds to the forward snapshot
- `merged_svi.xlsx` is direction-based:
  - the SVI trainer pairs backward and forward rows during training
  - the executable training sheet is `news_direction_audit`, not `gan_input_ready`

### 3. Train models

The preferred training CLI is:

```bash
python scripts/train/main.py --help
```

Merged vol-surface WGAN:

```bash
python scripts/train/main.py vol-xlsx --config configs/wgan/train_vol_xlsx.yaml
```

Merged SVI supervised training:

```bash
python scripts/train/main.py svi-xlsx --config configs/wgan/train_svi_xlsx.yaml
```

Dry runs:

```bash
python scripts/train/train_vol.py --config configs/wgan/train_vol_xlsx.yaml --dry-run
python scripts/train/train_svi.py --config configs/wgan/train_svi_xlsx.yaml --dry-run
```

Legacy daily-surface WGAN training is still available:

```bash
python scripts/train.py --config configs/wgan/train_default.yaml
```

## Training Modes

### Legacy daily-surface WGAN

- entrypoint: `scripts/train.py`
- config: `configs/wgan/train_default.yaml`
- dataset source: raw option data plus daily aggregated news embeddings
- target: proxy future surface forecasting

### Merged vol-surface WGAN

- entrypoint: `scripts/train/main.py vol-xlsx`
- config: `configs/wgan/train_vol_xlsx.yaml`
- dataset source: `merged_vol.xlsx`
- target: `current_surface + text_embedding -> target_surface`
- model: conditional WGAN-GP with reconstruction and arbitrage-aware penalties

### Merged SVI supervised trainer

- entrypoint: `scripts/train/main.py svi-xlsx`
- config: `configs/wgan/train_svi_xlsx.yaml`
- dataset source: `merged_svi.xlsx`, sheet `news_direction_audit`
- target: future padded SVI parameters plus future slice count
- model: supervised MLP regressor

## Current Training Features

The training stack now supports:

- resolved run-config snapshots under the metrics directory
- JSON and CSV metrics export
- loss-curve plots
- periodic checkpoints
- best-checkpoint saving
- optional early stopping
- optional `ReduceLROnPlateau`

### Best-checkpoint rules

- WGAN vol path: best model is selected by lowest `val_recon`
- SVI path: best model is selected by lowest `val_regression`

### Plateau LR scheduling

Available config fields:

```yaml
use_reduce_lr_on_plateau: false
reduce_lr_factor: 0.5
reduce_lr_patience: 8
reduce_lr_min_lr: 1.0e-5
```

The global config default is `false`, while individual experiment YAMLs can override it.

Current monitor metrics:

- WGAN vol path: `val_recon`
- SVI path: `val_regression`

## Output Artifacts

Typical output trees:

- `outputs/training/vol_xlsx`
- `outputs/training/svi_xlsx`
- `outputs/checkpoints` for the legacy path

Typical metrics artifacts:

- `training_metrics.json`
- `training_metrics.csv`
- `loss_curves.png`
- `best_checkpoint.json`
- `run_config_YYYYMMDD_HHMMSS.yaml`

Typical checkpoint artifacts:

- WGAN final:
  - `generator.pt`
  - `discriminator.pt`
- WGAN best:
  - `generator_best.pt`
  - `discriminator_best.pt`
- SVI final:
  - `svi_regressor.pt`
- SVI best:
  - `svi_regressor_best.pt`

## Text Embedding Modes

Merged-xlsx training supports:

- `hd`
- `lp`
- `concat`

Defaults remain aligned with earlier experiments:

- default mode is `hd`
- if `concat` is used, embedding dimension is inferred automatically from both embedding columns

## Installation

Requirements:

- Python 3.10+

Editable install:

```bash
python -m pip install -e .
```

If you are installing on a GPU server with CUDA 12.4 drivers, install the matching PyTorch wheel after the editable install:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Requirements-based install is also available:

```bash
python -m pip install -r requirements.txt
```

## Project Layout

```text
wgan_option/
├── configs/
│   ├── surface_builder/            # Surface-generation jobs
│   └── wgan/                       # Legacy + merged training configs
├── docs/                           # Research and training documentation
├── scripts/
│   ├── generate_surface/           # Unified generation CLI
│   ├── merge_file/                 # Workbook construction
│   ├── train/                      # Unified merged-xlsx training CLI
│   └── train.py                    # Legacy daily-surface training entrypoint
├── src/
│   ├── quantlib/                   # Vol surface, SVI, and numerical utilities
│   ├── market_data/                # Market-data contracts / DTO helpers
│   └── wgan_option/                # Trainers, loaders, configs, models
├── tests/
└── outputs/
```

## Recommended Reading

Project and experiment design:

- [docs/input_vol.md](docs/input_vol.md)
- [docs/input_svi.md](docs/input_svi.md)
- [docs/vol_surface_gan_architecture.md](docs/vol_surface_gan_architecture.md)

Training diagnostics:

- [docs/gan_model_detailed_architecture.md](docs/gan_model_detailed_architecture.md)
- [docs/training_loss_curves.md](docs/training_loss_curves.md)
- [docs/reduce_lr_on_plateau.md](docs/reduce_lr_on_plateau.md)

## Notes

- The merged vol workflow is currently the closest executable path to the thesis-facing vol-surface forecasting setup.
- The merged SVI workflow is a paired forecasting implementation, even though `merged_svi.xlsx` itself is direction-oriented.
- The repo still contains legacy workflows for comparison and transition, so not every script uses the same data representation.
