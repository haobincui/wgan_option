# wgan_option

`wgan_option` is a research codebase for building bond-option volatility representations and training text-conditioned forecasting models on them.

The repository is being developed as part of a PhD thesis chapter, so the code is organized around reproducibility, data lineage, and experiment comparability as much as around model training itself.

At a high level, the project asks how news embeddings and option-surface representations can be combined to study and forecast changes in bond-option volatility structure. The current repo supports both vol-surface workflows and SVI-parameter workflows, plus the supporting analysis steps needed to inspect model behavior after training.

## Architecture Overview

The main executable flow is:

```text
raw option data
-> scripts/generate_surface
-> data/processed/<model>-<data_range>/<run_ts>/
-> scripts/merge_file
-> merged_svi.xlsx / merged_vol.xlsx / merged_params.xlsx
-> scripts/train
-> outputs/training/<family>/<run_ts>/
-> scripts/generate_result or scripts/analyze_error
```

This is not just ETL. In thesis terms, it is the experimental data lineage that links raw inputs, generated surfaces, merged audit workbooks, trainable datasets, saved checkpoints, and post-training evaluation artifacts.

## `src/` Architecture

### `src/wgan_option`

Training and experiment runtime code.

- config models and CLI-facing config parsing for training, result generation, and error analysis
- merged-xlsx trainers for vol-surface WGAN training and SVI regressor training
- neural network definitions for generator, discriminator, GAN wrapper, and SVI regressor
- dataset loaders for merged workbooks and the older daily-surface training path
- shared helpers for inference, metrics export, run-directory layout, and training plots

This is the package to read first when reviewing model behavior, experiment configuration, or output artifact conventions.

### `src/quantlib`

Numerical and calendar core for surface construction and reconstruction.

- business-day calendars, holiday logic, daycount conventions, and date utilities
- implied-volatility surface abstractions and interpolation helpers
- raw option-surface construction from trade files
- lower-level quantitative utilities used by generation, reconstruction, and analysis workflows

This package is the main source of truth for how surfaces and SVI parameters are interpreted numerically.

### `src/market_data`

Market-data contracts and preprocessing helpers.

- contract parsing for futures and options
- DTOs for raw quote and trade records
- filtering and validation helpers for raw market-data files

This package is less central than `wgan_option` and `quantlib` for the current merged-workbook workflows, but it still documents important assumptions about the upstream data model.

### `src/volgan`

Standalone MLP-based conditional GAN for volatility surface generation.

- MLP generator and discriminator with Softplus activations and Sigmoid output
- BCE adversarial training with label smoothing (real=0.9, fake=0.0)
- automatic gradient matching to balance smoothness penalties against adversarial loss
- arbitrage constraints applied in post-processing only (not in training loss)
- see [src/volgan/README.md](src/volgan/README.md) for architecture and implementation details

### `src/cnn_wgan`

Standalone CNN-based Wasserstein GAN for volatility surface generation.

- convolutional surface encoder with residual blocks, text MLP encoder, and fusion MLP
- WGAN-GP adversarial training with gradient penalty for Lipschitz constraint
- calendar and butterfly arbitrage penalties applied during training
- weighted smoothness regularization across maturity and strike dimensions
- see [src/cnn_wgan/README.md](src/cnn_wgan/README.md) for architecture and implementation details

### `src/transformer_wgan`

Standalone Transformer-based Wasserstein GAN for volatility surface generation.

- Transformer encoder with learnable 2D positional embeddings and CLS token aggregation
- WGAN-GP adversarial training with Math SDPA backend for double-backward compatibility
- configurable objective: pure adversarial or adversarial + L1 reconstruction loss
- calendar/butterfly arbitrage penalties with optional constraint warmup
- LR scheduling, early stopping, and baseline-aware evaluation metrics
- see [src/transformer_wgan/README.md](src/transformer_wgan/README.md) for architecture and implementation details

### Shared Utilities

- `src/logger.py`
  - shared logging setup helper for consistent runtime logs
- `src/utils/draw.py`
  - local plotting helper for visualizing processed surface data
- `src/utils/output_paths.py`
  - dataset-aware output root resolution and checkpoint discovery
- `src/utils/training_paths.py`
  - run directory and generate-result directory layout helpers
- `src/utils/standalone_cli.py`
  - shared CLI entrypoint builder for standalone modules

## `scripts/` Jobs

The repo contains several script families. The preferred entrypoints are the unified CLIs under `scripts/generate_surface`, `scripts/train`, `scripts/generate_result`, and `scripts/analyze_error`.

### Core Pipeline Jobs

| Job | Entrypoint | Task | Main input | Main output | Status |
| --- | --- | --- | --- | --- | --- |
| `generate_surface` | `python scripts/generate_surface/main.py generate_surface ...` | Build minute-level surface or parameter outputs from raw option files | raw option data under `data/raw/option_data` plus a surface-builder config | one processed run directory under `data/processed/<model>-<data_range>/<run_ts>/` | Core pipeline |
| `merge_svi` | `python scripts/merge_file/merge_svi.py --input-dir ...` | Build a direction-level SVI audit workbook | one generated run directory plus the news embedding workbook | `merged_svi.xlsx` | Core pipeline |
| `merge_vol` | `python scripts/merge_file/merge_vol.py --input-dir ...` | Reconstruct paired current/target vol surfaces and build the training workbook | one generated run directory plus the news embedding workbook | `merged_vol.xlsx` | Core pipeline |
| `merge_params` | `python scripts/merge_file/merge_params.py --input-dir ...` | Build a model-neutral parameter audit workbook across surface models | one generated run directory plus the news embedding workbook | `merged_params.xlsx` | Core pipeline |
| `train vol-xlsx` | `python scripts/train/main.py vol-xlsx ...` | Train the merged vol-surface WGAN workflow | `merged_vol.xlsx` plus a training config | timestamped training artifacts under `outputs/training/vol_xlsx/` or inferred output root | Core pipeline, preferred training entrypoint |
| `train svi-xlsx` | `python scripts/train/main.py svi-xlsx ...` | Train the merged SVI supervised regressor workflow | `merged_svi.xlsx` plus a training config | timestamped training artifacts under `outputs/training/svi_xlsx/` or inferred output root | Core pipeline, preferred training entrypoint |

### Supporting Research Jobs

| Job | Entrypoint | Task | Main input | Main output | Status |
| --- | --- | --- | --- | --- | --- |
| `generate_result vol` | `python scripts/generate_result/main.py vol ...` | Re-run a trained vol model on selected samples and save comparison payloads | `merged_vol.xlsx` plus a saved generator checkpoint | JSON payloads, plots, and `summary.csv` | Supporting research |
| `generate_result svi` | `python scripts/generate_result/main.py svi ...` | Predict future SVI, reconstruct surfaces, and save sample-level comparisons | `merged_svi.xlsx` plus a saved SVI regressor checkpoint | JSON payloads, plots, and `summary.csv` | Supporting research |
| `generate_result plot` | `python scripts/generate_result/main.py plot --input-json ...` | Render a saved sample payload into comparison plots | one saved payload JSON | PNG comparison plots | Supporting research |
| `analyze_error vol` | `python scripts/analyze_error/main.py vol ...` | Compute distributional error summaries for generated future vol surfaces | `merged_vol.xlsx` plus a saved generator checkpoint | `errors.csv`, bootstrap outputs, and histograms | Supporting research |
| `analyze_error svi` | `python scripts/analyze_error/main.py svi ...` | Compute distributional error summaries after SVI prediction and surface reconstruction | `merged_svi.xlsx` plus a saved SVI regressor checkpoint | `errors.csv`, bootstrap outputs, and histograms | Supporting research |

### Standalone Model Training Jobs

| Job | Entrypoint | Task | Main input | Main output | Status |
| --- | --- | --- | --- | --- | --- |
| `train volgan` | `python scripts/volgan/main.py train --config ...` | Train the standalone MLP VolGAN on merged vol surfaces | `merged_vol.xlsx` plus a training config | timestamped artifacts under `outputs/training/volgan/` | Standalone pipeline |
| `train cnn-wgan` | `python scripts/cnn_wgan/main.py train --config ...` | Train the standalone CNN WGAN-GP on merged vol surfaces | `merged_vol.xlsx` plus a training config | timestamped artifacts under `outputs/training/cnn_wgan/` | Standalone pipeline |
| `train transformer-wgan` | `python scripts/transformer_wgan/main.py train --config ...` | Train the standalone Transformer WGAN-GP on merged vol surfaces | `merged_vol.xlsx` plus a training config | timestamped artifacts under `outputs/training/transformer_wgan/` | Standalone pipeline |
| `generate-result volgan` | `python scripts/volgan/main.py sample --config ...` | Generate scenarios from a trained VolGAN checkpoint | `merged_vol.xlsx` plus a saved checkpoint | JSON payloads, plots, and `summary.csv` | Standalone pipeline |
| `generate-result cnn-wgan` | `python scripts/cnn_wgan/main.py generate-result --config ...` | Generate scenarios from a trained CNN WGAN checkpoint | `merged_vol.xlsx` plus a saved checkpoint | JSON payloads, plots, and `summary.csv` | Standalone pipeline |
| `generate-result transformer-wgan` | `python scripts/transformer_wgan/main.py generate-result --config ...` | Generate scenarios from a trained Transformer WGAN checkpoint | `merged_vol.xlsx` plus a saved checkpoint | JSON payloads, plots, and `summary.csv` | Standalone pipeline |

### Maintenance and Data Utility Jobs

| Job | Entrypoint | Task | Main input | Main output | Status |
| --- | --- | --- | --- | --- | --- |
| `merge_raw_option_data` | `python scripts/merge_raw_option_data.py ...` | Merge raw gzip CSV files into one inspection dataset | a directory of raw `*.csv.gz` option files | one merged gzip CSV | Maintenance / data utility |
| `migrate_training_outputs` | `python scripts/train/migrate_training_outputs.py ...` | Move older training run directories into the newer `outputs/training/...` layout | legacy training output directories | migrated run directories in the normalized layout | Maintenance / migration |

### Convenience Wrappers

These are helpers for launching background jobs. They are not the primary APIs of the repo.

| Wrapper | Wraps | Purpose | Status |
| --- | --- | --- | --- |
| `run_train.sh` | `scripts/train/main.py` | background launcher for training jobs | Convenience wrapper |
| `run_analyze_error.sh` | `scripts/analyze_error/main.py` | background launcher for error-analysis jobs | Convenience wrapper |
| `run_minute_svi_excel_gpu.sh` | `scripts/generate_surface/main.py generate_surface ...` | background launcher for GPU `svi` + `excel` generation | Convenience wrapper |
| `run_volgan_svi_excel.sh` | `scripts/volgan/main.py` | background launcher for VolGAN train + generate-result | Convenience wrapper |
| `run_cnn_wgan_svi_excel.sh` | `scripts/cnn_wgan/main.py` | background launcher for CNN WGAN train + generate-result | Convenience wrapper |
| `run_transformer_wgan_svi_excel.sh` | `scripts/transformer_wgan/main.py` | background launcher for Transformer WGAN train + generate-result | Convenience wrapper |

## Semantics That Matter

- `backward` means the current snapshot aligned with the original news timestamp.
- `forward` means the future snapshot aligned with the forward offset window.
- In practice, this means `backward = current` and `forward = future` throughout the merged workflows.
- `merged_vol.xlsx` is already paired:
  - one usable row is already `current_surface -> target_surface`
  - this is the training-ready workbook for vol-surface forecasting
- `merged_svi.xlsx` is direction-level, not pair-level:
  - the SVI training path pairs `backward` and `forward` rows later during training
  - the executable pairing logic uses `news_direction_audit`, not the direction-level `gan_input_ready` sheet
- merged-xlsx training supports three text embedding modes:
  - `hd`
  - `lp`
  - `concat`
- the default text mode remains `hd`, which keeps the merged workflows aligned with the earlier experiments

## Canonical Commands

Install the package in editable mode:

```bash
python -m pip install -e .
```

Build one processed minute-surface run:

```bash
python scripts/generate_surface/main.py generate_surface \
  --device gpu \
  --model svi \
  --data_range excel \
  --config configs/surface_builder/svi/generate_surface-svi-excel.yaml
```

Build merged workbooks from one processed run:

```bash
python scripts/merge_file/merge_svi.py --input-dir data/processed/svi-excel/<run_ts>
python scripts/merge_file/merge_vol.py --input-dir data/processed/svi-excel/<run_ts>
```

Train the preferred merged-workbook paths:

```bash
python scripts/train/main.py vol-xlsx --config configs/wgan/train_vol_xlsx.yaml
python scripts/train/main.py svi-xlsx --config configs/wgan/train_svi_xlsx.yaml
```

Run post-training inspection:

```bash
python scripts/generate_result/main.py vol --config configs/wgan/train_vol_xlsx.yaml
python scripts/analyze_error/main.py vol --config configs/analyze_error/vol.yaml
```

Train the standalone model variants:

```bash
python scripts/volgan/main.py train --config configs/volgan/train_lp.yaml
python scripts/cnn_wgan/main.py train --config configs/cnn_wgan/train_lp.yaml
python scripts/transformer_wgan/main.py train --config configs/transformer_wgan/train_lp.yaml
```

Or use the convenience shell wrappers (background execution):

```bash
bash run_volgan_svi_excel.sh configs/volgan/train_lp.yaml
bash run_cnn_wgan_svi_excel.sh configs/cnn_wgan/train_lp.yaml
bash run_transformer_wgan_svi_excel.sh configs/transformer_wgan/train_lp.yaml
```

For flags, advanced configuration, and workflow-specific details, read the script-level READMEs under `scripts/`.

## Project Layout

```text
wgan_option/
├── configs/
│   ├── analyze_error/             # Error-analysis configs
│   ├── generate_result/           # Post-training inference configs
│   ├── surface_builder/           # Surface-generation configs by model
│   ├── wgan/                      # Merged-workbook training configs
│   ├── volgan/                    # Standalone VolGAN training and sampling configs
│   ├── cnn_wgan/                  # Standalone CNN WGAN training configs
│   └── transformer_wgan/          # Standalone Transformer WGAN training configs
├── data/                          # Raw inputs and processed run directories
├── docs/                          # Thesis design notes and architecture writeups
├── scripts/
│   ├── analyze_error/             # Post-training error distribution analysis
│   ├── generate_result/           # Post-training sample generation and plotting
│   ├── generate_surface/          # Unified minute surface-generation CLI
│   ├── merge_file/                # Workbook construction from generated runs
│   ├── train/                     # Unified merged-xlsx training CLI
│   ├── volgan/                    # Standalone VolGAN train and sample CLI
│   ├── cnn_wgan/                  # Standalone CNN WGAN train and generate-result CLI
│   ├── transformer_wgan/          # Standalone Transformer WGAN train and generate-result CLI
│   └── merge_raw_option_data.py   # Raw-data inspection utility
├── src/
│   ├── market_data/               # Contract parsing and raw-data DTO helpers
│   ├── quantlib/                  # Numerical and calendar core
│   ├── utils/                     # Shared utilities (output paths, training paths, CLI helpers)
│   ├── wgan_option/               # Training, inference, configs, and loaders
│   ├── volgan/                    # Standalone MLP VolGAN module (BCE adversarial)
│   ├── cnn_wgan/                  # Standalone CNN WGAN-GP module
│   ├── transformer_wgan/          # Standalone Transformer WGAN-GP module
│   └── logger.py                  # Shared logging helper
├── tests/                         # Script, quantlib, and workflow tests
│   └── test_standalone_wgan/      # Tests for standalone WGAN modules
├── run_train.sh                   # Training launcher helper
├── run_analyze_error.sh           # Analyze-error launcher helper
├── run_minute_svi_excel_gpu.sh    # Generate-surface launcher helper
├── run_volgan_svi_excel.sh        # VolGAN launcher helper
├── run_cnn_wgan_svi_excel.sh      # CNN WGAN launcher helper
├── run_transformer_wgan_svi_excel.sh  # Transformer WGAN launcher helper
└── README.md
```

## Where To Read Next

Thesis design and experiment framing:

- [docs/input_vol.md](docs/input_vol.md)
- [docs/input_svi.md](docs/input_svi.md)
- [docs/vol_surface_gan_architecture.md](docs/vol_surface_gan_architecture.md)
- [docs/current_executable_workflows.md](docs/current_executable_workflows.md)

Script-level operational details:

- [scripts/README.md](scripts/README.md)
- [scripts/generate_surface/README.md](scripts/generate_surface/README.md)
- [scripts/merge_file/README.md](scripts/merge_file/README.md)
- [scripts/train/README.md](scripts/train/README.md)
- [scripts/generate_result/README.md](scripts/generate_result/README.md)
- [scripts/analyze_error/README.md](scripts/analyze_error/README.md)

Training diagnostics and model notes:

- [docs/gan_model_detailed_architecture.md](docs/gan_model_detailed_architecture.md)
- [docs/svi_regressor_architecture.md](docs/svi_regressor_architecture.md)
- [docs/training_loss_curves.md](docs/training_loss_curves.md)
- [docs/reduce_lr_on_plateau.md](docs/reduce_lr_on_plateau.md)

Standalone module architecture:

- [src/volgan/README.md](src/volgan/README.md)
- [src/cnn_wgan/README.md](src/cnn_wgan/README.md)
- [src/transformer_wgan/README.md](src/transformer_wgan/README.md)

## Notes

- The preferred merged-workbook training entrypoint is `scripts/train/main.py`; older daily-surface training logic still lives under `src/wgan_option`.
- The merged vol workflow is currently the closest executable path to the thesis-facing current-surface -> future-surface forecasting setup.
- The merged SVI workflow is a paired forecasting implementation, even though the source workbook remains direction-oriented.
- The script families under `scripts/` are intentionally thin wrappers around reusable logic in `src/`.
- The standalone modules (`volgan`, `cnn_wgan`, `transformer_wgan`) are fully independent of `src/wgan_option` and share only the `merged_vol.xlsx` workbook format and utilities under `src/utils/`.


# Pipelines


## 1. run vol xlsx


## 2. run raw vol 
```shell
python -m pip install -e .

python scripts/generate_surface/main.py generate_surface \
  --device gpu \
  --config configs/surface_builder/raw/generate_surface-raw-excel-pipeline.yaml

python scripts/merge_file/merge_vol.py \
  --input-dir data/processed/raw-excel/20260410-raw-01

# python scripts/train/main.py vol-xlsx \
#   --config configs/wgan/train_vol_xlsx_24gb_raw_lp_aggressive_es.yaml

python scripts/train/main.py vol-xlsx \
  --config configs/wgan/train_vol_xlsx_24gb_raw_lp_aggressive_es_noconta.yaml


python scripts/generate_result/main.py vol \
  --config configs/wgan/train_vol_xlsx.yaml
```
