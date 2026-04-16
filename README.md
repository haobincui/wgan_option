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

### `src/film_wgan`

Standalone CNN WGAN-GP whose generator and critic are conditioned on the news embedding via FiLM (Feature-wise Linear Modulation).

- text encoder produces per-channel `(gamma, beta)` that modulate every convolution over the current surface
- single CNN pass per forward (no dual-branch surface encoder), so "text → current → future" is the dominant causal path
- WGAN-GP training with calendar / butterfly arbitrage penalties, smoothness regularization, and optional reconstruction loss
- used as an architectural baseline against `stylemod_wgan`

### `src/stylemod_wgan`

Standalone StyleGAN-style WGAN-GP that injects a global style vector into a CNN generator.

- style vector built from concatenated `(surface_summary, text_features)` plus noise; style demodulation on weights
- ModulatedConv2d stack + StyledResidualConvBlocks generate a log-IV delta added back to the current surface
- FiLM-conditioned critic reused from `film_wgan` for fair adversarial comparison
- same WGAN-GP + arbitrage / smoothness / reconstruction constraint set as the other CNN variants

### `src/crossattn_wgan`

Standalone WGAN-GP that uses cross-attention between the surface tokens and the text embedding instead of FiLM/style modulation.

- surface-token queries attend to text-key/value projections, letting each spatial location pick up different news context
- shares the merged-vol workbook format and WGAN-GP constraint stack with the other standalone modules
- intended as a comparison point for FiLM (per-channel) vs cross-attention (per-token) text conditioning

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
| `train film-wgan` | `python scripts/film_wgan/main.py train --config ...` | Train the FiLM-conditioned CNN WGAN-GP on merged vol surfaces | `merged_vol.xlsx` plus a training config | timestamped artifacts under `outputs/training/film_wgan/` | Standalone pipeline |
| `train stylemod-wgan` | `python scripts/stylemod_wgan/main.py train --config ...` | Train the StyleGAN-style WGAN-GP on merged vol surfaces | `merged_vol.xlsx` plus a training config | timestamped artifacts under `outputs/training/stylemod_wgan/` | Standalone pipeline |
| `train crossattn-wgan` | `python scripts/crossattn_wgan/main.py train --config ...` | Train the cross-attention-conditioned WGAN-GP on merged vol surfaces | `merged_vol.xlsx` plus a training config | timestamped artifacts under `outputs/training/crossattn_wgan/` | Standalone pipeline |
| `generate-result volgan` | `python scripts/volgan/main.py sample --config ...` | Generate scenarios from a trained VolGAN checkpoint | `merged_vol.xlsx` plus a saved checkpoint | JSON payloads, plots, and `summary.csv` | Standalone pipeline |
| `generate-result cnn-wgan` | `python scripts/cnn_wgan/main.py generate-result --config ...` | Generate scenarios from a trained CNN WGAN checkpoint | `merged_vol.xlsx` plus a saved checkpoint | JSON payloads, plots, and `summary.csv` | Standalone pipeline |
| `generate-result transformer-wgan` | `python scripts/transformer_wgan/main.py generate-result --config ...` | Generate scenarios from a trained Transformer WGAN checkpoint | `merged_vol.xlsx` plus a saved checkpoint | JSON payloads, plots, and `summary.csv` | Standalone pipeline |
| `generate-result film-wgan` | `python scripts/film_wgan/main.py generate-result --config ...` | Generate scenarios from a trained Film WGAN checkpoint | `merged_vol.xlsx` plus a saved checkpoint | JSON payloads, plots, and `summary.csv` | Standalone pipeline |
| `generate-result stylemod-wgan` | `python scripts/stylemod_wgan/main.py generate-result --config ...` | Generate scenarios from a trained StyleMod WGAN checkpoint | `merged_vol.xlsx` plus a saved checkpoint | JSON payloads, plots, and `summary.csv` | Standalone pipeline |
| `generate-result crossattn-wgan` | `python scripts/crossattn_wgan/main.py generate-result --config ...` | Generate scenarios from a trained CrossAttn WGAN checkpoint | `merged_vol.xlsx` plus a saved checkpoint | JSON payloads, plots, and `summary.csv` | Standalone pipeline |

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
| `run_film_wgan_svi_excel.sh` | `scripts/film_wgan/main.py` | background launcher for Film WGAN training | Convenience wrapper |
| `run_stylemod_wgan_svi_excel.sh` | `scripts/stylemod_wgan/main.py` | background launcher for StyleMod WGAN training | Convenience wrapper |
| `run_all_tests.sh` | `pytest` | run the project test suite in one shot | Convenience wrapper |
| `run_merge_vol.sh` | `scripts/merge_file/merge_vol.py` | background launcher for merged-vol workbook construction | Convenience wrapper |
| `run_generate-surface_svi_excel_parallel_bg.sh` | `scripts/generate_surface/main.py` | parallel background launcher for SVI + excel surface generation | Convenience wrapper |

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
│   ├── transformer_wgan/          # Standalone Transformer WGAN training configs
│   ├── film_wgan/                 # Standalone Film (FiLM) WGAN training configs
│   ├── stylemod_wgan/             # Standalone StyleMod WGAN training configs
│   └── crossattn_wgan/            # Standalone CrossAttention WGAN training configs
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
│   ├── film_wgan/                 # Standalone Film WGAN train and generate-result CLI
│   ├── stylemod_wgan/             # Standalone StyleMod WGAN train and generate-result CLI
│   ├── crossattn_wgan/            # Standalone CrossAttention WGAN train and generate-result CLI
│   └── merge_raw_option_data.py   # Raw-data inspection utility
├── src/
│   ├── market_data/               # Contract parsing and raw-data DTO helpers
│   ├── quantlib/                  # Numerical and calendar core
│   ├── utils/                     # Shared utilities (output paths, training paths, CLI helpers)
│   ├── wgan_option/               # Training, inference, configs, and loaders
│   ├── volgan/                    # Standalone MLP VolGAN module (BCE adversarial)
│   ├── cnn_wgan/                  # Standalone CNN WGAN-GP module
│   ├── transformer_wgan/          # Standalone Transformer WGAN-GP module
│   ├── film_wgan/                 # Standalone FiLM-conditioned WGAN-GP module
│   ├── stylemod_wgan/             # Standalone StyleGAN-style modulated WGAN-GP module
│   ├── crossattn_wgan/            # Standalone cross-attention-conditioned WGAN-GP module
│   └── logger.py                  # Shared logging helper
├── tests/                         # Script, quantlib, and workflow tests
│   └── test_standalone_wgan/      # Tests for standalone WGAN modules
├── run_train.sh                   # Training launcher helper
├── run_analyze_error.sh           # Analyze-error launcher helper
├── run_minute_svi_excel_gpu.sh    # Generate-surface launcher helper
├── run_volgan_svi_excel.sh        # VolGAN launcher helper
├── run_cnn_wgan_svi_excel.sh      # CNN WGAN launcher helper
├── run_transformer_wgan_svi_excel.sh  # Transformer WGAN launcher helper
├── run_film_wgan_svi_excel.sh     # Film WGAN launcher helper
├── run_stylemod_wgan_svi_excel.sh # StyleMod WGAN launcher helper
├── run_merge_vol.sh               # merged-vol workbook launcher helper
├── run_all_tests.sh               # test-suite launcher helper
├── run_generate-surface_svi_excel_parallel_bg.sh  # parallel surface-generation launcher
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
- The standalone modules (`volgan`, `cnn_wgan`, `transformer_wgan`, `film_wgan`, `stylemod_wgan`, `crossattn_wgan`) are fully independent of `src/wgan_option` and share only the `merged_vol.xlsx` workbook format and utilities under `src/utils/`.
- Among the CNN-based WGAN variants, `film_wgan` realizes "text → current → future" most directly by letting text embeddings modulate every convolution over the current surface via FiLM; `stylemod_wgan` folds text into a global style vector, and `crossattn_wgan` lets each surface token attend to text features.

## Model Guide

The executable models in this repo split into two families. The mainline merged-workbook paths live under `src/wgan_option`: `vol-xlsx` is the preferred surface-forecasting route, while `svi-xlsx` is the paired parameter-forecasting route. The standalone baselines (`volgan`, `cnn_wgan`, `transformer_wgan`, `film_wgan`, `stylemod_wgan`, `crossattn_wgan`) share `merged_vol.xlsx` as input but change the conditioning mechanism, inductive bias, and training dynamics for research comparison.

### Merged Vol WGAN (vol-xlsx)

#### What it predicts

This model forecasts a future volatility surface from one already paired `merged_vol.xlsx` row:

- `current_surface = backward/current`
- `text_embedding = news embedding at the current timestamp`
- `future_surface = forward/future`

It is the main executable path that most directly matches the thesis framing of `text + current surface -> future surface`.

#### Architecture

The mainline merged-vol trainer is a conditional CNN WGAN-GP. Its generator encodes the current surface, compresses the text embedding with an MLP, concatenates those features with noise, predicts a residual delta, and reconstructs the future surface through a positive residual map. The critic scores candidate `(current, future, text)` tuples.

```text
Generator
current_surface ──► Surface CNN Encoder ──► surface_features ───────────┐
text_embedding ──► Text MLP Encoder ─────► text_features ───────────────┼─► concat ─► Fusion MLP ─► delta
noise ───────────────────────────────────────────────────────────────────┘
                                                                                               │
                                                                                               ▼
                                                           future = softplus(current + delta) + 1e-4

Critic
[current_surface, future_surface] ──► Joint CNN Encoder ──► pair_features ───────┐
text_embedding ─────────────────────► Text MLP Encoder ─► text_features ──────────┼─► classifier ─► score
```

#### Strengths

- It is the closest current executable match to the thesis-facing surface forecasting question.
- The CNN encoder gives a clear local inductive bias over strike and maturity while keeping the predictor simpler than attention-heavy variants.
- The mainline runtime already integrates WGAN-GP, reconstruction pressure, arbitrage penalties, smoothness penalties, scheduling, metrics, and downstream inspection scripts.
- `merged_vol.xlsx` is already pair-based, so the `backward/current -> forward/future` lineage is explicit before training starts.

#### Limitations

- Text enters as a global feature vector, so spatially specific text effects are only learned indirectly through the fusion head and critic.
- Adversarial training is harder to stabilize and explain than the supervised SVI path.
- Direct surface forecasting can work well empirically but is less structurally interpretable than forecasting SVI parameters first.

#### Pipeline and CLI

- Upstream data lineage: `generate_surface -> merge_vol -> merged_vol.xlsx`
- Main training entrypoint:

```shell
python scripts/train/main.py vol-xlsx --config configs/wgan/train_vol_xlsx.yaml
```

- Post-training inspection:

```shell
python scripts/generate_result/main.py vol --config configs/wgan/train_vol_xlsx.yaml
python scripts/analyze_error/main.py vol --config configs/analyze_error/vol.yaml
```

### Merged SVI Regressor (svi-xlsx)

#### What it predicts

This path predicts future padded SVI parameters and future slice count from paired `backward/current -> forward/future` SVI samples. Unlike `merged_vol.xlsx`, `merged_svi.xlsx` remains direction-level. The trainer pairs usable `backward` and `forward` rows from `news_direction_audit` at runtime by `news_row_id`.

#### Architecture

The model is a supervised MLP regressor rather than a GAN. It flattens the current padded SVI representation, appends mask and count information, encodes the current side and the text side separately, fuses them in a shared trunk, and then predicts both the future SVI vector and the future slice-count class.

```text
current_svi_matrix + current_mask + current_slice_count
        │
        └──► flatten / concatenate ──► Current Encoder ────────────────┐
text_embedding ──────────────────────► Text Encoder ────────────────────┼─► Shared Trunk ──► Regression Head ─► future_svi_flat
                                                                        │
                                                                        └──────────────────► Count Head ─────► future_slice_count
```

#### Strengths

- It keeps the SVI representation explicit, which is valuable for thesis experiments about representation choice.
- Supervised training is usually more stable and easier to diagnose than adversarial training.
- Predicting slice count as a separate head keeps the varying-slice structure visible instead of hiding it inside a surface image tensor.
- The output is easier to inspect slice-by-slice than direct surface pixels.

#### Limitations

- The model depends heavily on upstream SVI calibration quality and on the runtime pairing logic staying clean.
- Fixed-width padding with `max_slices` is pragmatic but constrains how much term-structure variation can be represented directly.
- It models parameter vectors, not the full conditional distribution of future surfaces, so scenario diversity is limited relative to GAN paths.

#### Pipeline and CLI

- Upstream data lineage: `generate_surface -> merge_svi -> merged_svi.xlsx -> runtime backward/current + forward/future pairing`
- Main training entrypoint:

```shell
python scripts/train/main.py svi-xlsx --config configs/wgan/train_svi_xlsx.yaml
```

- Post-training inspection:

```shell
python scripts/generate_result/main.py svi --config configs/wgan/train_svi_xlsx.yaml
python scripts/analyze_error/main.py svi --config configs/analyze_error/svi.yaml
```

### VolGAN

#### What it predicts

VolGAN is the lowest-capacity standalone baseline. It flattens the current surface, concatenates it with the text embedding and noise, and predicts a future log-IV increment. The discriminator then judges whether a candidate delta looks realistic under the same condition.

#### Architecture

This model uses an MLP generator and an MLP discriminator with BCE adversarial training rather than WGAN-GP. It keeps the conditioning path simple on purpose.

```text
Generator
current_surface_flat ──┐
text_embedding ────────┼─► concat ─► MLP ─► delta
noise ─────────────────┘
                                        │
                                        ▼
                   future = exp(log(current) + delta)

Discriminator
current_surface_flat ──┐
text_embedding ────────┼─► concat ─► MLP ─► probability(real | current, text, delta)
candidate_delta ───────┘
```

#### Strengths

- It is the cheapest baseline to train and the easiest one to reason about end-to-end.
- Flattened inputs make it a useful control for asking how much spatial inductive bias actually matters.
- BCE plus gradient-matched smoothness penalties gives a historically simple benchmark against the WGAN families.

#### Limitations

- Flattening discards the 2D geometry of the surface, so local strike/maturity structure must be relearned by a plain MLP.
- BCE GAN objectives are typically less stable than WGAN-GP under the same data conditions.
- Arbitrage is handled in post-processing rather than built into the training loss.

#### Pipeline and CLI

- Upstream data lineage: `generate_surface -> merge_vol -> merged_vol.xlsx`
- CLI behavior: `train` only trains; `pipeline` runs `train -> generate-result`; `sample` and `generate-result` rerun scenario generation from an existing run or checkpoint.

```shell
python scripts/volgan/main.py train --config configs/volgan/train_lp.yaml
python scripts/volgan/main.py sample --config configs/volgan/train_lp.yaml
python scripts/volgan/main.py generate-result --config configs/volgan/train_lp.yaml
python scripts/volgan/main.py pipeline --config configs/volgan/train_lp.yaml
```

### CNN WGAN

#### What it predicts

CNN WGAN is the standalone convolutional baseline for forecasting future surfaces from paired `merged_vol.xlsx` rows. It uses the current surface as a 2D grid, the text embedding as a global condition, and noise as a scenario source.

#### Architecture

The generator uses a surface CNN encoder plus a text MLP encoder, then predicts a log-IV delta with a fusion MLP. The critic sees stacked current and future surfaces plus the same text condition.

```text
Generator
current_surface ──► Surface CNN Encoder ──► surface_features ───────────┐
text_embedding ──► Text MLP Encoder ─────► text_features ───────────────┼─► concat ─► Fusion MLP ─► delta
noise ───────────────────────────────────────────────────────────────────┘
                                                                                               │
                                                                                               ▼
                                                           future = exp(log(current) + delta)

Critic
[current_surface, future_surface] ──► Joint CNN Encoder ──► pair_features ───────┐
text_embedding ─────────────────────► Text MLP Encoder ─► text_features ──────────┼─► classifier ─► score
```

#### Strengths

- It gives a clean convolutional baseline with explicit 2D surface structure but without extra conditioning machinery.
- The architecture is easier to interpret than Transformer, FiLM, or StyleMod variants.
- WGAN-GP plus calendar, butterfly, smoothness, and optional reconstruction penalties make it a stronger research baseline than a plain GAN.

#### Limitations

- Text conditioning stays global, so the model cannot explicitly route different news effects to different surface regions.
- It is less expressive than FiLM, StyleMod, or CrossAttention for studying richer text-conditioning mechanisms.
- As a standalone baseline, it duplicates some of the mainline merged-vol ideas rather than replacing them.

#### Pipeline and CLI

- Upstream data lineage: `generate_surface -> merge_vol -> merged_vol.xlsx`
- CLI behavior: `train` defaults to `run_pipeline()`; `sample` is a thin alias-style generate path; `generate-result` is the explicit generate alias.

```shell
python scripts/cnn_wgan/main.py train --config configs/cnn_wgan/train_lp_gen128_disc128.yaml
python scripts/cnn_wgan/main.py sample --config configs/cnn_wgan/train_lp_gen128_disc128.yaml
python scripts/cnn_wgan/main.py generate-result --config configs/cnn_wgan/train_lp_gen128_disc128.yaml
```

### Transformer WGAN

#### What it predicts

Transformer WGAN forecasts future surfaces by treating the surface grid as a token sequence. It augments those surface tokens with one text token and one noise token, then uses self-attention to model longer-range structure than a local CNN can capture directly.

#### Architecture

The generator projects each surface location into token space, adds learnable 2D positional embeddings, prepends text and noise tokens, and uses a Transformer encoder to predict one delta per grid point. The critic uses a CLS token over paired current/future surface tokens.

```text
Generator
current_surface ──► surface tokens + 2D positional embeddings ───────────────┐
text_embedding ──► text token ────────────────────────────────────────────────┼─► Transformer Encoder ─► output head ─► delta(HxW)
noise ───────────► noise token ───────────────────────────────────────────────┘
                                                                                                 │
                                                                                                 ▼
                                                             future = softplus(current + delta) + 1e-4

Critic
[current_surface, future_surface] ──► paired surface tokens + positions ──────┐
text_embedding ─────────────────────► text token ──────────────────────────────┼─► Transformer Encoder ─► CLS head ─► score
learned CLS token ──────────────────────────────────────────────────────────────┘
```

#### Strengths

- Self-attention can express long-range strike/maturity dependencies more directly than local convolution.
- The architecture is useful when the research question is about global surface coordination rather than only local smoothness.
- The generator/critic setup makes it a natural comparison point against CNN-based condition mechanisms.

#### Limitations

- Training cost is higher than the CNN variants, and WGAN-GP on a Transformer critic is harder to stabilize.
- The critic’s gradient-penalty path requires math SDPA rather than faster attention kernels during the double-backward step.
- Tokenized modeling is less immediately interpretable than explicit FiLM or StyleMod conditioning when the goal is to explain how text enters the surface path.

#### Pipeline and CLI

- Upstream data lineage: `generate_surface -> merge_vol -> merged_vol.xlsx`
- CLI behavior: `train` defaults to `run_pipeline()`; `sample` is a thin alias-style generate path; `generate-result` is the explicit generate alias.

```shell
python scripts/transformer_wgan/main.py train --config configs/transformer_wgan/train_lp.yaml
python scripts/transformer_wgan/main.py sample --config configs/transformer_wgan/train_lp.yaml
python scripts/transformer_wgan/main.py generate-result --config configs/transformer_wgan/train_lp.yaml
```

### FiLM WGAN

#### What it predicts

FiLM WGAN is the most direct standalone implementation of the repo’s intended `text -> current -> future` pathway. The current surface remains the main carrier of local structure, but the news embedding modulates every convolutional stage through feature-wise affine transforms.

#### Architecture

The generator first compresses the text embedding into `text_features`, then uses those same features to modulate each convolutional block over the current surface. After the FiLM-conditioned stack, the model fuses surface features, text features, and noise to predict a log-IV delta.

```text
text_embedding ──► text_encoder ──► text_features ───────────┐
                                                             │
current_surface ──► conv1 ──► FiLM1(·, text_features) ──► LeakyReLU
                              ↑
                         text directly modulates current-surface features
                    ──► conv2 ──► FiLM2(·, text_features) ──► LeakyReLU
                    ──► conv3 ──► FiLM3(·, text_features) ──► LeakyReLU
                    ──► FiLM-ResBlocks (each block applies FiLM twice with text_features)
                                  │
                                  ▼
                         surface_features (text-modulated)
                                  │
                                  ▼
    concat([surface_features, text_features, noise]) ─► fusion MLP ─► delta
                                                                         │
                                                                         ▼
                                             future = exp(log(current) + delta)

Critic
[current_surface, future_surface] ──► conv stack ──► FiLM at each stage with text_features ──► classifier ──► score
text_embedding ─────────────────────► text_encoder ────────────────────────────────────────────┘
```

#### Strengths

- The text-conditioning mechanism is explicit at every convolutional stage rather than only at the fusion layer.
- It matches the research intuition that news should reshape how the model reads the current surface, not only how it post-processes a global summary.
- Compared with StyleMod and CrossAttention, the conditioning path stays relatively easy to inspect channel by channel.

#### Limitations

- Text still arrives as one global vector, so modulation is channel-wise rather than location-specific.
- Strong early conditioning can make optimization more sensitive than the plain CNN baseline.
- It is more expressive than the plain CNN baseline, but less flexible than token-level attention for spatially heterogeneous text effects.

#### Pipeline and CLI

- Upstream data lineage: `generate_surface -> merge_vol -> merged_vol.xlsx`
- CLI behavior: `train` defaults to `run_pipeline()`; `sample` is a thin alias-style generate path; `generate-result` is the explicit generate alias.

```shell
python scripts/film_wgan/main.py train --config configs/film_wgan/train_lp_gen128_disc128.yaml
python scripts/film_wgan/main.py sample --config configs/film_wgan/train_lp_gen128_disc128.yaml
python scripts/film_wgan/main.py generate-result --config configs/film_wgan/train_lp_gen128_disc128.yaml
```

### StyleMod WGAN

#### What it predicts

StyleMod WGAN predicts future surfaces from the same paired `merged_vol.xlsx` rows, but it routes the condition through a global style vector rather than through direct per-layer FiLM on the current-surface branch.

#### Architecture

The generator summarizes the current surface into a condition surface summary, encodes the text embedding, combines both into a base style, adds a noise-derived style component, and then uses modulated convolutions and styled residual blocks to produce the delta. The critic reuses the FiLM-conditioned design so that the main generator-side comparison stays focused on conditioning choice.

```text
current_surface ──► condition_surface_encoder ──► surface_summary ───────┐
text_embedding ──► text_encoder ─────────────────► text_features ─────────┼─► concat ─► condition_vector ─► style_base ──┐
noise ───────────► style_noise MLP ────────────────────────────────────────────────────────────────────────────────────────┘
                                                                                                                          │
                                                                                                                          ├─► global_style
                                                                                                                          ▼
current_surface ──► conv stem ──► modulated conv1 ──► modulated conv2 ──► styled residual blocks ──► surface_features ───┐
                                                                                                                            │
                                      concat([surface_features, condition_vector, global_style]) ───────────────────────────┘
                                                                                                                            │
                                                                                                                            ▼
                                                                                     fusion MLP ─► delta ─► future = exp(log(current) + delta)

Critic
[current_surface, future_surface] ──► FiLM-conditioned conv stack with text_features ──► classifier ─► score
text_embedding ─────────────────────► text_encoder ──────────────────────────────────────┘
```

#### Strengths

- A global style vector is a strong inductive bias for coherent surface-wide shifts such as level, skew, or curvature moves.
- It separates condition summarization from the modulated generator path, which is useful for comparing against FiLM’s more direct conditioning.
- Reusing the FiLM critic keeps the architectural comparison centered on generator conditioning rather than on a completely new adversary.

#### Limitations

- One global style can blur location-specific text effects that might matter at particular maturities or strikes.
- The extra condition-surface encoder and style stack make the generator more complex than the plain CNN or FiLM baseline.
- Interpretation is less direct than FiLM because the condition is mixed into a latent style space before modulation happens.

#### Pipeline and CLI

- Upstream data lineage: `generate_surface -> merge_vol -> merged_vol.xlsx`
- CLI behavior: `train` defaults to `run_pipeline()`; `sample` is a thin alias-style generate path; `generate-result` is the explicit generate alias.

```shell
python scripts/stylemod_wgan/main.py train --config configs/stylemod_wgan/train_lp_gen64_disc64_tuned.yaml
python scripts/stylemod_wgan/main.py sample --config configs/stylemod_wgan/train_lp_gen64_disc64_tuned.yaml
python scripts/stylemod_wgan/main.py generate-result --config configs/stylemod_wgan/train_lp_gen64_disc64_tuned.yaml
```

### CrossAttention WGAN

#### What it predicts

CrossAttention WGAN forecasts future surfaces by letting the current-surface feature map attend to text-derived virtual tokens. It is the standalone baseline that most explicitly asks whether text should act locally over surface locations instead of only globally over channels or style vectors.

#### Architecture

The generator first builds a CNN feature map from the current surface, compresses the text embedding, expands that compressed text into a small bank of virtual tokens, and applies multi-head cross-attention from surface queries to text keys/values. The attended feature map is then flattened and fused with the text vector and noise to predict the delta.

```text
current_surface ──► Surface CNN Encoder ──► surface_feature_map ──► Cross-Attention(surface queries, text keys/values) ──► attended_surface_features
                                                                                               ▲
text_embedding ──► Text Encoder ──► text_features ──► text_to_tokens ──────────────────────────┘

flatten(attended_surface_features) ───────────────────────────────────────────────┐
text_features ─────────────────────────────────────────────────────────────────────┼─► concat ─► Fusion MLP ─► delta
noise ─────────────────────────────────────────────────────────────────────────────┘                           │
                                                                                                              ▼
                                                                                          future = exp(log(current) + delta)

Critic
[current_surface, future_surface] ──► joint CNN encoder ──► cross-attention with text tokens ──► classifier ─► score
text_embedding ─────────────────────► text encoder ─► virtual text tokens ───────────────────────┘
```

#### Strengths

- It offers the clearest standalone test of whether text should affect different surface regions differently.
- Cross-attention is more expressive than global fusion when the text signal may matter differently across maturities and strikes.
- It keeps a CNN surface stem, so the model still benefits from local geometric bias before attention is applied.

#### Limitations

- It is more computationally involved than the FiLM and plain CNN baselines.
- The text side is still compressed into a single embedding before token expansion, so the attention mechanism is richer than the input text representation itself.
- Interpreting head-level attention is possible, but less straightforward than reading FiLM channel modulation.

#### Pipeline and CLI

- Upstream data lineage: `generate_surface -> merge_vol -> merged_vol.xlsx`
- CLI behavior: `train` defaults to `run_pipeline()`; `sample` is a thin alias-style generate path; `generate-result` is the explicit generate alias.

```shell
python scripts/crossattn_wgan/main.py train --config configs/crossattn_wgan/train_lp_gen128_disc128.yaml
python scripts/crossattn_wgan/main.py sample --config configs/crossattn_wgan/train_lp_gen128_disc128.yaml
python scripts/crossattn_wgan/main.py generate-result --config configs/crossattn_wgan/train_lp_gen128_disc128.yaml
```


# Pipelines

Each pipeline walks end-to-end from raw inputs to a trained checkpoint and post-training artifacts. Unless noted, commands assume the repo root as the working directory, `py312` conda env is active, and the package is installed editable (`python -m pip install -e .`).

## 0. Environment bootstrap

```shell
# one-time: create env, install deps, and install the repo editable
conda create -n py312 python=3.12 -y
conda activate py312
python -m pip install -r requirements.txt
python -m pip install -e .

# quick sanity check
bash run_all_tests.sh
```

## 1. run SVI vol xlsx (preferred merged-workbook path)

End-to-end: build SVI surface → merge paired vol workbook → train merged vol-surface WGAN → post-training inspection.

```shell
# 1.1 build one processed SVI minute-surface run (GPU)
python scripts/generate_surface/main.py generate_surface \
  --device gpu \
  --model svi \
  --data_range excel \
  --config configs/surface_builder/svi/generate_surface-svi-excel.yaml
# or parallel background:
# bash run_generate-surface_svi_excel_parallel_bg.sh

# 1.2 merge into the training-ready paired workbook
python scripts/merge_file/merge_vol.py \
  --input-dir data/processed/svi-excel/<run_ts>
# or background:
# bash run_merge_vol.sh data/processed/svi-excel/<run_ts>

# 1.3 train merged vol-surface WGAN (preferred entrypoint)
python scripts/train/main.py vol-xlsx \
  --config configs/wgan/train_vol_xlsx_24gb_lp_aggressive_es.yaml
# background launcher:
# bash run_train.sh configs/wgan/train_vol_xlsx_24gb_lp_aggressive_es.yaml

# 1.4 post-training inspection (generate samples + error analysis)
python scripts/generate_result/main.py vol \
  --config configs/generate_result/vol_best.yaml
python scripts/analyze_error/main.py vol \
  --config configs/analyze_error/vol.yaml
# background: bash run_analyze_error.sh configs/analyze_error/vol.yaml
```

## 2. run raw vol xlsx

Raw-excel data path (no SVI parameterization, surfaces built directly from trades).

```shell
python scripts/generate_surface/main.py generate_surface \
  --device gpu \
  --config configs/surface_builder/raw/generate_surface-raw-excel-pipeline.yaml

python scripts/merge_file/merge_vol.py \
  --input-dir data/processed/raw-excel/<run_ts>

python scripts/train/main.py vol-xlsx \
  --config configs/wgan/train_vol_xlsx_24gb_raw_lp_aggressive_es_noconta.yaml

python scripts/generate_result/main.py vol \
  --config configs/wgan/train_vol_xlsx.yaml
```

## 3. run SVI xlsx (supervised SVI regressor)

```shell
python scripts/train/main.py svi-xlsx \
  --config configs/wgan/train_svi_xlsx.yaml

python scripts/generate_result/main.py svi \
  --config configs/wgan/train_svi_xlsx.yaml

python scripts/analyze_error/main.py svi \
  --config configs/analyze_error/svi.yaml
```

## 4. Standalone model pipelines

Each standalone module shares `merged_vol.xlsx` as input and writes to `outputs/training/<model>/`. This section is the quick-start index only. For model rationale, strengths, limitations, and exact CLI semantics, see [Model Guide](#model-guide), especially [VolGAN](#volgan), [CNN WGAN](#cnn-wgan), [Transformer WGAN](#transformer-wgan), [FiLM WGAN](#film-wgan), [StyleMod WGAN](#stylemod-wgan), and [CrossAttention WGAN](#crossattention-wgan).

### 4.1 VolGAN (MLP baseline)

```shell
# full standalone pipeline: train -> generate-result
python scripts/volgan/main.py pipeline --config configs/volgan/train_lp.yaml

# rerun scenario generation from an existing run/checkpoint
python scripts/volgan/main.py sample --config configs/volgan/train_lp.yaml
# or: bash run_volgan_svi_excel.sh configs/volgan/train_lp.yaml
```

### 4.2 CNN WGAN

```shell
# `train` already defaults to the full pipeline: training + generate-result
python scripts/cnn_wgan/main.py train --config configs/cnn_wgan/train_lp_gen128_disc128.yaml
python scripts/cnn_wgan/main.py sample --config configs/cnn_wgan/train_lp_gen128_disc128.yaml
# or: bash run_cnn_wgan_svi_excel.sh configs/cnn_wgan/train_lp_gen128_disc128.yaml
```

### 4.3 Transformer WGAN

```shell
# `train` already defaults to the full pipeline: training + generate-result
python scripts/transformer_wgan/main.py train --config configs/transformer_wgan/train_lp.yaml
python scripts/transformer_wgan/main.py sample --config configs/transformer_wgan/train_lp.yaml
# or: bash run_transformer_wgan_svi_excel.sh configs/transformer_wgan/train_lp.yaml
```

### 4.4 Film WGAN (FiLM-conditioned, "text → current → future" baseline)

```shell
# `train` already defaults to the full pipeline: training + generate-result
python scripts/film_wgan/main.py train --config configs/film_wgan/train_lp_gen128_disc128.yaml
python scripts/film_wgan/main.py sample --config configs/film_wgan/train_lp_gen128_disc128.yaml
# or: bash run_film_wgan_svi_excel.sh configs/film_wgan/train_lp_gen128_disc128.yaml
```

### 4.5 StyleMod WGAN (StyleGAN-style text+noise style vector)

```shell
# `train` already defaults to the full pipeline: training + generate-result
python scripts/stylemod_wgan/main.py train --config configs/stylemod_wgan/train_lp_gen64_disc64_tuned.yaml
python scripts/stylemod_wgan/main.py sample --config configs/stylemod_wgan/train_lp_gen64_disc64_tuned.yaml
# or: bash run_stylemod_wgan_svi_excel.sh configs/stylemod_wgan/train_lp_gen64_disc64_tuned.yaml
```

### 4.6 CrossAttention WGAN

```shell
# `train` already defaults to the full pipeline: training + generate-result
python scripts/crossattn_wgan/main.py train --config configs/crossattn_wgan/train_lp_gen128_disc128.yaml
python scripts/crossattn_wgan/main.py sample --config configs/crossattn_wgan/train_lp_gen128_disc128.yaml
```

## 5. Run monitoring & management

Most `run_*.sh` wrappers write to `logs/<module>/<run_ts>/` with `run.log` and `run.pid`:

```shell
# follow training logs
tail -f logs/stylemod_wgan/<run_ts>/run.log

# check if a background job is still running
kill -0 "$(cat logs/stylemod_wgan/<run_ts>/run.pid)" 2>/dev/null && echo running || echo stopped

# stop a background job
kill "$(cat logs/stylemod_wgan/<run_ts>/run.pid)"
```

Training artifacts (checkpoints, metrics CSV/JSON, loss curves PNG, resolved config) live under `outputs/training/<module>/<data_range>/<run_ts>/`. The best checkpoint is `<module>_best.pt`; `metrics/best_checkpoint.json` records `best_epoch` and `best_metric`.
