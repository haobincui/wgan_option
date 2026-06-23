# Project Onboarding for Future Agents

This document is a practical English handoff for coding agents working in this
repository. It summarizes the current understanding of the project, the thesis
intent, the executable pipeline, model families, experiment artifacts, and
known caveats.

It is not a polished thesis chapter. It is meant to help a future model or
engineer get productive quickly without flattening the research semantics that
matter for the dissertation work.

## 1. What This Project Is

This repository is a PhD research codebase for studying and forecasting bond
option volatility structures using text embeddings and option-surface
representations.

At a high level, the research question is:

```text
Can news text embeddings help forecast short-horizon changes in bond-option
volatility structures?
```

The project compares several representations of the same underlying volatility
forecasting problem:

- daily proxy volatility-like surfaces from older raw-option workflows
- SVI parameter representations
- volatility surfaces reconstructed from SVI or related surface models
- standalone GAN/WGAN model families that differ mainly in how text conditions
  the surface transition

This is a research codebase, not a generic product repo. Changes should be
judged by whether they preserve or improve:

- experimental reproducibility
- data lineage from raw trades to model inputs
- clarity of current/future semantics
- comparability across surface representations and model families
- thesis-facing explanation of quality filters and modeling choices

## 2. How To Read Docs vs Code

The project has two overlapping sources of truth:

- `docs/` describes thesis design, intended data semantics, architecture notes,
  and saved experiment reviews.
- `src/`, `scripts/`, `configs/`, and `tests/` describe the current executable
  implementation.

Do not assume that a mismatch means either side is simply wrong.

Use this rule:

- treat `docs/` as the thesis design record and experiment interpretation layer
- treat `src/` and `scripts/` as the current runnable reality
- call out mismatches explicitly before changing behavior

Important design docs to read first:

- `docs/current_executable_workflows.md`
- `docs/input_vol.md`
- `docs/input_svi.md`
- `docs/vol_surface_gan_architecture.md`
- `docs/svi_regressor_architecture.md`
- `docs/gan_model_detailed_architecture.md`
- `docs/training_results_comparison.md`

The root `AGENTS.md` is also important because it describes project-specific
semantics and agent expectations.

## 3. Critical Semantics

The most important semantic convention is the direction convention:

```text
backward = current = the original news timestamp
forward  = future  = news timestamp + offset_minutes
```

The default offset is 5 minutes.

These names appear throughout generated JSON, merge workbooks, dataloaders,
training code, result generation, and analysis. Do not collapse them into
generic "input" and "target" labels too early when changing the project.

The current practical mapping is:

- `backward` side becomes `current_surface` or current SVI
- `forward` side becomes `target_surface` or future SVI
- one forecasting example is usually `current + text -> future`

This distinction is part of the thesis lineage. It must remain explicit in
new datasets, configs, metrics, and documentation.

## 4. Repository Map

Core packages:

- `src/wgan_option`
  - Main training and experiment runtime code.
  - Contains config parsing, merged-xlsx dataloaders, mainline WGAN, SVI
    regressor, deterministic vol regressor, result generation helpers, metrics,
    and artifact writing.
- `src/quantlib`
  - Numerical and calendar core.
  - Contains daycount/calendar utilities, implied volatility logic, SVI/SABR/raw
    surface classes, interpolation, and surface reconstruction.
- `src/market_data`
  - Contract parsing, DTOs, and market-data helper types.
  - Less central than `wgan_option` and `quantlib` for the current merged-xlsx
    workflows, but still relevant for upstream data assumptions.

Pipeline scripts:

- `scripts/generate_surface`
  - Unified minute-level surface generation entrypoint.
  - Current command family is `generate_surface` with model and data-range
    selectors.
- `scripts/merge_file`
  - Builds Excel workbooks that connect generated surface outputs to the news
    embedding workbook.
  - Main scripts: `merge_svi.py`, `merge_vol.py`, `merge_params.py`.
- `scripts/train`
  - Mainline merged-workbook training entrypoint.
  - Current subcommands include `vol-xlsx`, `vol-regression-xlsx`, and
    `svi-xlsx`.
- `scripts/generate_result`
  - Runs trained mainline models on selected samples and writes comparison
    payloads, plots, and summaries.
- `scripts/analyze_error`
  - Computes distributional error summaries and related plots.

Standalone model families:

- `src/volgan` and `scripts/volgan`
- `src/cnn_wgan` and `scripts/cnn_wgan`
- `src/transformer_wgan` and `scripts/transformer_wgan`
- `src/film_wgan` and `scripts/film_wgan`
- `src/stylemod_wgan` and `scripts/stylemod_wgan`
- `src/crossattn_wgan` and `scripts/crossattn_wgan`

These standalone families generally share `merged_vol.xlsx` as input but change
the architecture, conditioning mechanism, and training dynamics.

## 5. End-To-End Data Pipeline

The practical pipeline is:

```text
raw option trades
  + news embedding workbook
  -> scripts/generate_surface/main.py
  -> data/processed/<model>-<data_range>/<run_ts>/
     - surface-<model>-<data_range>.json
     - surface-<model>-<data_range>-precalib-points.csv
     - surface-resolved_config.yaml
  -> scripts/merge_file/merge_svi.py
  -> scripts/merge_file/merge_vol.py
  -> scripts/merge_file/merge_params.py
  -> merged_svi.xlsx / merged_vol.xlsx / merged_params.xlsx
  -> scripts/train/main.py
  -> outputs/training/<family>/<dataset>/<run_ts>/
  -> scripts/generate_result/main.py or standalone generate-result
  -> scripts/analyze_error/main.py
```

Older processed runs may still use legacy names:

- `minute_svi_params.json`
- `minute_svi_precalib_points.csv`

The merge helpers currently prefer the new `surface-*` names and fall back to
legacy names for compatibility.

### Raw Inputs

The expected raw inputs are:

- option trade files under `data/raw/option_data`
- news embedding workbook under `data/raw/text_embedding/`

In this workspace review, the local `data/` directory was not present. The
understanding in this document is based on code, docs, configs, tests, and
saved `outputs/` artifacts, not direct inspection of local raw or merged data
files.

### Generated Surface Outputs

`generate_surface` currently supports four surface models:

- `svi`
  - fits SVI parameters slice by slice
- `sabr`
  - fits SABR parameters slice by slice
- `cubic`
  - stores filtered discrete IV slices for cubic-spline reconstruction
- `raw`
  - stores filtered discrete IV slices with no calibration

It supports three data ranges:

- `all`
  - scan all eligible minutes in input trade files
- `window`
  - generate surfaces inside a window around target UTC timestamps
- `excel`
  - load target timestamps from the news workbook, then run window-style
    generation

The main thesis-facing path is usually `svi` plus `excel` or a saved
`svi-excel` / `svi-all` run, depending on the experiment snapshot.

## 6. Workbook Semantics

The merge stage is not just ETL. It is the data-lineage layer that connects:

- raw option observations
- calibrated or reconstructed surface parameters
- news rows and embeddings
- fit-quality diagnostics
- training-candidate flags
- final model inputs

### `merged_vol.xlsx`

`merged_vol.xlsx` is already pair-based.

The training-facing sheet is:

```text
gan_input_ready
```

One usable row means:

```text
news row
  backward/current side -> current_surface_flat
  forward/future side   -> target_surface_flat
  same news text        -> hd_embedding / lp_embedding
```

The current mainline vol dataloader reads `gan_input_ready`, filters
`training_candidate_flag == 1` when present, sorts chronologically by
`news_timestamp_utc`, and creates train/validation splits without randomizing
the time order.

Important sheets:

- `news_surface_pair_audit`
  - one row per paired news example
  - contains current and target quality diagnostics
- `surface_side_detail`
  - one row per current/target side
  - useful for auditing one side independently
- `gan_input_ready`
  - pair-level training rows

### `merged_svi.xlsx`

`merged_svi.xlsx` is direction-level, not pair-level.

The executable SVI trainer does not use its `gan_input_ready` sheet as paired
training data.

The current SVI trainer reads:

```text
news_direction_audit
```

Then it:

1. filters rows with `training_candidate_flag == 1`
2. groups by `news_row_id`
3. requires both `backward` and `forward`
4. treats `backward` as current and `forward` as future
5. sorts paired samples chronologically by the current side timestamp

Important sheets:

- `news_direction_audit`
  - one row per `news_row_id x direction`
  - canonical audit table for SVI quality and lineage
- `svi_slice_detail`
  - one row per SVI slice
  - useful for maturity-slice fit diagnostics
- `gan_input_ready`
  - filtered direction-level export
  - not the current SVI trainer input

### Quality Labels and Training Flags

The merge code applies quality gates before setting
`training_candidate_flag = 1`.

Key checks include:

- surface/SVI parameters exist
- parameters are not placeholders
- there are raw points that passed precalibration filters
- exact slice point ratio is at least `0.5`
- weighted IV RMSE is at most `0.05`

Common labels or exclusion reasons include:

- `usable`
- `no_svi`
- `placeholder`
- `no_raw_points`
- `poor`
- `poor_fit`
- `low_exact_slice_ratio`
- `high_weighted_iv_rmse`

These are thesis-relevant quality controls, not incidental spreadsheet cleanup.

## 7. Fixed Surface Grid

Merged vol-surface workflows reconstruct surfaces on a fixed grid:

```text
strike_bins = 16
maturity_bins = 16
moneyness_min = 0.7
moneyness_max = 1.3
maturity_min_days = 7
maturity_max_days = 365
```

The shape is serialized as `[16, 16]`, with maturity as rows and strike as
columns.

Flattening order is:

```text
for maturity in maturity_days_grid:
    for strike in strike_grid:
        append surface[maturity, strike]
```

SVI reconstruction uses `quantlib.vol_surface.algo.svi_surface.SviVolSurface`.
The implementation interpolates SVI parameters across business days, evaluates
total variance in log-moneyness, and converts total variance back to implied
volatility.

## 8. Mainline Training Paths

### `vol-xlsx`

Command:

```bash
python scripts/train/main.py vol-xlsx --config configs/wgan/train_vol_xlsx.yaml
```

Purpose:

- train the mainline conditional WGAN-GP on `merged_vol.xlsx`
- forecast future reconstructed vol surfaces from current reconstructed vol
  surfaces and text embeddings

Model family:

- generator:
  - CNN surface encoder
  - text MLP encoder
  - latent noise
  - fusion MLP
  - residual surface delta
  - `softplus(current + delta) + 1e-4` positivity map
- critic:
  - sees stacked current and candidate future surfaces
  - also receives text embedding
  - outputs Wasserstein critic score

Training objective includes:

- WGAN adversarial loss
- L1 reconstruction loss
- calendar arbitrage penalty
- butterfly arbitrage penalty
- smoothness penalty
- optional delta-shrink penalty
- WGAN-GP gradient penalty for the critic

Automated selection can monitor metrics such as:

- `val_recon`
- `val_current_recon`
- `val_baseline_gap`
- `val_hybrid_score`
- `val_calendar`
- `val_butterfly`

### `vol-regression-xlsx`

Command:

```bash
python scripts/train/main.py vol-regression-xlsx --config configs/wgan/train_vol_regression_xlsx.yaml
```

Purpose:

- train a deterministic residual surface forecaster on the same paired
  `merged_vol.xlsx` input
- provide a non-adversarial baseline for surface forecasting

Model:

- CNN surface encoder
- text MLP encoder
- fusion MLP
- residual future surface prediction with positive output map

Losses:

- L1 reconstruction
- optional calendar/butterfly/smoothness penalties
- optional delta-shrink penalty

This path is useful when the thesis comparison needs to separate the benefit of
surface/text features from the complexity and instability of adversarial
training.

### `svi-xlsx`

Command:

```bash
python scripts/train/main.py svi-xlsx --config configs/wgan/train_svi_xlsx.yaml
```

Purpose:

- train a supervised SVI parameter forecaster
- predict future padded SVI parameters and future slice count from current SVI
  plus text

The trainer reads `merged_svi.xlsx` sheet `news_direction_audit`, not
`gan_input_ready`.

SVI tensorization:

- feature order:

```text
business_days, a, b, rho, m, sigma
```

- default `max_slices = 4`
- each side becomes a padded `[4, 6]` matrix
- padding rows are masked out
- current input vector contains:
  - flattened normalized current matrix
  - current mask
  - normalized current slice count
- future target contains:
  - flattened normalized future matrix
  - future mask
  - future count class

Normalization:

- fitted on the train partition only
- uses valid slices from both current and future matrices in the train samples
- ignores padded rows
- saved to `metrics/normalization_stats.json`

Model:

- current SVI encoder
- text encoder
- shared MLP trunk
- future-parameter regression head
- future-count classification head

Loss:

- masked Smooth L1 regression loss
- cross-entropy count loss
- total loss = regression + `count_loss_weight * count_loss`

This path is a representation-level comparison point against surface
forecasting.

### Legacy Daily-Surface WGAN

Older files:

- `scripts/train.py`
- `src/wgan_option/train.py`
- `src/wgan_option/utils/dataloader.py`

This path builds proxy daily surfaces from raw option data and trains:

```text
current_surface + text_embedding -> future_surface
```

It is useful as a baseline and historical context, but it is not the closest
current implementation to the minute-SVI-derived thesis workflow.

## 9. Standalone Model Families

The standalone model families generally train on `merged_vol.xlsx` and use the
same broad forecasting task:

```text
current_surface + text_embedding + optional noise -> future_surface
```

They are research comparison paths for architecture and conditioning choices.

### VolGAN

Entrypoint:

```bash
python scripts/volgan/main.py train --config configs/volgan/train_lp.yaml
```

Role:

- low-capacity MLP GAN baseline
- flattened surface input
- BCE adversarial objective
- smoothness penalties with gradient matching
- arbitrage checks mainly in post-processing

Use it as a simple baseline, not as the strongest current model.

### CNN WGAN

Entrypoint:

```bash
python scripts/cnn_wgan/main.py train --config configs/cnn_wgan/train_lp_gen128_disc128.yaml
```

Role:

- strongest simpler standalone baseline
- convolutional surface encoder
- global text MLP conditioning at fusion
- WGAN-GP objective
- calendar/butterfly/smoothness penalties
- uses log-IV delta reconstruction in the standalone implementation

This family is easier to explain than FiLM, StyleMod, Transformer, or
CrossAttention and is useful as the main architecture baseline.

### Transformer WGAN

Entrypoint:

```bash
python scripts/transformer_wgan/main.py train --config configs/transformer_wgan/train_lp.yaml
```

Role:

- tokenizes the surface grid
- uses text and noise tokens
- self-attention can model long-range strike/maturity relationships
- WGAN-GP on Transformer critic requires attention backend care because of
  double-backward gradient penalty

Saved results currently make this more of a negative control than a leading
model under the stored configurations.

### FiLM WGAN

Entrypoint:

```bash
python scripts/film_wgan/main.py train --config configs/film_wgan/train_lp_gen128_disc128.yaml
```

Role:

- currently the strongest saved surface-forecasting family
- text embedding modulates convolutional layers through FiLM
  feature-wise affine transforms
- tests the idea that news should change how the model reads the current
  surface, not only how it post-processes a global surface summary
- later experiments focus on short-end and near-ATM behavior

This is the most important standalone family for current thesis-facing
empirical results.

### StyleMod WGAN

Entrypoint:

```bash
python scripts/stylemod_wgan/main.py train --config configs/stylemod_wgan/train_lp_gen64_disc64_tuned.yaml
```

Role:

- StyleGAN-like global style vector
- style built from surface summary, text features, and noise
- modulated convolutions generate future-surface delta
- useful for testing whether a coherent global style is better than direct
  FiLM conditioning

Saved results show it is trainable but not currently competitive with CNN or
FiLM.

### CrossAttention WGAN

Entrypoint:

```bash
python scripts/crossattn_wgan/main.py train --config configs/crossattn_wgan/train_lp_gen128_disc128.yaml
```

Role:

- surface feature tokens attend to text-derived virtual tokens
- tests location-specific text conditioning
- more expressive than global fusion but more complex

Saved results show it is stable but weaker than the best CNN/FiLM runs in the
current snapshot.

## 10. Canonical Commands

Python prerequisite:

- use Python `>=3.10`, as required by `pyproject.toml`
- older Python versions can fail during import because the code uses modern
  type syntax

Install:

```bash
python -m pip install -e .
```

Generate SVI surfaces from news-aligned timestamps:

```bash
python scripts/generate_surface/main.py generate_surface \
  --device gpu \
  --model svi \
  --data_range excel \
  --config configs/surface_builder/svi/generate_surface-svi-excel.yaml
```

Merge generated outputs into workbooks:

```bash
python scripts/merge_file/merge_svi.py --input-dir data/processed/svi-excel/<run_ts>
python scripts/merge_file/merge_vol.py --input-dir data/processed/svi-excel/<run_ts>
python scripts/merge_file/merge_params.py --input-dir data/processed/svi-excel/<run_ts>
```

Train mainline merged-workbook models:

```bash
python scripts/train/main.py vol-xlsx --config configs/wgan/train_vol_xlsx.yaml
python scripts/train/main.py vol-regression-xlsx --config configs/wgan/train_vol_regression_xlsx.yaml
python scripts/train/main.py svi-xlsx --config configs/wgan/train_svi_xlsx.yaml
```

Dry run or inspect resolved configs:

```bash
python scripts/train/main.py vol-xlsx --config configs/wgan/train_vol_xlsx.yaml --print-config
python scripts/train/main.py svi-xlsx --config configs/wgan/train_svi_xlsx.yaml --dry-run
```

Run mainline result generation:

```bash
python scripts/generate_result/main.py vol \
  --config configs/wgan/train_vol_xlsx.yaml \
  --set split=val \
  --set selection_mode=first_n \
  --set limit=5

python scripts/generate_result/main.py svi \
  --config configs/wgan/train_svi_xlsx.yaml \
  --set split=val \
  --set selection_mode=first_n \
  --set limit=5
```

Run error analysis:

```bash
python scripts/analyze_error/main.py vol --config configs/analyze_error/vol.yaml
python scripts/analyze_error/main.py svi --config configs/analyze_error/svi.yaml
```

Train standalone families:

```bash
python scripts/volgan/main.py train --config configs/volgan/train_lp.yaml
python scripts/cnn_wgan/main.py train --config configs/cnn_wgan/train_lp_gen128_disc128.yaml
python scripts/transformer_wgan/main.py train --config configs/transformer_wgan/train_lp.yaml
python scripts/film_wgan/main.py train --config configs/film_wgan/train_lp_gen128_disc128.yaml
python scripts/stylemod_wgan/main.py train --config configs/stylemod_wgan/train_lp_gen64_disc64_tuned.yaml
python scripts/crossattn_wgan/main.py train --config configs/crossattn_wgan/train_lp_gen128_disc128.yaml
```

## 11. Metrics and Experiment Interpretation

Do not mix metrics across generations of experiments without qualification.

Important metric families:

- `val_recon`
  - older reconstruction metric
  - used by legacy and some mainline WGAN runs
- `val_current_recon`
  - persistence/current-surface baseline error
- `val_baseline_gap`
  - model error minus current-surface baseline error
  - negative is better than persistence
- `val_mae_gap_vs_current`
  - common broad paired-surface gap metric in standalone runs
  - negative is better
- `val_short_atm_mae_gap_vs_current`
  - short-end near-ATM weighted metric
- `val_atm_short_pure_mae_gap_vs_current`
  - stricter pure short-ATM metric used by later FiLM experiments
- `val_hybrid_score`
  - combines reconstruction with baseline-gap penalty in some runs

Use `docs/training_loss_curves.md` for metric definitions and
`docs/training_results_comparison.md` for saved-run interpretation.

## 12. Current Empirical Takeaways

The current saved-output summary in `docs/training_results_comparison.md`
indicates:

- FiLM WGAN is currently the strongest saved surface-forecasting family.
- The best broad paired-surface gap is reported for
  `outputs/training/film_wgan/svi-excel/20260417_180233`.
- The best pure short-ATM checkpoint result is reported for
  `outputs/training/film_wgan/svi-excel/20260417_131244`.
- CNN WGAN is the strongest simpler baseline, especially
  `outputs/training/cnn_wgan/svi-excel/20260414_123032`.
- CrossAttention WGAN is stable but weaker than the best CNN/FiLM runs.
- StyleMod WGAN is trainable but not currently competitive.
- VolGAN is useful as a simple low-capacity reference.
- Transformer WGAN is a useful negative-control family under the saved
  configurations.
- Legacy raw/svi WGAN runs use different data semantics and metric schemas, so
  they should not be directly ranked with later standalone gap-based runs.

Important caveat:

- Early `vol-xlsx` run reviews showed strong no-text/current-surface baselines.
  Any thesis claim about text usefulness must compare against those baselines
  honestly.

## 13. Artifacts and Reproducibility

Typical training run directories contain:

```text
outputs/training/<family>/<dataset>/<run_ts>/
  checkpoints/
  metrics/
  samples/
  run.log
```

Important artifacts:

- `metrics/training_resolved_config.yaml`
- `metrics/training_metrics.csv`
- `metrics/training_metrics.json`
- `metrics/best_checkpoint.json`
- `metrics/loss_curves.png`
- `generate_result/<checkpoint_name>/summary.csv`
- `generate_result/<checkpoint_name>/samples/*.json`
- `generate_result/<checkpoint_name>/plots/*.png`

Some FiLM runs also write ATM-vol time-series artifacts under:

```text
generate_result/atm_vol/
```

For thesis work, prefer citing runs that have:

- resolved config snapshot
- best checkpoint metadata
- metrics CSV/JSON
- loss curves
- generated sample summaries
- sample-level plots or JSON payloads

## 14. Tests Worth Knowing

Targeted tests that protect important semantics:

```bash
python -m unittest discover -s tests/test_scripts -p 'test_merge_file.py'
python -m unittest discover -s tests/test_scripts -p 'test_merge_vol.py'
python -m unittest discover -s tests/test_scripts -p 'test_train_xlsx.py'
python -m unittest discover -s tests/test_scripts -p 'test_generate_surface_config_jobs.py'
python -m unittest discover -s tests/test_vol -p 'test_svi_calibration.py'
```

What they cover:

- merge argument parsing and workbook layout
- direction-level SVI workbook semantics
- paired vol workbook semantics
- quality labels and training-candidate filtering
- chronological train/validation split
- `hd`, `lp`, and `concat` text embedding modes
- SVI train-only normalization
- config overrides and training path derivation
- generated-surface config parsing

For docs-only changes, no code tests are usually required. For runtime changes,
run the relevant targeted tests and consider the repository-specific verification
skill if available.

## 15. Reference Literature Folder

The `ref/` directory contains relevant volatility and GAN references:

- `VolGAN  A Generative Model for Arbitrage-Free Implied Volatility Surfaces.pdf`
- `GAN-Enhanced_Implied_Volatility_Surface_Reconstruction_for_Option_Pricing_Error_Mitigation.pdf`
- `Deep learning volatility  a deep neural network perspective on pricing and calibration in  rough  volatility models.pdf`

In the environment used to write this onboarding note, full PDF text extraction
tools were not installed. Only metadata/title-level information was inspected.
Do not treat this note as a literature review of those PDFs.

High-level positioning based on metadata and filenames:

- VolGAN is a reference for generative implied-volatility-surface modeling and
  arbitrage-aware constraints.
- GAN-enhanced IV surface reconstruction is a reference for using GANs to
  reconstruct implied volatility surfaces and reduce option-pricing error.
- Deep learning volatility is a reference for neural pricing and calibration in
  volatility models, including rough-volatility context.

If thesis text depends on exact claims from these papers, install a PDF text
tool or read the PDFs directly before citing details.

## 16. Known Caveats and Risks

### Local Data Availability

At the time this note was written, this workspace did not contain a local
`data/` directory. The code and docs reference paths such as:

```text
data/raw/option_data
data/raw/text_embedding/news_with_openai_embeddings_large.xlsx
data/processed/svi-excel/<run_ts>/merged_vol.xlsx
data/processed/svi-all/20260330-01/merged_vol.xlsx
```

Do not claim that these data files were inspected unless they are present in the
active environment and have actually been read.

### Output Naming Has Evolved

Older docs and configs may mention `minute_svi_*` artifacts or
`data/processed/svi/<run_ts>`. Current generation prefers:

```text
data/processed/<model>-<data_range>/<run_ts>/
surface-<model>-<data_range>.json
surface-<model>-<data_range>-precalib-points.csv
```

Merge code supports both naming schemes. When updating docs or scripts, be
explicit about which generation of artifacts a command expects.

### Packaging Note

The current `pyproject.toml` package include list explicitly includes several
standalone packages, but a review noticed that `film_wgan` and `crossattn_wgan`
were not explicitly listed there while their scripts import them through the
repo root and `src/` path setup.

This may be fine for editable/script execution, but it is a reproducibility risk
for clean packaging or non-editable installs. Verify before relying on those
modules in a fresh environment.

### Metrics Are Not One Scoreboard

Do not build one global ranking table unless it labels metric families clearly.
The old mainline WGAN runs, gap-based standalone runs, and short-ATM FiLM runs
do not all optimize or checkpoint the same metric.

### SVI Workbook Is Not Paired

This is the most common implementation mistake:

- `merged_vol.xlsx` is paired.
- `merged_svi.xlsx` is direction-level.
- SVI training pairs rows at runtime from `news_direction_audit`.

Do not train SVI directly from `merged_svi.xlsx` `gan_input_ready` unless the
schema and trainer are intentionally changed.

### Quality Filters Are Research Semantics

Columns such as `weighted_iv_rmse`, `exact_slice_point_ratio`,
`placeholder_flag`, and `training_candidate_flag` are part of the experiment
definition. Do not remove or bypass them simply to make a loader accept more
rows.

## 17. Practical Guidance for Future Agents

When modifying this repository:

1. Identify whether the change affects thesis design, executable behavior, or
   both.
2. Preserve `backward/current` and `forward/future` semantics in names,
   comments, workbook fields, metrics, and docs.
3. For merged vol changes, start with `src/wgan_option/utils/merged_xlsx_*`,
   `src/wgan_option/train_vol_xlsx.py`, and `src/wgan_option/models/gan_model.py`.
4. For SVI changes, start with `src/wgan_option/utils/merged_xlsx_*`,
   `src/wgan_option/train_svi_xlsx.py`, and
   `src/wgan_option/models/svi_regressor.py`.
5. For generation changes, follow the `scripts/generate_surface` and
   `src/wgan_option/surface_generation` structure.
6. For merge schema changes, update tests that assert workbook semantics.
7. For model comparisons, keep persistence/no-text baselines visible.
8. For thesis-facing conclusions, cite saved artifacts and metric definitions,
   not only final-epoch losses.

The strongest contributions in this codebase are usually those that make the
experiment more reproducible, make representation choices more explicit, or make
the bridge between design docs and runnable code clearer.
