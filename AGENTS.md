# AGENTS Guide

This file is a practical project map for coding agents working in `wgan_option`.
It is written for a **research codebase**, not a generic product repo: this project is being developed as part of a **PhD thesis chapter**, and many changes should be judged by whether they improve experimental clarity, data lineage, and reproducibility, not just software neatness.

Use this file together with the design docs under [docs](/Users/haobincui/Documents/wgan_option/docs):

- [input_svi.md](/Users/haobincui/Documents/wgan_option/docs/input_svi.md)
- [input_vol.md](/Users/haobincui/Documents/wgan_option/docs/input_vol.md)
- [vol_surface_gan_architecture.md](/Users/haobincui/Documents/wgan_option/docs/vol_surface_gan_architecture.md)

Those docs describe the **research framing and target dataset/model design**.
This `AGENTS.md` describes the **current implementation state and project conventions**.
When docs and code differ, do not assume one is wrong automatically:

- treat `docs/` as the intended thesis design / experiment specification
- treat `src/` and `scripts/` as the current executable reality
- call out mismatches explicitly instead of silently "fixing" one side

## 1. Project Purpose

This repo has two closely related tracks:

- **Vol-surface / SVI generation**
  - Build daily surfaces or minute-level SVI fits from raw option data.
  - Main code lives under [scripts/generate_surface](/Users/haobincui/Documents/wgan_option/scripts/generate_surface) and [src/quantlib](/Users/haobincui/Documents/wgan_option/src/quantlib).
- **Model training**
  - Train models that use text embeddings plus either vol surfaces or SVI parameters.
  - Main code lives under [src/wgan_option](/Users/haobincui/Documents/wgan_option/src/wgan_option) and [scripts/train](/Users/haobincui/Documents/wgan_option/scripts/train).

From the thesis perspective, the chapter's high-level research theme is:

- use **text embeddings + option-surface representations** to study/predict the evolution of bond-option volatility structures
- compare different surface representations:
  - daily proxy vol-like surfaces
  - SVI parameter representations
  - SVI-reconstructed vol surfaces
- keep no-arbitrage and fit-quality concerns explicit rather than treating the surface as an arbitrary image tensor

There are now three training modes in the repo:

- **Legacy training**
  - [scripts/train.py](/Users/haobincui/Documents/wgan_option/scripts/train.py)
  - Uses [train_default.yaml](/Users/haobincui/Documents/wgan_option/configs/wgan/train_default.yaml)
  - Trains on daily proxy surfaces built directly from raw option trades.
- **Merged vol-surface training**
  - [train_vol.py](/Users/haobincui/Documents/wgan_option/scripts/train/train_vol.py)
  - Trains WGAN-GP on `merged_vol.xlsx`.
- **Merged SVI training**
  - [train_svi.py](/Users/haobincui/Documents/wgan_option/scripts/train/train_svi.py)
  - Trains a supervised MLP regressor on paired SVI rows from `merged_svi.xlsx`.

This means the repo currently contains both:

- **published / chapter-facing workflows**
  - the ideas described in `docs/`
- **transitional engineering workflows**
  - legacy daily training
  - merged xlsx training paths that help test whether the research data representation is usable

## 2. Repo Map

### Core packages

- [src/wgan_option](/Users/haobincui/Documents/wgan_option/src/wgan_option)
  - Training code.
  - Contains config, dataloaders, GAN model, new SVI regressor, and trainer wrappers.
- [src/quantlib](/Users/haobincui/Documents/wgan_option/src/quantlib)
  - Calendar logic, daycount, implied vol / SVI / surface utilities.
  - This is the numerical core behind surface generation and SVI reconstruction.
- [src/market_data](/Users/haobincui/Documents/wgan_option/src/market_data)
  - Market-data contracts and DTO helpers.
  - Less central than `quantlib` and `wgan_option` for the current workflows.

### Script layers

- [scripts/generate_surface](/Users/haobincui/Documents/wgan_option/scripts/generate_surface)
  - Unified entrypoint for daily-surface and minute-SVI generation.
  - Structured as:
    - `main.py`: top-level CLI
    - `common/`: shared config/time-window logic
    - `surface_cpu/` and `surface_gpu/`: execution backends
- [scripts/merge_file](/Users/haobincui/Documents/wgan_option/scripts/merge_file)
  - Merge generated SVI outputs with the news embedding workbook.
  - Current scripts:
    - [merge_svi.py](/Users/haobincui/Documents/wgan_option/scripts/merge_file/merge_svi.py)
    - [merge_vol.py](/Users/haobincui/Documents/wgan_option/scripts/merge_file/merge_vol.py)
- [scripts/train](/Users/haobincui/Documents/wgan_option/scripts/train)
  - New merged-xlsx training entrypoints.
  - Structured like `generate_surface`:
    - [main.py](/Users/haobincui/Documents/wgan_option/scripts/train/main.py): unified CLI
    - [common.py](/Users/haobincui/Documents/wgan_option/scripts/train/common.py): shared CLI/config parsing
    - [train_vol.py](/Users/haobincui/Documents/wgan_option/scripts/train/train_vol.py): vol-surface training wrapper
    - [train_svi.py](/Users/haobincui/Documents/wgan_option/scripts/train/train_svi.py): SVI training wrapper

### Configs

- [configs/surface_builder](/Users/haobincui/Documents/wgan_option/configs/surface_builder)
  - Config-driven surface generation jobs.
- [configs/wgan](/Users/haobincui/Documents/wgan_option/configs/wgan)
  - `train_default.yaml`: legacy daily-surface training
  - `train_vol_xlsx.yaml`: merged vol training
  - `train_svi_xlsx.yaml`: merged SVI training

### Tests

- [tests/test_scripts](/Users/haobincui/Documents/wgan_option/tests/test_scripts)
  - End-to-end and CLI-style script tests.
- [tests/test_vol](/Users/haobincui/Documents/wgan_option/tests/test_vol)
  - SVI / vol interpolation tests.
- [tests/test_quantlib](/Users/haobincui/Documents/wgan_option/tests/test_quantlib)
  - Lower-level quantlib checks.

## 3. Data Flow

The practical workflow is:

1. **Raw inputs**
   - option trades under `data/raw/option_data`
   - news embeddings workbook under `data/raw/text_embedding/news_with_openai_embeddings_large.xlsx`
2. **Minute SVI generation**
   - produces a result directory like `data/processed_excel_20260330-01`
   - key files:
     - `minute_svi_params.json`
     - `minute_svi_precalib_points.csv`
     - `minute_svi_params.log`
3. **Merge**
   - [merge_svi.py](/Users/haobincui/Documents/wgan_option/scripts/merge_file/merge_svi.py) produces `merged_svi.xlsx`
   - [merge_vol.py](/Users/haobincui/Documents/wgan_option/scripts/merge_file/merge_vol.py) produces `merged_vol.xlsx`
4. **Training**
   - `merged_vol.xlsx` feeds vol-surface WGAN training
   - `merged_svi.xlsx` feeds paired SVI regression training

For thesis work, this flow is not just ETL. It is the **experimental data lineage**.
When modifying this repo, preserve the ability to answer:

- which raw source produced a given training example
- whether the example came from `backward/current` or `forward/future`
- whether SVI was directly consumed or first reconstructed into a vol surface
- whether a sample passed a quality filter or was only included as an audit row

## 4. Important Semantic Rules

These are the most important repo-specific semantics to keep straight.

### Minute SVI direction semantics

- `backward` corresponds to the news row's original `timestamp_utc`
- `forward` corresponds to `timestamp_utc + 5 minutes`

### `merge_vol.py` semantics

- Produces **paired** rows.
- One row in `gan_input_ready` means:
  - `current_surface` = `backward`
  - `target_surface` = `forward`
- This is already in the training-ready shape for vol-surface modeling.

### `merge_svi.py` semantics

- Produces **direction-level** rows, not paired rows.
- `gan_input_ready` there is still one row per direction.
- `train_svi.py` does **not** train from `merge_svi`'s `gan_input_ready`.
- Instead, it reads `news_direction_audit` and pairs `backward` + `forward` on the fly using `news_row_id`.

This is an important thesis-code distinction:

- `docs/input_svi.md` is mainly an **audit and representation-design** document
- the current executable `train_svi.py` is a **paired forecasting implementation**
- do not collapse those two roles conceptually when changing the project

### Text embedding semantics

The repo now supports three text modes for merged-xlsx training:

- `hd`
- `lp`
- `concat`

Defaults remain aligned with the old code:

- default mode is `hd`
- if `concat` is used, embedding dimension is inferred automatically

When working on experiments, remember that text handling is part of the research design:

- `HD_embedding` is still the repo default because it matches the earlier training code and architecture note
- `LP_embedding` and `concat` exist to support ablation and representation-comparison experiments

## 5. Current Training Architecture

### Legacy daily-surface path

- Dataloader: [dataloader.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/utils/dataloader.py)
- Trainer: [train.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/train.py)
- Model: [gan_model.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/models/gan_model.py)

This path builds daily proxy surfaces from raw trades and trains:

- `current_surface + text_embedding -> future_surface`

This path matches the architecture discussion in [vol_surface_gan_architecture.md](/Users/haobincui/Documents/wgan_option/docs/vol_surface_gan_architecture.md), but it is still based on a **proxy surface**, not the newer SVI-derived chapter datasets.

### Merged vol path

- Loader: [merged_xlsx.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/utils/merged_xlsx.py)
- Trainer wrapper: [train_vol_xlsx.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/train_vol_xlsx.py)
- Reuses the existing WGAN-GP stack in [gan_model.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/models/gan_model.py)

This path reads:

- `current_surface_flat`
- `target_surface_flat`
- `hd_embedding` / `lp_embedding`

from `merged_vol.xlsx`.

Conceptually, this is the closest current implementation to the target chapter workflow described in [input_vol.md](/Users/haobincui/Documents/wgan_option/docs/input_vol.md):

- `backward` SVI -> reconstructed current vol surface
- `forward` SVI -> reconstructed target vol surface

### Merged SVI path

- Loader: [merged_xlsx.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/utils/merged_xlsx.py)
- Trainer wrapper: [train_svi_xlsx.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/train_svi_xlsx.py)
- Model: [svi_regressor.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/models/svi_regressor.py)

This path:

- pairs usable `backward` and `forward` rows from `news_direction_audit`
- pads SVI slices to `max_slices = 4`
- uses per-slice features:
  - `business_days`
  - `a`
  - `b`
  - `rho`
  - `m`
  - `sigma`
- predicts future padded SVI parameters plus future slice count

Conceptually, this is a pragmatic implementation of the research direction described in [input_svi.md](/Users/haobincui/Documents/wgan_option/docs/input_svi.md), but note:

- the docs are still more audit-oriented
- the code has already moved to a paired training implementation

## 6. Current Output Locations

Common output areas:

- [outputs/training/vol_xlsx](/Users/haobincui/Documents/wgan_option/outputs/training/vol_xlsx)
  - merged vol training artifacts
- [outputs/training/svi_xlsx](/Users/haobincui/Documents/wgan_option/outputs/training/svi_xlsx)
  - merged SVI training artifacts
- [outputs/checkpoints](/Users/haobincui/Documents/wgan_option/outputs/checkpoints)
  - legacy training artifacts

Current example merged data:

- [merged_svi.xlsx](/Users/haobincui/Documents/wgan_option/data/processed_excel_20260330-01/merged_svi.xlsx)
- [merged_vol.xlsx](/Users/haobincui/Documents/wgan_option/data/processed_excel_20260330-01/merged_vol.xlsx)

These example outputs are not just fixtures; they are currently the main concrete artifacts tying the chapter design to runnable code.

## 7. Commands You Will Actually Use

### Install

```bash
python -m pip install -e .
```

### Generate minute SVI from config

```bash
python scripts/generate_surface/main.py minute-svi-excel --device gpu --config configs/surface_builder/minute-svi-excel.yaml
```

### Build merged training workbooks

```bash
python scripts/merge_file/merge_svi.py --input-dir data/processed_excel_20260330-01
python scripts/merge_file/merge_vol.py --input-dir data/processed_excel_20260330-01
```

### Train from merged xlsx

```bash
python scripts/train/main.py vol-xlsx --config configs/wgan/train_vol_xlsx.yaml
python scripts/train/main.py svi-xlsx --config configs/wgan/train_svi_xlsx.yaml
```

### Dry runs

```bash
python scripts/train/train_vol.py --config configs/wgan/train_vol_xlsx.yaml --dry-run
python scripts/train/train_svi.py --config configs/wgan/train_svi_xlsx.yaml --dry-run
```

### Targeted test runs

```bash
python -m unittest discover -s tests/test_scripts -p 'test_merge_file.py'
python -m unittest discover -s tests/test_scripts -p 'test_merge_vol.py'
python -m unittest discover -s tests/test_scripts -p 'test_train_xlsx.py'
python -m unittest discover -s tests/test_vol -p 'test_svi_calibration.py'
```

## 8. Things That Are Easy To Get Wrong

- Do not assume `merge_svi.py` already produces `current_svi -> future_svi` rows. It does not.
- Do not train SVI from `merged_svi.xlsx` sheet `gan_input_ready`; pair from `news_direction_audit`.
- Do not assume the root [scripts/train.py](/Users/haobincui/Documents/wgan_option/scripts/train.py) knows about merged xlsx workflows. It is still the legacy entrypoint.
- Do not overwrite `outputs/checkpoints` when working on merged-xlsx training. Use the dedicated `outputs/training/vol_xlsx` and `outputs/training/svi_xlsx` trees.
- Keep the `backward/current` and `forward/future` distinction explicit in any new feature touching merged SVI or merged vol data.
- The repo uses `apply_patch`-style manual edits well; avoid ad hoc file rewriting patterns when making small focused changes.
- Do not treat `docs/input_svi.md` and `docs/input_vol.md` as mere documentation cosmetics. They are part of the thesis design record.
- Do not silently simplify away audit columns or quality flags unless the change is intentionally narrowing the research question.
- When changing data schemas, think about downstream experimental comparability, not just whether the code still runs.

## 9. If You Need To Extend The Project

Use these extension points first:

- Add new merged-data parsing logic to [merged_xlsx.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/utils/merged_xlsx.py)
- Add new training wrappers in [src/wgan_option](/Users/haobincui/Documents/wgan_option/src/wgan_option)
- Keep script entrypoints thin under [scripts/train](/Users/haobincui/Documents/wgan_option/scripts/train)
- For generation workflows, follow the [scripts/generate_surface](/Users/haobincui/Documents/wgan_option/scripts/generate_surface) structure:
  - common logic in `common/`
  - thin backend-specific entrypoints
  - one top-level `main.py`

For thesis-friendly changes, prefer this order:

1. update or check the corresponding design doc in `docs/`
2. preserve raw -> generated -> merged -> trainable data lineage
3. add tests that protect semantic meaning, not only importability
4. only then optimize or refactor internal code shape

## 10. Practical Summary

If you only remember five things, remember these:

1. `src/quantlib` builds surfaces and SVI; `src/wgan_option` trains models.
2. `scripts/generate_surface` creates `minute_svi_params.json` and `minute_svi_precalib_points.csv`.
3. `scripts/merge_file` turns those outputs plus news embeddings into `merged_svi.xlsx` and `merged_vol.xlsx`.
4. `merged_vol.xlsx` is already pair-based; `merged_svi.xlsx` is not, and SVI pairing happens during training.
5. The new preferred training CLI is [scripts/train/main.py](/Users/haobincui/Documents/wgan_option/scripts/train/main.py), not just the legacy root [scripts/train.py](/Users/haobincui/Documents/wgan_option/scripts/train.py).

## 11. What Good Contributions Look Like Here

For this project, a strong change usually has these properties:

- it makes the chapter experiment more reproducible
- it makes the data representation more explicit
- it preserves or clarifies `backward/current` vs `forward/future`
- it improves the bridge between design docs and runnable code
- it keeps room for ablation or comparison across representations instead of hard-coding one research choice too early
