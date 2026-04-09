# Bond Option Vol Surface Forecasting with Text Embeddings and Arbitrage-Aware Conditional WGAN

This document describes the intended thesis-facing architecture and the current executable implementation for vol-surface forecasting in this repo.

It should be read together with:

- `docs/current_executable_workflows.md`
- `docs/input_vol.md`
- `docs/input_svi.md`
- `docs/svi_regressor_architecture.md`
- `docs/gan_model_detailed_architecture.md`
- `docs/training_loss_curves.md`
- `docs/reduce_lr_on_plateau.md`

## 1. Research Objective

The core forecasting target is:

\[
(\text{CurrentSurface}_t,\ \text{TextEmbedding}_t) \rightarrow \text{FutureSurface}_{t+h}
\]

where:

- `CurrentSurface_t` is the option volatility surface observed at the current time
- `TextEmbedding_t` is the embedding of the associated news item
- `FutureSurface_{t+h}` is the future surface at the forecast horizon

In the current thesis direction, the preferred executable representation is:

- build minute-level SVI parameters from raw option data
- reconstruct current and future vol surfaces from those SVI slices
- train the conditional WGAN on paired current/future surfaces in `merged_vol.xlsx`

The repo also still supports the older daily proxy-surface workflow for comparison and backwards compatibility.

## 2. Executable Training Paths in the Repo

For a command-oriented walkthrough of the current runnable pipeline, see:

- `docs/current_executable_workflows.md`

There are now three training modes, and only two of them are directly relevant to this architecture note.

### 2.1 Legacy daily-surface WGAN

Files:

- `scripts/train.py`
- `src/wgan_option/train.py`
- `src/wgan_option/utils/dataloader.py`

This path:

- builds proxy daily surfaces directly from raw option trade files
- aggregates text embeddings at daily frequency
- trains a conditional WGAN on the daily proxy representation

This remains useful as a baseline or compatibility path, but it is not the closest current implementation to the thesis chapter's minute-SVI-based surface pipeline.

### 2.2 Merged vol-surface WGAN

Files:

- `scripts/train/main.py`
- `scripts/train/train_vol.py`
- `src/wgan_option/train_vol_xlsx.py`
- `src/wgan_option/utils/merged_xlsx.py`
- `src/wgan_option/models/gan_model.py`

This is the current executable path that most closely matches the intended chapter workflow.

It trains on `merged_vol.xlsx`, where each row already contains a pair:

- `current_surface_flat`: backward/current surface
- `target_surface_flat`: forward/future surface
- `hd_embedding` or `lp_embedding`: text embedding for the news row

The effective supervised target is:

\[
(\text{current reconstructed vol surface},\ \text{text embedding}) \rightarrow \text{future reconstructed vol surface}
\]

Current preferred config entrypoint:

- `python scripts/train/main.py vol-xlsx --config configs/wgan/train_vol_xlsx.yaml`

### 2.3 Merged SVI supervised regressor

Files:

- `scripts/train/main.py`
- `scripts/train/train_svi.py`
- `src/wgan_option/train_svi_xlsx.py`
- `src/wgan_option/models/svi_regressor.py`

This path does not use the WGAN architecture, but it is important context for the overall project.

It predicts:

- future padded SVI parameters
- future slice count

from:

- current padded SVI parameters
- text embedding

This path is complementary to the surface WGAN path and supports representation comparison at the thesis level.

For the detailed architecture of this SVI forecasting path, see:

- `docs/svi_regressor_architecture.md`

## 3. Data Lineage for the Preferred Vol Workflow

The corresponding config and workflow overview is summarized in:

- `docs/current_executable_workflows.md`

The practical data flow for the merged vol-surface path is:

1. raw option trades
2. minute SVI calibration
3. merge with news embeddings
4. paired current/future surface training

### 3.1 Minute SVI generation

Main entrypoint:

- `scripts/generate_surface/main.py`

Important subcommand:

- `minute-svi-excel`

Preferred config file:

- `configs/surface_builder/svi/minute-svi-excel.yaml`

Typical artifacts:

- `minute_svi_params.json`
- `minute_svi_precalib_points.csv`

### 3.2 Merge into paired vol workbook

Main file:

- `scripts/merge_file/merge_vol.py`

This script reconstructs current and target vol surfaces from SVI slices and writes `merged_vol.xlsx`.

Important semantic rule:

- `backward` corresponds to current
- `forward` corresponds to future

So in the training-ready sheet:

- `current_surface_flat` is the backward/current surface
- `target_surface_flat` is the forward/future surface

This distinction should remain explicit in any future extension to the pipeline.

### 3.3 Workbook training input

Training uses the `gan_input_ready` sheet from `merged_vol.xlsx`.

Preferred training config:

- `configs/wgan/train_vol_xlsx.yaml`

The loader performs:

- embedding selection by mode: `hd`, `lp`, or `concat`
- chronological train/validation split
- reshaping of flat surface arrays into `[B, 1, H, W]`

The chronological split matters because validation then behaves like a time-ordered holdout instead of a random split.

## 4. Model Architecture

The current WGAN vol model lives in:

- `src/wgan_option/models/generator.py`
- `src/wgan_option/models/discriminator.py`
- `src/wgan_option/models/gan_model.py`

### 4.1 Generator

Inputs:

- current surface: `[B, 1, H, W]`
- text embedding: `[B, E]`
- noise: `[B, Z]`

Design:

1. encode the current surface with a CNN-style encoder
2. encode the text embedding with an MLP
3. fuse those features with latent noise
4. map the fused representation to a surface delta
5. apply `softplus(current + delta)` so the predicted future surface remains positive

Conceptually, the generator learns a conditional surface-transition map rather than unconditional surface synthesis.

### 4.2 Critic

Inputs:

- candidate future surface, either real or generated
- current surface
- text embedding

Design:

1. concatenate current and future surfaces along the channel dimension
2. encode them with the critic network
3. fuse with text features
4. output a Wasserstein score

The critic therefore evaluates whether the proposed future surface looks plausible given both the current surface and the associated text signal.

## 5. Training Objective

The generator objective in the current implementation is:

\[
\mathcal{L}_G
=
\mathcal{L}_{adv}
+ \lambda_{recon}\mathcal{L}_{recon}
+ \lambda_{cal}\mathcal{L}_{calendar}
+ \lambda_{bfly}\mathcal{L}_{butterfly}
+ \lambda_{smooth}\mathcal{L}_{smooth}
\]

Only the enabled constraint terms are included in the actual loss.

### 5.1 Adversarial term

WGAN generator term:

\[
\mathcal{L}_{adv} = -\mathbb{E}[D(fake)]
\]

This pushes the generator toward producing surfaces that the critic scores more like real future surfaces.

### 5.2 Reconstruction term

The implementation uses L1 reconstruction:

\[
\mathcal{L}_{recon} = L1(fake\_future,\ real\_future)
\]

This keeps the model tied to the forecasting target instead of optimizing only for adversarial realism.

### 5.3 Calendar arbitrage penalty

The calendar constraint is based on total variance monotonicity:

\[
\partial_{\tau}(\sigma^2 \tau) \ge 0
\]

The code computes total variance on adjacent maturity bins and penalizes negative increments.

### 5.4 Butterfly arbitrage penalty

The butterfly constraint targets strike-direction convexity:

\[
\frac{\partial^2 C}{\partial K^2} \ge 0
\]

The code converts generated implied vol into Black call prices, takes a second finite difference along strike, and penalizes the negative part.

### 5.5 Smoothness penalty

The smoothness penalty adds squared first differences along:

- maturity direction
- strike direction

This discourages visually noisy local oscillations in the generated surface.

### 5.6 Critic objective

The critic uses the standard WGAN-GP form:

\[
\mathcal{L}_D = \mathbb{E}[D(fake)] - \mathbb{E}[D(real)] + \lambda_{gp}\mathcal{L}_{gp}
\]

where the gradient penalty encourages approximate 1-Lipschitz behavior.

## 6. Current Runtime Training Behavior

The current merged vol WGAN implementation includes several experiment-management features beyond the earlier version of this project.

### 6.1 Metrics and plots

During training, the code writes:

- `training_metrics.json`
- `training_metrics.csv`
- `loss_curves.png`

The primary WGAN monitoring metric is:

- `val_recon`

For the definitions of all curve variables, see:

- `docs/training_loss_curves.md`

### 6.2 Best checkpoint

When validation is available, the training loop now saves:

- `generator_best.pt`
- `discriminator_best.pt`
- `best_checkpoint.json`

Best-model selection is based on the lowest:

\[
val\_{recon}
\]

This is more appropriate for forecasting than selecting the last epoch by default.

### 6.3 Early stopping

The current config surface supports optional early stopping:

- `use_early_stopping`
- `early_stopping_patience`
- `early_stopping_min_delta`

Early stopping is disabled by default, but when enabled it also monitors `val_recon`.

### 6.4 `ReduceLROnPlateau`

The current code also supports optional validation-driven learning-rate decay:

- `use_reduce_lr_on_plateau`
- `reduce_lr_factor`
- `reduce_lr_patience`
- `reduce_lr_min_lr`

For the WGAN path:

- generator and discriminator both use `ReduceLROnPlateau`
- both monitor `val_recon`

This is intended for late-stage refinement when validation improvement has flattened out.

For the exact scheduler behavior and formulas, see:

- `docs/reduce_lr_on_plateau.md`

### 6.5 Downstream inspection tooling

After training, the repo's current supported inspection layer is:

- `scripts/generate_result/main.py`
- `scripts/analyze_error/main.py`

These scripts sit downstream of the merged training workflows and are part of the practical experiment loop, even though they are not themselves part of the WGAN architecture.

## 7. Why the Merged Vol Path Matters

The merged vol path is important because it is the clearest bridge between thesis design and runnable code.

Relative to the legacy daily path, it has three advantages:

1. it keeps the current/future pairing explicit at the row level
2. it is built from minute-level SVI-derived information rather than only daily proxy surfaces
3. it preserves more audit information about fit quality and sample usability in the workbook lineage

This makes it a better experimental base for studying whether text embeddings help forecast option-surface evolution.

## 8. Current Constraints and Research Caveats

The repo is executable and experiment-ready, but several caveats remain important.

### 8.1 Representation caveat

The merged vol workbook stores reconstructed surfaces, not raw SVI parameters.
That is ideal for direct surface forecasting, but it means any forecasting error mixes:

- true future uncertainty
- SVI fit quality
- reconstruction/interpolation effects

### 8.2 Data-quality caveat

Training quality still depends on the upstream minute-SVI calibration quality.

Fields such as:

- `weighted_iv_rmse`
- `exact_slice_point_ratio`
- `training_candidate_flag`

should remain part of experiment interpretation, not just preprocessing.

### 8.3 Experiment-selection caveat

Because this is a time-ordered forecasting problem, model comparison should emphasize:

- validation reconstruction quality
- structural quality of predicted surfaces
- reproducible run artifacts

This comparison set now includes:

- the text-conditioned merged-vol WGAN
- the merged-SVI supervised regressor

and not only final-epoch adversarial losses.

## 9. Suggested Experiment Priorities

For the current architecture, the most useful next experiment directions are:

1. compare text modes: `hd`, `lp`, and `concat`
2. compare merged vol forecasting against merged SVI forecasting
3. use best-checkpoint selection instead of last-epoch selection
5. evaluate whether validation-driven LR reduction improves late-stage `val_recon`

## 10. Key Code Locations

- config surface: `src/wgan_option/config.py`
- merged vol dataloader: `src/wgan_option/utils/merged_xlsx.py`
- vol trainer wrapper: `src/wgan_option/train_vol_xlsx.py`
- legacy trainer wrapper: `src/wgan_option/train.py`
- generator: `src/wgan_option/models/generator.py`
- critic: `src/wgan_option/models/discriminator.py`
- WGAN loss and training loop: `src/wgan_option/models/gan_model.py`
- training artifact writers: `src/wgan_option/utils/training_artifacts.py`
