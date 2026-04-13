# Forecasting Bond Option SVI Parameters with Text Embeddings and a Supervised Regressor

This document describes the intended thesis-facing architecture and the current executable implementation for merged-SVI forecasting in this repo.

It should be read together with:

- `docs/input_svi.md`
- `docs/current_executable_workflows.md`
- `docs/vol_surface_gan_architecture.md`
- `docs/training_loss_curves.md`

This is not a GAN document.
It explains the current SVI forecasting path that trains a supervised regressor on paired `backward/current -> forward/future` SVI samples.

## 1. Research Objective

The core forecasting target for this path is:

\[
(\text{CurrentSVI}_t,\ \text{TextEmbedding}_t) \rightarrow \text{FutureSVI}_{t+h}
\]

where:

- `CurrentSVI_t` is the current SVI representation associated with the news row's `backward` direction
- `TextEmbedding_t` is the embedding of the associated news item
- `FutureSVI_{t+h}` is the future SVI representation associated with the same news row's `forward` direction

In the current thesis direction, this path matters because it predicts the SVI representation itself rather than a reconstructed surface.

That makes it useful for:

- representation-level forecasting experiments
- comparing parameter forecasting against surface forecasting
- keeping the SVI structure explicit instead of hiding it inside a downstream surface image

Unlike the merged vol path, this implementation does not use adversarial training.
The current executable choice is a supervised regressor because the target is a structured parameter set:

- future per-slice SVI parameters
- future valid-slice count

So this path should be understood as:

- a forecasting model on SVI parameters
- a complement to the merged vol WGAN path
- a direct comparison point for the thesis question of which representation is most useful

## 2. Executable Training Path in the Repo

The current merged-SVI path is implemented by:

- `scripts/train/main.py`
- `scripts/train/train_svi.py`
- `src/wgan_option/train_svi_xlsx.py`
- `src/wgan_option/utils/merged_xlsx.py`
- `src/wgan_option/models/svi_regressor.py`

Current preferred command:

```bash
python scripts/train/main.py svi-xlsx --config configs/wgan/train_svi_xlsx.yaml
```

Relative to the other two training modes in the repo:

- the legacy daily-surface path predicts proxy future surfaces from raw option data
- the merged vol path predicts future reconstructed vol surfaces from paired workbook rows
- the merged SVI path predicts future padded SVI parameters and future slice count from paired SVI rows

This makes the merged SVI path the current executable implementation that is closest to:

\[
(\text{current SVI},\ \text{text}) \rightarrow \text{future SVI}
\]

rather than:

\[
(\text{current surface},\ \text{text}) \rightarrow \text{future surface}
\]

## 3. Data Lineage and Sample Semantics

The practical data flow for the merged SVI path is:

1. raw option trades
2. minute SVI calibration
3. merge with news embeddings into `merged_svi.xlsx`
4. runtime `backward/current -> forward/future` pairing
5. supervised SVI training

### 3.1 Source workbook and row filtering

The trainer reads:

- workbook: `merged_svi.xlsx`
- sheet: `news_direction_audit`

This is important because `merged_svi.xlsx` itself is still direction-level, not pair-level.

Before pairing, the loader filters rows to:

- `training_candidate_flag == 1`

This means the executable trainer only sees rows that have already passed the upstream quality gate.

### 3.2 Runtime pairing

Runtime training samples are created in `load_svi_paired_samples(...)`.

The pairing rule is:

- group usable rows by `news_row_id`
- require both `backward` and `forward`
- treat `backward` as current
- treat `forward` as future

The repo semantics remain:

- `backward = current = timestamp_utc`
- `forward = future = timestamp_utc + 5 minutes`

So one executable sample is **not** one workbook row.
It is one runtime pair built from two workbook rows belonging to the same `news_row_id`.

This is why the current SVI training path does **not** read `gan_input_ready`.
That sheet remains a direction-level export, while the executable trainer pairs rows on the fly from `news_direction_audit`.

### 3.3 Chronological ordering

After pairing, samples are ordered chronologically by the `backward/current` side's `news_timestamp_utc`.

That ordered sequence is then split by `train_ratio`, so the validation side behaves like a time-ordered holdout rather than a random sample.

## 4. Tensor Construction and Normalization

The SVI tensorization logic lives in `src/wgan_option/utils/merged_xlsx.py`.

### 4.1 Per-slice feature order

The fixed feature order is:

```text
business_days, a, b, rho, m, sigma
```

So each valid SVI slice contributes 6 scalar values.

### 4.2 Fixed-width SVI matrix

For each current or future side, the loader builds a padded matrix of shape:

\[
[\text{max\_slices}, 6]
\]

using the configured `max_slices`.

In the current default config:

- `max_slices = 4`

So each side is represented as a `4 x 6` matrix.

The loader also creates:

- a slice mask of shape `[max_slices]`
- a slice count

Rows beyond the valid slice count are padded with zeros and masked out.
If a sample has zero slices or more than `max_slices`, it is rejected by the data loader.

### 4.3 Normalization

Normalization is fit on the training partition only.

More precisely:

- the code first computes the chronological split index
- it fits normalization statistics using only train-side paired samples
- within those train samples, it uses valid slices from both current and future matrices
- masked padding rows do not contribute to the fitted mean or standard deviation

The saved normalization payload contains:

- `feature_order`
- `mean`
- `std`
- `max_slices`
- `feature_dim`

This is important for reproducibility because the saved model is tied to the normalization used during training.

### 4.4 Final model inputs and targets

Let:

- `S = max_slices`
- `F = 6`

Then the model tensors are:

#### Current SVI input vector

The current side is converted into:

- flattened normalized current matrix: length `S * F`
- current mask: length `S`
- normalized slice-count scalar: length `1`

So:

\[
\text{current\_input\_dim} = S \cdot F + S + 1
\]

Under the default `S = 4`:

\[
\text{current\_input\_dim} = 4 \cdot 6 + 4 + 1 = 29
\]

#### Text embedding vector

The text vector is selected by mode:

- `hd`
- `lp`
- `concat`

The default config uses:

- `text_embedding_mode = hd`
- `embedding_dim = 1024`

If `concat` is used, the effective dimension is inferred from the workbook vectors instead of being hard-coded.

#### Future regression target

The future side is converted into the flattened normalized future matrix:

\[
\text{regression\_dim} = S \cdot F
\]

Under the default `S = 4`:

\[
\text{regression\_dim} = 24
\]

#### Future mask and future count target

The loader also produces:

- `future_mask`: shape `[S]`
- `future_count`: a zero-based class label derived from `(valid future slice count - 1)`

So the count classifier predicts one of `S` classes, corresponding conceptually to:

- 1 valid future slice
- 2 valid future slices
- 3 valid future slices
- 4 valid future slices

when `max_slices = 4`.

## 5. Model Architecture

The current model lives in `src/wgan_option/models/svi_regressor.py`.

It is a dual-encoder MLP with a shared trunk and two output heads.

### 5.1 High-level structure

The model has five pieces:

1. current SVI encoder
2. text encoder
3. fusion trunk
4. regression head
5. count head

At a high level, the computation is:

\[
\text{CurrentSVI vector} \rightarrow \text{current encoder}
\]

\[
\text{Text embedding} \rightarrow \text{text encoder}
\]

\[
[\text{current features},\ \text{text features}] \rightarrow \text{shared trunk}
\]

\[
\text{shared trunk} \rightarrow
\begin{cases}
\text{future SVI parameter head} \\
\text{future slice-count head}
\end{cases}
\]

### 5.2 Current SVI encoder

The current SVI vector is passed through:

1. `Linear(current_input_dim, hidden_dim)`
2. `LayerNorm(hidden_dim)`
3. `LeakyReLU(0.2)`
4. `Dropout(dropout)`

### 5.3 Text encoder

The text embedding is passed through the same style of block:

1. `Linear(embedding_dim, hidden_dim)`
2. `LayerNorm(hidden_dim)`
3. `LeakyReLU(0.2)`
4. `Dropout(dropout)`

### 5.4 Fusion trunk

The encoded current-SVI and text features are concatenated, then processed by:

1. `Linear(hidden_dim * 2, hidden_dim)`
2. `LayerNorm(hidden_dim)`
3. `LeakyReLU(0.2)`
4. `Dropout(dropout)`
5. `Linear(hidden_dim, hidden_dim)`
6. `LeakyReLU(0.2)`

This trunk creates one shared hidden representation used by both prediction tasks.

### 5.5 Output heads

The model then branches into two heads:

- regression head: `Linear(hidden_dim, regression_dim)`
- count head: `Linear(hidden_dim, count_classes)`

So the model predicts:

- future padded SVI parameters as one flat vector
- future slice-count logits as a classification output

### 5.6 Default dimensions

The current default config is:

- `svi_hidden_dim = 256`
- `svi_dropout = 0.1`
- `max_slices = 4`
- `embedding_dim = 1024`

Under the default non-concat setup, that means:

- `current_input_dim = 29`
- `regression_dim = 24`
- `count_classes = 4`

This is a compact MLP architecture rather than a sequence model or a GAN.
That is a deliberate modeling choice: the current implementation treats padded SVI slices as a fixed-width structured feature vector and predicts the future structure directly.

## 6. Training Objective

The trainer optimizes two losses at the same time.

### 6.1 Regression loss

The first task is future SVI parameter regression.

Current implementation:

- masked Smooth L1 loss
- applied only on valid future slices

If we write:

- `\hat{Y}` for predicted future flattened SVI parameters
- `Y` for target future flattened SVI parameters
- `M` for the expanded future mask

then conceptually:

\[
\mathcal{L}_{reg}
=
\frac{\sum M \odot \text{SmoothL1}(\hat{Y}, Y)}{\sum M}
\]

Masking matters because padded rows are not real supervision targets.
Without masking, the model would be rewarded for predicting zeros on nonexistent future slices.

### 6.2 Count loss

The second task is future slice-count prediction.

Current implementation:

- cross-entropy loss on the zero-based future count class

This asks the model to predict how many valid future slices will exist, up to `max_slices`.

### 6.3 Total loss

The total loss is:

\[
\mathcal{L}_{total}
=
\mathcal{L}_{reg}
 \lambda_{count}\mathcal{L}_{count}
\]

where:

- `\lambda_{count} = count_loss_weight`

In the current default config:

- `count_loss_weight = 0.2`

So regression remains the main objective, while count prediction acts as an auxiliary structural target.

## 7. Current Runtime Training Behavior

The training loop lives in `src/wgan_option/train_svi_xlsx.py`.

### 7.1 Data split and loader behavior

The trainer uses the chronological order prepared by `merged_xlsx.py`.

It then:

- uses `train_ratio` to choose the split point
- shuffles batches inside the train partition
- keeps validation batches unshuffled

If the dataset is too small to produce a validation partition, the trainer still runs, but validation-dependent features are disabled.

### 7.2 Optimizer and monitored metric

The current trainer uses:

- `Adam`

The primary monitored validation metric is:

- `val_regression`

This is the key automated decision metric for the SVI path.
Best-checkpoint tracking, optional early stopping, and optional `ReduceLROnPlateau` all use `val_regression`.

### 7.3 Best checkpoint, early stopping, and scheduler

When validation is available:

- best checkpoint tracking is enabled
- the best checkpoint is the one with the lowest `val_regression`

Optional features:

- `use_early_stopping`
- `early_stopping_patience`
- `early_stopping_min_delta`
- `use_reduce_lr_on_plateau`
- `reduce_lr_factor`
- `reduce_lr_patience`
- `reduce_lr_min_lr`

If validation is unavailable, those validation-dependent controls are automatically disabled.

### 7.4 Artifact layout

For merged-xlsx training, the trainer rewrites artifact paths under a timestamped run directory beneath `output_root`.

For the current SVI config, the default root is:

- `outputs/training/svi_xlsx`

One run writes artifacts such as:

- resolved config snapshot: `metrics/run_config_<run_ts>.yaml`
- normalization stats: `metrics/normalization_stats.json`
- metrics json: `metrics/training_metrics.json`
- metrics csv: `metrics/training_metrics.csv`
- loss curves: `metrics/loss_curves.png`
- periodic checkpoints: `checkpoints/svi_regressor_epoch_*.pt`
- best model: `checkpoints/svi_regressor_best.pt`
- best-checkpoint metadata: `metrics/best_checkpoint.json`
- final model: `checkpoints/svi_regressor.pt`

The checkpoint payload also stores:

- `state_dict`
- resolved config values
- embedding dimension
- current input dimension
- regression dimension
- `max_slices`
- normalization stats

So the saved artifacts are not only model weights; they also preserve the tensorization context needed to interpret those weights later.

## 8. Why the Merged SVI Path Matters

The merged SVI path is important because it isolates the parameter representation directly.

Relative to the merged vol path, it has three important differences:

1. it predicts future SVI structure directly instead of a reconstructed surface grid
2. it keeps maturity-slice parameters explicit rather than hiding them inside an image-like tensor
3. it makes representation comparison easier at the thesis level, because the forecast target is still in the SVI parameter space

That does not make it automatically better than the merged vol path.
It means the two paths answer slightly different experimental questions:

- merged vol asks whether text helps forecast the future surface directly
- merged SVI asks whether text helps forecast the future structural parameterization that generated the surface

## 9. Current Constraints and Research Caveats

The current SVI path is executable and experiment-ready, but several caveats remain important.

### 9.1 Upstream-fit caveat

Training quality depends on the upstream SVI calibration and merge quality.

Fields such as:

- `training_candidate_flag`
- `weighted_iv_rmse`
- `exact_slice_point_ratio`

still matter for experiment interpretation, even though the trainer itself reads only the already-filtered rows.

### 9.2 Representation-width caveat

`max_slices` is a hard modeling assumption.

That means:

- the current model expects a bounded number of slices
- the padded representation is only valid within that bound
- samples with too many slices are not silently stretched into a longer representation

### 9.3 Count-modeling caveat

Future slice count is modeled as a classification problem, not a continuous regression problem.

So the model learns a discrete set of count outcomes tied to the configured `max_slices`.

### 9.4 Constraint caveat

Unlike the vol-surface WGAN path, the current SVI regressor does not include explicit arbitrage or surface-shape penalties in its loss.

Its inductive bias comes mainly from:

- the structured SVI representation
- the quality filtering in the workbook
- the multi-task setup of parameter regression plus count prediction

### 9.5 Workbook-versus-sample caveat

This path is a paired forecasting implementation, but the workbook remains direction-level for audit purposes.

That distinction is intentional and should be preserved:

- workbook rows remain useful for fit-quality analysis
- runtime samples remain the true forecasting unit used by the trainer

## 10. Key Code Locations

- config: `configs/wgan/train_svi_xlsx.yaml`
- CLI entrypoint: `scripts/train/train_svi.py`
- trainer wrapper: `src/wgan_option/train_svi_xlsx.py`
- paired-xlsx loader: `src/wgan_option/utils/merged_xlsx.py`
- model: `src/wgan_option/models/svi_regressor.py`
