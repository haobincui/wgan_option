# Temporal Residual FiLM-WGAN: Detailed Model Architecture

Last implementation review: 2026-08-07 (UTC)

This document describes the model that is currently used by the intraday RQ1
experiments. It replaces the earlier description of a single-surface CNN GAN.
The active model is a persistence-anchored, stochastic, temporal FiLM-WGAN that
encodes a masked sequence of raw-volatility surfaces and adds a gated text
residual to a no-text forecast.

The source of truth is:

- `src/film_wgan/models.py`
- `src/film_wgan/sequence_data.py`
- `src/film_wgan/trainer.py`
- `src/film_wgan/inference.py`
- `src/film_wgan/losses.py`
- `scripts/rq1_event30/event30_dataset.py`
- `configs/film_wgan/train_rq1_event30_textbase.yaml`

The older `FilmWGANGenerator` and `FilmWGANCritic` classes remain available for
legacy single-surface checkpoints. When `surface_input_mode=rolling_history`,
the trainer selects `TemporalFilmWGANGenerator` and
`TemporalFilmWGANCritic`, which are the focus of this document.

## 1. Forecasting Task

For the current 30-minute-history configuration, the model predicts the next
5-minute volatility surface from six 5-minute surface positions:

```text
history = [S_(t-25), S_(t-20), S_(t-15), S_(t-10), S_(t-5), S_t]
target  = S_(t+5)
```

The text-conditioned experiment uses news published strictly before the
forecast origin:

```text
t - 5 minutes <= news_time < t
```

Factiva publication timestamps are interpreted in `Europe/London`. The data
builder rejects future news, keeps the history and target within one CME
continuous session, and requires at least 3 of the 6 history positions. The
latest position `S_t` must always be valid.

The active RQ1 data source is a raw-volatility surface. Option implied-volatility
points are interpolated onto a fold-specific 16 x 16 grid; the sequence loader
requires `surface_model=raw`. No SVI parameters or SVI reconstruction enter the
current sequence experiment.

The conditional distribution is represented as:

```text
p(S_(t+5) | S_(t-25:t), support masks, text condition)
```

The generator predicts a log-IV increment relative to the persistence anchor
`S_t`, rather than directly predicting an unconstrained volatility level.

## 2. Current Configuration Snapshot

The architecture is configurable. The following values are the active defaults
in `train_rq1_event30_textbase.yaml`:

| Setting | Value |
|---|---:|
| Surface grid | 16 maturities x 16 moneyness points |
| History length | 6 positions |
| History spacing | 5 minutes |
| Minimum valid history | 3 positions |
| Forecast horizon | 5 minutes |
| History channels | 3 |
| Text-condition dimension | 389 |
| LP semantic dimension after PCA | 128 |
| Frame CNN base channels | 32 |
| Temporal hidden dimension | 256 |
| Temporal residual MLP blocks | 2 |
| Text hidden/output dimensions | 256 / 64 |
| Noise dimension | 64 |
| Fusion hidden dimension | 512 |
| Text dropout | 0.30 |
| Initial text gate | 0.0 |
| Batch size | 64 |
| Critic updates per generator update | 5 |
| Training epochs | 30 |
| Monte Carlo draws at evaluation | 32 |

The 60-minute predecessor remains configurable with 12 positions, but it is
not the configuration summarized in the tables below.

## 3. Data Tensor Contract

### 3.1 History tensor

The sequence dataset emits:

```text
history_features: [B, T, 3, H, W]
```

For the active configuration:

```text
[B, 6, 3, 16, 16]
```

The three channels at each time step are:

1. Train-normalized log implied volatility.
2. Raw-data support mask for each grid cell.
3. A step-validity mask broadcast over the full grid.

Formally, for step `j` and grid cell `c`:

```text
x[j,c,0] = (log(IV[j,c]) - train_mean[c]) / train_std[c]
x[j,c,1] = support[j,c]
x[j,c,2] = step_valid[j]
```

Unsupported cells and missing steps are set to zero after normalization. The
support and validity channels let the network distinguish a missing value from
a genuine normalized value near zero. No forward fill is used.

### 3.2 Target

The supervised target is the cell-wise log-IV increment:

```text
delta_target = log(S_(t+5)) - log(S_t)
```

It is normalized with train-only, cell-specific mean and standard deviation and
returned as a 256-dimensional vector. Losses and metrics use only the
intersection of current and target raw support:

```text
evaluation_support = support(S_t) AND support(S_(t+5))
```

### 3.3 Train-only transforms

The following artifacts are fitted separately within each rolling fold using
only its training partition:

- the 16 x 16 raw support grid;
- current/history log-IV normalization;
- target log-delta normalization;
- LP text PCA;
- news-count standardization.

Validation and test data cannot change these transforms. Their paths and hashes
are recorded in resolved configs and experiment artifacts.

## 4. Text Condition

### 4.1 Article pooling

For each forecast origin, articles in the strict prior 5-minute window are
deduplicated first by `ArticleID` and then audited by embedding SHA256. LP
embeddings are pooled as:

```text
pooled_lp = L2_normalize(mean(L2_normalize(article_embedding_i)))
```

A PCA transform is fitted on unique pooled training-news vectors only:

```text
pooled LP embedding -> train-fold PCA -> 128 semantic coordinates
```

PCA whitening is disabled. If a fold has fewer than 128 identifiable
components, the transformed vector is zero-padded to 128 dimensions.

### 4.2 The 389-dimensional compatibility layout

The active event30 dataset retains the earlier three-bucket interface so all
sequence variants have the same input width:

```text
indices   0:128   = prior-5-minute LP PCA features
indices 128:384   = reserved zero padding for two legacy semantic buckets
indices 384:389   = auxiliary news-arrival features
```

The five auxiliary values are:

```text
[
  zscore(log1p(article_count)),
  0,
  0,
  has_news,
  latest_news_recency_minutes / 5
]
```

The two zero entries are reserved for compatibility with the earlier
three-bucket count representation. They do not carry information in the strict
prior-5-minute experiment.

### 4.3 Controlled variants

All variants use exactly 389 input coordinates and the same downstream network:

| Variant | Semantic coordinates | Arrival/count coordinates | `has_text` |
|---|---|---|---:|
| `no_text` | all zero | all zero | 0 |
| `news_count` | all zero | observed auxiliaries | 1 for news rows |
| `matched_lp` | matched LP PCA-128 | observed auxiliaries | 1 for news rows |
| `shuffled_lp` | train-donor LP PCA-128 | target-row auxiliaries | 1 for news rows |

Shuffled donors come only from the training partition and must have disjoint
article IDs from the target row. No-news rows are zero in every text array.

`normalize_text_embedding=false` is intentional: these features have already
been transformed using fold-train-only artifacts.

## 5. Temporal Surface Encoder

The generator and critic have separate temporal encoders with identical
topology and independent parameters.

### 5.1 Shared frame CNN

Each history frame `[B,3,16,16]` is encoded by the same CNN:

| Layer | Output shape |
|---|---|
| `Conv2d(3,32,3,padding=1)` + LeakyReLU | `[B,32,16,16]` |
| `Conv2d(32,64,3,stride=2,padding=1)` + LeakyReLU | `[B,64,8,8]` |
| `Conv2d(64,128,3,stride=2,padding=1)` + LeakyReLU | `[B,128,4,4]` |
| `AdaptiveAvgPool2d(1)` + flatten | `[B,128]` |

The frame encoder is shared across all six time positions, preserving a common
spatial representation while limiting parameter growth.

### 5.2 Masked GRU

The sequence of 128-dimensional frame vectors is passed to:

```text
GRUCell(input_size=128, hidden_size=256)
```

At each step, the candidate hidden state is computed normally. It replaces the
previous state only when the broadcast step-validity channel is greater than
0.5. Missing positions therefore do not update the recurrent state:

```text
h_candidate[j] = GRUCell(frame_feature[j], h[j-1])
h[j] = h_candidate[j] if valid[j] else h[j-1]
```

After the final step, two residual MLP blocks refine the temporal state:

```text
h <- LeakyReLU(h + Linear(LeakyReLU(Linear(h))))
```

The temporal encoder output is `[B,256]`.

## 6. Generator

### 6.1 Inputs and output

The temporal generator receives:

```text
history       [B,6,3,16,16]
text          [B,389]
has_text      [B,1]
noise z       [B,64], z ~ N(0,I)
```

For backward API compatibility, the Python argument carrying the history is
still named `current_surface`. In rolling-history mode it contains the complete
five-dimensional history tensor, not only `S_t`.

It returns a normalized log-IV delta:

```text
delta_norm    [B,256]
```

### 6.2 No-text stochastic backbone

The temporal state `h` and noise `z` are concatenated:

```text
[h,z]: 256 + 64 = 320
```

The base forecast head is:

```text
Linear(320,512)
LeakyReLU(0.2)
Linear(512,512)
LeakyReLU(0.2)
Linear(512,256)
```

Its output, `base_delta`, is the no-text stochastic forecast. It does not depend
on the text encoder.

### 6.3 Text encoder

The text branch is:

```text
Linear(389,256)
LayerNorm(256)
LeakyReLU(0.2)
Dropout(0.30)
Linear(256,64)
LeakyReLU(0.2)
Dropout(0.30)
```

The 64-dimensional output is multiplied by `has_text`. This second mask is
applied after the MLP, preventing linear biases from creating an implicit text
condition for no-text samples.

### 6.4 Residual vector FiLM adapter

Text modulates only the 256-dimensional temporal bottleneck. A linear layer maps
the text feature to `gamma` and `beta`:

```text
[gamma,beta] = Linear(64,512)(text_feature)
h_film       = (1 + gamma) * h + beta
```

The FiLM projection is initialized to zero, so it initially acts as the
identity. The text adapter receives the modulation difference and the encoded
text:

```text
adapter_input = concat(h_film - h, text_feature)  # 256 + 64 = 320
text_delta    = Linear(256,256)(Dropout(LeakyReLU(Linear(320,256))))
```

The final adapter layer has small random weights (`std=1e-3`) and zero bias.

### 6.5 Zero-initialized scalar gate

The final generator equation is:

```text
delta_norm = base_delta
           + has_text * tanh(text_gate) * text_delta
```

`text_gate` is initialized at zero. Therefore, before Stage-B learning:

```text
G(history, any_text, z, has_text=1)
= G(history, zeros, z, has_text=0)
```

for an identical initialized backbone and noise draw. The small random adapter
initialization gives the scalar gate a non-zero learning signal; after the gate
moves away from zero, gradients also flow into the adapter.

This construction is central to the controlled experiment: text is asked to
learn an incremental residual rather than relearning the complete surface
dynamics.

### 6.6 Reconstructing volatility levels

The generated normalized delta is returned to log-IV units using train-only
statistics. Volatility levels are reconstructed from the latest actual surface:

```text
future_log_iv = clamp(log(S_t) + delta, log(1e-4), log(5.0))
S_hat_(t+5)   = exp(future_log_iv)
```

Thus `delta=0` is exactly the persistence forecast. The final volatility is
bounded to `[1e-4,5.0]` for numerical stability.

## 7. Projection Critic

The critic receives:

```text
candidate future surface [B,1,16,16]
history condition         [B,6,3,16,16]
text condition            [B,389]
has_text                  [B,1]
```

It has three branches.

### 7.1 History branch

The history uses an independent copy of the temporal CNN + masked GRU encoder
described above and produces `[B,256]`.

### 7.2 Candidate-future branch

The normalized candidate future surface is encoded by a one-channel frame CNN:

```text
Conv2d(1,32) -> Conv2d(32,64,stride=2)
-> Conv2d(64,128,stride=2) -> AdaptiveAvgPool -> [B,128]
```

The history and future representations are concatenated and mapped to a joint
512-dimensional feature:

```text
joint = LeakyReLU(Linear(256 + 128, 512))
```

### 7.3 Unconditional and text-projection scores

The critic text encoder is:

```text
Linear(389,256) -> LeakyReLU -> Dropout(0.30)
-> Linear(256,64) -> LeakyReLU
```

The final score is:

```text
D = Linear(512,1)(joint)
  + has_text * dot(Linear(512,64)(joint), text_feature) / sqrt(64)
```

There is no sigmoid: the output is a WGAN critic score, not a probability.
When `has_text=0`, the score is strictly unconditional with respect to the text
branch.

## 8. Parameter Counts

For the active 16 x 16, 6-step, 389-dimensional configuration:

| Module | Parameters |
|---|---:|
| Generator backbone | 1,211,200 |
| Generator text adapter and gate | 298,049 |
| Generator total | 1,509,249 |
| Critic total | 1,092,289 |
| Generator + critic | 2,601,538 |

The no-text and text variants instantiate the same generator architecture and
parameter count. Their controlled difference is the text values and
`has_text`, not network width.

## 9. Paired Stage-A / Stage-B Training

The continuation workflow is designed as follows:

1. **Stage A no-text parent**: train the temporal stochastic backbone on all
   eligible origins with a zero 389-dimensional text vector and `has_text=0`.
2. **Stage B continued no-text**: initialize from the corresponding Stage-A
   generator and continue on the news-conditioned comparison sample.
3. **Stage B text variants**: initialize from the exact same Stage-A generator,
   then train `news_count`, `matched_lp`, and `shuffled_lp` branches.
4. **Fresh critic**: critic weights and optimizer state are not inherited by
   Stage B.

The loader audits parent checkpoint SHA256, surface shape, input mode, history
length, support artifact, text-transform policy, and architecture version.
Current temporal checkpoints use schema version 5 and architecture identifier
`intraday_temporal_residual_film_v1`; the checkpoint also records parent and
transform lineage needed to reproduce the continuation initialization.

When an initial parent is configured, the generator optimizer uses separate
parameter groups:

```text
backbone learning rate     = 2e-6
text adapter learning rate = 2e-5
```

The implementation supports freezing the backbone for the first `N` epochs.
The current event30 base config sets `freeze_backbone_epochs=0`; any experiment
claiming a five-epoch freeze must show `freeze_backbone_epochs=5` in that run's
resolved config. Without a parent checkpoint, the full generator uses the
standard generator learning rate `2e-5`.

## 10. Training Objectives

### 10.1 Critic objective

For matched real samples and generated samples, the base WGAN-GP objective is:

```text
L_D = mean(D(fake)) - mean(D(real)) + gradient_penalty
```

The gradient penalty coefficient ramps from zero to `lambda_gp=20` over the
first 200 critic steps.

For a batch containing at least two text-conditioned rows, text embeddings are
rotated within the conditioned subset to construct mismatched real/text pairs.
With `lambda_mismatch=0.5`, the adversarial part becomes:

```text
L_D_adv = (mean(D(fake))
           + 0.5 * mean(D(real, mismatched_text))) / 1.5
          - mean(D(real, matched_text))
```

This asks the projection critic to identify both unrealistic surfaces and
incorrect surface-text alignment. The mismatch term is disabled for no-text
batches.

### 10.2 Generator objective

For the active configuration:

```text
L_G = lambda_adv * (-mean(D(fake)))
    + 20 * weighted_surface_MAE
    + 30 * short_ATM_MAE
    + 1e-4 * FiLM_regularization
```

The generator adversarial coefficient is zero through epoch 10 and becomes
`lambda_adv=0.1` afterwards. The critic still trains during this period.

`weighted_surface_MAE` uses support-aware weights. Cells satisfying:

```text
abs(moneyness - 1.0) <= 0.02
maturity_days <= 21
```

receive an 8x raw weight before the template is normalized to mean one. The
separate short-ATM MAE applies only to the same eligible supported cells. The
current setup therefore deliberately emphasizes this region through both the
weighted reconstruction term and a dedicated loss.

Calendar-arbitrage, butterfly-arbitrage, and smoothness penalties exist in the
code but are disabled in the active config (`use_*=false`, coefficient zero).
The current model must not be described as arbitrage-constrained during
training.

Both optimizers use Adam with `beta1=0.5` and `beta2=0.9`. The configured critic
learning rate is `1e-5`, and five critic updates are performed per generator
update.

## 11. Validation and Checkpoint Selection

Validation is chronological and fold-specific. The principal checkpoint metric
is unweighted support-aware validation surface MAE:

```text
checkpoint_metric = val_mae
```

Additional checkpoints track MAE gap versus persistence and short-ATM MAE gap.
Epochs 1-10 are excluded from paper-facing checkpoint selection:

```text
checkpoint_warmup_epochs = 10
```

Early stopping patience is seven validation epochs. Test data is not used for
normalization, PCA, checkpoint selection, or early stopping.

## 12. Scenario Generation and Point Forecast

At evaluation time, each sample receives 32 reproducible Gaussian noise draws.
The draw seed is derived from:

```text
(experiment seed, surface_pair_id, draw index)
```

Each draw produces one future surface. The active aggregation settings are:

```text
reweight_beta_mode = fixed
reweight_beta      = 0.0
aggregation_mode   = weighted_mean
```

`beta=0` makes all 32 scenario weights equal. Although the inference module can
reweight full rectangular surfaces by an arbitrage penalty, that mechanism has
no effect under the current config. The reported point forecast is therefore
the simple Monte Carlo mean.

## 13. Evaluation Metrics

All deterministic metrics are calculated only on the current-target support
intersection.

### 13.1 Point metrics

- `surface_mae`: mean absolute IV error over all supported grid cells.
- `surface_rmse`: root mean squared IV error over supported cells.
- `short_atm_mae`: MAE where `abs(moneyness-1)<=0.02` and maturity is at most
  21 days.
- `mae_gap_vs_current`: model MAE minus persistence MAE; negative is better.
- `win_flag_vs_current`: one when model MAE is lower than persistence MAE.

The raw support does not justify a fixed 7-day ATM extrapolation in the current
experiment, so `report_atm7_metric=false`.

### 13.2 Probabilistic metrics

- Energy score, normalized by the square root of the supported cell count.
- Local-neighbour variogram score across adjacent strike and maturity cells.
- Pointwise 50%, 80%, and 90% interval coverage.
- Mean interval width at each nominal level.
- Mean absolute calibration error across the three coverage levels.
- Scenario spread around the weighted mean.
- Effective scenario count and Monte Carlo standard error of surface MAE.

These intervals are raw generator intervals. Validation conformal calibration
is a separate post-processing step and does not change the neural architecture.

### 13.3 Financial consistency metrics

Calendar and butterfly violation rates are computed only when a sample has full
rectangular support. For partial raw support, the implementation records that
the metric is unavailable instead of treating missing cells as valid surface
values.

## 14. End-to-End Forward Pass

```text
raw IV history and masks
    [B,6,3,16,16]
        |
        v
shared frame CNN at each time
    [B,6,128]
        |
        v
masked GRU + 2 residual MLP blocks
    h = [B,256]
        |
        +---------------------------+
        |                           |
        v                           v
concat Gaussian z [B,64]       text [B,389]
        |                       MLP -> [B,64]
        v                           |
base forecast head                 v
base_delta [B,256]          vector FiLM + adapter
        |                    text_delta [B,256]
        +------------+--------------+
                     v
base_delta + has_text * tanh(gate) * text_delta
                     |
                     v
denormalize log-IV delta
                     |
                     v
exp(log(S_t) + delta)
                     |
                     v
predicted S_(t+5) [B,16,16]
```

## 15. Interpretation of the Controlled Comparison

The principal incremental-text contrast should compare `matched_lp` with
`continued_no_text` initialized from the same Stage-A parent, fold, and seed:

```text
text_effect = no_text_error - matched_lp_error
positive text_effect => matched LP text lowers forecast error
```

`news_count` tests whether news arrival intensity explains the result without
semantics. `shuffled_lp` tests whether correct article-surface alignment matters.
Because the gate starts at zero and the architecture width is identical, an
initial performance difference cannot be attributed to a larger text network's
uncontrolled base forecast.

The comparison still requires paired samples, the same support masks, the same
parent checkpoint SHA, and all pre-registered seeds. The architecture alone
does not guarantee that text will improve MAE.

## 16. Known Limitations

1. The 389-dimensional input contains 256 reserved zeros in the current strict
   prior-5-minute dataset. This preserves compatibility but is not a minimal
   representation.
2. The scalar text gate can remain close to zero when the incremental text
   signal is weak relative to reconstruction loss.
3. Strong reconstruction and short-ATM coefficients can dominate the relatively
   small adversarial coefficient.
4. Thirty-two latent draws do not guarantee calibrated prediction intervals;
   empirical undercoverage requires calibration or changes to distributional
   training.
5. Static-arbitrage training constraints are currently disabled, and partial
   raw support limits ex-post arbitrage diagnostics.
6. Results from the 2022-2023 rolling folds are development evidence, not an
   untouched future-period confirmation.

## 17. Code Reading Guide

| Concern | Implementation |
|---|---|
| Temporal generator and critic | `src/film_wgan/models.py` |
| Raw sequence loading and masks | `src/film_wgan/sequence_data.py` |
| Text pooling/PCA and strict news window | `scripts/rq1_event30/event30_dataset.py` |
| Parent loading and optimizer groups | `src/film_wgan/trainer.py` |
| WGAN-GP and MAE losses | `src/film_wgan/losses.py` |
| Monte Carlo scenarios and metrics | `src/film_wgan/inference.py` |
| Active hyperparameter snapshot | `configs/film_wgan/train_rq1_event30_textbase.yaml` |

For any reported experiment, the archived `training_resolved_config.yaml`,
split manifest, support artifact, text transform, parent-checkpoint audit, and
checkpoint SHA take precedence over defaults shown in this document.
