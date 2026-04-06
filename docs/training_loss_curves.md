# Training Loss Curves and Metrics

This document explains the metric names that appear in:

- `outputs/vol_xlsx/metrics/training_metrics.json`
- `outputs/vol_xlsx/metrics/training_metrics.csv`
- `outputs/vol_xlsx/metrics/loss_curves.png`
- `outputs/svi_xlsx/metrics/training_metrics.json`
- `outputs/svi_xlsx/metrics/training_metrics.csv`
- `outputs/svi_xlsx/metrics/loss_curves.png`

It is based on the current implementation in:

- `src/wgan_option/models/gan_model.py`
- `src/wgan_option/train_svi_xlsx.py`

## 1. How to Read the Curves

In this repo, the loss-curve figure is a compact diagnostic tool, not a single "scoreboard".
Different variables measure different things:

- training fit quality
- validation fit quality
- adversarial game balance
- arbitrage-related shape constraints
- learning-rate state

Because of that, not every metric should be interpreted with "lower is always better".

The most important rule is:

- for the vol-surface WGAN path, the primary model-selection metric is `val_recon`
- for the SVI supervised path, the primary model-selection metric is `val_regression`

## 2. WGAN Vol Training Metrics

This section covers the merged vol-surface training path used by `vol-xlsx`.

### 2.1 Metrics shown in `loss_curves.png`

The current WGAN loss-curve plot contains two panels:

- Primary losses:
  - `g_recon`
  - `val_recon`
  - `g_total`
  - `d_total`
  - `gp`
- Constraint losses:
  - `g_calendar`
  - `g_butterfly`
  - `g_smooth`
  - `val_calendar`
  - `val_butterfly`

### 2.2 Generator-side metrics

#### `g_total`

Generator total loss for one epoch.

Current implementation:

\[
g\_total
= g\_{adv}
 \lambda_{recon} \cdot g\_{recon}
 \lambda_{calendar} \cdot g\_{calendar}
 \lambda_{butterfly} \cdot g\_{butterfly}
 \lambda_{smooth} \cdot g\_{smooth}
\]

Only the enabled constraint terms are included.

Interpretation:

- this is the full optimization target for the generator
- it mixes several quantities with different scales
- it can be positive or negative
- it is useful for monitoring training dynamics, but it is not the best standalone model-selection metric

#### `g_adv`

Generator adversarial loss.

Current implementation:

\[
g\_{adv} = -\mathbb{E}[D(fake)]
\]

Interpretation:

- this measures how successfully the generator fools the critic
- lower values are not automatically "better" in isolation
- it should be read together with `d_total`, `d_real`, `d_fake`, and `val_recon`

#### `g_recon`

Generator reconstruction loss on the training set.

Current implementation:

\[
g\_{recon} = L1(fake\_future,\ real\_future)
\]

Interpretation:

- this is the average absolute difference between predicted and target future surfaces
- lower is better
- this is usually the most intuitive training-fit metric for the WGAN path

### 2.3 Constraint metrics

These metrics penalize undesirable surface shapes.

#### `g_calendar`

Training calendar-arbitrage penalty on generated surfaces.

The code computes total variance:

\[
w(\tau, k) = \sigma(\tau, k)^2 \tau
\]

and penalizes negative maturity-direction increments:

\[
\text{ReLU}\left(-(w_{\tau+1} - w_{\tau})\right)
\]

Interpretation:

- lower is better
- near zero means the generated surface is closer to satisfying the calendar no-arbitrage condition
- this is a shape/feasibility metric, not a direct forecast-accuracy metric

#### `g_butterfly`

Training butterfly-arbitrage penalty on generated surfaces.

The code:

- converts the generated implied-vol surface into Black call prices
- takes the second finite difference along strike
- penalizes negative curvature

Interpretation:

- lower is better
- near zero means the surface is closer to convexity in strike and therefore closer to no butterfly arbitrage

#### `g_smooth`

Training smoothness penalty on generated surfaces.

The code adds squared first differences along:

- maturity direction
- strike direction

Interpretation:

- lower means the generated surface is smoother
- this helps suppress noisy local oscillations
- if this becomes too dominant, the model may oversmooth the surface

### 2.4 Critic-side metrics

#### `d_total`

Critic total loss for one epoch.

Current implementation:

\[
d\_{total} = \mathbb{E}[D(fake)] - \mathbb{E}[D(real)] + gp
\]

Interpretation:

- this is the critic optimization target in WGAN-GP
- lower is not directly comparable to forecast accuracy
- it mainly tells you whether the adversarial game is training stably

#### `d_real`

Average critic score on real future surfaces.

Interpretation:

- larger than `d_fake` is usually expected in a healthy WGAN setup
- if `d_real` and `d_fake` become very close late in training, it often means the critic is finding it harder to distinguish real from generated samples

#### `d_fake`

Average critic score on generated future surfaces.

Interpretation:

- compare this with `d_real`
- a shrinking gap between `d_real` and `d_fake` may indicate that generated samples are becoming more realistic

#### `gp`

Gradient penalty term in WGAN-GP.

Current implementation:

\[
gp = \lambda_{gp} \cdot \mathbb{E}\left[(\|\nabla_{\hat{x}}D(\hat{x})\|_2 - 1)^2\right]
\]

Interpretation:

- lower and stable values are usually desirable
- this term helps keep the critic approximately 1-Lipschitz
- a very large or exploding `gp` can indicate unstable critic behavior

### 2.5 Validation metrics

#### `val_recon`

Validation reconstruction loss.

Current implementation:

\[
val\_{recon} = L1(fake\_future,\ real\_future)
\]

Interpretation:

- lower is better
- this is the main validation metric for the WGAN path
- best-checkpoint selection and learning-rate reduction monitor this metric

#### `val_calendar`

Validation calendar-arbitrage penalty on generated validation surfaces.

Interpretation:

- lower is better
- used to check whether arbitrage-aware shape quality also holds outside the training set

#### `val_butterfly`

Validation butterfly-arbitrage penalty on generated validation surfaces.

Interpretation:

- lower is better
- complements `val_calendar` when assessing the structural quality of generated validation surfaces

### 2.6 Learning-rate metrics

#### `g_lr`

Generator learning rate recorded for that epoch.

Interpretation:

- this is not a loss
- it shows the learning rate used by the generator optimizer during the completed epoch
- if `ReduceLROnPlateau` is enabled, this value can decrease across epochs

#### `d_lr`

Discriminator learning rate recorded for that epoch.

Interpretation:

- this is not a loss
- it tracks the critic optimizer learning rate
- with the current implementation, generator and discriminator both use validation-driven `ReduceLROnPlateau` on `val_recon`

### 2.7 Practical Reading Guide for WGAN Curves

If you are judging model quality, prioritize these metrics in this order:

1. `val_recon`
2. `val_calendar` and `val_butterfly`
3. `g_recon`
4. `gp`

Use `g_total`, `d_total`, `g_adv`, `d_real`, and `d_fake` mainly to understand adversarial training dynamics, not as direct performance rankings.

## 3. SVI Supervised Training Metrics

This section covers the merged SVI training path used by `svi-xlsx`.

### 3.1 Metrics shown in `loss_curves.png`

The current SVI loss-curve plot contains two panels:

- Primary losses:
  - `train_total`
  - `val_total`
  - `train_regression`
  - `val_regression`
- Count losses:
  - `train_count`
  - `val_count`

### 3.2 Core loss definitions

The SVI trainer predicts two things at the same time:

- future continuous SVI parameters
- future slice count

#### `train_regression`

Training regression loss on future SVI parameters.

Current implementation:

- masked Smooth L1 loss
- only valid future slices contribute to the regression error

Interpretation:

- lower is better
- this is the main fit metric for future SVI parameter prediction

#### `val_regression`

Validation regression loss on future SVI parameters.

Interpretation:

- lower is better
- this is the main validation metric for the SVI path
- best-checkpoint selection and learning-rate reduction monitor this metric

#### `train_count`

Training cross-entropy loss for predicting the number of future valid slices.

Interpretation:

- lower is better
- this captures whether the model is learning the discrete future slice-count target

#### `val_count`

Validation cross-entropy loss for future slice count.

Interpretation:

- lower is better
- used as an auxiliary validation signal

#### `train_total`

Training total loss.

Current implementation:

\[
train\_{total} = train\_{regression} + \text{count\_loss\_weight} \cdot train\_{count}
\]

Interpretation:

- this is the optimization target used during training
- it combines the continuous SVI regression task and the slice-count classification task

#### `val_total`

Validation total loss with the same definition:

\[
val\_{total} = val\_{regression} + \text{count\_loss\_weight} \cdot val\_{count}
\]

Interpretation:

- lower is better
- useful as a broad validation diagnostic
- model selection still focuses on `val_regression`

### 3.3 Learning-rate metric

#### `lr`

Optimizer learning rate recorded for that epoch.

Interpretation:

- this is not a loss
- it records the learning rate used during the completed epoch
- if `ReduceLROnPlateau` is enabled, it can decrease based on `val_regression`

## 4. Which Metrics Are Used for Automated Decisions

### WGAN vol path

- best checkpoint: lowest `val_recon`
- early stopping: monitored on `val_recon`
- `ReduceLROnPlateau`: monitored on `val_recon`

### SVI path

- best checkpoint: lowest `val_regression`
- early stopping: monitored on `val_regression`
- `ReduceLROnPlateau`: monitored on `val_regression`

## 5. Common Misreadings

### "Why can `g_total` be negative?"

Because `g_total` contains the adversarial term `g_adv = -E[D(fake)]`.
In WGAN training, adversarial terms do not behave like ordinary non-negative losses.

### "Is the smallest `d_total` always the best model?"

No.
`d_total` is mainly a critic-training diagnostic.
The more important forecast-quality metric is `val_recon`.

### "Why are `val_calendar` and `val_butterfly` important if `val_recon` already looks good?"

Because a surface can be numerically close to the target and still have poor structural shape properties.
These two metrics help check whether the prediction remains closer to no-arbitrage behavior.

### "Why are learning rates in the metrics file?"

They are included so the training record shows not only the loss values, but also when optimizer behavior changed during a run.
That is useful for experiment reproducibility and for diagnosing late-stage plateau behavior.

## 6. Short Recommendation

When reading a training run for thesis analysis:

- use `val_recon` or `val_regression` as the first summary metric
- use arbitrage or count-related metrics as secondary diagnostics
- use adversarial losses and learning-rate traces to explain training dynamics, not to replace the main validation metric
