# Modification Suggestions

## 1. Highest-priority code changes

### 1.1 Add best-checkpoint saving

Track the best model using `val_recon` and save dedicated artifacts such as:

- `generator_best.pt`
- `discriminator_best.pt`
- `best_metrics.json`, or an explicit best-epoch entry in the main metrics file

Why this should come first:

- it directly resolves the “final model vs. best model” problem
- it improves reproducibility
- it makes thesis reporting cleaner

### 1.2 Add early stopping

Use `val_recon` as the main monitored metric and add logic such as:

- `patience = 10` or `15`
- stop when validation does not improve for the full patience window
- always keep the best checkpoint

Why this matters:

- training termination becomes experiment-aware
- GPU time is used more efficiently
- cross-run comparison becomes more disciplined

### 1.3 Add `training_metrics.csv`

Export a CSV alongside `training_metrics.json`.

The CSV should at least include:

- `epoch`
- `d_total`
- `g_total`
- `g_adv`
- `g_recon`
- `gp`
- `g_calendar`
- `g_butterfly`
- `g_smooth`
- `val_recon`
- `val_calendar`
- `val_butterfly`

Why this matters:

- easier batch analysis
- easier thesis table generation
- easier use with pandas and Excel

### 1.4 Add fixed validation-sample visualizations

At evaluation time, save plots for a fixed subset of validation samples, including:

- `current_surface`
- `predicted_future_surface`
- `target_future_surface`
- `absolute_error`

Why this matters:

- makes error structure visible
- helps detect oversmoothing
- is much more useful for thesis discussion and defense slides

## 2. Second-priority experiment changes

### 2.1 Add baseline comparisons

At minimum, add:

- a persistence baseline: `prediction = current_surface`
- a no-text baseline

If time permits, also consider:

- a purely supervised regression baseline
- a non-adversarial variant using reconstruction loss only

### 2.2 Add region-wise error analysis

Do not rely only on overall `val_recon`. Break the error down by:

- short / medium / long maturities
- near-ATM / left wing / right wing regions
- stronger-quality / weaker-quality samples

This aligns better with the thesis framing, where surface structure matters.

### 2.3 Add month-level or rolling-window validation summaries

Because the validation set is the later time segment, also report:

- monthly validation error summaries
- rolling-window validation error summaries

This helps detect whether the model degrades in specific periods.

## 3. Hyperparameter suggestions

### 3.1 Do a small sweep before redesigning the architecture

This run does not show obvious instability, so a full GAN redesign is not the first move.

A smaller, targeted sweep is more justified.

### 3.2 Try a higher `lambda_recon`

Test values such as:

- `lambda_recon = 15`
- `lambda_recon = 20`

Why:

- the adversarial term becomes large late in training
- if forecasting accuracy matters more than purely adversarial realism, a stronger reconstruction term is a reasonable next step

### 3.3 Try a lower `discriminator_iter`

Test:

- `discriminator_iter = 3`

Why:

- late in training, `d_real - d_fake` is already very small
- the critic may no longer need `5` updates per generator step
- this may improve efficiency and reduce unnecessary adversarial oscillation

### 3.4 Add learning-rate scheduling

Introduce a scheduler driven by `val_recon`, for example a plateau-based reduction.

Why:

- the model clearly enters a plateau late in training
- a fixed learning rate for all `100` epochs is unlikely to be ideal

## 4. Text-conditioning experiment suggestions

This run uses `hd` mode. To support the thesis comparison across representations, run at least:

- `hd`
- `lp`
- `concat`
- `no-text`

The goal is not only to compare the final loss, but to answer:

- whether text adds measurable predictive value
- which embedding representation is most useful for surface forecasting

## 5. Recommended implementation order

Proceed in this order:

1. Add best-checkpoint saving
2. Add early stopping
3. Add `training_metrics.csv`
4. Add fixed validation-sample visualization
5. Add baseline comparison
6. Run a small hyperparameter sweep

## 6. Most important interpretation to preserve

This run should not be labeled a failed run.

The more accurate interpretation is:

- the model has learned a meaningful mapping
- the constraint terms are healthy
- late-stage validation performance is stable
- the main weaknesses are experiment management, model selection, and evaluation presentation, not fundamental training collapse

