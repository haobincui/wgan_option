# `use_reduce_lr_on_plateau` Explained

This document explains what `use_reduce_lr_on_plateau` means in this repo, how it works in the current implementation, and what formula is used when the learning rate is reduced.

It is based on the current code in:

- `src/wgan_option/config.py`
- `src/wgan_option/models/gan_model.py`
- `src/wgan_option/train_svi_xlsx.py`

## 1. What `use_reduce_lr_on_plateau` Means

`use_reduce_lr_on_plateau` is a boolean config flag.

- `false`: keep the optimizer learning rate fixed during training
- `true`: enable PyTorch `ReduceLROnPlateau`, so the learning rate is reduced automatically when the monitored validation metric stops improving

In this repo, the default is:

```yaml
use_reduce_lr_on_plateau: false
```

So the scheduler is available, but not active unless you turn it on explicitly.

## 2. Where It Is Used

### WGAN vol training

In the vol-surface WGAN path:

- generator optimizer uses `ReduceLROnPlateau`
- discriminator optimizer uses `ReduceLROnPlateau`
- both monitor `val_recon`

That means the scheduler decision is validation-driven, not based only on epoch count.

### SVI supervised training

In the SVI path:

- the optimizer uses `ReduceLROnPlateau`
- it monitors `val_regression`

## 3. When the Scheduler Is Active

The scheduler is created only if both conditions hold:

1. `use_reduce_lr_on_plateau = true`
2. a validation split is available

If there is no validation loader, the code disables the scheduler automatically and logs that clearly.

This is important, because `ReduceLROnPlateau` needs a validation metric such as `val_recon` or `val_regression`.

## 4. Related Config Fields

The scheduler behavior is controlled by these config values:

```yaml
use_reduce_lr_on_plateau: false
reduce_lr_factor: 0.5
reduce_lr_patience: 8
reduce_lr_min_lr: 1.0e-5
```

Their meanings are:

- `reduce_lr_factor`: multiplicative decay factor
- `reduce_lr_patience`: how long to wait before reducing LR when improvement stalls
- `reduce_lr_min_lr`: lower bound for the learning rate

## 5. Core Update Formula

When plateau reduction is triggered, the scheduler updates the learning rate by:

\[
\text{new\_lr} = \max(\text{old\_lr} \times \text{factor},\ \text{min\_lr})
\]

In this repo, with the recommended default values:

\[
\text{new\_lr} = \max(\text{old\_lr} \times 0.5,\ 10^{-5})
\]

### Example

If the current learning rate is:

\[
2 \times 10^{-4}
\]

then the first reduction becomes:

\[
\max(2 \times 10^{-4} \times 0.5,\ 10^{-5}) = 10^{-4}
\]

The next reduction becomes:

\[
\max(10^{-4} \times 0.5,\ 10^{-5}) = 5 \times 10^{-5}
\]

The sequence then continues until it reaches the floor:

\[
2 \times 10^{-4}
\rightarrow
10^{-4}
\rightarrow
5 \times 10^{-5}
\rightarrow
2.5 \times 10^{-5}
\rightarrow
1.25 \times 10^{-5}
\rightarrow
10^{-5}
\]

After the learning rate reaches `min_lr`, it will not go lower.

## 6. What "Plateau" Means Here

The current code constructs PyTorch `ReduceLROnPlateau` with:

- `mode="min"`
- configured `factor`
- configured `patience`
- configured `min_lr`

So the scheduler expects the monitored metric to decrease.

That means:

- for WGAN, smaller `val_recon` is better
- for SVI, smaller `val_regression` is better

Under the current installed PyTorch defaults, the scheduler also uses:

- `threshold = 1e-4`
- `threshold_mode = "rel"`
- `cooldown = 0`
- `eps = 1e-8`

With `mode="min"` and `threshold_mode="rel"`, an epoch counts as a meaningful improvement only if:

\[
\text{current\_metric} < \text{best\_metric} \times (1 - \text{threshold})
\]

So with the default threshold:

\[
\text{current\_metric} < \text{best\_metric} \times (1 - 10^{-4})
\]

If that condition is not met for long enough, the scheduler treats the metric as being on a plateau and reduces the learning rate.

## 7. Role of `reduce_lr_patience`

`reduce_lr_patience` controls how many non-improving validation checks the scheduler tolerates before lowering the learning rate.

With the repo default:

```yaml
reduce_lr_patience: 8
```

the scheduler waits through a short plateau window before reducing LR.

Practical meaning:

- if validation is still improving, keep the current LR
- if validation stops improving for multiple epochs, lower LR to allow finer late-stage optimization

This is especially useful in this repo because:

- WGAN training can enter a late-stage validation plateau
- adversarial training can remain noisy even when the main validation metric has mostly stabilized
- a smaller LR can help with controlled refinement instead of continuing with the original step size

## 8. Order of Operations During Training

### WGAN path

For each epoch, the current logic is:

1. run training batches
2. compute validation metrics
3. update best-checkpoint tracking using `val_recon`
4. step the generator and discriminator plateau schedulers using `val_recon`
5. save periodic checkpoints if needed
6. apply early-stopping logic if enabled

### SVI path

For each epoch, the current logic is:

1. run training epoch
2. compute validation metrics
3. update best-checkpoint tracking using `val_regression`
4. step the plateau scheduler using `val_regression`
5. save periodic checkpoints if needed
6. apply early-stopping logic if enabled

## 9. Metrics Files and Curve Interpretation

When the scheduler is enabled, the learning rate is also written into the metrics files.

### WGAN

- `g_lr`
- `d_lr`

### SVI

- `lr`

These are not losses.
They are recorded so the experiment log shows when LR changed during training.

One subtle but important detail:

- the metric row for epoch `t` stores the learning rate used during epoch `t`
- if plateau reduction is triggered after validation at epoch `t`, the lower LR is applied starting from epoch `t + 1`

So the reduced LR will usually appear in the next epoch's metrics row.

## 10. Relationship to Early Stopping

`ReduceLROnPlateau` and early stopping are separate mechanisms.

- `ReduceLROnPlateau` says: "validation improvement has slowed, so try a smaller LR"
- early stopping says: "validation has not improved enough for long enough, so stop training"

In this repo they can be enabled together.

This is often a useful combination:

1. first reduce LR when validation enters a plateau
2. then stop only if the smaller LR still does not produce improvement

## 11. Practical Interpretation for `vol_xlsx`

For the vol-surface WGAN path, enabling `use_reduce_lr_on_plateau` means:

- do not decay LR just because many epochs have passed
- decay LR only when `val_recon` stops improving enough
- let the generator and discriminator both move into a smaller-step late-stage regime

This is more suitable for the current project than a fixed epoch-only decay, because the WGAN validation curve can plateau at different times across runs.

## 12. Short Summary

`use_reduce_lr_on_plateau` turns on validation-driven learning-rate decay.

In formula form, when a plateau is detected:

\[
\text{new\_lr} = \max(\text{old\_lr} \times \text{reduce\_lr\_factor},\ \text{reduce\_lr\_min\_lr})
\]

In this repo:

- WGAN monitors `val_recon`
- SVI monitors `val_regression`
- no validation split means the scheduler is disabled automatically
- the default recommendation is conservative and optional, not always-on
