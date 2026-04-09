# Vol Training Run Review (2026-04-09)

This note summarizes the empirical results currently stored under `outputs/training/svi-all/` on branch `fix_vol`.

Despite the directory name `svi-all`, these runs are the recent `vol-xlsx` WGAN experiments against `data/processed/svi-all/20260330-01/merged_vol.xlsx`.
The output family is named after the shared data range rather than the model type, so this directory should be read as:

- `outputs/training/svi-all/*`
  - recent vol-surface WGAN runs on the `svi-all` merged dataset

This document complements:

- `docs/training_loss_curves.md`
- `docs/vol_surface_gan_architecture.md`

## 1. Runs Reviewed

The directory currently contains three kinds of runs:

- completed training runs with full metrics
- partial runs that only recorded a few epochs
- dry runs or aborted starts with config snapshots but no training metrics

### 1.1 Completed or usable runs

These are the runs with enough recorded metrics to compare validation behavior.

| Run | Mode | Capacity | Batch | LR | Best epoch | Best `val_recon` | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `20260409_154906` | `none` | `512 / 256` | `16` | `2e-4` | `4` | `0.035691` | Best overall run so far |
| `20260409_162036` | `lp` | `1024 / 786` | `16` | `1e-4` | `3` | `0.035869` | Best text-conditioned run so far |
| `20260409_154804` | `lp` | `512 / 256` | `16` | `2e-4` | `3` | `0.036456` | Strong baseline |
| `20260409_161819` | `lp` | `768 / 384` | `16` | `1e-4` | `1` | `0.036995` | Widened model, but improvement was limited |
| `20260409_154839` | `concat` | `512 / 256` | `16` | `2e-4` | `7` | `0.037638` | Worse than `lp` and `none` |
| `20260409_162904` | `lp` | `2048 / 1024`, base `64`, res `2` | `64` | `8e-5` | `24` | `0.038584` | Larger 24GB-style model, but worse validation fit |
| `20260409_162741` | `lp` | `2048 / 1024`, base `64`, res `2` | `64` | `8e-5` | `5` | `0.039517` | Same family as above, early stopped sooner |
| `20260409_163223` | `lp` | `4096 / 2048`, base `96`, res `4` | `128` | `6e-5` | `14` | `0.040598` | Most aggressive run so far, but validation got worse |
| `20260409_160317` | `hd` | `768 / 384` | `16` | `1e-4` | `1` | `0.041052` | Only one epoch recorded, treat as inconclusive |

Capacity above is shown as `gen_hidden_dim / disc_hidden_dim`.

### 1.2 Dry runs or incomplete setup runs

These runs should not be used for model comparison:

- `20260409_154827`
  - no config or metrics artifacts were found
- `20260409_160300`
  - dry run for widened `hd` config
- `20260409_162541`
  - dry run for `24gb_lp_es`
- `20260409_162634`
  - dry run for `24gb_none_es`
- `20260409_162947`
  - dry run for `24gb_lp_aggressive_es`

## 2. Main Findings

### 2.1 The current best run is still the small `none` baseline

The lowest validation reconstruction error in this directory is:

- run `20260409_154906`
- `text_embedding_mode = none`
- `best val_recon = 0.035691`

This is important because it means the current text-conditioned variants have not yet surpassed the no-text baseline on the main selection metric.

For the thesis, this does not prove text is useless, but it does show that:

- text signal is not yet giving a clear gain in the current executable WGAN setup
- the current architecture may be using text weakly or noisily
- a no-text baseline must remain part of any honest ablation story

### 2.2 The best text-conditioned run is moderately widened `lp`, not the largest model

The strongest text-conditioned result so far is:

- run `20260409_162036`
- `text_embedding_mode = lp`
- `gen_hidden_dim = 1024`
- `disc_hidden_dim = 786`
- `batch_size = 16`
- `learning_rate = 1e-4`
- `best val_recon = 0.035869`

This slightly improved over the smaller `lp` baseline run `20260409_154804` (`0.036456`), so moderate widening did help.

However, the improvement was small.
The result suggests that:

- widening from the original small MLP-based head can help
- there is some headroom in the text-conditioned path
- the gain is not large enough to justify arbitrarily scaling the model without changing the training regime

### 2.3 Larger 24GB-scale models did not improve `val_recon`

The larger recent runs were:

- `20260409_162904`
  - `lp`, batch `64`, base channels `64`, residual blocks `2`
  - best `val_recon = 0.038584`
- `20260409_163223`
  - `lp`, batch `128`, base channels `96`, residual blocks `4`
  - best `val_recon = 0.040598`

Both are clearly worse than the best batch-`16` runs.

The direct conclusion is:

- more capacity alone did not improve the main validation metric on this dataset

The likely reasons are:

- the train set is still small (`2187` samples)
- larger batches sharply reduce the number of generator updates per epoch
- the more aggressive models may be smoothing or stabilizing the critic without learning a sharper forecast map

## 3. Training Dynamics

### 3.1 Best validation often arrives very early

For the stronger small-batch runs:

- `20260409_154804` (`lp`)
  - best at epoch `3`
- `20260409_154906` (`none`)
  - best at epoch `4`
- `20260409_162036` (`lp`)
  - best at epoch `3`

This means the best model often appears within the first few epochs.
Later epochs usually degrade validation quality.

Examples:

- `20260409_154906`
  - best `val_recon = 0.035691` at epoch `4`
  - final `val_recon = 0.046277` at epoch `100`
- `20260409_162036`
  - best `val_recon = 0.035869` at epoch `3`
  - final `val_recon = 0.043543` at epoch `15`

This strongly supports keeping:

- best-checkpoint selection by `val_recon`
- early stopping for future experiments

### 3.2 Large batches reduce useful update count

With `2187` train samples, the generator update count per epoch is approximately:

- batch `16` -> `137` generator updates per epoch
- batch `64` -> `35` generator updates per epoch
- batch `128` -> `18` generator updates per epoch

The best checkpoints appeared after roughly:

- `20260409_154906` (`none`, batch `16`)
  - `548` generator updates to best
- `20260409_162036` (`lp`, batch `16`)
  - `411` generator updates to best
- `20260409_162904` (`lp`, batch `64`)
  - `840` generator updates to best
- `20260409_163223` (`lp`, batch `128`)
  - `252` generator updates to best

So although the larger-batch runs use more memory, they are not necessarily getting more useful optimization pressure.

In practice this means:

- using 24GB mainly to increase batch size was not a good trade-off here
- the current dataset size favors smaller batches and more updates

### 3.3 Larger models often lowered `gp`, but not validation error

The larger runs tended to have lower gradient-penalty levels near their best epochs:

- `20260409_154906`
  - best-epoch `gp = 0.1234`
- `20260409_162036`
  - best-epoch `gp = 0.1020`
- `20260409_162904`
  - best-epoch `gp = 0.0605`
- `20260409_163223`
  - best-epoch `gp = 0.0752`

This suggests the critic may be behaving more smoothly in the larger runs.
But smoother critic dynamics did not translate into better `val_recon`.

So at least for the current dataset and objective:

- lower `gp` alone is not a sufficient success criterion
- the main scoreboard still needs to be `val_recon`

## 4. Interpretation by Text Mode

### 4.1 `none`

`none` is currently the strongest mode on the main validation metric.

Interpretation:

- this is the best pure surface-transition baseline
- it should remain in every comparison table
- if later text-conditioned models do not beat it, that should be stated explicitly in thesis-facing writeups

### 4.2 `lp`

`lp` is currently the best text-conditioned mode.

Interpretation:

- among the text-conditioned variants tested so far, `lp` is the safest default
- moderate widening improved `lp`
- extreme scaling did not help under the current training recipe

### 4.3 `concat`

`concat` underperformed relative to `lp`.

Interpretation:

- concatenating both embeddings did not automatically add useful signal
- this may reflect noisy conditioning rather than a true upper bound on multimodal input quality

### 4.4 `hd`

The only widened `hd` run in this directory recorded one epoch and is not enough for a reliable conclusion.

Interpretation:

- there is not yet enough evidence to rank `hd` fairly against the others in the new widened setting

## 5. Recommended Next Steps

If the immediate goal is better `val_recon`, the next experiments should prioritize:

1. Keep batch size small or moderate.
   - Recommended range: `16` to `32`
   - The current data size does not support very large batches well.

2. Keep the model only moderately larger than the original baseline.
   - The current best text-conditioned result came from moderate widening, not the most aggressive architecture.

3. Leave early stopping enabled.
   - The best checkpoint usually appears very early.
   - Training far beyond the best epoch often hurts validation quality.

4. Continue using `none` as the primary no-text control.
   - It is currently the best overall result in this directory.

5. Treat `lp` as the main text-conditioned branch.
   - It is currently the strongest text-conditioned option in the recent runs.

## 6. Practical Recommendation

For the next round, the most defensible default experiment is:

- use `lp`
- keep batch size at `16` or `32`
- keep early stopping on
- prefer moderate widening over maximum-capacity scaling

If the goal is to maximize the current validation metric rather than test text usefulness, then:

- the best run so far is still the `none` baseline (`20260409_154906`)

If the goal is to preserve text conditioning while staying close to the best observed score, then:

- the best current candidate is `20260409_162036`

## 7. Caveats

- This review only uses the artifacts currently present under `outputs/training/svi-all/`.
- Not every run in that directory is complete; some are dry runs or partial smoke tests.
- These comparisons use `val_recon` as the main metric, following the current implementation and docs.
- This document is an empirical run review, not a proof of final model ranking across all seeds.
