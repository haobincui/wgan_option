# Comparison Analysis

## Compared Runs

- Prior run: `outputs/vol_xlsx_20260406-03`
- Current run: `outputs/vol_xlsx_20260406-04`

Shared core setup:

- `data/processed_excel_20260330-01/merged_vol.xlsx`
- `gan_input_ready`
- `hd` text mode
- `100` epochs
- `batch_size = 16`
- `learning_rate = 0.0002`
- `discriminator_iter = 5`
- calendar / butterfly / smooth constraints all enabled
- early stopping disabled

Main workflow difference:

- `04` enables `ReduceLROnPlateau`
- `03` does not record a scheduler in its config snapshot

## Main Comparison Conclusion

`04` should not be described as a better run than `03`.

The more accurate interpretation is:

- `03` is stronger on validation accuracy
- `04` is much stronger on late-stage stability
- `04` therefore looks more controlled, but also more conservative

This is consistent with a scheduler-driven tradeoff:

- less oscillation
- less aggressive late-stage exploration
- weaker best and final validation reconstruction

## Metric Comparison

### Best validation points

- `03` best overall `val_recon`: epoch `85`, `0.036803`
- `04` best overall `val_recon`: epoch `27`, `0.037823`

Difference:

- `04 - 03 = +0.001020`

So `03` is better at its best point.

### Best saved 10-epoch checkpoints

- `03` best saved periodic checkpoint: epoch `80`, `0.037560`
- `04` best saved periodic checkpoint: epoch `30`, `0.038482`

Difference:

- `04 - 03 = +0.000922`

Again, `03` is better.

### Final epoch comparison

- `03` final `val_recon`: `0.038589`
- `04` final `val_recon`: `0.039700`

Difference:

- `04 - 03 = +0.001111`

So the scheduler-enabled run also finishes worse at the final epoch.

### Late-stage average validation performance

Tail-20 mean `val_recon`:

- `03`: `0.038824`
- `04`: `0.039578`

Difference:

- `04 - 03 = +0.000754`

This shows that `04` is not only worse at its best point, but also slightly worse on late-stage average validation quality.

### Final fit and regularization metrics

Relative to `03`, the final epoch of `04` is worse on the main tracked final quantities:

- `final gp`: `+0.018658`
- `final g_recon`: `+0.004314`
- `final g_calendar`: `+0.000069`
- `final g_butterfly`: `+0.000049`
- `final g_smooth`: `+0.000020`

So the calmer late-stage trajectory in `04` does not translate into tighter final fit or better final shape-control metrics.

## Stability Comparison

This is where `04` clearly improves on `03`.

Tail-20 standard deviation of `val_recon`:

- `03`: `0.001322`
- `04`: `0.000370`

Difference:

- `04 - 03 = -0.000952`

Tail-20 standard deviation of `g_total`:

- `03`: `1.676371`
- `04`: `0.041037`

Difference:

- `04 - 03 = -1.635334`

This is a very large stability improvement. Late-stage dynamics in `04` are dramatically less noisy than in `03`.

The right interpretation is not that `04` wins overall. It is that `04` demonstrates that the scheduler can strongly suppress oscillation, but under the current settings it suppresses it at some cost to validation accuracy.

## Scheduler Interpretation

The scheduler behavior in `04` is central to the comparison.

- first reduction at epoch `22`: `2e-4 -> 1e-4`
- best epoch at `27`
- further reductions at `37`, `46`, `55`, and `64`
- learning rate reaches the floor `1e-5` by epoch `64`

This sequence fits the observed training pattern:

- the run is still capable of reaching its best point shortly after the first reduction
- once multiple reductions accumulate, the trajectory becomes highly stable
- that stability does not lead to a better validation basin than `03`

So the current scheduler settings look plausible as a stabilization mechanism, but not yet as the best accuracy-oriented default.

## Workflow / Artifact Comparison

In artifact management, `03` and `04` are broadly at parity.

Both runs contain:

- `training_metrics.csv`
- `training_metrics.json`
- `best_checkpoint.json`
- `generator_best.pt`
- `discriminator_best.pt`
- periodic checkpoints
- final checkpoints
- `loss_curves.png`

That means the main difference between `03` and `04` is not experiment bookkeeping. It is training dynamics under the scheduler-enabled configuration.

## Status of Issues Raised in the `03` Review

The `03` review package identified several concrete improvement items. Against those items, the `04` run shows a mixed picture.

### Already addressed in `04`

#### Best-checkpoint workflow was preserved

This recommendation has been carried forward successfully.

`04` still contains:

- `generator_best.pt`
- `discriminator_best.pt`
- `best_checkpoint.json`

This remains important because the best epoch in `04` is epoch `27`, which is not aligned with the periodic `save_every = 10` checkpoint cadence.

#### Late-stage adversarial oscillation was reduced materially

This is the clearest substantive improvement relative to the `03` review suggestions.

The `03` review recommended prioritizing lower late-stage oscillation and explicitly suggested a validation-driven scheduler as a next trial.

`04` implements that idea with `ReduceLROnPlateau`, and the stability effect is clear:

- tail-20 `val_recon` standard deviation: `0.001322 -> 0.000370`
- tail-20 `g_total` standard deviation: `1.676371 -> 0.041037`

So the stability problem identified in `03` was addressed experimentally.

### Partially addressed in `04`

#### Best-to-best comparison is now easier to do, but not fully encoded in metadata

The presence of `best_checkpoint.json` and dedicated best checkpoint files continues to support the `03` recommendation to compare best-to-best rather than final-to-final.

However, the metadata is still sparse enough that reviewers must reopen the full metrics history to recover context around the best epoch.

So this issue is improved at the workflow level, but not fully solved at the artifact-description level.

### Still not addressed in `04`

#### `best_checkpoint.json` is still not relocation-safe

This remains unfixed.

The archived run is stored under:

- `outputs/vol_xlsx_20260406-04`

But `best_checkpoint.json` still points to:

- `outputs/vol_xlsx/checkpoints/generator_best.pt`
- `outputs/vol_xlsx/checkpoints/discriminator_best.pt`

So the metadata still reflects the original run location rather than the archived folder.

#### `best_checkpoint.json` is still too minimal for quick interpretation

The `03` review recommended adding fields such as:

- `final_epoch`
- `final_metric`
- `best_vs_final_delta`
- whether best is also final

These are still absent in `04`.

The current metadata remains limited to:

- `monitor_metric`
- `best_epoch`
- `best_metric`
- artifact paths

That is usable, but not yet review-friendly enough for fast archive comparison.

#### The best epoch’s full metric row is still not stored in metadata

This also remains unfixed.

To understand the best checkpoint fully, the reviewer still has to reopen `training_metrics.csv` or `training_metrics.json` to recover quantities such as:

- `gp`
- `g_calendar`
- `g_butterfly`
- `g_smooth`
- learning-rate state at the best epoch

That is exactly the extra friction highlighted by the `03` review.

### Interpretation of the overall status

The fairest summary is:

- the stability-focused experimental suggestion from `03` was acted on
- the experiment-management improvements introduced before `03` were retained
- the metadata-quality issues identified in `03` were not yet cleaned up in `04`

So `04` should be seen as:

- a genuine follow-up on the scheduler/stability recommendation
- not yet a full closure of the review backlog opened in `03`

## Important Caveats

`04` should not be presented as a model improvement simply because its curves are smoother.

The evidence points to:

- more stable late-stage training
- weaker best validation accuracy
- weaker final validation accuracy
- slightly weaker final constraint-related metrics

The archive portability issue also remains in `04`:

- the archived folder is `outputs/vol_xlsx_20260406-04`
- `best_checkpoint.json` still stores artifact paths under `outputs/vol_xlsx/checkpoints/...`

So the run is reviewable and reproducible in practice, but the metadata is still not fully self-contained after archiving.
