# Comparison Analysis

## Compared Runs

- Pre-change run: `outputs/vol_xlsx_20260406-02`
- Post-change run: `outputs/vol_xlsx_20260406-03`

Both runs use the same main modeling setup:

- `merged_vol.xlsx`
- `gan_input_ready`
- `hd` text mode
- `100` epochs
- `batch_size = 16`
- `learning_rate = 0.0002`
- `discriminator_iter = 5`

The post-change config snapshot includes extra fields because of the new training infrastructure:

- `use_early_stopping: false`
- `early_stopping_patience: 10`
- `early_stopping_min_delta: 0.0`
- explicit `use_*_constraint` flags

These additional fields do not by themselves imply a behavioral change, because early stopping was disabled in the run.

## Main Comparison Conclusion

The code changes improved experiment management and checkpoint selection clearly.  
They did not produce a clean, across-the-board modeling improvement in this pair of runs.

The fairest interpretation is:

- model quality is broadly comparable across the two runs
- `03` is slightly better at its best point and in late-stage average validation performance
- `02` is slightly better at the final epoch
- the biggest reliable gain from the code changes is better artifact management, not a decisive gain in forecasting accuracy

## Metric Comparison

### Best validation points

- `02` best overall `val_recon`: epoch `3`, `0.036831`
- `03` best overall `val_recon`: epoch `85`, `0.036803`

Difference:

- `03 - 02 = -0.000028`

This is an improvement for `03`, but it is very small.

### Best saved checkpoints under periodic saving

- `02` best periodic checkpoint: epoch `90`, `0.037607`
- `03` best periodic checkpoint: epoch `80`, `0.037560`

Difference:

- `03 - 02 = -0.000047`

Again, this is a small improvement for `03`.

### Final epoch comparison

- `02` final `val_recon`: `0.037832`
- `03` final `val_recon`: `0.038589`

Difference:

- `03 - 02 = +0.000757`

So the post-change run ends slightly worse if one compares only the last epoch.

### Late-stage average validation performance

Tail-20 mean `val_recon`:

- `02`: `0.039183`
- `03`: `0.038824`

Difference:

- `03 - 02 = -0.000359`

This supports the view that `03` is slightly better in the late-stage validation region on average.

### Late-stage stability

Tail-20 standard deviation of `val_recon`:

- `02`: `0.001735`
- `03`: `0.001322`

This suggests the validation reconstruction metric is a bit steadier in `03`.

At the same time, tail-20 standard deviation of `g_total` is larger in `03`:

- `02`: `0.833643`
- `03`: `1.676371`

This means the adversarial side remains more oscillatory in `03`, even though `val_recon` itself is not worse on average late in training.

## Constraint and Regularization Comparison at Final Epoch

Relative to `02`, the final epoch of `03` is slightly worse on every tracked final constraint-like metric:

- `val_recon`: `+0.000757`
- `g_recon`: `+0.000487`
- `gp`: `+0.002014`
- `g_calendar`: `+0.000025`
- `g_butterfly`: `+0.000028`
- `g_smooth`: `+0.000094`

These are not catastrophic differences, but they reinforce the point that the post-change code should not be credited with a direct modeling improvement from this comparison alone.

## Workflow Improvement Comparison

This is where `03` clearly beats `02`.

`02` provides:

- `training_metrics.json`
- `loss_curves.png`
- periodic checkpoints
- final `generator.pt` / `discriminator.pt`

`03` additionally provides:

- `training_metrics.csv`
- `best_checkpoint.json`
- `generator_best.pt`
- `discriminator_best.pt`

This is especially important because:

- the true best epoch in `03` is `85`
- epoch `85` is not a periodic checkpoint
- the new code preserves it automatically

In other words, even though the numeric forecasting improvement is modest, the reproducibility and experiment-selection improvement is real and meaningful.

## Important Caveat Found During Comparison

The saved `best_checkpoint.json` inside `03` points to:

- `outputs/vol_xlsx/checkpoints/generator_best.pt`
- `outputs/vol_xlsx/checkpoints/discriminator_best.pt`

But the archived comparison directory is:

- `outputs/vol_xlsx_20260406-03/...`

This indicates that the run was likely produced under the original `outputs/vol_xlsx` path and then moved or copied into a timestamped folder afterward.

As a result:

- the archived folder contains the correct files
- the paths inside `best_checkpoint.json` are no longer relocation-safe

That is an experiment-archive problem worth fixing.

