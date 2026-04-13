# Result Analysis

## Review Target

- Run directory: `outputs/vol_xlsx_20260406-04`
- Metrics file: `outputs/vol_xlsx_20260406-04/metrics/training_metrics.json`
- CSV metrics file: `outputs/vol_xlsx_20260406-04/metrics/training_metrics.csv`
- Best-checkpoint metadata: `outputs/vol_xlsx_20260406-04/metrics/best_checkpoint.json`
- Config snapshot: `outputs/vol_xlsx_20260406-04/metrics/run_config_20260406_151145.yaml`

## High-Level Conclusion

The `04` run is a healthy and stable WGAN training run.

It does not show evidence of collapse, and it preserves the usual pattern seen in the vol-surface experiments:

- training reconstruction improves substantially
- arbitrage-related penalties remain small
- validation loss reaches a usable region and then stays tightly bounded

The main distinguishing feature of this run is stability, not peak accuracy.

Relative to recent runs, the late stage is much calmer, but the best and final `val_recon` are weaker than the stronger `03` run. The most plausible reading is that the current `ReduceLROnPlateau` setup is trading off some predictive sharpness for smoother late-stage dynamics.

## Key Metrics

- Epochs recorded: `100`
- Final epoch `val_recon`: `0.039700`
- Best overall `val_recon`: epoch `27`, `0.037823`
- Best saved 10-epoch checkpoint by `val_recon`: epoch `30`, `0.038482`
- Tail-20 mean `val_recon`: `0.039578`
- Tail-20 standard deviation of `val_recon`: `0.000370`
- Tail-20 standard deviation of `g_total`: `0.041037`
- Final `g_recon`: `0.023680`
- Final `gp`: `0.037551`
- Final `g_calendar`: `0.000381`
- Final `g_butterfly`: `0.000225`
- Final `g_smooth`: `0.004380`

## Improvement Across Training

From epoch `1` to epoch `100`:

- `g_recon`: `0.103690 -> 0.023680` (`-77.16%`)
- `val_recon`: `0.048758 -> 0.039700` (`-18.58%`)
- `g_calendar`: `0.002172 -> 0.000381` (`-82.46%`)
- `g_butterfly`: `0.002129 -> 0.000225` (`-89.42%`)
- `g_smooth`: `0.013538 -> 0.004380` (`-67.65%`)
- `gp`: `0.143108 -> 0.037551` (`-73.76%`)

These numbers support the view that the generator learns a materially better forecast than its initialization state while keeping the surface-shape penalties under control.

## Learning-Rate Schedule Behavior

This run enables `ReduceLROnPlateau`:

- `use_reduce_lr_on_plateau = true`
- `reduce_lr_factor = 0.5`
- `reduce_lr_patience = 8`
- `reduce_lr_min_lr = 1e-5`

The recorded learning-rate reductions occurred at:

- epoch `22`: `2e-4 -> 1e-4`
- epoch `37`: `1e-4 -> 5e-5`
- epoch `46`: `5e-5 -> 2.5e-5`
- epoch `55`: `2.5e-5 -> 1.25e-5`
- epoch `64`: `1.25e-5 -> 1e-5`

Important observations:

- `val_recon` at epoch `22`: `0.044480`
- best `val_recon` at epoch `27`: `0.037823`
- `val_recon` at epoch `64`: `0.039308`
- final `val_recon` at epoch `100`: `0.039700`

This suggests the first learning-rate reduction did not immediately harm the run. The model still improved and reached its best validation point five epochs later.

However, after the schedule kept reducing the rate to `1e-5`, the run settled into a narrow late-stage band rather than finding a better validation basin. The scheduler appears to have reduced oscillation effectively, but it did not produce a stronger optimum than the better recent baseline.

## Best-vs-Final Checkpoint Interpretation

The run's best checkpoint is not the final checkpoint.

- `best_checkpoint.json` records `best_epoch = 27`
- best metric: `val_recon = 0.037823`
- final epoch `100`: `val_recon = 0.039700`
- final minus best delta: `+0.001877`

This matters because the periodic `save_every = 10` workflow would only have preserved epoch `30` as the best saved regular checkpoint.

The current best-checkpoint workflow therefore still provides real value in `04`, even though the run itself is not stronger than `03`.

For downstream evaluation and thesis reporting, `generator_best.pt` is the correct default artifact, not `generator.pt`.

## Cautions

The late-stage smoothness of `04` should not be misread as better overall modeling quality.

- the best validation point happens early, at epoch `27`
- the run never surpasses that point later
- the final checkpoint is clearly worse than the best checkpoint

Also, the archived metadata is still not relocation-safe:

- the archived run directory is `outputs/vol_xlsx_20260406-04`
- `best_checkpoint.json` still points to `outputs/vol_xlsx/checkpoints/...`

That does not invalidate the run, but it does weaken archive portability for later thesis audit and reproduction work.
