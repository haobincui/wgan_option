# Result Analysis

## Review Target

- Run directory: `outputs/vol_xlsx_20260406-03`
- Metrics file: `outputs/vol_xlsx_20260406-03/metrics/training_metrics.json`
- CSV metrics file: `outputs/vol_xlsx_20260406-03/metrics/training_metrics.csv`
- Best-checkpoint metadata: `outputs/vol_xlsx_20260406-03/metrics/best_checkpoint.json`
- Config snapshot: `outputs/vol_xlsx_20260406-03/metrics/run_config_20260406_135650.yaml`

## High-Level Conclusion

The post-change run is a healthy training run.

It does not show evidence of collapse, and it preserves the same overall learning pattern as the earlier vol-surface WGAN experiments:

- reconstruction improves substantially over training
- arbitrage-related penalties remain small and controlled
- validation loss reaches its best region in the late stage

The main practical gain of this run is not a dramatic jump in model quality.  
It is that the run now records and preserves the best validation checkpoint automatically.

## Key Metrics

- Epochs recorded: `100`
- Final epoch `val_recon`: `0.038589`
- Best overall `val_recon`: epoch `85`, `0.036803`
- Best saved 10-epoch checkpoint by `val_recon`: epoch `80`, `0.037560`
- Final `g_recon`: `0.019366`
- Final `gp`: `0.018894`
- Final `g_calendar`: `0.000312`
- Final `g_butterfly`: `0.000176`
- Final `g_smooth`: `0.004359`

## Improvement Across Training

From epoch `1` to epoch `100`:

- `g_recon`: `0.103961 -> 0.019366` (`-81.37%`)
- `val_recon`: `0.047970 -> 0.038589` (`-19.56%`)
- `g_calendar`: `0.002219 -> 0.000312` (`-85.94%`)
- `g_butterfly`: `0.001887 -> 0.000176` (`-90.66%`)
- `g_smooth`: `0.010501 -> 0.004359` (`-58.49%`)
- `gp`: `0.164833 -> 0.018894` (`-88.54%`)

These numbers support the interpretation that the generator is learning a better forecast while keeping shape regularity under control.

## What the New Infrastructure Captured Correctly

This run now contains artifacts that did not exist in the earlier workflow:

- `training_metrics.csv`
- `best_checkpoint.json`
- `generator_best.pt`
- `discriminator_best.pt`

This matters because the true best validation epoch in this run is epoch `85`, which is not a periodic `save_every=10` checkpoint.

Without the new best-checkpoint mechanism, the workflow would only have preserved:

- epoch `80`
- epoch `90`
- final epoch `100`

The new logic therefore captured a checkpoint that the old workflow would have missed.

## Cautions

The final epoch is not the best epoch.

- final epoch `100`: `val_recon = 0.038589`
- best epoch `85`: `val_recon = 0.036803`

So even though the training run is valid, downstream evaluation should use `generator_best.pt`, not `generator.pt`, if the goal is best validation performance.

