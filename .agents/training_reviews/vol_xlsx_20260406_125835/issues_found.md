# Issues Found

## 1. The final model is not necessarily the best validation model

The current training workflow saves a checkpoint every `10` epochs and then saves `generator.pt` and `discriminator.pt` at the very end.

That means:

- `generator.pt` is only the last-epoch model
- it is not guaranteed to be the best validation model
- for this run, the better saved checkpoint candidate is epoch `90`

This creates avoidable ambiguity for reproducibility and thesis reporting.

## 2. There is no best-checkpoint mechanism

The training loop does not automatically track the best model using `val_recon`, and it does not save a dedicated artifact such as `generator_best.pt`.

Consequences:

- model selection is manual
- multi-run comparison becomes cumbersome
- downstream evaluation can accidentally use the final model instead of the best validation model

## 3. There is no early stopping

The validation curve is already in a relatively stable plateau late in training. Running to a fixed `100` epochs is not harmful here, but there is no explicit stopping policy tied to validation quality.

Consequences:

- unnecessary training time
- weaker comparability across runs
- no natural link between stopping time and best validation behavior

## 4. There is no CSV export for the metrics

The current `outputs/vol_xlsx/metrics` directory contains:

- `training_metrics.json`
- `loss_curves.png`

It does not contain a structured `training_metrics.csv`.

Consequences:

- more friction for run-to-run comparison
- more friction for thesis tables and appendix summaries
- less convenient downstream analysis with pandas or Excel

## 5. Evaluation is too aggregate and lacks sample-level visualization

The current validation outputs track:

- `val_recon`
- `val_calendar`
- `val_butterfly`

These are useful, but they do not show:

- which maturity regions are hardest
- whether wings are oversmoothed
- whether the generator underfits local structure
- whether text conditioning meaningfully changes concrete predictions

## 6. There is no baseline comparison

The current run shows internal improvement, but not improvement against simple baselines such as:

- persistence baseline: predict `future_surface = current_surface`
- no-text baseline: remove the text embedding

Without those comparisons, it is still hard to quantify:

- how much the text input helps
- how much the GAN setup improves over simpler predictors

## 7. `g_total` can be misread

The generator total loss combines adversarial, reconstruction, and constraint terms.

In this run:

- `g_total` is strongly negative in the middle stage
- it turns positive again late in training

That does not mean the model becomes worse later. It mainly reflects the adversarial term changing sign and magnitude. Without documentation, future readers may misinterpret this curve.

