# RQ1 Strict Out-of-Sample Experiment Implementation

> Plan timestamp: 2026-07-22 14:17:16 UTC
> Status: code implementation complete; formal 25-run experiment not started

## Research Question

RQ1 asks whether the LP text-conditioned FiLM-WGAN provides incremental
out-of-sample predictive information for the five-minute Treasury-option
implied-volatility surface relative to an otherwise identical no-text model
and quantitative persistence benchmarks.

The pre-specified primary difference is:

```text
Delta_text = no_text_error - text_error
Delta_text > 0 means the text model has lower forecast error.
```

The claim is predictive, not causal. The core models are persistence,
`film_wgan_text`, and `film_wgan_no_text`. The full RQ1 design also includes
`film_wgan_shuffled_text`, `concat_wgan_text`, deterministic
`film_cnn_text`, and `pca_ridge_no_text`.

## Method

### Data and splitting

- Main input: `data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx`.
- Group rows by `(current_snapshot_time_utc, target_snapshot_time_utc)`.
- Sort groups chronologically and split them 70/15/15 into train,
  validation, and untouched test sets.
- Keep every duplicate article row for one surface pair in the same split.
- Fit all normalization and statistical preprocessing on train only.
- Use validation only for checkpoint and hyperparameter selection.
- Access test only after all checkpoints and experiment metadata are frozen.

The audited data contain 3,711 rows and 2,913 unique surface pairs. The
expected grouped split is:

```text
train       2039 pairs / 2639 rows
validation   437 pairs /  533 rows
test         437 pairs /  539 rows
```

### Controlled model comparisons

- `film_wgan_text`: matched LP embedding, FiLM conditioning, stochastic WGAN.
- `film_wgan_no_text`: 1024-dimensional zero LP input with the same network.
- `film_wgan_shuffled_text`: split-local deterministic permutation that never
  takes text from the same surface pair.
- `concat_wgan_text`: stochastic WGAN with text used only by concatenation.
- `film_cnn_text`: deterministic FiLM CNN with no noise, critic, or GP.
- `pca_ridge_no_text`: train-only PCA of log-IV deltas and Ridge forecast from
  the current log surface.

Neural models use seeds `42, 101, 202, 303, 404`. No best-seed selection is
permitted. The paper-facing checkpoint is the lowest validation surface MAE
after epoch 10.

### Metrics

```text
primary point metric       surface_mae
secondary point metrics    short_atm_mae, atm7_abs_err
probabilistic metrics      energy score, 50/80/90% interval coverage,
                           interval width, calibration error, scenario spread
financial consistency      calendar and butterfly violation rates
```

Short ATM is fixed at `abs(strike - 1.0) <= 0.06` and maturity at most 90
days. Energy score uses arbitrage-weighted scenarios and is normalized by the
square root of the 16x16 grid size. The violation tolerance is `1e-8`.

### Inference

Article-row errors are first averaged within each unique surface pair. Primary
uncertainty uses a 10,000-repetition trading-day cluster bootstrap with seed
`20260722`, supplemented by a Diebold-Mariano/HAC test, pair-level win rate,
and seed-level paired t/Wilcoxon tests. Secondary and architecture tests report
Holm-adjusted p-values.

## Pre-Implementation State

The current loader sorts rows chronologically but still uses row-level 80/20
train/validation splitting. There is no untouched test set. Legacy `none`
uses a one-dimensional zero vector, `lambda_adv=0` still leaves stochastic
noise and critic training active, and there is no concatenation ablation.
Generate-result focuses on weighted-mean point errors and does not report
proper probabilistic scores or violation rates. Previous validation-selected
and best-seed results therefore remain exploratory.

## Implemented State

The completed RQ1 workflow provides a frozen split manifest, exactly
controlled text/no-text and architecture comparisons, proper deterministic
and quantitative baselines, point and distributional evaluation, dependence-
aware inference, and a self-contained experiment archive under:

```text
outputs/experiments/rq1_incremental_text_<run_ts>/
```

The archive will preserve input hashes, configs, git state, permutation maps,
run registry, validation-selected checkpoints, test predictions, comparison
tables, manifest, and validation summary.

## Planned Interfaces

```yaml
split_strategy: grouped_chronological
train_ratio: 0.70
val_ratio: 0.15
test_ratio: 0.15
split_manifest_path: ""
text_embedding_mode: lp        # zero_lp for the strict no-text model
text_alignment_mode: matched   # or permuted
text_permutation_seed: 20260722
forecast_mode: stochastic_wgan # or deterministic
conditioning_mode: film        # or concat
```

Legacy configs and checkpoints without these fields continue to load with the
existing FiLM-WGAN and two-way split defaults.

## Acceptance Criteria

- No surface pair crosses a split and normalization is train-only.
- Text and no-text networks have identical trainable parameter counts.
- Shuffled text is reproducible and never uses a same-pair donor.
- Deterministic mode creates no critic, noise, GP, or critic optimizer.
- Historical FiLM-WGAN checkpoints remain loadable.
- All 25 neural runs retain resolved configs and an epoch-after-10 checkpoint.
- Each test result contains 539 rows and comparison tables contain 437 pairs.
- Final tables report every seed and never select the best test/validation seed.

## Implementation Record

Implementation completed on 2026-07-22 UTC.

### Implemented files

- `src/film_wgan/config.py`: grouped three-way split, controlled text input,
  forecast/conditioning modes, calibration and violation settings.
- `src/film_wgan/data.py`: SHA256-bound split manifests, `surface_pair_id`,
  train-only normalization, `zero_lp`, split-local text permutation, and a
  reproducible permutation audit table.
- `src/film_wgan/models.py` and `src/film_wgan/trainer.py`: FiLM/concat and
  stochastic/deterministic execution, optional critic stack, versioned
  checkpoints, parameter counts, and legacy checkpoint compatibility.
- `src/film_wgan/inference.py` and `src/film_wgan/arbitrage.py`: test split
  generation, energy score, interval calibration, scenario spread, and
  calendar/butterfly violation rates while retaining historical output fields.
- `src/film_wgan/baselines/pca_ridge.py`: train-only PCA-Ridge baseline with
  validation-only hyperparameter selection.
- `configs/film_wgan/train_rq1_textbase.yaml`: frozen text-base training
  configuration. Its network and loss settings match the archived
  `20260417_131244` text run; only the pre-registered RQ1 data, split,
  checkpoint, and model-mode fields change.
- `scripts/rq1/rq1_experiment.py` and `scripts/rq1/*.sh`: preparation,
  sequential 25-run matrix, background execution, monitoring, checkpoint
  freezing, test generation, PCA-Ridge, comparison, validation, and packaging.
  Preparation copies the original `20260417_131244` resolved config and rejects
  any non-pre-registered hyperparameter divergence.
- `tests/test_standalone_wgan/test_film_wgan_module.py` and
  `tests/test_scripts/test_rq1_experiment.py`: semantic and statistical tests.
- `tests/test_argument_order_hygiene.py`: updated the existing hygiene audit to
  follow the current canonical surface-config implementation path under `src/`.

### Actual split audit

The real enriched SVI workbook was loaded in `py312` and produced:

```text
split       rows   unique surface pairs
train       2639   2039
validation   533    437
test         539    437
total       3711   2913
cross-split surface pairs = 0
```

The text and strict no-text generators both have 40,484,000 parameters under
the frozen architecture. The deterministic model initializes no critic and
uses zero-dimensional noise.

### Verification

Commands and results:

```bash
conda run -n py312 python -m unittest \
  tests.test_standalone_wgan.test_film_wgan_module \
  tests.test_scripts.test_rq1_experiment
# 33 tests passed

bash run_all_tests.sh
# 46 tests passed

conda run -n py312 python -m compileall -q \
  src/film_wgan scripts/rq1 tests/test_scripts/test_rq1_experiment.py \
  tests/test_standalone_wgan/test_film_wgan_module.py

for file in scripts/rq1/*.sh; do bash -n "$file"; done
git diff --check
```

Real-workbook dry runs succeeded for matched text, strict no-text, shuffled
text, concat conditioning, and deterministic FiLM variants. No test forecasts
or test metrics were inspected during implementation.

### Formal run commands

Prepare the immutable experiment inputs and start the five-model, five-seed
matrix sequentially in the background on one GPU:

```bash
EXPERIMENT_ROOT=$(ENV_NAME=py312 bash scripts/rq1/prepare_experiment.sh | tail -1)

CUDA_VISIBLE_DEVICES=1 ENV_NAME=py312 \
  bash scripts/rq1/start_training_matrix_background.sh "$EXPERIMENT_ROOT"

ENV_NAME=py312 bash scripts/rq1/monitor_training.sh "$EXPERIMENT_ROOT"
```

Only after all 25 validation-selected checkpoints are complete, freeze the
checkpoint registry, generate the untouched test split, build tables, and
package the archive:

```bash
ENV_NAME=py312 bash scripts/rq1/collect_checkpoints.sh "$EXPERIMENT_ROOT"

CUDA_VISIBLE_DEVICES=1 ENV_NAME=py312 \
  bash scripts/rq1/run_generate_test_matrix.sh "$EXPERIMENT_ROOT"

ENV_NAME=py312 bash scripts/rq1/build_comparison_archive.sh "$EXPERIMENT_ROOT"
ENV_NAME=py312 bash scripts/rq1/package_experiment.sh "$EXPERIMENT_ROOT"
```

### Known limitations

- The 25 formal neural training runs are intentionally not launched as part of
  code implementation; therefore no thesis result table exists yet.
- Five seeds provide limited power for seed-level t and exact signed-rank
  inference. The trading-day cluster bootstrap and HAC diagnostics remain
  important complements.
- The deterministic and persistence forecasts are degenerate predictive
  distributions. Their zero-width intervals are valid under the common metric
  API but should not be interpreted as calibrated probabilistic forecasts.
- The split is frozen for the current workbook SHA256. Any workbook change
  requires a new experiment root and manifest rather than reusing this split.

## Formal Run Completion

The 25-run matrix and untouched test evaluation completed on
`2026-07-23`. Final artifacts are stored in:

```text
outputs/experiments/rq1_incremental_text_20260722-145823/
```

Acceptance results:

```text
training runs complete        = 25 / 25
after-epoch-10 checkpoints    = 25 / 25
test rows per neural run      = 539
test surface pairs            = 437
neural test rows              = 13,475
best-seed selection           = false
validation status             = ok
```

The generation workflow was corrected to resolve `rq1_test_json` relative to
the run directory. RQ1 also disables the optional full-dataset ATM timeseries
pass after the selected test samples have been written; this avoids repeated
inference over all 3,711 rows while preserving all 539 test JSON payloads,
probabilistic metrics, arbitrage metrics, and run metadata.

The preregistered surface-MAE primary test does not support LP text superiority:
`no_text_minus_text=-0.000201`, cluster-bootstrap 95% CI
`[-0.000395, 0.000023]`, two-sided `p=0.0600`. Matched text does outperform the
shuffled-text placebo on short-ATM MAE, but no-text and PCA-Ridge remain
stronger point-forecast baselines. Full interpretation is recorded in:

```text
outputs/experiments/rq1_incremental_text_20260722-145823/docs/results_summary.md
```
