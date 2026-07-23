# RQ1 Pair-Level Text + Residual FiLM-WGAN Implementation

Implementation timestamp (UTC): `2026-07-23 13:30:14`

## Research Scope

This implementation changes RQ1 from an article-row comparison to a nested,
surface-pair-level incremental-text experiment:

```text
article rows
-> one pooled LP vector per current/target surface pair
-> fold-train-only PCA-128
-> zero-gated residual FiLM adapter
-> initialization from the paired no-text generator
```

The primary difference remains:

```text
Delta_text = no_text_error - text_error
Delta_text > 0 => matched LP text has lower error
```

The available news workbook ends in 2023. Therefore, all new output tables are
named `development_*`; they must not be described as untouched final
confirmation.

The execution default was switched on `2026-07-23` to the raw-vol workbook:

```text
data/processed/raw-excel/raw_vol_w5_s3_20260629-144428/merged_vol_rq2_text.xlsx
surface_model = raw
workbook sha256 = 470e4f542159bdbbfe76bce65cce7ed55daeb9f92f0e82614fa4419bc9893f84
```

These surfaces are produced directly from raw implied-volatility points using
linear strike interpolation with boundary clamping and linear interpolation in
total variance across maturities. They are not reconstructed from SVI
parameters. The prepare command rejects a workbook whose `surface_model` is not
exactly `raw`.

## Implemented Core Code

### Pair-level data and lineage

Implemented in:

```text
src/film_wgan/data.py
src/film_wgan/text_transform.py
```

`sample_unit=surface_pair` now:

1. validates that all article rows in a pair have identical surfaces, grids,
   and current/target timestamps;
2. requires `news_timestamp_utc == current_snapshot_time_utc`;
3. requires an exact five-minute forecast horizon;
4. maps `sample_id=news_<id>` to the 1-based row in the raw news workbook;
5. verifies exact LP embedding equality between merged and raw workbooks;
6. deduplicates by `ArticleID`, then by float32 embedding SHA256;
7. pools text as:

   ```text
   L2_normalize(mean(L2_normalize(unique_article_embedding_i)))
   ```

Each pair retains source sample IDs, article IDs, source files, news count,
unique embedding count, pooling mode, and `has_text`.

### Train-only text transform

New config fields:

```yaml
text_preprocessing_mode: raw_l2 | pca | coordinate_zscore
text_transform_path: <fold>/text_transform.npz
text_pca_components: 128
text_pca_whiten: false
```

For PCA, the artifact stores the mean, components, explained variance, input
and output dimensions, train-pair SHA256, source workbook SHA256, and artifact
SHA256. Loading fails when either the fold train pairs or workbook hash differs.

Both matched text and no-text load the same fold artifact. The no-text branch is
zeroed only after transformation, so it has the same 128-dimensional network
shape while remaining exactly zero.

### Residual FiLM and projection critic

Implemented in:

```text
src/film_wgan/models.py
src/film_wgan/losses.py
src/film_wgan/trainer.py
src/film_wgan/inference.py
```

The generator computes:

```text
base_delta = surface_backbone(current_surface, noise)
text_delta = text_adapter(bottleneck_features, PCA_text)
forecast_delta = base_delta + has_text * tanh(text_gate) * text_delta
```

`text_gate=0` gives bitwise-identical predictions for arbitrary text and the
no-text mask. The adapter has a small random output initialization so the gate
has a nonzero learning direction. The post-encoder `has_text` mask prevents
linear-layer bias from creating an implicit no-text condition.

The projection critic computes an unconditional score plus a masked
surface/text projection. Real surfaces with split-local mismatched text are
used as additional negatives only when text is present.

### Paired two-stage optimization

New fields:

```yaml
initial_generator_checkpoint_path: ""
freeze_backbone_epochs: 5
backbone_learning_rate: 2.0e-6
text_adapter_learning_rate: 2.0e-5
```

Matched and shuffled residual variants load the no-text generator from the same
fold and seed. The surface backbone is frozen for epochs 1-5 and unfrozen at
epoch 6. The critic is always newly initialized. Checkpoints record parent
path/SHA256, transform path/SHA256, architecture version, epoch, and freeze
state.

Legacy article-row FiLM/concat configs and checkpoint keys remain supported.

## Rolling Development Workflow

New files:

```text
configs/film_wgan/train_rq1_pair_textbase.yaml
scripts/rq1_pair/rq1_pair_experiment.py
scripts/rq1_pair/prepare_experiment.sh
scripts/rq1_pair/run_training_matrix.sh
scripts/rq1_pair/resume_training_matrix.sh
scripts/rq1_pair/start_training_matrix_background.sh
scripts/rq1_pair/monitor_training.sh
scripts/rq1_pair/collect_checkpoints.sh
scripts/rq1_pair/run_generate_test_matrix.sh
scripts/rq1_pair/build_comparison_archive.sh
scripts/rq1_pair/package_experiment.sh
```

The four expanding folds use non-overlapping 2023 outer tests:

| Outer test | Train pairs | Validation pairs | Test pairs |
|---|---:|---:|---:|
| 2023Q1 | 1621 | 425 | 521 |
| 2023Q2 | 2046 | 521 | 333 |
| 2023Q3 | 2567 | 333 | 365 |
| 2023Q4 | 2900 | 365 | 378 |

Development seeds:

```text
42, 202, 404
```

Variants:

```text
pair_pca_no_text_residual
pair_pca_text_residual_pretrained
pair_pca_shuffled_residual_pretrained
pair_pca_text_full_film
pair_pca_text_concat
pair_l2_text_full_film
```

This is `4 folds x 3 seeds x 6 variants = 72 runs`. The implementation does
not automatically start this matrix.

## Statistical Outputs

The comparison stage matches predictions by fold, seed, and surface pair. It
produces:

```text
comparisons/development_test_sample_metrics.csv
comparisons/development_model_metrics_by_fold_seed.csv
comparisons/development_pairwise_differences.csv
comparisons/development_fold_seed_differences.csv
comparisons/development_seed_direction_summary.csv
comparisons/development_seed_level_tests.csv
comparisons/development_cluster_bootstrap_ci.csv

final_tables/development_rq1_primary_test.csv
final_tables/development_rq1_point_metric_contrasts.csv
final_tables/development_rq1_metrics_by_fold_seed.csv
final_tables/development_rq1_probabilistic_metrics.csv
final_tables/development_rq1_financial_consistency.csv
```

The primary table uses seed-averaged pair errors and a trading-day cluster
bootstrap. Seed-level paired t-tests and exact Wilcoxon tests are supplementary.
Secondary point metrics receive Holm adjustment within each contrast.

## Verified Results

Commands completed in `py312`:

```bash
python -m unittest tests.test_standalone_wgan.test_film_wgan_module
python -m unittest tests.test_scripts.test_rq1_pair_experiment
python -m compileall -q src/film_wgan scripts/rq1_pair
for f in scripts/rq1_pair/*.sh; do bash -n "$f"; done
git diff --check
```

Observed:

```text
legacy FiLM tests:       27 passed
new pair/RQ1 tests:      10 passed
real rolling counts:      exact for all four folds
PCA dimensions:           1024 -> 128 for every fold
initial SVI Q1 dry-run:   passed on CPU
raw-vol fold counts:      1621/425/521 through 2900/365/378
```

The initial SVI-data transform explained-variance ratios were approximately:

```text
2023Q1 fold: 0.8078
2023Q2 fold: 0.8000
2023Q3 fold: 0.7917
2023Q4 fold: 0.7884
```

## Commands

Prepare a reproducible experiment:

```bash
bash scripts/rq1_pair/prepare_experiment.sh
```

Start the 72-run matrix in the background:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/rq1_pair/start_training_matrix_background.sh
```

Monitor:

```bash
bash scripts/rq1_pair/monitor_training.sh
```

After training:

```bash
bash scripts/rq1_pair/collect_checkpoints.sh
bash scripts/rq1_pair/run_generate_test_matrix.sh
bash scripts/rq1_pair/build_comparison_archive.sh
bash scripts/rq1_pair/package_experiment.sh
```

## Claim Boundary

The three-seed rolling experiment supports model development and temporal
stability analysis. It does not replace the planned five-seed confirmation on
new 2024+ news and option data. The implementation improves nesting and text
utilization but does not guarantee a positive `Delta_text`; any final claim
must follow the observed dependence-aware interval and cross-seed/fold
stability.
