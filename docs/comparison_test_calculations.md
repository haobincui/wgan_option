# FiLM WGAN Comparison Test Calculations

This document consolidates the completed comparison tests for the thesis-facing FiLM WGAN experiments. It records what was compared, how the paired differences were defined, how p-values and confidence intervals were calculated, and where the full CSV outputs are stored.

本文档不是新的实验计划，而是对已经完成的对比测试做统一记录，方便后续论文写作时追溯每个 statistic 的来源和方向。

## 1. Scope and Source Archives

The comparison tests cover four model variants:

```text
text          = FiLM WGAN with LP text embedding
no_text       = FiLM WGAN without text
bow           = FiLM WGAN with n-gram BoW log-count text features
llm_sentiment = FiLM WGAN with ChatGPT-style multi-dimensional sentiment scores
```

Important representation correction:

```text
bow           = unigram/bigram n-gram frequency with log1p count weighting
llm_sentiment = Sun-style ChatGPT score vector, not Loughran-McDonald dictionary sentiment
```

For the 2026-06-27 BoW / sentiment reruns, the enriched workbook used by training was:

```text
data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx
```

and the feature source matched:

```text
data/processed/text_features/rq2/20260625-075653/
```

The `20260623-rq2` feature directory is not the data lineage for the 2026-06-27 reruns.

The paper-facing checkpoint rule is:

```text
exclude epochs <= 10
select the after-warmup checkpoint by validation MAE gap vs current baseline
```

Main comparison archives:

```text
outputs/comparison/film_wgan_20260417_131244_vs_notext_20260623_083328_after10_epoch12/
outputs/comparison/film_wgan_text_20260417_131244_vs_bow_llm_sentiment_20260623_154232_after10_val_mae/
outputs/comparison/four_model_mae_self_tests_after10_current_baseline/
```

The chronological split is shared across the generated all-sample result files:

```text
train = first 2968 samples
eval  = final 743 samples
all   = all 3711 samples
```

The evaluation split is the main non-leakage split for thesis claims because it is chronologically later than the training samples.

## 2. Metric Definitions

All metrics are lower-is-better error metrics.

### 2.1 Surface MAE

`surface_mae` is the mean absolute error over the full volatility-surface grid:

```text
surface_mae_i = mean(|generated_surface_i - target_surface_i|)
```

The current / persistence baseline is:

```text
current_surface_i -> target_surface_i
current_surface_mae_i = mean(|current_surface_i - target_surface_i|)
```

### 2.2 Short-ATM MAE

`short_atm_mae` is the mean absolute error over the short-maturity near-ATM band:

```text
abs(strike - 1.0) <= 0.06
maturity_days <= 90
```

For the current grid this includes the near-ATM strikes around 1.0 and maturities up to 90 days. It is a band-level metric, not a single grid point.

### 2.3 7d ATM Absolute Error

`atm7_abs_err` / `atm_7d_err` is the absolute error at the nearest ATM strike and shortest maturity:

```text
nearest ATM strike = 0.9800000190734863
shortest maturity  = 7.0 days
```

It is a single-cell metric:

```text
atm7_abs_err_i = |generated_vol_i(7d, nearest_ATM) - target_vol_i(7d, nearest_ATM)|
```

### 2.4 Gap Metrics

Gap metrics subtract the current baseline error from the model error:

```text
surface_gap_i   = model_surface_mae_i - current_surface_mae_i
short_atm_gap_i = model_short_atm_mae_i - current_short_atm_mae_i
atm7_gap_i      = model_atm7_abs_err_i - current_atm7_abs_err_i
```

Negative gap means the model improves over the current / persistence baseline.

For model-vs-model paired differences, the gap rows have the same paired difference as the corresponding MAE rows when the current baseline is shared between the two models.

## 3. Paired t-test Logic

All t-tests are paired one-sample t-tests over per-sample differences.

For a paired difference vector:

```text
d_i = error_A_i - error_B_i
```

the t-statistic is:

```text
t = mean(d) / (std(d, ddof=1) / sqrt(n))
df = n - 1
```

The outputs store:

```text
p_value_two_sided
p_value_*_lower_one_sided
t_ci_95_low
t_ci_95_high
```

The one-sided p-value direction depends on the difference definition in each table. The direction is documented explicitly below.

## 4. Bootstrap Logic

Bootstrap tests use paired resampling at the sample level. For each model comparison or model-vs-current self-test:

```text
resampling unit = one matched sample
bootstrap_iterations = 10000
seed = 20260625
CI = percentile CI of the uncentered bootstrap mean-difference distribution
```

For the four-model self-tests, bootstrap p-values are computed from a centered bootstrap null distribution. The stored p-values therefore test whether the observed mean difference is extreme under:

```text
H0: mean(difference) = 0
```

## 5. Text vs No-text Paired Comparison

Full output:

```text
outputs/comparison/film_wgan_20260417_131244_vs_notext_20260623_083328_after10_epoch12/t_tests/paired_t_test_p_values_text_vs_notext_after10_epoch12.csv
outputs/comparison/film_wgan_20260417_131244_vs_notext_20260623_083328_after10_epoch12/t_tests/paired_t_test_p_values_compact_main_metrics_text_vs_notext_after10_epoch12.csv
```

Difference definition:

```text
difference_definition = notext_minus_text
d_i = no_text_error_i - text_error_i
```

Interpretation:

```text
d_i > 0  => text has lower error for that sample
d_i < 0  => no_text has lower error for that sample
```

Main paired t-test rows:

| split | metric | text mean | no-text mean | mean(no-text - text) | t | two-sided p | text-lower one-sided p | interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| train | surface MAE | 0.027644 | 0.028089 | 0.000445 | 7.002 | 3.11e-12 | 1.56e-12 | text better |
| train | short-ATM MAE | 0.019546 | 0.020001 | 0.000455 | 9.913 | 8.20e-23 | 4.10e-23 | text better |
| train | 7d ATM abs err | 0.024487 | 0.025176 | 0.000689 | 7.266 | 4.73e-13 | 2.36e-13 | text better |
| eval | surface MAE | 0.023450 | 0.023484 | 0.000034 | 0.392 | 0.695 | 0.348 | not significant |
| eval | short-ATM MAE | 0.017196 | 0.017225 | 0.000028 | 0.396 | 0.692 | 0.346 | not significant |
| eval | 7d ATM abs err | 0.021383 | 0.022159 | 0.000776 | 4.668 | 3.61e-06 | 1.81e-06 | text better |
| all | surface MAE | 0.026804 | 0.027167 | 0.000363 | 6.746 | 1.75e-11 | 8.77e-12 | text better |
| all | short-ATM MAE | 0.019076 | 0.019445 | 0.000370 | 9.352 | 1.45e-20 | 7.25e-21 | text better |
| all | 7d ATM abs err | 0.023866 | 0.024572 | 0.000706 | 8.530 | 2.10e-17 | 1.05e-17 | text better |

Thesis interpretation:

- After applying the after-warmup checkpoint rule, LP text is significantly better than no-text on eval 7d ATM error.
- On eval surface MAE and short-ATM MAE, text has a slightly lower mean error, but the paired t-tests do not reject equality.
- On train and all, text is significantly better across the three main error metrics.

## 6. Text vs BoW and LLM Sentiment Paired Comparisons

Full t-test output:

```text
outputs/comparison/film_wgan_text_20260417_131244_vs_bow_llm_sentiment_20260623_154232_after10_val_mae/t_tests/paired_t_test_p_values_compact_main_metrics.csv
```

Full bootstrap output:

```text
outputs/comparison/film_wgan_text_20260417_131244_vs_bow_llm_sentiment_20260623_154232_after10_val_mae/bootstrap_ci/paired_difference_bootstrap_ci_compact_main_metrics.csv
```

Difference definitions:

```text
text_vs_bow:
  d_i = bow_error_i - text_error_i

text_vs_llm_sentiment:
  d_i = llm_sentiment_error_i - text_error_i
```

Interpretation:

```text
d_i > 0  => text has lower error
d_i < 0  => variant has lower error
```

### 6.1 Paired t-test Results

| comparison | split | metric | mean difference | t | two-sided p | lower-error one-sided result |
|---|---|---:|---:|---:|---:|---|
| text vs bow | train | surface MAE | -0.000508 | -5.482 | 4.55e-08 | BoW better |
| text vs bow | train | short-ATM MAE | -0.000744 | -7.011 | 2.92e-12 | BoW better |
| text vs bow | train | 7d ATM abs err | -0.002627 | -12.953 | 2.28e-37 | BoW better |
| text vs bow | eval | surface MAE | -0.000220 | -1.569 | 0.117 | not significant |
| text vs bow | eval | short-ATM MAE | 0.000591 | 3.880 | 0.000114 | text better |
| text vs bow | eval | 7d ATM abs err | 0.001002 | 2.350 | 0.0190 | text better |
| text vs bow | all | surface MAE | -0.000450 | -5.682 | 1.44e-08 | BoW better |
| text vs bow | all | short-ATM MAE | -0.000477 | -5.262 | 1.51e-07 | BoW better |
| text vs bow | all | 7d ATM abs err | -0.001901 | -10.283 | 1.78e-24 | BoW better |
| text vs llm_sentiment | train | surface MAE | 0.000480 | 9.128 | 1.26e-19 | text better |
| text vs llm_sentiment | train | short-ATM MAE | 0.000581 | 7.057 | 2.11e-12 | text better |
| text vs llm_sentiment | train | 7d ATM abs err | 0.000140 | 1.441 | 0.150 | not significant |
| text vs llm_sentiment | eval | surface MAE | 0.000075 | 1.089 | 0.276 | not significant |
| text vs llm_sentiment | eval | short-ATM MAE | -0.000059 | -0.467 | 0.641 | not significant |
| text vs llm_sentiment | eval | 7d ATM abs err | -0.000374 | -1.908 | 0.0568 | llm_sentiment better only by one-sided t-test |
| text vs llm_sentiment | all | surface MAE | 0.000399 | 8.995 | 3.67e-19 | text better |
| text vs llm_sentiment | all | short-ATM MAE | 0.000453 | 6.404 | 1.70e-10 | text better |
| text vs llm_sentiment | all | 7d ATM abs err | 0.000037 | 0.422 | 0.673 | not significant |

### 6.2 Bootstrap CI Results

Bootstrap rows use the same paired difference definitions as the t-tests above.

| comparison | split | metric | mean difference | 95% bootstrap CI | CI excludes 0 | interpretation |
|---|---|---:|---:|---|---:|---|
| text vs bow | train | surface MAE | -0.000508 | [-0.000689, -0.000330] | yes | BoW better |
| text vs bow | train | short-ATM MAE | -0.000744 | [-0.000956, -0.000543] | yes | BoW better |
| text vs bow | train | 7d ATM abs err | -0.002627 | [-0.003041, -0.002226] | yes | BoW better |
| text vs bow | eval | surface MAE | -0.000220 | [-0.000491, 0.000057] | no | not resolved |
| text vs bow | eval | short-ATM MAE | 0.000591 | [0.000282, 0.000884] | yes | text better |
| text vs bow | eval | 7d ATM abs err | 0.001002 | [0.000079, 0.001715] | yes | text better |
| text vs llm_sentiment | train | surface MAE | 0.000480 | [0.000372, 0.000578] | yes | text better |
| text vs llm_sentiment | train | short-ATM MAE | 0.000581 | [0.000414, 0.000740] | yes | text better |
| text vs llm_sentiment | train | 7d ATM abs err | 0.000140 | [-0.000056, 0.000331] | no | not resolved |
| text vs llm_sentiment | eval | surface MAE | 0.000075 | [-0.000062, 0.000212] | no | not resolved |
| text vs llm_sentiment | eval | short-ATM MAE | -0.000059 | [-0.000312, 0.000190] | no | not resolved |
| text vs llm_sentiment | eval | 7d ATM abs err | -0.000374 | [-0.000781, -0.000011] | yes | llm_sentiment lower by percentile CI |

Thesis interpretation:

- BoW is strong on train and all, but on eval the LP text model is significantly better for short-ATM MAE and 7d ATM error.
- For eval 7d ATM, `llm_sentiment` has a lower point estimate than LP text. The two-sided paired t-test is borderline (`p = 0.0568`), while the one-sided t-test and percentile bootstrap support a lower `llm_sentiment` error. This should be described carefully rather than as a robust two-sided result.

## 7. Four-Model Self Tests vs Current Baseline

This test family is not a pairwise model comparison. It tests whether each model is significantly better than the current / persistence baseline.

Full output:

```text
outputs/comparison/four_model_mae_self_tests_after10_current_baseline/unified_sample_mae_metrics.csv
outputs/comparison/four_model_mae_self_tests_after10_current_baseline/model_vs_current_paired_t_tests.csv
outputs/comparison/four_model_mae_self_tests_after10_current_baseline/model_vs_current_bootstrap_tests.csv
outputs/comparison/four_model_mae_self_tests_after10_current_baseline/model_vs_current_mae_self_test_compact.csv
```

Difference definition:

```text
d_i = model_error_i - current_baseline_error_i
```

Interpretation:

```text
d_i < 0 => model improves over current baseline
d_i > 0 => current baseline is better
```

The `current_mean` column is:

```text
mean(current_baseline_error_i)
```

for the same model, split, and metric row. It is computed from `current_surface` vs `target_surface`, using the same sample IDs as the model row. Because all models are evaluated on the same 3711 samples, `current_mean` is shared for a given split and metric, aside from floating-point alignment checks.

### 7.1 Compact Results

| model | split | metric | model mean | current mean | mean(model-current) | t-test two-sided p | model-lower one-sided p | bootstrap 95% CI | bootstrap one-sided p |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| text | train | surface MAE | 0.027644 | 0.030116 | -0.002472 | 8.41e-17 | 4.21e-17 | [-0.003056, -0.001912] | 1.00e-04 |
| text | train | short-ATM MAE | 0.019546 | 0.021261 | -0.001715 | 4.44e-25 | 2.22e-25 | [-0.002041, -0.001393] | 1.00e-04 |
| text | train | 7d ATM abs err | 0.024487 | 0.025454 | -0.000967 | 1.85e-07 | 9.26e-08 | [-0.001333, -0.000602] | 1.00e-04 |
| text | eval | surface MAE | 0.023450 | 0.025855 | -0.002405 | 3.28e-07 | 1.64e-07 | [-0.003340, -0.001519] | 1.00e-04 |
| text | eval | short-ATM MAE | 0.017196 | 0.018124 | -0.000928 | 2.85e-04 | 1.42e-04 | [-0.001426, -0.000420] | 3.00e-04 |
| text | eval | 7d ATM abs err | 0.021383 | 0.022622 | -0.001239 | 0.00662 | 0.00331 | [-0.002221, -0.000452] | 0.0101 |
| no_text | eval | surface MAE | 0.023484 | 0.025855 | -0.002371 | 1.90e-06 | 9.49e-07 | [-0.003360, -0.001398] | 1.00e-04 |
| no_text | eval | short-ATM MAE | 0.017225 | 0.018124 | -0.000900 | 0.00114 | 5.69e-04 | [-0.001437, -0.000360] | 5.00e-04 |
| no_text | eval | 7d ATM abs err | 0.022159 | 0.022622 | -0.000463 | 0.293 | 0.146 | [-0.001407, 0.000280] | 0.143 |
| bow | eval | surface MAE | 0.023230 | 0.025855 | -0.002625 | 2.91e-08 | 1.45e-08 | [-0.003551, -0.001721] | 1.00e-04 |
| bow | eval | short-ATM MAE | 0.017787 | 0.018124 | -0.000337 | 0.285 | 0.143 | [-0.000977, 0.000279] | 0.141 |
| bow | eval | 7d ATM abs err | 0.022385 | 0.022622 | -0.000237 | 0.767 | 0.384 | [-0.002025, 0.001038] | 0.347 |
| llm_sentiment | eval | surface MAE | 0.023525 | 0.025855 | -0.002330 | 5.94e-07 | 2.97e-07 | [-0.003238, -0.001447] | 1.00e-04 |
| llm_sentiment | eval | short-ATM MAE | 0.017137 | 0.018124 | -0.000987 | 6.38e-04 | 3.19e-04 | [-0.001568, -0.000436] | 4.00e-04 |
| llm_sentiment | eval | 7d ATM abs err | 0.021008 | 0.022622 | -0.001614 | 0.00459 | 0.00229 | [-0.002847, -0.000677] | 0.00760 |

The compact CSV contains all 36 rows:

```text
4 models x 3 splits x 3 metrics = 36 rows
```

The table above highlights all eval rows plus the text train rows, because eval is the main no-leakage split and text is the primary paper model. The remaining train/all rows are preserved in the compact CSV with the same columns and calculation logic.

### 7.2 Self-Test Interpretation

On the eval split:

- `text` significantly improves over current baseline for surface MAE, short-ATM MAE, and 7d ATM error.
- `no_text` significantly improves over current baseline for surface MAE and short-ATM MAE, but not for 7d ATM error.
- `bow` significantly improves over current baseline for surface MAE, but not for short-ATM MAE or 7d ATM error.
- `llm_sentiment` significantly improves over current baseline for all three eval metrics.

This self-test answers a different question from pairwise model comparisons. A model can be significantly better than current baseline while still not being significantly better than another model.

## 8. Thesis-Facing Reading of 7d ATM

If the claim is restricted to the eval split and the 7d nearest-ATM metric:

- LP text is significantly better than no-text in the paired model-vs-model test:

```text
mean(no_text - text) = 0.000776
two-sided p = 3.61e-06
text-lower one-sided p = 1.81e-06
```

- LP text is significantly better than BoW in the paired model-vs-model test:

```text
mean(bow - text) = 0.001002
two-sided p = 0.0190
text-lower one-sided p = 0.00952
```

- LLM sentiment has a lower eval 7d ATM point estimate than LP text:

```text
text mean          = 0.021383
llm_sentiment mean = 0.021008
mean(llm_sentiment - text) = -0.000374
two-sided p = 0.0568
llm-lower one-sided p = 0.0284
```

Therefore the most defensible wording is:

```text
On the eval split, LP text FiLM WGAN significantly outperforms no-text and BoW on 7d nearest-ATM error. LLM sentiment has a slightly lower 7d ATM point estimate, but the two-sided paired t-test is borderline rather than conventionally significant.
```

It is not statistically clean to claim that LP text is better than all three alternatives on eval 7d ATM, because the LLM sentiment comparison does not support that direction.

## 9. Validation and Audit Trail

The four-model self-test archive records the validation checks:

```text
outputs/comparison/four_model_mae_self_tests_after10_current_baseline/validation_summary.csv
outputs/comparison/four_model_mae_self_tests_after10_current_baseline/text_source_consistency_check.csv
outputs/comparison/four_model_mae_self_tests_after10_current_baseline/source_manifest.csv
```

Validation facts:

```text
text sample count          = 3711
no_text sample count       = 3711
bow sample count           = 3711
llm_sentiment sample count = 3711

train rows per model = 2968
eval rows per model  = 743
all rows per model   = 3711

compact self-test rows = 36
text source consistency max diff = 0.0
```

The `text_source_consistency_check.csv` confirms that the LP text sample metrics used in the text/no-text archive and in the RQ2 BoW/LLM archive match exactly for the checked main metrics. This prevents the four-model table from mixing incompatible LP text outputs.

## 10. Output File Map

Text vs no-text:

```text
outputs/comparison/film_wgan_20260417_131244_vs_notext_20260623_083328_after10_epoch12/final_comparison/
outputs/comparison/film_wgan_20260417_131244_vs_notext_20260623_083328_after10_epoch12/t_tests/
```

Text vs BoW / LLM sentiment:

```text
outputs/comparison/film_wgan_text_20260417_131244_vs_bow_llm_sentiment_20260623_154232_after10_val_mae/final_comparison/
outputs/comparison/film_wgan_text_20260417_131244_vs_bow_llm_sentiment_20260623_154232_after10_val_mae/bootstrap_ci/
outputs/comparison/film_wgan_text_20260417_131244_vs_bow_llm_sentiment_20260623_154232_after10_val_mae/t_tests/
```

Four-model self tests vs current baseline:

```text
outputs/comparison/four_model_mae_self_tests_after10_current_baseline/
```

Related RQ2 documentation:

```text
docs/rq2_bow_film_wgan.md
docs/rq2_llm_sentiment_film_wgan.md
```
