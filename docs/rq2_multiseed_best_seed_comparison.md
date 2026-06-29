# RQ2 Multi-Seed Best-Seed Comparison

This document records the current **best-seed-after-selection** comparison for the RQ2 FiLM WGAN experiments.

本文档记录当前采用的补充分析口径：

```text
1. Train multiple seeds for each model.
2. For each model and each metric, select the seed with the lowest eval MAE.
3. Compare LP text best seed against each baseline best seed using paired sample-level differences.
```

Important caveat:

```text
This is a best-seed-after-selection analysis.
It is useful as a supplementary result, but it is optimistic and should not replace the multi-seed seed-level robustness test.
```

## Experiment Source

Experiment archive:

```text
outputs/experiments/rq2_multiseed_textbase_20260627/
```

Models:

```text
text          = FiLM WGAN with LP text embedding
no_text       = FiLM WGAN without text
bow           = FiLM WGAN with n-gram BoW log-count text features
llm_sentiment = FiLM WGAN with ChatGPT-style multi-dimensional sentiment scores
```

Seeds:

```text
42, 101, 202, 303, 404
```

Split:

```text
eval = final chronological 743 samples
```

Checkpoint rule:

```text
exclude epochs <= 10
select checkpoint with lowest val_mae_gap_vs_current
checkpoint file = film_wgan_best_val_mae_gap_vs_current.pt
```

Source CSV files:

```text
outputs/experiments/rq2_multiseed_textbase_20260627/final_tables/best_seed_by_model_metric_eval.csv
outputs/experiments/rq2_multiseed_textbase_20260627/final_tables/best_seed_pairwise_text_vs_baselines_eval_p_values.csv
outputs/experiments/rq2_multiseed_textbase_20260627/final_tables/best_seed_eval_mae_vs_current_p_values.csv
```

## Methodology

For each model and each metric:

```text
best_seed(model, metric) = argmin_seed mean_eval_MAE(model, seed, metric)
```

The three metrics are:

```text
surface_mae
short_atm_mae
atm7_abs_err
```

Pairwise comparison uses matched eval samples:

```text
diff_i = baseline_best_seed_error_i - text_best_seed_error_i
```

Interpretation:

```text
diff > 0 => text has lower MAE / text is better
diff < 0 => baseline has lower MAE / baseline is better
```

The p-value is a paired one-sample t-test over the 743 eval-sample differences:

```text
H0: mean(diff) = 0
p_text_better_one_sided tests mean(diff) > 0
```

For best-seed vs current baseline:

```text
diff_i = model_error_i - current_baseline_error_i
diff < 0 => model improves over current/persistence baseline
p_model_better_one_sided tests mean(diff) < 0
```

## Best Seed By Model And Metric

All values are eval-split MAE. Lower is better.

| metric | model | best seed | best seed MAE | gap vs current | win rate vs current |
|---|---|---:|---:|---:|---:|
| surface_mae | text | 404 | 0.022863 | -0.002992 | 0.613728 |
| surface_mae | no_text | 101 | 0.022906 | -0.002949 | 0.641992 |
| surface_mae | bow | 404 | 0.022853 | -0.003002 | 0.602961 |
| surface_mae | llm_sentiment | 303 | 0.022909 | -0.002947 | 0.619112 |
| short_atm_mae | text | 42 | 0.017219 | -0.000905 | 0.567968 |
| short_atm_mae | no_text | 202 | 0.017022 | -0.001102 | 0.550471 |
| short_atm_mae | bow | 42 | 0.017332 | -0.000792 | 0.567968 |
| short_atm_mae | llm_sentiment | 303 | 0.017391 | -0.000733 | 0.543742 |
| atm7_abs_err | text | 42 | 0.021374 | -0.001248 | 0.499327 |
| atm7_abs_err | no_text | 202 | 0.021832 | -0.000790 | 0.449529 |
| atm7_abs_err | bow | 42 | 0.021519 | -0.001103 | 0.502019 |
| atm7_abs_err | llm_sentiment | 101 | 0.022152 | -0.000470 | 0.450875 |

Best model by metric under this selected-seed rule:

```text
surface_mae    => bow seed 404, with text seed 404 extremely close
short_atm_mae  => no_text seed 202
atm7_abs_err   => text seed 42
```

## Text Best Seed Vs Baseline Best Seed

Difference definition:

```text
baseline_minus_text_diff = baseline_best_seed_error - text_best_seed_error
positive diff => text lower MAE / better
negative diff => baseline lower MAE / better
```

| metric | comparison | text seed | baseline seed | text MAE | baseline MAE | baseline_minus_text_diff | two-sided p | text better p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| surface_mae | text vs no_text | 404 | 101 | 0.022863 | 0.022906 | 0.000043 | 0.642628 | 0.321314 |
| surface_mae | text vs bow | 404 | 404 | 0.022863 | 0.022853 | -0.000010 | 0.850482 | 0.574759 |
| surface_mae | text vs llm_sentiment | 404 | 303 | 0.022863 | 0.022909 | 0.000045 | 0.604257 | 0.302129 |
| short_atm_mae | text vs no_text | 42 | 202 | 0.017219 | 0.017022 | -0.000197 | 0.006075 | 0.996963 |
| short_atm_mae | text vs bow | 42 | 42 | 0.017219 | 0.017332 | 0.000113 | 0.023898 | 0.011949 |
| short_atm_mae | text vs llm_sentiment | 42 | 303 | 0.017219 | 0.017391 | 0.000172 | 0.074565 | 0.037283 |
| atm7_abs_err | text vs no_text | 42 | 202 | 0.021374 | 0.021832 | 0.000458 | 0.004131 | 0.002066 |
| atm7_abs_err | text vs bow | 42 | 42 | 0.021374 | 0.021519 | 0.000145 | 0.116099 | 0.058049 |
| atm7_abs_err | text vs llm_sentiment | 42 | 101 | 0.021374 | 0.022152 | 0.000778 | 0.000001 | 0.000001 |

Detailed p-value interpretation:

```text
surface_mae:
  text is not significantly different from no_text, bow, or llm_sentiment.

short_atm_mae:
  text best seed is significantly better than bow best seed by one-sided and two-sided tests.
  text best seed has one-sided support against llm_sentiment, but the two-sided p-value is 0.074565.
  no_text best seed is significantly better than text best seed.

atm7_abs_err:
  text best seed is significantly better than no_text best seed.
  text best seed is significantly better than llm_sentiment best seed.
  text best seed is better than bow best seed in mean, but only marginal under one-sided testing and not significant at 5%.
```

## Best Seed Vs Current Baseline

Difference definition:

```text
model_minus_current = model_error - current_baseline_error
negative diff => model lower MAE than current/persistence baseline
```

| metric | model | best seed | model MAE | current MAE | model_minus_current | two-sided p | model better p |
|---|---|---:|---:|---:|---:|---:|---:|
| surface_mae | text | 404 | 0.022863 | 0.025855 | -0.002992 | 4.38e-10 | 2.19e-10 |
| surface_mae | no_text | 101 | 0.022906 | 0.025855 | -0.002949 | 1.56e-10 | 7.81e-11 |
| surface_mae | bow | 404 | 0.022853 | 0.025855 | -0.003002 | 2.65e-10 | 1.33e-10 |
| surface_mae | llm_sentiment | 303 | 0.022909 | 0.025855 | -0.002947 | 2.62e-11 | 1.31e-11 |
| short_atm_mae | text | 42 | 0.017219 | 0.018124 | -0.000905 | 0.000439 | 0.000220 |
| short_atm_mae | no_text | 202 | 0.017022 | 0.018124 | -0.001102 | 9.74e-05 | 4.87e-05 |
| short_atm_mae | bow | 42 | 0.017332 | 0.018124 | -0.000792 | 0.000946 | 0.000473 |
| short_atm_mae | llm_sentiment | 303 | 0.017391 | 0.018124 | -0.000733 | 0.000852 | 0.000426 |
| atm7_abs_err | text | 42 | 0.021374 | 0.022622 | -0.001248 | 0.006396 | 0.003198 |
| atm7_abs_err | no_text | 202 | 0.021832 | 0.022622 | -0.000790 | 0.100466 | 0.050233 |
| atm7_abs_err | bow | 42 | 0.021519 | 0.022622 | -0.001103 | 0.012440 | 0.006220 |
| atm7_abs_err | llm_sentiment | 101 | 0.022152 | 0.022622 | -0.000470 | 0.309928 | 0.154964 |

Under the selected-seed rule, the best seeds for all four model families improve over the current baseline for `surface_mae` and `short_atm_mae`. For `atm7_abs_err`, the strongest current-baseline improvements are from `text seed 42` and `bow seed 42`.

## Thesis-Facing Interpretation

This best-seed analysis supports a narrow, metric-specific text advantage:

```text
On the eval split, after selecting each model's best seed per metric, LP text achieves the lowest 7d ATM absolute error.
```

For 7d ATM:

```text
text seed 42 MAE = 0.021374
no_text best seed MAE = 0.021832, p_text_better = 0.002066
bow best seed MAE = 0.021519, p_text_better = 0.058049
llm_sentiment best seed MAE = 0.022152, p_text_better = 6.23e-07
```

For short-ATM:

```text
text seed 42 is better than bow seed 42 and has one-sided support over llm_sentiment seed 303.
However, no_text seed 202 is significantly better than text seed 42.
```

For full-surface MAE:

```text
text seed 404 is close to the best result, but none of the text-vs-baseline differences are statistically significant.
```

Final caution for paper writing:

```text
Because the seed is selected using eval MAE, this analysis is selection-biased and should be presented as supplementary.
The primary robust comparison should still report the multi-seed seed-level paired tests.
```

Suggested wording:

```text
As a supplementary best-seed analysis, LP text gives the strongest 7d ATM performance on the chronological evaluation split. However, this selected-seed result should be interpreted cautiously because model seeds are chosen after observing eval MAE; the multi-seed seed-level tests remain the more conservative evidence.
```
