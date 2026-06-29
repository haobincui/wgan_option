# RQ2 N-gram Log-count BoW FiLM WGAN Baseline

This note documents the completed RQ2 BoW experiment. 目标是说明 BoW n-gram log-count text representation 如何计算、如何并入 `merged_vol` training workbook、如何训练 FiLM WGAN，以及本次 after-warmup comparison 的结论。This is a thesis-facing reproducibility record, not a new experiment proposal.

## 1. Research Role

RQ2 asks whether LLM-based text representations contain more useful information than traditional text-mining baselines. The BoW baseline is designed as a controlled representation ablation:

- Downstream model is unchanged: same FiLM WGAN architecture, surface inputs, losses, split, and evaluation protocol.
- Only the text representation changes: `text_embedding_mode=bow` reads `bow_embedding` instead of LP embedding.
- The baseline therefore tests whether a high-dimensional unigram/bigram log-count representation of the same `LP` text can compete with the original LP-text FiLM WGAN.

## 2. Feature Calculation Logic

Raw input:

```text
data/raw/text_embedding/news_with_openai_embeddings_large.xlsx
```

The BoW feature generation uses the `LP` text column:

```bash
python scripts/generate_rq2_text_features.py \
  --news-xlsx data/raw/text_embedding/news_with_openai_embeddings_large.xlsx \
  --output-dir data/processed/text_features/rq2/20260625-075653 \
  --text-column LP \
  --target-dim 1024 \
  --model gpt-5.4-mini
```

BoW-specific implementation lives in `src/bow/features.py`:

- Tokenization uses the regex `[A-Za-z][A-Za-z'-]*`, lower-cased.
- The representation uses unigram and bigram n-grams, `ngram_range=(1, 2)`.
- The vocabulary is the top `target_dim=1024` n-grams by corpus frequency, tie-broken lexicographically.
- Each feature value is `log1p(count)` for that n-gram in the article.
- The output is aligned to exactly 1024 dimensions.
- Each vector is serialized as JSON text in `bow_embedding`.

Output artifact:

```text
data/processed/text_features/rq2/20260625-075653/bow_features.xlsx
```

Output columns:

```text
news_row_id
article_id
source_file
bow_embedding
bow_dim
```

Actual manifest facts from `bow_manifest.json`:

```text
row_count       = 14900
target_dim      = 1024
ngram_range     = [1, 2]
backend         = python_counter
representation  = ngram_frequency
weighting       = log1p_count
vocabulary_size = 1024
```

The fitted preprocessing artifacts are also saved:

```text
data/processed/text_features/rq2/20260625-075653/bow_vocabulary.json
```

## 3. Workbook Enrichment

The RQ2 feature workbook is merged into the vol-surface training workbook with:

```bash
python scripts/rq2/enrich_merged_vol.py \
  --merged-vol data/processed/svi-excel/20260410-174929/merged_vol.xlsx \
  --bow-features data/processed/text_features/rq2/20260625-075653/bow_features.xlsx \
  --sentiment-features data/processed/text_features/rq2/20260625-075653/llm_sentiment_features.xlsx \
  --output data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx
```

The enrichment script preserves all original workbook sheets. It joins feature rows using `news_row_id` when available. If a sheet only has `sample_id`, it parses IDs of the form `news_<id>` and uses that numeric ID as the join key. For the training sheet `gan_input_ready`, missing RQ2 feature matches are treated as an error.

Final training data path:

```text
data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx
```

## 4. Training Protocol

Launcher:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/rq2/run_film_wgan_bow.sh
```

Resolved training config:

```text
outputs/training/film_wgan/bow/20260623_154232/metrics/training_resolved_config.yaml
```

Important resolved settings:

```text
data_path            = data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx
sheet_name           = gan_input_ready
text_embedding_mode  = bow
train_ratio          = 0.8
normalize_text_embedding = true
output_root          = outputs/training/film_wgan/bow
```

The chronological split is:

```text
train = first 2968 samples
eval  = final 743 samples
all   = 3711 samples
```

Model and loss settings are inherited from `configs/film_wgan/train_lp_exp_F5.yaml`:

```text
noise_dim              = 128
gen_base_channels      = 112
disc_base_channels     = 112
gen_res_blocks         = 4
disc_res_blocks        = 2
text_hidden_dim        = 768
text_out_dim           = 384
fusion_hidden_dim      = 2048
batch_size             = 128
critic_iter            = 5
lambda_gp              = 20.0
lambda_adv             = 0.1
adv_warmup_epochs      = 10
lambda_recon           = 20.0
recon_weight_mode      = short_atm_band
recon_atm_range        = 0.06
recon_atm_short_end_max_days = 90.0
recon_atm_multiplier   = 8.0
use_atm_short_loss     = true
lambda_atm_short       = 50.0
atm_short_range        = 0.04
atm_short_max_days     = 60.0
```

The FiLM WGAN data loader reads `bow_embedding` through `src/film_wgan/data.py` when `text_embedding_mode=bow`. It requires a non-empty `bow_embedding` column and parses the serialized JSON list into a `float32` vector.

## 5. Checkpoint and Generate-Result Process

Training output:

```text
outputs/training/film_wgan/bow/20260623_154232
```

The run stopped at epoch 18 because early stopping was enabled. Training-time best checkpoint tracking used:

```text
primary metric = val_atm_short_pure_mae_gap_vs_current
checkpoint_warmup_epochs = 5
selection_start_epoch = 6
```

Training primary best:

```text
epoch 11
val_atm_short_pure_mae_gap_vs_current = -0.0007551380529972149
checkpoint = outputs/training/film_wgan/bow/20260623_154232/checkpoints/film_wgan_best.pt
```

For the paper-facing comparison, the archive enforces the stronger warmup rule requested later:

```text
eligible epochs: epoch > 10
selection metric: lowest val_mae_gap_vs_current
```

Selected paper-facing checkpoint:

```text
epoch 14
val_mae_gap_vs_current = -0.002624642565733616
checkpoint = outputs/training/film_wgan/bow/20260623_154232/checkpoints/film_wgan_best_val_mae_gap_vs_current.pt
```

Generate-result was run on the full chronological sample set:

```bash
conda run -n py312 python scripts/film_wgan/main.py generate-result \
  --config outputs/training/film_wgan/bow/20260623_154232/metrics/training_resolved_config.yaml \
  --checkpoint outputs/training/film_wgan/bow/20260623_154232/checkpoints/film_wgan_best_val_mae_gap_vs_current.pt \
  --output-dir after10_val_mae_all_json \
  --split all \
  --selection-mode all \
  --selection-count 0 \
  --no-plot
```

Generated outputs:

```text
outputs/training/film_wgan/bow/20260623_154232/after10_val_mae_all_json/
```

The final comparison archive is:

```text
outputs/comparison/film_wgan_text_20260417_131244_vs_bow_llm_sentiment_20260623_154232_after10_val_mae/
```

## 6. Result Summary

Source table:

```text
outputs/comparison/film_wgan_text_20260417_131244_vs_bow_llm_sentiment_20260623_154232_after10_val_mae/final_comparison/thesis_compact_table_after10_val_mae.csv
```

Lower is better for all MAE and gap metrics. For gap metrics, more negative means larger improvement over the current-surface persistence baseline.

| split | metric | text | bow | llm_sentiment | bow_minus_text | llm_sentiment_minus_text | best_lower_is_better |
|:--|:--|--:|--:|--:|--:|--:|:--|
| train | surface_mae | 0.027644 | 0.027136 | 0.028124 | -0.000508 | 0.000480 | bow |
| train | surface_gap | -0.002472 | -0.002980 | -0.001992 | -0.000508 | 0.000480 | bow |
| train | short_atm_mae | 0.019546 | 0.018802 | 0.020127 | -0.000744 | 0.000581 | bow |
| train | short_atm_gap | -0.001715 | -0.002459 | -0.001134 | -0.000744 | 0.000581 | bow |
| train | atm7_abs_err | 0.024487 | 0.021860 | 0.024627 | -0.002627 | 0.000140 | bow |
| train | atm7_gap | -0.000967 | -0.003594 | -0.000827 | -0.002627 | 0.000140 | bow |
| eval | surface_mae | 0.023450 | 0.023230 | 0.023525 | -0.000220 | 0.000075 | bow |
| eval | surface_gap | -0.002405 | -0.002625 | -0.002330 | -0.000220 | 0.000075 | bow |
| eval | short_atm_mae | 0.017196 | 0.017787 | 0.017137 | 0.000591 | -0.000059 | llm_sentiment |
| eval | short_atm_gap | -0.000928 | -0.000337 | -0.000987 | 0.000591 | -0.000059 | llm_sentiment |
| eval | atm7_abs_err | 0.021383 | 0.022385 | 0.021008 | 0.001002 | -0.000374 | llm_sentiment |
| eval | atm7_gap | -0.001239 | -0.000237 | -0.001614 | 0.001002 | -0.000374 | llm_sentiment |
| all | surface_mae | 0.026804 | 0.026354 | 0.027203 | -0.000450 | 0.000399 | bow |
| all | surface_gap | -0.002459 | -0.002909 | -0.002060 | -0.000450 | 0.000399 | bow |
| all | short_atm_mae | 0.019076 | 0.018599 | 0.019528 | -0.000477 | 0.000453 | bow |
| all | short_atm_gap | -0.001558 | -0.002034 | -0.001105 | -0.000477 | 0.000453 | bow |
| all | atm7_abs_err | 0.023866 | 0.021965 | 0.023902 | -0.001901 | 0.000037 | bow |
| all | atm7_gap | -0.001022 | -0.002922 | -0.000985 | -0.001901 | 0.000037 | bow |

BoW is strongest on the `train` and `all` splits for the three main paper-facing metrics:

- `surface_mae`
- `short_atm_mae`
- `atm7_abs_err`

On `eval`, BoW has the lowest full-surface MAE, but it is worse than LP text and LLM sentiment on short-ATM MAE and 7d ATM absolute error.

## 7. Bootstrap Confidence Intervals

Source table:

```text
outputs/comparison/film_wgan_text_20260417_131244_vs_bow_llm_sentiment_20260623_154232_after10_val_mae/bootstrap_ci/paired_difference_bootstrap_ci_compact_main_metrics.csv
```

Bootstrap method:

```text
paired percentile bootstrap
resampling unit = matched global_index / sample_id
iterations = 10000
seed = 20260625
difference = bow - text
```

For error metrics, negative difference means BoW is lower/better than LP text.

| split | metric | bow_minus_text | 95% CI low | 95% CI high | interpretation |
|:--|:--|--:|--:|--:|:--|
| train | surface_mae | -0.000508 | -0.000689 | -0.000330 | BoW better; CI excludes zero |
| train | short_atm_mae | -0.000744 | -0.000956 | -0.000543 | BoW better; CI excludes zero |
| train | atm7_abs_err | -0.002627 | -0.003041 | -0.002226 | BoW better; CI excludes zero |
| eval | surface_mae | -0.000220 | -0.000491 | 0.000057 | CI includes zero |
| eval | short_atm_mae | 0.000591 | 0.000282 | 0.000884 | LP text better; CI excludes zero |
| eval | atm7_abs_err | 0.001002 | 0.000079 | 0.001715 | LP text better; CI excludes zero |
| all | surface_mae | -0.000450 | -0.000606 | -0.000294 | BoW better; CI excludes zero |
| all | short_atm_mae | -0.000477 | -0.000654 | -0.000298 | BoW better; CI excludes zero |
| all | atm7_abs_err | -0.001901 | -0.002275 | -0.001553 | BoW better; CI excludes zero |

## 8. Paired t-test P-values

Source table:

```text
outputs/comparison/film_wgan_text_20260417_131244_vs_bow_llm_sentiment_20260623_154232_after10_val_mae/t_tests/paired_t_test_p_values_compact_main_metrics.csv
```

The paired t-test is computed on:

```text
difference = metric_bow - metric_text
H0: mean(difference) = 0
```

Rows are paired by `global_index` and `sample_id`.

| split | metric | bow_minus_text | t statistic | p two-sided | one-sided p: BoW lower | one-sided p: text lower |
|:--|:--|--:|--:|--:|--:|--:|
| train | surface_mae | -0.000508 | -5.48225 | 4.55185e-08 | 2.27593e-08 | 1 |
| train | short_atm_mae | -0.000744 | -7.01068 | 2.92428e-12 | 1.46214e-12 | 1 |
| train | atm7_abs_err | -0.002627 | -12.9531 | 2.28136e-37 | 1.14068e-37 | 1 |
| eval | surface_mae | -0.000220 | -1.56899 | 0.117078 | 0.0585388 | 0.941461 |
| eval | short_atm_mae | 0.000591 | 3.88049 | 0.000113518 | 0.999943 | 5.67592e-05 |
| eval | atm7_abs_err | 0.001002 | 2.34984 | 0.0190427 | 0.990479 | 0.00952137 |
| all | surface_mae | -0.000450 | -5.68155 | 1.43745e-08 | 7.18725e-09 | 1 |
| all | short_atm_mae | -0.000477 | -5.26167 | 1.50831e-07 | 7.54153e-08 | 1 |
| all | atm7_abs_err | -0.001901 | -10.2833 | 1.78111e-24 | 8.90554e-25 | 1 |

The t-test agrees with the bootstrap summary:

- Train: BoW is significantly better than LP text on all three main metrics.
- Eval: BoW's surface MAE advantage is not statistically resolved; LP text is significantly better on short-ATM and 7d ATM.
- All: BoW is significantly better than LP text on all three main metrics.

## 9. Thesis Interpretation

BoW is not merely a weak traditional baseline in this run. Under the final after-10 `val_mae_gap_vs_current` checkpoint rule, BoW gives the best aggregate `all` performance across the main metrics. However, the eval split shows a more nuanced pattern: the full-surface result slightly favors BoW but is not significant, while short-ATM and 7d ATM favor LP text.

For thesis writing, the safest interpretation is:

```text
N-gram log-count BoW captures useful lexical information and can outperform the LP embedding on aggregate surface metrics in this experiment, but its advantage is not uniform across out-of-sample short-end ATM diagnostics.
```

This means RQ2 should not be framed as a simple monotonic ordering where LLM embedding always dominates traditional text-mining. The evidence supports a representation-dependent tradeoff: BoW is strong on broad aggregate metrics, while LP text remains competitive or better on the eval short-end ATM region.
