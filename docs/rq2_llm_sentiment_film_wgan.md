# RQ2 Loughran-McDonald Sentiment FiLM WGAN Baseline

This note documents the completed RQ2 LLM-sentiment experiment. Here `llm_sentiment` means a Loughran-McDonald dictionary sentiment feature baseline, not a generative LLM inference step. The purpose is to record the feature calculation logic, FiLM WGAN training process, checkpoint selection, generate-result procedure, and statistical comparison results.

## 1. Research Role

The sentiment baseline is a traditional text-mining baseline for RQ2. It asks whether a compact financial dictionary sentiment representation can provide useful text signal compared with the original LP-text FiLM WGAN.

Controlled design:

- Same downstream FiLM WGAN architecture as LP text and BoW.
- Same paired volatility-surface samples and chronological split.
- Same loss configuration and evaluation protocol.
- Only the text representation changes: `text_embedding_mode=llm_sentiment` reads `sentiment_embedding`.

## 2. Feature Calculation Logic

Raw input:

```text
data/raw/text_embedding/news_with_openai_embeddings_large.xlsx
```

Text column:

```text
LP
```

Dictionary:

```text
data/reference/Loughran-McDonald_MasterDictionary_1993-2025.csv
```

Feature-generation command:

```bash
python scripts/generate_rq2_text_features.py \
  --news-xlsx data/raw/text_embedding/news_with_openai_embeddings_large.xlsx \
  --output-dir data/processed/text_features/rq2/20260623-rq2 \
  --text-column LP \
  --target-dim 1024 \
  --dictionary-path data/reference/Loughran-McDonald_MasterDictionary_1993-2025.csv
```

Sentiment implementation lives in `src/llm_sentiment/features.py`. It tokenizes each `LP` text with the regex:

```text
[A-Za-z][A-Za-z'-]*
```

The Loughran-McDonald-style categories are:

```text
positive
negative
uncertainty
litigious
strong_modal
weak_modal
constraining
```

For each text, the base vector has 17 dimensions:

```text
7 category counts
7 category shares = counts / token_count
net_positive = positive_count - negative_count
polarity = net_positive / max(1, positive_count + negative_count)
token_count
```

The 17-dimensional base vector is zero-padded to the shared text-feature width:

```text
target_dim = 1024
```

Output artifact:

```text
data/processed/text_features/rq2/20260623-rq2/llm_sentiment_features.xlsx
```

Output columns:

```text
news_row_id
article_id
source_file
sentiment_embedding
sentiment_dim
sentiment_dictionary_source
```

Actual manifest facts from `llm_sentiment_manifest.json`:

```text
row_count       = 14900
target_dim      = 1024
base_feature_dim = 17
dictionary_source = data/reference/Loughran-McDonald_MasterDictionary_1993-2025.csv
```

## 3. Workbook Enrichment

The sentiment feature workbook is merged into the RQ2 training workbook together with BoW:

```bash
python scripts/rq2/enrich_merged_vol.py \
  --merged-vol data/processed/svi-excel/20260410-174929/merged_vol.xlsx \
  --bow-features data/processed/text_features/rq2/20260623-rq2/bow_features.xlsx \
  --sentiment-features data/processed/text_features/rq2/20260623-rq2/llm_sentiment_features.xlsx \
  --output data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx
```

Final enriched workbook:

```text
data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx
```

The enrichment script preserves original sheets and joins features using `news_row_id`; if only `sample_id` exists, it parses IDs of the form `news_<id>`. For `gan_input_ready`, missing feature matches are not allowed.

During FiLM WGAN loading, `src/film_wgan/data.py` maps both `sentiment` and `llm_sentiment` modes to the `sentiment_embedding` column. The column must be non-empty and is parsed as a serialized numeric vector.

## 4. Training Protocol

Launcher:

```bash
CUDA_VISIBLE_DEVICES=1 ./run_film_wgan_llm_sentiment.sh
```

Resolved training config:

```text
outputs/training/film_wgan/llm_sentiment/20260623_154232/metrics/training_resolved_config.yaml
```

Important resolved settings:

```text
data_path            = data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx
sheet_name           = gan_input_ready
text_embedding_mode  = llm_sentiment
train_ratio          = 0.8
normalize_text_embedding = true
output_root          = outputs/training/film_wgan/llm_sentiment
```

The chronological split is the same as BoW and LP text:

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

## 5. Checkpoint and Generate-Result Process

Training output:

```text
outputs/training/film_wgan/llm_sentiment/20260623_154232
```

The run stopped at epoch 20 because early stopping was enabled. Training-time best checkpoint tracking used:

```text
primary metric = val_atm_short_pure_mae_gap_vs_current
checkpoint_warmup_epochs = 5
selection_start_epoch = 6
```

Training primary best:

```text
epoch 13
val_atm_short_pure_mae_gap_vs_current = -0.0011691697127364994
checkpoint = outputs/training/film_wgan/llm_sentiment/20260623_154232/checkpoints/film_wgan_best.pt
```

For the paper-facing comparison, the archive enforces:

```text
eligible epochs: epoch > 10
selection metric: lowest val_mae_gap_vs_current
```

Selected paper-facing checkpoint:

```text
epoch 12
val_mae_gap_vs_current = -0.0023298367319759435
checkpoint = outputs/training/film_wgan/llm_sentiment/20260623_154232/checkpoints/film_wgan_best_val_mae_gap_vs_current.pt
```

Generate-result command:

```bash
conda run -n py312 python scripts/film_wgan/main.py generate-result \
  --config outputs/training/film_wgan/llm_sentiment/20260623_154232/metrics/training_resolved_config.yaml \
  --checkpoint outputs/training/film_wgan/llm_sentiment/20260623_154232/checkpoints/film_wgan_best_val_mae_gap_vs_current.pt \
  --output-dir after10_val_mae_all_json \
  --split all \
  --selection-mode all \
  --selection-count 0 \
  --no-plot
```

Generated outputs:

```text
outputs/training/film_wgan/llm_sentiment/20260623_154232/after10_val_mae_all_json/
```

Final comparison archive:

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

The Loughran-McDonald sentiment baseline is strongest on the eval split for:

- `short_atm_mae`
- `atm7_abs_err`

However, it is worse than LP text on train and all for the broad surface and short-ATM metrics.

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
difference = llm_sentiment - text
```

For error metrics, negative difference means LLM sentiment is lower/better than LP text.

| split | metric | llm_sentiment_minus_text | 95% CI low | 95% CI high | interpretation |
|:--|:--|--:|--:|--:|:--|
| train | surface_mae | 0.000480 | 0.000372 | 0.000578 | LP text better; CI excludes zero |
| train | short_atm_mae | 0.000581 | 0.000414 | 0.000740 | LP text better; CI excludes zero |
| train | atm7_abs_err | 0.000140 | -0.000056 | 0.000331 | CI includes zero |
| eval | surface_mae | 0.000075 | -0.000062 | 0.000212 | CI includes zero |
| eval | short_atm_mae | -0.000059 | -0.000312 | 0.000190 | CI includes zero |
| eval | atm7_abs_err | -0.000374 | -0.000781 | -0.000011 | LLM sentiment better; CI excludes zero |
| all | surface_mae | 0.000399 | 0.000307 | 0.000482 | LP text better; CI excludes zero |
| all | short_atm_mae | 0.000453 | 0.000312 | 0.000588 | LP text better; CI excludes zero |
| all | atm7_abs_err | 0.000037 | -0.000133 | 0.000206 | CI includes zero |

## 8. Paired t-test P-values

Source table:

```text
outputs/comparison/film_wgan_text_20260417_131244_vs_bow_llm_sentiment_20260623_154232_after10_val_mae/t_tests/paired_t_test_p_values_compact_main_metrics.csv
```

The paired t-test is computed on:

```text
difference = metric_llm_sentiment - metric_text
H0: mean(difference) = 0
```

Rows are paired by `global_index` and `sample_id`.

| split | metric | llm_sentiment_minus_text | t statistic | p two-sided | one-sided p: LLM sentiment lower | one-sided p: text lower |
|:--|:--|--:|--:|--:|--:|--:|
| train | surface_mae | 0.000480 | 9.12757 | 1.26113e-19 | 1 | 6.30567e-20 |
| train | short_atm_mae | 0.000581 | 7.05672 | 2.11349e-12 | 1 | 1.05674e-12 |
| train | atm7_abs_err | 0.000140 | 1.44066 | 0.149786 | 0.925107 | 0.0748928 |
| eval | surface_mae | 0.000075 | 1.08909 | 0.276466 | 0.861767 | 0.138233 |
| eval | short_atm_mae | -0.000059 | -0.466978 | 0.640653 | 0.320326 | 0.679674 |
| eval | atm7_abs_err | -0.000374 | -1.90814 | 0.0567587 | 0.0283793 | 0.971621 |
| all | surface_mae | 0.000399 | 8.99544 | 3.67411e-19 | 1 | 1.83706e-19 |
| all | short_atm_mae | 0.000453 | 6.40418 | 1.70108e-10 | 1 | 8.50539e-11 |
| all | atm7_abs_err | 0.000037 | 0.422457 | 0.672716 | 0.663642 | 0.336358 |

The statistical evidence is mixed:

- Train: LP text is significantly better than LLM sentiment for `surface_mae` and `short_atm_mae`; `atm7_abs_err` is not significant.
- Eval: LLM sentiment improves `atm7_abs_err`; the one-sided t-test supports this at 5%, but the two-sided p-value is 0.0567587, so the evidence is borderline under a two-sided convention.
- All: LP text is significantly better for `surface_mae` and `short_atm_mae`; `atm7_abs_err` is not significant.

## 9. Thesis Interpretation

The Loughran-McDonald sentiment representation is much more compressed than BoW or LP embeddings: only 17 base sentiment/count features are non-zero before zero padding to 1024 dimensions. It therefore tests a narrower hypothesis: whether dictionary-based financial sentiment categories alone capture enough information for volatility-surface forecasting.

The result is not uniformly favorable:

```text
LLM sentiment is useful for eval 7d ATM error, but it does not outperform LP text on the broader train/all surface and short-ATM metrics.
```

For thesis writing, the safest interpretation is:

```text
Dictionary sentiment captures a targeted short-end ATM signal in the eval window, but it loses too much lexical and semantic detail to dominate the LP text representation across the full surface.
```

This supports using Loughran-McDonald sentiment as a traditional text-mining baseline, but not as a replacement for richer text embeddings in the main FiLM WGAN model.
