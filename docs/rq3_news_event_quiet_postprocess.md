# RQ3 News-Event vs No-News Quiet Post-Processing

This note documents the general news-arrival extension to RQ3. It is a post-processing experiment, not a retraining pipeline.

## Purpose

The original RQ3 event study focuses on scheduled FOMC press releases. This extension broadens the contrast:

```text
news_event = all usable news-row samples
quiet      = no-news samples sampled from a 5-minute surface time grid
```

The goal is to test whether LP text conditioning has larger marginal prediction value when news arrives than during no-news periods.

## Dataset Construction

Build a news/quiet workbook with:

```bash
python scripts/rq3/main.py build-news-quiet-workbook \
  --source-merged-vol <merged_vol_rq2_text.xlsx> \
  --surface-all-json <surface-svi-all.json or surface-raw-all.json> \
  --news-xlsx data/raw/text_embedding/news_with_openai_embeddings_large.xlsx \
  --output-workbook <news_quiet_eval_workbook.xlsx> \
  --horizon-minutes 5 \
  --quiet-grid-minutes 5 \
  --quiet-buffer-minutes 60
```

Event rows come from the source merged workbook and retain their real LP, BoW, and sentiment embeddings. Quiet rows are constructed from `surface-*-all.json`; both `t` and `t+5min` must exist, and `t` must be at least 60 minutes away from every news timestamp.

Quiet text embeddings are zero vectors with the same dimensions as the source workbook. No `has_news` mask is added to the model input in this post-processing experiment.

## Analysis Commands

Workbook jump analysis:

```bash
python scripts/rq3/main.py news-quiet-workbook \
  --workbook <news_quiet_eval_workbook.xlsx> \
  --output-dir <workbook_analysis_dir> \
  --split all
```

Model-result grouping:

```bash
python scripts/rq3/main.py news-quiet-result \
  --output-dir <model_result_analysis_dir> \
  --text-label <text_model_label> \
  --result <label>=<summary.csv> \
  --result <label>=<summary.csv>
```

The result summaries must be generated from the news/quiet workbook, because `summary.csv` must contain `event_group`.

## Interpretation

The main text-value statistic is:

```text
text_advantage = baseline_error - text_error
DiD = text_advantage(news_event) - text_advantage(quiet)
```

Positive `DiD` means LP text provides more incremental value for news-event samples than for quiet samples.

This design is not a causal news-shock identification strategy. News arrival is endogenous, and quiet samples are evaluated with zero text embeddings using checkpoints trained on news-row data. Results should be reported as predictive contrast / robustness evidence.
