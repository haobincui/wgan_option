# WGAN Input Design

## Current State

This document remains the thesis-facing design and audit spec for SVI-based training data.
For the command-oriented view of what is runnable today, see [current_executable_workflows.md](current_executable_workflows.md).
For the detailed architecture of the current executable SVI trainer, see [svi_regressor_architecture.md](svi_regressor_architecture.md).

The repo now has two executable training paths that matter for this design:

- the legacy daily-surface WGAN path in [dataloader.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/utils/dataloader.py) and `scripts/train.py`
- the merged-SVI supervised path in `scripts/train/main.py svi-xlsx` and `src/wgan_option/train_svi_xlsx.py`

The legacy daily path still builds supervised samples in this shape:

- `current_surface`
- `text_embedding`
- `future_surface`

The merged-SVI path now does consume minute-SVI-derived workbook rows, but it does so with one important distinction:

- `merged_svi.xlsx` is still written as a direction-level audit workbook
- executable pairing happens later inside `src/wgan_option/utils/merged_xlsx.py`
- the current trainer reads `news_direction_audit` and pairs `backward` and `forward` rows by `news_row_id`

So this document should now be read as design target plus implementation notes:

- the audit-oriented workbook design still matters for thesis traceability
- direct merged-SVI training is now executable
- `gan_input_ready` in this workbook is still a staging/export view, not the current trainer's actual sheet input

## Target Goal

Target direction for the next generation of the training dataset:

`text_embedding + current_svi -> future_svi`

Before changing the GAN input format, we need to verify that minute SVI parameters are a good representation of the raw option data. The first deliverable is therefore an **SVI audit dataset**, not the final GAN training dataset.

## Canonical Sample Definition

The canonical unit for this workflow is:

`one row = one news item × one direction`

Direction rules:

- `backward` uses `timestamp_utc`
- `forward` uses `timestamp_utc_plus_5m`

Join rules:

- raw option rows join by `calibration_datetime_utc`
- SVI rows join by direction-level `snapshot_time_utc`

Source column mapping for the news side:

- `article_id <- ArticleID`
- `source_file <- SourceFile`
- `hd_text <- HD`
- `lp_text <- LP`
- `hd_embedding <- HD_embedding`
- `lp_embedding <- LP_embedding`
- `hd_dim <- HD_dim`
- `lp_dim <- LP_dim`
- `news_timestamp_utc <- timestamp_utc`

## Recommended XLSX Outputs

### 1. `news_direction_audit`

Purpose:
fit-quality and data-alignment check before training.

Headers:

```text
sample_id,
news_row_id,
article_id,
source_file,
direction,
news_timestamp_utc,
matched_snapshot_time_utc,
json_target_timestamp_utc,
hd_text,
lp_text,
hd_embedding,
lp_embedding,
hd_dim,
lp_dim,
has_csv_points,
raw_point_count,
raw_point_pass_count,
raw_point_fail_count,
raw_point_pass_ratio,
raw_business_days_count,
raw_business_days_min,
raw_business_days_max,
strike_min,
strike_max,
percent_strike_min,
percent_strike_max,
has_svi_params,
svi_slice_count,
svi_business_days_min,
svi_business_days_max,
svi_business_days_list,
svi_a_list,
svi_b_list,
svi_rho_list,
svi_m_list,
svi_sigma_list,
exact_slice_point_count,
exact_slice_point_ratio,
nearest_slice_gap_days_mean,
nearest_slice_gap_days_max,
weighted_iv_rmse,
weighted_iv_mae,
max_abs_iv_error,
weighted_total_var_rmse,
svi_boundary_flag,
svi_placeholder_flag,
fit_quality_label,
training_candidate_flag,
exclude_reason
```

Notes:

- This is the main table for deciding whether a sample is usable.
- `svi_*_list` fields come from JSON `svi_params`.
- `weighted_iv_rmse`, `weighted_iv_mae`, `max_abs_iv_error`, and `weighted_total_var_rmse` are computed from matched raw option points plus SVI slices.
- `svi_boundary_flag` and `svi_placeholder_flag` are quality-control flags, not raw source fields.

### 2. `svi_slice_detail`

Purpose:
inspect each maturity slice used in a `news × direction` sample.

Headers:

```text
sample_id,
news_row_id,
direction,
matched_snapshot_time_utc,
json_target_timestamp_utc,
slice_index,
business_days,
a,
b,
rho,
m,
sigma,
raw_point_count_on_slice,
raw_point_pass_count_on_slice,
percent_strike_min_on_slice,
percent_strike_max_on_slice,
weighted_iv_rmse_slice,
weighted_iv_mae_slice,
max_abs_iv_error_slice,
boundary_flag_slice,
placeholder_flag_slice
```

Notes:

- `a`, `b`, `rho`, `m`, `sigma`, and `business_days` come from one SVI slice.
- Slice-level fit metrics are computed after matching raw option rows to the same or nearest `business_days`.

### 3. `gan_input_ready`

Purpose:
final training-ready table after filtering the audit table.

Headers:

```text
sample_id,
news_timestamp_utc,
direction,
matched_snapshot_time_utc,
hd_embedding,
lp_embedding,
hd_dim,
lp_dim,
svi_business_days_list,
svi_a_list,
svi_b_list,
svi_rho_list,
svi_m_list,
svi_sigma_list,
raw_point_pass_count,
exact_slice_point_ratio,
weighted_iv_rmse,
fit_quality_label,
training_candidate_flag
```

Notes:

- This table is derived from `news_direction_audit`.
- Only rows with acceptable fit quality should be promoted into this table.
- It remains a useful staging/export dataset for filtered direction-level rows.
- The current executable SVI trainer does **not** read this sheet directly.
- Today's runtime pairing path reads `news_direction_audit` and forms `backward -> forward` pairs by `news_row_id`.

## Field Provenance

### Directly available from the merged news workbook

From [news_with_openai_embeddings_large_merged.xlsx](/Users/haobincui/Documents/wgan_option/data/raw/text_embedding/news_with_openai_embeddings_large_merged.xlsx):

- `news_row_id`
- `article_id` via `ArticleID`
- `source_file` via `SourceFile`
- `hd_text` via `HD`
- `lp_text` via `LP`
- `hd_embedding`
- `lp_embedding`
- `hd_dim`
- `lp_dim`
- `news_timestamp_utc` via `timestamp_utc`
- `timestamp_utc_plus_5m`

### Directly available from the raw precalibration CSV

From [minute_svi_precalib_points.csv](/Users/haobincui/Documents/wgan_option/data/processed/svi/20260330-01/minute_svi_precalib_points.csv):

- `trade_datetime_utc`
- `calibration_datetime_utc`
- `business_days`
- `maturity_date`
- `contract_id`
- `option_type`
- `strike`
- `price`
- `spot`
- `percent_strike`
- `implied_vol`
- `passes_precalib_filter`
- `filter_reason`
- `weight`

### Directly available from the surface JSON

From [minute_svi_params.json](/Users/haobincui/Documents/wgan_option/data/processed/svi/20260330-01/minute_svi_params.json):

- `json_target_timestamp_utc`
- `snapshot_time_utc`
- `surface_model`
- `surface_params.business_days`
- `surface_params.a`
- `surface_params.b`
- `surface_params.rho`
- `surface_params.m`
- `surface_params.sigma`

### Derived after matching

The following fields are **not** stored directly in one source file. They must be computed after matching news rows, raw option points, and SVI slices:

- `sample_id`
- `matched_snapshot_time_utc`
- `has_csv_points`
- `raw_point_count`
- `raw_point_pass_count`
- `raw_point_fail_count`
- `raw_point_pass_ratio`
- `raw_business_days_count`
- `raw_business_days_min`
- `raw_business_days_max`
- `strike_min`
- `strike_max`
- `percent_strike_min`
- `percent_strike_max`
- `has_svi_params`
- `svi_slice_count`
- `svi_business_days_min`
- `svi_business_days_max`
- `svi_business_days_list`
- `svi_a_list`
- `svi_b_list`
- `svi_rho_list`
- `svi_m_list`
- `svi_sigma_list`
- `exact_slice_point_count`
- `exact_slice_point_ratio`
- `nearest_slice_gap_days_mean`
- `nearest_slice_gap_days_max`
- `weighted_iv_rmse`
- `weighted_iv_mae`
- `max_abs_iv_error`
- `weighted_total_var_rmse`
- `svi_boundary_flag`
- `svi_placeholder_flag`
- `fit_quality_label`
- `training_candidate_flag`
- `exclude_reason`
- all slice-level summary metrics in `svi_slice_detail`

## Important Notes

### Embedding Storage

Keep embeddings as serialized vector strings inside xlsx.
Do **not** expand them into 1024 separate Excel columns.
They should be converted into tensors later inside `src/wgan_option`.

### Why This Order

We should work in this order:

1. verify `SVI vs raw option data`
2. select usable `news × direction` samples
3. pair usable `backward` and `forward` rows into forecasting examples
4. compare SVI forecasting against other representations at the experiment level

### Migration Impact on Training Code

Current executable state:

- legacy daily training still reads daily news embeddings, builds surface tensors, and trains on `current_surface + text_embedding -> future_surface`
- merged-SVI training now reads `merged_svi.xlsx`
- runtime sample pairing happens from `news_direction_audit`, not from `gan_input_ready`
- the model input family is already `current_svi + text -> future_svi`

Research-facing design implications:

- `news_direction_audit` remains the canonical audit table for deciding whether a row is usable
- `gan_input_ready` remains a filtered direction-level export, useful for lineage and sanity checks
- any future schema change should preserve the distinction between direction-level audit data and runtime paired forecasting samples

## Validation Checklist

When using this document as the input spec, confirm these points:

- every field listed above can be mapped to the merged news workbook, the precalib CSV, the SVI JSON, or a deterministic computation based on them
- the document distinguishes thesis-facing audit design from the current executable SVI trainer
- the current executable SVI trainer reads `news_direction_audit` and pairs rows by `news_row_id`
- `gan_input_ready` in `merged_svi.xlsx` remains direction-level and is not the trainer's current sheet input
- the canonical sample unit remains `news × direction`
- `backward` always uses `timestamp_utc`
- `forward` always uses `timestamp_utc_plus_5m`
