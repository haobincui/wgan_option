# WGAN Input Design

## Current State

The current training pipeline in [dataloader.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/utils/dataloader.py) builds supervised samples in this shape:

- `current_surface`
- `text_embedding`
- `future_surface`

Today, `src/wgan_option` does **not** consume minute SVI parameters directly.
The current code path:

- loads daily news embeddings from xlsx
- aggregates embeddings by date
- builds daily surface tensors from raw option trades
- trains a model of `surface + text -> future_surface`

So the new SVI-based workflow is a migration target, not a capability the training code already has.

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
- It is a staging dataset for the future GAN pipeline, not something the current `src/wgan_option` dataloader can read directly yet.

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

From [minute_svi_precalib_points.csv](/Users/haobincui/Documents/wgan_option/data/processed_excel_20260330-01/minute_svi_precalib_points.csv):

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

### Directly available from the SVI JSON

From [minute_svi_params.json](/Users/haobincui/Documents/wgan_option/data/processed_excel_20260330-01/minute_svi_params.json):

- `json_target_timestamp_utc`
- `snapshot_time_utc`
- `svi_params.business_days`
- `svi_params.a`
- `svi_params.b`
- `svi_params.rho`
- `svi_params.m`
- `svi_params.sigma`

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
3. refactor `src/wgan_option` from surface forecasting to SVI forecasting

### Migration Impact on Training Code

Current training code:

- reads daily news embeddings
- builds surface tensors
- trains on `current_surface + text_embedding -> future_surface`

Future training code should:

- read `gan_input_ready`
- parse SVI list fields into model features
- change the input from `surface + text` to `SVI + text`
- change the target from `future_surface` to `future_svi`

## Validation Checklist

When using this document as the input spec, confirm these points:

- every field listed above can be mapped to the merged news workbook, the precalib CSV, the SVI JSON, or a deterministic computation based on them
- the document does **not** claim that the current `src/wgan_option` code already supports direct SVI training
- the canonical sample unit remains `news × direction`
- `backward` always uses `timestamp_utc`
- `forward` always uses `timestamp_utc_plus_5m`
