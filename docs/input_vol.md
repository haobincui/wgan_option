# WGAN Vol-Surface Input Design

## Current State

The current training pipeline in [dataloader.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/utils/dataloader.py) builds supervised samples in this shape:

- `current_surface`
- `text_embedding`
- `future_surface`

Today, `src/wgan_option` does **not** read SVI-derived vol surfaces from xlsx.
The current code path:

- loads daily news embeddings from xlsx
- aggregates embeddings by date
- builds daily proxy surface tensors from raw option trades
- trains a model of `current_surface + text_embedding -> future_surface`

So the SVI-derived vol-surface workflow described here is a migration target, not something the current code already supports directly.

## Target Goal

Target direction for the next generation of the training dataset:

`text_embedding + current_vol_surface -> target_vol_surface`

In this design:

- `current_vol_surface` is reconstructed from the same news row's `backward` SVI result
- `target_vol_surface` is reconstructed from the same news row's `forward` SVI result
- both surfaces come from SVI params, not directly from raw option price tensors

Before migrating the WGAN input, we first need to verify that the SVI-derived vol surfaces are a good representation of the raw option data.

## Canonical Sample Definition

The canonical unit for this workflow is:

`one row = one news item`

Pairing rules:

- `current_surface` comes from the same news row's `backward` SVI result at `timestamp_utc`
- `target_surface` comes from the same news row's `forward` SVI result at `timestamp_utc_plus_5m`
- if either side is missing or unusable, the pair is not training-ready

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

## Surface Reconstruction Rule

Vol surfaces are reconstructed from SVI on a fixed `16 x 16` grid.

Use the same grid range as current `src/wgan_option` config in [config.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/config.py):

- `strike_bins = 16`
- `maturity_bins = 16`
- `moneyness_min = 0.7`
- `moneyness_max = 1.3`
- `maturity_min_days = 7`
- `maturity_max_days = 365`

The reconstructed surface should be produced from SVI using the logic available in [svi_surface.py](/Users/haobincui/Documents/wgan_option/src/quantlib/vol_surface/algo/svi_surface.py).

Serialization rule for xlsx:

- `strike_grid` is stored as a list string
- `maturity_days_grid` is stored as a list string
- each surface is stored as a flattened list string
- flatten order is fixed:
  - iterate `maturity_days_grid` outer
  - iterate `strike_grid` inner

## Recommended XLSX Outputs

### 1. `news_surface_pair_audit`

Purpose:
audit whether a news row can form a usable `backward -> forward` surface pair.

Headers:

```text
sample_id,
news_row_id,
article_id,
source_file,
news_timestamp_utc,
current_snapshot_time_utc,
target_snapshot_time_utc,
current_json_target_timestamp_utc,
target_json_target_timestamp_utc,
hd_text,
lp_text,
hd_embedding,
lp_embedding,
hd_dim,
lp_dim,
strike_grid,
maturity_days_grid,
surface_shape,
current_has_svi,
target_has_svi,
current_svi_slice_count,
target_svi_slice_count,
current_surface_flat,
target_surface_flat,
current_raw_point_count,
target_raw_point_count,
current_raw_point_pass_count,
target_raw_point_pass_count,
current_exact_slice_point_ratio,
target_exact_slice_point_ratio,
current_weighted_iv_rmse,
target_weighted_iv_rmse,
current_weighted_iv_mae,
target_weighted_iv_mae,
current_max_abs_iv_error,
target_max_abs_iv_error,
current_weighted_total_var_rmse,
target_weighted_total_var_rmse,
current_boundary_flag,
target_boundary_flag,
current_placeholder_flag,
target_placeholder_flag,
pair_quality_label,
training_candidate_flag,
exclude_reason
```

Pair labeling rules:

- `no_current_svi`
- `no_target_svi`
- `current_placeholder`
- `target_placeholder`
- `no_current_raw_points`
- `no_target_raw_points`
- `poor_fit`
- `usable`

Medium-strict training rule:

- both sides must have SVI
- neither side can be placeholder
- both sides need `raw_point_pass_count > 0`
- both sides need `exact_slice_point_ratio >= 0.5`
- both sides need `weighted_iv_rmse <= 0.05`

Notes:

- This is the main table for deciding whether a sample is usable.
- `current_surface_flat` and `target_surface_flat` are reconstructed from SVI, not copied directly from any raw source file.
- `pair_quality_label` should summarize the quality of the entire `backward -> forward` pair.

### 2. `surface_side_detail`

Purpose:
keep per-side audit separate so current and target quality can be inspected independently.

One row per `news × side`, where `side in {current_back, target_forward}`.

Headers:

```text
sample_id,
news_row_id,
side,
source_direction,
matched_snapshot_time_utc,
json_target_timestamp_utc,
has_svi,
svi_slice_count,
svi_business_days_list,
svi_a_list,
svi_b_list,
svi_rho_list,
svi_m_list,
svi_sigma_list,
strike_grid,
maturity_days_grid,
surface_flat,
surface_min,
surface_max,
surface_mean,
surface_std,
raw_point_count,
raw_point_pass_count,
exact_slice_point_ratio,
weighted_iv_rmse,
weighted_iv_mae,
max_abs_iv_error,
weighted_total_var_rmse,
boundary_flag,
placeholder_flag,
side_quality_label
```

Notes:

- `source_direction` is `backward` for current and `forward` for target.
- `surface_flat` is the flattened vol surface reconstructed on the fixed grid.
- `side_quality_label` should be derived from that side's own fit statistics.

### 3. `gan_input_ready`

Purpose:
final training-ready table for the future vol-surface WGAN pipeline.

Headers:

```text
sample_id,
news_timestamp_utc,
current_snapshot_time_utc,
target_snapshot_time_utc,
hd_embedding,
lp_embedding,
hd_dim,
lp_dim,
strike_grid,
maturity_days_grid,
surface_shape,
current_surface_flat,
target_surface_flat,
current_weighted_iv_rmse,
target_weighted_iv_rmse,
pair_quality_label,
training_candidate_flag
```

Notes:

- This table is derived from `news_surface_pair_audit`.
- Only rows with acceptable pair quality should be promoted into this table.
- This is the future dataloader target, not something the current `src/wgan_option` code already reads.

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

- `current_json_target_timestamp_utc`
- `target_json_target_timestamp_utc`
- `snapshot_time_utc`
- `svi_params.business_days`
- `svi_params.a`
- `svi_params.b`
- `svi_params.rho`
- `svi_params.m`
- `svi_params.sigma`

### Derived after matching and reconstruction

The following fields are **not** stored directly in one source file. They must be computed after matching news rows, raw option points, and SVI slices:

- `sample_id`
- `current_snapshot_time_utc`
- `target_snapshot_time_utc`
- `strike_grid`
- `maturity_days_grid`
- `surface_shape`
- `current_surface_flat`
- `target_surface_flat`
- `current_has_svi`
- `target_has_svi`
- `current_svi_slice_count`
- `target_svi_slice_count`
- `current_raw_point_count`
- `target_raw_point_count`
- `current_raw_point_pass_count`
- `target_raw_point_pass_count`
- `current_exact_slice_point_ratio`
- `target_exact_slice_point_ratio`
- `current_weighted_iv_rmse`
- `target_weighted_iv_rmse`
- `current_weighted_iv_mae`
- `target_weighted_iv_mae`
- `current_max_abs_iv_error`
- `target_max_abs_iv_error`
- `current_weighted_total_var_rmse`
- `target_weighted_total_var_rmse`
- `current_boundary_flag`
- `target_boundary_flag`
- `current_placeholder_flag`
- `target_placeholder_flag`
- `pair_quality_label`
- `training_candidate_flag`
- `exclude_reason`
- all side-level metrics in `surface_side_detail`

## Important Notes

### Embedding Storage

Keep embeddings as serialized vector strings inside xlsx.
Do **not** expand them into 1024 separate Excel columns.
They should be converted into tensors later inside `src/wgan_option`.

### Surface Storage

Keep each reconstructed vol surface as a flattened list string inside xlsx.
Do **not** expand each surface cell into a separate Excel column.
The training pipeline should parse the string and reshape it back into `16 x 16`.

### Why This Order

We should work in this order:

1. verify reconstructed vol surfaces against raw option data
2. form usable `backward -> forward` paired samples
3. migrate `src/wgan_option` from raw/proxy surface forecasting to SVI-derived vol-surface forecasting

### Migration Impact on Training Code

Current training code:

- reads daily news embeddings
- builds daily proxy surfaces
- trains on `current_surface + text_embedding -> future_surface`

Future training code should:

- read pair-level xlsx rows
- parse `current_surface_flat` and `target_surface_flat`
- keep the model input family as `surface + text`
- change the surface source from raw/proxy-built to SVI-reconstructed
- change the target from next-day proxy surface to same-news forward SVI-derived surface

## Validation Checklist

When using this document as the input spec, confirm these points:

- current WGAN input shape still matches [dataloader.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/utils/dataloader.py)
- grid defaults still match [config.py](/Users/haobincui/Documents/wgan_option/src/wgan_option/config.py)
- SVI-to-vol reconstruction is supported by [svi_surface.py](/Users/haobincui/Documents/wgan_option/src/quantlib/vol_surface/algo/svi_surface.py)
- the document does **not** claim that the current `src/wgan_option` code already supports this SVI-derived vol-surface workflow directly
- the canonical sample unit remains one news item paired as `backward -> forward`
