# merge_file

This directory contains the Excel merge stage that connects generated minute-surface outputs with the news embedding workbook.

It builds the two merged workbooks used by the later training and inference pipelines:

- `merged_svi.xlsx`
- `merged_vol.xlsx`
- `merged_params.xlsx`

Both scripts consume the same upstream generation outputs:

- `surface-<model>-<data_range>.json`
- `surface-<model>-<data_range>-precalib-points.csv`
- optional `surface-resolved_config.yaml`

The merge loaders prefer the new `surface-*` names and still fall back to legacy
`minute_svi_*` filenames when reading older processed runs.

and combine them with:

- `data/raw/text_embedding/news_with_openai_embeddings_large.xlsx`

## Entry Points

Build the SVI workbook:

```bash
python scripts/merge_file/merge_svi.py --input-dir data/processed/svi-all/20260330-01
```

Build the vol workbook:

```bash
python scripts/merge_file/merge_vol.py --input-dir data/processed/svi-all/20260330-01
```

Build the model-neutral parameter audit workbook:

```bash
python scripts/merge_file/merge_params.py --input-dir data/processed/svi-all/20260330-01
```

## Shared Input Semantics

The merge step relies on the minute-SVI direction convention:

- `backward` corresponds to the news row’s original `timestamp_utc`
- `forward` corresponds to `timestamp_utc + offset_minutes`

All three scripts:

1. load the news Excel file
2. derive UTC timestamps from `PD` and `ET`
3. load pre-calibration CSV rows
4. load the model-aware surface JSON output
5. align news timestamps to generated surface snapshots
6. compute audit-quality statistics
7. write a multi-sheet workbook

## merge_svi.py

Purpose:

- create an audit workbook centered on direction-level SVI data

Output file:

- `merged_svi.xlsx`

Main sheets:

- `news_direction_audit`
- `svi_slice_detail`
- `gan_input_ready`

### news_direction_audit

Granularity:

- one row per `news_row_id` and direction

Each row stores:

- the original news metadata and embeddings
- the matched snapshot time
- raw point statistics from the pre-calibration CSV
- SVI parameter lists
- fit-quality diagnostics
- `training_candidate_flag`

This sheet is the main audit and traceability table for SVI-based workflows.

### svi_slice_detail

Granularity:

- one row per SVI slice

It records slice-level SVI parameters and slice-level fit diagnostics.

### gan_input_ready

Granularity:

- still one row per direction

Important semantic note:

- this sheet is not yet a `current -> future` paired training table
- the later SVI training code pairs `backward` and `forward` rows on the fly using `news_row_id`

### Quality Label Logic

`merge_svi.py` marks direction rows as usable only when the row has valid SVI information and acceptable fit quality.

Examples of exclusion reasons include:

- no SVI
- placeholder SVI
- no raw points
- low exact-slice ratio
- high weighted IV RMSE

Only rows with `fit_quality_label == "usable"` become training candidates.

## merge_vol.py

Purpose:

- create a paired vol-surface workbook for current -> future forecasting
- reconstruct surfaces from model-aware `surface_model/surface_params` payloads

Output file:

- `merged_vol.xlsx`

Main sheets:

- `news_surface_pair_audit`
- `surface_side_detail`
- `gan_input_ready`

### news_surface_pair_audit

Granularity:

- one row per paired sample

Each row combines:

- the `backward` side as current
- the `forward` side as target

It stores:

- current and target snapshot times
- the reconstructed current and target vol surfaces
- current and target fit diagnostics
- pair-level quality label
- `training_candidate_flag`

### surface_side_detail

Granularity:

- one row per side

This sheet breaks the pair into:

- `current_back`
- `target_forward`

and records side-specific fit diagnostics and the reconstructed surface itself.

### gan_input_ready

Granularity:

- one row per paired sample

This is already in the training-ready shape expected by the merged vol workflow.

Each usable row contains:

- text embeddings
- current surface
- target surface
- surface grid metadata
- pair quality metadata

### Pair Semantics

This is the key distinction from `merge_svi.py`:

- `merge_vol.py` already produces paired current/target samples
- `merge_svi.py` does not

So `merged_vol.xlsx` can be used directly for current-surface -> future-surface model training.

## merge_params.py

Purpose:

- create a model-neutral parameter audit workbook for `svi`, `sabr`, and `cubic`

Output file:

- `merged_params.xlsx`

Main sheets:

- `news_direction_audit`
- `surface_slice_detail`
- `gan_input_ready`

This workbook keeps parameter-level audit fields generic:

- `surface_model`
- `surface_slice_count`
- `surface_business_days_list`
- `surface_param_json`
- `slice_param_json`

## Surface Reconstruction Logic

`merge_vol.py` reconstructs a vol surface from model-aware slices on a fixed grid.

The grid is defined by the default config values for:

- strike bins
- maturity bins
- minimum and maximum moneyness
- minimum and maximum maturity days

For each maturity on the grid, the script interpolates surface slices across business-day terms and then computes model implied vol across the strike grid.

## Typical Workflow Position

The merge stage sits between generation and training:

```text
surface-<model>-<data_range>.json
+ surface-<model>-<data_range>-precalib-points.csv
-> merge_svi.py / merge_vol.py / merge_params.py
-> merged_svi.xlsx / merged_vol.xlsx / merged_params.xlsx
-> train / generate_result / analyze_error
```

## Typical Commands

Build both merged workbooks from the same processed directory:

```bash
python scripts/merge_file/merge_svi.py --input-dir data/processed/svi-all/20260330-01
python scripts/merge_file/merge_vol.py --input-dir data/processed/svi-all/20260330-01
python scripts/merge_file/merge_params.py --input-dir data/processed/svi-all/20260330-01
```

Use a different forward offset:

```bash
python scripts/merge_file/merge_svi.py --input-dir data/processed/svi-all/20260330-01 --offset-minutes 10
python scripts/merge_file/merge_vol.py --input-dir data/processed/svi-all/20260330-01 --offset-minutes 10
```
