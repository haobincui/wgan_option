# Current Executable Workflows

This document is the code-facing companion to the thesis design notes in:

- [input_svi.md](input_svi.md)
- [input_vol.md](input_vol.md)
- [svi_regressor_architecture.md](svi_regressor_architecture.md)
- [vol_surface_gan_architecture.md](vol_surface_gan_architecture.md)

It describes the current runnable pipeline in `scripts/` and `src/`, without replacing the research framing in those design documents.

## 1. End-to-End Pipeline

The current preferred executable flow is:

```text
raw option trades
    -> scripts/generate_surface/main.py
    -> data/processed/<model>/<run_ts>/minute_svi_params.json + minute_svi_precalib_points.csv
    -> scripts/merge_file/merge_svi.py / scripts/merge_file/merge_vol.py / scripts/merge_file/merge_params.py
    -> merged_svi.xlsx / merged_vol.xlsx / merged_params.xlsx
    -> scripts/train/main.py
    -> outputs/training/*
    -> scripts/generate_result/main.py / scripts/analyze_error/main.py
```

At a high level:

1. `generate_surface` builds minute-level SVI outputs from raw option data.
2. `merge_file` joins those outputs to the news-embedding workbook.
3. `train` consumes the merged workbook in either vol-surface or SVI mode.
4. `generate_result` and `analyze_error` inspect trained runs and checkpoints.

## 2. Preferred Commands

Generate minute surfaces from the nested surface-builder config:

```bash
python scripts/generate_surface/main.py minute-svi-excel --device gpu --model svi --config configs/surface_builder/svi/minute-svi-excel.yaml
```

Build the merged workbooks:

```bash
python scripts/merge_file/merge_svi.py --input-dir data/processed/svi/20260330-01
python scripts/merge_file/merge_vol.py --input-dir data/processed/svi/20260330-01
python scripts/merge_file/merge_params.py --input-dir data/processed/svi/20260330-01
```

Train from merged xlsx:

```bash
python scripts/train/main.py vol-xlsx --config configs/wgan/train_vol_xlsx.yaml
python scripts/train/main.py svi-xlsx --config configs/wgan/train_svi_xlsx.yaml
```

Downstream inspection is handled by:

- `scripts/generate_result/main.py`
- `scripts/analyze_error/main.py`

Those scripts work against explicit checkpoint paths or resolved run artifacts and are part of the supported current workflow.

## 3. Workbook Semantics That Must Stay Explicit

### `merged_vol.xlsx`

`merged_vol.xlsx` is already pair-based.

Its training-facing `gan_input_ready` sheet stores:

- `current_surface_flat` from the news row's `backward` direction
- `target_surface_flat` from the same news row's `forward` direction
- text embeddings in `hd`, `lp`, or `concat` mode

This is the current executable training input for `scripts/train/main.py vol-xlsx`.

### `merged_svi.xlsx`

`merged_svi.xlsx` is direction-based, not pair-based.

Important distinction:

- `news_direction_audit` is the current executable source for SVI training
- `train_svi_xlsx.py` pairs `backward` and `forward` rows at runtime using `news_row_id`
- `gan_input_ready` remains a filtered direction-level export, not the trainer's current sheet input

For the detailed model and training internals of this path, see [svi_regressor_architecture.md](svi_regressor_architecture.md).

This distinction matters for thesis lineage: the workbook stays audit-friendly even though training is now paired.

### Direction semantics

The repo consistently uses:

- `backward` = current = `timestamp_utc`
- `forward` = future = `timestamp_utc + 5 minutes`

Any new workflow should preserve those semantics explicitly instead of collapsing them into generic "input" and "target" labels too early.

## 4. Config Architecture

### Surface generation configs

The `generate_surface` entrypoint now expects a nested `surface_builder` config structure.

Typical shape:

```yaml
surface_builder:
  job: minute-svi-excel
  output_dir: data/processed
  minute_svi:
    model: svi
    run_ts: 20260330-01
    output_dir: data/processed/${model}/${run_ts}
  minute_svi_excel:
    option_data_glob: data/raw/option_data/**/*.csv
    ...
```

Important points:

- use an explicit top-level `surface_builder:` mapping
- keep shared values under that root
- put job-specific options under the matching section name

### Training configs

Merged training configs live under `configs/wgan/`.

Key examples:

- `configs/wgan/train_vol_xlsx.yaml`
- `configs/wgan/train_svi_xlsx.yaml`

Current conventions:

- `output_root` controls the training artifact tree for merged-xlsx runs
- `sheet_name` selects the workbook sheet to read
- `text_embedding_mode` supports `hd`, `lp`, and `concat`

The current default text mode remains `hd`.

## 5. Relationship to the Design Docs

The design docs and the code are intentionally not identical in role:

- [input_svi.md](input_svi.md) remains the audit and representation-design record for SVI-based experiments
- [input_vol.md](input_vol.md) remains the vol-surface workbook design record
- [vol_surface_gan_architecture.md](vol_surface_gan_architecture.md) remains the WGAN architecture note

This overview page only answers a different question:

`What can we run in the repo today, and how do those runnable steps map back to the thesis design?`
