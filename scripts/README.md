# scripts

This directory contains the runnable research workflows for the repository.

## Pipeline

```text
raw option files
-> scripts/generate_surface
-> data/processed/<model>-<data_range>/<run_ts>/
-> scripts/merge_file
-> merged_svi.xlsx / merged_vol.xlsx / merged_params.xlsx
-> scripts/train
-> outputs/training/<model>-<data_range>/<run_ts>/
-> scripts/generate_result or scripts/analyze_error
```

## Main Script Areas

- `scripts/generate_surface`
  - unified minute surface-generation CLI
  - supports `--model {svi,sabr,cubic,raw}` and `--data_range {all,window,excel}`
- `scripts/merge_file`
  - converts one generated run directory into merged audit and training workbooks
- `scripts/train`
  - trains on merged workbook datasets
- `scripts/generate_result`
  - runs saved checkpoints on selected workbook samples
- `scripts/analyze_error`
  - computes error distributions and bootstrap summaries from saved checkpoints

## Script-Level READMEs

- `scripts/generate_surface/README.md`
- `scripts/merge_file/README.md`
- `scripts/train/README.md`
- `scripts/generate_result/README.md`
- `scripts/analyze_error/README.md`
