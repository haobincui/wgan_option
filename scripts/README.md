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

- `scripts/rq123`
  - canonical corrected raw-vol RQ1-RQ3 orchestration
  - background/resume/monitor/final-validation/package entrypoints
- `scripts/rq1_pair`
  - pair-level LP versus controlled no-text rolling experiment
  - shared Stage-A parent and residual Stage-B continuation
- `scripts/rq2_pair`
  - train-fold-only BoW log-count and ChatGPT-score representation tests
- `scripts/rq3`
  - frozen all-OOS conditional predictive-robustness analysis
  - scheduled-vs-ordinary matching retained as secondary robustness
- `scripts/raw_vol`
  - corrected Black76 raw-vol dataset construction and hard-stop audits
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

## Canonical Corrected Run

Run from the repository root:

```bash
GPU_IDS="0 1" RUNS_PER_GPU=2 \
bash scripts/rq123/start_corrected_pipeline_background.sh
```

Monitor:

```bash
bash scripts/rq123/monitor_corrected_pipeline.sh
```

The corrected workflow fixes `Europe/London` Factiva timestamps, disjoint
`[t-5,t)` / `[t,t+5)` windows, CME TY option expiry, Black76 with frozen rates
and prior futures, OTM filtering, fold-train-only raw support, strict text
lineage, and dependence-aware RQ1-RQ3 inference.

To rebuild only the training workbook with the frozen CME Treasury session
calendar and a five-minute origin tolerance:

```bash
bash scripts/rq123/build_session_aligned_dataset.sh
```

This mode keeps in-session news at its publication minute. News published
during the daily halt, weekend, or a frozen 2022-2023 holiday closure is moved
to the next continuous-session open. The selected raw-vol origin must be the
first complete pair within five minutes of that scheduled origin, and the
full `[origin-5, origin+5]` interval must remain inside one session.

RQ dataset construction reads the 728 daily 2022-2023 `0#TY+` files only. It
does not read `ty_plus_merged.csv.gz`, avoiding duplicate source traversal.

Superseded RQ orchestration is preserved under
`scripts/archive/pre_corrected_20260727/`; it is not a current entrypoint.

## Script-Level READMEs

- `scripts/generate_surface/README.md`
- `scripts/merge_file/README.md`
- `scripts/train/README.md`
- `scripts/generate_result/README.md`
- `scripts/analyze_error/README.md`
