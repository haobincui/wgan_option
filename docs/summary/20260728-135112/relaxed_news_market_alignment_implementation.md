# Relaxed News-Market Alignment Implementation

Implementation time: `2026-07-28 13:51:12 UTC`

## Objective

Increase the raw-vol training sample without treating a missing exact
publication-minute surface as evidence that no usable market observation
exists. The implementation maps each article once to the earliest complete
five-minute raw-vol pair at or after news availability.

## Frozen Rules

```text
source timezone = Europe/London
publication availability lag = 0 minutes
forecast horizon = 5 minutes
intraday tolerance = 15 minutes
maximum forward shift = 72 hours
session-shifted rows = included in main dataset
sample unit = unique surface pair
collision policy = pool articles at the same earliest pair
```

For origin `o`:

```text
current = trades in [o - 5min, o)
target  = trades in [o, o + 5min)
```

The matching rule is:

```text
o = earliest valid pair origin where news_time <= o <= news_time + 72h
```

Thus no origin can precede the news. A valid pair requires raw surfaces at
both `o` and `o + 5min`.

## Implemented Components

- `src/wgan_option/surface_generation/market_index.py`
  - versioned SQLite schema;
  - resumable raw-file and calibration-batch state;
  - candidate anchor construction;
  - complete five-minute pair index;
  - deterministic forward alignment and invariants.
- `scripts/raw_vol/relaxed_time_pipeline.py`
  - scan of positive-price, positive-volume option minutes;
  - weekly window-calibration batches;
  - canonical backward-anchor ingestion;
  - precalibration provenance retention;
  - aligned JSON/CSV/workbook materialization;
  - strict-vs-relaxed coverage gate.
- `src/wgan_option/merge/merge_vol_core.py`
  - optional `alignment_csv_path`;
  - effective origin/target lookup;
  - alignment audit columns passed into `gan_input_ready`.
- `scripts/rq123/run_relaxed_time_pipeline.sh`
  - dataset build, RQ2 enrichment, validation, and optional RQ1-RQ3 run.
- `scripts/rq123/start_relaxed_time_pipeline_background.sh`
  - detached process-group launcher.

Strict output directories are never overwritten.

## Validation

Synthetic tests cover:

```text
exact match
1-minute and 15-minute intraday shifts
weekend/session shift
UTC ordering across London DST
no pair in 72 hours
origin-before-news rejection
earliest eligible origin
fixed five-minute non-overlapping windows
same-pair collision pooling
SQLite idempotent resume
```

A real one-day smoke test used:

```text
0#TY+_2022-11-23_2022-11-24.csv.gz
valid option observation minutes = 417
candidate anchors = 852
valid raw surface anchors = 71
complete five-minute pairs = 20
stored precalibration rows = 2044
```

Candidate construction intentionally excludes futures-only minutes and option
rows without positive price and volume; the old all-trade-minute scan produced
1388 candidates for the same file. The completed batch resumed in about one
second without duplicate rows. A news item available at
`2022-11-23T14:10:00Z` matched to
`2022-11-23T14:11:00Z`, with target `14:16:00Z`; the resulting workbook row
was `usable`, and the corrected raw-vol validator passed.

## Full Background Command

From the repository root:

```bash
GPU_IDS="0 1" RUNS_PER_GPU=2 \
bash scripts/rq123/start_relaxed_time_pipeline_background.sh
```

Dataset-only preflight:

```bash
RUN_DOWNSTREAM=0 \
bash scripts/rq123/start_relaxed_time_pipeline_background.sh
```

The full run must pass the dynamic coverage gate before training:

```text
relaxed article rows > strict article rows
relaxed unique pairs > strict unique pairs
```

The final analysis must stratify results by `exact`, `intraday_shift`, and
`session_shift`.
