# RQ1-RQ3 Europe/London Timestamp Recalculation

Implementation time (UTC): `2026-07-25 18:33:49`

## Worktree

```text
path:
/home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone

branch:
wgan_rq123_london_timezone

base commit:
26f3a23291eb78aa2019505ed61aa0ffcdd8d767
```

The worktree combines the RQ1 no-text continuation implementation, the RQ2
representation continuation implementation, and the scheduled-news RQ3
implementation. Existing worktrees and historical outputs are not modified.

## Timestamp Correction

Factiva publication fields are interpreted as:

```text
date column       = PD
time column       = ET
source timezone   = Europe/London
output timezone   = UTC
```

`Europe/London` applies GMT in winter and BST in summer. For example:

```text
2023-12-30 07:35 Europe/London -> 2023-12-30T07:35:00Z
2023-07-01 08:30 Europe/London -> 2023-07-01T07:30:00Z
```

This setting applies only to Factiva `PD/ET`. Official scheduled releases whose
calendar records use `America/New_York` retain that timezone.

## Recalculation Scope

The timestamp correction changes the option-data windows used to construct
current and target surfaces. The following artifacts must therefore be rebuilt:

1. raw-vol surface JSON and pre-calibration points;
2. `merged_vol.xlsx` and `merged_vol_rq2_text.xlsx`;
3. `surface_pair_id`, rolling fold manifests and article-to-pair lineage;
4. fold-train-only LP PCA, BoW vocabulary/PCA and sentiment scaling;
5. all RQ1 Stage-A, continuation, LP, shuffled and architecture runs;
6. all RQ2 BoW and ChatGPT sentiment continuation runs;
7. all test predictions, comparison tables and statistical tests;
8. RQ3 scheduled-news matching and inference.

Article text, LP embeddings, article-level ChatGPT scores, raw responses, raw
option files and official release calendars remain reusable.

## Code Controls

- `src/wgan_option/news_time.py` is the canonical Factiva parser.
- Surface generation and merge use the same timezone.
- A pre-surface audit cross-checks converted UTC times against GMT stamps embedded
  in Factiva LP text and requires at least a 95% match rate within two minutes.
- Merge reads `surface-resolved_config.yaml` and rejects a conflicting timezone.
- Merged workbooks preserve local timestamp, timezone, UTC offset and parse status.
- RQ1 rejects workbooks not marked `Europe/London`.
- RQ2 rejects RQ1 archives not marked `Europe/London`.
- RQ3 rejects RQ1/RQ2 archives or workbooks with a different source timezone.
- Rolling fold counts are read from the newly generated lineage files, not from
  historical New-York-time sample counts.

## Parallel Training

The scheduler uses configurable GPU slots:

```text
GPU_IDS="0 1"
RUNS_PER_GPU=2
total concurrent runs=4
```

RQ1 preserves dependencies:

```text
phase 1: 12 Stage-A no-text parents
phase 2: 72 continuation/text/placebo/ablation runs
```

Phase 2 starts only after every Stage-A parent is complete. RQ2 runs its 24 BoW
and sentiment branches after RQ1 evaluation is complete.

Scheduler smoke validation with two GPUs and two slots per GPU:

```text
RQ1 Stage-A tasks       = 12, 3 per worker
RQ1 Stage-B/ablation    = 72, 18 per worker
RQ2 representation runs = 24, 6 per worker
```

If two runs exceed one GPU's memory, restart the same pipeline timestamp with:

```bash
RUN_TS=<existing_ts> GPU_IDS="0 1" RUNS_PER_GPU=1 \
bash scripts/rq123_london/start_full_recalculation_background.sh
```

Completed runs and completed master stages are reused.

## Full Background Command

Run from this worktree root:

```bash
cd /home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone

GPU_IDS="0 1" \
RUNS_PER_GPU=2 \
bash scripts/rq123_london/start_full_recalculation_background.sh
```

Monitor:

```bash
bash scripts/rq123_london/monitor_full_recalculation.sh
```

The master pipeline performs:

```text
shared raw-data preflight
-> Europe/London raw-vol coverage scan
-> selected workbook enrichment
-> RQ1 prepare/train/evaluate
-> RQ2 prepare/train/evaluate
-> RQ3 scheduled-news analysis
```

Primary output registries:

```text
outputs/experiments/rq123_london_recalculation_<run_ts>/
outputs/experiments/rq1_pair_text_raw_vol_continuation_london_<run_ts>/
outputs/experiments/rq2_pair_representation_raw_vol_continuation_london_<run_ts>/
outputs/experiments/rq3_scheduled_news_regime_raw_vol_london_<run_ts>/
```

## Verification

Completed targeted checks:

```text
Factiva timestamp and DST tests
surface config tests
merge_svi / merge_vol / merge_params tests
RQ1 pair continuation tests
RQ2 representation continuation tests
RQ3 event/news-quiet/scheduled-news tests
Python compileall
shell syntax checks
git diff --check
```

Actual full-workbook timestamp audit:

```text
rows                         = 14900
timestamp parse status ok    = 14900
LP rows with embedded GMT    = 3171
GMT matches within 2 minutes = 3094
match rate                   = 97.5717%
required rate                = 95%
status                       = ok
```

The expensive coverage scan and model matrix are intentionally not started by
the implementation step.
