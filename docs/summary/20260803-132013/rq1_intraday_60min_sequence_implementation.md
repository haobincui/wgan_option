# RQ1 Intraday 60-Minute Sequence Implementation

Implementation timestamp (UTC): `2026-08-03 13:20:13`

## Scope

This branch adds a new RQ1 development experiment without changing the legacy
single-surface `rq1_pair` workflow. The research question is whether LP text
observed during the previous hour improves the next five-minute raw-vol surface
forecast beyond a temporal no-text model.

```text
history = S_(t-55), ..., S_t at five-minute positions
target  = S_(t+5)
news    = publications available in (t-60, t]
```

The implementation is on:

```text
branch:   rq1-intraday-60m-sequence
worktree: /home/haobin_cui/research_files_space_2/wgan_option-rq1-intraday-60m
base:     origin/master d9a9a07
```

This remains a 2023 rolling-development analysis. It must not be described as
an untouched 2024+ confirmation test.

## Frozen Data Rules

- Surface source: raw implied-volatility points reconstructed by the existing
  linear strike / linear total-variance maturity interpolation.
- Market index: schema version 2, `surface_model=raw` only, five-minute
  backward windows, positive-price-and-volume option minutes.
- Origin and target: a valid raw pair `t -> t+5` on the five-minute grid.
- Session rule: the complete interval `[t-60,t+5]` must remain inside one CME
  Treasury Globex continuous session, including frozen holiday closures.
- History: 12 positions, at least 6 valid; missing surfaces are zero with
  separate support and step-valid masks. `S_t` is always present.
- News timezone: `Europe/London` with zero publication availability lag.
- Future news is rejected by construction because the lookup interval ends at
  `t` with an inclusive upper boundary.
- News buckets: `[0,15)`, `[15,30)`, and `[30,60)` minutes before `t`.
- Frozen inputs use reflink copies when supported and ordinary copies otherwise;
  hard links are not used. A non-empty SQLite WAL blocks preparation, and the
  frozen market index is subsequently opened as immutable read-only.

Input SHA256 values used by the full prepare validation:

```text
market_surface_index.sqlite:
9bcefb2afad7fba841926cb318521dcf3c96ec3160c39c8e1df3a1119e07301c

news_with_openai_embeddings_large.xlsx:
88bd2fdc9be55cd0b5c714645933bc53c4ce4f6748315d734cdf454dd8f38aa6

cme_treasury_globex_closures_2022_2023.csv:
cb043278433002d58efb11402c1f84075736110f580ede070e6ac7d39b0cefd3
```

## Text Representation

Within each time bucket:

```text
unique ArticleID
-> L2-normalize each LP embedding
-> mean
-> final L2 normalization
-> fold-train-only PCA-128 without whitening
```

The condition dimension is:

```text
3 * 128 semantic coordinates
+ 3 standardized log1p bucket counts
+ has_news
+ standardized latest-news recency
= 389
```

The same train-fold PCA and scalar transforms are reused by every variant in a
fold. `no_text` is exactly 389 zeros with `has_text=0`. `news_count` retains
only arrival/count fields. `shuffled_lp` preserves arrival fields while using
train-only donor ArticleIDs; it cannot draw from validation/test or the same
ArticleID.

Repeated adjacent windows can contain the same set of articles. PCA fits each
unique bucket ArticleID set once to avoid mechanically overweighting repeated
news, while its lineage metadata hashes the complete ordered set of training
surface-pair IDs and separately hashes the unique pool keys.

The source workbook contains 27 duplicated ArticleIDs with conflicting LP
embedding hashes. The frozen rule keeps the earliest available row, then the
lowest `news_row_id`, and records every conflict in
`article_id_embedding_conflicts.csv`.

## Dataset Validation

The full real-data prepare run reproduced:

```text
five-minute sequence candidates = 4,394
with news in previous 60m        = 2,936
without news                    = 1,458
complete 12/12 histories         = 473
```

The plan's fold counts are pre-support candidates. A train-only 16x16 raw
support grid can have no current/target support intersection for some pairs;
these rows are explicitly written to `support_grid_dropped_samples.csv`.
There is no silent deletion.

| Fold | Candidate train/val/test | Retained train/val/test | Support drops |
|---|---:|---:|---:|
| 2023Q1 | 1547 / 551 / 361 | 1227 / 446 / 286 | 320 / 105 / 75 |
| 2023Q2 | 2098 / 361 / 371 | 1645 / 296 / 280 | 453 / 65 / 91 |
| 2023Q3 | 2459 / 371 / 658 | 1963 / 280 / 508 | 496 / 91 / 150 |
| 2023Q4 | 2830 / 658 / 906 | 2246 / 511 / 740 | 584 / 147 / 166 |

The retained 2023 outer-test sample count is `1,814`. This differs from the
pre-support count `2,296`; downstream registries and statistics read the actual
fold manifests rather than hard-coding either number.

Short-ATM eligibility is also recorded per sample. A row without supported
cells in `abs(moneyness-1)<=0.02, maturity<=21` is reported as missing rather
than as zero error.

| Fold | Train eligible/total | Validation eligible/total | Test eligible/total |
|---|---:|---:|---:|
| 2023Q1 | 1084 / 1227 | 384 / 446 | 264 / 286 |
| 2023Q2 | 1476 / 1645 | 267 / 296 | 254 / 280 |
| 2023Q3 | 1792 / 1963 | 247 / 280 | 426 / 508 |
| 2023Q4 | 2040 / 2246 | 433 / 511 | 663 / 740 |

## Model And Training

The sequence condition has shape `[B,12,3,16,16]`:

```text
channel 0 = normalized log IV
channel 1 = raw observed-support mask
channel 2 = step-valid mask
```

A shared 2D CNN encodes each frame. A masked `GRUCell` updates state only for
valid steps. The temporal state feeds a persistence-anchored stochastic delta
generator. LP features enter through a small residual FiLM adapter with a zero
initialized scalar gate. At gate zero, conditioned and no-text generator output
is bitwise identical for the same history/noise.

Training matrix for seeds `42,202,404`:

```text
history_no_text_parent
history_no_text_continued
current_only_no_text
history_news_count
history_matched_lp
history_shuffled_lp

4 folds * 3 seeds * 6 variants = 72 runs
```

Stage A trains the history no-text parent. Stage B continued/count/LP/shuffled
branches load the exact same fold/seed parent generator; each critic starts
fresh from the same seed. Text branches freeze the temporal/surface backbone
for five epochs, then use backbone LR `2e-6` and adapter LR `2e-5`.
Checkpoints are selected only after epoch 10 by minimum validation surface MAE.
The current-only control masks the first 11 steps at the dataset and inference
boundaries, but normalization remains the same train-fold normalization fitted
from the complete raw history. This isolates history availability from scaling.

## Statistics

Primary difference:

```text
continued_no_text_error - matched_lp_error
positive => LP text improves the forecast
```

Samples are first matched by fold, seed, and `surface_pair_id`; differences are
then averaged across seeds for each origin. Inference uses a trading-day cluster
bootstrap (10,000 iterations, seed `20260722`), daily HAC/DM with lag 5, and
seed-level paired t/Wilcoxon diagnostics. No best-seed selection is permitted.
Outputs include the pair-level win/tie rate and effective paired-origin count.
Holm adjustment is applied separately within each metric and preregistered
family: primary incremental text, control ablations, and persistence context.

Primary point metrics are `surface_mae` and `short_atm_mae`, where short ATM is
`abs(moneyness-1)<=0.02` and maturity `<=21` business days. The raw-vol 7-day
metric is excluded because seven days is not consistently observed support.

## Implemented Files

Core:

```text
src/film_wgan/config.py
src/film_wgan/data.py
src/film_wgan/sequence_data.py
src/film_wgan/models.py
src/film_wgan/trainer.py
src/film_wgan/inference.py
```

Experiment layer:

```text
configs/film_wgan/train_rq1_sequence_textbase.yaml
scripts/rq1_sequence/dataset_builder.py
scripts/rq1_sequence/rq1_sequence_experiment.py
scripts/rq1_sequence/prepare_experiment.sh
scripts/rq1_sequence/run_training_matrix.sh
scripts/rq1_sequence/start_training_matrix_background.sh
scripts/rq1_sequence/monitor_training.sh
scripts/rq1_sequence/run_results_pipeline.sh
scripts/rq1_sequence/start_results_pipeline_background.sh
scripts/rq1_sequence/package_experiment.sh
```

Tests:

```text
tests/test_standalone_wgan/test_rq1_sequence.py
```

## Commands

Prepare the frozen experiment:

```bash
conda activate py312
bash scripts/rq1_sequence/prepare_experiment.sh
```

Start training in the background, with one run per GPU by default:

```bash
GPU_IDS="0 1" RUNS_PER_GPU=1 \
bash scripts/rq1_sequence/start_training_matrix_background.sh
```

Single-GPU examples:

```bash
GPU_IDS="0" RUNS_PER_GPU=1 \
bash scripts/rq1_sequence/start_training_matrix_background.sh

GPU_IDS="0" RUNS_PER_GPU=2 \
bash scripts/rq1_sequence/start_training_matrix_background.sh
```

Monitor and build results:

```bash
bash scripts/rq1_sequence/monitor_training.sh

GPU_IDS="0 1" RUNS_PER_GPU=1 \
bash scripts/rq1_sequence/start_results_pipeline_background.sh
```

All scripts discover the exact root from
`outputs/experiments/rq1_intraday_sequence_latest.txt`; set
`EXPERIMENT_ROOT=...` to resume a specific archive.

## Verification Performed

```text
10 RQ1 sequence tests passed, including toy raw SQLite and checkpoint inference
41 targeted sequence/raw-support/legacy tests passed, 1 skipped
61 repository tests passed
real-data prepare completed for all four folds
real-data no-text/matched/count/shuffled/current-only loaders passed
temporal trainer dry-run passed on 1,227/446/286 Q1 samples
one discriminator and one generator update passed with finite losses
temporal checkpoint load and generate-result smoke path passed
parallel task audit produced 12 Stage-A + 60 second-phase tasks
Ruff checks passed
compileall passed
shell syntax checks passed
git diff --check passed
```

## Known Limitations

- Results remain development evidence because 2024+ news is unavailable.
- Train-only rectangular support grids reduce the candidate sample; both
  candidate and retained counts must be reported.
- The first version predicts only `t+5`; direct 15/30-minute heads remain future
  extensions.
- Parallel capacity depends on GPU memory. Start at one run per GPU, then raise
  `RUNS_PER_GPU` only after checking actual memory consumption.
