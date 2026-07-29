# Raw-Vol Interpolation RQ1/RQ2 Workflow

This note documents the raw-vol alternative-input workflow for thesis analysis.
It repeats the RQ1/RQ2 FiLM WGAN experiment design using volatility surfaces
reconstructed directly from raw option implied-volatility points.

## Purpose

The previous RQ1/RQ2 experiments use SVI-reconstructed volatility surfaces:

```text
backward SVI fit -> current_surface
forward SVI fit  -> target_surface
```

The raw-vol workflow replaces the market-state input/target construction with:

```text
raw option prices -> implied vol points -> raw interpolation surface
```

The text representations and downstream FiLM WGAN training protocol remain the
same. This is a robustness / alternative input-space experiment, not a strict
same-sample replacement for the SVI-vol main results.

## Raw-Vol Surface Construction

The interpolation logic is implemented in:

```text
src/quantlib/vol_surface/algo/raw_surface.py
```

The raw surface parameter schema is:

```text
business_days
percent_strikes
implied_vols
```

The reconstruction rule is fixed as:

```text
strike direction:
  average duplicate percent-strike observations inside each maturity slice
  linearly interpolate implied-vol total variance over percent strike
  clamp outside observed strike bounds

maturity direction:
  linearly interpolate total variance between maturity slices
  convert total variance back to implied volatility
```

This deliberately avoids SVI/SABR/cubic smoothing. The goal is to isolate the
effect of using raw observed IV points as the surface source.

## Dataset Policy

The sample policy is:

```text
raw available first
```

That means the raw-vol workbook uses every news sample for which both current
and target raw-interpolated surfaces pass the quality filters. It does not
force the raw-vol sample set to match the previous 3711 SVI-vol samples.

Important implication for the thesis:

```text
Raw-vol results should be reported as robustness / alternative input evidence.
They should not be directly substituted for the SVI-vol main tables unless the
sample-set difference is explicitly discussed.
```

## Dataset Build Commands

One-command full background pipeline:

```bash
CUDA_VISIBLE_DEVICES=1 \
bash scripts/raw_vol/run_full_raw_vol_rq_background.sh
```

This starts a background driver that runs coverage scan, RQ2 workbook
enrichment, multi-seed training, generate-result, comparison archive building,
and zip packaging in sequence.

Monitor:

```bash
tail -f outputs/experiments/raw_vol_rq1_rq2_<run_ts>/logs/driver/full_raw_vol_pipeline.log
```

Single dataset build:

```bash
ENV_NAME=py312 \
RUN_TS=raw_vol_rq_YYYYMMDD_HHMMSS \
WINDOW_MINUTES=5 \
MIN_STRIKES_PER_EXPIRY=2 \
bash scripts/raw_vol/prepare_raw_vol_dataset.sh
```

The paper-facing raw-vol dataset uses a bounded fallback policy:

```text
option_filter_mode = otm_preferred_itm_fallback
max_itm_moneyness_distance = 0.05
```

For each maturity/strike, valid OTM observations take precedence. An ITM
observation is eligible only when `abs(strike / futures - 1) <= 0.05` and no
valid OTM observation exists at that strike. Every maturity containing ITM
fallback points must retain at least one OTM strike, and the number of ITM
fallback strikes cannot exceed the number of OTM strikes. Black-76 price
bounds, the 60-second last-prior futures constraint, positive-volume
weighting, and the pre-calibration IV cap remain unchanged.

Coverage scan:

```bash
ENV_NAME=py312 \
WINDOW_CANDIDATES="3 5 10 15 30 60" \
bash scripts/raw_vol/coverage_scan_raw_vol.sh
```

The scan records:

```text
raw_vol_coverage_scan.csv
raw_vol_selected_dataset.json
```

By default the scan also updates:

```text
data/processed/raw-excel/rq_raw_vol_selected -> <selected raw-vol run>
```

The raw-vol training launchers use this selected path unless `DATA_PATH` is
overridden explicitly.

Selection rule:

```text
primary = maximum usable_pairs
tie-breaker = smaller window_minutes, then smaller min_strikes_per_expiry
hard stop = usable_pairs < 100
low-power warning = usable_pairs < 1000
```

After selecting a raw-vol run directory, enrich it with RQ2 text features:

```bash
FEATURE_DIR=data/processed/text_features/rq2/<feature_run> \
bash scripts/raw_vol/enrich_raw_vol_rq2_text.sh data/processed/raw-excel/<raw_run>
```

Expected dataset artifacts:

```text
data/processed/raw-excel/<raw_run>/
  surface-raw-excel.json
  surface-raw-excel-precalib-points.csv
  surface-raw-excel.log
  surface-resolved_config.yaml
  merged_vol.xlsx
  merged_vol_rq2_text.xlsx
  raw_vol_dataset_validation.json
```

## Training Protocol

Base config:

```text
configs/film_wgan/train_raw_vol_textbase.yaml
```

Allowed model overrides:

```text
text:
  text_embedding_mode=lp
  normalize_text_embedding=true

no_text:
  text_embedding_mode=none
  normalize_text_embedding=false

bow:
  text_embedding_mode=bow
  normalize_text_embedding=true

llm_sentiment:
  text_embedding_mode=llm_sentiment
  normalize_text_embedding=true
```

Single-run launchers:

```bash
DATA_PATH=data/processed/raw-excel/<raw_run>/merged_vol_rq2_text.xlsx \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/rq1_raw_vol/run_film_wgan_text_shortatm.sh

DATA_PATH=data/processed/raw-excel/<raw_run>/merged_vol_rq2_text.xlsx \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/rq1_raw_vol/run_film_wgan_notext_shortatm.sh

DATA_PATH=data/processed/raw-excel/<raw_run>/merged_vol_rq2_text.xlsx \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/rq2_raw_vol/run_film_wgan_bow.sh

DATA_PATH=data/processed/raw-excel/<raw_run>/merged_vol_rq2_text.xlsx \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/rq2_raw_vol/run_film_wgan_llm_sentiment.sh
```

Multi-seed launch:

```bash
EXP_ROOT=outputs/experiments/raw_vol_rq1_rq2_<run_ts> \
DATA_PATH=data/processed/raw-excel/<raw_run>/merged_vol_rq2_text.xlsx \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/raw_vol_multiseed/start_training_matrix_background.sh
```

Monitor:

```bash
EXP_ROOT=outputs/experiments/raw_vol_rq1_rq2_<run_ts> \
bash scripts/raw_vol_multiseed/monitor_training.sh
```

Generate all-split JSON after training:

```bash
EXP_ROOT=outputs/experiments/raw_vol_rq1_rq2_<run_ts> \
DATA_PATH=data/processed/raw-excel/<raw_run>/merged_vol_rq2_text.xlsx \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/raw_vol_multiseed/start_generate_matrix_background.sh
```

Build comparison archive:

```bash
EXP_ROOT=outputs/experiments/raw_vol_rq1_rq2_<run_ts> \
bash scripts/raw_vol_multiseed/build_comparison_archive.sh
```

## Comparison Outputs

Primary metrics:

```text
surface_mae
short_atm_mae
atm7_abs_err
```

The comparison archive recomputes sample-level metrics from generated JSON
rather than relying only on `summary.csv`.

Expected final outputs:

```text
outputs/experiments/raw_vol_rq1_rq2_<run_ts>/
  comparisons/sample_metrics_all_models_all_seeds.csv
  comparisons/model_metric_summary_by_seed.csv
  comparisons/pairwise_diff_summary_by_seed.csv
  comparisons/seed_level_tests.csv
  comparisons/sample_level_tests_by_seed.csv
  final_tables/raw_vol_eval_main_metrics_by_seed.csv
  final_tables/raw_vol_seed_level_tests.csv
  final_tables/raw_vol_text_vs_baselines.csv
  final_tables/raw_vol_best_seed_eval_comparison.csv
  validation_summary.json
  manifest.csv
```

Checkpoint selection:

```text
epoch > 10
selection metric = lowest val_mae_gap_vs_current
checkpoint = film_wgan_best_val_mae_gap_vs_current.pt
```

Pairwise direction:

```text
baseline_minus_text > 0 => LP text lower MAE / better
baseline_minus_text < 0 => baseline lower MAE / better
```

## Reproducibility Notes

The experiment root snapshots:

```text
input config
raw-vol workbook
RQ2 feature artifacts
script snapshots
git state
training resolved configs
selected checkpoint registry
generate-result configs
comparison tables
```

For thesis writing, report the raw-vol usable sample count from:

```text
data/processed/raw-excel/<raw_run>/raw_vol_dataset_validation.json
```

and keep it separate from the SVI-vol sample count.

## Relaxed News-to-Market Alignment

The strict dataset uses the Factiva availability minute as the exact forecast
origin. The relaxed dataset preserves the same five-minute forecasting
semantics while allowing a news item to wait for the next complete market
surface pair:

```text
current surface window = [origin - 5min, origin)
target surface window  = [origin, origin + 5min)

origin = earliest valid pair origin satisfying:
news_available_time <= origin <= news_available_time + 72h
```

Classification:

```text
shift = 0 minutes       -> exact
shift = 1..15 minutes   -> intraday_shift
shift = 16..4320 minutes -> session_shift
```

An origin before news availability is invalid. A market origin is valid only
when raw-vol surfaces exist at both `origin` and `origin + 5min`. Multiple
articles can map to one pair; downstream `surface_pair` sampling deduplicates
articles by `ArticleID` and applies the existing equal-weight mean-L2 text
pooling. One article is never assigned to multiple pairs.

Candidate anchors are constructed only from raw option observations with
positive price and volume. Futures-only minutes and invalid option rows cannot
create calibration jobs, although futures trades within an option-triggered
five-minute window remain available for the Black-76 underlying match.

The reusable index is stored under:

```text
data/processed/raw-market-index/<index_id>/
  market_surface_index.sqlite
  market_surface_audit.csv
  precalibration_audit.csv
  valid_5m_pair_index.csv
  source_manifest.csv
```

The aligned dataset is stored separately from strict outputs:

```text
data/processed/raw-excel-relaxed/<run_ts>/
  news_market_alignment.csv
  unmatched_news.csv
  alignment_coverage_summary.csv
  surface-raw-excel.json
  surface-raw-excel-precalib-points.csv
  merged_vol.xlsx
  merged_vol_rq2_text.xlsx
  validation_summary.json
```

Training is blocked unless the relaxed `gan_input_ready` sheet increases both
article-row coverage and unique surface-pair coverage relative to the strict
baseline. Main tables must report `exact`, `intraday_shift`, and
`session_shift` strata because cross-session news age is a different economic
regime even though all rows use the same fixed five-minute target horizon.

Background run:

```bash
GPU_IDS="0 1" RUNS_PER_GPU=2 \
bash scripts/rq123/start_relaxed_time_pipeline_background.sh
```

Set `RUN_DOWNSTREAM=0` to build and validate only the index/workbook before
starting RQ1-RQ3 training.
