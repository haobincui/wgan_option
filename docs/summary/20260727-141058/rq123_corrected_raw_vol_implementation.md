# RQ1-RQ3 Corrected Raw-Vol Implementation

## Metadata

```text
implementation_started_utc: 2026-07-27T14:10:58Z
worktree: /home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone
branch: rq123_corrected_raw_support
base_commit: e5baccc9ae4244baf9216a1445e07b758d46df74
backup_tag: pre-rq123-audit-cleanup-20260727-135547
```

## Scope

This implementation rebuilds the active RQ1-RQ3 pipeline around local-support
raw TY option implied-volatility surfaces. It implements the decisions recorded
in:

```text
docs/summary/20260727-130201/rq123_input_and_methodology_audit.md
```

The active research sequence is:

1. RQ1: incremental LP-text value relative to a controlled no-text model.
2. RQ2: LP semantic embeddings relative to BoW log-count and frozen ChatGPT
   sentiment scores.
3. RQ3: conditional predictive robustness around scheduled news releases.

This implementation does not add SSVI/SSVI robustness, does not restore RQ4,
and does not interpret RQ3 as a causal monetary-policy event study.

## Required Corrections

- Disjoint five-minute half-open current and target windows.
- Deterministic CBOT Treasury-option expiration and symbol-year resolution.
- Per-option last-prior matching futures trade with a finite staleness cap.
- Black-76 inversion using a frozen, no-lookahead Treasury curve.
- OTM-only option observations and at least two expiry slices per surface.
- Fold-train-only local grids and per-sample observed-support masks.
- Support-masked training losses, MAE, probabilistic scores, and arbitrage
  diagnostics; unsupported raw-vol 7d ATM metrics are removed.
- Reproducible text lineage, explicit missing-text masks, duplicate-text
  sensitivity, common random numbers, variogram score, and Monte Carlo error.
- One active set of RQ1/RQ2/RQ3 orchestration scripts and one resumable
  background pipeline.

## Implementation Record

The following sections are appended as work completes:

```text
implemented files
reference-data SHA256
test commands and results
corrected dataset counts
fold-specific support/grid counts
known limitations
final background command
```

## Implemented Components

### Corrected Raw-Vol Labels

Reusable market and label logic is implemented under:

```text
src/wgan_option/market/
src/wgan_option/surface_generation/data_helperd/
src/wgan_option/surface_generation/backend/
```

The corrected path now enforces:

```text
current trade window = [t - 5min, t)
target trade window  = [t, t + 5min)
pricing model        = Black76
underlying match     = last prior TY futures trade, <= 60 seconds stale
option observations  = OTM only, positive volume
IV aggregation       = volume-weighted median
minimum expiries     = 2
Factiva timezone     = Europe/London
```

The raw dataset validator hard-fails unless the resolved settings, temporal
audit, accepted precalibration rows, and exact frozen-rate-curve SHA256 all
pass.

### Local-Support Model Input

Implemented in:

```text
src/film_wgan/support.py
src/film_wgan/config.py
src/film_wgan/data.py
src/film_wgan/models.py
src/film_wgan/trainer.py
src/film_wgan/inference.py
```

Each rolling fold derives its grid and support only from train pairs. The
neural input has two channels:

```text
channel 0 = log implied volatility
channel 1 = observed-support mask
```

Normalization, reconstruction loss, validation MAE, probabilistic scores, and
test metrics use the support mask. The raw-vol primary point metrics are:

```text
surface_mae
short_atm_mae
supported_shortest_atm_abs_err
```

`7d ATM` is intentionally disabled because 7 days lies below observed raw
maturity support. Broad-grid calendar/butterfly summaries are marked
`not_reported_for_irregular_raw_support`; arbitrage-constrained SSVI is outside
this implementation scope.

### Text, Stochastic, And Lineage Controls

Implemented in:

```text
src/film_wgan/text_lineage.py
scripts/rq123/audit_text_lineage.py
scripts/rq123/audit_sentiment_scores.py
```

The audit records row-specific fallback article IDs, exact text/embedding
hashes, 64-bit SimHash near-duplicate components, and explicit exclusions for
empty LP text. RQ1 reports exact- and near-duplicate exclusion sensitivities.

Validation and inference use common random numbers keyed by:

```text
(seed, surface_pair_id, scenario_draw)
```

Outputs include energy score, variogram score, interval calibration, scenario
spread, and Monte Carlo standard error. The ChatGPT sentiment artifact is
treated as a frozen compressed baseline, not ground truth. The pipeline creates
a deterministic 120-row human annotation template and 50-article, two-repeat
stability template. Their status remains pending until a reviewer and
independent repeat calls complete them.

### Corrected RQ Inference

RQ1:

```text
scripts/rq1_pair/
```

- shared Stage-A no-text parent;
- equal-budget continued no-text, LP residual, shuffled LP, FiLM/concat
  ablations;
- four non-overlapping 2023 outer quarters and seeds 42/202/404;
- trading-day cluster bootstrap, seed tests, and daily DM/HAC;
- no best-seed selection.

RQ2:

```text
scripts/rq2_pair/
```

- imports the exact RQ1 parent/LP/no-text artifacts by SHA256;
- train-fold-only BoW vocabulary/log-count/PCA;
- train-fold-only scaling for frozen ChatGPT scores;
- LP-vs-BoW and LP-vs-sentiment primary contrasts with Holm adjustment,
  cluster bootstrap, DM/HAC, and seed-level tests.

RQ3:

```text
scripts/rq3/scheduled_news_regime.py
```

- uses all frozen OOS news-origin predictions as the primary sample;
- estimates scheduled-release conditional text advantage with event-family and
  fold fixed effects plus forecast-origin raw-support controls;
- release/trading-day cluster-robust inference;
- Giacomini-White conditional predictive-ability and overall DM/HAC tests;
- preserves greedy scheduled-vs-ordinary matching only as secondary
  robustness;
- does not make a causal event-study claim.

## Input And Output Reproducibility

The canonical pipeline is:

```text
scripts/rq123/
```

Before dataset construction it snapshots, hard-links or copies, and hashes:

```text
728 daily raw option files covering 2022-2023
Factiva news/LP workbook
BoW and ChatGPT-score artifacts/cache
RQ1/RQ2/RQ3/raw configs
frozen Treasury curve
CME expiry-rule reference
scheduled macro event calendar
```

It also saves the git commit/status, binary worktree patch, untracked-file
list, run parameters, per-stage status, complete output manifest, and final
cross-RQ validation. Final validation re-hashes every source snapshot.

The old broad glob also matched `ty_plus_merged.csv.gz` after the daily files,
which duplicated the source stream and could not safely contribute to already
closed target windows. The corrected RQ pipeline accepts only:

```text
data/raw/option_data/0#TY+/0#TY+_202[23]-*.csv.gz
```

and validates an exact count of 728 daily inputs; the aggregate file is
explicitly excluded.

Superseded orchestration was preserved under:

```text
scripts/archive/pre_corrected_20260727/
```

The active project root and `scripts/` directory now point users to only the
corrected workflow.

## Reference Data SHA256

```text
US Treasury 2022 official CSV:
c33cb2758e5e34c2c23b7ab759d18681091860f3f9e5960112859f64f31cd814

US Treasury 2023 official CSV:
edb7f3904e11dcaa502cdcc03ad9b1b16c893802d166ce9bc5fa1118251249d4

Combined 2022-2023 frozen curve:
919f4d288e6f5212156f7fafadfba76b208698d9c091669c3aaff7c30f305db8

CME TY monthly expiry-rule reference:
41ccef88d9b03968a315ac88526820f2cdfcde01c8938519f080cf634f4602e3
```

## Verification

Executed with:

```text
/home/haobin_cui/.conda/envs/py312/bin/python
PYTHONPATH=src
```

Results:

```text
market/raw-support/window/news/audit tests: 33 passed
FiLM-WGAN/support tests:                 29 passed, 1 skipped
RQ1/RQ2/RQ3 integration tests:          28 passed
new sentiment/snapshot/raw-audit tests:   4 passed
compileall:                               passed
active shell syntax checks:               passed
git diff --check:                         passed
```

Full frozen-workbook smoke audits:

```text
news rows                            = 14900
usable LP rows                       = 14872
excluded empty-LP/nonzero-vector rows= 28
missing ArticleID rows               = 149
exact embedding duplicate groups     = 266
exact LP text duplicate groups       = 839
near SimHash component groups        = 1573

text-lineage elapsed / peak RSS       = 46.45s / 1.30GB
sentiment-audit elapsed / peak RSS    = 67.96s / 0.89GB
manual sentiment audit sample         = 120
repeat-score requests                 = 50 articles x 2
```

The corrected raw dataset and model matrix were not started during code
implementation. Therefore corrected usable-pair counts, fold support counts,
selected epochs, and RQ1/RQ2/RQ3 numeric findings are intentionally pending.

## Final Background Command

Run from this worktree root:

```bash
cd /home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone
conda activate py312

GPU_IDS="0 1" \
RUNS_PER_GPU=2 \
PUBLICATION_AVAILABILITY_LAG_MINUTES=0 \
bash scripts/rq123/start_corrected_pipeline_background.sh
```

Do not set a global `CUDA_VISIBLE_DEVICES`: the launcher assigns physical GPU
IDs to individual workers.

Monitor:

```bash
bash scripts/rq123/monitor_corrected_pipeline.sh
```

Availability-lag sensitivity is a separate expensive matrix and should start
only after the primary run passes final validation:

```bash
LAGS="1 2 5" GPU_IDS="0 1" RUNS_PER_GPU=2 \
bash scripts/rq123/start_availability_lag_sensitivity_background.sh
```

## Known Limitations

- This remains 2023 rolling-development evidence; a 2024+ frozen confirmation
  sample is still required.
- Black76 is the frozen primary inversion convention. American early-exercise
  sensitivity is not implemented in this pass.
- Human sentiment validation and independent repeat scoring require external
  completion; templates are archived and status is explicit.
- Raw-vol claims are restricted to the train-derived observed-support domain.
- RQ3 identifies conditional predictive performance, not causal semantic or
  monetary-policy effects.
- Arbitrage-constrained SSVI robustness was explicitly deferred.
