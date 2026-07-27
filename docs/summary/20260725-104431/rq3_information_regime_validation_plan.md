# RQ3 Information-Regime Validation Plan

> Design date: 2026-07-25 UTC  
> Methodology re-review: 2026-07-25 UTC  
> Evidence status: development design based on the completed RQ1/RQ2
> raw-vol continuation experiments  
> Controlling protocol: Section 13. Where Sections 1-12 conflict with
> Section 13, Section 13 takes precedence.

The first twelve sections preserve the initial literature-backed design so the
evolution of the research method remains auditable. A subsequent methodology
review identified limitations in making FOMC or zero-text quiet samples the
primary source of inference. Section 13 records the revised design that should
be implemented and used for thesis-facing RQ3 results.

## 1. Research Positioning

RQ1 and RQ2 establish the first two steps of the thesis argument:

```text
RQ1: Does text provide incremental predictive information?
RQ2: Does semantic LP representation preserve more useful information than
     BoW log-count and ChatGPT sentiment representations?
RQ3: Under which information regimes is that semantic value strongest?
```

The revised RQ3 formulation is:

> Is the incremental predictive value of LP semantic embeddings stronger
> during pre-specified scheduled high-information news windows than during
> observably similar ordinary-news windows?

RQ3 is a conditional predictive-ability and information-regime robustness
test. It is not, by itself, a causal estimate of the effect of news on the
implied volatility surface.

The central quantities are:

```text
text_advantage
= no_text_error - lp_text_error

scheduled_news_increment
= text_advantage(scheduled_news)
  - text_advantage(matched_ordinary_news)
```

Interpretation:

```text
text_advantage > 0
=> LP text has lower forecast error than no-text

scheduled_news_increment > 0
=> the marginal value of LP text is larger in scheduled high-information
   windows than in matched ordinary-news windows
```

A true no-news quiet comparison remains useful, but it is an appendix
robustness experiment requiring mixed news/no-news retraining. It is not the
primary frozen-model RQ3 test.

## 2. Literature Support

### 2.1 Text and volatility forecasting

Rahimikia, Zohren, and Poon show that financial-news representations contain
out-of-sample volatility information and that the improvement is stronger on
high-volatility or volatility-jump days. They also find that combining a news
signal with an established quantitative volatility benchmark is more useful
than using the news signal alone. This directly supports:

```text
no-text quantitative backbone
+ residual semantic text module
+ separate normal/high-volatility evaluation
```

Reference:
[Rahimikia, Zohren, and Poon, Realised Volatility Forecasting: Machine Learning via Financial Word Embedding](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3895272)

### 2.2 Central-bank communication and option-implied distributions

Vergote and Puigvert Gutierrez analyze intraday, option-implied distributions
around ECB Governing Council communications. Their results indicate that the
entire implied distribution can react to central-bank communication and that
the press conference can provide information beyond the policy-rate decision.

This supports evaluating:

```text
full-surface MAE
short-ATM MAE
7d ATM error
energy score
coverage and calibration
scenario spread
```

It also supports separating the policy statement and press conference rather
than treating them as one event.

Reference:
[Vergote and Puigvert Gutierrez, Interest Rate Expectations and Uncertainty during ECB Governing Council Days](https://ideas.repec.org/a/eee/jbfina/v36y2012i10p2804-2823.html)

### 2.3 Policy actions versus policy communication

Gurkaynak, Sack, and Swanson show that a single target-rate surprise does not
fully describe FOMC announcement effects. A separate future-policy-path factor,
closely associated with FOMC statements, has substantial effects on longer-term
Treasury yields. This supports testing semantic communication conditional on
the quantitative market state instead of treating the rate decision as the only
information channel.

Reference:
[Gurkaynak, Sack, and Swanson, Do Actions Speak Louder Than Words?](https://www.federalreserve.gov/econres/feds/do-actions-speak-louder-than-words-the-response-of-asset-prices-to-monetary-policy-actions-and-statements.htm)

### 2.4 High-frequency event windows

Nakamura and Steinsson identify monetary-policy and information effects using
changes in financial instruments over a 30-minute window around scheduled
Federal Reserve announcements. This supports a narrow, pre-specified FOMC
window, but it does not justify treating every news arrival as a causal policy
shock.

Reference:
[Nakamura and Steinsson, High-Frequency Identification of Monetary Non-Neutrality](https://academic.oup.com/qje/article-abstract/133/3/1283/4828341)

Fleming and Remolona document a two-stage Treasury-market response to public
information: an immediate price adjustment followed by continued trading,
volatility, and liquidity adjustment. This supports reporting the model's
five-minute horizon as primary and using a longer announcement window as a
separate robustness definition.

Reference:
[Fleming and Remolona, Price Formation and Liquidity in the U.S. Treasury Market](https://doi.org/10.1111/0022-1082.00172)

Ederington and Lee distinguish scheduled information releases, for which
uncertainty may be priced before the release and resolved afterward, from
unscheduled releases, which can increase option-implied uncertainty. Scheduled
FOMC events and general news should therefore not be pooled into a single event
category.

Reference:
[Ederington and Lee, The Creation and Resolution of Market Uncertainty](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/creation-and-resolution-of-market-uncertainty-the-impact-of-information-releases-on-implied-volatility/7D6E476DAEBC8D2EEBF6EC0E6320F5C8)

### 2.5 Conditional forecast evaluation

Giacomini and White provide a direct framework for asking whether relative
forecast performance changes with the economic information state. This is more
appropriate for RQ3 than separately running unrelated t-tests in event and
quiet subsamples.

Reference:
[Giacomini and White, Tests of Conditional Predictive Ability](https://doi.org/10.1111/j.1468-0262.2006.00718.x)

Diebold and Mariano support inference on paired forecast-loss differentials
while allowing serial dependence and non-quadratic loss functions.

Reference:
[Diebold and Mariano, Comparing Predictive Accuracy](https://doi.org/10.1080/07350015.1995.10524599)

For the very small number of FOMC clusters, conventional cluster-robust
asymptotics can over-reject. Cameron, Gelbach, and Miller motivate bootstrap
procedures that explicitly account for clustered dependence and small-cluster
problems.

Reference:
[Cameron, Gelbach, and Miller, Bootstrap-Based Improvements for Inference with Clustered Errors](https://www.nber.org/papers/t0344)

## 3. Current FOMC Sample Feasibility

The completed RQ2 rolling outer-test archive contains:

```text
independent test surface pairs = 1,597
2023 scheduled FOMC releases   = 8
```

Matching the frozen test pairs to the existing FOMC press-release calendar
produces:

| Event definition | Matched surface pairs | FOMC meetings represented |
|---|---:|---:|
| `[0,+5]` minutes | 7 | 7 of 8 |
| `[0,+30]` minutes | 13 | 8 of 8 |
| Nakamura-Steinsson `[-10,+20]` | 14 | 8 of 8 |
| Symmetric `[-30,+30]` | 16 | 8 of 8 |

Consequences:

1. The 2023 FOMC analysis is economically informative but statistically
   underpowered.
2. The event, rather than each article row, must be the primary inference unit.
3. The existing seven to eight meetings cannot support a conventional
   large-sample cluster p-value as the only evidence for RQ3.
4. A final confirmation should extend the frozen design to 2024 and later
   FOMC meetings.

## 4. Initial Two-Layer Design

This section records the initial design. After methodology re-review:

```text
RQ3-A FOMC analysis       = retained as a supporting case study
RQ3-B news versus quiet   = moved to an appendix requiring mixed-data retraining
revised primary analysis  = scheduled macro news versus matched ordinary news
```

The complete controlling design is specified in Section 13.

### 4.1 RQ3-A: Scheduled FOMC event validation

This layer reuses the frozen RQ1/RQ2 checkpoints. It does not select a new seed
or checkpoint based on FOMC performance.

Primary comparison:

```text
fomc_text_advantage
= no_text_surface_mae - lp_surface_mae
```

Primary forecast definition:

```text
forecast horizon = current surface to current + 5 minutes
FOMC event label = news timestamp in [statement release, release + 5 minutes]
```

Robustness event definitions:

```text
Nakamura-Steinsson window = [-10,+20] minutes
post-release window       = [0,+30] minutes
```

The 30-minute definition is an event-classification window. It does not turn a
five-minute model into a 30-minute forecasting model. A genuine 30-minute
forecast requires a separately defined target or a pre-specified sequence of
one-step forecasts.

The 14:00 Eastern FOMC statement and the 14:30 Eastern press conference should
be represented as separate event types:

```text
FOMC_STATEMENT
FOMC_PRESS_CONFERENCE
```

Recommended inference:

1. Average all eligible rows within each meeting and seed.
2. Average the paired model difference across the pre-specified seeds.
3. Report every meeting's text advantage.
4. Use an exact sign or event-level randomization test.
5. Report small-cluster bootstrap/HAC results only as supporting diagnostics.
6. Label the 2023 output `development_event_evidence`.

### 4.2 RQ3-B: Appendix-only general-news versus quiet comparison

A formal quiet comparison requires mixed news/no-news training. A model trained
only on news rows should not be evaluated on normalized zero text and treated
as if this were an in-distribution no-news input.

Sample definitions:

```text
news_event:
  existing usable news-surface pair

quiet:
  valid five-minute surface pair
  no news within +/-60 minutes
  no scheduled macro announcement within +/-60 minutes
```

Quiet controls should be matched without target information. Matching variables
must be available at the forecast origin:

```text
quarter
weekday
time of day, preferably within +/-15 minutes
current IV level
current surface slope and curvature
surface coverage and quality
recent pre-forecast volatility
```

Target jump, target MAE, and future liquidity must not be used for matching.

Primary matching uses no replacement. The selected quiet sample IDs and all
matching variables must be frozen in a manifest shared across models and seeds.

## 5. Controlled Model Inputs

All models must receive the same news-arrival indicator. Otherwise LP versus
no-text would mix semantic value with the value of merely knowing that news
arrived.

Recommended decomposition:

```text
common status feature:
  learned embedding of has_news

LP semantic feature:
  has_news * transformed LP embedding

no-text semantic feature:
  zero vector
```

The event label itself is used for evaluation, not supplied to the primary
forecast model. A scheduled-event-calendar indicator may be added only as a
separate robustness model and must be supplied to all representations.

Primary models:

```text
mixed continued no-text
mixed LP residual FiLM-WGAN
```

Secondary representation controls:

```text
mixed BoW residual FiLM-WGAN
mixed ChatGPT sentiment residual FiLM-WGAN
```

All Stage-B branches must:

```text
start from the same mixed-data Stage-A no-text checkpoint
use the same rolling folds and seeds
use the same training budget and losses
use the same validation checkpoint rule
use identical news/quiet manifests
```

## 6. Statistical Estimands

For test pair `i` and seed `s`:

```text
d_i,s = no_text_error_i,s - lp_error_i,s
```

Positive values mean that LP has lower error. Average the difference across
seeds before sample-level inference:

```text
d_i = mean_s(d_i,s)
```

Estimate the conditional predictive-ability specification:

```text
d_i = alpha
    + beta_news * ordinary_news_i
    + beta_fomc * FOMC_i
    + controls_i
    + fold fixed effects
    + error_i
```

Quiet is the omitted category:

```text
beta_news > 0
=> LP has greater incremental value for ordinary news than quiet

beta_fomc > 0
=> LP has greater incremental value for FOMC than quiet

beta_fomc - beta_news > 0
=> LP has additional value for FOMC beyond ordinary news
```

Primary test:

```text
metric   = surface_mae
contrast = text_advantage(news) - text_advantage(quiet)
```

Secondary tests:

```text
short_atm_mae
atm7_abs_err
energy_score
coverage and calibration
scenario_spread
LP versus BoW regime interaction
LP versus ChatGPT sentiment regime interaction
```

Inference:

```text
trading-day block/cluster bootstrap
Giacomini-White conditional predictive-ability test
DM/HAC loss-differential test
seed-level direction and paired tests
Holm correction for secondary metrics
```

## 7. High-Volatility and Jump Robustness

Two regimes should be distinguished.

Ex-ante high-volatility state:

```text
current-state volatility measure
> threshold estimated using fold training data only
```

This regime is known at the forecast origin and can support an operational
conditional-forecasting interpretation.

Ex-post jump diagnostic:

```text
current-to-target persistence surface error
> fold-training 90th percentile
```

This follows the high-volatility/jump motivation in Rahimikia et al., but the
label uses the realized target and is therefore diagnostic rather than
real-time tradable information.

Report:

```text
jump_increment
= text_advantage(jump) - text_advantage(non_jump)
```

## 8. Placebo and Falsification Tests

At minimum:

```text
FOMC dates shifted by +/-1 trading day
same-clock non-event dates
same-day mismatched text
lead text, which should not improve a valid real-time forecast
matched LP text versus zero-text counterfactual in the same LP checkpoint
```

The event advantage should weaken or disappear under the placebo definitions.
A placebo result as strong as the true event result invalidates a strong RQ3
interpretation.

## 9. Code and Output Design

Reusable model/data behavior belongs in:

```text
src/film_wgan/
  mixed news/quiet dataset support
  has_news status embedding
  semantic-feature masking
  train-only transforms
  metadata passthrough
```

RQ3-specific orchestration belongs in:

```text
scripts/rq3/
  event-calendar validation
  quiet candidate construction
  matching
  FOMC labels
  conditional predictive-ability statistics
  placebo construction
  archive generation
```

Recommended archive:

```text
outputs/experiments/rq3_information_regime_raw_vol_<run_ts>/
  inputs/
    event_calendars/
    mixed_split_manifests/
    quiet_matching_manifest.csv
    configs/
    git_state.txt
  training_runs/
  generated_results/
  comparisons/
    rq3_conditional_predictive_ability.csv
    rq3_fomc_event_level_results.csv
    rq3_news_quiet_differences.csv
    rq3_high_vol_jump_results.csv
    rq3_placebo_results.csv
  final_tables/
    thesis_rq3_primary_news_increment.csv
    thesis_rq3_fomc_event_validation.csv
    thesis_rq3_distributional_metrics.csv
  docs/
  validation_summary.json
  manifest.csv
```

## 10. Pre-Specified Decision Rule

Formal support for the general news/quiet RQ3 hypothesis requires:

```text
mean event_increment > 0
95% trading-day cluster confidence interval > 0
Holm-adjusted p-value < 0.05
at least 2 of 3 development seeds have the same direction
placebo tests do not reproduce the result
```

FOMC 2023 results should be treated as meeting-level supporting evidence because
only eight meetings are available. A strong FOMC conclusion requires an
expanded, frozen 2024+ confirmation sample.

## 11. Surface-Representation Consistency

The completed continuation RQ1/RQ2 experiments use raw-vol interpolated
surfaces. RQ3 should therefore use raw-vol as its primary surface input to
preserve the progressive RQ1-RQ3 comparison:

```text
RQ1 raw-vol result
-> RQ2 raw-vol representation comparison
-> RQ3 raw-vol information-regime validation
```

SVI-reconstructed surfaces may be added as a separate surface-construction
robustness check. Absolute MAE levels should not be compared directly between
raw-vol and SVI samples unless their pair sets are explicitly aligned.

## 12. Recommended Thesis Claim

Appropriate:

> The incremental out-of-sample forecasting value of semantic news embeddings
> is concentrated in, or becomes stronger during, pre-specified
> high-information regimes.

Not supported without a separate causal identification strategy:

> Semantic news causes changes in the Treasury-option implied volatility
> surface.

## 13. Methodology Re-review and Revised Controlling Design

### 13.1 Problems identified in the initial design

The methodology re-review identified five issues that materially affect the
interpretation of RQ3.

#### Issue 1: FOMC is too sparse for the primary test

The frozen 2023 outer-test sample contains only eight FOMC meetings and seven
`[0,+5]` forecast pairs. Treating article rows or five-minute observations as
independent would create pseudo-replication; treating meetings as independent
leaves too little power for a conventional primary significance test.

Therefore:

```text
FOMC remains economically important
but becomes a meeting-level supporting case study
```

#### Issue 2: the current zero-text quiet evaluation is out of distribution

The completed RQ1/RQ2 models were trained on news-linked surface pairs. Passing
a zero embedding at no-news timestamps changes the input distribution and does
not establish how a model trained on mixed news/no-news data would behave.

In addition, the legacy event-study labeling logic assigns every
non-announcement row to `quiet`, even though many such rows are ordinary-news
observations. This label must not be used as evidence for a formal no-news
control group.

Therefore:

```text
frozen news-trained model + zero text at no-news timestamps
= descriptive OOD stress test only

formal no-news quiet comparison
= separate mixed-data training experiment
```

#### Issue 3: news versus quiet is strongly confounded

Scheduled news and no-news timestamps differ systematically in time of day,
market liquidity, quote density, current IV level, surface coverage, recent
volatility, and calendar conditions. A raw difference between event and quiet
MAE cannot be interpreted as a difference in semantic text value.

The primary control must therefore be another news observation with comparable
forecast-origin state, not an unmatched no-news timestamp.

#### Issue 4: retraining changes the research object

Adding quiet samples to training can be a useful new experiment, but it changes
the Stage-A backbone, the training distribution, and possibly the learned
forecasting task. It cannot be presented as a pure conditional re-analysis of
the frozen RQ1/RQ2 evidence.

RQ3 should first reuse the frozen RQ1/RQ2 test predictions. Mixed-data
retraining is separately labeled as an appendix extension.

#### Issue 5: the claim must be metric-specific

The completed raw-vol continuation results imply:

```text
RQ1:
  LP is better than continued no-text on surface MAE and short-ATM MAE
  LP is not better on 7d ATM absolute error

RQ2:
  LP is better than BoW and ChatGPT sentiment on the primary surface MAE
  secondary-metric evidence is weaker and not uniform

benchmark context:
  the neural models do not robustly beat persistence on every metric
```

RQ3 therefore cannot pre-specify a universal claim that text improves every
part of the surface. The primary outcome remains `surface_mae`; short-ATM and
7d ATM are separately interpreted secondary outcomes.

### 13.2 Revised research question

The controlling RQ3 question is:

> Conditional on an observably similar market state and news arrival, is the
> incremental out-of-sample value of LP semantic embeddings greater during
> pre-specified scheduled high-information releases than during ordinary-news
> periods?

This creates the intended progression:

```text
RQ1:
  Does LP text add predictive information beyond an identical no-text model?

RQ2:
  Is the semantic LP representation more useful than BoW log-count and
  ChatGPT sentiment summaries?

RQ3:
  Is that incremental semantic value concentrated in economically
  high-information news regimes?
```

RQ3 remains a conditional predictive-ability and robustness analysis. Event
timing improves the economic interpretation of the regime label, but the
analysis does not identify the causal effect of a statement's semantic content.

### 13.3 Frozen evidence and model policy

The primary RQ3 analysis must reuse the completed raw-vol rolling-development
predictions:

```text
RQ1 source:
outputs/experiments/
  rq1_pair_text_raw_vol_continuation_20260723-143511

RQ2 source:
outputs/experiments/
  rq2_pair_representation_raw_vol_continuation_20260724-145528

folds:
2023Q1, 2023Q2, 2023Q3, 2023Q4

seeds:
42, 202, 404
```

Primary models:

```text
LP residual FiLM-WGAN
continued no-text residual model
```

Secondary representation comparisons:

```text
LP versus BoW log-count residual model
LP versus ChatGPT sentiment residual model
```

Rules:

1. Do not retrain or fine-tune a model using RQ3 labels.
2. Do not select a seed, checkpoint, event window, or event category using RQ3
   forecast errors.
3. Use all three pre-specified seeds.
4. Match predictions by `fold`, `seed`, and `surface_pair_id`.
5. Preserve source prediction files by hard link or copy and record SHA256.
6. Any rerun is allowed only to restore missing metadata; its prediction arrays
   must match the frozen source within a declared numerical tolerance.

### 13.4 Scheduled high-information event calendar

The primary event calendar expands beyond FOMC so the analysis is not driven by
only eight meetings. Pre-specified Tier-1 release families are:

```text
FOMC policy statement
CPI
nonfarm payrolls / Employment Situation
PPI
advance GDP
retail sales
ISM manufacturing
ISM services
```

This follows the event-study literature showing that scheduled macroeconomic
announcements generate rapid price discovery and volatility responses. See:
[Andersen, Bollerslev, Diebold, and Vega, Micro Effects of Macro Announcements](https://pubs.aeaweb.org/doi/10.1257/000282803321455151).

Each calendar row must contain:

```text
event_id
event_family
release_name
release_time_local
release_timezone
release_time_utc
scheduled_or_unscheduled
official_source
source_retrieval_date
calendar_version
```

Release times must come from official agencies or archived official calendars.
UTC conversion must be timezone-aware and preserve daylight-saving behavior.
The calendar is frozen before forecast errors are joined.

Primary scheduled-news label:

```text
release_time_utc
<= current_snapshot_time_utc
<= release_time_utc + 5 minutes
```

This aligns the event label with the model's five-minute forecast horizon.
`[-10,+20]` and `[0,+30]` are robustness classification windows only; they do
not convert the model into a 30-minute forecaster.

When multiple releases overlap:

1. retain every associated event ID in the audit table;
2. assign a deterministic primary family using a frozen priority rule;
3. report a robustness result excluding overlapping releases.

### 13.5 Matched ordinary-news control

The primary control group consists of ordinary-news surface pairs, not no-news
quiet pairs.

Eligibility:

```text
sample is a frozen RQ1/RQ2 test news pair
sample is at least 60 minutes from every scheduled event in the calendar
sample has complete model predictions for all compared models and seeds
```

Matching variables must be observed at or before the forecast origin:

```text
rolling fold and calendar quarter
weekday
time of day
current surface level
current ATM level
current surface slope and curvature
recent pre-forecast volatility
surface point count / coverage
raw-vol interpolation quality indicators
article count and news-cluster size
```

Forbidden matching variables:

```text
target surface
realized current-to-target jump
future liquidity or quote count
model forecast error
any statistic calculated from the outer-test target
```

Primary matching procedure:

1. Fit scaling and any propensity/distance model using forecast-origin
   covariates only.
2. Require exact match on rolling fold.
3. Require ordinary news to be within 15 clock minutes of the scheduled sample
   where support permits.
4. Use nearest-neighbor matching without replacement.
5. Freeze matched-set IDs and weights before joining model errors.
6. Report unmatched event rows instead of forcing poor matches.

Required balance diagnostics:

```text
event/control counts
number of independent release days
standardized mean differences before and after matching
distance distribution
common-support exclusions
effective sample size
```

An absolute post-match standardized mean difference above `0.10` is flagged.
The result remains reportable, but the affected covariate and sensitivity
analysis must be disclosed.

### 13.6 Primary estimand and inference

For model loss `L`, test pair `i`, and seed `s`:

```text
d_i,s = L(no_text_i,s) - L(LP_i,s)

d_i,s > 0
=> LP has lower forecast error
```

Average across the three pre-specified seeds:

```text
d_i = mean_s(d_i,s)
```

For matched set `m`, define:

```text
tau_m
= d_scheduled,m
  - weighted_mean(d_ordinary_news,m)

tau
= mean_m(tau_m)
```

Interpretation:

```text
tau > 0
=> LP's incremental value over no-text is larger during scheduled
   high-information news than during matched ordinary news
```

The regression form is:

```text
d_i = matched_set_fixed_effect
    + tau * scheduled_news_i
    + gamma' * forecast_origin_controls_i
    + fold_fixed_effect
    + error_i
```

Primary inference:

```text
metric                = surface_mae
model contrast         = no_text minus LP
event window           = [0,+5] minutes
bootstrap iterations   = 10000
bootstrap seed         = 20260722
resampling cluster     = trading day / independent release day
```

Report:

```text
mean tau
95% cluster-bootstrap confidence interval
raw two-sided p-value
pre-specified one-sided LP-better p-value
matched-set randomization p-value
Giacomini-White conditional predictive-ability result
DM/HAC diagnostic on daily loss differentials
fold-, seed-, event-family-, and release-level directions
```

The three seeds are not treated as a large independent sample. Seed-level
paired t-tests may be shown as diagnostics, but cannot replace event/day-level
inference.

### 13.7 Secondary and heterogeneity analysis

Secondary metrics:

```text
short_atm_mae
atm7_abs_err
energy_score, if frozen scenario outputs are available
coverage and calibration, if computed identically across models
```

Secondary model contrasts:

```text
BoW error - LP error
ChatGPT sentiment error - LP error
```

Multiple-testing policy:

```text
one primary test:
  no multiplicity adjustment required

secondary metric family:
  Holm adjustment

secondary representation family:
  Holm adjustment

alternative-window family:
  Holm adjustment
```

The 7d ATM result must be reported even if its direction differs from surface
MAE. A positive primary surface result does not justify claiming that LP is
better at the shortest ATM point.

### 13.8 FOMC meeting-level case study

FOMC remains a focused economic case study:

```text
primary FOMC window     = [0,+5]
robustness windows      = [-10,+20], [0,+30]
inference unit          = meeting
```

Required outputs:

```text
one row per meeting, fold, model contrast, metric, and seed
seed-averaged meeting difference
exact sign/randomization result
leave-one-meeting-out estimates
statement and press-conference labels kept separate
```

The 2023 FOMC sample must be described as underpowered development evidence.
It cannot be the sole source of a confirmatory RQ3 p-value. A stronger FOMC
claim requires a frozen 2024+ extension.

### 13.9 Ex-ante information-state robustness

Rahimikia, Zohren, and Poon motivate testing whether text is especially useful
in high-volatility states. Formal regimes must be known at forecast origin:

```text
current IV level
recent pre-forecast volatility
current surface displacement
article count
news-cluster size
semantic novelty measured relative to fold-training news
```

Thresholds and semantic-novelty transforms must be fitted on each fold's
training data only. The formal interaction is:

```text
high_state_increment
= text_advantage(high_state)
  - text_advantage(normal_state)
```

An ex-post jump label based on the realized target remains a descriptive
diagnostic and must not be presented as a real-time conditioning variable.

### 13.10 Placebo and falsification requirements

At minimum, run:

```text
scheduled dates shifted by +1 trading day
scheduled dates shifted by -1 trading day
same-clock ordinary-news pseudo-events
same-day mismatched LP text
lead/future text
fold-preserving shuffled text
leave-one-event-family-out analysis
```

Strong placebo effects comparable to the true scheduled-news estimate weaken
or invalidate the information-regime interpretation.

### 13.11 No-news quiet appendix

There are two permissible no-news analyses.

#### Descriptive frozen-model stress test

The existing frozen news-trained models may be evaluated with zero text on
valid no-news pairs only if the output is labeled:

```text
OOD zero-text stress test
not formal evidence for RQ3
```

Its errors must not be pooled with the primary scheduled-versus-ordinary-news
table.

#### Formal mixed-data robustness experiment

A formal news/no-news comparison requires new training:

```text
training data = matched mixture of news and no-news surface pairs
common input  = has_news indicator supplied to every model
LP input      = has_news * transformed LP embedding
no-text input = zero semantic vector with the same has_news indicator
```

All branches must share the same mixed-data Stage-A parent, folds, seeds,
training budget, validation rule, and sample weights. Quiet rows require:

```text
valid five-minute current/target raw-vol surfaces
no news within +/-60 minutes
no scheduled macro release within +/-60 minutes
matched forecast-origin state
```

Because this retraining changes the data-generating distribution and backbone,
the result belongs in an appendix robustness table and must be named separately
from the frozen RQ1/RQ2 continuation evidence.

### 13.12 Revised output archive

Primary frozen-model archive:

```text
outputs/experiments/rq3_scheduled_news_regime_raw_vol_<run_ts>/
  inputs/
    source_prediction_manifest.csv
    event_calendars/
    event_calendar_manifest.csv
    matching_config.yaml
    frozen_model_sha256.csv
    git_state.txt
  matched_samples/
    scheduled_news_samples.csv
    ordinary_news_candidates.csv
    matched_set_manifest.csv
    matching_balance.csv
  comparisons/
    development_rq3_sample_loss_differentials.csv
    development_rq3_primary_cluster_bootstrap.csv
    development_rq3_conditional_predictive_ability.csv
    development_rq3_secondary_metrics.csv
    development_rq3_representation_interactions.csv
    development_rq3_event_family_results.csv
    development_rq3_fomc_meeting_results.csv
    development_rq3_high_information_states.csv
    development_rq3_placebo_results.csv
  final_tables/
    development_thesis_rq3_primary_scheduled_news.csv
    development_thesis_rq3_secondary_metrics.csv
    development_thesis_rq3_fomc_case_study.csv
    development_thesis_rq3_placebo_summary.csv
  docs/
  validation_summary.json
  manifest.csv
```

Any formal mixed-data quiet experiment uses a separate archive:

```text
outputs/experiments/rq3_mixed_news_quiet_raw_vol_<run_ts>/
```

### 13.13 Acceptance checks

The primary archive is valid only if:

```text
all source model/checkpoint/prediction SHA256 values are recorded
all 3 seeds and 4 rolling folds are retained
no RQ3 outcome was used for checkpoint or seed selection
every model comparison is exactly matched by surface_pair_id
the event calendar was frozen before error merging
all event times are timezone-aware and source-audited
ordinary-news controls are >=60 minutes from scheduled releases
matching uses forecast-origin variables only
matching balance and common support are reported
bootstrap resamples independent trading/release days
placebo outputs exist
FOMC output is labeled development case-study evidence
7d ATM and persistence limitations are stated in the final summary
```

If fewer than 20 independent scheduled release days survive matching, the
pooled result is explicitly labeled underpowered development evidence and no
confirmatory thesis claim is made.

### 13.14 Pre-specified interpretation rule

Full support for the revised primary RQ3 hypothesis requires:

```text
mean scheduled_news_increment > 0
95% trading-day/release-day cluster CI lies above 0
one-sided conditional p-value < 0.05
at least 2 of 3 seeds have the same positive direction
the result is not driven by one fold or one event family
date-shift and same-clock placebos do not reproduce the effect
```

Partial support is reported when the mean direction is positive but one or
more inference or stability conditions fail.

No support is reported when the mean is non-positive, the result is dominated
by one event family, or the placebo effects are comparable to the true effect.

### 13.15 Revised thesis claim

Appropriate if the primary rule is satisfied:

> Holding the quantitative surface state and news arrival approximately
> constant, the incremental out-of-sample forecasting value of LP semantic
> embeddings is greater during pre-specified scheduled high-information
> releases than during matched ordinary-news periods.

Appropriate if only some metrics support the hypothesis:

> The conditional value of semantic text is concentrated in selected regions
> or loss functions, rather than uniformly improving the entire volatility
> surface forecast.

Not supported by this design:

> LP text causally changes the implied volatility surface.

Also not supported unless a separate benchmark result establishes it:

> LP FiLM-WGAN universally outperforms persistence across all surface regions.

## 14. Implementation Record (2026-07-25 UTC)

The revised frozen-model RQ3 workflow is implemented in:

```text
scripts/rq3/scheduled_news_regime.py
scripts/rq3/main.py
configs/rq3/scheduled_news_regime_raw_vol.yaml
data/reference/rq3_scheduled_macro_events_2023.csv
tests/test_scripts/test_rq3_scheduled_news.py
```

The event calendar contains 84 source-audited 2023 releases across CPI, PPI,
nonfarm payrolls, retail sales, advance GDP, ISM manufacturing, ISM services,
and FOMC policy announcements. Local release times, time zones, UTC times, and
official source URLs are retained for audit. The runtime validates the local
time-to-UTC conversion before merging any outcomes.

The implementation reads the frozen raw-vol outer-test predictions from:

```text
RQ1:
outputs/experiments/rq1_pair_text_raw_vol_continuation_20260723-143511

RQ2:
outputs/experiments/rq2_pair_representation_raw_vol_continuation_20260724-145528
```

It verifies the duplicate LP and no-text prediction rows across the two
archives within a tolerance of `1e-8`. No model is retrained, no seed is
selected using RQ3 outcomes, and all three seeds and four rolling folds are
retained.

The primary executable command is:

```bash
bash scripts/rq3/start_scheduled_news_regime_background.sh
```

Progress and the completed validation summary are read with:

```bash
bash scripts/rq3/monitor_scheduled_news_regime.sh
```

The foreground equivalent is:

```bash
conda run -n py312 \
  python scripts/rq3/main.py scheduled-news-regime \
  --config configs/rq3/scheduled_news_regime_raw_vol.yaml
```

The background wrapper uses `nohup` and `setsid`, writes PID and log files
under `outputs/rq3/logs/`, and writes a timestamped reproducibility archive
under:

```text
outputs/experiments/rq3_scheduled_news_regime_raw_vol_<run_ts>/
```

### 14.1 Verification outcome

A full 10,000-iteration verification run completed successfully on
2026-07-25. The primary `[0,+5]` window contained:

```text
scheduled forecast pairs = 22
matched sets             = 21
unique releases          = 19
independent release days = 16
event families retained  = 6
```

Because only 16 independent release days survived matching, the software
correctly labels the result `development_underpowered`, below the
pre-specified minimum of 20. Nine forecast-origin balance covariates also
exceeded `|SMD| = 0.10` after matching. These diagnostics are preserved in
`validation_summary.json` and `matched_samples/matching_balance.csv`; they do
not cause silent sample or specification changes.

The verification run did not satisfy the full-support rule. This is an
empirical outcome, not a pipeline failure. It demonstrates that the code
distinguishes a completed reproducible analysis from sufficient evidence for
the RQ3 hypothesis.

### 14.2 Tests completed

```text
tests.test_scripts.test_rq3_scheduled_news = 5 passed
tests.test_scripts.test_rq3_event_study     = 12 passed
tests.test_scripts.test_rq3_news_quiet      = 8 passed
Python compileall                          = passed
shell syntax checks                       = passed
git diff --check                          = passed
background start/monitor/completion        = passed
```
