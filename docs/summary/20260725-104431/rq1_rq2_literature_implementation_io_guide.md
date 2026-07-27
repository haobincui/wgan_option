# RQ1/RQ2 Literature-Grounded Implementation, Results, and I/O Guide

Generated at: `2026-07-25 10:44:31 UTC`

Continuation result update: `2026-07-25 13:49:00 UTC`

## 1. Purpose And Reading Guide

本文档将四类信息放在同一份可复现记录中：

1. RQ1 和 RQ2 在论文中的递进研究逻辑。
2. 与当前研究最相关的 financial text and volatility literature。
3. 当前代码实际实现的 text/no-text 输入处理、FiLM-WGAN conditioning、训练和统计流程。
4. RQ1/RQ2 的 canonical inputs、outputs、历史归档和 claim boundary。

文档区分三种 artifact status：

```text
canonical current
  当前应优先查看和引用的实现或结果。

historical development
  已完成并保留用于比较、诊断和方法演进说明，但不等同于当前口径。

superseded
  早期生成方式或旧结果，只用于审计，不应与 canonical artifacts 混用。
```

最重要的结论边界是：

```text
Canonical RQ1:
  controlled raw-vol pair-level continuation experiment
  same Stage-A parent and same Stage-B budget for text/no-text

Canonical RQ2:
  completed pair-level representation continuation experiment
  LP vs BoW vs ChatGPT sentiment vs continued no-text

Evidence boundary:
  both are 2023 rolling-development evidence
  neither is an untouched 2024+ confirmation
```

本次结果更新的 source provenance：

| RQ | Codex section | Worktree | Branch | Canonical experiment |
|---|---|---|---|---|
| RQ1 | `019f89e2-3edf-71f3-8b07-91278efcc99a` | `/home/haobin_cui/research_files_space_2/wgan_option-rq1-no-text-continuation` | `wgan_rq1_no_text_continuation` | `outputs/experiments/rq1_pair_text_raw_vol_continuation_20260723-143511/` |
| RQ2 | `019f945a-e939-7d82-853d-0a6c43ca5b9f` | `/home/haobin_cui/research_files_space_2/wgan_option-rq2-pair-continuation` | `wgan_rq2_pair_continuation` | `outputs/experiments/rq2_pair_representation_raw_vol_continuation_20260724-145528/` |

数值以 experiment archive 中的 CSV/JSON 为 source of truth。Codex section
用于定位实验过程、设计决策和完成状态，不替代机器可读结果文件。

## 2. Progressive Research Logic

论文围绕 `text embedding + FiLM-WGAN` 的核心创新组织为：

```text
Current implied-volatility surface
              +
financial-news representation
              +
random scenario noise
              |
              v
    text-conditioned FiLM-WGAN
              |
              v
conditional distribution of the future IV surface
```

RQ1 和 RQ2 的关系不是两个独立模型比赛，而是递进关系：

```text
RQ1: Does text add incremental predictive value?
     文本是否在强 surface-history benchmark 之外提供增量信息？

RQ2: Which text representation carries that value?
     如果文本有效，语义 embedding、lexical BoW 和压缩 sentiment
     中哪一种保留了最有用的信息？
```

对应的论文问题可以写为：

> **RQ1:** Does a text-conditioned FiLM-WGAN improve out-of-sample intraday
> implied-volatility-surface forecasting relative to a nested no-text model?

> **RQ2:** Under an otherwise identical forecasting design, do dense semantic
> embeddings provide more useful incremental information than lexical
> log-count features and compressed ChatGPT sentiment scores?

RQ1 的主要差值固定为：

```text
Delta_text = no_text_error - text_error

Delta_text > 0  => text has lower error
Delta_text = 0  => equal predictive accuracy
Delta_text < 0  => no-text has lower error
```

RQ2 的 text-versus-baseline 差值使用同一正方向：

```text
Delta_representation = baseline_error - text_error

Delta_representation > 0 => LP text has lower error
```

## 3. Literature Foundation

### 3.1 Central Reference: Rahimikia, Zohren, And Poon

The closest financial forecasting reference is:

- Eghbal Rahimikia, Stefan Zohren, and Ser-Huang Poon,
  [Realised Volatility Forecasting: Machine Learning via Financial Word Embedding](https://arxiv.org/html/2108.00480v6),
  arXiv version 6, 13 April 2026.

The paper asks whether news text contains predictive information for realised
volatility and whether that information adds value to standard
volatility-history benchmarks. Its main design lessons for this project are:

1. **Volatility persistence is a strong benchmark.**

   News-only forecasting is informative, but it does not automatically dominate
   the strongest HAR-family volatility-history model. This is directly relevant
   because the current IV surface is an even stronger five-minute predictor of
   the target IV surface.

2. **The most defensible role of text is incremental.**

   Their more stable improvement comes from combining the NLP signal with a
   leading volatility benchmark. This supports a nested design in which the
   no-text surface model supplies the base forecast and text learns only a
   correction.

3. **Text value is regime dependent.**

   The paper finds stronger news usefulness on high-volatility days. A small
   unconditional average improvement can therefore conceal a materially larger
   effect in high-information or jump regimes.

4. **Domain and content relevance matter.**

   Finance-specific and stock-related text is more informative than broad,
   general news in their setting. Text coverage alone is not sufficient:
   relevance to the forecasted market state matters.

5. **Out-of-sample and economic evaluation are essential.**

   The paper does not infer text value from in-sample loss alone. It evaluates
   forecast performance out of sample and distinguishes statistical from
   economic value.

The methodological mapping to this project is:

| Rahimikia et al. component | This project | Interpretation |
|---|---|---|
| HAR-family volatility benchmark | no-text surface backbone | strong quantitative state forecast |
| news-based NLP forecast | pair-level LP text adapter | incremental news signal |
| benchmark + NLP ensemble | zero-gated residual FiLM | nested conditional correction |
| high-volatility-day analysis | RQ3 event/high-information analysis | state-dependent text value |
| embedding comparison | RQ2 LP/BoW/sentiment comparison | representation mechanism |

This is an intellectual and design mapping, not a claim that the models are
identical. Important differences are:

| Dimension | Rahimikia et al. | This project |
|---|---|---|
| Target | realised volatility | full implied-volatility surface |
| Horizon | longer-horizon RV forecasting | five-minute intraday transition |
| Numerical benchmark | HAR-family RV history | current IV surface and no-text FiLM-WGAN |
| Text representation | financial word embeddings and NLP model | article-level LP embeddings pooled by surface pair |
| Forecast object | scalar volatility | 16x16 surface and predictive scenarios |
| Combination | forecast ensemble | nested residual conditioning inside generator |

The zero-gated residual FiLM implementation should therefore be described as:

> Inspired by the strong-benchmark-plus-news result in Rahimikia, Zohren, and
> Poon, the model treats text as an incremental correction to a pre-trained
> surface-only forecast rather than requiring text to relearn the entire
> volatility dynamics.

It should not be described as a replication of their model.

### 3.2 Supporting Financial Text Literature

#### High-frequency relevance and novelty

[Groß-Klußmann and Hautsch (2011)](https://doi.org/10.1016/j.jempfin.2010.11.009)
study high-frequency market reactions to machine-readable news. Their results
emphasize that relevance and novelty are important for filtering noisy
intraday news.

Implications here:

```text
do not treat all articles as equally informative;
deduplicate repeated or near-identical articles;
audit timestamp alignment;
retain article count, source, and novelty metadata;
consider relevance-weighted pooling as a future ablation.
```

#### Frequency and unusualness of news

[Calomiris and Mamaysky (2019)](https://www.nber.org/papers/w24430) show that
topic-specific sentiment, news frequency, and unusualness or entropy have
predictive content for returns, volatility, and drawdowns.

Implications here:

```text
the number and diversity of articles may be predictive;
semantic embedding alone may omit news-arrival intensity;
news_count and source diversity should remain in pair-level audit output.
```

#### Task-specific text signal

[Ke, Kelly, and Xiu (2019)](https://www.nber.org/papers/w26186) construct a
supervised text signal tailored to return prediction and distinguish it from
generic dictionary sentiment.

Implications here:

```text
generic semantic quality does not guarantee target-specific predictive value;
all task-specific projection must be fitted on training folds only;
RQ2 should compare rich semantic text with genuinely compressed baselines.
```

### 3.3 Set-Valued News Inputs

When several articles share one current/target surface pair, the input is a set
of articles, not several independent forecasting labels.

[Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)
provides the permutation-invariant foundation for set-valued learning.
[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) extends this
idea with attention over interactions among set elements.

Current primary implementation:

```text
fixed, transparent, permutation-invariant mean-L2 pooling
```

Future ablation:

```text
attention or relevance-weighted pooling
```

The fixed mean remains the primary low-variance baseline because the current
independent training sample is small relative to the text dimension.

### 3.4 Conditioning And Alignment Literature

[FiLM](https://ojs.aaai.org/index.php/AAAI/article/view/11671) defines
feature-wise affine modulation by conditioning information. FiLM supports the
mechanism used here, but its success in visual reasoning does not imply that
financial text must improve IV-surface MAE.

[Gated Multimodal Units](https://arxiv.org/abs/1702.01992) motivate learning
how strongly one modality should influence another. This supports an explicit
text gate rather than forcing text to affect every forecast.

[ControlNet](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)
locks a strong pretrained backbone and connects conditional controls through
zero-initialized layers. The analogy supports:

```text
pretrained no-text backbone
zero-initialized text contribution
initial text forecast equal to base forecast
progressive learning of conditional corrections
```

ControlNet is cross-domain architectural inspiration, not direct financial
evidence.

[Reed et al. (2016)](https://proceedings.mlr.press/v48/reed16.html) use
matching-aware discrimination for text-conditioned generation. This supports
using real surfaces paired with incorrect text as negative examples, so the
critic cannot succeed solely by judging surface realism while ignoring text.

### 3.5 Forecast Evaluation Literature

[Tashman (2000)](https://doi.org/10.1016/S0169-2070(00)00065-0) motivates
rolling-origin evaluation and multiple test periods for more reliable
out-of-sample evidence.

[Diebold and Mariano](https://www.nber.org/papers/t0169) provide a framework
for equal-predictive-accuracy tests using forecast loss differences with
serial dependence.

This project therefore uses:

```text
non-overlapping rolling outer-test quarters;
surface-pair-level matching;
trading-day cluster bootstrap;
seed-level paired diagnostics;
multiple-comparison adjustment for secondary contrasts.
```

## 4. Shared Data Lineage

The high-level data flow is:

```text
raw option implied-volatility points
    -> raw-vol interpolation
    -> current/target 16x16 IV surfaces

raw financial news
    -> LP semantic embedding
    -> BoW log-count representation
    -> ChatGPT sentiment representation

surfaces + news representations
    -> merged_vol_rq2_text.xlsx
    -> RQ1/RQ2 model training
    -> generated sample forecasts
    -> pair-level statistical comparison
```

### Shared raw news source

```text
data/raw/text_embedding/news_with_openai_embeddings_large.xlsx
```

This workbook is approximately 267 MB and contains 14,900 news rows. It is the
source for:

```text
LP article text
LP semantic embeddings
ArticleID
SourceFile
RQ2 BoW features
RQ2 ChatGPT sentiment features
```

### Current raw-vol surface workbook

```text
data/processed/raw-excel/raw_vol_w5_s3_20260629-144428/merged_vol_rq2_text.xlsx
```

The surface model is `raw`, not SVI. The interpolation rule is:

```text
strike or moneyness direction:
  linear interpolation with boundary clamping

maturity direction:
  linear interpolation in total variance
  followed by conversion back to implied volatility
```

This workbook is the canonical surface input for the revised RQ1.

### Historical SVI surface workbook

```text
data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx
```

This workbook is the input for the historical SVI RQ2 multi-seed experiment.
Absolute MAE values from SVI and raw-vol experiments should not be compared as
if they were measured on identical surface construction errors.

## 5. Revised RQ1: Current Canonical Implementation

### 5.1 Research question and hypothesis

RQ1 asks whether matched LP text supplies incremental information after
conditioning on the current IV surface:

```text
H0: E[no_text_error - text_error] = 0
H1: E[no_text_error - text_error] > 0
```

Primary metric:

```text
surface_mae
```

Secondary local metrics:

```text
short_atm_mae
atm7_abs_err
```

The short-ATM mask is:

```text
abs(strike - 1.0) <= 0.06
maturity_days <= 90
```

### 5.2 Canonical code paths

Configuration:

```text
configs/film_wgan/train_rq1_pair_textbase.yaml
```

Reusable implementation:

```text
src/film_wgan/data.py
src/film_wgan/text_transform.py
src/film_wgan/models.py
src/film_wgan/losses.py
src/film_wgan/trainer.py
src/film_wgan/inference.py
```

Experiment orchestration:

```text
scripts/rq1_pair/
```

The workflow directory includes preparation, training, resume, checkpoint
collection, test generation, comparison, monitoring, and packaging scripts.

### 5.3 Surface-pair sample unit

The previous article-row design could create:

```text
text_A -> target_surface_Y
text_B -> target_surface_Y
text_C -> target_surface_Y
```

The revised design creates:

```text
{text_A, text_B, text_C} -> one target_surface_Y
```

For every `surface_pair_id`, the loader requires identical:

```text
current surface
target surface
strike/maturity grid
current timestamp
target timestamp
```

The loader additionally verifies:

```text
news_timestamp_utc == current_snapshot_time_utc
target_snapshot_time_utc - current_snapshot_time_utc == 5 minutes
```

Article lineage is recovered from `sample_id=news_<news_row_id>` and checked
against the raw news workbook. Articles are deduplicated first by `ArticleID`
and then by float32 embedding SHA256.

Pair-level LP pooling is:

```text
u_i = embedding_i / ||embedding_i||_2

pooled_text =
    mean(u_i) / ||mean(u_i)||_2
```

The pair metadata retains:

```text
source_sample_ids
article_ids
source_files
news_count
unique_embedding_count
pooling_mode
has_text
```

### 5.4 Fold-train-only text transformation

The primary representation is:

```text
pair-level pooled LP embedding
    -> PCA fitted only on the fold's train pairs
    -> 128 components
    -> no whitening
```

Each fold stores:

```text
text_transform.npz
text_transform_metadata.json
train_pair_ids_sha256
input_workbook_sha256
components
mean
explained_variance
artifact_sha256
```

The same transformer is loaded by text and no-text models. No-text is set to a
128-dimensional zero vector after the shared transformation, so the text and
no-text networks have the same input shape.

This fixes two historical asymmetries:

```text
historical text:
  1024 variable dimensions with coordinate z-score

historical no-text:
  constant input with much lower effective complexity
```

### 5.5 Zero-gated residual FiLM

The generator is nested around the no-text forecast:

```text
base_delta =
    no_text_surface_backbone(current_surface, noise)

text_delta =
    text_adapter(base_features, transformed_text)

forecast_delta =
    base_delta
    + has_text * tanh(text_gate) * text_delta
```

Properties:

```text
text_gate starts at 0;
the initial text forecast equals the no-text forecast;
the adapter learns only an incremental correction;
has_text is applied after the text encoder;
linear-layer bias cannot create an implicit no-text condition.
```

Text encoder:

```text
128 -> 256 -> 64
dropout = 0.30
```

The critic uses projection conditioning:

```text
score =
    unconditional_surface_score
    + has_text * projection(surface_features, text_features)
```

During text training, incorrectly matched real-surface/text pairs from the
same split are additional negatives. The mismatch loss is disabled for
no-text.

### 5.6 Paired two-stage training

For each fold and seed:

1. Train `pair_pca_no_text_residual`.
2. Initialize matched and shuffled residual generators from that exact
   no-text generator checkpoint.
3. Freeze the surface backbone for epochs 1 to 5.
4. Train the text adapter and gate during the frozen stage.
5. Unfreeze at epoch 6.
6. Use:

   ```text
   backbone learning rate = 2e-6
   text adapter rate      = 2e-5
   ```

7. Reinitialize the critic rather than importing the no-text critic optimizer
   state.
8. Select checkpoints after the warmup period using validation surface MAE.

This pairing reduces seed-to-seed optimization differences between the text
and no-text forecasts.

### 5.7 Rolling development design

The available news workbook ends in 2023, so this is development evidence
rather than untouched future-period confirmation.

| Outer test | Train pairs | Validation pairs | Test pairs |
|---|---:|---:|---:|
| 2023Q1 | 1621 | 425 | 521 |
| 2023Q2 | 2046 | 521 | 333 |
| 2023Q3 | 2567 | 333 | 365 |
| 2023Q4 | 2900 | 365 | 378 |

Seeds:

```text
42, 202, 404
```

Variants:

```text
pair_pca_no_text_residual
pair_pca_no_text_continued
pair_pca_text_residual_pretrained
pair_pca_shuffled_residual_pretrained
pair_pca_text_full_film
pair_pca_text_concat
pair_l2_text_full_film
```

Total:

```text
original rolling matrix:
  4 folds x 3 seeds x 6 variants = 72 runs

controlled no-text continuation:
  4 folds x 3 seeds x 1 variant = 12 runs

canonical continuation archive:
  84 training runs and 84 test summaries
```

The four non-overlapping test quarters contain:

```text
1597 unique surface pairs
229 trading-day clusters
```

The additional `pair_pca_no_text_continued` branch is essential. It starts
from the same fold/seed Stage-A no-text parent as the LP residual branch and
receives the same Stage-B epoch budget, five-epoch backbone freeze and
low-learning-rate continuation schedule. Therefore the primary RQ1 contrast
does not attribute a generic benefit of additional optimization to text.

### 5.8 RQ1 input directory map

| Status | Path | Purpose |
|---|---|---|
| canonical source | `data/raw/text_embedding/news_with_openai_embeddings_large.xlsx` | article text, LP embeddings, IDs, source metadata |
| canonical source | `data/processed/raw-excel/raw_vol_w5_s3_20260629-144428/merged_vol_rq2_text.xlsx` | raw-vol current/target surfaces and text columns |
| canonical config | `configs/film_wgan/train_rq1_pair_textbase.yaml` | frozen base behavior for revised RQ1 |
| canonical worktree | `/home/haobin_cui/research_files_space_2/wgan_option-rq1-no-text-continuation` | branch `wgan_rq1_no_text_continuation` |
| frozen experiment copy | `outputs/experiments/rq1_pair_text_raw_vol_continuation_20260723-143511/inputs/data/merged_vol_rq2_text.xlsx` | exact training workbook snapshot |
| frozen experiment copy | `outputs/experiments/rq1_pair_text_raw_vol_continuation_20260723-143511/inputs/data/news_with_openai_embeddings_large.xlsx` | exact news workbook snapshot |
| frozen experiment config | `outputs/experiments/rq1_pair_text_raw_vol_continuation_20260723-143511/inputs/configs/` | source and resolved base configs |
| fold definitions | `outputs/experiments/rq1_pair_text_raw_vol_continuation_20260723-143511/inputs/folds/` | split manifests and train-only PCA transforms |
| input audit | `outputs/experiments/rq1_pair_text_raw_vol_continuation_20260723-143511/inputs/input_manifest.csv` | input hashes and lineage |
| git audit | `outputs/experiments/rq1_pair_text_raw_vol_continuation_20260723-143511/inputs/git_state.txt` | source-control state |

### 5.9 RQ1 output directory map

Experiment root:

```text
outputs/experiments/rq1_pair_text_raw_vol_continuation_20260723-143511/
```

| Path | Purpose |
|---|---|
| `training_runs/` | 84 fold/seed/variant training runs |
| `generated_results/` | 84 validation-selected outer-test results |
| `checkpoint_selection/selected_checkpoints.csv` | selected checkpoint for every run |
| `checkpoint_selection/resolved_config_audit.csv` | cross-run config audit |
| `checkpoint_selection/paired_stage_validation.csv` | parent, transform and Stage-B continuation audit |
| `registry/launch_registry.csv` | training launch registry |
| `registry/generate_registry.csv` | generate-result registry |
| `comparisons/development_test_sample_metrics.csv` | sample-level model metrics |
| `comparisons/development_pairwise_differences.csv` | matched pairwise loss differences |
| `comparisons/development_cluster_bootstrap_ci.csv` | trading-day cluster bootstrap |
| `comparisons/development_seed_level_tests.csv` | seed-level t/Wilcoxon diagnostics |
| `final_tables/development_rq1_primary_controlled_incremental_text.csv` | controlled primary surface-MAE result |
| `final_tables/development_rq1_point_metric_contrasts.csv` | all point-metric ablations |
| `final_tables/development_rq1_parent_continuation_diagnostics.csv` | Stage-A versus continued branches |
| `final_tables/development_rq1_metrics_by_fold_seed.csv` | fold/seed-level model metrics |
| `final_tables/development_rq1_probabilistic_metrics.csv` | energy, coverage, width, calibration |
| `final_tables/development_rq1_financial_consistency.csv` | calendar and butterfly violations |
| `validation_summary.json` | counts, directions, status, and claim limit |
| `manifest.csv` | complete experiment artifact inventory |

### 5.10 RQ1 main model means

The following values are pair-weighted over the four outer folds and three
seeds:

| Model | Surface MAE | Short-ATM MAE | 7d ATM abs. error |
|---|---:|---:|---:|
| persistence/current surface | **0.011118705** | **0.014306672** | **0.019746464** |
| Stage-A residual no-text | 0.011562755 | 0.014590728 | 0.020073115 |
| continued no-text | 0.011434435 | 0.014520338 | 0.020120361 |
| shuffled LP residual | 0.011414894 | 0.014512090 | 0.020134764 |
| matched LP residual | **0.011400046** | **0.014505638** | 0.020137706 |

Lower is better.

### 5.11 Controlled RQ1 primary result

Difference:

```text
continued_no_text_error - matched_text_error
positive => matched LP text is better
```

| Metric | Continued no-text | LP text | Difference | Relative change | Trading-day cluster 95% CI | Bootstrap p, two-sided | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| surface MAE | 0.011434435 | **0.011400046** | **+0.000034389** | **+0.3007%** | **[0.000018860, 0.000053846]** | `<0.0001` | `<0.0001` |
| short-ATM MAE | 0.014520338 | **0.014505638** | **+0.000014699** | **+0.1012%** | **[0.000006131, 0.000022664]** | 0.0008 | 0.0016 |
| 7d ATM abs. error | **0.020120361** | 0.020137706 | **-0.000017345** | **-0.0862%** | **[-0.000033493, -0.000002366]** | 0.0244 | 0.0244 |

The source CSV stores `0.0` when none of 10,000 bootstrap draws produces a
null-centered statistic at least as extreme in the tested direction. The
document reports this conservatively as `p < 1/10000 = 0.0001`, rather than
claiming a literal zero probability.

Direction and seed diagnostics:

- Surface MAE is better for text in `3/3` seed averages and `12/12`
  fold-seed combinations. The seed-level paired t-test is `p=0.0439`.
- Short-ATM MAE is better in `3/3` seed averages and `10/12` fold-seed
  combinations. Its seed-level t-test is only `p=0.1117`; cluster evidence is
  stronger than the three-seed diagnostic.
- 7d ATM supports text in only `1/3` seed averages and `4/12` fold-seed
  combinations. The controlled result favors no-text at this single grid
  location.

The central RQ1 claim is therefore narrow but positive: LP text supplies a
small incremental improvement for the full surface and short-ATM region after
controlling for the extra optimization stage. It does not improve every local
metric.

### 5.12 Why the continuation control changes the interpretation

Two comparisons separate continuation from text:

```text
continuation effect:
  Stage-A no-text error - continued no-text error

text versus parent:
  Stage-A no-text error - LP residual error
```

| Contrast | Metric | Mean difference | 95% CI | Two-sided p |
|---|---|---:|---:|---:|
| continuation effect | surface MAE | +0.000128320 | [0.000084684, 0.000169671] | `<0.0001` |
| continuation effect | short-ATM MAE | +0.000070390 | [0.000028018, 0.000111592] | 0.0014 |
| text vs Stage-A parent | surface MAE | +0.000162709 | [0.000113783, 0.000212915] | `<0.0001` |
| text vs Stage-A parent | short-ATM MAE | +0.000085090 | [0.000043999, 0.000127724] | `<0.0001` |

Text improves surface MAE by about `1.41%` relative to the Stage-A parent, but
continuing no-text alone improves it by about `1.11%`. Once the shared
continuation budget is controlled, the net text contribution is about
`0.301%`. This is the appropriate incremental-text estimate; the larger
text-versus-parent difference must not be attributed entirely to semantics.

### 5.13 Semantic-alignment placebo

Difference:

```text
shuffled_text_error - matched_text_error
positive => correct text-surface alignment is better
```

| Metric | Mean difference | 95% CI | Two-sided p | Holm p |
|---|---:|---:|---:|---:|
| surface MAE | **0.000014848** | **[0.000003614, 0.000027659]** | 0.0072 | 0.0216 |
| short-ATM MAE | 0.000006451 | [-0.000000877, 0.000013443] | 0.0780 | 0.1560 |
| 7d ATM abs. error | -0.000002942 | [-0.000017124, 0.000010164] | 0.7002 | 0.7002 |

The surface result rejects the interpretation that any random variable passed
through the text branch produces the same gain. Correct semantic alignment
matters for the full-surface forecast in this development sample. However,
the surface effect is positive in `3/3` seed averages but only `8/12`
fold-seed combinations, and the seed-level t-test is `p=0.3958`. This placebo
is supportive rather than independently conclusive.

### 5.14 Residual package versus independently trained full FiLM

Difference:

```text
full_film_error - residual_pretrained_error
positive => residual package is better
```

| Metric | Mean difference | 95% CI | Two-sided p | Holm p |
|---|---:|---:|---:|---:|
| surface MAE | **0.000208460** | **[0.000156727, 0.000255930]** | `<0.0001` | `<0.0001` |
| short-ATM MAE | **0.000212017** | **[0.000163834, 0.000263217]** | `<0.0001` | `<0.0001` |
| 7d ATM abs. error | -0.000001304 | [-0.000117263, 0.000116464] | 0.9846 | 0.9846 |

This is consistent with the Rahimikia et al. design lesson: text is more
effective as a controlled addition to a strong quantitative forecast than as
a condition that must jointly relearn all volatility dynamics from scratch.

It remains an interpretation of this experiment, not proof that residual FiLM
is universally superior.

### 5.15 Persistence benchmark

Persistence uses the current IV surface directly as the five-minute forecast:

| Model | Surface MAE | Short-ATM MAE | 7d ATM abs. error |
|---|---:|---:|---:|
| persistence | **0.011118705** | **0.014306672** | **0.019746464** |
| LP residual | 0.011400046 | 0.014505638 | 0.020137706 |
| continued no-text | 0.011434435 | 0.014520338 | 0.020120361 |
| shuffled LP residual | 0.011414894 | 0.014512090 | 0.020134764 |

LP is descriptively worse than persistence by approximately `2.53%`,
`1.39%`, and `1.98%` for the three metrics. The trading-day cluster intervals
for model-minus-persistence comparisons nevertheless do not provide robust
evidence that any learned branch differs from persistence after multiplicity
adjustment. The archive therefore supports neither “the model beats
persistence” nor a definitive inferiority claim.

This benchmark is substantively important. At a five-minute horizon,
volatility persistence is extremely strong; the RQ1 result concerns the
incremental ranking among learned models, not dominance over the current
surface.

### 5.16 Probabilistic calibration and financial consistency

Pair-weighted probabilistic diagnostics:

| Model | Energy score | Coverage 50% | Coverage 80% | Coverage 90% | Calibration error | Scenario spread |
|---|---:|---:|---:|---:|---:|---:|
| LP residual | **0.0158694** | 3.50% | 6.52% | 8.17% | **0.6727** | 0.000446 |
| continued no-text | 0.0159252 | 3.40% | 6.38% | 7.96% | 0.6742 | 0.000449 |

LP has a slightly lower descriptive energy score and calibration error, but
the empirical coverages are far below their nominal `50/80/90%` levels.
Scenario spreads and interval widths are very small. The generated
distribution is therefore severely under-dispersed, so the probabilistic
outputs cannot currently support a well-calibrated uncertainty claim.

Financial-consistency diagnostics:

| Model | Calendar violation rate | Butterfly violation rate |
|---|---:|---:|
| LP residual | 14.0218% | 9.4942% |
| continued no-text | 14.0216% | 9.4610% |

The differences are negligible and slightly unfavorable to text for both
rates. Text improves point forecast MAE in selected regions but does not
improve the measured no-arbitrage consistency.

### 5.17 RQ1 claim boundary

Allowed statement:

> Across four non-overlapping 2023 rolling outer-test quarters and three
> pre-specified seeds, the pair-level PCA LP residual FiLM-WGAN reduced
> full-surface and short-ATM MAE relative to a no-text branch receiving the
> same parent initialization and continuation budget. The matched-versus-
> shuffled surface result provides additional evidence that correct semantic
> alignment matters.

Not allowed:

```text
the model has been confirmed on untouched 2024+ data;
text improves every local IV-surface metric;
the learned model significantly outperforms persistence;
the generated scenarios are probabilistically calibrated;
text improves calendar or butterfly consistency;
the result proves a causal effect of news;
```

Final confirmation still requires later news and option data not used in
developing this architecture.

The machine-readable archive records the same boundary:

```text
validation_summary.json:
  status = ok
  development_only = true
  claim_limit = No untouched 2024+ confirmation data are available.
```

## 6. Historical RQ1 Predecessors

### 6.1 Uncontrolled rolling predecessor

Previous rolling root:

```text
outputs/experiments/rq1_pair_text_raw_vol_rolling_20260723-140323/
```

This 72-run archive established pair-level pooling, train-only PCA, rolling
folds, shuffled-text and architecture ablations. Its original comparison used
the Stage-A no-text parent directly against a Stage-B LP model. Because the LP
branch received an additional optimization stage while no-text did not, its
larger text advantage (`+0.00016664` surface MAE) combines semantic value with
continuation value.

The archive remains useful for implementation and ablation history, but
`rq1_pair_text_raw_vol_continuation_20260723-143511` is the canonical
incremental-text result.

### 6.2 Strict grouped-split diagnostic

Historical root:

```text
outputs/experiments/rq1_incremental_text_20260722-145823/
```

This strict grouped-split experiment remains useful because it diagnosed:

```text
article-row pseudoreplication;
multiple distinct texts mapped to one surface transition;
coordinate z-score distortion of nearly unit-normalized LP geometry;
1024-dimensional conditional input with few independent surface pairs;
same nominal parameter count but different effective text/no-text complexity;
independently optimized text and no-text models.
```

It should be cited as the development evidence that motivated the pair-level
residual design. It should not be pooled numerically with the rolling RQ1
result.

## 7. Historical RQ2: Representation Comparison

### 7.1 Research question

Historical RQ2 compares:

```text
text          = LP dense semantic embedding
no_text       = no text input
bow           = unigram/bigram log-count representation
llm_sentiment = ChatGPT three-score sentiment representation
```

The intended controlled interpretation is:

```text
same surfaces
same downstream FiLM-WGAN
same seed set
different text representation
```

However, these completed experiments predate the revised pair-level residual
RQ1 and therefore have important limitations described below.

### 7.2 Historical feature lineage and frozen sentiment source

Raw text:

```text
data/raw/text_embedding/news_with_openai_embeddings_large.xlsx
```

Historical full-corpus feature root and frozen ChatGPT artifact source:

```text
data/processed/text_features/rq2/20260625-075653/
```

Files:

```text
bow_features.xlsx
bow_manifest.json
bow_vocabulary.json
llm_sentiment_features.xlsx
llm_sentiment_manifest.json
openai_sentiment_cache.jsonl
```

#### LP semantic embedding

```text
source column = LP embedding in the news/enriched workbook
dimension     = 1024
role          = dense semantic and contextual representation
```

#### Historical full-corpus BoW

Actual `bow_manifest.json`:

```text
text column        = LP
rows               = 14900
ngram range        = [1, 2]
vocabulary size    = 1024
feature dimension  = 1024
backend            = python_counter
weighting          = log1p_count
reference method   = Manela_Moreira_2017_style_ngram_frequency
```

For article \(i\) and vocabulary term \(j\):

```text
bow_ij = log(1 + ngram_count_ij)
```

The canonical continuation RQ2 does **not** use this full-corpus vocabulary as
its model input. It fits a new vocabulary and PCA inside each rolling fold
using train articles only. The full-corpus artifact is retained for lineage
and audit.

#### Frozen ChatGPT sentiment source

Actual completed `llm_sentiment_manifest.json`:

```text
rows             = 14900
artifact model   = gpt-5.4-mini
prompt version   = sun2026_zero_shot_chatgpt_v1
reasoning effort = low
input limit      = 6000 characters
base dimension   = 3
target dimension = 1024
API errors       = 0
```

The three scored dimensions are:

```text
macroeconomic_uncertainty
institutional_action
risk_off_intensity
```

Each score lies in `[0, 1]`. The remaining 1021 positions are zero. This is a
compressed theory-driven baseline, not a representation with the same
effective information capacity as LP.

The current source-code default model can change over time. Reproduction must
follow the archived manifest and cache, not the current default constant in
`src/llm_sentiment/features.py`.

### 7.3 Superseded RQ2 features

Do not mix the canonical features above with:

```text
data/processed/text_features/rq2/20260623-rq2/
```

That directory used:

```text
BoW:
  TF-IDF vocabulary size 5000
  TruncatedSVD to 1024

sentiment:
  Loughran-McDonald dictionary
  17 base features
```

Those are materially different representations and are retained only for
audit.

### 7.4 Historical RQ2 input workbooks

SVI:

```text
data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx
```

Raw-vol:

```text
data/processed/raw-excel/raw_vol_w5_s3_20260629-144428/merged_vol_rq2_text.xlsx
```

Historical base config:

```text
configs/film_wgan/train_rq2_multiseed_textbase.yaml
```

Historical design:

```text
sample unit            = article row
split                  = chronological 80/20
seeds                  = 42, 101, 202, 303, 404
checkpoint selection   = after warmup, validation metric
conditioning           = independently trained full FiLM
historical no-text     = text_embedding_mode=none
```

### 7.5 Historical RQ2 output roots

SVI multi-seed:

```text
outputs/archive/outputs_pre_20260722-135827/experiments/
  rq2_multiseed_textbase_20260627/
```

Raw-vol multi-seed:

```text
outputs/archive/outputs_pre_20260722-135827/experiments/
  raw_vol_rq1_rq2_20260629-144428/
```

Each archive contains:

```text
inputs/
training_runs/
logs/
registry/
checkpoint_selection/
comparisons/
final_tables/
manifest.csv
validation_summary.json
```

Older single-run comparison archives:

```text
outputs/archive/outputs_pre_20260722-135827/comparison/
```

These include early text/no-text and text/BoW/sentiment comparisons. They are
not the primary multi-seed evidence.

## 8. Historical RQ2 Results

### 8.1 SVI all-seed eval means

Source:

```text
outputs/archive/outputs_pre_20260722-135827/experiments/
  rq2_multiseed_textbase_20260627/final_tables/
  thesis_eval_main_metrics.csv
```

Values are mean across five seeds, with cross-seed standard deviation in
parentheses:

| Representation | Surface MAE | Short-ATM MAE | 7d ATM abs. error |
|---|---:|---:|---:|
| LP text | 0.02316374 (0.00023370) | 0.01772635 (0.00034939) | 0.02309296 (0.00126156) |
| no-text | 0.02316790 (0.00022987) | **0.01733014 (0.00024999)** | **0.02236702 (0.00035340)** |
| BoW | **0.02309490 (0.00021175)** | 0.01769430 (0.00033327) | 0.02287111 (0.00094936) |
| ChatGPT sentiment | 0.02317589 (0.00018237) | 0.01757840 (0.00014708) | 0.02308118 (0.00071246) |

The SVI experiment does not support a simple claim that LP text uniformly
dominates the alternatives.

### 8.2 SVI seed-level LP-text contrasts

Difference:

```text
baseline_error - LP_text_error
positive => LP text is better
```

| Baseline | Metric | Mean seed difference | Two-sided t p | One-sided p, text better | Text-win seeds |
|---|---|---:|---:|---:|---:|
| no-text | surface MAE | 0.00000415 | 0.9529 | 0.4764 | 3/5 |
| no-text | short-ATM MAE | -0.00039620 | 0.0258 | 0.9871 | 1/5 |
| no-text | 7d ATM abs. error | -0.00072594 | 0.2528 | 0.8736 | 1/5 |
| BoW | surface MAE | -0.00006884 | 0.1329 | 0.9335 | 0/5 |
| BoW | short-ATM MAE | -0.00003205 | 0.6579 | 0.6711 | 3/5 |
| BoW | 7d ATM abs. error | -0.00022185 | 0.4880 | 0.7560 | 2/5 |
| sentiment | surface MAE | 0.00001215 | 0.9103 | 0.4551 | 3/5 |
| sentiment | short-ATM MAE | -0.00014795 | 0.3412 | 0.8294 | 2/5 |
| sentiment | 7d ATM abs. error | -0.00001179 | 0.9815 | 0.5093 | 2/5 |

This table should be used instead of selecting the best seed after inspecting
eval MAE.

### 8.3 Raw-vol all-seed eval means

Source:

```text
outputs/archive/outputs_pre_20260722-135827/experiments/
  raw_vol_rq1_rq2_20260629-144428/final_tables/
  thesis_eval_main_metrics.csv
```

| Representation | Surface MAE | Short-ATM MAE | 7d ATM abs. error |
|---|---:|---:|---:|
| LP text | 0.00935920 (0.00001208) | 0.01145808 (0.00005781) | 0.01661545 (0.00046994) |
| no-text | **0.00933257 (0.00001551)** | 0.01142604 (0.00009191) | **0.01626029 (0.00065897)** |
| BoW | 0.00936084 (0.00000708) | 0.01146155 (0.00003123) | 0.01670775 (0.00049864) |
| ChatGPT sentiment | 0.00933359 (0.00001456) | **0.01141883 (0.00005282)** | 0.01657126 (0.00044181) |

Again, this historical full-FiLM design does not show uniform LP dominance.

### 8.4 Raw-vol seed-level LP-text contrasts

| Baseline | Metric | Mean seed difference | Two-sided t p | One-sided p, text better | Text-win seeds |
|---|---|---:|---:|---:|---:|
| no-text | surface MAE | -0.00002664 | 0.0087 | 0.9957 | 0/5 |
| no-text | short-ATM MAE | -0.00003204 | 0.5759 | 0.7120 | 1/5 |
| no-text | 7d ATM abs. error | -0.00035516 | 0.0698 | 0.9651 | 1/5 |
| BoW | surface MAE | 0.00000163 | 0.7319 | 0.3659 | 2/5 |
| BoW | short-ATM MAE | 0.00000347 | 0.8924 | 0.4462 | 4/5 |
| BoW | 7d ATM abs. error | 0.00009230 | 0.5987 | 0.2994 | 2/5 |
| sentiment | surface MAE | -0.00002561 | 0.0088 | 0.9956 | 0/5 |
| sentiment | short-ATM MAE | -0.00003925 | 0.3119 | 0.8441 | 1/5 |
| sentiment | 7d ATM abs. error | -0.00004419 | 0.8547 | 0.5726 | 2/5 |

### 8.5 Why historical RQ2 does not contradict the continuation results

The historical and revised designs differ in more than text representation:

| Component | Historical RQ2 | Revised RQ1 |
|---|---|---|
| Sample unit | article row | unique surface pair |
| Duplicate label handling | repeated target surfaces | one target per pair |
| LP preprocessing | coordinate z-score | mean-L2 then train-only PCA-128 |
| no-text | historical `none` | same-dimensional transformed zero input |
| Conditioning | independently trained full FiLM | no-text-initialized residual FiLM |
| Evaluation | one chronological 80/20 split | four rolling train/val/test folds |
| Main uncertainty | five-seed test | trading-day cluster bootstrap plus seed diagnostics |
| BoW fitting | full news corpus | fold-train-only in canonical continuation RQ2 |

The historical RQ2 results answer:

> How did the three text representations behave in the earlier article-row
> full-FiLM pipeline?

They do not answer:

> Which representation is best inside the revised pair-level residual
> FiLM-WGAN?

That question is answered by the canonical continuation experiment in
Section 9.

### 8.6 Best-seed files

Both historical RQ2 archives contain files such as:

```text
final_tables/best_seed_by_model_metric_eval.csv
final_tables/best_seed_pairwise_text_vs_baselines_eval_p_values.csv
```

They are retained for audit and exploratory analysis only. Selecting a
different seed for every model and metric after observing eval MAE creates
post-selection optimism. These files must not be the main thesis evidence.

The main historical RQ2 evidence is:

```text
final_tables/thesis_eval_main_metrics.csv
final_tables/thesis_text_vs_baselines.csv
comparisons/seed_level_tests.csv
```

## 9. Canonical RQ2: Pair-Level Representation Continuation

### 9.1 Experiment identity and controlled design

Worktree and branch:

```text
worktree:
  /home/haobin_cui/research_files_space_2/wgan_option-rq2-pair-continuation

branch:
  wgan_rq2_pair_continuation
```

Experiment root:

```text
outputs/experiments/
  rq2_pair_representation_raw_vol_continuation_20260724-145528/
```

Source RQ1 experiment:

```text
/home/haobin_cui/research_files_space_2/wgan_option-rq1-no-text-continuation/
  outputs/experiments/
  rq1_pair_text_raw_vol_continuation_20260723-143511/
```

The four comparison branches are:

```text
continued_no_text = imported RQ1 Stage-B continuation control
lp                = imported RQ1 matched LP residual branch
bow               = newly trained Stage-B BoW residual branch
llm_sentiment     = newly trained Stage-B sentiment residual branch
```

RQ2 imports 12 Stage-A parents, 12 continued no-text results and 12 LP results
from RQ1. It trains only:

```text
2 new representations x 4 folds x 3 seeds = 24 new runs
```

All four branches use the same raw-vol surface pairs, rolling folds, seeds,
128-dimensional text interface, residual FiLM architecture, parent generator,
Stage-B budget and validation checkpoint rule. The new BoW and sentiment
branches use a fresh critic under the same seed. Initialization hashes and
parent SHA256 are audited before comparison.

Completed archive status:

```text
24/24 new checkpoints
24 imported LP/no-text model-fold-seed results
48/48 four-model test results
19164 model-sample rows
1597 unique outer-test pairs per seed
229 trading-day clusters
pipeline_status = completed
comparison_status = ok
missing_outputs = 0
completed_at = 2026-07-25 11:27 UTC
```

### 9.2 Representation construction

All representations first deduplicate articles by `ArticleID` and LP embedding
SHA256, then aggregate at `surface_pair_id`.

```text
LP:
  mean(L2(unique article LP embeddings))
  -> final L2 normalization
  -> fold-train-only PCA-128

BoW:
  train corpus = unique articles from fold train pairs only
  unigram/bigram vocabulary = top 1024 deterministic frequency-ranked terms
  pair vector = L2(log1p(sum unique-article term counts))
  -> fold-train-only PCA-128, no whitening

ChatGPT sentiment:
  model = gpt-5.4-mini
  prompt = sun2026_zero_shot_chatgpt_v1
  pair mean of:
    macroeconomic_uncertainty
    institutional_action
    risk_off_intensity
  -> fold-train-only z-score
  -> zero-pad from 3 to 128 dimensions

continued no-text:
  same 128-dimensional model interface
  -> all-zero transformed vector
```

Validation and test articles cannot influence the BoW vocabulary, PCA or
sentiment scaling. This removes the full-corpus feature leakage risk present
in the historical article-row RQ2.

### 9.3 RQ2 input and output map

Canonical input/config paths:

| Path | Purpose |
|---|---|
| `inputs/data/merged_vol_rq2_text.xlsx` | frozen raw-vol pair workbook |
| `inputs/data/news_with_openai_embeddings_large.xlsx` | frozen article/LP source |
| `inputs/text_features/llm_sentiment_features.xlsx` | frozen ChatGPT score artifact |
| `inputs/text_features/llm_sentiment_manifest.json` | model, prompt and dimension lineage |
| `inputs/text_features/openai_sentiment_cache.jsonl` | per-article response cache |
| `inputs/imported_rq1/` | imported RQ1 checkpoints, registries and validation |
| `inputs/rq1_import_manifest.csv` | source/target path, link mode and SHA256 |
| `inputs/configs/train_rq2_pair_textbase.yaml` | frozen RQ2 base config |

Core result paths:

| Path | Purpose |
|---|---|
| `checkpoint_selection/four_branch_initialization_audit.csv` | parent and initialization equality audit |
| `comparisons/development_rq2_test_sample_metrics.csv` | all four model sample metrics |
| `comparisons/development_rq2_pairwise_differences.csv` | matched pair differences |
| `comparisons/development_rq2_cluster_bootstrap_ci.csv` | trading-day cluster bootstrap |
| `comparisons/development_rq2_dm_hac_tests.csv` | daily DM-style HAC tests |
| `comparisons/development_rq2_seed_level_tests.csv` | seed t-test and exact Wilcoxon diagnostics |
| `final_tables/development_rq2_primary_lp_vs_baselines.csv` | primary LP-vs-representation tests |
| `final_tables/development_rq2_incremental_value_vs_no_text.csv` | representation incremental value |
| `final_tables/development_rq2_vs_persistence.csv` | persistence benchmark |
| `final_tables/development_rq2_model_overall_metrics.csv` | fold/seed model means |
| `final_tables/development_rq2_result_summary.json` | machine-readable H2 decision |
| `validation_summary.json` | output counts and evidence boundary |
| `manifest.csv` | complete artifact inventory |

### 9.4 Primary Surface MAE results

Primary difference:

```text
representation_error - LP_error
positive => LP has lower surface MAE
```

Pair-weighted model means and dependence-aware tests:

| Model | Surface MAE | Baseline minus LP | Cluster 95% CI | Bootstrap p | Holm p | DM/HAC p |
|---|---:|---:|---:|---:|---:|---:|
| LP | **0.011400046** | - | - | - | - | - |
| BoW | 0.011414586 | **+0.000014540** | **[0.000003223, 0.000027025]** | 0.0092 | **0.0156** | 0.0188 |
| LLM sentiment | 0.011400817 | **+0.000000772** | **[0.000000177, 0.000001452]** | 0.0078 | **0.0156** | 0.0267 |
| continued no-text | 0.011434435 | **+0.000034389** | **[0.000018720, 0.000053615]** | 0.0002 | **0.0008** | `<0.000001` |

Relative MAE reductions:

```text
LP relative to BoW               = 0.1274%
LP relative to LLM sentiment     = 0.0068%
LP relative to continued no-text = 0.3007%
```

The two pre-specified LP-vs-representation tests form one Holm family. Both
adjusted p-values are below 0.05 and both cluster-CI lower bounds are positive.
The archived pre-registered decision is therefore:

```text
H2 support = full_support
```

This decision concerns `surface_mae`. It does not imply that LP dominates
every local metric. The sentiment difference is statistically detectable but
economically extremely small: its absolute MAE difference is below
`8e-7`, only about `0.0068%`.

### 9.5 Secondary metrics

Differences remain `baseline error - LP error`:

| Baseline | Short-ATM difference | Cluster p | Holm p | 7d ATM difference | Cluster p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| BoW | +0.000006703 | 0.0746 | 0.2238 | -0.000000081 | 0.9751 | 1.0000 |
| LLM sentiment | +0.000000270 | 0.1004 | 0.2238 | +0.000001418 | 0.0604 | 0.1812 |
| continued no-text | **+0.000014699** | 0.0016 | **0.0080** | -0.000017345 | 0.0186 | 0.0744 |

Interpretation:

- LP significantly improves short-ATM MAE relative to continued no-text after
  Holm correction.
- LP has favorable short-ATM point estimates relative to BoW and sentiment,
  but neither survives secondary-family correction.
- LP and BoW have effectively identical 7d ATM error.
- LP's 7d point estimate is slightly lower than sentiment but is not
  significant after correction.
- Continued no-text has lower 7d ATM error than LP. The raw cluster test
  detects this adverse direction, while the broader RQ2 Holm family raises the
  adjusted p-value to `0.0744`.

Thus the representation ranking is strongest for the full surface, partially
supported for the short-ATM region, and unresolved or unfavorable at the
single 7d ATM point.

### 9.6 Seed, fold and time-dependence diagnostics

Surface-MAE direction:

| Contrast | Positive seed averages | Positive fold-seed combinations | Seed t-test, two-sided | Exact Wilcoxon |
|---|---:|---:|---:|---:|
| LP vs BoW | 3/3 | 6/12 | 0.3943 | 0.25 |
| LP vs sentiment | 3/3 | 9/12 | 0.0862 | 0.25 |
| LP vs continued no-text | 3/3 | 12/12 | 0.0439 | 0.25 |

The trading-day cluster bootstrap and DM/HAC tests use much richer temporal
information than three seed averages. Conversely, cluster-level significance
does not replace cross-seed replication. With only three seeds, the exact
Wilcoxon test cannot attain a conventional two-sided 5% threshold. The
strongest stability result is LP versus no-text; LP versus BoW is significant
in the pooled time-aware tests but varies substantially by quarter and seed.

### 9.7 Incremental value of each representation and persistence

Relative to continued no-text:

| Representation | Surface difference | Holm p | Short-ATM difference | Holm p | 7d ATM difference | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| LP | +0.000034389 | 0.0008 | +0.000014699 | 0.0080 | -0.000017345 | 0.0744 |
| BoW | +0.000019849 | 0.0008 | +0.000007997 | 0.0048 | -0.000017263 | 0.0012 |
| sentiment | +0.000033617 | 0.0008 | +0.000014430 | 0.0080 | -0.000018763 | 0.0680 |

All text representations improve full-surface and short-ATM point metrics
relative to continued no-text, while all are worse on the 7d ATM point. LP
has the lowest overall surface MAE, but the result also shows that the useful
incremental signal is not exclusive to dense embeddings.

Persistence means:

```text
surface_mae   = 0.011118705
short_atm_mae = 0.014306672
atm7_abs_err  = 0.019746464
```

All four learned branches have larger average errors. None robustly beats
persistence after within-metric Holm adjustment. RQ2 therefore ranks text
representations inside the continuation FiLM-WGAN framework; it does not
establish that this framework beats the five-minute persistence benchmark.

### 9.8 Reconciliation of RQ1 and RQ2 statistics

RQ2 imports the exact RQ1 LP and continued no-text predictions. Their model
means and paired mean differences therefore match:

```text
surface difference   = +0.000034389
short-ATM difference = +0.000014699
7d ATM difference    = -0.000017345
```

The two archives independently execute bootstrap resampling and use different
Holm families. Small differences in reported confidence bounds or raw Monte
Carlo p-values are therefore expected. In particular, RQ1 adjusts the three
metrics within the controlled text contrast, whereas RQ2 adjusts secondary
metrics across a larger set of representation contrasts. Statistical values
should always be cited from the archive belonging to the research question
being discussed; they must not be mixed into a synthetic p-value.

### 9.9 RQ2 claim boundary

Allowed statement:

> Under the same raw-vol surface pairs, Stage-A parent checkpoints, residual
> FiLM architecture, rolling folds, seeds and Stage-B budget, LP semantic
> embeddings achieved significantly lower full-surface MAE than fold-train-
> only BoW, compressed ChatGPT sentiment and continued no-text. The advantage
> was small and was most robust for the aggregate surface metric.

Not allowed:

```text
LP is better for every strike, maturity or local metric;
LP has a substantively large advantage over ChatGPT sentiment;
all learned models outperform persistence;
three seeds provide high-powered distribution-free inference;
the result is an untouched 2024+ confirmation;
the predictive contrast establishes a causal effect of news.
```

## 10. Unified Reproducibility Index

| RQ | Artifact status | Representation/surface | Path | Purpose |
|---|---|---|---|---|
| shared | canonical | raw news and LP | `data/raw/text_embedding/news_with_openai_embeddings_large.xlsx` | source text, embeddings, IDs |
| RQ1 | canonical | raw-vol | `data/processed/raw-excel/raw_vol_w5_s3_20260629-144428/merged_vol_rq2_text.xlsx` | revised RQ1 source workbook |
| RQ1 | canonical | pair-level residual | `configs/film_wgan/train_rq1_pair_textbase.yaml` | base training/generation config |
| RQ1 | canonical current | controlled LP/no-text continuation | `outputs/experiments/rq1_pair_text_raw_vol_continuation_20260723-143511/` in `wgan_option-rq1-no-text-continuation` | 84-run controlled rolling archive |
| RQ1 | historical development | uncontrolled rolling matrix | `outputs/experiments/rq1_pair_text_raw_vol_rolling_20260723-140323/` | 72-run predecessor and architecture diagnostics |
| RQ1 | historical development | strict grouped split | `outputs/experiments/rq1_incremental_text_20260722-145823/` | diagnostic predecessor |
| RQ2 | canonical current | LP/BoW/sentiment/no-text continuation | `outputs/experiments/rq2_pair_representation_raw_vol_continuation_20260724-145528/` in `wgan_option-rq2-pair-continuation` | completed four-model rolling comparison |
| RQ2 | canonical config | pair-level representation residual | `configs/film_wgan/train_rq2_pair_textbase.yaml` in the RQ2 worktree | frozen RQ2 training/generation behavior |
| RQ2 | frozen source artifact | ChatGPT sentiment | `data/processed/text_features/rq2/20260625-075653/` | model/prompt manifest and response cache |
| RQ2 | audit only | full-corpus BoW | `data/processed/text_features/rq2/20260625-075653/` | not used as continuation model input |
| RQ2 | superseded features | TF-IDF/SVD and dictionary | `data/processed/text_features/rq2/20260623-rq2/` | audit only |
| RQ2 | historical development | SVI, five seeds | `outputs/archive/outputs_pre_20260722-135827/experiments/rq2_multiseed_textbase_20260627/` | historical SVI representation comparison |
| RQ2 | historical development | raw-vol, five seeds | `outputs/archive/outputs_pre_20260722-135827/experiments/raw_vol_rq1_rq2_20260629-144428/` | historical raw-vol representation comparison |
| RQ2 | superseded/intermediate | older comparisons | `outputs/archive/outputs_pre_20260722-135827/comparison/` | single-run and intermediate audit |

## 11. Fastest Inspection Order

### Canonical RQ1 continuation

Use worktree:

```text
/home/haobin_cui/research_files_space_2/wgan_option-rq1-no-text-continuation
```

Read in this order:

```text
outputs/experiments/rq1_pair_text_raw_vol_continuation_20260723-143511/

1. final_tables/development_rq1_primary_controlled_incremental_text.csv
2. final_tables/development_rq1_point_metric_contrasts.csv
3. final_tables/development_rq1_parent_continuation_diagnostics.csv
4. comparisons/development_seed_level_tests.csv
5. final_tables/development_rq1_probabilistic_metrics.csv
6. final_tables/development_rq1_financial_consistency.csv
7. checkpoint_selection/paired_stage_validation.csv
8. validation_summary.json
9. manifest.csv
```

### Canonical RQ2 continuation

Use worktree:

```text
/home/haobin_cui/research_files_space_2/wgan_option-rq2-pair-continuation
```

Read in this order:

```text
outputs/experiments/
  rq2_pair_representation_raw_vol_continuation_20260724-145528/

1. final_tables/development_rq2_result_summary.json
2. final_tables/development_rq2_primary_lp_vs_baselines.csv
3. final_tables/development_rq2_incremental_value_vs_no_text.csv
4. final_tables/development_rq2_vs_persistence.csv
5. comparisons/development_rq2_cluster_bootstrap_ci.csv
6. comparisons/development_rq2_dm_hac_tests.csv
7. comparisons/development_rq2_seed_level_tests.csv
8. checkpoint_selection/four_branch_initialization_audit.csv
9. inputs/rq1_import_manifest.csv
10. validation_summary.json
11. manifest.csv
```

### Historical RQ2 archives

SVI:

```text
outputs/archive/outputs_pre_20260722-135827/experiments/
  rq2_multiseed_textbase_20260627/final_tables/
```

Read:

```text
thesis_eval_main_metrics.csv
thesis_text_vs_baselines.csv
```

Raw-vol:

```text
outputs/archive/outputs_pre_20260722-135827/experiments/
  raw_vol_rq1_rq2_20260629-144428/final_tables/
```

Read:

```text
thesis_eval_main_metrics.csv
thesis_text_vs_baselines.csv
```

### Frozen feature lineage

```text
data/processed/text_features/rq2/20260625-075653/
  llm_sentiment_manifest.json
  openai_sentiment_cache.jsonl
```

Continuation-specific BoW vocabulary, PCA and pair audits must be read from
the canonical RQ2 experiment's per-fold input artifacts, not from the
full-corpus `bow_vocabulary.json`.

## 12. Thesis Interpretation

### 12.1 What canonical RQ1 supports

The controlled continuation evidence supports:

> Holding the Stage-A parent, Stage-B continuation budget, rolling folds and
> seeds fixed, pair-level LP residual conditioning produces a small reduction
> in full-surface and short-ATM MAE relative to continued no-text.

The matched-versus-shuffled surface comparison strengthens the interpretation
that correct semantic alignment contributes information. The continuation
control also shows why the effect must be described as small: most of the
original text-versus-parent gain can be reproduced by continuing no-text
training.

RQ1 does not support LP at the single 7d ATM point, calibrated scenario
intervals, improved no-arbitrage consistency or superiority to persistence.

### 12.2 What canonical RQ2 supports

The pair-level continuation RQ2 supports the pre-specified surface-MAE
hypothesis:

```text
LP surface MAE < BoW surface MAE
LP surface MAE < ChatGPT sentiment surface MAE
H2 support = full_support
```

Both cluster-bootstrap comparisons survive their two-test Holm family and are
also supported by daily HAC tests. LP also remains better than continued
no-text for surface and short-ATM MAE.

The magnitude qualification is essential:

```text
LP vs BoW improvement           = 0.1274%
LP vs sentiment improvement     = 0.0068%
LP vs continued no-text         = 0.3007%
```

Therefore the evidence supports superior aggregate surface accuracy for LP,
not a large universal representation advantage. Short-ATM representation
differences are mostly unresolved after adjustment, and 7d ATM does not
support uniform LP dominance.

### 12.3 Statistical and economic interpretation

The strongest inference combines:

```text
matched pair-level errors
trading-day cluster bootstrap
daily DM/HAC tests
pre-specified Holm families
three pre-specified seeds
four non-overlapping test quarters
```

Cluster and HAC significance address repeated observations and time
dependence. The three-seed diagnostics address optimization variability but
have low power, especially for exact Wilcoxon inference. Statistical
significance must be reported together with effect size and fold/seed
direction.

Persistence remains the strongest average five-minute forecast. The learned
models' contribution is currently a representation and incremental-text
result inside the FiLM-WGAN framework, not a demonstrated replacement for the
current-surface forecast.

### 12.4 Coherent chapter narrative

The defensible narrative is:

```text
1. Historical experiments showed that naive article-row full-FiLM conditioning
   did not reliably extract incremental LP information.

2. Input and architecture audits identified duplicate surface labels,
   distorted embedding geometry, excessive conditional capacity, and
   independent optimization as plausible mechanisms.

3. Literature on news-based volatility forecasting suggested treating text as
   an incremental signal on top of a strong volatility benchmark.

4. The pair-level residual architecture produced positive preliminary
   evidence, but a continuation audit showed that additional optimization was
   an important confound.

5. Canonical RQ1 controlled that confound with a same-budget continued
   no-text branch and retained a smaller but significant surface/short-ATM
   LP advantage.

6. Canonical RQ2 then held the model and continuation design fixed and showed
   that LP achieved the lowest aggregate surface MAE relative to fold-train-
   only BoW and compressed ChatGPT sentiment.

7. The local 7d result, persistence benchmark, weak interval calibration and
   2023-only sample prevent a broad claim that the full forecasting system is
   uniformly superior.
```

This preserves the negative and null historical results as part of the
research process while keeping the final claims aligned with the actual
evidence. The final thesis should present these results as rolling development
evidence until a genuinely untouched 2024+ confirmation sample is available.

### 12.5 Paper-ready conclusion and prohibited overclaim

Recommended conclusion:

> 在相同 parent checkpoint 和相同 continuation 预算下，pair-level LP
> residual FiLM-WGAN 对整体曲面和 short-ATM MAE 提供了小幅增量改善。进一步
> 固定模型、fold、seed 和训练设计后，LP 的 aggregate surface MAE 显著低于
> fold-train-only BoW、ChatGPT sentiment 和 continued no-text。

Do not write:

```text
LP is better for every local IV metric;
the model significantly beats persistence;
the result is confirmed on untouched 2024+ data;
statistical significance implies a large LP-versus-sentiment economic gain;
the predictive result identifies a causal news effect.
```
