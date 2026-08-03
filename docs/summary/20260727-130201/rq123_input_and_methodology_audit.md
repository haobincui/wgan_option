# RQ1-RQ3 Input and Methodology Audit

## Audit Metadata

```text
audit_time_utc: 2026-07-27T13:02:01Z
worktree: /home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone
branch: wgan_rq123_london_timezone
git_commit: 26f3a23291eb78aa2019505ed61aa0ffcdd8d767
active_pipeline: outputs/experiments/rq123_london_recalculation_20260725-185142
rq1_experiment: outputs/experiments/rq1_pair_text_raw_vol_continuation_london_20260725-185142
raw_dataset: data/processed/raw-excel/raw_vol_w5_s3_london_20260725-185142
```

本审计基于当前代码、raw-vol 生成产物、新闻 workbook、RQ1/RQ2 frozen
artifacts、RQ3 scheduled-news 实现，以及相关官方规则和预测评估文献。审计目的
不是选择更有利于 LP text 的设定，而是确认输入标签、时间顺序和统计口径是否足以
支持论文中的严格样本外结论。

## Overall Assessment

RQ1 到 RQ3 的研究问题具有清晰的递进关系：

1. RQ1 检验 LP text 相对 no-text 是否提供 incremental predictive information。
2. RQ2 在相同 quantitative backbone 下比较 LP、BoW 和 ChatGPT sentiment
   representations。
3. RQ3 检验 text predictive value 是否在 scheduled-news/high-information
   regimes 中更强。

该逻辑具有研究价值。Rahimikia, Zohren and Poon 的研究支持将金融文本与传统
volatility benchmark 结合，而不是单独依赖文本。但该文献只说明文本可能提供增量
信息，并不保证 text model 必然取得更小 MAE：

- Rahimikia, Zohren and Poon, *Realised Volatility Forecasting: Machine
  Learning via Financial Word Embedding*:
  https://arxiv.org/abs/2108.00480

当前 rolling folds、fold-train-only text transforms、共享 Stage-A no-text
parents、128-dimensional zero no-text、BoW train-only vocabulary、multi-seed 和
pair/day-level inference 的总体方向是合理的。

但是，当前 raw-vol 标签构造存在三个 P0 问题。它们会直接改变 target surface，
因此当前 London-timezone runs 只能保留为 exploratory diagnostics，不能作为
confirmatory thesis evidence。

## P0-1: Current and Target Windows Overlap

相关实现：

```text
src/wgan_option/surface_generation/data_helperd/window.py:255-286
```

当 `window_minutes=5` 时，当前代码使用：

```text
current minute buckets = t-5, t-4, ..., t
target minute buckets  = t, t+1, ..., t+5
```

`t` 这一完整 minute bucket 同时进入 current 和 target。新闻时间先被 floor 到
分钟，因此 current surface 还可能使用新闻 nominal time 之后约 60 秒内的交易。

本次 precalibration audit：

```text
precalibration points                  = 233699
points after nominal current anchor    = 39491
post-anchor fraction                   = 16.898%
```

这构成 temporal leakage，也会人为增强 persistence baseline 和预测模型。

必须改成不相交的 half-open windows：

```text
current = [t - 5min, t)
target  = [t, t + 5min)
```

重建时必须通过：

```text
current_trade_time < t
target_trade_time >= t
current/target trade overlap = 0
post-origin trades in current = 0
```

如果 Factiva 只能提供 minute-level publication time，应将 `t` 明确定义为该分钟
起点，并额外报告 `+1/+2/+5min` availability-lag sensitivity。

## P0-2: TY Option Expiration Is Incorrect

相关实现：

```text
src/wgan_option/surface_generation/data_helperd/all.py:667-699
src/market_data/contract_handler/contract.py:40-104
```

代码为 option contract 使用 `ContractTerminationRule.EndOfMonth`，并将其解释为
named option month 的最后一个工作日。

CBOT Rule 19A01.I 对 Treasury quarterly/serial options 的规则并非 named month
月末。通常应在 named option month 前一个月月底之前、满足规则的最后一个星期五
终止交易：

https://www.cmegroup.com/rulebook/CBOT/II/19A.pdf

当前规则会系统性扭曲：

```text
time-to-expiry
business_days
implied volatility
total variance
maturity-grid placement
surface interpolation/extrapolation
```

此外：

```text
expiration_time_utc = 20:00:00
data_date            = 2026-03-09
```

固定 UTC time 没有处理 Chicago daylight-saving time；contract year inference
还引用运行时 `today().year`，导致历史重跑结果依赖执行日期。

应新增 CME Treasury option contract calendar：

1. 按 standard/serial/weekly contract class 解析 symbol。
2. 在 `America/Chicago` 中确定 last trading/expiration datetime。
3. 最后转换为 UTC。
4. 用官方 2022-2023 expiration calendar 抽样核对。
5. 年份推断只依赖 trade date 和 contract symbol，不依赖系统当前日期。

## P0-3: The Fixed Raw-Vol Grid Is Mostly Extrapolated

相关实现：

```text
src/quantlib/vol_surface/algo/raw_surface.py:89-128
```

模型固定目标网格为：

```text
moneyness = 0.70 ... 1.30
maturity  = 7 ... 365 business days
grid      = 16 x 16
```

本次 raw observations 的典型范围为：

```text
source moneyness range          approximately 0.972 ... 1.029
median minimum maturity         approximately 37 business days
median maximum maturity         approximately 49 business days
```

4630 usable surface pairs 对应 9260 个 current/target directions：

```text
1 expiry slice = 4456
2 expiry slices = 3128
3 expiry slices = 1579
4 expiry slices =   97
```

Support audit 结果：

```text
7d below observed minimum maturity       = 9260 / 9260
average strike-grid clamp fraction        approximately 91.5%
average full-grid support upper bound     below 0.8%
directions with zero supported grid node  = 4950 / 9260
```

当前 raw interpolation 在 strike 方向 clamp 到 observed boundary。在 maturity
低于第一张 slice 时，它保留第一张 slice 的 total variance，再除以 requested
day。因而 `7d ATM vol` 是外推值，不是 7d market quote 的插值值。

相关文献说明简单 variance interpolation/extrapolation 可能引入静态套利或不稳定
尾部：

- Le Floc'h, *Arbitrages in the Volatility Surface Interpolation and
  Extrapolation*: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2175001
- Fengler, *Arbitrage-Free Smoothing of the Implied Volatility Surface*:
  https://ideas.repec.org/a/taf/quantf/v9y2009i4p417-428.html
- Gatheral and Jacquier, *Arbitrage-Free SVI Volatility Surfaces*:
  https://arxiv.org/abs/1204.0646

推荐修改：

```text
primary raw-vol grid = train-fold liquid support domain
primary raw-vol MAE  = evaluated only on an observed/support mask
minimum expiries     = at least 2 for maturity interpolation
7d ATM metric        = removed unless actual 7d quotes exist
broad 7d-365 surface = arbitrage-constrained SVI/SSVI primary representation
raw interpolation    = local-support robustness representation
```

不得用 test folds 选择 support grid。每个 rolling fold 的 grid/mask 必须仅由该
fold 的 train data 冻结。

## P1-1: Implied-Volatility Inversion Needs a Contract-Model Audit

相关实现：

```text
src/wgan_option/surface_generation/backend/surface_cpu/all.py:41-59
```

当前实现把 futures price 作为 `spot`，并令 `r=0, q=0` 调用
Black-Scholes-style inversion。Black (1976) 是 European futures-option model，
而 CBOT Treasury options 允许提前行权。

参考：

- Black, *The Pricing of Commodity Contracts*:
  https://www.sciencedirect.com/science/article/pii/0304405x76900246/pdf
- CME/CBOT Rule 19A, nature and exercise of Treasury options:
  https://www.cmegroup.com/rulebook/CBOT/II/19A.pdf
- *Pricing American Options: Managing Risk with Early Exercise*:
  https://rpc.cfainstitute.org/research/financial-analysts-journal/1986/pricing-american-options-managing-risk-with-early-exercise

在到期日修正后，应比较：

```text
Black-76 with maturity-matched discount curve
American lattice or Barone-Adesi-Whaley sensitivity
call-only / put-only / OTM-only / parity-filtered surfaces
```

还应避免用整个六分钟窗口 futures VWAP 为所有 option trades 提供同一个 underlying
price。更合理的是按 option trade timestamp 同步 nearest futures quote/trade，并
报告 stale-price、spread 和 liquidity filters。

## P1-2: Factiva Timestamp Is Plausible but Not Fully Proven

相关实现：

```text
src/wgan_option/news_time.py:91-163
```

`Europe/London` 对当前 workbook 有较强内部支持：

```text
total news rows                         = 14900
parse success                           = 14900
rows with embedded GMT cross-check      = 3171
within 2 minutes                        = 3094
agreement rate                          = 97.57%
```

在 usable subset 中仍有 15 个 mismatches，其中至少两个约为 60 分钟偏差。Factiva
关于 GMT 的搜索日期说明不必然等同于 ET publication field 的 timezone definition。

应保存原始导出设置和 timezone evidence，人工解决 77 个 mismatches，并报告：

```text
Europe/London primary
fixed UTC sensitivity
exclude timezone-mismatch rows
+1/+2/+5min publication-availability lag
```

即使 publication timestamp 正确，也不等于文本在该时刻已经可以被自动交易系统
完整获取。

## P1-3: Selection and Text-Lineage Risks

数据选择：

```text
raw news rows = 14900
usable pairs  = 4630
usable rate   = 31.07%
2022 usable   = 2803
2023 usable   = 1827
```

因此论文 estimand 应表述为：

> News observations with sufficient contemporaneous TY option liquidity to
> construct both current and five-minute-ahead surfaces.

不能直接推广到全部 Factiva news。应比较 selected/excluded rows 的年份、时段、
source、headline/text length 和 event-family distributions。

文本 lineage 的问题：

1. `src/film_wgan/data.py:507` 将缺失 `ArticleID` 转为字符串 `"nan"`，同一 pair
   的多个缺失 ID 可能被错误地当作同一文章。
2. 28 rows 的 LP 为空，其中 10 rows 进入 usable sample，但 frozen LP embedding
   非零，无法从保存的文本重现。
3. LP embedding artifact 没有完整记录 model version、API settings、
   preprocessing、token truncation 和 generation time。
4. 发现 190 个 exact-embedding duplicate groups，共 487 rows；其中部分跨 pair
   或 split。新闻 wire boilerplate 和重复更新可能让 test text 与 train text
   近乎相同。

应：

```text
use pd.isna before ArticleID fallback
recover or exclude empty-text/nonzero-embedding rows
freeze embedding generation manifest and cache
audit exact and near-duplicate text clusters
report sensitivity excluding recurring market-wrap boilerplate
```

Manela and Moreira 在基于新闻文本构造不确定性指标时也显式处理重复出现的日常标题：

https://amoreira2.github.io/alan-moreira.github.io/NVIX_published.pdf

## RQ1 and RQ2 Assessment

以下实现方向合理：

```text
rolling non-overlapping outer-test quarters
fold-train-only LP PCA
fold-train-only BoW vocabulary and PCA
shared Stage-A no-text parent by fold and seed
same 128-dimensional interface for text and no-text
zero-initialized residual text branch
all seeds reported without best-seed selection
pair/day-level dependence-aware inference
```

需要补充：

- Validation MAE 使用 Monte Carlo scenarios 时应固定 common random numbers，并
  报告 Monte Carlo standard error，避免 checkpoint ranking 受随机采样噪声影响。
- Empty BoW/sentiment input 经 centering、PCA 或 z-score 后可能变为非零，因此
  missing-text row 必须使用显式 `has_text=0` mask。
- ChatGPT sentiment 是 custom low-dimensional score baseline，不是 ground truth；
  需要人工标注子样本、prompt/model version、raw response cache 和 repeat-scoring
  stability。
- 若论文强调 WGAN distributional contribution，不能只比较 mean-surface MAE。

Probabilistic outputs 应同时报告：

```text
energy score
variogram score
marginal interval coverage
calibration error
scenario spread
calendar and butterfly violation rates
```

Energy score 对错误 cross-grid dependence 的识别能力有限，variogram score 可作为
补充：

- Scheuerer and Hamill, *Variogram-Based Proper Scoring Rules for
  Probabilistic Forecasts of Multivariate Quantities*:
  https://repository.library.noaa.gov/view/noaa/22327/

## RQ3 Assessment

相关实现：

```text
configs/rq3/scheduled_news_regime_raw_vol.yaml
scripts/rq3/scheduled_news_regime.py:554-660
scripts/rq3/scheduled_news_regime.py:708-905
scripts/rq3/scheduled_news_regime.py:1100-1302
```

当前实现完成的是：

> Conditional forecast-performance analysis for Factiva news observations
> published around scheduled macroeconomic releases.

它不是 Nakamura and Steinsson 意义上的 causal high-frequency event study。
Nakamura and Steinsson 使用官方 FOMC announcement 周围的 30-minute asset-price
change，并把窄窗内意外利率变化解释为 monetary-policy news：

https://academic.oup.com/qje/article-abstract/133/3/1283/4828341

当前 RQ3 的 current snapshot 仍是 Factiva article publication time。即使该时间落在
official release window，文章也可能在市场已经发生反应后才发布，而且当前数据没有
announcement surprise magnitude。

Treasury 市场确实会在五分钟宏观公告窗口产生显著反应：

- Bollerslev, Cai and Song, *Intraday Periodicity, Long Memory Volatility,
  and Macroeconomic Announcement Effects in the US Treasury Bond Market*:
  https://www.sciencedirect.com/science/article/pii/S0927539800000025

推荐将 RQ3 分成两种证据：

### Primary: Conditional Predictive Robustness

在所有 out-of-sample pairs 上构造：

```text
text_advantage = baseline_error - text_error
```

回归或检验：

```text
text_advantage ~ scheduled_news_indicator
                 + event-family fixed effects
                 + fold fixed effects
                 + forecast-origin controls
```

使用 release/trading-day clustered inference，并报告 Giacomini-White conditional
predictive ability test：

https://doi.org/10.1111/j.1468-0262.2006.00718.x

总体 equal predictive accuracy 继续报告 Diebold-Mariano/HAC：

https://doi.org/10.1080/07350015.1995.10524599

当前 greedy matched event-vs-ordinary-news analysis 可保留为 secondary robustness，
但匹配 covariates 必须基于修正后、真实有支撑的 raw surface features。

### Mechanism: Official Release Event Study

如果要提出 causal/event-shock interpretation，应重新构造：

```text
surface before official release
surface after official release
survey-standardized macro surprise
FOMC target/path or information-shock measures
```

此分析不能使用 Factiva article time 替代 official release time。

## Required Rebuild Order

```text
1. Correct CME option expiration and deterministic year parsing.
2. Make current/target raw-trade windows disjoint.
3. Recompute IV with synchronized futures inputs and model sensitivities.
4. Define train-only liquid support grids and support masks.
5. Remove unsupported raw-vol 7d ATM as a primary metric.
6. Rebuild raw surfaces and rerun coverage/lineage validation.
7. Fix empty text, ArticleID, duplicate and embedding-manifest issues.
8. Rerun RQ1.
9. Rerun RQ2 from the corrected shared RQ1 parents.
10. Run RQ3 as conditional robustness; add official-release/surprise analysis
    only if causal interpretation is required.
```

Before training resumes, the dataset build should hard-fail unless:

```text
current/target raw trade overlap = 0
post-origin current trades       = 0
sampled CME expiries             = official calendar
train-derived support mask       = frozen and archived
primary metrics                  = inside observed support
input/config/artifact SHA256     = complete
```

## Thesis Claim After Correction

A defensible target claim is:

> Conditional on a liquid, empirically supported region of the TY implied
> volatility surface, LP semantic information provides incremental
> out-of-sample predictive value when introduced as a residual conditioning
> signal to a quantitative FiLM-WGAN backbone.

The current data do not support the broader claim that raw interpolation accurately reconstructs
and predicts the full 7d-365d, 70%-130% surface. The corrected experiments must be interpreted
according to the observed support and must report negative as well as positive text results.

## Pipeline Stop Record

Immediately before the requested stop:

```text
top-level PID/PGID: 3844478
command: bash scripts/rq123_london/run_full_recalculation.sh
active stage: RQ1 results-pipeline / test JSON generation
active child: pair_pca_text_residual_pretrained, 2023Q4, seed 42
training matrix status: 84/84 training runs had completed
RQ2/RQ3 downstream recalculation: not yet started by the top-level pipeline
```

The stop action is intended to preserve all completed training, checkpoints, partial test JSON,
registries and logs. No output is to be deleted.

Actual stop verification:

```text
term_signal_sent_utc: 2026-07-27T13:04:00Z approximately
verification_time_utc: 2026-07-27T13:04:26Z
target_process_group: 3844478
target_processes_remaining: 0
forced_kill_required: false
resolved training configs preserved: 84
checkpoint files preserved: 810
partial generated sample JSON preserved: 26946
experiment log files preserved: 151
experiment size preserved: 163601236089 bytes
```

One GPU-visible `table_c7.py` process remained after the stop. Its command and process group were
unrelated to this RQ123 worktree and experiment, so it was intentionally left running.





做一个滚动，e.g. 使用20天 (t - 20) 来预测下一个， 保证信息都拿到，

attention weight， ， 历史某一个时间点对今天可能会有影响，，不需要手动处理weights。


冲击对图像冲击的解释性， 不同的冲击大小对图像冲击的程度。

使用MC来模拟，来做情景分析，讲究实用角度。
