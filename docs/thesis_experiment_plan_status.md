# Thesis Experiment Plan and Current Status

这份文档面向博士论文第三章 `docs/chapter3.tex` 的写作和后续实验推进。它不是最终论文正文，
而是一份 chapter-facing planning note：把第三章当前叙事、项目已经实现的实验结果、以及还需要补充
的材料对应起来。

当前整理依据：

- `docs/chapter3.tex`
- `docs/project_onboarding_for_agents.md`
- `docs/training_results_comparison.md`
- 当前 `outputs/training` 中保存的训练结果目录和指标总结
- 当前代码中已经实现的 generation / merge / training / result / analysis workflow

注意：本工作区当前没有可直接检查的本地 `data/` 目录。因此这里对 raw data 和 merged workbook
的理解来自代码、docs、configs、tests 和 saved `outputs/` artifacts，而不是重新读取本地原始数据。

## 1. 实验计划 / Experiment Plan

### 1.1 Chapter-Level Research Question

第三章题目是：

```text
Volatility Surface Prediction with Textual Information
```

当前项目实验的核心问题应保持为：

```text
Can news text embeddings help forecast short-horizon changes in bond-option
volatility structures?
```

这和 `chapter3.tex` 中的 RQ1-RQ4 对应关系如下：

- RQ1 Predictive Accuracy:
  - 检验 textual embeddings 是否比 purely quantitative inputs 更能提升 IVS prediction accuracy。
- RQ2 Model Comparison:
  - 比较 LLM-based embeddings 与更传统 text representation / no-text baseline 的信息含量。
- RQ3 Intraday Dynamics:
  - 检验模型是否能捕捉 news / policy announcement 后的 intraday IVS changes。
- RQ4 Practical Implications:
  - 从 option pricing、hedging、risk management 角度解释预测改进的实际意义。

当前 `outputs/` 结果主要覆盖 RQ1 和 RQ3 的 surface-forecasting 部分；RQ2 和 RQ4 还需要更系统的
ablation、baseline 和 thesis interpretation。

### 1.2 Contribution Mapping

`chapter3.tex` 当前写了四项 contribution。实验计划需要逐项给出证据：

- Textual Information in IVS Modeling
  - 用 Dow Jones Newswires 的 news text，通过 LLM embeddings 进入 IVS forecasting model。
- Hybrid Modeling Framework
  - 当前项目已经实现 current volatility surface + text embedding -> future volatility surface。
- Benchmarking Against Established Models
  - 当前已有 persistence、legacy WGAN、VolGAN、CNN WGAN、FiLM WGAN 等比较。
  - 还需要把 SABR / cubic / SVI / deterministic regression / no-text baseline 的关系整理成论文表格。
- Intraday Prediction of IVS
  - 代码语义是 `backward/current` -> `forward/future`，默认 5-minute horizon。
  - 这应成为第三章实证部分的核心 framing。

### 1.3 Data Lineage Plan

第三章的数据叙事已经包括两类数据：

- Numerical data:
  - U.S. 10-Year Treasury Note futures and options
  - CME-listed contracts
  - Refinitiv DataScope limit order book / tick data
  - raw extracted period: January 1, 2022 to December 31, 2023
  - chapter focus period: March 17, 2022 to July 27, 2023, covering ten consecutive FFR increases
- Textual data:
  - Dow Jones Newswires
  - Federal Reserve related news
  - final dataset: 14,872 relevant news
  - date range: January 27, 2022 to December 30, 2023
  - fields: `HD`, `LP`, `PD`, `ET`
  - current chapter text says the model uses `LP` as the main news content

项目实验中的 executable data lineage 应写成：

```text
raw option trades
  + Dow Jones news embeddings
  -> implied volatility / SVI or surface generation
  -> merged_svi.xlsx / merged_vol.xlsx
  -> trainable current-to-future pairs
  -> model training
  -> generate_result / analyze_error
  -> thesis tables and figures
```

关键语义必须保持明确：

- `backward/current` = news row 的原始 `timestamp_utc`
- `forward/future` = `timestamp_utc + offset_minutes`，当前默认是 5 minutes
- `merged_vol.xlsx` 是 pair-level workbook，训练直接读取 `gan_input_ready`
- `merged_svi.xlsx` 是 direction-level workbook，SVI training 从
  `news_direction_audit` 中按 `news_row_id` 配对 `backward` 和 `forward`

### 1.4 Surface Construction Plan

第三章当前写法是：

- 使用 Black Model / Black-76 从 option price 反推出 implied volatility
- 使用 5-minute interval 构造 IVS
- 强调 FOMC announcement 前后 futures price 和 IVS 的变化

项目代码层面的 surface construction 还包括：

- SVI slice calibration
- SABR slice calibration
- cubic spline reconstruction
- raw filtered IV slice storage
- fixed grid reconstruction for merged vol workflow

论文中建议把 SVI / SABR / cubic / raw 写成 surface construction and representation choices，
不要把它们和最终 forecasting model 混在同一段。Black-76 是 IV extraction layer，SVI/SABR/cubic
是 surface representation layer，WGAN/FiLM/CNN 是 forecasting model layer。

### 1.5 Model Plan

`chapter3.tex` 当前 model section 已经确定以下主线：

- CGAN / conditional GAN
- WGAN / Wasserstein-1 distance for training stability
- no-arbitrage constraints:
  - butterfly arbitrage
  - calendar arbitrage
- model input:
  - IVS at time `t`
  - text-embedding vectors at time `t`
- model output:
  - IVS at time `t+1`
- later subsection placeholders:
  - `Text-Embeddings`
  - `CNN for Feature Extraction`
  - `FiLM Layer for Conditional Information Encoding`
  - `FiLM-GAN Model Architecture`

当前实验计划应该围绕 FiLM WGAN 作为 thesis main model 展开，同时保留以下 comparison families：

- persistence / current-surface baseline
- no-text baseline
- deterministic `vol-regression-xlsx`
- `svi-xlsx` SVI regressor
- VolGAN
- CNN WGAN
- Transformer WGAN
- StyleMod WGAN
- CrossAttention WGAN
- legacy raw/svi WGAN

### 1.6 Evaluation Plan

Evaluation 需要分别服务于 RQ1-RQ4：

- RQ1:
  - `val_mae_gap_vs_current`
  - `val_mae`
  - current-surface persistence baseline
  - no-text / text embedding ablations
- RQ2:
  - `hd`, `lp`, `concat`, no-text
  - chapter 中提到的 BoW / sentiment scores 目前还没有完整实现，需要作为 missing experiment 或 limitation
- RQ3:
  - 5-minute horizon
  - news timestamp alignment
  - short-end / near-ATM reaction around policy announcements
- RQ4:
  - short-ATM errors
  - option pricing / hedging relevance
  - no-arbitrage diagnostics

Metric caveat 必须写清楚：legacy `val_recon`、broad gap-based metrics、short-ATM metrics 不能混成
一个排行榜。它们回答的问题不同。

## 2. 目前已经实现的 / Implemented So Far

### 2.1 Chapter Draft State

`docs/chapter3.tex` 当前已经有：

- Introduction
- Research Question and Contribution
- Literature Review
- Model Details
- Non-arbitrage Condition
- Conditional GAN / WGAN framing
- Dataset section:
  - U.S. 10-Year T-Note futures and options
  - Refinitiv DataScope
  - Black-76 IV extraction
  - Dow Jones Newswires text dataset
  - `text-embedding-3-large`
  - 512-dimensional and 1,024-dimensional embedding discussion
- Empirical Result section placeholder
- Main Results placeholder
- Robustness Tests placeholder
- Conclusion placeholder

Chapter 中仍有明显 TODO / 空缺：

- `Text-Embeddings` subsection 还需要补全获取 embedding 的具体流程。
- `CNN for Feature Extraction` subsection 还没有内容。
- `Feature-wise Linear Modulation (FiLM)` subsection 还没有内容。
- `FiLM-GAN Model Architecture` subsection 还没有内容。
- `Empirical Result`, `Main Results`, `Robustness Tests`, `Conclusion` 还需要用 outputs 结果填充。

### 2.2 Pipeline Implementation

项目已经实现完整的 executable workflow：

- `scripts/generate_surface/main.py`
  - supports SVI / SABR / cubic / raw surface generation
  - supports `all`, `window`, and `excel` data ranges
- `scripts/merge_file`
  - `merge_svi.py`
  - `merge_vol.py`
  - `merge_params.py`
- `scripts/train/main.py`
  - `vol-xlsx`
  - `vol-regression-xlsx`
  - `svi-xlsx`
- `scripts/generate_result/main.py`
  - sample-level result generation for trained mainline models
- `scripts/analyze_error/main.py`
  - error-distribution analysis for vol and SVI result payloads
- standalone model CLIs
  - `scripts/volgan/main.py`
  - `scripts/cnn_wgan/main.py`
  - `scripts/transformer_wgan/main.py`
  - `scripts/film_wgan/main.py`
  - `scripts/stylemod_wgan/main.py`
  - `scripts/crossattn_wgan/main.py`

### 2.3 Workbook and Pairing Semantics

已经实现并在 docs/tests 中固定的关键语义：

- `merged_vol.xlsx`
  - already pair-based
  - `gan_input_ready` 中一行就是一个 usable forecasting example
  - current side = `backward`
  - target side = `forward`
- `merged_svi.xlsx`
  - direction-level, not pair-level
  - `gan_input_ready` 是 direction-level filtered export
  - current SVI training 从 `news_direction_audit` 中配对 `backward` 和 `forward`

这一点要进入第三章 data construction 或 empirical setup。否则读者会误以为 SVI workbook 和 vol workbook
都已经是同一种 pair-level training table。

### 2.4 Saved Output Snapshot

根据 `docs/training_results_comparison.md`，当前 `outputs/training` 快照包含：

| Status | Count |
| --- | ---: |
| `finished` | 30 |
| `partial/interrupted` | 13 |
| `early stopped` | 8 |
| `no metrics` | 2 |

合计 53 个 discovered run directories，其中 51 个有 `training_metrics.csv`，2 个是 setup/dry 或
aborted runs，没有 epoch metrics。

模型族覆盖：

| Family | Runs |
| --- | ---: |
| `film_wgan` | 18 |
| `cnn_wgan` | 15 |
| `volgan` | 6 |
| `transformer_wgan` | 5 |
| `legacy_wgan` | 5 |
| `stylemod_wgan` | 3 |
| `crossattn_wgan` | 1 |

### 2.5 Current Empirical Findings

当前保存结果支持以下阶段性结论：

- FiLM WGAN 是当前保存结果中最强的 surface-forecasting family。
  - best broad paired-surface gap:
    `outputs/training/film_wgan/svi-excel/20260417_180233`
  - `val_mae_gap_vs_current = -0.002785`
  - best epoch = 18
- Best pure short-ATM checkpoint 来自：
  - `outputs/training/film_wgan/svi-excel/20260417_131244`
  - `val_atm_short_pure_mae_gap_vs_current = -0.001601`
  - best epoch = 9
- CNN WGAN 是最强的 simpler baseline。
  - best stable CNN reference:
    `outputs/training/cnn_wgan/svi-excel/20260414_123032`
  - `val_mae_gap_vs_current = -0.001999`
  - `val_mae = 0.023857`
  - best epoch = 19
- CrossAttention WGAN stable but weaker than CNN/FiLM。
  - single completed run:
    `outputs/training/crossattn_wgan/svi-excel/20260414_230946`
  - best gap = `-0.000700`
- StyleMod WGAN 已经 trainable，但当前不够 competitive。
  - tuned run `20260414_233543` selected checkpoint gap is positive:
    `+0.000851`
- VolGAN 是 low-capacity reference。
  - best gap-aware run reaches `val_mae_gap_vs_current = -0.000432`
  - useful as a simple baseline, weaker than CNN/FiLM
- Transformer WGAN 在当前 snapshot 中更像 negative control。
  - measured reconstruction errors are much worse than paired current-surface baselines
- Legacy raw/svi WGAN runs 只适合作为 lineage/context。
  - 它们使用 legacy `val_recon` 或 `val_hybrid_score`
  - 不应该直接和 later standalone gap-based `svi-excel` runs 排名

### 2.6 Thesis-Relevant Interpretation So Far

当前结果可以支持第三章 empirical result 的初步结构：

- 模型可以在部分设置下 beat current-surface persistence baseline。
- Architecture matters:
  - VolGAN 明显弱
  - CNN WGAN 是强 baseline
  - FiLM WGAN 当前最好，最贴合 `chapter3.tex` 中 FiLM-GAN 主线
- Text conditioning 的 thesis claim 还需要 no-text / text-mode ablation 支撑。
- Short-end near-ATM region 值得单独讨论，因为它更接近 policy announcement 后的实际交易和 hedging
  relevance。
- 结果表述时必须把 broad-surface improvement 和 short-ATM improvement 分开。

## 3. 还需要补充的 / Remaining Work

### 3.1 Fill Chapter Model Details

第三章 model section 需要补全：

- `Text-Embeddings`
  - 明确使用 Dow Jones `LP`，同时解释 `HD` / `LP` / embedding dimensions 的关系。
  - 说明 `text-embedding-3-large` 的 512D / 1024D 实验设计。
  - 如果最终代码默认使用 `HD_embedding` 或 `LP_embedding`，论文需要和实验配置对齐。
- `CNN for Feature Extraction`
  - 解释 surface grid 的 maturity-strike layout。
  - 说明 CNN 如何提取 local smile / term-structure pattern。
- `FiLM Layer for Conditional Information Encoding`
  - 解释 text embedding 如何生成 per-channel modulation。
  - 强调 FiLM 的作用不是简单 concat，而是让 text condition 改变 surface feature processing。
- `FiLM-GAN Model Architecture`
  - 写清楚 generator、critic/discriminator、noise、current surface、text embedding、future surface target。
  - 和当前最强 saved results 对齐，避免只写 generic GAN。

### 3.2 Text Usefulness and RQ2 Ablations

还需要补充或整理：

- `hd` vs `lp` vs `concat` text embedding mode 对比
- no-text baseline
- same architecture with/without text conditioning
- same data split 下 persistence baseline、deterministic baseline、text-conditioned model 的直接比较
- BoW / sentiment score 是否作为正式 baseline
  - `chapter3.tex` 的 RQ2 提到 traditional text-mining approaches
  - 当前项目主要是 LLM embedding / no-text / text-mode ablation
  - 如果不实现 BoW 或 sentiment score，需要在 thesis 中调整 RQ2 或作为 limitation

### 3.3 Representation and Benchmarking

第三章 contribution 中写了 benchmarking against established models。还需要把 benchmark 口径整理清楚：

- deterministic `vol-regression-xlsx` vs adversarial `vol-xlsx`
- `svi-xlsx` SVI parameter regressor vs SVI-reconstructed surface forecasting
- FiLM/CNN surface models vs SVI regressor
- SVI / SABR / cubic / raw representation 是否进入正式 table
- Heston / SABR / spline 这些 literature benchmark 和当前 executable benchmark 的边界

不能只把不同历史 run 的最优数字放在一起。每个 benchmark table 必须说明 data split、metric、
sample universe 和 checkpoint metric。

### 3.4 Statistical Robustness and Robustness Tests

`chapter3.tex` 已经有 `Robustness Tests` 占位。建议补充：

- seed repeat 或至少记录 seed sensitivity
- bootstrap confidence intervals
- time-split robustness
- sample-level win rate and distribution plots
- error by maturity bucket / moneyness bucket
- short-ATM subset vs full-surface subset 的 paired comparison
- around-announcement subset vs all-news subset

如果论文要说 text-conditioned FiLM 明显优于 CNN 或 persistence，最好有 bootstrap / paired test 支撑。

### 3.5 Empirical Result Tables and Figures

`Empirical Result` 和 `Main Results` 需要整理成论文可用材料：

- model-family comparison table
- broad `val_mae_gap_vs_current` table
- short-ATM table
- no-text / text-mode ablation table
- sample-level generated vs real surface plots
- current / generated future / real future term-structure and smile views
- FOMC event examples: futures movement and IVS change around announcement
- data lineage figure: raw option trades + news embeddings -> generated surfaces -> merged workbook -> model training
- quality-filter summary: usable vs excluded samples and exclusion reasons

图表中需要标注 metric family，避免把 legacy `val_recon`、gap-based metrics 和 short-ATM metrics 混成一个
scoreboard。

### 3.6 Reproducibility and Clean Environment

还需要补充 clean-run 相关事项：

- Python requirement: project requires Python `>=3.10`
- 当前本地工作区没有 `data/` 目录，raw data 和 merged workbook 需要在实验环境中确认
- 保存每个 thesis-cited run 的:
  - resolved config
  - best checkpoint metadata
  - metrics CSV/JSON
  - generated result samples
  - plots and summary tables
- 确认 `film_wgan` 和 `crossattn_wgan` 在 clean install 下的 packaging 问题
  - 当前 `pyproject.toml` package include list 未显式包含 `film_wgan*` 和 `crossattn_wgan*`
  - script execution may still work through repo-root / `src` path setup
  - clean packaging or non-editable install 前需要验证

### 3.7 Writing Tasks for `chapter3.tex`

建议后续按以下顺序推进第三章：

1. 对齐 RQ：
   - 如果不实现 BoW / sentiment score baseline，调整 RQ2，避免承诺 traditional text-mining comparison。
2. 补全 `Text-Embeddings`：
   - 数据字段、tokenization、embedding model、dimension、HD/LP 使用口径。
3. 补全 `CNN` / `FiLM` / `FiLM-GAN`：
   - 用当前最强 FiLM WGAN 实现写 architecture，而不是泛泛写 GAN。
4. 补全 empirical setup：
   - data split、5-minute horizon、workbook semantics、quality filters、metrics。
5. 写 main results：
   - FiLM broad gap、FiLM short-ATM、CNN baseline、negative-control families。
6. 写 robustness tests：
   - no-text / text-mode ablation、seed / bootstrap、subsample robustness。
7. 写 conclusion：
   - 回答 RQ1-RQ4，并明确 limitations。

最后修改日期：2026-06-11
