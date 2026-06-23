# 实验计划逻辑评审 / Experiment Plan Logic Review

评审对象：`docs/thesis_experiment_plan_status.md`（详细实验计划，后续会落成 `chapter3.tex` 实证部分）。
交叉依据：`docs/project_onboarding_for_agents.md`、`docs/training_results_comparison.md`、`docs/chapter3.tex`。

本文档**只评审实验计划本身的逻辑**：实验设计能否真正回答 RQ、有没有 confound / bias / 时序漏洞、
interim conclusion 是否被它引用的证据支撑、计划内部是否自洽。不评审 chapter3 的文字（已知 TODO）。

分级：**P0 = 逻辑硬伤，不修则结论站不住**；**P1 = 设计缺口，影响说服力**；**P2 = 增强严谨性**。

---

## 0. 总体判断 / Overall

**计划逻辑做得好的地方（先肯定）：**
- current/future（`backward`/`forward`）语义贯穿始终，时间方向不含糊。
- 明确区分 metric family，并自己写下"不要把 legacy `val_recon` / gap-based / short-ATM 混成一个排行榜"（§1.6、§3.3）——这是对的。
- 诚实记录了 onboarding §12 的 caveat："早期 no-text/current-surface baseline 很强"。
- §1.4 已经有"IV extraction / surface representation / forecasting"三层分离的雏形。

**但存在 4 个会让核心结论站不住的逻辑问题（按重要性）：**
1. **§2.5 用的主指标 `val_mae_gap_vs_current` 衡量的是"模型 vs persistence"，不是"text vs no-text"。** 全论文的中心 claim 是"text 有用"，但计划用来支撑它的所有数字都没把 text 的贡献单独隔离出来。**这是最致命的逻辑断层。**（详见 §1.1）
2. **RQ2 的实验被偷换了概念。** 计划把 RQ2 默默改成"LLM vs no-text"，但 no-text（删掉文本）不是 "traditional text-mining"。逻辑上回答不了原 RQ2。（§1.2）
3. **没有独立 test set，val 同时用于 checkpoint 选择和结果报告 → selection bias。** §2.5 全是"best epoch = X"的 validation 最优值，是乐观偏差。（§2.1）
4. **跨 family 的排名是 confounded 的。** §2.5 拿不同 run、可能不同 split/config/checkpoint-metric 的最优数字排出 "FiLM > CNN > ..."，而 §3.3 自己又说"不能这么做"——计划自相矛盾。（§2.2）

下面逐项展开。

---

## 1. RQ ↔ 实验设计的逻辑闭环（逐个 RQ 检查）

判定标准：**给定计划描述的实验，能否在不引入 confound 的前提下回答这个 RQ？**

### 1.1 RQ1（text vs 纯数值）— 核心 confound：gap-vs-persistence ≠ text usefulness（P0）

- **RQ1 要回答**："加入 textual embedding 是否比纯 quantitative input 更准。" → 这是一个**受控对比**：同 architecture、同 split、**只切换 text on/off**，看误差差异。
- **计划现状的逻辑漏洞**：§1.6 RQ1 和 §2.5 的所有证据都用 `val_mae_gap_vs_current`（= 模型误差 − persistence 误差）。但是：
  - **一个完全没有 text 的模型，只要学到 surface 动力学，也能 beat persistence。** 所以 "gap < 0" 只证明"模型比'明天=今天'强"，**完全不证明 text 有贡献**。
  - §2.5 写 "FiLM gap = −0.002785 → FiLM 最强"，这个推理链断在"最强 ≠ text 有用"。
- **必须改的逻辑**：RQ1 的主结果**不能**是 `gap_vs_current`，而应该是
  ```
  Δ_text = (no-text 模型误差) − (text 模型误差)，同架构同 split
  ```
  并对这个差做 paired 检验。`gap_vs_current` 只能当"模型是否有用"的辅助指标，不能当"text 是否有用"的主指标。
- **落地**：把 "same architecture with/without text"（§3.2 已列）从"还需补充"提升为 **RQ1 的第一主实验**，且必须有配对显著性。现在它被埋在 ablation 列表里，地位太低。
- ⚠️ 结合 onboarding §12（no-text baseline 很强），很可能 `Δ_text` 很小甚至不稳定。**计划必须预设"如果 text 提升不显著"的诚实写法**，否则一旦实验出来 text 没用，整章 claim 崩盘。建议现在就在计划里写好两套叙事分支（text 显著 / text 仅在 announcement 子集显著 / text 不显著但 representation 有价值）。

### 1.2 RQ2（LLM vs traditional text-mining）— 概念被偷换（P0）

- **RQ2 要回答**："LLM embedding vs 传统 text-mining（BoW / sentiment）的信息含量。"
- **计划的逻辑问题**：§1.1 把 RQ2 重述为"LLM-based embeddings 与更传统 text representation / **no-text baseline** 的信息含量"。**no-text 不是 "traditional text representation"** —— 删掉文本和"用更弱的文本表示"是两回事。用 no-text 对比，回答的是 RQ1（text 有没有用），**根本没碰 RQ2（哪种 text 表示更好）**。
- §3.2 自己也承认 BoW/sentiment 没实现。所以现状是：**RQ2 没有任何能回答它的实验。**
- **二选一（必须现在决定，不能拖到写作）**：
  - **(A) 补一个 traditional baseline**：BoW + TF-IDF，或 Loughran-McDonald 金融情感词典打分，喂进**同一个 forecasting head**（保持下游一致，只换 text 表示）。这样 RQ2 才真正成立。工作量中等。
  - **(B) 改 RQ2 的范围（若时间紧）**：把 RQ2 改成"LLM embedding 的**表示选择**"：`hd vs lp vs concat`、`512D vs 1024D`。但要在 limitation 明说"未与 BoW/sentiment 直接对比"。
  - 注意：(B) 下 `hd/lp/concat/512/1024` 都还是 LLM embedding 内部的变体，**不能**被包装成"LLM vs traditional"。措辞要诚实。

### 1.3 RQ3（announcement 后的 intraday dynamics）— 缺 event-study 设计 + 时序因果（P0/P1）

两个独立的逻辑问题：

**(a) RQ3 是 event-study，但计划把它当成全样本任务（P1）**
- RQ3 问的是"**immediately after major policy announcements**"。这是一个**子集**（FOMC / 关键 Fed 讲话附近），不是全部 14,872 条 news。
- 计划现状：所有 news 都按 5-min pair 同等对待，"around-announcement subset vs all-news subset" 只出现在 §3.4 robustness，地位太低。
- **逻辑修正**：RQ3 的主实验应是 **event-study**：圈定 announcement 时间窗（如 FOMC 公布 ±N 分钟），单独评估模型在这些窗口里对 IVS 跳变的跟踪能力，并和"安静时段"对比。否则无法 claim "模型捕捉到了 announcement 反应"——你只是在全样本上平均，announcement 信号被稀释。

**(b) 5-min horizon 的因果时序必须写死，否则有 look-ahead 嫌疑（P0）**
- 定义是 `current = news timestamp`，`future = news timestamp + 5min`，text embedding 在 current 时刻已知。
- **致命问题**：news 在 `timestamp` 时刻**已经公开**，那么 current 时刻的 surface **可能已经部分反映了这条 news**。于是：
  - 模型用"已含 news 信息的 current surface" + "news embedding" 预测 +5min。预测信号到底是什么？是"news 公布后的**持续 drift**"，还是泄漏？
  - 必须澄清：current surface 的快照窗口是否**严格早于或等于** news 时刻、future 窗口是否**严格在 +5min**、两个窗口**不重叠**。
- **计划要补一段明确的 timing/causality 说明**：news 公布时刻 vs surface 快照时刻的先后、是否有信息泄漏、为什么 +5min 的 surface 仍有可预测的增量。这段不写清楚，RQ3 会被 examiner 一句"这是不是 look-ahead"打掉。
- **附带（P1）**：为什么是 5 分钟？做 horizon sensitivity（1 / 5 / 15 / 30 min），证明结论对 horizon 不脆弱，顺带强化 intraday claim。

### 1.4 RQ4（pricing / hedging / risk 的实际意义）— 没有真实实验（P1）

- **RQ4 要回答**："预测改进对 option pricing、hedging、risk management 的实际意义。"
- **计划现状**：§1.6 RQ4 只列了 short-ATM errors、"option pricing / hedging relevance"、no-arbitrage diagnostics。**"relevance" 是断言，不是测量**——计划里没有任何一个把 surface 转成 price 或 hedge ratio 的实验。
- **逻辑修正（二选一）**：
  - **(A) 加一个小 case study**：用 generated future surface 给某些 OTM 合约定价 / 算 Vega-Gamma hedge ratio，对比"用 persistence surface"，报告 pricing error（bp 或 $）或 hedge P&L 改进。哪怕只做几个 announcement 事件，也比纯断言强。
  - **(B) 显式降范围**：把 RQ4 改成"discussion-level implication"，用 short-ATM 误差改进 + no-arbitrage 满足度做**定性**论证，并在 limitation 承认"未做真实 pricing/hedging backtest"。
- 现状若不改，RQ4 在答辩时只能空谈，是明显短板。

---

## 2. 评测方法论的逻辑漏洞

### 2.1 没有独立 test set → selection bias（P0）

- onboarding 显示只有 chronological **train/val** 两分，没有 test。计划 §2.5 全是 "best epoch = 18 / 9 / 19" 的 **validation 最优值**。
- **逻辑问题**：val 既用于 early-stopping / checkpoint 选择，又用于报告最终数字。"在 val 上选最好的 epoch，再报告这个 val 数字" = 用了 model selection 的信息去报告，**系统性乐观偏差**。跨 53 个 run 选最优，偏差更大（multiple comparisons）。
- **必须改**：三分 **train / val / test**（chronological），val 选模型，**test 报结果**。所有进论文的主数字来自 test。这是博士论文级别的硬要求，examiner 一定查。
- 如果数据量不够三分，至少要：固定 checkpoint-selection 协议（如固定 epoch 预算 + 单一 selection metric），并 honest 声明 val=report 的偏差。

### 2.2 跨 family 排名是 confounded 的（P0）

- §2.5 排出 FiLM(−0.002785) > CNN(−0.001999) > CrossAttn(−0.000700) > VolGAN(−0.000432) > StyleMod(+0.000851)。
- 但这些来自**不同 run**，很可能 **split / config / text-mode / checkpoint-selection-metric / seed 都不同**。§3.3 计划自己写了"不能只把不同历史 run 的最优数字放一起"——**§2.5 正好违反了 §2.3/§3.3 自己定的规则**。计划内部自相矛盾。
- **逻辑修正**：要 claim "FiLM 最好"，必须做一次**受控 head-to-head**：
  - 同一 data split、同一 text mode（如都用 lp）、同一 checkpoint-selection metric、同一 seed 协议、同一 epoch 预算；
  - 在**同一 test 集、同一 eval surface 空间**上比较。
- §2.5 现在的排名只能当"探索性观察 / 初筛"，**不能进论文主结论表**。计划要明确区分"探索性快照"和"受控正式对比"。

### 2.3 "negative control" 与 StyleMod 的措辞逻辑（P1）

- §2.5 把 Transformer WGAN 叫 "negative control"。**逻辑问题**：negative control 应是被**设计**成预期无效的对照，而不是"恰好没调好的模型"。把一个 underperforming 的模型事后命名为 negative control，是 post-hoc rationalization，examiner 会问"你是不是只是没把 Transformer 调出来"。
  - **改法**：要么公平地调一遍 Transformer（同预算），要么如实写"under the tested configurations, Transformer underperformed"，**不要**叫它 control。
- StyleMod selected gap = **+0.000851（比 persistence 还差）**。把一个连 persistence 都没打过的模型列进对比没问题，但计划要如实说它"未能 beat persistence"，并讨论是否 selection/training 不充分，而不是轻描淡写成 "not competitive"。

### 2.4 统计显著性缺失（P1）

- §2.5 的差距是 0.0008 ~ 0.0028 这种量级的单点数字，没有任何方差 / CI / 检验。**"FiLM 比 CNN 好 0.0008" 在没有 CI 时是没有意义的断言。**
- §3.4 已列 seed repeat / bootstrap / paired test —— 把它从"还需补充"提升为**所有主对比表的强制要求**：
  - seed repeat（≥3–5），报 mean ± std；
  - 对 test 误差做 **paired bootstrap** 或 **Diebold-Mariano**；
  - **sample-level win rate**（模型在多少比例样本上 beat 对手）。
- 没有这些，任何"A 优于 B"的排序都不可辩护。

---

## 3. 数据与预测目标的逻辑问题

### 3.1 预测目标是 SVI-smoothed surface，label 本身是 fit 而非 ground truth（P0/P1）

- 主路径是 `svi-excel`：target surface 是**从 SVI 参数重建的 16×16 grid**（onboarding §7）。
- **逻辑问题**：模型学的是"预测一张**被 SVI 平滑过的** surface"，而"真实未来 surface"本身也是一次 SVI 拟合，**不是市场原始 IV**。于是：
  - SVI 的拟合误差变成 **label noise**；
  - 模型在"平滑面"上的低误差，可能只是因为平滑面更好预测，而非真捕捉了市场。
- **计划要明确写死**：prediction target 到底是 (a) SVI 重建面，还是 (b) raw filtered IV 点？如果是 (a)，必须承认"目标是平滑表示"，并最好补一个"在 raw IV 点上评估"的 robustness（避免 SVI 既当 input 表示又当 evaluation 真值的循环论证）。
- 这点直接关系到"误差到底测的是什么"，是评测有效性的根。

### 3.2 quality filter 可能系统性排除 announcement 样本（与 RQ3 冲突）（P0）

- 质量门槛：`weighted_iv_rmse ≤ 0.05`、`exact_slice_point_ratio ≥ 0.5`，排除 `poor_fit` / `no_raw_points` 等（onboarding §6）。
- **逻辑冲突**：announcement 后的分钟正是**最 volatile、quote 最乱、SVI 最难拟合**的时刻 → 这些样本更可能 `poor_fit` 被过滤掉。**但 RQ3 恰恰要研究 announcement 后的反应。** 过滤器可能把 RQ3 最想要的样本删掉了。
- **必须做的逻辑检查**：统计 around-announcement 样本 vs 安静样本的**通过率**。如果 announcement 样本被过滤得更狠，RQ3 的样本是 selected/biased 的，结论不可推广。
- 计划 §3.5 提到 "usable vs excluded summary"——把它**和 RQ3 绑定**：必须报告 announcement 窗口内的 usable rate。

### 3.3 vol workbook 与 svi workbook 是不同 universe，跨表比较要统一 eval space（P1）

- §3.3 计划比较"FiLM/CNN surface models vs SVI regressor"。但：
  - surface models 训练在 `merged_vol.xlsx`（pair-level）；
  - SVI regressor 训练在 `merged_svi.xlsx`（direction-level，runtime pairing）；
  - 两者 quality filter / usable universe **可能不同**。
- **逻辑问题**：不同样本集上的误差**不可直接比**。要 claim "surface model 优于 SVI regressor"，必须：
  - 把两者评估到**同一 surface IV grid 空间**（SVI regressor 的预测参数 → 重建成同样 16×16 grid）；
  - 在**同一交集样本**上比较。
- 计划要写明这个 common evaluation protocol，否则 §3.3 的对比是 apples-to-oranges。

### 3.4 5-min horizon 的 baseline 含义（P2）

- persistence baseline = "future = current"。在 5-min horizon 上，IVS 变化很小，persistence 本身就**非常强**——这放大了 §1.1 的问题（beat persistence 容易，证明 text 有用难）。
- 计划应预期：在如此短的 horizon 上，所有模型相对 persistence 的 gap 都会小。所以**绝对 gap 数字小不代表模型差**，但也意味着**显著性检验更不可省**。这个 horizon 选择的含义要在计划里点明。

---

## 4. Baseline / Benchmark 设计逻辑

### 4.1 缺一个 trivial 统计 baseline（P1）

- 计划 §1.5 的 baseline 全是 neural（deterministic CNN regression 已是最简的）。**缺一个真正 trivial 的统计 baseline**：如 ridge / linear regression from (flatten current surface + text embedding) → future surface。
- **逻辑作用**：它回答"到底需不需要深度模型 / 对抗训练"。如果 linear baseline 已接近 FiLM，那 GAN 的复杂度不被支持。博士论文需要这个"下界"来证明模型复杂度的必要性。

### 4.2 benchmark 要按"受控变量轴"组织，不要 flat leaderboard（P1）

- 计划倾向把所有 family 放一张排行榜。**更有说服力的逻辑**是把对比组织成**沿单一变量的 ablation 轴**，每张表只动一个变量：
  - 轴 1（text）：no-text vs text（同架构）→ RQ1；
  - 轴 2（text 表示）：lp vs hd vs concat / 512 vs 1024 → RQ2(B)；
  - 轴 3（adversarial vs deterministic）：vol-xlsx WGAN vs vol-regression → 隔离对抗训练的价值；
  - 轴 4（architecture）：VolGAN vs CNN vs FiLM（同 text、同 split）→ 隔离架构；
  - 轴 5（representation）：SVI vs SABR vs cubic 作为表示的拟合质量。
- 每个 benchmark 隔离一个 claim。flat leaderboard 把多个变量混在一起，无法归因。

### 4.3 Heston/SABR 作为 forecasting benchmark 的承诺（P1）

- chapter3 Contribution 3 承诺 benchmark against Heston/SABR/spline。但在本项目里 SVI/SABR/cubic 是 **representation layer**，不是 forecasting baseline。
- 计划 §1.4/§3.3 已意识到要分层，**但要把话说死**：如果不真的把 SABR/Heston 做成"预测 t+1 surface"的 forecaster，就**不要**在 contribution 里承诺"benchmark against Heston/SABR as forecasters"。representation-level 的拟合对比和 forecasting-level 的预测对比是两件事，计划要分开列。

---

## 5. 计划内部一致性 / 自相矛盾（汇总）

| # | 矛盾点 | 位置 |
| --- | --- | --- |
| 1 | §2.3/§3.3 说"不要混排不同 run 的最优数字"，但 §2.5 正是这么排出 family 名次 | §2.5 vs §3.3 |
| 2 | §1.1 把 RQ2 重述为含 "no-text baseline"，与 chapter RQ2 的 "traditional text-mining" 口径不一致 | §1.1 vs chapter RQ2 |
| 3 | 中心 claim 是 "text 有用"，但所有引用证据用的是 "gap vs persistence"（测的是模型 vs persistence，不是 text vs no-text） | §2.5 全节 |
| 4 | RQ3 强调 announcement，但样本处理是全样本平均 + quality filter 可能删掉 announcement 样本 | §1.6 RQ3 vs §3.2/§3.5 |
| 5 | Transformer 被叫 "negative control" 但实为 underperforming run | §2.5 |

把这 5 条理顺，计划的逻辑就自洽了。

---

## 6. 修订优先级清单

| 顺序 | 任务 | 优先级 | 对应 |
| --- | --- | --- | --- |
| 1 | RQ1 主指标改为 Δ_text=(no-text)−(text)，配对显著性；预设 text 不显著的诚实分支 | P0 | §1.1 |
| 2 | 引入 train/val/**test** 三分，主数字一律来自 test | P0 | §2.1 |
| 3 | RQ2 决策：补 BoW/sentiment baseline，或改范围并在 limitation 声明 | P0 | §1.2 |
| 4 | RQ3 改 event-study + 写死 timing/causality（防 look-ahead） | P0 | §1.3 |
| 5 | 检查 quality filter 对 announcement 样本的通过率 | P0 | §3.2 |
| 6 | 写死 prediction target（SVI 重建面 vs raw 点），补 raw-IV 评估的 robustness | P0/P1 | §3.1 |
| 7 | family 对比改为受控 head-to-head（同 split/text/metric/seed），§2.5 降级为探索性 | P0 | §2.2 |
| 8 | 所有主对比加 seed repeat + bootstrap CI + win rate | P1 | §2.4 |
| 9 | RQ4 决策：加 pricing/hedging case study，或降为 discussion 并声明 | P1 | §1.4 |
| 10 | benchmark 改按"受控变量轴"组织；加 trivial linear baseline | P1 | §4.1/§4.2 |
| 11 | 统一跨表 eval space（SVI regressor vs surface model 同 grid 同交集样本） | P1 | §3.3 |
| 12 | horizon sensitivity（1/5/15/30 min） | P2 | §1.3/§3.4 |
| 13 | 修正 Transformer "negative control" / StyleMod 措辞 | P2 | §2.3 |

---

## 7. 新的详细实验计划（修订版）/ Revised Detailed Experiment Plan

本节把 §1–§6 的评审逐条落成**可执行的实验计划**。它替代旧 `thesis_experiment_plan_status.md` 的 §1.5/§1.6 评测部分，
目标是：**每个对比只动一个变量、所有主数字来自 test set、所有排序带显著性、每个 RQ 有唯一对应实验**。
落成 `chapter3.tex` 时，Empirical Result 一节直接按这里的 Phase 顺序写。

### 7.0 全局协议（适用于所有 Phase）/ Global Protocol

这是后面一切实验的前提，先一次性定死，避免每个实验各自为政。

1. **三分数据切分（chronological）**：按 `news_timestamp_utc` 排序后切 **train / val / test = 70 / 15 / 15**（比例可调，但必须时间顺序、不 shuffle）。
   - `val`：early-stopping + checkpoint 选择 + 超参搜索。
   - `test`：**只在最终报告时碰一次**，不参与任何选择。所有进论文的数字来自 test。
2. **统一 evaluation space**：所有 forecasting 模型一律评估在**同一张重建 16×16 IV grid**、**同一交集 usable 样本**、**同一 metric** 上。SVI regressor 的预测参数先重建成同样的 grid 再比（解决 §3.3）。
3. **预测目标写死**：primary target = **SVI 重建的 16×16 surface**；同时保留 **raw filtered IV 点**上的二次评估作为 robustness（解决 §3.1 的循环论证）。
4. **checkpoint 选择协议**：预先注册**单一** selection metric（建议 `val_mae`，broad surface），固定 epoch 预算。不准事后换 metric 选 checkpoint。
5. **seed 协议**：每个 config 跑 **5 个 seed**，test 上报 **mean ± std**。
6. **显著性协议**：任何"A 优于 B"的结论都要给：**paired bootstrap CI**（per-sample test 误差，≥10k 重采样）+ **Diebold-Mariano** 检验 + **sample-level win rate**。
7. **metric registry**：每个 metric 只定义一次，三类 metric（broad `val_mae` / short-ATM / legacy `val_recon`）**永不**同表混排。
8. **pre-registration**：在看 test 之前冻结 split / 超参 / selection metric / 假设。把这份冻结记录存进 run artifact。

> ⚠️ §2.5 现有的 53 个 run 全部降级为 **exploratory 初筛**，只用于挑出值得正式 head-to-head 的 family（FiLM / CNN / VolGAN），**不进论文主结论表**。

### 7.1 Phase 0 — 数据与样本固化（前置，无模型）

| 项 | 内容 | 解决评审 |
| --- | --- | --- |
| 0.1 | 生成并冻结 train/val/test split manifest（存 sample ID 列表） | §2.1 |
| 0.2 | **quality-filter 审计表**：announcement 窗口 vs 安静时段的 usable 通过率 | §3.2 |
| 0.3 | 定义 **announcement event 日历**：10 次 FFR hike 的 FOMC 公布时刻 + 关键 Fed 讲话，窗口 ±N min（建议 N=30，作敏感性） | §1.3a |
| 0.4 | 确认 prediction target = SVI 重建面；同时导出 raw-IV 评估点 | §3.1 |
| 0.5 | 写死 **timing 构造**：current snapshot 窗口严格 ≤ news ts、future 严格在 +5min、两窗不重叠、embedding 在 current 时刻可得 | §1.3b |

**产出**：split manifest、quality-filter 审计表（进论文，支撑 §3.2 的样本代表性辩护）、event 日历、timing 构造说明（进 chapter3 empirical setup，防 look-ahead 质疑）。

> 若 0.2 发现 announcement 样本被过滤得明显更狠，需在 Phase 0 就决定补救（放宽 announcement 窗口的 fit 阈值并单独标注，或在 limitation 显式声明 RQ3 样本偏向安静时段）。

### 7.2 Phase 1（RQ1）— text 有用性，**全论文中心实验**

这是把 §1.1 的核心缺陷修掉的实验。**主指标是 Δ_text，不是 gap_vs_current。**

- **设计**：固定**一个**架构（main = FiLM WGAN；在 CNN WGAN 上复现作 robustness），**只切换 text on/off**，其余完全相同（同 split、同超参、同 5 seeds）。
  - 条件 A：no-text（generator/critic 去掉 text 通道）
  - 条件 B：text（lp embedding）
- **主指标**：
  ```
  Δ_text = MAE_test(no-text) − MAE_test(text)
  ```
  报 paired bootstrap CI + DM 检验 + win rate。Δ_text 显著 > 0 才支持"text 有用"。
- **辅助指标**：两条件各自的 `gap_vs_current`（只用来说明"模型 vs persistence"，不充当 text 证据）。
- **预设三套诚实叙事分支（现在就写好，防结果翻车）**：
  - **B1**：Δ_text 在全样本显著 → text 直接支撑 RQ1，主线成立。
  - **B2**：Δ_text 仅在 announcement 子集显著（见 Phase 3）→ 改写 claim 为"text 的价值集中在 event 时段"，仍是有力结论。
  - **B3**：Δ_text 不显著 → 诚实报 null，把 contribution 重心移到 architecture（FiLM）/ representation，RQ1 写成"在本数据与 horizon 下，text 未带来显著 broad 提升"。**这不是失败，是诚实的博士结论。**

**产出**：Table RQ1（{FiLM, CNN} × {no-text, text}，test MAE mean±std、Δ_text、CI、p、win rate）。

### 7.3 Phase 2（RQ2）— text 表示对比 ✅ 路线 A（已定）

**决策已定：走路线 A**——实现 **BoW + TF-IDF** 和 **Loughran-McDonald 金融情感打分** 两个 traditional text baseline，**喂进与 Phase 1 完全相同的 forecasting head**（只换 text 表示层，下游 FiLM-WGAN 架构、split、协议全部一致）。
- Table RQ2-A：**no-text → BoW(TF-IDF) → sentiment(LM) → LLM-embedding** 的递进对比（同协议）。期望误差单调下降才支持"LLM embedding 信息含量更高"。
- 同时把 LLM 内部表示对比（`lp vs hd vs concat`、`512D vs 1024D`）作为 Axis 2 的子表，回答"哪种 LLM 表示最好"。
- RQ2 这样才真正回答"LLM vs traditional text-mining"，chapter RQ2 措辞**不需要降级**。
- 执行落点见 §8.3。

### 7.4 Phase 3（RQ3）— announcement event-study + horizon 敏感性

修掉 §1.3 的两个问题。

- **(a) Event-study（主实验）**：把 **test 集**切成 *announcement-window 样本* vs *quiet 样本*，分别评估 {persistence, no-text, text} 三者。
  - **核心假设**：Δ_text 在 announcement 窗口 **大于** quiet 窗口。这是证明"text 捕捉 policy 反应"的最干净位置，也直接对应 Phase 1 的 B2 分支。
  - 配 **event case study 图**：选 2–3 个 FOMC（如 2022-11-02），画 announcement 前后 IVS 跳变 + 模型预测的跟踪。
- **(b) Timing/causality**：把 Phase 0.5 写死的构造在 empirical setup 正式陈述，明确"无 look-ahead"（current 窗口不晚于 news、future 严格 +5min）。
- **(c) Horizon sensitivity**：重复主实验于 horizon ∈ {1, 5, 15, 30} min，证明结论不脆弱。

**产出**：Table RQ3（announcement vs quiet × {persistence, no-text, text}，含 Δ_text 分组对比）；horizon-sensitivity 曲线；event case-study 图。

### 7.5 Phase 4（RQ4）— pricing / hedging 实际意义 ✅ 路线 A（已定）

**决策已定：走路线 A**——做真实的 **option pricing / hedging case study**：
- 选若干 announcement 事件（与 Phase 3 event 日历共用），用 **generated future surface** 给选定 OTM 合约定价 / 算 **Vega-Gamma hedge ratio**；
- 对比用 **persistence surface**（future = current）的结果；
- 报告 **pricing error（bp 或 $）** 与 **hedge P&L / hedge-error** 改进；
- 同时报告 short-ATM 误差改进 + no-arbitrage 满足率作为补充。

**产出**：Table RQ4（pricing error + hedge error，generated vs persistence）+ 事件级 pricing 图。执行落点见 §8.5。

### 7.6 Phase 5 — 受控 benchmark（按变量轴，取代 flat leaderboard）

修掉 §2.2/§4.2。**每张表只动一个变量**，全部在 7.0 协议下、同 test 集、同 eval space：

| 轴 | 对比 | 隔离的变量 | RQ |
| --- | --- | --- | --- |
| Axis 1 | no-text vs text | textual information | RQ1（Phase 1） |
| Axis 2 | BoW/sentiment vs LLM-embedding 或 lp/hd/concat/512/1024 | text 表示 | RQ2（Phase 2） |
| Axis 3 | vol-xlsx WGAN vs vol-regression-xlsx | adversarial vs deterministic | 方法贡献 |
| Axis 4 | **linear/ridge** vs VolGAN vs CNN vs FiLM | 架构容量（含 trivial 下界） | 模型选择 |
| Axis 5 | SVI vs SABR vs cubic（拟合质量，非预测） | surface representation 层 | Contribution 3 |

- **Axis 4 是"FiLM 最好"这个 claim 真正被赚到的地方**：同 text/split/metric/seed 的 head-to-head，配显著性。新增 **trivial linear/ridge baseline** 作下界，证明深度/对抗的必要性（§4.1）。
- **Axis 5 是 representation-level，不是 forecasting**：明确与 forecasting 表分开，避免 §4.3 的 Heston/SABR 过度承诺。

**产出**：每轴一张表，各自隔离一个 claim。

### 7.7 Phase 6 — 统计与 robustness（横切所有 Phase）

把 §2.4/§3.1 落地，作为每个主表的标准列/附录：
- 5-seed mean ± std；paired bootstrap CI；DM 检验；sample-level win rate；
- error by **maturity bucket / moneyness bucket**；
- **raw-IV 点上的二次评估**（对照 SVI 重建面，破循环论证）；
- announcement vs all-news 分组（与 Phase 3 共享）。

### 7.8 RQ → 实验 → 产出 映射表

| RQ | 主实验 | 主指标 | 产出 | Phase |
| --- | --- | --- | --- | --- |
| RQ1 predictive accuracy | no-text vs text（同架构） | **Δ_text** + CI + win rate | Table RQ1 | 1 |
| RQ2 text 表示 | BoW/sentiment vs LLM（A）或 LLM 内部（B） | test MAE 递进 | Table RQ2 | 2 |
| RQ3 intraday/announcement | event-study + horizon | 分组 Δ_text、跳变跟踪 | Table RQ3 + case 图 | 3 |
| RQ4 practical | pricing/hedging case（A）或定性（B） | pricing error / short-ATM | Table/图 RQ4 | 4 |
| benchmark | 5 条受控轴 | 各轴 test MAE + 显著性 | 5 张轴表 | 5 |

### 7.9 执行顺序与里程碑

```
Phase 0（split + 审计 + timing）           ← 必须最先，所有实验的地基
   ↓
Phase 1（RQ1 中心实验：Δ_text）            ← 决定整章叙事走 B1/B2/B3 哪条
   ↓
Phase 2（RQ2 决策 A/B）  Phase 3（RQ3 event-study）   ← 可并行
   ↓
Phase 4（RQ4 决策 A/B）
   ↓
Phase 5（受控 benchmark 5 轴）
   ↓
Phase 6（统计/robustness，横切回填各表）
```

- **最小可交付路径（MVP）**：Phase 0 → 1 → 3 → 6。这条就能支撑一篇诚实合格的章节（RQ1 + RQ3 + 显著性）。
- **完整路径**：再加 Phase 2(A) + Phase 4(A) + Phase 5，论文更强。

### 7.10 决策状态

| # | 决策 | 状态 |
| --- | --- | --- |
| 1 | RQ2 路线 | ✅ **A**：补 BoW/TF-IDF + Loughran-McDonald sentiment baseline |
| 2 | RQ4 路线 | ✅ **A**：真做 pricing / hedging case study |
| 3 | main 架构 | ✅ **FiLM WGAN 主线**，CNN WGAN 作复现 robustness |
| 4 | test 比例 | ⏳ 暂定 70/15/15（chronological），Phase 0 出样本量后再确认 |
| 5 | announcement 窗口 N | ⏳ 暂定 ±30 min，Phase 3 做敏感性 |

> 1/2/3 已定，可直接启动 Phase 0；4/5 是数值参数，在 Phase 0/3 落地时定即可，不阻塞开工。具体执行步骤见 §8。

---

## 8. 执行手册 / Execution Playbook（FiLM 主线，RQ2/RQ4 路线 A）

本节把 §7 的 Phase 落成**具体到文件/命令**的可执行步骤。已对照当前代码（`src/film_wgan/`、`configs/film_wgan/`、`scripts/film_wgan/`）。
**先看 §8.0 的代码改动——有两处是后面所有 Phase 的硬前置，不改无法做。**

### 8.0 必须先做的代码改动（3 处硬前置）

当前 standalone FiLM WGAN 的实现有两个缺口，直接挡住 §7 的中心实验，必须先补：

**(1) no-text 模式不存在 → 挡住 RQ1（P0，必做）**
- 现状：`src/film_wgan/data.py` 的 `_parse_embedding()`（约 L42–L52）只支持 `hd / lp / concat`，没有"关掉文本"的选项。
- 改法（最小侵入）：
  - 在 `_parse_embedding` 增加 `none` 分支，返回一个**固定维度的零向量**（维度对齐 lp，使 generator/critic 的 text 输入维度不变，保证 no-text 与 text 跑**同一架构**，只是 condition 信息为零）。
  - 同步：`FilmWGANTrainConfig.text_embedding_mode` 允许 `"none"`；当 `none` 时把 `normalize_text_embedding` 强制设 False（零向量不要再除以 std）。
  - 校验：`embedding_dim`（`data.py` 里由样本推断）在 `none` 下要等于 lp 维度，否则模型 text 分支 shape 不一致。
- 这样 RQ1 的 `no-text vs text` 就是**严格同架构**，Δ_text 才干净。

**(2) 只有 train/val 两分 → 挡住"test 报告"（P0，必做）**
- 现状：`create_train_val_bundle()`（`data.py` L291–L339）经 `_split_index()`（L269–L274）做 **2-way** 切分；`split_samples()`（L355–L366）只认 `train/val/all`。
- 改法：
  - 增 `test_ratio`（或直接 `val_ratio` + `test_ratio`），把 chronological 切成 **train / val / test**；normalization 仍**只用 train**（`_compute_normalization_stats` 当前就是只吃 train_items，保持）。
  - `split_samples` 增加 `"test"` 分支，供 generate_result / 评估在 test 上取数。
  - bundle 增加 `test_items`，供最终报告。
- 报告纪律：**train 训练、val 选 checkpoint（`checkpoint_metric=val_mae`）、test 只在最后评估一次**。

**(3) 评估指标补 raw-IV + 配对统计（P1，Phase 6 用）**
- 现状 metric 在重建 16×16 grid（log-IV delta）上算。补：在 raw filtered IV 点上的二次评估（解决 §3.1），以及把 per-sample 误差**落盘**（供 bootstrap/DM/win-rate，Phase 6）。
- 落点：FiLM 的评估在 `trainer.py` / `inference.py` / `short_atm_study.py`，per-sample 误差导出加在 `generate_result.py` 的 summary 里。

> CNN WGAN 作复现 robustness，同样的 (1)(2) 改动要在 `src/cnn_wgan/` 对应文件复制一遍。建议把 no-text/3-way-split 抽成**共享 util**，避免两套实现漂移。

### 8.1 Phase 0 执行（数据/样本/timing 固化）

1. **3-way split manifest**：用 §8.0(2) 改好的 split，跑一次 dry 导出 `train/val/test` 的 `sample_id` 列表与时间边界，存成 `data/processed/.../split_manifest.json`，之后所有实验固定引用它。
2. **event 日历**：新建 `data/reference/fomc_events.csv`（10 次 FFR hike 的 FOMC 公布 UTC 时刻 + 关键 Fed 讲话）。这是外部已知信息，手工整理即可。
3. **announcement 标注（无需改训练码）**：写一个分析脚本，读样本的 `news_timestamp_utc`，与 `fomc_events.csv` 比对，给每条样本打 `is_announcement_window`（|Δt| ≤ N min）。
4. **quality-filter announcement 审计**（解决 §3.2）：用样本 metadata（`pair_quality_label`、`current/target_weighted_iv_rmse`，见 `data.py` L243–L247）统计 announcement vs quiet 的 usable 通过率，出一张审计表。若 announcement 通过率明显低，按 §7.1 的备注处理。
5. **timing 陈述**：用 `current_snapshot_time_utc` / `target_snapshot_time_utc`（`data.py` L236–L237）核对 current ≤ news ≤ future 且 future−current=5min，写进 empirical setup 防 look-ahead。

**产出**：split_manifest、fomc_events.csv、announcement 标注列、quality 审计表、timing 说明。

### 8.2 Phase 1 执行（RQ1：Δ_text，中心实验）

1. **配置**：复制 `configs/film_wgan/train_lp_gen128_disc128.yaml` 成两个：
   - `train_film_text_lp.yaml`：`text_embedding_mode: lp`
   - `train_film_notext.yaml`：`text_embedding_mode: none`（用 §8.0(1)）
   - 两者**除 text_embedding_mode 外完全一致**，`checkpoint_metric: val_mae`（broad，按 §7.0 协议），加 `test_ratio`。
2. **5 seeds**：每个 config 跑 seed ∈ {41,42,43,44,45}：
   ```bash
   for s in 41 42 43 44 45; do
     python scripts/film_wgan/main.py train --config configs/film_wgan/train_film_text_lp.yaml --set seed=$s
     python scripts/film_wgan/main.py train --config configs/film_wgan/train_film_notext.yaml --set seed=$s
   done
   ```
3. **test 评估**：对每个 run 在 **test** 上 generate_result，导出 per-sample 误差。
4. **主结果**：`Δ_text = MAE_test(notext) − MAE_test(text)`，5-seed mean±std + paired bootstrap CI + DM + win rate（§8.0(3) 的落盘误差喂给统计脚本）。
5. **CNN 复现**：在 `cnn_wgan` 上重复 1–4，验证 Δ_text 的方向一致（robustness）。
6. **按 Δ_text 结果选叙事分支 B1/B2/B3（见 §7.2）**。

### 8.3 Phase 2 执行（RQ2 路线 A：traditional text baseline）

1. **特征构造**（新模块，如 `src/text_baselines/`）：
   - **BoW + TF-IDF**：从原始 `LP` 文本（news 工作簿里有原文；workbook 仅存 embedding 时需回源 `data/raw/text_embedding/`）构造 TF-IDF 向量，可 SVD 降维到与 lp 可比的维度。
   - **Loughran-McDonald sentiment**：用 LM 金融词典算 positive/negative/uncertainty 等计数特征。
2. **集成点**：在 merge 阶段或一个预处理步把这些特征写成与 `lp_embedding` 同构的列（如 `bow_embedding` / `sentiment_embedding`），这样 `_parse_embedding` 加两个 mode（`bow` / `sentiment`）即可复用**同一 FiLM-WGAN 下游**——保证"只换 text 表示，下游不变"。
3. **跑同协议**：与 Phase 1 相同 split/seed/test。
4. **Table RQ2-A**：`none < bow < sentiment < lp(LLM)` 的 test MAE 递进；并附 LLM 内部 `lp/hd/concat`、`512/1024` 子表。

> 注意：512D vs 1024D 需要重新生成对应维度的 OpenAI embedding（`text-embedding-3-large` 支持 `dimensions` 参数）。若只有 1024D 已生成，512D 要么重生成、要么 PCA 降维并在文中说明。

### 8.4 Phase 3 执行（RQ3：event-study + horizon）

1. **event-study（纯后处理，不重训）**：用 Phase 0 的 `is_announcement_window` 把 **test** 切成 announcement vs quiet 两组，分别算 {persistence, no-text, text} 的误差与 **Δ_text**。
2. **核心假设检验**：`Δ_text(announcement) > Δ_text(quiet)`，做分组配对检验。
3. **case study 图**：选 2–3 个 FOMC（如 2022-11-02），用 generate_result 画 announcement 前后 IVS 跳变与模型跟踪（FiLM 已有 plotting/`short_atm_study.py` 可复用）。
4. **horizon sensitivity**：需要 horizon≠5min 的 merged workbook（offset_minutes 在 surface 生成/merge 阶段控制）。生成 {1,5,15,30}min 的 pair workbook，各重训主对比，画 horizon 曲线。

> horizon 实验成本最高（要重生成 workbook + 重训）。MVP 阶段可只做 5min；horizon 作为 Phase 5 之后的增强。

### 8.5 Phase 4 执行（RQ4 路线 A：pricing / hedging）

1. **新分析脚本**（如 `scripts/pricing_case/main.py`），输入：Phase 1 text 模型在 announcement 事件上的 **generated future surface** 与 **persistence surface** 与 **real future surface**。
2. **定价**：用 `src/quantlib`（已有 Black-76 / IV / vega 逻辑）对选定 OTM 合约，由各 surface 的 IV → 反算 option price；对比 generated vs persistence 相对 real 的 **pricing error（bp 或 $）**。
3. **hedging**：从 surface 取 ATM/near-ATM 的 **Vega / Gamma**，构造简单 delta-vega hedge，比较用 generated vs persistence surface 的 **hedge error / P&L**。
4. **产出**：Table RQ4 + 事件级图。范围控制在几个代表性 FOMC 即可，重在"测量而非断言"。

### 8.6 Phase 5 执行（受控 benchmark 5 轴）

全部在 §7.0 协议、同 split/seed/test/eval-space 下：
- **Axis 1**（text）：复用 Phase 1。
- **Axis 2**（text 表示）：复用 Phase 2/3。
- **Axis 3**（adversarial vs deterministic）：`scripts/train/main.py vol-xlsx` vs `vol-regression-xlsx`，同 `merged_vol.xlsx`。
- **Axis 4**（架构 + trivial 下界）：**新增 linear/ridge baseline**（flatten current surface + text → future，sklearn 即可）vs `volgan` vs `cnn_wgan` vs `film_wgan`，同 text(lp)/split/seed → 这是"FiLM 最好"被真正赚到的 head-to-head。
- **Axis 5**（representation 拟合，非预测）：用 `scripts/generate_surface` 跑 `svi/sabr/cubic`，比 surface 拟合 RMSE（representation-level，单独成表，**不与 forecasting 表混**）。

### 8.7 Phase 6 执行（统计/robustness 横切回填）

- 写一个统一统计脚本，吃各 run 的 per-sample test 误差（§8.0(3) 落盘），产出：5-seed mean±std、paired bootstrap CI、DM、win rate、按 maturity/moneyness bucket 的误差、raw-IV 二次评估。
- 每张主表统一加这几列，保证所有"A>B"都带显著性。

### 8.8 改动/新增文件清单

| 类型 | 文件 | 作用 | Phase |
| --- | --- | --- | --- |
| 改 | `src/film_wgan/data.py`（`_parse_embedding`、`_split_index`、`create_train_val_bundle`、`split_samples`） | no-text 模式 + 3-way split | 8.0 |
| 改 | `src/film_wgan/config.py` | `text_embedding_mode="none"`、`test_ratio` 字段 | 8.0 |
| 改 | `src/cnn_wgan/*`（对应文件） | 同步 no-text + 3-way split（建议抽共享 util） | 8.0 |
| 改 | `src/film_wgan/generate_result.py` / `inference.py` | per-sample 误差落盘 + raw-IV 评估 | 8.0(3) |
| 新 | `configs/film_wgan/train_film_text_lp.yaml`、`train_film_notext.yaml` | RQ1 双条件 | 8.2 |
| 新 | `data/reference/fomc_events.csv` | event 日历 | 8.1 |
| 新 | `scripts/analyze_announcement/`（或并入 analyze_error） | announcement 标注 + quality 审计 + event-study | 8.1/8.4 |
| 新 | `src/text_baselines/`（BoW/TF-IDF + LM sentiment）+ merge 集成 | RQ2 路线 A | 8.3 |
| 新 | `scripts/pricing_case/` | RQ4 pricing/hedging | 8.5 |
| 新 | linear/ridge baseline（小脚本） | Axis 4 下界 | 8.6 |
| 新 | 统一统计脚本（bootstrap/DM/win-rate） | Phase 6 | 8.7 |

### 8.9 建议起步顺序

```
8.0(1) no-text 模式  +  8.0(2) 3-way split      ← 先改这两处，是一切的前置
   ↓
8.1 Phase 0（split manifest + event 日历 + quality 审计）
   ↓
8.2 Phase 1（RQ1 Δ_text，FiLM；CNN 复现）        ← 中心结果，决定叙事分支
   ↓
8.3 / 8.4 并行（RQ2 baseline / RQ3 event-study）
   ↓
8.5（RQ4 pricing）→ 8.6（5 轴 benchmark）→ 8.7（统计回填）
```

> 下一步若要我动手，建议从 **§8.0(1)+(2) 的代码改动**开始（no-text 模式 + 3-way split），这是解锁全部实验的前置，工作量小、收益最大。

---

**一句话总结**：计划的**数据语义和工程链路是清楚的**，但**实验逻辑有一个中心缺陷和三个方法论缺陷**——
中心缺陷是"用 gap-vs-persistence 去支撑 text-usefulness"（隔离不出文本贡献）；三个方法论缺陷是
"无独立 test set 的 selection bias、跨 family 的 confounded 排名、缺统计显著性"。
再加上 RQ2 概念偷换、RQ3 缺 event-study/timing、RQ4 无真实 pricing 实验、以及 quality filter 与 RQ3 的样本冲突。
**这些都是设计层面、现在就能在计划里修正的逻辑问题**，修好后再落成 chapter3 实证，结论才经得起答辩质询。

最后整理日期：2026-06-11
