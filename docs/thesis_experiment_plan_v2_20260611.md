# 论文第三章 修订版详细实验计划 (v2)

**创建日期：2026-06-11**

本文档是博士论文第三章 *Volatility Surface Prediction with Textual Information* 的**修订版实验计划**，
面向执行。它取代旧 `docs/thesis_experiment_plan_status.md` 中的评测设计部分，依据 `docs/chapter3_review_suggestions.md`
的逻辑评审结论重写，并对照了当前代码（`src/film_wgan/`、`src/cnn_wgan/`、`configs/`、`scripts/`）。

执行完本计划后，按 Phase 顺序填充 `chapter3.tex` 的 Empirical Result / Main Results / Robustness / Conclusion。

---

## 0. 为什么需要 v2（一句话）

旧计划的中心缺陷：用 `val_mae_gap_vs_current`（= 模型 vs persistence）去支撑"text 有用"，
但这个指标根本隔离不出文本的贡献——一个**没有文本**的模型只要学到 surface 动力学也能 beat persistence。
v2 把中心实验改成 **Δ_text =（no-text 误差）−（text 误差）**，并补齐 test set、显著性检验、受控对比，
使每个 RQ 都能在不引入 confound 的前提下被回答。

---

## 1. 锁定决策

| # | 决策项 | 选定 |
| --- | --- | --- |
| 1 | RQ2 路线 | **A**：实现 n-gram frequency BoW 与 Loughran-McDonald (LM) sentiment 两个 traditional text baseline，喂同一下游 |
| 2 | RQ4 路线 | **A**：真做 option pricing / hedging case study（用 generated surface 定价/对冲，对比 persistence） |
| 3 | 主架构 | **FiLM WGAN 主线**，CNN WGAN 作复现 robustness |
| 4 | 数据切分 | train/val/test = **70/15/15**（chronological，按 `news_timestamp_utc`），Phase 0 出样本量后微调 |
| 5 | announcement 窗口 | **±30 min** 起步，Phase 3 做敏感性 |

---

## 2. 设计原则（贯穿全计划）

1. **每个对比只动一个变量**（one variable per comparison）——否则结论无法归因。
2. **所有进论文的主数字来自 test set**——val 只用于选模型，test 只碰一次。
3. **任何"A 优于 B"都带显著性**——paired bootstrap CI + Diebold-Mariano (DM) + sample-level win rate。
4. **三类 metric 永不混排**——broad `val_mae`、short-ATM、legacy `val_recon` 分表，各自标注。
5. **预设诚实叙事分支**——text 显著 / 仅 announcement 显著 / 不显著，三套写法都先准备好。

---

## 3. 全局实验协议（所有 Phase 共用，先定死）

| 项 | 规定 |
| --- | --- |
| **数据切分** | chronological train/val/test = 70/15/15，按 `news_timestamp_utc` 排序后切，不 shuffle |
| **normalization** | 只在 **train** 上拟合（current/delta/text 统计量），val/test 复用 |
| **eval space** | 所有 forecasting 模型评估在**同一重建 16×16 IV grid**、**同一交集 usable 样本**、**同一 metric**；SVI regressor 预测参数先重建成同 grid 再比 |
| **prediction target** | primary = SVI 重建 16×16 surface（log-IV delta）；secondary = raw filtered IV 点（robustness，破循环论证） |
| **checkpoint 选择** | 预注册**单一** metric = `val_mae`（broad），固定 epoch 预算，不事后换 metric |
| **seed** | 每 config 跑 5 seeds（41–45），test 上报 mean ± std |
| **显著性** | per-sample test 误差落盘 → paired bootstrap CI（≥10k）+ DM + win rate |
| **pre-registration** | 看 test 前冻结 split / 超参 / selection metric / 假设，存进 run artifact |

> **现有 53 个 saved run 全部降级为 exploratory 初筛**，仅用于挑出值得正式 head-to-head 的 family（FiLM / CNN / VolGAN），**不进论文主结论表**。

---

## 4. RQ → 实验 → 产出 映射总表

| RQ | 主实验 | 主指标 | 产出 | Phase |
| --- | --- | --- | --- | --- |
| **RQ1** predictive accuracy | no-text vs text（同 FiLM 架构） | **Δ_text** + CI + win rate | Table RQ1 | 1 |
| **RQ2** LLM vs traditional | none → BoW → LM-sentiment → LLM-embedding（同下游） | test MAE 递进 + 显著性 | Table RQ2-A | 2 |
| **RQ3** intraday/announcement | event-study（announcement vs quiet）+ horizon | 分组 Δ_text、IVS 跳变跟踪 | Table RQ3 + case 图 | 3 |
| **RQ4** practical | pricing/hedging case（generated vs persistence） | pricing error(bp/$)、hedge error | Table RQ4 + 事件图 | 4 |
| benchmark | 5 条受控变量轴 | 各轴 test MAE + 显著性 | 5 张轴表 | 5 |

---

## 5. 必做的代码前置（3 处，挡住后续所有 Phase）

> 这是看代码后确认的真实障碍，不改无法做 RQ1 与 test 报告。

### 5.1 no-text 模式不存在 → 挡住 RQ1（P0）
- **现状**：`src/film_wgan/data.py` 的 `_parse_embedding()`（约 L42–L52）只支持 `hd / lp / concat`。
- **改法**：
  - 增 `none` 分支，返回**固定维度零向量**（维度对齐 lp，使 generator/critic 的 text 输入维度不变 → no-text 与 text 跑**严格同一架构**，只是 condition 为零）。
  - `FilmWGANTrainConfig.text_embedding_mode` 允许 `"none"`；为 `none` 时强制 `normalize_text_embedding=False`（零向量不除 std）。
  - 校验 `embedding_dim`（由样本推断）在 `none` 下 = lp 维度。

### 5.2 只有 train/val 两分 → 挡住 test 报告（P0）
- **现状**：`create_train_val_bundle()`（`data.py` L291–L339）经 `_split_index()`（L269–L274）做 2-way；`split_samples()`（L355–L366）只认 `train/val/all`。
- **改法**：
  - 增 `test_ratio`（或 `val_ratio`+`test_ratio`），chronological 切 train/val/test；normalization 仍只用 train（`_compute_normalization_stats` 当前即只吃 train_items，保持）。
  - `split_samples` 增 `"test"` 分支；bundle 增 `test_items`。
- **纪律**：train 训练 / val 选 checkpoint / test 仅最终评估一次。

### 5.3 评估补 raw-IV + per-sample 误差落盘 → Phase 6 用（P1）
- 现 metric 在重建 grid 上算。补：raw filtered IV 点上的二次评估；per-sample test 误差导出（喂 bootstrap/DM/win-rate）。
- **落点**：`src/film_wgan/generate_result.py` / `inference.py` / `short_atm_study.py` 的 summary。

> CNN WGAN 同样需要 5.1+5.2。建议把 no-text / 3-way-split 抽成**共享 util**，避免两套实现漂移。

---

## 6. Phase 0 — 数据 / 样本 / timing 固化（前置，无模型）

| 步 | 内容 | 说明 |
| --- | --- | --- |
| 0.1 | 生成 3-way **split manifest**（train/val/test 的 `sample_id` + 时间边界） | 存 `data/processed/.../split_manifest.json`，之后固定引用 |
| 0.2 | 整理 **event 日历** `data/reference/fomc_events.csv` | 10 次 FFR hike 的 FOMC 公布 UTC 时刻 + 关键 Fed 讲话，外部已知信息 |
| 0.3 | **announcement 标注**（后处理，不改训练码） | 按样本 `news_timestamp_utc` 与 event 比对，打 `is_announcement_window`（\|Δt\| ≤ 30min） |
| 0.4 | **quality-filter announcement 审计** | 用 metadata（`pair_quality_label`、`current/target_weighted_iv_rmse`，`data.py` L243–247）统计 announcement vs quiet 的 usable 通过率 |
| 0.5 | **timing/causality 陈述** | 用 `current_snapshot_time_utc`/`target_snapshot_time_utc`（L236–237）核对 current ≤ news ≤ future 且 future−current=5min，写进 empirical setup 防 look-ahead |

**关键检查（解决评审 §3.2）**：若 0.4 显示 announcement 样本通过率明显低于 quiet，必须在此处决定补救（放宽 announcement 窗口 fit 阈值并单独标注 / 或在 limitation 明确 RQ3 样本偏向安静时段）。

**产出**：split_manifest、fomc_events.csv、announcement 标注列、quality 审计表、timing 说明。

---

## 7. Phase 1 — RQ1：Δ_text 中心实验

**目标**：在严格同架构下隔离文本贡献。这是全章的中心结果。

1. **配置**：复制 `configs/film_wgan/train_lp_gen128_disc128.yaml` 为两份，**除 text_embedding_mode 外完全一致**：
   - `configs/film_wgan/train_film_text_lp.yaml`：`text_embedding_mode: lp`
   - `configs/film_wgan/train_film_notext.yaml`：`text_embedding_mode: none`
   - 两者均设 `checkpoint_metric: val_mae`、加 `test_ratio`。
2. **5 seeds 训练**：
   ```bash
   for s in 41 42 43 44 45; do
     python scripts/film_wgan/main.py train --config configs/film_wgan/train_film_text_lp.yaml --set seed=$s
     python scripts/film_wgan/main.py train --config configs/film_wgan/train_film_notext.yaml --set seed=$s
   done
   ```
3. **test 评估**：每个 run 在 **test** 上 generate_result，导出 per-sample 误差。
4. **主结果**：
   ```
   Δ_text = MAE_test(no-text) − MAE_test(text)
   ```
   报 5-seed mean±std + paired bootstrap CI + DM + win rate。Δ_text 显著 >0 才支持"text 有用"。
   辅助列：各自 `gap_vs_current`（只说明"模型 vs persistence"，不当 text 证据）。
5. **CNN 复现**：在 `cnn_wgan` 重复 1–4，验证 Δ_text 方向一致。
6. **据结果选叙事分支**（见 §16）。

**产出**：Table RQ1（{FiLM, CNN} × {no-text, text}：test MAE、Δ_text、CI、p、win rate）。

---

## 8. Phase 2 — RQ2（路线 A）：text representation baseline

**目标**：回答"LLM embedding vs sparse text / LLM sentiment representation 的信息含量"，让 RQ2 真正成立。

1. **特征构造**（新模块 `src/text_baselines/`）：
   - **n-gram frequency BoW**：参考 Manela and Moreira 风格的传统文本表示，从原始 `LP` 文本（回源 `data/raw/text_embedding/`）构造 unigram/bigram frequency features，作为迁移到本文数据上的 sparse text baseline。
   - **Sun-style LLaMA sentiment**：参考 Sun (2026) 的 zero-shot LLaMA 3 multi-dimensional sentiment decomposition，按 `macroeconomic_uncertainty`、`institutional_action`、`risk_off_intensity` 三个维度给每条新闻打分。
2. **集成点（保证"只换 text 表示，下游不变"）**：在 merge / 预处理阶段把这些特征写成与 `lp_embedding` 同构的列（如 `bow_embedding` / `sentiment_embedding`）→ `_parse_embedding` 加 `bow` / `sentiment` 两个 mode → 复用同一 FiLM-WGAN 下游。
3. **同协议跑**：与 Phase 1 相同 split / seed / test。
4. **产出 Table RQ2-A**：`none < bow < sentiment < lp(LLM)` 的 test MAE 递进（期望单调）；附 LLM 内部子表 `lp/hd/concat`、`512D/1024D`。

> **512D vs 1024D 提醒**：`text-embedding-3-large` 支持 `dimensions` 参数。若只有 1024D 已生成，512D 需重新生成或 PCA 降维并在文中说明。
>
> **文献方法定位**：BoW baseline 只复用 Manela and Moreira 式的 n-gram / bag-of-words frequency text representation。它不复现原论文的 NVIX/SVR 下游，也不把原论文结果与本文结果直接比较；比较发生在本文数据、本文 split、同一 FiLM-WGAN 下游和同一 metrics 下。

---

## 9. Phase 3 — RQ3：event-study + horizon sensitivity

**目标**：证明模型捕捉 announcement 后的 intraday 反应，并堵 look-ahead 质疑。

1. **event-study（纯后处理，不重训）**：用 Phase 0 的 `is_announcement_window` 把 **test** 切成 announcement vs quiet，分别算 {persistence, no-text, text} 的误差与 **Δ_text**。
2. **核心假设**：`Δ_text(announcement) > Δ_text(quiet)`，做分组配对检验。**这是即使 broad Δ_text 偏小时，仍能证明 text 价值的最干净位置（对应分支 B2）**。
3. **case study 图**：选 2–3 个 FOMC（如 2022-11-02），用 generate_result + `short_atm_study.py` 画 announcement 前后 IVS 跳变与模型跟踪。
4. **horizon sensitivity**：生成 {1,5,15,30}min 的 pair workbook（offset 在 surface 生成/merge 阶段控制），各重训主对比，画 horizon 曲线。

> **成本提示**：horizon 实验需重生成 workbook + 重训，成本最高。MVP 阶段只做 5min；horizon 作为 Phase 5 之后的增强。

**产出**：Table RQ3（announcement vs quiet × {persistence, no-text, text}，含分组 Δ_text）+ horizon 曲线 + event case 图。

---

## 10. Phase 4 — RQ4（路线 A）：pricing / hedging case study

**目标**：用真实测量（而非断言）说明预测改进的实际意义。

1. **新分析脚本** `scripts/pricing_case/main.py`，输入 Phase 1 text 模型在 announcement 事件上的 **generated future surface**、**persistence surface**、**real future surface**。
2. **pricing**：用 `src/quantlib`（已有 Black-76 / IV / vega）对选定 OTM 合约，由各 surface 的 IV → 反算 price；比较 generated vs persistence 相对 real 的 **pricing error（bp 或 $）**。
3. **hedging**：从 surface 取 ATM/near-ATM 的 **Vega / Gamma**，构造简单 delta-vega hedge，比较用 generated vs persistence surface 的 **hedge error / P&L**。
4. **范围**：几个代表性 FOMC 即可，重在"测量"。

**产出**：Table RQ4（pricing error + hedge error，generated vs persistence）+ 事件级 pricing 图。

---

## 11. Phase 5 — 受控 benchmark（5 条变量轴，取代 flat leaderboard）

全部在 §3 协议、同 split/seed/test/eval-space 下，**每张表只动一个变量**：

| 轴 | 对比 | 隔离变量 | 落点 |
| --- | --- | --- | --- |
| Axis 1 | no-text vs text | textual information | 复用 Phase 1 |
| Axis 2 | none/bow/sentiment/LLM + lp/hd/concat/512/1024 | text 表示 | 复用 Phase 2 |
| Axis 3 | `vol-xlsx` WGAN vs `vol-regression-xlsx` | adversarial vs deterministic | `scripts/train/main.py`，同 `merged_vol.xlsx` |
| Axis 4 | **linear/ridge** vs VolGAN vs CNN vs FiLM | 架构容量（含 trivial 下界） | 新增 sklearn linear baseline + 各 family，同 text(lp)/split/seed |
| Axis 5 | SVI vs SABR vs cubic（拟合 RMSE） | surface representation 层（非预测） | `scripts/generate_surface`，单独成表 |

- **Axis 4 是"FiLM 最好"被真正赚到的 head-to-head**；trivial linear baseline 作下界，证明深度/对抗的必要性。
- **Axis 5 是 representation-level，不是 forecasting**，单独成表，**绝不与 forecasting 表混**（避免 Heston/SABR 过度承诺）。

---

## 12. Phase 6 — 统计 / robustness（横切回填所有主表）

统一统计脚本，吃各 run 的 per-sample test 误差（§5.3 落盘），产出并回填每张主表：
- 5-seed mean ± std；paired bootstrap CI；DM 检验；sample-level win rate；
- error by **maturity bucket / moneyness bucket**；
- **raw-IV 点二次评估**（对照 SVI 重建面）；
- announcement vs all-news 分组（与 Phase 3 共享）。

---

## 13. 产物与目录约定

每个进论文的 run 必须保存：
```
outputs/training/<family>/<dataset>/<run_ts>/
  metrics/training_resolved_config.yaml   # 含冻结的 pre-registration
  metrics/training_metrics.csv / .json
  metrics/best_checkpoint.json            # val 上选定
  generate_result/<ckpt>/summary.csv      # 含 per-sample test 误差
  generate_result/<ckpt>/samples/*.json
  generate_result/<ckpt>/plots/*.png
```
引用时一律给：resolved config + best checkpoint metadata + metrics + per-sample 误差 + plots。

---

## 14. 文件改动 / 新增清单

| 类型 | 文件 | 作用 | Phase |
| --- | --- | --- | --- |
| 改 | `src/film_wgan/data.py`（`_parse_embedding` / `_split_index` / `create_train_val_bundle` / `split_samples`） | no-text 模式 + 3-way split | 5.1/5.2 |
| 改 | `src/film_wgan/config.py` | `text_embedding_mode="none"`、`test_ratio` 字段 | 5.1/5.2 |
| 改 | `src/cnn_wgan/*` 对应文件 | 同步 no-text + 3-way split（建议共享 util） | 5.1/5.2 |
| 改 | `src/film_wgan/generate_result.py` / `inference.py` | per-sample 误差落盘 + raw-IV 评估 | 5.3 |
| 新 | `configs/film_wgan/train_film_text_lp.yaml`、`train_film_notext.yaml` | RQ1 双条件 | 7 |
| 新 | `data/reference/fomc_events.csv` | event 日历 | 6 |
| 新 | `scripts/analyze_announcement/`（或并入 `analyze_error`） | announcement 标注 + quality 审计 + event-study | 6/9 |
| 新 | `src/text_baselines/`（n-gram frequency BoW + LM sentiment）+ merge 集成 | RQ2 路线 A | 8 |
| 新 | `scripts/pricing_case/` | RQ4 pricing/hedging | 10 |
| 新 | linear/ridge baseline 小脚本 | Axis 4 下界 | 11 |
| 新 | 统一统计脚本（bootstrap/DM/win-rate） | Phase 6 | 12 |

---

## 15. 执行顺序与里程碑

```
代码前置：5.1 no-text 模式  +  5.2 3-way split          ← 先改这两处，解锁一切
   ↓
Phase 0（split manifest + event 日历 + quality 审计 + timing）
   ↓
Phase 1（RQ1 Δ_text，FiLM；CNN 复现）                    ← 中心结果，决定叙事分支
   ↓
Phase 2（RQ2 baseline）  ‖  Phase 3（RQ3 event-study）   ← 可并行
   ↓
Phase 4（RQ4 pricing）→ Phase 5（5 轴 benchmark）→ Phase 6（统计回填）
```

- **MVP（最小可交付）路径**：代码前置 → Phase 0 → Phase 1 → Phase 3 → Phase 6。这条即可支撑一篇诚实合格章节（RQ1 + RQ3 + 显著性）。
- **完整路径**：再加 Phase 2 + Phase 4 + Phase 5（兑现 RQ2/RQ4 路线 A 与全 benchmark）。

**建议起步**：先做 §5.1 + §5.2 的代码改动（no-text 模式 + 3-way split），工作量小、解锁全部实验。

---

## 16. 风险与诚实叙事分支（Phase 1 出结果后据此走）

| 分支 | 触发条件 | 章节叙事 |
| --- | --- | --- |
| **B1** | Δ_text 在全样本显著 >0 | text 直接支撑 RQ1，主线成立 |
| **B2** | Δ_text 仅在 announcement 子集显著（Phase 3） | claim 改为"text 价值集中在 event 时段"，仍是有力结论 |
| **B3** | Δ_text 不显著 | 诚实报 null，contribution 重心移到 architecture(FiLM)/representation；RQ1 写"在本数据与 5-min horizon 下 text 未带来显著 broad 提升"——这是诚实的博士结论，不是失败 |

> onboarding 已记录 "no-text/current-surface baseline 很强"，**B2/B3 的概率不低，现在就把三套写法备好**，避免结果出来后被动。

---

## 17. 待定数值参数（不阻塞开工，落地时定）

- test 比例：暂定 70/15/15，Phase 0 出样本量后确认 test 是否够支撑显著性。
- announcement 窗口 N：暂定 ±30min，Phase 3 做敏感性。
- seeds：暂定 5 个（41–45），若方差大可增。
- BoW / embedding 目标维度：Phase 2 落地时定。

---

**最后整理日期：2026-06-11**
