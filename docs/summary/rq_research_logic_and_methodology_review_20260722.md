# RQ Research Logic and Methodology Review

> Review timestamp: 2026-07-22 13:30:46 UTC  
> Review scope: `docs/chapter3.tex`, RQ1/RQ2/RQ3 experiment documentation, FiLM-WGAN data/training/evaluation code, SVI/raw-vol inputs, comparison procedures, and relevant literature.

## Overall Assessment

论文主线具有明确研究价值：把高频 Treasury option IVS forecasting、金融文本表征和新闻事件反应结合起来，交叉点相对少见。既有研究分别证明了 IVS 动态可预测、文本包含波动率信息、宏观新闻会造成 Treasury 市场的高频跳跃，但尚不足以覆盖当前研究的完整问题。[Cont and da Fonseca](https://www.maths.ox.ac.uk/node/29718)、[Medvedev and Wang](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.22302)、[Manela and Moreira](https://ideas.repec.org/a/eee/jfinec/v123y2017i1p137-162.html)、[Bollerslev et al.](https://scholars.duke.edu/individual/pub644374)

但是，当前结果还不能作为论文最终实证结论。研究问题本身成立，模型框架也有合理基础，但数据切分、模型选择和统计推断存在会直接影响结论有效性的重大问题。

## Critical Findings

### 1. Current `eval` Split Is Not Chronological

这是目前最严重的问题。

Loader 没有先按时间排序，而是直接按 workbook 行号切分前 80% 和后 20%：

```text
src/film_wgan/data.py:312
```

对 SVI workbook 的直接审计结果为：

```text
total rows                         = 3711
backward timestamp steps           = 1722
train time range                   = 2022-01 to 2023-12
eval time range                    = 2022-12 to 2023-12
surface pairs shared across split  = 6
```

raw-vol workbook 同样没有按时间排列，并且也有 6 个 current/target surface pairs 同时出现在 train 和 eval。

因此，`docs/comparison_test_calculations.md` 中将 evaluation split 描述为 chronological non-leakage split 的表述目前不成立。现有结果不属于严格的 out-of-time evaluation。

### 2. The Same Eval Data Is Used for Selection and Inference

当前 checkpoint selection 和 early stopping 都使用 validation metrics：

```text
configs/film_wgan/train_rq2_multiseed_textbase.yaml:48-54
```

随后，又在同一批 743 个 eval rows 上完成：

```text
1. Select the lowest-MAE seed for each model/metric.
2. Compare selected seeds.
3. Calculate paired p-values on the same rows.
```

这个数据集只能称为 validation set，不能称为 untouched test set。反复使用同一批数据进行模型选择和显著性检验会产生乐观的 selection bias。[Cawley and Talbot](https://jmlr.org/papers/v11/cawley10a.html)、[White](https://doi.org/10.1111/1468-0262.00152)

因此，当前 best-seed comparison 只能作为 exploratory/supplementary analysis，不能作为论文主结论。

### 3. Row-Level T-Tests and Bootstrap Ignore Dependence

3711 行训练数据只对应 2913 个唯一新闻时点和 surface pairs：

```text
duplicate rows       = 798
maximum multiplicity = 7
```

多个新闻 article rows 可能共享完全相同的 current and target surfaces。当前 paired t-test 和 bootstrap 将每一行视为独立 observation，并按 row-level IID resampling。这通常会低估标准误并夸大显著性。

应当：

```text
1. Keep every identical surface pair in the same split.
2. Aggregate or cluster inference by unique surface timestamp.
3. Use trading-day/time-block stationary bootstrap or HAC variance.
4. Use a Diebold-Mariano-style predictive accuracy test where appropriate.
```

参考：[Diebold and Mariano](https://www.nber.org/papers/t0169)、[Politis and Romano](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870)

### 4. BoW Vocabulary Uses Full-Sample News

当前 RQ2 feature script 先读取完整新闻 workbook，再使用所有新闻文本建立 BoW vocabulary：

```text
scripts/generate_rq2_text_features.py:157-160
src/bow/features.py:74-80
```

这没有使用 volatility target，但利用了未来时期的文本词频和 vocabulary distribution，属于 transductive preprocessing leakage。

严格实验应当：

```text
fit vocabulary on training-period news only
freeze vocabulary
transform validation/test news without refitting
```

### 5. Current Evaluation Does Not Establish the Value of a Generative WGAN

当前配置中：

```text
lambda_adv       = 0.1
lambda_recon     = 20.0
lambda_atm_short = 30.0
eval_mc_samples  = 32
aggregation      = arbitrage-weighted mean
```

训练主要由 reconstruction 和 short-ATM point losses 驱动，最终结果又以生成场景的 weighted mean 计算 MAE。这可以验证 hybrid point forecaster，但尚不能证明 WGAN 学到了有效的 conditional predictive distribution。

如果论文保留 generative-model contribution，需要增加：

```text
probabilistic calibration
prediction interval coverage
energy score or another multivariate proper score
scenario diversity and dependence structure
deterministic FiLM-CNN ablation with lambda_adv=0
```

VolGAN 类研究重点评估生成场景、联合分布与经济用途，而不仅是生成均值 MAE。[VolGAN](https://pmc.ncbi.nlm.nih.gov/articles/PMC13060012/)、[Gneiting and Raftery](https://doi.org/10.1198/016214506000001437)

### 6. RQ3 Quiet Evaluation Is an OOD Robustness Diagnostic

当前 RQ3 news/quiet 设计已经在文档中承认：

```text
quiet text embedding = zero vector
text model was trained on news rows
no has_news mask is part of the model input
no retraining is performed
```

此外，raw zero vector 进入模型前还会使用 training text mean/std 做 normalization：

```text
src/film_wgan/data.py:199-205
```

所以 text model 实际看到的不是中性零条件，而是一个偏离训练分布的 normalized vector。

Nakamura and Steinsson 的 30-minute window 用于识别 scheduled monetary-policy surprises，不能把一般的 all-news versus quiet comparison 自动转化为 causal event study。[Nakamura and Steinsson](https://ideas.repec.org/a/oup/qjecon/v133y2018i3p1283-1330..html)

因此应将当前 RQ3 表述为：

```text
news-regime predictive contrast
out-of-distribution robustness diagnostic
not a causal news-shock estimate
```

### 7. RQ4 Currently Has No Empirical Test

RQ4 询问 IVS prediction 对 option pricing、hedging、risk management 和 high-frequency trading 的实际意义，但当前 empirical results 仍然只有空的 section structure：

```text
docs/chapter3.tex:40
docs/chapter3.tex:521-525
```

MAE 改善不能自动证明经济价值。要保留 RQ4，至少应增加：

```text
future option-price prediction error
delta/vega hedging P&L
transaction costs and bid-ask spreads
portfolio turnover
VaR/ES or predictive coverage
risk-adjusted economic value
```

否则应将 RQ4 从独立 research question 降为 Discussion/Practical Implications。

### 8. Technical Statements in the Chapter Need Correction

#### GAN/WGAN

Standard GAN 不是简单地最小化 KL divergence；在 optimal discriminator 条件下，其经典目标与 Jensen-Shannon divergence 相关。WGAN critic 也不输出 0/1 probability。

Relevant locations:

```text
docs/chapter3.tex:114
docs/chapter3.tex:258
```

参考：[Original GAN](https://arxiv.org/abs/1406.2661)、[WGAN](https://proceedings.mlr.press/v70/arjovsky17a.html)

#### Black-76 Formula and American Exercise

Chapter 中 Black-76 call/put 公式遗漏了 futures term 的 discount factor：

```text
docs/chapter3.tex:316-320
```

代码中的定价公式本身正确，但 surface generation 使用 `r=0`：

```text
src/wgan_option/surface_generation/backend/surface_cpu/all.py:47-55
```

论文还明确说明 Treasury options 是 American style，因此需要解释为什么使用 European Black-76 inversion，并提供 American-pricing 或 early-exercise robustness。[Black 1976](https://www.sciencedirect.com/science/article/pii/0304405x76900246/pdf)、[CME contract description](https://www.cmegroup.com/education/articles-and-reports/ultra-10-year-us-treasury-note-futures)

#### Arbitrage-Free Claims

当前实现是在有限 strike/maturity grid 上计算 soft calendar and butterfly penalties：

```text
src/film_wgan/arbitrage.py:23-44
```

因此论文不应写成 `ensure arbitrage-free`，更准确的说法是：

```text
penalizes sampled-grid static-arbitrage violations
```

真正的 arbitrage-free SVI parameterization 需要满足更严格的参数和全域条件。[Gatheral and Jacquier](https://ideas.repec.org/a/taf/quantf/v14y2014i1p59-71.html)

#### Missing Benchmarks and Data-Lineage Mismatch

论文声称将预测结果与 Heston、SABR 和 spline approaches 对比：

```text
docs/chapter3.tex:55
```

但当前结果没有对应的 forecasting benchmarks。SVI/raw interpolation 是 surface construction robustness，不等于 Heston/SABR forecasting benchmark。

论文样本期写为 2022-03-17 至 2023-07-27，而实际 SVI training workbook 覆盖：

```text
2022-01-27 to 2023-12-27
```

Relevant location:

```text
docs/chapter3.tex:287
```

## RQ-by-RQ Assessment

| RQ | Research value | Current methodology assessment |
|---|---|---|
| RQ1: text vs no-text | High. It directly tests incremental predictive content in financial text. | Conceptually correct, but requires a true chronological test and shuffled-text placebo. |
| RQ2: LP vs BoW vs sentiment | Medium-high. It can distinguish semantic representations from compressed/word-frequency signals. | Reasonable controlled framework, but BoW must be fitted on training text only and best-eval-seed inference must be removed. |
| RQ3: event/news robustness | Medium-high. High-frequency Treasury announcement effects have strong literature support. | Scheduled FOMC can support an event-study design; all-news/quiet is a non-causal robustness contrast. |
| RQ4: economic implications | Potentially valuable. | Not currently answered because no pricing, hedging, or risk-management experiment exists. |

## Recommended Final Research Design

### Data Split

```text
1. Parse and sort by current_snapshot_time_utc.
2. Group all rows sharing the same current/target surface pair.
3. Keep each group entirely within one split.
4. Use train/validation/test or rolling-origin evaluation.
5. Never use final test data for checkpoint, seed, prompt, metric, or window selection.
```

### Multi-Seed Evaluation

```text
report all pre-specified seeds
report mean and standard deviation across seeds
optionally report a pre-specified seed ensemble
do not choose the lowest-MAE seed on the final test
```

With five seeds, seed-level tests have low power. They should be combined with timestamp/day-clustered out-of-sample loss comparisons, not replaced by best-seed sample-level tests.

### Metrics and Multiple Testing

Pre-specify one primary metric before looking at final test results:

```text
surface_mae    = natural primary metric for full-IVS forecasting
short_atm_mae  = secondary metric, or primary only if the thesis explicitly focuses on short ATM
atm7_abs_err   = single-cell secondary diagnostic
```

For multiple models, metrics, windows, and event definitions, use Holm correction, false-discovery control, or a Model Confidence Set rather than selecting the most favorable p-value.[Hansen, Lunde and Nason](https://pure.au.dk/portal/en/publications/the-model-confidence-set/)

### RQ1

Recommended baselines:

```text
persistence/current surface
no-text FiLM model
shuffled or mismatched text placebo
deterministic FiLM-CNN with lambda_adv=0
simple PCA/VAR or comparable quantitative forecasting model
```

The main claim should be `incremental predictive content`, not a causal effect of text on volatility.

### RQ2

```text
fit BoW vocabulary on training-period news only
freeze vocabulary before validation/test transformation
freeze ChatGPT prompt/model/version before final test
compare all representations using the same downstream model and seeds
```

The three-dimensional ChatGPT sentiment vector padded to 1024 dimensions should be described as a compressed-information baseline, not as an equal-information-capacity embedding.

### RQ3

The primary event analysis should use out-of-sample predictions around pre-specified FOMC or macroeconomic announcements. The all-news/quiet analysis can be retained as robustness.

For a formal quiet comparison, retrain with both news and no-news samples and provide an explicit `has_news` indicator or learned missing-text token. Additional placebo tests should include shuffled text, same-day mismatched text, and lagged/lead text.

### RQ4

Either add a complete economic evaluation or remove RQ4 as a standalone research question. If retained, map predicted surfaces into American-option prices and evaluate hedging and risk-management outcomes after transaction costs.

## Final Conclusion

The research questions have genuine value, and the combination of five-minute Treasury-option IVS forecasting with financial text provides a defensible and potentially distinctive contribution.

The current implementation, however, should be treated as exploratory evidence rather than final thesis evidence. Before claiming that LP text significantly outperforms no-text, BoW, or sentiment, the experiments must be rerun after correcting:

```text
non-chronological splitting
train/eval surface-pair overlap
validation reuse for checkpoint/seed selection and p-values
row-level dependence in statistical inference
full-sample BoW vocabulary fitting
generative-model evaluation and claim alignment
```

After these corrections, RQ1-RQ3 can form a coherent and publishable empirical sequence:

```text
RQ1: Does text provide incremental out-of-sample predictive information?
RQ2: Which text representation preserves the most useful predictive information?
RQ3: Does that predictive value remain or increase in high-information event regimes?
```

RQ4 should only remain if statistical gains are connected to actual pricing, hedging, or risk-management outcomes.

---

## Revised RQ1-RQ3 Research Framework

> Revision timestamp: 2026-07-22 13:52:27 UTC  
> Revision decision: Remove RQ4 as a standalone research question and reorganize RQ1-RQ3 as a progressive research sequence centered on the `text embedding + FiLM-WGAN` methodological innovation.

### Core Research Positioning

The central research question should be unified as:

> **Can semantic information embedded in financial news improve the conditional forecasting of intraday Treasury-option implied volatility surfaces, and under what information environments is this improvement most pronounced?**

中文表述：

> 金融新闻中的语义信息能否提高美国国债期权隐含波动率曲面的日内预测能力，这种预测价值来自何种文本表征，并且在什么信息环境下最明显？

The entire study should be organized around one methodological innovation:

```text
Current IVS + Text Embedding + Random Noise
                    ↓
      Text-conditioned FiLM-WGAN
                    ↓
Conditional distribution of future IVS
```

Existing IVS forecasting research predominantly uses numerical variables and historical volatility surfaces. Studies such as [Medvedev and Wang](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.22302) demonstrate that deep-learning models can forecast IVS dynamics. The financial-text literature separately shows that news contains information about macroeconomic conditions and volatility, as documented by [Manela and Moreira](https://ideas.repec.org/a/eee/jfinec/v123y2017i1p137-162.html) and [Bybee et al.](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13377).

The proposed contribution lies at the intersection of these research streams: using semantic financial-news information to condition the intraday generative forecast distribution of a Treasury-option implied volatility surface.

### Progressive Logic Across the Three RQs

The three research questions should answer a progressive sequence:

```text
RQ1: Does text help?                  Existence of incremental value
RQ2: Which text representation helps? Representation mechanism
RQ3: When does text help most?         Boundary condition and robustness
```

The logical relationship is:

```text
RQ1 establishes whether textual information has incremental predictive value.
RQ2 identifies whether that value is specifically associated with semantic representation.
RQ3 examines whether the identified semantic value persists or becomes stronger in high-information regimes.
```

This produces a more coherent thesis narrative than treating the three questions as independent experiments.

### RQ1: Incremental Predictive Value

Recommended formulation:

> **RQ1: Does the proposed text-conditioned FiLM-WGAN improve out-of-sample intraday IVS forecasting relative to an otherwise identical no-text model and quantitative persistence benchmarks?**

中文：

> 在严格样本外预测中，相对于结构完全相同的 no-text 模型和 persistence baseline，加入文本嵌入的 FiLM-WGAN 能否提高日内 IVS 预测准确度？

RQ1 is the foundation of the study. Its purpose is to establish:

```text
text information adds incremental predictive value
```

The core comparison should contain:

```text
current/persistence baseline
no-text FiLM-WGAN
text-conditioned FiLM-WGAN
```

Two additional architecture ablations are recommended:

```text
text + deterministic FiLM-CNN with lambda_adv=0
text + WGAN using simple concatenation instead of FiLM
```

These comparisons distinguish three different questions:

```text
Does text add information?
Does FiLM conditioning improve the use of that information?
Does the WGAN component add distributional forecasting value?
```

Recommended hypothesis:

> **H1:** The text-conditioned FiLM-WGAN produces lower out-of-sample IVS forecast errors than the no-text model and persistence baseline.

Metrics should be fixed before inspecting the final test results:

```text
primary point metric       = surface_mae
secondary point metrics    = short_atm_mae, atm7_abs_err
probabilistic metrics      = energy score, interval coverage, calibration
financial consistency      = calendar/butterfly violation rate
```

If the analysis reports only weighted-mean MAE, it can demonstrate point-forecast effectiveness but cannot establish the value of the WGAN predictive distribution.

### RQ2: Representation Mechanism

Recommended formulation:

> **RQ2: Do dense semantic text embeddings provide greater predictive value for intraday IVS forecasting than lexical BoW features and compressed LLM-based sentiment scores under the same FiLM-WGAN architecture?**

中文：

> 在保持数据、模型结构、训练配置和随机种子一致的条件下，语义文本嵌入是否比 BoW log-count 和 ChatGPT sentiment scores 更有效地提取与 IVS 变化相关的信息？

RQ2 builds directly on RQ1:

```text
RQ1 confirms whether text helps.
RQ2 identifies which type of textual information drives that improvement.
```

The controlled comparison should include:

```text
no_text
LP semantic embedding
BoW unigram/bigram log-count
ChatGPT multidimensional sentiment scores
```

These representations correspond to different levels of textual information:

```text
BoW             = lexical occurrence information
LLM sentiment   = compressed theory-driven scores
LP embedding    = dense semantic and contextual representation
```

Recommended hypothesis:

> **H2:** Dense LP semantic embeddings outperform BoW and compressed sentiment features because they preserve contextual and narrative information relevant to IVS dynamics.

The following controls are required for a valid comparison:

```text
BoW vocabulary fitted on training-period text only
same grouped chronological train/validation/test split
same downstream architecture and training losses
same pre-specified seeds
no best-seed selection on validation or test
all-seed mean and uncertainty reported
```

The three-dimensional ChatGPT sentiment vector padded to 1024 dimensions should be described as a compressed-information baseline, not as a representation with the same effective information capacity as the LP embedding.

### RQ3: Information-Regime Robustness

Recommended formulation:

> **RQ3: Is the incremental predictive value of semantic text embeddings stronger during high-information event windows, particularly around scheduled policy announcements and news arrivals, than during quiet market periods?**

中文：

> LP semantic embedding 的增量预测价值是否在 FOMC 等高信息事件窗口和一般新闻到达时更强，而在无新闻 quiet periods 中减弱？

RQ3 follows from the first two questions:

```text
RQ1 establishes that text has predictive value.
RQ2 establishes that semantic representation is the preferred representation.
RQ3 examines when that semantic information is most valuable.
```

The central quantities should be defined as:

```text
text_advantage
= baseline_error - text_error

event_increment
= text_advantage(event) - text_advantage(quiet)
```

Interpretation:

```text
text_advantage > 0
=> the text model has lower error than the baseline

event_increment > 0
=> text has greater marginal predictive value in event periods
```

RQ3 should contain two layers.

#### Primary Analysis: Scheduled Policy Events

Use a pre-specified FOMC press-release calendar and strictly out-of-sample forecasts:

```text
scheduled FOMC event windows
pre-specified event horizon
out-of-sample predictions only
pre-specified text and no-text models
```

The model's original forecast horizon is five minutes, so the five-minute horizon should remain the primary forecasting definition. The 30-minute window used by Nakamura and Steinsson can be retained as policy-announcement robustness rather than replacing the model's main horizon.[Nakamura and Steinsson](https://ideas.repec.org/a/oup/qjecon/v133y2018i3p1283-1330..html)

#### Secondary Analysis: General News vs Quiet

```text
news_event = any usable news arrival
quiet      = no news within a pre-specified buffer
```

If this comparison is included in the main RQ3 evidence, the model should be retrained using both news and quiet samples and should receive an explicit missingness signal:

```text
has_news indicator
learned no-news embedding or missing-text token
```

The current design using a news-trained checkpoint and a normalized zero embedding should remain an OOD robustness diagnostic rather than formal event-versus-quiet evidence.

All RQ3 conclusions must use predictive rather than causal language:

```text
Appropriate:
semantic text has greater predictive value in high-information regimes

Not supported without causal identification:
text causes volatility changes
```

### Unified Methodological Innovation

The methodological contribution can be described through four components.

#### 1. Multimodal Forecasting

The model jointly uses the current IVS and a financial-news embedding instead of relying only on historical numerical market variables.

#### 2. FiLM Conditioning

Text is not only concatenated with surface features. It produces feature-wise scale and shift parameters that modulate the CNN representation of the current IVS.

#### 3. Conditional Generative Forecasting

WGAN-GP and latent noise are used to generate conditional future-IVS scenarios rather than only a single deterministic prediction.

#### 4. Finance-Informed Regularization

Calendar, butterfly, smoothness, reconstruction, and short-ATM penalties incorporate financial structure and thesis-specific forecast priorities into training.

The recommended contribution statement is:

> We propose a text-conditioned FiLM-WGAN framework for the conditional generative forecasting of intraday Treasury-option implied volatility surfaces.

The model should not be described as guaranteeing fully arbitrage-free surfaces. The current implementation penalizes static-arbitrage violations on the sampled strike/maturity grid.

### Revised Contributions

After removing RQ4, the paper's contributions can be presented as:

1. Propose a text-conditioned FiLM-WGAN for intraday Treasury-option IVS forecasting.
2. Test the incremental out-of-sample predictive value of financial text relative to quantitative-only models.
3. Distinguish the predictive content of semantic embeddings, BoW log-count features, and ChatGPT sentiment scores.
4. Examine whether text value is concentrated around FOMC announcements and other high-information news windows.
5. Test whether the results are robust to SVI-reconstructed and raw-interpolated volatility surfaces.

The raw-vol experiment should not become a separate research question. It belongs in the robustness section:

```text
main surface representation       = SVI-reconstructed surface
robustness surface representation = raw-vol interpolated surface
```

Because the two surface-construction approaches may produce different usable samples, the paper should compare text advantages within each dataset rather than directly interpreting their absolute MAE scales as equivalent.

### Revised Chapter Structure

```text
1. Introduction
   Research gap
   Central methodological innovation
   Progressive RQ1-RQ3

2. Literature Review
   IVS dynamics and forecasting
   Financial text and volatility
   High-frequency announcement effects

3. Data and Surface Construction
   Option and news data
   Black-76 and IV extraction
   SVI main surface
   Raw-vol robustness surface

4. Text-Conditioned FiLM-WGAN
   Surface encoder
   Text representations
   FiLM conditioning
   WGAN-GP
   Financial penalties
   Forecast generation

5. Empirical Design
   Grouped chronological train/validation/test split
   Multi-seed protocol
   Baselines and architecture ablations
   Point and probabilistic metrics
   Statistical inference

6. RQ1: Incremental Predictive Value of Text

7. RQ2: Text Representation Comparison

8. RQ3: Information-Regime Robustness

9. Additional Robustness
   SVI vs raw-vol
   Alternative forecast horizons
   Checkpoint robustness
   Shuffled-text and mismatched-text placebos

10. Conclusion and Practical Discussion
```

Practical pricing, hedging, and risk-management implications can still be discussed in the conclusion, but they are no longer presented as an empirically answered RQ unless the corresponding economic tests are added later.

### Final Research Narrative

The complete research logic can be summarized as:

> 本文首先检验金融文本是否能为日内 Treasury-option IVS 提供超越历史曲面和 quantitative-only model 的增量预测信息；在确认文本价值后，进一步比较 LP semantic embedding、BoW log-count 和 ChatGPT sentiment scores，以识别语义表征是否是预测改进的主要来源；最后考察这种语义信息的增量价值是否在 FOMC 和一般新闻到达等高信息环境中更加明显。

In compact form:

```text
RQ1: Whether text works
        ↓
RQ2: Why semantic text representation works better
        ↓
RQ3: When semantic text information matters most
```

This sequence keeps `text embedding + FiLM-WGAN` as the methodological center of the chapter while assigning RQ1-RQ3 the progressive roles of **whether it works, why it works, and when it matters most**.
