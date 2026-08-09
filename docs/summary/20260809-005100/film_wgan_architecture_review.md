# FiLM-WGAN 文本条件化方案架构与实验设计审查

## 1. 审查范围与总体结论

本文档审查以下模型方案：

- no-text temporal backbone；
- zero-gated residual FiLM text adapter；
- projection critic；
- matched/mismatched text discrimination。

审查基于 GitHub 仓库 [`haobincui/wgan_option`](https://github.com/haobincui/wgan_option) 的 `master` 分支，固定至提交 [`d9a9a07`](https://github.com/haobincui/wgan_option/commit/d9a9a07c122a6810a956719bb438b11e14271864)。本文结论来自静态代码和实验设计审查，不等同于完整训练结果的实证验证。

总体而言，该方案在方法逻辑上成立，而且明显优于将文本向量直接拼接到普通 conditional GAN 中的简单设计。特别是 Stage A no-text parent、Stage B continuation/matched/shuffled 的嵌套实验设计，可以较好地控制额外训练轮数、初始权重和模型容量带来的影响。代码中没有发现明显的 FiLM 初始化错误、WGAN 损失方向错误或 projection score 维度错误。

不过，在启动最终大规模实验之前，建议优先处理以下三个问题：

1. projection critic 中的文本 dropout 会污染 Wasserstein score、mismatch margin 和 gradient penalty；
2. 当前 gradient penalty 将 unsupported IVS cells 的梯度纳入范数；
3. projection critic 可能利用“当前 IVS—文本”关系识别匹配，而不是学习文本与未来 IVS 变化之间的关系。

此外，当前模型更准确的名称是：

> a no-text IVS transition backbone augmented with a zero-gated residual text adapter and bottleneck FiLM modulation, together with a projection critic and a text–surface matching objective

它不是在所有 CNN 层使用 FiLM 的模型，也不是严格意义上处理多期序列的 temporal backbone。

---

## 2. 代码中的实际生成器结构

在 `residual_film` 模式下，生成器可以概括为

\[
\widehat{\Delta}_{t+1}
=
\Delta_{\mathrm{base}}(\boldsymbol{\sigma}_t,\mathbf z_t)
+
h_t\tanh(g)\,
A\!\left(
\left[
\operatorname{vec}\!\left(
\operatorname{FiLM}(\mathbf x_t,\mathbf e_t)-\mathbf x_t
\right),
\mathbf e_t
\right]
\right),
\]

其中：

- \(\boldsymbol{\sigma}_t\) 是当前 IVS；
- \(\mathbf z_t\) 是随机噪声；
- \(\Delta_{\mathrm{base}}\) 是不使用文本的基础预测分支；
- \(\mathbf x_t\) 是 surface backbone 的 bottleneck feature；
- \(\mathbf e_t\) 是文本编码；
- \(A(\cdot)\) 是 residual text adapter；
- \(g\) 是零初始化的全局 text gate；
- \(h_t\in\{0,1\}\) 是 `has_text` mask。

最终预测通过基础变化量和文本残差共同形成。相关实现见 [generator residual-FiLM 路径](https://github.com/haobincui/wgan_option/blob/d9a9a07c122a6810a956719bb438b11e14271864/src/film_wgan/models.py#L150-L264)。

### 2.1 零 gate 初始化的作用

当 \(g=0\) 时，\(\tanh(g)=0\)，因此

\[
\widehat{\Delta}_{t+1}
=
\Delta_{\mathrm{base}}(\boldsymbol{\sigma}_t,\mathbf z_t).
\]

这保证 Stage B 文本模型在初始化时与已训练的 Stage A no-text model 完全嵌套，不会因为新增随机模块立即破坏原预测。FiLM projection 同样使用零初始化，并通过 \((1+\gamma)x+\beta\) 实现初始恒等映射。这一实现是正确的。

但在 gate 为零时，第一步任务梯度只能直接更新 gate，不能更新 text encoder、FiLM projection 和 adapter 主体：

\[
\frac{\partial\widehat{\Delta}}{\partial g}
=\Delta_{\mathrm{text}},
\qquad
\frac{\partial\widehat{\Delta}}
{\partial\theta_{\mathrm{adapter}}}=0.
\]

代码通过对 adapter 最后一层施加很小的随机初始化，使 gate 仍能获得非零梯度。这不是实现错误，但需要监控 gate 是否在训练中真正离开零点。

### 2.2 “FiLM 效应”的识别限制

当前 residual adapter 同时接收：

1. \(\operatorname{FiLM}(\mathbf x_t,\mathbf e_t)-\mathbf x_t\)；
2. 直接输入的文本编码 \(\mathbf e_t\)。

因此，adapter 可以绕过 FiLM 分支，直接通过 late text input 学习预测修正。当前 matched-text 与 no-text 的表现差异识别的是整个 residual text-conditioning package，而不是 FiLM 的独立贡献。

若论文需要说明 FiLM 本身有效，至少应增加：

- residual adapter with bottleneck FiLM：当前主模型；
- text-MLP-only adapter：移除 bottleneck FiLM，但保留直接文本、gate、projection critic 和 mismatch loss；
- FiLM-only adapter：保留 FiLM difference，移除直接 \(\mathbf e_t\) 输入。

### 2.3 “Temporal backbone”的命名限制

当前 backbone 只接收当期 IVS、support mask 和随机噪声，没有显式输入多个滞后 IVS，也没有 temporal convolution、RNN 或 temporal attention。因此更准确的名称是：

> no-text IVS transition backbone

如果论文希望使用 “temporal backbone”，应加入滞后 IVS window、时间间隔特征，或者使用 TCN、RNN、Transformer 等显式时间结构。

---

## 3. Projection critic

当前 critic 可概括为

\[
D(\boldsymbol{\sigma}_{t+1},\boldsymbol{\sigma}_t,\mathbf w_t)
=
u(\boldsymbol{\phi}_t)
+
h_t
\frac{
\left\langle P\boldsymbol{\phi}_t,\mathbf e_t\right\rangle
}{\sqrt{d_e}},
\]

其中

\[
\boldsymbol{\phi}_t
=
\phi(\boldsymbol{\sigma}_t,\boldsymbol{\sigma}_{t+1}).
\]

第一项是 unconditional score，第二项衡量 surface representation 与 text representation 的一致性。critic 输出实数且没有 sigmoid，并使用 \(\sqrt{d_e}\) 进行尺度调整，因此其基本形式和方向是正确的。实现见 [projection critic](https://github.com/haobincui/wgan_option/blob/d9a9a07c122a6810a956719bb438b11e14271864/src/film_wgan/models.py#L267-L402)。

### 3.1 当前 IVS 条件捷径

projection feature 同时由当前和未来 IVS 构成。当文本被错配时，critic 可能只需判断文本与当前市场状态 \(\boldsymbol{\sigma}_t\) 是否一致，而不必判断文本是否能够解释未来 IVS 变化。

这会产生两个问题：

1. generator 不能修改当前 IVS，因此该部分 score 可能无法提供有效的生成梯度；
2. gradient penalty 只对 future surface 求导，无法抑制 critic 对当前 IVS 条件的过度依赖。

建议将 critic 分成两个表示分支：

- unconditional branch 继续使用 \((\boldsymbol{\sigma}_t,\boldsymbol{\sigma}_{t+1})\)；
- projection branch 使用 transition representation，例如

\[
\Delta\log\boldsymbol{\sigma}_{t+1}
=
\log\boldsymbol{\sigma}_{t+1}
-
\log\boldsymbol{\sigma}_t.
\]

同时建议记录

\[
R_{\mathrm{proj}}
=
\frac{
\left\|\nabla_{\boldsymbol{\sigma}_{t+1}}D_{\mathrm{proj}}\right\|
}{
\left\|\nabla_{\boldsymbol{\sigma}_t}D_{\mathrm{proj}}\right\|
}.
\]

如果该比率长期很低，说明 projection branch 主要依赖 generator 无法改变的当前条件。

### 3.2 Projection bias

当前 `surface_projection` 使用带 bias 的线性层，从而引入额外的 text-only term：

\[
\mathbf b^{\top}\mathbf e_t.
\]

理论上，当 matched 和 mismatched batch 包含相同文本集合时，该项可能在 batch mean 中抵消；但在 critic dropout 存在时不一定逐步精确抵消。建议将该层改为 `bias=False`。

---

## 4. Matched/mismatched text discrimination

代码中的 critic adversarial loss 不是简单的

\[
D_{\mathrm{fake}}
+\lambda_{\mathrm{mis}}D_{\mathrm{mismatch}}
-D_{\mathrm{real}},
\]

而是

\[
\mathcal L_D^{\mathrm{adv}}
=
\frac{
\mathbb E[D_{\mathrm{fake}}]
+
\lambda_{\mathrm{mis}}\mathbb E[D_{\mathrm{mismatch}}]
}{1+\lambda_{\mathrm{mis}}}
-
\mathbb E[D_{\mathrm{real}}].
\]

完整目标还包括 gradient penalty：

\[
\mathcal L_D
=
\mathcal L_D^{\mathrm{adv}}
+
\lambda_{\mathrm{GP}}\mathcal L_{\mathrm{GP}}.
\]

实现见 [trainer 中的 critic loss](https://github.com/haobincui/wgan_option/blob/d9a9a07c122a6810a956719bb438b11e14271864/src/film_wgan/trainer.py#L461-L506)。

当 \(\lambda_{\mathrm{mis}}=0.5\) 时，fake 和 mismatch 在负样本池中的实际权重分别为 \(2/3\) 和 \(1/3\)，而不是未经归一化的 1 和 0.5。实现本身合理，但论文公式应与该归一化版本完全一致。

由于引入了匹配负样本，完整 critic objective 不再只是一个纯粹的 Wasserstein-1 estimator。因此论文中建议使用：

> a conditional WGAN-GP augmented with a text–surface matching objective

### 4.1 当前负样本构造的局限

训练中使用 batch 内 `torch.roll` 构造 mismatched text。对于 batch size 大于 1 的情况，它可以保证 donor index 不等于 target index，因此不存在直接的索引错误。但是，它不能保证：

- donor text 与 target text 在内容上不同；
- 两个文本不属于同一新闻事件；
- mismatched pair 不是因为相隔很远的市场状态而过于容易识别；
- donor 和 target 都具有有效文本。

建议未来根据 `surface_pair_id`、text hash 或 article ID 构造显式 derangement，并记录：

- duplicate-donor rate；
- target–donor embedding cosine similarity；
- matched–mismatched score gap；
- projection score 对 future surface 的梯度。

更严格的 negative controls 可以采用同日、同周、相同波动状态或相同文章来源内的 hard negatives。

---

## 5. 三项高优先级修改

### P0-1：将 critic text dropout 设为零

当前相同的 `text_dropout=0.30` 同时进入 generator 和 projection critic。由于 real、fake、mismatched 和 GP 分别调用 critic，同一文本会经历不同 dropout masks。这会导致：

- Wasserstein score difference 混入随机文本编码噪声；
- matched/mismatched margin 不再只反映配对关系；
- GP 约束的是随机子网络，而不是一个稳定 critic 的输入梯度；
- generator 在不同 forward 中接收到不一致的 critic gradient。

建议拆分配置：

```yaml
generator_text_dropout: 0.30
critic_text_dropout: 0.00
```

generator adapter 可以保留适量 dropout；WGAN-GP critic 最好保持确定性。当前配置和 critic text encoder 分别见 [15-seed 配置](https://github.com/haobincui/wgan_option/blob/d9a9a07c122a6810a956719bb438b11e14271864/configs/film_wgan/train_rq1_pair_textbase_15seed.yaml)及 [critic text encoder](https://github.com/haobincui/wgan_option/blob/d9a9a07c122a6810a956719bb438b11e14271864/src/film_wgan/models.py#L300-L314)。

### P0-2：使用 support-aware gradient penalty

当前 trainer 先将 real 和 fake surfaces 乘以 support mask，再把已经被置零的张量传给 GP。GP 随后将其作为新的叶节点，并计算 critic 对所有网格位置的梯度。因此 unsupported cells 虽然输入值为零，仍可能产生非零梯度，并进入整体梯度范数。

对于不规则且稀疏的 IVS support，这可能使 gradient penalty 主要约束缺乏经济意义的位置。

建议采用以下计算顺序：

1. 在未 mask 的 real 与 fake future surfaces 之间进行插值；
2. 令插值 surface `requires_grad_(True)`；
3. 将 `interpolated * support_mask` 输入 critic；
4. 对未 mask 的 `interpolated` 求导。

形式上，令

\[
\widehat{\boldsymbol{\sigma}}
=
\alpha\boldsymbol{\sigma}_{t+1}
+
(1-\alpha)\boldsymbol{\sigma}^{G}_{t+1},
\]

critic 接收

\[
\widehat{\boldsymbol{\sigma}}
\odot\mathbf M_t,
\]

但梯度对 \(\widehat{\boldsymbol{\sigma}}\) 计算。这样 unsupported cells 的梯度会自然变为零。当前 GP 见 [losses.py](https://github.com/haobincui/wgan_option/blob/d9a9a07c122a6810a956719bb438b11e14271864/src/film_wgan/losses.py#L19-L53)。

### P0-3：限制 projection shortcut

建议组合采用以下措施：

- projection branch 改用 future transition feature；
- `surface_projection` 设置 `bias=False`；
- 使用 current-regime-matched hard negatives；
- 记录 projection 对 current/future surface 的梯度比例；
- 分别记录 unconditional score 与 projection score 的均值、方差和尺度。

这项修改的目标不是削弱 critic，而是确保文本条件对 generator 提供与未来 IVS 变化有关的可行动梯度。

---

## 6. 实验识别设计

### 6.1 当前设计的主要优势

当前 RQ1 设计具有以下优点：

- Stage A 训练统一的 no-text parent；
- Stage B 的 continued no-text、matched text 和 shuffled text 从同一 fold/seed parent checkpoint 开始；
- Stage B 使用 fresh critic，并审计 parent checkpoint SHA；
- PCA 只使用每个 fold 的训练文本拟合；
- matched、shuffled 和 no-text 共享相同的 PCA artifact；
- shuffled text 在 split 内置换，并排除相同 surface-pair donor；
- 四个 expanding folds 分别使用互不重叠的 2023Q1–Q4 outer test；
- 推断阶段根据 training seed、surface pair ID 和 draw index 生成相同场景噪声，从而在 variants 之间实现 common random numbers。

相关实现见 [RQ1 实验脚本](https://github.com/haobincui/wgan_option/blob/d9a9a07c122a6810a956719bb438b11e14271864/scripts/rq1_pair/rq1_pair_experiment.py)和 [scenario sampling](https://github.com/haobincui/wgan_option/blob/d9a9a07c122a6810a956719bb438b11e14271864/src/film_wgan/inference.py#L135-L177)。

当前 `master` 的默认实验为 15 seeds、4 folds、7 variants，共 420 runs。旧 implementation memo 中的 3 seeds/84 runs 已经与代码不一致，论文和复现文档应统一引用 frozen commit、实际 launcher 和 `experiment_design.json`。

### 6.2 Primary contrast 的解释边界

`matched` 相对于 `continuation` 不只是增加文本输入，还同时启用了：

- residual text adapter；
- projection conditioning；
- FiLM/adapter regularization；
- matched/mismatched critic loss。

因此

\[
\Delta_{\mathrm{package}}
=
L_{\mathrm{continued\ no\ text}}
-L_{\mathrm{matched}}
\]

识别的是完整 text-conditioning package 的增量预测表现，不能单独解释为文本语义或 FiLM 的效果。

较有说服力的证据链应同时包含：

\[
\Delta_{\mathrm{package}}>0,
\qquad
\Delta_{\mathrm{alignment}}>0,
\qquad
\Delta_{\mathrm{placebo}}\approx0,
\]

其中：

\[
\Delta_{\mathrm{alignment}}
=L_{\mathrm{shuffled}}-L_{\mathrm{matched}},
\]

\[
\Delta_{\mathrm{placebo}}
=L_{\mathrm{continuation}}-L_{\mathrm{shuffled}}.
\]

建议对 shuffled 与 continuation 使用预先设定实质性界限的 equivalence test，例如 TOST。仅仅发现二者差异“不显著”，不能证明二者等价。

### 6.3 Shuffled placebo 的稳健性

当前所有训练 seeds 共用一个基础 `text_permutation_seed`，因此 donor mapping 的偶然性没有进入不确定性估计。建议增加：

- 5–10 个独立 shuffle seeds；
- 同日或同周内 shuffle；
- 相同市场状态、news count 或文章来源内 shuffle。

全局 shuffled text 可以检验总体配对信息，但不能单独证明文本包含超越时间和市场状态代理的语义信息。

### 6.4 Common random numbers 的准确表述

训练过程中虽然 variants 共享初始权重和 training seed，但 mismatch forward、dropout、不同计算图和 CUDA 非确定性会使后续 RNG 流分离。因此准确表述应是：

- paired initialization and paired training seeds；
- exact common random numbers during evaluation。

当前 evaluation noise seed 与 training seed 绑定，因此跨 seed 方差同时包含训练随机性和 Monte Carlo 场景噪声。建议设置独立且固定的 `evaluation_noise_seed`，或保存统一 evaluation noise bank，供所有 training seeds 和 variants 使用。

---

## 7. 其他重要风险

### 7.1 模型容量

按照当前 16×16 grid、PCA-128、112 base channels、4 个 generator residual blocks、2 个 critic residual blocks 和 2048 fusion hidden 的配置估算：

- generator 可训练参数约 37.28 million；
- critic 可训练参数约 12.20 million；
- 合计约 49.48 million。

而各 rolling fold 的训练 surface pairs 约为 1,621–2,900。模型容量相对样本量较大，可能放大训练 pair 记忆、projection shortcut 和跨 seed 不稳定性。

建议先进行小规模容量消融：

- base channels：112 → 64；
- generator residual blocks：4 → 2；
- critic residual blocks：2 → 1；
- fusion hidden：2048 → 512 或 1024。

模型容量应根据 rolling validation、matched–shuffled gap 和跨 seed 稳定性选择，而不应只依据训练损失。

### 7.2 Noise collapse 与分布预测解释

当前配置中重构相关损失明显强于 adversarial loss，并对 short-ATM cells 进一步加权。这可能使 stochastic generator 退化为条件均值或中位数预测，并忽略随机噪声 \(\mathbf z_t\)。

建议报告 latent sensitivity：

\[
R_z
=
\mathbb E\left[
\left\|G(\mathbf c_t,\mathbf z_1)
-G(\mathbf c_t,\mathbf z_2)\right\|_1
\right],
\]

并同时检查：

- scenario spread；
- predictive interval width；
- empirical coverage；
- energy score 或 variogram score；
- `noise_dim=0` 消融。

如果生成场景几乎不随 \(\mathbf z_t\) 变化，则更准确的模型描述是 adversarially regularized point forecaster，而不是完整的 conditional distribution generator。

### 7.3 Adversarial warm-up

当前 generator 的 adversarial weight 在前 10 epochs 为零，之后直接跳至 0.1，而 critic 从训练开始就进行多次更新。建议将 adversarial weight 在若干 epochs 内线性增加，并监控 critic margin 与 generator gradient norm，避免突然开启造成优化震荡。

### 7.4 Unsupported outputs

训练和评价指标使用 support mask，但 inference 仍保存完整矩形 generated surface。unsupported cells 没有可靠的经济含义。建议额外保存 unsupported cells 为 `NaN` 的 supported-only 输出，并要求所有下游图表与指标强制使用 support mask。

### 7.5 无套利性质的表述

当前 active RQ1 配置关闭了 calendar、butterfly 和 smoothness constraints。因此该模型不能被描述为 arbitrage-constrained WGAN，也不能仅根据当前训练目标声称改善了无套利性质。

---

## 8. 建议补充的测试与诊断

现有测试已经覆盖 FiLM identity、zero-gate nesting、`has_text` mask、projection mask、backbone freeze/unfreeze 和 GP finite smoke test。正式实验前建议补充以下测试。

### 8.1 单元与回归测试

1. 验证 gate 为零时只有 gate 获得第一步任务梯度；
2. 验证若干 optimizer steps 后 gate、adapter 和 text encoder 能够打开；
3. 验证训练模式下 critic 对相同输入输出确定，避免 dropout 污染；
4. 对 mismatch loss 的归一化公式进行数值测试；
5. 端到端检查 matched score、mismatched score 和 margin；
6. 验证 donor pair ID 与 text hash 均不同；
7. 验证 support-aware GP 下 unsupported gradient 为零；
8. 验证 unsupported-cell perturbation 不改变受支持区域的损失和指标；
9. 验证 generator step 不留下无用的 critic gradients；
10. 分别测试 unconditional term 和 projection term 的数值与梯度。

### 8.2 训练监控

建议每个 epoch 记录：

- `text_gate` 及其 gradient；
- text delta RMS 与 base delta RMS；
- \(\|\operatorname{FiLM}(x,e)-x\|\)；
- text encoder、adapter 和 FiLM projection 的 gradient norm；
- unconditional score 与 projection score；
- matched–mismatched critic margin；
- projection 对 current/future surface 的 gradient ratio；
- generator 对替换文本的预测敏感度；
- generator 对替换噪声的预测敏感度；
- volatility floor/ceiling saturation rate；
- supported 与 unsupported gradient norm。

---

## 9. 建议的实验执行顺序

在启动完整 420-run matrix 之前，建议先使用一个 fold、三个 seeds 完成以下诊断：

1. 将 critic text dropout 设为零；
2. 修复 support-aware gradient penalty；
3. 修改或诊断 projection branch 的 current-condition shortcut；
4. 确认 text gate 能够离开零点；
5. 比较 text-MLP-only、FiLM-only 和完整 residual adapter；
6. 检查固定条件、更换 \(\mathbf z_t\) 后输出是否真正变化；
7. 比较 full-capacity 与 reduced-capacity model；
8. 通过诊断后冻结 commit、配置和 evaluation noise bank，再运行完整实验矩阵。

---

## 10. 论文可以支持的结论边界

如果 primary confidence interval 排除零，当前设计可以支持：

> Across four rolling 2023 development folds, the matched-text conditional training package achieved lower out-of-sample supported-grid IVS forecast MAE than an equally continued no-text model, conditional on the selected data, architecture, and evaluation procedure.

只有当 matched-text 模型稳定优于多个时间或市场状态匹配的 shuffled controls 时，才可以进一步表述：

> The results are consistent with pair-specific textual alignment providing incremental predictive information beyond the text-conditioning machinery.

当前设计不能直接支持以下结论：

- 新闻对 IVS 具有因果影响；
- 预测改善完全来自文本语义；
- 改善可以单独归因于 FiLM；
- 模型在 2024 年以后或其他市场具有外部有效性；
- 模型优于传统预测方法或达到 state of the art；
- 模型显著改善概率校准或无套利性质；
- 增加训练 seeds 等同于新的 out-of-time confirmation。

若需要证明相对于传统模型的绝对 forecast skill，还应将以下基准正式纳入相同 rolling evaluation pipeline：

- current-surface persistence；
- surface-PCA + Ridge；
- surface-PCA + text-PCA + Ridge；
- deterministic CNN/L1 forecaster。

---

## 11. 最终评价

该方案可以继续作为博士论文的主要模型框架。其最有价值的部分是 Stage A/Stage B 的嵌套训练设计、continued no-text control、matched/shuffled text comparison，以及 fold-train-only text transformation。这些安排明显提升了结果的可解释性和可复现性。

当前最主要的风险不是公式写错，而是：

1. critic 学会识别文本与当前市场状态的相关性，但没有向 generator 提供与未来 IVS 变化有关的有效梯度；
2. 最终表现被错误地单独归因于 FiLM，而实际识别的是完整 text-conditioning package；
3. 强重构损失和过大的模型容量使随机噪声被忽略；
4. 现有 shuffled placebo 和统计推断不足以支持较强的语义或因果结论。

在修复 critic dropout、support-aware GP 和 projection shortcut，并完成小规模诊断与关键消融之后，该方案能够形成一套较有说服力、适合博士论文报告的文本条件化 IVS forecasting framework。

---

## 参考方法文献

- Perez, E., Strub, F., de Vries, H., Dumoulin, V., and Courville, A. (2018). [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871).
- Miyato, T. and Koyama, M. (2018). [cGANs with Projection Discriminator](https://openreview.net/forum?id=ByS1VpgRZ).
- Reed, S., Akata, Z., Yan, X., Logeswaran, L., Schiele, B., and Lee, H. (2016). [Generative Adversarial Text to Image Synthesis](https://proceedings.mlr.press/v48/reed16.html).
- Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., and Courville, A. (2017). [Improved Training of Wasserstein GANs](https://papers.neurips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf).
