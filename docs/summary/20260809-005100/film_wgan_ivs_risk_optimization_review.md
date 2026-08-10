# 文本条件 FiLM-WGAN 隐含波动率曲面模型  
## 风险评估、优化方案与相关文献综述

**文档日期：** 2026年8月10日  
**研究对象：** 新闻文本条件下的下一期隐含波动率曲面（IVS）概率生成模型  
**当前核心架构：** No-text Stage A + Zero-gated Residual FiLM + Projection Critic + Mismatched-text Negatives + WGAN-GP

---

## 目录

1. [执行摘要](#1-执行摘要)  
2. [当前模型结构与训练目标](#2-当前模型结构与训练目标)  
3. [风险优先级矩阵](#3-风险优先级矩阵)  
4. [概率分布退化与噪声失效风险](#4-概率分布退化与噪声失效风险)  
5. [Projection Critic 的识别捷径风险](#5-projection-critic-的识别捷径风险)  
6. [Critic Dropout 导致条件比较不一致](#6-critic-dropout-导致条件比较不一致)  
7. [Gradient Penalty 与 Support Mask 不一致](#7-gradient-penalty-与-support-mask-不一致)  
8. [Mismatched-text Negatives 的构造风险](#8-mismatched-text-negatives-的构造风险)  
9. [静态无套利约束缺失](#9-静态无套利约束缺失)  
10. [Zero-gated FiLM 的梯度启动问题](#10-zero-gated-film-的梯度启动问题)  
11. [Matched 与 Continuation 的识别问题](#11-matched-与-continuation-的识别问题)  
12. [文本表示、尺度捷径与时间泄漏](#12-文本表示尺度捷径与时间泄漏)  
13. [模型容量与样本规模不匹配](#13-模型容量与样本规模不匹配)  
14. [Stage A 的时间序列建模不足](#14-stage-a-的时间序列建模不足)  
15. [推荐的第二版模型结构](#15-推荐的第二版模型结构)  
16. [评价体系与统计推断](#16-评价体系与统计推断)  
17. [最小可识别实验矩阵](#17-最小可识别实验矩阵)  
18. [实施顺序与停止规则](#18-实施顺序与停止规则)  
19. [相关文献与相似方案](#19-相关文献与相似方案)  
20. [论文贡献边界与建议表述](#20-论文贡献边界与建议表述)  
21. [代码级修改清单](#21-代码级修改清单)  
22. [结论](#22-结论)  

---

# 1. 执行摘要

该研究方案具有较强的应用价值：它试图将新闻文本嵌入引入下一期隐含波动率曲面概率预测，并通过 FiLM、projection discriminator、mismatched-text negatives 和 WGAN-GP 识别文本与未来曲面变化之间的关系。

但当前版本最主要的风险并不是通常意义上的“GAN 不稳定”，而是更深层的**模型识别失败**：

> 模型可能在验证集 MAE、判别器分数或 matched-text 指标上表现良好，但实际上没有学到新闻文本对下一期 IVS 条件分布的增量预测信息。

当前训练结构容易收敛为：

> 一个高容量、接近确定性的点预测器，加上一个主要利用当前市场状态、文本范数、新闻缺失模式或宏观 regime 来识别文本匹配关系的判别器。

模型面临六个最紧迫问题：

1. **逐样本重构损失可能压制噪声，使不同 latent noise 生成近乎相同的曲面。**
2. **Projection critic 可能主要识别“当前 IVS 与文本是否匹配”，而不是“文本是否解释未来 IVS 增量”。**
3. **Critic 中的 dropout 使 real、fake、matched 和 mismatched 条件表示不一致。**
4. **Gradient penalty 可能在无报价、无效 support 网格上施加错误约束。**
5. **Mismatch 构造可能混入无文本样本、重复新闻及容易识别的跨 regime 负样本。**
6. **无套利约束关闭，使生成曲面即使平滑、误差较低，也可能不满足基本期权价格约束。**

建议在大规模运行完整实验矩阵前，先执行以下 P0 修改：

- 将 critic dropout 设为 0；
- projection branch 改为只观察未来变化或 Stage A 残差；
- projection linear layer 取消 bias；
- 将 gradient penalty 改为 support-aware；
- 只在有效文本样本中构造显式 derangement；
- 使用 Energy Score 等 proper scoring rule 训练概率分布；
- 增加生成方差、coverage、PIT、协方差和无套利诊断；
- 使用 matched、shuffled、metadata-only、FiLM-only 等严格消融。

---

# 2. 当前模型结构与训练目标

## 2.1 两阶段结构

当前研究框架可概括为：

\[
\text{Stage A: No-text IVS transition model}
\]

\[
\text{Stage B: Stage A}
+\text{zero-gated residual FiLM}
+\text{projection critic}
+\text{mismatched-text negatives}.
\]

Stage A 学习不使用文本的曲面转移：

\[
\sigma_{t+1}
\sim
p_A\left(
\sigma_{t+1}
\mid
\sigma_t,z
\right).
\]

Stage B 在 Stage A 基础上加入新闻文本条件：

\[
\sigma_{t+1}
\sim
p_B\left(
\sigma_{t+1}
\mid
\sigma_t,e_t,z
\right),
\]

其中：

- \(\sigma_t\)：当前 IVS；
- \(e_t\)：预测时点可获得的文本 embedding；
- \(z\)：随机噪声；
- FiLM：使用文本向量调制卷积特征；
- residual adapter：对 no-text 输出增加文本条件修正；
- projection critic：通过曲面特征与文本特征内积实现条件判别；
- mismatched-text negatives：将真实曲面与错误文本组成负样本。

## 2.2 当前生成器输出

生成器预测 log-IV 增量：

\[
\Delta_{t+1}
=
G(\sigma_t,e_t,z),
\]

并重构未来曲面：

\[
\widehat{\sigma}_{t+1}
=
\exp
\left[
\log(\sigma_t)+\Delta_{t+1}
\right].
\]

对未来波动率设置上下限可以防止训练初期数值溢出，但 clipping 也可能掩盖 generator 输出爆炸问题，因此应同时记录 clipping rate。

## 2.3 当前损失结构

当前 generator objective 包括：

\[
\mathcal L_G
=
\lambda_{\mathrm{adv}}\mathcal L_{\mathrm{adv}}
+
\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}
+
\lambda_{\mathrm{ATM}}\mathcal L_{\mathrm{ATM}}
+
\lambda_{\mathrm{smooth}}\mathcal L_{\mathrm{smooth}}
+
\lambda_{\mathrm{arb}}\mathcal L_{\mathrm{arb}}
+
\lambda_{\mathrm{FiLM}}\mathcal R_{\mathrm{FiLM}}.
\]

当前代表性配置包括：

- reconstruction loss 权重较大；
- short-ATM 区域额外加权；
- ATM-short pure MAE 另有独立损失；
- adversarial loss 在前若干 epoch 关闭；
- checkpoint 主要根据验证集 MAE 选择；
- calendar、butterfly 和 smooth constraints 在当前主要配置中关闭。

这套激励结构更接近“带对抗正则的点预测”，而不是严格意义上的“条件分布学习”。

---

# 3. 风险优先级矩阵

| 优先级 | 风险 | 典型症状 | 主要后果 | 核心优化 |
|---|---|---|---|---|
| P0 | 噪声失效、分布退化 | 固定条件改变 \(z\)，输出几乎不变 | WGAN 退化成点预测器 | Energy Score、variogram score、latent usage diagnostics |
| P0 | Projection critic 走捷径 | 判别器依赖当前 IVS、文本范数或偏置 | 无法识别文本对未来曲面的增量信息 | projection 只看未来残差，`bias=False` |
| P0 | Critic 使用 dropout | real/fake/mismatch 的文本编码随机不同 | critic ranking 与 GP 噪声较大 | critic dropout 设为 0 |
| P0 | GP 未严格遵守 support mask | 无报价网格也产生梯度约束 | critic 容量被无效位置占用 | masked input + masked gradient norm |
| P0 | mismatch 混入无文本样本 | mismatch loss 含 unconditional score | 识别“是否有新闻”而非新闻语义 | 仅有文本样本内 derangement |
| P0 | 无套利约束关闭 | 曲面误差低但产生价格套利 | 生成结果缺乏金融可用性 | 总方差/价格空间约束或无套利 decoder |
| P1 | matched 与 continuation 不可直接归因 | 两组训练路径和有效更新参数不同 | 无法将改善解释为文本或 FiLM 效应 | 因子化消融设计 |
| P1 | zero gate 梯度启动慢 | gate 长期接近 0 | 文本路径长期不学习 | 小正值初始化或分阶段解冻 |
| P1 | 参数量相对样本过大 | seed 差异大，训练集表现明显更好 | 记忆样本与 regime，泛化不稳 | 降低 5–10 倍容量，latent-space 建模 |
| P1 | 文本尺度和元数据捷径 | embedding norm 与预测高度相关 | 模型利用新闻数量和市场状态 | LayerNorm/L2、metadata-only control |
| P1 | 文本时点泄漏 | 测试期新闻或后训练 encoder 信息进入模型 | 回测结果高估 | point-in-time 数据审计 |
| P1 | Stage A 时序信息不足 | 对波动率持续性和 regime shift 反应差 | no-text baseline 太弱 | 加入多日因子、VIX、收益率和流动性 |
| P2 | 评价体系偏向 MAE | MAE 好但 coverage 和尾部差 | 无法证明概率预测有效 | proper scores + calibration + economic tests |

---

# 4. 概率分布退化与噪声失效风险

## 4.1 风险来源

当前重构损失近似为：

\[
\mathcal L_{\mathrm{rec}}
=
\mathbb E_{(c,y),z}
\left[
\|G(c,z)-y\|_1
\right],
\]

其中：

- \(c\) 是当前曲面、文本及其他条件；
- \(y\) 是单个实际观察到的下一期曲面；
- \(z\) 是随机噪声。

对每个固定训练样本 \((c,y)\)，所有 \(z\) 都被要求接近同一个 \(y\)。因此，模型不因生成多个不同但合理的情景而获得奖励。

逐点 L1 的最优解倾向于条件中位数：

\[
G(c,z_1)
=
G(c,z_2)
=
\cdots
\approx
\operatorname{Median}(Y\mid c).
\]

当 reconstruction loss 和 ATM loss 权重明显高于 adversarial loss 时，generator 最容易选择忽略 \(z\)。

## 4.2 不能仅根据损失权重判断主导项

不同损失的量纲不同，因此不能只比较：

\[
20,\ 30,\ 0.1
\]

这些 raw coefficients。

应记录每个 loss 对共享 generator 参数产生的梯度范数：

\[
g_j
=
\left\|
\nabla_{\theta_{\mathrm{shared}}}
\lambda_j\mathcal L_j
\right\|_2.
\]

若 reconstruction 和 ATM loss 的梯度范数长期比 adversarial loss 大一个或多个数量级，则可以确认概率生成目标被点预测目标压制。

## 4.3 必须执行的 latent usage diagnostics

对固定条件 \(c\)，生成 \(K=128\) 或 \(256\) 个场景：

\[
Y_k=G(c,z_k).
\]

至少计算以下指标。

### 场景间平均距离

\[
R_z
=
\frac{1}{K(K-1)}
\sum_{k\neq l}
\|Y_k-Y_l\|_1.
\]

### 生成方差与真实预测残差方差之比

\[
R_{\mathrm{var}}
=
\frac{
\operatorname{Var}_z[G(c,z)]
}{
\operatorname{Var}[Y-\widehat Y_{\mathrm{point}}\mid c]
}.
\]

### 协方差矩阵有效秩

令生成情景协方差为 \(\Sigma_G\)，则：

\[
\operatorname{EffectiveRank}(\Sigma_G)
=
\exp
\left[
-\sum_i p_i\log p_i
\right],
\quad
p_i=\frac{\lambda_i}{\sum_j\lambda_j}.
\]

### 经济因子分散度

检查不同 \(z\) 是否产生不同的：

- ATM level；
- short-end level；
- skew；
- term slope；
- curvature；
- left-tail/right-tail wing；
- calendar spread。

如果只有网格噪声变化，而经济因子几乎不变，也不能视为有效概率生成。

## 4.4 推荐使用 Energy Score

对同一个条件生成 \(K\) 个情景：

\[
Y_1,\ldots,Y_K,
\]

Energy Score 的样本估计为：

\[
\mathcal L_{\mathrm{ES}}
=
\frac{1}{K}
\sum_{k=1}^{K}
\|Y_k-y\|_2
-
\frac{1}{2K(K-1)}
\sum_{k\neq l}
\|Y_k-Y_l\|_2.
\]

第一项要求生成情景接近真实观察；第二项奖励合理的场景分散，阻止 mode collapse。

推荐目标：

\[
\mathcal L_G
=
\lambda_{\mathrm{ES}}\mathcal L_{\mathrm{ES}}
+
\lambda_{\mathrm{adv}}\mathcal L_{\mathrm{adv}}
+
\lambda_{\mathrm{ATM}}\mathcal L_{\mathrm{ATM}}
+
\lambda_{\mathrm{arb}}\mathcal L_{\mathrm{arb}}.
\]

建议：

- 训练时每个条件使用 \(K=4\) 或 \(8\)；
- 保留较小的 ATM guardrail；
- 不再同时使用两个高度重叠、权重都很大的 ATM 点损失；
- checkpoint 以 validation Energy Score 为主；
- 同时报告 variogram score，以弥补 Energy Score 对高维依赖结构可能不够敏感的问题。

## 4.5 替代或辅助方法

可以考虑：

- latent reconstruction；
- mode-seeking loss；
- mutual information regularization；
- conditional normalizing flow；
- conditional diffusion；
- latent flow matching。

但这些方法仍需 proper scoring rule 和 calibration 评价，不能只看生成图是否“看起来多样”。

---

# 5. Projection Critic 的识别捷径风险

## 5.1 当前结构

当前 projection critic 可以抽象为：

\[
D(c,y,e)
=
D_{\mathrm{uncond}}(c,y)
+
\frac{
\left\langle
P\phi(c,y),\psi(e)
\right\rangle
}{\sqrt d}.
\]

其中：

- \(c\)：当前 IVS；
- \(y\)：未来 IVS；
- \(e\)：文本；
- \(\phi(c,y)\)：当前和未来曲面拼接后的 joint feature；
- \(\psi(e)\)：文本编码。

## 5.2 Bias 导致 text-only score

若 projection layer 使用 bias：

\[
P\phi(c,y)
=
W\phi(c,y)+b,
\]

则：

\[
\langle P\phi(c,y),\psi(e)\rangle
=
\langle W\phi(c,y),\psi(e)\rangle
+
\langle b,\psi(e)\rangle.
\]

其中：

\[
\langle b,\psi(e)\rangle
\]

完全不依赖未来曲面。这给判别器提供了 text-only shortcut。

建议：

```python
self.surface_projection = nn.Linear(
    transition_feature_dim,
    text_out_dim,
    bias=False,
)
```

## 5.3 当前 IVS—文本匹配捷径

新闻文本通常与当前市场状态高度相关。例如：

- 危机新闻对应较高当前 IV；
- 政策会议新闻对应特定期限结构；
- 公司事件新闻对应局部 skew；
- 新闻密度与 VIX regime 相关。

如果 projection branch 同时观察当前和未来曲面，判别器可能只判断：

\[
\text{当前 IVS 是否与文本匹配},
\]

而不是：

\[
\text{文本是否解释下一期 IVS 的增量变化}.
\]

生成器无法改变当前 IVS，因此这一 shortcut 是 generator 无法对抗的。

## 5.4 推荐结构

保留 unconditional branch 使用 joint state：

\[
D_{\mathrm{uncond}}
=
u\left(
\phi_{\mathrm{joint}}(c,y)
\right).
\]

但 projection branch 只观察 transition residual：

\[
r
=
\log \sigma_{t+1}
-
\log \sigma_t,
\]

或者更强的识别定义：

\[
r
=
\log \sigma_{t+1}
-
\log \widehat\sigma^{A}_{t+1},
\]

其中 \(\widehat\sigma^{A}_{t+1}\) 是 Stage A no-text 预测。

新的 critic：

\[
D(c,y,e)
=
u\left(
\phi_{\mathrm{joint}}(c,y)
\right)
+
h
\frac{
\left\langle
P\phi_{\Delta}(r),
\operatorname{LN}(e)
\right\rangle
}{\sqrt d}.
\]

这样 projection score 的经济含义变为：

> 文本是否能够解释 no-text baseline 尚未解释的未来曲面变化。

## 5.5 Shortcut diagnostics

需要训练或评估以下诊断模型：

1. `current IVS + text`，不输入未来 IVS；
2. `future IVS + text`，不输入当前 IVS；
3. 仅文本；
4. 仅 embedding norm；
5. 仅文章数、来源数、文本长度；
6. 文本向量方向置零但保留 norm；
7. matched text 与未来曲面匹配，但与当前曲面不匹配；
8. same-regime hard mismatch；
9. cross-regime easy mismatch。

还应计算 projection score 的输入梯度比例：

\[
\rho_{\mathrm{grad}}
=
\frac{
\|\partial D_{\mathrm{proj}}/\partial y_{t+1}\|
}{
\|\partial D_{\mathrm{proj}}/\partial y_t\|+\epsilon
}.
\]

若 \(\rho_{\mathrm{grad}}\ll 1\)，说明 critic 主要使用当前状态，而不是未来结果。

---

# 6. Critic Dropout 导致条件比较不一致

## 6.1 问题机制

若 critic 的文本编码器使用 dropout，则同一文本 \(e\) 在以下调用中可能产生不同表示：

- real score；
- fake score；
- mismatch score；
- gradient penalty；
- validation ranking。

因此，critic 实际比较的是：

\[
D(y_{\mathrm{real}},\widetilde e_1)
\quad\text{与}\quad
D(y_{\mathrm{fake}},\widetilde e_2),
\]

其中：

\[
\widetilde e_1\neq \widetilde e_2
\]

可能只是因为 dropout mask 不同，而不是文本不同。

## 6.2 后果

- real–fake score gap 噪声增加；
- matched–mismatched ranking 不稳定；
- GP 梯度方向不稳定；
- 同一 batch 内条件不再严格一致；
- 小样本环境下 seed sensitivity 进一步上升。

## 6.3 建议

拆分 generator 和 critic 的 dropout：

```yaml
generator_text_dropout: 0.10
critic_text_dropout: 0.00
```

critic 的正则化优先使用：

- 更小网络；
- weight decay；
- spectral normalization；
- R1 regularization；
- early stopping；
- data augmentation 或 feature noise，但必须在 real/fake/mismatch 中保持一致。

---

# 7. Gradient Penalty 与 Support Mask 不一致

## 7.1 当前风险

对稀疏 IVS 数据，部分网格没有真实报价。当前训练通常先将 real/fake surface 乘以 support mask：

\[
Y^{M}=Y\odot M.
\]

然后在 mask 后的 real/fake 之间插值并求 gradient penalty。

即使无效位置的输入值被置为 0，critic 对这些坐标的偏导仍可能非零。若 GP 直接计算所有坐标的梯度范数：

\[
\|\nabla_{\widehat Y}D\|_2,
\]

无报价位置仍然影响 Lipschitz penalty。

## 7.2 后果

- critic 被迫控制无效网格方向的梯度；
- 不同样本 support 密度不同，GP 的有效强度不同；
- sparse surface 样本的梯度范数可能被系统性扭曲；
- critic 容量被无意义位置占用。

## 7.3 推荐实现

先在未 mask 的变量空间插值：

\[
\widehat Y
=
\alpha Y_{\mathrm{real}}
+
(1-\alpha)Y_{\mathrm{fake}},
\]

设置：

\[
\widehat Y.\mathrm{requires\_grad}=True.
\]

输入 critic 时再 mask：

\[
D(\widehat Y\odot M,c,e).
\]

对输入梯度进行 mask：

\[
g_M
=
M\odot
\nabla_{\widehat Y}
D(\widehat Y\odot M,c,e).
\]

可根据有效网格数进行归一化：

\[
\widetilde g_M
=
\frac{g_M}{
\sqrt{\max(\sum M,1)}
}.
\]

GP：

\[
\mathcal L_{\mathrm{GP}}
=
\lambda_{\mathrm{GP}}
\mathbb E
\left[
\left(
\|\widetilde g_M\|_2-\tau
\right)^2
\right].
\]

其中 \(\tau\) 应与归一化定义一致。

## 7.4 单元测试

应增加：

```python
assert grad[mask == 0].abs().max() < tolerance
```

并测试：

- 全 support；
- 部分 support；
- 每个样本 support 不同；
- support 为空时的错误处理；
- mask density 改变时 GP 不应机械性增大。

---

# 8. Mismatched-text Negatives 的构造风险

## 8.1 无文本样本混入 mismatch

若使用 batch roll：

```python
donor_indices = torch.roll(indices, shifts=1)
```

同时 roll `has_text`，则 donor 可能是无文本样本。此时 projection 项被 mask，mismatch score 退化为 unconditional score，却仍被计入 mismatch loss。

最终模型可能学习：

- 是否有新闻；
- 新闻缺失模式；
- 新闻数量；
- 当前 regime；
- unconditional real/fake discrimination；

而不是文本语义与未来 IVS 是否匹配。

## 8.2 正确的有效样本筛选

仅在：

\[
\mathcal I
=
\{i: \text{has\_text}_i=1\}
\]

中构造 mismatch。

```python
valid_idx = torch.where(
    has_text.squeeze(-1).bool()
)[0]
```

然后生成显式 derangement：

\[
\pi(i)\neq i.
\]

还应禁止：

- 文本 hash 相同；
- 同一新闻转载；
- 相同事件 ID；
- 高度相似标题；
- 相同聚合新闻集合。

## 8.3 Easy negatives 与 hard negatives

随机跨期 mismatch 容易产生以下 easy negatives：

- 危机新闻与低波动曲面；
- 财报季新闻与非财报时期；
- 高 VIX regime 文本与低 VIX regime 曲面；
- 高新闻密度与低新闻密度样本。

critic 可以通过 regime 差异识别，而无需理解语义。

推荐 hard negative matching 条件：

- 同一交易日；
- 相邻交易日；
- 相似当前 IVS level；
- 相似 VIX；
- 相似收益率；
- 相似文章数量；
- 相似 embedding norm；
- 相同来源构成；
- 但事件语义不同。

## 8.4 将 mismatch objective 与 WGAN objective 分离

可以使用 margin ranking loss：

\[
\mathcal L_{\mathrm{rank}}
=
\max
\left[
0,
m
-
s(c,y,e^+)
+
s(c,y,e^-)
\right].
\]

或者使用 InfoNCE：

\[
\mathcal L_{\mathrm{NCE}}
=
-\log
\frac{
\exp[s(c,y,e^+)/\tau]
}{
\sum_j
\exp[s(c,y,e_j)/\tau]
}.
\]

模型应准确描述为：

> Conditional WGAN-GP with a text–surface contrastive or ranking regularizer.

不应将包含 mismatch ranking 的 critic loss 全部解释为标准 Wasserstein distance estimator。

---

# 9. 静态无套利约束缺失

## 9.1 平滑不等于无套利

IV surface 平滑并不保证由其隐含的 option prices 满足：

- 对执行价单调；
- 对执行价凸；
- 对期限一致；
- 合理边界条件。

因此：

\[
\mathcal L_{\mathrm{smooth}}\approx 0
\]

并不意味着：

\[
\mathcal L_{\mathrm{arb}}\approx 0.
\]

## 9.2 应处理的静态无套利条件

对标准化 call price \(C(K,T)\)，至少需要：

### 执行价单调性

\[
\frac{\partial C}{\partial K}\le 0.
\]

### 执行价凸性

\[
\frac{\partial^2 C}{\partial K^2}\ge 0.
\]

对应无 butterfly arbitrage。

### 期限单调性

在合适的远期价格和贴现框架下：

\[
C(K,T_2)\ge C(K,T_1),
\quad T_2>T_1.
\]

### 总方差约束

令：

\[
w(k,T)
=
\sigma^2(k,T)T.
\]

许多无套利条件在 total variance 或 SVI/eSSVI 参数空间中更自然。

## 9.3 三种优化路径

### 路径 A：无套利参数化 decoder

生成器不直接输出 raw IV grid，而输出：

- SVI/eSSVI 参数；
- discrete local volatility；
- arbitrage-free neural decoder 的 latent factors。

\[
z_{\mathrm{IV}}
\rightarrow
\theta_{\mathrm{arb}}
\rightarrow
\sigma(k,T).
\]

优点：

- 金融约束强；
- 结果可解释；
- 易于生成可交易曲面。

缺点：

- 参数化限制表达能力；
- 需要稳定标定；
- 极端市场下可能失配。

### 路径 B：可微无套利 penalty

将生成 IV 转为价格或 total variance，再计算：

\[
\mathcal L_{\mathrm{arb}}
=
\lambda_{\mathrm{cal}}\mathcal L_{\mathrm{calendar}}
+
\lambda_{\mathrm{bf}}\mathcal L_{\mathrm{butterfly}}
+
\lambda_{\mathrm{mono}}\mathcal L_{\mathrm{monotonicity}}.
\]

必须在真实的非均匀 strike/maturity grid 上正确计算有限差分，不能默认网格等距。

### 路径 C：生成后 refinement

先得到概率场景：

\[
Y_k^{\mathrm{raw}},
\]

再投影到无套利集合：

\[
Y_k^{\mathrm{arb}}
=
\Pi_{\mathcal A}
\left(
Y_k^{\mathrm{raw}}
\right).
\]

需要分别报告 refinement 前后：

- Energy Score；
- coverage；
- variance；
- factor correlation；
- tail quantiles；
- arbitrage violation rate；
- 投影距离。

否则 refinement 可能把概率分布压缩成过度平滑的窄分布。

## 9.4 当前代码中无套利离散化的额外风险

若 butterfly penalty 使用简单二阶差分：

\[
C_{j+1}-2C_j+C_{j-1},
\]

则只有在 strike grid 等距时才严格对应 convexity。

对于非均匀 \(K_j\)，应使用斜率差：

\[
\frac{C_{j+1}-C_j}{K_{j+1}-K_j}
-
\frac{C_j-C_{j-1}}{K_j-K_{j-1}}
\ge 0.
\]

calendar penalty 也应确认：

- 是否使用统一 forward moneyness；
- 利率和股息处理是否一致；
- strike grid 在不同 maturity 上是否可比较；
- 期限排序是否严格正确。

---

# 10. Zero-gated FiLM 的梯度启动问题

## 10.1 当前残差结构

文本增量近似为：

\[
\Delta_{\mathrm{text}}
=
\tanh(g)
A(c,e,z),
\]

初始：

\[
g_0=0.
\]

于是：

\[
\tanh(g_0)=0.
\]

对 adapter 参数 \(\theta_A\)：

\[
\frac{\partial \Delta_{\mathrm{text}}}
{\partial \theta_A}
=
\tanh(g)
\frac{\partial A}{\partial \theta_A}.
\]

当 \(g=0\)：

\[
\frac{\partial \Delta_{\mathrm{text}}}
{\partial \theta_A}
=0.
\]

因此文本 adapter 和 residual FiLM 分支初始几乎无法获得来自最终输出的梯度。

## 10.2 可能的训练路径

训练开始时主要更新：

\[
\frac{\partial \Delta_{\mathrm{text}}}{\partial g}
=
\operatorname{sech}^2(g)
A(c,e,z).
\]

但若 adapter 最后一层初始化极小：

\[
A(c,e,z)\approx 0,
\]

则 gate 梯度也可能很小，导致文本支路长时间停留在 no-text 状态。

## 10.3 优化方案

### 方案 1：小正值 gate 初始化

例如：

\[
g_0=0.01
\quad\text{或}\quad
0.02.
\]

这样仍近似保持 Stage A，同时允许文本支路立即获得梯度。

### 方案 2：分阶段解冻

建议：

1. 前 1–2 epoch 冻结 Stage A backbone；
2. 训练 text encoder、adapter 和 gate；
3. 解冻 bottleneck FiLM；
4. 再解冻最后一个或两个 backbone block；
5. 只有验证指标支持时才全量解冻。

### 方案 3：零初始化输出投影而非完全关闭整个分支

可以保持文本分支内部可训练，但使其最终输出 projection 初始为零，类似 zero convolution 或 ReZero residual scaling。

## 10.4 必须记录的训练监控

每个 epoch 记录：

- raw gate；
- \(\tanh(g)\)；
- gate gradient；
- text encoder gradient norm；
- FiLM gamma/beta gradient norm；
- adapter gradient norm；
- text delta RMS；
- base delta RMS；
- \(\mathrm{RMS}(\Delta_{\mathrm{text}})/
  \mathrm{RMS}(\Delta_{\mathrm{base}})\)；
- matched 与 shuffled 的 gate 演化差异。

若 matched 和 shuffled 的 gate 都快速增大，可能只是架构容量增加，而非真实语义信息。

---

# 11. Matched 与 Continuation 的识别问题

## 11.1 为什么不能直接归因

Matched 与 continuation 分支可能同时在以下方面不同：

- 是否有 text encoder；
- 是否有 residual adapter；
- 是否有 FiLM；
- 是否有 projection critic；
- 是否有 mismatch objective；
- 哪些参数被冻结；
- 有效学习率；
- 有效训练步数；
- optimizer parameter groups；
- regularization；
- early stopping trajectory。

因此：

\[
\text{Matched}
-
\text{Continuation}
\]

不能直接解释为：

- 文本语义效应；
- FiLM 效应；
- projection critic 效应；
- mismatch objective 效应。

它只能解释为：

> 完整 text-conditioned training package 相对于 continuation package 的总增量。

## 11.2 冻结阶段的非对称性

如果 continuation 分支：

- 没有文本路径；
- backbone 被冻结；
- generator 的其他路径也未更新；

那么冻结阶段可能几乎没有有效 generator learning。

Matched 分支则可能更新：

- text encoder；
- adapter；
- gate；
- projection-related parameters。

此时两组拥有不同的有效训练预算。

## 11.3 正确识别文本语义

最重要的预设比较应为：

\[
\text{Full matched}
>
\text{Full shuffled}.
\]

该比较检验：

> 正确语义对齐是否带来增量预测信息。

还应检验：

\[
\text{Full shuffled}
\approx
\text{Continuation}.
\]

但“不显著”不能直接解释为“相等”。应执行等效性检验，并预先设定 margin：

\[
-\delta
<
\Delta
<
\delta.
\]

## 11.4 Common random numbers

为降低模型比较方差，应尽量保证：

- 相同 Stage A parent checkpoint；
- 相同 minibatch order；
- 相同 latent noise；
- 相同初始化；
- 相同训练步数；
- 相同解冻 schedule；
- 相同数据 split；
- 相同 evaluation MC draws。

---

# 12. 文本表示、尺度捷径与时间泄漏

## 12.1 Embedding norm 可能携带非语义信息

即使单篇文章 embedding 已归一化，若多篇文章进行均值聚合：

\[
e_t
=
\frac{1}{N_t}
\sum_{n=1}^{N_t}
\widetilde e_{t,n},
\]

其范数仍可能反映：

- 文章数量；
- 新闻内容一致性；
- 事件集中度；
- 来源结构；
- 新闻噪声；
- 市场 regime。

模型可能利用：

\[
\|e_t\|
\]

而不是 embedding direction 中的语义。

## 12.2 PCA 与 normalization

建议：

1. PCA 仅在每个训练 fold 上拟合；
2. validation/test 只使用训练期 transform；
3. PCA 后增加 LayerNorm 或 L2 normalization；
4. 比较 whitened 与 non-whitened；
5. 保存 transform hash 和数据 lineage；
6. 检查 embedding 各主成分与 VIX、收益率和新闻数量的相关性。

## 12.3 Metadata-only baseline

应显式建立只使用以下变量的模型：

- 文章数；
- 来源数；
- 文本长度；
- 发布时间分布；
- embedding norm；
- 新闻缺失 indicator；
- 标题重复率；
- 来源集中度。

如果 metadata-only 已能达到接近 full-text 的表现，则不能将 full model 的改善解释为新闻语义。

## 12.4 文本聚合方法

简单均值适合作为基线，但可能丢失：

- 相反新闻之间的冲突；
- 新闻重要性；
- 新闻时效差异；
- 来源可信度；
- 事件之间的层级关系。

可比较：

- mean pooling；
- time-decay pooling；
- attention pooling；
- set transformer；
- source-aware attention；
- event-cluster pooling；
- top-k salient news selection。

## 12.5 Point-in-time 泄漏

必须审计：

- 发布时间；
- 抓取时间；
- 修订时间；
- 时区；
- market close cutoff；
- overnight 新闻归属；
- 同一新闻跨日转载；
- 测试期新闻是否参与 PCA；
- 文本 encoder 是否使用预测期之后语料训练；
- embedding 文件是否在完整样本上预处理。

现代预训练 embedding 在测试期之后训练，不一定构成直接标签泄漏，但会影响“严格 point-in-time”解释。应至少加入：

- TF–IDF；
- 历史情绪词典；
- point-in-time encoder；
- random projection；
- metadata-only；

作为稳健性检验。

---

# 13. 模型容量与样本规模不匹配

## 13.1 风险

若模型参数达到数千万，而训练样本只有约数千个 surface pairs，则存在明显的高容量风险：

- critic 记忆真实样本；
- generator 记忆 transition；
- 文本路径记忆重大事件；
- shuffled placebo 也可能过拟合；
- seed 间方差大；
- validation checkpoint selection 偏差高；
- regime out-of-sample 表现不稳定。

参数共享和卷积结构可以缓解，但无法消除该问题。

## 13.2 建议初始缩减

| 参数 | 当前代表性值 | 建议诊断值 |
|---|---:|---:|
| Generator base channels | 112 | 48 或 64 |
| Critic base channels | 112 | 48 或 64 |
| Fusion hidden dimension | 2048 | 256 或 512 |
| Generator residual blocks | 4 | 2 |
| Critic residual blocks | 2 | 1–2 |
| Noise dimension | 128 | 8、16、32 |
| Text PCA dimension | 128 | 16、32、64 |
| Critic dropout | 0.30 | 0 |
| Gate initial value | 0 | 0.01 或分阶段训练 |

这些值是用于风险诊断的起点，不是最终最优超参数。

## 13.3 推荐低维 IVS latent space

可先通过 PCA、VAE 或结构化因子得到：

\[
f_t
=
E(\sigma_t),
\]

其中 \(f_t\) 包括：

- level；
- short-end level；
- term slope；
- skew；
- curvature；
- wing asymmetry；
- volatility-of-volatility factor。

在 latent space 生成：

\[
f_{t+1}
\sim
p_\theta
\left(
f_{t+1}
\mid
f_t,e_t,x_t
\right),
\]

再使用无套利 decoder：

\[
\widehat\sigma_{t+1}
=
D_{\mathrm{arb}}(f_{t+1}).
\]

优势：

- 参数量明显下降；
- 概率分布更容易估计；
- 因子解释更清晰；
- 协方差和 calibration 更容易评价；
- 可将文本作用解释为对 level、skew、term slope 等因子的影响。

---

# 14. Stage A 的时间序列建模不足

## 14.1 当前问题

仅输入当前曲面：

\[
\sigma_t
\rightarrow
\sigma_{t+1}
\]

不足以显式表达：

- 多日波动率持续性；
- 均值回复；
- regime transition；
- leverage effect；
- volatility clustering；
- 新闻冲击后的衰减；
- support/liquidity 状态。

## 14.2 推荐输入

Stage A 至少可加入：

\[
c_t^A
=
[
f_t,
f_{t-1},
\ldots,
f_{t-L},
\operatorname{EWMA}(f),
r_t,
|r_t|,
r_t^2,
\mathrm{RV}_t,
\mathrm{VIX}_t,
\mathrm{liquidity}_t,
\mathrm{support}_t
].
\]

可考虑：

- 最近 5、10、20 日 latent factors；
- 标的收益率；
- realized volatility；
- VIX；
- bid–ask spread；
- option volume/open interest；
- 可用网格数量；
- 市场事件日；
- 到期日滚动信息。

## 14.3 基线顺序

不应直接将大型 Transformer 作为唯一时序基线。建议先比较：

1. current-surface/random walk；
2. factor AR/VAR；
3. HAR-style factor model；
4. PCA + LSTM；
5. conditional WGAN；
6. conditional diffusion；
7. latent flow matching；
8. temporal CNN/Transformer。

Stage A 必须足够强，否则 Stage B 的“文本增量”可能只是弥补 no-text baseline 的明显缺陷，而非真正的语义贡献。

---

# 15. 推荐的第二版模型结构

# 15.1 Stage A：低维 no-text 概率基线

首先构造 IVS latent factors：

\[
f_t=E(\sigma_t).
\]

输入：

\[
c_t^A
=
[
f_t,
\operatorname{EWMA}(f),
r_t,
|r_t|,
\mathrm{RV}_t,
\mathrm{VIX}_t,
\mathrm{liquidity}_t
].
\]

生成基础情景：

\[
f_{t+1}^{A,k}
=
G_A(c_t^A,z_k).
\]

Stage A 单独通过以下评价：

- MAE/RMSE；
- Energy Score；
- calibration；
- covariance；
- regime robustness；
- no-arbitrage reconstruction。

## 15.2 Stage B：文本 residual correction

文本标准化：

\[
\widetilde e_t
=
\operatorname{LN}
\left[
\operatorname{PCA}(e_t)
\right].
\]

文本 residual：

\[
f_{t+1}^{B,k}
=
f_{t+1}^{A,k}
+
g_t
R_\theta
(c_t^A,\widetilde e_t,z_k).
\]

其中：

- \(g_t\) 可以是全局 scalar；
- 也可以是因子级 gate；
- 还可以由文本和当前市场状态动态决定。

但动态 gate 需要防止直接利用文本 norm 和 regime 走捷径。

## 15.3 FiLM-only 与 adapter-only 分离

为证明 FiLM 的具体价值，应分别构建：

### FiLM-only

文本只通过 FiLM 调制 latent/surface feature：

\[
h_l'
=
(1+\gamma_l(e))\odot h_l
+
\beta_l(e).
\]

不再把文本直接拼接到最终 residual adapter。

### Adapter-only

文本只进入 residual MLP：

\[
\Delta_{\mathrm{text}}
=
A(c,e,z).
\]

### Full model

FiLM + residual adapter。

只有比较三者，才能判断改善来自：

- feature modulation；
- direct text residual；
- 额外容量；
- projection critic。

## 15.4 推荐 critic

\[
D
=
u
\left[
\phi_{\mathrm{joint}}(f_t,f_{t+1})
\right]
+
\frac{
\left\langle
P\phi_r
\left(
f_{t+1}
-
\widehat f_{t+1}^{A}
\right),
\widetilde e_t
\right\rangle
}{
\sqrt d
}.
\]

关键约束：

- projection layer `bias=False`；
- projection branch 不直接看当前曲面；
- critic dropout 为 0；
- mismatch ranking 单独计算；
- GP 仅作用于有效 support；
- 文本 embedding 进行 normalization；
- projection feature 和 text feature 维度较小。

## 15.5 推荐 generator objective

\[
\mathcal L_G
=
\lambda_{\mathrm{ES}}\mathcal L_{\mathrm{ES}}
+
\lambda_{\mathrm{adv}}\mathcal L_{\mathrm{adv}}
+
\lambda_{\mathrm{ATM}}\mathcal L_{\mathrm{ATM}}
+
\lambda_{\mathrm{arb}}\mathcal L_{\mathrm{arb}}
+
\lambda_{\mathrm{reg}}\mathcal R.
\]

其中：

- \(\mathcal L_{\mathrm{ES}}\)：概率分布主目标；
- \(\mathcal L_{\mathrm{adv}}\)：高阶结构辅助目标；
- \(\mathcal L_{\mathrm{ATM}}\)：经济重点区域 guardrail；
- \(\mathcal L_{\mathrm{arb}}\)：静态无套利约束；
- \(\mathcal R\)：gate、FiLM 或参数正则化。

对抗权重使用平滑 ramp：

\[
\lambda_{\mathrm{adv}}(t)
=
\lambda_{\max}
\min
\left[
1,
\frac{t-t_0}{T_{\mathrm{ramp}}}
\right].
\]

避免在单个 epoch 从 0 突然跳至固定值。

---

# 16. 评价体系与统计推断

## 16.1 点预测指标

保留：

- MAE；
- RMSE；
- weighted MAE；
- short-ATM MAE；
- ATM pure MAE；
- level factor error；
- skew factor error；
- term slope error；
- current-surface/random-walk improvement。

点预测结果仍然重要，但不能作为概率生成模型的唯一选择指标。

## 16.2 概率预测指标

至少加入：

### Energy Score

\[
\operatorname{ES}
=
\mathbb E
\|Y-y\|
-
\frac{1}{2}
\mathbb E
\|Y-Y'\|.
\]

### Variogram Score

用于评价网格间依赖结构：

\[
\operatorname{VS}
=
\sum_{i,j}
w_{ij}
\left[
|y_i-y_j|^p
-
\mathbb E|Y_i-Y_j|^p
\right]^2.
\]

### Marginal CRPS

对每个网格或 latent factor 单独计算。

### Coverage

报告：

- 50%；
- 80%；
- 90%；

预测区间 coverage。

### Interval width

coverage 好但区间过宽也没有实际价值，因此必须联合报告 sharpness。

### PIT / Rank Histogram

检查是否：

- U 型：分布过窄；
- 倒 U 型：分布过宽；
- 偏斜：系统性 bias；
- 非均匀：分布形状错误。

### Generated spread / empirical residual spread

\[
R_{\mathrm{spread}}
=
\frac{
\operatorname{SD}_z[\widehat Y]
}{
\operatorname{SD}[Y-\widehat Y_{\mathrm{center}}]
}.
\]

### Covariance diagnostics

- covariance Frobenius error；
- correlation matrix error；
- effective rank；
- PCA eigenvalue spectrum；
- factor tail dependence。

## 16.3 金融有效性指标

- calendar violation rate；
- butterfly violation rate；
- monotonicity violation rate；
- violation magnitude；
- option portfolio repricing error；
- delta–vega hedge P&L；
- VaR exception rate；
- ES exceedance；
- stress-day coverage；
- short maturity 和 ATM 区域的经济损失；
- surface refinement distortion。

## 16.4 Regime 分层

至少分层报告：

- 高/低 VIX；
- 正/负大收益；
- 危机/平稳时期；
- 高/低新闻密度；
- 有文本/无文本；
- 高/低 support density；
- short/long maturity；
- ATM/OTM；
- 重大事件日/普通交易日。

## 16.5 统计推断

时间序列数据不能把所有网格点视为独立观察。

建议：

- 以交易日为基本比较单位；
- week/month block bootstrap；
- fold-level aggregation；
- seed-level full distribution；
- paired loss difference；
- Diebold–Mariano 类检验；
- multiple-testing adjustment；
- equivalence test；
- 预先设定主要指标和主要比较。

避免：

- 只报告最佳 seed；
- 用网格点数量虚增样本量；
- 根据 test set 选择模型；
- 在多项指标中挑选最有利结果。

---

# 17. 最小可识别实验矩阵

## 17.1 核心实验组

| 组别 | Generator 文本结构 | Projection Critic | Mismatch | 文本输入 | 识别目的 |
|---|---:|---:|---:|---|---|
| A. Continuation | 无 | 无 | 无 | 无 | no-text 基线 |
| B. Full Matched | FiLM + Adapter | 有 | 有 | 正确匹配 | 完整文本方案 |
| C. Full Shuffled | FiLM + Adapter | 有 | 有 | 打乱 | 语义对齐 placebo |
| D. Metadata-only | Adapter/FiLM | 有 | 有 | 新闻元数据 | 识别非语义捷径 |
| E. Adapter-only | 仅 residual adapter | 可选 | 可选 | 正确匹配 | 识别直接文本残差 |
| F. FiLM-only | 仅 FiLM | 可选 | 可选 | 正确匹配 | 识别 FiLM 机制 |
| G. Projection-only | 无 generator 文本路径 | 有 | 有 | 正确匹配 | 识别 critic conditioning |
| H. Mismatch-only | 可选 | 有 | 有 | 正确匹配 | 识别 mismatch objective |
| I. Norm-only | 简化 | 有 | 有 | embedding norm | 检查尺度捷径 |
| J. Future-text placebo | 完整 | 有 | 有 | 不可用未来文本 | 泄漏上界诊断 |

## 17.2 主要预设假设

### 假设 H1：文本语义有增量预测价值

\[
\operatorname{Score}
(\text{Matched})
<
\operatorname{Score}
(\text{Shuffled}).
\]

这里 score 应以 Energy Score、CRPS 或主要经济损失为主，而不是只使用 MAE。

### 假设 H2：破坏语义后不优于 no-text

\[
\text{Shuffled}
\approx
\text{Continuation}.
\]

应使用等效性检验。

### 假设 H3：Full model 优于 metadata-only

\[
\text{Full Matched}
>
\text{Metadata-only}.
\]

否则改善可能来自新闻数量和市场 regime。

### 假设 H4：FiLM 提供独立机制价值

\[
\text{Full}
>
\text{Adapter-only}.
\]

若 FiLM-only 和 adapter-only 表现相近，则论文不应强调 FiLM 是主要创新来源。

## 17.3 Shuffle 设计

必须：

- 使用多个 shuffle seed；
- 显式禁止 self-match；
- 禁止重复新闻；
- 禁止相同事件；
- 保持新闻数量分布；
- 保持 embedding norm 分布；
- 最好在相似 regime 内 shuffle。

至少区分：

- global shuffle；
- within-day shuffle；
- within-regime shuffle；
- hard-negative semantic shuffle。

---

# 18. 实施顺序与停止规则

## 18.1 第一阶段：先修复实现风险

按顺序完成：

1. critic dropout 设为 0；
2. projection layer 改为 `bias=False`；
3. projection branch 改看未来增量或 Stage A residual；
4. support-aware GP；
5. mismatch 只在有文本样本中计算；
6. explicit derangement；
7. 文本 LayerNorm/L2；
8. 记录每项损失的 gradient norm；
9. 记录 gate、FiLM 和 adapter 的梯度；
10. 记录输出 clipping rate。

## 18.2 第二阶段：单 fold 小规模诊断

仅运行：

- continuation；
- matched；
- shuffled；
- metadata-only。

每组使用少量 seed，并对固定条件生成 256 个情景。

## 18.3 停止规则

满足以下任一条件时，应停止扩展大规模实验并先修改模型：

### 噪声塌缩

\[
R_z
\approx 0.
\]

### 生成方差严重不足

\[
R_{\mathrm{spread}}
<0.2
\]

或明显低于合理范围。

### Coverage 严重不足

例如 nominal 90% 区间覆盖率显著低于目标。

### Projection shortcut

current-only + text 模型的 mismatch accuracy 接近 full critic。

### Metadata shortcut

metadata-only 与 full-text 表现接近。

### 无套利违反严重

大量生成情景需要较大 refinement 才能满足静态无套利。

### Gate 不启动

多次 seed 下 gate 和文本分支梯度长期接近 0。

## 18.4 第三阶段：重构概率目标

- 引入 Energy Score；
- 每个条件训练 \(K=4\) 或 \(8\) 个情景；
- 减少重复 ATM loss；
- adversarial loss 平滑 warm-up；
- checkpoint metric 改为 distribution score；
- 增加 variogram score。

## 18.5 第四阶段：机制消融

执行：

- FiLM-only；
- Adapter-only；
- Projection-only；
- Mismatch-only；
- Full model。

## 18.6 第五阶段：完整 fold × seed

只有当以下条件同时成立后再扩大实验：

- matched 优于 shuffled；
- shuffled 与 continuation 等效；
- full 优于 metadata-only；
- latent noise 有效；
- coverage 合理；
- 无套利违反受控；
- 结果对 seed 不过度敏感；
- no-text Stage A 是强基线。

---

# 19. 相关文献与相似方案

截至 2026年8月10日，公开研究中尚未发现与以下完整组合完全相同的方案：

> 新闻文本条件 + 下一日 IVS 概率生成 + WGAN-GP + zero-gated residual FiLM + projection critic + mismatched-text negatives。

这一判断基于当前可获得的检索结果，并不构成严格的“文献不存在”证明。

## 19.1 与 IVS 概率生成最接近

### VolGAN

**主题：** 条件 GAN、隐含波动率曲面情景生成、金融约束。  
**相关性：** 与当前方案在“conditional GAN + IVS scenario generation”方面最接近。  
**差异：** 不使用新闻文本、FiLM 和 text mismatch objective。  

链接：  
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4617536>

### Forecasting Implied Volatility Surfaces with Generative Diffusion Models

**主题：** 条件 diffusion、下一日 IVS 概率预测、历史市场变量、无套利处理。  
**相关性：** 与研究目标高度一致。  
**差异：** 使用 diffusion，不使用文本条件 WGAN。  

链接：  
<https://arxiv.org/html/2511.07571v2>

### Deep Hedging: Learning to Simulate Equity Option Markets

**主题：** WGAN-GP、期权市场生成、discrete local volatility、无套利表示。  
**启发：** 不一定直接生成 raw IV grid，可在更具金融结构的表示空间生成。  

链接：  
<https://arxiv.org/pdf/1911.01700>

### Two-Step Framework for Arbitrage-Free IVS Prediction

**主题：** PCA/VAE 低维表示、latent dynamics、无套利 decoder。  
**启发：** 可作为高容量 raw-surface WGAN 的低维替代或强基线。  

链接：  
<https://arxiv.org/abs/2106.07177>

### Latent Flow Matching for Arbitrage-Aware IVS Generation

**主题：** latent representation、flow matching、无套利 penalty。  
**启发：** 支持“先压缩、再生成、后约束”的架构。  

链接：  
<https://arxiv.org/html/2608.00616v2>

### Decoupled Probabilistic Forecasting and Arbitrage-Aware Refinement

**主题：** 概率预测与无套利 refinement 解耦。  
**启发：** 可用于生成后投影，但必须评估分布扭曲。  

链接：  
<https://arxiv.org/html/2607.29220v1>

## 19.2 与文本—波动率预测接近

### Degree of Irrationality / Text Sentiment and IVS Forecasting

**主题：** 文本情绪、下一日 IVS 预测。  
**相关性：** 是“文本 + IVS”方向的重要近邻。  
**差异：** 更偏传统预测，不是概率生成 WGAN。  

链接：  
<https://arxiv.org/abs/2405.11730>

### Realised Volatility Forecasting with Financial Word Embeddings

**主题：** 金融文本 embedding、realised volatility forecasting。  
**启发：** 支持文本具有波动率增量预测信息，但目标不是完整 IVS。  

链接：  
<https://arxiv.org/html/2108.00480v5>

### M2VN

**主题：** 新闻与时间序列跨模态对齐、point-in-time language model、look-ahead bias。  
**启发：** 对文本时点控制、跨模态识别和实验设计具有直接参考价值。  

链接：  
<https://arxiv.org/pdf/2510.20699>

### FININ 等新闻—金融多模态模型

**主题：** 多新闻交互、跨模态金融预测、新闻聚合。  
**启发：** 简单均值聚合可能无法处理大量新闻及相互冲突信息。  

链接：  
<https://aclanthology.org/2024.findings-emnlp.189.pdf>

## 19.3 当前模型组件的方法论先例

### FiLM

Feature-wise Linear Modulation：

\[
h'
=
(1+\gamma(e))\odot h+\beta(e).
\]

链接：  
<https://arxiv.org/abs/1709.07871>

### Projection Discriminator

通过条件 embedding 与样本 feature 的内积实现条件判别。

链接：  
<https://openreview.net/forum?id=ByS1VpgRZ>

### Matching-aware Adversarial Training

使用 matched 和 mismatched 条件对增强条件一致性。

链接：  
<https://proceedings.mlr.press/v48/reed16.pdf>

### WGAN-GP

在真实和生成样本插值点上施加 gradient penalty。

链接：  
<https://proceedings.neurips.cc/paper_files/paper/7159-improved-training-of-wasserstein-gans.pdf>

### ControlNet / Zero Initialization

通过零初始化 residual conditional branch，在保持预训练 backbone 初始行为的同时加入控制条件。

链接：  
<https://arxiv.org/abs/2302.05543>

### ReZero

使用可学习 residual scaling 改善深层网络训练稳定性。

### Scoring-rule-trained Generative Forecasting

使用 Energy Score 等 proper scoring rules 直接训练隐式概率模型。

链接：  
<https://jmlr.org/papers/volume25/23-0038/23-0038.pdf>

### Proper Scoring Rules

proper scoring rule 的核心性质是：模型在报告真实预测分布时最小化期望损失。

链接：  
<https://www.tandfonline.com/doi/abs/10.1198/016214506000001437>

### Arbitrage-free SVI

用于理解 SVI/SSVI 类曲面的静态无套利条件。

链接：  
<https://arxiv.org/abs/1204.0646>

### Deep Smoothing

将拟合、平滑和无套利约束放入可微神经网络框架。

链接：  
<https://arxiv.org/abs/1906.05065>

### BicycleGAN

通过 latent consistency 缓解 conditional generation 中的输出退化。

链接：  
<https://proceedings.neurips.cc/paper/2017/file/819f46e52c25763a55cc642422644317-Paper.pdf>

---

# 20. 论文贡献边界与建议表述

## 20.1 不建议的表述

不建议将研究贡献简单描述为：

- 首次使用 FiLM 预测 IVS；
- 首次使用 projection critic；
- 首次把文本与 WGAN 结合；
- 证明新闻对 IVS 有因果影响。

原因：

- FiLM、projection discriminator、WGAN-GP、matching-aware training 均已有成熟方法文献；
- 当前设计是观察性预测研究，无法直接识别新闻的因果效应；
- 仅组合已有组件，方法论创新性可能不足。

## 20.2 更合理的研究贡献

建议表述为：

> 本文提出一种面向下一期隐含波动率曲面场景预测的文本条件概率生成框架。该框架以 no-text IVS transition model 为基线，通过 residual text conditioning、feature-wise modulation、文本—未来曲面增量对齐、proper scoring-rule training 和无套利约束，识别新闻文本对 IVS 条件分布的增量预测信息。

可拆分为四项贡献：

1. **经济问题贡献**  
   研究新闻文本是否改善完整 IVS 条件分布，而不是单一波动率指标。

2. **识别设计贡献**  
   使用 matched、shuffled、metadata-only 和 hard negatives 区分语义信息、新闻强度和市场 regime。

3. **概率预测贡献**  
   使用 proper scoring rules、calibration 和 scenario covariance 评价生成分布。

4. **金融结构贡献**  
   将静态无套利约束与文本条件概率生成结合。

## 20.3 结论边界

可以声称：

- 新闻文本具有增量预测信息；
- 正确语义对齐改善概率预测；
- 文本主要影响特定期限或 moneyness 区域；
- 文本改善高波动 regime 下的情景覆盖；
- 某种文本注入结构优于其他结构。

不应直接声称：

- 新闻导致 IVS 变化；
- 文本 embedding 识别了结构性因果冲击；
- projection critic 分数等于经济机制；
- matched 优于 continuation 就证明 FiLM 有效。

---

# 21. 代码级修改清单

## 21.1 Generator

- [ ] 将 `text_gate_initial_value` 从 0 调整为小正值，或使用分阶段训练；
- [ ] 记录 gate、adapter、FiLM 梯度；
- [ ] 记录 `text_delta/base_delta`；
- [ ] 加入多样本 Energy Score；
- [ ] 降低 fusion hidden dimension；
- [ ] 降低 noise dimension 并做敏感性分析；
- [ ] 将 FiLM-only 与 adapter-only 结构分离；
- [ ] 记录输出触碰 volatility floor/ceiling 的比例；
- [ ] 对无文本样本验证文本分支严格为 0；
- [ ] 对不同 \(z\) 进行 latent sensitivity unit test。

## 21.2 Critic

- [ ] `critic_text_dropout = 0`；
- [ ] projection layer 使用 `bias=False`；
- [ ] projection branch 只输入 future delta 或 Stage A residual；
- [ ] 对文本特征应用 LayerNorm/L2；
- [ ] unconditional branch 和 conditional branch 分开记录；
- [ ] 记录 projection score 占总 score 的比例；
- [ ] 记录对 current/future surface 的输入梯度；
- [ ] 考虑 spectral normalization 或 R1；
- [ ] 防止 critic 容量远大于有效样本规模。

## 21.3 Gradient Penalty

- [ ] 对未 mask 的插值变量设置 `requires_grad=True`；
- [ ] 输入 critic 前应用 support mask；
- [ ] 梯度范数只计算有效网格；
- [ ] 处理每个样本不同 support；
- [ ] 对 support density 做归一化；
- [ ] 增加无效网格梯度为零的测试。

## 21.4 Mismatch

- [ ] 仅从 `has_text=1` 样本中选择 anchor；
- [ ] donor 也必须 `has_text=1`；
- [ ] 显式 derangement；
- [ ] 禁止同一文本 hash；
- [ ] 禁止同一事件；
- [ ] 禁止新闻转载重复；
- [ ] 增加 same-regime hard negatives；
- [ ] 单独报告 matched、random mismatch 和 hard mismatch score；
- [ ] 将 ranking loss 与 WGAN loss 分开记录。

## 21.5 Data Pipeline

- [ ] PCA 只在训练 fold 上拟合；
- [ ] 保存 PCA transform hash；
- [ ] 检查 text embedding normalization；
- [ ] 加入 metadata-only features；
- [ ] 审计发布时间、时区、收盘截点；
- [ ] 审计 overnight 新闻归属；
- [ ] 审计重复新闻；
- [ ] 审计 source concentration；
- [ ] 审计测试期数据是否参与 transform；
- [ ] 使用固定 split manifest。

## 21.6 Arbitrage

- [ ] 在价格或总方差空间计算约束；
- [ ] 对非均匀 strike grid 使用正确有限差分；
- [ ] 增加 call monotonicity；
- [ ] 增加 butterfly convexity；
- [ ] 增加 calendar consistency；
- [ ] 报告 violation rate 和 magnitude；
- [ ] refinement 前后分别评价；
- [ ] 考虑无套利 latent decoder。

## 21.7 Evaluation

- [ ] Energy Score；
- [ ] Variogram Score；
- [ ] grid/factor CRPS；
- [ ] 50/80/90% coverage；
- [ ] interval width；
- [ ] PIT/rank histogram；
- [ ] scenario covariance；
- [ ] effective rank；
- [ ] latent noise sensitivity；
- [ ] arbitrage violation；
- [ ] option portfolio repricing；
- [ ] hedge P&L；
- [ ] regime-level results；
- [ ] block bootstrap；
- [ ] equivalence test；
- [ ] 多 seed 完整分布。

---

# 22. 结论

当前文本条件 FiLM-WGAN 方案的核心研究问题是成立的：新闻文本可能包含当前 IVS 和传统市场变量之外的增量信息，而且这种信息可能影响未来曲面的 level、skew、term structure 和尾部情景。

但要将该方案发展为可信的实证研究或高水平论文，需要从“模型是否能拟合未来曲面”提升到以下四个更严格的问题：

1. **模型是否真的使用了 latent noise，并学习了条件分布，而不是条件中位数？**
2. **critic 是否识别了文本对未来曲面增量的关系，而不是当前市场状态、文本范数和新闻缺失模式？**
3. **matched-text 改善是否能通过 shuffled、metadata-only、FiLM-only 等对照被准确归因？**
4. **生成情景是否满足 calibration、依赖结构和静态无套利要求？**

最优先的修改不是扩大训练规模，而是：

- 修复 critic shortcut；
- 修复 dropout 和 support-aware GP；
- 重构 mismatch negatives；
- 引入 proper scoring rules；
- 建立严格的识别型消融；
- 将无套利约束纳入训练或 decoder；
- 降低模型容量；
- 强化 Stage A 的时间序列基线。

在这些问题得到解决后，模型的主要学术贡献可以定位为：

> 识别新闻文本对下一期隐含波动率曲面条件分布的增量预测价值，并提供兼顾概率校准、跨模态对齐与静态无套利的生成式预测框架。

---

## 附：当前代码与配置参考

- 模型结构：  
  <https://github.com/haobincui/wgan_option/blob/d9a9a07c122a6810a956719bb438b11e14271864/src/film_wgan/models.py>

- 训练逻辑：  
  <https://github.com/haobincui/wgan_option/blob/d9a9a07c122a6810a956719bb438b11e14271864/src/film_wgan/trainer.py>

- 损失函数：  
  <https://github.com/haobincui/wgan_option/blob/d9a9a07c122a6810a956719bb438b11e14271864/src/film_wgan/losses.py>

- 无套利 penalty：  
  <https://github.com/haobincui/wgan_option/blob/d9a9a07c122a6810a956719bb438b11e14271864/src/film_wgan/arbitrage.py>

- 当前代表性配置：  
  <https://github.com/haobincui/wgan_option/blob/d9a9a07c122a6810a956719bb438b11e14271864/configs/film_wgan/train_rq1_pair_textbase_15seed.yaml>
