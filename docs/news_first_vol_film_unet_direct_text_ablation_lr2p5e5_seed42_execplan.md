# FiLM LR=2.5e-5 单-Seed文本表示消融 ExecPlan

## 目标

在不重训已完成matched-LP模型的前提下，使用seed 42、四个rolling folds和统一
5分钟alignment，直接从相同随机初态训练以下四个FiLM U-Net + NoLP Critic条件：

- `lp_shuffle`
- `no_text`
- `bow`
- `sentiment`

四个新条件均固定Generator backbone LR=`5e-7`、text encoder LR=`2.5e-6`、
FiLM projection LR=`2.5e-5`、Critic LR=`5e-7`，最多240 epochs、min30、
patience20。正式矩阵是4 arms × 4 folds = 16 jobs；没有parent、continuation或
branch replay。

## 冻结比较基准

matched-LP不进入新训练registry。分析只读取并复验已完成high-LR seed42 root中
`film_lr_2p5e5`的四个fold checkpoint、500行pair metrics和训练诊断。配置冻结：

- 外部root的完整output manifest SHA
- pair metrics、training summary、checkpoint allowlist各自SHA
- 四个Generator checkpoint SHA

分析展示名固定为`lp_matched`。因此最终统计矩阵为16个新训练cell加4个冻结
reference cell，共5 arms × 4 folds、2,500行pair evidence。

## 数据与评价门禁

- exact-TTM 16×16、raw-joint、current-support-masked、identity residual
- train/validation由每个fold的半开区间创建；训练阶段禁止test loader
- 16个checkpoint全部冻结并写入allowlist之后才能打开test数据
- MC64 noise bank在每个新fold的四个arm之间共享
- 10,000次`fold → paired CME-session` bootstrap
- matched LP相对四个替代条件为预声明对比；结果仅标记
  `retrospective_rolling_development_single_seed_descriptive`

## 启动与恢复

正式root创建前先运行完整16-job一轮benchmark。两张GPU各8 workers通过
`<20 GiB/GPU`和RAM `<85%`门禁后冻结8/GPU；否则使用4/GPU、两wave。正式root
一旦prepare就不改变并发。supervisor使用PID、排他锁、journal和全SHA恢复校验。

```bash
nohup setsid env PYTHONPATH=src:. \
  /home/haobin_cui/.conda/envs/py312/bin/python \
  -m scripts.rq3.news_first_vol_film_unet_direct_text_ablation_lr2p5e5_seed42 \
  run-pipeline --resume \
  --config configs/rq3/news_first_vol_film_unet_direct_text_ablation_lr2p5e5_seed42.yaml \
  --output-dir outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_text_ablation_lr2p5e5_seed42_exact_ttm_rolling_v1 \
  > outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_text_ablation_lr2p5e5_seed42_exact_ttm_rolling_v1_control/pipeline.log \
  2>&1 < /dev/null &
```

## 验收

- 16 unique jobs、GPU 8/8、四arm共享相同seed42 G/D epoch-0 SHA
- 每个job的四组LR及0.1×floor写入并复验完整epoch trace
- no-text overlay逐位为0；shuffle无fixed point；BoW/scaler仅train拟合
- 16个新预测cell=2,000行，冻结LP reference=500行，合并=2,500行
- 外部reference任何路径、SHA、checkpoint或arm漂移均fail closed
- Ruff、py_compile、相关单测、`git diff --check`和项目强制verification完成

## 当前进度

- [x] 独立配置和16-cell contract
- [x] thin wrapper、冻结reference绑定和组合分析
- [x] 单元测试与ExecPlan
- [ ] CPU/静态verification
- [ ] 双GPU完整一轮benchmark（等待现有5-seed任务释放GPU）
- [ ] 后台正式流水线
