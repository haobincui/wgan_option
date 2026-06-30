# RQ3 Event Study Calculation Logic

本文档解释 RQ3 的 event / no-event 后处理逻辑。当前 RQ3 pipeline 位于 `scripts/rq3/`，它不训练模型，也不修改 raw data；它只读取已有 volatility workbook、event calendar 或 generate-result summary，并输出 announcement-window vs quiet 的分组统计。

RQ3 对应论文计划中的问题：

```text
intraday/announcement event-study: announcement vs quiet + horizon
```

当前实现的核心目标是回答：

```text
在 policy announcement / major event 附近，current -> target IVS 的 5-minute 跳变是否不同于 quiet 样本？
如果已有模型输出，模型误差和 text improvement 是否在 announcement window 中不同于 quiet window？
```

## 1. Pipeline Overview

RQ3 当前有三个 CLI 子命令：

```bash
python scripts/rq3/main.py write-event-template ...
python scripts/rq3/main.py workbook ...
python scripts/rq3/main.py result ...
```

三者的用途不同：

- `write-event-template`: 生成空的 event calendar CSV 模板。
- `workbook`: 读取 `merged_vol.xlsx` 或 `merged_vol_rq2_text.xlsx`，按 event window 拆分样本，并计算 current -> target IVS jump。
- `result`: 读取已有 `generate_result/summary.csv`，按 event window 拆分模型结果，并统计模型误差指标。

最重要的是：`workbook` 模式不依赖任何训练结果，适合先完成 RQ3 的 data split、vol jump tracking 和 quality audit；`result` 模式用于之后把 text / no-text / baseline 模型输出接进同一个 event-study 框架。

## 2. Event Calendar

RQ3 不在代码中硬编码 FOMC 或 policy events。事件由 CSV 文件提供。

论文主口径使用 FOMC press release / policy statement release time，推荐路径是：

```text
data/reference/fomc_press_release_events.csv
```

该文件保存 2022-2023 FOMC statement release 的 UTC 时间，`event_type` 为：

```text
FOMC_PRESS_RELEASE
```

旧的模板兼容路径仍保留为：

```text
data/reference/fomc_events.csv
```

`fomc_events.csv` 可用于后续合并其他 FOMC-related events，但不是当前 paper-facing RQ3 主输入。

固定 schema 是：

```text
event_id,event_time_utc,event_name,event_type
```

示例：

```csv
event_id,event_time_utc,event_name,event_type
2022-11-02-fomc,2022-11-02T18:00:00Z,FOMC rate decision,FOMC
```

必需列：

- `event_id`
- `event_time_utc`

可选但推荐填写：

- `event_name`
- `event_type`

可以先生成模板：

```bash
python scripts/rq3/main.py write-event-template \
  --output data/reference/fomc_events.csv
```

然后手工填入真实 event time。所有时间都应使用 UTC，并写成 ISO format，例如 `2022-11-02T18:00:00Z`。

从当前版本开始，paper-facing RQ3 默认要求 event calendar 非空，并且要求至少一个样本落入 announcement window。如果只是做 quiet-only diagnostic，可以显式增加：

```bash
--allow-zero-announcement
```

## 3. Event Window Labeling

RQ3 使用 `news_timestamp_utc` 作为样本时间，与 event calendar 中的 `event_time_utc` 比较。

论文主口径采用 Nakamura and Steinsson (2018, QJE) 的 high-frequency monetary policy event-study 思路：围绕 FOMC statement / press release 使用 30-minute window，从 release 前 10 分钟到 release 后 20 分钟。因此 FOMC press release 的 paper-facing 主窗口是：

```text
[-10, +20] minutes around event_time_utc
```

对应 CLI 参数：

```bash
--pre-window-minutes 10 --post-window-minutes 20
```

代码仍然保留其他窗口作为 robustness / sensitivity analysis：

```text
--pre-window-minutes 0 --post-window-minutes 10  =>  Vergote-style post-release 10 minutes
--window-minutes 30  =>  [-30, +30] minutes
--window-minutes 3   =>  [-3, +3] minutes
```

对每个样本，代码计算：

```text
delta_minutes = (news_timestamp_utc - event_time_utc) in minutes
```

如果使用 Nakamura-Steinsson-style asymmetric window，则 event 条件是：

```text
-10 <= delta_minutes <= 20
```

一般化写法是：

```text
-pre_window_minutes <= delta_minutes <= post_window_minutes
```

如果没有提供 `--pre-window-minutes` / `--post-window-minutes`，则回退为旧的 symmetric window：

```text
abs(delta_minutes) <= window_minutes
```

则该样本被标记为：

```text
is_announcement_window = True
event_group = announcement
```

否则标记为：

```text
is_announcement_window = False
event_group = quiet
```

如果一个样本同时落入多个 event window，代码选择距离样本时间最近的 event。输出中会保留：

```text
event_id
event_name
event_type
event_time_utc
event_time_delta_minutes
event_window_mode
event_pre_window_minutes
event_post_window_minutes
```

## 4. Chronological Split

`workbook` 模式默认只分析 holdout subset：

```text
split = val
train_ratio = 0.8
```

这与当前 FiLM WGAN executable reality 一致：代码没有单独的 test split，而是按时间排序后取前 80% 为 train，后 20% 为 val。

具体步骤：

1. 按 `news_timestamp_utc` 和原始行顺序排序。
2. 计算：

```text
split_idx = int(total_samples * train_ratio)
```

3. 前 `split_idx` 行为 `train`，剩余行为 `val`。
4. 默认 `split=val`，因此 RQ3 默认分析后 20% holdout 样本。

也可以显式指定：

```bash
--split train
--split val
--split all
```

论文写作中，如果称为 test / holdout，需要说明当前代码层面对应的是 chronological validation holdout。

## 5. Workbook Mode: IVS Jump Metrics

`workbook` 模式读取 workbook 的 `gan_input_ready` sheet，默认 sheet name 是：

```text
gan_input_ready
```

可读取：

```text
merged_vol.xlsx
merged_vol_rq2_text.xlsx
```

必需列：

```text
news_timestamp_utc
current_surface_flat
target_surface_flat
strike_grid
maturity_days_grid
```

可选但会保留或用于 quality audit 的列：

```text
sample_id
global_index
current_snapshot_time_utc
target_snapshot_time_utc
pair_quality_label
current_weighted_iv_rmse
target_weighted_iv_rmse
training_candidate_flag
```

如果存在 `training_candidate_flag`，代码会优先使用 `training_candidate_flag == 1` 的行；如果过滤后为空，则保留原始行。

### 5.1 Surface Reconstruction

每一行中的 surface 是 serialized flat list。代码先解析：

```text
strike_grid
maturity_days_grid
current_surface_flat
target_surface_flat
```

然后恢复 surface shape：

```text
surface_shape = (len(maturity_days_grid), len(strike_grid))
```

因此：

```text
current_surface = reshape(current_surface_flat, surface_shape)
target_surface  = reshape(target_surface_flat, surface_shape)
```

这里的语义是：

- `current_surface`: backward/current IVS
- `target_surface`: forward/future IVS

也就是当前项目中的 5-minute forecasting pair。

### 5.2 Broad IVS Jump

对每个样本计算：

```text
diff = target_surface - current_surface
abs_diff = abs(diff)
```

然后输出三个 broad IVS jump metrics：

```text
surface_jump_mae = mean(abs_diff)
surface_jump_rmse = sqrt(mean(diff^2))
surface_jump_max_abs = max(abs_diff)
```

这些指标衡量的是真实 IVS 在 current -> target 之间发生了多大变化，不是模型误差。

### 5.3 ATM-Short Jump

RQ3 还单独跟踪 short-end near-ATM 区域，因为它更接近 policy announcement 后的短端市场反应。

ATM / short-end 的定义与现有 FiLM plotting helper 对齐：

```text
atm_idx = argmin(abs(strike_grid - 1.0))
short_idx = argmin(maturity_days_grid)
```

然后取：

```text
atm_short_current_vol = current_surface[short_idx, atm_idx]
atm_short_target_vol  = target_surface[short_idx, atm_idx]
```

输出：

```text
atm_short_signed_jump = atm_short_target_vol - atm_short_current_vol
atm_short_abs_jump = abs(atm_short_signed_jump)
atm_strike = strike_grid[atm_idx]
short_maturity_days = maturity_days_grid[short_idx]
```

注意：这里使用的是最近 ATM strike 和最短 maturity 的单点 volatility，不是 weighted short-ATM band。

## 6. Workbook Mode Outputs

`workbook` 模式输出目录默认在：

```text
outputs/rq3/rq3_<UTC timestamp>/
```

主要输出：

```text
rq3_labeled_samples.csv
rq3_group_summary.csv
rq3_event_summary.csv
rq3_event_vs_quiet_tests.csv
rq3_quality_audit.csv
rq3_run_manifest.json
plots/rq3_atm_short_event_cases/*.png
```

### `rq3_labeled_samples.csv`

一行对应一个样本，包含：

```text
sample_id
global_index
news_timestamp_utc
current_snapshot_time_utc
target_snapshot_time_utc
pair_quality_label
current_weighted_iv_rmse
target_weighted_iv_rmse
surface_jump_mae
surface_jump_rmse
surface_jump_max_abs
atm_short_current_vol
atm_short_target_vol
atm_short_signed_jump
atm_short_abs_jump
is_announcement_window
event_group
event_id
event_name
event_type
event_time_utc
event_time_delta_minutes
```

### `rq3_group_summary.csv`

按 `event_group` 汇总：

```text
announcement
quiet
```

对核心 jump metrics 输出 mean 和 median：

```text
surface_jump_mae_mean
surface_jump_mae_median
surface_jump_rmse_mean
surface_jump_rmse_median
surface_jump_max_abs_mean
surface_jump_max_abs_median
atm_short_abs_jump_mean
atm_short_abs_jump_median
atm_short_signed_jump_mean
atm_short_signed_jump_median
```

### `rq3_event_summary.csv`

只汇总 announcement-window 样本，按 `event_id` 分组，输出：

```text
event_id
event_name
event_type
event_time_utc
sample_count
first_news_timestamp_utc
last_news_timestamp_utc
surface_jump_mae_mean
atm_short_abs_jump_mean
```

### `rq3_event_vs_quiet_tests.csv`

对 announcement-window 和 quiet samples 做事件窗口差异检验。输出指标包括：

```text
surface_jump_mae
surface_jump_rmse
surface_jump_max_abs
atm_short_abs_jump
atm_short_signed_jump
```

方向固定为：

```text
announcement_minus_quiet > 0
```

含义是 announcement window 的 IVS jump 更大。表中包括 group sample counts、mean/median、Welch t-test p-value、one-sided event-greater p-value，以及 bootstrap percentile CI。

### `rq3_quality_audit.csv`

用于检查 quality filter 是否系统性排除 announcement 样本。它按：

```text
event_group
pair_quality_label
```

统计：

```text
sample_count
group_total
sample_rate
```

如果 announcement 样本的 poor-fit 或 excluded 比例明显高于 quiet，需要在论文 limitation 或 robustness 中说明。

### Case Plot

如果未指定 `--no-plots`，代码会为最多 `--max-case-events` 个 event 画图：

```text
plots/rq3_atm_short_event_cases/<event_id>.png
```

图中只包含：

```text
Current ATM-short IV
Target ATM-short IV
```

不包含模型预测。这一点很重要：当前 workbook mode 的图是 IVS jump tracking，不是 model prediction tracking。

## 7. Result Mode: Model Output Grouping

`result` 模式读取已有 `generate_result/summary.csv`，不重新生成预测，不训练模型。

命令格式：

```bash
python scripts/rq3/main.py result \
  --events-csv data/reference/fomc_press_release_events.csv \
  --output-dir outputs/rq3/rq3_results_$(date -u +%Y%m%d-%H%M%S) \
  --pre-window-minutes 10 \
  --post-window-minutes 20 \
  --result text=outputs/.../summary.csv \
  --result notext=outputs/.../summary.csv
```

每个 `--result` 使用：

```text
label=path
```

例如：

```text
text=outputs/training/film_wgan/svi-excel/.../generate_result/film_wgan_best/summary.csv
notext=outputs/training/film_wgan_notext/svi-excel/.../generate_result/film_wgan_best/summary.csv
```

`result` mode 要求 summary CSV 至少包含：

```text
news_timestamp_utc
```

如果存在以下指标，会按 `model_label` 和 `event_group` 汇总 mean / median：

```text
mae
current_mae
mae_gap_vs_current
win_flag_vs_current
short_atm_mae_gap_vs_current
atm_short_pure_mae_gap_vs_current
```

输出：

```text
rq3_result_labeled_samples.csv
rq3_result_group_metrics.csv
rq3_result_manifest.json
```

`result` mode 是之后计算 RQ3 分组 `Delta_text` 的接口。典型逻辑是比较同一 event group 内：

```text
text model metric - no-text model metric
```

当前脚本先负责统一 event labeling 和分组汇总；显著性检验、bootstrap 或 paired comparison 可以在这些 CSV 的基础上继续做。

## 8. Canonical Commands

生成 event calendar 模板：

```bash
python scripts/rq3/main.py write-event-template \
  --output data/reference/fomc_events.csv
```

填好真实 events 后，运行 paper-facing workbook mode。主窗口采用 Nakamura-Steinsson-style `[-10,+20]` minutes：

```bash
python scripts/rq3/main.py workbook \
  --merged-vol data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx \
  --events-csv data/reference/fomc_press_release_events.csv \
  --output-dir outputs/rq3/rq3_fomc_press_release_ns30_$(date -u +%Y%m%d-%H%M%S) \
  --pre-window-minutes 10 \
  --post-window-minutes 20 \
  --split val \
  --train-ratio 0.8
```

可选：对已有 model result 做 event/no-event 分组：

```bash
python scripts/rq3/main.py result \
  --events-csv data/reference/fomc_press_release_events.csv \
  --output-dir outputs/rq3/rq3_results_fomc_press_release_ns30_$(date -u +%Y%m%d-%H%M%S) \
  --pre-window-minutes 10 \
  --post-window-minutes 20 \
  --result text=outputs/training/film_wgan/svi-excel/20260417_180233/generate_result/film_wgan_best/summary.csv
```

## 9. Caveats / 注意事项

- RQ3 当前是 post-processing pipeline，不训练模型。
- `workbook` mode 测试的是真实 current -> target IVS jump，不是 prediction error。
- `result` mode 才用于模型误差和 text/no-text 分组比较。
- Event calendar 必须由外部 CSV 提供；代码不会自动知道 FOMC 时间。当前主文件是 `data/reference/fomc_press_release_events.csv`。
- 空 event calendar 或零 announcement 命中默认会报错；`--allow-zero-announcement` 只用于 diagnostic，不用于论文主结果。
- 当前默认 `split=val`，对应代码层面的 chronological validation holdout；论文中如称 test，需要说明 split 语义。
- Paper-facing 主窗口是 FOMC press release 的 `[-10,+20]min`，对应 Nakamura and Steinsson (2018, QJE) 的 30-minute high-frequency event-study 思路。
- Vergote-style `[0,+10]min`、`--window-minutes 3` 和 `--window-minutes 30` 是 robustness，不是当前主口径。
- 当前 ATM-short jump 是 nearest-ATM / shortest-maturity 单点，不是 weighted ATM-short band。
- 如果 `training_candidate_flag` 存在，workbook mode 会优先保留 `training_candidate_flag == 1` 的样本。
- `rq3_quality_audit.csv` 必须和 RQ3 结论一起看，避免 announcement 样本因 quality filter 被系统性筛掉而导致 selection bias。
