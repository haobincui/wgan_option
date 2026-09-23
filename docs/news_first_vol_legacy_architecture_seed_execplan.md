# Legacy 2×2 Generator×Critic 多 Seed 实验 ExecPlan

## 目标

在 exact-TTM 16×16、legacy widths、real_text、5m、LR=5e-7 的固定条件下，比较
`Concat/FiLM Generator × LP/NoLP Critic`。训练窗口为 `t<2023-07-01`，只在共同Q3
面板评价。Q4在本实验中禁止读取。

## 进度

- [x] 明确兼容边界：无release tag；新增branch-local CLI/schema，不修改旧接口。
- [x] 冻结2×2架构、3 seeds、anchor、统计门槛和外部reference SHA。
- [x] 完成独立orchestrator、analysis、report与测试。
- [x] 外部reference已增加本地静态快照与正式Q3 prediction/panel-universe绑定。
- [x] 完成独立CLI接线及安全默认`prepare`。
- [x] terminal manifest绑定不可变registry快照，并覆盖中断恢复与registry篡改拒绝。
- [x] 在全新临时root完成9/9真实dry-run。
- [x] 完成targeted tests、相关回归、Ruff、py_compile、diff-check与强制verification。
- [ ] 由主agent完成prelaunch QA并后台启动9个新增训练任务。

## 决策记录

- FiLM G + NoLP Critic 的legacy/5m/seed 42、202、404从正式容量实验只读引用；
  按job ID及artifact SHA锁定，不按时间戳扫描checkpoint；正式Q3 prediction与manifest
  直接冻结复用，不对external cell重新推理。
- 原始 `Concat G + LP Critic` 为anchor；其余三个cell构成Holm-3 family。
- 成功门槛为candidate−anchor平均MAE差<0、95% CI上界<0、Holm p<0.05，且
  至少2/3 seeds不劣。
- 不启动refit，不创建Q4窗口、loader、prediction或统计产物。
- Candidate−anchor Holm-3是primary；cell−persistence Holm-4和G/D/interaction
  factorial Holm-3分别是secondary family，不参与重新选择。

## 验证记录

- 最终代码对应的临时root：`/tmp/legacy_architecture_prelaunch_dryrun.0E7SAM/experiment`；
  9/9本地任务均为`dry_run_passed`，三份正式reference通过复验，Q4保持forbidden。
- 专用及相关回归共79项通过；Ruff format/check、`py_compile`和`git diff --check`通过。
- 强制verification脚本已执行；仓库没有`make format` target，因此在第一步以基础设施错误退出，
  未覆盖上述项目原生替代验证结果。

## 产物

- 9个本地训练任务registry/status与3个外部reference manifest条目。
- Q3 pair metrics、cell scores、三项candidate−anchor paired bootstrap/Holm结果。
- 冻结selection JSON、训练诊断、Markdown/HTML报告、QA与输出SHA manifest。
