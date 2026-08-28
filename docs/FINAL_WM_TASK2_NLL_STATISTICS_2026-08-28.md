# Final-WM Task 2：NLL 判决统计口径（2026-08-28）

## 结论

v0.7 的 O1/T1/J1 正式 NLL 门已从百分比改善改为相同验证窗口上的绝对配对差：

`ΔNLL = NLL_arm - NLL_baseline`

每个窗口先计算差值，再按 UTC 日聚合并做 day-block bootstrap。单 seed 只有在 95% CI 上界
`< 0` 时才计为改善；完整单元仍要求至少 2/3 seeds 通过，并同时满足各自 continuity / stability
门。训练目标没有改动。

## 审计理由与处置

- NLL 可因量纲或密度基准改变而整体平移，百分比改善对该平移不具不变性；负或近零 baseline 还会被
  `clamp_min(1e-9)` 放大为无意义比例。因此删除 O1/T1/J1 的 NLL 百分比阈值判决。
- paired helper 强制 baseline/arm 的 tensor shape 与 day ids 完全一致；不一致立即 fail-closed。
- J1 原先 staged 与 joint 使用不同抽样 seed；现重新在 staged 的同一批窗口上评估 joint 后再判决。
- CRPS/MAE 保留相对改善及 CI，只作为正尺度上的实用效应量，不参与 NLL 门。
- 历史 `audit_verdicts.py` 明确限制为 v0.2 replay，并把旧阈值局部化；它不能审计 v0.7 新结果。

## 验证

- 加常数平移前后，`ΔNLL` 点估计、CI 和 seed pass 保持不变；
- 负 baseline NLL 下，`ΔNLL` 仍给出有限且方向正确的结果；
- `python -m pytest tests/final_wm/test_evaluation.py tests/final_wm/test_matrix_smoke.py -q`
  → `21 passed in 162.33s`；
- 未训练、未访问 test split、未改模型结构或历史结果。
