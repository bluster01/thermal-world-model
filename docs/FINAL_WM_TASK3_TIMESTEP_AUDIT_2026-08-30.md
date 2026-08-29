# Final WM Task 3 时序语义修复审计

> 日期：2026-08-30  
> 状态：`LOCAL VERIFIED / HOLD / TEST LOCKED`  
> 范围：可信度修复计划 Task 3；不含 Task 4 支持域、正式训练、Linux 执行或论文判决。

## 1. 审计结论

Task 3 的两个时序缺陷均成立，并已用最小修改关闭：

1. leakage probe 旧实现推进一次 `state_t -> state_{t+1}`，却使用第二个未来时刻的
   boundary/observation，并把 `state_{t+1}` 特征与第一个未来 action 配对，形成半步混合；
2. CF replay 旧实现用 `history.obs[-1]` 与 `history.boundary[0]`、`history.actions[0]`
   初始化，随后又重放完整 history，既不共时又重复消费起点。

统一语义现为：

```text
state_t + boundary_t + action_t -> state_{t+1} -> observation_{t+1}
```

本次修复只纠正评估语义，不改世界模型结构、训练目标、数据、阈值或既有历史结果。

## 2. 修复办法

- leakage probe 只采一个未来步；closure 特征取当前 `state_t, boundary_t`，action 取
  `action_t`，物理 transition 推进一次后以同一步的 `observation_{t+1}` 计算 residual；
- CF replay 用 history 第一个共时 observation/boundary/action 三元组初始化，只重放
  索引 1 以后的 boundary/action；单点 history 直接返回初始化状态；
- CF-1 增加 `baseline_mae` 与逐通道值，避免仅凭 delta 相等掩盖共同基线偏差；
- runner 将修正后的 leakage 证据接入 `leakage_v07`。Task 4 尚未完成，因此
  `support_domain_v07` 仍为空，正式 R1 按 fail-closed 合同继续为 `INCOMPLETE`。

## 3. 验证

- 定向回归：
  `python -m pytest tests/final_wm/test_cf_probes.py tests/final_wm/test_evaluation.py tests/final_wm/test_matrix_smoke.py -q`
  -> `31 passed`；
- 完整本地回归：`python -m pytest tests/final_wm -q` -> `164 passed`；
- toy transition 手算覆盖了 feature/action/residual 的同一步对齐；
- teacher==student identity 同时满足 `baseline_mae < 1e-6` 与 `delta_mae < 1e-6`；
- 未访问 test split，未运行正式训练，未生成 Linux 授权。

## 4. 结果边界与下一步

历史 leakage/CF 数字不因代码修复自动重判，也不能直接用于论文升级。Task 3 只证明评估
实现的时间索引已经自洽；逐样本反事实支持域仍未关闭。下一项严格按计划执行 Task 4，
在其完成前保持 `ready_for_linux=false`、test locked、论文冻结。
