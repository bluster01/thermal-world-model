# Final WM Task 4 逐样本反事实支持域审计

> 日期：2026-08-30
> 状态：`LOCAL VERIFIED / HOLD / TEST LOCKED`
> 范围：可信度修复计划 Task 4；不含 Task 5、正式训练、Linux 执行或论文 verdict。

## 1. 审计结论

Task 4 的主缺陷成立：旧 `action_support_from_history()` 把整个 batch 展平后求统一
阀位范围，因此一个窗口可以借用另一个窗口的动作支持域。旧 R1 step response、CF-1、
CF-3 和 CF-4 还直接调用 transition，绕过了 `model.counterfactual()` 的支持域门。

这会使“域内反事实”证据不可信，但不说明模型主体架构错误。本次只修正式评估路径，
不改 transition、observer、closure、训练目标、数据、阈值或历史实验产物。

## 2. 解决办法

- `ActionSupport.lo/hi` 改为逐窗口 `(B, 2)` tensor；每个窗口只使用自己的 history
  极值和冻结 margin，禁止跨窗口借域；
- `contains()` 强制 batch 对齐，并按待检 action 的 dtype/device 比较；CPU 合同已验证，
  CUDA 可用时执行同一条件测试；
- `model.counterfactual()` 保持默认拒绝任何越界；增加受形状检查的 `initial_state`
  参数，仅用于 Task 3 已冻结的 CF-1 replay abduction；
- R1 step response、CF-1、CF-3、CF-4 及再湿消融统一走
  `model.counterfactual()`，报告 factual/counterfactual 的逐步 mask、`support_rate`、
  `n_unsupported` 和 `allow_extrapolation`；
- R1 runner 为保留完整诊断显式采集越界轨迹，但任一正式 H18/H60 方向探针越界时，
  adjudicator 增加 `counterfactual_support_violation` 并将 verdict 降为 `INCOMPLETE`；
- D-SYN 已知 teacher 和 CF-3/CF-4 结构诊断可显式允许外推，但必须保存 mask。其域外
  数字只能标为合成/结构诊断，不能升级成现场经验支持或论文因果证据。

## 3. 验证

- 红测阶段：5 个预期失败，分别覆盖 batch 隔离、跨窗口借域、两条正式路径绕行和
  外推证据字段缺失；
- Task 4 定向回归：
  `python -m pytest tests/final_wm/test_contracts.py tests/final_wm/test_model.py tests/final_wm/test_cf_probes.py tests/final_wm/test_evaluation.py tests/final_wm/test_matrix_smoke.py -q`
  -> `56 passed`；
- 完整本地回归：`python -m pytest tests/final_wm -q` -> `168 passed`；
- device 归一化后合同回归：`tests/final_wm/test_contracts.py` -> `13 passed`；
- 未访问 test split，未运行正式训练，未生成 Linux 授权。

## 4. 结果边界与下一步

Task 4 证明的是支持域门和正式调用路径已经闭合，不会自动重判历史 R1/CF 结果。
尤其是显式外推产生的 CF-3/CF-4 数字仍需按支持率分层解释。下一项是 Task 5：确认
D-SYN teacher 参数扰动真实发生并可审计。在 Task 5–8 完成前继续保持
`ready_for_linux=false`、test locked；论文结论不自动升级。
