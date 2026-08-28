# Final-WM Task 1 与新增探针包复审（2026-08-28）

## 1. 结论

复审对象为 `c252c59`（Task 1：v0.7 required-evidence / fail-closed）以及其前置远端提交
`d198783`（执行侧 direction / event-window / plant-DiD 探针包）。结论如下：

- Task 1 的主设计成立；经本稿所列两项最小修正后，未发现需要改模型结构的大逻辑漏洞；
- v0.7 仍按设计保持 `INCOMPLETE`，直至后续 paired-NLL、leakage 和 support-domain 证据闭合；
- `d198783` 可作为诊断和协议设计依据，当前不可升级为论文级因果或反事实证据；
- test split 继续锁定，不授权 Linux 正式重跑，不自动修改论文结论。

## 2. 已修正的阻塞项

### A1. 定常稳定性初态混入历史瞬态

原 `constant_condition_stability()` 用 observer/history 生成初态，随后才冻结边界和动作，因而
可能把“最后一个历史点到第一个未来工况”的跳变计入漂移。J1 又用真实变化的未来边界与动作评价
“稳定性”，会惩罚正常动态响应并偏爱过度平坦的模型。

处置：T1/J1 统一使用首个未来条件、最后历史观测锚定的 `initial_steady_state`；边界和动作在整个
rollout 内冻结。删除不再需要的 observed-action `rollout_stability`，不增加新指标。

### A2. 本机 pandas 兼容阻断全量验证

`DatetimeIndex.as_unit()` 在本机 pandas 不存在，使 `test_data_v2.py` 的 9 项测试全部失败。
`DatetimeIndex.asi8` 对当前纳秒时间索引已直接给出纳秒整数，因此移除多余的 `as_unit("ns")`，
不改变时间戳换算语义。

## 3. 新增执行侧探针包的审计结论

### P1. Plant-DiD 置换 p 值不能报告为 0

代码和 JSON 以 `extreme_count / 200` 报告 Monte Carlo p 值；当 200 次均未越界时写成 `0.0`。
有限置换次数不能支持零概率。正式使用时应改为
`(extreme_count + 1) / (n_placebo + 1)`，并同时报告 `n_placebo=200`；当前最小可报告值应为
`1/201 ≈ 0.00498`。现有 JSON 不回写，保留为历史原始产物。

### P2. Factual counterfactual 的支持域只检查第一步

`direction_factual_probe.py` 的未来动作随真实轨迹变化，但 `in_support` 只检查 `step[i, 0]`。
因此“usable”并不保证整个 H18/H60 反事实序列处于该窗口历史动作支持域。正式重发时必须逐步检查，
并同时报告 step-level 与 whole-window support；修复前这些 PASS 只作诊断。

### P3. 反事实效应小于 factual rollout 误差量级

报告中的 H60 factual rollout MAE 约为 7–8 °C，而多数阀门反事实均值约为 0.1–0.2 °C。
成对积分可以抵消一部分共模误差，但这不足以证明效应量可信。论文使用前必须给出事件窗口上的
模型适配性/校准证据和支持域证据；在此之前只允许表述为“模型内部方向诊断”，不得表述为已验证的
plant causal effect。

### P4. DiD 的适用边界

独立复核脚本覆盖了事件数、控制污染、置换零分布和 Wilson 区间，方法具有诊断价值；但 matching
仍依赖 load、温度和单一 pre-trend 的可观测混杂控制，parallel-trends / no-unmeasured-confounding
不能由当前数据证明。故 DiD 只作为对象侧三角验证，不作为 world model 因果正确性的单独门禁。

## 4. 后续解决顺序

1. Task 2：正式 NLL 判决改为按日配对的绝对差 `ΔNLL = arm - baseline`，要求 CI 上界 `< 0`；
   CRPS/MAE 只报告相对效应量，不参与 NLL 符号敏感判决。
2. 后续 support-domain 任务统一逐步支持域口径，并修正有限置换 p 值；不为旧探针单独扩建框架。
3. 完成 leakage、数据质量、血缘绑定与全量本地验证后，才冻结 Linux v0.7 重发命令。

## 5. 验证记录

- `python -m py_compile`：`d198783` 新增的 7 个 Python 探针脚本全部通过；
- 定向回归：`30 passed`；
- 全量回归：`python -m pytest tests/final_wm -q` → `154 passed in 194.18s`；
- 本轮没有访问 test split、没有训练、没有修改 checkpoint 或历史结果 JSON。
