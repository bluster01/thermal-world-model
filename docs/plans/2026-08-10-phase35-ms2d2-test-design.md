# Phase 3.5-MS2-D2 One-Shot Synthetic Test Design

## Material Passport

- Material Type: content-addressed confirmatory test design
- Evidence Scope: `synthetic_order_pressure_test_not_field_causality`
- Upstream: D2 validation `AUDITED_SCREENING_PASS`
- Status: IMPLEMENTED / TEST_AUTHORIZED
- Test Boundary: 只读取原 21 个 validation-selected checkpoints；不重训、不访问 A/B、不启动 D3

## 1. 研究问题

在独立 synthetic test trajectories 上，显式三极点 graybox 相对同预算二极点的响应改善，其 95% CI 下界是否逐 seed 保持至少 10%，同时复现 oracle 与绝对误差门禁？

## 2. 备选方案

| 方案 | 优点 | 风险 | 决定 |
|---|---|---|---|
| A. 跳过 test，直接 D3 | 最快 | 把 validation selection 当确认性证据 | 拒绝 |
| B. 只评估 two/three/oracle 9 checkpoints | 最小计算 | 无法检查完整矩阵结构与 mechanism diagnostics 的 split 稳定性 | 不采用 |
| C. 评估全部 21 checkpoints，主门只含预注册三项 | 完整配对、成本仅推理、无事后选模 | 产物较多 | **采用** |

## 3. 冻结样本与统计单位

- checkpoints：归档中 7 candidates × 3 seeds = 21，全量内容寻址；
- test split：生成器已有独立 `test` offset，每 seed 256 episodes；
- action profiles：hold/step/pulse/ramp/multi-step；
- 配对：同 seed 的七个候选必须共享完全相同的 trajectory hash；不同 seed 不得复用；
- bootstrap：以 episode 为顶层单位，在 action profile 内有放回重采样，10,000 次；bootstrap seed 为 `20260813 + seed`；
- checkpoint selector 仍是历史 validation effect MAE，test 不参与任何选择。

## 4. 确认门禁

全部逐 seed判定：

1. 21/21 artifact 与结构门禁通过；
2. `d2_g3_oracle_structure` clean NMAE `<0.05`；
3. `d2_g3_three_pole` clean NMAE `<0.10`；
4. `d2_g3_three_pole` 相对 `d2_g2_two_pole` 的 paired/profile-stratified bootstrap 95% CI 下界 `>=0.10`。

任一主门失败即按冻结结果关闭，不重试、不改阈值、不补 seed。DeepONet、Koopman、PI-ODE 与 delay-compensation 不进入确认门，也不能因 test 排名升级。

## 5. 单列诊断

- three-pole/oracle 的 sorted-tau log-MAE；
- no-delay truth 下 delay-compensation 的 expected delay 与 zero-step mass；
- profile 与 H1/H6/H18/H60 异质性；
- validation→test NMAE 漂移。

诊断不改变主门。即使 test 通过，也只能确认该 known-truth order-pressure 设计中的响应优势，不能证明现场存在三个唯一物理状态，不能区分真实高阶惯性与输运迟延。

## 6. 一次访问与停止规则

runner 在生成首个 test metric 前写 root ledger；任一已有 root/run ledger、test metrics 或 test summary 都拒绝执行。执行中断后不删除 ledger 重跑，由 Supervisor 审计 partial 状态。test 完成后先回传原始产物，再由本地复算；D2 关闭前 D3/MS5 保持冻结。
