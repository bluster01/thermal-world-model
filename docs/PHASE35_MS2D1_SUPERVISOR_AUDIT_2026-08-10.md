# Phase 3.5-MS2-D1 Supervisor Audit（2026-08-10）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-10
- Verification Status: VERIFIED（validation 汇总、checkpoint、history 与归档均已独立复核）
- Overall Confidence: CAUTION（独立 synthetic test 尚未访问）
- Version Label: phase35_ms2d_d1_supervisor_audit_v1
- Training Commit: `95d1dbeebe23068e2a018edb44e6255a75faf6dd`
- Linux Return Commit: `7cf2b14`
- Evidence Scope: known-truth synthetic validation；不是现场因果、闭环或物理参数真值证据

## 1. Supervisor 判决

当前判决是 **`AUDITED_SCREENING_PASS / TEST_AUTHORIZED`**，不是 D1 最终关闭。

1. 18/18 validation 产物与结构门禁可复现；oracle 正对照 3 seeds clean NMAE 为 0.0201–0.0211，均小于 0.05。
2. learned-delay 相对 no-delay 的 validation 点改善为 20.25%–23.11%，逐 seed 达到冻结的 20% screening 门槛。
3. 但独立的逐 episode、action-profile 分层 bootstrap 诊断显示，3 seeds 的 95% CI 下界均未达到 20%：约 19.58%、17.26%、17.96%。因此 validation 只支持“值得进入一次性 test”，不能直接支持确认性结论。
4. learned simplex 的期望迟延接近真值（2.03–2.20 steps），但真值 ±1 step 的概率质量只有 0.538–0.579，低于 0.80。当前可守结论是“显式迟延容量改善响应”，不是“20 s 迟延被唯一辨识”。
5. D2 继续冻结。D1 只有在不重训、不改阈值、不增 seed 的一次性 synthetic test 通过后，才进入关闭判决。

## 2. 独立复核范围

| 检查 | 结果 |
|---|---|
| validation 汇总重算 | 忽略 `aggregation_git_sha` 与 Linux 后补的 archive 元数据后，逐字段一致 |
| checkpoint archive | 18 个预期成员，无绝对路径/`..`；tar 内 checkpoint SHA 与 manifest 全匹配 |
| checkpoint 内容 | protocol、candidate、seed、训练 SHA、operator/training/synthetic config 与 manifest 一致；参数全有限 |
| history | 18/18 epoch 连续、数值有限；按照 `score < best - 1e-8` 重放后，best epoch 与 checkpoint 选择一致 |
| 训练执行来源 | 18/18 manifest 均为 `95d1dbe`；冻结执行路径相对当前返回 commit 无差异 |
| test 隔离 | 18/18 `test_accessed=false`、`test_authorized=false`；无 test metrics/episode ledger |
| 运行完整性 | 18/18 完成；history 长度 232–300，9 个 run 达到 300 epoch cap |

归档 SHA256：`7ee6393993d209cee255e8d7f09b7d376a135b63fc70572b0c6993d44e1a05f4`。validation summary SHA256：`be29f9a3a9f4993c31cd8ebf8076f4f4b926c46e4d7ddf9eb321fd273faa6197`。

## 3. 结果解释

### 3.1 主对比

| seed | no-delay NMAE | learned-delay NMAE | 点改善 | validation 判定 |
|---:|---:|---:|---:|---|
| 0 | 0.05333 | 0.04101 | 23.11% | screening PASS |
| 1 | 0.06093 | 0.04859 | 20.25% | screening PASS（擦线） |
| 2 | 0.05202 | 0.04103 | 21.13% | screening PASS |

这里的 20% 是 validation 点阈值，不是置信区间阈值。checkpoint 又由 validation 选择，所以用同一 validation 宣布确认性优势会同时承受 sampling uncertainty 与 selection optimism。一次性 test 将冻结为：逐 seed、配对 episode、按 action profile 分层、10,000 次 bootstrap，要求 95% CI 下界均不低于 20%。

### 3.2 异质性诊断

- 非 hold profile 的改善方向均为正，没有观察到 Simpson 方向反转；但幅度明显异质：step 约 14%–24%，pulse 约 32%–37%，ramp 约 9%–15%，multi-step 约 14%–22%。
- hold profile 中两模型误差同为零，它只验证参考恒等性，不提供迟延模型优越性信息。
- 改善主要集中在短中期：H6 约 75%–83%，H18 约 56%–62%；H60 只有约 6%–12%，H1 因 20 s 真值迟延两者均为零。

因此 D1 的潜在正结论应限定为“显式迟延结构改善早中期动作响应重建”，不能扩写成整个 600 s 时域都等比例改善。

### 3.3 参数诊断

期望迟延接近 2 steps，但 simplex 权重分散在 0–4 steps。动力学时间常数、增益、有效开度映射和迟延核能够互相补偿；仅凭输出轨迹不能把分布式迟延核解释成现场唯一的输运迟延。这是结构可辨识性限制，不是通过更多 seed 就能自动消失的优化噪声。

## 4. 统计谬误扫描（11/11）

| 风险 | 结论 | 处理 |
|---|---|---|
| Simpson's paradox | NOTE | profile 方向未反转，但幅度异质；分 profile 保留 |
| Ecological fallacy | CAUTION | synthetic episode 结论不得外推现场机组/工况 |
| Berkson's bias | N/A | 已知真值生成器不存在入院式选择机制 |
| Collider bias | N/A | 本 Gate 不按结果或中介变量筛选 episode |
| Base-rate neglect | NOTE | 五类 action profile 固定覆盖；hold 不贡献主对比信息 |
| Regression to the mean | NOTE | 非极值前后对比；但 validation 选 checkpoint，故需独立 test |
| Survivorship bias | PASS | 18/18 全纳入，无失败 run 删除 |
| Look-elsewhere effect | PASS | primary 对比预注册；DeepONet 数值较低不事后升级为冠军 |
| Researcher degrees of freedom | NOTE | 矩阵/阈值/seed 已在训练前冻结；test 协议再次 content-addressed 冻结 |
| Correlation implies causation | CAUTION | synthetic known-truth 可解性不等于现场阀位 `do()` 因果效应 |
| Reverse causality | N/A | 生成器的 action→response 时序已知；现场逆因果问题留给 MS3/MS4 |

## 5. Provenance 修正

Linux 初稿 [`PHASE35_MS2D1_VALIDATION_REVIEW_2026-08-10.md`](PHASE35_MS2D1_VALIDATION_REVIEW_2026-08-10.md) 的“18/18 manifest 含 python/torch/cuda/device/platform”不正确。实际 manifest 只有 `device` 与 `torch_version`，没有完整 Python、CUDA runtime、CUDA availability、platform 字段；也未回传独立 stdout/stderr/environment snapshot。

这不改变数值复算与 checkpoint 完整性，但降低环境级复现实证的完整度。一次性 test runner 已把完整环境写入 root access ledger，并由汇总器 fail-closed 校验。

## 6. 一次性 test 决策

授权文件 [`ms2d_delay_test_authorization.json`](../configs/phase3_5/ms2d_delay_test_authorization.json) 固定：

- 原 18 个 validation-selected checkpoints，全量评估，不重训；
- 3 个原 seeds，每 seed 256 个独立 synthetic test episodes；
- oracle clean NMAE 每 seed `<0.05`；
- learned-delay vs no-delay 配对、profile 分层 bootstrap 10,000 次，每 seed 95% CI 下界 `≥0.20`；
- delay 参数恢复继续单列诊断，不并入响应确认门禁；
- 任一失败均按冻结结果报告，不重试、不调阈值、不增加 seed。

test 若通过，能够确认“在该 known-truth pure-delay 压力设计中，显式迟延模块相对同骨架 no-delay 的响应优势可复现”。它仍不能确认现场 20 s 迟延、真实喷水流量映射、串级 PID 闭环因果或完整世界模型。
