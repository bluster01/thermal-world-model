# Phase 3.5-MS2-D2 Validation Supervisor Audit（2026-08-10）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-10
- Verification Status: VERIFIED（汇总、checkpoint、history、archive 与 validation episode bootstrap 已独立复核）
- Overall Confidence: CAUTION（validation screening 阳性；independent synthetic test 尚未访问）
- Version Label: phase35_ms2d2_supervisor_audit_v1
- Training Commit: `fa6933c718fda884e0a33d4f0371b4bd0cd54fc8`
- Linux Return Commit: `aedf1be5cfdf7482bde9a02b71611bb4e4875879`
- Evidence Scope: known-truth synthetic order-pressure validation；不是现场阶次、迟延或因果证据

## 1. Supervisor 判决

当前判决为 **`AUDITED_SCREENING_PASS / TEST_AUTHORIZED`**，不是 D2 已确认关闭。

1. 21/21 validation artifacts 与结构门禁通过；三阶 R50 oracle clean NMAE 为 0.0214–0.0226，逐 seed `<0.05`。
2. 三极点主模型 clean NMAE 为 0.0403、0.0520、0.0410，逐 seed `<0.10`。
3. 三极点相对同预算二极点的 validation 点改善为 25.18%、18.56%、28.10%，逐 seed超过预注册 10%。
4. 独立重建 validation episodes 后，配对、profile 分层 10,000 次 bootstrap 的 95% CI 下界为 20.73%、15.08%、22.28%，仍逐 seed高于 10%。这支持一次性 test 的功效预算，但 validation 同时参与 checkpoint 选择，不能替代 test。
5. 无 pure-delay 真值下，二极点+learned-delay 仍学习到 2.16–2.40 steps 的期望迟延，零步质量只有 0.241–0.297，且均值 NMAE 0.0456 接近三极点 0.0444。这是有限 horizon 下机制补偿/不可辨识证据，不授权把 delay 路线升级为已证实机制。

## 2. 独立复核范围

| 检查 | 结果 |
|---|---|
| Linux 写入范围 | 模型/配置/测试/注册表均未修改；新增一个越权 review，已降级为 `UNVERIFIED_REMOTE_REPORT` |
| 执行日志 | train/summary exit code 均为 0；21/21 顺序完成；无 NaN/OOM/traceback/warning |
| manifest | 21/21 execution SHA、matrix SHA、config、seed、环境和 test lock 一致 |
| validation 汇总 | 从归档恢复 checkpoint 后重跑；除 `aggregation_git_sha` 和 Linux 后附 archive 元数据外，科学字段逐字段一致 |
| checkpoint archive | 21 个安全相对路径成员；SHA256 与 manifest 全匹配；参数全有限 |
| checkpoint 内容 | protocol、route、seed、git SHA、operator/training/synthetic config、best epoch 与 manifest 一致 |
| history | 21/21 best-epoch 重放一致；长度 237–300，14 runs 到达 300 epoch cap |
| split 隔离 | 无 `metrics_test.json`、episode test metrics 或 test ledger；`test_accessed=false` |
| 环境 | Python 3.11.15、PyTorch 2.11.0+cu130、CUDA runtime 13.0、device=cuda |

内容 pins：

- matrix SHA256：`dfa01ad4124c452f3fd5de2f22b0d384f56041fab4837e7c7e5f05c23a854c26`
- validation summary SHA256：`4061353d9c4ef058a6db3dd969505452a3163a9fe957b13aee32251bd11ce701`
- checkpoint archive SHA256：`e8d6d8064fcef58ecb1da154727379fa781b2d80c63afc0aee02d1e2cc25c43f`

## 3. 主门禁与异质性

| seed | two-pole NMAE | three-pole NMAE | 点改善 | validation bootstrap 95% CI |
|---:|---:|---:|---:|---:|
| 0 | 0.05386 | 0.04030 | 25.18% | [20.73%, 29.81%] |
| 1 | 0.06383 | 0.05199 | 18.56% | [15.08%, 22.33%] |
| 2 | 0.05707 | 0.04104 | 28.10% | [22.28%, 34.30%] |

非 hold profiles 的改善方向均为正，没有观察到 Simpson 方向反转；但幅度异质。seed 0 的 ramp 只有 7.44%，其余 step/pulse/ramp/multi-step 为 12.25%–43.27%。改善集中在 H6/H18，H60 只有 13.64%–15.86%；H1 两模型均为零误差。确认性 test 继续使用整体配对 episode、profile 分层 bootstrap，不追加逐 profile 必须过 10% 的事后门禁。

## 4. 参数与表示边界

- 三极点主模型 sorted-tau log-MAE 为 0.133–0.149；oracle 为 0.169–0.191，均低于诊断阈值 0.35。
- 由于候选本身被固定为三极点，tau 诊断只能支持“给定三极点容量下近似恢复时间常数集合”，不能单独证明模型从数据中选择出了真实阶次。
- DeepONet validation 均值 0.0419 略低于三极点 0.0444，但它是预注册 secondary reference，且没有状态 continuation/物理参数合同，不能按 validation 榜事后升级为主路线。
- delay-compensation 的接近表现说明三阶惯性和分布式迟延在该 600 s 支持域内可互相逼近；不等于现场两个机制物理等价。

## 5. 统计谬误扫描（11/11）

| 风险 | 结论 | 处理 |
|---|---|---|
| Simpson's paradox | NOTE | profile 方向未反转，但保留幅度异质性 |
| Ecological fallacy | CAUTION | synthetic 结果不外推现场机组或设备状态 |
| Berkson's bias | N/A | 已知真值生成器无该选择机制 |
| Collider bias | N/A | 未按中介或结果筛选 episode |
| Base-rate neglect | NOTE | 五类 profile 人工均衡，不代表现场动作基率 |
| Regression to the mean | NOTE | validation 选 checkpoint；必须由独立 test 确认 |
| Survivorship bias | PASS | 21/21 全纳入，无失败 run 删除 |
| Look-elsewhere effect | PASS | 主对比和阈值预注册；secondary 不升级 |
| Garden of forking paths | PASS/NOTE | matrix/seed/gate 已冻结；下一 test content-addressed 冻结 |
| Correlation implies causation | CAUTION | known-truth 可解性不支持现场 `do(valve)` |
| Reverse causality | N/A | synthetic action→response 方向由生成器定义 |

## 6. 一次性 test 决策

授权原 21 个 validation-selected checkpoints 进入一次性 synthetic test，不重训、不调参、不增 seed。确认门禁逐 seed要求：oracle clean NMAE `<0.05`；三极点 clean NMAE `<0.10`；三极点相对二极点的配对、profile 分层 bootstrap 95% CI 下界 `>=0.10`。tau 与伪迟延继续只作诊断。test 审计完成前不启动 D3、MS5 或现场数据实验。
