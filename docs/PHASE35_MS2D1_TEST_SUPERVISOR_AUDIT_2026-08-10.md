# Phase 3.5-MS2-D1 Synthetic Test Supervisor Audit（2026-08-10）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-10
- Verification Status: VERIFIED（产物、聚合、独立 bootstrap 与访问状态已复核）
- Overall Confidence: CONFIRMATORY FAIL AT PREDECLARED 20% MARGIN
- Version Label: phase35_ms2d1_test_supervisor_audit_v1
- Authorization Commit: `6665405`
- Linux Result Commit: `dc2939c`
- Evidence Scope: known-truth synthetic test；不是现场因果、闭环或真实迟延证据

## 1. Supervisor 最终判决

MS2-D1 状态为 **`CLOSED / TEST_NOT_CONFIRMED_AT_20PCT_MARGIN`**。

1. 18/18 test runs、run ledger、episode metrics 与 root ledger 完整；结构门禁和 oracle 正对照均通过。
2. learned-delay 相对 no-delay 的 test 点改善为 20.39%、21.70%、22.50%，方向与 validation 一致。
3. 预注册确认标准要求逐 seed、配对 episode、profile 分层 bootstrap 的 95% CI 下界均 `>=0.20`。实际下界为 0.1720、0.1845、0.1878，三 seed 均未通过。因此不能声称“至少 20% 的优势得到独立 test 确认”。
4. 期望迟延为 2.03–2.20 steps，但真值 ±1 step 质量仍只有 0.538–0.579。响应容量有效不等于 20 s 迟延被唯一辨识。
5. 按冻结停止规则，不重试、不调阈值、不增 seed，也不把 secondary candidate 的排序升级成事后主结论。

可守表述是：在该 synthetic pure-delay 设计中，显式迟延结构的改善方向跨 validation/test 稳定，但预注册的 20% 最小效应没有获得独立 test 的区间确认。

## 2. 独立复核结果

| 检查 | 结果 |
|---|---|
| canonical summary 重算 | 与 `summary_test.json` 逐字段一致；SHA256 `fc9f29397489207c4684a48489492f71b8e4a558dfb8bc89e112681b862335b0` |
| test run 数 | 18/18，3 seeds × 6 candidates |
| episode 聚合 | 18/18 `episode_metrics_test.json` 可精确重建对应 run metrics |
| trajectory 配对 | 同 seed 六候选共享同一 trajectory hash；不同 seed 的 hash 不同 |
| test 样本 | 每 seed 256 episodes；profile 为 hold/step/pulse/ramp/multi-step |
| 一次性访问 | root/run ledger 均 completed；重复执行被拒绝 |
| manifest 变化 | 相对授权版本只改变冻结允许的 test access/ledger/episode 字段 |
| 内容 pins | authorization、matrix、validation summary、checkpoint archive 均匹配 |
| 专项回归 | `pytest tests/phase35 -q`：96 passed（审计时仓库版本） |

内容 pins：

- authorization SHA256：`d1e04ce03ffeb2b0ab204f9c4131b352c8c58a10d049161d9e1be49755427ffe`
- matrix SHA256：`f5164bfb5f0e3e2d7d2dd0dfbb83934d8a573fb8a5ee874e4a71d9d11392f933`
- validation summary SHA256：`be29f9a3a9f4993c31cd8ebf8076f4f4b926c46e4d7ddf9eb321fd273faa6197`
- checkpoint archive SHA256：`7ee6393993d209cee255e8d7f09b7d376a135b63fc70572b0c6993d44e1a05f4`

root ledger 记录的 evaluation SHA 为 `61edbc7`，而授权 commit 为 `6665405`。两者在 matrix、runner、模型、训练与测试代码上无差异；`61edbc7` 只新增了一个随后撤回的 Linux 自审文档，因此数值执行与冻结协议代码等价。该偏差记录为 provenance advisory，不阻塞本次数值判决。远端未提交独立 stdout/stderr 文件；环境已写入 root ledger。以后仍要求完整回传命令日志。

## 3. 确认门禁

| seed | observed improvement | frozen 95% CI | independent NumPy 50k bootstrap | 判定 |
|---:|---:|---:|---:|---|
| 0 | 0.2039 | [0.1720, 0.2397] | [0.1719, 0.2390] | FAIL |
| 1 | 0.2170 | [0.1845, 0.2502] | [0.1848, 0.2513] | FAIL |
| 2 | 0.2250 | [0.1878, 0.2656] | [0.1878, 0.2647] | FAIL |

oracle clean NMAE 为 0.0206、0.0212、0.0223，逐 seed 小于 0.05，证明 test 生成—加载—评测链有效。它不改变主对比的 FAIL。

## 4. 异质性与时域诊断

- 非 hold profiles 的改善方向全部为正，无 Simpson 方向反转；step、pulse、ramp、multi-step 的幅度明显异质。
- pulse 改善最大，约 0.303–0.337；ramp 最弱，约 0.093–0.187。
- 改善集中在 H6/H18；H60 仅约 0.068–0.158。不能把结果扩写成整个 600 s 时域均有同等收益。
- hold 的 clean response 为零，只验证 reference identity，不提供模型优越性信息。

## 5. 统计谬误扫描（11/11）

| 风险 | 结论 | 处理 |
|---|---|---|
| Simpson's paradox | NOTE | profile 方向不反转，但保留异质性报告 |
| Ecological fallacy | CAUTION | synthetic 结论不外推现场 |
| Berkson's bias | N/A | 已知真值生成器无该选择机制 |
| Collider bias | N/A | 未按中介或结果筛 episode |
| Base-rate neglect | NOTE | profile 比例由人工均衡，不代表现场基率 |
| Regression to the mean | NOTE | validation 选 checkpoint；独立 test 已缓解 |
| Survivorship bias | PASS | 18/18 全纳入 |
| Look-elsewhere effect | PASS | primary contrast 与 margin 预注册 |
| Researcher degrees of freedom | PASS/NOTE | 内容 pins 与一次访问冻结；没有 retry |
| Correlation implies causation | CAUTION | 不能升级为现场 `do(valve)` |
| Reverse causality | N/A | synthetic 生成方向已知 |

## 6. 后续边界

D1 不提供把 learned-delay 直接传播到主架构的依据。MS2-D2 必须作为独立的结构阶次压力诊断：真值取消 pure delay、加入第三惯性极点，主对比是同预算三极点与二极点；learned-delay 仅检查遗漏惯性是否被误读成延迟。D2 的任何 synthetic 阳性仍不能证明现场三阶结构、真实喷水流量或完整世界模型。
