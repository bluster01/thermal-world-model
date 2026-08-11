# Phase 3.5-MS2-D3 Validation Supervisor Audit（2026-08-11）

## Material Passport

- Material Type: synthetic validation audit
- Origin Skill: academic-research-suite / experiment-agent / validate
- Verification Status: VERIFIED for artifact replay and validation statistics
- Evidence Scope: `synthetic_colored_disturbance_pressure_validation_not_field_causality`
- Execution Commit: `040cb2791f1547007256561c59ea8c9e8e3343ea`
- Linux Return Commit: `f8a48ec`
- Decision: `CLOSED / VALIDATION_STRESS_PASS / NO_TEST_BY_BUDGET_DECISION`

## 1. 判决

D3 validation 的冻结主门逐 seed通过，因此按项目负责人 2026-08-11 的算力决策直接关闭，不再授权 D3 synthetic test，进入 MS5。该关闭只表示：D2 三阶 response advantage 在一个预注册的 action-independent stationary AR(1) output nuisance 下通过了 validation stress screen。它不是独立 test 确认，也不支持现场扰动谱、现场阶次、扰动 observer、状态闭合或因果反事实。

| seed | oracle clean NMAE | three-pole clean NMAE | 相对 two-pole 改善 | 冻结 10k CI | 独立 50k CI |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.04460 | 0.06170 | 15.15% | [11.54%, 18.95%] | [11.58%, 18.98%] |
| 1 | 0.03575 | 0.06334 | 13.87% | [10.76%, 17.21%] | [10.77%, 17.19%] |
| 2 | 0.03966 | 0.05578 | 17.76% | [14.30%, 21.34%] | [14.23%, 21.30%] |

冻结阈值为 oracle `<0.05`、three-pole `<0.10`、paired/profile-stratified bootstrap CI 下界 `>=0.10`；21/21 artifact 与结构合同也全部通过。seed 1 的 CI 下界仅比阈值高约 0.76 个百分点，因此结论应保留为边界通过，不写成强鲁棒性。

## 2. 可复现性与交付审计

- Linux 训练和汇总 exit code 均为 0；stderr 只有 21 条预期进度行，无 traceback、OOM 或非有限值。
- 21/21 manifest、history、metrics 和 episode files 存在；同 seed七候选 trajectory hash 一致，不同 seed互异。
- episode aggregate 独立重算与 metrics 最大绝对差 `3.35e-8`。
- 从 checkpoint archive 在临时目录还原 21 个 `.pt` 后，原 fail-closed summary 可完整重放并再次得到 `all_primary_gates_pass=true`。
- 独立 NumPy PCG64、50,000 次 profile-stratified paired bootstrap 的判决与冻结 10,000 次一致。
- Linux 提交未包含单独 `.pt`，但两个 tar 均含 21 个同名字、逐成员字节一致的 checkpoint。

### Provenance advisory

Linux 原始 `summary_stdout.log` 指向 `results/phase3_5/ms2d_disturbance/checkpoints_validation.tar`（SHA256 `3ea767dc...`）；提交后的 `summary_validation.json` 只把 archive path/hash 改为 `results/phase3_5/archive/ms2d_d3_checkpoints_validation.tar`（SHA256 `6864473c...`）。两个 tar 的 21 个 checkpoint 成员逐字节一致，科学指标完全相同，但该后处理越过了“远端原样回传、只写 output root”的协议。D3 不再开 test，故此项记为 provenance advisory，不为补容器元数据重跑训练；MS5 禁止远端改写 summary。

## 3. 非阻断诊断

- D3/D2 mean clean-NMAE ratio：three-pole 1.34、oracle 1.71，说明有色 nuisance 明显抬高恢复误差但未消除主优势。
- tau-set log-MAE：three-pole 0.180–0.188，oracle 0.238–0.262，均低于诊断阈值 0.35；不代表现场三个唯一物理状态。
- 无 true delay 时 learned-delay 仍给出 1.63–2.05 steps，零步质量 0.320–0.429，诊断继续失败；不得解释为现场迟延。
- step/ramp 的若干 seed 描述性改善低于 10%，H60 的 seed 0/1 仅约 6%–7%；逐 profile/horizon 从未是主门，不能事后改判，但必须限制“所有轨迹/所有时域稳健”的表述。
- secondary representation 排名不进入 Gate；D3 不产生路线冠军。

## 4. Fallacy Scan（11/11）

| 类型 | 判定 |
|---|---|
| Simpson's paradox | CAUTION：总体方向为正，但 profile/horizon 幅度差异明显 |
| Ecological fallacy | CAUTION：synthetic episode 不能外推现场机组或负荷段 |
| Berkson's paradox | N/A：生成器未按结果筛选 episode |
| Collider bias | N/A：主分析未条件化处理中介或结果 |
| Base-rate neglect | NOTE：五类 action profile 人工近均衡，不代表现场基率 |
| Regression to the mean | CAUTION：validation 同时参与 checkpoint 选择和最终 Gate |
| Survivorship bias | PASS：21/21 runs 全纳入；无失败 run 被删除 |
| Look-elsewhere effect | PASS/NOTE：主门预注册；secondary 与子组不升级 |
| Garden of forking paths | PASS：扰动强度、seed、阈值和 bootstrap 预先冻结 |
| Correlation implies causation | CAUTION：known-truth 生成器不建立现场 `do(valve)` |
| Reverse causality | N/A for generator；现场闭环问题仍未解决 |

## 5. 下游边界

D3 关闭后只放行 MS5 synthetic full `free+response` coupling。MS5 必须检验总预测看似准确时 response component 是否被 free head 吸收；MS2-J 的 response-internal staged 失败不能代替该实验。MS3、MS4、模型选择和论文仍冻结。
