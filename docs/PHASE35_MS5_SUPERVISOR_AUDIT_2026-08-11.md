# Phase 3.5-MS5 Supervisor Audit

- 审计日期：2026-08-11
- Linux 结果提交：`1fb6a23`
- 冻结执行提交：`af31495c8099062c60114dfb14647a53e9b9feb1`
- 协议：`phase3.5-ms5-v1`
- 最终标签：`CLOSED / VALIDATION_ONLY_COMPONENT_RECOVERY_PASS / JOINT_SELECTED / STAGED_PROTOCOL_REJECTED`

## 1. Material Passport

| 项 | 审计结果 |
|---|---|
| 研究问题 | 在冻结 synthetic `free+response` 真值和 context-policy correlation 下，仅用 total temperature loss 能否恢复两个组件 |
| 数据范围 | synthetic train/validation；未访问 synthetic test 或真实 A/B 数据 |
| 矩阵 | 4 modes × 3 seeds = 12 runs，12/12 完整 |
| 主选择规则 | oracle 必须先过；joint 全 seed 过门即选 joint；staged 只在 joint 失败时有资格候补 |
| 原始产物 | manifest、history、checkpoint、episode metrics、stdout/stderr、summary、checkpoint archive |
| 可复现性 | 从 checkpoint 重算 metrics；从 run JSON 重建 summary/archive；均与提交结果一致 |
| 证据范围 | synthetic validation 的组件恢复与协议选择，不是现场因果、反事实或闭环证据 |

## 2. 冻结主结果

| 模式 | total clean NMAE | free clean NMAE | response clean NMAE | response amplitude ratio | 判决 |
|---|---:|---:|---:|---:|---|
| component oracle | 0.0117–0.0126 | 0.0102–0.0112 | 0.0449–0.0458 | 0.991–0.998 | 3/3 PASS |
| joint total | 0.0139–0.0145 | 0.0127–0.0134 | 0.0473–0.0498 | 0.988–0.994 | 3/3 PASS，选中 |
| staged total | 0.1618–0.2051 | 0.1596–0.2048 | 0.1846–0.2639 | 0.840–0.990 | 3/3 FAIL |
| free only | 0.0824–0.0861 | 0.0366–0.0421 | 1.0000 | 0.000 | 负控成立 |

`staged/joint` total-error ratio 为 `11.14、13.09、14.11`，远高于冻结的 `1.10` 上限。因此当前冻结三阶段训练协议被拒绝。这个结论针对“hold-only free 预训练 → 冻结 free 训练 response → 低学习率联合恢复”的具体协议，不能推广成“所有分阶段训练都必然有害”。

free-only 的 total NMAE 可低至 `0.082–0.086`，但 response NMAE 精确为 `1.0`、amplitude ratio 精确为 `0`。这直接验证了本 Gate 的核心陷阱：总预测误差较低不等于动作响应组件已恢复。

## 3. 独立重放与产物审计

1. `summary_validation.json` 与 Linux 原始 `summary_stdout.log` 科学内容逐字节一致。
2. 12 个 manifest 均绑定执行 SHA `af31495...`，均记录 `test_accessed=false`。
3. 从 episode 文件独立聚合，最大绝对差约 `1.03e-7`。
4. 从 12 个最佳 checkpoint 在 CPU 重新生成 validation split 并重算指标，最大绝对差 `2.39e-7`。
5. 同 seed 四种模式共享相同真值轨迹哈希；三个 seed 的轨迹哈希互不相同。
6. checkpoint archive 含 21 个成员，成员哈希全部闭合；从原始 run JSON 重建 archive 后 SHA-256 仍为 `011ff92f480a0acfd0c49792499355661bb4bdfcdd361725de7e6c50188b8a69`。
7. train/summary 进程均 exit 0；无 traceback、OOM 或 non-finite 日志。

验证状态：`VERIFIED`。

## 4. 分阶段失败定位

冻结掩码和梯度路径工作正常：Stage A 只有 free 梯度，Stage B 只有 response 梯度，Stage C 两支均有非零梯度。失败不是“分支没有参与反向传播”或 summary 错误。

Stage A 只用 hold 子集训练 free，结束时 total/free NMAE 仍约 `0.21–0.22`；Stage B 的 response 可继续下降，但无法修复 free；Stage C 的低学习率联合恢复只将 total NMAE 降到 `0.16–0.21`。因此失败点是弱 Stage-A 初始化与恢复预算不足的组合。MS3 采用简单 joint 训练，不携带该 staged 协议。

## 5. 异质性与时域诊断

joint 的 response NMAE 在 step/pulse/ramp/multi-step 子组约为 `0.039–0.074`，没有子组方向反转；ramp 相对最难，但仍低于主 Gate 的 `0.15`。H6 相对误差个别 seed 超过 `0.10`，其真值尺度仅约 `2e-4–4e-4 °C`，因此只作 near-zero horizon 诊断，不改变冻结主判决。H18/H60 与主结论一致。

## 6. Linux 预检偏差

Linux 在 126 条预检中出现 1 条红项后继续执行，形式上偏离“全绿才运行”的远端协议。红项是 legacy `clean_effect` 原始 float32 bytes 的绝对哈希在 x86 与 aarch64 上因 `torch.exp` 最末 bit 不同；action 哈希一致，同一 aarch64 机器在 MS5 修改前后得到相同 effect 哈希，证明 MS5 对 legacy path 为 no-op。

该偏差不改变 12-run 科学产物：本地已完成权重级重算和 archive 重建。但测试已改为冻结 `1e-6` 量化后的跨平台哈希。未来远端遇到任一预检失败仍必须停止并回传，不得自行裁决继续。

## 7. 11 项统计谬误扫描

| 风险 | 结论 | 处理 |
|---|---|---|
| Simpson 悖论 | 低 | profile 子组无结论反转；保留 ramp 较难的诊断 |
| 生态谬误 | 高边界风险 | synthetic aggregate 不外推到现场 episode 或机组 |
| Berkson 偏差 | 低 | generator 未按 outcome 选择样本 |
| collider 偏差 | 低于本 Gate | 未按后处理变量筛选；真实闭环 Gate 仍需专门处理 |
| base-rate neglect | 中 | synthetic profile 近似均衡，不代表现场动作基率 |
| regression to mean | 中 | validation 用于 checkpoint 选择，故只称 development/validation 证据 |
| survivorship bias | 低 | 12/12 runs 全纳入，无弃 seed |
| look-elsewhere effect | 低 | 模式、seed、主门槛预先冻结；profile/horizon 仅诊断 |
| forking paths | 低 | 无超参补跑；跨平台哈希修复不改变模型或结果 |
| correlation→causation | 高边界风险 | known-truth 只支持 synthetic 机制，不支持现场 `do(valve)` |
| reverse causality | 不适用于 generator | 真实串级 PID 中仍是 MS3/MS4 的核心风险 |

统计置信度：对 MS5 协议选择为 `SOLID`；对任何现场科学外推为 `CAUTION / NOT ESTABLISHED`。

## 8. Supervisor 决策

1. 关闭 MS5 validation，不追加 synthetic test（按冻结预算决定）。
2. 选择 `ms5_joint_total` 作为 MS3 的训练策略；拒绝当前 staged 协议。
3. MS3 只做真实 A/B observational validation：检验条件预测、动作分支非坍缩和 A/B 一致性，不把 logged future valve 的预测增益解释为因果响应。
4. `do(valve)`、SP→阀位→温度闭环响应和真实反事实继续留给 MS4；在 MS4 前不能声称完整物理响应已成立。
