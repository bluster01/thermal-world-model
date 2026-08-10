# Phase 3.5-MS 上下文恢复快照

> 更新：2026-08-10。新会话先读本文，再运行状态检查器。机器状态以 `configs/phase3_5/experiment_registry.json` 为准；本文解释为什么。

## 1. 恢复命令

```bash
git pull --ff-only origin main
python experiments/phase3_5/experiment_status.py --check --json
```

若命令中的 `active_gate`、`linux_authorized_gate` 或 deprecated track 与本文不同，先审计最新 commit，不凭聊天记忆继续实验。

## 2. 不可丢失的项目决定

1. Phase 4 暂停；先完成 Phase 3.5-MS 系列，不提前进入论文收口。
2. 旧 E1–E5 已废弃，只保留为历史失败与数据识别边界，不再作为当前 Gate、候选选择或主实验目录。
3. 现场 A/B 交叉链按现场事实冻结，不再验证；现场为串级 PID。
4. 喷水流量传感器不可靠；实际阀位是有效喷水作用代理，不是 kg/s。
5. 本地负责设计、代码、测试、冻结矩阵和审计；Linux 只在指定 commit 执行冻结命令并原样回传。
6. 完整顺序为 `MS0 → MS1 → MS2-V/C/J → MS2-D1/D2/D3 → MS5 → MS3 → MS4 → 模型选择/论文`。
7. MS2-J 的 response-internal staged 失败不能外推到 MS5；MS5 必须单独检验 `free+response` 耦合与动作吸收。

## 3. 当前可信实验状态

| Gate | 状态 | 可守结论 |
|---|---|---|
| MS0 | CLOSED | reference identity、prefix causality、state continuation 合同可测试 |
| MS1 | CLOSED | 同型二阶 known-truth 可解；存在 inverse crime，不设冠军 |
| MS2-V | CLOSED | learned monotone 在 R50 truth 下相对 identity 双层 PASS；`K/phi` 不可拆分 |
| MS2-C | CLOSED | scheduled K/τ 相对 global 双层 PASS |
| MS2-J | CLOSED | joint 相对两个单模块双层 PASS；staged 10% 非劣双层 FAIL；采用 joint |
| MS2-D1 | READY_FOR_LINUX | 当前 active/authorized Gate：20 s pure-delay validation；不授权 test |
| MS2-D2/D3 | PLANNED | 三阶惯性、未建模扰动，按序进入 |
| MS5 | PLANNED | 完整 `free+response` 耦合，不被 MS2-J 替代 |
| MS3 | PLANNED | A/B validation-only 真实数据适配，不称因果 |
| MS4 | PLANNED | 用 SP held-step 验证串级闭环响应，不恢复旧 E 匹配 |

## 4. Linux 最新同步与本地审计

Linux 在 `5260d3f` 完成 MS2-J test，27/27 root/run ledger 为 completed。基于每 seed 256 episode、按 action profile 分层、10,000 次 bootstrap：

- joint vs 两个单模块的改善为 0.77–0.88，95% CI 下界 0.73–0.89，PASS；
- staged/joint ratio 为 1.14–1.20，95% CI 上界 1.21–1.32（seed 0 下界附近不影响上界判决），非劣 FAIL；
- staged 相对 Stage A 改善 0.73–0.74，CI 下界 0.68–0.77，PASS。

本地已从 episode JSON 重新运行同一 bootstrap，三组 Gate 与 `summary_test.json` 逐字段完全一致。

### Provenance advisories

1. ledger 的 `evaluation_git_sha=78904ba` 不在远端对象库；可达的 pin-fix commit 是 `b170689`。冻结代码/产物一致，但以后 ledger 只能记录可达 SHA 或显式记录 rewrite alias。
2. `summary_validation.json` 的 Windows working-tree SHA 为 `788b1bc2…`，Git blob/Linux LF SHA 为 `212e53cc…`。这是 CRLF/LF 差异，不是内容语义变化。以后 JSON pin 使用 LF 或 canonical JSON；`.gitattributes` 已冻结 LF。

## 5. 当前下一步

MS2-D1 只加入固定 20 s pure delay。主对比是同一 scheduled monotone graybox 的 learned-delay vs no-delay；fixed-delay oracle 是正对照，Koopman/PI-ODE/DeepONet 是 secondary representation references。正式 validation 为 18 runs；本地 88 项 Phase3.5 测试、compile、CPU smoke、dry-run 与 fail-closed 汇总测试已通过，现仅授权 Linux validation，不授权 test。Linux 只能执行 [`experiments/phase3_5/README.md`](../experiments/phase3_5/README.md) 第 11 节。设计见 [`plans/2026-08-10-phase35-ms2d-pressure-design.md`](plans/2026-08-10-phase35-ms2d-pressure-design.md)。

## 6. 上下文读取优先级

1. `configs/phase3_5/experiment_registry.json`：机器状态与关键脚本；
2. 本文：项目决定、推理链与恢复命令；
3. `TODO.md`：当前人工任务队列；
4. 当前 Gate 的 design/implementation plan；
5. 各 Gate review：只提供结果，不自行改变后续授权。

历史 handoff、E 系列设计和论文草稿若与前五项冲突，按历史材料处理。
