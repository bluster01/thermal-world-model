# Thermal World Model TODO

> 更新：2026-08-11。本文是项目唯一人工任务队列；机器状态见 `configs/phase3_5/experiment_registry.json`。MS5 已完成权重级审计，当前唯一授权为 Phase 3.5-MS3 A/B observational validation。旧 E1–E5 已废弃，Phase 4 暂停。

## 当前主线

完整顺序保持为：

```text
MS0 → MS1 → MS2-V/C/J → MS2-D1/D2/D3 → MS5 → MS3 → MS4 → 模型选择/论文
```

当前目的不是提前写论文，而是把已通过 synthetic component-recovery 门的完整模型迁移到真实 A/B：

\[
\widehat T_{1:H}=f_{free}(history,context)+g_{response}(context,a_{1:H},r_{1:H})
\]

MS5 已回答在冻结已知真值下动作响应不会被 joint `free` 分支吸收。MS3 现在检查真实观测数据中的条件预测、动作非坍缩和时间对齐；`free` 不读取未来动作，`g_response(c,r,r)=0`，开阀长期增益非正。喷水流量不作真值，现场实际阀位仅作为有效喷水作用代理；logged-action 增益仍不等于因果效应。

## Gate 总表

| Gate | 问题 | 当前结论/状态 | 下一动作 |
|---|---|---|---|
| MS0 | 统一多步响应合同 | ✓ CLOSED | 保持冻结 |
| MS1 | 同型 known-truth 可解性 | ✓ CLOSED | 不设路线冠军 |
| MS2-V/C/J | 非线性、工况调度和联合耦合 | ✓ CLOSED | joint 为 response 主训练；不外推 MS5 |
| MS2-D1 | pure-delay 压力 | ✓ 阴性关闭 | 不重试、不传播 delay 结构 |
| MS2-D2 | 三阶惯性压力 | ✓ test 确认 | 仅 frozen known-truth 响应优势 |
| MS2-D3 | colored nuisance 压力 | ✓ validation-only 关闭 | 不补 test；进入 MS5 |
| MS5 | 完整 `free+response` 动作吸收 | ✓ CLOSED | joint 选中；冻结 staged 协议拒绝 |
| **MS3** | A/B 真实数据适配 | ▶ **READY_FOR_LINUX** | 12-run validation-only；不访问 test |
| MS4 | SP→阀位→温度闭环响应 | ◻ 冻结 | 等 MS3；不恢复旧 E 匹配 |

## D3 收口

Linux 回传的 21/21 runs 已由本地重放：oracle clean NMAE `0.0357–0.0446`，三阶 `0.0558–0.0633`；三阶相对二阶的冻结 bootstrap 95% CI 下界 `10.8%–14.3%`，逐 seed达到 10% 门槛。独立 50k bootstrap 判决一致。按预算决定标签为：

```text
CLOSED / VALIDATION_STRESS_PASS / NO_TEST_BY_BUDGET_DECISION
```

它不是独立 test，不能支持现场扰动谱、现场唯一阶次或因果反事实。权威审计见 [D3 Supervisor Audit](docs/PHASE35_MS2D3_SUPERVISOR_AUDIT_2026-08-11.md)。

## MS5 收口

Linux 12/12 runs 已完成，本地从 checkpoint 重新生成 validation 并复算组件指标，最大差 `2.39e-7`；archive 21/21 成员闭合并可确定性重建。joint 每 seed response NMAE `0.047–0.050`、amplitude ratio `0.988–0.994`，全过；staged/joint total error ratio `11.14–14.11`，全失败；free-only 虽 total NMAE `0.082–0.086`，但 response NMAE=`1`、amplitude=`0`。最终标签：

```text
CLOSED / VALIDATION_ONLY_COMPONENT_RECOVERY_PASS /
JOINT_SELECTED / STAGED_PROTOCOL_REJECTED
```

权威审计见 [MS5 Supervisor Audit](docs/PHASE35_MS5_SUPERVISOR_AUDIT_2026-08-11.md)。

## 当前唯一任务：MS3

### 冻结 12-run 矩阵

| 候选 | side/seeds | 作用 |
|---|---|---|
| `ms3_joint_total` | A/B×seeds 0/1/2 | MS5 选中的真实适配主策略 |
| `ms3_free_only` | A/B×seeds 0/1/2 | prediction-only 负控 |

合计 12 runs；history=96、horizon=60，最多 40 epochs×100 updates。数据从冻结 SHA 的 `all_merged_10s.csv` 构造两个控制回路 cache：A阀→右(B)温、B阀→左(A)温。只运行 chronological train/validation，不访问 test。

### 判决规则

1. 12/12 manifest/history/checkpoint/episode/结构合同闭合；
2. joint/free logged MAE ratio `≤1.05`；
3. 动态阀位 support `≥512 windows` 且 `≥5 UTC days`；
4. 动态 response mean absolute effect `≥0.02°C`，全体最大绝对 effect `≤20°C`；
5. logged 相对保持阀位、相对置乱 delta-path 的 UTC-day bootstrap 95% CI 下界均 `>0`；
6. 每侧至少 2/3 seeds 过门，双侧均过才进入 MS4；
7. 任一失败原样回传，不补阈值、seed 或超参数扫描。

详细协议见 [MS3 设计](docs/plans/2026-08-11-phase35-ms3-real-adaptation-design.md)。

## 本地 / Linux 分工

| 本地 / Codex | Linux 远端 |
|---|---|
| 设计协议、写代码和测试、冻结矩阵、审计日志与重算统计、迁移状态 | 只在指定干净 commit 执行 README 第 17 节命令，原样提交结果 |
| 只有本地可改 TODO、注册表、阈值、结论和 Supervisor 文档 | 不改 `configs/src/experiments/tests/docs`，不挑 seed、不补跑、不后处理 summary |

## MS3 执行清单

| 顺序 | 任务 | 负责人 | 状态 |
|---:|---|---|---|
| 1 | MS5 checkpoint/episode/archive 独立审计与关闭 | 本地 | ✓ |
| 2 | 冻结 all_merged source SHA 与交叉 side mapping | 本地 | ✓ |
| 3 | cross-cache、joint/free 训练、runner、summary TDD | 本地 | ✓ |
| 4 | 专项测试、完整回归、compile、dry-run、状态检查 | 本地 | ▶ 收尾 |
| 5 | 生成两个 cache 并执行 12-run validation | Linux | ◻ 当前唯一远端任务 |
| 6 | checkpoint/episode/UTC-day bootstrap 独立复算 | 本地 | ◻ 等回传 |
| 7 | 若双侧 2/3 seed 过门，冻结 MS4 闭环模型验证 | 本地 | ◻ 不提前启动 |

Linux 唯一运行说明见 [experiments/phase3_5/README.md](experiments/phase3_5/README.md)。

## 不可提前声称

- MS5 PASS 只证明冻结 synthetic policy correlation 下的 component recovery，不证明现场 free/response 分解；
- MS3 的 logged-action 增益仍可能来自串级 PID 反馈内生性，不能声称 `do(valve)`；
- 当前不能声称完整状态 simulator、可识别反事实、闭环可用或最终路线冠军；
- Koopman、PI-ODE、DeepONet 与 Fan 路线不在 MS3 当前矩阵中重新赛马；
- MS4、模型选择与论文在 MS3 本地审计前继续冻结。
