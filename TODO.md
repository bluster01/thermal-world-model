# Thermal World Model TODO

> 更新：2026-08-11。本文是项目唯一人工任务队列；机器状态见 `configs/phase3_5/experiment_registry.json`。MS3-D 已完成并审计：B 阀位响应更持久，但现场局部/末端热响应未支持 checkpoint 的 4.63 倍侧差。MS3-R Gate-A 框架已本地验证并签发单批 Linux 授权：仅运行 `ms3r_gate_a_v1`，一次 attempt、2 小时硬停、禁止自动重试和扩展实验；不访问 test、不启动 Gate B/C 或 MS4。旧 E1–E5 已废弃，Phase 4 暂停。

## 当前主线

完整顺序保持为：

```text
MS0 → MS1 → MS2-V/C/J → MS2-D1/D2/D3 → MS5 → MS3 → MS3-D → MS3-R → MS4(HOLD)
```

当前目的不是提前写论文，而是解释为什么同一完整模型迁移到真实 A/B 后只在 B 回路保留动作响应：

\[
\widehat T_{1:H}=f_{free}(history,context)+g_{response}(context,a_{1:H},r_{1:H})
\]

MS5 已回答在冻结已知真值下动作响应不会被 joint `free` 分支吸收。MS3 的真实观测验证现已完成：`free` 不读取未来动作，`g_response(c,r,r)=0`，开阀长期增益非正；但 A 侧 response non-collapse 3/3 失败。喷水流量不作真值，现场实际阀位仅作为有效喷水作用代理；logged-action 增益仍不等于因果效应。

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
| **MS3** | A/B 真实数据适配 | ✓ **AUDITED FAIL / ASYMMETRIC** | B 3/3 PASS；A 0/3 non-collapse FAIL；不重跑、不访问 test |
| **MS3-D** | A/B 响应不对称诊断 | ✓ **AUDITED** | 模型 A attenuation 未获现场热链路支持；B 阀位持久性更强；单侧 plant 归因不足 |
| **MS3-R** | 点位辨识、分支归因与真实模型扩充 | ▶ **READY_FOR_LINUX** | 仅 Gate-A 单批授权；190 tests PASS，无科学 PASS |
| MS4 | SP→阀位→温度闭环响应 | ◻ HOLD | MS3-R 冻结前不启动；不恢复旧 E 匹配 |

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

## MS3/MS3-D 结果与当前唯一任务：MS3-R Gate A

Linux 12/12 runs 已由本地从 12 个 checkpoint、8,192-anchor episode 与 UTC-day bootstrap 独立重放。archive/trajectory/结构门闭合，test 未访问。冻结结论为：

```text
AUDITED / OBSERVATIONAL_VALIDATION_FAIL_ASYMMETRIC /
NO_RETRY / MS4_HOLD
```

B 动态平均绝对响应为 `0.04289–0.04851°C`，3/3 seeds 的 logged-vs-baseline/shuffled 日块 CI 下界均大于 0；A 仅 `0.00663–0.00854°C`，3/3 均未过 `0.02°C` non-collapse 门。B/A 动态效应比为 `5.03–7.32`，而 B/A 动作剂量中位数比仅 `1.052–1.059`。权威审计见 [MS3 Supervisor Audit](docs/PHASE35_MS3_SUPERVISOR_AUDIT_2026-08-11.md)。

MS3-D 的独立事件/日块复算误差为 0。主层为 A=41、B=42 个事件，各 19 日、17 个可配对日。B-A 的阀位响应差在 H300/H600 为 `+2.947 [+0.544,+5.486]` 与 `+2.627 [+1.655,+4.107] %/°C-SP`；局部温降、阀位归一化温降和 H600 末温主对比均跨 0。严格 600 s 且阀位稳定仅 A=1/B=3，主层另一回路安静仅 A=2/B=1，故不能升级成单侧 plant gain 或等价性结论。权威审计见 [MS3-D Supervisor Audit](docs/PHASE35_MS3D_SUPERVISOR_AUDIT_2026-08-11.md)。

MS3-R 采用三个批次级大门，避免逐小实验审批：Gate A 一次执行点位/时序/placebo/输入秩，Gate B 一次执行分支归因/串级闭合/不变性/IV，Gate C 才执行真实模型筛查与正式比较。当前只实现 Gate A；`Tin-Tout` 是局部 plant-response 主证据，末温是延迟下游验证。分支只称“历史状态/未测扰动残差分支”和“受约束阀位响应分支”，不得凭 head 名称升级为燃烧或喷水物理模型。完整冻结设计见 [MS3-R 设计](docs/plans/2026-08-11-phase35-ms3r-response-identification-design.md)。

### 已执行的冻结 12-run 矩阵

| 候选 | side/seeds | 作用 |
|---|---|---|
| `ms3_joint_total` | A/B×seeds 0/1/2 | MS5 选中的真实适配主策略 |
| `ms3_free_only` | A/B×seeds 0/1/2 | prediction-only 负控 |

合计 12 runs；history=96、horizon=60，最多 40 epochs×100 updates。数据从冻结 SHA 的 `all_merged_10s.csv` 构造两个控制回路 cache：A阀→右(B)温、B阀→左(A)温。只运行 chronological train/validation，不访问 test。

### 已执行的判决规则

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
| 设计 MS3-R 分段/MIMO response-identification；冻结信息流、selector 和消融 | 当前无授权任务；不得重跑 MS3、访问 test 或启动 MS4 |
| 只有本地可改 TODO、注册表、阈值、结论和 Supervisor 文档 | 等本地冻结新矩阵/命令后才能恢复执行角色 |

## MS3 执行清单

| 顺序 | 任务 | 负责人 | 状态 |
|---:|---|---|---|
| 1 | MS5 checkpoint/episode/archive 独立审计与关闭 | 本地 | ✓ |
| 2 | 冻结 all_merged source SHA 与交叉 side mapping | 本地 | ✓ |
| 3 | cross-cache、joint/free 训练、runner、summary TDD | 本地 | ✓ |
| 4 | pandas 2/3 纳秒修复、专项/完整回归、compile、dry-run、状态检查 | 本地 | ✓ |
| 5 | 用 v1.1 覆盖旧 cache 并执行 12-run validation | Linux | ✓；summary exit 2 为科学失败 |
| 6 | checkpoint/episode/UTC-day bootstrap 与连续日块稳健性复算 | 本地 | ✓ |
| 7 | 记录 asymmetric FAIL，冻结重跑/test/MS4 | 本地 | ✓ |
| 8 | MS3-D 稳态 A/B 经验响应与 checkpoint IRF 对齐 | 本地 | ✓；独立复算通过 |
| 9 | 冻结 MS3-R response-identification、分支语义与三大门协议 | 本地 | ✓ |
| 10 | 实现并测试 Gate A 点位、placebo、residual excitation、结构信息流与输入秩框架 | 本地 | ✓；190 tests PASS |
| 11 | Gate A 批量 Linux 执行与一次性本地审计 | Linux/本地 | ▶ `ms3r_gate_a_v1` 单批已授权 |

Linux 历史运行说明保留在 [experiments/phase3_5/README.md](experiments/phase3_5/README.md)，当前不构成重复运行授权。

## 不可提前声称

- MS5 PASS 只证明冻结 synthetic policy correlation 下的 component recovery，不证明现场 free/response 分解；
- MS3 的 logged-action 增益仍可能来自串级 PID 反馈内生性，不能声称 `do(valve)`；
- 当前不能声称完整状态 simulator、可识别反事实、闭环可用或最终路线冠军；
- Koopman、PI-ODE、DeepONet 与 Fan 路线不在 MS3 当前矩阵中重新赛马；
- MS4、模型选择与论文在 MS3 本地审计前继续冻结。
