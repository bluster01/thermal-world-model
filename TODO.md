# Thermal World Model TODO

> 更新：2026-08-12。本文是项目唯一人工任务队列；机器状态见 `configs/phase3_5/experiment_registry.json`。RM3 已最终审计；RM3-A 双向容量匹配与local/terminal Pareto执行包已本地验证，冻结5个新配置×2 folds×3 seeds=30 runs，并复用旧18个锚点。Hermes尚未授权；test、自动科学 PASS 和 MS4 仍禁止。旧 E1–E5 已废弃，Phase 4 暂停。

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
| **MS3-R** | 点位辨识、分支归因与真实模型扩充 | ◉ **RM3 AUDITED / RM3-A LOCAL VERIFIED** | 30-run新矩阵已闭合；未授权Hermes，test/MS4 HOLD |
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

## MS3/MS3-D 结果与当前唯一任务：MS3-R Gate C 本地设计

Linux 12/12 runs 已由本地从 12 个 checkpoint、8,192-anchor episode 与 UTC-day bootstrap 独立重放。archive/trajectory/结构门闭合，test 未访问。冻结结论为：

```text
AUDITED / OBSERVATIONAL_VALIDATION_FAIL_ASYMMETRIC /
NO_RETRY / MS4_HOLD
```

B 动态平均绝对响应为 `0.04289–0.04851°C`，3/3 seeds 的 logged-vs-baseline/shuffled 日块 CI 下界均大于 0；A 仅 `0.00663–0.00854°C`，3/3 均未过 `0.02°C` non-collapse 门。B/A 动态效应比为 `5.03–7.32`，而 B/A 动作剂量中位数比仅 `1.052–1.059`。权威审计见 [MS3 Supervisor Audit](docs/PHASE35_MS3_SUPERVISOR_AUDIT_2026-08-11.md)。

MS3-D 的独立事件/日块复算误差为 0。主层为 A=41、B=42 个事件，各 19 日、17 个可配对日。B-A 的阀位响应差在 H300/H600 为 `+2.947 [+0.544,+5.486]` 与 `+2.627 [+1.655,+4.107] %/°C-SP`；局部温降、阀位归一化温降和 H600 末温主对比均跨 0。严格 600 s 且阀位稳定仅 A=1/B=3，主层另一回路安静仅 A=2/B=1，故不能升级成单侧 plant gain 或等价性结论。权威审计见 [MS3-D Supervisor Audit](docs/PHASE35_MS3D_SUPERVISOR_AUDIT_2026-08-11.md)。

MS3-R 采用三个批次级大门，避免逐小实验审批：Gate A 一次执行点位/时序/placebo/输入秩；Gate B 只做最后一轮点位闭合，不训练世界模型；Gate C 才执行真实模型筛查与正式比较。Gate B 主门固定为 UTC 日配对的 60/180 s `Tin-Tout` 正确路径减错侧路径、正滞后减 `|lead|`；2×2 MIMO、common/differential、不变性和 SP-IV feasibility 同批输出，后四者不能自动升级因果结论。完整设计见 [Gate B 设计](docs/plans/2026-08-11-phase35-ms3r-gateb-point-closure-design.md)。

Gate B 的四个冻结配对主门均通过：A/B specificity 日中位数为 `0.5149/0.3950`，simultaneous 97.5% 下界为 `0.4266/0.3033`；A/B timing 为 `0.5950/0.4590`，下界为 `0.5478/0.4020`。但 A 两个 family 各有 2/24 反向日；SP-IV partial R² 仅 `0.0141/0.0040`，末温 H600 错侧路径远大于正确对角。结论只升级到短时局部条件 MIMO，详见 [Gate B Supervisor Audit](docs/PHASE35_MS3R_GATEB_SUPERVISOR_AUDIT_2026-08-11.md)。

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
| 冻结假设/矩阵/代码；只跑micro-cache smoke；收到结果后做checkpoint/trajectory replay和唯一Supervisor判决 | 执行冻结RM2 54-run完整train/validation矩阵；按CUDA设备池并行，逐run一次attempt |
| 只有本地可改 TODO、注册表和 Supervisor 文档 | 只回传机器产物/日志；不改代码、配置、阈值、seed、fold，不访问test或启动MS4 |

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
| 11 | Gate A 批量 Linux 执行与一次性本地审计 | Linux/本地 | ✓ 条件通过；8/8产物闭合，rank复算误差≤1.55e-15 |
| 12 | Gate B 配对路径、MIMO、不变性与IV设计/代码 | 本地 | ✓ local_verified；4项专项、194项全回归通过 |
| 13 | Gate B 单次 validation 执行与 cache-free replay 审计 | Linux/本地 | ✓ 主门PASS；IV/末温路线不通过；授权关闭 |
| 14 | Gate C measured-boundary latent MIMO 架构与消融冻结 | 本地 | ✓ 双接口设计和实施计划已冻结 |
| 15 | Gate C contracts/data/model/training/synthetic/CLI 框架 | 本地 | ✓ 24项专项通过；全量回归见本地验证记录，未授权Linux |
| 16 | A1phys/LPV-Koopman/PI-ODE/DeepONet 路线特定实现与端到端合成恢复 | 本地 | ✓ 独立方程、held-out合成训练及free×excitation负控制通过；不替代真实值 |
| 17 | 四路线真实 1/100 RM0-A | 本地 | ✓ 审计为 underfit；四路线不可排名，禁止重跑同协议 |
| 18 | baseline-anchored + response-only action auxiliary RM0-B | 本地 | ✓ baseline有效；terminal无增益，response分解不唯一，禁止路线排名 |
| 19 | 固定 A1phys 的真实 RM1-A attribution | 本地 | ✓ 容量塌缩未观察；local supervision 必须保留；不设冠军 |
| 20 | RM2 日块响应与 rolling-fold 稳健性设计/代码 | 本地 | ✓ 54-run矩阵、九候选micro-smoke；真实cache 1-update smoke闭合 |
| 21 | RM2 完整train/validation并行训练与机器汇总 | Hermes | ✓ 54/54完成，无failure，test未访问 |
| 22 | RM2 checkpoint/archive/trajectory cache-free Supervisor审计 | 本地 | ✓ 220项ledger闭合；条件动作路径复现，operator gain未识别 |
| 23 | RM3 OOF nuisance residualization与正交响应矩校准设计 | 本地 | ✓ 合同/正交矩/R-loss/负控制通过 |
| 24 | RM3 joint-latent physical interfaces与公平预测表 | 本地 | ✓ 结构框架与输出域合同通过 |
| 25 | RM3真实H60统一数据适配、micro smoke与冻结矩阵 | 本地 | ✓ 六候选forward/backward；48-run envelope闭合 |
| 26 | RM3长训runner、selector/reporting与artifact ledger | 本地 | ✓ 36预测+12校准；六候选reduced-cache长训通过 |
| 27 | RM3独立Hermes授权提交与48-unit train/validation执行 | 本地授权 / Hermes执行 | ✓ 48/48 complete，exit 0，test未访问 |
| 28 | RM3 cache-free replay、NNLS修复与checkpoint补传 | 本地审计 / Hermes补传 | ✓ 168项ledger、36 checkpoint strict load闭合 |
| 29 | RM3-A P3/P4/P5容量匹配与local/terminal权衡消融设计 | 本地 | ✓ 双向容量匹配+两档Pareto，30新runs |
| 30 | RM3-A runner/reporting/完整产物合同与本地smoke | 本地 | ✓ 五候选one-update和dry-run通过；未授权Hermes |
| 31 | RM3-A独立授权与30-run validation执行 | 本地授权 / Hermes执行 | ▶ 等待授权；旧18 runs禁止重跑 |

Linux 历史命令保留在 [experiments/phase3_5/README.md](experiments/phase3_5/README.md)；当前均不构成运行授权。

## 不可提前声称

- MS5 PASS 只证明冻结 synthetic policy correlation 下的 component recovery，不证明现场 free/response 分解；
- MS3 的 logged-action 增益仍可能来自串级 PID 反馈内生性，不能声称 `do(valve)`；
- 当前不能声称完整状态 simulator、可识别反事实、闭环可用或最终路线冠军；
- Koopman、PI-ODE、DeepONet 进入 RM2 稳定性比较，但单 seed/fold MAE 不得产生路线冠军；Fan 路线仍不进入当前矩阵；
- MS4、模型选择与论文在 MS3 本地审计前继续冻结。
