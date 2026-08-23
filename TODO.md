# Thermal World Model TODO

> 更新：2026-08-18。本文是项目唯一人工任务队列；机器状态见 `configs/phase3_5/experiment_registry.json`。RM3-B1 的 22/22 validation 已完成独立 paired audit 并关闭：1 项支持结构化简、5 项混合、2 项拒绝，不生成 RM3-B2。最终世界模型 pipeline 的本地接口包 `src/final_wm/`（observer/boundary/Fan2020-UDE transition/action-blind closure/observation/controller/装配层）已完成并通过 82 项本地合同与 micro-smoke 测试；下一判决点是 O1/B1/T1/R1/J1/K1 判别实验矩阵的冻结与独立授权提交。Linux 授权为空，test 与 MS4 继续锁定。

## 最终 Pipeline 组装（当前本地任务）

| 项目 | 状态 | 边界/下一步 |
|---|---|---|
| Ad hoc 孤立分支资产回收 | ✓ SELECTIVE SNAPSHOT | 已导入 `physical_models/fan2020_ude`；不含权重、轨迹、图片和日志 |
| 物理模型身份 | ✓ FROZEN AS CANDIDATE | Fan2020-inspired UDE transition，不是完全白箱或 plant truth |
| 已有证据链 | ✓ MAPPED | 区分已支持、部分支持和缺口；历史 PASS 不自动升级 |
| 最终 pipeline 架构 | ✓ DESIGN v0.1 | Observer + Boundary + Fan2020-UDE + action-blind closure + Observation + Koopman student |
| RM3-B1 原始结果 | ✓ AUDITED / CLOSED | 22/22 与 ledger 闭合；不生成 B2、不访问 test |
| 新模型正式包 | ✓ INTERFACES + MICRO-SMOKE | `src/final_wm/`：observer/boundary/transition/closure/observation/controller/model + 82 项本地测试通过；未授权长训 |
| 判别实验 O1/B1/T1/R1/J1/K1 | ✓ 侧A v0.2 判决已审计 → v0.3 修正案已生效 | 复算审计 11/11 全过；v0.3 冻结：R1 规则修订 + 物理修复批①→⑤立项（时滞/再湿契约/先验锚定/全锚定）+ gray-box 时序分工架构；侧B 延至 AE 阶段 |
| FMTS 2026 论文 | ❄ FROZEN-DRAFT | 初稿在仓（6pp+3图）但含已撤回数字；解冻条件见对齐审计 §5.1（证据链齐全前不补全） |

## 当前主线

完整顺序保持为：

```text
MS0 → MS1 → MS2-V/C/J → MS2-D1/D2/D3 → MS5 → MS3 → MS3-D → MS3-R(CLOSED)
                                                                    ↓
                                                  FINAL WORLD MODEL PIPELINE (ACTIVE)
                                                                    ↓
                                                               MS4 (HOLD)
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
| **MS3-R** | 点位辨识、分支归因与真实模型扩充 | ✓ **RM3-B1 AUDITED / CLOSED** | 对角化简支持；5 mixed、2 rejected；不生成 B2 |
| **Final WM** | Observer+Boundary+Fan2020-UDE+Closure+Observation | ✓ **INTERFACES + MICRO-SMOKE LOCAL PASS** | `src/final_wm/` 与 `tests/final_wm/` 完成；O1/B1/T1/R1/J1/K1 判别实验待独立提交授权 Hermes |
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

## MS3/MS3-D 结果与当前唯一任务：RM3-B 审计约束设计

Linux 12/12 runs 已由本地从 12 个 checkpoint、8,192-anchor episode 与 UTC-day bootstrap 独立重放。archive/trajectory/结构门闭合，test 未访问。冻结结论为：

```text
AUDITED / OBSERVATIONAL_VALIDATION_FAIL_ASYMMETRIC /
NO_RETRY / MS4_HOLD
```

B 动态平均绝对响应为 `0.04289–0.04851°C`，3/3 seeds 的 logged-vs-baseline/shuffled 日块 CI 下界均大于 0；A 仅 `0.00663–0.00854°C`，3/3 均未过 `0.02°C` non-collapse 门。B/A 动态效应比为 `5.03–7.32`，而 B/A 动作剂量中位数比仅 `1.052–1.059`。权威审计见 [MS3 Supervisor Audit](docs/PHASE35_MS3_SUPERVISOR_AUDIT_2026-08-11.md)。

MS3-D 的独立事件/日块复算误差为 0。主层为 A=41、B=42 个事件，各 19 日、17 个可配对日。B-A 的阀位响应差在 H300/H600 为 `+2.947 [+0.544,+5.486]` 与 `+2.627 [+1.655,+4.107] %/°C-SP`；局部温降、阀位归一化温降和 H600 末温主对比均跨 0。严格 600 s 且阀位稳定仅 A=1/B=3，主层另一回路安静仅 A=2/B=1，故不能升级成单侧 plant gain 或等价性结论。权威审计见 [MS3-D Supervisor Audit](docs/PHASE35_MS3D_SUPERVISOR_AUDIT_2026-08-11.md)。

MS3-R 采用三个批次级大门，避免逐小实验审批：Gate A 一次执行点位/时序/placebo/输入秩；Gate B 只做最后一轮点位闭合，不训练世界模型；Gate C 才执行真实模型筛查与正式比较。Gate B 主门固定为 UTC 日配对的 60/180 s `Tin-Tout` 正确路径减错侧路径、正滞后减 `|lead|`；2×2 MIMO、common/differential、不变性和 SP-IV feasibility 同批输出，后四者不能自动升级因果结论。完整设计见 [Gate B 设计](docs/plans/2026-08-11-phase35-ms3r-gateb-point-closure-design.md)。

RM3-AV 已完成。AV2 判决为 `SUPPORTED=30 / MIXED=3`：P5 terminal 优势主要由 bypass 贡献；action shield 可放大响应并改善侧别/时序 placebo，但损害 terminal/local MAE；4000 updates 不足；full MIMO 与三极点/迟延没有额外证据；C31 第二窗口递推低于 persistence。RM3-B1 因此只保留 C28/C29/C30 三个角色锚点与八个单模块配对，禁止全量堆叠。矩阵固定 22 units、统一 8000 updates；只有两个 folds 同方向且合同门通过的模块才可在本地审计后进入 B2。

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
| 最终 pipeline 接口、测试与 micro-smoke 已完成；下一步是冻结判别实验矩阵/预算后才能授权训练 | 当前无授权任务；不得继续执行 RM3-B1、生成 B2 或复用历史命令 |
| 只有本地可改 TODO、注册表和 Supervisor 文档并给出唯一审计判决 | 后续获新 commit 授权时只执行冻结矩阵并回传机器产物；不改代码/配置/阈值，不访问 test |

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
| 31 | RM3-A独立授权与30-run validation执行 | 本地授权 / Hermes执行 | ✓ 30/30 complete；旧18 runs未重跑 |
| 32 | RM3/RM3-A独立架构审计与审计意见实验化 | 本地 | ✓ RM3-AV设计冻结；审计意见均降为待证伪命题 |
| 33 | RM3-AV0恢复诊断、冻结推理消融与数据/反馈审计代码 | 本地 | ✓ RM2 54 + RM3/RM3-A 66 = 120 checkpoint/ledger闭合；11模式函数干预、rank/state/assumption ledger通过 |
| 34 | RM3-AV1 32候选×2 folds×seed0宽筛代码、矩阵与dry-run | 本地 | ✓ 64 units闭合；C00-C31逐候选一更新与完整产物smoke通过 |
| 35 | RM3-AV0/AV1单次批量执行与原始产物回传 | Hermes | ✓ AV1 64/64；AV0 120 checkpoint闭合；test未访问 |
| 36 | RM3-AV2 cache-free replay与逐项四态审计 | 本地 | ✓ 30 SUPPORTED / 3 MIXED；无冠军；授权关闭 |
| 37 | 按 AV2 输入清单重写 RM3-B paired composition 设计与 B1 runner | 本地 | ✓ 11候选、8配对、22 units；一更新与400项回归通过 |
| 38 | RM3-B1 单次 validation 执行与原始产物回传 | Hermes | ✓ 22/22；不得重试或生成B2 |
| 39 | RM3-B1 checkpoint/ledger/paired verdict 本地审计 | 本地 | ✓ 110+3 ledger闭合；1 supported simplification / 5 mixed / 2 rejected |
| 40 | 最终世界模型 transition/observer/boundary/closure 接口与 micro-smoke | 本地 | ✓ `src/final_wm/` 九模块 + 82 项专项测试通过；不授权长训 |
| 41 | O1/B1/T1/R1/J1 判别实验矩阵冻结（含 D0 数据合同与 D-SYN 门禁） | 本地 | ✓ 矩阵 v0.1 冻结；K1 条件 HOLD；待独立授权提交 |
| 42 | 判别矩阵执行代码（D0管道/训练/评估/runner）与本地 smoke | 本地 | ✓ `src/final_wm/{data,training,evaluation,diagnostics}.py` + `experiments/final_wm/run_matrix.py`；95 项专项测试通过；quick dry-run 验证过 |
| 43 | D0 数据审计整合与双侧桥接 | Linux执行+本地审计 | ✓ 14/14 通道 HIGH 闭合、四项质量门全过（82.6天×2侧）；`import_dual_canonical` 桥接交叉阀位映射 + 冻结 75/15/10 切分；96 项测试通过 |
| 44 | 首轮执行回传修复与矩阵 v0.2 修正 | 本地 | ✓ 执行侧修复复核接受（GPU搬运/R1 import）；R1 路径另修 2 个潜伏缺陷；runner 断点续跑+增量落盘；T1 预算统一 60/10 + 收敛诊断入 ledger；101 项测试通过 |
| 45 | 证据链对齐审计（证据链/路线图v1/论文初稿） | 本地 | ✓ 审计闭合：撤回口径×4、R1 规则 v0.3 提案、修复批①-⑤冻结、论文 FROZEN-DRAFT + 解冻条件 |
| 46 | 侧A v0.2 判决复算审计 + 本地证据包 | 本地 | ✓ 复算 11/11 全过（audit_verdicts.py）；auditpack 入仓并复现核心机制；IAPWS 网格已回传、provisional 全解除；论文 O1 写反已记录 |
| 47 | 矩阵修正案 v0.3（R1 规则修订 + 修复批①-⑤ + 架构路线登记） | 本地 | ✓ 已冻结生效；侧B 延至 AE 阶段；②③④ 已实施完毕（见 48），①随后 |
| 48 | 物理修复批 ②③④ 实施（混合时滞/再湿契约/先验锚定） | 本地 | ✓ 设计冻结+代码+契约测试；D-SYN 探针门禁通过；τ_mix 先验锚定 adhoc2 learnlag（80s）并锁可学习性 |
| 49 | Hermes 重跑失败修复（指纹/容差/quick覆写） | 本地 | ✓ 指纹纳入模型结构指纹（根因）；红项容差 1e-5（裁定A）；quick 档写 `_quick.json` 不覆写已审计产物；回执 results/final_wm/rerun_failure_response_20260820.md |
| 50 | 训练提速与归因纪律（用户反馈 15h + 改动捆绑） | 本地 | ✓ 子步抽取 `_substep` + runner `--compile`（aot_eager 逐位一致已验）；指纹加 git tree hash；split_runs 缓存（采样 59×）；并行实测 0.9× 撤回，分段计时入 ledger |
| 51 | seed0 重跑审计与 R1 门修复 | 本地+Hermes | ✓ leakage 伪影说接受并协议化（shuffle 对照，delta>5% 判据）；R1 seed0 暂定 PASS（待 seed1/2）；**已裁定 T1 减臂 closure_cons×3**（runbook 在回执追加4）；增益缺口重判：对名义线性上界 2.7-7.6×，阀门非线性+窄激励（phase1 D1 证据）下真实局部增益本就低，量级不入 R1 判据 |
| 52 | 世界模型声明证件清单 + CF 评测梯批准 | 本地 | ✓ 清单入档 results/final_wm/world_model_credential_checklist_20260821.md（A-G 七组 20 证，L0-L4 分级声明）；**修正口述错误：v0.2 真值 O1 hybrid REJECTED、B1 REJECTED、T1 closure SUPPORTED**；CF-2 待 R1 后立项 |
| 53 | FMTS 排期 + CF/D1 探针实施 | 本地 | ✓ CFP 核实（8/30 23:59 UTC=北京 8/31 07:59，4 页正文，欢迎负结果）；CF-1/3/4 + D1 探针落地（evaluation-only，无指纹影响），CF-1 接 dsyn、CF-3/4/D1 接 auditpack；倒排计划 docs/plans/2026-08-21-fmts-schedule-and-protocol-plan.md |
| 54 | R1 三 seed 判决审计：seed1 leakage 边际案 | 本地+Hermes | ✓ **案结：REJECTED 成立，不修门**。16 重 shuffle 零分布（std 0.2-0.3pp）证实 seed1 delta_vs_mean 5.15pp > 5pp 为真实泄漏（~17σ），非统计噪声；三 seed percentile 全 1.00（伪影地板之上普遍有真信号，仅 seed1 过门）；k=1 复现逐位一致。闭合审计 results/final_wm/leakage_marginal_case_closure_20260821.md；泄漏根因修复列入 AE 阶段 |
| 55 | 修复批①实施（五点锚定+观测器锚定相对化+干湿端点固定） | 本地 | ✓ H1 分解诊断定位双根因（出口锚不定 −18.1/−6.3°C；hybrid 融合拖偏 h 锚 +2.7/+1.5°C）；①-A 二分反演喷水侧状态 + **干湿混合零喷水湿漏 4.76% 端点修复**（实施中实测发现，~−6°C 偏差的真根因）；①-B 观测器改锚定相对修正（零初始化=精确锚，hybrid 仅慢状态，压力分段 22.064 MPa 软指示）；契约测试含反演往返/掩码/湿带阶跃；全套 128/128 + D-SYN quick 门禁过（59% > 30%）；残余登记 sh1_out −3.6°C 结构性（AE 候选）。重跑 runbook：results/final_wm/repair1_rerun_runbook_20260821.md |
| 56 | 修复批①重跑审计 + runner 完整性修复 + ③bis 提案 | 本地 | ✓ 审计 results/final_wm/repair1_rerun_audit_20260822.md：D-SYN 全量 PASS×3、T1 closure_cons×3 重训（H1 箱均值 5.3-13.3→0.35-1.15°C，①达标）、R1 REJECTED（seed0 28/32<1.0 门，**泄漏三 seed 全清白 0.55-1.09pp**——①消除泄漏签名）、**O1 陈旧无效**（legacy 无指纹续跑洞复陈 v0.2 判决，已修+回归测试，summary 冲撞同修）；**F3 再湿项幅值使 v1 下游反号**（aW 消融证伪：sh2_in +2.7→−0.3），③bis 消融臂提案待批；执行侧两笔热修复审通过；130/130。O1 重跑 runbook 已发 |
| 57 | 修正案 v0.4 实施（已批准）：再湿消融臂 + 判决实验 | 本地 | ✓ 实施+判决：`rewet_ablate` 配置位（默认 False 护③冻结语义）+ `_norew` 后缀解析（partition 误用被契约测试当场捕获，改 removesuffix）+ `closure_cons_norew` 证据臂（指纹隔离已验）+ 契约测试；runner 增 `--r1-arm`（独立块/文件不动冻结 r1）+ auditpack 非默认臂独立文件名；quick 档实证无区分度（2ep 双臂 ~10800）→ 判决档 10ep 对跑：**三判据全真采纳 norew**（intact v1 下游反号复现 sh2_in+4.63/final+2.31 @60s vs norew 全负；val NLL 3.494≤3.545+0.05；v2 更强 −0.454）；v04_rewet_decision.py 规则即代码，判决 JSON 入档；修正案文档 + 全档终审 runbook（norew×3+R1双栈+leakdist×3+auditpack）已发执行侧；132/132 |
| 58 | O1 重跑审计 + summary 权威版重建 | 本地+Hermes | ✓ 执行侧 1ab13c8 完成 O1 三臂×3（44eda5875 栈，9/9 新鲜、无续跑标记）：**learned SUPPORTED 3/3（+0.305/0.218/0.273）、hybrid SUPPORTED 3/3（+0.241±0.001）、steady 2.316-2.325**——①-B 锚定相对化把学习修正从 v0.2 净伤害翻转为稳定净收益，A2 证件 ❌→✅；F1 resume 洞三重闭环；ledger rebase 冲突按时间序并集解决（本地 quick 档 2 条留痕标注）；matrix_summary 权威版重建 r1 块（来自 r1_report.json，REJECTED：seed0 28/32 方向门、泄漏三清）；证件清单 O1 行更新 |
| 59 | v0.4 全档终审独立审计 | 本地+Hermes | ✓ 执行侧 07d6d91 完成 norew×3 全档+R1+leakdist+auditpack；本地逐项复核：**预注册判据全满足**（R1 norew SUPPORTED 3/3 方向 32/32×3——intact seed0 的 28/32 消失、泄漏 0.13-0.58pp 三清且 seed1 边际案不复存在；leakdist delta_vs_mean ≤0.6pp；auditpack 消融恒等自检逐位一致；val NLL 中位 1.272 vs intact 1.228）。**执行侧报告两处勘误**：intact 中位误引 1.260（实为 1.228，真差值 +0.044 非 +0.012，过门边际仅 0.006）；H18 MAE 数字无源（权威 metrics：norew 3.22/2.74/2.80 vs intact 2.65/2.79/2.62，均差 +0.23°C，不入终审门）。审计 results/final_wm/v04_final_adjudication_audit_20260823.md；**已裁定：A ✓ 生产臂切换 norew（runner 冻结路径不改，口径由裁定承载）；B=B1 补跑 physics_only×3 重发 T1 比较判决**（runbook results/final_wm/v04_t1_reissue_runbook_20260823.md 已发执行侧；判决用 runner 函数对 metrics 文件计算并留痕，不改冻结嵌套对；若要改 runner 默认比较臂另起 v0.5） |
| 60 | T1 比较判决重发（B1 回传审计） | 本地+Hermes | ✓ 执行侧 5fa5623 physics_only×3 新鲜（1.260/1.290/1.260，stop=cap×3，arm-filter 纪律保持）；设计侧用 runner 冻结函数只读重放，与执行侧预演逐位一致：**norew vs physics_only MIXED 1/3（seed0 −0.178 显著更差、seed1 +0.069 过门、seed2 −0.013）；intact closure vs physics_only REJECTED 0/3**——v0.2 "T1 SUPPORTED" 在修复①栈上被推翻，**第三负结果成立：闭包无显著精度增益**（锚定修复强化物理基线所致）；物理基线撞上限仍在降→判决对闭包已是宽松方向。审计 results/final_wm/t1_verdict_reissue_20260823.md；清单 T1 行已标推翻。**R1 physics_only 已跑（已批准）：REJECTED——seed1 瞬态 27/32 方向门失败**（稳态三 seed 全对，失败仅在 60s 控制相关瞬态；无闭包→leakage/residual_quantiles 语义空缺，runner 标记 skipped 最小修补+132/132 回归）。**三方终审闭合：physics_only 方向 REJECTED / intact closure 方向 REJECTED / norew 闭包 SUPPORTED 3/3——生产臂是唯一精度平价且方向持证配置**，闭包价值主张由响应保真救活（瞬态校正）；r1_report_physics_only.json 与 summary 块 r1_physics_only 入档 |

Linux 历史命令保留在 [experiments/phase3_5/README.md](experiments/phase3_5/README.md)，仅供复现历史批次；当前 registry 的 Linux 授权为空，任何旧命令都不得继续执行。

## 不可提前声称

- MS5 PASS 只证明冻结 synthetic policy correlation 下的 component recovery，不证明现场 free/response 分解；
- MS3 的 logged-action 增益仍可能来自串级 PID 反馈内生性，不能声称 `do(valve)`；
- 当前不能声称完整状态 simulator、可识别反事实、闭环可用或最终路线冠军；
- Koopman、PI-ODE、DeepONet 进入 RM2 稳定性比较，但单 seed/fold MAE 不得产生路线冠军；Fan 路线仍不进入当前矩阵；
- MS4、模型选择与论文在 MS3 本地审计前继续冻结。
