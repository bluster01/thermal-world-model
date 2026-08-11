# Thermal World Model

伊敏 6 号机主汽温数据驱动与灰箱世界模型研究。项目分别回答预测是否准确、实际阀门响应是否可信，以及这些证据是否足以支持控制应用。

> **当前判断（2026-08-11）**：Phase 4 暂停，先完成 Phase 3.5-MS 全系列，尚未进入论文收口。MS3 的 12-run A/B observational validation 已完成 checkpoint/episode 独立重放，结论为不对称科学失败：B 回路 3/3 seeds 通过，A 回路 0/3 response non-collapse 失败。不得重跑、降低门槛或只保留 B；test 与正式 MS4 继续冻结。当前唯一任务是本地 MS3-D：用稳态 SP→实际阀位→温度经验响应判断 A 侧是真实增益较弱还是模型 response absorption。Linux 当前无授权；旧 E1–E5 仍废弃。

## 当前入口

- [唯一活任务队列](TODO.md)
- [上下文恢复快照](docs/PHASE35_CONTEXT_SNAPSHOT.md)
- [机器实验注册表](configs/phase3_5/experiment_registry.json)
- [Phase 3.5 主线实验上下文](docs/PHASE35_MAINLINE_CONTEXT.md)
- [历史 E1–E5 实验设计](docs/PHASE3_5_EXPERIMENT_DESIGN.md)
- [Linux 执行手册](experiments/phase3_5/README.md)
- [项目状态与证据边界](docs/PROJECT_STATUS.md)
- [最终世界模型证据阶梯](docs/WORLD_MODEL_EVIDENCE_LADDER.md)
- [文档地图](docs/README.md)
- [实验地图](experiments/README.md)
- [历史结果索引](results/README.md)

Phase 4 的 [实验计划](docs/PHASE4_EXPERIMENT_PLAN.md) 和 Fan2017/2020/2021 路线保留为暂停的未来工作，不进入当前预算或论文判决。

## 当前主要实验目的

完整目标是验证结构化多步响应从“已知真值可解”到“结构失配稳健”、再到“完整 `free+response` 不吸收动作”、真实数据适配和现场闭环响应的一条连续证据链。D2/D3 与 MS5 已完成 synthetic 方法门；MS5 选择 joint，拒绝当前 staged 协议。MS3 进一步显示真实数据迁移不能由 synthetic PASS 直接保证：两侧预测非劣与动态 support 均过门，但只有 B 侧形成稳定的 logged-action 条件预测增益，A 侧响应分支明显偏弱。当前先完成不训练的 A/B 不对称诊断；这仍不是 `do(valve)`。完整顺序和停止规则见 [主线实验上下文](docs/PHASE35_MAINLINE_CONTEXT.md)，机器状态可运行 `python experiments/phase3_5/experiment_status.py --check --json` 查询。

## Phase 3.5 研究命题

现场喷水流量传感器不准，不能作为监督真值。项目采用实际二级减温阀反馈开度作为可审计代理，但不把 `%` 阀位伪装成 `kg/s` 流量。阀位到有效喷水作用允许是 A/B 侧独立的单调非线性映射：

```text
SP → 控制器/执行机构 → 阀门指令 → 实际阀位
                                      ↓ 单调非线性代理
                                有效喷水作用 → 温度响应
```

这也给出一个可检验解释：只用 Δ阀位会丢失绝对工作点；相同开度变化在不同基准阀位处可能对应不同有效喷水变化。因此 Phase 3.5 不只比较预测 MAE，还必须验证真实阀门事件的方向、迟延、剂量和模型 IRF。

证据边界是“闭环历史数据下的观测物理一致性”，不是随机干预因果效应。A1phys 是带方向与惯性约束的灰箱模型，不是质量/能量守恒方程模型。

## 历史 E1–E5（已废弃）

| ID | 问题 | 主对照 | 输出 |
|---|---|---|---|
| E1 | Δ阀位失败是否源于缺失绝对基准？ | Δ无基准 / Δ+基准 / 绝对阀位 | 预测误差与动作响应 |
| E2 | 阀位作用是否需要非线性和速率项？ | identity / fixed equal-percentage R=50 / learned monotone / learned+rate | 非劣预测、IRF-WMAE、剂量一致性 |
| E3 | 现场实际阀门是否产生可辨认温度响应？ | 隔离开关阀事件 / matched quiet controls | 二减与主汽温经验 IRF、日块 CI |
| E4 | A1phys 是否复现 E3？ | logged valve / constant-valve counterfactual | direction、lag、IRF-WMAE、dose monotonicity |
| E5 | SP 改变但阀位未执行应如何解释？ | 600 s no-execution / fast executed / ambiguous | 模型阀门效应、真实温度变化与层级证据 |

该表只记录旧路线及其失败原因，不再定义当前实验或论文主线。现场验证已迁移到 MS3/MS4：MS3 做 A/B validation-only 适配，MS4 使用 SP held-step 的串级闭环响应锚点。

## 最终目标与当前距离

世界模型不是一个更准的温度预测器。项目将最终能力拆成五个合同：状态/动作语义、独立预测、状态闭合仿真、可识别反事实和分级闭环。当前大致处于 `C0 PARTIAL + C1 DEVELOPMENT ONLY`：

- 当前模型不生成完整下一状态，也没有 30–60 min 自由递推稳定性证据，因此尚不是可验证的仿真器；
- 实际阀位处于 PID 反馈闭环，仍未建立开环 `do(valve)` reference；SP held-step 只提供闭环响应锚点，因此改变模型阀位得到的曲线仍只能称 action sensitivity；
- 旧 MPC 使用同构 plant 且协议存在缺陷，没有独立闭环效用证据；
- A1phys 是二阶惯性灰箱先验，没有质量/能量守恒、执行器标定和完整热状态闭合。

逐级缺口、实验门禁与停止规则见 [主汽温世界模型证据阶梯](docs/WORLD_MODEL_EVIDENCE_LADDER.md)。

## A1phys-V 架构

模型写成：

```text
T_hat = f_free(历史状态与历史阀位)
      + g_phys(未来阀位轨迹相对当前阀位的变化 | 历史工况)
```

关键约束：

- 自由预测头只读取处理前历史，不读取未来阀位；
- 未来阀位恒定在当前基准时，`g_phys = 0` 精确成立；
- 开阀的长期温度增益约束为非正，时间常数为正；
- 有效开度曲线单调且固定 `(0,0)`、`(100,100)` 端点；
- 预测增量直接使用 °C，避免归一化后物理量纲混乱；
- checkpoint 只按 validation 预测指标选择，物理指标作预注册门禁；
- test 由独立命令显式解锁，并写不可重复访问 ledger。

## 数据事实与口径

- 旧 A/B 原始文件是异步稀疏 historian CSV，旧框架用 causal LOCF 重建；MS3 改用 SHA 冻结的 377 列 `all_merged_10s.csv` 显式构造交叉控制回路 cache。源文件含 282 个非 10 s transition，窗口构造器硬排除所有跨缺口窗口。
- `二级减温调节阀设定` 是温度 SP，不是阀门开度设定；SP、指令、实际阀位属于不同控制层。
- 历史只读分析显示，在稳定工况的 SP 事件中约 4% 在 60 s 内实际阀位几乎不变；这可以由执行链时延、死区、限幅、控制器条件或 tag 层级解释，不能把 SP 直接当 plant action。
- 喷水流量只作诊断，不进入 loss、事件剂量、模型选择或核心结论。
- A/B 独立训练与报告，不能把两侧记录拼成独立样本扩大显著性。

原始生产数据不提交 Git。本地已知 A 侧路径为：

```text
C:\Users\14020\Desktop\时间预测模型\AA数据中心\伊敏12.10\merged_data\A侧主汽温全数据4.csv
```

B 侧位于同一目录；当前 MS3 数据源是 `merged_all_data/all_merged_10s.csv`。远端路径通过环境变量传入，不写进版本化配置。

## 仓库骨架

```text
thermal-world-model/
├── README.md
├── TODO.md
├── configs/phase3_5/experiment_matrix.json  # 42-run 开发矩阵与门禁
├── src/phase35/                             # 数据、模型、事件、训练、统计、汇总
├── experiments/
│   ├── phase1_dynamics/                     # 历史预测基线
│   ├── phase2_mpc/                          # 历史 MPC 探索
│   ├── phase3_feedforward/                  # 旧 SP/CFE/A1phys 原型
│   └── phase3_5/                            # 当前正式训练/评估入口
├── tests/phase35/                           # Phase 3.5 协议与模型测试
├── docs/PHASE3_5_EXPERIMENT_DESIGN.md
├── results/                                 # 正式结果返回后入库摘要
└── data/                                    # 本地数据入口，不含生产数据
```

旧 `phase3_feedforward` 保留用于历史追溯，不再作为正式训练入口。新实现隔离在 `src/phase35`，避免继承旧脚本的 test-selection、CFI fallback 和 ΔSP estimand。

## 历史 42-run 开发矩阵（已废弃）

```text
7 configs × 2 sides × 3 development seeds = 42 runs
```

配置包括 `free_only`、Δ无基准、Δ+基准、绝对 identity、固定等百分比 `R=50`、可学习 monotone、可学习 monotone+rate。该矩阵已经审计并废弃：不再补 seed、不打开旧模型 test，也不用于当前选模。`R=50` 只作为 exp_201 产生的工程先验对照，不代表流量真值。

## 本地与 Linux 分工

| 本地 / Codex | Linux 远端 |
|---|---|
| 设计实验、写代码与测试、冻结矩阵、审计事件/统计、复算结果和写论文 | 在固定 commit 上生成缓存、执行训练/评估、保存环境与日志并原样回传 |
| 修复必须形成新 commit；决定是否可进入下一 Gate | 不改模型、阈值、split、seed，不补挑 run，不自行下论文结论 |

详细命令见 [Phase 3.5 Linux 执行手册](experiments/phase3_5/README.md)。

## 本地验证

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest tests/phase35 -q
python -m compileall -q src/phase35 experiments/phase3_5
python experiments/phase3_5/experiment_status.py --check --json
python experiments/phase3_5/ms3_real_adaptation.py --dry-run
```

当前 Phase 3.5 专项测试覆盖 synthetic Gate、MS5 component recovery、跨平台确定性、MS3 交叉 cache/gap-aware anchor、joint/free 信息流、矩阵漂移拒绝、UTC-day 判决与 CLI smoke。最新通过数以仓库测试命令输出为准。旧 42 个真实数据 development runs 已完成但路线废弃；独立真实数据模型 test 未执行，A/B 旧事件 test 标签已在探索脚本中暴露，未来 MS4 正式事件证据必须使用冻结新时间块或明确内部验证边界。

## 历史证据限制

- M7 是旧协议下的强预测 baseline，不是已定性的最终模型。
- exp_106/112 逐 epoch 读取 test 选 checkpoint，不能作为独立 test 结论。
- exp_201 的 A 侧三 seed pilot 中，固定 `R=50` 的 ff10 变体 test-Jacobian 负方向为 100%×3，no-freeze 变体均值约 98.3%，优于原始绝对阀位的约 60–75%；但它使用手工曲线、test 逐轮评估和 test-Jacobian/CFI 选模，只能生成 Phase 3.5 假设。
- exp_112 所谓 CFI 由于 ground-truth 文件/事件数不匹配实际走 fallback；0.869/0.821 不可作为 P2 CFE 证据。
- Koopman free-head 的旧 MAE pilot 只否定该实现，不能关闭 controlled Koopman 路线。
- 旧 MPC 与 PID 比较存在同构 plant 与协议问题，不能外推现场控制优势。

完整审计见 [Supervisor 总审](docs/SUPERVISOR_REVIEW_2026-08-07.md) 和 [项目状态](docs/PROJECT_STATUS.md)。

## 论文可守表述

基于当前两条证据臂，可写：

> a physics-guided multistep action-response framework with synthetic known-truth validation and fail-closed identifiability auditing on industrial closed-loop data

这不是“已验证 world model”的同义改写。仍不能写“完全物理模型”“真实喷水流量已恢复”“随机干预因果效应已证明”“状态闭合 simulator 已成立”或“可直接进入闭环”。论文图表默认英文，研究与工程记录可使用中文。
