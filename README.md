# Thermal World Model

面向超超临界直流锅炉主汽温（SST）的数据驱动与灰箱世界模型研究。项目关注三个彼此独立的问题：预测是否准确、动作干预响应是否可信、模型是否适合进入预测控制或现场监督环节。

> **当前判断（2026-08-07）**：项目尚未完成模型定性，也没有尚未被开发流程访问的独立测试结论。M7 与 A1phys 只保留为 baseline；exp_112 是 selection-on-test 且 CFI 走了 fallback，只能作为该 Koopman `free_head` 的探索性负面信号，不能“关闭”它或 controlled Koopman 路线。Phase 4 优先把 Fan20-SST 作为 central physical candidate，只有通过闭合/可辨识门禁后，才嵌套 Fan17/21 组件并比较三条动态表达。

项目的权威状态与任务入口：

- [活任务队列](TODO.md)
- [Supervisor 总审](docs/SUPERVISOR_REVIEW_2026-08-07.md)
- [Phase 4 实验计划](docs/PHASE4_EXPERIMENT_PLAN.md)
- [项目状态](docs/PROJECT_STATUS.md)
- [历史任务说明](docs/CURRENT_TASKS.md)
- [远端实验协议](docs/REMOTE_EXPERIMENT_PROTOCOL.md)
- [文档地图](docs/README.md)
- [实验地图](experiments/README.md)
- [结果索引](results/README.md)

## 研究对象

- 数据：伊敏 6 号机，10 s 采样，约 70.7 万条记录，40 个数值变量。
- 目标：末级过热器出口汽温。
- 已发现的控制相关列：一级/二级减温阀相关量、`二级减温调节阀设定`、给水、燃料、负荷、压力和流量等；其 tag 语义、单位、测点层级与 Fan 变量映射仍待 DCS/工程核验。
- 主要困难：60–90 s 以上的纯迟延、约 600–1000 s 的慢热惯性、闭环运行数据中的共因混杂、宽负荷非平稳性，以及真实干预事件稀少。

## 当前研究进展

| 工作流 | 状态 | 当前结论 |
|---|---|---|
| 开环预测基线 | 已完成一轮 | Direct WM / M7 在旧协议下是较强的短窗预测 baseline，但需在 Phase 4 新 split 上重训；预测精度不能证明干预响应正确。 |
| MPC 仿真 | 已审计，降级为历史探索 | MPC 与 PID 使用同构神经世界作为 plant，且动作通道弱因果，不能据此宣称真实控制优势。 |
| 观测事件评测原型 | 正在重审 | 动作编码和跨时程指标已有积累，但 CFE 不是因果真值，现有 matching、test 选模和同名异量纲 CFI 仍需重建。 |
| A1phys 灰箱先验 | 候选基线 | `T_hat = f_free(x) + g(x,a)` 且动作分支满足 `g(x,0)=0`；它建模监督层 ΔSP 闭环响应，不是 Fan20 plant-level 物理模型。 |
| Koopman free-head | 探索性负结果 | exp_112 的 test-selected MAE 未优于 MLP free-head；CFI 无效，且该实现不代表 controlled/Fan-state Koopman。 |
| Fan 灰箱可微模型 | 尚未实现 | Fan20 SST 两级喷水焓值链是 central skeleton；Fan17 慢蓄热与 Fan21 mismatch/负荷调度作为组件，均待验证。 |
| 论文 | 方法与证据整理中 | 论文主线应强调“预测、干预和控制效用是不同层级”，模型选择仍开放。 |

## 不应混淆的两组“三路线”

仓库中已有 [exp_020_koopman_vs_gru.py](experiments/phase1_dynamics/exp_020_koopman_vs_gru.py)，比较了：

1. GRU decoder；
2. 受控 Koopman decoder；
3. 简化 Euler Neural ODE decoder。

这组实验使用早期 11 状态与阀位动作框架，只比较潜在解码器，**没有实现 Fan 的质量/能量守恒、焓值传递或宽负荷参数化**，因此不能作为物理路线的最终判决。

后续需要在统一协议下验证的可微动力学方向是：

1. **Fan-structured Neural ODE**：保留 `dx/dt = A(x) + B(x)u`、守恒项和可观测状态，用神经网络学习未知系数。
2. **Controlled Koopman**：在 Fan 对应状态/输出上学习受控线性潜在演化，而不是只替换预测头。
3. **时变灰箱混合模型**：以 Fan 2020/2021 的分段焓值、能量不匹配和负荷相关参数为骨架，使用低维神经修正或时变算子。

三条路线必须先冻结相同的 Fan 物理内容，再共享数据切分、输入输出、训练预算、预测指标、观测事件指标和物理门禁。否则“物理内容”和“动态表达”会被混为一个因素。详细门控见 [Phase 4 实验计划](docs/PHASE4_EXPERIMENT_PLAN.md)。

## Fan 2017/2020/2021 的角色

| 文献 | 可复用结构 | 在本项目中的状态 |
|---|---|---|
| Fan 2017 | 4 状态、3 输入、3 输出非线性 ODE；显式金属蓄热 | Fan20 已有制粉动态，只把显式金属状态作为组件候选，未实现 |
| Fan 2020 | 7 状态、两级喷水、逐段焓值传递、SST 物性关系 | 主汽温 central skeleton **候选**，须先过 Gate 1，未实现 |
| Fan 2021 | 宽负荷 CCS 的能量不匹配、节流损失、负荷相关参数；不直接建模 SST/喷水 | 作为 Fan20 宽负荷组件候选，未实现 |

当前变量表只证明“存在可能的列名对应”，不证明物理量可用。喷水质量流量不能由阀位直接替代，给水焓需要正确的给水压力，`T3` 测点位置仍需 P&ID 核验，`ut` 不得用结果相关的负荷变化率替代。Phase 4 需先完成单位、tag、时间对齐、物性与代理敏感性审计。

## 仓库骨架

```text
thermal-world-model/
├── README.md                         # 项目入口与最新口径
├── TODO.md                           # 唯一活任务队列
├── src/                              # 早期通用模型/数据代码；不是当前完整实现
├── experiments/
│   ├── phase1_dynamics/              # 预测基线、消融、早期 GRU/Koopman/ODE
│   ├── phase2_mpc/                   # MPC 探索与统一评测框架
│   └── phase3_feedforward/           # SP、CFE、A1phys 与当前因果架构实验
├── tests/                            # 现有评测协议回归测试
├── docs/
│   ├── PROJECT_STATUS.md             # 唯一的现状汇总
│   ├── SUPERVISOR_REVIEW_2026-08-07.md # 代码/论文/方法总审
│   ├── PHASE4_EXPERIMENT_PLAN.md      # 门控实验与统计计划
│   ├── CURRENT_TASKS.md              # 旧版任务说明，供历史追踪
│   ├── README.md                     # 文档导航
│   └── plans/                        # 整理/实施计划
├── results/                          # 已入库的实验摘要；模型 checkpoint 通常忽略
├── figures/                          # 论文与诊断图
├── archive/                          # 已作废协议和历史结果
└── data/                             # 本地数据入口；仓库不包含原始生产数据
```

本轮采用保守整理：历史脚本、结果与引用路径均保持不动。Phase 4 将新增隔离且 import-safe 的 `src/phase4/` 实现，不搬迁旧实验；主模型定性后再决定是否清理或迁移历史代码。

## 本地与 Linux 远端分工

本项目将“实验研发”和“正式算力执行”明确分开：

| 环节 | 本地工作区（本仓库） | Linux 远端 |
|---|---|---|
| 研究 | 梳理证据、提出假设、设计公平对照与判决标准 | 不临时改变研究问题或评测口径 |
| 实现 | 编写模型、训练脚本、测试、smoke 配置和结果 schema | 按指定 commit 与命令运行 |
| 验证 | 做静态检查、单元测试和小规模 smoke；审查数据泄漏与指标实现 | 使用真实数据/GPU 完成多 seed、长 epoch 正式实验 |
| 结果 | 读取日志和 JSON，复算指标、检查异常、形成结论并更新文档 | 回传 commit、命令、环境、日志、结果和退出状态 |

代码写完只到 `implemented`，本地 smoke 通过只到 `smoke_passed`；还需 manifest、固定 tag/commit、阈值、数据 hash 和 Supervisor 放行，才是 `ready_for_remote`。只有 Linux 结果返回并经过本地审计，实验才能标记为 `audited` 或 `concluded`。远端运行失败时不直接热修代码，应回传日志，由本地修复并生成新的 commit 后重跑。

完整交接格式见 [远端实验协议](docs/REMOTE_EXPERIMENT_PROTOCOL.md)。

## 关键代码与实验入口

| 文件 | 用途 | 定位 |
|---|---|---|
| [exp_025_unified_benchmark.py](experiments/phase1_dynamics/exp_025_unified_benchmark.py) | 统一预测基线与数据入口 | 广泛被后续实验复用 |
| [eval_protocol.py](experiments/phase2_mpc/eval_protocol.py) | MPC 协议原型与回归测试对象 | PID 方向/工作点/导数仍有 P0 缺陷 |
| [causal_eval.py](experiments/phase3_feedforward/causal_eval.py) | 动作构造、DiD 与 CFE 指标 | 历史原型；matching、量纲和 fail-closed 待修 |
| [causal_arch.py](experiments/phase3_feedforward/causal_arch.py) | A1/A3/B1、`g(x,0)=0` 与 Koopman free-head | 当前候选架构模块 |
| [exp_106_causal_arch.py](experiments/phase3_feedforward/exp_106_causal_arch.py) | 因果架构训练与比较 | 历史入口；逐 epoch 读取 test 选模 |
| [exp_112_koopman_full.py](experiments/phase3_feedforward/exp_112_koopman_full.py) | Koopman free-head 对照 | 探索性 pilot；CFI fallback，不作路线判决 |

## 主要证据

- 在该历史 supervisory tag 与旧协议的观测事件中，响应显示明显迟延；因此不应强制单步符号立即成立，但这既不是 plant-level 事实，也不是随机干预因果估计。
- Phase 2 审计表明：开环预测好不等于闭环可控，旧版 MPC 优势结论不能外推到真实锅炉。
- CFE 迭代修复了二阶差分动作和单末点评分等局部问题，但当前事件匹配不足以构成因果 ground truth，且 exp_106/112 仍以 test 逐 epoch 选模。
- exp_112 的三个 seed 中，每个 seed 都逐 epoch 在 test 上取最小 MAE 后再求均值：MLP 0.8467、Koopman free-head 0.8902、null 1.5467，Koopman−MLP 为 +0.0435 °C。所有 run 都在 50-epoch cap 前早停。所谓 0.869/0.821 则是对同 16 个 test 事件逐 epoch 取最大 fallback 分数后的均值，不是 P2 DiD/CFE，也不是独立测试结果。

完整证据分级见 [项目状态](docs/PROJECT_STATUS.md)。

## 数据与运行环境

历史实验约定从仓库根目录运行，并使用 Conda 环境 `Alloftime`：

```powershell
conda activate Alloftime
python experiments/phase3_feedforward/exp_106_causal_arch.py --help
pytest -q
```

以上命令需在数据路径和依赖配置完成后运行。当前 `pytest` 还存在一个既有收集错误：测试桩缺少 `TimeXerWM`；详情与修复任务已记录在项目状态和当前任务中。

注意：仓库尚无 `requirements.txt`、`environment.yml` 或 `pyproject.toml`，环境不可完全复现；这是当前工程任务之一。

`data/伊敏6号机` 在 Git 中是指向原作者 Linux 路径的符号链接。Windows 检出时它可能变成只含 `/home/bluster/Desktop/AI` 的普通文本文件，运行实验前需要将数据路径映射到本机实际 CSV。生产数据不应提交到仓库。

## 研究与实验规范

1. 不使用“最终模型”“路线关闭”等全局表述，除非明确限定到具体架构、数据、协议和指标。
2. 预测指标与干预指标分开报告；不能用 MAE 替代因果有效性。
3. 同一 task 内的候选路线使用相同切分、task-qualified 事件集、预算和多 seed 评测；Task P/S 不共用 action/event manifest 或排行榜。
4. 新实验必须有 `if __name__ == '__main__':`，结果目录不得覆盖其他协议。
5. 结论必须记录反例、失效边界和被推翻的旧口径。
6. 先更新根 [TODO](TODO.md)、[项目状态](docs/PROJECT_STATUS.md) 与 Phase 4 证据文档，再更新根 README。
7. 正式实验必须记录 Git commit、完整命令、数据指纹、seed、环境和输出目录；Linux 结果审计前不得写成项目结论。

## 相关文档

- [Supervisor 代码、论文与方法论总审](docs/SUPERVISOR_REVIEW_2026-08-07.md)
- [Phase 4 门控实验计划](docs/PHASE4_EXPERIMENT_PLAN.md)
- [Fan 三篇整合解读](docs/Fan_三篇整合_热工控制入门.md)
- [伊敏变量与 Fan 模型对照](docs/伊敏40列_vs_Fan模型变量对照.md)
- [Neural ODE、Deep Koopman 与 Koopa](docs/Neural_ODE_Koopman_三篇关键论文.md)
- [因果评测框架](docs/causal_eval_framework.md)
- [2026-08-06 审查记录（历史，部分结论已被总审撤回）](docs/session_2026-08-06_review_v2.md)
- [Phase 2 最终审查](docs/phase2_final_audit.md)

论文图表默认使用英文；研究记录与工程文档可使用中文。
