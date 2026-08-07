# Thermal World Model

面向超超临界直流锅炉主汽温（SST）的数据驱动与灰箱世界模型研究。项目关注三个彼此独立的问题：预测是否准确、动作干预响应是否可信、模型是否适合进入预测控制或现场监督环节。

> **当前判断（2026-08-07）**：项目尚未完成模型定性。M7、A1phys、Neural ODE、Koopman 和 Fan 灰箱模型都不能被称为“最终模型”。最新实验只关闭了 **Koopman 作为 A1phys `free_head`** 的具体实现路线；Fan 2017/2020/2021 启发的可微物理模型仍未完成公平验证。

项目的权威状态与任务入口：

- [项目状态](docs/PROJECT_STATUS.md)
- [当前任务](docs/CURRENT_TASKS.md)
- [远端实验协议](docs/REMOTE_EXPERIMENT_PROTOCOL.md)
- [文档地图](docs/README.md)
- [实验地图](experiments/README.md)
- [结果索引](results/README.md)

## 研究对象

- 数据：伊敏 6 号机，10 s 采样，约 70.7 万条记录，40 个数值变量。
- 目标：末级过热器出口汽温。
- 可用控制相关量：一级/二级减温阀指令与反馈、主汽温设定值、给水、燃料、负荷、压力、流量等。
- 主要困难：60–90 s 以上的纯迟延、约 600–1000 s 的慢热惯性、闭环运行数据中的共因混杂、宽负荷非平稳性，以及真实干预事件稀少。

## 当前研究进展

| 工作流 | 状态 | 当前结论 |
|---|---|---|
| 开环预测基线 | 已完成一轮 | Direct WM / M7 提供较强的短窗预测基线，但预测精度不能证明干预因果正确。 |
| MPC 仿真 | 已审计，降级为历史探索 | MPC 与 PID 使用同构神经世界作为 plant，且动作通道弱因果，不能据此宣称真实控制优势。 |
| 因果评测框架 CFE | 已建立 | 已修复动作编码、测试区间泄漏、单一 600 s 末点评分等问题；采用 DiD 真值和跨时程 CFI。 |
| A1phys 灰箱先验 | 候选基线 | `T_hat = f_free(x) + g(x,a)` 且 `g(x,0)=0`，两级惯性先验表现稳定，但尚未和 Fan 方程路线公平比较。 |
| Koopman free-head | 当前实现已关闭 | exp_112 中 3 seeds × 50 epochs 未优于 MLP free-head；这不否定 controlled Koopman 或 Fan-state Koopman。 |
| Fan 灰箱可微模型 | 尚未实现 | Fan 2017 基础 ODE、Fan 2020 SST 两级喷水焓值链、Fan 2021 宽负荷能量不匹配/时变参数仍待验证。 |
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

三条路线必须共享数据切分、输入输出、训练预算、预测指标和 CFE 干预指标，才有可比性。详细验收条件见 [当前任务](docs/CURRENT_TASKS.md)。

## Fan 2017/2020/2021 的角色

| 文献 | 可复用结构 | 在本项目中的状态 |
|---|---|---|
| Fan 2017 | 4 状态、3 输入、3 输出非线性 ODE；金属蓄热与状态相关传热 | 已精读，未实现 |
| Fan 2020 | 7 状态、两级喷水、逐段焓值传递、SST 物性关系 | 已精读，与主汽温任务最直接，未实现 |
| Fan 2021 | 宽负荷、能量不匹配、节流损失、时变参数 | 已精读，未实现 |

数据变量对照表显示大部分状态可直接获得或通过 IAPWS-IF97 间接计算；主要缺口是汽轮机调门开度 `ut`。在建立可微模型前，仍需完成单位、时间对齐、焓值计算和代理变量有效性验证。

## 仓库骨架

```text
thermal-world-model/
├── README.md                         # 项目入口与最新口径
├── src/                              # 早期通用模型/数据代码；不是当前完整实现
├── experiments/
│   ├── phase1_dynamics/              # 预测基线、消融、早期 GRU/Koopman/ODE
│   ├── phase2_mpc/                   # MPC 探索与统一评测框架
│   └── phase3_feedforward/           # SP、CFE、A1phys 与当前因果架构实验
├── tests/                            # 现有评测协议回归测试
├── docs/
│   ├── PROJECT_STATUS.md             # 唯一的现状汇总
│   ├── CURRENT_TASKS.md              # 当前任务和验收标准
│   ├── README.md                     # 文档导航
│   └── plans/                        # 整理/实施计划
├── results/                          # 已入库的实验摘要；模型 checkpoint 通常忽略
├── figures/                          # 论文与诊断图
├── archive/                          # 已作废协议和历史结果
└── data/                             # 本地数据入口；仓库不包含原始生产数据
```

本轮采用保守整理：历史脚本、结果与引用路径均保持不动。待模型路线定性后，再决定是否把活跃实现迁移为正式 Python 包。

## 本地与 Linux 远端分工

本项目将“实验研发”和“正式算力执行”明确分开：

| 环节 | 本地工作区（本仓库） | Linux 远端 |
|---|---|---|
| 研究 | 梳理证据、提出假设、设计公平对照与判决标准 | 不临时改变研究问题或评测口径 |
| 实现 | 编写模型、训练脚本、测试、smoke 配置和结果 schema | 按指定 commit 与命令运行 |
| 验证 | 做静态检查、单元测试和小规模 smoke；审查数据泄漏与指标实现 | 使用真实数据/GPU 完成多 seed、长 epoch 正式实验 |
| 结果 | 读取日志和 JSON，复算指标、检查异常、形成结论并更新文档 | 回传 commit、命令、环境、日志、结果和退出状态 |

代码写完或本地 smoke 通过，只表示 **ready for remote**；只有 Linux 结果返回并经过本地审计，实验才能标记为 **audited** 或 **concluded**。远端运行失败时不直接热修代码，应回传日志，由本地修复并生成新的 commit 后重跑。

完整交接格式见 [远端实验协议](docs/REMOTE_EXPERIMENT_PROTOCOL.md)。

## 关键代码与实验入口

| 文件 | 用途 | 定位 |
|---|---|---|
| [exp_025_unified_benchmark.py](experiments/phase1_dynamics/exp_025_unified_benchmark.py) | 统一预测基线与数据入口 | 广泛被后续实验复用 |
| [eval_protocol.py](experiments/phase2_mpc/eval_protocol.py) | MPC 公平协议与回归测试对象 | 历史控制探索的稳定基础设施 |
| [causal_eval.py](experiments/phase3_feedforward/causal_eval.py) | 动作构造、DiD 与 CFE 指标 | 当前有效评测模块 |
| [causal_arch.py](experiments/phase3_feedforward/causal_arch.py) | A1/A3/B1、`g(x,0)=0` 与 Koopman free-head | 当前候选架构模块 |
| [exp_106_causal_arch.py](experiments/phase3_feedforward/exp_106_causal_arch.py) | 因果架构训练与比较 | 当前实验主入口之一 |
| [exp_112_koopman_full.py](experiments/phase3_feedforward/exp_112_koopman_full.py) | Koopman free-head 完整对照 | 仅关闭该具体变体 |

## 主要证据

- 真实事件研究支持减温动作存在明显迟延，不能用单步符号正则强行制造即时响应。
- Phase 2 审计表明：开环预测好不等于闭环可控，旧版 MPC 优势结论不能外推到真实锅炉。
- CFE 审计修复了二阶差分动作、训练区间事件泄漏和单末点评分等问题。
- exp_112 的三 seed 汇总：A1phys 的平均最佳 MAE 为 0.8467、平均最佳 CFI 为 0.869；Koopman free-head 分别为 0.8902 和 0.821；移除 free-head 后 MAE 退化至 1.5467。结果支持保留 free dynamics，但不支持当前 Koopman 预测头。

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
3. 所有候选路线使用相同切分、事件集、预算和多 seed 评测。
4. 新实验必须有 `if __name__ == '__main__':`，结果目录不得覆盖其他协议。
5. 结论必须记录反例、失效边界和被推翻的旧口径。
6. 先更新 [项目状态](docs/PROJECT_STATUS.md) 与 [当前任务](docs/CURRENT_TASKS.md)，再更新根 README。
7. 正式实验必须记录 Git commit、完整命令、数据指纹、seed、环境和输出目录；Linux 结果审计前不得写成项目结论。

## 相关文档

- [Fan 三篇整合解读](docs/Fan_三篇整合_热工控制入门.md)
- [伊敏变量与 Fan 模型对照](docs/伊敏40列_vs_Fan模型变量对照.md)
- [Neural ODE、Deep Koopman 与 Koopa](docs/Neural_ODE_Koopman_三篇关键论文.md)
- [因果评测框架](docs/causal_eval_framework.md)
- [完整代码与结果审查 v2](docs/session_2026-08-06_review_v2.md)
- [Phase 2 最终审查](docs/phase2_final_audit.md)

论文图表默认使用英文；研究记录与工程文档可使用中文。
