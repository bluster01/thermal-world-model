# 项目状态

> 更新：2026-08-10。本文是项目现状的权威入口；历史文档保留当时结论，不自动代表当前判断。

## 一句话状态

项目已完成 MS0、MS1、MS2-V/C/J 和 MS2-D1；MS2-J 联合模块双层 PASS、response-internal staged 非劣双层 FAIL。D1 test 的点改善方向稳定，但确认性 CI 下界 17.2–18.8% 未达预注册 20%，故以 `TEST_NOT_CONFIRMED_AT_20PCT_MARGIN` 关闭，且 delay 参数仍不可唯一辨识。项目**尚未完成真实模型定性，也未进入论文收口**。当前 active Gate 只授权 MS2-D2 的 21-run synthetic validation；后续仍需 D3、MS5 完整耦合、MS3 真实适配和 MS4 闭环响应。旧 E1–E5 已废弃；Phase 4 继续暂停。恢复入口见 [`PHASE35_CONTEXT_SNAPSHOT.md`](PHASE35_CONTEXT_SNAPSHOT.md)。

## Phase 3.5 当前状态

| 项目 | 状态 | 当前边界 |
|---|---|---|
| 原始数据口径 | 已完成只读探索，待 Linux manifest | A/B 为异步稀疏 historian；不能把原始行称为 10 s 样本 |
| 动作定义 | 已冻结 | 主动作是实际二级减温阀反馈开度；SP 是监督层信号；喷水流量不作真值 |
| 数据框架 | 已实现 | causal LOCF 10 s grid、staleness、SHA256、60/20/20 split 内窗口 |
| A1phys-V | 已实现 | 历史阀位可作处理前状态；未来阀位只进入受约束干预分支 |
| exp_201 pilot | 已回传并审计降级 | A 侧固定 `R=50`：ff10 三 seed 为 100%×3，no-freeze 三 seed 均值约 98.3%；原始绝对阀位多为 60–75%。均为 test-selected，且手工先验未标定，不能作正式证据 |
| 核心实验 | 42/42 development runs 已完成并审计 | E1 PASS；E2/E3/E5 INCONCLUSIVE；E4 BLOCKED；没有 test 候选 |
| 训练选择 | 已修正 | validation-only canonical checkpoint；test 显式解锁并写 ledger |
| 统计 | 已运行并 fail-closed 审计 | A/B 分报；UTC 日块 bootstrap；seed 仅表示优化波动；matching balance/common support 未过 |
| 正式结果 | 仅 development validation | 模型 test 尚未打开；A/B 旧 event test 标签已暴露，未来事件证据需新时间块 |

Phase 3.5 的目标不是证明质量/能量守恒，而是建立分层物理一致性：阀门动作可辨认、经验温度响应可复核、模型反事实响应能复现该曲线、SP 未执行时模型不制造阀门效应。完整协议见 [`PHASE3_5_EXPERIMENT_DESIGN.md`](PHASE3_5_EXPERIMENT_DESIGN.md)。

## 最终世界模型距离

最终系统需分别通过状态/动作语义、独立预测、状态闭合仿真、可识别反事实和分级闭环五项合同。当前只达到 `C0 PARTIAL + C1 DEVELOPMENT ONLY`：

| 最终能力 | 当前证据 | 判决 |
|---|---|---|
| 未见时间预测 | 历史预测基线与 Phase 3.5 development validation | 尚缺新时间块独立 test、校准与工况分层 |
| 递推仿真 | 当前主要输出固定 horizon 温度与低阶动作响应 | 未建立完整下一状态、自由 rollout 稳定性或守恒闭合 |
| 反事实推演 | 有代码方向/零动作约束；E3 reference 未通过 | BLOCKED；action sensitivity 不能写成 `do(action)` |
| 闭环嵌入 | 旧 MPC 探索存在同构 plant 与协议缺陷 | 未建立独立 plant/HIL、策略支持、实时性和安全证据 |

权威缺口矩阵和 W0–W6 放行路径见 [`WORLD_MODEL_EVIDENCE_LADDER.md`](WORLD_MODEL_EVIDENCE_LADDER.md)。

## 证据分级

| 等级 | 含义 | 使用规则 |
|---|---|---|
| A | 真实数据、协议已审计、结果可追溯 | 可作为论文主张，但仍需写明适用边界 |
| B | 多 seed 或多协议支持，但依赖代理真值/模型仿真 | 可作为候选结论，不外推到现场闭环 |
| C | 单 seed、旧协议、消融或机制假设 | 只用于生成下一步实验 |
| X | 已发现协议错误或被后续结果推翻 | 仅保留历史，不再引用为有效结论 |

## 当前可信结论

| 结论 | 等级 | 主要依据 |
|---|---|---|
| 历史 supervisory tag 的观测响应存在迟延，短时不应强制最终方向 | B/C | `phase1_report.md`；tag/estimand 与匹配协议仍待重审 |
| 绝对阀位在旧预测协议中优于差分阀位 | C | exp_012 及后续旧基线；需在新 split、同 estimand 下复核 |
| RevIN 与 per-variable TCN 在旧预测消融中有正面信号 | C | exp_025；主要为单 seed/test 已参与开发 |
| 开环预测精度不能证明干预因果或闭环控制效用 | A/B | CFE 审计、Phase 2 最终审查 |
| `g(x,0)=0` 是显式动作分支的代码恒等式 | C | `causal_arch.py`；它不提供因果识别，也不保证全模型零动作输出 |
| A1phys 的两级惯性结构在当前旧协议中值得保留 | C | exp_106/110/112；存在 test 选模、观测 reference 与 action estimand 问题 |
| 当前 Koopman free-head 的 MAE pilot 未优于 MLP free-head | C | exp_112，3 seeds、最多 50 epochs 且均早停；test-selected，仅限该实现 |

目前没有满足“未参与开发的 test + 协议审计 + 统计不确定性”的 A 级模型比较。已有 A/A-B 事实不能自动升级任何模型。

## 尚未定性的候选模型

### 1. M7 / Direct WM

它是强预测基线和大量后续实验的基础设施，但动作响应存在条件期望与干预效应混淆。保留为纯数据驱动 baseline，不再称“最终模型”。

### 2. 历史 A1phys 与 Phase 3.5 A1phys-V

当前形式为：

```text
T_hat(x, a) = f_free(x) + g_phys(x, a)
g_phys(x, 0) = 0
```

旧 `g_phys` 使用工况相关增益和两级一阶惯性，是监督层 ΔSP 到闭环主汽温响应的灰箱先验。Phase 3.5 新实现改为实际绝对阀位轨迹，并允许 A/B 侧独立的单调有效开度映射；constant-valve 路径严格为零、开阀长期增益非正。它仍不含质量守恒、能量守恒、焓值传递和完整锅炉状态方程，因此只能称 physics-guided gray-box，不能称 Fan plant 物理模型。

exp_112 三 seed 的**探索性、test-selected** MAE 汇总（每 seed 逐 epoch 取 test 最小值后再求均值）：

| 变体 | 最佳 MAE 均值 | 与 MLP 差值 | 可用判断 |
|---|---:|---:|---|
| A1phys + MLP free-head | 0.8467 | — | 旧协议参考 |
| A1phys + Koopman free-head | 0.8902 | +0.0435 °C | 该实现的负面 pilot；不能关闭路线 |
| A1phys without free-head | 1.5467 | +0.7000 °C | 旧协议的 3 个探索性 seed 均更高；无独立 CI |

exp_112 指向的 `results/cfe_groundtruth_p2/did_response.json` 在仓库不存在，且训练期 test-only 事件数与 P2 val+test 事件数不匹配，因此所谓 0.869/0.821 “最佳 CFI”实际是每 seed 对 16 个 test 事件逐 epoch 取最大 `sign(ΔSP)+gain` fallback 后的均值，而不是 P2 CFE/DiD。脚本还分别保存 test-selected `best_mae` 与 `best_causal`。这些 CFI 数字撤销证据资格，MAE 也不能作独立测试估计。

### 3. Fan 灰箱模型

| 骨架 | 关键状态/机制 | 数据覆盖 | 实现状态 |
|---|---|---|---|
| Fan 2017 | 4 状态非线性 ODE、显式金属蓄热 | Fan20 已含制粉动态；`Tj` 未观测，加入后须避免热量双计 | 金属组件候选，未实现 |
| Fan 2020 | 7 状态、两级喷水、分段焓值与 SST | 与任务最匹配；喷水质量流量、给水压力和部分测点仍缺/不确定 | 中央骨架，未实现 |
| Fan 2021 | 4 状态宽负荷 CCS、整炉能量不匹配、节流损失、时变参数 | 不直接含 SST/喷水；mismatch 需映射到 Fan20 分段热量，throttle 只作用完整 CCS | 组件候选，未实现 |

详细变量映射见 [`伊敏40列_vs_Fan模型变量对照.md`](伊敏40列_vs_Fan模型变量对照.md)。映射“存在”不等于数据可用；阀位不能直接当 `kg/s` 喷水流量，主汽压不能直接当给水压力，`T3` 需要测点图核实，`ut` 不得用可能泄漏结果的负荷变化率替代。

## 动作层级与研究对象

项目区分两个不可混榜的任务：

1. **plant-level**：喷水流量或经核验的有效阀位代理 → Fan20 plant → `Tst`；
2. **supervisory-level**：监督层 SP/ΔSP → controller/actuator → valve/spray → plant → `Tst`。

exp_025 使用实际绝对阀位，exp_106/旧 A1phys 使用 `二级减温调节阀设定` 的一阶差分，Fan20 使用两级喷水质量流量。它们不是同一 estimand，禁止放在同一模型冠军表中。Phase 3.5 只比较阀门级配置；SP 仅作执行链负对照。

## 三类可微动力学路线的真实状态

| 路线 | 仓库已有工作 | 尚缺什么 |
|---|---|---|
| PI-ODE | exp_020 使用纯神经动力学与固定 Euler 小步；MS1 在同型二阶 synthetic truth 达噪声下限；MS2-V/C validation/test 误差较低 | MS2 未把 PI-ODE 预注册为主要对照或冠军；PI-ODE 自身的联合非线性/显式调度、Fan 方程、部分可观测状态与真实响应仍未验证 |
| Controlled Koopman | exp_020 的潜在受控 decoder；exp_112 的 Koopman free-head；MS1 稳定受控模态算子通过结构门禁 | MS2 的 global-context 与四模态阀门版本只作次要对照，不能据此赋予真实 Koopman 谱解释 |
| 灰箱 / DeepONet 算子 | MS1 证明 2P 灰箱同型可解；MS2-V/C validation+test 的两个预注册响应对比均通过；MS2-J 联合模块 validation+test 双层通过 | 单调响应模块存在 `K/phi/动力学` 补偿，尚未辨识出真实阀门曲线；MS2-D、完整 free+response、真实数据迁移与 Fan 物理闭合仍缺 |

因此，exp_020 和 exp_112 都不能回答“Fan 物理微分模型是否优于当前模型”。Phase 3.5-MS 新框架也只回答低维动作响应的表示/优化可行性，不等同于 Fan 模型比较或现场因果验证。

Phase 3.5-MS 的统一 estimand、四类路线公式、损失/指标定义、可辨识性边界和来源核验见 [`PHASE35_MS_METHODS_AND_REFERENCES.md`](PHASE35_MS_METHODS_AND_REFERENCES.md)。

MS1 的准确结论是“synthetic 同型可解性 PASS”，不是模型冠军。MS2-V/C validation+test 已独立复核：两个主对比的逐 seed paired-episode CI 下界均远高于 20%，但 MS2-V 的 learned `phi` 没有恢复真值曲线，DeepONet 也可隐式表达非线性；论文不能写成“阀门映射必要且已辨识”。MS2-J 一次性 test 已执行（`5260d3f`）：联合模块相对两个单模块消融在 validation+test 双层通过（test CI 下界 0.73–0.89 >> 20%），staged 非劣双层失败（test ratio 1.14–1.20，CI 上界 1.09–1.32 > 1.10）——主训练方案定 joint，staged 仅作阴性消融；不重训、不扩矩阵。见 [`PHASE35_MS2J_TEST_REVIEW_2026-08-10.md`](PHASE35_MS2J_TEST_REVIEW_2026-08-10.md)。

## 已降级或作废的结论

| 旧表述 | 当前状态 | 原因 |
|---|---|---|
| M7 是最终模型 | 降级 | 只在预测基线与旧协议中领先，未与 Fan 灰箱路线比较 |
| DWM-MPC 比 PID 更好 | X | 同构 plant、动作弱因果、部分协议不公平，不能外推 |
| 单点 600 s CFI 足以选模型 | X | 会奖励“末点对、过程错”；历史聚合 CFI 仍有量纲/权重问题，Phase 4 只作分解诊断、不选模 |
| Koopman 路线整体关闭 | X | exp_112 只评估 Koopman free-head |
| A1phys 已是物理模型 | X | 当前仅含低阶惯性先验，没有守恒方程 |
| P2 DiD/CFE 是因果 ground truth | X | 闭环观测匹配缺关键混杂、balance/pre-trend/placebo；应称观测事件响应参考 |
| exp_112 的 0.869/0.821 是 P2 CFI | X | GT 文件/事件长度不匹配，实际走同名 fallback |
| Fan 三篇是三个平行 SST 全模型 | X | Fan20 直接覆盖 SST；Fan17/21 更适合作为嵌套机制 |

## 当前工程状态

- 工作边界：本地负责实验设计、代码实现、测试与结果审计；Linux 远端负责真实数据/GPU 上的正式运行。
- “已实现”不等于“已验证”：只有远端结果返回并完成本地审计，实验才能进入结论层。
- 当前正式逻辑位于 `src/phase35/` 与 `experiments/phase3_5/`；`phase3_feedforward/` 只作历史追溯。
- 仓库没有锁定依赖或可复现环境文件。
- `data/伊敏6号机` 是 Linux 符号链接，Windows 检出不可直接使用。
- Phase 3.5 已新增数据、模型、事件、训练、统计、汇总和 CLI smoke 专项测试；最新通过数以仓库测试命令输出为准。全仓历史测试仍有 `TimeXerWM` 测试桩及硬编码 CSV 导入收集错误，不能由专项测试代替或隐去。
- `eval_protocol.py` 的 PID 物理方向、零误差工作点和导数项实现存在 P0 缺陷；旧控制结果不得恢复证据等级。
- `exp_106/112` 在 test 上逐 epoch 评估并选 checkpoint；`exp_109/110` 又合并 val+test 构造/筛选事件。
- 148 个 Python 文件中有 88 个没有 `if __name__ == '__main__'` guard；多个实验依赖导入副作用、全局变量与 `sys.path/sys.argv` 修改。
- 当前代码对 NaN 静默置零、丢弃时间信息，尚无 episode/gap-aware 窗口与 split manifest。
- 在模型路线定性前，不进行大规模源码迁移或历史目录重排。

## 下一判决点

下一判决点是 MS2-D2 三阶惯性 validation：在无 pure delay、R50+context scheduling、truth tau `[40,70,210] s` 下，要求三极点 oracle 每 seed clean NMAE `<0.05`，三极点主模型每 seed `<0.10`，且相对同预算二极点每 seed改善 `≥10%`。tau 集合恢复和二极点+learned-delay 的虚假迟延只作诊断，不并入主门禁。注册表现为 `ready_for_linux`；test 与 D3 均未授权。

完整顺序为 MS2-D1/D2/D3 → MS5 → MS3 → MS4 → 模型选择/论文。活队列见根目录 [`TODO.md`](../TODO.md)；Phase 4 保持暂停。
