# 项目状态

> 更新：2026-08-07。本文是项目现状的权威入口；历史文档保留当时结论，不自动代表当前判断。

## 一句话状态

项目已完成预测基线、MPC 方法探索和第一轮观测事件评测，但**尚未完成模型定性，也没有独立 lockbox 结论**。历史 test 时段已被多轮开发与逐 epoch 选模访问；A1phys 仅保留为监督层 baseline，Fan20 主汽温骨架及三类动态表达尚未在统一协议下验证。

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

### 2. A1phys

当前形式为：

```text
T_hat(x, a) = f_free(x) + g_phys(x, a)
g_phys(x, 0) = 0
```

`g_phys` 使用工况相关增益和两级一阶惯性。它是监督层 ΔSP 到闭环主汽温响应的灰箱先验，不含 controller/actuator、质量守恒、能量守恒、焓值传递和完整锅炉状态方程，因此不能称为 Fan plant 物理模型。

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

Phase 4 冻结两个不可混榜的任务：

1. **plant-level**：喷水流量或经核验的有效阀位代理 → Fan20 plant → `Tst`；
2. **supervisory-level**：监督层 SP/ΔSP → controller/actuator → valve/spray → plant → `Tst`。

exp_025 使用实际绝对阀位，exp_106/A1phys 使用 `二级减温调节阀设定` 的一阶差分，Fan20 使用两级喷水质量流量。它们目前不是同一 estimand，禁止放在同一模型冠军表中。

## 三类可微动力学路线的真实状态

| 路线 | 仓库已有工作 | 尚缺什么 |
|---|---|---|
| Neural ODE | exp_020 使用纯神经动力学与固定 Euler 小步 | Fan 方程、守恒结构、部分可观测状态、严谨 ODE solver、公平多 seed |
| Controlled Koopman | exp_020 的潜在受控 decoder；exp_112 的 Koopman free-head | 在 Fan 状态和控制输入上建模、非平稳工况、物理输出约束、独立干预评测 |
| 时变灰箱混合 | A1phys 有低阶惯性；文档调研了 Koopa/时变参数 | Fan 2020/2021 骨架、负荷调度、能量不匹配、低维残差与参数漂移验证 |

因此，exp_020 和 exp_112 都不能回答“Fan 物理微分模型是否优于当前模型”。

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
- 活跃逻辑主要在 `experiments/phase3_feedforward/`，`src/` 仍是早期原型。
- 仓库没有锁定依赖或可复现环境文件。
- `data/伊敏6号机` 是 Linux 符号链接，Windows 检出不可直接使用。
- 现有测试主要覆盖 Phase 2 评测协议，不覆盖 CFE、A1phys 或 Fan 方程。
- 当前 `pytest` 在收集阶段失败：`tests/test_eval_protocol.py` 注入的基线模块桩缺少 `TimeXerWM`，而 `eval_protocol.py` 已新增该导入；这是既有测试桩漂移，不是本次文档整理引入。
- `eval_protocol.py` 的 PID 物理方向、零误差工作点和导数项实现存在 P0 缺陷；旧控制结果不得恢复证据等级。
- `exp_106/112` 在 test 上逐 epoch 评估并选 checkpoint；`exp_109/110` 又合并 val+test 构造/筛选事件。
- 148 个 Python 文件中有 88 个没有 `if __name__ == '__main__'` guard；多个实验依赖导入副作用、全局变量与 `sys.path/sys.argv` 修改。
- 当前代码对 NaN 静默置零、丢弃时间信息，尚无 episode/gap-aware 窗口与 split manifest。
- 在模型路线定性前，不进行大规模源码迁移或历史目录重排。

## 下一判决点

只有依次通过以下 Gate，才讨论“主模型”定性：

1. Gate 0：tag/action、episode/split、validation-only 选模、fail-closed metrics 和测试地基；
2. Gate 1：IAPWS 与 Fan20-SST 方程闭合、合成恢复、数值稳定和可辨识性；
3. Gate 2：Fan17/21 仅作为嵌套组件的最小消融；
4. Gate 3：固定物理内容后公平比较 ODE、controlled Koopman、time-varying gray-box；
5. Gate 4：rolling folds、负荷/action worst-group 与 cluster-aware 统计；
6. Gate 5：5 seeds、一个 canonical checkpoint；有新数据时一次批量 locked-final，无新数据时明确标注 internal-final。

活队列见根目录 [`TODO.md`](../TODO.md)，完整判决规则见 [`PHASE4_EXPERIMENT_PLAN.md`](PHASE4_EXPERIMENT_PLAN.md)。
