# 当前任务

> **历史任务设计 / 已被新计划取代。** 本文保留 2026-08-07 早期梳理，不能作为活队列。当前任务只在根目录 [`TODO.md`](../TODO.md) 维护；判决协议见 [`PHASE4_EXPERIMENT_PLAN.md`](PHASE4_EXPERIMENT_PLAN.md)，审查依据见 [`SUPERVISOR_REVIEW_2026-08-07.md`](SUPERVISOR_REVIEW_2026-08-07.md)。

> 原目标：建立能公平判定 Fan 灰箱与三类可微动力学路线的实验闭环。新计划进一步把 Fan20 设为 central SST skeleton、Fan17/21 设为嵌套组件，并拆分 plant/supervisory 两个 estimand。

## 执行分工与状态机

- **本地**：研究审计、实验设计、代码与测试、smoke、远端运行包、结果复核和结论维护。
- **Linux 远端**：固定 commit 上的真实数据/GPU 正式训练与评测，回传标准化产物。

每项实验依次使用以下状态，禁止跨级：

```text
designed → implemented → smoke_passed → ready_for_remote
         → remote_running → results_returned → audited → concluded
```

`implemented` 或 `smoke_passed` 不能写成“实验完成”；`results_returned` 也必须经过本地复算和协议审计。交接清单见 [REMOTE_EXPERIMENT_PROTOCOL.md](REMOTE_EXPERIMENT_PROTOCOL.md)。

## P0：冻结术语与有效基线

**输出**：所有新文档和实验统一使用以下名称。

- `M7 / DirectWM`：纯数据驱动预测 baseline。
- `A1phys`：带两级惯性干预先验的灰箱 baseline，不称“物理模型”。
- `Koopman-free`：exp_112 的具体 free-head 变体，不代表 controlled Koopman 全路线。
- `Fan20`：主汽温 central plant skeleton；`Fan17 / Fan21`：待验证的嵌套机制，不是三个平行 SST 全模型。

**验收**：根 README、项目状态和后续实验标题不再出现无范围限定的“最终模型”或“路线关闭”。

## P1：Fan 数据可观测性与物性层

这是所有物理微分路线的共同前置任务。

### P1.1 数据字典与单位审计

**输入**：40 列 CSV、`伊敏40列_vs_Fan模型变量对照.md`。

**输出**：机器可读 schema，至少包含列名、单位、采样率、缺失率、有效范围、传感器/指令/反馈属性和 Fan 变量映射。

**验收**：Ne、pst、Dfw、uB、pm、Tm、Dst、Tst、两级喷水代理都能唯一映射；任何单位假设显式记录。

### P1.2 焓值与物性计算

**输出**：基于 IAPWS-IF97 的 `hm`、`hst`、`hfw` 计算与测试。

**验收**：

- 随机抽样工况落在正确水/汽区域；
- 温压单位换算有单元测试；
- 物性异常不会被静默截断；
- 计算结果与 Fan 文献量级和现场常识一致。

### P1.3 缺失 `ut` 的处理

比较机组负荷变化率、AGC 指令、燃料主控输出作为代理；同时保留“不使用 `ut`”的可识别子模型。

**验收**：代理变量需通过滞后相关、事件响应和稳定性检查，不能仅凭相关系数选取。

## P2：建立统一判决协议

**输出**：所有模型共用的数据集、训练预算与评测 API。

### 固定条件

- 相同 70/15/15 时间切分，事件只来自测试区间。
- 相同历史窗口和预测时域；短窗 180 s 与长窗 600 s 分开报告。
- 相同参数预算或同时报告参数量、训练时间和推理时间。
- 相同 seed 列表；筛选阶段至少 3 seeds，定稿阶段至少 5 seeds。
- checkpoint 选择只使用验证集，测试集只做一次最终评估。

### 指标

1. 预测：逐 horizon MAE/RMSE、工况分层误差、概率校准。
2. 干预：DiD/CFE、方向、归一化增益、形状、到峰时间。
3. 物理：守恒残差、状态范围、稳定性、零动作不变量。
4. 泛化：负荷分层、升降负荷、稀有大动作、时间外推。
5. 工程：训练时长、推理延迟、显存、失败率。

**验收**：不得用同一测试事件选择 checkpoint；不得用单一 600 s 末点替代完整响应曲线。

## P3：Fan 三种物理骨架的最小实现

先验证方程和数据是否闭合，再加入复杂神经网络。

### P3.1 Fan17-min

实现 4 状态非线性 ODE 的最小可微版本，未知常数先用可训练标量或低阶负荷函数，不加入高容量残差网络。

**通过条件**：积分稳定、状态有物理意义、在固定窗口上优于 persistence，并能正确响应燃料/给水代理。

### P3.2 Fan20-SST

实现两级喷水与逐段焓值链，主输出为 Tst，并保留 pst/hst/T3 等辅助状态或输出。

**通过条件**：两级喷水路径方向正确、快慢通路可区分、物性关系残差可量化；阀位作为流量代理时必须单独报告局限。

### P3.3 Fan21-wide-load

实现能量不匹配项和负荷相关/时变参数，先不加入节流损失或将其作为可选消融，避免 `ut` 缺失导致不可识别。

**通过条件**：宽负荷分层误差优于固定参数版本，参数变化平滑且不依赖测试集调参。

## P4：三类可微动力学表达的公平比较

在 P3 中最可识别的共同状态空间上比较，而不是同时改变数据、状态和 decoder。

### Route A：Fan-structured Neural ODE

- 已知守恒项显式编码；未知传热/修正系数由小网络给出。
- 使用可靠 ODE solver，并与固定步 Euler 做数值误差对照。

### Route B：Controlled Koopman

- 编码 Fan 状态，使用 `z_{t+1}=K(x_or_load)z_t+B(x_or_load)u_t`。
- 输出端恢复物理状态并计算物性/守恒残差。
- exp_112 的 free-head 结果仅作为负面对照。

### Route C：时变灰箱混合

- Fan 2020/2021 方程作为主干；低维残差只修正传热系数、能量不匹配或时变参数。
- 可评估 Koopa/时变算子，但不得让高容量残差绕过物理主干。

**判决规则**：任何路线只有同时改善预测或 OOD 表现，并保持干预/物理指标不退化，才进入下一轮。单一 MAE 优势不足以胜出。

## P5：与当前 A1phys 的关键对照

统一比较：

| 模型 | 作用 |
|---|---|
| Persistence / ARX | 最低复杂度参考 |
| M7 / DirectWM | 纯数据预测上限参考 |
| A1phys | 低阶干预先验参考 |
| Fan17-min | 基础守恒 ODE |
| Fan20-SST | 与主汽温最直接的物理模型 |
| Fan21-wide-load | 宽负荷与时变能力 |
| Route A/B/C | 同一状态空间上的动力学表达比较 |

最终模型可以是单一路线，也可以是 Fan20-SST 主干 + Fan21 时变项的组合；在实验完成前不预设答案。

## P6：工程补齐

- 修复 `tests/test_eval_protocol.py` 的模块桩，使其覆盖 `eval_protocol.py` 当前导入的 `TimeXerWM`，恢复测试收集。
- 新增可复现环境文件与最小 CPU smoke test。
- 为数据 schema、物性层、ODE 积分、零动作不变量和 CFE 增加测试。
- 待模型定性后，再把活跃模块迁入 `src/thermal_world_model/`。
- checkpoint、生产数据和大体积中间文件继续排除在 Git 外。
- 为远端运行提供统一 manifest、日志与结果 schema，确保每次结果能追溯到唯一 commit。

## 暂不做

- 不继续扩展旧 MPC-vs-PID 主表。
- 不根据 exp_112 宣布所有 Koopman 方法无效。
- 不在 Fan 方程未闭合前增加复杂控制器。
- 不移动或重命名现有 `exp_0XX` 脚本。
