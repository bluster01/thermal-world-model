# Phase 3.5 — 从闭环SP到plant valve：estimand修正

> 日期：2026-08-08
> 状态：**实验完成（2026-08-08）** — 证据链见 §5；原设计稿 §0-4 保留
> 前置：Phase 3 收口论文设计稿（PHASE3_PAPER_DESIGN.md）
> 起因：1s 数据验证揭示 A1phys 学到的方向是 supervisor tracking，不是 plant 物理

---

## 0. 推理链（保留）

### 0.1 触发

用户质疑："设定值变了，阀门的控制动作没有变，A1phys 的方向正确是怎么来的？"

### 0.2 数据验证

**10s 数据**（A侧主汽温全数据03_cleaned_10s.csv，707,725行）：
- SP 变化（|ΔSP|>1°C）仅占 0.76% 的 timestep
- 阀位变化占 89.23% 的 timestep
- SP 变化时阀位反向（SP↑→关阀，PID 正确行为）：64.1%
- SP 变化后 600s 内阀位几乎不动（<0.5%）：仅 1.5%

**结论：SP 极少变化，但变化时阀位正确响应。SP 不变时阀位变化由温度偏差驱动（混杂）。**

**1s 数据**（A侧主汽温全数据4.csv，70,020,907行，59ms 中位采样间隔）：
- 前向填充 → 1s 重采样 → 500,000 秒
- SP 变化方向正确率从 64%（10s）提升到 84%（1s）
- 但外源 valve 信号仅占 0.2%，混杂占 99.8%

**结论：1s 数据让 SP→valve 方向更干净，但信号太稀疏，不足以做训练信号。**

### 0.3 传递函数辨识

A1phys 的 `g_phys(x, ΔSP)` 用 n_lag=2 的级联 LTI 拟合闭环传递函数：

```
G_cl(s) = dT / dSP
        = controller(s) · plant(s) / (1 + controller(s) · plant(s))
```

不是 plant 传递函数 G_p(s) = dT / dvalve。

**"方向正确"的真实原因**：SP↑ → PID 关阀 → 减少喷水 → 温度↑。dSP 和 dT 同向是因为**控制器 tracking property**，不是 g_phys 学到了物理因果。

### 0.4 和 Phase 1 阀位实验的本质区别

Phase 1（exp_011/012/025）用阀位训练单塔模型，结论是"混杂压倒因果——开阀↔升温是假象"。

但 Phase 1 的架构是 `[x_t ‖ valve_t] → encoder → decoder → T̂`，所有信号混在一起，没有地方分流混杂和因果。

Phase 3.5 的方案是 **A1phys 的架构 × Phase 1 的 action（阀位）**：

```
x_t → f_free(x_t) → T̂_free                      ← 吸收混杂
         +
x_t, valve_t → g_plant(x_t, valve_t) → T̂_plant  ← g(x,0)≡0，只学增量
         ↓
      T̂ = T̂_free + T̂_plant
```

**本质区别**：不在阀位数据（一样），在架构——双分支有地方把"温度自己波动导致的阀位变化"和"阀位变化对温度的影响"分开。

### 0.5 核心假设（待验证）

> **H₁**：在 f_free 吸收混杂后，g_plant 从残余信号中能学到正确的物理方向（开阀→降温，∂T/∂valve < 0）。
>
> **H₀**（零假设）：g_plant 学到的是随机方向或无响应——因为残差里混杂成分已被 f_free 吸收，剩余物理信号太弱。

**验证方式**：对 g_plant 做独立干预测试——固定 x_t，对 valve_t 做 ±Δ 扰动，看 ∂T̂_plant/∂valve 的符号是否为负。

**SP 事件审计**：在 SP 变化事件的 t₀ 时刻，看 g_plant 对实际阀位变化的响应方向是否与物理一致。

---

## 1. 实验设计

### 1.1 基线模型

| ID | 名称 | 架构 | Action 变量 | 说明 |
|----|------|------|------------|------|
| P35-A1 | A1phys_valve | f_free + g_plant | 二级减温调节门阀位（绝对值） | **本实验核心** |
| P35-A2 | A1phys_valve_diff | f_free + g_plant | 二级减温调节门阀位（一阶差分） | 对照：差分信号的物理可学性 |
| P35-A3 | A1phys_sp | f_free + g_phys | 二级减温调节阀设定（ΔSP） | Phase 3 已有，作为闭环基线上限 |
| P35-B1 | M7_valve | 单塔 DirectWM | 阀位绝对值 | Phase 1 架构复现，混杂对照组 |
| P35-B2 | B1glb_valve | TimeXer 单塔 | 阀位绝对值 | 因果对照组 |

### 1.2 实验步骤

**S1：重跑 A1phys_valve（P35-A1）--freeze-free=10, seeds=0/1/2**
- 复用 exp_106 框架，action 从 ΔSP 换成绝对阀位
- 输出：MAE + CFI + g_plant 雅可比方向 + SP 事件方向审计

**S2：对照 A1phys_valve_diff（P35-A2）--seeds=0**
- 给定 Phase 1 证明差分信号太弱，先跑 1 seed 验证
- 如果 MAE 和方向不可用，放弃该分支

**S3：对照组 M7_valve（P35-B1）--seeds=0**
- 验证单塔架构是否仍学反方向
- 如果单塔方向正确 → 10s→1s 数据变更后混杂减弱

**S4：SP 事件方向审计**
- 对 P35-A1 的 g_plant，在 SP 事件 t₀ 时刻取 valve 变化
- 检查 g_plant(valve_t+1..valve_t+5) 的方向是否物理正确
- 对比 A1phys_sp 的 g_phys 在同一事件上的方向

**S5：雅可比方向全局审计**
- 对所有 test 时间步，计算 ∂g_plant/∂valve 的符号
- 统计负号（物理正确）vs 正号（混杂噪声）vs 零（无响应）

### 1.3 成功标准

| 指标 | 准入门槛 | 目标 |
|------|---------|------|
| g_plant 雅可比符号（全局） | ∂g/∂v < 0 占比 > 70% | > 85% |
| SP 事件方向 | 阀位↑ → T̂_plant ↓ > 70% | > 80% |
| MAE | < 1.0°C | < 0.9°C |
| SP→T 方向（作为对照） | > 90% | 已确认 |

### 1.4 成本

| 实验 | GPU 训练次数 | 预计时间 |
|------|------------|---------|
| P35-A1 × 3 seeds | 3 次 | 2-3h |
| P35-A2 × 1 seed | 1 次 | 45min |
| P35-B1 × 1 seed | 1 次 | 45min |
| 方向审计（CPU） | — | 10min |
| **总计** | **5 次 GPU** | **~5h** |

---

## 2. 与 Phase 3 论文的关系

### 2.1 如果 H₁ 成立（g_plant 学到正确方向）

**叙事升级**：
- Phase 3 论文从 "supervisory closed-loop WM" 升级为 "结构分解使 plant-level 物理方向可学"
- 核心贡献从 "g(x,0)≡0 保证 SP 方向" 变成 "g(x,0)≡0 使 f_free 吸收混杂，g_plant 从混杂数据中学到正确物理方向"
- 传递函数从 "辨识闭环 G_cl" 变成 "辨识 plant G_p"

**文章结构调整**：
- §1 Introduction：加一句 "即使在不加物理约束的情况下，结构分解本身已足够分离混杂效应"
- §3 Method：加一节 g_plant(valve) 的实现
- §4 Experiments：加 valve action 的结果表
- §6 Discussion：加 "为什么单塔失败而双分支成功"

### 2.2 如果 H₀ 成立（g_plant 学不到）

**叙事保留**：
- Phase 3 论文不动，仍以 supervisory closed-loop 为主线
- 加一节 Limitation 或 Discussion：为什么 valve action 在双分支下仍不可学 → 根因在混杂占比 99.8%、物理信号占比 0.2%，不是架构问题
- 为 Phase 4 的 Fan20 物理骨架提供最强动机——"观测数据即使双分支也不能分离混杂，必须引入物理方程作为额外约束"

**不丢人**：一条负结果、有清晰根因分析、有下一步指向的 Discussion 比一个勉强做出来的正向结果更有说服力。

---

## 3. 如果 H₁ 成立：论文叙事 v3

> **题目方向**：
> Decomposing endogenous actions: Structure-inspired separation of confounding from causal response in industrial world models

> **Key Idea**：
> 闭环数据中 actuator 信号被混杂主导（valve↑↔temp↑ 占 71.6%）。单塔黑箱模型学到统计关联而非物理因果——但不换数据、不加物理方程约束的情况下，仅通过**结构分解 f_free(x) + g(x,a)** 和 **g(x,0)≡0 不变量**，我们让模型自动将混杂效应分配给 f_free、物理响应分配给 g，在 valve→temperature 的 plant-level 测试上得到物理正确的方向，而单塔对照模型全部失败。

> **三句话贡献**：
> 1. 结构不变量 g(x,0)≡0 不仅是架构恒等式，更是一种**自监督的前向因果分离机制**：它通过强制零基线把模型推入一个参数空间，使 f_free 自然吸收混杂、g 被残余的物理信号驱动
> 2. 在伊敏主汽温阀位数据上，双分支模型给出 ∂T̂/∂valve < 0（物理正确），而单塔对照全部给出正号（混杂错误）
> 3. 传递函数从 SP→T 升级为 valve→T（plant-level），n_lag=2 现在辨识的是 喷水→温度 的物理滞后

---

## 4. n_lag 消融（附带）

Phase 3 的 n_lag 消融已证明：
- 一阶：60s 无死区（FAIL）
- 二阶：最优（RMSE=0.022）
- ≥4阶：变差（不可辨识参数）

Phase 3.5 对 valve action 只需验证：
- n_lag=2 是否仍最优？
- K 和 τ 在 valve action 下的物理意义是否一致？（K 应更大——直接喷水比 SP 间接作用更直接）

---

## 5. 实验结果（2026-08-08，commit a7fff72）

### 5.1 执行轨迹（按用户驱动的表示修正演进）

| 阶段 | action 表示 | 方向 jac:neg (3seed) | gain@180s | 关键发现 |
|------|------------|----------------------|-----------|---------|
| v3 基线 | Δvalve (cumsum) | **0%**（复核） | ~+200 (错) | 技能表"100%正确"是假记录；混杂方向全程 |
| abs | 绝对开度（去中位数） | 65-68% | ~-1.5 | 用户"绝对值"直觉部分验证；ff 扫描(10/20/30)无效 |
| **flow** | 等百分比流量 R=50 | **100% ×3** | ~-1.5 | **表示决定方向可学习性**；K 100% 负 |
| **flow+λ0.2** | + SP-IV 增益校准 | **100% ×3** | **-77 (均值)** | **under-gain 修复：-1.5 → 真值区间** |

### 5.2 关键结论

**H₁ 成立但机制修正**：双分支结构 + g(x,0)≡0 是必要条件（null 变体 63%、单塔全错），但**决定性因素是 action 的物理量表示**——同一架构同一数据，仅换表示：0% → 65% → 100%。

1. **Δvalve 不可用**（方向 0%）；**绝对开度部分可用**（65-68%，高开度层崩塌 38%）；**等百分比流量表示**（F/Fmax = R^(V/100-1), R=50）方向 **3/3 seed 100%**，全开度层 89-100%。机制：等百分比特性在常规工作区（5-25%）斜率小 → 混杂关联"温度高→开阀"在流量空间被压缩。
2. **ff 扫描无效**（10/20/30 全 65%）→ 训练策略不是瓶颈；best_cfi 选点逻辑有缺陷（固定在 ep5），必须用 best_gain 检查点（loss_gain 最小）。
3. **SP 阶跃 = 工具变量**（用户指出）：plant 增益真值 = (dT/dSP)/(dV/dSP)，用 CEM 匹配 DiD 响应（r18/r60）+ 30s 阀位响应：**180s 真值 -90~-130 m°C/%**。阀位自身事件的 DiD 不可用（方向率 39%，混杂主导）。
4. **增益校准**（--lambda-gain，扰动口径 L_gain）：λ=0.2 → gain -50~-96（真值 ~73%），方向保持 100%，代价 MAE +43%（λ=0.5 崩）。
5. **"形状符合等百分比理论"是输入变换伪影**（增益曲线 = 常数 K × dF/dV），非学习成果；**0.006°C/% "物理地板"是 ARX 模型数字**（同样被混杂收缩），非真值。

### 5.3 论文叙事（v5，2026-08-09 审计后降级口径）

> **注意**: 2026-08-09 Supervisor 审计 (docs/PHASE3_5_LINUX_REVIEW_2026-08-09.md) 后, 本节的 v4 表述被降级。
> 以下为审计后可写范围; "SP-IV truth"、"结构性保证"、"完全物理响应"、"三项全绿" 等表述已撤回。

> **当前可写 (pilot 范围)**:
> 1. 在 A 侧开发 pilot 中, action 表示从 Δvalve 换为等百分比流量 proxy (R=50, 未标定) 后,
>    对齐评估下的模型扰动方向保持为负 (valve↑→T↓), 而 Δvalve 表示下方向不可学;
>    该 pilot 用于生成假设, 不作为独立物理验证。
> 2. SP 阶跃事件曾提供 observational gain reference (-90~-130 m°C/%@180s), 但 1s 数据
>    first-stage 诊断 (2026-08-09) 证明该参考是选择性样本 (dv30·dsp<0 事后筛选;
>    全样本 SP→阀位 R²<0.07, 41.6% 事件 30s 内阀位不动) —— 已弃用。SP 阶跃在伊敏
>    数据上不是外生工具变量 (运行干预, 非准稳态试验)。
> 3. 显式增益正则 (λ=0.15-0.2) 能把模型内部 180s 扰动 gain 从 ~-1.5 拉向 reference 量级
>    (-50~-96 m°C/%), 方向保持为负 —— calibration target recovery, 不是独立 gain 验证。
>    独立验证需在 B 侧/未来时间块以未参与训练的 reference 进行 (计划中)。

### 5.4 成功标准对照

| 指标 | 门槛 | 结果 |
|------|------|------|
| 雅可比方向 >70% | >85% | **100%**（flow, 3seed） |
| SP 事件方向 | >70% | 校准后保持 100% |
| MAE | <1.0°C | 1.38（校准代价，基线 0.95） |
| gain 校准 | — | -77 m°C/%（真值 ~73%） |

### 5.5 诚实边界

- 方向与增益在 **best_gain 检查点**（训练中途 loss_gain 最小时）达成，非训练终点；训练后期校准与 NLL 竞争导致波动
- 增益校准代价 MAE +43%（λ=0.5 崩; 同口径复核实际 ~+7.6% 于各自 final checkpoint）
- R=50 是工程先验（equal-percentage valve proxy），未用伊敏实际阀特性曲线标定
- SP-event reference 基于 79 事件（180s）/15 事件（600s），中位数口径稳健，精确值有噪声

---

## 6. 2026-08-09 Supervisor 审计回应

> 审计全文: docs/PHASE3_5_LINUX_REVIEW_2026-08-09.md

### 已接受并处理

| 项 | 状态 |
|---|---|
| P0-1 split offset bug (eval_jacobian/eval_gain_180 状态与action基线错位) | 🔧 修复中 (r2), 加 split-offset 单测, 重跑评估; 现有数值不作为对齐后的正式结果 |
| P0-2 SP-IV 工具变量假设不闭合 | ✅ **数据证实弱工具** (r3, 1s): 365事件中严格稳态S层n=1; SP→阀位first-stage R²<0.07, 41.6%事件30s内阀位不动; 原"SP-IV真值"(-90~-130 m°C/%)是 dv30·dsp<0 事后筛选的选择性样本 — 已弃用, 降级为"选择性观测参考", 不再作为校准目标依据 |
| P0-3 校准与验证同目标 | ✅ 降级为 calibration target recovery; 独立验证计划: B 侧/未来块, reference 不参与训练 (r4) |
| P1-1 时间单位错 (t+600=6000s) | ✅ v1 脚本结果弃用; v2 使用 r18/r60 并已注明步长 |
| P1-2 best_gain 非 validation checkpoint | ✅ 承认; 正式口径改用 validation MAE 选点, gain 作门禁 (r4) |
| P1-3 R=50 表述 | ✅ 一律称 equal-percentage valve proxy |
| P1-4 K(x) 单位换算不可复核 | ✅ 承认; τ 饱和下界作为负面诊断; K 换算需固定 split + 每窗口 std (随 r2 修复) |
| P1-5 MAE +43% 同口径 | ✅ 修正为 ~+7.6% (各自 final checkpoint), 非公平估计 |
| P1-6 exp_201 逐 epoch 看 test | ✅ 承认; 作为历史 pilot, 不进正式 leaderboard; gain 试验迁入 src/phase35 (r4) |

### 执行顺序 (审计建议 1-5)

1. ✅ cache manifests 证据闭合 (docs/phase35_cache_evidence_2026-08-09.md)
2. 🔧 修 exp_201 flow 评估 offset + 单测; 撤下图中 truth/learns-the-nonlinearity 表述
3. 🔧 gain 试验迁入正式 Phase 3.5 口径 (validation-only)
4. 🔧 SP-event reference S/D 分层 + first-stage/balance/pretrend/placebo 审计 (1s 数据)
5. ⏳ 42-run 已执行 (commit 4f8d89a), 等审计解锁
