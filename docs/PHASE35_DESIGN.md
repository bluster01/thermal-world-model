# Phase 3.5 — 从闭环SP到plant valve：estimand修正

> 日期：2026-08-08
> 状态：设计稿 — 等实验验证后 push
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
