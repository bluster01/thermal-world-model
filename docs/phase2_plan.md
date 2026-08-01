# Phase 2 规划：离线 MPC 验证（含 Phase 1 审稿人评审）

> 日期: 2026-08-01 | 作者: cwoc + Hermes
> 方法: nature-reviewer skill（3 审稿人）+ academic-paper skill（贡献叙事驱动）
> 依据: phase1_status.md、experiment_design.md v0.1、exp_001-017、事件研究

---

## 0. 一句话结论

**重大修改（Major Revision）级结论：Phase 1 的每个致命问题都有明确修复实验，且耗时 1-2 天——不是方向性错误，是完整性缺失。Phase 2 的核心任务是让"贡献 4：能回答 if-then"在数据上站住。**

---

## 1. nature-reviewer 评审摘要（3 审稿人 + 综合）

### Reviewer 1（因果推断/方法学）
- **R1-M1 (CRITICAL)** 共因混杂处理是 post-hoc 叙事而非预设设计。exp_013-016 五个实验（~3 天算力）验证了一个错误前提（单步符号正则）。正确顺序：先事件研究（30 分钟）确立物理时标，再设计约束
- **R1-M2 (MAJOR)** "动作条件化"因果证据仍缺失：模型响应 vs 事件研究真值直接对比从未做，t1=+0.186 是共因拟合
- **R1-M3 (MAJOR)** exp_015 早停 bug（best@17<warmup20）→"未受正则的模型"被当正则结果分析，需排查同类结论

### Reviewer 2（工业控制/系统辨识）
- **R2-M1 (CRITICAL)** 零 baseline：LSTM dynamics / n4sid / DeepThermal / Exp-0 全部没有
- **R2-M2 (MAJOR)** "可微"卖点未兑现：MPC 方案 A 是随机采样+Top-K，不利用可微性
- **R2-M3 (MAJOR)** 训练-推理错配：K=5 训练（50s）vs H=18 推理（180s），物理响应 120s+ 无监督覆盖
- **R2-M4 (MINOR)** MPC 约束未用 DCS 实测阀位速率限制

### Reviewer 3（时序预测/不确定性）
- **R3-M1 (CRITICAL)** σ 校准系统性失败：|error|/σ 0.34→5.28，σ 随步反缩，probabilistic 卖点崩塌
- **R3-M2 (MAJOR)** 敏感性评测时标错误贯穿实验史（t1/t3 vs 物理 60-120s）
- **R3-M3 (MINOR)** 种子固定策略未文档化

### Cross-review 共识风险
1. **因果声称无支撑**（核心 claim 证据最弱）
2. **零 baseline**（无对比 = 无位置）
3. **训练-推理-物理三错配**（时标三者不一致）

---

## 2. 论文故事线（Phase 2 要服务的）

```
图1 动机: 火电大迟延 + PID 共因 → 预测模型不够, 需要因果世界模型
图2 方法: 滞后动作嵌入 + 长程监督 + (可选)长程正则
图3 结果: MPC 反事实仿真 vs PID 真实轨迹 → 温度偏差↓ 超调↓
表1 消融: 滞后/正则/时标
表2 baseline: vs LSTM/n4sid/Exp-0
表3 分工况: 11 工况误差与控制质量
```

---

## 3. Phase 1.5（前置修复，1-2 天）— 三个 CRITICAL 的修复

1. **1c 因果证实**：模型响应曲线 vs 事件研究真值曲线直接对比（开/关阀双侧，H=8-12），量化方向+幅度+时标匹配度
2. **1e baseline**：LSTM dynamics + n4sid + Exp-0 直接多步预测（同 test 集、同 seed）
3. **11 工况拆分评估**：单脚本，论文必需

**依赖**: exp_017_B/C 结果决定 1c 最终方案（K=12 长监督够用→纯验证；不够→加长程正则调参）

---

## 4. Phase 2 主实验设计（修订 experiment_design §3）

| 实验 | 设计 | 成功标准 | 论文位置 |
|---|---|---|---|
| 2a 反事实仿真 | 50 条测试轨迹，MPC H=10 与 PID 对比 | 温度 RMSE 降 20%+ 且不违反约束 | 图3 |
| 2b 约束满足 | 超温次数/幅度 | 0 次超温上限 | 表 |
| 2c 动作代价 | 总阀位变化 TV | MPC TV ≤ 1.5×PID | 表 |
| 2d 分工况 | 11 工况逐工况 | 最差工况不劣于 PID | 表3 |

### MPC 求解器决策（回应 R2-M2）
- 先做**随机采样+Top-K（random shooting）**：1 天内出图，但论文须注明非可微优化
- 时间允许则升级 **CEM（交叉熵法）**：10 行代码，比 random shooting 好一个量级，仍属采样类
- 目标函数: `(T-T_set)² + λ₁Δv₁² + λ₂Δv₂²`，约束 `540≤T≤575`、`|Δv|≤5%/step`
- **H=10（100s）**：物理响应区起点，兼顾计算量。不用 H=5（看不到阀效应，MPC 退化为纯预测器——exp_014 已证实的坑）

### Phase 2 诚实边界（Discussion 素材）
- 离线反事实 ≠ 在线闭环：模型误差累积，MPC 依赖每步状态重置（data assimilation）
- 共因混杂未完全消除：长程响应仍可能带 PID 统计残留
- 可微优化未兑现：**建议论文标题弱化为 "action-conditioned world model" 而非 "differentiable"**

---

## 5. 路线图

```
Phase 1.5 (1-2天): 1c证实 → baseline → 11工况     ← 等 exp_017_B/C 结果定 1c 方案
Phase 2   (3-4天): random shooting MPC → CEM → 反事实仿真 → 分工况
论文骨架   (并行): tech-paper-template 起骨架, intro-drafter 写引言
```

---

## 6. 论文定位修正建议

- 标题方向: "Action-conditioned world model for main steam temperature control in thermal power plants"（弱化 differentiable）
- 核心贡献重排:
  1. 事件研究驱动的物理时标验证方法论（这是独有资产，审稿人认可的严谨性来源）
  2. 滞后动作嵌入 + 长程监督的世界模型（方法）
  3. 真实 1000MW 机组数据 + 11 工况 + MPC 反事实仿真（应用）
