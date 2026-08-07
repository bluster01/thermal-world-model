# 论文叙事重构方案 (2026-08-05)

> **历史叙事 / 已撤回作为论文方案。** 文中的“因果响应实证”“现场实证”和相关 CFI 数字没有达到当前 Supervisor 的识别、独立测试和产物标准；仓库也没有足以核验“已投运现场实证”的 artifact。新论文主线与证据边界见 [`SUPERVISOR_REVIEW_2026-08-07.md`](SUPERVISOR_REVIEW_2026-08-07.md) 和 [`PHASE4_EXPERIMENT_PLAN.md`](PHASE4_EXPERIMENT_PLAN.md)。

> 基于: phase2_final_audit.md (数据审查) + §25.6 (方向决策) + 文献调研

---

## 1. 核心叙事转变

**旧叙事**: "我们用深度世界模型做 MPC, 比 PID 控制更好"

**新叙事 (v2, 因果主线)**: "我们构建了主汽温世界模型, **以干预因果响应为核心主线** — 设计动作参与注意力的架构 (M9DSP), 实证其能学习 ΔSP→温度的因果方向与中程增益; 在此基础上落地预测驱动+监督模式"

| 维度 | 旧 | 新 |
|------|-----|-----|
| 定位 | 控制器替代 | **干预响应可学的世界模型** + 预测顾问 |
| 主结果 | MPC vs PID 闭环对比 | **动作因果保真度 (方向/增益/时标)** + 预测精度 + 现场监督实证 |
| 架构卖点 | VarAttn 消融 | **动作 cross-attention 注入** (动作 token 参与每层注意力) |
| 因果 | "无翻转" (错误) | **中程方向可靠 (89%@180s) / 远程受限 (共因)** — 架构×数据双因素 |
| Phase 2 | 核心贡献 | 方法论探索 (发现边界) |
| TV | 深度 WM 优势 | MPC 框架结构性优势, 非模型独有 |

---

## 2. 论文结构

### I. 引言

- 火电灵活性需求: 深调峰、快速变负荷 → 主汽温控制是关键瓶颈
- 现有 PID 控制的局限: 纯反馈, 无预见性, 大迟延 (60-90s) 下超调
- 世界模型的潜力: 学习动态转移, 支持前瞻预测
- **本文定位**: 不是用 WM 替代 PID 闭环控制, 而是用 WM 做预测驱动+监督模式
- 引用: DreamerV3 (WM 通用性), TD-MPC2 (task-oriented), 工业数字孪生文献

### II. 世界模型架构

- **M9DSP**: TimeXer 式动作 cross-attention — 动作以 token 参与每层注意力 (act_attn: GLB↔动作), 非展平拼接
- 对比 DirectWM (动作 Linear 展平→decoder 稠密混合, 不经过任何注意力)
- RevIN + PerVarTCN + VarAttn + β-NLL; 40维输入 → 1维目标 (末级过热器出口汽温)
- task-oriented 设计 (TD-MPC2 哲学): 只预测控制相关变量
- **架构-因果证据** (varattn_causality_analysis.md): 动作注入方式是因果保真度的决定因素 — 展平注入长时程衰减 33-48%, 动作 cross-attn 单调增长 (+40/+48%)
- 引用: RevIN (Kim 2022), PatchTST (Nie 2023), β-NLL (Seitzer 2022), iTransformer (Liu 2024), TimeXer (2025)

### III. 开环预测能力 (Phase 1 主线)

- **统一协议消融** (exp_025): M0-M7 + B1-B6, 同协议对比
  - RevIN 必需 (去掉 MAE ×70)
  - per-variable TCN 结构关键
  - VarAttn 无害但不提升精度
  - 单目标合理 (信息量集中)
- **保真度** (exp_044): 1200s MAE 1.2°C, 亚线性收敛
- **基线对比**: 线性 SSM (N4SID) 长程发散 → 非线性必要; DLinear/Mamba/iTransformer 基线
- **概率校准** (exp_024): β-NLL σ=0.87, 优于 persistence 31%
- 引用: N4SID (Van Overschee 1994), DLinear (Zeng 2023), Mamba (Gu 2023), PETS (Chua 2018)

### IV. 因果保真度: 主线证据链 (exp_096-102 + 事件研究)

> **本章是核心贡献**: 世界模型能否学习干预 (ΔSP) 的因果响应? 我们给出了"能, 但有时标边界"的实证。

- **物理基准** (exp_093/099): SP 跟随时标 600s (97%), 180s 仅 17% (0.36°C@ΔSP2.07); 前 60s 零响应纯延迟
- **事件研究** (真实数据): 物理滞后 60-90s, 开阀 120s 后 −0.59°C
- **架构决定因果保真度**:
  - M5-DSP (展平注入): 180s 方向 75%, 响应 0.05°C (物理 14%)
  - M7-DSP (概率+展平): H=60 下 600s 方向 45%≈随机; 长时程响应衰减 (与 varattn 分析观察 C 同构)
  - **M9DSP (动作 cross-attn)**: **180s 方向 89%**, 300s 84%; 中程响应 +28-32%; 短窗 MAE 0.361
- **强化训练不可行** (exp_098/099/100): action dropout 无效 (0.06°C); 物理叠加过冲 6 倍; H=60 覆盖 97% 时标仍 FAIL (600s 方向 41-45%)
- **边界**: 中程 (≤300s) 方向可靠; 远程 (>300s) 受共因混杂限制 — 架构×数据双因素
- **诊断**: 观测数据下 WM 学到条件期望 E[T|a] 为主, 干预效应 P(T|do(a)) 需架构支撑 (动作参与注意力) + 时标匹配
- 引用: Lambert 2020 (objective mismatch), Pearl因果阶梯

### V. 从预测到控制的边界 (Phase 2 方法论探索)

> **本章不是"我们比 PID 好", 而是"我们发现了什么做不到"**

- **MPC 仿真实验** (S1-S6): 在世界模型仿真中对比 MPC vs PID
  - RMSE 不显著 (p=0.25-0.86): 动作通道弱因果, 工况主导, 测不出控制效果
  - TV 降低: 归因于 MPC 框架 (预测+优化), 非深度 WM — 线性 MPC TV 更低 (S6)
  - S3 因果安全结论无效 (判定符号反 + 2/3 反演)
- **Objective mismatch**: 开环精度好 (MAE 0.31°C) ≠ 闭环控制有效
  - 训练数据 = 运行员自然轨迹; MPC 规划动作超出分布 → 退化
  - 小幅动作 (<10%, MPC 式) 落在训练分布内 → 方向正确
  - 大幅持续阶跃 (>10%, 运行员式) 超出分布 → 共因方向反演
- **仿真对象同构**: MPC 预测模型与 plant 同架构 → 闭环对比不具备外推有效性
- 引用: Lambert 2020, train-test gap (2512.09929), closed-loop performance prediction (2607.01736), WM evaluation ladder (2606.15032)

### VI. 预测驱动+监督模式 (因果主线的落地)

- **SP 通道** (exp_093): 阶跃跟随率 92%, 增益 −2.0%/°C; 时标 600s (180s 仅 17%)
- **沙盒预测** (exp_095): SP 操作场景 180s MAE 0.30°C
- **ΔSP 建模** (exp_096/101/102): M9DSP 180s 方向 89% — 预测器在 180s 窗口内对干预方向的判断可靠
- **现场实证**: 已投运 180s 预测前馈 + 误差补偿 + 前后效果数据
- **架构**: WM 预测 (含干预方向) → 前馈补偿 → PID 反馈 (人机协同, 非替代)
- 引用: Actionable WM for Industrial Process Control (2503.01411), Neuromancer (Drgona 2023), Graph WM Rolling MPC (Liu 2026)

### VII. 方法论教训

1. **有预测 ≠ 有因果**: 条件期望 vs 干预效应 (Pearl L1 vs L2)
2. **有敏感性 ≠ 有物理**: LSTM 共因方向 vs WM 干预方向
3. **开环精度 ≠ 闭环效用**: objective mismatch (Lambert 2020)
4. **仿真闭环 ≠ 真实闭环**: 同构 plant 不具备外推有效性
5. **符号正则 = 伪物理**: 把物理滞后压成伪响应
6. **TV 优势归因**: MPC 框架结构性差异, 非深度 WM 独有
7. **工业 WM 定位**: 预测器+顾问 > 控制器替代

---

## 3. 新增引用

| # | 论文 | arXiv/DOI | 用途 |
|---|------|-----------|------|
| 14 | Lambert et al. 2020, "Objective Mismatch in MBRL" | arXiv:2102.03023 | §IV/V: 预测精度≠控制效用 |
| 15 | "Predicting Closed-Loop Performance of Latent WM" | arXiv:2607.01736 | §V: 开环精度与闭环性能脱节 |
| 16 | "Closing the Train-Test Gap in WM" | arXiv:2512.09929 | §V: 训练分布vs规划分布差异 |
| 17 | "Actionable WM for Industrial Process Control" | arXiv:2503.01411 | §VI: 工业WM做预测+建议范式 |
| 18 | "How Should WM Be Evaluated for Embodied DM" | arXiv:2606.15032 | §V: L0-L7评估阶梯 |
| 19 | "Predictive but Not Plannable: RC-aux" | arXiv:2605.07278 | §V: 预测准确≠可规划 |
| 20 | "Imagined Rollouts are Kinematic, Not Dynamic" | arXiv:2607.05966 | §V: WM rollout vs 真实动力学差异 |

---

## 4. 与现有文档的对应

| 现有文档 | 在新叙事中的角色 |
|---------|----------------|
| Phase 1 实验全部 | §II-IV 主体 (开环预测+消融+因果) |
| Phase 2 S1-S6 | §V "边界探索" (诚实报告局限性) |
| experiment_audit.md | §VII "方法论教训" 的素材 |
| §25.6 决策 | §VI 落地方向的决策依据 |
| phase2_final_audit.md | §V 数据问题的详细支撑 |
