# 对照论文原文 — 方向指导

> 保存时间: 2026-08-05
> 来源: docs/references.md (2026-07-30 建立的核心参考清单)
> 用途: 论文方向指导 — 世界模型预测驱动 + 现场监督模式

## 三篇核心对照论文

### 1. Differentiable World Model for Offline RL
- **文件**: `dwm_offline_rl_2603.22430.pdf` (arXiv 2603.22430, 2026, 14页)
- **核心**: 扩散世界模型 s_{t+1}=f_θ(s_t,a_t,ε_t) + 推理时梯度 MPC
- **对当前方向的指导**:
  - 纯 Markov 假设不适用火电大迟延 — 必须用历史窗口
  - 可微世界模型 + 轨迹优化 = 我们的沙盒优化 SP 曲线的技术路线
  - 局限警示: 完备状态环境才成立 — 火电温度由未观测工况主导,
    控制动作因果弱 — 这正是我们放弃"控阀"主线的理论依据

### 2. Graph Spatiotemporal World-Model-Driven Rolling MPC
- **文件**: `graph_worldmodel_mpc_electronics_2231.pdf`
  (Electronics 15(11):2231, DOI 10.3390/electronics15112231, 2026, 32页)
- **作者**: Junling Liu, Xiaojun Wang, Leilei Wang, Yu Song
  (内蒙古电力集团 + 天津理工大学)
- **核心**: 图时空世界模型 (Graph Encoder + GRU + 自回归展开训练)
  + 物理一致性约束 + 显式嵌入 MPC
- **对当前方向的指导**:
  - **历史窗口设计** → 解决大迟延 (与我们 W=96 一致)
  - **自回归展开训练** → 减少累积误差 (我们 rollout 训练的做法来源)
  - **物理一致性约束** → 温度上下限、阀位速率限制可嵌入
  - 24步预测 NRMSE=4.28%, 月运营成本降 6.07% — "预测精度+经济调度"
    的论文叙事范式 (类似我们的"预测精度+SP优化")

### 3. Differentiable Predictive Control (DPC) / Neuromancer
- **文件**: `neuromancer_dpc_2011.03699.pdf` (arXiv 2011.03699v2, 28页)
- **作者**: Ján Drgoňa, Karol Kiš, Aaron Tuor, Draguna Vrabie, Martin Klaúčo (PNNL)
- **核心**: 神经状态空间模型 + 通过闭环动力学反传的 MPC 损失优化
- **对当前方向的指导**:
  - **可微闭环训练** — 控制策略端到端优化
  - **建筑 HVAC 案例** — 类比火电过程控制 (监督式设定值控制)
  - Neuromancer 开源框架 (github.com/pnnl/neuromancer) 可参考实现

## 与新方向 (预测驱动 + 监督模式) 的映射

| 论文要点 | 我们的落地 |
|---|---|
| 历史窗口 + 自回归训练 | M5/M7 世界模型 (W=96, rollout 18 步) |
| 可微轨迹优化 | 沙盒中 MPC 优化 SP 曲线 |
| 物理一致性约束 | 温度上下限/SP 速率限制 (监督模式安全) |
| 预测精度 + 调度收益 | 沙盒 MAE 0.30°C + 现场前馈效果 |

## 备注

- 下载时间 2026-08-05, 均已验证 PDF 完整 (14/32/28 页)
- references.md (核心三篇) + phase1_references.md (13 篇完整引用清单)
