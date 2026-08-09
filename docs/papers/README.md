# 对照论文原文 — 方向指导

> 保存时间: 2026-08-05
> 来源: docs/references.md (2026-07-30 核心三篇 + 2026-08-05 叙事重构新增 7 篇)
> 用途: 论文方向指导 — 世界模型预测驱动 + 现场监督模式
> 对应文档: docs/narrative_restructure.md (叙事重构方案)

## A. 核心对照论文 (2026-07-30 建立)

### 1. Differentiable World Model for Offline RL
- **文件**: `dwm_offline_rl_2603.22430.pdf` (arXiv 2603.22430, 2026, 14页)
- **核心**: 扩散世界模型 s_{t+1}=f_θ(s_t,a_t,ε_t) + 推理时梯度 MPC
- **指导意义**: 纯 Markov 假设不适用火电大迟延 — 必须用历史窗口; 可微世界模型
  + 轨迹优化 = 沙盒优化 SP 曲线的技术路线; 局限警示支撑"放弃控阀"决策

### 2. Graph Spatiotemporal World-Model-Driven Rolling MPC
- **文件**: `graph_worldmodel_mpc_electronics_2231.pdf`
  (Electronics 15(11):2231, DOI 10.3390/electronics15112231, 2026, 32页)
- **作者**: Junling Liu et al. (内蒙古电力集团 + 天津理工大学)
- **核心**: 图时空世界模型 + 物理一致性约束 + 显式嵌入 MPC; 24步 NRMSE=4.28%
- **指导意义**: 历史窗口/自回归展开训练/物理约束 → 我们的 W=96/rollout 18 步;
  "预测精度+调度收益"叙事范式

### 3. Differentiable Predictive Control (DPC) / Neuromancer
- **文件**: `neuromancer_dpc_2011.03699.pdf` (arXiv 2011.03699v2, 28页)
- **作者**: Ján Drgoňa et al. (PNNL)
- **核心**: 神经状态空间模型 + 通过闭环动力学反传的 MPC 损失优化
- **指导意义**: 可微闭环训练; 建筑 HVAC 案例类比火电; 开源框架可参考

## B. Objective Mismatch 框架 (2026-08-05 叙事重构新增)

### 4. Objective Mismatch in Model-based RL
- **文件**: `objective_mismatch_2102.03023.pdf` (arXiv 2102.03023, 20页)
- **作者**: Nathan Lambert et al. (2020)
- **核心**: 训练目标 (一步预测似然) 与下游目标 (闭环性能) 不相关; 任务局部精确即可
- **本项目对应**: WM 开环 MAE 0.31°C 但持续阶跃因果反演 — 典型 objective mismatch

### 5. Closed-Loop Performance Prediction
- **文件**: `closedloop_perf_pred_2607.01736.pdf` (arXiv 2607.01736, 2026, 19页)
- **核心**: validation loss 持续改善但闭环性能崩溃; ROF 是最强预测因子
- **本项目对应**: 不能用开环 MAE 判据闭环 MPC 有效性

### 6. Train-Test Gap (分布差异)
- **文件**: `train_test_gap_2512.09929.pdf` (arXiv 2512.09929, 2025, 25页)
- **核心**: 训练=行为策略轨迹, 测试=规划器动作 → OOD; 规划轨迹误差系统性更高
- **本项目对应**: 训练=运行员自然操作; MPC 大幅阶跃超分布 → 退化共因方向

### 7. RC-aux (预测准确 ≠ 可规划)
- **文件**: `rc_aux_2605.07278.pdf` (arXiv 2605.07278, 2026, 35页)
- **核心**: 短程预测训练 vs 长程规划搜索 → 时空 mismatch
- **本项目对应**: WM 短程好但 H=18 rollout 因果方向退化

### 8. Kinematic Not Dynamic (rollout 诊断)
- **文件**: `kinematic_not_dynamic_2607.05966.pdf` (arXiv 2607.05966, 2026, 9页)
- **核心**: WM imagined rollout 是运动学而非动力学; 摩擦不变性
- **本项目对应**: WM rollout 在大幅扰动下不反映真实物理响应

### 9. WM Evaluation Ladder
- **文件**: `wm_eval_ladder_2606.15032.pdf` (arXiv 2606.15032, 2026, 27页)
- **核心**: L0-L7 评估阶梯 (视觉→干预推理→策略评估→策略优化)
- **本项目对应**: Phase 1 = L1-L2 (开环+因果), Phase 2 = L4 (闭环) 发现边界

### 10. Actionable World Models for Industrial Process Control
- **文件**: `actionable_wm_2503.01411.pdf` (arXiv 2503.01411, IEEE SDS 2025, 8页)
- **核心**: JEPA + contrastive learning; 不做闭环控制, 提供 control action 建议
- **本项目对应**: §25.6 转向"预测驱动+监督模式"的文献先例 — 最重要的一篇

## 与新方向 (预测驱动 + 监督模式) 的映射

| 论文要点 | 我们的落地 |
|---|---|
| 历史窗口 + 自回归训练 | M5/M7 世界模型 (W=96, rollout 18 步) |
| 可微轨迹优化 | 沙盒中 MPC 优化 SP 曲线 |
| 物理一致性约束 | 温度上下限/SP 速率限制 (监督模式安全) |
| 预测精度 + 调度收益 | 沙盒 MAE 0.30°C + 现场前馈效果 |
| objective mismatch / OOD / RC-aux | 控阀失败的 4 条理论解释 (叙事重构核心) |
| Actionable WM (不做闭环) | 监督模式定位的直接文献先例 |

## 备注

- 全部 PDF 已验证完整 (8-35 页)
- phase1_references.md 另有 13 篇组件/基线引用 (RevIN/PatchTST/β-NLL/iTransformer/DLinear/Mamba/N4SID/TD-MPC2/DreamerV3/PETS)

## C. 主汽温系统辨识方法论文 (2026-08-09 新增, E3 方法学修正)

### 11. Cao 2021 主蒸汽温度系统辨识 (导前区/惰性区分段)
- **文件**: `cao2021_msts_pso_identification.pdf` (系统仿真学报 33(10), 9页)
- **核心**: 直接闭环辨识 + 改进PSO; 导前区(阀位→减温器出口θ2) + 惰性区(θ2→主汽温θ1)
  分别拟合 G(s)=K·e^(−τs)/(Ts+1); 3s采样6000点, 前3000辨识后3000验证
- **指导意义**: E3 不应只用事件方向率 — 应做分段 FOPDT 传递函数辨识

### 12. Brolese 2022 水管锅炉汽温控制 (PoliMi 硕士论文)
- **文件**: `IDENTIFICATION_METHODS_SUMMARY_2026-08-09.md` 内附全文缓存路径
- **核心**: 喷水减温器=代数混合块(无动态, 负增益 K<0), 动态在阀门低通(τ≈1s)
  + 过热器金属蓄热(惰性区 τ_outer≈30s, k≈0.5); 阶跃辨识 k=g2/g1, τ=settle/5
- **指导意义**: 导前区快(秒~数十秒)、惰性区慢(数十秒~分钟) 的物理先验

### 方法学修正 (2026-08-09)
事件方向率 (旧) → 分段 FOPDT 辨识 (新): 阀位→减温器出口(导前区)→末过出口(惰性区),
直接闭环数据辨识, 输出 K/T/τ 参数而非 0-100% 方向率。详见
`IDENTIFICATION_METHODS_SUMMARY_2026-08-09.md`
