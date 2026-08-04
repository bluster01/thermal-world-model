# 世界模型三篇关键参考

> 保存时间: 2026-07-30

## 1. Differentiable World Model for Offline RL (arXiv 2603.22430, 2026)

**作者**: 未署名（多机构合作）
**核心思想**: 用扩散模型构建可微世界模型 s_{t+1} = f_θ(s_t, a_t, ε_t)，在推理时通过世界模型展开做 MPC 梯度优化

**模型设计**:
- 纯 Markov 假设: s_t, a_t → 扩散去噪 → s_{t+1}
- 扩散采样器 f_θ 完全可微，支持梯度反传
- 配套奖励模型 r_ξ 和终端价值函数 Q_φ
- 推理时: 从当前状态 s_t 展开 H 步 → Monte Carlo 估计梯度 → 更新策略参数 → 执行第一步

**局限对火电**: MuJoCo 环境满足 Markov 性（位置+速度完备），火电大迟延系统不满足

**要点**:
- 扩散模型作为 transition model 的创新
- 梯度通过扩散采样过程反传的定理(Theorem 4.1)
- MPC 中使用 Monte Carlo 近似 + 梯度上升更新策略参数
- D4RL MuJoCo 基准: 18个数据集，一致优于离线RL基线

## 2. Graph Spatiotemporal World Model Rolling MPC (Electronics 2026)

**作者**: Junling Liu, Xiaojun Wang, Leilei Wang, Yu Song
**单位**: 内蒙古电力集团 + 天津理工大学
**核心思想**: 用图时空世界模型学习多能耦合系统状态转移，嵌入 MPC 做经济调度

**模型设计（关键架构）**:
- 输入: 历史窗口 W 步(x_{t-W+1:t}, u_{t-W+1:t}, d_{t-W+1:t})
- Graph Encoder: 每步独立图消息传递（注意力加权）
- Temporal Encoder: GRU 序列编码 → z_t
- Decoder: z_t → s_{t+1}
- 自回归展开训练: 用多步 rollout loss + 物理一致性约束

**训练目标**:
- 多步预测 loss: Σ w_k * MSE(s_pred_k, s_true_k)
- 物理一致性: 平衡残差 + 存储边界 + 爬坡约束
- 直接训练 rollout（不是只训一步）

**MPC 集成**:
- 世界模型作为 MPC 的显式转移约束
- 分位数安全收紧: 预测残差 → 收紧约束边界
- 24步预测 NRMSE=4.28%，月运营成本降低 6.07%

**适用火电的关键点**:
- 历史窗口设计 → 解决大迟延
- 自回归展开训练 → 减少累积误差
- 物理一致性约束 → 可借鉴（温度上下限、阀位速率限制）
- MPC 显式嵌入 → 框架可直接复用

## 3. Neuromancer / Differentiable Predictive Control (PNNL)

**作者**: Jan Drgona, Aaron Tuor, Draguna Vrabie (PNNL)
**核心思想**: 用可微编程将 ML 模型嵌入 MPC，实现端到端控制策略学习

**开源**: github.com/pnnl/neuromancer (PyTorch, 1.4k stars)
**模型类型**: Physics-informed NN, Neural ODE, Koopman operator, RNN/GRU

**DPC 工作流**:
- 系统辨识: 用神经网络学习动力学（支持物理约束）
- 控制策略: 可微 MPC 或用 RL 微调
- 在线部署: 滚动优化，支持约束处理

**要点**:
- Neuromancer 框架的模块化设计可参考
- 支持多种模型类型（ODE/RNN/Transformer）
- 内置约束处理和安全保障
- 建筑物 HVAC 控制案例（类比火电过程控制）

## 关键结论

1. **不能做纯 Markov** — DWM 的设计仅适用于完备状态系统
2. **必须用历史窗口** — Graph World Model 是正确方向
3. **训练必须包含多步 rollout** — 不能只训一步预测
4. **物理约束可嵌入训练** — 温度上下限、阀位速率
5. **MPC 显式嵌入世界模型** — 不是分离的预测+优化

---

## 4. Objective Mismatch — 预测精度 ≠ 控制效用 (Phase 2 叙事重构新增)

### 4.1 Lambert et al. 2020 — 理论框架

**标题**: Objective Mismatch in Model-based Reinforcement Learning
**arXiv**: 2102.03023

- 世界模型训练目标 (一步预测似然) 与下游目标 (闭环控制性能) 不相关
- 全局精确的模型既非充分也非必要; 任务局部精确即可
- **本项目对应**: WM 开环 MAE 0.31°C (精度好), 但 S3 持续阶跃因果反演 (控制效用差) — 典型 objective mismatch

### 4.2 Closed-Loop Performance Prediction — 实证

**arXiv**: 2607.01736 (2026)

- validation loss 和多步 RMSE 持续改善, 但闭环性能早已崩溃
- 最强预测因子是 Reward Observability Fraction (ROF), 不是预测精度
- **本项目对应**: 不能用开环 MAE 作为闭环 MPC 有效性的判据

### 4.3 Train-Test Gap — 分布差异

**arXiv**: 2512.09929 (2025)

- 训练数据 = 专家/行为策略轨迹; 测试时 = 规划器产生的动作序列 → OOD
- 规划轨迹的预测误差系统性高于专家轨迹
- **本项目对应**: 训练数据 = 运行员自然操作; MPC 大幅阶跃超出分布 → 退化到共因方向; 小幅动作 (<10%) 在分布内 → 方向正确

### 4.4 RC-aux — 预测准确 ≠ 可规划

**arXiv**: 2605.07278 (2026)

- 短程预测训练 vs 长程规划搜索 → 时空 mismatch
- Euclidean 距离不反映有限步可达性
- **本项目对应**: WM 短程预测好, 但长程 rollout (H=18) 下因果方向退化

### 4.5 Kinematic Not Dynamic — rollout 诊断

**arXiv**: 2607.05966 (2026)

- WM imagined rollout 是运动学 (kinematic) 而非动力学 (dynamic)
- 摩擦不变性: 物理响应随摩擦变化, WM rollout 不变
- **本项目对应**: WM rollout 在大幅扰动下不反映真实物理响应

### 4.6 WM Evaluation Ladder — 评估框架

**arXiv**: 2606.15032 (2026)

- L0-L7 评估阶梯: 视觉合理性 → 干预推理 → 策略评估 → 策略优化
- 开环评估 (L1) vs 闭环评估 (L4): 很多 WM 开环强、闭环骤降
- **本项目对应**: Phase 1 = L1-L2 (开环预测+因果敏感性); Phase 2 = L4 (闭环) 发现边界

---

## 5. 工业 WM 定位 — 预测+建议而非闭环控制 (Phase 2 叙事重构新增)

### 5.1 Actionable World Models for Industrial Process Control

**arXiv**: 2503.01411 (2025), IEEE SDS 2025

- JEPA + contrastive learning 学习 action-aware 表示
- 不做闭环控制, 而是: 预测动作后果 → 为操作员提供 control action 建议
- 注塑成型案例: 80 样本训练, 实时调整控制参数
- **本项目对应**: §25.6 转向"预测驱动+监督模式"的文献先例

### 5.2 工业神经网络控制器现状

**来源**: noga.es/en/blog/nn-controllers-real-industrial-deployments (2025)

- 全球确认在真实工业系统上闭环运行的 NN 控制器 ≈ 10 个
- Digital twin = 离线仿真, 不闭环; Soft sensor = 监督角色, 有时闭环
- "MPC already works. The business case for NN control must compare against best-in-class MPC, not against a poorly tuned PID baseline."
- **本项目对应**: WM 应定位为 digital twin / soft sensor 层 (预测+监督), 不是 NN controller 层

### 5.3 Graph Spatiotemporal WM Rolling MPC (已有, 重新定位)

**作者**: Junling Liu et al., Electronics 2026

- 历史窗口 + 自回归展开 + 物理一致性约束 + MPC 显式嵌入
- 24步预测 NRMSE=4.28%, 月运营成本降低 6.07%
- **重新定位**: 从"我们的 MPC 也可以做到"变为"多能耦合场景的 WM-MPC 参考, 但火电主汽温动作通道弱因果, 需要不同策略"
