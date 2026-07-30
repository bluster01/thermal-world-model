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
