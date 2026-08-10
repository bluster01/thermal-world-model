# Phase 3.5-MS2-J 联合模块与分阶段训练设计

> 状态：frozen for local implementation and Linux validation。只开放 validation；synthetic test 未授权。证据范围为 `synthetic_joint_coupling_validation_not_field_causality`。

## 1. 为什么是这一 Gate

MS2-V 已证明在 R50 合成真值下，显式单调模块相对同一二阶灰箱的 identity 假设显著降低响应误差；MS2-C 已证明读取 context 的变参数表示显著优于全局参数。但两个模块此前位于不同生成器中，不能回答它们同时存在时是否互相补偿、参数塌缩或训练不收敛。

本轮考虑过三条路径：直接进入 MS3 真实 A/B 适配、一次加入纯迟延/三阶/未建模扰动、先做联合耦合。直接进入 MS3 会把优化失败与现场不可辨识混合；广义 MS2-D 会同时改变过多机制。故选择最小的 MS2-J：只合并已经单独验证过的两个真值轴，并对 joint-from-scratch 与 staged training 作可审计比较。纯迟延、阶次扩展、扰动和真实数据继续 HOLD。

## 2. 已知真值与 estimand

统一 estimand 仍为参考阀位轨迹相对于实际阀位轨迹的温度增量响应：

\[
g(c,a,r)=T(c,a)-T(c,r),\qquad g(c,r,r)=0.
\]

真值同时使用 equal-percentage R50 有效开度和 context-dependent gain/time constants：

\[
u_t^*=\phi_{R50}(a_t)-\phi_{R50}(r_t),
\]

\[
K(c)=K_0\exp\{0.35\tanh(c_0)\},\quad
\tau_j(c)=\tau_{j,0}\exp\{0.30\tanh(c_{j})\}.
\]

其余冻结值：`K0=-0.10 °C/effective-%`、`tau=[70,210] s`、`dt=10 s`、horizon 60、train/validation/test=`1024/256/256`、noise `0.02 °C`。动作仍含 hold/step/pulse/ramp/multi-step。该系统是 synthetic known truth，不把阀位解释成喷水质量流量，也不证明现场的 R50 或 K/τ 形式。

## 3. 冻结候选矩阵

单一 regime，9 candidates × 3 seeds = **27 validation runs**：

| Candidate | 非线性开度 | context 调度 | 训练 | 角色 |
|---|---|---|---|---|
| `j_g2_identity_global` | 无 | 无 | joint | 双缺失负对照 |
| `j_g2_monotone_global` | learned monotone | 无 | joint | 仅阀门模块 |
| `j_g2_identity_scheduled` | 无 | 有 | joint | 仅调度模块 |
| `j_g2_monotone_scheduled_joint` | learned monotone | 有 | joint | 双模块主模型 |
| `j_g2_monotone_scheduled_staged` | learned monotone | 有 | staged | 稳定性主对照 |
| `j_g2_r50_scheduled` | true R50 prior | 有 | joint | opening-map 正对照；不参加冠军 |
| `j_k4_monotone` | learned monotone | 无显式调度 | joint | Koopman-family 次要对照 |
| `j_pi_monotone` | learned monotone | neural context closure | joint | PI-ODE 次要对照 |
| `j_deeponet` | raw opening，可隐式非线性 | causal context branch | joint | 灵活算子次要对照 |

各路线共享 batch、总 epoch cap、validation selector 和结构门禁。次要路线只回答表示能力，不把数值最低者事后改写成物理冠军。

## 4. Staged training 冻结协议

总 cap 与 joint 模型相同为 300 epochs，分成：

1. **Stage A（120）**：schedule 权重固定为零，只训练 base K/τ 与 monotone opening；学习跨工况平均响应。
2. **Stage B（90）**：冻结 base K/τ 与 opening，只训练 gain/tau schedules；学习工况偏移。
3. **Stage C（90）**：全部解冻，以主学习率的 0.2 倍联合微调。

每个 stage 使用同一 noisy validation effect MAE 选该 stage checkpoint；下一 stage 从上一 stage 的最佳权重开始。最终 checkpoint 只能来自 Stage C（允许 Stage C epoch 0 保留 Stage B 边界权重），并保存 A/B/C checkpoint SHA、每阶段 optimizer updates、validation 指标和 trainable parameter names。不得根据 validation 改阶段长度或解冻顺序。

## 5. 预注册判决

所有 27 runs 首先必须满足 reference identity=0、future-action leakage=0、有限 rollout/state、正开阀长期响应为负；Graybox/Koopman 谱半径必须小于 1。

两个主要判决分别处理表示和优化：

1. **联合模块增量价值**：`j_g2_monotone_scheduled_joint` 相对两个单模块消融 `j_g2_monotone_global` 与 `j_g2_identity_scheduled`，三个 seed 的 clean NMAE 均至少改善 20%。
2. **staged 稳定性**：staged 三个 seed 全部完成 A/B/C，无非有限损失或参数；最终 clean NMAE 不超过 joint-from-scratch 的 1.10 倍；并且最终相对 Stage A 边界至少改善 20%。这是一项 non-inferiority/stability Gate，不预设 staged 必须优于 joint。

validation 只用于上述 screening。test 必须在本地审计 checkpoint、阶段日志和参数健康后另行授权；Linux 不得自行打开 test、追加 seed 或改变主对比。

## 6. 可复现性补丁

上一轮发现 Torch 2.11/CUDA 与 Torch 2.5/CPU 会生成不同的 RNG trajectory digest，虽然权重重算指标差异仅 1.47%。因此 MS2-J 每个 manifest/ledger 必须记录 Python、Torch、CUDA、device 和平台版本；validation batch 记录 trajectory design SHA。跨环境复算按 environment-sensitive 协议，以结构一致和主指标相对差 <10% 判定，不再声称 RNG trajectory 跨 Torch 版本逐位相同。

