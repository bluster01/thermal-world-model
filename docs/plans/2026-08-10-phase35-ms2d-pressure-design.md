# Phase 3.5-MS2-D 结构压力设计

## Material Passport

- Material Type: sequential synthetic experiment design
- Evidence Scope: `synthetic_delay_pressure_validation`
- Status: CODE/PROTOCOL FROZEN；READY FOR LINUX VALIDATION
- Upstream Evidence: MS1、MS2-V/C/J validation+test
- Field Boundary: 不读取 A/B 现场 test；不恢复已废弃 E1–E5

## 1. 目的

MS2-V/C/J 已证明在二阶、R50 有效开度和 context 调度的已知真值中，联合灰箱能够恢复多步响应。但该真值仍没有显式纯迟延、额外惯性阶次或动作无关扰动。MS2-D 用顺序 Gate 判断当前正结论是否依赖过于理想的生成器，而不是一次铺开大矩阵。

## 2. 三个顺序子 Gate

| Gate | 单独改变的真值轴 | 主要问题 | 状态 |
|---|---|---|---|
| D1 | 固定 20 s 纯迟延 | 显式 causal delay module 是否优于无迟延灰箱？ | 本轮实现 |
| D2 | 二阶改为三阶串联惯性 | 二阶结论是否依赖阶次同型？ | D1 审计后 |
| D3 | action-independent colored disturbance | 响应模块能否在不可预测背景扰动下保持结构与相对误差？ | D2 审计后 |

D1 不改变 R50 开度、context 调度、动作 profile、样本量、噪声、horizon 或训练预算。这样 primary contrast 只归因于 delay representation。

## 3. D1 真值与候选

真值先把有效剂量延迟 (d^*=2) 个 10 s step，再进入已冻结的 scheduled two-pole cascade：

\[
u_t^d=\begin{cases}0,&t<d^*,\\u_{t-d^*},&t\ge d^*,\end{cases}
\qquad
T_t=G_{2P,K(c),\tau(c)}(u^d) .
\]

学习型延迟使用长度为 5 的 causal simplex：

\[
\widetilde u_t=\sum_{d=0}^{4}w_d u_{t-d},
\qquad w_d\ge0,\quad\sum_dw_d=1 .
\]

| Candidate | 角色 |
|---|---|
| `d1_g2_no_delay` | 同一联合灰箱的 delay 消融 |
| `d1_g2_learned_delay` | 主模型；learned causal delay simplex |
| `d1_g2_oracle_delay` | R50 + fixed 2-step delay 正对照，不参加冠军 |
| `d1_k4_monotone` | stable modal 表示参照 |
| `d1_pi_monotone` | PI-ODE/closure 表示参照 |
| `d1_deeponet` | flexible fixed-horizon 表示参照 |

正式 validation 为 `6 candidates × 3 seeds = 18 runs`。test 默认不存在，必须在 validation 产物、本地复算和 checkpoint 归档后另行授权。

## 4. 预注册 Gate

1. 18/18 必须通过 reference identity、future-action leakage、post-change sensitivity、有限性、长期方向和适用路线谱半径门禁。
2. oracle 三个 seed 的 clean NMAE 必须均低于 0.05；否则判生成—优化链失败，不解释其它模型。
3. learned-delay 相对 no-delay 的 paired clean NMAE 改善必须每个 seed 至少 20%。validation 只作 screening；若以后授权 test，使用 paired episode stratified bootstrap 的 95% CI 下界判定。
4. 参数可辨识性单列两个诊断：expected delay error `|E_w[d] - 2| ≤ 1 step`，以及真值 ±1 step 邻域概率质量 `Σ_{d=1}^3 w_d ≥ 0.80`。learned logits 初始化偏向 `d=0`（不是均匀分布，避免初始期望值恰好等于 2 step）。响应 Gate 通过但参数诊断失败时，只能声称 delay capacity 有效，不能声称真实迟延已恢复。
5. flexible routes 只作 secondary representation reference；不根据最低 validation 数值事后改写路线冠军。

## 5. D1 之后

D1 validation 审计只决定是否允许一次性 synthetic test，以及 D2 是否需要保留 learned-delay 模块。它不直接放行真实数据。D1–D3 完成后进入 MS5 完整 `free+response` 耦合，再进入 MS3 A/B validation 和重构后的 MS4 现场闭环响应验证。
