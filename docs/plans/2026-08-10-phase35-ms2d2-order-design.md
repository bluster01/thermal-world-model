# Phase 3.5-MS2-D2 三阶惯性结构压力诊断设计

## Material Passport

- Material Type: sequential synthetic diagnostic design
- Evidence Scope: `synthetic_order_pressure_validation_not_field_causality`
- Upstream: MS2-D1 validation screening positive / independent test not confirmed at the 20% margin
- Status: READY_FOR_LINUX VALIDATION
- Field Boundary: 不读取 synthetic test 或 A/B 数据；不恢复 E1–E5；不把 D1 阳性传播到 D2

## 1. 决策与研究问题

D1 证明了显式迟延容量的改善方向稳定，却没有在独立 test 确认“改善至少 20%”，且迟延核没有唯一恢复。因此 D2 不再问“已成立的 delay-aware 模型能否继续获胜”，而问：

> 当真值只增加一个可分辨的惯性状态时，显式三阶结构是否相对同训练预算的二阶结构产生至少 10% 的稳定响应改善？二阶+learned-delay 是否会把漏掉的惯性误解释为伪迟延？

这个问题是结构压力诊断，不是路线冠军赛，也不以 D1 为正结论前提。

## 2. 备选路径

| 路径 | 处理 | 结论 |
|---|---|---|
| A. D1 后停止 MS2-D | 不再检验阶次与扰动 | 放弃；M 系列证据链不完整 |
| B. 将 learned-delay 固定为 D2 主模型 | 在三阶真值继续传播 D1 | 放弃；违反 D1 independent test 未确认边界 |
| C. 正交阶次诊断 | 取消真值纯迟延，只增加第三惯性；delay 仅作替代机制诊断 | **采用** |

## 3. 真值与单轴变化

D2 保持 D1/MS2-J 的 R50 有效开度、context 调度、动作 profiles、噪声、样本量、600 s horizon 和训练预算；真值 `input_delay_steps=0`。原二阶时间常数 `[70,210] s` 前增加一个可解析的 `40 s` pole：

\[
x^{(1)}_{t+1}=a_1(c)x^{(1)}_t+(1-a_1(c))\phi_{R50}(u_t),
\]

\[
x^{(j)}_{t+1}=a_j(c)x^{(j)}_t+(1-a_j(c))x^{(j-1)}_t,
\quad j=2,3,
\]

\[
\Delta T_t=K(c)x^{(3)}_t,
\qquad \tau^*=[40,70,210]\ \mathrm{s}.
\]

第三 pole 在 10 s 采样下有 4 个采样时间常数，既不是近似瞬时项，也不是新的纯迟延。二阶模型仍可用有效时间常数逼近它，因此这是非平凡结构失配而非人为制造的必胜任务。

## 4. 冻结候选

| Candidate | 角色 | 与真值关系 |
|---|---|---|
| `d2_g2_two_pole` | primary ablation | scheduled monotone 二阶；漏掉一个 pole |
| `d2_g3_three_pole` | primary model | scheduled monotone 三阶；只比 ablation 多一个 pole |
| `d2_g3_oracle_structure` | positive control | 三阶 + 已知 R50 opening family；参数仍由 validation 训练 |
| `d2_g2_delay_compensation` | alternative-mechanism diagnostic | 二阶 + learned delay；检查伪迟延是否吸收漏阶 |
| `d2_k4_monotone` | secondary representation | stable controlled Koopman reference |
| `d2_pi_monotone` | secondary representation | nominal two-pole PI-ODE + closure reference |
| `d2_deeponet` | secondary representation | causal fixed-horizon flexible reference |

正式 validation 为 `7 candidates × 3 seeds = 21 runs`。secondary 与 delay-compensation 都不参与主门禁，也不能因 validation 数值最低而升级为冠军。

## 5. 预注册门禁

主门禁全部逐 seed 判定：

1. 21/21 artifact 与 reference identity、future-action leakage、有限性、post-change sensitivity、长期方向及适用路线谱半径通过；
2. `d2_g3_oracle_structure` clean NMAE `<0.05`；
3. `d2_g3_three_pole` clean NMAE `<0.10`；
4. `d2_g3_three_pole` 相对 `d2_g2_two_pole` clean NMAE 点改善 `≥0.10`。

10% 是复杂度升级的最小工程收益：三阶只增加一个物理状态，低于该幅度不支持在主架构中保留额外阶次。validation 只作 screening；只有四项都通过并完成 checkpoint 归档/本地审计，才可另行设计一次性 test，以 paired episode、profile-stratified bootstrap 的 95% CI 下界 `≥0.10` 确认。

## 6. 单列可辨识性诊断

以下不进入主门禁：

- 三阶模型与 oracle 的 sorted-τ log-MAE；pole 的排列本身不可辨识，不按标签比较；
- `d2_g2_delay_compensation` 的 expected delay 与 `w_0`。真值没有纯迟延；若该模型仍分配非零迟延并改善响应，说明“额外惯性”和“分布式迟延”在有限 horizon 内可互相补偿；
- 各 action profile 与 H1/H6/H18/H60 的改善异质性。

因此即使三阶响应门通过，也只能声称 order-aware capacity 有用；只有参数诊断同时稳定，才讨论时间常数集合的近似恢复，不能声称现场存在三个唯一物理设备状态。

## 7. 停止规则

- Oracle 失败：生成—优化链失败，停止解释其它路线；不访问 test。
- 三阶主模型绝对误差或 10% 相对门失败：D2 记为结构复杂度未获支持；不调阈值、不补 seed。
- Delay compensation 接近或优于三阶：记录为机制不可辨识证据，不事后改为 delay 路线胜出。
- D2 validation 审计前不启动 D3、MS5 或任何 test。
