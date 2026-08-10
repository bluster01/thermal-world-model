# Phase 3.5-MS2-D3 Colored-Disturbance Pressure Design

## Material Passport

- Material Type: preregistered synthetic experiment design
- Evidence Scope: `synthetic_colored_disturbance_pressure_validation_not_field_causality`
- Upstream: MS2-D2 `CLOSED / CONFIRMED_SYNTHETIC_ORDER_RESPONSE`
- Status: IMPLEMENTED / LOCALLY VERIFIED / READY FOR LINUX VALIDATION
- Boundary: 只检验 action-independent temporal nuisance；不引入 free head、不访问 A/B、不启动 MS5

## 1. 单一研究问题

在 D2 已确认的 R50、context scheduling、三极点 `[40,70,210] s`、无 pure delay 真值上加入不可由 response operator 观察的 action-independent colored disturbance 后，显式三极点相对同预算二极点的 clean action-response 优势是否仍能逐 seed保持？

D3 不要求 response operator 预测不可观测扰动。主指标仍是已知真值 `clean_effect`；带扰动的 `effect_mae` 只用于 validation checkpoint 选择和误差地板诊断。

## 2. 备选设计与取舍

| 方案 | 回答的问题 | 风险 | 决定 |
|---|---|---|---|
| A. 输出端平稳 AR(1) nuisance | 条件均值响应能否从时间相关背景中恢复 | 不模拟扰动与物理状态耦合 | **采用；与 MS5 边界最清楚** |
| B. 扰动注入某个热力状态 | 含过程传播的扰动下是否稳健 | 改变 latent estimand，混入未观测状态闭合问题 | 推迟到 MS5/后续 simulator Gate |
| C. 多频谱×多幅度网格 | 扰动谱鲁棒曲线 | 矩阵膨胀且产生事后挑强度空间 | D3 不采用；必要时另立后续 robustness scan |

采用 A 是因为 D3 的冻结问题是“action-independent colored disturbance”，不是完整过程扰动世界。将扰动注入热力状态会同时测试 free-state estimation、disturbance observer 和 response identification，无法归因失败来源。

## 3. 冻结生成式

动作 clean response 与 D2 完全相同。额外扰动为每个 episode 独立的平稳 AR(1)：

\[
\rho=\exp(-\Delta t/\tau_d),\qquad
d_{-1}\sim\mathcal N(0,\sigma_d^2),
\]

\[
d_t=\rho d_{t-1}+\sigma_d\sqrt{1-\rho^2}\,\epsilon_t,
\quad \epsilon_t\sim\mathcal N(0,1),
\]

\[
y_t^{target}=g_{clean}(c,a,r)_t+e_t+d_t.
\]

冻结参数：`dt=10 s`、`tau_d=120 s`、`sigma_d=0.03 °C`、原 white noise `sigma_e=0.02 °C`。因此 `rho≈0.9200`。扰动在 toggled spec 下不得改变 context、action、reference、profile 或 clean response；现有非 D3 regimes 的输出必须逐值保持。

该扰动是“未观测背景输出扰动”，不是主汽温现场过程噪声的标定模型。幅度和相关时间只构成中等强度 synthetic pressure，不代表现场频谱。

## 4. 冻结矩阵

基底 truth：D2 的 `third_order_r50_context_scheduled`，改名为 `third_order_r50_context_scheduled_colored_disturbance`。候选仍为 7×3 seeds=21：

1. `d3_g2_two_pole`：同预算二极点主消融；
2. `d3_g3_three_pole`：三极点主模型；
3. `d3_g3_oracle_structure`：三极点+R50 正对照；
4. `d3_g2_delay_compensation`：扰动/遗漏阶次被 delay capacity 吸收的诊断；
5. `d3_k4_monotone`；
6. `d3_pi_monotone`；
7. `d3_deeponet`。

operator、training budget、train/validation/test 样本数、生成器 base seed `20260813`、training seeds `[0,1,2]` 和 checkpoint selector 与 D2 相同。这样 toggled D2/D3 的 context、动作与 clean truth 可逐值配对；不得因为 colored nuisance 增 epoch、补 seed或按 clean truth 选 checkpoint。

## 5. Validation 主门与统计单位

runner 为每个 validation-selected checkpoint 保存 `episode_metrics_validation.json`。同 seed 七候选必须共享 trajectory design hash；配对 bootstrap 以 episode 为单位、在 hold/step/pulse/ramp/multi-step profile 内重采样，10,000 次，seed 为 `20260814 + training_seed`。

主门逐 seed要求：

1. 21/21 artifact、结构门和 disturbance contract 通过；
2. `d3_g3_oracle_structure` clean NMAE `<0.05`；
3. `d3_g3_three_pole` clean NMAE `<0.10`；
4. 三极点相对二极点的 paired/profile-stratified bootstrap 95% CI 下界 `>=0.10`。

validation 同时参与 checkpoint 选择，所以即使全过也只判 `AUDITED_SCREENING_PASS`，随后才可内容寻址授权一次性 test。任一主门失败则按原阈值关闭，不增加数据、不调扰动强度。

## 6. 非阻断诊断

- realized disturbance mean、std、lag-1 autocorrelation 与理论 `rho`；
- observed-effect MAE 与 clean-effect NMAE 的差异；
- D3 相对 D2 的 candidate/seed clean NMAE ratio；
- action profile 与 H1/H6/H18/H60 异质性；
- tau 集合与 no-true-delay 下 learned-delay；
- secondary representation 排名。

这些诊断不创建新的通过门槛。尤其不能因 DeepONet 排名、delay capacity 或某个 profile 结果事后改变主对比。

## 7. 解释与停止边界

D3 PASS 最多支持：“冻结的结构化响应在一个 action-independent AR(1) 输出扰动设计下仍可恢复。”它不支持现场扰动频谱、扰动 observer、完整状态闭合、自由递推仿真或闭环控制。D3 validation 审计前不访问 test；D3 关闭前不启动 MS5、MS3 或 MS4。
