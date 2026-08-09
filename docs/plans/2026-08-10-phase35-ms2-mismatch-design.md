# Phase 3.5-MS2 结构失配实验设计

> 状态：frozen for Linux validation。MS2 只部署 validation；synthetic test CLI 尚未授权。

## 1. 研究问题

MS1 已证明低维多步响应模型能恢复同型二阶真值。MS2 不做“大路线冠军赛”，而是把现场最相关的两个不确定模块拆成独立问题：

- **RQ-V**：当阀门开度与有效喷水作用为单调非线性时，显式绝对开度映射是否必要？
- **RQ-C**：当减温响应增益和时间常数随工况变化时，context-scheduled 多步 A1phys 是否比全局参数模型更可解？

两条轴使用不同 synthetic regime、独立榜单和独立主对比，不做跨 regime 单榜。阳性结果仍只属于 synthetic method feasibility。

## 2. 真值与候选

### MS2-V：`valve_nonlinear_r50`

真值先将绝对阀位通过归一化 equal-percentage R50 映射，再计算动作路径与参考路径的有效开度差，最后进入固定二阶串联惯性。为避免 12%–48% 低开度区的有效剂量被 `σ=0.02°C` 噪声淹没，该 regime 冻结 `K=-0.10°C/%effective`；R50 与该增益都只是 known-truth benchmark，不解释成现场质量流量或现场增益标定。

| Candidate | 作用 |
|---|---|
| `v_g2_identity` | 线性开度负对照 |
| `v_g2_r50_oracle` | 知道真值映射的正对照，不参加方法冠军 |
| `v_g2_monotone` | 论文主模块：端点归一化的可学习单调映射 |
| `v_k4_monotone` | 单调映射＋稳定受控模态算子 |
| `v_pi_monotone` | 单调映射＋名义 ODE/神经闭合 |
| `v_deeponet` | 不显式指定 R50 的固定时域算子对照 |

主对比是 `v_g2_monotone - v_g2_identity`。`v_g2_r50_oracle` 只确认数据与优化链可解。

### MS2-C：`context_scheduled_2p`

真值保持 identity opening map，但令

\[
K(c)=K_0\exp\{0.35\tanh(c_0)\},\qquad
\tau_j(c)=\tau_{j,0}\exp\{0.30\tanh(c_{j+1})\}.
\]

所有时间常数保持正值，增益保持非正。context 在 MS2-C 首次成为真值物理调度变量；MS1 的 context 仍是干扰变量。

| Candidate | 作用 |
|---|---|
| `c_g2_global` | 全局二阶灰箱负对照 |
| `c_g2_scheduled` | 论文主模块：有界对数尺度工况调度 A1phys-MS |
| `c_k4_global` | 不读取 context 的稳定模态负对照 |
| `c_pi_ode` | context 进入小型闭合项 |
| `c_deeponet` | context 进入 causal branch |

主对比是 `c_g2_scheduled - c_g2_global`。其余路线用于判断“显式参数调度”和“隐式函数逼近”是否表现不同。

## 3. 冻结预算与指标

- 2 regimes，11 candidates，3 seeds，共 **33 validation runs**。
- 每 run：train 1024、validation 256、horizon 60、dt 10 s。
- 最大 300 epochs、patience 30；唯一 checkpoint selector 仍是带噪 validation effect MAE。
- MS2 test 当前不可访问；本地审计 33 个 validation artifacts 后另行冻结授权。

Synthetic known truth 允许评估时读取无噪声响应。主报告指标为 `clean_effect_nmae`、`clean_effect_mae`、H1/H6/H18/H60 clean MAE 和 clean-direction；带噪 MAE 保留，用于训练选择和与 MS1 噪声下限对照。

### 预声明门禁

1. 33/33 reference identity、future-action leakage、post-change action sensitivity、finite rollout 和正阶跃终值降温方向必须通过；否则对应 run 不进入 test。
2. Graybox/Koopman 谱半径必须 `<1`；PI-ODE 只把名义谱半径作为诊断。
3. 每个主模块相对其负对照的三组 paired-seed clean NMAE 必须方向一致。
4. 预声明的最小有意义改善为 clean NMAE 相对下降 **20%**；未达到则报告 module not supported/inconclusive，不用次要路线补写阳性结论。
5. 正对照 `v_g2_r50_oracle` 若不能进入低 clean-error 区域，则 MS2-V 判为 benchmark/optimization failure，禁止比较其它候选。

20% 是 synthetic module-screening 阈值，不是现场工程温差阈值。正式 test 需要相同 trajectory 的 paired episode bootstrap 95% CI；3 个优化 seed 单独展示。

## 4. 可复现性与产物

每 run 必须回传：

```text
manifest.json
history.json
metrics_validation.json
checkpoint_best_val.pt
```

manifest 记录 checkpoint SHA-256。由于 `.pt` 默认不入 Git，Linux 必须把 33 个 checkpoint 打包为独立归档并同时回传归档 SHA-256；只 push JSON 不算 `reproducibility_passed`。本地用 manifest hash 校验后，才能部署 synthetic test evaluator。

## 5. 暂停项

- 三阶/纯迟延与未建模扰动：MS2-V/C 收口后再决定是否需要；
- 真实 A/B 数据适配：等待 MS2 module 选择，且只能先做 validation-only 观测预测；
- 现场因果、闭环部署、Fan 方程：证据层级不变，仍不由 synthetic 实验授权。
