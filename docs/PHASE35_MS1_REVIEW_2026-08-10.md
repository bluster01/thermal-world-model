# Phase 3.5-MS1 Supervisor Review（2026-08-10）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-10
- Verification Status: ANALYZED
- Version Label: phase35_ms1_review_v1
- Source Commit: `1636f0745eaaff80670a0e393cbad025d9ca9862`
- Training Commit: `fba7311ac47ee77a5a241ece2753e727f69b68d7`
- Evidence Scope: synthetic known-truth method feasibility；不是 A/B 现场因果验证

## 1. Supervisor 判决

**MS1 PASS，但只通过“可解性正对照”。** 结果证明当前代码中的多步 Graybox-2P、PI-ODE 和 Causal DeepONet 能在同分布、已知二阶串联惯性真值下逼近噪声下限；Graybox-2P 还能恢复接近真值的全局增益与时间常数。该结果允许 Phase 3.5-MS 进入结构失配实验，不允许写成路线冠军、宽负荷验证、真实阀门因果响应或闭环世界模型成立。

## 2. 数值复算

真值观测噪声为独立高斯噪声 `σ=0.02°C`，其理论绝对误差下限为

\[
\mathbb E|\epsilon|=\sigma\sqrt{2/\pi}=0.0159577\ ^\circ\mathrm C.
\]

下表由仓库 18 个 `metrics_validation.json`、`metrics_test.json` 和 manifest 重新聚合；均值与标准差只描述 3 个优化 seed，不是 episode-level 统计置信区间。

| Route | Validation MAE | Synthetic test MAE | Test RMSE | 判读 |
|---|---:|---:|---:|---|
| Graybox-1P | 0.018977 ± 0.000063 | 0.018926 ± 0.000161 | 0.023867 | 一阶结构失配，且训练仍碰到 epoch cap |
| Graybox-2P | 0.015952 ± 0.000082 | 0.015983 ± 0.000096 | 0.020030 | 达噪声下限；MS1 inverse-crime 正对照 |
| Koopman-K2 | 0.016644 ± 0.000066 | 0.016644 ± 0.000055 | 0.020905 | 略高于噪声下限；训练未完全平台化 |
| Koopman-K4 | 0.016586 ± 0.000066 | 0.016585 ± 0.000055 | 0.020830 | 略优于 K2，但不足以宣称路线优势 |
| PI-ODE | 0.015956 ± 0.000084 | 0.015989 ± 0.000094 | 0.020036 | 达噪声下限；闭合项使参数不具唯一物理解读 |
| Causal DeepONet | 0.015969 ± 0.000085 | 0.016021 ± 0.000101 | 0.020081 | 达噪声下限；仅固定 60-step horizon |

Graybox-2P 三个 seed 学到 `K=-0.04070~-0.04077°C/%`、`τ1=62.8~64.9 s`、`τ2=224.0~228.0 s`，对照真值 `K=-0.04°C/%`、`τ=(70,210)s`。这是“同型模型可恢复”的阳性证据，也正因为同型而构成 inverse crime。

## 3. 结构与协议审计

| 项目 | 结果 | 边界 |
|---|---|---|
| reference identity | 18/18 为 0 | 证明代码恒等式，不证明现场无混杂 |
| future-action leakage | 18/18 为 0 | 证明预测接口前缀因果 |
| finite rollout/state | 18/18 通过 | 只覆盖 60×10 s 合成时域 |
| Graybox/Koopman 谱半径 | 小于 1 | 是对应递推块的结构稳定性 |
| PI-ODE nominal spectral radius | 小于 1 | 只约束名义块，不是含神经闭合项的 Lyapunov 证明 |
| positive-step direction | 全部为负 | 是有限探针，不是全输入域的形式证明 |
| test access | 每 run 一次，ledger `completed` | 提交顺序支持先训练、后冻结、再统一 test |

Validation 与 test 来自同一生成器、不同动作和噪声 seed。两者 MAE 接近的准确表述是“未观察到同生成器 split degradation”，不能泛化成“无过拟合”或“对结构失配稳健”。

## 4. 必须保留的 P1 限制

1. **Clean truth 指标缺失。** 旧指标把带噪 `target_effect` 当方向真值，因此六条路线的方向率都约为 0.697，几乎完全不能区分模型。MS2 必须同时报告无噪声 `clean_effect` MAE/NMAE 和 clean-direction。
2. **复杂路线碰到预算上限。** Graybox-1P、Koopman-K2/K4、DeepONet 共 12/18 条 history 跑满 100 epochs，末 10 epochs 仍有下降；它们与早停路线的差异混合了表示和优化预算效应。MS2 统一提高到 300-epoch cap、patience 30，并继续报告 best epoch、optimizer updates 和 wall-clock。
3. **冻结权重未入仓。** 18 个 `checkpoint_best_val.pt` 被 `.gitignore` 排除，manifest 也没有 checkpoint hash；目前只能复算 JSON 汇总，不能从原权重独立重建预测。因此本报告是 `ANALYZED`，不是 `VERIFIED`。MS2 manifest 必须记录 SHA-256，Linux 必须通过独立归档回传 checkpoint。
4. **没有 episode-level 不确定性。** `3 seeds` 只反映优化波动。后续 synthetic test 必须在相同 trajectory 上做 paired episode bootstrap；seed 作为重复优化层单独报告，不能用 seed 标准差替代样本不确定性。

## 5. 11 类统计谬误扫描

- Coverage: **11/11 checked**。
- Simpson / ecological / Berkson / collider：MS1 为人工生成的单总体，未发现适用证据；后续 context 分层必须复查 Simpson。
- Base-rate neglect / regression-to-mean / survivorship：不适用于当前完整 synthetic cohort。
- Look-elsewhere：若从六路线中只挑最低 test MAE 会产生风险；因此 MS1 不设冠军。
- Garden of forking paths：矩阵先冻结，风险较低；但 test 后再调预算不得回写 MS1 结论。
- Correlation≠causation / reverse causality：synthetic structural truth 内部可以谈已知干预响应；不得外推到现场观测数据。

## 6. 下一 Gate

进入 MS2，但只打开两个互相独立的结构轴：

1. `MS2-V`：阀位—有效作用的单调非线性；
2. `MS2-C`：增益和时间常数随 context 调度。

纯迟延、未建模扰动、真实 A/B 适配继续 HOLD，直到上述两个轴至少一个形成稳定、可复算的模块结论。
