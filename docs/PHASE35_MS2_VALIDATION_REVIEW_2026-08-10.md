# Phase 3.5-MS2 Validation Review（2026-08-10）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-10
- Verification Status: VERIFIED（本地已逐 run 校验 archive/manifest SHA，并在 CPU 重新加载 33 个冻结权重复算 validation）
- Version Label: phase35_ms2_validation_review_v1
- Source Commit: `ff9412a89bfabca3c456725a8fe3b8f300769c6f`（validation 产物归档 commit）
- Training Commit: `f3401631edae60b42f8832024de7098305e4d0d7`（manifest git_sha，33/33 一致）
- Evidence Scope: synthetic mismatch method feasibility；不是 A/B 现场因果验证

## 1. 审计判决

**MS2 validation PASS——两个结构轴的主对比均稳定超过预声明门槛。** 33/33 runs 结构门禁全过、正对照进入低 clean-error 区域、两个主对比（阀门单调非线性、context 工况调度）相对其负对照的 paired-seed clean NMAE 改善约 90%，远超预声明 20% 最小有意义门槛，3 个优化 seed 方向完全一致。

该结果授权 MS2 进入 synthetic test 单次访问；不允许写成现场阀门非线性标定、现场工况依赖验证或闭环世界模型成立。

## 2. 数值复算

以下由仓库 33 个 `metrics_validation.json` 与 manifest 重新聚合；均值/标准差只描述 3 个优化 seed，不是 episode-level 统计置信区间。

### MS2-V（`valve_nonlinear_r50`，真值含 equal-percentage R50 绝对开度映射）

| Candidate | clean NMAE | 相对 identity | 判读 |
|---|---:|---:|---|
| `v_g2_identity`（负对照） | 0.3552 ± 0.0202 | — | 线性开度假设在非线性真值下误差最大 |
| `v_g2_monotone`（主模块） | 0.0353 ± 0.0025 | **−90.1%** | 单调模块恢复支持域内增量响应；不能解释为真值阀门曲线恢复 |
| `v_g2_r50_oracle`（正对照） | **0.0043 ± 0.0010** | −98.8% | 真值映射注入 → 优化链可解性确认 |
| `v_k4_monotone` | 0.0820 ± 0.0022 | −76.9% | Koopman 表示下单调映射恢复较差 |
| `v_pi_monotone` | 0.0397 ± 0.0017 | −88.8% | ODE 块 + 单调映射接近主模块 |
| `v_deeponet` | 0.0195 ± 0.0014 | −94.5% | 非 oracle 中数值最低的固定时域对照；不据此改写预注册主结论 |

paired-seed 相对改善（monotone vs identity）：[0.8957, 0.9060, 0.9005]，mean 0.9007，方向一致 ✅

### MS2-C（`context_scheduled_2p`，真值 K/τ 随 context 调度）

| Candidate | clean NMAE | 相对 global | 判读 |
|---|---:|---:|---|
| `c_g2_global`（负对照） | 0.2059 ± 0.0135 | — | 全局参数无法吸收工况调度 |
| `c_g2_scheduled`（主模块） | 0.0215 ± 0.0025 | **−89.6%** | 有界对数尺度调度恢复真值变化 |
| `c_k4_global` | 0.2295 ± 0.0110 | **比 global graybox 高 11.4%** | 不读 context 的模态算子同样失败 |
| `c_pi_ode` | 0.0245 ± 0.0014 | −88.1% | context 进闭合项亦可 |
| `c_deeponet` | 0.0221 ± 0.0022 | −89.3% | context 进 causal branch 亦可 |

paired-seed 相对改善（scheduled vs global）：[0.8910, 0.8918, 0.9046]，mean 0.8958，方向一致 ✅

关键结构信息：MS2-C 中两个「不读 context」的负对照（c_g2_global、c_k4_global）显著落后，三个「读 context」路线（scheduled/pi_ode/deeponet）收敛到同一低误差区（0.0215–0.0245）——该 regime 的辨识瓶颈是「context 是否进入物理调度」，而非模型族。

## 3. 结构与协议审计

| 项目 | 结果 | 边界 |
|---|---|---|
| reference identity | 33/33 为 0 | 代码恒等式，不证明现场无混杂 |
| future-action leakage | 33/33 为 0 | 预测接口前缀因果 |
| post-change sensitivity | 全部非零（max_c 0.127 量级） | 动作分支对干预敏感 |
| positive-step direction | 全部为负（−0.131 量级） | 有限探针，非全输入域形式证明 |
| finite rollout/state | 33/33 通过 | 只覆盖 60×10 s 合成时域 |
| Graybox/Koopman 谱半径 | 全部 <1（如 0.9624） | 递推块结构稳定性，非 Lyapunov 证明 |
| checkpoint 归档 | tar 含 33 .pt，文件大小 215,040 bytes，归档 SHA `1124e356…`；逐 run 与 manifest checkpoint_sha256 比对 33/33 匹配 | 权重级可复算性已满足（P1-3 关闭） |
| test access | **未访问**（test_accessed=false，33/33） | 单次访问配额完整保留 |
| split | validation only | MS2 使用独立 `stable_cascade_v2` 非线性/调度真值；当前只验证 validation，尚不能讨论 test split degradation |

## 4. P1 限制跟踪

1. **Clean truth 指标**（MS1 P1-1）：✅ 已落实。全部主指标为 `clean_effect_nmae/mae` + clean-direction（33/33 = 1.0），带噪 MAE 仅作训练选择与噪声底对照。
2. **预算上限**（MS1 P1-2）：部分缓解。cap 300 / patience 30；`v_deeponet`、`v_k4_monotone`、`v_g2_r50_oracle`（1/3 seed）仍碰 cap——灵活路线在失配 regime 需要更多预算，其优势/劣势混合了表示与优化预算效应，不据此定冠军。
3. **权重归档**（MS1 P1-3）：✅ 关闭。本地不仅核对独立 tar 和 33/33 单文件 hash，还在 CPU 重载全部权重并重新生成 validation。Graybox/Koopman/PI-ODE 指标几乎逐位一致；CUDA/CPU DeepONet 的最大 clean NMAE 绝对差为 `4.27×10^-4`，两个主对比复算为 90.07%/89.58%，判决不变。
4. **Episode-level 不确定性**（MS1 P1-4）：未做（validation 阶段不要求）。正式 test 必须在相同 trajectory 上做 paired episode bootstrap 95% CI；3 个优化 seed 单独展示，不用 seed 标准差替代样本不确定性。

## 5. 11 类统计谬误扫描

- Coverage: **11/11 checked**。
- Simpson / ecological / Berkson / collider：MS2 双 regime 为独立生成总体，未发现跨 regime 汇总问题；MS2-C 的 context 分层显示「读 context vs 不读」是主导因素，后续现场 context 分层必须复查 Simpson。
- Look-elsewhere：MS2-V 中 `v_deeponet` 数值最优（0.0195），但设计预声明 deeponet 为「灵活算子对照」非主模块；不得事后把对照改写成冠军。主结论只来自两个预声明主对比。
- Garden of forking paths：矩阵与门禁先冻结，本复核未改动任何运行配置；test 后不得回写本结论。
- Base-rate / survivorship / regression-to-mean：完整 synthetic cohort，不适用。
- Correlation≠causation：synthetic structural truth 内部可谈已知干预响应；不外推到现场观测数据。

## 6. 下一 Gate

**MS2 synthetic test 授权建议：批准，条件如下**

1. test 与 validation 同生成器、不同动作/噪声 seed；每 run 单次访问，ledger 记录 `test_accessed`。
2. 主对比在相同 trajectory 上做 paired episode bootstrap 95% CI（20% 门槛重新以 CI 下界判定）；3 个优化 seed 单独展示。
3. 对 10 个非 oracle 候选 + 1 个 oracle 正对照部署 test evaluator（11 candidates 全部，oracle 只作 benchmark 复核）。
4. checkpoint 仍来自训练 commit `f340163`；测试在本授权 commit 执行，并逐项证明 matrix、训练入口及模型/生成器/训练代码与 `f340163` 内容等价。ledger 同时记录 training/evaluation SHA，不能伪称两个 SHA 相同。
5. 预声明主结论边界不变：阳性只属 `synthetic_mismatch_validation`，不授权现场 E3/E4、不授权 MS3 真实数据适配（后者需另行设计 validation-only 观测预测）。

MS2-D（纯迟延、阶次扩展、未建模扰动）与 MS3（真实 A/B 适配）继续 HOLD，直到 MS2 test 收口。

## 7. 判读边界（写入论文时的表述约束）

- 「单调非线性响应模块相对 identity 改善」仅对 equal-percentage R50 型合成真值和当前支持域成立；由于 `K/phi` 补偿，本结果既不证明现场一定需要同一映射，也不构成现场阀门曲线标定。
- 「工况调度可辨识」仅对 K/τ 对数尺度随 context 平滑变化成立；现场是否有同类调度、调度尺度未知，需 MS3 观测预测检验。
- 90% 相对改善的数值本身是 synthetic module-screening 信号，不是现场温差改善的承诺。

## 8. 本地参数与分层复核补充

- MS2-V 的主模块在 step/pulse/ramp/multi-step 四类动作上的 clean MAE 相对 identity 均下降 85%–92%，在参考开度四分位内均下降 66%–98%，所以总体改善不是单一动作或单一开度区间驱动。
- 但 learned `phi` **没有恢复 R50 曲线本身**：三个 seed 在训练支持区间的 knot-value RMSE 约 13.3–14.0 effective percentage points，学到的 `K≈-0.0375` 也与真值 `-0.10` 明显不同。模型依靠 `K`、`phi` 和有限时域动力学补偿恢复局部增量响应。因此论文只能主张“monotone response module 有效”，不能主张“阀门特性曲线被辨识”。
- MS2-C 的参数调度恢复较强但不完美：三个 seed 的 `K(c)` 相关系数为 0.991–0.995、相对 MAE 2.7%–3.3%；`tau1(c)` 相关 0.940–0.976、相对 MAE 7.0%–11.5%；`tau2(c)` 相关 0.925–0.964、相对 MAE 6.9%–9.3%。三个 context 维度的各四分位内，scheduled 相对 global 的 response MAE 均下降约 85%–92%。
