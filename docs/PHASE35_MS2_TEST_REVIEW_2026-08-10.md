# Phase 3.5-MS2 Synthetic Test Review（2026-08-10）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate（one-shot synthetic access）
- Origin Date: 2026-08-10
- Verification Status: VERIFIED（environment-sensitive；见 §4）
- Version Label: phase35_ms2_test_review_v1
- Training Commit: `f3401631edae60b42f8832024de7098305e4d0d7`（manifest git_sha，33/33 一致）
- Evaluation Commit: `f4e0612`（Codex 授权 + test evaluator 部署；frozen execution paths 与训练 commit 逐文件 diff 一致）
- Result Commit: `6a7bd8b`（Linux test artifacts）
- Evidence Scope: synthetic_mismatch_test_not_field_causality；不是 A/B 现场因果验证

## 1. 判决

**MS2 synthetic test PASS。** 33/33 runs 单次访问完成，artifact/结构门禁全过；两个预注册主对比的 paired stratified bootstrap（256 episodes/seed × 3 seeds × 10k replicates）相对改善约 88–90%，**95% CI 下界全部 ≥0.859，远超 20% synthetic module-screening 门槛**。11 个候选的 test/validation clean NMAE 相对变化均在 ±6.8% 内；这只说明同一生成分布下未见明显 split degradation，不是 OOD 泛化。oracle test 为 0.004276、validation 为 0.004264（相差 0.30%），是近似复现而非逐位相同。

## 2. 主对比（paired episode bootstrap，按 action profile 分层）

### MS2-V `valve_nonlinear_r50`：v_g2_monotone vs v_g2_identity

| seed | obs rel. imp. | 95% CI | CI 下界 ≥ 20% |
|---|---:|---:|---|
| 0 | 0.8804 | [0.8594, 0.8991] | ✅ |
| 1 | 0.9029 | [0.8842, 0.9191] | ✅ |
| 2 | 0.8871 | [0.8669, 0.9045] | ✅ |

### MS2-C `context_scheduled_2p`：c_g2_scheduled vs c_g2_global

| seed | obs rel. imp. | 95% CI | CI 下界 ≥ 20% |
|---|---:|---:|---|
| 0 | 0.8966 | [0.8840, 0.9084] | ✅ |
| 1 | 0.9004 | [0.8906, 0.9092] | ✅ |
| 2 | 0.9041 | [0.8907, 0.9150] | ✅ |

两个 regime 的 6/6 seed 判定 `ci_lower_exceeds_20pct=True`，`primary_contrasts_pass=True`。CI 半宽 <0.02，判定无边界模糊。

## 3. 候选 test 榜（clean NMAE，3 seeds）

| Candidate | Test | Validation | Δ |
|---|---:|---:|---|
| v_g2_r50_oracle | **0.0043** ± 0.0011 | 0.0043 ± 0.0010 | 0.0 |
| v_deeponet | 0.0203 ± 0.0038 | 0.0195 ± 0.0014 | +0.0008 |
| c_g2_scheduled | 0.0209 ± 0.0007 | 0.0215 ± 0.0025 | −0.0006 |
| c_deeponet | 0.0211 ± 0.0020 | 0.0221 ± 0.0022 | −0.0010 |
| c_pi_ode | 0.0248 ± 0.0011 | 0.0245 ± 0.0014 | +0.0003 |
| v_g2_monotone | 0.0368 ± 0.0029 | 0.0353 ± 0.0025 | +0.0015 |
| v_pi_monotone | 0.0424 ± 0.0029 | 0.0397 ± 0.0017 | +0.0027 |
| v_k4_monotone | 0.0817 ± 0.0040 | 0.0820 ± 0.0022 | −0.0003 |
| c_g2_global | 0.2100 ± 0.0014 | 0.2059 ± 0.0135 | +0.0041 |
| c_k4_global | 0.2278 ± 0.0030 | 0.2295 ± 0.0110 | −0.0017 |
| v_g2_identity | 0.3354 ± 0.0099 | 0.3552 ± 0.0202 | −0.0198 |

最大绝对 Δ=0.0198（identity，误差大故波动大），最大相对变化为 `v_pi_monotone` 的 +6.8%，主模块绝对 Δ<0.003。同一生成分布下无明显 split degradation；不能外推到新的物理 regime。

## 4. 协议审计

| 项目 | 结果 |
|---|---|
| 单次访问 | 33/33 ledger `completed`；evaluator 拒绝重复/部分访问（metrics/episode/ledger 任一存在即拒） |
| 冻结代码等价 | evaluator 对 6 个 FROZEN_EXECUTION_PATHS 逐文件 diff 训练 commit `f340163` vs 当前 HEAD，全部一致 |
| checkpoint 完整性 | 33/33 manifest.checkpoint_sha256 == 磁盘实际 SHA；checkpoint 内 protocol/route/seed/git_sha 与 manifest 一致 |
| 同轨迹配对 | 6 个 regime×seed 组内分别只有一个 `trajectory_design_sha256`；同组 6 个或 5 个候选的 episode_ids/profile_ids 全等 |
| episode 完整性 | 33/33：256 条 episode、profile {0..4} 全覆盖、H1/H6/H18/H60 齐、无 NaN/负数 |
| bootstrap 单元 | `paired_episode_stratified_by_action_profile`，10k replicates，seed 20260810+regime+seed |
| 结构门禁 | 33/33 全过（reference identity=0、leakage=0、有限 rollout、方向约束、谱半径<1） |
| 本地独立复算 | archive SHA=`1124e356…`，33/33 checkpoint/metadata 匹配；固定 bootstrap summary 与 Linux 文件 byte-exact |
| 环境敏感性 | Linux 为 Torch 2.11.0+cu130，本地为 Torch 2.5.0+cpu；本地权重推理最大主指标相对差 1.47%（低于 10% 环境敏感容差），但 33/33 trajectory digest 不同，说明数据 RNG 未跨 Torch 版本逐位冻结 |

因此保留 `VERIFIED`，但限定为 **environment-sensitive reproducible**。Linux 内部配对和原始 bootstrap 判决有效；不能声称跨环境生成了逐位相同的 test trajectory。ledger 未记录 evaluation Torch/Python/CUDA 版本，这是本轮 P1 provenance 缺口，后续 synthetic runner 必须补齐。

## 5. 结论与边界（写入论文的表述）

1. **显式单调模块优于同一二阶灰箱中的 identity 假设**：合成 R50 真值下，identity clean NMAE 0.335 vs monotone 0.037，改善 CI 下界 0.859。但 `K/phi/动力学` 存在补偿，learned `phi` 未恢复真值曲线；而使用 raw action 的 DeepONet 也能隐式表达非线性且误差更低。因此不能写“该阀门映射必要且已辨识”，只能写“线性 identity 二阶灰箱不足，显式单调模块是补充非线性响应容量的一种可解释实现”。
2. **context 信息对该合成变参数真值显著有用**：不读 context 的全局模型 NMAE 0.21–0.23，读取 context 的 scheduled graybox/PI-ODE/DeepONet 为 0.021–0.025。scheduled graybox 在 validation 的 K/τ 恢复较强，但 test 主对比只证明 response 受益，不能把所有灵活路线都解释成物理参数辨识。
3. **oracle 约 0.0043**：确认同型正对照和当前优化链没有明显 benchmark failure；不排除非 oracle 路线仍受预算、参数化或可辨识性限制。
4. 阳性只属 `synthetic_mismatch_test`；**不授权**现场 E3/E4、闭环部署、MS3 真实数据因果断言、Fan 方程验证。
5. deeponet 数值最低（0.0203）但按预注册是"灵活算子对照"非主模块；不据此改写冠军或主结论。

## 6. 统计谬误扫描

- **Coverage: 11/11 checked。** Simpson：非 hold 的 step/pulse/ramp/multi-step 在 6 个 seed 中均保持正向改善（最低约 79%），未见方向反转；hold 的真值/误差均为零，不解释相对改善。
- Ecological、base-rate、regression-to-mean、survivorship、reverse-causality：对独立生成的 synthetic episode 主对比不适用。Berkson/collider：没有按结果筛 episode 或控制后验变量。
- Look-elsewhere/garden-of-forking-paths：两个主对比、seed、20% 门槛与 bootstrap 算法均在 test 前冻结；其余 11-candidate 排名只作 secondary，不改写冠军。
- Correlation≠causation：**CAUTION**。synthetic intervention 只支持生成器内机制恢复；对现场阀位、喷水流量、主汽温和闭环因果均不得外推。

## 7. 下一 Gate

**MS2-V/C 收口。** 下一步不同时铺开 MS2-D 和 MS3。按既有 TODO“MS5 在 MS3 前”，先做一个窄的联合耦合 Gate：在同一 synthetic truth 中同时启用 R50 非线性与 context 调度，对比双模块 joint-from-scratch、分阶段训练和单模块消融。它只回答多模块能否共同收敛；纯迟延/未建模扰动及真实 A/B 适配继续 HOLD。
