# Phase 3.5-MS2 Synthetic Test Review（2026-08-10）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: test（one-shot synthetic access）
- Origin Date: 2026-08-10
- Verification Status: VERIFIED
- Version Label: phase35_ms2_test_review_v1
- Training Commit: `f3401631edae60b42f8832024de7098305e4d0d7`（manifest git_sha，33/33 一致）
- Evaluation Commit: `f4e0612`（Codex 授权 + test evaluator 部署；frozen execution paths 与训练 commit 逐文件 diff 一致）
- Evidence Scope: synthetic_mismatch_test_not_field_causality；不是 A/B 现场因果验证

## 1. 判决

**MS2 synthetic test PASS。** 33/33 runs 单次访问完成，artifact/结构门禁全过；两个预注册主对比的 paired stratified bootstrap（256 episodes/seed × 3 seeds × 10k replicates）相对改善约 88–90%，**95% CI 下界全部 ≥0.859，远超 20% 最小有意义门槛**。validation 与 test 的 clean NMAE 逐候选一致（无 split degradation），oracle 正对照 0.0043 与 validation 完全复现，确认优化链与数据生成可解性成立。

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

最大 Δ=0.0198（identity，误差大故波动大），主模块 Δ<0.003。无 split degradation 证据。

## 4. 协议审计

| 项目 | 结果 |
|---|---|
| 单次访问 | 33/33 ledger `completed`；evaluator 拒绝重复/部分访问（metrics/episode/ledger 任一存在即拒） |
| 冻结代码等价 | evaluator 对 6 个 FROZEN_EXECUTION_PATHS 逐文件 diff 训练 commit `f340163` vs 当前 HEAD，全部一致 |
| checkpoint 完整性 | 33/33 manifest.checkpoint_sha256 == 磁盘实际 SHA；checkpoint 内 protocol/route/seed/git_sha 与 manifest 一致 |
| 同轨迹配对 | 每 regime×seed 的 33 组 `trajectory_design_sha256` 唯一（同一批 test trajectories），episode_ids/profile_ids 全等 |
| episode 完整性 | 33/33：256 条 episode、profile {0..4} 全覆盖、H1/H6/H18/H60 齐、无 NaN/负数 |
| bootstrap 单元 | `paired_episode_stratified_by_action_profile`，10k replicates，seed 20260810+regime+seed |
| 结构门禁 | 33/33 全过（reference identity=0、leakage=0、有限 rollout、方向约束、谱半径<1） |

## 5. 结论与边界（写入论文的表述）

1. **阀门绝对开度非线性映射在合成失配真值下是必要结构**：线性假设 clean NMAE 0.335 vs 单调模块 0.037，改善 CI 下界 0.859。只证明"该结构可辨识且显著优于线性"，不证明现场阀门真实开度曲线形状（未知，需现场标定）。
2. **context 调度是工况依赖可辨识的必要通道**：不读 context 的全局参数模型 0.21–0.23 vs 读 context 路线 0.021–0.025。三个不同架构收敛同一误差区 → 瓶颈是"context 是否接入物理参数调度"，非模型族。
3. **oracle 0.0043 复现**：数据生成与优化链可解性双重确认，排除 benchmark/optimization failure。
4. 阳性只属 `synthetic_mismatch_test`；**不授权**现场 E3/E4、闭环部署、MS3 真实数据因果断言、Fan 方程验证。
5. deeponet 数值最低（0.0203）但按预注册是"灵活算子对照"非主模块；不据此改写冠军或主结论。

## 6. 下一 Gate

**MS2 收口。** 两个结构轴均形成稳定、可复算的模块结论（validation + test 双层证据）。下一步按 TODO 顺序：

- MS2-D（纯迟延、阶次扩展、未建模扰动）：由用户决定是否铺开（当前 HOLD）；
- MS3（真实 A/B 数据适配）：可启动 validation-only 观测预测设计，但**不称因果**；需先出书面设计稿（PASS 条件：A/B 分榜、预测指标非劣、与合成结论方向一致的模块选择）。
