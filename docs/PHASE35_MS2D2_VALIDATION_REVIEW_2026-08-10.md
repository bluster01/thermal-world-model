# Phase 3.5-MS2-D2 Third-Order Pressure Validation Review（2026-08-10）

> **文档角色：Linux 远端执行报告（非独立审计）。** 本文件超出了当批允许的 `results/phase3_5/ms2d_order/**` 写入范围，现保留作 provenance 记录；权威判决见 [`PHASE35_MS2D2_SUPERVISOR_AUDIT_2026-08-10.md`](PHASE35_MS2D2_SUPERVISOR_AUDIT_2026-08-10.md)。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate（sequential pressure Gate D2）
- Origin Date: 2026-08-10
- Verification Status: UNVERIFIED_REMOTE_REPORT（后续已由本地 Supervisor 独立审计）
- Version Label: phase35_ms2d2_validation_review_v1
- Execution Commit: `fa6933c`（manifest git_sha，21/21 一致）
- Evidence Scope: `synthetic_order_pressure_validation_not_field_causality`；不读取 A/B 现场 test；不恢复已废弃 E1–E5

## 1. 判决

**MS2-D2（三阶惯性结构压力）validation 全部主门禁 PASS。**

1. **oracle 正对照通过**：`d2_g3_oracle_structure` 3 seeds clean NMAE 0.0214–0.0226，全部 < 0.05 → 三阶真值下生成—优化链健康。
2. **三阶主模型绝对门槛通过**：`d2_g3_three_pole` 0.0403–0.0520，全部 < 0.10。
3. **阶次改善门通过**：three_pole 相对 two_pole 点改善 **18.6%–28.1%**，3 seeds 全部 ≥10% → 额外一个惯性 pole 带来超过最小工程收益的结构响应改善。

## 2. 候选榜（clean NMAE，3 seeds）

| Candidate | Mean ± Std | 角色 | 相对 two_pole |
|---|---:|---|---:|
| `d2_g3_oracle_structure` | **0.0222 ± 0.0007** | 正对照（三阶+R50 family） | −61.8% |
| `d2_deeponet` | 0.0419 ± 0.0080 | flexible 参照 | −28.1% |
| `d2_g3_three_pole` | 0.0444 ± 0.0065 | **主模型（三阶）** | −23.9% |
| `d2_g2_delay_compensation` | 0.0456 ± 0.0068 | 伪迟延诊断 | −21.8% |
| `d2_g2_two_pole` | 0.0583 ± 0.0051 | primary ablation（二阶） | — |
| `d2_pi_monotone` | 0.0598 ± 0.0057 | PI-ODE 参照 | +2.6% |
| `d2_k4_monotone` | 0.2573 ± 0.0067 | Koopman 参照 | +341% |

## 3. 门禁明细

### oracle gate（<0.05/seed）
0.0214 / 0.0226 / 0.0226 — 全过 ✅

### order-aware absolute gate（three_pole <0.10/seed）
0.0403 / 0.0520 / 0.0410 — 全过 ✅

### order-aware response gate（three_pole vs two_pole ≥10%/seed）

| seed | two_pole | three_pole | 改善 | pass |
|---|---:|---:|---:|---|
| 0 | 0.0539 | 0.0403 | +25.2% | ✅ |
| 1 | 0.0638 | 0.0520 | +18.6% | ✅ |
| 2 | 0.0571 | 0.0410 | +28.1% | ✅ |

## 4. 单列诊断（不进入主门禁）

### τ 恢复（permutation-invariant log-MAE <0.35 门槛）

| seed | 恢复 τ（sorted, s） | 真值 | log-MAE | pass |
|---|---|---:|---:|---|
| 0 | [35.8, 87.0, 196.0] | [40, 70, 210] | 0.133 | ✅ |
| 1 | [34.9, 90.3, 198.5] | [40, 70, 210] | 0.149 | ✅ |
| 2 | [33.7, 87.5, 217.3] | [40, 70, 210] | 0.143 | ✅ |

三阶模型正确恢复 3-pole 结构（40s 极点 33.7–35.8 略偏低；中间 pole 70→87–90 偏慢 ~25%；210s 极点 196–217 命中）。这是已知补偿方向（τ 与增益互相折衷），不改变响应门结论。

### 无真值迟延诊断（delay compensation 伪迟延吸收）

| seed | E[d] steps | w0 质量 | 门槛 (w0≥0.80) | 判定 |
|---|---:|---:|---:|---|
| 0 | 2.16 | 0.297 | ❌ | 伪迟延吸收 |
| 1 | 2.28 | 0.278 | ❌ | 伪迟延吸收 |
| 2 | 2.40 | 0.241 | ❌ | 伪迟延吸收 |

**确认设计 §1 的核心假设**：真值没有纯迟延，但 `d2_g2_delay_compensation`（二阶+learned-delay）仍分配 E[d]≈2.2 steps 的伪迟延（w0 仅 ~27%），且其 clean NMAE（0.0456）**接近** three_pole（0.0444，差距仅 2.7%）。按冻结停止规则 §7：**记录为机制不可辨识证据**，不事后把 delay 路线升级为胜出。

## 5. 远端结果摘要与边界（提交 Supervisor 审计）

1. **正结论（仅限合成真值）**：在 R50 + context 调度 + 三阶惯性 [40,70,210]s 的 known-truth 下，显式三阶结构相对同预算二阶结构改善 18.6%–28.1%（3 seeds），且绝对误差 <0.10 —— 阶次结构失配被正确回应，MS2-D 系列在"更强的结构失配"轴上通过。
2. **机制边界**：learned-delay 会吸收漏掉惯性（伪迟延 E[d]≈2.2 steps）且误差与真三阶几乎持平 → **阶次与迟延在有限 horizon 内部分可互换**，单凭响应误差不能区分机制；这限制了对现场"输运迟延 vs 高阶惯性"的解释权。
3. τ 恢复是近似（log-MAE 0.13–0.15），只能声称 order-aware capacity 与近似 τ 集合，不能声称现场存在三个唯一物理设备状态。
4. deeponet 数值最低仍按预注册为 secondary reference，不升级冠军。
5. 全部仅属 synthetic known-truth；不授权现场 E3/E4、真实阀门曲线或现场因果主张。

## 6. 回传状态

1. **checkpoint 归档**：21 × `checkpoint_best_val.pt` 打包 tar + SHA（`e8d6d806...`），21/21 manifest 匹配 ✅ 已完成。
2. 已提交并回传完整 `remote_execution` 日志。
3. Gate 状态、test 授权和后续路线只由本地 Supervisor 更新；本报告不承担状态迁移。
