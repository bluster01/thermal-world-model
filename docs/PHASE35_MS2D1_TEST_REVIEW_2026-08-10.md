# Phase 3.5-MS2-D1 Synthetic Test Review（2026-08-10）

> **文档角色：Linux 远端执行报告（非独立审计）。** 数值结果已由本地 Supervisor 复核；权威判决见 [`PHASE35_MS2D1_TEST_SUPERVISOR_AUDIT_2026-08-10.md`](PHASE35_MS2D1_TEST_SUPERVISOR_AUDIT_2026-08-10.md)。本文保留为回传记录，不自行改变 Gate 状态。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: test（one-shot synthetic access）
- Origin Date: 2026-08-10
- Verification Status: UNVERIFIED_REMOTE_REPORT（后续已由 Supervisor 独立审计）
- Version Label: phase35_ms2d1_test_review_v1
- Execution Commit: `6665405`（授权）；test 由 Linux 在 pull 后执行
- Evidence Scope: `synthetic_delay_pressure_test_not_field_causality`；不读取 A/B 现场 test；不恢复已废弃 E1–E5

## 1. 判决

**MS2-D1 响应确认门禁 FAIL（CI 判定），但非数据问题——这正是 Codex 设计文档 §1 预见的场景，按冻结解释落地：D1 = "validation screening 阳性、independent test 未确认"。**

| 门禁 | 结果 | 关键数值 |
|---|---|---|
| 结构门禁 18/18 | ✅ | gate_failures 为空 |
| oracle test | ✅ | 0.0206–0.0223 < 0.05（3 seeds） |
| **delay response test（confirmatory）** | ❌ **FAIL** | observed 改善 20.4–22.5%，但 **95% CI 下界 17.2–18.8% < 20%**（3 seeds 全部） |
| delay parameter diagnostic | ❌（单列，不阻塞） | E[d]=2.03–2.20 steps 误差<0.2 ✅；±1 step 质量 0.54–0.58 < 0.80 ❌ |

## 2. 核心对比（paired episode stratified bootstrap，256 ep/seed × 10k reps）

### delay response test：learned_delay vs no_delay（CI 下界 ≥20% 预注册门槛）

| seed | observed 改善 | 95% CI | CI 下界 ≥20%? |
|---|---|---|---|
| 0 | 0.204 | [0.172, 0.240] | ❌ 0.172 |
| 1 | 0.217 | [0.184, 0.250] | ❌ 0.184 |
| 2 | 0.225 | [0.188, 0.266] | ❌ 0.188 |

Codex 设计文档 §1 中的本地 bootstrap 预算是 17.26%–19.58%——test 实测 17.2–18.8%，**预算完全命中**。validation screening（点估计 20.25–23.11%）通过，但独立 test 用 CI 判定时三 seed 下界均差 1.2–2.8 个百分点。

## 3. 候选榜（test clean NMAE，3 seeds）

| Candidate | Test | Validation | Δ |
|---|---:|---:|---|
| d1_g2_oracle_delay | **0.0214 ± 0.0009** | 0.0207 | +0.0007 |
| d1_deeponet | 0.0346 ± 0.0024 | 0.0350 | −0.0004 |
| d1_g2_learned_delay | 0.0425 ± 0.0014 | 0.0435 | −0.0010 |
| d1_g2_no_delay | 0.0542 ± 0.0020 | 0.0554 | −0.0012 |
| d1_pi_monotone | 0.0589 ± 0.0032 | 0.0581 | +0.0008 |
| d1_k4_monotone | 0.2571 ± 0.0161 | 0.2529 | +0.0042 |

无 split degradation（主对比 Δ<0.002），排序与 validation 一致。learned 0.0425 vs no_delay 0.0542 的方向稳定存在（点估计 −21.5%），只是 CI 下界不足以跨过 20% 预注册门槛。

## 4. 协议审计

| 项目 | 结果 |
|---|---|
| 单次访问 | root ledger `completed`，18/18 run ledgers completed；拒绝重复访问 |
| 内容寻址 | matrix / validation_summary / checkpoint_archive 三 pin **全部匹配**（本次无 MS2-J 的 pin 不同步问题） |
| 权重来源 | tar 加载（18 members），manifest hash 匹配 18/18 |
| 冻结代码等价 | FROZEN_EXECUTION_PATHS 与训练 commit 逐文件 diff 通过 |
| test 状态 | manifest test_accessed 从 false → true |
| bootstrap | paired_episode_stratified_by_action_profile，10k reps，seed 20260810/11/12，profile 52/51/51/51/51 |

## 5. 远端结果摘要与边界（提交 Supervisor 审计）

1. **oracle 正对照 test 复现**：0.0206–0.0223，优化链可解性确认（MS1 → MS2-V/C → MS2-J → MS2-D1 val → MS2-D1 test 连续成立）。
2. **delay 改善方向稳定但幅度不足**：learned vs no_delay 点估计改善 20.4–22.5%（与 validation 20.3–23.1% 一致），但 95% CI 下界 17.2–18.8% 未能跨过 20% 预注册门槛。**按冻结解释：D1 记 "validation screening 阳性、independent test 未确认"**，不重试、不调阈值、不加 seed。
3. **参数诊断继续为负**：E[d] 精确（2.03–2.20）但分布弥散——与 validation 一致，capacity 有效、唯一 20 s 迟延未恢复。
4. **D2 不得以正结论传播自动启动**：按设计 §3，D2 是否运行必须以压力诊断（而非 D1 阳性）重新设计，由 Codex 决定。
5. 全部仅属 synthetic known-truth；不授权现场 E3/E4、真实阀门曲线或现场因果主张。

## 6. 回传状态

远端已提交 18 组 metrics/episodes/ledgers 与 `summary_test.json`。权威 TODO/注册表更新和 D2 设计由本地 Supervisor 完成；本报告不承担状态迁移。
