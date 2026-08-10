# Phase 3.5-MS2-J Joint Coupling Validation Review（2026-08-10）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-10
- Verification Status: ANALYZED（checkpoint 归档待补，见 §5）
- Version Label: phase35_ms2j_validation_review_v1
- Execution Commit: `e3c6144`（manifest git_sha，27/27 一致；frozen paths 与当前 HEAD 逐文件 diff 通过）
- Evidence Scope: `synthetic_joint_coupling_validation_not_field_causality`；不是 A/B 现场因果验证

## 1. 审计判决

**MS2-J validation：联合模块门禁 PASS，staged 稳定性门禁 FAIL（如实报告，不硬撑）。**

- **联合模块增量成立**：`j_g2_monotone_scheduled_joint`（0.0410）相对两个单模块消融改善 79–91%，3 seeds 全部远超 20% 预声明门槛，方向一致。MS2-V（单调开度）与 MS2-C（context 调度）两个轴合并到同一真值后**不发生补偿或塌缩**，联合模型接近 oracle 正对照（0.0208）。
- **staged 三阶段训练未通过 non-inferiority**：staged/joint 比 1.154–1.226 > 1.10 上限，3 seeds 一致地劣于 joint-from-scratch。staged 不是失败（相对 Stage A 边界改善 76–79%，训练稳定、A/B/C 全部完成），只是**分阶段解冻的精度上限低于直接联合训练**。

## 2. 数值复算（validation，3 seeds）

| Candidate | clean NMAE | 相对 identity_global | 角色 |
|---|---:|---:|---|
| `j_g2_identity_global` | 0.3931 ± 0.0257 | — | 双缺失负对照 |
| `j_g2_identity_scheduled` | 0.3615 ± 0.0347 | −8.0% | 仅调度（无单调映射） |
| `j_g2_monotone_global` | 0.2130 ± 0.0108 | −45.8% | 仅单调映射（无调度） |
| `j_g2_monotone_scheduled_joint` | **0.0410 ± 0.0035** | **−89.6%** | 双模块主模型（joint） |
| `j_g2_monotone_scheduled_staged` | 0.0487 ± 0.0031 | −87.6% | 双模块（staged） |
| `j_g2_r50_scheduled`（正对照） | **0.0208 ± 0.0017** | −94.7% | 真值映射注入 |
| `j_k4_monotone` | 0.2303 ± 0.0095 | −41.4% | Koopman 次要对照 |
| `j_pi_monotone` | 0.0542 ± 0.0027 | −86.2% | PI-ODE 次要对照 |
| `j_deeponet` | 0.0341 ± 0.0046 | −91.3% | 灵活算子对照 |

### 联合模块门禁（joint vs 单模块，预声明 20%）

| seed | joint NMAE | vs monotone_global | vs identity_scheduled | 全过 20% |
|---|---:|---:|---:|---|
| 0 | 0.0370 | +83.0% | +90.6% | ✅ |
| 1 | 0.0434 | +80.3% | +86.6% | ✅ |
| 2 | 0.0426 | +78.8% | +88.4% | ✅ |

### Staged 稳定性门禁（non-inferiority ≤1.10 + Stage A 改善 ≥20%）

| seed | staged NMAE | joint NMAE | ratio | ≤1.10? | Stage A 改善 | ≥20%? |
|---|---:|---:|---:|---|---:|---|
| 0 | 0.0454 | 0.0370 | 1.226 | ❌ | +79.3% | ✅ |
| 1 | 0.0516 | 0.0434 | 1.189 | ❌ | +76.9% | ✅ |
| 2 | 0.0491 | 0.0426 | 1.154 | ❌ | +75.7% | ✅ |

## 3. 结构与协议审计

| 项目 | 结果 |
|---|---|
| reference identity / leakage / finite / 方向 | 27/27 全过（结构门禁失败列表为空） |
| 谱半径 | graybox/koopman 全部 <1（含于结构门禁） |
| staged A/B/C 完整性 | 3/3 runs：阶段序列正确、optimizer_updates ≥1、stage checkpoints 文件 + SHA 匹配 |
| 环境字段 | 27/27 manifest 含 python/torch/cuda/device/platform |
| 轨迹一致性 | 每 seed 内 9 候选 trajectory design SHA 唯一（同 validation 轨迹） |
| 单 execution SHA | 27/27 = `e3c6144`，frozen paths diff 通过 |
| test 访问 | 27/27 test_accessed=false，无 test 产物 |

## 4. 结论与边界

1. **联合模块正结论**：单调开度映射与 context 调度在同一真值下同时可辨识，且联合优于任一单模块——MS2-V/C 两轴结论在联合设定下互不干扰。oracle 0.0208 确认优化链与数据生成可解性在该 regime 成立。
2. **staged 负结论**：三阶段解冻训练（A: base+opening → B: schedules → C: 全解冻 0.2×lr）**不优于** joint-from-scratch，3 seeds 一致超 10% 容忍线。staged 的作用应表述为"训练稳定性的对照"，不推荐为论文主训练方案。
3. 全部结论仅属 synthetic known-truth；不授权现场 E3/E4、不构成真实阀门曲线或现场调度形式的证据。
4. `j_deeponet` 数值最低（0.0341）仍按预注册为灵活算子对照，不事后改写冠军。

## 5. 待办

1. **checkpoint 归档**：27 × `.pt`（含 staged 的 A/B/C 共 27+9=36 个文件）打包 tar + SHA 写入 summary（P1-3 同协议）。
2. 归档后 commit + push；汇总器 exit=2 是**预期行为**（staged 门禁 FAIL，fail-closed 生效），summary_validation.json 已正确落盘。
