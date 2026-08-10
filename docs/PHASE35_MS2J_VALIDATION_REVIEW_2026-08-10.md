# Phase 3.5-MS2-J Joint Coupling Validation Review（2026-08-10）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-10
- Verification Status: VERIFIED（36/36 checkpoint 已归档并完成独立哈希/可加载/有限性复核）
- Version Label: phase35_ms2j_validation_review_v1
- Execution Commit: `e3c6144`（manifest git_sha，27/27 一致；frozen paths 与当前 HEAD 逐文件 diff 通过）
- Evidence Scope: `synthetic_joint_coupling_validation_not_field_causality`；不是 A/B 现场因果验证

## 1. 审计判决

**MS2-J validation：联合模块门禁 PASS，staged 稳定性门禁 FAIL（如实报告，不硬撑）。**

- **联合模块增量成立**：`j_g2_monotone_scheduled_joint`（0.0410）相对两个单模块消融改善 79–91%，3 seeds 全部远超 20% 预声明门槛，方向一致。MS2-V（单调开度）与 MS2-C（context 调度）两个轴合并到同一真值后没有发生**响应层面的塌缩**；参数层面仍存在已知的 `K/phi` 补偿，不声称分别恢复真实参数。
- **staged 三阶段训练未通过 non-inferiority**：staged/joint 比 1.154–1.226 > 1.10 上限，3 seeds 一致地劣于 joint-from-scratch。A/B/C 均完成且相对 Stage A 改善 76–79%，说明训练链可运行；但当前 staged 协议的整体门禁仍是 FAIL。现有 validation 不能证明这是所有 staged 方案的普遍“精度上限”。

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

1. **联合模块正结论**：单调开度映射与 context 调度在同一真值下的**联合响应**可被当前模型识别，且联合优于任一单模块。oracle 0.0208 与接近真值的 base gain 支持该合成 regime 的生成—优化链可解；learned `K` 与 `phi` 仍不可拆分识别。
2. **staged 负结论**：三阶段解冻训练（A: base+opening → B: schedules → C: 全解冻 0.2×lr）**不优于** joint-from-scratch，3 seeds 一致超 10% 容忍线。staged 的作用应表述为"训练稳定性的对照"，不推荐为论文主训练方案。
3. 全部结论仅属 synthetic known-truth；不授权现场 E3/E4、不构成真实阀门曲线或现场调度形式的证据。
4. `j_deeponet` 在非-oracle learned candidates 中数值最低（0.0341），仍按预注册为灵活算子对照，不事后改写物理冠军。

## 5. 本地复核与后续授权

1. `ms2j_checkpoints_validation.tar` 共 36 个文件，archive SHA256=`3005fd4b2c3b96bedaf53e273ad39f440670fec5b23eb60e7d64e540d3e06f52`；36/36 与 manifest 匹配、可加载且参数有限。
2. 本地 Torch 2.5/CPU 从权重重放 27 个 validation run：结构门禁一致；主指标相对差最大 0.894%，低于预注册 10% environment-sensitive 容忍线。trajectory digest 因 Torch 2.11/CUDA 与 2.5/CPU RNG 差异而 27/27 不逐位相同，按协议不伪装成 bitwise replay。
3. 已另行冻结一次性 MS2-J synthetic test；它确认混合结论，不改变 validation 的 overall FAIL，也不授权重训或现场 test。
