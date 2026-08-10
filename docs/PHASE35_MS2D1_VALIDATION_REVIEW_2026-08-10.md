# Phase 3.5-MS2-D1 Pure-Delay Pressure Validation Review（2026-08-10）

> Supervisor 复核已完成；本文件的环境字段与归档待办有事实性过时。最终审计与一次性 test 判决以 [MS2-D1 Supervisor Audit](PHASE35_MS2D1_SUPERVISOR_AUDIT_2026-08-10.md) 为准。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate（sequential pressure Gate D1）
- Origin Date: 2026-08-10
- Verification Status: ANALYZED（checkpoint 归档见 §6）
- Version Label: phase35_ms2d_d1_validation_review_v1
- Execution Commit: `95d1dbe`（manifest git_sha，18/18 一致）
- Evidence Scope: `synthetic_delay_pressure_validation`；不读取 A/B 现场 test；不恢复已废弃 E1–E5

## 1. 判决

**MS2-D1（固定 20 s 纯迟延压力）validation 全部主要门禁 PASS。**

1. **oracle 正对照通过**：`d1_g2_oracle_delay` 3 seeds clean NMAE 0.0201–0.0211，全部 < 0.05 门槛 → 生成—优化链在纯迟延 regime 健康，其他模型结论可解释。
2. **显式 causal delay 模块有效**：`d1_g2_learned_delay`（0.0435）相对 `d1_g2_no_delay`（0.0554）改善 20.3%–23.1%，3 seeds 全 ≥20% 预注册门槛（seed=1 擦线 20.25%，但 3 seeds 方向一致）。
3. **延迟可辨识性部分成立（预注册预期分支）**：expected delay 2.03–2.20 steps（真值 2 steps = 20 s），误差 ≤0.20 step，3/3 within_one_step ✅；但 ±1 step 邻域质量仅 0.54–0.58 < 0.80 门槛，权重分布弥散（w≈[0.25, 0.05, 0.27, 0.23, 0.19]）。按设计 §4.4：**只能声称 delay capacity 有效，不能声称真实 20 s 迟延已恢复**。

## 2. 候选榜（clean NMAE，3 seeds）

| Candidate | Mean ± Std | 角色 | 相对 no_delay |
|---|---:|---|---:|
| `d1_g2_oracle_delay` | **0.0207 ± 0.0005** | 正对照（R50 + fixed 2-step delay） | −62.7% |
| `d1_deeponet` | 0.0350 ± 0.0037 | 灵活 fixed-horizon 参照 | −36.9% |
| `d1_g2_learned_delay` | 0.0435 ± 0.0044 | **主模型（learned causal simplex）** | −21.5% |
| `d1_pi_monotone` | 0.0581 ± 0.0021 | PI-ODE/closure 参照 | +4.8% |
| `d1_g2_no_delay` | 0.0554 ± 0.0048 | delay 消融 | — |
| `d1_k4_monotone` | 0.2529 ± 0.0263 | stable modal 参照 | +356% |

## 3. 门禁明细

### oracle gate（<0.05/seed）

| seed | clean NMAE | pass |
|---|---:|---|
| 0 | 0.0201 | ✅ |
| 1 | 0.0211 | ✅ |
| 2 | 0.0208 | ✅ |

### delay response gate（learned vs no_delay ≥20%/seed）

| seed | no_delay | learned | 改善 | pass |
|---|---:|---:|---:|---|
| 0 | 0.0533 | 0.0410 | +23.1% | ✅ |
| 1 | 0.0609 | 0.0486 | +20.3% | ✅ |
| 2 | 0.0520 | 0.0410 | +21.1% | ✅ |

### delay identification diagnostic（单列，不阻塞主门禁）

| seed | E[d] steps | error | ±1 step 质量 | 集中? |
|---|---:|---:|---:|---|
| 0 | 2.195 | 0.195 | 0.579 | ❌ |
| 1 | 2.030 | 0.030 | 0.538 | ❌ |
| 2 | 2.034 | 0.034 | 0.561 | ❌ |

权重分布：`[≈0.25, ≈0.05, ≈0.27, ≈0.23, ≈0.19]`——simplex 将质量分散在 d∈{0,2,3,4}，仅轻微偏向 d=2。学习型延迟表达的是"延迟容量 + 分布式补偿"，不是唯一的 2-step 尖峰。这与 MS2-V 的 `K/phi` 补偿结论同族：**参数不可唯一辨识时，只承诺容量与响应改善，不承诺物理参数恢复**。

## 4. 协议审计

| 项目 | 结果 |
|---|---|
| 18/18 结构门禁 | reference identity / future-action leakage / post-change sensitivity / 有限性 / 长期方向 / 谱半径 全过（failures 列表为空） |
| 环境字段 | 18/18 manifest 只有 `device`/`torch_version`；缺少 python/cuda runtime/platform，见 Supervisor Audit |
| 单 execution SHA | 18/18 = `95d1dbe` |
| test 访问 | 18/18 test_accessed=false；`test_authorized=false`（dry-run 与 manifest 双确认） |
| 训练预算 | 300 epochs 上限，无 test 泄漏路径 |

## 5. 结论与边界

1. **正结论（仅限合成真值）**：在固定 20 s 纯迟延 + R50 非线性 + context 调度的已知真值下，显式 causal delay simplex 比无迟延灰箱改善 20–23%（3 seeds 一致）——MS2 系列的联合灰箱结论**不依赖零迟延生成器**，D1 压力通过。
2. **边界声明**：learned delay 分布不唯一（±1 step 质量 <0.80），论文只能写"delay-aware 表示带来响应改善"，不能写"20 s 迟延被精确恢复"。`K/phi` 补偿结论在延迟轴上同样成立。
3. flexible routes（deeponet 0.0350 数值最低）仍按预注册为 secondary reference，不事后改写冠军。
4. 全部仅属 synthetic known-truth；不授权现场 E3/E4、真实阀门曲线或现场因果主张。

## 6. 待办

1. **checkpoint 归档已完成**：18 × `checkpoint_best_val.pt`，SHA256 为 `7ee6393993d209cee255e8d7f09b7d376a135b63fc70572b0c6993d44e1a05f4`。
2. Supervisor 已完成独立复算并授权冻结的一次性 synthetic test。
3. D2/D3 不自动启动；等待 D1 test 审计。
