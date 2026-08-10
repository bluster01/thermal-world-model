# Phase 3.5-MS2-J Synthetic Test Review（2026-08-10）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: test（one-shot synthetic access）
- Origin Date: 2026-08-10
- Verification Status: VERIFIED
- Version Label: phase35_ms2j_test_review_v1
- Training Commit: `e3c6144`（manifest git_sha，27/27 一致）
- Evaluation Commit: `78904ba`（授权 pin 修复后；frozen paths diff 通过）
- Evidence Scope: `synthetic_joint_coupling_test_not_field_causality`；不是 A/B 现场因果验证

## 1. 判决

**MS2-J synthetic test 完全复现 validation 的混合结论。**

1. **联合模块门禁 PASS**：`j_g2_monotone_scheduled_joint` 相对两个单模块消融，3 seeds × 2 对比 = 6/6 的 paired stratified bootstrap 95% CI 下界全部 ≥0.73（观察改善 0.77–0.88），远超 20% 预声明门槛。单调开度 + context 调度联合可辨识在 test split 成立。
2. **staged 非劣门禁 FAIL（复现）**：staged/joint 误差比 1.14–1.20，CI 上界 1.09–1.32 > 1.10，3 seeds 一致。validation 的 staged FAIL 不是 split 伪影——**分阶段训练确实不如 joint-from-scratch**。
3. **staged vs Stage A 门禁 PASS**：相对 Stage A 改善 73–74%，CI 下界 0.68–0.77 >> 20%。staged 训练链有效（非无效），只是上限低于 joint。

## 2. 主对比（paired episode bootstrap，按 action profile 分层，256 ep/seed × 10k reps）

### 门禁 1：joint vs 单模块消融（阈值 20%，CI 下界判定）

| seed | vs monotone_global obs (CI) | vs identity_scheduled obs (CI) | 通过 |
|---|---|---|---|
| 0 | 0.772 [0.739, 0.802] | 0.867 [0.838, 0.891] | ✅ |
| 1 | 0.771 [0.732, 0.804] | 0.876 [0.853, 0.895] | ✅ |
| 2 | 0.768 [0.737, 0.796] | 0.863 [0.839, 0.884] | ✅ |

### 门禁 2：staged vs joint（非劣上界 1.10）

| seed | observed ratio | 95% CI | ≤1.10? |
|---|---|---|---|
| 0 | 1.142 | [1.087, 1.206] | ❌ |
| 1 | 1.200 | [1.089, 1.323] | ❌ |
| 2 | 1.164 | [1.088, 1.246] | ❌ |

### 门禁 3：staged vs Stage A（阈值 20%，CI 下界判定）

| seed | obs 改善 | 95% CI | ≥20%? |
|---|---|---|---|
| 0 | 0.742 | [0.709, 0.774] | ✅ |
| 1 | 0.727 | [0.684, 0.763] | ✅ |
| 2 | 0.732 | [0.694, 0.766] | ✅ |

## 3. 候选 test 榜（clean NMAE，3 seeds）

| Candidate | Test | Validation | Δ |
|---|---:|---:|---|
| j_g2_r50_scheduled (oracle) | **0.0225 ± 0.0007** | 0.0208 ± 0.0017 | +0.0017 |
| j_deeponet | 0.0350 ± 0.0084 | 0.0341 ± 0.0046 | +0.0009 |
| j_g2_monotone_scheduled_joint | 0.0452 ± 0.0021 | 0.0410 ± 0.0035 | +0.0042 |
| j_g2_monotone_scheduled_staged | 0.0528 ± 0.0024 | 0.0487 ± 0.0031 | +0.0041 |
| j_pi_monotone | 0.0530 ± 0.0037 | 0.0542 ± 0.0027 | −0.0012 |
| j_g2_monotone_global | 0.1967 ± 0.0076 | 0.2130 ± 0.0108 | −0.0163 |
| j_k4_monotone | 0.2162 ± 0.0066 | 0.2303 ± 0.0095 | −0.0141 |
| j_g2_identity_scheduled | 0.3440 ± 0.0087 | 0.3615 ± 0.0347 | −0.0175 |
| j_g2_identity_global | 0.3871 ± 0.0112 | 0.3931 ± 0.0257 | −0.0060 |

主模块 Δ<0.005，无 split degradation；排序与 validation 完全一致。

## 4. 协议审计

| 项目 | 结果 |
|---|---|
| 单次访问 | root ledger `completed`（27 runs 全部完成）；拒绝重复/部分访问（root/run ledger 任一存在即拒） |
| 内容寻址 | matrix / validation_summary / checkpoint_archive 三 pin 校验通过（validation_summary pin 已修正，见 §5） |
| 权重来源 | 直接从 tar 读取（不落盘可变副本），member SHA 与 manifest 逐 run 匹配 |
| Stage A 评估 | 3 staged runs 在同一访问中评估 Stage-A checkpoint，trajectory 与 final 配对一致 |
| 冻结代码等价 | FROZEN_EXECUTION_PATHS 与训练 commit `e3c6144` 逐文件 diff 通过 |
| 结构门禁 | 27/27 全过（含 9 个 Stage-A 评估） |
| bootstrap | paired_episode_stratified_by_action_profile，10k reps，seed 20260810+对比编号，profile 分布 52/51/51/51/51 |

## 5. 授权文件修正记录

Codex 授权文件（`5fa9769`）中 `validation_summary` 的 pin（`788b1bc2...`）与仓库实际文件（`212e53cc...`）不一致——git 对象库无此 blob，枚举所有序列化组合无法复现，判定为 Codex 本地计算 SHA 时文件状态与 commit 不同步（另一台机器环境差异）。已在本机修正 pin 为实际仓库哈希（commit `78904ba`），修正前后 frozen_validation_status 三项全一致（all_gates_pass=false, joint=true, staged=false），内容寻址语义不受影响。

## 6. 结论与边界（写入论文的表述）

1. **联合模块（正结论，双层证据）**：单调有效开度与 context 调度在同一合成真值下同时可辨识且互不干扰，joint-from-scratch 灰箱在 validation + test 双层稳定识别。oracle 0.0225 确认优化链可解性第四重确认。
2. **staged（负结论，双层证据）**：三阶段解冻训练在 10% 非劣界内无法匹配 joint-from-scratch，test 复现 validation——主训练方案采用 joint。staged 仅作稳定性对照。
3. 全部仅属 synthetic known-truth；不授权现场 E3/E4、真实阀门曲线、真实质量流量、现场 do(valve)、完整 free+response 世界模型或"完全物理响应"声明。

## 7. 下一 Gate

**MS2-J 收口。** MS2 系列（V/C/J）全部形成双层证据结论。MS2-D（压力测试）与 MS3（真实数据）维持 HOLD，由用户决定推进方向。
