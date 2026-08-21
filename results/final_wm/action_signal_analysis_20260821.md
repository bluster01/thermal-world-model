# 动作信号分析报告：leakage 门伪影 + T1 臂收敛建议（2026-08-21）

执行侧（Hermes，DGX）| 数据源：修复批 `d2bfa19` 之后 T1+R1 seed0 重跑（`--compile --tf32`，四臂并行）

---

## 1. 背景

修复批（②③④）后重跑 T1 四臂 × seed0，R1 单 seed（seed0）判决：**REJECTED**。
触发门：leakage（aware_relative_improvement = 23.9% > 5% 门槛）；另两门通过
（runtime_blind_ok ✓；direction frac_negative = 1.0 ✓）。

本报告用两个判别实验证明该 REJECTED 是探针伪影，并给出动作增益的物理量化与臂收敛建议。

## 2. leakage 门验证为探针伪影

### 2.1 打乱 null 实验

同一探针协议（512 训练窗 + 128 验证窗、20 epoch、同 seed），aware 分支的阀位特征
**跨样本随机置换**后再训：

| 条件 | aware_relative_improvement |
|---|---|
| 真阀位关联 | 23.88% |
| 阀位打乱（关联破坏） | 23.24% |
| Δ（真实信息） | **0.64% ≈ 0** |

打乱后改善率几乎不变 → 23.9% 的改善**不来自阀位-残差关联**，而是探针结构伪影。

### 2.2 机制：20 epoch 全批训练欠拟合

blind 探针（纯闭包特征）在不同训练轮数下的验证拟合度（val_mse_norm，1.0 = 零解释力）：

| 训练轮数 | val_mse_norm | 残差方差解释率 |
|---|---|---|
| 20（协议值） | 0.907 | 9% |
| 80 | 0.482 | 52% |
| 200 | 0.377 | 62% |

协议 20 epoch 时 blind 探针严重欠拟合；aware 多两个输入维度恰好改变了优化轨迹、
拟合得更好 → "23.9% 改善"主要是优化伪影。收敛后 blind 单独就能解释 62% 残差方差
（闭包特征确实与逐步误差强相关——这是快动态/混合滞后欠建模的迹象，与"leakage"概念无关）。

### 2.3 结论

- 当前 leakage 门（improvement > 5%）**不可靠**：无信息量的打乱阀位即可触发。
- R1 seed0 的 REJECTED 判定**无效**；有效信息是：残差与阀位的真实关联 ≈ 0。

## 3. 三门对照：v0.2 vs 修复批

| 门 | v0.2 审计（6305b50） | 修复批 seed0 |
|---|---|---|
| runtime_blind_ok | ✓ | ✓ |
| direction (frac_negative) | ✗（方向不稳定，DirectWM 症状） | ✓ 1.0，mean −0.228°C/+5% |
| leakage | ✓（3 seed 全 false） | ✗ 23.9% —— 已证伪影 |

**修复批解决了 v0.2 的方向不稳定问题**（阀位-温度方向 100% 一致）；新触发的门为伪影。
若 leakage 门按 2.1 打乱控制修正，seed0 三门全过。

## 4. 动作信号量级：三参照对比

| 量 | 数值（v2 二级阀口径） |
|---|---|
| 模型 step 响应（60 步 rollout，+5% 阶跃） | **−0.091°C / 2%** |
| 混合参考（auditpack mixing_reference） | −0.53 ~ −1.48°C / 2% |
| 真实工况事件（event_study v2-down，n=6） | h6 +0.37°C、h18 +0.79°C（关阀升温，符号正确；67–83% 正确率） |
| DirectWM 动作审计（历史，同口径） | −0.013°C / 2%，方向随 fold/seed 漂移 |

结论：**比 DirectWM 强 ~7 倍、方向稳定——可辨识；但仍比混合参考弱 6–16 倍——增益不足**。

**caveat（重要）**：mixing_reference 是**零延迟稳态混合估计**（Δh/cp 比值，假设喷水瞬间
混合），而模型 step 响应是含时滞环节（tau_mix 级 ~350s 量级）的 60 步瞬态。模型在 600s
窗口内未必到达稳态，直接用瞬态增益除以零延迟稳态参考会**夸大差距**。要公平对比需取
模型更长 rollout 的稳态值（如 200+ 步）或给参考加上同阶时滞后再比。该参考本身的量级
是否适用也需对侧复核（假设 delta_h=1674 kJ/kg、cp=2.2 kJ/kgK 均为固定先验）。

## 5. 建议

### 5.1 leakage 门修复（协议侧）

任选其一：
- **打乱控制**：以 aware(真)-aware(打乱) 的 Δ 作为 suspect 判据（同架构、同训练量），
  阈值可按 v0.2 三 seed 数据校准；或
- **训到收敛**：blind/aware 均训至验证收敛（如 80+ epoch 或早停）再比改善率。

### 5.2 T1 只保留 closure_cons 臂

- 修复批 seed0 四臂排序：closure_cons 1.407 < latent4 1.416 < closure_steam 1.421 < physics_only 1.511；
- v0.2 时代 closure_cons 亦多次胜出（历史训练结论一致）；
- R1 权重复用即来自 closure_cons。
**建议后续 T1 只训 closure_cons × 3 seeds**，省 75% 训练资源，其余三臂归档。

### 5.3 执行侧附记

- 4 路并行在此机实测 ≈ 0.9× 串行（GB10 多进程争用），单进程仍是最快路径；
  若采纳 5.2，3 seeds 串行 ~4-5h 可完成。
- tf32 平价门通过：student val NLL 差 0.0017%（<1% 门槛）。
- 本报告所有结论为 **seed0 单 seed 暂定**，正式判决需 3 seeds。

---

*产物：auditpack_A.json、r1_report.json、matrix_summary_sideA.json、ledger（seed0 final 块），
seed1/2 半成品已回退 v0.2 审计态。*
