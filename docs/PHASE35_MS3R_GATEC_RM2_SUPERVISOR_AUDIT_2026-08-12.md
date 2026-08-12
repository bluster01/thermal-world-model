# Phase 3.5 MS3-R Gate C RM2 Supervisor Audit

> 日期：2026-08-12。范围：真实现场闭环数据、train/validation、2 个 expanding folds、3 seeds。本文不是独立 test、开环 plant identification、喷水流量标定或闭环投运授权。

## 判决

```text
RM2_COMPLETE /
CONDITIONAL_ACTION_PATH_REPRODUCED /
OPERATOR_GAIN_NOT_IDENTIFIED
```

RM2 证明 measured-boundary 模型可以稳定学习 observed-policy 预测，并在局部 `Tin−Tout` 上形成跨 seed、fold、UTC 日复现的动作依赖槽；但它同时给出了更重要的反证：四种 operator 在预测质量近似时学出相差数倍的响应幅值，因此当前 closed-loop auxiliary loss 没有唯一识别物理增益，不能选择 Koopman、PI-ODE、DeepONet 或 A1phys 的“物理冠军”。

保留 `A2_a1_sched_base` 作为预注册工程参考，保留 `C1_common_only` 作为部分辨识基线。Linux 授权关闭，test 和 MS4 继续冻结。

## 完整性

- 结果提交 `c095279`，执行代码提交 `c179482a18d244b54567f35cd858a4c110d022d3`；
- 9 candidates × 3 seeds × 2 folds = 54/54 完成，均为单次 attempt；
- 54 个 checkpoint 从归档流式恢复并与各 run ledger 核对，216 个逐 run ledger 条目和 4 个 root ledger 条目全部逐字节匹配；
- F0/F1 各 27 个 manifest 的 stats/selector/final anchor SHA 均唯一，selector 与 final reporting 不相交；
- 54/54 structural selector checks 通过，无 failure，test 未访问，本地没有重新训练。

## 结果总表

数值为 6 个 fold-seed run 的均值。`logged advantage` 是同一 residual local prediction 下，真实未来阀位相对置乱未来阀位的 local MAE 优势；由于该量参与了动作辅助训练，它只能说明模型使用了动作，不能单独证明因果或物理增益。

| 候选 | shared score↓ | local MAE °C↓ | terminal MAE °C↓ | logged effect |·| °C | logged advantage °C |
|---|---:|---:|---:|---:|---:|
| paired-free | 0.246113 | 1.755750 | **1.108099** | 0 | 0 |
| A1 additive | 0.245561 | 1.713933 | 1.112521 | 0.250634 | 0.527977 |
| **A1 scheduled base** | 0.245333 | 1.687937 | 1.123411 | 0.319175 | 0.739465 |
| A1 scheduled large | 0.245453 | **1.675576** | 1.131575 | 0.318421 | 0.737031 |
| LPV-Koopman | 0.244777 | 1.705414 | 1.111525 | **1.088478** | **3.410408** |
| PI-Neural-ODE | 0.245377 | 1.697605 | 1.121236 | 0.434899 | 1.110204 |
| causal DeepONet | 0.245449 | 1.697088 | 1.121327 | 0.492753 | 1.297915 |
| common-only | 0.245286 | 1.689045 | 1.123557 | 0.310911 | 0.678113 |
| no downstream latent | **0.241379** | 1.694274 | 1.151530 | 0.315262 | 0.716960 |

## 关键对比

1. **显式 A1 vs paired-free。** Additive A1 的 local MAE 在 6/6 配对 run 均改善，均值改善 `0.04182°C`；shared score 只改善 `0.000552`，terminal MAE反而平均增加 `0.00442°C`。这支持局部动作槽的工程价值，但不是整体仿真精度已经改善。
2. **scheduled vs additive。** Scheduled-base 的 local MAE 平均再改善 `0.02599°C`，但只在 4/6 run 改善，terminal MAE平均增加 `0.01089°C`；不能仅凭 logged advantage 更大宣称 scheduled 物理上更真。
3. **capacity scan。** Scheduled large 相对 base 的 logged effect 只变化 `−0.000753°C`，local MAE 在 6/6 改善；未观察到 free capacity 增大导致 response 消失。它排除了一种简单的 branch stealing 解释，但不证明分解唯一。
4. **common/differential。** Common-only 相对 full-MIMO 的 shared score、local 和 terminal 几乎不变（`−0.000047/+0.001108/+0.000146`）。因此当前 RM2 没有证明 differential response 模态对预测是必要的；支持域外只能优先报告 common spray mode。
5. **downstream latent。** 去掉 downstream latent 后 terminal MAE 在 6/6 配对 run 变差，均值 `+0.02812°C`。虽然 composite score 因权重结构反而更低，但这不能覆盖一致的末温退化；latent downstream block 应保留。

## 为什么不能选择 operator 冠军

A1 scheduled、Koopman、PI-ODE、DeepONet 的 shared score 只分布在 `0.24478–0.24545`，local MAE 只分布在 `1.68794–1.70541°C`；但 logged response 幅值分别为 `0.319/1.088/0.435/0.493°C`。Koopman 相对 A1 的响应放大约 `3.41×`，却没有对应的 local 预测改善，反而平均差 `0.01748°C`。

用相同 validation episode 将 learned pure-action effect 对阀位 dose 做 H60/H180 斜率诊断，四路线都保持正对角和对角占优；但相对 Gate B 的条件局部 MIMO 对角，A1 只恢复约 `8.6%–13.2%`，Koopman约 `25.0%–38.9%`，PI-ODE约 `11.6%–17.6%`，DeepONet约 `13.1%–20.0%`。Gate B 也不是开环真值，不能把“更接近 Gate B”当成冠军规则；这些差异只能证明当前 raw logged-future-valve auxiliary objective 允许多种尺度分解。

所有受约束候选在每个 fold-seed 的 15 个 UTC 日上，logged 相对 shuffled 的日均优势均为正。但该对比使用了训练时同口径的 logged-action auxiliary，属于动作使用/一致性诊断，不是独立 causal placebo。

## 论文与模型边界

RM2 目前可以支撑：

- measured-boundary、disturbance-conditioned 的 closed-loop world-model 架构；
- observed-policy 600 s 多步预测；
- 局部 `Tin−Tout` 动作依赖、稳定递推、constant-action identity；
- common-mode 小范围支持域反事实作为待校准接口。

RM2 仍不能支撑：

- 唯一的喷水/阀位 plant gain；
- 任意 `do(valve)`、独立双侧 differential 反事实；
- 喷水流量真值或阀位—流量标定；
- Koopman/PI-ODE/DeepONet/A1phys 的物理冠军；
- test、MS4 或闭环投运。

## 下一步：RM3

下一步不增加更复杂 operator，先修正 estimand。RM3 应采用 out-of-fold nuisance residualization/R-loss 思路：

1. 用 past history、SP、Tin 和工况预测未来阀位与局部温降，严格产生 OOF action/outcome innovations；
2. 冻结 residual branch，显式 response operator 只通过 `outcome innovation ≈ G(c)·action innovation` 的正交矩训练，不能再用 raw logged future valve 直接制造 auxiliary advantage；
3. primary gate 采用 UTC 日/连续块的 H60/H180 correct-minus-wrong、lead/placebo 和 common/differential 支持，而不是单一 MAE；
4. 先校准 A1 scheduled 与 common-only 两个参考，再决定是否值得让其它 operator 进入相同正交目标；
5. 若跨 fold 的响应尺度仍不能收敛，只报告部分辨识区间/鲁棒模型集合，并转向现场小幅预注册 SP 激励。

机器审计见 `results/phase3_5/ms3r_gatec_rm2/supervisor_audit_validation.json`，复算入口为 `experiments/phase3_5/audit_ms3r_gatec_rm2.py`。
