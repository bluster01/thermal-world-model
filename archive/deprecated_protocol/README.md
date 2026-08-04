# 归档: 协议作废的 Phase 2 闭环结果

> 归档日期: 2026-08-04
> 原因: 主协议 `experiments/phase2_mpc/exp_027_dwm_mpc.py` 的 `simulate()` 存在 4 项结构性缺陷,
>       所有依赖它产出的闭环 MPC vs PID 数值均不可信。
> 处置: 移入本目录留档 (不删除), 待新协议 `experiments/phase2_mpc/eval_protocol.py` 重跑后作新旧对比。

---

## 1. 为何作废 — 四项协议缺陷

| 缺陷 | 问题 | 对数值的影响方向 |
|------|------|-----------------|
| **P0-A** 假 PID | PID 组使用历史阀位录像回放 (`test_raw[gi:gi+H_OUT, VALVE_IDX]`), 并非反馈控制器。PID 无法感知仿真中注入的扰动, MPC 每步重规划却可以。 | **系统性高估 MPC 优势**, 扰动越大高估越多 |
| **P0-B** 自评分 | MPC 组与 PID 组的温度均由**被测世界模型自己**预测。M5 在 M5 世界跑、M7 在 M7 世界跑, 两者"真值"不同源。 | **模型误差被抵消 → 高估闭环性能**; 且跨模型 RMSE **不可比** |
| **P0-C** 非物理扰动 | 扰动为标量偏置直接叠加在输出温度上, 而窗口内负荷/煤量/流量等 39 维状态仍为历史真值 → 物理自相矛盾。对 ARX 而言这是纯不可建模噪声。 | **人为放大线性 MPC 的崩溃**; 深度 WM 相对受益 |
| **P0-D** 动作未回填 | 窗口推进时只覆盖温度列, 阀位列仍写入历史真值 → 世界模型看到的动作历史并非控制器实际执行的动作。 | 闭环因果链断裂, 影响不确定 |
| 附加 | MPC 侧有 KF + SMA6 执行端平滑, PID 侧没有 | 高估 MPC 的动作平滑性 (TV 指标) |
| 附加 | 超温指标以**步数**统计 (`(y > T_MAX).sum()`), 文档中却按秒表述 | 超温时间数值口径混乱 (如 "超温 1s" 不可能是整数步) |

**综合判断**: 归档结果给了 MPC **三重不公平优势** (闭环 vs 回放、自评分、输出侧扰动)。
已公布的 "MPC vs PID 改善 −21.1%" 等结论在修复协议后预计明显缩水, 必须以新协议重跑为准。

---

## 2. 归档内容

### results/ — 闭环仿真结果 (共 29 组)

| 实验 | 内容 | 需重跑的替代实验 |
|------|------|-----------------|
| `exp_027_M7` | DWM-MPC 主循环 (grad/cem, α 扫描) | S1 |
| `exp_029_mpc_conditions` | 分工况 MPC 评估 | S7 |
| `exp_032_sp_traj_ab` | SP 轨迹目标 A/B | — |
| `exp_034_M11`, `exp_035_joint`, `exp_037_sp_ff`, `exp_039_joint_ff`, `exp_040_pipeline_const` | SP/联合通道闭环前馈 | Phase 3 重设计 |
| `exp_041_fair_H10`, `exp_042_trajectories` | 公平协议 + 轨迹图数据 | S7 |
| `exp_045_lti`, `exp_046_stats` | LTI 对照 + Wilcoxon 检验 | S4, S7 |
| `exp_047_mstep`, `exp_050_mstep_dist` | M_STEP 扫描 (无扰动/扰动) | S2 |
| `exp_048_horizon`, `exp_048_robustness`, `exp_049_disturbance` | 视野扫描 / 鲁棒性 / 扰动注入 | S2, S6 |
| `exp_051_boundary_fix`, `exp_051_boundary_fix_H10`, `exp_052_overlap`, `exp_063_smooth_scan`, `exp_064_final_smooth` | 边界跳变修复与平滑策略扫描 (hard/blend/inert/overlap) | S2 后重扫 |
| `exp_053_hplan`, `exp_059_hplan_full`, `exp_059b_hplan_newbench`, `exp_059c_hplan_fixed` | H_PLAN 扫描 (四代, 结论互相冲突) | **S2 (定论 H=10 vs 18)** |
| `exp_054_horizon_eval`, `exp_055_stepwise`, `exp_056_horizon_curves`, `exp_058_single_model` | 预测视野与逐步曲线 | S2 |
| `exp_060_risk` | CVaR 风险敏感 MPC (λ 扫描) | **S1 成本口径 (c)** |
| `exp_062_final_eval` | 阶段性最终评测 | S7 |
| `exp_067_linear_mpc`, `exp_068_linear_mpc` | 线性 ARX-MPC 对照 | **S4 (崩溃机制诊断)** |
| `exp_070_ensemble` | 集成 WM + grad/cem | S7 |
| `exp_073_policy` | IQL / TD3+BC / SAC 直接策略 | **S5 (baseline 公平性)** |
| `exp_074_det_wm`, `exp_074_pressline` | 确定性 WM 对照 / 压线控制 | **S1 (M5 vs M7 重判)** |

### figures/ — 由上述结果绘制的图 (共 19 张)

`fig3_clean` `fig3_mpc_trajectories` `fig_h1h18_trajectories` `fig_hplan_sweep` `fig_kf_effect`
`fig_m5_vs_m7_gap` `fig_matched_control` `fig_mpc_kf_6tracks` `fig_mpc_kf_sma6_6tracks`
`fig_multi_track_compare` `fig_multi_trajectories` `fig_offset_tradeoff` `fig_rejudge_curves`
`fig_smooth_modes` `fig_step_final` `fig_step_final2` `fig_step_response`
`fig_supervisory_mpc_smoke` `fig_m12_effect`

其中 `fig_m5_vs_m7_gap.png` 对应 exp_094 的主模型切换依据 —— 该结论受 **P0-B 自评分** 影响最严重
(M5 与 M7 在各自世界里评测), 是本次归档的核心动机。

---

## 3. 未归档 (仍然有效)

以下结果**不依赖闭环仿真**, 不受协议缺陷影响, 保留在 `results/`:

- `exp_001`–`exp_026b`: Phase 1 开环预测与消融 (训练/验证/测试指标、rollout 曲线、敏感性、σ 校准)
- `exp_028_diff_verify`: 世界模型可微性验证 (模型属性, 非闭环)
- `exp_031_sp_events`: 历史数据 SP 事件研究 (数据事实)
- `exp_033_pi_params`, `exp_036_pi_v2`, `exp_038_sp_valve`: PI 控制器与 SP→阀位系统辨识 (数据驱动)
- `exp_044_fidelity`: WM 开环保真度 (1200s rollout MAE)
- `exp_071_rl_data`: 离线 RL 数据集

保留的图为事件研究 / 系统辨识 / 开环对比类。

> 待复核: `fig_aug_effect` `fig_deeponet_effect` `fig_physreg_effect` `fig_m13phys_effect`
> 这几张为模型变体效果图, 若其数值来自闭环仿真则应一并归档。

---

## 4. 恢复方式

```bash
git mv archive/deprecated_protocol/results/<exp_id> results/<exp_id>
```

完整历史保存在 git 中, 归档为 `git mv` 操作, 未丢失任何提交记录。
