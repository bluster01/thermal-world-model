# results/ 索引 — 实验编号 → 阶段/协议/文档

> 命名: `results/exp_0XX_<描述>/`（目录，含 agg JSON + checkpoints）或 `exp_0XX_<描述>.json`（单文件）。
> 同一实验不同协议必须分目录或加后缀（如 `exp_051_boundary_fix_H10/`）。

## Phase 1 — 世界模型 (文档: docs/phase1_report.md, phase1_conclusions_audit.md)

| 目录/文件 | 内容 |
|---|---|
| exp_001 ~ exp_007 | 早期架构/损失扫描 (MSE/GaussNLL/β-NLL) |
| exp_010_* | 组件消融 (VarAttn/MLP/RevIN) |
| exp_011_* | 动作编码消融 (bypass/FiLM/decoder) |
| exp_012_absvalve | 绝对阀位 (差分→绝对, 关键修复) |
| exp_013_signreg_* | 符号正则扫描 (结论: 有害) |
| exp_014_* exp_015_dualvalve | 延迟/双阀符号正则 |
| exp_016_* | 符号正则消融矩阵 (8配置) |
| exp_017_* | 多步符号正则 (t≥8约束) |
| exp_018_A exp_021_A | LSTM/GRU 基线 |
| exp_020_n4sid.json | N4SID 线性基线 |
| exp_022_direct_wm exp_023_direct_aligned | Direct WM (多步direct) |
| exp_024_sigma_persistence.json | σ 持续性 |
| exp_025_* (M0-M11, B1-B6, scaler) | **统一基准消融 → M7 定稿** (results/exp_025_M7/) |
| exp_026* exp_028 | 可微性校验 (backprop-through-WM) |

## Phase 2 — DWM-MPC (文档: docs/phase2_results.md §1-12)

| 目录/文件 | 内容 | 章节 |
|---|---|---|
| exp_027_M7/ | MPC 主循环 (grad/CEM, α/H 扫描) | §1-7 |
| exp_029_mpc_conditions.json | 11工况评测 | §2 |
| exp_031_sp_events.json | SP 事件研究 | — |
| exp_032_sp_traj_ab.json | SP轨迹目标 vs 标量 (方案1) | — |
| exp_034_M11 exp_035_joint | 路线A变体 (SP联合) | — |
| exp_036~040 | 路线B 前馈/PI (见下) | — |
| exp_041_fair_H10.json | 公平对比 (PID也走WM) | §8 |
| exp_042_trajectories.json | 多轨迹 9/9 全胜 | §8-9 |
| exp_044_fidelity/ | 保真度 1200s MAE 1.2°C | §8 |
| exp_045_lti/ | LTI-MPC 线性基线 (+14.6% 增量) | §8 |
| exp_046_stats/ | Wilcoxon p<1e-4 | §8.3 |
| exp_047_mstep.json | M_STEP 扫描 (旧, H=10 截断) | §8.4 |
| exp_048_horizon/ | 预测长度扫描 60-180s | — |
| exp_048_robustness.json | WM噪声鲁棒性 | §8.4 |
| exp_049_disturbance.json | 扰动响应 (MPC↓34% vs 无控制) | §9 |
| exp_050_mstep_dist/ | **扰动世界 M_STEP×拼接扫描** | §10 |
| exp_051_boundary_fix/ | **边界修复三方案 H=18** | §11 |
| exp_051_boundary_fix_H10/ | 边界修复 H=10 (主协议验证) | §11.4 |
| exp_052_overlap/ | **重叠一致性 (平滑切换)** | §12 |

## Phase 3 / 路线B — SP前馈 (文档: docs/phase2_plan.md)

| 目录/文件 | 内容 |
|---|---|
| exp_033_pi_params.json exp_036_pi_v2.json | PI 控制器行为辨识 (SP→阀位) |
| exp_037_sp_ff.json | SP 前馈拟合 |
| exp_038_sp_valve.json | SP→阀位→温度 通道分析 |
| exp_039_joint_ff/ exp_040_pipeline_const_*.json | 联合优化 (结论: 阀位>联合>SP) |
