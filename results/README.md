# results/ 索引 — 实验编号 → 阶段/协议/文档

> 命名: `results/exp_0XX_<描述>/`（目录，含 agg JSON + checkpoints）或 `exp_0XX_<描述>.json`（单文件）。
> 同一实验不同协议必须分目录或加后缀（如 `exp_051_boundary_fix_H10/`）。

> 当前项目口径以 [`TODO.md`](../TODO.md)、[`SUPERVISOR_REVIEW_2026-08-07.md`](../docs/SUPERVISOR_REVIEW_2026-08-07.md) 和 [`PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md) 为准。这里记录结果位置，不表示历史数字是独立测试结果或仍支持当时结论。

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
| exp_025_* (M0-M11, B1-B6, scaler) | 历史统一基准消融 → M7 预测 baseline（非全项目定稿） |
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

## CFE / 观测事件架构 — 历史原型

> P0 审计：这些目录没有提供因果 ground truth。exp_106/112 逐 epoch 使用 test 选 checkpoint；exp_109/110 合并 val+test 事件；同名 CFI 存在不同量纲与 silent fallback。正式 Phase 4 不直接复用其 best 数字。

| 目录/文件 | 内容 | 当前解释 |
|---|---|---|
| `cfe_groundtruth/` | 第一版匹配 DiD 响应 | 小事件集；观测闭环 protocol prototype |
| `cfe_groundtruth_p2/` | 扩展匹配 DiD 响应，79 个 val+test 事件 | 观测事件响应参考；不是 ground truth，且 val/test 混合 |
| `exp_106_causal_arch/` | A1/A3/B1 训练结果与多变体 | test-selected 历史探索；A1phys 仅保留 baseline 身份 |
| `exp_107_did_eval/did_eval.json` | 跨时程 DiD/CFI 评测 | 历史诊断；不能替代独立 test |
| `baselines_exp110/results.json` | M5/M7/M9、A1phys 与 B1 基线 | val+test 事件筛选的历史横向审查 |
| `exp_111_koopman_free/summary.json` | Koopman free-head 初步对照 | 预实验 |
| `exp_112_koopman_full/summary.json` | 3 seeds、50-epoch cap（9 runs 均早停）的 MLP/Koopman/null 对照 | MAE 是 test-selected pilot；CFI 实际为 n=16 fallback，不能关闭任何路线 |

`cfe_groundtruth_p2/` 只入库 `did_response.npz`；exp_112 写死读取不存在的 `did_response.json`，且其 test-only 事件长度与 P2 不同，所以结果里的 0.869/0.821 不是 P2 CFI。`final_causal` 无 `cfe` 字段也印证了 fallback。

Fan20-SST 灰箱主干、Fan17/21 嵌套组件、Fan-state controlled Koopman 和时变灰箱路线目前没有正式结果目录，尚未实现与验证。Phase 4 新结果应写入 content-addressed、不可覆盖目录，并保留 raw predictions 与 manifest。
