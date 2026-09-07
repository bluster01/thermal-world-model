# v07 norew_local_replay — 数值一致性重评估报告（2026-09-07, Linux 执行侧）

## 协议

- 范围：v07 full reissue 双侧 T1 全部已产出 checkpoint（5 臂 × 3 seeds = 30；sideB seed2 组 5 臂训练未完成，MISSING）
- 模式：**只重评估、不重新训练**；不读 test（SPLIT_VAL=验证段）；无任何训练调用
- 命令：`evaluate_windows`（与 run_matrix._train_and_eval 完全同参）：n_windows=256, batch_size=32, history_steps=96, horizon=18, boundary_mode=spec.boundary_mode, seed=50_000+spec.seed, device=cuda
- 精度档位：`torch.set_float32_matmul_precision("highest")`（v07 跑批未带 `--tf32`，ledger flags 确认 matmul_precision=highest、compile_substep=false；本机重放同样不 compile）
- 对比基准：`side{A,B}/metrics/<run_id>.pt` 的 committed (256,18) 张量（逐元素，非统计）

## 结果

| 侧 | 已评 | exact（mae+nll 逐位=0） | 均值差 | 最大绝对差 | day_ids 一致 |
|---|---|---|---|---|---|
| A | 15/15 | **15/15** | 0.0 | 0.0 | ✓ |
| B | 10/15 | **10/10** | 0.0 | 0.0 | ✓ |
| 合计 | 25 | **25/25** | 0.0 | 0.0 | ✓ |

MISSING（预期，非缺陷）：sideB t1_physics_only_seed2 / closure_cons_seed2 / closure_steam_seed2 / latent4_seed2 / closure_cons_norew_seed2——sideB 矩阵尚在训练（seed2 组未落盘），完成后可补跑。

## 与 Windows 侧重放的对照（数值差异定位结论）

| 来源 | sideA t1_physics_only_seed0 H18 MAE | 与 1.749003 差 |
|---|---|---|
| committed（原保存，runner 落盘） | 1.749002814 | — |
| **Linux 本机重放** | **1.749002814** | **0.000000000（逐位）** |
| Linux 重放 + matmul_precision='high'（tf32 档对照实验） | 1.749041796 | +3.9e-5（均值）；单窗 max 0.0011 |
| Windows 重放（对侧） | 1.748900 | −0.000103（均值）；逐窗逐步 max 0.007672 |

**结论**：原保存值为 Linux 侧产物；本机同一 ckpt 的评估**确定性可复现（bit-exact）**。Windows 侧差异与数据/权重/源码身份无关（对侧已验证对齐，day_ids 一致证明窗口采样同源），属**平台/精度实现差异**——其特征（相对 6e-5、单点 0.45%）与 tf32 档对照实验同量级但更大（2.6×），叠加因素待对侧三查确认：

1. `torch.get_float32_matmul_precision()` 是否 = "highest"（v07 协议值；he 若为 high/medium 即接近根因）
2. `torch.__version__` / `torch.version.cuda` / GPU 型号（cuDNN→GRU kernel 累加顺序差异是 1e-4 级常见来源）
3. 若前两项正常：导出同一窗口 `model.forecast` 首步 `temps_mu` 中间张量逐位对比（Linux 侧可 30s 导出）

## 判定建议（供 Supervisor/用户裁决）

本机重放 25/25 bit-exact ⇒ 重评估协议在本机**完全可复现**，committed 数字无漂移。Windows 侧差异不影响本机重评估结论；若对侧需在同一环境重放，请先完成上列三查，再按"统计等价"口径（既定跨机惯例：比统计量不比字节）判定，而非抬高容差直接放行。

## 产物

- `replay_summary.json` — 逐臂完整记录（wall_s/mae/nll 各口径 diff/max_at 位置/day_ids_match/exact）
- `replay_stdout.log` — 全量运行日志
