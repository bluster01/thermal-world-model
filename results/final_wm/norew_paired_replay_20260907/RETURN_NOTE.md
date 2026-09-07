# norew paired replay — Linux 回传说明（2026-09-07/08）

## 执行信息

- 执行单：`results/final_wm/norew_linux_replay_runbook_20260907.md`（commit db9447b，分支 codex/norew-inference-replay）
- 执行位置：`/home/bluster/thermal-world-model-norew-replay`（detached worktree @ db9447b）
- 冻结命令（原样）：
  `python -u experiments/final_wm/replay_norew_pairs.py --package /home/bluster/final_wm_v07_full_reissue_v1 --out /home/bluster/final_wm_norew_paired_replay_20260907`
- 环境：Alloftime（torch 2.11.0+cu130, cuda 13.0, cudnn 91900, numpy/pandas 同 v07 批次）；GPU NVIDIA GB10
- 模式：纯推理重放；`training=false, leakage_probe=false, full_R1_verdict=未裁决, test_values_accessed=false`
- 完成状态：COMPLETE（无错误、无重试）

## 产物清单（本目录）

- `protocol.json` — 协议与环境记录（matmul=highest; cuda_matmul_tf32=false; cudnn_tf32=true）
- `isolation.json` — 数据隔离证明：original_rows=707709, decoded=636937（test 段 0 行解码）
- `t1_{physics_only,closure_cons,closure_cons_norew}_seed{0,1,2}.json` — 9 个 run 报告（预测重放检查 + 36 组响应单元：双阀 × H18/H60 × 3 seeds × 3 arms，含支持掩码）
- `t1_*_prediction.pt` — 9 个重放预测张量（mae/nll/crps/day_ids，与 saved 逐位一致）
- `summary.json` — 汇总（6 组等 UTC 日加权配对 MAE CI + runs 全量）
- `paired_replay_stdout.log` / `paired_replay_stderr.log` — 全量日志

## 核验结果

| 项 | 结果 |
|---|---|
| 预测重放（9/9） | `bit_exact=true, allclose=true`（mae/nll/crps max_abs_difference=0.0；day_ids 由重建索引断言相等） |
| 响应探针（36 单元） | 全部产出（含 n_unsupported 与支持掩码；越界单元如实保留，不产生 supported 结论） |
| 参数不变 | 9/9 断言通过（named_parameters 逐张量相等） |
| test 访问 | 0 行 |
| 训练/泄漏探针 | 未调用 |

## 数据观察（非裁决，供 Supervisor）

- paired MAE（equal-day bootstrap CI，n_days=13）：
  - closure_cons → closure_cons_norew：point +0.204~+0.375（CI 全正）——norew 口径预测 MAE 更差
  - physics_only → closure_cons：point −0.084~−0.168（CI 全负）——闭包臂优于无闭包
- 响应 delta 同 v07 方向门样本一致（norew 臂 H18 delta 约 −0.06~−0.10°C，H60 −0.24~−0.32°C；n_unsupported 18~180 随 horizon 增长，与 R1 report 的 support violation 同源）
