# Final-WM 可信度修复 Task 5–8 本地审计（2026-09-01）

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-01
- Verification Status: VERIFIED
- Integrity Pass Date: 2026-09-01
- Version Label: validation_v1
- Upstream Dependencies: code_plan_v1, canonical_v2.2, matrix_v0.7

## 范围与结论

本审计只验证 v0.6/v0.7 全量重发的本地实现、冻结协议和可执行性，不包含 Linux 正式训练结果。

**结论：Task 5–8 本地闭合，批次 `final_wm_v07_full_reissue_v1` 可交 Linux 一次性执行。** test 仍锁定；旧结果仍为 historical/superseded；本结论不升级论文 verdict。

## 逐项结果

| Task | 审计结果 | 关键证据 |
|---|---|---|
| 5 D-SYN | PASS | teacher 直接遍历 `transition.raw`；quick 实测扰动 34 个参数且 L2>0；no-op fail-closed |
| 6 raw quality | PASS | range/coverage 在 clip/fill 前执行；派生通道按源覆盖率判定；两路喷水负零漂使用已审计 raw range，随后 clip |
| 7 anchors/identity | PASS | validation anchor seed 每 run 固定；record/properties/init/anchor 内容哈希进入 resume fingerprint；quick/full 目录互斥 |
| 7 manifest | PASS | clean-start preflight、完整 run/seed、D-SYN、summary、ledger、checkpoint、metrics 与输入 SHA-256 绑定；包内相对路径可跨机复核 |
| 8 freeze | PASS | 每侧 39 个训练 run；所有训练臂 120/20；R1=`closure_cons_norew`；双侧分报；唯一 Linux runbook 和 registry gate 一致 |

## 生产式实物 smoke

1. 从本地 3.99 GB `all_merged_10s.csv` 重建 side A canonical v2.2：`n=707,709`，7/7 alignment corr=1.0000，raw/source quality 全过。
2. 在该新制品上顺序跑通 D-SYN、O1、T1、B1、J1、R1 quick。
3. summary 含五个正式单元；R1 arm=`closure_cons_norew`、status=`SMOKE`；不存在 `manifest.json`，证明 quick 未越权生成权威证据。

quick 数值不用于科学判决；它只验证接口、产物隔离和完整控制流。

## 验证记录

- `python -m pytest tests/final_wm -q`：193 passed。
- `python -m pytest tests/phase35/test_experiment_status.py tests/final_wm/test_jepa.py -q`：24 passed。
- `python -m pytest tests/final_wm/test_audit_manifest.py tests/phase35/test_experiment_status.py -q`：10 passed。
- `python -m compileall -q src/final_wm experiments/final_wm`：PASS。
- registry/config JSON 全解析；`experiment_status.py --check --json`：valid=true，active/linux gate 均为 `final_world_model_pipeline`。
- `git diff --check`：无内容错误；仅根 README 的既有 CRLF 提示。

## Linux 后续门

Linux 只能执行 `results/final_wm/v07_full_reissue_runbook_20260901.md`。任何异常原样停止，不自动重试。回传后先验 manifest，再只读复算各单元判决；在此之前 `results_returned=false`、`audited=false`、论文/test 均不解锁。
