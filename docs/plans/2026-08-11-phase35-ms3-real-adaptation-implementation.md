# Phase 3.5-MS3 实施计划

## 目标

实现并冻结 A/B 交叉控制回路的 validation-only 真实适配批次；Linux 只生成 cache、训练 12 runs、运行冻结汇总并原样回传。

## 本地实现清单

1. `prepare_ms3_cross_data.py`
   - 校验 4 GB source size/SHA；
   - 一次扫描生成 A阀→右温、B阀→左温两个 cache；
   - 校验严格 10 s、写 source/mapping/age provenance。
2. `real_training.py`
   - past-only encoder/free 与 3-pole response；
   - joint/free-only 两模式；
   - validation selector 与 logged/baseline/shuffled diagnostics；
   - checkpoint、history、manifest、metrics、episode metrics。
3. `ms3_real_adaptation.py`
   - 冻结矩阵内容哈希；
   - 12-run dry-run/execute/skip-existing；
   - cache mapping/source pins 与 test-artifact fail closed。
4. `summarize_ms3_real_adaptation.py`
   - 12/12 artifact/trajectory/structural replay；
   - UTC-day block bootstrap；
   - 2/3 seeds per side、双侧主判决；
   - deterministic checkpoint archive。
5. 测试
   - cross mapping、matrix mutation、action-blind/reference identity；
   - shuffled delta design；
   - joint/free-only CPU smoke；
   - 双侧 2/3 seed 决策树。

## 放行顺序

```text
MS5 weight-level audit
→ cross-platform legacy hash fix
→ MS3 design/matrix/code
→ focused tests + full phase35 regression
→ clean commit + push
→ Linux preflight/cache build
→ 12-run validation
→ frozen summary + raw logs
→ local checkpoint/episode/bootstrap audit
→ close or diagnose MS3
```

任一 preflight 红项必须停止。Linux 不裁决“非科学问题后继续”，不改源代码/阈值/状态文档。summary exit 2 表示科学门失败但产物仍须全部提交。

## 预期产物

每 run：`checkpoint_best_val.pt`、`history.json`、`manifest.json`、`metrics_validation.json`、`episode_metrics_validation.json`。

批次：`summary_validation.json`、`checkpoints_validation.tar`、cache manifests、完整 stdout/stderr/exit code/environment/command/git SHA。test artifacts 必须不存在。
