# 执行侧 runbook：T1 比较判决重发——physics_only ×3（裁定 B1）

2026-08-23 设计侧发。前置：用户裁定 A 已采纳（生产臂 = closure_cons_norew）。
本 runbook 只补一块缺口：当前栈上 physics_only 基线未训，T1 比较判决无法重发。

## 0. 前置

```bash
cd ~/thermal-world-model && git pull   # 取 ≥ cf1ecb2（含终审审计与裁定记录）
python -m pytest tests/final_wm/ -q    # 预期 132/132
```

## 1. physics_only ×3 全档（预估 ~1.5-2.5h，臂无 closure 网络更轻）

```bash
python -m experiments.final_wm.run_matrix --phase matrix \
  --record data/canonical_sideA.npz --side A --units t1 \
  --arm-filter physics_only \
  --properties-npz data/iapws_surrogate.npz --out artifacts/final_wm \
  --device auto --compile --tf32
```

验收：ledger 三条 final（arm=physics_only, seeds 0/1/2, commit=HEAD,
无 RESUMED）；metrics 文件 `metrics/t1_physics_only_seed{0,1,2}.pt`。
**arm-filter 下 runner 不写 T1 verdict 块**（保护纪律，预期行为）；
比较判决由设计侧用 runner 的 `_seed_passes`/THRESH_T1_NLL 对
norew 与 physics_only 的 metrics 文件计算，留痕入审计。

## 2. 回传

ledger.jsonl + metrics/t1_physics_only_seed*.pt（checkpoints 按需）。

## 注意

- 本 run 只产生基线指标；不得手工编辑任何 verdict 块。
- physics_only 与 norew 同栈同预算（epochs=60/patience=10），比较对称。
