# 执行侧长任务 runbook：侧B完整复现矩阵（2026-08-23 设计侧发）

目的：把侧A终审结论在**独立镜像侧**复现一遍，做实或证伪——
1. F3 复现检验：intact 闭包在侧B是否复现 v1 下游反号？norew 是否同样修复？
2. T1 重发判决复现：闭包精度平价结论在侧B是否成立？
3. 方向证件排他性复现：physics_only / intact / norew 三方 R1 对照在侧B重跑。
4. O1/D-SYN 证件复现。

数据侦察（本地已查）：侧B = 同时段同切分镜像侧；v1 均值 0.221、76% 时段 >10%
开度——**v1 活跃，再湿辨识性问题在侧B成立**，消融对照有意义。

## 0. 前置与停损点

```bash
cd ~/thermal-world-model && git pull   # 须 ≥ f22b9a1（含 R1 无闭包 skipped 修补）
python -m pytest tests/final_wm/ -q    # 预期 132/132
mkdir -p artifacts/final_wm_sideB
```

- **全部产物写 `--out artifacts/final_wm_sideB`**（检查点名与侧A相同，必须隔离目录）。
- record 路径用 `artifacts/final_wm/canonical_sideB.npz`（仓库内规范位置；
  `data/` 路径已废弃）。
- 门槛/判决函数全部沿用冻结 matrix_spec（侧不可知），不得改任何阈值。
- **停损点**：步骤 1 D-SYN 若不过门 → 停止后续训练，回传结果等设计侧裁定。

## 1. D-SYN 侧B门禁（~1h，gate）

```bash
python -m experiments.final_wm.run_matrix --phase dsyn \
  --record artifacts/final_wm/canonical_sideB.npz --side B \
  --out artifacts/final_wm_sideB --device auto --compile --tf32
```

验收：`dsyn_verdict.json` PASS×3（预期 gap 类指标 >30% 门）。

## 2. T1 三臂 ×3（~12-15h，主体）

顺序固定（先基线后闭包，便于中途回传）：

```bash
for ARM in physics_only closure_cons closure_cons_norew; do
  python -m experiments.final_wm.run_matrix --phase matrix \
    --record artifacts/final_wm/canonical_sideB.npz --side B --units t1 \
    --arm-filter $ARM \
    --properties-npz artifacts/final_wm/iapws_surrogate.npz \
    --out artifacts/final_wm_sideB --device auto --compile --tf32
done
```

验收：ledger 9 条 final（3 臂 ×3 seed，commit=HEAD，无 RESUMED）；
metrics 9 件。arm-filter 下不写 verdict 块（纪律保持）。

## 3. R1 三臂对照（<30min，探针）

```bash
for ARM in physics_only closure_cons closure_cons_norew; do
  python -m experiments.final_wm.run_matrix --phase matrix \
    --record artifacts/final_wm/canonical_sideB.npz --side B --units r1 \
    --r1-arm $ARM \
    --properties-npz artifacts/final_wm/iapws_surrogate.npz \
    --out artifacts/final_wm_sideB --device auto --compile --tf32
done
```

产物：`r1_report_{physics_only,closure_cons,closure_cons_norew}.json` +
summary 三个独立块。核心读数：各 seed direction frac_negative（60s 与 240s）。

## 4. O1 三模式 ×3（~7h）

```bash
python -m experiments.final_wm.run_matrix --phase matrix \
  --record artifacts/final_wm/canonical_sideB.npz --side B --units o1 \
  --properties-npz artifacts/final_wm/iapws_surrogate.npz \
  --out artifacts/final_wm_sideB --device auto --compile --tf32
```

（O1 无 arm-filter → 会写 o1 verdict 块，正常。）

## 5. 收尾探针（<1h）

```bash
for s in 0 1 2; do
  python -m experiments.final_wm.run_matrix --phase leakdist \
    --record artifacts/final_wm/canonical_sideB.npz \
    --checkpoint artifacts/final_wm_sideB/checkpoints/t1_closure_cons_norew_seed${s}.pt \
    --arm closure_cons_norew --seed ${s} \
    --properties-npz artifacts/final_wm/iapws_surrogate.npz \
    --out artifacts/final_wm_sideB --device auto --compile --tf32
done
python -m experiments.final_wm.run_matrix --phase auditpack \
  --record artifacts/final_wm/canonical_sideB.npz --side B \
  --checkpoint artifacts/final_wm_sideB/checkpoints/t1_closure_cons_seed0.pt \
  --arm closure_cons --seed 0 \
  --properties-npz artifacts/final_wm/iapws_surrogate.npz \
  --out artifacts/final_wm_sideB --device auto --compile --tf32
python -m experiments.final_wm.run_matrix --phase auditpack \
  --record artifacts/final_wm/canonical_sideB.npz --side B \
  --checkpoint artifacts/final_wm_sideB/checkpoints/t1_closure_cons_norew_seed0.pt \
  --arm closure_cons_norew --seed 0 \
  --properties-npz artifacts/final_wm/iapws_surrogate.npz \
  --out artifacts/final_wm_sideB --device auto --compile --tf32
```

（双 auditpack：intact 的 rewetting_ablation 探针给出侧B aW 效应幅度——
F3 复现的关键读数；norew 的应恒等。）

## 6. 回传清单

`artifacts/final_wm_sideB/` 的 ledger.jsonl、matrix_summary_sideB.json、
metrics/ 全部、dsyn_verdict.json、r1_report_*.json、leakdist_*.json、
auditpack_*.json。checkpoints 按需。T1/R1 比较判决由设计侧复核后重发，
执行侧只跑产物、不写判决。

预估总时长 ~20-24h 串行。完成后设计侧出《侧B复现审计》做实或降级侧A结论。
