# 执行侧 runbook：修正案 v0.4 全档判决（closure_cons_norew ×3 + 双栈探针）

2026-08-22 设计侧发。前置：本地判决档已按预注册规则**采纳 norew**
（`results/final_wm/v04_rewet_decision_20260822.json`：判据 i/ii/iii 全真——
intact v1 下游反号复现 sh2_in +4.63/final +2.31 @60s，norew 全负；
val NLL 3.494 ≤ 3.545+0.05；v2 不退化）。本 runbook 执行**终审**：
全档三 seed 对跑 + 双栈探针。

## 0. 前置

```bash
cd ~/thermal-world-model && git pull   # 取 v0.4 提交（本地 HEAD）
python -m pytest tests/final_wm/ -q    # 预期 132/132
```

- 指纹纪律：v0.4 改 training/transition 代码 → git tree hash 变 →
  **closure_cons×3 既有新鲜 ckpt 不可续跑**（预期行为）。判决对照直接引用
  890bd15 栈的**已审计指标**（metrics/pt 文件无指纹门槛，探针加载不受影响，
  state_dict 形状未变）。
- 不改任何阈值/门；norew 臂不进 T1 verdict `nested` 对（arm-filter 保护）。

## 1. T1 消融臂全档（~5h）

```bash
python -m experiments.final_wm.run_matrix --phase matrix \
  --record data/canonical_sideA.npz --side A --units t1 \
  --arm-filter closure_cons_norew \
  --properties-npz data/iapws_surrogate.npz --out artifacts/final_wm \
  --device auto --compile --tf32
```

验收：ledger 三条 final（arm=closure_cons_norew, seeds 0/1/2, commit=HEAD）；
metrics 文件 `metrics/t1_closure_cons_norew_seed{0,1,2}.pt`。

## 2. R1 消融栈证据门（~40min）

```bash
python -m experiments.final_wm.run_matrix --phase matrix \
  --record data/canonical_sideA.npz --side A --units r1 \
  --r1-arm closure_cons_norew \
  --properties-npz data/iapws_surrogate.npz --out artifacts/final_wm \
  --device auto --compile --tf32
```

产物：`r1_report_closure_cons_norew.json` + summary 块 `r1_closure_cons_norew`
（**不动** `r1` 块与 `r1_report.json`）。关注：三 seed 的
direction.frac_negative（intact 栈 32/32,32/32,28/32）与 leakage
delta_vs_shuffle（intact 栈 0.55/1.09/~1pp）。

## 3. leakdist 消融栈 ×3（~2h）

```bash
for s in 0 1 2; do
  python -m experiments.final_wm.run_matrix --phase leakdist \
    --record data/canonical_sideA.npz \
    --checkpoint artifacts/final_wm/checkpoints/t1_closure_cons_norew_seed${s}.pt \
    --arm closure_cons_norew --seed ${s} \
    --properties-npz data/iapws_surrogate.npz --out artifacts/final_wm \
    --device auto --compile --tf32
done
```

## 4. auditpack 消融栈 seed0（~1.5h）

```bash
python -m experiments.final_wm.run_matrix --phase auditpack \
  --record data/canonical_sideA.npz --side A \
  --checkpoint artifacts/final_wm/checkpoints/t1_closure_cons_norew_seed0.pt \
  --arm closure_cons_norew --seed 0 \
  --properties-npz data/iapws_surrogate.npz --out artifacts/final_wm \
  --device auto --compile --tf32
```

产物：`auditpack_sideA_closure_cons_norew.json`（不覆写 intact 的
`auditpack_sideA.json`）。rewetting_ablation 探针在 norew 栈上应为恒等
（aW 已冻结→消融无效应）——这是臂正确性的自检。

## 5. O1 重跑（已发 runbook，独立进行，不变）

按 `repair1_rerun_runbook_20260821.md` §5 原样执行（现栈修复陈旧判决）。

## 6. 回传清单

`artifacts/final_wm/` 的 ledger.jsonl、matrix_summary_sideA.json、
metrics/t1_closure_cons_norew_seed*.pt、r1_report_closure_cons_norew.json、
leakdist_closure_cons_norew_seed*.json、auditpack_sideA_closure_cons_norew.json、
以及 O1 重跑全部产物。checkpoints 按需（大）。

## 判决规则（终审，预注册于修正案 v0.4）

采纳 norew 为侧A生产臂 当且仅当三 seed 一致满足：
方向全对（R1 frac_negative=1.0 且 auditpack/探针 v1 下游无反号）
且 val NLL 三 seed 中位不差于 intact 栈（890bd15 已审计值）+0.05。
否则保持③现状并将 v1 反号登记为结构局限入论文"限制"节。
