"""Two-side coupling diagnostic (2026-08-26): is single-side modeling misspecified?

Data-only (no GPU). Uses the four valve feedbacks recoverable from the npz set:
  sideA v1 old = 一级B, sideA v2.1 valve1 = 一级A, valve2 = 二级B
  sideB v1 old = 一级A, sideB v2.1 valve1 = 一级B, valve2 = 二级A
Questions:
  Q1 fair mid-training comparison of the running arms at matched epochs
  Q2 per-load-bin valve statistics (is 一级A degenerate at high load, where the
     corrected-record arm exploded?)
  Q3 common/differential decomposition of stage-1 (how much is separable at all)
  Q4 own-side vs cross-side correlation of each valve with the observed temps,
     by load bin -- does the cross-side path dominate (Gate B H600 finding)?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.contracts import BOUNDARY_ELEMENTS, OBSERVATION_ELEMENTS

CH_FINAL = OBSERVATION_ELEMENTS.index("final_outlet_temp")
CH_SH1O = OBSERVATION_ELEMENTS.index("sh1_outlet_temp")
IDX_FLOW = BOUNDARY_ELEMENTS.index("steam_flow")
A = ROOT / "artifacts/final_wm"
P = ROOT / "results/final_wm/probes_20260824"


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100 or np.std(x[m]) < 1e-9 or np.std(y[m]) < 1e-9:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def stuck_rate(x, tol=1e-6):
    return float((np.abs(np.diff(x)) < tol).mean())


print("=" * 70)
print("Q1  matched-epoch val_nll comparison of the two corrected-record arms")
print("=" * 70)


def val_curve(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    out, seen_final = {}, False
    for r in rows:
        if r.get("final"):
            seen_final = True
            continue
        if "val_nll" in r and "epoch" in r and not seen_final:
            out[r["epoch"]] = r["val_nll"]
    return out


v1fix_arm1 = val_curve(P / "v1fix_probe/ledger.jsonl")
rewet = val_curve(P / "rewet_probe/ledger.jsonl")
common = sorted(set(v1fix_arm1) & set(rewet))
print(f"norew(corrected) epochs={len(v1fix_arm1)}  intact(corrected) epochs={len(rewet)}")
for ep in [e for e in (5, 10, 15, 20, 25, 27) if e in common]:
    print(f"  ep{ep:3d}: norew={v1fix_arm1[ep]:8.3f}   intact={rewet[ep]:8.3f}")
if common:
    tail = common[-5:]
    print(f"  matched-epoch mean over {tail}: "
          f"norew={np.mean([v1fix_arm1[e] for e in tail]):.3f}  "
          f"intact={np.mean([rewet[e] for e in tail]):.3f}")

print()
print("=" * 70)
print("Q2/Q3/Q4  four-valve structure on the shared canonical span")
print("=" * 70)

a_old = np.load(A / "canonical_sideA.npz")
b_old = np.load(A / "canonical_sideB.npz")
a_new = np.load(A / "canonical_sideA_v2.npz")
b_new = np.load(A / "canonical_sideB_v2.npz")
TRIM = 12  # v2 trims 12 leading rows
valves = {
    "stage1_A": a_new["actions"][:, 0],
    "stage1_B": b_new["actions"][:, 0],
    "stage2_B": a_new["actions"][:, 1],
    "stage2_A": b_new["actions"][:, 1],
}
old_v1_A = a_old["actions"][TRIM:, 0]   # = 一级B (wrong side, what training used)
temps = {
    "left_final": a_new["obs"][:, CH_FINAL],
    "right_final": b_new["obs"][:, CH_FINAL],
    "left_sh1o": a_new["obs"][:, CH_SH1O],
    "right_sh1o": b_new["obs"][:, CH_SH1O],
}
flow = a_new["boundary"][:, IDX_FLOW]
split = a_new["split"]
print(f"span n={len(flow)}  flow range {flow.min():.0f}-{flow.max():.0f}")
print(f"sanity: old sideA v1 vs stage1_B corr={corr(old_v1_A, valves['stage1_B']):.6f} "
      f"(should be ~1.0)")

edges = np.quantile(flow, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
print("\n[Q2] per-load-bin valve stats (mean / std / stuck-rate)")
hdr = "  bin  flow-range      " + "".join(f"{k:>26}" for k in valves)
print(hdr)
for i in range(5):
    lo, hi = edges[i], edges[i + 1]
    m = (flow >= lo) & (flow <= hi if i == 4 else flow < hi)
    row = f"  Q{i+1}  {lo:5.0f}-{hi:5.0f}  "
    for k, v in valves.items():
        vv = v[m]
        row += f"  {vv.mean():5.3f}/{vv.std():5.3f}/{stuck_rate(vv):4.2f}"
    print(row)

print("\n[Q3] stage-1 common/differential decomposition by load bin")
s1a, s1b = valves["stage1_A"], valves["stage1_B"]
for i in range(5):
    lo, hi = edges[i], edges[i + 1]
    m = (flow >= lo) & (flow <= hi if i == 4 else flow < hi)
    c = (s1a[m] + s1b[m]) / 2
    d = (s1a[m] - s1b[m]) / 2
    tot = np.var(s1a[m]) + np.var(s1b[m])
    print(f"  Q{i+1}: corr(1A,1B)={corr(s1a[m], s1b[m]):+.3f}  "
          f"var_common={np.var(c):.5f} ({2*np.var(c)/tot*100:4.1f}%)  "
          f"var_diff={np.var(d):.5f} ({2*np.var(d)/tot*100:4.1f}%)")

print("\n[Q4] valve -> temperature correlation, own-side vs cross-side, by load bin")
print("     (sideA obs = LEFT temps; wiring: 1A->left, 2B->left, 1B->right, 2A->right)")
pairs = [("stage1_A", "left_final", "own"), ("stage1_B", "left_final", "cross"),
         ("stage2_B", "left_final", "own"), ("stage2_A", "left_final", "cross"),
         ("stage1_A", "left_sh1o", "own"), ("stage1_B", "left_sh1o", "cross")]
for i in range(5):
    lo, hi = edges[i], edges[i + 1]
    m = (flow >= lo) & (flow <= hi if i == 4 else flow < hi)
    parts = [f"{v}->{t}[{tag}]={corr(valves[v][m], temps[t][m]):+.3f}"
             for v, t, tag in pairs]
    print(f"  Q{i+1}: " + "  ".join(parts[:4]))
    print(f"       " + "  ".join(parts[4:]))

print("\n[Q4b] val-split only (the probe evaluation domain)")
mval = split == 1
for i in range(5):
    lo, hi = edges[i], edges[i + 1]
    m = mval & (flow >= lo) & (flow <= hi if i == 4 else flow < hi)
    if m.sum() < 500:
        print(f"  Q{i+1}: n={int(m.sum())} too few")
        continue
    print(f"  Q{i+1} n={int(m.sum()):6d}: "
          f"1A->left={corr(valves['stage1_A'][m], temps['left_final'][m]):+.3f}  "
          f"1B->left={corr(valves['stage1_B'][m], temps['left_final'][m]):+.3f}  "
          f"2B->left={corr(valves['stage2_B'][m], temps['left_final'][m]):+.3f}  "
          f"2A->left={corr(valves['stage2_A'][m], temps['left_final'][m]):+.3f}")
print("\ndone")
