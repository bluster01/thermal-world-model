"""Independent design-side verification of the valve-alignment forensics.

Claims under test (execution-side report 2026-08-25):
  C1: canonical sideA (left temps) valve1 == 一级B反馈 (mis-wired; should be 一级A)
  C2: canonical sideA valve2 == 二级B反馈 (correct cross for stage-2)
  C3: canonical sideB (right temps) valve1 == 一级A反馈, valve2 == 二级A反馈
  C4: physical wiring -- stage-1 same-side, stage-2 cross (lag-30s diff corr)
  C5: 一级A/B valves correlate ~+0.784 (mitigation factor)
"""
import numpy as np
import pandas as pd

ALL = r"C:\Users\14020\Desktop\时间预测模型\AA数据中心\伊敏12.10\merged_all_data\all_merged_10s.csv"
V1A, V1B = "artifacts/final_wm/canonical_sideA_v2.npz", "artifacts/final_wm/canonical_sideB_v2.npz"

V = ["过热器一级减温器A侧喷水调节门阀位反馈", "过热器一级减温器B侧喷水调节门阀位反馈",
     "过热器二级减温器A侧喷水调节门阀位反馈", "过热器二级减温器B侧喷水调节门阀位反馈"]
T = ["选择后左侧一过喷水减温器出口", "选择后右侧一过喷水减温器出口",
     "选择后左侧二过喷水减温器出口", "选择后右侧二过喷水减温器出口"]

g = pd.read_csv(ALL, usecols=["time"] + V + T)
g["ep"] = pd.to_datetime(g["time"]).astype("int64") // 10**9
g = g.set_index("ep").sort_index()


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1])


print("== C1/C2/C3: canonical per-side actions vs 377 valve feedback columns")
for tag, path in (("sideA", V1A), ("sideB", V1B)):
    r = np.load(path)
    ts = r["timestamps"].astype(np.int64)
    sub = g.reindex(ts)
    for j, vn in enumerate(("valve1", "valve2")):
        vals = [corr(r["actions"][:, j].astype(np.float64) * 100.0,
                     pd.to_numeric(sub[c], errors="coerce").to_numpy(float)) for c in V]
        best = int(np.argmax(np.abs(vals)))
        print(f"  {tag} {vn}: " + " ".join(f"{x:+.3f}" for x in vals)
              + f"  -> best |corr| = {V[best][:14]}...")

print("\n== C5: 一级A vs 一级B feedback correlation")
print(f"  corr = {corr(pd.to_numeric(g[V[0]], errors='coerce').to_numpy(float), pd.to_numeric(g[V[1]], errors='coerce').to_numpy(float)):+.3f}")

print("\n== C4: physical wiring, valve-diff -> temp-diff at lag +3 steps (30s)")
gv = {c: pd.to_numeric(g[c], errors="coerce").to_numpy(float) for c in V + T}
for v in V:
    dv = np.diff(gv[v])
    outs = []
    for t in T:
        dt_ = np.diff(gv[t])
        n = len(dv) - 3
        outs.append(corr(dv[:n], dt_[3:3 + n]))
    print(f"  d({v[:16]}...) -> " + " ".join(f"{x:+.3f}" for x in outs))
print("  temps order: 左一过 | 右一过 | 左二过 | 右二过")
