"""Decisive valve-wiring adjudication: perturbation-response analysis.

Question: for each attemperator stage, which side's outlet temperature does
each valve physically drive?

  H_double_cross (user reading of DCS piping, two red crosses):
      stage1: A->右, B->左 ; stage2: A->左, B->右
  H_data_lagdiff (yesterday's lag-30s reading):
      stage1: A->左, B->右 ; stage2: A->右, B->左

Method 1: d(valve) -> d(att_out) correlation at lags 0..12 steps (0-120s),
          restricted to windows where the valve actually moved (|dv| > 0.5%).
Method 2: isolated events -- |dv_X| > 2% while the other three valves move
          < 0.2%; sign + magnitude of temp response at best lag.
"""
import numpy as np
import pandas as pd

ALL = r"C:\Users\14020\Desktop\时间预测模型\AA数据中心\伊敏12.10\merged_all_data\all_merged_10s.csv"

V1A = "过热器一级减温器A侧喷水调节门阀位反馈"
V1B = "过热器一级减温器B侧喷水调节门阀位反馈"
V2A = "过热器二级减温器A侧喷水调节门阀位反馈"
V2B = "过热器二级减温器B侧喷水调节门阀位反馈"
T1L = "选择后左侧一过喷水减温器出口"
T1R = "选择后右侧一过喷水减温器出口"
T2L = "选择后左侧二过喷水减温器出口"
T2R = "选择后右侧二过喷水减温器出口"

g = pd.read_csv(ALL, usecols=[V1A, V1B, V2A, V2B, T1L, T1R, T2L, T2R])
for c in g.columns:
    g[c] = pd.to_numeric(g[c], errors="coerce")
g = g.interpolate(limit_direction="both")


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.mean() < 0.5:
        return np.nan
    sx, sy = x[m].std(), y[m].std()
    if sx < 1e-9 or sy < 1e-9:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


print("== Method 1: d(valve)->d(temp) by lag (only steps where |dv|>0.5) ==")
for stage, (va, vb, tl, tr) in {"一级": (V1A, V1B, T1L, T1R),
                                "二级": (V2A, V2B, T2L, T2R)}.items():
    for vname, vcol in (("A阀", va), ("B阀", vb)):
        dv = np.diff(g[vcol].to_numpy(float))
        moved = np.abs(dv) > 0.5
        row_l, row_r = [], []
        for lag in range(0, 13):
            n = len(dv) - lag
            sel = moved[:n]
            dl = np.diff(g[tl].to_numpy(float))
            dr = np.diff(g[tr].to_numpy(float))
            row_l.append(corr(dv[:n][sel], dl[lag:lag + n][sel]))
            row_r.append(corr(dv[:n][sel], dr[lag:lag + n][sel]))
        bl = int(np.nanargmin(row_l)); br = int(np.nanargmin(row_r))
        print(f"  {stage}{vname}: 左温 best lag={bl*10:>3}s r={row_l[bl]:+.3f} | "
              f"右温 best lag={br*10:>3}s r={row_r[br]:+.3f} | "
              f"lag剖面 L={'/'.join(f'{x:+.2f}' for x in row_l[:7])}")
        print(f"           lag剖面 R={'/'.join(f'{x:+.2f}' for x in row_r[:7])}")

print("\n== Method 2: isolated single-valve events (|dv|>2, others<0.2) ==")
for stage, (va, vb, tl, tr) in {"一级": (V1A, V1B, T1L, T1R),
                                "二级": (V2A, V2B, T2L, T2R)}.items():
    allv = [np.diff(g[c].to_numpy(float)) for c in (V1A, V1B, V2A, V2B)]
    for vname, vcol in (("A阀", va), ("B阀", vb)):
        dv = np.diff(g[vcol].to_numpy(float))
        others = [o for o, c in zip(allv, (V1A, V1B, V2A, V2B)) if c != vcol]
        iso = (np.abs(dv) > 2.0)
        for o in others:
            iso &= (np.abs(o) < 0.2)
        n = len(dv) - 6
        dl = np.diff(g[tl].to_numpy(float)); dr = np.diff(g[tr].to_numpy(float))
        rl = corr(dv[:n][iso[:n]], dl[3:3 + n][iso[:n]])   # 30s lag
        rr = corr(dv[:n][iso[:n]], dr[3:3 + n][iso[:n]])
        print(f"  {stage}{vname}: n_events={int(iso[:n].sum()):>6}  "
              f"左温响应 r={rl:+.3f}  右温响应 r={rr:+.3f}")
