#!/usr/bin/env python3
"""exp_046_stats.py — 统计显著性 (Phase 2.5 任务3)
paired Wilcoxon signed-rank on 50 tracks (公平协议 exp_041 per_track)
"""
import json, os, sys
import numpy as np
from scipy import stats

d = json.load(open("results/exp_041_fair_H10.json"))
rows = d['per_track']
print(f"n = {len(rows)} 轨迹 (公平协议: 两组均 WM 闭环)")

# 每轨迹的 RMSE (vs SP) — per_track 里是 rmse_mpc_vs_sp / rmse_pidwm_vs_sp
rm_mpc = np.array([r['rmse_mpc_vs_sp'] for r in rows])
rm_pid = np.array([r['rmse_pidwm_vs_sp'] for r in rows])
std_mpc = np.array([r['std_mpc'] for r in rows])
std_pid = np.array([r['std_pidwm'] for r in rows])

def report(name, x, y, better='lower'):
    diff = x - y
    w, p = stats.wilcoxon(diff)
    n_win = int(np.sum((x < y) if better == 'lower' else (x > y)))
    med = np.median(diff)
    print(f"{name}: 中位差 {med:+.4f} | 胜 {n_win}/{len(x)} | W={w:.0f} p={p:.2e} {'✅显著' if p < 0.05 else '❌不显著'}")

print("\n===== paired Wilcoxon (DWM-MPC vs PID-WM, 50 条) =====")
report("RMSE vs SP", rm_mpc, rm_pid)
report("温度 std", std_mpc, std_pid)

# 效应量 (Cohen's d)
def cohens_d(x, y):
    d_ = x - y
    return d_.mean() / d_.std(ddof=1)
print(f"\n效应量 Cohen's d (RMSE): {cohens_d(rm_mpc, rm_pid):.3f} (大效应 >0.8)")
print(f"效应量 Cohen's d (std):  {cohens_d(std_mpc, std_pid):.3f}")

# 中位数 + IQR 报告
print("\n中位数 (IQR):")
for nm, x, y in [('RMSE', rm_mpc, rm_pid), ('std', std_mpc, std_pid)]:
    print(f"  {nm}: MPC {np.median(x):.3f} [{np.percentile(x,25):.3f},{np.percentile(x,75):.3f}] | "
          f"PID-WM {np.median(y):.3f} [{np.percentile(y,25):.3f},{np.percentile(y,75):.3f}]")

out = {'n': len(rows),
       'wilcoxon_rmse': {'W': float(stats.wilcoxon(rm_mpc - rm_pid).statistic),
                          'p': float(stats.wilcoxon(rm_mpc - rm_pid).pvalue)},
       'wilcoxon_std': {'W': float(stats.wilcoxon(std_mpc - std_pid).statistic),
                         'p': float(stats.wilcoxon(std_mpc - std_pid).pvalue)},
       'cohens_d_rmse': float(cohens_d(rm_mpc, rm_pid)),
       'cohens_d_std': float(cohens_d(std_mpc, std_pid))}
os.makedirs("results/exp_046_stats", exist_ok=True)
json.dump(out, open("results/exp_046_stats/wilcoxon.json", 'w'), indent=2)
print("\nSaved: results/exp_046_stats/wilcoxon.json")
