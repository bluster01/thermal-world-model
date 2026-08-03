#!/usr/bin/env python3
"""
exp_064_final_smooth.py — 平滑方案定稿评测 (论文主表)
========================================================
配置: H_PLAN=18, M_STEP=6, 新基准 (每步SP), 3起点集×50轨迹, 双场景 (无扰动/扰动)
对比: MPC-ovl05_hard5 vs MPC-inert05 vs PID (同扰动序列)
指标: RMSE/IAE/ITAE/TV/jump/viol/超温积分/超温时间 + Wilcoxon (150轨迹配对)
用法: python exp_064_final_smooth.py [--smoke]
"""
import os, sys, json, time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv

SMOKE = '--smoke' in sys.argv
M.SP_TRAJ = 0
M.DIST_AMP = 0.3
M.M_STEP = 6
M.H_PLAN = 18
M.BENCH_SP_EACH = True
N_TRACKS = 2 if SMOKE else 50
SEEDS = [42] if SMOKE else [42, 7, 13]
OUT_DIR = os.environ.get('OUT_DIR', 'results/exp_064_final_smooth')
os.makedirs(OUT_DIR, exist_ok=True)
CONFIGS = [('ovl05_hard5', 'overlap', 0.05, 5.0), ('inert05', 'inert05', 0.0, 0.0)]

wm = M.load_wm()
KEYS = ['rmse_mpc', 'rmse_pid', 'iae_mpc', 'iae_pid', 'itae_mpc', 'itae_pid',
        'act_tv_mpc', 'act_tv_pid', 'viol_mpc', 'overtemp_int_mpc', 'overtemp_int_pid',
        'overtemp_mpc', 'overtemp_pid']

def run(name, dist_amp):
    M.DIST_AMP = dist_amp
    rows = []
    for seed in SEEDS:
        np.random.seed(seed)
        starts = np.random.choice(range(len(M.test_raw) - M.W - M.H_OUT - 120),
                                  N_TRACKS, replace=False)
        for s in starts:
            mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, 'grad')
            m = M.metrics(mpc_t, pid_t, tset, mpc_a, pid_a)
            T = len(mpc_a); M_ = M.M_STEP
            jumps = [float(np.abs(mpc_a[i] - mpc_a[i - 1]).mean()) for i in range(M_, T, M_) if i < T]
            tv = float(np.abs(np.diff(mpc_a, axis=0)).sum()) / 2.0
            m['jump_mean'] = float(np.mean(jumps)) if jumps else float('nan')
            m['jump_share'] = float(np.sum(jumps) / tv) if jumps and tv > 0 else float('nan')
            rows.append({k: m[k] for k in KEYS + ['jump_mean', 'jump_share']})
    json.dump(rows, open(f"{OUT_DIR}/{name}.json", 'w'), indent=2)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    agg['jump_mean'] = float(np.mean([r['jump_mean'] for r in rows]))
    agg['jump_share'] = float(np.mean([r['jump_share'] for r in rows]))
    agg['viol_mpc'] = float(np.sum([r['viol_mpc'] for r in rows]))
    agg['overtemp_mpc'] = float(np.sum([r['overtemp_mpc'] for r in rows]))
    agg['overtemp_pid'] = float(np.sum([r['overtemp_pid'] for r in rows]))
    p = stats.wilcoxon([r['rmse_mpc'] for r in rows], [r['rmse_pid'] for r in rows]).pvalue
    print(f"=== {name} (n={len(rows)}) ===")
    print(f"  MPC: RMSE {agg['rmse_mpc']:.3f} | IAE {agg['iae_mpc']:.1f} | ITAE {agg['itae_mpc']:.0f} "
          f"| TV {agg['act_tv_mpc']:.3f} | jump {agg['jump_mean']:.3f} ({agg['jump_share']*100:.1f}%) "
          f"| viol {agg['viol_mpc']:.0f} | 超温积分 {agg['overtemp_int_mpc']:.2f} | 超温 {agg['overtemp_mpc']:.0f}s")
    print(f"  PID: RMSE {agg['rmse_pid']:.3f} | IAE {agg['iae_pid']:.1f} | ITAE {agg['itae_pid']:.0f} "
          f"| TV {agg['act_tv_pid']:.3f} | 超温积分 {agg['overtemp_int_pid']:.2f} | 超温 {agg['overtemp_pid']:.0f}s")
    print(f"  Wilcoxon (MPC vs PID RMSE): p={p:.2e}")
    return agg

t0 = time.time()
for name, fix, lam3, hd in CONFIGS:
    M.FIX_MODE = fix
    M.LAMBDA3 = lam3
    M.HARD_DELTA = hd
    run(f"{name}_dist", 0.3)
    run(f"{name}_nodist", 0.0)
print(f"\n===== 平滑定稿完成 ({(time.time()-t0)/60:.1f}min) =====")
print(f"Saved: {OUT_DIR}/")
