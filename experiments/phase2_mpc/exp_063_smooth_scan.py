#!/usr/bin/env python3
"""平滑方案快速对比 (新基准, 10轨迹, H_PLAN=18): hard5 vs hard2 vs inert05 vs inert025 vs ovl05_hard5"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv

M.SP_TRAJ = 0
M.DIST_AMP = 0.3
M.M_STEP = 6
M.H_PLAN = int(os.environ.get('H_PLAN', 18))
M.BENCH_SP_EACH = True
N_TRACKS = 10
OUT_DIR = os.environ.get('OUT_DIR', 'results/exp_063_smooth_scan')
os.makedirs(OUT_DIR, exist_ok=True)
# 正确组合: overlap 软参考 + hard 边界 (ovl05_hard5 = LAMBDA3=0.05 + HARD_DELTA=5)
MODES = [('hard5', 0.0), ('hard2', 0.0), ('inert05', 0.0), ('inert025', 0.0)]
OVL_CFG = [('overlap', 0.05, 5.0)]  # (FIX_MODE, LAMBDA3, HARD_DELTA)

wm = M.load_wm()
N = len(M.test_raw)
np.random.seed(42)
starts = np.random.choice(range(N - M.W - M.H_OUT - 120), N_TRACKS, replace=False)

def boundary_stats(mpc_a):
    T = len(mpc_a); M_ = M.M_STEP
    jumps = [float(np.abs(mpc_a[i] - mpc_a[i - 1]).mean()) for i in range(M_, T, M_) if i < T]
    tv = float(np.abs(np.diff(mpc_a, axis=0)).sum()) / 2.0
    return (float(np.mean(jumps)) if jumps else float('nan'),
            float(np.sum(jumps) / tv) if jumps and tv > 0 else float('nan'))

print(f"H_PLAN={M.H_PLAN} | 新基准 (每步SP) | 10轨迹")
print(f"{'FIX':>12} | {'RMSE':>7} {'IAE':>8} {'ITAE':>8} {'TV_m':>6} {'jump':>6} {'viol':>4} {'超温积分':>7}")
for mode, lam3 in MODES:
    M.FIX_MODE = mode
    M.LAMBDA3 = lam3
    M.HARD_DELTA = 0.0
    rows = []
    for s in starts:
        mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, 'grad')
        m = M.metrics(mpc_t, pid_t, tset, mpc_a, pid_a)
        jm, js = boundary_stats(mpc_a)
        rows.append({**{k: m[k] for k in ['rmse_mpc', 'iae_mpc', 'itae_mpc', 'act_tv_mpc',
                                          'viol_mpc', 'overtemp_int_mpc']}, 'jump_mean': jm})
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg['viol_mpc'] = float(np.sum([r['viol_mpc'] for r in rows]))
    print(f"{mode:>12} | {agg['rmse_mpc']:>7.3f} {agg['iae_mpc']:>8.1f} {agg['itae_mpc']:>8.0f} "
          f"{agg['act_tv_mpc']:>6.3f} {agg['jump_mean']:>6.3f} {agg['viol_mpc']:>4.0f} {agg['overtemp_int_mpc']:>7.3f}")
    json.dump({'agg': agg, 'per_track': rows}, open(f"{OUT_DIR}/{mode}.json", 'w'), indent=2)

# overlap 组合 (ovl05_hard5)
for mode, lam3, hd in OVL_CFG:
    M.FIX_MODE = mode
    M.LAMBDA3 = lam3
    M.HARD_DELTA = hd
    rows = []
    for s in starts:
        mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, 'grad')
        m = M.metrics(mpc_t, pid_t, tset, mpc_a, pid_a)
        jm, js = boundary_stats(mpc_a)
        rows.append({**{k: m[k] for k in ['rmse_mpc', 'iae_mpc', 'itae_mpc', 'act_tv_mpc',
                                          'viol_mpc', 'overtemp_int_mpc']}, 'jump_mean': jm})
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg['viol_mpc'] = float(np.sum([r['viol_mpc'] for r in rows]))
    print(f"{mode}{lam3}_h{int(hd):>2} | {agg['rmse_mpc']:>7.3f} {agg['iae_mpc']:>8.1f} {agg['itae_mpc']:>8.0f} "
          f"{agg['act_tv_mpc']:>6.3f} {agg['jump_mean']:>6.3f} {agg['viol_mpc']:>4.0f} {agg['overtemp_int_mpc']:>7.3f}")
    json.dump({'agg': agg, 'per_track': rows}, open(f"{OUT_DIR}/ovl05_hard5.json", 'w'), indent=2)
print(f"Saved: {OUT_DIR}/")
