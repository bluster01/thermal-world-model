#!/usr/bin/env python3
"""找 SP 跳变轨迹, A/B 验证评测基准修正 (每步SP vs 块起点SP)"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv

M.SP_TRAJ = 0
M.DIST_AMP = 0.3
M.M_STEP = 6
M.H_PLAN = 10
M.FIX_MODE = 'hard5'
M.HARD_DELTA = 0.0

N = len(M.test_raw)
W = M.W
# 找 SP 跳变大的候选起点: 窗口内 SP 最大跳变
np.random.seed(42)
cands = np.random.choice(range(N - W - M.H_OUT - 120), 200, replace=False)
best = []
for s in cands:
    sp_win = M.test_raw[s + W:s + W + 120, M.SP_IDX]
    d = np.abs(np.diff(sp_win)).max()
    best.append((d, s))
best.sort(reverse=True)
print("SP 最大单步跳变 top5 候选:", [(round(d, 2), s) for d, s in best[:5]])

wm = M.load_wm()
print(f"{'起点':>7} | {'旧基准RMSE':>10} {'新基准RMSE':>10} {'Δ':>8} | {'旧IAE':>7} {'新IAE':>7}")
for d, s in best[:5]:
    M.BENCH_SP_EACH = False
    mpc_t, pid_t, tset_old, mpc_a, pid_a = M.simulate(wm, s, 'grad')
    m_old = M.metrics(mpc_t, pid_t, tset_old, mpc_a, pid_a)
    M.BENCH_SP_EACH = True
    mpc_t, pid_t, tset_new, mpc_a, pid_a = M.simulate(wm, s, 'grad')
    m_new = M.metrics(mpc_t, pid_t, tset_new, mpc_a, pid_a)
    print(f"{s:>7} | {m_old['rmse_mpc']:>10.3f} {m_new['rmse_mpc']:>10.3f} "
          f"{m_new['rmse_mpc']-m_old['rmse_mpc']:>+8.3f} | {m_old['iae_mpc']:>7.1f} {m_new['iae_mpc']:>7.1f}")
