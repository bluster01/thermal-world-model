#!/usr/bin/env python3
"""
exp_053_hplan_scan.py — MPC 规划视野 H_PLAN 扫描
====================================================
回答: MPC 用什么规划视野最优? (exp_048 回答 WM 训练预测长度 H_OUT; 本实验回答 MPC 的 H_PLAN)

设计: H_PLAN ∈ {6, 10, 14, 18} × FIX_MODE ∈ {none, hard5}
  - M_STEP=6 (60s执行块), H_OUT=18, DIST_AMP=0.3 (扰动主协议), grad, α=0.5, SP_TRAJ=0
  - 10 轨迹 seed42, 1200s 闭环
  - hard5 = 当前推荐默认 (边界免费修复); none 作对照 (exp_050 已有 H=10/18 锚点)
指标: RMSE/std/TV/jump/viol (与 exp_051/052 同口径)
用法: python exp_053_hplan_scan.py [--smoke]
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv

SMOKE = '--smoke' in sys.argv
M.SP_TRAJ = 0
M.DIST_AMP = 0.3
M.M_STEP = 6
N_TRACKS = 2 if SMOKE else 10
OUT_DIR = os.environ.get('OUT_DIR', 'results/exp_053_hplan')
os.makedirs(OUT_DIR, exist_ok=True)

wm = M.load_wm()

def boundary_stats(mpc_a):
    """从实际执行流算重规划边界跳变 (与 exp_051/052 同口径)"""
    T = len(mpc_a)
    M_ = M.M_STEP
    jumps = []
    for b in range(1, T // M_):
        i = b * M_
        if i >= T:
            break
        jumps.append(float(np.abs(mpc_a[i] - mpc_a[i - 1]).mean()))
    total_var = float(np.abs(np.diff(mpc_a, axis=0)).sum()) / 2.0
    jumps = np.array(jumps)
    return {
        'jump_mean': float(jumps.mean()) if len(jumps) else float('nan'),
        'jump_max': float(jumps.max()) if len(jumps) else float('nan'),
        'jump_share': float(jumps.sum() / total_var) if len(jumps) and total_var > 0 else float('nan'),
        'n_boundaries': int(len(jumps)),
    }

CONFIGS = []
for hp in [6, 10, 14, 18]:
    for fix in ['none', 'hard5']:
        CONFIGS.append((hp, fix))

results = {}
t0 = time.time()
print(f"{'H_PLAN':>7} {'FIX':>6} | {'RMSE_m':>7} {'RMSE_p':>7} {'std_m':>6} {'TV_m':>6} "
      f"{'jump_m':>7} {'jump_max':>8} {'jump%TV':>7} {'viol':>4}")
for hp, fix in CONFIGS:
    M.H_PLAN = hp
    M.FIX_MODE = fix
    M.HARD_DELTA = 0.0
    tc = time.time()
    N = len(M.test_raw)
    np.random.seed(42)
    starts = np.random.choice(range(N - M.W - M.H_OUT - 120), N_TRACKS, replace=False)
    rows = []
    for s in starts:
        mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, 'grad')
        m = M.metrics(mpc_t, pid_t, tset, mpc_a, pid_a)
        bs = boundary_stats(mpc_a)
        row = {'rmse_mpc': m['rmse_mpc'], 'rmse_pid': m['rmse_pid'],
               'std_mpc': m['temp_std_mpc'], 'tv_mpc': m['act_tv_mpc'],
               'tv_pid': m['act_tv_pid'], 'viol_mpc': m['viol_mpc'], **bs}
        rows.append(row)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg['jump_max'] = float(np.max([r['jump_max'] for r in rows]))
    agg['viol_mpc'] = float(np.sum([r['viol_mpc'] for r in rows]))
    results[f'h{hp}_{fix}'] = {'agg': agg, 'per_track': rows}
    print(f"{hp:>7} {fix:>6} | {agg['rmse_mpc']:>7.3f} {agg['rmse_pid']:>7.3f} {agg['std_mpc']:>6.3f} "
          f"{agg['tv_mpc']:>6.3f} {agg['jump_mean']:>7.3f} {agg['jump_max']:>8.3f} "
          f"{agg['jump_share']*100:>6.1f}% {agg['viol_mpc']:>4.0f}  ({(time.time()-tc)/60:.1f}min)")
    json.dump({'agg': agg, 'per_track': rows}, open(f"{OUT_DIR}/h{hp}_{fix}.json", 'w'), indent=2)

print(f"\n===== H_PLAN 扫描完成 ({(time.time()-t0)/60:.1f}min) =====")
print(f"Saved: {OUT_DIR}/")
