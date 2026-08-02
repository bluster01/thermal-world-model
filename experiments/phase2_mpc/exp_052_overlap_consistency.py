#!/usr/bin/env python3
"""
exp_052_overlap_consistency.py — 重叠一致性 (用户想法: 利用预测后半程平滑切换)
================================================================================
exp_051 的 blend (执行旧尾部) 已证明陈旧信息代价高 (+31% RMSE)。
本实验实现改良版: 旧计划未执行段作为**软轨迹参考**进入新计划的优化目标:
    J += LAMBDA3 · Σ_{j<M_STEP} |a_new[j] − a_old[M_STEP+j]|²
执行仍用新计划 (新鲜状态), 平滑性由目标函数构造保证, 不牺牲新信息。

配置: M_STEP=6, H_PLAN=18, DIST_AMP=0.3, grad, α=0.5, SP_TRAJ=0, 10 轨迹 seed42
  none          : 基线
  hard5         : exp_051 免费修复 (边界限幅 δ=5)
  ovl01 / ovl05 : 重叠一致性 LAMBDA3=0.1 / 0.5
  ovl05_hard5   : 组合 (重叠 + 边界限幅)
用法: python exp_052_overlap_consistency.py [--smoke]
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
M.H_PLAN = 18
N_TRACKS = 2 if SMOKE else 10
OUT_DIR = 'results/exp_052_overlap'
os.makedirs(OUT_DIR, exist_ok=True)

wm = M.load_wm()
CONFIGS = [
    ('none', {}),
    ('hard5', {'FIX_MODE': 'hard5'}),
    ('ovl01', {'FIX_MODE': 'overlap', 'LAMBDA3': 0.1}),
    ('ovl05', {'FIX_MODE': 'overlap', 'LAMBDA3': 0.5}),
    ('ovl05_hard5', {'FIX_MODE': 'overlap', 'LAMBDA3': 0.5, 'HARD_DELTA': 5.0}),
]

def boundary_stats(mpc_a):
    """从实际执行流算重规划边界跳变 (与 exp_051 同口径)"""
    T = len(mpc_a)
    M_ = M.M_STEP
    jumps = []
    for b in range(1, T // M_):
        i = b * M_
        if i >= T:
            break
        jumps.append(float(np.abs(mpc_a[i] - mpc_a[i - 1]).mean()))
    total_var = float(np.abs(np.diff(mpc_a, axis=0)).sum()) / 2.0  # per-valve-mean 口径
    jumps = np.array(jumps)
    return {
        'jump_mean': float(jumps.mean()) if len(jumps) else float('nan'),
        'jump_max': float(jumps.max()) if len(jumps) else float('nan'),
        'jump_share': float(jumps.sum() / total_var) if len(jumps) and total_var > 0 else float('nan'),
        'n_boundaries': int(len(jumps)),
    }

results = {}
t0 = time.time()
print(f"{'CFG':>12} | {'RMSE_m':>7} {'RMSE_p':>7} {'std_m':>6} {'TV_m':>6} {'TV_p':>6} "
      f"{'jump_m':>7} {'jump_max':>8} {'jump%TV':>7} {'viol':>4}")
for name, over in CONFIGS:
    M.FIX_MODE = over.get('FIX_MODE', 'none')
    M.LAMBDA3 = over.get('LAMBDA3', 0.1)
    M.HARD_DELTA = over.get('HARD_DELTA', 0.0)
    M.OVERLAP_REF = None
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
    results[name] = {'agg': agg, 'per_track': rows, 'fix': str(over)}
    print(f"{name:>12} | {agg['rmse_mpc']:>7.3f} {agg['rmse_pid']:>7.3f} {agg['std_mpc']:>6.3f} "
          f"{agg['tv_mpc']:>6.3f} {agg['tv_pid']:>6.3f} {agg['jump_mean']:>7.3f} "
          f"{agg['jump_max']:>8.3f} {agg['jump_share']*100:>6.1f}% {agg['viol_mpc']:>4.0f}  ({(time.time()-tc)/60:.1f}min)")
    json.dump({'agg': agg, 'per_track': rows, 'fix': str(over)},
              open(f"{OUT_DIR}/{name}.json", 'w'), indent=2)

print(f"\n===== 重叠一致性对比完成 ({(time.time()-t0)/60:.1f}min) =====")
print(f"Saved: {OUT_DIR}/")
