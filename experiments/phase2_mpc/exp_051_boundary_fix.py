#!/usr/bin/env python3
"""
exp_051_boundary_fix.py — 边界跳变修复方案对比
================================================
对 exp_050 确认的抖动主根源 (重规划边界跳变占 TV 28-46%) 测试三种修复:
  none    : 基线 (现状, 整段继承 warm-start)
  hard2/5 : 优化器内硬约束 a[0] ∈ a_last±δ (rate constraint, 保留约束最优性)
  blend   : 加权融合 — 执行块 = w_j·旧计划未执行段 + (1−w_j)·新计划, w 线性 1→0
            (连续性由构造保证: 旧计划内部 |Δa|≤5)
  inert05/025: 惯性块 — 一阶惯性作用于执行流, K=0.5/0.25

协议: 与 exp_050 一致 — M_STEP=6, H_PLAN=18, DIST_AMP=0.3, SP_TRAJ=0,
      grad, α=0.5, 10 轨迹 seed42。闭环一致性: blend/inert 执行≠计划时
      用实际执行动作重算 WM 温度 (exp_027 FIX_MODE 内置)。
指标: RMSE/σ/TV + 边界跳变 (从实际执行流计算): jump_mean/max/share
      share = Σ边界跳变 / 总动作变差 (占 TV 比例)
用法: python exp_051_boundary_fix.py [--smoke]
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
M.H_PLAN = int(os.environ.get('H_PLAN', 18))
N_TRACKS = 2 if SMOKE else 10
OUT_DIR = 'results/exp_051_boundary_fix'
os.makedirs(OUT_DIR, exist_ok=True)

wm = M.load_wm()
MODES = os.environ.get('MODES', 'none,hard2,hard5,blend,inert05,inert025').split(',')

def boundary_stats(mpc_a):
    """从实际执行流算重规划边界跳变 (块起点 vs 上一块终点)"""
    T = len(mpc_a)
    M_ = M.M_STEP
    jumps = []
    for b in range(1, T // M_):
        i = b * M_
        if i >= T:
            break
        jumps.append(float(np.abs(mpc_a[i] - mpc_a[i - 1]).mean()))
    total_var = float(np.abs(np.diff(mpc_a, axis=0)).sum())
    jumps = np.array(jumps)
    return {
        'jump_mean': float(jumps.mean()) if len(jumps) else float('nan'),
        'jump_max': float(jumps.max()) if len(jumps) else float('nan'),
        'jump_share': float(jumps.sum() / total_var) if len(jumps) and total_var > 0 else float('nan'),
        'n_boundaries': int(len(jumps)),
    }

results = {}
t0 = time.time()
print(f"{'FIX':>8} | {'RMSE_m':>7} {'RMSE_p':>7} {'std_m':>6} {'TV_m':>6} {'TV_p':>6} "
      f"{'jump_m':>7} {'jump_max':>8} {'jump%TV':>7}")
for mode in MODES:
    M.FIX_MODE = mode
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
               'tv_pid': m['act_tv_pid'], **bs}
        rows.append(row)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg['jump_max'] = float(np.max([r['jump_max'] for r in rows]))
    results[mode] = {'agg': agg, 'per_track': rows}
    print(f"{mode:>8} | {agg['rmse_mpc']:>7.3f} {agg['rmse_pid']:>7.3f} {agg['std_mpc']:>6.3f} "
          f"{agg['tv_mpc']:>6.3f} {agg['tv_pid']:>6.3f} {agg['jump_mean']:>7.3f} "
          f"{agg['jump_max']:>8.3f} {agg['jump_share']*100:>6.1f}%  ({(time.time()-tc)/60:.1f}min)")
    json.dump({'agg': agg, 'per_track': rows}, open(f"{OUT_DIR}/{mode}.json", 'w'), indent=2)

print(f"\n===== 边界跳变修复对比完成 ({(time.time()-t0)/60:.1f}min) =====")
print(f"Saved: {OUT_DIR}/")
