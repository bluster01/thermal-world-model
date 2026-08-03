#!/usr/bin/env python3
"""
exp_059_hplan_full.py — MPC 规划视野 H_PLAN 全扫 1-18 步
============================================================
用户需求: MPC 的规划 horizon 测试 1-18 步全部 (exp_053 只测了 6/10/14/18)
协议: M_STEP=6, H_OUT=18, DIST_AMP=0.3, grad, α=0.5, SP_TRAJ=0, FIX_MODE=hard5
      10 轨迹 seed42 (与 exp_053 同协议, 可直接衔接)
指标: RMSE (主指标保留) + IAE/ITAE/TV/jump/viol/超温积分 (论文补充指标)
注: H_PLAN < M_STEP 时 n_exec=H_PLAN (计划耗尽即重规划), 执行粒度=H_PLAN
用法: python exp_059_hplan_full.py [--smoke]
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
M.FIX_MODE = 'hard5'
M.HARD_DELTA = 0.0
N_TRACKS = 2 if SMOKE else 10
OUT_DIR = os.environ.get('OUT_DIR', 'results/exp_059_hplan_full')
os.makedirs(OUT_DIR, exist_ok=True)
H_LIST = [1, 3] if SMOKE else list(range(1, 19))

wm = M.load_wm()

def boundary_stats(mpc_a):
    T = len(mpc_a)
    M_ = min(M.M_STEP, M.H_PLAN)  # 实际执行粒度 (H_PLAN<M_STEP 时计划耗尽即重规划)
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

results = {}
t0 = time.time()
print(f"{'H_PLAN':>6} | {'RMSE':>7} {'IAE':>8} {'ITAE':>8} {'TV_m':>6} {'jump':>6} "
      f"{'jump%TV':>7} {'viol':>4} {'超温积分':>7} {'超温s':>5}")
for hp in H_LIST:
    M.H_PLAN = hp
    tc = time.time()
    N = len(M.test_raw)
    np.random.seed(42)
    starts = np.random.choice(range(N - M.W - M.H_OUT - 120), N_TRACKS, replace=False)
    rows = []
    for s in starts:
        mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, 'grad')
        m = M.metrics(mpc_t, pid_t, tset, mpc_a, pid_a)
        bs = boundary_stats(mpc_a)
        row = {k: m[k] for k in ['rmse_mpc', 'iae_mpc', 'itae_mpc', 'act_tv_mpc',
                                 'viol_mpc', 'overtemp_int_mpc', 'overtemp_mpc']}
        row.update(bs)
        rows.append(row)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg['jump_max'] = float(np.max([r['jump_max'] for r in rows]))
    agg['viol_mpc'] = float(np.sum([r['viol_mpc'] for r in rows]))
    agg['overtemp_mpc'] = float(np.sum([r['overtemp_mpc'] for r in rows]))
    results[str(hp)] = {'agg': agg, 'per_track': rows}
    print(f"{hp:>6} | {agg['rmse_mpc']:>7.3f} {agg['iae_mpc']:>8.1f} {agg['itae_mpc']:>8.0f} "
          f"{agg['act_tv_mpc']:>6.3f} {agg['jump_mean']:>6.3f} {agg['jump_share']*100:>6.1f}% "
          f"{agg['viol_mpc']:>4.0f} {agg['overtemp_int_mpc']:>7.3f} {agg['overtemp_mpc']:>5.0f}  ({(time.time()-tc)/60:.1f}min)")
    json.dump({'agg': agg, 'per_track': rows}, open(f"{OUT_DIR}/h{hp}.json", 'w'), indent=2)

print(f"\n===== H_PLAN 全扫完成 ({(time.time()-t0)/60:.1f}min) =====")
print(f"Saved: {OUT_DIR}/")
