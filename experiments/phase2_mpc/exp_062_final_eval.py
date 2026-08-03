#!/usr/bin/env python3
"""
exp_062_final_eval.py — 主协议定稿评测 (论文主表)
====================================================
配置: M_STEP=6, H_PLAN=10, FIX_MODE=hard5, DIST_AMP=0.3, SP_TRAJ=0, grad
基准: 每步真实 SP (BENCH_SP_EACH=True, 2026-08-03 修正)
协议: 3 起点集 (seed 42/7/13) × 50 轨迹, 双场景 (无扰动 / 扰动世界)
对比: MPC(hard5) vs PID (同扰动序列)
指标: RMSE/IAE/ITAE/TV/jump/viol/超温积分/超温时间 (per-track + 3-seed 均值±std + Wilcoxon)
用法: python exp_062_final_eval.py [--smoke]
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
M.H_PLAN = 10
M.FIX_MODE = 'hard5'
M.HARD_DELTA = 0.0
M.BENCH_SP_EACH = True
N_TRACKS = 2 if SMOKE else 50
SEEDS = [42] if SMOKE else [42, 7, 13]
OUT_DIR = os.environ.get('OUT_DIR', 'results/exp_062_final_eval')
os.makedirs(OUT_DIR, exist_ok=True)

wm = M.load_wm()

KEYS = ['rmse_mpc', 'rmse_pid', 'iae_mpc', 'iae_pid', 'itae_mpc', 'itae_pid',
        'act_tv_mpc', 'act_tv_pid', 'viol_mpc', 'overtemp_int_mpc', 'overtemp_int_pid',
        'overtemp_mpc', 'overtemp_pid']

def run_scene(scene, dist_amp):
    """返回 per-track 指标列表 [{...}] 3-seed 合并"""
    M.DIST_AMP = dist_amp
    rows = []
    for seed in SEEDS:
        np.random.seed(seed)
        starts = np.random.choice(range(len(M.test_raw) - M.W - M.H_OUT - 120),
                                  N_TRACKS, replace=False)
        for s in starts:
            mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, 'grad')
            m = M.metrics(mpc_t, pid_t, tset, mpc_a, pid_a)
            T = len(mpc_a)
            M_ = M.M_STEP
            jumps = [float(np.abs(mpc_a[i] - mpc_a[i - 1]).mean())
                     for i in range(M_, T, M_) if i < T]
            tv = float(np.abs(np.diff(mpc_a, axis=0)).sum()) / 2.0
            m['jump_mean'] = float(np.mean(jumps)) if jumps else float('nan')
            m['jump_share'] = float(np.sum(jumps) / tv) if jumps and tv > 0 else float('nan')
            m['scene'] = scene
            rows.append({k: m[k] for k in KEYS + ['jump_mean', 'jump_share']})
    return rows

t0 = time.time()
out = {}
for scene, dist in [('nodist', 0.0), ('dist', 0.3)]:
    rows = run_scene(scene, dist)
    out[scene] = rows
    agg = {}
    for k in KEYS:
        agg[k] = float(np.mean([r[k] for r in rows]))
    agg['jump_mean'] = float(np.mean([r['jump_mean'] for r in rows]))
    agg['jump_share'] = float(np.mean([r['jump_share'] for r in rows]))
    agg['viol_mpc'] = float(np.sum([r['viol_mpc'] for r in rows]))
    agg['overtemp_mpc'] = float(np.sum([r['overtemp_mpc'] for r in rows]))
    agg['overtemp_pid'] = float(np.sum([r['overtemp_pid'] for r in rows]))
    n = len(rows)
    print(f"=== {scene} (n={n}) ===")
    print(f"  MPC: RMSE {agg['rmse_mpc']:.3f} | IAE {agg['iae_mpc']:.1f} | ITAE {agg['itae_mpc']:.0f} "
          f"| TV {agg['act_tv_mpc']:.3f} | jump {agg['jump_mean']:.3f} ({agg['jump_share']*100:.1f}%) "
          f"| viol {agg['viol_mpc']:.0f} | 超温积分 {agg['overtemp_int_mpc']:.2f} | 超温 {agg['overtemp_mpc']:.0f}s")
    print(f"  PID: RMSE {agg['rmse_pid']:.3f} | IAE {agg['iae_pid']:.1f} | ITAE {agg['itae_pid']:.0f} "
          f"| TV {agg['act_tv_pid']:.3f} | 超温积分 {agg['overtemp_int_pid']:.2f} | 超温 {agg['overtemp_pid']:.0f}s")
    p = stats.wilcoxon([r['rmse_mpc'] for r in rows], [r['rmse_pid'] for r in rows]).pvalue
    print(f"  Wilcoxon (MPC vs PID, RMSE): p={p:.2e}")
    json.dump(rows, open(f"{OUT_DIR}/{scene}.json", 'w'), indent=2)

print(f"\n===== 主协议定稿评测完成 ({(time.time()-t0)/60:.1f}min) =====")
print(f"Saved: {OUT_DIR}/")
