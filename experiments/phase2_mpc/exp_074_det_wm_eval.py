#!/usr/bin/env python3
"""
exp_074_det_wm_eval.py — P4: 确定性 WM (M5) 闭环评测
=====================================================
同协议: H_PLAN=18, ovl05_hard5, M_STEP=6, 150轨迹×3起点, 扰动+无扰动, 每步SP基准
对照: M7(概率) vs M5(确定性,同架构MSE训练) — 不确定性机制的价值
用法: python exp_074_det_wm_eval.py [--smoke]
"""
import os, sys, json, time
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMOKE = '--smoke' in sys.argv
OUT_DIR = os.environ.get('OUT_DIR', 'results/exp_074_det_wm')
os.makedirs(OUT_DIR, exist_ok=True)

def load_m5():
    from experiments.phase1_dynamics.exp_025_unified_benchmark import build_model
    model = build_model('M5').to(DEVICE).eval()
    ck = torch.load('results/exp_025_M5/checkpoints/best_model.pth',
                    map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict'])
    return model

# ============ 评测 (同 exp_064 协议) ============
M.SP_TRAJ = 0; M.M_STEP = 6; M.H_PLAN = 18
M.FIX_MODE = 'overlap'; M.LAMBDA3 = 0.05; M.HARD_DELTA = 5.0
M.BENCH_SP_EACH = True
M.RISK_LAMBDA = 0.0
N_TRACKS = 2 if SMOKE else 50
SEEDS = [42] if SMOKE else [42, 7, 13]
KEYS = ['rmse_mpc', 'rmse_pid', 'iae_mpc', 'iae_pid', 'itae_mpc', 'itae_pid',
        'act_tv_mpc', 'act_tv_pid', 'viol_mpc', 'overtemp_int_mpc', 'overtemp_int_pid',
        'overtemp_mpc', 'overtemp_pid', 'jump_mean', 'jump_share']

def run(wm, scene, dist_amp):
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
            rows.append({k: m[k] for k in KEYS})
    json.dump(rows, open(f'{OUT_DIR}/{scene}.json', 'w'), indent=2)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    agg['viol_mpc'] = float(np.sum([r['viol_mpc'] for r in rows]))
    agg['overtemp_mpc'] = float(np.sum([r['overtemp_mpc'] for r in rows]))
    p = stats.wilcoxon([r['rmse_mpc'] for r in rows], [r['rmse_pid'] for r in rows]).pvalue
    print(f"=== M5确定性WM {scene} (n={len(rows)}) ===")
    print(f"  M5: RMSE {agg['rmse_mpc']:.3f} | IAE {agg['iae_mpc']:.1f} | TV {agg['act_tv_mpc']:.3f} "
          f"| jump {agg['jump_mean']:.3f} | viol {agg['viol_mpc']:.0f} | 超温 {agg['overtemp_mpc']:.0f}s | p={p:.2e}")
    return agg

if __name__ == '__main__':
    t0 = time.time()
    wm5 = load_m5()
    a_dist = run(wm5, 'dist', 0.3)
    a_nod = run(wm5, 'nodist', 0.0)
    print(f"\n===== 确定性WM评测完成 ({(time.time()-t0)/60:.1f}min) =====")
    print(f"Saved: {OUT_DIR}/")
