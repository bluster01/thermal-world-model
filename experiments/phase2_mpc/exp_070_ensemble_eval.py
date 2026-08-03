#!/usr/bin/env python3
"""
exp_070_ensemble_eval.py — P2: PETS式 (集成+CEM) 闭环评测
===========================================================
EnsembleWM: 3×M7 (h18 + seed7/13), μ_ens=mean(μ_i), σ_ens²=mean(σ_i²)+var(μ_i) (aleatoric+epistemic, PETS标准)
评测矩阵: ①集成+CEM (PETS式) ②集成+grad (分离规划器因素) ③对照 M7+grad (exp_064已有)
协议: H_PLAN=18, ovl05_hard5, M_STEP=6, 3起点×50轨迹, 扰动+无扰动, 每步SP基准
用法: python exp_070_ensemble_eval.py [--smoke]
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
from experiments.phase1_dynamics import exp_025_unified_benchmark as E

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMOKE = '--smoke' in sys.argv
OUT_DIR = 'results/exp_070_ensemble'
os.makedirs(OUT_DIR, exist_ok=True)
CK_DIR = 'results/exp_069_ensemble/checkpoints'

def load_model(path):
    E.H_OUT = 18
    model = E.build_model('M7').to(DEVICE)
    ck = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    return model

class EnsembleWM:
    """3×M7 概率集成: μ_ens=mean(μ), σ_ens²=mean(σ²)+var(μ)"""
    def __init__(self, models):
        self.models = models
        self.probabilistic = True
    def __call__(self, x_hist, a_full):
        mus, lvs = [], []
        for m in self.models:
            mu, lv = m(x_hist, a_full)
            mus.append(mu); lvs.append(lv)
        mu_stack = torch.stack(mus, 0)                    # [n, B, H]
        lv_stack = torch.stack(lvs, 0)                    # [n, B, H]
        mu = mu_stack.mean(0)                             # [B, H]
        sig2 = torch.exp(lv_stack).mean(0) + mu_stack.var(0)  # aleatoric + epistemic
        lv = torch.log(sig2.clamp_min(1e-6))
        return mu, lv

# ============ 评测 ============
M.SP_TRAJ = 0; M.M_STEP = 6; M.H_PLAN = 18
M.FIX_MODE = 'overlap'; M.LAMBDA3 = 0.05; M.HARD_DELTA = 5.0
M.BENCH_SP_EACH = True
M.RISK_LAMBDA = 0.0
N_TRACKS = 2 if SMOKE else 50
SEEDS = [42] if SMOKE else [42, 7, 13]
KEYS = ['rmse_mpc', 'rmse_pid', 'iae_mpc', 'iae_pid', 'itae_mpc', 'itae_pid',
        'act_tv_mpc', 'act_tv_pid', 'viol_mpc', 'overtemp_int_mpc', 'overtemp_int_pid',
        'overtemp_mpc', 'overtemp_pid', 'jump_mean', 'jump_share']

def run(wm, planner, scene, dist_amp):
    M.DIST_AMP = dist_amp
    rows = []
    for seed in SEEDS:
        np.random.seed(seed)
        starts = np.random.choice(range(len(M.test_raw) - M.W - M.H_OUT - 120),
                                  N_TRACKS, replace=False)
        for s in starts:
            mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, planner)
            m = M.metrics(mpc_t, pid_t, tset, mpc_a, pid_a)
            T = len(mpc_a); M_ = M.M_STEP
            jumps = [float(np.abs(mpc_a[i] - mpc_a[i - 1]).mean()) for i in range(M_, T, M_) if i < T]
            tv = float(np.abs(np.diff(mpc_a, axis=0)).sum()) / 2.0
            m['jump_mean'] = float(np.mean(jumps)) if jumps else float('nan')
            m['jump_share'] = float(np.sum(jumps) / tv) if jumps and tv > 0 else float('nan')
            rows.append({k: m[k] for k in KEYS})
    json.dump(rows, open(f'{OUT_DIR}/{planner}_{scene}.json', 'w'), indent=2)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    agg['viol_mpc'] = float(np.sum([r['viol_mpc'] for r in rows]))
    agg['overtemp_mpc'] = float(np.sum([r['overtemp_mpc'] for r in rows]))
    p = stats.wilcoxon([r['rmse_mpc'] for r in rows], [r['rmse_pid'] for r in rows]).pvalue
    print(f"=== 集成{planner} {scene} (n={len(rows)}) ===")
    print(f"  集成+{planner}: RMSE {agg['rmse_mpc']:.3f} | IAE {agg['iae_mpc']:.1f} | TV {agg['act_tv_mpc']:.3f} "
          f"| jump {agg['jump_mean']:.3f} | viol {agg['viol_mpc']:.0f} | 超温 {agg['overtemp_mpc']:.0f}s | p={p:.2e}")
    return rows

if __name__ == '__main__':
    t0 = time.time()
    models = [load_model(f'{CK_DIR}/seed7.pth'), load_model(f'{CK_DIR}/seed13.pth'),
              load_model('results/exp_048_horizon/checkpoints/h18.pth')]
    ens = EnsembleWM(models)
    print(f"ensemble 成员: {len(models)}")
    # 评测矩阵
    run(ens, 'cem', 'dist', 0.3)
    run(ens, 'cem', 'nodist', 0.0)
    run(ens, 'grad', 'dist', 0.3)
    run(ens, 'grad', 'nodist', 0.0)
    print(f"\n===== 集成评测完成 ({(time.time()-t0)/60:.1f}min) =====")
    print(f"Saved: {OUT_DIR}/")
