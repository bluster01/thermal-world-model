#!/usr/bin/env python3
"""
exp_068_linear_mpc_eval.py — P1: ΔT-ARX 线性 MPC 闭环评测
============================================================
同框架换模型: LinearARX 实现 wm 接口, 复用 exp_027 simulate+grad planner
协议: H_PLAN=18, ovl05_hard5, M_STEP=6, 3起点集×50轨迹, 扰动+无扰动, 每步SP基准
对比: DWM-MPC (M7) / 线性MPC / PID, Wilcoxon
用法: python exp_068_linear_mpc_eval.py [--smoke]
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
from experiments.phase1_dynamics.exp_025_unified_benchmark import TARGET_IDX, VALVE_IDX, SP_IDX

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMOKE = '--smoke' in sys.argv
ORDER = 12
COEF = np.load('results/exp_067_linear_mpc/arx_dT_coef.npy')

class LinearARX:
    """ΔT-ARX 增量模型, 实现 wm 接口: __call__(x_hist, a_future) -> (mu [B,H], lv [B,H])
    纯 torch 可微 (grad planner 需要)"""
    def __init__(self, coef, order=12):
        self.coef = torch.FloatTensor(coef).to(DEVICE)
        self.order = order
        self.probabilistic = False
    def __call__(self, x_hist, a_future):
        B = x_hist.shape[0]
        H = a_future.shape[1] // 2
        a = a_future.reshape(B, H, 2)                        # [B, H, 2]
        dev = x_hist.device
        d_hist = torch.diff(x_hist[:, :, TARGET_IDX], dim=1)[:, -ORDER:].flip(1)  # ΔT 最近在前
        V1h = x_hist[:, -ORDER:, VALVE_IDX[0]].flip(1)
        V2h = x_hist[:, -ORDER:, VALVE_IDX[1]].flip(1)
        SPh = x_hist[:, -6:, SP_IDX].flip(1)
        t_cur = x_hist[:, -1, TARGET_IDX]                    # [B]
        mus = []
        for k in range(H):
            idxs = torch.arange(k, k + ORDER, device=dev).clamp(max=H - 1)
            v1 = a[:, idxs, 0].flip(1)                        # 最近(k)在前, 与训练特征顺序一致
            v2 = a[:, idxs, 1].flip(1)
            r = torch.cat([d_hist, v1, v2, SPh,
                           torch.ones(B, 1, device=dev)], 1)  # [B, 42]
            dd = r @ self.coef                                # [B]
            t_cur = t_cur + dd
            mus.append(t_cur)
            d_hist = torch.cat([dd.unsqueeze(1), d_hist[:, :-1]], 1)
        mu = torch.stack(mus, 1)                              # [B, H]
        lv = torch.full_like(mu, 2 * np.log(0.01))            # 确定性模型: σ=0.01
        return mu, lv

# ============ 评测 ============
M.SP_TRAJ = 0; M.M_STEP = 6; M.H_PLAN = 18
M.FIX_MODE = 'overlap'; M.LAMBDA3 = 0.05; M.HARD_DELTA = 5.0
M.BENCH_SP_EACH = True
M.RISK_LAMBDA = 0.0
N_TRACKS = 2 if SMOKE else 50
SEEDS = [42] if SMOKE else [42, 7, 13]
OUT_DIR = os.environ.get('OUT_DIR', 'results/exp_068_linear_mpc')
os.makedirs(OUT_DIR, exist_ok=True)

arx = LinearARX(COEF, ORDER)
KEYS = ['rmse_mpc', 'rmse_pid', 'iae_mpc', 'iae_pid', 'itae_mpc', 'itae_pid',
        'act_tv_mpc', 'act_tv_pid', 'viol_mpc', 'overtemp_int_mpc', 'overtemp_int_pid',
        'overtemp_mpc', 'overtemp_pid']

def run(scene, dist_amp):
    M.DIST_AMP = dist_amp
    rows = []
    for seed in SEEDS:
        np.random.seed(seed)
        starts = np.random.choice(range(len(M.test_raw) - M.W - M.H_OUT - 120),
                                  N_TRACKS, replace=False)
        for s in starts:
            mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(arx, s, 'grad')
            m = M.metrics(mpc_t, pid_t, tset, mpc_a, pid_a)
            T = len(mpc_a); M_ = M.M_STEP
            jumps = [float(np.abs(mpc_a[i] - mpc_a[i - 1]).mean()) for i in range(M_, T, M_) if i < T]
            tv = float(np.abs(np.diff(mpc_a, axis=0)).sum()) / 2.0
            m['jump_mean'] = float(np.mean(jumps)) if jumps else float('nan')
            m['jump_share'] = float(np.sum(jumps) / tv) if jumps and tv > 0 else float('nan')
            rows.append({k: m[k] for k in KEYS + ['jump_mean', 'jump_share']})
    json.dump(rows, open(f"{OUT_DIR}/{scene}.json", 'w'), indent=2)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    agg['jump_mean'] = float(np.mean([r['jump_mean'] for r in rows]))
    agg['viol_mpc'] = float(np.sum([r['viol_mpc'] for r in rows]))
    agg['overtemp_mpc'] = float(np.sum([r['overtemp_mpc'] for r in rows]))
    p = stats.wilcoxon([r['rmse_mpc'] for r in rows], [r['rmse_pid'] for r in rows]).pvalue
    print(f"=== 线性MPC {scene} (n={len(rows)}) ===")
    print(f"  线性MPC: RMSE {agg['rmse_mpc']:.3f} | IAE {agg['iae_mpc']:.1f} | ITAE {agg['itae_mpc']:.0f} "
          f"| TV {agg['act_tv_mpc']:.3f} | jump {agg['jump_mean']:.3f} | viol {agg['viol_mpc']:.0f} "
          f"| 超温 {agg['overtemp_mpc']:.0f}s")
    print(f"  PID:    RMSE {agg['rmse_pid']:.3f} | IAE {agg['iae_pid']:.1f} | TV {agg['act_tv_pid']:.3f}")
    print(f"  Wilcoxon (线性MPC vs PID): p={p:.2e}")

t0 = time.time()
run('dist', 0.3)
run('nodist', 0.0)
print(f"\n===== 线性MPC评测完成 ({(time.time()-t0)/60:.1f}min) =====")
print(f"Saved: {OUT_DIR}/")
