#!/usr/bin/env python3
"""
exp_086_final_main.py — Phase2 最终收口主协议 (论文主表定稿版)
================================================================
定稿配置 (2026-08-04):
  - 模型: M7 (Direct WM, 概率, β=−0.3) — 最终模型架构
  - 规划: grad, H_PLAN=18, M_STEP=6, FIX_MODE=ovl05_hard5 (LAMBDA3=0.05, HARD_DELTA=5)
  - 成本: RMSE-only (LAMBDA1=0, LAMBDA2=0, 用户: 不控制动作)
  - 执行端: KF(q=0.01) + SMA6 (EXEC_KF=0.01, EXEC_SMA=6)
  - 协议: 3 起点集 (seed 42/7/13) × 50 轨迹 = 150, 双场景 (无扰动 / 扰动 DIST_AMP=0.3)
  - 基准: 每步真实 SP (BENCH_SP_EACH=True)
  - 对比: MPC vs PID (同扰动序列, 同世界)
指标: RMSE/IAE/ITAE/TV/jump/viol/超温积分/超温时间 (per-track json + 3-seed 均值 + 配对 Wilcoxon)
用法: python exp_086_final_main.py [--smoke]
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

# ===== 最终定稿配置 (2026-08-04) =====
M.SP_TRAJ = 0
M.DIST_AMP = 0.3
M.M_STEP = 6
M.H_PLAN = 18
M.FIX_MODE = 'overlap'      # ovl05_hard5
M.LAMBDA3 = 0.05
M.HARD_DELTA = 5.0
M.BENCH_SP_EACH = True
M.LAMBDA1 = 0.0             # RMSE-only
M.LAMBDA2 = 0.0             # RMSE-only
M.LAMBDA1_2ND = 0.0
M.EXEC_KF = float(os.environ.get('EXEC_KF', '0.01'))   # 执行端 KF q=0.01
M.EXEC_SMA = int(os.environ.get('EXEC_SMA', '6'))      # 执行端 SMA6

N_TRACKS = 2 if SMOKE else 50
SEEDS = [42] if SMOKE else [42, 7, 13]
OUT_DIR = os.environ.get('OUT_DIR', 'results/exp_086_final_main')
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[cfg] H_PLAN={M.H_PLAN} M_STEP={M.M_STEP} FIX={M.FIX_MODE}(λ3={M.LAMBDA3},δ={M.HARD_DELTA}) "
      f"L1={M.LAMBDA1} L2={M.LAMBDA2} KF={M.EXEC_KF} SMA={M.EXEC_SMA} "
      f"seeds={SEEDS} tracks={N_TRACKS}")

wm = M.load_wm()
KEYS = ['rmse_mpc', 'rmse_pid', 'iae_mpc', 'iae_pid', 'itae_mpc', 'itae_pid',
        'act_tv_mpc', 'act_tv_pid', 'viol_mpc', 'overtemp_int_mpc', 'overtemp_int_pid',
        'overtemp_mpc', 'overtemp_pid']

def run(dist_amp, tag):
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
            rows.append({k: m[k] for k in KEYS + ['jump_mean', 'jump_share']})
    json.dump(rows, open(f"{OUT_DIR}/{tag}.json", 'w'), indent=2)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    agg['jump_mean'] = float(np.mean([r['jump_mean'] for r in rows]))
    agg['jump_share'] = float(np.mean([r['jump_share'] for r in rows]))
    agg['viol_mpc'] = float(np.sum([r['viol_mpc'] for r in rows]))
    agg['overtemp_mpc'] = float(np.sum([r['overtemp_mpc'] for r in rows]))
    agg['overtemp_pid'] = float(np.sum([r['overtemp_pid'] for r in rows]))
    p_rmse = stats.wilcoxon([r['rmse_mpc'] for r in rows], [r['rmse_pid'] for r in rows]).pvalue
    p_tv = stats.wilcoxon([r['act_tv_mpc'] for r in rows], [r['act_tv_pid'] for r in rows]).pvalue
    n = len(rows)
    print(f"=== {tag} (n={n}) ===")
    print(f"  MPC: RMSE {agg['rmse_mpc']:.3f} | IAE {agg['iae_mpc']:.1f} | ITAE {agg['itae_mpc']:.0f} "
          f"| TV {agg['act_tv_mpc']:.3f} | jump {agg['jump_mean']:.3f} ({agg['jump_share']*100:.1f}%) "
          f"| viol {agg['viol_mpc']:.0f} | 超温积分 {agg['overtemp_int_mpc']:.2f} | 超温 {agg['overtemp_mpc']:.0f}s")
    print(f"  PID: RMSE {agg['rmse_pid']:.3f} | IAE {agg['iae_pid']:.1f} | ITAE {agg['itae_pid']:.0f} "
          f"| TV {agg['act_tv_pid']:.3f} | 超温积分 {agg['overtemp_int_pid']:.2f} | 超温 {agg['overtemp_pid']:.0f}s")
    print(f"  Wilcoxon: RMSE p={p_rmse:.2e} | TV p={p_tv:.2e}")
    return agg

t0 = time.time()
out = {}
out['ovl05_hard5_kf_sma6_dist'] = run(0.3, 'ovl05_hard5_kf_sma6_dist')
out['ovl05_hard5_kf_sma6_nodist'] = run(0.0, 'ovl05_hard5_kf_sma6_nodist')
json.dump(out, open(f"{OUT_DIR}/summary.json", 'w'), indent=2)
print(f"\n===== 最终主协议完成 ({(time.time()-t0)/60:.1f}min) =====")
print(f"Saved: {OUT_DIR}/summary.json")
