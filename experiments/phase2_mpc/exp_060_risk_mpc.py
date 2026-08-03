#!/usr/bin/env python3
"""
exp_060_risk_mpc.py — 风险敏感 MPC: J = E[e²] + λ·CVaR_α(超温尾部)
====================================================================
概率 WM 该打的牌: M7 输出 (μ, σ), 规划目标加入超温尾部风险
  CVaR_t = μ_t + k_α·σ_t (k_0.95=2.0627, 正态假设)
  J += λ · Σ relu(CVaR_t − T_MAX)²

评测: λ ∈ {0, 0.5, 2, 10} × 扰动协议 (M_STEP=6, H_PLAN=10, hard5, DIST_AMP=0.3)
指标: RMSE (主) + IAE/ITAE/TV/jump/viol/超温积分/超温时间 + 风险激活率
风险激活率 = 规划步中 CVaR>T_MAX 的比例 — 确认风险项是否被真实触发
用法: python exp_060_risk_mpc.py [--smoke]
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv

SMOKE = '--smoke' in sys.argv
M.SP_TRAJ = 0
M.DIST_AMP = float(os.environ.get('DIST_AMP', 0.3))
M.M_STEP = 6
M.H_PLAN = 10
M.FIX_MODE = 'hard5'
M.HARD_DELTA = 0.0
M.RISK_SIGMA_ADD = M.DIST_AMP / np.sqrt(1.0 - 0.81)  # 扰动随机游走稳态 std (自相关0.9)
N_TRACKS = 2 if SMOKE else 10
OUT_DIR = os.environ.get('OUT_DIR', 'results/exp_060_risk')
os.makedirs(OUT_DIR, exist_ok=True)
LAMBDAS = [float(x) for x in os.environ.get('LAMBDAS', '0,0.5,2,10').split(',')]

wm = M.load_wm()

def boundary_stats(mpc_a):
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

# 风险激活率: 需要钩子统计 plan 内 CVaR>T_MAX 的步数 → 在 plan_grad 后检查
# 简化: 用模块级计数器, 由 plan_grad_wrap 统计 (每次规划后)
_real_plan_grad = M.plan_grad
RISK_HITS = []

def plan_grad_wrap(wm_, x_hist, t_set, a_last, a_init, sp_fut=None):
    a_plan, Js = _real_plan_grad(wm_, x_hist, t_set, a_last, a_init, sp_fut)
    if M.RISK_LAMBDA > 0:
        # 统计当前计划的风险激活 (a_plan 下的 CVaR 超阈值步数)
        with torch.no_grad():
            if M.H_PLAN < M.H_OUT:
                tail = a_plan[-1:].repeat(M.H_OUT - M.H_PLAN, 1)
                a_full = torch.cat([a_plan, tail], 0)
            else:
                a_full = a_plan[:M.H_OUT]
            mu, lv = wm_(x_hist, a_full.reshape(1, -1))
            sig = torch.exp(lv[0, :M.H_PLAN] * 0.5)
            if M.RISK_SIGMA_ADD > 0:
                sig = torch.sqrt(sig ** 2 + M.RISK_SIGMA_ADD ** 2)
            cvar = mu[0, :M.H_PLAN] + M.CVAR_K * sig
            RISK_HITS.append(int((cvar > M.T_MAX).sum().item()))
    return a_plan, Js

M.plan_grad = plan_grad_wrap

results = {}
t0 = time.time()
print(f"{'λ':>5} | {'RMSE':>7} {'IAE':>8} {'ITAE':>8} {'TV_m':>6} {'jump':>6} {'viol':>4} "
      f"{'超温积分':>7} {'超温s':>5} {'激活步/计划步':>12}")
for lam in LAMBDAS:
    M.RISK_LAMBDA = lam
    RISK_HITS.clear()
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
    n_plan_steps = len(RISK_HITS) * M.H_PLAN
    agg['risk_activation'] = float(np.sum(RISK_HITS)) / n_plan_steps if n_plan_steps else 0.0
    results[str(lam)] = {'agg': agg, 'per_track': rows}
    print(f"{lam:>5} | {agg['rmse_mpc']:>7.3f} {agg['iae_mpc']:>8.1f} {agg['itae_mpc']:>8.0f} "
          f"{agg['act_tv_mpc']:>6.3f} {agg['jump_mean']:>6.3f} {agg['viol_mpc']:>4.0f} "
          f"{agg['overtemp_int_mpc']:>7.3f} {agg['overtemp_mpc']:>5.0f} "
          f"{agg['risk_activation']*100:>9.2f}%  ({(time.time()-tc)/60:.1f}min)")
    json.dump({'agg': agg, 'per_track': rows}, open(f"{OUT_DIR}/lam{lam}.json", 'w'), indent=2)

print(f"\n===== 风险敏感 MPC 对比完成 ({(time.time()-t0)/60:.1f}min) =====")
print(f"Saved: {OUT_DIR}/")
