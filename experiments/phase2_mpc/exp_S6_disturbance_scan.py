#!/usr/bin/env python3
"""
exp_S6_disturbance_scan.py — S6: 扰动谱敏感性 (2026-08-04)
============================================================
4 种扰动谱 × (M7 + PID + 线性MPC), 50 轨迹 × seed 42, 成本 cost_d (S1 最优口径)
  (a) σ=0.3 ρ=0.9 低频随机游走 (当前)   (b) σ=0.3 ρ=0.0 高频白噪声
  (c) σ=0.8 ρ=0.9 大幅低频            (d) σ=0.8 ρ=0.0 大幅高频
判定: M7 全谱优于 PID → 鲁棒性稳健; 高频下优势消失 → 诚实报告边界
用法: python exp_S6_disturbance_scan.py [--smoke]
"""
import argparse, json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.phase2_mpc.exp_S1_fair_comparison import (  # noqa: E402
    WORLD_IDS, CONTROLLERS, make_starts, summarize, paired_test)
from experiments.phase2_mpc.eval_protocol import (  # noqa: E402
    CostConfig, Disturbance, MPCController, PIDController, WorldSim, load_wm,
    run_episode, metrics, H_OUT)

from experiments.phase2_mpc.exp_S4_linear_mpc import LinearARX, ORDER, COEF  # noqa: E402

OUT_DIR = 'results/exp_S6'
SPECTRA = [(0.3, 0.9), (0.3, 0.0), (0.8, 0.9), (0.8, 0.0)]
COST = 'd'                       # S1 最优口径 (M7 j_total 最低)


def run_arm_spec(controller_factory, world, starts_by_seed, sigma, rho, n_steps):
    rows = []
    for s_seed, starts in starts_by_seed.items():
        for k, st in enumerate(starts):
            dist = Disturbance(sigma=sigma, rho=rho, mode='physical',
                               seed=hash((s_seed, k)) % (2 ** 31))
            ep = run_episode(controller_factory(), world, st, n_steps=n_steps, dist=dist)
            if len(ep['temp']) < 10:
                continue
            m = metrics(ep)
            m.update(arm='x', start_seed=s_seed, track=int(st))
            rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    n_tracks = 2 if args.smoke else 50
    n_steps = 40 if args.smoke else 120
    seeds = [42]
    os.makedirs(OUT_DIR, exist_ok=True)
    world = WorldSim(WORLD_IDS, controller_ids=CONTROLLERS)
    wm7 = load_wm('M7')
    arx = LinearARX(COEF, ORDER)
    cost = CostConfig(COST)

    all_out = {}
    for (sigma, rho) in SPECTRA:
        tag = f's{sigma}_r{rho}'
        starts_by_seed = {s: make_starts(s, n_tracks) for s in seeds}
        pid = run_arm_spec(lambda: PIDController(), world, starts_by_seed, sigma, rho, n_steps)
        m7 = run_arm_spec(lambda: MPCController(wm7, cost, h_plan=18, m_step=6),
                          world, starts_by_seed, sigma, rho, n_steps)
        lin = run_arm_spec(lambda: MPCController(arx, cost, h_plan=18, m_step=6),
                           world, starts_by_seed, sigma, rho, n_steps)
        s_pid, s_m7, s_lin = summarize(pid), summarize(m7), summarize(lin)
        p1, n1 = paired_test(m7, pid, 'rmse')
        p2, n2 = paired_test(lin, pid, 'rmse')
        print(f"[S6] {tag}: PID {s_pid['rmse']:.3f} | M7 {s_m7['rmse']:.3f} (p={p1:.2e}) "
              f"| Lin {s_lin['rmse']:.3f} (p={p2:.2e})")
        print(f"       TV: PID {s_pid['tv']:.3f} | M7 {s_m7['tv']:.3f} | Lin {s_lin['tv']:.3f} | "
              f"j_total: PID {s_pid['j_total']:.3f} | M7 {s_m7['j_total']:.3f} | Lin {s_lin['j_total']:.3f}")
        all_out[tag] = dict(PID=s_pid, M7=s_m7, Lin=s_lin,
                            p_m7_vs_pid=float(p1), p_lin_vs_pid=float(p2), n=n1)

    json.dump(all_out, open(f"{OUT_DIR}/s6_spectra.json", 'w'), indent=2, default=float)
    print(f"\nSaved: {OUT_DIR}/s6_spectra.json")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"runtime: {(time.time()-t0)/60:.1f} min")
