#!/usr/bin/env python3
"""
exp_S2_hplan_scan.py — S2: 公平协议下 H_PLAN 扫描 (2026-08-04)
================================================================
M7 主线模型, H_PLAN ∈ {6, 10, 14, 18}, M_STEP=6, 成本=综合工程代价 cost_d
(S1 中 M7 在 cost_d 双场景表现最优且含 TV/超温 — 配置扫描用综合口径最稳)
评测: 150 轨迹 × 3 起点集 × 扰动场景, 配对 Wilcoxon

判定 (docs/supplementary_experiments.md §2):
  RMSE 差异 p>0.05 + TV 差异显著 → 选 TV 更低
  RMSE 差异 p<0.05 → 选 RMSE 更低
用法: python exp_S2_hplan_scan.py [--smoke]
"""
import argparse, json, os, sys, time
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from experiments.phase2_mpc.exp_S1_fair_comparison import (  # noqa: E402
    N_TRACKS, START_SEEDS, N_STEPS, M_STEP, DIST_SIGMA, DIST_RHO, WORLD_IDS,
    CONTROLLERS, make_starts, run_arm, summarize, paired_test)
from experiments.phase2_mpc.eval_protocol import (  # noqa: E402
    CostConfig, Disturbance, MPCController, PIDController, WorldSim, load_wm)

HP = [6, 10, 14, 18]
COST = 'd'                       # 综合工程代价 (RMSE+0.1·超温积分+0.01·TV)
OUT_DIR = 'results/exp_S2'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    n_tracks = 2 if args.smoke else N_TRACKS
    seeds = START_SEEDS[:1] if args.smoke else START_SEEDS
    n_steps = 40 if args.smoke else N_STEPS
    os.makedirs(OUT_DIR, exist_ok=True)
    starts_by_seed = {s: make_starts(s, n_tracks) for s in seeds}

    print(f"[S2] 世界={WORLD_IDS} 模型=M7 成本=cost_{COST} H_PLAN={HP} 轨迹={len(seeds)}×{n_tracks}")
    world = WorldSim(WORLD_IDS, controller_ids=CONTROLLERS)
    wm7 = load_wm('M7')
    cost = CostConfig(COST)

    all_rows, summary = [], {}
    pid_rows = run_arm('PID', lambda: PIDController(), world, starts_by_seed, True, n_steps)
    all_rows += pid_rows; summary['PID'] = summarize(pid_rows)
    print(f"  PID  {summary['PID']}")

    for h in HP:
        arm = f'M7/H{h}'
        rows = run_arm(arm, lambda h=h: MPCController(wm7, cost, h_plan=h, m_step=M_STEP),
                       world, starts_by_seed, True, n_steps)
        all_rows += rows; summary[arm] = summarize(rows)
        p_r, n = paired_test(rows, pid_rows, 'rmse')
        print(f"  {arm:10s} {summary[arm]} | vs PID p_rmse={p_r:.2e} (n={n})")

    # 配对对比: H=10 (当前定稿) vs 其他
    verdict = {}
    r10 = [r for r in all_rows if r['arm'] == 'M7/H10']
    for h in HP:
        if h == 10: continue
        rh = [r for r in all_rows if r['arm'] == f'M7/H{h}']
        v = {}
        for key in ('rmse', 'overtemp_s', 'overtemp_int', 'tv', 'j_total'):
            p, n = paired_test(r10, rh, key)
            m10, mh = np.mean([r[key] for r in r10]), np.mean([r[key] for r in rh])
            v[key] = dict(H10=float(m10), Hh=float(mh), p=p, n=n,
                          winner='H10' if m10 < mh else f'H{h}')
        verdict[f'H{h}'] = v
        print(f"  [对比] H10 vs H{h}: rmse {v['rmse']['H10']:.3f}/{v['rmse']['Hh']:.3f} "
              f"(p={v['rmse']['p']:.2e}) | tv {v['tv']['H10']:.3f}/{v['tv']['Hh']:.3f} "
              f"(p={v['tv']['p']:.2e}) | j_total {v['j_total']['H10']:.3f}/{v['j_total']['Hh']:.3f} (p={v['j_total']['p']:.2e})")

    out = dict(protocol=dict(world=WORLD_IDS, model='M7', cost=COST, h_plan=HP,
                             m_step=M_STEP, n_steps=n_steps, dist=dict(on=True,
                             sigma=DIST_SIGMA, rho=DIST_RHO, mode='physical'),
                             start_seeds=seeds, n_tracks=n_tracks),
               summary=summary, verdict=verdict, per_track=all_rows)
    fn = os.path.join(OUT_DIR, f"s2_hplan{'_smoke' if args.smoke else ''}.json")
    json.dump(out, open(fn, 'w'), indent=2, default=float)
    print(f"\nSaved: {fn}  ({(time.time() - time.time()) / 60:.0f} min)")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
