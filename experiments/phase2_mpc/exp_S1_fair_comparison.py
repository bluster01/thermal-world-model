#!/usr/bin/env python3
"""
exp_S1_fair_comparison.py — S1: M5 vs M7 公平重判 (推翻 exp_094 循环论证)
==========================================================================
问题: exp_094 在 "RMSE-only" 成本下判定 M5(确定性) 优于 M7(概率), 据此切换主模型。
      这是循环论证 —— 先决定不用 σ, 再发现不带 σ 的模型更好。
      且原协议存在自评分 (用被测 WM 评测自己) → M5/M7 在各自世界里跑, RMSE 不可比。

本实验在修复后的公平协议 (eval_protocol.py: P0-A/B/C/D) 下, 以 4 种成本口径重判:
  a) rmse_only : 仅跟踪误差            → 预期 M5 胜 (σ 无用武之地)
  b) overtemp  : + 非对称超温惩罚(μ)    → 检验单靠均值能否压超温
  c) cvar      : + CVaR 尾部风险(μ+kσ) → 概率模型专属能力
  d) total     : 综合工程代价           → 工程实际排序

判定:
  M7 在 (b)(c)(d) 胜出        → M5 切换是循环论证, M7 恢复主模型
  M5 在全部口径胜出            → M5 确实更优, 但需解释超温安全代价
  仅 (c) M7 胜                → 概率 WM 价值限于风险规划, 叙事需调整

用法:
  python experiments/phase2_mpc/exp_S1_fair_comparison.py --smoke        # 冒烟 (2轨迹)
  python experiments/phase2_mpc/exp_S1_fair_comparison.py                # 全量
  python experiments/phase2_mpc/exp_S1_fair_comparison.py --costs a c    # 指定口径
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.phase2_mpc.eval_protocol import (  # noqa: E402
    CostConfig, Disturbance, MPCController, PIDController, WorldSim,
    load_wm, metrics, run_episode, test_raw, W, H_OUT)

# ── 协议参数 ──
CONTROLLERS = ['M5', 'M7']            # 被测控制器模型
WORLD_IDS = ['M0', 'M8', 'M9']        # 评测世界集成 (与控制器不相交 — P0-B 契约)
START_SEEDS = [42, 7, 13]             # 起点集 seed
N_TRACKS = 50                         # 每起点集轨迹数 → 150 条
N_STEPS = 120
H_PLAN, M_STEP = 10, 6
DIST_SIGMA, DIST_RHO = 0.3, 0.9
OUT_DIR = 'results/exp_S1'


def make_starts(seed, n):
    rng = np.random.default_rng(seed)
    hi = len(test_raw) - W - H_OUT - N_STEPS - 2
    return rng.choice(hi, size=n, replace=False)


def run_arm(name, controller_factory, world, starts_by_seed, dist_on, n_steps):
    """跑一个方法臂, 返回 per-track 指标列表"""
    rows = []
    for s_seed, starts in starts_by_seed.items():
        for k, st in enumerate(starts):
            # 关键: 同一 track 的扰动序列由 (s_seed, k) 决定 → 所有方法共享同一扰动实现
            dist = Disturbance(sigma=DIST_SIGMA if dist_on else 0.0, rho=DIST_RHO,
                               mode='physical', seed=hash((s_seed, k)) % (2 ** 31))
            ep = run_episode(controller_factory(), world, st, n_steps=n_steps, dist=dist)
            if len(ep['temp']) < 10:
                continue
            m = metrics(ep)
            m.update(arm=name, start_seed=s_seed, track=int(st))
            rows.append(m)
    return rows


def summarize(rows, keys=('rmse', 'overtemp_s', 'overtemp_int', 'tv', 'iae', 'j_total')):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def paired_test(rows_a, rows_b, key):
    """配对 Wilcoxon (按 (start_seed, track) 对齐)"""
    ia = {(r['start_seed'], r['track']): r[key] for r in rows_a}
    ib = {(r['start_seed'], r['track']): r[key] for r in rows_b}
    common = sorted(set(ia) & set(ib))
    if len(common) < 10:
        return float('nan'), 0
    va = [ia[c] for c in common]
    vb = [ib[c] for c in common]
    if np.allclose(va, vb):
        return 1.0, len(common)
    return float(stats.wilcoxon(va, vb).pvalue), len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='冒烟: 2 轨迹 1 起点集 40 步')
    ap.add_argument('--costs', nargs='+', default=['a', 'b', 'c', 'd'])
    ap.add_argument('--no-dist', action='store_true', help='无扰动场景')
    ap.add_argument('--hybrid', action='store_true', help='附加 S1b: M5均值+M7σ 混合臂')
    ap.add_argument('--hplan', type=int, default=10, help='H_PLAN (定稿 18, S2 结论)')
    args = ap.parse_args()

    n_tracks = 2 if args.smoke else N_TRACKS
    seeds = START_SEEDS[:1] if args.smoke else START_SEEDS
    n_steps = 40 if args.smoke else N_STEPS
    dist_on = not args.no_dist
    tag = 'nodist' if args.no_dist else 'dist'

    os.makedirs(OUT_DIR, exist_ok=True)
    starts_by_seed = {s: make_starts(s, n_tracks) for s in seeds}

    print(f"[S1] 评测世界 (集成): {WORLD_IDS} | 控制器: {CONTROLLERS}")
    print(f"[S1] 场景={tag} 成本口径={args.costs} 轨迹={len(seeds)}×{n_tracks} 步长={n_steps}")
    world = WorldSim(WORLD_IDS, controller_ids=CONTROLLERS)
    wms = {m: load_wm(m) for m in CONTROLLERS}

    all_rows, summary = [], {}
    t0 = time.time()

    # ── 基线: 真闭环 PID (P0-A) ──
    pid_rows = run_arm('PID', lambda: PIDController(), world, starts_by_seed, dist_on, n_steps)
    all_rows += pid_rows
    summary['PID'] = summarize(pid_rows)
    print(f"  PID            {summary['PID']}")

    # ── MPC 臂: 模型 × 成本口径 ──
    for cv in args.costs:
        for mid in CONTROLLERS:
            cost = CostConfig(cv, sigma_add=DIST_SIGMA if (cv == 'c' and dist_on) else 0.0)
            if cost.needs_sigma() and not getattr(wms[mid], 'probabilistic', True):
                print(f"  skip {mid}/cost_{cv}: 确定性模型无 σ, CVaR 不适用")
                continue
            arm = f"{mid}/cost_{cv}"
            rows = run_arm(arm, lambda m=mid, c=cost: MPCController(
                wms[m], c, h_plan=args.hplan, m_step=M_STEP),
                world, starts_by_seed, dist_on, n_steps)
            all_rows += rows
            summary[arm] = summarize(rows)
            p_r, n = paired_test(rows, pid_rows, 'rmse')
            print(f"  {arm:14s} {summary[arm]} | vs PID p_rmse={p_r:.2e} (n={n})")

    # ── S1b: M5 均值 + M7 σ 混合 (解耦架构) ──
    if args.hybrid and 'c' in args.costs:
        from experiments.phase2_mpc.eval_protocol import HybridWM  # 可选扩展
        cost = CostConfig('c', sigma_add=DIST_SIGMA if dist_on else 0.0)
        hyb = HybridWM(wms['M5'], wms['M7'])
        rows = run_arm('M5mu+M7sig/cost_c', lambda: MPCController(
            hyb, cost, h_plan=args.hplan, m_step=M_STEP),
            world, starts_by_seed, dist_on, n_steps)
        all_rows += rows
        summary['M5mu+M7sig/cost_c'] = summarize(rows)
        print(f"  hybrid         {summary['M5mu+M7sig/cost_c']}")

    # ── M5 vs M7 逐口径配对检验 (核心判定) ──
    verdict = {}
    for cv in args.costs:
        a5 = [r for r in all_rows if r['arm'] == f"M5/cost_{cv}"]
        a7 = [r for r in all_rows if r['arm'] == f"M7/cost_{cv}"]
        if not a5 or not a7:
            continue
        v = {}
        for key in ('rmse', 'overtemp_s', 'overtemp_int', 'j_total'):
            p, n = paired_test(a5, a7, key)
            m5, m7 = np.mean([r[key] for r in a5]), np.mean([r[key] for r in a7])
            v[key] = dict(M5=float(m5), M7=float(m7), p=p, n=n,
                          winner='M5' if m5 < m7 else 'M7')
        verdict[f"cost_{cv}"] = v
        w = v['j_total']
        print(f"  [判定] cost_{cv}: j_total M5={w['M5']:.3f} M7={w['M7']:.3f} "
              f"→ {w['winner']} (p={w['p']:.2e})")

    out = dict(protocol=dict(world=WORLD_IDS, controllers=CONTROLLERS,
                             h_plan=args.hplan, m_step=M_STEP, n_steps=n_steps,
                             dist=dict(on=dist_on, sigma=DIST_SIGMA, rho=DIST_RHO,
                                       mode='physical'),
                             start_seeds=seeds, n_tracks=n_tracks),
               summary=summary, verdict=verdict, per_track=all_rows)
    fn = os.path.join(OUT_DIR, f"s1_{tag}{'_smoke' if args.smoke else ''}.json")
    json.dump(out, open(fn, 'w'), indent=2, default=float)
    print(f"\nSaved: {fn}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == '__main__':
    main()
