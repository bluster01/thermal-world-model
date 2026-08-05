#!/usr/bin/env python3
"""
exp_S3b_causal_scan_v2.py — S3b 重写版: 因果方向正确性扫描 (2026-08-05)
=======================================================================
替代 exp_S3_causal_scan.py。旧版有 4 个缺陷, 本版逐条修复:

  [B1] 判定符号反了 (致命)
       开减温阀 → 喷水增加 → 降温。物理正确 = ΔT < 0。
       旧版把 ΔT<0 判为"翻转", ΔT>0 判为"正常", 结论完全反向。
       本版: inverted := (ΔT > 0), 即升温才是因果反演。

  [B2] V1 污染
       旧版 a_step[0,:,0] = a_base[0,0,0] 把一级阀冻结在首值,
       而基准里一级阀是随真实轨迹变化的 → ΔT 混入了"冻结V1"的效应。
       本版: 未被扰动的那个阀严格保持 a_base 不变。

  [B3] 阶跃定义混淆
       旧版 a_step[:,1] = v2_0 + amp 是"恒定保持", 与基准的真实变化轨迹相比,
       差异里混入了"抹掉V2自然变化"的效应。
       本版默认 mode='shift': a_step[:,v] = a_base[:,v] + amp (保留自然变化),
       纯粹隔离阶跃量; mode='hold' 保留旧行为供对照。

  [B4] n=3 且三个起点结论互相矛盾
       本版默认 n_starts=80, 报告反演率 + Wilson 95% CI, 并按初始阀位分层。

新增:
  - 双阀扫描 (一级=长通路, 二级=短通路) — Phase1 分析指出长通路才是问题所在
  - 饱和过滤: v0+amp 触顶 45 时实际阶跃被削, 标记并剔除
  - --worlds 可指定世界组成, 用于检验"§24 翻转是 M7 单模型伪影"这一说法

用法:
  python experiments/phase2_mpc/exp_S3b_causal_scan_v2.py --smoke
  python experiments/phase2_mpc/exp_S3b_causal_scan_v2.py
  python experiments/phase2_mpc/exp_S3b_causal_scan_v2.py --worlds M7 --tag m7only
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.phase2_mpc.exp_S1_fair_comparison import (  # noqa: E402
    WORLD_IDS, CONTROLLERS, make_starts)
from experiments.phase2_mpc.eval_protocol import (  # noqa: E402
    WorldSim, DEVICE, W, H_OUT, VALVE_IDX, test_raw)

AMPS = [1, 2, 3, 5, 7, 10, 15, 20]      # 阶跃幅度 (%)
STEPS = [3, 6, 12, 18]                  # 评估步 (×10s = 30/60/120/180s)
VALVES = {0: '一级(长通路)', 1: '二级(短通路)'}
U_LO, U_HI = 0.0, 45.0                  # 与 eval_protocol.Actuator 一致
SAT_TOL = 0.5                           # 实际阶跃 < 0.5×名义 → 判为饱和, 剔除
OUT_DIR = 'results/exp_S3'

# 物理先验: 开阀(+amp) → 喷水增加 → 降温 → ΔT 应为负
EXPECTED_SIGN = -1


def wilson_ci(k, n, z=1.96):
    """反演率的 Wilson 95% 置信区间 (小样本比二项正态近似可靠)"""
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def scan(world, starts, amps=AMPS, mode='shift'):
    """对每个起点 × 每个阀 × 每个幅度, 测各步 ΔT"""
    rows = []
    for s in starts:
        s = int(s)
        win0 = torch.FloatTensor(test_raw[s:s + W]).unsqueeze(0).to(DEVICE)
        a_base = torch.FloatTensor(
            test_raw[s + W:s + W + H_OUT, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu_base = world.predict(win0, a_base[0])

        for v in VALVES:
            v0 = float(test_raw[s + W, VALVE_IDX[v]])
            for amp in amps:
                a_step = a_base.clone()          # [B2] 另一个阀原样保留
                if mode == 'shift':
                    # [B3] 保留自然变化, 整体平移 amp
                    a_step[0, :, v] = torch.clamp(a_base[0, :, v] + amp, U_LO, U_HI)
                else:  # 'hold' — 旧行为, 恒定保持
                    a_step[0, :, v] = min(v0 + amp, U_HI)

                # 饱和检测: 实际平均施加的阶跃量
                eff = float((a_step[0, :, v] - a_base[0, :, v]).mean())
                saturated = eff < SAT_TOL * amp

                with torch.no_grad():
                    mu_step = world.predict(win0, a_step[0])

                rec = dict(start=s, valve=v, amp=amp, v0=v0,
                           eff_amp=eff, saturated=bool(saturated))
                for k in STEPS:
                    dT = float(mu_step[k - 1] - mu_base[k - 1])
                    rec[f'dT{k}'] = dT
                    # 物理正确 = 降温(负); 升温(正) = 因果反演
                    rec[f'inv{k}'] = bool(dT * EXPECTED_SIGN < 0)
                rows.append(rec)
    return rows


def aggregate(rows):
    """按 阀×幅度×步 汇总反演率"""
    agg = {}
    for v in VALVES:
        for amp in AMPS:
            rs = [r for r in rows if r['valve'] == v and r['amp'] == amp
                  and not r['saturated']]
            if not rs:
                continue
            entry = dict(n=len(rs), v0_mean=float(np.mean([r['v0'] for r in rs])))
            for k in STEPS:
                inv = sum(1 for r in rs if r[f'inv{k}'])
                lo, hi = wilson_ci(inv, len(rs))
                entry[f'step{k}'] = dict(
                    inv_rate=inv / len(rs), inv_n=inv, ci=[lo, hi],
                    mean_dT=float(np.mean([r[f'dT{k}'] for r in rs])),
                    median_dT=float(np.median([r[f'dT{k}'] for r in rs])))
            agg[f'valve{v}_amp{amp}'] = entry
    return agg


def stratify_by_v0(rows, valve, step=12, n_bins=4):
    """按初始阀位分层看反演率 — 检验因果正确性是否依赖工作点"""
    rs = [r for r in rows if r['valve'] == valve and not r['saturated']]
    if not rs:
        return {}
    v0s = np.array([r['v0'] for r in rs])
    qs = np.quantile(v0s, np.linspace(0, 1, n_bins + 1))
    out = {}
    for b in range(n_bins):
        lo, hi = qs[b], qs[b + 1]
        sel = [r for r in rs if (lo <= r['v0'] <= hi if b == n_bins - 1
                                 else lo <= r['v0'] < hi)]
        if not sel:
            continue
        inv = sum(1 for r in sel if r[f'inv{step}'])
        cl, ch = wilson_ci(inv, len(sel))
        out[f'q{b+1}_[{lo:.1f},{hi:.1f}]'] = dict(
            n=len(sel), inv_rate=inv / len(sel), ci=[cl, ch],
            mean_dT=float(np.mean([r[f'dT{step}'] for r in sel])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n-starts', type=int, default=80)
    ap.add_argument('--mode', choices=['shift', 'hold'], default='shift')
    ap.add_argument('--worlds', nargs='+', default=None,
                    help='世界模型组成 (默认 M0 M8 M9)')
    ap.add_argument('--tag', default='v2')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    worlds = args.worlds or WORLD_IDS
    n_starts = 6 if args.smoke else args.n_starts

    world = WorldSim(worlds, controller_ids=CONTROLLERS)
    starts = make_starts(42, n_starts)

    print(f"[S3b-v2] 世界={worlds} 起点={n_starts} 幅度={AMPS} 模式={args.mode}")
    print(f"[S3b-v2] 物理先验: 开阀 → 降温, 故 ΔT>0 = 因果反演\n")

    t0 = time.time()
    rows = scan(world, starts, mode=args.mode)
    agg = aggregate(rows)

    # ── 主表: 反演率 ──
    for v, vname in VALVES.items():
        print(f"  ── 阀 {v} {vname} ──")
        print(f"  {'amp':>4} {'n':>4} | " +
              " | ".join(f"{'t'+str(k)+' inv%':>10}" for k in STEPS) +
              " | " + " | ".join(f"{'t'+str(k)+' ΔT':>9}" for k in STEPS))
        for amp in AMPS:
            e = agg.get(f'valve{v}_amp{amp}')
            if not e:
                print(f"  {amp:>4} {'—':>4} | (全部饱和)")
                continue
            inv_s = " | ".join(f"{e[f'step{k}']['inv_rate']*100:>9.1f}%" for k in STEPS)
            dt_s = " | ".join(f"{e[f'step{k}']['mean_dT']:>+9.3f}" for k in STEPS)
            print(f"  {amp:>4} {e['n']:>4} | {inv_s} | {dt_s}")
        print()

    # ── 分层: 初始阀位 ──
    strat = {}
    for v in VALVES:
        st = stratify_by_v0(rows, v, step=12)
        strat[f'valve{v}'] = st
        if st:
            print(f"  ── 阀 {v} 按初始阀位分层 (t12 反演率) ──")
            for k, d in st.items():
                print(f"    {k:>22}: n={d['n']:>3} inv={d['inv_rate']*100:>5.1f}% "
                      f"CI[{d['ci'][0]*100:.1f},{d['ci'][1]*100:.1f}]% "
                      f"ΔT={d['mean_dT']:+.3f}")
            print()

    # ── 判定 ──
    print("  [判定]")
    verdict = {}
    for v, vname in VALVES.items():
        worst = None
        for amp in AMPS:
            e = agg.get(f'valve{v}_amp{amp}')
            if not e:
                continue
            r = e['step12']['inv_rate']
            if worst is None or r > worst[1]:
                worst = (amp, r, e['step12']['ci'])
        if worst is None:
            continue
        amp, r, ci = worst
        verdict[f'valve{v}'] = dict(worst_amp=amp, worst_inv_rate=r, ci=ci)
        if ci[0] > 0.5:
            msg = f"多数工作点因果反演 (CI下界 {ci[0]*100:.0f}% > 50%) — 严重"
        elif ci[0] > 0.05:
            msg = f"存在显著反演 (CI下界 {ci[0]*100:.0f}% > 5%) — 需报告"
        elif ci[1] < 0.05:
            msg = f"反演可忽略 (CI上界 {ci[1]*100:.0f}% < 5%) — 安全"
        else:
            msg = "证据不足, 需增大样本"
        print(f"    阀{v} {vname}: 最差 amp=+{amp}% 反演率 {r*100:.1f}% "
              f"CI[{ci[0]*100:.1f},{ci[1]*100:.1f}]% → {msg}")

    out = dict(
        protocol=dict(worlds=worlds, n_starts=n_starts, amps=AMPS, steps=STEPS,
                      mode=args.mode, expected_sign=EXPECTED_SIGN,
                      sat_tol=SAT_TOL, u_hi=U_HI),
        aggregate=agg, stratified=strat, verdict=verdict, rows=rows)
    path = f"{OUT_DIR}/s3b_causal_scan_{args.tag}.json"
    json.dump(out, open(path, 'w'), indent=2, default=float)
    print(f"\nSaved: {path}")
    print(f"runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
