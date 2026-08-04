#!/usr/bin/env python3
"""
exp_S3_causal_scan.py — S3b: 世界模型因果方向翻转阈值扫描 (2026-08-04)
======================================================================
评测世界 M0+M8+M9 集成: 阀位阶跃幅度 {1,2,3,5,7,10,15,20}%, 持续保持,
测 60s/120s 的 ΔT 方向 → 找方向翻转临界幅度 (§24 持续阶跃翻转在新世界的量化)

判定 (docs/supplementary_experiments.md §3b):
  翻转阈值 > +10% → MPC 实际动作 (<5%) 远在安全区
  翻转阈值 ≤ +3% → MPC 动作可能进入翻转区
用法: python exp_S3_causal_scan.py [--smoke]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.phase2_mpc.exp_S1_fair_comparison import (  # noqa: E402
    WORLD_IDS, CONTROLLERS, make_starts)
from experiments.phase2_mpc.eval_protocol import (  # noqa: E402
    WorldSim, DEVICE, W, H_OUT, TARGET_IDX, VALVE_IDX, test_raw)

AMPS = [1, 2, 3, 5, 7, 10, 15, 20]     # 阶跃幅度 (%)
OUT_DIR = 'results/exp_S3'


def scan(world, starts, amps=AMPS, n_seeds=3):
    rows = []
    for si, s in enumerate(starts[:n_seeds]):
        for amp in amps:
            # 基准: 动作保持当前值 (零阶跃)
            win0 = torch.FloatTensor(test_raw[s:s+W]).unsqueeze(0).to(DEVICE)
            a_base = torch.FloatTensor(test_raw[s+W:s+W+H_OUT, VALVE_IDX]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mu_base = world.predict(win0, a_base[0])
            v2_0 = float(test_raw[s+W, VALVE_IDX[1]])
            # 阶跃: V2 跳 amp% 并持续
            a_step = a_base.clone()
            a_step[0, :, 1] = min(v2_0 + amp, 45.0)
            a_step[0, :, 0] = a_base[0, 0, 0]  # V1 不变
            with torch.no_grad():
                mu_step = world.predict(win0, a_step[0])
            dT6 = float(mu_step[5] - mu_base[5])     # 60s ΔT
            dT12 = float(mu_step[11] - mu_base[11])  # 120s ΔT
            rows.append(dict(start=int(s), amp=amp, v2_0=v2_0,
                             dT6=dT6, dT12=dT12,
                             dir6=1 if dT6 > 0 else (-1 if dT6 < 0 else 0),
                             dir12=1 if dT12 > 0 else (-1 if dT12 < 0 else 0)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    world = WorldSim(WORLD_IDS, controller_ids=CONTROLLERS)
    starts = make_starts(42, 5)
    rows = scan(world, starts, n_seeds=3 if not args.smoke else 2)
    # 汇总: 每幅度下方向一致率
    print(f"[S3b] 世界={WORLD_IDS} 起点={len(set(r['start'] for r in rows))} "
          f"幅度={AMPS} | 120s 方向 (正=温度升)")
    flip = {}
    for amp in AMPS:
        rs = [r for r in rows if r['amp'] == amp]
        dirs = [r['dir12'] for r in rs]
        pos = sum(1 for d in dirs if d > 0); neg = sum(1 for d in dirs if d < 0)
        mean_dT = np.mean([r['dT12'] for r in rs])
        flip[amp] = dict(pos=pos, neg=neg, mean_dT=float(mean_dT))
        print(f"  +{amp:2d}%: ΔT12 {mean_dT:+.3f}°C (正{pos}/负{neg})")
    # 翻转阈值: 找方向反转 (正→负) 的临界幅度
    print("\n  [判定]")
    for amp in AMPS:
        if flip[amp]['pos'] == 0 and flip[amp]['neg'] > 0:
            print(f"  幅度 +{amp}% 起方向翻转 (全部负) — 阈值 ≤ {amp}%")
            break
    else:
        print("  全部幅度方向为正 — 无翻转 (阈值 > 20%)")
    json.dump(dict(rows=rows, flip=flip), open(f"{OUT_DIR}/s3_causal_scan.json", 'w'),
              indent=2, default=float)
    print(f"Saved: {OUT_DIR}/s3_causal_scan.json")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"runtime: {time.time()-t0:.1f}s")
