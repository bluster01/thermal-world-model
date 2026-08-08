#!/usr/bin/env python3
"""G3 参数健康摘要: 42 个 checkpoint 的 gain/τ/map/rate 参数分布与边界命中率。

审计 (PHASE3_5_VALIDATION_AUDIT_2026-08-09 §9 G3) 要求回传参数摘要:
- gain: 分布 + 接近零命中率 (塌缩检测)
- tau: 分布 + tau_min/tau_max 边界命中率
- monotone map: 各段斜率
- rate branch: 分布
"""
import os, sys, json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.phase35.data import load_cache, valid_window_anchors, extract_windows
from src.phase35.matrix import load_matrix, expand_matrix
from src.phase35.schema import TARGET_COLUMN, VALVE_COLUMN
from src.phase35.training import build_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RUN_ROOT = 'results/phase3_5/runs'
CACHE = {'A': '/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/phase3_5_cache/cache_A.npz',
         'B': '/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/phase3_5_cache/cache_B.npz'}
N_ANCHORS = 256

def main():
    matrix = load_matrix('configs/phase3_5/experiment_matrix.json')
    runs = expand_matrix(matrix)
    summaries = []
    for run in runs:
        run_dir = os.path.join(RUN_ROOT, run.run_id)
        ckpt = os.path.join(run_dir, 'checkpoint_best_val.pt')
        if not os.path.exists(ckpt):
            print(f'MISS {run.run_id}'); continue
        cache = load_cache(CACHE[run.side])
        model, features = build_model(run.config, cache, DEVICE)
        ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        model.eval()
        anchors = valid_window_anchors(cache, 'validation', features, TARGET_COLUMN,
                                       VALVE_COLUMN, run.config.window, run.config.horizon)
        rng = np.random.default_rng(0)
        anchors = anchors[rng.choice(len(anchors), min(N_ANCHORS, len(anchors)), replace=False)]
        w = extract_windows(cache, anchors, features, TARGET_COLUMN, VALVE_COLUMN,
                            run.config.window, run.config.horizon)
        hist = torch.from_numpy(w['history']).to(DEVICE)
        fv = torch.from_numpy(w['future_valve']).to(DEVICE)
        bv = torch.from_numpy(w['baseline_valve']).to(DEVICE)
        with torch.no_grad():
            out = model(hist, fv, bv)
        gain = out['gain'].cpu().numpy().ravel()
        tau = out['tau'].cpu().numpy()
        tau_min, tau_max = 1.5, 120.0
        slopes = None
        if run.config.opening_map == 'monotone':
            kv = model.opening_map.knot_values().detach().cpu().numpy()
            knots = model.opening_map.knots.detach().cpu().numpy()
            slopes = (np.diff(kv) / np.maximum(np.diff(knots), 1e-9)).tolist()
        rate_mean = None
        if run.config.rate_branch:
            rng2 = np.random.default_rng(1)
            a2 = anchors[rng2.choice(len(anchors), min(64, len(anchors)), replace=False)]
            w2 = extract_windows(cache, a2, features, TARGET_COLUMN, VALVE_COLUMN,
                                 run.config.window, run.config.horizon)
            with torch.no_grad():
                out2 = model(torch.from_numpy(w2['history']).to(DEVICE),
                             torch.from_numpy(w2['future_valve']).to(DEVICE),
                             torch.from_numpy(w2['baseline_valve']).to(DEVICE))
            if out2.get('rate_gain') is not None:
                rate_mean = float(out2['rate_gain'].cpu().numpy().mean())
        s = {
            'run_id': run.run_id, 'config': run.config.config_id, 'side': run.side, 'seed': run.seed,
            'gain_mean': float(gain.mean()), 'gain_p05': float(np.quantile(gain, .05)),
            'gain_p95': float(np.quantile(gain, .95)),
            'gain_near_zero': float(np.mean(np.abs(gain) < 1e-3)),
            'tau_mean': float(tau.mean()), 'tau_p05': float(np.quantile(tau, .05)),
            'tau_p95': float(np.quantile(tau, .95)),
            'tau_at_min': float(np.mean(tau <= tau_min * 1.01)),
            'tau_at_max': float(np.mean(tau >= tau_max * 0.99)),
            'map_slopes': slopes,
            'rate_mean': rate_mean,
        }
        summaries.append(s)
        print(f"{run.run_id:<40} gain={s['gain_mean']:+.4f} (near0 {s['gain_near_zero']:.0%})  "
              f"tau={s['tau_mean']:6.1f}s (min {s['tau_at_min']:.0%}/max {s['tau_at_max']:.0%})  "
              f"rate={'n/a' if rate_mean is None else f'{rate_mean:.4f}'}")
    with open('results/phase3_5/param_summary_validation.json', 'w') as f:
        json.dump(summaries, f, indent=1)
    print(f'\nsaved: results/phase3_5/param_summary_validation.json ({len(summaries)} runs)')

if __name__ == '__main__':
    main()
