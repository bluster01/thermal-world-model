#!/usr/bin/env python3
"""
exp_054_horizon_eval.py — exp_048 显著性补评估
====================================================
用已训练 checkpoint (h06/h09/h12/h18.pth, results/exp_048_horizon/checkpoints/)
重跑 50 轨迹 1200s 自回归闭环 rollout, 存 per-track MAE,
配对 Wilcoxon 检验: 120s 凸起 (1.322) vs 90s (1.264) vs 180s (1.200) 是偶然还是必然
协议与 exp_048 完全一致: 真实阀位动作 + 温度预测回填, 其余列真实, 50轨迹 seed42
用法: python exp_054_horizon_eval.py
"""
import os, sys, json, time
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import config as cfg
from experiments.phase1_dynamics.exp_025_unified_benchmark import (
    build_model, test_raw, VALVE_IDX, TARGET_IDX, H_OUT)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CK_DIR = 'results/exp_048_horizon/checkpoints'
N_TRACKS = 50
N_STEPS = 120
OUT = 'results/exp_054_horizon_eval'
os.makedirs(OUT, exist_ok=True)

W = cfg.WINDOW_SIZE

def sim_rollout(wm, track_idx, h, n_steps):
    """自回归闭环: 真实动作 + 温度预测回填 (exp_048 同款)"""
    win = torch.FloatTensor(test_raw[track_idx:track_idx + W]).unsqueeze(0).to(DEVICE)
    temps = []
    for k in range(n_steps):
        gi = track_idx + W + k
        a_real = torch.FloatTensor(test_raw[gi:gi + h, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, _ = wm(win, a_real)
            y1 = mu[0, 0].item()
        next_row = torch.FloatTensor(test_raw[gi]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        next_row[0, 0, TARGET_IDX] = y1
        win = torch.cat([win[:, 1:, :], next_row], 1)
        temps.append(y1)
    return np.array(temps)

np.random.seed(42)
N = len(test_raw)
starts = np.random.choice(range(N - W - H_OUT - N_STEPS - 50), N_TRACKS, replace=False)

results = {}
for h in [6, 9, 12, 18]:
    import experiments.phase1_dynamics.exp_025_unified_benchmark as E
    E.H_OUT = h
    model = build_model('M7').to(DEVICE).eval()
    ck = torch.load(f'{CK_DIR}/h{h:02d}.pth', map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict'])
    per = []
    for s in starts:
        pred = sim_rollout(model, s, h, N_STEPS)
        real = test_raw[s + W:s + W + N_STEPS, TARGET_IDX]
        per.append(float(np.abs(pred - real).mean()))
    per = np.array(per)
    results[str(h * 10)] = {'per_track_mae': [float(x) for x in per],
                            'mean': float(per.mean()), 'std': float(per.std())}
    print(f"H={h*10:>3}s: 1200s MAE {per.mean():.3f} ± {per.std():.3f}")

print("\n=== 配对 Wilcoxon (50轨迹) ===")
for a, b in [('90', '120'), ('120', '180'), ('90', '180'), ('60', '180')]:
    pa = np.array(results[a]['per_track_mae'])
    pb = np.array(results[b]['per_track_mae'])
    p = stats.wilcoxon(pa, pb).pvalue
    print(f"  {a}s vs {b}s: Δ={pb.mean()-pa.mean():+.3f}  p={p:.4f} {'**' if p<0.05 else ''}")

json.dump(results, open(f'{OUT}/horizon_eval.json', 'w'), indent=2)
print(f"\nSaved: {OUT}/horizon_eval.json")
