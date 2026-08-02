#!/usr/bin/env python3
"""
11 工况分工况评估 — M7 (Direct WM, β 固定 -0.3) 最终模型
=================================================================
复用 exp_019 的工况分类 (TCN-Improved-GRU evaluate_by_condition.py),
对 M7 做 test 集分工况 rollout 精度 + 敏感性 + σ 校准。
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from collections import Counter

WM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WM_ROOT, 'src'))
sys.path.insert(0, WM_ROOT)
import config as cfg
from data_loader import load_raw_data
from experiments.phase1_dynamics.exp_025_unified_benchmark import (
    build_model, test_raw, VALVE_IDX, TARGET_IDX, H_OUT, eval_sensitivity, eval_sigma_calib)

# TCN 项目分类函数
TCN_ROOT = '/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU'
sys.path.insert(0, TCN_ROOT)
sys.path.insert(0, os.path.join(TCN_ROOT, 'utils'))
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from evaluate_by_condition import classify_conditions

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COND_CN = {
    'steady': '稳态', 'wcr_mismatch': '水煤比失配', 'load_reversal': '负荷反转',
    'ramp_up': '升负荷', 'ramp_down': '降负荷', 'soot_blow': '锅炉吹灰',
    'aux_steam': '辅汽/抽汽扰动', 'fast_ramp_up': '快速升负荷',
    'fast_ramp_down': '快速降负荷', 'spray_oscillation': '减温水波动',
    'large_oscillation': '大幅波动-其他',
}

# ===== 数据 =====
csv_path = os.path.join(cfg.DATA_DIR, cfg.TRAIN_FILE)
df = pd.read_csv(csv_path)
if 'date' in df.columns:
    df.set_index('date', inplace=True)

state_data, delta_actions, valve_abs = load_raw_data()
raw_data = np.concatenate([state_data, valve_abs], axis=1)
n_total = len(raw_data)
n_train = int(n_total * 0.70); n_val_end = int(n_total * 0.85)
test_start = n_val_end
N = len(raw_data) - n_val_end  # test 长度 (应与 test_raw 一致)

print("工况分类中 (全数据, 约1-2min)...")
condition_labels, condition_stats, load_rate = classify_conditions(df)
print("分类完成")

# ===== 加载 M7 =====
model = build_model('M7').to(DEVICE).eval()
ck = torch.load("results/exp_025_M7/checkpoints/best_model.pth",
                map_location=DEVICE, weights_only=True)
model.load_state_dict(ck['model_state_dict']); model.eval()
print(f"模型: M7 best@{ck['epoch']}")

# ===== 分工况 rollout 评估 =====
W, H = cfg.WINDOW_SIZE, H_OUT
np.random.seed(42)
idxs = np.random.choice(range(N - W - H), 2000, replace=False)

def sample_condition(global_start):
    labels = condition_labels[global_start + W : global_start + W + H]
    cnt = Counter(labels)
    non_steady = [lbl for lbl in cnt if lbl != 'steady']
    if non_steady:
        return max(non_steady, key=lambda x: cnt[x])
    return 'steady'

print(f"\n评估 {len(idxs)} 样本 (H={H})...")
errs = {}
for j, i in enumerate(idxs):
    gi = test_start + i
    cond = sample_condition(gi)
    xh = torch.FloatTensor(test_raw[i:i+W]).unsqueeze(0).to(DEVICE)
    af = torch.FloatTensor(test_raw[i+W:i+W+H, VALVE_IDX]).unsqueeze(0).to(DEVICE)
    tt = test_raw[i+W:i+W+H, TARGET_IDX]
    with torch.no_grad():
        mu, _ = model(xh, af)
        e = np.abs(mu[0].cpu().numpy() - tt)
    errs.setdefault(cond, []).append((e[0], e[-1]))
    if (j+1) % 500 == 0:
        print(f"  {j+1}/{len(idxs)}")

print("\n" + "="*80)
print("11 工况 rollout 评估 (M7, test 集, H=18)")
print("="*80)
print(f"{'工况':<14} {'n':>5} {'Step0':>8} {'Step17':>8} {'×Grow':>7} {'占比':>6}")
rows = []
for cond, lst in sorted(errs.items(), key=lambda x: -len(x[1])):
    e0 = np.mean([x[0] for x in lst])
    e17 = np.mean([x[1] for x in lst])
    rows.append({'condition': cond, 'n': len(lst), 'step0': e0, 'step17': e17,
                 'growth': e17/e0 if e0 > 0 else float('nan'),
                 'ratio': len(lst)/len(idxs)*100})
    print(f"{COND_CN.get(cond, cond):<14} {len(lst):>5} {e0:>8.3f} {e17:>8.3f} "
          f"{e17/e0 if e0>0 else float('nan'):>7.2f} {len(lst)/len(idxs)*100:>5.1f}%")

all0 = np.mean([x[0] for lst in errs.values() for x in lst])
all17 = np.mean([x[1] for lst in errs.values() for x in lst])
print(f"\n总体: step0={all0:.3f} step17={all17:.3f} ×{all17/all0:.1f}")

result = {'model': 'M7', 'n_samples': len(idxs), 'H': H,
          'per_condition': rows, 'overall': {'step0': float(all0), 'step17': float(all17)}}
with open("results/exp_025_M7_conditions.json", 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=float)
print("\nSaved: results/exp_025_M7_conditions.json")
