#!/usr/bin/env python3
"""
11 工况分工况评估 — 世界模型 L3_W1_l0.00 在 test 集各工况的 rollout 精度
=========================================================================
审稿人缺口: 数据号称 11 工况, 从未分工况评估 (论文需要"各工况误差分布"表)。

方法:
1. 复用 TCN-Improved-GRU 项目的 classify_conditions (evaluate_by_condition.py)
   — 物理机理分类, 11 类工况 (稳态/水煤比失配/负荷反转/升降负荷/吹灰/抽汽/...)
2. 对世界模型 test 集逐样本标注工况 (预测窗口内多数非稳态标签)
3. 每工况: rollout MAE @ H=18 + 起点/终点 + 样本数
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch

WM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WM_ROOT, 'src'))
sys.path.insert(0, WM_ROOT)
import config as cfg
from data_loader import load_raw_data

# TCN 项目分类函数
TCN_ROOT = '/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU'
sys.path.insert(0, TCN_ROOT)
sys.path.insert(0, os.path.join(TCN_ROOT, 'utils'))
from evaluate_by_condition import classify_conditions

# 世界模型滞后结构 (L3)
import experiments.phase1_dynamics.exp_016_ablation_sweep as exp016
exp016.LAGS = [0, 3, 6, 9]
exp016.N_LAGS = len(exp016.LAGS)
from experiments.phase1_dynamics.exp_016_ablation_sweep import WorldModel_Lag

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
test_data = raw_data[n_val_end:]

# 工况标注 (全数据逐行)
print("工况分类中 (全数据, 约1-2min)...")
condition_labels, condition_stats, load_rate = classify_conditions(df)
print("分类完成")

# ===== 加载模型 =====
ck = torch.load("results/exp_016_L3_W1_l0.00/checkpoints/best_model.pth",
                map_location=DEVICE, weights_only=True)
model = WorldModel_Lag().to(DEVICE)
model.load_state_dict(ck['model_state_dict']); model.eval()
print(f"模型: L3_W1_l0.00 best@{ck['epoch']}")

# ===== 分工况 rollout 评估 =====
W, H = cfg.WINDOW_SIZE, 18
N = len(test_data)
# test 集在原始数据中的起点
test_start = n_val_end

np.random.seed(42)
idxs = np.random.choice(range(N - W - H), 1000, replace=False)

# 每样本: 计算工况标签 (预测窗口内多数非稳态)
from collections import Counter

def sample_condition(global_start):
    """global_start: 样本在原始全数据的起始索引"""
    labels = condition_labels[global_start + W : global_start + W + H]
    cnt = Counter(labels)
    non_steady = [lbl for lbl in cnt if lbl != 'steady']
    if non_steady:
        return max(non_steady, key=lambda x: cnt[x])
    return 'steady'

print(f"\n评估 {len(idxs)} 样本 (H={H})...")
errs = {}   # condition -> list of (err0, err17)
for j, i in enumerate(idxs):
    gi = test_start + i  # 全局索引
    cond = sample_condition(gi)
    sw = test_data[i:i+W, :cfg.N_STATE]; aw = test_data[i:i+W, cfg.N_STATE:]
    xt = torch.FloatTensor(np.concatenate([sw, aw], 1)).unsqueeze(0).to(DEVICE)
    fa = torch.FloatTensor(test_data[i+W:i+W+H, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
    tt = test_data[i+W:i+W+H, cfg.TARGET_IDX]
    with torch.no_grad():
        tr = model.rollout(xt, fa, mode='sliding')
        e = np.abs(tr[0,:,cfg.TARGET_IDX].cpu().numpy() - tt)
    errs.setdefault(cond, []).append((e[0], e[-1]))
    if (j+1) % 250 == 0:
        print(f"  {j+1}/{len(idxs)}")

print("\n" + "="*80)
print("11 工况 rollout 评估 (test 集, H=18)")
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

# 全样本汇总
all0 = np.mean([x[0] for lst in errs.values() for x in lst])
all17 = np.mean([x[1] for lst in errs.values() for x in lst])
print(f"\n总体: step0={all0:.3f} step17={all17:.3f} ×{all17/all0:.1f}")

result = {'model': 'L3_W1_l0.00', 'n_samples': len(idxs), 'H': H,
          'per_condition': rows, 'overall': {'step0': float(all0), 'step17': float(all17)}}
with open("results/exp_019_condition_eval.json", 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=float)
print("\nSaved: results/exp_019_condition_eval.json")
