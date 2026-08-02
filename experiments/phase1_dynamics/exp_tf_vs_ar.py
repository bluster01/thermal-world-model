#!/usr/bin/env python3
"""
世界模型评测范式对照 — 为什么世界模型 MAE 高于直接多步预测模型
=================================================================
TCN-iTransformer-Prob (Exp-0) 是"直接多步预测": 96步历史 → 一次输出18步,
无误差累积 → avg 0.330 / step17 0.586。

世界模型是"自回归 rollout": 每步用上一步的预测喂回窗口, 误差逐级累积。

本脚本定量分离两个因素:
1. teacher-forcing (每步喂真实状态): 测模型单步精度上界 — 若接近 Exp-0,
   说明模型本身学得不差, 差距全来自自回归累积
2. 自回归 (标准 rollout): 现在的 0.767

对比: L3_W1_l0.00 (K5无正则) / L3_W1_l0.10 (有正则) / 017_B (K12)
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from data_loader import load_raw_data

import experiments.phase1_dynamics.exp_016_ablation_sweep as exp016
exp016.LAGS = [0, 3, 6, 9]
exp016.N_LAGS = len(exp016.LAGS)
from experiments.phase1_dynamics.exp_016_ablation_sweep import WorldModel_Lag

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state_data, delta_actions, valve_abs = load_raw_data()
raw_data = np.concatenate([state_data, valve_abs], axis=1)
n_total = len(raw_data)
n_val_end = int(n_total * 0.85)
test_data = raw_data[n_val_end:]

W, H = cfg.WINDOW_SIZE, 18


@torch.no_grad()
def eval_teacher_forcing(model, raw, n=500):
    """每步喂真实状态 (仅动作来自真实序列) — 单步精度上界"""
    model.eval(); N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-H), n, replace=False)
    err = np.zeros((n, H))
    for j, i in enumerate(idxs):
        sw = raw[i:i+W, :cfg.N_STATE]
        aw = raw[i:i+W, cfg.N_STATE:]
        for t in range(H):
            xt = torch.FloatTensor(np.concatenate([sw, aw], 1)).unsqueeze(0).to(DEVICE)
            mu, _ = model(xt)
            # 只预测下一步, 然后窗口用真实状态推进
            err[j, t] = abs(mu[0, cfg.TARGET_IDX].item() - raw[i+W+t, cfg.TARGET_IDX])
            sw = np.concatenate([sw[1:], raw[i+W+t:i+W+t+1, :cfg.N_STATE]], 0)
            aw = np.concatenate([aw[1:], raw[i+W+t:i+W+t+1, cfg.N_STATE:]], 0)
    return err.mean(0)


@torch.no_grad()
def eval_autoregressive(model, raw, n=500):
    """标准自回归 rollout (与 eval_rollout 相同)"""
    model.eval(); N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-H), n, replace=False)
    err = np.zeros((n, H))
    for j, i in enumerate(idxs):
        sw = raw[i:i+W, :cfg.N_STATE]; aw = raw[i:i+W, cfg.N_STATE:]
        xt = torch.FloatTensor(np.concatenate([sw, aw], 1)).unsqueeze(0).to(DEVICE)
        fa = torch.FloatTensor(raw[i+W:i+W+H, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
        tt = raw[i+W:i+W+H, cfg.TARGET_IDX]
        tr = model.rollout(xt, fa, mode='sliding')
        err[j] = np.abs(tr[0,:,cfg.TARGET_IDX].cpu().numpy()-tt)
    return err.mean(0)


def load_model(ckpt_path, lag_cls):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    m = lag_cls().to(DEVICE)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m


models = {
    'L3_W1_l0.00 (K5无正则)': 'results/exp_016_L3_W1_l0.00/checkpoints/best_model.pth',
    'L3_W1_l0.10 (K5有正则)': 'results/exp_016_L3_W1_l0.10/checkpoints/best_model.pth',
}

# exp_017 的模型类不同 (K12), 单独处理
import experiments.phase1_dynamics.exp_017_multistep_signreg as exp017
exp017.LAGS = [0, 3, 6, 9]
exp017.N_LAGS = len(exp017.LAGS)

print(f"{'模型':<28} {'TF step0':>9} {'TF step8':>9} {'TF step17':>9} {'AR step0':>9} {'AR step17':>9}")
for name, path in models.items():
    m = load_model(path, WorldModel_Lag)
    tf = eval_teacher_forcing(m, test_data)
    ar = eval_autoregressive(m, test_data)
    print(f"{name:<28} {tf[0]:>9.3f} {tf[8]:>9.3f} {tf[-1]:>9.3f} {ar[0]:>9.3f} {ar[-1]:>9.3f}")
    print(f"  TF曲线: " + " ".join(f"{x:.2f}" for x in tf))
    print(f"  AR曲线: " + " ".join(f"{x:.2f}" for x in ar))

# 017_B
m = load_model('results/exp_017_B/checkpoints/best_model.pth', exp017.WorldModel_Lag)
tf = eval_teacher_forcing(m, test_data)
ar = eval_autoregressive(m, test_data)
print(f"{'017_B (K12无正则)':<28} {tf[0]:>9.3f} {tf[8]:>9.3f} {tf[-1]:>9.3f} {ar[0]:>9.3f} {ar[-1]:>9.3f}")
print(f"  TF曲线: " + " ".join(f"{x:.2f}" for x in tf))
print(f"  AR曲线: " + " ".join(f"{x:.2f}" for x in ar))

print("\n参照 (直接多步预测, 无累积): TCN-iTransformer-Prob avg=0.330 step17=0.586")
