#!/usr/bin/env python3
"""
Phase 1 收尾 — ① σ 校准评估 ② 1b 成功标准验证 + persistence baseline
=====================================================================
审稿人 R3-M1: σ 校准系统性失败 (|error|/σ=0.34→5.28, σ随步反缩)。
审稿人 1b: 成功标准 "H=18 < 2×H=1" 从未形式化验证; 无 persistence 对照。

评测模型: L3_W1_l0.00 (K5无正则, 当前最物理模型) + Direct WM 40列 (exp_023, 新最优)

σ 校准: NLL 模型输出的 σ 是否匹配实际 |error| (理想 |error|/σ ≈ 1.0)
persistence: ŝ_{t+k} = s_t (温度不变) — 时序预测的底线 baseline
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from data_loader import load_raw_data

import experiments.phase1_dynamics.exp_016_ablation_sweep as exp016
exp016.LAGS = [0, 3, 6, 9]
exp016.N_LAGS = len(exp016.LAGS)
from experiments.phase1_dynamics.exp_016_ablation_sweep import WorldModel_Lag

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W, H = cfg.WINDOW_SIZE, 18

state_data, delta_actions, valve_abs = load_raw_data()
raw_data = np.concatenate([state_data, valve_abs], axis=1)
n_total = len(raw_data)
n_val_end = int(n_total * 0.85)
test_data = raw_data[n_val_end:]

# ============ 1. Persistence baseline (1b) ============
print("="*70)
print("1b 成功标准验证 + Persistence baseline")
print("="*70)
np.random.seed(42)
Nt = len(test_data)
idxs = np.random.choice(range(Nt - W - H), 500, replace=False)

persist_err = np.zeros((len(idxs), H))
for j, i in enumerate(idxs):
    s0 = test_data[i + W - 1, cfg.TARGET_IDX]  # 窗口末位真实温度
    tt = test_data[i + W:i + W + H, cfg.TARGET_IDX]
    persist_err[j] = np.abs(s0 - tt)  # ŝ_{t+k} = s_t

pm = persist_err.mean(0)
print(f"\nPersistence (ŝ=s_t) rollout MAE:")
print(f"  step0={pm[0]:.3f} step8={pm[8]:.3f} step17={pm[-1]:.3f} ×{pm[-1]/pm[0]:.1f}")

# ============ 2. σ 校准 (L3_W1_l0.00, NLL 模型) ============
print("\n" + "="*70)
print("σ 校准: |error|/σ (理想=1.0) — L3_W1_l0.00")
print("="*70)
ck = torch.load("results/exp_016_L3_W1_l0.00/checkpoints/best_model.pth",
                map_location=DEVICE, weights_only=True)
model = WorldModel_Lag().to(DEVICE)
model.load_state_dict(ck['model_state_dict']); model.eval()

abs_err = np.zeros((len(idxs), H))
sigma_vals = np.zeros((len(idxs), H))
with torch.no_grad():
    for j, i in enumerate(idxs):
        sw = test_data[i:i+W, :cfg.N_STATE]; aw = test_data[i:i+W, cfg.N_STATE:]
        xt = torch.FloatTensor(np.concatenate([sw, aw], 1)).unsqueeze(0).to(DEVICE)
        fa = torch.FloatTensor(test_data[i+W:i+W+H, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
        tt = test_data[i+W:i+W+H, cfg.TARGET_IDX]
        tr = model.rollout(xt, fa, mode='sliding')  # [1, H, 11]
        # σ: 用单步 forward 逐步取 logvar (rollout 不返回 sigma)
        ss = sw.copy(); aa = aw.copy()
        sig_list = []
        for t in range(H):
            xt_t = torch.FloatTensor(np.concatenate([ss, aa], 1)).unsqueeze(0).to(DEVICE)
            mu_t, lv_t = model(xt_t)
            sig_list.append(torch.exp(lv_t[0, cfg.TARGET_IDX]*0.5).item())
            ss = np.concatenate([ss[1:], tr[0, t, :cfg.N_STATE].cpu().numpy().reshape(1,-1)], 0)
            aa = np.concatenate([aa[1:], fa[0, t:t+1].cpu().numpy()], 0)
        abs_err[j] = np.abs(tr[0,:,cfg.TARGET_IDX].cpu().numpy() - tt)
        sigma_vals[j] = np.array(sig_list)

ratio = abs_err / (sigma_vals + 1e-8)
print(f"\n{'step':<6}{'|err|':>8}{'σ':>8}{'|err|/σ':>9}")
for s in [0, 4, 8, 12, 17]:
    print(f"{s:<6}{abs_err.mean(0)[s]:>8.3f}{sigma_vals.mean(0)[s]:>8.3f}{ratio.mean(0)[s]:>9.2f}")
print(f"\n  σ 随步: {sigma_vals.mean(0)[0]:.3f} → {sigma_vals.mean(0)[-1]:.3f} "
      f"({'收缩' if sigma_vals.mean(0)[-1] < sigma_vals.mean(0)[0] else '扩张'})")
print(f"  |err|/σ 全程均值: {ratio.mean():.2f} (理想 1.0)")

# ============ 3. Direct WM 40列 σ 校准 (exp_023) ============
print("\n" + "="*70)
print("σ 校准: |error|/σ — Direct WM 40列 (exp_023, 新最优)")
print("="*70)
import pandas as pd
import experiments.phase1_dynamics.exp_023_direct_aligned as exp23

ck2 = torch.load("results/exp_023_direct_aligned/checkpoints/best_model.pth",
                 map_location=DEVICE, weights_only=True)
m2 = exp23.DirectAligned().to(DEVICE)
m2.load_state_dict(ck2['model_state_dict']); m2.eval()

df_full = pd.read_csv(os.path.join(cfg.DATA_DIR, cfg.TRAIN_FILE))
NUMERIC_COLS = [c for c in df_full.columns if c != 'date']
data_all = df_full[NUMERIC_COLS].values.astype(np.float32)
data_all = np.nan_to_num(data_all, nan=0.0)
n_val2 = int(len(data_all) * 0.85)
test2 = data_all[n_val2:]
TARGET_IDX2 = NUMERIC_COLS.index('末级过热器出口汽温')
VALVE_IDX2 = [NUMERIC_COLS.index('一级减温调节门阀位'), NUMERIC_COLS.index('二级减温调节门阀位')]

abs_err2 = np.zeros((len(idxs), H))
sigma_vals2 = np.zeros((len(idxs), H))
with torch.no_grad():
    for j, i in enumerate(idxs):
        xh = torch.FloatTensor(test2[i:i+W]).unsqueeze(0).to(DEVICE)
        af = torch.FloatTensor(test2[i+W:i+W+H, VALVE_IDX2]).unsqueeze(0).to(DEVICE)
        mu, lv = m2(xh, af)
        sig2 = torch.exp(lv * 0.5)
        abs_err2[j] = np.abs(mu[0].cpu().numpy() - test2[i+W:i+W+H, TARGET_IDX2])
        sigma_vals2[j] = sig2[0].cpu().numpy()

ratio2 = abs_err2 / (sigma_vals2 + 1e-8)
print(f"\n{'step':<6}{'|err|':>8}{'σ':>8}{'|err|/σ':>9}")
for s in [0, 4, 8, 12, 17]:
    print(f"{s:<6}{abs_err2.mean(0)[s]:>8.3f}{sigma_vals2.mean(0)[s]:>8.3f}{ratio2.mean(0)[s]:>9.2f}")
print(f"\n  σ 随步: {sigma_vals2.mean(0)[0]:.3f} → {sigma_vals2.mean(0)[-1]:.3f}")
print(f"  |err|/σ 全程均值: {ratio2.mean():.2f} (理想 1.0)")

# ============ 保存 ============
result = {
    'persistence_mae': pm.tolist(),
    'l3_sigma_calib': {'ratio': ratio.mean(0).tolist(), 'sigma': sigma_vals.mean(0).tolist(),
                       'abs_err': abs_err.mean(0).tolist(), 'overall_ratio': float(ratio.mean())},
    'direct23_sigma_calib': {'ratio': ratio2.mean(0).tolist(), 'sigma': sigma_vals2.mean(0).tolist(),
                             'abs_err': abs_err2.mean(0).tolist(), 'overall_ratio': float(ratio2.mean())},
}
with open("results/exp_024_sigma_persistence.json", 'w') as f:
    json.dump(result, f, indent=2, default=float)
print("\nSaved: results/exp_024_sigma_persistence.json")

# 1b 成功标准: H=18 vs 2×H=1
print("\n" + "="*70)
print("1b 成功标准: H=18 < 2×H=1 ?")
print("="*70)
print(f"  L3_W1_l0.00: H=1={abs_err.mean(0)[0]:.3f}, H=18={abs_err.mean(0)[-1]:.3f}, "
      f"H18/H1={abs_err.mean(0)[-1]/abs_err.mean(0)[0]:.2f} "
      f"({'PASS' if abs_err.mean(0)[-1] < 2*abs_err.mean(0)[0] else 'FAIL'})")
print(f"  Persistence: H=1={pm[0]:.3f}, H=18={pm[-1]:.3f}, "
      f"H18/H1={pm[-1]/pm[0]:.2f}")
print(f"  模型 vs Persistence @H18: {abs_err.mean(0)[-1]:.3f} vs {pm[-1]:.3f} "
      f"({'优于' if abs_err.mean(0)[-1] < pm[-1] else '劣于'} persistence)")
