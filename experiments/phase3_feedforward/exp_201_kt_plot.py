#!/usr/bin/env python3
"""K(x)/τ(x) 可解释性: 干预分支增益/时间常数 vs 负荷 (校准后模型)。

验证: K 应全负 (流量↑→T↓), 量级合理 (-0.1°C/% 开度 ≈ 归一化 -0.035/%), 
且随工况变化 (负荷依赖的增益异质性)。
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_proj = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _proj)
sys.path.insert(0, os.path.join(_proj, 'experiments', 'phase1_dynamics'))

import causal_arch as CA
from exp_025_unified_benchmark import cfg as E_cfg, data_all, N_FEAT, TARGET_IDX

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = E_cfg.WINDOW_SIZE; H = 60; n_val_end = 601566
raw = data_all
N_MAX = len(raw) - W - H

m = CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H, intervention='phys', cumsum_out=False,
                        probabilistic=True, n_lag=2, free_head_type='mlp',
                        alpha_init=0.0, clamp_interv=15.0, k_init=0.05, integrate=False).to(DEVICE)
m.load_state_dict(torch.load('results/exp_201_valve_action/A1phys_valve_noff_s0_flow_lg0.2/best_gain.pth',
                             map_location=DEVICE))
m.eval()

rng = np.random.default_rng(0)
idxs = rng.integers(0, N_MAX, size=300)
K_list, tau_list, load_list = [], [], []
for i in idxs:
    x = torch.from_numpy(raw[i:i + W, :N_FEAT]).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, s_flat = m.encode(x)
        K, tau = m.interv.params(s_flat)
    K_list.append(K.item())
    tau_list.append(tau[0].cpu().numpy())
    load_list.append(float(np.nanmean(raw[i:i + W, 0])))   # 机组负荷

K = np.array(K_list); tau = np.array(tau_list); load = np.array(load_list)
# 反归一化 K 到开度空间: mu 反归一化 ×std, action 是 flow 单位
# K 的物理含义: °C(归一化) / flow-unit; 转 °C/%开度 需 ×ss×(dflow/dV)⁻¹
ss = m.revin._std[0, 0, TARGET_IDX].item()
dflow_dV_at_20 = np.log(50.0) * 50.0 ** (0.2 - 1.0) / 100.0   # 20% 开度处
K_open = K * ss * dflow_dV_at_20 * 1000   # m°C/%(开度)

print(f'K: mean={K_open.mean():.1f} med={np.median(K_open):.1f} m°C/%  neg={(K_open<0).mean():.1%}')
t_mean = tau.mean(0)
print(f'tau: mean={t_mean[0]:.1f}/{t_mean[1]:.1f} steps  ({(t_mean*10)[0]:.0f}s/{(t_mean*10)[1]:.0f}s 级联)')

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), dpi=150)
ax = axes[0]
ax.scatter(load, K_open, s=10, alpha=0.5, color='#c0392b')
ax.axhline(0, color='k', lw=0.6)
ax.axhspan(-130, -90, color='#2ecc71', alpha=0.15, label='SP-IV truth')
ax.set_xlabel('Load (MW)'); ax.set_ylabel('K (m°C/%, at 20% opening)')
ax.set_title('(a) Gain K(x) vs load — all negative,\nSP-IV truth range shaded', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(load, tau[:, 0] * 10, s=10, alpha=0.5, color='#2980b9', label='τ₁')
ax.scatter(load, tau[:, 1] * 10, s=10, alpha=0.5, color='#e67e22', label='τ₂')
ax.set_xlabel('Load (MW)'); ax.set_ylabel('τ (s)')
ax.set_title('(b) Time constants vs load\n(cascade n_lag=2)', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
ax.hist(K_open, bins=40, color='#c0392b', alpha=0.7)
ax.axvline(-100, color='k', ls='--', lw=0.8, label='SP-IV truth ≈ -100')
ax.axvline(0, color='k', lw=0.6)
ax.set_xlabel('K (m°C/%)'); ax.set_ylabel('count')
ax.set_title('(c) K distribution (n=300 windows)', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
out = 'results/exp_201_valve_action/fig_kt_interpret.png'
plt.savefig(out, bbox_inches='tight')
print('saved:', out)
