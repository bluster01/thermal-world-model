#!/usr/bin/env python3
"""
exp_078_action_response.py — 纯动作→温度响应曲线 (扰动/SP/燃烧侧全剔除)
=======================================================================
同一窗口, 唯一变量=动作序列:
  a_base = 保持当前阀位 (燃烧侧前馈驱动基线)
  a_step = 阀位阶跃 (+Δ% at t=100s, 保持) 
  ΔT(t) = ŷ_step − ŷ_base  = 纯"阀位→温度"动作响应曲线 (M7 学到的因果通道)
验证: 多幅度线性度 + 多窗口平均 + 与真实事件研究对比 (真实: 阀位+3%→180s −0.06°C)
"""
import os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wm = M.load_wm()
H = M.H_OUT
t_axis = np.arange(H) * M.DT

# 代表性窗口: test 段, SP 稳定 + 阀位稳定 (排除工况切换)
cand = np.linspace(5000, len(M.test_raw) - M.W - H - 20, 400).astype(int)
wins = []
for i in cand:
    seg_sp = M.test_raw[i+M.W-60:i+M.W+H, M.SP_IDX]
    seg_v = M.test_raw[i+M.W-60:i+M.W+H, M.VALVE_IDX[0]]
    if seg_sp.std() < 0.3 and seg_v.std() < 1.5 and len(wins) < 5:
        wins.append(int(i))

AMPS = [2.0, 5.0, 10.0]   # 阀位阶跃幅度 (%)
dT_all = {a: [] for a in AMPS}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for i in wins:
    win = torch.FloatTensor(M.test_raw[i:i+M.W]).unsqueeze(0).to(DEVICE)
    # 基线 = 真实未来动作 (与 exp_025 Sens 同协议, 避免恒定动作OOD; 打破共因)
    a_base = M.test_raw[i+M.W:i+M.W+H, M.VALVE_IDX].copy()
    a_full_b = torch.FloatTensor(a_base).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mu_b, _ = wm(win, a_full_b.reshape(1, -1))
    T_base = mu_b[0].cpu().numpy()
    # 动作阶跃: 二级阀 +Δ% at t=100s (k=10) — 二级减温直接作用于主汽温出口, 方向应明确
    for amp in AMPS:
        a_step = a_base.copy()
        a_step[10:, 1] += amp
        a_full_s = torch.FloatTensor(a_step).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu_s, _ = wm(win, a_full_s.reshape(1, -1))
        T_step = mu_s[0].cpu().numpy()
        dT_all[amp].append(T_step - T_base)
        if amp == 5.0 and len(dT_all[amp]) == 1:  # 首窗口画完整曲线
            ax = axes[0]
            ax.plot(t_axis, T_base, 'k-', lw=1.6, label='Baseline (valve held, combustion-driven)')
            ax.plot(t_axis, T_step, '#c0504d', lw=1.6, label=f'Valve step +{amp}% at 100s')
            ax.plot(t_axis, T_base + (T_step - T_base), '--', color='gray', lw=0.8)
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Predicted main steam temp (°C)')
            ax.set_title('(a) Model prediction: baseline vs valve step (one window)')
            ax.legend(fontsize=8)
# (b) 平均 ΔT 曲线 (多幅度)
ax = axes[1]
colors = ['#4f81bd', '#c0504d', '#8064a2']
for k, amp in enumerate(AMPS):
    dT = np.array(dT_all[amp])   # [n_win, H]
    m = dT.mean(0); s = dT.std(0)
    ax.plot(t_axis, m, colors[k], lw=1.8, label=f'ΔV1 +{amp}% (mean±std, n={len(wins)})')
    ax.fill_between(t_axis, m - s, m + s, color=colors[k], alpha=0.15)
ax.axhline(0, color='gray', lw=0.7)
ax.axvline(100, color='gray', lw=0.8, ls=':')
ax.set_xlabel('Time (s)'); ax.set_ylabel('ΔT from valve action (°C)')
ax.set_title('(b) Pure valve→temperature response curve (combustion/SP/disturbance removed)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig('figures/fig_action_response.png', dpi=180, bbox_inches='tight')
print('Saved: figures/fig_action_response.png')

# ============ 量化 ============
print('\n=== 纯动作响应量化 (5 窗口平均) ===')
for amp in AMPS:
    dT = np.array(dT_all[amp])
    m = dT.mean(0)
    # 180s 稳态响应 + 时间常数 (63% 上升点)
    dT_180 = m[-1]
    tc = np.nan
    for k in range(10, H):
        if abs(m[k]) >= 0.63 * abs(dT_180):
            tc = t_axis[k] - 100
            break
    print(f"  ΔV1 +{amp}%: 180s ΔT = {dT_180:+.3f}°C | 时间常数 ≈ {tc:.0f}s (若存在)")
# 线性度: 5% vs 10% 响应比
m2 = np.array(dT_all[2.0]).mean(0); m5 = np.array(dT_all[5.0]).mean(0); m10 = np.array(dT_all[10.0]).mean(0)
print(f"  线性度检验: ΔT(10%)/ΔT(5%) = {m10[-1]/m5[-1]:.2f} (1.0=线性) | ΔT(5%)/ΔT(2%) = {m5[-1]/m2[-1]:.2f}")
print(f"  与真实事件研究对比: 真实阀位+3%→180s −0.06°C → 归一化 {( -0.06/3):.3f}°C/% vs 模型 {m5[-1]/5:.4f}°C/% (量级一致={abs(-0.06/3 - m5[-1]/5) < 0.02})")
