#!/usr/bin/env python3
"""
exp_095_sandbox_sp_validation.py — 世界模型仿真沙盒: SP规划动作下的预测精度 (2026-08-04)
==========================================================================================
核心问题: 世界模型作为沙盒, 在运行人员 SP 规划动作 (阶跃) 下, 预测温度 vs 现场实际温度误差?
  - 557 个 SP 阶跃事件 (稳定工况, exp_093 筛选)
  - M5 从 onset 前 96 步状态 + 实际 SP/阀位轨迹 → 预测 180s 温度 vs 实际
  - 指标: 预测 RMSE (SP变化时段 vs 平稳基线) | 方向正确率 | 跟随捕捉率 (温度跟随 SP 是否被预测)
用法: python exp_095_sandbox_sp_validation.py [--smoke]
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase2_mpc'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

SMOKE = '--smoke' in sys.argv
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
raw = E.data_all
N = len(raw)
W = 96
H_OUT = 18
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_V  = E.NUMERIC_COLS.index('二级减温调节门阀位')

# ===== 加载 M5 =====
ck = torch.load('results/exp_025_M5/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
wm = E.build_model('M5').to(DEVICE).eval()
wm.load_state_dict(ck['model_state_dict'])
print('[load] M5 OK')

# ===== 事件筛选 (同 exp_093: |ΔSP|>1°C, 间隔60步, 工况稳定, SP保持) =====
dsp = np.abs(np.diff(raw[:, I_SP]))
onsets = []
for i in np.where(dsp > 1.0)[0] + 1:
    if not onsets or i - onsets[-1] >= 60:
        onsets.append(i)
stable = [o for o in onsets if o + 60 < N and np.abs(np.diff(raw[max(0,o-20):min(N,o+20), E.NUMERIC_COLS.index('机组负荷')])).max() <= 3.0]
kept = [o for o in stable if np.abs(raw[o:o+61, I_SP] - raw[o, I_SP]).max() <= 0.3]
events = kept[:200] if SMOKE else kept
print(f"[events] SP阶跃+稳定+SP保持: {len(events)}")

# ===== 平稳基线事件 (随机非阶跃窗口) =====
rng = np.random.default_rng(42)
n_ev = len(events)
calm = []
for _ in range(n_ev):
    while True:
        c = int(rng.integers(W + 60, N - 60))
        if np.abs(np.diff(raw[c-20:c+20, I_SP])).max() <= 0.15 and c not in events:
            calm.append(c); break
print(f"[events] 平稳基线: {len(calm)}")

def predict_temps(onset, lead):
    """从 onset−W−lead 步起点 (窗口末=阶跃前), 给定实际 SP/阀位轨迹, 预测 18 步温度"""
    s = onset - W - lead
    if s < 0 or s + W + H_OUT >= N:
        return None
    win = torch.FloatTensor(raw[s:s+W]).unsqueeze(0).to(DEVICE)
    a = raw[s+W:s+W+H_OUT, E.NUMERIC_COLS.index('一级减温调节门阀位'):E.NUMERIC_COLS.index('二级减温调节门阀位')+1]
    if len(a) < H_OUT:
        a = np.pad(a, ((0, H_OUT-len(a)), (0, 0)), mode='edge')
    a_fut = torch.FloatTensor(a).reshape(1, -1).to(DEVICE)
    with torch.no_grad():
        mu, _ = wm(win, a_fut)
    return mu[0].cpu().numpy()  # [H_OUT]

# ===== 预测精度: 起点 = onset (阶跃时刻) =====
errs_on, errs_lead, errs_calm = [], [], []
dir_ok, follow_ok, follow_actual = [], [], []
for o in events:
    # onset 起点
    p = predict_temps(o, 0)
    if p is None: continue
    actual = raw[o:o+H_OUT, I_T]
    errs_on.append(np.abs(p - actual).mean())
    # onset 前 90s 起点 (现场前馈视角: 提前预测)
    p2 = predict_temps(o, 9)
    if p2 is not None:
        errs_lead.append(np.abs(p2 - actual).mean())
    # 方向正确率: SP 阶跃后 180s 温度净变化方向
    d_sp = raw[o, I_SP] - raw[o-1, I_SP]
    d_act = actual[-1] - raw[o-1, I_T]
    d_pred = p[-1] - raw[o-1, I_T]
    dir_ok.append(1 if np.sign(d_pred) == np.sign(d_act) else 0)
    # 跟随捕捉: 现场温度跟随 SP (|T600−SP|<0.3), 预测是否也显示跟随
    follow_actual.append(1 if abs(raw[o+30, I_T] - raw[o, I_SP]) < 0.3 else 0)
    follow_ok.append(1 if abs(p[-1] - raw[o, I_SP]) < 0.3 else 0)
for c in calm:
    p = predict_temps(c, 0)
    if p is None: continue
    errs_calm.append(np.abs(p - raw[c:c+H_OUT, I_T]).mean())

errs_on = np.array(errs_on); errs_lead = np.array(errs_lead); errs_calm = np.array(errs_calm)
dir_ok = np.array(dir_ok); follow_ok = np.array(follow_ok); follow_actual = np.array(follow_actual)
print(f"\n===== 沙盒预测精度 (180s 预测) =====")
print(f"  SP阶跃场景 (onset起点): RMSE {errs_on.mean():.3f}°C (中位 {np.median(errs_on):.3f}, p90 {np.percentile(errs_on,90):.3f})")
print(f"  SP阶跃场景 (提前90s起点): RMSE {errs_lead.mean():.3f}°C (中位 {np.median(errs_lead):.3f})")
print(f"  平稳基线: RMSE {errs_calm.mean():.3f}°C (中位 {np.median(errs_calm):.3f})")
print(f"  方向正确率 (SP阶跃后温度趋势): {dir_ok.mean()*100:.0f}%")
print(f"  现场实际跟随 SP (300s): {follow_actual.mean()*100:.0f}% | 预测显示跟随 (180s): {follow_ok.mean()*100:.0f}%")

# ===== 方向筛选: SP 驱动 vs 响应 (共因) 事件 =====
# SP 驱动事件: 实际温度变化方向与 SP 阶跃方向一致 (SP 降→温度降) — 因果可预测
# SP 响应事件: 温度逆 SP 走 (运行人员响应工况的干预, 共因) — 模型无法预测 (不知道未来工况)
drv_ok, resp_ok, drv_err, resp_err = [], [], [], []
for o in events:
    d_sp = raw[o, I_SP] - raw[o-1, I_SP]
    d_act = raw[o+H_OUT-1, I_T] - raw[o-1, I_T]
    is_drive = (np.sign(d_act) == np.sign(d_sp)) and abs(d_sp) > 0.3
    p = predict_temps(o, 0)
    if p is None: continue
    err = np.abs(p - raw[o:o+H_OUT, I_T]).mean()
    if is_drive:
        drv_ok.append(o); drv_err.append(err)
    else:
        resp_ok.append(o); resp_err.append(err)
drv_err = np.array(drv_err); resp_err = np.array(resp_err)
print(f"\n===== SP 驱动 vs 响应事件分类 =====")
print(f"  SP驱动事件 (温度随SP方向, n={len(drv_ok)}): 预测 RMSE {drv_err.mean():.3f}°C (中位 {np.median(drv_err):.3f})")
print(f"  SP响应事件 (温度逆SP, n={len(resp_ok)}): 预测 RMSE {resp_err.mean():.3f}°C (中位 {np.median(resp_err):.3f})")

# ===== 图 =====
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].hist(errs_on, bins=30, alpha=0.7, color='#c0504d', label=f'SP step (n={len(errs_on)})')
axes[0].hist(errs_calm, bins=30, alpha=0.6, color='#4f81bd', label=f'Calm (n={len(errs_calm)})')
axes[0].axvline(errs_on.mean(), color='#c0504d', ls='--', lw=1.2)
axes[0].axvline(errs_calm.mean(), color='#4f81bd', ls='--', lw=1.2)
axes[0].set_xlabel('180s prediction RMSE (°C)'); axes[0].set_ylabel('Count')
axes[0].set_title('Prediction error distribution (sandbox vs field)')
axes[0].legend(fontsize=8)
axes[1].bar(['SP-step\n(all)', 'SP-step\n(90s lead)', 'Calm\nbaseline', 'SP-drive', 'SP-response'],
            [errs_on.mean(), errs_lead.mean(), errs_calm.mean(), drv_err.mean(), resp_err.mean()],
            color=['#c0504d', '#e8a04c', '#4f81bd', '#2e8b57', '#8b0000'])
axes[1].set_ylabel('Mean prediction RMSE (°C)')
axes[1].set_title('Sandbox accuracy: SP-driven vs SP-responsive events')
for i, v in enumerate([errs_on.mean(), errs_lead.mean(), errs_calm.mean(), drv_err.mean(), resp_err.mean()]):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
# 典型事件: 预测 vs 实际 (驱动事件)
n_show = 0
for o in drv_ok[:6]:
    p = predict_temps(o, 0)
    if p is None: continue
    t_ax = np.arange(H_OUT) * 10
    axes[2].plot(t_ax, raw[o:o+H_OUT, I_T], color='#4f81bd', lw=1.2)
    axes[2].plot(t_ax, p, color='#c0504d', lw=1.2, ls='--')
    axes[2].axhline(raw[o, I_SP], color='gray', lw=0.6, ls=':')
    n_show += 1
axes[2].set_xlabel('Time since SP step (s)'); axes[2].set_ylabel('Outlet temp (°C)')
axes[2].set_title(f'Field (solid) vs sandbox (dashed) — SP-driven events (n={n_show})')
fig.tight_layout()
fig.savefig('figures/fig_sandbox_sp_validation.png', dpi=170, bbox_inches='tight')
print('\nSaved: figures/fig_sandbox_sp_validation.png')
