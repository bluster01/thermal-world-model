#!/usr/bin/env python3
"""
exp_092_bigstep_loop_id.py — 大信号副回路辨识 (2026-08-04)
============================================================
小信号 ARX 增益 (-0.17) 被平衡点附近波动主导, 低估真实增益 11 倍。
本脚本: 只在 |ΔSP|>1°C 大阶跃事件窗口内拟合 ΔV2_t = Σa·ΔV2 + Σb·e, e=SP−T
  - 大事件 = PI 大幅动作 = 大信号增益可辨识
  - 合成验证; 输出阶跃响应/增益/时标 + 图
用法: python exp_092_bigstep_loop_id.py [--smoke]
"""
import os, sys
import numpy as np
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
cols = E.NUMERIC_COLS
I_SP = cols.index('二级减温调节阀设定')
I_V2 = cols.index('二级减温调节门阀位')
I_T  = cols.index('末级过热器出口汽温')
raw = E.data_all
P, Q = 4, 6
PRE, POST = 5, 20          # 事件窗口: onset-5 .. onset+20
THR = 1.0                  # |ΔSP| 阈值
GAP = 30
N_EV = 300 if SMOKE else 900
print(f"[cfg] 大信号副回路: |ΔSP|>{THR}°C 事件窗口拟合 ARX({P},{Q}), {N_EV} 事件")

def fit_arx(u, y, p=P, q=Q):
    T = len(y)
    X, Y = [], []
    for t in range(max(p, q), T):
        X.append([y[t-i] for i in range(1, p+1)] + [u[t-j] for j in range(q+1)])
        Y.append(y[t])
    X = np.array(X); Y = np.array(Y)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    yhat = X @ coef
    r2 = 1 - float(((Y-yhat)**2).sum()) / float(((Y-Y.mean())**2).sum())
    return coef[:p], coef[p:], r2

# ===== 事件窗口拼接 =====
dsp = np.abs(np.diff(raw[:, I_SP]))
idxs = np.where(dsp > THR)[0] + 1
onsets = []
for i in idxs:
    if not onsets or i - onsets[-1] >= GAP:
        onsets.append(i)
onsets = onsets[:N_EV]
print(f"[events] 大阶跃 {len(onsets)} 个")
Us, Ys = [], []
for o in onsets:
    if o - PRE < 0 or o + POST >= len(raw):
        continue
    w0, w1 = o - PRE, o + POST
    e_w = raw[w0:w1, I_SP] - raw[w0:w1, I_T]
    v_w = raw[w0:w1, I_V2]
    Us.append(e_w); Ys.append(np.diff(v_w))
U = np.concatenate(Us); Y = np.concatenate(Ys)
print(f"[data] 窗口样本 {len(Y)}")

# ===== 合成验证 (事件数据上, 与真实管线同路径: 事件窗口拼接) =====
rng = np.random.default_rng(0)
a_true = np.array([0.30, -0.05, 0.02, -0.01])
b_true = np.array([-1.2, -0.8, -0.4, -0.2, -0.1, -0.05, 0.0])  # 大信号增益
n_syn = 500000
sp_syn = 567 + (rng.random(n_syn) < 0.002).astype(float) * rng.normal(0, 0.8, n_syn)
t_syn = 567 + np.cumsum(rng.normal(0, 0.05, n_syn))
e_syn = sp_syn - t_syn
y_syn = np.zeros(n_syn)
for t in range(60, n_syn):
    y_syn[t] = (a_true @ y_syn[t-1:t-1-P:-1]) + (b_true @ e_syn[t-Q:t+1][::-1]) + rng.normal(0, 0.02)
# 同路径: 事件窗口拼接
dsp_s = np.abs(np.diff(sp_syn))
idx_s = np.where(dsp_s > THR)[0] + 1
ons_s = []
for i in idx_s:
    if not ons_s or i - ons_s[-1] >= GAP:
        ons_s.append(i)
Us_s, Ys_s = [], []
for o in ons_s[:N_EV]:
    if o - PRE < 0 or o + POST >= n_syn:
        continue
    w0, w1 = o - PRE, o + POST
    Us_s.append(e_syn[w0:w1]); Ys_s.append(y_syn[w0:w1])  # y_syn 已是 ΔV2 (ARX 输出), 不 diff
U_s = np.concatenate(Us_s); Y_s = np.concatenate(Ys_s)
a_s, b_s, r2_s = fit_arx(U_s, Y_s)
def step_resp(a, b, steps=40):
    p, q = len(a), len(b)
    y = np.zeros(steps)
    for t in range(steps):
        acc = sum(a[i-1]*y[t-i] for i in range(1, p+1) if t-i >= 0)
        acc += sum(b[j] for j in range(q) if t-j >= 0)
        y[t] = acc
    return y
sr_true = step_resp(a_true, b_true)
sr_fit = step_resp(a_s, b_s)
# 阶跃响应曲线恢复度 (激励不足→系数非唯一, 但响应/增益可辨识 — 验证实际使用量)
resp_err = float(np.abs(sr_fit[:30] - sr_true[:30]).max())
G_fit = b_s.sum() / (1 - a_s.sum()); G_true = b_true.sum() / (1 - a_true.sum())
dir_ok = np.sign(sr_fit[3]) == np.sign(sr_true[3]) if abs(sr_true[3]) > 1e-6 else True
g_ok = abs(G_fit - G_true) / max(abs(G_true), 1e-9) < 0.10
ok = r2_s > 0.95 and g_ok and dir_ok
print(f"[syn] R²={r2_s:.4f} | G: 拟合 {G_fit:.3f} vs 真值 {G_true:.3f} (相对误差 {abs(G_fit-G_true)/max(abs(G_true),1e-9)*100:.1f}%) | 方向 {'PASS' if dir_ok else 'FAIL'} | {'PASS' if ok else 'FAIL'}")
if not ok:
    sys.exit(1)

# ===== 真实数据 =====
a, b, r2 = fit_arx(U, Y)
print(f"\n[real] R²={r2:.4f} | a={np.round(a,4)} | b={np.round(b,4)}")
G1 = b.sum() / (1 - a.sum())
def step_resp(a, b, steps=60):
    p, q = len(a), len(b)
    y = np.zeros(steps)
    for t in range(steps):
        acc = sum(a[i-1]*y[t-i] for i in range(1, p+1) if t-i >= 0)
        acc += sum(b[j] for j in range(q) if t-j >= 0)
        y[t] = acc
    return y
sr = step_resp(a, b)
t63 = next((i for i, v in enumerate(sr) if abs(v) >= 0.63*abs(G1)), None)
print(f"[real] 大信号增益 G(1)={G1:.3f} (%阀位/°C误差) | 方向 {'关阀' if G1<0 else '开阀'} | 63%时间 {t63*10 if t63 else None}s")
print(f"[real] 阶跃响应(前15步, e=+1°C): {' '.join(f'{v:+.2f}' for v in sr[:15])}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(np.arange(len(sr))*10, sr, 'o-', ms=3, lw=1.5, color='#c0504d')
axes[0].axhline(G1, color='gray', ls='--', lw=0.8, label=f'G(1)={G1:.2f}')
axes[0].axhline(0, color='gray', lw=0.7)
axes[0].set_title(f'Big-signal inner-loop: e=+1°C -> V2 (ARX({P},{Q}), |ΔSP|>1°C events)')
axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Valve V2 Δ (%)'); axes[0].legend(fontsize=8)
axes[1].bar(range(len(a)), a, label='a (ΔV2)')
axes[1].bar(range(len(a), len(a)+len(b)), b, label='b (e)')
axes[1].axhline(0, color='gray', lw=0.7); axes[1].legend(fontsize=8)
axes[1].set_title('ARX coefficients (big-signal)')
fig.tight_layout()
fig.savefig('figures/fig_bigstep_loop_id.png', dpi=170, bbox_inches='tight')
print('Saved: figures/fig_bigstep_loop_id.png')
