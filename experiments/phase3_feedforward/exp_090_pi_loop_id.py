#!/usr/bin/env python3
"""
exp_090_pi_loop_id.py — 监督模式副回路辨识 v2: PI式误差驱动模型 (2026-08-04)
============================================================================
现场温度 PI 的输入输出模型: ΔV2_t = Σa_i·ΔV2_{t-i} + Σb_j·e_{t-j}, e = SP − T
  - e 含温度反馈 (比开环 ARX(SP→V2) 完整: SP 阶跃 → e 突变 → 大幅动作 → T 变 → e 回)
  - 方向/增益/时标全从数据学 (不假设 Kp/Ki)
  - 先合成验证系数恢复; 阶跃响应: SP +1°C → ΔV2/e → 稳态; 输出 图+表
用法: python exp_090_pi_loop_id.py [--smoke]
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
NMAX = 800000 if not SMOKE else 40000
print(f"[cfg] PI式副回路 ARX({P},{Q}): ΔV2 ~ e(SP−T) | 样本 {NMAX}")

def fit_arx(u, y, p=P, q=Q):
    T = len(y)
    X, Y = [], []
    for t in range(max(p, q), T):
        row = [y[t - i] for i in range(1, p + 1)] + [u[t - j] for j in range(0, q + 1)]
        X.append(row); Y.append(y[t])
    X = np.array(X); Y = np.array(Y)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    yhat = X @ coef
    r2 = 1 - float(((Y - yhat) ** 2).sum()) / float(((Y - Y.mean()) ** 2).sum())
    return coef[:p], coef[p:], r2

# ===== 合成验证 (e→ΔV2, 已知 PI 行为) =====
rng = np.random.default_rng(0)
a_true = np.array([0.30, -0.05, 0.02, -0.01])
b_true = np.array([-0.15, -0.20, -0.12, -0.05, -0.02, -0.01, 0.0])  # e>0(需热)→关阀(ΔV2<0)
n_syn = 200000
sp_syn = 567 + (rng.random(n_syn) < 0.003).astype(float) * rng.normal(0, 0.5, n_syn)
t_syn = 567 + np.cumsum(rng.normal(0, 0.05, n_syn))  # 温度慢漂
e_syn = sp_syn - t_syn
y_syn = np.zeros(n_syn)
for t in range(60, n_syn):
    y_syn[t] = (a_true @ y_syn[t-1:t-1-P:-1]) + (b_true @ e_syn[t-Q:t+1][::-1]) + rng.normal(0, 0.02)
a_s, b_s, r2_s = fit_arx(e_syn, y_syn)
print(f"[syn] R²={r2_s:.4f} | a 恢复 {np.round(a_s,3)} vs {a_true} | b 恢复 {np.round(b_s,3)} vs {b_true}")
if not (np.allclose(a_s, a_true, atol=0.05) and np.allclose(b_s, b_true, atol=0.05)):
    print("[syn] FAIL — 停止"); sys.exit(1)
print("[syn] PASS")

# ===== 真实数据: e = SP − T → ΔV2 =====
e = raw[:NMAX, I_SP] - raw[:NMAX, I_T]
dy = np.diff(raw[:NMAX, I_V2])
a, b, r2 = fit_arx(e[:-1], dy)
print(f"\n[real] R²={r2:.4f} | a={np.round(a,4)} | b={np.round(b,4)}")

# 阶跃响应: SP +1°C (e 阶跃 +1, 温度不动理想化)
def step_resp(a, b, steps=60):
    p, q = len(a), len(b)
    y = np.zeros(steps)
    for t in range(steps):
        acc = sum(a[i-1]*y[t-i] for i in range(1, p+1) if t-i >= 0)
        acc += sum(b[j] for j in range(q) if t-j >= 0)
        y[t] = acc
    return y
sr = step_resp(a, b)
G1 = b.sum() / (1 - a.sum())
t_ax = np.arange(len(sr)) * 10
t63 = next((i for i, v in enumerate(sr) if abs(v) >= 0.63*abs(G1)), None)
print(f"[real] 稳态 G(1)={G1:.4f} (%阀位/°C误差) | 方向: {'关阀(e>0)' if G1<0 else '开阀(e>0)'} | 63%时间 {t63*10 if t63 else None}s")
print(f"[real] 阶跃响应(前15步, e=+1°C): {' '.join(f'{v:+.3f}' for v in sr[:15])}")

# 时标/方向对照: e 与 ΔV2 的互相关 (e 领先时 ΔV2 响应)
lag = 20
L = len(dy)
cc = []
for k in range(lag):
    cc.append(np.corrcoef(e[:L-k], dy[k:])[0, 1])
print(f"[real] corr(e_t, ΔV2_{{{'+' if False else ''}t+k}}) 峰值 @ k={int(np.argmax(np.abs(cc)))}: {cc[np.argmax(np.abs(cc))]:+.3f} (负=误差大时关阀)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(t_ax, sr, 'o-', ms=3, lw=1.5, color='#c0504d')
axes[0].axhline(G1, color='gray', ls='--', lw=0.8, label=f'G(1)={G1:.3f}')
axes[0].axhline(0, color='gray', lw=0.7)
axes[0].set_title(f'PI-loop step response: e=+1°C -> V2 (ARX({P},{Q}))')
axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Valve V2 Δ (%)'); axes[0].legend(fontsize=8)
axes[1].plot(range(lag), cc, 'o-', ms=3)
axes[1].axhline(0, color='gray', lw=0.7)
axes[1].set_title('Cross-correlation e(t) vs ΔV2(t+k)')
axes[1].set_xlabel('Lag k (×10s)')
fig.tight_layout()
fig.savefig('figures/fig_pi_loop_id.png', dpi=170, bbox_inches='tight')
print('\nSaved: figures/fig_pi_loop_id.png')
