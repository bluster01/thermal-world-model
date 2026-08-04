#!/usr/bin/env python3
"""
exp_088_sp_v2_arx.py — 副回路辨识: SP(idx36) → 阀位(idx39) 因果响应 (监督模式)
=================================================================================
目的: 监督模式虚拟世界需要"MPC 输出 SP → 副回路 → 阀位"模型。
方法: 差分域 ARX (ΔV2_t = Σa_i·ΔV2_{t-i} + Σb_j·ΔSP_{t-j}) — 剔共因 (观测长程被工况主导, exp_087)
  - 差分消除趋势共因; 阶跃响应稳态 G(1)=(Σb)/(1-Σa)
  - 先合成验证系数恢复 (exp_083 教训: 滞后列索引易错位, 拟合前必须验证)
  - 输出: 阶跃响应曲线 + 时标/增益/方向表 + 图 (全英文)

用法: python exp_088_sp_v2_arx.py [--smoke]
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

SMOKE = '--smoke' in sys.argv
cols = E.NUMERIC_COLS
I_SP  = cols.index('二级减温调节阀设定')
I_V2  = cols.index('二级减温调节门阀位')
raw = E.data_all

P, Q = 4, 6        # ARX 阶数 (ΔV2 自回归 4 阶, ΔSP 输入 6 阶 = 60s 输入记忆)
NMAX = 800000 if not SMOKE else 40000
print(f"[cfg] ARX({P},{Q}) 差分域 ΔSP→ΔV2 | 样本 {NMAX}")

def fit_arx(u, y, p=P, q=Q):
    """差分域 ARX: y_t = Σ_{i=1..p} a_i y_{t-i} + Σ_{j=0..q} b_j u_{t-j}
    列索引: y[p-i : T-i]  (exp_083 教训: y[i:T-p+i] 是未来列!)"""
    T = len(y)
    X, Y = [], []
    for t in range(max(p, q), T):
        row = [y[t - i] for i in range(1, p + 1)] + [u[t - j] for j in range(0, q + 1)]
        X.append(row); Y.append(y[t])
    X = np.array(X); Y = np.array(Y)
    coef, res, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
    yhat = X @ coef
    ss_res = float(((Y - yhat) ** 2).sum()); ss_tot = float(((Y - Y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    a = coef[:p]; b = coef[p:]
    return a, b, r2

def step_response(a, b, steps=60):
    """单位阶跃响应 (u=1 从 t=0 起): 递归计算"""
    p, q = len(a), len(b)
    y = np.zeros(steps)
    for t in range(steps):
        acc = 0.0
        for i in range(1, p + 1):
            if t - i >= 0:
                acc += a[i - 1] * y[t - i]
        for j in range(q):
            if t - j >= 0:
                acc += b[j]
        y[t] = acc
    return y

# ===== 1. 合成验证 (系数恢复) =====
rng = np.random.default_rng(0)
a_true = np.array([0.60, -0.10, 0.05, -0.02])
b_true = np.array([0.0, 0.0, 0.12, 0.20, 0.15, 0.08, 0.03])  # 滞后 2 步起响应 (20s)
n_syn = 200000
u_syn = (rng.random(n_syn) < 0.002).astype(float) * rng.normal(0, 1, n_syn)  # 稀疏阶跃
y_syn = np.zeros(n_syn)
for t in range(60, n_syn):
    y_syn[t] = (a_true @ y_syn[t - 1:t - 1 - P:-1]) + (b_true @ u_syn[t - Q:t + 1][::-1]) + rng.normal(0, 0.02)
a_s, b_s, r2_s = fit_arx(u_syn, y_syn)
print(f"[syn] R²={r2_s:.4f} | a 恢复: {np.round(a_s,3)} vs {a_true} | b 恢复: {np.round(b_s,3)} vs {b_true}")
syn_ok = np.allclose(a_s, a_true, atol=0.05) and np.allclose(b_s, b_true, atol=0.05)
print(f"[syn] 系数恢复 {'PASS' if syn_ok else 'FAIL'}")
if not syn_ok:
    print("[syn] 合成验证失败, 停止 (拟合管线有 bug, 不能用于真实数据)")
    sys.exit(1)

# ===== 2. 真实数据: 差分域 ARX =====
du = np.diff(raw[:NMAX, I_SP])          # ΔSP
dy = np.diff(raw[:NMAX, I_V2])          # ΔV2
a, b, r2 = fit_arx(du, dy)
print(f"\n[real] ΔARX R²={r2:.4f} | a={np.round(a,4)} | b={np.round(b,4)}")

# 阶跃响应
sr = step_response(a, b, steps=60)
G1 = b.sum() / (1 - a.sum())
t_axis = np.arange(len(sr)) * 10
# 时标: 达 63% 稳态时间 (一阶等效) / 首次过 30% 峰值
pk = np.abs(sr).max()
t63 = next((i for i, v in enumerate(sr) if np.abs(v) >= 0.63 * abs(G1)), None)
t30 = next((i for i, v in enumerate(sr) if np.abs(v) >= 0.30 * pk), None)
print(f"[real] 稳态增益 G(1) = {G1:.4f} (°C SP → % 阀位)")
print(f"[real] 方向: {'关阀 (SP↑→V2↓)' if G1 < 0 else '开阀 (SP↑→V2↑)'} | 63%时间: {t63*10 if t63 else None}s | 30%峰值: {t30*10 if t30 else None}s")
print(f"[real] 阶跃响应(前15步, 1°C SP): {' '.join(f'{v:+.3f}' for v in sr[:15])}")

# ===== 3. 图 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(t_axis, sr, 'o-', ms=3, lw=1.5, color='#4f81bd')
axes[0].axhline(G1, color='gray', ls='--', lw=0.8, label=f'Steady-state G(1)={G1:.3f}')
axes[0].axhline(0, color='gray', lw=0.7)
axes[0].set_title(f'Unit SP step (1°C) -> valve V2 (ARX({P},{Q}), diff-domain)')
axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Valve V2 Δ (%)'); axes[0].legend(fontsize=8)
axes[1].plot(np.arange(len(a)) + 1, a, 's-', label='a (ΔV2 AR)')
axes[1].plot(np.arange(len(b)), b, 'o-', label='b (ΔSP input)')
axes[1].axhline(0, color='gray', lw=0.7)
axes[1].set_title('ARX coefficients'); axes[1].set_xlabel('Lag (×10s)'); axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig('figures/fig_sp_v2_arx.png', dpi=170, bbox_inches='tight')
print('\nSaved: figures/fig_sp_v2_arx.png')
