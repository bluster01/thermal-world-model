"""
exp_083_combustion_partial_id.py — 剔除燃烧侧扰动的系统辨识 (v2: 差分域)
================================================================
回答: 持续阶跃翻转是"燃烧侧扰动"还是"物理实效"?

v2 修正 (v1 问题: 近单位根AR吸收输入效应 / 准稳态窗口欠定):
  1. 差分域 ARX: Δy_t = Σa·Δy滞后 + Σb·Δu滞后 + Σc·Δd滞后
     → 输入=Δ阀位 (与世界模型动作定义一致), 阶跃响应=Δu脉冲累积
  2. 准稳态阈值自适应: 30步窗口极差 < 全数据分位数 (报告实际覆盖)
  3. 岭回归 (X'X+λI) 防欠定

三个对照 (同一协议):
  A. SISO-全数据: 仅阀位 → 燃烧侧混杂未剔除 (预期可能翻转)
  B. SISO-准稳态: 燃烧侧平稳窗口内 (天然剔除)
  C. MISO-全数据: 燃烧侧7测点进模型, 阀位通道=偏效应 (统计剔除)

若 B/C 阶跃响应为负 (开阀→降温, 60-90s滞后), 而 A 翻转 → 实锤燃烧侧混杂。
用法: python exp_083_combustion_partial_id.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = os.path.expanduser(
    "~/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT/A侧主汽温全数据03_cleaned_10s.csv"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

P = 12            # ARX阶数 (120s记忆, 覆盖60-90s滞后)
DT = 10.0         # 采样秒
W_STEADY = 30     # 准稳态窗口长度 (5分钟)
H_RESP = 30       # 阶跃响应长度 (300s)
RIDGE = 1e-3      # 岭正则

COMBUSTION_COLS = [
    '机组负荷', '主蒸汽压力', '机组负荷变化率', '主蒸汽流量',
    '未校正总煤量', '总二次风量', '主给水流量',
]
VALVE_COLS = ['一级减温调节门阀位', '二级减温调节门阀位']
Y_COL = '末级过热器出口汽温'


def ridge_arx(y, u, d, p=P, lam=RIDGE):
    """差分域 ARX 岭回归. y,u,d: [T]/[T,nu]/[T,nd] 已标准化+差分.
    返回 (theta, r2_diff, resid_std)."""
    T = len(y)
    nu = u.shape[1]
    nd = 0 if d is None else d.shape[1]
    D = np.zeros((T, nd)) if d is None else d
    cols = []
    for i in range(1, p + 1):
        cols.append(y[i:T - p + i])
    for j in range(nu):
        for i in range(1, p + 1):
            cols.append(u[i:T - p + i, j])
    for j in range(nd):
        for i in range(1, p + 1):
            cols.append(D[i:T - p + i, j])
    X = np.column_stack(cols + [np.ones(T - p)])
    yt = y[p:T]
    XtX = X.T @ X
    n = len(yt)
    XtX.flat[:: XtX.shape[0] + 1] += lam * np.trace(XtX) / n   # 岭
    theta = np.linalg.solve(XtX, X.T @ yt)
    yhat = X @ theta
    ss_res = float(np.sum((yt - yhat) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return theta, r2


def step_response(theta, nu, nd, u_idx, y_scale, dy_scale):
    """Δ阀位脉冲(→阀位阶跃 +1σ)的 ΔT 累积响应. 返回 [H_RESP] °C."""
    p = P
    dyhist = np.zeros(p)          # Δy_{t-1-i}
    duhist = np.zeros((p, nu))    # Δu_{t-1-i}
    ddhist = np.zeros((p, nd))
    duhist[0, u_idx] = 1.0        # t=-1 施加 Δu 脉冲
    resp = []
    cum = 0.0
    for _ in range(H_RESP):
        x = np.concatenate([dyhist.ravel(),
                            duhist.T.ravel(),
                            ddhist.T.ravel(),
                            [1.0]])
        dyt = float(theta @ x)
        cum += dyt
        resp.append(cum * dy_scale * y_scale)   # dy_scale: Δ标准化→Δ物理; 累积→物理°C
        dyhist = np.roll(dyhist, 1); dyhist[0] = dyt
        duhist = np.roll(duhist, 1, axis=0); duhist[0] = 0.0
        ddhist = np.roll(ddhist, 1, axis=0)
    return np.array(resp)


def steady_mask(df, frac=0.35, w=W_STEADY):
    """准稳态 mask: 30步窗口内 load/coal/air 极差 < 各自 P(frac) 阈值.
    返回 (mask, thresh_dict)."""
    T = len(df)
    vals = {}
    for key, col in [('load', '机组负荷'), ('coal', '未校正总煤量'), ('air', '总二次风量')]:
        s = df[col].values
        # 滚动窗口极差 (每步窗口 [t, t+w))
        ptp = np.array([np.ptp(s[t:t + w]) for t in range(T - w + 1)])
        ptp_full = np.full(T, np.nan)
        ptp_full[:T - w + 1] = ptp
        vals[key] = ptp_full
    thresh = {k: np.nanpercentile(v, frac * 100) for k, v in vals.items()}
    mask = np.zeros(T, dtype=bool)
    for t in range(T):
        if np.isnan(vals['load'][t]):
            continue
        if (vals['load'][t] <= thresh['load'] and
                vals['coal'][t] <= thresh['coal'] and
                vals['air'][t] <= thresh['air']):
            mask[t:t + w] = True
    return mask, thresh


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.apply(pd.to_numeric, errors='coerce')
    T = len(df)
    print(f"数据: {T} 行 x {df.shape[1]} 列 ({T*DT/3600:.0f} h)")

    y = df[Y_COL].values.astype(float)
    U = df[VALVE_COLS].values.astype(float)
    D = df[COMBUSTION_COLS].values.astype(float)

    # 差分 (与世界模型动作定义一致)
    dy = np.diff(y); du = np.diff(U, axis=0); dd = np.diff(D, axis=0)
    dy_std, du_std, dd_std = dy.std(), du.std(0), dd.std(0)
    dyz = dy / dy_std
    duz = du / du_std
    ddz = dd / dd_std
    y_scale = 1.0 / dy_std * dy_std   # dy_scale*y_scale 语义见下
    # 阶跃响应: 1σ Δu 脉冲 → 累积 Δy; 每步 Δy 物理量 = dyz * dy_std (°C)

    # ---- 准稳态 mask ----
    mask, thresh = steady_mask(df)
    n_steady = int(mask.sum())
    print(f"准稳态阈值: load≤{thresh['load']:.1f}MW  coal≤{thresh['coal']:.1f}t/h  air≤{thresh['air']:.1f}t/h")
    print(f"准稳态覆盖: {n_steady} 行 ({n_steady/T*100:.1f}%)")

    # ---- A. SISO 全数据 ----
    thetaA, r2A = ridge_arx(dyz, duz, None)
    # ---- B. SISO 准稳态 ----
    dyzs, duzs = dyz[mask[:-1]], duz[mask[:-1]]
    thetaB, r2B = ridge_arx(dyzs, duzs, None)
    # ---- C. MISO 全数据 (燃烧侧剔除) ----
    thetaC, r2C = ridge_arx(dyz, duz, ddz)

    nu, nd = 2, D.shape[1]
    respA = [step_response(thetaA, nu, 0, j, dy_std, 1.0) for j in range(nu)]
    respB = [step_response(thetaB, nu, 0, j, dy_std, 1.0) for j in range(nu)]
    respC = [step_response(thetaC, nu, nd, j, dy_std, 1.0) for j in range(nu)]

    t_axis = np.arange(H_RESP) * DT
    print(f"\n差分域 R²:  A={r2A:.4f}  B={r2B:.4f}  C={r2C:.4f}")
    for j, vname in enumerate(VALVE_COLS):
        rA, rB, rC = respA[j], respB[j], respC[j]
        w60 = slice(6, 10)
        print(f"\n{vname} (+1σ 阶跃 → ΔT):")
        print(f"  峰值: A={rA.max():+.4f}°C(t={t_axis[rA.argmax()]:.0f}s)  "
              f"B={rB.max():+.4f}°C(t={t_axis[rB.argmax()]:.0f}s)  "
              f"C={rC.max():+.4f}°C(t={t_axis[rC.argmax()]:.0f}s)")
        print(f"  谷值: A={rA.min():+.4f}°C(t={t_axis[rA.argmin()]:.0f}s)  "
              f"B={rB.min():+.4f}°C(t={t_axis[rB.argmin()]:.0f}s)  "
              f"C={rC.min():+.4f}°C(t={t_axis[rC.argmin()]:.0f}s)")
        print(f"  60-90s均值: A={rA[w60].mean():+.4f}  B={rB[w60].mean():+.4f}  C={rC[w60].mean():+.4f}")

    # ---- 图 (全英文, Applied Energy 风格) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for j, vname in enumerate(['Spray valve 1', 'Spray valve 2']):
        ax = axes[j]
        for resp, lab, c in [(respA[j], 'A: SISO, all data', '#888888'),
                             (respB[j], 'B: SISO, steady-combustion windows', '#1a7f37'),
                             (respC[j], 'C: MISO, combustion partialled', '#b35900')]:
            ax.plot(t_axis, resp, label=lab, color=c, lw=1.8)
        ax.axvspan(60, 90, color='orange', alpha=0.15, label='physical lag 60-90 s')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ΔT to +1σ valve step (°C)')
        ax.set_title(f'{vname} → main steam T\n(step response, diff-domain ARX p=12)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('Combustion-side confounding: partialled system ID (v2)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, 'fig_combustion_partial_id.png')
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
