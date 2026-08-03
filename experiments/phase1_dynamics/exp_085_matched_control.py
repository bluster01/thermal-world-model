"""
exp_085_matched_control.py — 匹配对照: 开阀的因果效应 (剔除燃烧侧扰动)
================================================================
对每个阀位阶跃事件, 找一个"反事实"对照时段:
  - 同一机组, 事件前后 ±12h 内
  - 对照时段无大阀位动作 (|Δv| < 0.5%)
  - 匹配事件前300s: 温度斜率 / 负荷极差 / 煤量极差 / 温度水平
1:1 最近邻匹配后, 效应曲线 = [T(t+τ)−T(t)] − [T(s+τ)−T(s)], τ=0..300s

事件组与对照组的共同趋势 (燃烧侧/AGC/吹灰等) 被差分抵消,
剩下的就是"开阀"的纯因果效应 (含PI反调: 先降后回)。

若匹配对照曲线在 60-90s 后显著为负 → 物理方向确认;
若幅度仍 ≪ WM 持续阶跃翻转幅度 (+0.65°C) → 翻转不可能是物理实效。
用法: python exp_085_matched_control.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = os.path.expanduser(
    "~/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT/A侧主汽温全数据03_cleaned_10s.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

DT = 10.0
BASE_LO, BASE_HI = 30, 300      # 匹配特征窗口: 事件前 [300, 30] s
H_EVT = 30                       # 效应窗口: 事件后 300s
MIN_GAP = 9
SEARCH_H = 12 * 3600 // 10       # 搜索 ±12h (步)
LOAD, COAL, AIR = '机组负荷', '未校正总煤量', '总二次风量'
Y_COL = '末级过热器出口汽温'
V1, V2 = '一级减温调节门阀位', '二级减温调节门阀位'


def roll_ptp(x, w):
    """滑动窗口极差 (窗口 [t-w, t))"""
    n = len(x)
    cs = np.concatenate([[0], np.cumsum(x)])
    out = np.full(n, np.nan)
    for t in range(w, n):
        out[t] = x[t - w:t].max() - x[t - w:t].min()
    return out


def roll_slope(y, w):
    """滑动窗口线性斜率 (窗口 [t-w, t), °C/10s 步)"""
    n = len(y)
    xx = np.arange(w, dtype=float)
    denom = w * (xx ** 2).sum() - xx.sum() ** 2
    csy = np.concatenate([[0], np.cumsum(y)])
    csxy = np.concatenate([[0], np.cumsum(y * np.arange(n))])
    out = np.full(n, np.nan)
    for t in range(w, n):
        sy = csy[t] - csy[t - w]
        sxy = csxy[t] - csxy[t - w] - (t - w) * sy
        out[t] = (w * sxy - xx.sum() * sy) / denom
    return out


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.apply(pd.to_numeric, errors='coerce')
    T = len(df)
    print(f"数据: {T} 行 ({T*DT/3600:.0f} h)")

    y = df[Y_COL].values.astype(float)
    v1, v2 = df[V1].values.astype(float), df[V2].values.astype(float)
    load, coal = df[LOAD].values.astype(float), df[COAL].values.astype(float)
    d1, d2 = np.abs(np.diff(v1)), np.abs(np.diff(v2))

    # ---- 事件 (开阀) ----
    th1, th2 = np.percentile(d1, 99.9), np.percentile(d2, 99.9)
    cand = np.where((d1 >= th1) | (d2 >= th2))[0]
    dv = np.array([np.sign(v1[t+1]-v1[t]) if d1[t] >= th1 else np.sign(v2[t+1]-v2[t]) for t in cand])
    keep, dvk = [], []
    last = -1e9
    for t, s in zip(cand, dv):
        if t - last >= MIN_GAP:
            keep.append(t); dvk.append(s); last = t
    ev, ev_dir = np.array(keep), np.array(dvk)
    open_ev = ev[ev_dir > 0]
    close_ev = ev[ev_dir < 0]
    ok = (open_ev >= BASE_HI + SEARCH_H) & (open_ev + H_EVT < T - SEARCH_H)
    open_ev = open_ev[ok]
    okc = (close_ev >= BASE_HI + SEARCH_H) & (close_ev + H_EVT < T - SEARCH_H)
    close_ev = close_ev[okc]
    print(f"开阀事件 {len(open_ev)} | 关阀事件 {len(close_ev)}")

    # ---- 预计算匹配特征 (每时刻) ----
    W = 30
    slope_T = roll_slope(y, W) * 6.0          # °C/min
    ptp_load = roll_ptp(load, W)
    ptp_coal = roll_ptp(coal, W)
    print("特征预计算完成")

    # 无动作 mask: 前后90s内 |Δv| < 0.5%
    no_act = np.ones(T, bool)
    big = np.where((d1 >= 0.5) | (d2 >= 0.5))[0]
    for t in big:
        no_act[max(0, t-9):t+10] = False
    print(f"无动作时段覆盖 {no_act.mean()*100:.1f}%")

    def matched_effect(evs, label, flip=False):
        """开阀方向统一的匹配效应曲线. flip=True: 关阀事件取反."""
        effects = []
        n_ok = 0
        for t in evs:
            sgn = -1 if (flip and np.sign(y[t+1]-y[t]) != 0 and False) else 1
            # 候选区间: [t-SEARCH_H, t-60] ∪ [t+60, t+SEARCH_H]
            r1 = slice(max(0, t-SEARCH_H), t-6)
            r2 = slice(t+6, min(T, t+SEARCH_H))
            idx = np.concatenate([np.arange(r1.start, r1.stop), np.arange(r2.start, r2.stop)])
            idx = idx[no_act[idx]]
            idx = idx[(~np.isnan(slope_T[idx])) & (~np.isnan(ptp_load[idx])) &
                      (~np.isnan(ptp_coal[idx]))]
            if len(idx) < 100:
                continue
            # 特征向量 (标准化)
            feat_t = np.array([slope_T[t], ptp_load[t], ptp_coal[t], y[t]])
            feat_s = np.stack([slope_T[idx], ptp_load[idx], ptp_coal[idx], y[idx]], 1)
            scale = feat_s.std(0) + 1e-9
            dist = np.abs((feat_s - feat_t) / scale).sum(1)
            s = idx[dist.argmin()]
            # 效应: [T(t+τ)−T(t)] − [T(s+τ)−T(s)]
            base_t, base_s = y[t], y[s]
            seg_t = y[t:t+H_EVT] - base_t
            seg_s = y[s:s+H_EVT] - base_s
            effects.append(seg_t - seg_s)
            n_ok += 1
        E = np.array(effects)
        m, se = E.mean(0), E.std(0) / np.sqrt(len(E))
        print(f"{label}: 匹配成功 {n_ok}/{len(evs)}")
        return m, 1.96 * se, n_ok

    t_axis = np.arange(H_EVT) * DT
    m_open, ci_open, n_open = matched_effect(open_ev, '开阀事件')
    m_close, ci_close, n_close = matched_effect(close_ev, '关阀事件', flip=True)

    w60 = slice(6, 10)
    for nm, m, ci, n in [('开阀', m_open, ci_open, n_open), ('关阀(取反)', m_close, ci_close, n_close)]:
        print(f"\n{nm} (n={n}): 60-90s={m[w60].mean():+.4f}°C  峰值谷值: "
              f"min={m.min():+.4f}°C(t={t_axis[m.argmin()]:.0f}s)  max={m.max():+.4f}°C(t={t_axis[m.argmax()]:.0f}s)")
        print(f"  300s累计={m[-1]:+.4f}°C | 120-180s均值={m[12:18].mean():+.4f}°C")

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for m, ci, n, lab, c in [(m_open, ci_open, n_open, f'Open valve (n={n_open})', '#1a7f37'),
                             (m_close, ci_close, n_close, f'Close valve, sign-flipped (n={n_close})', '#b35900')]:
        ax.plot(t_axis, m, lw=2, color=c, label=lab)
        ax.fill_between(t_axis, m - ci, m + ci, color=c, alpha=0.2)
    ax.axvspan(60, 90, color='orange', alpha=0.15, label='physical lag 60-90 s')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('Time after valve step (s)')
    ax.set_ylabel('Causal ΔT vs matched control (°C)')
    ax.set_title('Matched-control estimate: pure valve effect on main steam T')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_matched_control.png')
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
