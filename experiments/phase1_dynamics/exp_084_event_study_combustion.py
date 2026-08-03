"""
exp_084_event_study_combustion.py — 阀位阶跃事件研究, 按燃烧侧活跃度分层
================================================================
直接回答: 持续开阀→主汽温"翻转", 是燃烧侧扰动还是物理实效?

方法 (非参数, 零模型假设):
1. 事件: 10s 内 |Δ阀位| ≥ 阈值 (一级/二级减温阀) 的时刻
2. 独立性: ±90s 内其他事件剔除 (只保留间隔>90s 的事件)
3. 趋势校正: 事件前 [t-600, t-60] 线性外推基线, ΔT = T - 基线外推
4. 分层: 事件前 300s 窗口内 负荷/煤量/风量 极差
     - 平稳层: 三个极差都 < 各自 P40 (燃烧侧基本不动)
     - 活跃层: 至少一个极差 > 各自 P60
5. 输出: 两层 0-300s 平均 ΔT 曲线 ± 95% CI, 方向 + 60-90s 时标

若平稳层显著为负 (开阀→降温) 而活跃层翻转/为正 → 实锤燃烧侧混杂。
用法: python exp_084_event_study_combustion.py
"""
import os
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

DT = 10.0
BASE_LO, BASE_HI = 60, 600   # 基线窗口: 事件前 [600, 60] s
H_EVT = 30                   # 事件后 300s
MIN_GAP = 9                  # 事件最小间隔 90s (步)

LOAD, COAL, AIR = '机组负荷', '未校正总煤量', '总二次风量'
Y_COL = '末级过热器出口汽温'
V1, V2 = '一级减温调节门阀位', '二级减温调节门阀位'


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.apply(pd.to_numeric, errors='coerce')
    T = len(df)
    print(f"数据: {T} 行 ({T*DT/3600:.0f} h)")

    y = df[Y_COL].values.astype(float)
    v1, v2 = df[V1].values.astype(float), df[V2].values.astype(float)
    d1, d2 = np.abs(np.diff(v1)), np.abs(np.diff(v2))

    # 事件阈值: 各自 P99.9 的 Δ阀位
    th1, th2 = np.percentile(d1, 99.9), np.percentile(d2, 99.9)
    print(f"Δ阀位 P99.9: v1={th1:.2f}%  v2={th2:.2f}%  (10s内)")

    cand = np.where((d1 >= th1) | (d2 >= th2))[0]
    # 事件方向: 开阀 (+1) / 关阀 (-1); 一级二级分别
    dv = np.zeros(len(cand))
    for k, t in enumerate(cand):
        dv[k] = np.sign(v1[t + 1] - v1[t]) if d1[t] >= th1 else np.sign(v2[t + 1] - v2[t])
    # 独立性: 间隔 ≥ MIN_GAP
    keep, dv_keep = [], []
    last = -1e9
    for t, sgn in zip(cand, dv):
        if t - last >= MIN_GAP:
            keep.append(t)
            dv_keep.append(sgn)
            last = t
    ev = np.array(keep)
    ev_dir = np.array(dv_keep)
    print(f"候选事件 {len(cand)} → 独立事件 {len(ev)} "
          f"(开阀 {(ev_dir > 0).sum()} / 关阀 {(ev_dir < 0).sum()})")

    # 排除太靠前/靠后的 (基线/窗口不足)
    ok = (ev >= BASE_HI) & (ev + H_EVT < T)
    ev, ev_dir = ev[ok], ev_dir[ok]
    print(f"可用事件 {len(ev)}")

    # 燃烧活跃度: 事件前 300s 窗口极差
    W = 30
    ptp_load = np.array([np.ptp(df[LOAD].values[t - W:t]) for t in ev])
    ptp_coal = np.array([np.ptp(df[COAL].values[t - W:t]) for t in ev])
    ptp_air = np.array([np.ptp(df[AIR].values[t - W:t]) for t in ev])
    # 分层阈值: 各自 P40 / P60
    l40, l60 = np.percentile(ptp_load, [40, 60])
    c40, c60 = np.percentile(ptp_coal, [40, 60])
    a40, a60 = np.percentile(ptp_air, [40, 60])
    steady = (ptp_load <= l40) & (ptp_coal <= c40) & (ptp_air <= a40)
    active = (ptp_load > l60) | (ptp_coal > c60) | (ptp_air > a60)
    print(f"分层: 平稳层 {steady.sum()} 事件 | 活跃层 {active.sum()} 事件 | "
          f"中间 {(~steady & ~active).sum()} 事件 (弃)")
    print(f"平稳层阈值: load≤{l40:.1f}MW  coal≤{c40:.1f}t/h  air≤{a40:.1f}t/h")


    def avg_curve(idx, label, sign=+1):
        """事件后平均 ΔT 曲线 (趋势校正: 基线线性外推).
        sign=+1: 开阀事件; sign=-1: 关阀事件 (响应取反 → 统一为"开阀方向")."""
        curves = []
        for t, sgn in zip(ev[idx], ev_dir[idx]):
            base = np.polyfit(np.arange(BASE_LO, BASE_HI), y[t - BASE_HI:t - BASE_LO], 1)
            seg = y[t - BASE_LO:t + H_EVT]
            base_line = np.polyval(base, np.arange(BASE_LO, BASE_LO + len(seg)))
            curves.append((seg - base_line) * (sgn if sign == +1 else -sgn))
        C = np.array(curves)                      # [n_ev, 60+H_EVT]
        post = C[:, BASE_LO:]                     # [n_ev, H_EVT]
        m, se = post.mean(0), post.std(0) / np.sqrt(len(post))
        return m, 1.96 * se, len(post)

    t_axis = np.arange(H_EVT) * DT
    m_ss, ci_ss, n_ss = avg_curve(steady, 'steady', sign=+1)
    m_ac, ci_ac, n_ac = avg_curve(active, 'active', sign=+1)
    # 开阀/关阀拆分统计
    m_open, ci_open, n_open = avg_curve(np.ones(len(ev), bool), 'open', sign=+1)
    m_close, ci_close, n_close = avg_curve(np.ones(len(ev), bool), 'close', sign=-1)

    # 关键指标
    w60 = slice(6, 10)
    print(f"\n[开阀方向统一] 平稳层 (n={n_ss}): 60-90s均值={m_ss[w60].mean():+.3f}°C  "
          f"300s累计={m_ss[-1]:+.3f}°C  谷值={m_ss.min():+.3f}°C(t={t_axis[m_ss.argmin()]:.0f}s)")
    print(f"[开阀方向统一] 活跃层 (n={n_ac}): 60-90s均值={m_ac[w60].mean():+.3f}°C  "
          f"300s累计={m_ac[-1]:+.3f}°C  谷值={m_ac.min():+.3f}°C(t={t_axis[m_ac.argmin()]:.0f}s)")
    print(f"[拆开] 开阀事件 (n={n_open}): 300s累计={m_open[-1]:+.3f}°C")
    print(f"[拆开] 关阀事件 (n={n_close}, 响应取反): 300s累计={m_close[-1]:+.3f}°C")

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_axis, m_ss, color='#1a7f37', lw=2, label=f'Steady combustion (n={n_ss})')
    ax.fill_between(t_axis, m_ss - ci_ss, m_ss + ci_ss, color='#1a7f37', alpha=0.2)
    ax.plot(t_axis, m_ac, color='#b35900', lw=2, label=f'Active combustion (n={n_ac})')
    ax.fill_between(t_axis, m_ac - ci_ac, m_ac + ci_ac, color='#b35900', alpha=0.2)
    ax.axvspan(60, 90, color='orange', alpha=0.15, label='physical lag 60-90 s')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('Time after valve step (s)')
    ax.set_ylabel('ΔT vs baseline extrapolation (°C)')
    ax.set_title('Valve step events: main steam T response, stratified by combustion activity')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_event_study_combustion.png')
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
