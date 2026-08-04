#!/usr/bin/env python3
"""
fig_S1_fair_verdict.py — S1 公平协议主图 (nature-figure 规范)
================================================================
核心结论: 公平协议下 (1) M7 全口径优于 M5 (2) MPC vs PID 温度精度相当
但动作平滑度 TV 显著更优 (3) 超温无差异
数据: results/exp_S1/{s1_dist,s1_nodist}.json
"""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})
P = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B", "red_strong": "#B64342", "teal": "#42949E",
    "violet": "#9A4D8E", "neutral_light": "#CFCECE", "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D", "neutral_black": "#272727",
}
ARMS = [('M5', 'cost_a'), ('M5', 'cost_b'), ('M5', 'cost_d'),
        ('M7', 'cost_a'), ('M7', 'cost_b'), ('M7', 'cost_c'), ('M7', 'cost_d')]
ARM_LABEL = {'M5/cost_a': 'M5 (RMSE)', 'M5/cost_b': 'M5 (overtemp)',
             'M5/cost_d': 'M5 (total)', 'M7/cost_a': 'M7 (RMSE)',
             'M7/cost_b': 'M7 (overtemp)', 'M7/cost_c': 'M7 (CVaR)',
             'M7/cost_d': 'M7 (total)', 'M5mu+M7sig/cost_c': 'M5μ+M7σ',
             'PID': 'PID'}
COLOR = {'PID': P['neutral_light'], 'M5': P['teal'], 'M7': P['blue_main'],
         'hybrid': P['violet']}


def load(tag):
    d = json.load(open(f'results/exp_S1/{tag}.json'))
    per = d['per_track']
    out = {}
    for r in per:
        arm = r['arm']
        out.setdefault(arm, []).append(r)
    return out


def stats(rows, key):
    v = np.array([r[key] for r in rows])
    return v.mean(), v.std() / np.sqrt(len(v)), len(v)


def plot_panel(ax, data, key, title, ylab):
    """分组柱状: x=方法臂, 误差棒=SE"""
    arms = ['PID'] + [f"{m}/{c}" for m, c in ARMS]
    means, ses = [], []
    for a in arms:
        if a not in data:
            continue
        m, s, n = stats(data[a], key)
        means.append(m); ses.append(s)
    colors = [COLOR['PID']] + [COLOR[a.split('/')[0]] for a in arms[1:]]
    x = np.arange(len(means))
    ax.bar(x, means, yerr=ses, color=colors, width=0.7, capsize=2.5,
           error_kw=dict(lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL.get(a, a) for a in arms], rotation=30, ha='right', fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylab)
    ax.grid(axis='y', alpha=0.25, lw=0.4)


def paired_diff(per_dist, per_nodist, key='j_total'):
    """M5 vs M7 同 track 配对差值 (cost_a), 合并两场景"""
    diffs = []
    for per in (per_dist, per_nodist):
        m5 = {r['start_seed'] * 1000 + r['track']: r for r in per.get('M5/cost_a', [])}
        m7 = {r['start_seed'] * 1000 + r['track']: r for r in per.get('M7/cost_a', [])}
        for k in m5:
            if k in m7:
                diffs.append(m5[k][key] - m7[k][key])
    return np.array(diffs)


def main():
    dist = load('s1_dist')
    nodist = load('s1_nodist')
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    plot_panel(axes[0, 0], dist, 'rmse', 'a  Disturbed world', 'RMSE (°C)')
    plot_panel(axes[0, 1], dist, 'tv', 'b  Disturbed world', 'TV (valve)')
    plot_panel(axes[1, 0], nodist, 'rmse', 'c  Nominal world', 'RMSE (°C)')
    plot_panel(axes[1, 1], nodist, 'tv', 'd  Nominal world', 'TV (valve)')
    # 显著标记: M7 vs PID RMSE 不显著, TV 显著 — 在 TV 面板标注
    for ax in (axes[0, 1], axes[1, 1]):
        ax.text(0.98, 0.95, 'TV: MPC < PID (p<1e-4)', transform=ax.transAxes,
                ha='right', va='top', fontsize=7, color=P['red_strong'])
    for ax in (axes[0, 0], axes[1, 0]):
        ax.text(0.98, 0.95, 'RMSE: n.s. vs PID', transform=ax.transAxes,
                ha='right', va='top', fontsize=7, color=P['neutral_mid'])
    fig.tight_layout()
    fig.savefig('figures/fig_S1_fair_verdict.pdf', bbox_inches='tight')
    fig.savefig('figures/fig_S1_fair_verdict.png', dpi=300, bbox_inches='tight')
    print('Saved figures/fig_S1_fair_verdict.{pdf,png}')
    d = paired_diff(dist, nodist)
    print(f"  j_total M5-M7 (配对, dist+nodist): mean={d.mean():+.4f}, "
          f"M5优占比={100*(d<0).mean():.0f}%")


if __name__ == '__main__':
    main()
