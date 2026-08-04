#!/usr/bin/env python3
"""
fig_S1_cases.py — S1 代表性轨迹 case 图 (nature-figure 规范)
==============================================================
从 s1_dist.json per_track 选 3 个代表性 track, 重跑闭环保存轨迹:
  A. TV 差异最大 (MPC 平滑 vs PID 抖动最明显)
  B. PID 超温 track (安全 case)
  C. M7 j_total 最优 track
每个 case: 温度+SP 曲线 (上) / 阀位曲线 (下)
"""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})
P = {"blue_main": "#0F4D92", "red_strong": "#B64342", "neutral_light": "#CFCECE",
     "neutral_dark": "#4D4D4D", "gold": "#FFD700"}

from experiments.phase2_mpc.exp_S1_fair_comparison import (  # noqa: E402
    WORLD_IDS, CONTROLLERS, make_starts, N_STEPS, M_STEP, DIST_SIGMA, DIST_RHO)
from experiments.phase2_mpc.eval_protocol import (  # noqa: E402
    CostConfig, Disturbance, MPCController, PIDController, WorldSim, load_wm,
    run_episode, test_raw)


CASES = {
    'A': ('42', 19331, 'Disturbed case — MPC −39% RMSE'),
    'B': ('13', 83523, 'Extreme case — MPC mitigates but overtemp persists'),
    'C': ('13', 52730, 'Quiet case — comparable RMSE, TV −61%'),
}


def run_case(world, wm7, cost, s_seed, track, n_steps=120):
    starts = make_starts(s_seed, 50)
    k = list(starts).index(track) if track in starts else 0
    dist = Disturbance(sigma=DIST_SIGMA, rho=DIST_RHO, mode='physical',
                       seed=hash((s_seed, k)) % (2 ** 31))
    ep_pid = run_episode(PIDController(), world, track, n_steps=n_steps, dist=dist)
    ep_mpc = run_episode(MPCController(wm7, cost, h_plan=18, m_step=M_STEP),
                         world, track, n_steps=n_steps, dist=dist)
    return ep_pid, ep_mpc


def main():
    world = WorldSim(WORLD_IDS, controller_ids=CONTROLLERS)
    wm7 = load_wm('M7')
    cost = CostConfig('d')
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 7.2), sharex='col')
    for row, (tag, (s_seed, track, label)) in enumerate(CASES.items()):
        ep_pid, ep_mpc = run_case(world, wm7, cost, int(s_seed), track)
        T = min(len(ep_mpc['temp']), len(ep_pid['temp']))
        t = np.arange(T) * 10 / 60  # 分钟
        ax1, ax2 = axes[row]
        ax1.plot(t, ep_mpc['temp'][:T], color=P['blue_main'], lw=1.0, label='WM-MPC (M7)')
        ax1.plot(t, ep_pid['temp'][:T], color=P['red_strong'], lw=1.0, alpha=0.85, label='PID')
        ax1.plot(t, ep_mpc['sp'][:T], color=P['neutral_light'], lw=0.8, ls='--', label='SP')
        # 工况: 负荷 (右轴)
        gi = track + 96
        ax1b = ax1.twinx()
        ax1b.plot(t, test_raw[gi:gi + T, 0], color=P['gold'], lw=0.8, alpha=0.7, label='Load (MW)')
        ax1b.set_ylabel('Load (MW)', fontsize=7, color=P['gold'])
        ax1b.tick_params(axis='y', labelsize=7, colors=P['gold'])
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title(f'{label}  (seed {s_seed}, track {track})', fontsize=9, loc='left')
        ax1.grid(alpha=0.25, lw=0.4)
        ax2.plot(t, ep_mpc['act'][:T, 1], color=P['blue_main'], lw=1.0, label='WM-MPC')
        ax2.plot(t, ep_pid['act'][:T, 1], color=P['red_strong'], lw=1.0, alpha=0.85, label='PID')
        ax2.set_ylabel('Valve 2 (%)')
        ax2.set_xlabel('Time (min)')
        ax2.grid(alpha=0.25, lw=0.4)
        if row == 0:
            ax1.legend(loc='upper right', fontsize=7, ncol=3)
            ax2.legend(loc='upper right', fontsize=7)
    fig.tight_layout()
    fig.savefig('figures/fig_S1_cases.pdf', bbox_inches='tight')
    fig.savefig('figures/fig_S1_cases.png', dpi=300, bbox_inches='tight')
    print('Saved figures/fig_S1_cases.{pdf,png}')


if __name__ == '__main__':
    main()
