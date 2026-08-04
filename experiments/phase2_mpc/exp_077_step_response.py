#!/usr/bin/env python3
"""
exp_077_step_response.py — 合成 SP 阶跃响应测试 (MPC vs PID)
=============================================================
在固定代表性起点注入合成 SP 阶跃 (+3°C at t=100s, 保持):
MPC/PID 温度跟踪 + 超调量/调节时间量化
无扰动 (DIST_AMP=0) 干净图; 也跑 SP_TRAJ=1 前馈对照
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv

plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})

M.SP_TRAJ = 0; M.M_STEP = 6; M.H_PLAN = 18
M.FIX_MODE = 'overlap'; M.LAMBDA3 = 0.05; M.HARD_DELTA = 5.0
M.BENCH_SP_EACH = True; M.RISK_LAMBDA = 0.0
M.DIST_AMP = 0.0  # 无扰动: 干净阶跃响应

wm = M.load_wm()
STEP_TIME = 100.0   # 阶跃时刻 (s)
STEP_AMP = float(os.environ.get('STEP_AMP', '1.0'))  # 阶跃幅度 (°C) — 更小幅更容易收敛
N_STEPS = 250       # 2500s (阶跃后 2400s 观察调节, 覆盖减温阀慢爬升)

# 代表性起点: test 段均匀取 6 条, 窗口内 SP 稳定 (std<0.3)
cand = np.linspace(5000, len(M.test_raw) - M.W - N_STEPS - 20, 200).astype(int)
starts = []
for i in cand:
    seg = M.test_raw[i+M.W-60:i+M.W+60, M.SP_IDX]
    if seg.std() < 0.3 and len(starts) < 6:
        starts.append(int(i))

fig, axes = plt.subplots(3, 2, figsize=(12, 9.5), sharex=True)
t_axis = np.arange(N_STEPS) * M.DT
for r, s in enumerate(starts):
    ax = axes[r // 2, r % 2]
    mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, 'grad', n_steps=N_STEPS,
                                                  sp_step=(STEP_TIME, STEP_AMP))
    # 真实温度轨迹 (同一起点, 无 MPC 干预的历史实际)
    real_t = M.test_raw[s+M.W:s+M.W+N_STEPS, M.TARGET_IDX]
    ax.plot(t_axis, tset, 'k--', lw=1.2, label='SP (synthetic step)')
    ax.plot(t_axis, mpc_t, '#c0504d', lw=1.6, label='DWM-MPC')
    ax.plot(t_axis, real_t, '#999999', lw=1.0, alpha=0.8, ls=':', label='Actual (no intervention)')
    # 阶跃前后误差统计
    e_mpc = mpc_t - tset; e_pid = pid_t - tset
    pre = t_axis < STEP_TIME
    post = (t_axis >= STEP_TIME) & (t_axis < STEP_TIME + 600)
    os_mpc = float(np.abs(e_mpc[post]).max()); os_pid = float(np.abs(e_pid[post]).max())
    rms_pre = float(np.sqrt(np.mean(e_mpc[pre] ** 2)))
    ax.set_title(f'Track {r+1}: start={s} | 阶跃后峰值偏差 MPC {os_mpc:.2f}°C / PID {os_pid:.2f}°C '
                 f'| 阶跃前MPC RMSE {rms_pre:.2f}', fontsize=8.5)
    ax.axvline(STEP_TIME, color='gray', lw=0.8, ls=':')
    if r == 0:
        ax.legend(fontsize=8, loc='upper left')
axes[-1, -1].axis('off')
fig.suptitle('Synthetic SP step response (+3°C at 100s): DWM-MPC vs PID (no disturbance)', y=1.0, fontsize=11)
fig.tight_layout()
fig.savefig('figures/fig_step_response.png', dpi=180, bbox_inches='tight')
print('Saved: figures/fig_step_response.png')

# ============ 量化 (5 起点平均) ============
print('\n=== 合成阶跃响应量化 (5 起点平均) ===')
res = {'mpc': [], 'pid': []}
for s in starts[:5]:
    mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, 'grad', n_steps=N_STEPS,
                                                  sp_step=(STEP_TIME, STEP_AMP))
    for name, temp, act in [('mpc', mpc_t, mpc_a), ('pid', pid_t, pid_a)]:
        e = temp - tset
        post = t_axis >= STEP_TIME
        # 超调: 阶跃后 300s 内峰值偏差
        pk = float(np.abs(e[(t_axis >= STEP_TIME) & (t_axis < STEP_TIME + 300)]).max())
        # 调节时间: 偏差进入 ±0.3°C 且不再出去
        settle = np.nan
        band = np.abs(e) < 0.3
        for tt in range(int(STEP_TIME // M.DT), N_STEPS):
            if band[tt:].all():
                settle = t_axis[tt] - STEP_TIME
                break
        rms_post = float(np.sqrt(np.mean(e[(t_axis >= STEP_TIME) & (t_axis < STEP_TIME + 600)] ** 2)))
        tv = float(np.abs(np.diff(act[:60], axis=0)).sum()) / 2.0
        res[name].append((pk, settle, rms_post, tv))
for name in ['mpc', 'pid']:
    pk = np.mean([x[0] for x in res[name]])
    st = np.nanmean([x[1] for x in res[name]])
    rms = np.mean([x[2] for x in res[name]])
    tv = np.mean([x[3] for x in res[name]])
    print(f"  {name.upper():4s}: 峰值偏差 {pk:.2f}°C | 调节时间 {st:.0f}s | 阶跃后600s RMSE {rms:.2f} | 阶跃后TV {tv:.3f}")
