#!/usr/bin/env python3
"""
exp_089_supervisory_mpc.py — 监督模式虚拟世界冒烟 v2 (2026-08-04)
==================================================================
执行链: MPC 优化 SP 序列(idx36) → 真PID副回路(exp_027 _simulate_pid 同款)
        → 阀位 V2 → M7 WM → 出口温度 → 误差 e=T−SP 回副回路 (闭环自洽)
  - 规划: 18 步可微闭环 rollout (PI 线性 + WM 可微, 梯度穿透), 目标=温度跟踪目标轨迹
  - 对比: mpc(优化SP) vs step(一步切SP, 现状) vs fixed(SP不动)
  - V1 冻结 (监督模式第一版只管二级减温)
用法: python exp_089_supervisory_mpc.py [--smoke]
"""
import os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase2_mpc'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMOKE = '--smoke' in sys.argv
DT = 10.0
W = M.W
H_OUT = M.H_OUT
H_PLAN = 18
M_STEP = 6
N_STEPS = 150 if not SMOKE else 80
SP_STEP = (500.0, 1.5)           # 合成目标阶跃: 500s 时 +1.5°C
SP_LO, SP_HI = 560.0, 572.0
DSP_MAX = 0.3                    # 无扰斜坡限幅: |ΔSP| ≤ 0.3°C/10s
LAMBDA_SP = 0.2                  # SP 偏离当前值惩罚 (轻度, 允许前瞻调整)
KP, KI = 40.0, 8.0               # 副回路 PID (exp_027 默认, 现场 PI 模型)
U_LO, U_HI = 0.0, 45.0           # 阀位执行限幅
INERT = 0.5                      # 阀门一阶惯性系数

def rollout(wm, win, sp_seq, target, a_cur, pid_I, pid_e_prev, v1_last, gi, grad=True):
    """18 步闭环 rollout: SP → PI → V2 → WM → T → e (可微, 返回温度轨迹 [H])"""
    H = len(sp_seq)
    temps = []
    a_cur_v = torch.tensor(a_cur, dtype=torch.float32, device=DEVICE, requires_grad=False)
    pid_I_v = torch.tensor(pid_I, dtype=torch.float32, device=DEVICE)
    pid_e_v = torch.tensor(pid_e_prev, dtype=torch.float32, device=DEVICE)
    win_v = win.clone()
    gi_v = gi
    e_sum = 0.0
    for t in range(H):
        y_prev = win_v[0, -1, M.TARGET_IDX].detach()
        e = sp_seq[t] - y_prev
        pid_I_v = pid_I_v + e * DT
        u = KP * e + KI * pid_I_v + 0.0 * (e - pid_e_v) / DT
        pid_e_v = e
        a_cur_v = a_cur_v + INERT * (u - a_cur_v)
        a_cur_c = a_cur_v.clamp(U_LO, U_HI)
        a_full = torch.full((H_OUT, 2), v1_last, device=DEVICE)
        a_full[:, 1] = a_cur_c
        mu, _ = wm(win_v, a_full.reshape(1, -1))
        y = mu[0, 0]
        temps.append(y)
        # 窗口推进: 真实非温度状态 + 预测温度
        if gi_v + W + 1 < len(M.test_raw):
            next_row = torch.FloatTensor(M.test_raw[gi_v + W]).unsqueeze(0).unsqueeze(0).to(DEVICE)
            next_row[0, 0, M.TARGET_IDX] = y.detach() if not grad else y
            win_v = torch.cat([win_v[:, 1:, :], next_row], 1)
        gi_v += 1
        e_sum = e_sum + e
    return torch.stack(temps)

def plan_sp(wm, win, target, a_cur, pid_I, pid_e_prev, v1_last, gi, sp_now, iters=250):
    """Adam 优化 SP 序列: J = Σw(ŷ−target)² + λ_sp(SP−sp_now)² + 2·ΣΔSP²"""
    sp_seq = torch.full((H_PLAN,), sp_now, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([sp_seq], lr=0.02)
    w = torch.linspace(1.0, 0.8, H_PLAN, device=DEVICE)
    last_J = float('inf')
    for it in range(iters):
        opt.zero_grad()
        temps = rollout(wm, win, sp_seq, target, a_cur, pid_I, pid_e_prev, v1_last, gi, grad=True)
        J = ((w * (temps - target) ** 2).sum() / H_PLAN
             + LAMBDA_SP * ((sp_seq - sp_now) ** 2).sum() / H_PLAN
             + 2.0 * ((sp_seq[1:] - sp_seq[:-1]) ** 2).sum() / H_PLAN)
        J.backward()
        opt.step()
        with torch.no_grad():
            sp_seq.data = sp_seq.data.clamp(SP_LO, SP_HI)
            d = sp_seq.data - sp_now
            sp_seq.data = sp_now + d.clamp(-DSP_MAX * H_PLAN, DSP_MAX * H_PLAN)
        last_J = float(J.item())
    return sp_seq.detach(), last_J

def simulate_sup(wm, track_idx, mode, seed=42, sp_step=SP_STEP):
    """监督模式闭环: 每块规划/给定 SP → 逐步执行 (PI+V2+WM) → 推进"""
    np.random.seed(seed)
    raw = M.test_raw
    i = int(track_idx)
    N = len(raw)
    temps, sps, v2s, tsets = [], [], [], []
    v1_last = float(raw[i + W - 1, M.VALVE_IDX[0]])
    a_cur = float(raw[i + W - 1, M.VALVE_IDX[1]])
    sp_now = float(raw[i + W - 1, M.SP_IDX])
    pid_I, pid_e_prev = 0.0, 0.0
    win = torch.FloatTensor(raw[i:i + W]).unsqueeze(0).to(DEVICE)
    t = 0
    while t < N_STEPS:
        gi = i + t
        if gi + W + 1 >= N:
            break
        t_now = t * DT
        target = sp_now if t_now < sp_step[0] else sp_now + sp_step[1]
        if mode == 'mpc':
            sp_plan, J = plan_sp(wm, win, target, a_cur, pid_I, pid_e_prev, v1_last, gi, sp_now)
            sp_exec = sp_plan[:M_STEP].cpu().numpy()
        elif mode == 'step':
            sp_exec = np.full(M_STEP, target)
        else:
            sp_exec = np.full(M_STEP, sp_now)
        # 执行 M_STEP 步 (PI + WM 逐步)
        for k in range(M_STEP):
            if t + k >= N_STEPS:
                break
            gi2 = gi + k
            if gi2 + W + 1 >= N:
                break
            sp_v = float(sp_exec[k])
            y_meas = float(win[0, -1, M.TARGET_IDX])
            e = sp_v - y_meas
            pid_I = float(np.clip(pid_I + e * DT, -300.0, 300.0))
            u = KP * e + KI * pid_I
            pid_e_prev = e
            a_cur = float(np.clip(a_cur + INERT * (u - a_cur), U_LO, U_HI))
            a_full = torch.full((H_OUT, 2), v1_last, device=DEVICE)
            a_full[:, 1] = a_cur
            with torch.no_grad():
                mu, _ = wm(win, a_full.reshape(1, -1))
            y_j = float(mu[0, 0])
            temps.append(y_j); sps.append(sp_v); v2s.append(a_cur); tsets.append(target)
            next_row = torch.FloatTensor(raw[gi2 + W]).unsqueeze(0).unsqueeze(0).to(DEVICE)
            next_row[0, 0, M.TARGET_IDX] = y_j
            win = torch.cat([win[:, 1:, :], next_row], 1)
        t += M_STEP
    return np.array(temps), np.array(sps), np.array(v2s), np.array(tsets)

if __name__ == '__main__':
    wm = M.load_wm(); wm.eval()
    track = 95560
    print(f"[cfg] 监督模式 v2 (PID副回路): track={track} SP_STEP={SP_STEP} H_PLAN={H_PLAN} M_STEP={M_STEP}")
    t0 = time.time()
    res = {}
    for mode in ['mpc', 'step', 'fixed']:
        temps, sps, v2s, tsets = simulate_sup(wm, track, mode)
        rmse = float(np.sqrt(((temps - tsets) ** 2).mean()))
        # 阶跃后稳态 (最后 30 步) 偏差
        tail = temps[-30:] - tsets[-30:]
        res[mode] = dict(temps=temps, sps=sps, v2s=v2s, tsets=tsets, rmse=rmse,
                         tail_bias=float(tail.mean()), tail_std=float(tail.std()))
        print(f"  [{mode}] RMSE {rmse:.3f}°C | 尾段偏差 {res[mode]['tail_bias']:+.3f}±{res[mode]['tail_std']:.3f} | "
              f"SP {sps.min():.2f}-{sps.max():.2f} | V2 {v2s.min():.1f}-{v2s.max():.1f} | {time.time()-t0:.0f}s")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    colors = {'mpc': '#c0504d', 'step': '#4f81bd', 'fixed': '#888888'}
    t_ax = np.arange(len(res['mpc']['temps'])) * DT
    for mode, r in res.items():
        axes[0].plot(t_ax, r['temps'], lw=1.4, label=f"{mode} (RMSE {r['rmse']:.3f}°C)", color=colors[mode])
    axes[0].plot(t_ax, res['mpc']['tsets'], 'k--', lw=1.2, label='Target (SP step)')
    axes[0].set_ylabel('Outlet temp (°C)'); axes[0].legend(fontsize=8)
    axes[0].set_title('Supervisory MPC (PID inner loop): SP optimization vs baselines')
    for mode, r in res.items():
        axes[1].plot(np.arange(len(r['sps'])) * DT, r['sps'], lw=1.4, color=colors[mode])
    axes[1].set_ylabel('SP (°C)')
    for mode, r in res.items():
        axes[2].plot(np.arange(len(r['v2s'])) * DT, r['v2s'], lw=1.4, color=colors[mode])
    axes[2].set_ylabel('Valve V2 (%)'); axes[2].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig('figures/fig_supervisory_mpc_smoke.png', dpi=170, bbox_inches='tight')
    print('Saved: figures/fig_supervisory_mpc_smoke.png')
