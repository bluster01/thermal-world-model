#!/usr/bin/env python3
"""
exp_091_supervisory_v3.py — 监督模式虚拟世界 v3: 短程 e 驱动副回路 (2026-08-04)
==============================================================================
副回路 (exp_090 短程系数): ΔV2_t = Σa·ΔV2 + Σb·e, e = SP − T (含 WM 温度反馈)
  - 短程可靠 (e>0→关阀 -0.17@10s), 长程由 WM 闭环负反馈自洽
  - MPC 优化 SP 序列: 18步可微 rollout (副回路+WM 联合, 梯度穿透)
对比: mpc(优化SP) vs step(一步切SP) vs fixed(SP不动)
指标: RMSE/尾段偏差/超调/V2范围/SP范围
用法: python exp_091_supervisory_v3.py [--smoke]
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
SP_STEP = (500.0, 1.0)           # 合成目标阶跃: 500s 时 +1.0°C (现场: SP范围555-575, 单步可到5°C)
SP_LO, SP_HI = 555.0, 575.0      # 现场 SP 动作范围
DSP_MAX = 5.0                    # 无扰斜坡限幅 (°C/10s) — 现场单步上限 5°C
LAMBDA_SP = 0.1
V2_LO, V2_HI = 0.0, 45.0

# ===== 副回路模型 (大信号标定 + 积分, exp_092/093) =====
# 离散 PI: V2 += Kp·(e−e_prev) + Ki·e·DT  — 积分消除静差 (温度最终=SP, 现场跟随率92% @600s)
# 标定: Kp=2.0 %/°C (大阶跃 ΔV2/ΔSP≈−2.0), Ti≈80s (240s 到 92% → Ki=Kp/Ti≈0.025/s)
K_PI = 2.0
KI_PI = 0.025
P, Q = 0, 0

def inner_loop(sp_seq, win, v2_hist, e_hist, v2_last, gi, grad=True):
    """副回路+WM 闭环 rollout: SP → e → PI(比例+积分) → V2 → WM → T (可微)"""
    H = len(sp_seq)
    v2 = v2_last.clone()
    e_prev = torch.zeros((), device=DEVICE)
    win_v = win.clone()
    gi_v = gi
    temps, v2s = [], []
    for t in range(H):
        y_prev = win_v[0, -1, M.TARGET_IDX].detach()
        e = sp_seq[t] - y_prev
        v2 = v2 + K_PI * (e - e_prev) + KI_PI * e * DT
        e_prev = e
        v2 = v2.clamp(V2_LO, V2_HI)
        a_full = torch.full((H_OUT, 2), float(win_v[0, -1, M.VALVE_IDX[0]]), device=DEVICE)
        a_full[:, 1] = v2
        mu, _ = wm(win_v, a_full.reshape(1, -1))
        y = mu[0, 0]
        temps.append(y); v2s.append(v2)
        if gi_v + W + 1 < len(M.test_raw):
            next_row = torch.FloatTensor(M.test_raw[gi_v + W]).unsqueeze(0).unsqueeze(0).to(DEVICE)
            next_row[0, 0, M.TARGET_IDX] = y.detach() if not grad else y
            win_v = torch.cat([win_v[:, 1:, :], next_row], 1)
        gi_v += 1
    return torch.stack(temps), torch.stack(v2s)

def plan_sp(wm, win, target, v2_hist, e_hist, v2_last, gi, sp_now, iters=200):
    sp_seq = torch.full((H_PLAN,), sp_now, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([sp_seq], lr=0.03)
    w = torch.linspace(1.0, 0.8, H_PLAN, device=DEVICE)
    for _ in range(iters):
        opt.zero_grad()
        temps, _ = inner_loop(sp_seq, win, v2_hist, e_hist, v2_last, gi, grad=True)
        J = ((w * (temps - target) ** 2).sum() / H_PLAN
             + LAMBDA_SP * ((sp_seq - sp_now) ** 2).sum() / H_PLAN
             + 1.0 * ((sp_seq[1:] - sp_seq[:-1]) ** 2).sum() / H_PLAN)
        J.backward()
        opt.step()
        with torch.no_grad():
            sp_seq.data = sp_seq.data.clamp(SP_LO, SP_HI)
            d = sp_seq.data - sp_now
            sp_seq.data = sp_now + d.clamp(-DSP_MAX * H_PLAN, DSP_MAX * H_PLAN)
    return sp_seq.detach()

def simulate_sup(wm, track_idx, mode, seed=42, sp_step=SP_STEP):
    np.random.seed(seed)
    raw = M.test_raw
    i = int(track_idx)
    N = len(raw)
    temps, sps, v2s, tsets = [], [], [], []
    a_cur = float(raw[i + W - 1, M.VALVE_IDX[1]])
    sp_now = float(raw[i + W - 1, M.SP_IDX])
    dv2_w = np.diff(raw[i + W - P - 1:i + W, M.VALVE_IDX[1]])
    e_w = raw[i + W - Q - 1:i + W, M.SP_IDX] - raw[i + W - Q - 1:i + W, M.TARGET_IDX]
    v2_hist = [torch.tensor(v, dtype=torch.float32, device=DEVICE) for v in dv2_w[::-1]]
    e_hist = [torch.tensor(v, dtype=torch.float32, device=DEVICE) for v in e_w[::-1]]
    v2_last = torch.tensor(a_cur, dtype=torch.float32, device=DEVICE)
    win = torch.FloatTensor(raw[i:i + W]).unsqueeze(0).to(DEVICE)
    t = 0
    while t < N_STEPS:
        gi = i + t
        if gi + W + 1 >= N:
            break
        t_now = t * DT
        target = sp_now if t_now < sp_step[0] else sp_now + sp_step[1]
        if mode == 'mpc':
            sp_plan = plan_sp(wm, win, target, v2_hist, e_hist, v2_last, gi, sp_now)
            sp_exec = sp_plan[:M_STEP].cpu().numpy()
        elif mode == 'step':
            sp_exec = np.full(M_STEP, target)
        else:
            sp_exec = np.full(M_STEP, sp_now)
        # 执行 M_STEP 步 (无梯度)
        with torch.no_grad():
            sp_v = torch.tensor(sp_exec, dtype=torch.float32, device=DEVICE)
            t_exec, v_exec = inner_loop(sp_v, win, v2_hist, e_hist, v2_last, gi, grad=False)
            t_exec = t_exec.cpu().numpy(); v_exec = v_exec.cpu().numpy()
        for k in range(min(M_STEP, N_STEPS - t)):
            temps.append(float(t_exec[k])); sps.append(float(sp_exec[k]))
            v2s.append(float(v_exec[k])); tsets.append(target)
        # 更新副回路状态 (块末)
        v2_last = torch.tensor(float(v_exec[min(M_STEP, N_STEPS - t) - 1]), device=DEVICE)
        dv2_blk = np.diff(np.concatenate([[a_cur], v_exec[:min(M_STEP, N_STEPS - t)]]))
        e_blk = np.array(sp_exec[:min(M_STEP, N_STEPS - t)]) - t_exec[:min(M_STEP, N_STEPS - t)]
        v2_hist = [torch.tensor(v, dtype=torch.float32, device=DEVICE) for v in dv2_blk[::-1][:P]] + v2_hist
        v2_hist = v2_hist[:P]
        e_hist = [torch.tensor(v, dtype=torch.float32, device=DEVICE) for v in e_blk[::-1][:Q]] + e_hist
        e_hist = e_hist[:Q]
        a_cur = float(v2_last)
        t += M_STEP
        i += M_STEP
        if i + W >= N:
            break
        win = torch.FloatTensor(raw[i:i + W]).unsqueeze(0).to(DEVICE)
    return np.array(temps), np.array(sps), np.array(v2s), np.array(tsets)

if __name__ == '__main__':
    wm = M.load_wm(); wm.eval()
    track = 95560
    print(f"[cfg] 监督模式 v3 (短程e副回路): track={track} SP_STEP={SP_STEP}")
    t0 = time.time()
    res = {}
    for mode in ['mpc', 'step', 'fixed']:
        temps, sps, v2s, tsets = simulate_sup(wm, track, mode)
        rmse = float(np.sqrt(((temps - tsets) ** 2).mean()))
        tail = temps[-30:] - tsets[-30:]
        res[mode] = dict(temps=temps, sps=sps, v2s=v2s, tsets=tsets, rmse=rmse,
                         tail_bias=float(tail.mean()), tail_std=float(tail.std()))
        print(f"  [{mode}] RMSE {rmse:.3f} | 尾段 {res[mode]['tail_bias']:+.3f}±{res[mode]['tail_std']:.3f} | "
              f"SP {sps.min():.2f}-{sps.max():.2f} | V2 {v2s.min():.1f}-{v2s.max():.1f} | {time.time()-t0:.0f}s")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    colors = {'mpc': '#c0504d', 'step': '#4f81bd', 'fixed': '#888888'}
    t_ax = np.arange(len(res['mpc']['temps'])) * DT
    for mode, r in res.items():
        axes[0].plot(t_ax, r['temps'], lw=1.4, label=f"{mode} (RMSE {r['rmse']:.3f})", color=colors[mode])
    axes[0].plot(t_ax, res['mpc']['tsets'], 'k--', lw=1.2, label='Target')
    axes[0].set_ylabel('Outlet temp (°C)'); axes[0].legend(fontsize=8)
    axes[0].set_title('Supervisory MPC v3 (short-horizon e-driven inner loop)')
    for mode, r in res.items():
        axes[1].plot(np.arange(len(r['sps'])) * DT, r['sps'], lw=1.4, color=colors[mode])
    axes[1].set_ylabel('SP (°C)')
    for mode, r in res.items():
        axes[2].plot(np.arange(len(r['v2s'])) * DT, r['v2s'], lw=1.4, color=colors[mode])
    axes[2].set_ylabel('Valve V2 (%)'); axes[2].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig('figures/fig_supervisory_mpc_v3.png', dpi=170, bbox_inches='tight')
    print('Saved: figures/fig_supervisory_mpc_v3.png')
