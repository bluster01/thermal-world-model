#!/usr/bin/env python3
"""
exp_041_fair_compare.py — 公平协议: PID 也走 WM 闭环
======================================================
修正 exp_027 协议缺陷: MPC温度=WM预测 vs PID温度=真实 (不公平, 模型预测天然平滑)
修正后: 两组都在 WM 闭环虚拟世界 — 对比动作策略质量
  PID-WM: 真实阀位动作 + WM 预测温度 (窗口推进)
  MPC:    MPC优化动作 + WM 预测温度
指标: RMSE/std vs 真实SP, TV, 违规
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
# exp_027 解析 sys.argv — 隔离 import, 参数恢复后读取
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
from exp_027_dwm_mpc import load_wm, W, H_OUT, DEVICE, test_raw, VALVE_IDX, SP_IDX, TARGET_IDX
import exp_027_dwm_mpc as M
sys.argv = _argv

N_TRACKS = int(sys.argv[1]) if len(sys.argv) > 1 else 50

wm = load_wm()
M.SP_TRAJ = 0  # 标量目标 (真实SP)

def sim_pid_wm(track_idx, n_steps=120):
    """PID 动作 (真实阀位) + WM 闭环预测温度 — 与 MPC 同一虚拟世界"""
    win = torch.FloatTensor(test_raw[track_idx:track_idx+W]).unsqueeze(0).to(DEVICE)
    temps = []
    for k in range(n_steps):
        gi = track_idx + W + k
        a_real = torch.FloatTensor(test_raw[gi:gi+H_OUT, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, _ = wm(win, a_real)
            y1 = mu[0, 0].item()
        next_row = torch.FloatTensor(test_raw[gi]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        next_row[0, 0, TARGET_IDX] = y1
        win = torch.cat([win[:, 1:, :], next_row], 1)
        temps.append(y1)
    return np.array(temps)

np.random.seed(42)
N = len(test_raw)
starts = np.random.choice(range(N - W - H_OUT - 120), N_TRACKS, replace=False)
rows, t0 = [], time.time()
for k, s in enumerate(starts):
    # MPC 闭环 (已有)
    mpc_t, _, tset, mpc_a, _ = M.simulate(wm, s, 'grad', n_steps=120)
    # PID-WM 闭环 (公平)
    pid_wm_t = sim_pid_wm(s, 120)
    # 真实 PID (参考)
    pid_real = test_raw[s+W:s+W+120, TARGET_IDX]
    sp_traj = test_raw[s+W:s+W+120, SP_IDX]
    def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
    rows.append({
        'rmse_mpc_vs_sp': rmse(np.array(mpc_t), sp_traj),
        'rmse_pidwm_vs_sp': rmse(pid_wm_t, sp_traj),
        'rmse_pidreal_vs_sp': rmse(pid_real, sp_traj),
        'std_mpc': float(np.std(mpc_t)), 'std_pidwm': float(np.std(pid_wm_t)),
        'std_pidreal': float(np.std(pid_real)),
        'tv_mpc': float(np.abs(np.diff(np.array(mpc_a)[:, 1])).sum()),
        'tv_pid': float(np.abs(np.diff(test_raw[s+W:s+W+120, VALVE_IDX[1]])).sum()),
    })
    if (k+1) % 10 == 0: print(f"  [{k+1}/{N_TRACKS}]")

agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
print("\n===== 公平协议 (两组均 WM 闭环) =====")
print(f"  RMSE vs SP: MPC {agg['rmse_mpc_vs_sp']:.3f} | PID-WM {agg['rmse_pidwm_vs_sp']:.3f} | PID真实 {agg['rmse_pidreal_vs_sp']:.3f}")
print(f"  RMSE改善 (MPC vs PID-WM): {(1-agg['rmse_mpc_vs_sp']/agg['rmse_pidwm_vs_sp'])*100:+.1f}%")
print(f"  温度std: MPC {agg['std_mpc']:.3f} | PID-WM {agg['std_pidwm']:.3f} | PID真实 {agg['std_pidreal']:.3f}")
print(f"  std改善 (MPC vs PID-WM): {(1-agg['std_mpc']/agg['std_pidwm'])*100:+.1f}%")
print(f"  动作TV: MPC {agg['tv_mpc']:.2f} vs PID {agg['tv_pid']:.2f} ({(1-agg['tv_mpc']/agg['tv_pid'])*100:+.1f}%)")

json.dump({'agg': agg, 'per_track': rows},
          open(f"results/exp_041_fair_H10.json", 'w'), indent=2, default=float)
print(f"\nSaved: results/exp_041_fair_H10.json (耗时 {(time.time()-t0)/60:.1f}min)")
