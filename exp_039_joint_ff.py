#!/usr/bin/env python3
"""
exp_039_joint_ff.py — 联合优化 v2: SP + 阀位, 带前馈结构
==========================================================
修正 exp_035 (联合发散): SP 变化通过前馈映射物理化到阀位
  a_eff(t) = a_seq(t) − 0.82·(SP(t) − SP_now)   仅二级阀 (PI 执行机构)
  SP轨迹 = SP_now + cumsum(ΔSP)
  M10 输入: (a_eff, SP轨迹) — 自洽无 OOD
动作: [a₁, a₂, ΔSP], 全链路可微
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from experiments.phase1_dynamics.exp_025_unified_benchmark import build_model
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
from exp_027_dwm_mpc import W, H_OUT, DEVICE, test_raw, VALVE_IDX, SP_IDX, TARGET_IDX
sys.argv = _argv

N_TRACKS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
H_PLAN = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ALPHA_FF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.82  # 前馈增益 (SP→二级阀)
LAMBDA_SP = 2.0      # SP 变化惩罚
DSP_MAX = 1.0        # 单步 ΔSP 上限
ETA = 0.05
E_STEPS = 30

wm = build_model('M10').to(DEVICE).eval()
ck = torch.load("results/exp_025_M10/checkpoints/best_model.pth", map_location=DEVICE, weights_only=True)
wm.load_state_dict(ck['model_state_dict'])

def build_obj(x_hist, a_seq, dsp, sp_now, t_set):
    """a_seq [H,2] 阀位基线, dsp [H] SP增量 → J"""
    w = torch.linspace(1.0, 0.8, H_PLAN, device=DEVICE)
    sp_traj = sp_now + torch.cumsum(dsp, 0)            # [H] SP 轨迹
    # 填充到 H_OUT
    if H_PLAN < H_OUT:
        tail_a = a_seq[-1:].repeat(H_OUT - H_PLAN, 1)
        a_full = torch.cat([a_seq, tail_a], 0)
        tail_s = sp_traj[-1:].repeat(H_OUT - H_PLAN)
        sp_full = torch.cat([sp_traj, tail_s], 0)
    else:
        a_full, sp_full = a_seq[:H_OUT], sp_traj[:H_OUT]
    # 前馈结构: 二级阀 = 基线 − ALPHA_FF·(SP−SP_now)
    ff = ALPHA_FF * (sp_full - sp_now)                 # [H_OUT]
    a_eff = a_full.clone()
    a_eff[:, 1] = a_eff[:, 1] - ff                     # 只作用二级阀
    a_eff = a_eff.clamp(0, 100)
    mu, _ = wm(x_hist, a_eff.reshape(1, -1), sp_full.unsqueeze(0))
    mu = mu[0, :H_PLAN]
    err = (mu - t_set) ** 2
    J = (w * err).sum() / H_PLAN
    J = J + 0.5 * err[-1]                              # 终端
    if H_PLAN > 1:
        J = J + 0.1 * ((a_eff[1:] - a_eff[:-1]) ** 2).sum()
    J = J + LAMBDA_SP * (dsp ** 2).sum()
    J = J + 2.0 * F.relu(sp_traj - 580).pow(2).sum() + 2.0 * F.relu(550 - sp_traj).pow(2).sum()
    return J, sp_traj.detach()

def plan_joint(x_hist, sp_now, t_set, a_last):
    a = a_last.unsqueeze(0).repeat(H_PLAN, 1).clone().detach().requires_grad_(True)
    dsp = torch.zeros(H_PLAN, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([a, dsp], lr=ETA)
    for _ in range(E_STEPS):
        opt.zero_grad()
        J, _ = build_obj(x_hist, a, dsp, sp_now, t_set)
        J.backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(0, 100)
            dsp.clamp_(-DSP_MAX, DSP_MAX)
    with torch.no_grad():
        _, sp_traj = build_obj(x_hist, a, dsp, sp_now, t_set)
        sp_traj = sp_now + torch.cumsum(dsp, 0)
        if H_PLAN < H_OUT:
            sp_full = torch.cat([sp_traj, sp_traj[-1:].repeat(H_OUT - H_PLAN)], 0)
            a_full = torch.cat([a, a[-1:].repeat(H_OUT - H_PLAN, 1)], 0)
        else:
            sp_full, a_full = sp_traj[:H_OUT], a[:H_OUT]
        ff = ALPHA_FF * (sp_full - sp_now)
        a_eff = a_full.clone(); a_eff[:, 1] = a_eff[:, 1] - ff
        a_eff = a_eff.clamp(0, 100)
    return a.detach(), dsp.detach(), sp_full.detach(), a_eff.detach()

def simulate(track_idx, n_steps=120):
    win = torch.FloatTensor(test_raw[track_idx:track_idx+W]).unsqueeze(0).to(DEVICE)
    sp_now = torch.tensor(float(test_raw[track_idx+W, SP_IDX]), device=DEVICE)
    a_last = torch.FloatTensor(test_raw[track_idx+W, VALVE_IDX]).to(DEVICE)
    mpc_t, pid_t, mpc_sp, pid_sp, mpc_a = [], [], [], [], []
    for k in range(n_steps):
        gi = track_idx + W + k
        pid_t.append(test_raw[gi, TARGET_IDX]); pid_sp.append(test_raw[gi, SP_IDX])
        t_set = torch.tensor(np.mean(win[0, :, TARGET_IDX].cpu().numpy()), device=DEVICE)
        a_plan, dsp_plan, sp_full, a_eff = plan_joint(win, sp_now, t_set, a_last)
        # 执行第一步
        sp_new = sp_now + dsp_plan[0]
        sp_now = sp_new.clamp(550, 580)
        a1 = a_eff[0]
        a_last = a1
        with torch.no_grad():
            mu, _ = wm(win, a_eff.reshape(1, -1), sp_full.unsqueeze(0))
            y1 = mu[0, 0].item()
        next_row = torch.FloatTensor(test_raw[gi]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        next_row[0, 0, TARGET_IDX] = y1
        next_row[0, 0, SP_IDX] = sp_now.item()
        win = torch.cat([win[:, 1:, :], next_row], 1)
        mpc_t.append(y1); mpc_sp.append(sp_now.item()); mpc_a.append(a1.cpu().numpy())
    return mpc_t, pid_t, mpc_sp, pid_sp, np.array(mpc_a)

np.random.seed(42)
N = len(test_raw)
starts = np.random.choice(range(N - W - H_OUT - 120), N_TRACKS, replace=False)
all_m, t0 = [], time.time()
for k, s in enumerate(starts):
    mpc_t, pid_t, mpc_sp, pid_sp, mpc_a = simulate(s, 120)
    tset = np.array(pid_sp)
    rmse_m = np.sqrt(np.mean((np.array(mpc_t) - tset)**2))
    rmse_p = np.sqrt(np.mean((np.array(pid_t) - tset)**2))
    std_m, std_p = np.std(mpc_t), np.std(pid_t)
    tv_a = np.abs(np.diff(mpc_a[:, 1])).sum()
    tv_sp = np.abs(np.diff(np.array(mpc_sp))).sum()
    dsp_used = np.abs(np.diff(np.array(mpc_sp))).max()
    all_m.append({'rmse_mpc': rmse_m, 'rmse_pid': rmse_p, 'std_mpc': std_m, 'std_pid': std_p,
                  'tv_a2': tv_a, 'tv_sp': tv_sp, 'dsp_max': dsp_used})
    if (k+1) % 10 == 0: print(f"  [{k+1}/{N_TRACKS}] RMSE {rmse_m:.3f}/{rmse_p:.3f}")

agg = {k: float(np.mean([m[k] for m in all_m])) for k in all_m[0]}
print("\n===== 联合优化 v2 (SP前馈+阀位) 汇总 =====")
print(f"  RMSE: {agg['rmse_mpc']:.3f} vs PID {agg['rmse_pid']:.3f} ({(1-agg['rmse_mpc']/agg['rmse_pid'])*100:+.1f}%)")
print(f"  温度std: {agg['std_mpc']:.3f} vs {agg['std_pid']:.3f} ({(1-agg['std_mpc']/agg['std_pid'])*100:+.1f}%)")
print(f"  二级阀TV: {agg['tv_a2']:.2f} | SP TV: {agg['tv_sp']:.2f} | ΔSP最大: {agg['dsp_max']:.2f}")
out = {'n_tracks': N_TRACKS, 'H': H_PLAN, 'alpha_ff': ALPHA_FF, 'lambda_sp': LAMBDA_SP,
       'dsp_max': DSP_MAX, 'agg': agg, 'per_track': all_m}
os.makedirs("results/exp_039_joint_ff", exist_ok=True)
json.dump(out, open(f"results/exp_039_joint_ff/mpc_joint_ff_H{H_PLAN}_a{ALPHA_FF}.json", 'w'), indent=2, default=float)
print(f"Saved: results/exp_039_joint_ff/mpc_joint_ff_H{H_PLAN}_a{ALPHA_FF}.json (耗时 {(time.time()-t0)/60:.1f}min)")
