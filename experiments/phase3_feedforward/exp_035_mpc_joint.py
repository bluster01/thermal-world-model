#!/usr/bin/env python3
"""
exp_035_mpc_joint.py — 联合优化: MPC 同时出阀位 + SP
=====================================================
动作空间 [Δa₁, Δa₂, ΔSP] (H×3), 模型 M10 (双条件 WM)
对比: 路线A(纯阀位) / PID
指标: 温度RMSE/std + 阀位TV + SP幅度/平滑 + 违规
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
import config as cfg
from experiments.phase1_dynamics.exp_025_unified_benchmark import build_model
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
from exp_027_dwm_mpc import W, H_OUT, DEVICE, test_raw, VALVE_IDX, SP_IDX, TARGET_IDX
sys.argv = _argv

MODEL_ID = 'M10'
N_TRACKS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
H_PLAN = int(sys.argv[2]) if len(sys.argv) > 2 else 10
LAMBDA1 = 0.1          # 阀位平滑
LAMBDA_SP = 2.0        # SP 变化惩罚
DSP_MAX = 1.5          # 单步 ΔSP 上限
CLIP_DELTA = 5.0       # 阀位单步变化上限
ETA = 0.05
E_STEPS = 30

wm = build_model(MODEL_ID).to(DEVICE).eval()
ck = torch.load(f"results/exp_025_{MODEL_ID}/checkpoints/best_model.pth", map_location=DEVICE, weights_only=True)
wm.load_state_dict(ck['model_state_dict'])

def build_obj(x_hist, a_seq, dsp, sp_now, t_set):
    """a_seq [H,2] 阀位, dsp [H] SP增量 → J"""
    w = torch.linspace(1.0, 0.8, H_PLAN, device=DEVICE)
    if H_PLAN < H_OUT:
        tail = a_seq[-1:].repeat(H_OUT - H_PLAN, 1)
        a_full = torch.cat([a_seq, tail], 0)
        sp_traj = sp_now + torch.cumsum(torch.cat([dsp, dsp[-1:].repeat(H_OUT - H_PLAN)]), 0)
    else:
        a_full = a_seq[:H_OUT]
        sp_traj = sp_now + torch.cumsum(dsp[:H_OUT], 0)
    mu, _ = wm(x_hist, a_full.reshape(1, -1), sp_traj.unsqueeze(0))
    mu = mu[0, :H_PLAN]
    err = (mu - t_set) ** 2
    J = (w * err).sum() / H_PLAN
    J = J + ALPHA * err[-1]
    if H_PLAN > 1:
        J = J + LAMBDA1 * ((a_seq[1:] - a_seq[:-1]) ** 2).sum()
    J = J + LAMBDA_SP * (dsp ** 2).sum()
    J = J + 2.0 * F.relu(sp_traj - 580).pow(2).sum() + 2.0 * F.relu(550 - sp_traj).pow(2).sum()
    return J

ALPHA = 0.5
def plan_joint(x_hist, sp_now, t_set, a_last):
    a = a_last.unsqueeze(0).repeat(H_PLAN, 1).clone().detach().requires_grad_(True)
    dsp = torch.zeros(H_PLAN, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([a, dsp], lr=ETA)
    for _ in range(E_STEPS):
        opt.zero_grad()
        J = build_obj(x_hist, a, dsp, sp_now, t_set)
        J.backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(0, 100)
            a[1:] = torch.clamp(a[1:] - a[:-1], -CLIP_DELTA, CLIP_DELTA) + a[:-1]
            dsp.clamp_(-DSP_MAX, DSP_MAX)
    with torch.no_grad():
        sp_traj = sp_now + torch.cumsum(dsp[:H_OUT], 0) if H_PLAN >= H_OUT else sp_now + torch.cumsum(torch.cat([dsp, dsp[-1:].repeat(H_OUT - H_PLAN)]), 0)
    return a.detach(), sp_traj.detach()

def simulate(track_idx, n_steps=120):
    win = torch.FloatTensor(test_raw[track_idx:track_idx+W]).unsqueeze(0).to(DEVICE)
    sp_now = torch.tensor(float(test_raw[track_idx+W, SP_IDX]), device=DEVICE)
    a_last = torch.FloatTensor(test_raw[track_idx+W, VALVE_IDX]).to(DEVICE)
    mpc_t, pid_t, mpc_sp, pid_sp, mpc_a = [], [], [], [], []
    for k in range(n_steps):
        gi = track_idx + W + k
        pid_t.append(test_raw[gi, TARGET_IDX]); pid_sp.append(test_raw[gi, SP_IDX])
        t_set = torch.tensor(np.mean(win[0, :, TARGET_IDX].cpu().numpy()), device=DEVICE)  # 窗口均值目标
        a_plan, sp_traj = plan_joint(win, sp_now, t_set, a_last)
        a1, sp_new = a_plan[0], sp_now + (sp_traj[0] - sp_now)
        sp_now = sp_new.clamp(550, 580)
        a_last = a1
        with torch.no_grad():
            if H_PLAN < H_OUT:
                a_full = torch.cat([a_plan, a_plan[-1:].repeat(H_OUT - H_PLAN, 1)], 0)
                sp_full = sp_traj  # plan_joint 已返回 H_OUT 长
            else:
                a_full, sp_full = a_plan[:H_OUT], sp_traj[:H_OUT]
            mu, _ = wm(win, a_full.reshape(1, -1), sp_full.unsqueeze(0))
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
    t_set_arr = np.array(pid_sp)
    rmse_m = np.sqrt(np.mean((np.array(mpc_t) - t_set_arr)**2))
    rmse_p = np.sqrt(np.mean((np.array(pid_t) - t_set_arr)**2))
    std_m, std_p = np.std(mpc_t), np.std(pid_t)
    tv_a = np.abs(np.diff(mpc_a[:, 1])).sum()  # 二级阀 TV
    tv_sp = np.abs(np.diff(np.array(mpc_sp))).sum()
    tv_sp_pid = np.abs(np.diff(np.array(pid_sp))).sum()
    all_m.append({'rmse_mpc': rmse_m, 'rmse_pid': rmse_p, 'std_mpc': std_m, 'std_pid': std_p,
                  'tv_a2': tv_a, 'tv_sp': tv_sp, 'tv_sp_pid': tv_sp_pid})
    if (k+1) % 10 == 0: print(f"  [{k+1}/{N_TRACKS}] RMSE {rmse_m:.3f}/{rmse_p:.3f}")

agg = {k: float(np.mean([m[k] for m in all_m])) for k in all_m[0]}
print("\n===== 联合优化 (阀位+SP) 汇总 =====")
print(f"  RMSE: {agg['rmse_mpc']:.3f} vs PID {agg['rmse_pid']:.3f} ({(1-agg['rmse_mpc']/agg['rmse_pid'])*100:+.1f}%)")
print(f"  温度std: {agg['std_mpc']:.3f} vs {agg['std_pid']:.3f} ({(1-agg['std_mpc']/agg['std_pid'])*100:+.1f}%)")
print(f"  二级阀TV: {agg['tv_a2']:.2f} | SP TV: {agg['tv_sp']:.2f} vs PID {agg['tv_sp_pid']:.2f}")
out = {'n_tracks': N_TRACKS, 'H': H_PLAN, 'lambda1': LAMBDA1, 'lambda_sp': LAMBDA_SP,
       'dsp_max': DSP_MAX, 'agg': agg, 'per_track': all_m}
os.makedirs("results/exp_035_joint", exist_ok=True)
json.dump(out, open(f"results/exp_035_joint/mpc_joint_H{H_PLAN}.json", 'w'), indent=2, default=float)
print(f"Saved: results/exp_035_joint/mpc_joint_H{H_PLAN}.json (耗时 {(time.time()-t0)/60:.1f}min)")
