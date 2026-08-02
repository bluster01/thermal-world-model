#!/usr/bin/env python3
"""
exp_034_mpc_sp.py — 路线B: MPC 前馈调节设定值 (SP 为动作通道)
================================================================
与路线A (MPC直接出阀位) 对比。伊敏落地方式: MPC 输出 SP 轨迹 → 现有 PI 回路执行。

闭环仿真:
  MPC-B: 优化 SP 轨迹 (动作=ΔSP) → M10(历史, SP轨迹) → 温度 (M10 隐式含 PI 闭环行为)
  PID:   真实数据 (真实 SP → 真实 PI → 温度)

指标: 温度 RMSE/std vs PID + SP 轨迹平滑性 (ΔSP 限制) + 隐含阀位 (辨识PI后验估算)
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from experiments.phase1_dynamics.exp_025_unified_benchmark import build_model
# exp_027 会解析 sys.argv — 隔离: 用默认参数 import 取数据常量
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
from exp_027_dwm_mpc import W, H_OUT, DEVICE, test_raw, VALVE_IDX, SP_IDX, TARGET_IDX
sys.argv = _argv

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else 'M10'
N_TRACKS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
H_PLAN = int(sys.argv[3]) if len(sys.argv) > 3 else 10
LAMBDA_SP = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0   # SP 变化惩罚
DSP_MAX = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0     # 单步 ΔSP 上限 (°C)
ETA_SP = 0.3                                                    # SP 规划步长
E_STEPS = 40

wm = build_model(MODEL_ID).to(DEVICE).eval()
ck = torch.load(f"results/exp_025_{MODEL_ID}/checkpoints/best_model.pth", map_location=DEVICE, weights_only=True)
wm.load_state_dict(ck['model_state_dict'])
assert getattr(wm, 'use_sp', False), f"{MODEL_ID} 不是 SP 条件模型!"

def build_obj_sp(x_hist, dsp, sp_now, t_set):
    """J = Σwₜ(ŷₜ−T_set)² + λ_sp·ΣΔSP² + 软约束
    dsp: [H] 可微, SP 轨迹 = sp_now + cumsum(dsp)"""
    w = torch.linspace(1.0, 0.8, H_PLAN, device=DEVICE)
    sp_traj = sp_now + torch.cumsum(dsp, 0)         # [H] SP 轨迹
    # 填充到 H_OUT 给 M10 (后段保持)
    if H_PLAN < H_OUT:
        tail = sp_traj[-1:].repeat(H_OUT - H_PLAN)
        sp_full = torch.cat([sp_traj, tail], 0)
    else:
        sp_full = sp_traj[:H_OUT]
    # M10 签名: forward(x_hist, a_future, sp_future); 路线B 无阀位动作 → a_future 用真实阀位
    a_fut = x_hist.new_zeros(1, H_OUT * 2)  # 占位 (M10 内部 use_sp 分支需要 a_future reshape)
    mu, _ = wm(x_hist, a_fut, sp_full.unsqueeze(0))
    mu = mu[0, :H_PLAN]
    err = (mu - t_set) ** 2
    J = (w * err).sum() / H_PLAN
    J = J + LAMBDA_SP * (dsp ** 2).sum()            # SP 变化惩罚 (防抖)
    J = J + 2.0 * F.relu(sp_traj - 580).pow(2).sum() + 2.0 * F.relu(550 - sp_traj).pow(2).sum()
    return J, sp_traj.detach()

def plan_sp(wm, x_hist, sp_now, t_set):
    dsp = torch.zeros(H_PLAN, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([dsp], lr=ETA_SP)
    for _ in range(E_STEPS):
        opt.zero_grad()
        J, _ = build_obj_sp(x_hist, dsp, sp_now, t_set)
        J.backward()
        opt.step()
        with torch.no_grad():
            dsp.clamp_(-DSP_MAX, DSP_MAX)
    with torch.no_grad():
        _, sp_traj = build_obj_sp(x_hist, dsp, sp_now, t_set)
    return sp_traj, dsp.detach()

def simulate_sp(wm, track_idx, n_steps=120):
    """闭环: 每步 MPC-B 出 SP → M10 预测温度 → 窗口推进"""
    W_ = W
    win = torch.FloatTensor(test_raw[track_idx:track_idx+W_]).unsqueeze(0).to(DEVICE)
    sp_now = torch.tensor(float(test_raw[track_idx+W_, SP_IDX]), device=DEVICE)
    t_set = sp_now.clone()
    mpc_t, mpc_sp, pid_t, pid_sp = [], [], [], []
    for k in range(n_steps):
        gi = track_idx + W_ + k
        # PID 参考 (真实)
        pid_t.append(test_raw[gi, TARGET_IDX]); pid_sp.append(test_raw[gi, SP_IDX])
        # MPC-B 规划
        t_set = torch.tensor(float(test_raw[gi, SP_IDX]), device=DEVICE)  # 目标=真实SP轨迹(运行人员意图)
        sp_traj, _ = plan_sp(wm, win, sp_now, t_set)
        # 执行第一步 → 更新 SP (给 PI 回路)
        sp_now = sp_now + (sp_traj[0] - sp_now) * 1.0  # 全量执行 (可调系数)
        sp_now = sp_now.clamp(550, 580)
        # WM 闭环预测温度 (M11, 用 MPC 的 SP 轨迹)
        with torch.no_grad():
            sp_full = torch.cat([sp_traj, sp_traj[-1:].repeat(H_OUT - H_PLAN)]) if H_PLAN < H_OUT else sp_traj[:H_OUT]
            a_fut = win.new_zeros(1, H_OUT * 2)  # M11 无阀位动作, 占位
            mu, _ = wm(win, a_fut, sp_full.unsqueeze(0))
            y1 = mu[0, 0].item()
        # 窗口推进 (温度列替换为预测, SP 列替换为 MPC 的 SP)
        next_row = torch.FloatTensor(test_raw[gi]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        next_row[0, 0, TARGET_IDX] = y1
        next_row[0, 0, SP_IDX] = sp_now.item()
        win = torch.cat([win[:, 1:, :], next_row], 1)
        mpc_t.append(y1); mpc_sp.append(sp_now.item())
    return mpc_t, pid_t, mpc_sp, pid_sp

# ===== 主跑 =====
np.random.seed(42)
N = len(test_raw)
starts = np.random.choice(range(N - W - H_OUT - 120), N_TRACKS, replace=False)
all_m = []
t0 = time.time()
for k, s in enumerate(starts):
    mpc_t, pid_t, mpc_sp, pid_sp = simulate_sp(wm, s, 120)
    # 温度指标 (T_set = 真实 SP)
    rmse_m = np.sqrt(np.mean((np.array(mpc_t) - np.array(pid_sp))**2))
    rmse_p = np.sqrt(np.mean((np.array(pid_t) - np.array(pid_sp))**2))
    std_m, std_p = np.std(mpc_t), np.std(pid_t)
    # SP 轨迹质量: 平滑性 (TV), 幅度
    sp_arr = np.array(mpc_sp)
    tv_m = np.abs(np.diff(sp_arr)).sum()
    tv_p = np.abs(np.diff(np.array(pid_sp))).sum()
    m = {'rmse_mpc': rmse_m, 'rmse_pid': rmse_p, 'temp_std_mpc': std_m, 'temp_std_pid': std_p,
         'sp_tv_mpc': tv_m, 'sp_tv_pid': tv_p}
    all_m.append(m)
    if (k+1) % 10 == 0: print(f"  [{k+1}/{N_TRACKS}] RMSE {rmse_m:.3f}/{rmse_p:.3f}")

agg = {k: float(np.mean([m[k] for m in all_m])) for k in all_m[0]}
print("\n===== 路线B: MPC-SP 汇总 =====")
print(f"  RMSE: MPC {agg['rmse_mpc']:.3f} vs PID {agg['rmse_pid']:.3f} ({(1-agg['rmse_mpc']/agg['rmse_pid'])*100:+.1f}%)")
print(f"  温度std: {agg['temp_std_mpc']:.3f} vs {agg['temp_std_pid']:.3f} ({(1-agg['temp_std_mpc']/agg['temp_std_pid'])*100:+.1f}%)")
print(f"  SP TV: MPC {agg['sp_tv_mpc']:.1f} vs PID {agg['sp_tv_pid']:.1f} ({(1-agg['sp_tv_mpc']/agg['sp_tv_pid'])*100:+.1f}%)")

out = {'model': MODEL_ID, 'n_tracks': N_TRACKS, 'H': H_PLAN, 'lambda_sp': LAMBDA_SP,
       'dsp_max': DSP_MAX, 'agg': agg, 'per_track': all_m}
os.makedirs(f"results/exp_034_{MODEL_ID}", exist_ok=True)
fn = f"results/exp_034_{MODEL_ID}/mpc_sp_H{H_PLAN}_l{LAMBDA_SP}.json"
json.dump(out, open(fn, 'w'), indent=2, default=float)
print(f"Saved: {fn} (耗时 {(time.time()-t0)/60:.1f}min)")
