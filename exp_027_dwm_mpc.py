#!/usr/bin/env python3
"""
exp_027_dwm_mpc.py — Phase 2a: DWM-MPC 主循环 (梯度规划 vs CEM 对照)
=====================================================================
主路线 A — 梯度优化动作序列 (backprop through WM):
  初始化 a⁽⁰⁾ (warm-start: 上一步最优 或 PID 基线)
  E 轮 Adam 梯度上升: J = Σwₜ(ŷₜ−T_set)² + λ₁ΣΔa² + λ₂Σ(a−a_last)² + α(ŷ_H−T_set)²
  执行 a₁ (receding horizon, 每步真实状态重建窗口)

对照路线 B — CEM: 同目标函数, 200 采样 + 精英加权 + σ_min=0.05

评测: 反事实仿真 (真实状态推进 + MPC 动作) vs PID 真实轨迹
  指标: 温度 RMSE vs T_set / 超温次数 / 总变差 TV / |Δa| 违规
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from experiments.phase1_dynamics.exp_025_unified_benchmark import (
    build_model, test_raw, VALVE_IDX, TARGET_IDX, H_OUT, train_min, span_g)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else 'M7'   # 世界模型
PLANNER = sys.argv[2] if len(sys.argv) > 2 else 'grad'  # grad | cem
N_TRACKS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
H_PLAN = int(sys.argv[4]) if len(sys.argv) > 4 else 10  # 规划视野 (≤ H_OUT=18)

W = cfg.WINDOW_SIZE
N_FEAT = 40
T_SET_MODE = 'window_mean'   # 温度目标: 窗口均值 (每样本自适应)
ALPHA = 0.5                  # 终端价值系数 (启发式 T1)
ETA = 0.05                   # 梯度规划步长
E_STEPS = 30                 # 内层梯度步数
LAMBDA1 = 0.1                # 动作变化惩罚 (平滑性)
LAMBDA2 = 0.05               # 动作偏离 last 惩罚
N_CEM_SAMPLES = 200          # CEM 采样数
N_CEM_ELITE = 20             # 精英数
CEM_ITERS = 5
CEM_SIGMA_MIN = 0.05
CLIP_DELTA = 5.0             # |Δa| ≤ 5%/step 硬约束
T_MIN, T_MAX = 540., 575.    # 软约束区间


def load_wm():
    model = build_model(MODEL_ID).to(DEVICE).eval()
    ck = torch.load(f"results/exp_025_{MODEL_ID}/checkpoints/best_model.pth",
                    map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict'])
    return model


def build_objective(wm, x_hist, a_seq, t_set, a_last):
    """J = Σwₜ(ŷₜ−T_set)² + 终端 + 平滑 + 偏离 + 软约束
    a_seq: [H, 2] 可微, 返回 J (标量, 可反传)"""
    w = torch.linspace(1.0, 0.8, H_PLAN, device=DEVICE)
    # 规划序列 [H_PLAN,2] → 填充到 H_OUT 步 → [B, H_OUT*2] (与训练一致)
    if H_PLAN < H_OUT:
        tail = a_seq[-1:].repeat(H_OUT - H_PLAN, 1)
        a_full = torch.cat([a_seq, tail], 0)
    else:
        a_full = a_seq[:H_OUT]
    mu, _ = wm(x_hist, a_full.reshape(1, -1))
    mu = mu[0, :H_PLAN]                       # [H] 单目标
    err = (mu - t_set) ** 2
    J = (w * err).sum() / H_PLAN
    # 终端价值 (启发式 T1)
    J = J + ALPHA * err[-1]
    # 平滑 + 偏离
    if a_seq.shape[0] > 1:
        J = J + LAMBDA1 * ((a_seq[1:] - a_seq[:-1]) ** 2).sum()
    J = J + LAMBDA2 * ((a_seq - a_last) ** 2).sum()
    # 软约束: 超区间惩罚
    over = F.relu(mu - T_MAX).pow(2).sum() + F.relu(T_MIN - mu).pow(2).sum()
    J = J + 2.0 * over
    return J


def plan_grad(wm, x_hist, t_set, a_last, a_init=None):
    """梯度规划: Adam 上升 E 轮, warm-start
    a_last: [2] 上一步阀位 → 初始化为 H 步恒定序列"""
    if a_init is not None:
        a = a_init.clone().detach().requires_grad_(True)
    else:
        a = a_last.unsqueeze(0).repeat(H_PLAN, 1).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=ETA)
    Js = []
    for _ in range(E_STEPS):
        opt.zero_grad()
        J = build_objective(wm, x_hist, a, t_set, a_last)
        J.backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(0, 100)                    # 阀位物理范围
            a[1:] = torch.clamp(a[1:] - a[:-1], -CLIP_DELTA, CLIP_DELTA) + a[:-1]  # |Δa|≤5
        Js.append(J.item())
    return a.detach(), Js


def build_objective_batch(wm, x_hist, a_seqs, t_set, a_last):
    """批量目标: a_seqs [N, H, 2] → J [N] (CEM 用, GPU 并行)
    与 build_objective 相同的 J 结构"""
    N = a_seqs.shape[0]
    w = torch.linspace(1.0, 0.8, H_PLAN, device=DEVICE)
    if H_PLAN < H_OUT:
        tail = a_seqs[:, -1:, :].repeat(1, H_OUT - H_PLAN, 1)
        a_full = torch.cat([a_seqs, tail], 1)
    else:
        a_full = a_seqs[:, :H_OUT, :]
    x_rep = x_hist.repeat(N, 1, 1)
    mu, _ = wm(x_rep, a_full.reshape(N, -1))
    mu = mu[:, :H_PLAN]
    err = (mu - t_set) ** 2
    J = (w * err).sum(1) / H_PLAN
    J = J + ALPHA * err[:, -1]
    if H_PLAN > 1:
        J = J + LAMBDA1 * ((a_seqs[:, 1:] - a_seqs[:, :-1]) ** 2).sum((1, 2))
    J = J + LAMBDA2 * ((a_seqs - a_last.unsqueeze(0).unsqueeze(0)) ** 2).sum((1, 2))  # [1,1,2] 广播
    over = F.relu(mu - T_MAX).pow(2).sum(1) + F.relu(T_MIN - mu).pow(2).sum(1)
    J = J + 2.0 * over
    return J


def plan_cem(wm, x_hist, t_set, a_last):
    """CEM: 200 采样 (GPU 批量) + 精英加权, 5 轮
    a_last: [2] → 初始均值 H 步恒定"""
    a_mean = a_last.unsqueeze(0).repeat(H_PLAN, 1).clone()
    a_std = torch.full((H_PLAN, 2), 5.0, device=DEVICE)
    for _ in range(CEM_ITERS):
        samples = a_mean + a_std * torch.randn(N_CEM_SAMPLES, H_PLAN, 2, device=DEVICE)
        samples.clamp_(0, 100)
        Js = build_objective_batch(wm, x_hist, samples, t_set, a_last)
        _, idx = torch.topk(Js, N_CEM_ELITE, largest=False)
        elite = samples[idx]
        a_mean = elite.mean(0)
        a_std = elite.std(0).clamp_(CEM_SIGMA_MIN, 10)
    return a_mean.detach(), [Js.min().item()]


def simulate(wm, track_idx, planner, n_steps=120, seed=42):
    """反事实仿真: 每步策略动作 → WM 闭环预测温度 → 窗口推进 (receding horizon)
    协议 (plan.md §2.1): PID 组=真实阀位→真实温度(基准); WM-MPC 组=MPC 动作→WM 预测温度。
    公平性: 两条轨迹都在 WM+真实非温度状态推进下跑, 模型误差对两策略一视同仁。
    track_idx: test 集起始索引
    返回: (mpc_temp, pid_temp, t_set_traj, mpc_actions, pid_actions)
    """
    np.random.seed(seed)
    i = track_idx
    N = len(test_raw)
    pid_temp, mpc_temp, t_set_traj = [], [], []
    pid_actions, mpc_actions = [], []
    a_last = torch.FloatTensor(test_raw[i+W, VALVE_IDX]).to(DEVICE)
    a_init = None
    # 初始窗口: 真实
    win = torch.FloatTensor(test_raw[i:i+W]).unsqueeze(0).to(DEVICE)
    for t in range(n_steps):
        gi = i + t
        if gi + W + 1 >= N: break
        # PID 真实动作/温度 (下一步)
        pid_a = test_raw[gi+W, VALVE_IDX]
        pid_t = test_raw[gi+W, TARGET_IDX]
        # MPC 规划 (基于当前窗口)
        t_set = torch.tensor(np.mean(win[0, :, TARGET_IDX].cpu().numpy()), dtype=torch.float32, device=DEVICE)
        if planner == 'grad':
            a_plan, Js = plan_grad(wm, win, t_set, a_last, a_init)
        else:
            a_plan, Js = plan_cem(wm, win, t_set, a_last)
        a1 = a_plan[0]
        # WM 闭环: MPC 动作序列(填充到H_OUT) → 预测第一步温度
        with torch.no_grad():
            if H_PLAN < H_OUT:
                a_full = torch.cat([a_plan, a_plan[-1:].repeat(H_OUT - H_PLAN, 1)], 0)
            else:
                a_full = a_plan[:H_OUT]
            mu, _ = wm(win, a_full.reshape(1, -1))
            y1 = mu[0, 0].item()
        # 窗口推进: 真实数据下一行 (温度列替换为模型预测)
        next_row = torch.FloatTensor(test_raw[gi+W]).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,40]
        next_row[0, 0, TARGET_IDX] = y1
        win = torch.cat([win[:, 1:, :], next_row], 1)
        # 记录
        mpc_temp.append(y1)
        pid_temp.append(pid_t)
        t_set_traj.append(t_set.item())
        mpc_actions.append(a1.cpu().numpy())
        pid_actions.append(pid_a)
        a_last = a1
        a_init = a_plan  # warm-start
    return (np.array(mpc_temp), np.array(pid_temp), np.array(t_set_traj),
            np.array(mpc_actions), np.array(pid_actions))


def metrics(mpc_t, pid_t, t_set_traj, mpc_a, pid_a):
    """温度 RMSE vs T_set + 波动/超温/动作TV/违规"""
    m = {}
    m['rmse_mpc'] = float(np.sqrt(np.mean((mpc_t - t_set_traj)**2)))
    m['rmse_pid'] = float(np.sqrt(np.mean((pid_t - t_set_traj)**2)))
    m['temp_std_mpc'] = float(mpc_t.std())
    m['temp_std_pid'] = float(pid_t.std())
    m['temp_range_mpc'] = float(mpc_t.max() - mpc_t.min())
    m['temp_range_pid'] = float(pid_t.max() - pid_t.min())
    # 动作 TV (总变差) — 越小越平滑
    m['act_tv_mpc'] = float(np.abs(np.diff(mpc_a, axis=0)).mean())
    m['act_tv_pid'] = float(np.abs(np.diff(pid_a, axis=0)).mean())
    # |Δa| 违规 (阈值 5)
    m['viol_mpc'] = int((np.abs(np.diff(mpc_a, axis=0)) > CLIP_DELTA).sum())
    m['viol_pid'] = int((np.abs(np.diff(pid_a, axis=0)) > CLIP_DELTA).sum())
    # 超温 (>575)
    m['overtemp_mpc'] = int((mpc_t > T_MAX).sum())
    m['overtemp_pid'] = int((pid_t > T_MAX).sum())
    return m


def main():
    wm = load_wm()
    print(f"WM: {MODEL_ID} | planner: {PLANNER} | H={H_PLAN} | α={ALPHA} | η={ETA} E={E_STEPS}")
    N = len(test_raw)
    np.random.seed(42)
    starts = np.random.choice(range(N - W - H_OUT - 120), N_TRACKS, replace=False)
    all_m, all_act = [], []
    t0 = time.time()
    for k, s in enumerate(starts):
        mpc_t, pid_t, tset, mpc_a, pid_a = simulate(wm, s, PLANNER)
        m = metrics(mpc_t, pid_t, tset, mpc_a, pid_a)
        all_m.append(m)
        print(f"  [{k+1}/{N_TRACKS}] RMSE {m['rmse_mpc']:.3f}/{m['rmse_pid']:.3f} "
              f"TV {m['act_tv_mpc']:.3f}/{m['act_tv_pid']:.3f} "
              f"超温 {m['overtemp_mpc']}/{m['overtemp_pid']} "
              f"违规 {m['viol_mpc']}/{m['viol_pid']}")
    # 汇总
    agg = {}
    for k in all_m[0]:
        agg[k] = float(np.mean([m[k] for m in all_m]))
    print(f"\n===== {MODEL_ID} {PLANNER} H={H_PLAN} 汇总 ({N_TRACKS} 轨迹) =====")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")
    out = {'model': MODEL_ID, 'planner': PLANNER, 'n_tracks': N_TRACKS, 'H': H_PLAN,
           'alpha': ALPHA, 'eta': ETA, 'e_steps': E_STEPS, 'agg': agg,
           'per_track': all_m}
    os.makedirs(f"results/exp_027_{MODEL_ID}", exist_ok=True)
    fn = f"results/exp_027_{MODEL_ID}/mpc_{PLANNER}_H{H_PLAN}.json"
    json.dump(out, open(fn, 'w'), indent=2, default=float)
    print(f"Saved: {fn} (耗时 {(time.time()-t0)/60:.1f}min)")


if __name__ == '__main__':
    main()
