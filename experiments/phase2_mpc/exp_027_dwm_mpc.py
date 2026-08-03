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
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
import config as cfg
from experiments.phase1_dynamics.exp_025_unified_benchmark import (
    build_model, test_raw, VALVE_IDX, TARGET_IDX, H_OUT, train_min, span_g)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else 'M7'   # 世界模型
PLANNER = sys.argv[2] if len(sys.argv) > 2 else 'grad'  # grad | cem
N_TRACKS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
H_PLAN = int(sys.argv[4]) if len(sys.argv) > 4 else 10  # 规划视野 (≤ H_OUT=18)
ALPHA = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5  # 终端价值系数 (α 扫描用)
SP_TRAJ = int(sys.argv[6]) if len(sys.argv) > 6 else 1     # 方案1: SP轨迹目标 (1=前馈目标, 0=标量目标)
FIX_MODE = sys.argv[7] if len(sys.argv) > 7 else 'none'    # 边界跳变修复: none|hard2|hard5|blend|inert05|inert025

W = cfg.WINDOW_SIZE
N_FEAT = 40
T_SET_MODE = 'real_sp'     # 温度目标: 数据中真实设定值 (二级减温调节阀设定, csv idx 37 = 数据 idx 36)
SP_IDX = 36                # 数据列: 二级减温调节阀设定 (主汽温 SP, SP−PV 均值+0.49°C 验证)
ETA = 0.05                   # 梯度规划步长
E_STEPS = 30                 # 内层梯度步数
LAMBDA1 = 0.1                # 动作变化惩罚 (平滑性)
LAMBDA2 = 0.05               # 动作偏离 last 惩罚
LAMBDA3 = 0.1                # 重叠一致性惩罚 (FIX_MODE='overlap': 新计划头部偏离旧计划尾部)
OVERLAP_REF = None           # 旧计划引用 (simulate 每次重规划前设置; 首块 None)
HARD_DELTA = 0.0             # 边界硬约束幅值 (与 FIX_MODE 独立, 支持 overlap+hard 组合; 0=关)
RISK_LAMBDA = 0.0            # 风险敏感代价权重: J += λ·Σ relu(CVaR_α(超温尾部)−T_MAX)² (0=关)
CVAR_ALPHA = 0.95            # CVaR 分位 (正态假设: k_α = φ(Φ⁻¹(α))/(1−α))
CVAR_K = 2.0627              # α=0.95 → k=2.0627 (φ(1.6449)/0.05)
RISK_SIGMA_ADD = 0.0         # 额外扰动不确定性叠加: σ_total=√(σ_wm²+σ_add²) (扰动世界必须加, 否则风险项看不见扰动)
BENCH_SP_EACH = True         # 评测基准: True=每步真实SP (2026-08-03修正) / False=块起点SP (旧)
SIM_COLLECT_SIGMA = False    # 收集每步预测σ (置信带用, exp_066; 收集到 SIM_SIGMA_BUF)
SIM_SIGMA_BUF = []
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


def build_objective(wm, x_hist, a_seq, t_set, a_last, sp_fut=None):
    """J = Σwₜ(ŷₜ−T_set)² + 终端 + 平滑 + 偏离 + 软约束
    a_seq: [H, 2] 可微, 返回 J (标量, 可反传)
    sp_fut: [H] 未来设定值轨迹 — 目标轨迹版 (方案1): t_set 标量时用标量, sp_fut 给定时目标=轨迹"""
    w = torch.linspace(1.0, 0.8, H_PLAN, device=DEVICE)
    # 规划序列 [H_PLAN,2] → 填充到 H_OUT 步 → [B, H_OUT*2] (与训练一致)
    if H_PLAN < H_OUT:
        tail = a_seq[-1:].repeat(H_OUT - H_PLAN, 1)
        a_full = torch.cat([a_seq, tail], 0)
    else:
        a_full = a_seq[:H_OUT]
    mu, lv = wm(x_hist, a_full.reshape(1, -1))
    mu = mu[0, :H_PLAN]                       # [H] 单目标
    # 目标: 标量 t_set (默认) 或未来 SP 轨迹 (方案1: SP 前馈目标)
    if sp_fut is not None:
        target = sp_fut[:H_PLAN]
    else:
        target = t_set
    err = (mu - target) ** 2
    J = (w * err).sum() / H_PLAN
    # 风险敏感项: CVaR_α 超温尾部 (概率 WM 的 aleatoric σ, 正态假设)
    #   CVaR_t = mu_t + k_α·σ_t; 风险 = relu(CVaR_t − T_MAX)² — 超温尾部概率进规划目标
    if RISK_LAMBDA > 0 and lv is not None:
        sig = torch.exp(lv[0, :H_PLAN] * 0.5)          # 物理空间 σ (denorm_out 返回 lv=2logσ)
        if RISK_SIGMA_ADD > 0:                          # 扰动方差叠加 (扰动世界的总不确定性)
            sig = torch.sqrt(sig ** 2 + RISK_SIGMA_ADD ** 2)
        cvar = mu + CVAR_K * sig
        risk = F.relu(cvar - T_MAX)
        J = J + RISK_LAMBDA * (risk ** 2).sum()
    # 终端价值 (启发式 T1)
    J = J + ALPHA * err[-1]
    # 平滑 + 偏离
    if a_seq.shape[0] > 1:
        J = J + LAMBDA1 * ((a_seq[1:] - a_seq[:-1]) ** 2).sum()
    J = J + LAMBDA2 * ((a_seq - a_last) ** 2).sum()
    # 重叠一致性: 新计划头部 (执行段) 软钉在旧计划未执行段 (平滑切换, 非blend式陈旧执行)
    if FIX_MODE == 'overlap' and OVERLAP_REF is not None:
        m = min(M_STEP, a_seq.shape[0], len(OVERLAP_REF) - M_STEP)
        if m > 0:
            ref = OVERLAP_REF[M_STEP:M_STEP + m]
            J = J + LAMBDA3 * ((a_seq[:m] - ref) ** 2).sum()
    # 软约束: 超区间惩罚
    over = F.relu(mu - T_MAX).pow(2).sum() + F.relu(T_MIN - mu).pow(2).sum()
    J = J + 2.0 * over
    return J


def plan_grad(wm, x_hist, t_set, a_last, a_init=None, sp_fut=None):
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
        J = build_objective(wm, x_hist, a, t_set, a_last, sp_fut)
        J.backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(0, 100)                    # 阀位物理范围
            if FIX_MODE in ('hard2', 'hard5'):  # 边界硬约束: 首步钳制在 a_last±δ (rate constraint)
                delta = 2.0 if FIX_MODE == 'hard2' else 5.0
                a[0:1] = torch.clamp(a[0:1], a_last - delta, a_last + delta)
            elif HARD_DELTA > 0:                # 独立硬约束幅值 (支持 overlap+hard 组合)
                a[0:1] = torch.clamp(a[0:1], a_last - HARD_DELTA, a_last + HARD_DELTA)
            # 内部限幅必须在 a[0] 钳制之后: 否则 a[1] 相对未钳制的 a[0] 限幅, 距 a_last 可达 δ+5
            a[1:] = torch.clamp(a[1:] - a[:-1], -CLIP_DELTA, CLIP_DELTA) + a[:-1]  # |Δa|≤5
        Js.append(J.item())
    return a.detach(), Js


def build_objective_batch(wm, x_hist, a_seqs, t_set, a_last, sp_fut=None):
    """批量目标: a_seqs [N, H, 2] → J [N] (CEM 用, GPU 并行)
    与 build_objective 相同的 J 结构; sp_fut 给定=目标轨迹 (方案1)"""
    N = a_seqs.shape[0]
    w = torch.linspace(1.0, 0.8, H_PLAN, device=DEVICE)
    if H_PLAN < H_OUT:
        tail = a_seqs[:, -1:, :].repeat(1, H_OUT - H_PLAN, 1)
        a_full = torch.cat([a_seqs, tail], 1)
    else:
        a_full = a_seqs[:, :H_OUT, :]
    x_rep = x_hist.repeat(N, 1, 1)
    mu, lv = wm(x_rep, a_full.reshape(N, -1))
    mu = mu[:, :H_PLAN]
    if sp_fut is not None:
        target = sp_fut[:H_PLAN]
    else:
        target = t_set
    err = (mu - target) ** 2
    J = (w * err).sum(1) / H_PLAN
    if RISK_LAMBDA > 0 and lv is not None:  # CEM 路径同款风险项
        sig = torch.exp(lv[:, :H_PLAN] * 0.5)
        cvar = mu + CVAR_K * sig
        risk = F.relu(cvar - T_MAX)
        J = J + RISK_LAMBDA * (risk ** 2).sum(1)
    J = J + ALPHA * err[:, -1]
    if H_PLAN > 1:
        J = J + LAMBDA1 * ((a_seqs[:, 1:] - a_seqs[:, :-1]) ** 2).sum((1, 2))
    J = J + LAMBDA2 * ((a_seqs - a_last.unsqueeze(0).unsqueeze(0)) ** 2).sum((1, 2))  # [1,1,2] 广播
    if FIX_MODE == 'overlap' and OVERLAP_REF is not None:  # CEM 路径同款重叠一致性
        m = min(M_STEP, a_seqs.shape[1], len(OVERLAP_REF) - M_STEP)
        if m > 0:
            ref = OVERLAP_REF[M_STEP:M_STEP + m].unsqueeze(0).unsqueeze(0)
            J = J + LAMBDA3 * ((a_seqs[:, :m] - ref) ** 2).sum((1, 2))
    over = F.relu(mu - T_MAX).pow(2).sum(1) + F.relu(T_MIN - mu).pow(2).sum(1)
    J = J + 2.0 * over
    return J


def plan_cem(wm, x_hist, t_set, a_last, sp_fut=None):
    """CEM: 200 采样 (GPU 批量) + 精英加权, 5 轮
    a_last: [2] → 初始均值 H 步恒定"""
    a_mean = a_last.unsqueeze(0).repeat(H_PLAN, 1).clone()
    a_std = torch.full((H_PLAN, 2), 5.0, device=DEVICE)
    for _ in range(CEM_ITERS):
        samples = a_mean + a_std * torch.randn(N_CEM_SAMPLES, H_PLAN, 2, device=DEVICE)
        samples.clamp_(0, 100)
        Js = build_objective_batch(wm, x_hist, samples, t_set, a_last, sp_fut)
        _, idx = torch.topk(Js, N_CEM_ELITE, largest=False)
        elite = samples[idx]
        a_mean = elite.mean(0)
        a_std = elite.std(0).clamp_(CEM_SIGMA_MIN, 10)
    return a_mean.detach(), [Js.min().item()]


M_STEP = 6  # 多步执行: 每步执行 a_plan[0:M_STEP], 窗口推进 M_STEP 步 (对齐动作时标 60s)
DIST_AMP = 0.3  # 过程扰动幅度 (°C/步 随机游走, 模拟负荷/燃料扰动; 0=无扰动)

def _dist_rng(seed):
    return np.random.default_rng(seed)

def simulate(wm, track_idx, planner, n_steps=120, seed=42):
    """反事实仿真: 每步策略动作 → WM 闭环预测温度 → 窗口推进 (receding horizon)
    协议 (plan.md §2.1): PID 组=真实阀位→真实温度(基准); WM-MPC 组=MPC 动作→WM 预测温度。
    多步执行: 每次规划执行 M_STEP 步 (动作效应时标 60s, 短程 ŷ₁ 对动作响应≈0 需对齐)
    公平性: 两条轨迹都在 WM+真实非温度状态推进下跑, 模型误差对两策略一视同仁。
    track_idx: test 集起始索引
    返回: (mpc_temp, pid_temp, t_set_traj, mpc_actions, pid_actions)
    """
    global OVERLAP_REF  # 模块级引用: build_objective 的重叠一致性项需要 (函数内赋值默认局部)
    np.random.seed(seed)
    i = track_idx
    N = len(test_raw)
    pid_temp, mpc_temp, t_set_traj = [], [], []
    pid_actions, mpc_actions = [], []
    a_last = torch.FloatTensor(test_raw[i+W, VALVE_IDX]).to(DEVICE)
    a_init = None
    rng = _dist_rng(seed + track_idx) if DIST_AMP > 0 else None
    d_state = 0.0
    # 初始窗口: 真实
    win = torch.FloatTensor(test_raw[i:i+W]).unsqueeze(0).to(DEVICE)
    if SIM_COLLECT_SIGMA:
        SIM_SIGMA_BUF.clear()
    t = 0
    while t < n_steps:  # 2026-08-03 修正: 原 for t in range(0,n_steps,M_STEP) 固定20块, H_PLAN<M_STEP 时 n_exec 截断导致轨迹变短 (H=1→20步)
        gi = i + t
        if gi + W + M_STEP >= N: break
        # MPC 规划 (基于当前窗口, 每次 M_STEP 步执行一次)
        t_set = torch.tensor(float(test_raw[gi+W, SP_IDX]), dtype=torch.float32, device=DEVICE)  # 真实 SP (当前值)
        sp_fut = torch.FloatTensor(test_raw[gi+W:gi+W+H_OUT, SP_IDX]).to(DEVICE)  # 未来 SP 轨迹
        if not SP_TRAJ:
            sp_fut = None  # 标量目标模式: 用当前 SP
        if planner == 'grad':
            OVERLAP_REF = a_init  # 旧计划引用 (重叠一致性用; 首块 None)
            a_plan, Js = plan_grad(wm, win, t_set, a_last, a_init, sp_fut)
        else:
            OVERLAP_REF = a_init
            a_plan, Js = plan_cem(wm, win, t_set, a_last, sp_fut)
        # WM 闭环: MPC 动作序列 → 预测 M_STEP 步温度
        with torch.no_grad():
            if H_PLAN < H_OUT:
                a_full = torch.cat([a_plan, a_plan[-1:].repeat(H_OUT - H_PLAN, 1)], 0)
            else:
                a_full = a_plan[:H_OUT]
            if getattr(wm, 'use_sp', False):
                mu, lv = wm(win, a_full.reshape(1, -1), sp_fut.unsqueeze(0))
            else:
                mu, lv = wm(win, a_full.reshape(1, -1))
        # 多步执行: 依次执行 a_plan[0..M_STEP-1], 窗口逐步推进 (对应预测温度)
        n_exec = min(M_STEP, len(a_plan), len(mu[0]), n_steps - t)
        # 边界跳变修复: 构造实际执行块 a_exec (none/hard: 执行=计划; blend/inert: 执行≠计划)
        prev_plan = a_init  # 旧计划 (warm-start 继承来源), 首块为 None
        blended = FIX_MODE == 'blend' and prev_plan is not None and len(prev_plan) > M_STEP
        if blended:
            # 加权融合: a_exec[j] = w_j·a_old[M_STEP+j] + (1−w_j)·a_plan[j], w_j 线性 1→0
            # 连续性由构造保证: j=0 时完全继承旧计划未执行段, 而旧计划内部 |Δa|≤5
            a_exec = a_plan[:n_exec].clone()
            for j in range(n_exec):
                w = 1.0 - j / float(n_exec)
                a_old_step = prev_plan[min(M_STEP + j, len(prev_plan) - 1)]
                a_exec[j] = w * a_old_step + (1.0 - w) * a_plan[j]
        elif FIX_MODE.startswith('inert'):
            # 惯性块: 一阶惯性环节作用于执行流 a_exec[j] = a_exec[j-1] + K·(a_plan[j] − a_exec[j-1])
            K = 0.5 if FIX_MODE == 'inert05' else 0.25
            a_exec = a_plan[:n_exec].clone()
            a_prev_ex = a_last
            for j in range(n_exec):
                a_exec[j] = a_prev_ex + K * (a_plan[j] - a_prev_ex)
                a_prev_ex = a_exec[j]
        else:
            a_exec = a_plan[:n_exec]
        # PID 参考: 真实动作 + WM 闭环预测 (同一扰动世界, 公平协议)
        with torch.no_grad():
            a_pid_full = torch.FloatTensor(test_raw[gi:gi+H_OUT, VALVE_IDX]).unsqueeze(0).to(DEVICE)
            if getattr(wm, 'use_sp', False):
                mu_pid, _ = wm(win, a_pid_full.reshape(1, -1), sp_fut.unsqueeze(0))
            else:
                mu_pid, _ = wm(win, a_pid_full.reshape(1, -1))
        # 执行≠计划时: 用实际执行动作重算 WM 温度 (闭环一致性, 否则评测的是未执行的动作)
        if blended or FIX_MODE.startswith('inert'):
            if H_PLAN < H_OUT:
                tail = a_plan[-1:].repeat(H_OUT - H_PLAN, 1)
                a_full_exec = torch.cat([a_exec, a_plan[n_exec:], tail], 0)
            else:
                a_full_exec = torch.cat([a_exec, a_plan[n_exec:]], 0)[:H_OUT]
            with torch.no_grad():
                if getattr(wm, 'use_sp', False):
                    mu_exec, lv_exec = wm(win, a_full_exec.reshape(1, -1), sp_fut.unsqueeze(0))
                else:
                    mu_exec, lv_exec = wm(win, a_full_exec.reshape(1, -1))
        else:
            mu_exec, lv_exec = mu, lv
        for j in range(n_exec):
            gi_j = gi + j
            if gi_j + W + 1 >= N: break
            if SIM_COLLECT_SIGMA:
                SIM_SIGMA_BUF.append(float(torch.exp(lv_exec[0, j] * 0.5).item()))
            pid_a = test_raw[gi_j+W, VALVE_IDX]
            if rng is not None:  # 过程扰动 (两策略共享同一扰动序列)
                d_state = 0.9 * d_state + rng.normal(0, DIST_AMP)
            pid_t = mu_pid[0, j].item() + (d_state if rng is not None else 0.0)
            y_j = mu_exec[0, j].item()
            if rng is not None:
                y_j = y_j + d_state
            next_row = torch.FloatTensor(test_raw[gi_j+W]).unsqueeze(0).unsqueeze(0).to(DEVICE)
            next_row[0, 0, TARGET_IDX] = y_j
            win = torch.cat([win[:, 1:, :], next_row], 1)
            mpc_temp.append(y_j)
            pid_temp.append(pid_t)
            t_set_traj.append(float(test_raw[gi_j + W, SP_IDX]) if BENCH_SP_EACH else t_set.item())
            mpc_actions.append(a_exec[j].cpu().numpy())
            pid_actions.append(pid_a)
        a_last = a_exec[n_exec - 1]
        a_init = a_plan  # warm-start (整段继承)
        t += n_exec
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
    # 积分指标 (dx=1步=10s; 单位 °C·10s) — 论文补充指标 (IAE/ITAE/超温积分)
    m['iae_mpc'] = float(np.trapz(np.abs(mpc_t - t_set_traj)))
    m['iae_pid'] = float(np.trapz(np.abs(pid_t - t_set_traj)))
    m['itae_mpc'] = float(np.trapz(np.arange(len(mpc_t)) * np.abs(mpc_t - t_set_traj)))
    m['itae_pid'] = float(np.trapz(np.arange(len(pid_t)) * np.abs(pid_t - t_set_traj)))
    m['overtemp_int_mpc'] = float(np.trapz(np.maximum(mpc_t - T_MAX, 0.0)))
    m['overtemp_int_pid'] = float(np.trapz(np.maximum(pid_t - T_MAX, 0.0)))
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
    fn = f"results/exp_027_{MODEL_ID}/mpc_{PLANNER}_H{H_PLAN}_a{ALPHA}.json"
    json.dump(out, open(fn, 'w'), indent=2, default=float)
    print(f"Saved: {fn} (耗时 {(time.time()-t0)/60:.1f}min)")


if __name__ == '__main__':
    main()
