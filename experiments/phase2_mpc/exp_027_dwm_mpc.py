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
LAMBDA1 = float(os.environ.get('LAMBDA1', '0.1'))   # 动作变化惩罚 (平滑性) — env可覆盖 (2026-08-04 最终配置=0)
LAMBDA1_2ND = 0.0            # 二阶平滑 (Δ²a², 抑制驻波/standing wave; 2026-08-03)
EXEC_SMA = int(os.environ.get('EXEC_SMA', '1'))  # 执行端因果移动平均窗 (>1: 滤驻波留趋势; 2026-08-03)
EXEC_KF = float(os.environ.get('EXEC_KF', '0.0'))  # 执行端卡尔曼速度过程噪声 (>0: 启用; 用户建议 2026-08-03)
EXEC_DEADZONE = float(os.environ.get('EXEC_DEADZONE', '0.0'))  # 执行死区 (变化<thr 保持; 2026-08-03)
PID_KP = float(os.environ.get('PID_KP', '40.0'))   # 真 PID 增益 (阶跃对比用; 2026-08-03)
PID_KI = float(os.environ.get('PID_KI', '8.0'))
PID_KD = float(os.environ.get('PID_KD', '0.0'))
# ── 安全 MPC (2026-08-03, 现场投切边界必需): 状态密度检测 + PID 回退 ──
SAFE_Z_THR = float(os.environ.get('SAFE_Z_THR', '0.0'))   # 触发回退的 z-score 阈值 (0=关)
SAFE_HOLD = int(os.environ.get('SAFE_HOLD', '3'))         # 连续 N 块低置信才回退 (滞回)
SAFE_RELEASE = int(os.environ.get('SAFE_RELEASE', '5'))   # 连续 M 块高置信才重新投 MPC
SAFE_FEATS = [TARGET_IDX, SP_IDX, VALVE_IDX[0], VALVE_IDX[1]]  # 置信度特征
DIST_MEAN = {}; DIST_STD = {}   # 全段分布统计 (数据加载后填充; 2026-08-03)
SAFE_LOG = []                   # (t, score, feature, action_mode) 回退记录
OFFSET_GAIN = float(os.environ.get('OFFSET_GAIN', '0.0'))  # offset-free 补偿增益 (治模型系统偏差; 2026-08-03)
OFFSET_EMA = float(os.environ.get('OFFSET_EMA', '0.3'))    # 偏差估计 EMA 系数
LAMBDA2 = float(os.environ.get('LAMBDA2', '0.05'))   # 动作偏离 last 惩罚 — env可覆盖 (2026-08-04 最终配置=0)
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
INT_LAMBDA = float(os.environ.get('INT_LAMBDA', '0.0'))  # 积分项权重 (等效PI积分, 治静差; 2026-08-03)
INT_GAIN = float(os.environ.get('INT_GAIN', '0.0'))      # 跨规划周期积分增益 (SP偏置=PI的I项; 2026-08-03)
DT = 10                      # 采样间隔 (s) — 合成 SP 阶跃时间轴 (2026-08-03)
N_CEM_ELITE = 20             # 精英数
CEM_ITERS = 5
CEM_SIGMA_MIN = 0.05
CLIP_DELTA = 5.0             # |Δa| ≤ 5%/step 硬约束
T_MIN, T_MAX = 540., 575.    # 软约束区间
# ── 压线控制 (2026-08-03 用户要求: 贴线运行 + 阀位留安全裕度) ──
# 压线语义 (冒烟 exp_074 证实): 超温重罚→操作点下移→平均温度反而降 (mean_err -0.23→-0.40)。
# 正确结构 = 欠温重罚 (掉温=效率损失, 平均气温考核) + 超温 571-575 轻罚 + T_MAX=575 软约束兜底安全。
W_OVER = 1.0               # 超温权重 (e>0; 575 由软约束兜底, 这里只需轻)
W_UNDER = 1.0              # 欠温权重 (e<0; >1=压线: 掉温重罚, 平均气温顶上去)
LAMBDA_U = 0.0             # 阀位安全裕度惩罚 (0=关; >0: 接近 U_LO/U_HI 时惩罚, 留余量)
U_LO = np.array([2.0, 0.0])  # 阀位安全带下界 (per-valve; 数据 p5≈[0.2,-0.75], 取整留余量)
U_HI = np.array([43.0, 32.0])  # 阀位上界 (数据 p95≈[43.6,32.6], 校准 2026-08-03)


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
    # 非对称压线代价: 欠温重罚/超温轻罚 (用户: 贴线运行, 提高平均气温)
    # 冒烟证据 (exp_074): 超温重罚→操作点下移→均值降; 欠温重罚→均值顶上去 (压线)
    if W_UNDER != 1.0 or W_OVER != 1.0:
        e = mu - target
        w_e = torch.where(e > 0, torch.full_like(e, W_OVER), torch.full_like(e, W_UNDER))
        err = w_e * e ** 2
    J = (w * err).sum() / H_PLAN
    # 积分项 (等效 PI 积分作用, 治静差, 2026-08-03): (Σ原始偏差/H)² — 持续偏差累积迫使动作
    if INT_LAMBDA > 0:
        e_lin = mu - target
        J = J + INT_LAMBDA * (e_lin.sum() / H_PLAN) ** 2
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
    # 二阶平滑 (惩罚加速度 Δ²a, 抑制高频驻波/standing wave, 保留低频趋势; 2026-08-03)
    if LAMBDA1_2ND > 0 and a_seq.shape[0] > 2:
        d2 = a_seq[2:] - 2.0 * a_seq[1:-1] + a_seq[:-2]
        J = J + LAMBDA1_2ND * (d2 ** 2).sum()
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
    # 阀位安全裕度: 接近 U_LO/U_HI 时惩罚 (留余量应对扰动; 用户: 阀门动作有上限, 留够安全余度)
    if LAMBDA_U > 0:
        u_lo = torch.as_tensor(U_LO, dtype=torch.float32, device=DEVICE)
        u_hi = torch.as_tensor(U_HI, dtype=torch.float32, device=DEVICE)
        u_pen = F.relu(a_seq - u_hi).pow(2).sum() + F.relu(u_lo - a_seq).pow(2).sum()
        J = J + LAMBDA_U * u_pen
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
    if INT_LAMBDA > 0:  # 积分项 (等效 PI 积分, 与 build_objective 一致)
        e_lin = mu - target
        J = J + INT_LAMBDA * (e_lin.sum(1) / H_PLAN) ** 2
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
            ref = OVERLAP_REF[M_STEP:M_STEP + m]  # [m, 2] 广播到 [N, m, 2] (2026-08-03修复: 原双unsqueeze形状错)
            J = J + LAMBDA3 * ((a_seqs[:, :m] - ref) ** 2).sum((1, 2))
    over = F.relu(mu - T_MAX).pow(2).sum(1) + F.relu(T_MIN - mu).pow(2).sum(1)
    J = J + 2.0 * over
    if LAMBDA_U > 0:  # 批量路径阀位裕度 (CEM)
        u_lo = torch.as_tensor(U_LO, dtype=torch.float32, device=DEVICE)
        u_hi = torch.as_tensor(U_HI, dtype=torch.float32, device=DEVICE)
        u_pen = F.relu(a_seqs - u_hi).pow(2).sum((1, 2)) + F.relu(u_lo - a_seqs).pow(2).sum((1, 2))
        J = J + LAMBDA_U * u_pen
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

def _safe_score(win_np):
    """状态置信度: 窗口关键特征均值 vs 全段分布的 z-score 最大绝对值
    返回 (score, 最差特征名) — 现场投切边界判定 (2026-08-03)"""
    zs = {}
    for fi in SAFE_FEATS:
        m = float(np.mean(win_np[:, fi])); s = float(np.std(win_np[:, fi]))
        zm = (m - DIST_MEAN[fi]) / (DIST_STD[fi] + 1e-8)
        zs[fi] = abs(zm)
    fi_worst = max(zs, key=zs.get)
    return zs[fi_worst], fi_worst

def _simulate_pid(wm, track_idx, n_steps=250, sp_step=None, seed=42):
    """真 PID 闭环 (2026-08-03): 每步 e=SP−T → u=Kp·e+Ki·∫e+Kd·de → WM 预测 → 回填
    与 MPC 同一虚拟世界协议 (WM+扰动), 公平对比阶跃响应控制参数"""
    np.random.seed(seed + track_idx)
    i = int(track_idx)
    rng = _dist_rng(seed + track_idx) if DIST_AMP > 0 else None
    d_state = 0.0
    win = torch.FloatTensor(test_raw[i:i+W]).unsqueeze(0).to(DEVICE)
    sp_base = float(test_raw[i+W-1, SP_IDX]) if sp_step is not None else None
    pid_I, pid_e_prev = 0.0, 0.0
    a_cur = float(test_raw[i+W-1, VALVE_IDX[1]])
    mpc_temp, pid_temp, t_set_traj, mpc_actions = [], [], [], []
    for t in range(n_steps):
        gi = i + t
        if gi + W + 1 >= len(test_raw): break
        sp_now = float(test_raw[gi+W, SP_IDX])
        if sp_step is not None:
            sp_now = sp_base + (sp_step[1] if t * DT >= sp_step[0] else 0.0)
        y_meas = float(win[0, -1, TARGET_IDX])
        e = sp_now - y_meas
        pid_I = float(np.clip(pid_I + e * DT, -300.0, 300.0))
        u = PID_KP * e + PID_KI * pid_I + PID_KD * (e - pid_e_prev) / DT
        pid_e_prev = e
        a_cur = float(np.clip(a_cur + 0.5 * (u - a_cur), 0.0, 45.0))  # 一阶惯性执行 (阀门实际速率)
        a_full = torch.full((1, H_OUT * 2), float(test_raw[gi+W-1, VALVE_IDX[0]]), device=DEVICE)
        a_full[0, 1::2] = a_cur
        with torch.no_grad():
            if getattr(wm, 'use_sp', False):
                mu, _ = wm(win, a_full, torch.FloatTensor([sp_now] * H_OUT).unsqueeze(0).to(DEVICE))
            else:
                mu, _ = wm(win, a_full)
        y_j = float(mu[0, 0])
        if rng is not None:
            d_state = 0.9 * d_state + rng.normal(0, DIST_AMP)
            y_j = y_j + d_state
        next_row = torch.FloatTensor(test_raw[gi+W]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        next_row[0, 0, TARGET_IDX] = y_j
        win = torch.cat([win[:, 1:, :], next_row], 1)
        mpc_temp.append(y_j)
        pid_temp.append(float(test_raw[gi+W, TARGET_IDX]))
        t_set_traj.append(sp_now)
        mpc_actions.append(np.array([a_full[0, 0].item(), a_cur]))
    return (np.array(mpc_temp), np.array(pid_temp), np.array(t_set_traj),
            np.array(mpc_actions), np.zeros((len(mpc_temp), 2)))

def simulate(wm, track_idx, planner, n_steps=120, seed=42, sp_step=None):
    """反事实仿真: 每步策略动作 → WM 闭环预测温度 → 窗口推进 (receding horizon)
    协议 (plan.md §2.1): PID 组=真实阀位→真实温度(基准); WM-MPC 组=MPC 动作→WM 预测温度。
    多步执行: 每次规划执行 M_STEP 步 (动作效应时标 60s, 短程 ŷ₁ 对动作响应≈0 需对齐)
    公平性: 两条轨迹都在 WM+真实非温度状态推进下跑, 模型误差对两策略一视同仁。
    track_idx: test 集起始索引
    sp_step: (step_time_s, step_amp) 合成 SP 阶跃注入 (阶跃响应测试, 2026-08-03):
             SP(t) = SP_base + amp·1(t ≥ step_time), SP_base = 窗口末真实 SP (恒定基准)
             评测基准 t_set_traj 同步用合成 SP
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
    sp_base = float(test_raw[i+W-1, SP_IDX]) if sp_step is not None else None
    int_state = 0.0          # 积分状态 (等效 PI 的 I 项, 跨规划周期累积; 2026-08-03)
    t_set_eff_prev = None
    # 执行端卡尔曼滤波状态 (位置+速度, 跨块连续; 2026-08-03) — 滤驻波且不延迟趋势
    a0 = test_raw[i+W-1, VALVE_IDX]
    kf_state = [{'p': float(a0[ch]), 'v': 0.0, 'P': np.eye(2)} for ch in range(2)]
    d_est = 0.0   # offset-free 模型偏差估计 (d = y_meas − y_pred, EMA; 2026-08-03)
    # 安全 MPC 状态 (2026-08-03)
    if not DIST_MEAN:
        for fi in SAFE_FEATS:
            DIST_MEAN[fi] = float(np.mean(test_raw[:, fi]))
            DIST_STD[fi] = float(np.std(test_raw[:, fi]))
    low_cnt = high_cnt = 0; safe_active = False; safe_I = 0.0
    t = 0
    while t < n_steps:  # 2026-08-03 修正: 原 for t in range(0,n_steps,M_STEP) 固定20块, H_PLAN<M_STEP 时 n_exec 截断导致轨迹变短 (H=1→20步)
        gi = i + t
        if gi + W + M_STEP >= N: break
        # MPC 规划 (基于当前窗口, 每次 M_STEP 步执行一次)
        if sp_step is not None:  # 合成 SP: 恒定基准 + 阶跃
            t_now = t * DT
            step_on = t_now >= sp_step[0]
            sp_now = sp_base + (sp_step[1] if step_on else 0.0)
            t_set = torch.tensor(sp_now, dtype=torch.float32, device=DEVICE)
            fut_base = np.full(H_OUT, sp_now)
            sp_fut = torch.FloatTensor(fut_base).to(DEVICE)
            if SP_TRAJ:
                t_on = np.arange(H_OUT) * DT + t_now
                fut = np.where(t_on >= sp_step[0], sp_base + sp_step[1], sp_base)
                sp_fut = torch.FloatTensor(fut).to(DEVICE)
        else:  # 真实 SP (默认)
            t_set = torch.tensor(float(test_raw[gi+W, SP_IDX]), dtype=torch.float32, device=DEVICE)  # 真实 SP (当前值)
            sp_fut = torch.FloatTensor(test_raw[gi+W:gi+W+H_OUT, SP_IDX]).to(DEVICE)  # 未来 SP 轨迹
            if not SP_TRAJ:
                sp_fut = None  # 标量目标模式: 用当前 SP
        # 积分状态: 用上次累积值计算 SP 偏置 (等效 PI 的 I 项, 治静差; 2026-08-03)
        t_set_eff = t_set - OFFSET_GAIN * d_est + INT_GAIN * int_state   # offset-free: 模型偏差补偿
        t_set_eff_prev = float(t_set_eff)
        # ── 安全 MPC: 状态置信度检测 → PID 回退 (滞回; 2026-08-03 现场投切边界) ──
        safe_fallback = False
        if SAFE_Z_THR >= 0:
            score, fi_w = _safe_score(test_raw[gi:gi+W])
            if SAFE_Z_THR > 0:
                if score > SAFE_Z_THR:
                    low_cnt += 1; high_cnt = 0
                else:
                    high_cnt += 1; low_cnt = 0
                if low_cnt >= SAFE_HOLD: safe_active = True
                if high_cnt >= SAFE_RELEASE: safe_active = False
            SAFE_LOG.append((t, score, fi_w, 'SAFE' if safe_active else 'MPC'))
            if safe_active:
                y_m = float(win[0, -1, TARGET_IDX])
                e_s = float(t_set_eff) - y_m
                safe_I = float(np.clip(safe_I + e_s * DT, -300.0, 300.0))
                u_s = float(np.clip(PID_KP * e_s + PID_KI * safe_I, 0.0, 45.0))
                # 平滑接管: 相对上一执行动作限幅 (同 MPC 的 HARD_DELTA), 避免跳变触发模型失真
                u_s = float(np.clip(u_s, float(a_last[1]) - 5.0, float(a_last[1]) + 5.0))
                a_exec = torch.stack([torch.full((n_exec,), float(a_last[0]), device=DEVICE),
                                      torch.full((n_exec,), u_s, device=DEVICE)], 1)
                safe_fallback = True
        if planner == 'grad' and not safe_fallback:
            OVERLAP_REF = a_init  # 旧计划引用 (重叠一致性用; 首块 None)
            a_plan, Js = plan_grad(wm, win, t_set_eff, a_last, a_init, sp_fut)
        elif not safe_fallback:
            OVERLAP_REF = a_init
            a_plan, Js = plan_cem(wm, win, t_set_eff, a_last, sp_fut)
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
        # 执行端卡尔曼滤波 (位置+速度, 跨块连续; 用户建议 2026-08-03) — 滤驻波且跟踪趋势斜坡
        if EXEC_KF > 0 and not safe_fallback:
            q_vel, r = EXEC_KF, 1.0
            F = np.array([[1.0, 1.0], [0.0, 1.0]])
            H = np.array([1.0, 0.0]).reshape(1, 2)
            a_kf = []
            for j in range(n_exec):
                z = a_plan[j].cpu().numpy()
                row = []
                for ch in range(2):
                    st = kf_state[ch]
                    x = F @ np.array([st['p'], st['v']])
                    P = F @ st['P'] @ F.T + np.diag([0.01, q_vel])
                    S = float(H @ P @ H.T + r)
                    K = P @ H.T / S
                    innov = z[ch] - float(H @ x)
                    x = x + K.flatten() * innov
                    P = (np.eye(2) - K @ H) @ P
                    st['p'], st['v'], st['P'] = float(x[0]), float(x[1]), P
                    row.append(st['p'])
                a_kf.append(row)
            a_exec = torch.FloatTensor(a_kf).to(DEVICE)
        # 执行端因果移动平均 (2026-08-03; KF之后: 测残余锯齿是否还需SMA — 理论上滞后>收益)
        if EXEC_SMA > 1 and a_exec is not None and not safe_fallback:
            w = EXEC_SMA
            a_sm = []
            for j in range(n_exec):
                lo = max(0, j - w + 1)
                a_sm.append(a_exec[lo:j + 1].mean(0))
            a_exec = torch.stack(a_sm)
        # 死区: 相对上一执行动作变化 < 阈值 → 保持 (抑制残余锯齿; 2026-08-03)
        if EXEC_DEADZONE > 0 and not safe_fallback:
            a_dz_ref = a_last
            a_dz = []
            for j in range(n_exec):
                if torch.abs(a_exec[j] - a_dz_ref).max() < EXEC_DEADZONE:
                    a_dz.append(a_dz_ref)
                else:
                    a_dz.append(a_exec[j])
                    a_dz_ref = a_exec[j]
            a_exec = torch.stack(a_dz)
        # PID 参考: 真实动作 + WM 闭环预测 (同一扰动世界, 公平协议)
        with torch.no_grad():
            a_pid_full = torch.FloatTensor(test_raw[gi:gi+H_OUT, VALVE_IDX]).unsqueeze(0).to(DEVICE)
            if getattr(wm, 'use_sp', False):
                mu_pid, _ = wm(win, a_pid_full.reshape(1, -1), sp_fut.unsqueeze(0))
            else:
                mu_pid, _ = wm(win, a_pid_full.reshape(1, -1))
        # 执行≠计划时: 用实际执行动作重算 WM 温度 (闭环一致性, 否则评测的是未执行的动作)
        if blended or FIX_MODE.startswith('inert') or safe_fallback:
            if safe_fallback:  # 回退块: 全序列用回退动作 (不混旧规划残留)
                a_full_exec = torch.cat([a_exec, a_exec[-1:].repeat(H_OUT - n_exec, 1)], 0)
            elif H_PLAN < H_OUT:
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
            if sp_step is not None and BENCH_SP_EACH:  # 合成 SP 基准
                t_j = (gi_j - i) * DT
                t_set_traj.append(sp_base + (sp_step[1] if t_j >= sp_step[0] else 0.0))
            else:
                t_set_traj.append(float(test_raw[gi_j + W, SP_IDX]) if BENCH_SP_EACH else t_set.item())
            mpc_actions.append(a_exec[j].cpu().numpy())
            pid_actions.append(pid_a)
        a_last = a_exec[n_exec - 1]
        a_init = a_plan if not safe_fallback else None  # warm-start (整段继承; safe回退无规划)
        # 积分状态更新: 块内每执行步累积偏差 (等效 PI 积分; 2026-08-03)
        if INT_GAIN > 0 and t > 0:
            for j in range(n_exec):
                int_state = float(np.clip(int_state + (float(mpc_temp[-n_exec + j]) - t_set_eff_prev), -10.0, 10.0))
        # offset-free 模型偏差估计: d = y_meas − y_pred = 扰动状态 (EMA; 2026-08-03)
        if OFFSET_GAIN > 0:
            d_est = float(OFFSET_EMA * d_state + (1 - OFFSET_EMA) * d_est)
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
    # 压线指标 (2026-08-03): 平均温度 / 平均偏差 (正=贴线上方) / 阀位距安全带距离 (越大越安全)
    m['mean_temp_mpc'] = float(mpc_t.mean())
    m['mean_temp_pid'] = float(pid_t.mean())
    m['mean_err_mpc'] = float((mpc_t - t_set_traj).mean())
    m['mean_err_pid'] = float((pid_t - t_set_traj).mean())
    m['valve_margin_mpc'] = float(np.minimum(U_HI - mpc_a, mpc_a - U_LO).min(1).mean())
    m['valve_margin_pid'] = float(np.minimum(U_HI - pid_a, pid_a - U_LO).min(1).mean())
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
