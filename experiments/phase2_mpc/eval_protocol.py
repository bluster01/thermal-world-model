#!/usr/bin/env python3
"""
eval_protocol.py — 公平评测协议 (P0-A/B/C 修复版)
==================================================
修复 exp_027 主协议的四个结构性缺陷:

  P0-A  真 PID 闭环          : PID 组原为历史阀位录像回放 (open-loop replay),
                               无法看见扰动 → 与 MPC 的闭环对比不对等。
                               本模块 PID 为真反馈控制器, 与 MPC 同世界同扰动同执行器。

  P0-B  第三方评测世界        : 原协议用被测 WM 自己预测温度 (自评分), 且 M5/M7 各自
                               在各自世界里跑 → RMSE 不可比。本模块用与控制器不相交的
                               模型集成作为"世界", 所有控制器在同一世界推进。

  P0-C  物理侧扰动            : 原协议把扰动作为标量偏置直接加在输出温度上, 而窗口内
                               其余 39 维状态仍为真实历史 → 物理矛盾, 且对 ARX 是纯噪声。
                               本模块扰动注入负荷/煤量/流量等物理通道, 由世界模型传导到温度。

  P0-D  窗口动作回填          : 原协议推进窗口时只覆盖温度列, 阀位列仍为历史真值 →
                               世界模型看到的动作历史不是控制器实际执行的动作。
                               本模块同步回填阀位列。

统一执行器: PID 与 MPC 共用同一执行器模型 (一阶惯性 + 速率限幅 + 行程限幅),
           消除"MPC 有 KF/SMA 平滑而 PID 没有"的不对等。

用法: 作为库导入, 见 exp_S1_fair_comparison.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, os.path.join(_ROOT, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfg  # noqa: E402
from experiments.phase1_dynamics.exp_025_unified_benchmark import (  # noqa: E402
    build_model, test_raw, VALVE_IDX, TARGET_IDX, H_OUT, NUMERIC_COLS)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

W = cfg.WINDOW_SIZE
DT = 10                    # 采样间隔 (s)
SP_IDX = 36                # 二级减温调节阀设定 = 主汽温 SP
T_MIN, T_MAX = 540.0, 575.0
CVAR_K = 2.0627            # α=0.95 正态 CVaR 系数

# ── 物理扰动通道 (P0-C): 负荷 / 煤量 / 主蒸汽流量 ──
# 这些是真实机组的主要外部扰动源, 经由过热器传热动力学传导到主汽温。
_PHYS_NAMES = ['机组负荷', '未校正总煤量', '主蒸汽流量']
PHYS_DIST_IDX = [NUMERIC_COLS.index(n) for n in _PHYS_NAMES if n in NUMERIC_COLS]
# 各通道扰动的相对幅度权重 (相对该列训练集标准差)
PHYS_DIST_W = np.array([1.0, 1.0, 0.6][:len(PHYS_DIST_IDX)], dtype=np.float32)
_PHYS_STD = test_raw[:, PHYS_DIST_IDX].std(0) if PHYS_DIST_IDX else np.zeros(0, np.float32)


# ══════════════════════════════════════════════════════════════════
# 执行器 (PID / MPC 共用 — 消除执行端不对等)
# ══════════════════════════════════════════════════════════════════
class Actuator:
    """阀门执行器: 一阶惯性 + 速率限幅 + 行程限幅

    u_cmd (控制器指令) → u_act (实际阀位)
      u_act[t] = u_act[t-1] + K·(u_cmd − u_act[t-1])   一阶惯性
      |u_act[t] − u_act[t-1]| ≤ rate                    速率限幅
      lo ≤ u_act[t] ≤ hi                                行程限幅
    """

    def __init__(self, u0, inertia=0.5, rate=5.0, lo=0.0, hi=45.0):
        self.u = np.asarray(u0, dtype=np.float64).copy()
        self.K, self.rate, self.lo, self.hi = inertia, rate, lo, hi

    def step(self, u_cmd):
        u_cmd = np.asarray(u_cmd, dtype=np.float64)
        target = self.u + self.K * (u_cmd - self.u)
        delta = np.clip(target - self.u, -self.rate, self.rate)
        self.u = np.clip(self.u + delta, self.lo, self.hi)
        return self.u.copy()


# ══════════════════════════════════════════════════════════════════
# P0-C: 物理侧扰动发生器
# ══════════════════════════════════════════════════════════════════
class Disturbance:
    """AR(1) 有色噪声扰动

    mode='physical' (默认, P0-C 修复): 扰动注入负荷/煤量/流量物理通道,
        由世界模型传导到主汽温 —— 物理自洽, 对线性/非线性模型公平。
    mode='output'   (legacy, 仅用于与旧结果对齐): 标量偏置直接加在输出温度。

    sigma: 扰动强度 (physical 模式下为各通道标准差的倍数; output 模式下为 °C/步)
    rho:   自相关系数 (0=白噪声/高频, 0.9=随机游走/低频)
    """

    def __init__(self, sigma=0.3, rho=0.9, mode='physical', seed=0):
        self.sigma, self.rho, self.mode = sigma, rho, mode
        self.rng = np.random.default_rng(seed)
        self.state = 0.0
        self.enabled = sigma > 0

    def step(self):
        """推进一步, 返回归一化扰动强度 d (无量纲)"""
        if not self.enabled:
            return 0.0
        self.state = self.rho * self.state + self.rng.normal(0.0, 1.0)
        # AR(1) 稳态方差归一化 → 保证不同 rho 下扰动能量可比 (关键: 否则 rho 大幅度自动变大)
        norm = np.sqrt(1.0 - self.rho ** 2) if self.rho < 1.0 else 1.0
        return float(self.state * norm * self.sigma)

    def apply_to_row(self, row, d):
        """physical 模式: 扰动物理通道 (就地修改 row 副本并返回)"""
        if not self.enabled or self.mode != 'physical' or not PHYS_DIST_IDX:
            return row
        row = row.copy()
        row[PHYS_DIST_IDX] = row[PHYS_DIST_IDX] + d * PHYS_DIST_W * _PHYS_STD
        return row

    def apply_to_temp(self, temp, d):
        """output 模式 (legacy): 偏置加在温度上"""
        if not self.enabled or self.mode != 'output':
            return temp
        return temp + d


# ══════════════════════════════════════════════════════════════════
# P0-B: 第三方评测世界 (与控制器模型不相交的集成)
# ══════════════════════════════════════════════════════════════════
def load_wm(model_id, seed=None, root='results'):
    """加载世界模型权重。seed 非空时读 exp_025_{id}_s{seed}/。"""
    tag = f"{model_id}_s{seed}" if seed is not None else model_id
    path = os.path.join(root, f"exp_025_{tag}", 'checkpoints', 'best_model.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少权重: {path}")
    model = build_model(model_id).to(DEVICE).eval()
    ck = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict'])
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class WorldSim:
    """评测世界: 若干世界模型的集成均值, 只用于状态推进, 不参与规划。

    公平性契约 (P0-B): 构造时断言世界成员与控制器模型不相交。
    集成均值降低单模型偏置, 使不同控制器面对同一个"物理世界"。
    """

    def __init__(self, model_ids, controller_ids=(), seed=None, root='results'):
        overlap = set(model_ids) & set(controller_ids)
        assert not overlap, f"评测世界与控制器模型重叠 (自评分): {overlap}"
        self.ids = list(model_ids)
        self.models = [load_wm(m, seed=seed, root=root) for m in self.ids]

    @torch.no_grad()
    def predict(self, win, a_future, sp_fut=None):
        """win [1,W,40], a_future [H_OUT,2] → mu [H_OUT] (集成均值, 物理空间)"""
        a_flat = a_future.reshape(1, -1)
        outs = []
        for m in self.models:
            if getattr(m, 'use_sp', False) and sp_fut is not None:
                mu, _ = m(win, a_flat, sp_fut.unsqueeze(0))
            else:
                mu, _ = m(win, a_flat)
            outs.append(mu[0])
        return torch.stack(outs).mean(0)


class HybridWM(torch.nn.Module):
    """S1b: 解耦均值-不确定性 — 确定性模型给 μ, 概率模型给 σ

    动机: 若确定性模型 (M5) 均值更准而缺 σ, 概率模型 (M7) σ 校准好但均值略逊,
          则 μ 与 σ 分别取各自最优, 可能优于任一单模型。
    接口与 DirectWM 一致: forward(x_hist, a_flat[, sp_fut]) → (mu, lv)
    注意: μ 路径保留梯度 (MPC 需反传), σ 路径 detach (仅作风险度量, 不引导梯度穿过 M7)。
    """

    def __init__(self, mu_model, sigma_model):
        super().__init__()
        self.mu_model, self.sigma_model = mu_model, sigma_model
        self.probabilistic = True
        self.use_sp = getattr(mu_model, 'use_sp', False)

    def forward(self, x_hist, a_flat, sp_fut=None):
        if getattr(self.mu_model, 'use_sp', False) and sp_fut is not None:
            mu, _ = self.mu_model(x_hist, a_flat, sp_fut)
        else:
            mu, _ = self.mu_model(x_hist, a_flat)
        with torch.no_grad():
            if getattr(self.sigma_model, 'use_sp', False) and sp_fut is not None:
                _, lv = self.sigma_model(x_hist, a_flat, sp_fut)
            else:
                _, lv = self.sigma_model(x_hist, a_flat)
        return mu, (lv.detach() if lv is not None else None)


# ══════════════════════════════════════════════════════════════════
# 成本函数配置 (S1: 四种成本口径)
# ══════════════════════════════════════════════════════════════════
class CostConfig:
    """MPC 成本函数配置

    variant:
      'a' rmse_only : 仅跟踪误差 + 硬边界软约束 (旧"RMSE-only"口径)
      'b' overtemp  : + 非对称超温惩罚 (作用于均值 mu)
      'c' cvar      : + CVaR 风险惩罚 (作用于 mu + k·σ, 需概率模型)
      'd' total     : 综合工程代价 = 跟踪 + 超温积分 + 动作总变差
    """

    PRESETS = {
        'a': dict(w_over=1.0, w_cvar=0.0, w_ot_int=0.0, w_tv=0.0),
        'b': dict(w_over=5.0, w_cvar=0.0, w_ot_int=0.0, w_tv=0.0),
        'c': dict(w_over=1.0, w_cvar=2.0, w_ot_int=0.0, w_tv=0.0),
        'd': dict(w_over=1.0, w_cvar=0.0, w_ot_int=0.1, w_tv=0.01),
    }

    def __init__(self, variant='a', lam_smooth=0.0, lam_dev=0.0,
                 alpha_term=0.5, sigma_add=0.0):
        assert variant in self.PRESETS, f"未知成本口径: {variant}"
        self.variant = variant
        for k, v in self.PRESETS[variant].items():
            setattr(self, k, v)
        self.lam_smooth = lam_smooth      # Σ(Δa)²  平滑
        self.lam_dev = lam_dev            # Σ(a−a_last)²  偏离
        self.alpha_term = alpha_term      # 终端权重
        self.sigma_add = sigma_add        # 扰动方差叠加 (CVaR 用)

    def needs_sigma(self):
        return self.w_cvar > 0

    def __call__(self, mu, sigma, target, a_seq, a_last):
        """返回标量代价 (可反传)"""
        h = mu.shape[0]
        w = torch.linspace(1.0, 0.8, h, device=mu.device)
        e = mu - target
        J = (w * e.pow(2)).sum() / h
        J = J + self.alpha_term * e[-1].pow(2)
        # 硬边界软约束 (始终存在 — 这是安全底线, 不是可选项)
        J = J + 2.0 * (F.relu(mu - T_MAX).pow(2).sum() + F.relu(T_MIN - mu).pow(2).sum())
        # (b) 非对称超温惩罚: 均值超温区加重
        if self.w_over != 1.0:
            J = J + (self.w_over - 1.0) * F.relu(e).pow(2).sum() / h
        # (c) CVaR 风险: 尾部超温概率进目标 (概率模型专属)
        if self.w_cvar > 0 and sigma is not None:
            s = sigma
            if self.sigma_add > 0:
                s = torch.sqrt(s.pow(2) + self.sigma_add ** 2)
            J = J + self.w_cvar * F.relu(mu + CVAR_K * s - T_MAX).pow(2).sum()
        # (d) 综合工程代价: 超温积分 + 动作总变差
        if self.w_ot_int > 0:
            J = J + self.w_ot_int * F.relu(mu - T_MAX).sum()
        if self.w_tv > 0 and a_seq.shape[0] > 1:
            J = J + self.w_tv * (a_seq[1:] - a_seq[:-1]).abs().sum()
        # 常规正则
        if self.lam_smooth > 0 and a_seq.shape[0] > 1:
            J = J + self.lam_smooth * (a_seq[1:] - a_seq[:-1]).pow(2).sum()
        if self.lam_dev > 0:
            J = J + self.lam_dev * (a_seq - a_last).pow(2).sum()
        return J


# ══════════════════════════════════════════════════════════════════
# 控制器
# ══════════════════════════════════════════════════════════════════
class PIDController:
    """P0-A: 真闭环 PI 控制器 (抗积分饱和)

    与 MPC 共用同一执行器, 同一世界, 同一扰动序列。
    只输出阀位指令, 执行器动力学在外部统一施加。
    """
    name = 'pid'

    def __init__(self, kp=40.0, ki=8.0, kd=0.0, i_clip=300.0, u_lo=0.0, u_hi=45.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_clip, self.u_lo, self.u_hi = i_clip, u_lo, u_hi
        self.I, self.e_prev = 0.0, 0.0

    def reset(self):
        self.I, self.e_prev = 0.0, 0.0

    def act(self, win, sp_now, sp_fut, a_last, **_):
        y = float(win[0, -1, TARGET_IDX])
        e = sp_now - y
        u_unsat = self.kp * e + self.ki * self.I + self.kd * (e - self.e_prev) / DT
        # 抗饱和: 仅在未饱和时积分 (conditional integration)
        if self.u_lo < u_unsat < self.u_hi:
            self.I = float(np.clip(self.I + e * DT, -self.i_clip, self.i_clip))
        self.e_prev = e
        u = float(np.clip(self.kp * e + self.ki * self.I + self.kd * (e - self.e_prev) / DT,
                          self.u_lo, self.u_hi))
        # 返回 1 步指令 (PID 无预测视野)
        return np.array([a_last[0], u], dtype=np.float64), 1


class MPCController:
    """梯度规划 MPC (backprop through WM)

    控制器模型 (self.wm) 与评测世界严格分离 (P0-B)。
    """
    name = 'mpc'

    def __init__(self, wm, cost: CostConfig, h_plan=10, m_step=6,
                 eta=0.05, e_steps=30, rate=5.0, u_lo=0.0, u_hi=45.0,
                 control_both=False):
        self.wm, self.cost = wm, cost
        self.h_plan, self.m_step = h_plan, m_step
        self.eta, self.e_steps = eta, e_steps
        self.rate, self.u_lo, self.u_hi = rate, u_lo, u_hi
        self.control_both = control_both   # False: 仅二级阀 (与 PID 同执行通道, 公平)
        self.a_init = None

    def reset(self):
        self.a_init = None

    def _rollout(self, win, a_seq, sp_fut, a_hist1):
        """a_seq [h_plan] (二级阀) 或 [h_plan,2] → wm 预测 (mu, sigma)"""
        if self.control_both:
            a2 = a_seq
        else:
            a2 = torch.stack([a_hist1.expand(a_seq.shape[0]), a_seq], 1)
        if self.h_plan < H_OUT:
            a2 = torch.cat([a2, a2[-1:].repeat(H_OUT - self.h_plan, 1)], 0)
        else:
            a2 = a2[:H_OUT]
        if getattr(self.wm, 'use_sp', False) and sp_fut is not None:
            mu, lv = self.wm(win, a2.reshape(1, -1), sp_fut.unsqueeze(0))
        else:
            mu, lv = self.wm(win, a2.reshape(1, -1))
        sigma = None
        if lv is not None and self.cost.needs_sigma():
            sigma = torch.exp(lv[0, :self.h_plan] * 0.5)
        return mu[0, :self.h_plan], sigma, a2

    def act(self, win, sp_now, sp_fut, a_last, **_):
        a_last_t = torch.tensor(float(a_last[1]), device=DEVICE)
        a_hist1 = torch.tensor(float(a_last[0]), device=DEVICE)
        if self.a_init is not None and len(self.a_init) == self.h_plan:
            a0 = self.a_init.clone()
        else:
            a0 = a_last_t.repeat(self.h_plan).clone()
        a = a0.detach().requires_grad_(True)
        opt = torch.optim.Adam([a], lr=self.eta)
        tgt = sp_fut[:self.h_plan] if sp_fut is not None else torch.tensor(sp_now, device=DEVICE)
        for _ in range(self.e_steps):
            opt.zero_grad()
            mu, sigma, _ = self._rollout(win, a, sp_fut, a_hist1)
            J = self.cost(mu, sigma, tgt, a.unsqueeze(1), a_last_t)
            J.backward()
            opt.step()
            with torch.no_grad():
                # 速率约束: 首步相对 a_last, 后续逐步 (与执行器一致)
                a[0] = torch.clamp(a[0], a_last_t - self.rate, a_last_t + self.rate)
                a[1:] = torch.clamp(a[1:] - a[:-1], -self.rate, self.rate) + a[:-1]
                a.clamp_(self.u_lo, self.u_hi)
        plan = a.detach()
        self.a_init = torch.cat([plan[self.m_step:], plan[-1:].repeat(
            min(self.m_step, self.h_plan))])[:self.h_plan]
        cmds = np.stack([np.full(self.m_step, float(a_last[0])),
                         plan[:self.m_step].cpu().numpy()], 1)
        return cmds, self.m_step


# ══════════════════════════════════════════════════════════════════
# 统一仿真主循环
# ══════════════════════════════════════════════════════════════════
def run_episode(controller, world, start, n_steps=120, dist: Disturbance = None,
                actuator_kw=None):
    """统一闭环仿真: 控制器 → 执行器 → 世界 → 窗口推进

    公平性保证:
      - PID 与 MPC 走同一函数、同一世界、同一执行器、同一扰动序列
      - 窗口推进时同步回填 温度(世界输出) + 阀位(执行器实际值) + 物理扰动通道 (P0-D/C)

    返回 dict: temp / sp / act / dist
    """
    controller.reset()
    i = int(start)
    N = len(test_raw)
    win = torch.FloatTensor(test_raw[i:i + W]).unsqueeze(0).to(DEVICE)
    act = Actuator(test_raw[i + W - 1, VALVE_IDX], **(actuator_kw or {}))
    a_last = act.u.copy()
    temps, sps, acts, ds = [], [], [], []

    t = 0
    while t < n_steps:
        gi = i + t
        if gi + W + H_OUT + 1 >= N:
            break
        sp_now = float(test_raw[gi + W, SP_IDX])
        sp_fut = torch.FloatTensor(test_raw[gi + W:gi + W + H_OUT, SP_IDX]).to(DEVICE)
        cmds, n_exec = controller.act(win, sp_now, sp_fut, a_last)
        cmds = np.atleast_2d(cmds)
        n_exec = min(n_exec, n_steps - t, len(cmds))

        for j in range(n_exec):
            gi_j = gi + j
            if gi_j + W + H_OUT + 1 >= N:
                break
            # 1) 执行器: 指令 → 实际阀位 (PID/MPC 同一模型)
            a_real = act.step(cmds[j])
            # 2) 扰动推进 (物理通道)
            d = dist.step() if dist is not None else 0.0
            # 3) 世界推进: 用实际阀位 (未来段保持) 预测下一步温度
            a_fut = torch.FloatTensor(np.tile(a_real, (H_OUT, 1))).to(DEVICE)
            sp_f = torch.FloatTensor(test_raw[gi_j + W:gi_j + W + H_OUT, SP_IDX]).to(DEVICE)
            y = float(world.predict(win, a_fut, sp_f)[0])
            if dist is not None:
                y = dist.apply_to_temp(y, d)
            # 4) 窗口推进: 回填 温度 + 阀位 + 物理扰动通道 (P0-C/D)
            row = test_raw[gi_j + W].copy()
            if dist is not None:
                row = dist.apply_to_row(row, d)
            row[TARGET_IDX] = y
            row[VALVE_IDX] = a_real
            nr = torch.FloatTensor(row).view(1, 1, -1).to(DEVICE)
            win = torch.cat([win[:, 1:, :], nr], 1)

            temps.append(y)
            sps.append(float(test_raw[gi_j + W, SP_IDX]))
            acts.append(a_real.copy())
            ds.append(d)
            a_last = a_real
        if n_exec <= 0:
            break
        t += n_exec

    return dict(temp=np.array(temps), sp=np.array(sps),
                act=np.array(acts), dist=np.array(ds))


def metrics(ep):
    """控制品质指标。超温统一以 (s) 为单位 (1 步 = DT = 10s)。"""
    y, sp, a = ep['temp'], ep['sp'], ep['act']
    e = y - sp
    m = {
        'rmse': float(np.sqrt(np.mean(e ** 2))),
        'iae': float(np.trapz(np.abs(e))) * DT,
        'itae': float(np.trapz(np.arange(len(e)) * np.abs(e))) * DT * DT,
        'tv': float(np.abs(np.diff(a, axis=0)).mean()) if len(a) > 1 else 0.0,
        'mean_err': float(e.mean()),
        # 超温: 时间 (s) 与积分 (°C·s) — 口径统一 (修 exp_027 的步数/秒混用)
        'overtemp_s': float((y > T_MAX).sum() * DT),
        'overtemp_int': float(np.maximum(y - T_MAX, 0.0).sum() * DT),
        'undertemp_s': float((y < T_MIN).sum() * DT),
        'max_temp': float(y.max()),
        'n_steps': int(len(y)),
    }
    # 综合工程代价 (与 CostConfig 'd' 同权重, 用于跨方法排序)
    m['j_total'] = m['rmse'] + 0.1 * m['overtemp_int'] / DT + 0.01 * m['tv']
    return m
