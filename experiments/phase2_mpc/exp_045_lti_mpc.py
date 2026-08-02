#!/usr/bin/env python3
"""
exp_045_lti_mpc.py — 线性 MPC 基线 (ARX 模型) — Phase 2.5 任务2
=================================================================
目的: 证明深度 WM 比经典线性 MPC 有增量
1. ARX 辨识: y(t)=Σa_i y(t-i)+Σb_j u(t-d-j), u=[一级阀,二级阀], 时延 d=6 (60s)
2. 线性 MPC (同框架: H=10, α=0.5, M_STEP=6, 梯度优化)
3. 公平评测: 所有控制器在 WM 闭环世界 (规划模型不同, 评测世界统一)
   对比: DWM-MPC vs ARX-MPC vs PID-WM
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
from exp_027_dwm_mpc import load_wm, W, H_OUT, DEVICE, test_raw, VALVE_IDX, SP_IDX, TARGET_IDX
import exp_027_dwm_mpc as M
sys.argv = _argv

from experiments.phase1_dynamics.exp_025_unified_benchmark import train_raw

N_TRACKS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NA = 3      # 温度自回归阶数 (30s)
NB = 2      # 阀位回归阶数
D = 6       # 时延 (60s, 动作→温度物理时标)

# ===== 1. ARX 辨识 (train 段) =====
u_tr = train_raw[:, VALVE_IDX]      # [N, 2]
y_tr = train_raw[:, TARGET_IDX]     # [N]
N_USE = 200000
rows, targets = [], []
for t in range(D + NB, N_USE):
    feat = list(y_tr[t-NA:t])                       # 温度自回归
    feat += list(u_tr[t-D:t-D+NB, 0])               # 一级阀 (时延后)
    feat += list(u_tr[t-D:t-D+NB, 1])               # 二级阀 (时延后)
    rows.append(feat); targets.append(y_tr[t])
X = np.array(rows); Y = np.array(targets)
print(f"ARX 特征: {X.shape} (na={NA}, nb={NB}, d={D})")
# 岭回归
lam = 1e-3
XTX = X.T @ X + lam * np.eye(X.shape[1])
beta = np.linalg.solve(XTX, X.T @ Y)
y_hat = X @ beta
print(f"ARX 拟合: R²={1-np.sum((y_hat-Y)**2)/np.sum((Y-Y.mean())**2):.4f}, 单步MAE={np.abs(y_hat-Y).mean():.3f}°C")

# 开环多步验证 (val 段)
def arx_predict_openloop(y0, u_seq, steps):
    """递推预测: y(t) = beta·feat(t)"""
    y_hist = list(y0)
    preds = []
    for k in range(steps):
        t = len(y_hist)
        feat = list(y_hist[t-NA:t])
        feat += list(u_seq[t-D:t-D+NB, 0])
        feat += list(u_seq[t-D:t-D+NB, 1])
        y_next = beta @ feat
        y_hist.append(y_next); preds.append(y_next)
    return np.array(preds)

# val 段开环验证 (用真实动作, 递推预测)
from experiments.phase1_dynamics.exp_025_unified_benchmark import val_raw
u_val = val_raw[:, VALVE_IDX]; y_val = val_raw[:, TARGET_IDX]
S0 = 1000
y_hat_ol = arx_predict_openloop(y_val[S0-D:S0], u_val[S0-D+1:S0+200], 120)
y_true_ol = y_val[S0+1:S0+121]
print(f"开环 120 步预测: MAE={np.abs(y_hat_ol-y_true_ol).mean():.3f}°C (真实动作)")
np.savez("results/exp_045_lti/arx_model.npz", beta=beta, na=NA, nb=NB, d=D)

# ===== 2. ARX-MPC 规划 (梯度, 与 WM 同框架) =====
beta_t = torch.FloatTensor(beta).to(DEVICE)

def arx_predict_tensor(y_hist, u_hist, a_seq):
    """a_seq [H,2] 可微 → 预测 [H] (ARX 递推, torch)"""
    y = list(y_hist.detach().cpu().numpy())
    preds = []
    for k in range(a_seq.shape[0]):
        t = len(y)
        # 输入阀位: 历史 (k<D 用历史, 之后用动作序列)
        if k < D:
            i0 = min(t - D, len(u_hist) - NB)   # 防越界 (窗口末尾用最后可用)
            u1 = u_hist[i0:i0+NB, 0]; u2 = u_hist[i0:i0+NB, 1]
            feat = torch.cat([torch.FloatTensor(y[t-NA:t]), torch.FloatTensor(u1), torch.FloatTensor(u2)])
        else:
            # 动作序列生效: 取 a_seq[k-D : k-D+NB] (不足时用边界)
            j = k - D
            idx = np.clip([j, j+1], 0, a_seq.shape[0]-1)
            feat = torch.cat([torch.FloatTensor(y[t-NA:t]).to(DEVICE),
                              a_seq[idx[0], 0].reshape(1), a_seq[idx[1], 0].reshape(1),
                              a_seq[idx[0], 1].reshape(1), a_seq[idx[1], 1].reshape(1)])
        y_next = (beta_t @ feat.to(DEVICE)).clone()
        y.append(y_next.item())
        preds.append(y_next)
    return torch.stack(preds)

def plan_arx(win, t_set, a_last):
    """ARX-MPC 梯度规划 (与 WM MPC 相同 J 结构)"""
    y_hist = win[0, :, TARGET_IDX].detach().cpu()
    u_hist = win[0, :, VALVE_IDX].detach().cpu()
    a = a_last.unsqueeze(0).repeat(M.H_PLAN, 1).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=0.05)
    w = torch.linspace(1.0, 0.8, M.H_PLAN, device=DEVICE)
    for _ in range(30):
        opt.zero_grad()
        pred = arx_predict_tensor(y_hist, u_hist, a)
        err = (pred - t_set) ** 2
        J = (w * err).sum() / M.H_PLAN + 0.5 * err[-1]
        if M.H_PLAN > 1:
            J = J + 0.1 * ((a[1:] - a[:-1]) ** 2).sum()
        J.backward()
        opt.step()
        with torch.no_grad(): a.clamp_(0, 100)
    return a.detach()

# ===== 3. 公平评测: 所有控制器在 WM 闭环世界 =====
wm = load_wm()
np.random.seed(42)
N = len(test_raw)
starts = np.random.choice(range(N - W - H_OUT - 120), N_TRACKS, replace=False)

def sim_controller(track_idx, controller, n_steps=120, m_step=M.M_STEP):
    win = torch.FloatTensor(test_raw[track_idx:track_idx+W]).unsqueeze(0).to(DEVICE)
    a_last = torch.FloatTensor(test_raw[track_idx+W, VALVE_IDX]).to(DEVICE)
    temps = []
    for k in range(0, n_steps, m_step):
        gi = track_idx + W + k
        t_set = torch.tensor(float(test_raw[gi, SP_IDX]), device=DEVICE)
        if controller == 'dwm':
            a_plan, _ = M.plan_grad(wm, win, t_set, a_last, None, None)
        else:  # arx
            a_plan = plan_arx(win, t_set, a_last)
        with torch.no_grad():
            if M.H_PLAN < H_OUT:
                a_full = torch.cat([a_plan, a_plan[-1:].repeat(H_OUT - M.H_PLAN, 1)], 0)
            else:
                a_full = a_plan[:H_OUT]
            mu, _ = wm(win, a_full.reshape(1, -1))
        for j in range(min(m_step, len(mu[0]))):
            gij = gi + j
            if gij >= track_idx + W + n_steps: break
            y_j = mu[0, j].item()
            next_row = torch.FloatTensor(test_raw[gij]).unsqueeze(0).unsqueeze(0).to(DEVICE)
            next_row[0, 0, TARGET_IDX] = y_j
            win = torch.cat([win[:, 1:, :], next_row], 1)
            temps.append(y_j)
        a_last = a_plan[min(m_step, len(a_plan)) - 1]
    return np.array(temps)

def sim_pid_wm(track_idx, n_steps=120):
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

rows, t0 = [], time.time()
for k, s in enumerate(starts):
    tset = test_raw[s+W:s+W+120, SP_IDX]
    mpc_t = sim_controller(s, 'dwm')
    lti_t = sim_controller(s, 'arx')
    pid_t = sim_pid_wm(s)
    def rmse(a): return float(np.sqrt(np.mean((a - tset)**2)))
    rows.append({'rmse_dwm': rmse(mpc_t), 'rmse_arx': rmse(lti_t), 'rmse_pid': rmse(pid_t),
                 'std_dwm': float(np.std(mpc_t)), 'std_arx': float(np.std(lti_t)), 'std_pid': float(np.std(pid_t))})
    if (k+1) % 5 == 0: print(f"  [{k+1}/{N_TRACKS}]")

agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
print("\n===== 控制器对比 (WM 闭环世界, 公平协议) =====")
print(f"  RMSE: DWM-MPC {agg['rmse_dwm']:.3f} | ARX-MPC {agg['rmse_arx']:.3f} | PID-WM {agg['rmse_pid']:.3f}")
print(f"  std:  DWM-MPC {agg['std_dwm']:.3f} | ARX-MPC {agg['std_arx']:.3f} | PID-WM {agg['std_pid']:.3f}")
print(f"  深度WM增量 (vs ARX): RMSE {(1-agg['rmse_dwm']/agg['rmse_arx'])*100:+.1f}% | std {(1-agg['std_dwm']/agg['std_arx'])*100:+.1f}%")

json.dump({'agg': agg, 'per_track': rows, 'arx': {'na': NA, 'nb': NB, 'd': D,
          'r2_fit': float(1-np.sum((y_hat-Y)**2)/np.sum((Y-Y.mean())**2)),
          'mae_openloop': float(np.abs(y_hat_ol-y_true_ol).mean())}},
          open("results/exp_045_lti/compare.json", 'w'), indent=2, default=float)
print(f"\nSaved: results/exp_045_lti/compare.json (耗时 {(time.time()-t0)/60:.1f}min)")
