#!/usr/bin/env python3
"""
n4sid 线性状态空间 baseline (审稿人 R2-M1 经典系统辨识对照)
=============================================================
延迟嵌入 + SVD 压缩的子空间辨识 (N4SID 核心思想):

1. 状态 = 过去 p 步的 I/O 延迟嵌入: z_t = [y_{t-p:t}, u_{t-p:t}] 展平
2. SVD 压缩到 n 维 (去噪, 子空间法标志性步骤)
3. 线性最小二乘拟合一步动力学: z_{t+1} ≈ A z_t + B u_t; 观测 y_t ≈ C z_t
4. 与世界模型同协议: 给定未来动作序列, 自回归展开 H=18 步, test 集, seed 42

这等价于线性 SSM (ARX→状态空间), 是审稿人认可的系统辨识 baseline。
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data

# ===== 数据 (与世界模型同切分) =====
state_data, delta_actions, valve_abs = load_raw_data()
raw_data = np.concatenate([state_data, valve_abs], axis=1)
n_total = len(raw_data)
n_train = int(n_total * 0.70); n_val_end = int(n_total * 0.85)
train_data = raw_data[:n_train]
test_data = raw_data[n_val_end:]

ny, nu = cfg.N_STATE, cfg.N_ACTION
W, H = cfg.WINDOW_SIZE, 18
print(f"数据: train {len(train_data)} / test {len(test_data)} | y={ny} u={nu}")

# 标准化 (y 量纲差异大)
y_mean, y_std = train_data[:, :ny].mean(0), train_data[:, :ny].std(0)
u_mean, u_std = train_data[:, ny:].mean(0), train_data[:, ny:].std(0)
y_tr = (train_data[:, :ny] - y_mean) / y_std
u_tr = (train_data[:, ny:] - u_mean) / u_std


def fit_linear_dynamics(y, u, p=10, n_comp=25):
    """
    延迟嵌入 + SVD 压缩 + 线性动力学拟合
    返回: (proj, A, B, C) — proj: 压缩矩阵; A: z_{t+1}=A z_t+B u_t; C: y_t≈C z_t
    """
    T = len(y)
    # 延迟嵌入矩阵: 每行 = [y_{t-p+1:t+1}, u_{t-p+1:t+1}] 展平
    dim = (ny + nu) * p
    Z = np.zeros((T - p, dim))
    for t in range(p, T):
        Z[t - p] = np.concatenate([
            y[t-p+1:t+1].flatten(),
            u[t-p+1:t+1].flatten(),
        ])
    # 目标: 下一时刻的 y
    Ynext = y[p:T]  # [T-p, ny]

    # SVD 压缩 (子空间法: 去噪 + 降维)
    Uz, Sz, Vtz = np.linalg.svd(Z, full_matrices=False)
    n_comp = min(n_comp, len(Sz))
    proj = Vtz[:n_comp].T  # [dim, n_comp] — 压缩投影
    Zc = Z @ proj          # [T-p, n_comp] 压缩状态

    # 一步转移: z_{t+1} ≈ A z_t + B u_{t+1}... 用 z_t 和当前 u 预测下一状态
    # 简化: 直接线性映射 Zc_t → Zc_{t+1}
    A = np.linalg.lstsq(Zc[:-1], Zc[1:], rcond=None)[0]     # [n, n]
    # 观测: y_t ≈ C z_t
    C = np.linalg.lstsq(Zc, Ynext, rcond=None)[0].T          # [ny, n]
    # B: 输入对状态的直接效应 (残差部分)
    resid = Zc[1:] - Zc[:-1] @ A
    B = np.linalg.lstsq(u[p+1:T], resid, rcond=None)[0].T   # [n, nu]
    return proj, A, B, C


def rollout_linear(proj, A, B, C, y_hist, u_hist, fa, target_idx):
    """
    从历史窗口末尾开始, 驱动未来动作展开 H 步
    y_hist: [W, ny] 原始尺度; u_hist: [W, nu]; fa: [H, nu] 未来动作
    """
    # 历史窗口末尾的延迟嵌入 (用窗口最后 p 步)
    p = (proj.shape[0] // (ny + nu))
    seg_y = (y_hist[-p:] - y_mean) / y_std
    seg_u = (u_hist[-p:] - u_mean) / u_std
    z = np.concatenate([seg_y.flatten(), seg_u.flatten()])
    z = z @ proj  # 压缩状态

    preds = []
    fa_n = (fa - u_mean) / u_std
    for t in range(H):
        yp = C @ z
        yp_raw = yp * y_std + y_mean
        preds.append(yp_raw[target_idx])
        z = A @ z + B @ fa_n[t]
        # 更新延迟嵌入: 用预测值滚动 (简化: 状态 z 已含全部记忆)
    return np.array(preds)


print("拟合线性动力学 (延迟嵌入 p=10, SVD 压缩 n=25)...")
p, n_comp = 10, 25
proj, A, B, C = fit_linear_dynamics(y_tr, u_tr, p=p, n_comp=n_comp)
print(f"  压缩状态维度: {n_comp}")

# ===== 评测 (与世界模型同协议: test, seed 42, H=18) =====
np.random.seed(42)
Nt = len(test_data)
idxs = np.random.choice(range(Nt - W - H), 500, replace=False)
err = np.zeros((len(idxs), H))
for j, i in enumerate(idxs):
    y_hist = test_data[i:i+W, :ny]
    u_hist = test_data[i:i+W, ny:]
    fa = test_data[i+W:i+W+H, ny:]
    tt = test_data[i+W:i+W+H, cfg.TARGET_IDX]
    pred = rollout_linear(proj, A, B, C, y_hist, u_hist, fa, cfg.TARGET_IDX)
    err[j] = np.abs(pred - tt)

m = err.mean(0)
print(f"\nN4SID 线性SSM rollout (test, H=18):")
print(f"  step0={m[0]:.4f} step17={m[-1]:.4f} ×{m[-1]/m[0]:.1f}")
print(f"  曲线: " + " ".join(f"{x:.3f}" for x in m))

result = {'model': 'n4sid_linear_ssm', 'order': n_comp, 'embed_p': p,
          'rollout_mae': m.tolist()}
with open("results/exp_020_n4sid.json", 'w') as f:
    json.dump(result, f, indent=2)
print("Saved: results/exp_020_n4sid.json")
