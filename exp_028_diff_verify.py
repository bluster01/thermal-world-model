#!/usr/bin/env python3
"""
exp_028_diff_verify.py — Phase 2c: DWM-MPC 可微性验证
========================================================
1. 梯度检查: 解析 ∂J/∂a vs 有限差分 (余弦/相对误差) — 确认 backprop 正确
2. 规划收敛: J 随内层 Adam 迭代 E 的下降曲线
3. 多起点: 5 个随机起点 → 最优 J 分布 (局部最优严重度)
4. 梯度方向: ∂T/∂a 是否物理 (开阀降温)
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from exp_027_dwm_mpc import (
    load_wm, build_objective, H_PLAN, ETA, E_STEPS, H_OUT, W, DEVICE,
    test_raw, VALVE_IDX, TARGET_IDX)

np.random.seed(42); torch.manual_seed(42)

# ===== 1. 梯度检查 =====
print("=" * 60)
print("1. 梯度检查: 解析 vs 有限差分 (∂J/∂a)")
wm = load_wm()
i = np.random.randint(0, len(test_raw) - W - H_OUT - 5)
x_hist = torch.FloatTensor(test_raw[i:i+W]).unsqueeze(0).to(DEVICE)
t_set = torch.tensor(np.mean(test_raw[i:i+W, TARGET_IDX]), dtype=torch.float32, device=DEVICE)
a_last = torch.FloatTensor(test_raw[i+W, VALVE_IDX]).to(DEVICE)

a0 = a_last.unsqueeze(0).repeat(H_PLAN, 1).clone().detach().requires_grad_(True)
J = build_objective(wm, x_hist, a0, t_set, a_last)
J.backward()
grad_analytic = a0.grad.clone().cpu().numpy()

eps = 0.5  # 相对动作尺度 (float32 安全)
grad_fd = np.zeros_like(grad_analytic)
for h in range(H_PLAN):
    for dim in range(2):
        ap = a0.clone().detach(); am = a0.clone().detach()
        ap[h, dim] += eps; am[h, dim] -= eps
        Jp = build_objective(wm, x_hist, ap, t_set, a_last).item()
        Jm = build_objective(wm, x_hist, am, t_set, a_last).item()
        grad_fd[h, dim] = (Jp - Jm) / (2 * eps)

cos = np.dot(grad_analytic.ravel(), grad_fd.ravel()) / (
    np.linalg.norm(grad_analytic) * np.linalg.norm(grad_fd) + 1e-12)
rel = np.linalg.norm(grad_analytic - grad_fd) / (np.linalg.norm(grad_fd) + 1e-12)
print(f"  余弦相似度: {cos:.6f} | 相对误差: {rel:.6f}")
print(f"  {'✅ 可微性成立' if cos > 0.999 else '❌ 梯度不一致!'}")

# ===== 2. 规划收敛 =====
print("=" * 60)
print("2. 规划收敛: J vs Adam 迭代步数")
a = a_last.unsqueeze(0).repeat(H_PLAN, 1).clone().detach().requires_grad_(True)
opt = torch.optim.Adam([a], lr=ETA)
Js = []
for e in range(E_STEPS):
    opt.zero_grad()
    J = build_objective(wm, x_hist, a, t_set, a_last)
    J.backward()
    opt.step()
    with torch.no_grad():
        a.clamp_(0, 100)
    Js.append(J.item())
print(f"  J: {Js[0]:.4f} → {Js[-1]:.4f} ({len(Js)} 步, 降幅 {(Js[0]-Js[-1])/Js[0]*100:.1f}%)")
print(f"  前5步: {[round(x,3) for x in Js[:5]]}")
print(f"  后5步: {[round(x,3) for x in Js[-5:]]}")

# ===== 3. 多起点 =====
print("=" * 60)
print("3. 多起点: a_last 附近 ±10 扰动 × 5 → 最优 J 分布")
print("   (完全随机 0-100 是分布外外推, 无意义; MPC 实际是 warm-start)")
starts = []
base = a_last.unsqueeze(0).repeat(H_PLAN, 1).clone()
for s in range(5):
    a_init = (base + (torch.rand(H_PLAN, 2, device=DEVICE) - 0.5) * 20).clamp(0, 100)
    a_init.requires_grad_(True)
    opt2 = torch.optim.Adam([a_init], lr=ETA)
    for _ in range(E_STEPS):
        opt2.zero_grad()
        J = build_objective(wm, x_hist, a_init, t_set, a_last)
        J.backward()
        opt2.step()
        with torch.no_grad(): a_init.clamp_(0, 100)
    starts.append(J.item())
print(f"  各起点最优 J: {[round(x,4) for x in starts]}")
print(f"  mean={np.mean(starts):.4f} std={np.std(starts):.4f} (std 小=局部最优不严重)")

# ===== 4. 梯度物理方向 =====
print("=" * 60)
print("4. 梯度方向: ∂T(12)/∂a₂(0) (开阀应降温)")
a_seq = a_last.unsqueeze(0).repeat(H_OUT, 1).clone().detach().requires_grad_(True)
mu, _ = wm(x_hist, a_seq.reshape(1, -1))
mu[0, 12].backward()
g = a_seq.grad[0, 1].item()  # 首步二级阀
print(f"  ∂T(12)/∂a₂(0) = {g:+.4f} °C/阀位")
print(f"  {'✅ 物理方向 (开阀降温)' if g < 0 else '❌ 反物理!'}")

out = {'grad_check': {'cos': float(cos), 'rel_err': float(rel)},
       'convergence': {'J0': Js[0], 'Jf': Js[-1], 'trace': Js},
       'multi_start': {'Js': starts, 'mean': float(np.mean(starts)), 'std': float(np.std(starts))},
       'phys_dir': float(g)}
json.dump(out, open("results/exp_028_diff_verify.json", 'w'), indent=2)
print("\nSaved: results/exp_028_diff_verify.json")
