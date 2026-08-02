"""
exp_026_grad_check.py — 可微分性验证 (Phase 2 地基)
====================================================
验证 Direct WM (exp_025 M0) 对动作序列的可微性:
1. 解析梯度 vs 有限差分 (∂J/∂a 一致性)
2. 梯度方向物理性: ∂T/∂a₂ 长时标应 <0 (开阀降温)
3. 规划收敛: Adam 梯度上升 E 步, J 应下降

J = Σ w_t (ŷ_t − T_set)² + λ Σ Δa²    (MPC 目标函数)
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from experiments.phase1_dynamics.exp_025_unified_benchmark import (
    build_model, train_raw, test_raw, VALVE_IDX, TARGET_IDX, H_OUT)

DEVICE = torch.device('cpu')  # 梯度检查用 CPU, 不干扰 GPU 训练
W = cfg.WINDOW_SIZE
H = H_OUT

# ============ 1. 加载模型 ============
model = build_model('M0').to(DEVICE).eval()
ck = torch.load("results/exp_025_M0/checkpoints/best_model.pth",
                map_location=DEVICE, weights_only=True)
model.load_state_dict(ck['model_state_dict'])
print(f"模型加载 OK | {sum(p.numel() for p in model.parameters()):,} 参数")

# ============ 2. 取样本窗口 ============
np.random.seed(0)
i = np.random.randint(0, len(test_raw) - W - H)
x_hist = torch.FloatTensor(test_raw[i:i+W]).unsqueeze(0).to(DEVICE)          # [1, W, 40]
a_init = torch.FloatTensor(test_raw[i+W:i+W+H, VALVE_IDX]).clone()            # [H, 2]
t_target = float(test_raw[i+W:i+W+H, TARGET_IDX].mean())                     # 设定值=窗口均值
print(f"样本 idx={i} | T_set={t_target:.2f}°C | a_init 范围 {a_init.min():.1f}-{a_init.max():.1f}")


def objective(a_seq):
    """MPC 目标: J = Σw_t(ŷ_t−T_set)² + λΣΔa² (a_seq: [H,2], requires_grad)"""
    mu, _ = model(x_hist, a_seq.unsqueeze(0))
    w = torch.linspace(1.0, 0.6, H)
    j = (w * (mu[0] - t_target) ** 2).sum()
    reg = 0.01 * ((a_seq[1:] - a_seq[:-1]) ** 2).sum()
    return j + reg


# ============ 3. 解析梯度 vs 有限差分 ============
a = a_init.clone().requires_grad_(True)
J = objective(a)
J.backward()
grad_analytic = a.grad.clone()  # [H, 2]

eps = 0.5  # 相对动作尺度(0-53)合理: 1% 阀位 (float32 下 1e-3 太小会数值下溢)
grad_fd = torch.zeros_like(a)
with torch.no_grad():
    for t in range(H):
        for k in range(2):
            ap = a.clone(); ap[t, k] += eps
            am = a.clone(); am[t, k] -= eps
            grad_fd[t, k] = (objective(ap) - objective(am)) / (2 * eps)

denom = grad_analytic.abs().max().item()
rel_err = (grad_analytic - grad_fd).abs().max().item() / (denom + 1e-12)
cos_sim = torch.nn.functional.cosine_similarity(
    grad_analytic.reshape(1, -1), grad_fd.reshape(1, -1)).item()
print(f"\n=== 梯度检查 ===")
print(f"解析梯度范数: {grad_analytic.norm():.4f} | 有限差分范数: {grad_fd.norm():.4f}")
print(f"最大相对误差: {rel_err:.2e} (应 <1e-2)")
print(f"余弦相似度:   {cos_sim:.6f} (应 >0.999)")
print(f"判定: {'✅ 可微性成立' if rel_err < 1e-2 and cos_sim > 0.999 else '❌ 梯度不一致!'}")

# ============ 4. 梯度方向物理性 ============
print(f"\n=== 梯度方向物理性 (∂T/∂a₂ < 0 = 开阀降温) ===")
with torch.no_grad():
    for t in [0, 4, 8, 12]:
        # 解析: ∂J/∂a₂(t) ≈ 2w_t(ŷ_t−T_set)·∂ŷ_t/∂a₂(t), 用中心差分量 ∂ŷ_t/∂a₂ 更直接
        ap = a.clone(); ap[t, 1] += eps
        am = a.clone(); am[t, 1] -= eps
        mu_p, _ = model(x_hist, ap.unsqueeze(0))
        mu_m, _ = model(x_hist, am.unsqueeze(0))
        dT_da2 = (mu_p[0, 12] - mu_m[0, 12]) / (2 * eps)  # 对第12步温度的敏感性
        print(f"∂T(12)/∂a₂({t}): {dT_da2:+.4f} °C/阀位 {'✅ 物理' if dT_da2 < 0 else '⚠️ 反物理'}")

# ============ 5. 规划收敛 (Adam 梯度上升, 与 MPC 同协议) ============
print(f"\n=== 规划收敛 (Adam, lr=0.05, 30 步) ===")
a_opt = a_init.clone().requires_grad_(True)
opt = torch.optim.Adam([a_opt], lr=0.05)
J0 = objective(a_opt).item()
Js = [J0]
for e in range(30):
    opt.zero_grad()
    Je = objective(a_opt)
    Je.backward()
    opt.step()
    Js.append(Je.item())
Jf = Js[-1]
print(f"J: {J0:.4f} → {Jf:.4f} ({'↓ 收敛' if Jf < J0 else '↑ 发散!'})")
print(f"最优动作序列首步: a₂={a_opt[0,1].item():.2f} (原 {a_init[0,1].item():.2f})")

# ============ 6. 全梯度路径验证 (MPC 反事实) ============
print(f"\n=== 端到端可微: 从随机动作到最优的梯度链路 ===")
a_rnd = (torch.rand(H, 2) * 40 + 5).requires_grad_(True)
J_rnd = objective(a_rnd)
J_rnd.backward()
print(f"随机动作 J={J_rnd.item():.4f}, 梯度范数={a_rnd.grad.norm():.4f} "
      f"({'✅ 梯度链路完整' if a_rnd.grad.norm() > 0 else '❌ 梯度为零'})")

result = {'grad_rel_err': float(rel_err), 'grad_cos_sim': float(cos_sim),
          'J0': J0, 'Jf': Jf, 'converged': Jf < J0}
json.dump(result, open("results/exp_026_grad_check.json", 'w'), indent=2)
print("\nSaved: results/exp_026_grad_check.json")
