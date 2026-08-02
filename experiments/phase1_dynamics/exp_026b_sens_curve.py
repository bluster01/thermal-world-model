#!/usr/bin/env python3
"""
M0 敏感性曲线 vs 事件研究真值 — 严格验证首步物理方向
======================================================
问题: 首步扰动的 ∂T/∂a₂ < 0 方向是否可靠?

方法 (与 exp_1c 同协议, 但针对 Direct WM):
1. M0 完整敏感性曲线: 二级阀首步 +10 → ΔT(t) for t=1..17 全时标
2. 真值 (事件研究, memory): 开阀 60-120s+ 起效, 120s 附近 −0.59°C, 时标 60-90s+
3. 判定: 方向 (t≥8 应 <0), 时标 (响应从 t≈6-8 开始显著), 单调性 (随 t 增强)
4. 额外: 一级阀对照 + 首步扰动 vs 全步扰动 (确认共因 vs 因果)
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from experiments.phase1_dynamics.exp_025_unified_benchmark import (
    build_model, test_raw, VALVE_IDX, TARGET_IDX, H_OUT)

DEVICE = torch.device('cpu')
W = cfg.WINDOW_SIZE
H = H_OUT

model = build_model('M0').to(DEVICE).eval()
ck = torch.load("results/exp_025_M0/checkpoints/best_model.pth",
                map_location=DEVICE, weights_only=True)
model.load_state_dict(ck['model_state_dict'])
print(f"M0 加载 OK | {sum(p.numel() for p in model.parameters()):,} 参数")


@torch.no_grad()
def full_curve(adim, delta, perturb_mode='first', n=200, seed=42):
    """完整 ΔT(t) 曲线. perturb_mode: 'first'=仅首步, 'all'=全步"""
    model.eval(); N = len(test_raw)
    np.random.seed(seed)
    idxs = np.random.choice(range(N - W - H), n, replace=False)
    curves = np.zeros((n, H))
    for j, i in enumerate(idxs):
        x_hist = torch.FloatTensor(test_raw[i:i+W]).unsqueeze(0).to(DEVICE)
        a_fut = torch.FloatTensor(test_raw[i+W:i+W+H, VALVE_IDX]).clone()
        mu_b, _ = model(x_hist, a_fut.unsqueeze(0))
        bp = mu_b[0].cpu().numpy()
        a_p = a_fut.clone()
        if perturb_mode == 'first':
            a_p[0, adim] = np.clip(a_p[0, adim] + delta, 0, 100)
        else:
            a_p[:, adim] = np.clip(a_p[:, adim] + delta, 0, 100)
        mu_p, _ = model(x_hist, a_p.unsqueeze(0))
        pp = mu_p[0].cpu().numpy()
        curves[j] = pp - bp
    return curves.mean(0)


print("\n===== M0 敏感性曲线 vs 事件研究真值 =====")
print("真值: 开阀 60-120s 起效, ~120s 降温, 时标 60-90s+; 一级阀类似但较弱")

for adim, name in [(1, '二级阀'), (0, '一级阀')]:
    c_first = full_curve(adim, +10.0, 'first')
    c_all = full_curve(adim, +10.0, 'all')
    print(f"\n--- {name} 首步扰动 +10 → ΔT(t) ---")
    print("t:  " + "  ".join(f"{t:4d}" for t in range(0, H, 2)))
    print("ΔT: " + "  ".join(f"{c_first[t]:+.3f}" for t in range(0, H, 2)))
    t8, t12, t17 = c_first[7], c_first[11], c_first[17]
    dir_ok = t8 < 0 and t12 < 0 and t17 < 0
    # 时标: t1 应接近 0 (60-90s 才起效), 响应强度随 t 增强 (前 8 步)
    t1_small = abs(c_first[0]) < 0.15
    growing = c_first[11] < c_first[3]
    print(f"  判定: 方向(全部<0)={'✅' if dir_ok else '❌'} | "
          f"t1≈0(60s内无效应)={'✅' if t1_small else '⚠️'} | "
          f"随时间增强={'✅' if growing else '⚠️'}")
    print(f"  {name}首步 vs 全步 @t12: {c_first[11]:+.3f} vs {c_all[11]:+.3f} "
          f"(差异={'✅ 首步测因果' if abs(c_first[11]) > abs(c_all[11]) else '⚠️ 全步反而更强'})")

result = {'method': 'M0 sensitivity curve vs event study truth'}
json.dump(result, open("results/exp_026b_sens_curve.json", 'w'), indent=2)
print("\nSaved: results/exp_026b_sens_curve.json")
