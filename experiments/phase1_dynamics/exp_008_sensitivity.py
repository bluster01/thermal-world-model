"""
exp_008_sensitivity.py — 动作敏感性分析 (1c)
固定 s_t，改变 a_t ±5%/±10%/±20%，观察 ŝ_{t+1} 主汽温变化
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data
from world_model import WorldModel

EXP_DIR = "results/exp_008"

# Load best models
N_PATCHES = (cfg.WINDOW_SIZE - 16) // 8 + 1  # = 11

def load_model(variant, device):
    ckpt = torch.load(os.path.join(EXP_DIR, "checkpoints", f"{variant}_best.pth"),
                      map_location=device, weights_only=True)
    
    common = dict(n_state=cfg.N_STATE, n_action=cfg.N_ACTION,
                  window_size=cfg.WINDOW_SIZE, d_model=cfg.D_MODEL,
                  n_heads=cfg.N_HEADS, dropout=cfg.DROPOUT,
                  rollout_mode='sliding', probabilistic=True)
    
    model = WorldModel(**common)
    
    if variant == 'mlp_backbone':
        # Rebuild MLP backbone matching the ablation script
        model.var_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(N_PATCHES * cfg.D_MODEL, cfg.D_MODEL * 4),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.D_MODEL * 4, cfg.D_MODEL * 2),
            nn.GELU(),
            nn.Linear(cfg.D_MODEL * 2, cfg.D_MODEL),
        )
    
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def action_sensitivity(model, raw_data, device, n_samples=500, deltas=[-0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20]):
    """
    对每个样本: 固定 s_hist，在基准动作上加减 delta，
    记录主汽温预测变化。返回 {delta: [ΔT samples]}
    """
    W = cfg.WINDOW_SIZE
    N = len(raw_data)
    np.random.seed(42)
    indices = np.random.choice(range(N - W - 1), n_samples, replace=False)

    results = {d: [] for d in deltas}

    for idx in indices:
        s_hist = raw_data[idx:idx+W, :cfg.N_STATE]
        a_hist = raw_data[idx:idx+W, cfg.N_STATE:]

        # 基准预测
        x_base = np.concatenate([s_hist, a_hist], axis=1)
        x_t = torch.FloatTensor(x_base).unsqueeze(0).to(device)
        mu_base, _ = model(x_t)
        t_base = mu_base[0, cfg.TARGET_IDX].item()

        # 改变最后一个动作
        for d in deltas:
            if d == 0:
                results[d].append(0.0)
                continue
            a_perturbed = a_hist.copy()
            a_perturbed[-1] = a_hist[-1] * (1 + d)  # 按比例缩放
            x_pert = np.concatenate([s_hist, a_perturbed], axis=1)
            x_t = torch.FloatTensor(x_pert).unsqueeze(0).to(device)
            mu_pert, _ = model(x_t)
            t_pert = mu_pert[0, cfg.TARGET_IDX].item()
            results[d].append(t_pert - t_base)

    return results


def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')

    state_data, delta_actions, _ = load_raw_data()
    raw_data = np.concatenate([state_data, delta_actions], axis=1)

    # 只用 full (最佳一步精度) 和 mlp_backbone (最佳长期)
    for variant in ['full', 'mlp_backbone']:
        ckpt_path = os.path.join(EXP_DIR, "checkpoints", f"{variant}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {variant}: no checkpoint")
            continue

        print(f"\n{'='*60}")
        print(f"  Action Sensitivity: {variant}")
        print(f"{'='*60}")

        model = load_model(variant, device)
        deltas = [-0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20]
        sens = action_sensitivity(model, raw_data, device, n_samples=500, deltas=deltas)

        # 主汽温变化统计
        print(f"\n  阀位变化 → 主汽温Δ(°C)  [500 samples, mean ± std]")
        print(f"  {'Δ action':>10}  {'Δ T mean':>10}  {'Δ T std':>10}  {'方向':>6}")
        print(f"  {'-'*42}")

        for d in deltas:
            if d == 0:
                continue
            arr = np.array(sens[d])
            direction = "升" if np.mean(arr) > 0 else "降"
            print(f"  {d:>+8.0%}    {np.mean(arr):>+10.4f}  {np.std(arr):>10.4f}  {direction:>6}")

        # 检查总体因果方向: 开阀(正) → 温度上升/下降？
        pos_deltas = [d for d in deltas if d > 0]
        neg_deltas = [d for d in deltas if d < 0]
        pos_mean = np.mean([np.mean(sens[d]) for d in pos_deltas])
        neg_mean = np.mean([np.mean(sens[d]) for d in neg_deltas])

        print(f"\n  → 正方向(开阀) 平均ΔT: {pos_mean:+.4f}°C")
        print(f"  → 负方向(关阀) 平均ΔT: {neg_mean:+.4f}°C")
        print(f"  → 响应对称性: {abs(pos_mean/neg_mean) if neg_mean != 0 else 'N/A':.2f}x")


if __name__ == '__main__':
    main()
