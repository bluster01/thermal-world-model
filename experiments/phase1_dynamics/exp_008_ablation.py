"""
exp_008_ablation.py — 模型组件消融实验
========================================
基线: exp_006 最佳配置 (β-NLL, β=-0.3, warmup=20)

消融变体:
  A. Full (baseline)    — 完整模型复现 (作对照)
  B. Zero Actions       — 动作通道置零 (模型架构不变, 测动作信息贡献)
  C. No VarAttn         — 去掉 VariableAttention 层
  D. MLP Backbone        — 替换 PerVariableTCN → MLP
  E. No RevIN            — 去掉 RevIN 归一化

评测:
  - 1a: 一步预测 RMSE (各变量 + 主汽温)
  - 1b: 自回归展开 H={5,10,20,30} RMSE 曲线
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data
from world_model import WorldModel

EXP_ID = "exp_008"
EXP_DIR = os.path.join("results", EXP_ID)
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(os.path.join(EXP_DIR, "checkpoints"), exist_ok=True)

ROLLOUT_K = 5
ROLLOUT_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2]
BETA = -0.3
BETA_WARMUP = 20

N_PATCHES = (cfg.WINDOW_SIZE - 16) // 8 + 1  # = 11


class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0.0, eps=1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps
    
    def forward(self, mu, logvar, target):
        logvar = torch.clamp(logvar, -20.0, 20.0)
        var = torch.exp(logvar) + self.eps
        nll = 0.5 * (logvar + (target - mu) ** 2 / var)
        if self.beta != 0:
            nll = var.detach() ** self.beta * nll
        return nll.mean()


# ============= 消融模型构建 =============

def make_model(variant, device):
    """构建消融变体，返回 (model, use_zero_actions)"""
    common = dict(
        n_state=cfg.N_STATE, n_action=cfg.N_ACTION,
        window_size=cfg.WINDOW_SIZE,
        d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        dropout=cfg.DROPOUT, rollout_mode='sliding',
        probabilistic=True,
    )

    if variant == 'no_varattn':
        return WorldModel(n_var_layers=0, **common), False

    elif variant == 'mlp_backbone':
        model = WorldModel(**common)
        # 替换 PerVariableTCN → Flatten + MLP
        # PerVariableTCN 签入: [B*N, n_patches, d_model] → [B*N, d_model]
        model.var_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),                                    # [B*N, n_patches*d_model]
            nn.Linear(N_PATCHES * cfg.D_MODEL, cfg.D_MODEL * 4),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.D_MODEL * 4, cfg.D_MODEL * 2),
            nn.GELU(),
            nn.Linear(cfg.D_MODEL * 2, cfg.D_MODEL),
        ).to(device)
        return model, False

    elif variant == 'no_revin':
        model = WorldModel(**common)
        # 替换 RevIN 为恒等映射（保留接口兼容）
        class IdentityRevIN(nn.Module):
            def __init__(self, num_features):
                super().__init__()
                self.eps = 1e-5
                self.affine = False  # 跳过 denorm 中的 affine 步骤
                self.register_buffer('weight', torch.ones(num_features))
                self.register_buffer('bias', torch.zeros(num_features))
            def forward(self, x, mode='norm'):
                B, W, N = x.shape
                self._mean = torch.zeros(B, 1, N, device=x.device)
                self._std  = torch.ones(B, 1, N, device=x.device)
                return x
        model.revin = IdentityRevIN(cfg.N_STATE + cfg.N_ACTION)
        return model, False

    elif variant == 'zero_actions':
        # 同样架构, 动作通道全置 0
        return WorldModel(**common), True

    else:  # full
        return WorldModel(**common), False


# ============= 训练 =============

def make_batch(raw_data, indices, use_zero_actions):
    """构造训练批次 [B, W, total] 和 [B, K, n_state] 目标"""
    W, K = cfg.WINDOW_SIZE, ROLLOUT_K
    BS = len(indices)

    x_batch = np.zeros((BS, W, cfg.N_STATE + cfg.N_ACTION), dtype=np.float32)
    future_a_batch = np.zeros((BS, K, cfg.N_ACTION), dtype=np.float32)
    future_s_batch = np.zeros((BS, K, cfg.N_STATE), dtype=np.float32)

    for i, idx in enumerate(indices):
        # 历史窗口: [W, N_TOTAL]
        x_hist = raw_data[idx:idx+W, :]

        if use_zero_actions:
            x_hist[:, cfg.N_STATE:] = 0.0  # 零掉动作
            future_a_batch[i] = 0.0
        else:
            future_a_batch[i] = raw_data[idx+W:idx+W+K, cfg.N_STATE:]

        x_batch[i] = x_hist
        future_s_batch[i] = raw_data[idx+W:idx+W+K, :cfg.N_STATE]

    return (torch.FloatTensor(x_batch),
            torch.FloatTensor(future_a_batch),
            torch.FloatTensor(future_s_batch))


def train_epoch(model, raw_data, device, optimizer, criterion, use_zero_actions, BS=256, steps=500):
    model.train()
    W, K = cfg.WINDOW_SIZE, ROLLOUT_K
    N = len(raw_data)
    total_loss = 0.0

    for _ in range(steps):
        indices = np.random.randint(0, N - W - K, size=BS)
        x_t, a_seq_t, s_true_t = make_batch(raw_data, indices, use_zero_actions)
        x_t, a_seq_t, s_true_t = x_t.to(device), a_seq_t.to(device), s_true_t.to(device)

        optimizer.zero_grad()

        # Sliding rollout: 保持窗口, 逐步替换预测
        state_win = x_t[:, :, :cfg.N_STATE].clone()
        action_win = x_t[:, :, cfg.N_STATE:].clone()
        total_step_loss = 0.0

        for k in range(K):
            x_step = torch.cat([state_win, action_win], dim=2)
            mu, logvar = model(x_step)
            step_loss = ROLLOUT_WEIGHTS[k] * criterion(mu, logvar, s_true_t[:, k, :])
            total_step_loss += step_loss

            # 滑动窗口: 去掉最早, 推入预测
            state_win = torch.cat([state_win[:, 1:, :], mu.unsqueeze(1).detach()], dim=1)
            action_win = torch.cat([action_win[:, 1:, :], a_seq_t[:, k:k+1, :]], dim=1)

        total_step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += total_step_loss.item()

    return total_loss / steps


@torch.no_grad()
def validate(model, raw_data, device, use_zero_actions, n_samples=200):
    """快速验证: step-0 和 step-4 MAE"""
    model.eval()
    W, K = cfg.WINDOW_SIZE, ROLLOUT_K
    N = len(raw_data)
    mae_short, mae_long = 0.0, 0.0

    for _ in range(n_samples):
        idx = np.random.randint(0, N - W - K)
        x_t, a_seq_t, s_true_t = make_batch(raw_data, [idx], use_zero_actions)
        x_t, a_seq_t, s_true_t = x_t.to(device), a_seq_t.to(device), s_true_t.to(device)

        s_traj = model.rollout(x_t, a_seq_t, mode='sliding')
        mae_short += (s_traj[0, 0, cfg.TARGET_IDX] - s_true_t[0, 0, cfg.TARGET_IDX]).abs().item()
        mae_long  += (s_traj[0, min(4, K-1), cfg.TARGET_IDX] - s_true_t[0, min(4, K-1), cfg.TARGET_IDX]).abs().item()

    return mae_short / n_samples, mae_long / n_samples


@torch.no_grad()
def evaluate_rollout(model, raw_data, device, use_zero_actions, H, n_samples=500):
    """自回归展开 H 步, 返回 per-step MAE array 和 RMSE array"""
    W = cfg.WINDOW_SIZE
    N = len(raw_data)
    np.random.seed(42)
    indices = np.random.choice(range(N - W - H), n_samples, replace=False)

    errors_mae = np.zeros(H)
    errors_sq = np.zeros(H)

    for idx in indices:
        s_hist = raw_data[idx:idx+W, :cfg.N_STATE]
        a_hist = raw_data[idx:idx+W, cfg.N_STATE:]

        future_a_raw = raw_data[idx+W:idx+W+H, cfg.N_STATE:]
        if use_zero_actions:
            a_hist[:] = 0
            future_a_raw[:] = 0

        x_hist = np.concatenate([s_hist, a_hist], axis=1)
        x_t = torch.FloatTensor(x_hist).unsqueeze(0).to(device)
        a_t = torch.FloatTensor(future_a_raw).unsqueeze(0).to(device)
        mainT_true = raw_data[idx+W:idx+W+H, cfg.TARGET_IDX]

        s_traj = model.rollout(x_t, a_t, mode='sliding')
        preds = s_traj[0, :, cfg.TARGET_IDX].cpu().numpy()

        errors_mae += np.abs(preds - mainT_true)
        errors_sq += (preds - mainT_true) ** 2

    return errors_mae / n_samples, np.sqrt(errors_sq / n_samples)


@torch.no_grad()
def evaluate_one_step(model, raw_data, device, use_zero_actions, n_samples=2000):
    """一步预测 per-variable RMSE"""
    model.eval()
    W = cfg.WINDOW_SIZE
    N = len(raw_data)
    np.random.seed(42)
    indices = np.random.choice(range(N - W - 1), n_samples, replace=False)

    errors_sq = np.zeros(cfg.N_STATE)
    for idx in indices:
        x_hist = raw_data[idx:idx+W, :].copy()
        if use_zero_actions:
            x_hist[:, cfg.N_STATE:] = 0
        x_t = torch.FloatTensor(x_hist).unsqueeze(0).to(device)
        s_true = raw_data[idx + W, :cfg.N_STATE]
        mu, _ = model(x_t)
        errors_sq += (mu[0].cpu().numpy() - s_true) ** 2

    return np.sqrt(errors_sq / n_samples)


# ============= 单变体训练循环 =============

def train_variant(variant, raw_data, train_data, val_data, test_data, device):
    print(f"\n{'='*55}")
    print(f"  [{EXP_ID}] Ablation: {variant}")
    print(f"{'='*55}")

    model, use_zero_actions = make_model(variant, device)
    model = model.to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Zero actions: {use_zero_actions}")

    criterion = BetaNLLLoss(beta=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_mae = float('inf')
    patience_counter = 0
    t0 = time.time()
    best_epoch = 0

    for epoch in range(1, cfg.EPOCHS + 1):
        # Beta warmup
        if epoch <= BETA_WARMUP:
            criterion.beta = 0.0
        else:
            progress = min((epoch - BETA_WARMUP) / 10, 1.0)
            criterion.beta = BETA * progress

        train_loss = train_epoch(model, train_data, device, optimizer, criterion, use_zero_actions)
        val_mae0, val_mae4 = validate(model, val_data, device, use_zero_actions)

        scheduler.step(val_mae4)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:6.2f} | "
                  f"Val-MAE(0): {val_mae0:.4f} | Val-MAE(4): {val_mae4:.4f}")

        if val_mae4 < best_val_mae - 0.001:
            best_val_mae = val_mae4
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_mae4': val_mae4},
                os.path.join(EXP_DIR, "checkpoints", f"{variant}_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"  早停于 epoch {epoch}, 最佳 epoch {best_epoch}")
                break

    train_time = time.time() - t0

    # Load best
    ckpt = torch.load(os.path.join(EXP_DIR, "checkpoints", f"{variant}_best.pth"),
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 1a: One-step RMSE
    print(f"\n  1a: 一步预测 per-variable RMSE...")
    one_step_rmse = evaluate_one_step(model, test_data, device, use_zero_actions)

    # 1b: Autoregressive unrolling
    print(f"  1b: 自回归展开 H=5,10,20,30...")
    rollout_results = {}
    for H_step in [5, 10, 20, 30]:
        mae, rmse = evaluate_rollout(model, test_data, device, use_zero_actions, H_step)
        growth = mae[-1] / (mae[0] + 1e-8) if mae[0] > 0 else float('inf')
        rollout_results[f"H{H_step}"] = {
            'mae': mae.tolist(),
            'rmse': rmse.tolist(),
            'mae_final': float(mae[-1]),
            'rmse_final': float(rmse[-1]),
            'mae_growth': float(growth),
        }
        print(f"    H={H_step:2d}: step0-{mae[0]:.4f} → step{H_step-1}-{mae[-1]:.4f} "
              f"(growth ×{growth:.1f})  |  step0-RMSE: {rmse[0]:.4f} → {rmse[-1]:.4f}")

    result = {
        'variant': variant,
        'params': sum(p.numel() for p in model.parameters()),
        'train_time_min': train_time / 60,
        'best_epoch': ckpt['epoch'],
        'zero_actions': use_zero_actions,
        'one_step_rmse': {str(i): float(v) for i, v in enumerate(one_step_rmse)},
        'one_step_rmse_target': float(one_step_rmse[cfg.TARGET_IDX]),
        'rollout': rollout_results,
    }

    # Per-variant summary
    print(f"  → 1-step RMSE(主汽温): {one_step_rmse[cfg.TARGET_IDX]:.4f}")
    print(f"  → H=30 final MAE: {rollout_results['H30']['mae_final']:.4f}")

    return result


# ============= Main =============

def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    state_data, delta_actions, _ = load_raw_data()
    raw_data = np.concatenate([state_data, delta_actions], axis=1)
    n_total = len(raw_data)
    n_train = int(n_total * 0.70)
    n_val_end = int(n_total * 0.85)
    train_data = raw_data[:n_train]
    val_data = raw_data[n_train:n_val_end]
    test_data = raw_data[n_val_end:]

    print(f"[{EXP_ID}] Component Ablation Study")
    print(f"  Data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    print(f"  Config: N_STATE={cfg.N_STATE}, N_ACTION={cfg.N_ACTION}, W={cfg.WINDOW_SIZE}")
    print(f"  n_patches={N_PATCHES}, d_model={cfg.D_MODEL}")

    variants = ['full', 'zero_actions', 'no_varattn', 'mlp_backbone', 'no_revin']
    all_results = {}

    for variant in variants:
        result = train_variant(variant, raw_data, train_data, val_data, test_data, device)
        all_results[variant] = result

    # ============================
    #           汇总
    # ============================
    print(f"\n\n{'='*80}")
    print(f"  ABLATION SUMMARY — {EXP_ID}")
    print(f"{'='*80}")

    # Table: one-step + H=5,10,20,30 final MAE
    header_parts = [f"{'Variant':<16}", f"{'1-step':>8}", f"{'H=5':>8}",
                    f"{'H=10':>8}", f"{'H=20':>8}", f"{'H=30':>8}", f"{'×Grow':>6}"]
    print("  " + " ".join(header_parts))
    print("  " + "-" * 62)

    baseline_mae = {}
    for variant in variants:
        r = all_results[variant]
        row = f"  {variant:<16} {r['one_step_rmse_target']:>8.4f}"
        for h in [5, 10, 20, 30]:
            key = f"H{h}"
            mae_f = r['rollout'][key]['mae_final']
            if variant == 'full':
                baseline_mae[h] = mae_f
            row += f" {mae_f:>8.4f}"
        row += f" {r['rollout']['H30']['mae_growth']:>5.1f}x"
        print(row)

    # Δ vs Full
    print(f"\n  Δ vs Full (%):")
    print(f"  {'Variant':<16} {'1-step':>8} {'H=5':>8} {'H=10':>8} {'H=20':>8} {'H=30':>8}")
    full_r = all_results['full']
    for variant in ['zero_actions', 'no_varattn', 'mlp_backbone', 'no_revin']:
        r = all_results[variant]
        d1 = (r['one_step_rmse_target'] - full_r['one_step_rmse_target']) / full_r['one_step_rmse_target'] * 100
        row = f"  {variant:<16} {d1:>+7.1f}%"
        for h in [5, 10, 20, 30]:
            d = (r['rollout'][f'H{h}']['mae_final'] - baseline_mae[h]) / baseline_mae[h] * 100
            row += f" {d:>+7.1f}%"
        print(row)

    # Save
    out_path = os.path.join(EXP_DIR, "ablation_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  ✓ Results saved: {out_path}")

    # Save easy-to-read text summary
    summary_path = os.path.join(EXP_DIR, "summary.txt")
    with open(summary_path, 'w') as f:
        for variant in variants:
            r = all_results[variant]
            f.write(f"\n## {variant}\n")
            f.write(f"  params: {r['params']:,}\n")
            f.write(f"  best_epoch: {r['best_epoch']}\n")
            f.write(f"  1-step RMSE(target): {r['one_step_rmse_target']:.4f}\n")
            for h in [5, 10, 20, 30]:
                rr = r['rollout'][f'H{h}']
                f.write(f"  H={h}: step0 MAE={rr['mae'][0]:.4f} → final MAE={rr['mae_final']:.4f} "
                        f"(×{rr['mae_growth']:.1f})\n")
    print(f"  ✓ Summary: {summary_path}")


if __name__ == '__main__':
    main()
