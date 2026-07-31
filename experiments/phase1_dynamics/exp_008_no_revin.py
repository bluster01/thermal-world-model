"""exp_008_no_revin.py — 补跑 no_revin 消融"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data
from world_model import WorldModel

EXP_ID = "exp_008"
EXP_DIR = os.path.join("results", EXP_ID)

BETA = -0.3
BETA_WARMUP = 20
ROLLOUT_K = 5
ROLLOUT_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2]


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


class IdentityRevIN(nn.Module):
    """恒等 RevIN — 所有属性兼容但不做归一化"""
    def __init__(self, num_features):
        super().__init__()
        self.eps = 1e-5
        self.affine = False
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
    def forward(self, x, mode='norm'):
        B, W, N = x.shape
        self._mean = torch.zeros(B, 1, N, device=x.device)
        self._std  = torch.ones(B, 1, N, device=x.device)
        return x


def make_model(device):
    model = WorldModel(
        n_state=cfg.N_STATE, n_action=cfg.N_ACTION,
        window_size=cfg.WINDOW_SIZE, d_model=cfg.D_MODEL,
        n_heads=cfg.N_HEADS, dropout=cfg.DROPOUT,
        rollout_mode='sliding', probabilistic=True)
    model.revin = IdentityRevIN(cfg.N_STATE + cfg.N_ACTION)
    model = model.to(device)
    return model


def make_batch(raw_data, indices):
    W, K = cfg.WINDOW_SIZE, ROLLOUT_K
    BS = len(indices)
    x_batch = np.zeros((BS, W, cfg.N_STATE + cfg.N_ACTION), dtype=np.float32)
    future_a_batch = np.zeros((BS, K, cfg.N_ACTION), dtype=np.float32)
    future_s_batch = np.zeros((BS, K, cfg.N_STATE), dtype=np.float32)
    for i, idx in enumerate(indices):
        x_batch[i] = raw_data[idx:idx+W, :]
        future_a_batch[i] = raw_data[idx+W:idx+W+K, cfg.N_STATE:]
        future_s_batch[i] = raw_data[idx+W:idx+W+K, :cfg.N_STATE]
    return (torch.FloatTensor(x_batch),
            torch.FloatTensor(future_a_batch),
            torch.FloatTensor(future_s_batch))


def train_epoch(model, raw_data, device, optimizer, criterion, BS=256, steps=500):
    model.train()
    W, K = cfg.WINDOW_SIZE, ROLLOUT_K
    N = len(raw_data)
    total_loss = 0.0
    for _ in range(steps):
        indices = np.random.randint(0, N - W - K, size=BS)
        x_t, a_seq_t, s_true_t = make_batch(raw_data, indices)
        x_t, a_seq_t, s_true_t = x_t.to(device), a_seq_t.to(device), s_true_t.to(device)
        optimizer.zero_grad()
        state_win = x_t[:, :, :cfg.N_STATE].clone()
        action_win = x_t[:, :, cfg.N_STATE:].clone()
        total_step_loss = 0.0
        for k in range(K):
            x_step = torch.cat([state_win, action_win], dim=2)
            mu, logvar = model(x_step)
            step_loss = ROLLOUT_WEIGHTS[k] * criterion(mu, logvar, s_true_t[:, k, :])
            total_step_loss += step_loss
            state_win = torch.cat([state_win[:, 1:, :], mu.unsqueeze(1).detach()], dim=1)
            action_win = torch.cat([action_win[:, 1:, :], a_seq_t[:, k:k+1, :]], dim=1)
        total_step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += total_step_loss.item()
    return total_loss / steps


@torch.no_grad()
def validate(model, raw_data, device, n_samples=200):
    model.eval()
    W, K = cfg.WINDOW_SIZE, ROLLOUT_K
    N = len(raw_data)
    mae0, mae4 = 0.0, 0.0
    for _ in range(n_samples):
        idx = np.random.randint(0, N - W - K)
        x_t, a_seq_t, s_true_t = make_batch(raw_data, [idx])
        x_t, a_seq_t, s_true_t = x_t.to(device), a_seq_t.to(device), s_true_t.to(device)
        s_traj = model.rollout(x_t, a_seq_t, mode='sliding')
        mae0 += (s_traj[0, 0, cfg.TARGET_IDX] - s_true_t[0, 0, cfg.TARGET_IDX]).abs().item()
        mae4 += (s_traj[0, min(4, K-1), cfg.TARGET_IDX] - s_true_t[0, min(4, K-1), cfg.TARGET_IDX]).abs().item()
    return mae0 / n_samples, mae4 / n_samples


@torch.no_grad()
def evaluate_rollout(model, raw_data, device, H, n_samples=500):
    W = cfg.WINDOW_SIZE
    N = len(raw_data)
    np.random.seed(42)
    indices = np.random.choice(range(N - W - H), n_samples, replace=False)
    errors_mae = np.zeros(H)
    errors_sq = np.zeros(H)
    for idx in indices:
        s_hist = raw_data[idx:idx+W, :cfg.N_STATE]
        a_hist = raw_data[idx:idx+W, cfg.N_STATE:]
        future_a = raw_data[idx+W:idx+W+H, cfg.N_STATE:]
        x_t = torch.FloatTensor(np.concatenate([s_hist, a_hist], axis=1)).unsqueeze(0).to(device)
        a_t = torch.FloatTensor(future_a).unsqueeze(0).to(device)
        mainT_true = raw_data[idx+W:idx+W+H, cfg.TARGET_IDX]
        s_traj = model.rollout(x_t, a_t, mode='sliding')
        preds = s_traj[0, :, cfg.TARGET_IDX].cpu().numpy()
        errors_mae += np.abs(preds - mainT_true)
        errors_sq += (preds - mainT_true) ** 2
    return errors_mae / n_samples, np.sqrt(errors_sq / n_samples)


@torch.no_grad()
def evaluate_one_step(model, raw_data, device, n_samples=2000):
    model.eval()
    W = cfg.WINDOW_SIZE
    N = len(raw_data)
    np.random.seed(42)
    indices = np.random.choice(range(N - W - 1), n_samples, replace=False)
    errors_sq = np.zeros(cfg.N_STATE)
    for idx in indices:
        x_t = torch.FloatTensor(raw_data[idx:idx+W, :]).unsqueeze(0).to(device)
        s_true = raw_data[idx + W, :cfg.N_STATE]
        mu, _ = model(x_t)
        errors_sq += (mu[0].cpu().numpy() - s_true) ** 2
    return np.sqrt(errors_sq / n_samples)


def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    state_data, delta_actions, _ = load_raw_data()
    raw_data = np.concatenate([state_data, delta_actions], axis=1)
    n_total = len(raw_data)
    n_train = int(n_total * 0.70)
    n_val_end = int(n_total * 0.85)
    train_data = raw_data[:n_train]
    val_data = raw_data[n_train:n_val_end]
    test_data = raw_data[n_val_end:]

    print(f"[{EXP_ID}] Ablation: no_revin")
    model = make_model(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = BetaNLLLoss(beta=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_mae = float('inf')
    patience_counter = 0
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        if epoch <= BETA_WARMUP:
            criterion.beta = 0.0
        else:
            criterion.beta = BETA * min((epoch - BETA_WARMUP) / 10, 1.0)

        train_loss = train_epoch(model, train_data, device, optimizer, criterion)
        val_mae0, val_mae4 = validate(model, val_data, device)
        scheduler.step(val_mae4)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:6.2f} | "
                  f"Val-MAE(0): {val_mae0:.4f} | Val-MAE(4): {val_mae4:.4f}")

        if val_mae4 < best_val_mae - 0.001:
            best_val_mae = val_mae4
            best_epoch = epoch
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_mae4': val_mae4},
                       os.path.join(EXP_DIR, "checkpoints", "no_revin_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"  早停于 epoch {epoch}, 最佳 epoch {best_epoch}")
                break

    train_time = time.time() - t0
    ckpt = torch.load(os.path.join(EXP_DIR, "checkpoints", "no_revin_best.pth"),
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"\n  1a: 一步预测 per-variable RMSE...")
    one_step_rmse = evaluate_one_step(model, test_data, device)

    print(f"  1b: 自回归展开 H=5,10,20,30...")
    for H_step in [5, 10, 20, 30]:
        mae, rmse = evaluate_rollout(model, test_data, device, H_step)
        g = mae[-1] / (mae[0] + 1e-8)
        print(f"    H={H_step:2d}: step0-{mae[0]:.4f} → step{H_step-1}-{mae[-1]:.4f} "
              f"(growth ×{g:.1f})  |  RMSE: {rmse[0]:.4f} → {rmse[-1]:.4f}")

    print(f"\n  → 1-step RMSE(主汽温): {one_step_rmse[cfg.TARGET_IDX]:.4f}")
    print(f"  → Train time: {train_time/60:.1f} min")
    print(f"  → Best epoch: {ckpt['epoch']}")


if __name__ == '__main__':
    main()
