"""
exp_005_train.py — 概率世界模型 (Gaussian NLL loss)
====================================================
改动 vs exp_004:
- WorldModel 输出 (μ, logσ²) 双头
- 损失: Gaussian Negative Log-Likelihood
- 训练: 多步 rollout loss K=5, 每步用 NLL
- 评测: Sliding vs GRU, 同时输出 σ (预测不确定性)

实验编号: exp_005
"""
import os, sys, json, time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data
from world_model import WorldModel

EXP_ID = "exp_005"
EXP_DIR = os.path.join("results", EXP_ID)
os.makedirs(os.path.join(EXP_DIR, "checkpoints"), exist_ok=True)

ROLLOUT_K = 5
ROLLOUT_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2]

EXP_META = {
    "exp_id": EXP_ID,
    "name": "Probabilistic World Model — Gaussian NLL loss",
    "date": datetime.now().isoformat(),
    "changes_vs_exp004": [
        "损失: MSE → Gaussian NLL (GaussNLLLoss)",
        "模型输出: ŝ → (μ, logσ²) 双头",
        "Rollout 用 μ 做确定性预测, σ 评估不确定性",
    ],
    "rollout_k": ROLLOUT_K,
    "rollout_weights": ROLLOUT_WEIGHTS,
    "loss": "gaussian_nll",
}

with open(os.path.join(EXP_DIR, "meta.json"), 'w') as f:
    json.dump(EXP_META, f, indent=2, default=str)


class GaussNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood Loss
    
    NLL = 0.5 * mean( log(σ²) + (y - μ)² / σ² )
    
    其中 σ² = exp(logvar), 数值稳定处理:
    - logvar 钳在 [-20, 20] 防止 exp 溢出
    - σ² 加 1e-6 防止除零
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, mu, logvar, target):
        logvar = torch.clamp(logvar, -20.0, 20.0)
        var = torch.exp(logvar) + self.eps
        loss = 0.5 * (logvar + (target - mu) ** 2 / var)
        return loss.mean()


def train_epoch(model, raw_data, device, optimizer, criterion, BS=256, steps=500):
    """多步 rollout + NLL loss 训练 (sliding 模式)"""
    model.train()
    W, K = cfg.WINDOW_SIZE, ROLLOUT_K
    N = len(raw_data)
    total_loss = 0.0
    total_nll = 0.0
    total_mse = 0.0
    
    for _ in range(steps):
        indices = np.random.randint(0, N - W - K, size=BS)
        
        x_hist_batch, a_seq_batch, s_true_batch = [], [], []
        for idx in indices:
            s_win = raw_data[idx:idx+W, :cfg.N_STATE]
            a_win = raw_data[idx:idx+W, cfg.N_STATE:]
            x_hist = np.concatenate([s_win, a_win], axis=1)
            future_a = raw_data[idx+W:idx+W+K, cfg.N_STATE:]
            future_s = raw_data[idx+W:idx+W+K, :cfg.N_STATE]
            x_hist_batch.append(x_hist)
            a_seq_batch.append(future_a)
            s_true_batch.append(future_s)
        
        x_hist_t = torch.FloatTensor(np.stack(x_hist_batch)).to(device)
        a_seq_t = torch.FloatTensor(np.stack(a_seq_batch)).to(device)
        s_true_t = torch.FloatTensor(np.stack(s_true_batch)).to(device)
        
        optimizer.zero_grad()
        
        states = x_hist_t[:, :, :cfg.N_STATE]
        actions = x_hist_t[:, :, cfg.N_STATE:]
        total_step_loss = 0.0
        
        for k in range(K):
            x_t = torch.cat([states, actions], dim=2)
            mu, logvar = model(x_t)
            step_loss = ROLLOUT_WEIGHTS[k] * criterion(mu, logvar, s_true_t[:, k, :])
            total_step_loss += step_loss
            
            # 记录 MSE 用于监控
            total_mse += F.mse_loss(mu, s_true_t[:, k, :]).item() * ROLLOUT_WEIGHTS[k]
            
            # 用 μ 更新窗口 (detach 阻断梯度回传到上一步)
            states = torch.cat([states[:, 1:, :], mu.unsqueeze(1).detach()], dim=1)
            actions = torch.cat([actions[:, 1:, :], a_seq_t[:, k:k+1, :]], dim=1)
        
        total_step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += total_step_loss.item()
        total_nll += total_step_loss.item()
    
    return total_loss / steps, total_mse / steps


@torch.no_grad()
def validate(model, raw_data, device, n_samples=200):
    """验证: step0 和 step4 MAE (用 μ 计算)"""
    model.eval()
    W, K = cfg.WINDOW_SIZE, ROLLOUT_K
    N = len(raw_data)
    
    mae0, mae4 = 0.0, 0.0
    for _ in range(n_samples):
        idx = np.random.randint(0, N - W - K)
        s_win = raw_data[idx:idx+W, :cfg.N_STATE]
        a_win = raw_data[idx:idx+W, cfg.N_STATE:]
        x_t = torch.FloatTensor(np.concatenate([s_win, a_win], axis=1)).unsqueeze(0).to(device)
        future_a = torch.FloatTensor(raw_data[idx+W:idx+W+K, cfg.N_STATE:]).unsqueeze(0).to(device)
        s_true = torch.FloatTensor(raw_data[idx+W:idx+W+K, :cfg.N_STATE]).unsqueeze(0).to(device)
        
        s_traj = model.rollout(x_t, future_a, mode='sliding')
        mae0 += torch.abs(s_traj[0, 0, cfg.TARGET_IDX] - s_true[0, 0, cfg.TARGET_IDX]).item()
        mae4 += torch.abs(s_traj[0, min(4, K-1), cfg.TARGET_IDX] - s_true[0, min(4, K-1), cfg.TARGET_IDX]).item()
    
    return mae0 / n_samples, mae4 / n_samples


def evaluate_full_rollout(model, raw_data, device, mode='sliding', n_samples=500):
    """完整18步 rollout 评测 (用 μ)"""
    W, H = cfg.WINDOW_SIZE, cfg.ROLLOUT_LEN
    N = len(raw_data)
    np.random.seed(42)
    indices = np.random.choice(range(N - W - H), n_samples, replace=False)
    
    errors = np.zeros((n_samples, H))
    
    for i, idx in enumerate(indices):
        s_win = raw_data[idx:idx+W, :cfg.N_STATE]
        a_win = raw_data[idx:idx+W, cfg.N_STATE:]
        x_t = torch.FloatTensor(np.concatenate([s_win, a_win], axis=1)).unsqueeze(0).to(device)
        future_a = torch.FloatTensor(raw_data[idx+W:idx+W+H, cfg.N_STATE:]).unsqueeze(0).to(device)
        mainT_true = raw_data[idx+W:idx+W+H, cfg.TARGET_IDX]
        
        s_traj = model.rollout(x_t, future_a, mode=mode)
        mainT_pred = s_traj[0, :, cfg.TARGET_IDX].detach().cpu().numpy()
        errors[i] = np.abs(mainT_pred - mainT_true)
    
    return errors.mean(axis=0), np.sqrt((errors**2).mean(axis=0))


def evaluate_uncertainty(model, raw_data, device, mode='sliding', n_samples=200):
    """评估 σ 是否校准良好 (σ vs 实际 |error|)"""
    W, H = cfg.WINDOW_SIZE, cfg.ROLLOUT_LEN
    N = len(raw_data)
    np.random.seed(42)
    indices = np.random.choice(range(N - W - H), n_samples, replace=False)
    
    all_sigma = np.zeros((n_samples, H))
    all_error = np.zeros((n_samples, H))
    
    for i, idx in enumerate(indices):
        s_win = raw_data[idx:idx+W, :cfg.N_STATE]
        a_win = raw_data[idx:idx+W, cfg.N_STATE:]
        x_t = torch.FloatTensor(np.concatenate([s_win, a_win], axis=1)).unsqueeze(0).to(device)
        future_a = torch.FloatTensor(raw_data[idx+W:idx+W+H, cfg.N_STATE:]).unsqueeze(0).to(device)
        mainT_true = raw_data[idx+W:idx+W+H, cfg.TARGET_IDX]
        
        s_traj, sigma_traj = model.rollout(x_t, future_a, mode=mode, return_stats=True)
        mainT_pred = s_traj[0, :, cfg.TARGET_IDX].detach().cpu().numpy()
        mainT_sigma = sigma_traj[0, :, cfg.TARGET_IDX].detach().cpu().numpy()
        
        all_sigma[i] = mainT_sigma
        all_error[i] = np.abs(mainT_pred - mainT_true)
    
    return all_sigma, all_error


def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"[{EXP_ID}] Probabilistic World Model — Gaussian NLL (K={ROLLOUT_K})")
    
    state_data, delta_actions, valve_abs = load_raw_data()
    raw_data = np.concatenate([state_data, delta_actions], axis=1)  # [T, 13]
    
    n_total = len(raw_data)
    n_train = int(n_total * 0.70)
    n_val_end = int(n_total * 0.85)
    
    train_data = raw_data[:n_train]
    val_data = raw_data[n_train:n_val_end]
    test_data = raw_data[n_val_end:]
    
    print(f"  数据: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    print(f"  N_STATE={cfg.N_STATE}, N_ACTION={cfg.N_ACTION}, TARGET_IDX={cfg.TARGET_IDX}")
    
    model = WorldModel(
        n_state=cfg.N_STATE, n_action=cfg.N_ACTION,
        window_size=cfg.WINDOW_SIZE,
        d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        n_var_layers=cfg.N_VAR_LAYERS, n_tcn_layers=cfg.N_TCN_LAYERS,
        dropout=cfg.DROPOUT, rollout_mode='sliding',
        probabilistic=True,
    ).to(device)
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = GaussNLLLoss()
    
    best_val_mae = float('inf')
    patience_counter = 0
    history = {'train_nll': [], 'train_mse': [], 'val_mae0': [], 'val_mae4': []}
    
    print(f"  训练中...")
    t0 = time.time()
    
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_mse = train_epoch(model, train_data, device, optimizer, criterion)
        val_mae0, val_mae4 = validate(model, val_data, device)
        
        history['train_nll'].append(train_loss)
        history['train_mse'].append(train_mse)
        history['val_mae0'].append(val_mae0)
        history['val_mae4'].append(val_mae4)
        
        scheduler.step(val_mae4)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | NLL: {train_loss:.4f} | MSE: {train_mse:.4f} | Val MAE(0): {val_mae0:.4f} | Val MAE(4): {val_mae4:.4f}")
        
        if val_mae4 < best_val_mae - 0.001:
            best_val_mae = val_mae4
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_mae4': val_mae4, 'val_mae0': val_mae0,
            }, os.path.join(EXP_DIR, "checkpoints", "best_model.pth"))
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"  早停于 epoch {epoch}")
            break
    
    train_time = time.time() - t0
    
    # 加载最佳模型 → 完整评测
    ckpt = torch.load(os.path.join(EXP_DIR, "checkpoints", "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"\n  训练完成: {train_time/60:.1f}min | Best epoch: {ckpt['epoch']}")
    print(f"\n  完整18步Rollout评测 (500条轨迹)...")
    
    # Sliding rollout
    sliding_mae, sliding_rmse = evaluate_full_rollout(model, test_data, device, mode='sliding', n_samples=500)
    
    # GRU rollout
    gru_mae, gru_rmse = evaluate_full_rollout(model, test_data, device, mode='gru', n_samples=500)
    
    # 不确定性校准
    sigma_sliding, error_sliding = evaluate_uncertainty(model, test_data, device, mode='sliding')
    
    # 输出对比
    print(f"\n{'='*70}")
    print(f"  {EXP_ID}: Probabilistic World Model — Gaussian NLL")
    print(f"{'='*70}")
    print(f"  {'Step':>5} {'Time':>6} {'Sliding MAE':>11} {'GRU MAE':>11} {'σ_sliding':>10}")
    print(f"  {'-'*55}")
    
    for s in [0, 1, 2, 4, 7, 10, 14, 17]:
        avg_sigma = sigma_sliding[:, s].mean()
        print(f"  {s:>5} {(s+1)*10:>5}s {sliding_mae[s]:>11.4f} {gru_mae[s]:>11.4f} {avg_sigma:>10.4f}")
    
    print(f"\n  Sliding MAE 增长: {sliding_mae[0]:.4f} → {sliding_mae[-1]:.4f} (×{sliding_mae[-1]/sliding_mae[0]:.2f})")
    print(f"  GRU MAE 增长:     {gru_mae[0]:.4f} → {gru_mae[-1]:.4f} (×{gru_mae[-1]/gru_mae[0]:.2f})")
    
    # σ 校准分析
    mean_error = error_sliding.mean(axis=0)
    mean_sigma = sigma_sliding.mean(axis=0)
    cal_ratio = mean_error / (mean_sigma + 1e-8)
    print(f"\n  σ 校准 (σ vs |error|):")
    for s in [0, 4, 10, 17]:
        print(f"    Step {s:2d}: |err|={mean_error[s]:.4f}, σ={mean_sigma[s]:.4f}, ratio={cal_ratio[s]:.2f} (理想=1.0)")
    print(f"    平均 ratio: {cal_ratio.mean():.2f}")
    
    # vs exp_004
    old_sliding = [0.1941, 0.2250, 0.2583, 0.3276, 0.5251, 0.7488, 1.0255, 1.0124]
    old_steps = [0, 1, 2, 4, 7, 10, 14, 17]
    imp_sliding = []
    for s, m in zip(old_steps, old_sliding):
        imp_sliding.append((m - sliding_mae[s]) / m * 100)
    
    print(f"\n  vs exp_004 Sliding (MSE):")
    for s, imp in zip(old_steps, imp_sliding):
        print(f"    Step {s:2d}: {imp:+5.1f}%")
    print(f"    平均: {np.mean(imp_sliding):+.1f}%")
    
    # 保存
    results = {
        **EXP_META,
        'train_time_min': train_time / 60,
        'best_epoch': ckpt['epoch'],
        'sliding_mae': sliding_mae.tolist(),
        'sliding_rmse': sliding_rmse.tolist(),
        'gru_mae': gru_mae.tolist(),
        'gru_rmse': gru_rmse.tolist(),
        'sigma_cal_ratio': cal_ratio.tolist(),
        'mean_sigma_per_step': mean_sigma.tolist(),
        'mean_error_per_step': mean_error.tolist(),
        'improvement_vs_exp004': {str(s): imp for s, imp in zip(old_steps, imp_sliding)},
        'avg_improvement': float(np.mean(imp_sliding)),
        'history': history,
    }
    with open(os.path.join(EXP_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  结果已保存: {EXP_DIR}/results.json")


if __name__ == '__main__':
    main()
