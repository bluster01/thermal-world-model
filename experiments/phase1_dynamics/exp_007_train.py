"""
exp_007_train.py — 端到端 μ 优化 (MSE, 无σ)
===========================================
改动 vs exp_005/006:
- 回退: probabilistic=False, 纯 MSE 损失
- 保留: NLL训练带来的架构改进 (WorldModel v3)
- 目标: 扔掉σ负担, 只看μ能到多少

实验编号: exp_007
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

EXP_ID = "exp_007"
EXP_DIR = os.path.join("results", EXP_ID)
os.makedirs(os.path.join(EXP_DIR, "checkpoints"), exist_ok=True)

ROLLOUT_K = 5
ROLLOUT_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2]

EXP_META = {
    "exp_id": EXP_ID,
    "name": "End-to-end μ optimization — MSE only, no σ head",
    "date": datetime.now().isoformat(),
    "changes_vs_exp004": [
        "WorldModel v3 architecture (probabilistic=False → pure MSE decoder)",
        "回退: 扔掉σ双头, 纯MSE端到端优化μ",
        "对比: 验证NLL训练带来的μ收益是否来自架构而非损失函数",
    ],
    "rollout_k": ROLLOUT_K,
    "rollout_weights": ROLLOUT_WEIGHTS,
    "loss": "mse",
}

with open(os.path.join(EXP_DIR, "meta.json"), 'w') as f:
    json.dump(EXP_META, f, indent=2, default=str)


def train_epoch(model, raw_data, device, optimizer, BS=256, steps=500):
    """多步 rollout + MSE loss"""
    model.train()
    W, K = cfg.WINDOW_SIZE, ROLLOUT_K
    N = len(raw_data)
    total_loss = 0.0
    
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
            s_pred = model(x_t)  # deterministic: [B, N_STATE]
            step_loss = ROLLOUT_WEIGHTS[k] * F.mse_loss(s_pred, s_true_t[:, k, :])
            total_step_loss += step_loss
            
            states = torch.cat([states[:, 1:, :], s_pred.unsqueeze(1).detach()], dim=1)
            actions = torch.cat([actions[:, 1:, :], a_seq_t[:, k:k+1, :]], dim=1)
        
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


def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"[{EXP_ID}] End-to-end μ — MSE only (K={ROLLOUT_K})")
    
    state_data, delta_actions, valve_abs = load_raw_data()
    raw_data = np.concatenate([state_data, delta_actions], axis=1)
    
    n_total = len(raw_data)
    n_train = int(n_total * 0.70)
    n_val_end = int(n_total * 0.85)
    
    train_data = raw_data[:n_train]
    val_data = raw_data[n_train:n_val_end]
    test_data = raw_data[n_val_end:]
    
    print(f"  数据: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    model = WorldModel(
        n_state=cfg.N_STATE, n_action=cfg.N_ACTION,
        window_size=cfg.WINDOW_SIZE,
        d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        n_var_layers=cfg.N_VAR_LAYERS, n_tcn_layers=cfg.N_TCN_LAYERS,
        dropout=cfg.DROPOUT, rollout_mode='sliding',
        probabilistic=False,  # ← 纯确定性, 不输出σ
    ).to(device)
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_mae = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_mae0': [], 'val_mae4': []}
    
    print(f"  训练中...")
    t0 = time.time()
    
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = train_epoch(model, train_data, device, optimizer)
        val_mae0, val_mae4 = validate(model, val_data, device)
        
        history['train_loss'].append(train_loss)
        history['val_mae0'].append(val_mae0)
        history['val_mae4'].append(val_mae4)
        
        scheduler.step(val_mae4)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val MAE(0): {val_mae0:.4f} | Val MAE(4): {val_mae4:.4f}")
        
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
    
    ckpt = torch.load(os.path.join(EXP_DIR, "checkpoints", "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"\n  训练完成: {train_time/60:.1f}min | Best epoch: {ckpt['epoch']}")
    print(f"\n  完整18步Rollout评测 (500条轨迹)...")
    
    sliding_mae, sliding_rmse = evaluate_full_rollout(model, test_data, device, mode='sliding')
    gru_mae, gru_rmse = evaluate_full_rollout(model, test_data, device, mode='gru')
    
    print(f"\n{'='*65}")
    print(f"  {EXP_ID}: End-to-end μ — MSE only")
    print(f"{'='*65}")
    print(f"  {'Step':>5} {'Time':>6} {'Sliding MAE':>11} {'GRU MAE':>11}")
    print(f"  {'-'*42}")
    
    for s in [0, 1, 2, 4, 7, 10, 14, 17]:
        print(f"  {s:>5} {(s+1)*10:>5}s {sliding_mae[s]:>11.4f} {gru_mae[s]:>11.4f}")
    
    growth = sliding_mae[-1] / sliding_mae[0]
    print(f"\n  Sliding MAE 增长: {sliding_mae[0]:.4f} → {sliding_mae[-1]:.4f} (×{growth:.2f})")
    
    # vs exp_004 (baseline MSE)
    old_004 = [0.1941, 0.2250, 0.2583, 0.3276, 0.5251, 0.7488, 1.0255, 1.0124]
    old_steps = [0, 1, 2, 4, 7, 10, 14, 17]
    imp_004 = [(old_004[i] - sliding_mae[s]) / old_004[i] * 100 for i, s in enumerate(old_steps)]
    print(f"\n  vs exp_004 (baseline MSE, v2 arch):")
    for s, imp in zip(old_steps, imp_004):
        print(f"    Step {s:2d}: {imp:+5.1f}%")
    print(f"    平均: {np.mean(imp_004):+.1f}%")
    
    # vs exp_005 (best μ from NLL)
    old_005 = [0.0772, 0.1098, 0.1603, 0.2512, 0.3960, 0.5582, 0.7857, 0.9528]
    imp_005 = [(old_005[i] - sliding_mae[s]) / old_005[i] * 100 for i, s in enumerate(old_steps)]
    print(f"\n  vs exp_005 (GaussNLL):")
    for s, imp in zip(old_steps, imp_005):
        print(f"    Step {s:2d}: {imp:+5.1f}%")
    print(f"    平均: {np.mean(imp_005):+.1f}%")
    
    # vs exp_006 (β-NLL)
    old_006 = [0.0834, 0.1144, 0.1615, 0.2475, 0.3747, 0.5058, 0.6788, 0.8082]
    imp_006 = [(old_006[i] - sliding_mae[s]) / old_006[i] * 100 for i, s in enumerate(old_steps)]
    print(f"\n  vs exp_006 (β-NLL):")
    for s, imp in zip(old_steps, imp_006):
        print(f"    Step {s:2d}: {imp:+5.1f}%")
    print(f"    平均: {np.mean(imp_006):+.1f}%")
    
    results = {
        **EXP_META,
        'train_time_min': train_time / 60,
        'best_epoch': ckpt['epoch'],
        'sliding_mae': sliding_mae.tolist(),
        'sliding_rmse': sliding_rmse.tolist(),
        'gru_mae': gru_mae.tolist(),
        'gru_rmse': gru_rmse.tolist(),
        'improvement_vs_exp004': {str(s): imp for s, imp in zip(old_steps, imp_004)},
        'improvement_vs_exp005': {str(s): imp for s, imp in zip(old_steps, imp_005)},
        'improvement_vs_exp006': {str(s): imp for s, imp in zip(old_steps, imp_006)},
        'history': history,
    }
    with open(os.path.join(EXP_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  结果已保存: {EXP_DIR}/results.json")


if __name__ == '__main__':
    main()
