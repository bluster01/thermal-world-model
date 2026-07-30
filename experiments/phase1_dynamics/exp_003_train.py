"""
exp_003_train.py — 实验 1a: 多步 rollout loss 训练
==================================================
基于 Graph World Model 论文的多步展开训练范式:
  L = Σ_{k=1}^{K} w_k · MSE(ŝ_{t+k}, s_{t+k})

目的: 让模型学会 '展开友好' 的预测, 减少累积误差

实验编号: exp_003
日期: 2026-07-30
对比: exp_001 (一步loss) vs exp_003 (多步rollout loss)
"""
import os, sys, json, time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import get_dataloaders
from world_model import WorldModel


EXP_ID = "exp_003"
EXP_DIR = os.path.join("results", EXP_ID)
os.makedirs(os.path.join(EXP_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(EXP_DIR, "logs"), exist_ok=True)

# 多步 rollout 训练配置
ROLLOUT_K = 5                     # 展开步数
ROLLOUT_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2]  # 递减权重: 近步更重要

EXP_META = {
    "exp_id": EXP_ID,
    "name": "WorldModel v2 多步 rollout loss 训练",
    "date": datetime.now().isoformat(),
    "phase": "Phase 1 / Experiment 1a",
    "objective": f"minimize weighted {ROLLOUT_K}-step rollout MSE",
    "rollout_k": ROLLOUT_K,
    "rollout_weights": ROLLOUT_WEIGHTS,
    "config": {k: v for k, v in vars(cfg).items() if not k.startswith('__') and not callable(v)},
}

with open(os.path.join(EXP_DIR, "meta.json"), 'w') as f:
    json.dump(EXP_META, f, indent=2, default=str)


def train_one_epoch_rollout(model, loader, optimizer, device):
    """多步 rollout loss 训练"""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    for x_hist, s_next in loader:
        x_hist = x_hist.to(device)
        B = x_hist.size(0)
        
        # 取未来动作 (从窗口末尾取)
        future_a = x_hist[:, -ROLLOUT_K:, cfg.N_STATE:]  # [B, K, N_action]
        
        # 前向: encode + rollout K步
        optimizer.zero_grad()
        z_t = model.encode(x_hist)
        
        # GRU rollout with grad (训练时需要保留计算图)
        h = z_t
        rollout_loss = 0.0
        
        for k in range(ROLLOUT_K):
            s_next_pred, h = model.state_decoder_gru(
                z_t=None, a_t=future_a[:, k, :], h_prev=h
            )
            # 注意: s_next_pred 在归一化空间, 需要 denorm 再算 loss
            # 简化: 直接用归一化空间算 loss (因为 RevIN 是线性变换)
            # 或者: 在 rollout 前 denorm
            rollout_loss += ROLLOUT_WEIGHTS[k] * nn.functional.mse_loss(
                s_next_pred, s_next[:, k * cfg.N_STATE: (k+1) * cfg.N_STATE] 
                if k == 0 else s_next
            )
        
        # 简化版: 由于 s_next 只有一步真值, 我们分步计算
        
        rollout_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += rollout_loss.item() * B
        n_samples += B
    
    return total_loss / n_samples


def train_one_epoch_sliding_rollout(model, loader, optimizer, device):
    """
    多步 rollout loss 训练 (Sliding 模式)
    
    对每个样本:
      1. 编码历史窗口 → z_t
      2. 展开 K 步, 每步喂真实动作
      3. 用真实 s_{t+k} 计算 loss
    """
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    # 需要连续窗口数据 — 不能用 shuffled DataLoader
    # 改用 random sampling from raw data
    
    return 0.0  # placeholder


def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"[{EXP_ID}] 设备: {device}")
    print(f"[{EXP_ID}] 多步 rollout loss 训练 (K={ROLLOUT_K})")
    print(f"[{EXP_ID}] ⚠️ rollout loss 训练需要连续时间窗口, 使用原始数据采样")
    
    # 从原始CSV加载数据
    import pandas as pd
    csv_path = os.path.join(cfg.DATA_DIR, cfg.TRAIN_FILE)
    df = pd.read_csv(csv_path)
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
    raw_data = df[cfg.FEATURE_COLUMNS].values.astype(np.float32)
    raw_data = np.nan_to_num(raw_data, nan=0.0)
    
    # 时间序列切分
    n_total = len(raw_data)
    n_train = int(n_total * 0.70)
    
    train_data = raw_data[:n_train]
    val_data = raw_data[n_train:int(n_total * 0.85)]
    test_data = raw_data[int(n_total * 0.85):]
    
    print(f"  数据: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    W = cfg.WINDOW_SIZE
    K = ROLLOUT_K
    
    # 模型
    model = WorldModel(rollout_mode='sliding').to(device)  # 用滑动窗口模式
    print(f"[{EXP_ID}] 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_mae = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae_step0': [], 'val_mae_step4': []}
    
    BS = 256
    STEPS_PER_EPOCH = 500  # 每epoch采样500个batch
    
    print(f"[{EXP_ID}] 开始训练 (每epoch采样{STEPS_PER_EPOCH}个batch)...")
    t0 = time.time()
    
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        for step in range(STEPS_PER_EPOCH):
            # 随机采样连续窗口
            indices = np.random.randint(0, len(train_data) - W - K, size=BS)
            
            x_hist_batch = []
            a_seq_batch = []
            s_true_batch = []  # [BS, K, N_state]
            
            for idx in indices:
                s_win = train_data[idx:idx+W, :]
                a_win = train_data[idx:idx+W, cfg.ACTION_INDICES]
                x_hist = np.concatenate([s_win, a_win], axis=1)
                
                future_a = train_data[idx+W:idx+W+K, cfg.ACTION_INDICES]
                future_s = train_data[idx+W:idx+W+K, :]
                
                x_hist_batch.append(x_hist)
                a_seq_batch.append(future_a)
                s_true_batch.append(future_s)
            
            x_hist_t = torch.FloatTensor(np.stack(x_hist_batch)).to(device)  # [BS, W, 16]
            a_seq_t = torch.FloatTensor(np.stack(a_seq_batch)).to(device)    # [BS, K, 2]
            s_true_t = torch.FloatTensor(np.stack(s_true_batch)).to(device)  # [BS, K, 14]
            
            optimizer.zero_grad()
            
            # Sliding rollout K 步 (保留计算图)
            states = x_hist_t[:, :, :cfg.N_STATE]
            actions = x_hist_t[:, :, cfg.N_STATE:]
            total_step_loss = 0.0
            
            for k in range(K):
                x_t = torch.cat([states, actions], dim=2)
                s_pred = model.forward(x_t)  # [BS, N_state], 保留梯度
                
                # 第k步 loss
                step_loss = nn.functional.mse_loss(s_pred, s_true_t[:, k, :])
                total_step_loss += ROLLOUT_WEIGHTS[k] * step_loss
                
                # 滑动窗口
                states = torch.cat([states[:, 1:, :], s_pred.unsqueeze(1).detach()], dim=1)
                actions = torch.cat([actions[:, 1:, :], a_seq_t[:, k:k+1, :]], dim=1)
            
            total_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += total_step_loss.item()
        
        avg_train_loss = epoch_loss / STEPS_PER_EPOCH
        history['train_loss'].append(avg_train_loss)
        
        # 验证: 随机采样100个val样本
        model.eval()
        val_loss = 0.0
        val_mae_step0 = 0.0
        val_mae_step4 = 0.0
        n_val = 100
        
        val_indices = np.random.randint(0, len(val_data) - W - K, size=n_val)
        
        with torch.no_grad():
            for idx in val_indices:
                s_win = val_data[idx:idx+W, :]
                a_win = val_data[idx:idx+W, cfg.ACTION_INDICES]
                x_hist_np = np.concatenate([s_win, a_win], axis=1)
                x_hist_t = torch.FloatTensor(x_hist_np).unsqueeze(0).to(device)
                
                future_a = val_data[idx+W:idx+W+K, cfg.ACTION_INDICES]
                future_s = val_data[idx+W:idx+W+K, :]
                a_seq_t = torch.FloatTensor(future_a).unsqueeze(0).to(device)
                s_true_t = torch.FloatTensor(future_s).unsqueeze(0).to(device)
                
                s_traj = model.rollout(x_hist_t, a_seq_t, mode='sliding')  # [1, K, 14]
                
                err0 = torch.abs(s_traj[0, 0, cfg.TARGET_IDX] - s_true_t[0, 0, cfg.TARGET_IDX])
                err4 = torch.abs(s_traj[0, min(4, K-1), cfg.TARGET_IDX] - s_true_t[0, min(4, K-1), cfg.TARGET_IDX])
                
                val_mae_step0 += err0.item()
                val_mae_step4 += err4.item()
                val_loss += nn.functional.mse_loss(s_traj, s_true_t).item()
        
        val_mae_step0 /= n_val
        val_mae_step4 /= n_val
        val_loss /= n_val
        
        history['val_loss'].append(val_loss)
        history['val_mae_step0'].append(val_mae_step0)
        history['val_mae_step4'].append(val_mae_step4)
        
        scheduler.step(val_mae_step4)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | "
                  f"Val MAE(step0): {val_mae_step0:.4f} | Val MAE(step4): {val_mae_step4:.4f}")
        
        if val_mae_step4 < best_val_mae - 0.001:
            best_val_mae = val_mae_step4
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae_step4': val_mae_step4,
                'meta': EXP_META,
            }, os.path.join(EXP_DIR, "checkpoints", "best_model.pth"))
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"  早停于 epoch {epoch}")
            break
    
    train_time = time.time() - t0
    print(f"[{EXP_ID}] 训练完成, 耗时 {train_time/60:.1f} min")
    
    # 测试: 完整18步 rollout 评测
    print(f"\n[{EXP_ID}] 完整18步 rollout 评测...")
    ckpt = torch.load(os.path.join(EXP_DIR, "checkpoints", "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    H = cfg.ROLLOUT_LEN
    N_TEST = 500
    test_start_indices = np.random.choice(
        range(len(test_data) - W - H), N_TEST, replace=False
    )
    
    errors = np.zeros((N_TEST, H))
    
    with torch.no_grad():
        for i, idx in enumerate(test_start_indices):
            s_win = test_data[idx:idx+W, :]
            a_win = test_data[idx:idx+W, cfg.ACTION_INDICES]
            x_hist_np = np.concatenate([s_win, a_win], axis=1)
            x_hist_t = torch.FloatTensor(x_hist_np).unsqueeze(0).to(device)
            
            future_a = test_data[idx+W:idx+W+H, cfg.ACTION_INDICES]
            a_seq_t = torch.FloatTensor(future_a).unsqueeze(0).to(device)
            
            mainT_true = test_data[idx+W:idx+W+H, cfg.TARGET_IDX]
            
            s_traj = model.rollout(x_hist_t, a_seq_t, mode='sliding')
            mainT_pred = s_traj[0, :, cfg.TARGET_IDX].cpu().numpy()
            
            errors[i] = np.abs(mainT_pred - mainT_true)
    
    step_mae = errors.mean(axis=0)
    step_rmse = np.sqrt((errors ** 2).mean(axis=0))
    
    print(f"\n{'='*60}")
    print(f"  实验 {EXP_ID} 完成 — Rollout Loss 训练")
    print(f"{'='*60}")
    print(f"  训练时间: {train_time/60:.1f} min | 最佳 Epoch: {ckpt['epoch']}")
    print(f"\n  Rollout MAE (主汽温, {N_TEST}条轨迹):")
    print(f"  {'Step':>5} {'Time':>6} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*30}")
    for s in [0, 1, 2, 4, 7, 10, 14, 17]:
        print(f"  {s:>5} {(s+1)*10:>5}s {step_mae[s]:>8.4f} {step_rmse[s]:>8.4f}")
    print(f"\n  MAE 增长: {step_mae[0]:.4f} → {step_mae[-1]:.4f} (×{step_mae[-1]/step_mae[0]:.2f})")
    
    # 与 exp_001 对比
    exp_001_mae = [0.1183, 0.2376, 0.3624, 0.6084, 0.9118, 1.1211, 1.3194, 1.4258]
    exp_001_steps = [0, 1, 2, 4, 7, 10, 14, 17]
    improvement = []
    for s, mae_001 in zip(exp_001_steps, exp_001_mae):
        imp = (mae_001 - step_mae[s]) / mae_001 * 100
        improvement.append(imp)
    avg_imp = np.mean(improvement)
    
    print(f"\n  vs exp_001 (一步loss训练) 改进:")
    for s, imp in zip(exp_001_steps, improvement):
        arrow = "↑" if imp > 0 else "↓"
        print(f"    Step {s:2d}: {abs(imp):5.1f}% {arrow}")
    print(f"    平均改进: {avg_imp:+.1f}%")
    
    results = {
        **EXP_META,
        'train_time_min': train_time / 60,
        'best_epoch': ckpt['epoch'],
        'step_mae': step_mae.tolist(),
        'step_rmse': step_rmse.tolist(),
        'improvement_vs_exp001': {str(s): imp for s, imp in zip(exp_001_steps, improvement)},
        'avg_improvement': avg_imp,
        'history': history,
    }
    with open(os.path.join(EXP_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  结果已保存: {EXP_DIR}/results.json")


if __name__ == '__main__':
    main()
