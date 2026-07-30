"""
exp_002_train.py — 实验 1a 对比: GRU vs Sliding rollout
========================================================
训练 WorldModel v2，对比两种 rollout 模式的一步预测精度和展开精度

实验编号: exp_002
日期: 2026-07-30
目标: (1) 一步全状态预测精度 (2) GRU vs Sliding rollout 对比
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


# ===== 实验记录 =====
EXP_ID = "exp_002"
EXP_DIR = os.path.join("results", EXP_ID)
os.makedirs(os.path.join(EXP_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(EXP_DIR, "logs"), exist_ok=True)

EXP_META = {
    "exp_id": EXP_ID,
    "name": "WorldModel v2 一步预测 + GRU/Sliding rollout 对比",
    "date": datetime.now().isoformat(),
    "phase": "Phase 1 / Experiment 1a",
    "model": "WorldModel v2 (Encoder + GRU Cell decoder)",
    "note": "对比 GRU rollout 和 Sliding rollout 的展开精度与速度",
    "config": {k: v for k, v in vars(cfg).items() if not k.startswith('__') and not callable(v)},
}

with open(os.path.join(EXP_DIR, "meta.json"), 'w') as f:
    json.dump(EXP_META, f, indent=2, default=str)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, s_next in loader:
        x, s_next = x.to(device), s_next.to(device)
        optimizer.zero_grad()
        s_pred = model(x)
        loss = criterion(s_pred, s_next)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_one_step(model, loader, device):
    """一步预测评估"""
    model.eval()
    total_mae = torch.zeros(cfg.N_STATE, device=device)
    n_samples = 0
    for x, s_next in loader:
        x, s_next = x.to(device), s_next.to(device)
        s_pred = model(x)
        total_mae += torch.abs(s_pred - s_next).sum(dim=0)
        n_samples += x.size(0)
    per_var_mae = (total_mae / n_samples).cpu().numpy()
    return {
        'mae_per_var': per_var_mae.tolist(),
        'mae_main_temp': float(per_var_mae[cfg.TARGET_IDX]),
        'mae_avg': float(per_var_mae.mean()),
    }


@torch.no_grad()
def evaluate_rollout(model, loader, device, max_batches=200):
    """自回归展开评估: 对比 GRU 和 Sliding 模式"""
    model.eval()
    results = {'gru': [], 'sliding': []}
    
    for batch_idx, (x_hist, s_next) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        
        x_hist = x_hist.to(device)
        B = x_hist.size(0)
        
        # 用真实动作序列做展开
        a_seq = x_hist[:, -cfg.PRED_LEN:, cfg.N_STATE:]  # 取窗口最后 PRED_LEN 步的动作
        # 注意: 这里用历史动作做"展开", 不是MPC规划的动作
        # 真实使用场景中 a_seq 来自 MPC 求解器
        
        for mode in ['gru', 'sliding']:
            s_traj = model.rollout(x_hist, a_seq, mode=mode)  # [B, H, N_state]
            
            # 计算每步的 MAE
            # 注意: rollout预测的是未来, 但我们没有真实未来的 (s_next只是一步)
            # 这里简化为: 对比 rollout 的第1步预测 vs 一步预测
            step1_pred = s_traj[:, 0, :]  # [B, N_state]
            mae_main_temp = float(torch.abs(step1_pred[:, cfg.TARGET_IDX] - s_next[:, cfg.TARGET_IDX]).mean())
            results[mode].append(mae_main_temp)
    
    return {
        'gru_mae_main_temp_mean': float(np.mean(results['gru'])),
        'sliding_mae_main_temp_mean': float(np.mean(results['sliding'])),
        'gru_mae_main_temp_std': float(np.std(results['gru'])),
        'sliding_mae_main_temp_std': float(np.std(results['sliding'])),
    }


def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"[{EXP_ID}] 设备: {device}")
    print(f"[{EXP_ID}] 模型类型: WorldModel v2 (GRU Cell decoder)")
    
    train_loader, val_loader, test_loader = get_dataloaders()
    
    model = WorldModel(
        n_state=cfg.N_STATE, n_action=cfg.N_ACTION,
        window_size=cfg.WINDOW_SIZE,
        d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        n_var_layers=cfg.N_VAR_LAYERS, n_tcn_layers=cfg.N_TCN_LAYERS,
        patch_len=cfg.PATCH_LEN, stride=cfg.STRIDE, dropout=cfg.DROPOUT,
        rollout_mode='gru',
    ).to(device)
    print(f"[{EXP_ID}] 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_mae = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_mae': [], 'val_mae_main_temp': []}
    
    print(f"[{EXP_ID}] 开始训练...")
    t0 = time.time()
    
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_one_step(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_metrics['mae_avg'])
        history['val_mae_main_temp'].append(val_metrics['mae_main_temp'])
        
        scheduler.step(val_metrics['mae_main_temp'])
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                  f"Val MAE(avg): {val_metrics['mae_avg']:.4f} | "
                  f"Val MAE(mainT): {val_metrics['mae_main_temp']:.4f}")
        
        if val_metrics['mae_main_temp'] < best_val_mae - 0.001:
            best_val_mae = val_metrics['mae_main_temp']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae_main_temp': val_metrics['mae_main_temp'],
                'meta': EXP_META,
            }, os.path.join(EXP_DIR, "checkpoints", "best_model.pth"))
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"  早停于 epoch {epoch}")
            break
    
    train_time = time.time() - t0
    print(f"[{EXP_ID}] 训练完成, 耗时 {train_time/60:.1f} min")
    print(f"[{EXP_ID}] Best Val MAE (mainT): {best_val_mae:.4f}")
    
    # 加载最佳模型
    ckpt = torch.load(os.path.join(EXP_DIR, "checkpoints", "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # 测试集评估
    print(f"\n[{EXP_ID}] 测试集评估...")
    test_metrics = evaluate_one_step(model, test_loader, device)
    
    # rollout 对比较少量样本
    print(f"[{EXP_ID}] Rollout 对比 (GRU vs Sliding)...")
    t_start = time.time()
    rollout_metrics_gru = evaluate_rollout(model, test_loader, device, max_batches=100)
    t_gru = time.time() - t_start
    rollout_metrics_gru['time_ms_per_batch'] = t_gru / 100 * 1000
    
    t_start = time.time()
    rollout_metrics_sliding = evaluate_rollout(model, test_loader, device, max_batches=100)
    t_sliding = time.time() - t_start
    rollout_metrics_sliding['time_ms_per_batch'] = t_sliding / 100 * 1000
    
    # 输出
    print(f"\n{'='*60}")
    print(f"  实验 {EXP_ID} 完成")
    print(f"{'='*60}")
    print(f"  训练时间: {train_time/60:.1f} min | 最佳 Epoch: {ckpt['epoch']}")
    print(f"\n  一步预测:")
    print(f"    Test MAE (avg):   {test_metrics['mae_avg']:.4f}")
    print(f"    Test MAE (mainT): {test_metrics['mae_main_temp']:.4f}")
    print(f"\n  Rollout 对比:")
    print(f"    GRU     mainT MAE: {rollout_metrics_gru['gru_mae_main_temp_mean']:.4f} ± {rollout_metrics_gru['gru_mae_main_temp_std']:.4f}")
    print(f"    Sliding mainT MAE: {rollout_metrics_sliding['sliding_mae_main_temp_mean']:.4f} ± {rollout_metrics_sliding['sliding_mae_main_temp_std']:.4f}")
    print(f"    GRU 速度:     {rollout_metrics_gru['time_ms_per_batch']:.0f}ms/batch")
    print(f"    Sliding 速度: {rollout_metrics_sliding['time_ms_per_batch']:.0f}ms/batch")
    
    print(f"\n  逐变量一步 MAE:")
    for i, (name, mae) in enumerate(zip(cfg.FEATURE_COLUMNS, test_metrics['mae_per_var'])):
        marker = " ← TARGET" if i == cfg.TARGET_IDX else ""
        print(f"    [{i:2d}] {name:20s}: {mae:.4f}{marker}")
    
    # 保存
    results = {
        **EXP_META,
        'train_time_min': train_time / 60,
        'best_epoch': ckpt['epoch'],
        'best_val_mae_main_temp': best_val_mae,
        'test_one_step': test_metrics,
        'rollout_gru': rollout_metrics_gru,
        'rollout_sliding': rollout_metrics_sliding,
        'history': history,
    }
    with open(os.path.join(EXP_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  结果已保存: {EXP_DIR}/results.json")


if __name__ == '__main__':
    main()
