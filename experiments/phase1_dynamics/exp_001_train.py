"""
exp_001_train.py — 实验 1a: 世界模型一步预测训练
==================================================
训练动作条件化世界模型，验证 s_t, a_t → s_{t+1} 的一步动力学拟合能力

实验编号: exp_001
日期: 2026-07-30
目标: 一步全状态预测精度 (14变量逐变量RMSE/MAE)
"""
import os, sys, json, time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import config as cfg
from data_loader import get_dataloaders
from world_model import WorldModel


# ===== 实验记录 =====
EXP_ID = "exp_001"
EXP_DIR = os.path.join("results", EXP_ID)
os.makedirs(os.path.join(EXP_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(EXP_DIR, "logs"), exist_ok=True)

EXP_META = {
    "exp_id": EXP_ID,
    "name": "世界模型一步预测训练",
    "date": datetime.now().isoformat(),
    "phase": "Phase 1 / Experiment 1a",
    "model": "WorldModel (sliding window + action-conditioned)",
    "objective": "一步全状态预测 (s_t, a_t → s_{t+1})",
    "config": {k: v for k, v in vars(cfg).items() if not k.startswith('__') and not callable(v)},
}

with open(os.path.join(EXP_DIR, "meta.json"), 'w') as f:
    json.dump(EXP_META, f, indent=2, default=str)


# ===== 训练函数 =====
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
def evaluate(model, loader, device):
    """评估全状态预测: 逐变量 MAE + 主汽温 MAE"""
    model.eval()
    total_mae = torch.zeros(cfg.N_STATE, device=device)
    n_samples = 0
    
    for x, s_next in loader:
        x, s_next = x.to(device), s_next.to(device)
        s_pred = model(x)
        abs_err = torch.abs(s_pred - s_next)
        total_mae += abs_err.sum(dim=0)
        n_samples += x.size(0)
    
    per_var_mae = (total_mae / n_samples).cpu().numpy()
    return {
        'mae_per_var': per_var_mae.tolist(),
        'mae_main_temp': float(per_var_mae[cfg.TARGET_IDX]),
        'mae_avg': float(per_var_mae.mean()),
    }


def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"[{EXP_ID}] 设备: {device}")
    
    # 数据
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # 模型
    model = WorldModel(
        n_state=cfg.N_STATE, n_action=cfg.N_ACTION,
        window_size=cfg.WINDOW_SIZE,
        d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        n_var_layers=cfg.N_VAR_LAYERS, n_tcn_layers=cfg.N_TCN_LAYERS,
        patch_len=cfg.PATCH_LEN, stride=cfg.STRIDE, dropout=cfg.DROPOUT,
    ).to(device)
    print(f"[{EXP_ID}] 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练配置
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 训练循环
    best_val_mae = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_mae': [], 'val_mae_main_temp': []}
    
    print(f"[{EXP_ID}] 开始训练...")
    t0 = time.time()
    
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_metrics['mae_avg'])
        history['val_mae_main_temp'].append(val_metrics['mae_main_temp'])
        
        scheduler.step(val_metrics['mae_main_temp'])
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                  f"Val MAE(avg): {val_metrics['mae_avg']:.4f} | "
                  f"Val MAE(mainT): {val_metrics['mae_main_temp']:.4f}")
        
        # 早停
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
    
    # 测试集评估
    print(f"[{EXP_ID}] 加载最佳模型, 评估测试集...")
    ckpt = torch.load(os.path.join(EXP_DIR, "checkpoints", "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"  实验 {EXP_ID} 完成")
    print(f"{'='*60}")
    print(f"  训练时间: {train_time/60:.1f} min")
    print(f"  最佳 Epoch: {ckpt['epoch']}")
    print(f"  Test MAE (avg):    {test_metrics['mae_avg']:.4f}")
    print(f"  Test MAE (mainT):  {test_metrics['mae_main_temp']:.4f}")
    print(f"\n  逐变量 MAE:")
    for i, (name, mae) in enumerate(zip(cfg.FEATURE_COLUMNS, test_metrics['mae_per_var'])):
        marker = " ← TARGET" if i == cfg.TARGET_IDX else ""
        print(f"    [{i:2d}] {name:20s}: {mae:.4f}{marker}")
    
    # 保存结果
    results = {
        **EXP_META,
        'train_time_min': train_time / 60,
        'best_epoch': ckpt['epoch'],
        'best_val_mae_main_temp': best_val_mae,
        'test_metrics': test_metrics,
        'history': history,
    }
    with open(os.path.join(EXP_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  结果已保存: {EXP_DIR}/results.json")


if __name__ == '__main__':
    main()
