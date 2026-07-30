"""
data_loader.py — 世界模型数据加载器 v2
改动:
- 状态: 12维纯物理状态 (去掉了设定值)
- 动作: 差分阀位 (Δv1, Δv2) 而非绝对值
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg


class WorldModelDataset(Dataset):
    """一步预测数据集: (x_t, s_{t+1})"""
    def __init__(self, csv_path, split='train', val_ratio=0.15, test_ratio=0.15):
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # 加载12维状态
        state_data = df[cfg.FEATURE_COLUMNS].values.astype(np.float32)
        state_data = np.nan_to_num(state_data, nan=0.0)
        
        # 加载2维阀位绝对值 → 计算差分动作
        valve_abs = df[cfg.VALVE_ABS_COLS].values.astype(np.float32)  # [T, 2]
        valve_abs = np.nan_to_num(valve_abs, nan=0.0)
        
        # 差分: a_t = v_t - v_{t-1}, a_0 = 0
        delta_actions = np.zeros_like(valve_abs)
        delta_actions[1:] = valve_abs[1:] - valve_abs[:-1]  # [T, 2]
        
        # 合并: x = [state, delta_action]
        self.data = np.concatenate([state_data, delta_actions], axis=1)  # [T, 14]
        self.n_total = cfg.N_STATE + cfg.N_ACTION  # 14
        
        # 切分
        n_total = len(self.data)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_val - n_test
        
        if split == 'train':
            start, end = 0, n_train
        elif split == 'val':
            start, end = n_train, n_train + n_val
        elif split == 'test':
            start, end = n_train + n_val, n_total
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.data = self.data[start:end]
        self.n_samples = len(self.data) - cfg.WINDOW_SIZE
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # x_t = [s_{t-W:t} ‖ Δv_{t-W:t}] → 14维 (12状态 + 2差分动作)
        x_t = self.data[idx:idx + cfg.WINDOW_SIZE, :]  # [W, 14]
        
        # s_{t+1} = 12维纯状态
        s_next = self.data[idx + cfg.WINDOW_SIZE, :cfg.N_STATE]  # [N_STATE]
        
        return torch.FloatTensor(x_t), torch.FloatTensor(s_next)


def load_raw_data(csv_path=None):
    """加载完整原始数据 (状态12维 + 差分动作2维)"""
    if csv_path is None:
        csv_path = os.path.join(cfg.DATA_DIR, cfg.TRAIN_FILE)
    
    df = pd.read_csv(csv_path)
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
    
    state_data = df[cfg.FEATURE_COLUMNS].values.astype(np.float32)
    state_data = np.nan_to_num(state_data, nan=0.0)
    
    valve_abs = df[cfg.VALVE_ABS_COLS].values.astype(np.float32)
    valve_abs = np.nan_to_num(valve_abs, nan=0.0)
    
    delta_actions = np.zeros_like(valve_abs)
    delta_actions[1:] = valve_abs[1:] - valve_abs[:-1]
    
    return state_data, delta_actions, valve_abs


def get_dataloaders(batch_size=None, num_workers=0):
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    
    csv_path = os.path.join(cfg.DATA_DIR, cfg.TRAIN_FILE)
    
    train_ds = WorldModelDataset(csv_path, split='train')
    val_ds = WorldModelDataset(csv_path, split='val')
    test_ds = WorldModelDataset(csv_path, split='test')
    
    print(f"数据加载 (clean state + delta action): train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    x, s_next = next(iter(train_loader))
    print(f"x shape: {x.shape}  (B, W, N_state+N_action=14)")
    print(f"s_next shape: {s_next.shape}  (B, N_state=12)")
    print(f"\n特征:")
    for i, name in enumerate(cfg.FEATURE_COLUMNS):
        print(f"  [{i:2d}] {name}")
    print(f"\n动作 (差分阀位):")
    print(f"  [12] Δ一级阀位")
    print(f"  [13] Δ二级阀位")
    print(f"\nx range: [{x[:,:,cfg.N_STATE:].min():.3f}, {x[:,:,cfg.N_STATE:].max():.3f}] (Δ% per step)")
