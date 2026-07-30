"""
data_loader.py — 世界模型数据加载器
从 Exp-0 CSV 数据构造 (x_t, s_{t+1}) 对
x_t = [s_{t-W:t}, a_{t-W:t}]
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg


class WorldModelDataset(Dataset):
    """
    世界模型数据集
    - 输入: x_t = concat(s_{t-W:t}, a_{t-W:t})  shape: [W, N_state + N_action]
    - 输出: s_{t+1}  shape: [N_state]
    
    动作定义: a_t = [valve1_t, valve2_t] (绝对值, 非差分)
    """
    
    def __init__(self, csv_path, split='train', val_ratio=0.15, test_ratio=0.15):
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # 确保列存在
        for col in cfg.FEATURE_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"列 '{col}' 不在CSV中。可用列: {list(df.columns)}")
        
        data = df[cfg.FEATURE_COLUMNS].values.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        
        # 时间序列切分 (不打乱顺序)
        n_total = len(data)
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
        
        self.data = data[start:end]
        self.n_samples = len(self.data) - cfg.WINDOW_SIZE
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 状态窗口: s_{t-W:t}
        s_window = self.data[idx:idx + cfg.WINDOW_SIZE, :]  # [W, N_state]
        
        # 动作窗口: a_{t-W:t} (阀位绝对值)
        a_window = self.data[idx:idx + cfg.WINDOW_SIZE, cfg.ACTION_INDICES]  # [W, N_action]
        
        # 输入: 拼接 x_t
        x_t = np.concatenate([s_window, a_window], axis=1)  # [W, N_state + N_action]
        
        # 目标: s_{t+1}
        s_next = self.data[idx + cfg.WINDOW_SIZE, :]  # [N_state]
        
        return torch.FloatTensor(x_t), torch.FloatTensor(s_next)


def get_dataloaders(batch_size=None, num_workers=0):
    """返回 train/val/test DataLoader"""
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    
    csv_path = os.path.join(cfg.DATA_DIR, cfg.TRAIN_FILE)
    
    train_ds = WorldModelDataset(csv_path, split='train')
    val_ds = WorldModelDataset(csv_path, split='val')
    test_ds = WorldModelDataset(csv_path, split='test')
    
    print(f"数据加载: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 快速测试
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    x, s_next = next(iter(train_loader))
    print(f"x shape: {x.shape}")       # [B, W, N_state+N_action]
    print(f"s_next shape: {s_next.shape}")  # [B, N_state]
    print(f"x range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"s_next range: [{s_next.min():.3f}, {s_next.max():.3f}]")
