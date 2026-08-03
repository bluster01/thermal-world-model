#!/usr/bin/env python3
"""
exp_069_ensemble_train.py — P2: 训练 M7 ensemble (H_OUT=18, 多seed)
=====================================================================
成员1 = exp_048 的 h18.pth (seed 42), 新训 seed 7/13 (同架构同数据同超参只差seed)
用法: python exp_069_ensemble_train.py [--smoke] [SEED_LIST='7,13']
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
from src import config as cfg
sys.argv = _argv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMOKE = '--smoke' in sys.argv
SEEDS = [int(s) for s in os.environ.get('SEED_LIST', '7,13').split(',')]
EPOCH_CAP = 1 if SMOKE else cfg.EPOCHS
OUT_DIR = 'results/exp_069_ensemble'
os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)

def train_one(seed):
    E.H_OUT = 18
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = E.build_model('M7').to(DEVICE)
    prob = model.probabilistic
    crit = E.BetaNLLLoss(beta=E.BETA)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    best_m, pc, be = float('inf'), 0, 0
    t0 = time.time()
    for ep in range(1, EPOCH_CAP + 1):
        crit.beta = E.BETA
        nll = E.train_epoch(model, E.train_raw, opt, crit, prob)
        v0, v4 = E.validate(model, E.val_raw, prob)
        sched.step(v4)
        if ep % 10 == 0 or ep == 1:
            print(f"  seed={seed} E{ep:3d} | NLL:{nll:7.0f} | V0:{v0:.4f} | V4:{v4:.4f} | {time.time()-t0:.0f}s", flush=True)
        if v4 < best_m - 0.001:
            best_m, be, pc = v4, ep, 0
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict()},
                       f'{OUT_DIR}/checkpoints/seed{seed}.pth')
        else:
            pc += 1
        if pc >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"  seed={seed} Stop@{ep} best@{be} ({time.time()-t0:.0f}s)", flush=True)
            break
    return be

if __name__ == '__main__':
    for seed in SEEDS:
        if os.path.exists(f'{OUT_DIR}/checkpoints/seed{seed}.pth'):
            print(f"seed={seed} 已存在, 跳过")
            continue
        print(f"=== 训练 seed={seed} ===", flush=True)
        be = train_one(seed)
    print(f"完成. checkpoints: {OUT_DIR}/checkpoints/")
