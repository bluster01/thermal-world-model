#!/usr/bin/env python3
"""
exp_048_horizon_scan.py — WM 预测长度扫描 (找 pivot)
=====================================================
H ∈ {6,9,12,18} (60/90/120/180s @10s), 统一 20min(1200s) 自回归闭环 rollout 对比 MAE。
动机: H_OUT=18(180s) 可能不是最优预测长度 — 预测太短→自回归累积误差大,
     太长→direct 一步到位误差大, 中间应有 pivot。

评估 (每配置同一协议):
  1. direct eval_rollout (n=500) — 每步 MAE
  2. 自回归闭环 1200s (真实动作 + 温度预测回填, 其余列真实, exp_044 协议) 50 条轨迹
     → 1200s 总 MAE + 逐 horizon MAE (60/90/120/180/300/600/900/1200s)

训练: M7 协议 (β-NLL fixed -0.3, RevIN, seed 42, 早停 v4 统一)
用法: python exp_048_horizon_scan.py [--smoke]
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
import config as cfg
from experiments.phase1_dynamics import exp_025_unified_benchmark as E

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMOKE = '--smoke' in sys.argv
H_LIST = [6] if SMOKE else [6, 9, 12, 18]
N_TRACKS = 2 if SMOKE else 50
N_STEPS = 120          # 20min @10s
EPOCH_CAP = 3 if SMOKE else cfg.EPOCHS
OUT_DIR = "results/exp_048_horizon"
os.makedirs(f"{OUT_DIR}/checkpoints", exist_ok=True)


def train_wm(h):
    """训练 M7, 预测长度 H_OUT=h, 返回 (model, best_epoch)"""
    E.H_OUT = h
    np.random.seed(42); torch.manual_seed(42)
    model = E.build_model('M7').to(DEVICE)
    prob = model.probabilistic
    crit = E.BetaNLLLoss(beta=E.BETA)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
                            weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    best_m, pc, be = float('inf'), 0, 0
    for ep in range(1, EPOCH_CAP + 1):
        crit.beta = E.BETA  # β fixed -0.3
        nll = E.train_epoch(model, E.train_raw, opt, crit, prob)
        v0, v4 = E.validate(model, E.val_raw, prob)
        sched.step(v4)
        if ep % 10 == 0 or ep == 1:
            print(f"  H={h:>2} E{ep:3d} | NLL:{nll:7.0f} | V0:{v0:.4f} | V4:{v4:.4f}")
        if v4 < best_m - 0.001:
            best_m, be, pc = v4, ep, 0
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict()},
                       f"{OUT_DIR}/checkpoints/h{h:02d}.pth")
        else:
            pc += 1
        if pc >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"  H={h:>2} Stop@{ep} best@{be}")
            break
    ck = torch.load(f"{OUT_DIR}/checkpoints/h{h:02d}.pth",
                    map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    return model, be


def sim_rollout(wm, track_idx, h, n_steps):
    """自回归闭环: 真实动作 + 温度预测回填 (exp_044 协议)"""
    W = cfg.WINDOW_SIZE
    win = torch.FloatTensor(E.test_raw[track_idx:track_idx + W]).unsqueeze(0).to(DEVICE)
    temps = []
    for k in range(n_steps):
        gi = track_idx + W + k
        a_real = torch.FloatTensor(E.test_raw[gi:gi + h, E.VALVE_IDX]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, _ = wm(win, a_real)
            y1 = mu[0, 0].item()
        next_row = torch.FloatTensor(E.test_raw[gi]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        next_row[0, 0, E.TARGET_IDX] = y1
        win = torch.cat([win[:, 1:, :], next_row], 1)
        temps.append(y1)
    return np.array(temps)


results = {}
for h in H_LIST:
    t0 = time.time()
    print(f"\n===== H_OUT={h} ({h * 10}s 预测) =====")
    model, be = train_wm(h)
    # 1. direct 多步预测
    mae_direct = E.eval_rollout(model, E.test_raw, model.probabilistic, n=500)
    # 2. 自回归闭环 20min (1200s)
    np.random.seed(42)
    N = len(E.test_raw)
    starts = np.random.choice(range(N - cfg.WINDOW_SIZE - h - N_STEPS - 50),
                              N_TRACKS, replace=False)
    hz_steps = [6, 9, 12, 18, 30, 60, 90, 120]   # 60s..1200s
    hz_err = {s: [] for s in hz_steps}
    mae_full = []
    for s in starts:
        pred = sim_rollout(model, s, h, N_STEPS)
        real = E.test_raw[s + cfg.WINDOW_SIZE:s + cfg.WINDOW_SIZE + N_STEPS,
                          E.TARGET_IDX]
        mae_full.append(np.abs(pred - real).mean())
        for hz in hz_steps:
            hz_err[hz].append(np.abs(pred[:hz] - real[:hz]).mean())
    res = {
        'h_steps': h, 'horizon_s': h * 10, 'best_epoch': be,
        'direct_mae_degC': [float(x) for x in mae_direct],
        'direct_avg_degC': float(mae_direct.mean()),
        'direct_last_degC': float(mae_direct[-1]),
        'rollout_mae_1200s_degC': float(np.mean(mae_full)),
        'rollout_mae_by_horizon_s': {str(s * 10): float(np.mean(hz_err[s]))
                                     for s in hz_steps},
    }
    results[str(h)] = res
    print(f"  direct avg {res['direct_avg_degC']:.3f} | last {res['direct_last_degC']:.3f} | "
          f"rollout 1200s {res['rollout_mae_1200s_degC']:.3f}")
    print(f"  耗时 {(time.time() - t0) / 60:.1f}min")

json.dump(results, open(f"{OUT_DIR}/scan.json", 'w'), indent=2, default=float)

print("\n===== 汇总 =====")
print(f"{'H(s)':>6} {'direct avg':>10} {'direct last':>12} {'rollout1200s':>13} "
      f"{'60s':>6} {'120s':>6} {'300s':>6} {'600s':>6} {'1200s':>6}")
for h in H_LIST:
    r = results[str(h)]
    hz = r['rollout_mae_by_horizon_s']
    print(f"{r['horizon_s']:>5}s {r['direct_avg_degC']:>10.3f} {r['direct_last_degC']:>12.3f} "
          f"{r['rollout_mae_1200s_degC']:>13.3f} {hz['60']:>6.3f} {hz['120']:>6.3f} "
          f"{hz['300']:>6.3f} {hz['600']:>6.3f} {hz['1200']:>6.3f}")
print(f"\nSaved: {OUT_DIR}/scan.json")
