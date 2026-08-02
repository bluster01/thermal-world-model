#!/usr/bin/env python3
"""
exp_044_wm_fidelity.py — WM 保真度验证 (Phase 2.5 任务1)
==========================================================
目的: 证明 WM 虚拟世界有物理锚点 — 同动作序列下 WM 闭环预测 vs 真实温度,
      误差随 horizon 累积曲线

方法: 50 条轨迹, 真实阀位动作 + WM 闭环预测 (sim_pid_wm 逐步推进, 温度列=模型预测,
      其余列=真实) vs 真实温度。逐 horizon 计算 MAE/RMSE, 对比:
      - WM 单步预测误差 (eval_rollout avg 0.295°C)
      - 误差累积斜率 (亚线性=可信, 超线性=漂移)
      - 温度统计量保真 (均值/方差/超调)
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
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
from exp_027_dwm_mpc import load_wm, W, H_OUT, DEVICE, test_raw, VALVE_IDX, SP_IDX, TARGET_IDX
import exp_027_dwm_mpc as M
sys.argv = _argv

N_TRACKS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
N_STEPS = 120

wm = load_wm()

def sim_pid_wm(track_idx, n_steps):
    """真实阀位动作 + WM 闭环预测温度 (逐步推进, 其余列真实)"""
    win = torch.FloatTensor(test_raw[track_idx:track_idx+W]).unsqueeze(0).to(DEVICE)
    temps = []
    for k in range(n_steps):
        gi = track_idx + W + k
        a_real = torch.FloatTensor(test_raw[gi:gi+H_OUT, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, _ = wm(win, a_real)
            y1 = mu[0, 0].item()
        next_row = torch.FloatTensor(test_raw[gi]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        next_row[0, 0, TARGET_IDX] = y1
        win = torch.cat([win[:, 1:, :], next_row], 1)
        temps.append(y1)
    return np.array(temps)

np.random.seed(42)
N = len(test_raw)
starts = np.random.choice(range(N - W - H_OUT - N_STEPS - 50), N_TRACKS, replace=False)

# 逐 horizon 误差
HORIZONS = [10, 30, 60, 90, 120]
err_mae = {h: [] for h in HORIZONS}
err_rmse = {h: [] for h in HORIZONS}
stats = []
t0 = time.time()
for k, s in enumerate(starts):
    pred = sim_pid_wm(s, N_STEPS)
    real = test_raw[s+W:s+W+N_STEPS, TARGET_IDX]
    for h in HORIZONS:
        e = pred[:h] - real[:h]
        err_mae[h].append(np.abs(e).mean())
        err_rmse[h].append(np.sqrt(np.mean(e**2)))
    stats.append({'mae_full': float(np.abs(pred-real).mean()),
                  'rmse_full': float(np.sqrt(np.mean((pred-real)**2))),
                  'bias': float((pred-real).mean()),
                  'std_ratio': float(np.std(pred)/np.std(real)),
                  'range': float(np.percentile(pred,99)-np.percentile(pred,1))})
    if (k+1) % 10 == 0: print(f"  [{k+1}/{N_TRACKS}]")

print("\n===== WM 保真度验证 (真实动作序列 + WM 闭环预测 vs 真实温度) =====")
print(f"{'horizon':>8} {'MAE(°C)':>10} {'RMSE(°C)':>10} {'累积斜率':>10}")
prev = 0
for h in HORIZONS:
    mae = np.mean(err_mae[h]); rmse = np.mean(err_rmse[h])
    slope = mae / h * 10  # 每100步的MAE增幅
    print(f"{h*10:>6}s {mae:>10.3f} {rmse:>10.3f} {slope:>10.3f}")

print("\n温度统计量保真:")
print(f"  均值偏差: {np.mean([st['bias'] for st in stats]):+.3f}°C")
print(f"  std比 (pred/real): {np.mean([st['std_ratio'] for st in stats]):.3f}")
print(f"  单步预测误差 (WM eval_rollout): 0.295°C (对比基准)")
print(f"  120s MAE / 单步误差: {np.mean(err_mae[120])/0.295:.1f}x")

out = {'n': N_TRACKS, 'horizons': HORIZONS,
       'mae': {str(h): float(np.mean(err_mae[h])) for h in HORIZONS},
       'rmse': {str(h): float(np.mean(err_rmse[h])) for h in HORIZONS},
       'stats': stats}
os.makedirs("results/exp_044_fidelity", exist_ok=True)
json.dump(out, open("results/exp_044_fidelity/wm_fidelity.json", 'w'), indent=2, default=float)
print(f"\nSaved: results/exp_044_fidelity/wm_fidelity.json (耗时 {(time.time()-t0)/60:.1f}min)")
