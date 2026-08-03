#!/usr/bin/env python3
"""
exp_073_policy_eval.py — P3: 策略闭环评测 (M7 当仿真器)
=========================================================
同协议: 150轨迹×3起点×双协议(扰动/无扰动), 每步SP基准, 与MPC评测公平对比
策略每步: 状态(窗口末40维观测,z-score) → π → 动作(反归一化+clip) → M7预测1步 → 扰动 → 窗口推进
用法: python exp_073_policy_eval.py --method iql [--smoke]
"""
import os, sys, json, argparse, time
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv
from experiments.phase2_mpc.exp_072_rl_train import Policy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/exp_073_policy'
os.makedirs(OUT, exist_ok=True)

def load_policy(method, seed):
    ck = torch.load(f'results/exp_072_rl/checkpoints/{method}_seed{seed}.pth',
                    map_location=DEVICE, weights_only=False)
    pi = Policy().to(DEVICE)
    pi.load_state_dict(ck['pi']); pi.eval()
    am = torch.FloatTensor(ck['am']).to(DEVICE)
    astd = torch.FloatTensor(ck['astd']).to(DEVICE)
    sm = torch.FloatTensor(ck['sm']).squeeze().to(DEVICE)
    ss = torch.FloatTensor(ck['ss']).squeeze().to(DEVICE)
    return pi, am, astd, sm, ss

def roll_policy(wm, pi, am, astd, sm, ss, track_idx, dist_amp, n_steps=120):
    """策略闭环: 返回 (temp, tset, actions)"""
    i = track_idx
    win = torch.FloatTensor(M.test_raw[i:i+M.W]).unsqueeze(0).to(DEVICE)
    rng = np.random.RandomState(42 + i) if dist_amp > 0 else None
    d_state = 0.0
    temps, tsets, acts = [], [], []
    amin, amax = M.test_raw[:, M.VALVE_IDX].min(0), M.test_raw[:, M.VALVE_IDX].max(0)
    amin_t = torch.FloatTensor(amin).to(DEVICE); amax_t = torch.FloatTensor(amax).to(DEVICE)
    for t in range(n_steps):
        gi_j = i + t
        s = (win[0, -1, :40] - sm) / ss
        a_norm = pi(s.unsqueeze(0)).squeeze(0)            # tanh [-1,1]
        a_raw = a_norm * astd + am
        a_raw = a_raw.clamp(amin_t, amax_t)
        a_full = a_raw.unsqueeze(0).repeat(M.H_OUT, 1).unsqueeze(0)
        with torch.no_grad():
            mu, _ = wm(win, a_full.reshape(1, -1))
        y_j = mu[0, 0].item()
        if rng is not None:
            d_state = 0.9 * d_state + rng.normal(0, dist_amp)
            y_j += d_state
        next_row = torch.FloatTensor(M.test_raw[gi_j+M.W]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        next_row[0, 0, M.TARGET_IDX] = y_j
        win = torch.cat([win[:, 1:, :], next_row], 1)
        temps.append(y_j)
        tsets.append(float(M.test_raw[gi_j+M.W, M.SP_IDX]))
        acts.append(a_raw.detach().cpu().numpy())
    return np.array(temps), np.array(tsets), np.array(acts)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['iql', 'td3bc'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    SMOKE = args.smoke
    M.SP_TRAJ = 0; M.BENCH_SP_EACH = True
    wm = M.load_wm()
    pi, am, astd, sm, ss = load_policy(args.method, args.seed)
    N_TRACKS = 2 if SMOKE else 50
    SEEDS = [42] if SMOKE else [42, 7, 13]
    t0 = time.time()
    for scene, da in [('dist', 0.3), ('nodist', 0.0)]:
        rows = []
        for seed in SEEDS:
            np.random.seed(seed)
            starts = np.random.choice(range(len(M.test_raw) - M.W - M.H_OUT - 120),
                                      N_TRACKS, replace=False)
            for s in starts:
                temps, tsets, acts = roll_policy(wm, pi, am, astd, sm, ss, int(s), da)
                e = temps - tsets
                rm = float(np.sqrt(np.mean(e ** 2)))
                iae = float(np.trapz(np.abs(e)))
                itae = float(np.trapz(np.arange(len(e)) * np.abs(e)))
                tv = float(np.abs(np.diff(acts, axis=0)).sum()) / 2.0
                rows.append({'rmse': rm, 'iae': iae, 'itae': itae, 'tv': tv,
                             'overtemp': int((temps > 575).sum())})
        json.dump(rows, open(f'{OUT}/{args.method}_seed{args.seed}_{scene}.json', 'w'), indent=2)
        a = {k: float(np.mean([r[k] for r in rows])) for k in ['rmse', 'iae', 'itae', 'tv']}
        a['overtemp'] = float(np.sum([r['overtemp'] for r in rows]))
        print(f"=== {args.method} seed{args.seed} {scene} (n={len(rows)}) ===")
        print(f"  {args.method}: RMSE {a['rmse']:.3f} | IAE {a['iae']:.1f} | ITAE {a['itae']:.0f} "
              f"| TV {a['tv']:.3f} | 超温 {a['overtemp']:.0f}s")
    print(f"===== 完成 ({(time.time()-t0)/60:.1f}min) =====")
