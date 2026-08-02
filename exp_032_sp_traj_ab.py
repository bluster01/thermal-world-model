#!/usr/bin/env python3
"""
exp_032_sp_traj_ab.py — 方案1验证: SP轨迹目标 vs 标量目标 (SP跳变事件对比)
==========================================================================
同一批 SP 跳变事件, 两种目标模式跑 MPC:
  A. 标量目标 (SP_TRAJ=0): J = Σwₜ(ŷₜ−SP_now)²  ← 当前协议 (滞后一拍)
  B. 轨迹目标 (SP_TRAJ=1): J = Σwₜ(ŷₜ−SP_future)²  ← 方案1 (前馈)
指标: 到达时间 / 窗口RMSE / 超调 / 动作提前量
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from exp_027_dwm_mpc import load_wm, simulate, W, H_OUT, DEVICE, test_raw, VALVE_IDX, SP_IDX, TARGET_IDX

wm = load_wm()
N = len(test_raw)
dsp = np.abs(np.diff(test_raw[:, SP_IDX]))
events = np.where(dsp > 2.0)[0]
events = [e for e in events if e > W + 100 and e < N - W - H_OUT - 180]
np.random.seed(42)
ev_sel = sorted(np.random.choice(events, min(20, len(events)), replace=False))
print(f"SP 跳变事件: {len(ev_sel)}")

def settle_time(temp, sp_after, tol=0.3):
    within = np.abs(temp - sp_after) < tol
    for t in range(len(within) - 10):
        if within[t:t+10].all():
            return t
    return len(within)

res = {0: [], 1: []}
for k, e in enumerate(ev_sel):
    i0 = e + 1 - W
    if i0 < 0: continue
    sp_after = test_raw[e+1, SP_IDX]
    pid_t = test_raw[e+1:e+1+180, TARGET_IDX]
    for mode in [0, 1]:
        import exp_027_dwm_mpc as M
        M.SP_TRAJ = mode
        mpc_t, _, _, mpc_a, _ = simulate(wm, i0, 'grad', n_steps=180)
        off = 0
        if off + 180 > len(mpc_t): break
        align = mpc_t[off:off+180]
        act = mpc_a[off:off+180]
        settle = settle_time(align, sp_after)
        rmse = np.sqrt(np.mean((align - sp_after)**2))
        os_val = np.max(align[:60]) - sp_after if sp_after > test_raw[e, SP_IDX] else sp_after - np.min(align[:60])
        # 动作提前量: SP突变后18步内二级阀累计动作
        act_ff = np.abs(act[1:19, 1] - act[0, 1]).sum()
        res[mode].append({'settle': settle, 'rmse': float(rmse), 'os': float(max(os_val, 0)), 'act_ff': float(act_ff)})
    print(f"  [{k+1}/{len(ev_sel)}] ΔSP={sp_after-test_raw[e,SP_IDX]:+.1f} | "
          f"标量: RMSE {res[0][-1]['rmse']:.3f} | 轨迹: RMSE {res[1][-1]['rmse']:.3f}")

print("\n===== 汇总 =====")
for mode, name in [(0, '标量目标 (滞后)'), (1, '轨迹目标 (前馈)')]:
    d = res[mode]
    if not d: continue
    print(f"{name}: settle={np.mean([x['settle'] for x in d]):.0f} 步 | "
          f"RMSE={np.mean([x['rmse'] for x in d]):.3f} | "
          f"超调={np.mean([x['os'] for x in d]):.3f} | "
          f"突变后动作量={np.mean([x['act_ff'] for x in d]):.2f}")
if res[0] and res[1]:
    print(f"\n轨迹 vs 标量: RMSE {(1-np.mean([x['rmse'] for x in res[1]])/np.mean([x['rmse'] for x in res[0]]))*100:+.1f}% | "
          f"settle {(1-np.mean([x['settle'] for x in res[1]])/np.mean([x['settle'] for x in res[0]]))*100:+.1f}% | "
          f"动作量 {(np.mean([x['act_ff'] for x in res[1]])/np.mean([x['act_ff'] for x in res[0]])-1)*100:+.1f}%")

json.dump({'scalar': res[0], 'traj': res[1]},
          open("results/exp_032_sp_traj_ab.json", 'w'), indent=2, default=float)
print("\nSaved: results/exp_032_sp_traj_ab.json")
