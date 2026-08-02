#!/usr/bin/env python3
"""
exp_031_sp_event_eval.py — SP 跳变事件评测: M7 隐式前馈验证
============================================================
提取 test 集 SP 跳变事件, 对比 PID(真实) vs M7-MPC(闭环) 的:
1. 响应延迟: SP 跳变后温度到达 (新SP±0.3°C) 的时间
2. 超调: 到达过程中的最大过冲
3. 跳变窗口内 RMSE
验证假设: MPC 从历史窗口看到扰动先兆 → 提前动作 → 更快到达新 SP
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from exp_027_dwm_mpc import load_wm, simulate, W, H_OUT, DEVICE, test_raw, VALVE_IDX, SP_IDX, TARGET_IDX

wm = load_wm()  # M7
N = len(test_raw)

# SP 跳变事件
dsp = np.abs(np.diff(test_raw[:, SP_IDX]))
events = np.where(dsp > 2.0)[0]
events = [e for e in events if e > W + 50 and e < N - W - H_OUT - 180]
print(f"test 集 SP 跳变事件: {len(events)}")

# 用与主跑相同的 seed 选取迹 (避免重复计算, 只取前 20 个事件)
np.random.seed(42)
ev_sel = sorted(np.random.choice(events, min(20, len(events)), replace=False))

def settle_time(temp, sp_after, tol=0.3):
    """到达 |T−SP_new|<tol 且保持 10 步的最早时间"""
    within = np.abs(temp - sp_after) < tol
    for t in range(len(within) - 10):
        if within[t:t+10].all():
            return t
    return len(within)  # 未到达

results = []
for k, e in enumerate(ev_sel):
    i0 = e - 20  # 跳变前 20 步开始仿真 (窗口需含跳变前历史)
    sp_before, sp_after = test_raw[e, SP_IDX], test_raw[e+1, SP_IDX]
    # PID 真实轨迹 (跳变后 180 步)
    pid_t = test_raw[e+1:e+1+180, TARGET_IDX]
    # MPC 闭环 (从跳变前开始, 覆盖跳变)
    mpc_t, _, _, _, _ = simulate(wm, i0, 'grad', n_steps=180)
    # 对齐: simulate 从 i0 起, MPC 温度对应 i0+W+t; 跳变发生在 e+1 → 偏移 off = e+1-(i0+W)
    off = e + 1 - (i0 + W)
    if off < 0 or off + 180 > len(mpc_t): continue
    mpc_align = mpc_t[off:off+180]
    # 指标
    pid_settle = settle_time(pid_t, sp_after)
    mpc_settle = settle_time(mpc_align, sp_after)
    pid_rmse = np.sqrt(np.mean((pid_t - sp_after)**2))
    mpc_rmse = np.sqrt(np.mean((mpc_align - sp_after)**2))
    pid_os = np.max(pid_t[:60]) - sp_after if sp_after > sp_before else sp_after - np.min(pid_t[:60])
    mpc_os = np.max(mpc_align[:60]) - sp_after if sp_after > sp_before else sp_after - np.min(mpc_align[:60])
    results.append({'event': int(e), 'dsp': float(sp_after - sp_before),
                    'pid_settle': int(pid_settle), 'mpc_settle': int(mpc_settle),
                    'pid_rmse': float(pid_rmse), 'mpc_rmse': float(mpc_rmse),
                    'pid_os': float(max(pid_os, 0)), 'mpc_os': float(max(mpc_os, 0))})
    print(f"  [{k+1}/{len(ev_sel)}] ΔSP={sp_after-sp_before:+.1f}°C | settle: PID {pid_settle} vs MPC {mpc_settle} 步 | RMSE: {pid_rmse:.3f}/{mpc_rmse:.3f}")

print("\n===== 汇总 =====")
agg = {'n': len(results)}
for k in ['pid_settle', 'mpc_settle', 'pid_rmse', 'mpc_rmse', 'pid_os', 'mpc_os']:
    agg[k] = float(np.mean([r[k] for r in results]))
settle_improv = (1 - agg['mpc_settle'] / agg['pid_settle']) * 100
rmse_improv = (1 - agg['mpc_rmse'] / agg['pid_rmse']) * 100
os_improv = (1 - agg['mpc_os'] / agg['pid_os']) * 100
print(f"  到达时间: PID {agg['pid_settle']:.0f} vs MPC {agg['mpc_settle']:.0f} 步 ({settle_improv:+.1f}%)")
print(f"  窗口RMSE: PID {agg['pid_rmse']:.3f} vs MPC {agg['mpc_rmse']:.3f} ({rmse_improv:+.1f}%)")
print(f"  超调: PID {agg['pid_os']:.3f} vs MPC {agg['mpc_os']:.3f} ({os_improv:+.1f}%)")

json.dump({'agg': agg, 'per_event': results},
          open("results/exp_031_sp_events.json", 'w'), indent=2, default=float)
print("\nSaved: results/exp_031_sp_events.json")
