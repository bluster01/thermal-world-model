#!/usr/bin/env python3
"""
exp_050_mstep_dist_scan.py — 扰动世界 M_STEP × 拼接模式 扫描
================================================================
回答两个问题:
  1. 扰动世界 (DIST_AMP=0.3, 主协议) 下 M_STEP 最优在哪? (旧扫描在无扰动协议下, 空白)
  2. 滚动优化指令拼接 (warm-start 模式) 是否是抖动根源?
     - off  : 常量重启 (a_init=None, 只有 LAMBDA2 锚定 a_last)
     - on   : 现状 — 整段旧计划直接作新规划初值 (exp_027 默认)
     - shift: 标准 receding-horizon — 丢弃已执行 M_STEP 段, 尾部补齐
  附加统计: 每次重规划边界跳变 |a_new[0] − a_last| (块间不连续的直接度量),
           及其占总动作变差 TV 的比例 → 定量回答"拼接是否是抖动根源"。

协议: 与主协议一致 — grad, α=0.5, SP_TRAJ=0, DIST_AMP=0.3, 10 条轨迹 (seed 42)
      H_PLAN=18 (使 M_STEP=12/18 为真实多步执行, 旧扫描 H=10 下 12/18 被截断为 10 步)
      + 锚点配置 M_STEP=6/H_PLAN=10 (精确复现主协议)
用法: python exp_050_mstep_dist_scan.py [--smoke]
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv

SMOKE = '--smoke' in sys.argv
M.SP_TRAJ = 0          # 标量目标 (真实 SP, 主协议)
M.DIST_AMP = 0.3       # 扰动世界 (主协议)
M.ALPHA = 0.5
N_TRACKS = 2 if SMOKE else 10
OUT_DIR = 'results/exp_050_mstep_dist'
os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)

DEVICE = M.DEVICE
if __name__ == '__main__':
    wm = M.load_wm()

    # ---- 拼接模式: 包装 plan_grad (simulate 内部逻辑不动, 保持同一代码路径) ----
    _real_plan_grad = M.plan_grad
    WS_MODE = 'on'          # off | on | shift
    JUMPS = []              # 每 track 的重规划边界跳变 (排除首块: a_init 为 None 时不算)

    def plan_grad_wrap(wm_, x_hist, t_set, a_last, a_init, sp_fut=None):
        global WS_MODE
        if WS_MODE == 'off':
            a_init_use = None
        elif WS_MODE == 'shift':
            if a_init is not None and len(a_init) > M.M_STEP:
                tail = a_init[M.M_STEP:]
                tail = torch.cat([tail, a_init[-1:].repeat(M.M_STEP, 1)], 0)
                a_init_use = tail[:len(a_init)]
            else:
                a_init_use = a_init
        else:
            a_init_use = a_init
        a_plan, Js = _real_plan_grad(wm_, x_hist, t_set, a_last, a_init_use, sp_fut)
        if a_init is not None:   # 非首块 → 重规划边界
            JUMPS.append(float((a_plan[0] - a_last).abs().mean().item()))
        return a_plan, Js

    M.plan_grad = plan_grad_wrap

    def run_config(m_step, h_plan, ws):
        """跑 N_TRACKS 条轨迹, 返回聚合指标"""
        global WS_MODE, JUMPS
        WS_MODE = ws
        M.M_STEP = m_step
        M.H_PLAN = h_plan
        N = len(M.test_raw)
        np.random.seed(42)
        starts = np.random.choice(range(N - M.W - M.H_OUT - 120), N_TRACKS, replace=False)
        rows = []
        for k, s in enumerate(starts):
            JUMPS = []
            mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, 'grad')
            m = M.metrics(mpc_t, pid_t, tset, mpc_a, pid_a)
            jumps = np.array(JUMPS)
            tv = float(np.abs(np.diff(mpc_a, axis=0)).mean())  # 与 act_tv_mpc 同口径
            row = {
                'rmse_mpc': m['rmse_mpc'], 'rmse_pid': m['rmse_pid'],
                'std_mpc': m['temp_std_mpc'], 'tv_mpc': m['act_tv_mpc'],
                'tv_pid': m['act_tv_pid'], 'viol_mpc': m['viol_mpc'],
                'boundary_jump_mean': float(jumps.mean()) if len(jumps) else float('nan'),
                'boundary_jump_share': float(jumps.sum() / (tv * (len(mpc_a) - 1)))
                                       if len(jumps) and len(mpc_a) > 1 else float('nan'),
                'n_blocks': len(jumps) + 1,
            }
            rows.append(row)
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        return agg, rows

    CONFIGS = []
    for m_step in [6, 12, 18]:
        for ws in ['off', 'on', 'shift']:
            CONFIGS.append((m_step, 18, ws))
    CONFIGS.append((6, 10, 'on'))   # 锚点: 精确主协议

    results = {}
    t0 = time.time()
    print(f"{'M_STEP':>6} {'H_PLAN':>7} {'ws':>6} | {'RMSE_m':>7} {'RMSE_p':>7} {'std_m':>6} "
          f"{'TV_m':>6} {'TV_p':>6} {'jump':>6} {'jump%TV':>7} {'nblk':>5}")
    for m_step, h_plan, ws in CONFIGS:
        tc = time.time()
        agg, rows = run_config(m_step, h_plan, ws)
        results[f'm{m_step}_h{h_plan}_{ws}'] = {'agg': agg, 'per_track': rows}
        print(f"{m_step:>6} {h_plan:>7} {ws:>6} | {agg['rmse_mpc']:>7.3f} {agg['rmse_pid']:>7.3f} "
              f"{agg['std_mpc']:>6.3f} {agg['tv_mpc']:>6.2f} {agg['tv_pid']:>6.2f} "
              f"{agg['boundary_jump_mean']:>6.3f} {agg['boundary_jump_share']*100:>6.1f}% "
              f"{agg['n_blocks']:>5.0f}  ({(time.time()-tc)/60:.1f}min)")
        json.dump({'agg': agg, 'per_track': rows},
                  open(f"{OUT_DIR}/m{m_step}_h{h_plan}_{ws}.json", 'w'), indent=2)

    print(f"\n===== 扰动世界 M_STEP×拼接 扫描完成 ({(time.time()-t0)/60:.1f}min) =====")
    print(f"Saved: {OUT_DIR}/")
