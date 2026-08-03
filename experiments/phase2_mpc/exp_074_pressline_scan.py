#!/usr/bin/env python3
"""
exp_074_pressline_scan.py — 压线控制扫描 (2026-08-03 用户要求)
================================================================
目标: 主汽温"贴线运行" — 减少震荡的同时提高平均气温 (压线到 SP 附近),
      阀门动作留安全余度 (不顶到上限).

三个维度 (全部默认=主协议原行为, 不破坏 exp_064 定稿):
  1. ASYMM_RATIO — 非对称温度代价 (超温重罚/欠温轻罚)
  2. LAMBDA_U / U_LO,U_HI — 阀位安全裕度惩罚 (接近安全带边界时罚)
  3. (对照) 对称基线 ASYMM=1, LAMBDA_U=0

协议: 主协议 (H=18 + ovl05_hard5 + DIST_AMP=0.3 + M_STEP=6 + 每步SP基准)
指标: RMSE / IAE / ITAE / 超温积分 / 平均温度 / 平均偏差 mean_err / TV / jump / viol / valve_margin
      平均温度↑ + TV↓ + 超温积分↓ + valve_margin↑ = 压线控制达成

用法:
  python exp_074_pressline_scan.py --smoke            # 2 轨迹快速验证链路
  python exp_074_pressline_scan.py                    # 全量 10 轨迹 × 配置
  python exp_074_pressline_scan.py --asym 3 --lam-u 0.1 --u-hi 90 --n-tracks 50   # 单配置定稿
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_027_dwm_mpc as M  # 复用主协议模块 (模块级常量可覆盖)
M.MODEL_ID = 'M7'  # ⚠️ exp_027 模块级读 sys.argv[1] (import 副作用): 强制主模型, 防 --smoke 被当 MODEL_ID

# ── 主协议基线 (与 exp_064 定稿一致) ──
M.H_PLAN = 18
M.FIX_MODE = 'overlap'
M.LAMBDA3 = 0.05
M.HARD_DELTA = 5.0
M.DIST_AMP = 0.3
M.SP_TRAJ = 1
M.BENCH_SP_EACH = True

CFG_LIST = [
    # (tag, W_OVER, W_UNDER, LAMBDA_U, U_LO, U_HI) — 安全带按数据 p5/p95 校准 (valve0 [0.2,43.6], valve1 [-0.75,32.6])
    ('base_sym',      1.0, 1.0, 0.00, [2.0, 0.0], [43.0, 32.0]),   # 对称基线 = 主协议
    ('under3',        1.0, 3.0, 0.00, [2.0, 0.0], [43.0, 32.0]),   # 欠温重罚×3 (压线)
    ('under5',        1.0, 5.0, 0.00, [2.0, 0.0], [43.0, 32.0]),   # 欠温重罚×5
    ('u_margin',      1.0, 1.0, 0.10, [2.0, 0.0], [43.0, 32.0]),   # 阀位裕度
    ('under3_u',      1.0, 3.0, 0.10, [2.0, 0.0], [43.0, 32.0]),   # 组合
    ('under3_u_tight',1.0, 3.0, 0.10, [5.0, 3.0], [40.0, 29.0]),   # 更紧安全带 (更强裕度)
]


def run_cfg(tag, w_over, w_under, lam_u, u_lo, u_hi, n_tracks, smoke):
    M.W_OVER = w_over
    M.W_UNDER = w_under
    M.LAMBDA_U = lam_u
    M.U_LO = np.asarray(u_lo, dtype=float)
    M.U_HI = np.asarray(u_hi, dtype=float)
    wm = M.load_wm()
    np.random.seed(42)
    starts = np.random.choice(range(len(M.test_raw) - M.W - M.H_OUT - 120),
                              n_tracks, replace=False)
    all_m = []
    t0 = time.time()
    for k, s in enumerate(starts):
        mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, 'grad')
        all_m.append(M.metrics(mpc_t, pid_t, tset, mpc_a, pid_a))
        if k % 5 == 0 or k == n_tracks - 1:
            m = all_m[-1]
            print(f"  [{tag} {k+1}/{n_tracks}] RMSE {m['rmse_mpc']:.3f}/{m['rmse_pid']:.3f} "
                  f"T̄ {m['mean_temp_mpc']:.2f} 超温 {m['overtemp_mpc']:.0f}s "
                  f"TV {m['act_tv_mpc']:.3f} margin {m['valve_margin_mpc']:.1f}", flush=True)
    agg = {}
    for kk in all_m[0]:
        agg[kk] = float(np.mean([m[kk] for m in all_m]))
    out = {'tag': tag, 'w_over': w_over, 'w_under': w_under, 'lam_u': lam_u,
           'u_lo': u_lo, 'u_hi': u_hi, 'n_tracks': n_tracks, 'agg': agg, 'per_track': all_m}
    os.makedirs('results/exp_074_pressline', exist_ok=True)
    fn = f"results/exp_074_pressline/{tag}.json"
    json.dump(out, open(fn, 'w'), indent=2, default=float)
    print(f"  [{tag}] saved {fn} ({(time.time()-t0)/60:.1f}min) "
          f"| RMSE {agg['rmse_mpc']:.3f} T̄ {agg['mean_temp_mpc']:.2f} "
          f"mean_err {agg['mean_err_mpc']:+.3f} 超温积分 {agg['overtemp_int_mpc']:.2f} "
          f"TV {agg['act_tv_mpc']:.3f} margin {agg['valve_margin_mpc']:.1f}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n-tracks', type=int, default=10)
    ap.add_argument('--w-over', type=float, default=None)  # 单配置模式
    ap.add_argument('--w-under', type=float, default=None)
    ap.add_argument('--lam-u', type=float, default=None)
    ap.add_argument('--u-lo', type=float, default=None)
    ap.add_argument('--u-hi', type=float, default=None)
    ap.add_argument('--tag', type=str, default='custom')
    args = ap.parse_args()

    n_tracks = 2 if args.smoke else args.n_tracks
    if args.w_over is not None or args.w_under is not None:  # 单配置
        cfg = [(args.tag, args.w_over or 1.0, args.w_under or 1.0, args.lam_u or 0.0,
                [args.u_lo] if args.u_lo is not None else [2.0, 0.0],
                [args.u_hi] if args.u_hi is not None else [43.0, 32.0])]
    else:
        cfg = CFG_LIST if not args.smoke else CFG_LIST[:3]  # 冒烟只跑 3 个代表配置
    print(f"exp_074 压线扫描 | smoke={args.smoke} | n_tracks={n_tracks} | cfg={len(cfg)}")
    t0 = time.time()
    rows = []
    for tag, w_over, w_under, lam_u, u_lo, u_hi in cfg:
        agg = run_cfg(tag, w_over, w_under, lam_u, u_lo, u_hi, n_tracks, args.smoke)
        rows.append((tag, w_over, w_under, lam_u, u_lo, u_hi, agg))
    # 汇总表
    print("\n===== 压线扫描汇总 =====")
    print(f"{'cfg':>16} | {'RMSE':>6} {'T̄mpc':>6} {'T̄pid':>6} {'mean_err':>8} "
          f"{'超温积分':>7} {'超温s':>5} {'TV':>5} {'viol':>4} {'margin':>6}")
    for tag, w_over, w_under, lam_u, u_lo, u_hi, a in rows:
        print(f"{tag:>16} | {a['rmse_mpc']:>6.3f} {a['mean_temp_mpc']:>6.2f} "
              f"{a['mean_temp_pid']:>6.2f} {a['mean_err_mpc']:>+8.3f} "
              f"{a['overtemp_int_mpc']:>7.2f} {a['overtemp_mpc']:>5.0f} "
              f"{a['act_tv_mpc']:>5.3f} "
              f"{a['viol_mpc']:>4.0f} {a['valve_margin_mpc']:>6.1f}")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f}min")


if __name__ == '__main__':
    main()
