#!/usr/bin/env python3
"""
exp_101_realpid_vs_mpc.py — 真实增量式PID vs DWM-MPC 闭环对比 (路线A 小实验)
================================================================================
背景: E4 曾因缺真实控制器 baseline 被 BLOCK。2026-08-09 已从 BMCS2.ppp 解析并
      验证现场主调 (POU #107, 伊敏二减A侧) 的增量式 PI 参数 (强驱动段 OLS slope≈1.07),
      副调标定增益 K=2.0%/°C (exp_092 大信号辨识)。现在用真实参数跑公平闭环对比。

协议: 复用 eval_protocol P0-A/B/C 公平协议 (第三方评测世界 + 物理侧扰动 + 共用执行器)
  世界: M0/M8/M9 集成 (与控制器不相交)
  扰动: physical 模式 (负荷/煤量/流量通道), sigma=0.3, rho=0.9 — 与 S1 主表一致
  SP  : 真实 SP 序列 (test 段逐点)
  执行器: Actuator (一阶惯性+速率限幅+行程限幅, 三臂共用)

臂:
  pid_real  : 真实增量式 PI (POU#107) + 副调比例 K_sub=2.0%/°C → 阀位指令
              Δmid(k) = −(Kp(E,load)·ΔE + Kp/Ti(E,load)·E·Δt)
              Kp = FX44(E)×FX49(load), Ti = FX45(E)×FX50(load)   [已验证]
  pid_legacy: eval_protocol.PIDController (kp=40, ki=8) — 旧主表虚拟 PID 对照
  mpc_m7    : DWM-MPC (M7 规划, CostConfig 'd' 综合代价, H=10, M_STEP=6)

判定: mpc_m7 不劣于 pid_real (配对 RMSE / TV / 超温积分) → 路线A 在真实参数下成立;
      mpc 输 → 旧结论 (MPC>PID) 是虚拟 PID 参数造成的假象, 需修正论文叙事。

用法: python exp_101_realpid_vs_mpc.py [--smoke]
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.phase2_mpc.eval_protocol import (  # noqa: E402
    CostConfig, Disturbance, MPCController, PIDController, WorldSim, load_wm,
    run_episode, metrics, DEVICE, DT, TARGET_IDX, SP_IDX, T_MIN, T_MAX)

# ══════════════════════════════════════════════════════════════════
# 真实增量式 PI (POU #107) — 参数来自 pid_parameter_traces.md, 已验证 slope≈1
# ══════════════════════════════════════════════════════════════════
FX44_X = [-12, -10, -8, -5, -3, 3, 5, 8, 10, 12]
FX44_Y = [0.6, 0.6, 0.8, 1.0, 1.2, 1.2, 1.0, 0.8, 0.6, 0.6]
FX45_X = [-12, -10, -8, -5, -3, 3, 5, 8, 10, 12]
FX45_Y = [800, 650, 550, 450, 350, 350, 450, 550, 650, 800]
FX49_X = [150, 200, 300, 400, 500, 600]
FX49_Y = [1.0, 1.0, 1.0, 1.0, 0.8, 0.5]
FX50_X = [150, 200, 300, 400, 500, 600]
FX50_Y = [1.0, 1.0, 1.0, 1.0, 1.2, 1.6]
K_SUB = 2.0          # 副调比例增益 (%/°C, exp_092 大信号标定)
LOAD_IDX = 0         # NUMERIC_COLS: 机组负荷


def fx(xs, ys, x):
    return float(np.interp(np.clip(x, xs[0], xs[-1]), xs, ys))


class RealPIDController:
    """真实增量式 PI 主调 + 副调比例, 输出二级阀位指令。

    与 pid_repro 已验证公式完全一致 (事件域 slope≈1.068):
        Δmid(k) = −(Kp(k)·ΔE(k) + Kp(k)/Ti(k)·E(k)·Δt)
        Δv(k)   = K_SUB · Δmid(k)          (副调秒级跟随 → 比例近似)
        v(k)    = v(k−1) + Δv(k)
    """
    name = 'pid_real'

    def __init__(self, u_lo=0.0, u_hi=45.0, k_sub=K_SUB):
        self.u_lo, self.u_hi = u_lo, u_hi
        self.k_sub = k_sub
        self.reset()

    def reset(self):
        self.e_prev = None
        self.v_cmd = None

    def act(self, win, sp_now, sp_fut, a_last, **_):
        y = float(win[0, -1, TARGET_IDX])
        load = float(win[0, -1, LOAD_IDX])
        E = y - float(sp_now)
        Kp = fx(FX44_X, FX44_Y, E) * fx(FX49_X, FX49_Y, load)
        Ti = max(fx(FX45_X, FX45_Y, E) * fx(FX50_X, FX50_Y, load), 1.0)
        if self.e_prev is None:
            self.e_prev = E
        dE = E - self.e_prev
        self.e_prev = E
        dmid = -(Kp * dE + Kp / Ti * E * DT)          # 增量式 PI (10s 步长)
        if self.v_cmd is None:
            self.v_cmd = float(a_last[1])             # 从实际阀位开始
        self.v_cmd = float(np.clip(self.v_cmd + self.k_sub * dmid, self.u_lo, self.u_hi))
        return np.array([a_last[0], self.v_cmd], dtype=np.float64), 1


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════
def run_arm(name, controller_factory, world, starts_by_seed, dist_on, n_steps):
    rows = []
    for s_seed, starts in starts_by_seed.items():
        for k, st in enumerate(starts):
            dist = Disturbance(sigma=0.3 if dist_on else 0.0, rho=0.9,
                               mode='physical', seed=hash((s_seed, k)) % (2 ** 31))
            ep = run_episode(controller_factory(), world, st, n_steps=n_steps, dist=dist)
            if len(ep['temp']) < 10:
                continue
            m = metrics(ep)
            m.update(arm=name, start_seed=s_seed, track=int(st))
            m['ep'] = ep
            rows.append(m)
    return rows


def summarize(rows, keys=('rmse', 'tv', 'overtemp_s', 'overtemp_int', 'iae', 'j_total')):
    df = pd.DataFrame([{k: r[k] for k in keys} for r in rows])
    return df.describe().loc[['mean', 'std', 'min', '50%', 'max']].round(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    n_tracks = 2 if args.smoke else 20
    seeds = [42]
    n_steps = 40 if args.smoke else 120
    dist_on = True

    from experiments.phase2_mpc.exp_S1_fair_comparison import make_starts
    starts_by_seed = {s: make_starts(s, n_tracks) for s in seeds}

    CONTROLLER_IDS = ['M7']          # 被测控制器模型 (MPC 规划用 M7)
    world = WorldSim(['M0', 'M8', 'M9'], controller_ids=CONTROLLER_IDS)
    wm_m7 = load_wm('M7')

    print(f"[exp_101] 世界集成 M0/M8/M9 | 控制器: pid_real(真实POU#107) / pid_legacy / mpc_m7")
    print(f"[exp_101] 规模: {len(seeds)} seed × {n_tracks} tracks × {n_steps} 步 | 扰动: physical σ=0.3 ρ=0.9")

    arms = {
        'pid_real': lambda: RealPIDController(),
        'pid_legacy': lambda: PIDController(),
        'mpc_m7': lambda: MPCController(wm_m7, CostConfig('d'), h_plan=10, m_step=6),
    }
    all_rows = {}
    for name, fac in arms.items():
        rows = run_arm(name, fac, world, starts_by_seed, dist_on, n_steps)
        all_rows[name] = rows
        print(f"\n=== {name} (n={len(rows)}) ===")
        print(summarize(rows).to_string())

    # ── 配对对比: MPC vs PID-Real (同 track) ──
    print("\n=== 配对对比 mpc_m7 vs pid_real (每 track RMSE/TV/超温积分) ===")
    by_track = {}
    for r in all_rows['pid_real'] + all_rows['mpc_m7']:
        by_track.setdefault(r['track'], {})[r['arm']] = r
    paired = []
    for tr, d in sorted(by_track.items()):
        if 'pid_real' in d and 'mpc_m7' in d:
            paired.append((tr, d))
    wins = {'rmse': 0, 'tv': 0, 'overtemp_int': 0}
    for tr, d in paired:
        for k in wins:
            if d['mpc_m7'][k] < d['pid_real'][k]:
                wins[k] += 1
    print(f"配对轨迹数: {len(paired)}")
    for k, w in wins.items():
        print(f"  {k}: MPC 胜 {w}/{len(paired)} ({w / len(paired) * 100:.0f}%)")
    if paired:
        d_r = np.array([d['pid_real']['rmse'] for _, d in paired])
        d_m = np.array([d['mpc_m7']['rmse'] for _, d in paired])
        print(f"  RMSE: pid_real 中位 {np.median(d_r):.3f} | mpc_m7 中位 {np.median(d_m):.3f} | "
              f"配对差 (M−R) 中位 {np.median(d_m - d_r):+.3f}")
        d_tv_r = np.array([d['pid_real']['tv'] for _, d in paired])
        d_tv_m = np.array([d['mpc_m7']['tv'] for _, d in paired])
        print(f"  TV:   pid_real 中位 {np.median(d_tv_r):.3f} | mpc_m7 中位 {np.median(d_tv_m):.3f}")

    # ── case 曲线: MPC vs PID-Real 差 RMSE 最大/中位/最小 三轨迹 ──
    if len(paired) >= 3 and not args.smoke:
        diffs = sorted(paired, key=lambda x: x[1]['mpc_m7']['rmse'] - x[1]['pid_real']['rmse'])
        picks = [diffs[0], diffs[len(diffs) // 2], diffs[-1]]
        fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=False)
        for row, (tr, d) in enumerate(picks):
            for col, arm in enumerate(['pid_real', 'mpc_m7']):
                ax = axes[row, col]
                ep = d[arm]['ep']
                t = np.arange(len(ep['temp'])) * DT / 60.0
                ax.plot(t, ep['sp'], 'k--', lw=1.0, label='SP')
                ax.plot(t, ep['temp'], 'b-', lw=1.0, label='PV')
                ax.axhline(T_MAX, color='r', ls=':', lw=0.8)
                ax.set_title(f"track {tr} | {arm} | RMSE={d[arm]['rmse']:.2f}°C, TV={d[arm]['tv']:.2f}%/step")
                ax.set_ylabel('degC')
                ax.legend(fontsize=8)
                ax2 = ax.twinx()
                ax2.plot(t, ep['act'][:, 1], 'g-', lw=0.8, alpha=0.7, label='valve')
                ax2.set_ylabel('valve %', color='g')
                ax2.tick_params(axis='y', labelcolor='g')
        axes[0, 0].set_title(axes[0, 0].get_title() + "  [MPC worst]")
        axes[-1, 0].set_title(axes[-1, 0].get_title() + "  [MPC best]")
        plt.tight_layout()
        outdir = os.path.join(_ROOT, 'results', 'exp_101_realpid_vs_mpc')
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, 'case_curves.png'), dpi=110)
        print(f"\nsaved case_curves.png")

    # ── 保存 ──
    outdir = os.path.join(_ROOT, 'results', 'exp_101_realpid_vs_mpc')
    os.makedirs(outdir, exist_ok=True)
    for name, rows in all_rows.items():
        slim = [{k: v for k, v in r.items() if k != 'ep'} for r in rows]
        json.dump(slim, open(os.path.join(outdir, f'{name}.json'), 'w'), indent=1, default=str)
    print(f"results saved to {outdir}")


if __name__ == '__main__':
    main()
