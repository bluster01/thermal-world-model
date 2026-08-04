#!/usr/bin/env python3
"""
exp_S4_linear_mpc.py — S4: 线性 MPC 公平协议重验 (2026-08-04)
================================================================
S4a: 开环预测对比 — M7 vs ARX (ΔT-ARX) 在物理扰动下逐步预测误差累积, 定位崩溃根因
S4b: 线性 MPC (ARX) ± EMA offset-free 补偿 vs PID, 150 轨迹公平协议

判定 (docs/supplementary_experiments.md §4):
  加补偿后显著改善 → 崩溃部分是可补偿偏差
  改善有限 → 线性模型结构性局限
用法: python exp_S4_linear_mpc.py [--smoke]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.phase2_mpc.exp_S1_fair_comparison import (  # noqa: E402
    N_TRACKS, START_SEEDS, N_STEPS, M_STEP, DIST_SIGMA, DIST_RHO, WORLD_IDS,
    CONTROLLERS, make_starts, run_arm, summarize, paired_test)
from experiments.phase2_mpc.eval_protocol import (  # noqa: E402
    CostConfig, Disturbance, MPCController, PIDController, WorldSim, load_wm,
    run_episode, DEVICE, W, H_OUT, TARGET_IDX)

sys.path.insert(0, os.path.join(_ROOT, 'experiments', 'phase2_mpc'))

# ===== LinearARX 内联 (exp_068 无 __main__ 保护, import 会触发全量评测 — 不能 import) =====
ORDER = 12
COEF = np.load('results/exp_067_linear_mpc/arx_dT_coef.npy')


class LinearARX:
    """ΔT-ARX 增量模型 (复制自 exp_068): __call__(x_hist, a_future) -> (mu [B,H], lv [B,H])"""
    def __init__(self, coef, order=12):
        self.coef = torch.FloatTensor(coef).to(DEVICE)
        self.order = order
        self.probabilistic = False
    def __call__(self, x_hist, a_future):
        B = x_hist.shape[0]
        H = a_future.shape[1] // 2
        a = a_future.reshape(B, H, 2)
        dev = x_hist.device
        d_hist = torch.diff(x_hist[:, :, TARGET_IDX], dim=1)[:, -ORDER:].flip(1)
        V1h = x_hist[:, -ORDER:, 37].flip(1)
        V2h = x_hist[:, -ORDER:, 38].flip(1)
        SPh = x_hist[:, -6:, 36].flip(1)
        t_cur = x_hist[:, -1, TARGET_IDX]
        mus = []
        for k in range(H):
            idxs = torch.arange(k, k + ORDER, device=dev).clamp(max=H - 1)
            v1 = a[:, idxs, 0].flip(1)
            v2 = a[:, idxs, 1].flip(1)
            r = torch.cat([d_hist, v1, v2, SPh, torch.ones(B, 1, device=dev)], 1)
            dd = r @ self.coef
            t_cur = t_cur + dd
            mus.append(t_cur)
            d_hist = torch.cat([dd.unsqueeze(1), d_hist[:, :-1]], 1)
        mu = torch.stack(mus, 1)
        lv = torch.full_like(mu, 2 * np.log(0.01))
        return mu, lv

OUT_DIR = 'results/exp_S4'
COST = 'a'                        # RMSE-only 定稿口径


class OffsetMPCController(MPCController):
    """EMA 偏差补偿: 窗口末实际温度 vs 模型预测 → 补偿加到 rollout μ"""
    def __init__(self, wm, cost, h_plan=10, m_step=6, offset_gain=0.3, **kw):
        super().__init__(wm, cost, h_plan=h_plan, m_step=m_step, **kw)
        self.offset, self.gain = 0.0, offset_gain

    def _rollout(self, win, a_seq, sp_fut, a_hist1):
        mu, sigma, a2 = super()._rollout(win, a_seq, sp_fut, a_hist1)
        return mu + self.offset, sigma, a2

    def act(self, win, sp_now, sp_fut, a_last):
        with torch.no_grad():
            # 一维动作 (二级阀, control_both=False), _rollout 内部 stack 成 [h_plan,2]
            a_ref = torch.FloatTensor([float(a_last[1])] * self.h_plan).to(DEVICE)
            mu0, _, _ = super()._rollout(win, a_ref, sp_fut,
                                         torch.tensor(float(a_last[0]), device=DEVICE))
            y_meas = float(win[0, -1, TARGET_IDX])
            self.offset += self.gain * (y_meas - float(mu0[0]))
        return super().act(win, sp_now, sp_fut, a_last)


def s4a(world, starts_by_seed, dist_on=True, n_trajs=10, n_steps=60):
    """开环预测对比: M7 vs ARX, 每 6 步重新预测 18 步, 对比前 6 步误差 vs 世界实际"""
    from experiments.phase2_mpc.eval_protocol import test_raw as raw
    wm7 = load_wm('M7')
    arx = LinearARX(COEF, ORDER)
    arx.probabilistic = False

    errs = {m: [] for m in ('M7', 'ARX')}
    for seed, starts in list(starts_by_seed.items())[:1]:
        for s in starts[:n_trajs]:
            gi = s + W
            for j in range(0, n_steps, M_STEP):
                if gi + H_OUT >= len(raw):
                    break
                win = torch.FloatTensor(raw[gi-W:gi]).unsqueeze(0).to(DEVICE)
                a_ref = torch.FloatTensor(raw[gi:gi+H_OUT, 37:39]).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    mu7, _ = wm7(win, a_ref.reshape(1, -1))
                    mu_a, _ = arx(win, a_ref.reshape(1, -1))
                y_true = raw[gi:gi+M_STEP, TARGET_IDX]
                errs['M7'].append(np.abs(mu7[0, :M_STEP].cpu().numpy() - y_true).mean())
                errs['ARX'].append(np.abs(mu_a[0, :M_STEP].cpu().numpy() - y_true).mean())
                gi += M_STEP
    for m in ('M7', 'ARX'):
        e = np.array(errs[m])
        print(f"  [S4a] {m}: 6步预测 MAE {e.mean():.3f}°C (中位 {np.median(e):.3f}, p90 {np.percentile(e,90):.3f})")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    n_tracks = 2 if args.smoke else N_TRACKS
    seeds = START_SEEDS[:1] if args.smoke else START_SEEDS
    n_steps = 40 if args.smoke else N_STEPS
    os.makedirs(OUT_DIR, exist_ok=True)
    starts_by_seed = {s: make_starts(s, n_tracks) for s in seeds}

    print(f"[S4] 世界={WORLD_IDS} 成本=cost_{COST} 轨迹={len(seeds)}×{n_tracks}")
    world = WorldSim(WORLD_IDS, controller_ids=CONTROLLERS)
    arx = LinearARX(COEF, ORDER)
    arx.probabilistic = False
    cost = CostConfig(COST)

    # ── S4a: 开环预测对比 ──
    print("\n===== S4a: 开环预测 (M7 vs ARX, 物理扰动下) =====")
    s4a(world, starts_by_seed, n_trajs=n_tracks)

    # ── S4b: 闭环评测 ──
    print("\n===== S4b: 闭环 (150轨迹协议) =====")
    all_rows, summary = [], {}
    pid_rows = run_arm('PID', lambda: PIDController(), world, starts_by_seed, True, n_steps)
    all_rows += pid_rows; summary['PID'] = summarize(pid_rows)
    print(f"  PID            {summary['PID']}")

    for arm, fac in [
        ('LinMPC', lambda: MPCController(arx, cost, h_plan=10, m_step=M_STEP)),
        ('LinMPC+off', lambda: OffsetMPCController(arx, cost, h_plan=10, m_step=M_STEP, offset_gain=0.3)),
    ]:
        rows = run_arm(arm, fac, world, starts_by_seed, True, n_steps)
        all_rows += rows; summary[arm] = summarize(rows)
        p_r, n = paired_test(rows, pid_rows, 'rmse')
        print(f"  {arm:10s} {summary[arm]} | vs PID p_rmse={p_r:.2e} (n={n})")

    # 补偿效果
    r0 = [r for r in all_rows if r['arm'] == 'LinMPC']
    r1 = [r for r in all_rows if r['arm'] == 'LinMPC+off']
    if r0 and r1:
        for key in ('rmse', 'tv', 'j_total'):
            p, n = paired_test(r0, r1, key)
            print(f"  [补偿] {key}: {np.mean([r[key] for r in r0]):.3f} → {np.mean([r[key] for r in r1]):.3f} (p={p:.2e})")

    out = dict(protocol=dict(world=WORLD_IDS, cost=COST, h_plan=10, m_step=M_STEP,
                             dist=dict(on=True, sigma=DIST_SIGMA, rho=DIST_RHO, mode='physical'),
                             start_seeds=seeds, n_tracks=n_tracks, offset_gain=0.3),
               summary=summary, per_track=all_rows)
    fn = os.path.join(OUT_DIR, f"s4_linear{'_smoke' if args.smoke else ''}.json")
    json.dump(out, open(fn, 'w'), indent=2, default=float)
    print(f"\nSaved: {fn}")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
