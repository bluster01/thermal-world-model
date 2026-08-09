#!/usr/bin/env python3
"""G3 参数健康摘要 v2 (审计 5B 修复版): 42 个 checkpoint 的 gain/τ/rate 参数分布与边界命中率。

修复项 (2026-08-09, 审计 §3.1 + TODO 5B):
- tau 分两个 stage 报告, 且换算为真实秒 (10s 采样步 × 10)
- rate_gain 直接从 model.forward 读取 (model 已补返回)
- 排除 free_only 未训练 physics 分支 (gain=-0.05/τ=18 只是初始化)
- 输出 checkpoint SHA256 + anchor 抽样 hash
- 固定真实阀位扰动 (+5% 开度) 报告 action IRF, 禁止跨 opening map 直接比较 raw K
"""
import os, sys, json, hashlib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.phase35.data import load_cache, valid_window_anchors, extract_windows
from src.phase35.matrix import load_matrix, expand_matrix
from src.phase35.schema import TARGET_COLUMN, VALVE_COLUMN
from src.phase35.training import build_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RUN_ROOT = 'results/phase3_5/runs'
CACHE = {'A': '/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/phase3_5_cache/cache_A.npz',
         'B': '/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/phase3_5_cache/cache_B.npz'}
N_ANCHORS = 256
STEP_SECONDS = 10
TAU_MIN, TAU_MAX = 1.5, 120.0  # 采样步
FIXED_PERTURB_PCT = 5.0        # 固定真实阀位扰动 (%) 用于跨 map 可比 IRF


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    matrix = load_matrix('configs/phase3_5/experiment_matrix.json')
    runs = expand_matrix(matrix)
    summaries = []
    for run in runs:
        run_dir = os.path.join(RUN_ROOT, run.run_id)
        ckpt_path = os.path.join(run_dir, 'checkpoint_best_val.pt')
        if not os.path.exists(ckpt_path):
            print(f'MISS {run.run_id}'); continue
        cache = load_cache(CACHE[run.side])
        model, features = build_model(run.config, cache, DEVICE)
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        model.eval()
        anchors = valid_window_anchors(cache, 'validation', features, TARGET_COLUMN,
                                       VALVE_COLUMN, run.config.window, run.config.horizon)
        rng = np.random.default_rng(0)
        anchors = anchors[rng.choice(len(anchors), min(N_ANCHORS, len(anchors)), replace=False)]
        w = extract_windows(cache, anchors, features, TARGET_COLUMN, VALVE_COLUMN,
                            run.config.window, run.config.horizon)
        hist = torch.from_numpy(w['history']).to(DEVICE)
        fv = torch.from_numpy(w['future_valve']).to(DEVICE)
        bv = torch.from_numpy(w['baseline_valve']).to(DEVICE)
        with torch.no_grad():
            out = model(hist, fv, bv)
        gain = out['gain'].cpu().numpy().ravel()
        tau = out['tau'].cpu().numpy()  # [n, 2]: stage1, stage2 (采样步)
        rate = out['rate_gain'].cpu().numpy().ravel() if out['rate_gain'] is not None else None

        # 固定扰动 IRF: baseline_valve + FIXED_PERTURB_PCT 开度 (真实阀位单位, 跨 map 可比)
        with torch.no_grad():
            fv_perturb = (bv + FIXED_PERTURB_PCT)[:, None].expand(bv.shape[0], run.config.horizon).to(DEVICE)
            eff = model.intervention_effect(hist, fv_perturb, bv)
        irf_h60 = eff[:, -1].cpu().numpy().mean()  # °C @600s

        s = {
            'run_id': run.run_id, 'config': run.config.config_id, 'side': run.side, 'seed': run.seed,
            'action_mode': run.config.action_mode, 'opening_map': run.config.opening_map,
            'rate_branch': run.config.rate_branch, 'free_head': run.config.free_head,
            # gain (采样步的每开度%增益)
            'gain_mean': float(gain.mean()), 'gain_p05': float(np.quantile(gain, .05)),
            'gain_p95': float(np.quantile(gain, .95)),
            'gain_near_zero': float(np.mean(np.abs(gain) < 1e-3)),
            # tau stage1 (步 + 秒)
            'tau1_steps_mean': float(tau[:, 0].mean()), 'tau1_seconds_mean': float(tau[:, 0].mean() * STEP_SECONDS),
            'tau1_p05': float(np.quantile(tau[:, 0], .05)), 'tau1_p95': float(np.quantile(tau[:, 0], .95)),
            'tau1_at_min': float(np.mean(tau[:, 0] <= TAU_MIN * 1.01)),
            'tau1_at_max': float(np.mean(tau[:, 0] >= TAU_MAX * 0.99)),
            # tau stage2 (步 + 秒)
            'tau2_steps_mean': float(tau[:, 1].mean()), 'tau2_seconds_mean': float(tau[:, 1].mean() * STEP_SECONDS),
            'tau2_p05': float(np.quantile(tau[:, 1], .05)), 'tau2_p95': float(np.quantile(tau[:, 1], .95)),
            'tau2_at_min': float(np.mean(tau[:, 1] <= TAU_MIN * 1.01)),
            'tau2_at_max': float(np.mean(tau[:, 1] >= TAU_MAX * 0.99)),
            # rate gain
            'rate_mean': float(rate.mean()) if rate is not None else None,
            'rate_near_zero': float(np.mean(np.abs(rate) < 1e-3)) if rate is not None else None,
            # 固定扰动 IRF (真实阀位 +5%, °C @600s)
            'fixed_irf_h60_mean_c': float(irf_h60),
            # 溯源
            'checkpoint_sha256': sha256_file(ckpt_path)[:16],
            'anchor_hash': hashlib.sha256(anchors.astype(np.int64).tobytes()).hexdigest()[:16],
            'n_anchors': int(len(anchors)),
        }
        summaries.append(s)
        trained = '' if run.config.action_mode == 'none' and run.config.free_head is False else ''
        trained = '(free-only)' if run.config.action_mode == 'none' and not run.config.rate_branch else ''
        rate_str = 'n/a' if rate is None else f"{s['rate_mean']:.4f}"
        print(f"{run.run_id:<40} gain={s['gain_mean']:+.4f} (near0 {s['gain_near_zero']:.0%})  "
              f"tau1={s['tau1_seconds_mean']:6.0f}s tau2={s['tau2_seconds_mean']:6.0f}s  "
              f"rate={rate_str}  "
              f"IRF+5%={s['fixed_irf_h60_mean_c']:+.4f}°C {trained}")
    with open('results/phase3_5/param_summary_validation.json', 'w') as f:
        json.dump(summaries, f, indent=1)
    print(f'\nsaved: results/phase3_5/param_summary_validation.json ({len(summaries)} runs)')


if __name__ == '__main__':
    main()
