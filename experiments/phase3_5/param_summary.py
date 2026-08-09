#!/usr/bin/env python3
"""G3 参数健康摘要: 保留 42-run 闭合，并只用 36 个已训练物理分支作健康判断。

修复项 (2026-08-09, 审计 §3.1 + TODO 5B):
- tau 分两个 stage 报告, 且换算为真实秒 (10s 采样步 × 10)
- rate_gain 直接从 model.forward 读取 (model 已补返回)
- 给 free_only 未训练 physics 分支显式打标并排除于健康统计
- 输出 checkpoint SHA256 + anchor 抽样 hash
- 固定真实阀位扰动 (+5% 开度) 报告 action IRF, 禁止跨 opening map 直接比较 raw K
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.phase35.data import load_cache, valid_window_anchors, extract_windows
from src.phase35.matrix import load_matrix, expand_matrix
from src.phase35.schema import TARGET_COLUMN, VALVE_COLUMN
from src.phase35.training import build_model, git_sha

RUN_ROOT = 'results/phase3_5/runs'
CACHE = {'A': '/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/phase3_5_cache/cache_A.npz',
         'B': '/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/phase3_5_cache/cache_B.npz'}
N_ANCHORS = 256
FIXED_PERTURB_PCT = 5.0        # 固定真实阀位扰动 (%) 用于跨 map 可比 IRF


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_status_paths() -> list[str] | None:
    try:
        output = subprocess.check_output(
            ['git', '-C', ROOT, 'status', '--porcelain'], stderr=subprocess.DEVNULL, text=True
        )
        return [line[3:].strip() for line in output.splitlines() if line.strip()]
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', default='configs/phase3_5/experiment_matrix.json')
    parser.add_argument('--run-root', default=RUN_ROOT)
    parser.add_argument('--cache-a', default=CACHE['A'])
    parser.add_argument('--cache-b', default=CACHE['B'])
    parser.add_argument('--output', default='results/phase3_5/param_summary_validation.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--allow-dirty', action='store_true', help='仅调试；正式产物禁止从 dirty tree 生成')
    args = parser.parse_args()
    dirty_paths_at_start = git_status_paths()
    dirty_non_result_paths = [
        path for path in (dirty_paths_at_start or []) if not path.replace('\\', '/').startswith('results/')
    ]
    if dirty_non_result_paths and not args.allow_dirty:
        parser.error(f'working tree has uncommitted non-result files: {dirty_non_result_paths}')
    generated_git_sha = git_sha(ROOT)
    device = torch.device(args.device)
    caches = {'A': args.cache_a, 'B': args.cache_b}
    matrix = load_matrix(args.matrix)
    runs = expand_matrix(matrix)
    summaries = []
    for run in runs:
        run_dir = os.path.join(args.run_root, run.run_id)
        ckpt_path = os.path.join(run_dir, 'checkpoint_best_val.pt')
        if not os.path.exists(ckpt_path):
            print(f'MISS {run.run_id}'); continue
        cache = load_cache(caches[run.side])
        model, features = build_model(run.config, cache, device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        model.eval()
        anchors = valid_window_anchors(cache, 'validation', features, TARGET_COLUMN,
                                       VALVE_COLUMN, run.config.window, run.config.horizon)
        rng = np.random.default_rng(0)
        anchors = anchors[rng.choice(len(anchors), min(N_ANCHORS, len(anchors)), replace=False)]
        w = extract_windows(cache, anchors, features, TARGET_COLUMN, VALVE_COLUMN,
                            run.config.window, run.config.horizon)
        hist = torch.from_numpy(w['history']).to(device)
        fv = torch.from_numpy(w['future_valve']).to(device)
        bv = torch.from_numpy(w['baseline_valve']).to(device)
        with torch.no_grad():
            out = model(hist, fv, bv)
        gain = out['gain'].cpu().numpy().ravel()
        tau = out['tau'].cpu().numpy()  # [n, 2]: stage1, stage2 (采样步)
        rate = out['rate_gain'].cpu().numpy().ravel() if out['rate_gain'] is not None else None
        step_seconds = int(cache.metadata.get('step_seconds', 0))
        if step_seconds <= 0:
            raise ValueError(f'{run.run_id}: cache step_seconds must be positive')
        tau_min, tau_max = model.physics.tau_min, model.physics.tau_max
        physics_parameters_trained = run.config.action_mode != 'none'

        # 固定扰动 IRF: baseline_valve + FIXED_PERTURB_PCT 开度 (真实阀位单位, 跨 map 可比)
        with torch.no_grad():
            fv_perturb = (bv + FIXED_PERTURB_PCT)[:, None].expand(bv.shape[0], run.config.horizon).to(DEVICE)
            eff = model.intervention_effect(hist, fv_perturb, bv)
        irf_h60 = eff[:, -1].cpu().numpy().mean()  # °C @600s

        s = {
            'run_id': run.run_id, 'config': run.config.config_id, 'side': run.side, 'seed': run.seed,
            'action_mode': run.config.action_mode, 'opening_map': run.config.opening_map,
            'rate_branch': run.config.rate_branch, 'free_head': run.config.free_head,
            'physics_parameters_trained': physics_parameters_trained,
            # gain (每 effective-dose unit；不同 opening map 的 raw K 不直接比较)
            'gain_mean': float(gain.mean()), 'gain_p05': float(np.quantile(gain, .05)),
            'gain_p95': float(np.quantile(gain, .95)),
            'gain_near_zero': float(np.mean(np.abs(gain) < 1e-3)),
            # tau stage1 (步 + 秒)
            'tau1_steps_mean': float(tau[:, 0].mean()), 'tau1_seconds_mean': float(tau[:, 0].mean() * step_seconds),
            'tau1_steps_p05': float(np.quantile(tau[:, 0], .05)), 'tau1_steps_p95': float(np.quantile(tau[:, 0], .95)),
            'tau1_at_min': float(np.mean(tau[:, 0] <= tau_min * 1.01)),
            'tau1_at_max': float(np.mean(tau[:, 0] >= tau_max * 0.99)),
            # tau stage2 (步 + 秒)
            'tau2_steps_mean': float(tau[:, 1].mean()), 'tau2_seconds_mean': float(tau[:, 1].mean() * step_seconds),
            'tau2_steps_p05': float(np.quantile(tau[:, 1], .05)), 'tau2_steps_p95': float(np.quantile(tau[:, 1], .95)),
            'tau2_at_min': float(np.mean(tau[:, 1] <= tau_min * 1.01)),
            'tau2_at_max': float(np.mean(tau[:, 1] >= tau_max * 0.99)),
            # rate gain
            'rate_mean': float(rate.mean()) if rate is not None else None,
            'rate_near_zero': float(np.mean(np.abs(rate) < 1e-3)) if rate is not None else None,
            # 固定扰动 IRF (真实阀位 +5%, °C @600s)
            'fixed_irf_h60_mean_c': float(irf_h60),
            # 溯源
            'checkpoint_sha256': sha256_file(ckpt_path),
            'anchor_index_sha256': hashlib.sha256(anchors.astype(np.int64).tobytes()).hexdigest(),
            'anchor_input_sha256': hashlib.sha256(
                w['history'].tobytes() + w['future_valve'].tobytes() + w['baseline_valve'].tobytes()
            ).hexdigest(),
            'cache_source_sha256': (cache.metadata.get('source') or {}).get('sha256'),
            'cache_side': cache.metadata.get('side'),
            'step_seconds': step_seconds,
            'generated_git_sha': generated_git_sha,
            'git_dirty_at_start': bool(dirty_paths_at_start),
            'dirty_non_result_paths_at_start': dirty_non_result_paths,
            'n_anchors': int(len(anchors)),
        }
        summaries.append(s)
        trained = '' if physics_parameters_trained else '(excluded: untrained physics branch)'
        rate_str = 'n/a' if rate is None else f"{s['rate_mean']:.4f}"
        print(f"{run.run_id:<40} gain={s['gain_mean']:+.4f} (near0 {s['gain_near_zero']:.0%})  "
              f"tau1={s['tau1_seconds_mean']:6.0f}s tau2={s['tau2_seconds_mean']:6.0f}s  "
              f"rate={rate_str}  "
              f"IRF+5%={s['fixed_irf_h60_mean_c']:+.4f}°C {trained}")
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as f:
        json.dump(summaries, f, indent=1)
    print(f'\nsaved: {output} ({len(summaries)} runs; '
          f'{sum(s["physics_parameters_trained"] for s in summaries)} trained physics branches)')


if __name__ == '__main__':
    main()
