#!/usr/bin/env python3
"""
exp_112_koopman_full.py — Koopman free_head 完整 50-epoch 实验
=================================================================
P0 实验: 完整训练 + unfreeze, 验证 Koopman 特征值能否学到物理衰减率。

对比:
  A1phys        (baseline): MLP free_head + InterventionPhysics, ff10
  A1phys_koopman (α=3.0):   KoopmanFreeHead + InterventionPhysics, ff10+ai3.0
  A1phys_null   (ceiling):  free_head=None, 纯干预驱动 T̂ = g(x,a)

关键改动 vs exp_111:
  - 50 epochs (vs 8): freeze 10 + unfreeze 40, 给 Koopman 特征值时间学习
  - alpha_init=3.0: λ=tanh(3)≈0.995 匹配热惯性 ~1000s 先验
  - A1phys_null: 验证 pure-intervention 天花板
  - patience=20: 允许充分收敛

协议:
  W=96, H=60, BS=256, STEPS=500, AdamW LR 1e-3
  双 ckpt: best_mae + best_causal
"""
import os
import sys
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_106_causal_arch import (
    train_one, DEVICE, W, N_FEAT, TARGET_IDX, OUT_ROOT,
    VARIANTS, PROFILE_K, build_model, eval_mae, eval_causal,
    train_raw, test_raw
)
import causal_eval as CE
import causal_arch as CA

# ===================================================== 配置
VARIANTS_TO_RUN = ['A1phys', 'A1phys_koopman', 'A1phys_null']
N_EPOCHS = 50
SEEDS = [0, 1, 2]
P_FREEZE = 10              # ff10: 冻结 free 分支前 10 epoch
ALPHA_INIT = 3.0           # Koopman α 初始化 (λ=tanh(3)≈0.995)
PATIENCE = 20              # 早停

OUT_DIR = 'results/exp_112_koopman_full'
os.makedirs(OUT_DIR, exist_ok=True)

# DiD ground truth (P2 expanded, n=79)
GT_PATH = 'results/cfe_groundtruth_p2/did_response.json'


def load_gt():
    if os.path.exists(GT_PATH):
        with open(GT_PATH) as f:
            return json.load(f)
    return None


def print_header():
    print("=" * 90)
    print(f"exp_112_koopman_full — Koopman 完整 50-epoch 实验")
    print(f"  variants: {VARIANTS_TO_RUN}")
    print(f"  seeds: {SEEDS}")
    print(f"  epochs: {N_EPOCHS} (freeze-free={P_FREEZE}, unfreeze={N_EPOCHS - P_FREEZE})")
    print(f"  alpha_init: {ALPHA_INIT} (λ≈{np.tanh(ALPHA_INIT):.3f})")
    print(f"  patience: {PATIENCE}")
    print(f"  Device: {DEVICE}")
    print("=" * 90)


def main():
    print_header()
    gt = load_gt()
    if gt:
        print(f"[gt] DiD 真值已载入 (n_ev={gt.get('n_ev', '?')})")
    else:
        print("[gt] DiD 真值未找到 → CFI 退化为 sign(ΔSP) 口径")

    all_results = {}

    for variant in VARIANTS_TO_RUN:
        print(f"\n{'─' * 80}")
        print(f"  ▶ {variant}")
        print(f"{'─' * 80}")
        variant_results = []

        for seed in SEEDS:
            print(f"\n  [{variant} s{seed}] training...")
            t0 = time.time()

            # Koopman 变体使用 alpha_init, null 变体无 free_head
            ai = ALPHA_INIT if 'koopman' in variant else 0.0

            res = train_one(
                variant, seed,
                smoke=False,
                gt=gt,
                epochs=N_EPOCHS,
                flat_weight=False,
                h_override=None,
                loss_type='nll',
                freeze_free_epochs=P_FREEZE,
                lambda_gain=0.0,
                n_lag=2,
                alpha_init=ai,
                patience=PATIENCE,
                min_delta=1e-4,
            )

            elapsed = time.time() - t0
            variant_results.append(res)
            print(f"  [{variant} s{seed}] done ({elapsed:.0f}s) | "
                  f"best MAE {res['best']['mae']:.4f}@ep{res['best']['mae_ep']} | "
                  f"best CFI {res['best']['cfi']:.3f}@ep{res['best']['cfi_ep']}")

        all_results[variant] = variant_results

    # ============================================ 汇总报告
    print("\n" + "=" * 90)
    print("对比报告")
    print("=" * 90)

    header = (f"{'变体':18s} {'seed':4s} "
              f"{'MAE':>7s} {'CFI':>7s} "
              f"{'gain_600':>9s} {'gain_180':>9s} "
              f"{'dir_600':>7s} {'dir_180':>7s} "
              f"{'n_param':>8s} {'min':>6s}")
    print(header)
    print("-" * 90)

    for variant in VARIANTS_TO_RUN:
        for i, r in enumerate(all_results[variant]):
            seed = SEEDS[i]
            fc = r.get('final_causal', {})
            prof = fc.get('profile', {}) if fc else {}
            g600 = prof.get('600s', {})
            g180 = prof.get('180s', {})
            gain_600 = g600.get('gain_norm', float('nan'))
            gain_180 = g180.get('gain_norm', float('nan'))
            dir_600 = g600.get('dir_dsp', float('nan'))
            dir_180 = g180.get('dir_dsp', float('nan'))

            print(f"{variant:18s} {seed:4d} "
                  f"{r['best']['mae']:7.4f} {r['best']['cfi']:7.3f} "
                  f"{gain_600:9.4f} {gain_180:9.4f} "
                  f"{dir_600:6.0%} {dir_180:6.0%} "
                  f"{r['n_param']:8d} {r.get('minutes', 0):5.0f}min")

    # ============================================ 聚合统计
    print("\n" + "-" * 90)
    print("聚合 (mean ± std over seeds)")
    print("-" * 90)

    for variant in VARIANTS_TO_RUN:
        results = all_results[variant]
        maes = [r['best']['mae'] for r in results]
        cfis = [r['best']['cfi'] for r in results]
        gains600 = []
        gains180 = []
        for r in results:
            fc = r.get('final_causal', {})
            prof = fc.get('profile', {}) if fc else {}
            g600 = prof.get('600s', {})
            g180 = prof.get('180s', {})
            gains600.append(g600.get('gain_norm', float('nan')))
            gains180.append(g180.get('gain_norm', float('nan')))

        print(f"{variant:18s} "
              f"MAE={np.mean(maes):.4f}±{np.std(maes):.4f}  "
              f"CFI={np.mean(cfis):.3f}±{np.std(cfis):.3f}  "
              f"gain600={np.nanmean(gains600):.4f}±{np.nanstd(gains600):.4f}  "
              f"gain180={np.nanmean(gains180):.4f}±{np.nanstd(gains180):.4f}")

    # ============================================ A1phys_null 参数数
    if 'A1phys_null' in all_results:
        n_null = all_results['A1phys_null'][0]['n_param']
        n_mlp = all_results['A1phys'][0]['n_param']
        n_koop = all_results['A1phys_koopman'][0]['n_param']
        print(f"\n  参数数: A1phys_null={n_null} (基线={n_mlp}, Δ={n_mlp - n_null}), "
              f"Koopman={n_koop}")

    # Koopman 特征值 (可解释性) — 遍历 best_mae ckpt
    if 'A1phys_koopman' in all_results:
        print("\n" + "-" * 90)
        print("Koopman 特征值 (可解释性, best_mae ckpt)")
        print("-" * 90)
        for i, r in enumerate(all_results['A1phys_koopman']):
            ckpt_path = os.path.join(
                OUT_ROOT,
                f'A1phys_koopman_s{SEEDS[i]}_ff{P_FREEZE}_ai{ALPHA_INIT}',
                'checkpoints', 'best_mae.pth'
            )
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = build_model('A1phys_koopman', H=60, n_lag=2,
                                    alpha_init=ALPHA_INIT)
                model.load_state_dict(ckpt['model_state_dict'])
                ev = model.free_head.eigenvalues()
                n_near1 = np.sum(np.abs(ev) > 0.99)
                n_near0 = np.sum(np.abs(ev) < 0.01)
                top5 = np.sort(np.abs(ev))[-5:][::-1]
                print(f"  s{SEEDS[i]} ep{ckpt['ep']}: "
                      f"|λ|_max={np.max(np.abs(ev)):.4f}  "
                      f"near1={n_near1}/{len(ev)}  near0={n_near0}/{len(ev)}  "
                      f"top5={[f'{x:.4f}' for x in top5]}")
                del model

        # 也检查 best_causal
        print("\n  (best_causal ckpt):")
        for i, r in enumerate(all_results['A1phys_koopman']):
            ckpt_path = os.path.join(
                OUT_ROOT,
                f'A1phys_koopman_s{SEEDS[i]}_ff{P_FREEZE}_ai{ALPHA_INIT}',
                'checkpoints', 'best_causal.pth'
            )
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = build_model('A1phys_koopman', H=60, n_lag=2,
                                    alpha_init=ALPHA_INIT)
                model.load_state_dict(ckpt['model_state_dict'])
                ev = model.free_head.eigenvalues()
                n_near1 = np.sum(np.abs(ev) > 0.99)
                n_near0 = np.sum(np.abs(ev) < 0.01)
                top5 = np.sort(np.abs(ev))[-5:][::-1]
                print(f"  s{SEEDS[i]} ep{ckpt['ep']}: "
                      f"|λ|_max={np.max(np.abs(ev)):.4f}  "
                      f"near1={n_near1}/{len(ev)}  near0={n_near0}/{len(ev)}  "
                      f"top5={[f'{x:.4f}' for x in top5]}")
                del model

    # 保存
    summary = {
        'config': dict(variants=VARIANTS_TO_RUN, seeds=SEEDS,
                       epochs=N_EPOCHS, freeze_free_epochs=P_FREEZE,
                       alpha_init=ALPHA_INIT, patience=PATIENCE),
        'results': {v: all_results[v] for v in VARIANTS_TO_RUN},
    }
    out_path = os.path.join(OUT_DIR, 'summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[✓] Results saved to {out_path}")


if __name__ == '__main__':
    main()
