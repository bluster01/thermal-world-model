#!/usr/bin/env python3
"""
exp_106_causal_arch.py — 强动作因果架构训练与消融 (2026-08-05)
==============================================================
设计稿: docs/causal_eval_framework.md  L4 + §6

变体 (--variant):
  A1mlp     残差分解 + 乘性门控 (A1+A2)
  A1phys    残差分解 + 一阶惯性结构化响应 (A1+A3)
  A1both    A3 物理主干 + A1 小残差修正
  A1mlp_cs  A1mlp + C1 增量累积输出
  A1physcs  A1phys + C1 增量累积输出
  B1glb     TimeXer + GLB-only head (强制动作通路)
  B1flat    TimeXer + FlattenHead (复现现 M9DSP, B1 的对照)

协议 (全变体固定, 保证消融可归因):
  W=96, BS=256, STEPS=500/ep, AdamW LR 1e-3 WD 1e-5, ReduceLROnPlateau
  动作构造一律走 causal_eval.build_action (L0/P0.1)
  事件只取 test 区间 (修 exp_100-102 的因果指标训练集泄漏)
  双 checkpoint: best-MAE 与 best-CAUSAL 各存一份 (L0/P0.5)

用法:
  python exp_106_causal_arch.py --variant A1phys --seed 0
  python exp_106_causal_arch.py --variant A1phys --seed 0 --smoke
  python exp_106_causal_arch.py --all --seeds 0,1,2,3,4          # 串行跑全部
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

import causal_eval as CE
import causal_arch as CA

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = E.cfg.WINDOW_SIZE
N_FEAT = E.N_FEAT
TARGET_IDX = E.TARGET_IDX
n_train, n_val_end = 495407, 601566
OUT_ROOT = 'results/exp_106_causal_arch'

VARIANTS = {
    'A1mlp':    dict(kind='res', intervention='mlp',  cumsum_out=False, H=60),
    'A1phys':   dict(kind='res', intervention='phys', cumsum_out=False, H=60),
    'A1both':   dict(kind='res', intervention='both', cumsum_out=False, H=60),
    'A1mlp_cs': dict(kind='res', intervention='mlp',  cumsum_out=True,  H=60),
    'A1physcs': dict(kind='res', intervention='phys', cumsum_out=True,  H=60),
    'B1glb':    dict(kind='timexer', head_mode='glb',     H=60),
    'B1flat':   dict(kind='timexer', head_mode='flatten', H=60),
}

PROFILE_K = [(2, '30s'), (5, '60s'), (11, '120s'), (17, '180s'),
             (29, '300s'), (41, '420s'), (59, '600s')]


def build_model(variant, H):
    v = VARIANTS[variant]
    if v['kind'] == 'res':
        return CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H,
                                   intervention=v['intervention'],
                                   cumsum_out=v['cumsum_out'], probabilistic=True)
    return CA.TimeXerCausalWM(N_FEAT, TARGET_IDX, H,
                              head_mode=v['head_mode'], probabilistic=True)


# ===================================================== 数据
raw = E.data_all
N = len(raw)
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_LD = E.NUMERIC_COLS.index('机组负荷')
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)
I_DSP = 40
train_raw = raw41[:n_train]
test_raw = raw41[n_val_end:]


def make_batch(H, bs, rng):
    """训练取样。动作一律走 CE.build_action → 与评测同构 (L0/P0.1)。"""
    idxs = rng.integers(0, len(train_raw) - W - H, size=bs)
    X = np.stack([train_raw[i:i + W, :N_FEAT] for i in idxs])
    A = np.stack([CE.build_action(train_raw, int(i), W, H, I_DSP) for i in idxs])
    Y = np.stack([train_raw[i + W:i + W + H, TARGET_IDX] for i in idxs])
    return (torch.from_numpy(X).float().to(DEVICE),
            torch.from_numpy(A).float().unsqueeze(-1).to(DEVICE),
            torch.from_numpy(Y).float().to(DEVICE))


@torch.no_grad()
def eval_mae(model, H, n=200, seed=0):
    model.eval()
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, len(test_raw) - W - H, size=n)
    errs = []
    for i in idxs:
        x = torch.from_numpy(test_raw[i:i + W, :N_FEAT]).float().unsqueeze(0).to(DEVICE)
        a = torch.from_numpy(CE.build_action(test_raw, int(i), W, H, I_DSP))
        mu, _ = model(x, a.reshape(1, H, 1).to(DEVICE))
        errs.append(np.abs(mu[0].cpu().numpy() - test_raw[i + W:i + W + H, TARGET_IDX]).mean())
    return float(np.mean(errs))


@torch.no_grad()
def eval_causal(model, H, gt=None):
    """因果评测。事件只取 test 区间 (lo=n_val_end) → 无训练集泄漏。

    gt: exp_104 产出的 DiD 真值 dict; None 时退化为 sign(ΔSP) 口径 (仅供训练期监控)。
    """
    model.eval()
    wrap = CE.ModelWrapper(model, raw, raw41, W, H, I_DSP, DEVICE)
    ev, dv = CE.select_events(raw, I_SP, I_LD, H=H, lo=n_val_end, W=W)
    if len(ev) == 0:
        return None
    m, _, keep = CE.model_response(wrap, ev, dv)
    dvk = dv[keep]
    prof = {}
    for k, lab in PROFILE_K:
        if k >= H:
            continue
        resp_c = m[:, k] * dvk                       # 还原为 °C
        prof[lab] = dict(k=int(k),
                         gain_norm=float(m[:, k].mean()),
                         resp_abs=float(np.abs(resp_c).mean()),
                         dir_dsp=float((np.sign(resp_c) == np.sign(dvk)).mean()))
    out = dict(n_ev=int(len(dvk)), profile=prof)
    if gt is not None:
        gtk = gt.get(f'H{H}', gt)   # exp_104 JSON: {H60: {...}, H18: {...}}; 退化用顶层级
        R = np.array(gtk['R_true'], dtype=np.float32)
        ceil = np.array(gtk['sgn_ceiling'], dtype=np.float32)
        r = np.array(gtk.get('r', np.zeros((len(m), H))), dtype=np.float32)
        if len(r) == len(m):                        # 事件集一致才配对
            ks = [(k, l) for k, l in PROFILE_K if k < H]
            out['cfe'] = CE.causal_metrics(m, r, R, ceil, ks)
            out['cfi'] = CE.cfi(out['cfe'], ks[-1][1])
    if 'cfi' not in out:
        # 退化 score: 归一化增益接近 1 且方向高 (仅训练期 checkpoint 选择用)
        lab = '600s' if H == 60 else '180s'
        g = prof[lab]['gain_norm']
        out['cfi'] = float(0.5 * min(max(g, 0.0), 1.0 / max(g, 1e-6)) +
                           0.5 * prof[lab]['dir_dsp'])
    return out


# ===================================================== 训练
class SafeBetaNLL(torch.nn.Module):
    """β=0 标准高斯 NLL + σ clamp (H=60 下 β<0 膨胀会数值不稳, 见 debug_nan_h60)。"""

    def __init__(self, beta=0.0):
        super().__init__(); self.beta = beta

    def forward(self, mu, lv, tgt, w=None):
        lv = torch.clamp(lv, -6., 20.)
        v = torch.exp(lv) + 1e-4
        nll = 0.5 * (lv + (tgt - mu) ** 2 / v)
        if self.beta != 0:
            nll = v.detach() ** self.beta * nll
        if w is not None:
            nll = nll * w
        return nll.mean()


def train_one(variant, seed, smoke=False, gt=None, epochs=None, flat_weight=False,
              patience=20, min_delta=1e-4, h_override=None, loss_type='nll'):
    H = h_override if h_override is not None else VARIANTS[variant]['H']
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    outdir = os.path.join(OUT_ROOT, f'{variant}_s{seed}')
    if h_override is not None:
        outdir = os.path.join(OUT_ROOT, f'{variant}_H{h_override}_s{seed}')
    if flat_weight:
        outdir += '_flatw'
    if loss_type != 'nll':
        outdir += f'_{loss_type}'
    os.makedirs(os.path.join(outdir, 'checkpoints'), exist_ok=True)

    model = build_model(variant, H).to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters())
    CA.check_zero_action_identity(model, N_FEAT, H, DEVICE)
    print(f"[{variant} s{seed}] {n_param/1e6:.2f}M params | g(x,0)=0 自检 PASS")

    BS, STEPS = 256, 500
    NEPOCH = 4 if smoke else (epochs or E.cfg.EPOCHS)
    if loss_type == 'mae':
        crit = torch.nn.L1Loss()
    elif loss_type == 'huber':
        crit = torch.nn.SmoothL1Loss()
    else:
        crit = SafeBetaNLL(beta=0.0)
    use_nll = (loss_type == 'nll')
    opt = torch.optim.AdamW(model.parameters(), lr=E.cfg.LEARNING_RATE,
                            weight_decay=E.cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    tw = torch.ones(H, device=DEVICE) if flat_weight else \
        torch.from_numpy(np.linspace(1.0, 0.6, H).astype(np.float32)).to(DEVICE)

    best = dict(mae=None, mae_ep=None, cfi=None, cfi_ep=None)
    curve, wait = [], 0
    t0 = time.time()
    for ep in range(1, NEPOCH + 1):
        model.train()
        losses = []
        for _ in range(STEPS):
            x, a, y = make_batch(H, BS, rng)
            mu, lv = model(x, a)
            if use_nll:
                loss = crit(mu, lv, y, tw)
            else:
                loss = crit(mu, y)  # MAE/Huber: 只用 mu
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        mae = eval_mae(model, H)
        cz = eval_causal(model, H, gt)
        cfi = cz['cfi'] if cz else float('nan')
        sched.step(mae)
        curve.append(dict(ep=ep, loss=float(np.mean(losses)), mae=mae, cfi=cfi))

        tag = []
        if best['mae'] is None or mae < best['mae'] - min_delta:
            best.update(mae=mae, mae_ep=ep)
            torch.save({'model_state_dict': model.state_dict(), 'ep': ep, 'mae': mae,
                        'cfi': cfi, 'variant': variant, 'seed': seed, 'H': H},
                       os.path.join(outdir, 'checkpoints', 'best_mae.pth'))
            tag.append('*MAE')
            wait = 0
        else:
            wait += 1
        if np.isfinite(cfi) and (best['cfi'] is None or cfi > best['cfi']):
            best.update(cfi=cfi, cfi_ep=ep)
            torch.save({'model_state_dict': model.state_dict(), 'ep': ep, 'mae': mae,
                        'cfi': cfi, 'variant': variant, 'seed': seed, 'H': H},
                       os.path.join(outdir, 'checkpoints', 'best_causal.pth'))
            tag.append('*CFI')
        g600 = cz['profile'].get('600s', cz['profile'].get('180s', {})).get('gain_norm', float('nan'))
        print(f"  ep{ep:3d} loss {np.mean(losses):7.4f} | MAE {mae:.4f} | CFI {cfi:.3f} "
              f"| gain {g600:+.3f} | wait {wait:2d} {' '.join(tag)}")

        if not smoke and wait >= patience:
            print(f"  → 早停: MAE 连续 {patience} epoch 无改善 (best {best['mae']:.4f}@ep{best['mae_ep']})")
            break

    CA.check_zero_action_identity(model, N_FEAT, H, DEVICE)     # 训练后不变量仍须成立
    res = dict(variant=variant, seed=seed, H=H, n_param=int(n_param),
               best=best, curve=curve, final_causal=eval_causal(model, H, gt),
               minutes=round((time.time() - t0) / 60, 1))
    with open(os.path.join(outdir, 'result.json'), 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[{variant} s{seed}] done {res['minutes']}min | "
          f"best MAE {best['mae']:.4f}@ep{best['mae_ep']} | best CFI {best['cfi']:.3f}@ep{best['cfi_ep']}")
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return res


# ===================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='A1phys', choices=list(VARIANTS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seeds', default=None, help='逗号分隔, 覆盖 --seed')
    ap.add_argument('--all', action='store_true', help='跑全部变体')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--flat-weight', action='store_true', help='L5: 时间权重全 1')
    ap.add_argument('--h', type=int, default=None, help='L5: 覆盖 H (默认取 VARIANTS 定义)')
    ap.add_argument('--gt', default='results/cfe_groundtruth/did_response.json',
                    help='exp_104 的 DiD 真值; 不存在则退化为 sign(ΔSP) 口径')
    ap.add_argument('--loss', default='nll', choices=['nll', 'mae', 'huber'],
                    help='P3: 损失函数 (nll/mae/huber)')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    gt = None
    if os.path.exists(args.gt):
        with open(args.gt) as f:
            gt = json.load(f)
        print(f"[gt] 载入 DiD 真值 {args.gt} (n_ev={gt.get('n_ev')})")
    else:
        print(f"[gt] {args.gt} 不存在 → CFI 退化为 sign(ΔSP) 口径 (仅供 ckpt 选择, "
              f"最终结论须等 exp_104)")

    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else [args.seed]
    variants = list(VARIANTS) if args.all else [args.variant]
    os.makedirs(OUT_ROOT, exist_ok=True)

    print(f"[protocol] W={W} N_FEAT={N_FEAT} TARGET_IDX={TARGET_IDX} DEVICE={DEVICE}")
    CE.assert_train_eval_identity(train_raw, raw41, W, 60, I_DSP, n_probe=200)
    CE.assert_train_eval_identity(test_raw, test_raw, W, 60, I_DSP, n_probe=200)
    print("[protocol] L0/P0.2 动作编码往返一致性 PASS")

    allres = []
    for v in variants:
        for s in seeds:
            allres.append(train_one(v, s, args.smoke, gt, args.epochs, args.flat_weight,
                                   h_override=args.h, loss_type=args.loss))

    with open(os.path.join(OUT_ROOT, 'summary.json'), 'w') as f:
        json.dump(allres, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print(f"{'变体':10s} {'seed':>4s} {'best MAE':>9s} {'best CFI':>9s} {'末点gain':>9s} {'方向':>7s}")
    print("=" * 78)
    for r in allres:
        fc = r['final_causal']
        lab = '600s' if r['H'] == 60 else '180s'
        p = fc['profile'][lab] if fc else {}
        print(f"{r['variant']:10s} {r['seed']:4d} {r['best']['mae']:9.4f} "
              f"{r['best']['cfi']:9.3f} {p.get('gain_norm', float('nan')):9.3f} "
              f"{p.get('dir_dsp', float('nan'))*100:6.0f}%")
    print("\n物理参考: 600s 归一化增益 ≈ 0.97, 180s ≈ 0.17 (exp_099)")
    print("注意: DiD 真值 (exp_104) 就位前, gain/方向仍是 sign(ΔSP) 口径, 不作最终结论。")


if __name__ == '__main__':
    main()
