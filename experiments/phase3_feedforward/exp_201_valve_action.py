#!/usr/bin/env python3
"""
exp_201_valve_action.py — Phase 3.5: 阀位 action 的双分支模型验证
===================================================================
设计稿: docs/PHASE35_DESIGN.md

Fix v2: action 从绝对阀位改为 Δvalve（一阶差分），使 cumsum(Δvalve)
       恢复阀位偏差轨迹，物理含义与ΔSP→cumsum→SP偏差一致。
       外加 g_plant 输出 clamp[-15, +15]°C 防止递归发散。

Fix v3 (abs mode): --action-mode abs 用绝对阀位(去训练集中位数)作为 action，
       integrate=False 不做 cumsum。保留绝对工作点信息(阀门开度-流量非线性:
       30% 处 +1% 与 80% 处 +1% 流量增量不同, Δvalve 表示无法区分)。
       去中位数保证 g(x,0)=0 语义: 阀位处于典型开度时干预项为 0。

用法:
  python exp_201_valve_action.py --variant A1phys_valve --seed 0 --ff 10
  python exp_201_valve_action.py --variant A1phys_valve --seed 0 --ff 10 --action-mode abs
  python exp_201_valve_action.py --variant A1phys_valve_noff --seed 0 --ff 0
  python exp_201_valve_action.py --variant A1phys_null_valve --seed 0
"""

import os, sys, json, time, argparse
import numpy as np; import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_proj = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _proj)
sys.path.insert(0, os.path.join(_proj, 'experiments', 'phase1_dynamics'))

import causal_arch as CA
import causal_eval as CE

from exp_025_unified_benchmark import cfg as E_cfg, data_all, N_FEAT, TARGET_IDX, NUMERIC_COLS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = E_cfg.WINDOW_SIZE  # 96
H = 60

n_train = 495407
n_val_end = 601566

I_V2 = NUMERIC_COLS.index('二级减温调节门阀位')   # valve position
I_SP = NUMERIC_COLS.index('二级减温调节阀设定')    # for event selection only

OUT_ROOT = 'results/exp_201_valve_action'

# ─── Data: Δvalve / absolute valve / flow-proportional valve as action ───
raw = data_all
dvalve_col = np.diff(raw[:, I_V2], prepend=raw[0, I_V2])  # Δvalve
dsp_col    = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])    # ΔSP (events only)
v_med_train = float(np.median(raw[:n_train, I_V2]))        # 训练集阀位中位数 (abs 模式参考点)
absvalve_col = raw[:, I_V2] - v_med_train                  # 绝对阀位去中位数
# 等百分比流量特性先验 (用户: 阀门开度-流量非线性): F/Fmax = R^(V/100 - 1), R=50
# u_flow ∈ [0, 1], 去训练集中位数保持 g(x,0)=0 语义 (典型开度→流量中位→干预 0)
R_FLOW = 50.0
v_pct = np.clip(raw[:, I_V2], 0.0, 100.0) / 100.0
flow_col = (R_FLOW ** (v_pct - 1.0) - 1.0 / R_FLOW) / (1.0 - 1.0 / R_FLOW)  # [0,1]
flow_med_train = float(np.median(flow_col[:n_train]))
flow_col = flow_col - flow_med_train                                           # centered
raw42 = np.concatenate([raw, dvalve_col[:, None], absvalve_col[:, None],
                        flow_col[:, None], dsp_col[:, None]], 1)
I_DVALVE = 40  # Δvalve column
I_AVALVE = 41  # absolute valve (centered) column
I_FLOW   = 42  # equal-percentage flow (centered) column
I_DSP    = 43  # ΔSP column (events only)

train_raw = raw42[:n_train]
test_raw  = raw42[n_val_end:]

# ─── Action construction ───
def valve_to_flow(v):
    """等百分比流量变换: F/Fmax = R^(V/100-1), 归一化到 [0,1] 再去中位数。"""
    v_pct = np.clip(np.asarray(v, dtype=np.float64), 0.0, 100.0) / 100.0
    f = (R_FLOW ** (v_pct - 1.0) - 1.0 / R_FLOW) / (1.0 - 1.0 / R_FLOW)
    return (f - flow_med_train).astype(np.float32)


def build_valve_action(raw42, s, W_val, H_val, override=None, mode='delta'):
    """Read Δvalve (delta), centered absolute valve (abs) or flow-proportional
    valve (flow) as action.
    Delta mode: model's forward() cumsums internally → valve deviation trajectory.
    Abs/flow: integrate=False → u = a directly, absolute level retained."""
    if override is not None:
        a = np.asarray(override, dtype=np.float32)
        if a.ndim == 0: a = np.full(H_val, float(a), dtype=np.float32)
        return a
    if mode == 'flow':
        # 扰动在开度空间 → 变换到流量空间 (物理语义: 操作变量是开度)
        v_win = raw42[s + W_val:s + W_val + H_val, I_V2]
        return valve_to_flow(v_win)
    icol = I_DVALVE if mode == 'delta' else I_AVALVE
    return raw42[s + W_val:s + W_val + H_val, icol].astype(np.float32)


def make_batch(H_val, bs, rng, mode='delta'):
    idxs = rng.integers(0, len(train_raw) - W - H_val, size=bs)
    X = np.stack([train_raw[i:i + W, :N_FEAT] for i in idxs])
    A = np.stack([build_valve_action(train_raw, int(i), W, H_val, mode=mode) for i in idxs])
    Y = np.stack([train_raw[i + W:i + W + H_val, TARGET_IDX] for i in idxs])
    return (torch.from_numpy(X).float().to(DEVICE),
            torch.from_numpy(A).float().unsqueeze(-1).to(DEVICE),
            torch.from_numpy(Y).float().to(DEVICE))


@torch.no_grad()
def eval_mae(model, H_val, n=200, seed=0, mode='delta'):
    model.eval()
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, len(test_raw) - W - H_val, size=n)
    errs = []
    for i in idxs:
        i_int = int(i)
        x = torch.from_numpy(test_raw[i_int:i_int + W, :N_FEAT]).float().unsqueeze(0).to(DEVICE)
        a = build_valve_action(test_raw, i_int, W, H_val, mode=mode)
        mu, _ = model(x, torch.from_numpy(a).float().reshape(1, H_val, 1).to(DEVICE))
        gt = test_raw[i_int + W:i_int + W + H_val, TARGET_IDX]
        errs.append(np.abs(mu[0].cpu().numpy() - gt).mean())
    return float(np.mean(errs))


@torch.no_grad()
def eval_jacobian(model, H_val, n=100, seed=42, delta=0.1, mode='delta'):
    """Finite-difference direction test. valve↑ → T↓ = correct physics
    delta mode: δ=0.1/step, cumsum → 6% cumulative valve deviation.
    abs/flow:   δ=5.0 whole-window valve +5% (flow mode: perturb in OPENING
    space V±δ, then transform to flow space)."""
    model.eval()
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, len(test_raw) - W - H_val, size=n)
    neg = pos = zero = 0
    for i in idxs:
        i_int = int(i)
        x = torch.from_numpy(test_raw[i_int:i_int + W, :N_FEAT]).float().unsqueeze(0).to(DEVICE)
        if mode == 'flow':
            v_win = raw[i_int:i_int + W + H_val, I_V2]
            a_up = valve_to_flow(np.clip(v_win[W:], 0, 100) + delta)
            a_dn = valve_to_flow(np.clip(v_win[W:], 0, 100) - delta)
        else:
            a = build_valve_action(test_raw, i_int, W, H_val, mode=mode)
            a_up = a.copy(); a_up += delta
            a_dn = a.copy(); a_dn -= delta
        mu_up, _ = model(x, torch.from_numpy(a_up).float().reshape(1, H_val, 1).to(DEVICE))
        mu_dn, _ = model(x, torch.from_numpy(a_dn).float().reshape(1, H_val, 1).to(DEVICE))
        diff = (mu_up - mu_dn).mean().item()
        if abs(diff) < 1e-6: zero += 1
        elif diff < 0: neg += 1
        else: pos += 1
    return dict(neg=neg/n, pos=pos/n, zero=zero/n, n=n)


# ─── Model ───
def build_model(variant, mode='delta'):
    if variant in ('A1phys_valve', 'A1phys_valve_noff', 'A1phys_null_valve'):
        # delta mode: K_init=0.01 (valve K ≈ 0.01× SP K, u ∈ ±4%)
        # abs mode:   K_init=0.002 (u ∈ ±30% centered, sig ≈ ±0.06 normalized)
        # flow mode:  K_init=0.05 (u ∈ [-0.024, 0.98], sig ≈ ±0.05 normalized)
        k_init = {'delta': 0.01, 'abs': 0.002, 'flow': 0.05}[mode]
        return CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H,
                                   intervention='phys', cumsum_out=False,
                                   probabilistic=True, n_lag=2,
                                   free_head_type='mlp' if variant != 'A1phys_null_valve' else None,
                                   alpha_init=0.0,
                                   clamp_interv=15.0,      # ±15°C output bound
                                   k_init=k_init,
                                   integrate=(mode == 'delta'))
    else:
        raise ValueError(f'Unknown variant: {variant}')


class SafeBetaNLL(torch.nn.Module):
    def __init__(self, beta=0.0):
        super().__init__(); self.beta = beta

    def forward(self, mu, lv, tgt, w=None):
        lv = torch.clamp(lv, -6., 20.)
        v = torch.exp(lv) + 1e-4
        nll = 0.5 * (lv + (tgt - mu) ** 2 / v)
        if self.beta != 0: nll = v.detach() ** self.beta * nll
        if w is not None: nll = nll * w
        return nll.mean()


def train_one(variant, seed, smoke=False, epochs=None, freeze_free_epochs=0, mode='delta'):
    if smoke: epochs = 3
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    suffix = f'_s{seed}'
    if freeze_free_epochs > 0: suffix += f'_ff{freeze_free_epochs}'
    if mode != 'delta': suffix += f'_{mode}'
    outdir = os.path.join(OUT_ROOT, f'{variant}{suffix}')
    os.makedirs(outdir, exist_ok=True)

    model = build_model(variant, mode=mode).to(DEVICE)
    print(f'{variant} seed={seed}: {sum(p.numel() for p in model.parameters()):,} params')

    if freeze_free_epochs > 0 and hasattr(model, 'free_head') and model.free_head is not None:
        for p in model.free_head.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    criterion = SafeBetaNLL(beta=0.0)

    max_epochs = epochs if epochs else (20 if smoke else 150)
    patience = 5 if smoke else 20
    best_mae = float('inf'); best_cfi = 0.0
    best_mae_ep = 0; best_cfi_ep = 0
    curve = []

    for ep in range(1, max_epochs + 1):
        if freeze_free_epochs > 0 and ep == freeze_free_epochs + 1 and hasattr(model, 'free_head') and model.free_head is not None:
            for p in model.free_head.parameters(): p.requires_grad = True
            for pg in opt.param_groups: pg['lr'] = 1e-3

        model.train()
        opt.zero_grad()
        X, A, Y = make_batch(H, 256, rng, mode=mode)
        mu, lv = model(X, A)
        loss = criterion(mu, lv, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        scheduler.step(loss.item())

        if ep % 5 == 0 or ep == 1:
            mae = eval_mae(model, H, n=50, seed=seed, mode=mode)
            jac = eval_jacobian(model, H, n=30, seed=seed,
                                delta=0.1 if mode == 'delta' else 5.0, mode=mode)
            cfi_fallback = 0.5 * jac['neg'] + 0.5 * (1.0 if mae < 1.5 else 0.5)

            curve.append(dict(ep=ep, loss=loss.item(), mae=mae, cfi=cfi_fallback,
                             jac_neg=jac['neg'], jac_pos=jac['pos'], jac_zero=jac['zero']))

            if mae < best_mae:
                best_mae = mae; best_mae_ep = ep
                torch.save(model.state_dict(), os.path.join(outdir, 'best_mae.pth'))
            if cfi_fallback > best_cfi:
                best_cfi = cfi_fallback; best_cfi_ep = ep
                torch.save(model.state_dict(), os.path.join(outdir, 'best_cfi.pth'))

            print(f'  ep{ep:3d} loss={loss.item():.4f} mae={mae:.4f} '
                  f'jac:neg={jac["neg"]:.1%} pos={jac["pos"]:.1%} z={jac["zero"]:.1%}')

        if (ep - best_mae_ep > patience and ep > max(freeze_free_epochs, 0) + 5):
            print(f'  early stop at ep {ep}')
            break

    model.load_state_dict(torch.load(os.path.join(outdir, 'best_cfi.pth')))
    final_mae = eval_mae(model, H, n=200, seed=99, mode=mode)
    final_jac = eval_jacobian(model, H, n=100, seed=99,
                              delta=0.1 if mode == 'delta' else 5.0, mode=mode)

    result = dict(variant=variant, seed=seed, H=H, mode=mode,
                  v_med_train=v_med_train,
                  best=dict(mae=best_mae, mae_ep=best_mae_ep, cfi=best_cfi, cfi_ep=best_cfi_ep),
                  final=dict(mae=final_mae, jac_neg=final_jac['neg'],
                             jac_pos=final_jac['pos'], jac_zero=final_jac['zero']),
                  curve=curve)
    with open(os.path.join(outdir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=float)

    print(f'  DONE: best_mae={best_mae:.4f} best_cfi={best_cfi:.4f} '
          f'final_mae={final_mae:.4f} final_jac_neg={final_jac["neg"]:.1%}')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='A1phys_valve')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ff', type=int, default=10)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--action-mode', default='delta', choices=['delta', 'abs', 'flow'],
                    help='delta: Δvalve + cumsum (v3 baseline); abs: absolute valve, '
                         'no cumsum; flow: equal-percentage flow transform (R=50)')
    args = ap.parse_args()

    variants = [args.variant] if not args.all else ['A1phys_valve', 'A1phys_valve_noff', 'A1phys_null_valve']
    seeds = [args.seed] if not args.all else [int(s) for s in args.seeds.split(',')]

    for v in variants:
        for s in seeds:
            print(f'\n{"="*60}\n{v} seed={s} mode={args.action_mode}\n{"="*60}')
            train_one(v, s, smoke=args.smoke, freeze_free_epochs=args.ff, mode=args.action_mode)
