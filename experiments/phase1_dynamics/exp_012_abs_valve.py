"""
exp_012_abs_valve.py — 绝对阀位替换差分阀位
=============================================
假设: Δ阀位在10s分辨率下无信息 → 改用绝对阀位
绝对阀位=当前喷水开度 → 模型可直接学到"开度30%→温降速率X"
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data
from world_model import WorldModel

DEVICE = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

ROLLOUT_K, BETA, BETA_WARMUP = 5, -0.3, 20
BS, STEPS = 256, 500
ROLLOUT_W = [1.0, 0.8, 0.6, 0.4, 0.2]


class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0., eps=1e-6):
        super().__init__(); self.beta = beta; self.eps = eps
    def forward(self, mu, lv, tgt):
        lv = torch.clamp(lv, -20., 20.)
        v = torch.exp(lv) + self.eps
        nll = 0.5 * (lv + (tgt - mu)**2 / v)
        if self.beta != 0: nll = v.detach()**self.beta * nll
        return nll.mean()


# ===== Data: load absolute valve positions =====
state_data, delta_actions, valve_abs = load_raw_data()
print(f"state_data: {state_data.shape}, valve_abs: {valve_abs.shape}")
print(f"valve_abs range: [{valve_abs[:,0].min():.4f}, {valve_abs[:,0].max():.4f}], "
      f"[{valve_abs[:,1].min():.4f}, {valve_abs[:,1].max():.4f}]")

# Use absolute positions instead of deltas
raw_data = np.concatenate([state_data, valve_abs], axis=1)  # [T, 13]

n_total = len(raw_data)
n_train = int(n_total * 0.70)
n_val_end = int(n_total * 0.85)
train_data = raw_data[:n_train]
val_data = raw_data[n_train:n_val_end]
test_data = raw_data[n_val_end:]
print(f"Data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")


# ===== Training =====
def train_epoch(model, raw, opt, crit):
    model.train(); W, K = cfg.WINDOW_SIZE, ROLLOUT_K; N = len(raw)
    total = 0.
    for _ in range(STEPS):
        idxs = np.random.randint(0, N-W-K, size=BS)
        xb, ab, sb = [], [], []
        for i in idxs:
            sw = raw[i:i+W, :cfg.N_STATE]; aw = raw[i:i+W, cfg.N_STATE:]
            xb.append(np.concatenate([sw, aw], 1))
            ab.append(raw[i+W:i+W+K, cfg.N_STATE:])
            sb.append(raw[i+W:i+W+K, :cfg.N_STATE])
        xt = torch.FloatTensor(np.stack(xb)).to(DEVICE)
        at = torch.FloatTensor(np.stack(ab)).to(DEVICE)
        st = torch.FloatTensor(np.stack(sb)).to(DEVICE)
        opt.zero_grad()
        ss = xt[:,:,:cfg.N_STATE]; aa = xt[:,:,cfg.N_STATE:]
        tsl = 0.
        for k in range(K):
            mu, lv = model(torch.cat([ss, aa], 2))
            sl = ROLLOUT_W[k] * crit(mu, lv, st[:,k])
            tsl += sl
            ss = torch.cat([ss[:,1:], mu.unsqueeze(1).detach()], 1)
            aa = torch.cat([aa[:,1:], at[:,k:k+1]], 1)
        tsl.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); total += tsl.item()
    return total/STEPS


@torch.no_grad()
def validate(model, raw, n=200):
    model.eval(); W, K = cfg.WINDOW_SIZE, ROLLOUT_K; N = len(raw)
    m0, m4 = 0., 0.
    for _ in range(n):
        i = np.random.randint(0, N-W-K)
        sw = raw[i:i+W, :cfg.N_STATE]; aw = raw[i:i+W, cfg.N_STATE:]
        xt = torch.FloatTensor(np.concatenate([sw, aw], 1)).unsqueeze(0).to(DEVICE)
        fa = torch.FloatTensor(raw[i+W:i+W+K, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
        st = torch.FloatTensor(raw[i+W:i+W+K, :cfg.N_STATE]).unsqueeze(0).to(DEVICE)
        tr = model.rollout(xt, fa, mode='sliding')
        m0 += (tr[0,0,cfg.TARGET_IDX]-st[0,0,cfg.TARGET_IDX]).abs().item()
        m4 += (tr[0,min(4,K-1),cfg.TARGET_IDX]-st[0,min(4,K-1),cfg.TARGET_IDX]).abs().item()
    return m0/n, m4/n


@torch.no_grad()
def eval_rollout(model, raw, H=18, n=500):
    model.eval(); W = cfg.WINDOW_SIZE; N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-H), n, replace=False)
    err = np.zeros((n, H))
    for j, i in enumerate(idxs):
        sw = raw[i:i+W, :cfg.N_STATE]; aw = raw[i:i+W, cfg.N_STATE:]
        xt = torch.FloatTensor(np.concatenate([sw, aw], 1)).unsqueeze(0).to(DEVICE)
        fa = torch.FloatTensor(raw[i+W:i+W+H, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
        tt = raw[i+W:i+W+H, cfg.TARGET_IDX]
        tr = model.rollout(xt, fa, mode='sliding')
        err[j] = np.abs(tr[0,:,cfg.TARGET_IDX].cpu().numpy()-tt)
    return err.mean(0)


@torch.no_grad()
def eval_sensitivity(model, raw, n=500):
    """Absolute valve: ±10% of range perturbation, not ±% of current value"""
    model.eval(); W = cfg.WINDOW_SIZE; N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-1), n, replace=False)
    res = {}
    # Absolute perturbation in valve units (valves range ~0-100%)
    abs_deltas = [-10., -5., -2., -1., 1., 2., 5., 10.]
    for adim in range(cfg.N_ACTION):
        r = {d: [] for d in abs_deltas}
        for i in idxs:
            sh = raw[i:i+W, :cfg.N_STATE]
            ah = raw[i:i+W, cfg.N_STATE:].copy()
            xt = torch.FloatTensor(np.concatenate([sh, ah], 1)).unsqueeze(0).to(DEVICE)
            mb, _ = model(xt); tb = mb[0, cfg.TARGET_IDX].item()
            for d in abs_deltas:
                ap = ah.copy()
                ap[-1, adim] += d  # absolute perturbation
                ap[-1, adim] = np.clip(ap[-1, adim], 0, 100)  # physical range
                xp = torch.FloatTensor(np.concatenate([sh, ap], 1)).unsqueeze(0).to(DEVICE)
                mp, _ = model(xp)
                r[d].append(mp[0, cfg.TARGET_IDX].item()-tb)
        res[f'action_{adim}'] = {d: (np.mean(v), np.std(v)) for d, v in r.items()}
    return res


@torch.no_grad()
def eval_multistep_sensitivity(model, raw, H=8, n=200):
    """多步敏感性: 扰动 step0 动作, 观察 step1-7 的 ΔT"""
    model.eval(); W = cfg.WINDOW_SIZE; N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-H), n, replace=False)
    abs_deltas = [-10., -5., -2., -1., 1., 2., 5., 10.]
    for adim in range(cfg.N_ACTION):
        name = ['喷水阀', '其他阀'][adim]
        print(f'\n  [{name} 多步扰动]')
        header = "  " + "".join([f"{'Δa':>8}   "]) + "".join([f"{f't={s}':>8}  " for s in [1,2,3,5,7]])
        print(header)
        for d in abs_deltas:
            dT_steps = {s: [] for s in range(1, H)}
            for i in idxs:
                sh = raw[i:i+W, :cfg.N_STATE].copy()
                ah = raw[i:i+W, cfg.N_STATE:].copy()
                fa = raw[i+W:i+W+H, cfg.N_STATE:].copy()
                # Baseline
                xt = torch.FloatTensor(np.concatenate([sh, ah], 1)).unsqueeze(0).to(DEVICE)
                at = torch.FloatTensor(fa).unsqueeze(0).to(DEVICE)
                trb = model.rollout(xt, at, mode='sliding')
                bp = trb[0,:,cfg.TARGET_IDX].cpu().numpy()
                # Perturbed
                ap = ah.copy(); ap[-1, adim] = np.clip(ap[-1, adim] + d, 0, 100)
                fap = fa.copy(); fap[0, adim] = np.clip(fap[0, adim] + d, 0, 100)
                xt = torch.FloatTensor(np.concatenate([sh, ap], 1)).unsqueeze(0).to(DEVICE)
                at = torch.FloatTensor(fap).unsqueeze(0).to(DEVICE)
                trp = model.rollout(xt, at, mode='sliding')
                pp = trp[0,:,cfg.TARGET_IDX].cpu().numpy()
                for s in range(1, H):
                    dT_steps[s].append(pp[s] - bp[s])
            # Print row
            row = f'  {d:>+8.1f}  '
            for s in [1,2,3,5,7]:
                arr = np.array(dT_steps[s])
                row += f'{np.mean(arr):>+8.4f}  '
            print(row)


# ===== Main =====
def main():
    # Standard model, same as exp_006
    model = WorldModel(
        n_state=cfg.N_STATE, n_action=cfg.N_ACTION,
        window_size=cfg.WINDOW_SIZE, d_model=cfg.D_MODEL,
        n_heads=cfg.N_HEADS, n_var_layers=cfg.N_VAR_LAYERS,
        n_tcn_layers=cfg.N_TCN_LAYERS, dropout=cfg.DROPOUT,
        rollout_mode='sliding', probabilistic=True,
    ).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    exp_dir = "results/exp_012_absvalve"
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)

    crit = BetaNLLLoss(beta=BETA)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

    best_m, pc, be = float('inf'), 0, 0; t0 = time.time()
    for ep in range(1, cfg.EPOCHS + 1):
        crit.beta = 0. if ep <= BETA_WARMUP else BETA * min((ep - BETA_WARMUP) / 10, 1.)
        tl = train_epoch(model, train_data, opt, crit)
        v0, v4 = validate(model, val_data); sched.step(v4)
        if ep % 10 == 0 or ep == 1:
            print(f"  E{ep:3d} | NLL:{tl:7.0f} | V0:{v0:.4f} | V4:{v4:.4f}")
        if v4 < best_m - 0.001: best_m, be, pc = v4, ep, 0; torch.save(
            {'epoch':ep, 'model_state_dict':model.state_dict()},
            f"{exp_dir}/checkpoints/best_model.pth")
        else: pc += 1
        if pc >= cfg.EARLY_STOPPING_PATIENCE: print(f"  Stop@{ep} best@{be}"); break

    ck = torch.load(f"{exp_dir}/checkpoints/best_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    print(f"\nTrain: {(time.time()-t0)/60:.1f}min Best@{be}")

    # Rollout
    mae = eval_rollout(model, test_data)
    print(f"\nRollout(test): {mae[0]:.4f}→{mae[-1]:.4f} (×{mae[-1]/mae[0]:.1f})")

    # 1-step sensitivity (absolute perturbation)
    sens = eval_sensitivity(model, test_data)
    print(f"\n1-step Sensitivity:")
    for adim in range(2):
        name = ['喷水阀', '其他阀'][adim]
        r = sens[f'action_{adim}']
        print(f"  [{name}]")
        for d in [-10, -5, -2, -1, 1, 2, 5, 10]:
            m, s = r[d]; print(f"    Δ={d:+4.1f} → ΔT={m:+.4f}°C (±{s:.4f})")

    # Multi-step sensitivity
    print(f"\nMulti-step Sensitivity:")
    eval_multistep_sensitivity(model, test_data)

    # Save
    result = {'params': sum(p.numel() for p in model.parameters()),
              'best_epoch': be, 'rollout_mae': mae.tolist(),
              'sensitivity_1step': {k: {str(d): list(v) for d, v in s.items()} for k, s in sens.items()}}
    with open(f"{exp_dir}/results.json", 'w') as f: json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
