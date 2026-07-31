"""
exp_015_dualvalve_signreg.py — 双阀符号正则 + 多步监督
=======================================================
基于 exp_013 (符号正则) + exp_014B (多步权重) 的组合:

1. 符号正则约束 BOTH 阀门: 一级/二级减温阀 开阀→降温 (∂T/∂a < 0)
2. 多步权重 [1.0, 0.6, 1.0, 0.6, 1.2]: 强化第3/5步预测监督
3. 绝对阀位 (exp_012 验证有效)

物理依据 (互相关分析):
  二级减温阀 → 二级减温器出口温度: lead10s r=-0.035 (短时程物理响应明确)
  二级减温阀 → 主汽温: lead10s r=-0.024, lead120s r=+0.058 (短负长正)
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

ROLLOUT_K, BETA, BETA_WARMUP = 5, -0.3, 20
BS, STEPS = 256, 500
ROLLOUT_W = [1.0, 0.6, 1.0, 0.6, 1.2]     # 多步监督权重 (强化 3/5 步)
SIGN_LAMBDA = 0.1                          # 符号损失权重
SIGN_DELTA = 5.0                           # 扰动幅度 (绝对阀位 %)
SIGN_WARMUP = 20                           # 符号损失 warmup

# 约束两个阀: adim=0 (一级), adim=1 (二级) 都要求 开阀→降温
CONSTRAINED_DIMS = [0, 1]


class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0., eps=1e-6):
        super().__init__(); self.beta = beta; self.eps = eps
    def forward(self, mu, lv, tgt):
        lv = torch.clamp(lv, -20., 20.)
        v = torch.exp(lv) + self.eps
        nll = 0.5 * (lv + (tgt - mu)**2 / v)
        if self.beta != 0: nll = v.detach()**self.beta * nll
        return nll.mean()


# ===== Data =====
state_data, delta_actions, valve_abs = load_raw_data()
raw_data = np.concatenate([state_data, valve_abs], axis=1)
n_total = len(raw_data)
n_train = int(n_total * 0.70); n_val_end = int(n_total * 0.85)
train_data, val_data = raw_data[:n_train], raw_data[n_train:n_val_end]
test_data = raw_data[n_val_end:]
print(f"Data: {len(train_data)}+{len(val_data)}+{len(test_data)} | 绝对阀位")
print(f"Constrained dims: {CONSTRAINED_DIMS} | λ={SIGN_LAMBDA} Δ={SIGN_DELTA}% "
      f"weights={ROLLOUT_W}")


# ===== Training =====
def train_epoch(model, raw, opt, crit, epoch):
    model.train(); W, K = cfg.WINDOW_SIZE, ROLLOUT_K; N = len(raw)
    total_nll, total_sign = 0., 0.
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

        # NLL rollout (多步权重)
        ss = xt[:,:,:cfg.N_STATE]; aa = xt[:,:,cfg.N_STATE:]
        nll_loss = 0.
        for k in range(K):
            mu, lv = model(torch.cat([ss, aa], 2))
            nll_loss += ROLLOUT_W[k] * crit(mu, lv, st[:,k])
            ss = torch.cat([ss[:,1:], mu.detach().unsqueeze(1)], 1)
            aa = torch.cat([aa[:,1:], at[:,k:k+1]], 1)

        # 符号正则: 双阀 (开阀→降温)
        sign_loss = torch.tensor(0., device=DEVICE)
        if epoch > SIGN_WARMUP:
            ss0 = xt[:,:,:cfg.N_STATE]
            aa0 = xt[:,:,cfg.N_STATE:].clone()
            mu_orig, _ = model(torch.cat([ss0, aa0], 2))
            t_orig = mu_orig[:, cfg.TARGET_IDX]
            for adim in CONSTRAINED_DIMS:
                aa_open = aa0.clone()
                aa_open[:, -1, adim] += SIGN_DELTA
                mu_open, _ = model(torch.cat([ss0, aa_open], 2))
                t_open = mu_open[:, cfg.TARGET_IDX]
                aa_close = aa0.clone()
                aa_close[:, -1, adim] -= SIGN_DELTA
                mu_close, _ = model(torch.cat([ss0, aa_close], 2))
                t_close = mu_close[:, cfg.TARGET_IDX]
                # 开阀应降温 (t_open < t_orig), 关阀应升温 (t_close > t_orig)
                sign_loss += F.relu(t_open - t_orig).mean() + F.relu(t_orig - t_close).mean()

        loss = nll_loss + SIGN_LAMBDA * sign_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        total_nll += nll_loss.item(); total_sign += sign_loss.item()
    return total_nll / STEPS, total_sign / STEPS


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
def eval_sensitivity(model, raw, n=200):
    model.eval(); W = cfg.WINDOW_SIZE; N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-13), n, replace=False)
    abs_deltas = [-10., -5., -2., -1., 1., 2., 5., 10.]
    results = {}
    for adim in range(2):
        vname = ['一级减温阀', '二级减温阀'][adim]
        print(f"\n  [{vname}] 扰动 step0 → Δ主汽温")
        header = "  " + "Δ".rjust(7) + "  " + "  ".join([f"t={s}".rjust(9) for s in [1,2,3,5,8,12]])
        print(header)
        r = {}
        for d in abs_deltas:
            dT = {s: [] for s in [1,2,3,5,8,12]}
            for i in idxs:
                sh = raw[i:i+W, :cfg.N_STATE].copy()
                ah = raw[i:i+W, cfg.N_STATE:].copy()
                fa = raw[i+W:i+W+13, cfg.N_STATE:].copy()
                xt = torch.FloatTensor(np.concatenate([sh, ah], 1)).unsqueeze(0).to(DEVICE)
                at = torch.FloatTensor(fa).unsqueeze(0).to(DEVICE)
                trb = model.rollout(xt, at, mode='sliding')
                bp = trb[0,:,cfg.TARGET_IDX].cpu().numpy()
                ap = ah.copy(); ap[-1, adim] = np.clip(ap[-1, adim]+d, 0, 100)
                fap = fa.copy(); fap[0, adim] = np.clip(fap[0, adim]+d, 0, 100)
                xt = torch.FloatTensor(np.concatenate([sh, ap], 1)).unsqueeze(0).to(DEVICE)
                at = torch.FloatTensor(fap).unsqueeze(0).to(DEVICE)
                trp = model.rollout(xt, at, mode='sliding')
                pp = trp[0,:,cfg.TARGET_IDX].cpu().numpy()
                for s in [1,2,3,5,8,12]:
                    dT[s].append(pp[s]-bp[s])
            row = f"  {d:>+7.1f}  " + "  ".join([f"{np.mean(dT[s]):>+9.4f}" for s in [1,2,3,5,8,12]])
            print(row)
            for s in [1,2,3,5,8,12]:
                r[f'{d}_{s}'] = float(np.mean(dT[s]))
        results[f'action_{adim}'] = r
    return results


def main():
    model = WorldModel(
        n_state=cfg.N_STATE, n_action=cfg.N_ACTION,
        window_size=cfg.WINDOW_SIZE, d_model=cfg.D_MODEL,
        n_heads=cfg.N_HEADS, n_var_layers=cfg.N_VAR_LAYERS,
        n_tcn_layers=cfg.N_TCN_LAYERS, dropout=cfg.DROPOUT,
        rollout_mode='sliding', probabilistic=True,
    ).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    exp_dir = "results/exp_015_dualvalve"
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)

    crit = BetaNLLLoss(beta=BETA)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

    best_m, pc, be = float('inf'), 0, 0; t0 = time.time()
    for ep in range(1, cfg.EPOCHS + 1):
        crit.beta = 0. if ep <= BETA_WARMUP else BETA * min((ep - BETA_WARMUP) / 10, 1.)
        nll, sign = train_epoch(model, train_data, opt, crit, ep)
        v0, v4 = validate(model, val_data); sched.step(v4)
        if ep % 10 == 0 or ep == 1:
            print(f"  E{ep:3d} | NLL:{nll:7.0f} | Sign:{sign:.4f} | V0:{v0:.4f} | V4:{v4:.4f}")
        if v4 < best_m - 0.001: best_m, be, pc = v4, ep, 0; torch.save(
            {'epoch':ep, 'model_state_dict':model.state_dict()},
            f"{exp_dir}/checkpoints/best_model.pth")
        else: pc += 1
        if pc >= cfg.EARLY_STOPPING_PATIENCE: print(f"  Stop@{ep} best@{be}"); break

    ck = torch.load(f"{exp_dir}/checkpoints/best_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    print(f"\nTrain: {(time.time()-t0)/60:.1f}min Best@{be}")

    mae = eval_rollout(model, test_data)
    print(f"\nRollout(test): {mae[0]:.4f}→{mae[-1]:.4f} (×{mae[-1]/mae[0]:.1f})")

    print(f"\n多步敏感性:")
    sens = eval_sensitivity(model, test_data)

    result = {'constrained_dims': CONSTRAINED_DIMS, 'lambda': SIGN_LAMBDA,
              'delta': SIGN_DELTA, 'weights': ROLLOUT_W,
              'best_epoch': be, 'rollout_mae': mae.tolist(), 'sensitivity': sens}
    with open(f"{exp_dir}/results.json", 'w') as f: json.dump(result, f, indent=2)
    print(f"\nSaved: {exp_dir}/results.json")


if __name__ == '__main__':
    main()
