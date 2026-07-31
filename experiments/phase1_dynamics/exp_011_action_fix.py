"""
exp_011_action_fix.py — 动作条件化修复实验 (4方案)
===================================================
A. Action Scale ×10      — RevIN前放大动作
B. Action Bypass RevIN   — 动作跳过归一化，独立嵌入
C. FiLM Conditioning     — 动作生成γ,β调制状态特征
D. Decoder-Only Action   — Encoder纯状态，动作只进Decoder

评测: 1c动作敏感性 + 1b rollout MAE
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data
from world_model import (WorldModel, RevIN, PatchEmbedding,
                         VariableAttention, PerVariableTCN, GRUStateDecoder)

DEVICE = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')

ROLLOUT_K, BETA, BETA_WARMUP = 5, -0.3, 20
BS, STEPS = 256, 500
ROLLOUT_W = [1.0, 0.8, 0.6, 0.4, 0.2]


class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0., eps=1e-6):
        super().__init__()
        self.beta = beta; self.eps = eps
    def forward(self, mu, logvar, target):
        logvar = torch.clamp(logvar, -20., 20.)
        var = torch.exp(logvar) + self.eps
        nll = 0.5 * (logvar + (target - mu)**2 / var)
        if self.beta != 0:
            nll = var.detach() ** self.beta * nll
        return nll.mean()


# ======================================================================
#  A. Action Scale ×10
# ======================================================================
class ModelA_Scale(nn.Module):
    def __init__(self, scale=10.):
        super().__init__()
        self.scale = scale
        self.wm = WorldModel(n_state=11, n_action=2, window_size=96, d_model=64,
                             n_heads=4, n_var_layers=2, n_tcn_layers=2,
                             dropout=0.1, rollout_mode='sliding', probabilistic=True)
    def forward(self, x):
        xs = x.clone(); xs[:,:,cfg.N_STATE:] *= self.scale; return self.wm(xs)
    def rollout(self, x, a, **kw):
        return self.wm.rollout(x, a * self.scale, **kw)


# ======================================================================
#  B. Action Bypass RevIN
# ======================================================================
class ModelB_Bypass(nn.Module):
    """状态走 RevIN+TCN, 动作走独立 MLP, VarAttn 前拼接"""
    def __init__(self):
        super().__init__()
        d = cfg.D_MODEL
        # State path
        self.revin = RevIN(cfg.N_STATE + cfg.N_ACTION)
        self.patch = PatchEmbedding(cfg.WINDOW_SIZE, 16, 8, d)
        self.np = self.patch.n_patches
        self.tcn = PerVariableTCN(self.np, d, cfg.N_TCN_LAYERS, cfg.DROPOUT)

        # Action path
        self.act_emb = nn.Sequential(
            nn.Linear(cfg.N_ACTION * cfg.WINDOW_SIZE, d * 2),
            nn.GELU(), nn.Linear(d * 2, d),
        )
        # Combine: N_state state vars + 1 action token = N_state+1 tokens
        n_tokens = cfg.N_STATE + 1
        self.var_attn = nn.ModuleList([VariableAttention(d, cfg.N_HEADS, cfg.DROPOUT)
                                       for _ in range(cfg.N_VAR_LAYERS)])
        self.decoder = nn.Sequential(
            nn.Linear(n_tokens * d, d * 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, d * 2), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 2, cfg.N_STATE * 2),
        )
        self.probabilistic = True

    def forward(self, x):
        B, W, _ = x.shape
        s = x[:, :, :cfg.N_STATE]; a = x[:, :, cfg.N_STATE:]

        # State: RevIN(only state part) → Patch → TCN
        x_norm = self.revin(x, mode='norm')
        s_norm = x_norm[:, :, :cfg.N_STATE]
        var_tokens = [self.patch(s_norm[:,:,i]) for i in range(cfg.N_STATE)]
        var_tokens = torch.stack(var_tokens, 1)  # [B,Ns,np,d]
        var_tokens = var_tokens.reshape(B*cfg.N_STATE, self.np, cfg.D_MODEL)
        s_repr = self.tcn(var_tokens).reshape(B, cfg.N_STATE, cfg.D_MODEL)

        # Action: 整窗展平 → MLP → token
        a_token = self.act_emb(a.reshape(B, -1)).unsqueeze(1)  # [B,1,d]

        # Concat + VarAttn
        tokens = torch.cat([s_repr, a_token], 1)  # [B, Ns+1, d]
        for attn in self.var_attn:
            tokens, _ = attn(tokens)

        # Decode (include action token for prediction)
        raw = self.decoder(tokens.reshape(B, -1))

        # RevIN denorm
        ms = self.revin._mean[:,:,:cfg.N_STATE]; ss = self.revin._std[:,:,:cfg.N_STATE]
        w = self.revin.weight[:cfg.N_STATE]; b = self.revin.bias[:cfg.N_STATE]
        mu_n = raw[:,:cfg.N_STATE]; lv_n = raw[:,cfg.N_STATE:]
        mu_n2 = mu_n.unsqueeze(1)
        if self.revin.affine: mu_n2 = (mu_n2 - b) / (w + self.revin.eps)
        mu = (mu_n2 * ss + ms).squeeze(1)
        sig = torch.exp(lv_n * 0.5) * ss.squeeze(1)
        lv = 2.0 * torch.log(sig + 1e-8)
        return mu, lv

    def rollout(self, x_hist, a_seq, mode='sliding', return_stats=False):
        B, W, _ = x_hist.shape; H = a_seq.shape[1]
        sw = x_hist[:,:,:cfg.N_STATE]; aw = x_hist[:,:,cfg.N_STATE:]
        preds = []
        for t in range(H):
            mu, _ = self(torch.cat([sw, aw], 2))
            preds.append(mu)
            sw = torch.cat([sw[:,1:,:], mu.unsqueeze(1)], 1)
            aw = torch.cat([aw[:,1:,:], a_seq[:,t:t+1,:]], 1)
        return torch.stack(preds, 1)


# ======================================================================
#  C. FiLM Conditioning
# ======================================================================
class ModelC_FiLM(nn.Module):
    """状态走 RevIN+TCN+VarAttn, 动作生成γ/β调制各变量特征"""
    def __init__(self):
        super().__init__()
        d = cfg.D_MODEL
        self.wm = WorldModel(n_state=11, n_action=2, window_size=96, d_model=64,
                             n_heads=4, n_var_layers=2, n_tcn_layers=2,
                             dropout=0.1, rollout_mode='sliding', probabilistic=True)
        # FiLM: 动作 → 2×N_state 个 γ,β
        self.film_mlp = nn.Sequential(
            nn.Linear(cfg.N_ACTION * cfg.WINDOW_SIZE, d * 4),
            nn.GELU(), nn.Linear(d * 4, cfg.N_STATE * 2),
        )

    def forward(self, x):
        B, W, _ = x.shape
        a = x[:, :, cfg.N_STATE:]

        # 获取状态特征 (hijack internal forward halfway)
        wm = self.wm
        x_norm = wm.revin(x, mode='norm')
        # Per-variable encoding
        var_tokens = []
        for i in range(cfg.N_STATE + cfg.N_ACTION):
            var_tokens.append(wm.patch_embed(x_norm[:,:,i]))
        var_tokens = torch.stack(var_tokens, 1)  # [B,N,np,d]
        var_tokens = var_tokens.reshape(B*(cfg.N_STATE+cfg.N_ACTION), wm.n_patches, cfg.D_MODEL)
        var_repr = wm.var_encoder(var_tokens)  # [B*N, d]
        var_repr = var_repr.reshape(B, cfg.N_STATE+cfg.N_ACTION, cfg.D_MODEL)
        var_repr_state = var_repr[:, :cfg.N_STATE, :]  # [B, Ns, d]

        # FiLM: 动作 → γ,β
        film = self.film_mlp(a.reshape(B, -1))  # [B, Ns*2]
        gamma = film[:, :cfg.N_STATE].unsqueeze(2)  # [B, Ns, 1]
        beta = film[:, cfg.N_STATE:].unsqueeze(2)

        # Modulate
        var_repr_state = gamma * var_repr_state + beta

        # VarAttn (all 13 tokens)
        for attn_layer in wm.var_attention_layers:
            var_repr, _ = attn_layer(var_repr)

        # Decode (all tokens, decoder was built for 13*64=832)
        var_flat = var_repr.reshape(B, (cfg.N_STATE + cfg.N_ACTION) * cfg.D_MODEL)
        raw_out = wm.state_decoder_direct(var_flat)

        # Denorm
        ms = wm.revin._mean[:,:,:cfg.N_STATE]; ss = wm.revin._std[:,:,:cfg.N_STATE]
        w = wm.revin.weight[:cfg.N_STATE]; b = wm.revin.bias[:cfg.N_STATE]
        mu_n = raw_out[:,:cfg.N_STATE]; lv_n = raw_out[:,cfg.N_STATE:]
        mu_n2 = mu_n.unsqueeze(1)
        if wm.revin.affine: mu_n2 = (mu_n2 - b) / (w + wm.revin.eps)
        mu = (mu_n2 * ss + ms).squeeze(1)
        sig = torch.exp(lv_n * 0.5) * ss.squeeze(1)
        lv = 2.0 * torch.log(sig + 1e-8)
        return mu, lv

    def rollout(self, x_hist, a_seq, mode='sliding', return_stats=False):
        B, W, _ = x_hist.shape; H = a_seq.shape[1]
        sw = x_hist[:,:,:cfg.N_STATE]; aw = x_hist[:,:,cfg.N_STATE:]
        preds = []
        for t in range(H):
            mu, _ = self(torch.cat([sw, aw], 2))
            preds.append(mu)
            sw = torch.cat([sw[:,1:,:], mu.unsqueeze(1)], 1)
            aw = torch.cat([aw[:,1:,:], a_seq[:,t:t+1,:]], 1)
        return torch.stack(preds, 1)


# ======================================================================
#  D. Decoder-Only Action
# ======================================================================
class ModelD_DecoderOnly(nn.Module):
    """Encoder纯状态, 动作只进Decoder的GRU cell"""
    def __init__(self):
        super().__init__()
        d = cfg.D_MODEL
        self.wm = WorldModel(n_state=11, n_action=0, window_size=96, d_model=64,
                             n_heads=4, n_var_layers=2, n_tcn_layers=2,
                             dropout=0.1, rollout_mode='sliding', probabilistic=True)
        # Override decoder to take action
        self.wm.state_decoder_direct = nn.Sequential(
            nn.Linear(cfg.N_STATE * d + cfg.N_ACTION, d * 4),
            nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, d * 2), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 2, cfg.N_STATE * 2),
        )
        self.wm.probabilistic = True

    def forward(self, x):
        # x: [B, W, N_state + N_action]; encoder uses state only; decoder gets last action
        B, W, _ = x.shape
        s = x[:,:,:cfg.N_STATE]
        a_last = x[:, -1, cfg.N_STATE:]  # [B, 2]

        wm = self.wm
        x_norm = wm.revin(s, mode='norm')
        var_tokens = [wm.patch_embed(x_norm[:,:,i]) for i in range(cfg.N_STATE)]
        var_tokens = torch.stack(var_tokens, 1)
        var_tokens = var_tokens.reshape(B * cfg.N_STATE, wm.n_patches, cfg.D_MODEL)
        var_repr = wm.var_encoder(var_tokens).reshape(B, cfg.N_STATE, cfg.D_MODEL)
        for attn in wm.var_attention_layers:
            var_repr, _ = attn(var_repr)

        z = var_repr.reshape(B, cfg.N_STATE * cfg.D_MODEL)
        raw = self.wm.state_decoder_direct(torch.cat([z, a_last], 1))

        # Denorm
        ms = self.wm.revin._mean[:,:,:cfg.N_STATE]; ss = self.wm.revin._std[:,:,:cfg.N_STATE]
        wq = self.wm.revin.weight[:cfg.N_STATE]; bq = self.wm.revin.bias[:cfg.N_STATE]
        mu_n = raw[:,:cfg.N_STATE]; lv_n = raw[:,cfg.N_STATE:]
        mu_n2 = mu_n.unsqueeze(1)
        if self.wm.revin.affine: mu_n2 = (mu_n2 - bq) / (wq + self.wm.revin.eps)
        mu = (mu_n2 * ss + ms).squeeze(1)
        sig = torch.exp(lv_n * 0.5) * ss.squeeze(1)
        return mu, 2.0 * torch.log(sig + 1e-8)

    def rollout(self, x_hist, a_seq, mode='sliding', return_stats=False):
        B, W, _ = x_hist.shape; H = a_seq.shape[1]
        sw = x_hist[:,:,:cfg.N_STATE]
        preds = []
        for t in range(H):
            a_cur = a_seq[:, t:t+1, :]  # [B, 1, 2]
            if t == 0:
                aw = x_hist[:,:,cfg.N_STATE:]
            else:
                aw = torch.cat([aw[:,1:,:], a_cur], 1)
            x_step = torch.cat([sw, aw], 2)
            mu, _ = self(x_step)
            preds.append(mu)
            sw = torch.cat([sw[:,1:,:], mu.unsqueeze(1)], 1)
        return torch.stack(preds, 1)


# ======================================================================
#                       Training & Eval
# ======================================================================
def train_epoch(model, raw_data, optimizer, criterion, scale_act=1.0):
    model.train(); W, K = cfg.WINDOW_SIZE, ROLLOUT_K; N = len(raw_data)
    total = 0.
    for _ in range(STEPS):
        idxs = np.random.randint(0, N - W - K, size=BS)
        xb, ab, sb = [], [], []
        for i in idxs:
            sw = raw_data[i:i+W, :cfg.N_STATE]
            aw = raw_data[i:i+W, cfg.N_STATE:] * scale_act
            xb.append(np.concatenate([sw, aw], 1))
            ab.append(raw_data[i+W:i+W+K, cfg.N_STATE:] * scale_act)
            sb.append(raw_data[i+W:i+W+K, :cfg.N_STATE])
        xt = torch.FloatTensor(np.stack(xb)).to(DEVICE)
        at = torch.FloatTensor(np.stack(ab)).to(DEVICE)
        st = torch.FloatTensor(np.stack(sb)).to(DEVICE)
        optimizer.zero_grad()
        ss = xt[:,:,:cfg.N_STATE]; aa = xt[:,:,cfg.N_STATE:]
        tsl = 0.
        for k in range(K):
            mu, lv = model(torch.cat([ss, aa], 2))
            sl = ROLLOUT_W[k] * criterion(mu, lv, st[:, k])
            tsl += sl
            ss = torch.cat([ss[:,1:], mu.unsqueeze(1).detach()], 1)
            aa = torch.cat([aa[:,1:], at[:,k:k+1]], 1)
        tsl.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); total += tsl.item()
    return total / STEPS


@torch.no_grad()
def validate(model, raw_data, scale_act=1.0, n=200):
    model.eval(); W, K = cfg.WINDOW_SIZE, ROLLOUT_K; N = len(raw_data)
    m0, m4 = 0., 0.
    for _ in range(n):
        i = np.random.randint(0, N - W - K)
        sw = raw_data[i:i+W, :cfg.N_STATE]; aw = raw_data[i:i+W, cfg.N_STATE:] * scale_act
        xt = torch.FloatTensor(np.concatenate([sw, aw], 1)).unsqueeze(0).to(DEVICE)
        fa = torch.FloatTensor(raw_data[i+W:i+W+K, cfg.N_STATE:] * scale_act).unsqueeze(0).to(DEVICE)
        st = torch.FloatTensor(raw_data[i+W:i+W+K, :cfg.N_STATE]).unsqueeze(0).to(DEVICE)
        tr = model.rollout(xt, fa, mode='sliding')
        m0 += (tr[0,0,cfg.TARGET_IDX] - st[0,0,cfg.TARGET_IDX]).abs().item()
        m4 += (tr[0,min(4,K-1),cfg.TARGET_IDX] - st[0,min(4,K-1),cfg.TARGET_IDX]).abs().item()
    return m0/n, m4/n


@torch.no_grad()
def eval_rollout(model, raw_data, scale_act=1.0, H=18, n=500):
    model.eval(); W = cfg.WINDOW_SIZE; N = len(raw_data)
    np.random.seed(42); idxs = np.random.choice(range(N - W - H), n, replace=False)
    err = np.zeros((n, H))
    for j, i in enumerate(idxs):
        sw = raw_data[i:i+W, :cfg.N_STATE]; aw = raw_data[i:i+W, cfg.N_STATE:] * scale_act
        xt = torch.FloatTensor(np.concatenate([sw, aw], 1)).unsqueeze(0).to(DEVICE)
        fa = torch.FloatTensor(raw_data[i+W:i+W+H, cfg.N_STATE:] * scale_act).unsqueeze(0).to(DEVICE)
        tt = raw_data[i+W:i+W+H, cfg.TARGET_IDX]
        tr = model.rollout(xt, fa, mode='sliding')
        err[j] = np.abs(tr[0,:,cfg.TARGET_IDX].cpu().numpy() - tt)
    return err.mean(0)


@torch.no_grad()
def eval_sensitivity(model, raw_data, scale_act=1.0, n=500):
    model.eval(); W = cfg.WINDOW_SIZE; N = len(raw_data)
    np.random.seed(42); idxs = np.random.choice(range(N - W - 1), n, replace=False)
    res = {}
    for adim in range(cfg.N_ACTION):
        r = {d: [] for d in [-0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2]}
        for i in idxs:
            sh = raw_data[i:i+W, :cfg.N_STATE]
            ah = raw_data[i:i+W, cfg.N_STATE:].copy() * scale_act
            xt = torch.FloatTensor(np.concatenate([sh, ah], 1)).unsqueeze(0).to(DEVICE)
            mb, _ = model(xt); tb = mb[0, cfg.TARGET_IDX].item()
            for d in r:
                ap = ah.copy(); ap[-1, adim] *= (1+d)
                xp = torch.FloatTensor(np.concatenate([sh, ap], 1)).unsqueeze(0).to(DEVICE)
                mp, _ = model(xp)
                r[d].append(mp[0, cfg.TARGET_IDX].item() - tb)
        res[f'action_{adim}'] = {d: (np.mean(v), np.std(v)) for d, v in r.items()}
    return res


# ======================================================================
def run(name, model, raw, train, val, test, scale=1.):
    print(f"\n{'='*55}\n  exp_011_{name}\n{'='*55}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}  Scale: {scale}")

    d = f"results/exp_011_{name}"; os.makedirs(f"{d}/checkpoints", exist_ok=True)
    crit = BetaNLLLoss(beta=BETA)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

    best_m, pc, be = float('inf'), 0, 0; t0 = time.time()
    for ep in range(1, cfg.EPOCHS + 1):
        crit.beta = 0. if ep <= BETA_WARMUP else BETA * min((ep - BETA_WARMUP) / 10, 1.)
        tl = train_epoch(model, train, opt, crit, scale)
        v0, v4 = validate(model, val, scale); sched.step(v4)
        if ep % 10 == 0 or ep == 1:
            print(f"  E{ep:3d} | NLL:{tl:7.0f} | V0:{v0:.4f} | V4:{v4:.4f}")
        if v4 < best_m - 0.001: best_m, be, pc = v4, ep, 0; torch.save(
            {'epoch':ep, 'model_state_dict':model.state_dict(), 'val_mae4':v4},
            f"{d}/checkpoints/best_model.pth")
        else: pc += 1
        if pc >= cfg.EARLY_STOPPING_PATIENCE: print(f"  Stop@{ep} best@{be}"); break

    tm = time.time() - t0
    ck = torch.load(f"{d}/checkpoints/best_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict']); model.eval()

    mae = eval_rollout(model, test, scale)
    print(f"\n  Rollout: {mae[0]:.4f}→{mae[-1]:.4f} (×{mae[-1]/mae[0]:.1f})")

    sens = eval_sensitivity(model, test, scale)
    print(f"  Sensitivity (Δ喷水阀 ±20%):")
    r0 = sens['action_0']
    for dp in [-0.2, -0.1, 0.1, 0.2]:
        m, s = r0[dp]; print(f"    {dp:+5.0%} → {m:+.4f}°C (±{s:.4f})")

    r = {'name': name, 'scale': scale,
         'params': sum(p.numel() for p in model.parameters()),
         'train_min': tm/60, 'best_ep': be,
         'rollout_mae': mae.tolist(),
         'sensitivity': {k: {str(d): list(v) for d, v in s.items()} for k, s in sens.items()}}
    with open(f"{d}/results.json", 'w') as f: json.dump(r, f, indent=2)
    return r


def main():
    sd, da, _ = load_raw_data(); raw = np.concatenate([sd, da], 1)
    nt = int(len(raw)*.70); nv = int(len(raw)*.85)
    train, val, test = raw[:nt], raw[nt:nv], raw[nv:]
    print(f"Data: {len(train)}+{len(val)}+{len(test)}")

    all_r = {}

    # A done: scale×10 → still zero sensitivity

    # B: Bypass

    # B: Bypass
    all_r['B_bypass'] = run('B_bypass', ModelB_Bypass().to(DEVICE),
                             raw, train, val, test, scale=1.)

    # C: FiLM
    all_r['C_film'] = run('C_film', ModelC_FiLM().to(DEVICE),
                           raw, train, val, test, scale=1.)

    # D: DecoderOnly
    all_r['D_decoder'] = run('D_decoder', ModelD_DecoderOnly().to(DEVICE),
                              raw, train, val, test, scale=1.)

    # Summary
    print(f"\n\n{'='*70}\n  ACTION FIX SUMMARY\n{'='*70}")
    print(f"  {'Variant':<15} {'Step0':>8} {'Step17':>8} {'±20%→ΔT':>10}")
    print(f"  {'-'*45}")
    print(f"  {'exp_006 bug':<15} {'0.0834':>8} {'0.8082':>8} {'±0.001':>10}")
    for k, r in all_r.items():
        m = r['rollout_mae']
        dT = list(r['sensitivity']['action_0'].values())[-1][0] if 'action_0' in r['sensitivity'] else 0
        print(f"  {k:<15} {m[0]:>8.4f} {m[-1]:>8.4f} {abs(dT):>10.4f}")


if __name__ == '__main__':
    main()
