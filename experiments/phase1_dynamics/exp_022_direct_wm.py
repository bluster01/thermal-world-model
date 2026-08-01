"""
exp_022_direct_wm.py — Direct 多步世界模型 (无自回归累积)
===========================================================
动机: 自回归 rollout 误差累积 (0.12→0.767) 是世界模型 vs Exp-0 (0.33)
差距的全部来源。Direct 多步预测 (一次前向输出 H 步) 无累积, 且可以
条件于未来动作序列 — 这是 DWM 论文的做法, 可能同时拿到:
  - 动作条件化 (MPC 需要: 给定规划动作 → 直接预测整条轨迹)
  - 无累积精度 (对标 Exp-0 的 0.586@step17)

架构:
  [状态历史 96步 ‖ 动作历史 96步 ‖ 未来动作 H步] → Encoder → MLP → [ŝ_{t+1..t+H}]
  与 Exp-0 的 direct multi-step 完全同评测协议 (test 集, seed 42, H=18)

用法: python exp_022_direct_wm.py
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data
from world_model import RevIN, PatchEmbedding, PerVariableTCN, VariableAttention

DEVICE = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')

BETA, BETA_WARMUP = -0.3, 20
BS, STEPS = 256, 500
H_OUT = 18  # 直接输出 18 步


class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0., eps=1e-6):
        super().__init__(); self.beta = beta; self.eps = eps
    def forward(self, mu, lv, tgt):
        lv = torch.clamp(lv, -20., 20.)
        v = torch.exp(lv) + self.eps
        nll = 0.5 * (lv + (tgt - mu)**2 / v)
        if self.beta != 0: nll = v.detach()**self.beta * nll
        return nll.mean()


class DirectWorldModel(nn.Module):
    """Direct 多步世界模型: 历史状态+动作 + 未来动作 → 一次输出 H 步"""
    def __init__(self):
        super().__init__()
        d = cfg.D_MODEL
        W = cfg.WINDOW_SIZE
        # 状态编码: RevIN → Patch → PerVariableTCN (与世界模型同款)
        self.revin = RevIN(cfg.N_STATE)
        self.patch = PatchEmbedding(W, 16, 8, d)
        self.np = self.patch.n_patches
        self.tcn = PerVariableTCN(self.np, d, cfg.N_TCN_LAYERS, cfg.DROPOUT)
        # 动作编码: 历史动作 + 未来动作 展平后 MLP
        self.action_enc = nn.Sequential(
            nn.Linear((W + H_OUT) * cfg.N_ACTION, d * 2),
            nn.GELU(), nn.Dropout(cfg.DROPOUT),
        )
        # 融合: 状态表示 (11*d) + 动作编码 → 直接输出 H 步
        self.decoder = nn.Sequential(
            nn.Linear(cfg.N_STATE * d + d * 2, d * 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, d * 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, H_OUT * cfg.N_STATE * 2),  # H 步 × (μ, logσ²)
        )
        self.probabilistic = True

    def forward(self, s_hist, a_hist, a_future):
        """
        s_hist: [B, W, N_STATE]
        a_hist: [B, W, N_ACTION]
        a_future: [B, H_OUT, N_ACTION]
        Returns: mu [B, H_OUT, N_STATE], lv [B, H_OUT, N_STATE]
        """
        B = s_hist.shape[0]
        d = cfg.D_MODEL
        # 状态编码
        s_norm = self.revin(s_hist, mode='norm')
        var_tokens = [self.patch(s_norm[:, :, i]) for i in range(cfg.N_STATE)]
        var_tokens = torch.stack(var_tokens, 1)
        var_tokens = var_tokens.reshape(B * cfg.N_STATE, self.np, d)
        s_repr = self.tcn(var_tokens).reshape(B, cfg.N_STATE, d)
        # 动作编码 (历史+未来)
        a_all = torch.cat([a_hist, a_future], 1)  # [B, W+H, N_ACTION]
        a_feat = self.action_enc(a_all.reshape(B, -1))  # [B, 2d]
        # 融合
        z = torch.cat([s_repr.reshape(B, -1), a_feat], 1)
        raw = self.decoder(z)  # [B, H*N*2]
        raw = raw.reshape(B, H_OUT, cfg.N_STATE, 2)
        mu_n = raw[..., 0]
        lv_n = raw[..., 1]
        # denorm μ, σ (用当前窗口统计)
        ms = self.revin._mean[:, :, :cfg.N_STATE]   # [B, 1, N]
        ss = self.revin._std[:, :, :cfg.N_STATE]    # [B, 1, N]
        w = self.revin.weight[:cfg.N_STATE]
        b = self.revin.bias[:cfg.N_STATE]
        mu_n2 = mu_n  # [B, H, N]
        if self.revin.affine: mu_n2 = (mu_n2 - b) / (w + self.revin.eps)
        mu = mu_n2 * ss + ms                        # [B, H, N] 广播 OK
        sig = torch.exp(lv_n * 0.5) * ss            # [B, H, N] * [B,1,N]
        lv = 2.0 * torch.log(sig + 1e-8)
        return mu, lv


# ===== Data =====
state_data, delta_actions, valve_abs = load_raw_data()
raw_data = np.concatenate([state_data, valve_abs], axis=1)
n_total = len(raw_data)
n_train = int(n_total * 0.70); n_val_end = int(n_total * 0.85)
train_data, val_data = raw_data[:n_train], raw_data[n_train:n_val_end]
test_data = raw_data[n_val_end:]
print(f"Data: {len(train_data)}+{len(val_data)}+{len(test_data)} | Direct H={H_OUT}")


def train_epoch(model, raw, opt, crit, epoch):
    model.train(); W = cfg.WINDOW_SIZE; H = H_OUT; N = len(raw)
    total_nll = 0.
    for _ in range(STEPS):
        idxs = np.random.randint(0, N-W-H, size=BS)
        sh, ah, af, st = [], [], [], []
        for i in idxs:
            sh.append(raw[i:i+W, :cfg.N_STATE])
            ah.append(raw[i:i+W, cfg.N_STATE:])
            af.append(raw[i+W:i+W+H, cfg.N_STATE:])
            st.append(raw[i+W:i+W+H, :cfg.N_STATE])
        s_hist = torch.FloatTensor(np.stack(sh)).to(DEVICE)
        a_hist = torch.FloatTensor(np.stack(ah)).to(DEVICE)
        a_fut = torch.FloatTensor(np.stack(af)).to(DEVICE)
        s_true = torch.FloatTensor(np.stack(st)).to(DEVICE)
        opt.zero_grad()
        mu, lv = model(s_hist, a_hist, a_fut)
        # 逐步权重: 近步高权 (与世界模型 W1 类似)
        w = torch.linspace(1.0, 0.6, H, device=DEVICE)
        nll_loss = (w * crit(mu, lv, s_true).mean(dim=0)).sum() / H
        nll_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        total_nll += nll_loss.item()
    return total_nll / STEPS


@torch.no_grad()
def validate(model, raw, n=200):
    model.eval(); W = cfg.WINDOW_SIZE; H = H_OUT; N = len(raw)
    m0, m4 = 0., 0.
    for _ in range(n):
        i = np.random.randint(0, N-W-H)
        s_hist = torch.FloatTensor(raw[i:i+W, :cfg.N_STATE]).unsqueeze(0).to(DEVICE)
        a_hist = torch.FloatTensor(raw[i:i+W, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
        a_fut = torch.FloatTensor(raw[i+W:i+W+H, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
        s_true = raw[i+W:i+W+H, :cfg.N_STATE]
        mu, _ = model(s_hist, a_hist, a_fut)
        m0 += (mu[0,0,cfg.TARGET_IDX]-s_true[0,cfg.TARGET_IDX]).abs().item()
        m4 += (mu[0,4,cfg.TARGET_IDX]-s_true[4,cfg.TARGET_IDX]).abs().item()
    return m0/n, m4/n


@torch.no_grad()
def eval_rollout(model, raw, n=500):
    model.eval(); W = cfg.WINDOW_SIZE; H = H_OUT; N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-H), n, replace=False)
    err = np.zeros((n, H))
    for j, i in enumerate(idxs):
        s_hist = torch.FloatTensor(raw[i:i+W, :cfg.N_STATE]).unsqueeze(0).to(DEVICE)
        a_hist = torch.FloatTensor(raw[i:i+W, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
        a_fut = torch.FloatTensor(raw[i+W:i+W+H, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
        tt = raw[i+W:i+W+H, cfg.TARGET_IDX]
        mu, _ = model(s_hist, a_hist, a_fut)
        err[j] = np.abs(mu[0,:,cfg.TARGET_IDX].cpu().numpy()-tt)
    return err.mean(0)


@torch.no_grad()
def eval_sensitivity(model, raw, n=200):
    """扰动未来动作首位 → 观察整条轨迹响应 (direct 模式)"""
    model.eval(); W = cfg.WINDOW_SIZE; H = H_OUT; N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-H), n, replace=False)
    abs_deltas = [-10., -5., -2., -1., 1., 2., 5., 10.]
    results = {}
    for adim in range(2):
        r = {}
        for d in abs_deltas:
            dT = {s: [] for s in [1, 3, 8, 12]}
            for i in idxs:
                s_hist = torch.FloatTensor(raw[i:i+W, :cfg.N_STATE]).unsqueeze(0).to(DEVICE)
                a_hist = torch.FloatTensor(raw[i:i+W, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
                a_fut = torch.FloatTensor(raw[i+W:i+W+H, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
                mu_b, _ = model(s_hist, a_hist, a_fut)
                bp = mu_b[0,:,cfg.TARGET_IDX].cpu().numpy()
                # 扰动未来动作首位
                a_p = a_fut.clone(); a_p[0, 0, adim] = torch.clamp(a_p[0, 0, adim]+d, 0, 100)
                mu_p, _ = model(s_hist, a_hist, a_p)
                pp = mu_p[0,:,cfg.TARGET_IDX].cpu().numpy()
                for s in [1, 3, 8, 12]:
                    dT[s].append(pp[s]-bp[s])
            for s in [1, 3, 8, 12]:
                r[f'{d}_{s}'] = float(np.mean(dT[s]))
        results[f'action_{adim}'] = r
    return results


def main():
    model = DirectWorldModel().to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    exp_dir = "results/exp_022_direct_wm"
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)

    crit = BetaNLLLoss(beta=BETA)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

    best_m, pc, be = float('inf'), 0, 0; t0 = time.time()
    for ep in range(1, cfg.EPOCHS + 1):
        crit.beta = 0. if ep <= BETA_WARMUP else BETA * min((ep - BETA_WARMUP) / 10, 1.)
        nll = train_epoch(model, train_data, opt, crit, ep)
        v0, v4 = validate(model, val_data); sched.step(v4)
        if ep % 10 == 0 or ep == 1:
            print(f"  E{ep:3d} | NLL:{nll:7.0f} | V0:{v0:.4f} | V4:{v4:.4f}")
        if v4 < best_m - 0.001: best_m, be, pc = v4, ep, 0; torch.save(
            {'epoch':ep, 'model_state_dict':model.state_dict()},
            f"{exp_dir}/checkpoints/best_model.pth")
        else: pc += 1
        if pc >= cfg.EARLY_STOPPING_PATIENCE: print(f"  Stop@{ep} best@{be}"); break

    ck = torch.load(f"{exp_dir}/checkpoints/best_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    print(f"\nTrain: {(time.time()-t0)/60:.1f}min Best@{be}")

    mae = eval_rollout(model, test_data)
    print(f"\nDirect rollout(test): {mae[0]:.4f}→{mae[-1]:.4f} (×{mae[-1]/mae[0]:.1f})")
    sens = eval_sensitivity(model, test_data)
    print(f"  Sens (二级 ±10, t1/t12): {sens['action_1']['10.0_1']:+.3f} / {sens['action_1']['10.0_12']:+.3f}")

    result = {'model': 'direct_wm', 'H': H_OUT, 'best_epoch': be,
              'rollout_mae': mae.tolist(), 'sensitivity': sens}
    with open(f"{exp_dir}/results.json", 'w') as f: json.dump(result, f, indent=2)
    print(f"Saved: {exp_dir}/results.json")


if __name__ == '__main__':
    main()
