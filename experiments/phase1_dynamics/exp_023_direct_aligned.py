"""
exp_023_direct_aligned.py — Direct WM 协议对齐实验 (收尾)
===========================================================
目标: 厘清 Direct WM (0.789) vs Exp-0 (0.586) 精度差异的来源。

假设: 差异来自协议不对等, 不是架构:
  1. 输入: Exp-0 用 40 列全特征, Direct WM 只用 11状态+2动作
  2. 输出: Exp-0 只预测主汽温(单变量), Direct WM 预测 11 维全状态

对齐方案 (唯一保留差异 = 未来动作注入):
  - 输入: 40 列全特征 (同 Exp-0)
  - 输出: 只预测主汽温 18 步 (单变量头)
  - 训练: 同 β-NLL + 逐步权重
  - 评测: test 集 500 样本 seed 42, H=18 (与世界模型内部可比)

若对齐后 ≈ Exp-0 (0.586) → 差异全是协议造成, Direct+动作条件化可行;
若仍 0.7+ → 架构本身有问题 (MLP 融合未来动作).
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data
from world_model import RevIN, PatchEmbedding, PerVariableTCN

DEVICE = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')

BETA, BETA_WARMUP = -0.3, 20
BS, STEPS = 256, 500
H_OUT = 18

# 40 列全特征 (全 CSV 数值列, 除 date; 目标列保持在原位)
CSV_PATH = os.path.join(cfg.DATA_DIR, cfg.TRAIN_FILE)
df_full = pd.read_csv(CSV_PATH)
NUMERIC_COLS = [c for c in df_full.columns if c != 'date']
N_FEAT = len(NUMERIC_COLS)
TARGET_IDX = NUMERIC_COLS.index('末级过热器出口汽温')  # 主汽温在 40 列中的位置
VALVE_IDX = [NUMERIC_COLS.index('一级减温调节门阀位'),
             NUMERIC_COLS.index('二级减温调节门阀位')]
print(f"全特征 {N_FEAT} 列 | 主汽温 idx={TARGET_IDX} | 阀位 idx={VALVE_IDX}")


class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0., eps=1e-6):
        super().__init__(); self.beta = beta; self.eps = eps
    def forward(self, mu, lv, tgt):
        lv = torch.clamp(lv, -20., 20.)
        v = torch.exp(lv) + self.eps
        nll = 0.5 * (lv + (tgt - mu)**2 / v)
        if self.beta != 0: nll = v.detach()**self.beta * nll
        return nll.mean()


class DirectAligned(nn.Module):
    """Direct WM 对齐版: 40列历史 + 未来阀位 → 18步主汽温 (单变量)"""
    def __init__(self):
        super().__init__()
        d = cfg.D_MODEL
        W = cfg.WINDOW_SIZE
        self.revin = RevIN(N_FEAT)  # 40 列一起归一化
        self.patch = PatchEmbedding(W, 16, 8, d)
        self.np = self.patch.n_patches
        self.tcn = PerVariableTCN(self.np, d, cfg.N_TCN_LAYERS, cfg.DROPOUT)
        # 未来阀位序列编码
        self.action_enc = nn.Sequential(
            nn.Linear(H_OUT * 2, d * 2),
            nn.GELU(), nn.Dropout(cfg.DROPOUT),
        )
        # 融合 → 18 步主汽温 (μ, logσ²)
        self.decoder = nn.Sequential(
            nn.Linear(N_FEAT * d + d * 2, d * 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, d * 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, H_OUT * 2),  # H 步 × (μ, logσ²) 单变量
        )
        self.probabilistic = True

    def forward(self, x_hist, a_future):
        """
        x_hist: [B, W, N_FEAT] (40列历史, 含当前阀位)
        a_future: [B, H, 2] 未来阀位序列
        Returns: mu [B, H], lv [B, H]
        """
        B = x_hist.shape[0]
        d = cfg.D_MODEL
        x_n = self.revin(x_hist, mode='norm')
        var_tokens = [self.patch(x_n[:, :, i]) for i in range(N_FEAT)]
        var_tokens = torch.stack(var_tokens, 1)
        var_tokens = var_tokens.reshape(B * N_FEAT, self.np, d)
        s_repr = self.tcn(var_tokens).reshape(B, N_FEAT, d)

        # 未来阀位: 用历史窗口末位阀位归一化 (近似; 主要看模型能否利用)
        a_feat = self.action_enc(a_future.reshape(B, -1))  # [B, 2d]

        z = torch.cat([s_repr.reshape(B, -1), a_feat], 1)
        raw = self.decoder(z)  # [B, H*2]
        raw = raw.reshape(B, H_OUT, 2)
        mu_n = raw[..., 0]  # [B, H]
        lv_n = raw[..., 1]

        # denorm 主汽温 (用窗口统计)
        ms = self.revin._mean[:, :, TARGET_IDX]   # [B, 1]
        ss = self.revin._std[:, :, TARGET_IDX]    # [B, 1]
        w = self.revin.weight[TARGET_IDX]
        b = self.revin.bias[TARGET_IDX]
        mu_n2 = mu_n  # [B, H]
        if self.revin.affine: mu_n2 = (mu_n2 - b) / (w + self.revin.eps)
        mu = mu_n2 * ss + ms                      # [B,H]*[B,1]+[B,1] → [B,H] ✓
        sig = torch.exp(lv_n * 0.5) * ss          # [B,H]*[B,1] → [B,H]
        lv = 2.0 * torch.log(sig + 1e-8)
        return mu, lv


# ===== 数据: 40 列全特征 (与世界模型同切分 70/15/15) =====
data_all = df_full[NUMERIC_COLS].values.astype(np.float32)
data_all = np.nan_to_num(data_all, nan=0.0)
n_total = len(data_all)
n_train = int(n_total * 0.70); n_val_end = int(n_total * 0.85)
train_data, val_data = data_all[:n_train], data_all[n_train:n_val_end]
test_data = data_all[n_val_end:]
print(f"Data: {len(train_data)}+{len(val_data)}+{len(test_data)} | 40列全特征 | 单目标主汽温")


def train_epoch(model, raw, opt, crit, epoch):
    model.train(); W = cfg.WINDOW_SIZE; H = H_OUT; N = len(raw)
    total_nll = 0.
    for _ in range(STEPS):
        idxs = np.random.randint(0, N-W-H, size=BS)
        xh, af, tt = [], [], []
        for i in idxs:
            xh.append(raw[i:i+W])
            af.append(raw[i+W:i+W+H, VALVE_IDX])  # 未来阀位序列
            tt.append(raw[i+W:i+W+H, TARGET_IDX])
        x_hist = torch.FloatTensor(np.stack(xh)).to(DEVICE)
        a_fut = torch.FloatTensor(np.stack(af)).to(DEVICE)
        t_true = torch.FloatTensor(np.stack(tt)).to(DEVICE)
        opt.zero_grad()
        mu, lv = model(x_hist, a_fut)
        w = torch.linspace(1.0, 0.6, H, device=DEVICE)
        nll_loss = (w * crit(mu, lv, t_true).mean(dim=0)).sum() / H
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
        x_hist = torch.FloatTensor(raw[i:i+W]).unsqueeze(0).to(DEVICE)
        a_fut = torch.FloatTensor(raw[i+W:i+W+H, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        mu, _ = model(x_hist, a_fut)
        m0 += (mu[0,0]-raw[i+W,TARGET_IDX]).abs().item()
        m4 += (mu[0,4]-raw[i+W+4,TARGET_IDX]).abs().item()
    return m0/n, m4/n


@torch.no_grad()
def eval_rollout(model, raw, n=500):
    model.eval(); W = cfg.WINDOW_SIZE; H = H_OUT; N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-H), n, replace=False)
    err = np.zeros((n, H))
    for j, i in enumerate(idxs):
        x_hist = torch.FloatTensor(raw[i:i+W]).unsqueeze(0).to(DEVICE)
        a_fut = torch.FloatTensor(raw[i+W:i+W+H, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        tt = raw[i+W:i+W+H, TARGET_IDX]
        mu, _ = model(x_hist, a_fut)
        err[j] = np.abs(mu[0].cpu().numpy()-tt)
    return err.mean(0)


@torch.no_grad()
def eval_sensitivity(model, raw, n=200):
    """扰动未来阀位首位 → 轨迹响应"""
    model.eval(); W = cfg.WINDOW_SIZE; H = H_OUT; N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-H), n, replace=False)
    abs_deltas = [-10., -5., -2., -1., 1., 2., 5., 10.]
    results = {}
    for adim in range(2):
        r = {}
        for d in abs_deltas:
            dT = {s: [] for s in [1, 3, 8, 12]}
            for i in idxs:
                x_hist = torch.FloatTensor(raw[i:i+W]).unsqueeze(0).to(DEVICE)
                a_fut = torch.FloatTensor(raw[i+W:i+W+H, VALVE_IDX]).unsqueeze(0).to(DEVICE)
                mu_b, _ = model(x_hist, a_fut)
                bp = mu_b[0].cpu().numpy()
                a_p = a_fut.clone(); a_p[0, 0, adim] = torch.clamp(a_p[0, 0, adim]+d, 0, 100)
                mu_p, _ = model(x_hist, a_p)
                pp = mu_p[0].cpu().numpy()
                for s in [1, 3, 8, 12]:
                    dT[s].append(pp[s]-bp[s])
            for s in [1, 3, 8, 12]:
                r[f'{d}_{s}'] = float(np.mean(dT[s]))
        results[f'action_{adim}'] = r
    return results


def main():
    model = DirectAligned().to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    exp_dir = "results/exp_023_direct_aligned"
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
    print(f"\nDirect对齐 rollout(test): {mae[0]:.4f}→{mae[-1]:.4f} (×{mae[-1]/mae[0]:.1f})")
    sens = eval_sensitivity(model, test_data)
    print(f"  Sens (二级 ±10, t1/t12): {sens['action_1']['10.0_1']:+.3f} / {sens['action_1']['10.0_12']:+.3f}")

    result = {'model': 'direct_aligned_40col_singletarget', 'H': H_OUT, 'n_feat': N_FEAT,
              'best_epoch': be, 'rollout_mae': mae.tolist(), 'sensitivity': sens}
    with open(f"{exp_dir}/results.json", 'w') as f: json.dump(result, f, indent=2)
    print(f"Saved: {exp_dir}/results.json")
    print(f"\n对照: Exp-0 (40列, 单目标, direct) avg=0.330 step17=0.586")
    print(f"对照: Direct WM 13列多目标 (exp_022) 0.169→0.789")


if __name__ == '__main__':
    main()
