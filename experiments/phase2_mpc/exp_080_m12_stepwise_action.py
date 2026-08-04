#!/usr/bin/env python3
"""
exp_080_m12_stepwise_action.py — 方案1: M12 逐时间步动作注入 (因果 cross-attn)
==============================================================================
对比 M7 (act_lin 压缩 18步→1特征, 长程动作响应方向翻转):
  M12: 动作 [B,H,2] → per-step Linear(2→d) → [B,H,d] (保留时域步)
       状态特征 → 投影 [B,H,d] (per-step 查询)
       因果 cross-attn: 第k步查询 attend 前k步动作键值 → 每步 MLP → [B,H,2]
验证: 同 check_causal (首步扰动/持续阶跃 两种协议) + 画对比图 M7 vs M12
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv
import config as cfg

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMOKE = '--smoke' in sys.argv
EPOCHS = 3 if SMOKE else 60
PATIENCE = 6 if SMOKE else 10
EXP_DIR = 'results/exp_025_M12'
os.makedirs(f'{EXP_DIR}/checkpoints', exist_ok=True)
W = cfg.WINDOW_SIZE; H = E.H_OUT; d = cfg.D_MODEL

class CausalActionCrossAttn(nn.Module):
    """第k步查询 attend 前k步动作 (因果) — 动作时域模式保留"""
    def __init__(self, d, nhead=4):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.kv_proj = nn.Linear(d, d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=cfg.DROPOUT, batch_first=True)
        self.norm = nn.LayerNorm(d)
    def forward(self, q, kv):
        # q: [B,H,d] (状态条件), kv: [B,H,d] (动作特征)
        q = self.q_proj(q)
        kv = self.kv_proj(kv)
        attn_mask = torch.triu(torch.ones(H, H, device=q.device), diagonal=1).bool()  # 因果
        out, _ = self.attn(q, kv, kv, attn_mask=attn_mask, need_weights=False)
        return self.norm(q + out)

class M12(E.RevINModel):
    """逐时间步动作注入: 压缩式 act_lin → 因果 cross-attn 逐步对齐"""
    def __init__(self, probabilistic=True):
        super().__init__()
        self.probabilistic = probabilistic
        self.use_action = True
        self.patch = E.PatchEmbedding(W, 16, 8, d)
        self.np = self.patch.n_patches
        self.glb_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.exog_lin = nn.Linear(W, d)
        self.enc_layers = nn.ModuleList([E.TimeXerLayer(d, 4, cfg.DROPOUT) for _ in range(cfg.N_TCN_LAYERS)])
        self.norm = nn.LayerNorm(d)
        # 状态 → per-step 查询 [B, (np+1)*d] → [B, H, d]
        self.state_proj = nn.Sequential(nn.Linear((self.np + 1) * d, H * d), nn.GELU())
        # 动作: per-step 编码 (保留时域) [B, H, 2] → [B, H, d]
        self.step_act = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        # 因果 cross-attn + 每步输出
        self.causal_attn = CausalActionCrossAttn(d)
        self.step_head = nn.Sequential(nn.Linear(d * 2, d * 2), nn.GELU(),
                                       nn.Linear(d * 2, 2 if probabilistic else 1))
    def forward(self, x_hist, a_future=None):
        B = x_hist.shape[0]
        x_n = self.revin(x_hist, mode='norm')
        xt = x_n[:, :, E.TARGET_IDX]
        zt = self.patch(xt)
        glb = self.glb_token.expand(B, -1, -1)
        x = torch.cat([zt, glb], 1)
        exog_idx = [i for i in range(E.N_FEAT) if i != E.TARGET_IDX]
        ze = self.exog_lin(x_n[:, :, exog_idx].permute(0, 2, 1))
        za = self.step_act(a_future)                      # [B, H, 2] → [B, H, d] 逐步 (train_epoch 格式)
        for layer in self.enc_layers:
            x = layer(x, ze, za)
        x = self.norm(x)
        state_q = self.state_proj(x.reshape(B, -1)).reshape(B, H, d)
        out = self.causal_attn(state_q, za)                     # 因果: 第k步看前k步动作
        out = torch.cat([state_q, out], -1)
        raw = self.step_head(out)                               # [B, H, 2]
        if self.probabilistic:
            mu_n, lv_n = raw[..., 0], raw[..., 1]
        else:
            mu_n, lv_n = raw.squeeze(-1), None
        return self.denorm_out(mu_n, lv_n)      # 2026-08-03修复: 缺 denorm_out, 输出空间错误 (V4 43 vs M7 0.14)

# ============ 训练 (同 exp_025 M7 协议, 无扰动增强) ============
def train():
    model = M12().to(DEVICE)
    crit = E.BetaNLLLoss(beta=-0.3)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    best_m, pc, be = float('inf'), 0, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        crit.beta = -0.3
        nll = E.train_epoch(model, E.train_raw, opt, crit, True)
        v0, v4 = E.validate(model, E.val_raw, True); sched.step(v4)
        if ep % 10 == 0 or ep == 1:
            print(f"  E{ep:3d} | NLL:{nll:7.0f} | V0:{v0:.4f} | V4:{v4:.4f}", flush=True)
        if v4 < best_m - 0.001:
            best_m, be, pc = v4, ep, 0
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict()},
                       f'{EXP_DIR}/checkpoints/best_model.pth')
        else:
            pc += 1
        if pc >= PATIENCE:
            print(f"  Stop@{ep} best@{be}"); break
    ck = torch.load(f'{EXP_DIR}/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    mae = E.eval_rollout(model, E.test_raw, True)
    print(f"  M12 训练完成 best@{be} ({time.time()-t0:.0f}s) | Rollout avg={mae.mean():.4f}")
    return model

# ============ 因果一致性验证 ============
def check_causal(model):
    model.eval()
    np.random.seed(7)
    idxs = np.random.choice(range(len(E.test_raw) - W - H), 50, replace=False)
    dT_first = {s: [] for s in [1, 3, 8, 12]}
    dT_step = {s: [] for s in [1, 3, 8, 12]}
    for i in idxs:
        x_hist = torch.FloatTensor(E.test_raw[i:i+W]).unsqueeze(0).to(DEVICE)
        a_fut = torch.FloatTensor(E.test_raw[i+W:i+W+H, E.VALVE_IDX]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu_b, _ = model(x_hist, a_fut)
        bp = mu_b[0].cpu().numpy()
        a1 = a_fut.clone(); a1[0, 0, 1] = torch.clamp(a1[0, 0, 1] + 10, 0, 100)
        with torch.no_grad():
            mu1, _ = model(x_hist, a1)
        pp1 = mu1[0].cpu().numpy()
        a2 = a_fut.clone(); a2[0, 10:, 1] = torch.clamp(a2[0, 10:, 1] + 10, 0, 100)
        with torch.no_grad():
            mu2, _ = model(x_hist, a2)
        pp2 = mu2[0].cpu().numpy()
        for s in [1, 3, 8, 12]:
            dT_first[s].append(pp1[s] - bp[s]); dT_step[s].append(pp2[s] - bp[s])
    return {s: float(np.mean(dT_first[s])) for s in [1, 3, 8, 12]}, \
           {s: float(np.mean(dT_step[s])) for s in [1, 3, 8, 12]}

if __name__ == '__main__':
    model = train()
    f12, s12 = check_causal(model)
    # 原 M7
    m7 = E.build_model('M7').to(DEVICE).eval()
    ck7 = torch.load('results/exp_025_M7/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
    m7.load_state_dict(ck7['model_state_dict'])
    f7, s7 = check_causal(m7)
    print(f"\n  首步扰动 t1/t3/t8/t12:  M7  {[f'{f7[s]:+.3f}' for s in [1,3,8,12]]}")
    print(f"                           M12 {[f'{f12[s]:+.3f}' for s in [1,3,8,12]]}")
    print(f"  持续阶跃 t1/t3/t8/t12:  M7  {[f'{s7[s]:+.3f}' for s in [1,3,8,12]]}")
    print(f"                           M12 {[f'{s12[s]:+.3f}' for s in [1,3,8,12]]}")
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ts = [1, 3, 8, 12]
    axes[0].plot(ts, [f7[s] for s in ts], 'o-', color='#c0504d', label='M7 (compress enc)')
    axes[0].plot(ts, [f12[s] for s in ts], 's--', color='#4f81bd', label='M12 (stepwise causal attn)')
    axes[0].axhline(0, color='gray', lw=0.7); axes[0].set_title('(a) First-step pulse (+10% V2)')
    axes[0].set_xlabel('Prediction step t (×10s)'); axes[0].set_ylabel('ΔT (°C)'); axes[0].legend()
    axes[1].plot(ts, [s7[s] for s in ts], 'o-', color='#c0504d', label='M7 (compress enc)')
    axes[1].plot(ts, [s12[s] for s in ts], 's--', color='#4f81bd', label='M12 (stepwise causal attn)')
    axes[1].axhline(0, color='gray', lw=0.7)
    axes[1].set_title('(b) Sustained step from t=10 (+10% V2) — physical sign: negative')
    axes[1].set_xlabel('Prediction step t (×10s)'); axes[1].legend()
    fig.suptitle('Action→temperature response: M7 (compressed) vs M12 (stepwise causal attention)', fontsize=10.5)
    fig.tight_layout()
    fig.savefig('figures/fig_m12_effect.png', dpi=180, bbox_inches='tight')
    print('Saved: figures/fig_m12_effect.png')
