#!/usr/bin/env python3
"""
exp_082_deeponet_wm.py — 方案3b: DeepONet 式算子世界模型 (M13)
================================================================
DeepONet 核心: G(u)(y) = Σ_p b_p(u)·t_p(y)
  branch: 窗口状态(TCN) + 动作函数(per-step编码) → [B, p]  (输入函数 u)
  trunk:  时间坐标 k/H (sin/cos 周期编码) → [B, H, p]       (查询位置 y)
  输出: 内积 → [B, H] 温度轨迹
灰盒热力学: 可学习惯性核 τ (60s/120s 初始化) — 减温阀效应经传热惯性延迟,
  模型结构上"知道"动作效应不能瞬时作用于早期预测 (缓解 t1 伪响应)
训练同 M7 协议 + 方向一致性验证 + 图
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
EXP_DIR = 'results/exp_025_M13'
os.makedirs(f'{EXP_DIR}/checkpoints', exist_ok=True)
W = cfg.WINDOW_SIZE; H = E.H_OUT; d = cfg.D_MODEL; P = 256

class DeepONetWM(E.RevINModel):
    """DeepONet 算子世界模型: 动作函数→branch, 时间坐标→trunk, 内积输出 + 惯性灰盒核"""
    def __init__(self, probabilistic=True):
        super().__init__()
        self.probabilistic = probabilistic
        self.use_action = True
        # ---- branch: 状态编码 (同 M7: patch+GLB+exog+TimeXerLayer) ----
        self.patch = E.PatchEmbedding(W, 16, 8, d)
        self.np = self.patch.n_patches
        self.glb_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.exog_lin = nn.Linear(W, d)
        self.enc_layers = nn.ModuleList([E.TimeXerLayer(d, 4, cfg.DROPOUT) for _ in range(cfg.N_TCN_LAYERS)])
        self.norm = nn.LayerNorm(d)
        # ---- branch: 动作函数编码 (per-step, 保留时域) ----
        self.act_enc = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        # ---- branch head: 状态+动作 → [B, P] ----
        self.branch_head = nn.Sequential(
            nn.Linear((self.np + 1) * d + H * d, d * 2), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 2, P))
        # ---- trunk: 时间坐标 → [B, H, P] ----
        self.trunk = nn.Sequential(
            nn.Linear(4, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, P))
        # ---- 灰盒惯性核: 减温阀效应经一阶惯性延迟 (τ 可学习, 60s/120s 初始化) ----
        self.tau = nn.Parameter(torch.tensor([60.0, 120.0]))
        self.alpha = nn.Parameter(torch.tensor([0.7, 0.3]))   # 核权重 (softmax 归一化)
        # ---- 输出头: 内积 → μ/lv ----
        self.out_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.lv_lin = nn.Linear(P, 2)

    def forward(self, x_hist, a_future=None):
        B = x_hist.shape[0]
        x_n = self.revin(x_hist, mode='norm')
        zt = self.patch(x_n[:, :, E.TARGET_IDX])
        glb = self.glb_token.expand(B, -1, -1)
        x = torch.cat([zt, glb], 1)
        exog_idx = [i for i in range(E.N_FEAT) if i != E.TARGET_IDX]
        ze = self.exog_lin(x_n[:, :, exog_idx].permute(0, 2, 1))
        za = self.act_enc(a_future)                       # [B, H, d] per-step
        for layer in self.enc_layers:
            x = layer(x, ze, za)
        s_repr = self.norm(x)                             # [B, np+1, d]
        b = self.branch_head(torch.cat([s_repr.reshape(B, -1), za.reshape(B, -1)], 1))  # [B, P]
        # trunk: 时间坐标 (k/H, sin, cos, k/H²)
        k = torch.arange(H, device=DEVICE).float() / H
        y = torch.stack([k, torch.sin(2 * np.pi * k), torch.cos(2 * np.pi * k), k ** 2], -1)  # [H, 4]
        t = self.trunk(y).unsqueeze(0).expand(B, -1, -1)  # [B, H, P]
        mu_n = (b.unsqueeze(1) * t).sum(-1) * self.out_scale  # [B, H] 算子内积
        # 灰盒惯性核: 动作变化经一阶滞后 → 增量修正 (物理结构先验)
        dv = torch.diff(a_future, dim=1, prepend=a_future[:, :1] - a_future[:, :1])  # [B, H, 2] 动作变化
        dv = torch.cumsum(dv, 1)                          # 累计变化 (阶跃保持)
        al = torch.softmax(self.alpha, 0)
        kern = torch.zeros(H, device=DEVICE)
        for j, tau in enumerate(self.tau):
            tau = torch.abs(tau) + 10.0                    # 正约束, ≥10s
            kern = kern + al[j] * (1 - torch.exp(-torch.arange(H, device=DEVICE).float() * 10.0 / tau))
        kern = kern.unsqueeze(0).unsqueeze(-1)             # [1, H, 1]
        phys_inc = (kern * dv).sum(-1)                     # [B, H] 惯性响应
        mu_n = mu_n + 0.05 * phys_inc
        lv_n = self.lv_lin(t.reshape(B * H, P)).reshape(B, H, 2)[..., 1] * 0.1
        if self.probabilistic:
            return self.denorm_out(mu_n, lv_n)
        return self.denorm_out(mu_n, None)

# ============ 训练 (同 M7 协议) ============
def train():
    model = DeepONetWM().to(DEVICE)
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
    print(f"  M13-DeepONet 训练完成 best@{be} ({time.time()-t0:.0f}s) | Rollout avg={mae.mean():.4f}")
    return model

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
    f13, s13 = check_causal(model)
    m7 = E.build_model('M7').to(DEVICE).eval()
    ck7 = torch.load('results/exp_025_M7/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
    m7.load_state_dict(ck7['model_state_dict'])
    f7, s7 = check_causal(m7)
    print(f"\n  首步扰动 t1/t3/t8/t12:  M7    {[f'{f7[s]:+.3f}' for s in [1,3,8,12]]}")
    print(f"                           M13   {[f'{f13[s]:+.3f}' for s in [1,3,8,12]]}")
    print(f"  持续阶跃 t1/t3/t8/t12:  M7    {[f'{s7[s]:+.3f}' for s in [1,3,8,12]]}")
    print(f"                           M13   {[f'{s13[s]:+.3f}' for s in [1,3,8,12]]}")
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ts = [1, 3, 8, 12]
    axes[0].plot(ts, [f7[s] for s in ts], 'o-', color='#c0504d', label='M7')
    axes[0].plot(ts, [f13[s] for s in ts], 's--', color='#4f81bd', label='M13 DeepONet')
    axes[0].axhline(0, color='gray', lw=0.7); axes[0].set_title('(a) First-step pulse (+10% V2)')
    axes[0].set_xlabel('Prediction step t (×10s)'); axes[0].set_ylabel('ΔT (°C)'); axes[0].legend()
    axes[1].plot(ts, [s7[s] for s in ts], 'o-', color='#c0504d', label='M7')
    axes[1].plot(ts, [s13[s] for s in ts], 's--', color='#4f81bd', label='M13 DeepONet')
    axes[1].axhline(0, color='gray', lw=0.7)
    axes[1].set_title('(b) Sustained step from t=10 (+10% V2) — physical sign: negative')
    axes[1].set_xlabel('Prediction step t (×10s)'); axes[1].legend()
    fig.suptitle('DeepONet operator WM vs M7: action response consistency', fontsize=10.5)
    fig.tight_layout()
    fig.savefig('figures/fig_deeponet_effect.png', dpi=180, bbox_inches='tight')
    print('Saved: figures/fig_deeponet_effect.png')
