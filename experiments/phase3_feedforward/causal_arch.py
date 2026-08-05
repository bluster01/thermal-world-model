#!/usr/bin/env python3
"""
causal_arch.py — 强动作因果架构 (2026-08-05)
=============================================
设计稿: docs/causal_eval_framework.md §6

诊断 (为何 DirectWM / TimeXerWM 必然弱因果):
  [缺陷1] 动作只进 1 个 token, 输出走 FlattenHead
          TimeXerLayer 的 act_attn/exog_attn 只更新 glb 一个 token, 而 head 是
          Linear((np+1)*d, H*2) = 12 token 展平 → 动作仅占 head 输入 1/12,
          另外 11 个 patch token 是主汽温自身历史的恒等残差通路。
          最省力解 = 从内生通路外推, 压低 GLB 权重。
          注: M9b 把 head 从 GLB-only 改成 FlattenHead ("消除信息瓶颈"),
              而那个瓶颈恰恰是因果的强制通路 → v2 提精度砍因果。
  [缺陷2] 纯加性注入无法表达工况相关增益
          DirectWM: cat([s_repr, a_feat])→MLP;  TimeXer: 残差相加
          两者只能表达 f(state) + g(action)。但 exp_099 观测响应比例中位 0.17
          且分布很宽 = K(state) × action 的乘性增益。加性结构数学上无法表示。
  [缺陷3] direct multi-horizon 无时间因果约束
          head 一次并行吐 H 步, 无单调性/时间常数约束 → 响应剖面可任意非单调。
  [缺陷4] 动作在训练分布上近乎冗余 (SP 可由负荷/温度趋势预测) → 梯度饥饿。
          任何带旁路通路的架构都会饿死动作通道。

本模块提供的架构:
  ResidualCausalWM  A1 残差分解: T̂ = f_free(x) + g(x,a),  架构性保证 g(x,0) ≡ 0
                    A2 乘性门控 phi(x) ⊙ psi(a)         (对付缺陷2)
                    A3 一阶惯性结构化响应 K(x), tau(x)   (对付缺陷3, 最强先验)
                    C1 增量累积输出 cumsum              (对付缺陷3, 正交)
  TimeXerCausalWM   B1 head_mode='glb' 强制所有输出经过唯一携带动作的 token

g(x,0) ≡ 0 的精确实现 (非正则, 是恒等式):
  动作分支全程 bias=False, 激活 GELU(0)=0  →  psi(0)=0
  输出投影 bias=False                      →  g_out(phi ⊙ 0) = 0
  故 pred(x,a) − pred(x,0) = g(x,a) 恒等于一个专用子网络的输出:
    * 有独立梯度, free path 吸收不了
    * 可直接用 L1 的 DiD 真值监督 (评测真值 → 训练信号, 消灭 objective mismatch)
    * 可加物理约束 (符号/单调/增益上下界)
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import config as cfg
from world_model import RevIN, PatchEmbedding, PerVariableTCN, VariableAttention


# ==================================================================== 干预分支
class InterventionMLP(nn.Module):
    """A1+A2: 乘性门控干预分支。全程 bias=False → psi(0)=0 → g(x,0)=0 恒成立。

    g(x, a) = W_out( phi(s_repr) ⊙ psi(a) )
    phi 允许 bias (工况门控与动作无关); psi 与 W_out 严禁 bias。
    """

    def __init__(self, d_state, d, H):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(H, d, bias=False), nn.GELU(),
            nn.Linear(d, d, bias=False), nn.GELU())      # GELU(0)=0 → psi(0)=0
        self.phi = nn.Sequential(
            nn.Linear(d_state, d), nn.GELU(), nn.Linear(d, d))
        self.g_out = nn.Linear(d, H, bias=False)         # bias=False → g(x,0)=0

    def forward(self, s_repr_flat, a):
        return self.g_out(self.phi(s_repr_flat) * self.psi(a))   # [B, H]


class InterventionPhysics(nn.Module):
    """A3: 一阶惯性 (可级联) 结构化干预分支。

    物理形式: SP 偏移轨迹 u(k) = cumsum(ΔSP)(k) 驱动的一阶滞后
        r(k) = r(k-1) + (K(x)·u(k) − r(k-1)) / tau(x)
    级联 n_lag 次得到 S 形 (等效纯迟延), 无需显式 delay 参数。

    对 u 严格线性 → a=0 ⇒ u=0 ⇒ r≡0, 故 g(x,0)=0 恒成立 (无 bias 需求)。
    K(x) 与 tau(x) 由工况决定 → 直接建模 exp_099 的增益异质性, 且可解释:
    K(x) 可画成负荷的函数, 与 DiD 分层真值逐箱对比。
    """

    def __init__(self, d_state, d, H, n_lag=2, tau_min=2.0, tau_max=400.0, tau_init=18.0):
        super().__init__()
        self.H, self.n_lag = H, n_lag
        self.tau_min, self.tau_max = tau_min, tau_max
        self.trunk = nn.Sequential(nn.Linear(d_state, d), nn.GELU())
        self.k_head = nn.Linear(d, 1)                        # 增益 K(x)
        self.tau_head = nn.Linear(d, n_lag)                  # 各级时间常数
        nn.init.zeros_(self.tau_head.weight)
        nn.init.zeros_(self.k_head.weight)
        nn.init.constant_(self.k_head.bias, 1.0)             # 初始增益 1.0 (物理稳态)
        # tau 按物理时标初始化: exp_099 测得 600s 响应 ≈97% → 2 级级联需 tau≈18 步。
        # 若用 sigmoid(0)=0.5 的默认零 bias, tau=201 步(2010s), 600s 仅走到 4%,
        # 响应几乎为 0 → K 的梯度饿死, A3 分支训不起来。
        frac0 = (tau_init - tau_min) / (tau_max - tau_min)
        nn.init.constant_(self.tau_head.bias, float(torch.logit(torch.tensor(frac0))))

    def params(self, s_repr_flat):
        h = self.trunk(s_repr_flat)
        K = self.k_head(h)                                            # [B,1]
        frac = torch.sigmoid(self.tau_head(h))                        # [B,n_lag] ∈(0,1)
        tau = self.tau_min + (self.tau_max - self.tau_min) * frac     # [B,n_lag]
        return K, tau

    def forward(self, s_repr_flat, a):
        # a: [B,H] 一阶差分 ΔSP → u: [B,H] SP 偏移轨迹 (相对 onset 前)
        u = torch.cumsum(a, dim=1)
        K, tau = self.params(s_repr_flat)
        sig = K * u                                                   # [B,H] 稳态目标
        for L in range(self.n_lag):
            alpha = (1.0 / tau[:, L:L + 1]).clamp(1e-3, 1.0)          # [B,1]
            out, r = [], torch.zeros_like(sig[:, 0])
            for k in range(self.H):
                r = r + alpha[:, 0] * (sig[:, k] - r)
                out.append(r)
            sig = torch.stack(out, dim=1)
        return sig                                                    # [B,H]


class InterventionBoth(nn.Module):
    """A3 物理主干 + A1 小残差修正 (物理形式不足时兜底)。"""

    def __init__(self, d_state, d, H, n_lag=2, res_scale=0.1):
        super().__init__()
        self.phys = InterventionPhysics(d_state, d, H, n_lag)
        self.res = InterventionMLP(d_state, d, H)
        self.res_scale = res_scale

    def forward(self, s_repr_flat, a):
        return self.phys(s_repr_flat, a) + self.res_scale * self.res(s_repr_flat, a)


def make_intervention(mode, d_state, d, H, n_lag=2):
    if mode == 'mlp':
        return InterventionMLP(d_state, d, H)
    if mode == 'phys':
        return InterventionPhysics(d_state, d, H, n_lag)
    if mode == 'both':
        return InterventionBoth(d_state, d, H, n_lag)
    raise ValueError(f"未知 intervention mode: {mode}")


# ==================================================================== 基类
class CausalWMBase(nn.Module):
    """RevIN 归一化/反归一化基类 (与 exp_025.RevINModel 数值等价, 但不依赖其模块)。"""

    def __init__(self, n_feat, target_idx):
        super().__init__()
        self.n_feat, self.target_idx = n_feat, target_idx
        self.revin = RevIN(n_feat)
        self.use_revin = True
        self.use_action = True

    def denorm_out(self, mu_n, lv_n):
        ms = self.revin._mean[:, :, self.target_idx]
        ss = self.revin._std[:, :, self.target_idx]
        w = self.revin.weight[self.target_idx]; b = self.revin.bias[self.target_idx]
        mu_n2 = mu_n
        if self.revin.affine:
            mu_n2 = (mu_n2 - b) / (w + self.revin.eps)
        mu = mu_n2 * ss + ms
        if lv_n is not None:
            sig = torch.exp(lv_n * 0.5) * ss
            return mu, 2.0 * torch.log(sig + 1e-8)
        return mu, None


# ==================================================================== A1 主架构
class ResidualCausalWM(CausalWMBase):
    """A1 残差分解因果世界模型。

        T̂ = f_free(x) + g(x, a),   g(x, 0) ≡ 0 (架构恒等式)

    编码器沿用 Phase 1 已验证组件 (RevIN + Patch + PerVarTCN + VarAttn), 保证与
    M5/M7-DSP 的编码器同条件, 差异只在动作注入路径 → 消融可归因。

    Args:
        intervention: 'mlp' (A1+A2 乘性) | 'phys' (A3 一阶惯性) | 'both'
        cumsum_out:   C1 增量累积输出参数化 (归一化空间锚定 T[o-1])
        probabilistic: σ 头只由 free 分支产生 (不需要 g(x,0)=0 性质)
        free_action_blind: True=f_free 完全看不到动作 (标准 A1);
                           False 保留 (调试用, 会破坏可归因性)
    """

    def __init__(self, n_feat, target_idx, H, intervention='mlp', n_lag=2,
                 cumsum_out=False, probabilistic=True, free_action_blind=True):
        super().__init__(n_feat, target_idx)
        d = cfg.D_MODEL
        W = cfg.WINDOW_SIZE
        self.H, self.probabilistic, self.cumsum_out = H, probabilistic, cumsum_out
        self.intervention_mode = intervention
        assert free_action_blind, "free_action_blind=False 会破坏 A1 可归因性"

        self.patch = PatchEmbedding(W, cfg.PATCH_LEN, cfg.STRIDE, d)
        self.np = self.patch.n_patches
        self.tcn = PerVariableTCN(self.np, d, cfg.N_TCN_LAYERS, cfg.DROPOUT)
        self.varattn = VariableAttention(d, cfg.N_HEADS, cfg.DROPOUT)
        d_state = n_feat * d

        self.free_head = nn.Sequential(
            nn.Linear(d_state, d * 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, d * 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, H * 2 if probabilistic else H))
        self.interv = make_intervention(intervention, d_state, d, H, n_lag)
        if cumsum_out:
            # free_head 末层零初始化: 初始预测 = anchor 持久化 (强基线),
            # 同时消除 cumsum 带来的 ~200x 梯度放大 (每个增量影响所有后续步)。
            nn.init.zeros_(self.free_head[-1].weight)
            nn.init.zeros_(self.free_head[-1].bias)

    def encode(self, x_hist):
        B = x_hist.shape[0]; d = cfg.D_MODEL
        x_n = self.revin(x_hist, mode='norm')
        tokens = torch.stack([self.patch(x_n[:, :, i]) for i in range(self.n_feat)], 1)
        s = self.tcn(tokens.reshape(B * self.n_feat, self.np, d)).reshape(B, self.n_feat, d)
        s, _ = self.varattn(s)
        return x_n, s.reshape(B, -1)

    def forward(self, x_hist, a_future=None):
        B = x_hist.shape[0]
        x_n, s_flat = self.encode(x_hist)
        raw = self.free_head(s_flat)
        if self.probabilistic:
            raw = raw.reshape(B, self.H, 2)
            free_n, lv_n = raw[..., 0], raw[..., 1]
        else:
            free_n, lv_n = raw.reshape(B, self.H), None

        a = a_future.reshape(B, self.H)
        g_n = self.interv(s_flat, a)                       # g(x,0) ≡ 0

        if self.cumsum_out:
            # C1: free 分支输出解释为归一化空间的逐步增量, 锚定 onset 前一步真实值。
            # RevIN 可逆且 denorm 为仿射 → denorm(anchor_n) 精确等于 T[o-1]。
            # 注意: cumsum 只作用于 free 分支。干预分支输出的已是**响应量级**
            # (InterventionPhysics 的 r(k) 即响应曲线), 再 cumsum 会二次积分。
            anchor = x_n[:, -1, self.target_idx].unsqueeze(1)          # [B,1]
            mu_n = anchor + torch.cumsum(free_n, dim=1) + g_n
        else:
            mu_n = free_n + g_n

        if lv_n is not None:
            lv_n = torch.clamp(lv_n, -6., 20.)
        return self.denorm_out(mu_n, lv_n)

    # --------- 供评测/可解释性使用 (不参与训练) ---------
    @torch.no_grad()
    def intervention_effect(self, x_hist, a_future):
        """直接取干预分支输出 (物理空间), 与 pred(a)−pred(0) 恒等。"""
        B = x_hist.shape[0]
        _, s_flat = self.encode(x_hist)
        g_n = self.interv(s_flat, a_future.reshape(B, self.H))
        ss = self.revin._std[:, :, self.target_idx]
        w = self.revin.weight[self.target_idx]
        return g_n / (w + self.revin.eps) * ss if self.revin.affine else g_n * ss

    @torch.no_grad()
    def physics_params(self, x_hist):
        """A3 专用: 返回 (K, tau) 供可解释性作图 (K 对负荷的函数 vs DiD 分层真值)。"""
        mod = self.interv.phys if self.intervention_mode == 'both' else self.interv
        if not isinstance(mod, InterventionPhysics):
            return None
        _, s_flat = self.encode(x_hist)
        return mod.params(s_flat)


# ==================================================================== B1 变体
class TimeXerCausalLayer(nn.Module):
    """与 exp_025.TimeXerLayer 结构一致 (self → act cross → exog cross → FFN)。
    复制而非继承: exp_025.TimeXerLayer 依赖模块级 H_OUT, 此处 H 显式传参。"""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, dropout, batch_first=True)
        self.act_attn = nn.MultiheadAttention(d, heads, dropout, batch_first=True)
        self.exog_attn = nn.MultiheadAttention(d, heads, dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 4, d))
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.n3 = nn.LayerNorm(d); self.n4 = nn.LayerNorm(d)

    def forward(self, x, ze, za):
        x = x + self.self_attn(x, x, x)[0]; x = self.n1(x)
        glb = x[:, -1:, :]
        glb = glb + self.act_attn(glb, za, za)[0]; glb = self.n2(glb)
        glb = glb + self.exog_attn(glb, ze, ze)[0]; glb = self.n3(glb)
        x = torch.cat([x[:, :-1, :], glb], 1)
        x = x + self.ffn(x); x = self.n4(x)
        return x


class TimeXerCausalWM(CausalWMBase):
    """B1: TimeXer + head_mode 开关 — "精度 vs 因果"权衡的直接对照。

    head_mode='flatten' : 复现现 M9DSP (Linear((np+1)*d, H*2))
                          动作只经 glb 1 个 token → 仅占 head 输入 1/12
    head_mode='glb'     : Linear(d, H*2), 仅取 glb
                          强制所有输出经过唯一携带动作的 token (= M9 v1 的瓶颈)

    两者只差 head, 其余完全相同 → 差异可归因于"动作是否被内生旁路稀释"。
    """

    def __init__(self, n_feat, target_idx, H, head_mode='glb', probabilistic=True):
        super().__init__(n_feat, target_idx)
        d = cfg.D_MODEL
        W = cfg.WINDOW_SIZE
        assert head_mode in ('flatten', 'glb')
        self.H, self.head_mode, self.probabilistic = H, head_mode, probabilistic
        self.patch = PatchEmbedding(W, cfg.PATCH_LEN, cfg.STRIDE, d)
        self.np = self.patch.n_patches
        self.glb_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.exog_lin = nn.Linear(W, d)
        self.act_lin = nn.Linear(H, d)
        self.enc_layers = nn.ModuleList([
            TimeXerCausalLayer(d, cfg.N_HEADS, cfg.DROPOUT) for _ in range(cfg.N_TCN_LAYERS)])
        self.norm = nn.LayerNorm(d)
        d_head = (self.np + 1) * d if head_mode == 'flatten' else d
        self.head = nn.Linear(d_head, H * 2 if probabilistic else H)
        self.exog_idx = [i for i in range(n_feat) if i != target_idx]

    def forward(self, x_hist, a_future=None):
        B = x_hist.shape[0]
        x_n = self.revin(x_hist, mode='norm')
        zt = self.patch(x_n[:, :, self.target_idx])
        x = torch.cat([zt, self.glb_token.expand(B, -1, -1)], 1)
        ze = self.exog_lin(x_n[:, :, self.exog_idx].permute(0, 2, 1))
        za = self.act_lin(a_future.reshape(B, self.H, 1).permute(0, 2, 1))
        for layer in self.enc_layers:
            x = layer(x, ze, za)
        x = self.norm(x)
        feat = x.reshape(B, -1) if self.head_mode == 'flatten' else x[:, -1, :]
        out = self.head(feat)
        if self.probabilistic:
            out = out.reshape(B, self.H, 2)
            return self.denorm_out(out[..., 0], torch.clamp(out[..., 1], -6., 20.))
        return self.denorm_out(out.reshape(B, self.H), None)


# ==================================================================== 自检
@torch.no_grad()
def check_zero_action_identity(model, n_feat, H, device, B=8, tol=1e-5):
    """断言 g(x,0) ≡ 0: pred(x, a=0) 必须与 a 的取值无关到机器精度。

    这是 A1 架构的核心不变量, 训练前后都应成立。若失败说明动作分支引入了 bias。
    """
    model.eval()
    x = torch.randn(B, cfg.WINDOW_SIZE, n_feat, device=device)
    a0 = torch.zeros(B, H, 1, device=device)
    mu_a, _ = model(x, a0)
    mu_b, _ = model(x, torch.zeros_like(a0))
    assert torch.allclose(mu_a, mu_b, atol=tol), "a=0 两次前向不一致 (dropout 未关?)"
    if isinstance(model, ResidualCausalWM):
        g = model.intervention_effect(x, a0)
        assert g.abs().max().item() < tol, \
            f"g(x,0) != 0, max|g|={g.abs().max().item():.3e} — 动作分支存在 bias"
        # 非零动作必须产生非零效应 (否则分支已死)
        g1 = model.intervention_effect(x, torch.randn(B, H, 1, device=device) * 2.0)
        assert g1.abs().max().item() > 1e-8, "g(x,a) 恒为 0 — 干预分支已死"
    return True
