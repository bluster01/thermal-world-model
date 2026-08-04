#!/usr/bin/env python3
"""
exp_081_phys_reg.py — 方案3a: 物理一致性正则 (hinge)
=====================================================
对开阀扰动样本 (二级阀 +5%, 首步/持续混合): 
  hinge = mean(relu(ΔT_pred)) — 开阀→预测温度必须 ≤0 (物理: 减温阀开大降温)
  loss = NLL + λ·hinge (λ 可扫)
只约束二级阀 (一级阀物理方向弱/不确定)。对比 M7/M7aug/M12 的方向一致性。
"""
import os, sys, time
import numpy as np
import torch

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
LAMBDA_H = float(os.environ.get('LAMBDA_H', '1.0'))   # hinge 权重
EXP_DIR = f'results/exp_025_M7phys_l{LAMBDA_H}'
os.makedirs(f'{EXP_DIR}/checkpoints', exist_ok=True)
W = cfg.WINDOW_SIZE; H = E.H_OUT; BS = 64

def train_epoch_phys(model, raw, opt, crit):
    """NLL + λ·hinge(配对扰动: 开阀+5预测 ≤ 关阀−5预测) — 差值正则无法整体偏移作弊
    扰动=首步/持续混合, 二级阀"""
    model.train(); N = len(raw)
    total = 0.
    for _ in range(E.STEPS):
        idxs = np.random.randint(0, N - W - H, size=BS)
        xh, af, tt = [], [], []
        for i in idxs:
            xh.append(raw[i:i+W]); af.append(raw[i+W:i+W+H, E.VALVE_IDX])
            tt.append(raw[i+W:i+W+H, E.TARGET_IDX])
        x_hist = torch.FloatTensor(np.stack(xh)).to(DEVICE)
        a_fut = torch.FloatTensor(np.stack(af)).to(DEVICE)
        t_true = torch.FloatTensor(np.stack(tt)).to(DEVICE)
        # 50% 样本: 配对扰动 (二级阀 +5 与 −5, 相同模式)
        aug = torch.rand(BS, device=DEVICE) < 0.5
        a_pos, a_neg = a_fut.clone(), a_fut.clone()
        if aug.any():
            first = aug & (torch.rand(BS, device=DEVICE) < 0.6)
            step = aug & ~first
            if first.any():
                a_pos[first, 0, 1] = (a_pos[first, 0, 1] + 5).clamp(0, 100)
                a_neg[first, 0, 1] = (a_neg[first, 0, 1] - 5).clamp(0, 100)
            if step.any():
                for j in step.nonzero()[:, 0]:
                    k = int(torch.randint(1, H, (1,)))
                    a_pos[j, k:, 1] = (a_pos[j, k:, 1] + 5).clamp(0, 100)
                    a_neg[j, k:, 1] = (a_neg[j, k:, 1] - 5).clamp(0, 100)
        opt.zero_grad()
        mu_aug, lv_aug = model(x_hist, a_pos)
        w = torch.linspace(1.0, 0.6, H, device=DEVICE)
        loss = (w * crit(mu_aug, lv_aug, t_true).mean(dim=0)).sum() / H
        if aug.any():
            mu_neg, _ = model(x_hist, a_neg)             # 关阀样本 (有梯度, 差值正则)
            hinge = torch.relu(mu_aug[aug] - mu_neg[aug]).mean()
            loss = loss + LAMBDA_H * hinge
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); total += loss.item()
    return total / E.STEPS

def check_causal(model, n_wins=50, onsets=(3,), amps=(10.0, 20.0)):
    """新协议 (2026-08-03 修方向): 持续阶跃 onset=3 步起 (覆盖60-90s物理滞后视野),
    采样点相对 onset (onset+1/3/8/12), 全 horizon ΔT 曲线 + 5-95 分位带
    判据: ①时标(onset前≈0) ②方向(onset后≤0) ③幅度带(vs ARX −0.01~−0.06°C/10%@60-90s) ④线性度(+20%≈2×+10%)"""
    model.eval()
    np.random.seed(7)
    idxs = np.random.choice(range(len(E.test_raw) - W - H), n_wins, replace=False)
    out = {}
    for amp in amps:
        dT = np.zeros((n_wins, H))
        for n, i in enumerate(idxs):
            x_hist = torch.FloatTensor(E.test_raw[i:i+W]).unsqueeze(0).to(DEVICE)
            a_fut = torch.FloatTensor(E.test_raw[i+W:i+W+H, E.VALVE_IDX]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mu_b, _ = model(x_hist, a_fut)
            bp = mu_b[0].cpu().numpy()
            a2 = a_fut.clone()
            for o in onsets:
                a2[0, o:, 1] = torch.clamp(a2[0, o:, 1] + amp, 0, 100)
            with torch.no_grad():
                mu2, _ = model(x_hist, a2)
            pp2 = mu2[0].cpu().numpy()
            dT[n] = pp2 - bp
        out[f'dT{int(amp)}'] = {'mean': dT.mean(0), 'lo': np.percentile(dT, 5, 0),
                                'hi': np.percentile(dT, 95, 0), 'onset': onsets[0]}
    return out

def judge(res, name):
    """四判据: ①时标 ②方向 ③幅度带(ARX基准) ④线性度"""
    o = res['dT10']['onset']
    m10 = res['dT10']['mean']; m20 = res['dT20']['mean']
    pre = float(np.abs(m10[:o]).mean())                     # ① onset前应≈0
    post = m10[o+1:o+5]                                      # ② onset+1..+4 (20-50s)
    late = m10[o+3:o+7]                                      # ③ onset+3..+6 (60-90s, ARX窗口)
    lin = m20[o+3:o+7].mean() / (m10[o+3:o+7].mean() + 1e-9)  # ④ 线性度
    print(f"\n[{name}] 判据 (持续阶跃 +10% V2, onset={o}步={o*10}s):")
    print(f"  ①时标: onset前(|k<{o}) ΔT均值 = {pre:+.4f}°C  {'✓≈0' if abs(pre) < 0.03 else '✗≠0 (压缩伪影)'}")
    d_ok = bool((post <= 0).all())
    print(f"  ②方向: onset后 k={o+1}..{o+4} ({10*(o+1)}-{10*(o+4)}s) = {[f'{v:+.3f}' for v in post]}  {'✓≤0' if d_ok else '✗>0 (翻转)'}")
    amp_ok = -0.06 <= late.mean() <= -0.01
    print(f"  ③幅度: 60-90s窗口均值 = {late.mean():+.4f}°C (ARX因果基准 −0.01~−0.06)  {'✓带内' if amp_ok else '✗带外'}")
    print(f"  ④线性度: ΔT(+20%)/ΔT(+10%) = {lin:.2f} (期望≈2.0)  {'✓' if 1.2 <= lin <= 3.0 else '✗'}")
    return {'时标': abs(pre) < 0.03, '方向': d_ok, '幅度': amp_ok, '线性度': 1.2 <= lin <= 3.0}

if __name__ == '__main__':
    # ===== 重读模式 (2026-08-03 修方向): 不重训, 用新判据评测已训 checkpoint =====
    # 三个已训模型: M7 / M7phys(λ=1.0) / M13-DeepONet
    models = {
        'M7': ('results/exp_025_M7/checkpoints/best_model.pth', 'M7'),
        'M7phys': ('results/exp_025_M7phys_l1.0/checkpoints/best_model.pth', 'M7'),
        'M13': ('results/exp_025_M13/checkpoints/best_model.pth', 'M13'),
    }
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'M7': '#c0504d', 'M7phys': '#4f81bd', 'M13': '#8064a2'}
    for tag, (ckp, arch) in models.items():
        model = (E.build_model('M7') if arch == 'M7' else
                 __import__('exp_082_deeponet_wm', fromlist=['DeepONetWM']).DeepONetWM()).to(DEVICE)
        ck = torch.load(ckp, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ck['model_state_dict']); model.eval()
        res = check_causal(model)
        judge(res, tag)
        o = res['dT10']['onset']
        m10, lo, hi = res['dT10']['mean'], res['dT10']['lo'], res['dT10']['hi']
        t_ax = np.arange(H) * 10
        ax.plot(t_ax, m10, colors[tag], lw=1.8, label=tag)
        ax.fill_between(t_ax, lo, hi, color=colors[tag], alpha=0.12)
    ax.axhline(0, color='gray', lw=0.7)
    ax.axvline(30, color='gray', lw=0.8, ls=':')
    # ARX 因果基准带 (−0.01~−0.06 °C @ 60-90s per 10% V2)
    ax.axhspan(-0.06, -0.01, xmin=0.38, xmax=0.55, color='green', alpha=0.15)
    ax.text(145, -0.028, 'ARX causal band\n(−0.01~−0.06 °C @60-90s)', fontsize=7.5, color='green', va='center')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('ΔT from sustained +10% V2 step (°C)')
    ax.set_title('Sustained valve step response (onset=30s): full-horizon curves ±5-95% band')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig('figures/fig_rejudge_curves.png', dpi=180, bbox_inches='tight')
    print('\nSaved: figures/fig_rejudge_curves.png')
