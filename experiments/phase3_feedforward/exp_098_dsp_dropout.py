#!/usr/bin/env python3
"""
exp_098_dsp_dropout.py — ΔSP 动作通道强化训练 (action dropout, 2026-08-05)
============================================================================
问题: exp_097 诊断 M5-DSP 对 ΔSP 响应仅 0.05°C (物理杠杆 0.63°C/°C@180s 的 2.7%)
方案: 训练时以 P=0.4 概率整序列置零 ΔSP 动作 → 模型被迫学"ΔSP→温度"增益
      (状态捷径消除: ΔSP=0 学状态基线, ΔSP=real 学基线+增量)
协议: 同 exp_096 (MSE, 100ep, LR 1e-3, BS 256) + action dropout
评测: 训练后自动跑 ①动作增益测试 (响应幅度 vs 物理 1.31°C, 方向) ②原指标回归
判定: PASS ≥0.40°C 且方向≥80% | GOOD ≥0.65°C | FAIL → B方案(物理校准)
用法: python exp_098_dsp_dropout.py [--smoke]
"""
import os, sys, time, json
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

SMOKE = '--smoke' in sys.argv
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
H_OUT = E.H_OUT
W = E.cfg.WINDOW_SIZE
P_DROPOUT = 0.4

# ===== 数据: 构造 ΔSP 列 =====
raw = E.data_all
N = len(raw)
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_V1 = E.NUMERIC_COLS.index('一级减温调节门阀位')
I_V2 = E.NUMERIC_COLS.index('二级减温调节门阀位')
I_LD = E.NUMERIC_COLS.index('机组负荷')
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)
I_DSP = raw41.shape[1] - 1
n_train = 495407
train_raw = raw41[:n_train]; test_raw = raw41[n_train + 106159:]
print(f"[data] train {train_raw.shape} test {test_raw.shape}")

# ===== M5-DSP =====
class M5DSP(E.DirectWM):
    def __init__(self):
        super().__init__(use_action=True, use_patch=True, per_variable=True,
                         use_varattn=True, probabilistic=False)
        d = E.cfg.D_MODEL
        self.action_enc = nn.Sequential(
            nn.Linear(H_OUT, d * 2), nn.GELU(), nn.Dropout(E.cfg.DROPOUT))

model = M5DSP().to(DEVICE).to(DTYPE)
print(f"[model] M5-DSP + action dropout P={P_DROPOUT}")

# ===== 训练 =====
LR = 1e-3; WD = 1e-5; BS = 256
NEPOCH = 4 if SMOKE else 100
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
N_TRAIN = len(train_raw)
t0 = time.time()
best_mae, best_ep, curve = None, None, []
for ep in range(NEPOCH):
    model.train()
    idxs = np.random.permutation(N_TRAIN - W - H_OUT)
    losses, n = 0.0, 0
    for bi in range(0, len(idxs), BS):
        bid = idxs[bi:bi+BS]
        X = [train_raw[i:i+W, :40] for i in bid]
        A = [train_raw[i+W:i+W+H_OUT, I_DSP] for i in bid]
        Y = [train_raw[i+W:i+W+H_OUT, E.TARGET_IDX] for i in bid]
        x = torch.FloatTensor(np.stack(X)).to(DEVICE)
        a = torch.FloatTensor(np.stack(A)).unsqueeze(-1).to(DEVICE)
        y = torch.FloatTensor(np.stack(Y)).to(DEVICE)
        # === action dropout: 整序列置零 ===
        mask = (torch.rand(len(bid), device=DEVICE) < P_DROPOUT)
        a[mask] = 0.0
        mu, _ = model(x, a)
        loss = nn.functional.mse_loss(mu, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses += loss.item() * len(bid); n += len(bid)
    # test (200 随机窗口)
    model.eval()
    te_errs = []
    with torch.no_grad():
        te_idxs = np.arange(len(test_raw) - W - H_OUT)
        np.random.shuffle(te_idxs)
        for i in te_idxs[:200]:
            xh = torch.FloatTensor(test_raw[i:i+W, :40]).unsqueeze(0).to(DEVICE)
            af = torch.FloatTensor(test_raw[i+W:i+W+H_OUT, I_DSP]).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            mu, _ = model(xh, af)
            tt = test_raw[i+W:i+W+H_OUT, E.TARGET_IDX]
            te_errs.append(np.abs(mu[0].cpu().numpy() - tt).mean())
    te_mae = float(np.mean(te_errs))
    print(f"  ep {ep+1:3d} | train loss {losses/n:.4f} | test MAE {te_mae:.4f}°C | {time.time()-t0:.0f}s")
    if best_mae is None or te_mae < best_mae:
        best_mae, best_ep = te_mae, ep + 1
        os.makedirs('results/exp_098_dsp_dropout/checkpoints', exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep + 1},
                   'results/exp_098_dsp_dropout/checkpoints/best_model.pth')
    curve.append((ep + 1, losses / n, te_mae))

with open('results/exp_098_dsp_dropout/metrics.json', 'w') as f:
    json.dump({'best_epoch': best_ep, 'best_test_mae': best_mae, 'curve': curve,
               'p_dropout': P_DROPOUT}, f, indent=2)
print(f"\n[done] best ep {best_ep} MAE {best_mae:.4f}°C")

# ===== 评测 1: 动作增益测试 (134 事件) =====
ck = torch.load('results/exp_098_dsp_dropout/checkpoints/best_model.pth',
                map_location=DEVICE, weights_only=True)
model.load_state_dict(ck['model_state_dict'])
model.eval()

dsp_abs = np.abs(np.diff(raw[:, I_SP]))
onsets = []
for i in np.where(dsp_abs > 1.0)[0] + 1:
    if not onsets or i - onsets[-1] >= 60:
        onsets.append(i)
stable = [o for o in onsets if o + 60 < N and
          np.abs(np.diff(raw[max(0, o-20):min(N, o+20), I_LD])).max() <= 3.0]
kept = [o for o in stable if np.abs(raw[o:o+61, I_SP] - raw[o, I_SP]).max() <= 0.3]
print(f"[events] {len(kept)}")

def pred(s, a_override=None):
    if s < 0 or s + W + H_OUT >= N:
        return None
    win = torch.FloatTensor(raw[s:s+W]).unsqueeze(0).to(DEVICE)
    if a_override is None:
        a = np.diff(raw41[s+W-1:s+W+H_OUT, I_DSP])
    else:
        a = a_override
    a_f = torch.FloatTensor(a).reshape(1, -1).to(DEVICE)
    with torch.no_grad():
        mu, _ = model(win, a_f)
    return mu[0].cpu().numpy()

resp_amp, resp_dir = [], []
for o in kept:
    s = o - W
    p_real = pred(s)
    p_zero = pred(s, np.zeros(H_OUT))
    if p_real is None or p_zero is None: continue
    dT = p_real[-1] - p_zero[-1]
    ds = raw[o, I_SP] - raw[o-1, I_SP]
    resp_amp.append(abs(dT))
    resp_dir.append(1 if np.sign(dT) == np.sign(ds) else 0)
resp_amp = np.array(resp_amp); resp_dir = np.array(resp_dir)
print("\n===== 动作增益测试 (134 事件) =====")
print(f"  响应幅度: mean {resp_amp.mean():.4f}°C | med {np.median(resp_amp):.4f} | p90 {np.percentile(resp_amp,90):.4f}")
print(f"  方向正确率: {resp_dir.mean()*100:.1f}%")
print(f"  物理基准: 1.31°C (0.63°C/°C@180s × 2.07°C) → 学到比例 {resp_amp.mean()/1.31*100:.0f}%")
verdict = 'GOOD (≥0.65)' if resp_amp.mean() >= 0.65 else ('PASS (≥0.40)' if resp_amp.mean() >= 0.40 else 'FAIL (<0.40 → B方案)')
print(f"  判定: {verdict}")

# ===== 评测 2: 原指标回归 (SP 事件 MAE + 方向, 对标 exp_097) =====
errs, dirs = [], []
for o in kept:
    s = o - W
    p = pred(s)
    if p is None: continue
    actual = raw[o:o+H_OUT, I_T]
    errs.append(np.abs(p - actual).mean())
    prev_T = raw[o-1, I_T]
    dirs.append(1 if np.sign(p[-1]-prev_T) == np.sign(actual[-1]-prev_T) else 0)
errs, dirs = np.array(errs), np.array(dirs)
print(f"\n===== 原指标回归 (对标 exp_097: MAE 0.301 方向 87%) =====")
print(f"  全事件 MAE {errs.mean():.3f} | 方向 {dirs.mean()*100:.0f}%")
print(f"  回归检查: MAE 劣化 {errs.mean()-0.301:+.3f} ({'✓≤+0.05' if errs.mean()-0.301 <= 0.05 else '✗>+0.05'})")

with open('results/exp_098_dsp_dropout/gain_test.json', 'w') as f:
    json.dump({'resp_mean': float(resp_amp.mean()), 'resp_med': float(np.median(resp_amp)),
               'resp_p90': float(np.percentile(resp_amp, 90)), 'dir': float(resp_dir.mean()),
               'phys_ref': 1.31, 'mae': float(errs.mean()), 'dir_orig': float(dirs.mean()),
               'verdict': verdict}, f, indent=2)
print('Saved: results/exp_098_dsp_dropout/gain_test.json')
