#!/usr/bin/env python3
"""
1c 因果证实 — 模型响应曲线 vs 事件研究真值曲线
================================================
审稿人 R1-M2/R3-M2: "动作条件化的因果证据缺失, 且评测时标错误"。

方法:
1. 模型侧: 用 eval_sensitivity 的扰动协议 (历史末位+未来首位动作 ±Δ), 但输出
   全时标 t1..t13 响应曲线 (而非只 1/3/8/12)
2. 真值侧: 事件研究趋势校正曲线 (去共因后的物理增量), 开阀/关阀双侧
3. 对比: 方向匹配 (t≥8 符号), 幅度比 (模型/真值), 时标匹配 (峰值位置)

模型: L3_W1_l0.00 (无正则=物理一致) vs L3_W1_l0.10 (有正则=伪物理)
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import config as cfg
from data_loader import load_raw_data

# 关键: 覆盖 exp_016 模块的 LAGS 为 L3, 使 WorldModel_Lag 实例化为滞后结构
import experiments.phase1_dynamics.exp_016_ablation_sweep as exp016
exp016.LAGS = [0, 3, 6, 9]
exp016.N_LAGS = len(exp016.LAGS)
from experiments.phase1_dynamics.exp_016_ablation_sweep import WorldModel_Lag

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state_data, delta_actions, valve_abs = load_raw_data()
raw_data = np.concatenate([state_data, valve_abs], axis=1)
n_total = len(raw_data)
n_train = int(n_total * 0.70); n_val_end = int(n_total * 0.85)
test_data = raw_data[n_val_end:]

H = 13  # 敏感性展开步数


@torch.no_grad()
def model_response_curve(model, raw, adim, delta, n=200, seed=42):
    """扰动历史末位+未来首位动作 → 全时标 ΔT 曲线 (与 eval_sensitivity 同协议)"""
    model.eval(); W = cfg.WINDOW_SIZE; N = len(raw)
    np.random.seed(seed); idxs = np.random.choice(range(N-W-H), n, replace=False)
    curves = np.zeros((n, H))
    for j, i in enumerate(idxs):
        sh = raw[i:i+W, :cfg.N_STATE].copy()
        ah = raw[i:i+W, cfg.N_STATE:].copy()
        fa = raw[i+W:i+W+H, cfg.N_STATE:].copy()
        xt = torch.FloatTensor(np.concatenate([sh, ah], 1)).unsqueeze(0).to(DEVICE)
        at = torch.FloatTensor(fa).unsqueeze(0).to(DEVICE)
        trb = model.rollout(xt, at, mode='sliding')
        bp = trb[0,:,cfg.TARGET_IDX].cpu().numpy()
        ap = ah.copy(); ap[-1, adim] = np.clip(ap[-1, adim]+delta, 0, 100)
        fap = fa.copy(); fap[0, adim] = np.clip(fap[0, adim]+delta, 0, 100)
        xt = torch.FloatTensor(np.concatenate([sh, ap], 1)).unsqueeze(0).to(DEVICE)
        at = torch.FloatTensor(fap).unsqueeze(0).to(DEVICE)
        trp = model.rollout(xt, at, mode='sliding')
        pp = trp[0,:,cfg.TARGET_IDX].cpu().numpy()
        curves[j] = pp - bp
    return curves.mean(0)


def event_study_truth(thr=3.0, pre=3, post=13):
    """事件研究趋势校正曲线 (去共因物理增量), 开阀+thr / 关阀-thr"""
    from event_study_valve_close import find_events, trend_adjusted
    up, dn = find_events(valve_abs[:, 1], thr, thr)
    ru = trend_adjusted(state_data[:, cfg.TARGET_IDX], up, pre=pre)
    rd = trend_adjusted(state_data[:, cfg.TARGET_IDX], dn, pre=pre)
    # trend_adjusted 返回 61 步, 取前 post+1 步
    return ru.mean(0)[:post+1], rd.mean(0)[:post+1], len(up), len(dn)


def load_model(config_name):
    exp_dir = f"results/exp_016_{config_name}"
    ck = torch.load(f"{exp_dir}/checkpoints/best_model.pth", map_location=DEVICE, weights_only=True)
    m = WorldModel_Lag().to(DEVICE)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m


if __name__ == '__main__':
    print("="*100)
    print("1c 因果证实: 模型响应曲线 vs 事件研究真值 (二级减温阀 ±10)")
    print("="*100)

    # 事件研究真值 (趋势校正, 物理增量)
    ru, rd, n_up, n_dn = event_study_truth(thr=3.0)
    t = np.arange(H) * 10

    print(f"\n事件研究真值 (趋势校正, ±3%, 开阀{n_up}次/关阀{n_dn}次):")
    print(f"  开阀(物理降温): " + " ".join(f"{x:+.2f}" for x in ru))
    print(f"  关阀(物理升温): " + " ".join(f"{x:+.2f}" for x in rd))

    for cname in ['L3_W1_l0.00', 'L3_W1_l0.10']:
        model = load_model(cname)
        # 开阀 +10 → 应降温 (负); 关阀 -10 → 应升温 (正)
        cur_open = model_response_curve(model, test_data, adim=1, delta=10.0)   # 开阀
        cur_close = model_response_curve(model, test_data, adim=1, delta=-10.0) # 关阀

        print(f"\n{'='*100}")
        print(f"模型 {cname}:")
        print(f"  开阀+10 → ΔT: " + " ".join(f"{x:+.3f}" for x in cur_open))
        print(f"  关阀-10 → ΔT: " + " ".join(f"{x:+.3f}" for x in cur_close))

        # 与真值对比 (t>=8 物理响应区)
        print(f"\n  [t>=8 物理响应区对比]")
        print(f"  {'step':>5} {'time':>6} | {'真值(开阀)':>10} {'模型(开阀)':>10} {'匹配':>4} | {'真值(关阀)':>10} {'模型(关阀)':>10} {'匹配':>4}")
        for k in [8, 9, 10, 11, 12]:
            match_o = '✓' if np.sign(ru[k]) == np.sign(cur_open[k]) else '✗'
            match_c = '✓' if np.sign(rd[k]) == np.sign(cur_close[k]) else '✗'
            print(f"  {k:>5} {t[k]:>5.0f}s | {ru[k]:+10.3f} {cur_open[k]:+10.3f} {match_o:>4} | {rd[k]:+10.3f} {cur_close[k]:+10.3f} {match_c:>4}")

        # 幅度比 (t12)
        r_open = abs(cur_open[12]) / abs(ru[12]) if ru[12] != 0 else float('nan')
        r_close = abs(cur_close[12]) / abs(rd[12]) if rd[12] != 0 else float('nan')
        print(f"\n  幅度比 t12 (模型/真值): 开阀 {r_open:.2f} | 关阀 {r_close:.2f} (1.0=量级正确)")

        # 时标匹配: 模型响应峰值位置 vs 真值 (开阀谷值)
        imin_m = np.argmin(cur_open[:13]); imin_t = np.argmin(ru[:13])
        print(f"  开阀谷值位置: 模型 t{imin_m} ({t[imin_m]:.0f}s) vs 真值 t{imin_t} ({t[imin_t]:.0f}s)")
