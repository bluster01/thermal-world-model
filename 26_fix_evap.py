#!/usr/bin/env python3
"""26_fix_evap.py: FIXB 喷水蒸发干燥动力学 — E0Evap 灰盒本体结构改动 + 重辨识 3 seeds

设计稿 FIXB_DESIGN.md (预注册冻结):
  B1: 湿态窗口 sh1_out 首步偏差 ≤8°C (现21-25)
  B2: 湿态开环耦合阶跃 τ63∈[240,900]s (无残差)
  B3: e0-evap rollout ≤10 (现 e0 12.7)
  B4: 干态开环 K<0
  B5审计: 学得 τ_evap/α_w/m_dry0 + 干态退化检查
结构: 液滴状态 m_liq (dm/dt=Dsw−m/τ_evap) + 壁面供热 q_wall=α_w·(Tm−Tsat)·(1−dry)
      + 干燥度门控输出 T=Tsat+dry·(T_of_ph(h_mix+q_wall/D)−Tsat)
"""
import importlib.util
import json
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _imp(p, n):
    spec = importlib.util.spec_from_file_location(n, os.path.join(os.getcwd(), p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t02 = _imp("02_train.py", "t02")
r09 = _imp("09_residual.py", "r09")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T_of_ph = t02.T_of_ph
tsat_poly = t02.tsat_poly

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
TRAIN_N = t02.TRAIN_N
VAL_N = t02.VAL_N
SEQ = t02.SEQ
P_CRIT = t02.P_CRIT
N = 600
DT = 10.0
V2 = 6
OP_WET, OP_DRY = 40161, 40437

EVAP_PRIORS = dict(t02.E0_PRIORS)
EVAP_PRIORS.update({
    "tau_evap": 15.0,     # s 蒸发时间常数 (共享)
    "aW1": 150.0,         # kW/K 减温器1出口湿混合物壁面给热
    "aW2": 150.0,         # kW/K 减温器2出口
    "m_dry0": 30.0,       # kg 干燥阈值液滴量 (dry=σ(3·(m_dry0−m)/m_dry0))
})
EVAP_KEYS = list(EVAP_PRIORS.keys())


def sp_init(target):
    if target > 30.0:
        return float(target)
    return float(np.log(np.expm1(target)))


class E0Evap(nn.Module):
    """灰盒 + 蒸发干燥状态。共享参数从已训 e0 热启动。"""

    def __init__(self, warm_state=None):
        super().__init__()
        self.raw = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(0.0 if k.startswith("b") else sp_init(1.0),
                                         dtype=torch.float32))
            for k in EVAP_KEYS})
        self._g = {"M": ["M0", "M1", "M2"], "UA": ["UA0", "UA1", "UA2"],
                   "Cm": ["Cm0", "Cm1", "Cm2"], "k": ["k0", "k1", "k2"],
                   "dTm": ["dTm0", "dTm1", "dTm2"]}
        if warm_state is not None:
            own = self.state_dict()
            for k, v in warm_state.items():
                if k in own and own[k].shape == v.shape:
                    own[k] = v.clone()
            self.load_state_dict(own)

    def val(self, k):
        if k.startswith("b"):
            return EVAP_PRIORS[k] * torch.tanh(self.raw[k])
        return EVAP_PRIORS[k] * F.softplus(self.raw[k])

    def tri(self, grp):
        return torch.stack([EVAP_PRIORS[k] * F.softplus(self.raw[k]) for k in self._g[grp]], dim=0)

    def k_of(self, pm):
        a = torch.sigmoid((P_CRIT - pm) / t02.K_BLEND)
        dpm = pm - t02.PM_REF
        k0 = a * self.val("k0") + (1.0 - a) * self.val("k0d") + self.val("b0") * dpm
        k1 = a * self.val("k1") + (1.0 - a) * self.val("k1d") + self.val("b1") * dpm
        k2 = a * self.val("k2") + (1.0 - a) * self.val("k2d") + self.val("b2") * dpm
        return torch.stack([k0.expand_as(pm), k1.expand_as(pm), k2.expand_as(pm)], dim=0)

    def th_of(self, pm):
        a = torch.sigmoid((P_CRIT - pm) / t02.K_BLEND)
        return (a * self.val("th1") + (1.0 - a) * self.val("th1d"),
                a * self.val("th2") + (1.0 - a) * self.val("th2d"))

    def integrate(self, exo, h, Tm, rB, m1, m2, steps, return_states=False):
        """exo: (B,steps,9); h/Tm:(3,B); rB:(B,); m1,m2:(B,) 液滴质量。返回 (B,steps,5)。"""
        Bsz = exo.shape[0]
        M = self.tri("M")[:, None]
        UA = self.tri("UA")[:, None]
        Cm = self.tri("Cm")[:, None]
        tauB = self.val("tauB")
        tau_evap = self.val("tau_evap")
        aW1 = self.val("aW1")
        aW2 = self.val("aW2")
        m_dry0 = self.val("m_dry0")
        D, uB, pm, Tm_sep, Tfw, v1, v2, p_out, W = [exo[:, :, j] for j in range(9)]
        h_sw = t02.hliq_of_T(Tfw)
        p0 = pm + (p_out - pm) / 3.0
        p1 = pm + 2.0 * (p_out - pm) / 3.0
        hsep = t02.h_sep_of(pm, Tm_sep)
        out_list = []
        th1_0, th2_0 = self.th_of(pm[:, 0])
        s_den0 = th1_0 * v1[:, 0] + th2_0 * v2[:, 0] + 1e-6
        W0 = W[:, 0].clamp(min=0.0)
        Dsw1 = t02.KAPPA * W0 * (th1_0 * v1[:, 0]) / s_den0
        Dsw2 = t02.KAPPA * W0 * (th2_0 * v2[:, 0]) / s_den0
        hm1 = (D[:, 0] * h[0] + Dsw1 * h_sw[:, 0]) / (D[:, 0] + Dsw1 + 1e-6)
        hm2 = (D[:, 0] * h[1] + Dsw2 * h_sw[:, 0]) / (D[:, 0] + Dsw2 + 1e-6)
        for t in range(steps):
            k_t = self.k_of(pm[:, t])
            th1_t, th2_t = self.th_of(pm[:, t])
            s_den = th1_t * v1[:, t] + th2_t * v2[:, t] + 1e-6
            Wt = W[:, t].clamp(min=0.0)
            for _ in range(t02.N_SUB):
                ts = T_of_ph(torch.stack([p0[:, t], p1[:, t], p_out[:, t]]), h)
                # 干燥度与壁面供热 (湿混合物向管壁吸热, 金属失热)
                dry1 = torch.sigmoid(3.0 * (m_dry0 - m1) / m_dry0)
                dry2 = torch.sigmoid(3.0 * (m_dry0 - m2) / m_dry0)
                tsat0 = t02.tsat_poly(p0[:, t])
                tsat1 = t02.tsat_poly(p1[:, t])
                q_w1 = aW1 * (Tm[0] - tsat0) * (1.0 - dry1)
                q_w2 = aW2 * (Tm[1] - tsat1) * (1.0 - dry2)
                Q = UA * (Tm - ts)
                Tm0_in = (k_t[0] * rB / 3600.0 + UA[0] * ts[0] - q_w1) / Cm[0]
                Tm1_in = (k_t[1] * rB / 3600.0 + UA[1] * ts[1] - q_w2) / Cm[1]
                Tm2_in = (k_t[2] * rB / 3600.0 + UA[2] * ts[2]) / Cm[2]
                Tm_in = torch.stack([Tm0_in, Tm1_in, Tm2_in])
                Tm = (Tm + t02.DT_SUB * Tm_in) / (1.0 + t02.DT_SUB * UA / Cm)
                # 下游段入口焓 = 混合焓 + 壁热 (干燥后总焓)
                h_in1 = hm1 + q_w1 / (D[:, t] + 1e-6)
                h_in2 = hm2 + q_w2 / (D[:, t] + 1e-6)
                hin = torch.stack([hsep[:, t], h_in1, h_in2])
                h = (h + t02.DT_SUB * (D[:, t][None, :] * hin + Q) / M) / (
                    1.0 + t02.DT_SUB * D[:, t][None, :] / M)
                h = t02._ste_clamp(h, t02.H_LO, t02.H_HI)
                Dsw1 = t02.KAPPA * Wt * (th1_t * v1[:, t]) / s_den
                Dsw2 = t02.KAPPA * Wt * (th2_t * v2[:, t]) / s_den
                hm1 = (D[:, t] * h[0] + Dsw1 * h_sw[:, t]) / (D[:, t] + Dsw1 + 1e-6)
                hm2 = (D[:, t] * h[1] + Dsw2 * h_sw[:, t]) / (D[:, t] + Dsw2 + 1e-6)
                # 液滴质量: 喷水喂入 + 蒸发消耗
                m1 = m1 + t02.DT_SUB * (Dsw1 - m1 / tau_evap)
                m2 = m2 + t02.DT_SUB * (Dsw2 - m2 / tau_evap)
                m1 = m1.clamp(min=0.0)
                m2 = m2.clamp(min=0.0)
                rB = rB + t02.DT_SUB * (uB[:, t] - rB) / tauB
            # 输出: 减温器出口 = 干燥度门控的壁热过热
            dry1 = torch.sigmoid(3.0 * (m_dry0 - m1) / m_dry0)
            dry2 = torch.sigmoid(3.0 * (m_dry0 - m2) / m_dry0)
            tsat0 = t02.tsat_poly(p0[:, t])
            tsat1 = t02.tsat_poly(p1[:, t])
            q_w1o = aW1 * (Tm[0] - tsat0) * (1.0 - dry1)
            q_w2o = aW2 * (Tm[1] - tsat1) * (1.0 - dry2)
            h_o1 = hm1 + q_w1o / (D[:, t] + 1e-6)
            h_o2 = hm2 + q_w2o / (D[:, t] + 1e-6)
            T_o1 = tsat0 + dry1 * (T_of_ph(p0[:, t], h_o1) - tsat0)
            T_o2 = tsat1 + dry2 * (T_of_ph(p1[:, t], h_o2) - tsat1)
            p = torch.stack([p0[:, t], p0[:, t], p1[:, t], p1[:, t], p_out[:, t]])
            hh = torch.stack([h[0], h_o1, h[1], h_o2, h[2]])
            T_all5 = T_of_ph(p, hh)
            T_out = torch.stack([T_all5[0],
                                 tsat0 + dry1 * (T_all5[1] - tsat0),
                                 T_all5[2],
                                 tsat1 + dry2 * (T_all5[3] - tsat1),
                                 T_all5[4]])
            out_list.append(T_out)
        out = torch.stack(out_list, dim=2).permute(1, 2, 0)
        if return_states:
            return out, h, Tm, rB, m1, m2
        return out


def init_states_evap(model, init_rows, obs_T):
    """窗口起点状态 + 液滴稳态初值 m = Dsw·τ_evap。"""
    D = init_rows[:, 0]
    pm = init_rows[:, 2]
    p_out = init_rows[:, 7]
    p0 = pm + (p_out - pm) / 3.0
    p1 = pm + 2.0 * (p_out - pm) / 3.0
    h0 = t02.h_of_pT(p0, obs_T[:, 0])
    h1 = t02.h_of_pT(p1, obs_T[:, 2])
    h2 = t02.h_of_pT(p_out, obs_T[:, 4])
    ts0 = t02.T_of_ph(p0, h0)
    ts1 = t02.T_of_ph(p1, h1)
    ts2 = t02.T_of_ph(p_out, h2)
    ts = torch.stack([ts0, ts1, ts2])
    rB0 = init_rows[:, 1].clone()
    Tm = ts + model.k_of(pm) * rB0[None, :] / 3600.0 / model.tri("UA")[:, None] \
        + model.tri("dTm")[:, None]
    # 液滴稳态: m = Dsw·τ_evap (喷水率×蒸发时间)
    tau_evap = model.val("tau_evap")
    th1_0, th2_0 = model.th_of(pm)
    v1_0 = init_rows[:, 5]
    v2_0 = init_rows[:, 6]
    s_den0 = th1_0 * v1_0 + th2_0 * v2_0 + 1e-6
    W0 = init_rows[:, 8].clamp(min=0.0)
    Dsw1 = t02.KAPPA * W0 * (th1_0 * v1_0) / s_den0
    Dsw2 = t02.KAPPA * W0 * (th2_0 * v2_0) / s_den0
    m1 = Dsw1 * tau_evap
    m2 = Dsw2 * tau_evap
    return torch.stack([h0, h1, h2]), Tm, rB0, m1, m2


def train_evap(df, seed, warm_state, fast=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_s = 25 if fast else 5
    va_s = 100 if fast else 20
    Xtr, Ytr, Itr, Itr_T = t02.e0_build_windows(df, 0, TRAIN_N, tr_s)
    Xva, Yva, Iva, Iva_T = t02.e0_build_windows(df, TRAIN_N, TRAIN_N + VAL_N, va_s)
    print(f"[evap s{seed}] train={len(Xtr)} val={len(Xva)}", flush=True)
    model = E0Evap(warm_state).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    w5 = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=DEVICE)
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr).to(DEVICE)
    Itr_t = torch.from_numpy(Itr).to(DEVICE)
    ItrT_t = torch.from_numpy(Itr_T).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    Yva_t = torch.from_numpy(Yva).to(DEVICE)
    Iva_t = torch.from_numpy(Iva).to(DEVICE)
    IvaT_t = torch.from_numpy(Iva_T).to(DEVICE)

    def fwd(exo, init_rows, obs_T):
        h, Tm, rB, m1, m2 = init_states_evap(model, init_rows, obs_T)
        return model.integrate(exo, h, Tm, rB, m1, m2, exo.shape[1])

    max_ep = 2 if fast else 50
    best_va, best_state, patience = 1e9, None, 0
    n_batch = len(Xtr_t) // 256
    n_ep_done = 0
    for ep in range(max_ep):
        n_ep_done = ep + 1
        model.train()
        perm = torch.randperm(len(Xtr_t), device=DEVICE)
        for b in range(n_batch):
            i = perm[b * 256: (b + 1) * 256]
            pred = fwd(Xtr_t[i], Itr_t[i], ItrT_t[i])
            loss = (((pred - Ytr_t[i]) ** 2) * w5).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = fwd(Xva_t, Iva_t, IvaT_t)
            va = (((pred - Yva_t) ** 2) * w5).mean().item()
        if va < best_va:
            best_va, patience = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT, f"model_e0_evap_seed{seed}.pt"))
    print(f"[evap s{seed}] {n_ep_done}ep val={best_va:.3f} τ_evap={model.val('tau_evap').item():.1f}s "
          f"aW1={model.val('aW1').item():.0f} aW2={model.val('aW2').item():.0f} "
          f"m_dry0={model.val('m_dry0').item():.0f}kg", flush=True)
    return model, best_va


def main():
    df = r09.load_e0_df()
    warm = torch.load(os.path.join(OUT, "model_e0_seed0.pt"), map_location=DEVICE, weights_only=True)
    summ = {"train": {}, "judge": {}}
    for sd in (0, 1, 2):
        model, va = train_evap(df, sd, warm)
        summ["train"][str(sd)] = {"val_mse": round(va, 4),
                                  "tau_evap": round(float(model.val("tau_evap").item()), 1),
                                  "aW1": round(float(model.val("aW1").item()), 0),
                                  "aW2": round(float(model.val("aW2").item()), 0),
                                  "m_dry0": round(float(model.val("m_dry0").item()), 1)}

    # 验证用 seed0
    model0 = E0Evap(warm).to(DEVICE)
    model0.load_state_dict(torch.load(os.path.join(OUT, "model_e0_evap_seed0.pt"),
                                      map_location=DEVICE, weights_only=True))
    model0.eval()
    for p in model0.parameters():
        p.requires_grad_(False)

    E = df[r09.E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    Ea = E.to_numpy(np.float32)
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)
    pm_all = Ea[:, 2]

    # B1: 湿态窗口 sh1_out 首步偏差
    wet_idx = np.where((pm_all[START: START + t02.ROLL_STEPS] <= P_CRIT))[0] + START
    wet_idx = wet_idx[wet_idx + 1 < len(T_all)]
    errs = []
    with torch.no_grad():
        for r in wet_idx:
            row = Ea[r]
            obs = T_all[r]
            exo_t = torch.tensor(Ea[r: r + 1], device=DEVICE)[None, :, :]
            h, Tm, rB, m1, m2 = init_states_evap(
                model0, torch.tensor(row, device=DEVICE)[None, :],
                torch.tensor(obs, device=DEVICE)[None, :])
            out, *_ = model0.integrate(exo_t, h, Tm, rB, m1, m2, 1)
            errs.append(float(out[0, 0, 1] - T_all[r + 1, 1]))
    errs = np.array(errs)
    bias = float(np.mean(errs))
    rmse1 = float(np.sqrt(np.mean(errs ** 2)))
    print(f"[B1] wet sh1_out first-step: n={len(errs)} bias={bias:.2f}°C rmse={rmse1:.2f}°C", flush=True)

    # B3: rollout (无残差)
    preds = np.empty((t02.ROLL_STEPS, 5), dtype=np.float32)
    T_sens_state = None
    h, Tm, rB, m1, m2 = None, None, None, None, None
    with torch.no_grad():
        for t in range(t02.ROLL_STEPS):
            row = Ea[START + t]
            exo_t = torch.tensor(row, device=DEVICE)[None, None, :]
            if t == 0:
                obs = T_all[START]
                h, Tm, rB, m1, m2 = init_states_evap(
                    model0, torch.tensor(row, device=DEVICE)[None, :],
                    torch.tensor(obs, device=DEVICE)[None, :])
            out, h, Tm, rB, m1, m2 = model0.integrate(exo_t, h, Tm, rB, m1, m2, 1,
                                                     return_states=True)
            preds[t] = out[0, 0].cpu().numpy()
    truths = T_all[START: START + t02.ROLL_STEPS]
    rmse_main = float(np.sqrt(np.mean((preds[:, 4] - truths[:, 4]) ** 2)))
    dry_mask = pm_all[START: START + t02.ROLL_STEPS] > P_CRIT
    rmse_dry = float(np.sqrt(np.mean((preds[dry_mask, 4] - truths[dry_mask, 4]) ** 2)))
    rmse_wet = float(np.sqrt(np.mean((preds[~dry_mask, 4] - truths[~dry_mask, 4]) ** 2)))
    print(f"[B3] rollout main={rmse_main:.2f} dry={rmse_dry:.2f} wet={rmse_wet:.2f}", flush=True)

    # B2/B4: 开环阶跃 (v2+5% + W联动, 无残差)
    pm_seg = pm_all[START: START + t02.ROLL_STEPS]
    k_w_state = {}
    for state, msk in (("wet", pm_seg <= P_CRIT), ("dry", pm_seg > P_CRIT)):
        sub = Ea[START: START + t02.ROLL_STEPS][msk]
        A = np.stack([sub[:, 5], sub[:, 6], np.ones(len(sub))], 1)
        coef, _, _, _ = np.linalg.lstsq(A, sub[:, 8], rcond=None)
        k_w_state[state] = float(np.clip(coef[1] / np.mean(sub[:, 8]), 0.5, 4.0))

    def run_step(row_idx, d_v2, W_mul):
        row, obs = Ea[row_idx], T_all[row_idx]
        exo = torch.tensor(row, device=DEVICE)[None, None, :].repeat(1, N, 1).clone()
        exo[0, :, V2] = exo[0, :, V2] + d_v2
        exo[0, :, 8] = exo[0, :, 8] * W_mul
        h, Tm, rB, m1, m2 = init_states_evap(
            model0, torch.tensor(row, device=DEVICE)[None, :],
            torch.tensor(obs, device=DEVICE)[None, :])
        out, *_ = model0.integrate(exo, h, Tm, rB, m1, m2, N)
        return out[0, :, 4].cpu().numpy()

    for name, row_idx, state in (("wet", OP_WET, "wet"), ("dry", OP_DRY, "dry")):
        kw = k_w_state[state]
        base = run_step(row_idx, 0.0, 1.0)
        step = run_step(row_idx, 0.05, 1.0 + kw * 0.05)
        d = step - base
        K = float(np.mean(d[-60:]))
        idx = np.where(d <= 0.63 * K)[0] if K < 0 else np.where(d >= 0.63 * K)[0]
        tau63 = int(idx[0]) * DT if len(idx) else None
        summ[f"step_{name}"] = {"K": round(K, 3), "tau63_s": tau63}
        print(f"[step {name}] K={K:.3f} τ63={tau63}s", flush=True)

    B1 = bool(abs(bias) <= 8.0)
    B2 = bool(summ["step_wet"]["tau63_s"] is not None
              and 240 <= summ["step_wet"]["tau63_s"] <= 900)
    B3 = bool(rmse_main <= 10.0)
    B4 = bool(summ["step_dry"]["K"] < 0)
    judge = {"B1": B1, "B2": B2, "B3": B3, "B4": B4,
             "bias_sh1out": round(bias, 2), "rollout_main": round(rmse_main, 2),
             "verdict": "PASS" if (B1 and B2 and B3 and B4) else "FAIL"}
    summ["judge"] = judge
    with open(os.path.join(OUT, "fixb_evap_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2, default=str)
    print("=== FIXB 判定 ===")
    print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    t_axis = np.arange(N) * DT / 60.0
    for name, color in (("wet", "#c55a11"), ("dry", "#8b008b")):
        row_idx = OP_WET if name == "wet" else OP_DRY
        kw = k_w_state[name]
        base = run_step(row_idx, 0.0, 1.0)
        step = run_step(row_idx, 0.05, 1.0 + kw * 0.05)
        d = step - base
        K = float(np.mean(d[-60:]))
        ax.plot(t_axis, d / K, lw=1.4, color=color, label=f"{name} open-loop (norm)")
    ax.axhline(0.63, color="0.5", ls=":", lw=0.8)
    ax.set_title("e0-evap open-loop step")
    ax.set_xlabel("time (min)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar([0, 1], [abs(bias), 21.4], color=["#2e8b57", "crimson"], alpha=0.8)
    ax.axhline(8.0, color="0.3", ls=":", lw=1, label="B1 gate 8°C")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["e0-evap bias", "old e0 bias"])
    ax.set_title("wet sh1_out first-step bias")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"FIXB evaporation — verdict={judge['verdict']} "
                 f"(B1={B1} B2={B2} B3={B3} B4={B4})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig27_evap.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig27_evap.png", flush=True)


if __name__ == "__main__":
    main()
