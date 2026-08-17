#!/usr/bin/env python3
"""23_dry_tau63_probe.py: FIXC 归因 — 干态 τ63=10s 是残差W捷径还是灰盒干态喷水前载物理?

qslow 干态仍瞬时响应 (τ63=10s)。qslow 残差含 W 特征 (no_v12), W 联动阶跃让残差瞬间反应。
qnal (FIX3) 残差无 W (no_act=True)。对比两模型干态耦合阶跃 τ63:
  - qnal τ63 也 ~10s → 物理前载 (灰盒干态喷水增量效应递减), 需结构修复
  - qnal τ63 显著更大 → W 捷径是主因, 修复 = 残差也去 W (qna-lag)
另测: 纯灰盒 e0 (无残差) 干态耦合阶跃 τ63 作物理上限参考。
"""
import importlib.util
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _imp(p, n):
    spec = importlib.util.spec_from_file_location(n, os.path.join(os.getcwd(), p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t02 = _imp("02_train.py", "t02")
r09 = _imp("09_residual.py", "r09")
r15 = _imp("15_fix_learnlag.py", "r15")
r22 = _imp("22_fix_slowdyn.py", "r22")
import numpy as np
import torch

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
P_CRIT = t02.P_CRIT
N = 600
DT = 10.0
V2 = 6
OP_DRY = 40437


def main():
    df = r09.load_e0_df()
    E = df[r09.E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    Ea = E.to_numpy(np.float32)
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)
    pm_all = Ea[:, 2]
    model0 = r09.load_e0(0)

    pm_seg = pm_all[START: START + t02.ROLL_STEPS]
    sub = Ea[START: START + t02.ROLL_STEPS][pm_seg > P_CRIT]
    A = np.stack([sub[:, 5], sub[:, 6], np.ones(len(sub))], 1)
    coef, _, _, _ = np.linalg.lstsq(A, sub[:, 8], rcond=None)
    k_w_dry = float(np.clip(coef[1] / np.mean(sub[:, 8]), 0.5, 4.0))
    row, obs = Ea[OP_DRY], T_all[OP_DRY]
    u0, W0 = float(row[V2]), float(row[8])

    # 基线 (qslow 用 r22, qnal/e0 用 r15 路径统一: 手动 per-step)
    results = {}

    def run_step(fwd, base_cb):
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        z_lag = None
        base = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                base[t], h, Tm, rB, T_sens, z_lag = fwd(row, h, Tm, rB, T_sens, z_lag, u0, W0)
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        z_lag = None
        dT = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                dT[t], h, Tm, rB, T_sens, z_lag = fwd(row, h, Tm, rB, T_sens, z_lag,
                                                      u0 + 0.05, W0 * (1 + k_w_dry * 0.05))
        d = dT - base
        K = float(np.mean(d[-60:]))
        idx = np.where(d <= 0.63 * K)[0] if K < 0 else np.where(d >= 0.63 * K)[0]
        tau63 = int(idx[0]) * DT if len(idx) else None
        return {"K": round(K, 3), "tau63_s": tau63,
                "resp_step1": round(float(d[0]), 3), "resp_step2": round(float(d[1]), 3)}

    # qslow (W in features, lagged residual)
    mod_qslow = r22.QnaLagSlow().to(DEVICE)
    mod_qslow.load_state_dict(torch.load(os.path.join(OUT, "model_res_qslow_seed0.pt"),
                                         map_location=DEVICE, weights_only=True))
    mod_qslow.eval()
    for p in mod_qslow.parameters():
        p.requires_grad_(False)

    def fwd_qslow(row, h, Tm, rB, T_sens, z_lag, v2v, Wv):
        exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
        exo[0, 0, V2] = v2v
        exo[0, 0, 8] = Wv
        out, h, Tm, rB, hm1, hm2, T_sens, z_lag = r22.integrate_slow(
            model0, mod_qslow, exo, h, Tm, rB, 1, T_sens=T_sens, z_lag=z_lag)
        return float(out[0, 0, 4]), h, Tm, rB, T_sens, z_lag

    results["qslow(W_in)"] = run_step(fwd_qslow, None)

    # qnal (no W, lagged τ_sw/τ_sens, no residual lag)
    mod_qnal = r15.QnaLag().to(DEVICE)
    mod_qnal.load_state_dict(torch.load(os.path.join(OUT, "model_res_qnal_seed0.pt"),
                                        map_location=DEVICE, weights_only=True))
    mod_qnal.eval()
    for p in mod_qnal.parameters():
        p.requires_grad_(False)

    def fwd_qnal(row, h, Tm, rB, T_sens, z_lag, v2v, Wv):
        exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
        exo[0, 0, V2] = v2v
        exo[0, 0, 8] = Wv
        out, h, Tm, rB, hm1, hm2, T_sens = r15.integrate_learn(
            model0, mod_qnal, exo, h, Tm, rB, 1, T_sens=T_sens)
        return float(out[0, 0, 4]), h, Tm, rB, T_sens, None

    results["qnal(no_W)"] = run_step(fwd_qnal, None)

    # e0 纯灰盒 (无残差, 无滞后) — 物理前载上限参考: 用 r15.integrate_learn 但残差置零不可行,
    # 直接模拟: 用 qnal 的 integrate 但 mlp 权重置零
    for p in mod_qnal.mlp.parameters():
        p.data.zero_()
    results["e0_greybox_only"] = run_step(fwd_qnal, None)

    with open(os.path.join(OUT, "dry_tau63_probe.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    for k, v in results.items():
        print(f"[{k}] {v}", flush=True)
    print("=== 判定 ===")
    if results["qnal(no_W)"]["tau63_s"] is not None and \
            results["qnal(no_W)"]["tau63_s"] > 3 * DT:
        print("qnal τ63 > 30s → W 捷径是干态瞬时响应的主因 → 修复 = 残差去 W (qna-lag)", flush=True)
    else:
        print("qnal τ63 ≤ 30s → 灰盒干态物理前载是主因 → 需结构修复 (干态喷水动力学)", flush=True)


if __name__ == "__main__":
    main()
