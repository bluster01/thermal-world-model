#!/usr/bin/env python3
"""37_direct_wm_object_probe.py — Direct WM 对象阶跃响应探针（纯推理，复用 36 的 ckpt）

目的: 在 Q32-T 对象面板同口径干预 (±0.02 阀位分位持久阶跃) 下量化 Direct WM 的
响应能力 (符号/增益/时标), 把纯数据点的响应弱点如实写进折中曲线。

口径差异 (诚实记录):
  - Q32-T 对象面板干预 = 阀位 ±0.02 + 喷水 W 乘子 (耦合阶跃, 物理模型有 W 输入)
  - Direct WM 只有未来阀位序列是动作输入 (无 W 通道) → 本探针 = 阀位 ±0.02 持久阶跃
  - 两者增益不可直接等值对比, 只比符号/方向占比/时标形态

协议:
  - 6 个 ckpt (F0/F1 × s42/s0/s7), 折匹配评测窗
  - 每折 200 锚点 (seed 42), 干预: v1 ±0.02 / v2 ±0.02 / 双阀 ±0.02, 全程 18 步持久
  - 输出: 逐步 ΔT 曲线 (中位数), 方向占比, 稳态增益 (末3步均值), τ63 (18步内可估则报)
"""
import importlib.util, json, os
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("dm36", "36_direct_wm.py")
dm36 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm36)

DEVICE = dm36.DEVICE
FOLDS = dm36.FOLDS
OUT_DIR = "out/direct_wm_object_probe"
INTERVENTIONS = {"v1": 0, "v2": 1, "both": None}   # None = 双阀同时
N_ANCHOR = 200


def load_model(fold, seed, cols, target_idx):
    m = dm36.DirectAligned(len(cols), target_idx).to(DEVICE)
    ck = torch.load(f"{dm36.OUT_DIR}/direct_wm_{fold}_s{seed}.pt",
                    map_location=DEVICE, weights_only=True)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


@torch.no_grad()
def step_response(model, raw, seg, rng, valve_idx, valve_sel, delta):
    """持久 ±0.02 阶跃 → 每锚点 ΔT 曲线 [H]"""
    a, b = seg
    idxs = rng.choice(np.arange(a, b - dm36.W - dm36.H), N_ANCHOR, replace=False)
    dT = np.zeros((N_ANCHOR, dm36.H))
    for j, i in enumerate(idxs):
        x_hist = torch.FloatTensor(raw[i:i + dm36.W]).unsqueeze(0).to(DEVICE)
        a_fut = torch.FloatTensor(raw[i + dm36.W:i + dm36.W + dm36.H, valve_idx]).unsqueeze(0).to(DEVICE)
        mu_b, _ = model(x_hist, a_fut)
        base = mu_b[0].cpu().numpy()
        a_p = a_fut.clone()
        if valve_sel is None:
            a_p[0, :, 0] = a_p[0, :, 0] + delta
            a_p[0, :, 1] = a_p[0, :, 1] + delta
        else:
            a_p[0, :, valve_sel] = a_p[0, :, valve_sel] + delta
        mu_p, _ = model(x_hist, a_p)
        dT[j] = mu_p[0].cpu().numpy() - base
    return dT


def tau63(curve):
    """曲线 (18步) 的 τ63 步数; 未达稳态返回 None"""
    ss = curve[-1]
    if abs(ss) < 1e-4:
        return None
    target = 0.63 * ss
    idx = np.where(np.abs(curve) >= np.abs(target))[0]
    return int(idx[0]) if len(idx) else None


def main():
    raw, cols = dm36.load_dev()
    target_idx = cols.index("末级过热器出口汽温")
    valve_idx = [cols.index("一级减温调节门阀位"), cols.index("二级减温调节门阀位")]
    print(f"[37] dev rows={len(raw)} | anchors={N_ANCHOR} | target={target_idx} valves={valve_idx} | device={DEVICE}")
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for fold in ["F0", "F1"]:
        ev = FOLDS[fold]["eval"]
        for seed in [42, 0, 7]:
            model = load_model(fold, seed, cols, target_idx)
            rng = np.random.default_rng(42)   # 锚点固定跨模型
            for name, sel in INTERVENTIONS.items():
                for delta in [-0.02, 0.02]:
                    dT = step_response(model, raw, ev, rng, valve_idx, sel, delta)
                    med = np.median(dT, axis=0)
                    steady = float(np.mean(med[-3:]))
                    sign_ok = float(np.mean(np.sign(dT[:, -1]) == np.sign(delta) * -1))
                    t63 = tau63(med)
                    key = f"{fold}_s{seed}_{name}_{delta:+.3f}"
                    results[key] = {"median_curve": med.tolist(), "steady_c": steady,
                                    "direction_correct_fraction": sign_ok,
                                    "tau63_steps": t63, "n": N_ANCHOR}
                    print(f"  {key}: steady={steady:+.4f} sign={sign_ok:.2f} tau63={t63}")
    summary = {"experiment": "direct_wm_object_probe",
               "intervention": "valve +/-0.02 persistent 18 steps (no W channel)",
               "ckpts": ["direct_wm_%s_s%d.pt" % (f, s) for f in ["F0", "F1"] for s in [42, 0, 7]],
               "anchor_seed": 42, "n_anchor": N_ANCHOR, "results": results}
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
