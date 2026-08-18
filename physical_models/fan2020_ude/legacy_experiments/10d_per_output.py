#!/usr/bin/env python3
"""10d_per_output.py: 各输出(RMSE/MAE/bias)明细表 — 从已落盘 rollout npz 直接计算, 不重训不重评"""
import importlib.util
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _imp(p, n):
    spec = importlib.util.spec_from_file_location(n, os.path.join(os.getcwd(), p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t02 = _imp("02_train.py", "t02")
r09 = _imp("09_residual.py", "r09")
import numpy as np
import pandas as pd

OUT = t02.OUT
START = r09.START
P_CRIT = t02.P_CRIT
OUTS = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "main"]

# pm 序列 (rollout 1800 步)
df = pd.read_csv(t02.CSV, usecols=["分离器出口压力"], dtype=np.float32) \
    .iloc[t02.WIN_START: t02.WIN_START + t02.WIN].ffill().bfill().reset_index(drop=True)
pm_roll = df["分离器出口压力"].to_numpy(np.float32)[START: START + t02.ROLL_STEPS]
wet_mask = pm_roll <= P_CRIT

SPECS = {
    "e0-post": ("res_e0", [None]),
    "rb":      ("res_rb", [0, 1, 2]),
    "qh":      ("res_qh", [0, 1, 2]),
    "qspl":    ("res_qspl", [0, 1, 2]),
    "qtm":     ("res_qtm", [0]),
    "qcon":    ("res_qcon", [0]),
    "v2":      ("v2", [0, 1, 2]),
    "v2o":     ("v2o", [0, 1, 2]),
    "v0":      ("v0", [0, 1, 2]),
}


def _load(tag, s):
    f = os.path.join(OUT, f"rollout_{tag}_seed{s}.npz" if s is not None else f"rollout_{tag}.npz")
    return np.load(f)


def per_out(p, t):
    rmse = np.sqrt(np.mean((p - t) ** 2, axis=0))
    mae = np.mean(np.abs(p - t), axis=0)
    bias = np.mean(p - t, axis=0)
    return rmse, mae, bias


print("=" * 100)
print(f"rollout 1800步 ({t02.ROLL_STEPS}×10s=5h) — wet {wet_mask.mean()*100:.0f}% / dry {(~wet_mask).mean()*100:.0f}%")
print("=" * 100)
print(f"{'variant':8s} | " + " | ".join(f"{o:>18s}" for o in OUTS))
print("-" * 100)
rows = {}
for name, (tag, seeds) in SPECS.items():
    rmses, maes, biases = [], [], []
    for s in seeds:
        d = _load(tag, s)
        preds, truths = d["preds"], d["truths"]
        if preds.shape[0] > t02.ROLL_STEPS:
            preds, truths = preds[:t02.ROLL_STEPS], truths[:t02.ROLL_STEPS]
        r, m, b = per_out(preds, truths)
        rmses.append(r)
        maes.append(m)
        biases.append(b)
    rm = np.mean(rmses, axis=0)
    ma = np.mean(maes, axis=0)
    bi = np.mean(biases, axis=0)
    rs = np.std(rmses, axis=0)
    rows[name] = (rm, ma, bi, rs)
    cell = " | ".join(f"{rm[j]:6.2f}±{rs[j]:4.2f}" for j in range(5))
    print(f"{name:8s} | " + " | ".join(f"{rm[j]:>18.2f}" for j in range(5)))
print("-" * 100)
print(f"{'variant':8s} | " + " | ".join(f"{o:>18s}" for o in OUTS))
for name in SPECS:
    ma = rows[name][1]
    print(f"{name:8s} | " + " | ".join(f"{ma[j]:>18.2f}" for j in range(5)))
print("-" * 100)
print(f"{'variant':8s} | " + " | ".join(f"{o:>18s}" for o in OUTS))
for name in SPECS:
    bi = rows[name][2]
    print(f"{name:8s} | " + " | ".join(f"{bi[j]:>+18.2f}" for j in range(5)))

print()
print("=" * 100)
print("湿/干分层 RMSE (主要变体)")
print("=" * 100)
for name in ("e0-post", "v2", "rb", "qh", "qspl", "qcon"):
    tag, seeds = SPECS[name]
    d = _load(tag, seeds[0])
    preds, truths = d["preds"][:t02.ROLL_STEPS], d["truths"][:t02.ROLL_STEPS]
    print(f"\n{name} (seed0):")
    print(f"  {'output':8s} | {'wet RMSE':>10s} | {'dry RMSE':>10s} | {'wet MAE':>10s} | {'dry MAE':>10s} | {'wet bias':>10s} | {'dry bias':>10s}")
    for j, o in enumerate(OUTS):
        rw = np.sqrt(np.mean((preds[wet_mask, j] - truths[wet_mask, j]) ** 2))
        rd = np.sqrt(np.mean((preds[~wet_mask, j] - truths[~wet_mask, j]) ** 2))
        mw = np.mean(np.abs(preds[wet_mask, j] - truths[wet_mask, j]))
        md = np.mean(np.abs(preds[~wet_mask, j] - truths[~wet_mask, j]))
        bw = np.mean(preds[wet_mask, j] - truths[wet_mask, j])
        bd = np.mean(preds[~wet_mask, j] - truths[~wet_mask, j])
        print(f"  {o:8s} | {rw:10.2f} | {rd:10.2f} | {mw:10.2f} | {md:10.2f} | {bw:+10.2f} | {bd:+10.2f}")
