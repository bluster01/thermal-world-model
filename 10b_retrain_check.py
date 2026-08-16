#!/usr/bin/env python3
"""10b_retrain_check.py: ra/rc seed0 新代码重训 vs 旧ckpt eval — 代码路径等价性验证

背景: Step① 中 ra/rc 旧ckpt种子(旧代码训练)与重训种子(新代码训练)出现差异
(rc s0=6.10 vs s1=7.36/s2=7.41; ra s0=5.81/s1=4.78 vs s2=6.23)。
rb(无锚)重训种子与旧ckpt一致(2.09/2.13/2.00), 疑点集中在带锚变体。
本检查: 用新代码重训 ra/rc seed0, 对比旧ckpt eval 数字。
  一致(<0.1) → 代码路径等价, 差异=锚定变体种子方差
  不一致 → 代码改动引入回归, 停止 Step② 排查
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
import numpy as np
import torch

df = r09.load_e0_df()
model0 = r09.load_e0(0)
mu_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].mean().to_numpy(np.float32)
sd_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].std().replace(0, 1.0).to_numpy(np.float32)
pm_roll = df["分离器出口压力"].to_numpy(np.float32)[r09.START: r09.START + t02.ROLL_STEPS]

OLD = {"ra": {"rollout": 5.19, "dry": 5.81, "wet": 4.59},
       "rc": {"rollout": 5.33, "dry": 6.10, "wet": 4.57}}
res = {}
for v, mode, anchor, out in [("ra", "q", True, 3), ("rc", "dk", True, 3)]:
    m, va, ep = r09.train_res(df, 0, v + "_chk", mode, anchor, False, out=out)
    m.eval()
    with torch.no_grad():
        r, preds, truths = r09.rollout_res(model0, m, df, r09.START, t02.ROLL_STEPS, mode, anchor)
        sr = r09.strat_rollout(preds, truths, pm_roll, mu_o, sd_o)
    roll = round(r["rmse_main"], 4)
    d = sr["dry"]["rmse_main"]
    w = sr["wet"]["rmse_main"]
    dd = round(abs(roll - OLD[v]["rollout"]), 4)
    ok = bool(dd < 0.1)
    res[v] = {"rollout_new": roll, "rollout_old": OLD[v]["rollout"],
              "rollout_diff": dd, "dry": d, "wet": w,
              "val_mse": round(va, 4), "ep": ep, "equiv": ok}
    print(f"[chk {v}] new={roll} old={OLD[v]['rollout']} diff={dd} "
          f"dry={d} wet={w} val={va:.2f} ep={ep} equiv={ok}", flush=True)
res["verdict"] = "EQUIV" if all(res[v]["equiv"] for v in ("ra", "rc")) else "REGRESSION"
with open(os.path.join(t02.OUT, "retrain_check.json"), "w") as f:
    json.dump(res, f, indent=2)
print(f"[chk] verdict={res['verdict']} saved out/retrain_check.json", flush=True)
