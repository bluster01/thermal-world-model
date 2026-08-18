#!/usr/bin/env python3
"""10e_align_check.py: rollout 真值对齐审计 — 灰盒 preds[t]=T̂(start+t+1) vs truths[t]=T(start+t)?

假设: e0_rollout/rollout_res 初始化用 T[start] 且第一步就 integrate(exo[start])
→ preds[0] 是 T̂(start+10s), 但 truths[0]=T_all[start] → 1 步错位。
v2 baseline_rollout: window=历史[start-60,start-1] → pred 对齐 T(start+t) → 应无错位。
判别: 灰盒把 truth 右移 1 步 rmse 应下降 (错位证实); v2 右移应上升 (本就对齐)。
"""
import importlib.util
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _imp(p, n):
    spec = importlib.util.spec_from_file_location(n, os.path.join(os.getcwd(), p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t02 = _imp("02_train.py", "t02")
import numpy as np

OUT = t02.OUT
OUTS = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "main"]
N = t02.ROLL_STEPS


def rmse5(p, t):
    return np.sqrt(np.mean((p - t) ** 2, axis=0))


def test(tag, s):
    f = os.path.join(OUT, f"rollout_{tag}_seed{s}.npz" if s is not None else f"rollout_{tag}.npz")
    d = np.load(f)
    preds, truths = d["preds"][:N], d["truths"][:N]
    r_now = rmse5(preds, truths[0:N])            # 现行协议 (可能错位)
    r_shift = rmse5(preds[0:N - 1], truths[1:N])  # 右移 1 步 (对齐假设)
    d_std = truths[1:N].std(0) - truths[0:N - 1].std(0)  # 10s 差分漂移参考
    drift10 = (truths[1:N] - truths[0:N - 1]).std(0)
    print(f"\n{tag} s{s}: 现行 vs 右移1步")
    print(f"  {'out':8s} | {'现行RMSE':>9s} | {'右移RMSE':>9s} | {'Δ':>7s} | {'10s漂移std':>10s}")
    for j, o in enumerate(OUTS):
        print(f"  {o:8s} | {r_now[j]:9.2f} | {r_shift[j]:9.2f} | {r_shift[j]-r_now[j]:+7.2f} | {drift10[j]:10.2f}")


print("=" * 70)
print("灰盒家族 (假设错位 → 右移应改善):")
print("=" * 70)
test("res_rb", 0)
test("res_qh", 0)
test("res_e0", None)

print("\n" + "=" * 70)
print("黑盒 v2 (假设已对齐 → 右移应恶化):")
print("=" * 70)
test("v2", 0)
