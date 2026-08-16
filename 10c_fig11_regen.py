#!/usr/bin/env python3
"""10c_fig11_regen.py: 重生成 fig11 (修复CJK标题tofu) + 打印 qspl λ 统计

复用 Step② 已训练 ckpt (reuse=True), 重跑 rollout+windowed 评估后重画图。
同时输出 qspl 学到的分配系数 λ 的每段统计 (均值/中位数/占比)。
"""
import argparse
import importlib.util
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _imp(p, n):
    spec = importlib.util.spec_from_file_location(n, os.path.join(os.getcwd(), p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t02 = _imp("02_train.py", "t02")
r10 = _imp("10_refine.py", "r10")
import numpy as np

dummy = argparse.Namespace(step="inject", smoke_short=False)
df, model0, mu_o, sd_o, pm_roll, n_roll = r10.load_ctx(dummy)

recs = {}
for v in ("rb", "qtm", "qh", "qcon", "qspl"):
    recs[v] = r10.run_variant(df, model0, v, [0], False, n_roll, mu_o, sd_o, pm_roll,
                              reuse=True)

lam_hist = r10.collect_lam(df, model0, r10.load_res_ckpt("qspl"))
print("=== qspl λ stats (test windows) ===")
for j, hh in enumerate(lam_hist):
    print(f"seg{j}: mean={hh.mean():.3f} std={hh.std():.3f} p50={np.median(hh):.3f} "
          f"P(λ>0.5)={np.mean(hh > 0.5):.3f} (λ=1→全Tm)", flush=True)

judge = r10.judge_inject(recs)
print("judge:", judge, flush=True)
r10.fig11_inject(recs, judge, lam_hist)
print("[fig11] regenerated", flush=True)
