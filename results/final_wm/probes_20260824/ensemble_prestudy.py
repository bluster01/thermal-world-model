"""Zero-cost ensemble pre-study on cached predictions (2026-08-25).

Estimates the achievable gain of stacking armA (physics) + iTransformer (bb)
before training any meta-model:
1. global blend sweep  w*phys + (1-w)*bb
2. oracle upper bound   per-window min(phys, bb)
3. feature-based switch: |phys predicted displacement| > tau  -> phys else bb
   (feature available at inference time; sweep tau)
4. per-window blend weight predicted by ridge on inference-time features,
   evaluated with leave-one-day-out CV
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/final_wm/probes_20260824/retrain_probe"

z0 = np.load(ROOT / "results/final_wm/probes_20260824/plots_mainsteam/predictions_cache.npz")
za = np.load(OUT / "armA_budget/preds_armA_budget.npz")
phys, actual, bb = za["pred"], z0["actual"], z0["bb_pred"]
days = z0["days"]
N, H = actual.shape

maeP = np.abs(phys - actual).mean(axis=1)
maeB = np.abs(bb - actual).mean(axis=1)
print(f"baseline: armA {maeP.mean():.3f} | bb {maeB.mean():.3f}")

# ---- 1. global blend ----
print("\n[1] global blend w*phys + (1-w)*bb:")
best = (1.0, maeP.mean())
for w in np.arange(0, 1.01, 0.1):
    m = np.abs(w * phys + (1 - w) * bb - actual).mean(axis=1).mean()
    print(f"  w={w:.1f}: {m:.3f}")
    if m < best[1]:
        best = (w, m)
print(f"  best global w={best[0]:.1f} -> {best[1]:.3f}")

# ---- 2. oracle ----
oracle = np.minimum(maeP, maeB).mean()
print(f"\n[2] oracle per-window min: {oracle:.3f}  (ceiling)")
print(f"    theoretical max gain vs bb: {maeB.mean() - oracle:.3f}")

# ---- 3. feature switch ----
disp_phys = np.abs(phys[:, -1] - phys[:, 0])   # predicted displacement
print("\n[3] switch on |phys predicted displacement|:")
for tau in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0):
    use_phys = disp_phys > tau
    comb = np.where(use_phys[:, None], phys, bb)
    m = np.abs(comb - actual).mean(axis=1).mean()
    print(f"  tau={tau}: n_phys={use_phys.sum():3d} | MAE {m:.3f}")

# ---- 4. ridge combiner, leave-one-day-out CV ----
feats = np.stack([
    disp_phys,
    np.abs(bb[:, -1] - bb[:, 0]),
    np.abs(phys - bb).mean(axis=1),           # disagreement
    maeP,  # not usable at inference -- EXCLUDE in the honest variant
], axis=1)
feats_honest = feats[:, :3]
print("\n[4] ridge w(phys) = f(features), leave-one-day-out CV:")

def ridge_fit(X, y, lam=1.0):
    Xc = X - X.mean(0); yc = y - y.mean()
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    beta = np.linalg.solve(A, Xc.T @ yc)
    return beta, y.mean(), X.mean(0)

def loo_day_eval(X, y, phys, bb, actual):
    udays = np.unique(days)
    preds = np.zeros_like(actual)
    for d in udays:
        tr = days != d; te = days == d
        beta, mu, xm = ridge_fit(X[tr], y[tr])
        w = np.clip((X[te] - xm) @ beta + mu, 0, 1)
        preds[te] = w[:, None] * phys[te] + (1 - w[:, None]) * bb[te]
    return np.abs(preds - actual).mean(axis=1).mean()

m_honest = loo_day_eval(feats_honest, maeP, phys, bb, actual)
m_all = loo_day_eval(feats, maeP, phys, bb, actual)
print(f"  honest features (disp_phys, disp_bb, disagreement): {m_honest:.3f}")
print(f"  +leaky feature (phys MAE, not inference-legal):     {m_all:.3f}")

# distribution of per-window best-model
print(f"\nwindows where phys strictly better than bb: {(maeP < maeB).sum()}")
print(f"median maeP {np.median(maeP):.3f} maeB {np.median(maeB):.3f}")
