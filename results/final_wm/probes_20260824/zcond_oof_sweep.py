"""Full-val OOF sweep + NLL sigma decomposition (2026-08-29, execution-side).

Audit plan eval_protocol_audit_20260829.md §3: the frozen 256-window protocol
is a random sample; this sweep covers the WHOLE val split (stride 9, ~11.8k
windows) for control + zcond A + zcond B on the SAME windows, and decomposes
the Gaussian NLL into the (err/sigma)^2 term and the log-sigma^2 term --
answering whether NLL gains come from point accuracy or sigma calibration.

No gates are evaluated here; the frozen 256-window protocol stays the
decision metric.  This is a robustness/mechanism evidence layer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.final_wm.analysis import STEAM_FLOW_INDEX
from src.final_wm.data import SPLIT_VAL
from src.final_wm.model import HistoryWindow
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from src.final_wm.water_coal import WaterCoalRecord

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))
from a5_water_coal_probe import BaseRecordView, probe_spec, _error_summary
from zcond_schedule_probe import promote_zcond, z_stats, K_MAX

DEVICE = "cuda"
OUT = P / "zcond_schedule_probe" / "oof_sweep"
OUT.mkdir(parents=True, exist_ok=True)
RECORD = ROOT / "artifacts/final_wm/canonical_sideA_v2.npz"
CONTROL_CKPT = P / "a5_water_coal_probe" / "control" / "checkpoints" / "t1_a5_filtered_control_seed0.pt"
ARM_CKPT = {
    "a": P / "zcond_schedule_probe" / "arm_a" / "checkpoints" / "t1_zcond_k_seed0.pt",
    "b": P / "zcond_schedule_probe" / "arm_b" / "checkpoints" / "t1_zcond_all_seed0.pt",
}
W, H, STRIDE = 96, 18, 9


def build_windows(record, view):
    """Stride-walk the val split: every STRIDE-th valid window start."""
    span = W + H
    starts = []
    for s, e in record.split_runs(SPLIT_VAL):
        if e - s < span:
            continue
        starts.extend(range(s + W, e - H + 1, STRIDE))
    idx = torch.tensor(starts, dtype=torch.long)
    hist_off = torch.arange(-W, 0)
    fut_off = torch.arange(0, H)
    hist_idx = idx[:, None] + hist_off[None, :]
    fut_idx = idx[:, None] + fut_off[None, :]
    obs = view.obs
    act = view.actions
    bnd = view.boundary
    fut_obs = view.obs
    return {
        "idx": idx,
        "hist_obs": obs[hist_idx], "hist_act": act[hist_idx], "hist_bnd": bnd[hist_idx],
        "fut_obs": fut_obs[fut_idx], "fut_act": act[fut_idx], "fut_bnd": bnd[fut_idx],
        "load": bnd[idx, STEAM_FLOW_INDEX],
    }


@torch.no_grad()
def sweep(model, wins, batch=256):
    n = wins["idx"].shape[0]
    mu_all, sig_all, tgt_all, load_all = [], [], [], []
    for i in range(0, n, batch):
        j = slice(i, min(i + batch, n))
        hist = HistoryWindow(wins["hist_obs"][j].to(DEVICE), wins["hist_act"][j].to(DEVICE),
                             wins["hist_bnd"][j].to(DEVICE))
        r = model.forecast(hist, wins["fut_act"][j].to(DEVICE), boundary_mode="oracle",
                           true_future_boundary=wins["fut_bnd"][j].to(DEVICE))
        ch = 4  # final_outlet_temp
        mu_all.append(r.temps_mu[:, :, ch].cpu())
        sig_all.append(r.temps_sigma[:, :, ch].cpu())
        tgt_all.append(wins["fut_obs"][j][:, :, ch])
        load_all.append(wins["load"][j])
    mu = torch.cat(mu_all)
    sig = torch.cat(sig_all)
    tgt = torch.cat(tgt_all)
    load = torch.cat(load_all)
    err = (tgt - mu).abs()
    per_window = err.mean(dim=1)  # per-window H18 MAE, ch4
    edges = torch.quantile(load, torch.tensor([0.2, 0.4, 0.6, 0.8]).to(load.device))
    bidx = torch.bucketize(load, edges)
    bin_means = [float(per_window[bidx == i].mean()) for i in range(5)]
    # NLL decomposition (Gaussian): nll = 0.5*((e/s)^2 + log(2 pi s^2))
    e = (tgt - mu)
    err_term = 0.5 * ((e / sig) ** 2)
    logsig_term = 0.5 * torch.log(2 * np.pi * sig ** 2)
    return {
        "n_windows": int(n),
        "overall_h18_mae": float(per_window.mean()),
        "bins_q1q5": bin_means,
        "spread_ratio": float(max(bin_means) / min(bin_means)) if all(bin_means) else None,
        "nll_err_term": float(err_term.mean()),
        "nll_logsigma_term": float(logsig_term.mean()),
        "nll_total": float((err_term + logsig_term).mean()),
        "mean_abs_err_c": float(err.mean()),
        "mean_sigma_c": float(sig.mean()),
        "sigma_err_ratio": float(sig.mean() / (err.mean() + 1e-12)),
    }


def main() -> None:
    record = WaterCoalRecord(RECORD)
    view = BaseRecordView(record)
    props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz",
                                 device=DEVICE)
    wins = build_windows(record, view)
    print(f"[OOF] windows={wins['idx'].shape[0]} (val stride {STRIDE})")

    control = build_world_model(probe_spec("a5_filtered_control"), props).to(DEVICE)
    control.load_state_dict(torch.load(CONTROL_CKPT, map_location=DEVICE,
                                       weights_only=False)["state_dict"])
    control.eval()
    center, scale = z_stats(control, view)

    results = {}
    results["control"] = sweep(control, wins)
    for name, groups in (("a", ("k",)), ("b", ("k", "tau", "ua"))):
        model = promote_zcond(build_world_model(probe_spec(f"zcond_{'k' if name=='a' else 'all'}"), props),
                              center, scale, groups).to(DEVICE)
        model.load_state_dict(torch.load(ARM_CKPT[name], map_location=DEVICE,
                                         weights_only=False)["state_dict"])
        model.eval()
        results[name] = sweep(model, wins)

    report = {"stride": STRIDE, "windows": int(wins["idx"].shape[0]),
              "record": str(RECORD), "models": results,
              "note": "robustness layer; frozen 256-window protocol remains the decision metric"}
    (OUT / "oof_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    for name, r in results.items():
        print(f"[OOF {name}] H18={r['overall_h18_mae']:.4f} spread={r['spread_ratio']:.3f} "
              f"nll={r['nll_total']:.3f} (err {r['nll_err_term']:.3f} / logσ {r['nll_logsigma_term']:.3f}) "
              f"σ/err={r['sigma_err_ratio']:.2f}")
    print(f"written {OUT / 'oof_report.json'}")


if __name__ == "__main__":
    main()
