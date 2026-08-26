"""Observer encoder swap probe: GRU -> per-channel TCN / iTransformer tokens.

2026-08-25, execution-side exploratory. NO src changes (config_fingerprint
safe): variants subclass ProbabilisticObserver in this file and override
only encode(); heads/pressure-features/anchor structure untouched.
Training loop replicates train_arm verbatim (same functions imported).

Arms (seed0, epochs=120/patience=20, oracle, conservative_norew):
  sanity  : standard GRU observer, 15 epochs -> val curve must match
            armA_budget seed0 ledger within noise (loop-fidelity check)
  enc_tcn : per-channel causal-ish TCN encoder
  enc_itx : iTransformer-style variables-as-tokens encoder
Eval: canonical probe set (sideA val, 256 windows seed 50k, H18, ch4).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.analysis import WindowErrors, binning_stats, STEAM_FLOW_INDEX
from src.final_wm.contracts import (ACTION_ELEMENTS, BOUNDARY_ELEMENTS,
                                    OBSERVATION_ELEMENTS)
from src.final_wm.data import (SPLIT_TRAIN, SPLIT_VAL, CanonicalRecord,
                               sample_windows)
from src.final_wm.evaluation import evaluate_windows
from src.final_wm.model import FinalWorldModel
from src.final_wm.observer import ProbabilisticObserver
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from src.final_wm.contracts import (WorldModelConfig, TransitionConfig,
                                    ClosureConfig, ObserverConfig,
                                    BoundaryModelConfig)
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
OUT = ROOT / "results/final_wm/probes_20260824/encoder_probe"
OUT.mkdir(parents=True, exist_ok=True)
CH = OBSERVATION_ELEMENTS.index("final_outlet_temp")
N_WIN = 256
EVAL_SEED = 50_000
torch.backends.cuda.matmul.allow_tf32 = True

record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)


def make_config(spec):
    norew = spec.closure_mode.endswith("_norew")
    closure_mode = spec.closure_mode.removesuffix("_norew")
    return WorldModelConfig(
        transition=TransitionConfig(latent_dim=spec.latent_dim, rewet_ablate=norew),
        closure=ClosureConfig(injection_mode=closure_mode),
        observer=ObserverConfig(history_steps=spec.history_steps, latent_dim=spec.latent_dim),
        boundary=BoundaryModelConfig(history_steps=spec.history_steps),
        boundary_mode=spec.boundary_mode,
        initial_state_mode=spec.initial_state_mode,
    )


# ---------------------------------------------------------------------------
# Encoder variants
# ---------------------------------------------------------------------------

class TCNEncoderObserver(ProbabilisticObserver):
    """Per-channel temporal conv stack; heads unchanged."""

    def __init__(self, config, layout, conv_hidden=64):
        super().__init__(config, layout)
        self.n_ch = len(OBSERVATION_ELEMENTS) + len(ACTION_ELEMENTS) + len(BOUNDARY_ELEMENTS)
        d = conv_hidden
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, d, kernel_size=7, padding=3), nn.GELU(),
                nn.Conv1d(d, d, kernel_size=5, padding=2), nn.GELU(),
            ) for _ in range(self.n_ch)
        ])
        self.proj = nn.Linear(self.n_ch * d, config.d_hidden)

    def encode(self, history_obs, history_actions, history_boundary):
        self._check_history(history_obs, history_actions, history_boundary)
        obs_n = (history_obs - self.obs_loc) / self.obs_scale
        act_n = (history_actions - self.action_loc) / self.action_scale
        bnd_n = (history_boundary - self.boundary_loc) / self.boundary_scale
        x = torch.cat([obs_n, act_n, bnd_n], dim=-1)          # (B, 96, 14)
        outs = [c(x[..., i:i + 1].transpose(1, 2)).mean(dim=2)   # (B, d)
                for i, c in enumerate(self.convs)]
        return self.proj(torch.cat(outs, dim=-1))


class ITransformerEncoderObserver(ProbabilisticObserver):
    """Variables-as-tokens: 14 channel tokens, cross-variable attention."""

    def __init__(self, config, layout, d=64, layers=2, heads=4):
        super().__init__(config, layout)
        self.n_ch = len(OBSERVATION_ELEMENTS) + len(ACTION_ELEMENTS) + len(BOUNDARY_ELEMENTS)
        self.tok = nn.Linear(96, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True,
                                         dropout=0.1, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.proj = nn.Linear(d, config.d_hidden)

    def encode(self, history_obs, history_actions, history_boundary):
        self._check_history(history_obs, history_actions, history_boundary)
        obs_n = (history_obs - self.obs_loc) / self.obs_scale
        act_n = (history_actions - self.action_loc) / self.action_scale
        bnd_n = (history_boundary - self.boundary_loc) / self.boundary_scale
        x = torch.cat([obs_n, act_n, bnd_n], dim=-1)          # (B, 96, 14)
        tok = self.tok(x.transpose(1, 2))                     # (B, 14, d)
        z = self.enc(tok)                                     # (B, 14, d)
        return self.proj(z.mean(dim=1))


VARIANTS = {"gru": None, "tcn": TCNEncoderObserver, "itx": ITransformerEncoderObserver}


def build_with_variant(spec, variant: str) -> FinalWorldModel:
    cfg = make_config(spec)
    m = FinalWorldModel(cfg, props)
    if variant != "gru":
        m.observer = VARIANTS[variant](cfg.observer, m.layout)
    return m


# ---------------------------------------------------------------------------
# Training loop (verbatim replica of train_arm essentials)
# ---------------------------------------------------------------------------

def train_variant(spec, variant: str, out_dir: Path, max_epochs: int | None = None,
                  anchor_path: Path | None = None):
    torch.manual_seed(spec.seed)
    model = build_with_variant(spec, variant).to(DEVICE)
    if anchor_path is not None:
        payload = torch.load(anchor_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=False)
        print(f"  [{variant}] anchored from {anchor_path.name} (strict=False)", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=spec.lr)
    gen = torch.Generator().manual_seed(spec.seed)
    epochs = max_epochs if max_epochs is not None else spec.epochs
    best_val, best_epoch, since_best = float("inf"), -1, 0
    val_history = []
    ledger_path = out_dir / "ledger.jsonl"
    t0 = time.time()
    with ledger_path.open("a", encoding="utf-8") as ledger:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for _ in range(spec.batches_per_epoch):
                batch = sample_windows(record, SPLIT_TRAIN, spec.batch_size,
                                       spec.history_steps, spec.horizon, gen)
                history = batch.history.__class__(
                    obs=batch.history.obs.to(DEVICE),
                    actions=batch.history.actions.to(DEVICE),
                    boundary=batch.history.boundary.to(DEVICE),
                )
                future_actions = batch.future_actions.to(DEVICE)
                future_obs = batch.future_obs.to(DEVICE)
                future_boundary = batch.future_boundary.to(DEVICE)
                result = model.forecast(
                    history, future_actions, boundary_mode=spec.boundary_mode,
                    true_future_boundary=future_boundary if spec.boundary_mode == "oracle" else None,
                )
                loss = model.observation_nll(result.temps_mu, result.temps_sigma, future_obs)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()
                train_loss += float(loss)
                torch.cuda.synchronize()
            train_loss /= spec.batches_per_epoch
            val = evaluate_windows(
                model, record, 1, n_windows=spec.eval_windows, batch_size=spec.eval_batch,
                history_steps=spec.history_steps, horizon=spec.horizon,
                boundary_mode=spec.boundary_mode, seed=10_000 + epoch, device=DEVICE,
            )
            val_nll = float(val.nll.mean())
            val_history.append(val_nll)
            entry = {"variant": variant, "epoch": epoch, "train_loss": train_loss,
                     "val_nll": val_nll, "wall_seconds": time.time() - t0}
            ledger.write(json.dumps(entry) + "\n")
            ledger.flush()
            print(f"  [{variant}] ep{epoch} train={train_loss:.3f} val={val_nll:.3f}", flush=True)
            if val_nll < best_val - 1e-4:
                best_val, best_epoch, since_best = val_nll, epoch, 0
                torch.save({"state_dict": model.state_dict()}, out_dir / "best.pt")
            else:
                since_best += 1
                if since_best >= spec.patience and max_epochs is None:
                    break
    return model, {"best_val_nll": best_val, "best_epoch": best_epoch,
                   "epochs_run": epoch + 1, "val_tail": val_history[-5:],
                   "wall_seconds": time.time() - t0}


def eval_probe_set(model, tag):
    model.eval()
    gen = torch.Generator().manual_seed(EVAL_SEED)
    errs, loads, days, preds, acts = [], [], [], [], []
    done = 0
    with torch.no_grad():
        while done < N_WIN:
            bsz = min(32, N_WIN - done)
            b = sample_windows(record, SPLIT_VAL, bsz, 96, 18, gen)
            hist = b.history.__class__(obs=b.history.obs.to(DEVICE),
                                       actions=b.history.actions.to(DEVICE),
                                       boundary=b.history.boundary.to(DEVICE))
            r = model.forecast(hist, b.future_actions.to(DEVICE), boundary_mode="oracle",
                               true_future_boundary=b.future_boundary.to(DEVICE))
            preds.append(r.temps_mu[:, :, CH].cpu())
            acts.append(b.future_obs[:, :, CH].cpu())
            errs.append((b.future_obs.to(DEVICE) - r.temps_mu).abs().cpu())
            loads.append(b.future_boundary[:, 0, STEAM_FLOW_INDEX])
            days.append(b.day_ids)
            done += bsz
    we = WindowErrors(abs_err=torch.cat(errs), load=torch.cat(loads),
                      day_ids=torch.cat(days))
    bins = binning_stats(we)
    ch4 = bins["H18"]["final_outlet_temp"]
    np.savez_compressed(OUT / f"preds_{tag}.npz",
                        pred=torch.cat(preds).numpy(), actual=torch.cat(acts).numpy(),
                        load=we.load.numpy(), days=we.day_ids.numpy())
    return {"overall_h18_mae": float(np.mean(ch4["bin_means"])),
            "bins_q1q5": ch4["bin_means"]}


if __name__ == "__main__":
    spec = ms._base("t1", "closure_cons_norew", 0, boundary_mode="oracle",
                    initial_state_mode="hybrid", closure_mode="conservative_norew",
                    epochs=120, patience=20, batch_size=32, batches_per_epoch=200)
    # ---- sanity arm: 15 epochs, GRU, compare vs armA seed0 ledger ----
    san_dir = OUT / "sanity_gru15"
    san_dir.mkdir(parents=True, exist_ok=True)
    _, san = train_variant(spec, "gru", san_dir, max_epochs=15)
    ref = [json.loads(l) for l in open(
        ROOT / "results/final_wm/probes_20260824/retrain_probe/armA_budget/ledger.jsonl")]
    ref_vals = [x["val_nll"] for x in ref if "val_nll" in x][:15]
    mine = [json.loads(l)["val_nll"] for l in open(san_dir / "ledger.jsonl")]
    diffs = [abs(a - b) for a, b in zip(mine, ref_vals)]
    print(f"[sanity] max |delta val_nll| vs armA seed0 = {max(diffs):.5f} "
          f"(mean {np.mean(diffs):.5f})", flush=True)
    (OUT / "sanity_report.json").write_text(json.dumps(
        {"mine": mine, "ref": ref_vals, "max_abs_diff": float(max(diffs))}, indent=2))

    report = {}
    for variant in ("tcn", "itx"):
        vdir = OUT / f"enc_{variant}"
        vdir.mkdir(parents=True, exist_ok=True)
        print(f"[enc_{variant}] training", flush=True)
        model, tr = train_variant(spec, variant, vdir)
        model.load_state_dict(torch.load(vdir / "best.pt", map_location=DEVICE,
                                         weights_only=False)["state_dict"])
        ev = eval_probe_set(model, f"enc_{variant}")
        print(f"[enc_{variant}] H18 ch4 overall={ev['overall_h18_mae']:.3f} "
              f"bins={[round(x, 3) for x in ev['bins_q1q5']]} | "
              f"best_val={tr['best_val_nll']:.3f}@{tr['best_epoch']}", flush=True)
        report[f"enc_{variant}"] = {"train": tr, "eval": ev}
        (OUT / "report.json").write_text(json.dumps(report, indent=2))
        del model
        torch.cuda.empty_cache()
    print(json.dumps(report, indent=2))
