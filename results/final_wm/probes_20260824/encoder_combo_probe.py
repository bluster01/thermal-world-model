"""Combo arm: iTransformer observer encoder + s1-constant anchor (seed0).

2026-08-25. Stacks the two positive results: enc_itx (0.666 vs GRU 0.723)
and armC anchor (0.478 vs unanchored 0.723). Reuses the bit-verified
replicated training loop from encoder_swap_probe.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

sp = importlib.util.spec_from_file_location(
    "esp", ROOT / "results/final_wm/probes_20260824/encoder_swap_probe.py")
esp = importlib.util.module_from_spec(sp)
sp.loader.exec_module(esp)

from experiments.final_wm import matrix_spec as ms

OUT = ROOT / "results/final_wm/probes_20260824/encoder_probe"
ANCHOR = (ROOT / "results/final_wm/probes_20260824/retrain_probe/anchor_assets"
          / "anchor_init_s1constants_seed0.pt")

spec = ms._base("t1", "closure_cons_norew", 0, boundary_mode="oracle",
                initial_state_mode="hybrid", closure_mode="conservative_norew",
                epochs=120, patience=20, batch_size=32, batches_per_epoch=200)

vdir = OUT / "enc_itx_anchor"
vdir.mkdir(parents=True, exist_ok=True)
print("[enc_itx_anchor] training", flush=True)
model, tr = esp.train_variant(spec, "itx", vdir, anchor_path=ANCHOR)
model.load_state_dict(torch.load(vdir / "best.pt", map_location=esp.DEVICE,
                                 weights_only=False)["state_dict"])
ev = esp.eval_probe_set(model, "enc_itx_anchor")
print(f"[enc_itx_anchor] H18 ch4 overall={ev['overall_h18_mae']:.3f} "
      f"bins={[round(x, 3) for x in ev['bins_q1q5']]} | "
      f"best_val={tr['best_val_nll']:.3f}@{tr['best_epoch']}", flush=True)
(OUT / "report_itx_anchor.json").write_text(
    json.dumps({"train": tr, "eval": ev}, indent=2))
print("done")
