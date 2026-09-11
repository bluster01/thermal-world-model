"""Linux entry point: fixed-window training, raw predictions and valve probes.

Run with --smoke for a tiny engineering test; full runs require IAPWS properties.
Existing output directories are never overwritten or silently resumed.
"""
import argparse
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from .data import RichRecord, make_indices
from .models import RichBlackbox, RichFusion
from .manifests import sha256
from .spec import serialized_spec, structured_specs, BLACKBOX_SPEC
from src.final_wm.contracts import action_support_from_history, FinalWMProtocolError
from src.final_wm.properties import load_grid_properties


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def day_mean(values, days):
    return np.stack([values[days == d].mean(0) for d in np.unique(days)]).mean(0)


def cumulative(errors):
    return errors.cumsum(-1) / np.arange(1, errors.shape[-1] + 1)


def block_interval(values, days):
    blocks = np.stack([values[days == d].mean(0) for d in np.unique(days)])
    rng = np.random.default_rng(11000)
    samples = blocks[rng.integers(len(blocks), size=(1000, len(blocks)))].mean(1)
    return {"mean": blocks.mean(0).tolist(), "ci95": np.quantile(samples,[.025,.975],axis=0).tolist(),
            "days": len(blocks), "degenerate_single_day": len(blocks) == 1}


@torch.no_grad()
def evaluate(model, record, starts, device, response=False, batch_size=32):
    model.eval()
    collected = {}
    def add(name, value):
        collected.setdefault(name, []).append(value.detach().cpu().numpy())
    for idx in starts.split(batch_size):
        h, ext, actions, boundary, target, days = record.batch(idx, 1, device)
        def predict(a, b):
            out = model(h, ext, a, b)
            pred = out if isinstance(out, torch.Tensor) else out.temps_mu[:, :, 4]
            if not torch.isfinite(pred).all():
                raise FinalWMProtocolError("nonfinite prediction")
            return pred
        add("starts", idx)
        add("days", days)
        if not response:
            add("prediction", predict(actions, boundary))
            add("target", target[:, :, 4])
            add("persistence", h.obs[:, -1:, 4].expand(-1, 18))
        else:
            actions = h.actions[:, -1:].expand(-1, 18, -1).clone()
            boundary = h.boundary[:, -1:].expand(-1, 18, -1).clone()
            base = predict(actions, boundary)
            if not torch.equal(base, predict(actions.clone(), boundary.clone())):
                raise FinalWMProtocolError("identical-action identity gate failed")
            add("base", base)
            support = action_support_from_history(h.actions, margin=0.05)
            for valve in (0, 1):
                step = actions.clone()
                step[:, :, valve] = (step[:, :, valve] + 0.05).clamp(max=1.0)
                add(f"valve{valve+1}_prediction", predict(step, boundary))
                add(f"valve{valve+1}_support", support.contains(step) & support.contains(actions))
                add(f"valve{valve+1}_dose", step[:, 0, valve] - actions[:, 0, valve])
    return {k: np.concatenate(v) for k, v in collected.items()}


def train_one(arm, seed, spec, record, indices, mean, std, properties, device, out, smoke):
    torch.manual_seed(seed)
    np.random.seed(seed)
    blackbox = arm == "blackbox_itransformer"
    model = (RichBlackbox(mean, std) if blackbox else RichFusion(spec, mean, std, properties)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    # Each pair of hybrid arms shares both initialization seed and minibatch draws.
    generator = torch.Generator().manual_seed(20000 + seed)
    batch = 128 if blackbox else spec.batch_size
    every = 500 if blackbox else spec.batches_per_epoch
    cap = 3000 if blackbox else spec.epochs * every
    patience = 3 if blackbox else spec.patience
    if smoke:
        batch, every, cap, patience = 2, 1, 2, 2
    run_dir = out / f"{arm}_seed{seed}"
    run_dir.mkdir()
    best, stale, best_step = float("inf"), 0, -1
    active_parameters = set()
    started = time.time()
    with (run_dir / "ledger.jsonl").open("w", encoding="utf-8") as ledger:
        for step in range(1, cap + 1):
            model.train()
            selection = torch.randint(len(indices["train"]), (batch,), generator=generator)
            h, ext, a, b, y, _ = record.batch(indices["train"][selection], 0, device)
            result = model(h, ext, a, b)
            loss = ((result - y[:, :, 4]) ** 2).mean() if blackbox else model.base.observation_nll(
                result.temps_mu, result.temps_sigma, y)
            if not torch.isfinite(loss):
                raise FinalWMProtocolError("nonfinite training loss")
            opt.zero_grad()
            loss.backward()
            active_parameters.update(n for n,p in model.named_parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 if blackbox else 10.0,
                                          error_if_nonfinite=True)
            opt.step()
            if step % every == 0:
                raw = evaluate(model, record, indices["validation"], device)
                score = float(day_mean(np.abs(raw["prediction"] - raw["target"]).mean(1), raw["days"]))
                ledger.write(json.dumps({"step": step, "loss": float(loss.detach()),
                                        "validation_mainsteam_mae": score}) + "\n")
                ledger.flush()
                if score < best:
                    best, stale, best_step = score, 0, step
                    torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": seed,
                        "spec": BLACKBOX_SPEC if blackbox else asdict(spec),
                        "model_config": None if blackbox else asdict(model.base.config),
                        "protocol": serialized_spec(), "step": step,
                        "identity_sha256": sha256(out / "identity.json"),
                        "normalization": {"mean": mean, "std": std}}, run_dir / "best.pt")
                else:
                    stale += 1
                if stale >= patience:
                    break
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False)["state_dict"])
    raw = evaluate(model, record, indices["validation"], device)
    probe = evaluate(model, record, indices["response"], device, response=True)
    np.savez_compressed(run_dir / "prediction.npz", **raw)
    np.savez_compressed(run_dir / "response.npz", **probe)
    curve = day_mean(cumulative(np.abs(raw["prediction"] - raw["target"])), raw["days"])
    report = {"arm": arm, "seed": seed, "status": "complete", "updates": step,
        "best_step": best_step, "stop_reason": "patience" if stale >= patience else "cap",
        "seconds": time.time()-started, "parameters_instantiated": sum(p.numel() for p in model.parameters()),
        "parameters_with_gradients": sum(p.numel() for n,p in model.named_parameters() if n in active_parameters),
        "parameter_names_with_gradients": sorted(active_parameters),
        "cumulative_mae": curve.tolist(), "H18": float(curve[-1]), "identity_gate": True,
        "single_step_mae": day_mean(np.abs(raw["prediction"]-raw["target"]),raw["days"]).tolist(),
        "persistence_cumulative_mae": day_mean(cumulative(np.abs(raw["persistence"]-raw["target"])),raw["days"]).tolist(),
        "prediction_sha256": sha256(run_dir / "prediction.npz"),
        "response_sha256": sha256(run_dir / "response.npz"), "checkpoint_sha256": sha256(run_dir / "best.pt")}
    write_json(run_dir / "report.json", report)
    return report


def summarize(out, reports, smoke):
    complete = [r for r in reports if r["status"] == "complete"]
    selection = "INCOMPLETE"
    lookup = {(r["arm"], r["seed"]): r for r in complete}
    if not smoke and all((arm, s) in lookup for arm in ("fusion_gru_norew", "fusion_token_xattn_norew") for s in (0,1,2)):
        gru = np.array([lookup["fusion_gru_norew", s]["H18"] for s in (0,1,2)])
        token = np.array([lookup["fusion_token_xattn_norew", s]["H18"] for s in (0,1,2)])
        selection = "fusion_token_xattn_norew" if (token < gru).sum() >= 2 and token.mean() <= gru.mean() else "fusion_gru_norew"
    stats = {}
    for r in complete:
        key = f"{r['arm']}_seed{r['seed']}"
        with np.load(out / key / "response.npz") as raw:
            stats[key] = {}
            for v in (1,2):
                delta = raw[f"valve{v}_prediction"] - raw["base"]
                support = raw[f"valve{v}_support"].all(1)
                stats[key][f"valve{v}"] = {"curve": day_mean(delta, raw["days"]).tolist(),
                    "day_block_interval": block_interval(delta, raw["days"]),
                    "negative_fraction": day_mean((delta < 0).astype(float), raw["days"]).tolist(),
                    "unsupported_windows": int((~support).sum()), "n": len(delta),
                    "supported_curve": day_mean(delta[support], raw["days"][support]).tolist() if support.any() else None}
    pairs = {}
    for a, b in combinations(sorted(set(r["arm"] for r in complete)), 2):
        for seed in (0,1,2):
            if (a,seed) not in lookup or (b,seed) not in lookup:
                continue
            with np.load(out / f"{a}_seed{seed}" / "prediction.npz") as ra, np.load(out / f"{b}_seed{seed}" / "prediction.npz") as rb:
                if not np.array_equal(ra["starts"], rb["starts"]) or not np.array_equal(ra["target"], rb["target"]):
                    raise FinalWMProtocolError("unmatched paired outputs")
                diff = np.abs(ra["prediction"]-ra["target"]).mean(1)-np.abs(rb["prediction"]-rb["target"]).mean(1)
                days = ra["days"]
                blocks = np.array([diff[days == d].mean() for d in np.unique(days)])
                rng = np.random.default_rng(11000)
                boot = blocks[rng.integers(len(blocks), size=(1000,len(blocks)))].mean(1)
                pairs[f"{a} minus {b} seed{seed}"] = {"mean": float(blocks.mean()),
                    "ci95": np.quantile(boot,[.025,.975]).tolist(), "days": len(blocks)}
    seed_summary = {}
    for arm in sorted(set(r["arm"] for r in complete)):
        group = [r for r in complete if r["arm"] == arm]
        curves = np.array([r["cumulative_mae"] for r in group])
        seed_summary[arm] = {"seeds": [r["seed"] for r in group], "mean": curves.mean(0).tolist(),
            "seed_min": curves.min(0).tolist(), "seed_max": curves.max(0).tolist(),
            "seed_std": curves.std(0,ddof=1).tolist() if len(group)>1 else None,
            "three_seed_complete": len(group)==3}
        for v in (1,2):
            response = np.array([stats[f"{arm}_seed{r['seed']}"][f"valve{v}"]["curve"] for r in group])
            seed_summary[arm][f"valve{v}"] = {"mean": response.mean(0).tolist(),
                "seed_min": response.min(0).tolist(), "seed_max": response.max(0).tolist()}
    write_json(out / "summary.json", {"status": "SMOKE" if smoke else "VALIDATION_EXPLORATORY",
        "seed_summary": seed_summary,
        "runs": reports, "selected_hybrid": selection, "response": stats, "paired_H18": pairs,
        "locked_test_evaluated": False, "plant_response_truth": False})


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--record", required=True)
    ap.add_argument("--mapping", default="configs/final_wm/channel_mapping_v2.json")
    ap.add_argument("--properties")
    ap.add_argument("--reference-manifest", default="artifacts/final_wm/v07_full_reissue_v1/sideA/manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not args.smoke and not args.properties:
        ap.error("full runs require --properties (IAPWS grid)")
    if not args.smoke:
        reference = json.loads(Path(args.reference_manifest).read_text(encoding="utf-8"))
        if reference.get("side") != "A":
            ap.error("reference manifest must identify Side A")
        for name, path in (("record",args.record),("properties",args.properties)):
            if sha256(path) != reference["inputs"][name]["sha256"]:
                ap.error(f"{name} does not match the audited v0.7 input identity")
    out = Path(args.out)
    if out.exists():
        ap.error("output path exists; preserve it and choose a new path")
    record = RichRecord(args.record, args.mapping)
    indices = make_indices(record, args.smoke)
    mean, std = record.normalization()
    out.mkdir(parents=True)
    np.savez_compressed(out / "indices.npz", **{k:v.numpy() for k,v in indices.items()})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    roots = [Path("src/final_wm"), Path(__file__).parent]
    sources = {str(p).replace("\\", "/"): sha256(p) for root in roots for p in sorted(root.glob("*.py"))}
    write_json(out / "identity.json", {"protocol": serialized_spec(), "smoke": args.smoke,
        "git_head": revision, "sources": sources, "record": sha256(args.record),
        "mapping": sha256(args.mapping), "properties": sha256(args.properties) if args.properties else "ANALYTIC_SMOKE_ONLY",
        "indices": sha256(out / "indices.npz"), "python": platform.python_version(),
        "reference_manifest": sha256(args.reference_manifest) if not args.smoke else None,
        "torch": torch.__version__, "device": args.device, "normalization_mean": mean.tolist(),
        "cuda_version": torch.version.cuda,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "normalization_std": std.tolist()})
    reports = []
    seeds = (0,) if args.smoke else (0,1,2)
    specs = {(s.arm, s.seed):s for s in structured_specs(seeds)}
    arms = ["blackbox_itransformer", "greybox_steady_none", "fusion_gru_norew", "fusion_token_xattn_norew"]
    for arm in arms:
        for seed in seeds:
            print(f"Starting {arm} seed{seed}", flush=True)
            try:
                props = load_grid_properties(args.properties) if args.properties else None
                report = train_one(arm, seed, specs.get((arm,seed)), record, indices, mean, std,
                                   props, args.device, out, args.smoke)
            except Exception as exc:
                report = {"arm": arm, "seed": seed, "status": "failed", "error": str(exc)}
                write_json(out / f"{arm}_seed{seed}_failure.json", report)
            reports.append(report)
            write_json(out / "progress.json", reports)
    summarize(out, reports, args.smoke)
    if any(r["status"] != "complete" for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
