"""Verdict audit: replay side-A matrix verdicts from returned raw artifacts.

Independent of the runner's in-memory path: reloads per-run metrics from
disk, recomputes day-paired bootstrap CIs via the frozen code
(`relative_improvement_ci`), and compares every verdict against
`matrix_summary_sideA.json`.  Also checks ledger completeness/consistency,
the D-SYN gate, and the split-sides SHA.  Reads validation artifacts only.

Usage:
    python experiments/final_wm/audit_verdicts.py --out artifacts/final_wm --side A
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.final_wm import matrix_spec as ms
from src.final_wm.data import SPLIT_VAL, CanonicalRecord
from src.final_wm.evaluation import (
    WindowMetrics,
    persistence_boundary_metrics,
    relative_improvement_ci,
)

TOL = 1e-6
# Historical v0.2 replay constants.  They stay local so v0.7 cannot reuse
# percentage-NLL thresholds for a new formal verdict.
V02_THRESH_O1_NLL = 0.05
V02_THRESH_T1_NLL = 0.02
V02_THRESH_J1_NLL = 0.03


def _load_metrics(path: Path) -> WindowMetrics:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = blob["metrics"] if "metrics" in blob else blob
    return WindowMetrics(**m)


def _replay_pairs(pairs, threshold, metric="nll") -> dict:
    details = []
    for base, arm in pairs:
        ci = relative_improvement_ci(base, arm, horizon=ms.HORIZON, metric=metric)
        ok = ci.point >= threshold and ci.ci_lo > 0.0
        details.append({"point": ci.point, "ci_lo": ci.ci_lo, "ci_hi": ci.ci_hi,
                        "n_days": ci.n_days, "pass": bool(ok)})
    passes = sum(d["pass"] for d in details)
    verdict = "SUPPORTED" if passes >= ms.MIN_SEED_PASSES else ("REJECTED" if passes == 0 else "MIXED")
    return {"verdict": verdict, "per_seed": details}


def _compare(name: str, recomputed: dict, recorded: dict, checks: list) -> None:
    ok = recomputed["verdict"] == recorded.get("verdict")
    seeds_ok = True
    for rec, rep in zip(recomputed["per_seed"], recorded.get("per_seed", [])):
        for key in ("point", "ci_lo", "ci_hi"):
            if abs(rec[key] - rep[key]) > TOL:
                seeds_ok = False
        if bool(rec["pass"]) != bool(rep["pass"]):
            seeds_ok = False
    checks.append({"unit": name, "verdict_match": ok, "per_seed_match": seeds_ok,
                   "recomputed_verdict": recomputed["verdict"],
                   "recorded_verdict": recorded.get("verdict")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/final_wm")
    parser.add_argument("--side", default="A")
    parser.add_argument("--record", default=None)
    args = parser.parse_args()

    out = Path(args.out)
    summary = json.loads((out / f"matrix_summary_side{args.side}.json").read_text(encoding="utf-8"))
    if summary.get("matrix_version") != "0.2":
        raise RuntimeError("audit_verdicts.py is the historical v0.2 replayer only")
    checks: list = []

    def m(run_id: str) -> WindowMetrics:
        return _load_metrics(out / "metrics" / f"{run_id}.pt")

    # O1
    for arm in ("learned", "hybrid"):
        pairs = [(m(f"o1_steady_seed{s}"), m(f"o1_{arm}_seed{s}")) for s in ms.SEEDS]
        _compare(f"o1_{arm}", _replay_pairs(pairs, V02_THRESH_O1_NLL),
                 summary["units"]["o1"][arm], checks)

    # T1 nested arms
    for arm, base_arm in [("closure_cons", "physics_only"), ("closure_steam", "closure_cons"),
                          ("latent4", "closure_cons")]:
        pairs = [(m(f"t1_{base_arm}_seed{s}"), m(f"t1_{arm}_seed{s}")) for s in ms.SEEDS]
        _compare(f"t1_{arm}_vs_{base_arm}", _replay_pairs(pairs, V02_THRESH_T1_NLL),
                 summary["units"]["t1"][f"{arm}_vs_{base_arm}"], checks)

    # B1: persistence baseline recomputed from the record (deterministic).
    record_path = args.record or summary.get("record")
    record = CanonicalRecord(record_path)
    pairs = []
    for s in ms.SEEDS:
        base = persistence_boundary_metrics(
            record, SPLIT_VAL, n_windows=256, batch_size=32,
            history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON, seed=60_000 + s)
        pairs.append((base, m(f"b1_gru_seed{s}_boundary")))
    _compare("b1", _replay_pairs(pairs, ms.THRESH_B1_CRPS, metric="crps"),
             summary["units"]["b1"], checks)

    # J1
    pairs = [(m(f"j1_staged_seed{s}"), m(f"j1_joint_seed{s}")) for s in ms.SEEDS]
    _compare("j1", _replay_pairs(pairs, V02_THRESH_J1_NLL), summary["units"]["j1"], checks)

    # R1: rule replay + cross-file consistency
    r1_file = json.loads((out / "r1_report.json").read_text(encoding="utf-8"))
    r1_sum = summary["units"]["r1"]
    expected = "SUPPORTED"
    for rep in r1_file["reports"]:
        if "error" in rep:
            expected = "MIXED"
            continue
        if not rep["runtime_blind_ok"] or rep["leakage"]["leakage_suspected"]:
            expected = "REJECTED"
        if rep["direction"]["frac_negative"] < 1.0:
            expected = "REJECTED"
    checks.append({
        "unit": "r1",
        "verdict_match": expected == r1_sum["verdict"] == r1_file["verdict"],
        "report_file_consistent": r1_file["reports"] == r1_sum["reports"],
        "recomputed_verdict": expected, "recorded_verdict": r1_sum["verdict"],
        "per_seed_match": True,
    })

    # Ledger completeness/consistency
    ledger = [json.loads(line) for line in (out / "ledger.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    finals = {}
    dup_count: dict[str, int] = {}
    for entry in ledger:  # last occurrence wins (duplicate-block convention)
        if entry.get("final"):
            finals[entry["run_id"]] = entry
            dup_count[entry["run_id"]] = dup_count.get(entry["run_id"], 0) + 1
    expected_runs = (
        [f"o1_{a}_seed{s}" for a in ("steady", "learned", "hybrid") for s in ms.SEEDS]
        + [f"t1_{a}_seed{s}" for a in ("physics_only", "closure_cons", "closure_steam", "latent4")
           for s in ms.SEEDS]
        + [f"b1_gru_seed{s}" for s in ms.SEEDS]
        + [f"j1_{a}_seed{s}" for a in ("joint", "staged_main") for s in ms.SEEDS]
        + [f"j1_staged_boundary_from_{s}_seed{s}" for s in ms.SEEDS]
    )
    missing = [r for r in expected_runs if r not in finals]
    conv_missing = [
        r for r in finals if r.startswith("t1_")
        and any(k not in finals[r] for k in ("stop_reason", "converged", "val_tail"))
    ]
    epochs_bad = [
        r for r in finals
        if finals[r].get("epochs_run", 0) > (finals[r].get("spec") or {}).get("epochs", 10**9)
    ]
    checks.append({
        "unit": "ledger",
        "entries": len(ledger), "final_runs": len(finals),
        "missing_runs": missing,
        "t1_convergence_fields_missing": conv_missing,
        "epochs_over_cap": epochs_bad,
        "duplicate_run_ids": sorted(r for r, c in dup_count.items() if c > 1),
        "commits": sorted({e.get("commit") for e in finals.values()}),
    })

    # Gates
    dsyn = json.loads((out / "dsyn_verdict.json").read_text(encoding="utf-8"))
    checks.append({"unit": "dsyn", "verdict_match": dsyn["verdict"] == "PASS",
                   "passes": dsyn["passes"], "quick": dsyn.get("quick")})
    split = json.loads((out / "split_sides_report.json").read_text(encoding="utf-8"))
    metas = {}
    for side, rel in split.get("records", {}).items():
        meta_path = Path(rel.replace(".npz", "_meta.json"))
        if meta_path.exists():
            metas[side] = json.loads(meta_path.read_text(encoding="utf-8"))
    shas = {s: m["provenance"]["dual_record_sha256"] for s, m in metas.items()}
    gates_ok = all(
        m["quality"]["gap_ratio"] <= 0.01
        and m["quality"]["valve_active_ratio"] >= 0.60
        and m["quality"]["days"] >= 60.0
        and m.get("test_locked") is True
        for m in metas.values()
    )
    splits_same = len({json.dumps(m["splits"], sort_keys=True) for m in metas.values()}) == 1
    checks.append({
        "unit": "split_sides",
        "verdict_match": len(shas) == 2 and len(set(shas.values())) == 1 and gates_ok and splits_same,
        "dual_record_sha256": sorted(set(shas.values())),
        "gates_ok": gates_ok,
        "splits_identical": splits_same,
    })

    audit = {"side": args.side, "matrix_version": summary.get("matrix_version"),
             "properties": summary.get("properties"), "checks": checks}
    (out / f"verdict_audit_side{args.side}.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[audit] side={args.side} version={summary.get('matrix_version')} "
          f"properties={summary.get('properties')}")
    for c in checks:
        if c["unit"] == "ledger":
            c["verdict_match"] = not (c["missing_runs"] or c["t1_convergence_fields_missing"]
                                      or c["epochs_over_cap"])
        flag = "OK " if c.get("verdict_match") and c.get("per_seed_match", True) else "FAIL"
        extra = ""
        if c["unit"] == "ledger":
            extra = (f" missing={c['missing_runs']} conv_missing={c['t1_convergence_fields_missing']}"
                     f" over_cap={c['epochs_over_cap']} commits={c['commits']}")
        print(f"[audit] {flag} {c['unit']}: {c.get('recorded_verdict', '')}"
              f" vs replay {c.get('recomputed_verdict', '')}{extra}")


if __name__ == "__main__":
    main()
