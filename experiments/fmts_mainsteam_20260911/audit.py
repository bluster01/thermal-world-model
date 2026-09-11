"""Read-only artifact and numerical replay audit of an FMTS run directory."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import RichRecord
from .manifests import sha256
from .models import RichBlackbox, RichFusion
from .run import evaluate
from src.final_wm.training import TrainSpec
from src.final_wm.properties import load_grid_properties
from src.final_wm.contracts import FinalWMProtocolError


def audit(out, record_path=None, properties_path=None, mapping="configs/final_wm/channel_mapping_v2.json", device="cpu"):
    out = Path(out)
    identity = json.loads((out / "identity.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    if sha256(out / "indices.npz") != identity["indices"]:
        raise FinalWMProtocolError("index manifest hash mismatch")
    if summary["locked_test_evaluated"]:
        raise FinalWMProtocolError("unexpected test access")
    record = None
    if record_path:
        if sha256(record_path) != identity["record"] or sha256(mapping) != identity["mapping"]:
            raise FinalWMProtocolError("replay input hash mismatch")
        if not identity["smoke"] and (not properties_path or sha256(properties_path) != identity["properties"]):
            raise FinalWMProtocolError("replay requires identical IAPWS properties")
        record = RichRecord(record_path, mapping)
    checked, replayed = 0, 0
    with np.load(out / "indices.npz") as indices:
        for report in summary["runs"]:
            if report["status"] != "complete":
                continue
            run_dir = out / f"{report['arm']}_seed{report['seed']}"
            for file, field in (("best.pt","checkpoint_sha256"),("prediction.npz","prediction_sha256"),("response.npz","response_sha256")):
                if sha256(run_dir / file) != report[field]:
                    raise FinalWMProtocolError(f"artifact hash mismatch: {run_dir / file}")
            payload = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            if payload["identity_sha256"] != sha256(out / "identity.json"):
                raise FinalWMProtocolError("checkpoint provenance mismatch")
            if record is not None:
                stats = payload["normalization"]
                props = load_grid_properties(properties_path) if properties_path else None
                model = (RichBlackbox(stats["mean"],stats["std"]) if report["arm"] == "blackbox_itransformer"
                         else RichFusion(TrainSpec(**payload["spec"]),stats["mean"],stats["std"],props)).to(device)
                model.load_state_dict(payload["state_dict"],strict=True)
            for name, response in (("prediction",False),("response",True)):
                expected = indices["response" if response else "validation"]
                with np.load(run_dir / f"{name}.npz") as raw:
                    if not np.array_equal(raw["starts"],expected):
                        raise FinalWMProtocolError("unmatched windows")
                    if record is not None:
                        replay = evaluate(model,record,torch.from_numpy(expected),device,response=response)
                        for key in raw.files:
                            if not np.allclose(replay[key],raw[key],rtol=1e-5,atol=1e-5):
                                raise FinalWMProtocolError(f"numerical replay mismatch: {name}/{key}")
                replayed += int(record is not None)
            checked += 1
    return {"runs_checked": checked, "array_sets_replayed": replayed,
            "failed_runs": [r for r in summary["runs"] if r["status"] != "complete"],
            "status": summary["status"], "complete": checked == (4 if identity["smoke"] else 12)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--out",required=True)
    ap.add_argument("--record")
    ap.add_argument("--properties")
    ap.add_argument("--mapping",default="configs/final_wm/channel_mapping_v2.json")
    ap.add_argument("--device",default="cpu")
    args = ap.parse_args()
    result = audit(args.out,args.record,args.properties,args.mapping,args.device)
    print(json.dumps(result,indent=2))
    if not result["complete"]:
        raise SystemExit(1)
