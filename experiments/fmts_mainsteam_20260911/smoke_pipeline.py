"""Generate reproducible synthetic v2.2 inputs and exercise all four arms."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from src.final_wm.synthetic import synthetic_canonical_arrays
from .run import main, write_json
from .audit import audit


if __name__ == "__main__":
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--out",required=True,help="new synthetic bundle directory")
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    arrays = synthetic_canonical_arrays(total_steps=2200,seed=11)
    centers = np.array([250,4,400,5,600,500,500,4,400])
    arrays["boundary_ext"] = (centers[None]*(1+np.random.default_rng(10).normal(0,.01,(2200,9)))).astype("float32")
    arrays["aux"] = np.zeros((2200,15),dtype="float32")
    arrays["mill_on"] = np.zeros((2200,8),dtype="uint8")
    path = root / "synthetic_v22.npz"
    np.savez_compressed(path,**arrays)
    sys.argv = ["run","--record",str(path),"--out",str(root / "runs"),"--smoke"]
    main()
    result = audit(root / "runs",path)
    write_json(root / "audit.json",result)
    print(result)
