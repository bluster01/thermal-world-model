"""Fixed train/validation windows with v2.2 history quality checks."""
import json
from pathlib import Path
import numpy as np
import torch
from src.final_wm.contracts import FinalWMProtocolError
from src.final_wm.data import SPLIT_TRAIN, SPLIT_VAL
from src.final_wm.data_v2 import CanonicalV2Record, BOUNDARY_EXT_ELEMENTS
from .manifests import windows_from_starts


class RichRecord(CanonicalV2Record):
    def __init__(self, path, mapping):
        super().__init__(path)
        if tuple(self.boundary_ext_elements) != BOUNDARY_EXT_ELEMENTS:
            raise FinalWMProtocolError("nine-column canonical v2.2 required")
        with np.load(path) as arrays:
            self.valid = torch.from_numpy(arrays["valid"].astype(bool))
        limits = json.loads(Path(mapping).read_text(encoding="utf-8"))["boundary_ext"]
        self.ext_valid = torch.isfinite(self.boundary_ext).all(1)
        for i, name in enumerate(BOUNDARY_EXT_ELEMENTS):
            lo, hi = limits[name]["range"]
            self.ext_valid &= (self.boundary_ext[:, i] >= lo) & (self.boundary_ext[:, i] <= hi)
        self.base_valid = self.valid & torch.isfinite(self.obs).all(1)
        self.base_valid &= torch.isfinite(self.actions).all(1) & torch.isfinite(self.boundary).all(1)

    def candidates(self, split, history=96, horizon=18):
        if split not in (SPLIT_TRAIN, SPLIT_VAL):
            raise FinalWMProtocolError("only train/validation are permitted")
        # Only historical extensions affect eligibility; never future extensions.
        starts = torch.arange(history, self.n - horizon + 1)
        def count(mask, lo, hi):
            prefix = torch.cat([torch.zeros(1, dtype=torch.long), mask.long().cumsum(0)])
            return prefix[hi] - prefix[lo]
        full = (self.split == split) & self.base_valid
        good = count(full, starts-history, starts+horizon) == history+horizon
        good &= count(self.ext_valid, starts-history, starts) == history
        discontinuity = torch.cat([torch.zeros(1, dtype=torch.bool), torch.diff(self.timestamps) != 10])
        good &= count(discontinuity, starts-history+1, starts+horizon) == 0
        return starts[good]

    def batch(self, starts, split, device):
        b = windows_from_starts(self, starts, split, 96, 18)
        indices = starts[:, None] + torch.arange(-96, 0)
        ext = self.boundary_ext[indices].to(device)
        h = b.history.__class__(*(t.to(device) for t in b.history))
        return h, ext, b.future_actions.to(device), b.future_boundary.to(device), b.future_obs.to(device), b.day_ids

    def normalization(self):
        mask = (self.split == SPLIT_TRAIN) & self.base_valid & self.ext_valid
        x = torch.cat([self.obs, self.actions, self.boundary, self.boundary_ext], -1)[mask]
        if len(x) < 2:
            raise FinalWMProtocolError("not enough valid train rows")
        return x.mean(0), x.std(0).clamp_min(1e-3)


def make_indices(record, quick=False):
    arrays = {}
    for name, split, n, seed in (("train", 0, 20000, 15000),
                                ("validation", 1, 256, 30000),
                                ("response", 1, 64, 60000)):
        candidates = record.candidates(split)
        if len(candidates) == 0:
            raise FinalWMProtocolError(f"no eligible {name} windows")
        n = min(n, 4) if quick else n
        gen = torch.Generator().manual_seed(seed)
        arrays[name] = candidates[torch.randint(len(candidates), (n,), generator=gen)]
    return arrays
