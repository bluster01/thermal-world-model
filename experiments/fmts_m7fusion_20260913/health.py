"""Observer-only diagnostics on fixed histories; no target or action sign score."""
import numpy as np
import torch
from src.final_wm.contracts import FinalWMProtocolError


@torch.no_grad()
def observer_health(model, record, starts, device='cpu'):
    model.eval()
    observer = model.base.observer
    collected = {}
    for idx in starts.split(32):
        history, extension, *_ = record.batch(idx, 1, device)
        extension = (extension-model.history_mean[-9:])/model.history_std[-9:]
        boundary = torch.cat([history.boundary, extension], -1)
        anchor = torch.zeros(len(idx), model.base.layout.dim, device=device)
        raw_values = []
        hook = observer.mu_head.register_forward_hook(
            lambda module, inputs, out: raw_values.append(out.detach()))
        try:
            observer.posterior(history.obs, history.actions, boundary, anchor)
            def reverse_past(x):
                return torch.cat([x[:, :-1].flip(1), x[:, -1:]], 1)
            observer.posterior(reverse_past(history.obs), reverse_past(history.actions),
                               reverse_past(boundary), anchor)
        finally:
            hook.remove()
        raw = raw_values[0].reshape(len(idx), -1)[:, 3:7]
        reversed_raw = raw_values[1].reshape(len(idx), -1)[:, 3:7]
        fraction = torch.tanh(raw)
        values = dict(starts=idx, raw=raw, correction_fraction=fraction,
                      tanh_derivative=1-fraction.square(),
                      reversed_history_fraction=torch.tanh(reversed_raw))
        for name, value in values.items():
            if not torch.isfinite(value).all():
                raise FinalWMProtocolError('nonfinite observer health arrays')
            collected.setdefault(name, []).append(value.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in collected.items()}


def summarize_health(raw):
    f = raw['correction_fraction']
    return dict(n=len(f), active_state_indices=[3,4,5,6],
                correction_fraction_mean=f.mean(0).tolist(),
                correction_fraction_std=f.std(0).tolist(),
                saturation_abs_gt_0p99=(np.abs(f)>.99).mean(0).tolist(),
                tanh_derivative_mean=raw['tanh_derivative'].mean(0).tolist(),
                reversal_mean_abs_change=np.abs(f-raw['reversed_history_fraction']).mean(0).tolist(),
                diagnostic_only=True, checkpoint_selector=False)
