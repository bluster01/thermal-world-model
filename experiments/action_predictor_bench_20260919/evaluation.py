"""Shared forecast and action probes. Only predicted temperatures enter feedback."""
import numpy as np
import torch

from .data import TRAIN_H, DT, unpack
from .models import Anchored


def forecast(model, history, actions, boundaries, *, mode='block'):
    """Controls have H+1 rows. Blocks use full32 feedback, never rolling truth."""
    horizon = actions.shape[1] - 1
    if horizon < 1 or boundaries.shape[1] != horizon + 1:
        raise ValueError('Expected H+1 left/right endpoint controls')
    if hasattr(model, 'forecast_plan'):
        return model.forecast_plan(history, actions, boundaries, mode=mode)
    if isinstance(model, Anchored):
        # Keep one nominal plan for the ENTIRE candidate comparison, not one
        # redefined from each candidate's generated block history.
        nominal = history[:, -1:, 5:7].expand_as(actions)
        return (forecast(model.predictor, history, nominal, boundaries, mode='block')
                + (forecast(model.response, history, actions, boundaries, mode=mode)
                   - forecast(model.response, history, nominal, boundaries, mode=mode)))
    if mode == 'native':
        if not model.native_long and horizon > TRAIN_H:
            raise ValueError('Native extrapolation unavailable for fixed direct head')
        return model(history, actions[:, :-1], boundaries[:, :-1])
    if mode != 'block':
        raise ValueError(mode)
    current, output = history, []
    for start in range(0, horizon, TRAIN_H):
        stop = min(start + TRAIN_H, horizon)
        prediction = model(current, actions[:, start:stop], boundaries[:, start:stop])
        output.append(prediction)
        # prediction[k] is T at t+k+1, paired with u/d at t+k+1, NOT t+k.
        rows = torch.cat((prediction, actions[:, start+1:stop+1], boundaries[:, start+1:stop+1]), -1)
        current = torch.cat((current, rows), 1)[:, -history.shape[1]:]
    return torch.cat(output, 1)


def metrics(prediction, truth, anchor):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    valid = np.isfinite(p).all((1, 2))
    result = {'windows': len(p), 'finite_window_fraction': float(valid.mean())}
    if not valid.all():
        return {**result, 'failure': 'Nonfinite predictions; errors not computed on a cherry-picked finite subset'}
    error = np.abs(p - y)
    result.update(mae_by_step_channel_C=error.mean(0).tolist(),
                  main_mae_C=float(error[:, :, 4].mean()), all_mae_C=float(error.mean()),
                  prediction_min_per_channel_C=p.min((0, 1)).tolist(),
                  prediction_max_per_channel_C=p.max((0, 1)).tolist(),
                  per_window_main_mae_C=error[:, :, 4].mean(1).tolist())
    for h in (1, 6, 18, 32, 64, 128, 256, 512):
        if h <= p.shape[1]:
            result[f'H{h}'] = {'main_mae_C': float(error[:, :h, 4].mean()),
                               'all_mae_C': float(error[:, :h].mean()),
                               'per_channel_mae_C': error[:, :h].mean((0, 1)).tolist(),
                               'endpoint_main_mae_C': float(error[:, h-1, 4].mean())}
    for lo, hi in ((32, 128), (128, 512)):
        if hi <= p.shape[1]:
            result[f'tail_{lo+1}_{hi}'] = {'main_mae_C': float(error[:, lo:hi, 4].mean()),
                                          'per_channel_mae_C': error[:, lo:hi].mean((0, 1)).tolist()}
    p0 = np.concatenate((np.asarray(anchor)[:, None, :5], p), 1)
    y0 = np.concatenate((np.asarray(anchor)[:, None, :5], y), 1)
    for lag in (6, 18, 60):
        if lag <= p.shape[1]:
            dp, dy = p0[:, lag:, 4] - p0[:, :-lag, 4], y0[:, lag:, 4] - y0[:, :-lag, 4]
            corr = np.corrcoef(dp.ravel(), dy.ravel())[0, 1] if dp.std() > 1e-12 and dy.std() > 1e-12 else None
            result[f'delta_{lag*DT}s'] = {'mae_C': float(np.abs(dp-dy).mean()),
                                         'correlation': None if corr is None else float(corr),
                                         'amplitude_ratio': float(np.abs(dp).mean() / max(np.abs(dy).mean(), 1e-12))}
    result['peak_time_mae_seconds'] = float(np.abs(p[:, :, 4].argmax(1)-y[:, :, 4].argmax(1)).mean()*DT)
    result['trough_time_mae_seconds'] = float(np.abs(p[:, :, 4].argmin(1)-y[:, :, 4].argmin(1)).mean()*DT)
    return result


@torch.no_grad()
def evaluate_forecasts(model, bank, device, batch_size=64):
    h, u, d, y = unpack(bank, device)
    modes = ['block_recorded', 'block_held_boundary']
    if model.native_long:
        modes.append('native_recorded')
    arrays, scores = {}, {}
    for name in modes:
        pred = []
        for i in range(0, len(bank), batch_size):
            hi, ui, di = h[i:i+batch_size], u[i:i+batch_size], d[i:i+batch_size]
            if name.endswith('held_boundary'):
                di = di[:, :1].expand_as(di)
            pi = forecast(model, hi, ui, di, mode='native' if name.startswith('native') else 'block')
            pred.append(pi.cpu().numpy())
        p = np.concatenate(pred)
        arrays[name] = p
        scores[name] = metrics(p, y.cpu().numpy(), h[:, -1].cpu().numpy())
        # Chronological reporting strata, not unseen-regime claims.
        mid = len(bank) // 2
        scores[name]['early_validation_main_mae_C'] = float(np.abs(p[:mid, :, 4] - bank[:mid, h.shape[1]:, 4]).mean())
        scores[name]['late_validation_main_mae_C'] = float(np.abs(p[mid:, :, 4] - bank[mid:, h.shape[1]:, 4]).mean())
    if 'native_recorded' in arrays:
        scores['native_block_gap_C'] = float(np.abs(arrays['native_recorded'] - arrays['block_recorded']).mean())
    return scores, arrays


def scenarios(length=160):
    """110 deterministic plans, 32-step burn-in +128-step scoring window."""
    burn = 32
    cases = []

    def add(shape, onset, dose, valves):
        x = np.maximum(np.arange(length + 1) - onset, 0)
        live = np.arange(length + 1) >= onset
        if shape == 'step': f = live.astype(float)
        elif shape == 'pulse': f = live & (x < 12)
        elif shape == 'ramp': f = live * np.minimum(x / 16, 1.)
        elif shape == 'sine': f = live * np.sin(2*np.pi*x/32)
        elif shape == 'smooth': f = live * (.6*np.sin(2*np.pi*x/48) + .4*np.sin(2*np.pi*x/19))
        elif shape == 'double_pulse': f = live & ((x < 8) | ((x >= 24) & (x < 32)))
        else: raise ValueError(shape)
        delta = (f[:, None] * np.asarray(valves)[None] * dose).astype(np.float32)
        ident = f'{shape}_at{onset-burn:+d}_dose{dose:+.2f}_v{valves[0]}_{valves[1]}'
        cases.append({'id': ident, 'shape': shape, 'onset': onset, 'relative_onset': onset-burn,
                      'dose': dose, 'valves': list(valves), 'delta': delta})
    for onset in (8, 32, 96, 144):  # pre-window, beginning, middle, end
        for dose in (-.06, -.03, -.01, .01, .03, .06):
            for valve in ((1, 0), (0, 1)):
                add('step', onset, dose, valve)
    for shape in ('pulse', 'ramp', 'sine', 'smooth'):
        for onset in (32, 96, 144):
            for dose in (-.03, .03):
                for valve in ((1, 0), (0, 1)):
                    add(shape, onset, dose, valve)
    for onset in (32, 96, 144):
        for valves in ((1, 1), (1, -1)):
            add('step', onset, .03, valves)
    for valves in ((1, 0), (0, 1)):
        add('double_pulse', 32, .03, valves)
        for onset in (63, 64, 65):
            add('step', onset, .03, valves)
    return cases


def response_summary(delta, case, valid, burn=32):
    """All quantities on model differences, not plant intervention labels."""
    info = {k: v for k, v in case.items() if k != 'delta'}
    info.update(eligible_windows=int(valid.sum()), excluded_out_of_bounds=int((~valid).sum()))
    if not valid.any():
        return {**info, 'status': 'no_eligible_windows'}
    x = delta[valid].astype(np.float64)
    if not np.isfinite(x).all():
        return {**info, 'status': 'nonfinite_response'}
    main, onset = x[:, :, 4], case['onset']
    p = main[:, burn:]
    peak = np.abs(main).max(1)
    threshold = np.maximum(1e-4, .1 * peak)
    crossed = np.abs(main[:, onset:]) >= threshold[:, None]
    latency = [int(np.flatnonzero(row)[0]) * DT if row.any() else None for row in crossed]
    info.update(status='ok', pre_onset_max_abs_C=float(np.abs(x[:, :onset]).max()) if onset else 0.,
                mean_curve_main_C=p.mean(0).tolist(),
                terminal_per_channel_C=x[:, -1].mean(0).tolist(),
                terminal_main_C=float(p[:, -1].mean()), peak_abs_main_C=float(np.abs(p).max(1).mean()),
                mean_abs_main_C=float(np.abs(p).mean()), integral_main_Cs=float(p.sum(1).mean()*DT),
                near_zero_fraction=float((np.abs(p).max(1) <= 1e-4).mean()),
                onset_to_10pct_peak_seconds=latency,
                tail_mean_main_C=float(main[:, -8:].mean()))
    active = np.flatnonzero(np.asarray(case['valves']))
    if len(active) == 1:
        unreachable = [0] if active[0] == 0 else [0, 1, 2]
        info['unreachable_max_abs_C'] = float(np.abs(x[:, :, unreachable]).max())
        if case['shape'] in ('step', 'pulse', 'ramp', 'double_pulse'):
            expected = -np.sign(case['dose'] * case['valves'][active[0]])
            reachable = [1, 2, 3, 4] if active[0] == 0 else [3, 4]
            response = x[:, onset:, reachable]
            info['opposite_sign_fraction'] = float((response * expected < -1e-4).mean())
            info['terminal_gain_C_per_pp'] = float(p[:, -1].mean() / (case['dose']*100))
    if case['shape'] in ('pulse', 'double_pulse'):
        end = onset + (12 if case['shape'] == 'pulse' else 32)
        info['post_release_tail_abs_C'] = float(np.abs(main[:, max(end, main.shape[1]-8):]).mean()) if end < main.shape[1] else None
        info['release_observed'] = end < main.shape[1]
    return info


def response_comparisons(curves):
    """Dose/symmetry/superposition on common eligible windows within each pair."""
    cases = scenarios()[:len(curves['case_ids'])]
    lookup = {(c['shape'], c['onset'], c['dose'], tuple(c['valves'])): i for i, c in enumerate(cases)}
    output = []
    for mode in ('block', 'native'):
        if mode not in curves: continue
        for onset in (32, 96, 144):
            for valve in ((1, 0), (0, 1)):
                keys = [('step', onset, dose, valve) for dose in (.01, .03, .06, -.03)]
                if not all(k in lookup for k in keys): continue
                parts = [curves[mode][lookup[k]] for k in keys]
                valid = np.logical_and.reduce([np.isfinite(x).all((1, 2)) for x in parts])
                if not valid.any(): continue
                small, medium, large, negative = [x[valid, :, 4].astype(float) for x in parts]
                output.append(dict(mode=mode, onset_relative=onset-32, valve=list(valve), common_windows=int(valid.sum()),
                    type='dose_and_symmetry',
                    positive_dose_terminal_order_fraction=float(((small[:, -1] >= medium[:, -1]-1e-4) & (medium[:, -1] >= large[:, -1]-1e-4)).mean()),
                    dose_linearity_mae_C=float(np.abs(large-2*medium).mean()),
                    plus_minus_symmetry_mae_C=float(np.abs(medium+negative).mean())))
            for pair in ((1, 1), (1, -1)):
                keys = [('step', onset, .03, pair), ('step', onset, .03, (1, 0)),
                        ('step', onset, .03 if pair[1] > 0 else -.03, (0, 1))]
                if not all(k in lookup for k in keys): continue
                parts = [curves[mode][lookup[k]] for k in keys]
                valid = np.logical_and.reduce([np.isfinite(x).all((1, 2)) for x in parts])
                if not valid.any(): continue
                both, first, second = [x[valid].astype(float) for x in parts]
                output.append(dict(mode=mode, onset_relative=onset-32, valves=list(pair), common_windows=int(valid.sum()),
                                   type='superposition', interaction_mae_C=float(np.abs(both-first-second).mean())))
    return output


@torch.no_grad()
def evaluate_responses(model, bank, device, windows=8, case_batch=4, limit=None):
    original_dtype = model.mean.dtype
    model.double()
    try:
        return _response_impl(model, bank, device, windows, case_batch, limit)
    finally:
        model.to(dtype=original_dtype)


def _response_impl(model, bank, device, windows, case_batch, limit):
    positions = np.linspace(0, len(bank)-1, min(windows, len(bank)), dtype=int)
    history = unpack(bank[positions], device)[0].double()
    u0 = history[:, -1:, 5:7].expand(-1, 161, -1)
    d0 = history[:, -1:, 7:13].expand(-1, 161, -1)
    cases = scenarios()[:limit]
    modes = ['block'] + (['native'] if model.native_long else [])
    curves, rows = {}, []
    for mode in modes:
        mode_curves = []
        for start in range(0, len(cases), case_batch):
            group = cases[start:start+case_batch]
            perturbation = torch.as_tensor(np.stack([c['delta'] for c in group]), device=device, dtype=history.dtype)
            u = u0[None] + perturbation[:, None]
            valid = ((u >= 0) & (u <= 1)).all((2, 3)).cpu().numpy()
            # Invalid plans are not simulated outside bounds and never scored.
            safe = torch.where(torch.as_tensor(valid, device=device)[:, :, None, None], u, u0[None])
            # Pair both branches in ONE batch: different GEMM batch shapes can
            # otherwise produce apparent ~1 ULP pre-action/upstream effects.
            count = len(group) * len(history)
            paired_u = torch.cat((safe.flatten(0, 1), u0.repeat(len(group), 1, 1)), 0)
            paired = forecast(model, history.repeat(2*len(group), 1, 1), paired_u,
                              d0.repeat(2*len(group), 1, 1), mode=mode)
            delta = (paired[:count] - paired[count:]).reshape(len(group), len(history), 160, 5).cpu().numpy()
            for j, case in enumerate(group):
                row = response_summary(delta[j], case, valid[j])
                row['mode'] = mode
                rows.append(row)
                stored = delta[j].copy()
                stored[~valid[j]] = np.nan
                mode_curves.append(stored)
        curves[mode] = np.stack(mode_curves)
    if model.native_long:
        for i in range(len(cases)):
            a, b = curves['native'][i], curves['block'][i]
            valid = np.isfinite(a).all((1, 2)) & np.isfinite(b).all((1, 2))
            rows[i]['native_block_response_gap_C'] = float(np.abs(a[valid]-b[valid]).mean()) if valid.any() else None
    return rows, {**curves, 'window_positions': positions,
                  'input_delta': np.stack([c['delta'] for c in cases]),
                  'case_ids': np.asarray([c['id'] for c in cases])}


def gradient_probe(model, bank, device):
    h, _, _, _ = unpack(bank[:2], device)
    h = h.to(dtype=model.mean.dtype)
    u = h[:, -1:, 5:7].expand(-1, TRAIN_H+1, -1).clone().requires_grad_(True)
    d = h[:, -1:, 7:13].expand(-1, TRAIN_H+1, -1)
    with torch.enable_grad():
        score = forecast(model, h, u, d)[:, -1, 4].mean()
        grad = torch.autograd.grad(score, u, allow_unused=True)[0] if score.requires_grad else None
    if grad is None:
        return {'gradient_abs_max_C_per_fraction': 0., 'finite_difference_C_per_fraction': 0., 'autograd_C_per_fraction': 0.}
    # A fixed input coordinate, central finite difference in valid support.
    eps = .001
    point = (0, TRAIN_H // 2, 1)
    if not eps <= float(u[point]) <= 1-eps:
        return {'status': 'fixed_coordinate_outside_fd_support'}
    with torch.no_grad():
        plus, minus = u.detach().clone(), u.detach().clone()
        plus[point] += eps
        minus[point] -= eps
        fd = (forecast(model, h, plus, d)[:, -1, 4].mean() - forecast(model, h, minus, d)[:, -1, 4].mean()) / (2*eps)
    return {'gradient_abs_max_C_per_fraction': float(grad.abs().max()),
            'finite_difference_C_per_fraction': float(fd), 'autograd_C_per_fraction': float(grad[point]),
            'absolute_difference': float(abs(fd-grad[point])), 'fd_epsilon_fraction': eps}
