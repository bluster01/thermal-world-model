"""Pure perturbations and descriptive statistics; no fitting or selection."""
import json
from pathlib import Path

import numpy as np
import torch

from src.final_wm.model import HistoryWindow
from experiments.fmts_mainsteam_20260911.run import day_mean

PACKAGE = Path(__file__).resolve().parent
P = json.loads((PACKAGE / 'protocol.json').read_text(encoding='utf-8'))


def scenarios():
    result = []
    for valve in (0, 1):
        for lag in P['history_block_end_lag_steps']:
            for dose in P['history_valve_doses']:
                result.append(dict(family='history', valve=valve, lag=lag, dose=dose))
        for onset in P['future_onset_steps']:
            for shape in ('pulse', 'step'):
                for dose in P['future_valve_doses']:
                    result.append(dict(family='future', valve=valve, onset=onset, shape=shape, dose=dose))
    for dose in P['spray_doses_tph']:
        result.append(dict(family='spray', onset=0, spray_dose=dose))
    for valve in (0, 1):
        for dose in P['history_valve_doses']:
            result.append(dict(family='joint', valve=valve, onset=0, shape='step', dose=dose,
                               spray_dose=float(np.sign(dose) * 2.)))
    assert len(result) == P['scenarios_per_checkpoint']
    return [dict(id=f's{i:03d}', **s) for i, s in enumerate(result)]


def perturb(h, actions, boundary, scenario):
    """Never rebuild future hold-last from modified history; inputs stay intact."""
    ha, a, b = h.actions.clone(), actions.clone(), boundary.clone()
    history = scenario['family'] == 'history'
    dose = torch.zeros_like(ha[:, :, 0] if history else a[:, :, 0])
    water_dose = torch.zeros_like(b[:, :, 0])
    if 'valve' in scenario:
        j = scenario['valve']
        if history:
            end = P['history_steps'] - scenario['lag']
            start = end - P['history_block_steps']
            old = ha[:, start:end, j].clone()
            ha[:, start:end, j] = (old + scenario['dose']).clamp(0., 1.)
            dose[:, start:end] = ha[:, start:end, j] - old
        else:
            start = scenario['onset']
            end = start + P['pulse_steps'] if scenario['shape'] == 'pulse' else P['horizon_steps']
            old = a[:, start:end, j].clone()
            a[:, start:end, j] = (old + scenario['dose']).clamp(0., 1.)
            dose[:, start:end] = a[:, start:end, j] - old
    if 'spray_dose' in scenario:
        j = P['spray_column']
        old = b[:, :, j].clone()
        b[:, :, j] = (old + scenario['spray_dose']).clamp(*P['spray_bounds_tph'])
        water_dose = b[:, :, j] - old
    return HistoryWindow(h.obs, ha, h.boundary), a, b, dose, water_dose


def opening_bins(positions, thresholds):
    positions, thresholds = np.asarray(positions), np.asarray(thresholds)
    assert positions.ndim == 2 and positions.shape[1] == 2 and thresholds.shape == (2, 2)
    return np.stack([np.digitize(positions[:, j], thresholds[j], right=False) for j in (0, 1)], axis=1)


def summarize_effect(delta, days, scenario):
    delta = np.asarray(delta, dtype=np.float64)
    days = np.asarray(days)
    assert delta.shape == (len(days), 18) and np.isfinite(delta).all()
    out = dict(n_windows=len(days), n_days=len(np.unique(days)))
    if not len(days):
        return out
    out.update(signed_curve_c=day_mean(delta, days).tolist(),
               absolute_curve_c=day_mean(np.abs(delta), days).tolist(),
               negative_fraction_curve=day_mean(delta < 0, days).tolist(),
               peak_abs_c=float(np.abs(delta).max()))
    onset = scenario.get('onset')
    out['pre_action_max_abs_c'] = float(np.abs(delta[:, :onset]).max()) if onset else None
    out['pre_action_mean_abs_c'] = float(day_mean(np.abs(delta[:, :onset]), days).mean()) if onset else None
    out['elapsed60_signed_c'] = float(day_mean(delta[:, onset+5], days)) if onset is not None else None
    out['elapsed60_absolute_c'] = float(day_mean(np.abs(delta[:, onset+5]), days)) if onset is not None else None
    return out


def factual_metrics(prediction, target, anchor, days):
    prediction, target = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    days, anchor = np.asarray(days), np.asarray(anchor, dtype=float)[:, None]
    out = dict(n_windows=len(days), n_days=len(np.unique(days)))
    if not len(days):
        return out
    assert prediction.shape == target.shape == (len(days), 18)
    dp = np.diff(np.concatenate([anchor, prediction], axis=1), axis=1)
    dt = np.diff(np.concatenate([anchor, target], axis=1), axis=1)
    correlation = float(np.corrcoef(dp.ravel(), dt.ravel())[0, 1]) if dp.std() > 1e-12 and dt.std() > 1e-12 else None
    out.update(level_mae_c=float(day_mean(np.abs(prediction-target).mean(1), days)),
               increment_mae_c=float(day_mean(np.abs(dp-dt).mean(1), days)),
               increment_correlation=correlation,
               correlation_weighting='pooled descriptive samples, not independent observations')
    return out


def scenario_summary(raw, context, scenario):
    delta = raw['prediction'].astype(float) - raw['base'].astype(float)
    days = context['days']
    out = summarize_effect(delta, days, scenario)
    out['unsupported_windows'] = int((~raw['support'].all(1)).sum())
    out['support_interpretation'] = 'marginal historical range sanity check, not conditional causal support'
    out['max_absolute_valve_dose_fraction'] = float(np.abs(raw['valve_dose']).max())
    out['max_absolute_spray_dose_tph'] = float(np.abs(raw['spray_dose']).max())
    if 'valve' in scenario:
        end = 96-scenario['lag'] if scenario['family'] == 'history' else (scenario['onset']+3 if scenario['shape'] == 'pulse' else 18)
        start = end-3 if scenario['family'] == 'history' else scenario['onset']
        actual = raw['valve_dose'][:, start:end]
        out['valve_dose_accounting'] = dict(
            requested_fraction=scenario['dose'],
            actual_mean_fraction=float(actual.mean()), actual_min_fraction=float(actual.min()),
            actual_max_fraction=float(actual.max()),
            clipped_windows=int((~np.isclose(actual, scenario['dose'], rtol=0, atol=1e-7)).any(1).sum()),
            zero_dose_windows=int((actual == 0).all(1).sum()))
    if 'spray_dose' in scenario:
        actual = raw['spray_dose']
        out['spray_dose_accounting'] = dict(
            requested_tph=scenario['spray_dose'], actual_mean_tph=float(actual.mean()),
            actual_min_tph=float(actual.min()), actual_max_tph=float(actual.max()),
            clipped_windows=int((~np.isclose(actual, scenario['spray_dose'], rtol=0, atol=1e-6)).any(1).sum()))
    # Actual per-window dose arrays are authoritative near clipping boundaries.
    if 'valve' in scenario:
        bins = context['opening_bin'][:, scenario['valve']]
        out['opening_strata'] = {name: summarize_effect(delta[bins == i], days[bins == i], scenario)
                                 for i, name in enumerate(('low', 'mid', 'high'))}
    return out


def opening_coverage(context):
    result = {}
    for j in (0, 1):
        positions, bins = context['history_actions'][:, -1, j], context['opening_bin'][:, j]
        result[str(j+1)] = dict(min_fraction=float(positions.min()), median_fraction=float(np.median(positions)),
            max_fraction=float(positions.max()), strata={name: dict(n_windows=int((bins == i).sum()),
            n_days=len(np.unique(context['days'][bins == i]))) for i, name in enumerate(('low', 'mid', 'high'))})
    return result


def factual_summary(raw, context):
    pred, truth, anchor, days = raw['prediction'], raw['target'], raw['persistence'][:, 0], raw['days']
    result = dict(all=factual_metrics(pred, truth, anchor, days), opening={}, movement={})
    excursion = np.abs(context['future_actions'] - context['history_actions'][:, -1:, :]).max(1)
    for j in (0, 1):
        result['opening'][str(j+1)] = {}
        result['movement'][str(j+1)] = {}
        movement = np.digitize(excursion[:, j], P['factual_movement_edges_fraction'])
        for dest, bins, labels in [(result['opening'][str(j+1)], context['opening_bin'][:, j], ('low', 'mid', 'high')),
                                   (result['movement'][str(j+1)], movement, ('lt2pp', '2to5pp', 'ge5pp'))]:
            for i, name in enumerate(labels):
                mask = bins == i
                dest[name] = factual_metrics(pred[mask], truth[mask], anchor[mask], days[mask])
    return result
