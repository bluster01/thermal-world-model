#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Natural-action event calibration for the bilateral pack (spec §6.1).

Finds isolated single-valve actions in the train split, matches each event with
non-action control windows that are similar *before* the action starts, and reports
local (0-180 s) and downstream (180-1280 s) temperature changes together with the
pre-action balance differences and the match quality.

Nothing here uses model outputs or picks events by model agreement.  All matching
features come from data before the event onset.  Events with insufficient post-action
rows, or with overlapping valve activity, are excluded and counted.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .bilateral_data import ACT_IDX, CONTEXT, NAMES
from .data import sha256

HERE = Path(__file__).resolve().parent
PACK = HERE / 'data/bilateral_AB_33pct.npz'
OUT = HERE.parents[1] / 'results/action_predictor_bench_20260919/bilateral_events_20260922'

VALVES = ['u1A', 'u1B', 'u2A', 'u2B']
LOCAL = {'u1A': (1, 0), 'u1B': (6, 5), 'u2A': (8, 7), 'u2B': (3, 2)}   # (downstream, upstream) per side
DOWN = {'u1A': 4, 'u1B': 9, 'u2A': 9, 'u2B': 4}                        # main outlet of the same side
SIDE = {'u1A': 0, 'u2B': 0, 'u1B': 5, 'u2A': 5}


def valve_track(pack, split='train'):
    """[N, 64 + 128] valve trajectories (history rows, then future control rows)."""
    hist = pack[f'hist30_{split}'][:, :, ACT_IDX]
    fut = pack[f'future_act_{split}'][:, 1:, :]
    return np.concatenate((hist, fut), axis=1)


def detect_events(track, dose=.02, guard=.002, merge=12):
    """Isolated single-valve actions -> {valve: [(window, onset_step), ...]}."""
    delta = np.diff(track, axis=1)
    events = {v: [] for v in VALVES}
    for j, v in enumerate(VALVES):
        move = np.abs(delta[:, :, j]) >= dose
        others = np.delete(np.abs(delta), j, axis=2) <= guard
        flag = move & others.all(axis=2)
        for n in np.flatnonzero(flag.any(axis=1)):
            kept = []
            for t in np.flatnonzero(flag[n]):
                if any(abs(t - prev) <= merge for prev in kept):
                    continue
                kept.append(int(t))
            events[v].extend((int(n), t) for t in kept)
    return events


def observed_temps(pack, split='train'):
    """[N, 64+128] observed temperatures: history rows then recorded future rows."""
    return np.concatenate((pack[f'hist30_{split}'][:, :, :10], pack[f'future_temp_{split}']), axis=1)


def event_rows(pack, split, events, temps, block=32):
    """Attach pre-action matching features (observed data strictly before the action).
    Calibrated events are restricted to history onsets with >= `block` pre-action rows,
    so every event and every control shares the same 32-row descriptor."""
    hist = pack[f'hist30_{split}']
    load = hist[:, :, NAMES.index('load')].mean(1)
    coal = hist[:, :, NAMES.index('coal')].mean(1)
    flow = hist[:, :, NAMES.index('steam_flow')].mean(1)
    rows = []
    for v in VALVES:
        s = SIDE[v]
        for n, t in events[v]:
            if not (block - 1 <= t <= CONTEXT - 1):        # history onsets only, >=32 pre rows
                continue
            pre = temps[n, t - block + 1:t + 1, s:s + 5]
            feats = np.concatenate([hist[n, t, ACT_IDX], pre.mean(0), pre[-1] - pre[0],
                                    [load[n], coal[n], flow[n]]])
            rows.append(dict(valve=v, window=int(n), onset=int(t), block_rows=block, features=feats))
    return rows


def quiet_windows(track, events):
    """Control pool = windows where the *same* event rule triggers nothing.
    (Feedback valves jitter continuously, so 'perfectly static' windows do not exist.)"""
    hit = np.zeros(len(track), dtype=bool)
    for v, pairs in events.items():
        for n, _ in pairs:
            hit[n] = True
    return ~hit


def window_features(hist, n, t, side, stats):
    """Pre-action descriptor: valve values at t, side temps (mean, slope), load/coal/flow means."""
    load, coal, flow = stats
    start = max(0, t - CONTEXT + 1)
    block = hist[n, start:t + 1, :]
    temps = block[:, side:side + 5]
    return np.concatenate([hist[n, t, ACT_IDX], temps.mean(0), temps[-1] - temps[0],
                           [load[n], coal[n], flow[n]]])


def match_controls(pack, split, rows, quiet, k=5, block=32):
    """Nearest control windows per event: standardized pre-action features, controls share
    the same 32-row history descriptor.  Distances use control-pool statistics."""
    hist = pack[f'hist30_{split}']
    stats = (hist[:, :, NAMES.index('load')].mean(1), hist[:, :, NAMES.index('coal')].mean(1),
             hist[:, :, NAMES.index('steam_flow')].mean(1))
    qidx = np.flatnonzero(quiet)
    qfeat = {s: np.stack([np.concatenate([hist[n, CONTEXT - 1, ACT_IDX],
                                          hist[n, -block:, s:s + 5].mean(0),
                                          hist[n, -1, s:s + 5] - hist[n, -block, s:s + 5],
                                          [stats[0][n], stats[1][n], stats[2][n]]]) for n in qidx])
             for s in (0, 5)}
    for r in rows:
        pool = qfeat[SIDE[r['valve']]]
        mu, sd = pool.mean(0), pool.std(0) + 1e-9
        z = (pool - mu) / sd
        zev = (r['features'] - mu) / sd
        d = np.sqrt(((z - zev) ** 2).sum(1))
        order = np.argsort(d)[:k]
        r['controls'] = qidx[order].tolist()
        r['distance'] = float(d[order[0]])
    return rows


def event_responses(pack, split, rows, horizon=128, controls=3):
    temp = np.concatenate((pack[f'hist30_{split}'][:, :, :10], pack[f'future_temp_{split}']), axis=1)
    out = {}
    for v in VALVES:
        curves, ctrl, diag = [], [], []
        for r in [x for x in rows if x['valve'] == v]:
            n, t = r['window'], r['onset']
            post = min(horizon, temp.shape[1] - 1 - t)
            if post < 18:
                continue
            lo, hi, down = LOCAL[v][0], LOCAL[v][1], DOWN[v]
            d = temp[n, t:t + post + 1, :] - temp[n, t, :]
            curves.append(d[:, [lo, hi, down]])
            for c in r['controls'][:controls]:
                dc = temp[c, t:t + post + 1, :] - temp[c, t, :]
                ctrl.append(dc[:, [lo, hi, down]])
            diag.append(dict(window=n, onset=t, post=post,
                             pre_local_gap_C=float(temp[n, t, lo] - temp[n, t, hi]),
                             pre_downstream_C=float(temp[n, t, down]),
                             match_distance=r['distance']))
        if curves:
            L = min(19, min(c.shape[0] for c in curves))          # 0-180 s at 10 s
            out[v] = dict(curves=np.stack([c[:L] for c in curves]),
                          control=np.stack([c[:L] for c in ctrl]) if ctrl else np.zeros((0, L, 3)),
                          diagnostics=diag)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', type=Path, default=PACK)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--dose', type=float, default=.02)
    ap.add_argument('--guard', type=float, default=.002)
    ap.add_argument('--controls', type=int, default=5)
    args = ap.parse_args()
    pack = {k: np.load(args.pack)[k] for k in np.load(args.pack).files}
    track = valve_track(pack, 'train')
    events = detect_events(track, dose=args.dose, guard=args.guard)
    rows = event_rows(pack, 'train', events, observed_temps(pack, 'train'))
    quiet = quiet_windows(track, events)
    rows = match_controls(pack, 'train', rows, quiet, args.controls)
    resp = event_responses(pack, 'train', rows)

    summary = {
        'pack': str(args.pack), 'pack_sha256': sha256(args.pack),
        'rule': dict(dose=args.dose, guard=args.guard, window_steps=6, merge_steps=12),
        'windows': int(len(track)), 'quiet_windows': int(quiet.sum()),
        'events_detected': {v: len(events[v]) for v in VALVES},
        'events_usable': {v: sum(1 for r in rows if r['valve'] == v) for v in VALVES},
        'match_distance_median': float(np.median([r['distance'] for r in rows])) if rows else None,
        'match_pool': 'windows where the same event rule triggers nothing (feedback valves jitter, '
                      'so perfectly static windows do not exist); events restricted to history onsets '
                      'with >=32 pre-action rows so every event/control shares one descriptor',
        'relaxed_sensitivity': 'rerun with --dose 0.01 --guard 0.004 as a separate sensitivity table',
        'local_pairs': {v: list(LOCAL[v]) for v in VALVES},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / 'event_responses.npz',
                        **{f'{v}_{k}': arr for v, d in resp.items() for k, arr in d.items()
                           if isinstance(arr, np.ndarray)})
    (args.out / 'event_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# 自然动作事件（train，未用任何模型输出）', '',
             f"- 规则：目标阀 60s 内变化 ≥{args.dose:.3f}（开度小数），其余三阀同期 ≤{args.guard:.3f}；重叠事件按 {12*10}s 合并",
             f"- 窗口 {summary['windows']}，无动作窗 {summary['quiet_windows']}（匹配池）",
             '', '| 阀 | 检出 | 可用（有 ≥180s 动作后观察） |', '|---|---:|---:|']
    for v in VALVES:
        lines.append(f"| {v} | {summary['events_detected'][v]} | {summary['events_usable'][v]} |")
    lines += ['', f"匹配距离中位数（标准化特征，仅用动作前信息）：{summary['match_distance_median']}", '',
              '近场对 = 同侧 (下游, 上游)；下游 = 同侧末级出口。曲线（事件/控制窗）在 event_responses.npz。',
              '事件数不足的阀按 spec 单列敏感性（--dose 0.01 --guard 0.004），不与主表混合。']
    (args.out / 'event_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
