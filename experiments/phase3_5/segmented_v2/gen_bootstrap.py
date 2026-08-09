#!/usr/bin/env python3
"""bootstrap_summary.json: V1 日块聚类 bootstrap CI 单独落盘 (审计 §9 产物)。"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'results/phase35_segmented_v2/boots‌trap_summary.json'.replace('\u200c', ''))

# 从 leading_summary.json 提取日块 CI (已按 UTC 日聚类 bootstrap 2000 次)
ls = json.load(open(os.path.join(ROOT, 'results/phase35_segmented_v2/leading_summary.json')))
out = {
    'method': 'day-clustered bootstrap, n_boot=2000, alpha=0.05, seed=42',
    'note': 'V0 INCONCLUSIVE → 以下仅 exploratory, 不构成确认证据',
    'by_side': {},
}
for side in ['A', 'B']:
    s = ls['summary'].get(side, {})
    out['by_side'][side] = {
        'n_events': s.get('n'),
        'days': s.get('days'),
        'paired_area_mean': s.get('paired_area_mean'),
        'paired_area_ci95': s.get('paired_area_ci95'),
        'ci_crosses_zero': bool(s.get('paired_area_ci95') and s['paired_area_ci95'][0] < 0 < s['paired_area_ci95'][1]),
    }
out['verdict'] = 'INCONCLUSIVE: 双侧 CI 均含 0, 事件不足(<30/回路) 无统计功效'
with open(os.path.join(ROOT, 'results/phase35_segmented_v2/bootstrap_summary.json'), 'w') as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
