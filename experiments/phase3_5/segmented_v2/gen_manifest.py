#!/usr/bin/env python3
"""生成 run_manifest.json (审计 §9): 脚本SHA、CSV SHA256、阈值、切分、环境、命令。"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_sha(path):
    try:
        r = subprocess.run(['git', '-C', path, 'rev-parse', 'HEAD'],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else 'n/a'
    except Exception:
        return 'n/a'


def git_status(path):
    try:
        r = subprocess.run(['git', '-C', path, 'status', '--porcelain'],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception:
        return 'n/a'


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    csv = '/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/cleaned_data/all_merged_10s.csv'
    manifest = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'git_sha': git_sha(root),
        'git_dirty': git_status(root),
        'csv_path': csv,
        'csv_sha256': sha256(csv),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'numpy': __import__('numpy').__version__,
        'pandas': __import__('pandas').__version__,
        'thresholds': {
            'sampling': '10s continuous grid, gaps excluded',
            'load_min_mw': 250,
            'pre_window_s': 600,
            'load_range_mw': 15, 'pressure_range_mpa': 0.4,
            'temp_range_c': 3.0, 'valve_range_pct': 0.3,
            'dose_def': 'median(post 30-60s) - median(pre 60s)',
            'main_threshold_pct': 3.0,
            'sensitivity_pct': [2.0, 5.0],
            'other_valve_limit': 'max(1%, 0.5*|dose|)',
            'min_event_gap_s': 600,
            'held_step': 'post-dose 300s valve range <= max(0.5%, 0.2*|dose|)',
        },
        'splits': {
            'development': '2025-12-24 ~ 2026-03-31',
            'internal_validation': '2026-04-01 ~ 2026-04-30',
            'robustness': '2026-05-01 ~ 2026-05-11',
            'note': '2026-05 previously viewed, robustness only; confirmatory needs new block',
        },
        'commands': [
            'python experiments/phase3_5/segmented_v2/v0_events.py --csv <csv> --out-dir results/phase35_segmented_v2',
            'python experiments/phase3_5/segmented_v2/v1_leading.py --csv <csv> --events results/phase35_segmented_v2/event_manifest.jsonl --out-dir results/phase35_segmented_v2',
            'python experiments/phase3_5/segmented_v2/v23_models.py --csv <csv> --events results/phase35_segmented_v2/event_manifest.jsonl --out-dir results/phase35_segmented_v2',
            'python experiments/phase3_5/segmented_v2/v4_placebo.py --csv <csv> --events results/phase35_segmented_v2/event_manifest.jsonl --out-dir results/phase35_segmented_v2',
        ],
        'verdict': {
            'v0': 'INCONCLUSIVE (A=7 events/5 days, B=6 events/5 days; <30 events/8 days; '
                  '12 open / 1 close → no bidirectional common support; held-step=2)',
            'v1': 'INCONCLUSIVE (paired area CI includes 0: A=[-0.27,+0.35], B=[-8.35,+0.74])',
            'v2': 'INCONCLUSIVE (n too small; A b_coef noisy, B cross direction consistent but n=6)',
            'v3': 'NOT PASSED (same-side coefficient not dominant: b_oth > b_same in all blocks; '
                  'collinear dual-side inputs; R2=1.0 autocorrelation-dominated)',
            'v4': 'NOT PASSED (B wrong-side > true; strata close-cells empty)',
            'conclusion': '85%/74% 与 0s 峰值仅 exploratory pilot; 交叉拓扑物理结论 NOT VERIFIED; '
                          'E3 保持 INCONCLUSIVE; E4 保持 BLOCKED',
        },
    }
    out = os.path.join(root, 'results/phase35_segmented_v2/run_manifest.json')
    with open(out, 'w') as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
