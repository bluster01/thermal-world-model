#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FMTS 补做探针：黑箱「未来喷水通道」消融（execution-side，3 seeds）。

问题（防守卡素材）：论文附录已声明 "Future total spray is visible to the black box
but ignored by action-driven physics"，且 spray_flow_total 标注
"unreliable; oracle diagnostics only"。本探针量化：把黑箱可见的未来喷水通道
（BOUNDARY_ELEMENTS[6]）屏蔽（置为该窗口历史末端保持值）后，H18 平均 MAE 的变化。

三组（全部在原始 256 验证窗 + 原始 20k 训练 bank 上，seeds 0/1/2）：
  A. control      — 原样重训（复现门：与 linux_full_v02 原报告对比）
  B. masked_train — 训练+验证/评估均屏蔽未来喷水（部署一致口径）
  C. masked_eval  — 仅对原 checkpoints 评估时屏蔽（不加训练，看已训模型的通道依赖）

不改 experiments/、不动冻结 checkpoint、不写 verdict。仅新增本目录产物。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.fmts_mainsteam_20260911.data import RichRecord  # noqa: E402
from experiments.fmts_mainsteam_20260911.models import RichBlackbox  # noqa: E402
from experiments.fmts_mainsteam_20260911.run import cumulative, day_mean  # noqa: E402
from src.final_wm.contracts import BOUNDARY_ELEMENTS  # noqa: E402

RECORD = Path('/home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz')
MAPPING = ROOT / 'configs/final_wm/channel_mapping_v2.json'
ORIG = ROOT / 'results/fmts_mainsteam_20260911/linux_full_v02'
INDICES = ORIG / 'indices.npz'
OUT = ROOT / 'results/fmts_spray_ablation_20260916/full'
SMOKE_OUT = ROOT / 'results/fmts_spray_ablation_20260916/smoke'
EXPECTED_RECORD_SHA = '24da77960e05e3636cc7b97a60a75e9b4ba470a3abb4f8ba920ddf11c6dad1d0'
SPRAY = BOUNDARY_ELEMENTS.index('spray_flow_total')  # 6

CKPT = 'best.pt'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False) + '\n',
                    encoding='utf-8')


class AblationRun:
    def __init__(self, args):
        self.device = args.device
        self.smoke = args.smoke
        self.record = RichRecord(str(RECORD), str(MAPPING))
        arr = np.load(INDICES)
        self.train_starts = torch.from_numpy(arr['train']).long()
        self.val_starts = torch.from_numpy(arr['validation']).long()
        self.mean, self.std = self.record.normalization()

    # ---------- batch with optional spray mask ----------
    def make_batch(self, starts: torch.Tensor, split: int, ablate: bool):
        h, ext, a, b, y, days = self.record.batch(starts, split, self.device)
        if ablate:
            hold = self.record.boundary[(starts - 1).cpu()][:, SPRAY].to(self.device)
            b = b.clone()
            b[:, :, SPRAY] = hold[:, None]
        return h, ext, a, b, y, days

    # ---------- evaluation identical in metric to run.py ----------
    @torch.no_grad()
    def evaluate(self, model, starts: torch.Tensor, split: int, ablate: bool,
                 batch_size: int = 32):
        model.eval()
        preds, errs, days_all = [], [], []
        for i in range(0, len(starts), batch_size):
            idx = starts[i:i + batch_size]
            h, ext, a, b, y, days = self.make_batch(idx, split, ablate)
            pred = model(h, ext, a, b)
            if not torch.isfinite(pred).all():
                raise RuntimeError('nonfinite prediction')
            preds.append(pred.detach().cpu().numpy())
            errs.append((pred - y[:, :, 4]).abs().detach().cpu().numpy())
            days_all.append(days.cpu().numpy() if hasattr(days, 'cpu') else np.asarray(days))
        err = np.concatenate(errs)
        days = np.concatenate(days_all)
        curve = day_mean(cumulative(err), days)
        return {'pred': np.concatenate(preds), 'err': err, 'days': days, 'curve': curve,
                'window_mae': err.mean(1),
                'H6': float(curve[5]), 'H12': float(curve[11]), 'H18': float(curve[-1])}

    # ---------- training loop copied from run.py (blackbox branch) ----------
    def train(self, seed: int, ablate: bool, out_dir: Path, *, cap: int, every: int,
              patience: int, batch: int, train_starts=None):
        train_starts = self.train_starts if train_starts is None else train_starts
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = RichBlackbox(self.mean, self.std).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        generator = torch.Generator().manual_seed(20000 + seed)
        best, stale, best_step = float('inf'), 0, -1
        started = time.time()
        out_dir.mkdir(parents=True, exist_ok=False)
        with (out_dir / 'ledger.jsonl').open('w', encoding='utf-8') as ledger:
            step = 0
            for step in range(1, cap + 1):
                model.train()
                selection = torch.randint(len(train_starts), (batch,), generator=generator)
                h, ext, a, b, y, _ = self.make_batch(train_starts[selection], 0, ablate)
                result = model(h, ext, a, b)
                loss = ((result - y[:, :, 4]) ** 2).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite training loss')
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                if step % every == 0:
                    raw = self.evaluate(model, self.val_starts, 1, ablate)
                    score = float(day_mean(raw['window_mae'], raw['days']))
                    ledger.write(json.dumps({'step': step, 'loss': float(loss.detach()),
                                             'validation_mainsteam_mae': score}) + '\n')
                    ledger.flush()
                    if score < best:
                        best, stale, best_step = score, 0, step
                        torch.save({'state_dict': model.state_dict(), 'arm': 'blackbox_itransformer',
                                    'seed': seed, 'ablate': ablate, 'step': step,
                                    'normalization': {'mean': self.mean.tolist(),
                                                      'std': self.std.tolist()}},
                                   out_dir / CKPT)
                    else:
                        stale += 1
                    if stale >= patience:
                        break
        ck = torch.load(out_dir / CKPT, map_location=self.device, weights_only=False)
        model.load_state_dict(ck['state_dict'])
        return {'model': model, 'updates': step, 'best_step': best_step,
                'seconds': time.time() - started,
                'param_count': sum(p.numel() for p in model.parameters())}


def replay_check(run: AblationRun, seed: int, tolerance=2e-4):
    """原 checkpoint 重放（不屏蔽）：与 linux_full_v02 存储预测逐点对比（复现门）。"""
    ck = torch.load(ORIG / f'blackbox_itransformer_seed{seed}' / CKPT,
                    map_location=run.device, weights_only=False)
    model = RichBlackbox(run.mean, run.std).to(run.device)
    model.load_state_dict(ck['state_dict'])
    raw = run.evaluate(model, run.val_starts, 1, ablate=False)
    stored = np.load(ORIG / f'blackbox_itransformer_seed{seed}' / 'prediction.npz')
    maxdiff = float(np.abs(raw['pred'] - stored['prediction']).max())
    report = json.loads((ORIG / f'blackbox_itransformer_seed{seed}' / 'report.json').read_text())
    return {'seed': seed, 'max_abs_pred_diff': maxdiff, 'gate_pass': maxdiff < tolerance,
            'H18_replay': raw['H18'], 'H18_stored': report['H18'],
            'H18_diff': raw['H18'] - report['H18']}


def paired_delta(days, a_mae, b_mae, rng_seed=11000):
    """Δ = b − a（按天 block bootstrap CI95）。"""
    diff = b_mae - a_mae
    blocks = np.array([diff[days == d].mean() for d in np.unique(days)])
    rng = np.random.default_rng(rng_seed)
    boot = blocks[rng.integers(len(blocks), size=(1000, len(blocks)))].mean(1)
    return {'mean': float(blocks.mean()), 'ci95': [float(np.quantile(boot, .025)),
                                                   float(np.quantile(boot, .975))],
            'days': int(len(blocks))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seeds', default='0,1,2')
    args = ap.parse_args()

    # input identity gates
    rec_sha = sha256(RECORD)
    if rec_sha != EXPECTED_RECORD_SHA:
        sys.exit(f'FATAL: record sha mismatch {rec_sha}')
    ident = json.loads((ORIG / 'identity.json').read_text())
    idx_sha = sha256(INDICES)
    if idx_sha != ident['indices']:
        sys.exit('FATAL: indices.npz does not match linux_full_v02 identity')

    run = AblationRun(args)
    seeds = [int(s) for s in args.seeds.split(',')]
    print(f'[setup] device={args.device} seeds={seeds} spray_idx={SPRAY} '
          f'train_windows={len(run.train_starts)} val_windows={len(run.val_starts)}', flush=True)

    if args.smoke:
        if SMOKE_OUT.exists():
            shutil.rmtree(SMOKE_OUT)
        SMOKE_OUT.mkdir(parents=True)
        out = {}
        rep = replay_check(run, 0)   # 全 256 验证窗，必须在截断前跑
        out['replay'] = rep
        print('[smoke] replay gate:', rep['gate_pass'], 'maxdiff=%.2e' % rep['max_abs_pred_diff'], flush=True)
        sm_train = run.train_starts[:512]
        run.val_starts = run.val_starts[:32]
        t = run.train(0, False, SMOKE_OUT / 'control_seed0', cap=6, every=2, patience=2,
                      batch=16, train_starts=sm_train)
        ev = run.evaluate(t['model'], run.val_starts, 1, ablate=False)
        out['control_seed0'] = {'updates': t['updates'], 'H18': ev['H18']}
        t = run.train(1, True, SMOKE_OUT / 'masked_train_seed1', cap=6, every=2, patience=2,
                      batch=16, train_starts=sm_train)
        ev2 = run.evaluate(t['model'], run.val_starts, 1, ablate=True)
        out['masked_seed1'] = {'updates': t['updates'], 'H18': ev2['H18']}
        write_json(SMOKE_OUT / 'smoke_summary.json', out)
        print('[smoke] OK ->', SMOKE_OUT / 'smoke_summary.json', flush=True)
        return

    if OUT.exists() and any(OUT.iterdir()) and not args.force:
        sys.exit(f'FATAL: {OUT} not empty; pass --force to replace')
    if OUT.exists() and args.force:
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    t0 = time.time()
    summary = {'protocol': 'FMTS-SPRAY-ABL', 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
               'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                          text=True, cwd=ROOT).stdout.strip(),
               'torch': torch.__version__, 'device': args.device,
               'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
               'cuda_tf32': torch.backends.cuda.matmul.allow_tf32,
               'spray_index': SPRAY, 'spray_element': BOUNDARY_ELEMENTS[SPRAY],
               'mask_rule': 'future boundary[:, :, 6] held at the window history-end value',
               'record_sha256': rec_sha, 'indices_sha256': idx_sha,
               'seeds': seeds, 'arms': {}, 'harness': {}}

    # -------- harness: replay original checkpoints (gate) --------
    print('[harness] replay original checkpoints...', flush=True)
    replays = [replay_check(run, s) for s in seeds]
    summary['harness']['replay'] = replays
    ok = all(r['gate_pass'] for r in replays)
    print('[harness] replay gate', 'PASS' if ok else 'FAIL',
          [f"{r['seed']}:{r['max_abs_pred_diff']:.1e}" for r in replays], flush=True)
    if not ok:
        sys.exit('FATAL: replay gate failed; environment does not reproduce stored predictions')

    # -------- Arm C: masked eval on original checkpoints --------
    print('[arm C] masked-eval on original checkpoints...', flush=True)
    armC = {}
    for s in seeds:
        ck = torch.load(ORIG / f'blackbox_itransformer_seed{s}' / CKPT,
                        map_location=run.device, weights_only=False)
        model = RichBlackbox(run.mean, run.std).to(run.device)
        model.load_state_dict(ck['state_dict'])
        raw = run.evaluate(model, run.val_starts, 1, ablate=True)
        armC[f'seed{s}'] = {'H6': raw['H6'], 'H12': raw['H12'], 'H18': raw['H18']}
        np.savez_compressed(OUT / f'masked_eval_seed{s}.npz',
                            err=raw['err'].astype(np.float32), days=raw['days'])
        print(f"  seed{s}: H18={raw['H18']:.4f}", flush=True)
    summary['arms']['masked_eval'] = armC

    # -------- Arm A: control retrain --------
    print('[arm A] control retrain...', flush=True)
    armA = {}
    for s in seeds:
        t = run.train(s, False, OUT / f'control_seed{s}', cap=3000, every=500, patience=3,
                      batch=128)
        raw = run.evaluate(t['model'], run.val_starts, 1, ablate=False)
        armA[f'seed{s}'] = {'H6': raw['H6'], 'H12': raw['H12'], 'H18': raw['H18'],
                            'updates': t['updates'], 'seconds': round(t['seconds'], 1),
                            'param_count': t['param_count']}
        np.savez_compressed(OUT / f'control_seed{s}.npz',
                            err=raw['err'].astype(np.float32), days=raw['days'])
        print(f"  seed{s}: H18={raw['H18']:.4f} ({t['seconds']:.0f}s)", flush=True)
    summary['arms']['control'] = armA

    # -------- Arm B: masked retrain --------
    print('[arm B] masked retrain...', flush=True)
    armB = {}
    for s in seeds:
        t = run.train(s, True, OUT / f'masked_train_seed{s}', cap=3000, every=500, patience=3,
                      batch=128)
        raw = run.evaluate(t['model'], run.val_starts, 1, ablate=True)
        armB[f'seed{s}'] = {'H6': raw['H6'], 'H12': raw['H12'], 'H18': raw['H18'],
                            'updates': t['updates'], 'seconds': round(t['seconds'], 1),
                            'param_count': t['param_count']}
        np.savez_compressed(OUT / f'masked_train_seed{s}.npz',
                            err=raw['err'].astype(np.float32), days=raw['days'])
        print(f"  seed{s}: H18={raw['H18']:.4f} ({t['seconds']:.0f}s)", flush=True)
    summary['arms']['masked_train'] = armB

    # -------- paired deltas --------
    def load_mae(tag, s):
        z = np.load(OUT / f'{tag}_seed{s}.npz')
        return z['err'].mean(1), z['days']
    compar = {}
    for name, tagb in (('B_minus_A', 'masked_train'), ('C_minus_A', 'masked_eval')):
        per = {}
        for s in seeds:
            mA, d = load_mae('control', s)
            mB, _ = load_mae(tagb, s)
            per[f'seed{s}'] = paired_delta(d, mA, mB)
        per['H18_delta_mean'] = float(np.mean(
            [summary['arms'][tagb][f'seed{s}']['H18'] - armA[f'seed{s}']['H18'] for s in seeds]))
        compar[name] = per
    summary['comparisons'] = compar
    summary['H18_summary'] = {
        lab: {'mean': float(np.mean([arm[f"seed{s}"]["H18"] for s in seeds])),
              'std': float(np.std([arm[f"seed{s}"]["H18"] for s in seeds], ddof=1))}
        for lab, arm in (('control', armA), ('masked_train', armB), ('masked_eval', armC))}
    summary['reference'] = {'gru_hybrid_H18': 0.4431, 'blackbox_orig_H18': 0.3706}
    summary['seconds_total'] = round(time.time() - t0, 1)

    write_json(OUT / 'summary.json', summary)

    # identity for the probe run
    write_json(OUT / 'probe_identity.json', {
        'git_head': summary['git_head'], 'command': ' '.join(sys.argv),
        'python': platform.python_version(), 'torch': torch.__version__,
        'device': args.device, 'record_sha256': rec_sha, 'indices_sha256': idx_sha,
        'orig_identity_sha256': sha256(ORIG / 'identity.json'),
        'ckpt_sha256': {f'seed{s}': sha256(ORIG / f'blackbox_itransformer_seed{s}' / CKPT)
                        for s in seeds}})

    # ---- auto-report ----
    def fmt(x): return f'{x:.4f}'
    rows = []
    for s in seeds:
        rows.append((s, armA[f'seed{s}']['H18'], armB[f'seed{s}']['H18'],
                     armC[f'seed{s}']['H18'], compar['B_minus_A'][f'seed{s}'],
                     compar['C_minus_A'][f'seed{s}']))
    L = []
    L.append('# FMTS 补做探针：黑箱「未来喷水通道」消融（execution-side）\n')
    L.append(f"- 日期：{summary['created']}；git `{summary['git_head'][:10]}`；"
             f"device `{args.device}`；torch `{torch.__version__}`\n"
             f"- 输入身份：record sha 已验证；indices 与 linux_full_v02 identity 一致\n"
             f"- 屏蔽规则：未来 boundary 第 7 通道（spray_flow_total）置为窗口历史末端值；"
             f"训练与评估同口径（masked_train 组）；masked_eval 组仅评估屏蔽、不重训\n")
    L.append('## 复现门\n')
    for r in replays:
        L.append(f"- seed{r['seed']}: 重放预测 vs 存储预测 max|Δ|={r['max_abs_pred_diff']:.2e}"
                 f"（H18 {r['H18_replay']:.4f} vs {r['H18_stored']:.4f}）")
    L.append('\n## 结果（H18 平均 MAE，验证段 256 窗 / 13 天，°C）\n')
    L.append('| seed | A control（原样重训） | B masked_train（重训+屏蔽） | C masked_eval（仅屏蔽评估） | ΔB−A [CI95] | ΔC−A [CI95] |')
    L.append('|---|---|---|---|---|---|')
    for s, ha, hb, hc, db, dc in rows:
        L.append(f"| {s} | {fmt(ha)} | {fmt(hb)} | {fmt(hc)} | "
                 f"{db['mean']:+.4f} [{db['ci95'][0]:+.4f},{db['ci95'][1]:+.4f}] | "
                 f"{dc['mean']:+.4f} [{dc['ci95'][0]:+.4f},{dc['ci95'][1]:+.4f}] |")
    hs = summary['H18_summary']
    L.append(f"| **均值±std** | **{hs['control']['mean']:.4f}±{hs['control']['std']:.4f}** | "
             f"**{hs['masked_train']['mean']:.4f}±{hs['masked_train']['std']:.4f}** | "
             f"**{hs['masked_eval']['mean']:.4f}±{hs['masked_eval']['std']:.4f}** | "
             f"{compar['B_minus_A']['H18_delta_mean']:+.4f} | {compar['C_minus_A']['H18_delta_mean']:+.4f} |")
    L.append(f"\n参考：GRU 融合 H18 = 0.4431；黑箱原报告 H18 = 0.3706。\n")
    L.append('## 解读与限定\n')
    L.append('- 本探针为 execution-side 证据包；不改论文、不写 verdict，供防守卡选用。\n'
             '- Δ>0 = 屏蔽后误差变大 = 原精度对「未来喷水可见性」有依赖；Δ≈0 = 该通道对黑箱精度贡献可忽略。\n'
             '- masked_train 组为部署一致口径（训练/评估都看不到未来喷水）；masked_eval 组反映已训模型的即时依赖，'
             '输入分布外故仅作参考。\n'
             '- 仅屏蔽未来窗口的喷水；历史段喷水保持可见（历史数据部署可得）。\n'
             f"- 总耗时 {summary['seconds_total']}s。\n")
    (OUT / 'REPORT_ZH.md').write_text('\n'.join(L), encoding='utf-8')

    print('[done] mean H18: control=%.4f  masked_train=%.4f  masked_eval=%.4f' %
          (hs['control']['mean'], hs['masked_train']['mean'], hs['masked_eval']['mean']), flush=True)
    print('[done] wrote', OUT / 'summary.json', 'and REPORT_ZH.md', flush=True)


if __name__ == '__main__':
    main()
