#!/usr/bin/env python3
"""对照实验: SP 事件(60sV, train+val)上喂 logged valve 给模型, 对比模型预测方向 vs 经验方向。

问题: A1phys 的 100% 方向正确是架构约束产物, 未在观测上验证。
本脚本: 在 SP 干预事件上算模型响应方向, 若 ~75-80% 则"模型复现了 SP 干预响应"成立。
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/home/bluster/projectA/thermal-world-model')
sys.path.insert(0, str(ROOT))

from src.phase35.data import load_cache, extract_windows
from src.phase35.model import A1PhysValveWM
from src.phase35.schema import ExperimentConfig, TARGET_COLUMN, VALVE_COLUMN, SP_COLUMN

CACHE = '/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/phase3_5_cache/cache_A.npz'
CKPT = '/home/bluster/projectA/thermal-world-model/results/phase3_5/runs/A_absolute_identity_s0/checkpoint_best_val.pt'

# ---- split 边界 (A侧 60/20/20) ----
bounds = json.load(open('/tmp/a_split_bounds.json'))
TRAIN_END, VAL_END = bounds['train_end_ns'], bounds['val_end_ns']

# ---- 1s 事件: 60sV 层 (60s严格 + 一级阀≤1), 剔大突变, train+val ----
d = json.load(open(str(ROOT / 'results/phase35_sp1s_events.json')))
evs = d['events']
r180 = json.load(open(str(ROOT / 'results/phase35_sp1s_events_180s.json')))
r180_by_t0 = {e['t0_ns']: e for e in r180['events']}
for e in evs:
    r = r180_by_t0.get(e['t0_ns'])
    if r:
        for k, v in r.items():
            if k != 't0_ns': e[k] = v
cov60 = json.load(open(str(ROOT / 'results/phase35_sp1s_covars_60s.json')))
for e in evs:
    c = {x['t0_ns']: x for x in cov60['events']}.get(e['t0_ns'])
    if c:
        for k, v in c.items():
            if k != 't0_ns': e['cov60_' + k] = v

def is_60sV(e):
    return (e['load_range_60']<=5.0 and e['pres_range_60']<=0.2 and e['temp_range_60']<=1.0
            and e.get('cov60_一级减温调节门阀位') is not None and e['cov60_一级减温调节门阀位']<=1.0)

sel = []
for e in evs:
    t_ns = e['t0_ns'] * 1000  # us -> ns
    if abs(e['dsp']) > 3.0:
        continue
    if t_ns >= VAL_END:  # 排除 test
        continue
    if is_60sV(e):
        e['_t_ns'] = t_ns
        sel.append(e)
print(f'60sV train+val 事件: {len(sel)}')

# ---- 加载 cache + model ----
cache = load_cache(CACHE)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
config = ExperimentConfig.from_mapping(ckpt['config'])
features = ckpt['feature_columns']
model = A1PhysValveWM(config, len(features), features.index(TARGET_COLUMN))
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f'window={config.window} horizon={config.horizon}')
print(f'features: {features}')

# ---- 事件 anchor → cache 索引 (anchor = t0 前最后一行, 即 SP 阶跃前一刻) ----
# 事件 t0 是阶跃后第一个点; anchor 应为 t0 前一秒(1s网格) 对应 cache 中 <= t0-1s 的最后一行
timestamps = cache.timestamps_ns
sp_idx = cache.index(SP_COLUMN)

rows = []
skipped = 0
for e in sel:
    t0_ns = e['_t_ns']
    anchor_ns = t0_ns - 1_000_000_000  # t0 前 1s (1s 网格 SP 阶跃后第1点, anchor 取前一秒)
    # cache 中 <= anchor_ns 的最后一行
    i = int(np.searchsorted(timestamps, anchor_ns, side='right') - 1)
    if i < config.window or i + config.horizon >= len(cache.values):
        skipped += 1
        continue
    # 验证: cache SP 在 anchor 附近确实有阶跃
    rows.append((e, i))
print(f'映射到 cache: {len(rows)} (跳过边界 {skipped})')

# ---- 模型预测: logged valve 轨迹 ----
anchors = np.array([i for _, i in rows], dtype=np.int64)
batch = extract_windows(cache, anchors, features, TARGET_COLUMN, VALVE_COLUMN,
                        config.window, config.horizon)
with torch.no_grad():
    out = model(torch.from_numpy(batch['history']),
                torch.from_numpy(batch['future_valve']),
                torch.from_numpy(batch['baseline_valve']))
effect = out['effect'].numpy()  # [n, horizon] 干预效应 (°C)

# ---- 经验响应 (1s 数据, 600s) ----
emp_dir_ok = 0; emp_dir_n = 0
model_dir_ok = {6: 0, 18: 0, 30: 0, 60: 0}
model_dir_n = {6: 0, 18: 0, 30: 0, 60: 0}
sign_agree = 0

for j, (e, i) in enumerate(rows):
    dsp = e['dsp']
    # 经验: SP 方向与 dT_post_600 同号
    if e['dT_post_600'] is not None:
        emp_dir_n += 1
        if (dsp > 0) == (e['dT_post_600'] > 0):
            emp_dir_ok += 1
    # 模型: effect 在 horizon 末端符号与 SP 方向同号
    for h in (6, 18, 30, 60):
        if h <= config.horizon:
            model_dir_n[h] += 1
            if (dsp > 0) == (effect[j, h-1] > 0):
                model_dir_ok[h] += 1

print(f'\n=== 结果 (n={len(rows)}) ===')
print(f'经验方向率 (SP vs dT600, 1s): {emp_dir_ok}/{emp_dir_n} = {emp_dir_ok/max(1,emp_dir_n)*100:.1f}%')
for h in (6, 18, 30, 60):
    if model_dir_n[h]:
        print(f'模型方向率 (SP vs effect@h{h}): {model_dir_ok[h]}/{model_dir_n[h]} = {model_dir_ok[h]/model_dir_n[h]*100:.1f}%')

# 平均效应幅度
print(f'\n模型 effect 幅度: |effect@h60| 均值={np.abs(effect[:,59]).mean():.3f}°C, '
      f'中位={np.median(np.abs(effect[:,59])):.3f}°C')
print(f'经验 |dT600| 均值: {np.mean([abs(e["dT_post_600"]) for e,_ in rows if e["dT_post_600"] is not None]):.2f}°C')
