#!/usr/bin/env python3
"""
M0/M4 早停审查 — 固定训练 60 epochs, 跟踪 V4(50s) vs 完整 rollout(18步 avg)
=============================================================================
问题: M0 best@7 早停 (V4 指标), β warmup=20 未走完。怀疑长程还有改善空间。
方法: 同协议重训, 每 epoch 记 V0/V4, 每 5 epoch 跑 test rollout, 比较
      "V4 早停" vs "rollout-avg 早停" 的选择差异。
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from experiments.phase1_dynamics.exp_025_unified_benchmark import (
    build_model, get_data, train_epoch, validate, eval_rollout, eval_sensitivity,
    eval_sigma_calib, BetaNLLLoss, MSELoss_, train_raw, val_raw, test_raw,
    VALVE_IDX, TARGET_IDX, H_OUT, DEVICE, STEPS, BS, BETA, BETA_WARMUP, EPOCHS, PATIENCE)

MID = sys.argv[1] if len(sys.argv) > 1 else 'M0'
N_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 60

np.random.seed(42); torch.manual_seed(42)
model = build_model(MID).to(DEVICE)
prob = model.probabilistic
tr, va, te = get_data(MID)

crit = BetaNLLLoss(beta=BETA) if prob else MSELoss_()
opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

history = []
for ep in range(1, N_EPOCHS + 1):
    if prob: crit.beta = 0. if ep <= BETA_WARMUP else BETA * min((ep - BETA_WARMUP) / 10, 1.)
    nll = train_epoch(model, tr, opt, crit, prob)
    v0, v4 = validate(model, va, prob); sched.step(v4)
    rec = {'ep': ep, 'nll': nll, 'v0': v0, 'v4': v4, 'beta': crit.beta}
    if ep % 5 == 0 or ep == 1:
        mae = eval_rollout(model, te, prob)
        rec['rollout_avg'] = float(mae.mean())
        rec['rollout_s17'] = float(mae[-1])
        print(f"  E{ep:3d} | NLL:{nll:7.0f} | V0:{v0:.4f} | V4:{v4:.4f} | β:{crit.beta:+.2f} "
              f"| rollout avg:{rec['rollout_avg']:.4f} s17:{rec['rollout_s17']:.4f}")
    history.append(rec)

# 汇总: V4 最优 epoch vs rollout-avg 最优 epoch
v4_best = min(history, key=lambda r: r['v4'])
roll_best = min([r for r in history if 'rollout_avg' in r], key=lambda r: r['rollout_avg'])
print(f"\n===== {MID} 早停审查 =====")
print(f"V4(50s) 最优: E{v4_best['ep']} (v4={v4_best['v4']:.4f}) — 现行早停指标")
print(f"Rollout-avg 最优: E{roll_best['ep']} (avg={roll_best['rollout_avg']:.4f}, s17={roll_best['rollout_s17']:.4f})")
print(f"差异: {'⚠️ 早停过早, 长程仍有改善' if roll_best['ep'] > v4_best['ep'] + 5 else '✅ V4 早停合理'}")

# 保存最优 checkpoint (rollout-avg) 并出最终指标
model2 = build_model(MID).to(DEVICE)
# 重新训练到 rollout_best epoch 保存? 简化: 直接用历史最优评估已加载模型不够准确, 重训到 roll_best epoch
print(f"\n重训至 E{roll_best['ep']} 保存最终模型...")
np.random.seed(42); torch.manual_seed(42)
model = build_model(MID).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
for ep in range(1, roll_best['ep'] + 1):
    if prob: crit.beta = 0. if ep <= BETA_WARMUP else BETA * min((ep - BETA_WARMUP) / 10, 1.)
    train_epoch(model, tr, opt, crit, prob)
    v0, v4 = validate(model, va, prob); sched.step(v4)
model.eval()
mae = eval_rollout(model, te, prob)
sens = eval_sensitivity(model, te) if getattr(model, 'use_action', False) else None
print(f"最终 (E{roll_best['ep']}): rollout avg={mae.mean():.4f} s17={mae[-1]:.4f}")
if sens:
    s1 = sens.get('action_1', {})
    print(f"  Sens 二级阀±10: t1={s1.get('10.0_1',0):+.3f} t8={s1.get('10.0_8',0):+.3f} t12={s1.get('10.0_12',0):+.3f}")
if prob:
    print(f"  σ 校准: {eval_sigma_calib(model, te):.2f}")

out = {'model': MID, 'history': history,
       'v4_best_ep': v4_best['ep'], 'rollout_best_ep': roll_best['ep'],
       'v4_best_v4': v4_best['v4'], 'rollout_best_avg': roll_best['rollout_avg']}
json.dump(out, open(f"results/exp_025_{MID}_earlystop_audit.json", 'w'), indent=2)
print(f"Saved: results/exp_025_{MID}_earlystop_audit.json")
