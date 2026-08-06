# Linux 代理 TODO (2026-08-06, 第二轮)

> 第一轮已完成: P0(exp_103) P1(exp_106 7变体+L5) P2(exp_104 DiD n=15) P3A(ff10/ff20) P3MAE n_lag消融 exp_107 exp_111 旧脚本修复
> 完整审查结论见 `docs/session_2026-08-06_full_review.md`
> 关键发现: P2扩样本未跑, P3B未生效, session doc的CFI_agg数字无法溯源, freeze反而恶化gain

---

## S1 — 修 .gitignore 放行 NPZ (立即, 1min)

`.gitignore` 含 `*.npz` 导致 `results/cfe_groundtruth/did_response.npz` 未入库。
在 `*.npz` 行后面加例外:

```
*.npz
!results/cfe_groundtruth/*.npz
!results/cfe_groundtruth_p2/*.npz
```

然后 `git add -f results/cfe_groundtruth/did_response.npz && git commit -m "fix: track DiD groundtruth NPZ"`

---

## S2 — 跑 exp_109 扩样本 DiD 真值 (CPU, ~30min)

**最高优先级。** 这是 S4/S5/S6 的前置条件。

```
python experiments/phase3_feedforward/exp_109_p2_expand.py
```

产出 `results/cfe_groundtruth_p2/did_response.npz` (n_ev≥60)。
跑完 `git add -f results/cfe_groundtruth_p2/did_response.npz && git commit`

---

## S3 — 修 exp_104 JSON 加 r/onsets 字段 (CPU, 5min)

`exp_104_did_groundtruth.py` 的 JSON 输出缺 `r` (逐事件响应) 和 `onsets` 字段，
导致 `exp_106` 的 `eval_causal()` sgn_pair 全部失效 (用 zeros fallback)。

在 `exp_104_did_groundtruth.py` 的 `results['H60']` 和 `results['H18']` 中加入:
```python
'r': did60['r'].tolist(),
'onsets': did60['onsets'].tolist(),
```

重跑 `python experiments/phase3_feedforward/exp_104_did_groundtruth.py` 更新 JSON。

---

## S4 — 跑 exp_110 补基线行 (CPU, ~20min)

S2 完成后:
```
python experiments/phase3_feedforward/exp_110_baselines.py
```

这会用 P2 扩样本真值评测 M5-DSP / M7DSP / M9DSP + 所有 A1phys 变体 + B1glb，
给出有基线参照的完整对比表。`git commit` 结果。

---

## S5 — 修 exp_107 用 cfi_agg + 重跑 (CPU, ~10min)

`exp_107_did_eval.py:111` 仍用旧 `CE.cfi(cfe, '600s')`。改为:
```python
agg = CE.cfi_agg(cfe, PROFILE_K)
cfi = agg['cfi']
```

同时修 line 91 的 H=18 gt key bug: `gt[f'onsets{H // 10}0']` → `gt[f'onsets{H}']`

重跑 `python experiments/phase3_feedforward/exp_107_did_eval.py`，`git commit` 结果。

---

## S6 — P3-C: 对 f_free 屏蔽 SP/ΔSP 输入通道 (GPU, ~2h)

**这是解决欠增益 0.65 的正确方向。** freeze 策略已证明无效 (gain 0.765→0.685)。

在 `causal_arch.py` 的 `ResidualCausalWM` 中，给 `free_head` 的输入去掉 SP 和 ΔSP 列:
- `N_FEAT=40`, `I_SP` 和 `I_DSP=40` 是第 39 和 40 列 (0-indexed: 38, 39)
- `free_x = x[:, :, [i for i in range(N_FEAT) if i not in (I_SP_idx, I_DSP)]]`
- 干预分支 `g(x,a)` 仍用完整输入 (含 SP/ΔSP)

新增变体 `A1phys_spblock` 到 `exp_106` 的 VARIANTS，跑:
```
python experiments/phase3_feedforward/exp_106_causal_arch.py --variant A1phys_spblock --seeds 0,1,2
```

预期: free path 无法从 SP 预测 ΔSP 的效应 → g(x,a) 被迫吸收全部干预信号 → gain ↑

---

## S7 — 多 seed (GPU, ~10h, 可过夜)

论文需要 n≥5 seeds。已有 seed 0，补 1-4:
```
python experiments/phase3_feedforward/exp_106_causal_arch.py --variant A1phys --seeds 1,2,3,4
python experiments/phase3_feedforward/exp_106_causal_arch.py --variant B1glb  --seeds 1,2,3,4
python experiments/phase3_feedforward/exp_106_causal_arch.py --variant B1flat --seeds 1,2,3,4
```

---

## 注意事项

1. **OMP 冲突**: `export KMP_DUPLICATE_LIB_OK=TRUE`
2. **g(x,0)=0 断言**: exp_106 训练前后各做一次
3. **session doc 数字修正**: `docs/session_2026-08-05_causal_arch_eval.md` 中 L6/L7 表格的 CFI_agg/gain_mean 无法溯源, 不要引用
4. **freeze 策略已否**: ff10/ff20 gain 低于 baseline, 不要继续调 freeze epoch 数
5. **MAE loss 已否**: gain 0.619 < NLL 0.765, 不要用 MAE loss
6. **baseline (NLL, no freeze) 是当前最优 gain 配置**
