# 完整代码与结果审查 (2026-08-06)

> 审查范围: commit 19156f7..23e3908 全部新增/修改文件 + result.json 交叉验证

---

## 一、执行状态总览

| 计划项 | 状态 | 说明 |
|---|---|---|
| P0 exp_103 recheck | ✅ 完成 | n=134, 5 模型, corrected action encoding |
| P1 exp_106 全 7 变体 | ✅ 完成 | seed 0, 含 L5 H/weight 解耦 |
| P2 exp_104 DiD 真值 | ✅ 完成 | n_ev=15, JSON+NPZ 均产出 (NPZ 被 .gitignore 排除) |
| P2 扩样本 exp_109 | ❌ 未跑 | `results/cfe_groundtruth_p2/` 不存在 |
| P3A freeze-free | ✅ 完成 | ff10, ff20 两个配置 |
| P3B gain calibration | ⚠️ 未生效 | P2 npz 不存在 → `lambda_gain` 被 fallback 为 0 |
| P3 MAE loss | ✅ 完成 | `A1phys_s0_mae` |
| P4 baselines exp_110 | ❌ 未跑 | 依赖 P2 npz |
| exp_107 DiD eval | ✅ 完成 | 但仍用旧 `cfi()` 非 `cfi_agg()` |
| exp_108 L5 eval | ❌ 未跑 | 依赖 P2 npz (line 30) |
| exp_111 case 图 | ✅ 完成 | 5 模型 × 3 事件 |
| n_lag=3 消融 | ✅ 完成 | `A1phys_s0_ff10_nl3` |
| 旧脚本修复 (9 个) | ✅ 完成 | commit fea922f |

---

## 二、数据完整性问题

### 2.1 NPZ 被 .gitignore 排除

`.gitignore` 含 `*.npz`，导致 `results/cfe_groundtruth/did_response.npz` 虽在 Linux 端生成但未入库。

**影响**:
- `exp_107` 在 Linux 端成功运行（NPZ 本地存在），但在 Windows 端无法复现
- `exp_108` 和 `exp_110` 依赖 P2 npz，从未运行
- `exp_106` P3B gain calibration loss 依赖 P2 npz，`lambda_gain` 被 fallback 为 0

**修复**: `.gitignore` 应对 `results/cfe_groundtruth/` 和 `results/cfe_groundtruth_p2/` 例外放行，或改用 JSON 存储逐事件数据。

### 2.2 exp_106 `eval_causal()` 的 sgn_pair 全部失效

`exp_106_causal_arch.py:139-143`:
```python
gtk = gt.get(f'H{H}', gt)   # exp_104 JSON: {H60: {...}, H18: {...}}
r = np.array(gtk.get('r', np.zeros((len(m), H))), dtype=np.float32)
if len(r) == len(m):                        # 事件集一致才配对
    out['cfe'] = CE.causal_metrics(m, r, R, ceil, ks)
```

`did_response.json` 的 `H60` 子对象**不含 `r` 字段**（只有 `R_true`, `ci_lo`, `ci_hi`, `sgn_ceiling`, `gain_ceiling`, `noise_floor`, `profile`）。`r` 在 NPZ 里但不在 JSON 里。

结果: `r = zeros((16, 60))`，`sgn_pair = (sign(m) == sign(0)).mean()` → 对所有 k 值，sign(0)=0，所以 sgn_pair ≈ 0.5（取决于 m 的正负分布）。`sgn_norm = (0.5-0.5)/(ceil-0.5) = 0/...` → sgn_norm ≈ 0 或 NaN。

**但 gain 和 shape_corr 仍然有效**：它们用 `mbar` (模型响应均值) vs `R_true` (DiD 真值均值)，不依赖 `r`。

**结论**: exp_106 训练期 CFI 值不可信（含 sgn 项），但 gain/shape/ttp 指标可信。exp_107 的事件对齐评测（用 NPZ 的 `r` 字段）是唯一可信的 sgn 评测。

### 2.3 exp_107 仍用旧 `cfi()` 而非 `cfi_agg()`

`exp_107_did_eval.py:111`:
```python
cfi = CE.cfi(cfe, '600s')
```

`cfi_agg()` 已添加到 `causal_eval.py:315`，但 exp_107 从未调用它。所有 `did_eval.json` 中的 CFI 值仍是 600s 单点 CFI。

session doc 声称的 "CFI_agg=0.833" 等数字**不存在于任何 result 文件中**，来源不明。

---

## 三、session_2026-08-05_causal_arch_eval.md 数字校验

### L4 表格 (exp_107 best_causal)

| 变体 | doc CFI | 实际 CFI | doc 600s GAIN | 实际 600s GAIN | doc SHAPE | 实际 SHAPE | doc TTP | 实际 TTP |
|---|---|---|---|---|---|---|---|---|
| B1glb | 0.979 | 0.979 ✓ | 1.002 | 1.002 ✓ | +0.97 | 0.97 ✓ | +6 | +6 ✓ |
| A1phys | 0.896 | 0.896 ✓ | 0.657 | 0.657 ✓ | +1.00 | 1.00 ✓ | +0 | +0 ✓ |
| A1mlp | 0.962 | 0.962 ✓ | 0.918 | 0.918 ✓ | +0.99 | 0.99 ✓ | +4 | +4 ✓ |

**L4 表格准确。** ✅

### L6 表格 (ff10)

| 指标 | doc 值 | 实际值 (result.json) | 判定 |
|---|---|---|---|
| ff10 MAE | 0.832 | best_mae=0.832@ep21 ✓ | ✅ 但 doc 说"MAE差(1.55)"→ 1.55 是 best_causal ep6 的 MAE |
| ff10 CFI_agg | 0.833 | **不存在** | ❌ cfi_agg 从未被计算 |
| ff10 gain_mean | 1.004 | **不存在** | ❌ 来源不明 |
| ff10+lg0.5 MAE | 0.815 | 0.8153 ✓ | ✅ |
| ff10+lg0.5 CFI_agg | 0.749 | **不存在** | ❌ |

**L6 表格的 CFI_agg 和 gain_mean 数字无法溯源。** ❌

### L7 表格 (n_lag 消融)

| 指标 | doc 值 | 实际值 | 判定 |
|---|---|---|---|
| nl2 CFI_agg | 0.833 | **不存在** | ❌ |
| nl2 gain_mean | 1.004 | **不存在** | ❌ |
| nl3 CFI_agg | 0.688 | **不存在** | ❌ |
| nl3 gain_mean | 0.554 | **不存在** | ❌ |
| nl2 完胜 nl3 | — | cfe gain: nl2=0.685 vs nl3=0.668 | ✅ 方向正确但差距远小于 doc 声称 |

**L7 表格数字全部无法溯源。** ❌

### doc 最终最优配置

> 架构: A1phys, H=60, linspace 权重, n_lag=2, freeze-free epochs=10

- H=60 linspace: ✅ (所有 A1phys 默认就是 H=60 linspace)
- n_lag=2: ✅ (cfe gain 0.685 > nl3 0.668，且 shape/ttp 相近)
- freeze-free=10: ⚠️ **存疑** — ff10 的 gain (0.685) **低于** baseline (0.765)，freeze 反而恶化了欠增益

---

## 四、P3A freeze-free 效果分析

### 4.1 gain 对比 (cfe, mbar/R_true)

| 变体 | 120s | 180s | 300s | 600s | shape | ttp | MAE |
|---|---|---|---|---|---|---|---|
| A1phys baseline | 1.042 | 0.716 | 0.730 | 0.765 | 0.99 | +1 | 0.858 |
| A1phys ff10 | 0.760 | 0.543 | 0.590 | 0.685 | 0.99 | +4 | 0.832 |
| A1phys ff20 | 1.010 | 0.699 | 0.727 | 0.808 | 0.99 | +2 | 0.857 |
| A1phys ff10+lg0.5 | 0.653 | 0.472 | 0.520 | 0.603 | 0.99 | +4 | 0.815 |
| A1phys mae loss | 0.843 | 0.573 | 0.580 | 0.619 | 0.99 | +1 | 0.879 |

### 4.2 关键发现

1. **freeze 10 epoch 使 gain 下降**: 600s gain 0.765→0.685 (−10%)，180s gain 0.716→0.543 (−24%)
   - 与预期相反：冻结 f_free 应逼 g(x,a) 先吃信号，但解冻后 f_free 重新抢回信号
   - ff20 (0.808) > ff10 (0.685) > baseline (0.765) — ff20 反而比 baseline 好，说明冻 20 epoch 让干预分支充分学习后再解冻

2. **lg0.5 实际未生效**: P2 npz 不存在 → `lambda_gain` 被 fallback 为 0。`ff10_lg0.5` 的 gain 低于 `ff10` 纯粹是训练随机性

3. **MAE loss gain 更差**: 0.619 vs NLL 0.765 — **推翻了 exp_107_review.md 中的假设**。原假设"NLL 对大误差样本降权导致梯度饥饿"不成立。可能原因: NLL 的 sigma 头对动态过程有正则化作用，MAE loss 无此效果

4. **所有 A1phys 变体 shape=0.99, ttp=1~4**: 物理结构保证了形状和时标的稳定性，不受训练策略影响

### 4.3 根因再分析

freeze 10 epoch 恶化 gain 的原因:

- 冻结期间干预分支独占梯度，但训练数据中大多数 batch 无 SP 阶跃 → g(x,a) 在非阶跃样本上学到 ≈0
- 解冻后 f_free 重新吸收信号，但干预分支已被"锁定"在低 gain 状态
- ff20 效果好于 ff10: 冻 20 epoch 让干预分支有更多时间在阶跃样本上学习 K/τ

**正确方向不是 freeze f_free，而是 P3-C: 对 f_free 屏蔽 SP/ΔSP 输入通道** — 从架构上切断 free path 获取动作信息的途径，而非用梯度冻结这种软约束。

---

## 五、exp_103 recheck 分析

n=134 事件 (全量 test 区间, 未放宽筛选), corrected action encoding:

| 模型 | H | MAE | 600s resp | 600s dir | 180s resp | 180s dir |
|---|---|---|---|---|---|---|
| M5DSP | 18 | 0.276 | — | — | 0.202 | 78% |
| M7DSP | 60 | 0.816 | 1.534 | 96% | 0.408 | 67% |
| M9DSP60 | 60 | 0.865 | 1.625 | 94% | 0.532 | 68% |
| M9DSP18 | 18 | 0.326 | — | — | 0.369 | 74% |
| M5DSP_DO | 18 | 0.299 | — | — | 0.119 | 87% |

DiD 真值 600s R_true = 0.725。M7DSP/M9DSP60 的 600s resp = 1.53/1.63 → **旧模型在长程严重过响应** (2x 以上)。

注意: exp_103 用的是 sign(ΔSP) 口径方向率，不是 DiD 口径，不能直接和新架构的 DiD gain 比。但过响应的结论是可靠的 — resp 值是绝对量。

---

## 六、代码 bug 清单

### 6.1 exp_107_did_eval.py:91 — H=18 gt key 错误 (已知, 未修)

```python
gt_onsets = gt[f'onsets{H // 10}0']   # H=18 → onsets10, 不存在
```

H=60 时 H//10*10=60 正确; H=18 时应为 `onsets18`。exp_108 用 `gt_key = H` 是对的。

### 6.2 exp_107_did_eval.py:116-119 — dv_k 切片与 m_a 事件对不上 (已知, 未修)

```python
resp_c = m_a[:, k] * dv_k[:len(m_a)]   # m_a 已按 onset 匹配筛过, dv_k 未对应筛选
```

影响 `prof_old` 旧口径对照 (未用于结论)。

### 6.3 exp_106 eval_causal — JSON 缺 r 字段导致 sgn 失效 (新发现)

`did_response.json` 不含 `r` (逐事件响应)，`eval_causal()` 用 zeros fallback → sgn_pair/sgn_norm 全部失效。

**修复方案**: exp_104 的 JSON 输出应加入 `r` 和 `onsets` 字段; 或 exp_106 改为读 NPZ。

### 6.4 exp_106 P3B gain calibration — 标量监督过于粗糙

`exp_106_causal_arch.py:263-266`:
```python
a_mag = a[:, 0].abs().mean()           # batch平均 |ΔSP|
ideal_g = a_mag * R_true_t.mean()      # scalar: 预期总响应
actual_g = g.mean()                    # scalar: 实际总响应
loss_gain = (actual_g - ideal_g).abs()
```

把 H=60 的完整响应曲线压成一个标量，完全丢失时程信息。即使 P2 npz 就位，这种监督也无法校准增益剖面。

**修复方案**: 改为逐时点监督 `loss_gain = ((g.mean(0) - a_mag * R_true_t).abs() / (R_true_t.abs() + eps)).mean()`。

### 6.5 .gitignore 排除 NPZ — 导致结果不可复现

`*.npz` 被 gitignore，所有 NPZ 结果文件不入库。

**修复**: 在 .gitignore 中添加例外:
```
*.npz
!results/cfe_groundtruth/*.npz
!results/cfe_groundtruth_p2/*.npz
```

---

## 七、可信结论 (仅基于可验证数据)

### 7.1 架构排名 (exp_107, n=15, best_causal ckpt, DiD 口径)

| 维度 | A1phys | B1glb | A1mlp | B1flat |
|---|---|---|---|---|
| 600s gain | 0.657 | **1.002** | 0.918 | 1.008 |
| 180s gain | **0.651** | 0.444 | 0.691 | 0.355 |
| SHAPE | **1.00** | 0.97 | 0.99 | 0.97 |
| TTP_err | **0** | +6 | +4 | +7 |
| gain 跨度 | **0.322** | 0.758 | 0.327 | 0.932 |
| 早期符号 | OK | **FAIL** | OK | **FAIL** |
| MAE | 0.858 | **0.843** | 0.869 | 0.862 |

**A1phys 在因果保真度核心维度 (shape/ttp/180s gain/gain稳定性/早期符号) 全面领先。**
**B1glb/B1flat 在 600s gain 和 MAE 领先，但前 60s 符号反转 + gain 跨度极大。**

### 7.2 n_lag=2 vs n_lag=3 (cfe gain, ff10 配置)

| | 120s | 180s | 300s | 600s | shape | ttp | MAE |
|---|---|---|---|---|---|---|---|
| n_lag=2 | 0.760 | 0.543 | 0.590 | 0.685 | 0.99 | +4 | 0.832 |
| n_lag=3 | 0.679 | 0.533 | 0.601 | 0.668 | 0.99 | +3 | 0.823 |

差距很小，n_lag=2 在 gain 上略优，n_lag=3 在 MAE 和 ttp 上略优。**不可分辨**，需多 seed 才能定论。

### 7.3 训练策略对 gain 的影响

| 策略 | 600s gain | 180s gain | MAE |
|---|---|---|---|
| baseline (NLL, no freeze) | **0.765** | **0.716** | 0.858 |
| ff20 | 0.808 | 0.699 | 0.857 |
| ff10 | 0.685 | 0.543 | 0.832 |
| MAE loss | 0.619 | 0.573 | 0.879 |

**baseline (NLL, no freeze) 已是最优 gain 配置。** 所有"改进"策略都使 gain 下降。

### 7.4 旧模型过响应

M7DSP/M9DSP60 的 600s 响应 (1.53/1.63) 远超 DiD 真值 (0.725)，旧模型在长程严重过响应。这为"新架构的欠增益 (0.65) 是更安全的失败模式"提供了对照。

---

## 八、下一步优先级

### 立即需做 (CPU, 无需训练)

1. **修 .gitignore**: 放行 `results/cfe_groundtruth/*.npz`
2. **跑 exp_109_p2_expand.py**: 生成 P2 扩样本 npz (n_ev≥60) — 这是所有后续步骤的前置
3. **修 exp_104 JSON**: 加入 `r` 和 `onsets` 字段，使 exp_106 训练期 sgn 可用
4. **跑 exp_110_baselines.py**: P2 npz 就位后，补 M5/M7/M9 基线行
5. **修 exp_107 用 cfi_agg()**: 替换 `CE.cfi()` → `CE.cfi_agg()`

### 需 GPU

6. **P3-C: 对 f_free 屏蔽 SP/ΔSP 输入通道**: 在 `ResidualCausalWM` 中给 `free_head` 的输入去掉 SP 和 ΔSP 列。这是从架构上解决信号竞争的正确方向
7. **多 seed (≥5)**: A1phys / B1glb / B1flat 各 5 seeds — 论文需要
8. **A3+B1 组合**: TimeXer 编码器 + 物理干预分支 — 兼顾精度与因果

### 论文写作 (可并行)

9. **方法章节已可写**: L0 协议 bug + L1 DiD 方法 + L2 CFE 框架 + g(x,0)≡0 推导 + 物理分支形式
10. **物理参数图**: `physics_params()` 接口已就绪，画 K(x)/tau(x) vs 负荷 — 论文最有说服力的一张图

---

## 九、session doc 修正清单

`docs/session_2026-08-05_causal_arch_eval.md` 需要以下修正:

1. L6 表格: CFI_agg 和 gain_mean 数字无法溯源 → 删除或标注"来源不明"
2. L6 "ff10 MAE差(1.55)": 改为 "best_causal ckpt 的 MAE=1.555 (ep6), best_mae ckpt 的 MAE=0.832 (ep21)"
3. L7 表格: CFI_agg/gain_mean 数字无法溯源 → 删除，替换为 cfe gain 对比
4. "ff10 纯因果最优": 改为 "ff10 gain 反而低于 baseline (0.685 vs 0.765), freeze 策略未达预期"
5. "gain_mean 0.425→0.624": 来源不明 → 删除
6. 最终最优配置 "freeze-free epochs=10": 加注 "gain 低于 baseline, 不推荐; 推荐保持 baseline (no freeze)"
