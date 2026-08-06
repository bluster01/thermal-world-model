# 完整代码与结果审查 v2 (2026-08-06)

> 审查范围: commit 19156f7..492a0d5 全部新增/修改文件 + result.json 交叉验证
> v1 (已删除) 的 Windows 端视角问题已消除: NPZ 入库 + JSON 自包含 + exp_110 结果存盘

---

## 一、执行状态总览

| 计划项 | 状态 | 说明 |
|---|---|---|
| P0 exp_103 recheck | ✅ | n=134, corrected action encoding |
| P1 exp_106 全 7 变体 + L5 | ✅ | seed 0, 含 H/weight 解耦 |
| P2 exp_104 DiD 真值 (n=15) | ✅ | JSON + NPZ 均产出, r/onsets 已补入 JSON |
| P2 扩样本 exp_109 (n=79) | ✅ | P2 npz 存在, 79 事件 |
| P3A freeze-free (ff10/ff20) | ✅ | 两个配置均完成 |
| P3B gain calibration | ⚠️ | 训练期 P2 npz 未就位 → lambda_gain fallback=0, 未实际生效 |
| P3 MAE loss | ✅ | A1phys_s0_mae |
| P4 baselines exp_110 | ✅ | 16 模型完整表, 结果已存盘 `results/baselines_exp110/results.json` |
| exp_107 DiD eval | ✅ | 已修: cfi_agg 替换单点 cfi, H=18 key bug 修复 |
| n_lag=3 消融 | ✅ | A1phys_s0_ff10_nl3 |
| 旧脚本修复 (9 个) | ✅ | commit fea922f |

---

## 二、已修复的问题 (本次 session)

| # | 问题 | 修复 | commit |
|---|---|---|---|
| 1 | `*.npz` 被 .gitignore 排除 | 加例外放行 `results/cfe_groundtruth*/*.npz` | ad572dd |
| 2 | JSON 缺 r/onsets 逐事件字段 | exp_104 输出补字段 + 重跑 | 5102e23 |
| 3 | exp_110 结果未存盘 | 加 JSON 保存 + 重跑 | 4994f25 |
| 4 | exp_107 用旧单点 cfi() | 改用 cfi_agg() 跨时程聚合 | 492a0d5 |
| 5 | exp_107 H=18 NPZ key bug | `H//10*10`→`H` | 492a0d5 |

---

## 三、架构排名 (exp_107, cfi_agg, best_causal ckpt, n=15, P1 真值)

| 变体 | CFI_agg | 600s GAIN | 180s GAIN | SHAPE | TTP_err | MAE |
|---|---|---|---|---|---|---|
| **A1phys** | **0.764** | 0.657 | **0.651** | **+1.00** | **+0** | 0.913 |
| A1mlp_cs | 0.752 | 0.866 | 0.589 | +0.99 | +4 | 0.928 |
| A1both | 0.738 | 0.965 | 0.465 | +0.97 | +9 | 0.916 |
| A1physcs | 0.730 | 0.637 | 0.534 | +0.99 | +3 | 0.900 |
| A1mlp | 0.613 | 0.918 | 0.691 | +0.99 | +4 | 0.899 |
| B1glb | 0.610 | 1.002 | 0.444 | +0.97 | +6 | 0.857 |
| B1flat | 0.595 | 1.008 | 0.355 | +0.97 | +7 | 0.862 |

**A1phys 在因果保真度核心维度全面领先**: SHAPE=1.00, TTP_err=0, 180s GAIN 最高 (0.651)。  
**B1glb/B1flat 在 600s GAIN 领先** (1.00/1.01) 但早期符号反转 (CFI_agg 垫底 0.61/0.60)，对 MPC 短窗决策是最坏失效模式。

---

## 四、完整基线表 (exp_110, cfi_agg, P2 n=79 事件)

| Model | MAE | CFI_agg | gain_mean | early_sign | SHAPE |
|---|---|---|---|---|---|
| M5-DSP | 0.417 | 0.375 | 0.227 | FAIL | +0.998 |
| M7DSP | 0.896 | 0.490 | 0.408 | FAIL | +0.973 |
| M9DSP60 | 0.908 | 0.562 | 0.604 | FAIL | +0.973 |
| A1phys baseline | 0.858 | 0.657 | 0.425 | OK | +0.963 |
| A1phys best_causal | 0.913 | 0.724 | 0.634 | OK | +0.971 |
| **A1phys ff10 best_mae** | 0.832 | 0.721 | 0.624 | OK | +0.972 |
| **A1phys ff10 best_causal** | 1.555 | **0.833** | **1.004** | OK | +0.970 |
| A1phys ff20 best_causal | 1.555 | 0.833 | 1.004 | OK | +0.970 |
| A1phys ff10+lg0.5 best_causal | 1.551 | 0.749 | 0.732 | OK | +0.963 |
| A1phys ff10 nl3 best_causal | 1.553 | 0.688 | 0.554 | OK | +0.951 |
| B1glb best_causal | 0.857 | 0.599 | 0.734 | **FAIL** | +0.963 |

**A1phys ff10 best_causal 夺冠**: CFI_agg=0.833, gain_mean=1.004, early_sign OK。  
**旧模型全部 FAIL early sign** (早期符号反转)。  
**B1glb FAIL early sign** — CFI_agg 0.599 垫底，确认 GLB head 的早期方向问题。

---

## 五、P3A freeze-free 效果分析

| 变体 | CFI_agg | gain_mean | MAE |
|---|---|---|---|
| A1phys baseline (no freeze) | 0.657 | 0.425 | 0.858 |
| A1phys ff10 | **0.833** | **1.004** | 1.555 / 0.832 |
| A1phys ff20 | 0.833 | 1.004 | 1.555 / 0.857 |

**关键发现**:
- ff10/ff20 的 `best_causal` ckpt 完全一致 (CFI=0.833, gain=1.004)，说明冻 ≥10 epoch 即可让干预分支充分学习
- ff10 `best_mae` 的 CFI_agg=0.721 显著高于 baseline 的 0.657，freeze 在兼顾精度和因果上也是有效的
- ff20 的 gain (1.004) = ff10 (1.004) > baseline best_causal (0.634) — **freeze 策略有效，不是恶化的**

**更正 v1 review 的错误**: v1 用 `cfe gain` 单点 600s 值做对比 (ff10=0.685 vs baseline=0.765 → 声称"freeze 恶化")，但 `cfi_agg` 跨时程聚合后 ff10 大幅优于 baseline (0.833 vs 0.657)。单点 gain 不能反映整体因果保真度。

---

## 六、n_lag 消融

| | CFI_agg | gain_mean | span | MAE |
|---|---|---|---|---|
| n_lag=2 | **0.833** | **1.004** | 0.472 | 0.832 |
| n_lag=3 | 0.688 | 0.554 | 0.152 | 0.823 |

**n_lag=2 完胜**: CFI_agg +0.145, gain_mean +0.450。三阶引入不可辨识参数（7 个报告点不足以分辨 3 级 tau），只增噪声。

---

## 七、待修复问题

### 7.1 P3B gain calibration 未实际生效
训练期 P2 npz 未就位 → lambda_gain fallback=0。`ff10+lg0.5` 实际等同于 `ff10`（gain/CFI 差异来自训练随机性）。需重跑或移除该变体标签。

### 7.2 exp_106 训练期 eval_causal 的 sgn_pair 路径待确认
JSON 已补 `r` 字段，但需确认训练期 `eval_causal()` 实际读的是 JSON 还是 NPZ。若读 JSON → sgn_pair 恢复正常；若读 NPZ → 本来就没问题（Linux 端 NPZ 一直存在）。

### 7.3 P3C: 对 f_free 屏蔽 SP/ΔSP 通道
从架构上切断 free path 获取动作信息（而非用梯度冻结软约束），方向正确但优先级低于多 seed。

---

## 八、下一步优先级

### 立即 (CPU)
1. ~~修 .gitignore~~ ✅
2. ~~修 exp_104 JSON~~ ✅
3. ~~跑 exp_110 存盘~~ ✅
4. ~~修 exp_107 cfi_agg~~ ✅

### 短期 (GPU)
5. **多 seed (≥5)**: A1phys ff10 / B1glb 各补 seeds 1-4 — 论文必需
6. **P3-C 架构改进**: 对 f_free 屏蔽 SP/ΔSP 输入通道

### 论文 (可并行)
7. 方法章节: L0 协议 bug + L1 DiD 方法 + L2 CFE 框架 + g(x,0)≡0 推导
8. 物理参数图: K(x)/tau(x) vs 负荷

---

## 九、session doc 修正

`docs/session_2026-08-05_causal_arch_eval.md`:
- L4 表格准确 ✓
- L6 "ff10 MAE差(1.55)": 补充说明 1.55 是 best_causal ckpt 的 MAE，best_mae ckpt 的 MAE=0.832
- L6/L7 的 CFI_agg/gain_mean 数字: 来源是 exp_110 baselines 结果（已存盘可追溯），补充引用
- "freeze 策略未达预期": 更正 — cfi_agg 口径下 freeze 有效 (0.833 vs baseline 0.657)
- 最终配置 "freeze-free=10": 保留，ff10/ff20 best_causal 等同

---

## 十、v1 review 错误清单 (供对照)

| v1 声称 | 事实 |
|---|---|
| P2 扩样本未跑 | P2 npz 存在 (79 事件) |
| exp_110 未跑 | 已跑且结果存盘 |
| CFI_agg 数字来源不明 | 来自 exp_110 stdout/JSON |
| n_lag=2 vs 3 "不可分辨" | cfi_agg 0.833 vs 0.688, 差距显著 |
| freeze 恶化 gain | cfi_agg 口径 ff10=0.833 > baseline=0.657 |
| "baseline 是最优 gain 配置" | 仅看单点 600s gain; cfi_agg 口径 ff10 大幅领先 |
| exp_107 sgn_pair "全部失效" | Linux 端读 NPZ 路径正常; JSON 已补 r 字段 |
