# RM3 / RM3-A 独立架构审计 + RM3-B 设计评审 (2026-08-13)

> 审计对象: `6c3746c..16cee58`, 重点 `2213000`(RM3) / `ba7a8f1`(RM3-A) / `16cee58`(RM3-B DRAFT)
> 方法: 直接从 66 个 `checkpoint_best_validation.pt` + 66 个 `episodes_validation.npz` + 12 个校准 NPZ + RM2 的 48 个 metrics 复算, 不依赖既有审计文档的叙述
> 立场: 只列可从产物复现的结论, 全部命令见 §7
> 状态: 本地独立审计, 非授权文件

---

## 0. 摘要

RM3/RM3-A 的**执行纪律与不宣布冠军的克制是对的**, RM3-B 的方向(可识别性门 + 显式非平稳)也是对的。但复算暴露一个此前未被识别的**根因**, 它同时解释了 RM2→RM3 的响应塌缩和 RM3-A 的"架构方向":

> **RM3 在"公平预测"的名义下删除了 RM2 的 `structure_penalty`(权重 0.10, 用 logged 阀位监督响应分支)。响应算子从此只被塌缩的阀位预测驱动, 幅值从 RM2 的 0.186 °C 掉到 0.028 °C(6.7×), 数值上变成惰性分支。**

连带三个后果:

1. **RM3 相对 RM2 是审计能力的倒退** — RM2 有持续基线、shuffled-action 安慰剂、效应幅值三套诊断, RM3 全部丢失。RM3-B 的 E1/E7/E8 大部分是在**重建 RM2 已有的东西**。
2. **RM3-A 的"P5 架构方向成立"结论不成立** — 双向容量匹配匹配错了轴。P5 赢终端是因为它独有一条 120 输出的 **action-invariant** 自由终端头; P3/P4 赢局部是因为它们独有 23.8k 参数的自由局部头。缩放 `d_model` 碰不到这个机制。
3. **RM3-A 自己的数据反证了 P5 有动作通路** — A4 的阀位 MAE 最差(+0.070, 0/6)而终端 MAE 最好(−0.024, 5/6)。**把阀位预测变差反而让终端变好。**

因此 §6 的建议是: **RM3-B 之前先做 RM3-0(零训练回放 + 恢复 RM2 诊断 + 一次 bypass 消融)**, 否则 RM3-B 会在一个错误的架构结论上叠加七篇论文的理论装置。

---

## 1. 根因: RM2 的 `structure_penalty` 被 RM3 删除

### 1.1 RM2 怎么训练响应分支

`src/phase35/multistep/gatec_rm2_training.py`:

```python
output = model(
    tensors["history"], tensors["future_sp"],
    boundary_mode="forecast_boundary",
    logged_future_valve_for_aux=tensors["valve"] if response_expected else None,
)
if response_expected:
    output["structure_penalty"] = F.smooth_l1_loss(
        output["logged_local_drop_prediction"] / scales.values["local"],
        tensors["local"]                       / scales.values["local"],
    )
loss = compute_gatec_loss(..., weights, ...)   # DEFAULT_WEIGHTS["structure"] = 0.10
```

`logged_local_drop_prediction = residual_local + response(context, logged_valve, baseline)`。
即 **响应算子被真实阀位轨迹以 0.10 权重直接监督**。预测路径 `local_drop_prediction` 仍然只用 forecast 阀位, 推理零泄漏。manifest 也如实标注:

```
logged_future_valve_role = "response_auxiliary_and_validation_diagnostic_only"
```

### 1.2 RM3 怎么训练响应分支

`src/phase35/multistep/rm3_training.py` 的 `rm3_scope_loss` 只有 valve/tin/local/terminal 四项, 从不传 `logged_future_valve_for_aux`。`rm3_prediction.RM3FairPredictionAdapter.forward` 更进一步**硬禁止**:

```python
if logged_future_valve is not None:
    raise Phase35ProtocolError(
        "RM3 logged future valve may enter only the declared M7 oracle candidate")
```

RM2 的 `structure` 权重 0.10 与 `rollout` 权重 0.111 一并消失, 权重从 `valve/tin/local/terminal = 0.167/0.167/0.278/0.278(+rollout 0.111)` 变成一律 `0.25`。

### 1.3 后果: 同路由同边界模式, 效应幅值塌 6.7×

`a1phys_three_pole` + `scheduled` + `boundary_mode="forecast_boundary"`, 两批次完全同构:

| 批次 | 候选 | 训练时 logged-aux 监督 | predicted effect (°C) | logged effect (°C) | local MAE (°C) | shuffled 安慰剂优势 (°C) |
|---|---|---|---:|---:|---:|---:|
| RM2 | `A2_a1_sched_base` | **有** (structure 0.10) | **0.1864** | 0.3192 | 1.6879 | **0.7395** |
| RM2 | `A3_a1_sched_large` | 有 | 0.1890 | 0.3184 | 1.6756 | 0.7370 |
| RM2 | `B1_koopman` | 有 | 0.6302 | 1.0885 | 1.7054 | 3.4104 |
| RM2 | `A0_paired_free` (无响应) | — | 0.0000 | 0.0000 | 1.7557 | 0.0000 |
| **RM3** | `P4_gatec_a1_scheduled` | **无** | **0.0277**† | 0.0462† | 1.6383 | **未测量** |

† RM3 未记录该指标; 由 `P4_gatec_a1_scheduled_F0_s0` 的 checkpoint 参数重建响应算子并在其自身 `valve_prediction` / `logged_valve` 上前向复算得到(§7 命令一)。logged/predicted 比值 RM2 1.71 vs RM3 1.67, 内部结构一致, 只是整体尺度缩了 6.7×。

**RM3 的 A1 分支输出只有 local MAE 的 1.7%**(0.0277 / 1.638)。这解释了 P4−P3 = −0.00056 °C, 也解释了 P3 与 P4 的 `best_update` 序列**逐元素完全相同** `[3700,3900,4000,4000,3900,4000]` — 响应分支对选择器损失的贡献低于 `minimum_score_improvement`(1e-4)。

### 1.4 这是过度收紧, 不是必要的防泄漏

RM3 的封禁把**训练期的识别锚点**当成了**推理期的 oracle** 处理。两者性质不同:

- RM2 的 aux 只进 `structure_penalty`, 是对一条**反事实分支**的正则; 报告用的 `local_drop_prediction` 走 forecast 阀位。
- 真正的 oracle 泄漏是 P0 那种: 把 logged 阀位喂进**被评测的预测路径**。

`gatec_model.MeasuredBoundaryMIMOWorldModel.forward` 的 `logged_future_valve_for_aux` 参数**在 RM3 代码里依然存在**, 只是没人调用。恢复成本约 10 行。

---

## 2. 上游: 阀位预测塌缩到近持续基线

即使恢复 structure penalty, forecast 路径本身仍然坏。

| 候选 | valve MAE | 持续基线 | 技巧 | 逐步&#124;Δvalve&#124; 预测/实测 | 水平跨度 预测/实测 |
|---|---:|---:|---:|---:|---:|
| P1 | 3.353 | 3.580 | 6.3% | 0.052 / 0.312 (**1/6**) | 3.03 / 9.01 |
| P3 | 3.089 | 3.580 | 13.7% | 0.073 / 0.312 | 3.34 / 9.01 |
| P4 | 3.085 | 3.580 | 13.8% | 0.073 / 0.312 | 3.32 / 9.01 |
| P5 | 3.240 | 3.580 | 9.5% | 0.058 / 0.312 | 3.19 / 9.01 |

阀位预测是**过度平滑的近持续轨迹**。响应算子输入 `dose = opening(valve) − opening(baseline_valve)`, 阀位几乎不动 ⇒ dose ≈ 0。

以持续(hold-last-value)为基线的终端技巧:

| 候选 | terminal MAE | 技巧@120s | @300s | @600s |
|---|---:|---:|---:|---:|
| P0 M7 **oracle** valve | 0.648 | 0.476 | 0.556 | 0.517 |
| P1 M7 predicted valve | 0.931 | 0.268 | 0.377 | 0.310 |
| **P2 M9 future SP** | 1.424 | **−0.079** | **−0.014** | **−0.007** |
| P3 Gate C paired free | 1.042 | 0.221 | 0.347 | 0.200 |
| P4 Gate C A1 scheduled | 1.029 | 0.236 | 0.351 | 0.206 |
| P5 hybrid joint latent | 0.957 | 0.249 | 0.365 | 0.282 |

- 精确知道未来阀位, 技巧**翻倍**(0.27→0.48 @120s)。这与 RM3-B §1 的 0.29 °C 判断一致, **同意 RM3-B 把 A4(PI 结构化阀位策略)列为主要改动**。
- **P2 (M9) 全时程低于持续基线** — 不是"未被选中", 是不如"保持不动"。Phase 3 的 M9DSP 叙事建立在这条线上, 需要在论文里明确处理。

RM3 的 `smooth_l1_loss` 对**绝对阀位**取平均, 天然奖励平滑; `CausalValveDecoder` 又只看 `future_sp` 和自身上一步输出, 看不到控制误差 `SP − T`。塌缩是结构性可预期的。

---

## 3. RM3 相对 RM2 的诊断能力倒退

RM2 的 `metrics_validation.json` 每个 run 都有(`gatec_rm2_training.py` L328-334):

```
persistence_valve_mae_c / persistence_tin_mae_c / persistence_local_mae_c / persistence_terminal_mae_c
valve_to_persistence / tin_to_persistence / local_to_persistence / terminal_to_persistence
logged_local_mae_c / shuffled_local_mae_c / logged_vs_shuffled_local_advantage_c
predicted_effect_mean_abs_c / logged_effect_mean_abs_c / logged_effect_h6 / logged_effect_h18
oracle_terminal_mae_c / stable_pole_max
```

**RM3 与 RM3-A 一个都没有。** `rm3_training._report` 只产出 `terminal_mae_c / valve_mae / tin_mae_c / local_mae_c / scope_selector_score`。

RM2 的 shuffled-action 安慰剂本来是有效的裁判(A2: logged local 1.523 vs shuffled 2.263, 优势 0.740 °C; A0 无响应分支时优势恰为 0.000)。RM3 丢弃它之后, 没有任何机制能发现响应分支已经死了 — 这正是 §1.3 需要靠外部复算才暴露的原因。

**对 RM3-B 的直接影响**: E1(幅值分歧门)、E7(rollout)、E8(分层) 里相当部分是重建 RM2 已有设施。建议 RM3-B 明确写成"恢复并升级 RM2 诊断集", 而不是新建, 以免重复实现并丢失可比性。

---

## 4. RM3-A 复算: 结论方向需要修正

### 4.1 复算结果 (30 runs 全部 complete, 与 ledger 一致)

| 候选 | d_model | 参数 | terminal | local | valve | tin | best_update |
|---|---:|---:|---:|---:|---:|---:|---|
| A0_p3_large | 77 | 120,928 | 1.0528 | 1.6403 | 3.1710 | 1.5957 | 3600–4000 |
| A1_p4_large | 77 | 121,103 | 1.0515 | 1.6443 | 3.1683 | 1.6013 | 3900–4000 |
| A2_p5_small | 52 | 83,649 | 0.9743 | 1.9772 | 3.3202 | 1.6795 | 2300–4000 |
| A3_p5_local35 | 64 | 122,301 | 0.9577 | 1.8837 | 3.3478 | 1.7725 | 3500–4000 |
| A4_p5_local50 | 64 | 122,301 | **0.9490** | **1.8317** | 3.4009 | 1.8398 | 2700–4000 |
| *(参考)* P3 | 64 | 87,258 | 1.0584 | 1.6387 | 3.1662 | 1.6023 | 3700–4000 |
| *(参考)* P5 | 64 | 122,301 | 0.9727 | 1.9510 | 3.3313 | 1.6676 | 3500–4000 |

配对差(6 folds×seeds):

```
A0_p3_large   − P5   terminal +0.0801 (0/6)   local −0.3107 (6/6)   valve −0.1603 (6/6)
A1_p4_large   − P5   terminal +0.0787 (0/6)   local −0.3068 (6/6)   valve −0.1630 (6/6)
A2_p5_small   − P3   terminal −0.0841 (6/6)   local +0.3385 (0/6)   valve +0.1539 (0/6)
A0_p3_large   − P3   terminal −0.0056 (4/6)   local +0.0016 (4/6)   valve +0.0047 (3/6)
A4_p5_local50 − P5   terminal −0.0238 (5/6)   local −0.1194 (6/6)   valve +0.0696 (0/6)
A4_p5_local50 − A3   terminal −0.0088 (5/6)   local −0.0520 (6/6)   valve +0.0531 (0/6)
```

### 4.2 反对意见 1: 容量匹配匹配错了轴

`A0 − P3`: 参数 +39%(87k→121k), `residual_head` 23,800→30,612, 结果 terminal −0.0056、local +0.0016、valve +0.0047 — **三项全在噪声内**。Gate C 根本不是容量受限的。

真正区分两个家族的是**自由头的位置**, 与 `d_model` 正交:

| | 自由局部头 | 自由终端头 | local MAE | terminal MAE |
|---|---|---|---:|---:|
| P3 / P4 / A0 / A1 | `residual_head` **23.8k–30.6k**, 120 自由输出 | **无** | **1.638–1.644** | 1.052–1.058 |
| P5 / A2 / A3 / A4 | **无**(共享 8×32 readout 的 2 行, 64 参数) | `terminal_bypass` 2.9k–3.1k, **120 自由输出** | 1.832–1.977 | **0.949–0.974** |

**每个家族恰好赢在它有自由头的那个通道。** 两个自由头的输出维度都固定为 `horizon*2 = 120`, 由水平决定, 与 `d_model` 无关 — 所以 RM3-A 的双向容量匹配在设计上**不可能**触及该机制。

局部头容量差 **372 倍**(23,800 vs 64)。把这个称为"joint latent 架构方向"缺乏依据。

### 4.3 反对意见 2: RM3-A 的数据反证 P5 有动作通路

```
A4_p5_local50 − P5 :  valve MAE +0.0696 (0/6, 最差)   terminal MAE −0.0238 (5/6, 最好)
A4_p5_local50 − A3 :  valve MAE +0.0531 (0/6)         terminal MAE −0.0088 (5/6)
```

沿 P5 → A3 → A4, 阀位 MAE 单调变差(3.331 → 3.348 → 3.401), tin 也变差(1.668 → 1.773 → 1.840), 而终端 MAE 单调变好(0.973 → 0.958 → 0.949)。

**若终端存在真实动作通路, 恶化阀位预测应当恶化终端。实测相反。** 结合 §1.3 的 0.028 °C 效应幅值, 结论是 P5 的终端输出对 `do(valve)` 基本无响应。

`rm3_joint_model.py` 的结构也支持这一点:

```
terminal          = terminal_physical + bypass
terminal_physical = baseline_terminal + state_readout(latent)[6:8]
latent 驱动       = boundary_projection(tin − baseline_tin) + local_response_projection(effect)
tin               = baseline_tin + tin_head(context)   # 纯 context, 无动作
effect            ≈ 0.028 °C                            # §1.3
bypass            = f(history only)                     # 按设计 action-invariant
```

`ActionInvariantTerminalBypass` 输出层权重范数在 **24/24 个 P5 家族 run** 中从初始 0.0438 长到 **14.80–19.35(340–440×)**, 无一例外。折内去均值后与 terminal MAE 相关 −0.386(n=24, 仅提示性, 不作结论)。

### 4.4 反对意见 3: A4 严格支配 P5, 所以 P5 不是 Pareto 点

`A4 − P5`: terminal −0.0238(5/6) **且** local −0.1194(6/6)。两项都更好 ⇒ **严格支配**, 不是权衡。

`ba7a8f1` 的 commit message 与 RM3-A 设计把 A3/A4 定位为 "Pareto 边界点", 但数据显示 **P5 的 balanced 权重根本不在 Pareto 前沿上**。而 P5 正是 RM3 终审用来对比 P3/P4 的参照点。RM3 的 "P5 终端优势" 是在一个被支配的工作点上测得的。

### 4.5 附带: 选择器分数跨候选不可比

A3/A4 的 `component_loss_weights` 分别是 `{0.15,0.15,0.35,0.35}` 和 `{0.10,0.10,0.50,0.30}`, manifest 记为 `validation_full_multitask_declared_component_weighted_loss`。**这是三个不同的目标函数。** 而 `configs/phase3_5/ms3r_rm3a_matrix.json` 的 `decision_contract.report_metrics` 里包含 `scope_selector_score` — 该项在跨候选表中无意义, 应删除。

### 4.6 全部 RM3/RM3-A 全多任务 run 都撞更新上限

| 候选 | best_update |
|---|---|
| P0 | 1700–3400 (patience 早停) |
| P1 | 2500–3700 |
| P3 / P4 | 3700–4000, 各 2 个 run 恰在 4000 |
| P5 | 3500–4000 |
| A0 / A1 | 3600–4000, 4 个 / 3 个 run 恰在 4000 |
| A2 / A3 / A4 | 2300–4000 |

P0/P1 靠 patience 收敛, 其余**在上限处仍在改善**。排名同时是"收敛速度"的陈述。RM3-B §5 的 RM3-B3 沿用 4000 上限(且 §10 把它列为不可变), 会继承该混淆。**建议至少补一次"提高上限"的收敛性检查, 或明示该上限是预算约束而非收敛判据。**

---

## 5. 其余架构与统计问题

### 5.1 符号正确是架构强制的, 不构成证据

`gatec_model.StableMIMOResponseBase._equilibrium_modes`:

```python
diagonal = F.softplus(self.diagonal_gain) + 1e-3                        # > 0
cross    = 0.35 * torch.sigmoid(self.cross_gain_logits) * diagonal.flip(0)  # > 0
```

`MonotoneValveOpening` 是 softmax 对 {x, x², x³} 的凸组合 ⇒ 单调增。且 `residual_head` **从不接收阀位**。因此开阀 ⇒ dose ≥ 0 ⇒ effect ≥ 0 ⇒ local_drop 增大, **模型在结构上不可能给出错误符号**。

任何基于本架构的"方向正确率"统计都是循环论证; 唯一可证伪的是幅值。这一点应写进论文 limitation, 也应阻止把 Phase 3 的 CFI/方向率指标搬过来。RM3-B §1 说"正确性由结构先验保证, 没有可证伪的统计裁判"——**同意, 且这里给出了具体机制**。

### 5.2 响应分支与自由分支信息集相同(Phase 3 问题原地复发)

```
valve    = valve_policy(context, future_sp, baseline_valve)
response = local_response(context, valve, baseline_valve)
local    = residual_head(context) + response.effect
```

`response` 是 `(context, future_sp)` 的确定函数, `residual_head` 拿到的正是 `context`。信息集几乎相同, 自由通路参数多 372 倍 ⇒ 自由通路吃掉全部信号。这与 Phase 3 `exp_106` 的 `f_free` 抢 ΔSP 信号**完全同构**, 动作从 ΔSP 换成了阀位。Phase 3 诊断出的正确修法(切断自由通路的动作输入, P3-C)**至今未执行**。

### 5.3 P3 vs P4 与初始化混淆

`MeasuredBoundaryMIMOWorldModel.__init__` 顺序: encoder → valve_policy → tin_forecaster → residual_head → **local_response** → downstream。P4 多 7 个响应张量, 消耗 RNG ⇒ `downstream` 初始化不同。实测同 fold 同 seed 下 35 个共享张量只有 2 个数值接近, 相对差最大的恰是响应算子之后构造的 `downstream`:

```
downstream.input_projection.weight   1.813
downstream.output_projection.weight  1.347
downstream.initial.weight            1.340
valve_policy.cell.weight_ih          0.420
```

结论方向(无优势)因幅度太小仍成立, 但**归因不成立**。修法: 每个子模块用独立 `torch.Generator`, 或把可选分支放到 `__init__` 最后构造。

### 5.4 三极点形状主张没有分辨力

复算确认修正后 NNLS 的 RMSE 为 0.024–0.072(与终审一致)。但同一批 `r0_trajectory_matrix` 换基底:

| 单元 | 3极点(60/180/600s) | 单极点(180s) | 2极点 | 幂律 {t^0.5,t,t^1.5} | 直线 t |
|---|---:|---:|---:|---:|---:|
| F0_s0_h18 | 0.0481 | 0.0658 | 0.0524 | 0.0599 | 0.0873 |
| F0_s0_h6 | 0.0608 | 0.0655 | 0.0629 | **0.0511** | 0.0671 |
| F0_s1_h6 | 0.0317 | 0.0344 | 0.0323 | **0.0210** | 0.0312 |
| F1_s0_h18 | 0.0242 | 0.0426 | 0.0299 | 0.0314 | 0.0608 |
| F1_s2_h6 | 0.0523 | 0.0537 | 0.0527 | **0.0450** | 0.0525 |

单极点几乎一样好; **幂律基底在 8/12 个单元里更好**; h6 上直线都够用。用 3 个非负系数拟合 6 或 18 个单调点不能识别极点结构。

**这直接影响 RM3-B 的 E6** — 把 NNLS 轨迹定为"经验响应参考锚点"是危险的: 该锚点本身不可辨识, 用它做形状距离等于用一条任意平滑单调曲线当真值。E6 必须先补模型选择(1/2/3 极点 vs 幂律 vs 线性, AIC/BIC 或样本外), 否则应降级为诊断而非锚点。

相关: `tau_max = 1200 s` 而水平仅 600 s。P4_F0_s0 学到的 tau 为 32.8/126.4/**788.7** s(权重 0.375/0.454/**0.171**), 第三极点在水平末只到 53% 稳态 — 窗内不可辨识。

### 5.5 校准 JSON 仍是修正前的错误数字

`rm3_calibration.py` 已是 exact active-set NNLS, 但 12 个 `calibration_validation.json` 里 `R1_a1_scheduled.projection_rmse` 仍是旧值:

| 单元 | 存档值 | 正确复算 |
|---|---:|---:|
| F0_s0_h18 | 3.3254 | 0.0481 |
| F0_s0_h6 | **22.3237** | 0.0608 |
| F1_s0_h18 | 1.2429 | 0.0242 |
| F1_s2_h6 | 17.3280 | 0.0523 |

修正值只存在于审计文档散文里。**读结果树会拿到错的数字。** 纯 CPU 后处理即可重生成, 无需重训。

### 5.6 跨侧耦合过大 / 跨折增益互换

R0 端点矩阵(h18):

| 单元 | A→A | A→B | B→A | B→B |
|---|---:|---:|---:|---:|
| F0_s0 | 0.620 | 0.060 | 0.089 | 0.307 |
| F0_s1 | 0.549 | 0.074 | 0.153 | 0.305 |
| F0_s2 | 0.538 | 0.088 | 0.118 | 0.314 |
| F1_s0 | 0.322 | 0.071 | 0.105 | 0.482 |
| F1_s1 | 0.352 | 0.106 | 0.133 | 0.442 |
| F1_s2 | 0.337 | 0.073 | 0.133 | 0.443 |

- F1 的 `B→A / A→A` 达 **33–40%**。A/B 是物理分离的减温器支路, 该量级更像**未除净的共模混杂**。12/12 单元 `R2 independent_channels_supported=False` 与此一致。
- A→A 折均 0.569→0.337(−41%), B→B 0.309→0.456(+47%), **两侧主导互换**。折内 seed 离散仅 ±0.04, 而激励能量基本相同(共模 13.0–18.8, 差模/共模 0.35–0.50) ⇒ **不是激励差异造成的**。

RM3-B §1 把它读作"时变、扰动条件响应"。**同样符合另一解释: 哪一侧在该时段被主动调节。** 若是后者, 估计量是观测性的而非因果的。RM3-B 的 E3(b)(fold 对增益差 bootstrap CI) 只能确认"有差异", 不能区分这两种解释 — **建议 E3 增加一项: 把增益差回归到工况变量 + 各侧阀位活跃度上**, 这也正好衔接 E8。

### 5.7 早期步增益为负

A→A 逐步(每 3 步, h18): `F0_s2: -0.31 0.12 0.32 0.39 0.43 0.46`; `F1_s1: -0.10 -0.08 0.07 0.19 0.29 0.31`。

物理先验下应 ≥0 并带死区。负值指向**未建模的瞬时控制耦合** — 正是 RM3-B A3(IDOL 瞬时耦合层)要处理的对象。**这条为 A3 提供了 RM3-B 目前缺少的经验依据**, 建议写进 A3 的动机。

### 5.8 指标全程按双侧池化

`rm3_training._report` 用 `elements = len(anchors) * horizon * 2`, 所有 MAE 都是 A/B 池化均值。但 §5.6 显示两侧行为差异大且跨折互换。RM2 记录了 `a_only_effect` / `b_only_effect`, RM3 又丢了。**RM3-B E8 应把"分侧"与"分工况"并列为强制维度。**

---

## 6. 对 RM3-B 设计的评审

### 6.1 同意的部分

| RM3-B 项 | 本审计的独立支持 |
|---|---|
| §1 "分解正确性由结构先验保证, 无可证伪裁判" | §5.1 给出具体机制: 符号被 softplus/sigmoid/单调开度强制 |
| §6 闭环反馈违反 IN 假设 | §5.2 响应与自由分支信息集相同, 是同一问题的可观测面 |
| A4 PI 结构化阀位策略 | §2 阀位技巧仅 6–14%, 逐步幅度 1/6; P0−P1 缺口 0.29 °C 复算确认 |
| A3 瞬时耦合层 | §5.7 早期步负增益提供直接经验依据 |
| A6 时变增益 AR | §5.6 跨折互换复算确认, 且排除了激励差异解释 |
| E4 充分变异性**训练前**资格门 | 同意方向; 但见 §6.2 第 4 条 |
| E7 递归 rollout / W3 门 | 同意; 注意 RM2 的 `rollout` 权重只是"后 1/3 水平的 teacher-forced 终端损失", **不是**递推, 命名易混淆 |
| §8 不可提前声称清单 | 完全同意, 建议原样冻结为论文用语 |

### 6.2 反对或需要修改的部分

1. **§9 "纯容量扩充: RM3-A 已证 terminal 优势是架构性的, 容量杠杆已耗尽" — 不成立。**
   见 §4.2: 容量匹配匹配了 `d_model`, 而机制是**自由头位置**(两个自由头输出维度都固定为 `horizon*2`, 与 `d_model` 正交)。`A0−P3` 三项指标全在噪声内, 说明 Gate C 不是容量受限, 而不是"架构方向成立"。
   **要求**: 在 RM3-B 之前补 **P5-nobypass** 与 **P3+bypass** 两个对照。若 P3+bypass 达到 ~0.95 terminal, "joint latent 架构方向"即被证伪, RM3-B 的架构基线需要重选。

2. **§1 三个"硬事实"缺第四个, 且第一个的解释需要修正。**
   RM2 的 `OPERATOR_GAIN_NOT_IDENTIFIED` 被归因为"residual head 与 response operator 互相补偿"。这成立, 但 RM3 的情况已经不同: **RM3 的响应分支不是"与残差互相补偿", 而是被删掉 structure penalty 后直接饿死到 1.7%**(§1)。两者需要区分, 否则 E1(幅值分歧门)会在 RM3 基线上得到"所有候选幅值都≈0, 分歧比值不稳定"的退化结果。
   **要求**: E1 除比值外, 必须加**幅值下限门**(如 `predicted_effect_mean_abs_c` 相对 local MAE 的占比 ≥ 预注册阈值), 否则"都接近 0"会被误判为"分歧小 ⇒ 通过"。

3. **§4 A 系列全部建立在"响应分支能收到激励"的前提上, 而该前提当前不成立。**
   A1(机制噪声 σ_r)、A2(下三角 Jacobian)、A5(工况开关)、A6(时变增益 AR) 都是在给一个输出 0.028 °C 的分支加结构。
   **要求**: 把"恢复 structure penalty(或等效的响应激励机制)+ A4 阀位策略"作为 A 系列的**前置条件**, 而不是并列项。在响应幅值恢复到 RM2 量级(≥0.15 °C)之前, A1/A2/A5/A6 不具备可测性。

4. **E4 用 `RM3MomentAudit.condition_number` 作充分变异性指标 — 不敏感。**
   §5.6 实测条件数 1.955–3.474, 差模/共模 0.338–0.578, **跨折几乎不变**, 却对应 A→A 增益 −41% 的巨大漂移。条件数在这里没有分辨力。
   **要求**: E4 的门槛不能只用动作 Gram 条件数, 需补**每侧独立**的剂量幅度与时间变化率指标(RM3-B 已提"每侧/每负荷 bin 剂量多样性", 请把它设为主指标而非补充)。

5. **E6 把 NNLS 轨迹当经验锚点 — 见 §5.4, 该锚点不可辨识。** 降级为诊断, 或先补模型选择。

6. **§10 "4000-update 上限不可变" 与 §4.6 冲突。**
   所有全多任务 run 都在上限处仍在改善。把预算约束写成不可变纪律, 等于把一个已知混淆冻结进后续所有批次。
   **要求**: 保留"跨批次同预算可比"的纪律, 但补一次一次性的收敛性诊断(单 fold 单 seed 跑到 12000 updates), 用于判断 4000 是否已进入平台。

7. **§5 批次表建议增补 RM3-B0 的内容。**
   RM3-B0 目前是"E1/E2/E3/E6 回放"。建议追加零训练的 **bypass-off / response-off / placebo-valve 推理消融**(现有 66 个 checkpoint 直接前向, 无需重训), 这三项的信息量高于 E2/E6 且成本更低。

8. **命名: "RM3-B" 之前应有一个 RM3-0。** 见 §6.3。

### 6.3 建议的执行顺序

```
RM3-0  (零训练 / 纯 CPU, 当天可完成)
  0.1  重生成 12 个 calibration JSON (修 §5.5 的错数字)
  0.2  在 66 个已冻结 checkpoint 上补 RM2 诊断集:
       persistence / oracle 基线, predicted+logged effect 幅值, shuffled-valve 安慰剂,
       分侧 A/B 指标                                     ← 需 Linux (本地无 data cache)
  0.3  推理消融: P5-nobypass, P4-noresponse, placebo(错侧阀位)  ← 需 Linux, 零训练
  0.4  R1 形状模型选择 (1/2/3 极点 vs 幂律 vs 线性)          ← 本地可做

RM3-1  (小规模训练, 验证根因)
  1.1  恢复 structure penalty (logged-aux, 权重 0.10), 单 fold 单 seed,
       检查 predicted_effect 是否回到 ~0.19 °C
  1.2  阀位解码器: Δvalve 损失 + 粗糙度匹配 + (SP−T) PID 特征输入   [= RM3-B A4]
  1.3  P3+bypass / P5-nobypass 正式训练对照 (决定 RM3-B 架构基线)
  1.4  收敛性诊断: 单 fold 单 seed 跑 12000 updates
  1.5  P3-C: 切断 residual_head 的动作相关输入 (Phase 3 遗留)

RM3-B  (按现设计, 但以 RM3-1 结论重选基线与前置条件)
```

RM3-0 全部是零训练回放, 不消耗 GPU, 且会改变 RM3-B 的矩阵设计。**不建议在 RM3-0 与 RM3-1.1/1.3 之前冻结 RM3-B 矩阵。**

### 6.4 对 §12 开放决策点的意见

| # | 意见 |
|---|---|
| 1 (E1 阈值 >2×) | 比值门不够, 必须并列幅值下限门(见 §6.2-2)。>2× 可从 RM2 四算子 0.319/1.088/0.435/0.493 反推: 极差 3.41×, 若阈值取 2× 则 RM2 判 FAIL — 这与 RM2 已有的 `OPERATOR_GAIN_NOT_IDENTIFIED` 判决一致, 阈值合理。 |
| 2 (KCI α) | 同意 α=0.05 + 中位数带宽; 但先诊断级一个批次的安排是对的, 建议诊断批次直接用 RM3-0 的冻结产物, 不新训。 |
| 3 (E4 门槛) | 见 §6.2-4, 条件数不可作主指标。 |
| 4 (首批 A1+A4+A3) | 建议改为 **A4 优先且单独先行**。A1/A3 依赖响应分支有激励(§6.2-3)。 |
| 5 (回放是否算新 Gate) | 建议算独立批次 RM3-0, 因为它会修改已发布数字(§5.5)并可能推翻 RM3-A 结论(§4.2), 影响面超出"回放"。 |
| 6 (P0c oracle-local 注入) | 支持; 注意与 §5.1 的循环论证问题区分 — P0c 是**误差归因**不是识别证据, 措辞需明确。 |
| 7 (A5 降级措辞) | 同意保守处理; 参考 `CAUSAL_REPRESENTATION_PAPER_READING_MAP.md` §4 表格的表述边界, 建议直接复用。 |

---

## 7. 复现命令

```powershell
# 命令一 — §1.3 响应分支幅值 (需要 checkpoint, 无需 data)
python - <<'PY'
import torch,torch.nn.functional as F,numpy as np
r='results/phase3_5/ms3r_rm3/prediction/P4_gatec_a1_scheduled_F0_s0'
sd=torch.load(r+'/checkpoint_best_validation.pt',map_location='cpu',weights_only=False)['model_state_dict']
d=np.load(r+'/episodes_validation.npz')
o='model.local_response.operator.'
opw=F.softmax(sd[o+'opening.power_logits'],0)
op=lambda v:(lambda n:opw[0]*n+opw[1]*n**2+opw[2]*n**3)((v/100).clamp(0,1))
dg=F.softplus(sd[o+'diagonal_gain'])+1e-3
cg=0.35*torch.sigmoid(sd[o+'cross_gain_logits'])*dg.flip(0)
mix=torch.stack((torch.stack((dg[0],cg[0])),torch.stack((cg[1],dg[1]))))
dec=torch.exp(-10/(20+1180*torch.sigmoid(sd[o+'tau_logits'])))
w=F.softmax(sd[o+'pole_weights'],1)
for nm,key in [('predicted','valve_prediction'),('logged','logged_valve')]:
    v=torch.from_numpy(d[key]); dose=op(v)-op(v[:,0])[:,None,:]
    eq=torch.einsum('bhi,oi->bho',dose,mix)
    md=torch.stack((.5*(eq[...,0]+eq[...,1]),.5*(eq[...,0]-eq[...,1])),-1)
    st=torch.zeros(len(v),2,3); out=[]
    for k in range(v.shape[1]):
        st=dec[None]*st+(1-dec[None])*md[:,k].unsqueeze(-1)
        m=(w[None]*st).sum(2); out.append(torch.stack((m[...,0]+m[...,1],m[...,0]-m[...,1]),-1))
    print(nm,'mean|effect| = %.4f C'%torch.stack(out,1).abs().mean())
PY

# 命令二 — §2 阀位塌缩 + 持续技巧
python - <<'PY'
import numpy as np,os
b='results/phase3_5/ms3r_rm3/prediction'
for c in sorted(os.listdir(b)):
    if not c.endswith('F0_s0'): continue
    d=np.load(f'{b}/{c}/episodes_validation.npz')
    t,p=d['terminal_target'].astype(float),d['terminal_prediction'].astype(float)
    sk=[1-np.abs(p[:,k]-t[:,k]).mean()/np.abs(t[:,k]-t[:,0]).mean() for k in (11,29,59)]
    v=''
    if 'valve_prediction' in d:
        vp,lv=d['valve_prediction'],d['logged_valve']
        v=' valveMAE=%.3f persist=%.3f dstep=%.3f/%.3f'%(np.abs(vp-lv).mean(),
          np.abs(lv-lv[:,0:1]).mean(),np.abs(np.diff(vp,1,1)).mean(),np.abs(np.diff(lv,1,1)).mean())
    print('%-30s MAE=%.4f skill %.3f/%.3f/%.3f%s'%(c,np.abs(p-t).mean(),*sk,v))
PY

# 命令三 — §4 RM3-A 配对差 + 自由头容量
python - <<'PY'
import json,os,torch,statistics as st
M={}
for pp in ('results/phase3_5/ms3r_rm3/prediction','results/phase3_5/ms3r_rm3a'):
    for r in sorted(os.listdir(pp)):
        f=f'{pp}/{r}/metrics_validation.json'
        if os.path.exists(f):
            m=json.load(open(f,encoding='utf-8')); M[(m['candidate_id'],r[-6:])]=m['metrics']
ks=sorted({k for _,k in M})
for a,b in [('A0_p3_large','P3_gatec_paired_free'),('A0_p3_large','P5_hybrid_joint_latent'),
            ('A2_p5_small','P3_gatec_paired_free'),('A4_p5_local50','P5_hybrid_joint_latent')]:
    for f in ('terminal_mae_c','local_mae_c','valve_mae'):
        d=[M[(a,k)][f]-M[(b,k)][f] for k in ks if (a,k) in M and (b,k) in M]
        print('%-14s-%-22s %-14s %+.4f (%d/%d)'%(a,b,f,st.mean(d),sum(x<0 for x in d),len(d)))
for pp,c in [('results/phase3_5/ms3r_rm3a','A0_p3_large'),('results/phase3_5/ms3r_rm3a','A4_p5_local50')]:
    sd=torch.load(f'{pp}/{c}_F0_s0/checkpoint_best_validation.pt',map_location='cpu',weights_only=False)['model_state_dict']
    print(c,'residual_head=%d bypass=%d bypassW=%.2f'%(
      sum(v.numel() for k,v in sd.items() if 'residual_head' in k),
      sum(v.numel() for k,v in sd.items() if 'terminal_bypass' in k),
      max([v.norm().item() for k,v in sd.items() if 'terminal_bypass.network.2.weight' in k]+[0])))
PY

# 命令四 — §5.4/§5.5 形状不可辨识 + 校准存档错值
python - <<'PY'
import numpy as np,os,json
from itertools import combinations
def nn(D,t):
    k=D.shape[1];b=np.zeros(k);e0=t@t
    for s in range(1,k+1):
        for a in combinations(range(k),s):
            c=np.linalg.lstsq(D[:,a],t,rcond=None)[0]
            if (c<-1e-12).any(): continue
            x=np.zeros(k); x[list(a)]=np.maximum(c,0); e=((D@x-t)**2).sum()
            if e<e0: b,e0=x,e
    return b
def rm(M,B):
    f=np.empty_like(M)
    for i in range(2):
        for j in range(2): f[:,i,j]=B@nn(B,M[:,i,j])
    return np.sqrt(((f-M)**2).mean())
p='results/phase3_5/ms3r_rm3/calibration'
for r in sorted(os.listdir(p)):
    M=np.load(f'{p}/{r}/orthogonal_residuals_validation.npz')['r0_trajectory_matrix']
    t=(np.arange(len(M))+1)*10.
    E=lambda ts:1-np.exp(-t[:,None]/np.array(ts)[None])
    stored=json.load(open(f'{p}/{r}/calibration_validation.json',encoding='utf-8'))['results']['R1_a1_scheduled']['projection_rmse']
    print('%-22s stored=%7.4f | 3p=%.4f 1p=%.4f pow=%.4f lin=%.4f'%(r,stored,
      rm(M,E([60,180,600])),rm(M,E([180])),rm(M,(t[:,None]/600)**np.array([.5,1,1.5])[None]),
      rm(M,(t[:,None]/600)**np.array([1.])[None])))
PY

# 命令五 — §3 RM2 诊断集(证明 RM3 丢失了什么)
python - <<'PY'
import json,os,collections,statistics as st
p='results/phase3_5/ms3r_gatec_rm2'; g=collections.defaultdict(list)
for r in sorted(os.listdir(p)):
    f=f'{p}/{r}/metrics_validation.json'
    if os.path.exists(f): g[r[:-6]].append(json.load(open(f,encoding='utf-8'))['metrics'])
sel=['predicted_effect_mean_abs_c','logged_effect_mean_abs_c','logged_vs_shuffled_local_advantage_c',
     'forecast_local_mae_c','persistence_terminal_mae_c','terminal_to_persistence']
print('%-26s '%'cand'+' '.join('%12s'%k[:12] for k in sel))
for c,v in sorted(g.items()):
    print('%-26s '%c+' '.join('%12.4f'%st.mean(x[k] for x in v) for k in sel))
PY
```

---

## 8. 与 Phase 3 的连续性

Phase 3 (`docs/session_2026-08-06_full_review.md`) 的三条结论在 Phase 3.5 全部复发:

| Phase 3 结论 | Phase 3.5 现状 |
|---|---|
| `f_free` 抢动作信号, 干预分支欠增益 0.65 | 同一机制; RM3 响应分支降到残差的 **1.7%** (§1.3 / §5.2) |
| freeze / MAE loss 无效, 正确方向是切断自由通路输入 | P3-C **仍未执行**(RM3-1.5) |
| 缺 naive 基线, MAE 无参照 | RM2 补齐过, **RM3/RM3-A 又丢失** (§3) |

**RM2 → RM3 的倒退模式值得单独记录**: 一个批次补齐的诊断设施, 在下一个批次的重构中被静默丢弃, 且没有任何门捕获这一点。建议把 RM2 诊断集(persistence / effect magnitude / placebo / 分侧) 提升为**跨批次强制产物契约**, 写进 `execution_contract.required_run_artifacts` 的同级位置。这比 RM3-B 新增的任何一个门都更能防止同类回归。
