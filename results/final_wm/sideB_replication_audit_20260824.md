# 侧B 全量复制矩阵审计（2026-08-24）

产物：执行侧 commit `b199d8d`（ledger 21 runs / 907 epoch 条，0 RESUMED）。
T1 判决由设计侧只读重放（`scripts/t1_verdict_sideB_replay.py`，runner 冻结函数
`_seed_passes`/`_verdict` + THRESH_T1_NLL=0.02，不改 runner、不写 summary），
结果落档 `results/final_wm/t1_verdict_sideB_replay_20260824.json`。

## 0. 溯源与完整性

- **commit 混合**：ledger 中 dsyn + t1_physics_only 记 `0922cac`，其余记
  `1b4f874`。已核实 `git diff 0922cac..1b4f874` 仅含
  `scripts/report_per_channel_mae.py`（+81，只读报表工具）——src/experiments
  训练路径零改动，训练完整性不受影响。执行侧 commit message 自述一致。
- **resume**：0 RESUMED，全新训练 21 runs（dsyn×3, t1×9, o1×9）。
- **arm-filter 纪律**：`matrix_summary_sideB.json` 无 t1 判决块（预期），
  r1/o1 判决块由执行侧 runner 写入（r1 三分臂为 `--r1-arm` 独立块，未互相覆盖）。

## 1. D-SYN 门（PASS 3/3，复现）

| seed | skeleton→student NLL | CF-1 terminal sign agreement |
|---|---|---|
| 0 | 316.90 → -0.025 | 1.000 |
| 1 | 312.47 → -0.105 | 1.000 |
| 2 | 417.58 → -0.297 | 0.992 |

侧B 通行证成立。

## 2. T1（设计侧重放）——**跨侧不一致，核心发现**

| 比较 | 侧B verdict | per-seed point (CI_lo) |
|---|---|---|
| closure_cons vs physics_only | **SUPPORTED 3/3** | 0.073(0.042) / 0.068(0.040) / 0.098(0.064) |
| closure_cons_norew vs physics_only | **SUPPORTED 3/3** | 0.103(0.051) / 0.088(0.059) / 0.111(0.081) |
| norew vs intact（参考） | MIXED 1/3 | 0.033 / 0.020 / 0.013 |

**与侧A 终审不一致**：侧A 重发判决为 closure 增益不支持
（norew vs physics_only 1/3 MIXED 且 seed0 为 -0.178 异常值；
closure_cons vs physics_only 0/3）。侧B 上闭包精度增益真实存在且三 seed
稳健（含 norew 臂 8.8-11.1%）。

**判读**：闭包的精度价值是**侧依赖**的——侧B 有增益、侧A 中性。这削弱
"闭包只为响应方向服务"的单侧口径，但不影响生产臂选择：norew 在侧B 上精度
不劣于 intact（MIXED 但全部点估计为正 0.013-0.033），同时是两侧唯一方向
认证臂。论文口径应改为"闭包精度增益侧依赖（B 正 / A 中性），方向认证跨侧
稳定"。

## 3. R1 三分臂对比（复现，核心 punchline 成立）

| arm | 侧B verdict | frac_negative（3 seeds） | mean_delta_c |
|---|---|---|---|
| physics_only | REJECTED | 0.41 / 0.41 / 0.53 | +0.047 / +0.041 / +0.034 |
| closure_cons（intact） | REJECTED | 0.03 / 0.22 / 0.13 | +0.060 / +0.064 / +0.044 |
| closure_cons_norew | **SUPPORTED 3/3** | 1.0 / 1.0 / 1.0 | -0.392 / -0.524 / -0.490 |

侧A 终审（norew 为唯一方向认证臂）在侧B 完全复现，且 norew 冷却幅度更强
（B: -0.39..-0.52 vs A: -0.27..-0.40）。steady 子探针三臂均 frac=1.0
（稳态段方向全对，瞬态段才是区分度来源）——与侧A 结构一致。

## 4. 泄漏门（复现）

- leakdist norew ×3，16-shuffle 零分布，`aware_percentile=1.0`，
  `leakage_suspected=false` ×3。
- seed2 aware improvement 为负（-0.255），复现侧A seed2 摆动（-0.109）——
  已知 seed 级摆动，零分布同步为负（mean -0.271），非泄漏。
- 注意：summary 内嵌 leak 探针 n_shuffles=1（runner 内联版），16-shuffle
  证据由独立 leakdist 文件承载，与侧A 证据结构相同。

## 5. O1（复现，hybrid 幅度减半）

| arm | 侧B point（3 seeds） | 侧A 对照 |
|---|---|---|
| learned | 0.259 / 0.244 / 0.259，CI 全过 | 0.305 / 0.218 / 0.273 |
| hybrid | 0.113 / 0.118 / 0.122，CI 全过 | ~0.241 |

learned 观察者增益跨侧稳定；hybrid 在侧B 减半但判决不变。

## 6. Auditpack（F3 因果归因复现）

- **rewetting ablation（intact 臂）**：mean_delta_c +0.058（frac 0.08）
  → rewet_zeroed -0.293（frac 1.0）。F3 元凶=润湿项，在侧B 独立复现。
- **norew 臂**：intact ≡ rewet_zeroed（-0.381，frac 1.0），冻结语义正确。
- **position_binned_gain**：intact 模型 v1/v2 全 bin 增益为正（+0.5..+2.1，
  升温方向，错误）；norew 全 bin 为负（-3.2..-9.3，冷却方向，正确）。
  数据侧增益符号噪声大（v1 高 bin n≤2），仅作方向参考。
- **单调性**：两臂 monotone_cooling=true（120 步 rollout，Δv 网格
  0.01-0.1 终端冷却单调递增）。
- **校准**：两臂 H18 95% 覆盖缺口均 ~0.084，与侧A 同量级。

## 7. 终审结论

**复现成立（跨侧稳定）**：D-SYN 通行证、R1 方向排他性（norew 唯一认证臂）、
F3 润湿项归因、O1 learned/hybrid 增益、泄漏门清白、单调性与校准量级。

**复现不成立（侧依赖）**：T1 闭包精度增益——侧B SUPPORTED 3/3 vs 侧A 不支持。
属科学发现而非工程事故：论文须按侧分别报告，禁止聚合为单一判决。

**生产臂维持**：closure_cons_norew——两侧唯一方向认证、侧B 精度增益最大臂、
侧A 精度中性。v0.4 终审裁定不受侧B T1 结果影响。
