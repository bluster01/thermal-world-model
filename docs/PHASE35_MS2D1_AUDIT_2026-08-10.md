# Phase 3.5-MS2-D1 本地审计（2026-08-10）

## Material Passport

- Material Type: checkpoint & parameter & log audit for MS2-D1 validation artifacts
- Scope: `results/phase3_5/ms2d_delay/` 18 runs 的 checkpoint、manifest、history
- Verification Status: VERIFIED（本审计）
- Evidence Boundary: 与 MS2-J 审计同款——只审计产物健康与语义，不扩展现场主张

## 1. 审计判决

**MS2-D1 validation 产物可信，建议批准下一步（Codex 决定：是否授权一次性 test、是否启动 D2 三阶惯性）。**

三层审计全部健康：
1. checkpoint 可加载性 18/18，零 NaN/Inf
2. oracle 参数恢复语义正确（四重确认）
3. 训练日志无异常（收敛正常、无退化）

## 2. checkpoint 健康

| 项 | 结果 |
|---|---|
| 可加载 | 18/18（torch.load weights_only=False, cpu） |
| NaN/Inf | 0 个参数张量 |
| 缺文件 | 无 |
| 归档 | `ms2d_d1_checkpoints_validation.tar` 18 文件，SHA `7ee6393d...`，manifest 匹配 18/18 |

## 3. 参数语义

### oracle_delay（R50 + fixed 2-step delay 正对照）

| seed | raw_gain | fixed_delay_weights | gain_schedule 常数项 |
|---|---|---|---|
| 0 | -2.274 | [0, 0, 1.0, 0] | 0.653 |
| 1 | -2.279 | [0, 0, 1.0, 0] | 0.664 |
| 2 | -2.279 | [0, 0, 1.0, 0] | 0.664 |

- `fixed_delay_weights=[0,0,1,0]` → one-hot 精确落在 d=2（真值 20s=2 steps）✅
- `raw_gain=-2.274` → `-softplus(-2.274)=-0.103` ≈ 真值 K0=-0.10（R50 非线性下界），与 MS2-J 审计一致（Codex 修正口径：softplus 非 exp）✅
- 优化链可解性**第四重确认**（MS1 → MS2-V/C → MS2-J → MS2-D1 oracle）

### learned_delay（主模型，causal simplex）

| seed | w0 | w1 | w2 | w3 | w4 | E[d] | ±1step 质量 |
|---|---|---|---|---|---|---|---|
| 0 | 0.215 | 0.048 | 0.270 | 0.262 | 0.206 | 2.195 | 0.579 |
| 1 | 0.267 | 0.057 | 0.250 | 0.231 | 0.195 | 2.030 | 0.538 |
| 2 | 0.253 | 0.060 | 0.275 | 0.227 | 0.186 | 2.034 | 0.561 |

- 权重跨 seed 结构一致（w1 显著小 ≈0.05，质量弥散在 {0,2,3,4}）→ 不是单一 seed 的偶然
- **语义解释**：simplex 表达"容量 + 分布式补偿"，不是唯一的 2-step 尖峰。与 MS2-V `K/phi` 补偿同族：响应可恢复、物理参数不唯一。
- 这正是设计 §4.4 预注册的预期分支："响应 Gate 通过但参数诊断失败时，只能声称 delay capacity 有效，不能声称真实迟延已恢复"

## 4. 训练日志

| candidate | seeds | best_val（range） | 说明 |
|---|---|---|---|
| no_delay | 3 | 0.0161–0.0163 | 251/300/300ep |
| learned_delay | 3 | 0.0160–0.0163 | 251/279/300ep |
| oracle_delay | 3 | 0.0159–0.0161 | 232/238/300ep |
| deeponet | 3 | 0.0160–0.0162 | 285/289/300ep |

- 全部正常收敛，best/last 差值 <0.001（无退化）
- 251ep 是正常收敛（early-stop 触发），非故障（与 MS2-J s1 同类）

## 5. 判定与建议

1. **产物可信**：validation 汇总（oracle 0.021<0.05、learned vs no-delay 20.3–23.1%、E[d]≈2.03–2.20）可作审计输入。
2. **建议 Codex**：
   - 若授权 test：主对比 = learned_delay vs no_delay（paired episode bootstrap CI），oracle 复现为第二判据；参数诊断保持单列不阻塞。
   - D2（三阶串联惯性）在 D1 审计通过后按序推进，不并行。
3. 不启动 D2/test 前，synthetic 矩阵保持现状。

## 6. 边界

- 全部仅属 synthetic known-truth；不授权现场 E3/E4、真实阀门曲线或现场因果主张。
- deeponet（0.0350）数值最低仍只作 secondary representation reference。
