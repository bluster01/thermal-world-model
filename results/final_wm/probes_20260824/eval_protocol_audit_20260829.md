# 评估口径与记录迁移审计（execution-side, 2026-08-29）

> 用户三问（基线 MAE 为何变低 / NLL 降而 MAE 不降 / 是否未做全量 OOF）
> 的事实回答 + 推理速度实测。本文件为归档事实，不作新判据。

## 1. 基线 H18 为何 0.484 → 0.426

v2.2 记录（`canonical_sideA_v2.npz`，meta `version=2.2`）声明
`v1_keys_verbatim = [boundary, obs, valid, timestamps, split]`——七通道数据逐字节
继承 v1，**唯独 actions 被替换**：

- **valve1（一级减温阀）在 v1 为错侧缺陷**：`results/final_wm/known_defect_v1_valve1_20260826.md`
  写明"双侧均错配到对侧"，旧通道含 ~20% 对侧噪声；v2.2 重新接线后
  `actions_continuity.valve1: corr_with_v1=0.784, mae_vs_v1=0.098`。
- valve2 本就正确（corr 0.99999，MAE 2.7e-5）。
- 另有 A5 质量门滤 ~5% 窗口（`unit_load > 160 MW`、`1 < water_coal_ratio < 8`、
  `fuel_corrected > 50 t/h`）+ 前 12 行 edge trim。

口径（`a5_water_coal_preimplementation_audit_20260828.md` §2.1）：旧 0.484/1.13×
**只保留作外部锚**；同一筛选记录上的匹配对照（0.4262）才是因果差值分母。
0.484→0.426 的三项构成（缺陷补丁 / 质量门 / trim）**未消融**，各自贡献未知。

## 2. val NLL 反复降而 H18 MAE 不降的机制

- 训练 loss 本身就是 Gaussian NLL（`training.py:219` `observation_nll(temps_mu,
  temps_sigma, future_obs)`），早停也按 val NLL。**优化器从未见过 MAE**。
- NLL = (err/σ)² + log σ²（至常数）：均值误差到残差底（train 残差 std
  0.1346°C；未观测炉膛侧驱动为信息极限的 v0.6 诊断）后，继续降 NLL 的唯一
  通道是 σ 校准（误差大处放 σ、小处收 σ），MAE 对 σ 无感。
- zcond A 实例：val NLL 1.039@86 vs 对照 1.114@98（明显更好），H18 +1.4%
  （差异 0.006°C ≈ 0.7 SE，噪声级）→ 收益疑似在 σ 校准不在点精度。
- **待实证**：把两臂 NLL 分解为 (err/σ)² 项与 log σ² 项，随 §3 的 OOF 扫一并做。

## 3. 评估口径缺口：未做全量 OOF（已确认）

- 冻结协议（PREREG §0）：主指标 = 256 随机窗（seed 50k）H18 MAE。val 切分 =
  15%（106,156 行），随机起点的窗采样，非 stride 滑窗、非交叉折。
- 统计效力：整体均值 SE ≈ 0.135/√256 ≈ 0.008°C（够紧）；分箱每箱 ~51 窗（薄），
  regime 切片更薄。
- **计划**：B 臂训完后，控制 + A + B 三臂**同一批窗口** stride 滑窗扫全量 val
  + NLL σ 分解，作为 256 窗协议的稳健性校验。判决口径仍以冻结协议为准，
  OOF 结果作证据层，不改判据。

## 4. 推理速度实测（2026-08-29；与 B 训练共享 GPU，绝对数偏保守，比例可靠）

| 批大小 | 实测 | 说明 |
|---|---|---|
| batch 1 | 366 ms/窗 | 部署口径（单窗预测） |
| batch 32 | 12 ms/窗 | 训练/协议评估口径 |
| batch 256 | 1 ms/窗 | 全量扫描口径 |

- 单窗/批32/批256 比例 = 31.7× / 9.1× → **瓶颈是批利用率，不是计算**；
  单窗含争抢 ≤0.4 s，远小于 DCS 10 s 周期，部署不存在推理速度问题。
- `torch.compile(_substep)` = 1.0×（且有 recompile 警告）：子步开销不是瓶颈，
  不追编译。
- 训练侧 `timing.eval_s ≈ 102–111 s` 为**全程累计**（每 epoch 评估 128 窗、
  seed=10000+epoch 逐轮变动、batch 32），约 8 ms/窗/epoch，与本节实测一致。
  此前按"单次 256 窗成本"解读属误读，由此推出的 OOF 3–4 h 估计作废。
- **全量 OOF 成本修正**：val 106,156 行、stride 9 ≈ 1.18 万窗，batch 256
  实测 1 ms/窗 → **1–2 分钟量级**（加数据采样/落盘开销，仍为分钟级）。
