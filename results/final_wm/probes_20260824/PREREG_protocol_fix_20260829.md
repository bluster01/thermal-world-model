# 固定协议 spec（PREREG, execution-side, 2026-08-29）

> 用户指令(2026-08-29)：协议 spec 先固定，所有未按此协议跑过的实验一律重跑，
> 公平比较。本文件是重跑的唯一协议来源；**旧裁决（v1fixed 记录下的一切数字）
> 自本文件起降级为 HISTORICAL**，不得与新协议数字混拼。
> 预注册纪律：本文件先定死，跑完只填结果，不事后改判据。

## 1. 记录（唯一合法数据源）

- `artifacts/final_wm/canonical_sideA_v2.npz`（version=2.2）
  - valve1 动作通道为**修正列**（known_defect_v1_valve1_20260826.md：
    v1 双侧错配对侧、~20% 对侧噪声；v2.2 corr_with_v1=0.784）
  - boundary/obs/split 逐字节继承 v1（meta: v1_keys_verbatim）
- **有效工况门**（A5 预注册口径）：`unit_load > 160 MW`、
  `1 < water_coal_ratio < 8`、`fuel_corrected > 50 t/h`；窗口必须完整落于
  连续有效段。7 通道视图 = `WaterCoalRecord.base_boundary`（BaseRecordView）。
- 该门滤除 ~5% 窗口 → **同筛选记录上的匹配对照** = 唯一因果分母
  （`t1_a5_filtered_control_seed0`，checkpoint 复用不重训）。

## 2. 训练 spec（全部臂统一）

| 项 | 值 |
|---|---|
| unit/arm/seed | t1 / <arm> / 0（探索性；晋级后补 1/2）|
| boundary_mode | oracle |
| initial_state_mode | hybrid（A2 消融除外，见其预注册目标）|
| closure_mode | conservative_norew |
| epochs/patience/batch/bpe/lr | 120 / 20 / 32 / 200 / 1e-3 |
| 物性 | GridThermoProperties（assert_grid 前置 + ledger 回读）|
| 训练记录 | BaseRecordView（7 通道，与对照同数据）|

## 3. 评估（双口径，判读规则先定）

- **判决口径**：256 窗（seed 50k）H18 MAE ch4 + Q1-Q5 分箱极差 + 方向门 v0.3
  （均值终端 ΔT<0、day-block CI<0、正方向占比≥0.60，H18/H60 同报）
- **判读基准**：全量 val OOF（stride 9，batch 256，同窗三臂）。2026-08-29
  审计证实 256 窗样本把效应放大至 4.6×——**幅度接近噪声带的结论必须以
  OOF 为准**；250 窗判决口径不变但解释必须引用 OOF。
- **门**：晋级 = H18 ≤ −5% 且分箱 ≤ +10% 且方向过；否决 = H18 ≥ +5% 或方向不过；
  不确定 = 其余。单种子全部标 "EXPLORATORY"。

## 4. 重跑队列（用户指令，全部按 §1-§3）

| 臂 | 机制源脚本 | 旧记录结果（HISTORICAL）| 状态 |
|---|---|---|---|
| A1 | lpv_uaphys_corrected_probe（α_τ=0, α_UA=0.8 钉死）| UA +12.2% | 排队 |
| A2 | a2_steady_init_probe（initial_state_mode=steady）| +60.7% | 排队 |
| A4 | a4_k_schedule_probe（k 负荷调度单旋钮）| −7.6%（分箱 1.48×）| 排队 |
| A6 | a6_pressure_schedule_probe（压力双调度）| +1.2% | 排队 |
| A46 | a46_joint_schedule_probe（A4+A6）| +27.7% REJECT | 排队 |
| A9 | lpv_schedule_probe --mix-only（τ_mix 仅）| +22.5% | 排队 |
| LPV-free | lpv_schedule_probe（自由 τ/UA 调度）| 7.788（作废口径）| 排队 |

- 分母统一复用 A5 匹配对照（0.4262@256窗 / 0.4340@OOF），不重训。
- 每臂先过 identity 门（w=0 / 机制关闭时位级恒等）再训；训完 256 窗 + OOF
  双口径出表，报告落 `results/final_wm/probes_20260824/protocol_rerun/<arm>/`。
- 判定函数与 A5/zcond 完全一致；本文件锁定后不改。
