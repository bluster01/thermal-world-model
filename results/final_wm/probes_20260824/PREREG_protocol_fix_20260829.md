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

| 臂 | 机制源脚本 | 旧记录结果（HISTORICAL）| 固定协议终审（2026-08-30）|
|---|---|---|---|
| A1 | lpv_uaphys（α_τ=0, α_UA=0.8 钉死）| UA +12.2% | **REJECT** 256 +22.7% / OOF +23.4% |
| A2 | a2_steady_init（steady 消融）| +60.7% | **REJECT**（消融臂：NN 初值修正价值）256 +71.8% / OOF +77.8% |
| A4 | a4_k_schedule（k 负荷单旋钮）| −7.6%（分箱 1.48×）| **REJECT** 256 +24.7% / OOF +23.1% |
| A6 | a6_pressure（压力双调度）| +1.2% | **REJECT** 256 +23.9% / OOF +30.9% |
| A46 | a46_joint（A4+A6）| +27.7% REJECT | **REJECT** 256 +17.5% / OOF +13.7% |
| A9 | lpv --mix-only | +22.5% | **REJECT** 256 +8.6% / OOF +10.1% |
| LPV-free | lpv_schedule（自由 τ/UA）| −3.6%（旧口径）| **REJECT** 256 +29.1% / OOF +30.4% |

## 5. 重跑终表与裁定（2026-08-30）

全部 7 臂与 zcond 系合并后的**固定协议矩阵**（对照 = 匹配筛选控制，
本批内 256w 0.4259 / OOF 0.4338；采样自 7 通道 view，与 A5 逐位 0.4262
相差 <0.1% 属采样路径差异，批内可比性不受影响）：

| 被调参数 × 条件 | 256w ΔH18 | OOF ΔH18 | 判决 |
|---|---|---|---|
| k × 负荷（A4）| +24.7% | +23.1% | REJECT（旧 −7.6% 翻转）|
| k × z（zcond A）| +22.7% | +28.2% | REJECT |
| k+压τ × 联合（A46）| +17.5% | +13.7% | REJECT（确认）|
| 压力τ（A6）| +23.9% | +30.9% | REJECT（旧 +1.2% 翻转，OOF 最毒）|
| UA × D-B 物理（A1）| +22.7% | +23.4% | REJECT（旧 +12.2% 翻转）|
| τ_mix 仅（A9）| +8.6% | +10.1% | REJECT（确认）|
| τ+UA × 自由 α（LPV-free）| +29.1% | +30.4% | REJECT（旧 −3.6% 翻转）|
| k+τ+UA × z（zcond B）| −3.6% | −0.78% | INCONCLUSIVE（OOF 噪声级）|
| 初值 × steady 消融（A2）| +71.8% | +77.8% | 消融值：NN 初值修正价值巨大 |

**裁定**：
1. **固定协议下 8/9 调度臂全 REJECT**，唯一未 REJECT 的是 zcond B（OOF 噪声级）。
   旧记录"有点信号"的三臂（A1/A4/A6/LPV-free）全部翻转——valve1 错侧缺陷
   曾洗出假信号。调度不是本问题瓶颈的结论以两倍证据坐实。
2. **变工况信息走状态不走参数**：A2 消融（+71.8%）与 a4 起所有调度臂
   （≤+30.9%）对照，唯一赚的路径是 encoder→初值。
3. 风煤比第四旋钮仍待对侧通道；若通道落地，其预期从本矩阵外推为
   REJECT 或 INCONCLUSIVE（所有显式工况条件器无一存活的先验下）。

- 分母统一复用 A5 匹配对照（0.4262@256窗 / 0.4340@OOF），不重训。
- 每臂先过 identity 门（w=0 / 机制关闭时位级恒等）再训；训完 256 窗 + OOF
  双口径出表，报告落 `results/final_wm/probes_20260824/protocol_rerun/<arm>/`。
- 判定函数与 A5/zcond 完全一致；本文件锁定后不改。
