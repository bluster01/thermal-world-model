# 最终世界模型判别实验矩阵 v0.1（冻结稿）

> 状态：**FROZEN MATRIX v0.1 / 待独立授权提交**。本文件冻结 O1/B1/T1/R1/J1 判别实验与前置 D0/D-SYN
> 门禁的假设、数据合同、预算、判决规则与禁止事项。K1 为条件实验，母模型未过门禁前保持 HOLD。
>
> 上游依据：[pipeline 设计稿](2026-08-18-final-world-model-pipeline-design.md) §6 初始矩阵、
> [实现记录](2026-08-18-final-world-model-implementation.md)、RM3-B1 审计判决。
> 本文只冻结矩阵；任何执行需独立授权提交，Linux 侧不得改代码/配置/阈值，不访问 test。

## 1. 数据输入合同（D0 前置）

数据源：伊敏 6 号机（660 MW 超临界）DCS 导出，Linux 侧 `/home/bluster/Desktop/AI` 目录树
（仓内 `data/伊敏6号机` 为路径占位符，数据不入仓）。

### 1.1 必需通道（与 `src/final_wm/contracts.py` 注册表一一对应）

| 注册表通道 | 物理量 | 单位 | 来源标签 |
|---|---|---|---|
| `steam_flow` | 主蒸汽流量 D | kg/s | 边界 |
| `coal_command` | 给煤指令 uB | t/h | 边界 |
| `separator_pressure` | 分离器出口压力 pm | MPa | 边界 |
| `separator_temperature` | 分离器出口温度 Tm_sep | degC | 边界 |
| `feedwater_temperature` | 喷水/给水温度 Tfw | degC | 边界 |
| `outlet_pressure` | 过热器出口压力 p_out | MPa | 边界 |
| `spray_flow_total` | 减温水总流量 W（oracle-only） | t/h | 边界（诊断） |
| `valve1_position` / `valve2_position` | 一级/二级减温水调节阀阀位 | 0..1（0-100% 归一化） | 动作 |
| 5 个 `OBSERVATION_ELEMENTS` | 屏过入/出口、高过入/出口、末过出口汽温 | degC | 观测 |

D0 必须产出**点位映射审计表**（DCS 点名 → 注册表通道 → 单位换算 → 可信度），未闭合的通道
映射直接判 MIXED 并阻断对应实验。RM3-R 审计遗留的 tag 证据缺口在此一次性收口。

### 1.2 质量门（fail-closed）

- 采样周期 10 s，时间戳连续性缺口率 < 1%；
- 每个通道的饱和/卡死段（连续 > 30 min 零方差）占比 < 5%；
- 阀位非饱和段（v1、v2 不同时长时间贴 0/100%）覆盖 ≥ 60%，否则动作通道激励不足，T1/R1 不得启动；
- 可用连续时长 ≥ 30 天等效；不足则由 D0 如实回传并缩小矩阵，不补数据、不降门槛。

### 1.3 切分与 canonical records

- 按时间顺序切 train / validation；**test 继续锁定**，本矩阵全部判决只用 validation；
- D0 生成唯一 canonical record（10 s 对齐、单位归一、通道按注册表序打包），后续所有实验只读
  canonical record；窗口化在线完成，不物化大窗口张量；
- 切分点、缺口掩码、映射表随产物回传，本地独立审计。

## 2. 实验单元

### D-SYN：同型可解性门禁（先行，真实数据前必过）

- 内容：`tests/final_wm/test_micro_smoke.py` 的扩展版 —— 同型 teacher 生成已知真值轨迹，
  observer+transition+closure 联合训练须恢复 teacher 温度轨迹（H18 NLL 相对零初始化骨架
  下降 ≥ 30%，3 seeds ≥ 2/3 过门）。
- 判决：FAIL 则整个矩阵 HOLD，回本地修接口；不进入真实数据。

### O1：初态后验

- 假设 H_O1：learned 后验（GRU q(x0|H)）相对观测锚定稳态初始化（steady 臂）在验证集上显著
  改善短期预测，且不破坏相邻窗口状态连续性。
- 臂：`steady` / `learned` / `hybrid`（learned 均值与 steady 的精度加权融合）。
- 度量：H6/H18 观测 NLL；相邻窗口 state-continuity 误差（归一化 L2）。
- 判决：H18 NLL 改善 ≥ 5%（UTC-day block bootstrap 95% CI 下界 > 0）且 continuity 误差不劣于
  steady 臂（CI 上界 ≤ steady 臂点估计）→ learned/hybrid 留用；否则回退 steady 臂并记 REJECTED。

### B1：边界预报

- 假设 H_B1：GRU 边界模型相对持久性基线（persistence: 未来=历史末值）显著改善边界预报，
  且 forecast 边界相对 oracle 的下游 rollout 退化有界。
- 度量：7 通道 H6/H18/H36 NLL/CRPS；下游 H18 温度 NLL 退化 Δ = NLL_forecast − NLL_oracle。
- 判决：H18 CRPS 改善 ≥ 3%（CI 下界 > 0）→ 留用；否则回退 persistence 边界并记 MIXED/REJECTED。
  Δ 仅作报告量，不设门槛（退化量级本身是发布证据）。

### T1：transition 结构消融

- 假设 H_T1：physics-only 骨架之外的结构（conservative closure / steam-only closure /
  latent block）只有在带来显著且稳定的验证收益时才保留。
- 臂：`physics-only` → `+closure(conservative)` → `+closure(steam_only)` → `+latent(4)`，
  嵌套比较。
- 度量：H1/H6/H18 验证 NLL；60 步定常 rollout 有界性（|drift| 与 settle，同本地合同口径）。
- 判决：每一层相对前一层 H18 NLL 改善 ≥ 2%（CI 下界 > 0）且 ≥ 2/3 seeds 过门才保留；
  任一臂 rollout 发散（非有限或 |drift| > 60 °C）直接 REJECTED 该臂。
- 负对照：`closure(conservative)` 与 `closure(steam_only)` 差异不显著时优先 conservative
  （能量守恒先验），并在报告中声明该选择是先验驱动而非数据驱动。

### R1：closure 因果与量级门

- 假设 H_R1：action-blind closure 吸收的是真实未测扰动而非动作/阀门因果泄漏。
- 检查项：
  1. 运行时 action-blindness（同本地合同测试，真实权重下复跑）；
  2. 残差功率分位数报告（p50/p90/max，kW），相对典型段级热流（≈1e5 kW）占比；
  3. 阶跃响应方向一致性审计：closure 开启/关闭两种配置下，v2 阶跃 +0.05 的长期终端温度
     响应必须为负；任一配置翻号 → closure REJECTED（E4 防线）；
  4. 负对照：action-aware closure 变体若显著优于 action-blind（H18 NLL 改善 > 5%，CI 下界>0），
     判 closure 存在动作泄漏，整臂 REJECTED，回报残差结构供本地归因。
- 判决：四项全过 → SUPPORTED；方向翻号或泄漏坐实 → REJECTED；其余 → MIXED。

### J1：联合端到端 vs 分阶段训练

- 假设 H_J1：联合 NLL 目标（observer+boundary+transition+closure+observation 端到端）相对
  分阶段训练不劣。
- 度量：H18 验证 NLL + 长 rollout（H36）稳定性；选择器口径同 MS5（H1/H6/H18 NLL/MAE/CRPS 全量报告）。
- 判决：joint H18 NLL 改善 ≥ 3%（CI 下界 > 0）且稳定性不劣 → joint 为训练主路线；
  否则分阶段留用。单 seed/单 fold 结果不产生路线冠军（MS5 纪律）。

### K1：Koopman student（条件 HOLD）

- 仅当 O1、T1、R1、J1 全部有留用判决后解冻；届时以单独冻结提交定义算子阶数、损失与
  student-vs-parent 偏差门。本矩阵不授权 K1 任何执行。

## 3. 公共判决纪律

- 全部判决只报 validation；**不访问 test**；
- 区间估计用 UTC-day block bootstrap（≥ 1000 次重抽样），报点估计 + 95% CI；
- 四态判决：SUPPORTED / MIXED / REJECTED + 规模护栏（<1% 或 <0.1 °C 只作尺度参考）；
- seeds ≥ 3，≥2/3 过门才算过门；预算内失败原样回传，不补阈值、seed 或超参扫描；
- 反事实指标只在动作支持域内计算；支持域外步骤必须带 `in_support=False` 标记上报
  （RM3-B1 §3.4 审计要求）；
- 每次运行保存 config hash / 代码 commit / 数据 SHA / 机器环境，产物只读追加。

## 4. 预算（Hermes，单次授权提交）

| 单元 | 规模 | 预算上限 |
|---|---|---|
| D0 数据审计 + canonical record | 1 次 | 2 h CPU |
| D-SYN 同型门禁 | 3 seeds | 1 h GPU |
| O1（3 臂 × 3 seeds） | 9 runs | 6 h GPU |
| B1（模型+基线 × 3 seeds） | 6 runs | 4 h GPU |
| T1（4 臂 × 3 seeds） | 12 runs | 12 h GPU |
| R1（复用 T1 权重 + 负对照 3 runs） | 3 runs + 审计 | 3 h GPU |
| J1（joint vs staged × 3 seeds） | 6 runs | 8 h GPU |
| **合计** | | **≤ 36 GPU 小时** |

## 5. 明确禁止事项

- 不访问 test；不生成 B2 式重试批次；不复活旧 runner；
- 不把 Direct-WM/物理分支历史数值当本矩阵先验之外的证据；
- 不在支持域外报告反事实结论；不允许以 observational fit 冒充 do-calculus 干预；
- 不修改 `src/final_wm/` 接口与合同以迎合数据；数据问题回 D0 如实判决；
- K1、MS4、论文写作继续锁定。
