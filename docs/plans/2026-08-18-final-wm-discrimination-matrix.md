# 最终世界模型判别实验矩阵 v0.2（冻结稿）

> 状态：**v0.7 LOCAL VERIFIED / READY FOR LINUX FULL REISSUE / TEST LOCKED**。原冻结稿 v0.2 及后续修正案
> 保留为历史；本文件冻结 O1/B1/T1/R1/J1 判别实验与前置 D0/D-SYN
> 门禁的假设、数据合同、预算、判决规则与禁止事项。K1 为条件实验，母模型未过门禁前保持 HOLD。
> v0.1 → v0.2 修正案见 §5（首轮执行回传后生效）。
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
| 5 个 `OBSERVATION_ELEMENTS` | 减温器站汽温：一级减温器入/出口、二级减温器入/出口、末过出口（即低过出口、屏过入口、屏过出口、高过入口、末过出口） | degC | 观测 |

D0 必须产出**点位映射审计表**（DCS 点名 → 注册表通道 → 单位换算 → 可信度），未闭合的通道
映射直接判 MIXED 并阻断对应实验。RM3-R 审计遗留的 tag 证据缺口在此一次性收口。

**D0 执行结果（2026-08-18 回传，本地审计认可）**：14/14 通道 HIGH 置信闭合
（`results/final_wm/d0/mapping_audit.json`）；A/B 双侧各 ≈70.8 万行、82.6 天、10 s 采样，四项
质量门全过（`quality_gates.json`）。数据为**双侧结构**：单级边界 7 通道共享，阀位与汽温按侧
成对；交叉控制映射（用户 2026-08-09 确认 + RM3 先例）为 A 侧阀控 B 侧温、B 侧阀控 A 侧温。
因此矩阵按侧独立执行：桥接 `run_matrix.py --phase split-sides` 产出每侧注册表格式记录，
矩阵 phase 逐侧跑两次（`--side A/B`），判决按侧报告。冻结点位映射：
`configs/final_wm/channel_mapping.json`。

### 1.2 质量门（fail-closed）

- 采样周期 10 s，时间戳连续性缺口率 < 1%；
- 每个通道的饱和/卡死段（连续 > 30 min 零方差）占比 < 5%；
- 阀位非饱和段（v1、v2 不同时长时间贴 0/100%）覆盖 ≥ 60%，否则动作通道激励不足，T1/R1 不得启动；
- 可用连续时长 ≥ 30 天等效；不足则由 D0 如实回传并缩小矩阵，不补数据、不降门槛。

### 1.3 切分与 canonical records

- 按时间顺序切分，冻结比例 **train 75% / validation 15% / test 10%（保留锁定）**；本矩阵全部
  判决只用 validation；D0 执行侧原提案 80/20 由本地审计改冻结为 75/15/10；
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
- 训练预算（v0.2 修正）：T1 全部臂统一 epochs ≤ 60 / patience 10；ledger 逐 run 记录
  `stop_reason` / `converged` / `val_tail` 收敛诊断，凡 `stop_reason=cap` 且 `val_tail` 仍下降的
  臂，其负贡献判决降级为 provisional 并在审计中单独标注（首轮 latent4 seed2 教训）。
- 负对照：`closure(conservative)` 与 `closure(steam_only)` 差异不显著时优先 conservative
  （能量守恒先验），并在报告中声明该选择是先验驱动而非数据驱动。

### R1：closure 因果与量级门

- 假设 H_R1：action-blind closure 吸收的是真实未测扰动而非动作/阀门因果泄漏。
- 检查项：
  1. 运行时 action-blindness（同本地合同测试，真实权重下复跑）；
  2. 残差功率分位数报告（p50/p90/max，kW），相对典型段级热流（≈1e5 kW）占比；
  3. 阶跃响应方向一致性审计：closure 开启/关闭两种配置下，v2 阶跃 +0.05 的长期终端温度
     响应必须为负；任一配置翻号 → closure REJECTED（E4 防线）；
  4. 负对照（操作化）：生产 closure 合同上不可读动作，负对照以独立残差泄漏探针实现
     （`src/final_wm/diagnostics.py:leakage_probe`）——两个独立探针分别用 [状态+白名单边界] 与
     [状态+边界+动作] 拟合冻结物理骨架的一步前观测残差；aware 探针验证 MSE 相对 blind 改善
     > 5% 判 closure 存在动作泄漏，整臂 REJECTED，回报残差结构供本地归因。探针为诊断件，
     不进入生产装配。
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

## 5. 修正案记录

### v0.2（2026-08-19，首轮执行回传后生效）

触发：Linux 首轮执行报告（`results/final_wm/execution_report.md`）。

1. **T1 训练预算统一上调**：默认 30 epochs / patience 5 对含潜变量臂偏小（latent4 seed2 撞顶
   仍在降，seed0/1 停在浅谷高原），T1 嵌套比较的负贡献判决存在预算混淆。修正为 T1 全部臂
   统一 60 / 10——均匀上调保持比较对称，早停仍约束已收敛臂的成本；O1/B1/J1 臂不变。
2. **收敛诊断入 ledger**：所有单元逐 run 记录 `stop_reason`（patience|cap）、`converged`、
   `val_tail`（末 5 轮 val NLL）；审计阶段按 §T1 所述规则降级未收敛臂的判决。
3. **runner 断点续跑与增量落盘**：run 级产物（checkpoint + metrics + spec 指纹）匹配时跳过
   重训；spec 变更（如本修正案）自动触发对应臂重训，其余臂复用；matrix_summary 每单元增量
   落盘。判决与训练由此解耦，`--units` 子集重跑安全。
4. **执行侧修复复核**：cc81cb3（GPU 设备搬运）与 f8ec07f（R1 import）经 Supervisor 复核接受，
   均补了回归测试；R1 路径另发现两个潜伏缺陷（`layout.state_dim`→`.dim`、`latent_raw`→
   `latent_step`）已修并纳入端到端 smoke。

### v0.3（2026-08-20，侧 A v0.2 判决审计闭合后生效）

触发：证据链对齐审计 §4/§5.2/§5.3 与用户裁定（修复批立项、侧 B 延至 AE 阶段）。

1. **R1 判决规则修订（标定错误修正）**：原"100% 窗口负响应"严于真实对象可观测行为——
   本地协议化事件研究（双阀 ±60 步污染排除，`src/final_wm/analysis.py`）测得 v2 60 步正确
   方向占比 0.68（up，n=22）/0.75（down，n=48）。修订为：**均值终端 ΔT < 0 且 day-block
   bootstrap CI 整体低于 0，且正确方向占比 ≥ 0.60**；H18/H60 两档同报。运行时盲检与泄漏
   探针规则不变。该修正不为现行模型开门（现行 R1 口径 0.19–0.34 仍失败）。
2. **物理修复批立项（顺序冻结 ①→⑤，逐批进矩阵重跑）**：
   - **① 全五点初态锚定 + 压力分段反演学习化**（靶：sh1_in H1 箱均值 5.3–13.3 °C，
     38× persistence 放弃签名；观测方程问题，顺带消除 O1 学习后验退化通道）；
   - **② 喷水→混合链路传输/蒸发时滞**（一阶滞后状态接入混合路径；靶：真实对象 H1≈0
     vs 模型瞬时过冲、60 步符号反转的动作保真根因之一）；
   - **③ 再湿项符号-量级硬契约**（q_w 上界 ≤ 喷水直接冷却项或质量平衡闭锁；靶：aW=0
     消融 0.27→1.00 的正反馈根因）；
   - **④ 喷水灵敏度先验锚定数据回归值**（dW/dv1=27.8、dW/dv2=70.0 t/h/满开度，
     auditpack 口径；消除 ~5× 先验失配，与②联合验证）；
   - **⑤ 参数 MLP 主线**：v0.2 后单调负荷偏差证据作废（网格物性复算 final 通道两 horizon
     平坦，between_ratio 0.069/0.024），**降级观察**，无新分箱证据不立项。
   每批为独立修正案子项，进矩阵重跑 T1+R1（①另重跑 O1）；预算、seeds、判决纪律不变。
3. **架构路线登记（gray-box split by time scale）**：物理管快动态与不变量；学习仅限
   action-blind **慢扰动状态**估计与误差校准（closure_cons 正式化为扰动估计器，SUPPORTED
   已实证）；边界**外生化**（oracle/计划值/场景库——B1 −2.0 CRPS 已否决学习式预测）；
   观测端稳态锚全锚定 + 慢偏置，不学习快状态后验（O1 −30%/−31% 已否决）。steam/latent4
   容量扩张路径封存（双口径 REJECTED）。
4. **侧 B 延期**：侧 A v0.2 判决已审计闭合（11/11）；侧 B 矩阵延至 AE/投稿补充阶段一并
   执行（`canonical_sideB.npz` 已在仓，届时按届时版本矩阵含修复批重测）。论文若先行，
   必须显式声明 single-side 范围并在 Limitations 落地。
5. **评测协议增强**：skill score vs persistence（10s 增量 MAE 基线 0.088/0.192/0.246/
   0.275/0.451，auditpack 口径）列入标准报告量；事件研究脚本已转正为 R1 参考带生成器
   （入仓、带测试，`--phase auditpack`）。

### v0.7（2026-08-28，可信度审计后的可执行合同修订）

触发：`docs/FINAL_WM_CREDIBILITY_AUDIT_2026-08-27.md` C1。v0.2 runner 只执行了各单元
部分指标，却仍写出方向性 verdict。v0.7 不改模型结构，先把文档中的必需证据变成机器可读合同：

| 单元 | 正式 verdict 的必需证据 |
|---|---|
| O1 | H6/H18 NLL、相邻窗口 state continuity、v0.7 paired-NLL 统计 |
| T1 | H1/H6/H18 NLL、60 步定常 drift/settle、v0.7 paired-NLL 统计 |
| B1 | 7 通道 H6/H18/H36、forecast-vs-oracle 下游 H18 NLL 退化 |
| J1 | H1/H6/H18 全指标、H18 NLL、H36 定常条件稳定性、v0.7 paired-NLL 统计 |
| R1 | runtime blindness、残差功率、双阀 H18/H60 方向+day-block CI、修正后的 leakage 与支持域证据 |

执行纪律：

1. 任一必需键缺失时只写 `INCOMPLETE`；quick 只写 `SMOKE`；partial seeds 或 `--arm-filter`
   只写 `INCOMPLETE`，不得暴露 `SUPPORTED/MIXED/REJECTED`；
2. summary 必须携带 `matrix_version` 与完整 `required_evidence`；只允许合并同版本、同 tier、同侧的
   增量 summary，禁止把 v0.2–v0.6 单元块带入 v0.7；
3. O1 continuity 使用相隔 18 步的相邻历史窗，候选臂 day-block CI 上界不高于 steady 点估计；
4. T1 定常稳定性沿用本地合同：全有限、终端最大漂移 ≤60 °C、末 6 步 settle ≤5 °C；
5. B1 H6/H18/H36 共用相同验证抽样；下游退化用同模型、同窗口的
   `NLL_forecast - NLL_oracle` day-block CI，仅报告不设门；
6. J1 H36 在相同冻结边界/动作、条件锚定初态下同报 joint/staged，全有限且满足 drift/settle 合同；“不劣”操作化为 joint 的
   terminal drift p95 不高于 staged；
7. R1 对 valve1/valve2 分别报告 H18/H60 的均值、95% day-block CI 与正确方向占比，判据沿用
   v0.3（均值<0、CI 上界<0、占比≥0.60）。
8. O1/T1/J1 的正式 NLL 门使用相同验证窗口上的 `ΔNLL = arm - baseline`，先按 UTC 日聚合再
   bootstrap；单 seed 仅当 95% CI 上界 `< 0` 才计通过，完整判决仍要求至少 2/3 seeds。
   NLL 不再使用百分比阈值；CRPS/MAE 的相对改善只作为实用效应量报告。
9. 2026-09-01 全量重发继承 v0.6-B 的显式 `epochs=120 / patience=20`，取代 v0.2 的历史
   60/10 预算；正式记录为 canonical v2.2 的 7 通道 base view，正式 R1 栈固定为
   `closure_cons_norew`。A5/LPV/zcond/JEPA-B 不并入本矩阵。

阶段护栏：Task 1–2 已闭合 C1 与 paired-NLL。`leakage_v07`、`support_domain_v07` 在对应修复
完成前保持缺失，因此 R1 仍为 `INCOMPLETE`；当前不授权 Linux 或正式重跑。

## 6. 明确禁止事项

- 不访问 test；不生成 B2 式重试批次；不复活旧 runner；
- 不把 Direct-WM/物理分支历史数值当本矩阵先验之外的证据；
- 不在支持域外报告反事实结论；不允许以 observational fit 冒充 do-calculus 干预；
- 不修改 `src/final_wm/` 接口与合同以迎合数据；数据问题回 D0 如实判决；
- K1、MS4、论文写作继续锁定。
