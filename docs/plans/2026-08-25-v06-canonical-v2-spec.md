# v0.6 canonical v2 数据规格（SPEC，待冻结）

> 依据：`docs/plans/2026-08-25-v06-all-merged-plan.md` Phase 1（用户已批准先做 Phase 1，
> 执行构建与重训后续再做）。本文是 canonical v2 的冻结规格：通道表、派生规则、
> 质量门、对齐门、指纹与共存纪律。实现：`src/final_wm/data_v2.py` +
> `configs/final_wm/channel_mapping_v2.json`。

## 1. 设计原则

1. **v1 证据链零扰动**：contracts.py 的注册表（boundary 7 / action 2 / obs 5）不增不改；
   v2 记录中 `boundary/obs/valid/timestamps/split` 五个键**逐位搬运**自 v1 npz
   （不重采样、不重切分）。`actions` 自 v2.1 起例外——按修正接线从 all_merged
   重建（known-defect 修复，见 §9）。
2. **扩展即附加**：新信号存 `boundary_ext`（v2.2 为 9 列）、`aux`（15 列）、`mill_on`（8 列
   二值）三个新键；v2 记录仍可被 v1 加载器原样打开（宽度契约不破）。
3. **fail-closed**：对齐门不过 → 构建中止；新通道质量门不过 → 构建中止。
4. **口径分列**：v2 下一切数字与 v1 冻结判决分列报告，禁止聚合。

## 2. boundary_ext（v2.2：9 列，Phase 2 机制臂的候选输入）

| 名称 | 源列（all_merged_10s.csv） | 单位/变换 | 物理角色 |
|---|---|---|---|
| `fuel_corrected` | 校正后总燃料量 | t/h | 实际热输入（治 coal_command=指令之疾） |
| `mill_count_on` | 派生：给煤机瞬时煤量×8 > 2.0 t/h 计数 | 台 | 磨组合规模 |
| `mill_gas_temp_wavg` | 派生：8×磨高温炉烟温度，流量加权（流量 clip≥0） | degC | 炉膛热分布/磨组合重心 |
| `flue_o2` | 烟气含氧量 | % | 过量空气/燃烧工况 |
| `secondary_air_total` | 总二次风量 | t/h | 配风强度 |
| `rh_gas_in_temp_a` | 立式低温再热器入口烟气温度(A) | degC | 炉膛出口烟温场（A 侧） |
| `rh_gas_in_temp_b` | 立式低温再热器入口烟气温度(B) | degC | 炉膛出口烟温场（B 侧） |
| `water_coal_ratio` | 水煤比 | 无量纲 | A5 真实 DCS 水煤比修正量 |
| `unit_load` | 机组负荷_GENERATOR_POWER | MW | A5 运行工况门与训练集内 `wc_ref(L)` |

## 3. aux（15 列，监督/诊断，Phase 1 不进模型输入）

| 名称 | 源列 | 用途 |
|---|---|---|
| `att1_in_temp_l/r` | 选择后左/右侧一过喷水减温器入口 | A2 闭包一级混合监督 |
| `att1_out_temp_l/r` | 选择后左/右侧一过喷水减温器出口 | A2 同上（级后） |
| `att2_in_temp_l/r` | 选择后二级减温器左/右侧入口蒸汽 | 二级级前参考 |
| `att2_out_temp_l/r` | 选择后左/右侧二过喷水减温器出口 | A2 二级混合监督 |
| `spray_flow_sh_total` | 过热器减温水总流量 | 校验既有 spray_flow_total |
| `spray_flow_rh_total` | 再热器减温水总流量 | 再热喷水（obs 外耦合） |
| `superheat_sep` | 汽水分离器出口过热度 | 干/湿态判别（A3 润湿门控特征） |
| `rh_steam_in_temp_l/r` | 选择后左/右侧再热器入口蒸汽温度 | 再热侧（A5 前提） |
| `rh_steam_out_temp_l/r` | 选择后左/右侧再热出口汽温 | 同上 |

`mill_on`（8 列 uint8）：给煤机瞬时煤量 k=1..8 > 2.0 t/h → 0/1。服务 Phase 0
磨切换事件研究与 A4 炉膛热潜态；**给煤量 0=停磨是物理真值，显式编码**。

## 4. 对齐门（alignment gate，fail-closed）

目的：证明 all_merged 与 v1 canonical 是同一时间网格上的同一物理过程。
步骤：
1. **网格包含**：v1 timestamps 必须在 all_merged 的 10s 网格上有精确 epoch
   匹配（插值上限 30s 与 v1 一致）。**内部行对不上 → 中止（fail-closed）**；
   首/尾连续越界段（v1 记录边缘超出源文件覆盖）允许裁边，每侧 ≤60 行
   （10 分钟），裁边量入 meta.edge_trim，保留行逐位搬运。实测依据：侧A v1
   首 12 行（2025-12-23 08:25-08:27）早于 all_merged 起点。
2. **数值交叉核验**（v1 通道 ← all_merged 候选列，阈值入 mapping）：

   | v1 通道 | 候选列 | min_corr | max_mae |
   |---|---|---|---|
   | steam_flow | 主蒸汽流量_60BKAO0312（×0.27778） | 0.999 | 2.0 kg/s |
   | separator_pressure | 选择后分离器最终出口压力 | 0.999 | 0.10 MPa |
   | separator_temperature | 分离器最终出口温度 | 0.990 | 3.0 °C |
   | feedwater_temperature | 选择后省煤器出口给水温度 | 0.990 | 3.0 °C |
   | outlet_pressure | 选择后左/右侧末级过热器出口压力（按 corr 自动选侧） | 0.999 | 0.10 MPa |
   | final_outlet_temp | 选择后左/右侧末级过热器出口汽温（按 corr 自动选侧） | 0.995 | 2.0 °C |
   | spray_flow_total | 过热器减温水总流量 | 0.980 | 3.0 t/h |

   自动选侧结果写入 meta.provenance（双侧 v2 构建应各自收敛到正确侧——
   侧选择本身即交叉验证）。**actions 不做数值交叉**（v1 阀位来自镜像双记录
   交叉映射，与 all_merged 阀位反馈口径不同源），仅靠网格包含门。
3. 交叉核验用**原始值**（清洗 clip 之前）计算。

## 5. 新通道质量门（fail-closed）

- 覆盖率 ≥ 99%（30s 插值上限后）；`mill_count_on`/`mill_on` 豁免 stuck 门
  （阶梯语义），其余新通道 stuck_ratio ≤ 0.05（30 分钟零方差段）。
- 物理量程（越限比例 > 0.1% → 中止；清洗 clip 规则入 mapping，gate 在 clip 后）：
  fuel_corrected [0,600] t/h；flue_o2 [0,15] %；mill_gas_temp_wavg [0,900] °C；
  rh_gas_in_temp_* [0,900] °C；secondary_air_total [0,2000] t/h；aux 温度
  [0,800] °C；spray 流量 [0,400] t/h；superheat_sep [-60,200] °C。
- 已知野值处置（侦察实测）：磨炉烟流量负值 → clip≥0 后参与加权；7 号磨停机
  炉烟温度 ~10°C 环境温度 → 权重=0 时不入加权，自然豁免。
- 原始质量门先于 clip：`spray_flow_sh_total` 已审计负零漂最低 −9.05 t/h，冻结
  `raw_range=[-10,400]`；`spray_flow_rh_total` 最低 −1.03 t/h，冻结
  `raw_range=[-2,400]`。通过 raw 门后才 clip 到模型范围 `[0,400]`；其他通道未声明
  `raw_range` 时直接以 `range` 作原始门，禁止用 clip 隐藏越界。
- v2.2 A5 两列的存储量程为：`water_coal_ratio` [0,10]、`unit_load` [0,800] MW；
  A5 采窗另按预注册运行工况门 `unit_load>160, 1<ratio<8, fuel>50` 排除停机与异常点。

## 6. 产物与指纹

- 输出：`artifacts/final_wm/canonical_side{A,B}_v2.npz` + `_v2_meta.json`
  （version=2；provenance 含 all_merged sha256、v1 npz sha256、mapping_v2
  sha256、对齐门逐通道报告、质量门报告、自动选侧结果）。
- v1 产物禁覆写；v2 与 v1 并存，指纹隔离（训练 spec 指纹在 Phase 2 纳入
  record 路径哈希，沿用现有 config_fingerprint 纪律）。

## 7. 执行与验收（后续发单）

- 执行侧：`python experiments/final_wm/build_canonical_v2.py --side A|B`
  （包装脚本已交付），双侧各 ~4 min（单遍 3.8GB usecols 读 + sha256 全文件哈希）。
- 验收：双侧对齐门全过、final_outlet_temp 自动选侧互为镜像、meta 完整、
  契约测试 100%。

## 8. 本地构建实绩（2026-08-25，设计侧首建）

- **双侧构建成功**：各 n=707,709（裁边 leading=12，trailing=0，v1 保留行
  逐位一致已断言）。
- **对齐门 7/7 全过且 corr=1.0000**（含单位换算 steam_flow ×1/3.6）。
- **自动选侧交叉验证成立**：final_outlet_temp 侧A→左侧列、侧B→右侧列
  （镜像正确）；outlet_pressure 双侧均选左侧列——末过出口左右压力近乎同一
  机组级信号，同侧选择符合预期，留痕不视为异常。
- **质量门**：新通道 coverage ≥0.99999、range_violation=0、stuck≈0（最坏
  spray_flow_sh_total 0.0003）。实测修正两处：spray 两 aux 通道加 clip≥0
  （5.2%/0.28% 微负传感器零漂）；outlet_pressure 候选列名词序订正。
- **指纹**：meta 含 v1 npz / all_merged / mapping 三 sha256，执行侧重建可
  逐字节核验；npz 不入 git（执行侧确定性重建），meta json 入库。
- 测试：`tests/final_wm/test_data_v2.py` 9 项（含裁边/越界/失配 fail-closed
  与逐位一致性），全套 141/141。
- **SUPERSEDED（2026-08-26）**：本次 v2.0 首建继承 v1 错侧 valve1，制品标记
  SUPERSEDED 不删除；有效制品以 v2.1 重建为准。

## 9. v2.1 修订：actions 按修正接线重建（2026-08-26 批准）

依据：`results/final_wm/known_defect_v1_valve1_20260826.md`。物理接线
（用户 2026-08-25 现场确认一级同侧 + 滞后差分复核）：

| 记录侧 | obs 温度侧 | actions[0]=valve1 | actions[1]=valve2 |
|---|---|---|---|
| A | 左 | 过热器一级减温器A侧喷水调节门阀位反馈 ÷100 | 过热器二级减温器B侧喷水调节门阀位反馈 ÷100 |
| B | 右 | 过热器一级减温器B侧喷水调节门阀位反馈 ÷100 | 过热器二级减温器A侧喷水调节门阀位反馈 ÷100 |

- 映射新增 `actions` 段（按侧给出两列 + 连续性门）；`build_canonical_v2` 增
  `--side` 参数（A/B）。量程容忍带 [-0.02, 1.0]：阀位反馈存在传感器负零漂
  （一级B 6.2%、二级A 17.9%、二级B 5.0%，界内 ≥-1%，DCS 画面实测可见
  "一减B位置 -0.9%"），终端 clip≥0，与 aux spray 通道同清洗规则。
- **连续性门（fail-closed）**：新 valve2 与 v1 旧 valve2 corr≥0.999 且
  mae≤0.02（v1 二级本就配得对，必须逐位等价）；新 valve1 与 v1 旧 valve1 的
  corr 记录入 meta.provenance（不设门，预计 ≈0.8）。
- 元数据：version=2.1；provenance 增 known_defect 引用与旧 actions 指纹。
- 加载器不变（键集合不变）；`CanonicalV2Record` 行为不变。

## 10. v2.2 修订：A5 真实水煤比上下文（2026-08-28）

- `boundary_ext` 末尾追加 `water_coal_ratio` 与 `unit_load`；v1 七通道契约及所有核心键不变。
- 两列只服务预注册 A5 独立机制臂：真实 ratio 进入 metal-power 修正，unit load 只用于
  运行工况门和在 train split 内拟合二次 `wc_ref(L)`，不得从 validation/test 拟合。
- A5 模型使用 7+2 的专用 oracle 视图；正式 v0.7 模型、历史 v2.1 结果和 paper verdict 不改。
