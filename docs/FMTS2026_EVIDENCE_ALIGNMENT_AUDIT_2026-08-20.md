# FMTS 2026 证据链对齐审计（Supervisor，2026-08-20）

> 触发：Linux 推送 `b93421e`（证据链/综述/逻辑链/路线图 v1）与 `234deb8`（论文初稿）。
> 本审计按四态规程核对三份文档之间、以及它们与冻结矩阵/已回传产物的一致性；
> 结论分为 SUPPORTED / CONFLICT（须改）/ UNVERIFIABLE（须补产物）/ PROTOCOL（规程问题）。
> 用户裁定：论文等证据链齐全再补全 —— 论文初稿正式登记为 FROZEN-DRAFT，解冻条件见 §5。

## 1. 规程问题（PROTOCOL）

| # | 事项 | 状态 | 处置 |
|---|---|---|---|
| P1 | 矩阵 §6 冻结"论文写作"，Linux 提前产出完整初稿 | 违规但用户已追认方向 | 登记 FROZEN-DRAFT，冻结至 §5 条件满足 |
| P2 | 论文 §4 引用侧 A 全套判决（O1/T1/B1/J1/R1 具体数字），但 `matrix_summary_sideA.json`、ledger、checkpoints 均未回传入仓 | **UNVERIFIABLE** | Linux 回传 `artifacts/final_wm/` 前，论文任何判决数字不得视为已证 |
| P3 | 证据链 §3/§4 与路线图 §6 的数值出自 Linux 本地 `/tmp` 脚本（residual_binning/error_floor/事件研究/消融），未入仓 | ~~UNVERIFIABLE~~ **已闭合（本地方案）** | 已由 Supervisor 重写为入仓模块 `src/final_wm/analysis.py`（事件研究/分箱/误差地板/喷水灵敏度/再湿消融）+ runner `--phase auditpack`，全部数值改为本地可复算口径；Linux /tmp 脚本不再作为证据源 |
| P4 | 执行报告仍写"重跑中止、等 Codex 修复"，但路线图 §6 已有 v0.2 重测数值 | 状态自相矛盾 | Linux 更新 execution_report.md 为真实状态（v0.2 跑到哪步、哪些 run 是新预算） |

## 2. 数字口径冲突（CONFLICT，全部须改）

1. **25–450× 与 −0.005~−0.015 / −0.45~−0.87 已撤回**（evidence_chain §1/§2），但仍存在于：
   论文摘要 L32-33、§3.1 L127-129、图 1 caption L236；`logic_chain.md` L9-10、L32-33。
   撤回原因：物理参考口径冲突未决时已撤回；新参考为数据锚定的 dW/dv + 混合能量平衡
   （v1：0.04–0.16 °C / v2：0.59–1.65 °C 每 +2%，稳态口径，随负荷带变化）。
2. **路线图 §6 "30–90× 差距坐实"是第三种口径**（模型 H1 −0.45 vs Direct −0.005~−0.015），
   与撤回口径、新参考三者互不一致 —— 统一为：Direct WM 响应引用审计表原值
   （−0.002~−0.014 等，含方向正确 seeds 计数），比值只对**同 horizon、同阀、同负荷带**的
   参考陈述；禁止跨口径压缩成单一倍数。
3. **论文 L130 "Line 2 一步响应 follows the energy balance (−0.2~−0.45)" 与事件研究直接冲突**：
   真实对象 H1≈0（响应分钟级建立，evidence_chain §3），模型 H1 比真实同尺度响应**大 10–30×**。
   正确表述：物理线 H1 响应是输出方程的构造性瞬时项，相对真实对象**过冲一个量级**，
   且 60 步符号反转 —— 这强化而非削弱张力叙事（两线都不过动作保真，路径不同）。
4. 细节一致性：论文 §2 写 "600 MW class"，D0 合同为 **660 MW**；T1 段 "seed 3" 应为 seed 2
   （seeds=0/1/2）；R1 探针步幅论文写 +5%，须与 `step_response_direction` 实际配置一致标注。

## 3. 与证据链一致的结论（SUPPORTED，可留用）

- D-SYN 3/3 PASS（99.3/107.0/136.5 vs 阈值口径）——与执行报告一致；
- 再湿反馈根因实锤（aW=0 消融：正确方向占比 0.12→0.94）——机制叙事完整、closure 已排除；
- 真实对象 60 步正确方向占比 ~0.61–0.67（v2 事件 613/744 个，n 充足）；
- v0.2 重测后 final 通道分箱偏差消失（+2.1~+5.4 → ±0.2 °C）→"参数 MLP 主证据失效、
  优先级重排"的路线图 v1 结论与收敛诊断设计目的一致，**接受**；
- 文献综述 32 条目与论文定位（D3×D4、verifiability over accuracy）无内部冲突。

## 4. R1 判决规则标定（v0.3 提案，冻结待激活）

事件研究证明真实对象在闭环混杂下 60 步正确方向占比仅 ~2/3，**冻结的 100% 规则比真实
对象可观测行为更严**，属标定错误而非模型单独失败。提案（与物理修复批一同激活，不单独
为现行模型开门——现行模型 0.12–0.19 在新规则下同样失败）：

- 方向判据改为：**均值终端 ΔT < 0 且 bootstrap CI 整体低于 0，且正确方向占比 ≥ 0.60**
  （事件研究参考带下沿留裕量）；H18/H60 两档都报；
- 运行时盲检与泄漏探针规则不变；
- 该修正必须写成矩阵文档 §5 修正案 v0.3，激活时点 = 物理修复批（§5.2）进矩阵重跑时。

## 5. 改进方案（冻结的优先级序）

### 5.0 规程前置（立即，阻塞论文与判决引用）

- Linux 回传 `artifacts/final_wm/` 全目录 + 更新执行报告 + 转正 /tmp 分析脚本；
- 本地完成侧 A 判决审计后，授权侧 B 执行（命令不变）。

### 5.1 论文解冻条件（全部满足才可补全）

1. v0.2 双侧判决经本地审计（产物在仓、数字可追溯 ledger/summary）；
2. §2 全部撤回/冲突数字清理完毕；
3. H1"能量平衡"表述按事件研究重写（过冲 + 反转 + 分钟级建立）；
4. 侧 B 判决补齐，或论文显式声明 single-side 范围并在 Limitations 落地；
5. 每个数值标注产物溯源（run_id / ledger 行 / summary 键）。

### 5.2 物理修复批（矩阵 A/B 判决闭合后立项，顺序冻结）

| 序 | 项 | 靶向证据 | 备注 |
|---|---|---|---|
| ① | 全五点初态锚定 + 压力分段反演学习化 | sh1_in H1 9.4 °C（38× persistence）、σ=8.8 放弃签名 | 路线图为①，代价小 |
| ② | 喷水→混合链路传输/蒸发时滞（tau_evap 接入混合路径或一阶滞后状态） | 真实 H1≈0 vs 模型瞬时过冲；sh1_out 0.44 比率 | 动作通道与精度同一杠杆 |
| ③ | 再湿项符号-量级契约（q_w 上界 ≤ 喷水直接冷却项，或质量平衡闭锁） | aW=0 消融 0.12→0.94 | 60 步反转根因 |
| ④ | 喷水灵敏度先验收紧（v1 ≈ 8 t/h/满开度，数据斜率锚定） | 模型 0.217 vs 数据 0.043 kg/s/2%（~5×） | 与②联合验证 |
| ⑤ | 参数 MLP 主线 | **证据已作废**（v0.2 后单调偏差消失） | 需新分箱证据重新立项，降级观察 |

### 5.3 评测协议增强（随 v0.3）

- skill score vs persistence 作为标准报告量（AR 一致性口径：10s 增量 MAE 基线）；
- R1 方向判据按 §4 修正；事件研究脚本转正为 R1 参考带生成器（入仓、带测试）。

## 6. 对 Linux 的下一步授权（本审计闭合后执行）

**路线更新（用户 2026-08-20 裁定）**：大产物不回传；侧 A 矩阵改为**本地执行**（本机 RTX 4070
+ CUDA torch），canonical 记录经带外通道拷贝到本机；侧 B 暂缓。

1. ~~回传 artifacts~~ → 改为：拷贝 `canonical_sideA.npz`（+ IAPWS 物性 npz）到本机
   `artifacts/final_wm/`；
2. 更新 `results/final_wm/execution_report.md` 为真实状态；
3. ~~/tmp 脚本转正~~ → 已由 Supervisor 本地协议化（`src/final_wm/analysis.py`，
   `--phase auditpack`），Linux 侧无需再传；
4. 侧 B 暂缓，等本地侧 A 判决审计通过后单独授权；
5. 论文目录 `docs/fmts2026/paper/` 冻结，禁止继续编辑直至 §5.1 条件满足。

### 本地侧 A 执行序列（冻结命令，数据到位后按序执行）

```bash
python -m pytest tests/final_wm/ -q                     # 门禁：110 项全过
python experiments/final_wm/run_matrix.py --phase dsyn --out artifacts/final_wm
python experiments/final_wm/run_matrix.py --phase matrix \
  --record artifacts/final_wm/canonical_sideA.npz --side A [--properties-npz <IAPWS>] \
  --out artifacts/final_wm
# 判决审计通过后补证据包（用 T1 closure_cons seed0 权重）：
python experiments/final_wm/run_matrix.py --phase auditpack \
  --record artifacts/final_wm/canonical_sideA.npz --side A \
  --checkpoint artifacts/final_wm/checkpoints/t1_closure_cons_seed0.pt --arm closure_cons --seed 0
```
