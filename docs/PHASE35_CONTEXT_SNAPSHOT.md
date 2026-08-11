# Phase 3.5-MS 上下文恢复快照

> 更新：2026-08-11。新会话先读本文，再运行状态检查器。机器状态以 `configs/phase3_5/experiment_registry.json` 为准；本文解释为什么。

## 1. 恢复命令

```bash
git pull --ff-only origin main
python experiments/phase3_5/experiment_status.py --check --json
```

若命令中的 `active_gate`、`linux_authorized_gate` 或 deprecated track 与本文不同，先审计最新 commit，不凭聊天记忆继续实验。

## 2. 不可丢失的项目决定

1. Phase 4 暂停；先完成 Phase 3.5-MS 系列，不提前进入论文收口。
2. 旧 E1–E5 已废弃，只保留为历史失败与数据识别边界，不再作为当前 Gate、候选选择或主实验目录。
3. 现场 A/B 交叉链按现场事实冻结，不再验证；现场为串级 PID。
4. 喷水流量传感器不可靠；实际阀位是有效喷水作用代理，不是 kg/s。
5. 本地负责设计、代码、测试、冻结矩阵和审计；Linux 只在指定 commit 执行冻结命令并原样回传。
6. 完整顺序为 `MS0 → MS1 → MS2-V/C/J → MS2-D1/D2/D3 → MS5 → MS3 → MS4 → 模型选择/论文`。
7. MS2-J 的 response-internal staged 失败不能外推到 MS5；MS5 必须单独检验 `free+response` 耦合与动作吸收。

## 3. 当前可信实验状态

| Gate | 状态 | 可守结论 |
|---|---|---|
| MS0 | CLOSED | reference identity、prefix causality、state continuation 合同可测试 |
| MS1 | CLOSED | 同型二阶 known-truth 可解；存在 inverse crime，不设冠军 |
| MS2-V | CLOSED | learned monotone 在 R50 truth 下相对 identity 双层 PASS；`K/phi` 不可拆分 |
| MS2-C | CLOSED | scheduled K/τ 相对 global 双层 PASS |
| MS2-J | CLOSED | joint 相对两个单模块双层 PASS；staged 10% 非劣双层 FAIL；采用 joint |
| MS2-D1 | CLOSED | test 改善方向稳定，但逐 seed CI 下界 17.2–18.8% 未达预注册 20%；参数诊断继续负，不重试 |
| MS2-D2 | CLOSED | test 三个主门逐 seed通过；确认 frozen known-truth 三阶响应优势，不确认现场阶次唯一性 |
| MS2-D3 | CLOSED | 21-run validation 主门通过；按预算不追加 test，不称 confirmatory |
| MS5 | CLOSED | joint component recovery validation 通过；冻结 staged 协议拒绝 |
| MS3 | AUDITED FAIL | B 3/3 PASS；A 0/3 response non-collapse FAIL；不重跑、不访问 test |
| MS3-D | LOCAL ONLY | 稳态 A/B `SP→阀位→温度` 经验响应与 checkpoint IRF 对齐，不训练 |
| MS4 | HOLD | MS3-D 前不启动；不恢复旧 E 匹配 |

## 4. Linux 最新同步与本地审计

Linux 在 `597180f` 回传 MS3 v1.1 validation，实际训练 commit 为 `798fcde`。12/12 runs、12-file checkpoint archive、manifest/cache timeline、结构门和 no-test 边界闭合；本地重建全部 validation anchors，逐 checkpoint 在 x86 CPU 重放，aggregate metric 最大差 `1.1623e-5`、单窗口最大差 `8.4734e-4°C`，冻结 CI 下界漂移 `<9.5e-8°C`，所有判决一致。B 回路动态效应 `0.04289–0.04851°C`、3/3 seeds 通过；A 仅 `0.00663–0.00854°C`、0/3 通过。B/A 效应比 `5.03–7.32`，动作剂量中位数比仅 `1.052–1.059`；标准化 +5% H60 checkpoint 响应也显示 B 约为 A 的 4–5 倍。最终标签 `AUDITED / OBSERVATIONAL_VALIDATION_FAIL_ASYMMETRIC / NO_RETRY / MS4_HOLD`。权威判决见 [`PHASE35_MS3_SUPERVISOR_AUDIT_2026-08-11.md`](PHASE35_MS3_SUPERVISOR_AUDIT_2026-08-11.md)。

Linux 在 `1fb6a23` 回传 MS5 validation，实际执行 commit 为 `af31495`。12/12 runs、21-file checkpoint archive 和 test-access 门闭合；本地从 12 个最佳权重重新生成 validation 后最大指标差 `2.39e-7`，archive 可确定性重建为同一 SHA。component oracle 与 joint 逐 seed全过，joint response NMAE `0.047–0.050`；staged/joint total-error ratio `11.14–14.11`，当前 staged 协议拒绝；free-only 的 response NMAE=`1`、amplitude=`0`，确认总预测误差不能替代组件审计。最终标签 `CLOSED / VALIDATION_ONLY_COMPONENT_RECOVERY_PASS / JOINT_SELECTED / STAGED_PROTOCOL_REJECTED`。Linux 曾在 aarch64 legacy raw-float SHA 预检红项后继续，属流程偏差；本地复核确认非科学回归，测试已换成 1e-6 量化哈希。权威结论见 [`PHASE35_MS5_SUPERVISOR_AUDIT_2026-08-11.md`](PHASE35_MS5_SUPERVISOR_AUDIT_2026-08-11.md)。

Linux 在 `f8a48ec` 回传 MS2-D3 validation，实际执行 commit 为 `040cb27`。21/21 runs、manifest/history/episode/checkpoint archive 和结构门闭合；本地 episode 重算最大差约 `3.35e-8`，独立 50,000 次 profile-stratified paired bootstrap 与冻结 10,000 次判决一致。oracle clean NMAE 为 0.0357–0.0446，三阶为 0.0558–0.0633，三阶相对二阶的冻结 CI 下界为 10.8%–14.3%。按负责人预算决定以 `CLOSED / VALIDATION_STRESS_PASS / NO_TEST_BY_BUDGET_DECISION` 关闭；validation 参与 checkpoint 选择，因此不是独立 test。归档 tar 的容器 path/hash 曾在远端 summary 后处理，但 21 个成员权重逐字节一致，记 provenance advisory 而不重跑。权威判决见 [`PHASE35_MS2D3_SUPERVISOR_AUDIT_2026-08-11.md`](PHASE35_MS2D3_SUPERVISOR_AUDIT_2026-08-11.md)。

MS5 的 4 modes×3 seeds 已关闭归档；它只回答 frozen truth 中 total-only 训练时 response 是否被 free head 吸收，不能覆盖 MS3 的真实 A/B 不对称失败。

Linux 在 `d97538f` 回传 MS2-D2 one-shot test，实际执行 commit 为 `c221403`。远端提交只修改 `results/phase3_5/ms2d_order/**`，21/21 root/run ledger 均为 completed，日志、环境、manifest、checkpoint pin 与冻结 authorization 全部闭合。本地 canonical 重建与 `summary_test.json` 完全一致，episode aggregate 最大差 `2.461e-08`；独立 NumPy PCG64 50,000 次 profile-stratified paired bootstrap 也保持同一判决。oracle clean NMAE 为 0.0211–0.0255，三阶为 0.0444–0.0465，三阶相对二阶点改善 23.74%–25.36%，冻结 95% CI 下界为 19.90%–21.22%，逐 seed高于 10%。因此 D2 以 `CLOSED / CONFIRMED_SYNTHETIC_ORDER_RESPONSE` 关闭。二极点+learned-delay 与 DeepONet 在有限 horizon 仍接近三阶，故不能升级为现场唯一阶次或迟延机制。权威判决见 [`PHASE35_MS2D2_TEST_SUPERVISOR_AUDIT_2026-08-11.md`](PHASE35_MS2D2_TEST_SUPERVISOR_AUDIT_2026-08-11.md)。

D3 的冻结设计与边界仍见 [`plans/2026-08-11-phase35-ms2d3-disturbance-design.md`](plans/2026-08-11-phase35-ms2d3-disturbance-design.md)，但该 Gate 已关闭，不再授权重复运行或 test。

Linux 在 `aedf1be` 回传 MS2-D2 validation，实际训练 commit 为 `fa6933c`。21/21 artifacts、manifest、history、checkpoint archive 与结构门禁已由本地逐项复核：oracle clean NMAE 0.0214–0.0226；三极点主模型 0.0403–0.0520；相对二极点点改善 18.56%–28.10%。独立重建 validation episodes 后，配对/profile 分层 10,000 次 bootstrap 的 95% CI 下界为 15.08%–22.28%，逐 seed 高于预注册 10%。因此只判 `AUDITED_SCREENING_PASS / TEST_AUTHORIZED`；validation 参与 checkpoint 选择，不能代替独立 test。权威判决见 [`PHASE35_MS2D2_SUPERVISOR_AUDIT_2026-08-10.md`](PHASE35_MS2D2_SUPERVISOR_AUDIT_2026-08-10.md)。

无 pure-delay truth 下，二极点+learned-delay 仍产生 2.16–2.40 steps 的期望迟延且零步质量仅 0.241–0.297。这说明遗漏阶次可被迟延容量补偿，不是现场迟延阳性。Linux 另写入了一个 validation review，违反远端只写结果/可选 `UNVERIFIED_REMOTE_REPORT` 的边界；该文件已降级，不能覆盖 Supervisor audit。

Linux 在 `dc2939c` 回传 MS2-D1 one-shot test。远端报告的数字经本地重新聚合、独立 50,000 次 NumPy bootstrap、manifest/episode/ledger/hash 审计后成立：oracle 为 0.0206–0.0223；learned-delay 点改善 20.4–22.5%，但冻结 95% CI 下界为 17.2–18.8%，未达 20%。因此 D1 以 `TEST_NOT_CONFIRMED_AT_20PCT_MARGIN` 关闭；不重试、不调阈值，也不把 learned-delay 当现场已证实结构。权威判决见 [`PHASE35_MS2D1_TEST_SUPERVISOR_AUDIT_2026-08-10.md`](PHASE35_MS2D1_TEST_SUPERVISOR_AUDIT_2026-08-10.md)。

Linux 在 `7cf2b14` 回传 MS2-D1 validation：18/18 artifacts/结构门禁通过；oracle clean NMAE 0.0201–0.0211；learned-delay 相对 no-delay 的逐 seed 点改善 20.25%–23.11%；learned delay 期望为 2.03–2.20 steps，但真值 ±1 step 质量仅 0.538–0.579。

本地重新汇总全部 JSON、解包并加载 18 个 checkpoint、重放 best-epoch 选择和配置/hash 校验，结果一致。独立的 validation episode bootstrap 诊断显示 95% CI 下界约 17.3%–19.6%，所以判为 `AUDITED_SCREENING_PASS`，不是确认性关闭。当前 content-addressed authorization 只允许原 18 checkpoints 的一次性 synthetic test；D2 继续冻结。完整判决见 [`PHASE35_MS2D1_SUPERVISOR_AUDIT_2026-08-10.md`](PHASE35_MS2D1_SUPERVISOR_AUDIT_2026-08-10.md)。

此前 MS2-J 状态保持不变：

Linux 在 `5260d3f` 完成 MS2-J test，27/27 root/run ledger 为 completed。基于每 seed 256 episode、按 action profile 分层、10,000 次 bootstrap：

- joint vs 两个单模块的改善为 0.77–0.88，95% CI 下界 0.73–0.89，PASS；
- staged/joint ratio 为 1.14–1.20，95% CI 上界 1.21–1.32（seed 0 下界附近不影响上界判决），非劣 FAIL；
- staged 相对 Stage A 改善 0.73–0.74，CI 下界 0.68–0.77，PASS。

本地已从 episode JSON 重新运行同一 bootstrap，三组 Gate 与 `summary_test.json` 逐字段完全一致。

### Provenance advisories

1. ledger 的 `evaluation_git_sha=78904ba` 不在远端对象库；可达的 pin-fix commit 是 `b170689`。冻结代码/产物一致，但以后 ledger 只能记录可达 SHA 或显式记录 rewrite alias。
2. `summary_validation.json` 的 Windows working-tree SHA 为 `788b1bc2…`，Git blob/Linux LF SHA 为 `212e53cc…`。这是 CRLF/LF 差异，不是内容语义变化。以后 JSON pin 使用 LF 或 canonical JSON；`.gitattributes` 已冻结 LF。

## 5. 当前下一步

MS3 已审计为 A/B 不对称 FAIL，Linux 授权已清空。当前只做本地 MS3-D：从稳定负荷、稳定主汽压力、处理前温度稳定的 SP held-step 估计 A/B 经验闭环响应，并与现有 checkpoint `±5%` IRF 对齐。动态工况只作分层描述；不重训、不改门槛、不访问 test。经验 A 若同样弱，后续转向 side-specific scale；经验 A 若与 B 接近，另立 response-identification 新协议。在 MS3-D 审计前，正式 MS4、模型选择和论文均保持 HOLD。

## 6. 上下文读取优先级

1. `configs/phase3_5/experiment_registry.json`：机器状态与关键脚本；
2. 本文：项目决定、推理链与恢复命令；
3. `TODO.md`：当前人工任务队列；
4. 当前 Gate 的 design/implementation plan；
5. 各 Gate review：只提供结果，不自行改变后续授权。

历史 handoff、E 系列设计和论文草稿若与前五项冲突，按历史材料处理。
