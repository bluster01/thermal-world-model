# Phase 3.5-MS2-D1 一次性 Synthetic Test 设计

## Material Passport

- Material Type: confirmatory synthetic test design
- Evidence Scope: `synthetic_delay_pressure_test_not_field_causality`
- Upstream: D1 validation screening at execution commit `95d1dbe`
- Status: FROZEN FOR IMPLEMENTATION
- Test Boundary: 不重训、不改阈值、不加 seed、不读取 A/B 数据

## 1. 为什么不能直接进入 D2

D1 validation 的预注册点估计门通过：learned-delay 相对 no-delay 的 clean NMAE 改善为 20.25%–23.11%。但 validation 同时承担 checkpoint 选择，且 seed 1 只高出 20% 门槛 0.25 个百分点。本地仅作精度诊断的 paired-episode bootstrap 显示三 seed 的 95% CI 下界为 17.26%–19.58%，均未达到 20%。该结果不推翻冻结的 validation screening PASS，却说明 D1 尚不能记为 confirmatory closed。

可选路径有三种：只测主模型/消融/正控、测试全部 18 个冻结 checkpoint、或跳过 test 直接进入 D2。采用第二种：一次性评估全部 18 个 checkpoint。这样不重新选模，能完整检查 secondary routes、结构合同和归档，也与 MS2-V/C/J 的一次访问协议一致。跳过 test 会把 validation 选择乐观带入 D2；只测三类虽然省计算，但会留下归档中其余 checkpoint 未经独立 split 验证的缺口。

## 2. 内容寻址与一次访问

授权文件同时 pin：训练矩阵、validation summary、18-checkpoint tar 及其 SHA-256。runner 在任何 test 输出产生前核对 18 个 manifest、archive member 名称/哈希、checkpoint 内部配置、单一 execution SHA、冻结代码等价性和 `test_accessed=false`。根 ledger 与每 run ledger 先写 `started`，完成后改为 `completed`；任何既有 root/run test artifact 都使再次访问失败。

test 只用 checkpoint 内的 synthetic spec 生成独立 `test` split（256 episodes/seed），不调用 optimizer。每 run 生成 aggregate metrics、paired episode metrics 和 access ledger，并只把 manifest 的 test 状态从 false 改为 true。

## 3. 冻结判决

确认门包括：

1. 18/18 artifact 与结构门通过；
2. fixed-delay oracle 每 seed test clean NMAE `<0.05`；
3. learned-delay 相对 no-delay 的 paired-episode、按 action profile 分层 10,000 次 bootstrap，三个 seed 的 95% CI 下界均 `≥0.20`。

迟延期望误差和真值 ±1 step 质量继续单列参数诊断，不进入响应确认门。若响应确认门失败，D1 记为“validation screening 阳性、independent test 未确认”，不重试、不调阈值；D2 是否仍运行必须以压力诊断而非正结论传播来重新设计。无论结果如何，都不能外推现场因果、真实 20 s 迟延或路线冠军。

## 4. 产物与下一状态

正式输出位于现有 `results/phase3_5/ms2d_delay/`：18 组 `metrics_test.json`、`episode_metrics_test.json`、`synthetic_test_access_ledger.json`，根 `synthetic_test_matrix_access_ledger.json` 和 `summary_test.json`。Linux 只执行冻结命令并回传；本地复算后才能把 D1 从 `test_completed` 改为 `closed`，再决定 D2。
