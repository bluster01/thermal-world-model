# Phase 3.5-MS2-J 一次性合成测试设计

> 状态：authorized for local implementation，尚未允许 Linux 访问 test。证据范围固定为 `synthetic_joint_coupling_test_not_field_causality`。

## 1. Supervisor 判决

MS2-J validation 是混合结果，而不是整体 PASS：联合非线性开度与工况调度模型相对两个单模块消融均通过 20% 增量门禁；staged 相对 joint-from-scratch 的误差比为 1.154–1.226，三个 seed 均未满足 1.10 非劣界。checkpoint、manifest、归档和 validation 数值已经本地复核，因此允许一次独立 synthetic test，目的仅为确认这两个预注册结论是否跨 split 保持。test 不得反向修改 validation 判决，不授权重训、调参、补 seed、A/B 真实数据 test 或现场因果主张。

## 2. 一次性访问与数据流

独立授权文件固定训练矩阵、validation summary 和 checkpoint archive 的 SHA256。runner 在任何 test 生成前完成全矩阵 preflight：27 个 manifest、27 个 canonical checkpoint、3 个 Stage-A checkpoint、训练代码等价性及所有既有 test/ledger 文件均须通过。权重直接从 tar 读取，不在结果目录恢复可变副本。通过后先写根级 `synthetic_test_matrix_access_ledger.json: started`，再为 27 个 run 依次写 metrics、episode metrics 与 run ledger；staged run 在同一次访问中额外评估已冻结的 Stage-A checkpoint。中途失败保留 started/partial 痕迹，禁止删除后重试。全部完成后根 ledger 才改为 completed。

## 3. 统计门禁

统计单位是完整 60-step episode；同 seed、同轨迹上的候选和基线配对，并按五种 action profile 分层 bootstrap 10,000 次。三个 seed 分别判定，不把 seed 当 episode，也不把 60 个时间点当独立样本。

1. `joint` 对 `monotone_global` 与 `identity_scheduled`：相对 clean MAE 改善的双侧 95% percentile CI 下界均须不低于 0.20。两个对比构成 intersection-union 门禁，必须同时通过，不做事后选优。
2. `staged` 对 `joint`：配对 clean MAE 比值的双侧 95% CI 上界须不高于 1.10。validation 已失败；test 只确认或显示 split 不一致，不能把单个 split 改写成 staged 优越。
3. `staged` 对 Stage A：相对改善 CI 下界须不低于 0.20，用于区分“分阶段完全无效”和“稳定但不如 joint”。

所有 27 个 canonical 模型同时执行结构诊断；oracle、DeepONet、PI-ODE、Koopman 只作预注册对照。汇总器只要任一 artifact、结构门禁或上述确认门禁失败即退出 code 2；这表示科学结果，不表示训练故障。

## 4. 允许结论

若 test 复现 validation，论文可写：在该联合 synthetic known-truth regime 中，单调有效开度与 context 调度同时存在时，联合灰箱响应模型能稳定识别响应；当前 staged schedule 不能在 10% 容忍度内匹配 joint-from-scratch，故主训练方案采用 joint。仍不得写成真实阀门曲线、真实质量流量、现场 `do(valve)`、完整 free+response 世界模型或“完全物理响应”。

## 5. 统计与推断反例审计

| 风险 | 本协议处理 |
|---|---|
| HARKing | test 前以内容寻址 authorization 冻结三个问题、阈值、seed 和 bootstrap；validation 的 staged FAIL 也写入授权，不能隐藏 |
| p-hacking / 重复试验 | test 只允许整矩阵一次访问；started/partial ledger 不得删除重试；禁止重训、改参、补 seed |
| 样本量不足 | 每 seed 256 条独立合成 episode；只作该生成器内确认，不以 3 个优化 seed 代替现场统计功效 |
| 多重比较 | joint 对两个预声明消融采用 intersection-union：每 seed 两项都过；其余路线只作次要对照 |
| 事后子组 | 五种 action profile 只用于预声明分层抽样，不据其事后挑选显著子组 |
| p 值误读 | 不报告孤立 p 值；报告效应量、配对 bootstrap CI 与工程门槛 |
| 统计显著但无工程意义 | 优效阈值 20%、非劣界 1.10 在 test 前冻结 |
| 独立性误设 | 统计单位为 episode，60 个时点不拆作独立样本；候选与基线使用同 episode 配对；seed 分别判定 |
| 将未显著当无效 | staged 未过非劣只写“未满足 1.10 界”；Stage-A 对比单独区分训练链有效与相对 joint 不足 |
| 分布假设失配 | 使用按 action profile 分层的非参数 paired bootstrap，不依赖正态误差假设 |
| 确认偏差 | 27 个候选、正/负对照和结构门禁全部回传；汇总器对预期 FAIL 仍退出 code 2，不择优隐藏 |
