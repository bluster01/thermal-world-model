# PREREG：JEPA 状态增强 B1–B4（v1，2026-08-30）

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-30
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

本文件在 Linux 训练前冻结。完整实现设计见 `docs/plans/2026-08-30-jepa-b-series-design.md`，机器可读参数见 `configs/final_wm/jepa_b_series_v1.json`。原始结果无论正负均按本文件解释，不回填阈值。

## 研究问题

在保留 11 维可解释物理态、Fan2020-UDE transition、固定物理解码和动作盲闭包的条件下，四种 JEPA 借鉴是否改善验证集预测与长程状态演化：

- B1：未来观测表示是否能塑形当前物理状态；
- B2：4 维、每 6 步更新的慢状态是否能摘要 240–560 s 在途信息；
- B3：训练期富通道与 7 通道观测的对应轨迹交叉预测，是否优于固定错配；
- B4：物理/残差分解与 static/dynamic latent consistency 是否改善长程漂移。

## 输入权限与防泄漏

- 数据：`canonical_sideA_v2.npz`，仅 train/validation；test locked。
- 所有臂都使用 A5 操作质量门：`unit_load > 160 MW`、`1 < water_coal_ratio < 8`、`fuel_corrected > 50 t/h`，完整窗口必须位于连续有效段。
- B3 特权分支仅在训练期读取已注册 32 维富通道；部署与 validation 主预测只读 C0 的 7+2 接口。
- B1/B3 的 representation target 与 B4 的 residual target encoder 禁止未来动作；B3 未来动作只进入共享 predictor。B4 的 deterministic physical anchor 使用同刻已记录动作完成物理反演，不进入 residual encoder，也不构成部署输入扩张。
- 标准化统计只拟合 train；validation 不参与均值、方差或模型选择外的任何拟合。
- B3-SHUFFLE 使用预计算可用窗口池上的固定无不动点循环移位；每条富通道历史与未来保持为完整但错误的另一条轨迹。

## 冻结预算与权重

- seed：仅 0；成功只触发后续固定 seeds 1/2，不构成支持结论。
- 训练：120 epochs，patience 20，batch 32，200 batches/epoch，Adam `1e-3`，grad clip 10。
- history/H：96/18；B2 slow dim/stride：4/6；B1 dim：8；B3 dim：16；B4 residual dim：4。
- 总损失：主 observation NLL 权重 1；JEPA 预测 0.1；Gaussian-CF 0.01；B4 static 0.05、dynamic 0.05。权重不搜索。
- 高斯正则为仓内实现的固定切片特征函数匹配（16 slices、17 knots、固定 seed），明确标为 SIGReg-style adaptation，不声称逐行复现 LeJEPA 官方实现。

## 预注册输出与裁定

主裁定采用 validation 上与 C0 完全配对的 256 窗：末端测点第 18 步 MAE 至少改善 5%，四负荷箱的同一末端误差极差恶化不超过 10%，且两阀 H18/H60 v0.3 方向门全过。五测点/全时程 MAE同时报告但不混入主门。若主 MAE 反而恶化至少 5%或任一方向门失败，记 `REJECT_EXPLORATORY_SEED0`；三门全过仅记 `PROMOTE_TO_FIXED_SEEDS_1_2`；其余记 `INCONCLUSIVE_EXPLORATORY_SEED0`。

B3/B4 另外报告 H36/H60 末端误差、相对 persistence 的 drift 与 UTC-day bias，但不设事后阈值。B3-SHUFFLE 只检验对应关系是否承载信息，不参与晋级。任何结果都不得恢复旧 CVAE-B3、不得宣称工业 JEPA 成熟、不得升级当前论文正式 verdict。

## 原文锚点

- Balestriero & LeCun, *LeJEPA*, arXiv:2511.08544v3。
- Maes et al., *LeWorldModel*, arXiv:2603.19312v3。
- Wen et al., *JEPA-x: Cross-Predictive Physics Grounding for Forecastable Latent Dynamics*, arXiv:2608.24044v2。
- Nie et al., *Phys-JEPA*, arXiv:2606.16076v1。

这些文献只提供机制和术语来源。JEPA-x 是机器人操作域三种子研究；Phys-JEPA 明示主表单种子且使用弱 descriptor。二者均不替代本项目自己的工业过程验证。
