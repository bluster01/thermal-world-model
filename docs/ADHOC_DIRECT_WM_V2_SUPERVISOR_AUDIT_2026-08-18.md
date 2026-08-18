# Direct-WM v2 Supervisor Audit

## 判决

```text
AUDITED / DEVELOPMENT-ONLY DIRECT-H18 ENDPOINT /
PREDICTION SIGNAL SUPPORTED /
DUAL-ACTION PHYSICAL RESPONSE NOT SUPPORTED /
NO TEST / NO RESPONSE CHAMPION
```

审计对象为孤立分支 `adhoc/lumped-enthalpy` 的 `81310b1..57fa018`，执行结果位于 `out/direct_wm_v2/`。本审计不把该分支并入主仓历史，只将代码、manifest 和小型结果摘要作为最终世界模型的证据输入。

## 1. 协议闭合

- 6/6 单元完成：F0/F1 × seeds 42/0/7；
- train、validation、evaluation 时间块不重叠，checkpoint 仅按 validation H18 MAE 选择；
- 固定 200 个 evaluation anchors，三个 seed 共享索引；
- reserved `[40000,50000)` 声明为未加载；
- `ef1d366` 仅为 `eval_sensitivity` 增加 `@torch.no_grad()`，没有改变模型、split、seed、dose 或 selector；
- checkpoint 未保留，故本次能复算 JSON 算术和代码合同，不能进行权重级 replay。

## 2. 预测结果复算

`rollout_mae` 是旧字段名，实际为一次前向同时输出 18 步的 direct conditional forecast，不是递归 rollout。

| Fold | step 0 MAE 均值 | step 17/H18 MAE 均值 | seed SD(H18) |
|---|---:|---:|---:|
| F0 | 0.2254°C | 0.7122°C | 0.0021°C |
| F1 | 0.2568°C | 0.9269°C | 0.0185°C |

数字与 Linux 记录的 `0.712/0.927` 一致。模型使用真实未来双阀位，因此该结果只能称 `logged-future-action conditional oracle`，不能称部署预测，也不能和递归物理模型的 H60 RMSE 直接比较。

## 3. 动作响应修正

喷水阀开大在长期方向上应对应降温。H18 平均敏感性复算如下：

| Fold / action | `-2%` 均值 | 方向正确 seeds | `+2%` 均值 | 方向正确 seeds |
|---|---:|---:|---:|---:|
| F0 / 一级阀 | -0.00215°C | 0/3 | +0.00542°C | 0/3 |
| F0 / 二级阀 | -0.00004°C | 1/3 | +0.00094°C | 0/3 |
| F1 / 一级阀 | -0.01395°C | 1/3 | +0.03972°C | 0/3 |
| F1 / 二级阀 | +0.00476°C | 2/3 | -0.01261°C | 3/3 |

因此远端文档中“F1 `-0.005~-0.015°C`、方向正确但弱”的表述只覆盖 F1 二级阀 `+2%`，遗漏一级阀持续反向以及 F0 不成立。可守结论是：

> Direct-WM 的动作通道总体较弱，而且方向随 fold、阀门和 seed 变化；它没有恢复稳定的双动作物理响应。

物理参考本身仍存在 `-0.036°C/2%` 与 `-0.45~-0.87°C/2%` 的口径冲突；在参考统一和事件分布 CI 缺失前，不保留“弱 25–450×”这一精确倍数，只保留“远小于候选物理对象响应且不稳定”的描述。

## 4. 对最终 Pipeline 的含义

该实验支持把高容量 Direct-WM 当作历史 encoder/预测 backbone 候选，但尚未证明它是：

- 可延续物理状态的概率 observer；
- Tin、负荷、压力等未来边界的 boundary model；
- 经过 coverage/calibration 验证的概率模型；
- 可用于动作替换的 response transition。

尤其是代码虽然输出 `mu/log variance`，结果没有 NLL、coverage、CRPS 或校准曲线，因此“σ 供应方”目前只是待验证接口，不是已有证据。

## 5. 后续处理

- 不再为旧 Direct-WM 增加 seed 或 probe；
- 不访问 reserved/test；
- 在最终 pipeline 的 O1/B1 中分别验证 state initialization 和 future boundary；
- 动作响应统一交给显式 Fan2020-UDE transition，并继续接受 wrong/lead/shuffled 和支持域门禁。
