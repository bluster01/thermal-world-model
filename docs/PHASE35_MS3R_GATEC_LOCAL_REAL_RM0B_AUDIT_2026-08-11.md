# Phase 3.5 MS3-R Gate C 本地真实 RM0-B 审计

日期：2026-08-11

## 监督结论

`AUDITED / BASELINE_ANCHOR_EFFECTIVE / PERSISTENCE_DIAGNOSTICS_PASS / TERMINAL_NO_GAIN / RESPONSE_DECOMPOSITION_NONUNIQUE / NO_ROUTE_RANKING / NO_LINUX_RELEASE`

RM0-B 使用与 RM0-A 完全相同的真实 1/100 train/validation anchors 和 seed 0，但采用新协议：observed persistence baseline + learned increment、180 optimizer updates，以及 logged future valve 仅进入 response auxiliary。它不是 RM0-A 重跑。

## 结果

| 路线 | composite | local MAE °C | local/persist | terminal MAE °C | terminal/persist | predicted effect °C | logged effect °C |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1phys | 0.319393 | 2.2238 | 0.9671 | 1.4757 | 1.0009 | 0.0175 | 0.0438 |
| LPV-Koopman | 0.319391 | 2.2239 | 0.9671 | 1.4760 | 1.0011 | 0.0150 | 0.0390 |
| PI neural ODE | 0.319263 | 2.2215 | 0.9661 | 1.4757 | 1.0009 | 0.0255 | 0.0637 |
| causal DeepONet | 0.319206 | 2.2206 | 0.9657 | 1.4759 | 1.0010 | 0.0377 | 0.0968 |
| persistence | — | 2.2995 | 1.0000 | 1.4744 | 1.0000 | — | — |

共同结果：valve/persistence 约 0.975，Tin/persistence 约 0.956，local/persistence 约 0.966–0.967；说明 baseline parameterization 修复了 RM0-A 的主要训练失败。

## 为什么仍不能排名路线

1. composite 最大差仅 `0.000187`，约为分数本身的 0.06%；
2. 与此同时 logged-action response 幅值相差 2.49 倍，predicted-action response 相差 2.51 倍；
3. 更大 response 幅值没有带来可分辨的 local/terminal MAE 改善；
4. terminal 没有优于 persistence，oracle Tin 也没有形成改善；
5. 单 seed、1/100、180 updates 只能作 RM0，不具备统计排名能力。

因此相同预测误差容许多个动作响应分解。DeepONet 幅值最大不能解释为“物理效果更强”或“方法更好”。

## 下一步

暂缓四 operator 路线赛马，进入真实 RM1-A attribution：

1. 固定 A1phys，比较 `paired_free`、additive base、scheduled small/base/large、scheduled terminal-only；
2. 所有候选使用同一真实 1/100 anchors、seed 和预算；
3. 同时报告 total/local/terminal 相对 persistence、predicted/logged response 幅值和 shuffled-action 诊断；
4. 若 free capacity 增大时 MAE不变但 response持续缩小，判为分解不唯一；
5. 只有 attribution 稳定后，才恢复 operator route 比较。

本批无 test 访问、无自动科学 PASS、无 Linux 授权。
