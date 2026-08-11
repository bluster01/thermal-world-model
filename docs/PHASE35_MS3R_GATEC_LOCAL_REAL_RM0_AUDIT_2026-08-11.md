# Phase 3.5 MS3-R Gate C 本地真实 1% RM0-A 审计

日期：2026-08-11

## 监督结论

`AUDITED / REAL_1PCT_RM0_UNDERFIT / NO_ROUTE_RANKING / PERSISTENCE_ANCHOR_MISSING / RESPONSE_BRANCH_WEAK / NO_LINUX_RELEASE`

本批首次使用冻结真实 A/B cache，而不是合成真值。四条路线均完成单 seed、60 optimizer updates、1/100 train anchors 和 1/100 validation anchors 的 CUDA 运行；数值、前缀因果、恒动作 identity、未来真值隔离和稳定 pole 检查均通过，且没有访问 test。

但该批不支持任何路线排名，也不支持“真实物理响应模型已验证”。

## 数据与执行闭合

- 源数据 SHA-256：`85a3f92648d5f88a4543f500859b200207fb55a32555900ca88f7c339c4e4da6`；
- 执行代码：`54687ac1c77ce2f7e8a24b5fd2b2e6f7df3bf5da`；
- 完整可用 train/validation anchors：695,780 / 229,785；
- 冻结子集：6,957 / 2,048；四路线 anchor SHA 完全一致；
- 环境：本地 `ALLoftime`、CUDA；每路线约 11–15 s，峰值显存约 67 MB；
- ledger 中 6 项运行产物逐项 SHA-256 复核一致。

## 真实结果与持久性基线

| 路线 | composite | valve MAE | Tin MAE °C | local-drop MAE °C | terminal MAE °C | predicted effect °C | logged-action effect °C |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1phys three-pole | 0.461654 | 3.8461 | 1.9963 | 5.5666 | 1.4739 | 0.00558 | 0.02524 |
| stable LPV-Koopman | 0.461439 | 3.8453 | 1.9971 | 5.5671 | 1.4728 | 0.00269 | 0.01305 |
| PI neural ODE | 0.461700 | 3.8452 | 1.9969 | 5.5735 | 1.4728 | 0.00508 | 0.02354 |
| causal DeepONet | 0.461952 | 3.8452 | 1.9981 | 5.5718 | 1.4742 | 0.00437 | 0.02100 |
| persistence baseline | — | 3.8140 | 2.0571 | 2.2995 | 1.4744 | — | — |

关键比较：

1. local-drop 比持久性差约 142%，说明当前 absolute residual local head 尚未学会最基本的局部状态基线；
2. terminal 与持久性几乎相同，oracle Tin 也没有系统改善，说明瓶颈不是单纯 future Tin forecast；
3. valve 略差于持久性，Tin 只比持久性改善约 3%；
4. 四路线 composite 最大差仅约 0.00051（约 0.11%），远小于可解释为路线差异的尺度；
5. predicted-action response 只有 0.0027–0.0056°C，logged-action response 也只有 0.013–0.025°C。`>1e-6` 的软件 non-collapse 门过松，只能排除精确零，不能作为科学门。

## 原因判断

这是一个有信息量的失败，不应通过扩大到 1/10 或直接增加 seed 掩盖：

- residual local head 从随机值直接预测约 10°C 的绝对 Tin−Tout，而不是从已观测基线温差预测增量；
- valve、Tin 和 terminal decoder 也未从 persistence identity 初始化；
- response 主 rollout 使用尚未学好的 predicted valve，削弱了动作梯度；
- local absolute loss 可被 residual/free 分支解释，不能确保 response 获得真实动作解释权；
- 60 updates 是工程 smoke，不足以形成方法排名。

## RM0-B 修正门

下一批仍使用真实 1/100，不先扩大数据：

1. valve、Tin、local-drop、terminal 全部改为 baseline + learned increment；
2. local residual 以当前 `Tin−Tout` 为显式 persistence anchor，输出层近零初始化；
3. logged future valve 只进入 response auxiliary，不进入 encoder/residual；同时保留 predicted-valve deploy rollout；
4. selector 增加相对 persistence 指标；local-drop 至少不得明显差于 persistence，terminal 必须报告 effect 而不能靠持平过门；
5. response non-collapse 改为相对工程/数据尺度诊断，不再使用 `1e-6` 作为科学阈值；
6. RM0-B 使用新的冻结配置和一次性预算，不重跑 RM0-A。

合成真值继续保留为理论可解性与负控制，不能替代上述真实值门禁。
