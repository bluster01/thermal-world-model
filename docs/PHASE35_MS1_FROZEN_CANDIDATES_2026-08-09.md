# MS1 冻结候选与 synthetic test 授权清单 (2026-08-09)

> 依据: experiments/phase3_5/README.md §7 — 结构门禁全部通过后冻结候选,
> 单个 checkpoint 一次性打开 synthetic test。证据范围: synthetic known-truth,
> 不扩展到 A/B 真实数据 test。

## 1. 结构门禁审计 (18/18 PASS)

| 门禁 | 判据 | 结果 |
|---|---|---|
| reference_identity_max_error | =0 | 全 0 ✅ |
| future_action_leakage_max_error | =0 | 全 0 ✅ |
| finite_effect / finite_state | True | 全 True ✅ |
| spectral_radius (stateful routes) | <1 | graybox 0.954-0.957 / koopman 0.987-0.989 / pi_ode 0.959 ✅ |
| positive_step_terminal_effect | <0 (方向约束) | -0.159 ~ -0.203 全负 ✅ |

## 2. 冻结候选 (6 routes × 3 seeds = 18 checkpoints)

全部 18 个 checkpoint 满足门禁, 全部冻结并授权 synthetic test 单次访问:

| route_id | seeds | val MAE (mean±std) | 备注 |
|---|---|---|---|
| graybox_1p | 0,1,2 | 0.0190±0.0001 | 阶次失配正对照 |
| graybox_2p | 0,1,2 | 0.0160±0.0001 | 与真值同构 (inverse crime) |
| koopman_k2 | 0,1,2 | 0.0166±0.0001 | |
| koopman_k4 | 0,1,2 | 0.0166±0.0001 | |
| pi_ode | 0,1,2 | 0.0160±0.0001 | |
| causal_deeponet | 0,1,2 | 0.0160±0.0001 | fixed-horizon |

## 3. synthetic test 访问记录

| run | test MAE | 访问时间 | ledger |
|---|---|---|---|
| (执行后填写) | | | synthetic_test_access_ledger.json |

## 4. 边界声明

- 本次授权仅 synthetic known-truth test (seed+100_003 偏移生成), 重复访问被拒绝
- 不构成对 A/B 真实数据 test 的任何授权
- MS1 阳性只记 synthetic_method_feasibility, 不恢复 E3/E4 现场因果结论
