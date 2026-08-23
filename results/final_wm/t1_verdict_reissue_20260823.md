# T1 比较判决重发（生产栈，2026-08-23）

方法：runner 冻结函数 `relative_improvement_ci`（H18 NLL，UTC 日块自助 1000 次）
+ `_seed_passes`（THRESH_T1_NLL=0.02 且 ci_lo>0）+ `_verdict`（MIN_SEED_PASSES=2），
对已入档 metrics 文件只读重放；不改 runner、不重训。基线：执行侧 5fa5623 的
physics_only ×3（best_val 1.260/1.290/1.260，stop=cap×3；commit 4b03ebf，
无 RESUMED，arm-filter 纪律保持——summary 未被写）。

## 官方判决（设计侧复核，与执行侧预演逐位一致）

| 比较 | seed0 | seed1 | seed2 | 判决 |
|---|---|---|---|---|
| **closure_cons_norew vs physics_only** | −0.178 CI[−0.262,−0.114] ✗ | +0.069 CI[+0.016,+0.117] ✓ | −0.013 CI[−0.076,+0.036] ✗ | **MIXED（1/3）** |
| closure_cons（intact）vs physics_only | +0.024 CI[−0.026,+0.070] ✗ | +0.038 CI[−0.007,+0.079] ✗ | −0.002 CI[−0.050,+0.042] ✗ | **REJECTED（0/3）** |

n_days=13/seed。

## 含义（必须如实进入论文叙事）

1. **v0.2 的"T1 closure SUPPORTED（2/3，+5.7%/6.9%）"在修复①栈上不成立**：
   锚定修复大幅强化了纯物理基线（H1 MAE 1.09-1.11°C），闭包的精度增益被吃光。
   v0.2 判决是旧栈产物——闭包当年的"增益"部分来自被锚定 bug 压制的物理基线。
2. **第三个负结果**：修复观测模型后，学习残差闭包在侧A H18 NLL 上**无显著增益**
   （norew 栈 MIXED，intact 栈 REJECTED）。再湿通路不仅是不可辨识，
   整个闭包的精度价值主张在侧A站不住。
3. **seed0 混淆**记录：norew seed0 显著劣于其物理基线（−0.178，CI 不含 0），
   与其 41ep 早停弱 seed 一致；physics_only 三 seed 全部撞 60ep 上限
   （仍在下降）——统一预算下物理基线或被低估，判决对闭包已是宽松方向。
4. **生产臂地位**：norew 生产臂（裁定 A）理由随之修正——不再是"闭包更准"，
   而是"精度与物理基线平价 + 完整证件链（O1/R1/leakdist/auditpack）仅存于
   闭包栈"。physics_only 未经 R1 证件检验。
5. ~~信息量最大的下一步~~ **已执行并裁定**（2026-08-23 11:1x，用户批准）：
   R1 探针跑 physics_only 栈。runner 两处最小修补（无闭包模型 leakage/
   residual_quantiles 标记 skipped 而非崩溃——探针在该栈上语义空缺），
   132/132 回归通过。

## R1 physics_only 结果（本地全档，r1_report_physics_only.json）

| seed | blind | 60s 方向 frac_neg | 240s 稳态 frac_neg | leakage |
|---|---|---|---|---|
| 0 | ✓ | 1.000（−0.198） | 1.000（−0.302） | skipped（无闭包，问题空缺） |
| 1 | ✓ | **0.844（27/32）✗** | 1.000（−0.387） | skipped |
| 2 | ✓ | 1.000（−0.167） | 1.000（−0.438） | skipped |

**判决：R1 physics_only = REJECTED**（seed1 瞬态方向门 0.844 < 1.0 冻结门）。

## 三方对照——闭包价值主张的终审

| 栈 | T1 精度（vs physics_only） | R1 方向门 |
|---|---|---|
| physics_only | （基线） | **REJECTED**（seed1 27/32 瞬态反号窗） |
| closure_cons（intact） | REJECTED 0/3 | REJECTED（seed0 28/32） |
| **closure_cons_norew（生产臂）** | MIXED 1/3（平价） | **SUPPORTED 3/3（32/32×3）** |

**norew 闭包栈是唯一同时满足精度平价与方向证件的配置。** 纯物理核对阀门
瞬态响应存在错误符号窗（其稳态方向正确，失败仅在 60s 控制相关瞬态）——
学习残差闭包（去再湿）承担的正是这段瞬态校正。生产臂（裁定 A）的最终
理由就此闭合：**精度平价 + 唯一方向证件**。

## 论文措辞建议（供解冻修订）

"修复观测模型后，学习闭包在侧A不再提供显著精度增益（T1 重发 MIXED/REJECTED）；
但纯物理基线在控制相关瞬态视界存在方向性错误窗（R1 REJECTED，seed1 27/32），
而再湿消融闭包是唯一通过方向证件门的配置（SUPPORTED 3/3）。生产配置因此
保留保守闭包（去再湿）：其价值不在精度增量而在干预响应的方向保真。"
