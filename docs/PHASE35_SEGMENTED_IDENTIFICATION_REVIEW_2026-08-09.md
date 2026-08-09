# Phase 3.5 分段辨识审计与验证协议（2026-08-09）

> 审计对象：`experiments/phase3_5/segmented_identification.py`、
> `results/phase35_segmented_identification.json`、Cao et al. (2021)。
> 本地已在同一份 `all_merged_10s.csv` 上精确复现结果 JSON。

## 1. Supervisor 决定

分离“阀门→二过出口”的导前区与“二过出口→末过”的惰性区，方向正确；但当前
85%/74% K<0 率与 0 s 脉冲峰值只能列为探索性 pilot，不能写成“传递函数级确认交叉布置”。

当前结论状态：

- 代码/数字可复现：`VERIFIED`；
- 交叉拓扑物理结论：`NOT VERIFIED`；
- Phase 3.5 E3：保持 `INCONCLUSIVE`；
- E4：继续 `BLOCKED`。

## 2. 必修问题

### P0-A：阶跃符号丢失

当前代码用 `abs(u)` 归一化响应：

```python
ys = (y - y0) / max(abs(u), 1e-9)
```

这会把开阀和关阀放进同一方向。K<0 不再等价于“开阀导致降温”，而只是事件后温度
是否下降。必须使用有符号剂量，或用 `sign(delta_v)` 显式方向化。

### P0-B：事件不是可辨识阶跃

现协议只要求阀位事件前 600 s 安静、事件后 20 s 近似稳定，但 FOPDT 拟合使用 300 s
输出。必须二选一：

1. held-step 主分析：阀位在完整 300 s 后窗保持；
2. trajectory 次分析：用真实阀位轨迹驱动离散 FOPDT/ARX，不再假设恒定阶跃。

事件前还必须同时检查负荷、主蒸汽压力和两侧主汽温稳定；时间缺口必须切断事件。

### P0-C：当前“脉冲响应”不是系统脉冲响应

`mean(du * dy_lag) / var(du)` 是未中心化的滞后互相关，不是预白化、ARX、FIR 或
工具变量辨识。闭环中它会保留共同扰动、反馈和同步测量造成的 0 s 峰值。因此：

- 不得再称“差分脉冲响应”或“预白化”；
- 另一侧温度不是合法负对照，两侧共享炉膛和负荷扰动；
- 0 s 惰性区峰值必须视为 common-mode 警报，而不是“即时金属传导”。

### P0-D：缺少真正的验证集

事件拟合和 R²均来自同一窗口。Cao2021 使用前 3000 点辨识、后 3000 点验证，并执行
滤波/去漂移和零初始处理；论文最终导前区、惰性区模型均为二阶惯性环节。当前脚本固定
一阶且没有 held-out 验证，不能宣称复现论文方法。

### P1：现有数字不支持“交叉拟合更好”

| 阀门 | 交叉 R² 中位数 | 非交叉 R² 中位数 |
|---|---:|---:|
| A | 0.19 | 0.32 |
| B | 0.19 | 0.23 |

A 阀仅 20 个事件，B 阀 172 个。只比较 K 的符号率、忽略验证误差，属于选择性判据。

## 3. 冻结的验证问题

### H1：导前区交叉占优

- A 标签阀位主要影响右侧二减温降；
- B 标签阀位主要影响左侧二减温降。

主输出不得只用出口温度，改用近端物理温降：

```text
drop_left  = T_in_left  - T_out2_left
drop_right = T_in_right - T_out2_right
```

### H2：惰性区同侧传递

- 右二过出口主要预测右末过；
- 左二过出口主要预测左末过；
- 有效模型必须具有非零纯迟延，且在独立时间块上优于错侧和 placebo。

### H3：控制层与 plant 层分开

SP、主调输出、阀位反馈、二过温度和末过温度分属不同层级。验证 H1/H2 时不得把
SP 当作 plant action；完整闭环链另行报告，不能用同一张冠军表替代。

## 4. V0–V4 验证矩阵

| Gate | 输入/比较 | 主统计量 | 放行门槛 |
|---|---|---|---|
| V0 数据与事件 | 10 s 连续网格；前 600 s 工况；双阀动作 | 事件数、方向、日期、缺口、pretrend | 每回路至少30事件、8独立日；否则 inconclusive |
| V1 导前区 2×2 | A/B 阀 × 左/右 `Tin-Tout` | 日块 bootstrap 的交叉减同侧响应 | A→右、B→左的95% CI均大于0 |
| V2 FOPDT/ARX | no-action、同侧、交叉、完整 MIMO | held-out NRMSE/R²、K/T/tau、基线提升 | 交叉验证误差占优；K物理方向正确；无边界塌缩 |
| V3 惰性区 | 左/右二过历史共同预测左/右末过 | blocked forecast loss、延迟、2×2系数 | 同侧项在独立块占优；最优延迟不能为0 s |
| V4 伪证与复现 | 输入错移±1 h、日内置换、错侧；负荷/开关/月分层 | 配对差、分层方向、placebo排名 | 真配对优于全部placebo，主要分层不反号 |

## 5. V0 事件定义

主分析阈值冻结如下：

- 采样间隔必须连续为 10 s；缺口前后窗口全部排除；
- 负荷 > 250 MW；
- 事件前 600 s：负荷 range ≤ 15 MW；主汽压力 range ≤ 0.4 MPa；两侧末过温度
  range ≤ 3°C；动作阀位 range ≤ 0.3%；
- 阀位剂量：`median(post 30–60 s) - median(pre 60 s)`，保留正负号；
- 主阈值 `|delta_v| >= 3%`，2%/5% 仅作预声明敏感性；
- 另一阀同期变化 `< max(1%, 0.5*|delta_v|)`；
- 独立事件间隔至少 600 s；
- held-step 分析要求后 300 s 内阀位保持在 `max(0.5%, 0.2*|delta_v|)` 范围；
- 同时报开阀、关阀数量，不允许用一侧方向外推另一侧。

不得以事件后的温度、负荷或压力结果筛选事件。

## 6. V1 主统计量

对侧 `s` 的有符号温降响应：

```text
R_s(h) = sign(delta_v) * [drop_s(t+h) - mean(drop_s, pre-window)]
```

配对主对比：

```text
A valve: R_right - R_left
B valve: R_left  - R_right
```

主终点为 30–300 s 响应面积；60/120/180/300 s 为冻结的轨迹诊断。置信区间按 UTC 日
或连续 episode 聚类 bootstrap，不能把采样点或同日密集事件当作独立样本。

## 7. V2/V3 模型要求

- 输入必须保留有符号剂量；
- held-step 可以拟合阶跃 FOPDT；trajectory 必须使用真实阀位输入序列；
- 阶次 `n in {1,2,3}`、延迟和阀门变换只在开发块选择；
- 验证块只评估一次；禁止按 validation/test 回调结构；
- 同报无动作基线、旧同侧、交叉、完整 2×2 MIMO；
- 惰性区同时输入两侧二过温度，并控制负荷、压力和温度历史；
- placebo 应使用时间错移/日内置换，不把另一侧简单称为负对照；
- 参数必须报告全部拟合，禁止先筛 K<0 再计算 K/T/tau 中位数；
- 输出 opening/closing、fit failure、边界命中、逐日分数与 bootstrap CI。

## 8. 数据切分与证据等级

当前全量数据及五月数据均已被用于形成交叉假设，不能再称独立 lockbox。当前阶段只能做
blocked internal validation：

1. development：2025-12 至 2026-03；
2. internal validation：2026-04；
3. 2026-05：仅作已查看的 robustness block；
4. 真正 confirmatory 证据必须使用未来新增时间块或另一机组。

## 9. Linux 必须回传的产物

```text
results/phase35_segmented_v2/
  run_manifest.json
  data_audit.json
  event_manifest.jsonl
  leading_response_by_event.csv
  leading_summary.json
  blocked_model_scores.csv
  parameter_health.json
  placebo_summary.json
  bootstrap_summary.json
  console.log
```

`run_manifest.json` 必须含脚本 git SHA、原始 CSV SHA256、列清单、时间范围、阈值、切分、
Python/依赖版本和完整命令。Linux 只执行冻结脚本，不调整阈值，不筛结果，不自行改结论。

## 10. 论文表述边界

V0–V4 未全部通过前，只能写：

> 分段分析发现与交叉回路一致的探索性信号，但现有闭环观测和事件支持不足以确认物理拓扑。

只有在未来未查看时间块上复现，且真实配对同时优于错侧、无动作基线与 placebo，才可写成
“交叉占优的 2×2 动态通道得到验证”。即使通过，也不能把阀位开度等同于喷水质量流量。
