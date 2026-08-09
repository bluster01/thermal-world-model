# Phase 3.5 审计回应⑤：1s 事件修复（5A）+ 参数摘要修复（5B）+ B 侧首次分析（2026-08-09）

> 对应审计 §5"下一轮 Linux 只做七件事"的 1/2/6/7 项，以及 P0-1/P0-2/P0-3/P1 全部硬问题。
> 判决：**A/B 侧 1s 事件协议现已可复算、可溯源；B 侧首次分析发现 SP 干预通道方向率
> ~50%（A 侧 73-78%）——B 侧二级阀通道物理响应更弱，A/B 差异为真实工程事实，
> 不是协议伪影。G3 参数摘要闭合：τ 全贴上界（真实秒 1059-1189s）、gain/rate 塌缩。**

---

## 1. 5A：1s 事件脚本修复（P0-1/P0-2/P1）

### 修复内容（`experiments/phase3_5/sp_events_1s_v2.py`）

1. **P0-1**：显式保存 `dv_3s/dv_10s/dv_30s/dv_60s/dv_180s/dv_600s` 六档，
   废弃误导性 `valve_dv_30s`（实为 3s）。合成时间轴单测 3/3
   （`tests/phase35/test_sp_events_v2.py`）逐个核对索引。
2. **P0-2**：参数化 `--side/--split/--input/--output`；事件写 `split` 标记
   （60/20/20 时间边界与 cache 一致）；输出 provenance（source SHA256、
   生成时间 UTC、git SHA、脚本路径、grid bounds）+ 拒绝漏斗
   （candidates/rejected_hold/rejected_window/kept）。
3. **P1**：`t0_ns` 存 epoch 纳秒（原为微秒）。

### 重跑结果

| 侧 | 候选 | 拒绝(hold) | 保留 | split (train/val/test) | S600 |
|---|---|---|---|---|---|
| A | 3106 | 2739 | 365 | 279/32/54 | 1 |
| B | 2524 | 2161 | 360 | 274/33/53 | 1 |

- A 侧 365 事件与旧版 t0 集合**完全重合**（365/365），确认事件检测无回归；
  旧版数字（含 compliance 82 事件 80.5% 方向率）为 3s 口径，已按审计要求不再引用。
- B 侧首次完整 1s SP 事件分析。
- 测试：`pytest tests/phase35` 28/28 通过。

## 2. B 侧稳态层分析（首次，同 A 侧协议）

B 侧 train+val 剔除大突变后 n=303（A 侧 306 等价口径）：

| 层 | A 侧 SP→阀位异号率 | B 侧 SP→阀位异号率 | A 侧 SP→ΔT 同号率 | B 侧 SP→ΔT 同号率 |
|---|---|---|---|---|
| 60s 严格 | 63.6% | 49.7% | 76.2% | 76.2% |
| 60sV (一级阀≤1) | 73.3% | 50.8% | 73.3% | 71.4% |
| 180s 中等 | 67.6% | 52.2% | 72.1% | 74.8% |
| 180sV (一级阀≤2) | 74.2% | 48.4% | 64.5% | 71.0% |
| 180s 严格 | 75.8% | 54.1% | 75.8% | 75.7% |
| **交集 60sV∩180sV** | **77.8%** | **45.0%** | 66.7% | 80.0% |

**解读**：B 侧温度确实按 SP 方向走（71-80%，与 A 侧同水平），
但二级减温阀**没有按 SP 反向响应**（异号率 ~50% 随机水平）。
即：B 侧 SP 阶跃后的温度变化**不是通过二级减温阀通道实现**的
（工况漂移/其他减温通道主导）。这与 E3 的 B 侧阀位事件方向率 0.057 互相印证。

A/B 差异是真实工程事实（可能原因：B 侧减温阀投入率低、手动模式占比高、
阀位 tag 语义不同），不构成协议伪影；但也意味着 **B 侧不能作为 A 侧的外部复现**，
审计 §5 的判断（"A/B 同属一台锅炉，不能包装成独立机组外部验证"）仍然成立。

## 3. 5B：G3 参数健康摘要修复（P1 §3.1）

### 修复内容

1. `model.py` forward 补返回 `rate_gain`（原缺失 → param_summary rate 全 null）。
2. τ 分 stage 报告并换算**真实秒**（10s 步 × 10）；`tau1_seconds/tau2_seconds`。
3. 排除 `free_only` 未训练分支（gain=-0.05/τ=180s 为初始化值，打标 `(free-only)`）。
4. 固定真实阀位扰动（+5% 开度）报告 action IRF（°C @600s），跨 opening map 可比，
   禁止直接比较 raw K。
5. checkpoint SHA256 + anchor 抽样 hash 溯源。

### 42-run 结果（非 free_only，36 runs）

| 指标 | 结果 |
|---|---|
| τ1/τ2 真实秒 | 均值 **1139s**（1059-1189s），两 stage 几乎相同 |
| τ 上界贴合 | 全部贴近 1200s 上界（tau_max=120 步） |
| gain near-zero | 中位 74%（delta_no_baseline 98-100% 塌缩） |
| rate gain | 6 个 rate 分支全部 ≈ -0.0000（塌缩） |
| IRF +5% 固定扰动 | 范围 -0.184 ~ 0°C；R50 最大（-0.11~-0.18），identity 最小（-0.06~-0.11）；全部负号（符号约束保持） |

**τ 单位修正的叙事升级**：真实秒 1059-1189s 全部贴 1200s 上界 →
干预动力学被推到 600s 预测窗**之外**（此前"107-119s"错误地看似在窗口内）。
与 gain/rate 塌缩共同构成 G3 FAIL 完整证据链。

## 4. 对 gate 账本的影响

| Gate | 状态 | 说明 |
|---|---|---|
| G0 代码 | ✅ | 28/28 测试，含新增合成时间轴索引单测 |
| G1 数据 | ✅ 改善 | 1s 事件 source SHA256/provenance/funnel 已备 |
| G2 事件 | A: 73-78% 方向率（train+val）；B: ~50% | A 侧可识别、B 侧不可识别；validation 单块事件数仍不足 |
| G3 模型 | **FAIL 闭合** | τ 全贴上界（真实秒）、gain 中位 74% 塌缩、rate 全塌缩、free-only 已排除 |
| G4/G5 | HOLD | 无候选；A test 已 exploratory，B test 继续冻结 |

## 5. 产物

- `experiments/phase3_5/sp_events_1s_v2.py`（v2 脚本）
- `tests/phase35/test_sp_events_v2.py`（合成时间轴单测）
- `results/phase35_sp1s_events_v2.json` / `_B.json`（A/B 事件 v2）
- `results/phase35_sp1s_events_v2_B_180s.json`、`results/phase35_sp1s_covars_v2_B_{60s,180s}.json`
- `src/phase35/model.py`（rate_gain 返回）、`experiments/phase3_5/param_summary.py`（v2）
- `results/phase3_5/param_summary_validation.json`（42-run 参数摘要 v2）
