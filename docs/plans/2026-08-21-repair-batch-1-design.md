# 物理修复批 ① 设计稿（矩阵修正案 v0.3 子项，2026-08-21）

> 状态：设计冻结。靶：H1 精度地板（sh1_in/sh2_in H1 MAE 3.7/3.9°C，14.5×/21.1×
> persistence）与出口通道锚定缺口（t=0 偏差 −18.1/−6.3°C）。
> 诊断依据：`experiments/final_wm/h1_anchor_decomposition.py` 对
> t1_closure_cons_seed0（修复批后 checkpoint，IAPWS 网格物性）512 窗实测，
> 原始 JSON 在 `results/final_wm/h1_anchor_decomposition_seed0.json`。

## 0. 根因分解（实测，非推测）

| 通道 | t=0 锚 MAE | H1 MAE | persistence | H1/pers |
|---|---|---|---|---|
| sh1_in | **0.0013（精确）** | 3.70（bias +2.68） | 0.255 | 14.5× |
| sh1_out | **18.81（bias −18.08）** | 1.37 | 0.451 | 3.0× |
| sh2_in | **0.0011（精确）** | 3.91（bias +1.46） | 0.185 | 21.1× |
| sh2_out | **6.45（bias −6.34）** | 0.81 | 0.297 | 2.7× |
| final | **0.0008（精确）** | 0.15 | 0.089 | 1.7× |

- 三点焓锚（ch0/2/4）在 t=0 构造性精确（网格物性 roundtrip 最差 0.003°C，
  已排除插值误差假说）；
- 出口通道（ch1/3）锚不定：喷水侧状态（dsw_lag, m_liq）按瞬态平衡初始化，
  与实测出口温度不自洽 → 偏差 −18.1/−6.3°C；
- H1 的入口通道灾难来自 **hybrid 融合**：学习后验在全状态维上与精确锚做精度
  加权，把 h 锚拖偏（首步 bias +2.68/+1.46°C）——这正是 O1 learned MIXED /
  hybrid REJECTED 的机制通道。

## ①-A 全五点锚定（`transition.initial_steady_state`）

三点焓锚不变；新增：以实测出口温度**反演喷水侧状态**。

对每个减温器 i∈{1,2}，给定已锚定的入口焓 h_in、边界 (D, p, tfw) 与实测出口
温度 T_obs，求解 lag_i 使得 `output_temperatures` 的对应通道等于 T_obs：

```
T_out(l) = tsat + dry(l) · (T(p, hm(l) + q_w(l)/D) − tsat)
dry(l) = sigmoid(3·(m_dry0 − l·τ_evap)/m_dry0)
hm(l)  = (D·h_in + l·h_spray)/(D + l)
q_w(l) = 再湿契约项（m = l·τ_evap 平衡库存）
```

`T_out(l)` 在 l≥0 上单调不升（喷得多→更冷；干点→干态直通；饱和端→tsat），
用 **24 次二分迭代**（batch 向量化）求解：

- T_obs ≥ T_out(0)（干端）：lag=0，残差不匹配记录于返回值诊断位；
- T_obs ≤ tsat（饱和端）：lag=l_max=100 kg/s，同样记录；
- m_liq = lag·τ_evap、lag 状态 = lag（与修复②契约一致）；
- 二分在 `torch.no_grad()` 下进行：喷水路径参数（th/varphi）经动力学步照常
  接收梯度，仅初态 lag 变为数据反演（本批语义即"锚定"）。

成功标准（契约测试）：实数据 512 窗 `g(x0,b0) − obs_0` 五通道 |bias| < 0.05°C
（超界窗口以 clamp 记录，允许存在但需计数）。

## ①-A 增补（实施中实测发现，2026-08-21）：干湿度混合端点固定

二分反演落地后真数据复测：ch3 偏差 −6.3→−0.38°C，但 ch1 卡在 −6.2°C 不动。
根因定位：原干湿度混合 `sigmoid(3·(m_dry0 − w)/m_dry0)` 在**零喷水处留 4.76%
湿漏**（sigmoid(3)=0.953），按典型过热度 ~125°C 折算恰为 ~−6°C 人工冷却——
真实零喷水出口低于模型干态地板，反演不可达。**修正**：端点固定为
`dry = sigmoid(6 − 11·w/m_dry0)`（dry(0)≈0.998，dry(m_dry0)≈0.007，中间单调）。
修后实测（512 窗，修复①栈、未训练参数）：

| 通道 | t=0 锚 MAE（修前→修后） |
|---|---|
| sh1_in / sh2_in / final | 0.001 → 0.001（精确保持） |
| sh1_out | 18.81 → 3.64（残余：一级零喷水偏移，见下） |
| sh2_out | 6.45 → **0.38** |

**残余登记**：sh1_out 残余 −3.6°C bias 为结构性——厂侧 sh1_out 在零喷水时低于
sh1_in 约 3.6°C（疑似一级阀座泄漏或级间散热/测点偏移，模型干态地板
T_in−0.3°C 够不到）。列入 AE 阶段候选（一级泄漏通道或出口偏移项），
本批不再扩范围。

## ①-B 后验改为锚定修正 + 压力分段（`observer.py` / `model.py`）

按架构路线登记（v0.3 §3"观测端稳态锚全锚定 + 慢偏置，不学习快状态后验"）：

1. **后验参数化改为锚定相对修正**：`mu = anchor_5pt + δ`，
   `δ = state_scale ⊙ 0.1·tanh(head(hidden))`，零初始化 → 未训练观测器
   精确返回锚（消除初始化阶段的 O1 退化通道）；
2. **修正掩码**（模式语义重定义，O1 随重跑重释）：
   - `steady`：纯五点锚；
   - `hybrid`：δ 仅作用于**非锚定慢状态**（tm×3, rb, latent）；h/dsw_lag/m_liq
     掩码为 0（快状态后验已被 O1 否决，不再学习）；
   - `learned`：δ 全维（含 h）——保留作为对照臂，检验"全维修正"是否仍退化；
3. **压力分段**：GRU hidden 拼接压力工况特征后入 δ 头——
   `softplus((pm − 22.064 MPa)/1 MPa)` 与 `softplus((22.064 − pm)/1)`（临界点
   两侧的软指示，亚临界/超临界 regime 的 T-h 反演灵敏度不同）+ 归一化 pm；
   σ 头不变（全维，softplus 下界）。

## 影响面与纪律

- `config_fingerprint` 变化（observer 结构 + initial_steady_state）→ 旧 checkpoint
  失效为预期；重跑范围按修正案：**T1 closure_cons×3 + O1（3 臂×3 seed）+ R1**，
  预算/seeds/判决纪律不变（T1 仍 60/10 early-stop-10，closure_cons 单臂裁定维持）；
- D-SYN 门禁须先过（学生仍须砍骨架 NLL ≥30%）；
- R1 判决沿用现行冻结判据（方向双探针 + blindness + leakage 单 shuffle 5pp 门）；
- 泄漏根因（seed1）不在本批范围，列 AE 阶段；
- 档案兼容：audit_verdicts.py 复算的是 v0.2 冻结产物，不受新代码影响
  （新产物落新指纹目录）。

## 执行排期（对齐 docs/plans/2026-08-21-fmts-schedule-and-protocol-plan.md）

- 8/21：本地实施 + 契约测试 + D-SYN quick 门禁；
- 8/22-8/23：Hermes 重跑 T1 closure_cons×3 → R1 → O1 三臂（~10-12h GPU）+ auditpack；
- 8/23 晚：判决落档，论文修订周开始。
