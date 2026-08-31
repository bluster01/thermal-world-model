# Phase 3 收口论文设计稿

> **历史设计 / 已由 2026-08-09 Phase 3.5 审计取代。** 本文关于“唯一物理正确响应”“n_lag=2 已由 DiD 辨识”“因果保真度 Pareto 最优”等表述没有通过当前 E3 common-support、balance、独立 test 与参数健康门禁，不得直接用于论文。当前可守叙事见 [`PHASE3_5_LINUX_RETURN_AUDIT_2026-08-09.md`](PHASE3_5_LINUX_RETURN_AUDIT_2026-08-09.md)；从预测器到最终世界模型的证据缺口见 [`WORLD_MODEL_EVIDENCE_LADDER.md`](WORLD_MODEL_EVIDENCE_LADDER.md)。

> 版本：2026-08-08
> 定位：Phase 3 (因果架构评估) 的收口论文，作为 Phase 4 (Fan20 物理建模) 的前置交付物
> 状态：设计稿，待 cwoc 确认后进入实验补全和写作

---

## 0. 一句话定位

在观测闭环数据下，黑箱世界模型学到的是条件期望 E[T|a] 而非干预效应 P(T|do(a))；我们用结构零动作不变量 g(x,0)≡0 和数据辨识的灰箱先验，使世界模型在不牺牲预测精度的前提下获得物理正确的干预响应形态。

---

## 1. 论文类型

**Technique Paper**

Key Idea 承载叙事：结构分解 T̂ = f_free(x) + g_phys(x,a) 加零动作不变量，使干预分支输出等于反事实差，可直接与观测事件响应对齐。这不是"又一个模型好一点"，是"唯一一个物理响应形态正确的模型"。

---

## 2. Thinking Template

| Stage | Content |
|-------|---------|
| **Research background** | 火电主汽温控制是深度调峰下的关键瓶颈。世界模型(WM)学习动态转移，支持前瞻预测，已有工作报告了良好开环精度(MAE 0.31-0.42°C)。但工业部署需要模型不仅预测准，还要对运行干预(减温水动作)的响应物理正确——否则预测器无法升级为控制顾问。 |
| **Limitation 1** | 黑箱WM学到的是条件期望E[T\|a]而非干预效应P(T\|do(a))。在闭环运行数据中，动作由温度偏差和扰动共同驱动，模型可从状态历史吸收平均动作效应，不需要理解动作→温度的因果方向。B1glb(标准注意力WM)在减温水阶跃后前60s给出与物理相反的响应方向(dir=0.44-0.50)。 |
| **Limitation 2** | 开环预测精度 ≠ 闭环控制效用(Lambert 2020 objective mismatch)。MAE 0.31°C 的模型在MPC仿真中与PID无显著差异——因为训练数据是运行员自然轨迹，MPC规划动作超出分布。现有评测用MAE选模，对因果结构不敏感：动作响应为零的模型照样拿好MAE(exp_011)。 |
| **Limitation 3** | 现有因果评测用600s末点CFI单标量，奖励"末点对、过程错"。B1glb靠末点gain=1.002拿CFI=0.979，但前60s方向反、shape差(ttp_err=+5)。需要跨时程聚合评测 + 早期方向硬惩罚。 |
| **Key Idea** | 结构分解 T̂ = f_free(x) + g_phys(x,a)，其中 g_phys(x,0)≡0 是架构恒等式(bias=False + GELU(0)=0)，使干预分支输出等于反事实差 ΔT_do(a) = g_phys(x,a)，可直接与DiD观测事件响应对齐，消灭objective mismatch。 |
| **Challenge 1** | 如何保证 g(x,0)≡0 在数据驱动训练中不被绕过？free分支 f_free 可从闭环状态历史吸收动作效应，使 g_phys 梯度饥饿，gain卡在0.65。 |
| **Challenge 2** | 物理分支的结构怎么定？不能拍脑袋——需要从观测数据中辨识传递函数阶数和参数。 |
| **Challenge 3** | 如何评测干预响应的正确性？没有随机干预，只有闭环观测事件。CFI单标量会奖励"末点对、过程错"。 |
| **Methodology topic sentence** | 我们提出一个结构不变量约束的灰箱世界模型，通过零动作恒等式和数据辨识的物理先验，在统一协议下实现预测精度和因果保真度的Pareto最优。 |
| **Module A (→ Challenge 1)** | **结构零动作不变量**：g_phys 使用 bias-free 线性层 + GELU 激活，保证 g(x,0)≡0 是代码恒等式而非正则约束。freeze-free 策略(前10 epoch冻结 f_free) 阻止 free path 提前吸收动作信号。 |
| **Module B (→ Challenge 2)** | **数据辨识灰箱先验**：从DiD观测事件响应拟合 n_lag∈{1,2,3,4,5}，一阶被数据否决(60s误差0.103 vs 真值0.029，无死区)，二阶最优(RMSE=0.022)，≥4阶变差。二阶级联LTI给出可读参数 K(x)、τ(x)，不是假设而是辨识结果。 |
| **Module C (→ Challenge 3)** | **跨时程因果保真度评测**：用ERC-WMAE(事件响应曲线加权MAE)替代600s单点CFI；早期(30-60s)方向硬惩罚；gain按gain_ceiling归一化；validation-only选模；matched observational event-response reference(不称因果ground truth)。 |
| **Contribution 1** | 揭示黑箱WM在观测闭环数据上学到E[T\|a]而非P(T\|do(a))——方向反转是结构性的，不是调参能修的(§4.1) |
| **Contribution 2** | 结构零动作不变量 g(x,0)≡0 是架构恒等式，使干预分支等于反事实差，唯一实现物理正确响应形态的模型(§4.2-4.3) |
| **Contribution 3** | n_lag=2从DiD响应数据中辨识(非假设)，一阶被否决、≥4阶变差，建立"观测→DiD→阶数→架构"的完整闭环(§4.4) |
| **Contribution 4** | 统一评测协议：ERC-WMAE + 早期方向 + validation-only选模 + matched observational reference(§3 + §5) |

---

## 3. 自洽检查

- **Check 1 Limitations → Key Idea**: ✅ 三个limitation(方向反转、精度≠效用、CFI末点)都被Key Idea(结构分解+零动作恒等式+反事实差对齐)直接回应
- **Check 2 Key Idea → Challenges**: ✅ 三个challenge(恒等式不被绕过、物理结构怎么定、怎么评测)都从Key Idea的实现中自然产生
- **Check 3 Challenges → Methodology**: ✅ Module A/B/C 一一对应
- **Check 4 Methodology → Contributions**: ✅ 四个贡献分别映射到§4.1-4.4

---

## 4. 论文结构

### §1 Introduction (1.5页)

- 火电灵活性 → 主汽温控制瓶颈
- WM潜力：学习动态转移，支持前瞻预测
- Gap：精度好≠干预响应正确≠控制有效
- 本文：结构不变量使WM获得物理正确的干预响应
- 贡献概述(4条)

### §2 Problem Formulation (1页)

- 2.1 主汽温预测任务：40维DCS历史 → 末级过热器出口汽温 H=60步
- 2.2 观测数据的因果阶梯：logged action a = π(T_{t-1}, d_t)，WM学到 E[T|a] = E[T|π(T,d)]，不是 P(T|do(a))
- 2.3 Pearl L1 vs L2：条件期望 vs 干预效应；无sequential exchangeability/support时只能称 observational response consistency

### §3 Method (2.5页)

- 3.1 结构分解：T̂ = f_free(x) + g_phys(x,a)
  - f_free: RevIN + PatchEmbedding + PerVariableTCN + VariableAttention 编码器(Phase 1 M5/M7主干)
  - g_phys: 二阶级联LTI，bias-free + GELU → g(x,0)≡0 恒等式
  - 干预分支输出 = 反事实差 ΔT_do(a) = g_phys(x,a)
- 3.2 数据辨识的灰箱先验
  - DiD matched event-response curve → n_lag 拟合
  - n_lag=2最优(RMSE=0.022)，一阶被否决(60s 0.103 vs 0.029)
  - K(x)、τ(x) 由工况状态调制 → 可解释参数
- 3.3 Freeze-free 训练策略
  - 前10 epoch冻结 f_free，强迫 g_phys 先吸收动作信号
  - 对比：无freeze时 gain=0.65(free path抢35%)，freeze后 gain→1.0

### §4 Evaluation Protocol (1.5页)

- 4.1 Matched observational closed-loop event-response reference
  - 事件onset由ΔSP定义；匹配只用处理前变量(负荷/温度水平/趋势/燃料/给水)
  - 报告 overlap, standardized difference, pre-trend, placebo onset
  - **不称因果ground truth**，称 matched observational response consistency
- 4.2 ERC-WMAE (替代CFI单标量)
  - 跨 120/180/300/420/600s 加权MAE
  - 30-60s dir_dsp 硬惩罚(方向反→直接FAIL)
  - gain按gain_ceiling归一化
- 4.3 Validation-only canonical checkpoint
  - test不参与逐epoch选模
  - 每seed一个validation最优checkpoint

### §5 Experiments (3页)

- 5.1 **方向一致性对比** (核心结果，Table 1 + Fig 2)

| Model | dir_dsp@30s | dir_dsp@60s | dir_dsp@180s | shape_corr | ttp_err | gain_span | MAE |
|-------|-------------|-------------|--------------|------------|---------|-----------|-----|
| M5-DSP | FAIL | FAIL | FAIL | 0.998 | — | — | 0.417 |
| M7-DSP | FAIL | FAIL | FAIL | 0.973 | — | — | 0.896 |
| M9-DSP60 | FAIL | FAIL | FAIL | 0.973 | — | — | 0.908 |
| B1glb (TimeXer) | 0.44 | 0.50 | 0.63 | 0.978 | +5 | 0.76 | 0.857 |
| B1flat | 0.50 | 0.50 | 0.63 | 0.959 | +8 | 0.93 | 0.862 |
| A1phys_null | 0.38 | 0.38 | 0.38 | — | — | — | 1.545 |
| **A1phys ff10 (3 seeds)** | **1.00** | **1.00** | **1.00** | **0.97** | **0** | **0.32** | **0.832** |

  - A1phys_null(无物理分支)dir=0.375 → 恒等式不是trivial的，free path确实需要g_phys约束
  - 所有黑箱模型前60s方向反 → E[T|a] ≠ P(T|do(a)) 是结构性的

- 5.2 **n_lag消融** (Fig 3)

| n_lag | CFI_agg | gain_mean | dir@30s | MAE | RMSE vs DiD |
|-------|---------|-----------|---------|-----|-------------|
| 1 | — | — | — | — | 0.045 (60s FAIL) |
| **2** | **0.833** | **1.004** | **1.00** | **0.832** | **0.022** |
| 3 | 0.688 | 0.554 | 0.875 | 0.823 | 0.024 |
| ≥4 | ↓ | ↓ | ↓ | — | 0.031+ |

  - 一阶失败方式很具体：无初始零斜率，不能表达30→60s平台(纯迟延)
  - 二阶级联得S形等效纯迟延，无需显式delay参数

- 5.3 **多seed稳定性** (Table 2)

| Seed | best_cfi | final_cfi | gain@180s | dir_all | best_mae |
|------|----------|-----------|-----------|---------|----------|
| 0 | 0.874 | 0.748 | 0.170 | 1.00 | 0.832 |
| 1 | 0.856 | 0.790 | 0.205 | 1.00 | 0.851 |
| 2 | 0.876 | 0.761 | 0.193 | 1.00 | 0.857 |

  - **dir_dsp=1.0 跨3个seed全时程一致** → 结构性保证，非seed噪声
  - gain有seed变异但方向不变

- 5.4 **物理参数可解释性** (Fig 4，论文最有说服力的图)
  - K(x) vs 机组负荷散点图
  - τ(x) vs 机组负荷散点图
  - 按负荷箱分层与DiD真值逐箱对比
  - `physics_params()` 接口已就绪

- 5.5 **精度对比**
  - A1phys MAE 0.832 vs M5-DSP 0.417 → 精度代价
  - 但 M5-DSP 方向反 → 精度无意义(对控制顾问)
  - A1phys 在"物理正确"约束下的精度上界

### §6 Discussion (1页)

- 6.1 E[T|a] vs P(T|do(a))：观测数据的根本限制
  - 没有安全激励/已知policy/充分状态 → 只能称 observational response consistency
  - 但结构不变量可以在不识别因果效应的情况下保证响应方向
- 6.2 欠增益0.65的诊断
  - free path吸收35%干预效应(梯度饥饿)
  - freeze-free策略部分修复(gain 0.65→1.0)
  - 完全修复方向：(A)干预分支优先训练 (B)增益校准损失 (C)对f_free屏蔽SP通道
- 6.3 通用性
  - 结构不变量 + 灰箱先验 + 因果审计框架不局限于主汽温
  - 任何有logged-action的工业被控对象都可用此框架辨识传递函数并审计WM干预响应
- 6.4 表示增强必须过方向证书（JEPA B 系列，seed0 探索批 7 个 distinct arms / 8 次执行）
  - 动机：多尺度慢态记忆（B2）是否能在不破坏干预响应的前提下提升精度
  - B2（慢态读物理状态=动作的函数）：H18 −5.3% 精度收益，但 valve2 方向证书破坏
    （H18 +0.010 / H60 +0.079°C 升温 vs c0 −0.105°C 降温；frac 0.29）→ REJECT
  - B5（动作盲化慢态，只读 boundary）：valve2 方向回正（−0.010°C）但效应强度
    稀释约 10 倍（frac 0.446 < 0.5），H18 −4.42% 未过 5% 门限，负荷极差 +10.9% → REJECT
  - B2→B5 的符号变化与幅值稀释**符合**慢态注入通道干扰动作响应的假说，但两臂是
    单种子探索比较，不能识别“精度收益与因果破坏同源”，也不能排除优化、容量或支持域差异
  - 结论：表示增强收益必须与方向证书联合验证；本批五个候选均未晋级。动作盲化修复了
    B2 的均值反号但没有通过 B5 的预注册幅值/均匀性/方向门；不得升写为机制级因果冲突
- 6.5 未来工作：多尺度损失与慢尺度假设
  - 现有慢态仍以步级观测 NLL 为损失 → 梯度迫使慢态拟合快动力学（含动作效应）
  - 待检验假设：慢态更新尺度 τ_slow 若大于当前响应审计窗口，可能减少慢态吸收快动作效应；
    现有 seed0 结果不能确定阈值，也不能把 60s 与 240-560s 的比较当作已识别因果尺度
  - 提议设计：慢态专用损失（预测未来一个慢窗的粗粒化状态——负荷趋势/过热度
    趋势/工况标签，损失每慢窗计算一次）+ 快通路步级损失 → 表征按时间尺度分层
  - 若立项须另行预注册成功判据（方向证书、幅值、负荷均匀性与慢态工况相关性），
    不从本批结果事后反推阈值
  - 正交方向：因果可辨识性条件（受控世界模型可辨识性 2607.22430；JEPA-x
    特权物理 2608.24044；LeJEPA 线性可辨识条件）与受限学习（注入通道符号约束）

### §7 Limitations (0.5页)

- 单机组(伊敏6号机)、10s采样、n=16-79事件
- 观测非随机干预，结论为internal validation
- 概率头未独立校准
- 物理分支是gray-box prior非完整物理模型(无守恒方程)

### §8 Conclusion (0.5页)

- 结构零动作不变量使WM获得物理正确的干预响应
- n_lag=2由数据辨识
- 黑箱WM的方向反转是结构性的
- 框架通用，可推广到其他工业控制场景

---

## 5. 图表规划

| 图号 | 内容 | 类型 | 工具 |
|------|------|------|------|
| Fig 1 | Motivated example: B1glb vs A1phys 对同一减温水阶跃的响应曲线，叠加DiD真值 | 双panel时序曲线 | Matplotlib |
| Fig 2 | 方法overview: 结构分解架构图(编码器→f_free + g_phys→T̂) | 流程图 | TikZ/Figma |
| Fig 3 | n_lag消融: DiD真值 + n_lag=1/2/3/4/5 拟合曲线 | 时序曲线面板 | Matplotlib |
| Fig 4 | 物理参数: K(x) 和 τ(x) vs 机组负荷散点图 + DiD分层真值 | 双panel散点 | Matplotlib |
| Fig 5 | 多seed稳定性: 3 seeds 的 gain profile 曲线 + 置信带 | 时序曲线 | Matplotlib |
| Table 1 | 方向一致性主表(7模型 × 7时点 + shape/ttp/gain/MAE) | 表格 | LaTeX |
| Table 2 | 多seed结果表 | 表格 | LaTeX |

---

## 6. Claim-Evidence Ledger

| Claim | 证据 | 等级 | 需要补 |
|-------|------|------|--------|
| 黑箱WM学到E[T\|a]非P(T\|do(a)) | B1glb/M5/M7/M9前60s dir<0.55 | A(3 seeds A1phys) | 多seed B1glb |
| g(x,0)≡0是架构恒等式 | causal_arch.py bias=False+GELU(0)=0; A1phys_null dir=0.375 | A | 代码已验证 |
| A1phys唯一物理正确响应形态 | dir=1.0×3seeds, shape=0.97, ttp=0, gain_span=0.32 | B | 补到5 seeds |
| n_lag=2由DiD数据辨识 | RMSE 1阶=0.045(FAIL) vs 2阶=0.022 vs 4阶=0.031 | A | 已有 |
| freeze-free策略有效 | gain 0.65→1.0, CFI_agg 0.657→0.833 | B | 多seed已验证(3/3一致) |
| 框架通用 | 理论论证 | C(未在其他对象验证) | 标注为future work |

---

## 7. 需要补的实验

| 编号 | 任务 | 目的 | 成本 | 优先级 |
|------|------|------|------|--------|
| E1 | A1phys ff10 补到5 seeds (已有s0/s1/s2) | 多seed统计 | 2次GPU训练 | P0 |
| E2 | B1glb 补到3 seeds | 对照组多seed | 2次GPU训练 | P0 |
| E3 | M5-DSP/M9DSP/DirectWM 同口径事件评测 | 基线行 | CPU重算(已有ckpt) | P0 |
| E4 | validation-only选模重跑 A1phys×3 | 修test选模 | 3次GPU训练 | P1 |
| E5 | n_lag∈{1,2,3} 消融×3 seeds | 阶数实证 | 6次GPU训练(已有2/3) | P1 |
| E6 | 匹配诊断(pre-trend/placebo/overlap) | 修CFE不是ground truth | CPU | P1 |
| E7 | K(x)/τ(x) 工况分布图 | 可解释性主图 | CPU(已有代码) | P2 |
| E8 | 放宽事件筛选到n≥60 | 修n=15 | CPU | P2 |

**总计**：约7次GPU训练(2-3天) + CPU分析(1-2天)
**时间线**：2-3个月含写作
**目标期刊**：Applied Energy / IEEE TEC

---

## 8. 与Phase 4的关系

本文(Phase 3收口)解决的是**通用方法论**问题：
- 任何工业被控对象的WM都可以用结构不变量+灰箱先验+因果审计来检验干预响应正确性
- 传递函数阶数从观测数据辨识，不预设物理方程

Phase 4解决的是**系统级物理建模**问题：
- Fan20代表火电整体运行过程(守恒方程闭合)，不只是一个传递函数
- 这是系统性与通用性的区别
- 两条线互补：Phase 3的方法论可以用来审计Phase 4的Fan20灰箱模型的干预响应

**叙事衔接**：本文§6.3点出"框架通用"→ Phase 4 follow-up："当物理方程可用时，灰箱先验可升级为完整物理模型，结构不变量仍作为干预响应正确性的保证"

---

## 9. 诚实性提醒

1. **不能称"物理模型"**：A1phys是gray-box prior(二阶LTI + 工况调制参数)，无能量/质量守恒。审稿人会抓这个词。用"physics-informed gray-box prior"。
2. **不能称"因果识别"**：观测闭环事件是matched observational response reference，不是因果ground truth。用"observational response consistency"。
3. **不能称"独立测试"**：当前是internal validation(test已被开发访问)。用"internal temporal validation"。如有新数据再升级。
4. **欠增益0.65必须写**：不能只报gain=1.0(freeze后best_causal)，要同时报baseline 0.65和freeze效果，否则审稿人发现cherry-picking。
5. **n=15/n=16必须写**：事件数少，方向一致性的结构性保证比CFI数字更重要。dir_dsp=1.0×3seeds是结构性结论，CFI=0.833是探索性数字。
