# CFE: 因果保真度评测框架 (2026-08-05)

> 动机: 架构实验 (exp_096-102) 之间不可比, 且评测通路存在决定性协议错误。
> 原则: 先建立过硬的评测框架, 再做架构消融。

---

## 0. 触发原因: 训练/评测动作编码不一致 (决定性 bug)

第 40 列 `raw41[:, I_DSP] = np.diff(SP)` 本身已是**一阶差分**。

| 位置 | 代码 | 语义 |
|------|------|------|
| 训练 | `A = train_raw[i+W:i+W+H, I_DSP]` | 一阶差分 ✅ |
| 评测 | `a = np.diff(raw41[s+W-1:s+W+H, I_DSP])` | **二阶差分** ❌ |

一次 `+2` 的 SP 阶跃下:

```
SP      [0, 0, 2, 2, 2, 2, 2, 2]
dsp 列  [0, 0, 2, 0, 0, 0, 0, 0]
训练 a  [2, 0, 0, 0, 0, 0]      隐含 SP 净变 +2
评测 a  [2,-2, 0, 0, 0, 0]      隐含 SP 净变  0   ← 偶极子, 自相消
```

**影响范围 — 9 个脚本** (grep `np.diff(raw41`):

| 脚本 | 行 | 性质 |
|------|-----|------|
| `exp_097_action_probe.py` | 70, 99 | 诊断 (99 行 `p_neg` 同样错) |
| `exp_097_sandbox_eval.py` | 104 | 评测 |
| `exp_097_fig_cases.py` | 86 | **论文图 v1** |
| `exp_097_fig_cases_v2.py` | 69 | **论文图 v2** |
| `exp_097_fig_cases_v3.py` | 75 | **论文图 v3 (主模型 M9DSP H=60, commit b31a9e9)** |
| `exp_098_dsp_dropout.py` | 131 | 评测 |
| `exp_099_phys_calib.py` | 66, 135 | 诊断 |
| `exp_100_m7dsp_h60.py` | 163 | 评测 |
| `exp_101/102_m9dsp_*.py` | 144 | 评测 |

注: 各脚本的**训练段**都是正确的一阶差分, 故**权重本身有效, 无需重训**; 只有推理/评测/出图通路需要修正后重跑。

**受污染的结论** (全部需重测):

| 数字 / 结论 | 出处 | 状态 |
|------|------|------|
| M5-DSP 响应 0.05°C / 方向 75% | exp_097 | 作废 |
| action dropout 无效 (0.060°C, 方向 67%) | exp_098 | 作废 |
| 残差-ΔSP 相关 −0.327 / 欠响应 0.096°C/°C / "B 方案不可行" | exp_099_phys_calib | 作废 |
| M7-DSP 600s 0.212°C / 方向 45% | exp_100 | 作废 |
| M9DSP H=60 180s 方向 89% / 600s 41% | exp_101 | 作废 |
| M9DSP H=18 180s 方向 65% / "末端降权假说" | exp_102 | 作废 |
| "瓶颈 = 观测数据共因混杂非窗口长度" | exp_100 v3 | **推理前提失效** |
| "主模型定案 = M9DSP H=60" | exp_102 commit | **定案依据失效, 需重判** |
| case 图 v1/v2/v3 的 WM 预测曲线 | exp_097_fig_cases* | **需重画** |

MAE 数字 (0.301 / 0.348 / 0.361) 受影响小 (MAE 对动作通道不敏感), 但偏悲观。
方向类指标 (`dir_60` 等基于 `pred(real a)` 的) 也受影响, 因为 `pred` 本身喂了错动作。

`docs/narrative_restructure.md` §IV 的证据链在重测前不得引用。

---

## 1. 现有评测的 5 个结构性缺陷

| # | 缺陷 | 表现 | CFE 对策 |
|---|------|------|---------|
| D1 | 训练/评测动作编码不一致 | 上文 | L0 单一函数 + 往返测试 |
| D2 | 真值是假设而非观测 | 方向正确率对标 `sign(ΔSP)`; 但 exp_099 显示 180s 物理响应比例中位仅 0.17 且跨零 | L1 DiD 匹配对照真值 |
| D3 | 选模型准则与被测目标相反 | best ckpt 由 test MAE 选; exp_011 已证 MAE 对因果盲 | L0 双 ckpt (best-MAE / best-CFI) |
| D4 | 架构对比多因素混杂 | M9DSP vs M7-DSP 同时改 backbone / 注入方式 / 输出头 / token 化 | L4 因子化消融 |
| D5 | n=1 | varattn 文档 §5 自述"多 seed 是唯一准入门槛", 未执行 | L0 seeds≥5 + 配对检验 |

附加混杂: H=60 vs H=18 同时改了 H、时间权重 `linspace(1,0.6,H)` 末端权重、
best-epoch 评判窗口 → "末端降权"假说未被隔离。

---

## 2. 框架分层

```
L0 协议卫生 (门禁)  →  L1 真值构造  →  L2 因果指标  →  L3 混杂控制  →  L4 架构消融  →  L5 H/权重解耦
```

### L0 — 协议卫生 (不过门不许报任何指标)

| 项 | 要求 | 落地 |
|---|------|------|
| P0.1 | 动作构造**唯一函数**, 训练与评测共用 | `causal_eval.build_action()` |
| P0.2 | 往返一致性: 训练集样本走评测通路, MAE 与训练 test MAE 一致 (tol 1e-3) | `exp_103` 自检段 |
| P0.3 | 反事实基线定义写死: `a=0` ≡ "SP 保持", on-manifold (数据中 SP 大部分时间确实保持) | 文档声明 + assert |
| P0.4 | seeds ≥ 5, 所有指标报 mean ± 95% CI, 模型间配对检验 | L4 起强制 |
| P0.5 | 双 ckpt: `best-MAE` 与 `best-CFI` 各存一份, 两个都报 | 训练脚本改造 |
| P0.6 | 指标必须附**噪声上限** (ceiling), 否则 "89%" 不可解释 | L1 split-half ceiling |

### L1 — 真值构造 (DiD 匹配对照事件研究)

用**差分中差分**估计观测因果响应, 替代 `sign(ΔSP)`。

- **处理组**: `|ΔSP| > 1.0`, 事件间隔 ≥ 60 步, 负荷稳定 (`max|Δload| ≤ 3` in ±20), SP 保持 (`max|SP[o:o+61]−SP[o]| ≤ 0.3`) — n≈134
- **对照组**: 粗化精确匹配 (CEM) 的平稳段 — 同负荷分箱 × 同 onset 前温度趋势分箱, 且窗口内无 SP 阶跃
- **逐事件归一化响应**:

  ```
  r_i(k) = [ ΔT_treat,i(k) − mean_j ΔT_ctrl,ij(k) ] / ΔSP_i
  ΔT(k) = T(k) − T(−1)
  ```

- **聚合**: `R_true(k) = mean_i r_i(k)`, bootstrap 95% CI; 按 `|ΔSP|` 分箱与负荷分箱分层
- **噪声上限**: 对照组 split-half, 用半数对照预测另半数 → `SGN_ceiling(k)`, `GAIN_ceiling(k)`
- **产出**: `results/cfe_groundtruth/did_response.json` — 一次做好, 所有模型共用

> 这一步本身是独立贡献: 主汽温对 ΔSP 干预的观测因果响应曲线及其时标。

### L2 — 因果保真度指标 (对标 L1 真值)

模型响应: `m_i(k) = [pred(real a) − pred(a=0)](k) / ΔSP_i`

| 指标 | 定义 | 说明 |
|------|------|------|
| `SGN_pair(k)` | `mean_i 1[sign(m_i)=sign(r_i)]` | 逐事件配对, 需对比 ceiling |
| `SGN_agg(k)` | `mean_i 1[sign(m_i)=sign(R_true)]` | 对齐旧口径, 便于回溯 |
| `GAIN(k)` | `mean(m_i(k)) / R_true(k)` | **报比值不报 °C**, 消除口径歧义; 目标 1.0 |
| `SHAPE` | `corr_k(m̄(k), R_true(k))` | 响应形状 |
| `TTP` | 达 50% 稳态响应的时刻 − 真值时刻 | 时标 |
| `MONO` | `m̄(k_end)/m̄(k_mid)` ÷ 真值同比 | varattn 观察 C 的正式化 |
| `LIN` | `m/δ` 在 δ∈{±0.5,±1,±2,±4}×ΔSP 的 CV | 增益线性度 |
| `SYM` | `|m(+δ)| / |m(−δ)|` | 偏离 1 = 共因泄漏征兆 |
| `HET` | 增益在负荷分箱间 CV, 与真值 CV 之比 | 直接检验 varattn H1 (增益自适应) |

复合分 `CFI = 0.35·SGN_norm + 0.30·min(GAIN,1/GAIN) + 0.20·SHAPE + 0.15·(1−|TTP|/k_max)`
(`SGN_norm = (SGN−0.5)/(ceiling−0.5)`, 即相对噪声上限的归一化) — 用于 P0.5 ckpt 选择。

### L3 — 混杂控制 (判决"这是因果还是相关")

| 测试 | 做法 | 判读 |
|------|------|------|
| `LEAK` | 屏蔽/打乱 x_hist 中 SP 相关列 (idx 36 及派生), 重测动作增益 | exp_099 已发现状态路径隐含 0.53°C/°C。若 `LEAK` 大 → "动作通道增益小"可能是**信息冗余下的合理配置**而非缺陷。**此项决定 §IV 结论走向** |
| `PLACEBO` | 在无真实阶跃的平稳段注入 ΔSP | 真因果模型仍应产生响应; 捷径模型 ≈0 |
| `DO_vs_SEE` | 观测臂: 真状态+真动作。干预代理臂: **匹配平稳段的状态** + 真动作 | 两臂之差 = 混杂量级, 量化 Pearl L1/L2 gap |
| `AUDIT` | 断言 x_hist 窗口不越过 onset; 断言动作序列与训练同构 | 固化为 assert |

### L4 — 架构消融 (只在 L0–L3 就位后做)

因子化: **只变动作注入路径**, 其余全固定 (同 backbone / H / 时间权重 / β / epochs / 数据 / 5 seeds / 同 ckpt 准则)。

| 变体 | 动作路径 | 隔离的因素 |
|------|---------|-----------|
| A0 | 无动作 | 地板值 — 所有增益须超过它才有意义 |
| A1 | flatten→concat→decoder | 现 M5-DSP / M7-DSP |
| A2 | 动作作为第 41 个 token 进 VarAttn | varattn §E5 的 M4d — **在 DirectWM 内隔离"动作参与注意力"** |
| A3 | GLB↔动作独立 cross-attn | 现 M9DSP |
| A4 | A3 但动作与外生状态**共享 softmax** | **直接检验 TimeXerLayer docstring 的核心声称**: "动作不应被 39 状态在 softmax 中稀释" |
| A5 | FiLM / 门控: 动作调制 decoder | 乘性 vs 加性注入 |
| A6 | 逐步注入 (GRU decoder, `src/world_model.py` 已有) | 时序对齐 vs 整段注入 |

关键对比:

- **A2 vs A1** → 隔离"动作进注意力"单因素 (不换 backbone), 这才是 H3 的干净检验
- **A3 vs A4** → 隔离"独立 softmax", M9DSP 唯一原创设计声称, **目前完全没测**
- **A3 vs A2** → 隔离 TimeXer backbone 的贡献

现状 A1 vs A3 一步跨 4 个因素, 任何归因都站不住。

### L5 — H / 时间权重解耦

`H ∈ {18, 60} × weight ∈ {flat, linspace(1→0.6)}` 2×2, 检验 exp_102 commit 自提的"末端降权"假说。
附加变体: 权重按 L1 的 `R_true(k)` 物理响应比例加权 (有物理依据的设计)。

---

## 3. 执行顺序与判决树

```
P0.1 共享模块 + P0.2 往返一致性        半天   [exp_103 自检段]
              ↓
L1 DiD 真值 + ceiling                  1 天   [exp_104]  ← 独立贡献, 一次性
              ↓
用修正协议重测已有 4 个 ckpt            2 小时 [exp_103]  ← 不需重训
              ↓
   ┌── 结论翻转 (动作增益其实不弱) ──> §IV 回到"因果可学", 直接进 L4
   └── 未翻转 ──> L3 LEAK / DO_vs_SEE 定位瓶颈 ──> 再决定 L4 是否值得
              ↓
L4 因子化消融 A0-A6 × 5 seeds          2-3 天 GPU
              ↓
L5 H/权重解耦 2×2 × 5 seeds            1-2 天 GPU
```

**关键判断**: 不要现在跑 L4。修正协议后重测已有 checkpoint 是最高性价比的一步 —
它可能直接推翻"动作增益 FAIL"主线结论, 而那会改变 §IV 的写法, 进而改变 L4 该测什么。

---

## 4. 交付物清单

| 文件 | 内容 | 状态 |
|------|------|------|
| `experiments/phase3_feedforward/causal_eval.py` | 共享模块: `build_action` / 事件筛选 / CEM 匹配 / DiD / L2 指标 / L3 探针 | 已建 |
| `experiments/phase3_feedforward/exp_103_protocol_recheck.py` | P0.2 往返测试 + 修正编码重测 4 个 ckpt (无需重训) | 已建 |
| `experiments/phase3_feedforward/causal_arch.py` | §6 架构: `ResidualCausalWM` (A1/A2/A3/C1) + `TimeXerCausalWM` (B1) + `g(x,0)=0` 不变量自检 | 已建 |
| `experiments/phase3_feedforward/exp_106_causal_arch.py` | §6 训练/消融, 7 变体 × 多 seed, 双 ckpt | 已建 |
| `exp_104_did_groundtruth.py` | L1 DiD 真值 + ceiling, 落 `results/cfe_groundtruth/` | 待建 |
| `exp_105_leakage_probe.py` | L3 LEAK / PLACEBO / DO_vs_SEE | 待建 |

---

## 4.1 逐行修复清单 (Linux 侧直接执行)

统一替换: 所有 `a = np.diff(raw41[s+W-1 : s+W+H, I_DSP])` →

```python
import causal_eval as CE
a = CE.build_action(raw41, s, W, H, I_DSP)          # 真实 ΔSP
a = CE.build_action(raw41, s, W, H, I_DSP, 0.0)     # SP 保持基线
```

| 文件 | 行 | 备注 |
|------|-----|------|
| `exp_097_action_probe.py` | 70, 99 | 99 行 `p_neg` 应为 `-np.abs(CE.build_action(...))` |
| `exp_097_sandbox_eval.py` | 104 | |
| `exp_097_fig_cases.py` | 86 | 修完重跑出图 |
| `exp_097_fig_cases_v2.py` | 69 | 修完重跑出图 |
| `exp_097_fig_cases_v3.py` | 75 | 修完重跑出图 (主模型图, 优先级最高) |
| `exp_098_dsp_dropout.py` | 131 | |
| `exp_099_phys_calib.py` | 66, 135 | |
| `exp_100_m7dsp_h60.py` | 163 | 训练段无需改 |
| `exp_101_m9dsp_h60.py` | 144 | 训练段无需改 |
| `exp_102_m9dsp_h18.py` | 144 | 训练段无需改 |

同时在各训练脚本的数据取样处也改为调用 `CE.build_action`, 消除"两份代码"的复发可能
(训练段当前语义正确, 但仍是独立实现)。

`exp_103_protocol_recheck.py` 会先跑 `CE.assert_train_eval_identity` 做门禁, 不过不许出数。

---

## 5. 需要修订的既有文档 (与重测结果无关)

1. `docs/narrative_restructure.md` §IV — 全部动作增益数字加"协议待修正"标注, 重测后重写
2. `docs/varattn_causality_analysis.md` §3.2 — 敏感性表补 n=1 说明; 该表用阀位动作 (非 ΔSP), 不受本 bug 影响, 但仍是 n=1
3. `experiment_audit.md` — 新增条目: 训练/评测动作编码不一致 (跨 9 个脚本)
4. `docs/phase3_sandbox_design.md` — 沙盒精度/消融判定 (MAE 0.301, 消融 Δ≈0) 的评测通路同样受影响, 需标注
5. commit `dcd4d2c` 的"主模型定案 = M9DSP H=60"与 `b31a9e9` 的 case 图 v3 — 需在重测后确认或撤回

---

## 6. 强动作因果架构候选

实现: `experiments/phase3_feedforward/causal_arch.py`, 训练脚本 `exp_106_causal_arch.py`

### 6.1 为何现有架构必然弱因果

| 缺陷 | 位置 | 机制 |
|------|------|------|
| **1 动作旁路** | `exp_025.TimeXerLayer.forward` | `act_attn`/`exog_attn` 只更新 `glb` 1 个 token, 而 head 是 `Linear((np+1)*d, H*2)` = 12 token 展平 → 动作仅占 head 输入 **1/12**, 另 11 个 patch token 是主汽温自身历史的恒等残差通路。最省力解 = 内生外推 + 压低 GLB 权重 |
| **2 加性注入** | `DirectWM` `cat([s_repr,a_feat])`; TimeXer 残差相加 | 只能表达 `f(state)+g(action)`。但 exp_099 观测响应比例中位 0.17 且分布宽 = `K(state)×action` 乘性增益 → **加性结构数学上无法表示**, 加宽加深无效 |
| **3 无时间约束** | `head: Linear(..., H*2)` | 并行吐 H 步, 无单调性/时间常数约束 → 响应剖面可任意非单调 (exp_101 远步衰减) |
| **4 动作冗余** | x_hist 含绝对 SP (idx 36) | SP 可由负荷/温度趋势预测 → 动作通道增量信息低 → 梯度饥饿。**任何带旁路的架构都会饿死动作通道** |

> 注: M9b 把 head 从 GLB-only 改成 FlattenHead ("消除 GLB 信息瓶颈") —— **而那个瓶颈恰恰是因果的强制通路**。v2 提精度砍因果, 与 Phase 1 exp_011 的 objective mismatch 主题同构。

### 6.2 候选架构

| ID | 架构 | 对付 | 实现 |
|----|------|------|------|
| **A1** | 残差分解 `T̂ = f_free(x) + g(x,a)`, **架构性保证 `g(x,0) ≡ 0`** | 1, 4 | `ResidualCausalWM` |
| **A2** | 乘性门控 `g = W_out(φ(x) ⊙ ψ(a))` | 2 | `InterventionMLP` |
| **A3** | 一阶惯性级联, 网络只出 `K(x), τ(x)` | 2, 3 | `InterventionPhysics` |
| **C1** | 增量累积输出 (归一化空间锚定 `T[o-1]`) | 3 | `cumsum_out=True` |
| **B1** | head 退回 GLB-only, 强制动作通路 | 1 | `TimeXerCausalWM(head_mode='glb')` |

**`g(x,0) ≡ 0` 的精确实现** (恒等式, 非正则):
动作分支全程 `bias=False` + `GELU(0)=0` → `ψ(0)=0`; 输出投影 `bias=False` → `g_out(φ ⊙ 0) = 0`。
物理分支对 `u = cumsum(a)` 严格线性 → `a=0 ⇒ u=0 ⇒ r≡0`。

由此 **`pred(x,a) − pred(x,0)` 恒等于一个专用子网络的输出**:
- 有独立梯度, free path 吸收不了 (直击缺陷 4)
- **可直接用 L1 的 DiD 真值监督** → 评测真值变训练信号, 消灭 D3 的 objective mismatch
- 可加物理约束 (符号/单调/增益上下界)
- `intervention_effect()` 可直接读出因果响应, 无需做差

### 6.3 关键实现细节 (已验证)

| 项 | 问题 | 处理 |
|----|------|------|
| C1 梯度放大 | cumsum 使梯度范数 28000 (非 cumsum 的 200×), 被 clip=1.0 压死 | free_head 末层零初始化 → 初始预测 = `T[o-1]` 持久化 (强基线), 梯度降至 500 |
| C1 二次积分 | 物理分支输出的**已是响应量级** `r(k)`, 再 cumsum 会二次积分 | cumsum **只作用于 free 分支** |
| A3 τ 初值 | `sigmoid(0)=0.5 → τ=201` 步 (2010s), 600s 仅到 4% → `K` 梯度饿死 | 按 exp_099 的 600s≈97% 反推 `τ_init=18` 步 |

A3 初始化后的响应剖面 (`K=1.0, τ=[18,18]`, 未训练):

| | 60s | 180s | 300s | 600s |
|---|---|---|---|---|
| A3 初值 | 0.052 | 0.278 | 0.507 | 0.839 |
| 物理 (exp_099) | — | 0.17 | — | 0.97 |

**模型开局就在物理附近且严格单调**, 只需学增益异质性 —— 这是 A3 相对 A1mlp (开局 ≈0) 的先验优势。

### 6.4 变体表 (`exp_106 --variant`)

`A1mlp` / `A1phys` / `A1both` / `A1mlp_cs` / `A1physcs` / `B1glb` / `B1flat`

协议全变体固定 (W=96, BS=256, STEPS=500/ep, AdamW 1e-3/1e-5, β=0, ReduceLROnPlateau),
动作一律走 `CE.build_action`, 事件只取 test 区间, 双 ckpt (`best_mae.pth` / `best_causal.pth`)。

关键对比: **`B1glb` vs `B1flat`** 只差 head → "精度 vs 因果"权衡的直接证据 (论文缺的那张图);
**`A1phys` vs `A1mlp`** 隔离物理先验的价值; **`*_cs` vs 无后缀** 隔离 C1。

### 6.5 已修的另一处审查问题

`exp_100/101/102` 的 134 个事件在**全量数据**上筛选, 含训练区间 → 因果指标有训练集泄漏。
`CE.select_events` 新增 `lo/hi/W` 区间过滤, `exp_106` 传 `lo=n_val_end, W=W`。旧结果重测时须一并修正。
