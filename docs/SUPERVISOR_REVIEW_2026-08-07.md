# Supervisor Review — 代码、论文与方法论总审

> 日期：2026-08-07
> 范围：`src/`、`experiments/phase1_dynamics/`、`experiments/phase2_mpc/`、`experiments/phase3_feedforward/`、`tests/`、已入库 JSON/NPZ、Fan 2017/2020/2021 原文及项目精读、当前论文叙事文档。
> 判决：**Major revision / not ready for model lock / not ready for manuscript claims**。

## 1. Executive decision

项目已经积累了有价值的预测基线、失败案例和协议审计，但当前没有任何模型达到“主模型定性”的证据标准。Phase 4 应被定义为**研究对象、可识别性和判决协议的重置阶段**，而不是下一轮模型赛马。

Supervisor 建议的主轴是：

1. 以 Fan20 的两级喷水和 SST 焓值链作为主汽温 plant skeleton；
2. Fan20 已含制粉迟延/惯性；只把 Fan17 的显式金属蓄热和 Fan21 的能量不匹配/负荷相关参数作为嵌套候选，并防止热量双计；
3. 在同一物理内容、同一状态和输入上比较 explicit ODE、controlled Koopman 和 time-varying gray-box；
4. 把 supervisory setpoint response 与 plant-level spray response 分层建模；
5. 在 validation 选模、一次批量 locked-final（或明确降级的 internal-final）、时间 episode 统计和物理闭合全部就位后，再讨论主模型。

当前最强的潜在论文贡献候选不是某个 Transformer 或 Koopman head，而是：

> 一个把监督层设定值响应与锅炉热力学 logged-action dynamics 分层建模的主汽温灰箱世界模型，并在统一协议下检验物理结构对宽负荷泛化、观测事件响应和物理闭合的价值。

## 2. 审查口径

本审查将证据分成四层：

| 层级 | 含义 | 使用规则 |
|---|---|---|
| Source fact | 原论文、代码或结果文件可直接核对 | 可陈述事实，不自动外推 |
| Project evidence | 在本仓库协议下复现或保存的结果 | 必须注明 split、checkpoint、seed、estimand |
| Inference | 根据事实提出的解释 | 明确标注为假设，需实验检验 |
| Unverified claim | 缺少数据、协议或产物 | 不进入论文摘要、结论或模型定性 |

训练 seed 不是现场独立重复；重叠滑窗不是独立样本；观测闭环事件不是随机干预；代码结构恒等式不是因果识别。

## 3. 可保留的资产

- 项目保留了大量负结果和后续审计，没有强行删除历史，这对科研可追溯性有价值。
- `causal_eval.build_action()` 已集中一阶 ΔSP 的训练/评测语义，修复了早期二阶差分错误。
- `select_events()` 支持时间区间和历史窗口边界，比旧脚本更接近可审计协议。
- A1phys 的级联递推具有严格时间顺序；其 `g(x,0)=0` 在动作分支上是有效的代码不变量。
- Phase 2 审计主动撤回了同构神经 plant 上的 MPC-vs-PID 过强结论。
- exp_112 至少执行了三个训练 seed，能作为当前 diagonal Koopman free-head 的负面 pilot。
- Fan 三篇原文 PDF/全文已保存在仓库，且 Fan20 与主汽温问题的对应关系判断是正确的。
- 本地设计/实现/审计与 Linux 远端正式运行的职责已经分开。

这些资产足以支持 Phase 4 开题，但不足以支持模型终局或论文定稿。

## 4. P0 scientific validity findings

### 4.1 test 被用于逐 epoch 选模

[`exp_106_causal_arch.py`](../experiments/phase3_feedforward/exp_106_causal_arch.py) 只构造 `train_raw` 与 `test_raw`，`eval_mae()` 和 `eval_causal()` 每个 epoch 都访问 test，并用它们保存 `best_mae`、`best_causal` 和触发 early stopping。exp_112 直接复用这一训练函数。

因此：

- exp_106/112 的 best checkpoint 是 selection-on-test；
- 相同的 200 个随机 test 窗口被反复访问，存在自适应过拟合；
- exp_107/110 再在同一时间段报告结果，不是独立测试；
- 最后 15% 已在大量历史实验中反复查看，也不再是严格意义的 lockbox。

**判决**：所有由该流程产生的“最佳”数字降为探索性。Phase 4 优先使用新未来时间段/另一机组作为 lockbox；若无法获得，只能做 nested blocked temporal validation，并限制外推表述。

### 4.2 exp_112 的 CFI 不是所宣称的 P2 CFE

[`exp_112_koopman_full.py`](../experiments/phase3_feedforward/exp_112_koopman_full.py) 读取 `results/cfe_groundtruth_p2/did_response.json`，但仓库只保存了 [`did_response.npz`](../results/cfe_groundtruth_p2/did_response.npz)。即使远端曾存在 JSON，训练期默认 test-only 事件约为 16 个，而 P2 是放宽阈值、val+test 合并的 79 个事件；`eval_causal()` 只有在数组长度完全相同时才进入 CFE，否则静默 fallback。

结果文件提供了直接证据：exp_112 的 `final_causal` 没有 `cfe` 字段，9 个 run 的 CFI 都与“末点 gain 启发式 + sign(ΔSP)”一致。

此外：

- exp_106 成功进入 DiD 分支时仍调用已标为 deprecated 的单点 `cfi()`；
- `cfi_agg()` 的四项权重和为 0.85，而 fallback CFI 满分为 1.0；
- 两者同名不同量纲，不能横向比较；
- exp_112 表中 best MAE 与 best CFI 来自不同 epoch，`final_causal` 又来自最后 epoch，单行混合了三个模型状态。

**判决**：README 中 0.869/0.821 的“最佳 CFI”撤销证据资格。三个 seed 上 Koopman−MLP 的 test-selected MAE 均为正，可保留为该具体 free-head 的负面 pilot，但不能关闭 controlled Koopman。

### 4.3 现有 action 混合了两个系统层级

- exp_025/DirectWM 主要使用未来绝对减温阀位。
- exp_106/A1phys 使用 `二级减温调节阀设定` 的一阶差分 ΔSP。
- Fan20 的控制输入是 `Dsw1/Dsw2` 喷水质量流量。

它们对应不同 estimand：

```mermaid
flowchart LR
  U["supervisory setpoint / ΔSP"] --> C["existing controller"]
  C --> V["valve command and position"]
  V --> Q["spray-water mass flow"]
  Q --> P["Fan20 thermodynamic plant"]
  P --> T["T3 / Tst"]
```

`ΔSP → Tst` 是 controller-mediated closed-loop response；`Dsw → Tst` 是 plant-level thermodynamic response。A1phys 当前学习的是前者的低阶近似，Fan20 描述的是后者。二者不能直接放在同一模型排行榜。

**判决**：Phase 4 必须先核实 DCS tag，分开 supervisory model 与 plant model，或显式串联 controller/actuator layer 与 Fan20 plant layer。

### 4.4 CFE 不是因果 ground truth

[`causal_eval.py`](../experiments/phase3_feedforward/causal_eval.py) 的 CEM 只匹配负荷箱和短期温度趋势；无精确匹配时退化为只匹配负荷。当前没有保存或检验：

- SP/动作基线、实际阀位、煤量、给水、压力、流量、其他操作和控制模式的平衡；
- overlap/common support；
- pre-trend、placebo、negative control 或未观测混杂敏感性；
- treatment/control ID、控制复用和时间窗口重叠；
- 基于运行 episode 的 cluster uncertainty。

P2 的 79 个事件由 28 个 validation 事件与 51 个 test 事件合并得到，正/负动作约为 53/26，最小间隔 68 步；逐事件 600 s 响应标准差约 1.82，噪声很大。普通事件 bootstrap 将自相关和重复对照当作独立样本，置信区间可能偏乐观。

所谓 `split-half ceiling` 仍共享同一处理轨迹，不是独立统计上限。30–60 s 真实响应接近零时，`sgn_norm` 的分母很小，早期符号硬惩罚会被少数事件主导。

**判决**：改称 **matched observational closed-loop event-response reference**。`g(x,0)=0` 只保证动作分支零输入为零；`f_free` 仍可从闭环状态历史中吸收平均动作效应，因此不提供因果识别。

### 4.5 现有 PID baseline 方向和实现错误

[`eval_protocol.py`](../experiments/phase2_mpc/eval_protocol.py) 定义 `e = SP - PV`，却用正 `kp * e` 增大减温阀。温度低于 SP 时，这会继续开阀降温；零偏差时绝对输出趋近 0，而不是维持工作点。`e_prev` 在计算最终输出前已更新，使 `kd` 项恒为 0。相应单测的自然语言说明与断言方向相反。

**判决**：历史 PID-vs-MPC 结果继续保持作废/探索定位。任何新控制实验必须使用带工作点的增量式 PI/PID，并在解析假世界中验证冷热两侧方向。

## 5. P1 code and engineering findings

### 5.1 exp_020 没有验证三种 Phase 4 表示/closure 路线

[`exp_020_koopman_vs_gru.py`](../experiments/phase1_dynamics/exp_020_koopman_vs_gru.py) 是早期 decoder prototype：

- 训练与 validation rollout 只有 5 步，H=18 只用于测速，没有 test H18 精度；
- Neural ODE 是固定 Euler 小步的纯神经 latent dynamics，不是 Fan-structured ODE，也没有 solver 收敛审计；
- Koopman 计算虚部后丢弃，实际使用实对角矩阵；控制映射 `B(a)` 是非线性网络；
- RevIN 编码后没有把绝对均值/方差交给 decoder，物理工况恢复不闭合。

**判决**：仅保留为早期失败原型，不能判决 Neural ODE、controlled Koopman 或 Fan gray-box。

### 5.2 direct multi-horizon 存在未来动作影响过去输出

`InterventionMLP`、`A1both` 和 TimeXer action token 都一次读取完整未来动作序列并一次输出完整 horizon。代码没有下三角 mask，因而 `u_j` 可以影响 `ŷ_k (j > k)`。只读 Jacobian 探针确认 A1mlp/A1both/B1glb 的晚期动作会改变早期输出；A1phys 的逐步递推没有该问题。

**判决**：Phase 4 使用逐步积分或严格 causal mask，并将 `∂ŷ_k/∂u_j = 0, j > k` 固化为测试。

### 5.3 exp_025 的时间权重与 checkpoint 目标不成立

`BetaNLLLoss`/`MSELoss_` 已在 loss 内把 batch 和 horizon 降成标量，训练循环再乘 horizon weight 只改变整体尺度，没有实现远近步差异权重。checkpoint 只按第 5 个预测点的 validation MAE 选择，却报告 H18 平均结果。

**判决**：不能把现有时间权重消融当机制证据；Phase 4 loss 必须保留逐步维度到加权之后，并统一 checkpoint 与 primary horizon metric。

### 5.4 概率与 RevIN 口径有错误

- 均值反归一化除以 RevIN affine weight，但 σ 没有除以 `|weight|`，会造成尺度错误。
- exp_025 用 `mean(|error|/σ)` 并称理想值为 1；标准高斯下该期望约为 `sqrt(2/pi)≈0.798`。
- 历史 β-NLL 使用 `β=-0.3`，其概率解释和 proper scoring 性质没有被验证。

**判决**：概率路线晚于确定性动力学定性；统一用 held-out NLL/CRPS、PIT 或 coverage，并修复物理尺度。

### 5.5 A1phys 的“物理参数”尚不可解释为物理量

- `K` 位于归一化温度空间，实际物理响应还会乘当前窗口目标标准差并除 RevIN weight；原始 `K` 不能直接和 °C/动作单位比较。
- `K` 符号不受约束，tau 虽有范围但依赖高容量 40 变量编码器。
- 结构没有质量守恒、能量守恒、焓值传递或喷淋混合。
- `A1phys_null` 在物理空间并非单纯 `T=g(x,a)`；零归一化输出经 RevIN 反变换后仍含窗口位置/尺度基线。
- freeze-free 没有冻结 patch 和 RevIN affine，free/intervention 的共同表示仍在变化。
- gain supervision 先在 batch/horizon 上压成标量，batch 又以零动作窗口为主，剂量符号/分母与目标长度还可能不匹配；exp_112 的 P2 文件缺失使该项没有按宣称工作。

**判决**：A1phys 是两级惯性 intervention prior，不是 Fan 物理模型；保留为 baseline。

### 5.6 数据与可复现性地基不足

- `np.nan_to_num(..., nan=0)` 将物理传感器缺失变成非物理零值。
- 日期被丢弃，没有审计 10 s 连续性、停机、启机、断点或冻结传感器。
- 多个脚本硬编码 `495407/601566` split 边界和列索引。
- 148 个 Python 文件中 88 个没有 `__main__` guard；大量实验在 import 时读整份 CSV、训练、画图或写结果。
- 多处通过修改 `sys.path`、`sys.argv` 和全局 `H_OUT` 复用脚本。
- `src/config.py` 是 11 状态，loader/注释和 `WorldModel` 默认值仍写 12/14/16 等旧维度。
- 仓库没有 dependency lock、CI、data schema/hash；Windows 数据入口只是指向 `/home/bluster/Desktop/AI` 的 24 字节符号链接文本。
- checkpoint 全部被忽略，结果 JSON 通常缺 commit、命令、环境、数据指纹和预测明细。
- 多个结果目录按固定实验名写入，可被同名重跑覆盖；没有 config hash、checkpoint SHA256 或 run-completeness marker。
- `pytest -q` 当前在收集阶段因 `TimeXerWM` stub 缺失而失败，且没有 CFE、A1phys、Fan、ODE 或 Koopman 测试。

**判决**：在模型定性前不移动历史脚本；为 Phase 4 新建小而独立、import-safe 的包与不可覆盖运行协议。

## 6. Fan papers and project interpretation

原始文献：

- Fan et al. 2017, Applied Energy, DOI [10.1016/j.apenergy.2016.11.074](https://doi.org/10.1016/j.apenergy.2016.11.074)
- Fan et al. 2020, Applied Thermal Engineering, DOI [10.1016/j.applthermaleng.2020.114912](https://doi.org/10.1016/j.applthermaleng.2020.114912)
- Fan et al. 2021, Energy, DOI [10.1016/j.energy.2021.120425](https://doi.org/10.1016/j.energy.2021.120425)

### 6.1 Fan17

原文是 dry-operation 下的 4-state、3-input、3-output boiler-turbine model，输入为 `uB/Dfw/ut`。它不使用本项目的 ΔSP，但并非“没有动作输入”；也不直接以 SST 为输出。对本项目的价值是制粉滞后、锅炉核心状态和金属蓄热组件，而不是完整主汽温模型。

Fan20 自身已经包含 `uB→rB` 的制粉迟延/惯性，因此 Fan17 不能再提供一个重复 mill component；真正可测试的增量是显式金属温度 `Tj`。加入 `Tj` 需要重推 Fan20 分段热量并保证关闭组件时严格恢复 core。项目精读中蒸汽流量函数也曾漏掉 `pst^-0.743` 与乘法括号，现已按原文纠正；实现仍需独立表点/第二实现核验。

### 6.2 Fan20

原文是 7-state、5-input、5-output 模型，显式包含两级喷水、分段焓值和 `Tst=f(pst,hst)`，是三篇中唯一直接覆盖 SST 的主骨架。

需要避免三种误读：

1. Fan20 不是在 Fan17 的四状态上简单增加三个状态；它改变了状态选择。
2. 原文 1 s 数据中的数秒响应不能直接移植为伊敏 10 s 数据上的可辨识结论。
3. 同时预测 `pst` 并不自动形成物性闭合，仍需 `hst` 和正确压力/焓值关系。

### 6.3 Fan21

原文是 4-state CCS model，不含 SST 和喷水。主要价值是宽负荷参数化和整炉 energy mismatch；throttle-loss 进入 `Ne` 动力学，只在保留 `ut/Ne/turbine` coupling 时相关，不能默认嵌入 `Dst` 外生的 SST 子系统。

项目精读中的能量项公式写错。原文为：

```text
Q1 = k1*rB + m/(Ne + g) * (Dfw - a*rB)
```

而不是 `k1*rB + Ne + γ(Dfw-αrB)`。原文也没有证明 `k1/λ` 的变化由设备老化、结焦或积灰导致；那只能作为项目推断。

项目精读还曾把 Case A baseline 与 Case B changed-parameter MARE 拼成同一改善链，并把 Fan20 的 `398/905/632/26 s` 数字误归到 Fan21；两处均已撤回。跨 case、跨论文数字不能作为组件效应。

**正确定位**：Fan21 为 Fan20 提供负荷相关 closure 候选，而不是独立 SST competitor。其整炉 `Q1` 不能直接叠加到 Fan20 已有的 `k11/k12/k13·rB`；实现必须冻结替换/分配规则并证明总能量不重复。

### 6.4 伊敏变量映射仍未闭合

- `uB`（煤流指令）与 `rB`（经过制粉延迟后的炉膛煤流状态）不能都等同于未校正总煤量。
- `Dsw1/2` 是 kg/s 喷水流量，阀位/指令没有阀特性、压差和水焓时不能直接进入质量/能量守恒。
- “主给水流量”的口径是否等于 Fan 总给水、是否包含喷水尚未核对。
- 以主汽压近似省煤器入口给水压力不能作为最终守恒口径。
- Fan `T3` 的测点是否等于“一级减温器出口温度”需用 P&ID/tag 说明确认。
- 用负荷变化率代理缺失 `ut` 可能泄漏结果变量。对 SST 子系统更合理的方案是把实测 `Dst` 作为外生扰动。
- `Tj` 不能预先判为低影响；它可作为 latent metal state，但会影响可识别性。
- Fan17/21 的 dry-operation 假设需和伊敏运行模式对齐。

因此现有“状态覆盖率/模型可用”表只能称候选映射，不能作为实现许可。

### 6.5 observational estimand 边界

Task P 的喷水/阀位来自闭环 controller 对温度与扰动的响应。仅有 action manifest、matching 和 profile likelihood 不能识别 `E[Y(u)-Y(u')|H_t]`。没有安全激励、已知 policy、充分状态、sequential exchangeability 与 support 时，论文只能称 **logged-action conditional dynamics** 与 observational event-response consistency。

同理，未来实测 `Dst/Ne` 可能是 post-treatment/result-related。用它们评测只得到条件场景预测；要讨论喷水总响应，必须在模型中内生传播，或使用在结果前独立冻结的外生场景/预测器。Task S 应输入完整 SP trajectory，`ΔSP` 只定义 event exposure。

## 7. Paper audit

### 7.1 当前没有可审查的 manuscript

仓库中有研究报告、叙事草案、原文 PDF 和精读笔记，但没有完整 manuscript、BibTeX/参考文献数据库、Methods/Results 对应稿或 claim-evidence matrix。因此“论文审查”目前实际是对**论文候选叙事和证据链**的审查。

### 7.2 可保留的论文级认识

- 开环预测、干预响应和控制效用是不同层级，不能互相替代。
- 当前同构神经 plant 的 MPC-vs-PID 不能外推到现场。
- `g(x,0)=0` 是可验证的架构不变量，但应称结构隔离而非因果证明。
- Fan20 是主汽温物理建模的合理起点；Fan17/21 是机制来源。
- exp_112 可以报告为“当前 diagonal Koopman free-head 的 exploratory negative result”，同时披露 test-selection 限制。

### 7.3 当前禁止声称

- M7、M9DSP、A1phys 或任何其他模型已经定性。
- DiD/CFE 是因果 ground truth。
- A1phys 是物理模型或已优于 Fan 路线。
- exp_112 关闭了 controlled Koopman，或 Koopman eigenvalues 等于真实物理时间常数。
- Neural ODE、Koopa 和 controlled Koopman 本身就是物理模型。
- 单次 N4SID 发散证明非线性模型必需。
- MPC 优于 PID、提高安全/能效/灵活性。
- 已投运现场实证或现场改善；仓库中没有对应版本、上线时间、对照设计、工况、样本量和结果产物。
- “RevIN 必需”“n_lag=2 完胜”等单 seed 或 test-selected 结论。

### 7.4 推荐故事线

暂定题目方向：

> Physics-guided and action-response-audited modeling of main-steam-temperature dynamics under wide-load operation

正文围绕：

1. 监督设定值层与物理喷水层的分离；
2. 通过闭合与可辨识门禁的 Fan20 SST 主干候选；
3. Fan17/Fan21 机制的嵌套消融；
4. black-box、low-order prior 与 physical gray-box 在预测、事件响应、OOD 和闭合上的 Pareto 比较；
5. 独立未来时间段、另一机组或 shadow-mode 验证。

内部协议 bug 是可信度修复，不宜包装成论文主贡献；能源过程建模结果应居于中心。

## 8. Phase 4 research questions

1. **RQ1 — task/estimand**：目标是完整 supervisory SP trajectory、阀位还是喷水流量？在什么附加假设或安全激励下，logged conditional response 才能升级为 controller-mediated/plant effect？
2. **RQ2 — closure**：在 10 s 采样和缺失测点下，Fan20 的最小 SST 方程能否闭合、稳定积分并恢复有意义的参数？
3. **RQ3 — mechanisms**：Fan17 explicit metal storage 和 Fan21 mismatch/load scheduling 在 Fan20 主干上分别带来什么独立增益，且能否避免状态/能量双计？
4. **RQ4 — representation**：固定状态、输入、closure、预算和协议后，structured ODE、fixed-operator controlled Koopman 与时变灰箱三种表示谁形成更好的预测—事件响应—物理—算力 Pareto 前沿？
5. **RQ5 — generalization**：候选能否跨时间、负荷、升降负荷、动作方向/幅度和控制模式泛化，并通过 lockbox/shadow-mode 验证？

详细实验路线与门禁见 [PHASE4_EXPERIMENT_PLAN.md](PHASE4_EXPERIMENT_PLAN.md)。

## 9. Final supervisor recommendation

**Go**：进入 Phase 4 Gate 0，修复 action semantics、split、event reference、测试和运行可追溯性。

**No-go**：立即运行 Fan×三路线大矩阵、补更多 exp_112 seeds、扩展 MPC 主表或撰写“最终模型”论文结论。
**Decision point**：只有 Fan20 core 在合成/公开数据和伊敏 train/validation 上通过物性、积分、可识别性和事件响应门禁后，才释放 Linux 多 seed route comparison。
