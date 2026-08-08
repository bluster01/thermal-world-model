# Phase 3.5：A1phys 阀门级核心验证

> 状态：实验、代码框架和开发矩阵已实现；本地专项测试通过，等待 Linux 数据审计与训练结果。
> 范围：承接 Phase 3 论文核心验证，不启动 Phase 4，不比较 Fan 路线。
> 数据：A/B 两侧原始异步历史数据；喷水流量仅作诊断，不作监督真值。

## Material Passport

| 字段 | 内容 |
|---|---|
| 研究问题 | 绝对阀位驱动的 A1phys 灰箱路径能否同时保持预测能力与阀门物理响应一致性？ |
| 主 estimand | 实际二级减温阀位轨迹变化后，二减出口温度和末过出口主汽温的匹配闭环观测响应 |
| 辅 estimand | SP 变化是否经指令与阀门形成实际执行动作 |
| 动作 | 二级减温调节门绝对阀位；Δ阀位仅作表征消融和速率辅助特征 |
| 主要输出 | 主汽温 H1/H6/H18/H30/H60 预测误差；阀门事件 IRF；负对照响应 |
| 证据等级 | 阻断时间验证下的观测物理一致性；不宣称随机干预因果效应 |

## 1. 审计结论与重建理由

旧 `exp_106`/`causal_arch.py` 有一个值得保留的核心：逐步惯性递推不会让未来动作影响过去输出，且干预分支可满足零动作恒等式。但旧路径不能直接续跑：它使用 ΔSP 作为动作，属于 supervisory closed-loop estimand；训练逐 epoch 访问 test 并据此早停、保存 checkpoint；缺少 DiD JSON 或事件长度不一致时会静默退化为同名 CFI；干预增益监督压成 batch 标量；RevIN 使 null 分支在物理空间并不等于零基线。

Phase 3.5 因而新建独立实现，不修改旧结果。主要改变是：

1. 动作从 ΔSP 改为绝对阀位轨迹；
2. “零干预”定义为未来阀位保持 onset 前基准，而不是原始阀位等于零；
3. 干预分支直接输出物理温度增量，避免 RevIN 量纲歧义；
4. checkpoint 只按 validation 预测损失选择，训练期间不可读取 test；
5. 事件 IRF、负对照和预测指标分开报告，禁止压成 CFI 单标量；
6. A/B 两侧分别学习有效开度映射和动力学参数。

最新拉取的 `exp_201` 进一步支持“开度非线性值得正式验证”：A 侧固定等百分比 `R=50` 的 ff10 三 seed 在最终 test-Jacobian 上均为 100% 负方向，no-freeze 三 seed 为 95%/100%/100%（均值约 98.3%），而原始绝对阀位多为 60–75%。但该脚本逐 epoch 读取 test，并以 test MAE/Jacobian fallback CFI 选择 `best_cfi`；`R=50` 也不是现场流量标定。因此多 seed 只说明 pilot 信号可重复，不修复选择偏差。exp_201 只决定 E2 增加 fixed-prior 对照，不参与门禁阈值、checkpoint 或论文结果表。

## 2. 备选方案与架构决策

| 方案 | 优点 | 风险 | 决策 |
|---|---|---|---|
| 继续修补 `exp_106` | 改动最少 | 动作层级、全局副作用和 test 选择耦合太深 | 拒绝 |
| 包装旧 `ResidualCausalWM` | 可复用 checkpoint | 仍保留 ΔSP/cumsum 与归一化量纲假设 | 仅作历史对照 |
| 新建 `src/phase35` | 数据、模型、评测契约可单测；Linux 入口干净 | 初始工程量较大 | 采用 |

高层数据流：

```text
raw asynchronous CSV
        │ causal LOCF + staleness audit
        ▼
10 s cache (values + ages + columns + timestamps)
        │ chronological train / validation / locked-test
        ├──────────────► forecast windows
        ├──────────────► isolated valve events + matched quiet controls
        └──────────────► SP-executed / SP-no-execution negative controls

past state + past valve ─► future-action-isolated free head ─┐
                                                ├─► T prediction
absolute valve path ─► effective opening ─► lag ┘
```

## 3. A1phys-V 模型

模型保持残差分解：

\[
\hat T_{t+1:t+H}=f_{free}(x_{t-W+1:t})+
g_{phys}(x_{t-W+1:t},v_{t+1:t+H},v_t).
\]

阀门有效开度使用 A/B 侧独立、端点归一化的单调分段线性函数：

\[
e_t=m_s(v_t),\qquad m_s(0)=0,\quad m_s(100)=100,
\quad \frac{\partial m_s}{\partial v}\ge 0.
\]

真正进入惯性环节的是相对基准剂量：

\[
d_k=m_s(v_{t+k})-m_s(v_t).
\]

两级惯性递推为：

\[
r_k^{(1)}=r_{k-1}^{(1)}+\alpha_1(Kd_k-r_{k-1}^{(1)}),
\]

\[
r_k^{(2)}=r_{k-1}^{(2)}+\alpha_2(r_k^{(1)}-r_{k-1}^{(2)}).
\]

其中长期阀门增益约束为 \(K\le0\)，时间常数为正。恒定阀位路径严格得到 \(d_k=0\) 和 \(g_{phys}=0\)。可选 rate 分支只表达阀门变化速度的短暂效应，不能改变稳态增益。

## 4. 核心实验 E1–E5

| ID | 配置/分析 | 主要问题 | 主要判据 |
|---|---|---|---|
| E1 | Δ阀位无基准、Δ阀位+基准重建、绝对阀位 | 旧 Δ阀位失败来自信息丢失还是训练偶然？ | validation/test MAE、IRF-WMAE、动作响应非退化 |
| E2 | identity、固定等百分比 `R=50`、可学习 monotone、monotone+rate | 显式阀门非线性是否有额外价值？ | 预测比值≤1.02；IRF 误差改善≥2% 或 dose monotonicity 提升≥0.05；否则不升级复杂度 |
| E3 | 隔离开/关阀事件 + 匹配 quiet controls | 真实响应方向、时标、剂量是什么？ | 二减 60/120 s、主汽温 180/300/600 s 的 DiD IRF 与 UTC 日块 bootstrap CI |
| E4 | 实际阀位 vs constant-valve counterfactual | A1phys 是否复现 E3 的经验 IRF？ | direction、lag error、IRF-WMAE、dose monotonicity |
| E5 | SP 变阀不变 vs SP 变阀也变 | 模型是否区分监督信号与实际物理动作？ | 每组≥10 events；no-execution 模型动作响应接近零；两组不混为同一 estimand |

`free_only` 是 E4 的必要负基线。`delta_with_baseline` 与 identity absolute path 在数学上应等价，它是“基准信息是否足以恢复”的正对照；真正检验绝对工作点价值的是 nonlinear opening。旧 ΔSP-A1phys 只进入 E5 的 supervisory 附表，不进入阀门级模型主榜。

## 5. 数据与统计协议

- 原始异步历史数据只读；按 tag 的真实更新时间做 10 s 因果前向保持。
- 缓存同时保存值和 staleness，禁止 `nan_to_num(..., 0)` 把缺测伪装成阀门关闭。
- 每侧按时间前 60%/中 20%/后 20% 切分；test 在开发期不可访问。
- 事件阈值、匹配变量、模型配置和指标在 test 前冻结。
- 匹配只用处理前变量：负荷、主汽温误差及前趋势、阀位基准及前趋势、二减出口温度。
- E3 至少需要 30 个 matched events、开/关阀各 10 个、10 个独立 UTC 日块；匹配后要求 `max|SMD|≤0.20`、60 s 主汽温 pretrend 均值差绝对值 `≤0.15 °C`，且按动作方向统一后的 H60 经验响应 95% block-bootstrap CI 上界低于 0。任一条件不通过都不能写成已验证。
- 推断顶层单位是 UTC calendar-day block；先对同一日事件求均值，再做日块 bootstrap。seed 只衡量优化波动，两者分开报告。
- A/B 独立报告，再检查结论方向一致性；不把两侧行拼成伪增样本。
- SP `no_execution` 要求实际阀位在完整 600 s 响应窗内变化不超过 deadband；仅 60 s 后才动作的事件归入 ambiguous，而不是未执行。
- 喷水总流量不参与 loss、事件剂量、模型选择或主结论。

## 6. 模型选择与停止规则

唯一 checkpoint selector 是 validation integrated forecast MAE。物理指标是预注册 gate，不参与逐 epoch 选择：

1. 数值有限，constant-valve identity 通过；
2. 相对 `free_only` 产生非零、方向正确的阀门响应；
3. 相对 `absolute_identity`，复杂模型预测不劣且至少一个预声明 IRF 指标改善；
4. 若 nonlinear 不能改善，则回退到 absolute identity，不把阴性消融包装成失败；
5. test 只由独立 `evaluate.py --split test` 命令访问一次。

## 7. 运行矩阵与预算

每侧配置：

| run family | action mode | opening map | rate branch |
|---|---|---|---|
| `free_only` | none | identity | off |
| `delta_no_baseline` | delta | identity | off |
| `delta_with_baseline` | delta+baseline | identity | off |
| `absolute_identity` | absolute | identity | off |
| `absolute_equal_percentage_r50` | absolute | fixed equal-percentage `R=50` | off |
| `absolute_nonlinear` | absolute | monotone | off |
| `absolute_nonlinear_rate` | absolute+delta | monotone | on |

开发矩阵为 7 configs × 2 sides × 3 seeds = 42 runs。固定 `R=50` 只是 exp_201 pilot 产生的工程先验，不是流量标定；可学习 monotone 用于检验该形状能否由数据支持。通过 validation gate 后，每侧最多保留两个候选进入 locked test；最终候选补到 5 seeds。seed 数不用于替代日块 CI。

## 8. Linux 与本地职责

本地负责：协议、代码、单测、smoke、配置矩阵和结果 schema。Linux 只执行仓库中已冻结的命令、保存环境/git SHA、训练，并运行汇总脚本；Linux 不修改模型、阈值或指标。每个 run 至少输出：

```text
manifest.json
history.json
metrics_validation.json
event_metrics_validation.json
event_manifest_validation.json
checkpoint_best_val.pt
```

显式 test 评估另输出：

```text
metrics_test.json
event_metrics_test.json
event_manifest_test.json
access_ledger.json
```

完整命令和回传清单见 [`experiments/phase3_5/README.md`](../experiments/phase3_5/README.md)。

## 9. 论文主张边界

若 E1–E5 全部通过，可称“absolute-valve-conditioned, physics-guided gray-box world model with layered observational response validation”。不能称 A1phys 为守恒方程物理模型，也不能将 matched closed-loop IRF 写成随机干预因果真值。
