# FMTS 2026 投稿文献调研清单

> 用途：NeurIPS 2026 FMTS Workshop 4 页论文（预测 vs 世界模型张力 + 判别矩阵评测套件）的
> Related Work 与定位论证。所有 arXiv ID 均经 API 核验；无 ID 条目已核标题/出处。
> ⭐ = 与张力结论直接相关的论文。
> 谱系论证骨架见 docs/paper_ROM_mother_of_world_models.md（ROM-mother 精读笔记）。

## 1. 世界模型谱系与评测

| # | 标题 | 作者 | 出处 | ID | 相关性 |
|---|---|---|---|---|---|
| 1 | World Models | D. Ha, J. Schmidhuber | NeurIPS 2018 | 1803.10122 | 现代 WM 解剖结构（V→latent→M→C）起点 |
| 2 | Mastering Diverse Domains through World Models | D. Hafner et al. | 2023 | 2301.04104 | DreamerV3：latent 想象长时域经验能力（非认证） |
| 3 | Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model | J. Schrittwieser et al. | Nature 2020 | 1911.08265 | MuZero：规划型 WM 代表 |
| 4 | TD-MPC2: Scalable, Robust World Models for Continuous Control | N. Hansen et al. | ICLR 2024 | 2310.16828 | 连续控制 WM 精度-规划折中 |
| 5 ⭐ | Understanding World or Predicting Future? A Comprehensive Survey of World Models | J. Ding et al. | 2024 | 2411.14499 | 题目即我们张力命题；对 WM 评测缺口的综述 |
| 6 | Is Sora a World Simulator? A Comprehensive Survey on General World Models | Z. Zhu et al. | 2024 | 2405.03520 | 生成式 WM 的"逼真≠物理一致"象限论 |
| 7 | World Models for Autonomous Driving: An Initial Survey | Y. Guan et al. | 2024 | 2403.02622 | 任务关键系统 WM 评测视角 |
| 8 ⭐ | Double Check Your State Before Trusting It | J. Wang et al. | NeurIPS 2022 | 2206.07989 | 模型自信任幻觉态——WM 验证缺口实证 |
| 9 | Hallucinating Value: A Pitfall of Dyna-Style Planning with Imperfect Environment Models | T. Jafferjee et al. | NeurIPS 2021 | (proceedings) | 不完美模型规划失效的先例 |

## 2. 物理信息时序建模

| # | 标题 | 作者 | 出处 | ID | 相关性 |
|---|---|---|---|---|---|
| 10 | Neural Ordinary Differential Equations | R. Chen et al. | NeurIPS 2018 | 1806.07366 | 连续动力学学习基础 |
| 11 | Universal Differential Equations for Scientific Machine Learning | C. Rackauckas et al. | 2020 | 2001.04385 | UDE：物理骨架+可学习参数——我们的方法论直接祖先 |
| 12 | Efficiently Modeling Long Sequences with Structured State Spaces | A. Gu et al. | ICLR 2022 | 2111.00396 | 状态空间时序架构基线 |
| 13 | Physics-Informed Neural Networks for Power Systems | G. Misyris, A. Venzke, S. Chatzivasileiadis | IEEE PES GM 2020 | (IEEE) | 电力过程物理信息学习 |
| 14 | Prediction of Superheated Steam Temperature in Thermal Power Plants Based on the iTransformer Model | (Sensors 26(10):3078) | Sensors 2026 | (MDPI) | 主汽温纯数据预测现状：iTransformer 类基线 |
| 15 | Learning Interactive Real-World Simulators | S. Yang et al. | Nature 2024 | 2310.06114 | UniSim：仿真一致性的通用学习器对照 |

## 3. 反事实 / 动作条件有效性

| # | 标题 | 作者 | 出处 | ID | 相关性 |
|---|---|---|---|---|---|
| 16 | Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series | J. Runge | UAI/PMLR 2020 | 2003.03685 | PCMCI：时序因果发现方法（动作通道审计工具） |
| 17 | Closed-loop identification revisited | U. Forssell, L. Ljung | Automatica 35(7):1215-1241, 1999 | (DOI) | 闭环数据混杂的经典表述——我们"伪响应"现象的理论先例 |
| 18 | Robust Agents Learn Causal World Models | Z. Kenton et al. | ICLR 2024 | (OpenReview) | 因果 WM 主张与我们的因果审计互补 |
| 19 ⭐ | Forecast Collapse in Time-Series Foundation Models | S. Wan et al. | 2026-08-14 | 2608.14106 | **TSFM 在反馈/交互设定下预报坍缩**——纯预测线在闭环中的失效证据 |

## 4. 时序基础模型（纯预测对照组）

| # | 标题 | 作者 | 出处 | ID | 相关性 |
|---|---|---|---|---|---|
| 20 | Chronos: Learning the Language of Time Series | A. Ansari et al. | TMLR 2024 | 2403.07815 | 概率预报 FM 代表 |
| 21 | A decoder-only foundation model for time-series forecasting | A. Das et al. | ICML 2024 | 2310.10688 | TimesFM：预报 FM 代表 |
| 22 | Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting | K. Rasul et al. | NeurIPS 2024 | 2310.08278 | 概率预报 FM 代表 |
| 23 | MOMENT: A Family of Open Time-series Foundation Models | M. Goswami et al. | ICML 2024 | 2402.03885 | 开源 FM 家族 |
| 24 | Zero-shot Imputation with Foundation Inference Models for Dynamical Systems | P. Seifner et al. | NeurIPS 2024 | 2402.07594 | MIND：动力学 FM（系统推断视角，非纯预报） |

## 5. 泄漏感知评测与可靠性

| # | 标题 | 作者 | 出处 | ID | 相关性 |
|---|---|---|---|---|---|
| 25 | Conformalized Quantile Regression | Y. Romano et al. | NeurIPS 2019 | 1905.03222 | 校准不确定性基线；干预漂移下覆盖保证的局限是 P2 论证点 |
| 26 | A Path Towards Autonomous Machine Intelligence | Y. LeCun | OpenReview 2022 | (无 arXiv) | JEPA/世界模型架构主张（可验证性缺省的反面教材） |

## 6. ROM/MOR 验证谱系（谱系句原始引用）

| # | 标题 | 作者 | 出处 | 相关性 |
|---|---|---|---|---|
| 27 | Turbulence and the dynamics of coherent structures I | L. Sirovich | Q. Appl. Math. 45(3):561-571, 1987 | POD snapshots 起点 |
| 28 | The dynamics of coherent structures in the wall region of a turbulent boundary layer | N. Aubry, P. Holmes, J. Lumley, E. Stone | JFM 192:115-173, 1988 | 低维 Galerkin ODE 动力学 |
| 29 | A Survey of Projection-Based Model Reduction Methods for Parametric Dynamical Systems | P. Benner, S. Gugercin, K. Willcox | SIAM Review 57(4):483-531, 2015 | 投影 ROM + 误差界总览 |
| 30 | Using Simplicity to Control Complexity | L. Sha | IEEE Software 18(4):20-28, 2001 | Simplex 架构：可验证处置层的系统级先例 |
| 31 | Control barrier function based quadratic programs for safety critical systems | A. Ames, X. Xu, J. Grizzle, P. Tabuada | IEEE TAC 62(8), 2017 | runtime assurance 现代形式 |
| 32 ⭐ | ROM: The Mother of World Models | R. Ghosh | arXiv 2026-07 | 2607.03198 | 谱系论证锚点：MOR 解剖=WM 解剖；可验证性（P2）是部署真约束 |

## 论文定位建议（150 字内）

本文占据 FMTS 的 D3×D4 交界：以任务关键热工过程为测试床，给出"预报精度不传递为世界模型资格"的实证案例——纯数据线精度高但动作通道近死、物理结构线精度低但可反事实可审计，并以判别矩阵（同型门禁+结构消融+泄漏审计）落地 P5 型诚实评测。区别于综述与纯方法：现场双侧数据、负结果、可复现评测协议三者齐备；措辞锚点：verifiability over accuracy（承 ROM-mother Claim 2）。
