# FMTS 2026 稿件文献体检报告（Fernandez 2019 五产出清单）

> 体检对象: `docs/fmts2026/paper/fmts_main_v2.tex`（4页, 430行）+ `lit_survey.md`（32条清单）+ `fmts_refs.bib`（15条）
> 方法: Fernandez (2019) 五产出清单 + 引用准确性抽查（arXiv API/原文核对）
> 日期: 2026-09-02

## 总评

文献工作**强于平均水平**（gap 论证多层、问题表述教科书级、证据边界诚实），但有一个**必须立即修的错误引用**（wan2026）和一个**关键综述未引**（ding2024）。按 Fernandez 的标准，最危险的不是"引少了"而是"引错了"——引错会被内行当场识破。

## 🔴 致命项

### F1. wan2026forecast 引用失实（3处引用，全文核心论据之一）
- 正文 Introduction: "recent evidence shows time-series foundation models **collapsing under feedback** despite strong open-loop accuracy \cite{wan2026forecast}"
- Discussion: "our claim about recursive forecasters rests on external evidence \cite{wan2026forecast}"
- **arXiv 2608.14106 原文核对结论**：全文 27 页无任何 feedback/closed-loop/recursive-rollout 实验；摘要主旨=1000 只美股小时回报的**横截面**预测变平（ranking 失效），机制是低可预测性限幅 + per-series 目标丢跨序列结构；贡献是 CalibRank 目标与评测盲点（per-series metrics 掩盖 cross-series 失败）。**与"反馈坍缩"无关。**
- 风险：该文作者（Huan Liu 等，7人）极可能出现在 NeurIPS 系评审圈；被识破=全文可信度受损（Fernandez: "引错 = 懒惰或不道德"）。
- 修复（三选一）：
  1. **删除 feedback 声明**，改写为 wan2026 真实贡献："per-series 评测会掩盖下游决策所需的跨序列结构失败"——这与其判别矩阵的"评测盲点"论点点**天然同构**，反而更贴；
  2. 若一定要"feedback 下坍缩"证据：需另找文献（recursive rollout 误差累积类），不可硬撑；
  3. Limitations 中 "rests on external evidence" 句必须同步改（不能引用不存在的证据）。

## ⚠️ 重要项

### F2. ding2024 在 bib 中未引（最刺眼的综述缺口）
- "Understanding World or Predicting Future? A Comprehensive Survey of World Models"（2411.14499）——**综述标题就是本文的张力命题**，已入 bib（ding2024understanding）却零引用。
- 评审若熟悉该综述，会认为作者没读最关键的对标综述。
- 修复：Introduction 张力句加一句（顺带完成产出5的理论深化）：
  "Surveys frame the same tension as 'understanding vs predicting' \cite{ding2024understanding}; this paper provides an empirical instantiation in a fault-critical setting, where the qualification question becomes operational."

### F3. runge2020pcmci 在 bib 中未引
- PCMCI（2003.03685）是动作通道因果审计的工具级对照，判别矩阵（§4）没有方法学锚点。
- 修复：§4 鉴别矩阵或审计小节引用（"state-of-the-art causal discovery on autocorrelated series \cite{runge2020pcmci}"）。

### F4. 研究问题承诺 vs 结果张力
- 问题（88–92行）承诺 "causal **sign and magnitude**"；Discussion（392–397行）明言 "we claim **direction**, not magnitude"（2.7–7.6× gap，CF3 未落地）。
- Fernandez 第3条：评审会检查你是否回答了承诺的问题。修复：问题句改为 "sign and (a verified bound on) magnitude"，或在 Introduction 明示分层承诺（direction now; magnitude under \texttt{<CF3>}）。

### F5. "to our knowledge, the first qualification protocol"（408行）
- 该声明需先核对 ding2024 / zhu2024（2405.03520）是否已有类似"资格协议"讨论；若有，降级为 "first in fault-critical closed-loop setting"。
- 本次未读 zhu2024 全文，标注待核。⚠️ 注意：F2 修复后此句更易被对比。

## ✅ 达标项

- **产出1 证明知识**：经典锚（forssell1999 闭环辨识）+ 最新锚（wan2026 两周前）+ 谱系锚（ghosh2026rom/benner2015/sha2001）；ghosh2026rom 经 arXiv API 核对作者（Rajat Ghosh）✅，但 **v2 标题已改**为 "Reduced-Order Models: The Mother of World Models"（bib 内标题需同步）；正文"verifiability over accuracy"承继关系与原文摘要一致 ✅。
- **产出2 识别 gap**：多层诊断（representation/timescale/confounding）超额达标；外部 call（jafferjee/wang 规划失效）使用得当。
- **产出4 定位**：D3×D4 交界定位清晰；两 tier 架构（proposer bounded by certifiable ROM）有完整谱系支撑。
- **范式匹配**：统计方法（paired bootstrap、2/3 seed）与证据等级匹配；行动通道"盲/方向漂移"判定用词准确。
- **批判标记**："disqualifying"/"the illusion"/"faked" 显性批判 ✅。
- **单机组诚实**：Limitations 写明 single unit + dry operation + two-phase unobserved ✅（但见 F6）。

## 🔧 次要项

### F6. 单机组只被动承认
- Limitations 承认 single unit，但未把 side-B/跨机组表述为设计。建议 Discussion 加一句："the dual-side protocol (side B deferred) is the first step toward cross-unit qualification"——把弱点转为研究设计。

### F7. bib 未引条目清理（5条）
- ha2018world（World Models 起点概念——可用 hafner 替代，删或引）、ding2024、ansari2024chronos、das2024timesfm、runge2020pcmci。按 Fernandez"不读不引"，**要么用要么删**（F2/F3 修复后剩 ha/chronos/timesfm 三条——chronos/timesfm 可留作 Line1 对照句或删）。

### F8. 预印本引用标注
- wan2026/ghosh2026 均为 arXiv 预印本（未同行评审），引用时建议标注 "arXiv preprint"（bib 的 journal 字段已标，正文可不再标注；workshop 投稿可接受）。

### F9. 领域内主汽温预测现状未引
- Sensors 2026 iTransformer 主汽温论文（lit_survey #14）未进正文。评审若做该领域会问 Line 1 与"主汽温纯数据预测现状"的关系。建议 Introduction 的 Line 1 描述处引一句（若版面允许）。

## 优先修复顺序

1. **F1 wan2026 引用失实**（三处，今天修——正确改写为"评测盲点"论证或删除声明）
2. **F2 ding2024 进正文**（一句，同时完成理论对话）
3. **F4 研究问题小幅改词**
4. **F3 PCMCI 进 §4**
5. **F5 first 声明核对/降级**
6. F6/F7/F9 按版面处理
