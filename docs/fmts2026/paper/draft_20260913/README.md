# FMTS 新论文初稿 — Linux v0.2 结果版（2026-09-13）

> **后续优化已立项**：旧 token 最佳权重在合成历史上出现饱和/弱历史区分；真实历史覆盖范围待 Linux 诊断。已按作者要求实现 [M7R1 + GNR1](../../../../experiments/fmts_m7fusion_20260913/RUN_LINUX.md)。本目录 PDF/数值保留为 v0.2 证据快照，不代表新 M7 模型结果，也不能作为“attention 架构本质更差”的结论。新稿更新等待两项完整结果审计。

本轮任务已收敛为：**主汽温精度与双阀动作响应的预测代价**。不是把 token 融合包装成胜出模型。
作者补充要求：纯灰箱的正向响应不能作为“物理响应正确”的依据。
已单独注册并准备 [GNR1 no-rewet 灰箱对照](../../../../experiments/fmts_greybox_norew_20260913/RUN_LINUX.md)，只新增三个种子完整拟合。
**当前 PDF 是 v0.2 证据审读稿；GNR1 已按作者 push 指令发布，尚未收到执行回执，其结果不能提前写入正文。**
按冻结规则保留 GRU：黑箱/灰箱/GRU/token 的 H18 MAE 分别为 0.371/0.969/0.443/0.654°C。
GRU 相对黑箱的精度代价为 **0.0725°C（19.6%）**。图示响应只代表模型行为，不代表已校准现场增益。

## 交付

- [英文 PDF](../../../../output/pdf/fmts_draft_20260913/en/paper_en.pdf)：正文 4 页；参考文献第 5 页；含附录共 9 页。
- [中文审读 PDF](../../../../output/pdf/fmts_draft_20260913/zh/paper_zh.pdf)：正文 4 页；参考文献第 5 页；含附录共 8 页。
- [英文 TeX](paper_en.tex)、[中文 TeX](paper_zh.tex)，各自附录及共享文献库。
- 图 1：工程流程与代码对应架构；图 2：累计精度折线 + 两阀响应曲线；附图：GRU/token 匹配观察器比较。
- `figures/` 含每图的中英文 SVG、PDF、360 dpi PNG。SVG 文字可编辑，无嵌入位图。
- [数值审计](audit.json)、[结论证据表](CLAIM_EVIDENCE_MAP.md)、[文档 QA](document_qa.json)、[构图与叙事合同](ARGUMENT_AND_FIGURE_CONTRACT.md)。

不覆盖 9/8 旧稿；不把旧的多目标数值混入新主表。新稿在独立 release worktree 保存，原工作区无关未提交内容未动。
本稿、图源和 GNR1 实验包按作者随后发出的 push 指令一同发布；本地未进行真实数据重训、解锁 test 或修改判据。README/TODO 与机器实验状态已同步。

## 审计与编译

Linux 返回提交 `399a60ce71905ad5236ae2692fe2c6021da8ffd1`，执行代码为 `74348fd699ec264660569abd5ea07ee069e84fc4`。
已有 `audit.py` 检查 12 个运行的产物哈希、检查点输入绑定和共同窗口；独立脚本重算 24 份原始数组。
Linux 提交说明声称 24 份推理数组重放通过，但未附单独 `audit.json`；本地没有再次从原始现场记录执行推理。
相关测试 10 项通过。双阀各 64 个历史均未触发注册 support 违规。

从仓库根目录：

```powershell
python docs/fmts2026/paper/draft_20260913/audit_evidence.py
python docs/fmts2026/paper/draft_20260913/build_figures.py
python -m pytest tests/final_wm/test_fmts_rich_pipeline.py tests/final_wm/test_fmts_mainsteam_experiment.py -q
```

用 Tectonic 或 XeLaTeX 编译 `paper_en.tex` / `paper_zh.tex`（需 BibTeX；中文使用 Windows 宋体/黑体/楷体与微软雅黑）。
文档检查脚本 `check_draft.py` 需 pypdf，使用 Codex 捆绑 Python 运行。
官方 `neurips_2026.sty` 保持原样；匿名作者块内 Affiliation/Address/email 是该官方样式的固定占位文本，不是泄露作者信息。
中文仅对摘要标签及可用字体做本地化，英文投稿排版不改官方参数。

绘图静态预检 11 pass / 0 fail；3 项提示均为交付格式说明：139.7 mm 符合本模板 5.5 英寸正文宽，
正式插图用矢量 PDF/SVG，360 dpi PNG 仅预览，因此不额外生成无用途 TIFF。

## 投稿前尚需作者确认

FMTS 官网要求正文最多 4 页、匿名；截至本次核验提交截止为 2026-09-16 11:59 UTC（北京时间 19:59）。
来源：[FMTS CFP](https://fmts-workshop.github.io/cfp.html)。
本稿符合页数边界，但仍是初稿；作者需确认工业动机表述、科学边界、匿名材料和数据共享权限。
锁定测试、现场响应校准及去 oracle 仍未授权执行；no-rewet 灰箱对照已单独注册、实现并发布给 Linux，等待正式回传。
