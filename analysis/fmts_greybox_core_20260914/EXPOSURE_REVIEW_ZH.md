# FMTS 时间外候选：历史使用审计

2026-09-14；只读历史源码、运行清单和既有结果。不读取原始 CSV/NPZ 目标，未运行新评分。
核查基线为 release worktree `15757e8`；新候选合同来自主工作区的另一条数据准备线。
该候选合同的原样快照随本审计保存为 `source_contract.snapshot.json`，不把另一条模型线的完整代码带入FMTS。

## 结论

**2026-03-16 至 2026-05-11 不能称从未用于开发的独立留出集。** 不只是“不确定”：同源数据在历史 Phase3.5 中已有拟合、延迟选择及评估记录。当前 FMTS 没访问这些目标，与项目历史从未接触是两件事。

可考虑的定位仅为：相对当前 canonical 训练时间段的时间外推/稳健性探索，明确披露此前用于其他历史模型开发与分析。不能称“全新 test”或“post-development confirmatory”。本轮不运行该探索；23通道适配、最终模型/窗口冻结和单独注册仍未完成。真正独立确认需要另外核实此前未使用的数据，本审计不声称所有可能数据都已用过。

## 可定位的证据链

1. `experiments/side_full_windows_20260913/contract.json`：候选全源共 1,192,329 行，SHA256 为 `85a3f92648d5f88a4543f500859b200207fb55a32555900ca88f7c339c4e4da6`。它的 `extension` 名义起点 epoch 1773619210，即 **2026-03-16 00:00:10 UTC**；再加 10,240 s 隔离带才是 **02:50:50 UTC**，不是源数据起点。末端 epoch 1778543960 为 2026-05-11 23:59:20 UTC；实际可用 FMTS 窗口尚未建立。
2. `results/phase35_segmented_v2/run_manifest.json`：2026-08-09 生成，绑定完全相同的 CSV SHA；登记 development=2025-12-24至2026-03-31、internal_validation=4月、robustness=5月1–11日，并明确写有 `2026-05 previously viewed, robustness only; confirmatory needs new block`。记录脚本命令及已形成的判决，不是只有一份未执行计划。
3. `results/phase35_segmented_v2/run_manifest_v0.json`：再次绑定同一 SHA、1,192,329 行及 2026-05-11 23:59:20 UTC 末端。
4. `experiments/phase3_5/segmented_v2/v23_models.py:38` 的真实过滤代码按3月底、4月、5月分块。`:192` 起循环 **每个** dev/val/rob 块，`:206` 起扫描延迟并调用 `arx_fit`，按拟合 R² 取参数；并不只像注释所写“dev 上选”。`:233` 起还在 dev 拟合后对 val/rob 做预测评估。
5. `results/phase35_segmented_v2/blocked_model_scores.json` 存在 `right_val/left_val/right_rob/left_rob` 拟合结果及 `*_devmodel_on_val/rob` 评估结果。val 拟合约251,527–251,530点，rob为95,005点；这些实际产物与上述执行路径相符，足以否定未接触候选段的说法。旧分区拟合/选择是既有历史，不在本任务修改代码或重算旧结果。

代码+运行身份+非空产物共同构成证据；不把“代码可以读取”单独当成“确实跑过”。历史 manifest 自身记录 dirty 源码，未提供当时逐文件 SHA；因此这里是对仓库归档代码和结果的交叉审计，不宣称重放认证了2026-08-09的整个环境。

## 保留的不确定性与排除的误读

- Phase1/exp106/exp112 某些入口按文件尾或固定行界评估；其旧结果缺少完整运行时源 SHA/行数，不能用今天的大 CSV 推定当时必读到了5月。此项保持“具体范围不可完全核实”，不影响上面已确定的 Phase3.5 使用证据。
- Phase3.5 cache 末端 epoch 1773619190 是 **2026-03-15 23:59:50 UTC**，不是3月16日后约2小时40分；不拿它冒充 extension 已访问证据。
- 现有 side-full loader 的 test/extension 禁用只描述该新 loader；不能抹去更早的分析使用。
- 本轮读取的是已有结果摘要，未打开原始留出温度数组、未新建评分结果，也未据此调 CORE1 的窗口、参数或判据。
- CORE1 仍只用原 canonical 验证索引，固定三种子权重，四格组件对照+原GNR参考；这个溯源发现不更改其推理注册。

## 留痕与处理

`temporal_candidate_metadata.json` 保存本次证据文件及审计时 SHA，状态为 `historically_exposed`；元数据预检将返回 `HISTORICAL_EXPOSURE_CONFIRMED`，始终 `scoring_enabled=false`。即便未来完成23通道适配，也不能把这项状态改成未接触。

流程纠正：在让作者回忆实验历史之前，先追查实际 loader、运行源身份和结果归档；split 名称与当前访问开关不能代替数据使用史。
