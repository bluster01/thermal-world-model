# FMTS-BLOCK1 本地交付验证 · 2026-09-15

## 已完成

- 注册和实现前计划先于执行代码保存。最初测试以缺少run模块失败；实现后通过。
- `python -m pytest tests/final_wm/test_fmts_block_rollout.py tests/final_wm/test_fmts_core.py -q -k 'not end_to_end_inference_and_audit'`：**30 passed, 1 deselected**。
- 新包20项：18点整块反馈/7次调用/120点末块截取；后续主汽温真值置毒不影响预测；同步其他输入；三类原forward首块逐元素一致且权重不变；共同资格/重复窗口/尾部输入与extension检查边界；等日/等seed聚合；私有输出目录；打包、保存数组审计、完整数值重放工程夹具；哈希/缺单元/指标/重放篡改拒绝；禁止覆盖。
- CORE1的10项无训练合同回归通过。明确排除旧 `test_end_to_end_inference_and_audit`，因为其parent fixture含微型合成训练；本轮测试没有优化器更新。
- 原v0.2黑箱3、保留GRU3、独立GNR1三份checkpoint均通过文件SHA、identity绑定、strict state_dict重建；只加载，不在本地实际记录上forward。
- GNR1原工件审计通过；parent identity/summary/indices SHA匹配注册；冻结parent source mismatch为空。
- run / audit 的 `--help` 可执行；`git diff --check` 通过。

## 尚未完成（不能当成结果）

- 本地没有经此注册SHA确认的实际canonical record可用于正式推理；未以其他本地npz替代。
- 无Linux领取、正式9单元推理或实际记录重放回执；未获得长程合格窗数量或长程MAE。
- 工程夹具中的“完整重放”只验证审计程序，不是科学证据；真实执行入口不提供smoke/训练/换协议选项。
- 无新正文/图/PDF改动；无test/extension评分，无新模型或重新辨识。
- 准备阶段未commit/push；作者随后已授权发布执行。发布回执见 `experiment_state.json` / `RESULTS.md`；既有论文、结果与状态的未提交修改保留，不混入本次发布。

## 放行条件

Linux使用冻结源和真实record/IAPWS完成9个单元，原H18门与新120点全重放通过；私有数组与公共回执都回到作者侧且哈希相符后，再审计绘图。仅公共汇总或仅文件完整不等于数据已到齐。
