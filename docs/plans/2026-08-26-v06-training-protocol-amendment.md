# 修正案 v0.6-B：v0.6 训练协议修订包（2026-08-26 批准实施）

> 依据：执行侧 8/25 探针批次（09606b9 预算、e0d8c8e 锚定、5b95636 速度）+
> 设计侧审计 `results/final_wm/probe_batch_audit_20260826.md`。
> 用户 2026-08-26 批准 A/B/C 三项。本修正案=B 的 src 落地。

## 已落地（本提交）

1. **canonical v2.1（A 项）**：actions 按裁决后的修正接线重建（一级同侧/二级交叉；
   裁决证据 `results/final_wm/known_defect_v1_valve1_20260826.md`），连续性门
   fail-closed（新 valve2 vs v1 corr≥0.999 且 mae≤0.02）。双侧已重建：
   侧A valve2 corr=0.9999985 mae=2.7e-5，侧B corr=0.9998541 mae=1.1e-3；
   valve1 与旧（错侧）通道 corr=0.7843（双侧）= 预期的 ~0.8 provenance 值。
   阀位反馈负零漂（一级B 6.2%、二级A 17.9%、二级B 5.0%，界内 ≥-1%）按
   [-0.02, 1.0] 容忍带通过量程门，终端 clip≥0；与 aux spray 清洗同规则。
2. **P1 提速**：`properties.py` 的 tsat 多项式系数与 `_psub[0]` 在构造期
   hoist 为 Python float，消灭每调用 5+1 次 DtoH 标量同步（执行侧 profile：
   占前向 CPU 58%）。**数值逐位不变**（回归测试
   `test_saturation_temperature_bit_identical_to_legacy_scalar_sync_path`）。
   `boundary.py` oracle 的 sigma 不动：float32 log 计算路径保持位一致，
   且每 batch 仅一次、占比可忽略。
3. **常数锚定（armC 协议一等公民）**：`TrainSpec.anchor_constants_checkpoint`
   + `apply_anchor_constants()`——仅从参考 checkpoint 拷贝 `transition.raw`
   34 个物理常数，网络全部新鲜 init；fail-closed（缺键/形状不符即拒）；
   与 `init_checkpoint` 互斥、与 `_norew` 消融臂不可组合（防覆盖 aW 钉值）；
   指纹经 `asdict(spec)` 自动覆盖。

## 协议口径（v0.6 矩阵执行侧须知）

- **预算**：v0.6 矩阵所有臂 epochs=120 / patience=20（TrainSpec 默认值不动，
  避免静默改变既有脚本语义；由 v0.6 运行配置显式给出）。
- **锚定源**：多种子臂统一以"最好盆地"臂的 checkpoint 为
  anchor_constants_checkpoint（探针实证：种子极差 0.30→0.06）。锚定臂
  与未锚定臂指纹不同，判决时分开列报。
- **速度**：P1 已生效零风险；P2（iters 24→16，漂移 0.05°C）与
  P3（全图 compile，漂移 0.027°C）**不进入 v0.6 首训**——数值口径保持
  与冻结证据可比，留待 v0.6 矩阵稳定后单列数字门评审。

## 关闭登记（C 项）

observer 编码器路线关闭：TCN 0.754（负）、iTransformer 0.666（仅贴生产带
0.597-0.652 上缘）、itx+锚定组合 0.652 劣于纯锚定 0.478/0.465——编码器
不构成对 GRU observer 的带内改进。证据：`results/final_wm/probes_20260824/encoder_probe/`。

## 测试

148/148（含 v2.1 数据 12 项、P1 位一致回归、锚定 3 项）。
