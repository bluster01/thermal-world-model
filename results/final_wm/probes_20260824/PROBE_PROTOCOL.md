# 探针侧训练协议（execution-side, 2026-08-26 用户指示）

> 用户指示：以后所有探针训练默认用加速后的配置。本文件登记探针侧（非矩阵）协议；
> v0.6 矩阵首训口径仍遵循修正案 B（docs/plans/2026-08-26-v06-training-protocol-amendment.md，
> P2/P3 不进入首训）。

## 探针训练默认配置

1. **P1（标量同步 hoist）**：已入 src（data_v2.py 同步修复已含），对新进程自动生效，
   位一致（回归测试），无需额外开关。
2. **compile_substep=True**：`train_arm(..., compile_substep=True)`——aot_eager 模式，
   逐位一致（任务 50 已验），step 段 ~2x。探针脚本统一传入。
3. **预算**：120/20（120 epochs / patience 20），batch 32 / bpe 200，oracle 模式。
4. **锚定**：涉及常数初始化时优先 `TrainSpec.anchor_constants_checkpoint`（修正案 B
   一等公民）；旧 init_checkpoint 全量锚定文件仅存量探针兼容用。
5. **评测**：256 窗 seed50k，H18 主汽温双口径，分箱表必报。

## 不采用（及原因）
- P2（锚定二分 24→16 迭代）：漂移 0.05°C，探针对比会掺数值噪声；src 级修改待门。
- P3（全图 torch.compile）：train_arm 未接；dynamo 对插值函数重编译风暴未修；
  若探针自行复刻训练环可用（如 encoder_swap 系列），需标注漂移 0.027°C。

## 探针脚本纪律（本轮教训）

- 每个臂独立 out_dir + 独立 run 命名，禁止同 run_id 共目录（checkpoint 互相覆盖）。
- 探针记录文件（npz/json）与对侧制品逐位核验后才用。
